import asyncio
import logging
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


TASK_ID = "bestbuy_sd_portable_gaming_laptop"
TASK_DESCRIPTION = (
    "I'm looking for a portable gaming laptop to purchase at a Best Buy store in San Diego, California. Please help me find one model that meets ALL of the following requirements: "
    "(1) Display with at least 144Hz refresh rate, "
    "(2) Display resolution of at least 1920×1080 pixels (Full HD), "
    "(3) Weight under 5 pounds, "
    "(4) Equipped with either an AMD Ryzen 7 or Ryzen 9 processor, OR an Intel Core Ultra 7 or Core Ultra 9 processor, "
    "(5) At least 6 hours of battery life for productivity tasks (non-gaming use), "
    "(6) At least 512GB SSD storage, "
    "(7) NVIDIA GeForce RTX 4050 or better dedicated graphics card (or equivalent AMD graphics), "
    "(8) Currently available for purchase at a Best Buy store location in San Diego, California. "
    "Please provide the laptop model name, manufacturer, and specific verification details for each of the eight requirements listed above, along with a reference link to confirm its availability at Best Buy San Diego."
)

SAN_DIEGO_STORE_KEYWORDS = [
    "San Diego", "Mission Valley", "Sports Arena", "Midway", "UTC", "University Town Center",
    "La Jolla", "Balboa", "Clairemont", "Mira Mesa"
]


class LaptopModel(BaseModel):
    model_name: Optional[str] = None
    manufacturer: Optional[str] = None


class ProposedModels(BaseModel):
    models: List[LaptopModel] = Field(default_factory=list)


class LaptopSpecs(BaseModel):
    model_name: Optional[str] = None
    manufacturer: Optional[str] = None
    refresh_rate_hz: Optional[str] = None
    resolution: Optional[str] = None
    weight_lb: Optional[str] = None
    cpu: Optional[str] = None
    battery_life_hours: Optional[str] = None
    storage: Optional[str] = None
    gpu: Optional[str] = None
    bestbuy_urls: List[str] = Field(default_factory=list)
    spec_urls: List[str] = Field(default_factory=list)
    other_urls: List[str] = Field(default_factory=list)


def prompt_extract_proposed_models() -> str:
    return """
Extract all distinct laptop models that the answer proposes or recommends for purchase. 
Rules:
- Count unique models only (ignore duplicates or color/storage variants as separate).
- Ignore generic families without a specific model identifier.
Return:
{
  "models": [
    {"model_name": string | null, "manufacturer": string | null},
    ...
  ]
}
If none are proposed, return an empty array.
"""


def prompt_extract_laptop_specs() -> str:
    return """
Extract the detailed information for the single, primary laptop the answer is recommending (if multiple are listed, choose the first or the one most strongly recommended).
Fields to extract:
- model_name: exact model name string
- manufacturer: brand/manufacturer
- refresh_rate_hz: the stated display refresh rate string (e.g., "144Hz", "165 Hz")
- resolution: the stated display resolution string (e.g., "1920 x 1080", "2560x1440", "4K (3840x2160)")
- weight_lb: the stated laptop weight string (prefer pounds if available; otherwise provide what's stated)
- cpu: the stated CPU string (e.g., "AMD Ryzen 7 7840HS", "Intel Core Ultra 7 155H")
- battery_life_hours: the stated battery life string for general productivity (non‑gaming) use (e.g., "up to 8 hours")
- storage: the stated storage string (e.g., "512GB SSD", "1TB PCIe SSD")
- gpu: the stated dedicated GPU string (e.g., "NVIDIA GeForce RTX 4060", "AMD Radeon RX 7700S")
- bestbuy_urls: array of all BestBuy links in the answer for this laptop (product page, store availability, etc.)
- spec_urls: array of manufacturer or retailer spec pages cited for this model (non-BestBuy also allowed)
- other_urls: any other cited URLs relevant to verifying the above specs
Return a JSON object with these fields. If a field is not provided in the answer, set it to null (or an empty array for URL lists).
IMPORTANT for URLs: extract only actual URLs present in the answer text (plain or markdown format). Do not invent.
"""


def _dedup_urls(urls: List[str]) -> List[str]:
    seen = {}
    for u in urls or []:
        if isinstance(u, str):
            s = u.strip()
            if s and s not in seen:
                seen[s] = True
    return list(seen.keys())


def _choose_sources(*url_lists: List[str]) -> Optional[List[str]]:
    merged = []
    for lst in url_lists:
        merged.extend(lst or [])
    dedup = _dedup_urls(merged)
    return dedup if dedup else None


async def _build_verification_tree(
    evaluator: Evaluator,
    root,
    proposed: ProposedModels,
    specs: LaptopSpecs
) -> None:
    # Root is sequential: First identification, then constraints
    # 1) Laptop Identification (parallel, critical)
    id_node = evaluator.add_parallel(
        id="Laptop_Identification",
        desc="Response provides the required laptop identifiers.",
        parent=root,
        critical=True
    )

    # Single model only (critical) — determine using extraction count
    model_count = sum(1 for m in (proposed.models or []) if (m.model_name and m.model_name.strip()))
    evaluator.add_custom_node(
        result=(model_count == 1),
        id="Single_Model_Only",
        desc="Response proposes exactly one laptop model (not multiple).",
        parent=id_node,
        critical=True
    )

    # Model name provided (critical)
    evaluator.add_custom_node(
        result=bool(specs.model_name and specs.model_name.strip()),
        id="Model_Name_Provided",
        desc="Response includes the laptop model name.",
        parent=id_node,
        critical=True
    )

    # Manufacturer provided (critical)
    evaluator.add_custom_node(
        result=bool(specs.manufacturer and specs.manufacturer.strip()),
        id="Manufacturer_Provided",
        desc="Response includes the laptop manufacturer/brand.",
        parent=id_node,
        critical=True
    )

    # 2) Constraints with verification (parallel, critical)
    cons_node = evaluator.add_parallel(
        id="Constraints_Met_With_Verification",
        desc="Each constraint below is explicitly verified with a spec/value (and link where required).",
        parent=root,
        critical=True
    )

    # Prepare sources
    bestbuy_urls = _dedup_urls(specs.bestbuy_urls or [])
    spec_urls = _dedup_urls(specs.spec_urls or [])
    other_urls = _dedup_urls(specs.other_urls or [])
    all_sources = _choose_sources(bestbuy_urls, spec_urls, other_urls)

    # Create leaves for requirements 1–7
    req1 = evaluator.add_leaf(
        id="Req1_RefreshRate",
        desc="States the display refresh rate and it is ≥ 144Hz.",
        parent=cons_node,
        critical=True
    )
    req2 = evaluator.add_leaf(
        id="Req2_Resolution",
        desc="States the display resolution and it is ≥ 1920×1080.",
        parent=cons_node,
        critical=True
    )
    req3 = evaluator.add_leaf(
        id="Req3_Weight",
        desc="States the laptop weight and it is < 5 lb.",
        parent=cons_node,
        critical=True
    )
    req4 = evaluator.add_leaf(
        id="Req4_CPU",
        desc="States the CPU model/family and it is AMD Ryzen 7/9 OR Intel Core Ultra 7/9.",
        parent=cons_node,
        critical=True
    )
    req5 = evaluator.add_leaf(
        id="Req5_BatteryLife",
        desc="States battery life for non-gaming productivity use and it is ≥ 6 hours.",
        parent=cons_node,
        critical=True
    )
    req6 = evaluator.add_leaf(
        id="Req6_Storage",
        desc="States SSD storage capacity and it is ≥ 512GB (SSD).",
        parent=cons_node,
        critical=True
    )
    req7 = evaluator.add_leaf(
        id="Req7_GPU",
        desc="States the dedicated GPU model and it is RTX 4050 or better (or equivalent AMD graphics).",
        parent=cons_node,
        critical=True
    )

    # Build claims
    model_id = f"{(specs.manufacturer or '').strip()} {(specs.model_name or '').strip()}".strip()
    claims_and_sources = [
        (
            f"The laptop {model_id} has a display refresh rate of at least 144Hz."
            + (f" The answer stated: '{specs.refresh_rate_hz}'." if specs.refresh_rate_hz else ""),
            all_sources,
            req1,
            "Accept values such as 144Hz, 165Hz, 240Hz, or higher. "
            "If only the exact numeric value appears (e.g., 165 Hz), that is acceptable as ≥144Hz."
        ),
        (
            f"The laptop {model_id} has a display resolution of at least 1920x1080 (Full HD) or higher."
            + (f" The answer stated: '{specs.resolution}'." if specs.resolution else ""),
            all_sources,
            req2,
            "Accept 1920×1080 (FHD) or any higher resolution such as 2560×1440 (QHD/WQHD) or 3840×2160 (4K/UHD)."
        ),
        (
            f"The laptop {model_id} weighs under 5 pounds (less than 5.00 lb)."
            + (f" The answer stated: '{specs.weight_lb}'." if specs.weight_lb else ""),
            all_sources,
            req3,
            "If weight is given in kilograms, convert using 1 kg ≈ 2.20462 lb. "
            "Base judgment on the product specs (not shipping weight)."
        ),
        (
            f"The laptop {model_id} is equipped with either an AMD Ryzen 7 or Ryzen 9 processor, "
            f"or an Intel Core Ultra 7 or Core Ultra 9 processor."
            + (f" The answer stated: '{specs.cpu}'." if specs.cpu else ""),
            all_sources,
            req4,
            "Accept CPUs explicitly containing 'AMD Ryzen 7' or 'AMD Ryzen 9', or 'Intel Core Ultra 7' or 'Intel Core Ultra 9'. "
            "Do NOT count older 'Intel Core i7/i9' unless it clearly says 'Core Ultra'."
        ),
        (
            f"The laptop {model_id} offers battery life of at least 6 hours for general productivity (non-gaming) use."
            + (f" The answer stated: '{specs.battery_life_hours}'." if specs.battery_life_hours else ""),
            all_sources,
            req5,
            "Manufacturer or retailer claims like 'up to 7 hours' or 'up to 8 hours' are acceptable. "
            "If multiple battery life numbers are given, use the non-gaming productivity estimate."
        ),
        (
            f"The laptop {model_id} has at least 512GB SSD storage."
            + (f" The answer stated: '{specs.storage}'." if specs.storage else ""),
            all_sources,
            req6,
            "Accept 512GB SSD or any larger SSD capacity (e.g., 1TB). Hybrid or HDD-only configurations do not satisfy."
        ),
        (
            f"The laptop {model_id} has a dedicated GPU that is an NVIDIA GeForce RTX 4050 or better, "
            f"or an equivalent AMD dedicated graphics GPU."
            + (f" The answer stated: '{specs.gpu}'." if specs.gpu else ""),
            all_sources,
            "Req7_GPU",
            "NVIDIA: accept RTX 4050/4060/4070/4080/4090 Laptop GPUs. "
            "AMD: accept recent Radeon dedicated laptop GPUs typically positioned for mid‑range or above gaming "
            "(e.g., RX 7600M/7600S/7700S/7800S). Integrated GPUs do not satisfy this."
        ),
    ]

    # Parallel verification for req1–req7
    await evaluator.batch_verify(claims_and_sources)

    # Req8: Split into (a) link provided and (b) availability verified with link(s)
    req8_block = evaluator.add_parallel(
        id="Req8_BestBuy_SanDiego_Availability_Block",
        desc="Best Buy San Diego availability with substantiating link(s).",
        parent=cons_node,
        critical=True
    )

    # 8a) Link provided (critical)
    evaluator.add_custom_node(
        result=bool(bestbuy_urls),
        id="Req8_Link_Provided",
        desc="At least one Best Buy URL for this model is provided in the answer.",
        parent=req8_block,
        critical=True
    )

    # 8b) Availability verified via Best Buy page(s) (critical)
    req8b = evaluator.add_leaf(
        id="Req8_BestBuy_SanDiego_Availability_With_Link",
        desc="Provides evidence the laptop is currently available for purchase at a Best Buy store location in San Diego, CA, via a substantiating link.",
        parent=req8_block,
        critical=True
    )

    availability_claim = (
        f"The Best Buy page(s) show that the laptop {model_id} is currently available for purchase "
        f"at a Best Buy store location in San Diego, California (e.g., pickup/availability at a San Diego store)."
    )
    add_ins = (
        "Look for pickup/availability indicators for San Diego stores. "
        "Accept store names or locations such as: "
        + ", ".join(SAN_DIEGO_STORE_KEYWORDS)
        + ". If the page shows pickup today/available at any San Diego store, count as supported. "
          "If only shipping is available with no San Diego store availability indicated, this does not satisfy."
    )
    await evaluator.verify(
        claim=availability_claim,
        node=req8b,
        sources=bestbuy_urls if bestbuy_urls else None,
        additional_instruction=add_ins
    )


async def evaluate_answer(
    client: Any,
    answer: str,
    agent_name: str,
    answer_name: str,
    cache: CacheFileSys,
    semaphore: asyncio.Semaphore,
    logger: logging.Logger,
    model: str = "o4-mini"
) -> Dict:
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.SEQUENTIAL,
        agent_name=agent_name,
        answer_name=answer_name,
        client=client,
        task_description=TASK_DESCRIPTION,
        answer=answer,
        global_cache=cache,
        global_semaphore=semaphore,
        logger=logger,
        default_model=model
    )

    # Extraction
    proposed_models = await evaluator.extract(
        prompt=prompt_extract_proposed_models(),
        template_class=ProposedModels,
        extraction_name="proposed_models"
    )
    chosen_specs = await evaluator.extract(
        prompt=prompt_extract_laptop_specs(),
        template_class=LaptopSpecs,
        extraction_name="chosen_laptop_specs"
    )

    # Custom info for debugging
    evaluator.add_custom_info(
        info={"san_diego_store_keywords": SAN_DIEGO_STORE_KEYWORDS},
        info_type="store_keywords",
        info_name="san_diego_store_keywords_used"
    )

    # Build verification tree and run checks
    await _build_verification_tree(evaluator, root, proposed_models, chosen_specs)

    return evaluator.get_summary()