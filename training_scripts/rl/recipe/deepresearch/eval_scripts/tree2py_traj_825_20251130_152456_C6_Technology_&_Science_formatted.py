import asyncio
import logging
from typing import Optional, List, Dict, Any, Tuple

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.llm_client.base_client import LLMClient


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "gaming_laptop_microcenter_exact3stores"
TASK_DESCRIPTION = """
Identify a gaming laptop that meets all of the following requirements:

1. Geographic Requirement:
   - The laptop must be available for purchase at physical Micro Center store locations in a U.S. state that has exactly 3 Micro Center stores.

2. Display Requirements:
   - Display size must be at least 16 inches (diagonal measurement)
   - Display refresh rate must be at least 240Hz
   - Display must use OLED panel technology

3. Graphics Requirements:
   - Must feature an NVIDIA GeForce RTX 5000-series GPU
   - The GPU must specifically be RTX 5080 or RTX 5090 (high-tier only)

4. Portability Requirement:
   - The laptop must weigh 5 pounds (2.27 kg) or less

5. Connectivity Requirement:
   - Must include at least one Thunderbolt port (Thunderbolt 3, 4, or 5)

6. Battery Requirement:
   - Battery capacity must be at least 90Wh

Provide the following information:
- The state where the laptop is available (that has exactly 3 Micro Center locations)
- The laptop manufacturer/brand and specific model name
- All relevant specifications with supporting reference URLs
"""


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class LaptopIdentity(BaseModel):
    brand: Optional[str] = None
    model: Optional[str] = None


class GeographyInfo(BaseModel):
    state: Optional[str] = None
    store_count_sources: List[str] = Field(default_factory=list, description="URLs that show the state has exactly 3 Micro Center stores")
    in_store_availability_sources: List[str] = Field(default_factory=list, description="URLs showing in-store availability in that state (e.g., Micro Center product/store pages)")


class SpecInfo(BaseModel):
    display_size: Optional[str] = None
    refresh_rate_hz: Optional[str] = None
    panel_type: Optional[str] = None
    gpu: Optional[str] = None
    weight: Optional[str] = None
    thunderbolt: Optional[str] = None
    battery_wh: Optional[str] = None
    spec_sources: List[str] = Field(default_factory=list, description="URLs substantiating the specs (manufacturer page, Micro Center product listing, etc.)")


class LaptopExtraction(BaseModel):
    laptop: Optional[LaptopIdentity] = None
    geography: Optional[GeographyInfo] = None
    specs: Optional[SpecInfo] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_laptop_candidate() -> str:
    return """
You will extract a single gaming laptop candidate described in the answer that claims to meet all constraints. If multiple candidates are provided, select the first clearly identified one.

Return a JSON with the following structure:

- laptop:
  - brand: Manufacturer or brand name (string)
  - model: Specific model identifier (string)

- geography:
  - state: The U.S. state claimed to have exactly 3 Micro Center store locations where the laptop is available for in-store purchase
  - store_count_sources: Array of URLs that explicitly substantiate that the named state has exactly 3 Micro Center store locations
  - in_store_availability_sources: Array of URLs that substantiate the laptop’s in-store availability in the named state (e.g., product page showing in-store stock for a store in that state)

- specs:
  - display_size: The diagonal display size as written (e.g., "16.1-inch", "16-inch") – keep as a string
  - refresh_rate_hz: The refresh rate as written (e.g., "240Hz", "300 Hz") – keep as a string
  - panel_type: Panel technology as written (e.g., "OLED", "IPS")
  - gpu: GPU as written (e.g., "NVIDIA GeForce RTX 5090 Laptop GPU")
  - weight: Weight as written (e.g., "4.8 lb", "2.1 kg") – keep as a string
  - thunderbolt: Thunderbolt wording as written (e.g., "Thunderbolt 4", "USB4 with Thunderbolt") – keep as a string
  - battery_wh: Battery capacity as written (e.g., "99.9 Wh", "90Wh") – keep as a string
  - spec_sources: Array of URLs that substantiate the laptop specifications (manufacturer page, Micro Center product page, retailer spec sheet, etc.)

IMPORTANT RULES:
- Extract only from the provided answer text.
- Do not invent or infer any URL. Only include URLs that appear in the answer (plain or markdown links).
- If a field is missing in the answer, set it to null (or an empty array for URL fields).
- If units vary (inches vs. cm, pounds vs. kg), keep them exactly as written.
- If multiple URLs are listed, include them all in the appropriate arrays.
"""


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _safe_list(v: Optional[List[str]]) -> List[str]:
    return v if isinstance(v, list) else []


def _name_or_generic(brand: Optional[str], model: Optional[str]) -> str:
    if brand and model:
        return f"{brand} {model}"
    if brand:
        return f"{brand} laptop"
    if model:
        return f"{model} laptop"
    return "the laptop"


async def _verify_with_optional_sources(
    evaluator: Evaluator,
    items: List[Tuple[str, Optional[List[str]], Any, Optional[str]]],
) -> None:
    """
    items: list of tuples (claim, sources, node, additional_instruction)
    If sources is empty or None, mark node failed directly. Otherwise batch verify.
    """
    to_verify: List[Tuple[str, List[str], Any, Optional[str]]] = []
    for claim, sources, node, add_ins in items:
        srcs = _safe_list(sources)
        if len(srcs) == 0:
            node.score = 0.0
            node.status = "failed"
        else:
            to_verify.append((claim, srcs, node, (add_ins or "None")))

    if to_verify:
        await evaluator.batch_verify(to_verify)


# --------------------------------------------------------------------------- #
# Verification builder                                                        #
# --------------------------------------------------------------------------- #
async def build_and_verify(evaluator: Evaluator, extracted: LaptopExtraction) -> None:
    """
    Build the rubric tree and run all verifications according to the provided rubric JSON.
    The top-level rubric is critical (all criteria must be satisfied).
    """
    # Prepare extracted fields with safe defaults
    brand = extracted.laptop.brand.strip() if (extracted.laptop and extracted.laptop.brand) else None
    model = extracted.laptop.model.strip() if (extracted.laptop and extracted.laptop.model) else None

    state = extracted.geography.state.strip() if (extracted.geography and extracted.geography.state) else None
    store_count_sources = _safe_list(extracted.geography.store_count_sources if extracted.geography else [])
    availability_sources = _safe_list(extracted.geography.in_store_availability_sources if extracted.geography else [])

    display_size = extracted.specs.display_size.strip() if (extracted.specs and extracted.specs.display_size) else None
    refresh_rate = extracted.specs.refresh_rate_hz.strip() if (extracted.specs and extracted.specs.refresh_rate_hz) else None
    panel_type = extracted.specs.panel_type.strip() if (extracted.specs and extracted.specs.panel_type) else None
    gpu = extracted.specs.gpu.strip() if (extracted.specs and extracted.specs.gpu) else None
    weight = extracted.specs.weight.strip() if (extracted.specs and extracted.specs.weight) else None
    thunderbolt = extracted.specs.thunderbolt.strip() if (extracted.specs and extracted.specs.thunderbolt) else None
    battery_wh = extracted.specs.battery_wh.strip() if (extracted.specs and extracted.specs.battery_wh) else None
    spec_sources = _safe_list(extracted.specs.spec_sources if extracted.specs else [])

    # Create a critical task root under evaluator.root (so the whole task is all-or-nothing)
    task_root = evaluator.add_parallel(
        id="task_root",
        desc="Identify one gaming laptop that meets all geographic and technical requirements, and provide supporting URLs for the claims",
        parent=evaluator.root,
        critical=True
    )

    # 1) Laptop identification (critical)
    evaluator.add_custom_node(
        result=bool(brand) and bool(model),
        id="laptop_identification",
        desc="Provide the laptop manufacturer/brand and specific model identifier",
        parent=task_root,
        critical=True
    )

    # 2) Geographic requirement (critical, parallel children)
    geo_node = evaluator.add_parallel(
        id="geographic_requirement",
        desc="Meets the geographic requirement (state with exactly 3 Micro Center stores) and is available for in-store purchase there",
        parent=task_root,
        critical=True
    )

    # 2.1) State has exactly three stores (critical leaf)
    state_leaf = evaluator.add_leaf(
        id="state_with_exactly_three_stores",
        desc="Names the U.S. state and it is verified to have exactly 3 Micro Center store locations",
        parent=geo_node,
        critical=True
    )
    state_value = state or ""
    claim_state = f"The U.S. state of '{state_value}' has exactly three (3) Micro Center store locations."
    # 2.2) In-store availability in that state (critical leaf)
    availability_leaf = evaluator.add_leaf(
        id="in_store_availability_in_that_state",
        desc="Laptop is available for purchase at a physical Micro Center store location in the named state (not online-only/general availability)",
        parent=geo_node,
        critical=True
    )
    laptop_name = _name_or_generic(brand, model)
    claim_avail = f"The {laptop_name} is available for purchase for in-store pickup or same-day purchase at a Micro Center store located in the state of '{state_value}'."

    # Execute geographic verifications (fail immediately if no sources)
    await _verify_with_optional_sources(
        evaluator,
        [
            (
                claim_state,
                store_count_sources,
                state_leaf,
                "Verify from the provided URL(s) that the named state has exactly 3 Micro Center store locations. "
                "Evidence could be a store locator page or an explicit statement like '3 stores in <state>'."
            ),
            (
                claim_avail,
                availability_sources,
                availability_leaf,
                "Verify from the provided URL(s) that this specific model is available in-store (e.g., 'In Stock' or 'In-Store Pickup') at a Micro Center store in the named state. "
                "Do not accept general/online-only availability as sufficient."
            ),
        ]
    )

    # 3) Display requirements (critical, parallel)
    display_node = evaluator.add_parallel(
        id="display_requirements",
        desc="Meets all display requirements",
        parent=task_root,
        critical=True
    )
    # 3.1) Size >= 16 inches
    size_leaf = evaluator.add_leaf(
        id="display_size_min_16",
        desc="Display size is at least 16 inches (diagonal)",
        parent=display_node,
        critical=True
    )
    claim_size = (
        f"The {laptop_name} has a display diagonal of at least 16 inches."
    )
    # 3.2) Refresh rate >= 240 Hz
    refresh_leaf = evaluator.add_leaf(
        id="refresh_rate_min_240hz",
        desc="Display refresh rate is at least 240 Hz",
        parent=display_node,
        critical=True
    )
    claim_refresh = (
        f"The {laptop_name} has a display refresh rate of at least 240 Hz."
    )
    # 3.3) OLED panel
    oled_leaf = evaluator.add_leaf(
        id="oled_panel",
        desc="Display panel technology is OLED",
        parent=display_node,
        critical=True
    )
    claim_oled = (
        f"The {laptop_name} uses an OLED display panel technology."
    )

    # 4) GPU requirement (critical leaf)
    gpu_leaf = evaluator.add_leaf(
        id="gpu_requirement",
        desc="GPU is NVIDIA GeForce RTX 5080 or RTX 5090 (which satisfies the RTX 5000-series constraint)",
        parent=task_root,
        critical=True
    )
    claim_gpu = (
        f"The {laptop_name} is equipped with an NVIDIA GeForce RTX 5080 or NVIDIA GeForce RTX 5090 laptop GPU."
    )

    # 5) Weight requirement (critical leaf)
    weight_leaf = evaluator.add_leaf(
        id="weight_requirement",
        desc="Laptop weight is 5 lb (2.27 kg) or less",
        parent=task_root,
        critical=True
    )
    claim_weight = (
        f"The {laptop_name} weighs 5.0 lb (2.27 kg) or less."
    )

    # 6) Thunderbolt requirement (critical leaf)
    tb_leaf = evaluator.add_leaf(
        id="thunderbolt_requirement",
        desc="Includes at least one Thunderbolt port (Thunderbolt 3, 4, or 5)",
        parent=task_root,
        critical=True
    )
    claim_tb = (
        f"The {laptop_name} includes at least one Thunderbolt port (Thunderbolt 3, Thunderbolt 4, or Thunderbolt 5)."
    )

    # 7) Battery requirement (critical leaf)
    battery_leaf = evaluator.add_leaf(
        id="battery_requirement",
        desc="Battery capacity is at least 90 Wh",
        parent=task_root,
        critical=True
    )
    claim_battery = (
        f"The {laptop_name} has a battery capacity of at least 90 Wh."
    )

    # Execute spec verifications. Fail fast if no spec sources are provided.
    await _verify_with_optional_sources(
        evaluator,
        [
            (
                claim_size,
                spec_sources,
                size_leaf,
                "From the provided URL(s), confirm the screen size. Consider equivalent forms (e.g., 16.0, 16.1, 16.2 inches). "
                "If listed in centimeters, ensure it corresponds to at least 16 inches."
            ),
            (
                claim_refresh,
                spec_sources,
                refresh_leaf,
                "From the provided URL(s), confirm the refresh rate is 240 Hz or higher. "
                "Accept equivalent formats like '≥ 240 Hz' or specific values such as 240 Hz, 300 Hz, etc."
            ),
            (
                claim_oled,
                spec_sources,
                oled_leaf,
                "From the provided URL(s), confirm the panel technology is explicitly OLED. "
                "Do not accept IPS, VA, or other non-OLED technologies."
            ),
            (
                claim_gpu,
                spec_sources,
                gpu_leaf,
                "From the provided URL(s), confirm the GPU model is exactly NVIDIA GeForce RTX 5080 or RTX 5090 (laptop GPU). "
                "Merely belonging to 5000-series is insufficient unless explicitly 5080 or 5090."
            ),
            (
                claim_weight,
                spec_sources,
                weight_leaf,
                "From the provided URL(s), confirm the listed weight is 5.0 lb (2.27 kg) or less. "
                "If shown in kilograms, convert mentally to ensure it is ≤ 2.27 kg. "
                "If multiple configurations are listed, ensure the specified model/config is within the limit."
            ),
            (
                claim_tb,
                spec_sources,
                tb_leaf,
                "From the provided URL(s), confirm at least one Thunderbolt port (3, 4, or 5). "
                "Accept 'USB4 with Thunderbolt' if it clearly denotes Thunderbolt compatibility. "
                "Do not accept plain USB-C without an explicit Thunderbolt reference."
            ),
            (
                claim_battery,
                spec_sources,
                battery_leaf,
                "From the provided URL(s), confirm the battery is at least 90 Wh. "
                "Accept values like 90 Wh, 91 Wh, 99.9 Wh, etc."
            ),
        ]
    )

    # 8) Supporting references (break down into concrete existence checks under a critical parent)
    refs_parent = evaluator.add_parallel(
        id="supporting_references",
        desc="Provides supporting reference URL(s) that substantiate the state’s Micro Center store count, the laptop’s in-store availability in that state, and each claimed specification (display, GPU, weight, Thunderbolt, battery)",
        parent=task_root,
        critical=True
    )
    evaluator.add_custom_node(
        result=len(store_count_sources) > 0,
        id="refs_store_count_present",
        desc="Has at least one URL supporting the state's 'exactly 3 Micro Center stores' claim",
        parent=refs_parent,
        critical=True
    )
    evaluator.add_custom_node(
        result=len(availability_sources) > 0,
        id="refs_instore_availability_present",
        desc="Has at least one URL supporting in-store availability of the laptop in the named state",
        parent=refs_parent,
        critical=True
    )
    evaluator.add_custom_node(
        result=len(spec_sources) > 0,
        id="refs_specs_present",
        desc="Has at least one URL supporting the claimed specifications (display, GPU, weight, Thunderbolt, battery)",
        parent=refs_parent,
        critical=True
    )

    # Optionally record some counts as custom info for transparency
    evaluator.add_custom_info(
        info={
            "state": state or "",
            "brand": brand or "",
            "model": model or "",
            "store_count_sources_count": len(store_count_sources),
            "availability_sources_count": len(availability_sources),
            "spec_sources_count": len(spec_sources),
        },
        info_type="debug",
        info_name="extracted_highlights"
    )


# --------------------------------------------------------------------------- #
# Main evaluation entry point                                                 #
# --------------------------------------------------------------------------- #
async def evaluate_answer(
    client: LLMClient,
    answer: str,
    agent_name: str,
    answer_name: str,
    cache: CacheFileSys,
    semaphore: asyncio.Semaphore,
    logger: logging.Logger,
    model: str = "o4-mini"
) -> Dict:
    """
    Entry point for evaluating an answer for the gaming laptop Micro Center (exactly 3 stores) task.
    """
    # Initialize evaluator
    evaluator = Evaluator()
    evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,   # We add our own critical task_root under this
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

    # Extract structured information from the answer
    extracted: LaptopExtraction = await evaluator.extract(
        prompt=prompt_extract_laptop_candidate(),
        template_class=LaptopExtraction,
        extraction_name="laptop_candidate"
    )

    # Build tree and verify according to rubric
    await build_and_verify(evaluator, extracted)

    # Return summary
    return evaluator.get_summary()