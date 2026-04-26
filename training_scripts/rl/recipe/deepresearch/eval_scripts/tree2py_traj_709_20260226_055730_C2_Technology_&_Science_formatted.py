import asyncio
import logging
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "si_c_battery_phone_2025"
TASK_DESCRIPTION = (
    "In late 2025, several smartphone manufacturers began adopting advanced silicon-carbon battery technology to "
    "achieve significantly higher battery capacities. Identify a smartphone model that meets ALL of the following "
    "specifications:\n\n"
    "1. Uses silicon-carbon (Si/C) battery technology\n"
    "2. Has a battery capacity of at least 7,000 mAh\n"
    "3. Was released or made available for pre-order in the United States between November 1, 2025, and December 31, 2025\n"
    "4. Supports wireless charging at a power level of 50W or higher\n"
    "5. Is powered by the Qualcomm Snapdragon 8 Elite Gen 5 processor\n\n"
    "Provide the specific model name of the smartphone and include reference URLs from official sources to verify each specification."
)

# Official source policy text for verification instructions
OFFICIAL_SOURCE_GUIDELINES = (
    "IMPORTANT SOURCE POLICY:\n"
    "- Treat a source as official only if it is one of the following:\n"
    "  (a) The smartphone manufacturer's official domain (e.g., brand.com), including official press releases/blog pages hosted on the brand's domain;\n"
    "  (b) Qualcomm's official domain (qualcomm.com) if verifying processor information;\n"
    "  (c) Major U.S. carrier official domains for release/pre-order information (e.g., verizon.com, att.com, t-mobile.com) or official U.S. retail partners (e.g., bestbuy.com);\n"
    "  (d) U.S.-localized manufacturer product pages or support pages (e.g., us.brand.com or brand.com/us).\n"
    "- Do NOT accept third-party news sites, rumor forums, or aggregators as sufficient support.\n"
    "- If the provided URL does not clearly appear to be an official source as defined, you should consider the claim not supported."
)


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class SmartphoneExtraction(BaseModel):
    # Core identification
    model_name: Optional[str] = None

    # Battery technology (Si/C)
    battery_tech_text: Optional[str] = None
    battery_tech_urls: List[str] = Field(default_factory=list)

    # Battery capacity
    battery_capacity_text: Optional[str] = None
    battery_capacity_urls: List[str] = Field(default_factory=list)

    # Wireless charging capability
    wireless_charging_text: Optional[str] = None
    wireless_charging_urls: List[str] = Field(default_factory=list)

    # Processor
    processor_text: Optional[str] = None
    processor_urls: List[str] = Field(default_factory=list)

    # U.S. release/pre-order window
    us_availability_text: Optional[str] = None
    us_availability_urls: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_smartphone() -> str:
    return """
    Extract a single smartphone model and the exact sources the answer uses to support each required specification.

    Required fields to extract:
    1) model_name: The exact smartphone model name as stated in the answer.
    2) battery_tech_text: The exact phrasing used in the answer for the battery technology (e.g., "silicon-carbon", "Si/C", "Si‑C").
    3) battery_tech_urls: All URLs in the answer that are intended to support the battery technology claim.
    4) battery_capacity_text: The capacity value or description mentioned in the answer (e.g., "7,200 mAh", "7000mAh").
    5) battery_capacity_urls: All URLs in the answer that are intended to support the battery capacity.
    6) wireless_charging_text: The wireless charging power level as written (e.g., "50W", "55 W wireless").
    7) wireless_charging_urls: All URLs in the answer that are intended to support the wireless charging specification.
    8) processor_text: The processor designation as written (e.g., "Qualcomm Snapdragon 8 Elite Gen 5").
    9) processor_urls: All URLs in the answer that are intended to support the processor specification.
    10) us_availability_text: The wording in the answer describing U.S. release or pre-order timing (e.g., dates, phrases like "available for pre-order in the U.S. on Nov 15, 2025").
    11) us_availability_urls: All URLs in the answer that are intended to support the U.S. release/pre-order timeframe.

    Instructions:
    - Only extract URLs that are explicitly present in the answer. Do not invent URLs.
    - If the answer provides a single combined sources section, assign the relevant URLs to each field only if they plausibly support that specific specification based on the answer's wording. If a URL clearly supports multiple specs, include it in the corresponding multiple lists.
    - If a field is not present in the answer, set its text to null and its URLs to an empty list.
    - If multiple smartphone models are mentioned, choose the one the answer most clearly claims meets all five requirements; if ambiguous, pick the first model mentioned.
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _has_urls(urls: Optional[List[str]]) -> bool:
    return bool(urls) and isinstance(urls, list) and len([u for u in urls if isinstance(u, str) and u.strip()]) > 0


def _combine_instruction(spec_instruction: str) -> str:
    return f"{spec_instruction}\n\n{OFFICIAL_SOURCE_GUIDELINES}"


# --------------------------------------------------------------------------- #
# Verification tree builders                                                  #
# --------------------------------------------------------------------------- #
async def add_us_market_release_check(
    evaluator: Evaluator,
    parent_node,
    model_name: Optional[str],
    us_text: Optional[str],
    us_urls: List[str]
) -> None:
    us_node = evaluator.add_parallel(
        id="us_market_release",
        desc="The device was released or made available for pre-order in the United States between November 1, 2025, and December 31, 2025",
        parent=parent_node,
        critical=True
    )

    evaluator.add_custom_node(
        result=_has_urls(us_urls),
        id="us_release_sources_present",
        desc="Sources are provided for U.S. release/pre-order timeframe",
        parent=us_node,
        critical=True
    )

    us_leaf = evaluator.add_leaf(
        id="us_release_supported",
        desc="U.S. release/pre-order timeframe is confirmed by official sources",
        parent=us_node,
        critical=True
    )

    claim_model = model_name or "the smartphone model"
    claim = (
        f"{claim_model} was released or available for pre-order in the United States between November 1, 2025 and "
        f"December 31, 2025 (inclusive)."
    )
    add_ins = _combine_instruction(
        "Accept either 'release' or 'available for pre-order' as satisfying the requirement. "
        "The page must clearly indicate the U.S. market context and a date within the inclusive range "
        "2025-11-01 to 2025-12-31. If multiple dates are shown, prefer the U.S.-specific date. "
        "Carrier or U.S. retail partner pages are acceptable for this check."
    )

    await evaluator.verify(
        claim=claim,
        node=us_leaf,
        sources=us_urls,
        additional_instruction=add_ins
    )


async def add_battery_requirements_checks(
    evaluator: Evaluator,
    parent_node,
    model_name: Optional[str],
    tech_text: Optional[str],
    tech_urls: List[str],
    cap_text: Optional[str],
    cap_urls: List[str]
) -> None:
    battery_node = evaluator.add_parallel(
        id="battery_requirements",
        desc="The device meets both battery technology and capacity requirements",
        parent=parent_node,
        critical=True
    )

    # Silicon-Carbon technology
    si_c_node = evaluator.add_parallel(
        id="silicon_carbon_technology",
        desc="The device uses silicon-carbon (Si/C) battery technology",
        parent=battery_node,
        critical=True
    )

    evaluator.add_custom_node(
        result=_has_urls(tech_urls),
        id="silicon_carbon_sources_present",
        desc="Sources are provided for silicon-carbon battery technology",
        parent=si_c_node,
        critical=True
    )

    si_leaf = evaluator.add_leaf(
        id="silicon_carbon_supported",
        desc="Silicon-carbon battery technology is supported by official sources",
        parent=si_c_node,
        critical=True
    )

    claim_model = model_name or "the smartphone model"
    claim = f"{claim_model} uses silicon‑carbon (Si/C) battery technology."
    add_ins = _combine_instruction(
        "Look for explicit mentions of 'silicon‑carbon', 'silicon carbon', 'Si/C', or 'Si‑C' in connection with the battery. "
        "Do not accept generic references to 'silicon anode' without clear indication of silicon‑carbon battery technology."
    )
    await evaluator.verify(
        claim=claim,
        node=si_leaf,
        sources=tech_urls,
        additional_instruction=add_ins
    )

    # Battery capacity >= 7000 mAh
    cap_node = evaluator.add_parallel(
        id="battery_capacity",
        desc="The battery capacity is at least 7,000 mAh",
        parent=battery_node,
        critical=True
    )

    evaluator.add_custom_node(
        result=_has_urls(cap_urls),
        id="battery_capacity_sources_present",
        desc="Sources are provided for battery capacity",
        parent=cap_node,
        critical=True
    )

    cap_leaf = evaluator.add_leaf(
        id="battery_capacity_supported",
        desc="Battery capacity ≥ 7,000 mAh is supported by official sources",
        parent=cap_node,
        critical=True
    )

    claim = f"{claim_model} has a battery capacity of at least 7,000 mAh."
    add_ins = _combine_instruction(
        "Confirm that the capacity shown on the page is ≥ 7000 mAh. Accept numeric variants such as '7000mAh', '7,000 mAh', or higher. "
        "If multiple variants or regional specs are present, ensure the capacity corresponds to the specific model/variant in question."
    )
    await evaluator.verify(
        claim=claim,
        node=cap_leaf,
        sources=cap_urls,
        additional_instruction=add_ins
    )


async def add_wireless_charging_check(
    evaluator: Evaluator,
    parent_node,
    model_name: Optional[str],
    wc_text: Optional[str],
    wc_urls: List[str]
) -> None:
    wc_node = evaluator.add_parallel(
        id="wireless_charging_capability",
        desc="The device supports wireless charging at 50W or higher power level",
        parent=parent_node,
        critical=True
    )

    evaluator.add_custom_node(
        result=_has_urls(wc_urls),
        id="wireless_charging_sources_present",
        desc="Sources are provided for wireless charging power",
        parent=wc_node,
        critical=True
    )

    wc_leaf = evaluator.add_leaf(
        id="wireless_charging_supported",
        desc="Wireless charging ≥ 50W is supported by official sources",
        parent=wc_node,
        critical=True
    )

    claim_model = model_name or "the smartphone model"
    claim = f"{claim_model} supports wireless charging at 50W or higher."
    add_ins = _combine_instruction(
        "Verify that the page clearly states wireless charging power of ≥ 50W. "
        "Do not confuse wired charging with wireless. Accept brand names for wireless charging (e.g., Qi2, MagSafe, proprietary names) "
        "so long as the power figure applies to wireless charging."
    )
    await evaluator.verify(
        claim=claim,
        node=wc_leaf,
        sources=wc_urls,
        additional_instruction=add_ins
    )


async def add_processor_check(
    evaluator: Evaluator,
    parent_node,
    model_name: Optional[str],
    proc_text: Optional[str],
    proc_urls: List[str]
) -> None:
    proc_node = evaluator.add_parallel(
        id="processor_specification",
        desc="The device is powered by the Qualcomm Snapdragon 8 Elite Gen 5 processor",
        parent=parent_node,
        critical=True
    )

    evaluator.add_custom_node(
        result=_has_urls(proc_urls),
        id="processor_sources_present",
        desc="Sources are provided for processor specification",
        parent=proc_node,
        critical=True
    )

    proc_leaf = evaluator.add_leaf(
        id="processor_supported",
        desc="Processor is confirmed as Qualcomm Snapdragon 8 Elite Gen 5 by official sources",
        parent=proc_node,
        critical=True
    )

    claim_model = model_name or "the smartphone model"
    claim = f"{claim_model} is powered by the Qualcomm Snapdragon 8 Elite Gen 5 processor."
    add_ins = _combine_instruction(
        "Allow minor naming variations (e.g., 'Snapdragon 8 Elite Gen5', 'SD 8 Elite Gen 5') as equivalent. "
        "Do not accept different generations (e.g., Gen 4) or different chip families."
    )
    await evaluator.verify(
        claim=claim,
        node=proc_leaf,
        sources=proc_urls,
        additional_instruction=add_ins
    )


# --------------------------------------------------------------------------- #
# Main evaluation entry point                                                 #
# --------------------------------------------------------------------------- #
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
        strategy=AggregationStrategy.PARALLEL,
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

    extracted = await evaluator.extract(
        prompt=prompt_extract_smartphone(),
        template_class=SmartphoneExtraction,
        extraction_name="smartphone_extraction"
    )

    # Add top-level node corresponding to "Smartphone_Identification"
    top_node = evaluator.add_parallel(
        id="smartphone_identification",
        desc="Identify a smartphone meeting all specified technical requirements",
        parent=root,
        critical=False
    )

    # Optional: Model name existence (non-critical informational gate)
    evaluator.add_custom_node(
        result=bool(extracted.model_name and extracted.model_name.strip()),
        id="model_name_provided",
        desc="Model name is provided in the answer",
        parent=top_node,
        critical=False
    )

    # US Market Release (CRITICAL)
    await add_us_market_release_check(
        evaluator=evaluator,
        parent_node=top_node,
        model_name=extracted.model_name,
        us_text=extracted.us_availability_text,
        us_urls=extracted.us_availability_urls
    )

    # Technical Specifications (CRITICAL)
    tech_specs_node = evaluator.add_parallel(
        id="technical_specifications",
        desc="The device meets all required technical specifications",
        parent=top_node,
        critical=True
    )

    # Battery requirements (Si/C + capacity)
    await add_battery_requirements_checks(
        evaluator=evaluator,
        parent_node=tech_specs_node,
        model_name=extracted.model_name,
        tech_text=extracted.battery_tech_text,
        tech_urls=extracted.battery_tech_urls,
        cap_text=extracted.battery_capacity_text,
        cap_urls=extracted.battery_capacity_urls
    )

    # Wireless charging
    await add_wireless_charging_check(
        evaluator=evaluator,
        parent_node=tech_specs_node,
        model_name=extracted.model_name,
        wc_text=extracted.wireless_charging_text,
        wc_urls=extracted.wireless_charging_urls
    )

    # Processor
    await add_processor_check(
        evaluator=evaluator,
        parent_node=tech_specs_node,
        model_name=extracted.model_name,
        proc_text=extracted.processor_text,
        proc_urls=extracted.processor_urls
    )

    # Add custom info for timeframe to aid downstream interpretation
    evaluator.add_custom_info(
        info={
            "required_us_time_window": "2025-11-01 to 2025-12-31 (inclusive)",
            "minimum_battery_capacity_mAh": "7000",
            "minimum_wireless_charging_watt": "50",
            "required_processor": "Qualcomm Snapdragon 8 Elite Gen 5"
        },
        info_type="constraints",
        info_name="constraint_parameters"
    )

    return evaluator.get_summary()