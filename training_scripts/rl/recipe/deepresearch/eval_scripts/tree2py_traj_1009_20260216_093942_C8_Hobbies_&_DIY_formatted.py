import asyncio
import logging
from typing import Optional, List, Dict, Any, Tuple

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "craft_vendor_planning_2026"
TASK_DESCRIPTION = """
I am a craft vendor planning to expand my business by participating in holiday craft markets across three major U.S. metropolitan areas during the 2026 holiday season (November-December 2026). I need to research and compile detailed vendor information for each of three metropolitan areas: Chicago, Illinois; Los Angeles, California; and Seattle, Washington.

For each metropolitan area, provide the following information:

1. Market Infrastructure: Identify at least one craft or art supply store in the area, including its name and complete street address
2. Typical Vendor Booth Specifications: The standard booth size used at craft fairs (typically 10x10 feet) and the general price range for holiday market booth fees in the area
3. Vendor Requirements: Whether liability insurance is typically required for craft market vendors and whether a sales tax permit is needed
4. Holiday Market Timing: The typical load-in time schedule for vendors (usually 8:00-9:00 AM) and when applications are typically due (usually 60-120 days before events)
5. Makerspace Resources: If seeking tool access for craft production, identify whether there is a community makerspace in the area with typical monthly membership fees

Provide this comprehensive vendor planning information for all three metropolitan areas, with all store addresses and facility information verifiable through reliable sources.
"""


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class MetroAreaExtraction(BaseModel):
    # Market Infrastructure
    store_name: Optional[str] = None
    store_address: Optional[str] = None
    store_sources: List[str] = Field(default_factory=list)

    # Typical Vendor Booth Specifications
    booth_size: Optional[str] = None
    booth_fee_range: Optional[str] = None
    booth_sources: List[str] = Field(default_factory=list)

    # Vendor Requirements
    liability_insurance_required: Optional[str] = None
    li_sources: List[str] = Field(default_factory=list)
    sales_tax_permit_required: Optional[str] = None
    tax_sources: List[str] = Field(default_factory=list)

    # Holiday Market Timing
    load_in_time_window: Optional[str] = None
    load_in_sources: List[str] = Field(default_factory=list)
    application_deadline_timeframe: Optional[str] = None
    app_deadline_sources: List[str] = Field(default_factory=list)

    # Makerspace Resources
    makerspace_exists: Optional[bool] = None
    makerspace_name: Optional[str] = None
    makerspace_name_sources: List[str] = Field(default_factory=list)
    makerspace_monthly_fee: Optional[str] = None
    makerspace_fee_sources: List[str] = Field(default_factory=list)
    makerspace_training_required: Optional[str] = None
    makerspace_training_duration: Optional[str] = None
    makerspace_training_sources: List[str] = Field(default_factory=list)


class VendorPlanningExtraction(BaseModel):
    chicago: Optional[MetroAreaExtraction] = None
    los_angeles: Optional[MetroAreaExtraction] = None
    seattle: Optional[MetroAreaExtraction] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_vendor_planning() -> str:
    return """
Extract structured vendor-planning information for three metropolitan areas: Chicago (Illinois), Los Angeles (California), and Seattle (Washington).

You must extract the following fields for each area. For each factual item, also extract all supporting source URLs that the answer explicitly provides (if any). If a field is missing, set it to null; if no sources are provided, return an empty list for the corresponding sources.

For each metro area, extract:

- store_name: The name of one craft/art supply store in the metro area.
- store_address: The complete street address for that store (street number, street name, city, state, ZIP).
- store_sources: Array of URLs that support the store and its address (e.g., store website, Google Maps).

- booth_size: The typical/standard craft-fair booth size used in the area (e.g., "10x10 ft").
- booth_fee_range: The general price range for holiday market booth fees in the area (e.g., "$150–$500").
- booth_sources: Array of URLs that support booth size and/or fee range for a representative event(s) in the area.

- liability_insurance_required: Whether liability insurance is typically required for craft market vendors in the area. Express in your own words but concise, such as "Yes, typically required", "No, not typically required", or "Varies by market".
- li_sources: Array of URLs that support the above statement (e.g., vendor handbook, event application).

- sales_tax_permit_required: Whether a sales tax permit/registration is required for vendors in the jurisdiction. Express concisely as above.
- tax_sources: Array of URLs that support the above statement (e.g., state Dept. of Revenue, event guidance).

- load_in_time_window: The typical vendor load-in time window (e.g., "8:00–9:00 AM").
- load_in_sources: Array of URLs that support typical load-in timing (e.g., sample event schedule in the area).

- application_deadline_timeframe: When vendor applications are typically due relative to the event date (e.g., "60–120 days before").
- app_deadline_sources: Array of URLs to event pages or guidelines supporting that timeframe.

- makerspace_exists: true/false indicating whether there is a community makerspace in the metro area.
- makerspace_name: If makerspace_exists is true, provide the name of at least one community makerspace.
- makerspace_name_sources: Array of URLs supporting the makerspace name and location (e.g., official website, about page).
- makerspace_monthly_fee: If a makerspace is provided, the typical monthly membership fee (e.g., "$60/month", "from $45/mo").
- makerspace_fee_sources: Array of URLs supporting the monthly fee.
- makerspace_training_required: If tool access is discussed, whether safety training is required (e.g., "Yes, safety orientation required").
- makerspace_training_duration: If training is discussed, typical duration (e.g., "1–2 hours").
- makerspace_training_sources: Array of URLs supporting training requirement/duration.

Return a JSON with the top-level keys: chicago, los_angeles, seattle. Each key maps to an object containing the above fields.

Special rules:
- Do not invent URLs; only extract URLs explicitly present in the answer.
- If a URL is missing a protocol, prepend http://.
- Keep values as strings to allow ranges and qualitative phrasing.
"""


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _has_text(x: Optional[str]) -> bool:
    return bool(x) and bool(str(x).strip())


def _has_sources(urls: Optional[List[str]]) -> bool:
    return bool(urls) and len(urls) > 0


def _normalize_requirement_label(label: Optional[str]) -> str:
    if not _has_text(label):
        return "unknown"
    s = str(label).strip().lower()
    if any(k in s for k in ["varies", "depends"]):
        return "varies"
    if any(k in s for k in ["no", "not required", "optional"]):
        return "no"
    if any(k in s for k in ["yes", "required", "must carry", "must provide"]):
        return "yes"
    return "unknown"


def _area_display(area_key: str) -> str:
    return {
        "chicago": "Chicago, Illinois",
        "los_angeles": "Los Angeles, California",
        "seattle": "Seattle, Washington",
    }.get(area_key, area_key)


def _collect_required_source_presence(area_key: str, area: Optional[MetroAreaExtraction]) -> Tuple[bool, List[str]]:
    """
    Compute whether all required items have at least one supporting source URL.
    Returns (result, missing_items_list).
    """
    if area is None:
        return False, [f"{area_key}: all fields missing"]
    missing: List[str] = []

    # Store
    if not (_has_text(area.store_name) and _has_text(area.store_address) and _has_sources(area.store_sources)):
        missing.append(f"{area_key}: store name/address or sources")

    # Booth specs (size or fee range must be present with sources? Rubric requires both; enforce both)
    if not (_has_text(area.booth_size) and _has_sources(area.booth_sources)):
        missing.append(f"{area_key}: booth size sources")
    if not (_has_text(area.booth_fee_range) and _has_sources(area.booth_sources)):
        missing.append(f"{area_key}: booth fee range sources")

    # Requirements
    if not (_has_text(area.liability_insurance_required) and _has_sources(area.li_sources)):
        missing.append(f"{area_key}: liability insurance sources")
    if not (_has_text(area.sales_tax_permit_required) and _has_sources(area.tax_sources)):
        missing.append(f"{area_key}: sales tax permit sources")

    # Timing
    if not (_has_text(area.load_in_time_window) and _has_sources(area.load_in_sources)):
        missing.append(f"{area_key}: load-in time sources")
    if not (_has_text(area.application_deadline_timeframe) and _has_sources(area.app_deadline_sources)):
        missing.append(f"{area_key}: application deadline sources")

    # Makerspace
    if area.makerspace_exists:
        if not (_has_text(area.makerspace_name) and _has_sources(area.makerspace_name_sources)):
            missing.append(f"{area_key}: makerspace name/sources")
        if not (_has_text(area.makerspace_monthly_fee) and _has_sources(area.makerspace_fee_sources)):
            missing.append(f"{area_key}: makerspace fee/sources")
        # Training is marked as critical in rubric ("if applicable"); treat as required when makerspace is discussed
        if not (_has_text(area.makerspace_training_required) and _has_text(area.makerspace_training_duration) and _has_sources(area.makerspace_training_sources)):
            missing.append(f"{area_key}: makerspace training requirement/duration/sources")

    return (len(missing) == 0), missing


# --------------------------------------------------------------------------- #
# Verification for each area                                                  #
# --------------------------------------------------------------------------- #
async def verify_metro_area(
    evaluator: Evaluator,
    parent_node,
    area_key: str,
    area_data: Optional[MetroAreaExtraction],
) -> None:
    """
    Build the verification subtree for a single metropolitan area.
    All nodes are critical because the parent in rubric is critical.
    """
    display = _area_display(area_key)

    area_node = evaluator.add_parallel(
        id=f"{area_key}_area_info",
        desc=f"{display} metropolitan area: provide all required planning information",
        parent=parent_node,
        critical=True
    )

    # 1) Craft store with full address
    if area_data and _has_text(area_data.store_name) and _has_text(area_data.store_address) and _has_sources(area_data.store_sources):
        node = evaluator.add_leaf(
            id=f"{area_key}_craft_store_with_full_address",
            desc="Provide at least one craft/art supply store with store name and complete street address",
            parent=area_node,
            critical=True
        )
        claim = f"The cited source(s) show a store named '{area_data.store_name}' with the full street address '{area_data.store_address}'."
        await evaluator.verify(
            claim=claim,
            node=node,
            sources=area_data.store_sources,
            additional_instruction="Accept the claim if the page clearly shows both the store name and the complete street address (street number, street name, city, state, ZIP). Official store websites or Google Maps pages are acceptable."
        )
    else:
        evaluator.add_custom_node(
            result=False,
            id=f"{area_key}_craft_store_with_full_address",
            desc="Provide at least one craft/art supply store with store name and complete street address",
            parent=area_node,
            critical=True
        )

    # 2) Booth size reported
    if area_data and _has_text(area_data.booth_size) and _has_sources(area_data.booth_sources):
        node = evaluator.add_leaf(
            id=f"{area_key}_booth_size_reported",
            desc="State the typical/standard craft-fair booth size for the area (e.g., 10x10 ft)",
            parent=area_node,
            critical=True
        )
        claim = f"A representative holiday market or craft fair in {display} uses a typical/standard booth size of '{area_data.booth_size}'."
        await evaluator.verify(
            claim=claim,
            node=node,
            sources=area_data.booth_sources,
            additional_instruction="It is sufficient if one or more representative local event pages show the standard booth size (e.g., 10x10). Allow minor formatting variations like 10’x10’."
        )
    else:
        evaluator.add_custom_node(
            result=False,
            id=f"{area_key}_booth_size_reported",
            desc="State the typical/standard craft-fair booth size for the area (e.g., 10x10 ft)",
            parent=area_node,
            critical=True
        )

    # 3) Booth fee range reported
    if area_data and _has_text(area_data.booth_fee_range) and _has_sources(area_data.booth_sources):
        node = evaluator.add_leaf(
            id=f"{area_key}_booth_fee_range_reported",
            desc="Provide a typical holiday-market booth-fee price range for the area",
            parent=area_node,
            critical=True
        )
        claim = f"A representative holiday market in {display} shows a typical booth fee range of '{area_data.booth_fee_range}'."
        await evaluator.verify(
            claim=claim,
            node=node,
            sources=area_data.booth_sources,
            additional_instruction="Representative events in the metro area are sufficient to support a typical fee range. Accept ranges or indicative 'starting at' values if clearly applicable."
        )
    else:
        evaluator.add_custom_node(
            result=False,
            id=f"{area_key}_booth_fee_range_reported",
            desc="Provide a typical holiday-market booth-fee price range for the area",
            parent=area_node,
            critical=True
        )

    # 4) Liability insurance requirement
    if area_data and _has_text(area_data.liability_insurance_required) and _has_sources(area_data.li_sources):
        node = evaluator.add_leaf(
            id=f"{area_key}_liability_insurance_requirement",
            desc="Address whether liability insurance is typically required for craft market vendors",
            parent=area_node,
            critical=True
        )
        label = _normalize_requirement_label(area_data.liability_insurance_required)
        if label == "yes":
            claim = f"In {display}, liability insurance is typically required for craft market vendors."
        elif label == "no":
            claim = f"In {display}, liability insurance is not typically required for craft market vendors."
        elif label == "varies":
            claim = f"In {display}, liability insurance requirements vary by market for craft vendors."
        else:
            claim = f"In {display}, there is a specific stated position on liability insurance requirements for craft market vendors as described: '{area_data.liability_insurance_required}'."
        await evaluator.verify(
            claim=claim,
            node=node,
            sources=area_data.li_sources,
            additional_instruction="Prefer vendor handbooks, event application pages, or official guidance. Accept if the cited pages clearly convey the requirement (or lack thereof), or indicate variability across events."
        )
    else:
        evaluator.add_custom_node(
            result=False,
            id=f"{area_key}_liability_insurance_requirement",
            desc="Address whether liability insurance is typically required for craft market vendors",
            parent=area_node,
            critical=True
        )

    # 5) Sales tax permit requirement
    if area_data and _has_text(area_data.sales_tax_permit_required) and _has_sources(area_data.tax_sources):
        node = evaluator.add_leaf(
            id=f"{area_key}_sales_tax_permit_requirement",
            desc="Address whether a sales tax permit/registration is required in the jurisdiction",
            parent=area_node,
            critical=True
        )
        label = _normalize_requirement_label(area_data.sales_tax_permit_required)
        if label == "yes":
            claim = f"In {display}, vendors typically need to register for sales tax or hold a sales tax permit."
        elif label == "no":
            claim = f"In {display}, vendors are typically not required to register for sales tax or hold a sales tax permit."
        elif label == "varies":
            claim = f"In {display}, sales tax permit requirements for vendors vary by circumstances or event."
        else:
            claim = f"In {display}, the answer states: '{area_data.sales_tax_permit_required}' about sales tax permits for vendors."
        await evaluator.verify(
            claim=claim,
            node=node,
            sources=area_data.tax_sources,
            additional_instruction="Prefer official state or city tax authority pages or event guidance that addresses vendor tax registration requirements."
        )
    else:
        evaluator.add_custom_node(
            result=False,
            id=f"{area_key}_sales_tax_permit_requirement",
            desc="Address whether a sales tax permit/registration is required in the jurisdiction",
            parent=area_node,
            critical=True
        )

    # 6) Load-in time window
    if area_data and _has_text(area_data.load_in_time_window) and _has_sources(area_data.load_in_sources):
        node = evaluator.add_leaf(
            id=f"{area_key}_load_in_time_window",
            desc="Provide the typical vendor load-in time window",
            parent=area_node,
            critical=True
        )
        claim = f"In {display}, representative event(s) show a typical vendor load-in time window of '{area_data.load_in_time_window}'."
        await evaluator.verify(
            claim=claim,
            node=node,
            sources=area_data.load_in_sources,
            additional_instruction="Look for event schedules or vendor info pages specifying vendor load-in times. Accept approximate times around 8:00–9:00 AM as typical if supported."
        )
    else:
        evaluator.add_custom_node(
            result=False,
            id=f"{area_key}_load_in_time_window",
            desc="Provide the typical vendor load-in time window",
            parent=area_node,
            critical=True
        )

    # 7) Application deadline timeframe
    if area_data and _has_text(area_data.application_deadline_timeframe) and _has_sources(area_data.app_deadline_sources):
        node = evaluator.add_leaf(
            id=f"{area_key}_application_deadline_timeframe",
            desc="Provide the typical application deadline timeframe (e.g., ~60–120 days prior)",
            parent=area_node,
            critical=True
        )
        claim = f"In {display}, representative event(s) indicate that vendor applications are typically due '{area_data.application_deadline_timeframe}' before the event."
        await evaluator.verify(
            claim=claim,
            node=node,
            sources=area_data.app_deadline_sources,
            additional_instruction="Representative event timelines are sufficient. Accept ranges and approximate phrasing if the source supports it."
        )
    else:
        evaluator.add_custom_node(
            result=False,
            id=f"{area_key}_application_deadline_timeframe",
            desc="Provide the typical application deadline timeframe (e.g., ~60–120 days prior)",
            parent=area_node,
            critical=True
        )

    # 8) Makerspace existence and name
    if area_data and area_data.makerspace_exists and _has_text(area_data.makerspace_name) and _has_sources(area_data.makerspace_name_sources):
        node = evaluator.add_leaf(
            id=f"{area_key}_makerspace_existence_and_name",
            desc="Indicate whether there is a community makerspace; identify at least one by name",
            parent=area_node,
            critical=True
        )
        claim = f"There is a community makerspace in {display} named '{area_data.makerspace_name}'."
        await evaluator.verify(
            claim=claim,
            node=node,
            sources=area_data.makerspace_name_sources,
            additional_instruction="A community makerspace is a public-access or membership-based workshop/hackerspace. Verify the name and its presence in the metro area using the official site or authoritative directory page."
        )
    else:
        evaluator.add_custom_node(
            result=False,
            id=f"{area_key}_makerspace_existence_and_name",
            desc="Indicate whether there is a community makerspace; identify at least one by name",
            parent=area_node,
            critical=True
        )

    # 9) Makerspace monthly fees
    if area_data and area_data.makerspace_exists and _has_text(area_data.makerspace_name) and _has_text(area_data.makerspace_monthly_fee) and _has_sources(area_data.makerspace_fee_sources):
        node = evaluator.add_leaf(
            id=f"{area_key}_makerspace_monthly_fees_if_applicable",
            desc="Provide typical monthly membership fees for the makerspace",
            parent=area_node,
            critical=True
        )
        claim = f"The typical monthly membership fee for '{area_data.makerspace_name}' is '{area_data.makerspace_monthly_fee}'."
        await evaluator.verify(
            claim=claim,
            node=node,
            sources=area_data.makerspace_fee_sources,
            additional_instruction="Check the membership or pricing page. Accept 'from $X/month' or tiered pricing that clearly indicates a monthly amount."
        )
    else:
        evaluator.add_custom_node(
            result=False,
            id=f"{area_key}_makerspace_monthly_fees_if_applicable",
            desc="Provide typical monthly membership fees for the makerspace",
            parent=area_node,
            critical=True
        )

    # 10) Tool safety training (if applicable)
    if area_data and area_data.makerspace_exists and _has_text(area_data.makerspace_training_required) and _has_text(area_data.makerspace_training_duration) and _has_sources(area_data.makerspace_training_sources):
        node = evaluator.add_leaf(
            id=f"{area_key}_tool_safety_training_if_applicable",
            desc="If makerspace tool access is discussed, note if training is required and typical duration",
            parent=area_node,
            critical=True
        )
        claim = f"At '{area_data.makerspace_name}', tool safety training is required and the typical duration is '{area_data.makerspace_training_duration}'."
        await evaluator.verify(
            claim=claim,
            node=node,
            sources=area_data.makerspace_training_sources,
            additional_instruction="Check safety orientation or tool training pages. Accept similar terminology (orientation, safety class) and allow stated durations around the value provided."
        )
    else:
        evaluator.add_custom_node(
            result=False,
            id=f"{area_key}_tool_safety_training_if_applicable",
            desc="If makerspace tool access is discussed, note if training is required and typical duration",
            parent=area_node,
            critical=True
        )


# --------------------------------------------------------------------------- #
# Global verifiability check                                                  #
# --------------------------------------------------------------------------- #
def add_global_verifiability_node(evaluator: Evaluator, parent_node, extraction: VendorPlanningExtraction) -> None:
    problems: List[str] = []

    for key in ["chicago", "los_angeles", "seattle"]:
        area: Optional[MetroAreaExtraction] = getattr(extraction, key, None)
        ok, missing_items = _collect_required_source_presence(key, area)
        if not ok:
            problems.extend(missing_items)

    result = len(problems) == 0
    evaluator.add_custom_node(
        result=result,
        id="global_verifiability_no_fabrication",
        desc="All factual claims are supported by provided source URLs; no fabrications",
        parent=parent_node,
        critical=True
    )
    evaluator.add_custom_info(
        info={"missing_or_unsupported_items": problems},
        info_type="diagnostics",
        info_name="global_verifiability_diagnostics"
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
) -> Dict[str, Any]:
    """
    Evaluate an answer for the craft-vendor planning task across Chicago, Los Angeles, and Seattle.
    """
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

    # Extraction
    extraction = await evaluator.extract(
        prompt=prompt_extract_vendor_planning(),
        template_class=VendorPlanningExtraction,
        extraction_name="vendor_planning_extraction"
    )

    # Root is critical; add global verifiability check (critical)
    add_global_verifiability_node(evaluator, root, extraction)

    # Area subtrees (must be critical because root is critical)
    # Chicago
    await verify_metro_area(
        evaluator=evaluator,
        parent_node=root,
        area_key="chicago",
        area_data=extraction.chicago
    )

    # Los Angeles
    await verify_metro_area(
        evaluator=evaluator,
        parent_node=root,
        area_key="los_angeles",
        area_data=extraction.los_angeles
    )

    # Seattle
    await verify_metro_area(
        evaluator=evaluator,
        parent_node=root,
        area_key="seattle",
        area_data=extraction.seattle
    )

    return evaluator.get_summary()