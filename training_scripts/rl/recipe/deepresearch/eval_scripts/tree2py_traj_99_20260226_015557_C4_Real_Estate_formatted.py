import asyncio
import logging
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.llm_client.base_client import LLMClient


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "meck_warehouse_tax_2026"
TASK_DESCRIPTION = (
    "Identify a Class A warehouse property currently available for lease in Mecklenburg County, North Carolina, "
    "that has a minimum clear height of 32 feet, a truck court depth of at least 130 feet, an ESFR sprinkler system, "
    "a minimum building area of 95,000 square feet, and is located within 2 miles of Charlotte Douglas International Airport. "
    "Additionally, verify the commercial property tax rate per $100 of assessed value applicable to properties in Mecklenburg County for fiscal year 2026."
)


# --------------------------------------------------------------------------- #
# Data models for structured extraction                                       #
# --------------------------------------------------------------------------- #
class PropertyListing(BaseModel):
    property_name: Optional[str] = None
    listing_url: Optional[str] = None
    alt_urls: List[str] = Field(default_factory=list)

    # Specs as strings to maximize compatibility with various formats
    clear_height: Optional[str] = None
    truck_court_depth: Optional[str] = None
    sprinkler_type: Optional[str] = None
    building_area: Optional[str] = None

    # Location / availability
    address: Optional[str] = None
    city: Optional[str] = None
    county: Optional[str] = None
    state: Optional[str] = None
    airport_proximity: Optional[str] = None
    available_for_lease: Optional[str] = None
    class_type: Optional[str] = None  # e.g., "Class A"


class TaxRateInfo(BaseModel):
    fiscal_year: Optional[str] = None
    rate_string: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class WarehouseAnswerExtraction(BaseModel):
    property: Optional[PropertyListing] = None
    tax: Optional[TaxRateInfo] = None


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_property() -> str:
    return """
    Select exactly one warehouse property listing that the answer claims meets the specified requirements in Mecklenburg County, NC.
    Extract the following fields strictly from the answer text:

    property_name: The property's name or building/park name, if mentioned; else null.
    listing_url: The primary URL to the property's listing page that contains detailed specifications; must be a URL found in the answer; else null.
    alt_urls: Any additional URLs in the answer that contain relevant property specifications or location context (e.g., a brochure, map, or subpage). Return an array; empty if none.
    clear_height: The stated clear height figure (e.g., "32' clear", "36 feet clear"), if present in the answer; else null.
    truck_court_depth: The stated truck court depth (e.g., "130' truck court depth"), if present in the answer; else null.
    sprinkler_type: The sprinkler type, e.g., "ESFR", if present; else null.
    building_area: The total building area or minimum area (e.g., "100,000 SF"), if present; else null.
    address: Street address if present; else null.
    city: City name if present; else null.
    county: County name if present; else null.
    state: State if present; else null.
    airport_proximity: Any explicit distance statement to Charlotte Douglas International Airport (e.g., "1.8 miles to CLT") if present; else null.
    available_for_lease: The exact phrase indicating availability (e.g., "for lease", "available for lease") if present; else null.
    class_type: The building class (e.g., "Class A") if mentioned; else null.

    IMPORTANT:
    - Only extract information explicitly present in the answer text.
    - For all URL fields, return actual URLs that appear in the answer (plain or markdown). Do not invent URLs.
    - If a URL is missing a protocol, prepend "http://".
    """


def prompt_extract_tax() -> str:
    return """
    Extract the Mecklenburg County property tax rate information as described in the answer.

    Fields:
    fiscal_year: The fiscal year mentioned for the property tax rate (e.g., "FY 2026"); else null.
    rate_string: The rate string exactly as presented (e.g., "49.27 cents per $100", or "$0.4927 per $100"); else null.
    sources: All URLs provided in the answer that support or reference this tax rate; return an array (empty if none).

    IMPORTANT:
    - Only extract URLs explicitly present in the answer.
    - Do not invent any numbers or URLs.
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _valid_url(url: Optional[str]) -> bool:
    if not url:
        return False
    u = url.strip()
    return u.startswith("http://") or u.startswith("https://")


def _merge_sources(*url_lists: List[str]) -> List[str]:
    result: List[str] = []
    seen = set()
    for lst in url_lists:
        for u in lst:
            if not u:
                continue
            uu = u.strip()
            if uu and uu not in seen:
                seen.add(uu)
                result.append(uu)
    return result


def _property_sources(prop: Optional[PropertyListing]) -> List[str]:
    if not prop:
        return []
    base = []
    if prop.listing_url and prop.listing_url.strip():
        base.append(prop.listing_url.strip())
    return _merge_sources(base, prop.alt_urls or [])


# --------------------------------------------------------------------------- #
# Verification tree construction and checks                                   #
# --------------------------------------------------------------------------- #
async def build_and_verify_warehouse_requirements(
    evaluator: Evaluator,
    root_node,
    extraction: WarehouseAnswerExtraction,
) -> None:
    """
    Build the verification subtree according to the rubric and perform verifications.
    """
    # Create the main parallel node for all warehouse requirements
    wh_node = evaluator.add_parallel(
        id="Warehouse_Property_Requirements",
        desc="Verify that the identified warehouse property meets all specified requirements for a Class A distribution facility in Mecklenburg County, NC",
        parent=root_node,
        critical=False
    )

    prop: Optional[PropertyListing] = extraction.property
    tax: Optional[TaxRateInfo] = extraction.tax

    # Prepare sources
    prop_sources = _property_sources(prop)
    tax_sources = (tax.sources if tax else []) or []

    # 1) Reference URL Provided (existence check)
    url_ok = _valid_url(prop.listing_url) if prop else False
    url_node = evaluator.add_custom_node(
        result=url_ok,
        id="Reference_URL_Provided",
        desc="A valid property listing URL is provided that contains the property specifications",
        parent=wh_node,
        critical=True
    )

    # Helper: extra prerequisite list (ensure spec verifications skip if URL missing)
    prereqs = [url_node]

    # 2) Clear Height ≥ 32 ft
    node_clear = evaluator.add_leaf(
        id="Clear_Height_Specification",
        desc="The property has a minimum clear height of 32 feet",
        parent=wh_node,
        critical=True
    )
    await evaluator.verify(
        claim="The property listing indicates the clear height is at least 32 feet.",
        node=node_clear,
        sources=prop_sources,
        additional_instruction="Look for 'clear height' specifications such as 32', 32 ft, 36 ft, etc. Accept equivalent phrasing like 'minimum clear 32 feet'.",
        extra_prerequisites=prereqs
    )

    # 3) Truck Court Depth ≥ 130 ft
    node_truck = evaluator.add_leaf(
        id="Truck_Court_Depth",
        desc="The property has a truck court depth of at least 130 feet",
        parent=wh_node,
        critical=True
    )
    await evaluator.verify(
        claim="The property listing indicates a truck court depth of at least 130 feet.",
        node=node_truck,
        sources=prop_sources,
        additional_instruction="Check for 'truck court depth' numbers such as 130', 135', etc. Accept clear equivalents and typical industrial site spec phrasing.",
        extra_prerequisites=prereqs
    )

    # 4) ESFR Sprinkler System
    node_esfr = evaluator.add_leaf(
        id="ESFR_Sprinkler_System",
        desc="The property is equipped with an ESFR (Early Suppression Fast Response) sprinkler system",
        parent=wh_node,
        critical=True
    )
    await evaluator.verify(
        claim="The property listing indicates the building is equipped with ESFR sprinklers.",
        node=node_esfr,
        sources=prop_sources,
        additional_instruction="Look for 'ESFR' or 'Early Suppression Fast Response' in the building specifications.",
        extra_prerequisites=prereqs
    )

    # 5) Building Area ≥ 95,000 SF
    node_area = evaluator.add_leaf(
        id="Building_Size_Requirement",
        desc="The property has a minimum building area of 95,000 square feet",
        parent=wh_node,
        critical=True
    )
    await evaluator.verify(
        claim="The property listing indicates total building area is at least 95,000 square feet.",
        node=node_area,
        sources=prop_sources,
        additional_instruction="Check for total building area or minimum building size in square feet. Accept ranges or values clearly ≥ 95,000 SF.",
        extra_prerequisites=prereqs
    )

    # 6) Mecklenburg County Location
    node_county = evaluator.add_leaf(
        id="Mecklenburg_County_Location",
        desc="The property is located in Mecklenburg County, North Carolina",
        parent=wh_node,
        critical=True
    )
    await evaluator.verify(
        claim="The property listing indicates the building is located in Mecklenburg County, North Carolina.",
        node=node_county,
        sources=prop_sources,
        additional_instruction="Verify the address or stated location. If the page shows 'Charlotte, NC', it is within Mecklenburg County.",
        extra_prerequisites=prereqs
    )

    # 7) Airport Proximity ≤ 2 miles to CLT
    node_airport = evaluator.add_leaf(
        id="Airport_Proximity",
        desc="The property is located within 2 miles of Charlotte Douglas International Airport",
        parent=wh_node,
        critical=True
    )
    await evaluator.verify(
        claim="The property is within 2 miles of Charlotte Douglas International Airport (CLT).",
        node=node_airport,
        sources=prop_sources,
        additional_instruction="Look for explicit distance statements such as '1.8 miles to CLT'. If the listing provides a precise distance ≤ 2 miles, accept it.",
        extra_prerequisites=prereqs
    )

    # 8) Available for Lease
    node_available = evaluator.add_leaf(
        id="Available_for_Lease",
        desc="The property is currently available for lease",
        parent=wh_node,
        critical=True
    )
    await evaluator.verify(
        claim="The property listing indicates the property is currently available for lease.",
        node=node_available,
        sources=prop_sources,
        additional_instruction="Look for 'for lease', 'available', 'space available', 'now leasing', or similar explicit availability statements.",
        extra_prerequisites=prereqs
    )

    # 9) Property Tax Rate Verification (Mecklenburg County FY 2026)
    node_tax = evaluator.add_leaf(
        id="Property_Tax_Rate_Verification",
        desc="The commercial property tax rate for Mecklenburg County is correctly identified as 0.4927 cents per $100 of assessed value for FY 2026",
        parent=wh_node,
        critical=True
    )
    await evaluator.verify(
        claim="For FY 2026, the Mecklenburg County property tax rate is $0.4927 per $100 of assessed value (i.e., 49.27 cents per $100).",
        node=node_tax,
        sources=tax_sources,
        additional_instruction="Verify from official county sources (e.g., Mecklenburg County budget/tax pages) that FY 2026 rate equals $0.4927 per $100 (49.27 cents per $100). Ignore municipal add-ons; confirm county rate.",
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
) -> Dict[str, Any]:
    """
    Evaluate an agent's answer for the Mecklenburg County warehouse and FY 2026 tax rate task.
    """
    # Initialize evaluator and root
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

    # Extract structured information from the answer
    extraction = await evaluator.extract(
        prompt=prompt_extract_property(),
        template_class=WarehouseAnswerExtraction,
        extraction_name="property_extraction"
    )

    # Extract tax information (separate extraction call to improve robustness)
    tax_extraction = await evaluator.extract(
        prompt=prompt_extract_tax(),
        template_class=TaxRateInfo,
        extraction_name="tax_rate_extraction"
    )

    # Merge tax info back into main extraction object
    if not extraction.tax:
        extraction.tax = tax_extraction
    else:
        # If both exist, prefer non-empty fields
        extraction.tax.fiscal_year = extraction.tax.fiscal_year or tax_extraction.fiscal_year
        extraction.tax.rate_string = extraction.tax.rate_string or tax_extraction.rate_string
        if tax_extraction.sources:
            extraction.tax.sources = list(set((extraction.tax.sources or []) + tax_extraction.sources))

    # Add ground truth information (for transparency)
    evaluator.add_ground_truth({
        "expected_tax_rate_fy2026": "$0.4927 per $100 (i.e., 49.27 cents per $100)",
        "requirements": {
            "clear_height_min_ft": 32,
            "truck_court_depth_min_ft": 130,
            "sprinkler": "ESFR",
            "building_area_min_sf": 95000,
            "county": "Mecklenburg County, NC",
            "airport_proximity_max_miles": 2,
            "availability": "for lease"
        }
    }, gt_type="expected_requirements")

    # Custom info: source statistics
    prop_sources = _property_sources(extraction.property)
    evaluator.add_custom_info(
        info={"property_sources_count": len(prop_sources), "tax_sources_count": len((extraction.tax.sources if extraction.tax else []))},
        info_type="source_statistics"
    )

    # Build and verify the rubric tree
    await build_and_verify_warehouse_requirements(evaluator, root, extraction)

    # Return structured evaluation summary
    return evaluator.get_summary()