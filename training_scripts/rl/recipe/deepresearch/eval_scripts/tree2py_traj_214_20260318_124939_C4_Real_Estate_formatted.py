import asyncio
import logging
from typing import List, Optional, Dict, Any

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "atl_warehouse_spec_eval"
TASK_DESCRIPTION = """
I am looking for a large-scale warehouse facility in the Atlanta metropolitan area suitable for modern distribution operations. Find one industrial warehouse property currently available for lease that meets the following specifications:

- Property size: Minimum 100,000 square feet
- Clear height: Minimum 32 feet to accommodate high-density racking systems
- Loading infrastructure: Minimum 25 dock-high loading doors
- Column spacing: Minimum 50 feet to support efficient warehouse layout
- Parking: Minimum 100 auto parking spaces
- Fire suppression: ESFR (Early Suppression Fast Response) system
- Location: Within 15 miles of Hartsfield-Jackson Atlanta International Airport
- Environmental certification: LEED certified (Silver or higher)

Provide the following information:
- Complete property address
- Total building size (square feet)
- Clear height (feet)
- Number of dock-high loading doors
- Column spacing dimensions
- Number of auto parking spaces
- Fire suppression system type
- Distance to Hartsfield-Jackson Atlanta International Airport
- Environmental certification level
- Direct link to the property listing or information page
"""

# --------------------------------------------------------------------------- #
# Data Models                                                                 #
# --------------------------------------------------------------------------- #
class PropertyItem(BaseModel):
    """One property as presented in the answer."""
    name: Optional[str] = None
    address: Optional[str] = None
    total_size_sqft: Optional[str] = None
    clear_height_ft: Optional[str] = None
    dock_doors_count: Optional[str] = None
    column_spacing: Optional[str] = None
    auto_parking_spaces: Optional[str] = None
    fire_suppression: Optional[str] = None
    distance_to_atl_miles: Optional[str] = None
    environmental_cert: Optional[str] = None
    listing_url: Optional[str] = None
    supporting_urls: List[str] = Field(default_factory=list)
    # Optional helpful fields for a couple of checks
    property_type: Optional[str] = None
    availability_status: Optional[str] = None


class PropertyExtraction(BaseModel):
    """All properties mentioned in the answer (should be exactly one, per requirements)."""
    properties: List[PropertyItem] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction Prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_properties() -> str:
    return """
    Extract all specific property entries mentioned in the answer. Each entry corresponds to one distinct industrial/warehouse property.
    For each property, extract the following fields exactly as stated in the answer:
    - name: The property or park name (if provided)
    - address: The complete property address as written
    - total_size_sqft: The stated total building size (keep units or number as written, e.g., "1,000,000 SF" or "1,000,000")
    - clear_height_ft: The stated clear height, e.g., "36'", "36 ft", or "36"
    - dock_doors_count: The stated number of dock-high doors (e.g., "40 dock doors", "40")
    - column_spacing: The column spacing dimensions as written (e.g., "50' x 54'")
    - auto_parking_spaces: The number of auto/car parking spaces as written
    - fire_suppression: The fire suppression system type as written (e.g., "ESFR", "ESFR sprinklers")
    - distance_to_atl_miles: The distance to Hartsfield-Jackson Atlanta International Airport as stated in the answer. If included with units, keep the number (e.g., "12", "12.5"). If no distance is provided in the answer, set null.
    - environmental_cert: The environmental certification level as written (e.g., "LEED Silver", "LEED Gold", etc.)
    - listing_url: A direct URL to the property's official listing or information page as provided in the answer. If multiple are provided, use the most direct official listing link.
    - supporting_urls: Any additional URLs in the answer that specifically pertain to this property (exclude unrelated links).
    - property_type: If the answer explicitly labels the property type (e.g., "industrial", "warehouse", "distribution"), extract it; otherwise null.
    - availability_status: If the answer explicitly indicates "for lease" or similar availability status, extract that phrase; otherwise null.

    Important:
    - Extract only what is explicitly present in the answer; do not infer.
    - If a field is missing for a property, set it to null (or empty list for supporting_urls).
    - If the answer mentions multiple properties, extract each into the 'properties' array, in the same order they appear in the answer.

    Return a JSON object with a single key:
    {
      "properties": [ ... PropertyItem objects ... ]
    }
    """


# --------------------------------------------------------------------------- #
# Helper Functions                                                            #
# --------------------------------------------------------------------------- #
def _combine_sources(prop: PropertyItem) -> List[str]:
    urls = []
    if prop.listing_url and isinstance(prop.listing_url, str) and prop.listing_url.strip():
        urls.append(prop.listing_url.strip())
    # Add supporting urls (unique)
    for u in prop.supporting_urls or []:
        if isinstance(u, str) and u.strip():
            if u.strip() not in urls:
                urls.append(u.strip())
    return urls


def _non_empty(s: Optional[str]) -> bool:
    return isinstance(s, str) and bool(s.strip())


# --------------------------------------------------------------------------- #
# Verification Subtree Builders                                               #
# --------------------------------------------------------------------------- #
async def build_and_verify_tree_for_property(evaluator: Evaluator, root, prop: PropertyItem, prop_count: int) -> None:
    """
    Build the verification tree according to the rubric, and submit verification calls.
    The rubric requires each condition to be critical (no partial credit if any fails).
    """

    # 1) single_property_provided (custom check against extraction result)
    evaluator.add_custom_node(
        result=(prop_count == 1),
        id="single_property_provided",
        desc="Response identifies exactly one specific warehouse property (not zero and not multiple).",
        parent=root,
        critical=True
    )

    # 2) direct_listing_link_provided (existence only; other checks rely on the link)
    listing_present = _non_empty(prop.listing_url)
    evaluator.add_custom_node(
        result=listing_present,
        id="direct_listing_link_provided",
        desc="A direct link (URL) to the property listing or an official information page is provided.",
        parent=root,
        critical=True
    )

    # Prepare shared sources
    all_urls = _combine_sources(prop)

    # 3) industrial_and_available_for_lease (split into two critical leaves under a critical group)
    ind_avail_group = evaluator.add_parallel(
        id="industrial_and_available_for_lease",
        desc="Property is industrial/warehouse space and is currently available for lease (supported by the listing/info page).",
        parent=root,
        critical=True
    )
    # 3a) Industrial/Warehouse use supported
    ind_leaf = evaluator.add_leaf(
        id="industrial_use_supported",
        desc="Listing supports that the property is industrial/warehouse/distribution use.",
        parent=ind_avail_group,
        critical=True
    )
    await evaluator.verify(
        claim="The listing indicates that the property is an industrial/warehouse/distribution facility (e.g., 'industrial', 'warehouse', 'distribution center', 'logistics').",
        node=ind_leaf,
        sources=all_urls,
        additional_instruction="Accept synonyms like 'industrial', 'warehouse', 'distribution', 'logistics center', 'fulfillment center'. If the page is for offices/retail only, fail."
    )
    # 3b) Available for lease supported
    avail_leaf = evaluator.add_leaf(
        id="available_for_lease_supported",
        desc="Listing supports that the property/space is currently available for lease.",
        parent=ind_avail_group,
        critical=True
    )
    await evaluator.verify(
        claim="The listing shows the property/space is currently available for lease (not just for sale or fully leased).",
        node=avail_leaf,
        sources=all_urls,
        additional_instruction="Look for 'for lease', 'available', 'space available', 'availability'. If clearly 'sold', 'leased', or only 'for sale', then fail."
    )

    # 4) complete_property_address_provided (split into reported + supported)
    addr_group = evaluator.add_parallel(
        id="complete_property_address_provided",
        desc="Complete property address is provided.",
        parent=root,
        critical=True
    )
    addr_reported = evaluator.add_custom_node(
        result=_non_empty(prop.address),
        id="property_address_reported",
        desc="Property address is reported in the answer.",
        parent=addr_group,
        critical=True
    )
    addr_supported = evaluator.add_leaf(
        id="property_address_supported_by_listing",
        desc="Listing supports the stated property address.",
        parent=addr_group,
        critical=True
    )
    await evaluator.verify(
        claim=f"The property address is '{prop.address or ''}'.",
        node=addr_supported,
        sources=all_urls,
        additional_instruction="Match tolerance: allow abbreviations (Rd vs Road), directional suffix/prefix (SE vs Southeast), and suite/bldg numbers. The page must clearly show the same address."
    )

    # 5) location_within_15_miles_of_airport (split into distance stated + within-15 supported)
    loc_group = evaluator.add_parallel(
        id="location_within_15_miles_of_airport",
        desc="Response states the distance to Hartsfield-Jackson Atlanta International Airport and the stated/verifiable distance is within 15 miles.",
        parent=root,
        critical=True
    )
    dist_stated = evaluator.add_custom_node(
        result=_non_empty(prop.distance_to_atl_miles),
        id="distance_to_atl_stated",
        desc="Distance to Hartsfield-Jackson Atlanta International Airport is stated in the answer.",
        parent=loc_group,
        critical=True
    )
    within_15_leaf = evaluator.add_leaf(
        id="within_15_miles_supported",
        desc="The property is within 15 miles of Hartsfield-Jackson Atlanta International Airport (ATL).",
        parent=loc_group,
        critical=True
    )
    await evaluator.verify(
        claim="The property is within 15 miles of Hartsfield-Jackson Atlanta International Airport (ATL).",
        node=within_15_leaf,
        sources=all_urls,
        additional_instruction="Prefer explicit distance statements on the page (e.g., '12 miles to ATL'). If the page clearly indicates proximity to ATL within 15 miles, accept. If no evidence on the page, fail."
    )

    # 6) size_at_least_100k_and_reported
    size_group = evaluator.add_parallel(
        id="size_at_least_100k_and_reported",
        desc="Total building size is stated (in square feet) and is at least 100,000 SF.",
        parent=root,
        critical=True
    )
    size_reported = evaluator.add_custom_node(
        result=_non_empty(prop.total_size_sqft),
        id="total_size_reported",
        desc="Total building size is reported in the answer.",
        parent=size_group,
        critical=True
    )
    size_supported = evaluator.add_leaf(
        id="total_size_meets_100k_supported",
        desc="Listing supports that total building size is at least 100,000 SF.",
        parent=size_group,
        critical=True
    )
    await evaluator.verify(
        claim=f"The total building size is {prop.total_size_sqft or ''} and this is at least 100,000 square feet.",
        node=size_supported,
        sources=all_urls,
        additional_instruction="Check building or total area statements. If a range is given, the lower bound must be ≥ 100,000 SF. If multiple buildings, total should still meet/eat least 100,000 SF."
    )

    # 7) clear_height_at_least_32_and_reported
    ch_group = evaluator.add_parallel(
        id="clear_height_at_least_32_and_reported",
        desc="Clear height is stated (in feet) and is at least 32 ft.",
        parent=root,
        critical=True
    )
    ch_reported = evaluator.add_custom_node(
        result=_non_empty(prop.clear_height_ft),
        id="clear_height_reported",
        desc="Clear height is reported in the answer.",
        parent=ch_group,
        critical=True
    )
    ch_supported = evaluator.add_leaf(
        id="clear_height_meets_32_supported",
        desc="Listing supports that clear height is at least 32 ft.",
        parent=ch_group,
        critical=True
    )
    await evaluator.verify(
        claim=f"The property clear height is {prop.clear_height_ft or ''} and this is at least 32 feet.",
        node=ch_supported,
        sources=all_urls,
        additional_instruction="Accept typical notations like 32', 36', 40'. If multiple areas have different clear heights, the main warehouse must have ≥ 32'."
    )

    # 8) dock_doors_at_least_25_and_reported
    dock_group = evaluator.add_parallel(
        id="dock_doors_at_least_25_and_reported",
        desc="Number of dock-high loading doors is stated and is at least 25.",
        parent=root,
        critical=True
    )
    dock_reported = evaluator.add_custom_node(
        result=_non_empty(prop.dock_doors_count),
        id="dock_doors_reported",
        desc="Number of dock-high loading doors is reported in the answer.",
        parent=dock_group,
        critical=True
    )
    dock_supported = evaluator.add_leaf(
        id="dock_doors_meet_25_supported",
        desc="Listing supports that dock-high loading doors are at least 25.",
        parent=dock_group,
        critical=True
    )
    await evaluator.verify(
        claim=f"The property has {prop.dock_doors_count or ''} dock-high loading doors, which is at least 25.",
        node=dock_supported,
        sources=all_urls,
        additional_instruction="Allow synonyms like 'dock doors', 'dock-high doors', 'dock positions'. Exclude grade-level doors from the dock-high count unless explicitly counted as dock-high."
    )

    # 9) column_spacing_at_least_50_and_reported
    col_group = evaluator.add_parallel(
        id="column_spacing_at_least_50_and_reported",
        desc="Column spacing dimensions are stated and meet/exceed 50 ft in the relevant spacing measure.",
        parent=root,
        critical=True
    )
    col_reported = evaluator.add_custom_node(
        result=_non_empty(prop.column_spacing),
        id="column_spacing_reported",
        desc="Column spacing dimensions are reported in the answer.",
        parent=col_group,
        critical=True
    )
    col_supported = evaluator.add_leaf(
        id="column_spacing_meets_50_supported",
        desc="Listing supports that column spacing meets/exceeds 50 ft.",
        parent=col_group,
        critical=True
    )
    await evaluator.verify(
        claim=f"The column spacing is {prop.column_spacing or ''} and meets/exceeds 50 ft in the standard grid.",
        node=col_supported,
        sources=all_urls,
        additional_instruction="If spacing is expressed as AxB (e.g., 50' x 54'), treat the smaller dimension as the controlling spacing; it must be ≥ 50'. Accept typical grid descriptors."
    )

    # 10) auto_parking_at_least_100_and_reported
    park_group = evaluator.add_parallel(
        id="auto_parking_at_least_100_and_reported",
        desc="Number of auto parking spaces is stated and is at least 100.",
        parent=root,
        critical=True
    )
    park_reported = evaluator.add_custom_node(
        result=_non_empty(prop.auto_parking_spaces),
        id="auto_parking_reported",
        desc="Auto/car parking count is reported in the answer.",
        parent=park_group,
        critical=True
    )
    park_supported = evaluator.add_leaf(
        id="auto_parking_meets_100_supported",
        desc="Listing supports that auto parking spaces are at least 100.",
        parent=park_group,
        critical=True
    )
    await evaluator.verify(
        claim=f"The property has {prop.auto_parking_spaces or ''} auto parking spaces, which is at least 100.",
        node=park_supported,
        sources=all_urls,
        additional_instruction="Look for 'auto parking', 'car parking', 'vehicle parking'. Do not count trailer parking toward auto parking."
    )

    # 11) fire_suppression_esfr_and_reported
    fire_group = evaluator.add_parallel(
        id="fire_suppression_esfr_and_reported",
        desc="Fire suppression system type is stated and is ESFR (Early Suppression Fast Response).",
        parent=root,
        critical=True
    )
    fire_reported = evaluator.add_custom_node(
        result=_non_empty(prop.fire_suppression),
        id="fire_suppression_reported",
        desc="Fire suppression system type is reported in the answer.",
        parent=fire_group,
        critical=True
    )
    fire_supported = evaluator.add_leaf(
        id="fire_suppression_is_esfr_supported",
        desc="Listing supports that the fire suppression is ESFR.",
        parent=fire_group,
        critical=True
    )
    await evaluator.verify(
        claim=f"The fire suppression system is {prop.fire_suppression or ''} and specifically ESFR (Early Suppression Fast Response).",
        node=fire_supported,
        sources=all_urls,
        additional_instruction="Accept: 'ESFR', 'ESFR sprinklers', 'ESFR fire suppression'. If only 'wet' or 'dry' is mentioned without ESFR, fail."
    )

    # 12) leed_silver_or_higher_and_reported
    leed_group = evaluator.add_parallel(
        id="leed_silver_or_higher_and_reported",
        desc="Environmental certification level is stated and is LEED Silver or higher.",
        parent=root,
        critical=True
    )
    leed_reported = evaluator.add_custom_node(
        result=_non_empty(prop.environmental_cert),
        id="leed_reported",
        desc="Environmental certification level is reported in the answer.",
        parent=leed_group,
        critical=True
    )
    leed_supported = evaluator.add_leaf(
        id="leed_silver_or_higher_supported",
        desc="Listing supports that the environmental certification is LEED Silver or higher.",
        parent=leed_group,
        critical=True
    )
    await evaluator.verify(
        claim=f"The property has environmental certification '{prop.environmental_cert or ''}', which is LEED Silver or higher.",
        node=leed_supported,
        sources=all_urls,
        additional_instruction="Accept: LEED Silver/Gold/Platinum (any version, e.g., v4/v4.1, BD+C/CS). 'LEED Certified' alone is below Silver and should fail."
    )


# --------------------------------------------------------------------------- #
# Main Evaluation Function                                                    #
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
    """
    Evaluate an answer for the Atlanta warehouse specification task.
    Returns a structured evaluation summary dictionary.
    """
    # Initialize evaluator
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
    extracted = await evaluator.extract(
        prompt=prompt_extract_properties(),
        template_class=PropertyExtraction,
        extraction_name="extracted_properties"
    )

    # Choose the primary property (first if multiple, placeholder if none)
    prop_count = len(extracted.properties or [])
    selected_prop = extracted.properties[0] if prop_count > 0 else PropertyItem()

    # Record custom info for debugging
    evaluator.add_custom_info(
        info={
            "reported_property_count": prop_count,
            "selected_property_overview": {
                "name": selected_prop.name,
                "address": selected_prop.address,
                "listing_url": selected_prop.listing_url
            }
        },
        info_type="extraction_overview",
        info_name="selection_info"
    )

    # Build verification tree aligned to rubric and run verifications
    await build_and_verify_tree_for_property(evaluator, root, selected_prop, prop_count)

    # Return summary
    return evaluator.get_summary()