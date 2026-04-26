import asyncio
import logging
from typing import Any, List, Optional, Dict

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys

# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "columbus_leed_oz_office"
TASK_DESCRIPTION = (
    "A technology company with 200 employees is planning to relocate its corporate headquarters to Columbus, Ohio to "
    "take advantage of the Qualified Opportunity Zone program and demonstrate environmental commitment. Find 3 commercial "
    "office buildings in Columbus, Ohio that meet ALL of the following requirements:\n\n"
    "1. Located within one of Columbus's designated Qualified Opportunity Zones (as defined by the city's 39 zones in "
    "Eastside, Northeast, Northwest, Westside, or Southside areas)\n"
    "2. LEED certified at any level (Certified, Silver, Gold, or Platinum)\n"
    "3. Minimum 30,000 square feet of available leasable office space (to accommodate 150 square feet per employee)\n"
    "4. Currently available for lease as of January 2026\n"
    "5. Class A office building designation\n"
    "6. Adequate parking with minimum ratio of 5 spaces per 1,000 square feet\n"
    "7. Within 15 miles of John Glenn Columbus International Airport\n"
    "8. Located in or accessible to downtown Columbus for employee convenience\n\n"
    "For each building, provide:\n"
    "- Building name and complete street address\n"
    "- Confirmation of Opportunity Zone location (specific zone area: Eastside, Northeast, Northwest, Westside, or Southside)\n"
    "- LEED certification level\n"
    "- Total square footage available for lease\n"
    "- Building class designation\n"
    "- Number of parking spaces and parking ratio\n"
    "- Approximate distance from CMH airport\n"
    "- Current lease rate\n"
    "- Reference URL(s) for verification"
)

VALID_OZ_AREAS = ["Eastside", "Northeast", "Northwest", "Westside", "Southside"]

# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class BuildingEntry(BaseModel):
    name: Optional[str] = None
    address: Optional[str] = None
    opportunity_zone_area: Optional[str] = None  # One of Eastside/Northeast/Northwest/Westside/Southside
    leed_level: Optional[str] = None             # Certified/Silver/Gold/Platinum (or equivalent wording)
    available_sqft: Optional[str] = None         # Keep as string to allow ranges or combined suites
    available_as_of_jan_2026: Optional[str] = None  # e.g., "Available now", "Available Jan 2026", "Currently leasing"
    building_class: Optional[str] = None         # Expect "Class A" or similar
    parking_spaces: Optional[str] = None         # e.g., "600 spaces"
    parking_ratio: Optional[str] = None          # e.g., "5/1,000", "5 per 1,000 SF"
    airport_distance_miles: Optional[str] = None # e.g., "12 miles", "~10 mi"
    downtown_access_note: Optional[str] = None   # e.g., "Downtown location", "10 min to downtown"
    lease_rate: Optional[str] = None             # e.g., "$22/SF/yr", "$18-20 NNN"
    urls: List[str] = Field(default_factory=list)


class BuildingsExtraction(BaseModel):
    buildings: List[BuildingEntry] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_buildings() -> str:
    return (
        "Extract up to the first 3 commercial office buildings presented in the answer that are located in Columbus, Ohio, "
        "and for each building, extract the following fields:\n"
        "1) name: Building name (string)\n"
        "2) address: Complete street address including city and state (string)\n"
        "3) opportunity_zone_area: Which area the building is in among {Eastside, Northeast, Northwest, Westside, Southside} "
        "(string; if unspecified in answer, set null)\n"
        "4) leed_level: LEED certification level (Certified/Silver/Gold/Platinum), exact wording from the answer if present (string)\n"
        "5) available_sqft: Total leasable office square footage the answer claims is available (string; can be a range or sum)\n"
        "6) available_as_of_jan_2026: Statement regarding current availability as of January 2026 (string; e.g., 'available now', "
        "'currently leasing', 'available January 2026'; set null if not mentioned)\n"
        "7) building_class: Building class designation (expect 'Class A') (string)\n"
        "8) parking_spaces: Number of parking spaces provided in the answer (string; set null if missing)\n"
        "9) parking_ratio: Parking ratio stated (string; e.g., '5/1000' or '5 spaces per 1,000 SF'; set null if missing)\n"
        "10) airport_distance_miles: Approximate distance to John Glenn Columbus International Airport (CMH) as a string "
        "(e.g., '12 miles', '~10 mi'; set null if not mentioned)\n"
        "11) downtown_access_note: Statement indicating the building is in downtown Columbus or offers objective evidence of "
        "downtown accessibility (string; e.g., 'downtown location', '8 minutes to downtown', 'direct transit to downtown'; set null if missing)\n"
        "12) lease_rate: Current lease rate (string; e.g., '$22/SF/yr', '$18-20 NNN'; set null if not mentioned)\n"
        "13) urls: A list of all reference URL(s) that the answer associates with this building. Extract only actual URLs "
        "explicitly present in the answer (can include listing pages, building sites, Google Maps links, city OZ map links, "
        "LEED directories, etc.).\n\n"
        "Return a JSON object: {\"buildings\": [ ... up to 3 entries ... ]}. "
        "If more than 3 buildings are present, include only the first 3. If fewer than 3 are present, include what is available. "
        "If any field is missing for a building, set it to null, and if no URLs are provided for a building, return an empty list for 'urls'."
    )


# --------------------------------------------------------------------------- #
# Verification helpers                                                        #
# --------------------------------------------------------------------------- #
def ordinal(n: int) -> str:
    return ["1st", "2nd", "3rd", "4th", "5th"][n - 1] if 1 <= n <= 5 else f"{n}th"


# --------------------------------------------------------------------------- #
# Per-building verification                                                   #
# --------------------------------------------------------------------------- #
async def verify_building(
    evaluator: Evaluator,
    parent_node,
    building: BuildingEntry,
    index: int
) -> None:
    """
    Build verification subtree for a single building.
    Each requirement is a critical leaf check; failing any will fail the building node.
    """
    order = ordinal(index + 1)
    bnode = evaluator.add_parallel(
        id=f"building_{index + 1}",
        desc=f"{order} building (score independently for partial credit).",
        parent=parent_node,
        critical=False  # Non-critical at building level to allow partial credit across buildings
    )

    # Collect sources for this building (gates many verifications)
    sources = building.urls if building and building.urls else []

    # Reference URLs provided (critical gate)
    evaluator.add_custom_node(
        result=bool(sources),
        id=f"building_{index + 1}_reference_urls_provided",
        desc="Provides reference URL(s) supporting the above claims.",
        parent=bnode,
        critical=True
    )

    # Building name provided (critical)
    evaluator.add_custom_node(
        result=bool(building and building.name and building.name.strip()),
        id=f"building_{index + 1}_building_name_provided",
        desc="Building name is provided.",
        parent=bnode,
        critical=True
    )

    # Complete street address provided and is in Columbus, Ohio (critical)
    addr_leaf = evaluator.add_leaf(
        id=f"building_{index + 1}_complete_street_address_provided",
        desc="Complete street address is provided and is in Columbus, Ohio.",
        parent=bnode,
        critical=True
    )
    address_str = building.address or ""
    await evaluator.verify(
        claim=f"The building's complete street address '{address_str}' is in Columbus, Ohio.",
        node=addr_leaf,
        sources=sources,
        additional_instruction="Confirm that the address text and/or page evidence clearly places the property in Columbus, OH. "
                               "Accept reasonable variations in formatting. If the address isn't on the page, the claim is not supported."
    )

    # Opportunity Zone confirmed with area (critical)
    oz_leaf = evaluator.add_leaf(
        id=f"building_{index + 1}_opportunity_zone_confirmed_with_area",
        desc="Confirms the building is within a designated Columbus Qualified Opportunity Zone and specifies the zone area (Eastside, Northeast, Northwest, Westside, or Southside).",
        parent=bnode,
        critical=True
    )
    oz_area = building.opportunity_zone_area or ""
    await evaluator.verify(
        claim=(
            f"This building is located within a Columbus Qualified Opportunity Zone in the '{oz_area}' area "
            f"(one of Eastside, Northeast, Northwest, Westside, or Southside)."
        ),
        node=oz_leaf,
        sources=sources,
        additional_instruction="Verify explicit Opportunity Zone evidence on the provided page(s). "
                               "Evidence could include OZ maps, city documents, or the listing page stating OZ status. "
                               "Allow synonyms (e.g., 'East Side' for Eastside). If OZ cannot be confirmed, fail."
    )

    # LEED certification level provided (critical)
    leed_leaf = evaluator.add_leaf(
        id=f"building_{index + 1}_leed_certification_level_provided",
        desc="Building is LEED certified and the certification level is stated (Certified/Silver/Gold/Platinum).",
        parent=bnode,
        critical=True
    )
    leed_level = building.leed_level or ""
    await evaluator.verify(
        claim=f"The building is LEED certified at the '{leed_level}' level (Certified/Silver/Gold/Platinum).",
        node=leed_leaf,
        sources=sources,
        additional_instruction="Confirm LEED certification and level via the provided source(s). "
                               "Accept equivalent phrasing; reject 'LEED-targeted' or 'seeking' unless certification is explicitly stated."
    )

    # Available leasable space meets minimum (critical)
    sqft_leaf = evaluator.add_leaf(
        id=f"building_{index + 1}_available_leasable_space_meets_minimum",
        desc="Available leasable office square footage is stated and is at least 30,000 sq ft.",
        parent=bnode,
        critical=True
    )
    sqft_text = building.available_sqft or ""
    await evaluator.verify(
        claim="The building has at least 30,000 square feet of available leasable office space.",
        node=sqft_leaf,
        sources=sources,
        additional_instruction="Check the listing details for available SF; ranges or combined suites summing ≥30,000 SF are acceptable. "
                               "If the page lacks clear evidence of ≥30,000 SF availability, fail."
    )

    # Available for lease as of January 2026 (critical)
    avail_leaf = evaluator.add_leaf(
        id=f"building_{index + 1}_available_for_lease_january_2026",
        desc="States the building is currently available for lease as of January 2026.",
        parent=bnode,
        critical=True
    )
    avail_text = building.available_as_of_jan_2026 or ""
    await evaluator.verify(
        claim="The building is currently available for lease as of January 2026.",
        node=avail_leaf,
        sources=sources,
        additional_instruction="Verify the page indicates current availability (e.g., 'available now', active leasing, available date ≤ Jan 2026, "
                               "or clearly contemporaneous leasing status). If timing cannot be reasonably inferred, fail."
    )

    # Class A designation (critical)
    class_leaf = evaluator.add_leaf(
        id=f"building_{index + 1}_class_a_designation",
        desc="Building is designated as Class A office space.",
        parent=bnode,
        critical=True
    )
    bclass_text = building.building_class or ""
    await evaluator.verify(
        claim="This property is designated as a Class A office building.",
        node=class_leaf,
        sources=sources,
        additional_instruction="Look for 'Class A' designation explicitly on the page(s). If only implied or absent, fail."
    )

    # Parking meets ratio and is reported (critical)
    parking_leaf = evaluator.add_leaf(
        id=f"building_{index + 1}_parking_meets_ratio_and_reported",
        desc="Provides parking spaces count and a parking ratio; ratio is at least 5 spaces per 1,000 sq ft.",
        parent=bnode,
        critical=True
    )
    spaces_txt = building.parking_spaces or ""
    ratio_txt = building.parking_ratio or ""
    await evaluator.verify(
        claim=(
            "The building provides parking with a ratio of at least 5 spaces per 1,000 square feet, and the parking spaces count "
            f"({spaces_txt}) and ratio ({ratio_txt}) are reported."
        ),
        node=parking_leaf,
        sources=sources,
        additional_instruction="Confirm either an explicit ratio ≥5/1,000 or sufficient evidence implying this ratio. "
                               "If ratio is below 5/1,000 or not reported clearly, fail."
    )

    # Airport distance within 15 miles and reported (critical)
    airport_leaf = evaluator.add_leaf(
        id=f"building_{index + 1}_airport_distance_reported_and_within_15_miles",
        desc="Provides approximate distance to John Glenn Columbus International Airport and the distance is within 15 miles.",
        parent=bnode,
        critical=True
    )
    dist_txt = building.airport_distance_miles or ""
    await evaluator.verify(
        claim="The building is within 15 miles of John Glenn Columbus International Airport (CMH).",
        node=airport_leaf,
        sources=sources,
        additional_instruction="Use page evidence such as explicit distance to CMH, a Google Maps route link, or location context to confirm ≤15 miles. "
                               "If the provided evidence does not substantiate ≤15 miles, fail."
    )

    # Downtown accessibility evidence (critical)
    downtown_leaf = evaluator.add_leaf(
        id=f"building_{index + 1}_downtown_accessibility_evidence",
        desc="States the building is in downtown Columbus OR provides an objective indicator of downtown accessibility (e.g., distance to downtown, typical drive/transit time, or direct transit connectivity).",
        parent=bnode,
        critical=True
    )
    downtown_note = building.downtown_access_note or ""
    await evaluator.verify(
        claim="The property is in downtown Columbus or has objective evidence of direct and convenient accessibility to downtown.",
        node=downtown_leaf,
        sources=sources,
        additional_instruction="Accept explicit 'downtown Columbus' location or objective indicators such as distance/time to downtown or direct transit links. "
                               "If there is no clear evidence of downtown location/accessibility, fail."
    )

    # Lease rate provided (critical)
    evaluator.add_custom_node(
        result=bool(building and building.lease_rate and building.lease_rate.strip()),
        id=f"building_{index + 1}_lease_rate_provided",
        desc="Current lease rate is provided.",
        parent=bnode,
        critical=True
    )


# --------------------------------------------------------------------------- #
# Main evaluation function                                                    #
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
    Evaluate an answer for the Columbus LEED/OZ office building task.
    """
    # Initialize evaluator (root is non-critical to allow partial credit across buildings)
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Buildings are evaluated independently
        agent_name=agent_name,
        answer_name=answer_name,
        client=client,
        task_description="Provide 3 commercial office buildings in Columbus, Ohio. Each building must meet all mandatory requirements and include all requested per-building details with reference URL(s).",
        answer=answer,
        global_cache=cache,
        global_semaphore=semaphore,
        logger=logger,
        default_model=model
    )

    # Extract building entries
    extracted = await evaluator.extract(
        prompt=prompt_extract_buildings(),
        template_class=BuildingsExtraction,
        extraction_name="buildings_extraction"
    )

    # Ensure we have exactly 3 buildings in list (pad with empty entries if needed; trim if too many)
    buildings: List[BuildingEntry] = list(extracted.buildings[:3])
    while len(buildings) < 3:
        buildings.append(BuildingEntry())

    # Add custom info: valid OZ areas
    evaluator.add_custom_info({"valid_oz_areas": VALID_OZ_AREAS}, info_type="meta", info_name="oz_area_options")

    # Build verification subtrees for each building
    for i in range(3):
        await verify_building(evaluator, root, buildings[i], i)

    # Return evaluation summary
    return evaluator.get_summary()