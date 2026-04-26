import asyncio
import logging
from typing import List, Optional, Dict, Any

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "dallas_uptown_classA_selection"
TASK_DESCRIPTION = (
    "I am a corporate real estate advisor helping a client identify premium office space options in Dallas, Texas. "
    "The client requires a comprehensive analysis of Class A office buildings in the Uptown district that meet specific criteria for their expanding operations.\n\n"
    "Find three Class A office buildings located in the Uptown district of Dallas, Texas that satisfy ALL of the following requirements:\n\n"
    "1. Building Classification: Must be designated as Class A office space\n"
    "2. Sustainability Certification: Must have LEED Gold or Platinum certification\n"
    "3. Building Scale: Must be at least 15 stories tall\n"
    "4. Floor Plate Size: Must have floor plates of at least 20,000 square feet to accommodate larger corporate tenant needs\n"
    "5. Parking: Must provide at least 4 parking spaces per 1,000 square feet of rentable area\n"
    "6. Fitness Amenity: Must have an on-site fitness center or gym facility\n"
    "7. Dining Facilities: Must have on-site dining options (restaurant, café, or food service facility)\n"
    "8. Conference Facilities: Must have conference center or meeting room facilities available to tenants\n"
    "9. Infrastructure: Must have fiber optic internet connectivity infrastructure\n"
    "10. Accessibility: Must be compliant with Americans with Disabilities Act (ADA) requirements\n"
    "11. Availability: Must have office space available for lease in 2026\n\n"
    "For each building, provide:\n"
    "- The building name and complete address\n"
    "- Verification of how it meets each of the 11 requirements listed above\n"
    "- A reference URL to the building's official property page, leasing information, or authoritative real estate listing"
)


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class BuildingItem(BaseModel):
    name: Optional[str] = None
    address: Optional[str] = None
    reference_urls: List[str] = Field(default_factory=list)


class BuildingsExtraction(BaseModel):
    buildings: List[BuildingItem] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_buildings() -> str:
    return (
        "Extract up to five buildings the answer proposes that meet the client's requirements for Class A office space "
        "in the Uptown district of Dallas. For each building, extract:\n"
        "1) name: the building's official or commonly used name as stated in the answer\n"
        "2) address: the complete street address as stated in the answer (include city and state if provided)\n"
        "3) reference_urls: all URLs cited in the answer that correspond to the building's official property page, "
        "   leasing information, or authoritative real estate listings (e.g., LoopNet, CoStar, landlord site). "
        "   Only extract URLs that are explicitly present in the answer text. If a URL is missing a protocol, prepend http://.\n\n"
        "Return a JSON object with a 'buildings' array. If any field is missing for a building, set it to null (for strings) "
        "or an empty array (for URLs). Do not hallucinate or infer URLs."
    )


# --------------------------------------------------------------------------- #
# Verification helpers                                                        #
# --------------------------------------------------------------------------- #
def _null_to_empty(value: Optional[str]) -> str:
    return value or ""


def _urls_or_none(urls: Optional[List[str]]) -> List[str]:
    if not urls:
        return []
    # Deduplicate while preserving order
    seen = set()
    cleaned: List[str] = []
    for u in urls:
        if not u:
            continue
        if u not in seen:
            seen.add(u)
            cleaned.append(u)
    return cleaned


# --------------------------------------------------------------------------- #
# Building verification                                                       #
# --------------------------------------------------------------------------- #
async def verify_building(
    evaluator: Evaluator,
    parent_node,
    building: BuildingItem,
    index: int,
) -> None:
    """
    Build the verification subtree and run checks for one building.
    All requirement leaves are verified against the provided reference URLs.
    """

    bidx = index + 1
    bname = _null_to_empty(building.name)
    baddr = _null_to_empty(building.address)
    urls = _urls_or_none(building.reference_urls)

    # Create a container node for this building (parallel aggregation)
    bnode = evaluator.add_parallel(
        id=f"building_{bidx}",
        desc=f"Building #{bidx} verification (Dallas Uptown, Class A, all criteria)",
        parent=parent_node,
        critical=False,
    )

    # Critical gating: ensure at least one reference URL is provided
    evaluator.add_custom_node(
        result=len(urls) > 0,
        id=f"building_{bidx}_reference_url",
        desc="Provide a valid reference URL for the building's official property page or listing",
        parent=bnode,
        critical=True
    )

    # Prepare all leaves
    # 1) Name
    name_node = evaluator.add_leaf(
        id=f"building_{bidx}_name",
        desc="Provide the building name",
        parent=bnode,
        critical=True
    )
    name_claim = (
        f"The official or commonly used name of this property matches '{bname}'. "
        "Minor variations in branding, articles (e.g., 'The'), or punctuation should still count as a match."
        " If the extracted name is blank or missing, this should be considered Incorrect."
    )

    # 2) Address
    addr_node = evaluator.add_leaf(
        id=f"building_{bidx}_address",
        desc="Provide the complete address of the building",
        parent=bnode,
        critical=True
    )
    addr_claim = (
        f"The property's street address matches '{baddr}' (allowing common abbreviations like St./Street, "
        "Ave./Avenue, and formatting differences). The city must be Dallas and the state Texas (TX). "
        "If the extracted address is blank or missing, this should be considered Incorrect."
    )

    # 3) Location in Uptown Dallas
    loc_node = evaluator.add_leaf(
        id=f"building_{bidx}_location",
        desc="Building must be located in the Uptown district of Dallas, Texas",
        parent=bnode,
        critical=True
    )
    loc_claim = (
        "This property is located in the Uptown district of Dallas, Texas. "
        "Accept phrasings like 'Uptown Dallas', 'Uptown submarket', or 'Uptown neighborhood'. "
        "Do not accept Downtown, Victory Park, Turtle Creek, or other districts unless Uptown is explicitly confirmed."
    )

    # 4) Class A designation
    classa_node = evaluator.add_leaf(
        id=f"building_{bidx}_class_a",
        desc="Building must be designated as Class A office space",
        parent=bnode,
        critical=True
    )
    classa_claim = (
        "This building is designated as Class A (or A+) office space. "
        "Accept terms like 'Class A' or 'trophy' (which implies Class A or higher). "
        "Do not accept Class B or Class C."
    )

    # 5) LEED Gold or Platinum
    leed_node = evaluator.add_leaf(
        id=f"building_{bidx}_leed",
        desc="Building must have LEED Gold or Platinum certification",
        parent=bnode,
        critical=True
    )
    leed_claim = (
        "This property has a LEED Gold or LEED Platinum certification (any LEED version acceptable, e.g., v4, EBOM, CS). "
        "Do NOT accept LEED Certified (basic) or LEED Silver as satisfying the requirement."
    )

    # 6) Height: at least 15 stories
    height_node = evaluator.add_leaf(
        id=f"building_{bidx}_height",
        desc="Building must be at least 15 stories tall",
        parent=bnode,
        critical=True
    )
    height_claim = (
        "This property has at least 15 stories or floors (e.g., '15 stories', 'floors: 18'). "
        "If the number of stories is below 15, the claim is not supported."
    )

    # 7) Floor plate size >= 20,000 SF
    floorplate_node = evaluator.add_leaf(
        id=f"building_{bidx}_floor_plate",
        desc="Building must have floor plates of at least 20,000 square feet",
        parent=bnode,
        critical=True
    )
    floorplate_claim = (
        "Typical or average office floor plate size at this property is at least 20,000 square feet. "
        "Accept mentions like 'typical floor size 25,000 RSF', 'floor plate ~22,000 SF', or similar. "
        "If multiple floor sizes are listed, accept if typical/representative floors meet or exceed 20,000 SF."
    )

    # 8) Parking ratio >= 4/1,000 RSF
    parking_node = evaluator.add_leaf(
        id=f"building_{bidx}_parking",
        desc="Building must provide at least 4 parking spaces per 1,000 square feet of rentable area",
        parent=bnode,
        critical=True
    )
    parking_claim = (
        "The parking ratio is at least 4 spaces per 1,000 square feet (≥ 4/1,000 RSF). "
        "Accept formats like '4 per 1,000', '4.0/1,000', '4/KSF'. If the ratio is below 4/1,000, do not support."
    )

    # 9) On-site fitness amenity
    fitness_node = evaluator.add_leaf(
        id=f"building_{bidx}_fitness",
        desc="Building must have an on-site fitness center or gym facility",
        parent=bnode,
        critical=True
    )
    fitness_claim = (
        "This property offers an on-site fitness center or gym (within the building or part of the property amenities). "
        "Nearby third-party gyms without an explicit on-site amenity should not be accepted."
    )

    # 10) On-site dining facilities
    dining_node = evaluator.add_leaf(
        id=f"building_{bidx}_dining",
        desc="Building must have on-site dining options such as a restaurant, café, or food service facility",
        parent=bnode,
        critical=True
    )
    dining_claim = (
        "This property provides on-site dining options (e.g., café, restaurant, food service) within the building or property. "
        "Nearby off-site dining without on-site availability should not be accepted."
    )

    # 11) Conference facilities
    conf_node = evaluator.add_leaf(
        id=f"building_{bidx}_conference",
        desc="Building must have conference center or meeting room facilities",
        parent=bnode,
        critical=True
    )
    conference_claim = (
        "This property offers a tenant-available conference center or shared meeting room facilities on-site."
    )

    # 12) Fiber optic connectivity
    fiber_node = evaluator.add_leaf(
        id=f"building_{bidx}_fiber",
        desc="Building must have fiber optic internet connectivity infrastructure",
        parent=bnode,
        critical=True
    )
    fiber_claim = (
        "This property has fiber optic internet connectivity infrastructure (e.g., 'fiber connectivity', 'fiber-ready', "
        "'lit building with fiber providers')."
    )

    # 13) ADA compliance
    ada_node = evaluator.add_leaf(
        id=f"building_{bidx}_ada",
        desc="Building must be compliant with Americans with Disabilities Act (ADA) accessibility requirements",
        parent=bnode,
        critical=True
    )
    ada_claim = (
        "This property is compliant with Americans with Disabilities Act (ADA) accessibility requirements or otherwise "
        "explicitly indicates accessibility compliance. Accept 'ADA compliant', 'accessible building', or equivalent phrasing."
    )

    # 14) Availability in 2026
    avail_node = evaluator.add_leaf(
        id=f"building_{bidx}_availability",
        desc="Building must have office space available for lease in 2026",
        parent=bnode,
        critical=True
    )
    availability_claim = (
        "There is office space available for lease in 2026 at this property. "
        "Support if the listing or official materials explicitly reference 2026 availability, delivery in 2026, "
        "or suites specifically available in 2026. Do not accept generic 'available now' without a 2026 reference."
    )

    # Batch verify all factual claims (all use the same building URLs)
    claims_and_sources = [
        (name_claim, urls, name_node, "Verify the building name shown on the page matches the extracted name. Allow minor variations and abbreviations."),
        (addr_claim, urls, addr_node, "Check the address text on the page. Be lenient to abbreviations, but city must be Dallas, state TX."),
        (loc_claim, urls, loc_node, "Look for 'Uptown Dallas' or synonymous phrasing on the page."),
        (classa_claim, urls, classa_node, "Look for market classification; accept 'Class A' or 'trophy' as fulfilling Class A."),
        (leed_claim, urls, leed_node, "Only Gold or Platinum should pass. Silver or 'Certified' do not satisfy."),
        (height_claim, urls, height_node, "Confirm '15+' floors/stories. Any count >=15 passes."),
        (floorplate_claim, urls, floorplate_node, "Look for 'typical floor size' or 'floor plate' info; >=20,000 SF passes."),
        (parking_claim, urls, parking_node, "Look for parking ratio string. Accept >=4/1,000 RSF."),
        (fitness_claim, urls, fitness_node, "Confirm an on-site fitness center/gym amenity, not just nearby."),
        (dining_claim, urls, dining_node, "Confirm on-site dining/café/food service in the building/property."),
        (conference_claim, urls, conf_node, "Confirm shared conference center/meeting rooms available to tenants."),
        (fiber_claim, urls, fiber_node, "Confirm 'fiber' connectivity is available/installed."),
        (ada_claim, urls, ada_node, "Confirm ADA or equivalent accessibility compliance language."),
        (availability_claim, urls, avail_node, "Look for explicit 2026 availability language (suites, delivery, or leasing timelines)."),
    ]

    await evaluator.batch_verify(claims_and_sources)


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
    Evaluate an answer for three Class A office buildings in Dallas Uptown meeting strict criteria.
    """
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Buildings evaluated independently
        agent_name=agent_name,
        answer_name=answer_name,
        client=client,
        task_description=TASK_DESCRIPTION,
        answer=answer,
        global_cache=cache,
        global_semaphore=semaphore,
        logger=logger,
        default_model=model,
    )

    # Extract proposed buildings from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_buildings(),
        template_class=BuildingsExtraction,
        extraction_name="proposed_buildings"
    )

    buildings = list(extracted.buildings)[:3]
    while len(buildings) < 3:
        buildings.append(BuildingItem())

    # Create top-level nodes for three buildings
    bnodes = []
    for i in range(3):
        # Parent nodes for each building (as per rubric, parallel under root)
        bnode = evaluator.add_parallel(
            id=f"Building_{i+1}",
            desc=f"{['First','Second','Third'][i] if i < 3 else f'#{i+1}th'} qualifying Class A office building in Dallas Uptown",
            parent=root,
            critical=False
        )
        bnodes.append(bnode)

    # Verify each building (in parallel)
    await asyncio.gather(
        *[verify_building(evaluator, bnodes[i], buildings[i], i) for i in range(3)]
    )

    # Optional: record the constraints as a reference in output
    evaluator.add_ground_truth({
        "constraints": {
            "district": "Uptown, Dallas, Texas",
            "class": "Class A",
            "leed": "Gold or Platinum",
            "min_stories": 15,
            "min_floor_plate_sf": 20000,
            "min_parking_ratio_per_1000_rsf": 4,
            "amenities": ["fitness", "dining", "conference"],
            "infrastructure": ["fiber optic connectivity"],
            "accessibility": "ADA compliant",
            "availability_year": 2026
        },
        "expected_buildings": 3
    }, gt_type="requirements")

    return evaluator.get_summary()