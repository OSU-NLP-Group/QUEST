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
TASK_ID = "class_a_warehouse_south_metro_2024plus"
TASK_DESCRIPTION = (
    "Identify a Class A industrial warehouse facility that was completed or began construction in 2024 or later in one "
    "of these three major southern U.S. metropolitan areas: Atlanta, Dallas-Fort Worth, or Tampa Bay. The facility must "
    "meet ALL of the following specifications:\n\n"
    "- Minimum building size of 500,000 square feet\n"
    "- Minimum clear height of 32 feet\n"
    "- At least 50 dock-high loading doors\n"
    "- Column spacing of at least 50 feet\n"
    "- Truck court depth of at least 130 feet\n"
    "- Cross-dock configuration\n"
    "- Located within 10 miles of a major interstate highway\n"
    "- ESFR (Early Suppression Fast Response) sprinkler system\n"
    "- Minimum 6-inch concrete floor slab\n"
    "- LED lighting system\n"
    "- Minimum building depth of 250 feet\n"
    "- At least 100 trailer parking spaces\n"
    "- Office space included in the facility\n"
    "- Speculative or build-to-suit development\n\n"
    "Provide the facility name, complete address, total square footage, and the name of the developer."
)


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class FacilitySpecs(BaseModel):
    # Required response fields
    facility_name: Optional[str] = None
    address: Optional[str] = None
    total_square_footage: Optional[str] = None
    developer: Optional[str] = None

    # Location & classification
    metro_area: Optional[str] = None  # Prefer one of: "Atlanta", "Dallas-Fort Worth", "Tampa Bay"
    city: Optional[str] = None
    state: Optional[str] = None
    classification: Optional[str] = None  # e.g., "Class A industrial", "Class A warehouse"

    # Timeline
    timeline_statement: Optional[str] = None  # e.g., "construction began in 2024", "delivers 2025"
    completion_year: Optional[str] = None
    construction_start_year: Optional[str] = None

    # Building/shell specs
    building_size: Optional[str] = None  # another representation of size if provided
    clear_height_ft: Optional[str] = None
    dock_doors_count: Optional[str] = None
    column_spacing_ft: Optional[str] = None
    truck_court_depth_ft: Optional[str] = None
    cross_dock: Optional[str] = None  # "cross-dock", "yes", or description
    interstate_access_desc: Optional[str] = None  # e.g., "adjacent to I-75", "2 miles to I-20"
    esfr_sprinklers: Optional[str] = None
    floor_slab_thickness_in: Optional[str] = None
    led_lighting: Optional[str] = None
    building_depth_ft: Optional[str] = None
    trailer_parking_spaces: Optional[str] = None
    office_space: Optional[str] = None  # e.g., "office to suit", "includes office"
    development_type: Optional[str] = None  # "speculative", "build-to-suit", "spec", "BTS"

    # All URLs cited in the answer (sources)
    source_urls: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_facility() -> str:
    return """
    Extract details for a single industrial warehouse facility that the answer proposes as meeting the specifications.
    If multiple facilities are mentioned, choose the first one that is presented as meeting the constraints and treat it as the primary facility.
    Return all fields as strings when possible; do not coerce into numbers. If a field is not present, return null.

    Required response fields to extract:
    - facility_name
    - address
    - total_square_footage
    - developer

    Location & classification:
    - metro_area: If the answer explicitly indicates Atlanta, Dallas-Fort Worth (or DFW), or Tampa Bay, return that exact short name ("Atlanta", "Dallas-Fort Worth", or "Tampa Bay"); otherwise null.
    - city
    - state
    - classification: e.g., "Class A industrial", "Class A warehouse", "Class A logistics"

    Timeline (2024+ requirement):
    - timeline_statement: any phrase like "delivers in 2025", "construction began in 2024", "completed 2024"
    - completion_year: just the year if mentioned (e.g., "2025")
    - construction_start_year: just the year if mentioned (e.g., "2024")

    Building/shell specifications (strings; copy exactly as presented if possible):
    - building_size: e.g., "1,020,000 SF", "1.1 MSF"
    - clear_height_ft: e.g., "36'", "40 ft"
    - dock_doors_count: e.g., "120", "110+"
    - column_spacing_ft: e.g., "50' x 56' typical", "52' x 56'"
    - truck_court_depth_ft: e.g., "130'", "180-foot truck courts"
    - cross_dock: e.g., "cross-dock", "two-sided loading", "cross dock facility"
    - interstate_access_desc: e.g., "adjacent to I-75", "3 miles to I-20"
    - esfr_sprinklers: e.g., "ESFR sprinklers", "ESFR"
    - floor_slab_thickness_in: e.g., "7-inch slab", "6\" slab"
    - led_lighting: e.g., "LED lighting"
    - building_depth_ft: e.g., "building depth 320'"
    - trailer_parking_spaces: e.g., "200 trailer stalls", "100+ trailer parking"
    - office_space: e.g., "office to suit", "includes office"
    - development_type: e.g., "speculative", "spec", "build-to-suit", "BTS"

    Sources:
    - source_urls: Extract all URLs present in the answer that are relevant to this facility (developer page, property brochure, listing, news article, etc.). These may be plain URLs or markdown links. Return as an array of strings.

    Keep values verbatim from the answer where possible. If any field is missing from the answer, return null for that field.
    """


# --------------------------------------------------------------------------- #
# Verification helpers                                                        #
# --------------------------------------------------------------------------- #
def _has_text(value: Optional[str]) -> bool:
    return bool(value and isinstance(value, str) and value.strip() != "")


def _sources_or_none(urls: List[str]) -> Optional[List[str]]:
    return urls if (isinstance(urls, list) and len(urls) > 0) else None


# --------------------------------------------------------------------------- #
# Build verification tree and run checks                                      #
# --------------------------------------------------------------------------- #
async def verify_facility(evaluator: Evaluator, root, specs: FacilitySpecs) -> None:
    # Prepare sources (multi-URL if available)
    sources = _sources_or_none(specs.source_urls)

    # Top-level critical group
    main_node = evaluator.add_parallel(
        id="Facility_Identification_and_Specifications",
        desc="Evaluate whether the response identifies one qualifying facility and provides the required fields, and whether that facility meets all stated constraints.",
        parent=root,
        critical=True,
    )

    # Required fields (critical)
    required_fields_node = evaluator.add_parallel(
        id="Required_Response_Fields",
        desc="The response must provide all requested identifying details for the facility.",
        parent=main_node,
        critical=True,
    )

    evaluator.add_custom_node(
        result=_has_text(specs.facility_name),
        id="Facility_Name_Provided",
        desc="Provide the facility name.",
        parent=required_fields_node,
        critical=True,
    )
    evaluator.add_custom_node(
        result=_has_text(specs.address),
        id="Complete_Address_Provided",
        desc="Provide the complete address of the facility.",
        parent=required_fields_node,
        critical=True,
    )
    evaluator.add_custom_node(
        result=_has_text(specs.total_square_footage) or _has_text(specs.building_size),
        id="Total_Square_Footage_Provided",
        desc="Provide the total square footage of the facility.",
        parent=required_fields_node,
        critical=True,
    )
    evaluator.add_custom_node(
        result=_has_text(specs.developer),
        id="Developer_Name_Provided",
        desc="Provide the name of the developer.",
        parent=required_fields_node,
        critical=True,
    )

    # Constraints (critical)
    constraints_node = evaluator.add_parallel(
        id="Facility_Constraints",
        desc="The identified facility must satisfy all specifications stated in the question/constraints.",
        parent=main_node,
        critical=True,
    )

    name_for_claim = specs.facility_name or "the facility"

    # Prepare leaves and batch verifications
    batch: List[tuple[str, Optional[List[str]], Any, Optional[str]]] = []

    # Geographic_Location
    node_geo = evaluator.add_leaf(
        id="Geographic_Location",
        desc="Facility is located in the Atlanta metropolitan area, Dallas-Fort Worth metropolitan area, or Tampa Bay area.",
        parent=constraints_node,
        critical=True,
    )
    claim_geo = (
        f"The facility '{name_for_claim}' is located in the Atlanta, Dallas-Fort Worth (DFW), or Tampa Bay metropolitan area. "
        f"If the webpage indicates a city or county that is part of one of these metros (e.g., Cobb/DeKalb/Gwinnett for Atlanta; "
        f"Dallas/Tarrant/Collin/Denton for DFW; Hillsborough/Pinellas/Pasco for Tampa Bay), consider this satisfied."
    )
    add_geo = (
        "Use the address, city, county, or metro descriptors on the page. If the city is a known suburb within the metro, "
        "count it as within the specified metro even if the metro name is not stated verbatim."
    )
    batch.append((claim_geo, sources, node_geo, add_geo))

    # Building_Classification
    node_class = evaluator.add_leaf(
        id="Building_Classification",
        desc="Facility is classified as Class A industrial or warehouse property.",
        parent=constraints_node,
        critical=True,
    )
    claim_class = f"The facility '{name_for_claim}' is a Class A industrial/warehouse property."
    add_class = (
        "Look for phrases like 'Class A industrial', 'Class A warehouse', 'Class A distribution' or equivalent. "
        "Do not accept Class B or C."
    )
    batch.append((claim_class, sources, node_class, add_class))

    # Development_Timeline
    node_timeline = evaluator.add_leaf(
        id="Development_Timeline",
        desc="Facility was completed or began construction in 2024 or later.",
        parent=constraints_node,
        critical=True,
    )
    claim_time = (
        f"The facility '{name_for_claim}' was completed or began construction in 2024 or later."
    )
    add_time = (
        "Accept phrasing like 'delivers in 2024/2025+', 'construction began in 2024+', 'completion 2024+'. "
        "If a range is given, ensure that either construction start or completion is 2024 or later."
    )
    batch.append((claim_time, sources, node_timeline, add_time))

    # Building_Size_Requirement
    node_size = evaluator.add_leaf(
        id="Building_Size_Requirement",
        desc="Facility has a minimum building size of 500,000 square feet.",
        parent=constraints_node,
        critical=True,
    )
    claim_size = (
        f"The facility '{name_for_claim}' has total building size at least 500,000 square feet."
    )
    add_size = (
        "Look for total building area figures like 'SF', 'square feet', or 'MSF'. If multiple buildings are discussed, "
        "ensure the one for this facility meets or exceeds 500,000 SF."
    )
    batch.append((claim_size, sources, node_size, add_size))

    # Clear_Height_Specification
    node_clear = evaluator.add_leaf(
        id="Clear_Height_Specification",
        desc="Facility has a minimum clear height of 32 feet.",
        parent=constraints_node,
        critical=True,
    )
    claim_clear = f"The facility '{name_for_claim}' has a clear height of at least 32 feet."
    add_clear = "Accept mentions like '32 ft clear', '36' clear', '40-foot clear'."
    batch.append((claim_clear, sources, node_clear, add_clear))

    # Loading_Dock_Count
    node_docks = evaluator.add_leaf(
        id="Loading_Dock_Count",
        desc="Facility has at least 50 dock-high loading doors.",
        parent=constraints_node,
        critical=True,
    )
    claim_docks = f"The facility '{name_for_claim}' has at least 50 dock-high loading doors."
    add_docks = (
        "Look for 'dock doors', 'dock-high doors', 'DHD', or total dock count. Trailer positions are not a substitute "
        "unless explicitly described as dock-high doors."
    )
    batch.append((claim_docks, sources, node_docks, add_docks))

    # Column_Spacing
    node_col = evaluator.add_leaf(
        id="Column_Spacing",
        desc="Facility has column spacing of at least 50 feet.",
        parent=constraints_node,
        critical=True,
    )
    claim_col = (
        f"The facility '{name_for_claim}' has column spacing of at least 50 feet."
    )
    add_col = (
        "Accept typical descriptions like '50' x 56'' or '52' x 56''. If a pair is given (e.g., 50' x 56'), "
        "use the smaller dimension for the minimum check; it must be ≥ 50'."
    )
    batch.append((claim_col, sources, node_col, add_col))

    # Truck_Court_Depth
    node_truck = evaluator.add_leaf(
        id="Truck_Court_Depth",
        desc="Facility has a truck court depth of at least 130 feet.",
        parent=constraints_node,
        critical=True,
    )
    claim_truck = f"The facility '{name_for_claim}' has truck court depth of at least 130 feet."
    add_truck = "Look for 'truck court', 'court depth', '130 ft', '185-foot truck courts', etc."
    batch.append((claim_truck, sources, node_truck, add_truck))

    # Cross_Dock_Configuration
    node_cross = evaluator.add_leaf(
        id="Cross_Dock_Configuration",
        desc="Facility has a cross-dock configuration.",
        parent=constraints_node,
        critical=True,
    )
    claim_cross = f"The facility '{name_for_claim}' has a cross-dock configuration (two-sided loading)."
    add_cross = "Accept 'cross-dock', 'cross dock', or clearly two-sided loading configurations."
    batch.append((claim_cross, sources, node_cross, add_cross))

    # Interstate_Access
    node_interstate = evaluator.add_leaf(
        id="Interstate_Access",
        desc="Facility is located within 10 miles of a major interstate highway.",
        parent=constraints_node,
        critical=True,
    )
    claim_interstate = (
        f"The facility '{name_for_claim}' is within 10 miles of a major interstate highway."
    )
    add_interstate = (
        "Accept descriptions like 'adjacent to I-__', 'immediate access to I-__', or explicit distances ≤ 10 miles. "
        "Proximity statements indicating 1-5 miles or immediate/on-interchange qualify."
    )
    batch.append((claim_interstate, sources, node_interstate, add_interstate))

    # Fire_Protection_System (ESFR)
    node_esfr = evaluator.add_leaf(
        id="Fire_Protection_System",
        desc="Facility is equipped with an ESFR (Early Suppression Fast Response) sprinkler system.",
        parent=constraints_node,
        critical=True,
    )
    claim_esfr = f"The facility '{name_for_claim}' is equipped with an ESFR sprinkler system."
    add_esfr = "Look for 'ESFR sprinklers' or 'ESFR'."
    batch.append((claim_esfr, sources, node_esfr, add_esfr))

    # Floor_Specifications
    node_floor = evaluator.add_leaf(
        id="Floor_Specifications",
        desc="Facility has a minimum 6-inch concrete (reinforced) floor slab.",
        parent=constraints_node,
        critical=True,
    )
    claim_floor = (
        f"The facility '{name_for_claim}' has a concrete floor slab thickness of at least 6 inches."
    )
    add_floor = "Accept '6\" slab', '7-inch slab', or greater. Reinforcement note is not required if thickness is sufficient."
    batch.append((claim_floor, sources, node_floor, add_floor))

    # LED_Lighting
    node_led = evaluator.add_leaf(
        id="LED_Lighting",
        desc="Facility has an LED lighting system.",
        parent=constraints_node,
        critical=True,
    )
    claim_led = f"The facility '{name_for_claim}' has LED lighting."
    add_led = "Look for 'LED lighting' or equivalent energy-efficient LED descriptions."
    batch.append((claim_led, sources, node_led, add_led))

    # Building_Depth
    node_depth = evaluator.add_leaf(
        id="Building_Depth",
        desc="Facility has a minimum building depth of 250 feet.",
        parent=constraints_node,
        critical=True,
    )
    claim_depth = f"The facility '{name_for_claim}' has a building depth of at least 250 feet."
    add_depth = (
        "Look for 'building depth' or comparable dimensional references (e.g., depth from dock wall to dock wall) "
        "that are ≥ 250 ft."
    )
    batch.append((claim_depth, sources, node_depth, add_depth))

    # Trailer_Parking
    node_trailer = evaluator.add_leaf(
        id="Trailer_Parking",
        desc="Facility has at least 100 trailer parking spaces.",
        parent=constraints_node,
        critical=True,
    )
    claim_trailer = f"The facility '{name_for_claim}' has at least 100 trailer parking spaces."
    add_trailer = (
        "Look for 'trailer parking', 'trailer stalls', 'trailer positions', or 'trailer drops' counts; "
        "ensure the number is ≥ 100."
    )
    batch.append((claim_trailer, sources, node_trailer, add_trailer))

    # Office_Space
    node_office = evaluator.add_leaf(
        id="Office_Space",
        desc="Facility includes office space.",
        parent=constraints_node,
        critical=True,
    )
    claim_office = f"The facility '{name_for_claim}' includes office space."
    add_office = "Accept 'office to suit', 'build-to-suit office', 'office included', or equivalent."
    batch.append((claim_office, sources, node_office, add_office))

    # Development_Type
    node_devtype = evaluator.add_leaf(
        id="Development_Type",
        desc="Facility is a speculative development or a build-to-suit project.",
        parent=constraints_node,
        critical=True,
    )
    claim_devtype = (
        f"The facility '{name_for_claim}' is either a speculative (spec) development or a build-to-suit (BTS) project."
    )
    add_devtype = "Accept 'speculative', 'spec', 'build-to-suit', 'BTS'."
    batch.append((claim_devtype, sources, node_devtype, add_devtype))

    # Run batch verification (parallel verifications with automatic precondition gating)
    await evaluator.batch_verify(batch)


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
    model: str = "o4-mini",
) -> Dict:
    """
    Evaluate an answer for the Class A warehouse facility specification task.
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
        default_model=model,
    )

    # Extract structured facility information
    specs: FacilitySpecs = await evaluator.extract(
        prompt=prompt_extract_facility(),
        template_class=FacilitySpecs,
        extraction_name="facility_extraction",
    )

    # Optional: record the target metros for reference
    evaluator.add_custom_info(
        {"allowed_metros": ["Atlanta", "Dallas-Fort Worth", "Tampa Bay"], "year_threshold": 2024},
        info_type="constraints_context",
        info_name="constraints_context",
    )

    # Build verification tree and run checks
    await verify_facility(evaluator, root, specs)

    # Return structured evaluation summary
    return evaluator.get_summary()