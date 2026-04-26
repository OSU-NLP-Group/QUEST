import asyncio
import logging
import math
import re
from typing import Any, Optional, List, Dict

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy

# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "hou_restaurant_code_ada"
TASK_DESCRIPTION = (
    "A commercial developer is planning to build a new full-service restaurant in Houston, Texas. "
    "The project specifications are: total building area of 6,500 square feet (3,900 square feet dining area with tables and chairs, "
    "2,600 square feet kitchen and support areas), commercial kitchen equipped with fryers, griddles, and ranges, and a parking lot with 42 total parking spaces. "
    "What are the mandatory building code and ADA compliance requirements that must be met for this restaurant development? "
    "Provide specific numerical thresholds, ratios, and technical specifications required by applicable building codes and standards."
)

# Project parameters (from the task description)
TOTAL_BUILDING_AREA_SQFT = 6500
DINING_AREA_SQFT = 3900
TOTAL_PARKING_SPACES = 42

# Ground-truth derivations for checks based on widely used scoping tables (ADA) and straightforward arithmetic
OCCUPANT_LOAD_FACTOR_SQFT_PER_PERSON = 15
EXPECTED_OCCUPANT_LOAD = round(DINING_AREA_SQFT / OCCUPANT_LOAD_FACTOR_SQFT_PER_PERSON)  # 260

# ADA Table 208.2 (typical scoping used; for total=42 spaces => 2 accessible)
def compute_required_accessible_spaces(total_spaces: int) -> int:
    if total_spaces <= 25:
        return 1
    elif total_spaces <= 50:
        return 2
    elif total_spaces <= 75:
        return 3
    elif total_spaces <= 100:
        return 4
    elif total_spaces <= 150:
        return 5
    elif total_spaces <= 200:
        return 6
    elif total_spaces <= 300:
        return 7
    elif total_spaces <= 400:
        return 8
    elif total_spaces <= 500:
        return 9
    elif total_spaces <= 1000:
        return math.ceil(total_spaces * 0.02)
    else:
        # 1001 and over: 20 + 1 for each 100 over 1000
        over_1000 = total_spaces - 1000
        return 20 + math.ceil(over_1000 / 100.0)

EXPECTED_ACCESSIBLE_SPACES = compute_required_accessible_spaces(TOTAL_PARKING_SPACES)
EXPECTED_VAN_SPACES = max(1, math.ceil(EXPECTED_ACCESSIBLE_SPACES / 6.0))  # At least 1 of every 6 accessible spaces or fraction thereof

# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class NFPA13Specs(BaseModel):
    """NFPA 13-related specifications extracted from the answer."""
    sprinkler_required_stated: Optional[bool] = None  # True only if explicitly stated; null otherwise
    reason_threshold_sqft_mentioned: Optional[str] = None  # e.g., "5,000 sq ft"
    minimum_spacing_value: Optional[str] = None       # e.g., "10 ft"
    maximum_coverage_value: Optional[str] = None      # e.g., "225 sq ft"


class OccupantLoadInfo(BaseModel):
    """Occupant load factor and computed value for the dining area."""
    factor_sqft_per_person: Optional[str] = None      # e.g., "15"
    dining_area_sqft_cited: Optional[str] = None      # e.g., "3,900 sq ft"
    occupant_load_value_reported: Optional[str] = None  # e.g., "260"


class TypeIHoodSpecs(BaseModel):
    """Type I hood requirement and installation specs extracted from the answer."""
    hood_required_over_grease_appliances_stated: Optional[bool] = None
    appliances_listed: List[str] = Field(default_factory=list)  # e.g., ["fryers", "griddles", "ranges"]
    hood_overhang_minimum: Optional[str] = None       # e.g., "6 inches"
    clearance_to_combustibles_minimum: Optional[str] = None  # e.g., "18 inches"


class ADAParkingInfo(BaseModel):
    """ADA accessible and van-accessible parking counts extracted from the answer."""
    total_parking_spaces_cited: Optional[str] = None
    accessible_spaces_count: Optional[str] = None
    van_accessible_spaces_count: Optional[str] = None


class RestaurantComplianceExtraction(BaseModel):
    """Top-level extraction structure for all compliance items."""
    nfpa13: Optional[NFPA13Specs] = None
    occupant_load: Optional[OccupantLoadInfo] = None
    hood: Optional[TypeIHoodSpecs] = None
    ada_parking: Optional[ADAParkingInfo] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_restaurant_compliance() -> str:
    return """
    Extract from the provided answer the specific compliance-related items listed below. Follow these strict rules:
    - Do not invent or infer values. Only extract what is explicitly present in the answer.
    - For boolean fields (e.g., “sprinkler_required_stated”), set to true ONLY if the answer explicitly states the requirement; set to null if not explicitly stated. Do not set false unless the answer explicitly states the requirement is NOT needed.
    - For numeric specifications, extract the value exactly as written (including units), e.g., “10 ft”, “225 sq ft”, “6 inches”.
    - If a field is missing in the answer, return null for that field.

    Fields to extract:

    nfpa13:
      - sprinkler_required_stated: true/null based on whether the answer explicitly states that an automatic fire sprinkler system is required because the fire area exceeds 5,000 sq ft (or equivalent phrasing).
      - reason_threshold_sqft_mentioned: the threshold cited (e.g., “5,000 sq ft”) if present in the answer.
      - minimum_spacing_value: the minimum sprinkler spacing value (distance, with units) per NFPA 13, if present.
      - maximum_coverage_value: the maximum coverage area per sprinkler (square feet or square meters, with units) per NFPA 13, if present.

    occupant_load:
      - factor_sqft_per_person: the occupant load factor used for dining areas with tables/chairs (e.g., “15 sq ft/person”), if present.
      - dining_area_sqft_cited: the dining area size cited in the answer (e.g., “3,900 sq ft”), if present.
      - occupant_load_value_reported: the occupant load value reported in the answer for the dining area, if present (e.g., “260”).

    hood:
      - hood_required_over_grease_appliances_stated: true/null based on whether the answer explicitly states that a Type I grease hood is required over the listed grease-producing appliances (fryers, griddles, ranges).
      - appliances_listed: list of any grease-producing appliances explicitly mentioned (e.g., “fryers”, “griddles”, “ranges”).
      - hood_overhang_minimum: the minimum hood overhang beyond the equipment (numeric + units), if present.
      - clearance_to_combustibles_minimum: the minimum clearance to combustibles (numeric + units), if present.

    ada_parking:
      - total_parking_spaces_cited: the total number of parking spaces as cited in the answer (if mentioned).
      - accessible_spaces_count: the number of accessible parking spaces the answer reports are required, if present.
      - van_accessible_spaces_count: the number of van-accessible spaces the answer reports are required, if present.
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def parse_first_int(value: Optional[str]) -> Optional[int]:
    if not value or not isinstance(value, str):
        return None
    m = re.search(r"\d+", value.replace(",", ""))
    return int(m.group(0)) if m else None


# --------------------------------------------------------------------------- #
# Verification subtrees                                                       #
# --------------------------------------------------------------------------- #
async def verify_nfpa13(evaluator: Evaluator, parent_node, extraction: Optional[NFPA13Specs]) -> None:
    """
    Verify the NFPA 13-related compliance: sprinkler requirement due to >5,000 sq ft fire area,
    and presence of minimum spacing and maximum coverage specifications.
    """
    nfpa_node = evaluator.add_parallel(
        id="fire_sprinkler_system_and_nfpa13_specs",
        desc="States that an automatic fire sprinkler system is required because the fire area exceeds 5,000 sq ft, "
             "and specifies the NFPA 13 layout criteria (minimum spacing and maximum coverage per sprinkler).",
        parent=parent_node,
        critical=True,
    )

    # Leaf 1: Answer states sprinkler is required due to threshold >5,000 sq ft
    spr_req_leaf = evaluator.add_leaf(
        id="sprinkler_required_due_to_5000_threshold",
        desc="Answer states sprinkler is required because fire area exceeds 5,000 sq ft.",
        parent=nfpa_node,
        critical=True,
    )
    claim1 = (
        "The answer states that an automatic fire sprinkler system is required because the fire area exceeds 5,000 square feet "
        f"for this 6,500 square foot project."
    )
    await evaluator.verify(
        claim=claim1,
        node=spr_req_leaf,
        additional_instruction=(
            "Confirm the answer explicitly ties the sprinkler requirement to exceeding a 5,000 sq ft fire area threshold. "
            "Allow minor wording variations (e.g., '>5000 sq ft', 'over 5000 sq ft')."
        )
    )

    # Leaf 2: Minimum sprinkler spacing spec is present with a numeric value
    min_spacing_leaf = evaluator.add_leaf(
        id="nfpa13_min_spacing_spec_present",
        desc="Answer specifies the minimum spacing between sprinklers (numeric distance) per NFPA 13.",
        parent=nfpa_node,
        critical=True,
    )
    claim2 = "The answer includes a numeric minimum sprinkler spacing requirement per NFPA 13 (e.g., a distance in feet or meters)."
    await evaluator.verify(
        claim=claim2,
        node=min_spacing_leaf,
        additional_instruction="Look for phrasing like 'minimum spacing between sprinklers' and ensure a number + unit is present."
    )

    # Leaf 3: Maximum coverage per sprinkler spec is present with a numeric value
    max_coverage_leaf = evaluator.add_leaf(
        id="nfpa13_max_coverage_spec_present",
        desc="Answer specifies the maximum coverage area per sprinkler (numeric square-foot value) per NFPA 13.",
        parent=nfpa_node,
        critical=True,
    )
    claim3 = "The answer includes a numeric maximum coverage area per sprinkler per NFPA 13 (e.g., a value in square feet)."
    await evaluator.verify(
        claim=claim3,
        node=max_coverage_leaf,
        additional_instruction="Look for terms like 'maximum coverage per sprinkler' and ensure a numeric area (e.g., sq ft) is present."
    )


async def verify_occupant_load(evaluator: Evaluator, parent_node, extraction: Optional[OccupantLoadInfo]) -> None:
    """
    Verify occupant load methodology: factor of 15 sq ft/person is used and correctly applied to 3,900 sq ft.
    """
    occ_node = evaluator.add_sequential(
        id="occupant_load_method_for_dining_area",
        desc="Uses occupant load factor 15 sq ft/person for dining with tables/chairs and correctly applies it to 3,900 sq ft.",
        parent=parent_node,
        critical=True,
    )

    # Leaf 1: Factor 15 sq ft/person is explicitly used
    factor_leaf = evaluator.add_leaf(
        id="occupant_load_factor_15_used",
        desc="Answer uses 15 sq ft/person factor for dining areas with tables/chairs.",
        parent=occ_node,
        critical=True,
    )
    claim_factor = (
        "The answer uses an occupant load factor of 15 square feet per person for dining areas with tables and chairs."
    )
    await evaluator.verify(
        claim=claim_factor,
        node=factor_leaf,
        additional_instruction="Allow minor wording variations (e.g., '15 sf/person', '15 sq ft per person', 'net 15')."
    )

    # Leaf 2: Correct occupant load is reported (3,900 / 15 ≈ 260)
    reported_value = extraction.occupant_load_value_reported if extraction else None
    reported_int = parse_first_int(reported_value)
    correct = (reported_int == EXPECTED_OCCUPANT_LOAD)
    evaluator.add_custom_node(
        result=bool(reported_int) and correct,
        id="occupant_load_correct_for_3900",
        desc=f"Correct occupant load for 3,900 sq ft at 15 sq ft/person is {EXPECTED_OCCUPANT_LOAD}, "
             f"and the answer reports {reported_int if reported_int is not None else 'null'}.",
        parent=occ_node,
        critical=True
    )


async def verify_type_i_hood(evaluator: Evaluator, parent_node, extraction: Optional[TypeIHoodSpecs]) -> None:
    """
    Verify Type I hood requirements and installation specs presence: overhang and clearance to combustibles.
    """
    hood_node = evaluator.add_parallel(
        id="type_i_hood_and_installation_specs",
        desc="Identifies Type I hood required over fryers, griddles, and ranges and includes minimum overhang and clearance specs.",
        parent=parent_node,
        critical=True,
    )

    # Leaf 1: Type I hood required over grease appliances
    hood_req_leaf = evaluator.add_leaf(
        id="type_i_hood_required_over_grease_appliances",
        desc="Answer states a Type I grease hood is required over fryers, griddles, and ranges.",
        parent=hood_node,
        critical=True,
    )
    claim_hood = "The answer states that a Type I grease hood is required over fryers, griddles, and ranges."
    await evaluator.verify(
        claim=claim_hood,
        node=hood_req_leaf,
        additional_instruction="Allow minor synonyms for appliances and hood naming; ensure Type I is explicit."
    )

    # Leaf 2: Minimum hood overhang beyond equipment is included with numeric value
    overhang_leaf = evaluator.add_leaf(
        id="hood_overhang_min_spec_included",
        desc="Answer includes a numeric minimum hood overhang beyond the cooking equipment.",
        parent=hood_node,
        critical=True,
    )
    claim_overhang = "The answer includes a numeric minimum hood overhang distance beyond the equipment."
    await evaluator.verify(
        claim=claim_overhang,
        node=overhang_leaf,
        additional_instruction="Look for values like '6 inches', '12 inches', etc., explicitly tied to hood overhang."
    )

    # Leaf 3: Minimum clearance to combustibles is included with numeric value
    clearance_leaf = evaluator.add_leaf(
        id="clearance_to_combustibles_min_spec_included",
        desc="Answer includes a numeric minimum clearance to combustibles for the hood/duct system.",
        parent=hood_node,
        critical=True,
    )
    claim_clearance = "The answer includes a numeric minimum clearance to combustibles for the hood or duct system."
    await evaluator.verify(
        claim=claim_clearance,
        node=clearance_leaf,
        additional_instruction="Look for explicit numeric clearance distances (e.g., inches) tied to combustibles."
    )


async def verify_ada_parking(evaluator: Evaluator, parent_node, extraction: Optional[ADAParkingInfo]) -> None:
    """
    Verify ADA accessible and van-accessible parking counts derived for 42 total spaces.
    """
    ada_node = evaluator.add_sequential(
        id="ada_accessible_and_van_parking_provisions",
        desc="Applies ADA ratios to 42 total spaces: correct accessible and van-accessible counts are reported.",
        parent=parent_node,
        critical=True,
    )

    # Leaf 1: Counts provided (accessible and van)
    accessible_str = extraction.accessible_spaces_count if extraction else None
    van_str = extraction.van_accessible_spaces_count if extraction else None
    accessible_int = parse_first_int(accessible_str)
    van_int = parse_first_int(van_str)

    counts_provided = (accessible_int is not None) and (van_int is not None)
    evaluator.add_custom_node(
        result=counts_provided,
        id="ada_counts_provided",
        desc=f"Answer provides both accessible and van-accessible counts. "
             f"Accessible: {accessible_str}, Van: {van_str}.",
        parent=ada_node,
        critical=True
    )

    # Leaf 2: Accessible count is correct for 42 spaces
    accessible_correct = (accessible_int == EXPECTED_ACCESSIBLE_SPACES)
    evaluator.add_custom_node(
        result=bool(accessible_int) and accessible_correct,
        id="ada_accessible_count_correct",
        desc=f"Required accessible spaces for {TOTAL_PARKING_SPACES} total spaces is {EXPECTED_ACCESSIBLE_SPACES}; "
             f"answer reports {accessible_int if accessible_int is not None else 'null'}.",
        parent=ada_node,
        critical=True
    )

    # Leaf 3: Van-accessible count is correct (at least 1 of every 6 accessible spaces or fraction thereof)
    van_correct = (van_int == EXPECTED_VAN_SPACES)
    evaluator.add_custom_node(
        result=bool(van_int) and van_correct,
        id="ada_van_count_correct",
        desc=f"Required van-accessible spaces (ceil of accessible/6) for {EXPECTED_ACCESSIBLE_SPACES} accessible spaces is {EXPECTED_VAN_SPACES}; "
             f"answer reports {van_int if van_int is not None else 'null'}.",
        parent=ada_node,
        critical=True
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
    """
    Evaluate the restaurant compliance answer against mandatory building code and ADA requirements.
    """
    # Initialize evaluator
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Independent criteria; partial credit allowed at root
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

    # Add top-level rubric node mirroring the JSON structure
    main_node = evaluator.add_parallel(
        id="restaurant_compliance_requirements",
        desc="Evaluate whether the answer identifies and specifies all mandatory building code and ADA compliance requirements with correct "
             "numerical thresholds/ratios/specifications and correct application to the provided project parameters.",
        parent=root,
        critical=False
    )

    # Extract structured information from the answer
    extraction = await evaluator.extract(
        prompt=prompt_extract_restaurant_compliance(),
        template_class=RestaurantComplianceExtraction,
        extraction_name="restaurant_compliance_extraction",
    )

    # Record ground truth / derived expectations for transparency
    evaluator.add_ground_truth({
        "project_parameters": {
            "total_building_area_sqft": TOTAL_BUILDING_AREA_SQFT,
            "dining_area_sqft": DINING_AREA_SQFT,
            "total_parking_spaces": TOTAL_PARKING_SPACES
        },
        "expected": {
            "occupant_load_factor_sqft_per_person": OCCUPANT_LOAD_FACTOR_SQFT_PER_PERSON,
            "expected_occupant_load_dining_area": EXPECTED_OCCUPANT_LOAD,
            "ada_required_accessible_spaces": EXPECTED_ACCESSIBLE_SPACES,
            "ada_required_van_accessible_spaces": EXPECTED_VAN_SPACES,
            "sprinkler_threshold_fire_area_sqft": 5000
        }
    })

    # Build verification subtrees
    await verify_nfpa13(evaluator, main_node, extraction.nfpa13)
    await verify_occupant_load(evaluator, main_node, extraction.occupant_load)
    await verify_type_i_hood(evaluator, main_node, extraction.hood)
    await verify_ada_parking(evaluator, main_node, extraction.ada_parking)

    # Return structured summary
    return evaluator.get_summary()