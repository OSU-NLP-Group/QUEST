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
TASK_ID = "nfl_smallest_stadium_2024_2025"
TASK_DESCRIPTION = (
    "Identify the NFL stadium with the smallest seating capacity for the 2024-2025 season. "
    "Provide the stadium's official name, its seating capacity, its complete address (including street address, "
    "city, state, and zip code), and describe one unique or notable feature of its playing surface."
)

# Ground-truth expectations encoded from the rubric constraints
EXPECTED_STADIUM_NAME = "Soldier Field"
EXPECTED_CAPACITY_NUM = 61500
EXPECTED_CAPACITY_STR = "61,500"
EXPECTED_TEAM = "Chicago Bears"
EXPECTED_ADDRESS_STREET = "1410 Special Olympics Drive"
EXPECTED_ADDRESS_CITY = "Chicago"
EXPECTED_ADDRESS_STATE = "IL"
EXPECTED_ADDRESS_ZIP = "60605"
EXPECTED_ADDRESS_FULL = f"{EXPECTED_ADDRESS_STREET}, {EXPECTED_ADDRESS_CITY}, {EXPECTED_ADDRESS_STATE} {EXPECTED_ADDRESS_ZIP}"
EXPECTED_SURFACE_FEATURE = "natural grass with an underground radiant heating system"


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class StadiumExtraction(BaseModel):
    stadium_name: Optional[str] = None
    seating_capacity: Optional[str] = None
    address_street: Optional[str] = None
    address_city: Optional[str] = None
    address_state: Optional[str] = None
    address_zip: Optional[str] = None
    address_full: Optional[str] = None
    home_team: Optional[str] = None
    playing_surface_feature: Optional[str] = None
    source_urls: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_stadium_info() -> str:
    return """
    Extract the following information about the NFL stadium identified in the answer for the 2024–2025 season:
    - stadium_name: The official name of the stadium presented in the answer (not a nickname).
    - seating_capacity: The seating capacity stated in the answer (keep as a string exactly as written, including commas).
    - address_street: The street address as presented (e.g., "1410 Special Olympics Drive").
    - address_city: The city.
    - address_state: The two-letter state abbreviation.
    - address_zip: The 5-digit ZIP code.
    - address_full: The complete address string exactly as written in the answer (street, city, state, zip).
    - home_team: The NFL team identified as using this stadium as home.
    - playing_surface_feature: A notable or unique feature of the playing surface described in the answer (e.g., material and heating system).
    - source_urls: List all URLs explicitly cited in the answer as sources supporting the claims. Include all relevant URLs from the answer text; do not invent any.

    If any field is not present in the answer, set it to null (for strings) or an empty list (for source_urls).
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def assemble_answer_address(extracted: StadiumExtraction) -> str:
    if extracted.address_full and extracted.address_full.strip():
        return extracted.address_full.strip()
    parts = []
    if extracted.address_street:
        parts.append(extracted.address_street.strip())
    city_state_zip = []
    if extracted.address_city:
        city_state_zip.append(extracted.address_city.strip())
    if extracted.address_state:
        city_state_zip.append(extracted.address_state.strip())
    cs = ", ".join(city_state_zip) if city_state_zip else ""
    if cs:
        parts.append(cs)
    if extracted.address_zip:
        if parts:
            parts[-1] = f"{parts[-1]} {extracted.address_zip.strip()}"
        else:
            parts.append(extracted.address_zip.strip())
    return ", ".join(parts).strip()


def safe(s: Optional[str]) -> str:
    return s if s else "N/A"


# --------------------------------------------------------------------------- #
# Verification subtrees                                                       #
# --------------------------------------------------------------------------- #
async def verify_stadium_identification(evaluator: Evaluator, parent_node, extracted: StadiumExtraction) -> None:
    """
    Build and verify the 'Stadium_Identification' critical parallel group:
    - Official_Stadium_Name_Is_Correct
    - Smallest_Among_Current_NFL_Stadiums
    - Seating_Capacity_Equals_61500
    - Home_Of_Chicago_Bears
    """
    sources = extracted.source_urls or []
    stadium_name = safe(extracted.stadium_name)
    reported_capacity = safe(extracted.seating_capacity)
    reported_team = safe(extracted.home_team)

    stadium_node = evaluator.add_parallel(
        id="Stadium_Identification",
        desc="Identify the correct stadium meeting the constraints (smallest capacity, specified capacity value, and home team constraint).",
        parent=parent_node,
        critical=True
    )

    # 1) Official_Stadium_Name_Is_Correct
    node_official_name = evaluator.add_leaf(
        id="Official_Stadium_Name_Is_Correct",
        desc="Provides the stadium’s official name and it matches authoritative/official sources for that stadium (not merely a nickname or incorrect variant).",
        parent=stadium_node,
        critical=True
    )
    claim_official_name = f"The official name of the stadium is '{stadium_name}'."
    add_ins_official_name = (
        "Verify using authoritative sources (official team site, stadium site, NFL page, city/park district). "
        "Do not accept nicknames or outdated sponsor names. "
        f"The answer explicitly reported the official name as: {stadium_name}. "
        "If the answer does not explicitly provide the official stadium name, mark incorrect."
    )
    await evaluator.verify(
        claim=claim_official_name,
        node=node_official_name,
        sources=sources,
        additional_instruction=add_ins_official_name
    )

    # 2) Smallest_Among_Current_NFL_Stadiums
    node_smallest = evaluator.add_leaf(
        id="Smallest_Among_Current_NFL_Stadiums",
        desc="The identified stadium is the smallest-capacity current NFL stadium for the 2024–2025 season, consistent with the constraints’ requirement.",
        parent=stadium_node,
        critical=True
    )
    subj_name = stadium_name if stadium_name != "N/A" else "the identified stadium"
    claim_smallest = f"Among current NFL stadiums for the 2024–2025 season, {subj_name} has the smallest seating capacity."
    add_ins_smallest = (
        "Use reputable/authoritative lists or official sources. "
        "Judge based on stated 'seating capacity' (not attendance records, expandable capacities, or standing room). "
        "Focus on the 2024–2025 season. If any source shows another current NFL home stadium has a smaller seating capacity, mark this claim incorrect."
    )
    await evaluator.verify(
        claim=claim_smallest,
        node=node_smallest,
        sources=sources,
        additional_instruction=add_ins_smallest
    )

    # 3) Seating_Capacity_Equals_61500
    node_capacity = evaluator.add_leaf(
        id="Seating_Capacity_Equals_61500",
        desc="The seating capacity stated is 61,500 (as specified in the constraints).",
        parent=stadium_node,
        critical=True
    )
    claim_capacity = "The stadium's seating capacity is 61,500."
    add_ins_capacity = (
        f"The answer reported the stadium capacity as: {reported_capacity}. "
        "Verify the capacity with sources and require that it equals 61,500 (allow minor formatting like commas). "
        "If the answer omits capacity or reports a value other than 61,500, mark this as incorrect."
    )
    await evaluator.verify(
        claim=claim_capacity,
        node=node_capacity,
        sources=sources,
        additional_instruction=add_ins_capacity
    )

    # 4) Home_Of_Chicago_Bears
    node_home = evaluator.add_leaf(
        id="Home_Of_Chicago_Bears",
        desc="The stadium is identified as the home of the Chicago Bears (as specified in the constraints).",
        parent=stadium_node,
        critical=True
    )
    claim_home = f"{subj_name} is the home stadium of the Chicago Bears."
    add_ins_home = (
        f"The answer stated the home team as: {reported_team}. "
        "Verify via official/authoritative sources (team site, stadium site, NFL). "
        "If the answer does not say Chicago Bears or gives a different team, mark incorrect."
    )
    await evaluator.verify(
        claim=claim_home,
        node=node_home,
        sources=sources,
        additional_instruction=add_ins_home
    )


async def verify_required_details(evaluator: Evaluator, parent_node, extracted: StadiumExtraction) -> None:
    """
    Build and verify the 'Required_Details' critical parallel group:
    - Address_Matches_Specified_Address
    - Playing_Surface_Feature_Matches_Constraint
    """
    sources = extracted.source_urls or []
    provided_address = assemble_answer_address(extracted)
    provided_surface = safe(extracted.playing_surface_feature)

    details_node = evaluator.add_parallel(
        id="Required_Details",
        desc="Provide the required complete address and a notable playing-surface feature under the constraints.",
        parent=parent_node,
        critical=True
    )

    # Address matches specified
    node_address = evaluator.add_leaf(
        id="Address_Matches_Specified_Address",
        desc=f"Provides the complete address and it matches: {EXPECTED_ADDRESS_FULL} (as specified in the constraints).",
        parent=details_node,
        critical=True
    )
    claim_address = f"The stadium's complete address is: {EXPECTED_ADDRESS_FULL}."
    add_ins_address = (
        f"The answer provided the address as: {provided_address if provided_address else 'N/A'}. "
        "Confirm with authoritative sources (stadium site, park district, official pages). "
        "Require an exact match to the specified address (minor variations like 'Dr' vs 'Drive' or case may be acceptable if clearly equivalent). "
        "If the answer omits the complete address (street, city, state, zip) or provides a different address, mark incorrect."
    )
    await evaluator.verify(
        claim=claim_address,
        node=node_address,
        sources=sources,
        additional_instruction=add_ins_address
    )

    # Playing surface feature matches constraint
    node_surface = evaluator.add_leaf(
        id="Playing_Surface_Feature_Matches_Constraint",
        desc="Describes a notable playing-surface feature consistent with the constraint: natural grass with an underground radiant heating system.",
        parent=details_node,
        critical=True
    )
    claim_surface = "The stadium has a natural grass playing surface with an underground radiant heating system."
    add_ins_surface = (
        f"The answer described the playing-surface feature as: {provided_surface}. "
        "Verify with authoritative sources. Accept synonymous phrasing such as 'under-soil heating', 'subsurface heating', or 'heating coils under the field'. "
        "If the answer omits a notable surface feature or describes something inconsistent with natural grass and an underground radiant heating system, mark incorrect."
    )
    await evaluator.verify(
        claim=claim_surface,
        node=node_surface,
        sources=sources,
        additional_instruction=add_ins_surface
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
    Evaluate an answer for the NFL smallest stadium 2024–2025 task.
    """
    # Initialize evaluator with a sequential root to reflect the overall workflow order
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
    extracted = await evaluator.extract(
        prompt=prompt_extract_stadium_info(),
        template_class=StadiumExtraction,
        extraction_name="stadium_extraction"
    )

    # Add ground-truth info (for transparency in summary)
    evaluator.add_ground_truth({
        "expected_stadium_name": EXPECTED_STADIUM_NAME,
        "expected_capacity": EXPECTED_CAPACITY_STR,
        "expected_home_team": EXPECTED_TEAM,
        "expected_address": EXPECTED_ADDRESS_FULL,
        "expected_playing_surface_feature": EXPECTED_SURFACE_FEATURE
    }, gt_type="expected_constraints")

    # Build verification tree according to rubric
    # 1) Stadium Identification (critical parallel group)
    await verify_stadium_identification(evaluator, root, extracted)

    # 2) Required Details (critical parallel group)
    await verify_required_details(evaluator, root, extracted)

    # Return evaluation summary
    return evaluator.get_summary()