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
TASK_ID = "ncaa_di_mbb_pa_nj_ct"
TASK_DESCRIPTION = """
Identify three NCAA Division I universities with active men's basketball programs, one from each of the following states: Pennsylvania, New Jersey, and Connecticut. For each university, provide: (1) the full institution name and specific city location, (2) the current athletic conference affiliation, (3) the name and tenure information of the current men's basketball head coach, (4) the official name and seating capacity of the primary home basketball arena or athletic facility, and (5) evidence that the university provides academic support services for student-athletes. Include reference URLs supporting each requirement.
"""


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class UniversityInfo(BaseModel):
    institution_name: Optional[str] = None
    city: Optional[str] = None
    state: Optional[str] = None  # Expect "Pennsylvania" / "New Jersey" / "Connecticut"
    # Location references (official site, Wikipedia, athletics site page that states city/location)
    location_urls: List[str] = Field(default_factory=list)

    # Program details
    division: Optional[str] = None  # Expect "NCAA Division I" or equivalent
    program_active: Optional[str] = None  # e.g., "active", "currently fields a team", or any text implying active
    conference: Optional[str] = None  # e.g., "Big East", "Atlantic 10", etc.
    program_urls: List[str] = Field(default_factory=list)  # URLs that support program info and/or conference

    # Head coach details
    head_coach_name: Optional[str] = None
    head_coach_tenure: Optional[str] = None  # e.g., "since 2022", "3rd season (2025-26)"
    coach_urls: List[str] = Field(default_factory=list)  # URLs supporting coach name/tenure

    # Facility details
    facility_name: Optional[str] = None  # official arena/facility name
    seating_capacity: Optional[str] = None  # keep as string to allow "10,506", "~9,500", etc.
    facility_urls: List[str] = Field(default_factory=list)  # URLs supporting facility name/capacity

    # Academic support services
    academic_support_description: Optional[str] = None  # Optional textual description if present
    academic_support_urls: List[str] = Field(default_factory=list)  # URLs showing academic support services


class UniversitiesExtraction(BaseModel):
    pennsylvania: Optional[UniversityInfo] = None
    new_jersey: Optional[UniversityInfo] = None
    connecticut: Optional[UniversityInfo] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_universities() -> str:
    return """
    From the provided answer, extract exactly one NCAA Division I university for each of the three U.S. states:
    - Pennsylvania
    - New Jersey
    - Connecticut

    For each state, extract an object with the following fields:
    - institution_name: full official name of the university (string)
    - city: the specific city or town (string)
    - state: the state name (string; must be "Pennsylvania", "New Jersey", or "Connecticut" as appropriate)
    - location_urls: list of URL(s) that support the institution name and city/state location (list of strings)

    - division: the NCAA division status for men's basketball (string; e.g., "NCAA Division I")
    - program_active: text indicating the men's basketball program is active (string; e.g., "active", or a phrase indicating current activity)
    - conference: the current athletic conference affiliation (string; e.g., "Big East", "Atlantic 10", etc.)
    - program_urls: list of URL(s) that support Division I status, active men's basketball program, and/or conference (list of strings)

    - head_coach_name: full name of the current men's basketball head coach (string)
    - head_coach_tenure: the tenure information, e.g., "since 2022" or "3rd season (2025-26)" (string)
    - coach_urls: list of URL(s) that support the head coach name and tenure information (list of strings)

    - facility_name: official name of the primary home basketball arena or athletic facility (string)
    - seating_capacity: the seating capacity as a specific number; you may keep commas or formatting (string)
    - facility_urls: list of URL(s) that support the arena name and seating capacity (list of strings)

    - academic_support_description: brief description or label for student-athlete academic support services (string; optional)
    - academic_support_urls: list of URL(s) that document academic support services for student-athletes (list of strings)

    IMPORTANT:
    - Extract only what is explicitly present in the answer. Do not invent or infer details not stated.
    - Return null for any field that is not provided in the answer. For URL lists, return an empty list if none are present.
    - For URLs, extract actual link strings. If presented as markdown links, extract the underlying URL. Only include valid URLs.
    - If the answer mentions multiple universities per state, choose the first one that reasonably fits; if none is provided for a state, return null for that state's object.

    The final JSON structure must be an object with keys: "pennsylvania", "new_jersey", "connecticut", each mapping to a UniversityInfo object as described.
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _safe_list(v: Optional[List[str]]) -> List[str]:
    return v if isinstance(v, list) else []


def _combine_sources(*url_lists: List[str]) -> List[str]:
    combined: List[str] = []
    for lst in url_lists:
        combined.extend(lst or [])
    # Deduplicate while preserving order
    seen = set()
    uniq: List[str] = []
    for u in combined:
        if u not in seen:
            seen.add(u)
            uniq.append(u)
    return uniq


# --------------------------------------------------------------------------- #
# Verification logic per university                                           #
# --------------------------------------------------------------------------- #
async def verify_university(
    evaluator: Evaluator,
    parent_node,
    info: Optional[UniversityInfo],
    state_label_id: str,         # e.g., "pennsylvania", "new_jersey", "connecticut"
    state_full_name: str,        # e.g., "Pennsylvania"
) -> None:
    """
    Build and verify the subtree for one university/state.
    This function is robust to missing info (info can be None).
    """
    uni_node = evaluator.add_parallel(
        id=f"university_{state_label_id}",
        desc=f"NCAA Division I university in {state_full_name} meeting all specified criteria",
        parent=parent_node,
        critical=False  # allow partial credit across states
    )

    # Prepare safe fields
    institution_name = info.institution_name if info and info.institution_name else ""
    city = info.city if info and info.city else ""
    state = info.state if info and info.state else ""
    location_urls = _safe_list(info.location_urls if info else [])

    division = info.division if info and info.division else ""
    program_active = info.program_active if info and info.program_active else ""
    conference = info.conference if info and info.conference else ""
    program_urls = _safe_list(info.program_urls if info else [])

    head_coach_name = info.head_coach_name if info and info.head_coach_name else ""
    head_coach_tenure = info.head_coach_tenure if info and info.head_coach_tenure else ""
    coach_urls = _safe_list(info.coach_urls if info else [])

    facility_name = info.facility_name if info and info.facility_name else ""
    seating_capacity = info.seating_capacity if info and info.seating_capacity else ""
    facility_urls = _safe_list(info.facility_urls if info else [])

    academic_support_description = info.academic_support_description if info and info.academic_support_description else ""
    academic_support_urls = _safe_list(info.academic_support_urls if info else [])

    # ------------------------ Location & Identification --------------------- #
    loc_node = evaluator.add_parallel(
        id=f"{state_label_id}_location_info",
        desc="University location and basic identification",
        parent=uni_node,
        critical=True
    )

    # location_reference: ensure there is at least one supporting URL
    location_ref_node = evaluator.add_custom_node(
        result=len(location_urls) > 0,
        id=f"{state_label_id}_location_reference",
        desc="Reference URL supporting location information",
        parent=loc_node,
        critical=True
    )

    # institution_name: presence is required
    evaluator.add_custom_node(
        result=(institution_name.strip() != ""),
        id=f"{state_label_id}_institution_name",
        desc="Full official university name is provided",
        parent=loc_node,
        critical=True
    )

    # city_identification: presence is required
    evaluator.add_custom_node(
        result=(city.strip() != ""),
        id=f"{state_label_id}_city_identification",
        desc="Specific city or town location is identified",
        parent=loc_node,
        critical=True
    )

    # state_location: verify with provided sources
    state_loc_leaf = evaluator.add_leaf(
        id=f"{state_label_id}_state_location",
        desc=f"University must be located in {state_full_name}",
        parent=loc_node,
        critical=True
    )
    claim_state = (
        f"This webpage shows that '{institution_name}' is located in "
        f"{(city + ', ') if city else ''}{state_full_name}."
    )
    await evaluator.verify(
        claim=claim_state,
        node=state_loc_leaf,
        sources=location_urls,
        extra_prerequisites=[location_ref_node],
        additional_instruction=(
            "Confirm that the institution's physical location is in the specified U.S. state. "
            "If a city is given, also confirm the city matches. Accept minor naming variants (e.g., 'Philadelphia' vs. 'City of Philadelphia')."
        )
    )

    # ------------------------ Basketball Program ---------------------------- #
    program_node = evaluator.add_parallel(
        id=f"{state_label_id}_basketball_program",
        desc="Men's basketball program information and current status",
        parent=uni_node,
        critical=True
    )

    # program_reference: ensure program URLs present
    program_ref_node = evaluator.add_custom_node(
        result=len(program_urls) > 0,
        id=f"{state_label_id}_program_reference",
        desc="Reference URL supporting basketball program information",
        parent=program_node,
        critical=True
    )

    # division_status
    division_leaf = evaluator.add_leaf(
        id=f"{state_label_id}_division_status",
        desc="University competes in NCAA Division I",
        parent=program_node,
        critical=True
    )
    claim_division = (
        f"The men's basketball program of '{institution_name}' competes in NCAA Division I."
    )
    await evaluator.verify(
        claim=claim_division,
        node=division_leaf,
        sources=program_urls,
        extra_prerequisites=[program_ref_node],
        additional_instruction=(
            "If the page explicitly states NCAA Division I, that suffices. "
            "If not explicit, membership in a recognized Division I conference (e.g., ACC, Big Ten, Big East, AAC, A-10, Ivy League, etc.) "
            "also suffices to conclude Division I."
        )
    )

    # active_program
    active_leaf = evaluator.add_leaf(
        id=f"{state_label_id}_active_program",
        desc="Has an active men's basketball program",
        parent=program_node,
        critical=True
    )
    claim_active = (
        f"The university '{institution_name}' currently fields an active men's basketball program (team)."
    )
    await evaluator.verify(
        claim=claim_active,
        node=active_leaf,
        sources=program_urls,
        extra_prerequisites=[program_ref_node],
        additional_instruction=(
            "Look for evidence such as a current season page, roster, schedule, or 'Men's Basketball' team page. "
            "If the page clearly indicates an ongoing or upcoming season, consider it active."
        )
    )

    # conference_affiliation
    conf_leaf = evaluator.add_leaf(
        id=f"{state_label_id}_conference_affiliation",
        desc="Current athletic conference affiliation is specified",
        parent=program_node,
        critical=True
    )
    claim_conf = (
        f"The current athletic conference affiliation for the men's basketball team is '{conference}'."
        if conference else
        "The page specifies the current athletic conference affiliation for the men's basketball team."
    )
    await evaluator.verify(
        claim=claim_conf,
        node=conf_leaf,
        sources=program_urls,
        extra_prerequisites=[program_ref_node],
        additional_instruction=(
            "Confirm the conference (e.g., 'BIG EAST' vs 'Big East' should be considered equivalent). "
            "If multiple conferences are discussed, focus on the current affiliation."
        )
    )

    # current_head_coach (sub-branch)
    coach_node = evaluator.add_parallel(
        id=f"{state_label_id}_current_head_coach",
        desc="Current head coach name and tenure information",
        parent=program_node,
        critical=True
    )

    # coach_reference: ensure coach URLs present
    coach_ref_node = evaluator.add_custom_node(
        result=len(coach_urls) > 0,
        id=f"{state_label_id}_coach_reference",
        desc="Reference URL supporting coach information",
        parent=coach_node,
        critical=True
    )

    # coach_name
    coach_name_leaf = evaluator.add_leaf(
        id=f"{state_label_id}_coach_name",
        desc="Full name of current head coach",
        parent=coach_node,
        critical=True
    )
    claim_coach_name = (
        f"The current head coach of the men's basketball team is '{head_coach_name}'."
        if head_coach_name else
        "The page identifies the current head coach of the men's basketball team."
    )
    await evaluator.verify(
        claim=claim_coach_name,
        node=coach_name_leaf,
        sources=_combine_sources(coach_urls, program_urls),
        extra_prerequisites=[coach_ref_node],
        additional_instruction=(
            "Look for explicit 'Head Coach' designation. Allow minor name variants (e.g., middle initials)."
        )
    )

    # coaching_tenure
    coach_tenure_leaf = evaluator.add_leaf(
        id=f"{state_label_id}_coaching_tenure",
        desc="Years of service or season information at current position",
        parent=coach_node,
        critical=True
    )
    claim_tenure = (
        f"The head coach's tenure information is reported as '{head_coach_tenure}'."
        if head_coach_tenure else
        "The page provides the head coach's tenure information (e.g., since what year or which season)."
    )
    await evaluator.verify(
        claim=claim_tenure,
        node=coach_tenure_leaf,
        sources=_combine_sources(coach_urls, program_urls),
        extra_prerequisites=[coach_ref_node],
        additional_instruction=(
            "Accept equivalent phrasings (e.g., 'since 2022' vs. '2022–present', '3rd season' vs. 'third season')."
        )
    )

    # ------------------------ Facility Information -------------------------- #
    facility_node = evaluator.add_parallel(
        id=f"{state_label_id}_facility_information",
        desc="Home basketball arena and facility specifications",
        parent=uni_node,
        critical=True
    )

    # facility_reference: ensure facility URLs present
    facility_ref_node = evaluator.add_custom_node(
        result=len(facility_urls) > 0,
        id=f"{state_label_id}_facility_reference",
        desc="Reference URL supporting facility information",
        parent=facility_node,
        critical=True
    )

    # arena_name
    arena_leaf = evaluator.add_leaf(
        id=f"{state_label_id}_arena_name",
        desc="Official name of home basketball arena or primary athletic facility",
        parent=facility_node,
        critical=True
    )
    claim_arena = (
        f"The primary home basketball arena/facility is called '{facility_name}'."
        if facility_name else
        "The page identifies the official name of the primary home basketball arena or facility."
    )
    await evaluator.verify(
        claim=claim_arena,
        node=arena_leaf,
        sources=facility_urls,
        extra_prerequisites=[facility_ref_node],
        additional_instruction=(
            "If multiple venues are mentioned, accept the primary/most frequently used arena for home games."
        )
    )

    # seating_capacity
    capacity_leaf = evaluator.add_leaf(
        id=f"{state_label_id}_seating_capacity",
        desc="Arena seating capacity is specified with a specific number",
        parent=facility_node,
        critical=True
    )
    claim_capacity = (
        f"The seating capacity of the home basketball arena is {seating_capacity}."
        if seating_capacity else
        "The page states a specific seating capacity (a numeric value) for the home basketball arena."
    )
    await evaluator.verify(
        claim=claim_capacity,
        node=capacity_leaf,
        sources=facility_urls,
        extra_prerequisites=[facility_ref_node],
        additional_instruction=(
            "Accept reasonable numeric formatting (commas, approximate signs). If a capacity range is shown, "
            "the stated capacity should match the commonly cited figure."
        )
    )

    # ------------------------ Academic Support ------------------------------ #
    support_node = evaluator.add_parallel(
        id=f"{state_label_id}_academic_support",
        desc="Student-athlete academic support services",
        parent=uni_node,
        critical=True
    )

    # support_reference: ensure academic support URLs present
    support_ref_node = evaluator.add_custom_node(
        result=len(academic_support_urls) > 0,
        id=f"{state_label_id}_support_reference",
        desc="Reference URL supporting academic support information",
        parent=support_node,
        critical=True
    )

    # support_program_exists
    support_exists_leaf = evaluator.add_leaf(
        id=f"{state_label_id}_support_program_exists",
        desc="University provides documented academic support services for student-athletes",
        parent=support_node,
        critical=True
    )
    claim_support = (
        f"The institution provides academic support services for student-athletes (e.g., advising, tutoring, learning center). "
        f"Evidence is documented on the referenced pages."
    )
    await evaluator.verify(
        claim=claim_support,
        node=support_exists_leaf,
        sources=academic_support_urls,
        extra_prerequisites=[support_ref_node],
        additional_instruction=(
            "Look for specific services such as 'Academic Support', 'Student-Athlete Services', 'Academic Center', "
            "'tutoring', 'study hall', or similar. The page should clearly pertain to student-athletes."
        )
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
    model: str = "o4-mini",
) -> Dict:
    """
    Evaluate an answer for the NCAA Division I universities (PA, NJ, CT) task.
    """
    evaluator = Evaluator()
    # Note: The provided rubric marks root as critical. However, the verification framework enforces that
    # critical parents must have all-critical children. Since state group nodes are non-critical to allow
    # partial credit across states, we initialize root as non-critical (default) to avoid structural conflict.
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

    # Extract structured information for the three states
    extracted = await evaluator.extract(
        prompt=prompt_extract_universities(),
        template_class=UniversitiesExtraction,
        extraction_name="universities_by_state",
    )

    # Build top-level nodes for each state (parallel, non-critical)
    pa_node = evaluator.add_parallel(
        id="university_1_pennsylvania",
        desc="NCAA Division I university in Pennsylvania meeting all specified criteria",
        parent=root,
        critical=False
    )
    nj_node = evaluator.add_parallel(
        id="university_2_new_jersey",
        desc="NCAA Division I university in New Jersey meeting all specified criteria",
        parent=root,
        critical=False
    )
    ct_node = evaluator.add_parallel(
        id="university_3_connecticut",
        desc="NCAA Division I university in Connecticut meeting all specified criteria",
        parent=root,
        critical=False
    )

    # Verify each university subtree
    await verify_university(
        evaluator=evaluator,
        parent_node=pa_node,
        info=extracted.pennsylvania,
        state_label_id="pennsylvania",
        state_full_name="Pennsylvania",
    )
    await verify_university(
        evaluator=evaluator,
        parent_node=nj_node,
        info=extracted.new_jersey,
        state_label_id="new_jersey",
        state_full_name="New Jersey",
    )
    await verify_university(
        evaluator=evaluator,
        parent_node=ct_node,
        info=extracted.connecticut,
        state_label_id="connecticut",
        state_full_name="Connecticut",
    )

    # Return structured summary
    return evaluator.get_summary()