import asyncio
import logging
import re
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator, AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "ncaa_fbs_coach_progression_rapid_advancement"
TASK_DESCRIPTION = (
    "Identify three NCAA Division I FBS football coaches who exemplify rapid career advancement through "
    "the traditional coaching pathway. Each coach must meet ALL of the following criteria:\n"
    "1) Began collegiate coaching as a Graduate Assistant (GA) at an NCAA institution; "
    "2) Served in at least one position coach role before becoming a coordinator; "
    "3) Achieved first DC/OC role within 6 years of starting as GA; "
    "4) Have held a DC/OC role at an NCAA Division I program; "
    "5) Currently employed at an NCAA Division I FBS program (as of March 2026); "
    "6) Have been employed at 3+ different universities; "
    "7) Hold at least a bachelor's degree.\n"
    "For each coach: provide name, current role/institution, education, complete career timeline "
    "(GA, position coach roles, first coordinator, current role), GA start year, first coordinator year, "
    "all universities employed, and URLs supporting the information."
)


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class PositionRole(BaseModel):
    role: Optional[str] = None
    institution: Optional[str] = None
    years: Optional[str] = None  # free-form, e.g., "2018–2019" or "2018"
    urls: List[str] = Field(default_factory=list)


class CoachItem(BaseModel):
    # Identity and current role
    name: Optional[str] = None
    current_position_title: Optional[str] = None
    current_institution: Optional[str] = None
    current_urls: List[str] = Field(default_factory=list)

    # Education
    education_degree: Optional[str] = None  # e.g., "B.S. in Exercise Science"
    education_institution: Optional[str] = None
    education_urls: List[str] = Field(default_factory=list)

    # Career progression essentials
    ga_start_institution: Optional[str] = None
    ga_start_year: Optional[str] = None
    ga_urls: List[str] = Field(default_factory=list)

    # Position coach roles (before coordinator)
    position_roles: List[PositionRole] = Field(default_factory=list)
    position_roles_urls: List[str] = Field(default_factory=list)

    # First coordinator milestone
    first_coordinator_title: Optional[str] = None  # "Defensive Coordinator", "Offensive Coordinator", etc.
    first_coordinator_institution: Optional[str] = None
    first_coordinator_year: Optional[str] = None
    first_coordinator_urls: List[str] = Field(default_factory=list)

    # Coverage and institutional diversity
    universities_employed: List[str] = Field(default_factory=list)

    # General/career timeline URLs
    career_urls: List[str] = Field(default_factory=list)


class CoachesExtraction(BaseModel):
    coaches: List[CoachItem] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_coaches() -> str:
    return """
Extract up to the first 3 coaches described in the answer that the author claims meet all the criteria. For each coach, return the following fields exactly as presented in the answer. Do not invent or infer beyond the answer content.

For each coach, extract:
- name
- current_position_title
- current_institution
- current_urls: all URLs cited to support current role at an NCAA Division I FBS program
- education_degree: the degree credential (e.g., "B.S. in X", "Bachelor's in Y")
- education_institution
- education_urls: URLs that support the education credential(s)
- ga_start_institution: the school where they first served as a Graduate Assistant (GA)
- ga_start_year: the year they first began as a GA (4-digit if present, otherwise a string as given)
- ga_urls: URLs that support the GA start (may overlap with career_urls)
- position_roles: an array; each object should include:
    - role (e.g., "Linebackers Coach", "Defensive Line Coach", "Quarterbacks Coach")
    - institution
    - years (free-form, as stated)
    - urls: URLs supporting this role
- position_roles_urls: additional URLs (if any) broadly supporting position-coach roles before coordinator
- first_coordinator_title (e.g., "Defensive Coordinator", "Offensive Coordinator", "Co-Defensive Coordinator")
- first_coordinator_institution
- first_coordinator_year
- first_coordinator_urls: URLs that support this first coordinator milestone
- universities_employed: list of all distinct universities they have been employed at in their collegiate coaching career
- career_urls: all URLs that document the overall career timeline (GA, position roles, first coordinator, etc.)

Rules:
- Only include URLs that are explicitly mentioned in the answer. If none are given for a field, use an empty list.
- Use plain strings; do not force numeric types.
- If any field is missing in the answer, set it to null or an empty array as appropriate.
- The 'coaches' array should contain at most 3 entries based on the order they appear in the answer.
"""


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _parse_year(text: Optional[str]) -> Optional[int]:
    if not text:
        return None
    m = re.search(r"(19|20)\d{2}", text)
    if not m:
        return None
    try:
        return int(m.group(0))
    except Exception:
        return None


def _dedup_urls(*url_lists: List[str]) -> List[str]:
    seen = set()
    result: List[str] = []
    for lst in url_lists:
        for u in lst:
            if not u:
                continue
            u_norm = u.strip()
            if u_norm and u_norm not in seen:
                seen.add(u_norm)
                result.append(u_norm)
    return result


def _safe_name(n: Optional[str]) -> str:
    return n or "the coach"


def _format_universities(unis: List[str]) -> str:
    cleaned = [u.strip() for u in unis if u and u.strip()]
    if not cleaned:
        return ""
    return ", ".join(cleaned)


# --------------------------------------------------------------------------- #
# Verification logic                                                          #
# --------------------------------------------------------------------------- #
async def verify_single_coach(
    evaluator: Evaluator,
    parent_node,
    coach: CoachItem,
    coach_idx_1based: int,
) -> None:
    """
    Build the verification subtree and perform checks for a single coach.
    The node IDs follow the rubric's naming pattern (Coach_1, Coach_2, Coach_3, etc.).
    """
    coach_node = evaluator.add_parallel(
        id=f"Coach_{coach_idx_1based}",
        desc=f"Coach #{coach_idx_1based} meets all specified career progression and qualification criteria",
        parent=parent_node,
        critical=False,  # allow partial credit across coaches
    )

    # -------------------- Career Progression (Critical group) --------------------
    career_node = evaluator.add_parallel(
        id=f"Coach_{coach_idx_1based}_Career_Progression",
        desc="Verification of the coach's complete career progression through required stages",
        parent=coach_node,
        critical=True,
    )

    # Aggregate career-related sources
    career_sources = _dedup_urls(
        coach.career_urls,
        coach.ga_urls,
        coach.first_coordinator_urls,
        coach.position_roles_urls,
        *(r.urls for r in coach.position_roles),
    )

    # Career URL existence (Critical gate for other career checks)
    evaluator.add_custom_node(
        result=len(career_sources) > 0,
        id=f"Coach_{coach_idx_1based}_Career_URL",
        desc="Provide URL reference documenting the coach's career progression timeline including GA start, position coach roles, and coordinator achievement",
        parent=career_node,
        critical=True,
    )

    # GA start verification (Critical)
    ga_leaf = evaluator.add_leaf(
        id=f"Coach_{coach_idx_1based}_GA_Start",
        desc="Verify the coach began their collegiate coaching career as a Graduate Assistant at an NCAA institution",
        parent=career_node,
        critical=True,
    )
    ga_claim_parts = []
    if coach.ga_start_institution:
        ga_claim_parts.append(
            f"{_safe_name(coach.name)} began their collegiate coaching career as a Graduate Assistant at {coach.ga_start_institution}."
        )
    else:
        ga_claim_parts.append(
            f"{_safe_name(coach.name)} began their collegiate coaching career as a Graduate Assistant at an NCAA institution."
        )
    if coach.ga_start_year:
        ga_claim_parts.append(f"This occurred around {coach.ga_start_year}.")
    ga_claim_parts.append("The institution is an NCAA member.")
    ga_claim = " ".join(ga_claim_parts)

    await evaluator.verify(
        claim=ga_claim,
        node=ga_leaf,
        sources=career_sources,
        additional_instruction=(
            "Look for language such as 'graduate assistant' or 'GA' and whether it is stated or implied as the "
            "start of their collegiate coaching career. Treat NCAA Division I (FBS/FCS) membership as NCAA membership. "
            "If multiple pages are provided, collectively assess the earliest coaching role."
        ),
    )

    # Position coach role before coordinator (Critical)
    pos_leaf = evaluator.add_leaf(
        id=f"Coach_{coach_idx_1based}_Position_Coach",
        desc="Verify the coach served in at least one position coach role before becoming a coordinator",
        parent=career_node,
        critical=True,
    )
    example_role_txt = ""
    if coach.position_roles:
        pr0 = coach.position_roles[0]
        if pr0.role and pr0.institution:
            example_role_txt = f"For example, served as {pr0.role} at {pr0.institution}."
        elif pr0.role:
            example_role_txt = f"For example, served as {pr0.role}."
    pos_claim = (
        f"Before first becoming a coordinator in {coach.first_coordinator_year or 'the year indicated'}, "
        f"{_safe_name(coach.name)} served in at least one position coach role (e.g., linebackers, defensive line, safeties, quarterbacks). "
        f"{example_role_txt}"
    )

    await evaluator.verify(
        claim=pos_claim,
        node=pos_leaf,
        sources=career_sources,
        additional_instruction=(
            "Confirm at least one explicit position coach role (not GA, not coordinator) appears in their career "
            "timeline prior to the first coordinator appointment."
        ),
    )

    # Coordinator achievement timeline (Critical group: split into atomic leaves)
    coord_tl_node = evaluator.add_parallel(
        id=f"Coach_{coach_idx_1based}_Coordinator_Achievement_Timeline",
        desc="Verify first DC/OC occurred at NCAA Division I and within 6 years of GA start",
        parent=career_node,
        critical=True,
    )

    ga_year = _parse_year(coach.ga_start_year)
    coord_year = _parse_year(coach.first_coordinator_year)
    years_provided = ga_year is not None and coord_year is not None
    within_6 = years_provided and (coord_year - ga_year <= 6) and (coord_year - ga_year >= 0)

    evaluator.add_custom_node(
        result=years_provided,
        id=f"Coach_{coach_idx_1based}_Coord_Years_Provided",
        desc="Years for GA start and first coordinator role are provided (parseable 4-digit years)",
        parent=coord_tl_node,
        critical=True,
    )

    evaluator.add_custom_node(
        result=bool(within_6),
        id=f"Coach_{coach_idx_1based}_Coord_Within_6_Years",
        desc="First coordinator role achieved within 6 years of GA start",
        parent=coord_tl_node,
        critical=True,
    )

    coord_div1_leaf = evaluator.add_leaf(
        id=f"Coach_{coach_idx_1based}_First_Coordinator_Div1",
        desc="First coordinator role was at an NCAA Division I program",
        parent=coord_tl_node,
        critical=True,
    )
    coord_inst = coach.first_coordinator_institution or "the stated institution"
    coord_title = coach.first_coordinator_title or "a coordinator role"
    coord_year_txt = coach.first_coordinator_year or "the year indicated"
    coord_claim = (
        f"{_safe_name(coach.name)}'s first coordinator role ({coord_title}) at {coord_inst} in {coord_year_txt} "
        f"was at an NCAA Division I program (FBS or FCS)."
    )
    coord_sources = _dedup_urls(coach.first_coordinator_urls, career_sources)
    await evaluator.verify(
        claim=coord_claim,
        node=coord_div1_leaf,
        sources=coord_sources,
        additional_instruction=(
            "Verify that the first coordinator appointment (OC/DC, including 'co-' titles) occurred at an NCAA Division I "
            "program (FBS or FCS). Accept evidence from official bios, rosters, press releases, or reputable media. "
            "If a page clearly states the school's Division I status or commonly recognized FBS/FCS affiliation, accept it."
        ),
    )

    # -------------------- Current Qualifications (Critical group) --------------------
    qual_node = evaluator.add_parallel(
        id=f"Coach_{coach_idx_1based}_Current_Qualifications",
        desc="Verification of the coach's current professional qualifications and standing",
        parent=coach_node,
        critical=True,
    )

    # Current/Education/Institutional history URL presence as a critical gate bundle
    status_url_node = evaluator.add_parallel(
        id=f"Coach_{coach_idx_1based}_Current_Status_URL",
        desc="Provide URL reference documenting the coach's current position, educational background, and institutional history",
        parent=qual_node,
        critical=True,
    )

    # Leaves under Current_Status_URL (all critical)
    evaluator.add_custom_node(
        result=len(_dedup_urls(coach.current_urls)) > 0,
        id=f"Coach_{coach_idx_1based}_Current_URL_Provided",
        desc="Current position URL(s) provided",
        parent=status_url_node,
        critical=True,
    )
    evaluator.add_custom_node(
        result=len(_dedup_urls(coach.education_urls)) > 0,
        id=f"Coach_{coach_idx_1based}_Education_URL_Provided",
        desc="Education URL(s) provided",
        parent=status_url_node,
        critical=True,
    )
    evaluator.add_custom_node(
        result=len(career_sources) > 0,
        id=f"Coach_{coach_idx_1based}_Institution_History_URL_Provided",
        desc="Institutional history (career) URL(s) provided",
        parent=status_url_node,
        critical=True,
    )

    # Education verification (Critical)
    edu_leaf = evaluator.add_leaf(
        id=f"Coach_{coach_idx_1based}_Education",
        desc="Verify the coach holds at least a bachelor's degree from an accredited institution",
        parent=qual_node,
        critical=True,
    )
    edu_inst = coach.education_institution or "the stated institution"
    edu_deg = coach.education_degree or "at least a bachelor's degree"
    edu_claim = (
        f"{_safe_name(coach.name)} holds at least a bachelor's degree (e.g., '{edu_deg}') from {edu_inst}."
    )
    await evaluator.verify(
        claim=edu_claim,
        node=edu_leaf,
        sources=_dedup_urls(coach.education_urls),
        additional_instruction=(
            "Accept bachelor's or higher degrees (e.g., BA, BS, B.S., B.A., master's, etc.). "
            "Verify from official bios, media guides, or reputable sources that the credential is stated."
        ),
    )

    # Current employment at NCAA Division I FBS as of March 2026 (Critical)
    curr_leaf = evaluator.add_leaf(
        id=f"Coach_{coach_idx_1based}_Current_Employment",
        desc="Verify the coach is currently employed at an NCAA Division I FBS program (as of March 2026)",
        parent=qual_node,
        critical=True,
    )
    curr_pos = coach.current_position_title or "a coaching role"
    curr_inst = coach.current_institution or "an NCAA Division I FBS program"
    curr_claim = (
        f"As of March 2026, {_safe_name(coach.name)} is employed as {curr_pos} at {curr_inst}, "
        "which competes in NCAA Division I FBS."
    )
    await evaluator.verify(
        claim=curr_claim,
        node=curr_leaf,
        sources=_dedup_urls(coach.current_urls),
        additional_instruction=(
            "Prefer official team/staff bios or athletics roster pages for 2025–2026 seasons. "
            "If the page reflects the 2025 season staff and there is no contrary evidence of departure before March 2026, "
            "treat it as current. Confirm FBS affiliation if mentioned."
        ),
    )

    # Institution diversity (Critical group)
    inst_div_node = evaluator.add_parallel(
        id=f"Coach_{coach_idx_1based}_Institution_Diversity",
        desc="Verify the coach has been employed at at least 3 different universities during their coaching career",
        parent=qual_node,
        critical=True,
    )

    unique_unis = sorted(set([u.strip() for u in coach.universities_employed if u and u.strip()]))
    evaluator.add_custom_node(
        result=len(unique_unis) >= 3,
        id=f"Coach_{coach_idx_1based}_Institution_Diversity_Count",
        desc="At least 3 distinct universities are listed for the coach's career",
        parent=inst_div_node,
        critical=True,
    )

    inst_div_leaf = evaluator.add_leaf(
        id=f"Coach_{coach_idx_1based}_Institution_Diversity_Supported",
        desc="The 3+ universities of employment are supported by cited sources",
        parent=inst_div_node,
        critical=True,
    )
    inst_div_claim = (
        f"{_safe_name(coach.name)} has been employed by at least three distinct universities in their collegiate coaching career, "
        f"including: {_format_universities(unique_unis)}."
    )
    await evaluator.verify(
        claim=inst_div_claim,
        node=inst_div_leaf,
        sources=career_sources,
        additional_instruction=(
            "Review the career timeline sources to confirm that the named universities reflect distinct institutions of employment. "
            "Minor naming variations (e.g., 'University of X' vs 'X University') should be considered equivalent."
        ),
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
    Evaluate an answer for NCAA Division I FBS football coaching rapid advancement criteria.
    """
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # independent evaluation of each coach
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

    # Extract structured information about coaches
    extracted = await evaluator.extract(
        prompt=prompt_extract_coaches(),
        template_class=CoachesExtraction,
        extraction_name="coaches_extraction",
    )

    # Select at most the first 3 coaches; pad with empty entries if fewer
    coaches = list(extracted.coaches[:3])
    while len(coaches) < 3:
        coaches.append(CoachItem())

    # Build verification subtree for each coach
    # The rubric's "Task_Root" is represented by the evaluator's root node here.
    verify_tasks = []
    for idx, coach in enumerate(coaches, start=1):
        verify_tasks.append(verify_single_coach(evaluator, root, coach, idx))

    await asyncio.gather(*verify_tasks)

    return evaluator.get_summary()