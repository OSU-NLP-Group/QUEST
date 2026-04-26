import asyncio
import logging
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.verification_tree import AggregationStrategy

# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "sec_universities_4"
TASK_DESCRIPTION = """
Identify four (4) distinct public universities that satisfy ALL of the following criteria simultaneously:

Geographic & Conference Requirements:
- Must be current member institutions of the Southeastern Conference (SEC) as of 2024
- Must be located in one of the following five states: Georgia, Florida, Alabama, Louisiana, or Texas
- Must be a public state university (not private)

Enrollment Requirements:
- Must have total undergraduate enrollment exceeding 30,000 students

Athletic Program Requirements:
- Must sponsor an NCAA Division I Football Bowl Subdivision (FBS) football program
- Must meet NCAA Division I sports sponsorship requirements:
  * Sponsor a minimum of 16 varsity intercollegiate teams (including football)
  * Include at least 6 men's or coeducational varsity teams
  * Include at least 8 all-female varsity teams

Governance Requirements:
- Must be governed by a Board of Trustees or Board of Regents (or equivalent state governing board)

Academic Classification Requirements:
- Must be classified as either a state flagship university OR a land-grant university (or both)
- Must offer doctoral degree programs

For each of the four universities identified, the answer should provide:
1. Official university name
2. State location
3. Current undergraduate enrollment figure
4. Total number of varsity athletic teams sponsored
5. Breakdown of men's/coeducational and women's teams
6. Name of the governing board
7. University classification (flagship, land-grant, or both)
8. URL references that verify each of the above requirements
"""

ALLOWED_STATES = {"Georgia", "Florida", "Alabama", "Louisiana", "Texas"}

# --------------------------------------------------------------------------- #
# Extraction Models                                                           #
# --------------------------------------------------------------------------- #
class UniversitySources(BaseModel):
    sec_membership: List[str] = Field(default_factory=list)
    fbs_football: List[str] = Field(default_factory=list)
    sports_sponsorship: List[str] = Field(default_factory=list)
    enrollment: List[str] = Field(default_factory=list)
    governance: List[str] = Field(default_factory=list)
    academic: List[str] = Field(default_factory=list)
    general: List[str] = Field(default_factory=list)


class UniversityItem(BaseModel):
    official_name: Optional[str] = None
    state: Optional[str] = None

    undergraduate_enrollment: Optional[str] = None
    total_varsity_teams: Optional[str] = None
    men_coed_teams: Optional[str] = None
    women_teams: Optional[str] = None

    governing_board: Optional[str] = None
    classification: Optional[str] = None  # e.g., "flagship", "land-grant", "both"
    doctoral_programs: Optional[str] = None  # e.g., "yes", "offers PhD/EdD", etc.

    sources: UniversitySources = Field(default_factory=UniversitySources)


class UniversitiesExtraction(BaseModel):
    universities: List[UniversityItem] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction Prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_universities() -> str:
    return """
    Extract up to four (4) universities presented in the answer and structure the information exactly as requested below.
    For each university, return:
    - official_name: The official full name of the university exactly as written in the answer
    - state: The U.S. state where the main campus is located (e.g., "Florida", "Georgia")
    - undergraduate_enrollment: The total undergraduate enrollment figure or phrasing as presented (string)
    - total_varsity_teams: The total number of varsity intercollegiate teams (string)
    - men_coed_teams: The number of men's or coeducational varsity teams (string)
    - women_teams: The number of women's varsity teams (string)
    - governing_board: The name of the governing board (e.g., "Board of Trustees", "Board of Regents of ...")
    - classification: The classification string as claimed in the answer, e.g., "flagship", "land-grant", "both", or any phrase indicating flagship/land-grant status
    - doctoral_programs: A short text indicating whether doctoral programs are offered (e.g., "yes", "offers doctoral programs", "PhD programs offered")
    - sources: Group URLs that the answer cites for each requirement into the following arrays. Only include URLs explicitly present in the answer.
        * sec_membership: URLs confirming SEC membership (as of 2024)
        * fbs_football: URLs confirming an NCAA Division I FBS football program
        * sports_sponsorship: URLs confirming varsity team counts and breakdowns
        * enrollment: URLs confirming undergraduate enrollment figures or thresholds
        * governance: URLs confirming governance board and structure
        * academic: URLs confirming flagship/land-grant classification and that doctoral programs are offered
        * general: any other relevant URLs cited for the university

    Rules:
    - Do not fabricate any information. If a specific field is not provided in the answer, set it to null (or an empty list for URLs).
    - Preserve numbers as strings exactly as they appear; do not convert to numeric values.
    - Only include URLs that are explicitly shown or linked in the answer. Accept plain URLs or markdown links and extract the actual URL.
    - Extract up to the first 4 universities mentioned in the answer, in order.
    
    Return a JSON object: { "universities": [ { ... }, ... ] }.
    """


# --------------------------------------------------------------------------- #
# Helper Utilities                                                            #
# --------------------------------------------------------------------------- #
def ordinal_word(idx: int) -> str:
    return ["First", "Second", "Third", "Fourth"][idx] if 0 <= idx < 4 else f"University {idx+1}"


def gather_sources(uni: UniversityItem, categories: List[str]) -> List[str]:
    out: List[str] = []
    for cat in categories:
        arr = getattr(uni.sources, cat, [])
        if isinstance(arr, list):
            for u in arr:
                if u and u not in out:
                    out.append(u)
    return out


def any_sources(uni: UniversityItem) -> List[str]:
    return gather_sources(
        uni,
        ["sec_membership", "fbs_football", "sports_sponsorship", "enrollment", "governance", "academic", "general"],
    )


# --------------------------------------------------------------------------- #
# Verification builder for one university                                     #
# --------------------------------------------------------------------------- #
async def verify_university(evaluator: Evaluator, parent_node, uni: UniversityItem, idx: int) -> None:
    u_id = f"university_{idx + 1}"
    u_node = evaluator.add_parallel(
        id=u_id,
        desc=f"{ordinal_word(idx)} qualifying university identification and verification",
        parent=parent_node,
        critical=False,
    )

    # ------------------- Athletic program requirements ------------------- #
    ath_node = evaluator.add_parallel(
        id=f"athletic_requirements_u{idx + 1}",
        desc=f"Athletic program requirements for University {idx + 1}",
        parent=u_node,
        critical=True,
    )

    # SEC membership (as of 2024)
    sec_leaf = evaluator.add_leaf(
        id=f"sec_membership_u{idx + 1}",
        desc=f"University {idx + 1} is a current SEC member as of 2024",
        parent=ath_node,
        critical=True,
    )
    sec_sources = gather_sources(uni, ["sec_membership", "sports_sponsorship", "general", "academic"])
    uni_name = uni.official_name or "the university in the provided sources"
    sec_claim = f"As of 2024, {uni_name} is a current member institution of the Southeastern Conference (SEC)."
    if sec_sources:
        await evaluator.verify(
            claim=sec_claim,
            node=sec_leaf,
            sources=sec_sources,
            additional_instruction="Verify SEC conference membership status as of 2024 on official conference pages or the university athletics site.",
        )

    # FBS football
    fbs_leaf = evaluator.add_leaf(
        id=f"fbs_football_u{idx + 1}",
        desc=f"University {idx + 1} sponsors NCAA Division I FBS football",
        parent=ath_node,
        critical=True,
    )
    fbs_sources = gather_sources(uni, ["fbs_football", "sports_sponsorship", "general"])
    fbs_claim = f"{uni_name} sponsors an NCAA Division I Football Bowl Subdivision (FBS) football program."
    if fbs_sources:
        await evaluator.verify(
            claim=fbs_claim,
            node=fbs_leaf,
            sources=fbs_sources,
            additional_instruction="Look for 'FBS', 'Football Bowl Subdivision', or 'NCAA Division I FBS' on official athletics/NCAA/EADA pages.",
        )

    # Sports sponsorship details
    sports_node = evaluator.add_parallel(
        id=f"sports_sponsorship_u{idx + 1}",
        desc="Verification of NCAA Division I sports sponsorship requirements",
        parent=ath_node,
        critical=True,
    )
    sports_srcs = gather_sources(uni, ["sports_sponsorship"])

    # Presence of sports verification URL (make critical to satisfy framework constraints)
    sports_url_presence = evaluator.add_custom_node(
        result=len(sports_srcs) > 0,
        id=f"sports_verification_url_u{idx + 1}",
        desc="URL reference confirming sports sponsorship details",
        parent=sports_node,
        critical=True,
    )

    # Minimum 16 varsity teams
    min16_leaf = evaluator.add_leaf(
        id=f"minimum_16_teams_u{idx + 1}",
        desc=f"University {idx + 1} sponsors at least 16 varsity teams",
        parent=sports_node,
        critical=True,
    )
    if sports_srcs:
        min16_claim = f"The athletics department of {uni_name} sponsors at least 16 varsity intercollegiate teams."
        await evaluator.verify(
            claim=min16_claim,
            node=min16_leaf,
            sources=sports_srcs,
            additional_instruction="Use official athletics/EADA/NCAA pages that list the number of varsity sports or enumerate them. 'Varsity sports/programs' wording is acceptable.",
            extra_prerequisites=[sports_url_presence],
        )

    # At least 6 men's or coed teams
    men6_leaf = evaluator.add_leaf(
        id=f"mens_teams_requirement_u{idx + 1}",
        desc=f"University {idx + 1} has at least 6 men's/coeducational teams",
        parent=sports_node,
        critical=True,
    )
    if sports_srcs:
        men6_claim = f"The athletics program at {uni_name} includes at least 6 men's or coeducational varsity teams."
        await evaluator.verify(
            claim=men6_claim,
            node=men6_leaf,
            sources=sports_srcs,
            additional_instruction="Pages that break down teams by men/coed vs. women are acceptable; if coed teams are listed, count them toward the 6 threshold.",
            extra_prerequisites=[sports_url_presence],
        )

    # At least 8 women's teams
    women8_leaf = evaluator.add_leaf(
        id=f"womens_teams_requirement_u{idx + 1}",
        desc=f"University {idx + 1} has at least 8 all-female teams",
        parent=sports_node,
        critical=True,
    )
    if sports_srcs:
        women8_claim = f"The athletics program at {uni_name} includes at least 8 women's varsity teams."
        await evaluator.verify(
            claim=women8_claim,
            node=women8_leaf,
            sources=sports_srcs,
            additional_instruction="Use official listings or counts of women's varsity sports on athletics/EADA/NCAA pages.",
            extra_prerequisites=[sports_url_presence],
        )

    # ---------------- Geographic location and enrollment requirements ---- #
    geo_node = evaluator.add_parallel(
        id=f"geographic_enrollment_u{idx + 1}",
        desc=f"Geographic location and enrollment requirements for University {idx + 1}",
        parent=u_node,
        critical=True,
    )

    # Location in allowed states
    state_leaf = evaluator.add_leaf(
        id=f"state_location_u{idx + 1}",
        desc=f"University {idx + 1} is located in Georgia, Florida, Alabama, Louisiana, or Texas",
        parent=geo_node,
        critical=True,
    )
    state_sources = gather_sources(uni, ["sec_membership", "academic", "general", "governance"])
    state_name = uni.state or "one of Georgia, Florida, Alabama, Louisiana, or Texas"
    state_claim = f"{uni_name} is located in {state_name}, which is one of Georgia, Florida, Alabama, Louisiana, or Texas."
    if state_sources:
        await evaluator.verify(
            claim=state_claim,
            node=state_leaf,
            sources=state_sources,
            additional_instruction="Confirm the state of the main campus on official university pages, state system pages, or other authoritative sources.",
        )

    # Public institution
    public_leaf = evaluator.add_leaf(
        id=f"public_institution_u{idx + 1}",
        desc=f"University {idx + 1} is a public university",
        parent=geo_node,
        critical=True,
    )
    public_sources = gather_sources(uni, ["academic", "governance", "general"])
    public_claim = f"{uni_name} is a public state university (not private)."
    if public_sources:
        await evaluator.verify(
            claim=public_claim,
            node=public_leaf,
            sources=public_sources,
            additional_instruction="Look for explicit 'public university', governance by state board, or membership in a state university system.",
        )

    # Presence of enrollment verification URL (critical)
    enroll_srcs = gather_sources(uni, ["enrollment"])
    enroll_url_presence = evaluator.add_custom_node(
        result=len(enroll_srcs) > 0,
        id=f"enrollment_verification_url_u{idx + 1}",
        desc="URL reference confirming enrollment data",
        parent=geo_node,
        critical=True,
    )

    # Enrollment exceeds 30,000 undergraduates
    enroll_leaf = evaluator.add_leaf(
        id=f"enrollment_threshold_u{idx + 1}",
        desc=f"University {idx + 1} has undergraduate enrollment exceeding 30,000",
        parent=geo_node,
        critical=True,
    )
    if enroll_srcs:
        if uni.undergraduate_enrollment:
            enroll_claim = f"The undergraduate enrollment at {uni_name} is {uni.undergraduate_enrollment}, which exceeds 30,000 students."
        else:
            enroll_claim = f"The undergraduate enrollment at {uni_name} exceeds 30,000 students."
        await evaluator.verify(
            claim=enroll_claim,
            node=enroll_leaf,
            sources=enroll_srcs,
            additional_instruction="Use official fact books, Common Data Set pages, or institutional research pages that report undergraduate totals.",
            extra_prerequisites=[enroll_url_presence],
        )

    # --------------------------- Governance structure -------------------- #
    gov_node = evaluator.add_parallel(
        id=f"governance_structure_u{idx + 1}",
        desc=f"Governance structure requirements for University {idx + 1}",
        parent=u_node,
        critical=True,
    )

    gov_srcs = gather_sources(uni, ["governance"])
    gov_url_presence = evaluator.add_custom_node(
        result=len(gov_srcs) > 0,
        id=f"governance_verification_url_u{idx + 1}",
        desc="URL reference confirming governance structure",
        parent=gov_node,
        critical=True,
    )

    gov_leaf = evaluator.add_leaf(
        id=f"board_governance_u{idx + 1}",
        desc=f"University {idx + 1} is governed by Board of Trustees or Board of Regents",
        parent=gov_node,
        critical=True,
    )
    if gov_srcs:
        board_name = uni.governing_board or "a Board of Trustees or Board of Regents (or equivalent state governing board)"
        gov_claim = f"{uni_name} is governed by {board_name} (a state governing board such as a Board of Trustees or Board of Regents)."
        await evaluator.verify(
            claim=gov_claim,
            node=gov_leaf,
            sources=gov_srcs,
            additional_instruction="Confirm the specific governing board on official governance pages, bylaws, state system pages, or university charters.",
            extra_prerequisites=[gov_url_presence],
        )

    # ------------------------- Academic classification ------------------- #
    acad_node = evaluator.add_parallel(
        id=f"academic_status_u{idx + 1}",
        desc=f"Academic classification and program requirements for University {idx + 1}",
        parent=u_node,
        critical=True,
    )

    acad_srcs = gather_sources(uni, ["academic"])
    acad_url_presence = evaluator.add_custom_node(
        result=len(acad_srcs) > 0,
        id=f"academic_verification_url_u{idx + 1}",
        desc="URL reference confirming academic status and programs",
        parent=acad_node,
        critical=True,
    )

    # Flagship or land-grant classification
    class_leaf = evaluator.add_leaf(
        id=f"flagship_landgrant_u{idx + 1}",
        desc=f"University {idx + 1} is classified as state flagship or land-grant university",
        parent=acad_node,
        critical=True,
    )
    if acad_srcs:
        if uni.classification:
            class_claim = f"{uni_name} is classified as {uni.classification}, which satisfies the requirement of being a state flagship or a land-grant university (or both)."
        else:
            class_claim = f"{uni_name} is classified as either a state flagship university or a land-grant university."
        await evaluator.verify(
            claim=class_claim,
            node=class_leaf,
            sources=acad_srcs,
            additional_instruction="Accept official university/system pages, state designations, or authoritative references noting flagship or land-grant status.",
            extra_prerequisites=[acad_url_presence],
        )

    # Offers doctoral programs
    doc_leaf = evaluator.add_leaf(
        id=f"doctoral_programs_u{idx + 1}",
        desc=f"University {idx + 1} offers doctoral degree programs",
        parent=acad_node,
        critical=True,
    )
    if acad_srcs:
        if uni.doctoral_programs:
            doc_claim = f"{uni_name} offers doctoral degree programs ({uni.doctoral_programs})."
        else:
            doc_claim = f"{uni_name} offers doctoral degree programs."
        await evaluator.verify(
            claim=doc_claim,
            node=doc_leaf,
            sources=acad_srcs,
            additional_instruction="Use graduate school/program catalogs or official degree listings confirming doctoral (e.g., PhD, EdD, etc.).",
            extra_prerequisites=[acad_url_presence],
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

    # Extract structured university data from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_universities(),
        template_class=UniversitiesExtraction,
        extraction_name="universities_extraction",
    )

    # Keep exactly four entries (pad with empty if needed)
    universities: List[UniversityItem] = (extracted.universities or [])[:4]
    while len(universities) < 4:
        universities.append(UniversityItem())

    # Optional info for the summary
    evaluator.add_custom_info(
        info={"allowed_states": sorted(list(ALLOWED_STATES))},
        info_type="constraints",
        info_name="allowed_states_for_task",
    )

    # Build verification subtrees for each of the four universities
    for i in range(4):
        await verify_university(evaluator, root, universities[i], i)

    # Return full evaluation summary
    return evaluator.get_summary()