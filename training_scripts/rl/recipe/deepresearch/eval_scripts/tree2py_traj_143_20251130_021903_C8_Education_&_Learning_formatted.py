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
TASK_ID = "hs_grad_states_ncaa_universities_2024"
TASK_DESCRIPTION = (
    "As a high school guidance counselor preparing materials for student-athletes, identify three U.S. states that both "
    "(1) require standardized exit exams for high school graduation as of 2024, and (2) require a minimum of 22 credits or more for high school graduation. "
    "For each of the three states you identify, provide the name of one NCAA Division I university located in that state that has an NCAA Division I football program. "
    "For each university, provide the following information: (a) the specific name of the regional accrediting body that accredits the university, "
    "(b) the minimum core GPA required for NCAA Division I eligibility, (c) the total number of core courses required for NCAA Division I eligibility, "
    "and (d) the number of years/credits of English required within those NCAA Division I core courses."
)


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class UniversityDetails(BaseModel):
    name: Optional[str] = None

    # General URLs the answer associates with the university (homepage, Wikipedia, athletics, etc.)
    website_urls: List[str] = Field(default_factory=list)

    # Location verification URLs (could overlap with website_urls)
    location_urls: List[str] = Field(default_factory=list)

    # URLs indicating NCAA Division I membership
    ncaa_division_urls: List[str] = Field(default_factory=list)

    # URLs indicating NCAA Division I football program/team
    football_urls: List[str] = Field(default_factory=list)

    # Accreditation details
    accreditor_name: Optional[str] = None
    accreditation_urls: List[str] = Field(default_factory=list)

    # NCAA initial eligibility details (as presented in the answer)
    ncaa_core_gpa: Optional[str] = None
    ncaa_core_courses_total: Optional[str] = None
    ncaa_english_requirement: Optional[str] = None
    ncaa_eligibility_urls: List[str] = Field(default_factory=list)


class StateEntry(BaseModel):
    state_name: Optional[str] = None

    # State requirement (1): standardized exit exam as of 2024
    exit_exam_requirement: Optional[str] = None   # e.g., "yes", "required", or a descriptive phrase
    exit_exam_urls: List[str] = Field(default_factory=list)

    # State requirement (2): minimum credits (22 or more)
    min_credits_requirement: Optional[str] = None  # e.g., "22", "23", "24", or a phrase like "24 credits"
    min_credits_urls: List[str] = Field(default_factory=list)

    # One qualifying in-state NCAA Division I university
    university: Optional[UniversityDetails] = None


class StatesExtraction(BaseModel):
    states: List[StateEntry] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_states() -> str:
    return """
    You will extract up to three state-and-university pairs exactly as presented in the answer text. Focus on the first three complete pairs if more than three are provided; if fewer than three are provided, extract whatever is available.

    For each state, extract:
    - state_name: the U.S. state's name
    - exit_exam_requirement: the answer's stated status/description regarding whether a standardized exit exam is required as of 2024 (e.g., "yes", "required", or a descriptive phrase explicitly indicating the requirement)
    - exit_exam_urls: all URLs the answer cites that support the claim about the state's exit exam requirement
    - min_credits_requirement: the minimum credits required to graduate high school in that state as presented in the answer (e.g., "22", "23", "24 credits", etc.)
    - min_credits_urls: all URLs the answer cites that support the claim about statewide minimum credits
    - university: an object for the NCAA Division I university in that state with the following:
        - name: the university's name
        - website_urls: any general URLs provided for the university (homepage, Wikipedia, athletics site, etc.)
        - location_urls: URLs supporting that the university is located in the identified state
        - ncaa_division_urls: URLs supporting that the university is an NCAA Division I member
        - football_urls: URLs supporting that the university has an NCAA Division I football program/team (FBS/FCS)
        - accreditor_name: the specific, named regional accreditor for the university (e.g., "SACSCOC", "HLC", "MSCHE", "NECHE", "WSCUC", "NWCCU")
        - accreditation_urls: URLs supporting the accreditation claim
        - ncaa_core_gpa: the minimum core GPA for NCAA Division I initial eligibility as stated in the answer (e.g., "2.3")
        - ncaa_core_courses_total: the total number of NCAA Division I core courses required as stated in the answer (e.g., "16")
        - ncaa_english_requirement: the number of years/credits of English required within those NCAA Division I core courses as stated in the answer (e.g., "4 years")
        - ncaa_eligibility_urls: URLs supporting the NCAA initial eligibility facts

    IMPORTANT:
    - Only extract URLs explicitly present in the answer (including markdown links). Do not invent or infer URLs.
    - If a field is missing or not stated, return null for single fields or an empty list for arrays.
    - Preserve the exact text formatting for numbers/phrases as they appear (e.g., keep "24 credits" instead of converting to a number).
    - Return a JSON object with a "states" array of at most 3 elements.
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _safe_sources(*lists: Optional[List[str]]) -> List[str]:
    """Flatten and deduplicate multiple URL lists."""
    merged: List[str] = []
    seen = set()
    for lst in lists:
        if not lst:
            continue
        for u in lst:
            if isinstance(u, str):
                u2 = u.strip()
                if u2 and u2 not in seen:
                    seen.add(u2)
                    merged.append(u2)
    return merged


def _has_text(s: Optional[str]) -> bool:
    return isinstance(s, str) and len(s.strip()) > 0


# --------------------------------------------------------------------------- #
# Verification logic per state                                                #
# --------------------------------------------------------------------------- #
async def verify_one_state(
    evaluator: Evaluator,
    parent,
    state_idx: int,
    state_entry: StateEntry,
) -> None:
    """
    Build and verify the tree for a single state with its associated university.
    Follows the rubric:
      - state_{i}: sequential
        - state_{i}_requirements: critical, parallel
          - state_{i}_exit_exam: critical leaf
          - state_{i}_min_credits: critical leaf
        - state_{i}_university: critical, parallel
          - state_{i}_university_in_state: critical leaf
          - state_{i}_university_ncaa_div1_member: critical leaf
          - state_{i}_university_div1_football: critical leaf
          - state_{i}_university_accreditation: critical leaf
          - state_{i}_ncaa_core_gpa: critical leaf
          - state_{i}_ncaa_core_courses: critical leaf
          - state_{i}_ncaa_english_requirement: critical leaf
    """
    sidx = state_idx + 1
    state_name = state_entry.state_name or ""

    # Create the state-level sequential node
    state_node = evaluator.add_sequential(
        id=f"state_{sidx}",
        desc=f"State {sidx} + associated university details",
        parent=parent,
        critical=False  # Non-critical at the top level to allow partial scoring across states
    )

    # -------- Requirements (critical, parallel) --------
    req_node = evaluator.add_parallel(
        id=f"state_{sidx}_requirements",
        desc=f"State {sidx} meets both graduation requirement constraints",
        parent=state_node,
        critical=True
    )

    # Leaf: Exit exam required as of 2024
    exit_exam_leaf = evaluator.add_leaf(
        id=f"state_{sidx}_exit_exam",
        desc="State requires standardized exit exams for high school graduation as of 2024",
        parent=req_node,
        critical=True
    )

    if not _has_text(state_entry.state_name):
        # Missing state name => cannot verify, mark as failed
        exit_exam_leaf.score = 0.0
        exit_exam_leaf.status = "failed"
    else:
        exit_exam_claim = (
            f"As of 2024, the U.S. state of {state_name} requires high school students to pass a standardized exit exam "
            f"(such as statewide end-of-course exams, Regents-style exams, or equivalent statewide graduation tests) to graduate."
        )
        exit_exam_sources = _safe_sources(state_entry.exit_exam_urls)
        await evaluator.verify(
            claim=exit_exam_claim,
            node=exit_exam_leaf,
            sources=exit_exam_sources,
            additional_instruction=(
                "Check the referenced statewide graduation requirements (not district-only) to confirm that passing a "
                "standardized exit exam is required for earning a high school diploma as of 2024. "
                "Accept commonly used statewide systems (e.g., Regents Exams, EOCs) that are explicitly tied to graduation."
            )
        )

    # Leaf: Minimum credits >= 22
    credits_leaf = evaluator.add_leaf(
        id=f"state_{sidx}_min_credits",
        desc="State requires a minimum of 22 credits or more for high school graduation",
        parent=req_node,
        critical=True
    )

    if not _has_text(state_entry.state_name):
        credits_leaf.score = 0.0
        credits_leaf.status = "failed"
    else:
        credits_claim = (
            f"{state_name} requires at least 22 credits for high school graduation at the statewide level "
            f"(i.e., the minimum statewide diploma credit requirement is 22 or higher)."
        )
        credits_sources = _safe_sources(state_entry.min_credits_urls)
        await evaluator.verify(
            claim=credits_claim,
            node=credits_leaf,
            sources=credits_sources,
            additional_instruction=(
                "Verify the statewide minimum credit requirement for a standard high school diploma. "
                "Do not rely on district-only policies. Consider synonymous terms like 'units' if used statewide. "
                "The claim is correct only if the statewide minimum is 22 or more credits."
            )
        )

    # -------- University requirements (critical, parallel) --------
    uni_node = evaluator.add_parallel(
        id=f"state_{sidx}_university",
        desc=f"One qualifying university in State {sidx} and required details",
        parent=state_node,
        critical=True
    )

    uni = state_entry.university or UniversityDetails()

    # Leaf: University located in the identified state
    in_state_leaf = evaluator.add_leaf(
        id=f"state_{sidx}_university_in_state",
        desc="University is located in the identified state",
        parent=uni_node,
        critical=True
    )
    if not (_has_text(uni.name) and _has_text(state_entry.state_name)):
        in_state_leaf.score = 0.0
        in_state_leaf.status = "failed"
    else:
        in_state_claim = f"{uni.name} is located in the U.S. state of {state_name}."
        in_state_sources = _safe_sources(uni.location_urls, uni.website_urls)
        await evaluator.verify(
            claim=in_state_claim,
            node=in_state_leaf,
            sources=in_state_sources,
            additional_instruction=(
                "Accept official university pages, athletics pages, government/education pages, or reputable directories "
                "that explicitly show the university's state location."
            )
        )

    # Leaf: NCAA Division I member institution
    d1_member_leaf = evaluator.add_leaf(
        id=f"state_{sidx}_university_ncaa_div1_member",
        desc="University is an NCAA Division I member institution",
        parent=uni_node,
        critical=True
    )
    if not _has_text(uni.name):
        d1_member_leaf.score = 0.0
        d1_member_leaf.status = "failed"
    else:
        d1_member_claim = f"{uni.name} competes in NCAA Division I athletics."
        d1_member_sources = _safe_sources(uni.ncaa_division_urls, uni.website_urls)
        await evaluator.verify(
            claim=d1_member_claim,
            node=d1_member_leaf,
            sources=d1_member_sources,
            additional_instruction=(
                "Confirm NCAA Division I membership via NCAA, conference, or official athletics/university pages. "
                "Do not accept club, intramural, or lower division references."
            )
        )

    # Leaf: University has an NCAA Division I football program
    football_leaf = evaluator.add_leaf(
        id=f"state_{sidx}_university_div1_football",
        desc="University has an NCAA Division I football program",
        parent=uni_node,
        critical=True
    )
    if not _has_text(uni.name):
        football_leaf.score = 0.0
        football_leaf.status = "failed"
    else:
        football_claim = (
            f"{uni.name} fields an NCAA Division I football team (FBS or FCS)."
        )
        football_sources = _safe_sources(uni.football_urls, uni.ncaa_division_urls, uni.website_urls)
        await evaluator.verify(
            claim=football_claim,
            node=football_leaf,
            sources=football_sources,
            additional_instruction=(
                "Accept NCAA pages, official athletics pages, or reputable conference pages that explicitly indicate "
                "the program competes in FBS or FCS (both are NCAA Division I subdivisions)."
            )
        )

    # Leaf: Accreditation (specific regional accreditor provided)
    accred_leaf = evaluator.add_leaf(
        id=f"state_{sidx}_university_accreditation",
        desc="Specific name of the regional accrediting body that accredits the university is provided",
        parent=uni_node,
        critical=True
    )
    if not (_has_text(uni.name) and _has_text(uni.accreditor_name)):
        accred_leaf.score = 0.0
        accred_leaf.status = "failed"
    else:
        accred_claim = f"{uni.name} is institutionally accredited by {uni.accreditor_name}."
        accred_sources = _safe_sources(uni.accreditation_urls, uni.website_urls)
        await evaluator.verify(
            claim=accred_claim,
            node=accred_leaf,
            sources=accred_sources,
            additional_instruction=(
                "Verify the institutional (regional) accreditor, not programmatic accreditors. "
                "Accept accreditor directories or official university accreditation pages that explicitly name the institution's regional accreditor "
                "(e.g., SACSCOC, HLC, MSCHE, NECHE, WSCUC, NWCCU)."
            )
        )

    # Leaf: NCAA Division I minimum core GPA is correctly reported (2.3)
    gpa_leaf = evaluator.add_leaf(
        id=f"state_{sidx}_ncaa_core_gpa",
        desc="NCAA Division I minimum core GPA requirement is provided (2.3)",
        parent=uni_node,
        critical=True
    )
    # We evaluate against the answer text: did the answer report 2.3 for DI minimum core GPA?
    gpa_claim = (
        "In the provided answer, the reported NCAA Division I minimum core GPA for initial eligibility equals 2.3 "
        "(allow minor formatting like '2.30' to be considered equivalent)."
    )
    # Use eligibility URLs if available; otherwise simple verify based on the answer text
    gpa_sources = _safe_sources(uni.ncaa_eligibility_urls)
    await evaluator.verify(
        claim=gpa_claim,
        node=gpa_leaf,
        sources=gpa_sources,
        additional_instruction=(
            "Judge by comparing the answer's stated value against the official NCAA threshold (2.3). "
            "This passes only if the answer clearly reports 2.3 (variants like 2.30 acceptable)."
        )
    )

    # Leaf: NCAA Division I total number of core courses is correctly reported (16)
    core_total_leaf = evaluator.add_leaf(
        id=f"state_{sidx}_ncaa_core_courses",
        desc="NCAA Division I total number of core courses required is provided (16)",
        parent=uni_node,
        critical=True
    )
    core_total_claim = (
        "In the provided answer, the reported total number of NCAA Division I core courses equals 16."
    )
    core_total_sources = _safe_sources(uni.ncaa_eligibility_urls)
    await evaluator.verify(
        claim=core_total_claim,
        node=core_total_leaf,
        sources=core_total_sources,
        additional_instruction=(
            "Judge by comparing the answer's stated value against the official NCAA requirement (16 core courses). "
            "Pass only if the answer clearly reports 16."
        )
    )

    # Leaf: NCAA Division I English requirement correctly reported (4 years/credits)
    english_leaf = evaluator.add_leaf(
        id=f"state_{sidx}_ncaa_english_requirement",
        desc="NCAA Division I English requirement within core courses is provided (4 years/credits)",
        parent=uni_node,
        critical=True
    )
    english_claim = (
        "In the provided answer, the reported NCAA Division I English requirement within the core courses equals 4 years (or 4 credits)."
    )
    english_sources = _safe_sources(uni.ncaa_eligibility_urls)
    await evaluator.verify(
        claim=english_claim,
        node=english_leaf,
        sources=english_sources,
        additional_instruction=(
            "Judge by comparing the answer's stated English requirement against the official NCAA requirement (4 years/credits of English). "
            "Pass only if the answer clearly reports 4."
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
    Entry point to evaluate an answer for the 'high school graduation states + NCAA university details' task.
    """
    # 1) Initialize evaluator
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Parallel across the 3 states
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

    # 2) Extract structured info
    extraction: StatesExtraction = await evaluator.extract(
        prompt=prompt_extract_states(),
        template_class=StatesExtraction,
        extraction_name="states_and_university_extraction",
    )

    # 3) Normalize to exactly 3 states: keep first 3, pad with empty if fewer
    states_list: List[StateEntry] = extraction.states[:3]
    while len(states_list) < 3:
        states_list.append(StateEntry())

    # 4) Build verification tree per state (sequential within each state)
    #    These are independent across states (root is parallel)
    tasks = []
    for idx in range(3):
        tasks.append(
            verify_one_state(evaluator, root, idx, states_list[idx])
        )
    # Execute verifications (sequentially within each state node function)
    for t in tasks:
        await t

    # 5) Return summary
    return evaluator.get_summary()