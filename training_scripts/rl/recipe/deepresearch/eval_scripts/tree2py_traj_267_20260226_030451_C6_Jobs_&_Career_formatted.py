import asyncio
import logging
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field
from obj_task_eval.llm_client.base_client import LLMClient

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "d2_to_fbs_coaches_chain"
TASK_DESCRIPTION = """
Identify current or recent (as of 2026) FBS head football coaches who previously won NCAA Division II national championships as head coaches and subsequently served as offensive or defensive coordinators at FBS programs before becoming FBS head coaches. For each coach you identify, provide the following information: (1) The Division II institution where they won the national championship(s), the specific year(s) of the championship victories, and their tenure period as head coach at that institution; (2) The FBS institution where they served as a coordinator, the specific type of coordinator role (offensive or defensive), and the years they held that position; (3) The FBS institution where they currently serve or recently served as head coach, the year they began in that role, and any notable achievements in that position; (4) A coherent timeline showing the progression from Division II head coach to FBS coordinator to FBS head coach. Provide reference URLs for all major career milestones.
"""


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class D2Phase(BaseModel):
    institution_name: Optional[str] = None
    championship_years: List[str] = Field(default_factory=list)
    head_coach_tenure_years: Optional[str] = None
    reference_urls: List[str] = Field(default_factory=list)


class CoordinatorPhase(BaseModel):
    fbs_institution: Optional[str] = None
    role_type: Optional[str] = None  # "offensive" or "defensive" coordinator; allow variants like "OC" / "DC"
    years: Optional[str] = None
    reference_urls: List[str] = Field(default_factory=list)


class HeadCoachPhase(BaseModel):
    fbs_institution: Optional[str] = None
    start_year: Optional[str] = None
    notable_achievements: List[str] = Field(default_factory=list)
    reference_urls: List[str] = Field(default_factory=list)


class CoachInfo(BaseModel):
    name: Optional[str] = None
    d2_phase: Optional[D2Phase] = None
    coordinator_phase: Optional[CoordinatorPhase] = None
    head_coach_phase: Optional[HeadCoachPhase] = None
    timeline_text: Optional[str] = None


class CoachesExtraction(BaseModel):
    coaches: List[CoachInfo] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_coaches() -> str:
    return """
    Extract up to five coaches described in the answer who fit the following chain:
    – They won at least one NCAA Division II national championship as a head coach (specify the D-II institution, championship year(s), and their head-coach tenure years at that institution).
    – After the D-II title(s), they served as an offensive or defensive coordinator at an FBS program (specify FBS institution, role type, and years).
    – After serving as a coordinator, they became an FBS head coach (specify FBS institution, start year, and at least one notable achievement in the FBS head-coaching role).

    For each coach, return a JSON object with these fields:
    - name: Full name of the coach
    - d2_phase:
        - institution_name
        - championship_years: array of years (strings)
        - head_coach_tenure_years: the tenure period as head coach at the D-II institution (e.g., "2012–2016")
        - reference_urls: array of URLs explicitly provided in the answer that support the D-II championship/tenure details
    - coordinator_phase:
        - fbs_institution
        - role_type: "offensive coordinator" or "defensive coordinator" (allow abbreviations like "OC" or "DC")
        - years: years served in the coordinator role (e.g., "2017–2018")
        - reference_urls: array of URLs explicitly provided in the answer that support the FBS coordinator details
    - head_coach_phase:
        - fbs_institution
        - start_year: year they began as FBS head coach
        - notable_achievements: array of notable achievements mentioned for the FBS head-coaching role (e.g., "won conference title in 2023")
        - reference_urls: array of URLs explicitly provided in the answer that support the FBS head-coaching details
    - timeline_text: a concise timeline text summarizing progression from D-II champion head coach → FBS coordinator → FBS head coach

    IMPORTANT:
    - Extract ONLY what is explicitly present in the answer. If a field is missing, set it to null (or an empty array as appropriate).
    - For all URL fields, extract the actual URLs mentioned (plain link or markdown); do not infer or create URLs.
    - If a URL is missing a protocol, prepend "http://".
    - Return an object with a "coaches" array containing objects for each coach, in the same order as presented in the answer. Limit to at most 5.
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def _safe(s: Optional[str]) -> str:
    return s or ""

def _join_years(years: List[str]) -> str:
    return ", ".join(y for y in years if (y or "").strip())

def _has_urls(urls: Optional[List[str]]) -> bool:
    return bool(urls) and len([u for u in urls if (u or "").strip()]) > 0

def _qualifies_chain(coach: Optional[CoachInfo]) -> bool:
    """Minimal presence-based chain qualification: D-II champ details + coordinator details + head coach details with URLs."""
    if coach is None or (coach.name or "").strip() == "":
        return False
    d2 = coach.d2_phase
    coord = coach.coordinator_phase
    hc = coach.head_coach_phase
    if d2 is None or coord is None or hc is None:
        return False
    d2_ok = (d2.institution_name or "").strip() != "" and len(d2.championship_years) > 0 and _has_urls(d2.reference_urls)
    coord_ok = (coord.fbs_institution or "").strip() != "" and (coord.role_type or "").strip() != "" and (coord.years or "").strip() != "" and _has_urls(coord.reference_urls)
    hc_ok = (hc.fbs_institution or "").strip() != "" and (hc.start_year or "").strip() != "" and _has_urls(hc.reference_urls)
    return d2_ok and coord_ok and hc_ok


# --------------------------------------------------------------------------- #
# Verification functions                                                      #
# --------------------------------------------------------------------------- #
async def verify_coach(
    evaluator: Evaluator,
    parent_node,
    coach: CoachInfo,
    coach_index_1based: int,
) -> None:
    """
    Build the verification subtree for a single coach. This mirrors the rubric:
    - qualification (critical parallel)
    - division_ii_phase_details (critical parallel)
    - fbs_coordinator_phase_details (critical parallel)
    - fbs_head_coach_phase_details (critical parallel)
    - coherent_career_timeline (critical leaf)
    """

    name = _safe(coach.name)

    # Top-level coach node (non-critical per rubric)
    coach_node = evaluator.add_parallel(
        id=f"coach_{coach_index_1based}",
        desc=f"Evaluation of the {coach_index_1based} coach provided (if present).",
        parent=parent_node,
        critical=False
    )

    # -------------------- Qualification -------------------- #
    qual_node = evaluator.add_parallel(
        id=f"coach_{coach_index_1based}_qualification",
        desc="Coach meets all qualifying criteria (D-II champion HC; later FBS OC/DC; later FBS head coach; current/recent as of 2026).",
        parent=coach_node,
        critical=True
    )

    d2 = coach.d2_phase or D2Phase()
    coord = coach.coordinator_phase or CoordinatorPhase()
    hc = coach.head_coach_phase or HeadCoachPhase()

    # d2_champion_as_head_coach
    leaf_d2_champ = evaluator.add_leaf(
        id=f"coach_{coach_index_1based}_d2_champion_as_head_coach",
        desc="Coach won ≥1 NCAA Division II national championship as a head coach.",
        parent=qual_node,
        critical=True
    )
    d2_claim = f"{name} won NCAA Division II national championship(s) as head coach at {_safe(d2.institution_name)} in years {_join_years(d2.championship_years)}."
    await evaluator.verify(
        claim=d2_claim,
        node=leaf_d2_champ,
        sources=d2.reference_urls,
        additional_instruction="Confirm explicit language such as 'NCAA Division II national champion' or equivalent. Allow minor wording variants. The institution and years should match."
    )

    # fbs_coordinator_after_d2_title (role & years verified; chronology separately)
    leaf_coord = evaluator.add_leaf(
        id=f"coach_{coach_index_1based}_fbs_coordinator_after_d2_title",
        desc="After the D-II championship(s), coach served as an offensive or defensive coordinator at an FBS program.",
        parent=qual_node,
        critical=True
    )
    coord_claim = f"Following the D-II title(s), {name} served as {_safe(coord.role_type)} coordinator at {_safe(coord.fbs_institution)} during {_safe(coord.years)}."
    await evaluator.verify(
        claim=coord_claim,
        node=leaf_coord,
        sources=coord.reference_urls,
        additional_instruction="Focus on verifying the coordinator role (OC/DC) and years at the named FBS program. The 'after' chronology is evaluated elsewhere."
    )

    # fbs_head_coach_after_coordinator
    leaf_hc_after = evaluator.add_leaf(
        id=f"coach_{coach_index_1based}_fbs_head_coach_after_coordinator",
        desc="After serving as FBS coordinator, coach became an FBS head coach.",
        parent=qual_node,
        critical=True
    )
    hc_after_claim = f"After serving as FBS coordinator, {name} became the FBS head coach at {_safe(hc.fbs_institution)} starting in {_safe(hc.start_year)}."
    await evaluator.verify(
        claim=hc_after_claim,
        node=leaf_hc_after,
        sources=hc.reference_urls,
        additional_instruction="Verify that the coach was appointed/served as head coach at the specified FBS institution starting in the stated year. Chronology relative to coordinator role is evaluated elsewhere."
    )

    # current_or_recent_fbs_head_coach (as of 2026)
    leaf_current_recent = evaluator.add_leaf(
        id=f"coach_{coach_index_1based}_current_or_recent_fbs_head_coach",
        desc="Coach is currently (as of 2026) or recently (within last 5 years) an FBS head coach.",
        parent=qual_node,
        critical=True
    )
    current_recent_claim = f"As of 2026, {name} is currently or has served within the last five years as an FBS head coach at {_safe(hc.fbs_institution)} (start year: {_safe(hc.start_year)})."
    await evaluator.verify(
        claim=current_recent_claim,
        node=leaf_current_recent,
        sources=hc.reference_urls,
        additional_instruction="If the start year is 2021–2026 inclusive or the page indicates present/current head coach status around 2025–2026, consider 'current/recent'."
    )

    # -------------------- Division II Phase Details -------------------- #
    d2_details = evaluator.add_parallel(
        id=f"coach_{coach_index_1based}_division_ii_phase_details",
        desc="Division II championship head-coaching phase details are provided with a supporting URL.",
        parent=coach_node,
        critical=True
    )

    # d2_institution_name
    leaf_d2_inst = evaluator.add_leaf(
        id=f"coach_{coach_index_1based}_d2_institution_name",
        desc="Division II institution name where championship(s) were won is correctly identified.",
        parent=d2_details,
        critical=True
    )
    d2_inst_claim = f"{name} won the D-II national championship(s) while head coach at {_safe(d2.institution_name)}."
    await evaluator.verify(
        claim=d2_inst_claim,
        node=leaf_d2_inst,
        sources=d2.reference_urls,
        additional_instruction="Confirm the institution name associated with the D-II national title(s)."
    )

    # d2_championship_years
    leaf_d2_years = evaluator.add_leaf(
        id=f"coach_{coach_index_1based}_d2_championship_years",
        desc="Specific year(s) of NCAA Division II national championship victory(ies) are provided.",
        parent=d2_details,
        critical=True
    )
    d2_years_claim = f"The NCAA Division II championship year(s) for {_safe(d2.institution_name)} under head coach {name} include: {_join_years(d2.championship_years)}."
    await evaluator.verify(
        claim=d2_years_claim,
        node=leaf_d2_years,
        sources=d2.reference_urls,
        additional_instruction="Verify that the years listed match the page(s). Allow minor formatting variations."
    )

    # d2_head_coach_tenure_years
    leaf_d2_tenure = evaluator.add_leaf(
        id=f"coach_{coach_index_1based}_d2_head_coach_tenure_years",
        desc="Tenure period (years) as head coach at the Division II institution is stated.",
        parent=d2_details,
        critical=True
    )
    d2_tenure_claim = f"{name} served as head coach at {_safe(d2.institution_name)} during {_safe(d2.head_coach_tenure_years)}."
    await evaluator.verify(
        claim=d2_tenure_claim,
        node=leaf_d2_tenure,
        sources=d2.reference_urls,
        additional_instruction="Check that the tenure period matches what is stated on the page(s)."
    )

    # d2_reference_url (existence check)
    evaluator.add_custom_node(
        result=_has_urls(d2.reference_urls),
        id=f"coach_{coach_index_1based}_d2_reference_url",
        desc="At least one reference URL supports the Division II championship milestone/details.",
        parent=d2_details,
        critical=True
    )

    # -------------------- FBS Coordinator Phase Details -------------------- #
    coord_details = evaluator.add_parallel(
        id=f"coach_{coach_index_1based}_fbs_coordinator_phase_details",
        desc="FBS coordinator phase details are provided with a supporting URL.",
        parent=coach_node,
        critical=True
    )

    # coordinator_fbs_institution
    leaf_coord_inst = evaluator.add_leaf(
        id=f"coach_{coach_index_1based}_coordinator_fbs_institution",
        desc="FBS institution where the coach served as coordinator is identified.",
        parent=coord_details,
        critical=True
    )
    coord_inst_claim = f"{name} served at {_safe(coord.fbs_institution)} as a coordinator."
    await evaluator.verify(
        claim=coord_inst_claim,
        node=leaf_coord_inst,
        sources=coord.reference_urls,
        additional_instruction="Verify the named FBS institution where the coordinator role occurred."
    )

    # coordinator_role_type
    leaf_coord_role = evaluator.add_leaf(
        id=f"coach_{coach_index_1based}_coordinator_role_type",
        desc="Coordinator role type (offensive or defensive) is stated.",
        parent=coord_details,
        critical=True
    )
    coord_role_claim = f"The coordinator role type for {name} at {_safe(coord.fbs_institution)} was {_safe(coord.role_type)}."
    await evaluator.verify(
        claim=coord_role_claim,
        node=leaf_coord_role,
        sources=coord.reference_urls,
        additional_instruction="Confirm whether the role was offensive coordinator (OC) or defensive coordinator (DC), allowing common abbreviations."
    )

    # coordinator_years
    leaf_coord_years = evaluator.add_leaf(
        id=f"coach_{coach_index_1based}_coordinator_years",
        desc="Years of service as FBS coordinator are provided.",
        parent=coord_details,
        critical=True
    )
    coord_years_claim = f"{name} served as {_safe(coord.role_type)} coordinator at {_safe(coord.fbs_institution)} during {_safe(coord.years)}."
    await evaluator.verify(
        claim=coord_years_claim,
        node=leaf_coord_years,
        sources=coord.reference_urls,
        additional_instruction="Confirm the years (or range of years) for the coordinator role."
    )

    # coordinator_reference_url (existence check)
    evaluator.add_custom_node(
        result=_has_urls(coord.reference_urls),
        id=f"coach_{coach_index_1based}_coordinator_reference_url",
        desc="At least one reference URL supports the FBS coordinator milestone/details.",
        parent=coord_details,
        critical=True
    )

    # -------------------- FBS Head Coach Phase Details -------------------- #
    hc_details = evaluator.add_parallel(
        id=f"coach_{coach_index_1based}_fbs_head_coach_phase_details",
        desc="FBS head-coaching phase details are provided with a supporting URL.",
        parent=coach_node,
        critical=True
    )

    # fbs_head_coach_institution
    leaf_hc_inst = evaluator.add_leaf(
        id=f"coach_{coach_index_1based}_fbs_head_coach_institution",
        desc="FBS institution where the coach serves/served as head coach is identified.",
        parent=hc_details,
        critical=True
    )
    hc_inst_claim = f"{name} served/serves as head coach at {_safe(hc.fbs_institution)}."
    await evaluator.verify(
        claim=hc_inst_claim,
        node=leaf_hc_inst,
        sources=hc.reference_urls,
        additional_instruction="Verify that the coach held/is holding the head coach position at the named FBS school."
    )

    # fbs_head_coach_start_year
    leaf_hc_start = evaluator.add_leaf(
        id=f"coach_{coach_index_1based}_fbs_head_coach_start_year",
        desc="Year the coach began as FBS head coach is stated.",
        parent=hc_details,
        critical=True
    )
    hc_start_claim = f"{name} began as head coach at {_safe(hc.fbs_institution)} in {_safe(hc.start_year)}."
    await evaluator.verify(
        claim=hc_start_claim,
        node=leaf_hc_start,
        sources=hc.reference_urls,
        additional_instruction="Confirm the appointment/start year for the head coach role."
    )

    # fbs_head_coach_notable_achievements
    leaf_hc_ach = evaluator.add_leaf(
        id=f"coach_{coach_index_1based}_fbs_head_coach_notable_achievements",
        desc="At least one notable achievement in the FBS head coaching role is provided.",
        parent=hc_details,
        critical=True
    )
    achievements_text = "; ".join([a for a in hc.notable_achievements if (a or "").strip()]) or "None stated"
    hc_ach_claim = f"At least one notable achievement for {name} as head coach at {_safe(hc.fbs_institution)} is accurate: {achievements_text}."
    await evaluator.verify(
        claim=hc_ach_claim,
        node=leaf_hc_ach,
        sources=hc.reference_urls,
        additional_instruction="Confirm that at least one listed achievement is supported (e.g., conference title, bowl win, award, ranking). Minor wording or formatting differences are acceptable."
    )

    # fbs_head_coach_reference_url (existence check)
    evaluator.add_custom_node(
        result=_has_urls(hc.reference_urls),
        id=f"coach_{coach_index_1based}_fbs_head_coach_reference_url",
        desc="At least one reference URL supports the FBS head coach milestone/details.",
        parent=hc_details,
        critical=True
    )

    # -------------------- Coherent Career Timeline -------------------- #
    leaf_timeline = evaluator.add_leaf(
        id=f"coach_{coach_index_1based}_coherent_career_timeline",
        desc="A coherent timeline is provided showing progression from D-II head coach (with championship) → FBS coordinator → FBS head coach, with correct chronological order and no temporal contradictions.",
        parent=coach_node,
        critical=True
    )
    timeline_claim = (
        f"Timeline for {name}: D-II head-coach tenure {_safe(d2.head_coach_tenure_years)} with title years {_join_years(d2.championship_years)}; "
        f"then FBS {_safe(coord.role_type)} coordinator at {_safe(coord.fbs_institution)} during {_safe(coord.years)}; "
        f"then FBS head coach at {_safe(hc.fbs_institution)} starting {_safe(hc.start_year)}. "
        f"The sequence is D-II champion head coach → FBS coordinator → FBS head coach in chronological order without contradictions."
    )
    await evaluator.verify(
        claim=timeline_claim,
        node=leaf_timeline,
        additional_instruction=(
            "Judge logical coherence only. Ensure the order is D-II head coach (with championship) → FBS OC/DC → FBS head coach, "
            "and that the years do not contradict (e.g., coordinator years after titles, head-coach start after coordinator). "
            "Allow approximate ranges; focus on order consistency."
        )
    )


# --------------------------------------------------------------------------- #
# Main evaluation entry point                                                 #
# --------------------------------------------------------------------------- #
async def evaluate_answer(
    client: LLMClient,
    answer: str,
    agent_name: str,
    answer_name: str,
    cache: CacheFileSys,
    semaphore: asyncio.Semaphore,
    logger: logging.Logger,
    model: str = "o4-mini"
) -> Dict[str, Any]:
    """
    Evaluate an answer for the D-II → FBS coordinator → FBS head coach chain task.
    """

    # Initialize evaluator
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Root parallel aggregation per rubric
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

    # IMPORTANT: Root must be non-critical to allow mixture of critical/non-critical children (framework constraint)
    root.critical = False

    # Extract structured coaches info
    extracted = await evaluator.extract(
        prompt=prompt_extract_coaches(),
        template_class=CoachesExtraction,
        extraction_name="coaches_extraction"
    )

    # Add a useful custom info entry
    evaluator.add_custom_info(
        info={"current_year_context": 2026, "max_coaches_evaluated": 5, "extracted_count": len(extracted.coaches)},
        info_type="meta",
        info_name="evaluation_context"
    )

    # Critical gate: at least one qualifying coach provided
    has_qualifying = any(_qualifies_chain(c) for c in (extracted.coaches or []))
    evaluator.add_custom_node(
        result=has_qualifying,
        id="at_least_one_qualifying_coach_provided",
        desc="Response identifies at least one coach who meets the full qualification chain (D-II champion head coach → FBS OC/DC after that → FBS head coach after that; current/recent as of 2026).",
        parent=root,
        critical=True
    )

    # Build verification subtrees for up to 5 coaches (pad with empty if fewer)
    coaches_list = list(extracted.coaches or [])
    while len(coaches_list) < 5:
        coaches_list.append(CoachInfo())

    for idx in range(5):
        coach = coaches_list[idx]
        try:
            await verify_coach(evaluator, root, coach, idx + 1)
        except Exception as e:
            # If verification building fails unexpectedly, create a failed leaf to capture the error
            err_leaf = evaluator.add_leaf(
                id=f"coach_{idx + 1}_verification_error",
                desc=f"Error building/verifying coach #{idx + 1} subtree: {str(e)}",
                parent=root,
                critical=False
            )
            # Mark as failed explicitly
            err_leaf.score = 0.0
            err_leaf.status = "failed"
            logger.error(f"Verification error for coach {idx + 1}: {e}")

    # Return summarized evaluation
    return evaluator.get_summary()