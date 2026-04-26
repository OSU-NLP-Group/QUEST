import asyncio
import logging
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator, AggregationStrategy


# -----------------------------------------------------------------------------
# Task-specific constants
# -----------------------------------------------------------------------------
TASK_ID = "fcs_to_fbs_hire_nov2024"
TASK_DESCRIPTION = (
    "In November 2024, a college football coach who had been serving as head coach at an FCS "
    "institution was hired as head coach by an FBS program in either the American Athletic Conference "
    "or Conference USA. This coach had established a successful record at their FCS institution over "
    "multiple seasons and had previously served as a head coach at the Division III level before moving "
    "to FCS.\n\n"
    "Identify this coach and provide: "
    "1) Coach & FBS institution details with announcement date; "
    "2) Educational credentials (bachelor's and master's, with institutions and years); "
    "3) FCS head coaching record details (tenure years, overall record, conference, notable achievements); "
    "4) Prior head coaching roles before FCS (including division level confirming Division III prior to FCS); "
    "5) Division transition details (confirm FCS-to-FBS and the FBS conference = AAC or C-USA); "
    "6) Hiring context (replaced coach and that coach's departure timing). "
    "Each major category must include at least one credible supporting source URL."
)


# -----------------------------------------------------------------------------
# Data Models for Extraction
# -----------------------------------------------------------------------------
class DegreeInfo(BaseModel):
    institution: Optional[str] = None
    graduation_year: Optional[str] = None


class EducationInfo(BaseModel):
    bachelors: Optional[DegreeInfo] = None
    masters: Optional[DegreeInfo] = None
    sources: List[str] = Field(default_factory=list)


class FCSRecordInfo(BaseModel):
    institution: Optional[str] = None
    tenure_years: Optional[str] = None
    overall_record: Optional[str] = None
    conference: Optional[str] = None
    notable_achievements: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class PriorHeadCoachPosition(BaseModel):
    institution: Optional[str] = None
    division_level: Optional[str] = None
    years: Optional[str] = None


class PriorExperienceInfo(BaseModel):
    positions: List[PriorHeadCoachPosition] = Field(default_factory=list)
    sources: List[str] = Field(default_factory=list)


class TransitionInfo(BaseModel):
    fbs_conference: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class HiringContextInfo(BaseModel):
    replaced_coach_name: Optional[str] = None
    departure_timing: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class CoachHireExtraction(BaseModel):
    coach_full_name: Optional[str] = None
    fbs_institution: Optional[str] = None
    official_announcement_date: Optional[str] = None
    coach_and_hire_sources: List[str] = Field(default_factory=list)

    education: Optional[EducationInfo] = None
    fcs_record: Optional[FCSRecordInfo] = None
    prior_experience: Optional[PriorExperienceInfo] = None
    transition: Optional[TransitionInfo] = None
    hiring_context: Optional[HiringContextInfo] = None


# -----------------------------------------------------------------------------
# Extraction Prompt
# -----------------------------------------------------------------------------
def prompt_extract_coach_hire() -> str:
    return """
You must extract structured information about a November 2024 hiring in which a then‑FCS head football coach was hired as an FBS head coach in either the American Athletic Conference (AAC) or Conference USA (C‑USA). The coach previously served as a head coach at the Division III level before their FCS role. Extract exactly what the answer explicitly states.

Return a single JSON object with these fields:

- coach_full_name: string | null
- fbs_institution: string | null
- official_announcement_date: string | null  (e.g., "November 20, 2024" or ISO date)
- coach_and_hire_sources: string[]  (URLs verifying coach, institution, and announcement date; at least one if provided)

- education: object | null
  - bachelors: object | null
    - institution: string | null
    - graduation_year: string | null
  - masters: object | null
    - institution: string | null
    - graduation_year: string | null
  - sources: string[]  (URLs verifying the degrees)

- fcs_record: object | null
  - institution: string | null
  - tenure_years: string | null  (e.g., "2020–2024" or similar)
  - overall_record: string | null  (e.g., "35–14")
  - conference: string | null
  - notable_achievements: string | null  (e.g., "2 conference titles; 3 playoff appearances")
  - sources: string[]  (URLs verifying these FCS details)

- prior_experience: object | null
  - positions: array of objects (prior head coaching roles BEFORE the FCS role)
    - institution: string | null
    - division_level: string | null  (e.g., "NCAA Division III")
    - years: string | null
  - sources: string[]  (URLs verifying prior head coaching roles and division level)

- transition: object | null
  - fbs_conference: string | null  (the hiring institution’s conference; must be AAC or Conference USA)
  - sources: string[]  (URLs verifying the FCS→FBS transition and conference)

- hiring_context: object | null
  - replaced_coach_name: string | null
  - departure_timing: string | null  (e.g., date of resignation/firing or when they departed)
  - sources: string[]  (URLs verifying replaced coach and departure timing)

EXTRACTION RULES:
- Extract only what appears explicitly in the answer. Do not invent values.
- For URL fields, extract actual URLs only. If no URL is present, return an empty array.
- If a subfield is missing in the answer, set it to null; do not guess.
- If multiple relevant URLs are present for a category, include all of them.
"""


# -----------------------------------------------------------------------------
# Helper Functions
# -----------------------------------------------------------------------------
def _is_nonempty(s: Optional[str]) -> bool:
    return bool(s and str(s).strip())


def _dedupe_urls(urls: List[str]) -> List[str]:
    seen = set()
    out = []
    for u in urls:
        if not _is_nonempty(u):
            continue
        if u not in seen:
            seen.add(u)
            out.append(u)
    return out


def _merge_sources(*url_lists: Optional[List[str]]) -> List[str]:
    merged: List[str] = []
    for lst in url_lists:
        if lst:
            merged.extend(lst)
    return _dedupe_urls(merged)


def _positions_to_text(positions: List[PriorHeadCoachPosition]) -> str:
    if not positions:
        return "None"
    chunks = []
    for p in positions:
        inst = p.institution or "Unknown Institution"
        divi = p.division_level or "Unknown Division"
        yrs = f", {p.years}" if _is_nonempty(p.years) else ""
        chunks.append(f"{inst} ({divi}{yrs})")
    return "; ".join(chunks)


# -----------------------------------------------------------------------------
# Verification Subtrees
# -----------------------------------------------------------------------------
async def build_and_verify_coach_and_institution(evaluator: Evaluator, parent, data: CoachHireExtraction) -> None:
    node = evaluator.add_parallel(
        id="coach_and_institution",
        desc="Provide the coach identity and hiring details with a credible source URL.",
        parent=parent,
        critical=True,
    )

    # Create all leaves before verification to ensure auto preconditions work
    leaf_name = evaluator.add_leaf(
        id="coach_full_name",
        desc="Provide the coach's full name.",
        parent=node,
        critical=True,
    )
    leaf_inst = evaluator.add_leaf(
        id="hiring_fbs_institution",
        desc="Provide the name of the FBS institution that hired the coach.",
        parent=node,
        critical=True,
    )
    leaf_date = evaluator.add_leaf(
        id="official_announcement_date",
        desc="Provide the official announcement date of the hire.",
        parent=node,
        critical=True,
    )
    # Source presence (existence check)
    src_present = evaluator.add_custom_node(
        result=bool(data.coach_and_hire_sources and len(data.coach_and_hire_sources) > 0),
        id="coach_and_hire_source_url",
        desc="Provide at least one credible source URL verifying the coach, institution, and announcement date.",
        parent=node,
        critical=True,
    )

    # Run verifications
    coach_name = data.coach_full_name or "None"
    fbs_inst = data.fbs_institution or "None"
    announce_date = data.official_announcement_date or "None"
    hire_sources = data.coach_and_hire_sources or []

    await evaluator.verify(
        claim=f"The hire announced in November 2024 names {coach_name} as the new head football coach.",
        node=leaf_name,
        sources=hire_sources,
        additional_instruction=(
            "Verify the press release or credible news explicitly names the new head coach. "
            "The context is a November 2024 hiring. Minor name variants are acceptable."
        ),
    )

    await evaluator.verify(
        claim=f"The FBS institution that hired the coach is {fbs_inst}.",
        node=leaf_inst,
        sources=hire_sources,
        additional_instruction="Check the source clearly identifies the hiring FBS institution.",
    )

    await evaluator.verify(
        claim=f"The official announcement date of the hire was {announce_date}.",
        node=leaf_date,
        sources=hire_sources,
        additional_instruction=(
            "The announcement must be in November 2024. If the stated date is not in November 2024, mark incorrect. "
            "Timezone or late-evening timestamps are fine as long as the public announcement reference is in November 2024."
        ),
    )


async def build_and_verify_education(evaluator: Evaluator, parent, data: CoachHireExtraction) -> None:
    node = evaluator.add_parallel(
        id="educational_credentials",
        desc="Provide the coach's bachelor's and master's degree details (institution and graduation year), with a credible source URL.",
        parent=parent,
        critical=True,
    )

    leaf_bach = evaluator.add_leaf(
        id="bachelors_degree",
        desc="Provide the bachelor’s degree institution and graduation year.",
        parent=node,
        critical=True,
    )
    leaf_mast = evaluator.add_leaf(
        id="masters_degree",
        desc="Provide the master’s degree institution and graduation year.",
        parent=node,
        critical=True,
    )
    src_present = evaluator.add_custom_node(
        result=bool(data.education and data.education.sources and len(data.education.sources) > 0),
        id="education_source_url",
        desc="Provide at least one credible source URL verifying the bachelor’s and master’s degree details.",
        parent=node,
        critical=True,
    )

    edu = data.education or EducationInfo()
    bach = edu.bachelors or DegreeInfo()
    mast = edu.masters or DegreeInfo()
    edu_sources = edu.sources or []

    await evaluator.verify(
        claim=f"The coach earned a bachelor's degree from {bach.institution or 'None'} in {bach.graduation_year or 'None'}.",
        node=leaf_bach,
        sources=edu_sources,
        additional_instruction="Verify both institution and year when possible; if year is missing in the page, mark unsupported.",
    )

    await evaluator.verify(
        claim=f"The coach earned a master's degree from {mast.institution or 'None'} in {mast.graduation_year or 'None'}.",
        node=leaf_mast,
        sources=edu_sources,
        additional_instruction="Verify both institution and year when possible; if year is missing in the page, mark unsupported.",
    )


async def build_and_verify_fcs_record(evaluator: Evaluator, parent, data: CoachHireExtraction) -> None:
    node = evaluator.add_parallel(
        id="fcs_head_coaching_record",
        desc="Provide the coach’s FCS head coaching tenure, record, conference, and notable achievements, with a credible source URL.",
        parent=parent,
        critical=True,
    )

    leaf_inst = evaluator.add_leaf(
        id="fcs_institution",
        desc="Identify the FCS institution where the coach served as head coach immediately prior to the FBS hire.",
        parent=node,
        critical=True,
    )
    leaf_years = evaluator.add_leaf(
        id="fcs_tenure_years",
        desc="Provide the years of tenure as FCS head coach.",
        parent=node,
        critical=True,
    )
    leaf_record = evaluator.add_leaf(
        id="overall_win_loss_record",
        desc="Provide the overall win-loss record as FCS head coach.",
        parent=node,
        critical=True,
    )
    leaf_conf = evaluator.add_leaf(
        id="conference_affiliation",
        desc="Identify the FCS institution’s conference affiliation during the coach’s tenure.",
        parent=node,
        critical=True,
    )
    leaf_achv = evaluator.add_leaf(
        id="notable_achievements",
        desc="Provide notable achievements (e.g., conference titles, playoff appearances) or explicitly state none, if applicable.",
        parent=node,
        critical=True,
    )
    src_present = evaluator.add_custom_node(
        result=bool(data.fcs_record and data.fcs_record.sources and len(data.fcs_record.sources) > 0),
        id="fcs_record_source_url",
        desc="Provide at least one credible source URL verifying the FCS tenure, record, conference, and achievements.",
        parent=node,
        critical=True,
    )

    rec = data.fcs_record or FCSRecordInfo()
    fcs_sources = rec.sources or []

    await evaluator.verify(
        claim=f"The coach served as head coach at the FCS institution {rec.institution or 'None'} immediately prior to the FBS hire.",
        node=leaf_inst,
        sources=fcs_sources,
        additional_instruction="Verify the specific FCS school named as the coach’s most recent head coaching position before the FBS hire.",
    )

    await evaluator.verify(
        claim=f"The coach's tenure as FCS head coach at {rec.institution or 'None'} spanned {rec.tenure_years or 'None'}.",
        node=leaf_years,
        sources=fcs_sources,
        additional_instruction="Verify that the tenure years match what is stated on the page (allow minor formatting like en-dashes).",
    )

    await evaluator.verify(
        claim=f"The overall win-loss record as FCS head coach at {rec.institution or 'None'} was {rec.overall_record or 'None'}.",
        node=leaf_record,
        sources=fcs_sources,
        additional_instruction="Verify the overall record is explicitly stated (e.g., 35–14).",
    )

    await evaluator.verify(
        claim=f"During the coach's FCS tenure at {rec.institution or 'None'}, the team's conference affiliation was {rec.conference or 'None'}.",
        node=leaf_conf,
        sources=fcs_sources,
        additional_instruction="Verify the named FCS conference for that period.",
    )

    await evaluator.verify(
        claim=f"Notable achievements at {rec.institution or 'None'} included: {rec.notable_achievements or 'None'}.",
        node=leaf_achv,
        sources=fcs_sources,
        additional_instruction="Verify achievements such as conference titles or playoff appearances, if claimed.",
    )


async def build_and_verify_prior_experience(evaluator: Evaluator, parent, data: CoachHireExtraction) -> None:
    node = evaluator.add_parallel(
        id="prior_head_coaching_experience",
        desc="Provide prior head coaching experience before the FCS role (including institution and division level), and confirm Division III head coaching occurred before FCS, with a credible source URL.",
        parent=parent,
        critical=True,
    )

    leaf_positions = evaluator.add_leaf(
        id="prior_head_coaching_positions_with_division",
        desc="List any prior head coaching positions held before the FCS role, including the institution and NCAA division level for each.",
        parent=node,
        critical=True,
    )
    leaf_d3 = evaluator.add_leaf(
        id="division_iii_before_fcs_confirmed",
        desc="Confirm the coach served as a head coach at the Division III level prior to becoming an FCS head coach.",
        parent=node,
        critical=True,
    )
    src_present = evaluator.add_custom_node(
        result=bool(data.prior_experience and data.prior_experience.sources and len(data.prior_experience.sources) > 0),
        id="prior_experience_source_url",
        desc="Provide at least one credible source URL verifying the prior head coaching experience and division level(s).",
        parent=node,
        critical=True,
    )

    pe = data.prior_experience or PriorExperienceInfo()
    pe_sources = pe.sources or []
    positions_text = _positions_to_text(pe.positions)

    await evaluator.verify(
        claim=f"Before becoming an FCS head coach, prior head coaching positions included: {positions_text}.",
        node=leaf_positions,
        sources=pe_sources,
        additional_instruction="Verify the institutions and NCAA division levels for prior head coaching roles.",
    )

    await evaluator.verify(
        claim="Before becoming an FCS head coach, the coach served as a head coach at the NCAA Division III level.",
        node=leaf_d3,
        sources=pe_sources,
        additional_instruction="Confirm explicitly that at least one prior head coaching role was NCAA Division III and that it predated the FCS head coaching role.",
    )


async def build_and_verify_transition(evaluator: Evaluator, parent, data: CoachHireExtraction) -> None:
    node = evaluator.add_parallel(
        id="division_transition_details",
        desc="Confirm the transition is from FCS head coach to FBS head coach and identify the FBS conference (must be AAC or C-USA), with a credible source URL.",
        parent=parent,
        critical=True,
    )

    leaf_trans = evaluator.add_leaf(
        id="fcs_to_fbs_transition_confirmed",
        desc="Confirm the hire represents a transition from an FCS head coaching role to an FBS head coaching role.",
        parent=node,
        critical=True,
    )
    leaf_conf = evaluator.add_leaf(
        id="fbs_conference_identified",
        desc="Identify the hiring institution’s FBS conference and ensure it is either the American Athletic Conference or Conference USA.",
        parent=node,
        critical=True,
    )
    src_present = evaluator.add_custom_node(
        result=bool(data.transition and data.transition.sources and len(data.transition.sources) > 0),
        id="transition_source_url",
        desc="Provide at least one credible source URL verifying the FCS-to-FBS transition and the hiring institution’s conference.",
        parent=node,
        critical=True,
    )

    rec = data.fcs_record or FCSRecordInfo()
    trans = data.transition or TransitionInfo()

    coach_name = data.coach_full_name or "the coach"
    fcs_inst = rec.institution or "None"
    fbs_inst = data.fbs_institution or "None"
    fbs_conf = trans.fbs_conference or "None"
    merged_sources = _merge_sources(
        data.coach_and_hire_sources if data.coach_and_hire_sources else [],
        rec.sources if rec.sources else [],
        trans.sources if trans.sources else [],
    )

    await evaluator.verify(
        claim=f"This hire represents a move from an FCS head coaching role at {fcs_inst} to an FBS head coaching role at {fbs_inst}.",
        node=leaf_trans,
        sources=merged_sources,
        additional_instruction="Verify that the coach was an FCS head coach immediately prior and is now hired as an FBS head coach.",
    )

    await evaluator.verify(
        claim=f"The hiring institution {fbs_inst} competes in the {fbs_conf}.",
        node=leaf_conf,
        sources=merged_sources,
        additional_instruction=(
            "Verify the program’s FBS conference and ensure it is either the American Athletic Conference (AAC) "
            "or Conference USA (C-USA). If it is not AAC or C-USA, mark this check as incorrect."
        ),
    )


async def build_and_verify_hiring_context(evaluator: Evaluator, parent, data: CoachHireExtraction) -> None:
    node = evaluator.add_parallel(
        id="hiring_context",
        desc="Provide the replaced coach’s name and when that coach departed, with a credible source URL.",
        parent=parent,
        critical=True,
    )

    leaf_replaced = evaluator.add_leaf(
        id="replaced_coach_name",
        desc="Identify the coach who was replaced by the hire.",
        parent=node,
        critical=True,
    )
    leaf_depart = evaluator.add_leaf(
        id="replaced_coach_departure_timing",
        desc="Provide when the replaced coach departed (e.g., fired/resigned date).",
        parent=node,
        critical=True,
    )
    src_present = evaluator.add_custom_node(
        result=bool(data.hiring_context and data.hiring_context.sources and len(data.hiring_context.sources) > 0),
        id="hiring_context_source_url",
        desc="Provide at least one credible source URL verifying the replaced coach and the departure timing.",
        parent=node,
        critical=True,
    )

    ctx = data.hiring_context or HiringContextInfo()
    ctx_sources = ctx.sources or []

    await evaluator.verify(
        claim=f"The coach replaced was {ctx.replaced_coach_name or 'None'}.",
        node=leaf_replaced,
        sources=ctx_sources,
        additional_instruction="Verify the page explicitly names the outgoing head coach being replaced.",
    )

    await evaluator.verify(
        claim=f"The replaced coach departed {ctx.departure_timing or 'None'}.",
        node=leaf_depart,
        sources=ctx_sources,
        additional_instruction="Verify the date/timing of the outgoing coach's departure (e.g., resignation or firing date).",
    )


# -----------------------------------------------------------------------------
# Main Evaluation Entry Point
# -----------------------------------------------------------------------------
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
    Evaluate an answer for the November 2024 FCS-to-FBS head coach hiring task.
    """
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
        default_model=model,
    )

    # Extraction
    extracted: CoachHireExtraction = await evaluator.extract(
        prompt=prompt_extract_coach_hire(),
        template_class=CoachHireExtraction,
        extraction_name="coach_hire_extraction",
    )

    # Build a critical sequential main node to reflect rubric's critical root
    main = evaluator.add_sequential(
        id="task_main",
        desc="Identify the November 2024 hire of an FCS head coach into an FBS head coach position (AAC or C-USA) who previously served as a Division III head coach, and provide all requested background details with credible sources.",
        parent=root,
        critical=True,
    )

    # Part 1: Coach & Institution
    await build_and_verify_coach_and_institution(evaluator, main, extracted)

    # Part 2: Remaining details (parallel critical group per rubric)
    details = evaluator.add_parallel(
        id="details",
        desc="Provide the remaining required background details, each with at least one credible source URL per major category.",
        parent=main,
        critical=True,
    )

    await build_and_verify_education(evaluator, details, extracted)
    await build_and_verify_fcs_record(evaluator, details, extracted)
    await build_and_verify_prior_experience(evaluator, details, extracted)
    await build_and_verify_transition(evaluator, details, extracted)
    await build_and_verify_hiring_context(evaluator, details, extracted)

    return evaluator.get_summary()