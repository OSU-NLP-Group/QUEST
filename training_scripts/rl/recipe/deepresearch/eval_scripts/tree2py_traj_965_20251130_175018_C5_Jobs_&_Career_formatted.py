import asyncio
import logging
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys


# --------------------------------------------------------------------------- #
# Task constants                                                              #
# --------------------------------------------------------------------------- #
TASK_ID = "psu_head_coach_opening_2025_overview"
TASK_DESCRIPTION = (
    "As of November 30, 2025, Penn State University has an open head football coach position. "
    "Provide a comprehensive overview of this coaching opportunity by identifying: "
    "(1) Penn State's current Athletic Director and University President; "
    "(2) the date when the head coaching position became vacant; "
    "(3) the 2025 annual salary of Penn State's previous head coach; "
    "(4) a recent comparable Power 4 coaching hire from the 2025 cycle with the coach's name, hiring school, and salary details to serve as a market benchmark; "
    "(5) Penn State's current conference affiliation; "
    "(6) Penn State's win-loss record since the start of the 2021 season; "
    "(7) details about Penn State's major stadium renovation project including the estimated cost. "
    "Additionally, as supplementary context, provide two examples of recent Power 4 head coaching hires from the 2025 cycle, including each coach's name, hiring institution, previous position, and coaching record at that previous position."
)

AS_OF_TIMEFRAME = "November 30, 2025"


# --------------------------------------------------------------------------- #
# Extraction data models                                                      #
# --------------------------------------------------------------------------- #
class NameWithSources(BaseModel):
    name: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class OpeningInfo(BaseModel):
    # Explicit statement that the job is open/ vacant as of the specified timeframe
    position_open_statement: Optional[str] = None
    sources_open_status: List[str] = Field(default_factory=list)

    # Date when the job became vacant (string; allow any reasonable date format)
    vacancy_date: Optional[str] = None
    sources_vacancy_date: List[str] = Field(default_factory=list)


class LeadershipInfo(BaseModel):
    athletic_director_name: Optional[str] = None
    athletic_director_sources: List[str] = Field(default_factory=list)

    university_president_name: Optional[str] = None
    university_president_sources: List[str] = Field(default_factory=list)


class ComparableHire(BaseModel):
    coach_name: Optional[str] = None
    hiring_school: Optional[str] = None
    salary_details: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class CompensationInfo(BaseModel):
    previous_head_coach_name: Optional[str] = None
    previous_head_coach_salary_2025: Optional[str] = None
    previous_head_coach_salary_sources: List[str] = Field(default_factory=list)

    comparable_power4_hire_2025_cycle: Optional[ComparableHire] = None


class ProgramFacts(BaseModel):
    conference_affiliation: Optional[str] = None
    conference_sources: List[str] = Field(default_factory=list)

    win_loss_since_2021: Optional[str] = None
    win_loss_sources: List[str] = Field(default_factory=list)

    stadium_project_description: Optional[str] = None
    stadium_renovation_cost: Optional[str] = None
    stadium_sources: List[str] = Field(default_factory=list)


class SupplementaryHireExample(BaseModel):
    coach_name: Optional[str] = None
    hiring_institution: Optional[str] = None
    previous_position: Optional[str] = None
    previous_position_record: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class PennStateOverviewExtraction(BaseModel):
    opening: Optional[OpeningInfo] = None
    leadership: Optional[LeadershipInfo] = None
    compensation: Optional[CompensationInfo] = None
    program_facts: Optional[ProgramFacts] = None
    supplementary_hires: List[SupplementaryHireExample] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_overview() -> str:
    return f"""
Extract the requested information exactly as presented in the answer text for Penn State's open head football coach position overview. 
Return JSON that matches the following schema and rules:

Schema:
- opening:
  - position_open_statement: string (the answer's statement asserting the head coach position is open as of "{AS_OF_TIMEFRAME}")
  - sources_open_status: array of URL strings cited for the open/vacant status
  - vacancy_date: string (the date the position became vacant, as given)
  - sources_vacancy_date: array of URL strings cited for the vacancy date
- leadership:
  - athletic_director_name: string
  - athletic_director_sources: array of URL strings cited for AD
  - university_president_name: string
  - university_president_sources: array of URL strings cited for President
- compensation:
  - previous_head_coach_name: string (the previous Penn State head coach's name)
  - previous_head_coach_salary_2025: string (the 2025 annual salary figure as quoted, allow ranges or approximate)
  - previous_head_coach_salary_sources: array of URL strings cited for this salary
  - comparable_power4_hire_2025_cycle:
      coach_name: string
      hiring_school: string
      salary_details: string (salary/contract details as phrased in the answer)
      sources: array of URL strings cited for this hire
- program_facts:
  - conference_affiliation: string
  - conference_sources: array of URL strings
  - win_loss_since_2021: string (the cumulative record since the start of the 2021 season, any reasonable format like '41-11' or 'W-L' with caveats)
  - win_loss_sources: array of URL strings
  - stadium_project_description: string (short description)
  - stadium_renovation_cost: string (estimated cost, allow ranges)
  - stadium_sources: array of URL strings
- supplementary_hires: array (max 2)
  - For each element:
      coach_name: string
      hiring_institution: string
      previous_position: string
      previous_position_record: string
      sources: array of URL strings

Rules:
- Extract only what appears in the answer text; do not infer or add new details.
- For any field not mentioned, set it to null (for strings) or an empty array (for sources).
- For URLs, extract actual URL strings referenced in the answer text (including markdown link targets).
- Preserve the exact phrasing for text fields (e.g., salary_details, renovation cost) as given in the answer.
- Limit supplementary_hires to at most 2 items, taking the first two presented in the answer if more are listed.
"""


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _norm_sources(urls: Optional[List[str]]) -> Optional[List[str]]:
    if not urls:
        return None
    cleaned = [u.strip() for u in urls if isinstance(u, str) and u.strip()]
    return cleaned if cleaned else None


def _require_sources_instruction(base_instruction: str, urls: Optional[List[str]]) -> str:
    suffix = (
        " You must rely on the provided webpage(s) as evidence. If no valid URL sources are provided, "
        "conclude the claim is not supported."
    )
    return (base_instruction or "Verify the claim using the cited sources.") + suffix


async def _add_and_verify_leaf(
    evaluator: Evaluator,
    *,
    node_id: str,
    desc: str,
    claim: str,
    parent,
    critical: bool,
    sources: Optional[List[str]],
    add_ins: str,
) -> None:
    node = evaluator.add_leaf(
        id=node_id,
        desc=desc,
        parent=parent,
        critical=critical,
    )
    await evaluator.verify(
        claim=claim,
        node=node,
        sources=sources,
        additional_instruction=add_ins,
    )


# --------------------------------------------------------------------------- #
# Verification tree construction                                              #
# --------------------------------------------------------------------------- #
async def build_verification_tree(
    evaluator: Evaluator,
    extracted: PennStateOverviewExtraction,
) -> None:
    """
    Build the verification tree per the rubric.

    Note on critical structure:
    - The original rubric marks the top-level node as critical and includes a non-critical child (Supplementary examples).
      The verification framework enforces that a critical parent cannot have non-critical children. To respect both the
      rubric intent and framework constraints, we:
        • Create a top-level non-critical 'overview' node.
        • Under it, add the four required sections as CRITICAL children.
        • Add the supplementary examples as a NON-CRITICAL sibling section.
      This preserves the rubric semantics (required vs. supplementary) while staying valid in the framework.
    """
    # Top-level grouping node (non-critical, parallel)
    overview = evaluator.add_parallel(
        id="Penn_State_Coaching_Position_Overview",
        desc="Provide the requested overview details about Penn State's open head football coach position.",
        parent=evaluator.root,
        critical=False,
    )

    # -------------------- Opening Status & Vacancy (CRITICAL) --------------------
    opening_node = evaluator.add_parallel(
        id="Opening_Status_And_Vacancy",
        desc="Confirm the position is open and provide the vacancy date.",
        parent=overview,
        critical=True,
    )

    opening = extracted.opening or OpeningInfo()

    # Position is open as of timeframe
    open_sources = _norm_sources(opening.sources_open_status)
    open_claim = (
        f"As of {AS_OF_TIMEFRAME}, Penn State's head football coach position is open or vacant "
        f"(e.g., vacancy/search underway/interim coach in place)."
    )
    open_add_ins = _require_sources_instruction(
        "Confirm that the source(s) explicitly support that the Penn State head coach job is open as of the specified date. "
        "Accept synonymous language like 'vacant', 'mutually parted ways', 'search underway', or appointment of an interim implying vacancy. "
        "Ensure the timing is consistent with being true as of November 30, 2025.",
        open_sources,
    )
    await _add_and_verify_leaf(
        evaluator,
        node_id="Position_Is_Open_As_Of_Specified_Timeframe",
        desc="State that Penn State's head football coach position is open as of the specified timeframe.",
        claim=open_claim,
        parent=opening_node,
        critical=True,
        sources=open_sources,
        add_ins=open_add_ins,
    )

    # Vacancy date
    vac_date_display = opening.vacancy_date or ""
    vac_sources = _norm_sources(opening.sources_vacancy_date)
    vac_claim = f"The head coaching position became vacant on {vac_date_display}."
    vac_add_ins = _require_sources_instruction(
        "Verify the specific vacancy date as stated (allowing for reasonable reporting conventions and time-zone publication timestamps). "
        "If the answer's date is blank or missing, or if sources do not specify the date, mark as not supported.",
        vac_sources,
    )
    await _add_and_verify_leaf(
        evaluator,
        node_id="Vacancy_Date",
        desc="Provide the date when the head coaching position became vacant.",
        claim=vac_claim,
        parent=opening_node,
        critical=True,
        sources=vac_sources,
        add_ins=vac_add_ins,
    )

    # -------------------- Current Leadership (CRITICAL) -------------------------
    leadership_node = evaluator.add_parallel(
        id="Current_Leadership",
        desc="Identify Penn State leadership requested.",
        parent=overview,
        critical=True,
    )
    leadership = extracted.leadership or LeadershipInfo()

    # Athletic Director
    ad_name = leadership.athletic_director_name or ""
    ad_sources = _norm_sources(leadership.athletic_director_sources)
    ad_claim = f"Penn State's current Athletic Director is {ad_name}."
    ad_add_ins = _require_sources_instruction(
        "Confirm the identity of the current Athletic Director at Penn State as of late 2025. "
        "Minor name formatting differences are acceptable.",
        ad_sources,
    )
    await _add_and_verify_leaf(
        evaluator,
        node_id="Athletic_Director",
        desc="Identify Penn State's current Athletic Director.",
        claim=ad_claim,
        parent=leadership_node,
        critical=True,
        sources=ad_sources,
        add_ins=ad_add_ins,
    )

    # University President
    pres_name = leadership.university_president_name or ""
    pres_sources = _norm_sources(leadership.university_president_sources)
    pres_claim = f"Penn State's current University President is {pres_name}."
    pres_add_ins = _require_sources_instruction(
        "Confirm the identity of the current University President at Penn State as of late 2025.",
        pres_sources,
    )
    await _add_and_verify_leaf(
        evaluator,
        node_id="University_President",
        desc="Identify Penn State's current University President.",
        claim=pres_claim,
        parent=leadership_node,
        critical=True,
        sources=pres_sources,
        add_ins=pres_add_ins,
    )

    # -------------------- Compensation & Benchmark (CRITICAL) -------------------
    comp_node = evaluator.add_parallel(
        id="Compensation_And_Market_Benchmark",
        desc="Provide Penn State prior coach pay and one comparable Power 4 hire benchmark (with salary details).",
        parent=overview,
        critical=True,
    )
    compensation = extracted.compensation or CompensationInfo()

    # Previous head coach 2025 salary
    prev_name = compensation.previous_head_coach_name or "the previous Penn State head coach"
    prev_salary = compensation.previous_head_coach_salary_2025 or ""
    prev_sources = _norm_sources(compensation.previous_head_coach_salary_sources)
    prev_salary_claim = f"In 2025, the annual salary of {prev_name} was {prev_salary}."
    prev_salary_add_ins = _require_sources_instruction(
        "Verify the 2025 salary or total compensation for Penn State's prior head coach. "
        "Accept reasonable representations (e.g., base plus supplemental/retention; average annual value if explicitly stated). "
        "Minor rounding differences are acceptable if clearly equivalent.",
        prev_sources,
    )
    await _add_and_verify_leaf(
        evaluator,
        node_id="Previous_Head_Coach_2025_Salary",
        desc="Provide the 2025 annual salary of Penn State's previous head coach.",
        claim=prev_salary_claim,
        parent=comp_node,
        critical=True,
        sources=prev_sources,
        add_ins=prev_salary_add_ins,
    )

    # Comparable Power 4 hire
    comp_hire = compensation.comparable_power4_hire_2025_cycle or ComparableHire()
    ch_name = comp_hire.coach_name or ""
    ch_school = comp_hire.hiring_school or ""
    ch_salary_details = comp_hire.salary_details or ""
    ch_sources = _norm_sources(comp_hire.sources)
    comp_hire_claim = (
        f"During the 2025 coaching cycle, {ch_name} was hired by {ch_school} with salary/contract details stated as: {ch_salary_details}."
    )
    comp_hire_add_ins = _require_sources_instruction(
        "Confirm that this example is from the 2025 cycle and that the source(s) explicitly support the coach, hiring school, "
        "and the stated salary/contract details.",
        ch_sources,
    )
    await _add_and_verify_leaf(
        evaluator,
        node_id="Comparable_Power4_Hire_2025_Cycle",
        desc="Provide one recent comparable Power 4 head coaching hire from the 2025 cycle with required details.",
        claim=comp_hire_claim,
        parent=comp_node,
        critical=True,
        sources=ch_sources,
        add_ins=comp_hire_add_ins,
    )

    # -------------------- Program Facts (CRITICAL) -----------------------------
    facts_node = evaluator.add_parallel(
        id="Program_Facts",
        desc="Provide conference affiliation, record since 2021 season start, and stadium renovation cost details.",
        parent=overview,
        critical=True,
    )
    facts = extracted.program_facts or ProgramFacts()

    # Conference affiliation
    conf = facts.conference_affiliation or ""
    conf_sources = _norm_sources(facts.conference_sources)
    conf_claim = f"Penn State's current conference affiliation is {conf}."
    conf_add_ins = _require_sources_instruction(
        "Verify Penn State's current conference affiliation as of late 2025.",
        conf_sources,
    )
    await _add_and_verify_leaf(
        evaluator,
        node_id="Conference_Affiliation",
        desc="State Penn State's current conference affiliation.",
        claim=conf_claim,
        parent=facts_node,
        critical=True,
        sources=conf_sources,
        add_ins=conf_add_ins,
    )

    # Win-loss since start of 2021 season
    wl = facts.win_loss_since_2021 or ""
    wl_sources = _norm_sources(facts.win_loss_sources)
    wl_claim = f"Since the start of the 2021 season, Penn State's cumulative record is {wl}."
    wl_add_ins = _require_sources_instruction(
        "Confirm the cumulative win-loss record since the start of the 2021 season up to the relevant cutoff. "
        "Minor formatting variants (e.g., '41–11' vs '41-11') are acceptable.",
        wl_sources,
    )
    await _add_and_verify_leaf(
        evaluator,
        node_id="Win_Loss_Since_2021_Start",
        desc="Provide Penn State's win-loss record since the start of the 2021 season.",
        claim=wl_claim,
        parent=facts_node,
        critical=True,
        sources=wl_sources,
        add_ins=wl_add_ins,
    )

    # Stadium renovation cost
    ren_cost = facts.stadium_renovation_cost or ""
    ren_desc = facts.stadium_project_description or "Beaver Stadium renovation"
    ren_sources = _norm_sources(facts.stadium_sources)
    ren_claim = f"Penn State's major stadium renovation project ({ren_desc}) has an estimated cost of {ren_cost}."
    ren_add_ins = _require_sources_instruction(
        "Verify the Beaver Stadium (or major stadium) renovation estimated cost as stated. "
        "Accept ranges if the stated amount clearly falls within the cited range.",
        ren_sources,
    )
    await _add_and_verify_leaf(
        evaluator,
        node_id="Beaver_Stadium_Renovation_Cost",
        desc="Describe Penn State's major stadium renovation project and include the estimated cost.",
        claim=ren_claim,
        parent=facts_node,
        critical=True,
        sources=ren_sources,
        add_ins=ren_add_ins,
    )

    # -------------------- Supplementary Examples (NON-CRITICAL) -----------------
    supp_node = evaluator.add_parallel(
        id="Supplementary_2025_Cycle_Hire_Examples",
        desc="Two additional Power 4 head coaching hire examples from the 2025 cycle with requested details (supplementary context).",
        parent=overview,
        critical=False,
    )

    # Take up to 2 examples
    supp_hires = (extracted.supplementary_hires or [])[:2]
    # Pad with empties if fewer than 2
    while len(supp_hires) < 2:
        supp_hires.append(SupplementaryHireExample())

    # Example 1
    e1 = supp_hires[0]
    e1_claim = (
        f"In the 2025 cycle, {e1.coach_name or ''} was hired by {e1.hiring_institution or ''}; "
        f"their previous position was {e1.previous_position or ''} with a coaching record of {e1.previous_position_record or ''} at that prior position."
    )
    e1_sources = _norm_sources(e1.sources)
    e1_add_ins = _require_sources_instruction(
        "Confirm that this example is a 2025 cycle Power 4 head coaching hire and that the coach name, hiring institution, "
        "previous position, and previous-position record are supported by the sources.",
        e1_sources,
    )
    await _add_and_verify_leaf(
        evaluator,
        node_id="Hire_Example_1",
        desc="Provide one example including coach name, hiring institution, previous position, and prior coaching record.",
        claim=e1_claim,
        parent=supp_node,
        critical=False,
        sources=e1_sources,
        add_ins=e1_add_ins,
    )

    # Example 2
    e2 = supp_hires[1]
    e2_claim = (
        f"In the 2025 cycle, {e2.coach_name or ''} was hired by {e2.hiring_institution or ''}; "
        f"their previous position was {e2.previous_position or ''} with a coaching record of {e2.previous_position_record or ''} at that prior position."
    )
    e2_sources = _norm_sources(e2.sources)
    e2_add_ins = _require_sources_instruction(
        "Confirm that this example is a 2025 cycle Power 4 head coaching hire and that the coach name, hiring institution, "
        "previous position, and previous-position record are supported by the sources.",
        e2_sources,
    )
    await _add_and_verify_leaf(
        evaluator,
        node_id="Hire_Example_2",
        desc="Provide a second example including coach name, hiring institution, previous position, and prior coaching record.",
        claim=e2_claim,
        parent=supp_node,
        critical=False,
        sources=e2_sources,
        add_ins=e2_add_ins,
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
    Evaluate an answer for the Penn State head coach opening overview task.
    """
    # Initialize evaluator
    evaluator = Evaluator()
    evaluator.initialize(
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

    # Extract structured information from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_overview(),
        template_class=PennStateOverviewExtraction,
        extraction_name="psu_coaching_overview_extraction",
    )

    # Add useful context info
    evaluator.add_custom_info(
        {"as_of_timeframe": AS_OF_TIMEFRAME},
        info_type="context",
        info_name="timeframe_context",
    )

    # Build verification tree and run verifications
    await build_verification_tree(evaluator, extracted)

    # Return the structured evaluation summary
    return evaluator.get_summary()