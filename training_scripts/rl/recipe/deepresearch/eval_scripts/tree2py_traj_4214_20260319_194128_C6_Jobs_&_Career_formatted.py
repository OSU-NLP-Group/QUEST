import asyncio
import logging
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.llm_client.base_client import LLMClient


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "syracuse_leadership_transition_2026"
TASK_DESCRIPTION = """
On July 1, 2026, Syracuse University is experiencing a significant leadership transition. Research and provide comprehensive information about this transition, including:

1. Outgoing Leader: Identify the Syracuse University Chancellor who is leaving on July 1, 2026. For this person, provide:
   - Their name
   - The institution they are moving to and their new position title
   - The athletic conference affiliation of their destination institution
   - The year they became Syracuse Chancellor and total years served
   - Their base salary at Syracuse in 2024
   - Their new base salary at the destination institution
   - The dollar amount of their base salary increase

2. Incoming Leader: Identify the person becoming Syracuse University's Chancellor on July 1, 2026. For this person, provide:
   - Their name
   - Their previous position at Syracuse University before this appointment
   - The historical significance of this internal promotion (specifically, when was the last time Syracuse promoted a chancellor from within?)
   - Approximately how many years they have worked at Syracuse University
   - Their military service background
   - Confirmation that they hold a terminal degree (PhD or equivalent)

3. Institutional Context: Provide:
   - Syracuse University's approximate annual budget
   - What number chancellor (in the university's history) the incoming leader will be

All information must be supported by reference URLs from reputable sources.
"""


# --------------------------------------------------------------------------- #
# Data models for structured extraction                                       #
# --------------------------------------------------------------------------- #
class OutgoingLeader(BaseModel):
    name: Optional[str] = None
    destination_institution: Optional[str] = None
    new_position_title: Optional[str] = None
    conference_affiliation: Optional[str] = None
    destination_sources: List[str] = Field(default_factory=list)

    start_year: Optional[str] = None
    total_years_served: Optional[str] = None
    tenure_sources: List[str] = Field(default_factory=list)

    su_base_salary_2024: Optional[str] = None
    new_base_salary: Optional[str] = None
    increase_amount: Optional[str] = None
    compensation_sources: List[str] = Field(default_factory=list)


class IncomingLeader(BaseModel):
    name: Optional[str] = None
    previous_internal_role: Optional[str] = None
    historical_significance: Optional[str] = None
    last_internal_promotion_year: Optional[str] = None

    years_at_syracuse: Optional[str] = None
    military_service: Optional[str] = None
    terminal_degree: Optional[str] = None
    background_sources: List[str] = Field(default_factory=list)


class TimelineInfo(BaseModel):
    effective_date: Optional[str] = None
    timeline_sources: List[str] = Field(default_factory=list)


class InstitutionalContext(BaseModel):
    annual_budget: Optional[str] = None
    chancellor_position_number: Optional[str] = None
    context_sources: List[str] = Field(default_factory=list)


class TransitionExtraction(BaseModel):
    outgoing: OutgoingLeader = OutgoingLeader()
    incoming: IncomingLeader = IncomingLeader()
    timeline: TimelineInfo = TimelineInfo()
    context: InstitutionalContext = InstitutionalContext()


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_transition() -> str:
    return """
Extract structured information about Syracuse University's leadership transition effective on July 1, 2026, exactly as presented in the answer. Return a single JSON object with the following nested fields:

outgoing:
  - name: the full name of the Syracuse University Chancellor who is leaving on July 1, 2026
  - destination_institution: the university or institution they are moving to
  - new_position_title: the title of the new position at the destination
  - conference_affiliation: the athletic conference of the destination institution (e.g., ACC, Big Ten)
  - destination_sources: an array of URLs that confirm destination_institution and new_position_title (include any relevant press releases, reputable news, or official pages)
  - start_year: the year they became Syracuse Chancellor (as a string, e.g., "2014")
  - total_years_served: total years served as Chancellor (string, allow approximate values like "12", "12+")
  - tenure_sources: an array of URLs that confirm start_year and/or total_years_served
  - su_base_salary_2024: their base salary at Syracuse University in 2024 (string with dollar formatting if present, e.g., "$750,000")
  - new_base_salary: their base salary at the destination institution (string with dollar formatting)
  - increase_amount: the dollar amount of the base salary increase (string with dollar formatting)
  - compensation_sources: an array of URLs confirming the Syracuse 2024 base salary, the new base salary, and/or the increase amount

incoming:
  - name: the full name of the person becoming Syracuse's Chancellor on July 1, 2026
  - previous_internal_role: their position at Syracuse prior to becoming Chancellor (e.g., "Provost", "Dean of ...")
  - historical_significance: a short phrase/sentence describing the historical significance of this internal promotion (specifically when was the last time Syracuse promoted a chancellor from within)
  - last_internal_promotion_year: the year of the last internal promotion before this one, if stated (as a string)
  - years_at_syracuse: approximately how many years they have worked at Syracuse (string, allow ranges or approximate wording)
  - military_service: short description of their military service background, if any
  - terminal_degree: the terminal degree they hold (e.g., "Ph.D. in ...", "J.D.", "M.F.A.") or a phrase stating they hold a terminal degree
  - background_sources: an array of URLs that confirm identity, previous_internal_role, historical_significance, years_at_syracuse, military_service, and/or terminal_degree

timeline:
  - effective_date: the effective date of the leadership transition (expect "July 1, 2026" or equivalent wording)
  - timeline_sources: an array of URLs that confirm the effective date(s)

context:
  - annual_budget: Syracuse University's approximate annual budget (string, allow rounded values like "$1.6B", "$1.5 billion")
  - chancellor_position_number: which numbered Chancellor the incoming leader will be (e.g., "13th")
  - context_sources: an array of URLs that confirm the budget and/or the Chancellor numbering

GENERAL RULES:
- Extract only what is explicitly stated in the answer.
- For all URL lists, include only valid, complete URLs (prepend http:// if protocol missing).
- If information is absent, set the string field to null or the list field to [].
"""


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def unite_urls(*lists: Optional[List[str]]) -> List[str]:
    seen = set()
    combined: List[str] = []
    for lst in lists:
        if not lst:
            continue
        for u in lst:
            if not u:
                continue
            if isinstance(u, str):
                v = u.strip()
                if v and v not in seen:
                    seen.add(v)
                    combined.append(v)
    return combined


async def add_verified_leaf(
    evaluator: Evaluator,
    parent_node,
    node_id: str,
    desc: str,
    claim: str,
    sources: Optional[List[str]] = None,
    critical: bool = True,
    additional_instruction: Optional[str] = None,
) -> None:
    leaf = evaluator.add_leaf(
        id=node_id,
        desc=desc,
        parent=parent_node,
        critical=critical,
    )
    await evaluator.verify(
        claim=claim,
        node=leaf,
        sources=sources,
        additional_instruction=additional_instruction or "None",
    )


# --------------------------------------------------------------------------- #
# Subtree builders                                                            #
# --------------------------------------------------------------------------- #
async def build_outgoing_subtree(
    evaluator: Evaluator,
    parent_node,
    outgoing: OutgoingLeader,
    timeline: TimelineInfo,
) -> None:
    outgoing_node = evaluator.add_parallel(
        id="Outgoing_Syracuse_Leader",
        desc="Identify and verify information about the leader leaving Syracuse University on July 1, 2026",
        parent=parent_node,
        critical=False,
    )

    # 1) Leader Identity (critical)
    identity_sources = unite_urls(
        outgoing.destination_sources,
        outgoing.tenure_sources,
        outgoing.compensation_sources,
        timeline.timeline_sources,
    )
    await add_verified_leaf(
        evaluator,
        outgoing_node,
        "Leader_Identity",
        "Correctly identify the name of the Syracuse University Chancellor who is leaving on July 1, 2026",
        claim=f"The Syracuse University Chancellor who is leaving on July 1, 2026 is {outgoing.name}.",
        sources=identity_sources,
        critical=True,
        additional_instruction="Confirm that the cited page(s) explicitly name the outgoing Syracuse University Chancellor associated with the July 1, 2026 transition.",
    )

    # 2) Destination Institution block
    dest_node = evaluator.add_parallel(
        id="Destination_Institution",
        desc="Identify where this leader is moving to",
        parent=outgoing_node,
        critical=False,
    )
    # Institution Name (critical)
    await add_verified_leaf(
        evaluator,
        dest_node,
        "Institution_Name",
        "Correctly identify the destination university",
        claim=f"The outgoing Syracuse Chancellor is moving to {outgoing.destination_institution}.",
        sources=outgoing.destination_sources,
        critical=True,
    )
    # New Position Title (critical)
    await add_verified_leaf(
        evaluator,
        dest_node,
        "New_Position_Title",
        "Correctly identify the position title at the new institution",
        claim=f"At {outgoing.destination_institution}, the outgoing Chancellor's new position title is {outgoing.new_position_title}.",
        sources=outgoing.destination_sources,
        critical=True,
        additional_instruction="Verify that the title appears on the cited page; minor wording differences are acceptable if they are equivalent.",
    )
    # Conference Affiliation (critical)
    await add_verified_leaf(
        evaluator,
        dest_node,
        "Conference_Affiliation",
        "Verify the athletic conference membership of the destination institution",
        claim=f"The athletic conference affiliation of {outgoing.destination_institution} is {outgoing.conference_affiliation}.",
        sources=outgoing.destination_sources,
        critical=True,
        additional_instruction="Confirm the NCAA Division I conference membership (e.g., ACC, Big Ten). Accept authoritative sources like official athletics sites or reputable news.",
    )
    # Reference URL Destination (critical)
    await add_verified_leaf(
        evaluator,
        dest_node,
        "Reference_URL_Destination",
        "Provide a valid reference URL confirming the destination institution and new role",
        claim=f"Sources confirm that {outgoing.name} is moving to {outgoing.destination_institution} to serve as {outgoing.new_position_title}.",
        sources=outgoing.destination_sources,
        critical=True,
        additional_instruction="At least one cited URL must clearly state both the destination institution and the new role/title.",
    )

    # 3) Syracuse Tenure Details
    tenure_node = evaluator.add_parallel(
        id="Syracuse_Tenure_Details",
        desc="Verify details about the leader's time at Syracuse University",
        parent=outgoing_node,
        critical=False,
    )
    # Start Year (critical)
    await add_verified_leaf(
        evaluator,
        tenure_node,
        "Start_Year",
        "Correctly identify the year this leader became Syracuse Chancellor",
        claim=f"{outgoing.name} became Syracuse University's Chancellor in {outgoing.start_year}.",
        sources=outgoing.tenure_sources,
        critical=True,
    )
    # Total Years (critical)
    await add_verified_leaf(
        evaluator,
        tenure_node,
        "Total_Years",
        "Correctly calculate the total number of years served as Chancellor",
        claim=f"{outgoing.name} served as Syracuse University's Chancellor for {outgoing.total_years_served} years.",
        sources=outgoing.tenure_sources,
        critical=True,
        additional_instruction="Allow approximate phrasing if the source implies the same duration; minor rounding is acceptable.",
    )
    # Reference URL Tenure (critical)
    await add_verified_leaf(
        evaluator,
        tenure_node,
        "Reference_URL_Tenure",
        "Provide a valid reference URL confirming tenure details",
        claim=f"Sources confirm that {outgoing.name} became Chancellor in {outgoing.start_year} and served about {outgoing.total_years_served} years.",
        sources=outgoing.tenure_sources,
        critical=True,
    )

    # 4) Compensation Comparison
    comp_node = evaluator.add_parallel(
        id="Compensation_Comparison",
        desc="Compare and verify compensation between old and new positions",
        parent=outgoing_node,
        critical=False,
    )
    # Syracuse Base Salary 2024 (critical)
    await add_verified_leaf(
        evaluator,
        comp_node,
        "Syracuse_Base_Salary",
        "Correctly identify the base salary at Syracuse University in 2024",
        claim=f"{outgoing.name}'s base salary at Syracuse University in 2024 was {outgoing.su_base_salary_2024}.",
        sources=outgoing.compensation_sources,
        critical=True,
        additional_instruction="Treat dollar formatting variations (commas, rounding) as acceptable if they are equivalent.",
    )
    # New Base Salary (critical)
    await add_verified_leaf(
        evaluator,
        comp_node,
        "New_Base_Salary",
        "Correctly identify the base salary at the new institution",
        claim=f"{outgoing.name}'s base salary at {outgoing.destination_institution} will be {outgoing.new_base_salary}.",
        sources=outgoing.compensation_sources,
        critical=True,
    )
    # Salary Increase (parallel aggregator)
    incr_node = evaluator.add_parallel(
        id="Salary_Increase",
        desc="Calculate or verify the base salary increase amount",
        parent=comp_node,
        critical=False,
    )
    # Increase Amount (critical)
    await add_verified_leaf(
        evaluator,
        incr_node,
        "Increase_Amount",
        "Correctly identify or calculate the dollar amount of base salary increase",
        claim=f"The base salary increase amount for {outgoing.name} is {outgoing.increase_amount}.",
        sources=outgoing.compensation_sources,
        critical=True,
        additional_instruction="If the page lists both old and new salaries, confirm the stated increase matches the difference (allow small rounding).",
    )
    # Reference URL Compensation (critical)
    await add_verified_leaf(
        evaluator,
        incr_node,
        "Reference_URL_Compensation",
        "Provide a valid reference URL confirming compensation figures",
        claim=f"Sources confirm the Syracuse 2024 base salary {outgoing.su_base_salary_2024}, the new base salary {outgoing.new_base_salary}, and an increase of {outgoing.increase_amount}.",
        sources=outgoing.compensation_sources,
        critical=True,
    )


async def build_incoming_subtree(
    evaluator: Evaluator,
    parent_node,
    incoming: IncomingLeader,
    timeline: TimelineInfo,
    context: InstitutionalContext,
) -> None:
    incoming_node = evaluator.add_parallel(
        id="Incoming_Syracuse_Leader",
        desc="Identify and verify information about the new Syracuse University Chancellor starting July 1, 2026",
        parent=parent_node,
        critical=False,
    )

    identity_sources = unite_urls(
        incoming.background_sources,
        timeline.timeline_sources,
        context.context_sources,
    )

    # New Leader Identity (critical)
    await add_verified_leaf(
        evaluator,
        incoming_node,
        "New_Leader_Identity",
        "Correctly identify the name of the incoming Syracuse University Chancellor",
        claim=f"The person becoming Syracuse University's Chancellor on July 1, 2026 is {incoming.name}.",
        sources=identity_sources,
        critical=True,
    )

    # Previous Internal Role (critical)
    await add_verified_leaf(
        evaluator,
        incoming_node,
        "Previous_Internal_Role",
        "Correctly identify the previous position this person held at Syracuse University before becoming Chancellor",
        claim=f"Before becoming Chancellor, {incoming.name} served as {incoming.previous_internal_role} at Syracuse University.",
        sources=incoming.background_sources,
        critical=True,
        additional_instruction="Accept minor wording differences if the role is clearly the same.",
    )

    # Historical Significance (critical)
    hist_phrase = incoming.historical_significance or ""
    if incoming.last_internal_promotion_year:
        hist_phrase = hist_phrase or f"first internal promotion since {incoming.last_internal_promotion_year}"
    await add_verified_leaf(
        evaluator,
        incoming_node,
        "Historical_Significance",
        "Verify the historical significance of this internal promotion",
        claim=f"This appointment is Syracuse's first chancellor promoted from within since {incoming.last_internal_promotion_year}. {hist_phrase}",
        sources=incoming.background_sources,
        critical=True,
        additional_instruction="The source should explicitly reference either the 'first since YEAR' notion or otherwise clearly indicate the long gap since the previous internal promotion.",
    )

    # Professional Background block
    prof_node = evaluator.add_parallel(
        id="Professional_Background",
        desc="Verify key aspects of the incoming leader's professional background",
        parent=incoming_node,
        critical=False,
    )
    # Years at Syracuse (critical)
    await add_verified_leaf(
        evaluator,
        prof_node,
        "Years_At_Syracuse",
        "Correctly identify approximately how many years this person has worked at Syracuse University",
        claim=f"{incoming.name} has worked at Syracuse University for approximately {incoming.years_at_syracuse} years.",
        sources=incoming.background_sources,
        critical=True,
        additional_instruction="Allow approximate values and minor rounding (±1–2 years).",
    )
    # Military Service (critical)
    await add_verified_leaf(
        evaluator,
        prof_node,
        "Military_Service",
        "Verify military service background",
        claim=f"{incoming.name} has military service background: {incoming.military_service}.",
        sources=incoming.background_sources,
        critical=True,
        additional_instruction="Confirm service branch or roles as described; concise summaries are acceptable if consistent with the source.",
    )
    # Terminal Degree (critical)
    await add_verified_leaf(
        evaluator,
        prof_node,
        "Terminal_Degree",
        "Verify possession of a terminal degree (PhD or equivalent)",
        claim=f"{incoming.name} holds a terminal degree (PhD or equivalent): {incoming.terminal_degree}.",
        sources=incoming.background_sources,
        critical=True,
        additional_instruction="Treat commonly recognized terminal degrees (e.g., Ph.D., J.D., M.D., Ed.D., D.M.A., M.F.A.) as terminal.",
    )
    # Reference URL Background (critical)
    await add_verified_leaf(
        evaluator,
        prof_node,
        "Reference_URL_Background",
        "Provide a valid reference URL confirming professional background details",
        claim=f"Sources confirm that {incoming.name} has ~{incoming.years_at_syracuse} years at Syracuse, has military service ({incoming.military_service}), and holds a terminal degree ({incoming.terminal_degree}).",
        sources=incoming.background_sources,
        critical=True,
    )


async def build_timeline_subtree(
    evaluator: Evaluator,
    parent_node,
    timeline: TimelineInfo,
) -> None:
    timeline_node = evaluator.add_parallel(
        id="Transition_Timeline",
        desc="Verify the timing of both leadership transitions",
        parent=parent_node,
        critical=False,
    )
    # Effective Date (critical)
    await add_verified_leaf(
        evaluator,
        timeline_node,
        "Effective_Date",
        "Correctly identify that both transitions occur on July 1, 2026",
        claim="Both the outgoing and incoming leadership changes at Syracuse University take effect on July 1, 2026.",
        sources=timeline.timeline_sources,
        critical=True,
        additional_instruction='Accept equivalent phrasing like "effective July 1, 2026" or "beginning July 1, 2026".',
    )
    # Reference URL Timeline (critical)
    await add_verified_leaf(
        evaluator,
        timeline_node,
        "Reference_URL_Timeline",
        "Provide a valid reference URL confirming the transition date",
        claim=f"Sources confirm the leadership transition effective date is {timeline.effective_date or 'July 1, 2026'}.",
        sources=timeline.timeline_sources,
        critical=True,
    )


async def build_context_subtree(
    evaluator: Evaluator,
    parent_node,
    context: InstitutionalContext,
) -> None:
    context_node = evaluator.add_parallel(
        id="Institutional_Context",
        desc="Provide relevant context about Syracuse University as an institution",
        parent=parent_node,
        critical=False,
    )
    # Annual Budget (critical)
    await add_verified_leaf(
        evaluator,
        context_node,
        "Annual_Budget",
        "Correctly identify Syracuse University's approximate annual budget",
        claim=f"Syracuse University's approximate annual budget is {context.annual_budget}.",
        sources=context.context_sources,
        critical=True,
        additional_instruction="Allow approximate values and rounding (e.g., $1.5B vs $1.6B) if they are reasonably close and clearly marked as approximate.",
    )
    # Chancellor Position Number (critical)
    await add_verified_leaf(
        evaluator,
        context_node,
        "Chancellor_Position_Number",
        "Correctly identify that the incoming leader will be Syracuse's 13th Chancellor",
        claim=f"The incoming leader will be Syracuse University's {context.chancellor_position_number} Chancellor.",
        sources=context.context_sources,
        critical=True,
        additional_instruction="Treat ordinal variations (e.g., '13', '13th') as equivalent.",
    )
    # Reference URL Context (critical)
    await add_verified_leaf(
        evaluator,
        context_node,
        "Reference_URL_Context",
        "Provide a valid reference URL confirming institutional context",
        claim=f"Sources confirm that Syracuse's budget is around {context.annual_budget} and that the incoming leader will be the {context.chancellor_position_number} Chancellor.",
        sources=context.context_sources,
        critical=True,
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
    model: str = "o4-mini",
) -> Dict[str, Any]:
    # Initialize evaluator (root: parallel aggregation as per rubric)
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

    # Extract structured transition info
    extraction = await evaluator.extract(
        prompt=prompt_extract_transition(),
        template_class=TransitionExtraction,
        extraction_name="leadership_transition_extraction",
    )

    # Build tree: Leadership_Transition_Research (root children per rubric)
    research_root = evaluator.add_parallel(
        id="Leadership_Transition_Research",
        desc="Verify comprehensive research on concurrent university leadership transitions occurring on July 1, 2026, involving Syracuse University",
        parent=root,
        critical=False,
    )

    # Subtrees
    await build_outgoing_subtree(evaluator, research_root, extraction.outgoing, extraction.timeline)
    await build_incoming_subtree(evaluator, research_root, extraction.incoming, extraction.timeline, extraction.context)
    await build_timeline_subtree(evaluator, research_root, extraction.timeline)
    await build_context_subtree(evaluator, research_root, extraction.context)

    # Return summary
    return evaluator.get_summary()