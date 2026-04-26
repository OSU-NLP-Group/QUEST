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
TASK_ID = "permit_rolling_4mo_quota64"
TASK_DESCRIPTION = (
    "I am planning a hiking trip to an iconic geological formation and need to identify the specific "
    "permit-required hiking destination that matches the following criteria: The advance lottery for permits operates "
    "on a rolling 4-month advance schedule, where applications submitted during any calendar month are for hiking dates "
    "occurring 4 months later. The lottery drawing occurs on the 1st day of the month following the application period "
    "at 9:00 AM Mountain Time. The permit system enforces a strict daily visitor quota of exactly 64 people, with a "
    "maximum group size limit of 6 people. The fee structure requires a $6 non-refundable application fee plus a $7 "
    "per-person recreation fee if selected. What is the name of this hiking destination, and what federal agency manages "
    "the permit area?"
)

# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class PermitExtraction(BaseModel):
    """
    Extracted information from the agent's answer for the permit destination and supporting sources by topic.
    """
    destination_name: Optional[str] = None
    managing_agency: Optional[str] = None

    # General cited sources for the destination (e.g., main permit page)
    general_sources: List[str] = Field(default_factory=list)

    # Topic-specific sources (URLs explicitly mentioned in the answer)
    application_period_sources: List[str] = Field(default_factory=list)
    lottery_draw_timing_sources: List[str] = Field(default_factory=list)
    advance_lottery_quota_sources: List[str] = Field(default_factory=list)
    confirmation_deadline_sources: List[str] = Field(default_factory=list)

    daily_quota_sources: List[str] = Field(default_factory=list)
    group_size_sources: List[str] = Field(default_factory=list)
    day_use_sources: List[str] = Field(default_factory=list)
    wilderness_sources: List[str] = Field(default_factory=list)

    application_fee_sources: List[str] = Field(default_factory=list)
    recreation_fee_sources: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_permit_info() -> str:
    """
    Build the extraction prompt for pulling destination name, managing agency, and topic-specific source URLs.
    """
    return (
        "Extract the following information from the provided answer text:\n"
        "1) destination_name: The specific hiking destination name (e.g., the exact trail/formation/permit area name).\n"
        "2) managing_agency: The managing federal agency of the permit area (e.g., Bureau of Land Management (BLM), National Park Service (NPS)).\n"
        "3) general_sources: All general URLs cited that describe the destination or permit system.\n"
        "4) application_period_sources: URLs that describe the monthly application period for dates occurring 4 months later.\n"
        "5) lottery_draw_timing_sources: URLs that describe the lottery draw occurring on the 1st day of the following month at 9:00 AM Mountain Time.\n"
        "6) advance_lottery_quota_sources: URLs that describe the advance lottery awarding permits to a maximum of 48 people per day.\n"
        "7) confirmation_deadline_sources: URLs that describe the requirement to confirm and pay by the 15th of the month following the lottery.\n"
        "8) daily_quota_sources: URLs that describe the strict daily visitor quota of exactly 64 people.\n"
        "9) group_size_sources: URLs that describe the maximum group size of 6 people.\n"
        "10) day_use_sources: URLs that describe day-use only and no overnight camping.\n"
        "11) wilderness_sources: URLs that describe the destination being within federally managed wilderness land.\n"
        "12) application_fee_sources: URLs that describe a $6 non-refundable application fee.\n"
        "13) recreation_fee_sources: URLs that describe a $7 per-person recreation fee charged only if selected.\n\n"
        "IMPORTANT:\n"
        "- Extract only information explicitly present in the answer text. Do not invent or infer missing details.\n"
        "- For each URL list, include only valid URLs explicitly mentioned. If none are present for a topic, return an empty list.\n"
        "- If destination_name or managing_agency is missing, set them to null.\n"
    )


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def combine_sources(specific: List[str], general: List[str]) -> List[str]:
    """
    Combine topic-specific sources with general sources, remove duplicates, keep order stable.
    """
    seen = set()
    out: List[str] = []
    for url in (specific + general):
        if not url:
            continue
        if url not in seen:
            seen.add(url)
            out.append(url)
    return out


# --------------------------------------------------------------------------- #
# Verification logic (tree construction)                                      #
# --------------------------------------------------------------------------- #
async def build_and_verify_tree(evaluator: Evaluator, info: PermitExtraction) -> None:
    """
    Build the verification tree following the rubric and run claim verifications.
    All non-leaf nodes and leaves under the main task node are critical to match rubric requirements.
    """
    # Top-level task node (critical), placed under evaluator.root (which is non-critical by design)
    task_root = evaluator.add_parallel(
        id="task_root",
        desc="The provided hiking destination correctly satisfies all specified permit system constraints",
        parent=evaluator.root,
        critical=True
    )

    # 1) Destination answer provided (critical parallel)
    dest_provided_node = evaluator.add_parallel(
        id="destination_answer_provided",
        desc="Both the hiking destination name and the managing federal agency are provided",
        parent=task_root,
        critical=True
    )

    # 1.1 Destination name provided (critical leaf via custom check)
    evaluator.add_custom_node(
        result=bool(info.destination_name and info.destination_name.strip()),
        id="destination_name_provided",
        desc="A specific hiking destination name is provided",
        parent=dest_provided_node,
        critical=True
    )

    # 1.2 Managing agency provided (critical leaf via custom check)
    evaluator.add_custom_node(
        result=bool(info.managing_agency and info.managing_agency.strip()),
        id="managing_agency_provided",
        desc="The managing federal agency is identified",
        parent=dest_provided_node,
        critical=True
    )

    # 2) Lottery system constraints (critical parallel)
    lottery_node = evaluator.add_parallel(
        id="lottery_system_constraints",
        desc="The identified destination's lottery system matches all specified timing and process requirements",
        parent=task_root,
        critical=True
    )

    # 2.1 Application period timing
    app_period_leaf = evaluator.add_leaf(
        id="application_period_timing",
        desc="Applications are submitted during a calendar month for hiking dates 4 months in the future",
        parent=lottery_node,
        critical=True
    )
    claim_app_period = (
        f"For {info.destination_name or 'the destination'}, the advance permit lottery accepts applications "
        "during each calendar month for hiking dates occurring 4 months later."
    )
    await evaluator.verify(
        claim=claim_app_period,
        node=app_period_leaf,
        sources=combine_sources(info.application_period_sources, info.general_sources),
        additional_instruction=(
            "Confirm a rolling schedule: applications in a given calendar month are for dates 4 months later."
        )
    )

    # 2.2 Lottery draw timing
    draw_timing_leaf = evaluator.add_leaf(
        id="lottery_draw_timing",
        desc="The lottery drawing occurs on the 1st day of the following month at 9:00 AM Mountain Time",
        parent=lottery_node,
        critical=True
    )
    claim_draw_timing = (
        f"For {info.destination_name or 'the destination'}, the lottery drawing occurs on the 1st day of the month "
        "following the application period at 9:00 AM Mountain Time."
    )
    await evaluator.verify(
        claim=claim_draw_timing,
        node=draw_timing_leaf,
        sources=combine_sources(info.lottery_draw_timing_sources, info.general_sources),
        additional_instruction="Verify both the date (1st day of following month) and time (9:00 AM MT)."
    )

    # 2.3 Advance lottery quota (48 per day from advance lottery)
    adv_quota_leaf = evaluator.add_leaf(
        id="advance_lottery_quota",
        desc="The advance lottery awards permits to a maximum of 48 people per day",
        parent=lottery_node,
        critical=True
    )
    claim_adv_quota = (
        f"For {info.destination_name or 'the destination'}, the advance lottery allocates up to 48 people per day."
    )
    await evaluator.verify(
        claim=claim_adv_quota,
        node=adv_quota_leaf,
        sources=combine_sources(info.advance_lottery_quota_sources, info.general_sources),
        additional_instruction="Some systems split total daily quota; verify that the advance lottery portion is 48."
    )

    # 2.4 Confirmation deadline
    confirm_deadline_leaf = evaluator.add_leaf(
        id="confirmation_deadline",
        desc="Permits must be confirmed and paid by the 15th of the month following the lottery",
        parent=lottery_node,
        critical=True
    )
    claim_confirm_deadline = (
        f"For {info.destination_name or 'the destination'}, selected applicants must confirm and pay by the 15th of "
        "the month following the lottery."
    )
    await evaluator.verify(
        claim=claim_confirm_deadline,
        node=confirm_deadline_leaf,
        sources=combine_sources(info.confirmation_deadline_sources, info.general_sources),
        additional_instruction="Confirm the specific deadline wording: by the 15th of the next month."
    )

    # 3) Capacity and use constraints (critical parallel)
    capacity_node = evaluator.add_parallel(
        id="capacity_and_use_constraints",
        desc="The identified destination's capacity limits and use restrictions match all specified requirements",
        parent=task_root,
        critical=True
    )

    # 3.1 Daily visitor quota = 64 people
    daily_quota_leaf = evaluator.add_leaf(
        id="daily_visitor_quota",
        desc="The daily visitor quota is exactly 64 people",
        parent=capacity_node,
        critical=True
    )
    claim_daily_quota = (
        f"The daily visitor quota for {info.destination_name or 'the destination'} is exactly 64 people."
    )
    await evaluator.verify(
        claim=claim_daily_quota,
        node=daily_quota_leaf,
        sources=combine_sources(info.daily_quota_sources, info.general_sources),
        additional_instruction="The page may list components (e.g., advance + daily lottery); confirm the total equals 64."
    )

    # 3.2 Maximum group size = 6 people
    group_size_leaf = evaluator.add_leaf(
        id="maximum_group_size",
        desc="The maximum group size is 6 people",
        parent=capacity_node,
        critical=True
    )
    claim_group_size = (
        f"The maximum group size for {info.destination_name or 'the destination'} is 6 people."
    )
    await evaluator.verify(
        claim=claim_group_size,
        node=group_size_leaf,
        sources=combine_sources(info.group_size_sources, info.general_sources),
        additional_instruction="Verify explicit group size limit equals six."
    )

    # 3.3 Day-use only, no overnight camping
    day_use_leaf = evaluator.add_leaf(
        id="day_use_only",
        desc="The permit is valid for day-use only with no overnight camping allowed",
        parent=capacity_node,
        critical=True
    )
    claim_day_use = (
        f"The permit for {info.destination_name or 'the destination'} is valid for day-use only; "
        "overnight camping is not allowed."
    )
    await evaluator.verify(
        claim=claim_day_use,
        node=day_use_leaf,
        sources=combine_sources(info.day_use_sources, info.general_sources),
        additional_instruction="Look for explicit language indicating day-use only and prohibition of overnight camping."
    )

    # 3.4 Located within federally managed wilderness land
    wilderness_leaf = evaluator.add_leaf(
        id="federal_wilderness_land",
        desc="The destination is located within federally managed wilderness land",
        parent=capacity_node,
        critical=True
    )
    claim_wilderness = (
        f"{info.destination_name or 'The destination'} is located within federally managed wilderness land."
    )
    await evaluator.verify(
        claim=claim_wilderness,
        node=wilderness_leaf,
        sources=combine_sources(info.wilderness_sources, info.general_sources),
        additional_instruction="Confirm the area is designated as 'Wilderness' under federal management (e.g., BLM/NPS)."
    )

    # 4) Fee structure (critical parallel)
    fee_node = evaluator.add_parallel(
        id="fee_structure",
        desc="The identified destination's fee structure matches all specified requirements",
        parent=task_root,
        critical=True
    )

    # 4.1 Application fee is $6 non-refundable
    app_fee_leaf = evaluator.add_leaf(
        id="application_fee_amount",
        desc="The application fee is $6 and is non-refundable",
        parent=fee_node,
        critical=True
    )
    claim_app_fee = (
        f"The application fee for {info.destination_name or 'the destination'} is $6 and is non-refundable."
    )
    await evaluator.verify(
        claim=claim_app_fee,
        node=app_fee_leaf,
        sources=combine_sources(info.application_fee_sources, info.general_sources),
        additional_instruction="Confirm exact dollar amount ($6) and that the application fee is non-refundable."
    )

    # 4.2 Recreation fee is $7 per person if selected
    rec_fee_leaf = evaluator.add_leaf(
        id="recreation_fee_amount",
        desc="The per-person recreation fee is $7 and is charged only if the permit is awarded",
        parent=fee_node,
        critical=True
    )
    claim_rec_fee = (
        f"The per-person recreation fee for {info.destination_name or 'the destination'} is $7, "
        "charged only if the permit is awarded."
    )
    await evaluator.verify(
        claim=claim_rec_fee,
        node=rec_fee_leaf,
        sources=combine_sources(info.recreation_fee_sources, info.general_sources),
        additional_instruction="Confirm exact dollar amount ($7) and that it is charged only upon award/selection."
    )

    # Record compact custom info for easier debugging
    evaluator.add_custom_info(
        info={
            "destination_name": info.destination_name,
            "managing_agency": info.managing_agency,
            "general_sources_count": len(info.general_sources)
        },
        info_type="extracted_meta",
        info_name="extracted_destination_meta"
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
    """
    Evaluate an agent's answer against the permit destination rubric.
    Returns a structured summary with the verification tree and final score.
    """
    # Initialize evaluator (framework root is non-critical by design)
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
    extracted_info = await evaluator.extract(
        prompt=prompt_extract_permit_info(),
        template_class=PermitExtraction,
        extraction_name="permit_extraction"
    )

    # Build verification tree and run checks
    await build_and_verify_tree(evaluator, extracted_info)

    # Return evaluation summary
    return evaluator.get_summary()