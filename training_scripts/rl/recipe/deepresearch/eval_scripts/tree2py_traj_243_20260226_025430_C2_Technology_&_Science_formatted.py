import asyncio
import logging
from typing import Any, Optional, List, Dict

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy

# -----------------------------------------------------------------------------
# Task-specific constants
# -----------------------------------------------------------------------------
TASK_ID = "fcc_outage_reporting_timeframes"
TASK_DESCRIPTION = (
    "According to the Federal Communications Commission (FCC) regulations for telecommunications outage reporting, "
    "identify the three mandatory notification timeframes for the following scenarios:\n\n"
    "1. The timeframe within which wireline, cable, satellite, wireless, and Signaling System 7 (SS7) providers must "
    "submit a preliminary notification to the Network Outage Reporting System (NORS) after determining that an outage is reportable.\n\n"
    "2. The timeframe within which these same providers must submit an initial outage report to NORS after discovering the outage.\n\n"
    "3. The timeframe within which covered 911 service providers must notify affected Public Safety Answering Points (PSAPs) after discovering an outage that affects a 911 call center.\n\n"
    "For each scenario, provide the specific timeframe mandated by the FCC and include a reference URL from the FCC's official website or other authoritative source that confirms this requirement."
)

# Expected requirements (ground truth)
EXPECTED_REQUIREMENTS = {
    "preliminary_nors": "120 minutes (two hours) after determining the outage is reportable",
    "initial_report": "72 hours (three calendar days) after discovering the outage",
    "psap_notification": "30 minutes after discovering an outage affecting a 911 call center",
}

# -----------------------------------------------------------------------------
# Data models for extraction
# -----------------------------------------------------------------------------
class OutageReportingExtraction(BaseModel):
    preliminary_notification_timeframe: Optional[str] = None
    preliminary_notification_sources: List[str] = Field(default_factory=list)

    initial_report_timeframe: Optional[str] = None
    initial_report_sources: List[str] = Field(default_factory=list)

    psap_notification_timeframe: Optional[str] = None
    psap_notification_sources: List[str] = Field(default_factory=list)


# -----------------------------------------------------------------------------
# Extraction prompt
# -----------------------------------------------------------------------------
def prompt_extract_outage_requirements() -> str:
    return """
    Extract the specific timeframes and any reference URLs provided in the answer for the three FCC outage reporting scenarios below.
    You MUST extract the timeframe text exactly as stated in the answer (e.g., "120 minutes", "two hours", "72 hours", "three calendar days", "30 minutes").
    Also extract all reference URLs explicitly mentioned in the answer for each scenario. Only include valid URLs.

    Scenarios:
    1) preliminary_notification_timeframe: The preliminary notification to NORS for wireline, cable, satellite, wireless, and SS7 providers after determining that an outage is reportable.
       preliminary_notification_sources: All URLs that the answer cites to support this preliminary notification timeframe.

    2) initial_report_timeframe: The deadline for submitting the initial outage report to NORS after discovering the outage.
       initial_report_sources: All URLs that the answer cites to support this initial report timeframe.

    3) psap_notification_timeframe: The notification timeframe for covered 911 service providers to notify affected PSAPs after discovering an outage affecting a 911 call center.
       psap_notification_sources: All URLs that the answer cites to support this PSAP notification timeframe.

    Rules:
    - If the answer uses synonymous phrases (e.g., "two hours" vs "120 minutes", "three calendar days" vs "72 hours"), extract the exact phrase used in the answer.
    - For URLs, include only actual URLs present in the answer (plain URLs or markdown links). Do not invent or infer URLs.
    - If any timeframe is missing in the answer, set that timeframe to null.
    - If no URLs are cited for a scenario, return an empty list for that scenario's sources.
    """


# -----------------------------------------------------------------------------
# Helper functions
# -----------------------------------------------------------------------------
def has_valid_urls(urls: List[str]) -> bool:
    """Basic validity check: at least one non-empty HTTP(S) URL."""
    return any(isinstance(u, str) and u.strip().startswith(("http://", "https://")) for u in urls)


# -----------------------------------------------------------------------------
# Verification subtrees
# -----------------------------------------------------------------------------
async def verify_preliminary_notification(
    evaluator: Evaluator,
    parent_node,
    extraction: OutageReportingExtraction,
) -> None:
    """
    Build verification nodes for the preliminary NORS notification timeframe (120 minutes) and its supporting reference(s).
    """
    group_node = evaluator.add_parallel(
        id="Initial_Notification_Timeframe",
        desc="Correctly states the preliminary NORS notification timeframe for wireline, cable, satellite, wireless, and SS7 providers, with supporting reference",
        parent=parent_node,
        critical=False,
    )

    # Leaf: timeframe correctness (simple check using the answer)
    tf_leaf = evaluator.add_leaf(
        id="120_Minute_Requirement",
        desc="Specifies that the preliminary NORS notification must be submitted within 120 minutes after determining the outage is reportable",
        parent=group_node,
        critical=True,
    )
    prelim_tf = extraction.preliminary_notification_timeframe or ""
    claim_tf = (
        f"The timeframe '{prelim_tf}' stated in the answer for the preliminary NORS notification is equivalent to "
        f"'120 minutes' (two hours) after determining the outage is reportable."
    )
    await evaluator.verify(
        claim=claim_tf,
        node=tf_leaf,
        additional_instruction=(
            "Judge equivalence generously: treat '120 minutes', 'two hours', '2 hours', "
            "'within two hours', and similar phrasings as equivalent. "
            "If the answer does not clearly provide such an equivalent timeframe, mark incorrect."
        ),
    )

    # Custom existence node: at least one reference URL provided
    ref_exist_node = evaluator.add_custom_node(
        result=has_valid_urls(extraction.preliminary_notification_sources),
        id="Preliminary_Notification_Reference_Provided",
        desc="At least one reference URL is provided for the preliminary NORS notification timeframe",
        parent=group_node,
        critical=True,
    )

    # Leaf: reference support via URLs
    ref_leaf = evaluator.add_leaf(
        id="Preliminary_Notification_Reference",
        desc="Provides a valid reference URL from the FCC's official website or other authoritative source confirming the 120-minute requirement",
        parent=group_node,
        critical=True,
    )
    claim_ref = (
        "This page explicitly confirms that the preliminary NORS notification must be submitted within 120 minutes "
        "(two hours) after determining the outage is reportable."
    )
    await evaluator.verify(
        claim=claim_ref,
        node=ref_leaf,
        sources=extraction.preliminary_notification_sources,
        additional_instruction=(
            "Only mark supported if the page explicitly states or clearly implies the 120-minute (two hours) preliminary notification rule. "
            "Prefer authoritative sources (e.g., fcc.gov, ecfr.gov, federalregister.gov). "
            "Minor wording variations are acceptable as long as the 120 minutes requirement is unambiguous."
        ),
    )


async def verify_initial_report(
    evaluator: Evaluator,
    parent_node,
    extraction: OutageReportingExtraction,
) -> None:
    """
    Build verification nodes for the initial outage report deadline (72 hours / three calendar days) and its supporting reference(s).
    """
    group_node = evaluator.add_parallel(
        id="Initial_Report_Deadline",
        desc="Correctly states the initial outage report submission deadline for standard providers, with supporting reference",
        parent=parent_node,
        critical=False,
    )

    # Leaf: timeframe correctness
    tf_leaf = evaluator.add_leaf(
        id="72_Hour_Requirement",
        desc="Specifies that the initial outage report must be submitted within 72 hours (three calendar days) after discovering the outage",
        parent=group_node,
        critical=True,
    )
    init_tf = extraction.initial_report_timeframe or ""
    claim_tf = (
        f"The timeframe '{init_tf}' stated in the answer for the initial outage report is equivalent to '72 hours' "
        f"(three calendar days) after discovering the outage."
    )
    await evaluator.verify(
        claim=claim_tf,
        node=tf_leaf,
        additional_instruction=(
            "Judge equivalence generously: treat '72 hours', 'three calendar days', "
            "'within 72 hours', and similar phrasings as equivalent. "
            "If the answer does not clearly provide such an equivalent timeframe, mark incorrect."
        ),
    )

    # Custom existence node: at least one reference URL provided
    ref_exist_node = evaluator.add_custom_node(
        result=has_valid_urls(extraction.initial_report_sources),
        id="Initial_Report_Reference_Provided",
        desc="At least one reference URL is provided for the initial outage report timeframe",
        parent=group_node,
        critical=True,
    )

    # Leaf: reference support via URLs
    ref_leaf = evaluator.add_leaf(
        id="Initial_Report_Reference",
        desc="Provides a valid reference URL from the FCC's official website or other authoritative source confirming the 72-hour requirement",
        parent=group_node,
        critical=True,
    )
    claim_ref = (
        "This page explicitly confirms that the initial outage report must be submitted within 72 hours "
        "(three calendar days) after discovering the outage."
    )
    await evaluator.verify(
        claim=claim_ref,
        node=ref_leaf,
        sources=extraction.initial_report_sources,
        additional_instruction=(
            "Only mark supported if the page explicitly states or clearly implies the 72-hour (three calendar days) initial report rule. "
            "Prefer authoritative sources (e.g., fcc.gov, ecfr.gov, federalregister.gov). "
            "Minor wording variations are acceptable as long as the 72 hours requirement is unambiguous."
        ),
    )


async def verify_psap_notification(
    evaluator: Evaluator,
    parent_node,
    extraction: OutageReportingExtraction,
) -> None:
    """
    Build verification nodes for the PSAP notification timeframe (30 minutes) and its supporting reference(s).
    """
    group_node = evaluator.add_parallel(
        id="PSAP_911_Notification",
        desc="Correctly states the PSAP notification requirement for covered 911 service providers, with supporting reference",
        parent=parent_node,
        critical=False,
    )

    # Leaf: timeframe correctness
    tf_leaf = evaluator.add_leaf(
        id="30_Minute_Requirement",
        desc="Specifies that covered 911 service providers must notify affected PSAPs within 30 minutes after discovering an outage affecting a 911 call center",
        parent=group_node,
        critical=True,
    )
    psap_tf = extraction.psap_notification_timeframe or ""
    claim_tf = (
        f"The timeframe '{psap_tf}' stated in the answer for PSAP notification is equivalent to '30 minutes' "
        f"after discovering an outage affecting a 911 call center."
    )
    await evaluator.verify(
        claim=claim_tf,
        node=tf_leaf,
        additional_instruction=(
            "Judge equivalence generously: treat '30 minutes', 'within 30 minutes', and similar phrasings as equivalent. "
            "If the answer does not clearly provide such an equivalent timeframe, mark incorrect."
        ),
    )

    # Custom existence node: at least one reference URL provided
    ref_exist_node = evaluator.add_custom_node(
        result=has_valid_urls(extraction.psap_notification_sources),
        id="PSAP_Notification_Reference_Provided",
        desc="At least one reference URL is provided for the PSAP notification timeframe",
        parent=group_node,
        critical=True,
    )

    # Leaf: reference support via URLs
    ref_leaf = evaluator.add_leaf(
        id="PSAP_Notification_Reference",
        desc="Provides a valid reference URL from the FCC's official website or other authoritative source confirming the 30-minute requirement",
        parent=group_node,
        critical=True,
    )
    claim_ref = (
        "This page explicitly confirms that covered 911 service providers must notify affected PSAPs within 30 minutes "
        "after discovering an outage affecting a 911 call center."
    )
    await evaluator.verify(
        claim=claim_ref,
        node=ref_leaf,
        sources=extraction.psap_notification_sources,
        additional_instruction=(
            "Only mark supported if the page explicitly states or clearly implies the 30-minute PSAP notification rule. "
            "Prefer authoritative sources (e.g., fcc.gov, ecfr.gov, federalregister.gov). "
            "Minor wording variations are acceptable as long as the 30 minutes requirement is unambiguous."
        ),
    )


# -----------------------------------------------------------------------------
# Main evaluation entry point
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
    Evaluate an answer for FCC outage reporting timeframes and references.
    """
    # Initialize evaluator with a parallel root (non-critical root to allow mixed children)
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

    # Extract structured info from the answer
    extraction = await evaluator.extract(
        prompt=prompt_extract_outage_requirements(),
        template_class=OutageReportingExtraction,
        extraction_name="outage_reporting_extraction",
    )

    # Top-level grouping node for the rubric
    top_node = evaluator.add_parallel(
        id="FCC_Outage_Reporting_Requirements",
        desc="Correctly identifies the FCC's telecommunications outage reporting timeframes and requirements for all three scenarios",
        parent=root,
        critical=False,  # Keep non-critical to comply with framework constraint (critical parent requires all children critical)
    )

    # Add ground truth info
    evaluator.add_ground_truth({
        "expected_preliminary_nors_timeframe": EXPECTED_REQUIREMENTS["preliminary_nors"],
        "expected_initial_report_timeframe": EXPECTED_REQUIREMENTS["initial_report"],
        "expected_psap_notification_timeframe": EXPECTED_REQUIREMENTS["psap_notification"],
    }, gt_type="expected_requirements")

    # Build verification subtrees
    await verify_preliminary_notification(evaluator, top_node, extraction)
    await verify_initial_report(evaluator, top_node, extraction)
    await verify_psap_notification(evaluator, top_node, extraction)

    # Return structured summary
    return evaluator.get_summary()