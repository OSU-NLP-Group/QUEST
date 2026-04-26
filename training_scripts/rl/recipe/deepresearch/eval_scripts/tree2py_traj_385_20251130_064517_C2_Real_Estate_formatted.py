import asyncio
import logging
from typing import Any, Optional, List, Dict

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy

# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "broker_ce_compare"
TASK_DESCRIPTION = (
    "You are considering becoming a real estate broker and want to compare the continuing "
    "education requirements between Washington D.C. and Maryland to help decide where to establish your practice. "
    "Research and provide the following information for each jurisdiction:\n\n"
    "For Washington D.C.:\n"
    "- The total number of continuing education hours required for real estate brokers\n"
    "- How frequently (the renewal period) the continuing education must be completed\n"
    "- A reference URL from an official or authoritative source\n\n"
    "For Maryland:\n"
    "- The total number of continuing education hours required for real estate brokers\n"
    "- The timeframe within which a licensee must complete additional training after assuming a broker, branch office manager, or team leader role\n"
    "- A reference URL from an official or authoritative source"
)

# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class DCRequirements(BaseModel):
    total_hours: Optional[str] = None
    renewal_period: Optional[str] = None
    source_urls: List[str] = Field(default_factory=list)


class MDRequirements(BaseModel):
    total_hours: Optional[str] = None
    broker_training_timeframe: Optional[str] = None
    source_urls: List[str] = Field(default_factory=list)


class CEComparisonExtraction(BaseModel):
    dc: Optional[DCRequirements] = None
    md: Optional[MDRequirements] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_ce_requirements() -> str:
    return """
    Extract the continuing education requirement details for Washington D.C. and Maryland as explicitly stated in the answer.

    For Washington D.C. (dc):
    - dc.total_hours: The total number of continuing education hours required for real estate brokers (as a string, e.g., "15 hours")
    - dc.renewal_period: How frequently (the renewal period) the continuing education must be completed (as a string, e.g., "every 2 years" or "biennially")
    - dc.source_urls: A list of one or more reference URLs explicitly provided in the answer that document the Washington D.C. real estate broker continuing education requirements

    For Maryland (md):
    - md.total_hours: The total number of continuing education hours required for real estate brokers (as a string)
    - md.broker_training_timeframe: The timeframe within which a licensee must complete additional training after assuming a broker, branch office manager, or team leader role (as a string, e.g., "within 90 days")
    - md.source_urls: A list of one or more reference URLs explicitly provided in the answer that document the Maryland real estate broker continuing education requirements and/or the specified timeframe

    Rules:
    - Return values exactly as written in the answer; do not infer or invent.
    - If any field is missing in the answer, set it to null (for strings) or an empty list (for URLs).
    - For URLs: extract only valid URLs explicitly present (including markdown links). If a URL is missing a protocol, prepend "http://".
    """


# --------------------------------------------------------------------------- #
# Jurisdiction verification helpers                                           #
# --------------------------------------------------------------------------- #
async def verify_dc_requirements(evaluator: Evaluator, parent_node, dc: Optional[DCRequirements]) -> None:
    # Parent node for Washington D.C. (critical, parallel aggregation)
    dc_node = evaluator.add_parallel(
        id="Washington_DC_Requirements",
        desc="Continuing education requirements for real estate brokers in Washington D.C.",
        parent=parent_node,
        critical=True
    )

    # Existence/gating check (critical)
    dc_exists = (
        dc is not None and
        (dc.total_hours or "").strip() != "" and
        (dc.renewal_period or "").strip() != "" and
        isinstance(dc.source_urls, list) and len(dc.source_urls) > 0
    )
    evaluator.add_custom_node(
        result=dc_exists,
        id="DC_Info_Provided",
        desc="Washington D.C. CE info is provided with total hours, renewal period, and at least one source URL",
        parent=dc_node,
        critical=True
    )

    # Leaf: DC total hours
    dc_hours_node = evaluator.add_leaf(
        id="DC_Total_Hours",
        desc="State the total number of continuing education hours required for Washington D.C. real estate brokers.",
        parent=dc_node,
        critical=True
    )
    dc_hours_claim = f"Washington D.C. requires {dc.total_hours if dc and dc.total_hours else ''} continuing education hours for real estate brokers."
    # Leaf: DC renewal period
    dc_renewal_node = evaluator.add_leaf(
        id="DC_Renewal_Period",
        desc="State how frequently (renewal period) the continuing education must be completed in Washington D.C.",
        parent=dc_node,
        critical=True
    )
    dc_renewal_claim = f"In Washington D.C., real estate broker continuing education must be completed {dc.renewal_period if dc and dc.renewal_period else ''}."

    # Leaf: DC source authoritative
    dc_source_node = evaluator.add_leaf(
        id="DC_Source_URL",
        desc="Provide a valid reference URL from an official or authoritative source documenting Washington D.C. real estate broker continuing education requirements.",
        parent=dc_node,
        critical=True
    )
    dc_source_claim = (
        "This webpage is an official or authoritative source that documents the continuing education "
        "requirements (hours and/or renewal period) for Washington D.C. real estate brokers."
    )

    # Batch verify the three leaf checks to avoid cross-sibling gating issues
    dc_sources = dc.source_urls if dc and isinstance(dc.source_urls, list) else []
    await evaluator.batch_verify([
        (
            dc_hours_claim,
            dc_sources,
            dc_hours_node,
            "Verify the page states the total number of continuing education hours required for real estate brokers (not salespersons) in Washington D.C. Allow equivalent phrasing."
        ),
        (
            dc_renewal_claim,
            dc_sources,
            dc_renewal_node,
            "Verify the page states the renewal period/frequency for Washington D.C. real estate broker continuing education (e.g., biennially/every 2 years). Allow equivalent phrasing."
        ),
        (
            dc_source_claim,
            dc_sources,
            dc_source_node,
            "Judge whether the page is official or authoritative (e.g., dc.gov domain or the District of Columbia Real Estate Commission/regulator) and clearly documents broker CE requirements."
        ),
    ])


async def verify_md_requirements(evaluator: Evaluator, parent_node, md: Optional[MDRequirements]) -> None:
    # Parent node for Maryland (critical, parallel aggregation)
    md_node = evaluator.add_parallel(
        id="Maryland_Requirements",
        desc="Continuing education requirements for real estate brokers in Maryland.",
        parent=parent_node,
        critical=True
    )

    # Existence/gating check (critical)
    md_exists = (
        md is not None and
        (md.total_hours or "").strip() != "" and
        (md.broker_training_timeframe or "").strip() != "" and
        isinstance(md.source_urls, list) and len(md.source_urls) > 0
    )
    evaluator.add_custom_node(
        result=md_exists,
        id="MD_Info_Provided",
        desc="Maryland CE info is provided with total hours, role-based training timeframe, and at least one source URL",
        parent=md_node,
        critical=True
    )

    # Leaf: MD total hours
    md_hours_node = evaluator.add_leaf(
        id="MD_Total_Hours",
        desc="State the total number of continuing education hours required for Maryland real estate brokers.",
        parent=md_node,
        critical=True
    )
    md_hours_claim = f"Maryland requires {md.total_hours if md and md.total_hours else ''} continuing education hours for real estate brokers."

    # Leaf: MD role-based training timeframe
    md_timeframe_node = evaluator.add_leaf(
        id="MD_Broker_Training_Timeframe",
        desc="State the timeframe within which a Maryland licensee must complete additional training after assuming a broker, branch office manager, or team leader role.",
        parent=md_node,
        critical=True
    )
    md_timeframe_claim = (
        f"In Maryland, a licensee who assumes a broker, branch office manager, or team leader role "
        f"must complete the additional training within {md.broker_training_timeframe if md and md.broker_training_timeframe else ''}."
    )

    # Leaf: MD source authoritative
    md_source_node = evaluator.add_leaf(
        id="MD_Source_URL",
        desc="Provide a valid reference URL from an official or authoritative source documenting Maryland real estate broker continuing education requirements.",
        parent=md_node,
        critical=True
    )
    md_source_claim = (
        "This webpage is an official or authoritative source that documents Maryland real estate broker "
        "continuing education requirements and/or the timeframe for additional training after assuming a broker/manager/team leader role."
    )

    # Batch verify the three leaf checks
    md_sources = md.source_urls if md and isinstance(md.source_urls, list) else []
    await evaluator.batch_verify([
        (
            md_hours_claim,
            md_sources,
            md_hours_node,
            "Verify the page states the total number of continuing education hours required for Maryland real estate brokers. Allow equivalent phrasing."
        ),
        (
            md_timeframe_claim,
            md_sources,
            md_timeframe_node,
            "Verify the specified timeframe within which a licensee must complete additional training after assuming a broker, branch office manager, or team leader role in Maryland."
        ),
        (
            md_source_claim,
            md_sources,
            md_source_node,
            "Judge whether the page is official or authoritative (e.g., maryland.gov/md.gov domain or Maryland Real Estate Commission/regulator) and clearly documents broker CE requirements and/or the role-based training timeframe."
        ),
    ])


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
    model: str = "o4-mini"
) -> Dict:
    """
    Evaluate an answer for the broker CE comparison task.
    """
    # Initialize evaluator
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Top-level parallel aggregation
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

    # Extract structured CE requirements from the answer
    extraction = await evaluator.extract(
        prompt=prompt_extract_ce_requirements(),
        template_class=CEComparisonExtraction,
        extraction_name="ce_requirements_extraction"
    )

    # Top-level critical node for this task
    top_node = evaluator.add_parallel(
        id="Real_Estate_Broker_CE_Requirements",
        desc="Provide continuing education requirement details for Washington D.C. and Maryland as requested.",
        parent=root,
        critical=True
    )

    # Verify DC and MD requirements
    await verify_dc_requirements(evaluator, top_node, extraction.dc)
    await verify_md_requirements(evaluator, top_node, extraction.md)

    # Return evaluation summary
    return evaluator.get_summary()