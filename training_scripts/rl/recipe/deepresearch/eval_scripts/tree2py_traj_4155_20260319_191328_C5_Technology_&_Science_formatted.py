import asyncio
import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "fcc_verizon_outage_2026_deadlines"
TASK_DESCRIPTION = """
On January 14, 2026, Verizon experienced a major network outage that began at approximately 12:30 PM Eastern Time and lasted over 10 hours, affecting more than 1.5 million customers and impacting 911 emergency services in some areas. Based on FCC regulations for network outage reporting and emergency service notification that were in effect at the time, determine the following compliance deadlines and requirements:

1. The exact date and time (Eastern Time) by which Verizon was required to submit its initial NORS (Network Outage Reporting System) notification under the 120-minute requirement
2. The exact date and time (Eastern Time) by which Verizon was required to submit its initial outage report to NORS under the 72-hour requirement
3. The specific deadline date by which Verizon was required to submit its final NORS outage report under the 30-day requirement
4. Confirmation that the outage met the FCC's minimum duration threshold for reportability
5. The exact time (Eastern Time) by which Verizon was required to provide initial notification to affected PSAPs (Public Safety Answering Points) regarding the 911-impacting outage, given that FCC PSAP notification rules took effect on April 15, 2025
6. The required minimum frequency for follow-up notifications to PSAPs until the outage was resolved

Provide all specific deadline dates and times, and include supporting URL references from official FCC documentation describing the reporting requirements and from reliable news sources documenting the Verizon outage details.
"""

# Fixed baseline for calculations (Eastern Time context)
BASE_START_ET = datetime(2026, 1, 14, 12, 30)  # 12:30 PM ET, Jan 14, 2026

EXPECTED_120M = BASE_START_ET + timedelta(minutes=120)     # 2:30 PM ET, Jan 14, 2026
EXPECTED_72H = BASE_START_ET + timedelta(hours=72)         # 12:30 PM ET, Jan 17, 2026
EXPECTED_30D_DATE = (BASE_START_ET + timedelta(days=30)).date()  # Feb 13, 2026
EXPECTED_PSAP_30M = BASE_START_ET + timedelta(minutes=30)  # 1:00 PM ET, Jan 14, 2026


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class ComplianceExtraction(BaseModel):
    # NORS deadlines from the answer
    notification_120m_deadline_et: Optional[str] = None
    initial_report_72h_deadline_et: Optional[str] = None
    final_report_30d_deadline_date: Optional[str] = None
    nors_requirement_urls: List[str] = Field(default_factory=list)

    # PSAP notification requirements from the answer
    initial_psap_notification_time_et: Optional[str] = None
    followup_frequency: Optional[str] = None
    psap_requirement_urls: List[str] = Field(default_factory=list)

    # Outage details and sources from the answer
    outage_start_time_et: Optional[str] = None
    outage_end_time_et: Optional[str] = None
    outage_duration_text: Optional[str] = None
    outage_sources: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_compliance() -> str:
    return """
    Extract the following information exactly as presented in the answer. Use strings for all times and dates. All times should be in Eastern Time as stated in the answer. For URLs, return the actual URL strings.

    Required fields:
    - notification_120m_deadline_et: The answer's stated exact deadline (ET) for the initial NORS notification (120-minute rule)
    - initial_report_72h_deadline_et: The answer's stated exact deadline (ET) for the initial NORS outage report (72-hour rule)
    - final_report_30d_deadline_date: The answer's stated specific deadline DATE for the final NORS outage report (30-day rule). Date only.
    - nors_requirement_urls: Array of URL(s) to official FCC documentation for NORS reporting requirements (should cover 120-minute, 72-hour, 30-day)

    - initial_psap_notification_time_et: The answer's stated exact time (ET) by which initial PSAP notification was required (30-minute rule)
    - followup_frequency: The answer's stated minimum frequency for PSAP follow-up notifications (e.g., "every 2 hours")
    - psap_requirement_urls: Array of URL(s) to documentation describing FCC PSAP notification requirements (30-minute initial, 2-hour follow-up; rules effective Apr 15, 2025)

    - outage_start_time_et: The answer's stated outage start time in ET (expected around 12:30 PM ET on Jan 14, 2026)
    - outage_end_time_et: The answer's stated outage end time in ET (if given; e.g., around 10:15–10:20 PM ET)
    - outage_duration_text: The answer's stated duration text (e.g., "over 10 hours")
    - outage_sources: Array of URL(s) to reliable news or official statements documenting the Verizon outage details (start time, duration, 911 impact, customer impact)

    Rules:
    - If any field is not present in the answer, set it to null (for strings) or [] (for arrays).
    - Extract only URLs explicitly present in the answer. Do not fabricate any URLs.
    - Preserve the exact textual formatting of times/dates as given in the answer.
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def fmt_human_et(dt: datetime) -> str:
    """Format a datetime in a human-friendly way with explicit ET suffix."""
    return dt.strftime("%B %-d, %Y %-I:%M %p ET") if hasattr(dt, "strftime") else str(dt)


def fmt_human_et_portable(dt: datetime) -> str:
    """Portable version that avoids %-d on Windows."""
    return dt.strftime("%B %d, %Y %I:%M %p ET").replace(" 0", " ")


def expected_strings() -> Dict[str, str]:
    """Return canonical human-readable expected strings for deadlines."""
    # Use portable formatter to avoid issues across environments
    return {
        "nors_120m": fmt_human_et_portable(EXPECTED_120M),
        "nors_72h": fmt_human_et_portable(EXPECTED_72H),
        "psap_30m": fmt_human_et_portable(EXPECTED_PSAP_30M),
        "nors_30d_date": EXPECTED_30D_DATE.strftime("%B %d, %Y").replace(" 0", " "),
    }


# --------------------------------------------------------------------------- #
# Verification builders                                                       #
# --------------------------------------------------------------------------- #
async def build_outage_verifications(evaluator: Evaluator, parent, data: ComplianceExtraction):
    """
    Build and verify the Outage_Characteristics_Verification subtree.
    Returns a dict of created nodes for potential dependency reference.
    """
    out_node = evaluator.add_parallel(
        id="Outage_Characteristics_Verification",
        desc="Verify that the January 14, 2026 Verizon outage met FCC thresholds for reportability and document key outage details",
        parent=parent,
        critical=False,
    )

    # 1) Duration threshold met (≥ 30 minutes)
    duration_threshold_leaf = evaluator.add_leaf(
        id="Duration_Threshold_Met",
        desc="Confirm that the outage met the FCC's minimum duration threshold of 30 minutes for reportability",
        parent=out_node,
        critical=True,
    )
    claim_duration_threshold = (
        "The Verizon outage on January 14, 2026 lasted at least 30 minutes; "
        "in fact, it lasted over 10 hours according to reliable sources."
    )
    await evaluator.verify(
        claim=claim_duration_threshold,
        node=duration_threshold_leaf,
        sources=data.outage_sources,
        additional_instruction=(
            "Use the provided outage source URLs to confirm that the outage duration clearly exceeded 30 minutes. "
            "If sources state it lasted 'over 10 hours', that fully satisfies this check."
        ),
    )

    # 2) Outage start time and duration
    start_and_duration_leaf = evaluator.add_leaf(
        id="Outage_Start_Time_and_Duration",
        desc="Identify the outage start time as approximately 12:30 PM ET on January 14, 2026, and confirm it lasted over 10 hours",
        parent=out_node,
        critical=True,
    )
    claim_start_duration = (
        "Reliable sources report that Verizon's outage on January 14, 2026 began around 12:30 PM Eastern Time and "
        "lasted over 10 hours, with restoration around approximately 10:15–10:20 PM ET."
    )
    await evaluator.verify(
        claim=claim_start_duration,
        node=start_and_duration_leaf,
        sources=data.outage_sources,
        additional_instruction=(
            "Allow reasonable approximations for the start time (around 12:30 PM ET) and end time window "
            "(around 10:15–10:20 PM ET). Focus on whether the sources support the approximate times and >10 hour duration."
        ),
    )

    # 3) 911 impact confirmation
    impact_911_leaf = evaluator.add_leaf(
        id="911_Impact_Confirmation",
        desc="Confirm that the outage impacted 911 emergency services in some areas",
        parent=out_node,
        critical=True,
    )
    claim_911 = (
        "Reliable sources report that the January 14, 2026 Verizon outage impacted 911 emergency services in some areas."
    )
    await evaluator.verify(
        claim=claim_911,
        node=impact_911_leaf,
        sources=data.outage_sources,
        additional_instruction="Verify that at least one source explicitly mentions impact to 911 or emergency calling in affected areas.",
    )

    # 4) Outage details references (reliable news/official, including start time and >1.5M customers impacted)
    details_reference_leaf = evaluator.add_leaf(
        id="Outage_Details_Reference",
        desc="Provide URL references documenting the outage details (start time, duration, customer impact, 911 disruption)",
        parent=out_node,
        critical=True,
    )
    claim_details_ref = (
        "At least one of the provided URLs is a reliable news source or an official statement about the January 14, 2026 "
        "Verizon outage and documents the start time (around 12:30 PM ET) and that more than 1.5 million customers were affected."
    )
    await evaluator.verify(
        claim=claim_details_ref,
        node=details_reference_leaf,
        sources=data.outage_sources,
        additional_instruction="Accept reputable national or local news outlets or official Verizon/FCC statements as reliable.",
    )

    return {
        "duration_threshold_leaf": duration_threshold_leaf,
        "start_and_duration_leaf": start_and_duration_leaf,
        "impact_911_leaf": impact_911_leaf,
        "details_reference_leaf": details_reference_leaf,
    }


async def build_nors_verifications(evaluator: Evaluator, parent, data: ComplianceExtraction, prereq_nodes: Optional[List] = None):
    """
    Build and verify the NORS_Reporting_Deadlines subtree.
    """
    nors_node = evaluator.add_parallel(
        id="NORS_Reporting_Deadlines",
        desc="Identify all NORS reporting deadlines that applied to Verizon based on the outage start time and FCC requirements",
        parent=parent,
        critical=False,
    )

    exp = expected_strings()

    # 1) 120-minute notification deadline (calculation check)
    leaf_120 = evaluator.add_leaf(
        id="120_Minute_Notification_Deadline",
        desc="Calculate and provide exact ET deadline for initial NORS notification (120 minutes from discovery/start)",
        parent=nors_node,
        critical=True,
    )
    claim_120 = (
        f"The answer states the 120-minute NORS notification deadline as '{data.notification_120m_deadline_et}'. "
        f"Given discovery at approximately 12:30 PM ET on January 14, 2026, "
        f"the correct 120-minute deadline is '{exp['nors_120m']}'. "
        "The stated deadline matches the correct deadline (allowing for equivalent formatting and ET notation)."
    )
    await evaluator.verify(
        claim=claim_120,
        node=leaf_120,
        additional_instruction="Judge whether the provided deadline equals 2:30 PM ET on January 14, 2026, allowing differences in formatting but not in the actual timepoint.",
        extra_prerequisites=prereq_nodes or [],
    )

    # 2) 72-hour initial report deadline (calculation check)
    leaf_72 = evaluator.add_leaf(
        id="72_Hour_Initial_Report_Deadline",
        desc="Calculate and provide exact ET deadline for initial NORS report (within 72 hours of discovery)",
        parent=nors_node,
        critical=True,
    )
    claim_72 = (
        f"The answer states the 72-hour initial NORS report deadline as '{data.initial_report_72h_deadline_et}'. "
        f"Based on 72 hours after 12:30 PM ET on January 14, 2026, the correct deadline is '{exp['nors_72h']}'. "
        "The stated deadline matches this correct deadline (format differences allowed, same ET timepoint required)."
    )
    await evaluator.verify(
        claim=claim_72,
        node=leaf_72,
        additional_instruction="Judge whether the provided deadline equals 12:30 PM ET on January 17, 2026, allowing different but equivalent textual formats.",
        extra_prerequisites=prereq_nodes or [],
    )

    # 3) 30-day final report deadline (calculation check; date only)
    leaf_30 = evaluator.add_leaf(
        id="30_Day_Final_Report_Deadline",
        desc="Provide the specific deadline date for the final NORS outage report (no later than 30 days after discovery)",
        parent=nors_node,
        critical=True,
    )
    claim_30 = (
        f"The answer states the 30-day final NORS report deadline date as '{data.final_report_30d_deadline_date}'. "
        f"Thirty days after January 14, 2026 is '{exp['nors_30d_date']}'. "
        "The stated date matches this correct deadline date (format differences allowed, same calendar date required)."
    )
    await evaluator.verify(
        claim=claim_30,
        node=leaf_30,
        additional_instruction="Judge whether the provided final report deadline date equals February 13, 2026.",
        extra_prerequisites=prereq_nodes or [],
    )

    # 4) NORS requirements references (FCC official documentation covers 120-min, 72-hour, 30-day)
    leaf_refs = evaluator.add_leaf(
        id="NORS_Requirements_Reference",
        desc="Provide FCC documentation URL(s) describing NORS 120-minute notification, 72-hour initial report, and 30-day final report requirements",
        parent=nors_node,
        critical=True,
    )
    claim_refs = (
        "The provided URL(s) are official FCC documentation that describe NORS reporting requirements, including: "
        "initial notification within 120 minutes of determining an outage is reportable, an initial outage report within 72 hours (3 days) "
        "of discovery, and a final report due no later than 30 days after discovery."
    )
    await evaluator.verify(
        claim=claim_refs,
        node=leaf_refs,
        sources=data.nors_requirement_urls,
        additional_instruction="Confirm the page(s) are FCC sources and explicitly mention all three: 120-minute initial notification, 72-hour initial report, and 30-day final report.",
    )

    return {
        "leaf_120": leaf_120,
        "leaf_72": leaf_72,
        "leaf_30": leaf_30,
        "leaf_refs": leaf_refs,
    }


async def build_psap_verifications(evaluator: Evaluator, parent, data: ComplianceExtraction, prereq_nodes: Optional[List] = None):
    """
    Build and verify the PSAP_Notification_Requirements subtree.
    """
    psap_node = evaluator.add_parallel(
        id="PSAP_Notification_Requirements",
        desc="Identify PSAP notification requirements that applied (30-minute initial, 2-hour follow-ups) under rules effective April 15, 2025",
        parent=parent,
        critical=False,
    )

    exp = expected_strings()

    # 1) 30-minute initial notification time (calculation from 12:30 PM ET)
    leaf_psap_30 = evaluator.add_leaf(
        id="30_Minute_Initial_Notification",
        desc="Provide exact ET time by which initial PSAP notification was required (within 30 minutes of discovery/start)",
        parent=psap_node,
        critical=True,
    )
    claim_psap_30 = (
        f"The answer states the PSAP initial notification deadline as '{data.initial_psap_notification_time_et}'. "
        f"Within 30 minutes of 12:30 PM ET on January 14, 2026 is '{exp['psap_30m']}'. "
        "The stated deadline matches this correct time (allowing for equivalent formatting and ET notation)."
    )
    await evaluator.verify(
        claim=claim_psap_30,
        node=leaf_psap_30,
        additional_instruction="Judge whether the provided PSAP initial notification deadline equals 1:00 PM ET on January 14, 2026.",
        extra_prerequisites=prereq_nodes or [],
    )

    # 2) 2-hour follow-up frequency (rule requirement; verify via URLs)
    leaf_followup = evaluator.add_leaf(
        id="2_Hour_Followup_Frequency",
        desc="Identify required minimum frequency for PSAP follow-up notifications (at least every 2 hours)",
        parent=psap_node,
        critical=True,
    )
    claim_followup = (
        "Under FCC PSAP outage notification rules effective April 15, 2025, providers must send follow-up notifications "
        "to affected PSAPs at least every 2 hours until the outage is resolved."
    )
    await evaluator.verify(
        claim=claim_followup,
        node=leaf_followup,
        sources=data.psap_requirement_urls,
        additional_instruction="Verify the minimum follow-up frequency is 'at least every 2 hours' until resolution.",
    )

    # 3) PSAP requirements references (must cover 30-minute initial and 2-hour follow-ups; effective April 15, 2025)
    leaf_psap_refs = evaluator.add_leaf(
        id="PSAP_Requirements_Reference",
        desc="Provide documentation URL(s) describing FCC PSAP notification rules (30-minute initial; 2-hour follow-up; effective Apr 15, 2025)",
        parent=psap_node,
        critical=True,
    )
    claim_psap_refs = (
        "The provided URL(s) document FCC PSAP outage notification requirements that took effect on April 15, 2025, "
        "including initial notification within 30 minutes and follow-up notifications at least every 2 hours until restoration."
    )
    await evaluator.verify(
        claim=claim_psap_refs,
        node=leaf_psap_refs,
        sources=data.psap_requirement_urls,
        additional_instruction="Prefer FCC official rule summaries, orders, or public notices that explicitly state the effective date and timing requirements.",
    )

    return {
        "leaf_psap_30": leaf_psap_30,
        "leaf_followup": leaf_followup,
        "leaf_psap_refs": leaf_psap_refs,
    }


# --------------------------------------------------------------------------- #
# Main evaluation function                                                    #
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
    Evaluate an answer for the FCC compliance deadlines and outage verification task.
    """
    # Initialize the evaluator
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Parallel aggregation at the top level per rubric
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
    data = await evaluator.extract(
        prompt=prompt_extract_compliance(),
        template_class=ComplianceExtraction,
        extraction_name="compliance_extraction",
    )

    # Add ground truth baselines for transparency
    evaluator.add_ground_truth({
        "baseline_outage_start_et": "January 14, 2026 12:30 PM ET",
        "expected_120_minute_deadline": expected_strings()["nors_120m"],
        "expected_72_hour_deadline": expected_strings()["nors_72h"],
        "expected_30_day_final_date": expected_strings()["nors_30d_date"],
        "expected_psap_30_minute_deadline": expected_strings()["psap_30m"],
        "psap_rules_effective_date": "April 15, 2025"
    })

    # Build the full verification tree according to the rubric

    # Parent container per rubric
    top = evaluator.add_parallel(
        id="FCC_Compliance_Deadline_Identification",
        desc="Verify all FCC reporting and PSAP notification deadlines for the Jan 14, 2026 Verizon outage",
        parent=root,
        critical=False,
    )

    # First, build and verify outage characteristics (used as logical baseline)
    outage_nodes = await build_outage_verifications(evaluator, top, data)

    # Then NORS reporting deadlines (calculation checks + FCC refs).
    # Use outage start/duration node as an additional prerequisite to calculations
    nors_nodes = await build_nors_verifications(
        evaluator,
        top,
        data,
        prereq_nodes=[outage_nodes["start_and_duration_leaf"]],
    )

    # PSAP notification requirements (calculation for initial + rule verification)
    psap_nodes = await build_psap_verifications(
        evaluator,
        top,
        data,
        prereq_nodes=[outage_nodes["start_and_duration_leaf"]],
    )

    # Return the final structured evaluation summary
    return evaluator.get_summary()