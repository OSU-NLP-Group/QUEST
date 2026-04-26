import asyncio
import logging
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy

# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "verizon_nors_2026_01_14"
TASK_DESCRIPTION = (
    "On January 14, 2026, Verizon experienced a major nationwide network outage that affected millions of wireless "
    "customers. Under the FCC's Network Outage Reporting System (NORS) rules, wireless service providers must submit "
    "reports according to specific timelines after discovering a reportable outage. An outage is reportable if it lasts "
    "at least 30 minutes and meets other thresholds.\n\n"
    "Based on when Verizon customers first began noticing the service disruption on January 14, 2026, and the NORS "
    "reporting requirements for wireless providers, determine:\n\n"
    "1. The approximate time when customers first began experiencing the outage on January 14, 2026 (this serves as the discovery time for NORS reporting purposes)\n"
    "2. Whether the outage duration exceeded the 30-minute NORS reportability threshold\n"
    "3. The deadline by which Verizon must submit the initial NORS notification (120 minutes after determining reportability)\n"
    "4. The deadline by which Verizon must submit the initial outage report (3 calendar days after determining reportability)\n"
    "5. The deadline by which Verizon must submit the final outage report (30 days after discovering the outage)\n\n"
    "Provide all deadlines in Eastern Time with specific dates and times where applicable."
)

# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class NORSAnswerExtraction(BaseModel):
    # Outage discovery timeline
    discovery_time_text: Optional[str] = None  # The textual time provided in the answer (e.g., "around 9:15 AM ET")
    discovery_datetime_et_iso: Optional[str] = None  # ISO 8601 in Eastern Time, exactly as provided in the answer; do not invent
    # Sources that support the outage start time/duration
    timeline_sources: List[str] = Field(default_factory=list)

    # Duration claim (reportability threshold)
    duration_exceeded_30min: Optional[bool] = None  # True if the answer explicitly states ≥30 minutes; otherwise null
    duration_evidence_text: Optional[str] = None     # Optional textual evidence phrase extracted from the answer

    # Deadlines stated in the answer (if any)
    initial_notification_deadline_et_text: Optional[str] = None
    initial_report_deadline_et_text: Optional[str] = None
    final_report_deadline_et_text: Optional[str] = None


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_nors_answer() -> str:
    return (
        "Extract the key facts and URLs the answer provides about the January 14, 2026 Verizon outage and the NORS deadlines.\n"
        "Required fields:\n"
        "1) discovery_time_text: The earliest time customers first began noticing the outage on Jan 14, 2026, as stated in the answer (text as-is).\n"
        "2) discovery_datetime_et_iso: The exact discovery time in Eastern Time if the answer provides a specific date+time. "
        "Return in ISO 8601 with timezone offset (e.g., 2026-01-14T09:15:00-05:00). If the answer does not give a precise timestamp, return null.\n"
        "3) timeline_sources: All URLs the answer cites that support the outage start time or duration. Include every URL mentioned (markdown links, plain URLs, etc.).\n"
        "4) duration_exceeded_30min: Return true if the answer explicitly claims the outage lasted 30+ minutes; otherwise return null.\n"
        "5) duration_evidence_text: The exact text from the answer supporting the 30+ minutes claim (if any).\n"
        "6) initial_notification_deadline_et_text: The initial NORS notification deadline stated in the answer (120 minutes after determining reportability), ET format; if not stated, return null.\n"
        "7) initial_report_deadline_et_text: The initial outage report deadline stated in the answer (3 calendar days after determining reportability), ET format; if not stated, return null.\n"
        "8) final_report_deadline_et_text: The final outage report deadline stated in the answer (30 days after discovery), ET format; if not stated, return null.\n\n"
        "Do not invent or infer any information not explicitly present in the answer. If a field cannot be determined from the answer, return null."
    )


# --------------------------------------------------------------------------- #
# Utilities                                                                   #
# --------------------------------------------------------------------------- #
def _parse_iso_datetime_et(dt_str: Optional[str]) -> Optional[datetime]:
    """
    Parse an ISO 8601 datetime string that includes a timezone offset (e.g., 2026-01-14T09:15:00-05:00).
    Returns an aware datetime with the same offset. If parsing fails or input is None, returns None.
    """
    if not dt_str:
        return None
    try:
        # Normalize trailing Z if present
        normalized = dt_str.strip().replace("Z", "+00:00")
        return datetime.fromisoformat(normalized)
    except Exception:
        return None


def _fmt_et(dt: datetime) -> str:
    """Format a timezone-aware datetime into a concise ET string for claims."""
    # Keep the provided offset; label as ET
    return dt.strftime("%Y-%m-%d %H:%M ET")


def _compute_expected_deadlines(discovery_et: Optional[datetime], reportable: bool) -> Dict[str, Optional[datetime]]:
    """
    Compute expected deadlines:
    - Reportability determination time: discovery + 30 minutes (only if reportable)
    - Initial notification: +120 minutes from determination time
    - Initial report: +3 days from determination time
    - Final report: +30 days from discovery
    Returns dict of datetimes (or None if cannot be computed).
    """
    expected: Dict[str, Optional[datetime]] = {
        "reportability_time_et": None,
        "initial_notification_deadline_et": None,
        "initial_report_deadline_et": None,
        "final_report_deadline_et": None,
    }
    if not discovery_et:
        return expected

    if reportable:
        reportability_time = discovery_et + timedelta(minutes=30)
        initial_notification = reportability_time + timedelta(minutes=120)
        initial_report = reportability_time + timedelta(days=3)
        expected["reportability_time_et"] = reportability_time
        expected["initial_notification_deadline_et"] = initial_notification
        expected["initial_report_deadline_et"] = initial_report

    # Final report is always based on discovery time (if the outage is reportable at all)
    if reportable:
        expected["final_report_deadline_et"] = discovery_et + timedelta(days=30)

    return expected


# --------------------------------------------------------------------------- #
# Verification helpers                                                        #
# --------------------------------------------------------------------------- #
async def _verify_outage_timeline(
    evaluator: Evaluator,
    parent_node,
    extraction: NORSAnswerExtraction,
) -> None:
    """
    Build and verify the 'Outage_Timeline_Determination' subtree:
    - Discovery time provided and supported by sources
    - Duration exceeded 30 minutes supported by sources
    """
    timeline_node = evaluator.add_parallel(
        id="Outage_Timeline_Determination",
        desc="Identify the key temporal parameters of the outage that trigger NORS reporting obligations",
        parent=parent_node,
        critical=True,
    )

    # ----- Outage_Discovery_Time group (critical) -----
    discovery_group = evaluator.add_parallel(
        id="Outage_Discovery_Time",
        desc="Identify when customers first began noticing service disruptions on January 14, 2026",
        parent=timeline_node,
        critical=True,
    )

    # Existence of a specific discovery time (ISO ET)
    discovery_time_provided = evaluator.add_custom_node(
        result=bool(extraction.discovery_datetime_et_iso),
        id="Outage_Discovery_Time_Provided",
        desc="A specific discovery time (ET ISO) is provided in the answer",
        parent=discovery_group,
        critical=True,
    )

    # Sources provided
    sources_provided = evaluator.add_custom_node(
        result=bool(extraction.timeline_sources),
        id="Outage_Discovery_Time_Sources_Provided",
        desc="Sources are provided to support the discovery time",
        parent=discovery_group,
        critical=True,
    )

    # Supported by sources (verify with URLs)
    discovery_supported = evaluator.add_leaf(
        id="Outage_Discovery_Time_Supported",
        desc="Discovery time is supported by cited sources",
        parent=discovery_group,
        critical=True,
    )
    discovery_dt = _parse_iso_datetime_et(extraction.discovery_datetime_et_iso)
    if discovery_dt:
        claim_discovery = (
            f"Customers first began noticing Verizon service disruptions around {_fmt_et(discovery_dt)} on January 14, 2026."
        )
    else:
        # If no parseable time, craft a generic claim from text (will likely fail if too vague)
        claim_discovery = (
            f"The outage was first noticed by customers on January 14, 2026 at the time stated in the answer: "
            f"'{extraction.discovery_time_text}'."
        )

    await evaluator.verify(
        claim=claim_discovery,
        node=discovery_supported,
        sources=extraction.timeline_sources,
        additional_instruction=(
            "Focus on the earliest time the outage was observed on January 14, 2026. Allow reasonable approximations "
            "(e.g., around HH:MM). If multiple sources provide slightly different timestamps within ~20 minutes, consider "
            "them consistent. The verification should be based on the URLs provided in the answer."
        ),
    )

    # ----- Outage_Duration_Verification group (critical) -----
    duration_group = evaluator.add_parallel(
        id="Outage_Duration_Verification",
        desc="Verify that the outage duration exceeded the 30-minute NORS reportability threshold",
        parent=timeline_node,
        critical=True,
    )

    duration_provided = evaluator.add_custom_node(
        result=(extraction.duration_exceeded_30min is not None),
        id="Outage_Duration_Provided",
        desc="The answer explicitly states whether duration exceeded 30 minutes",
        parent=duration_group,
        critical=True,
    )

    duration_supported = evaluator.add_leaf(
        id="Outage_Duration_Exceeds_30min",
        desc="The outage exceeded 30 minutes as supported by cited sources",
        parent=duration_group,
        critical=True,
    )
    claim_duration = "The Verizon outage on January 14, 2026 lasted longer than 30 minutes."
    await evaluator.verify(
        claim=claim_duration,
        node=duration_supported,
        sources=extraction.timeline_sources,
        additional_instruction=(
            "Verify that the sources indicate the outage persisted for at least 30 minutes (or longer), thereby meeting "
            "the NORS reportability threshold. If sources clearly imply a duration well over 30 minutes, consider this supported."
        ),
    )


async def _verify_deadline_group(
    evaluator: Evaluator,
    parent_node,
    group_id: str,
    group_desc: str,
    provided_text: Optional[str],
    expected_dt: Optional[datetime],
    derivation_note: str,
) -> None:
    """
    Build a critical sub-tree for a single deadline:
    - Provided in answer
    - Correct against computed expected value
    """
    group_node = evaluator.add_parallel(
        id=group_id,
        desc=group_desc,
        parent=parent_node,
        critical=True,
    )

    provided_node = evaluator.add_custom_node(
        result=bool(provided_text),
        id=f"{group_id}_Provided",
        desc=f"{group_desc} is provided in the answer",
        parent=group_node,
        critical=True,
    )

    correct_node = evaluator.add_leaf(
        id=f"{group_id}_Correct",
        desc=f"{group_desc} is computed correctly",
        parent=group_node,
        critical=True,
    )

    if expected_dt is None:
        # If we cannot compute expected DT, mark this leaf as failed directly (no verification call)
        correct_node.score = 0.0
        correct_node.status = "failed"
        return

    claim_text = f"The {group_desc.lower()} is {_fmt_et(expected_dt)}."
    await evaluator.verify(
        claim=claim_text,
        node=correct_node,
        additional_instruction=(
            f"Compare the claimed deadline to the answer. The expected deadline is {_fmt_et(expected_dt)}. "
            f"Derivation: {derivation_note}. Allow minor formatting variations, but the date/time (ET) must match."
        ),
    )


async def _verify_nors_deadlines(
    evaluator: Evaluator,
    parent_node,
    extraction: NORSAnswerExtraction,
    discovery_dt: Optional[datetime],
) -> None:
    """
    Build and verify the 'NORS_Deadline_Calculations' subtree:
    - Initial notification (120 minutes after determining reportability)
    - Initial report (3 calendar days after determining reportability)
    - Final report (30 days after discovering the outage)
    All deadlines are in ET.
    """
    deadlines_node = evaluator.add_parallel(
        id="NORS_Deadline_Calculations",
        desc="Calculate all three mandatory NORS reporting deadlines based on the outage discovery time",
        parent=parent_node,
        critical=True,
    )

    # Gate calculations with input validity: need discovery time and reportability=True
    calc_ready = evaluator.add_custom_node(
        result=(discovery_dt is not None and extraction.duration_exceeded_30min is True),
        id="Deadline_Calculation_Input_Valid",
        desc="Inputs sufficient to compute deadlines (discovery time present and outage confirmed reportable)",
        parent=deadlines_node,
        critical=True,
    )

    expected = _compute_expected_deadlines(discovery_dt, reportable=(extraction.duration_exceeded_30min is True))
    reportability_note = (
        f"Reportability determined at {_fmt_et(expected['reportability_time_et'])} (i.e., 30 minutes after discovery)"
        if expected.get("reportability_time_et") else
        "Reportability time could not be determined due to missing inputs"
    )

    # Initial Notification Deadline: +120 minutes after reportability time
    await _verify_deadline_group(
        evaluator=evaluator,
        parent_node=deadlines_node,
        group_id="Initial_Notification_Deadline",
        group_desc="Initial NORS notification deadline (120 minutes after determining reportability)",
        provided_text=extraction.initial_notification_deadline_et_text,
        expected_dt=expected["initial_notification_deadline_et"],
        derivation_note=f"{reportability_note}; initial notification due 120 minutes after.",
    )

    # Initial Report Deadline: +3 calendar days after reportability time
    await _verify_deadline_group(
        evaluator=evaluator,
        parent_node=deadlines_node,
        group_id="Initial_Report_Deadline",
        group_desc="Initial outage report deadline (3 calendar days after determining reportability)",
        provided_text=extraction.initial_report_deadline_et_text,
        expected_dt=expected["initial_report_deadline_et"],
        derivation_note=f"{reportability_note}; initial report due 3 calendar days after.",
    )

    # Final Report Deadline: +30 days after discovery
    await _verify_deadline_group(
        evaluator=evaluator,
        parent_node=deadlines_node,
        group_id="Final_Report_Deadline",
        group_desc="Final outage report deadline (30 days after discovering the outage)",
        provided_text=extraction.final_report_deadline_et_text,
        expected_dt=expected["final_report_deadline_et"],
        derivation_note="Final report due 30 days after the discovery time.",
    )

    # Record computed expectations in summary for transparency
    evaluator.add_ground_truth({
        "discovery_time_et": _fmt_et(discovery_dt) if discovery_dt else None,
        "reportability_time_et": _fmt_et(expected["reportability_time_et"]) if expected.get("reportability_time_et") else None,
        "initial_notification_deadline_et": _fmt_et(expected["initial_notification_deadline_et"]) if expected.get("initial_notification_deadline_et") else None,
        "initial_report_deadline_et": _fmt_et(expected["initial_report_deadline_et"]) if expected.get("initial_report_deadline_et") else None,
        "final_report_deadline_et": _fmt_et(expected["final_report_deadline_et"]) if expected.get("final_report_deadline_et") else None,
    }, gt_type="computed_expected_deadlines")


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
    Evaluate an answer for Verizon NORS reporting deadlines for the Jan 14, 2026 outage.
    """
    # Initialize evaluator (root node is a general container)
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

    # Extract structured information from the answer
    extraction: NORSAnswerExtraction = await evaluator.extract(
        prompt=prompt_extract_nors_answer(),
        template_class=NORSAnswerExtraction,
        extraction_name="nors_answer_extraction",
    )

    # Build the root critical assessment node to mirror rubric
    assessment_node = evaluator.add_sequential(
        id="NORS_Reporting_Compliance_Assessment",
        desc="Evaluate the complete set of NORS reporting deadlines applicable to the January 14, 2026 Verizon outage",
        parent=root,
        critical=True,
    )

    # 1) Verify outage timeline determination (discovery + duration)
    await _verify_outage_timeline(evaluator, assessment_node, extraction)

    # Parse discovery time for deadline calculations
    discovery_dt = _parse_iso_datetime_et(extraction.discovery_datetime_et_iso)

    # 2) Verify NORS deadline calculations (initial notification, initial report, final report)
    await _verify_nors_deadlines(evaluator, assessment_node, extraction, discovery_dt)

    # Add custom info for transparency
    evaluator.add_custom_info(
        info={
            "extracted": extraction.dict(),
            "notes": "Computed expected deadlines assume reportability determined when the outage exceeds the 30-minute threshold (i.e., discovery + 30 minutes). All deadlines expressed in ET.",
        },
        info_type="audit_trail",
        info_name="extraction_and_computation_notes",
    )

    return evaluator.get_summary()