import asyncio
import logging
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "fcc_nors_wireline_outage_assessment"
TASK_DESCRIPTION = (
    "A wireline telecommunications provider operating in California experiences a service disruption affecting "
    "25,000 customers. The outage begins at 2:00 PM on Monday and service is fully restored at 3:15 PM the same day. "
    "Based on FCC regulations for Network Outage Reporting System (NORS) submissions, determine: "
    "1. Is this outage reportable to the FCC? "
    "2. If reportable, what are the specific deadlines (in hours or days after discovery) for each required submission to NORS? "
    "Your answer should include: "
    "- Whether the outage meets the FCC reporting thresholds for wireline providers "
    "- The calculation method and result for determining reportability "
    "- All applicable reporting deadlines with their time frames"
)

# Ground truth values derived from the scenario
AFFECTED_USERS = 25_000
OUTAGE_DURATION_MINUTES = 75  # 2:00 PM to 3:15 PM same day
DURATION_THRESHOLD_MINUTES = 30
USER_MINUTES_THRESHOLD = 900_000
USER_MINUTES_COMPUTED = AFFECTED_USERS * OUTAGE_DURATION_MINUTES  # 1,875,000
EXPECTED_REPORTABLE = (OUTAGE_DURATION_MINUTES >= DURATION_THRESHOLD_MINUTES) and (USER_MINUTES_COMPUTED >= USER_MINUTES_THRESHOLD)

# Expected NORS deadlines (relative to discovery)
NORS_NOTIFICATION_DEADLINE = "within 120 minutes after discovery"
NORS_INITIAL_REPORT_DEADLINE = "within 72 hours after discovery"
NORS_FINAL_REPORT_DEADLINE = "within 30 days after discovery"


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class OutageAssessmentExtraction(BaseModel):
    # Outage timing as stated in the answer (strings to maximize compatibility)
    outage_start_time: Optional[str] = None
    outage_end_time: Optional[str] = None
    duration_minutes_stated: Optional[str] = None
    duration_threshold_minutes_stated: Optional[str] = None

    # User-minutes calculation as stated
    affected_users_stated: Optional[str] = None
    user_minutes_calc_method: Optional[str] = None  # e.g., "duration (minutes) × affected users"
    user_minutes_result: Optional[str] = None
    user_minutes_threshold_stated: Optional[str] = None

    # Explicit reportability conclusion
    reportability_conclusion: Optional[str] = None  # e.g., "reportable", "not reportable", "yes/no", etc.

    # NORS conditional requirements as stated
    nors_applicability_statement: Optional[str] = None  # e.g., "filings required"/"not required"/"N/A"
    notification_deadline: Optional[str] = None        # text as stated in answer
    initial_report_deadline: Optional[str] = None      # text as stated in answer
    final_report_deadline: Optional[str] = None        # text as stated in answer
    submission_method: Optional[str] = None            # e.g., "electronically via NORS"

    # Any URLs cited in the answer (if available)
    sources: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_outage_assessment() -> str:
    return """
    Extract the key elements the answer provides for determining FCC NORS reportability for this wireline outage and, if reportable, the submission requirements and deadlines.

    Return a JSON object with the following fields (use null for any missing item):
    - outage_start_time: The stated outage start time (string, as written).
    - outage_end_time: The stated outage end time (string, as written).
    - duration_minutes_stated: The stated outage duration in minutes (string, e.g., "75 minutes" or "75").
    - duration_threshold_minutes_stated: The stated minimum duration threshold used (string, e.g., "30 minutes" or "30").
    - affected_users_stated: The stated number of potentially affected users used in the calculation (string, e.g., "25000").
    - user_minutes_calc_method: The description of the method used to calculate user-minutes (expect something like "duration in minutes × potentially affected users").
    - user_minutes_result: The stated user-minutes value (string, as written).
    - user_minutes_threshold_stated: The stated user-minutes threshold for wireline providers (string, e.g., "900000" or "900,000").
    - reportability_conclusion: The explicit conclusion on whether the outage is reportable to the FCC (string, e.g., "reportable", "yes", "not reportable", "no").
    - nors_applicability_statement: Whether NORS filings are required (string, e.g., "required", "N/A", "not required").
    - notification_deadline: The stated Notification deadline (string, ideally including 'within 120 minutes' and 'after discovery').
    - initial_report_deadline: The stated Initial Communications Outage Report deadline (string, ideally including 'within 72 hours' and 'after discovery').
    - final_report_deadline: The stated Final Communications Outage Report deadline (string, ideally including 'within 30 days' and 'after discovery').
    - submission_method: The stated submission method (string, e.g., "electronically via NORS").
    - sources: An array of URLs (strings) that the answer explicitly cites as references for FCC thresholds/deadlines or definitions. Only include valid URLs that appear in the answer text.
    """


# --------------------------------------------------------------------------- #
# Helper to get sources (if any)                                              #
# --------------------------------------------------------------------------- #
def _sources_or_none(extracted: OutageAssessmentExtraction) -> Optional[List[str]]:
    if extracted and extracted.sources:
        # Remove obvious empties/spaces
        cleaned = [s for s in extracted.sources if isinstance(s, str) and s.strip()]
        return cleaned if cleaned else None
    return None


# --------------------------------------------------------------------------- #
# Verification logic                                                          #
# --------------------------------------------------------------------------- #
async def build_and_verify_tree(evaluator: Evaluator, extracted: OutageAssessmentExtraction) -> None:
    # Root already initialized as SEQUENTIAL via evaluator.initialize()

    # 1) Reportability determination node (parallel, critical)
    reportability_node = evaluator.add_parallel(
        id="Reportability_Determination",
        desc="Determines whether the outage is reportable for a wireline provider based on duration and user-minute thresholds, and states the conclusion.",
        parent=evaluator.root,
        critical=True
    )

    # 1.a) Duration threshold check (leaf, critical)
    dur_leaf = evaluator.add_leaf(
        id="Duration_Threshold_Check",
        desc="Correctly determines whether the outage duration meets/exceeds the 30-minute minimum threshold using the given start/end times.",
        parent=reportability_node,
        critical=True
    )
    # Construct claim using scenario data; allow minor phrasing differences
    dur_claim = (
        f"The outage lasted {OUTAGE_DURATION_MINUTES} minutes (from 2:00 PM to 3:15 PM the same day), "
        f"which meets or exceeds the 30-minute minimum threshold."
    )
    await evaluator.verify(
        claim=dur_claim,
        node=dur_leaf,
        additional_instruction=(
            "Judge whether the answer explicitly or implicitly recognizes that 2:00 PM to 3:15 PM same day is 75 minutes "
            "and that 75 ≥ 30, i.e., the duration threshold is satisfied. Minor wording variants are acceptable."
        )
    )

    # 1.b) User-minutes calculation method and result (leaf, critical)
    um_calc_leaf = evaluator.add_leaf(
        id="User_Minutes_Calculation_Method_And_Result",
        desc="Uses the correct method (duration in minutes × potentially affected users) and provides the resulting user-minutes value.",
        parent=reportability_node,
        critical=True
    )
    um_calc_claim = (
        f"The answer uses the correct user-minutes method (duration in minutes × potentially affected users) and calculates "
        f"25,000 × {OUTAGE_DURATION_MINUTES} = {USER_MINUTES_COMPUTED} user-minutes."
    )
    await evaluator.verify(
        claim=um_calc_claim,
        node=um_calc_leaf,
        additional_instruction=(
            "Check the answer text for both: (1) The method (duration in minutes times potentially affected users) and "
            f"(2) the numeric result matching {USER_MINUTES_COMPUTED}. Allow minor formatting (commas, spaces)."
        )
    )

    # 1.c) User-minutes threshold check (leaf, critical)
    um_threshold_leaf = evaluator.add_leaf(
        id="User_Minutes_Threshold_Check",
        desc="Correctly determines whether the calculated user-minutes meets/exceeds the 900,000 user-minutes threshold for wireline providers.",
        parent=reportability_node,
        critical=True
    )
    um_threshold_claim = (
        f"Given the calculated user-minutes of {USER_MINUTES_COMPUTED}, the answer correctly determines that this "
        f"meets or exceeds the 900,000 user-minutes threshold for wireline providers."
    )
    await evaluator.verify(
        claim=um_threshold_claim,
        node=um_threshold_leaf,
        additional_instruction=(
            "Focus on whether the answer asserts that 1,875,000 ≥ 900,000 (i.e., threshold is satisfied). "
            "You don't need to verify the FCC threshold number itself here; the check is about 'meets/exceeds'."
        )
    )

    # 1.d) Explicit reportability conclusion (leaf, critical)
    reportable_leaf = evaluator.add_leaf(
        id="Explicit_Reportability_Conclusion",
        desc="Explicitly answers whether the outage is reportable to the FCC (yes/no) consistent with the threshold checks.",
        parent=reportability_node,
        critical=True
    )
    reportable_claim = (
        "The answer explicitly concludes that the outage is reportable to the FCC (i.e., that NORS reporting applies), "
        "consistent with the duration and user-minutes thresholds being met."
    )
    await evaluator.verify(
        claim=reportable_claim,
        node=reportable_leaf,
        additional_instruction=(
            "Look for an explicit yes/no or equivalent phrasing. It must align with the prior checks (here, it should say reportable)."
        )
    )

    # 2) NORS submission requirements node (parallel, critical)
    nors_node = evaluator.add_parallel(
        id="NORS_Submission_Requirements_(Conditional)",
        desc="If the outage is reportable, provide all required NORS submissions and deadlines (relative to discovery) and indicate electronic submission via NORS; if not reportable, indicate that NORS submissions/deadlines are not required/applicable.",
        parent=evaluator.root,
        critical=True
    )

    # Extract sources, if any (to assist URL-supported verification)
    sources = _sources_or_none(extracted)

    # 2.a) Conditional applicability statement (leaf, critical)
    applicability_leaf = evaluator.add_leaf(
        id="Conditional_Applicability_Statement",
        desc="States whether NORS filings are required (reportable case) or not required/N/A (non-reportable case), consistent with the reportability conclusion.",
        parent=nors_node,
        critical=True
    )
    if EXPECTED_REPORTABLE:
        applicability_claim = (
            "Because the outage is reportable, the answer states that NORS filings are required (i.e., not N/A or not required)."
        )
    else:
        applicability_claim = (
            "Because the outage is not reportable, the answer states that NORS filings are not required or N/A."
        )
    await evaluator.verify(
        claim=applicability_claim,
        node=applicability_leaf,
        sources=sources,
        additional_instruction=(
            "Judge consistency with the prior reportability conclusion and check whether the answer explicitly indicates "
            "'required' vs 'not required'/'N/A' for NORS filings."
        )
    )

    # 2.b) Notification deadline conditional (leaf, critical)
    notif_leaf = evaluator.add_leaf(
        id="Notification_Deadline_Conditional",
        desc="If reportable: states the notification deadline is within 120 minutes of discovery; if not reportable: states notification is not required/N/A.",
        parent=nors_node,
        critical=True
    )
    if EXPECTED_REPORTABLE:
        notif_claim = (
            "The answer states that the Notification must be submitted within 120 minutes (2 hours) after discovery."
        )
    else:
        notif_claim = (
            "Because the outage is not reportable, the answer indicates the Notification is not required or N/A."
        )
    await evaluator.verify(
        claim=notif_claim,
        node=notif_leaf,
        sources=sources,
        additional_instruction=(
            "If reportable, verify the presence of 'within 120 minutes' (or '2 hours') and 'after discovery' wording. "
            "If not reportable, verify the answer indicates 'not required' or 'N/A'."
        )
    )

    # 2.c) Initial report deadline conditional (leaf, critical)
    initial_leaf = evaluator.add_leaf(
        id="Initial_Report_Deadline_Conditional",
        desc="If reportable: states the Initial Communications Outage Report deadline is within 72 hours after discovery; if not reportable: states initial report is not required/N/A.",
        parent=nors_node,
        critical=True
    )
    if EXPECTED_REPORTABLE:
        initial_claim = (
            "The answer states that the Initial Communications Outage Report is due within 72 hours after discovery."
        )
    else:
        initial_claim = (
            "Because the outage is not reportable, the answer indicates the Initial report is not required or N/A."
        )
    await evaluator.verify(
        claim=initial_claim,
        node=initial_leaf,
        sources=sources,
        additional_instruction=(
            "If reportable, verify the presence of 'within 72 hours' and 'after discovery' wording. "
            "If not reportable, verify 'not required' or 'N/A'."
        )
    )

    # 2.d) Final report deadline conditional (leaf, critical)
    final_leaf = evaluator.add_leaf(
        id="Final_Report_Deadline_Conditional",
        desc="If reportable: states the Final Communications Outage Report deadline is within 30 days after discovery; if not reportable: states final report is not required/N/A.",
        parent=nors_node,
        critical=True
    )
    if EXPECTED_REPORTABLE:
        final_claim = (
            "The answer states that the Final Communications Outage Report is due within 30 days after discovery."
        )
    else:
        final_claim = (
            "Because the outage is not reportable, the answer indicates the Final report is not required or N/A."
        )
    await evaluator.verify(
        claim=final_claim,
        node=final_leaf,
        sources=sources,
        additional_instruction=(
            "If reportable, verify the presence of 'within 30 days' and 'after discovery' wording. "
            "If not reportable, verify 'not required' or 'N/A'."
        )
    )

    # 2.e) Electronic submission via NORS conditional (leaf, critical)
    submission_leaf = evaluator.add_leaf(
        id="Electronic_Submission_Via_NORS_Conditional",
        desc="If reportable: states required reports are submitted electronically through FCC NORS; if not reportable: states submission via NORS is not required/N/A.",
        parent=nors_node,
        critical=True
    )
    if EXPECTED_REPORTABLE:
        submission_claim = (
            "The answer states that required reports are submitted electronically via the FCC's Network Outage Reporting System (NORS)."
        )
    else:
        submission_claim = (
            "Because the outage is not reportable, the answer indicates that submission via NORS is not required or N/A."
        )
    await evaluator.verify(
        claim=submission_claim,
        node=submission_leaf,
        sources=sources,
        additional_instruction=(
            "If reportable, look for phrasing such as 'electronically via NORS' or 'submitted through FCC NORS'. "
            "If not reportable, verify it indicates not required/N/A."
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
    model: str = "o4-mini"
) -> Dict:
    # Initialize evaluator with SEQUENTIAL root as required by the rubric (root orchestrates phases)
    evaluator = Evaluator()
    evaluator.initialize(
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
    extracted = await evaluator.extract(
        prompt=prompt_extract_outage_assessment(),
        template_class=OutageAssessmentExtraction,
        extraction_name="outage_assessment_extraction"
    )

    # Add ground truth context for transparency
    evaluator.add_ground_truth({
        "wireline_thresholds": {
            "duration_minutes_min": DURATION_THRESHOLD_MINUTES,
            "user_minutes_min": USER_MINUTES_THRESHOLD
        },
        "scenario_values": {
            "affected_users": AFFECTED_USERS,
            "duration_minutes": OUTAGE_DURATION_MINUTES,
            "user_minutes_computed": USER_MINUTES_COMPUTED
        },
        "expected_reportable": EXPECTED_REPORTABLE,
        "expected_deadlines_relative_to_discovery": {
            "notification": NORS_NOTIFICATION_DEADLINE,
            "initial_report": NORS_INITIAL_REPORT_DEADLINE,
            "final_report": NORS_FINAL_REPORT_DEADLINE
        }
    }, gt_type="expected_assessment")

    # Build verification tree and run checks
    await build_and_verify_tree(evaluator, extracted)

    # Return evaluation summary
    return evaluator.get_summary()