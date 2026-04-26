import asyncio
import logging
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys

# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "fcc_outage_2026_compliance"
TASK_DESCRIPTION = (
    "A facilities-based wireless telecommunications provider in the United States experiences a major network outage "
    "in 2026. The outage affects service across multiple states for 11 continuous hours and impacts approximately "
    "2 million customers. During the outage, affected customers' mobile devices display 'SOS only' or 'Emergency calls only' "
    "status. The provider's investigation determines that a software issue in their cloud-based core network infrastructure "
    "caused the service disruption. Multiple 911 public safety answering points (PSAPs) in the affected regions report "
    "difficulties receiving emergency calls during the outage. According to current FCC regulations for network outage "
    "reporting and telecommunications emergency preparedness standards, what are the key regulatory compliance requirements "
    "and thresholds that apply to this outage scenario? Your answer must specify all applicable FCC reporting thresholds, "
    "notification deadlines to the FCC and affected facilities, and the backup power infrastructure requirements that should "
    "have been in place."
)

# Ground-truth expectations used for verification claims
GT = {
    "duration_threshold": "at least 30 minutes",
    "user_minutes_threshold": "900,000 user-minutes (i.e., ≥ 900,000 user-minutes of telephony service)",
    "nors_notification_deadline": "within 120 minutes of discovering a reportable outage (NORS notification)",
    "initial_report_deadline": "within 3 calendar days (72 hours) of discovering the outage",
    "final_report_deadline": "within 30 days of discovering the outage",
    "psap_initial_notification": "within 30 minutes of discovery when 911 special facilities are potentially affected",
    "psap_followup_notification": "first follow-up within 2 hours after initial contact",
    "backup_power_requirement": (
        "CMRS providers must have 8 hours of backup power for cell sites, remote switches, and digital loop carrier "
        "system remote terminals"
    ),
}

# --------------------------------------------------------------------------- #
# Data models for structured extraction                                       #
# --------------------------------------------------------------------------- #
class ClaimField(BaseModel):
    stated_value: Optional[str] = None
    source_urls: List[str] = Field(default_factory=list)


class ApplicabilityInfo(BaseModel):
    stated: Optional[str] = None
    source_urls: List[str] = Field(default_factory=list)


class FCCComplianceExtraction(BaseModel):
    duration_threshold: Optional[ClaimField] = None
    user_minutes_threshold: Optional[ClaimField] = None
    nors_notification_deadline: Optional[ClaimField] = None
    initial_report_deadline: Optional[ClaimField] = None
    final_report_deadline: Optional[ClaimField] = None
    psap_initial_notification_deadline: Optional[ClaimField] = None
    psap_followup_notification_deadline: Optional[ClaimField] = None
    backup_power_requirement: Optional[ClaimField] = None
    explicit_applicability: Optional[ApplicabilityInfo] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_fcc_requirements() -> str:
    return """
    Extract, from the provided answer text, the specific FCC compliance requirements and thresholds requested. 
    For each item below, return:
      - stated_value: the exact value, threshold, or deadline as written in the answer (e.g., "30 minutes", "≥900,000 user-minutes", "within 120 minutes", "3 calendar days / 72 hours", "30 days", "within 30 minutes", "within 2 hours", "8 hours")
      - source_urls: all URLs explicitly cited in the answer that directly support that specific item. If none are cited, return an empty list.

    Fields to extract:
      1) duration_threshold
         - Meaning: FCC/NORS minimum outage duration threshold (expected correct value is "at least 30 minutes").
      2) user_minutes_threshold
         - Meaning: FCC/NORS user-impact threshold (expected correct value is "900,000 user-minutes" or "≥900,000 user-minutes").
      3) nors_notification_deadline
         - Meaning: Time to submit the initial NORS notification to the FCC after discovering a reportable outage (expected "within 120 minutes").
      4) initial_report_deadline
         - Meaning: Time to submit the initial outage report after discovery (expected "within 3 calendar days" or "within 72 hours").
      5) final_report_deadline
         - Meaning: Time to submit the final outage report after discovery (expected "within 30 days").
      6) psap_initial_notification_deadline
         - Meaning: Timeline for notifying affected 911 special facilities/PSAPs after discovery when potentially affected (expected "within 30 minutes").
      7) psap_followup_notification_deadline
         - Meaning: Timeline for the first follow-up to PSAPs after initial contact (expected "within 2 hours after initial contact").
      8) backup_power_requirement
         - Meaning: Applicable emergency backup power requirement for CMRS providers (expected "8 hours for cell sites, remote switches, and digital loop carrier system remote terminals").

    Also extract an explicit applicability statement for the scenario:
      9) explicit_applicability
         - stated: whether the answer explicitly states that the described outage scenario meets/triggers the FCC thresholds and the 911 notification obligations (e.g., “This outage triggers FCC reporting thresholds and PSAP notification requirements.”). If no such explicit statement was made, set to null.
         - source_urls: any URLs cited in the answer specifically to support this applicability statement (if any).

    Return a single JSON object with these fields. Do not invent values or URLs; use null for missing stated_value, and [] for missing URLs.
    """


# --------------------------------------------------------------------------- #
# Helper verification builders                                                #
# --------------------------------------------------------------------------- #
async def verify_requirement_item(
    evaluator: Evaluator,
    parent_node,
    *,
    item_id: str,
    item_desc: str,
    claim_text_expected_truth: str,
    extracted_field: Optional[ClaimField],
    acceptable_variants_note: str = "",
    critical: bool = True,
) -> None:
    """
    Create a container node with two critical leaf checks:
      1) The answer explicitly states the correct requirement/deadline/threshold.
      2) The cited source(s) (if provided) support that requirement/deadline/threshold.
    """
    container = evaluator.add_parallel(
        id=item_id,
        desc=item_desc,
        parent=parent_node,
        critical=critical,
    )

    # 1) Stated explicitly in the answer
    stated_leaf = evaluator.add_leaf(
        id=f"{item_id}_stated_in_answer",
        desc=f"{item_desc} — stated explicitly in the answer",
        parent=container,
        critical=True,
    )
    await evaluator.verify(
        claim=f"The answer explicitly states that {claim_text_expected_truth}.",
        node=stated_leaf,
        additional_instruction=(
            "Focus only on whether the answer contains this exact regulatory requirement or an equivalent phrasing. "
            "Allow equivalent expressions (e.g., 'half an hour' equals '30 minutes'; 'two hours' equals '120 minutes'; "
            "'3 calendar days' equals '72 hours'; '>= 900,000 user-minutes' equals '900,000 user-minutes'). "
            + (acceptable_variants_note or "")
        ),
    )

    # 2) Supported by cited sources (if any were provided)
    support_leaf = evaluator.add_leaf(
        id=f"{item_id}_supported_by_sources",
        desc=f"{item_desc} — supported by cited source(s)",
        parent=container,
        critical=True,
    )
    srcs = extracted_field.source_urls if (extracted_field and extracted_field.source_urls) else []
    await evaluator.verify(
        claim=f"According to the provided source(s), {claim_text_expected_truth}.",
        node=support_leaf,
        sources=srcs if srcs else None,
        additional_instruction=(
            "Judge only whether the webpage(s) explicitly support this regulatory requirement. If no sources were "
            "provided, judge based on your evaluation criteria and context, allowing minor wording variations."
        ),
    )


# --------------------------------------------------------------------------- #
# Main evaluation                                                             #
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
    # Initialize evaluator (root is non-critical by default and parallel aggregation)
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

    # Extract structured answers
    extracted = await evaluator.extract(
        prompt=prompt_extract_fcc_requirements(),
        template_class=FCCComplianceExtraction,
        extraction_name="fcc_compliance_extraction",
    )

    # Add ground-truth info for transparency
    evaluator.add_ground_truth(
        {
            "expected": GT,
            "notes": "These are the commonly cited FCC Part 4 (NORS) outage reporting thresholds/deadlines and PSAP notification timelines, "
                     "and the CMRS backup power requirement as specified in the rubric for this task.",
        },
        gt_type="ground_truth",
    )

    # Container for all FCC compliance checks (make non-critical to allow the mixed child criticalities)
    fcc_node = evaluator.add_parallel(
        id="FCC_Compliance_Requirements",
        desc=("Answer identifies the key FCC regulatory compliance requirements and thresholds applicable to the described "
              "outage scenario, including reporting thresholds, FCC deadlines, 911 special-facility notifications, and backup "
              "power requirements."),
        parent=root,
        critical=False,
    )

    # Build each requirement item as a critical container with two critical leaves
    await verify_requirement_item(
        evaluator,
        fcc_node,
        item_id="FCC_Reporting_Threshold_Min_Duration",
        item_desc="States the FCC/NORS minimum outage duration reporting threshold (outage lasts at least 30 minutes)",
        claim_text_expected_truth=f"the FCC/NORS minimum outage duration reporting threshold is {GT['duration_threshold']}",
        extracted_field=extracted.duration_threshold or ClaimField(),
        acceptable_variants_note="Explicitly accept '>= 30 minutes', '30+ minutes', or 'half an hour'.",
        critical=True,
    )

    await verify_requirement_item(
        evaluator,
        fcc_node,
        item_id="FCC_Reporting_Threshold_User_Minutes",
        item_desc="States the FCC/NORS user-impact reporting threshold (≥900,000 user-minutes of telephony service)",
        claim_text_expected_truth="the FCC/NORS user-impact reporting threshold is 900,000 user-minutes (i.e., at least 900,000 user-minutes)",
        extracted_field=extracted.user_minutes_threshold or ClaimField(),
        acceptable_variants_note="Accept '>= 900,000 user-minutes', '900k user-minutes', or '0.9 million user-minutes'.",
        critical=True,
    )

    await verify_requirement_item(
        evaluator,
        fcc_node,
        item_id="FCC_NORS_Notification_Deadline",
        item_desc="States the FCC/NORS notification deadline (submit NORS notification within 120 minutes of discovery)",
        claim_text_expected_truth=f"{GT['nors_notification_deadline']}",
        extracted_field=extracted.nors_notification_deadline or ClaimField(),
        acceptable_variants_note="Accept 'within 2 hours' as equivalent to 'within 120 minutes'.",
        critical=True,
    )

    await verify_requirement_item(
        evaluator,
        fcc_node,
        item_id="FCC_Initial_Outage_Report_Deadline",
        item_desc="States the FCC initial outage report deadline (within 3 calendar days / 72 hours of discovery)",
        claim_text_expected_truth=f"{GT['initial_report_deadline']}",
        extracted_field=extracted.initial_report_deadline or ClaimField(),
        acceptable_variants_note="Accept 'within 72 hours' as equivalent to 'within 3 calendar days'.",
        critical=True,
    )

    await verify_requirement_item(
        evaluator,
        fcc_node,
        item_id="FCC_Final_Outage_Report_Deadline",
        item_desc="States the FCC final outage report deadline (within 30 days of discovery)",
        claim_text_expected_truth=f"{GT['final_report_deadline']}",
        extracted_field=extracted.final_report_deadline or ClaimField(),
        acceptable_variants_note="Accept 'within 30 calendar days' or 'within one month' (as an approximate equivalent).",
        critical=True,
    )

    await verify_requirement_item(
        evaluator,
        fcc_node,
        item_id="911_Special_Facilities_Initial_Notification",
        item_desc="States the 911 special-facility initial notification timeline (report to affected facilities within 30 minutes of discovery)",
        claim_text_expected_truth=f"{GT['psap_initial_notification']}",
        extracted_field=extracted.psap_initial_notification_deadline or ClaimField(),
        acceptable_variants_note="Accept phrasing like 'no later than 30 minutes', 'within thirty minutes', or similar.",
        critical=True,
    )

    await verify_requirement_item(
        evaluator,
        fcc_node,
        item_id="911_Special_Facilities_Followup_Notification",
        item_desc="States the 911 special-facility follow-up timeline (first follow-up within 2 hours after initial contact)",
        claim_text_expected_truth=f"{GT['psap_followup_notification']}",
        extracted_field=extracted.psap_followup_notification_deadline or ClaimField(),
        acceptable_variants_note="Accept 'within two hours following initial notification'.",
        critical=True,
    )

    await verify_requirement_item(
        evaluator,
        fcc_node,
        item_id="Backup_Power_Infrastructure_Requirement",
        item_desc=("States the applicable emergency backup power requirement (CMRS providers must have 8 hours of backup "
                   "power for cell sites, remote switches, and digital loop carrier system remote terminals)"),
        claim_text_expected_truth=f"{GT['backup_power_requirement']}",
        extracted_field=extracted.backup_power_requirement or ClaimField(),
        acceptable_variants_note="Accept 'eight hours' as equivalent to '8 hours'.",
        critical=True,
    )

    # Explicit applicability to the scenario (non-critical)
    explicit_node = evaluator.add_parallel(
        id="Explicit_Applicability_To_Scenario",
        desc=("Explicitly indicates that the scenario meets/triggers the listed thresholds and therefore triggers the "
              "FCC reporting and 911 notification obligations."),
        parent=fcc_node,
        critical=False,
    )

    explicit_stated_leaf = evaluator.add_leaf(
        id="Explicit_Applicability_To_Scenario_stated",
        desc="The answer explicitly states that this scenario meets/triggers the listed FCC/NORS thresholds and PSAP notification duties",
        parent=explicit_node,
        critical=False,
    )
    await evaluator.verify(
        claim=(
            "The answer explicitly indicates that the described outage triggers FCC reportable outage thresholds and the "
            "911 special-facility/PSAP notification obligations."
        ),
        node=explicit_stated_leaf,
        additional_instruction=(
            "Accept synonyms like 'meets', 'exceeds', 'triggers', 'subject to', 'reportable', or 'obligations apply'."
        ),
    )

    logic_consistency_leaf = evaluator.add_leaf(
        id="Explicit_Applicability_To_Scenario_logic_check",
        desc="Based on the facts (11 hours; ~2M customers; PSAP difficulties), the thresholds are logically triggered",
        parent=explicit_node,
        critical=False,
    )
    await evaluator.verify(
        claim=(
            "Given the scenario facts—an 11-hour outage affecting ~2,000,000 customers across multiple states with PSAP 911 "
            "difficulties—the 30-minute minimum duration threshold is met, the user-minutes threshold (≥900,000) is far "
            "exceeded, and PSAP notification obligations are implicated."
        ),
        node=logic_consistency_leaf,
        additional_instruction=(
            "You may reason with the given scenario details only; do not require exact math, but confirm logical sufficiency."
        ),
    )

    # Return evaluation summary
    return evaluator.get_summary()