import asyncio
import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys


# --------------------------------------------------------------------------- #
# Task constants                                                              #
# --------------------------------------------------------------------------- #
TASK_ID = "fcc_part4_outage_compliance_2026_03_15"
TASK_DESCRIPTION = """A wireless telecommunications provider operates a cellular network serving 2,500,000 subscribers across 5,000 macro cell sites. On March 15, 2026, at 1:00 PM EST, the provider discovers a network equipment failure that disables 45 macro cell sites. The outage lasts 2 hours and 15 minutes before service is fully restored. The affected area includes coverage for three Public Safety Answering Points (PSAPs) that rely on the provider's network for 911 call delivery.

As the provider's regulatory compliance officer, you must determine:

1. Whether this outage meets the FCC's reporting thresholds under 47 CFR Part 4
2. If reportable, what specific notifications and reports are required to be filed with the FCC's Network Outage Reporting System (NORS)
3. The specific deadlines (date and time) for each required FCC filing
4. The calculation methodology and numerical result for determining the number of user-minutes affected
5. Whether 911 special facility notifications are required, and if so, the specific timing requirements and mandatory content elements that must be included
6. The regulatory citations that support each compliance obligation

Your analysis must reference the applicable FCC regulations and provide specific deadline calculations based on the outage discovery time of 1:00 PM EST on March 15, 2026.
"""


# --------------------------------------------------------------------------- #
# Extraction models                                                           #
# --------------------------------------------------------------------------- #
class ComplianceExtraction(BaseModel):
    # Overall determinations
    overall_reportable: Optional[str] = None  # e.g., "reportable" / "not reportable" or phrase containing conclusion

    # Duration threshold application
    duration_text: Optional[str] = None  # any statement of duration threshold application

    # User-minutes methodology and result
    user_minutes_methodology_text: Optional[str] = None
    user_minutes_value: Optional[str] = None  # accept "3,037,500" or "3.0375 million", etc.
    user_minutes_threshold_text: Optional[str] = None  # statement comparing to 900,000 threshold

    # NORS filings and deadlines (strings as stated by the answer)
    nors_conditionality_text: Optional[str] = None
    nors_initial_deadline_str: Optional[str] = None
    nors_initial_citation: Optional[str] = None
    nors_initial_report_deadline_str: Optional[str] = None
    nors_initial_report_citation: Optional[str] = None
    nors_final_report_deadline_str: Optional[str] = None
    nors_final_report_citation: Optional[str] = None

    # 911 PSAP notifications
    psap_applicability_text: Optional[str] = None
    psap_initial_deadline_str: Optional[str] = None
    psap_initial_citation: Optional[str] = None
    psap_method_text: Optional[str] = None
    psap_method_citation: Optional[str] = None
    psap_first_followup_deadline_str: Optional[str] = None
    psap_first_followup_rule_text: Optional[str] = None
    psap_first_followup_citation: Optional[str] = None
    psap_mandatory_elements: List[str] = Field(default_factory=list)
    psap_mandatory_elements_citation: Optional[str] = None

    # Generic citations and any URLs present in the answer (if any)
    citations: List[str] = Field(default_factory=list)
    urls: List[str] = Field(default_factory=list)


def prompt_extract_compliance() -> str:
    return """
    Extract the following items exactly as they appear in the answer. Do not invent information that is not present.

    Return a JSON object with these fields:
    - overall_reportable: The answer's explicit overall conclusion about whether the outage is reportable under 47 CFR § 4.9 (string).
    - duration_text: The sentence(s) where the answer applies the ≥30-minute duration threshold (string).
    - user_minutes_methodology_text: The sentence(s) describing the wireless user-minutes calculation method (string).
    - user_minutes_value: The computed user-minutes value stated in the answer (string; allow formats like "3,037,500" or "3.0375 million").
    - user_minutes_threshold_text: The sentence(s) where the answer compares the computed user-minutes to the 900,000 threshold (string).

    - nors_conditionality_text: The sentence(s) that state NORS filing obligations apply only if the outage is reportable, and the resulting implication for this scenario (string).
    - nors_initial_deadline_str: The specific deadline (date/time/time zone string) the answer provides for the initial NORS notification (+120 minutes after discovery).
    - nors_initial_citation: The specific citation the answer associates with the initial NORS notification (e.g., "47 CFR § 4.9(e)(1)").
    - nors_initial_report_deadline_str: The specific deadline (date/time/time zone string) the answer provides for the Initial Communications Outage Report (+72 hours).
    - nors_initial_report_citation: The specific citation used for the Initial Communications Outage Report (e.g., "47 CFR § 4.9(e)(4)").
    - nors_final_report_deadline_str: The specific deadline (date/time/time zone string) the answer provides for the Final Communications Outage Report (+30 days).
    - nors_final_report_citation: The specific citation used for the Final Communications Outage Report (e.g., "47 CFR § 4.9(e)(4)").

    - psap_applicability_text: The sentence(s) concluding whether 911 PSAP notifications are required with citation to 47 CFR § 4.9(h) (string).
    - psap_initial_deadline_str: The specific latest permissible initial PSAP notification deadline computed by the answer (date/time/time zone string).
    - psap_initial_citation: The citation associated with initial PSAP notification timing (e.g., "47 CFR § 4.9(h)(4)").
    - psap_method_text: The sentence(s) stating the required methods (telephone and written electronic) for PSAP notifications (string).
    - psap_method_citation: The citation used for methods (e.g., "47 CFR § 4.9(h)(3)").
    - psap_first_followup_deadline_str: The specific latest permissible first follow-up deadline computed by the answer (date/time/time zone string).
    - psap_first_followup_rule_text: The sentence(s) stating the first follow-up timing rule (no later than 2 hours after initial contact) (string).
    - psap_first_followup_citation: The citation used for the first follow-up rule (e.g., "47 CFR § 4.9(h)(5)").
    - psap_mandatory_elements: An array listing each mandatory PSAP notification content element the answer enumerates.
    - psap_mandatory_elements_citation: The citation associated with the mandatory content list (e.g., "47 CFR § 4.9(h)(2)").

    - citations: Array of all CFR citations mentioned anywhere in the answer.
    - urls: Array of any URLs explicitly present in the answer (do not fabricate; include only if provided).

    Important:
    - Use strings for times/dates exactly as written in the answer.
    - If any requested item is not present, return null (or an empty list where applicable).
    """


# --------------------------------------------------------------------------- #
# Ground truth computations for this scenario                                 #
# --------------------------------------------------------------------------- #
def _fmt_et(dt: datetime) -> str:
    # Format as "Month D, YYYY, H:MM AM/PM ET"
    month = dt.strftime("%B")
    day = dt.day
    year = dt.year
    hour_12 = dt.hour % 12 or 12
    ampm = "AM" if dt.hour < 12 else "PM"
    return f"{month} {day}, {year}, {hour_12}:{dt.minute:02d} {ampm} ET"


def compute_expected_values() -> Dict[str, Any]:
    # Scenario inputs
    subscribers = 2_500_000
    macro_sites = 5_000
    disabled_sites = 45
    outage_minutes = 2 * 60 + 15  # 135 minutes
    discovery = datetime(2026, 3, 15, 13, 0)  # 1:00 PM local Eastern time label

    # User-minutes math
    avg_users_per_site = subscribers / macro_sites  # 500.0
    affected_users = disabled_sites * avg_users_per_site  # 22,500.0
    user_minutes = int(affected_users * outage_minutes)  # 3,037,500

    # Thresholds
    duration_meets = outage_minutes >= 30
    user_minutes_threshold = 900_000
    user_minutes_meets = user_minutes >= user_minutes_threshold
    overall_reportable = duration_meets and user_minutes_meets  # True

    # Deadlines
    nors_initial = discovery + timedelta(minutes=120)  # +120 minutes
    nors_initial_report = discovery + timedelta(hours=72)  # +72 hours
    nors_final_report = discovery + timedelta(days=30)  # +30 days
    psap_initial = discovery + timedelta(minutes=30)  # +30 minutes
    psap_first_followup_latest = psap_initial + timedelta(hours=2)  # +2 hours from initial

    return {
        "inputs": {
            "subscribers": subscribers,
            "macro_sites": macro_sites,
            "disabled_sites": disabled_sites,
            "outage_minutes": outage_minutes,
            "discovery_et": _fmt_et(discovery),
        },
        "math": {
            "avg_users_per_site": int(avg_users_per_site),
            "affected_users": int(affected_users),
            "user_minutes": user_minutes,
        },
        "thresholds": {
            "duration_meets_30min": duration_meets,
            "user_minutes_threshold": user_minutes_threshold,
            "user_minutes_meets": user_minutes_meets,
            "overall_reportable": overall_reportable,
        },
        "deadlines": {
            "nors_initial": _fmt_et(nors_initial),  # Expected 3:00 PM ET, Mar 15, 2026
            "nors_initial_report": _fmt_et(nors_initial_report),  # 1:00 PM ET, Mar 18, 2026
            "nors_final_report": _fmt_et(nors_final_report),  # 1:00 PM ET, Apr 14, 2026
            "psap_initial": _fmt_et(psap_initial),  # 1:30 PM ET, Mar 15, 2026
            "psap_first_followup_latest": _fmt_et(psap_first_followup_latest),  # 3:30 PM ET, Mar 15, 2026
        }
    }


# --------------------------------------------------------------------------- #
# Verification helpers                                                        #
# --------------------------------------------------------------------------- #
TIME_TOLERANCE_INSTRUCTION = (
    "When checking date/time statements, allow reasonable formatting variations (e.g., 'Mar 15, 2026 3 PM ET' vs "
    "'March 15, 2026, 15:00 Eastern'). Treat 'ET', 'EST', and 'EDT' as equivalent labels for this task. "
    "Minor textual differences are acceptable as long as the same deadline is clearly conveyed."
)

NUMBER_TOLERANCE_INSTRUCTION = (
    "When checking numbers, allow thousand separators, spacing, or expressions like '3.0375 million' that are "
    "numerically equivalent to 3,037,500. Minor rounding differences (within ~1%) are acceptable if the method is correct."
)


# --------------------------------------------------------------------------- #
# Tree building: Reportability determination                                  #
# --------------------------------------------------------------------------- #
async def build_reportability_determination(
    evaluator: Evaluator,
    parent_node,
    extraction: ComplianceExtraction,
    expected: Dict[str, Any],
) -> None:
    node = evaluator.add_parallel(
        id="Reportability_Determination",
        desc="Determine whether the outage is reportable under 47 CFR § 4.9 by applying the duration and user-minutes thresholds to the scenario.",
        parent=parent_node,
        critical=True,
    )

    # 1) Duration threshold application (≥30 minutes)
    n1 = evaluator.add_leaf(
        id="Duration_Threshold_Determination",
        desc="Correctly applies the ≥30-minute duration threshold (47 CFR § 4.9, as provided) to the outage duration and states whether it is met.",
        parent=node,
        critical=True,
    )
    outage_minutes = expected["inputs"]["outage_minutes"]
    claim_1 = (
        f"The answer correctly applies the FCC outage duration threshold of at least 30 minutes (47 CFR § 4.9) "
        f"to a {outage_minutes}-minute outage (2 hours 15 minutes) and explicitly states that this threshold is met."
    )
    await evaluator.verify(claim=claim_1, node=n1)

    # 2) User-minutes methodology per 47 CFR § 4.9(e)(2)
    n2 = evaluator.add_leaf(
        id="User_Minutes_Methodology",
        desc="States the wireless user-minutes calculation methodology per 47 CFR § 4.9(e)(2): average users/site = total subscribers ÷ total macro sites; affected users = disabled sites × average users/site; user-minutes = affected users × outage duration (in minutes).",
        parent=node,
        critical=True,
    )
    claim_2 = (
        "The answer states the wireless user-minutes method per 47 CFR § 4.9(e)(2) as: "
        "average users per site = total subscribers ÷ total macro sites; "
        "affected users = disabled sites × average users per site; "
        "user-minutes = affected users × outage duration (in minutes). "
        "Minor wording or ordering variations are acceptable if they are functionally equivalent."
    )
    await evaluator.verify(claim=claim_2, node=n2)

    # 3) User-minutes numerical result
    n3 = evaluator.add_leaf(
        id="User_Minutes_Numerical_Result",
        desc="Computes a numerical user-minutes value consistent with the scenario inputs and the stated methodology.",
        parent=node,
        critical=True,
    )
    um = expected["math"]["user_minutes"]  # 3,037,500
    avg = expected["math"]["avg_users_per_site"]  # 500
    affected = expected["math"]["affected_users"]  # 22,500
    claim_3 = (
        f"The answer computes the wireless user-minutes for this scenario as {um} "
        f"(since 2,500,000 ÷ 5,000 = {avg} users/site; 45 × {avg} = {affected} affected users; "
        f"and {affected} × {outage_minutes} minutes = {um})."
    )
    await evaluator.verify(
        claim=claim_3,
        node=n3,
        additional_instruction=NUMBER_TOLERANCE_INSTRUCTION,
    )

    # 4) 900,000 user-minutes threshold comparison
    n4 = evaluator.add_leaf(
        id="User_Minutes_Threshold_Determination",
        desc="Correctly compares computed user-minutes to the 900,000 user-minutes threshold (47 CFR § 4.9(e), as provided) and states whether it is met.",
        parent=node,
        critical=True,
    )
    threshold = expected["thresholds"]["user_minutes_threshold"]  # 900,000
    claim_4 = (
        f"The answer compares the computed user-minutes to the FCC wireless threshold of {threshold} user-minutes "
        f"(47 CFR § 4.9(e)) and explicitly states that the threshold is met/exceeded."
    )
    await evaluator.verify(claim=claim_4, node=n4)

    # 5) Overall reportability conclusion
    n5 = evaluator.add_leaf(
        id="Overall_Reportability_Conclusion",
        desc="States an overall reportability conclusion (reportable vs not reportable) consistent with the duration and user-minutes threshold determinations.",
        parent=node,
        critical=True,
    )
    overall = "reportable" if expected["thresholds"]["overall_reportable"] else "not reportable"
    claim_5 = (
        f"The answer concludes the outage is {overall} under 47 CFR § 4.9 and the conclusion is consistent with "
        f"both the ≥30-minute duration threshold and the ≥{threshold} user-minutes threshold being met."
    )
    await evaluator.verify(claim=claim_5, node=n5)


# --------------------------------------------------------------------------- #
# Tree building: NORS filings and deadlines                                   #
# --------------------------------------------------------------------------- #
async def build_nors_requirements_and_deadlines(
    evaluator: Evaluator,
    parent_node,
    extraction: ComplianceExtraction,
    expected: Dict[str, Any],
) -> None:
    node = evaluator.add_parallel(
        id="NORS_Filing_Requirements_And_Deadlines",
        desc="Conditionally identifies required NORS filings and computes deadlines based on the discovery time (only if the outage is reportable; otherwise states no NORS filings are required).",
        parent=parent_node,
        critical=True,
    )

    # 1) Conditionality of NORS (only if reportable)
    c1 = evaluator.add_leaf(
        id="NORS_Conditionality",
        desc="Correctly states that NORS filings/notifications are required if (and only if) the outage is reportable; if not reportable, explicitly states that no NORS filings are required.",
        parent=node,
        critical=True,
    )
    claim_c1 = (
        "The answer explicitly states that FCC NORS filings/notifications are required if, and only if, the outage is reportable; "
        "and for this scenario, because the outage is reportable, the provider must file in NORS."
    )
    await evaluator.verify(claim=claim_c1, node=c1)

    # 2) Initial NORS notification (+120 minutes) with citation 47 CFR § 4.9(e)(1)
    c2 = evaluator.add_leaf(
        id="Initial_NORS_Notification",
        desc="If reportable: identifies the initial NORS notification requirement with citation (47 CFR § 4.9(e)(1)) and provides the computed deadline (discovery time + 120 minutes, with date/time/time zone).",
        parent=node,
        critical=True,
    )
    nors_initial_deadline = expected["deadlines"]["nors_initial"]
    claim_c2 = (
        "The answer identifies the initial NORS notification requirement and cites 47 CFR § 4.9(e)(1), "
        f"and it provides the correct deadline of {nors_initial_deadline}, which is 120 minutes after the 1:00 PM ET discovery time on March 15, 2026."
    )
    await evaluator.verify(
        claim=claim_c2,
        node=c2,
        additional_instruction=TIME_TOLERANCE_INSTRUCTION,
    )

    # 3) Initial Communications Outage Report (+72 hours) with citation 47 CFR § 4.9(e)(4)
    c3 = evaluator.add_leaf(
        id="Initial_Communications_Outage_Report",
        desc="If reportable: identifies the Initial Communications Outage Report requirement with citation (47 CFR § 4.9(e)(4)) and provides the computed deadline (discovery time + 72 hours/3 calendar days, with date/time/time zone).",
        parent=node,
        critical=True,
    )
    nors_initial_report_deadline = expected["deadlines"]["nors_initial_report"]
    claim_c3 = (
        "The answer identifies the Initial Communications Outage Report and cites 47 CFR § 4.9(e)(4), "
        f"and it provides the correct deadline of {nors_initial_report_deadline}, which is exactly 72 hours (3 calendar days) after discovery."
    )
    await evaluator.verify(
        claim=claim_c3,
        node=c3,
        additional_instruction=TIME_TOLERANCE_INSTRUCTION,
    )

    # 4) Final Communications Outage Report (+30 days) with citation 47 CFR § 4.9(e)(4)
    c4 = evaluator.add_leaf(
        id="Final_Communications_Outage_Report",
        desc="If reportable: identifies the Final Communications Outage Report requirement with citation (47 CFR § 4.9(e)(4)) and provides the computed deadline as a specific date AND time (with time zone) based on discovery time + 30 days.",
        parent=node,
        critical=True,
    )
    nors_final_report_deadline = expected["deadlines"]["nors_final_report"]
    claim_c4 = (
        "The answer identifies the Final Communications Outage Report and cites 47 CFR § 4.9(e)(4), "
        f"and it provides the correct deadline of {nors_final_report_deadline}, which is exactly 30 days after discovery, including a specific date and time with time zone."
    )
    await evaluator.verify(
        claim=claim_c4,
        node=c4,
        additional_instruction=TIME_TOLERANCE_INSTRUCTION,
    )


# --------------------------------------------------------------------------- #
# Tree building: 911 PSAP notifications                                       #
# --------------------------------------------------------------------------- #
async def build_911_requirements(
    evaluator: Evaluator,
    parent_node,
    extraction: ComplianceExtraction,
    expected: Dict[str, Any],
) -> None:
    node = evaluator.add_parallel(
        id="911_Special_Facility_Notification_Requirements",
        desc="Determine whether 911 special-facility notifications are required and, if required, provide timing, method, and mandatory content elements with supporting citations.",
        parent=parent_node,
        critical=True,
    )

    # 1) Applicability (PSAP notifications required) with citation to 47 CFR § 4.9(h)
    a1 = evaluator.add_leaf(
        id="911_Applicability",
        desc="Correctly determines whether 911 special-facility (PSAP) notifications are required given the scenario and states the conclusion with supporting citation(s) to the relevant 47 CFR § 4.9(h) provision(s).",
        parent=node,
        critical=True,
    )
    claim_a1 = (
        "Because the affected area includes coverage for three PSAPs that rely on the provider's network for 911 calls, "
        "the answer correctly concludes that 911 special-facility (PSAP) notifications are required and cites 47 CFR § 4.9(h)."
    )
    await evaluator.verify(claim=claim_a1, node=a1)

    # 2) Initial notification timing and deadline (ASAP but ≤30 minutes) with citation 47 CFR § 4.9(h)(4)
    a2 = evaluator.add_leaf(
        id="911_Initial_Notification_Timing_And_Deadline",
        desc="States the initial 911 notification timing rule (as soon as possible but no later than 30 minutes after discovery) with citation (47 CFR § 4.9(h)(4)) and computes the latest permissible deadline using the discovery time (date/time/time zone).",
        parent=node,
        critical=True,
    )
    psap_initial_deadline = expected["deadlines"]["psap_initial"]
    claim_a2 = (
        "The answer states the initial PSAP notification timing rule as 'as soon as possible but no later than 30 minutes after discovery' "
        "with citation to 47 CFR § 4.9(h)(4), and it provides the correct latest permissible deadline of "
        f"{psap_initial_deadline}."
    )
    await evaluator.verify(
        claim=claim_a2,
        node=a2,
        additional_instruction=TIME_TOLERANCE_INSTRUCTION,
    )

    # 3) Initial notification method (telephone + written electronic) with citation 47 CFR § 4.9(h)(3)
    a3 = evaluator.add_leaf(
        id="911_Initial_Notification_Method",
        desc="States the required transmission methods (telephone and written via electronic means) with citation (47 CFR § 4.9(h)(3)).",
        parent=node,
        critical=True,
    )
    claim_a3 = (
        "The answer states that PSAP notifications must be transmitted by both telephone and written electronic means, "
        "with citation to 47 CFR § 4.9(h)(3)."
    )
    await evaluator.verify(claim=claim_a3, node=a3)

    # 4) First follow-up timing and deadline (≤2 hours after initial) with citation 47 CFR § 4.9(h)(5)
    a4 = evaluator.add_leaf(
        id="911_First_Followup_Timing_And_Deadline",
        desc="States the first follow-up timing requirement (no later than 2 hours after the initial contact) with citation (47 CFR § 4.9(h)(5)) and provides a specific latest permissible follow-up deadline that is consistent with the initial notification time/deadline used in the answer.",
        parent=node,
        critical=True,
    )
    psap_first_followup_latest = expected["deadlines"]["psap_first_followup_latest"]
    claim_a4 = (
        "The answer states the first follow-up timing requirement as 'no later than 2 hours after the initial contact' with citation to 47 CFR § 4.9(h)(5), "
        "and it provides a concrete latest permissible follow-up deadline that is exactly 2 hours after the initial time it used. "
        f"If the answer uses the latest permissible initial deadline of {expected['deadlines']['psap_initial']}, then the latest permissible follow-up is {psap_first_followup_latest}."
    )
    await evaluator.verify(
        claim=claim_a4,
        node=a4,
        additional_instruction=TIME_TOLERANCE_INSTRUCTION,
    )

    # 5) Mandatory content elements with citation 47 CFR § 4.9(h)(2)
    a5 = evaluator.add_leaf(
        id="911_Mandatory_Content_Elements",
        desc="Lists the mandatory 911 notification content elements with citation (47 CFR § 4.9(h)(2)): unique outage identifier; provider contact info (name/phone/email); provider name; start date/time with time zone; types of services affected; geographic area affected; expected impact on the 911 facility; expected restoration date/time with time zone; best-known cause; notification type indicator (initial/update/final).",
        parent=node,
        critical=True,
    )
    claim_a5 = (
        "The answer lists all required PSAP notification content elements per 47 CFR § 4.9(h)(2): "
        "unique outage identifier; provider contact information (name, telephone number, and email address); provider name; "
        "start date/time with time zone; types of services affected; geographic area affected; expected impact on the 911 facility; "
        "expected restoration date/time with time zone; best-known cause of the outage; and a notification type indicator (initial, update, or final)."
    )
    await evaluator.verify(claim=claim_a5, node=a5)

    # 6) PSAP contact list special diligence with citation 47 CFR § 4.9(h)(1)
    a6 = evaluator.add_leaf(
        id="PSAP_Contact_List_Special_Diligence",
        desc="Identifies the special diligence obligation for maintaining accurate PSAP contact lists (including annual updates/documented verification attempts) with citation (47 CFR § 4.9(h)(1)).",
        parent=node,
        critical=True,
    )
    claim_a6 = (
        "The answer identifies the special diligence obligation to maintain accurate PSAP contact lists, including at least annual updates "
        "or documented verification attempts, with citation to 47 CFR § 4.9(h)(1)."
    )
    await evaluator.verify(claim=claim_a6, node=a6)


# --------------------------------------------------------------------------- #
# Main evaluation entry                                                       #
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
    # Initialize evaluator
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

    # Compute ground truth expectations and record as GT info
    expected = compute_expected_values()
    evaluator.add_ground_truth({
        "scenario_inputs": expected["inputs"],
        "expected_math": expected["math"],
        "expected_thresholds": expected["thresholds"],
        "expected_deadlines": expected["deadlines"],
        "regulatory_references": {
            "part": "47 CFR § 4.9",
            "wireless_user_minutes_method": "§ 4.9(e)(2)",
            "nors_initial": "§ 4.9(e)(1)",
            "nors_initial_and_final_reports": "§ 4.9(e)(4)",
            "psap_initial_timing": "§ 4.9(h)(4)",
            "psap_methods": "§ 4.9(h)(3)",
            "psap_first_followup": "§ 4.9(h)(5)",
            "psap_content": "§ 4.9(h)(2)",
            "psap_contact_list_diligence": "§ 4.9(h)(1)",
        }
    })

    # Extract structured info from the answer (for transparency/logging)
    extraction = await evaluator.extract(
        prompt=prompt_extract_compliance(),
        template_class=ComplianceExtraction,
        extraction_name="compliance_extraction",
    )

    # Build tree corresponding to rubric
    # Root node mirrors rubric description
    root.desc = "Evaluate FCC Part 4 reportability, required NORS filings/deadlines (if reportable), 911 special-facility notification duties, and user-minutes methodology/result, using the scenario’s discovery time."

    # Subtrees
    await build_reportability_determination(evaluator, root, extraction, expected)
    await build_nors_requirements_and_deadlines(evaluator, root, extraction, expected)
    await build_911_requirements(evaluator, root, extraction, expected)

    # Return final structured summary
    return evaluator.get_summary()