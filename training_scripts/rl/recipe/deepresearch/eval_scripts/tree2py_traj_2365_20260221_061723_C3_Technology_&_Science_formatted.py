import asyncio
import logging
from typing import Optional, List, Dict, Any, Union

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy

# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "fcc_part4_outage_compliance"
TASK_DESCRIPTION = (
    "Identify a major wireless telecommunications outage in the United States that occurred between January 1, 2020 "
    "and December 31, 2024, where the outage met FCC reportability thresholds under 47 CFR § 4.9 (lasting at least 30 minutes "
    "and potentially affecting at least 900,000 user-minutes). For this identified outage, verify complete compliance with "
    "FCC Part 4 reporting requirements by providing the following information with supporting reference URLs: "
    "(1) Reportability Verification: Confirm the outage duration, estimated user impact (in user-minutes if available), and "
    "the wireless provider's name. Provide a reference URL documenting these outage details. "
    "(2) Notification Timeline: Verify when the provider discovered the outage and when notification was submitted to the FCC's "
    "Network Outage Reporting System (NORS). Confirm this was within 120 minutes of discovery. Provide a reference URL. "
    "(3) Initial Report Timeline: Verify when the Initial Communications Outage Report was submitted to NORS. Confirm this was "
    "within 72 hours of discovery. Provide a reference URL. "
    "(4) Final Report Timeline: Verify when the Final Communications Outage Report was submitted to NORS. Confirm this was within "
    "30 days of discovery. Provide a reference URL. "
    "(5) Final Report Content: Verify that the Final Report included root cause analysis and was properly attested. Provide the "
    "documented root cause and a reference URL to the Final Report or FCC's published analysis containing this information. "
    "All timeline verifications must be based on documented evidence from official FCC reports, the telecommunications provider's "
    "public disclosures, or authoritative news sources citing official information."
)

# Useful global additional instruction emphasizing source requirements
BASE_SOURCE_INSTRUCTION = (
    "Rely strictly on the provided webpage content. Prefer official FCC sources, provider public disclosures, or "
    "authoritative news articles that explicitly cite official information. If the cited page is irrelevant or does not "
    "contain the claimed information, conclude 'not supported'. Allow reasonable formatting variations in names/timestamps."
)

# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class WithSources(BaseModel):
    """A field value accompanied by supporting URLs."""
    value: Optional[str] = None
    urls: List[str] = Field(default_factory=list)


class OutageInfo(BaseModel):
    """Structured extraction for one identified outage case."""
    # Eligibility
    provider: Optional[WithSources] = None
    occurred_in_us: Optional[WithSources] = None
    outage_date: Optional[WithSources] = None
    wireless_outage: Optional[WithSources] = None

    # Core reportability details
    outage_duration_minutes: Optional[WithSources] = None
    outage_start_time: Optional[WithSources] = None
    outage_restoration_time: Optional[WithSources] = None
    user_minutes: Optional[WithSources] = None
    users_affected: Optional[WithSources] = None
    provider_subject_to_part4: Optional[WithSources] = None

    # NORS timeline
    discovery_time: Optional[WithSources] = None
    notification_submission_time: Optional[WithSources] = None
    initial_report_submission_time: Optional[WithSources] = None
    final_report_submission_time: Optional[WithSources] = None

    # Notification content per 4.11
    notification_content_elements: Optional[WithSources] = None

    # Final report content requirements
    final_root_cause: Optional[WithSources] = None
    final_updates_changes_vs_initial: Optional[WithSources] = None
    final_attestation: Optional[WithSources] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_outage_info() -> str:
    return (
        "Extract a single US wireless telecommunications outage (2020-01-01 through 2024-12-31) that the answer identifies "
        "and intends to verify for FCC Part 4 compliance. Return the following fields. For each field, include:\n"
        "- value: The textual value exactly as stated in the answer (if present).\n"
        "- urls: A list of one or more supporting URLs explicitly cited in the answer that document this field.\n\n"
        "FIELDS TO EXTRACT:\n"
        "Eligibility:\n"
        "- provider\n"
        "- occurred_in_us (e.g., 'United States' or 'US')\n"
        "- outage_date (prefer ISO 8601 if available)\n"
        "- wireless_outage (e.g., 'wireless', 'cellular network outage')\n\n"
        "Core reportability details:\n"
        "- outage_duration_minutes (e.g., '45 minutes')\n"
        "- outage_start_time (ISO 8601 if possible)\n"
        "- outage_restoration_time (ISO 8601 if possible)\n"
        "- user_minutes (e.g., '1,200,000 user-minutes')\n"
        "- users_affected (e.g., '600,000 users')\n"
        "- provider_subject_to_part4 (e.g., 'subject to FCC Part 4 outage reporting')\n\n"
        "NORS timeline:\n"
        "- discovery_time (documented time the provider discovered the outage)\n"
        "- notification_submission_time (documented Notification submission time to NORS)\n"
        "- initial_report_submission_time (documented Initial Report submission time to NORS)\n"
        "- final_report_submission_time (documented Final Report submission time to NORS)\n\n"
        "Notification content per 47 CFR § 4.11:\n"
        "- notification_content_elements (summary confirming the Notification included: reporting entity name; date/time of outage onset; brief description of problem; service effects; geographic area affected; contact name and telephone number)\n\n"
        "Final report content requirements:\n"
        "- final_root_cause (documented root cause)\n"
        "- final_updates_changes_vs_initial (evidence Final Report included information not contained in or changed from the Initial Report)\n"
        "- final_attestation (evidence Final Report was attested under oath by someone authorized to legally bind the provider)\n\n"
        "STRICT RULES:\n"
        "1) Extract only one outage case. If multiple are present, choose the one with the most complete timeline and evidence.\n"
        "2) For each field, include only URLs explicitly present in the answer. Do not invent or infer URLs.\n"
        "3) If a field is missing or not documented, set value=null and urls=[].\n"
        "4) Prefer official FCC/Provider sources or authoritative news citing official information.\n"
        "5) Use full URLs including protocol (http/https)."
    )


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def _val(ws: Optional[WithSources]) -> str:
    return (ws.value or "").strip() if ws else ""


def _urls(ws: Optional[WithSources]) -> List[str]:
    return ws.urls if (ws and ws.urls) else []


def _combine_sources(*lists: Optional[List[str]]) -> List[str]:
    combined: List[str] = []
    seen = set()
    for lst in lists:
        if not lst:
            continue
        for u in lst:
            if not u:
                continue
            if u not in seen:
                combined.append(u)
                seen.add(u)
    return combined


def _normalize_sources_arg(lst: List[str]) -> Union[str, List[str], None]:
    if not lst:
        return None
    if len(lst) == 1:
        return lst[0]
    return lst


# --------------------------------------------------------------------------- #
# Verification subtrees                                                       #
# --------------------------------------------------------------------------- #
async def verify_outage_eligibility(evaluator: Evaluator, parent_node, info: OutageInfo) -> None:
    """Build and verify the Outage_Eligibility subtree."""
    elig_node = evaluator.add_parallel(
        id="Outage_Eligibility",
        desc="Verify the identified outage matches the scenario required by the question (US, date range, wireless outage).",
        parent=parent_node,
        critical=True
    )

    # Occurred in United States
    leaf_us = evaluator.add_leaf(
        id="Occurred_in_United_States_with_URL",
        desc="Provide evidence (with supporting URL) that the outage occurred in the United States.",
        parent=elig_node,
        critical=True
    )
    claim_us = "This outage occurred in the United States."
    src_us = _normalize_sources_arg(_combine_sources(_urls(info.occurred_in_us), _urls(info.provider), _urls(info.outage_date)))
    await evaluator.verify(
        claim=claim_us,
        node=leaf_us,
        sources=src_us,
        additional_instruction=BASE_SOURCE_INSTRUCTION
    )

    # Date in range 2020-01-01 through 2024-12-31
    leaf_date = evaluator.add_leaf(
        id="Date_In_Range_2020_Through_2024_with_URL",
        desc="Provide evidence (with supporting URL) that the outage date falls between 2020-01-01 and 2024-12-31 (inclusive).",
        parent=elig_node,
        critical=True
    )
    date_val = _val(info.outage_date)
    claim_date = f"The outage date '{date_val}' falls between 2020-01-01 and 2024-12-31 (inclusive)."
    src_date = _normalize_sources_arg(_urls(info.outage_date))
    await evaluator.verify(
        claim=claim_date,
        node=leaf_date,
        sources=src_date,
        additional_instruction=BASE_SOURCE_INSTRUCTION
    )

    # Wireless telecommunications outage
    leaf_wireless = evaluator.add_leaf(
        id="Wireless_Telecommunications_Outage_with_URL",
        desc="Provide evidence (with supporting URL) that the event was a wireless telecommunications/network outage.",
        parent=elig_node,
        critical=True
    )
    claim_wireless = "The event was a wireless telecommunications or cellular network outage."
    src_wireless = _normalize_sources_arg(_combine_sources(_urls(info.wireless_outage), _urls(info.provider)))
    await evaluator.verify(
        claim=claim_wireless,
        node=leaf_wireless,
        sources=src_wireless,
        additional_instruction=BASE_SOURCE_INSTRUCTION
    )


async def verify_reportability_details(evaluator: Evaluator, parent_node, info: OutageInfo) -> None:
    """Build and verify the Reportability_And_Core_Outage_Details subtree."""
    rep_node = evaluator.add_parallel(
        id="Reportability_And_Core_Outage_Details",
        desc="Provide the requested outage details and verify the specified FCC reportability basis (≥30 minutes and ≥900,000 user-minutes) with supporting URLs.",
        parent=parent_node,
        critical=True
    )

    # Provider name with URL
    leaf_provider = evaluator.add_leaf(
        id="Provider_Name_with_URL",
        desc="State the wireless provider’s name and provide a supporting URL.",
        parent=rep_node,
        critical=True
    )
    provider_val = _val(info.provider)
    claim_provider = f"The wireless provider involved in this outage was '{provider_val}'."
    src_provider = _normalize_sources_arg(_urls(info.provider))
    await evaluator.verify(
        claim=claim_provider,
        node=leaf_provider,
        sources=src_provider,
        additional_instruction=BASE_SOURCE_INSTRUCTION
    )

    # Provider subject to Part 4 with URL
    leaf_part4 = evaluator.add_leaf(
        id="Provider_Subject_To_Part_4_with_URL",
        desc="Provide evidence (with supporting URL) that the provider is a qualifying communications provider subject to FCC Part 4 outage reporting (per the constraints).",
        parent=rep_node,
        critical=True
    )
    claim_part4 = f"The provider '{provider_val}' is a qualifying communications provider subject to FCC Part 4 outage reporting."
    src_part4 = _normalize_sources_arg(_combine_sources(_urls(info.provider_subject_to_part4), _urls(info.provider)))
    await evaluator.verify(
        claim=claim_part4,
        node=leaf_part4,
        sources=src_part4,
        additional_instruction=BASE_SOURCE_INSTRUCTION
    )

    # Outage duration with URL (either explicit duration or onset/restoration timestamps)
    leaf_duration = evaluator.add_leaf(
        id="Outage_Duration_with_URL",
        desc="Provide the documented outage duration (or onset/restoration timestamps sufficient to derive duration) with a supporting URL.",
        parent=rep_node,
        critical=True
    )
    dur_val = _val(info.outage_duration_minutes)
    start_val = _val(info.outage_start_time)
    rest_val = _val(info.outage_restoration_time)

    if dur_val:
        claim_duration = f"The documented outage duration was approximately {dur_val}."
        src_duration = _normalize_sources_arg(_urls(info.outage_duration_minutes))
    else:
        claim_duration = f"The outage onset time was '{start_val}' and restoration time was '{rest_val}', which together document the outage duration."
        src_duration = _normalize_sources_arg(_combine_sources(_urls(info.outage_start_time), _urls(info.outage_restoration_time)))
    await evaluator.verify(
        claim=claim_duration,
        node=leaf_duration,
        sources=src_duration,
        additional_instruction=BASE_SOURCE_INSTRUCTION
    )

    # User impact (user-minutes) with URL
    leaf_user_minutes = evaluator.add_leaf(
        id="User_Impact_User_Minutes_with_URL",
        desc="Provide the documented estimated user impact in user-minutes (or sufficient documented inputs to compute user-minutes) with a supporting URL.",
        parent=rep_node,
        critical=True
    )
    um_val = _val(info.user_minutes)
    users_val = _val(info.users_affected)
    dur_for_users_val = _val(info.outage_duration_minutes)

    if um_val:
        claim_um = f"The estimated user impact was approximately {um_val}."
        src_um = _normalize_sources_arg(_urls(info.user_minutes))
    else:
        claim_um = (
            f"The outage affected approximately {users_val} users for {dur_for_users_val}, providing sufficient inputs to compute user-minutes."
        )
        src_um = _normalize_sources_arg(_combine_sources(_urls(info.users_affected), _urls(info.outage_duration_minutes)))
    await evaluator.verify(
        claim=claim_um,
        node=leaf_user_minutes,
        sources=src_um,
        additional_instruction=BASE_SOURCE_INSTRUCTION
    )

    # Reportable per § 4.9 specified threshold (≥30 minutes and ≥900,000 user-minutes) with URL
    leaf_thresh = evaluator.add_leaf(
        id="Reportable_Per_4_9_Specified_Threshold_with_URL",
        desc="Using documented evidence, verify the outage meets the specified reportability basis stated in the proposed question: duration ≥ 30 minutes AND potentially affecting ≥ 900,000 user-minutes. Provide a supporting URL for the underlying facts used to verify this.",
        parent=rep_node,
        critical=True
    )
    claim_thresh = (
        "Based on the documented evidence, the outage lasted at least 30 minutes and potentially affected at least 900,000 user-minutes."
    )
    src_thresh = _normalize_sources_arg(
        _combine_sources(
            _urls(info.outage_duration_minutes),
            _urls(info.outage_start_time),
            _urls(info.outage_restoration_time),
            _urls(info.user_minutes),
            _urls(info.users_affected)
        )
    )
    await evaluator.verify(
        claim=claim_thresh,
        node=leaf_thresh,
        sources=src_thresh,
        additional_instruction=(
            BASE_SOURCE_INSTRUCTION +
            " Verify both thresholds from the provided URLs (duration ≥30 minutes; user-minutes ≥900,000). "
            "If user-minutes are not explicitly given, confirm the inputs (users affected and duration) suffice to exceed 900,000 user-minutes."
        )
    )


async def verify_nors_timeline(evaluator: Evaluator, parent_node, info: OutageInfo) -> None:
    """Build and verify the NORS_Submissions_And_Timeline_Compliance subtree."""
    nors_node = evaluator.add_sequential(
        id="NORS_Submissions_And_Timeline_Compliance",
        desc="Verify required NORS submissions and timeline compliance using documented discovery time and NORS submission timestamps.",
        parent=parent_node,
        critical=True
    )

    # Discovery time with URL
    leaf_discovery = evaluator.add_leaf(
        id="Discovery_Time_with_URL",
        desc="Provide the documented time the provider discovered the outage (used for Part 4 deadline calculations) with a supporting URL.",
        parent=nors_node,
        critical=True
    )
    discovery_val = _val(info.discovery_time)
    src_discovery = _normalize_sources_arg(_urls(info.discovery_time))
    claim_discovery = f"The provider discovered the outage at '{discovery_val}'."
    await evaluator.verify(
        claim=claim_discovery,
        node=leaf_discovery,
        sources=src_discovery,
        additional_instruction=BASE_SOURCE_INSTRUCTION
    )

    # Notification compliance (parallel under sequential parent)
    notif_node = evaluator.add_parallel(
        id="Notification_Compliance",
        desc="Verify Notification submission timing and required Notification content per 47 CFR § 4.11.",
        parent=nors_node,
        critical=True
    )

    # Notification submission time to NORS with URL
    leaf_notif_time = evaluator.add_leaf(
        id="Notification_Submission_Time_to_NORS_with_URL",
        desc="Provide the documented Notification submission time to the FCC’s Network Outage Reporting System (NORS) with a supporting URL.",
        parent=notif_node,
        critical=True
    )
    notif_val = _val(info.notification_submission_time)
    src_notif_time = _normalize_sources_arg(_urls(info.notification_submission_time))
    claim_notif_time = f"The Notification was submitted to NORS at '{notif_val}'."
    await evaluator.verify(
        claim=claim_notif_time,
        node=leaf_notif_time,
        sources=src_notif_time,
        additional_instruction=BASE_SOURCE_INSTRUCTION
    )

    # Notification within 120 minutes
    leaf_notif_120 = evaluator.add_leaf(
        id="Notification_Within_120_Minutes",
        desc="Using the documented discovery time and Notification submission time, verify Notification was submitted within 120 minutes of discovery per 47 CFR § 4.9.",
        parent=notif_node,
        critical=True
    )
    claim_notif_120 = (
        f"Based on the documented discovery time '{discovery_val}' and Notification submission time '{notif_val}', "
        "the Notification was submitted within 120 minutes of discovery."
    )
    src_notif_120 = _normalize_sources_arg(_combine_sources(_urls(info.discovery_time), _urls(info.notification_submission_time)))
    await evaluator.verify(
        claim=claim_notif_120,
        node=leaf_notif_120,
        sources=src_notif_120,
        additional_instruction=BASE_SOURCE_INSTRUCTION
    )

    # Notification content per 4.11 with URL
    leaf_notif_content = evaluator.add_leaf(
        id="Notification_Content_Per_4_11_with_URL",
        desc="Provide evidence (with supporting URL) that the Notification included the 47 CFR § 4.11 elements listed in the constraints: reporting entity name; date/time of outage onset; brief description of problem; service effects; geographic area affected; contact name and telephone number.",
        parent=notif_node,
        critical=True
    )
    claim_notif_content = (
        "The Notification included the 47 CFR § 4.11 elements: reporting entity name; date/time of outage onset; brief description "
        "of the problem; service effects; geographic area affected; contact name and telephone number."
    )
    src_notif_content = _normalize_sources_arg(_urls(info.notification_content_elements))
    await evaluator.verify(
        claim=claim_notif_content,
        node=leaf_notif_content,
        sources=src_notif_content,
        additional_instruction=BASE_SOURCE_INSTRUCTION
    )

    # Initial report timeline (parallel)
    init_node = evaluator.add_parallel(
        id="Initial_Report_Timeline",
        desc="Verify Initial Communications Outage Report timing compliance (≤72 hours from discovery) with supporting URLs.",
        parent=nors_node,
        critical=True
    )

    leaf_initial_time = evaluator.add_leaf(
        id="Initial_Report_Submission_Time_to_NORS_with_URL",
        desc="Provide the documented Initial Report submission time to NORS with a supporting URL.",
        parent=init_node,
        critical=True
    )
    init_val = _val(info.initial_report_submission_time)
    src_initial_time = _normalize_sources_arg(_urls(info.initial_report_submission_time))
    claim_initial_time = f"The Initial Communications Outage Report was submitted to NORS at '{init_val}'."
    await evaluator.verify(
        claim=claim_initial_time,
        node=leaf_initial_time,
        sources=src_initial_time,
        additional_instruction=BASE_SOURCE_INSTRUCTION
    )

    leaf_initial_72 = evaluator.add_leaf(
        id="Initial_Report_Within_72_Hours",
        desc="Using the documented discovery time and Initial Report submission time, verify the Initial Report was submitted not later than 72 hours after discovery per 47 CFR § 4.9.",
        parent=init_node,
        critical=True
    )
    claim_initial_72 = (
        f"Based on the documented discovery time '{discovery_val}' and Initial Report submission time '{init_val}', "
        "the Initial Report was submitted within 72 hours of discovery."
    )
    src_initial_72 = _normalize_sources_arg(_combine_sources(_urls(info.discovery_time), _urls(info.initial_report_submission_time)))
    await evaluator.verify(
        claim=claim_initial_72,
        node=leaf_initial_72,
        sources=src_initial_72,
        additional_instruction=BASE_SOURCE_INSTRUCTION
    )

    # Final report timeline (parallel)
    final_node = evaluator.add_parallel(
        id="Final_Report_Timeline",
        desc="Verify Final Communications Outage Report timing compliance (≤30 days from discovery) with supporting URLs.",
        parent=nors_node,
        critical=True
    )

    leaf_final_time = evaluator.add_leaf(
        id="Final_Report_Submission_Time_to_NORS_with_URL",
        desc="Provide the documented Final Report submission time to NORS with a supporting URL.",
        parent=final_node,
        critical=True
    )
    final_val = _val(info.final_report_submission_time)
    src_final_time = _normalize_sources_arg(_urls(info.final_report_submission_time))
    claim_final_time = f"The Final Communications Outage Report was submitted to NORS at '{final_val}'."
    await evaluator.verify(
        claim=claim_final_time,
        node=leaf_final_time,
        sources=src_final_time,
        additional_instruction=BASE_SOURCE_INSTRUCTION
    )

    leaf_final_30 = evaluator.add_leaf(
        id="Final_Report_Within_30_Days",
        desc="Using the documented discovery time and Final Report submission time, verify the Final Report was submitted not later than 30 days after discovery per 47 CFR § 4.9.",
        parent=final_node,
        critical=True
    )
    claim_final_30 = (
        f"Based on the documented discovery time '{discovery_val}' and Final Report submission time '{final_val}', "
        "the Final Report was submitted within 30 days of discovery."
    )
    src_final_30 = _normalize_sources_arg(_combine_sources(_urls(info.discovery_time), _urls(info.final_report_submission_time)))
    await evaluator.verify(
        claim=claim_final_30,
        node=leaf_final_30,
        sources=src_final_30,
        additional_instruction=BASE_SOURCE_INSTRUCTION
    )


async def verify_final_report_content(evaluator: Evaluator, parent_node, info: OutageInfo) -> None:
    """Build and verify the Final_Report_Content_Requirements subtree."""
    content_node = evaluator.add_parallel(
        id="Final_Report_Content_Requirements",
        desc="Verify Final Report content requirements requested by the question and listed in constraints (root cause, updates/changes vs Initial, and attestation) with supporting URLs.",
        parent=parent_node,
        critical=True
    )

    # Root cause
    leaf_root = evaluator.add_leaf(
        id="Final_Report_Root_Cause_with_URL",
        desc="Provide the documented root cause and a supporting URL to the Final Report or FCC-published analysis containing the root cause information.",
        parent=content_node,
        critical=True
    )
    root_val = _val(info.final_root_cause)
    src_root = _normalize_sources_arg(_urls(info.final_root_cause))
    claim_root = f"The documented root cause of the outage was: {root_val}."
    await evaluator.verify(
        claim=claim_root,
        node=leaf_root,
        sources=src_root,
        additional_instruction=BASE_SOURCE_INSTRUCTION
    )

    # Updates/Changes vs Initial
    leaf_updates = evaluator.add_leaf(
        id="Final_Report_Includes_Updates_Or_Changes_vs_Initial_with_URL",
        desc="Provide evidence (with supporting URL) that the Final Report included information not contained in or changed from the Initial Report (as required by 47 CFR § 4.11 per the constraints).",
        parent=content_node,
        critical=True
    )
    src_updates = _normalize_sources_arg(_urls(info.final_updates_changes_vs_initial))
    claim_updates = (
        "The Final Report included information not contained in or changed from the Initial Report, consistent with 47 CFR § 4.11."
    )
    await evaluator.verify(
        claim=claim_updates,
        node=leaf_updates,
        sources=src_updates,
        additional_instruction=BASE_SOURCE_INSTRUCTION
    )

    # Attestation
    leaf_attest = evaluator.add_leaf(
        id="Final_Report_Attestation_with_URL",
        desc="Provide evidence (with supporting URL) that the Final Report was attested under oath as true/complete/accurate by a person authorized to legally bind the provider per 47 CFR § 4.11.",
        parent=content_node,
        critical=True
    )
    src_attest = _normalize_sources_arg(_urls(info.final_attestation))
    claim_attest = (
        "The Final Report was attested under oath by a person authorized to legally bind the provider, "
        "affirming that the report is true, complete, and accurate."
    )
    await evaluator.verify(
        claim=claim_attest,
        node=leaf_attest,
        sources=src_attest,
        additional_instruction=BASE_SOURCE_INSTRUCTION
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
    """
    Evaluate an answer for FCC Part 4 outage compliance across eligibility, reportability details,
    NORS timelines, and final report content.
    """
    # Initialize evaluator (root sequential to reflect ordered compliance checks)
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

    # Create overall critical node under root to enforce failure propagation
    overall = evaluator.add_sequential(
        id="Overall_FCC_Part_4_Compliance",
        desc=("Identify an eligible US wireless outage (2020–2024) that is reportable under the specified 47 CFR § 4.9 threshold "
              "(≥30 minutes and ≥900,000 user-minutes) and verify FCC Part 4 (including 4.9 timelines and 4.11 content) "
              "compliance using documented evidence and supporting URLs."),
        parent=root,
        critical=True
    )

    # Extract structured outage info from the answer
    outage_info = await evaluator.extract(
        prompt=prompt_extract_outage_info(),
        template_class=OutageInfo,
        extraction_name="outage_info_extraction"
    )

    # Optionally record constraints as ground truth context for clarity
    evaluator.add_ground_truth({
        "date_range": "2020-01-01 to 2024-12-31 (inclusive)",
        "reportability_thresholds": {
            "duration_minutes": ">= 30 minutes",
            "user_minutes": ">= 900,000 user-minutes"
        },
        "timeline_requirements": {
            "notification": "<= 120 minutes from discovery",
            "initial_report": "<= 72 hours from discovery",
            "final_report": "<= 30 days from discovery"
        },
        "content_requirements": [
            "Notification includes elements per 47 CFR § 4.11",
            "Final Report includes root cause, updates/changes vs Initial, and attestation"
        ]
    })

    # Build and verify subtrees in required order (sequential gating)
    await verify_outage_eligibility(evaluator, overall, outage_info)
    await verify_reportability_details(evaluator, overall, outage_info)
    await verify_nors_timeline(evaluator, overall, outage_info)
    await verify_final_report_content(evaluator, overall, outage_info)

    # Return standardized evaluation summary
    return evaluator.get_summary()