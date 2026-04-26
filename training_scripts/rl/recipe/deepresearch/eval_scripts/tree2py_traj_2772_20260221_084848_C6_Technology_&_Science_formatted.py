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
TASK_ID = "verizon_outage_2026_reg_compliance"
TASK_DESCRIPTION = """
On January 14, 2026, a major U.S. wireless telecommunications carrier experienced a nationwide service outage that lasted approximately 10 hours, affecting over 1.5 million customers in major metropolitan areas including Atlanta, New York City, Charlotte, Houston, Dallas, Philadelphia, Miami, and Ashburn. The outage began around noon Eastern Time, with many customers' phones displaying 'SOS only' mode, and service was not fully restored until approximately 10:20 PM ET. The carrier subsequently attributed the outage to a software issue and offered $20 credits to affected customers.

For this outage, verify the carrier's compliance with FCC regulatory requirements by providing the following information:

1. FCC Reporting Threshold Verification: Confirm that the outage met the mandatory reporting thresholds under 47 CFR § 4.9 for wireless carriers (minimum 30-minute duration and at least one of the following: affecting ≥900,000 user-minutes, ≥667 OC3-minutes, or potentially affecting 911/988 special facilities). Include a calculation or determination showing which threshold(s) were met.

2. FCC Notification Timeline Compliance: Verify whether the carrier complied with the required notification timeline of submitting a Notification to the FCC within 120 minutes of discovering the outage, an Initial Communications Outage Report within 72 hours, and a Final Communications Outage Report within 30 days (or commitment to do so).

3. PSAP Emergency Notification Compliance: Determine whether the outage potentially affected 911 special facilities, and if so, verify whether the carrier complied with the requirement to notify affected Public Safety Answering Points (PSAPs) within 30 minutes of discovery and provide follow-up notifications within 2 hours of initial contact.

4. Root Cause Documentation: Confirm that the carrier identified and publicly disclosed the root cause of the outage, and specify what that root cause was (software issue, hardware failure, cyberattack, or other).

5. Customer Remediation: Document the compensation offered to affected customers, including the amount and how customers could claim it.

6. Industry Reliability Standard Context: Define the telecommunications industry 'five nines' (99.999%) reliability standard in terms of maximum allowable annual downtime, and compare this single outage's duration to that annual allowance to provide context on whether this event alone would cause the carrier to fall below the industry gold standard for the year.

For each verification point, provide supporting URL references from official sources, news reports, or the carrier's public statements.
""".strip()


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class OutageComplianceExtraction(BaseModel):
    # Core outage facts
    carrier_name: Optional[str] = None
    outage_date: Optional[str] = None
    start_time_et: Optional[str] = None
    end_time_et: Optional[str] = None
    duration: Optional[str] = None  # e.g., "about 10 hours" or "600 minutes"
    affected_customers: Optional[str] = None  # e.g., "over 1.5 million"
    affected_cities: List[str] = Field(default_factory=list)

    # FCC timeline facts
    discovery_time_et: Optional[str] = None
    fcc_notification_time: Optional[str] = None  # if explicitly stated in the answer
    initial_report_time: Optional[str] = None  # if explicitly stated
    final_report_status: Optional[str] = None  # e.g., "committed to submit within 30 days", "submitted on ..."

    # PSAP/911
    impact_911: Optional[str] = None  # "yes", "no", or "unknown"
    psap_notification_time: Optional[str] = None
    psap_followup_time: Optional[str] = None

    # Root cause and remediation
    root_cause: Optional[str] = None  # e.g., "software issue"
    compensation_amount: Optional[str] = None  # e.g., "$20"
    compensation_method: Optional[str] = None  # e.g., "automatic bill credit", "claim portal", etc.

    # URL sources by category (only URLs explicitly present in the answer)
    urls_outage: List[str] = Field(default_factory=list)  # general outage reporting sources
    urls_thresholds: List[str] = Field(default_factory=list)  # CFR/FCC docs about thresholds
    urls_timeline: List[str] = Field(default_factory=list)  # sources about FCC notifications/reports timeline
    urls_psap: List[str] = Field(default_factory=list)  # sources about 911 impact/PSAP notifications
    urls_root_cause: List[str] = Field(default_factory=list)  # sources stating root cause
    urls_compensation: List[str] = Field(default_factory=list)  # sources on compensation details
    urls_reliability_standard: List[str] = Field(default_factory=list)  # five nines definition sources
    urls_duration: List[str] = Field(default_factory=list)  # explicit duration sources if separate


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_outage_compliance() -> str:
    return """
    Extract structured information about the January 14, 2026 wireless outage and compliance details as explicitly stated in the answer. Do not infer or add information not present in the answer text. When extracting URLs, include only those explicitly present in the answer (plain or markdown links). If some fields are not mentioned, return null or an empty list as appropriate.

    Required fields:
    - carrier_name: name of the carrier involved (e.g., Verizon)
    - outage_date: the date (e.g., "January 14, 2026")
    - start_time_et: approximate start time in ET if given (e.g., "around noon ET")
    - end_time_et: approximate end time in ET if given (e.g., "about 10:20 PM ET")
    - duration: outage duration as stated (e.g., "about 10 hours" or "600 minutes")
    - affected_customers: number or description of affected customers (e.g., "over 1.5 million")
    - affected_cities: list of cities explicitly named

    FCC timeline:
    - discovery_time_et: when the carrier became aware / when reports began (string if mentioned)
    - fcc_notification_time: when (if stated) the carrier submitted the initial Notification to FCC NORS (string; null if not stated)
    - initial_report_time: when (if stated) the Initial Report was submitted (string; null if not stated)
    - final_report_status: any statement about the Final Report status or commitment (string; null if not stated)

    PSAP/911:
    - impact_911: "yes", "no", or "unknown" based on whether the answer says the outage potentially affected 911/988 special facilities
    - psap_notification_time: when (if stated) PSAPs were first notified
    - psap_followup_time: when (if stated) follow-up notifications were made

    Root cause and remediation:
    - root_cause: specific cause as stated (e.g., "software issue", "hardware failure", "cyberattack") or null
    - compensation_amount: amount offered per customer (e.g., "$20") or null
    - compensation_method: how customers could receive/claim compensation (e.g., "automatic bill credit", "credit code via app") or null

    URLs (only include URLs explicitly present in the answer):
    - urls_outage: URLs about the outage overview, impact, cities, general coverage
    - urls_thresholds: URLs that describe FCC 47 CFR § 4.9 reporting thresholds for wireless carriers
    - urls_timeline: URLs that discuss FCC notifications/reports timing for this incident (Notification ≤120 min, Initial ≤72 hours, Final ≤30 days)
    - urls_psap: URLs about 911/988 impact and PSAP notifications for this incident
    - urls_root_cause: URLs where the carrier or credible sources stated the root cause
    - urls_compensation: URLs where the carrier or credible sources stated the compensation details and how to claim
    - urls_reliability_standard: URLs that define the "five nines" (99.999%) reliability standard and its annual downtime allowance
    - urls_duration: URLs that state this outage's duration or start/end times (if separate from urls_outage)
    """.strip()


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _combine_urls(*url_lists: Optional[List[str]]) -> List[str]:
    combined: List[str] = []
    seen = set()
    for lst in url_lists:
        if not lst:
            continue
        for u in lst:
            if u and u not in seen:
                combined.append(u)
                seen.add(u)
    return combined


def _has_urls(urls: Optional[List[str]]) -> bool:
    return bool(urls and len(urls) > 0)


# --------------------------------------------------------------------------- #
# Verification subtrees                                                       #
# --------------------------------------------------------------------------- #
async def verify_fcc_reporting_thresholds(evaluator: Evaluator, parent, data: OutageComplianceExtraction) -> None:
    """
    FCC Reporting Threshold Verification under 47 CFR § 4.9 (wireless):
    - Duration ≥ 30 minutes
    - At least one of: ≥900,000 user-minutes OR ≥667 OC3-minutes OR 911/988 potentially affected
    - Provide URL reference for the threshold definition
    """
    carrier = data.carrier_name or "the carrier"
    threshold_node = evaluator.add_parallel(
        id="FCC_Reporting_Threshold_Met",
        desc="Verify that the outage met FCC threshold criteria under 47 CFR § 4.9 requiring mandatory reporting",
        parent=parent,
        critical=True  # Critical compliance dimension
    )

    # Duration ≥ 30 minutes
    duration_leaf = evaluator.add_leaf(
        id="Duration_Threshold",
        desc="The outage lasted at least 30 minutes",
        parent=threshold_node,
        critical=True
    )
    duration_sources = _combine_urls(data.urls_duration, data.urls_outage)
    duration_claim = (
        f"Public reporting indicates that {carrier}'s January 14, 2026 outage lasted approximately "
        f"{data.duration or '10 hours'} (which is ≥ 30 minutes)."
    )
    await evaluator.verify(
        claim=duration_claim,
        node=duration_leaf,
        sources=duration_sources,
        additional_instruction="Verify that sources explicitly or implicitly show the outage lasted at least 30 minutes."
    )

    # At least one impact-scale threshold satisfied: prefer user-minutes (≥900,000)
    impact_leaf = evaluator.add_leaf(
        id="Impact_Scale_Threshold_Satisfied",
        desc="At least one of the quantitative impact thresholds was met (e.g., ≥900,000 user-minutes)",
        parent=threshold_node,
        critical=True
    )
    impact_sources = _combine_urls(data.urls_outage, data.urls_duration, data.urls_thresholds)
    affected_text = data.affected_customers or "over 1.5 million"
    duration_text = data.duration or "about 10 hours (≈600 minutes)"
    impact_claim = (
        f"Based on sources reporting that {affected_text} customers were affected and that the outage lasted {duration_text}, "
        f"the outage exceeded 900,000 user-minutes (e.g., 1,500,000 × 600 = 900,000,000 ≥ 900,000), satisfying "
        f"the user-minutes threshold in 47 CFR § 4.9 for wireless carriers."
    )
    await evaluator.verify(
        claim=impact_claim,
        node=impact_leaf,
        sources=impact_sources,
        additional_instruction="Check the reported number of affected customers and duration, compute user-minutes, and compare to the 900,000 user-minutes threshold."
    )

    # Threshold documentation URL must be provided
    threshold_url_present = evaluator.add_custom_node(
        result=_has_urls(data.urls_thresholds),
        id="Threshold_Documentation_URL_Provided",
        desc="At least one URL is provided that documents the FCC 47 CFR § 4.9 reporting thresholds",
        parent=threshold_node,
        critical=True
    )
    threshold_url_leaf = evaluator.add_leaf(
        id="Threshold_Documentation_URL_Supports",
        desc="Provided URL(s) document the FCC 47 CFR § 4.9 reporting thresholds for wireless carriers",
        parent=threshold_node,
        critical=True
    )
    threshold_doc_claim = (
        "These sources state the FCC outage reporting thresholds for wireless carriers under 47 CFR § 4.9: "
        "minimum duration of 30 minutes and at least one of the following—≥900,000 user-minutes, ≥667 OC3-minutes, "
        "or potentially affecting a 911/988 special facility."
    )
    await evaluator.verify(
        claim=threshold_doc_claim,
        node=threshold_url_leaf,
        sources=data.urls_thresholds,
        additional_instruction="Verify that the cited source(s) describe the outage reporting thresholds for wireless carriers."
    )


async def verify_fcc_timeline_compliance(evaluator: Evaluator, parent, data: OutageComplianceExtraction) -> None:
    """
    FCC Notification Timeline Compliance (47 CFR § 4.9):
    - Notification within 120 minutes of discovery
    - Initial report within 72 hours
    - Final report within 30 days (or commitment)
    Note: Specific NORS timestamps are often non-public; rely on explicit statements from official or credible sources where available.
    """
    carrier = data.carrier_name or "the carrier"
    timeline_node = evaluator.add_sequential(
        id="FCC_Notification_Timeline_Compliance",
        desc="Verify that the carrier complied with FCC-mandated notification and reporting timelines under 47 CFR § 4.9",
        parent=parent,
        critical=False  # Allow partial credit if public documentation is limited
    )

    # 1) Initial Notification within 120 minutes
    initial_group = evaluator.add_parallel(
        id="Initial_Notification_120_Minutes",
        desc="The carrier submitted a Notification to the FCC within 120 minutes of discovering the outage",
        parent=timeline_node,
        critical=False
    )
    init_url_present = evaluator.add_custom_node(
        result=_has_urls(data.urls_timeline),
        id="Timeline_Documentation_URL_Provided",
        desc="At least one URL is provided for FCC notification timeline information",
        parent=initial_group,
        critical=True
    )
    init_notify_leaf = evaluator.add_leaf(
        id="Timeline_Notification_Compliance",
        desc="Notification to FCC within 120 minutes of discovery is supported",
        parent=initial_group,
        critical=True
    )
    init_notify_claim = (
        f"Public statements or credible reporting indicate that {carrier} submitted the required FCC Notification "
        f"within 120 minutes of discovering the outage (or states compliance with this requirement)."
    )
    await evaluator.verify(
        claim=init_notify_claim,
        node=init_notify_leaf,
        sources=data.urls_timeline,
        additional_instruction="Look for statements such as 'we notified the FCC' and any timing context supporting ≤120 minutes."
    )

    # 2) Initial Report within 72 hours
    initial_report_group = evaluator.add_parallel(
        id="Initial_Report_72_Hours",
        desc="The carrier submitted an Initial Communications Outage Report within 72 hours of discovering the outage",
        parent=timeline_node,
        critical=False
    )
    init_report_url_present = evaluator.add_custom_node(
        result=_has_urls(data.urls_timeline),
        id="Initial_Report_URL_Provided",
        desc="At least one URL is provided for Initial Report timeline information",
        parent=initial_report_group,
        critical=True
    )
    init_report_leaf = evaluator.add_leaf(
        id="Initial_Report_Submitted_Within_72_Hours",
        desc="Initial Communications Outage Report within 72 hours is supported",
        parent=initial_report_group,
        critical=True
    )
    init_report_claim = (
        f"Public statements or credible reporting indicate that {carrier} submitted the Initial Communications "
        f"Outage Report within 72 hours of discovery (or explicitly committed to do so)."
    )
    await evaluator.verify(
        claim=init_report_claim,
        node=init_report_leaf,
        sources=data.urls_timeline,
        additional_instruction="Confirm the 72-hour Initial Report requirement is addressed (submitted or committed)."
    )

    # 3) Final Report within 30 days (or commitment)
    final_report_group = evaluator.add_parallel(
        id="Final_Report_30_Days",
        desc="The carrier submitted (or committed to submit) a Final Communications Outage Report within 30 days",
        parent=timeline_node,
        critical=False
    )
    final_report_url_present = evaluator.add_custom_node(
        result=_has_urls(data.urls_timeline),
        id="Final_Report_URL_Provided",
        desc="At least one URL is provided for Final Report information or carrier commitment",
        parent=final_report_group,
        critical=True
    )
    final_report_leaf = evaluator.add_leaf(
        id="Final_Report_Compliance_or_Commitment",
        desc="Final Communications Outage Report within 30 days is supported (submitted or committed)",
        parent=final_report_group,
        critical=True
    )
    final_report_claim = (
        f"Public statements or credible reporting indicate that {carrier} submitted the Final Communications "
        f"Outage Report within 30 days of discovery, or explicitly committed to submit within 30 days."
    )
    await evaluator.verify(
        claim=final_report_claim,
        node=final_report_leaf,
        sources=data.urls_timeline,
        additional_instruction="Confirm that the 30-day Final Report requirement is addressed by submission or explicit commitment."
    )


async def verify_psap_notification_compliance(evaluator: Evaluator, parent, data: OutageComplianceExtraction) -> None:
    """
    PSAP Emergency Notification Compliance (conditional):
    - Determine whether 911/988 special facilities were potentially affected.
    - If yes, verify notification to PSAPs within 30 minutes and follow-up within 2 hours.
    This section uses sequential gating: if 911 impact is not supported, subsequent PSAP requirements are skipped.
    """
    carrier = data.carrier_name or "the carrier"
    psap_node = evaluator.add_sequential(
        id="PSAP_Emergency_Notification_Compliance",
        desc="If the outage potentially affected 911 special facilities, verify the carrier complied with PSAP notification requirements",
        parent=parent,
        critical=False  # Conditional applicability; allow partial scoring
    )

    # First: Determine whether 911/988 facilities were potentially affected (gating)
    impact_group = evaluator.add_parallel(
        id="911_Impact_Determination",
        desc="Determine whether the outage potentially affected 911 special facilities",
        parent=psap_node,
        critical=True  # Gate the rest: must determine impact to proceed
    )
    impact_urls = _combine_urls(data.urls_psap, data.urls_outage)
    impact_url_present = evaluator.add_custom_node(
        result=_has_urls(impact_urls),
        id="Impact_Determination_URL_Provided",
        desc="At least one URL is provided for 911 impact information",
        parent=impact_group,
        critical=True
    )
    impact_leaf = evaluator.add_leaf(
        id="Special_Facility_Impact_Determined",
        desc="Evidence indicates the outage potentially affected 911 or 988 special facilities",
        parent=impact_group,
        critical=True
    )
    impact_claim = (
        f"Public reporting indicates that the January 14, 2026 outage by {carrier} potentially affected 911 or 988 special facilities."
    )
    await evaluator.verify(
        claim=impact_claim,
        node=impact_leaf,
        sources=impact_urls,
        additional_instruction="Verify whether sources explicitly indicate 911/988 was impacted (potentially or actually)."
    )

    # If impacted, verify PSAP notifications within 30 minutes
    psap_notify_group = evaluator.add_parallel(
        id="PSAP_Notification_30_Minutes",
        desc="If 911 facilities were affected, the carrier notified affected PSAPs within 30 minutes of discovery",
        parent=psap_node,
        critical=False
    )
    psap_notify_url_present = evaluator.add_custom_node(
        result=_has_urls(data.urls_psap),
        id="PSAP_Notification_URL_Provided",
        desc="At least one URL is provided for PSAP notification information",
        parent=psap_notify_group,
        critical=True
    )
    psap_notify_leaf = evaluator.add_leaf(
        id="PSAP_Notification_Timing_Verified",
        desc="Carrier notified PSAPs within 30 minutes of discovery",
        parent=psap_notify_group,
        critical=True
    )
    psap_notify_claim = (
        f"For this outage, sources indicate {carrier} notified affected PSAPs within 30 minutes of discovery (if 911 was affected)."
    )
    await evaluator.verify(
        claim=psap_notify_claim,
        node=psap_notify_leaf,
        sources=data.urls_psap,
        additional_instruction="Verify explicit mention of PSAP notifications and that the first contact was within 30 minutes.",
    )

    # Follow-up within 2 hours
    psap_follow_group = evaluator.add_parallel(
        id="PSAP_Followup_2_Hours",
        desc="If 911 facilities were affected, the carrier provided the first follow-up notification within 2 hours of initial contact",
        parent=psap_node,
        critical=False
    )
    psap_follow_url_present = evaluator.add_custom_node(
        result=_has_urls(data.urls_psap),
        id="Followup_URL_Provided",
        desc="At least one URL is provided for follow-up notification information",
        parent=psap_follow_group,
        critical=True
    )
    psap_follow_leaf = evaluator.add_leaf(
        id="Followup_Notification_Evidence_Verified",
        desc="Evidence of follow-up notifications to PSAPs within 2 hours",
        parent=psap_follow_group,
        critical=False
    )
    psap_follow_claim = (
        f"For this outage, sources indicate {carrier} provided follow-up notifications to PSAPs within 2 hours of the initial contact (if 911 was affected)."
    )
    await evaluator.verify(
        claim=psap_follow_claim,
        node=psap_follow_leaf,
        sources=data.urls_psap,
        additional_instruction="Verify mention of follow-up to PSAPs and that timing aligns with ≤2 hours from initial contact.",
    )


async def verify_root_cause_documentation(evaluator: Evaluator, parent, data: OutageComplianceExtraction) -> None:
    """
    Root Cause Documentation:
    - Carrier identified and disclosed the root cause.
    - Confirm it was a software issue (and not a cyberattack).
    - Provide URL reference(s).
    """
    carrier = data.carrier_name or "the carrier"
    root_cause_node = evaluator.add_parallel(
        id="Root_Cause_Documentation",
        desc="Verify that the carrier documented and disclosed the root cause of the outage",
        parent=parent,
        critical=True
    )

    root_url_present = evaluator.add_custom_node(
        result=_has_urls(data.urls_root_cause),
        id="Root_Cause_URL_Provided",
        desc="At least one URL is provided for root cause information",
        parent=root_cause_node,
        critical=True
    )

    cause_identified_leaf = evaluator.add_leaf(
        id="Root_Cause_Identified",
        desc="The carrier identified and disclosed the root cause",
        parent=root_cause_node,
        critical=True
    )
    cause_identified_claim = (
        f"Sources indicate that {carrier} publicly identified and disclosed the root cause of the January 14, 2026 outage."
    )
    await evaluator.verify(
        claim=cause_identified_claim,
        node=cause_identified_leaf,
        sources=data.urls_root_cause,
        additional_instruction="Verify a direct statement identifying the cause (not just speculation)."
    )

    software_issue_leaf = evaluator.add_leaf(
        id="Software_Issue_Confirmation",
        desc="The disclosed root cause was a software issue (not a cyberattack)",
        parent=root_cause_node,
        critical=True
    )
    software_issue_claim = (
        f"Sources indicate the outage root cause was a software issue (and not a cyberattack)."
    )
    await evaluator.verify(
        claim=software_issue_claim,
        node=software_issue_leaf,
        sources=data.urls_root_cause,
        additional_instruction="Confirm that the identified cause is a software issue and not a cyberattack."
    )


async def verify_customer_remediation(evaluator: Evaluator, parent, data: OutageComplianceExtraction) -> None:
    """
    Customer Remediation and Communications:
    - Public acknowledgment of the outage.
    - Compensation amount and method; include URLs.
    """
    carrier = data.carrier_name or "the carrier"
    remediation_node = evaluator.add_parallel(
        id="Customer_Remediation",
        desc="Verify that the carrier provided appropriate customer notification and remediation",
        parent=parent,
        critical=False
    )

    # Customer communication
    comms_group = evaluator.add_parallel(
        id="Customer_Communication",
        desc="The carrier publicly acknowledged the outage and communicated with affected customers",
        parent=remediation_node,
        critical=False
    )
    comms_url_present = evaluator.add_custom_node(
        result=_has_urls(data.urls_outage) or _has_urls(data.urls_compensation),
        id="Communication_URL_Provided",
        desc="At least one URL is provided for customer communication or public acknowledgment",
        parent=comms_group,
        critical=True
    )
    comms_leaf = evaluator.add_leaf(
        id="Public_Acknowledgment_Verified",
        desc="The carrier issued public statements acknowledging the outage",
        parent=comms_group,
        critical=False
    )
    comms_claim = (
        f"Sources show that {carrier} publicly acknowledged the January 14, 2026 outage (e.g., statements on website or social channels, or press coverage quoting the carrier)."
    )
    await evaluator.verify(
        claim=comms_claim,
        node=comms_leaf,
        sources=_combine_urls(data.urls_outage, data.urls_compensation),
        additional_instruction="Look for explicit acknowledgment from the carrier."
    )

    # Compensation
    comp_group = evaluator.add_parallel(
        id="Compensation_Offered",
        desc="The carrier offered compensation to affected customers",
        parent=remediation_node,
        critical=False
    )
    comp_url_present = evaluator.add_custom_node(
        result=_has_urls(data.urls_compensation),
        id="Compensation_URL_Provided",
        desc="At least one URL is provided for compensation information",
        parent=comp_group,
        critical=True
    )
    comp_amount_leaf = evaluator.add_leaf(
        id="Compensation_Amount_Verified",
        desc="Identify the compensation amount offered per affected customer",
        parent=comp_group,
        critical=False
    )
    comp_amount_text = data.compensation_amount or "$20"
    comp_amount_claim = (
        f"Sources indicate the compensation amount offered per affected customer was {comp_amount_text}."
    )
    await evaluator.verify(
        claim=comp_amount_claim,
        node=comp_amount_leaf,
        sources=data.urls_compensation,
        additional_instruction="Verify the dollar amount offered to affected customers (e.g., $20 credit)."
    )

    comp_method_leaf = evaluator.add_leaf(
        id="Compensation_Method_Verified",
        desc="Describe how customers can claim or receive the compensation",
        parent=comp_group,
        critical=False
    )
    comp_method_text = data.compensation_method or "a bill credit offered by the carrier"
    comp_method_claim = (
        f"Sources explain how customers could receive or claim the compensation (e.g., {comp_method_text})."
    )
    await evaluator.verify(
        claim=comp_method_claim,
        node=comp_method_leaf,
        sources=data.urls_compensation,
        additional_instruction="Look for details on the mechanism to deliver or claim the credit."
    )


async def verify_reliability_context(evaluator: Evaluator, parent, data: OutageComplianceExtraction) -> None:
    """
    Industry Reliability Standard Context:
    - Define 'five nines' (99.999%) and annual downtime allowance (≈5.26 minutes).
    - Compare this outage duration (~10 hours) to that allowance.
    """
    carrier = data.carrier_name or "the carrier"
    reliability_node = evaluator.add_parallel(
        id="Annual_Reliability_Standard_Context",
        desc="Provide context on whether this single outage caused the carrier to fall below the industry 'five nines' (99.999%) annual reliability standard",
        parent=parent,
        critical=False
    )

    # Industry standard definition
    std_group = evaluator.add_parallel(
        id="Industry_Standard_Definition",
        desc="Define the telecommunications industry 'five nines' reliability standard in terms of maximum allowable annual downtime",
        parent=reliability_node,
        critical=False
    )
    std_url_present = evaluator.add_custom_node(
        result=_has_urls(data.urls_reliability_standard),
        id="Standard_Definition_URL_Provided",
        desc="At least one URL is provided for the five nines standard definition",
        parent=std_group,
        critical=True
    )
    std_pct_leaf = evaluator.add_leaf(
        id="Five_Nines_Percentage",
        desc="State that 'five nines' means 99.999% uptime",
        parent=std_group,
        critical=False
    )
    await evaluator.verify(
        claim="In telecommunications reliability, 'five nines' refers to 99.999% uptime.",
        node=std_pct_leaf,
        sources=data.urls_reliability_standard,
        additional_instruction="Verify that the provided source(s) define five nines as 99.999% uptime."
    )
    std_limit_leaf = evaluator.add_leaf(
        id="Annual_Downtime_Limit",
        desc="State that 99.999% uptime allows maximum ~5.26 minutes of downtime per year",
        parent=std_group,
        critical=False
    )
    await evaluator.verify(
        claim="Five nines (99.999%) uptime corresponds to approximately 5.26 minutes of downtime per year.",
        node=std_limit_leaf,
        sources=data.urls_reliability_standard,
        additional_instruction="Verify the annual downtime allowance commonly cited for 99.999% availability."
    )

    # Outage duration comparison
    compare_group = evaluator.add_parallel(
        id="Outage_Duration_Comparison",
        desc="Compare this outage's duration (approximately 10 hours) to the annual allowable downtime under the five nines standard",
        parent=reliability_node,
        critical=False
    )
    duration_url_present = evaluator.add_custom_node(
        result=_has_urls(_combine_urls(data.urls_duration, data.urls_outage)),
        id="Duration_Comparison_URL_Provided",
        desc="At least one URL is provided for outage duration information",
        parent=compare_group,
        critical=True
    )
    duration_leaf = evaluator.add_leaf(
        id="Outage_Duration_Stated",
        desc="State the duration of the January 14, 2026 outage in hours or minutes",
        parent=compare_group,
        critical=False
    )
    duration_claim = (
        f"Public reporting states that {carrier}'s January 14, 2026 outage lasted approximately {data.duration or '10 hours (≈600 minutes)'}."
    )
    await evaluator.verify(
        claim=duration_claim,
        node=duration_leaf,
        sources=_combine_urls(data.urls_duration, data.urls_outage),
        additional_instruction="Verify the stated duration from the sources."
    )
    exceeds_leaf = evaluator.add_leaf(
        id="Exceeds_Annual_Allowance",
        desc="Confirm that a single 10-hour outage far exceeds the 5.26-minute annual allowance",
        parent=compare_group,
        critical=False
    )
    exceeds_claim = (
        "Given that five nines allows about 5.26 minutes of downtime per year, a single outage of roughly 10 hours "
        "(~600 minutes) far exceeds that annual allowance."
    )
    await evaluator.verify(
        claim=exceeds_claim,
        node=exceeds_leaf,
        sources=_combine_urls(data.urls_reliability_standard, data.urls_duration, data.urls_outage),
        additional_instruction="Use the standard definition and reported duration to confirm the comparison."
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
    model: str = "o4-mini",
) -> Dict:
    """
    Evaluate an answer for: Verizon January 14, 2026 outage regulatory compliance.
    """
    # Initialize evaluator (root is parallel, non-critical to allow partial credit across categories)
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

    # Build high-level root node mirroring the overview rubric (non-critical to allow partial scoring)
    overview_root = evaluator.add_parallel(
        id="Verizon_January_2026_Outage_Regulatory_Compliance",
        desc="Verify that the telecommunications carrier's handling of the January 14, 2026 outage complied with FCC regulations and industry context",
        parent=root,
        critical=False
    )

    # Extract structured information from the answer
    data = await evaluator.extract(
        prompt=prompt_extract_outage_compliance(),
        template_class=OutageComplianceExtraction,
        extraction_name="outage_compliance_extraction",
    )

    # Subtree verifications
    await verify_fcc_reporting_thresholds(evaluator, overview_root, data)
    await verify_fcc_timeline_compliance(evaluator, overview_root, data)
    await verify_psap_notification_compliance(evaluator, overview_root, data)
    await verify_root_cause_documentation(evaluator, overview_root, data)
    await verify_customer_remediation(evaluator, overview_root, data)
    await verify_reliability_context(evaluator, overview_root, data)

    # Return structured evaluation summary
    return evaluator.get_summary()