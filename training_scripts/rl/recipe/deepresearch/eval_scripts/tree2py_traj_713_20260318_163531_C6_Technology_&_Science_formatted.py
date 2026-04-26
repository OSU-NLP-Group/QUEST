import asyncio
import logging
from typing import Optional, List, Dict, Any
from urllib.parse import urlparse

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "fcc_outage_compliance_20260114"
TASK_DESCRIPTION = """
A major U.S. wireless telecommunications carrier experienced a network outage on January 14, 2026, that began at approximately 12:00 PM Eastern Time. The outage was caused by a software issue (not a hardware failure) and lasted for more than 10 hours, with service restoration occurring late that evening. At its peak, outage tracking services recorded over 168,000 customer reports. The outage affected both voice and data services across multiple major cities, preventing customers from making calls or accessing mobile data. Many customers' phones displayed "SOS" or "emergency calls only" status.

For this specific outage incident, identify and document all applicable FCC regulatory compliance requirements that the carrier must satisfy, including:
1. Whether the outage meets the threshold criteria for mandatory reporting under the FCC's Network Outage Reporting System (NORS)
2. The specific deadlines for submitting the initial outage report and final outage report to the FCC
3. The requirements for PSAP (Public Safety Answering Point) notifications if 911 services were potentially impacted
4. The requirements and timeline for customer notifications
5. The reference URLs from official FCC sources documenting each of these requirements

Your answer must identify all applicable regulatory requirements, calculate the specific compliance deadlines based on the outage discovery time, and provide official FCC documentation URLs for each requirement category.
"""

# Canonical discovery time and derived deadlines (Eastern Time)
DISCOVERY_ET = "January 14, 2026, 12:00 PM ET"
INITIAL_REPORT_DEADLINE_ET = "January 17, 2026, 12:00 PM ET"  # 72 hours (3 calendar days) after discovery
FINAL_REPORT_DEADLINE_DATE = "February 13, 2026"              # 30 days after discovery (date reference)
FINAL_REPORT_DEADLINE_ET = "February 13, 2026, 12:00 PM ET"   # 30 days after discovery (with time)
PSAP_INITIAL_DEADLINE_ET = "January 14, 2026, 12:30 PM ET"    # 30 minutes after discovery
CUSTOMER_MAX_DELAY_DEADLINE_ET = "February 13, 2026, 12:00 PM ET"  # If 30-day authorized delay is used


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class NORSSection(BaseModel):
    # Threshold determination
    threshold_criteria_text: Optional[str] = None
    threshold_determination: Optional[str] = None
    threshold_basis_text: Optional[str] = None
    threshold_reference_urls: List[str] = Field(default_factory=list)

    # Initial report
    initial_rule_text: Optional[str] = None
    initial_deadline_text: Optional[str] = None
    initial_reference_urls: List[str] = Field(default_factory=list)

    # Final report
    final_rule_text: Optional[str] = None
    final_deadline_text: Optional[str] = None
    final_reference_urls: List[str] = Field(default_factory=list)


class PSAPSection(BaseModel):
    applicability_text: Optional[str] = None
    initial_rule_text: Optional[str] = None
    initial_deadline_text: Optional[str] = None
    followup_rule_text: Optional[str] = None
    reference_urls: List[str] = Field(default_factory=list)


class CustomerNotificationSection(BaseModel):
    order_text: Optional[str] = None  # e.g., "notify FCC and federal law enforcement before customers"
    no_unreasonable_delay_text: Optional[str] = None
    delay_authorization_text: Optional[str] = None
    max_delay_deadline_text: Optional[str] = None
    reference_urls: List[str] = Field(default_factory=list)


class FCCComplianceExtraction(BaseModel):
    nors: Optional[NORSSection] = None
    psap: Optional[PSAPSection] = None
    customer: Optional[CustomerNotificationSection] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_fcc_compliance() -> str:
    return """
    Extract the FCC compliance content that the answer provides for the described January 14, 2026 outage.
    The outage discovery time is exactly: "January 14, 2026, 12:00 PM ET".
    Return a JSON object with this exact nested structure and fields. Use exact phrases the answer uses when possible.

    {
      "nors": {
        "threshold_criteria_text": string or null,                 // How the answer states the NORS threshold (e.g., ">=30 minutes AND (>=90,000 blocked calls OR >=667 OC3-minutes)"; cite 47 CFR § 4.9 if mentioned)
        "threshold_determination": string or null,                 // The clear determination (e.g., "meets", "does not meet", or "cannot determine/pending FCC metrics")
        "threshold_basis_text": string or null,                    // The basis/explanation used (e.g., mentions of "blocked calls", "OC3-minutes", or a statement that those metrics are not provided)
        "threshold_reference_urls": [urls...],                     // Official FCC/CFR documentation URLs the answer cites for the threshold rule

        "initial_rule_text": string or null,                       // How the answer states the initial NORS report deadline rule (e.g., "within 72 hours (3 calendar days) of discovery")
        "initial_deadline_text": string or null,                   // The answer's computed deadline for the initial report (from Jan 14, 2026 12:00 PM ET)
        "initial_reference_urls": [urls...],                       // Official URLs cited for the initial report rule

        "final_rule_text": string or null,                         // How the answer states the final NORS report deadline rule (e.g., "no later than 30 days after discovery")
        "final_deadline_text": string or null,                     // The answer's computed deadline/date for the final report
        "final_reference_urls": [urls...]                          // Official URLs cited for the final report rule
      },
      "psap": {
        "applicability_text": string or null,                      // Text that addresses conditional applicability (e.g., "if 911 potentially impacted")
        "initial_rule_text": string or null,                       // Rule for initial PSAP notification (must mention 30 minutes and effective date Apr 15, 2025 if provided)
        "initial_deadline_text": string or null,                   // The computed initial PSAP notification time (30 minutes after discovery)
        "followup_rule_text": string or null,                      // Rule for follow-up notifications (e.g., every 2 hours until restored)
        "reference_urls": [urls...]                                // Official URLs cited for PSAP rules
      },
      "customer": {
        "order_text": string or null,                              // The before/after order: notify FCC & federal law enforcement before customers
        "no_unreasonable_delay_text": string or null,              // States "without unreasonable delay" after the federal notifications
        "delay_authorization_text": string or null,                // States federal agencies may authorize delay up to 30 days
        "max_delay_deadline_text": string or null,                 // If a 30-day delay window is discussed, the computed latest customer notification time from discovery
        "reference_urls": [urls...]                                // Official URLs cited for customer notification rules
      }
    }

    Extraction rules:
    - Extract only what is explicitly stated in the answer.
    - For each URL field, extract full URLs; include only valid URLs.
    - If any item is missing in the answer, set that field to null or an empty array as appropriate.
    - Do not infer new deadlines; capture the exact computed deadlines the answer gives.
    - Keep the original wording (lightly normalized is OK) for all *_text fields.
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
OFFICIAL_DOMAIN_SUFFIXES = [
    "fcc.gov",      # includes docs.fcc.gov and other subdomains
    "ecfr.gov",
    "govinfo.gov"
]


def is_official_url(url: str) -> bool:
    try:
        host = urlparse(url).netloc.lower()
        if not host:
            return False
        return any(host == suf or host.endswith("." + suf) for suf in OFFICIAL_DOMAIN_SUFFIXES)
    except Exception:
        return False


def any_official_url(urls: Optional[List[str]]) -> bool:
    if not urls:
        return False
    return any(is_official_url(u) for u in urls if isinstance(u, str))


# --------------------------------------------------------------------------- #
# Verification subtrees                                                       #
# --------------------------------------------------------------------------- #
async def verify_nors_threshold(evaluator: Evaluator, parent, nors: Optional[NORSSection]) -> None:
    node = evaluator.add_parallel(
        id="NORS_Threshold_Determination",
        desc="Whether the outage meets the FCC threshold criteria for mandatory NORS reporting, per the provided threshold rule.",
        parent=parent,
        critical=True
    )

    # 1) Threshold_Criteria_Stated_Correctly (simple verify against the answer content)
    crit_leaf = evaluator.add_leaf(
        id="Threshold_Criteria_Stated_Correctly",
        desc="States the threshold criteria as given in constraints: duration ≥ 30 minutes AND (≥ 90,000 blocked calls OR ≥ 667 OC3-minutes), citing 47 CFR § 4.9.",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim=(
            "In the answer, the NORS threshold criteria are explicitly stated as: "
            "an outage lasting at least 30 minutes AND (at least 90,000 blocked calls OR at least 667 OC3-minutes), "
            "and the answer cites 47 CFR § 4.9. Allow minor wording variations (e.g., '>=', '≥', 'OC-3 minutes')."
        ),
        node=crit_leaf,
        additional_instruction="Search the provided answer text. Accept small formatting or wording variants; the core numeric criteria must be present together with a citation to '47 CFR § 4.9'."
    )

    # 2) Threshold_Conclusion_Provided
    conc_leaf = evaluator.add_leaf(
        id="Threshold_Conclusion_Provided",
        desc="Provides a clear determination (meets/does not meet/cannot determine pending missing FCC-defined metrics) consistent with the stated threshold criteria and available incident facts.",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim=(
            "The answer clearly states a determination about whether the outage meets the NORS threshold, using one of: "
            "'meets', 'does not meet', or 'cannot determine/pending FCC-defined metrics'."
        ),
        node=conc_leaf,
        additional_instruction="Look for an explicit determination phrase. It must be unambiguous."
    )

    # 3) Threshold_Uses_FCC_Defined_Metrics
    metrics_leaf = evaluator.add_leaf(
        id="Threshold_Uses_FCC_Defined_Metrics",
        desc="Bases the determination on FCC-defined metrics (blocked calls and/or OC3-minutes) or explicitly states that those metrics are not provided and therefore the threshold cannot be conclusively determined; does not substitute unrelated proxies as if they were the FCC metric.",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim=(
            "The basis for the threshold determination in the answer explicitly references FCC-defined metrics "
            "such as 'blocked calls' and/or 'OC3-minutes', OR explicitly states those metrics are not provided and "
            "therefore a conclusive determination cannot be made. The answer does not treat unrelated proxies (e.g., Downdetector counts) "
            "as if they were the FCC-defined metrics."
        ),
        node=metrics_leaf,
        additional_instruction="Read the determination rationale. Accept clear statements that metrics are unavailable as a valid basis for 'cannot determine'. Reject using customer complaints or outage-tracker counts as substitutes."
    )

    # 4) Threshold_Reference_URL_Official
    threshold_urls = (nors.threshold_reference_urls if nors else []) if nors is not None else []
    url_ok = any_official_url(threshold_urls)
    evaluator.add_custom_node(
        result=url_ok,
        id="Threshold_Reference_URL_Official",
        desc="Provides an official documentation URL for the threshold rule from either an official CFR source (e.g., ecfr.gov, govinfo.gov) or FCC.gov, per constraints.",
        parent=node,
        critical=True
    )


async def verify_nors_initial(evaluator: Evaluator, parent, nors: Optional[NORSSection]) -> None:
    node = evaluator.add_parallel(
        id="NORS_Initial_Report_Requirements",
        desc="Initial NORS reporting deadline rule and its computed deadline based on the outage discovery time.",
        parent=parent,
        critical=True
    )

    # 1) Initial_Report_Deadline_Rule_Identified
    rule_leaf = evaluator.add_leaf(
        id="Initial_Report_Deadline_Rule_Identified",
        desc="Identifies the initial report deadline rule: within 72 hours (3 calendar days) of discovering the outage.",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim=(
            "The answer states that the initial NORS report must be submitted within 72 hours (3 calendar days) of discovering the outage."
        ),
        node=rule_leaf,
        additional_instruction="Look for either 'within 72 hours' or 'within 3 calendar days of discovery'."
    )

    # 2) Initial_Report_Deadline_Calculated
    calc_leaf = evaluator.add_leaf(
        id="Initial_Report_Deadline_Calculated",
        desc=f"Correctly calculates the initial report deadline from January 14, 2026, 12:00 PM ET to {INITIAL_REPORT_DEADLINE_ET}.",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim=(
            f"The answer correctly computes the initial NORS report deadline from the discovery time ({DISCOVERY_ET}) "
            f"as {INITIAL_REPORT_DEADLINE_ET} (72 hours later)."
        ),
        node=calc_leaf,
        additional_instruction="Allow minor date/time formatting variations and 'ET/EST/Eastern Time' variants; the date and (approximate) time must match."
    )

    # 3) Initial_Report_Reference_URL_Official
    initial_urls = (nors.initial_reference_urls if nors else []) if nors is not None else []
    url_ok = any_official_url(initial_urls)
    evaluator.add_custom_node(
        result=url_ok,
        id="Initial_Report_Reference_URL_Official",
        desc="Provides an official documentation URL supporting the initial report deadline rule from either an official CFR source (e.g., ecfr.gov, govinfo.gov) or FCC.gov, per constraints.",
        parent=node,
        critical=True
    )


async def verify_nors_final(evaluator: Evaluator, parent, nors: Optional[NORSSection]) -> None:
    node = evaluator.add_parallel(
        id="NORS_Final_Report_Requirements",
        desc="Final NORS reporting deadline rule and its computed deadline based on the outage discovery time.",
        parent=parent,
        critical=True
    )

    # 1) Final_Report_Deadline_Rule_Identified
    rule_leaf = evaluator.add_leaf(
        id="Final_Report_Deadline_Rule_Identified",
        desc="Identifies the final report deadline rule: no later than 30 days after discovering the outage.",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="The answer states that the final NORS report is due no later than 30 days after discovering the outage.",
        node=rule_leaf,
        additional_instruction="Accept wording like 'within 30 days' or 'no later than 30 days after discovery'."
    )

    # 2) Final_Report_Deadline_Calculated
    calc_leaf = evaluator.add_leaf(
        id="Final_Report_Deadline_Calculated",
        desc=f"Correctly calculates the final report deadline as {FINAL_REPORT_DEADLINE_DATE} (30 days after January 14, 2026).",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim=(
            f"The answer correctly computes the final NORS report deadline as {FINAL_REPORT_DEADLINE_DATE}, "
            f"which is 30 days after {DISCOVERY_ET.split(',')[0] + ','} 2026."
        ),
        node=calc_leaf,
        additional_instruction="Allow minor date formatting variations; the date must be February 13, 2026. Time is optional."
    )

    # 3) Final_Report_Reference_URL_Official
    final_urls = (nors.final_reference_urls if nors else []) if nors is not None else []
    url_ok = any_official_url(final_urls)
    evaluator.add_custom_node(
        result=url_ok,
        id="Final_Report_Reference_URL_Official",
        desc="Provides an official documentation URL supporting the final report deadline rule from either an official CFR source (e.g., ecfr.gov, govinfo.gov) or FCC.gov, per constraints.",
        parent=node,
        critical=True
    )


async def verify_psap(evaluator: Evaluator, parent, psap: Optional[PSAPSection]) -> None:
    node = evaluator.add_parallel(
        id="PSAP_Notification_Requirements",
        desc="PSAP notification requirements and timeline if 911 services were potentially impacted, including follow-ups until restoration.",
        parent=parent,
        critical=True
    )

    # 1) PSAP_Applicability_Addressed
    app_leaf = evaluator.add_leaf(
        id="PSAP_Applicability_Addressed",
        desc="Explicitly addresses applicability conditioned on whether 911 services were potentially impacted (as required by the question/constraints).",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim=(
            "The answer explicitly conditions PSAP notification obligations on whether 911 service was potentially impacted "
            "(e.g., uses phrasing like 'if 911 is potentially affected')."
        ),
        node=app_leaf,
        additional_instruction="Confirm that the answer treats PSAP notification as conditional on potential 911 impact, not unconditional."
    )

    # 2) PSAP_Initial_Notification_Rule_Identified
    init_rule_leaf = evaluator.add_leaf(
        id="PSAP_Initial_Notification_Rule_Identified",
        desc="Identifies the rule: notify affected PSAPs within 30 minutes of discovering the potential 911 impact; includes the effective date (April 15, 2025) as provided in constraints.",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim=(
            "The answer states that affected PSAPs must be notified within 30 minutes of discovering the potential 911 impact "
            "and includes the effective date of this requirement as April 15, 2025."
        ),
        node=init_rule_leaf,
        additional_instruction="Both pieces must appear: 'within 30 minutes' and the effective date 'April 15, 2025'. Allow minor date formatting."
    )

    # 3) PSAP_Initial_Notification_Deadline_Calculated
    init_calc_leaf = evaluator.add_leaf(
        id="PSAP_Initial_Notification_Deadline_Calculated",
        desc=f"Calculates the 30-minute deadline from the discovery time as {PSAP_INITIAL_DEADLINE_ET} (if applicable).",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim=(
            f"The answer correctly computes the initial PSAP notification deadline as {PSAP_INITIAL_DEADLINE_ET}, "
            f"which is 30 minutes after {DISCOVERY_ET}."
        ),
        node=init_calc_leaf,
        additional_instruction="Allow small formatting differences; ensure 12:30 PM ET on Jan 14, 2026 is stated if applicability is asserted."
    )

    # 4) PSAP_Followup_Rule_Identified
    follow_leaf = evaluator.add_leaf(
        id="PSAP_Followup_Rule_Identified",
        desc="Identifies the follow-up requirement: provide follow-up notifications to PSAPs every 2 hours until service is restored (if applicable).",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="The answer states that follow-up notifications to PSAPs must be provided every 2 hours until service is restored.",
        node=follow_leaf,
        additional_instruction="Look for 'every 2 hours' (or 'two hours') and 'until service is restored' language."
    )

    # 5) PSAP_Reference_URL_Official
    psap_urls = (psap.reference_urls if psap else []) if psap is not None else []
    url_ok = any_official_url(psap_urls)
    evaluator.add_custom_node(
        result=url_ok,
        id="PSAP_Reference_URL_Official",
        desc="Provides official documentation URL(s) supporting PSAP notification requirements from either an official CFR source (e.g., ecfr.gov, govinfo.gov) or FCC.gov, per constraints.",
        parent=node,
        critical=True
    )


async def verify_customer(evaluator: Evaluator, parent, cust: Optional[CustomerNotificationSection]) -> None:
    node = evaluator.add_parallel(
        id="Customer_Notification_Requirements",
        desc="Customer notification requirements and timeline/order, including any permissible delay, supported by official FCC/CFR documentation URLs.",
        parent=parent,
        critical=True
    )

    # 1) Federal_Agencies_Before_Customers
    order_leaf = evaluator.add_leaf(
        id="Federal_Agencies_Before_Customers",
        desc="States the requirement: notify the FCC and federal law enforcement before notifying customers.",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="The answer states that the FCC and federal law enforcement must be notified before notifying customers.",
        node=order_leaf,
        additional_instruction="Look for mention of notifying 'FCC' and federal law enforcement (e.g., FBI/USSS) prior to customer notification."
    )

    # 2) Customer_Notification_No_Unreasonable_Delay
    delay_leaf = evaluator.add_leaf(
        id="Customer_Notification_No_Unreasonable_Delay",
        desc="States the requirement: notify customers without unreasonable delay after notifying federal agencies.",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="The answer states that customer notifications must be provided without unreasonable delay after notifying federal agencies.",
        node=delay_leaf,
        additional_instruction="Accept common phrasings conveying 'without unreasonable delay'."
    )

    # 3) Delay_Authorization_Up_To_30_Days
    auth_leaf = evaluator.add_leaf(
        id="Delay_Authorization_Up_To_30_Days",
        desc="States the allowance: federal agencies may authorize delaying customer notification for up to 30 days if necessary.",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="The answer states that federal agencies may authorize delaying customer notifications for up to 30 days.",
        node=auth_leaf,
        additional_instruction="Look for explicit 'up to 30 days' authorization conditioned on federal agencies."
    )

    # 4) Max_Delay_Deadline_Calculated_If_Used
    max_deadline_leaf = evaluator.add_leaf(
        id="Max_Delay_Deadline_Calculated_If_Used",
        desc=f"If the answer discusses the 30-day authorized delay as a time limit, it calculates the latest customer-notification time from the discovery time: {CUSTOMER_MAX_DELAY_DEADLINE_ET}, explicitly conditioned on federal authorization.",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim=(
            f"If a 30-day authorized delay is discussed, the answer also computes the latest permissible customer-notification time "
            f"as {CUSTOMER_MAX_DELAY_DEADLINE_ET}, explicitly conditioned on federal authorization."
        ),
        node=max_deadline_leaf,
        additional_instruction="If the answer does not discuss using the 30-day authorization, this item may be omitted in the answer; otherwise, the computed latest time must be correct. Accept minor formatting variations."
    )

    # 5) Customer_Notification_Reference_URL_Official
    cust_urls = (cust.reference_urls if cust else []) if cust is not None else []
    url_ok = any_official_url(cust_urls)
    evaluator.add_custom_node(
        result=url_ok,
        id="Customer_Notification_Reference_URL_Official",
        desc="Provides official documentation URL(s) supporting the customer notification requirements from either an official CFR source (e.g., ecfr.gov, govinfo.gov) or FCC.gov, per constraints.",
        parent=node,
        critical=True
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
    Evaluate an answer for FCC outage compliance requirements (2026-01-14 incident).
    """
    # Initialize evaluator with a critical parallel root
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
        default_model=model
    )
    # Make the root critical as per rubric; ensure all children under it are critical
    root.critical = True

    # Extract structured compliance info
    extraction: FCCComplianceExtraction = await evaluator.extract(
        prompt=prompt_extract_fcc_compliance(),
        template_class=FCCComplianceExtraction,
        extraction_name="fcc_compliance_extraction"
    )

    # Add ground truth info for deadlines to the summary
    evaluator.add_ground_truth({
        "discovery_time_et": DISCOVERY_ET,
        "expected_initial_report_deadline_et": INITIAL_REPORT_DEADLINE_ET,
        "expected_final_report_deadline_date": FINAL_REPORT_DEADLINE_DATE,
        "expected_final_report_deadline_et": FINAL_REPORT_DEADLINE_ET,
        "expected_psap_initial_deadline_et": PSAP_INITIAL_DEADLINE_ET,
        "expected_customer_max_delay_deadline_et": CUSTOMER_MAX_DELAY_DEADLINE_ET
    }, gt_type="expected_deadlines")

    # Build FCC_Compliance_Requirements_Identification (critical parallel)
    main_node = evaluator.add_parallel(
        id="FCC_Compliance_Requirements_Identification",
        desc="Evaluate whether the answer identifies applicable FCC compliance requirements for the described outage, calculates deadlines from the stated discovery time, and provides official FCC/CFR documentation URLs per the constraints.",
        parent=root,
        critical=True
    )

    # Subtrees
    await verify_nors_threshold(evaluator, main_node, extraction.nors if extraction else None)
    await verify_nors_initial(evaluator, main_node, extraction.nors if extraction else None)
    await verify_nors_final(evaluator, main_node, extraction.nors if extraction else None)
    await verify_psap(evaluator, main_node, extraction.psap if extraction else None)
    await verify_customer(evaluator, main_node, extraction.customer if extraction else None)

    # Return structured summary
    return evaluator.get_summary()