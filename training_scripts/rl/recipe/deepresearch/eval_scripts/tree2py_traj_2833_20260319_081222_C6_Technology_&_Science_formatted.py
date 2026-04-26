import asyncio
import logging
from typing import Any, List, Optional, Dict

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "telecom_outage_reg_analysis"
TASK_DESCRIPTION = """In January 2026, a major U.S. telecommunications carrier experienced a significant wireless network outage that lasted more than 10 hours and affected millions of customers. The outage was caused by a software issue, and many customers reported their phones were in "SOS mode." The carrier subsequently provided account credits to affected customers, and the FCC opened an investigation into the incident.

Research and provide comprehensive documentation addressing the following requirements:

1. Outage Event Identification: Identify the specific telecommunications carrier, the exact date the outage occurred, the root cause as stated by the carrier, the duration from onset to resolution, the specific time when service was restored, and an official URL source confirming these details.

2. FCC NORS Reporting Timeline: Document all applicable FCC Network Outage Reporting System (NORS) requirements for this outage, including:
   - Confirmation that the outage meets FCC reportability thresholds (minimum duration and service type)
   - The required timeframe for submitting the initial NORS notification for wireless providers
   - The required timeframe for submitting the initial outage report
   - The required timeframe for submitting the final outage report
   - Official URL sources for each of these FCC requirements

3. Business SLA Credit Analysis: For the carrier's business Internet Dedicated service customers, document:
   - The network availability percentage guaranteed in the SLA
   - The required timeframe for customers to open trouble tickets after learning of an outage to claim Network Unavailability credits
   - The formula and components used to calculate credits for Network Unavailability (specify what charges are included)
   - The Time to Repair (TTR) Service Level Standard target timeframe
   - Official URL source for the carrier's business SLA terms

4. Consumer Compensation Documentation: Document the consumer compensation procedures, including:
   - The specific dollar amount of account credits provided to affected consumer customers
   - The method by which consumer customers can redeem their credits
   - The different process used for small business customers
   - Official URL source for the carrier's compensation announcement

5. FCC Investigation Participation Process: Document the FCC investigation procedures, including:
   - The specific FCC docket number opened for this investigation
   - The date the FCC public notice was released
   - The deadline date for submitting public comments
   - The methods available for submitting comments (including electronic, paper, and email options)
   - The specific email address (if any) for submitting experience descriptions
   - Official URL source for the FCC public notice

All factual claims must be supported by official URL references from the telecommunications carrier, FCC, or other authoritative sources. Ensure that all dates, timeframes, dollar amounts, and procedural requirements are accurately documented based on publicly available official information.
"""


# --------------------------------------------------------------------------- #
# Data models for structured extraction                                       #
# --------------------------------------------------------------------------- #
class OutageExtraction(BaseModel):
    carrier: Optional[str] = None
    outage_date: Optional[str] = None  # e.g., "January 28, 2026"
    root_cause: Optional[str] = None   # e.g., "software issue"
    duration_text: Optional[str] = None  # e.g., "more than 10 hours", "approximately 12 hours"
    restoration_time: Optional[str] = None  # e.g., "10:30 PM ET", "around 11 p.m. ET"
    affected_millions_statement: Optional[str] = None  # textual evidence statement from answer
    sos_mode_statement: Optional[str] = None  # textual evidence statement from answer
    outage_sources: List[str] = Field(default_factory=list)  # official/authoritative URLs cited in answer


class NORSExtraction(BaseModel):
    thresholds_statement: Optional[str] = None  # statement about 30 minutes & wireless applicability
    thresholds_sources: List[str] = Field(default_factory=list)

    initial_notification_timeframe: Optional[str] = None  # e.g., "within 120 minutes after determining reportable"
    initial_notification_sources: List[str] = Field(default_factory=list)

    initial_outage_report_timeframe: Optional[str] = None  # e.g., "within 3 calendar days"
    initial_outage_report_sources: List[str] = Field(default_factory=list)

    final_outage_report_timeframe: Optional[str] = None  # e.g., "within 30 days"
    final_outage_report_sources: List[str] = Field(default_factory=list)


class BusinessSLAExtraction(BaseModel):
    sla_carrier: Optional[str] = None  # expected to match the outage carrier
    service_name: Optional[str] = None  # e.g., "Internet Dedicated"
    network_availability_pct: Optional[str] = None  # e.g., "100%"
    trouble_ticket_window: Optional[str] = None  # e.g., "within 4 hours of learning of the outage"
    credit_formula: Optional[str] = None  # clear textual formula
    ttr_target: Optional[str] = None  # e.g., "4 hours"
    sla_sources: List[str] = Field(default_factory=list)


class ConsumerCompensationExtraction(BaseModel):
    credit_amount: Optional[str] = None  # e.g., "$20"
    redemption_method: Optional[str] = None  # e.g., "carrier mobile app"
    small_business_process: Optional[str] = None  # e.g., "contacted directly"
    compensation_sources: List[str] = Field(default_factory=list)


class FCCInvestigationExtraction(BaseModel):
    investigation_opened_statement: Optional[str] = None
    public_notice_release_date: Optional[str] = None  # e.g., "January 28, 2026"
    comment_deadline: Optional[str] = None  # e.g., "March 16, 2026"
    submission_methods: List[str] = Field(default_factory=list)  # e.g., ["ECFS", "paper", "email"]
    experience_email: Optional[str] = None  # if specified; else null
    docket_number: Optional[str] = None
    public_notice_sources: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_outage() -> str:
    return """
    Extract the outage event details explicitly stated in the answer. Return:
    - carrier: The telecommunications carrier's name.
    - outage_date: The exact calendar date of the outage as stated (e.g., "January 28, 2026"). If range or multi-day is given, provide the specific date the outage occurred.
    - root_cause: The root cause as the carrier stated it (e.g., "software issue").
    - duration_text: The total duration from onset to resolution as text (e.g., "more than 10 hours", "about 12 hours").
    - restoration_time: The specific time when service was restored (e.g., "10:30 PM ET"). If multiple time zones are shown, keep the text as given.
    - affected_millions_statement: A short phrase from the answer indicating that the outage affected millions of customers (or null if not present).
    - sos_mode_statement: A short phrase from the answer indicating that many customers reported "SOS mode" (or null if not present).
    - outage_sources: All official/authoritative URLs that the answer cites for any of the above (e.g., carrier newsroom post, status page, FCC page, government page). Include only valid URLs that appear in the answer.
    """


def prompt_extract_nors() -> str:
    return """
    Extract FCC NORS requirements as stated in the answer. Return:
    - thresholds_statement: The statement that outages lasting at least 30 minutes are reportable and that wireless providers are covered by NORS (verbatim or concise paraphrase from the answer).
    - thresholds_sources: The official FCC URL(s) provided in the answer supporting the thresholds/applicability.
    - initial_notification_timeframe: The required timeframe for submitting the initial NORS notification for wireless providers (e.g., "within 120 minutes after determining reportable").
    - initial_notification_sources: The official FCC URL(s) supporting the initial notification timeframe.
    - initial_outage_report_timeframe: The required timeframe for submitting the initial outage report (e.g., "within 3 calendar days of discovering the outage").
    - initial_outage_report_sources: The official FCC URL(s) supporting the initial report timeframe.
    - final_outage_report_timeframe: The required timeframe for submitting the final outage report (e.g., "within 30 days of discovering the outage").
    - final_outage_report_sources: The official FCC URL(s) supporting the final report timeframe.
    Only include URLs that appear explicitly in the answer.
    """


def prompt_extract_sla() -> str:
    return """
    Extract the business Internet Dedicated SLA details as provided in the answer. Return:
    - sla_carrier: The carrier name for the SLA (should match the outage carrier if the answer claims so).
    - service_name: The service name (e.g., "Internet Dedicated").
    - network_availability_pct: The guaranteed network availability percentage stated in the SLA (e.g., "100%").
    - trouble_ticket_window: The timeframe customers must open tickets after learning of an outage to claim Network Unavailability credits (e.g., "within 4 hours of learning of the outage").
    - credit_formula: The precise formula and included charges for Network Unavailability credits (e.g., "one pro-rated day (1/30th) of monthly recurring charges plus access/line charges per cumulative hour or fraction of unavailability").
    - ttr_target: The Time to Repair (TTR) Service Level Standard target timeframe (e.g., "4 hours").
    - sla_sources: Official URL(s) for the carrier's business SLA terms cited in the answer. Include only URLs that appear explicitly in the answer.
    """


def prompt_extract_consumer_comp() -> str:
    return """
    Extract consumer compensation details as provided in the answer. Return:
    - credit_amount: The specific dollar amount of account credits (e.g., "$20").
    - redemption_method: How consumers redeem the credits (e.g., "via the carrier mobile app").
    - small_business_process: The process for small business customers (e.g., "contacted directly"), if described.
    - compensation_sources: The official carrier URL(s) cited in the answer supporting these compensation details. Include only URLs that appear explicitly in the answer.
    """


def prompt_extract_fcc_investigation() -> str:
    return """
    Extract FCC investigation public notice details and participation procedures as provided in the answer. Return:
    - investigation_opened_statement: A brief statement that FCC opened an investigation (from the answer).
    - public_notice_release_date: The date the FCC public notice was released (e.g., "January 28, 2026").
    - comment_deadline: The deadline date for submitting public comments (e.g., "March 16, 2026").
    - submission_methods: A list of submission methods mentioned (e.g., ["ECFS", "paper", "email"]).
    - experience_email: The specific email address for submitting experience descriptions if the public notice provides one; otherwise null.
    - docket_number: The FCC docket number for this investigation.
    - public_notice_sources: Official FCC public notice URL(s) cited in the answer. Include only URLs that appear explicitly in the answer.
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _nz_list(values: Optional[List[str]]) -> List[str]:
    return [v for v in (values or []) if isinstance(v, str) and v.strip()]


# --------------------------------------------------------------------------- #
# Section verification builders                                               #
# --------------------------------------------------------------------------- #
async def build_outage_verification(
    evaluator: Evaluator,
    parent,
    outage: OutageExtraction,
) -> None:
    node = evaluator.add_parallel(
        id="Outage_Event_Identification",
        desc="Identify and document the specific outage event described in the prompt/constraints.",
        parent=parent,
        critical=True,
    )
    srcs = _nz_list(outage.outage_sources)

    # 1) Carrier is a major U.S. wireless carrier; event is a wireless network outage
    n1 = evaluator.add_leaf(
        id="Carrier_Is_Major_US_Wireless_Carrier_With_Source",
        desc="Identifies the carrier and supports (via authoritative source) that it is a major U.S. telecommunications carrier and that the event was a wireless network outage.",
        parent=node,
        critical=True,
    )
    claim1 = (
        f"The provided source indicates that {outage.carrier or 'the carrier'} is a major U.S. wireless "
        f"telecommunications carrier and confirms that the event was a wireless network outage."
    )
    await evaluator.verify(
        claim=claim1,
        node=n1,
        sources=srcs,
        additional_instruction="Accept evidence that the page is an official carrier page (e.g., verizon.com, att.com, t-mobile.com) or FCC/government site. "
                               "Treat phrases like 'nationwide wireless network' or 'major U.S. carrier' as sufficient for 'major'. "
                               "The source must also indicate the incident was a wireless/mobile service outage."
    )

    # 2) Outage date in January 2026
    n2 = evaluator.add_leaf(
        id="Outage_Date_In_January_2026",
        desc="Provides the exact outage date and it is in January 2026.",
        parent=node,
        critical=True,
    )
    claim2 = f"The outage occurred on {outage.outage_date or 'a date in January 2026'}, which is in January 2026."
    await evaluator.verify(
        claim=claim2,
        node=n2,
        sources=srcs,
        additional_instruction="Verify that the page explicitly states a date in January 2026 for the outage event."
    )

    # 3) Root cause is software issue as stated by the carrier
    n3 = evaluator.add_leaf(
        id="Root_Cause_Is_Software_Issue_As_Stated_By_Carrier",
        desc="States the root cause as a software issue, explicitly attributing the claim to the carrier’s statement.",
        parent=node,
        critical=True,
    )
    claim3 = "The carrier stated that a software issue was the root cause of the outage."
    await evaluator.verify(
        claim=claim3,
        node=n3,
        sources=srcs,
        additional_instruction="The evidence should attribute the cause to the carrier's own statement (press release, status update, newsroom, or official communication)."
    )

    # 4) Duration >= 10 hours
    n4 = evaluator.add_leaf(
        id="Duration_Exceeds_Or_Equals_10_Hours",
        desc="Documents outage duration from onset to resolution and confirms duration is at least 10 hours.",
        parent=node,
        critical=True,
    )
    claim4 = "The outage lasted at least 10 hours from onset to full resolution."
    await evaluator.verify(
        claim=claim4,
        node=n4,
        sources=srcs,
        additional_instruction="Accept phrasing such as 'more than 10 hours', 'over 10 hours', or any stated start/end times that reasonably total 10+ hours."
    )

    # 5) Specific service restoration time provided
    n5 = evaluator.add_leaf(
        id="Service_Restoration_Time_Provided",
        desc="Provides the specific time when service was restored (resolution time).",
        parent=node,
        critical=True,
    )
    claim5 = "The source provides a specific time of day when service was restored (the resolution time)."
    await evaluator.verify(
        claim=claim5,
        node=n5,
        sources=srcs,
        additional_instruction="Look for an explicit clock time (e.g., '10:30 PM ET') associated with service restoration."
    )

    # 6) Affected millions of customers
    n6 = evaluator.add_leaf(
        id="Affected_Millions_Of_Customers",
        desc="Documents that the outage affected millions of customers.",
        parent=node,
        critical=True,
    )
    claim6 = "The outage affected millions of customers."
    await evaluator.verify(
        claim=claim6,
        node=n6,
        sources=srcs,
        additional_instruction="The page should explicitly state 'millions' or provide a customer count that is in the millions."
    )

    # 7) SOS mode reported
    n7 = evaluator.add_leaf(
        id="SOS_Mode_Reported_By_Many_Customers",
        desc="Documents that many customers reported 'SOS mode' during the outage.",
        parent=node,
        critical=True,
    )
    claim7 = "Many customers reported that their phones showed 'SOS mode' during the outage."
    await evaluator.verify(
        claim=claim7,
        node=n7,
        sources=srcs,
        additional_instruction="Look for explicit mention of 'SOS' or 'SOS mode' affecting many users as a symptom during the outage."
    )

    # 8) Official/authoritative URL confirming details
    n8 = evaluator.add_leaf(
        id="Official_URL_Source_For_Outage_Details",
        desc="Provides at least one official/authoritative URL source confirming the outage details (carrier/date/cause/duration/restoration and impact indicators).",
        parent=node,
        critical=True,
    )
    claim8 = "This source is an official or authoritative page (carrier or FCC/government) that confirms key outage details (carrier/date/cause/duration/restoration time or impact)."
    await evaluator.verify(
        claim=claim8,
        node=n8,
        sources=srcs,
        additional_instruction="Official pages include carrier domains (e.g., verizon.com) and FCC/government (.gov). The page must corroborate at least several of the specified details."
    )


async def build_nors_verification(
    evaluator: Evaluator,
    parent,
    nors: NORSExtraction,
) -> None:
    node = evaluator.add_parallel(
        id="FCC_NORS_Reporting_Timeline",
        desc="Documents FCC NORS reportability and required submission timeframes, with official sources, matching the constraint values.",
        parent=parent,
        critical=True,
    )

    # Thresholds: 30 minutes and wireless applicability
    n1 = evaluator.add_leaf(
        id="Reportability_Thresholds_Duration_At_Least_30_Minutes_And_Wireless_With_FCC_Source",
        desc="States that reportable outages must last at least 30 minutes and that the affected service type (wireless) is within NORS applicability, and provides an official FCC URL supporting these thresholds/applicability.",
        parent=node,
        critical=True,
    )
    claim1 = "Under FCC NORS rules, outages lasting at least 30 minutes are reportable and wireless providers are covered by these reporting requirements."
    await evaluator.verify(
        claim=claim1,
        node=n1,
        sources=_nz_list(nors.thresholds_sources),
        additional_instruction="Look for FCC rules or public-facing FCC documentation (e.g., 47 CFR Part 4, FCC guidance pages) confirming both the 30-minute threshold and wireless applicability."
    )

    # Initial notification within 120 minutes (wireless)
    n2 = evaluator.add_leaf(
        id="Initial_NORS_Notification_Within_120_Minutes_With_FCC_Source",
        desc="States that for wireless providers the initial NORS notification must be submitted within 120 minutes after determining an outage is reportable, and provides an official FCC URL source.",
        parent=node,
        critical=True,
    )
    claim2 = "Wireless providers must submit the initial NORS notification within 120 minutes after determining that an outage is reportable."
    await evaluator.verify(
        claim=claim2,
        node=n2,
        sources=_nz_list(nors.initial_notification_sources),
        additional_instruction="Verify with FCC rules/guidance that the 120-minute initial notification requirement applies."
    )

    # Initial report within 3 calendar days
    n3 = evaluator.add_leaf(
        id="Initial_Outage_Report_Within_3_Calendar_Days_With_FCC_Source",
        desc="States that the initial outage report must be submitted within 3 calendar days of discovering the outage, and provides an official FCC URL source.",
        parent=node,
        critical=True,
    )
    claim3 = "The initial outage report must be submitted within 3 calendar days of discovering the outage."
    await evaluator.verify(
        claim=claim3,
        node=n3,
        sources=_nz_list(nors.initial_outage_report_sources),
        additional_instruction="Confirm the 'within 3 calendar days' timing from an official FCC page or rule."
    )

    # Final report within 30 days
    n4 = evaluator.add_leaf(
        id="Final_Outage_Report_Within_30_Days_With_FCC_Source",
        desc="States that the final outage report must be submitted within 30 days of discovering the outage, and provides an official FCC URL source.",
        parent=node,
        critical=True,
    )
    claim4 = "The final outage report must be submitted within 30 days of discovering the outage."
    await evaluator.verify(
        claim=claim4,
        node=n4,
        sources=_nz_list(nors.final_outage_report_sources),
        additional_instruction="Confirm the 'within 30 days' timing from an official FCC page or rule."
    )


async def build_sla_verification(
    evaluator: Evaluator,
    parent,
    sla: BusinessSLAExtraction,
    outage: OutageExtraction,
) -> None:
    node = evaluator.add_parallel(
        id="Business_SLA_Credit_Analysis",
        desc="Documents business Internet Dedicated SLA terms relevant to availability, credits, and repair targets, matching the constraint values and supported by an official SLA URL.",
        parent=parent,
        critical=True,
    )
    sla_srcs = _nz_list(sla.sla_sources)
    carrier_for_instruction = outage.carrier or sla.sla_carrier or "the carrier"

    # Network availability 100% (intended: Verizon Internet Dedicated)
    n1 = evaluator.add_leaf(
        id="SLA_Network_Availability_Is_100_Percent_For_Verizon_Internet_Dedicated",
        desc="States that for Verizon business Internet Dedicated services, the network availability standard is 100%, with an official SLA URL citation.",
        parent=node,
        critical=True,
    )
    claim1 = "The business Internet Dedicated service SLA guarantees 100% network availability."
    await evaluator.verify(
        claim=claim1,
        node=n1,
        sources=sla_srcs,
        additional_instruction=f"Verify on the official SLA terms page for {carrier_for_instruction} business Internet Dedicated (or equivalent dedicated internet) service that the availability standard is 100%."
    )

    # Trouble ticket window within 4 hours of learning
    n2 = evaluator.add_leaf(
        id="Trouble_Ticket_Window_Is_Within_4_Hours_Of_Learning_Of_Outage",
        desc="States that customers must open trouble tickets within 4 hours of learning of the outage to claim Network Unavailability credits, with an official SLA URL citation.",
        parent=node,
        critical=True,
    )
    claim2 = "Customers must open trouble tickets within 4 hours of learning of the outage to be eligible for Network Unavailability credits."
    await evaluator.verify(
        claim=claim2,
        node=n2,
        sources=sla_srcs,
        additional_instruction="Confirm the 4-hour window requirement on the official SLA page."
    )

    # Credit formula matches constraint
    n3 = evaluator.add_leaf(
        id="Network_Unavailability_Credit_Formula_Matches_Constraint",
        desc="States that credits equal one pro-rated day of monthly recurring charges plus access/line charges per cumulative hour (or fraction) of unavailability, and specifies which charges are included, with an official SLA URL citation.",
        parent=node,
        critical=True,
    )
    claim3 = "Network Unavailability credits equal one pro-rated day (e.g., 1/30th) of the monthly recurring charges plus access/line charges per cumulative hour or fraction of unavailability."
    await evaluator.verify(
        claim=claim3,
        node=n3,
        sources=sla_srcs,
        additional_instruction="Accept equivalent phrasing such as '1/30th of MRC per hour or fraction' and explicit inclusion of local access/line charges when stated."
    )

    # TTR target is 4 hours
    n4 = evaluator.add_leaf(
        id="TTR_Target_Is_4_Hours",
        desc="States that the Time to Repair (TTR) Service Level Standard target is 4 hours, with an official SLA URL citation.",
        parent=node,
        critical=True,
    )
    claim4 = "The Time to Repair (TTR) Service Level Standard target for the business Internet Dedicated service is 4 hours."
    await evaluator.verify(
        claim=claim4,
        node=n4,
        sources=sla_srcs,
        additional_instruction="Verify that the SLA or SLS explicitly sets a 4-hour TTR target."
    )

    # Official URL source for the business SLA terms
    n5 = evaluator.add_leaf(
        id="Official_URL_Source_Business_SLA_Terms",
        desc="Provides an official URL source for the carrier's business SLA terms used for the availability/credit/TTR claims above.",
        parent=node,
        critical=True,
    )
    claim5 = "This URL is the official carrier business SLA terms page for the Internet Dedicated service."
    await evaluator.verify(
        claim=claim5,
        node=n5,
        sources=sla_srcs,
        additional_instruction=f"The URL should belong to {carrier_for_instruction}'s official domain and be a terms/SLA/SLS document for the dedicated internet service."
    )


async def build_consumer_comp_verification(
    evaluator: Evaluator,
    parent,
    comp: ConsumerCompensationExtraction,
    outage: OutageExtraction,
) -> None:
    node = evaluator.add_parallel(
        id="Consumer_Compensation_Documentation",
        desc="Documents consumer compensation amount and procedures for the outage, matching the constraint values and supported by an official carrier URL.",
        parent=parent,
        critical=True,
    )
    comp_srcs = _nz_list(comp.compensation_sources)

    # $20 credits
    n1 = evaluator.add_leaf(
        id="Consumer_Account_Credit_Is_20_Dollars",
        desc="States that the carrier provided $20 account credits to affected consumer customers, with an official URL source.",
        parent=node,
        critical=True,
    )
    claim1 = "The carrier provided $20 account credits to affected consumer customers."
    await evaluator.verify(
        claim=claim1,
        node=n1,
        sources=comp_srcs,
        additional_instruction="Confirm the amount is specifically $20 on the official carrier announcement."
    )

    # Redeem via mobile app
    n2 = evaluator.add_leaf(
        id="Consumer_Credits_Redeemable_Via_Carrier_Mobile_App",
        desc="States that consumer credits are redeemable via the carrier’s mobile app, with an official URL source.",
        parent=node,
        critical=True,
    )
    claim2 = "Consumer customers can redeem the outage credits via the carrier's mobile app."
    await evaluator.verify(
        claim=claim2,
        node=n2,
        sources=comp_srcs,
        additional_instruction="Look for explicit instructions indicating redemption through the official mobile app."
    )

    # Small business contacted directly
    n3 = evaluator.add_leaf(
        id="Small_Business_Customers_Contacted_Directly",
        desc="States that small business customers are contacted directly about their credits (a different process than consumers), with an official URL source.",
        parent=node,
        critical=True,
    )
    claim3 = "Small business customers are contacted directly by the carrier regarding credits (a different process than consumers)."
    await evaluator.verify(
        claim=claim3,
        node=n3,
        sources=comp_srcs,
        additional_instruction="Confirm the official announcement distinguishes small business process as 'contacted directly' or similar."
    )

    # Official URL source for compensation announcement
    n4 = evaluator.add_leaf(
        id="Official_URL_Source_Compensation_Announcement",
        desc="Provides an official URL source for the carrier's compensation announcement supporting the compensation details above.",
        parent=node,
        critical=True,
    )
    claim4 = "This URL is the carrier's official announcement detailing the consumer compensation (credits) for the outage."
    await evaluator.verify(
        claim=claim4,
        node=n4,
        sources=comp_srcs,
        additional_instruction="The URL should be on the official carrier domain and clearly be an announcement or customer notice."
    )


async def build_fcc_investigation_verification(
    evaluator: Evaluator,
    parent,
    fcc: FCCInvestigationExtraction,
) -> None:
    node = evaluator.add_parallel(
        id="FCC_Investigation_Participation_Process",
        desc="Documents FCC investigation public notice details and public participation procedures, matching the constraint values where specified, with an official FCC URL source.",
        parent=parent,
        critical=True,
    )
    fcc_srcs = _nz_list(fcc.public_notice_sources)

    # Investigation opened
    n1 = evaluator.add_leaf(
        id="FCC_Investigation_Is_Opened",
        desc="States that the FCC opened an investigation into the outage, supported by an official FCC URL.",
        parent=node,
        critical=True,
    )
    claim1 = "The FCC opened an investigation into the outage."
    await evaluator.verify(
        claim=claim1,
        node=n1,
        sources=fcc_srcs,
        additional_instruction="Verify this is stated on an official FCC page (fcc.gov), preferably a public notice."
    )

    # Public notice release date
    n2 = evaluator.add_leaf(
        id="FCC_Public_Notice_Release_Date_Is_Jan_28_2026",
        desc="States that the FCC public notice was released on January 28, 2026, supported by an official FCC URL.",
        parent=node,
        critical=True,
    )
    claim2 = "The FCC public notice was released on January 28, 2026."
    await evaluator.verify(
        claim=claim2,
        node=n2,
        sources=fcc_srcs,
        additional_instruction="Confirm the release date on the FCC public notice."
    )

    # Comment deadline
    n3 = evaluator.add_leaf(
        id="Public_Comment_Deadline_Is_Mar_16_2026",
        desc="States that the comment deadline is March 16, 2026, supported by an official FCC URL.",
        parent=node,
        critical=True,
    )
    claim3 = "The deadline for submitting public comments is March 16, 2026."
    await evaluator.verify(
        claim=claim3,
        node=n3,
        sources=fcc_srcs,
        additional_instruction="Confirm the stated deadline date on the FCC public notice."
    )

    # Submission methods include ECFS, paper, and email
    n4 = evaluator.add_leaf(
        id="Comment_Submission_Methods_Include_ECFS_Paper_Email",
        desc="Describes methods available for submitting comments and confirms they include ECFS (electronic), paper filing, and email options, supported by an official FCC URL.",
        parent=node,
        critical=True,
    )
    claim4 = "The available methods for submitting comments include ECFS (electronic), paper filing, and email."
    await evaluator.verify(
        claim=claim4,
        node=n4,
        sources=fcc_srcs,
        additional_instruction="Look for explicit instructions on the FCC notice covering ECFS, paper filing, and email submission methods (synonyms acceptable)."
    )

    # Experience description email address (if any)
    n5 = evaluator.add_leaf(
        id="Experience_Description_Email_Address_Provided_If_Specified",
        desc="Provides the specific FCC email address for submitting experience descriptions if one is given in the FCC public notice; otherwise explicitly states that no such email address is specified, supported by an official FCC URL.",
        parent=node,
        critical=True,
    )
    if fcc.experience_email and fcc.experience_email.strip():
        claim5 = f"The FCC public notice provides a specific email address for submitting experience descriptions: {fcc.experience_email.strip()}."
        add_ins5 = "Verify that a specific email address appears on the page for submitting experience descriptions."
    else:
        claim5 = "The FCC public notice does not specify any dedicated email address for submitting experience descriptions."
        add_ins5 = "Confirm that no specific dedicated email address for submitting experience descriptions is provided in the notice."
    await evaluator.verify(
        claim=claim5,
        node=n5,
        sources=fcc_srcs,
        additional_instruction=add_ins5
    )

    # Docket number provided
    n6 = evaluator.add_leaf(
        id="FCC_Docket_Number_Provided",
        desc="States the FCC docket number opened for this investigation, supported by an official FCC URL.",
        parent=node,
        critical=True,
    )
    claim6 = f"The FCC docket number for this investigation is {fcc.docket_number or '[docket number]' }."
    await evaluator.verify(
        claim=claim6,
        node=n6,
        sources=fcc_srcs,
        additional_instruction="Confirm the docket number as shown on the FCC public notice."
    )

    # Official public notice URL
    n7 = evaluator.add_leaf(
        id="Official_URL_Source_FCC_Public_Notice",
        desc="Provides an official URL source for the FCC public notice supporting the docket/date/deadline/submission details above.",
        parent=node,
        critical=True,
    )
    claim7 = "This URL is the official FCC public notice for the investigation."
    await evaluator.verify(
        claim=claim7,
        node=n7,
        sources=fcc_srcs,
        additional_instruction="The URL should be on the fcc.gov domain and be a public notice or equivalent official document."
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
    # Initialize evaluator (root is a non-critical container)
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

    # Extract all sections from the answer
    outage, nors, sla, comp, fcc = await asyncio.gather(
        evaluator.extract(
            prompt=prompt_extract_outage(),
            template_class=OutageExtraction,
            extraction_name="outage_event_extraction"
        ),
        evaluator.extract(
            prompt=prompt_extract_nors(),
            template_class=NORSExtraction,
            extraction_name="fcc_nors_extraction"
        ),
        evaluator.extract(
            prompt=prompt_extract_sla(),
            template_class=BusinessSLAExtraction,
            extraction_name="business_sla_extraction"
        ),
        evaluator.extract(
            prompt=prompt_extract_consumer_comp(),
            template_class=ConsumerCompensationExtraction,
            extraction_name="consumer_compensation_extraction"
        ),
        evaluator.extract(
            prompt=prompt_extract_fcc_investigation(),
            template_class=FCCInvestigationExtraction,
            extraction_name="fcc_investigation_extraction"
        ),
    )

    # Add a top-level critical parallel node to reflect rubric root
    top = evaluator.add_parallel(
        id="Telecommunications_Outage_Regulatory_Analysis",
        desc="Documentation of the January 2026 major U.S. wireless outage, related FCC NORS requirements, business SLA credit terms, consumer compensation, and FCC investigation participation procedures, with authoritative sources.",
        parent=root,
        critical=True,
    )

    # Build and verify each section
    await build_outage_verification(evaluator, top, outage)
    await build_nors_verification(evaluator, top, nors)
    await build_sla_verification(evaluator, top, sla, outage)
    await build_consumer_comp_verification(evaluator, top, comp, outage)
    await build_fcc_investigation_verification(evaluator, top, fcc)

    # Return unified summary
    return evaluator.get_summary()