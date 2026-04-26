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
TASK_ID = "verizon_jan_2026_outage_report"
TASK_DESCRIPTION = (
    "In January 2026, Verizon experienced a significant nationwide network outage that affected millions of customers "
    "and prompted regulatory scrutiny. Research and compile a comprehensive report that documents this incident across "
    "five key dimensions:\n\n"
    "1. Outage Event Documentation: Identify the specific date the outage occurred, how long it lasted, and how many "
    "customers were affected. Include the geographic scope of the disruption.\n"
    "2. Customer Compensation Program: Determine what monetary compensation Verizon provided to affected customers and "
    "how this compensation was distributed.\n"
    "3. FCC Regulatory Response: Find the specific email address the FCC established for collecting customer complaints "
    "and identify the submission deadline.\n"
    "4. Technical Root Cause Analysis: Identify the type of network technology involved and the specific technical failure.\n"
    "5. Comparative Network Reliability Context: Provide industry context with Opensignal metrics and carrier coverage "
    "statistics, and how many metro markets Verizon led in reliability pre‑outage.\n\n"
    "For each piece of information, provide the supporting source URL(s)."
)

# Expected ground-truth targets per rubric
EXPECTED = {
    "outage_date": "January 14, 2026",
    "outage_min_hours": 10,
    "customers_range_text": "between 1.5 and 2 million",
    "geographic_scope_text": "nationwide across the United States",
    "comp_amount_text": "$20 account credit",
    "comp_distribution_method_text": "credits were automatically applied to affected customer accounts",
    "comp_start_date": "January 15, 2026",
    "fcc_email": "VerizonOutage2026@fcc.gov",
    "fcc_deadline": "March 16, 2026",
    "network_tech_text": "5G standalone (SA) network",
    "root_cause_text": "software issue",
    "opensignal_reliability_text": "According to Opensignal's January 2025 report, T-Mobile and Verizon tied for Reliability Experience with 898 points",
    "tmobile_cov_98_text": "T-Mobile has 5G coverage for 98% of Americans",
    "tmobile_largest_5g_text": "T-Mobile claims to have the largest 5G network",
    "tmobile_roadtrip_96_2_text": "T-Mobile's 5G coverage rate is 96.2% based on road-trip testing",
    "verizon_5g_99_text": "Verizon claims 5G coverage to 99% of the U.S. population",
    "verizon_lte_gt99_text": "Verizon claims 4G LTE coverage to more than 99% of the U.S. population",
    "verizon_metro_100_text": "Verizon won reliability awards in 100 metro markets (RootMetrics 1H 2025)"
}


# --------------------------------------------------------------------------- #
# Data models for structured extraction                                       #
# --------------------------------------------------------------------------- #
class OutageEventExtraction(BaseModel):
    event_date: Optional[str] = None
    event_date_sources: List[str] = Field(default_factory=list)
    event_duration: Optional[str] = None
    event_duration_sources: List[str] = Field(default_factory=list)
    customers_affected: Optional[str] = None
    customers_affected_sources: List[str] = Field(default_factory=list)
    geographic_scope: Optional[str] = None
    geographic_scope_sources: List[str] = Field(default_factory=list)


class CustomerCompensationExtraction(BaseModel):
    compensation_amount: Optional[str] = None
    compensation_amount_sources: List[str] = Field(default_factory=list)
    compensation_distribution_method: Optional[str] = None
    compensation_distribution_method_sources: List[str] = Field(default_factory=list)
    compensation_start_date: Optional[str] = None
    compensation_start_date_sources: List[str] = Field(default_factory=list)


class FCCResponseExtraction(BaseModel):
    complaint_email: Optional[str] = None
    complaint_email_sources: List[str] = Field(default_factory=list)
    complaint_deadline: Optional[str] = None
    complaint_deadline_sources: List[str] = Field(default_factory=list)


class TechnicalRootCauseExtraction(BaseModel):
    network_technology: Optional[str] = None
    network_technology_sources: List[str] = Field(default_factory=list)
    root_cause: Optional[str] = None
    root_cause_sources: List[str] = Field(default_factory=list)


class ComparativeContextExtraction(BaseModel):
    opensignal_reliability_desc: Optional[str] = None
    opensignal_reliability_sources: List[str] = Field(default_factory=list)
    tmobile_coverage_98_desc: Optional[str] = None
    tmobile_coverage_98_sources: List[str] = Field(default_factory=list)
    tmobile_largest_5g_desc: Optional[str] = None
    tmobile_largest_5g_sources: List[str] = Field(default_factory=list)
    tmobile_roadtrip_96_2_desc: Optional[str] = None
    tmobile_roadtrip_96_2_sources: List[str] = Field(default_factory=list)
    verizon_5g_99_desc: Optional[str] = None
    verizon_5g_99_sources: List[str] = Field(default_factory=list)
    verizon_lte_gt99_desc: Optional[str] = None
    verizon_lte_gt99_sources: List[str] = Field(default_factory=list)
    verizon_metro_100_desc: Optional[str] = None
    verizon_metro_100_sources: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_outage_event() -> str:
    return """
Extract the outage event documentation details from the answer, exactly as stated.
Return a JSON object with these fields:
- event_date: The stated outage date (string, e.g., "January 14, 2026"), or null if not stated.
- event_date_sources: Array of URL(s) explicitly cited that support the outage date; empty array if none.
- event_duration: The stated outage duration (string exactly as in answer, e.g., "about 12 hours"), or null if not stated.
- event_duration_sources: Array of URL(s) explicitly cited that support the duration; empty array if none.
- customers_affected: The stated affected-customer count description (e.g., "1.8 million", "more than 1.5 million"), or null if not stated.
- customers_affected_sources: Array of URL(s) explicitly cited that support the count; empty array if none.
- geographic_scope: The stated geographic scope (e.g., "nationwide", "across the United States"), or null if not stated.
- geographic_scope_sources: Array of URL(s) explicitly cited that support the scope; empty array if none.

Rules:
- Only extract URLs explicitly present in the answer. If none are given for a field, return an empty array for that field.
- Preserve the exact phrasing as it appears in the answer for text fields.
"""


def prompt_extract_compensation() -> str:
    return """
Extract the Verizon customer compensation program details from the answer.
Return a JSON object with:
- compensation_amount: The stated monetary compensation (e.g., "$20 account credit"), or null if not stated.
- compensation_amount_sources: Array of URL(s) that support the amount; empty array if none.
- compensation_distribution_method: The stated distribution method (e.g., "automatically applied"), or null if not stated.
- compensation_distribution_method_sources: Array of URL(s) that support the method; empty array if none.
- compensation_start_date: The start date when credits began being applied (e.g., "January 15, 2026"), or null if not stated.
- compensation_start_date_sources: Array of URL(s) that support the start date; empty array if none.

Only extract URLs explicitly present in the answer.
"""


def prompt_extract_fcc() -> str:
    return """
Extract the FCC regulatory response details from the answer.
Return a JSON object with:
- complaint_email: The email address for FCC complaint intake about this outage, or null if not stated.
- complaint_email_sources: Array of URL(s) that support the email; empty array if none.
- complaint_deadline: The submission deadline date for complaints (e.g., "March 16, 2026"), or null if not stated.
- complaint_deadline_sources: Array of URL(s) that support the deadline; empty array if none.

Only extract URLs explicitly present in the answer.
"""


def prompt_extract_technical() -> str:
    return """
Extract the technical root cause details from the answer.
Return a JSON object with:
- network_technology: The stated network technology involved (e.g., "5G standalone (SA) network"), or null if not stated.
- network_technology_sources: Array of URL(s) that support the technology; empty array if none.
- root_cause: The stated root cause (e.g., "software issue"), or null if not stated.
- root_cause_sources: Array of URL(s) that support the root cause; empty array if none.

Only extract URLs explicitly present in the answer.
"""


def prompt_extract_comparative() -> str:
    return """
Extract the comparative network reliability context details from the answer.
Return a JSON object with:
- opensignal_reliability_desc: The stated Opensignal Reliability Experience comparison (e.g., "T-Mobile and Verizon tied with 898 points in January 2025"), or null.
- opensignal_reliability_sources: Array of URL(s) supporting this Opensignal claim; empty array if none.
- tmobile_coverage_98_desc: The stated T-Mobile 5G coverage claim for 98% of Americans, or null.
- tmobile_coverage_98_sources: Array of URL(s) supporting this claim; empty array if none.
- tmobile_largest_5g_desc: The stated T-Mobile "largest 5G network" claim (as a claim), or null.
- tmobile_largest_5g_sources: Array of URL(s) supporting this claim; empty array if none.
- tmobile_roadtrip_96_2_desc: The stated T-Mobile 5G coverage rate of 96.2% based on road-trip testing, or null.
- tmobile_roadtrip_96_2_sources: Array of URL(s) supporting this; empty array if none.
- verizon_5g_99_desc: The stated Verizon 5G coverage of 99% of the U.S. population, or null.
- verizon_5g_99_sources: Array of URL(s) supporting this; empty array if none.
- verizon_lte_gt99_desc: The stated Verizon 4G LTE coverage more than 99% of the U.S. population, or null.
- verizon_lte_gt99_sources: Array of URL(s) supporting this; empty array if none.
- verizon_metro_100_desc: The stated Verizon reliability leadership in 100 metro markets (RootMetrics 1H 2025), or null.
- verizon_metro_100_sources: Array of URL(s) supporting this; empty array if none.

Only extract URLs explicitly present in the answer. Preserve phrasing for text fields.
"""


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def has_value_and_sources(value: Optional[str], sources: List[str]) -> bool:
    return bool(value and value.strip()) and bool(sources and len(sources) > 0)


async def add_existence_gate(evaluator: Evaluator, parent, base_id: str, desc: str, value: Optional[str], sources: List[str]):
    return evaluator.add_custom_node(
        result=has_value_and_sources(value, sources),
        id=f"{base_id}_existence",
        desc=f"{desc} – value stated and at least one source URL provided",
        parent=parent,
        critical=True
    )


async def add_value_match_leaf(evaluator: Evaluator, parent, base_id: str, desc: str, claim: str, additional_instruction: str = "None"):
    node = evaluator.add_leaf(
        id=f"{base_id}_value_match",
        desc=desc,
        parent=parent,
        critical=True
    )
    await evaluator.verify(
        claim=claim,
        node=node,
        additional_instruction=additional_instruction
    )


async def add_source_supported_leaf(evaluator: Evaluator, parent, base_id: str, desc: str, claim: str, sources: List[str], additional_instruction: str = "None"):
    node = evaluator.add_leaf(
        id=f"{base_id}_source_supported",
        desc=desc,
        parent=parent,
        critical=True
    )
    await evaluator.verify(
        claim=claim,
        node=node,
        sources=sources,
        additional_instruction=additional_instruction
    )


# --------------------------------------------------------------------------- #
# Verification builders per section                                           #
# --------------------------------------------------------------------------- #
async def build_outage_event_checks(evaluator: Evaluator, parent, event: OutageEventExtraction):
    # Section node (critical parallel)
    section = evaluator.add_parallel(
        id="Outage_Event_Documentation",
        desc="Outage date, duration, customer impact scale, and geographic scope, each supported by source URL(s).",
        parent=parent,
        critical=True
    )

    # Event Date
    group_date = evaluator.add_parallel(
        id="Event_Date_With_Source",
        desc="States the outage date as January 14, 2026 AND provides supporting source URL(s).",
        parent=section,
        critical=True
    )
    await add_existence_gate(
        evaluator, group_date, "Event_Date_With_Source",
        "Outage date",
        event.event_date, event.event_date_sources
    )
    await add_value_match_leaf(
        evaluator, group_date, "Event_Date_With_Source",
        "The stated outage date matches the expected 'January 14, 2026'",
        claim=f"The stated outage date '{event.event_date}' matches 'January 14, 2026' allowing minor formatting like 'Jan. 14, 2026'.",
        additional_instruction="Allow minor formatting variants (e.g., Jan 14, 2026, Jan. 14, 2026). Case-insensitive."
    )
    await add_source_supported_leaf(
        evaluator, group_date, "Event_Date_With_Source",
        "Outage date is supported by cited source URL(s)",
        claim="The Verizon network outage occurred on January 14, 2026.",
        sources=event.event_date_sources,
        additional_instruction="Verify that the source explicitly states (or clearly shows) that the outage occurred on January 14, 2026. Minor date formatting is acceptable."
    )

    # Outage Duration
    group_duration = evaluator.add_parallel(
        id="Outage_Duration_With_Source",
        desc="States the outage duration as at least 10 hours AND provides supporting source URL(s).",
        parent=section,
        critical=True
    )
    await add_existence_gate(
        evaluator, group_duration, "Outage_Duration_With_Source",
        "Outage duration",
        event.event_duration, event.event_duration_sources
    )
    await add_value_match_leaf(
        evaluator, group_duration, "Outage_Duration_With_Source",
        "The stated outage duration indicates ≥ 10 hours",
        claim=f"The stated outage duration description '{event.event_duration}' indicates the outage lasted at least 10 hours.",
        additional_instruction="Interpret typical phrasing (e.g., 'about 12 hours', 'more than 10 hours', timeframe spanning ≥10h)."
    )
    await add_source_supported_leaf(
        evaluator, group_duration, "Outage_Duration_With_Source",
        "Outage duration (≥10 hours) is supported by cited source URL(s)",
        claim="The outage lasted at least 10 hours.",
        sources=event.event_duration_sources,
        additional_instruction="Confirm the source states or clearly implies the outage duration was 10 hours or more."
    )

    # Customers Affected
    group_customers = evaluator.add_parallel(
        id="Customers_Affected_With_Source",
        desc="States the affected-customer count as between 1.5 and 2 million AND provides supporting source URL(s).",
        parent=section,
        critical=True
    )
    await add_existence_gate(
        evaluator, group_customers, "Customers_Affected_With_Source",
        "Affected customer count",
        event.customers_affected, event.customers_affected_sources
    )
    await add_value_match_leaf(
        evaluator, group_customers, "Customers_Affected_With_Source",
        "The stated affected-customer count indicates 1.5–2.0 million",
        claim=f"The stated affected-customer count '{event.customers_affected}' indicates a value between 1.5 million and 2 million (inclusive).",
        additional_instruction="Treat phrasings like 'more than 1.5 million', 'around 1.8 million', or 'nearly 2 million' as within 1.5–2.0M."
    )
    await add_source_supported_leaf(
        evaluator, group_customers, "Customers_Affected_With_Source",
        "Affected-customer count (1.5–2.0 million) is supported by cited source URL(s)",
        claim="Between 1.5 and 2 million customers were affected by the Verizon outage.",
        sources=event.customers_affected_sources,
        additional_instruction="Numbers reasonably equivalent (e.g., 1.8M) should be accepted as within the 1.5–2.0M range."
    )

    # Geographic Scope
    group_scope = evaluator.add_parallel(
        id="Geographic_Scope_With_Source",
        desc="States the outage scope as nationwide across the United States AND provides supporting source URL(s).",
        parent=section,
        critical=True
    )
    await add_existence_gate(
        evaluator, group_scope, "Geographic_Scope_With_Source",
        "Geographic scope",
        event.geographic_scope, event.geographic_scope_sources
    )
    await add_value_match_leaf(
        evaluator, group_scope, "Geographic_Scope_With_Source",
        "The stated geographic scope indicates a nationwide U.S. outage",
        claim=f"The stated geographic scope '{event.geographic_scope}' indicates the outage was nationwide across the United States.",
        additional_instruction="Accept synonyms like 'nationwide', 'across the U.S.', 'across the United States' or similar."
    )
    await add_source_supported_leaf(
        evaluator, group_scope, "Geographic_Scope_With_Source",
        "Nationwide scope is supported by cited source URL(s)",
        claim="The outage affected users nationwide across the United States.",
        sources=event.geographic_scope_sources,
        additional_instruction="Source should explicitly say 'nationwide' or equivalent phrasing indicating U.S.-wide impact."
    )


async def build_compensation_checks(evaluator: Evaluator, parent, comp: CustomerCompensationExtraction):
    section = evaluator.add_parallel(
        id="Customer_Compensation_Program",
        desc="Verizon customer compensation amount and distribution details, supported by source URL(s).",
        parent=parent,
        critical=True
    )

    # Amount
    group_amt = evaluator.add_parallel(
        id="Compensation_Amount_With_Source",
        desc="States Verizon provided a $20 account credit to affected customers AND provides supporting source URL(s).",
        parent=section,
        critical=True
    )
    await add_existence_gate(
        evaluator, group_amt, "Compensation_Amount_With_Source",
        "Compensation amount",
        comp.compensation_amount, comp.compensation_amount_sources
    )
    await add_value_match_leaf(
        evaluator, group_amt, "Compensation_Amount_With_Source",
        "The stated compensation amount matches '$20 account credit'",
        claim=f"The stated compensation amount '{comp.compensation_amount}' matches '$20 account credit' (allowing minor phrasing variations).",
        additional_instruction="Accept equivalent phrasing such as '$20 credit', '$20 bill credit', or '$20 account credit'."
    )
    await add_source_supported_leaf(
        evaluator, group_amt, "Compensation_Amount_With_Source",
        "Compensation amount is supported by cited source URL(s)",
        claim="Verizon provided a $20 account credit to affected customers.",
        sources=comp.compensation_amount_sources,
        additional_instruction="Confirm the source explicitly states a $20 account/bill credit for affected customers."
    )

    # Distribution Method
    group_method = evaluator.add_parallel(
        id="Compensation_Distribution_Method_With_Source",
        desc="States the credits were automatically applied to affected customer accounts AND provides supporting source URL(s).",
        parent=section,
        critical=True
    )
    await add_existence_gate(
        evaluator, group_method, "Compensation_Distribution_Method_With_Source",
        "Compensation distribution method",
        comp.compensation_distribution_method, comp.compensation_distribution_method_sources
    )
    await add_value_match_leaf(
        evaluator, group_method, "Compensation_Distribution_Method_With_Source",
        "The stated distribution method matches 'automatically applied to affected accounts'",
        claim=f"The stated distribution method '{comp.compensation_distribution_method}' indicates the credits were automatically applied to affected customer accounts.",
        additional_instruction="Accept equivalent wording indicating automatic application without customer action."
    )
    await add_source_supported_leaf(
        evaluator, group_method, "Compensation_Distribution_Method_With_Source",
        "Distribution method is supported by cited source URL(s)",
        claim="The credits were automatically applied to affected customer accounts.",
        sources=comp.compensation_distribution_method_sources,
        additional_instruction="Verify that the source states credits are applied automatically (no customer action required)."
    )

    # Start Date
    group_start = evaluator.add_parallel(
        id="Compensation_Distribution_Start_Date_With_Source",
        desc="States the credits began being applied starting January 15, 2026 AND provides supporting source URL(s).",
        parent=section,
        critical=True
    )
    await add_existence_gate(
        evaluator, group_start, "Compensation_Distribution_Start_Date_With_Source",
        "Compensation start date",
        comp.compensation_start_date, comp.compensation_start_date_sources
    )
    await add_value_match_leaf(
        evaluator, group_start, "Compensation_Distribution_Start_Date_With_Source",
        "The stated start date matches 'January 15, 2026'",
        claim=f"The stated start date '{comp.compensation_start_date}' matches 'January 15, 2026' allowing minor formatting variations.",
        additional_instruction="Allow minor date formatting variants."
    )
    await add_source_supported_leaf(
        evaluator, group_start, "Compensation_Distribution_Start_Date_With_Source",
        "Start date is supported by cited source URL(s)",
        claim="The credits began being applied starting January 15, 2026.",
        sources=comp.compensation_start_date_sources,
        additional_instruction="Confirm the source explicitly states credits started on January 15, 2026."
    )


async def build_fcc_checks(evaluator: Evaluator, parent, fcc: FCCResponseExtraction):
    section = evaluator.add_parallel(
        id="FCC_Regulatory_Response",
        desc="FCC complaint intake email address and submission deadline, supported by source URL(s).",
        parent=parent,
        critical=True
    )

    # FCC complaint email
    group_email = evaluator.add_parallel(
        id="FCC_Complaint_Email_With_Source",
        desc="Provides the FCC complaint email address VerizonOutage2026@fcc.gov AND provides supporting source URL(s).",
        parent=section,
        critical=True
    )
    await add_existence_gate(
        evaluator, group_email, "FCC_Complaint_Email_With_Source",
        "FCC complaint email address",
        fcc.complaint_email, fcc.complaint_email_sources
    )
    await add_value_match_leaf(
        evaluator, group_email, "FCC_Complaint_Email_With_Source",
        "The stated FCC complaint email address matches 'VerizonOutage2026@fcc.gov'",
        claim=f"The stated FCC complaint email '{fcc.complaint_email}' matches 'VerizonOutage2026@fcc.gov' (case-insensitive).",
        additional_instruction="Emails should be compared case-insensitively. Minor whitespace differences should be ignored."
    )
    await add_source_supported_leaf(
        evaluator, group_email, "FCC_Complaint_Email_With_Source",
        "FCC complaint email address is supported by cited source URL(s)",
        claim="The FCC established the complaint email address VerizonOutage2026@fcc.gov for this outage.",
        sources=fcc.complaint_email_sources,
        additional_instruction="Verify the source explicitly lists VerizonOutage2026@fcc.gov for outage-related complaints."
    )

    # FCC complaint deadline
    group_deadline = evaluator.add_parallel(
        id="FCC_Complaint_Deadline_With_Source",
        desc="Provides the FCC complaint submission deadline as March 16, 2026 AND provides supporting source URL(s).",
        parent=section,
        critical=True
    )
    await add_existence_gate(
        evaluator, group_deadline, "FCC_Complaint_Deadline_With_Source",
        "FCC complaint deadline",
        fcc.complaint_deadline, fcc.complaint_deadline_sources
    )
    await add_value_match_leaf(
        evaluator, group_deadline, "FCC_Complaint_Deadline_With_Source",
        "The stated FCC complaint deadline matches 'March 16, 2026'",
        claim=f"The stated FCC complaint deadline '{fcc.complaint_deadline}' matches 'March 16, 2026' allowing minor formatting variations.",
        additional_instruction="Allow minor date formatting variants (e.g., Mar. 16, 2026)."
    )
    await add_source_supported_leaf(
        evaluator, group_deadline, "FCC_Complaint_Deadline_With_Source",
        "FCC complaint deadline is supported by cited source URL(s)",
        claim="The FCC complaint submission deadline for this outage was March 16, 2026.",
        sources=fcc.complaint_deadline_sources,
        additional_instruction="Verify the source explicitly states the complaint submission deadline date."
    )


async def build_technical_checks(evaluator: Evaluator, parent, tech: TechnicalRootCauseExtraction):
    section = evaluator.add_parallel(
        id="Technical_Root_Cause_Analysis",
        desc="Network technology involved and the technical failure/root cause, supported by source URL(s).",
        parent=parent,
        critical=True
    )

    # Network technology
    group_tech = evaluator.add_parallel(
        id="Network_Technology_With_Source",
        desc="States the outage involved Verizon's 5G standalone (SA) network AND provides supporting source URL(s).",
        parent=section,
        critical=True
    )
    await add_existence_gate(
        evaluator, group_tech, "Network_Technology_With_Source",
        "Network technology",
        tech.network_technology, tech.network_technology_sources
    )
    await add_value_match_leaf(
        evaluator, group_tech, "Network_Technology_With_Source",
        "The stated network technology matches '5G standalone (SA) network'",
        claim=f"The stated network technology '{tech.network_technology}' matches '5G standalone (SA) network' allowing slight phrasing differences.",
        additional_instruction="Accept equivalent phrasing such as '5G SA', '5G standalone', or '5G SA core'."
    )
    await add_source_supported_leaf(
        evaluator, group_tech, "Network_Technology_With_Source",
        "Network technology is supported by cited source URL(s)",
        claim="The outage involved Verizon's 5G standalone (SA) network.",
        sources=tech.network_technology_sources,
        additional_instruction="Verify the source clearly links the outage to Verizon's 5G standalone (SA) network."
    )

    # Root cause
    group_cause = evaluator.add_parallel(
        id="Root_Cause_With_Source",
        desc="States the root cause was a software issue (software failure) AND provides supporting source URL(s).",
        parent=section,
        critical=True
    )
    await add_existence_gate(
        evaluator, group_cause, "Root_Cause_With_Source",
        "Root cause",
        tech.root_cause, tech.root_cause_sources
    )
    await add_value_match_leaf(
        evaluator, group_cause, "Root_Cause_With_Source",
        "The stated root cause matches 'software issue'",
        claim=f"The stated root cause '{tech.root_cause}' matches 'software issue' allowing equivalent phrasing like 'software failure'.",
        additional_instruction="Accept 'software issue', 'software failure', or closely equivalent phrasing."
    )
    await add_source_supported_leaf(
        evaluator, group_cause, "Root_Cause_With_Source",
        "Root cause is supported by cited source URL(s)",
        claim="The root cause of the outage was a software issue.",
        sources=tech.root_cause_sources,
        additional_instruction="Verify the source explicitly points to a software issue/failure as the cause."
    )


async def build_comparative_checks(evaluator: Evaluator, parent, comp: ComparativeContextExtraction):
    section = evaluator.add_parallel(
        id="Comparative_Network_Reliability_Context",
        desc="Pre-outage comparative context including Opensignal reliability metrics, coverage statistics/claims, and Verizon metro-market reliability leadership, each supported by source URL(s).",
        parent=parent,
        critical=True
    )

    # Opensignal Reliability tie 898
    group_open = evaluator.add_parallel(
        id="Opensignal_Reliability_Comparison_With_Source",
        desc="States that (per Opensignal's January 2025 report) T-Mobile and Verizon tied for Reliability Experience with 898 points AND provides supporting source URL(s).",
        parent=section,
        critical=True
    )
    await add_existence_gate(
        evaluator, group_open, "Opensignal_Reliability_Comparison_With_Source",
        "Opensignal Reliability Experience tie description",
        comp.opensignal_reliability_desc, comp.opensignal_reliability_sources
    )
    await add_value_match_leaf(
        evaluator, group_open, "Opensignal_Reliability_Comparison_With_Source",
        "The stated Opensignal Reliability Experience comparison matches a tie at 898 points",
        claim=f"The stated description '{comp.opensignal_reliability_desc}' indicates that T-Mobile and Verizon tied for Reliability Experience with 898 points in January 2025.",
        additional_instruction="Allow minor paraphrases; the key is both tied at 898 points in Jan 2025 for Reliability Experience."
    )
    await add_source_supported_leaf(
        evaluator, group_open, "Opensignal_Reliability_Comparison_With_Source",
        "Opensignal tie is supported by cited source URL(s)",
        claim="According to Opensignal's January 2025 report, T-Mobile and Verizon tied for Reliability Experience with 898 points.",
        sources=comp.opensignal_reliability_sources,
        additional_instruction="Verify the source shows the 'Reliability Experience' category with both T-Mobile and Verizon at 898 points."
    )

    # T-Mobile 98% coverage
    group_tm_98 = evaluator.add_parallel(
        id="TMobile_5G_Coverage_98_Percent_With_Source",
        desc="States T-Mobile has 5G coverage for 98% of Americans AND provides supporting source URL(s).",
        parent=section,
        critical=True
    )
    await add_existence_gate(
        evaluator, group_tm_98, "TMobile_5G_Coverage_98_Percent_With_Source",
        "T-Mobile 5G coverage 98% claim",
        comp.tmobile_coverage_98_desc, comp.tmobile_coverage_98_sources
    )
    await add_value_match_leaf(
        evaluator, group_tm_98, "TMobile_5G_Coverage_98_Percent_With_Source",
        "The stated T-Mobile coverage claim matches '98% of Americans'",
        claim=f"The stated description '{comp.tmobile_coverage_98_desc}' indicates T-Mobile has 5G coverage for 98% of Americans.",
        additional_instruction="Accept equivalent phrasing like 'covers 98% of Americans' or '98% population coverage'."
    )
    await add_source_supported_leaf(
        evaluator, group_tm_98, "TMobile_5G_Coverage_98_Percent_With_Source",
        "T-Mobile 98% coverage claim is supported by cited source URL(s)",
        claim="T-Mobile has 5G coverage for 98% of Americans.",
        sources=comp.tmobile_coverage_98_sources,
        additional_instruction="Verify the source explicitly mentions 98% 5G coverage for Americans."
    )

    # T-Mobile largest 5G network claim
    group_tm_largest = evaluator.add_parallel(
        id="TMobile_Largest_5G_Network_Claim_With_Source",
        desc="States T-Mobile claims to have the largest 5G network AND provides supporting source URL(s).",
        parent=section,
        critical=True
    )
    await add_existence_gate(
        evaluator, group_tm_largest, "TMobile_Largest_5G_Network_Claim_With_Source",
        "T-Mobile 'largest 5G network' claim",
        comp.tmobile_largest_5g_desc, comp.tmobile_largest_5g_sources
    )
    await add_value_match_leaf(
        evaluator, group_tm_largest, "TMobile_Largest_5G_Network_Claim_With_Source",
        "The stated claim matches 'T-Mobile claims to have the largest 5G network'",
        claim=f"The stated description '{comp.tmobile_largest_5g_desc}' indicates that T-Mobile claims to have the largest 5G network.",
        additional_instruction="We only require that the source attributes this as T-Mobile's claim."
    )
    await add_source_supported_leaf(
        evaluator, group_tm_largest, "TMobile_Largest_5G_Network_Claim_With_Source",
        "T-Mobile 'largest 5G network' claim is supported by cited source URL(s)",
        claim="T-Mobile claims to have the largest 5G network.",
        sources=comp.tmobile_largest_5g_sources,
        additional_instruction="Verify that the source attributes 'largest 5G network' as a T-Mobile claim."
    )

    # T-Mobile road-trip 96.2% coverage
    group_tm_rt = evaluator.add_parallel(
        id="TMobile_RoadTrip_5G_Coverage_96_2_Percent_With_Source",
        desc="States T-Mobile's 5G coverage rate is 96.2% based on road-trip testing AND provides supporting source URL(s).",
        parent=section,
        critical=True
    )
    await add_existence_gate(
        evaluator, group_tm_rt, "TMobile_RoadTrip_5G_Coverage_96_2_Percent_With_Source",
        "T-Mobile road-trip coverage 96.2% claim",
        comp.tmobile_roadtrip_96_2_desc, comp.tmobile_roadtrip_96_2_sources
    )
    await add_value_match_leaf(
        evaluator, group_tm_rt, "TMobile_RoadTrip_5G_Coverage_96_2_Percent_With_Source",
        "The stated road-trip coverage rate matches 96.2%",
        claim=f"The stated description '{comp.tmobile_roadtrip_96_2_desc}' indicates T-Mobile's 5G coverage rate is 96.2% based on road-trip testing.",
        additional_instruction="Allow minor formatting like '96.2 percent'."
    )
    await add_source_supported_leaf(
        evaluator, group_tm_rt, "TMobile_RoadTrip_5G_Coverage_96_2_Percent_With_Source",
        "Road-trip 96.2% coverage is supported by cited source URL(s)",
        claim="T-Mobile's 5G coverage rate is 96.2% based on road-trip testing.",
        sources=comp.tmobile_roadtrip_96_2_sources,
        additional_instruction="Verify the source explicitly states 96.2% coverage from road-trip testing."
    )

    # Verizon 5G 99%
    group_vz_5g = evaluator.add_parallel(
        id="Verizon_5G_Coverage_99_Percent_With_Source",
        desc="States Verizon claims 5G coverage to 99% of the U.S. population AND provides supporting source URL(s).",
        parent=section,
        critical=True
    )
    await add_existence_gate(
        evaluator, group_vz_5g, "Verizon_5G_Coverage_99_Percent_With_Source",
        "Verizon 5G 99% coverage claim",
        comp.verizon_5g_99_desc, comp.verizon_5g_99_sources
    )
    await add_value_match_leaf(
        evaluator, group_vz_5g, "Verizon_5G_Coverage_99_Percent_With_Source",
        "The stated Verizon 5G coverage claim matches '99% of the U.S. population'",
        claim=f"The stated description '{comp.verizon_5g_99_desc}' indicates Verizon claims 5G coverage to 99% of the U.S. population.",
        additional_instruction="We only require that the source attributes this as a Verizon claim."
    )
    await add_source_supported_leaf(
        evaluator, group_vz_5g, "Verizon_5G_Coverage_99_Percent_With_Source",
        "Verizon 5G 99% coverage claim is supported by cited source URL(s)",
        claim="Verizon claims 5G coverage to 99% of the U.S. population.",
        sources=comp.verizon_5g_99_sources,
        additional_instruction="Verify the source attributes '99% 5G coverage' as a Verizon claim."
    )

    # Verizon LTE >99%
    group_vz_lte = evaluator.add_parallel(
        id="Verizon_4G_LTE_Coverage_More_Than_99_Percent_With_Source",
        desc="States Verizon claims 4G LTE coverage to more than 99% of the U.S. population AND provides supporting source URL(s).",
        parent=section,
        critical=True
    )
    await add_existence_gate(
        evaluator, group_vz_lte, "Verizon_4G_LTE_Coverage_More_Than_99_Percent_With_Source",
        "Verizon 4G LTE >99% coverage claim",
        comp.verizon_lte_gt99_desc, comp.verizon_lte_gt99_sources
    )
    await add_value_match_leaf(
        evaluator, group_vz_lte, "Verizon_4G_LTE_Coverage_More_Than_99_Percent_With_Source",
        "The stated Verizon LTE coverage claim matches 'more than 99%'",
        claim=f"The stated description '{comp.verizon_lte_gt99_desc}' indicates Verizon claims 4G LTE coverage to more than 99% of the U.S. population.",
        additional_instruction="We only require that the source attributes this as a Verizon claim."
    )
    await add_source_supported_leaf(
        evaluator, group_vz_lte, "Verizon_4G_LTE_Coverage_More_Than_99_Percent_With_Source",
        "Verizon 4G LTE >99% coverage claim is supported by cited source URL(s)",
        claim="Verizon claims 4G LTE coverage to more than 99% of the U.S. population.",
        sources=comp.verizon_lte_gt99_sources,
        additional_instruction="Verify the source attributes '>99% LTE coverage' as a Verizon claim."
    )

    # Verizon metro markets reliability leadership
    group_vz_metro = evaluator.add_parallel(
        id="Verizon_Metro_Markets_Reliability_Lead_With_Source",
        desc="States Verizon won reliability awards in 100 metro markets (RootMetrics 1H 2025) AND provides supporting source URL(s).",
        parent=section,
        critical=True
    )
    await add_existence_gate(
        evaluator, group_vz_metro, "Verizon_Metro_Markets_Reliability_Lead_With_Source",
        "Verizon reliability leadership in metro markets",
        comp.verizon_metro_100_desc, comp.verizon_metro_100_sources
    )
    await add_value_match_leaf(
        evaluator, group_vz_metro, "Verizon_Metro_Markets_Reliability_Lead_With_Source",
        "The stated leadership matches '100 metro markets (RootMetrics 1H 2025)'",
        claim=f"The stated description '{comp.verizon_metro_100_desc}' indicates Verizon won reliability awards in 100 metro markets in RootMetrics 1H 2025.",
        additional_instruction="Accept equivalent phrasing that clearly conveys 100 metro markets for RootMetrics 1H 2025."
    )
    await add_source_supported_leaf(
        evaluator, group_vz_metro, "Verizon_Metro_Markets_Reliability_Lead_With_Source",
        "Verizon metro markets leadership is supported by cited source URL(s)",
        claim="Verizon won reliability awards in 100 metro markets in RootMetrics 1H 2025.",
        sources=comp.verizon_metro_100_sources,
        additional_instruction="Verify the source explicitly states 100 metro markets leadership in RootMetrics 1H 2025."
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
    Evaluate an answer for the Verizon January 2026 outage comprehensive report task.
    """
    # Initialize evaluator at framework root (non-critical), then build our critical report root below it
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

    # Extract all sections (in parallel)
    event_task = evaluator.extract(
        prompt=prompt_extract_outage_event(),
        template_class=OutageEventExtraction,
        extraction_name="outage_event"
    )
    comp_task = evaluator.extract(
        prompt=prompt_extract_compensation(),
        template_class=CustomerCompensationExtraction,
        extraction_name="compensation"
    )
    fcc_task = evaluator.extract(
        prompt=prompt_extract_fcc(),
        template_class=FCCResponseExtraction,
        extraction_name="fcc_response"
    )
    tech_task = evaluator.extract(
        prompt=prompt_extract_technical(),
        template_class=TechnicalRootCauseExtraction,
        extraction_name="technical_root_cause"
    )
    comp_ctx_task = evaluator.extract(
        prompt=prompt_extract_comparative(),
        template_class=ComparativeContextExtraction,
        extraction_name="comparative_context"
    )

    event, comp, fcc, tech, comp_ctx = await asyncio.gather(
        event_task, comp_task, fcc_task, tech_task, comp_ctx_task
    )

    # Add ground truth to summary for transparency
    evaluator.add_ground_truth({
        "expected_values": EXPECTED
    }, gt_type="ground_truth")

    # Build top-level critical report node
    report_root = evaluator.add_parallel(
        id="Verizon_January_2026_Outage_Comprehensive_Report",
        desc="Research report covering the five required dimensions with supporting source URL(s) for each required piece of information.",
        parent=root,
        critical=True
    )

    # Build five critical sections under the report root
    await build_outage_event_checks(evaluator, report_root, event)
    await build_compensation_checks(evaluator, report_root, comp)
    await build_fcc_checks(evaluator, report_root, fcc)
    await build_technical_checks(evaluator, report_root, tech)
    await build_comparative_checks(evaluator, report_root, comp_ctx)

    # Return the structured evaluation summary
    return evaluator.get_summary()