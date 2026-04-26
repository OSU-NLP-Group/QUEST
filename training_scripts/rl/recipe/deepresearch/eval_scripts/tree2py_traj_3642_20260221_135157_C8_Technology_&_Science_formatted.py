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
TASK_ID = "major_carrier_outage_2024_2026"
TASK_DESCRIPTION = (
    "Analyze a major network outage event from one of the three largest US wireless carriers (AT&T, Verizon, or "
    "T-Mobile) that occurred between January 2024 and February 2026. Your analysis must include:\n\n"
    "1. Outage Event Identification: Identify and document a specific qualifying outage event, including the carrier "
    "name, exact date of occurrence, total duration (must be at least 30 minutes to meet FCC reportable thresholds), "
    "geographic areas or cities affected, and the peak number of customers or user reports impacted. Provide a "
    "reference URL from a credible news source or official statement documenting this event.\n\n"
    "2. FCC Compliance Requirements: Verify that this outage meets FCC Network Outage Reporting System (NORS) "
    "reporting criteria based on its duration of at least 30 minutes and impact scale. Confirm whether the carrier "
    "would be required to submit NORS notification within 120 minutes, an initial report within 3 calendar days, and a "
    "final report within 30 days after discovering the outage. If the outage affected 911 emergency services, document "
    "the 30-minute notification requirement for 911 call centers. Provide a reference URL to official FCC NORS "
    "requirements.\n\n"
    "3. Technical Infrastructure Analysis: Identify the primary network technology affected (4G LTE, 5G, or both) and "
    "document the typical coverage radius specifications for that technology in the affected area type (urban areas: "
    "0.25-1 mile radius; general areas: 1-3 miles radius). Identify the reported root cause category (such as software "
    "error, hardware failure, configuration issue, or external factor), and document the actual restoration timeline "
    "or the carrier's stated timeline. Assess whether this timeline aligns with typical carrier SLA standards of 4-5 "
    "hours for standard service restoration. Provide a reference URL with technical details about the outage cause or "
    "restoration process.\n\n"
    "4. Impact Assessment and Response: Identify which service types were impacted (voice calls, data, SMS, emergency "
    "services) and provide quantifiable customer impact metrics where available (number of users affected, call volume "
    "blocked, or user-minutes lost). Assess whether emergency communications (911 calls, public safety services) were "
    "affected. Document whether the carrier issued an official public statement, offered customer compensation or "
    "credits, and mentioned any redundancy or failover systems (diverse routes, backup systems, failover protocols). "
    "Calculate whether the outage duration would exceed the standard 99.9% uptime SLA that permits 8.76 hours of "
    "annual downtime. Provide a reference URL documenting customer impact, carrier response, or any FCC investigation "
    "related to this outage.\n\n"
    "All information must be supported by verifiable reference URLs from credible sources such as news outlets, "
    "official carrier statements, FCC documentation, or industry reports."
)


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class OutageEvent(BaseModel):
    carrier_name: Optional[str] = None
    outage_date: Optional[str] = None
    outage_duration: Optional[str] = None
    geographic_areas: List[str] = Field(default_factory=list)
    peak_impact: Optional[str] = None
    outage_reference_urls: List[str] = Field(default_factory=list)


class FCCCompliance(BaseModel):
    fcc_reference_urls: List[str] = Field(default_factory=list)


class TechnicalAnalysis(BaseModel):
    network_technology: Optional[str] = None  # e.g., "4G LTE", "5G", "both"
    coverage_specifications: Optional[str] = None  # e.g., "urban: 0.25–1 mile; general: 1–3 miles"
    root_cause_category: Optional[str] = None  # e.g., "software error", "hardware failure", etc.
    restoration_timeline: Optional[str] = None  # e.g., "service restored in ~4 hours"
    sla_compliance_assessment: Optional[str] = None  # e.g., "aligns with 4–5 hours" or "does not align"
    technical_reference_urls: List[str] = Field(default_factory=list)


class ImpactResponse(BaseModel):
    service_types_affected: List[str] = Field(default_factory=list)  # e.g., ["voice", "data", "SMS", "911"]
    customer_impact_metrics: Optional[str] = None  # numeric detail as text
    emergency_communications_impact: Optional[str] = None  # e.g., "911 affected", "No 911 impact"
    official_statement_urls: List[str] = Field(default_factory=list)
    redundancy_measures: Optional[str] = None  # e.g., "diverse routes, failover protocols"
    customer_compensation: Optional[str] = None  # e.g., "one-day credit offered"
    uptime_impact_calculation: Optional[str] = None  # e.g., "Does not exceed 8.76 hours"
    impact_reference_urls: List[str] = Field(default_factory=list)


class OutageAnalysisExtraction(BaseModel):
    outage_event: Optional[OutageEvent] = None
    fcc: Optional[FCCCompliance] = None
    technical: Optional[TechnicalAnalysis] = None
    impact: Optional[ImpactResponse] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_outage_analysis() -> str:
    return """
Extract the outage analysis details from the answer. Return a JSON object with the following structure:

{
  "outage_event": {
    "carrier_name": string | null,                // One of: "AT&T", "Verizon", "T-Mobile" (as presented)
    "outage_date": string | null,                 // Specific date (e.g., "February 22, 2024" or "2024-02-22")
    "outage_duration": string | null,             // Duration text (e.g., "4 hours", "45 minutes", "several hours")
    "geographic_areas": string[] ,                // Cities, states, or "nationwide" as listed
    "peak_impact": string | null,                 // Peak affected users or user reports (e.g., "1.7 million", "tens of thousands")
    "outage_reference_urls": string[]             // URLs cited for the event (news/official statements)
  },
  "fcc": {
    "fcc_reference_urls": string[]                // URLs to official FCC NORS requirements
  },
  "technical": {
    "network_technology": string | null,          // "4G LTE", "5G", or "both"
    "coverage_specifications": string | null,     // e.g., "urban: 0.25–1 mile; general: 1–3 miles"
    "root_cause_category": string | null,         // e.g., "software error", "hardware failure", "configuration issue", "external factor"
    "restoration_timeline": string | null,        // e.g., "restored in ~4 hours"
    "sla_compliance_assessment": string | null,   // e.g., "aligns with 4–5 hours" or "does not align"
    "technical_reference_urls": string[]          // URLs documenting technical details
  },
  "impact": {
    "service_types_affected": string[],           // e.g., ["voice", "data", "SMS", "emergency services"]
    "customer_impact_metrics": string | null,     // e.g., "1.7M customers affected", "123k user reports"
    "emergency_communications_impact": string | null, // e.g., "911 affected", "No 911 impact"
    "official_statement_urls": string[],          // URLs to carrier's official statements (if any)
    "redundancy_measures": string | null,         // e.g., "backup systems, failover protocols" (if mentioned)
    "customer_compensation": string | null,       // e.g., "bill credit", "day of service credit" (if any)
    "uptime_impact_calculation": string | null,   // e.g., "exceeds 8.76 hours" or "does not exceed"
    "impact_reference_urls": string[]             // URLs documenting customer impact, response, or investigation
  }
}

Rules:
- Extract exactly what the answer states; do not invent or infer missing values.
- For URLs, include only explicit HTTP/HTTPS links that appear in the answer.
- If any field is missing, set it to null or [] as appropriate.
"""


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _list_or_empty(values: Optional[List[str]]) -> List[str]:
    return values if values else []


def _str_or_empty(value: Optional[str]) -> str:
    return value if value else ""


def _join_list_readable(items: List[str]) -> str:
    if not items:
        return ""
    return "; ".join(items)


def _combine_unique(*lists: List[str]) -> List[str]:
    seen = set()
    res = []
    for lst in lists:
        for x in lst:
            if x and x not in seen:
                seen.add(x)
                res.append(x)
    return res


def _has_fcc_official_url(urls: List[str]) -> bool:
    return any(isinstance(u, str) and ("fcc.gov" in u.lower()) for u in urls)


# --------------------------------------------------------------------------- #
# Verification: Outage Event Identification                                   #
# --------------------------------------------------------------------------- #
async def verify_outage_event_identification(
    evaluator: Evaluator,
    parent_node,
    extraction: OutageAnalysisExtraction
) -> None:
    data = extraction.outage_event or OutageEvent()
    carrier = _str_or_empty(data.carrier_name)
    date_str = _str_or_empty(data.outage_date)
    duration_str = _str_or_empty(data.outage_duration)
    areas = _list_or_empty(data.geographic_areas)
    peak = _str_or_empty(data.peak_impact)
    outage_urls = _list_or_empty(data.outage_reference_urls)

    group = evaluator.add_parallel(
        id="Outage_Event_Identification",
        desc="Identification and documentation of a qualifying major network outage event",
        parent=evaluator.root,
        critical=False  # Set to non-critical to allow mixed child criticality
    )

    # Carrier Identity (critical)
    leaf_carrier = evaluator.add_leaf(
        id="Carrier_Identity",
        desc="The outage must be from one of the major US wireless carriers (AT&T, Verizon, or T-Mobile)",
        parent=group,
        critical=True
    )
    carrier_claim = f"This page documents an outage from {carrier}."
    await evaluator.verify(
        claim=carrier_claim,
        node=leaf_carrier,
        sources=outage_urls,
        additional_instruction=(
            "Verify the carrier name (AT&T, Verizon, or T-Mobile) referenced in the page. "
            "Minor variations (e.g., AT&T vs ATT) are acceptable."
        ),
    )

    # Outage Date (critical)
    leaf_date = evaluator.add_leaf(
        id="Outage_Date",
        desc="Provide the specific date when the outage occurred (must be between January 2024 and February 2026)",
        parent=group,
        critical=True
    )
    date_claim = (
        f"The outage occurred on {date_str}, which is between January 2024 and February 2026."
    )
    await evaluator.verify(
        claim=date_claim,
        node=leaf_date,
        sources=outage_urls,
        additional_instruction=(
            "Confirm the reported outage date on the page. "
            "Allow common date formats (e.g., 'Feb. 22, 2024', 'February 22, 2024', or '2024-02-22')."
        ),
    )

    # Outage Duration (critical)
    leaf_duration = evaluator.add_leaf(
        id="Outage_Duration",
        desc="Document the total duration of the outage (must meet FCC reportable threshold of at least 30 minutes)",
        parent=group,
        critical=True
    )
    duration_claim = (
        f"The outage duration was at least 30 minutes; the reported duration is: {duration_str}."
    )
    await evaluator.verify(
        claim=duration_claim,
        node=leaf_duration,
        sources=outage_urls,
        additional_instruction=(
            "Look for duration or time range in the article. If the article states 'hours' or a time window exceeding 30 minutes, that qualifies."
        ),
    )

    # Geographic Scope (critical)
    leaf_geo = evaluator.add_leaf(
        id="Geographic_Scope",
        desc="Identify the geographic areas or cities affected by the outage",
        parent=group,
        critical=True
    )
    areas_str = _join_list_readable(areas)
    geo_claim = f"The outage affected the following areas or regions: {areas_str}."
    await evaluator.verify(
        claim=geo_claim,
        node=leaf_geo,
        sources=outage_urls,
        additional_instruction=(
            "Consider the claim supported if the page indicates a nationwide impact or lists most of the major locations mentioned."
        ),
    )

    # Peak Impact Scale (non-critical)
    leaf_peak = evaluator.add_leaf(
        id="Peak_Impact_Scale",
        desc="Document the peak number of customers or user reports affected during the outage",
        parent=group,
        critical=False
    )
    peak_claim = f"The peak impact during the outage was: {peak}."
    await evaluator.verify(
        claim=peak_claim,
        node=leaf_peak,
        sources=outage_urls,
        additional_instruction=(
            "Look for numbers like '1.7 million customers', 'tens of thousands', or Downdetector peak report counts."
        ),
    )

    # Reference URL for Outage (critical presence check)
    evaluator.add_custom_node(
        result=len(outage_urls) > 0,
        id="Reference_URL_Outage",
        desc="Provide a reference URL from a news source or official statement documenting this outage event",
        parent=group,
        critical=True
    )


# --------------------------------------------------------------------------- #
# Verification: FCC Compliance Requirements                                   #
# --------------------------------------------------------------------------- #
async def verify_fcc_compliance_requirements(
    evaluator: Evaluator,
    parent_node,
    extraction: OutageAnalysisExtraction
) -> None:
    event = extraction.outage_event or OutageEvent()
    fcc = extraction.fcc or FCCCompliance()

    outage_urls = _list_or_empty(event.outage_reference_urls)
    fcc_urls = _list_or_empty(fcc.fcc_reference_urls)

    group = evaluator.add_parallel(
        id="FCC_Compliance_Requirements",
        desc="Verification of FCC NORS reporting and compliance obligations for the identified outage",
        parent=parent_node,
        critical=False  # Mixed criticality children
    )

    # NORS Reportability (critical) – verify ≥ 30 minutes & significant impact via outage sources
    leaf_reportable = evaluator.add_leaf(
        id="NORS_Reportability",
        desc="Confirm the outage meets FCC NORS reporting threshold (duration of at least 30 minutes and impact criteria)",
        parent=group,
        critical=True
    )
    reportable_claim = (
        "This outage lasted at least 30 minutes and impacted a significant number of users, "
        "therefore it meets FCC NORS reportability thresholds."
    )
    await evaluator.verify(
        claim=reportable_claim,
        node=leaf_reportable,
        sources=outage_urls,
        additional_instruction=(
            "Check that the source indicates the outage duration was ≥ 30 minutes and that it had a broad customer impact."
        ),
    )

    # Initial notification within 120 minutes (critical) – verify using FCC official doc
    leaf_notify_120 = evaluator.add_leaf(
        id="Initial_Notification_Timeframe",
        desc="Identify whether the carrier would be required to submit NORS notification within 120 minutes for this type of outage",
        parent=group,
        critical=True
    )
    notify_claim = "FCC NORS rules require an initial outage notification within 120 minutes of discovering a reportable outage."
    await evaluator.verify(
        claim=notify_claim,
        node=leaf_notify_120,
        sources=fcc_urls,
        additional_instruction=(
            "Verify on the FCC official page (fcc.gov) that an initial notification is due within 120 minutes for reportable outages."
        ),
    )

    # Initial report within 3 calendar days (critical)
    leaf_initial_3d = evaluator.add_leaf(
        id="Initial_Report_Deadline",
        desc="Confirm that an initial outage report must be submitted within 3 calendar days after discovering the outage",
        parent=group,
        critical=True
    )
    initial_claim = "FCC NORS rules require that an initial outage report be submitted within 3 calendar days after discovery of the outage."
    await evaluator.verify(
        claim=initial_claim,
        node=leaf_initial_3d,
        sources=fcc_urls,
        additional_instruction="Verify the 3 calendar days initial report requirement on the FCC NORS documentation.",
    )

    # Final report within 30 days (critical)
    leaf_final_30d = evaluator.add_leaf(
        id="Final_Report_Deadline",
        desc="Confirm that a final outage report must be submitted no later than 30 days after discovering the outage",
        parent=group,
        critical=True
    )
    final_claim = "FCC NORS rules require that a final outage report be submitted no later than 30 days after discovery of the outage."
    await evaluator.verify(
        claim=final_claim,
        node=leaf_final_30d,
        sources=fcc_urls,
        additional_instruction="Verify the 30 day final report requirement on the FCC NORS documentation.",
    )

    # 911 service impact (non-critical) – verify with outage sources whether 911 was affected
    leaf_911 = evaluator.add_leaf(
        id="911_Service_Impact",
        desc="If the outage affected 911 services, document whether 911 call centers would need to be notified within 30 minutes",
        parent=group,
        critical=False
    )
    # Use the answer's statement; allow verification against outage sources
    emergency_impact = _str_or_empty((extraction.impact or ImpactResponse()).emergency_communications_impact)
    impact_claim = (
        f"Emergency communications impact: {emergency_impact}. "
        "If 911 was affected, FCC requires 30-minute notification to PSAPs (911 call centers)."
    )
    await evaluator.verify(
        claim=impact_claim,
        node=leaf_911,
        sources=outage_urls,
        additional_instruction=(
            "Check whether the page indicates any impact to 911 or public safety answering points (PSAPs). "
            "The mention of 30-minute PSAP notification is a general FCC rule context."
        ),
    )

    # FCC Reference URL present and official (critical presence/officiality check)
    evaluator.add_custom_node(
        result=_has_fcc_official_url(fcc_urls),
        id="Reference_URL_FCC",
        desc="Provide a reference URL to official FCC NORS requirements documentation",
        parent=group,
        critical=True
    )


# --------------------------------------------------------------------------- #
# Verification: Technical Infrastructure Analysis                             #
# --------------------------------------------------------------------------- #
async def verify_technical_infrastructure_analysis(
    evaluator: Evaluator,
    parent_node,
    extraction: OutageAnalysisExtraction
) -> None:
    tech = extraction.technical or TechnicalAnalysis()
    event = extraction.outage_event or OutageEvent()

    tech_urls = _list_or_empty(tech.technical_reference_urls)
    outage_urls = _list_or_empty(event.outage_reference_urls)
    sources_pref = tech_urls if tech_urls else outage_urls

    group = evaluator.add_parallel(
        id="Technical_Infrastructure_Analysis",
        desc="Analysis of technical aspects including network architecture, coverage specifications, and restoration capabilities",
        parent=parent_node,
        critical=False  # Mixed criticality children
    )

    # Network Technology (critical)
    leaf_tech = evaluator.add_leaf(
        id="Network_Technology",
        desc="Identify the primary network technology affected (4G LTE, 5G, or both)",
        parent=group,
        critical=True
    )
    network_claim = f"The primary network technology affected was: { _str_or_empty(tech.network_technology) }."
    await evaluator.verify(
        claim=network_claim,
        node=leaf_tech,
        sources=sources_pref,
        additional_instruction="Confirm whether the page indicates 4G/LTE, 5G, or both were impacted."
    )

    # Coverage Specifications (non-critical) – general technical knowledge statement
    leaf_coverage = evaluator.add_leaf(
        id="Coverage_Specifications",
        desc="Document the typical coverage radius for the affected network technology in the impacted area type (urban: 0.25-1 mile, general: 1-3 miles)",
        parent=group,
        critical=False
    )
    coverage_text = _str_or_empty(tech.coverage_specifications)
    coverage_claim = (
        f"Typical coverage radius for { _str_or_empty(tech.network_technology) } is described as: {coverage_text}. "
        "Standard references: urban ~0.25–1 mile, general ~1–3 miles."
    )
    await evaluator.verify(
        claim=coverage_claim,
        node=leaf_coverage,
        sources=None,
        additional_instruction=(
            "This is a general engineering rule-of-thumb statement; accept reasonable phrasing consistent with the stated ranges."
        ),
    )

    # Root Cause Category (critical)
    leaf_root_cause = evaluator.add_leaf(
        id="Root_Cause_Category",
        desc="Identify the reported root cause category of the outage (e.g., software error, hardware failure, configuration issue, external factor)",
        parent=group,
        critical=True
    )
    root_cause_claim = f"The reported root cause category for the outage was: { _str_or_empty(tech.root_cause_category) }."
    await evaluator.verify(
        claim=root_cause_claim,
        node=leaf_root_cause,
        sources=sources_pref,
        additional_instruction="Confirm the stated cause category (software, hardware, configuration, external, etc.) from the cited source."
    )

    # Restoration Timeline (non-critical)
    leaf_restoration = evaluator.add_leaf(
        id="Restoration_Timeline",
        desc="Document the actual time taken to restore service or the carrier's stated restoration timeline",
        parent=group,
        critical=False
    )
    restoration_claim = f"The stated restoration timeline was: { _str_or_empty(tech.restoration_timeline) }."
    await evaluator.verify(
        claim=restoration_claim,
        node=leaf_restoration,
        sources=sources_pref,
        additional_instruction="Look for language like 'restored within X hours' or status updates indicating the timeline."
    )

    # SLA Compliance (non-critical) – reasoning check
    leaf_sla = evaluator.add_leaf(
        id="SLA_Compliance",
        desc="Assess whether the restoration timeline aligns with typical carrier SLA standards (4-5 hours for standard restoration)",
        parent=group,
        critical=False
    )
    sla_claim = (
        f"Based on the restoration timeline '{ _str_or_empty(tech.restoration_timeline) }', "
        f"the assessment is: { _str_or_empty(tech.sla_compliance_assessment) } relative to a 4–5 hour standard restoration SLA."
    )
    await evaluator.verify(
        claim=sla_claim,
        node=leaf_sla,
        sources=None,
        additional_instruction="Make a reasoned judgment based on the stated restoration time and the 4–5 hour standard restoration guideline."
    )

    # Technical Reference URL (critical presence check)
    evaluator.add_custom_node(
        result=len(tech_urls) > 0,
        id="Reference_URL_Technical",
        desc="Provide a reference URL documenting technical details about the outage cause or restoration",
        parent=group,
        critical=True
    )


# --------------------------------------------------------------------------- #
# Verification: Impact Assessment and Response                                #
# --------------------------------------------------------------------------- #
async def verify_impact_assessment_and_response(
    evaluator: Evaluator,
    parent_node,
    extraction: OutageAnalysisExtraction
) -> None:
    impact = extraction.impact or ImpactResponse()
    event = extraction.outage_event or OutageEvent()
    tech = extraction.technical or TechnicalAnalysis()

    impact_urls = _list_or_empty(impact.impact_reference_urls)
    official_urls = _list_or_empty(impact.official_statement_urls)
    outage_urls = _list_or_empty(event.outage_reference_urls)
    tech_urls = _list_or_empty(tech.technical_reference_urls)

    # Choose sources preference for impact verification
    sources_pref = impact_urls if impact_urls else _combine_unique(official_urls, outage_urls, tech_urls)

    group = evaluator.add_parallel(
        id="Impact_Assessment_And_Response",
        desc="Assessment of customer impact, service disruption scope, and carrier response actions",
        parent=parent_node,
        critical=False  # Mixed criticality children
    )

    # Service Types Affected (critical)
    leaf_services = evaluator.add_leaf(
        id="Service_Types_Affected",
        desc="Identify which service types were impacted (voice calls, data, SMS, emergency services)",
        parent=group,
        critical=True
    )
    services_str = _join_list_readable(_list_or_empty(impact.service_types_affected))
    services_claim = f"The impacted service types included: {services_str}."
    await evaluator.verify(
        claim=services_claim,
        node=leaf_services,
        sources=sources_pref,
        additional_instruction="Confirm whether voice, data, SMS, and/or emergency services (911) were impacted as claimed."
    )

    # Customer Impact Quantification (non-critical)
    leaf_metrics = evaluator.add_leaf(
        id="Customer_Impact_Quantification",
        desc="Document quantifiable metrics of customer impact (number of users affected, call volume blocked, user-minutes lost)",
        parent=group,
        critical=False
    )
    metrics_claim = f"Customer impact metrics reported: { _str_or_empty(impact.customer_impact_metrics) }."
    await evaluator.verify(
        claim=metrics_claim,
        node=leaf_metrics,
        sources=sources_pref,
        additional_instruction="Look for counts of users affected, peak reports, or minutes/hours of service unavailability."
    )

    # Emergency Communications Impact (critical)
    leaf_emergency = evaluator.add_leaf(
        id="Emergency_Communications_Impact",
        desc="Assess whether emergency communications (911 calls, public safety) were affected by the outage",
        parent=group,
        critical=True
    )
    emergency_claim = f"Emergency communications impact: { _str_or_empty(impact.emergency_communications_impact) }."
    await evaluator.verify(
        claim=emergency_claim,
        node=leaf_emergency,
        sources=sources_pref,
        additional_instruction="Verify specifically whether 911 calling or public safety communications were affected."
    )

    # Carrier Official Statement (non-critical) – presence check
    evaluator.add_custom_node(
        result=len(official_urls) > 0,
        id="Carrier_Official_Statement",
        desc="Document whether the carrier issued an official public statement acknowledging the outage",
        parent=group,
        critical=False
    )

    # Redundancy Measures (non-critical)
    leaf_redundancy = evaluator.add_leaf(
        id="Redundancy_Measures",
        desc="Identify any redundancy or failover systems that were mentioned in relation to this outage (diverse routes, backup systems, failover protocols)",
        parent=group,
        critical=False
    )
    redundancy_claim = f"Redundancy or failover systems mentioned: { _str_or_empty(impact.redundancy_measures) }."
    await evaluator.verify(
        claim=redundancy_claim,
        node=leaf_redundancy,
        sources=sources_pref,
        additional_instruction="Verify any mention of backup systems, diverse routes, or failover protocols."
    )

    # Customer Compensation (non-critical)
    leaf_comp = evaluator.add_leaf(
        id="Customer_Compensation",
        desc="Document whether the carrier offered any compensation, credits, or remediation to affected customers",
        parent=group,
        critical=False
    )
    comp_claim = f"Customer compensation/remediation: { _str_or_empty(impact.customer_compensation) }."
    await evaluator.verify(
        claim=comp_claim,
        node=leaf_comp,
        sources=sources_pref,
        additional_instruction="Confirm any credits or compensation described in the cited source(s)."
    )

    # Uptime Impact Calculation (non-critical) – reasoning check
    leaf_uptime = evaluator.add_leaf(
        id="Uptime_Impact_Calculation",
        desc="Calculate whether this outage duration would exceed the standard 99.9% uptime SLA (which permits 8.76 hours annual downtime)",
        parent=group,
        critical=False
    )
    # Use both event duration and any explicit uptime statement if provided
    duration_str = _str_or_empty((extraction.technical or TechnicalAnalysis()).restoration_timeline)
    if not duration_str:
        duration_str = _str_or_empty((extraction.outage_event or OutageEvent()).outage_duration)
    uptime_calc = _str_or_empty(impact.uptime_impact_calculation)
    uptime_claim = (
        f"Given the outage duration '{duration_str}', the assessment is: {uptime_calc} relative to 99.9% annual uptime (8.76 hours)."
    )
    await evaluator.verify(
        claim=uptime_claim,
        node=leaf_uptime,
        sources=None,
        additional_instruction="Use basic reasoning based on the stated duration to decide if it exceeds the 8.76 hours annual allowance."
    )

    # Reference URL for Impact/Response (critical presence check)
    evaluator.add_custom_node(
        result=len(sources_pref) > 0,
        id="Reference_URL_Impact",
        desc="Provide a reference URL documenting customer impact, carrier response, or FCC investigation of this outage",
        parent=group,
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
    Evaluate an answer for the major carrier outage analysis task (2024–2026).
    """
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

    # Record criticality adjustments due to framework constraint:
    # In this evaluator, any parent node that contains non-critical children is set to non-critical because the
    # framework enforces that critical parents can only have critical children.
    evaluator.add_custom_info(
        info={
            "note": "Adjusted group-node criticality to satisfy framework constraints: "
                    "parents with mixed critical/non-critical children are set to non-critical. "
                    "Child nodes retain their criticality to preserve gating semantics."
        },
        info_type="meta",
        info_name="criticality_adjustment"
    )

    # Extract structured data from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_outage_analysis(),
        template_class=OutageAnalysisExtraction,
        extraction_name="outage_analysis_extraction"
    )

    # Build a top-level analysis node to contain all groups (kept non-critical to allow mixed children)
    analysis_node = evaluator.add_parallel(
        id="Major_Carrier_Outage_Analysis",
        desc="Comprehensive analysis of a major US telecommunications carrier's network outage event from 2024-2026, including FCC compliance, technical details, and response measures",
        parent=root,
        critical=False
    )

    # Run verification groups
    await verify_outage_event_identification(evaluator, analysis_node, extracted)
    await verify_fcc_compliance_requirements(evaluator, analysis_node, extracted)
    await verify_technical_infrastructure_analysis(evaluator, analysis_node, extracted)
    await verify_impact_assessment_and_response(evaluator, analysis_node, extracted)

    return evaluator.get_summary()