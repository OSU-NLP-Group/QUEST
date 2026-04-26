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
TASK_ID = "telecom_outage_regulatory_2024_2026"
TASK_DESCRIPTION = (
    "Analyze the two major U.S. telecommunications carrier outages that occurred on January 14, 2026 (Verizon) and "
    "February 22, 2024 (AT&T). For each outage, provide a comprehensive regulatory compliance assessment that includes: "
    "(1) Basic Factual Information: Outage date, duration, number of affected customers or blocked calls, and resolution details. "
    "(2) Technical Cause Analysis: Identify and explain the root technical cause of each outage, including the network architecture or system involved. "
    "(3) Geographic and Service Impact: Document the geographic scope (specific major cities or nationwide coverage) and the types of services affected (voice, text, data, 911 emergency services). "
    "(4) FCC NORS Reportability Assessment: Evaluate whether each outage met the FCC Network Outage Reporting System (NORS) thresholds, specifically the 30-minute minimum duration requirement and the 900,000 user-minutes threshold for wireless outages, and conclude whether the outage was reportable under FCC regulations. "
    "(5) Required Reporting Timeline Analysis: For each reportable outage, identify the FCC-mandated deadlines for initial notification (120 minutes from discovery), initial outage report (3 calendar days), final outage report (30 days), and 911 call center notification (30 minutes when emergency services are affected). "
    "(6) Carrier Response (for Verizon): Document the customer compensation offered and the method of credit distribution. "
    "(7) FCC Investigation Status: Report on any FCC investigations, including public comment deadlines and submission methods. "
    "For every major factual claim, technical detail, regulatory requirement, and impact metric, provide a supporting URL reference from an authoritative source."
)


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class OutageBasicFacts(BaseModel):
    date: Optional[str] = None
    duration: Optional[str] = None
    impact_metric: Optional[str] = None  # e.g., "more than 1.5 million customers", "92 million blocked calls"
    resolution_time: Optional[str] = None  # e.g., "10:20 PM ET on January 14, 2026"
    geographic_scope: Optional[str] = None  # e.g., "nationwide", "major cities including ..."
    sources: List[str] = Field(default_factory=list)


class TechnicalInfo(BaseModel):
    cause_type: Optional[str] = None  # e.g., "software issue in 5G Standalone core network", "configuration update issue"
    architecture: Optional[str] = None  # e.g., "5G SA core", nullable for AT&T if not applicable
    sources: List[str] = Field(default_factory=list)


class GeographicImpact(BaseModel):
    major_cities: List[str] = Field(default_factory=list)  # e.g., ["New York City", "Atlanta", ...]
    sources: List[str] = Field(default_factory=list)


class ServiceImpact(BaseModel):
    services: List[str] = Field(default_factory=list)  # e.g., ["voice", "text", "data"]
    impact_911: Optional[str] = None  # e.g., "affected", "not affected", "limited impact"
    sources: List[str] = Field(default_factory=list)


class TimelineRequirements(BaseModel):
    initial_notification_minutes: Optional[str] = None  # e.g., "120 minutes"
    initial_report_timeline: Optional[str] = None       # e.g., "3 calendar days"
    final_report_timeline: Optional[str] = None         # e.g., "30 days"
    psap_notification_minutes: Optional[str] = None     # e.g., "30 minutes"
    sources: List[str] = Field(default_factory=list)    # FCC NORS timeline/PSAP notification references


class CarrierResponse(BaseModel):
    credit_amount: Optional[str] = None  # e.g., "$20"
    credit_method: Optional[str] = None  # e.g., "myVerizon app redemption", "automatic bill credit"
    sources: List[str] = Field(default_factory=list)


class InvestigationInfo(BaseModel):
    status: Optional[str] = None             # e.g., "FCC opened a formal investigation"
    comment_deadline: Optional[str] = None   # e.g., "March 16, 2026"
    comment_email: Optional[str] = None      # e.g., "VerizonOutage2026@fcc.gov"
    sources: List[str] = Field(default_factory=list)


class VerizonExtraction(BaseModel):
    basic: Optional[OutageBasicFacts] = None
    technical: Optional[TechnicalInfo] = None
    geographic: Optional[GeographicImpact] = None
    services: Optional[ServiceImpact] = None
    timeline: Optional[TimelineRequirements] = None
    response: Optional[CarrierResponse] = None
    investigation: Optional[InvestigationInfo] = None
    nors_docs: List[str] = Field(default_factory=list)  # FCC NORS requirement references (general requirements)


class ATTExtraction(BaseModel):
    basic: Optional[OutageBasicFacts] = None
    technical: Optional[TechnicalInfo] = None
    services: Optional[ServiceImpact] = None
    timeline: Optional[TimelineRequirements] = None
    investigation: Optional[InvestigationInfo] = None
    nors_docs: List[str] = Field(default_factory=list)  # FCC NORS requirement references (can be same as Verizon's)


class OutagesExtraction(BaseModel):
    verizon: Optional[VerizonExtraction] = None
    att: Optional[ATTExtraction] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_outages() -> str:
    return """
Extract structured information from the answer about two major U.S. carrier outages: Verizon (January 14, 2026) and AT&T (February 22, 2024).
Return a JSON with two top-level keys: "verizon" and "att". For each, extract the following sections and fields.

General rules:
- Extract only what is explicitly stated in the answer.
- Use strings for dates, times, durations, counts, and deadlines (e.g., "January 14, 2026", "over 10 hours", "more than 1.5 million customers", "10:20 PM ET", "120 minutes", "3 calendar days", "30 days", "30 minutes").
- For each section, also collect all supporting source URLs explicitly included in the answer for that section (put them into the corresponding 'sources' list). Ignore malformed URLs.

For "verizon":
- basic: { date, duration, impact_metric, resolution_time, geographic_scope, sources }
- technical: { cause_type, architecture, sources }
- geographic: { major_cities[], sources }
- services: { services[], impact_911, sources }
- timeline: { initial_notification_minutes, initial_report_timeline, final_report_timeline, psap_notification_minutes, sources }
- response: { credit_amount, credit_method, sources }
- investigation: { status, comment_deadline, comment_email, sources }
- nors_docs: [ list of FCC NORS requirement URLs explicitly mentioned in the answer ]

For "att":
- basic: { date, duration, impact_metric, resolution_time, geographic_scope, sources }
- technical: { cause_type, architecture, sources }  # architecture may be null if not stated
- services: { services[], impact_911, sources }
- timeline: { initial_notification_minutes, initial_report_timeline, final_report_timeline, psap_notification_minutes, sources }
- investigation: { status, comment_deadline, comment_email, sources }
- nors_docs: [ list of FCC NORS requirement URLs explicitly mentioned in the answer ]
"""


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _urls(urls: Optional[List[str]]) -> List[str]:
    if not urls:
        return []
    cleaned = []
    for u in urls:
        if isinstance(u, str) and u.strip():
            s = u.strip()
            if s.startswith("http://") or s.startswith("https://"):
                cleaned.append(s)
            else:
                # add protocol if missing, as Extractor guidelines allow
                cleaned.append("http://" + s)
    # Deduplicate while preserving order
    seen = set()
    unique = []
    for u in cleaned:
        if u not in seen:
            seen.add(u)
            unique.append(u)
    return unique


def _str_or_empty(s: Optional[str]) -> str:
    return s if isinstance(s, str) else ""


async def _verify_leaf_with_urls(
    evaluator: Evaluator,
    *,
    parent,
    node_id: str,
    desc: str,
    claim: str,
    urls: List[str],
    critical: bool,
    add_inst: str = "None",
    extra_prereq: Optional[List] = None,
):
    """Create a leaf and verify a claim using provided URLs; gate by extra prerequisites if any."""
    leaf = evaluator.add_leaf(
        id=node_id,
        desc=desc,
        parent=parent,
        critical=critical,
    )
    await evaluator.verify(
        claim=claim,
        node=leaf,
        sources=urls if urls else None,
        additional_instruction=add_inst,
        extra_prerequisites=extra_prereq or [],
    )
    return leaf


def _add_ref_presence_node(
    evaluator: Evaluator,
    *,
    parent,
    node_id: str,
    desc: str,
    urls: List[str],
    critical: bool = True,
):
    """Add a custom presence node asserting that at least one URL was provided."""
    return evaluator.add_custom_node(
        result=len(urls) > 0,
        id=node_id,
        desc=desc,
        parent=parent,
        critical=critical
    )


# --------------------------------------------------------------------------- #
# Verizon verification sub-tree                                               #
# --------------------------------------------------------------------------- #
async def build_verizon_tree(evaluator: Evaluator, parent, ex: Optional[VerizonExtraction]):
    # Parent node for Verizon analysis (non-critical to allow partial credit across carriers)
    vz_root = evaluator.add_parallel(
        id="Verizon_January_2026_Outage_Analysis",
        desc="Complete analysis of the Verizon January 14, 2026 outage",
        parent=parent,
        critical=False
    )

    # Safe access
    basic = ex.basic if ex else None
    tech = ex.technical if ex else None
    geo = ex.geographic if ex else None
    svc = ex.services if ex else None
    tml = ex.timeline if ex else None
    resp = ex.response if ex else None
    inv = ex.investigation if ex else None
    nors_docs = _urls(ex.nors_docs if ex else [])

    # 1) Basic Facts
    basic_node = evaluator.add_parallel(
        id="Verizon_Outage_Basic_Facts",
        desc="Accurate reporting of basic factual information about the Verizon outage",
        parent=vz_root,
        critical=True
    )

    metrics_node = evaluator.add_parallel(
        id="Verizon_Outage_Metrics",
        desc="Core outage metrics including date, duration, and affected count",
        parent=basic_node,
        critical=True
    )

    basic_sources = _urls(basic.sources if basic else [])

    # Documentation node and reference presence
    basic_doc_node = evaluator.add_parallel(
        id="Verizon_Basic_Facts_Documentation",
        desc="Source documentation for basic facts",
        parent=basic_node,
        critical=True
    )
    vz_basic_ref = _add_ref_presence_node(
        evaluator,
        parent=basic_doc_node,
        node_id="Verizon_Basic_Facts_Reference",
        desc="Provides valid URL reference for Verizon outage basic facts",
        urls=basic_sources,
        critical=True
    )

    # Leaves under metrics (all critical under a critical parent)
    await _verify_leaf_with_urls(
        evaluator,
        parent=metrics_node,
        node_id="Verizon_Date",
        desc="Outage date correctly identified as January 14, 2026",
        claim=f"The Verizon outage date is {_str_or_empty(basic.date)}.",
        urls=basic_sources,
        critical=True,
        add_inst="Allow reasonable date formatting variations; focus on whether the page states this outage's date.",
        extra_prereq=[vz_basic_ref]
    )

    vz_duration_leaf = await _verify_leaf_with_urls(
        evaluator,
        parent=metrics_node,
        node_id="Verizon_Duration",
        desc="Outage duration correctly reported as exceeding 10 hours",
        claim=f"The Verizon outage duration is reported as {_str_or_empty(basic.duration)}.",
        urls=basic_sources,
        critical=True,
        add_inst="Verify the duration statement as written in the answer; allow approximations like 'over 10 hours' or '10+ hours' if supported.",
        extra_prereq=[vz_basic_ref]
    )

    vz_affected_leaf = await _verify_leaf_with_urls(
        evaluator,
        parent=metrics_node,
        node_id="Verizon_Affected_Count",
        desc="Number of affected customers correctly reported as more than 1.5 million",
        claim=f"The number of affected Verizon customers is reported as {_str_or_empty(basic.impact_metric)}.",
        urls=basic_sources,
        critical=True,
        add_inst="Check that the cited page supports the reported magnitude of affected customers.",
        extra_prereq=[vz_basic_ref]
    )

    await _verify_leaf_with_urls(
        evaluator,
        parent=metrics_node,
        node_id="Verizon_Resolution_Time",
        desc="Resolution time correctly reported as 10:20 PM ET on January 14, 2026",
        claim=f"The Verizon outage resolution time is reported as {_str_or_empty(basic.resolution_time)}.",
        urls=basic_sources,
        critical=True,  # Promote to critical to satisfy framework constraint for critical parent
        add_inst="Allow minor variations in punctuation and time zone phrasing if equivalent.",
        extra_prereq=[vz_basic_ref]
    )

    # 2) Technical Cause
    tech_node = evaluator.add_parallel(
        id="Verizon_Technical_Cause",
        desc="Accurate identification and explanation of the technical cause",
        parent=vz_root,
        critical=True
    )

    tech_details_node = evaluator.add_parallel(
        id="Verizon_Technical_Details",
        desc="Technical cause and architecture details",
        parent=tech_node,
        critical=True
    )

    tech_doc_node = evaluator.add_parallel(
        id="Verizon_Technical_Documentation",
        desc="Source documentation for technical information",
        parent=tech_node,
        critical=True
    )

    tech_sources = _urls(tech.sources if tech else [])
    vz_tech_ref = _add_ref_presence_node(
        evaluator,
        parent=tech_doc_node,
        node_id="Verizon_Technical_Reference",
        desc="Provides valid URL reference for technical cause information",
        urls=tech_sources,
        critical=True
    )

    await _verify_leaf_with_urls(
        evaluator,
        parent=tech_details_node,
        node_id="Verizon_Cause_Type",
        desc="Cause correctly identified as software issue in 5G Standalone core network",
        claim=f"The technical cause is reported as {_str_or_empty(tech.cause_type)}.",
        urls=tech_sources,
        critical=True,
        add_inst="Verify that the source supports the stated cause (e.g., software issue, configuration, etc.).",
        extra_prereq=[vz_tech_ref]
    )

    await _verify_leaf_with_urls(
        evaluator,
        parent=tech_details_node,
        node_id="Verizon_Architecture_Type",
        desc="Network architecture correctly identified as 5G SA (Standalone) using 5G core",
        claim=f"The network architecture involved is reported as {_str_or_empty(tech.architecture)}.",
        urls=tech_sources,
        critical=True,
        add_inst="Check that the page supports the stated network architecture (e.g., 5G Standalone core).",
        extra_prereq=[vz_tech_ref]
    )

    # 3) Geographic Impact
    geo_node = evaluator.add_parallel(
        id="Verizon_Geographic_Impact",
        desc="Accurate reporting of geographic scope and affected locations",
        parent=vz_root,
        critical=True
    )
    geo_details_node = evaluator.add_parallel(
        id="Verizon_Geographic_Details",
        desc="Geographic scope details",
        parent=geo_node,
        critical=True
    )
    geo_doc_node = evaluator.add_parallel(
        id="Verizon_Geographic_Documentation",
        desc="Source documentation for geographic impact",
        parent=geo_node,
        critical=True
    )
    geo_sources = _urls(geo.sources if geo else [])
    vz_geo_ref = _add_ref_presence_node(
        evaluator,
        parent=geo_doc_node,
        node_id="Verizon_Geographic_Reference",
        desc="Provides valid URL reference for geographic impact information",
        urls=geo_sources,
        critical=True
    )

    # Prepare claim about major cities: require "at least three of specific cities"
    candidate_cities = {"New York City", "Atlanta", "Charlotte", "Houston", "Washington D.C.", "Washington DC", "Washington, D.C."}
    extracted_cities = set([c.strip() for c in (geo.major_cities if geo else []) if isinstance(c, str) and c.strip()])
    matched = [c for c in extracted_cities if c in candidate_cities]
    if len(matched) >= 3:
        cities_phrase = ", ".join(list(matched)[:5])
        geo_claim = f"The Verizon outage impacted at least three major U.S. cities among New York City, Atlanta, Charlotte, Houston, and Washington D.C.; specifically: {cities_phrase}."
    else:
        # Fall back to claim using any provided cities list
        cities_phrase = ", ".join(list(extracted_cities)[:5]) if extracted_cities else "major U.S. cities"
        geo_claim = f"The Verizon outage impacted major U.S. cities including {cities_phrase}."

    await _verify_leaf_with_urls(
        evaluator,
        parent=geo_details_node,
        node_id="Verizon_Major_Cities",
        desc="Major affected cities correctly identified (must include at least 3 of: New York City, Atlanta, Charlotte, Houston, Washington D.C.)",
        claim=geo_claim,
        urls=geo_sources,
        critical=True,
        add_inst="Verify that the referenced page(s) list the cited cities as affected during the outage. Allow reasonable city name variants (e.g., 'Washington, D.C.' vs 'Washington DC').",
        extra_prereq=[vz_geo_ref]
    )

    # 4) Service Impact
    svc_node = evaluator.add_parallel(
        id="Verizon_Service_Impact",
        desc="Accurate reporting of service disruptions and affected services",
        parent=vz_root,
        critical=True
    )
    svc_details_node = evaluator.add_parallel(
        id="Verizon_Service_Details",
        desc="Service disruption details",
        parent=svc_node,
        critical=True
    )
    svc_doc_node = evaluator.add_parallel(
        id="Verizon_Service_Documentation",
        desc="Source documentation for service impact",
        parent=svc_node,
        critical=True
    )
    svc_sources = _urls(svc.sources if svc else [])
    vz_svc_ref = _add_ref_presence_node(
        evaluator,
        parent=svc_doc_node,
        node_id="Verizon_Service_Impact_Reference",
        desc="Provides valid URL reference for service impact information",
        urls=svc_sources,
        critical=True
    )

    # Services types claim
    extracted_services = [s.strip().lower() for s in (svc.services if svc else []) if isinstance(s, str) and s.strip()]
    services_claim = "The Verizon outage affected the following service types: " + (", ".join(extracted_services) if extracted_services else "voice, text, and data") + "."
    await _verify_leaf_with_urls(
        evaluator,
        parent=svc_details_node,
        node_id="Verizon_Service_Types",
        desc="Affected services correctly identified as voice, text, and data services",
        claim=services_claim,
        urls=svc_sources,
        critical=True,
        add_inst="Map synonyms appropriately (e.g., SMS/text, data/mobile data).",
        extra_prereq=[vz_svc_ref]
    )

    # 911 impact claim
    impact_911_text = _str_or_empty(svc.impact_911).lower()
    if "not" in impact_911_text or "no" in impact_911_text:
        claim_911 = "911 emergency services were not affected by the Verizon outage."
    elif impact_911_text:
        claim_911 = "911 emergency services were affected by the Verizon outage."
    else:
        claim_911 = "The outage impact on 911 emergency services is as reported in the sources."
    await _verify_leaf_with_urls(
        evaluator,
        parent=svc_details_node,
        node_id="Verizon_911_Impact",
        desc="Impact on 911 emergency services correctly reported",
        claim=claim_911,
        urls=svc_sources,
        critical=True,
        add_inst="Accept phrasing indicating partial, localized, or indirect impacts if supported by the citation.",
        extra_prereq=[vz_svc_ref]
    )

    # 5) NORS Reportability
    nors_node = evaluator.add_parallel(
        id="Verizon_NORS_Reportability",
        desc="Assessment of whether outage met FCC NORS reporting thresholds",
        parent=vz_root,
        critical=True
    )
    assessment_seq = evaluator.add_sequential(
        id="Verizon_Reportability_Assessment",
        desc="Threshold assessment and conclusion",
        parent=nors_node,
        critical=True
    )
    nors_doc_node = evaluator.add_parallel(
        id="Verizon_NORS_Documentation",
        desc="Source documentation for FCC NORS requirements",
        parent=nors_node,
        critical=True
    )
    vz_nors_ref = _add_ref_presence_node(
        evaluator,
        parent=nors_doc_node,
        node_id="Verizon_NORS_Reference",
        desc="Provides valid URL reference for FCC NORS requirements",
        urls=nors_docs,
        critical=True
    )

    # Duration threshold (simple verify; depend on duration fact + NORS ref)
    vz_duration_threshold = evaluator.add_leaf(
        id="Verizon_Duration_Threshold",
        desc="Correctly assesses that outage met 30-minute minimum duration threshold",
        parent=assessment_seq,
        critical=True
    )
    await evaluator.verify(
        claim=f"Given the reported duration '{_str_or_empty(basic.duration)}', the Verizon outage met the FCC NORS 30-minute minimum duration threshold.",
        node=vz_duration_threshold,
        sources=None,
        additional_instruction="Use the verified outage duration and general NORS threshold knowledge (documented in the NORS reference) to judge this.",
        extra_prerequisites=[vz_duration_leaf, vz_nors_ref]
    )

    # User-minutes threshold (simple verify; depend on affected count + duration + NORS ref)
    vz_user_minutes = evaluator.add_leaf(
        id="Verizon_User_Minutes_Threshold",
        desc="Correctly assesses that outage met 900,000 user-minutes threshold (1.5M+ customers × 10+ hours far exceeds threshold)",
        parent=assessment_seq,
        critical=True
    )
    await evaluator.verify(
        claim=f"Given the reported impact '{_str_or_empty(basic.impact_metric)}' and duration '{_str_or_empty(basic.duration)}', the outage exceeded the FCC NORS 900,000 user-minutes threshold for wireless outages.",
        node=vz_user_minutes,
        sources=None,
        additional_instruction="Apply rough reasoning based on magnitudes; do not require exact arithmetic if the product clearly exceeds the threshold. Rely on earlier verified facts and the NORS documentation.",
        extra_prerequisites=[vz_affected_leaf, vz_duration_leaf, vz_nors_ref]
    )

    # Reportability conclusion (simple verify; depend on previous two threshold leaves + NORS ref)
    vz_reportable = evaluator.add_leaf(
        id="Verizon_Reportability_Conclusion",
        desc="Correctly concludes that outage was reportable under FCC NORS requirements",
        parent=assessment_seq,
        critical=True
    )
    await evaluator.verify(
        claim="The Verizon outage was reportable under FCC NORS requirements.",
        node=vz_reportable,
        sources=None,
        additional_instruction="Use the determination from the duration and user‑minutes threshold checks.",
        extra_prerequisites=[vz_duration_threshold, vz_user_minutes, vz_nors_ref]
    )

    # 6) Timeline Compliance (deadlines)
    tml_node = evaluator.add_parallel(
        id="Verizon_Timeline_Compliance",
        desc="Analysis of expected FCC reporting timeline compliance",
        parent=vz_root,
        critical=True
    )

    gen_deadlines_node = evaluator.add_parallel(
        id="Verizon_General_Reporting_Deadlines",
        desc="General NORS reporting deadlines",
        parent=tml_node,
        critical=True
    )

    emg_node = evaluator.add_parallel(
        id="Verizon_Emergency_Notification_Requirements",
        desc="911-specific notification requirements",
        parent=tml_node,
        critical=True
    )

    tml_doc_node = evaluator.add_parallel(
        id="Verizon_Timeline_Documentation",
        desc="Source documentation for timeline requirements",
        parent=tml_node,
        critical=True
    )
    tml_sources = _urls(tml.sources if tml else [])
    vz_tml_ref = _add_ref_presence_node(
        evaluator,
        parent=tml_doc_node,
        node_id="Verizon_Timeline_Reference",
        desc="Provides valid URL reference for FCC reporting timeline requirements",
        urls=tml_sources,
        critical=True
    )

    await _verify_leaf_with_urls(
        evaluator,
        parent=gen_deadlines_node,
        node_id="Verizon_Initial_Notification_Deadline",
        desc="Correctly identifies 120-minute initial notification deadline from outage discovery",
        claim=f"FCC requires an initial notification within {_str_or_empty(tml.initial_notification_minutes)} of outage discovery.",
        urls=tml_sources,
        critical=True,
        add_inst="The expected value is 120 minutes; verify that the cited rule states this timing requirement. Allow formatting variants.",
        extra_prereq=[vz_tml_ref]
    )
    await _verify_leaf_with_urls(
        evaluator,
        parent=gen_deadlines_node,
        node_id="Verizon_Initial_Report_Deadline",
        desc="Correctly identifies 3-calendar-day initial report deadline",
        claim=f"FCC requires an initial outage report within {_str_or_empty(tml.initial_report_timeline)}.",
        urls=tml_sources,
        critical=True,
        add_inst="The expected value is 3 calendar days; verify that the cited rule states this.",
        extra_prereq=[vz_tml_ref]
    )
    await _verify_leaf_with_urls(
        evaluator,
        parent=gen_deadlines_node,
        node_id="Verizon_Final_Report_Deadline",
        desc="Correctly identifies 30-day final report deadline",
        claim=f"FCC requires a final outage report within {_str_or_empty(tml.final_report_timeline)}.",
        urls=tml_sources,
        critical=True,
        add_inst="The expected value is 30 days; verify that the cited rule states this.",
        extra_prereq=[vz_tml_ref]
    )
    await _verify_leaf_with_urls(
        evaluator,
        parent=emg_node,
        node_id="Verizon_911_Notification_Requirement",
        desc="Correctly identifies 30-minute 911 call center notification requirement for affected areas",
        claim=f"FCC requires notifying impacted 911 call centers within {_str_or_empty(tml.psap_notification_minutes)} when emergency services are affected.",
        urls=tml_sources,
        critical=True,
        add_inst="The expected value is 30 minutes; verify that the cited rule states this requirement.",
        extra_prereq=[vz_tml_ref]
    )

    # 7) Carrier Response (non-critical section)
    resp_node = evaluator.add_parallel(
        id="Verizon_Carrier_Response",
        desc="Analysis of Verizon's response and customer compensation",
        parent=vz_root,
        critical=False
    )
    comp_node = evaluator.add_parallel(
        id="Verizon_Compensation_Details",
        desc="Customer compensation details",
        parent=resp_node,
        critical=False
    )
    resp_doc_node = evaluator.add_parallel(
        id="Verizon_Response_Documentation",
        desc="Source documentation for carrier response",
        parent=resp_node,
        critical=True
    )
    resp_sources = _urls(resp.sources if resp else [])
    vz_resp_ref = _add_ref_presence_node(
        evaluator,
        parent=resp_doc_node,
        node_id="Verizon_Response_Reference",
        desc="Provides valid URL reference for carrier response information",
        urls=resp_sources,
        critical=True
    )

    await _verify_leaf_with_urls(
        evaluator,
        parent=comp_node,
        node_id="Verizon_Credit_Amount",
        desc="Correctly reports $20 account credit offered to affected customers",
        claim=f"Verizon offered a customer credit of {_str_or_empty(resp.credit_amount)} in response to the outage.",
        urls=resp_sources,
        critical=False,
        add_inst="Verify the specific amount and its applicability to affected customers.",
        extra_prereq=[vz_resp_ref]
    )
    await _verify_leaf_with_urls(
        evaluator,
        parent=comp_node,
        node_id="Verizon_Credit_Distribution",
        desc="Correctly describes credit redemption method through myVerizon app",
        claim=f"The credit distribution/redemption method is {_str_or_empty(resp.credit_method)}.",
        urls=resp_sources,
        critical=False,
        add_inst="Verify the method (e.g., via myVerizon app, automatic bill credit) as stated by the source.",
        extra_prereq=[vz_resp_ref]
    )

    # 8) FCC Investigation
    inv_node = evaluator.add_parallel(
        id="Verizon_FCC_Investigation",
        desc="Information about FCC investigation and public comment process",
        parent=vz_root,
        critical=True
    )
    inv_details_node = evaluator.add_parallel(
        id="Verizon_Investigation_Details",
        desc="Investigation status details",
        parent=inv_node,
        critical=True
    )
    public_part_node = evaluator.add_parallel(
        id="Verizon_Public_Participation",
        desc="Public comment process details",
        parent=inv_node,
        critical=True
    )
    inv_doc_node = evaluator.add_parallel(
        id="Verizon_Investigation_Documentation",
        desc="Source documentation for FCC investigation",
        parent=inv_node,
        critical=True
    )
    inv_sources = _urls(inv.sources if inv else [])
    vz_inv_ref = _add_ref_presence_node(
        evaluator,
        parent=inv_doc_node,
        node_id="Verizon_Investigation_Reference",
        desc="Provides valid URL reference for FCC investigation information",
        urls=inv_sources,
        critical=True
    )

    await _verify_leaf_with_urls(
        evaluator,
        parent=inv_details_node,
        node_id="Verizon_Investigation_Status",
        desc="Correctly reports that FCC opened formal investigation",
        claim=f"The FCC investigation status is reported as: {_str_or_empty(inv.status)}.",
        urls=inv_sources,
        critical=True,
        add_inst="Verify that the FCC (or authoritative source) confirms a formal investigation.",
        extra_prereq=[vz_inv_ref]
    )
    await _verify_leaf_with_urls(
        evaluator,
        parent=public_part_node,
        node_id="Verizon_Comment_Deadline",
        desc="Correctly identifies March 16, 2026 as public comment deadline",
        claim=f"The public comment deadline is {_str_or_empty(inv.comment_deadline)}.",
        urls=inv_sources,
        critical=True,
        add_inst="Verify the deadline date and ensure it is clearly stated on the cited source.",
        extra_prereq=[vz_inv_ref]
    )
    await _verify_leaf_with_urls(
        evaluator,
        parent=public_part_node,
        node_id="Verizon_Comment_Email",
        desc="Correctly provides VerizonOutage2026@fcc.gov as comment submission email",
        claim=f"The public comment submission email is {_str_or_empty(inv.comment_email)}.",
        urls=inv_sources,
        critical=True,
        add_inst="Verify that the email address for public comment submission is as stated.",
        extra_prereq=[vz_inv_ref]
    )


# --------------------------------------------------------------------------- #
# AT&T verification sub-tree                                                  #
# --------------------------------------------------------------------------- #
async def build_att_tree(evaluator: Evaluator, parent, ex: Optional[ATTExtraction]):
    att_root = evaluator.add_parallel(
        id="ATT_February_2024_Outage_Analysis",
        desc="Complete analysis of the AT&T February 22, 2024 outage",
        parent=parent,
        critical=False
    )

    basic = ex.basic if ex else None
    tech = ex.technical if ex else None
    svc = ex.services if ex else None
    tml = ex.timeline if ex else None
    inv = ex.investigation if ex else None
    nors_docs = _urls(ex.nors_docs if ex else [])

    # 1) Basic Facts
    basic_node = evaluator.add_parallel(
        id="ATT_Outage_Basic_Facts",
        desc="Accurate reporting of basic factual information about the AT&T outage",
        parent=att_root,
        critical=True
    )
    metrics_node = evaluator.add_parallel(
        id="ATT_Outage_Metrics",
        desc="Core outage metrics including date, duration, and impact",
        parent=basic_node,
        critical=True
    )
    basic_doc_node = evaluator.add_parallel(
        id="ATT_Basic_Facts_Documentation",
        desc="Source documentation for basic facts",
        parent=basic_node,
        critical=True
    )

    basic_sources = _urls(basic.sources if basic else [])
    att_basic_ref = _add_ref_presence_node(
        evaluator,
        parent=basic_doc_node,
        node_id="ATT_Basic_Facts_Reference",
        desc="Provides valid URL reference for AT&T outage basic facts",
        urls=basic_sources,
        critical=True
    )

    att_date_leaf = await _verify_leaf_with_urls(
        evaluator,
        parent=metrics_node,
        node_id="ATT_Date",
        desc="Outage date correctly identified as February 22, 2024",
        claim=f"The AT&T outage date is {_str_or_empty(basic.date)}.",
        urls=basic_sources,
        critical=True,
        add_inst="Allow reasonable date formatting variations.",
        extra_prereq=[att_basic_ref]
    )
    att_duration_leaf = await _verify_leaf_with_urls(
        evaluator,
        parent=metrics_node,
        node_id="ATT_Duration",
        desc="Outage duration correctly reported as over 12 hours",
        claim=f"The AT&T outage duration is reported as {_str_or_empty(basic.duration)}.",
        urls=basic_sources,
        critical=True,
        add_inst="Verify the duration statement; allow approximations like 'over 12 hours' if supported.",
        extra_prereq=[att_basic_ref]
    )
    att_blocked_calls_leaf = await _verify_leaf_with_urls(
        evaluator,
        parent=metrics_node,
        node_id="ATT_Blocked_Calls",
        desc="Number of blocked calls correctly reported as more than 92 million",
        claim=f"The number of blocked calls (or equivalent impact) is reported as {_str_or_empty(basic.impact_metric)}.",
        urls=basic_sources,
        critical=True,
        add_inst="Verify the magnitude of blocked/failed calls (or equivalent impact) as reported.",
        extra_prereq=[att_basic_ref]
    )
    await _verify_leaf_with_urls(
        evaluator,
        parent=metrics_node,
        node_id="ATT_Geographic_Scope",
        desc="Geographic scope correctly reported as nationwide (all 50 states plus territories)",
        claim=f"The geographic scope is reported as {_str_or_empty(basic.geographic_scope)}.",
        urls=basic_sources,
        critical=True,
        add_inst="Verify that the scope is national or otherwise as stated in the answer and supported by the citation.",
        extra_prereq=[att_basic_ref]
    )

    # 2) Technical Cause
    tech_node = evaluator.add_parallel(
        id="ATT_Technical_Cause",
        desc="Accurate identification and explanation of the technical cause",
        parent=att_root,
        critical=True
    )
    tech_details_node = evaluator.add_parallel(
        id="ATT_Technical_Details",
        desc="Technical cause details",
        parent=tech_node,
        critical=True
    )
    tech_doc_node = evaluator.add_parallel(
        id="ATT_Technical_Documentation",
        desc="Source documentation for technical information",
        parent=tech_node,
        critical=True
    )

    tech_sources = _urls(tech.sources if tech else [])
    att_tech_ref = _add_ref_presence_node(
        evaluator,
        parent=tech_doc_node,
        node_id="ATT_Technical_Reference",
        desc="Provides valid URL reference for AT&T technical cause information",
        urls=tech_sources,
        critical=True
    )

    await _verify_leaf_with_urls(
        evaluator,
        parent=tech_details_node,
        node_id="ATT_Cause_Type",
        desc="Cause correctly identified as network update/configuration issue",
        claim=f"The technical cause is reported as {_str_or_empty(tech.cause_type)}.",
        urls=tech_sources,
        critical=True,
        add_inst="Verify that the cited page supports the stated root cause (e.g., software update, misconfiguration).",
        extra_prereq=[att_tech_ref]
    )

    # 3) Service Impact
    svc_node = evaluator.add_parallel(
        id="ATT_Service_Impact",
        desc="Accurate reporting of service disruptions",
        parent=att_root,
        critical=True
    )
    svc_details_node = evaluator.add_parallel(
        id="ATT_Service_Details",
        desc="Service disruption details",
        parent=svc_node,
        critical=True
    )
    svc_doc_node = evaluator.add_parallel(
        id="ATT_Service_Documentation",
        desc="Source documentation for service impact",
        parent=svc_node,
        critical=True
    )

    svc_sources = _urls(svc.sources if svc else [])
    att_svc_ref = _add_ref_presence_node(
        evaluator,
        parent=svc_doc_node,
        node_id="ATT_Service_Impact_Reference",
        desc="Provides valid URL reference for AT&T service impact information",
        urls=svc_sources,
        critical=True
    )

    extracted_services = [s.strip().lower() for s in (svc.services if svc else []) if isinstance(s, str) and s.strip()]
    services_claim = "The AT&T outage affected the following service types: " + (", ".join(extracted_services) if extracted_services else "voice calls, texts, and mobile data") + "."
    await _verify_leaf_with_urls(
        evaluator,
        parent=svc_details_node,
        node_id="ATT_Service_Types",
        desc="Affected services correctly identified (voice calls, texts, mobile data)",
        claim=services_claim,
        urls=svc_sources,
        critical=True,
        add_inst="Map synonyms appropriately (e.g., SMS/text, data/mobile data).",
        extra_prereq=[att_svc_ref]
    )

    impact_911_text = _str_or_empty(svc.impact_911).lower()
    if "not" in impact_911_text or "no" in impact_911_text:
        claim_911 = "911 emergency services were not affected by the AT&T outage."
    elif impact_911_text:
        claim_911 = "911 emergency services were affected by the AT&T outage."
    else:
        claim_911 = "The outage impact on 911 emergency services is as reported in the sources."
    await _verify_leaf_with_urls(
        evaluator,
        parent=svc_details_node,
        node_id="ATT_911_Impact",
        desc="Impact on 911 emergency services correctly reported",
        claim=claim_911,
        urls=svc_sources,
        critical=True,
        add_inst="Accept phrasing indicating partial, localized, or indirect impacts if supported.",
        extra_prereq=[att_svc_ref]
    )

    # 4) NORS Reportability
    nors_node = evaluator.add_parallel(
        id="ATT_NORS_Reportability",
        desc="Assessment of whether outage met FCC NORS reporting thresholds",
        parent=att_root,
        critical=True
    )
    assessment_seq = evaluator.add_sequential(
        id="ATT_Reportability_Assessment",
        desc="Threshold assessment and conclusion",
        parent=nors_node,
        critical=True
    )
    nors_doc_node = evaluator.add_parallel(
        id="ATT_NORS_Documentation",
        desc="Source documentation for FCC NORS requirements",
        parent=nors_node,
        critical=True
    )
    att_nors_ref = _add_ref_presence_node(
        evaluator,
        parent=nors_doc_node,
        node_id="ATT_NORS_Reference",
        desc="Provides valid URL reference for FCC NORS requirements (can be same as Verizon's)",
        urls=nors_docs,
        critical=True
    )

    att_duration_threshold = evaluator.add_leaf(
        id="ATT_Duration_Threshold",
        desc="Correctly assesses that outage met 30-minute minimum duration threshold",
        parent=assessment_seq,
        critical=True
    )
    await evaluator.verify(
        claim=f"Given the reported duration '{_str_or_empty(basic.duration)}', the AT&T outage met the FCC NORS 30-minute minimum duration threshold.",
        node=att_duration_threshold,
        sources=None,
        additional_instruction="Use the verified outage duration and the documented NORS requirement.",
        extra_prerequisites=[att_duration_leaf, att_nors_ref]
    )

    att_user_minutes = evaluator.add_leaf(
        id="ATT_User_Minutes_Threshold",
        desc="Correctly assesses that outage met 900,000 user-minutes threshold (92M+ blocked calls over 12+ hours far exceeds threshold)",
        parent=assessment_seq,
        critical=True
    )
    await evaluator.verify(
        claim=f"Given the reported impact '{_str_or_empty(basic.impact_metric)}' and duration '{_str_or_empty(basic.duration)}', the AT&T outage exceeded the 900,000 user-minutes threshold for wireless outages.",
        node=att_user_minutes,
        sources=None,
        additional_instruction="Apply rough magnitude reasoning using earlier verified facts and the NORS documentation.",
        extra_prerequisites=[att_blocked_calls_leaf, att_duration_leaf, att_nors_ref]
    )

    att_reportable = evaluator.add_leaf(
        id="ATT_Reportability_Conclusion",
        desc="Correctly concludes that outage was reportable under FCC NORS requirements",
        parent=assessment_seq,
        critical=True
    )
    await evaluator.verify(
        claim="The AT&T outage was reportable under FCC NORS requirements.",
        node=att_reportable,
        sources=None,
        additional_instruction="Use the determinations from the duration and user‑minutes threshold checks.",
        extra_prerequisites=[att_duration_threshold, att_user_minutes, att_nors_ref]
    )

    # 5) Timeline Compliance
    tml_node = evaluator.add_parallel(
        id="ATT_Timeline_Compliance",
        desc="Analysis of expected FCC reporting timeline compliance",
        parent=att_root,
        critical=True
    )
    gen_deadlines_node = evaluator.add_parallel(
        id="ATT_General_Reporting_Deadlines",
        desc="General NORS reporting deadlines",
        parent=tml_node,
        critical=True
    )
    emg_node = evaluator.add_parallel(
        id="ATT_Emergency_Notification_Requirements",
        desc="911-specific notification requirements",
        parent=tml_node,
        critical=True
    )
    tml_doc_node = evaluator.add_parallel(
        id="ATT_Timeline_Documentation",
        desc="Source documentation for timeline requirements",
        parent=tml_node,
        critical=True
    )
    tml_sources = _urls(tml.sources if tml else [])
    att_tml_ref = _add_ref_presence_node(
        evaluator,
        parent=tml_doc_node,
        node_id="ATT_Timeline_Reference",
        desc="Provides valid URL reference for FCC reporting timeline requirements (can be same as Verizon's)",
        urls=tml_sources,
        critical=True
    )

    await _verify_leaf_with_urls(
        evaluator,
        parent=gen_deadlines_node,
        node_id="ATT_Initial_Notification_Deadline",
        desc="Correctly identifies 120-minute initial notification deadline from outage discovery",
        claim=f"FCC requires an initial notification within {_str_or_empty(tml.initial_notification_minutes)} of outage discovery.",
        urls=tml_sources,
        critical=True,
        add_inst="Expected value: 120 minutes. Verify against the cited rule.",
        extra_prereq=[att_tml_ref]
    )
    await _verify_leaf_with_urls(
        evaluator,
        parent=gen_deadlines_node,
        node_id="ATT_Initial_Report_Deadline",
        desc="Correctly identifies 3-calendar-day initial report deadline",
        claim=f"FCC requires an initial outage report within {_str_or_empty(tml.initial_report_timeline)}.",
        urls=tml_sources,
        critical=True,
        add_inst="Expected value: 3 calendar days. Verify against the cited rule.",
        extra_prereq=[att_tml_ref]
    )
    await _verify_leaf_with_urls(
        evaluator,
        parent=gen_deadlines_node,
        node_id="ATT_Final_Report_Deadline",
        desc="Correctly identifies 30-day final report deadline",
        claim=f"FCC requires a final outage report within {_str_or_empty(tml.final_report_timeline)}.",
        urls=tml_sources,
        critical=True,
        add_inst="Expected value: 30 days. Verify against the cited rule.",
        extra_prereq=[att_tml_ref]
    )
    await _verify_leaf_with_urls(
        evaluator,
        parent=emg_node,
        node_id="ATT_911_Notification_Requirement",
        desc="Correctly identifies 30-minute 911 call center notification requirement for affected areas",
        claim=f"FCC requires notifying impacted 911 call centers within {_str_or_empty(tml.psap_notification_minutes)} when emergency services are affected.",
        urls=tml_sources,
        critical=True,
        add_inst="Expected value: 30 minutes. Verify against the cited rule.",
        extra_prereq=[att_tml_ref]
    )

    # 6) FCC Investigation (non-critical)
    inv_root = evaluator.add_parallel(
        id="ATT_FCC_Investigation",
        desc="Information about FCC investigation and findings",
        parent=att_root,
        critical=False
    )
    inv_details = evaluator.add_parallel(
        id="ATT_Investigation_Details",
        desc="Investigation status and findings",
        parent=inv_root,
        critical=False
    )
    inv_doc_node = evaluator.add_parallel(
        id="ATT_Investigation_Documentation",
        desc="Source documentation for FCC investigation",
        parent=inv_root,
        critical=True
    )
    inv_sources = _urls(inv.sources if inv else [])
    att_inv_ref = _add_ref_presence_node(
        evaluator,
        parent=inv_doc_node,
        node_id="ATT_Investigation_Reference",
        desc="Provides valid URL reference for AT&T FCC investigation information",
        urls=inv_sources,
        critical=True
    )

    await _verify_leaf_with_urls(
        evaluator,
        parent=inv_details,
        node_id="ATT_Investigation_Status",
        desc="Correctly reports that FCC conducted investigation",
        claim=f"The FCC investigation status for the AT&T outage is reported as: {_str_or_empty(inv.status)}.",
        urls=inv_sources,
        critical=False,
        add_inst="Verify that the FCC (or similarly authoritative source) confirms the investigation status.",
        extra_prereq=[att_inv_ref]
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
    Evaluate an answer for the U.S. carrier outage regulatory compliance assessment (Verizon 2026, AT&T 2024).
    """
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Root aggregates Verizon and AT&T in parallel
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

    # Root node for the entire task (set to non-critical to allow partial credit; JSON root "critical" adjusted to satisfy framework constraints)
    # Use existing root created by evaluator.initialize()

    # Extract structured info once
    extraction = await evaluator.extract(
        prompt=prompt_extract_outages(),
        template_class=OutagesExtraction,
        extraction_name="outages_extraction"
    )

    # Optional custom info to help debugging
    evaluator.add_custom_info(
        info={"required_cities": ["New York City", "Atlanta", "Charlotte", "Houston", "Washington D.C."]},
        info_type="rubric_hints",
        info_name="major_cities_requirement"
    )

    # Build Verizon and AT&T subtrees
    await build_verizon_tree(evaluator, root, extraction.verizon if extraction else None)
    await build_att_tree(evaluator, root, extraction.att if extraction else None)

    # Return final structured summary
    return evaluator.get_summary()