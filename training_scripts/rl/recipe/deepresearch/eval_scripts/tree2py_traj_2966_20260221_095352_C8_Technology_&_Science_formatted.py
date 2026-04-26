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
TASK_ID = "verizon_outage_part4_20260114"
TASK_DESCRIPTION = (
    "On January 14, 2026, Verizon experienced a major nationwide wireless network outage that disrupted cellular and "
    "data services for customers across the United States. Conduct a comprehensive evaluation of this outage against "
    "FCC Part 4 reporting requirements (47 CFR Part 4). Your assessment must determine: (1) whether the outage met the "
    "mandatory reporting thresholds; (2) document all required outage characteristics including onset time, restoration "
    "time, root cause, and service impact scope; (3) verify Verizon's compliance with notification and reporting timeline "
    "requirements (120-minute notification, 72-hour Initial Report, 30-day Final Report); (4) assess whether 911 special "
    "facilities were affected and if so, whether special notification requirements were met (30-minute notification, "
    "2-hour follow-up, telephone and electronic methods); and (5) document Verizon's public communication, customer "
    "remediation measures, and any FCC regulatory follow-up actions. For each element of your assessment, provide "
    "authoritative source references (URLs) that document your findings."
)


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class TimeWithSources(BaseModel):
    time_text: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class DurationWithSources(BaseModel):
    duration_text: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class RootCauseInfo(BaseModel):
    description: Optional[str] = None
    category: Optional[str] = None  # e.g., software, configuration, hardware
    sources: List[str] = Field(default_factory=list)


class ImpactScope(BaseModel):
    services: List[str] = Field(default_factory=list)  # e.g., voice, SMS, data, LTE/5G
    geography: Optional[str] = None  # e.g., nationwide, states/regions
    sources: List[str] = Field(default_factory=list)


class UserImpactInfo(BaseModel):
    description: Optional[str] = None  # narrative summary
    figures: Optional[str] = None      # e.g., estimated customers/users affected
    user_minutes: Optional[str] = None # e.g., user-minutes if reported
    sources: List[str] = Field(default_factory=list)


class ReportabilityThresholds(BaseModel):
    threshold_met: Optional[bool] = None  # whether Part 4 thresholds are met
    duration_minutes: Optional[str] = None
    user_minutes_impacted: Optional[str] = None
    calculation_details: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class TimelineCompliance(BaseModel):
    triggered: Optional[bool] = None  # whether this timeline requirement is applicable
    discovery_time: Optional[str] = None
    submission_time: Optional[str] = None
    compliant: Optional[bool] = None
    sources: List[str] = Field(default_factory=list)


class SimpleCompliance(BaseModel):
    compliant: Optional[bool] = None
    sources: List[str] = Field(default_factory=list)


class PublicComm(BaseModel):
    summary: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class CustomerRemediation(BaseModel):
    description: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class RegulatoryFollowUp(BaseModel):
    description: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class SpecialFacility911(BaseModel):
    impacted: Optional[bool] = None
    impact_sources: List[str] = Field(default_factory=list)

    trigger_met: Optional[bool] = None
    trigger_sources: List[str] = Field(default_factory=list)

    notify_30min: TimelineCompliance = TimelineCompliance()
    follow_up_2hr: TimelineCompliance = TimelineCompliance()
    dual_method_notification: SimpleCompliance = SimpleCompliance()
    required_material_information_included: SimpleCompliance = SimpleCompliance()


class ChangeManagement(BaseModel):
    process_improvements_if_network_change_related: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class ComprehensiveOutageExtraction(BaseModel):
    event_date: Optional[str] = None

    # Required outage characteristics
    onset_time: Optional[TimeWithSources] = None
    restoration_time: Optional[TimeWithSources] = None
    total_duration: Optional[DurationWithSources] = None
    root_cause: Optional[RootCauseInfo] = None
    service_impact_scope: Optional[ImpactScope] = None
    user_impact: Optional[UserImpactInfo] = None

    # Thresholds
    reportability: Optional[ReportabilityThresholds] = None

    # Part 4 reporting & timeline compliance
    fcc_timeline_120min: Optional[TimelineCompliance] = None
    fcc_timeline_72hr: Optional[TimelineCompliance] = None
    fcc_timeline_30day: Optional[TimelineCompliance] = None
    electronic_template_submission: Optional[SimpleCompliance] = None
    final_report_attestation: Optional[SimpleCompliance] = None

    # Special facility (911) assessment
    special_911: Optional[SpecialFacility911] = None

    # Public/customer/regulatory actions
    public_communication: Optional[PublicComm] = None
    customer_remediation: Optional[CustomerRemediation] = None
    fcc_regulatory_follow_up: Optional[RegulatoryFollowUp] = None

    # Change management (conditional)
    change_management: Optional[ChangeManagement] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_outage_evaluation() -> str:
    return """
    Extract structured information about the January 14, 2026 Verizon outage and FCC Part 4 compliance from the answer.
    Follow these rules:
    - Return exactly the fields in the JSON schema.
    - For every factual element, extract the authoritative source URLs explicitly present in the answer text. If none are present, return an empty list.
    - Use free-form strings for times/dates/durations (include timezone if available).
    - For booleans like 'threshold_met' or 'compliant', infer only if the answer explicitly claims it; otherwise set null.

    You must fill this JSON structure (keys must match):
    {
      "event_date": str | null,

      "onset_time": { "time_text": str|null, "sources": [url...] },
      "restoration_time": { "time_text": str|null, "sources": [url...] },
      "total_duration": { "duration_text": str|null, "sources": [url...] },
      "root_cause": { "description": str|null, "category": str|null, "sources": [url...] },
      "service_impact_scope": { "services": [str...], "geography": str|null, "sources": [url...] },
      "user_impact": { "description": str|null, "figures": str|null, "user_minutes": str|null, "sources": [url...] },

      "reportability": {
        "threshold_met": bool|null,
        "duration_minutes": str|null,
        "user_minutes_impacted": str|null,
        "calculation_details": str|null,
        "sources": [url...]
      },

      "fcc_timeline_120min": {
        "triggered": bool|null,
        "discovery_time": str|null,
        "submission_time": str|null,
        "compliant": bool|null,
        "sources": [url...]
      },
      "fcc_timeline_72hr": {
        "triggered": bool|null,
        "discovery_time": str|null,
        "submission_time": str|null,
        "compliant": bool|null,
        "sources": [url...]
      },
      "fcc_timeline_30day": {
        "triggered": bool|null,
        "discovery_time": str|null,
        "submission_time": str|null,
        "compliant": bool|null,
        "sources": [url...]
      },
      "electronic_template_submission": { "compliant": bool|null, "sources": [url...] },
      "final_report_attestation": { "compliant": bool|null, "sources": [url...] },

      "special_911": {
        "impacted": bool|null,
        "impact_sources": [url...],
        "trigger_met": bool|null,
        "trigger_sources": [url...],
        "notify_30min": {
          "triggered": bool|null,
          "discovery_time": str|null,
          "submission_time": str|null,
          "compliant": bool|null,
          "sources": [url...]
        },
        "follow_up_2hr": {
          "triggered": bool|null,
          "discovery_time": str|null,
          "submission_time": str|null,
          "compliant": bool|null,
          "sources": [url...]
        },
        "dual_method_notification": { "compliant": bool|null, "sources": [url...] },
        "required_material_information_included": { "compliant": bool|null, "sources": [url...] }
      },

      "public_communication": { "summary": str|null, "sources": [url...] },
      "customer_remediation": { "description": str|null, "sources": [url...] },
      "fcc_regulatory_follow_up": { "description": str|null, "sources": [url...] },

      "change_management": { "process_improvements_if_network_change_related": str|null, "sources": [url...] }
    }

    Notes and cues to guide extraction:
    - Reporting thresholds for wireless (per 47 CFR Part 4): duration ≥ 30 minutes AND potential impact ≥ 900,000 user-minutes.
    - Timelines if reportable: 120-minute electronic notification, 72-hour Initial Report, 30-day Final Report.
    - Special 911 facility notifications (47 CFR § 4.9(h)): notify PSAPs within 30 minutes, follow-up within 2 hours, via telephone and electronic means, include required material info.
    - For 'sources', only include URLs that are present in the answer (plain or markdown). If a citation lacks a URL, leave sources empty.
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def has_sources(sources: Optional[List[str]]) -> bool:
    return bool(sources) and len([u for u in sources if isinstance(u, str) and u.strip()]) > 0


def list_to_str(items: Optional[List[str]]) -> str:
    if not items:
        return ""
    return ", ".join([str(x) for x in items if x is not None])


def safe_text(text: Optional[str]) -> str:
    return text or ""


# --------------------------------------------------------------------------- #
# Verification builders                                                       #
# --------------------------------------------------------------------------- #
async def build_reportability_thresholds(
    evaluator: Evaluator,
    parent_node,
    ext: ComprehensiveOutageExtraction,
) -> Dict[str, Any]:
    node = evaluator.add_parallel(
        id="reportability_thresholds",
        desc="Determine whether the outage met FCC Part 4 mandatory reporting thresholds, with supporting calculations and authoritative source URL(s)",
        parent=parent_node,
        critical=True,
    )

    # Existence check for sources
    sources_ok = has_sources(ext.reportability.sources if ext.reportability else [])
    evaluator.add_custom_node(
        result=sources_ok,
        id="part4_threshold_sources_present",
        desc="Authoritative sources (URLs) are provided for Part 4 threshold determination",
        parent=node,
        critical=True,
    )

    # Leaf: threshold determination
    leaf = evaluator.add_leaf(
        id="part4_threshold_determination",
        desc="Determine whether mandatory reporting thresholds are met per constraints (duration ≥30 minutes AND potential impact ≥900,000 user-minutes) with calculation and authoritative sources",
        parent=node,
        critical=True,
    )

    thr = ext.reportability or ReportabilityThresholds()
    met_text = "met" if thr.threshold_met else "did not meet" if thr.threshold_met is False else "unknown"
    calc_text = safe_text(thr.calculation_details)
    duration_text = safe_text(thr.duration_minutes)
    user_min_text = safe_text(thr.user_minutes_impacted)

    claim = (
        f"The outage {met_text} FCC Part 4 wireless reporting thresholds "
        f"(requires duration ≥ 30 minutes AND potential impact ≥ 900,000 user-minutes). "
        f"Duration (minutes): '{duration_text}'. User-minutes impacted: '{user_min_text}'. "
        f"Calculation/details: '{calc_text}'."
    )
    await evaluator.verify(
        claim=claim,
        node=leaf,
        sources=thr.sources,
        additional_instruction=(
            "Use the cited authoritative URLs to confirm whether both conditions are satisfied. "
            "Accept reasonable wording variations. If sources indicate the outage is not reportable, the claim should reflect 'did not meet'. "
            "Relevant references: 47 CFR Part 4 wireless thresholds (user-minutes)."
        ),
    )
    return {"threshold_leaf": leaf}


async def build_required_outage_characteristics(
    evaluator: Evaluator,
    parent_node,
    ext: ComprehensiveOutageExtraction,
):
    node = evaluator.add_parallel(
        id="required_outage_characteristics",
        desc="Document required outage characteristics for FCC reporting, each supported by authoritative source URL(s)",
        parent=parent_node,
        critical=True,
    )

    # Onset time
    onset_sources = (ext.onset_time.sources if ext.onset_time else [])
    evaluator.add_custom_node(
        result=has_sources(onset_sources),
        id="onset_time_sources_present",
        desc="Sources provided for outage onset time",
        parent=node,
        critical=True,
    )
    onset_leaf = evaluator.add_leaf(
        id="onset_time",
        desc="Provide outage onset date/time with authoritative source URL(s)",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"The outage onset time was '{safe_text(ext.onset_time.time_text if ext.onset_time else None)}'.",
        node=onset_leaf,
        sources=onset_sources,
        additional_instruction="Verify the onset timestamp from the cited URLs (accept textual time expressions and timezone variants).",
    )

    # Restoration time
    rest_sources = (ext.restoration_time.sources if ext.restoration_time else [])
    evaluator.add_custom_node(
        result=has_sources(rest_sources),
        id="restoration_time_sources_present",
        desc="Sources provided for outage restoration time",
        parent=node,
        critical=True,
    )
    restoration_leaf = evaluator.add_leaf(
        id="restoration_time",
        desc="Provide restoration date/time with authoritative source URL(s)",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"The outage restoration time was '{safe_text(ext.restoration_time.time_text if ext.restoration_time else None)}'.",
        node=restoration_leaf,
        sources=rest_sources,
        additional_instruction="Verify restoration timestamp from the cited URLs; reasonable variants are acceptable.",
    )

    # Total duration
    dur_sources = (ext.total_duration.sources if ext.total_duration else [])
    evaluator.add_custom_node(
        result=has_sources(dur_sources),
        id="total_duration_sources_present",
        desc="Sources provided for total outage duration",
        parent=node,
        critical=True,
    )
    duration_leaf = evaluator.add_leaf(
        id="total_duration",
        desc="Provide total outage duration (derived or explicitly stated), with authoritative source URL(s)",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"The total outage duration was '{safe_text(ext.total_duration.duration_text if ext.total_duration else None)}'.",
        node=duration_leaf,
        sources=dur_sources,
        additional_instruction="If duration is derived from onset and restoration, confirm consistency with those cited times; accept rounded values.",
    )

    # Root cause
    rc_sources = (ext.root_cause.sources if ext.root_cause else [])
    evaluator.add_custom_node(
        result=has_sources(rc_sources),
        id="root_cause_sources_present",
        desc="Sources provided for root cause",
        parent=node,
        critical=True,
    )
    root_leaf = evaluator.add_leaf(
        id="root_cause",
        desc="Provide the best-known root cause (and category if available), with authoritative source URL(s)",
        parent=node,
        critical=True,
    )
    rc_desc = safe_text(ext.root_cause.description if ext.root_cause else None)
    rc_cat = safe_text(ext.root_cause.category if ext.root_cause else None)
    claim_rc = f"The best-known root cause is '{rc_desc}' (category: '{rc_cat}')."
    await evaluator.verify(
        claim=claim_rc,
        node=root_leaf,
        sources=rc_sources,
        additional_instruction="Confirm the root cause statement and category (if present) from cited authority (e.g., official statements, FCC filings).",
    )

    # Service impact scope
    si_sources = (ext.service_impact_scope.sources if ext.service_impact_scope else [])
    evaluator.add_custom_node(
        result=has_sources(si_sources),
        id="service_impact_scope_sources_present",
        desc="Sources provided for service impact scope",
        parent=node,
        critical=True,
    )
    svc_leaf = evaluator.add_leaf(
        id="service_impact_scope",
        desc="Document service impact scope (services affected and geographic scope), with authoritative source URL(s)",
        parent=node,
        critical=True,
    )
    services_text = list_to_str(ext.service_impact_scope.services if ext.service_impact_scope else [])
    geo_text = safe_text(ext.service_impact_scope.geography if ext.service_impact_scope else None)
    claim_svc = f"Services affected: [{services_text}]. Geographic scope: '{geo_text}'."
    await evaluator.verify(
        claim=claim_svc,
        node=svc_leaf,
        sources=si_sources,
        additional_instruction="Verify services (e.g., voice/SMS/data/5G/LTE) and geographic coverage (e.g., nationwide or named regions) from cited sources.",
    )

    # User impact
    ui_sources = (ext.user_impact.sources if ext.user_impact else [])
    evaluator.add_custom_node(
        result=has_sources(ui_sources),
        id="user_impact_sources_present",
        desc="Sources provided for user impact figures",
        parent=node,
        critical=True,
    )
    user_leaf = evaluator.add_leaf(
        id="user_impact",
        desc="Document user impact (e.g., estimated users affected or user-minutes), with authoritative source URL(s)",
        parent=node,
        critical=True,
    )
    ui_desc = safe_text(ext.user_impact.description if ext.user_impact else None)
    ui_fig = safe_text(ext.user_impact.figures if ext.user_impact else None)
    ui_um = safe_text(ext.user_impact.user_minutes if ext.user_impact else None)
    claim_ui = f"User impact: '{ui_desc}'. Figures: '{ui_fig}'. User-minutes: '{ui_um}'."
    await evaluator.verify(
        claim=claim_ui,
        node=user_leaf,
        sources=ui_sources,
        additional_instruction="Confirm reported user impact/figures and any user-minutes statements from cited URLs.",
    )


async def build_fcc_reporting_and_timeline_compliance(
    evaluator: Evaluator,
    parent_node,
    ext: ComprehensiveOutageExtraction,
):
    node = evaluator.add_parallel(
        id="fcc_reporting_and_timeline_compliance",
        desc="Verify Verizon’s compliance with FCC Part 4 notification/reporting deadlines and filing mechanics; state 'not triggered' with citation if not reportable",
        parent=parent_node,
        critical=True,
    )

    # 120-minute notification
    n120_sources = (ext.fcc_timeline_120min.sources if ext.fcc_timeline_120min else [])
    evaluator.add_custom_node(
        result=has_sources(n120_sources),
        id="fcc_120min_notification_sources_present",
        desc="Sources provided for 120-minute notification timeliness or not-triggered citation",
        parent=node,
        critical=True,
    )
    n120_leaf = evaluator.add_leaf(
        id="fcc_120min_notification_timeliness",
        desc="120-minute electronic notification timeliness or not-triggered (with citation)",
        parent=node,
        critical=True,
    )
    n120 = ext.fcc_timeline_120min or TimelineCompliance()
    if n120.triggered is False:
        claim_n120 = "The FCC 120-minute electronic notification requirement was not triggered for this outage (with cited authority)."
    elif n120.compliant:
        claim_n120 = (
            f"Verizon submitted the FCC electronic notification within 120 minutes of discovery. "
            f"Discovery: '{safe_text(n120.discovery_time)}'; Submission: '{safe_text(n120.submission_time)}'."
        )
    else:
        claim_n120 = (
            f"Verizon did not meet the FCC 120-minute electronic notification timeline. "
            f"Discovery: '{safe_text(n120.discovery_time)}'; Submission: '{safe_text(n120.submission_time)}'."
        )
    await evaluator.verify(
        claim=claim_n120,
        node=n120_leaf,
        sources=n120_sources,
        additional_instruction=(
            "Confirm applicability and timing using cited URLs. If the outage is not reportable, verify the 'not triggered' claim with authoritative citation (e.g., CFR text or official determination)."
        ),
    )

    # 72-hour Initial Report
    n72_sources = (ext.fcc_timeline_72hr.sources if ext.fcc_timeline_72hr else [])
    evaluator.add_custom_node(
        result=has_sources(n72_sources),
        id="fcc_72hr_initial_report_sources_present",
        desc="Sources provided for 72-hour Initial Report timeliness or not-triggered citation",
        parent=node,
        critical=True,
    )
    n72_leaf = evaluator.add_leaf(
        id="fcc_72hr_initial_report_timeliness",
        desc="72-hour Initial Communications Outage Report timeliness or not-triggered (with citation)",
        parent=node,
        critical=True,
    )
    n72 = ext.fcc_timeline_72hr or TimelineCompliance()
    if n72.triggered is False:
        claim_n72 = "The FCC 72-hour Initial Communications Outage Report requirement was not triggered for this outage (with cited authority)."
    elif n72.compliant:
        claim_n72 = (
            f"The Initial Communications Outage Report was submitted within 72 hours of discovery. "
            f"Discovery: '{safe_text(n72.discovery_time)}'; Submission: '{safe_text(n72.submission_time)}'."
        )
    else:
        claim_n72 = (
            f"The Initial Communications Outage Report was not submitted within 72 hours of discovery. "
            f"Discovery: '{safe_text(n72.discovery_time)}'; Submission: '{safe_text(n72.submission_time)}'."
        )
    await evaluator.verify(
        claim=claim_n72,
        node=n72_leaf,
        sources=n72_sources,
        additional_instruction="Verify applicability and timing per the cited sources; accept reasonable time formats.",
    )

    # 30-day Final Report
    n30_sources = (ext.fcc_timeline_30day.sources if ext.fcc_timeline_30day else [])
    evaluator.add_custom_node(
        result=has_sources(n30_sources),
        id="fcc_30day_final_report_sources_present",
        desc="Sources provided for 30-day Final Report timeliness or not-triggered citation",
        parent=node,
        critical=True,
    )
    n30_leaf = evaluator.add_leaf(
        id="fcc_30day_final_report_timeliness",
        desc="30-day Final Communications Outage Report timeliness or not-triggered (with citation)",
        parent=node,
        critical=True,
    )
    n30 = ext.fcc_timeline_30day or TimelineCompliance()
    if n30.triggered is False:
        claim_n30 = "The FCC 30-day Final Communications Outage Report requirement was not triggered for this outage (with cited authority)."
    elif n30.compliant:
        claim_n30 = (
            f"The Final Communications Outage Report was submitted within 30 days of discovery. "
            f"Discovery: '{safe_text(n30.discovery_time)}'; Submission: '{safe_text(n30.submission_time)}'."
        )
    else:
        claim_n30 = (
            f"The Final Communications Outage Report was not submitted within 30 days of discovery. "
            f"Discovery: '{safe_text(n30.discovery_time)}'; Submission: '{safe_text(n30.submission_time)}'."
        )
    await evaluator.verify(
        claim=claim_n30,
        node=n30_leaf,
        sources=n30_sources,
        additional_instruction="Verify applicability and timing per cited sources (CFR and filings/acknowledgments).",
    )

    # Electronic template submission (47 CFR § 4.11)
    ets_sources = (ext.electronic_template_submission.sources if ext.electronic_template_submission else [])
    evaluator.add_custom_node(
        result=has_sources(ets_sources),
        id="electronic_template_submission_sources_present",
        desc="Sources provided for electronic template submission compliance or not-triggered citation",
        parent=node,
        critical=True,
    )
    ets_leaf = evaluator.add_leaf(
        id="electronic_template_submission_compliance",
        desc="Required FCC outage submissions filed electronically via Commission-approved templates, or not triggered (with citation)",
        parent=node,
        critical=True,
    )
    ets_comp = ext.electronic_template_submission.compliant if ext.electronic_template_submission else None
    if ets_comp is True:
        claim_ets = "Required FCC outage submissions/reports were filed electronically using Commission-approved web-based outage report templates."
    else:
        claim_ets = "Electronic template submission requirement was not applicable/not triggered for this outage, as supported by the cited sources."
    await evaluator.verify(
        claim=claim_ets,
        node=ets_leaf,
        sources=ets_sources,
        additional_instruction="Confirm compliance or non-applicability per 47 CFR § 4.11 and cited authoritative sources.",
    )

    # Final report attestation (47 CFR § 4.11)
    att_sources = (ext.final_report_attestation.sources if ext.final_report_attestation else [])
    evaluator.add_custom_node(
        result=has_sources(att_sources),
        id="final_report_attestation_sources_present",
        desc="Sources provided for final report attestation or not-triggered citation",
        parent=node,
        critical=True,
    )
    att_leaf = evaluator.add_leaf(
        id="final_report_attestation",
        desc="Final Report included required attestation by authorized person, or not triggered (with citation)",
        parent=node,
        critical=True,
    )
    att_comp = ext.final_report_attestation.compliant if ext.final_report_attestation else None
    if att_comp is True:
        claim_att = "The Final Communications Outage Report included the required attestation by an authorized person who can legally bind the provider."
    else:
        claim_att = "Final report attestation requirement was not applicable/not triggered for this outage, as supported by the cited sources."
    await evaluator.verify(
        claim=claim_att,
        node=att_leaf,
        sources=att_sources,
        additional_instruction="Confirm presence of required attestation under 47 CFR § 4.11 or verify not triggered with authoritative citation.",
    )


async def build_special_911_assessment(
    evaluator: Evaluator,
    parent_node,
    ext: ComprehensiveOutageExtraction,
):
    seq = evaluator.add_sequential(
        id="special_facility_911_assessment",
        desc="Assess 911 special facility impact and (if applicable) verify compliance with special notification requirements",
        parent=parent_node,
        critical=True,
    )

    sp = ext.special_911 or SpecialFacility911()

    # 911 impact determination
    impact_sources = sp.impact_sources
    evaluator.add_custom_node(
        result=has_sources(impact_sources),
        id="911_impact_sources_present",
        desc="Sources provided to determine whether 911 special facilities were affected",
        parent=seq,
        critical=True,
    )
    impact_leaf = evaluator.add_leaf(
        id="911_impact_determination",
        desc="Determine whether 911 special facilities were affected; provide authoritative sources",
        parent=seq,
        critical=True,
    )
    impact_text = "were affected" if sp.impacted else "were not affected" if sp.impacted is False else "impact unknown"
    await evaluator.verify(
        claim=f"911 special facilities {impact_text} by the outage.",
        node=impact_leaf,
        sources=impact_sources,
        additional_instruction="Confirm PSAP/911 center impact using cited authoritative sources; accept reasonable wording variations.",
    )

    # 911 special facility notification trigger
    trig_sources = sp.trigger_sources
    evaluator.add_custom_node(
        result=has_sources(trig_sources),
        id="911_trigger_sources_present",
        desc="Sources provided to determine whether 911 special-facility notification requirements are triggered",
        parent=seq,
        critical=True,
    )
    trig_leaf = evaluator.add_leaf(
        id="911_special_facility_notification_trigger",
        desc="Determine whether special-facility notification requirements are triggered under constraints; support with authoritative sources",
        parent=seq,
        critical=True,
    )
    triggered_text = "are triggered" if sp.trigger_met else "are not triggered" if sp.trigger_met is False else "trigger unknown"
    await evaluator.verify(
        claim=(
            f"The 911 special-facility notification requirements {triggered_text} for this outage under "
            f"the constraints (affected 911 facility AND duration ≥ 30 minutes AND ≥ 900,000 user-minutes)."
        ),
        node=trig_leaf,
        sources=trig_sources,
        additional_instruction=(
            "Use cited sources to confirm applicability under 47 CFR § 4.9(h). If not triggered, the sources should explicitly support non-applicability."
        ),
    )

    # If triggered: verify each notification obligation (parallel)
    notif = evaluator.add_parallel(
        id="911_special_facility_notification_requirements_if_triggered",
        desc="If triggered: verify each required 911 notification obligation; else skip automatically",
        parent=seq,
        critical=True,
    )

    # 30-minute initial notification
    n30_sources = (sp.notify_30min.sources if sp.notify_30min else [])
    evaluator.add_custom_node(
        result=has_sources(n30_sources),
        id="notify_within_30_minutes_sources_present",
        desc="Sources provided for 30-minute 911 notification compliance",
        parent=notif,
        critical=True,
    )
    n30_leaf = evaluator.add_leaf(
        id="notify_within_30_minutes",
        desc="Verify 911 special facility notified no later than 30 minutes after discovery",
        parent=notif,
        critical=True,
    )
    n30_comp = sp.notify_30min.compliant if sp.notify_30min else None
    if n30_comp is True:
        n30_claim = (
            f"The affected 911 special facility was notified within 30 minutes of discovery. "
            f"Discovery: '{safe_text(sp.notify_30min.discovery_time)}'; Contact: '{safe_text(sp.notify_30min.submission_time)}'."
        )
    else:
        n30_claim = (
            f"The 30-minute notification requirement was not met. "
            f"Discovery: '{safe_text(sp.notify_30min.discovery_time)}'; Contact: '{safe_text(sp.notify_30min.submission_time)}'."
        )
    await evaluator.verify(
        claim=n30_claim,
        node=n30_leaf,
        sources=n30_sources,
        additional_instruction="Verify timing using cited sources (47 CFR § 4.9(h)(4)).",
    )

    # 2-hour follow-up
    f2_sources = (sp.follow_up_2hr.sources if sp.follow_up_2hr else [])
    evaluator.add_custom_node(
        result=has_sources(f2_sources),
        id="follow_up_within_2_hours_sources_present",
        desc="Sources provided for 2-hour follow-up notification compliance",
        parent=notif,
        critical=True,
    )
    f2_leaf = evaluator.add_leaf(
        id="follow_up_within_2_hours",
        desc="Verify first follow-up notification sent within 2 hours after initial contact",
        parent=notif,
        critical=True,
    )
    f2_comp = sp.follow_up_2hr.compliant if sp.follow_up_2hr else None
    if f2_comp is True:
        f2_claim = (
            f"The first follow-up notification was sent within 2 hours after initial 911 contact. "
            f"Initial Contact: '{safe_text(sp.follow_up_2hr.discovery_time)}'; Follow-up: '{safe_text(sp.follow_up_2hr.submission_time)}'."
        )
    else:
        f2_claim = (
            f"The 2-hour follow-up notification requirement was not met. "
            f"Initial Contact: '{safe_text(sp.follow_up_2hr.discovery_time)}'; Follow-up: '{safe_text(sp.follow_up_2hr.submission_time)}'."
        )
    await evaluator.verify(
        claim=f2_claim,
        node=f2_leaf,
        sources=f2_sources,
        additional_instruction="Verify timestamps per 47 CFR § 4.9(h)(5).",
    )

    # Dual method notification
    dm_sources = (sp.dual_method_notification.sources if sp.dual_method_notification else [])
    evaluator.add_custom_node(
        result=has_sources(dm_sources),
        id="dual_method_notification_sources_present",
        desc="Sources provided for dual method (telephone and electronic) notification compliance",
        parent=notif,
        critical=True,
    )
    dm_leaf = evaluator.add_leaf(
        id="dual_method_notification",
        desc="Verify notifications were sent via both telephone and electronic means",
        parent=notif,
        critical=True,
    )
    dm_comp = sp.dual_method_notification.compliant if sp.dual_method_notification else None
    dm_claim = (
        "Notifications to affected 911 special facilities were transmitted by both telephone and in writing via electronic means."
        if dm_comp is True
        else "Notifications did not use both telephone and electronic methods as required."
    )
    await evaluator.verify(
        claim=dm_claim,
        node=dm_leaf,
        sources=dm_sources,
        additional_instruction="Confirm dual-method delivery per 47 CFR § 4.9(h)(3).",
    )

    # Required material information included
    mi_sources = (sp.required_material_information_included.sources if sp.required_material_information_included else [])
    evaluator.add_custom_node(
        result=has_sources(mi_sources),
        id="required_material_information_included_sources_present",
        desc="Sources provided showing required material information elements included",
        parent=notif,
        critical=True,
    )
    mi_leaf = evaluator.add_leaf(
        id="required_material_information_included",
        desc="Verify 911 notifications included all required material information elements",
        parent=notif,
        critical=True,
    )
    mi_comp = sp.required_material_information_included.compliant if sp.required_material_information_included else None
    mi_claim = (
        "The 911 notification(s) included the material information elements required by 47 CFR § 4.9(h)(2)."
        if mi_comp is True
        else "The 911 notification(s) did not include all material information elements required by 47 CFR § 4.9(h)(2)."
    )
    await evaluator.verify(
        claim=mi_claim,
        node=mi_leaf,
        sources=mi_sources,
        additional_instruction="Confirm inclusion of required elements (e.g., cause, scope, time estimates) as per § 4.9(h)(2).",
    )


async def build_optional_actions(
    evaluator: Evaluator,
    parent_node,
    ext: ComprehensiveOutageExtraction,
):
    opt = evaluator.add_parallel(
        id="public_customer_and_regulatory_actions",
        desc="Document public communication, customer remediation, and FCC regulatory follow-up actions (partial credit allowed)",
        parent=parent_node,
        critical=False,
    )

    # Public communication
    pub_sources = (ext.public_communication.sources if ext.public_communication else [])
    evaluator.add_custom_node(
        result=has_sources(pub_sources),
        id="public_communication_sources_present",
        desc="Sources provided for Verizon public communications",
        parent=opt,
        critical=True,
    )
    pub_leaf = evaluator.add_leaf(
        id="public_communication",
        desc="Verizon public communications about the outage and restoration, with sources",
        parent=opt,
        critical=False,
    )
    await evaluator.verify(
        claim=f"Verizon issued public communications regarding the outage and restoration: '{safe_text(ext.public_communication.summary if ext.public_communication else None)}'.",
        node=pub_leaf,
        sources=pub_sources,
        additional_instruction="Confirm public statements/posts/press releases with cited URLs.",
    )

    # Customer remediation
    rem_sources = (ext.customer_remediation.sources if ext.customer_remediation else [])
    evaluator.add_custom_node(
        result=has_sources(rem_sources),
        id="customer_remediation_sources_present",
        desc="Sources provided for customer remediation measures",
        parent=opt,
        critical=True,
    )
    rem_leaf = evaluator.add_leaf(
        id="customer_remediation",
        desc="Customer remediation measures (credits/compensation/other), with sources",
        parent=opt,
        critical=False,
    )
    await evaluator.verify(
        claim=f"Verizon provided customer remediation measures: '{safe_text(ext.customer_remediation.description if ext.customer_remediation else None)}'.",
        node=rem_leaf,
        sources=rem_sources,
        additional_instruction="Confirm any credits/compensation or other customer measures with cited sources.",
    )

    # FCC regulatory follow-up
    reg_sources = (ext.fcc_regulatory_follow_up.sources if ext.fcc_regulatory_follow_up else [])
    evaluator.add_custom_node(
        result=has_sources(reg_sources),
        id="fcc_regulatory_follow_up_sources_present",
        desc="Sources provided for FCC regulatory follow-up actions",
        parent=opt,
        critical=True,
    )
    reg_leaf = evaluator.add_leaf(
        id="fcc_regulatory_follow_up",
        desc="FCC regulatory follow-up (inquiries, investigations, enforcement), with sources",
        parent=opt,
        critical=False,
    )
    await evaluator.verify(
        claim=f"FCC regulatory follow-up actions: '{safe_text(ext.fcc_regulatory_follow_up.description if ext.fcc_regulatory_follow_up else None)}'.",
        node=reg_leaf,
        sources=reg_sources,
        additional_instruction="Confirm any FCC inquiry/investigation/enforcement items with cited sources.",
    )

    # Conditional change-management/prevention (if applicable)
    cm_node = evaluator.add_parallel(
        id="change_management_and_prevention_if_applicable",
        desc="If outage caused by software/configuration/network change, document process improvements (partial credit allowed)",
        parent=parent_node,
        critical=False,
    )
    # Trigger based on root cause category
    rc_cat = safe_text(ext.root_cause.category if ext.root_cause else None).lower()
    trigger = any(k in rc_cat for k in ["software", "config", "configuration", "network change", "upgrade"])
    trigger_node = evaluator.add_custom_node(
        result=trigger,
        id="change_mgmt_trigger",
        desc="Trigger for change-management documentation (root cause indicates software/configuration/network change)",
        parent=cm_node,
        critical=False,
    )

    cm_sources = (ext.change_management.sources if ext.change_management else [])
    cm_leaf = evaluator.add_leaf(
        id="process_improvements_if_network_change_related",
        desc="Document change-management procedures and process improvements to prevent recurrence (if applicable), with sources",
        parent=cm_node,
        critical=False,
    )
    await evaluator.verify(
        claim=f"Process improvements/change-management (if applicable): '{safe_text(ext.change_management.process_improvements_if_network_change_related if ext.change_management else None)}'.",
        node=cm_leaf,
        sources=cm_sources,
        additional_instruction=(
            "Verify stated procedures/process improvements with cited URLs. "
            "This verification is applicable only if the trigger condition is true; otherwise it should be skipped."
        ),
        extra_prerequisites=[trigger_node],  # Skip if trigger failed
    )


# --------------------------------------------------------------------------- #
# Main evaluation function                                                    #
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
    Evaluate an answer for the Verizon January 14, 2026 outage FCC Part 4 compliance task.
    """
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Root stays non-critical; core subtree will be marked critical
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

    # Extract comprehensive structured info from the answer
    ext: ComprehensiveOutageExtraction = await evaluator.extract(
        prompt=prompt_extract_outage_evaluation(),
        template_class=ComprehensiveOutageExtraction,
        extraction_name="outage_evaluation_extraction",
    )

    # Build core FCC Part 4 compliance subtree (critical)
    core = evaluator.add_parallel(
        id="fcc_part4_core",
        desc="Core FCC Part 4 compliance evaluation for the January 14, 2026 Verizon outage",
        parent=root,
        critical=True,
    )

    # 1) Reportability thresholds
    thr_info = await build_reportability_thresholds(evaluator, core, ext)
    threshold_leaf = thr_info.get("threshold_leaf")

    # 2) Required outage characteristics
    await build_required_outage_characteristics(evaluator, core, ext)

    # 3) FCC reporting timeline & submission compliance
    await build_fcc_reporting_and_timeline_compliance(evaluator, core, ext)

    # 4) Special 911 assessment
    await build_special_911_assessment(evaluator, core, ext)

    # 5) Optional actions (public/customer/regulatory + change mgmt)
    await build_optional_actions(evaluator, root, ext)

    # Add contextual info about CFR references to the summary
    evaluator.add_custom_info(
        info={
            "references": [
                "47 CFR Part 4 (Communications Outage Reporting)",
                "47 CFR § 4.9(h) (911 special facility notifications)",
                "47 CFR § 4.11 (Electronic filing; template and attestation requirements)"
            ],
            "event_date": ext.event_date or "2026-01-14",
        },
        info_type="context",
        info_name="regulatory_context",
    )

    # Return summary
    return evaluator.get_summary()