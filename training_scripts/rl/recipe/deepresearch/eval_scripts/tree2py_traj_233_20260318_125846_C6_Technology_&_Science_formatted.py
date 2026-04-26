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
TASK_ID = "major_us_carrier_outage_2024_2026_nors"
TASK_DESCRIPTION = """
Identify a major telecommunications outage by one of the three largest U.S. wireless carriers (Verizon, AT&T, or T-Mobile) that occurred between January 1, 2024, and March 18, 2026, and that met the FCC's Network Outage Reporting System (NORS) reportable thresholds of lasting at least 30 minutes and affecting at least 900,000 user-minutes. For the identified outage, provide comprehensive documentation including: (1) the carrier name and specific date of the outage, (2) the official root cause as disclosed by the carrier or FCC (e.g., software issue, configuration error, hardware failure), (3) the complete timeline with start time, resolution time, and total duration in hours, (4) quantifiable impact metrics such as number of affected users, blocked calls, or peak Downdetector reports, (5) confirmation that the outage affected voice and/or data services on a nationwide or multi-state scale, and (6) evidence that the carrier complied with FCC NORS reporting requirements by filing an initial report within 3 calendar days and a final report within 30 days. All information must be supported by reference URLs from official carrier statements, news reports, or FCC documentation.
"""

ALLOWED_CARRIERS = ["Verizon", "AT&T", "T-Mobile"]
DATE_RANGE_TEXT = "between January 1, 2024 and March 18, 2026 inclusive"


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class OutageExtraction(BaseModel):
    # Identity
    carrier: Optional[str] = None
    carrier_sources: List[str] = Field(default_factory=list)

    outage_date: Optional[str] = None  # Prefer a single specific calendar date like "2024-02-22"
    date_sources: List[str] = Field(default_factory=list)

    # Root cause
    root_cause: Optional[str] = None  # e.g., "software update/configuration issue"
    root_cause_sources: List[str] = Field(default_factory=list)

    # Timeline
    start_time: Optional[str] = None  # full timestamp if possible, e.g., "2024-02-22 03:30 ET"
    start_sources: List[str] = Field(default_factory=list)

    resolution_time: Optional[str] = None  # full timestamp if possible
    resolution_sources: List[str] = Field(default_factory=list)

    duration_hours: Optional[str] = None  # keep as string to allow ranges/approx like "~5"
    duration_sources: List[str] = Field(default_factory=list)

    # Impact
    impact_metric: Optional[str] = None  # e.g., "tens of millions affected", "peaked at 70k reports"
    impact_sources: List[str] = Field(default_factory=list)

    nors_threshold_evidence: Optional[str] = None  # statement/calculation indicating ≥900,000 user-minutes met
    nors_threshold_sources: List[str] = Field(default_factory=list)

    # Scope
    scope_statement: Optional[str] = None  # nationwide / multi-state; voice/data affected
    scope_sources: List[str] = Field(default_factory=list)

    # NORS compliance
    nors_initial_report_evidence: Optional[str] = None  # evidence initial filed within 3 days
    nors_initial_sources: List[str] = Field(default_factory=list)

    nors_final_report_evidence: Optional[str] = None  # evidence final filed within 30 days
    nors_final_sources: List[str] = Field(default_factory=list)

    # Official updates
    company_updates_evidence: Optional[str] = None  # official statements/updates existence
    company_updates_sources: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_outage() -> str:
    return f"""
    From the provided answer text, extract details for exactly ONE specific U.S. wireless carrier outage that the answer is using as the primary example.
    The chosen outage must be by Verizon, AT&T, or T-Mobile and must fall {DATE_RANGE_TEXT}.
    Return all fields below. If a field is not present in the answer, return null for that field or an empty list for URL arrays.
    IMPORTANT for URLs:
      - Only include explicit URLs that appear in the answer text.
      - Accept plain URLs or markdown links; extract the actual URL.
      - Do not invent or infer URLs. If none are present for a field, use an empty list.

    Required fields to extract:
    1) carrier: one of "Verizon", "AT&T", or "T-Mobile" exactly as written in the answer (minor variants acceptable, but normalize only if the answer explicitly shows it).
       carrier_sources: URLs that explicitly reference the carrier for the outage.
    2) outage_date: the specific calendar date of the outage (e.g., "2024-02-22" or "Feb 22, 2024").
       date_sources: URLs that explicitly reference the outage date.
    3) root_cause: an official statement of cause disclosed by the carrier or FCC (e.g., "software update/configuration error", "hardware failure").
       root_cause_sources: URLs supporting that cause.
    4) start_time: a specific timestamp for when the outage started, if given.
       start_sources: URLs supporting the start time.
    5) resolution_time: a specific timestamp for when the outage was resolved, if given.
       resolution_sources: URLs supporting the resolution time.
    6) duration_hours: the total duration of the outage in hours; can be approximate text (e.g., "~5 hours") if exact is not present.
       duration_sources: URLs supporting duration (or enabling derivation from start/resolution).
    7) impact_metric: at least one quantitative impact metric (e.g., affected users, blocked calls, peak Downdetector reports).
       impact_sources: URLs supporting the impact metric.
    8) nors_threshold_evidence: a statement or derivation that the outage met or exceeded the FCC NORS ≥900,000 user-minutes threshold (e.g., explicit mention of NORS threshold, or clear evidence to conclude it).
       nors_threshold_sources: URLs that directly support that threshold determination (can overlap with impact/timeline sources if they explicitly establish the threshold).
    9) scope_statement: statement confirming the outage affected voice and/or data services on a nationwide or multi-state scale.
       scope_sources: URLs supporting the scope statement.
    10) nors_initial_report_evidence: evidence that the carrier filed an initial NORS report within 3 calendar days (72 hours) of outage discovery.
        nors_initial_sources: URLs supporting this timing.
    11) nors_final_report_evidence: evidence that the carrier filed a final NORS report within 30 days of outage discovery.
        nors_final_sources: URLs supporting this timing.
    12) company_updates_evidence: evidence that the carrier issued official public statements/updates about the outage (e.g., newsroom posts, press releases, status page, official social media).
        company_updates_sources: URLs supporting that official updates occurred.

    Output as JSON strictly conforming to the fields specified in the schema.
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _normalize_urls(urls: Optional[List[str]]) -> List[str]:
    if not urls:
        return []
    cleaned = []
    for u in urls:
        if not u:
            continue
        u = u.strip()
        if not u:
            continue
        # Basic sanity: ensure protocol
        if not (u.startswith("http://") or u.startswith("https://")):
            u = "http://" + u
        if u not in cleaned:
            cleaned.append(u)
    return cleaned


def _nonempty_text(s: Optional[str]) -> bool:
    return bool(s and str(s).strip())


def _add_sources_presence_node(evaluator: Evaluator, parent, base_id: str, desc: str, sources: List[str]) -> None:
    evaluator.add_custom_node(
        result=bool(sources and len(sources) > 0),
        id=f"{base_id}_sources_present",
        desc=desc,
        parent=parent,
        critical=True,
    )


def _add_value_presence_node(evaluator: Evaluator, parent, base_id: str, desc: str, value_present: bool) -> None:
    evaluator.add_custom_node(
        result=value_present,
        id=f"{base_id}_value_present",
        desc=desc,
        parent=parent,
        critical=True,
    )


def _carrier_to_canonical(name: Optional[str]) -> Optional[str]:
    if not name:
        return None
    s = name.strip().lower()
    s = s.replace("&", "and").replace("-", "").replace(" ", "")
    if "verizon" in s:
        return "Verizon"
    if s in ("att", "atandt", "atandtmobility", "attmobility"):
        return "AT&T"
    if "tmobile" in s or s == "tmo":
        return "T-Mobile"
    # Try exact allowed names
    for c in ALLOWED_CARRIERS:
        if c.lower().replace("-", "").replace("&", "and").replace(" ", "") == s:
            return c
    return name  # fallback original


# --------------------------------------------------------------------------- #
# Verification builders                                                       #
# --------------------------------------------------------------------------- #
async def build_outage_identity_checks(evaluator: Evaluator, parent, data: OutageExtraction):
    node = evaluator.add_parallel(
        id="outage_identity",
        desc="Outage identity satisfies the carrier and date constraints.",
        parent=parent,
        critical=True,
    )

    # Carrier check
    carrier_group = evaluator.add_parallel(
        id="carrier_check",
        desc="Carrier name is provided, is allowed, and is supported by reference URL(s).",
        parent=node,
        critical=True,
    )
    carrier_sources = _normalize_urls(data.carrier_sources)
    carrier_value_present = _nonempty_text(data.carrier)
    canonical_carrier = _carrier_to_canonical(data.carrier)

    _add_value_presence_node(
        evaluator,
        carrier_group,
        "carrier",
        "Carrier value is provided in the answer.",
        carrier_value_present,
    )
    _add_sources_presence_node(
        evaluator,
        carrier_group,
        "carrier",
        "Carrier reference URL(s) are provided.",
        carrier_sources,
    )

    carrier_leaf = evaluator.add_leaf(
        id="carrier_provided_and_allowed",
        desc="Carrier name is provided, is one of Verizon/AT&T/T-Mobile, and is supported by reference URL(s).",
        parent=carrier_group,
        critical=True,
    )
    carrier_claim = f"The outage was experienced by {data.carrier}. This carrier is one of Verizon, AT&T, or T-Mobile."
    await evaluator.verify(
        claim=carrier_claim,
        node=carrier_leaf,
        sources=carrier_sources,
        additional_instruction="Confirm the carrier on the provided pages matches the named carrier. Also confirm it is one of Verizon, AT&T, or T-Mobile. Allow reasonable naming variants such as 'AT&T Mobility' or 'T-Mobile US'.",
    )

    # Date check
    date_group = evaluator.add_parallel(
        id="date_check",
        desc=f"Specific outage date is provided, falls {DATE_RANGE_TEXT}, and is supported by reference URL(s).",
        parent=node,
        critical=True,
    )
    date_sources = _normalize_urls(data.date_sources)
    date_value_present = _nonempty_text(data.outage_date)

    _add_value_presence_node(
        evaluator,
        date_group,
        "date",
        "Outage date value is provided in the answer.",
        date_value_present,
    )
    _add_sources_presence_node(
        evaluator,
        date_group,
        "date",
        "Outage date reference URL(s) are provided.",
        date_sources,
    )

    date_leaf = evaluator.add_leaf(
        id="date_provided_and_in_range",
        desc=f"Specific outage date is provided, falls between 2024-01-01 and 2026-03-18 inclusive, and is supported by reference URL(s).",
        parent=date_group,
        critical=True,
    )
    date_claim = f"The outage took place on {data.outage_date}, which falls between 2024-01-01 and 2026-03-18 inclusive."
    await evaluator.verify(
        claim=date_claim,
        node=date_leaf,
        sources=date_sources,
        additional_instruction="Verify that the referenced pages explicitly identify the outage date and that the date falls within the specified range (inclusive). Allow reasonable date format variations and local time indications.",
    )


async def build_root_cause_checks(evaluator: Evaluator, parent, data: OutageExtraction):
    # Official root cause
    cause_group = evaluator.add_parallel(
        id="official_root_cause_group",
        desc="Officially disclosed root cause (carrier or FCC disclosure) is provided and supported by reference URL(s).",
        parent=parent,
        critical=True,
    )
    cause_sources = _normalize_urls(data.root_cause_sources)
    cause_value_present = _nonempty_text(data.root_cause)

    _add_value_presence_node(
        evaluator,
        cause_group,
        "root_cause",
        "Root cause text is provided in the answer.",
        cause_value_present,
    )
    _add_sources_presence_node(
        evaluator,
        cause_group,
        "root_cause",
        "Root cause reference URL(s) are provided.",
        cause_sources,
    )

    cause_leaf = evaluator.add_leaf(
        id="official_root_cause",
        desc="Officially disclosed root cause (carrier or FCC disclosure) is provided and supported by reference URL(s).",
        parent=cause_group,
        critical=True,
    )
    cause_claim = f"The official root cause of this outage was: {data.root_cause}. The cause is explicitly disclosed by the carrier or the FCC."
    await evaluator.verify(
        claim=cause_claim,
        node=cause_leaf,
        sources=cause_sources,
        additional_instruction="Confirm that the stated cause is explicitly attributed by the carrier (official statement or newsroom) or by the FCC (orders, notices, reports).",
    )


async def build_timeline_checks(evaluator: Evaluator, parent, data: OutageExtraction):
    tl_node = evaluator.add_parallel(
        id="timeline",
        desc="Timeline details are provided and satisfy the NORS duration threshold.",
        parent=parent,
        critical=True,
    )

    # Start time
    start_group = evaluator.add_parallel(
        id="start_time_group",
        desc="Outage start time is provided as a specific timestamp and supported by reference URL(s).",
        parent=tl_node,
        critical=True,
    )
    start_sources = _normalize_urls(data.start_sources)
    start_value_present = _nonempty_text(data.start_time)

    _add_value_presence_node(
        evaluator,
        start_group,
        "start_time",
        "Start time value is provided in the answer.",
        start_value_present,
    )
    _add_sources_presence_node(
        evaluator,
        start_group,
        "start_time",
        "Start time reference URL(s) are provided.",
        start_sources,
    )

    start_leaf = evaluator.add_leaf(
        id="start_time_timestamp",
        desc="Outage start time is provided as a specific timestamp and supported by reference URL(s).",
        parent=start_group,
        critical=True,
    )
    start_claim = f"The outage started at approximately {data.start_time}."
    await evaluator.verify(
        claim=start_claim,
        node=start_leaf,
        sources=start_sources,
        additional_instruction="Verify the page(s) provide a specific start time (allow approximate phrasing like 'around' or timezone variants).",
    )

    # Resolution time
    end_group = evaluator.add_parallel(
        id="resolution_time_group",
        desc="Outage resolution time is provided as a specific timestamp and supported by reference URL(s).",
        parent=tl_node,
        critical=True,
    )
    resolution_sources = _normalize_urls(data.resolution_sources)
    resolution_value_present = _nonempty_text(data.resolution_time)

    _add_value_presence_node(
        evaluator,
        end_group,
        "resolution_time",
        "Resolution time value is provided in the answer.",
        resolution_value_present,
    )
    _add_sources_presence_node(
        evaluator,
        end_group,
        "resolution_time",
        "Resolution time reference URL(s) are provided.",
        resolution_sources,
    )

    resolution_leaf = evaluator.add_leaf(
        id="resolution_time_timestamp",
        desc="Outage resolution time is provided as a specific timestamp and supported by reference URL(s).",
        parent=end_group,
        critical=True,
    )
    resolution_claim = f"The outage was resolved by approximately {data.resolution_time}."
    await evaluator.verify(
        claim=resolution_claim,
        node=resolution_leaf,
        sources=resolution_sources,
        additional_instruction="Verify the page(s) provide a specific resolution/end time (allow approximate phrasing and timezone differences).",
    )

    # Duration in hours
    duration_group = evaluator.add_parallel(
        id="duration_group",
        desc="Total duration is provided in hours (or is derivable) and supported by reference URL(s).",
        parent=tl_node,
        critical=True,
    )
    duration_sources = _normalize_urls(data.duration_sources)
    duration_value_present = _nonempty_text(data.duration_hours)

    _add_value_presence_node(
        evaluator,
        duration_group,
        "duration",
        "Total duration value (in hours) is provided in the answer.",
        duration_value_present,
    )
    _add_sources_presence_node(
        evaluator,
        duration_group,
        "duration",
        "Duration reference URL(s) are provided.",
        duration_sources,
    )

    duration_leaf = evaluator.add_leaf(
        id="total_duration_in_hours",
        desc="Total duration is provided in hours (or is clearly derivable from sourced start/end timestamps) and supported by reference URL(s).",
        parent=duration_group,
        critical=True,
    )
    duration_claim = f"The total duration of the outage was about {data.duration_hours} hours."
    await evaluator.verify(
        claim=duration_claim,
        node=duration_leaf,
        sources=duration_sources if duration_sources else list(
            dict.fromkeys(start_sources + resolution_sources)  # fallback to timestamps if needed
        ),
        additional_instruction="Accept clearly stated hour counts or values derivable from sourced start/end times.",
    )

    # Duration meets 30-minute threshold
    threshold_group = evaluator.add_parallel(
        id="duration_threshold_group",
        desc="Evidence shows the outage lasted at least 30 minutes (NORS threshold), supported by reference URL(s).",
        parent=tl_node,
        critical=True,
    )
    combined_tl_sources = _normalize_urls(list(dict.fromkeys(start_sources + resolution_sources + duration_sources)))
    _add_sources_presence_node(
        evaluator,
        threshold_group,
        "duration_30m",
        "Reference URL(s) provided to assess 30-minute threshold.",
        combined_tl_sources,
    )

    threshold_leaf = evaluator.add_leaf(
        id="duration_meets_30_min_threshold",
        desc="Evidence shows the outage lasted at least 30 minutes (NORS threshold), supported by reference URL(s).",
        parent=threshold_group,
        critical=True,
    )
    threshold_claim = "The outage lasted at least 30 minutes (meeting or exceeding the FCC NORS duration threshold)."
    await evaluator.verify(
        claim=threshold_claim,
        node=threshold_leaf,
        sources=combined_tl_sources,
        additional_instruction="Confirm that the start/end times or stated duration indicate ≥30 minutes duration.",
    )


async def build_impact_checks(evaluator: Evaluator, parent, data: OutageExtraction):
    impact_node = evaluator.add_parallel(
        id="impact",
        desc="Quantifiable impact metrics are provided and the NORS ≥900,000 user-minutes threshold is supported by evidence.",
        parent=parent,
        critical=True,
    )

    # Quantifiable metric
    metric_group = evaluator.add_parallel(
        id="impact_metric_group",
        desc="At least one quantifiable impact metric is provided and supported by reference URL(s).",
        parent=impact_node,
        critical=True,
    )
    impact_sources = _normalize_urls(data.impact_sources)
    metric_value_present = _nonempty_text(data.impact_metric)

    _add_value_presence_node(
        evaluator,
        metric_group,
        "impact_metric",
        "At least one quantifiable impact metric value is provided in the answer.",
        metric_value_present,
    )
    _add_sources_presence_node(
        evaluator,
        metric_group,
        "impact_metric",
        "Impact metric reference URL(s) are provided.",
        impact_sources,
    )

    metric_leaf = evaluator.add_leaf(
        id="quantifiable_impact_metric",
        desc="At least one quantifiable impact metric (e.g., affected users, blocked calls, or peak Downdetector reports) is provided and supported by reference URL(s).",
        parent=metric_group,
        critical=True,
    )
    metric_claim = f"A quantifiable impact metric is reported for this outage: {data.impact_metric}."
    await evaluator.verify(
        claim=metric_claim,
        node=metric_leaf,
        sources=impact_sources,
        additional_instruction="Confirm that the page(s) provide a quantitative metric (e.g., users affected, calls blocked, peak Downdetector reports).",
    )

    # NORS user-minutes threshold ≥900,000
    nors_group = evaluator.add_parallel(
        id="nors_threshold_group",
        desc="Evidence shows the outage meets the ≥900,000 user-minutes NORS threshold.",
        parent=impact_node,
        critical=True,
    )
    nors_sources_pref = _normalize_urls(data.nors_threshold_sources)
    # Allow fallback to impact/timeline sources if the same pages explicitly establish threshold facts.
    timeline_sources_fallback = _normalize_urls(
        list(
            dict.fromkeys(
                (data.start_sources or [])
                + (data.resolution_sources or [])
                + (data.duration_sources or [])
                + (data.impact_sources or [])
            )
        )
    )
    nors_sources = nors_sources_pref if len(nors_sources_pref) > 0 else timeline_sources_fallback

    _add_value_presence_node(
        evaluator,
        nors_group,
        "nors_threshold",
        "NORS threshold evidence text/value is provided in the answer.",
        _nonempty_text(data.nors_threshold_evidence),
    )
    _add_sources_presence_node(
        evaluator,
        nors_group,
        "nors_threshold",
        "Reference URL(s) supporting the ≥900,000 user-minutes threshold determination are provided.",
        nors_sources,
    )

    nors_leaf = evaluator.add_leaf(
        id="nors_user_minutes_threshold_met",
        desc="Provides evidence that the outage meets the ≥900,000 user-minutes NORS threshold (or an explicitly stated equivalent establishing that threshold), supported by reference URL(s).",
        parent=nors_group,
        critical=True,
    )
    nors_claim = "The outage meets or exceeds the FCC NORS reporting threshold of 900,000 user-minutes."
    await evaluator.verify(
        claim=nors_claim,
        node=nors_leaf,
        sources=nors_sources,
        additional_instruction="Accept explicit statements that the outage was NORS-reportable at the ≥900,000 user-minutes threshold, or clear quantitative evidence that establishes ≥900,000 user-minutes.",
    )


async def build_scope_checks(evaluator: Evaluator, parent, data: OutageExtraction):
    scope_group = evaluator.add_parallel(
        id="scope_of_outage_group",
        desc="Confirms the outage affected voice and/or data services on a nationwide or multi-state scale, supported by reference URL(s).",
        parent=parent,
        critical=True,
    )
    scope_sources = _normalize_urls(data.scope_sources)
    scope_value_present = _nonempty_text(data.scope_statement)

    _add_value_presence_node(
        evaluator,
        scope_group,
        "scope",
        "Scope statement value is provided in the answer.",
        scope_value_present,
    )
    _add_sources_presence_node(
        evaluator,
        scope_group,
        "scope",
        "Scope statement reference URL(s) are provided.",
        scope_sources,
    )

    scope_leaf = evaluator.add_leaf(
        id="scope_of_outage",
        desc="Confirms the outage affected voice and/or data services on a nationwide or multi-state scale, supported by reference URL(s).",
        parent=scope_group,
        critical=True,
    )
    scope_claim = f"The outage affected voice and/or data services on a nationwide or multi-state scale. Evidence: {data.scope_statement}."
    await evaluator.verify(
        claim=scope_claim,
        node=scope_leaf,
        sources=scope_sources,
        additional_instruction="Confirm that the page(s) indicate nationwide or multi-state scope and that voice and/or data services were affected. Allow reasonable paraphrasing.",
    )


async def build_nors_compliance_checks(evaluator: Evaluator, parent, data: OutageExtraction):
    nors_node = evaluator.add_parallel(
        id="nors_reporting_compliance",
        desc="Evidence is provided that NORS reporting timing requirements were met.",
        parent=parent,
        critical=True,
    )

    # Initial report within 3 days
    initial_group = evaluator.add_parallel(
        id="nors_initial_group",
        desc="Evidence that an initial NORS report was filed within 3 calendar days (72 hours) of outage discovery.",
        parent=nors_node,
        critical=True,
    )
    initial_sources = _normalize_urls(data.nors_initial_sources)
    initial_value_present = _nonempty_text(data.nors_initial_report_evidence)

    _add_value_presence_node(
        evaluator,
        initial_group,
        "nors_initial",
        "Initial NORS report evidence text/value is provided in the answer.",
        initial_value_present,
    )
    _add_sources_presence_node(
        evaluator,
        initial_group,
        "nors_initial",
        "Initial NORS report timing reference URL(s) are provided.",
        initial_sources,
    )

    initial_leaf = evaluator.add_leaf(
        id="initial_report_within_3_days",
        desc="Evidence that an initial NORS report was filed within 3 calendar days (72 hours) of outage discovery, supported by reference URL(s).",
        parent=initial_group,
        critical=True,
    )
    initial_claim = "The carrier filed an initial NORS report within 3 calendar days (72 hours) of outage discovery."
    await evaluator.verify(
        claim=initial_claim,
        node=initial_leaf,
        sources=initial_sources,
        additional_instruction="Look for explicit statements or FCC/official documents indicating the initial NORS filing occurred within 3 calendar days.",
    )

    # Final report within 30 days
    final_group = evaluator.add_parallel(
        id="nors_final_group",
        desc="Evidence that a final NORS report was filed within 30 days of outage discovery.",
        parent=nors_node,
        critical=True,
    )
    final_sources = _normalize_urls(data.nors_final_sources)
    final_value_present = _nonempty_text(data.nors_final_report_evidence)

    _add_value_presence_node(
        evaluator,
        final_group,
        "nors_final",
        "Final NORS report evidence text/value is provided in the answer.",
        final_value_present,
    )
    _add_sources_presence_node(
        evaluator,
        final_group,
        "nors_final",
        "Final NORS report timing reference URL(s) are provided.",
        final_sources,
    )

    final_leaf = evaluator.add_leaf(
        id="final_report_within_30_days",
        desc="Evidence that a final NORS report was filed within 30 days of outage discovery, supported by reference URL(s).",
        parent=final_group,
        critical=True,
    )
    final_claim = "The carrier filed the final NORS report within 30 days of outage discovery."
    await evaluator.verify(
        claim=final_claim,
        node=final_leaf,
        sources=final_sources,
        additional_instruction="Look for explicit statements or FCC/official documents indicating the final NORS filing occurred within 30 days.",
    )


async def build_company_updates_checks(evaluator: Evaluator, parent, data: OutageExtraction):
    updates_group = evaluator.add_parallel(
        id="official_company_updates_group",
        desc="Evidence shows the company issued public statements/updates about the outage through official channels, supported by reference URL(s).",
        parent=parent,
        critical=True,
    )
    updates_sources = _normalize_urls(data.company_updates_sources)
    updates_value_present = _nonempty_text(data.company_updates_evidence)

    _add_value_presence_node(
        evaluator,
        updates_group,
        "company_updates",
        "Official company updates evidence text/value is provided in the answer.",
        updates_value_present,
    )
    _add_sources_presence_node(
        evaluator,
        updates_group,
        "company_updates",
        "Reference URL(s) for official company statements/updates are provided.",
        updates_sources,
    )

    updates_leaf = evaluator.add_leaf(
        id="official_company_updates",
        desc="Evidence shows the company issued public statements/updates about the outage through official channels, supported by reference URL(s).",
        parent=updates_group,
        critical=True,
    )
    updates_claim = "The carrier issued public statements or updates about the outage via official channels (e.g., newsroom, press releases, status page, official social media)."
    await evaluator.verify(
        claim=updates_claim,
        node=updates_leaf,
        sources=updates_sources,
        additional_instruction="Confirm that at least one official channel (company newsroom, press page, status page, or verified social account) contains statements/updates about the outage.",
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
    Evaluate an answer for the major U.S. carrier outage task using the Mind2Web2 framework.
    """
    # Initialize evaluator and root
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

    # Create a critical task node (since framework root is non-critical by design)
    task_node = evaluator.add_parallel(
        id="task_root",
        desc="Identify one qualifying major outage and provide all required documented details with supporting reference URL(s).",
        parent=root,
        critical=True,
    )

    # Extraction
    extracted: OutageExtraction = await evaluator.extract(
        prompt=prompt_extract_outage(),
        template_class=OutageExtraction,
        extraction_name="outage_extraction",
    )

    # Record minimal custom info for debug/summary
    evaluator.add_custom_info(
        {
            "carrier": extracted.carrier,
            "outage_date": extracted.outage_date,
            "root_cause": extracted.root_cause,
            "duration_hours": extracted.duration_hours,
            "impact_metric": extracted.impact_metric,
        },
        info_type="extracted_overview",
        info_name="extracted_overview",
    )

    # Build verification subtrees
    await build_outage_identity_checks(evaluator, task_node, extracted)
    await build_root_cause_checks(evaluator, task_node, extracted)
    await build_timeline_checks(evaluator, task_node, extracted)
    await build_impact_checks(evaluator, task_node, extracted)
    await build_scope_checks(evaluator, task_node, extracted)
    await build_nors_compliance_checks(evaluator, task_node, extracted)
    await build_company_updates_checks(evaluator, task_node, extracted)

    # Return summary
    return evaluator.get_summary()