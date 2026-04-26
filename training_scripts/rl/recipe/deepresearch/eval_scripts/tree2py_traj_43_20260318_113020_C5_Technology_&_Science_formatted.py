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
TASK_ID = "outage_tracking_comparison_2026Q1"
TASK_DESCRIPTION = """Research and compare three real-time outage tracking services—Downdetector, GeoBlackout, and Outage.Report—by identifying their key operational features. Additionally, document one recent major cellular carrier outage in the United States that occurred in 2024 or later.

For each outage tracking service, provide the following information based on official sources:

For Downdetector:
1. How frequently does Downdetector update the status of monitored services?
2. What time period does Downdetector use to calculate the baseline for determining if reports are significantly elevated?
3. How many distinct status levels does Downdetector display for services?
4. How many services does Downdetector monitor globally?

For GeoBlackout:
1. How frequently does GeoBlackout update its outage map?
2. What level of geographic precision does GeoBlackout provide for outage locations?
3. How many major US network operators does GeoBlackout monitor for internet outages?
4. What is the magnitude of user reports GeoBlackout processes annually (provide the threshold mentioned on their site)?

For Outage.Report:
1. What timing information does Outage.Report display for each incident?
2. What geographic information does Outage.Report show for affected areas during incidents?
3. What is Outage.Report's geographic coverage scope?

For one recent major US carrier outage (must be from 2024 or later):
1. Which major cellular carrier experienced the outage?
2. On what specific date did the outage occur?
3. How long did the outage last?
4. What quantitative impact metrics were reported (e.g., number of customers affected, number of calls blocked)?
5. Provide a reference URL from a reliable source documenting this outage.

All information must be verifiable through the official websites of these services or credible news sources.
"""

# Expected facts per rubric (used to frame verification claims)
EXP_DD_UPDATE_FREQ = "every 4 minutes"
EXP_DD_BASELINE_PERIOD = "6 months"
EXP_DD_STATUS_LEVELS_DESC = 'three distinct status levels: "No problems", "Possible problems", and "Problems"'
EXP_DD_SERVICE_COUNT_THRESHOLD = "over 31,000"

EXP_GB_UPDATE_FREQ = "every minute"
EXP_GB_PRECISION = "address-level precision"
EXP_GB_US_OPERATORS_COUNT = "four"
EXP_GB_ANNUAL_REPORTS_MAG = "over 10 million"

EXP_OR_TIMING_INFO = "start time, end time, and total duration"
EXP_OR_GEOGRAPHY_INFO = "which countries were affected"
EXP_OR_COVERAGE_SCOPE = "global coverage"

EXP_OUTAGE_CARRIER = "Verizon"
EXP_OUTAGE_DATE = "January 14, 2026"
EXP_OUTAGE_DURATION = "over 10 hours"
EXP_OUTAGE_IMPACT = "more than 1.5 million customers affected"


# --------------------------------------------------------------------------- #
# Data models for structured extraction                                       #
# --------------------------------------------------------------------------- #
class DowndetectorInfo(BaseModel):
    update_frequency: Optional[str] = None
    baseline_period: Optional[str] = None
    status_levels: Optional[str] = None
    service_count: Optional[str] = None
    source_urls: List[str] = Field(default_factory=list)


class GeoBlackoutInfo(BaseModel):
    update_frequency: Optional[str] = None
    precision_level: Optional[str] = None
    us_operators_monitored: Optional[str] = None
    annual_reports_magnitude: Optional[str] = None
    source_urls: List[str] = Field(default_factory=list)


class OutageReportInfo(BaseModel):
    timing_info: Optional[str] = None
    geography_info: Optional[str] = None
    coverage_scope: Optional[str] = None
    source_urls: List[str] = Field(default_factory=list)


class CarrierOutageInfo(BaseModel):
    carrier: Optional[str] = None
    date: Optional[str] = None
    duration: Optional[str] = None
    impact_metrics: Optional[str] = None
    source_urls: List[str] = Field(default_factory=list)


class OutageTrackingExtraction(BaseModel):
    downdetector: Optional[DowndetectorInfo] = None
    geoblackout: Optional[GeoBlackoutInfo] = None
    outage_report: Optional[OutageReportInfo] = None
    major_outage: Optional[CarrierOutageInfo] = None


# --------------------------------------------------------------------------- #
# Extraction prompt helpers                                                   #
# --------------------------------------------------------------------------- #
def prompt_extract_all() -> str:
    return """
    Extract structured information that the answer explicitly provides about three outage-tracking services and one specific major US carrier outage. Do not infer or add anything not stated in the answer text.

    For each section below, extract fields exactly as they appear in the answer. If a field is not present in the answer, set it to null (or an empty array for URL lists).

    1) Downdetector (object 'downdetector'):
       - update_frequency: how frequently status is updated (verbatim from answer)
       - baseline_period: the time period used for baseline comparisons
       - status_levels: the number and/or names of distinct status levels shown (verbatim)
       - service_count: the number of services monitored globally as stated (e.g., "over 31,000")
       - source_urls: an array of URLs the answer cites for these Downdetector facts; include only real URLs present in the answer

    2) GeoBlackout (object 'geoblackout'):
       - update_frequency: how frequently the outage map updates
       - precision_level: level of geographic precision stated
       - us_operators_monitored: how many major US network operators for internet outages
       - annual_reports_magnitude: magnitude/threshold of user reports processed annually
       - source_urls: array of URLs the answer cites for these GeoBlackout facts

    3) Outage.Report (object 'outage_report'):
       - timing_info: what timing info is shown per incident (e.g., start/end/duration)
       - geography_info: what geographic info is shown for affected areas
       - coverage_scope: geographic coverage scope stated
       - source_urls: array of URLs the answer cites for these Outage.Report facts

    4) One major US carrier outage (object 'major_outage'):
       - carrier: the named carrier
       - date: the specific date given (verbatim, e.g., "January 14, 2026")
       - duration: how long the outage lasted (verbatim, e.g., "over 10 hours")
       - impact_metrics: the quantitative impact metrics stated (verbatim)
       - source_urls: array of URLs the answer cites (credible news or authoritative sources) documenting this outage

    Notes:
    - URLs must be explicitly present in the answer. Include only valid URLs. If none are given, return an empty array.
    - Preserve the answer’s exact wording for fields where applicable.
    """


# --------------------------------------------------------------------------- #
# Utility helpers                                                             #
# --------------------------------------------------------------------------- #
def _safe(val: Optional[str]) -> str:
    return val if (val is not None and isinstance(val, str)) else "MISSING"


def _srcs(srcs: Optional[List[str]]) -> List[str]:
    return srcs or []


# --------------------------------------------------------------------------- #
# Verification subtree builders                                               #
# --------------------------------------------------------------------------- #
async def build_downdetector_checks(evaluator: Evaluator, parent, dd: Optional[DowndetectorInfo]) -> None:
    node = evaluator.add_parallel(
        id="downdetector_service",
        desc="Downdetector: provide the requested operational features matching the stated constraints",
        parent=parent,
        critical=True
    )

    dd_sources = _srcs(dd.source_urls if dd else None)

    # Update frequency: every 4 minutes
    leaf = evaluator.add_leaf(
        id="downdetector_update_frequency",
        desc="States that Downdetector updates service status every 4 minutes",
        parent=node,
        critical=True
    )
    claim = "Downdetector updates monitored service status every 4 minutes."
    add_ins = (
        f"Judge Correct only if BOTH are true:\n"
        f"(A) The answer explicitly states this same value. The answer’s extracted value was: '{_safe(dd.update_frequency if dd else None)}'. "
        f"If the answer omits this detail or states a different value, mark Incorrect.\n"
        f"(B) At least one cited, official/authoritative Downdetector page explicitly supports the 'every 4 minutes' frequency "
        f"(allow minor wording variants like 'every four minutes'). If no working URL is provided or the page does not support it, mark Incorrect."
    )
    await evaluator.verify(claim=claim, node=leaf, sources=dd_sources, additional_instruction=add_ins)

    # Baseline period: 6 months
    leaf = evaluator.add_leaf(
        id="downdetector_baseline_period",
        desc="States that Downdetector uses a 6-month historical baseline for comparison",
        parent=node,
        critical=True
    )
    claim = "Downdetector uses a 6-month historical baseline to determine when reports are significantly elevated."
    add_ins = (
        f"Judge Correct only if BOTH are true:\n"
        f"(A) The answer explicitly states a 6-month baseline; extracted: '{_safe(dd.baseline_period if dd else None)}'.\n"
        f"(B) A cited official page supports the 6-month baseline. If missing/contradicted, mark Incorrect."
    )
    await evaluator.verify(claim=claim, node=leaf, sources=dd_sources, additional_instruction=add_ins)

    # Status levels: three distinct levels
    leaf = evaluator.add_leaf(
        id="downdetector_status_levels",
        desc='States that Downdetector displays three distinct status levels (No problems / Possible problems / Problems)',
        parent=node,
        critical=True
    )
    claim = 'Downdetector displays three distinct status levels for services: "No problems", "Possible problems", and "Problems".'
    add_ins = (
        f"Judge Correct only if BOTH are true:\n"
        f"(A) The answer explicitly states three distinct levels and/or these names; extracted: '{_safe(dd.status_levels if dd else None)}'.\n"
        f"(B) A cited official source page supports these three levels. If no URL or mismatch, mark Incorrect."
    )
    await evaluator.verify(claim=claim, node=leaf, sources=dd_sources, additional_instruction=add_ins)

    # Service count: over 31,000 services globally
    leaf = evaluator.add_leaf(
        id="downdetector_service_count",
        desc="States that Downdetector monitors over 31,000 services globally",
        parent=node,
        critical=True
    )
    claim = "Downdetector monitors over 31,000 services globally."
    add_ins = (
        f"Judge Correct only if BOTH are true:\n"
        f"(A) The answer claims a threshold at least 'over 31,000'; extracted: '{_safe(dd.service_count if dd else None)}'.\n"
        f"(B) A cited official/authoritative Downdetector page supports a figure that is ≥ 31,000. If none supports it or no URL, mark Incorrect."
    )
    await evaluator.verify(claim=claim, node=leaf, sources=dd_sources, additional_instruction=add_ins)


async def build_geoblackout_checks(evaluator: Evaluator, parent, gb: Optional[GeoBlackoutInfo]) -> None:
    node = evaluator.add_parallel(
        id="geoblackout_service",
        desc="GeoBlackout: provide the requested operational features matching the stated constraints",
        parent=parent,
        critical=True
    )

    gb_sources = _srcs(gb.source_urls if gb else None)

    # Update frequency: every minute
    leaf = evaluator.add_leaf(
        id="geoblackout_update_frequency",
        desc="States that GeoBlackout updates its outage map every minute",
        parent=node,
        critical=True
    )
    claim = "GeoBlackout updates its outage map every minute."
    add_ins = (
        f"Judge Correct only if BOTH are true:\n"
        f"(A) The answer states an equivalent 'every minute' frequency; extracted: '{_safe(gb.update_frequency if gb else None)}'.\n"
        f"(B) A cited official GeoBlackout page supports this. If not, mark Incorrect."
    )
    await evaluator.verify(claim=claim, node=leaf, sources=gb_sources, additional_instruction=add_ins)

    # Precision: address-level precision
    leaf = evaluator.add_leaf(
        id="geoblackout_precision",
        desc="States that GeoBlackout provides address-level precision for outage locations",
        parent=node,
        critical=True
    )
    claim = "GeoBlackout provides address-level precision for outage locations."
    add_ins = (
        f"Judge Correct only if BOTH are true:\n"
        f"(A) The answer explicitly claims address-level precision; extracted: '{_safe(gb.precision_level if gb else None)}'.\n"
        f"(B) A cited official page supports this claim. Otherwise, Incorrect."
    )
    await evaluator.verify(claim=claim, node=leaf, sources=gb_sources, additional_instruction=add_ins)

    # US operators monitored: four
    leaf = evaluator.add_leaf(
        id="geoblackout_us_operators",
        desc="States that GeoBlackout monitors four major US network operators for internet outages",
        parent=node,
        critical=True
    )
    claim = "GeoBlackout monitors four major US network operators for internet outages."
    add_ins = (
        f"Judge Correct only if BOTH are true:\n"
        f"(A) The answer states 'four' (or equivalent) for major US network operators; extracted: '{_safe(gb.us_operators_monitored if gb else None)}'.\n"
        f"(B) A cited official page supports monitoring four major US operators. Otherwise, Incorrect."
    )
    await evaluator.verify(claim=claim, node=leaf, sources=gb_sources, additional_instruction=add_ins)

    # Annual reports magnitude: over 10 million
    leaf = evaluator.add_leaf(
        id="geoblackout_annual_reports",
        desc="States that GeoBlackout processes over 10 million user reports annually",
        parent=node,
        critical=True
    )
    claim = "GeoBlackout processes over 10 million user reports annually."
    add_ins = (
        f"Judge Correct only if BOTH are true:\n"
        f"(A) The answer states an annual magnitude ≥ 10 million; extracted: '{_safe(gb.annual_reports_magnitude if gb else None)}'.\n"
        f"(B) A cited official GeoBlackout page supports this magnitude. Otherwise, Incorrect."
    )
    await evaluator.verify(claim=claim, node=leaf, sources=gb_sources, additional_instruction=add_ins)


async def build_outage_report_checks(evaluator: Evaluator, parent, orp: Optional[OutageReportInfo]) -> None:
    node = evaluator.add_parallel(
        id="outage_report_service",
        desc="Outage.Report: provide the requested operational features matching the stated constraints",
        parent=parent,
        critical=True
    )

    or_sources = _srcs(orp.source_urls if orp else None)

    # Timing info
    leaf = evaluator.add_leaf(
        id="outage_report_timing",
        desc="States that Outage.Report displays start time, end time, and total duration for each incident",
        parent=node,
        critical=True
    )
    claim = "Outage.Report displays, for each incident, the start time, the end time, and the total duration."
    add_ins = (
        f"Judge Correct only if BOTH are true:\n"
        f"(A) The answer claims these three timing elements; extracted: '{_safe(orp.timing_info if orp else None)}'.\n"
        f"(B) A cited official Outage.Report page supports that these are displayed. Otherwise, Incorrect."
    )
    await evaluator.verify(claim=claim, node=leaf, sources=or_sources, additional_instruction=add_ins)

    # Geography info
    leaf = evaluator.add_leaf(
        id="outage_report_geography",
        desc="States that Outage.Report shows which countries were affected during incidents",
        parent=node,
        critical=True
    )
    claim = "Outage.Report shows which countries were affected during incidents."
    add_ins = (
        f"Judge Correct only if BOTH are true:\n"
        f"(A) The answer claims country-level affected areas; extracted: '{_safe(orp.geography_info if orp else None)}'.\n"
        f"(B) A cited official page supports this. Otherwise, Incorrect."
    )
    await evaluator.verify(claim=claim, node=leaf, sources=or_sources, additional_instruction=add_ins)

    # Coverage scope
    leaf = evaluator.add_leaf(
        id="outage_report_coverage",
        desc="States that Outage.Report provides global coverage",
        parent=node,
        critical=True
    )
    claim = "Outage.Report provides global coverage."
    add_ins = (
        f"Judge Correct only if BOTH are true:\n"
        f"(A) The answer explicitly states global coverage; extracted: '{_safe(orp.coverage_scope if orp else None)}'.\n"
        f"(B) A cited official page supports global coverage. Otherwise, Incorrect."
    )
    await evaluator.verify(claim=claim, node=leaf, sources=or_sources, additional_instruction=add_ins)


async def build_major_outage_checks(evaluator: Evaluator, parent, mo: Optional[CarrierOutageInfo]) -> None:
    node = evaluator.add_parallel(
        id="major_carrier_outage",
        desc="One major US cellular carrier outage, satisfying the stated constraints and including a credible reference URL",
        parent=parent,
        critical=True
    )

    mo_sources = _srcs(mo.source_urls if mo else None)

    # Occurred in 2024 or later (pure logical check based on the answer's provided date)
    leaf = evaluator.add_leaf(
        id="outage_date_2024_or_later",
        desc="The outage occurred in 2024 or later (constraint satisfied by the specified outage date)",
        parent=node,
        critical=True
    )
    claim = "The outage date provided in the answer is in 2024 or later."
    add_ins = (
        f"Use ONLY the date stated in the answer to judge this constraint. The extracted date was: '{_safe(mo.date if mo else None)}'. "
        f"Consider common formats (e.g., 'January 14, 2026' or '2026-01-14'). "
        f"If the answer does not provide a date or provides an earlier date, mark Incorrect."
    )
    await evaluator.verify(claim=claim, node=leaf, additional_instruction=add_ins)

    # Carrier identified as Verizon (and supported)
    leaf = evaluator.add_leaf(
        id="outage_carrier",
        desc="Identifies the carrier as Verizon",
        parent=node,
        critical=True
    )
    claim = "The specified outage was experienced by Verizon, a major US cellular carrier."
    add_ins = (
        f"Judge Correct only if BOTH are true:\n"
        f"(A) The answer explicitly names Verizon as the carrier; extracted carrier: '{_safe(mo.carrier if mo else None)}'.\n"
        f"(B) At least one cited credible/authoritative source confirms the outage was Verizon's. Otherwise, Incorrect."
    )
    await evaluator.verify(claim=claim, node=leaf, sources=mo_sources, additional_instruction=add_ins)

    # Specific date: January 14, 2026 (and supported)
    leaf = evaluator.add_leaf(
        id="outage_date_specific",
        desc="Provides the specific outage date as January 14, 2026",
        parent=node,
        critical=True
    )
    claim = "The outage occurred on January 14, 2026 (equivalently written as 2026-01-14)."
    add_ins = (
        f"Judge Correct only if BOTH are true:\n"
        f"(A) The answer explicitly states this same specific date; extracted date: '{_safe(mo.date if mo else None)}'.\n"
        f"(B) A cited credible/authoritative source confirms the outage date as January 14, 2026. Otherwise, Incorrect."
    )
    await evaluator.verify(claim=claim, node=leaf, sources=mo_sources, additional_instruction=add_ins)

    # Duration: over 10 hours (and supported)
    leaf = evaluator.add_leaf(
        id="outage_duration",
        desc="States the outage lasted over 10 hours",
        parent=node,
        critical=True
    )
    claim = "The outage lasted over 10 hours."
    add_ins = (
        f"Judge Correct only if BOTH are true:\n"
        f"(A) The answer explicitly claims a duration over 10 hours; extracted: '{_safe(mo.duration if mo else None)}'.\n"
        f"(B) A cited credible/authoritative source supports 'over 10 hours'. Otherwise, Incorrect."
    )
    await evaluator.verify(claim=claim, node=leaf, sources=mo_sources, additional_instruction=add_ins)

    # Impact metrics: more than 1.5 million customers affected (and supported)
    leaf = evaluator.add_leaf(
        id="outage_impact_metrics",
        desc="Provides quantitative impact metrics including that more than 1.5 million customers were affected",
        parent=node,
        critical=True
    )
    claim = "More than 1.5 million customers were affected by the outage."
    add_ins = (
        f"Judge Correct only if BOTH are true:\n"
        f"(A) The answer explicitly states a quantitative impact ≥ 1.5 million customers; extracted: '{_safe(mo.impact_metrics if mo else None)}'.\n"
        f"(B) A cited credible/authoritative source supports this threshold. Otherwise, Incorrect."
    )
    await evaluator.verify(claim=claim, node=leaf, sources=mo_sources, additional_instruction=add_ins)

    # Credible source URL present and relevant
    leaf = evaluator.add_leaf(
        id="outage_credible_source_url",
        desc="Provides at least one working reference URL from a credible news or authoritative source documenting the outage",
        parent=node,
        critical=True
    )
    claim = "This page is a credible news or authoritative source that documents the Verizon outage around January 14, 2026 in the United States, including date and impact information."
    add_ins = (
        "Judge Correct only if at least one working URL is provided in the answer AND the page is credible (e.g., major reputable news outlets; "
        "official company, regulator, or agency pages) AND it documents the specific outage with date and impact details. "
        "If the answer provides no URLs, or only non-credible/irrelevant links, mark Incorrect."
    )
    await evaluator.verify(claim=claim, node=leaf, sources=mo_sources, additional_instruction=add_ins)


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
    Evaluate an answer for the outage-tracking comparison and major outage documentation task.
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
        default_model=model,
    )

    # Extract structured info from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_all(),
        template_class=OutageTrackingExtraction,
        extraction_name="outage_tracking_extraction"
    )

    # Add a critical top-level node to mirror the rubric (root from Evaluator is always non-critical)
    task_root = evaluator.add_parallel(
        id="task_root",
        desc="Compare the three specified outage-tracking services and document one major US cellular-carrier outage (2024 or later), satisfying all stated constraints",
        parent=root,
        critical=True
    )

    # Build verification subtrees
    await build_downdetector_checks(evaluator, task_root, extracted.downdetector if extracted else None)
    await build_geoblackout_checks(evaluator, task_root, extracted.geoblackout if extracted else None)
    await build_outage_report_checks(evaluator, task_root, extracted.outage_report if extracted else None)
    await build_major_outage_checks(evaluator, task_root, extracted.major_outage if extracted else None)

    # Optionally record expected rubric facts for transparency
    evaluator.add_ground_truth({
        "expected": {
            "Downdetector": {
                "update_frequency": EXP_DD_UPDATE_FREQ,
                "baseline_period": EXP_DD_BASELINE_PERIOD,
                "status_levels": EXP_DD_STATUS_LEVELS_DESC,
                "service_count_threshold": EXP_DD_SERVICE_COUNT_THRESHOLD
            },
            "GeoBlackout": {
                "update_frequency": EXP_GB_UPDATE_FREQ,
                "precision_level": EXP_GB_PRECISION,
                "us_operators_monitored": EXP_GB_US_OPERATORS_COUNT,
                "annual_reports_magnitude": EXP_GB_ANNUAL_REPORTS_MAG
            },
            "Outage.Report": {
                "timing_info": EXP_OR_TIMING_INFO,
                "geography_info": EXP_OR_GEOGRAPHY_INFO,
                "coverage_scope": EXP_OR_COVERAGE_SCOPE
            },
            "MajorOutage": {
                "carrier": EXP_OUTAGE_CARRIER,
                "date": EXP_OUTAGE_DATE,
                "duration": EXP_OUTAGE_DURATION,
                "impact_threshold": EXP_OUTAGE_IMPACT
            }
        }
    })

    return evaluator.get_summary()