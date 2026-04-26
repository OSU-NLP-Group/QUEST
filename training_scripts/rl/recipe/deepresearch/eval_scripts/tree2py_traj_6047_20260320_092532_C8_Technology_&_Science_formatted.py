import asyncio
import logging
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.llm_client.base_client import LLMClient


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "verizon_outage_jan2026"
TASK_DESCRIPTION = (
    "Compile a comprehensive report on the Verizon network outage that occurred in January 2026. "
    "Your report must include: (1) Basic Event Information: Identify the specific date the outage occurred, "
    "the total duration (including whether it exceeded 10 hours), the approximate start and end times in Eastern Time, "
    "and the number of customers affected. (2) Technical Root Cause: Describe what caused the outage, including "
    "the specific network component involved (which type of 5G core network), and confirm whether or not the outage "
    "was caused by a cyberattack. (3) Geographic Impact: List all five major U.S. cities that experienced the highest "
    "concentration of outage reports. (4) Service Disruptions: Specify which types of telecommunications services were disrupted "
    "(voice, text, data) and whether emergency 911 services were affected. (5) Response Actions: Identify what remedy Verizon "
    "offered to affected customers and what regulatory action the FCC took in response. (6) FCC Reporting Thresholds: State the "
    "minimum outage duration (in minutes) and the minimum user-minutes threshold that trigger FCC reporting requirements for wireless carriers. "
    "(7) FCC General Timing Requirements: Specify the maximum time allowed (in minutes or hours/days as appropriate) for wireless carriers to submit: "
    "(a) the initial NORS notification, (b) the Initial Communications Outage Report, and (c) the Final Communications Outage Report after discovering "
    "a reportable outage. (8) FCC Emergency Services Requirements: State the maximum time allowed (in minutes or hours) for wireless carriers to: "
    "(a) notify affected 911 facilities when 911 services are potentially affected, and (b) provide the first follow-up notification to those facilities. "
    "For each piece of information, provide supporting evidence with reference URLs from reliable sources."
)

# Ground truth expectations used to phrase claims (for verification with sources)
EXPECTED_OUTAGE_DATE = "January 14, 2026"
EXPECTED_START_TIME_DESC = "around 12:00 p.m. Eastern Time (noon ET)"
EXPECTED_END_TIME_DESC = "around 10:20 p.m. Eastern Time"
EXPECTED_AFFECTED_MIN = "more than 1.5 million"
TOP_CITIES = ["New York City", "Atlanta", "Charlotte", "Houston", "Washington, D.C."]


# --------------------------------------------------------------------------- #
# Data models for structured extraction                                       #
# --------------------------------------------------------------------------- #
class BasicEventInfo(BaseModel):
    outage_date: Optional[str] = None
    outage_date_sources: List[str] = Field(default_factory=list)

    start_time_et: Optional[str] = None
    start_time_sources: List[str] = Field(default_factory=list)

    end_time_et: Optional[str] = None
    end_time_sources: List[str] = Field(default_factory=list)

    duration_exceeded_10_hours: Optional[str] = None  # "yes" or "no"
    duration_sources: List[str] = Field(default_factory=list)

    affected_customers: Optional[str] = None
    affected_sources: List[str] = Field(default_factory=list)


class TechnicalRootCause(BaseModel):
    software_issue: Optional[str] = None  # "yes" if described as software-caused
    software_issue_sources: List[str] = Field(default_factory=list)

    core_component: Optional[str] = None  # e.g., "5G Standalone core", "5G SA core"
    core_component_sources: List[str] = Field(default_factory=list)

    not_cyberattack: Optional[str] = None  # "yes" if confirmed not a cyberattack
    not_cyberattack_sources: List[str] = Field(default_factory=list)


class CityItem(BaseModel):
    city: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class GeographicImpact(BaseModel):
    major_cities: List[CityItem] = Field(default_factory=list)


class ServiceDisruptions(BaseModel):
    voice_disrupted: Optional[str] = None  # "yes"/"no"
    voice_sources: List[str] = Field(default_factory=list)

    text_disrupted: Optional[str] = None  # "yes"/"no"
    text_sources: List[str] = Field(default_factory=list)

    data_disrupted: Optional[str] = None  # "yes"/"no"
    data_sources: List[str] = Field(default_factory=list)

    e911_affected: Optional[str] = None  # "yes"/"no"
    e911_sources: List[str] = Field(default_factory=list)


class ResponseActions(BaseModel):
    account_credits_offered: Optional[str] = None  # "yes"/"no"
    credits_sources: List[str] = Field(default_factory=list)

    fcc_investigation_launched: Optional[str] = None  # "yes"/"no"
    fcc_investigation_sources: List[str] = Field(default_factory=list)


class FCCThresholds(BaseModel):
    min_duration_minutes: Optional[str] = None
    min_duration_sources: List[str] = Field(default_factory=list)

    user_minutes_threshold: Optional[str] = None
    user_minutes_sources: List[str] = Field(default_factory=list)


class FCCGeneralTiming(BaseModel):
    initial_notification_minutes: Optional[str] = None
    initial_notification_sources: List[str] = Field(default_factory=list)

    initial_report_timing: Optional[str] = None
    initial_report_sources: List[str] = Field(default_factory=list)

    final_report_timing: Optional[str] = None
    final_report_sources: List[str] = Field(default_factory=list)


class FCC911Requirements(BaseModel):
    notify_911_minutes: Optional[str] = None
    notify_911_sources: List[str] = Field(default_factory=list)

    follow_up_first_hours: Optional[str] = None
    follow_up_sources: List[str] = Field(default_factory=list)


class OutageReportExtraction(BaseModel):
    basic: Optional[BasicEventInfo] = None
    technical: Optional[TechnicalRootCause] = None
    geography: Optional[GeographicImpact] = None
    service: Optional[ServiceDisruptions] = None
    response: Optional[ResponseActions] = None
    fcc_thresholds: Optional[FCCThresholds] = None
    fcc_timing: Optional[FCCGeneralTiming] = None
    fcc_911: Optional[FCC911Requirements] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_outage_report() -> str:
    return """
Extract the report details exactly as presented in the answer. Do NOT invent values. For each requested field, also extract all supporting source URLs that the answer cites for that specific field. Return null for any value not present in the answer and an empty list for missing sources.

Use this JSON schema:
- basic:
  - outage_date: string | null (e.g., "January 14, 2026")
  - outage_date_sources: string[] (URLs mentioned in the answer for the outage date)
  - start_time_et: string | null (e.g., "around noon ET", "about 12:00 p.m. ET")
  - start_time_sources: string[]
  - end_time_et: string | null (e.g., "around 10:20 p.m. ET")
  - end_time_sources: string[]
  - duration_exceeded_10_hours: string | null ("yes" or "no" only)
  - duration_sources: string[]
  - affected_customers: string | null (e.g., "over 1.5 million", "1.6 million")
  - affected_sources: string[]
- technical:
  - software_issue: string | null ("yes" if the answer states it was a software-caused outage; else "no")
  - software_issue_sources: string[]
  - core_component: string | null (e.g., "5G Standalone core", "5G SA core")
  - core_component_sources: string[]
  - not_cyberattack: string | null ("yes" if the answer states Verizon said it was NOT a cyberattack; else "no")
  - not_cyberattack_sources: string[]
- geography:
  - major_cities: array of { city: string, sources: string[] } for the top cities most affected that the answer lists
- service:
  - voice_disrupted: string | null ("yes"/"no")
  - voice_sources: string[]
  - text_disrupted: string | null ("yes"/"no")
  - text_sources: string[]
  - data_disrupted: string | null ("yes"/"no")
  - data_sources: string[]
  - e911_affected: string | null ("yes"/"no")
  - e911_sources: string[]
- response:
  - account_credits_offered: string | null ("yes"/"no")
  - credits_sources: string[]
  - fcc_investigation_launched: string | null ("yes"/"no")
  - fcc_investigation_sources: string[]
- fcc_thresholds:
  - min_duration_minutes: string | null (e.g., "30")
  - min_duration_sources: string[]
  - user_minutes_threshold: string | null (e.g., "900,000")
  - user_minutes_sources: string[]
- fcc_timing:
  - initial_notification_minutes: string | null (e.g., "120")
  - initial_notification_sources: string[]
  - initial_report_timing: string | null (e.g., "72 hours", "3 calendar days")
  - initial_report_sources: string[]
  - final_report_timing: string | null (e.g., "30 days")
  - final_report_sources: string[]
- fcc_911:
  - notify_911_minutes: string | null (e.g., "30")
  - notify_911_sources: string[]
  - follow_up_first_hours: string | null (e.g., "2 hours")
  - follow_up_sources: string[]

Rules for URL extraction:
- Extract only actual URLs shown in the answer (including markdown links).
- If a field has multiple supporting URLs in the answer, include all of them.
- If the answer provides a general sources section, attribute URLs to relevant fields when possible; otherwise leave specific field lists empty.
    """.strip()


# --------------------------------------------------------------------------- #
# Helpers                                                                     #
# --------------------------------------------------------------------------- #
def _combine_sources(*lists: Optional[List[str]]) -> List[str]:
    seen = set()
    result: List[str] = []
    for lst in lists:
        if not lst:
            continue
        for url in lst:
            if not url:
                continue
            u = url.strip()
            if u and u not in seen:
                seen.add(u)
                result.append(u)
    return result


def _safe_list(lst: Optional[List[str]]) -> List[str]:
    return list(lst or [])


def _norm_city(name: str) -> str:
    return (
        name.lower()
        .replace(",", " ")
        .replace(".", "")
        .replace("  ", " ")
        .strip()
    )


CITY_SYNONYMS = {
    "New York City": {"nyc", "new york city", "new york", "new york ny"},
    "Atlanta": {"atlanta", "atlanta ga"},
    "Charlotte": {"charlotte", "charlotte nc"},
    "Houston": {"houston", "houston tx"},
    "Washington, D.C.": {
        "washington, d.c.",
        "washington dc",
        "washington d.c",
        "dc",
        "district of columbia",
        "washington",
        "washington, dc",
    },
}


def _city_sources(geo: Optional[GeographicImpact], target_city: str) -> List[str]:
    if not geo or not geo.major_cities:
        return []
    target_norms = CITY_SYNONYMS.get(target_city, {target_city.lower()})
    collected: List[str] = []
    for item in geo.major_cities:
        if not item or not item.city:
            continue
        norm = _norm_city(item.city)
        if norm in target_norms:
            collected = _combine_sources(collected, item.sources)
    # As a soft fallback, if nothing matched by synonym, include all city sources
    if not collected:
        all_sources = []
        for item in geo.major_cities:
            all_sources = _combine_sources(all_sources, item.sources)
        collected = all_sources
    return collected


# --------------------------------------------------------------------------- #
# Verification builders                                                       #
# --------------------------------------------------------------------------- #
async def build_basic_event_facts(evaluator: Evaluator, parent, extracted: OutageReportExtraction) -> None:
    node = evaluator.add_parallel(
        id="Outage_Event_Basic_Facts",
        desc="Basic factual information about the outage event",
        parent=parent,
        critical=False,
    )

    basic = extracted.basic or BasicEventInfo()

    # Outage Date (leaf)
    outage_date_node = evaluator.add_leaf(
        id="Outage_Date",
        desc="The specific date when the outage occurred (January 14, 2026)",
        parent=node,
        critical=True,
    )
    outage_date_claim = f"The Verizon network outage occurred on {EXPECTED_OUTAGE_DATE}."
    await evaluator.verify(
        claim=outage_date_claim,
        node=outage_date_node,
        sources=_safe_list(basic.outage_date_sources),
        additional_instruction="Allow equivalent date formats (e.g., Jan 14, 2026)."
    )

    # Duration details (parallel sub-node)
    duration_parent = evaluator.add_parallel(
        id="Outage_Duration_Details",
        desc="Detailed timing information about the outage",
        parent=node,
        critical=False,
    )

    # Duration exceeded 10 hours
    dur_node = evaluator.add_leaf(
        id="Duration_Exceeded_10_Hours",
        desc="The outage lasted over 10 hours",
        parent=duration_parent,
        critical=True,
    )
    # Approximate Start Time
    start_node = evaluator.add_leaf(
        id="Approximate_Start_Time",
        desc="The outage began around noon Eastern Time",
        parent=duration_parent,
        critical=True,
    )
    # Approximate End Time
    end_node = evaluator.add_leaf(
        id="Approximate_End_Time",
        desc="The outage was resolved around 10:20 p.m. Eastern Time",
        parent=duration_parent,
        critical=True,
    )

    # Prepare batch verification for duration details
    duration_sources = _combine_sources(
        basic.duration_sources,
        basic.start_time_sources,
        basic.end_time_sources,
    )
    start_claim = f"The outage began {EXPECTED_START_TIME_DESC} on {EXPECTED_OUTAGE_DATE}."
    end_claim = f"The outage was resolved {EXPECTED_END_TIME_DESC} on {EXPECTED_OUTAGE_DATE}."
    dur_claim = "The outage lasted more than 10 hours."

    await evaluator.batch_verify(
        [
            (
                dur_claim,
                duration_sources,
                dur_node,
                "Accept statements like 'over 10 hours', 'about 10+ hours', or calculations implied by start/end times."
            ),
            (
                start_claim,
                _safe_list(basic.start_time_sources) or duration_sources,
                start_node,
                "Treat 'around noon ET', 'about 12 p.m. ET', or 'shortly after noon' as equivalent."
            ),
            (
                end_claim,
                _safe_list(basic.end_time_sources) or duration_sources,
                end_node,
                "Treat 'around 10:20 p.m. ET', 'about 10:20pm ET', or 'around 10 p.m. ET' as close equivalents."
            ),
        ]
    )

    # Affected users scale
    affected_node = evaluator.add_leaf(
        id="Affected_Users_Scale",
        desc="Number of customers affected by the outage (more than 1.5 million)",
        parent=node,
        critical=True,
    )
    affected_claim = "More than 1.5 million customers were affected by the outage."
    await evaluator.verify(
        claim=affected_claim,
        node=affected_node,
        sources=_safe_list(basic.affected_sources),
        additional_instruction="Accept 'at least 1.5 million', 'over 1.5 million', or specific higher counts (e.g., ~1.7 million) that imply >1.5M."
    )


async def build_technical_cause(evaluator: Evaluator, parent, extracted: OutageReportExtraction) -> None:
    node = evaluator.add_parallel(
        id="Outage_Technical_Cause",
        desc="Technical root cause information",
        parent=parent,
        critical=False,
    )

    tech = extracted.technical or TechnicalRootCause()

    sw_node = evaluator.add_leaf(
        id="Software_Issue_Identified",
        desc="The outage was caused by a software issue",
        parent=node,
        critical=True,
    )
    core_node = evaluator.add_leaf(
        id="5G_SA_Core_Network",
        desc="The software issue occurred in the 5G Standalone core network during a feature update",
        parent=node,
        critical=True,
    )
    nocyb_node = evaluator.add_leaf(
        id="Not_Cyberattack",
        desc="Verizon confirmed the outage was NOT a cyberattack or cybersecurity breach",
        parent=node,
        critical=True,
    )

    await evaluator.batch_verify(
        [
            (
                "The outage was caused by a software issue.",
                _safe_list(tech.software_issue_sources),
                sw_node,
                "Look for wording like 'software issue', 'software update problem', 'software-related', 'software change'."
            ),
            (
                "The software issue occurred in Verizon's 5G Standalone (SA) core network during a feature update.",
                _safe_list(tech.core_component_sources),
                core_node,
                "Accept variants like '5G SA core', 'standalone core', '5G standalone core', and references to a software/feature update rollout."
            ),
            (
                "Verizon confirmed the outage was not a cyberattack or cybersecurity breach.",
                _safe_list(tech.not_cyberattack_sources),
                nocyb_node,
                "Accept negative confirmations such as 'no evidence of a cyberattack', 'not a hack', or official denials."
            ),
        ]
    )


async def build_geo_impact(evaluator: Evaluator, parent, extracted: OutageReportExtraction) -> None:
    node = evaluator.add_parallel(
        id="Outage_Geographic_Impact",
        desc="Major cities with highest concentration of outage reports",
        parent=parent,
        critical=False,
    )

    geo = extracted.geography or GeographicImpact()

    # Build city leaves
    city_nodes = [
        ("New_York_City", "New York City was among the major cities most affected", "New York City"),
        ("Atlanta", "Atlanta was among the major cities most affected", "Atlanta"),
        ("Charlotte", "Charlotte was among the major cities most affected", "Charlotte"),
        ("Houston", "Houston was among the major cities most affected", "Houston"),
        ("Washington_DC", "Washington D.C. was among the major cities most affected", "Washington, D.C."),
    ]

    claims_and_nodes = []
    for node_id, desc, city_name in city_nodes:
        leaf = evaluator.add_leaf(
            id=node_id,
            desc=desc,
            parent=node,
            critical=True,
        )
        sources = _city_sources(geo, city_name)
        claim = f"{city_name} was among the major U.S. cities with the highest concentration of Verizon outage reports on {EXPECTED_OUTAGE_DATE}."
        add_ins = "Allow reasonable synonyms: NYC/New York; Washington, DC/ Washington, D.C.; variants with state abbreviations."
        claims_and_nodes.append((claim, sources, leaf, add_ins))

    await evaluator.batch_verify(claims_and_nodes)


async def build_service_impact(evaluator: Evaluator, parent, extracted: OutageReportExtraction) -> None:
    node = evaluator.add_parallel(
        id="Outage_Service_Impact",
        desc="Types of services disrupted by the outage",
        parent=parent,
        critical=False,
    )

    svc = extracted.service or ServiceDisruptions()

    voice_node = evaluator.add_leaf(
        id="Voice_Calls_Disrupted",
        desc="Voice calling service was disrupted",
        parent=node,
        critical=True,
    )
    text_node = evaluator.add_leaf(
        id="Text_Messages_Disrupted",
        desc="Text messaging service was disrupted",
        parent=node,
        critical=True,
    )
    data_node = evaluator.add_leaf(
        id="Data_Services_Disrupted",
        desc="Data/internet services were disrupted",
        parent=node,
        critical=True,
    )
    e911_node = evaluator.add_leaf(
        id="911_Emergency_Services_Affected",
        desc="911 emergency services were affected in some areas, with cities issuing alerts",
        parent=node,
        critical=True,
    )

    await evaluator.batch_verify(
        [
            (
                "Verizon customers experienced disruption to voice calling service during the outage.",
                _safe_list(svc.voice_sources),
                voice_node,
                "Accept reports of intermittent or widespread voice call failures, inability to place/receive calls."
            ),
            (
                "Verizon customers experienced disruption to text messaging (SMS/MMS) during the outage.",
                _safe_list(svc.text_sources),
                text_node,
                "Accept reports that SMS/MMS messages failed to send/receive or were significantly delayed."
            ),
            (
                "Verizon customers experienced disruption to data/internet services during the outage.",
                _safe_list(svc.data_sources),
                data_node,
                "Accept reports of loss of mobile data, inability to access internet, or degraded throughput."
            ),
            (
                "Emergency 911 services were affected in some localities, with officials issuing alerts/warnings about potential difficulties reaching 911.",
                _safe_list(svc.e911_sources),
                e911_node,
                "It's sufficient if some jurisdictions reported issues or issued alerts; nationwide 911 failure is NOT required."
            ),
        ]
    )


async def build_response_actions(evaluator: Evaluator, parent, extracted: OutageReportExtraction) -> None:
    node = evaluator.add_parallel(
        id="Outage_Response_Actions",
        desc="Actions taken by Verizon and regulators in response to the outage",
        parent=parent,
        critical=False,
    )

    resp = extracted.response or ResponseActions()

    credits_node = evaluator.add_leaf(
        id="Account_Credits_Offered",
        desc="Verizon confirmed it would provide account credits to affected customers",
        parent=node,
        critical=True,
    )
    fcc_inv_node = evaluator.add_leaf(
        id="FCC_Investigation_Launched",
        desc="The FCC launched an investigation into the outage",
        parent=node,
        critical=True,
    )

    await evaluator.batch_verify(
        [
            (
                "Verizon stated it would provide account or bill credits to customers affected by the outage.",
                _safe_list(resp.credits_sources),
                credits_node,
                "Accept 'bill credits', 'account credits', or similar compensation announcements; automatic or upon request."
            ),
            (
                "The Federal Communications Commission (FCC) launched an investigation into the Verizon outage.",
                _safe_list(resp.fcc_investigation_sources),
                fcc_inv_node,
                "Accept phrasing such as 'opened an inquiry', 'investigating', or official FCC announcements."
            ),
        ]
    )


async def build_fcc_thresholds(evaluator: Evaluator, parent, extracted: OutageReportExtraction) -> None:
    node = evaluator.add_parallel(
        id="FCC_Reportability_Thresholds",
        desc="FCC thresholds that determine whether an outage must be reported",
        parent=parent,
        critical=False,
    )

    th = extracted.fcc_thresholds or FCCThresholds()

    min_dur_node = evaluator.add_leaf(
        id="Minimum_Duration_30_Minutes",
        desc="Outages must last at least 30 minutes to be reportable under FCC rules",
        parent=node,
        critical=True,
    )
    user_minutes_node = evaluator.add_leaf(
        id="User_Minutes_Threshold_900000",
        desc="Wireless carriers must report outages potentially affecting at least 900,000 user-minutes",
        parent=node,
        critical=True,
    )

    await evaluator.batch_verify(
        [
            (
                "Under FCC rules, a reportable outage must last at least 30 minutes.",
                _safe_list(th.min_duration_sources),
                min_dur_node,
                "Prefer FCC rules or public notices; credible summaries are acceptable if they accurately reflect FCC requirements."
            ),
            (
                "Wireless carriers must report outages that potentially affect at least 900,000 user-minutes.",
                _safe_list(th.user_minutes_sources),
                user_minutes_node,
                "Prefer FCC sources; 'user minutes' means number of affected users times outage duration in minutes."
            ),
        ]
    )


async def build_fcc_general_timing(evaluator: Evaluator, parent, extracted: OutageReportExtraction) -> None:
    node = evaluator.add_parallel(
        id="FCC_General_Notification_Timing",
        desc="FCC timing requirements for general outage notifications and reports",
        parent=parent,
        critical=False,
    )

    tm = extracted.fcc_timing or FCCGeneralTiming()

    init_notif_node = evaluator.add_leaf(
        id="Initial_Notification_120_Minutes",
        desc="Wireless carriers must submit NORS notification within 120 minutes of discovering a reportable outage",
        parent=node,
        critical=True,
    )
    init_report_node = evaluator.add_leaf(
        id="Initial_Report_72_Hours",
        desc="Initial Communications Outage Report must be submitted within 72 hours (3 calendar days)",
        parent=node,
        critical=True,
    )
    final_report_node = evaluator.add_leaf(
        id="Final_Report_30_Days",
        desc="Final Communications Outage Report must be submitted within 30 days of discovering the outage",
        parent=node,
        critical=True,
    )

    await evaluator.batch_verify(
        [
            (
                "The initial NORS notification must be submitted within 120 minutes of discovering a reportable outage.",
                _safe_list(tm.initial_notification_sources),
                init_notif_node,
                "Look for FCC NORS timing requirements; 'two hours' is equivalent to 120 minutes."
            ),
            (
                "The Initial Communications Outage Report must be submitted within 72 hours (3 calendar days) of discovering the outage.",
                _safe_list(tm.initial_report_sources),
                init_report_node,
                "Accept '72 hours' or '3 calendar days' as equivalent phrasing."
            ),
            (
                "The Final Communications Outage Report must be submitted within 30 days of discovering the outage.",
                _safe_list(tm.final_report_sources),
                final_report_node,
                "Look for '30 days' deadline language in FCC requirements."
            ),
        ]
    )


async def build_fcc_911_requirements(evaluator: Evaluator, parent, extracted: OutageReportExtraction) -> None:
    node = evaluator.add_parallel(
        id="FCC_911_Special_Requirements",
        desc="Special FCC requirements when 911 facilities are affected",
        parent=parent,
        critical=False,
    )

    p911 = extracted.fcc_911 or FCC911Requirements()

    notify_node = evaluator.add_leaf(
        id="911_Facility_Notification_30_Minutes",
        desc="If 911 facilities are potentially affected, carriers must notify affected facilities within 30 minutes",
        parent=node,
        critical=True,
    )
    follow_node = evaluator.add_leaf(
        id="911_Follow_Up_2_Hours",
        desc="First follow-up notification to affected 911 facilities must occur within 2 hours of initial contact",
        parent=node,
        critical=True,
    )

    await evaluator.batch_verify(
        [
            (
                "If 911 facilities are potentially affected, carriers must notify the affected 911 facilities within 30 minutes.",
                _safe_list(p911.notify_911_sources),
                notify_node,
                "Prefer FCC 911 outage notification rules; '30 minutes' deadline is required."
            ),
            (
                "Carriers must provide the first follow-up notification to affected 911 facilities within 2 hours of the initial contact.",
                _safe_list(p911.follow_up_sources),
                follow_node,
                "Prefer FCC 911 rules or public notices; 'two hours' is equivalent to '2 hours'."
            ),
        ]
    )


# --------------------------------------------------------------------------- #
# Main evaluation entry                                                       #
# --------------------------------------------------------------------------- #
async def evaluate_answer(
    client: LLMClient,
    answer: str,
    agent_name: str,
    answer_name: str,
    cache: CacheFileSys,
    semaphore: asyncio.Semaphore,
    logger: logging.Logger,
    model: str = "o4-mini",
) -> Dict:
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,
        agent_name=agent_name,
        answer_name=answer_name,
        client=client,
        task_description="Comprehensive analysis of the Verizon January 2026 outage and FCC compliance requirements",
        answer=answer,
        global_cache=cache,
        global_semaphore=semaphore,
        logger=logger,
        default_model=model,
    )

    # Extraction
    extracted = await evaluator.extract(
        prompt=prompt_extract_outage_report(),
        template_class=OutageReportExtraction,
        extraction_name="outage_report_extraction",
    )

    # Optional: record expectations as ground truth context (not used for scoring directly)
    evaluator.add_ground_truth({
        "expected_outage_date": EXPECTED_OUTAGE_DATE,
        "expected_top_cities": TOP_CITIES,
        "fcc_reportability": {"min_duration_minutes": "30", "user_minutes_threshold": "900,000"},
        "fcc_general_timing": {
            "initial_notification": "120 minutes",
            "initial_report": "72 hours (3 calendar days)",
            "final_report": "30 days"
        },
        "fcc_911_requirements": {
            "facility_notification": "30 minutes",
            "first_follow_up": "2 hours"
        }
    }, gt_type="ground_truth")

    # Build tree (as per rubric)
    # Root node is already initialized as non-critical parallel aggregator
    await build_basic_event_facts(evaluator, root, extracted)
    await build_technical_cause(evaluator, root, extracted)
    await build_geo_impact(evaluator, root, extracted)
    await build_service_impact(evaluator, root, extracted)
    await build_response_actions(evaluator, root, extracted)
    await build_fcc_thresholds(evaluator, root, extracted)
    await build_fcc_general_timing(evaluator, root, extracted)
    await build_fcc_911_requirements(evaluator, root, extracted)

    return evaluator.get_summary()