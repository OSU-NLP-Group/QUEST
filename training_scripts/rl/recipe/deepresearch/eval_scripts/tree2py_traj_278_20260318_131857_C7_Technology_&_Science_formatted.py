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
TASK_ID = "verizon_outage_jan2026"
TASK_DESCRIPTION = (
    "Research the major Verizon wireless network outage that occurred in January 2026 and compile comprehensive factual "
    "information about the incident and applicable regulatory requirements. Your response must include the following details:\n"
    "1) Incident Timeline: outage date, approximate start time (with time zone), approximate total duration in hours, "
    "and the time when Verizon officially announced the outage was resolved (with time zone). "
    "2) Outage Cause: the root cause as officially stated by Verizon. "
    "3) FCC NORS requirements for wireless providers (47 CFR § 4.9): duration threshold (minutes), user‑minutes threshold, "
    "notification deadline (minutes), initial report deadline (hours or days), final report deadline (days). "
    "4) 911 special facility notifications (47 CFR § 4.9(h)): initial notification deadline (minutes), first follow‑up deadline (hours). "
    "5) Technical definition: what “SOS” or “SOS only” means during a cellular outage. "
    "6) A URL to Verizon’s official statement/update page about the outage resolution."
)


# --------------------------------------------------------------------------- #
# Data models for structured extraction                                       #
# --------------------------------------------------------------------------- #
class OutageDetails(BaseModel):
    date: Optional[str] = None  # e.g., "January 25, 2026"
    start_time_with_tz: Optional[str] = None  # e.g., "around 11:15 AM ET"
    duration_hours: Optional[str] = None  # keep as string to allow ranges like "4–6"
    resolution_time_with_tz: Optional[str] = None  # e.g., "9:30 PM ET"
    cause: Optional[str] = None  # root cause as stated by Verizon
    sources: List[str] = Field(default_factory=list)  # URLs cited in the answer regarding the outage details
    verizon_official_url: Optional[str] = None  # URL to Verizon’s official statement/update page


class FCCNORSDetails(BaseModel):
    duration_threshold_minutes: Optional[str] = None
    user_minutes_threshold: Optional[str] = None
    notification_timeline_minutes: Optional[str] = None
    initial_report_timeline: Optional[str] = None  # e.g., "72 hours" or "3 days"
    final_report_timeline_days: Optional[str] = None
    sources: List[str] = Field(default_factory=list)  # FCC regulation URLs used/cited


class SpecialFacility911Details(BaseModel):
    initial_notification_minutes: Optional[str] = None
    first_followup_hours: Optional[str] = None
    sources: List[str] = Field(default_factory=list)  # FCC regulation URLs used/cited for §4.9(h)


class TechnicalInfo(BaseModel):
    sos_meaning: Optional[str] = None  # explanation of "SOS" or "SOS only"
    sources: List[str] = Field(default_factory=list)  # vendor docs (Apple/Android), carrier docs, etc.


class ExtractionResult(BaseModel):
    outage: Optional[OutageDetails] = None
    fcc_nors: Optional[FCCNORSDetails] = None
    special_911: Optional[SpecialFacility911Details] = None
    technical: Optional[TechnicalInfo] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_all() -> str:
    return (
        "Extract the following information from the provided answer text exactly as written, without adding or inferring any "
        "missing values. When a requested field is not present, return null (or an empty list for URL arrays). Use strings for "
        "all scalar fields so that approximate or ranged values (e.g., 'around 2 hours', '3–4') are preserved.\n\n"
        "Return a single JSON object matching this schema:\n"
        "{\n"
        "  outage: {\n"
        "    date: string | null,\n"
        "    start_time_with_tz: string | null,\n"
        "    duration_hours: string | null,\n"
        "    resolution_time_with_tz: string | null,\n"
        "    cause: string | null,\n"
        "    sources: string[]  // all URLs in the answer specifically about the outage timeline/cause\n"
        "    verizon_official_url: string | null  // URL to Verizon’s official statement/update page about this outage\n"
        "  },\n"
        "  fcc_nors: {\n"
        "    duration_threshold_minutes: string | null,\n"
        "    user_minutes_threshold: string | null,\n"
        "    notification_timeline_minutes: string | null,\n"
        "    initial_report_timeline: string | null,  // 'X hours' or 'Y days'\n"
        "    final_report_timeline_days: string | null,\n"
        "    sources: string[]  // FCC rules or public notices cited for NORS thresholds/timelines\n"
        "  },\n"
        "  special_911: {\n"
        "    initial_notification_minutes: string | null,\n"
        "    first_followup_hours: string | null,\n"
        "    sources: string[]  // FCC rule sources for § 4.9(h)\n"
        "  },\n"
        "  technical: {\n"
        "    sos_meaning: string | null,\n"
        "    sources: string[]  // e.g., Apple/Android support articles or carrier docs cited in the answer\n"
        "  }\n"
        "}\n\n"
        "Rules:\n"
        "- Only include URLs explicitly present in the answer. Do not invent or search for new URLs.\n"
        "- Keep time zone indicators in the time fields if the answer provides them (e.g., 'ET', 'PT', 'Eastern Time').\n"
        "- If the answer uses approximations ('around', '~'), preserve that wording in the extracted value.\n"
    )


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def collect_sources(*candidate_lists: Optional[List[str]], extra: Optional[List[str]] = None) -> List[str]:
    urls: List[str] = []
    for lst in candidate_lists:
        if not lst:
            continue
        for u in lst:
            if isinstance(u, str):
                s = u.strip()
                if s and s not in urls:
                    urls.append(s)
    if extra:
        for u in extra:
            if isinstance(u, str):
                s = u.strip()
                if s and s not in urls:
                    urls.append(s)
    return urls


# --------------------------------------------------------------------------- #
# Verification logic                                                          #
# --------------------------------------------------------------------------- #
async def verify_all(evaluator: Evaluator, parent_node, info: ExtractionResult) -> None:
    # Create a grouping node to mirror the rubric tree
    group_node = evaluator.add_parallel(
        id="Verizon_January_2026_Outage_Information",
        desc="Comprehensive factual information about the Verizon wireless network outage that occurred in January 2026, including incident details and applicable FCC regulatory requirements",
        parent=parent_node,
        critical=False
    )

    outage = info.outage or OutageDetails()
    fcc = info.fcc_nors or FCCNORSDetails()
    sp911 = info.special_911 or SpecialFacility911Details()
    tech = info.technical or TechnicalInfo()

    # Prepare common source bundles
    outage_urls = collect_sources(outage.sources, extra=[outage.verizon_official_url] if outage.verizon_official_url else None)
    fcc_urls = collect_sources(fcc.sources)
    sp911_urls = collect_sources(sp911.sources)
    tech_urls = collect_sources(tech.sources)

    # 1) Outage_Date
    node = evaluator.add_leaf(
        id="Outage_Date",
        desc="The specific date when the major Verizon wireless outage occurred",
        parent=group_node,
        critical=True
    )
    date_val = outage.date or ""
    claim = f"The major Verizon wireless network outage in question occurred on {date_val}."
    await evaluator.verify(
        claim=claim,
        node=node,
        sources=outage_urls,
        additional_instruction="Verify the calendar date for the large Verizon wireless outage that happened in January 2026. "
                               "If multiple articles mention a specific date, prefer Verizon’s official update when available. "
                               "Small phrasing differences like 'on Jan. 25, 2026' vs 'January 25, 2026' are acceptable."
    )

    # 2) Outage_Start_Time
    node = evaluator.add_leaf(
        id="Outage_Start_Time",
        desc="The approximate time when the outage began, including time zone",
        parent=group_node,
        critical=True
    )
    start_val = outage.start_time_with_tz or ""
    claim = f"The outage began at approximately {start_val}."
    await evaluator.verify(
        claim=claim,
        node=node,
        sources=outage_urls,
        additional_instruction="Check whether sources state the approximate outage start time and include a time zone (e.g., ET/Eastern). "
                               "Allow reasonable approximations (±30 minutes) and equivalent time zone naming."
    )

    # 3) Outage_Duration
    node = evaluator.add_leaf(
        id="Outage_Duration",
        desc="The approximate total duration of the outage in hours",
        parent=group_node,
        critical=True
    )
    duration_val = outage.duration_hours or ""
    claim = f"The outage lasted approximately {duration_val} hours."
    await evaluator.verify(
        claim=claim,
        node=node,
        sources=outage_urls,
        additional_instruction="Verify an approximate total duration of the January 2026 Verizon outage. "
                               "If sources provide a range (e.g., 4–6 hours), a paraphrase such as 'about 5 hours' is acceptable."
    )

    # 4) Outage_Resolution_Time
    node = evaluator.add_leaf(
        id="Outage_Resolution_Time",
        desc="The specific time when Verizon officially announced the outage was resolved, including time zone",
        parent=group_node,
        critical=True
    )
    resolved_val = outage.resolution_time_with_tz or ""
    claim = f"Verizon officially announced that service was restored at approximately {resolved_val}."
    await evaluator.verify(
        claim=claim,
        node=node,
        sources=outage.verizon_official_url if outage.verizon_official_url else outage_urls,
        additional_instruction="Prefer Verizon’s official page as the primary evidence for the stated resolution time. "
                               "Allow minor rounding (±15 minutes) and equivalent time zone wording."
    )

    # 5) Outage_Cause
    node = evaluator.add_leaf(
        id="Outage_Cause",
        desc="The root cause of the outage as stated by Verizon",
        parent=group_node,
        critical=True
    )
    cause_val = outage.cause or ""
    claim = f"According to Verizon’s official statement, the root cause of the outage was: {cause_val}."
    await evaluator.verify(
        claim=claim,
        node=node,
        sources=outage.verizon_official_url if outage.verizon_official_url else outage_urls,
        additional_instruction="Verify the cause exactly as characterized by Verizon (e.g., software configuration, update error). "
                               "Paraphrases are acceptable if they clearly convey the same root cause."
    )

    # 6) FCC_Duration_Threshold
    node = evaluator.add_leaf(
        id="FCC_Duration_Threshold",
        desc="The minimum duration threshold (in minutes) for an outage to be considered reportable under FCC NORS requirements",
        parent=group_node,
        critical=True
    )
    thr_val = fcc.duration_threshold_minutes or ""
    claim = f"Under 47 CFR § 4.9 (NORS for wireless), the minimum duration threshold for a reportable outage is {thr_val} minutes."
    await evaluator.verify(
        claim=claim,
        node=node,
        sources=fcc_urls,
        additional_instruction="Use FCC Part 4 rules (47 CFR § 4.9) or official FCC guidance pages cited in the answer. "
                               "Do not rely on non-FCC sources unless they are authoritative summaries that quote the rule."
    )

    # 7) FCC_User_Minutes_Threshold
    node = evaluator.add_leaf(
        id="FCC_User_Minutes_Threshold",
        desc="The minimum number of user-minutes that a wireless outage must potentially affect to trigger FCC NORS reporting requirements",
        parent=group_node,
        critical=True
    )
    um_val = fcc.user_minutes_threshold or ""
    claim = f"Under 47 CFR § 4.9, the user-minutes threshold for a wireless outage to be reportable is {um_val} user-minutes."
    await evaluator.verify(
        claim=claim,
        node=node,
        sources=fcc_urls,
        additional_instruction="Confirm the numerical user‑minutes threshold from the FCC rules or official guidance pages."
    )

    # 8) FCC_Notification_Timeline
    node = evaluator.add_leaf(
        id="FCC_Notification_Timeline",
        desc="The maximum time window (in minutes) within which wireless providers must submit a Notification to the FCC after discovering a reportable outage",
        parent=group_node,
        critical=True
    )
    notif_val = fcc.notification_timeline_minutes or ""
    claim = f"Wireless providers must submit a NORS Notification to the FCC no later than {notif_val} minutes after discovering a reportable outage."
    await evaluator.verify(
        claim=claim,
        node=node,
        sources=fcc_urls,
        additional_instruction="Verify the Notification timing requirement in minutes for wireless providers under § 4.9."
    )

    # 9) FCC_Initial_Report_Timeline
    node = evaluator.add_leaf(
        id="FCC_Initial_Report_Timeline",
        desc="The maximum time window (in hours or days) within which providers must submit an Initial Communications Outage Report to the FCC after discovering the outage",
        parent=group_node,
        critical=True
    )
    init_val = fcc.initial_report_timeline or ""
    claim = f"Providers must submit the Initial Communications Outage Report within {init_val} after discovering the outage."
    await evaluator.verify(
        claim=claim,
        node=node,
        sources=fcc_urls,
        additional_instruction="Verify the timing for the Initial report submission window under § 4.9. "
                               "Units may be hours or days; allow equivalent conversions (e.g., 72 hours = 3 days)."
    )

    # 10) FCC_Final_Report_Timeline
    node = evaluator.add_leaf(
        id="FCC_Final_Report_Timeline",
        desc="The maximum time window (in days) within which providers must submit a Final Communications Outage Report to the FCC after discovering the outage",
        parent=group_node,
        critical=True
    )
    final_val = fcc.final_report_timeline_days or ""
    claim = f"Providers must submit the Final Communications Outage Report within {final_val} days after discovering the outage."
    await evaluator.verify(
        claim=claim,
        node=node,
        sources=fcc_urls,
        additional_instruction="Verify the timing for the Final report submission window (in days) under § 4.9."
    )

    # 11) 911_Facility_Initial_Notification_Timeline
    node = evaluator.add_leaf(
        id="911_Facility_Initial_Notification_Timeline",
        desc="The maximum time window (in minutes) within which providers must notify potentially affected 911 special facilities after discovering an outage affecting such facilities",
        parent=group_node,
        critical=True
    )
    sp_init_val = sp911.initial_notification_minutes or ""
    claim = f"When an outage affects 911 special facilities, providers must notify those facilities within {sp_init_val} minutes of discovery."
    await evaluator.verify(
        claim=claim,
        node=node,
        sources=sp911_urls if sp911_urls else fcc_urls,
        additional_instruction="Confirm the initial PSAP/911 special facility notification timing under § 4.9(h). "
                               "If the answer cites a specific FCC rule page for § 4.9(h), use that."
    )

    # 12) 911_Facility_Followup_Timeline
    node = evaluator.add_leaf(
        id="911_Facility_Followup_Timeline",
        desc="The maximum time window (in hours) for the first follow-up notification to potentially affected 911 special facilities after the initial contact",
        parent=group_node,
        critical=True
    )
    sp_follow_val = sp911.first_followup_hours or ""
    claim = f"The first follow-up notification to 911 special facilities must occur within {sp_follow_val} hours after the initial contact."
    await evaluator.verify(
        claim=claim,
        node=node,
        sources=sp911_urls if sp911_urls else fcc_urls,
        additional_instruction="Confirm the first follow‑up notification deadline for PSAPs/911 special facilities under § 4.9(h)."
    )

    # 13) SOS_Mode_Meaning
    node = evaluator.add_leaf(
        id="SOS_Mode_Meaning",
        desc="The technical meaning of 'SOS' or 'SOS only' display on a smartphone during a cellular network outage",
        parent=group_node,
        critical=True
    )
    sos_val = tech.sos_meaning or ""
    claim = f"When a smartphone displays 'SOS' or 'SOS only' during a cellular outage, it means: {sos_val}"
    await evaluator.verify(
        claim=claim,
        node=node,
        sources=tech_urls,
        additional_instruction="Verify against authoritative phone vendor or carrier documentation that 'SOS' indicates the device can still place emergency calls "
                               "(e.g., to 911) over any available network, while normal carrier service/data is unavailable."
    )

    # 14) Official_Verizon_Statement_URL
    node = evaluator.add_leaf(
        id="Official_Verizon_Statement_URL",
        desc="A reference URL to Verizon's official statement or update about the network outage resolution",
        parent=group_node,
        critical=True
    )
    verizon_url = outage.verizon_official_url or ""
    claim = "This page is an official Verizon page that provides an update or statement about the January 2026 wireless network outage and its resolution."
    await evaluator.verify(
        claim=claim,
        node=node,
        sources=verizon_url if verizon_url else outage_urls,
        additional_instruction="Confirm that the URL is on an official Verizon domain and clearly addresses the January 2026 outage and its resolution."
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
    Evaluate an answer for the Verizon January 2026 outage information task.
    """
    # Initialize evaluator
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

    # Extract structured information
    extracted = await evaluator.extract(
        prompt=prompt_extract_all(),
        template_class=ExtractionResult,
        extraction_name="extracted_outage_and_fcc_info"
    )

    # Optionally record a compact view of URLs provided
    evaluator.add_custom_info(
        {
            "outage_urls": (extracted.outage.sources if extracted.outage and extracted.outage.sources else []),
            "verizon_official_url": (extracted.outage.verizon_official_url if extracted.outage else None),
            "fcc_urls": (extracted.fcc_nors.sources if extracted.fcc_nors and extracted.fcc_nors.sources else []),
            "special_911_urls": (extracted.special_911.sources if extracted.special_911 and extracted.special_911.sources else []),
            "technical_urls": (extracted.technical.sources if extracted.technical and extracted.technical.sources else []),
        },
        info_type="urls_overview",
        info_name="urls_overview"
    )

    # Build verification tree and run checks
    await verify_all(evaluator, root, extracted)

    # Return the evaluation summary
    return evaluator.get_summary()