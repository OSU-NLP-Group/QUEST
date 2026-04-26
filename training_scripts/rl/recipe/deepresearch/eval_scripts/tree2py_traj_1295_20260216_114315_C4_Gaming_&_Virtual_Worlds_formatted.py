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
TASK_ID = "gaming_schedule_2026_02_10_12"
TASK_DESCRIPTION = """A gaming community manager is planning a week's content schedule and needs to avoid times when major platforms are unavailable or hosting significant broadcasts. For the period of February 10-12, 2026, provide the following information:

1. Steam's regular weekly maintenance:
   - The specific date of Steam's maintenance during this period
   - The time window when this maintenance typically occurs (include time zone)
   - The typical duration of Steam maintenance
   - A reference URL from Steam's official documentation confirming this maintenance schedule

2. PlayStation's State of Play broadcast:
   - The date of the State of Play event
   - The start time in at least two different time zones (must include Pacific Time)
   - The expected duration of the broadcast
   - A reference URL from PlayStation's official announcement

All times must include proper time zone designations, and all reference URLs must be from official company sources or documentation.
"""

# Ground truth expectations for strict evaluation
EXPECTED_STEAM = {
    "date": "Tuesday, February 10, 2026",
    "time_window": "approximately 3–6 PM Pacific Time (PT)",
    "duration": "20–30 minutes, up to 1 hour maximum"
}
EXPECTED_SOP = {
    "date": "Thursday, February 12, 2026",
    "times": ["2:00 PM PT", "5:00 PM ET", "11:00 PM CET"],
    "duration": "60+ minutes"
}

# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class SteamInfo(BaseModel):
    maintenance_date: Optional[str] = None
    time_window: Optional[str] = None
    duration: Optional[str] = None
    reference_urls: List[str] = Field(default_factory=list)


class StateOfPlayInfo(BaseModel):
    event_date: Optional[str] = None
    start_times: List[str] = Field(default_factory=list)  # Each element should include time and time zone
    duration: Optional[str] = None
    reference_urls: List[str] = Field(default_factory=list)


class ScheduleExtraction(BaseModel):
    steam: Optional[SteamInfo] = None
    state_of_play: Optional[StateOfPlayInfo] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_schedule() -> str:
    return """
    Extract the requested schedule details exactly as they appear in the answer. Organize the result into two top-level sections: 'steam' and 'state_of_play'.

    For 'steam', extract:
    - maintenance_date: The specific date (between Feb 10–12, 2026) the answer identifies for Steam's weekly maintenance.
    - time_window: The typical time window for Steam maintenance as written (include the time zone text if present, e.g., "3–6 PM PT" or "3–5 PM PST").
    - duration: The typical duration as written (e.g., "20–30 minutes", "up to one hour").
    - reference_urls: An array of all reference URLs cited for Steam maintenance. Include only URLs that appear in the answer verbatim.

    For 'state_of_play', extract:
    - event_date: The date the answer states for PlayStation's State of Play.
    - start_times: An array of the start time(s) with explicit time zones as written in the answer (e.g., "2:00 PM PT", "5:00 PM ET", "11:00 PM CET"). Include all that are listed.
    - duration: The expected duration as written (e.g., "60+ minutes").
    - reference_urls: An array of all reference URLs cited for the PlayStation announcement. Include only URLs that appear in the answer verbatim.

    Rules:
    - Return fields as strings exactly as written in the answer; do not normalize or infer.
    - If a field is missing in the answer, set it to null (or an empty array for URL lists).
    - For URLs, extract only valid-looking URLs that are explicitly present in the answer text (plain or in markdown).
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def domain_of(url: str) -> str:
    try:
        parsed = urlparse(url if (url.startswith("http://") or url.startswith("https://")) else f"http://{url}")
        return parsed.netloc.lower()
    except Exception:
        return ""


def filter_official_urls(urls: List[str], vendor: str) -> List[str]:
    if vendor == "steam":
        # Accept official Steam/Valve domains commonly used for support/docs/announcements
        allowed_substrings = ["steampowered.com", "steamcommunity.com", "valvesoftware.com"]
        return [u for u in urls if any(s in domain_of(u) for s in allowed_substrings)]
    if vendor == "playstation":
        # Accept official PlayStation domains
        allowed_substrings = ["playstation.com", "sony.com"]
        return [u for u in urls if any(s in domain_of(u) for s in allowed_substrings)]
    return []


def has_pt_timezone(text: Optional[str]) -> bool:
    if not text:
        return False
    t = text.lower()
    return (" pt" in t) or ("pst" in t) or ("pdt" in t) or ("pacific" in t)


def list_has_pt(times: List[str]) -> bool:
    for s in times:
        if has_pt_timezone(s):
            return True
    return False


# --------------------------------------------------------------------------- #
# Verification subtrees                                                       #
# --------------------------------------------------------------------------- #
async def build_steam_subtree(evaluator: Evaluator, parent, steam: SteamInfo) -> None:
    # 1) Steam maintenance date
    node_date = evaluator.add_sequential(
        id="steam_maintenance_date",
        desc="Correctly identify that Steam maintenance occurs on Tuesday, February 10, 2026",
        parent=parent,
        critical=True
    )
    evaluator.add_custom_node(
        result=steam is not None and steam.maintenance_date is not None and steam.maintenance_date.strip() != "",
        id="steam_date_provided",
        desc="Steam maintenance date is provided in the answer",
        parent=node_date,
        critical=True
    )
    leaf_date_match = evaluator.add_leaf(
        id="steam_date_matches_expected",
        desc="Answer states the Steam maintenance date during Feb 10–12, 2026 as Tuesday, February 10, 2026",
        parent=node_date,
        critical=True
    )
    claim_date = (
        "In the answer, the Steam regular weekly maintenance date (within the period Feb 10–12, 2026) "
        "is stated as Tuesday, February 10, 2026. Consider minor formatting variants like 'Tue, Feb 10, 2026' as a match."
    )
    await evaluator.verify(
        claim=claim_date,
        node=leaf_date_match,
        additional_instruction="Check the answer text only. This is a content match check."
    )

    # 2) Steam maintenance time window
    node_time = evaluator.add_sequential(
        id="steam_maintenance_time",
        desc="Correctly specify Steam maintenance time window as approximately 3-6 PM Pacific Time (or equivalent in other time zones)",
        parent=parent,
        critical=True
    )
    evaluator.add_custom_node(
        result=steam is not None and steam.time_window is not None and steam.time_window.strip() != "",
        id="steam_time_provided",
        desc="Steam maintenance time window is provided in the answer",
        parent=node_time,
        critical=True
    )
    evaluator.add_custom_node(
        result=has_pt_timezone(steam.time_window),
        id="steam_time_has_timezone",
        desc="Steam maintenance time window includes a Pacific Time designation (PT/PST/PDT/Pacific)",
        parent=node_time,
        critical=True
    )
    leaf_time_match = evaluator.add_leaf(
        id="steam_time_matches_expected",
        desc="Answer states the Steam maintenance time window as approximately 3–6 PM PT",
        parent=node_time,
        critical=True
    )
    claim_time = (
        "In the answer, the Steam maintenance time window is given as approximately 3–6 PM Pacific Time (PT). "
        "Treat close variants such as 'around 3–5 PM PT' or 'around 3 PM PT up to an hour' as acceptable for 'approximately 3–6 PM PT'."
    )
    await evaluator.verify(
        claim=claim_time,
        node=leaf_time_match,
        additional_instruction="This is a content check against the answer; accept minor wording differences."
    )

    # 3) Steam maintenance duration
    node_duration = evaluator.add_sequential(
        id="steam_maintenance_duration",
        desc="Correctly specify Steam maintenance typical duration as 20-30 minutes, up to 1 hour maximum",
        parent=parent,
        critical=True
    )
    evaluator.add_custom_node(
        result=steam is not None and steam.duration is not None and steam.duration.strip() != "",
        id="steam_duration_provided",
        desc="Steam maintenance typical duration is provided in the answer",
        parent=node_duration,
        critical=True
    )
    leaf_duration_match = evaluator.add_leaf(
        id="steam_duration_matches_expected",
        desc="Answer states the typical Steam maintenance duration as 20–30 minutes with up to 1 hour maximum",
        parent=node_duration,
        critical=True
    )
    claim_duration = (
        "In the answer, the typical duration of Steam maintenance is given as 20–30 minutes, with up to 1 hour maximum. "
        "Accept semantically equivalent wording (e.g., '20-30 mins' and 'up to one hour')."
    )
    await evaluator.verify(
        claim=claim_duration,
        node=leaf_duration_match,
        additional_instruction="Check the answer text only. Minor phrasing differences are acceptable."
    )

    # 4) Steam official reference URL
    node_ref = evaluator.add_sequential(
        id="steam_reference_url",
        desc="Provide a valid reference URL from Steam official sources documenting the weekly Tuesday maintenance schedule",
        parent=parent,
        critical=True
    )
    official_steam_urls = filter_official_urls(steam.reference_urls if steam else [], "steam")
    evaluator.add_custom_node(
        result=len(official_steam_urls) > 0,
        id="steam_official_url_present",
        desc="At least one official Steam/Valve reference URL is provided in the answer",
        parent=node_ref,
        critical=True
    )
    leaf_ref_supports = evaluator.add_leaf(
        id="steam_schedule_supported_by_official",
        desc="Official Steam page supports weekly Tuesday maintenance (afternoon Pacific Time, routine duration expectations)",
        parent=node_ref,
        critical=True
    )
    if official_steam_urls:
        claim_ref = (
            "This official Steam/Valve page documents that regular scheduled Steam maintenance occurs weekly on Tuesdays, "
            "typically in the afternoon Pacific Time (around 3 PM) and may last up to about an hour."
        )
        await evaluator.verify(
            claim=claim_ref,
            node=leaf_ref_supports,
            sources=official_steam_urls,
            additional_instruction=(
                "Focus on whether the page mentions weekly Tuesday maintenance and an afternoon Pacific Time window. "
                "If it mentions 'every Tuesday' and a timeframe around 3 PM Pacific (and/or up to roughly an hour), count as supported."
            )
        )
    else:
        # No official URL provided -> fail this leaf explicitly
        leaf_ref_supports.score = 0.0
        leaf_ref_supports.status = "failed"


async def build_sop_subtree(evaluator: Evaluator, parent, sop: StateOfPlayInfo) -> None:
    # 1) State of Play date
    node_date = evaluator.add_sequential(
        id="state_of_play_date",
        desc="Correctly identify State of Play date as Thursday, February 12, 2026",
        parent=parent,
        critical=True
    )
    evaluator.add_custom_node(
        result=sop is not None and sop.event_date is not None and sop.event_date.strip() != "",
        id="sop_date_provided",
        desc="State of Play event date is provided in the answer",
        parent=node_date,
        critical=True
    )
    leaf_date_match = evaluator.add_leaf(
        id="sop_date_matches_expected",
        desc="Answer states the State of Play date as Thursday, February 12, 2026",
        parent=node_date,
        critical=True
    )
    claim_date = (
        "In the answer, the PlayStation State of Play date is Thursday, February 12, 2026. "
        "Accept minor formatting differences such as 'Thu, Feb 12, 2026'."
    )
    await evaluator.verify(
        claim=claim_date,
        node=leaf_date_match,
        additional_instruction="Check the answer text only for the stated date."
    )

    # 2) State of Play start time(s)
    node_time = evaluator.add_sequential(
        id="state_of_play_time",
        desc="Correctly specify State of Play start time as 2:00 PM PT / 5:00 PM ET / 11:00 PM CET",
        parent=parent,
        critical=True
    )
    evaluator.add_custom_node(
        result=sop is not None and isinstance(sop.start_times, list) and len(sop.start_times) >= 2,
        id="sop_times_provided",
        desc="State of Play start times include at least two time zones",
        parent=node_time,
        critical=True
    )
    evaluator.add_custom_node(
        result=list_has_pt(sop.start_times if sop else []),
        id="sop_times_include_pt",
        desc="State of Play start times include a Pacific Time (PT/PST/PDT) entry",
        parent=node_time,
        critical=True
    )
    leaf_time_match = evaluator.add_leaf(
        id="sop_times_match_expected",
        desc="Answer states the start time as 2:00 PM PT / 5:00 PM ET / 11:00 PM CET",
        parent=node_time,
        critical=True
    )
    claim_time = (
        "In the answer, the PlayStation State of Play start time is given as 2:00 PM PT / 5:00 PM ET / 11:00 PM CET. "
        "Allow minor formatting/casing variations (e.g., '11pm CET' or '23:00 CET')."
    )
    await evaluator.verify(
        claim=claim_time,
        node=leaf_time_match,
        additional_instruction="Check the answer text only for those three time zone entries."
    )

    # 3) State of Play duration
    node_duration = evaluator.add_sequential(
        id="state_of_play_duration",
        desc="Correctly specify State of Play duration as 60+ minutes",
        parent=parent,
        critical=True
    )
    evaluator.add_custom_node(
        result=sop is not None and sop.duration is not None and sop.duration.strip() != "",
        id="sop_duration_provided",
        desc="State of Play expected duration is provided in the answer",
        parent=node_duration,
        critical=True
    )
    leaf_duration_match = evaluator.add_leaf(
        id="sop_duration_matches_expected",
        desc="Answer states the expected State of Play duration as 60+ minutes",
        parent=node_duration,
        critical=True
    )
    claim_duration = (
        "In the answer, the expected length of the State of Play broadcast is 60+ minutes (over an hour). "
        "Accept semantically equivalent wording (e.g., 'over 60 minutes', 'more than an hour')."
    )
    await evaluator.verify(
        claim=claim_duration,
        node=leaf_duration_match,
        additional_instruction="Check the answer text only for the duration phrasing."
    )

    # 4) State of Play official reference URL
    node_ref = evaluator.add_sequential(
        id="state_of_play_reference_url",
        desc="Provide a valid reference URL from PlayStation official sources announcing the February 12, 2026 State of Play",
        parent=parent,
        critical=True
    )
    official_ps_urls = filter_official_urls(sop.reference_urls if sop else [], "playstation")
    evaluator.add_custom_node(
        result=len(official_ps_urls) > 0,
        id="sop_official_url_present",
        desc="At least one official PlayStation reference URL is provided in the answer",
        parent=node_ref,
        critical=True
    )
    leaf_ref_supports = evaluator.add_leaf(
        id="sop_event_announced_on_date_supported",
        desc="Official PlayStation page announces a State of Play on Thursday, February 12, 2026",
        parent=node_ref,
        critical=True
    )
    if official_ps_urls:
        claim_ref = (
            "This official PlayStation announcement page states that a State of Play is scheduled for Thursday, February 12, 2026."
        )
        await evaluator.verify(
            claim=claim_ref,
            node=leaf_ref_supports,
            sources=official_ps_urls,
            additional_instruction=(
                "Focus on whether the page announces a State of Play and explicitly lists the date as February 12, 2026 (Thursday). "
                "If the page clearly announces the event on that date, count as supported."
            )
        )
    else:
        # No official URL provided -> fail this leaf explicitly
        leaf_ref_supports.score = 0.0
        leaf_ref_supports.status = "failed"


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
    # Initialize evaluator (root is non-critical by default; we enforce critical children)
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

    # Extract structured information from the answer
    extraction = await evaluator.extract(
        prompt=prompt_extract_schedule(),
        template_class=ScheduleExtraction,
        extraction_name="schedule_extraction"
    )

    # Add ground truth info for transparency
    evaluator.add_ground_truth({
        "expected_steam": EXPECTED_STEAM,
        "expected_state_of_play": EXPECTED_SOP,
        "period": "Feb 10–12, 2026"
    })

    # Build verification subtrees (each top-level node marked critical within its subtree)
    await build_steam_subtree(evaluator, root, extraction.steam or SteamInfo())
    await build_sop_subtree(evaluator, root, extraction.state_of_play or StateOfPlayInfo())

    # Return evaluation summary
    return evaluator.get_summary()