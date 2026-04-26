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
TASK_ID = "nyc_theater_1920s"
TASK_DESCRIPTION = """
Identify a historic theater in New York City that opened during the 1920s and has a seating capacity between 2,800 and 3,000. The theater must currently be operational and host live music concerts. Provide the theater's name, exact opening year, current seating capacity, and its street address in New York City.
""".strip()


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class TheaterExtraction(BaseModel):
    # Core fields required by the task
    name: Optional[str] = None
    opening_year: Optional[str] = None  # keep as string to be flexible (e.g., "1921", "1921 (opened)")
    capacity: Optional[str] = None      # keep as string to allow variants (e.g., "2,900", "about 2,900")
    address: Optional[str] = None

    # Source URLs (must be present in the answer text)
    primary_urls: List[str] = Field(default_factory=list)          # main references (official site, Wikipedia, etc.)
    opening_year_sources: List[str] = Field(default_factory=list)  # URLs supporting opening year
    capacity_sources: List[str] = Field(default_factory=list)      # URLs supporting capacity
    address_sources: List[str] = Field(default_factory=list)       # URLs supporting address / location
    live_music_sources: List[str] = Field(default_factory=list)    # URLs showing live music is hosted
    events_urls: List[str] = Field(default_factory=list)           # URLs for recent/upcoming event listings


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_theater_info() -> str:
    return """
    Extract the theater details exactly as provided in the answer. Do not invent any information.
    Return a single JSON object with the following fields:
    - name: The theater's name (string). If missing, null.
    - opening_year: The exact opening year stated (string). If missing, null.
    - capacity: The current seating capacity value as written (string). If missing, null.
    - address: The street address in New York City (string). If missing, null.

    Also extract URL sources exactly as they appear in the answer (plain links or markdown links), placing each URL in the most relevant list(s):
    - primary_urls: Main reference links for the theater (official site, Wikipedia, venue profile).
    - opening_year_sources: Links that support the theater's opening year.
    - capacity_sources: Links that support the current seating capacity.
    - address_sources: Links that support the NYC street address or location.
    - live_music_sources: Links showing the theater currently hosts live music concerts (e.g., 'Concerts', 'Events', 'Music' pages).
    - events_urls: Links to pages listing recent or upcoming live music events at the venue.

    Rules:
    - Include only URLs explicitly present in the answer text.
    - Do not deduplicate between lists; if a URL supports multiple facts, it can appear in multiple lists.
    - If a list has no URLs mentioned in the answer, return an empty list for that list.
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _dedupe_urls(urls: List[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for u in urls:
        if not u:
            continue
        u_stripped = u.strip()
        if not u_stripped:
            continue
        if u_stripped not in seen:
            seen.add(u_stripped)
            out.append(u_stripped)
    return out


def _merge_sources(*url_lists: List[str]) -> List[str]:
    merged: List[str] = []
    for lst in url_lists:
        merged.extend(lst or [])
    return _dedupe_urls(merged)


def _nonempty(s: Optional[str]) -> bool:
    return bool(s and s.strip())


# --------------------------------------------------------------------------- #
# Verification tree construction and checks                                   #
# --------------------------------------------------------------------------- #
async def build_and_verify_theater(evaluator: Evaluator, root, info: TheaterExtraction) -> None:
    # Parent node aggregating all criteria (set non-critical to allow partial credit where appropriate)
    main_node = evaluator.add_parallel(
        id="Theater_Identification",
        desc="Identifies a theater that satisfies all specified criteria",
        parent=root,
        critical=False
    )

    # 1) Theater Name Provided (Critical existence check)
    evaluator.add_custom_node(
        result=_nonempty(info.name),
        id="Theater_Name_Provided",
        desc="The theater's name is provided",
        parent=main_node,
        critical=True
    )

    # 2) Geographic Location (Critical group)
    geo_node = evaluator.add_parallel(
        id="Geographic_Location",
        desc="Verifies the theater is located in New York City",
        parent=main_node,
        critical=True
    )

    # 2.1) Address Provided (Critical existence check)
    evaluator.add_custom_node(
        result=_nonempty(info.address),
        id="Address_Provided",
        desc="Specific street address in New York City is provided",
        parent=geo_node,
        critical=True
    )

    # 2.2) NYC Location Confirmed (Critical, verify with URLs)
    nyc_node = evaluator.add_leaf(
        id="NYC_Location_Confirmed",
        desc="Theater is confirmed to be in New York City",
        parent=geo_node,
        critical=True
    )
    nyc_sources = _merge_sources(info.address_sources, info.primary_urls)
    theater_name = info.name or "the theater"
    nyc_claim = f"The venue named '{theater_name}' is located in New York City, New York (NYC)."
    await evaluator.verify(
        claim=nyc_claim,
        node=nyc_node,
        sources=nyc_sources,
        additional_instruction=(
            "Confirm that the venue is in New York City. Accept if the page clearly indicates "
            "'New York, NY' or a NYC borough (Manhattan, Brooklyn, Queens, The Bronx, Staten Island). "
            "If the venue is in a different city (e.g., Newark, Jersey City), mark as not supported."
        )
    )

    # 3) Historic Opening Period (Critical group)
    opening_node = evaluator.add_parallel(
        id="Historic_Opening_Period",
        desc="Verifies the theater opened in the 1920s",
        parent=main_node,
        critical=True
    )

    # 3.1) Opening Year Provided (Critical existence check)
    evaluator.add_custom_node(
        result=_nonempty(info.opening_year),
        id="Opening_Year_Provided",
        desc="Exact opening year is provided",
        parent=opening_node,
        critical=True
    )

    # 3.2) Opened In 1920s (Critical, verify with URLs)
    opened_1920s_node = evaluator.add_leaf(
        id="Opened_In_1920s",
        desc="Theater opening year is between 1920 and 1929 inclusive",
        parent=opening_node,
        critical=True
    )
    opening_sources = _merge_sources(info.opening_year_sources, info.primary_urls)
    year_text = info.opening_year or "UNKNOWN"
    opened_claim = (
        f"The venue '{theater_name}' opened in {year_text}, and that year lies between 1920 and 1929 inclusive."
    )
    await evaluator.verify(
        claim=opened_claim,
        node=opened_1920s_node,
        sources=opening_sources,
        additional_instruction=(
            "Use the webpage(s) to confirm the initial opening year (not a later renovation or reopening). "
            "If multiple dates are shown, prefer the original first opening year. "
            "Then check if that year is within 1920–1929 (inclusive)."
        )
    )

    # 4) Capacity Requirements (Critical group)
    capacity_node = evaluator.add_parallel(
        id="Capacity_Requirements",
        desc="Verifies the theater's seating capacity meets the specified range",
        parent=main_node,
        critical=True
    )

    # 4.1) Capacity Value Provided (Critical existence check)
    evaluator.add_custom_node(
        result=_nonempty(info.capacity),
        id="Capacity_Value_Provided",
        desc="Current seating capacity value is provided",
        parent=capacity_node,
        critical=True
    )

    # 4.2) Capacity In Range (Critical, verify with URLs)
    cap_in_range_node = evaluator.add_leaf(
        id="Capacity_In_Range",
        desc="Theater capacity is between 2,800 and 3,000 seats inclusive",
        parent=capacity_node,
        critical=True
    )
    capacity_sources = _merge_sources(info.capacity_sources, info.primary_urls)
    capacity_text = info.capacity or "UNKNOWN"
    capacity_claim = (
        f"The current seating capacity of '{theater_name}' is {capacity_text} seats, "
        "and this number is between 2,800 and 3,000 inclusive."
    )
    await evaluator.verify(
        claim=capacity_claim,
        node=cap_in_range_node,
        sources=capacity_sources,
        additional_instruction=(
            "Verify the CURRENT seating capacity from the page(s). Accept minor formatting (commas/spaces). "
            "If a range or multiple figures are given, use the 'current' or most authoritative figure. "
            "Then confirm it lies within [2800, 3000]."
        )
    )

    # 5) Current Operational Status (group set non-critical due to containing a non-critical sub-item)
    ops_node = evaluator.add_parallel(
        id="Current_Operational_Status",
        desc="Verifies the theater currently hosts live music performances",
        parent=main_node,
        critical=False
    )

    # 5.1) Hosts Live Music (Critical leaf under a non-critical group)
    live_music_node = evaluator.add_leaf(
        id="Hosts_Live_Music",
        desc="Theater is confirmed to currently host live music concerts",
        parent=ops_node,
        critical=True
    )
    live_music_sources = _merge_sources(info.live_music_sources, info.events_urls, info.primary_urls)
    live_music_claim = (
        f"The venue '{theater_name}' currently hosts live music concerts (bands, solo artists, or similar)."
    )
    await evaluator.verify(
        claim=live_music_claim,
        node=live_music_node,
        sources=live_music_sources,
        additional_instruction=(
            "Look for explicit evidence that the venue hosts live music concerts (not just theatrical plays). "
            "Accept artist tour dates, concert calendars, or official event listings mentioning 'concert', 'live music', "
            "or artist names. If only theatrical plays are shown with no live music concerts, mark as not supported."
        )
    )

    # 5.2) Recent or Upcoming Events Documented (Non-critical)
    recent_events_node = evaluator.add_leaf(
        id="Recent_Events_Documented",
        desc="Recent or upcoming music events are documented",
        parent=ops_node,
        critical=False
    )
    recent_events_claim = (
        f"There are recent or upcoming live music events documented for '{theater_name}'."
    )
    await evaluator.verify(
        claim=recent_events_claim,
        node=recent_events_node,
        sources=live_music_sources,
        additional_instruction=(
            "Prefer pages that show dates (recent past ~18 months or upcoming) for live music concerts at this venue. "
            "If the page lists music events without clear dates, you may still pass if it clearly indicates ongoing schedules."
        )
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
    Evaluate an answer for the NYC 1920s theater task.
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
        default_model=model,
    )

    # Extract structured theater info from the answer
    info: TheaterExtraction = await evaluator.extract(
        prompt=prompt_extract_theater_info(),
        template_class=TheaterExtraction,
        extraction_name="theater_extraction"
    )

    # Build verification tree and perform checks
    await build_and_verify_theater(evaluator, root, info)

    # Return the evaluation summary
    return evaluator.get_summary()