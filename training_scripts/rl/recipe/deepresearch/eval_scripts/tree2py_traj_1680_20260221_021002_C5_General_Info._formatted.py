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
TASK_ID = "milestones_2025_2026"
TASK_DESCRIPTION = """
For a comprehensive article about major entertainment and travel milestones during the 2025-2026 season, provide detailed information about the following four specific items:

1. MLB Award Winner: Identify the New York Yankees player who won the 2025 American League Silver Slugger Award as a second baseman. Provide: (a) the player's full name, (b) the specific position for which the award was won, (c) the number of home runs the player hit during the 2025 season, (d) whether this was the player's first career Silver Slugger Award, and (e) a reference URL.

2. Major Awards Ceremony: Provide details about the 68th Grammy Awards ceremony held in early 2026. Include: (a) the exact date of the ceremony, (b) the broadcast start times in both Eastern Time (ET) and Pacific Time (PT), (c) the specific venue name and the city where it was held, (d) the name of the host, and (e) a reference URL.

3. New Store Opening: Provide information about Buc-ee's first location in Ohio, which opened in Huber Heights in 2026. Include: (a) the complete street address (street number, street name, city, and state), (b) the major road intersection where it is located (identify both the Interstate and State Route), (c) the exact date of the grand opening, (d) the specific time of day the grand opening began, and (e) a reference URL.

4. Streaming Series Milestone: Provide details about Season 2 of Ted Danson's Netflix comedy series that premiered in late 2025. Include: (a) the full title of the series, (b) the season number, (c) the exact premiere date of Season 2, (d) the streaming platform where it is available, and (e) a reference URL.
""".strip()


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class MLBInfo(BaseModel):
    player_name: Optional[str] = None
    award_position: Optional[str] = None
    home_run_count_2025: Optional[str] = None
    first_career_silver_slugger: Optional[str] = None  # yes/no/true/false or descriptive text
    reference_url: Optional[str] = None


class GrammyInfo(BaseModel):
    ceremony_date: Optional[str] = None
    broadcast_time_et: Optional[str] = None
    broadcast_time_pt: Optional[str] = None
    venue_name: Optional[str] = None
    venue_city: Optional[str] = None
    host_name: Optional[str] = None
    reference_url: Optional[str] = None


class StoreInfo(BaseModel):
    full_address: Optional[str] = None
    intersection_interstate: Optional[str] = None   # e.g., "I-70"
    intersection_state_route: Optional[str] = None  # e.g., "State Route 235" / "SR-235"
    grand_opening_date: Optional[str] = None
    opening_time: Optional[str] = None
    reference_url: Optional[str] = None


class SeriesInfo(BaseModel):
    series_title: Optional[str] = None
    season_number: Optional[str] = None
    premiere_date: Optional[str] = None
    streaming_platform: Optional[str] = None
    reference_url: Optional[str] = None


class MilestonesExtraction(BaseModel):
    mlb_award_winner: Optional[MLBInfo] = None
    awards_ceremony: Optional[GrammyInfo] = None
    store_opening: Optional[StoreInfo] = None
    streaming_series: Optional[SeriesInfo] = None


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_milestones() -> str:
    return """
Extract the required fields for each of the four items from the provided answer text. Return a single JSON object with the following structure and fields. If any field is not found in the answer, return null for that field. Only extract URLs explicitly present in the answer.

Structure:
{
  "mlb_award_winner": {
    "player_name": string | null,
    "award_position": string | null,  // e.g., "second baseman", "2B"
    "home_run_count_2025": string | null,  // keep as string, may be a number like "23"
    "first_career_silver_slugger": string | null,  // indicate yes/no/true/false or a short phrase
    "reference_url": string | null
  },
  "awards_ceremony": {
    "ceremony_date": string | null,  // e.g., "February 1, 2026"
    "broadcast_time_et": string | null,  // e.g., "8:00 PM ET"
    "broadcast_time_pt": string | null,  // e.g., "5:00 PM PT"
    "venue_name": string | null,  // e.g., "Crypto.com Arena"
    "venue_city": string | null,  // e.g., "Los Angeles"
    "host_name": string | null,
    "reference_url": string | null
  },
  "store_opening": {
    "full_address": string | null,  // include street number, street name, city, state (and ZIP if given)
    "intersection_interstate": string | null,  // e.g., "I-70"
    "intersection_state_route": string | null, // e.g., "State Route 235" or "SR-235"
    "grand_opening_date": string | null,  // e.g., "June 15, 2026"
    "opening_time": string | null,  // e.g., "6:00 AM"
    "reference_url": string | null
  },
  "streaming_series": {
    "series_title": string | null,   // full title of Ted Danson's Netflix comedy series
    "season_number": string | null,  // e.g., "2"
    "premiere_date": string | null,  // exact date for Season 2 premiere
    "streaming_platform": string | null, // e.g., "Netflix"
    "reference_url": string | null
  }
}

Guidelines:
- Do not infer facts; only extract exactly what is present in the answer.
- For times, keep the format as provided (e.g., "8 PM ET", "8:00 p.m. ET"); do not normalize.
- For the MLB position, accept variants like "second baseman", "2B", or "second base".
- For yes/no flags, return the literal string from the answer (e.g., "yes", "no", "true", "false", or a descriptive phrase if used).
- For each 'reference_url', include exactly one URL if the answer provides one clearly; if multiple are given, prefer the most relevant/official one explicitly tied to that item in the answer; if none are given, return null.
""".strip()


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _url_present(u: Optional[str]) -> bool:
    if not u:
        return False
    s = u.strip()
    return s.startswith("http://") or s.startswith("https://")


def _truthy_string(s: Optional[str]) -> Optional[bool]:
    if s is None:
        return None
    v = s.strip().lower()
    yes_set = {"yes", "true", "y", "1", "first", "first career", "first-ever", "first ever"}
    no_set = {"no", "false", "n", "0", "second", "third", "not first"}
    if v in yes_set:
        return True
    if v in no_set:
        return False
    # Unknown wording; return None to avoid forcing a direction
    return None


# --------------------------------------------------------------------------- #
# Verification builders                                                       #
# --------------------------------------------------------------------------- #
async def verify_mlb_award_winner(evaluator: Evaluator, parent_node, mlb: Optional[MLBInfo]) -> None:
    agg = evaluator.add_parallel(
        id="MLB_Award_Winner",
        desc="Information about the New York Yankees player who won the 2025 AL Silver Slugger Award as a second baseman",
        parent=parent_node,
        critical=True
    )

    # URL presence (critical sibling gating)
    ref_url = mlb.reference_url if mlb else None
    evaluator.add_custom_node(
        result=_url_present(ref_url),
        id="mlb_reference_url",
        desc="A valid reference URL supporting the information is provided",
        parent=agg,
        critical=True
    )

    # Player name
    name_node = evaluator.add_leaf(
        id="mlb_player_name",
        desc="The full name of the player is provided",
        parent=agg,
        critical=True
    )
    name_val = mlb.player_name if mlb else None
    claim_name = f"According to this page, the New York Yankees player who won the 2025 American League Silver Slugger Award as a second baseman is {name_val}."
    # Award position
    pos_node = evaluator.add_leaf(
        id="mlb_award_position",
        desc="The specific position for which the award was won is identified as second baseman",
        parent=agg,
        critical=True
    )
    pos_val = mlb.award_position if mlb else None
    claim_pos = f"The page states that {name_val} won the 2025 American League Silver Slugger Award specifically as a second baseman (2B)."

    # Home run count (2025)
    hr_node = evaluator.add_leaf(
        id="mlb_home_run_count",
        desc="The number of home runs hit by the player in the 2025 season is provided",
        parent=agg,
        critical=True
    )
    hr_val = mlb.home_run_count_2025 if mlb else None
    claim_hr = f"During the 2025 MLB season, {name_val} hit {hr_val} home runs."

    # First career or not
    career_node = evaluator.add_leaf(
        id="mlb_career_first",
        desc="It is indicated whether this was the player's first Silver Slugger Award",
        parent=agg,
        critical=True
    )
    tf = _truthy_string(mlb.first_career_silver_slugger if mlb else None)
    if tf is True:
        claim_first = f"This page indicates that the 2025 award was {name_val}'s first career Silver Slugger Award."
    elif tf is False:
        claim_first = f"This page indicates that the 2025 award was NOT {name_val}'s first career Silver Slugger Award."
    else:
        # Fallback claim if unknown wording—ask page to confirm either way explicitly
        claim_first = f"This page states explicitly whether the 2025 award was {name_val}'s first career Silver Slugger Award."

    await evaluator.batch_verify([
        (
            claim_name,
            ref_url,
            name_node,
            "Allow minor name formatting differences. Ensure the page is about the 2025 AL Silver Slugger for second base and the New York Yankees player."
        ),
        (
            claim_pos,
            ref_url,
            pos_node,
            "Treat 'second baseman', 'second base', and '2B' as equivalent."
        ),
        (
            claim_hr,
            ref_url,
            hr_node,
            "Verify the 2025 season home run total for the named player. Accept minor phrasing differences but the number must match."
        ),
        (
            claim_first,
            ref_url,
            career_node,
            "Verify whether the page indicates this was the player's first career Silver Slugger. Accept clear statements like 'first', 'second', etc."
        ),
    ])


async def verify_awards_ceremony(evaluator: Evaluator, parent_node, grammy: Optional[GrammyInfo]) -> None:
    agg = evaluator.add_parallel(
        id="Awards_Ceremony",
        desc="Information about the 68th Grammy Awards ceremony held in early 2026",
        parent=parent_node,
        critical=True
    )

    ref_url = grammy.reference_url if grammy else None
    evaluator.add_custom_node(
        result=_url_present(ref_url),
        id="awards_reference_url",
        desc="A valid reference URL supporting the information is provided",
        parent=agg,
        critical=True
    )

    # Ceremony date
    date_node = evaluator.add_leaf(
        id="ceremony_date",
        desc="The exact date of the ceremony (February 1, 2026) is provided",
        parent=agg,
        critical=True
    )
    date_val = grammy.ceremony_date if grammy else None
    claim_date = f"The 68th Grammy Awards ceremony took place on {date_val}."

    # Broadcast time ET/PT
    time_node = evaluator.add_leaf(
        id="broadcast_time",
        desc="The broadcast start time in ET and PT is provided",
        parent=agg,
        critical=True
    )
    et = grammy.broadcast_time_et if grammy else None
    pt = grammy.broadcast_time_pt if grammy else None
    claim_time = f"The broadcast started at {et} ET ({pt} PT)."

    # Venue (name + city)
    venue_node = evaluator.add_leaf(
        id="venue",
        desc="The specific venue name and location city are provided",
        parent=agg,
        critical=True
    )
    venue_name = grammy.venue_name if grammy else None
    venue_city = grammy.venue_city if grammy else None
    claim_venue = f"The ceremony was held at {venue_name} in {venue_city}."

    # Host
    host_node = evaluator.add_leaf(
        id="host_name",
        desc="The name of the ceremony host is provided",
        parent=agg,
        critical=True
    )
    host_name = grammy.host_name if grammy else None
    claim_host = f"The ceremony was hosted by {host_name}."

    await evaluator.batch_verify([
        (
            claim_date,
            ref_url,
            date_node,
            "Verify the exact ceremony date for the 68th Grammy Awards. Accept reasonable date formatting like 'Feb. 1, 2026'."
        ),
        (
            claim_time,
            ref_url,
            time_node,
            "Verify the telecast start times in ET and PT. Accept variations like '8 pm ET / 5 pm PT' or '8:00 p.m. ET'."
        ),
        (
            claim_venue,
            ref_url,
            venue_node,
            "Verify the venue name and city for the ceremony. Allow minor name variations (e.g., abbreviations)."
        ),
        (
            claim_host,
            ref_url,
            host_node,
            "Verify the host name for the 68th Grammy Awards ceremony."
        ),
    ])


async def verify_store_opening(evaluator: Evaluator, parent_node, store: Optional[StoreInfo]) -> None:
    agg = evaluator.add_parallel(
        id="Store_Opening",
        desc="Information about Buc-ee's first Ohio location in Huber Heights",
        parent=parent_node,
        critical=True
    )

    ref_url = store.reference_url if store else None
    evaluator.add_custom_node(
        result=_url_present(ref_url),
        id="store_reference_url",
        desc="A valid reference URL supporting the information is provided",
        parent=agg,
        critical=True
    )

    # Full address
    addr_node = evaluator.add_leaf(
        id="full_address",
        desc="The complete street address including street number, street name, city, and state is provided",
        parent=agg,
        critical=True
    )
    addr = store.full_address if store else None
    claim_addr = f"Buc-ee's first Ohio location in Huber Heights has the address {addr}."

    # Intersection (Interstate and State Route)
    inter_node = evaluator.add_leaf(
        id="intersection_description",
        desc="The major road intersection (Interstate and State Route) is identified",
        parent=agg,
        critical=True
    )
    interstate = store.intersection_interstate if store else None
    state_route = store.intersection_state_route if store else None
    claim_intersection = f"The Huber Heights Buc-ee's is located near the intersection of {interstate} and {state_route}."

    # Grand opening date
    date_node = evaluator.add_leaf(
        id="grand_opening_date",
        desc="The exact date of the grand opening is provided",
        parent=agg,
        critical=True
    )
    date_val = store.grand_opening_date if store else None
    claim_date = f"The grand opening date was {date_val}."

    # Opening time
    time_node = evaluator.add_leaf(
        id="opening_time",
        desc="The specific time of day when the grand opening begins is provided",
        parent=agg,
        critical=True
    )
    time_val = store.opening_time if store else None
    claim_time = f"The grand opening began at {time_val}."

    await evaluator.batch_verify([
        (
            claim_addr,
            ref_url,
            addr_node,
            "Verify the complete street address for the Huber Heights Buc-ee's. Accept standard formatting variations."
        ),
        (
            claim_intersection,
            ref_url,
            inter_node,
            "Verify the major intersection near the store. Treat 'State Route 235', 'SR-235', and 'OH-235' as equivalent."
        ),
        (
            claim_date,
            ref_url,
            date_node,
            "Verify the exact grand opening date."
        ),
        (
            claim_time,
            ref_url,
            time_node,
            "Verify the stated opening time for the grand opening event. Accept minor formatting differences (e.g., '6 AM' vs '6:00 a.m.')."
        ),
    ])


async def verify_streaming_series(evaluator: Evaluator, parent_node, series: Optional[SeriesInfo]) -> None:
    agg = evaluator.add_parallel(
        id="Streaming_Series",
        desc="Information about Ted Danson's Netflix series 'A Man on the Inside' Season 2",
        parent=parent_node,
        critical=True
    )

    ref_url = series.reference_url if series else None
    evaluator.add_custom_node(
        result=_url_present(ref_url),
        id="series_reference_url",
        desc="A valid reference URL supporting the information is provided",
        parent=agg,
        critical=True
    )

    # Series title
    title_node = evaluator.add_leaf(
        id="series_title",
        desc="The full title of the series is provided",
        parent=agg,
        critical=True
    )
    title_val = series.series_title if series else None
    claim_title = f"Ted Danson's Netflix comedy series is titled '{title_val}'."

    # Season number
    season_node = evaluator.add_leaf(
        id="season_number",
        desc="The season number (Season 2) is specified",
        parent=agg,
        critical=True
    )
    season_val = series.season_number if series else None
    claim_season = f"The page refers to Season {season_val} of '{title_val}'."

    # Premiere date
    prem_node = evaluator.add_leaf(
        id="premiere_date",
        desc="The exact premiere date of Season 2 is provided",
        parent=agg,
        critical=True
    )
    prem_val = series.premiere_date if series else None
    claim_prem = f"Season {season_val} of '{title_val}' premiered on {prem_val}."

    # Streaming platform
    plat_node = evaluator.add_leaf(
        id="streaming_platform",
        desc="The streaming platform where it is available is identified",
        parent=agg,
        critical=True
    )
    plat_val = series.streaming_platform if series else None
    claim_platform = f"The series is available on {plat_val}."

    await evaluator.batch_verify([
        (
            claim_title,
            ref_url,
            title_node,
            "Verify the series title for Ted Danson's Netflix comedy series. Allow minor punctuation or article variations."
        ),
        (
            claim_season,
            ref_url,
            season_node,
            "Verify that the page clearly refers to the specified season number for the series."
        ),
        (
            claim_prem,
            ref_url,
            prem_node,
            "Verify the exact premiere date of the specified season. Accept reasonable date formatting variations."
        ),
        (
            claim_platform,
            ref_url,
            plat_node,
            "Verify the streaming platform where the series/season is available."
        ),
    ])


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
    Evaluate an answer for the 2025-2026 entertainment and travel milestones task.
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

    # Extract structured data
    extracted = await evaluator.extract(
        prompt=prompt_extract_milestones(),
        template_class=MilestonesExtraction,
        extraction_name="milestones_extraction"
    )

    # Create a critical top-level aggregator to reflect task-wide criticality
    top = evaluator.add_parallel(
        id="All_Milestones",
        desc="Evaluate all four 2025-2026 entertainment and travel milestones with their required attributes",
        parent=root,
        critical=True
    )

    # Verify each category under the critical top-level node
    await verify_mlb_award_winner(evaluator, top, extracted.mlb_award_winner)
    await verify_awards_ceremony(evaluator, top, extracted.awards_ceremony)
    await verify_store_opening(evaluator, top, extracted.store_opening)
    await verify_streaming_series(evaluator, top, extracted.streaming_series)

    return evaluator.get_summary()