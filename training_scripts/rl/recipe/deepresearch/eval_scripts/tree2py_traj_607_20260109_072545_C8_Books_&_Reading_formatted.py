import asyncio
import logging
from datetime import datetime, date
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "us_book_festivals_2024_seasons_states"
TASK_DESCRIPTION = (
    "Identify four major book festivals that took place in the United States during 2024. "
    "Each festival must be from a different U.S. state, and each must have occurred during a different season of the year. "
    "For each festival, provide the following information: the official festival name, the exact dates (including month and specific days), "
    "the city where it was held, and the state. Ensure that one festival occurred in winter (December 2023 - February 2024), "
    "one in spring (March - May 2024), one in summer (June - August 2024), and one in fall (September - November 2024). "
    "Include a verifiable URL reference for each festival."
)

# Season boundaries (inclusive)
WINTER_START = date(2023, 12, 1)
WINTER_END = date(2024, 2, 29)  # 2024 is leap year
SPRING_START = date(2024, 3, 1)
SPRING_END = date(2024, 5, 31)
SUMMER_START = date(2024, 6, 1)
SUMMER_END = date(2024, 8, 31)
FALL_START = date(2024, 9, 1)
FALL_END = date(2024, 11, 30)

SEASON_META = {
    "winter": {
        "label": "Winter",
        "desc": "A book festival that took place in Winter (December 2023 - February 2024)",
        "start": WINTER_START,
        "end": WINTER_END,
    },
    "spring": {
        "label": "Spring",
        "desc": "A book festival that took place in Spring (March - May 2024)",
        "start": SPRING_START,
        "end": SPRING_END,
    },
    "summer": {
        "label": "Summer",
        "desc": "A book festival that took place in Summer (June - August 2024)",
        "start": SUMMER_START,
        "end": SUMMER_END,
    },
    "fall": {
        "label": "Fall",
        "desc": "A book festival that took place in Fall (September - November 2024)",
        "start": FALL_START,
        "end": FALL_END,
    },
}

# 50 U.S. states map (abbr -> full)
US_STATE_ABBR_TO_FULL = {
    "AL": "Alabama", "AK": "Alaska", "AZ": "Arizona", "AR": "Arkansas", "CA": "California",
    "CO": "Colorado", "CT": "Connecticut", "DE": "Delaware", "FL": "Florida", "GA": "Georgia",
    "HI": "Hawaii", "ID": "Idaho", "IL": "Illinois", "IN": "Indiana", "IA": "Iowa",
    "KS": "Kansas", "KY": "Kentucky", "LA": "Louisiana", "ME": "Maine", "MD": "Maryland",
    "MA": "Massachusetts", "MI": "Michigan", "MN": "Minnesota", "MS": "Mississippi", "MO": "Missouri",
    "MT": "Montana", "NE": "Nebraska", "NV": "Nevada", "NH": "New Hampshire", "NJ": "New Jersey",
    "NM": "New Mexico", "NY": "New York", "NC": "North Carolina", "ND": "North Dakota", "OH": "Ohio",
    "OK": "Oklahoma", "OR": "Oregon", "PA": "Pennsylvania", "RI": "Rhode Island", "SC": "South Carolina",
    "SD": "South Dakota", "TN": "Tennessee", "TX": "Texas", "UT": "Utah", "VT": "Vermont",
    "VA": "Virginia", "WA": "Washington", "WV": "West Virginia", "WI": "Wisconsin", "WY": "Wyoming",
}
US_STATE_FULL_TO_ABBR = {v.lower(): k for k, v in US_STATE_ABBR_TO_FULL.items()}
US_STATE_FULL_SET = set(US_STATE_ABBR_TO_FULL.values())


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class Festival(BaseModel):
    """One festival entry."""
    name: Optional[str] = None
    start_date: Optional[str] = None  # ISO format YYYY-MM-DD if possible; if single-day, set same in start and end
    end_date: Optional[str] = None    # ISO format YYYY-MM-DD if possible
    date_text: Optional[str] = None   # Original date text from the answer
    city: Optional[str] = None
    state: Optional[str] = None       # Full state name (preferred) or 2-letter abbreviation
    urls: List[str] = Field(default_factory=list)  # URLs cited for this festival


class FestivalsBySeason(BaseModel):
    """Festivals extracted from the answer, one per season where available."""
    winter: Optional[Festival] = None
    spring: Optional[Festival] = None
    summer: Optional[Festival] = None
    fall: Optional[Festival] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_festivals_by_season() -> str:
    return """
    Extract at most one U.S. book or literary festival for each season of 2024 as presented in the answer text.
    Seasons and acceptable date windows:
    - winter: December 2023 through February 2024 (inclusive)
    - spring: March through May 2024 (inclusive)
    - summer: June through August 2024 (inclusive)
    - fall: September through November 2024 (inclusive)

    Selection rules (very important):
    - If the answer mentions multiple candidate festivals for the same season, pick the first one that fits the date window.
    - If the answer does not clearly state a festival for a season, return null for that season.
    - Do not invent or infer anything not clearly present in the answer text.

    For each extracted festival, return these fields:
    - name: official festival name as stated in the answer
    - start_date: the start date in ISO format YYYY-MM-DD if available; if a single-day event, use the same date for start_date and end_date; if the answer only provides month/day/year text, convert to ISO
    - end_date: the end date in ISO format YYYY-MM-DD if available; if the event is one day, use the same ISO date as start_date
    - date_text: the exact date text as written in the answer (e.g., "April 6–7, 2024" or "Jan 27, 2024")
    - city: city where it took place, if present
    - state: the U.S. state (full name preferred; two-letter abbreviation acceptable if that is what the answer uses)
    - urls: all URLs explicitly cited in the answer that directly support this festival; include official festival pages or reputable news coverage when present; if multiple URLs are present for this festival, include them all

    Return a JSON object with keys: winter, spring, summer, fall.
    If a field is missing in the answer, set it to null (or [] for urls).
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def _parse_iso_date(s: Optional[str]) -> Optional[date]:
    if not s:
        return None
    try:
        return datetime.strptime(s.strip(), "%Y-%m-%d").date()
    except Exception:
        return None


def _date_range_inclusive_within(start: Optional[date], end: Optional[date], low: date, high: date) -> bool:
    if (start is None) or (end is None):
        return False
    return (low <= start <= high) and (low <= end <= high)


def _is_valid_url(url: str) -> bool:
    if not url:
        return False
    u = url.strip()
    if not (u.startswith("http://") or u.startswith("https://")):
        return False
    return "." in u and " " not in u


def _normalize_us_state(state_raw: Optional[str]) -> Optional[str]:
    if not state_raw:
        return None
    s = state_raw.strip()
    if not s:
        return None
    s_no_dots = s.replace(".", "")
    s_upper = s_no_dots.upper()
    if s_upper in US_STATE_ABBR_TO_FULL:
        return US_STATE_ABBR_TO_FULL[s_upper]
    s_lower = s_no_dots.lower()
    if s_lower in US_STATE_FULL_TO_ABBR:
        full = s_no_dots.title()
        # Fix common title-case artifacts like "Of" -> "of"
        # But we can trust mapping by using dictionary key normalization:
        for full_name in US_STATE_FULL_SET:
            if full_name.lower() == s_lower:
                return full_name
    return None


def _urls_present(urls: List[str]) -> bool:
    return any(_is_valid_url(u) for u in urls)


def _season_desc(label: str) -> str:
    return {
        "Winter": "A book festival that took place in Winter (December 2023 - February 2024)",
        "Spring": "A book festival that took place in Spring (March - May 2024)",
        "Summer": "A book festival that took place in Summer (June - August 2024)",
        "Fall": "A book festival that took place in Fall (September - November 2024)",
    }[label]


# --------------------------------------------------------------------------- #
# Season verification                                                         #
# --------------------------------------------------------------------------- #
async def verify_one_season(
    evaluator: Evaluator,
    parent_node,
    season_key: str,
    season_info: Festival,
    window_start: date,
    window_end: date,
) -> None:
    label = SEASON_META[season_key]["label"]
    season_node = evaluator.add_parallel(
        id=f"{label}_Festival",
        desc=_season_desc(label),
        parent=parent_node,
        critical=False,
    )

    # Extract fields with safe defaults
    name = (season_info.name or "").strip() if season_info else ""
    city = (season_info.city or "").strip() if season_info else ""
    state_raw = (season_info.state or "").strip() if season_info else ""
    urls = season_info.urls if (season_info and season_info.urls) else []
    s_date = _parse_iso_date(season_info.start_date if season_info else None)
    e_date = _parse_iso_date(season_info.end_date if season_info else None)

    # 1) Name provided (critical)
    evaluator.add_custom_node(
        result=bool(name),
        id=f"{label}_Festival_Name",
        desc="Provides the official festival name",
        parent=season_node,
        critical=True,
    )

    # 2) Dates provided (critical) -- split for clarity
    dates_provided = (s_date is not None) and (e_date is not None)
    evaluator.add_custom_node(
        result=dates_provided,
        id=f"{label}_Festival_Dates_Provided",
        desc="Provides exact dates (start and end dates are specified to the day; for single-day events, start=end)",
        parent=season_node,
        critical=True,
    )

    # 3) Dates in season window (critical)
    dates_in_window = _date_range_inclusive_within(s_date, e_date, window_start, window_end)
    evaluator.add_custom_node(
        result=dates_in_window,
        id=f"{label}_Festival_Dates_In_Season",
        desc=f"Dates fall within the required season window ({window_start.isoformat()} to {window_end.isoformat()})",
        parent=season_node,
        critical=True,
    )

    # 4) City provided (critical)
    evaluator.add_custom_node(
        result=bool(city),
        id=f"{label}_Festival_City",
        desc="Provides the city where the festival was held",
        parent=season_node,
        critical=True,
    )

    # 5) State is a valid U.S. state (critical)
    normalized_state = _normalize_us_state(state_raw)
    evaluator.add_custom_node(
        result=normalized_state is not None,
        id=f"{label}_Festival_State_US",
        desc="Provides the state and it is a U.S. state (not a non-U.S. region)",
        parent=season_node,
        critical=True,
    )

    # 6) Reference URL presence (critical) — presence check separated
    urls_ok = _urls_present(urls)
    url_presence_node = evaluator.add_custom_node(
        result=urls_ok,
        id=f"{label}_Festival_Reference_URL_Present",
        desc="Provides at least one verifiable URL reference",
        parent=season_node,
        critical=True,
    )

    # 7) Reference URL is official or reputable (critical) — content-based verification
    ref_quality_node = evaluator.add_leaf(
        id=f"{label}_Festival_Reference_URL",
        desc="Provides at least one verifiable URL reference from an official festival website or reputable news source supporting the festival details",
        parent=season_node,
        critical=True,
    )
    ref_quality_claim = (
        f"At least one of these sources is either the official website for the '{name}' festival or a reputable news outlet's "
        f"coverage that confirms the festival took place and provides factual details about it."
    )
    await evaluator.verify(
        claim=ref_quality_claim,
        node=ref_quality_node,
        sources=urls,
        additional_instruction=(
            "Judge whether the source is official (e.g., the festival's own site, a city or institution page explicitly hosting the festival) "
            "or a reputable news outlet (recognized local/regional/national media). Ignore casual blogs or irrelevant pages."
        ),
    )

    # 8) Documented book/literary festival (critical) — content-based verification
    documented_node = evaluator.add_leaf(
        id=f"{label}_Festival_Documented_Book_Festival",
        desc="The provided reference(s) support that this is a real, documented literary/book festival event",
        parent=season_node,
        critical=True,
    )
    documented_claim = (
        f"The sources clearly indicate that '{name}' is a real book or literary festival (e.g., 'book festival', 'literary festival', "
        f"'festival of books', 'book fair' organized as a festival event), not merely a conference, trade expo, or a simple book sale."
    )
    await evaluator.verify(
        claim=documented_claim,
        node=documented_node,
        sources=urls,
        additional_instruction=(
            "Look for explicit wording like 'book festival', 'literary festival', 'festival of books', or similar. "
            "Confirm it is an event and not just a single author talk or a routine book sale."
        ),
    )

    # 9) Major or well-established (critical) — content-based verification
    major_node = evaluator.add_leaf(
        id=f"{label}_Festival_Major_Well_Established",
        desc="Includes verifiable support (via provided source) that the festival is major or well-established (e.g., longevity, scale, notable recognition)",
        parent=season_node,
        critical=True,
    )
    major_claim = (
        f"The sources provide evidence that the '{name}' festival is major or well-established, such as multi-year history "
        f"(annual for several years), large attendance, notable partners/sponsors, prominent authors, or recognition as a leading event."
    )
    await evaluator.verify(
        claim=major_claim,
        node=major_node,
        sources=urls,
        additional_instruction=(
            "Accept as evidence: mentions of many editions (e.g., '20th annual'), large attendance, significant media coverage, "
            "major sponsors/partners, being 'one of the largest' or widely recognized."
        ),
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
    Evaluate an answer for the US Book Festivals 2024 seasons/states task.
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

    # Extract festival info by season
    extracted: FestivalsBySeason = await evaluator.extract(
        prompt=prompt_extract_festivals_by_season(),
        template_class=FestivalsBySeason,
        extraction_name="festivals_by_season",
    )

    # Add context info for season windows
    evaluator.add_custom_info(
        info={
            "winter_window": {"start": WINTER_START.isoformat(), "end": WINTER_END.isoformat()},
            "spring_window": {"start": SPRING_START.isoformat(), "end": SPRING_END.isoformat()},
            "summer_window": {"start": SUMMER_START.isoformat(), "end": SUMMER_END.isoformat()},
            "fall_window": {"start": FALL_START.isoformat(), "end": FALL_END.isoformat()},
        },
        info_type="season_windows",
    )

    # Build season verification subtrees
    season_items = {
        "winter": extracted.winter,
        "spring": extracted.spring,
        "summer": extracted.summer,
        "fall": extracted.fall,
    }

    for skey, sdata in season_items.items():
        await verify_one_season(
            evaluator=evaluator,
            parent_node=root,
            season_key=skey,
            season_info=sdata if sdata else Festival(),
            window_start=SEASON_META[skey]["start"],
            window_end=SEASON_META[skey]["end"],
        )

    # Global state uniqueness (critical)
    # Only count valid U.S. states (normalize and ensure 4 distinct)
    normalized_states: List[str] = []
    for skey in ["winter", "spring", "summer", "fall"]:
        st = None
        fest = season_items.get(skey)
        if fest and fest.state:
            st = _normalize_us_state(fest.state)
        if st is not None:
            normalized_states.append(st)

    all_four_present = len(normalized_states) == 4
    all_distinct = len(set(normalized_states)) == 4 if all_four_present else False
    evaluator.add_custom_node(
        result=all_four_present and all_distinct,
        id="Global_State_Uniqueness",
        desc="All four festivals are located in four different U.S. states (no two festivals share the same state)",
        parent=root,
        critical=True,
    )

    # Return the evaluation summary
    return evaluator.get_summary()