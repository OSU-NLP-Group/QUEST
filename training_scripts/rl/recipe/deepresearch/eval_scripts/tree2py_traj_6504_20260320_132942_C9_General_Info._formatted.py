import asyncio
import logging
import re
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "entertainment_events_march_2026"
TASK_DESCRIPTION = """
Identify four distinct entertainment events occurring in March 2026, each from a different category, with complete and accurate details:
1) One new scripted TV series premiere (title, network/platform, exact premiere date in March 2026, time if available, genre, reliable reference URL)
2) One major streaming content release in March 2026 (title, platform, exact release date, season/episodes, release format, reliable reference URL)
3) One major theatrical movie release in March 2026 (title, theatrical release date, studio/distributor, genre, key creative, reliable reference URL)
4) One major live event in March 2026 (awards or sports) that is broadcast (name, type, broadcaster, exact date, time, venue/city, reliable reference URL)
All events must be verifiable via provided URLs and be in March 2026. Each event must be from a different category.
"""

ALLOWED_STREAMING_PLATFORMS = {
    "netflix", "prime video", "amazon prime video", "apple tv+", "apple tv plus", "hulu",
    "disney+", "disney plus", "peacock"
}

RELIABLE_DOMAIN_HINTS = [
    # Trade/entertainment outlets
    "deadline.com", "hollywoodreporter.com", "variety.com", "tvline.com", "ew.com",
    "entertainmentweekly.com", "thewrap.com",
    # Movie databases/outlets
    "boxofficemojo.com", "imdb.com", "rottentomatoes.com",
    # Networks/streamers (official)
    "netflix.com", "netflix.co", "tv.apple.com", "apple.com", "disneyplus.com", "disney.com",
    "hulu.com", "peacocktv.com", "peacock.com", "primevideo.com", "amazon.com",
    "max.com", "hbomax.com", "paramountplus.com",
    "abc.com", "cbs.com", "nbc.com", "fox.com", "fxnetworks.com", "amc.com", "amcplus.com",
    "pbs.org", "starz.com", "showtime.com", "paramount.com",
    # Studios
    "warnerbros.com", "wb.com", "universalpictures.com", "sony.com", "mgm.com", "lionsgate.com",
    # Sports/events orgs
    "espn.com", "nba.com", "nfl.com", "mlb.com", "nhl.com",
    # Award orgs
    "oscars.org", "emmys.com", "goldenglobes.com", "grammy.com"
]


# --------------------------------------------------------------------------- #
# Utility helpers                                                             #
# --------------------------------------------------------------------------- #
def _normalize_urls(urls: Optional[List[str]]) -> List[str]:
    if not urls:
        return []
    cleaned = []
    for u in urls:
        if not isinstance(u, str):
            continue
        s = u.strip()
        if not s:
            continue
        if not s.startswith("http://") and not s.startswith("https://"):
            s = "http://" + s
        cleaned.append(s)
    # Remove duplicates while preserving order
    seen = set()
    unique = []
    for u in cleaned:
        if u not in seen:
            unique.append(u)
            seen.add(u)
    return unique


def _domain_from_url(u: str) -> str:
    try:
        return urlparse(u).netloc.lower()
    except Exception:
        return ""


def any_reliable_url(urls: List[str]) -> bool:
    if not urls:
        return False
    for u in urls:
        d = _domain_from_url(u)
        for hint in RELIABLE_DOMAIN_HINTS:
            if hint in d:
                return True
    return False


def platform_allowed(platform: Optional[str]) -> bool:
    if not platform:
        return False
    p = platform.strip().lower()
    # Normalize known variants
    p = p.replace("plus", "+").replace(" +", "+").replace("tv +", "tv+").replace("tv+", "tv+")
    for allowed in ALLOWED_STREAMING_PLATFORMS:
        if allowed in p:
            return True
    return False


def looks_like_march_2026(date_text: Optional[str]) -> bool:
    if not date_text:
        return False
    s = date_text.lower().strip()
    if "2026" not in s:
        return False
    # Allow various representations of March
    march_patterns = [r"\bmar\b", r"\bmarch\b", r"2026-03", r"03/..?/2026", r"2026/03"]
    for pat in march_patterns:
        if re.search(pat, s):
            return True
    return False


def has_timezone(text: Optional[str]) -> bool:
    if not text:
        return False
    s = text.upper()
    # Common broadcast timezones
    tzs = ["ET", "PT", "CT", "MT", "GMT", "UTC", "BST", "CET", "EET", "AEDT", "AEST", "JST", "KST"]
    return any(tz in s for tz in tzs)


def reliability_additional_instruction() -> str:
    return (
        "Judge whether at least one of the provided URLs is from a reliable entertainment news outlet or "
        "an official network/studio/organizer website. Examples of reliable outlets include Deadline, "
        "The Hollywood Reporter, Variety, TVLine, Entertainment Weekly, The Wrap, Box Office Mojo, IMDb, "
        "Rotten Tomatoes, and official sites for networks/streamers/studios (e.g., netflix.com, tv.apple.com, "
        "disneyplus.com, hulu.com, peacocktv.com, abc.com, nbc.com), major sports leagues, or award organizations. "
        "You only need at least one clearly reliable link to pass. Reason about the domain name and branding."
    )


def fuzzy_name_match_instruction() -> str:
    return (
        "Allow reasonable name/title variations (case differences, punctuation, subtitles). "
        "Focus on whether the referenced page clearly corresponds to the same title/content."
    )


def date_check_instruction(prefix: str) -> str:
    return (
        f"Verify that the {prefix} date given in the answer matches the page and that the date is in March 2026. "
        "Accept various date formats (e.g., 'March 5, 2026', '2026-03-05')."
    )


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class TVSeriesPremiereEvent(BaseModel):
    title: Optional[str] = None
    network_or_platform: Optional[str] = None
    premiere_date: Optional[str] = None
    premiere_time: Optional[str] = None  # optional
    genre: Optional[str] = None  # optional
    reference_urls: List[str] = Field(default_factory=list)


class StreamingReleaseEvent(BaseModel):
    title: Optional[str] = None
    platform: Optional[str] = None
    release_date: Optional[str] = None
    season_or_episodes: Optional[str] = None  # optional text (e.g., "Season 3", "6 episodes")
    release_format: Optional[str] = None  # optional (e.g., "all at once", "weekly")
    reference_urls: List[str] = Field(default_factory=list)


class MovieReleaseEvent(BaseModel):
    title: Optional[str] = None
    theatrical_release_date: Optional[str] = None
    distributor: Optional[str] = None  # optional
    genre: Optional[str] = None  # optional
    key_creative: Optional[str] = None  # optional (director or lead actor)
    reference_urls: List[str] = Field(default_factory=list)


class LiveEventInfo(BaseModel):
    name: Optional[str] = None
    event_type: Optional[str] = None  # "award show" or "sports event"
    broadcaster: Optional[str] = None
    date: Optional[str] = None
    time: Optional[str] = None
    venue_or_city: Optional[str] = None  # optional
    reference_urls: List[str] = Field(default_factory=list)


class EntertainmentEventsExtraction(BaseModel):
    tv_series: Optional[TVSeriesPremiereEvent] = None
    streaming_release: Optional[StreamingReleaseEvent] = None
    movie_release: Optional[MovieReleaseEvent] = None
    live_event: Optional[LiveEventInfo] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_entertainment_events() -> str:
    return """
    Extract exactly one event for each category from the answer text. If multiple candidates are provided in a category, select the first one that clearly fits. If a field is not provided, set it to null. For URLs, extract actual URLs as strings.

    Categories and required JSON schema:

    tv_series:
      - title: official title of the new scripted TV series (string or null)
      - network_or_platform: the network or streaming platform (string or null)
      - premiere_date: exact premiere date in March 2026 (string or null)
      - premiere_time: premiere time if available (string or null)
      - genre: genre category (drama, comedy, limited series, etc.) if provided (string or null)
      - reference_urls: array of reference URLs confirming premiere details (array of strings, can be empty)

    streaming_release:
      - title: title of the content (string or null)
      - platform: streaming platform (string or null)
      - release_date: exact release date in March 2026 (string or null)
      - season_or_episodes: season number or episode count if applicable (string or null)
      - release_format: whether all episodes release at once or scheduled (string or null)
      - reference_urls: array of reference URLs confirming release (array of strings, can be empty)

    movie_release:
      - title: official title of the film (string or null)
      - theatrical_release_date: exact theatrical release date in March 2026 (string or null)
      - distributor: film studio or distributor if provided (string or null)
      - genre: primary genre if provided (string or null)
      - key_creative: at least one key creative (director or lead actor) if provided (string or null)
      - reference_urls: array of reference URLs confirming theatrical release (array of strings, can be empty)

    live_event:
      - name: official name of the event (string or null)
      - event_type: either "award show" or "sports event" (string or null)
      - broadcaster: network or platform broadcasting the event (string or null)
      - date: exact date in March 2026 (string or null)
      - time: broadcast time with time zone (string or null)
      - venue_or_city: venue name and/or city (string or null)
      - reference_urls: array of reference URLs confirming event details (array of strings, can be empty)
    """


# --------------------------------------------------------------------------- #
# Verification builders                                                       #
# --------------------------------------------------------------------------- #
async def verify_tv_series_premiere(evaluator: Evaluator, root) -> None:
    # Extracted data from evaluator recorded extractions (we re-extract here via model)
    # We assume extraction has been done already and stored; we will pass it in from caller
    pass


async def build_verify_tv_series_premiere(
    evaluator: Evaluator,
    parent,
    data: Optional[TVSeriesPremiereEvent],
) -> None:
    node = evaluator.add_sequential(
        id="Event_1_TV_Series_Premiere",
        desc="Identify and verify one new scripted TV series premiering in March 2026",
        parent=parent,
        critical=False,
    )

    # Normalize
    data = data or TVSeriesPremiereEvent()
    urls = _normalize_urls(data.reference_urls)

    # Required info (gate)
    required_ok = bool(data.title and data.network_or_platform and data.premiere_date and urls)
    evaluator.add_custom_node(
        result=required_ok,
        id="TV_required_info",
        desc="TV series: required info present (title, network/platform, March 2026 date, and at least one reference URL)",
        parent=node,
        critical=True,
    )

    # Identification (critical parallel)
    ident = evaluator.add_parallel(
        id="TV_Series_Identification",
        desc="Verify the show is a new scripted TV series premiering for the first time",
        parent=node,
        critical=True,
    )

    # Title verification
    leaf_title = evaluator.add_leaf(
        id="TV_Series_Name",
        desc=f"Provide the official title of the TV series: {data.title}",
        parent=ident,
        critical=True,
    )
    await evaluator.verify(
        claim=f"The official title of the TV series is '{data.title}'.",
        node=leaf_title,
        sources=urls,
        additional_instruction=fuzzy_name_match_instruction(),
    )

    # Newness verification
    leaf_new = evaluator.add_leaf(
        id="TV_Series_Newness",
        desc="Verify it is a new series (not a returning season)",
        parent=ident,
        critical=True,
    )
    await evaluator.verify(
        claim="This is a brand-new scripted television series premiering for the first time (not a returning season) in March 2026.",
        node=leaf_new,
        sources=urls,
        additional_instruction="Look for wording such as 'new series', 'series premiere', 'debut', or similar; ensure it's not season X of an existing series.",
    )

    # Scripted/format verification
    leaf_fmt = evaluator.add_leaf(
        id="TV_Series_Format",
        desc="Verify it is scripted (drama, comedy, or limited series)",
        parent=ident,
        critical=True,
    )
    genre_part = f" with genre '{data.genre}'" if data.genre else ""
    await evaluator.verify(
        claim=f"This TV series is scripted (drama/comedy/limited){genre_part}. It is not an unscripted reality, news, or sports program.",
        node=leaf_fmt,
        sources=urls,
        additional_instruction="Accept synonyms for 'limited series' and subgenres. Focus on scripted classification.",
    )

    # Details (non-critical parallel)
    details = evaluator.add_parallel(
        id="TV_Series_Details",
        desc="Provide complete and accurate broadcast/streaming details for the March 2026 premiere",
        parent=node,
        critical=False,
    )

    # Network/platform (critical within details)
    leaf_network = evaluator.add_leaf(
        id="Network_Platform",
        desc=f"Identify the network or streaming platform airing the series: {data.network_or_platform}",
        parent=details,
        critical=True,
    )
    await evaluator.verify(
        claim=f"The series will air on '{data.network_or_platform}'.",
        node=leaf_network,
        sources=urls,
        additional_instruction="Accept reasonable brand variations (e.g., 'ABC' vs 'ABC Network').",
    )

    # Exact premiere date (critical within details)
    leaf_date = evaluator.add_leaf(
        id="Exact_Premiere_Date",
        desc=f"Provide the exact premiere date in March 2026: {data.premiere_date}",
        parent=details,
        critical=True,
    )
    await evaluator.verify(
        claim=f"The premiere date is '{data.premiere_date}', and that date is in March 2026.",
        node=leaf_date,
        sources=urls,
        additional_instruction=date_check_instruction("premiere"),
    )

    # Premiere time (non-critical; optional)
    if data.premiere_time and data.premiere_time.strip():
        leaf_time = evaluator.add_leaf(
            id="Premiere_Time",
            desc=f"Provide the premiere time (if available): {data.premiere_time}",
            parent=details,
            critical=False,
        )
        await evaluator.verify(
            claim=f"The premiere time is '{data.premiere_time}'.",
            node=leaf_time,
            sources=urls,
            additional_instruction="If multiple time zones are listed, any correct mention matching the answer passes.",
        )
    else:
        evaluator.add_custom_node(
            result=True,
            id="Premiere_Time",
            desc="Premiere time is optional and not provided; treated as pass.",
            parent=details,
            critical=False,
        )

    # Genre category (non-critical; optional)
    if data.genre and data.genre.strip():
        leaf_gen = evaluator.add_leaf(
            id="Genre_Category",
            desc=f"Identify the genre category: {data.genre}",
            parent=details,
            critical=False,
        )
        await evaluator.verify(
            claim=f"The genre/category for the series is '{data.genre}'.",
            node=leaf_gen,
            sources=urls,
            additional_instruction="Accept reasonable subgenre phrasing; ensure it's consistent with being scripted.",
        )
    else:
        evaluator.add_custom_node(
            result=True,
            id="Genre_Category",
            desc="Genre not provided; optional; treated as pass.",
            parent=details,
            critical=False,
        )

    # Reference URL existence already gated; here verify reliability critically
    leaf_ref_rel = evaluator.add_leaf(
        id="Reference_URL_TV",
        desc="Provide a reliable reference URL confirming the premiere details",
        parent=details,
        critical=True,
    )
    await evaluator.verify(
        claim="At least one of the provided URLs is from a reliable entertainment outlet or the official network/streamer.",
        node=leaf_ref_rel,
        sources=urls,
        additional_instruction=reliability_additional_instruction(),
    )


async def build_verify_streaming_release(
    evaluator: Evaluator,
    parent,
    data: Optional[StreamingReleaseEvent],
) -> None:
    node = evaluator.add_sequential(
        id="Event_2_Streaming_Release",
        desc="Identify and verify one major content release on a streaming platform in March 2026",
        parent=parent,
        critical=False,
    )

    data = data or StreamingReleaseEvent()
    urls = _normalize_urls(data.reference_urls)

    # Required info (gate)
    required_ok = bool(data.title and data.platform and data.release_date and urls)
    evaluator.add_custom_node(
        result=required_ok,
        id="Streaming_required_info",
        desc="Streaming: required info present (title, platform, March 2026 date, and at least one reference URL)",
        parent=node,
        critical=True,
    )

    # Identification (critical parallel)
    ident = evaluator.add_parallel(
        id="Streaming_Content_Identification",
        desc="Verify the content is a significant release on a major streaming platform",
        parent=node,
        critical=True,
    )

    # Title
    leaf_title = evaluator.add_leaf(
        id="Content_Title",
        desc=f"Provide the official title of the streaming content: {data.title}",
        parent=ident,
        critical=True,
    )
    await evaluator.verify(
        claim=f"The official title of the streaming content is '{data.title}'.",
        node=leaf_title,
        sources=urls,
        additional_instruction=fuzzy_name_match_instruction(),
    )

    # Type (returning season/docuseries/special; not a new series)
    leaf_type = evaluator.add_leaf(
        id="Content_Type",
        desc="Verify it is a returning series season, documentary series, or special (not a new series)",
        parent=ident,
        critical=True,
    )
    await evaluator.verify(
        claim="This content is a returning series season, a documentary series, or a special event; it is NOT a brand-new scripted series premiere.",
        node=leaf_type,
        sources=urls,
        additional_instruction="Look for wording like 'Season X', 'Documentary series', 'Special'. Ensure it's not described as a 'new series' debut.",
    )

    # Platform allowed (critical, quick local check)
    evaluator.add_custom_node(
        result=platform_allowed(data.platform),
        id="Platform_Verification",
        desc=f"Verify it is on Netflix/Prime Video/Apple TV+/Hulu/Disney+/Peacock: {data.platform}",
        parent=ident,
        critical=True,
    )

    # Details (non-critical parallel)
    details = evaluator.add_parallel(
        id="Streaming_Release_Details",
        desc="Provide complete streaming release information for the March 2026 release",
        parent=node,
        critical=False,
    )

    # Streaming platform (critical within details, source-grounded)
    leaf_plat = evaluator.add_leaf(
        id="Streaming_Platform",
        desc=f"Identify the specific streaming platform: {data.platform}",
        parent=details,
        critical=True,
    )
    await evaluator.verify(
        claim=f"The content releases on '{data.platform}'.",
        node=leaf_plat,
        sources=urls,
        additional_instruction="Verify that the platform name matches what is stated on the referenced page(s).",
    )

    # Exact release date (critical)
    leaf_date = evaluator.add_leaf(
        id="Exact_Release_Date",
        desc=f"Provide the exact release date in March 2026: {data.release_date}",
        parent=details,
        critical=True,
    )
    await evaluator.verify(
        claim=f"The release date is '{data.release_date}', and that date is in March 2026.",
        node=leaf_date,
        sources=urls,
        additional_instruction=date_check_instruction("release"),
    )

    # Season/Episode info (optional)
    if data.season_or_episodes and data.season_or_episodes.strip():
        leaf_se = evaluator.add_leaf(
            id="Season_Episode_Info",
            desc=f"If applicable, provide season number or episode count: {data.season_or_episodes}",
            parent=details,
            critical=False,
        )
        await evaluator.verify(
            claim=f"The season/episode information is '{data.season_or_episodes}'.",
            node=leaf_se,
            sources=urls,
            additional_instruction="Minor formatting differences acceptable (e.g., 'Season 3' vs 'third season').",
        )
    else:
        evaluator.add_custom_node(
            result=True,
            id="Season_Episode_Info",
            desc="Season/episode info not provided; optional; treated as pass.",
            parent=details,
            critical=False,
        )

    # Release format (optional)
    if data.release_format and data.release_format.strip():
        leaf_rf = evaluator.add_leaf(
            id="Release_Format",
            desc=f"Specify if all episodes release at once or weekly/scheduled: {data.release_format}",
            parent=details,
            critical=False,
        )
        await evaluator.verify(
            claim=f"The release schedule is '{data.release_format}'.",
            node=leaf_rf,
            sources=urls,
            additional_instruction="Accept phrasing variants like 'full season at once', 'weekly rollout', etc.",
        )
    else:
        evaluator.add_custom_node(
            result=True,
            id="Release_Format",
            desc="Release format not provided; optional; treated as pass.",
            parent=details,
            critical=False,
        )

    # Reference reliability (critical)
    leaf_ref = evaluator.add_leaf(
        id="Reference_URL_Streaming",
        desc="Provide a reliable reference URL confirming the release details",
        parent=details,
        critical=True,
    )
    await evaluator.verify(
        claim="At least one of the provided URLs is from a reliable entertainment outlet or the official streamer.",
        node=leaf_ref,
        sources=urls,
        additional_instruction=reliability_additional_instruction(),
    )


async def build_verify_movie_release(
    evaluator: Evaluator,
    parent,
    data: Optional[MovieReleaseEvent],
) -> None:
    node = evaluator.add_sequential(
        id="Event_3_Movie_Release",
        desc="Identify and verify one major theatrical movie release in March 2026",
        parent=parent,
        critical=False,
    )

    data = data or MovieReleaseEvent()
    urls = _normalize_urls(data.reference_urls)

    # Required info
    required_ok = bool(data.title and data.theatrical_release_date and urls)
    evaluator.add_custom_node(
        result=required_ok,
        id="Movie_required_info",
        desc="Movie: required info present (title, theatrical release date, and at least one reference URL)",
        parent=node,
        critical=True,
    )

    # Identification (critical parallel)
    ident = evaluator.add_parallel(
        id="Movie_Identification",
        desc="Verify the film is a theatrical release opening in March 2026",
        parent=node,
        critical=True,
    )

    # Movie title
    leaf_title = evaluator.add_leaf(
        id="Movie_Title",
        desc=f"Provide the official title of the film: {data.title}",
        parent=ident,
        critical=True,
    )
    await evaluator.verify(
        claim=f"The official title of the film is '{data.title}'.",
        node=leaf_title,
        sources=urls,
        additional_instruction=fuzzy_name_match_instruction(),
    )

    # Theatrical verification
    leaf_theatrical = evaluator.add_leaf(
        id="Theatrical_Verification",
        desc="Verify it is a theatrical release (not direct-to-streaming)",
        parent=ident,
        critical=True,
    )
    await evaluator.verify(
        claim="This film has a theatrical release (in cinemas) in March 2026; it is not a direct-to-streaming release.",
        node=leaf_theatrical,
        sources=urls,
        additional_instruction="Look for phrases like 'theatrical release', 'in theaters', 'in cinemas', or similar.",
    )

    # Major release status
    leaf_major = evaluator.add_leaf(
        id="Major_Release_Status",
        desc="Verify it is a major/wide release or significant limited release",
        parent=ident,
        critical=True,
    )
    await evaluator.verify(
        claim="This release is either a wide release or a significant limited release (i.e., notable theatrical reach).",
        node=leaf_major,
        sources=urls,
        additional_instruction="Accept explicit 'wide release' notes or reputable reporting indicating significant scale.",
    )

    # Details (non-critical)
    details = evaluator.add_parallel(
        id="Movie_Release_Details",
        desc="Provide complete theatrical release information for the March 2026 release",
        parent=node,
        critical=False,
    )

    # Exact theatrical date (critical)
    leaf_date = evaluator.add_leaf(
        id="Exact_Theatrical_Date",
        desc=f"Provide the exact theatrical release date in March 2026: {data.theatrical_release_date}",
        parent=details,
        critical=True,
    )
    await evaluator.verify(
        claim=f"The theatrical release date is '{data.theatrical_release_date}', and that date is in March 2026.",
        node=leaf_date,
        sources=urls,
        additional_instruction=date_check_instruction("theatrical release"),
    )

    # Studio/Distributor (optional)
    if data.distributor and data.distributor.strip():
        leaf_dist = evaluator.add_leaf(
            id="Studio_Distributor",
            desc=f"Identify the film studio or distributor: {data.distributor}",
            parent=details,
            critical=False,
        )
        await evaluator.verify(
            claim=f"The distributor/studio is '{data.distributor}'.",
            node=leaf_dist,
            sources=urls,
            additional_instruction="Allow for common imprint/label variations under a parent studio.",
        )
    else:
        evaluator.add_custom_node(
            result=True,
            id="Studio_Distributor",
            desc="Distributor not provided; optional; treated as pass.",
            parent=details,
            critical=False,
        )

    # Movie genre (optional)
    if data.genre and data.genre.strip():
        leaf_genre = evaluator.add_leaf(
            id="Movie_Genre",
            desc=f"Identify the film's primary genre: {data.genre}",
            parent=details,
            critical=False,
        )
        await evaluator.verify(
            claim=f"The film's primary genre is '{data.genre}'.",
            node=leaf_genre,
            sources=urls,
            additional_instruction="Accept standard genre labels and close synonyms.",
        )
    else:
        evaluator.add_custom_node(
            result=True,
            id="Movie_Genre",
            desc="Genre not provided; optional; treated as pass.",
            parent=details,
            critical=False,
        )

    # Key creative (optional)
    if data.key_creative and data.key_creative.strip():
        leaf_creative = evaluator.add_leaf(
            id="Creative_Team",
            desc=f"Provide at least one key creative element (director or lead actor): {data.key_creative}",
            parent=details,
            critical=False,
        )
        await evaluator.verify(
            claim=f"A key creative element (director/lead) is '{data.key_creative}'.",
            node=leaf_creative,
            sources=urls,
            additional_instruction="Allow minor naming variations.",
        )
    else:
        evaluator.add_custom_node(
            result=True,
            id="Creative_Team",
            desc="Key creative not provided; optional; treated as pass.",
            parent=details,
            critical=False,
        )

    # Reference reliability (critical)
    leaf_ref = evaluator.add_leaf(
        id="Reference_URL_Movie",
        desc="Provide a reliable reference URL confirming the theatrical release details",
        parent=details,
        critical=True,
    )
    await evaluator.verify(
        claim="At least one of the provided URLs is from a reliable entertainment outlet or an official studio/distributor.",
        node=leaf_ref,
        sources=urls,
        additional_instruction=reliability_additional_instruction(),
    )


async def build_verify_live_event(
    evaluator: Evaluator,
    parent,
    data: Optional[LiveEventInfo],
) -> None:
    node = evaluator.add_sequential(
        id="Event_4_Live_Event",
        desc="Identify and verify one major live event (award show or sports event) in March 2026",
        parent=parent,
        critical=False,
    )

    data = data or LiveEventInfo()
    urls = _normalize_urls(data.reference_urls)

    # Required info (note: time is critical per rubric)
    required_ok = bool(data.name and data.event_type and data.broadcaster and data.date and data.time and urls)
    evaluator.add_custom_node(
        result=required_ok,
        id="Live_required_info",
        desc="Live Event: required info present (name, event_type, broadcaster, March 2026 date, time with timezone, and at least one reference URL)",
        parent=node,
        critical=True,
    )

    # Identification (critical parallel)
    ident = evaluator.add_parallel(
        id="Live_Event_Identification",
        desc="Verify the event is a live broadcast award show or major sports event",
        parent=node,
        critical=True,
    )

    # Event name
    leaf_name = evaluator.add_leaf(
        id="Event_Name",
        desc=f"Provide the official name of the live event: {data.name}",
        parent=ident,
        critical=True,
    )
    await evaluator.verify(
        claim=f"The official name of the event is '{data.name}'.",
        node=leaf_name,
        sources=urls,
        additional_instruction=fuzzy_name_match_instruction(),
    )

    # Event type
    # Local quick check to ensure event_type value is sensible
    type_allowed = (data.event_type or "").strip().lower() in {"award show", "awards show", "sports event", "sporting event"}
    evaluator.add_custom_node(
        result=type_allowed,
        id="Event_Type_Value_Check",
        desc=f"Event type value must be 'award show' or 'sports event': {data.event_type}",
        parent=ident,
        critical=True,
    )

    leaf_type = evaluator.add_leaf(
        id="Event_Type",
        desc=f"Identify if it is an award show or sports event: {data.event_type}",
        parent=ident,
        critical=True,
    )
    await evaluator.verify(
        claim=f"This event is an '{data.event_type}'.",
        node=leaf_type,
        sources=urls,
        additional_instruction="Look for phrases clearly indicating award show (e.g., 'Awards', 'Ceremony') or sporting event/tournament/game.",
    )

    # Major status
    leaf_major = evaluator.add_leaf(
        id="Major_Event_Status",
        desc="Verify it is a nationally/internationally significant event",
        parent=ident,
        critical=True,
    )
    await evaluator.verify(
        claim="This event is nationally or internationally significant.",
        node=leaf_major,
        sources=urls,
        additional_instruction="Clues: major awards (Oscars, Emmys, Grammys) or major league sports (e.g., NBA, NFL) or marquee international events.",
    )

    # Broadcast details (non-critical parallel, but with critical children)
    details = evaluator.add_parallel(
        id="Live_Event_Broadcast_Details",
        desc="Provide complete broadcast and event information for the March 2026 event",
        parent=node,
        critical=False,
    )

    # Broadcaster (critical)
    leaf_broad = evaluator.add_leaf(
        id="Broadcast_Network",
        desc=f"Identify the network or platform broadcasting the event: {data.broadcaster}",
        parent=details,
        critical=True,
    )
    await evaluator.verify(
        claim=f"The event is broadcast on '{data.broadcaster}'.",
        node=leaf_broad,
        sources=urls,
        additional_instruction="Accept branding variants ('ABC', 'ABC Network', 'ABC/Disney').",
    )

    # Exact date (critical)
    leaf_date = evaluator.add_leaf(
        id="Exact_Event_Date",
        desc=f"Provide the exact date of the event in March 2026: {data.date}",
        parent=details,
        critical=True,
    )
    await evaluator.verify(
        claim=f"The event date is '{data.date}', and that date is in March 2026.",
        node=leaf_date,
        sources=urls,
        additional_instruction=date_check_instruction("event"),
    )

    # Event time (critical, with timezone expectation)
    time_has_tz = has_timezone(data.time)
    evaluator.add_custom_node(
        result=time_has_tz,
        id="Event_Time_Timezone_Check",
        desc=f"Broadcast time should include a timezone: {data.time}",
        parent=details,
        critical=True,
    )

    leaf_time = evaluator.add_leaf(
        id="Event_Time",
        desc=f"Provide the broadcast time with time zone: {data.time}",
        parent=details,
        critical=True,
    )
    await evaluator.verify(
        claim=f"The broadcast time is '{data.time}'.",
        node=leaf_time,
        sources=urls,
        additional_instruction="Time zone abbreviations like ET, PT, GMT, UTC, etc., should be present.",
    )

    # Venue/city (optional)
    if data.venue_or_city and data.venue_or_city.strip():
        leaf_venue = evaluator.add_leaf(
            id="Event_Location_Venue",
            desc=f"Provide the venue name and/or city: {data.venue_or_city}",
            parent=details,
            critical=False,
        )
        await evaluator.verify(
            claim=f"The event location/venue is '{data.venue_or_city}'.",
            node=leaf_venue,
            sources=urls,
            additional_instruction="Accept either the venue name, the city, or both as a correct match.",
        )
    else:
        evaluator.add_custom_node(
            result=True,
            id="Event_Location_Venue",
            desc="Venue/city not provided; optional; treated as pass.",
            parent=details,
            critical=False,
        )

    # Reference reliability (critical)
    leaf_ref = evaluator.add_leaf(
        id="Reference_URL_Event",
        desc="Provide a reliable reference URL confirming the event details",
        parent=details,
        critical=True,
    )
    await evaluator.verify(
        claim="At least one of the provided URLs is from a reliable entertainment outlet, sports league/broadcaster, or the official event organizer.",
        node=leaf_ref,
        sources=urls,
        additional_instruction=reliability_additional_instruction(),
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

    # Extract structured info for all categories
    extracted: EntertainmentEventsExtraction = await evaluator.extract(
        prompt=prompt_extract_entertainment_events(),
        template_class=EntertainmentEventsExtraction,
        extraction_name="entertainment_events_march_2026",
    )

    # Record some custom info for diagnostics
    evaluator.add_custom_info(
        info={
            "allowed_streaming_platforms": sorted(list(ALLOWED_STREAMING_PLATFORMS)),
            "reliable_domain_hints_sample": RELIABLE_DOMAIN_HINTS[:10],
        },
        info_type="config",
        info_name="evaluation_config",
    )

    # Build verifications for each category
    await build_verify_tv_series_premiere(evaluator, root, extracted.tv_series)
    await build_verify_streaming_release(evaluator, root, extracted.streaming_release)
    await build_verify_movie_release(evaluator, root, extracted.movie_release)
    await build_verify_live_event(evaluator, root, extracted.live_event)

    return evaluator.get_summary()