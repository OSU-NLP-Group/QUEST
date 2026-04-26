import asyncio
import logging
from typing import List, Optional, Dict, Any, Set

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator, AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.llm_client.base_client import LLMClient


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "dwts_s34_freestyle_music"
TASK_DESCRIPTION = (
    "Identify three different freestyle performances from the Dancing with the Stars Season 34 finale that aired on "
    "November 25, 2025. For each of the three performances, provide the following information: "
    "(1) the celebrity contestant and their professional dancer partner; "
    "(2) complete song title(s) used in the freestyle performance (if multiple songs or a mashup was performed, identify all songs); "
    "(3) the performing artist name(s) for each song; "
    "(4) the official release year of each song; "
    "(5) the birthplace (including city and country or region) of each primary artist; and "
    "(6) the nationality of each primary artist. "
    "The three performances you select must collectively feature artists from at least two different countries to demonstrate "
    "the international diversity of music selections. Provide reference URLs from reliable sources to support all information."
)


# --------------------------------------------------------------------------- #
# Data models                                                                 #
# --------------------------------------------------------------------------- #
class ArtistInfo(BaseModel):
    name: Optional[str] = None
    birthplace: Optional[str] = None  # City + Country/Region if available
    nationality: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class SongInfo(BaseModel):
    title: Optional[str] = None
    artists: List[ArtistInfo] = Field(default_factory=list)  # Primary performing artist(s)
    release_year: Optional[str] = None  # Keep as string to be lenient (e.g., "2010", "2010/2011")
    sources: List[str] = Field(default_factory=list)  # URLs specifically about the song and/or its use in the performance


class PerformanceInfo(BaseModel):
    celebrity: Optional[str] = None
    pro_partner: Optional[str] = None
    performance_type: Optional[str] = None  # Expected "Freestyle"
    songs: List[SongInfo] = Field(default_factory=list)
    sources: List[str] = Field(default_factory=list)  # URLs supporting performance details (finale, song list, couple, etc.)


class ContextInfo(BaseModel):
    show_name: Optional[str] = None  # Expected "Dancing with the Stars"
    season: Optional[str] = None     # Expected "34"
    finale_air_date: Optional[str] = None  # Expected "November 25, 2025" or similar
    context_sources: List[str] = Field(default_factory=list)


class FreestyleExtraction(BaseModel):
    context: Optional[ContextInfo] = None
    performances: List[PerformanceInfo] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_freestyle_set() -> str:
    return """
    Extract exactly three freestyle performances from the answer text related to the Dancing with the Stars (DWTS) Season 34 finale.
    Return a JSON object with the following structure:

    {
      "context": {
        "show_name": string or null,
        "season": string or null,
        "finale_air_date": string or null,
        "context_sources": string[]   // URLs that support context such as season/finale date, official episode info, etc.
      },
      "performances": [
        {
          "celebrity": string or null,
          "pro_partner": string or null,
          "performance_type": string or null,   // e.g., "Freestyle"
          "songs": [
            {
              "title": string or null,
              "artists": [
                {
                  "name": string or null,
                  "birthplace": string or null,     // include city and country/region if available
                  "nationality": string or null,
                  "sources": string[]               // URLs specifically supporting artist bio info (birthplace/nationality)
                }
              ],
              "release_year": string or null,       // official release year; if ambiguous, extract the answer's provided year
              "sources": string[]                   // URLs supporting the song info and/or its use in this performance
            }
          ],
          "sources": string[]                       // URLs supporting couple identification, performance details, finale participation, dance style, and song usage
        },
        ...
      ]
    }

    Rules:
    - Extract up to three performances; if more than three are present, include only the first three.
    - The performances must be described as freestyle routines from the Season 34 finale.
    - Include all songs used in each freestyle, including mashups or multiple tracks, exactly as the answer states.
    - For each song, list the primary performing artist(s). For each primary artist, include birthplace and nationality if provided in the answer.
    - Extract all URLs exactly as they appear in the answer text (supporting context, performance details, songs, and artists). Do not invent URLs.
    - If certain details are missing in the answer, set those fields to null and leave arrays empty.
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _safe_str(s: Optional[str]) -> str:
    return s or ""

def _join_names(names: List[str]) -> str:
    return ", ".join([n for n in names if n])

def _dedupe_keep_order(items: List[str]) -> List[str]:
    seen = set()
    result = []
    for it in items:
        if not it:
            continue
        if it not in seen:
            seen.add(it)
            result.append(it)
    return result

def _filter_valid_urls(urls: List[str]) -> List[str]:
    return [u.strip() for u in urls if isinstance(u, str) and u.strip().startswith(("http://", "https://"))]

def _performance_all_sources(perf: PerformanceInfo) -> List[str]:
    urls: List[str] = []
    urls.extend(perf.sources or [])
    for song in perf.songs or []:
        urls.extend(song.sources or [])
        for artist in song.artists or []:
            urls.extend(artist.sources or [])
    return _filter_valid_urls(_dedupe_keep_order(urls))

def _context_all_sources(extracted: FreestyleExtraction) -> List[str]:
    urls: List[str] = []
    if extracted.context:
        urls.extend(extracted.context.context_sources or [])
    for perf in extracted.performances or []:
        urls.extend(_performance_all_sources(perf))
    return _filter_valid_urls(_dedupe_keep_order(urls))

def _couple_display(perf: PerformanceInfo) -> str:
    celeb = _safe_str(perf.celebrity)
    pro = _safe_str(perf.pro_partner)
    if celeb and pro:
        return f"{celeb} with {pro}"
    return (celeb or "Unknown celebrity") + ((" with " + pro) if pro else "")

def _format_song_list(perf: PerformanceInfo) -> str:
    lines = []
    for idx, song in enumerate(perf.songs or [], start=1):
        title = f"'{_safe_str(song.title)}'"
        artists = _join_names([_safe_str(a.name) for a in song.artists or []])
        if artists:
            lines.append(f"{idx}) {title} by {artists}")
        else:
            lines.append(f"{idx}) {title}")
    return "; ".join(lines) if lines else "None provided"

def _format_song_artists_map(perf: PerformanceInfo) -> str:
    parts = []
    for song in perf.songs or []:
        title = f"'{_safe_str(song.title)}'"
        artists = _join_names([_safe_str(a.name) for a in song.artists or []])
        parts.append(f"{title}: {artists if artists else 'Unknown'}")
    return "; ".join(parts) if parts else "No songs"

def _format_song_years_map(perf: PerformanceInfo) -> str:
    parts = []
    for song in perf.songs or []:
        title = f"'{_safe_str(song.title)}'"
        year = _safe_str(song.release_year)
        parts.append(f"{title}: {year if year else 'Unknown'}")
    return "; ".join(parts) if parts else "No songs"

def _format_artist_birthplaces(perf: PerformanceInfo) -> str:
    parts = []
    for song in perf.songs or []:
        for artist in song.artists or []:
            name = _safe_str(artist.name)
            birthplace = _safe_str(artist.birthplace)
            if name:
                parts.append(f"{name}: {birthplace if birthplace else 'Unknown'}")
    return "; ".join(parts) if parts else "No artists"

def _format_artist_nationalities(perf: PerformanceInfo) -> str:
    parts = []
    for song in perf.songs or []:
        for artist in song.artists or []:
            name = _safe_str(artist.name)
            nationality = _safe_str(artist.nationality)
            if name:
                parts.append(f"{name}: {nationality if nationality else 'Unknown'}")
    return "; ".join(parts) if parts else "No artists"

def _unique_couples(performances: List[PerformanceInfo]) -> Set[str]:
    uniq: Set[str] = set()
    for perf in performances:
        key = f"{_safe_str(perf.celebrity).strip().lower()}|{_safe_str(perf.pro_partner).strip().lower()}"
        uniq.add(key)
    return {k for k in uniq if k != "|"}

def _collect_nationalities(performances: List[PerformanceInfo]) -> Set[str]:
    result: Set[str] = set()
    for perf in performances:
        for song in perf.songs or []:
            for artist in song.artists or []:
                nat = _safe_str(artist.nationality).strip().lower()
                if nat:
                    result.add(nat)
    return result


# --------------------------------------------------------------------------- #
# Verification for one performance                                            #
# --------------------------------------------------------------------------- #
async def verify_one_performance(
    evaluator: Evaluator,
    parent_node,
    perf: PerformanceInfo,
    ordinal_label: str,  # "First", "Second", "Third"
    idx: int
) -> None:
    # Container for this performance (parallel, non-critical)
    perf_node = evaluator.add_parallel(
        id=f"{ordinal_label}_Performance",
        desc=f"Analysis of the {ordinal_label.lower()} identified performance",
        parent=parent_node,
        critical=False
    )

    # 1) Couple identification (Critical leaf)
    couple_leaf = evaluator.add_leaf(
        id=f"{ordinal_label}_Perf_Couple_Identification",
        desc="Celebrity contestant and professional dancer partner are identified",
        parent=perf_node,
        critical=True
    )
    couple_claim = (
        f"The couple in the selected performance is {_couple_display(perf)}, and this performance is part of the "
        f"Dancing with the Stars Season 34 finale on November 25, 2025."
    )
    await evaluator.verify(
        claim=couple_claim,
        node=couple_leaf,
        sources=_performance_all_sources(perf),
        additional_instruction="Verify that the named celebrity and pro partner appear together in a freestyle at the Season 34 finale."
    )

    # 2) Song analysis group (parallel, critical)
    song_group = evaluator.add_parallel(
        id=f"{ordinal_label}_Perf_Song_Analysis",
        desc="All songs used in the freestyle are identified with required song metadata",
        parent=perf_node,
        critical=True
    )

    # 2.1 All songs listed (Critical leaf)
    songs_listed_leaf = evaluator.add_leaf(
        id=f"{ordinal_label}_Perf_All_Songs_Listed",
        desc="All songs used (including mashups/multiple songs) are included",
        parent=song_group,
        critical=True
    )
    songs_list = _format_song_list(perf)
    songs_list_claim = (
        f"For the freestyle by {_couple_display(perf)} at the Season 34 finale (Nov 25, 2025), "
        f"the song list includes exactly: {songs_list}."
    )
    await evaluator.verify(
        claim=songs_list_claim,
        node=songs_listed_leaf,
        sources=_performance_all_sources(perf),
        additional_instruction="Check the sources for the listed freestyle music tracks for this specific couple's freestyle on the finale night."
    )

    # 2.2 Artist names for each song (Critical leaf)
    artist_names_leaf = evaluator.add_leaf(
        id=f"{ordinal_label}_Perf_Artist_Names",
        desc="Performing artist name(s) are provided for each song",
        parent=song_group,
        critical=True
    )
    song_artists_map = _format_song_artists_map(perf)
    artist_names_claim = (
        f"The performing artist(s) for each song used by {_couple_display(perf)} are as follows: {song_artists_map}."
    )
    await evaluator.verify(
        claim=artist_names_claim,
        node=artist_names_leaf,
        sources=_performance_all_sources(perf),
        additional_instruction="Verify the canonical performing artist(s) for each named song. Minor name variants are acceptable."
    )

    # 2.3 Release years for each song (Critical leaf)
    release_years_leaf = evaluator.add_leaf(
        id=f"{ordinal_label}_Perf_Release_Years",
        desc="Official release year is provided for each song",
        parent=song_group,
        critical=True
    )
    song_years_map = _format_song_years_map(perf)
    release_years_claim = (
        f"The official release year for each song used by {_couple_display(perf)} is: {song_years_map}."
    )
    await evaluator.verify(
        claim=release_years_claim,
        node=release_years_leaf,
        sources=_performance_all_sources(perf),
        additional_instruction="Check authoritative sources for the original release year (single or album). Minor discrepancies are acceptable if well-supported."
    )

    # 3) Artist bio group (parallel, critical)
    artist_bio_group = evaluator.add_parallel(
        id=f"{ordinal_label}_Perf_Artist_Bio",
        desc="Primary-artist birthplace and nationality provided for each song's primary artist",
        parent=perf_node,
        critical=True
    )

    # 3.1 Birthplaces (Critical leaf)
    birthplaces_leaf = evaluator.add_leaf(
        id=f"{ordinal_label}_Perf_Artist_Birthplaces",
        desc="Birthplace (city and country/region) provided for each primary artist",
        parent=artist_bio_group,
        critical=True
    )
    birthplace_map = _format_artist_birthplaces(perf)
    birthplace_claim = (
        f"The birthplaces for the primary artists featured in {_couple_display(perf)}'s freestyle are: {birthplace_map}."
    )
    await evaluator.verify(
        claim=birthplace_claim,
        node=birthplaces_leaf,
        sources=_performance_all_sources(perf),
        additional_instruction="Verify each artist's birthplace (city and country/region). If an artist has multiple associated locations, prefer birthplace."
    )

    # 3.2 Nationalities (Critical leaf)
    nationalities_leaf = evaluator.add_leaf(
        id=f"{ordinal_label}_Perf_Artist_Nationalities",
        desc="Nationality provided for each primary artist",
        parent=artist_bio_group,
        critical=True
    )
    nationality_map = _format_artist_nationalities(perf)
    nationality_claim = (
        f"The nationalities for the primary artists featured in {_couple_display(perf)}'s freestyle are: {nationality_map}."
    )
    await evaluator.verify(
        claim=nationality_claim,
        node=nationalities_leaf,
        sources=_performance_all_sources(perf),
        additional_instruction="Verify each artist's nationality. If dual nationality is reported, the stated one in the answer is acceptable."
    )

    # 4) References support (Critical leaf) — structural check for presence of URLs
    refs_leaf = evaluator.add_custom_node(
        result=(len(_performance_all_sources(perf)) >= 1),
        id=f"{ordinal_label}_Perf_References",
        desc="Reference URLs from reliable sources support the performance, song, and artist information provided",
        parent=perf_node,
        critical=True
    )


# --------------------------------------------------------------------------- #
# Main evaluation entry point                                                 #
# --------------------------------------------------------------------------- #
async def evaluate_answer(
    client: LLMClient,
    answer: str,
    agent_name: str,
    answer_name: str,
    cache: CacheFileSys,
    semaphore: asyncio.Semaphore,
    logger: logging.Logger,
    model: str = "o4-mini"
) -> Dict[str, Any]:
    # Initialize evaluator with sequential root (root is non-critical by framework design)
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.SEQUENTIAL,
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
    extracted: FreestyleExtraction = await evaluator.extract(
        prompt=prompt_extract_freestyle_set(),
        template_class=FreestyleExtraction,
        extraction_name="freestyle_extraction"
    )

    # Ensure exactly three performances (pad with empty if fewer; trim if more)
    performances: List[PerformanceInfo] = list((extracted.performances or [])[:3])
    while len(performances) < 3:
        performances.append(PerformanceInfo())

    # ---------------- Context Verification (critical, parallel) ----------------
    context_node = evaluator.add_parallel(
        id="Context_Verification",
        desc="Verification of event context and coverage requirements",
        parent=root,
        critical=True
    )

    # 1) Event specification (Critical leaf) — finale and date
    event_leaf = evaluator.add_leaf(
        id="Event_Specification",
        desc="Performances are from Dancing with the Stars Season 34 finale that aired on November 25, 2025",
        parent=context_node,
        critical=True
    )
    context_sources = _context_all_sources(extracted)
    event_claim = "The Dancing with the Stars Season 34 finale aired on November 25, 2025."
    await evaluator.verify(
        claim=event_claim,
        node=event_leaf,
        sources=context_sources,
        additional_instruction="Verify the show, season number (34), and the finale air date (Nov 25, 2025)."
    )

    # 2) Performance category (Critical leaf) — freestyle
    category_leaf = evaluator.add_leaf(
        id="Performance_Category",
        desc="Performances selected are freestyle performances from the finale",
        parent=context_node,
        critical=True
    )
    couples_list = "; ".join([_couple_display(p) for p in performances])
    category_claim = (
        f"The selected performances ({couples_list}) are freestyle routines from the Season 34 finale."
    )
    await evaluator.verify(
        claim=category_claim,
        node=category_leaf,
        sources=context_sources,
        additional_instruction="Check that each selected couple performed a Freestyle at the finale, using provided sources."
    )

    # 3) Exactly three different finalist couples (Critical) — structural existence/uniqueness
    uniq = _unique_couples(performances)
    exactly_three_leaf = evaluator.add_custom_node(
        result=(len(uniq) == 3),
        id="Coverage_Exactly_Three_Finalist_Couples",
        desc="Exactly three different finalist couples' freestyle performances are analyzed",
        parent=context_node,
        critical=True
    )

    # ---------------- Performance Set (non-critical, parallel) ----------------
    perf_set_node = evaluator.add_parallel(
        id="Performance_Set",
        desc="Collection of individual performance analyses",
        parent=root,
        critical=False
    )

    # First performance
    await verify_one_performance(
        evaluator=evaluator,
        parent_node=perf_set_node,
        perf=performances[0],
        ordinal_label="First",
        idx=0
    )

    # Second performance
    await verify_one_performance(
        evaluator=evaluator,
        parent_node=perf_set_node,
        perf=performances[1],
        ordinal_label="Second",
        idx=1
    )

    # Third performance
    await verify_one_performance(
        evaluator=evaluator,
        parent_node=perf_set_node,
        perf=performances[2],
        ordinal_label="Third",
        idx=2
    )

    # ---------------- Aggregate Requirements (critical, parallel) -------------
    aggregate_node = evaluator.add_parallel(
        id="Aggregate_Requirements",
        desc="Requirements that apply across all selected performances collectively",
        parent=root,
        critical=True
    )

    # Geographic diversity: at least two different countries (by nationalities)
    nationalities = _collect_nationalities(performances)
    geo_diverse_leaf = evaluator.add_custom_node(
        result=(len(nationalities) >= 2),
        id="Geographic_Diversity",
        desc="Across the three selected performances, the primary artists collectively represent at least two different countries",
        parent=aggregate_node,
        critical=True
    )

    # Add some custom info for transparency
    evaluator.add_custom_info(
        info={
            "unique_couples_count": len(uniq),
            "unique_couples_keys": list(uniq),
            "unique_artist_nationalities_count": len(nationalities),
            "unique_artist_nationalities": sorted(list(nationalities))
        },
        info_type="diagnostics",
        info_name="aggregate_diagnostics"
    )

    return evaluator.get_summary()