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
TASK_ID = "music_events_mar_2026"
TASK_DESCRIPTION = """
As of March 20, 2026, provide comprehensive information about the following five music-related events and performances:

1. BTS Album Release: What is the album name, release date, total number of tracks, and title track for BTS's new album that was released on March 20, 2026?

2. BTS Streaming Event: What is the official event name, exact date, and streaming time (in either KST or PT timezone) for BTS's live comeback concert that is streaming exclusively on Netflix on March 21, 2026?

3. Grammy Winner on Billboard: Olivia Dean won Best New Artist at the 2026 Grammy Awards. What is the title of her song and what position does it hold on the Billboard Hot 100 chart dated March 21, 2026?

4. Billboard Debut: What is the song name and chart position for Harry Styles' new song that debuted on the Billboard Hot 100 chart dated March 21, 2026, and was it the highest-ranking debut that week?

5. Broadway Performance: What is the name of the musical, the character name, the theatre name, and the departure date for Lea Michele's current Broadway performance that she will be leaving in June 2026?
"""


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class BTSAlbumInfo(BaseModel):
    album_name: Optional[str] = None
    release_date: Optional[str] = None
    total_tracks: Optional[str] = None
    title_track: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class BTSEventInfo(BaseModel):
    event_name: Optional[str] = None
    date: Optional[str] = None
    time: Optional[str] = None
    timezone: Optional[str] = None  # Expect e.g., "KST" or "PT"
    sources: List[str] = Field(default_factory=list)


class OliviaDeanInfo(BaseModel):
    grammy_award: Optional[str] = None  # Expect "Best New Artist" if present
    grammy_sources: List[str] = Field(default_factory=list)
    song_title: Optional[str] = None
    chart_position: Optional[str] = None  # Keep as string to allow "No. 12" etc.
    chart_date: Optional[str] = None  # Expect "March 21, 2026" if mentioned
    billboard_sources: List[str] = Field(default_factory=list)


class HarryStylesInfo(BaseModel):
    song_title: Optional[str] = None
    chart_position: Optional[str] = None
    chart_date: Optional[str] = None  # Expect "March 21, 2026"
    highest_ranking_debut: Optional[str] = None  # "yes"/"no" or textual claim
    billboard_sources: List[str] = Field(default_factory=list)


class LeaMicheleInfo(BaseModel):
    musical_name: Optional[str] = None
    character_name: Optional[str] = None
    theatre_name: Optional[str] = None
    departure_date: Optional[str] = None  # Should indicate June 2026 if present
    sources: List[str] = Field(default_factory=list)


class MusicEventsExtraction(BaseModel):
    bts_album: Optional[BTSAlbumInfo] = None
    bts_event: Optional[BTSEventInfo] = None
    olivia_dean: Optional[OliviaDeanInfo] = None
    harry_styles: Optional[HarryStylesInfo] = None
    lea_michele: Optional[LeaMicheleInfo] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_music_events() -> str:
    return """
    Extract structured information from the answer for five topics. Return exactly and only what the answer explicitly states.

    1) BTS album (released March 20, 2026):
       - album_name: string
       - release_date: string (as written in the answer)
       - total_tracks: string (e.g., "10", "10 tracks", "ten")
       - title_track: string
       - sources: array of URLs specifically cited for the album info

    2) BTS Netflix live streaming event (March 21, 2026):
       - event_name: string (official event name)
       - date: string (as written in the answer; should refer to March 21, 2026 in KST or PT)
       - time: string (e.g., "8 PM", "12:00 KST")
       - timezone: string (e.g., "KST" or "PT") if given; otherwise null
       - sources: array of URLs cited for the event

    3) Olivia Dean — Grammy and Billboard Hot 100 (chart dated March 21, 2026):
       - grammy_award: string (e.g., "Best New Artist") if mentioned; else null
       - grammy_sources: array of URLs cited for Grammy claim
       - song_title: string (the song of Olivia Dean referenced with Billboard)
       - chart_position: string (e.g., "No. 14", "14")
       - chart_date: string (e.g., "March 21, 2026") if present; else null
       - billboard_sources: array of URLs cited for chart claim

    4) Harry Styles — Billboard Hot 100 debut (chart dated March 21, 2026):
       - song_title: string
       - chart_position: string (e.g., "No. 8", "8")
       - chart_date: string (e.g., "March 21, 2026") if present; else null
       - highest_ranking_debut: string (e.g., "yes", "no", or phrased text); if missing, null
       - billboard_sources: array of URLs cited for this chart claim

    5) Lea Michele — Broadway performance (leaving in June 2026):
       - musical_name: string
       - character_name: string
       - theatre_name: string
       - departure_date: string (as written in the answer; should indicate June 2026)
       - sources: array of URLs cited for Broadway info

    Extraction rules:
    - Do not invent any information; if a field is missing, set it to null (or empty list for URLs).
    - For URL fields, return only valid URLs explicitly present in the answer (plain or in markdown).
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def sanitize_urls(urls: Optional[List[str]]) -> List[str]:
    if not urls:
        return []
    out = []
    seen = set()
    for u in urls:
        if not u:
            continue
        s = u.strip()
        if not s:
            continue
        if s not in seen:
            seen.add(s)
            out.append(s)
    return out


# --------------------------------------------------------------------------- #
# Verification functions                                                      #
# --------------------------------------------------------------------------- #
async def verify_bts_album(evaluator: Evaluator, parent) -> None:
    node = evaluator.add_parallel(
        id="BTS_ARIRANG_Album",
        desc="Provide the correct album name, release date (March 20, 2026), total track count, and title track name for BTS's new album",
        parent=parent,
        critical=False
    )

    info: Optional[BTSAlbumInfo] = getattr(evaluator, "_music_extract", MusicEventsExtraction()).bts_album  # type: ignore
    has_core = bool(info and info.album_name and info.release_date)
    sources = sanitize_urls(info.sources if info else [])
    existence = evaluator.add_custom_node(
        result=has_core and len(sources) > 0,
        id="bts_album_exists",
        desc="BTS album details are provided with at least one source URL",
        parent=node,
        critical=True
    )

    # Album name verification
    album_name_leaf = evaluator.add_leaf(
        id="bts_album_name_supported",
        desc="Album name is correctly stated and supported by the cited sources",
        parent=node,
        critical=True
    )
    album_name = info.album_name if info and info.album_name else ""
    await evaluator.verify(
        claim=f"BTS's new album released on March 20, 2026 is titled '{album_name}'.",
        node=album_name_leaf,
        sources=sources,
        additional_instruction="Allow minor variants and punctuation. Confirm this is BTS's album associated with the March 20, 2026 release."
    )

    # Release date verification
    release_date_leaf = evaluator.add_leaf(
        id="bts_album_release_date_supported",
        desc="Release date is March 20, 2026 and supported by sources",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The BTS album '{album_name}' was released on March 20, 2026.",
        node=release_date_leaf,
        sources=sources,
        additional_instruction="Accept if the source clearly indicates March 20, 2026 (especially in KST). Minor timezone mention is fine as long as March 20, 2026 is the official release date."
    )

    # Total tracks verification
    tracks_leaf = evaluator.add_leaf(
        id="bts_album_total_tracks_supported",
        desc="Total number of tracks is correctly stated and supported",
        parent=node,
        critical=True
    )
    total_tracks = info.total_tracks if info and info.total_tracks else ""
    await evaluator.verify(
        claim=f"The BTS album '{album_name}' contains {total_tracks} tracks.",
        node=tracks_leaf,
        sources=sources,
        additional_instruction="Accept synonyms like 'songs' or counts like '10 tracks'. If a deluxe or versioned count is specified in the answer, verify that exact count."
    )

    # Title track verification
    title_track_leaf = evaluator.add_leaf(
        id="bts_album_title_track_supported",
        desc="Title track is correctly stated and supported",
        parent=node,
        critical=True
    )
    title_track = info.title_track if info and info.title_track else ""
    await evaluator.verify(
        claim=f"The album '{album_name}' has the title track (lead single) '{title_track}'.",
        node=title_track_leaf,
        sources=sources,
        additional_instruction="Treat 'title track' and 'lead single' as equivalent if the source uses either phrasing."
    )


async def verify_bts_event(evaluator: Evaluator, parent) -> None:
    node = evaluator.add_parallel(
        id="BTS_Netflix_Event",
        desc="Provide the correct event name, exact date (March 21, 2026), and streaming time (in KST or PT) for BTS's live comeback concert streaming on Netflix",
        parent=parent,
        critical=False
    )

    info: Optional[BTSEventInfo] = getattr(evaluator, "_music_extract", MusicEventsExtraction()).bts_event  # type: ignore
    has_core = bool(info and info.event_name and info.date and info.time)
    sources = sanitize_urls(info.sources if info else [])
    evaluator.add_custom_node(
        result=has_core and len(sources) > 0,
        id="bts_event_exists",
        desc="BTS streaming event info (name/date/time) is provided with sources",
        parent=node,
        critical=True
    )

    # Official event name
    event_name_leaf = evaluator.add_leaf(
        id="bts_event_name_supported",
        desc="Official event name is correctly stated and supported",
        parent=node,
        critical=True
    )
    event_name = info.event_name if info and info.event_name else ""
    await evaluator.verify(
        claim=f"The official event name for the BTS live comeback concert streaming on Netflix is '{event_name}'.",
        node=event_name_leaf,
        sources=sources,
        additional_instruction="Allow minor punctuation/casing differences. Confirm this is the official name used by Netflix or BTS."
    )

    # Date verification (March 21, 2026)
    event_date_leaf = evaluator.add_leaf(
        id="bts_event_date_supported",
        desc="Streaming date is March 21, 2026 and supported by sources",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="The BTS live comeback concert streams on Netflix on March 21, 2026.",
        node=event_date_leaf,
        sources=sources,
        additional_instruction="Accept if the source specifies March 21, 2026 in KST or PT. Timezone labeling differences are acceptable as long as March 21, 2026 is stated."
    )

    # Time verification (KST or PT)
    time_leaf = evaluator.add_leaf(
        id="bts_event_time_supported",
        desc="Streaming time (KST or PT) is correctly stated and supported",
        parent=node,
        critical=True
    )
    time_str = info.time if info and info.time else ""
    tz = info.timezone if info and info.timezone else ""
    await evaluator.verify(
        claim=f"The streaming time for the Netflix event is {time_str} {tz} (or the equivalent local time if the source provides the other timezone).",
        node=time_leaf,
        sources=sources,
        additional_instruction="Accept if time is provided in either PT or KST and matches the stated time after reasonable timezone conversion. Minor formatting differences (e.g., 8PM vs 8:00 PM) are fine."
    )


async def verify_olivia_dean(evaluator: Evaluator, parent) -> None:
    node = evaluator.add_parallel(
        id="Olivia_Dean_Performance",
        desc="Confirm Olivia Dean's 2026 Grammy Best New Artist win and provide her Billboard Hot 100 song and position for the chart dated March 21, 2026",
        parent=parent,
        critical=False
    )

    info: Optional[OliviaDeanInfo] = getattr(evaluator, "_music_extract", MusicEventsExtraction()).olivia_dean  # type: ignore

    # Subnode: Grammy win
    grammy_node = evaluator.add_parallel(
        id="olivia_grammy_block",
        desc="Olivia Dean Grammy win verification",
        parent=node,
        critical=False
    )
    grammy_sources = sanitize_urls(info.grammy_sources if info else [])
    evaluator.add_custom_node(
        result=(bool(info and info.grammy_award) and len(grammy_sources) > 0),
        id="olivia_grammy_exists",
        desc="Olivia Dean Grammy claim present with sources",
        parent=grammy_node,
        critical=True
    )
    grammy_leaf = evaluator.add_leaf(
        id="olivia_grammy_bna_supported",
        desc="Olivia Dean won Best New Artist at the 2026 Grammy Awards",
        parent=grammy_node,
        critical=True
    )
    await evaluator.verify(
        claim="Olivia Dean won Best New Artist at the 2026 Grammy Awards.",
        node=grammy_leaf,
        sources=grammy_sources,
        additional_instruction="Confirm the exact category 'Best New Artist' and the year 2026."
    )

    # Subnode: Billboard chart
    bb_node = evaluator.add_parallel(
        id="olivia_billboard_block",
        desc="Olivia Dean Billboard Hot 100 position verification (chart dated March 21, 2026)",
        parent=node,
        critical=False
    )
    bb_sources = sanitize_urls(info.billboard_sources if info else [])
    evaluator.add_custom_node(
        result=(bool(info and info.song_title and info.chart_position) and len(bb_sources) > 0),
        id="olivia_billboard_exists",
        desc="Olivia Dean Billboard song/position claim present with sources",
        parent=bb_node,
        critical=True
    )
    olivia_song = info.song_title if info and info.song_title else ""
    olivia_pos = info.chart_position if info and info.chart_position else ""
    bb_leaf = evaluator.add_leaf(
        id="olivia_billboard_position_supported",
        desc="Olivia Dean song title and Hot 100 position supported on chart dated March 21, 2026",
        parent=bb_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"On the Billboard Hot 100 chart dated March 21, 2026, Olivia Dean's song '{olivia_song}' is at position {olivia_pos}.",
        node=bb_leaf,
        sources=bb_sources,
        additional_instruction="Allow formats like 'No. X' or just the number. Confirm the chart date is March 21, 2026."
    )


async def verify_harry_styles(evaluator: Evaluator, parent) -> None:
    node = evaluator.add_parallel(
        id="Harry_Styles_Billboard_Debut",
        desc="Provide the correct song name and chart position for Harry Styles' new song debut on the Billboard Hot 100 dated March 21, 2026, and whether it was the highest-ranking debut that week",
        parent=parent,
        critical=False
    )

    info: Optional[HarryStylesInfo] = getattr(evaluator, "_music_extract", MusicEventsExtraction()).harry_styles  # type: ignore
    bb_sources = sanitize_urls(info.billboard_sources if info else [])
    evaluator.add_custom_node(
        result=(bool(info and info.song_title and info.chart_position) and len(bb_sources) > 0),
        id="harry_billboard_exists",
        desc="Harry Styles Billboard debut claim present with sources",
        parent=node,
        critical=True
    )

    song = info.song_title if info and info.song_title else ""
    pos = info.chart_position if info and info.chart_position else ""

    # Position verification
    pos_leaf = evaluator.add_leaf(
        id="harry_billboard_position_supported",
        desc="Harry Styles song title and Hot 100 position supported on chart dated March 21, 2026",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim=f"On the Billboard Hot 100 chart dated March 21, 2026, Harry Styles' song '{song}' is at position {pos}.",
        node=pos_leaf,
        sources=bb_sources,
        additional_instruction="Allow 'No. X' or numeric formats. Confirm the date is March 21, 2026."
    )

    # Highest-ranking debut verification
    highest_debut_leaf = evaluator.add_leaf(
        id="harry_highest_debut_supported",
        desc="Harry Styles' song was the highest-ranking debut on that chart week",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="On the Billboard Hot 100 chart dated March 21, 2026, this Harry Styles song was the highest-ranking debut that week.",
        node=highest_debut_leaf,
        sources=bb_sources,
        additional_instruction="Accept synonyms like 'top debut' or 'highest new entry'. Verify relative to other debuts on the same chart."
    )


async def verify_lea_michele(evaluator: Evaluator, parent) -> None:
    node = evaluator.add_parallel(
        id="Lea_Michele_Broadway",
        desc="Provide the correct musical name, character name, theatre name, and departure date (June 2026) for Lea Michele's current Broadway performance",
        parent=parent,
        critical=False
    )

    info: Optional[LeaMicheleInfo] = getattr(evaluator, "_music_extract", MusicEventsExtraction()).lea_michele  # type: ignore
    has_core = bool(info and info.musical_name and info.character_name and info.theatre_name and info.departure_date)
    sources = sanitize_urls(info.sources if info else [])
    evaluator.add_custom_node(
        result=has_core and len(sources) > 0,
        id="lea_broadway_exists",
        desc="Lea Michele Broadway info present with sources",
        parent=node,
        critical=True
    )

    musical = info.musical_name if info and info.musical_name else ""
    character = info.character_name if info and info.character_name else ""
    theatre = info.theatre_name if info and info.theatre_name else ""
    dep_date = info.departure_date if info and info.departure_date else ""

    # Musical name
    musical_leaf = evaluator.add_leaf(
        id="lea_musical_supported",
        desc="Musical name is correctly stated and supported",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim=f"Lea Michele is currently performing in the Broadway musical '{musical}'.",
        node=musical_leaf,
        sources=sources,
        additional_instruction="Allow minor punctuation/casing differences. Confirm that this is her current Broadway show as stated in the answer."
    )

    # Character name
    character_leaf = evaluator.add_leaf(
        id="lea_character_supported",
        desc="Character name is correctly stated and supported",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim=f"In this production of '{musical}', Lea Michele plays the character '{character}'.",
        node=character_leaf,
        sources=sources,
        additional_instruction="Allow minor variations (e.g., nicknames) if unambiguous."
    )

    # Theatre name
    theatre_leaf = evaluator.add_leaf(
        id="lea_theatre_supported",
        desc="Theatre name is correctly stated and supported",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The production is staged at the '{theatre}' theatre on Broadway.",
        node=theatre_leaf,
        sources=sources,
        additional_instruction="Allow 'Theatre' vs 'Theater' spelling variants and inclusion/exclusion of 'The' if obviously the same venue."
    )

    # Departure date (June 2026)
    departure_leaf = evaluator.add_leaf(
        id="lea_departure_date_supported",
        desc="Departure date (in June 2026) is correctly stated and supported",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim=f"Lea Michele will depart the show in June 2026, specifically on {dep_date}.",
        node=departure_leaf,
        sources=sources,
        additional_instruction="Accept if the source clearly indicates June 2026; if a specific date is given, it should fall within June 2026."
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
    Evaluation entry point for the March 2026 music events task.
    """
    # Initialize evaluator and root (parallel aggregation across the 5 subtasks)
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

    # Extract structured info (single pass)
    extract_result: MusicEventsExtraction = await evaluator.extract(
        prompt=prompt_extract_music_events(),
        template_class=MusicEventsExtraction,
        extraction_name="music_events_extraction"
    )
    # Stash on evaluator for easy access in helpers
    setattr(evaluator, "_music_extract", extract_result)

    # Build the top-level node matching the rubric root
    music_root = evaluator.add_parallel(
        id="Music_Events_March_2026",
        desc="Provide accurate information about music events, releases, and performances in March 2026, including K-pop comebacks, Grammy winners' chart performance, Billboard debuts, and Broadway shows",
        parent=root,
        critical=False
    )

    # Run verifications for each subtask
    await verify_bts_album(evaluator, music_root)
    await verify_bts_event(evaluator, music_root)
    await verify_olivia_dean(evaluator, music_root)
    await verify_harry_styles(evaluator, music_root)
    await verify_lea_michele(evaluator, music_root)

    # Return structured summary
    return evaluator.get_summary()