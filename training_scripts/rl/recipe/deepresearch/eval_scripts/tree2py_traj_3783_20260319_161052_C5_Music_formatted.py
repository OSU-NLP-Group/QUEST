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
TASK_ID = "album_2024_grammy_fest_producer"
TASK_DESCRIPTION = """Identify an album that satisfies ALL of the following criteria:

1. The album must have won at least one Grammy Award at the 2024 Grammy Awards ceremony (the 66th Annual Grammy Awards held on February 4, 2024).

2. The album must have been released between January 1, 2022 and December 31, 2023 (inclusive).

3. The artist who created this album must have been listed as a headliner at one of the following major music festivals during 2024: Coachella Valley Music and Arts Festival, Glastonbury Festival, or Lollapalooza.

4. The artist's festival headline performance must have taken place between April 1, 2024 and August 31, 2024 (inclusive).

5. The festival venue where the artist performed as a headliner must have a seating/attendance capacity between 15,000 and 100,000 people.

6. At least one producer who is credited on the album must have won the Grammy Award for Producer of the Year, Non-Classical, at least once during the 2020s (between 2020 and 2029, inclusive).

Provide the album title and artist name that meets all these requirements.
"""

ALLOWED_FESTIVALS = {
    "Coachella Valley Music and Arts Festival",
    "Glastonbury Festival",
    "Lollapalooza",
}

RELEASE_START = "2022-01-01"
RELEASE_END = "2023-12-31"
HEADLINE_START = "2024-04-01"
HEADLINE_END = "2024-08-31"


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class ProducerEntry(BaseModel):
    name: Optional[str] = None
    # Years (as strings) in which this producer won "Producer of the Year, Non-Classical" during the 2020s.
    award_years_2020s: List[str] = Field(default_factory=list)
    # URLs that support the producer's Producer-of-the-Year win(s).
    award_sources: List[str] = Field(default_factory=list)
    # URLs that support that this person is a credited producer on the album.
    album_credit_sources: List[str] = Field(default_factory=list)


class AlbumExtraction(BaseModel):
    album_title: Optional[str] = None
    artist_name: Optional[str] = None

    # Grammy win (for the album) at 2024 Grammys
    grammy_sources: List[str] = Field(default_factory=list)

    # Release date and sources
    release_date: Optional[str] = None
    release_date_sources: List[str] = Field(default_factory=list)

    # Festival/headliner info
    festival_name: Optional[str] = None  # Expect one of ALLOWED_FESTIVALS
    festival_headliner_status: Optional[bool] = None  # If stated explicitly in the answer
    performance_date: Optional[str] = None  # Prefer ISO string if present; else free-form
    festival_sources: List[str] = Field(default_factory=list)

    # Venue capacity info
    venue_name: Optional[str] = None
    venue_capacity: Optional[str] = None  # Keep as string to allow ranges/approximate in answers
    venue_capacity_sources: List[str] = Field(default_factory=list)

    # Producers
    producers: List[ProducerEntry] = Field(default_factory=list)

    # General album sources (e.g., album Wikipedia, label page) that might help verify credits/release
    album_sources: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_album() -> str:
    return """
    Extract a single candidate album (the one the answer claims satisfies all constraints). Return the following fields:

    1) album_title: string (exact as in the answer)
    2) artist_name: string (exact as in the answer)

    3) grammy_sources: array of URLs that the answer cites to support that THIS ALBUM won at least one Grammy Award at the 66th Annual Grammy Awards (held Feb 4, 2024). Only include URLs explicitly present in the answer.

    4) release_date: string for the album's release date (as stated in the answer). Prefer an ISO-like date (YYYY-MM-DD) if available; otherwise keep the original text.
    5) release_date_sources: array of URLs the answer cites for the album release date.

    6) festival_name: the exact name of the 2024 festival where the artist is stated as a headliner in the answer. Restrict to one of:
       - "Coachella Valley Music and Arts Festival"
       - "Glastonbury Festival"
       - "Lollapalooza"
       If multiple are mentioned, pick the one the answer associates with headlining in 2024.
    7) festival_headliner_status: boolean if the answer explicitly states they were a headliner.
    8) performance_date: string of the headliner performance date at that 2024 festival (prefer an ISO-like date if provided; otherwise keep the original text). If multiple dates, put the most relevant one or a concise range string.
    9) festival_sources: array of URLs the answer cites for headline status and performance timing (lineup page, schedule, news article, etc.).

    10) venue_name: string name of the venue where the headliner performance occurred (e.g., Grant Park for Lollapalooza, Empire Polo Club for Coachella, Worthy Farm for Glastonbury), if provided in the answer.
    11) venue_capacity: string capturing the capacity/attendance number (or range) mentioned in the answer for that venue (daily permitted). Keep as text as stated in the answer.
    12) venue_capacity_sources: array of URLs the answer cites to support the capacity figure/range for the relevant festival venue.

    13) producers: an array of producer entries for at least the producers mentioned in the answer. For each producer:
        - name: string (producer's name as in the answer)
        - award_years_2020s: array of strings for any years in the 2020s (2020–2029) that the producer won "Grammy Award for Producer of the Year, Non-Classical", as claimed in the answer (if any)
        - award_sources: array of URLs from the answer that support the producer’s Producer-of-the-Year win(s)
        - album_credit_sources: array of URLs from the answer that support that this person is a credited producer on the album

    14) album_sources: array of any general album URLs (official site, label page, Wikipedia, Apple/Spotify pages, etc.) explicitly included in the answer.

    IMPORTANT URL RULES:
    - Only include actual URLs explicitly present in the answer text (including markdown links). Do not invent or infer URLs.
    - Exclude obviously invalid/malformed URLs.

    If any field is not present in the answer, return null for that field or an empty list for URL arrays as appropriate.
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _merge_urls(*lists: List[str]) -> List[str]:
    """Merge multiple URL lists, preserve order, and remove duplicates."""
    seen = set()
    merged: List[str] = []
    for lst in lists:
        for url in lst or []:
            if url and url not in seen:
                merged.append(url)
                seen.add(url)
    return merged


def _find_producer_with_2020s_award(producers: List[ProducerEntry]) -> Optional[Dict[str, Any]]:
    """Return a dict with 'name', 'year', 'sources' for the first qualifying producer found."""
    for p in producers or []:
        if not p.name:
            continue
        for y in p.award_years_2020s or []:
            try:
                yi = int(y.strip())
            except Exception:
                continue
            if 2020 <= yi <= 2029:
                sources = _merge_urls(p.award_sources, p.album_credit_sources)
                if sources:
                    return {"name": p.name, "year": yi, "sources": sources}
                else:
                    # Even if we find a year, keep going to find one that has sources
                    continue
    return None


# --------------------------------------------------------------------------- #
# Verification subtrees                                                       #
# --------------------------------------------------------------------------- #
async def add_grammy_compliance(evaluator: Evaluator, parent, ext: AlbumExtraction) -> None:
    node = evaluator.add_sequential(
        id="Grammy_Award_Compliance",
        desc="The album won at least one Grammy Award at the 2024 Grammy Awards ceremony",
        parent=parent,
        critical=True,
    )

    # Presence of reference URL(s)
    ref_present = evaluator.add_custom_node(
        result=bool(ext.grammy_sources),
        id="Grammy_Reference_URL",
        desc="Reference URL provided documenting the Grammy win",
        parent=node,
        critical=True,
    )

    # Verify the album actually won a 2024 Grammy
    leaf = evaluator.add_leaf(
        id="Grammy_Win_Verification",
        desc="The album won at least one Grammy Award at the 66th Annual Grammy Awards held on February 4, 2024",
        parent=node,
        critical=True,
    )
    album_title = ext.album_title or "the album"
    claim = f"The album '{album_title}' won at least one Grammy Award at the 66th Annual Grammy Awards held on February 4, 2024 (the 2024 Grammys)."
    await evaluator.verify(
        claim=claim,
        node=leaf,
        sources=ext.grammy_sources,
        additional_instruction="Confirm the specific album (not just the artist or a single track) is listed as a winner at the 2024 Grammys. Mentions like '66th Annual Grammy Awards' or '2024 Grammys' are acceptable.",
        extra_prerequisites=[ref_present],
    )


async def add_release_compliance(evaluator: Evaluator, parent, ext: AlbumExtraction) -> None:
    node = evaluator.add_sequential(
        id="Release_Date_Compliance",
        desc="The album was released within the specified timeframe",
        parent=parent,
        critical=True,
    )

    # Presence of release date reference URL(s)
    ref_present = evaluator.add_custom_node(
        result=bool(ext.release_date_sources),
        id="Release_Date_Reference_URL",
        desc="Reference URL provided documenting the album release date",
        parent=node,
        critical=True,
    )

    # Verify release date within 2022-01-01 to 2023-12-31
    leaf = evaluator.add_leaf(
        id="Release_Date_Verification",
        desc="The album was released between January 1, 2022 and December 31, 2023 (inclusive)",
        parent=node,
        critical=True,
    )
    album_title = ext.album_title or "the album"
    rd_text = ext.release_date or "the stated release date in the cited source(s)"
    sources = _merge_urls(ext.release_date_sources, ext.album_sources)
    claim = (
        f"The album '{album_title}' was released on {rd_text}, which falls between "
        f"January 1, 2022 and December 31, 2023 (inclusive)."
    )
    await evaluator.verify(
        claim=claim,
        node=leaf,
        sources=sources,
        additional_instruction="Use the cited page to determine the album's release date. "
                              "If multiple formats/regions are listed, accept the earliest official release within 2022–2023.",
        extra_prerequisites=[ref_present],
    )


async def add_festival_compliance(evaluator: Evaluator, parent, ext: AlbumExtraction) -> None:
    node = evaluator.add_sequential(
        id="Festival_Performance_Compliance",
        desc="The artist headlined a qualifying festival performance in 2024",
        parent=parent,
        critical=True,
    )

    # Presence of festival reference URL(s)
    ref_present = evaluator.add_custom_node(
        result=bool(ext.festival_sources),
        id="Festival_Reference_URL",
        desc="Reference URL provided documenting the festival performance details",
        parent=node,
        critical=True,
    )

    # Headliner status check
    headliner_leaf = evaluator.add_leaf(
        id="Festival_Headliner_Status",
        desc="The artist was listed as a headliner at Coachella 2024, Glastonbury 2024, or Lollapalooza 2024",
        parent=node,
        critical=True,
    )
    artist = ext.artist_name or "the artist"
    fest = ext.festival_name or "the festival"
    headliner_claim = (
        f"{artist} was a headliner at {fest} in 2024. The festival must be one of: "
        "Coachella Valley Music and Arts Festival, Glastonbury Festival, or Lollapalooza."
    )
    await evaluator.verify(
        claim=headliner_claim,
        node=headliner_leaf,
        sources=ext.festival_sources,
        additional_instruction="Verify the artist appears as a headliner (top billing) in the 2024 lineup for one of the allowed festivals. "
                              "Lineup posters or official schedules explicitly naming headliners are acceptable.",
        extra_prerequisites=[ref_present],
    )

    # Performance timeframe check
    timeframe_leaf = evaluator.add_leaf(
        id="Performance_Timeframe",
        desc="The festival headline performance occurred between April 1, 2024 and August 31, 2024 (inclusive)",
        parent=node,
        critical=True,
    )
    perf_when = ext.performance_date or "the date of the artist's headliner set at the festival in 2024"
    timeframe_claim = (
        f"{artist}'s headliner performance at {fest} took place on {perf_when}, which is between "
        f"April 1, 2024 and August 31, 2024 (inclusive)."
    )
    await evaluator.verify(
        claim=timeframe_claim,
        node=timeframe_leaf,
        sources=ext.festival_sources,
        additional_instruction="If an exact set date is not explicitly stated, infer from official schedule/lineup dates that the headliner set fell within the given window.",
        extra_prerequisites=[ref_present, headliner_leaf],
    )

    # Venue capacity range check
    capacity_leaf = evaluator.add_leaf(
        id="Venue_Capacity_Range",
        desc="The festival venue has a capacity between 15,000 and 100,000 attendees",
        parent=node,
        critical=True,
    )
    venue = ext.venue_name or "the festival venue"
    capacity_claim = (
        f"The capacity of {venue} is between 15,000 and 100,000 attendees (consider daily attendance if applicable)."
    )
    cap_sources = _merge_urls(ext.venue_capacity_sources, ext.festival_sources)
    await evaluator.verify(
        claim=capacity_claim,
        node=capacity_leaf,
        sources=cap_sources,
        additional_instruction="Use cited pages to determine the venue or daily festival capacity. "
                              "Approximate values within the range are acceptable; focus on daily capacity if multiple numbers exist.",
        extra_prerequisites=[ref_present],
    )


async def add_producer_compliance(evaluator: Evaluator, parent, ext: AlbumExtraction) -> None:
    node = evaluator.add_sequential(
        id="Producer_Credential_Compliance",
        desc="At least one producer on the album has won Grammy Producer of the Year in the 2020s",
        parent=parent,
        critical=True,
    )

    # Presence of at least some producer-related URL(s)
    any_producer_urls = any((p.award_sources or p.album_credit_sources) for p in (ext.producers or []))
    ref_present = evaluator.add_custom_node(
        result=bool(any_producer_urls),
        id="Producer_Reference_URL",
        desc="Reference URL provided documenting the producer's Grammy win and album credit",
        parent=node,
        critical=True,
    )

    # Identify a qualifying producer (has a 2020s award year and at least some sources)
    winner = _find_producer_with_2020s_award(ext.producers or [])
    prod_leaf = evaluator.add_leaf(
        id="Producer_Award_Verification",
        desc="At least one producer credited on the album won the Grammy Award for Producer of the Year, Non-Classical, at least once between 2020-2029",
        parent=node,
        critical=True,
    )

    album_title = ext.album_title or "the album"
    if winner:
        pname = winner["name"]
        year = winner["year"]
        psources = winner["sources"]
        claim = (
            f"{pname}, who is credited as a producer on '{album_title}', won the Grammy Award for Producer of the Year, "
            f"Non-Classical in {year}, which is within 2020–2029."
        )
        await evaluator.verify(
            claim=claim,
            node=prod_leaf,
            sources=psources,
            additional_instruction="The evidence page should explicitly support BOTH: (1) this person is credited as a producer on the album, "
                                  "(2) this person won the Producer of the Year, Non-Classical Grammy in the specified 2020s year.",
            extra_prerequisites=[ref_present],
        )
    else:
        # Fall back to a generic claim using whatever URLs exist (likely to fail, but keeps structure consistent)
        merged = _merge_urls(*[p.award_sources for p in ext.producers or []], *[p.album_credit_sources for p in ext.producers or []])
        fallback_claim = (
            f"At least one credited producer on '{album_title}' won the Grammy Award for Producer of the Year, "
            f"Non-Classical during the 2020s (2020–2029)."
        )
        await evaluator.verify(
            claim=fallback_claim,
            node=prod_leaf,
            sources=merged if merged else None,
            additional_instruction="Confirm that at least one named producer is both credited on the album and has a Producer of the Year (Non-Classical) Grammy win in the 2020s.",
            extra_prerequisites=[ref_present],
        )


# --------------------------------------------------------------------------- #
# Tree builder                                                                #
# --------------------------------------------------------------------------- #
async def build_verification_tree(evaluator: Evaluator, ext: AlbumExtraction) -> None:
    # Top-level container under root to mirror rubric; keep root non-critical by framework design.
    album_node = evaluator.add_parallel(
        id="Album_Identification",
        desc="The album correctly satisfies all specified requirements",
        parent=evaluator.root,
        critical=True,
    )

    # Four critical compliance branches
    await add_grammy_compliance(evaluator, album_node, ext)
    await add_release_compliance(evaluator, album_node, ext)
    await add_festival_compliance(evaluator, album_node, ext)
    await add_producer_compliance(evaluator, album_node, ext)


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
    Evaluate an answer for the album identification task (2024 Grammys + 2024 festival headline + producer award).
    Returns a structured summary with the verification tree and final score.
    """
    # Initialize evaluator and root
    evaluator = Evaluator()
    evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Root aggregation
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

    # Extract structured information from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_album(),
        template_class=AlbumExtraction,
        extraction_name="album_extraction",
    )

    # Record simple custom info for visibility (not ground truth)
    evaluator.add_custom_info(
        info={
            "allowed_festivals": list(ALLOWED_FESTIVALS),
            "release_window": [RELEASE_START, RELEASE_END],
            "headline_window": [HEADLINE_START, HEADLINE_END],
        },
        info_type="constraints",
        info_name="constraint_windows_and_festivals",
    )

    # Build verification tree and run checks
    await build_verification_tree(evaluator, extracted)

    # Return standardized summary
    return evaluator.get_summary()