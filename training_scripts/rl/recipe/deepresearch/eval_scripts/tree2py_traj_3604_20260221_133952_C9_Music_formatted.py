import asyncio
import logging
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy

# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "grammy_coachella_2026"
TASK_DESCRIPTION = (
    "Identify 2 artists who released albums during the 68th Annual Grammy Awards eligibility period "
    "(August 31, 2024 through August 30, 2025), received Album of the Year nominations for the 68th "
    "Grammy Awards, and are confirmed headliners for Coachella Valley Music and Arts Festival 2026. "
    "For each artist, provide: (1) Artist name, (2) Nominated album title, (3) Album release date, "
    "(4) Coachella 2026 performance weekend dates, (5) Festival venue name, city, and state, (6) Venue capacity, "
    "and (7) Reference URLs supporting the Grammy nomination, album release date, Coachella performance confirmation, "
    "and venue capacity information."
)

GRAMMY_ELIGIBILITY_START = "August 31, 2024"
GRAMMY_ELIGIBILITY_END = "August 30, 2025"

COACHELLA_WEEKEND1_RANGE = "April 10–12, 2026"
COACHELLA_WEEKEND2_RANGE = "April 17–19, 2026"
EXPECTED_VENUE_NAME = "Empire Polo Club"
EXPECTED_VENUE_CITY = "Indio"
EXPECTED_VENUE_STATE = "California"
EXPECTED_VENUE_CAPACITY = "90,000"

# --------------------------------------------------------------------------- #
# Data models                                                                 #
# --------------------------------------------------------------------------- #
class ArtistRecord(BaseModel):
    artist_name: Optional[str] = None
    nominated_album_title: Optional[str] = None
    album_release_date: Optional[str] = None

    weekend1_dates: Optional[str] = None  # e.g., "April 10-12, 2026" or "Apr 10–12, 2026"
    weekend2_dates: Optional[str] = None  # e.g., "April 17-19, 2026"

    venue_name: Optional[str] = None
    venue_city: Optional[str] = None
    venue_state: Optional[str] = None
    venue_capacity: Optional[str] = None

    grammy_urls: List[str] = Field(default_factory=list)     # Support nomination + album nominated work
    album_urls: List[str] = Field(default_factory=list)      # Support album release date
    coachella_urls: List[str] = Field(default_factory=list)  # Support headliner confirmation + (optionally) dates
    venue_urls: List[str] = Field(default_factory=list)      # Support venue capacity (and optionally location/name)


class ArtistsExtraction(BaseModel):
    artists: List[ArtistRecord] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_artists() -> str:
    return """
    Extract up to TWO artists from the answer who meet all of the following criteria:
    – They released an album during the 68th Annual Grammy eligibility period (August 31, 2024 through August 30, 2025).
    – They received an Album of the Year nomination for the 68th Grammy Awards.
    – They are confirmed headliners for Coachella Valley Music and Arts Festival 2026.

    For each artist, extract exactly the following fields as they appear in the answer:
    1) artist_name: The name of the artist.
    2) nominated_album_title: The title of the nominated album.
    3) album_release_date: The album's release date string (do NOT convert; keep original formatting).
    4) weekend1_dates: The stated performance dates for Coachella 2026 Weekend 1 (e.g., "April 10-12, 2026").
    5) weekend2_dates: The stated performance dates for Coachella 2026 Weekend 2 (e.g., "April 17-19, 2026").
    6) venue_name: The festival venue name.
    7) venue_city: The city of the venue.
    8) venue_state: The state of the venue.
    9) venue_capacity: The venue capacity string (keep formatting; e.g., "90,000").
    10) grammy_urls: Array of URLs that confirm the Album of the Year nomination and the nominated album.
    11) album_urls: Array of URLs that confirm the album release date.
    12) coachella_urls: Array of URLs that confirm Coachella 2026 headliner status (and optionally performance dates).
    13) venue_urls: Array of URLs that confirm the venue capacity (and optionally name/location).

    Rules:
    – Extract ONLY what is explicitly present in the answer; do not invent or infer missing data.
    – If an item is not provided, set it to null (or an empty array for URL fields).
    – Accept URLs presented as plain links or markdown; output full URL strings only.
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _first_n_artists(data: ArtistsExtraction, n: int = 2) -> List[ArtistRecord]:
    """Return the first n artists, padding with empty records if necessary."""
    artists = list(data.artists[:n])
    while len(artists) < n:
        artists.append(ArtistRecord())
    return artists


def _nonempty_str(s: Optional[str]) -> bool:
    return bool(s and s.strip())


def _has_urls(urls: Optional[List[str]]) -> bool:
    return bool(urls and len(urls) > 0)


def _combine_sources(*lists: List[str]) -> List[str]:
    combined: List[str] = []
    for lst in lists:
        combined.extend(lst or [])
    # Remove duplicates while preserving order
    seen = set()
    unique = []
    for url in combined:
        if url and url not in seen:
            seen.add(url)
            unique.append(url)
    return unique


# --------------------------------------------------------------------------- #
# Verification for a single artist                                            #
# --------------------------------------------------------------------------- #
async def verify_artist(
    evaluator: Evaluator,
    parent_node,
    artist: ArtistRecord,
    index: int,
) -> None:
    """
    Build and verify the subtree for one artist according to the rubric.
    """

    # Create top-level node for the artist (non-critical to allow partial credit per artist)
    artist_node = evaluator.add_parallel(
        id=f"artist_{index + 1}",
        desc="Qualifying artist with complete information" if index == 0 else "Second qualifying artist with complete information",
        parent=parent_node,
        critical=False
    )

    # 1) Artist identification and Grammy nomination status (critical)
    artist_ident_node = evaluator.add_parallel(
        id=f"artist_{index + 1}_artist_identification",
        desc="Artist name and Grammy nomination status",
        parent=artist_node,
        critical=True
    )

    # 1.1) Artist name provided (critical existence)
    evaluator.add_custom_node(
        result=_nonempty_str(artist.artist_name),
        id=f"artist_{index + 1}_artist_name_provided",
        desc="The artist's name is clearly stated",
        parent=artist_ident_node,
        critical=True
    )

    # 1.2) Grammy nomination verified (critical)
    grammy_verif_node = evaluator.add_parallel(
        id=f"artist_{index + 1}_grammy_nomination_verified",
        desc="Artist's Grammy nomination is verified",
        parent=artist_ident_node,
        critical=True
    )

    # 1.2.1) Grammy nomination confirmed (critical - verify by URLs)
    grammy_confirm_leaf = evaluator.add_leaf(
        id=f"artist_{index + 1}_grammy_nomination_confirmed",
        desc="Artist received Album of the Year nomination for 68th Grammy Awards",
        parent=grammy_verif_node,
        critical=True
    )
    grammy_claim = (
        f"{artist.artist_name or 'The artist'} received an Album of the Year nomination for the 68th Annual Grammy Awards."
    )
    await evaluator.verify(
        claim=grammy_claim,
        node=grammy_confirm_leaf,
        sources=artist.grammy_urls,
        additional_instruction="Verify using the cited pages that the artist is listed among Album of the Year nominees for the 68th Grammy Awards. Allow minor formatting differences."
    )

    # 1.2.2) Grammy reference URL provided (critical existence)
    evaluator.add_custom_node(
        result=_has_urls(artist.grammy_urls),
        id=f"artist_{index + 1}_grammy_reference_url",
        desc="URL reference confirming Grammy nomination",
        parent=grammy_verif_node,
        critical=True
    )

    # 2) Album details and eligibility (critical)
    album_details_node = evaluator.add_parallel(
        id=f"artist_{index + 1}_album_details",
        desc="Album information and eligibility",
        parent=artist_node,
        critical=True
    )

    # 2.1) Album identification (critical)
    album_ident_node = evaluator.add_parallel(
        id=f"artist_{index + 1}_album_identification",
        desc="Album title and nomination status",
        parent=album_details_node,
        critical=True
    )

    # 2.1.1) Album title stated (critical existence)
    evaluator.add_custom_node(
        result=_nonempty_str(artist.nominated_album_title),
        id=f"artist_{index + 1}_album_title_stated",
        desc="The nominated album title is provided",
        parent=album_ident_node,
        critical=True
    )

    # 2.1.2) Album is nominated work (critical - verify by Grammy URLs)
    nominated_work_leaf = evaluator.add_leaf(
        id=f"artist_{index + 1}_album_is_nominated_work",
        desc="Album is confirmed as the work nominated for Album of the Year",
        parent=album_ident_node,
        critical=True
    )
    nominated_work_claim = (
        f"The album '{artist.nominated_album_title or 'the album'}' is the work nominated for Album of the Year for {artist.artist_name or 'the artist'} at the 68th Grammy Awards."
    )
    await evaluator.verify(
        claim=nominated_work_claim,
        node=nominated_work_leaf,
        sources=artist.grammy_urls,
        additional_instruction="Confirm that the specific album title is the nominated Album of the Year entry for the named artist."
    )

    # 2.2) Album release validation (critical)
    album_release_node = evaluator.add_parallel(
        id=f"artist_{index + 1}_album_release_validation",
        desc="Album release date and eligibility verification",
        parent=album_details_node,
        critical=True
    )

    # 2.2.1) Release date provided (critical existence)
    evaluator.add_custom_node(
        result=_nonempty_str(artist.album_release_date),
        id=f"artist_{index + 1}_release_date_provided",
        desc="Album release date is stated",
        parent=album_release_node,
        critical=True
    )

    # 2.2.2) Eligibility period check (critical - simple logical verification)
    eligibility_leaf = evaluator.add_leaf(
        id=f"artist_{index + 1}_eligibility_period_check",
        desc="Album released between August 31, 2024 and August 30, 2025",
        parent=album_release_node,
        critical=True
    )
    eligibility_claim = (
        f"The release date '{artist.album_release_date or 'unknown'}' falls within the Grammy eligibility period "
        f"from {GRAMMY_ELIGIBILITY_START} through {GRAMMY_ELIGIBILITY_END} (inclusive)."
    )
    await evaluator.verify(
        claim=eligibility_claim,
        node=eligibility_leaf,
        additional_instruction="Judge whether the provided date string denotes a date within the inclusive range Aug 31, 2024 to Aug 30, 2025. Use common formats; allow minor formatting variations."
    )

    # 2.2.3) Album reference URL (critical - verify release date by URLs)
    album_ref_leaf = evaluator.add_leaf(
        id=f"artist_{index + 1}_album_reference_url",
        desc="URL reference supporting album release date",
        parent=album_release_node,
        critical=True
    )
    release_claim = (
        f"The album '{artist.nominated_album_title or 'the album'}' by {artist.artist_name or 'the artist'} was released on {artist.album_release_date or 'unknown'}."
    )
    await evaluator.verify(
        claim=release_claim,
        node=album_ref_leaf,
        sources=artist.album_urls,
        additional_instruction="Verify that the cited source explicitly states the album's release date as provided."
    )

    # 3) Coachella 2026 headliner confirmation (critical)
    coachella_node = evaluator.add_parallel(
        id=f"artist_{index + 1}_coachella_performance",
        desc="Coachella 2026 headliner confirmation",
        parent=artist_node,
        critical=True
    )

    # 3.1) Headliner confirmation (critical)
    headliner_node = evaluator.add_parallel(
        id=f"artist_{index + 1}_headliner_confirmation",
        desc="Artist confirmed as Coachella 2026 headliner",
        parent=coachella_node,
        critical=True
    )

    # 3.1.1) Coachella headliner confirmed (critical - verify by URLs)
    headliner_leaf = evaluator.add_leaf(
        id=f"artist_{index + 1}_coachella_confirmation",
        desc="Artist is confirmed as Coachella 2026 headliner",
        parent=headliner_node,
        critical=True
    )
    headliner_claim = (
        f"{artist.artist_name or 'The artist'} is confirmed as a headliner for Coachella Valley Music and Arts Festival 2026."
    )
    await evaluator.verify(
        claim=headliner_claim,
        node=headliner_leaf,
        sources=artist.coachella_urls,
        additional_instruction="Confirm headliner status via official announcements, reliable news outlets, or Coachella sources. Allow minor formatting differences."
    )

    # 3.1.2) Coachella reference URL provided (critical existence)
    evaluator.add_custom_node(
        result=_has_urls(artist.coachella_urls),
        id=f"artist_{index + 1}_coachella_reference_url",
        desc="URL reference confirming Coachella 2026 headliner status",
        parent=headliner_node,
        critical=True
    )

    # 3.2) Performance dates (critical)
    perf_dates_node = evaluator.add_parallel(
        id=f"artist_{index + 1}_performance_dates",
        desc="Performance date information",
        parent=coachella_node,
        critical=True
    )

    # 3.2.1) Weekend dates provided for both weekends (critical existence)
    evaluator.add_custom_node(
        result=_nonempty_str(artist.weekend1_dates) and _nonempty_str(artist.weekend2_dates),
        id=f"artist_{index + 1}_weekend_dates_provided",
        desc="Specific performance dates for both weekends are stated",
        parent=perf_dates_node,
        critical=True
    )

    # 3.2.2) Dates match Coachella 2026 weekends (critical - logical check)
    dates_match_leaf = evaluator.add_leaf(
        id=f"artist_{index + 1}_dates_match_coachella_2026",
        desc="Dates fall within April 10-12 (Weekend 1) or April 17-19 (Weekend 2), 2026",
        parent=perf_dates_node,
        critical=True
    )
    dates_match_claim = (
        f"The provided weekend dates '{artist.weekend1_dates or 'unknown'}' and '{artist.weekend2_dates or 'unknown'}' "
        f"fall within {COACHELLA_WEEKEND1_RANGE} (Weekend 1) and {COACHELLA_WEEKEND2_RANGE} (Weekend 2)."
    )
    await evaluator.verify(
        claim=dates_match_claim,
        node=dates_match_leaf,
        additional_instruction="Check whether the provided date strings plausibly denote dates within those exact April 2026 weekend windows; allow punctuation and dash variations."
    )

    # 4) Venue information (critical)
    venue_info_node = evaluator.add_parallel(
        id=f"artist_{index + 1}_venue_information",
        desc="Festival venue details",
        parent=artist_node,
        critical=True
    )

    # 4.1) Venue identification (critical)
    venue_ident_node = evaluator.add_parallel(
        id=f"artist_{index + 1}_venue_identification",
        desc="Venue name and location",
        parent=venue_info_node,
        critical=True
    )

    # 4.1.1) Venue name is stated as Empire Polo Club (critical - verify by sources if available)
    venue_name_leaf = evaluator.add_leaf(
        id=f"artist_{index + 1}_venue_name_provided",
        desc="Festival venue name is stated (Empire Polo Club)",
        parent=venue_ident_node,
        critical=True
    )
    venue_name_claim = (
        f"The festival venue is '{EXPECTED_VENUE_NAME}'."
    )
    await evaluator.verify(
        claim=venue_name_claim,
        node=venue_name_leaf,
        sources=_combine_sources(artist.coachella_urls, artist.venue_urls),
        additional_instruction="Use official Coachella or reliable venue pages to confirm the venue name is Empire Polo Club."
    )

    # 4.1.2) Venue location (Indio, California) provided (critical - verify by sources)
    venue_loc_leaf = evaluator.add_leaf(
        id=f"artist_{index + 1}_venue_location_provided",
        desc="Venue city (Indio) and state (California) are provided",
        parent=venue_ident_node,
        critical=True
    )
    venue_loc_claim = (
        f"Coachella takes place in {EXPECTED_VENUE_CITY}, {EXPECTED_VENUE_STATE}."
    )
    await evaluator.verify(
        claim=venue_loc_claim,
        node=venue_loc_leaf,
        sources=_combine_sources(artist.coachella_urls, artist.venue_urls),
        additional_instruction="Confirm venue location is Indio, California. Allow minor formatting variants (e.g., 'Indio, CA')."
    )

    # 4.2) Venue capacity verification (critical)
    venue_cap_node = evaluator.add_parallel(
        id=f"artist_{index + 1}_venue_capacity_verification",
        desc="Venue capacity information",
        parent=venue_info_node,
        critical=True
    )

    # 4.2.1) Venue capacity number provided (critical existence)
    evaluator.add_custom_node(
        result=_nonempty_str(artist.venue_capacity),
        id=f"artist_{index + 1}_venue_capacity_stated",
        desc="Venue capacity number is provided",
        parent=venue_cap_node,
        critical=True
    )

    # 4.2.2) Capacity is 90,000 (critical - logical check)
    capacity_match_leaf = evaluator.add_leaf(
        id=f"artist_{index + 1}_capacity_is_90000",
        desc="Stated capacity matches Empire Polo Club capacity of 90,000",
        parent=venue_cap_node,
        critical=True
    )
    capacity_claim = (
        f"The capacity of {EXPECTED_VENUE_NAME} is approximately {EXPECTED_VENUE_CAPACITY}."
    )
    await evaluator.verify(
        claim=capacity_claim,
        node=capacity_match_leaf,
        additional_instruction="Treat '90,000' and minor numeric formatting variants (e.g., 90000) as equivalent."
    )

    # 4.2.3) Venue reference URL (critical - verify capacity by URLs)
    venue_ref_leaf = evaluator.add_leaf(
        id=f"artist_{index + 1}_venue_reference_url",
        desc="URL reference confirming venue capacity",
        parent=venue_cap_node,
        critical=True
    )
    venue_ref_claim = (
        f"The capacity of {EXPECTED_VENUE_NAME} is about {EXPECTED_VENUE_CAPACITY}."
    )
    await evaluator.verify(
        claim=venue_ref_claim,
        node=venue_ref_leaf,
        sources=artist.venue_urls,
        additional_instruction="Use official venue pages, credible festival documentation, or reliable sources to confirm capacity near 90,000."
    )


# --------------------------------------------------------------------------- #
# Main evaluation entry                                                       #
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
    Evaluate an answer for the Grammy/Coachella 2026 task using the Mind2Web2 framework.
    """
    # Initialize evaluator; root is non-critical to allow partial credit across artists
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,
        agent_name=agent_name,
        answer_name=answer_name,
        client=client,
        task_description="Identify 2 artists who released Grammy-eligible albums and are confirmed Coachella 2026 headliners",
        answer=answer,
        global_cache=cache,
        global_semaphore=semaphore,
        logger=logger,
        default_model=model,
    )

    # Record ground truth context / constraints
    evaluator.add_ground_truth({
        "grammy_eligibility_period": {
            "start": GRAMMY_ELIGIBILITY_START,
            "end": GRAMMY_ELIGIBILITY_END
        },
        "coachella_2026_weekends": {
            "weekend_1": COACHELLA_WEEKEND1_RANGE,
            "weekend_2": COACHELLA_WEEKEND2_RANGE
        },
        "venue_expected": {
            "name": EXPECTED_VENUE_NAME,
            "city": EXPECTED_VENUE_CITY,
            "state": EXPECTED_VENUE_STATE,
            "capacity": EXPECTED_VENUE_CAPACITY
        }
    })

    # Extract the structured information from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_artists(),
        template_class=ArtistsExtraction,
        extraction_name="artists_extraction",
    )

    first_two = _first_n_artists(extracted, n=2)

    # Build verification trees for both artists
    tasks = []
    for idx, artist in enumerate(first_two):
        tasks.append(verify_artist(evaluator, root, artist, idx))

    await asyncio.gather(*tasks)

    # Return structured summary
    return evaluator.get_summary()