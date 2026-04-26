import asyncio
import logging
from typing import Any, List, Optional, Dict

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "rodeo2025_country_debut_spotify"
TASK_DESCRIPTION = (
    "Identify three country or contemporary country music artists who are making their debut performance at the 2025 "
    "Houston Livestock Show and Rodeo (RodeoHouston 2025, scheduled March 4-23, 2025 at NRG Stadium). For each artist, "
    "the following criteria must be met: (1) The artist must be officially listed in the 2025 RodeoHouston entertainment "
    "lineup as published on rodeohouston.com; (2) The artist must be making their first-ever appearance at RodeoHouston "
    "(debut performer in 2025); (3) The artist's scheduled performance date must fall between March 4 and March 23, 2025; "
    "(4) The artist must currently have between 5 million and 15 million monthly listeners on Spotify; (5) The artist must "
    "have an official Spotify artist profile accessible via the format open.spotify.com/artist/[artist_ID]; "
    "(6) The artist's primary genre must be Country or Contemporary Country as indicated on their Spotify profile; "
    "(7) The artist must have released at least 3 songs available on their Spotify profile. For each artist, provide: the full "
    "name, scheduled performance date, a direct link to the Spotify artist profile, the current number of monthly listeners on "
    "Spotify (formatted in millions with one decimal place), and a direct link to the official RodeoHouston 2025 lineup page."
)


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class ArtistItem(BaseModel):
    name: Optional[str] = None
    performance_date: Optional[str] = None  # e.g., "March 12, 2025" or "Mar 12, 2025"
    rodeo_lineup_url: Optional[str] = None  # rodeohouston.com URL that confirms performance
    spotify_url: Optional[str] = None       # open.spotify.com/artist/[ID]
    monthly_listeners_millions: Optional[str] = None  # e.g., "7.8M" or "7.8 million"
    genre_claim: Optional[str] = None  # If the answer states a genre; optional
    supporting_urls: List[str] = Field(default_factory=list)  # Any other sources the answer cites
    setlist_urls: List[str] = Field(default_factory=list)     # setlist.fm URLs, if any


class ArtistsExtraction(BaseModel):
    artists: List[ArtistItem] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_artists() -> str:
    return """
    Extract up to three artists that the answer claims meet the RodeoHouston 2025 debut and Spotify-related criteria.
    Keep the original order in which the artists appear in the answer. For each artist, extract the following fields:

    - name: The artist's full name (string).
    - performance_date: The artist's scheduled performance date at RodeoHouston 2025 as written in the answer (string; keep original format).
    - rodeo_lineup_url: The direct URL on rodeohouston.com that confirms the artist's RodeoHouston 2025 performance (must be a URL explicitly present in the answer).
    - spotify_url: The direct URL to the official Spotify artist profile in the exact format open.spotify.com/artist/[artist_ID] (must be a URL explicitly present in the answer).
    - monthly_listeners_millions: The current number of monthly listeners as stated in the answer (formatted in millions with one decimal when available; string like '7.8M' or '7.8 million'; if absent, set to null).
    - genre_claim: The primary genre as claimed in the answer for the artist (string; if absent, set to null).
    - supporting_urls: A list of any additional URLs the answer cites for this artist (press releases, news, social posts, Wikipedia, etc.). Include only URLs explicitly present in the answer. If none, use an empty list.
    - setlist_urls: A list of setlist.fm URLs explicitly present in the answer for this artist. If none, use an empty list.

    Return a JSON object with exactly one key 'artists', which is an array of up to 3 artist objects with the fields described above.
    If the answer lists more than three artists, only include the first three. If fewer, include whatever is present.
    Do not fabricate or infer any missing URLs or fields. Only use information explicitly stated in the answer text.
    """


# --------------------------------------------------------------------------- #
# Verification helpers                                                        #
# --------------------------------------------------------------------------- #
async def verify_single_artist(
    evaluator: Evaluator,
    parent_node,
    artist: ArtistItem,
    artist_index: int,
) -> None:
    """
    Build the verification subtree and run checks for a single artist.
    """
    index_human = artist_index + 1
    artist_name = artist.name or "(missing name)"

    # Create the artist-level parent parallel node
    artist_node = evaluator.add_parallel(
        id=f"artist_{index_human}",
        desc=f"Artist #{index_human} qualifying checks for RodeoHouston 2025 debut and Spotify criteria",
        parent=parent_node,
        critical=False
    )

    # Gating: Minimal fields existence (critical) to avoid meaningless downstream checks
    required_fields_ok = bool(artist.name and artist.rodeo_lineup_url and artist.spotify_url)
    evaluator.add_custom_node(
        result=required_fields_ok,
        id=f"artist_{index_human}_fields_present",
        desc=f"Artist #{index_human}: Required core fields present (name, rodeohouston lineup URL, spotify artist URL)",
        parent=artist_node,
        critical=True
    )

    # 1) Official RodeoHouston performer listing
    rh_performer_node = evaluator.add_leaf(
        id=f"artist_{index_human}_rodeohouston_performer",
        desc="The artist is officially listed as a performer in the 2025 RodeoHouston lineup on rodeohouston.com",
        parent=artist_node,
        critical=True
    )
    claim_rh_perf = (
        f"The official rodeohouston.com lineup page confirms that {artist_name} is scheduled to perform at "
        f"RodeoHouston 2025."
    )
    await evaluator.verify(
        claim=claim_rh_perf,
        node=rh_performer_node,
        sources=artist.rodeo_lineup_url,
        additional_instruction=(
            "Verify the artist's name appears on the page as part of the 2025 RodeoHouston entertainment lineup. "
            "Allow minor casing/formatting variations of the artist name."
        ),
    )

    # 2) Debut status: first-ever appearance at RodeoHouston in 2025
    debut_node = evaluator.add_leaf(
        id=f"artist_{index_human}_debut_status",
        desc="The artist is making their first-ever performance at RodeoHouston in 2025 (debut performer)",
        parent=artist_node,
        critical=True
    )
    debut_sources: List[str] = []
    if artist.rodeo_lineup_url:
        debut_sources.append(artist.rodeo_lineup_url)
    if artist.supporting_urls:
        debut_sources.extend(artist.supporting_urls)
    claim_debut = (
        f"The provided sources indicate that {artist_name} will make their first-ever appearance at RodeoHouston "
        f"in 2025 (i.e., a Rodeo debut)."
    )
    await evaluator.verify(
        claim=claim_debut,
        node=debut_node,
        sources=debut_sources if debut_sources else None,
        additional_instruction=(
            "Look for explicit phrases such as 'Rodeo debut', 'first time at RodeoHouston', 'first appearance', "
            "or equivalent language on the provided sources. If none of the sources indicate debut status, mark as not supported."
        ),
    )

    # 3) Performance date must be between March 4 and March 23, 2025
    date_node = evaluator.add_leaf(
        id=f"artist_{index_human}_performance_date",
        desc="The artist's performance is scheduled between March 4-23, 2025",
        parent=artist_node,
        critical=True
    )
    perf_date_str = artist.performance_date or "(date missing)"
    claim_date = (
        f"The rodeohouston.com lineup page shows that {artist_name} is scheduled to perform on {perf_date_str}, "
        f"and this date falls between March 4 and March 23, 2025 (inclusive)."
    )
    await evaluator.verify(
        claim=claim_date,
        node=date_node,
        sources=artist.rodeo_lineup_url,
        additional_instruction=(
            "Confirm the page lists a specific 2025 date for this artist. Accept reasonable date formats (e.g., "
            "'March 7, 2025', 'Mar 7, 2025', '03/07/2025'). Ensure the date is within Mar 4–Mar 23, 2025."
        ),
    )

    # 4) Monthly listeners range (5M–15M) on Spotify
    listeners_node = evaluator.add_leaf(
        id=f"artist_{index_human}_monthly_listeners",
        desc="The artist has between 5 million and 15 million monthly listeners on Spotify",
        parent=artist_node,
        critical=True
    )
    claim_listeners = (
        "This Spotify artist page shows that the artist currently has between 5,000,000 and 15,000,000 monthly listeners."
    )
    await evaluator.verify(
        claim=claim_listeners,
        node=listeners_node,
        sources=artist.spotify_url,
        additional_instruction=(
            "Use the monthly listeners figure displayed near the top of the Spotify artist page. "
            "Small day-to-day fluctuations and rounding are acceptable; judge whether it clearly falls between 5M and 15M."
        ),
    )

    # 5) Spotify profile URL validity (official artist profile page)
    spotify_profile_node = evaluator.add_leaf(
        id=f"artist_{index_human}_spotify_profile",
        desc="The artist has an official Spotify artist profile accessible via open.spotify.com/artist/[ID] with the profile URL provided",
        parent=artist_node,
        critical=True
    )
    claim_spotify = (
        f"The provided URL is the official Spotify artist profile page for {artist_name} "
        f"(located under open.spotify.com/artist/...)."
    )
    await evaluator.verify(
        claim=claim_spotify,
        node=spotify_profile_node,
        sources=artist.spotify_url,
        additional_instruction=(
            "Confirm this is an artist profile (not an album/track/playlist). The URL path should contain '/artist/'. "
            "The page should present the artist's header and profile elements."
        ),
    )

    # 6) Genre is Country or Contemporary Country on Spotify profile
    genre_node = evaluator.add_leaf(
        id=f"artist_{index_human}_genre",
        desc="The artist's primary genre is Country or Contemporary Country as indicated on their Spotify profile",
        parent=artist_node,
        critical=True
    )
    claim_genre = (
        "This Spotify artist profile indicates the artist's primary genre is Country or Contemporary Country (or a clear variant including the word 'Country')."
    )
    await evaluator.verify(
        claim=claim_genre,
        node=genre_node,
        sources=artist.spotify_url,
        additional_instruction=(
            "Look on the Spotify artist page for textual indicators of genre (e.g., About section or tags). "
            "Accept clear variants containing 'Country' such as 'Country', 'Contemporary Country', 'Country Pop'. "
            "If the page provides no genre information at all, mark as not supported."
        ),
    )

    # 7) Track count: at least 3 songs on Spotify
    track_count_node = evaluator.add_leaf(
        id=f"artist_{index_human}_track_count",
        desc="The artist has released at least 3 songs available on their Spotify profile",
        parent=artist_node,
        critical=True
    )
    claim_tracks = (
        "This Spotify artist profile shows at least 3 tracks available (for example, via 'Popular' tracks or discography)."
    )
    await evaluator.verify(
        claim=claim_tracks,
        node=track_count_node,
        sources=artist.spotify_url,
        additional_instruction=(
            "Check the artist page for a list of 'Popular' songs or discography content. "
            "If at least 3 distinct tracks are visible/available, consider this supported."
        ),
    )

    # 8) Non-critical: setlist.fm presence in 2024 or 2025
    setlist_node = evaluator.add_leaf(
        id=f"artist_{index_human}_setlist_presence",
        desc="The artist has at least one documented concert performance in 2024 or 2025 recorded on setlist.fm",
        parent=artist_node,
        critical=False  # Non-critical per rubric
    )
    claim_setlist = (
        f"Setlist.fm shows at least one concert performance for {artist_name} in either 2024 or 2025."
    )
    await evaluator.verify(
        claim=claim_setlist,
        node=setlist_node,
        sources=artist.setlist_urls if artist.setlist_urls else None,
        additional_instruction=(
            "Verify on provided setlist.fm page(s) that there is at least one event with a date in 2024 or 2025. "
            "If no setlist.fm URL is provided, or the page shows no such events, mark as not supported."
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
    Evaluate an answer for the RodeoHouston 2025 debut artist task.
    """
    # Initialize evaluator (root: parallel aggregation)
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

    # Record contextual ground truth window (for transparency only)
    evaluator.add_ground_truth({
        "event": "RodeoHouston 2025",
        "venue": "NRG Stadium",
        "date_window_inclusive": ["2025-03-04", "2025-03-23"]
    })

    # Extract up to 3 artists from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_artists(),
        template_class=ArtistsExtraction,
        extraction_name="artists_extraction",
    )

    # Normalize to exactly 3 artist slots
    artists: List[ArtistItem] = list(extracted.artists or [])
    if len(artists) > 3:
        artists = artists[:3]
    while len(artists) < 3:
        artists.append(ArtistItem())

    # Build verification tree for each artist in parallel under root
    for idx, artist in enumerate(artists):
        await verify_single_artist(evaluator, root, artist, idx)

    # Return evaluator summary with verification tree and scores
    return evaluator.get_summary()