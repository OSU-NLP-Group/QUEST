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
TASK_ID = "festival_artists_june_2026"
TASK_DESCRIPTION = """
Identify three distinct music artists who are scheduled to perform at major United States music festivals in June 2026 and meet all of the following criteria:

1. The artist must be part of the official lineup for either the Bonnaroo Music & Arts Festival (June 11-14, 2026, Manchester, TN) or The Governors Ball (June 5-7, 2026, New York City).

2. The artist must have received at least one nomination at the 68th Annual Grammy Awards held in 2026.

3. The artist must have had at least one song appear on the Billboard Hot 100 chart at any point in 2026 (as of March 19, 2026).

4. The artist must currently have at least 10 million monthly listeners on Spotify.

5. The artist's primary music genre must be identifiable as one of the major Billboard chart categories: Pop, R&B/Hip-Hop, Rock, Country, or Dance/Electronic.

6. The artist must have at least one scheduled concert tour date in the United States during 2026.

7. The artist must have released at least one collaboration song (either featuring another artist or being featured on another artist's song) that is commercially available.

8. The artist must have performed or be scheduled to perform at a venue with a capacity of at least 5,000 people during their 2026 tour.

For each of the three artists you identify, please provide:
- Artist name
- Which festival(s) they are performing at (Bonnaroo and/or Governors Ball)
- Which specific day(s) of the festival they are scheduled to perform
- At least one Grammy 2026 nomination category (and indicate if they are a Best New Artist nominee)
- Their highest Billboard Hot 100 chart position achieved in 2026
- Their current Spotify monthly listeners count (in millions)
- Their primary genre
- At least one example of a collaboration song
- A reference URL for each major piece of information (festival lineup, Grammy nominations, Billboard chart, Spotify statistics, tour dates, collaboration)
"""

ALLOWED_GENRES = ["Pop", "R&B/Hip-Hop", "Rock", "Country", "Dance/Electronic"]


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class FestivalInfo(BaseModel):
    name: Optional[str] = None  # e.g., "Bonnaroo Music & Arts Festival" or "The Governors Ball"
    days: List[str] = Field(default_factory=list)  # e.g., ["Friday, June 12"]
    lineup_url: Optional[str] = None  # URL confirming lineup/schedule


class GrammyInfo(BaseModel):
    nominations: List[str] = Field(default_factory=list)  # e.g., ["Record of the Year"]
    is_best_new_artist: Optional[bool] = None
    source_urls: List[str] = Field(default_factory=list)  # official nominee page(s)


class ChartInfo(BaseModel):
    had_hot100_2026: Optional[bool] = None
    peak_position_2026: Optional[str] = None  # string to allow "Top 10", "No. 3", "3"
    source_urls: List[str] = Field(default_factory=list)  # Billboard links or reputable aggregators


class SpotifyInfo(BaseModel):
    monthly_listeners_millions: Optional[str] = None  # e.g., "18.2", "18M", "18.2 million"
    profile_url: Optional[str] = None  # Spotify artist URL
    source_urls: List[str] = Field(default_factory=list)  # Stats/third-party pages if any


class GenreInfo(BaseModel):
    primary_genre: Optional[str] = None  # should be one of ALLOWED_GENRES
    source_urls: List[str] = Field(default_factory=list)  # source for genre classification


class TourInfo(BaseModel):
    us_tour_date_2026: Optional[str] = None  # any one concrete US date in 2026, free text ok
    venue_name: Optional[str] = None
    venue_capacity: Optional[str] = None  # string to allow "10,000", "5000+"
    source_urls: List[str] = Field(default_factory=list)  # tour page, ticketing, venue pages
    venue_urls: List[str] = Field(default_factory=list)  # optional dedicated venue capacity pages


class CollaborationInfo(BaseModel):
    song_title: Optional[str] = None  # example of a collab track
    collaborators: List[str] = Field(default_factory=list)  # featured artists
    source_urls: List[str] = Field(default_factory=list)  # streaming page, press, etc.


class ArtistRecord(BaseModel):
    name: Optional[str] = None
    festivals: List[FestivalInfo] = Field(default_factory=list)
    grammy: Optional[GrammyInfo] = None
    chart: Optional[ChartInfo] = None
    spotify: Optional[SpotifyInfo] = None
    genre: Optional[GenreInfo] = None
    tour: Optional[TourInfo] = None
    collaboration: Optional[CollaborationInfo] = None


class ArtistsExtraction(BaseModel):
    artists: List[ArtistRecord] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_artists() -> str:
    return f"""
Extract up to three distinct artists from the answer and structure the following fields for each artist. Only extract information that is explicitly present in the answer text. For all URL fields, extract the concrete URLs exactly as shown (plain links or markdown links are both acceptable). If a piece of information is missing, return null or an empty list accordingly.

For each artist, extract:
- name: The artist name exactly as written in the answer.

- festivals: An array of festival objects, each with:
  - name: Festival name as presented in the answer (e.g., "Bonnaroo Music & Arts Festival" or "The Governors Ball").
  - days: Array of specific day(s) the artist is scheduled to perform (e.g., ["Friday, June 12"]).
  - lineup_url: URL that confirms the lineup/schedule for this artist/festival.

- grammy: Object with:
  - nominations: Array of at least one 2026 (68th Annual) Grammy nomination category mentioned (e.g., ["Record of the Year"]).
  - is_best_new_artist: true/false if the answer indicates Best New Artist nominee status; otherwise null if not mentioned.
  - source_urls: Array of URL(s) that confirm the Grammy nomination(s).

- chart: Object with:
  - had_hot100_2026: true if the answer indicates at least one song appeared on the Billboard Hot 100 in 2026 (as of March 19, 2026); false if explicitly contradicted; null if not stated.
  - peak_position_2026: The highest Hot 100 position achieved in 2026 as mentioned (string; allow formats like "3", "No. 3", "Top 10").
  - source_urls: URL(s) that substantiate the Billboard chart appearance/peak.

- spotify: Object with:
  - monthly_listeners_millions: A numeric-like string exactly as written indicating current Spotify monthly listeners in millions (e.g., "18.2", "18M", "18.2 million").
  - profile_url: The Spotify artist profile URL if present.
  - source_urls: Any additional URL(s) used to substantiate Spotify statistics (can be same as profile_url or third-party trackers).

- genre: Object with:
  - primary_genre: The primary music genre identified in the answer text; must be one of: {ALLOWED_GENRES}.
  - source_urls: URL(s) that support the genre classification.

- tour: Object with:
  - us_tour_date_2026: A specific 2026 U.S. tour date (free text is fine: city, venue, and/or date) if present.
  - venue_name: Venue name associated with a 2026 date if mentioned.
  - venue_capacity: Venue capacity if mentioned (string is fine).
  - source_urls: URL(s) confirming the 2026 U.S. tour date(s).
  - venue_urls: URL(s) that substantiate the venue’s stated capacity (if provided).

- collaboration: Object with:
  - song_title: An example collaboration song title (either featuring another artist or where this artist is featured).
  - collaborators: Array of collaborator artist names mentioned.
  - source_urls: URL(s) that confirm the collaboration and that it is commercially available.

General rules:
- Do not fabricate URLs or facts not shown in the answer.
- If a URL is missing a protocol, you may prepend http:// as needed.
- If more than three artists are listed, extract the first three in order of appearance.
"""


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _unique_urls(urls: List[Optional[str]] | None) -> List[str]:
    if not urls:
        return []
    out: List[str] = []
    seen = set()
    for u in urls:
        if not u:
            continue
        u = u.strip()
        if not u:
            continue
        if u not in seen:
            out.append(u)
            seen.add(u)
    return out


def _merge_url_lists(*lists: List[str]) -> List[str]:
    merged: List[str] = []
    for lst in lists:
        for u in lst:
            if u and u not in merged:
                merged.append(u)
    return merged


# --------------------------------------------------------------------------- #
# Verification logic per artist                                               #
# --------------------------------------------------------------------------- #
async def verify_single_artist(evaluator: Evaluator, parent_node, artist: ArtistRecord, artist_idx: int) -> None:
    """
    Build the verification subtree for a single artist according to the rubric.
    artist_idx is 1-based (1, 2, 3) to match the rubric IDs.
    """
    name = artist.name or f"Artist #{artist_idx}"

    # Artist node (parallel aggregation; non-critical as in rubric)
    artist_node = evaluator.add_parallel(
        id=f"artist_{artist_idx}",
        desc=f"{['First','Second','Third'][artist_idx-1]} qualifying artist with complete verification of all requirements",
        parent=parent_node,
        critical=False
    )

    # ---------------- Festival participation ----------------
    fest_node = evaluator.add_parallel(
        id=f"artist_{artist_idx}_festival_participation",
        desc=f"Festival participation verification for {['first','second','third'][artist_idx-1]} artist",
        parent=artist_node,
        critical=False
    )

    # Collect festival URLs and days
    fest_urls: List[str] = []
    has_any_days = False
    fest_names: List[str] = []
    if artist.festivals:
        for f in artist.festivals:
            fest_urls.extend(_unique_urls([f.lineup_url]))
            fest_names.append(f.name or "")
            if f.days and any(d.strip() for d in f.days):
                has_any_days = True
    fest_urls = _unique_urls(fest_urls)

    # Critical: Source URL provided for lineup confirmation
    evaluator.add_custom_node(
        result=len(fest_urls) > 0,
        id=f"artist_{artist_idx}_festival_source_url",
        desc="Reference URL provided for festival lineup confirmation",
        parent=fest_node,
        critical=True
    )

    # Critical: Specific day(s) specified
    evaluator.add_custom_node(
        result=has_any_days,
        id=f"artist_{artist_idx}_festival_day_specification",
        desc="Specific day(s) of the festival when the artist is scheduled to perform is provided",
        parent=fest_node,
        critical=True
    )

    # Critical: Artist confirmed in official lineup of Bonnaroo (June 11-14, 2026) or Governors Ball (June 5-7, 2026)
    fest_lineup_leaf = evaluator.add_leaf(
        id=f"artist_{artist_idx}_festival_lineup_membership",
        desc="Artist is confirmed in the official lineup of either Bonnaroo Music & Arts Festival (June 11-14, 2026) or The Governors Ball (June 5-7, 2026)",
        parent=fest_node,
        critical=True
    )
    lineup_claim = (
        f"The artist {name} appears in the official 2026 lineup for either the Bonnaroo Music & Arts Festival "
        f"(June 11–14, 2026, Manchester, TN) or The Governors Ball (June 5–7, 2026, New York City)."
    )
    await evaluator.verify(
        claim=lineup_claim,
        node=fest_lineup_leaf,
        sources=fest_urls,
        additional_instruction="Verify that the source page is an official or authoritative lineup/schedule page for 2026 and that the artist is listed. Minor name variants are acceptable."
    )

    # ---------------- Awards (Grammy nominations) ----------------
    awards_node = evaluator.add_parallel(
        id=f"artist_{artist_idx}_awards_recognition",
        desc=f"Awards and nominations verification for {['first','second','third'][artist_idx-1]} artist",
        parent=artist_node,
        critical=False
    )
    grammy_sources = _unique_urls(artist.grammy.source_urls if artist.grammy else [])

    # Critical: Grammy source URL present
    evaluator.add_custom_node(
        result=len(grammy_sources) > 0,
        id=f"artist_{artist_idx}_grammy_source_url",
        desc="Reference URL provided for Grammy nomination verification",
        parent=awards_node,
        critical=True
    )

    # Critical: At least one nomination at the 68th Annual Grammy Awards (2026)
    grammy_nom_leaf = evaluator.add_leaf(
        id=f"artist_{artist_idx}_grammy_nomination",
        desc="Artist received at least one nomination at the 68th Annual Grammy Awards (2026)",
        parent=awards_node,
        critical=True
    )
    grammy_claim = f"{name} received at least one nomination at the 68th Annual Grammy Awards in 2026."
    await evaluator.verify(
        claim=grammy_claim,
        node=grammy_nom_leaf,
        sources=grammy_sources,
        additional_instruction="Look for official nominee lists or reputable coverage confirming at least one 2026 Grammy nomination for the artist."
    )

    # Non-critical: Best New Artist status (verify only if claimed True; otherwise pass)
    if artist.grammy and artist.grammy.is_best_new_artist:
        bna_leaf = evaluator.add_leaf(
            id=f"artist_{artist_idx}_best_new_artist_status",
            desc="If applicable, indication of whether artist is a Best New Artist nominee",
            parent=awards_node,
            critical=False
        )
        bna_claim = f"{name} was nominated for Best New Artist at the 68th Annual Grammy Awards (2026)."
        await evaluator.verify(
            claim=bna_claim,
            node=bna_leaf,
            sources=grammy_sources,
            additional_instruction="Confirm that the artist is among the Best New Artist nominees for the 68th Grammys (2026)."
        )
    else:
        evaluator.add_custom_node(
            result=True,
            id=f"artist_{artist_idx}_best_new_artist_status",
            desc="If applicable, indication of whether artist is a Best New Artist nominee (not claimed, so not required)",
            parent=awards_node,
            critical=False
        )

    # ---------------- Chart performance (Billboard Hot 100 in 2026) ----------------
    chart_node = evaluator.add_parallel(
        id=f"artist_{artist_idx}_chart_performance",
        desc=f"Billboard Hot 100 chart presence verification for {['first','second','third'][artist_idx-1]} artist",
        parent=artist_node,
        critical=False
    )
    chart_sources = _unique_urls(artist.chart.source_urls if artist.chart else [])

    # Critical: Chart source URL present
    evaluator.add_custom_node(
        result=len(chart_sources) > 0,
        id=f"artist_{artist_idx}_chart_source_url",
        desc="Reference URL provided for Billboard chart verification",
        parent=chart_node,
        critical=True
    )

    # Critical: Presence on Hot 100 in 2026 (as of March 19, 2026)
    presence_leaf = evaluator.add_leaf(
        id=f"artist_{artist_idx}_billboard_presence",
        desc="Artist had at least one song appear on the Billboard Hot 100 chart in 2026",
        parent=chart_node,
        critical=True
    )
    presence_claim = f"At least one song by {name} appeared on the Billboard Hot 100 chart at some point in 2026 (as of March 19, 2026)."
    await evaluator.verify(
        claim=presence_claim,
        node=presence_leaf,
        sources=chart_sources,
        additional_instruction="Confirm any 2026 Hot 100 week that includes a song by this artist."
    )

    # Critical: Highest position provided and supported
    peak_leaf = evaluator.add_leaf(
        id=f"artist_{artist_idx}_peak_position",
        desc="Highest chart position achieved on Billboard Hot 100 in 2026 is provided",
        parent=chart_node,
        critical=True
    )
    peak_value = artist.chart.peak_position_2026 if artist.chart else None
    peak_claim = f"{name}'s highest Billboard Hot 100 chart position achieved in 2026 was {peak_value}."
    await evaluator.verify(
        claim=peak_claim,
        node=peak_leaf,
        sources=chart_sources,
        additional_instruction="Verify the claimed highest 2026 Hot 100 position. Reasonable equivalences like 'No. 3' vs '3' are acceptable."
    )

    # ---------------- Spotify streaming metrics ----------------
    stream_node = evaluator.add_parallel(
        id=f"artist_{artist_idx}_streaming_metrics",
        desc=f"Spotify streaming statistics verification for {['first','second','third'][artist_idx-1]} artist",
        parent=artist_node,
        critical=False
    )

    # Collect Spotify sources (profile + extras)
    spotify_urls: List[str] = []
    if artist.spotify:
        spotify_urls = _merge_url_lists(
            _unique_urls([artist.spotify.profile_url]),
            _unique_urls(artist.spotify.source_urls)
        )

    # Critical: streaming source URL present
    evaluator.add_custom_node(
        result=len(spotify_urls) > 0,
        id=f"artist_{artist_idx}_streaming_source_url",
        desc="Reference URL provided for Spotify statistics verification",
        parent=stream_node,
        critical=True
    )

    # Critical: Spotify availability
    availability_leaf = evaluator.add_leaf(
        id=f"artist_{artist_idx}_spotify_availability",
        desc="Artist's music is confirmed available on Spotify",
        parent=stream_node,
        critical=True
    )
    availability_claim = f"{name} has an active Spotify artist profile with available music."
    await evaluator.verify(
        claim=availability_claim,
        node=availability_leaf,
        sources=spotify_urls,
        additional_instruction="The page should show a legitimate Spotify artist profile or reputable proof of availability."
    )

    # Critical: >= 10M monthly listeners (current)
    listeners_leaf = evaluator.add_leaf(
        id=f"artist_{artist_idx}_monthly_listeners_threshold",
        desc="Artist has at least 10 million monthly listeners on Spotify",
        parent=stream_node,
        critical=True
    )
    listeners_value = artist.spotify.monthly_listeners_millions if artist.spotify else None
    listeners_claim = f"{name} currently has at least 10 million monthly listeners on Spotify (reported value: {listeners_value})."
    await evaluator.verify(
        claim=listeners_claim,
        node=listeners_leaf,
        sources=spotify_urls,
        additional_instruction="Allow small rounding differences; acceptable if value >= 10.0 million. If given as '18M' or '18.0 million', treat as 18.0."
    )

    # ---------------- Genre classification ----------------
    genre_node = evaluator.add_parallel(
        id=f"artist_{artist_idx}_genre_classification",
        desc=f"Genre identification for {['first','second','third'][artist_idx-1]} artist",
        parent=artist_node,
        critical=False
    )

    genre_sources = _unique_urls(artist.genre.source_urls if artist.genre else [])
    evaluator.add_custom_node(
        result=len(genre_sources) > 0,
        id=f"artist_{artist_idx}_genre_source_url",
        desc="Reference URL provided for genre classification verification",
        parent=genre_node,
        critical=True
    )

    genre_leaf = evaluator.add_leaf(
        id=f"artist_{artist_idx}_primary_genre",
        desc="Artist's primary genre is identified as one of the major Billboard chart categories (Pop, R&B/Hip-Hop, Rock, Country, or Dance/Electronic)",
        parent=genre_node,
        critical=True
    )
    genre_value = (artist.genre.primary_genre if artist.genre else None) or "Unknown"
    genre_claim = (
        f"{name}'s primary genre can be identified as '{genre_value}', which is one of the following: "
        f"{', '.join(ALLOWED_GENRES)}."
    )
    await evaluator.verify(
        claim=genre_claim,
        node=genre_leaf,
        sources=genre_sources,
        additional_instruction="Confirm that the classification aligns with reputable sources and that the stated genre is among the allowed Billboard-style categories."
    )

    # ---------------- Tour information ----------------
    tour_node = evaluator.add_parallel(
        id=f"artist_{artist_idx}_tour_information",
        desc=f"Tour activity and venue verification for {['first','second','third'][artist_idx-1]} artist",
        parent=artist_node,
        critical=False
    )

    tour_urls = _unique_urls(artist.tour.source_urls if artist.tour else [])
    venue_urls = _unique_urls(artist.tour.venue_urls if artist.tour else [])
    all_tour_related = _merge_url_lists(tour_urls, venue_urls)

    evaluator.add_custom_node(
        result=len(tour_urls) > 0,  # Require at least a tour schedule source
        id=f"artist_{artist_idx}_tour_source_url",
        desc="Reference URL provided for tour schedule verification",
        parent=tour_node,
        critical=True
    )

    # Critical: Has at least one scheduled US tour date in 2026
    us_tour_leaf = evaluator.add_leaf(
        id=f"artist_{artist_idx}_us_tour_date",
        desc="Artist has at least one scheduled concert tour date in the United States during 2026",
        parent=tour_node,
        critical=True
    )
    us_tour_value = (artist.tour.us_tour_date_2026 if artist.tour else None) or "a 2026 U.S. date"
    us_tour_claim = f"{name} has at least one scheduled U.S. tour date in 2026 (example mentioned in the answer: {us_tour_value})."
    await evaluator.verify(
        claim=us_tour_claim,
        node=us_tour_leaf,
        sources=tour_urls,
        additional_instruction="Confirm a 2026 U.S. date (city and/or venue) from the tour source(s). Ticketing pages and official site listings are acceptable."
    )

    # Critical: Performed/scheduled at a venue with capacity >= 5,000 in 2026
    capacity_leaf = evaluator.add_leaf(
        id=f"artist_{artist_idx}_venue_capacity",
        desc="Artist has performed or is scheduled to perform at a venue with capacity of at least 5,000 people during 2026",
        parent=tour_node,
        critical=True
    )
    venue_name = (artist.tour.venue_name if artist.tour else None) or "a 5,000+ capacity venue"
    capacity_value = (artist.tour.venue_capacity if artist.tour else None) or ">=5,000"
    capacity_claim = (
        f"During 2026, {name} performed or is scheduled to perform at {venue_name}, which has a capacity of at least 5,000 (claimed capacity: {capacity_value})."
    )
    await evaluator.verify(
        claim=capacity_claim,
        node=capacity_leaf,
        sources=all_tour_related,
        additional_instruction="Use tour/venue sources to confirm capacity >= 5,000. If an exact figure is shown and >= 5000, accept."
    )

    # ---------------- Collaboration ----------------
    collab_node = evaluator.add_parallel(
        id=f"artist_{artist_idx}_collaboration",
        desc=f"Collaboration work verification for {['first','second','third'][artist_idx-1]} artist",
        parent=artist_node,
        critical=False
    )

    collab_sources = _unique_urls(artist.collaboration.source_urls if artist.collaboration else [])

    evaluator.add_custom_node(
        result=len(collab_sources) > 0,
        id=f"artist_{artist_idx}_collaboration_source_url",
        desc="Reference URL provided for collaboration verification",
        parent=collab_node,
        critical=True
    )

    collab_leaf = evaluator.add_leaf(
        id=f"artist_{artist_idx}_collaboration_existence",
        desc="Artist has released at least one collaboration song (featuring another artist or being featured on another artist's song)",
        parent=collab_node,
        critical=True
    )
    collab_title = (artist.collaboration.song_title if artist.collaboration else None) or "a collaboration track"
    collab_with = ", ".join(artist.collaboration.collaborators) if (artist.collaboration and artist.collaboration.collaborators) else "another artist"
    collab_claim = f"{name} has released at least one commercially available collaboration song, for example '{collab_title}' with {collab_with}."
    await evaluator.verify(
        claim=collab_claim,
        node=collab_leaf,
        sources=collab_sources,
        additional_instruction="Streaming pages, official announcements, or reputable coverage that show a released collab are acceptable."
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
    model: str = "o4-mini"
) -> Dict:
    """
    Evaluate an answer for the 'festival_artists_june_2026' task using the Mind2Web2 framework.
    """
    # Initialize evaluator (root kept non-critical to comply with tree constraints)
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

    # Extract structured artist info
    extracted = await evaluator.extract(
        prompt=prompt_extract_artists(),
        template_class=ArtistsExtraction,
        extraction_name="artists_extraction"
    )

    # Normalize to exactly three artists (pad with empties if fewer)
    artists: List[ArtistRecord] = list(extracted.artists[:3])
    while len(artists) < 3:
        artists.append(ArtistRecord())

    # Build verification subtrees for three artists in parallel style
    # We perform verifications sequentially within each artist for clarity,
    # but across artists, they're independent siblings under the root (parallel aggregation at root).
    for i in range(3):
        await verify_single_artist(evaluator, root, artists[i], i + 1)

    # Return summarized evaluation result
    return evaluator.get_summary()