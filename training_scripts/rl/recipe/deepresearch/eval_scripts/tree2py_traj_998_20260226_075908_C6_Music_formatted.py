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
TASK_ID = "country_artists_2024_milestones"
TASK_DESCRIPTION = """
Identify two different country music artists who achieved the following distinct milestones in 2024:

Artist A: This artist achieved their 10th career #1 hit on the Billboard Country Airplay chart in 2024. This #1 song was a collaboration featuring Jelly Roll and came from an album that was released on September 29, 2023, which contains exactly 12 tracks.

Artist B: This artist won the ACM Triple Crown Award in 2024. To qualify for this award, the artist won ACM New Female Artist of the Year in 2022, ACM Female Artist of the Year in both 2023 and 2024, and ACM Entertainer of the Year in 2024 (for the first time). This artist also released a new album on August 23, 2024, which contains exactly 14 tracks and includes a collaboration with Miranda Lambert.

For each artist, provide:
1. The artist's name
2. For Artist A: the title of the #1 song and the album title
3. For Artist B: the title of the 2024 album
4. Reference URLs that verify each piece of information
"""

# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class ArtistAInfo(BaseModel):
    """Extracted data for Artist A"""
    name: Optional[str] = None
    song_title: Optional[str] = None
    album_title: Optional[str] = None
    # Optional fields from the answer (not strictly required for verification, but may be present)
    album_release_date: Optional[str] = None
    album_track_count: Optional[str] = None
    # Source URLs grouped by topic
    hit_sources: List[str] = Field(default_factory=list)       # URLs verifying #1 chart achievement, 10th milestone, and Jelly Roll collaboration
    album_sources: List[str] = Field(default_factory=list)     # URLs verifying album release date and track count


class ArtistBInfo(BaseModel):
    """Extracted data for Artist B"""
    name: Optional[str] = None
    album_title: Optional[str] = None
    # Optional fields from the answer (not strictly required for verification, but may be present)
    album_release_date: Optional[str] = None
    album_track_count: Optional[str] = None
    # Source URLs grouped by topic
    triple_crown_sources: List[str] = Field(default_factory=list)  # URLs verifying awards history and 2024 Triple Crown qualification
    album_sources: List[str] = Field(default_factory=list)         # URLs verifying 2024 album details


class BothArtistsExtraction(BaseModel):
    """Top-level extraction capturing both artists."""
    artist_a: Optional[ArtistAInfo] = None
    artist_b: Optional[ArtistBInfo] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_artists() -> str:
    return """
    Extract structured information for two distinct country music artists described in the answer.

    Artist A (2024 10th #1 milestone):
    - name: The artist's full name.
    - song_title: Title of the #1 song (the one that hit #1 in 2024).
    - album_title: The album title from which the #1 song came.
    - hit_sources: A list of URL(s) that verify the #1 achievement in 2024, that it was the artist's 10th career #1, and that the song features Jelly Roll (e.g., official charts, Billboard Country Airplay articles, credible news sources).
    - album_sources: A list of URL(s) that verify the album was released on September 29, 2023 and that the album has exactly 12 tracks (e.g., official artist site, label site, streaming platforms showing release date and track count).
    - album_release_date: If the answer provides a date for the album release, extract it verbatim as a string; otherwise null.
    - album_track_count: If the answer provides a track count, extract it verbatim as a string; otherwise null.

    Artist B (2024 ACM Triple Crown + album):
    - name: The artist's full name.
    - album_title: Title of the 2024 album.
    - triple_crown_sources: A list of URL(s) that verify: ACM New Female Artist of the Year (2022), ACM Female Artist of the Year (2023 and 2024), ACM Entertainer of the Year (2024, first time). Prefer official ACM site or credible news outlets.
    - album_sources: A list of URL(s) that verify: album released on August 23, 2024, album contains exactly 14 tracks, and album includes a collaboration with Miranda Lambert (e.g., official artist/label sites or streaming platforms showing track list and credits).
    - album_release_date: If the answer provides a date for the album release, extract it verbatim as a string; otherwise null.
    - album_track_count: If the answer provides a track count, extract it verbatim as a string; otherwise null.

    IMPORTANT:
    - Strictly extract only what the answer states. Do not invent any data.
    - For each URL list, include only valid, complete URLs explicitly present in the answer. If none are provided for a category, return an empty list.
    - If any field is missing in the answer, set it to null (or an empty list for URL fields).
    """


# --------------------------------------------------------------------------- #
# Verification helper functions                                               #
# --------------------------------------------------------------------------- #
async def verify_artist_a(evaluator: Evaluator, parent_node, info: ArtistAInfo) -> None:
    """
    Build and verify the Artist A subtree:
    - Identity provided
    - Hit details (with references)
    - Album details (with references)
    """
    # Parent: Artist A (sequential)
    artist_a_node = evaluator.add_sequential(
        id="artist_a",
        desc="Artist who achieved 10th career #1 hit on Billboard Country Airplay in 2024 from a 2023 album collaboration",
        parent=parent_node,
        critical=False,  # Allow partial scoring independently of Artist B
    )

    # 1) Identity (critical existence)
    evaluator.add_custom_node(
        result=bool(info and info.name and info.name.strip()),
        id="artist_a_identity",
        desc="Artist A correctly identified (name provided)",
        parent=artist_a_node,
        critical=True
    )

    # 2) Hit details (parallel, all children critical)
    hit_node = evaluator.add_parallel(
        id="artist_a_hit_details",
        desc="Details about the #1 hit achievement",
        parent=artist_a_node,
        critical=True
    )

    # Existence prerequisites under hit details
    evaluator.add_custom_node(
        result=bool(info and info.song_title and info.song_title.strip()),
        id="artist_a_song_title_provided",
        desc="The title of the #1 song is provided",
        parent=hit_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=bool(info and info.hit_sources and len(info.hit_sources) > 0),
        id="artist_a_hit_details_references",
        desc="URL references provided for #1 hit details",
        parent=hit_node,
        critical=True
    )

    # Chart achievement in 2024
    chart_leaf = evaluator.add_leaf(
        id="artist_a_chart_achievement",
        desc="Song reached #1 on Billboard Country Airplay chart in 2024",
        parent=hit_node,
        critical=True
    )
    chart_claim = f"The song '{(info.song_title or '').strip()}' by {(info.name or '').strip()} reached #1 on the Billboard Country Airplay chart in 2024."
    await evaluator.verify(
        claim=chart_claim,
        node=chart_leaf,
        sources=info.hit_sources,
        additional_instruction="Verify that the song achieved #1 specifically on the Billboard Country Airplay chart during calendar year 2024."
    )

    # Career milestone: 10th career #1
    milestone_leaf = evaluator.add_leaf(
        id="artist_a_career_milestone",
        desc="This was the artist's 10th career #1 hit",
        parent=hit_node,
        critical=True
    )
    milestone_claim = f"This #1 for {(info.name or '').strip()} was their 10th career #1 hit on the Billboard Country Airplay chart."
    await evaluator.verify(
        claim=milestone_claim,
        node=milestone_leaf,
        sources=info.hit_sources,
        additional_instruction="Confirm that the referenced sources explicitly state this #1 is the artist's 10th career #1 on Country Airplay."
    )

    # Collaboration featuring Jelly Roll
    collab_leaf = evaluator.add_leaf(
        id="artist_a_collaboration_feature",
        desc="Song features a collaboration with Jelly Roll",
        parent=hit_node,
        critical=True
    )
    collab_claim = f"The song '{(info.song_title or '').strip()}' is a collaboration featuring Jelly Roll (e.g., credited as 'feat. Jelly Roll' or 'Jelly Roll & {(info.name or '').strip()}')."
    await evaluator.verify(
        claim=collab_claim,
        node=collab_leaf,
        sources=info.hit_sources,
        additional_instruction="Check credits and descriptions in the provided sources to confirm Jelly Roll is featured on the #1 song."
    )

    # 3) Album details (parallel, all children critical)
    album_node = evaluator.add_parallel(
        id="artist_a_album_details",
        desc="Details about the 2023 album containing the #1 hit",
        parent=artist_a_node,
        critical=True
    )

    # Existence prerequisites under album details
    evaluator.add_custom_node(
        result=bool(info and info.album_title and info.album_title.strip()),
        id="artist_a_album_title_provided",
        desc="Album title is provided",
        parent=album_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=bool(info and info.album_sources and len(info.album_sources) > 0),
        id="artist_a_album_details_references",
        desc="URL references provided for album details",
        parent=album_node,
        critical=True
    )

    # Album release date: September 29, 2023
    album_date_leaf = evaluator.add_leaf(
        id="artist_a_album_release_date",
        desc="Album was released on September 29, 2023",
        parent=album_node,
        critical=True
    )
    album_date_claim = f"The album '{(info.album_title or '').strip()}' was released on September 29, 2023."
    await evaluator.verify(
        claim=album_date_claim,
        node=album_date_leaf,
        sources=info.album_sources,
        additional_instruction="Verify the standard release date for the album (not deluxe editions or reissues) is September 29, 2023."
    )

    # Album track count: exactly 12 tracks
    album_tracks_leaf = evaluator.add_leaf(
        id="artist_a_album_track_count",
        desc="Album contains exactly 12 tracks",
        parent=album_node,
        critical=True
    )
    album_tracks_claim = f"The album '{(info.album_title or '').strip()}' contains exactly 12 tracks."
    await evaluator.verify(
        claim=album_tracks_claim,
        node=album_tracks_leaf,
        sources=info.album_sources,
        additional_instruction="Use official tracklists or credible platforms to confirm the standard album edition includes exactly 12 tracks."
    )


async def verify_artist_b(evaluator: Evaluator, parent_node, info: ArtistBInfo) -> None:
    """
    Build and verify the Artist B subtree:
    - Identity provided
    - Triple Crown awards verification (with references)
    - 2024 album details verification (with references)
    """
    # Parent: Artist B (sequential)
    artist_b_node = evaluator.add_sequential(
        id="artist_b",
        desc="Artist who won ACM Triple Crown Award in 2024",
        parent=parent_node,
        critical=False
    )

    # 1) Identity (critical existence)
    evaluator.add_custom_node(
        result=bool(info and info.name and info.name.strip()),
        id="artist_b_identity",
        desc="Artist B correctly identified (name provided)",
        parent=artist_b_node,
        critical=True
    )

    # 2) Triple Crown verification (parallel, all children critical)
    triple_node = evaluator.add_parallel(
        id="artist_b_triple_crown",
        desc="ACM Triple Crown qualification verified",
        parent=artist_b_node,
        critical=True
    )

    # Existence of references
    evaluator.add_custom_node(
        result=bool(info and info.triple_crown_sources and len(info.triple_crown_sources) > 0),
        id="artist_b_triple_crown_references",
        desc="URL references provided for ACM Triple Crown awards",
        parent=triple_node,
        critical=True
    )

    # New Female Artist of the Year (2022)
    new_artist_leaf = evaluator.add_leaf(
        id="artist_b_new_artist_award",
        desc="Won ACM New Female Artist of the Year in 2022",
        parent=triple_node,
        critical=True
    )
    new_artist_claim = f"{(info.name or '').strip()} won the ACM New Female Artist of the Year in 2022."
    await evaluator.verify(
        claim=new_artist_claim,
        node=new_artist_leaf,
        sources=info.triple_crown_sources,
        additional_instruction="Confirm the ACM New Female Artist of the Year award for 2022 from ACM or credible sources."
    )

    # Female Artist of the Year (2023 and 2024)
    female_awards_leaf = evaluator.add_leaf(
        id="artist_b_female_artist_awards",
        desc="Won ACM Female Artist of the Year in both 2023 and 2024",
        parent=triple_node,
        critical=True
    )
    female_awards_claim = f"{(info.name or '').strip()} won ACM Female Artist of the Year in both 2023 and 2024."
    await evaluator.verify(
        claim=female_awards_claim,
        node=female_awards_leaf,
        sources=info.triple_crown_sources,
        additional_instruction="Confirm both years (2023 AND 2024) for ACM Female Artist of the Year."
    )

    # Entertainer of the Year (2024, first time)
    eoty_leaf = evaluator.add_leaf(
        id="artist_b_entertainer_award_2024",
        desc="Won ACM Entertainer of the Year in 2024 for the first time",
        parent=triple_node,
        critical=True
    )
    eoty_claim = f"{(info.name or '').strip()} won ACM Entertainer of the Year in 2024 for the first time."
    await evaluator.verify(
        claim=eoty_claim,
        node=eoty_leaf,
        sources=info.triple_crown_sources,
        additional_instruction="Verify 2024 Entertainer of the Year and that it was the artist's first time receiving Entertainer of the Year."
    )

    # 3) 2024 album details (parallel, all children critical)
    album2024_node = evaluator.add_parallel(
        id="artist_b_2024_album",
        desc="Details about the 2024 album",
        parent=artist_b_node,
        critical=True
    )

    # Existence prerequisites
    evaluator.add_custom_node(
        result=bool(info and info.album_title and info.album_title.strip()),
        id="artist_b_2024_album_title_provided",
        desc="Album title is provided",
        parent=album2024_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=bool(info and info.album_sources and len(info.album_sources) > 0),
        id="artist_b_2024_album_references",
        desc="URL references provided for 2024 album details",
        parent=album2024_node,
        critical=True
    )

    # Release date: August 23, 2024
    rel_date_leaf = evaluator.add_leaf(
        id="artist_b_2024_album_release_date",
        desc="Album was released on August 23, 2024",
        parent=album2024_node,
        critical=True
    )
    rel_date_claim = f"The album '{(info.album_title or '').strip()}' by {(info.name or '').strip()} was released on August 23, 2024."
    await evaluator.verify(
        claim=rel_date_claim,
        node=rel_date_leaf,
        sources=info.album_sources,
        additional_instruction="Confirm the standard release date for the album is August 23, 2024."
    )

    # Track count: exactly 14 tracks
    track_count_leaf = evaluator.add_leaf(
        id="artist_b_2024_album_track_count",
        desc="Album contains exactly 14 tracks",
        parent=album2024_node,
        critical=True
    )
    track_count_claim = f"The album '{(info.album_title or '').strip()}' contains exactly 14 tracks."
    await evaluator.verify(
        claim=track_count_claim,
        node=track_count_leaf,
        sources=info.album_sources,
        additional_instruction="Use official tracklists or credible platforms to confirm the standard edition includes exactly 14 tracks."
    )

    # Collaboration with Miranda Lambert
    collab_ml_leaf = evaluator.add_leaf(
        id="artist_b_2024_album_collaboration",
        desc="Album includes a collaboration with Miranda Lambert",
        parent=album2024_node,
        critical=True
    )
    collab_ml_claim = f"The album '{(info.album_title or '').strip()}' includes a collaboration with Miranda Lambert."
    await evaluator.verify(
        claim=collab_ml_claim,
        node=collab_ml_leaf,
        sources=info.album_sources,
        additional_instruction="Confirm at least one track on the album credits Miranda Lambert as a featured or collaborating artist."
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
    Evaluate an answer for the 2024 country music milestones task.
    """
    # Initialize evaluator
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Root parallel to allow independent artist evaluation
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

    # Extract both artists' data from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_artists(),
        template_class=BothArtistsExtraction,
        extraction_name="artists_extraction",
    )

    # Build subtrees and verify
    await verify_artist_a(evaluator, root, extracted.artist_a or ArtistAInfo())
    await verify_artist_b(evaluator, root, extracted.artist_b or ArtistBInfo())

    # Return the evaluation summary
    return evaluator.get_summary()