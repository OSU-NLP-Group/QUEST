import asyncio
import logging
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from mind2web2.llm_client.base_client import LLMClient
from mind2web2.utils.cache_filesys import CacheFileSys
from mind2web2.evaluator import Evaluator

# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "spotify_artists"
TASK_DESCRIPTION = """
Identify five music artists who currently have between 10 million and 20 million monthly listeners on Spotify. For each artist, provide a direct link to their official Spotify profile, the current number of monthly listeners on Spotify (formatted in millions, e.g., 12 million), and a link to one playlist that includes at least one song where the artist is the main performer (not just featured on someone else's song). Also provide the title of the song by the artist that appears in the playlist. The playlist must not be created by the artist themselves or by Spotify Official.
"""

MONTHLY_LISTENERS_MARGIN = 1  # 1M margin allowed

# --------------------------------------------------------------------------- #
# Data models for extraction                                                 #
# --------------------------------------------------------------------------- #
class ArtistInfo(BaseModel):
    """Information about a single artist"""
    name: Optional[str] = None
    spotify_url: Optional[str] = None
    monthly_listeners: Optional[float] = None  # in millions
    playlist_url: Optional[str] = None
    song_title: Optional[str] = None


class ExtractedArtists(BaseModel):
    """Collection of artist information extracted from the answer"""
    artists: List[ArtistInfo] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_artists() -> str:
    return """
    Please extract information about music artists mentioned in the answer. For each artist, extract:
    1. The artist's name
    2. The URL to their Spotify profile
    3. Their monthly listeners count on Spotify (convert to millions as a number, e.g., "12 million" → 12)
    4. The URL to a playlist that contains their song
    5. The title of their song that appears in the playlist
    
    Extract this information for ALL artists mentioned in the answer, even if there are more than 5.
    
    If any field is missing for an artist, set it to null. For monthly listeners, extract just the number in millions (e.g., 15 for "15 million").
    """


def prompt_extract_monthly_listeners_from_spotify_url() -> str:
    return """
    Please extract the exact number of monthly listeners for this artist from the Spotify webpage. 
    - Express this number in millions as a simple number (e.g., 12 for 12 million listeners)
    - Only extract the number itself, without any text or units
    - If the page shows monthly listeners in a format like "15,432,789", convert it to millions (15)
    - Round to the nearest million if needed
    - If you can't find the monthly listeners count, return null
    """


class MonthlyListeners(BaseModel):
    """Monthly listeners extraction model"""
    monthly_listeners: Optional[float] = None


# --------------------------------------------------------------------------- #
# Artist verification functions                                               #
# --------------------------------------------------------------------------- #
async def verify_single_artist(
    evaluator: Evaluator,
    parent_node,
    artist_info: ArtistInfo,
    index: int
) -> None:
    """
    Verify information for a single artist. Each artist has several
    verification criteria that must all be met.
    """
    # Create artist node (sequential for step-by-step verification)
    artist_node = evaluator.add_sequential(
        id=f"artist_{index}",
        desc=f"Verification for artist #{index+1}: {artist_info.name if artist_info.name else 'Unknown'}",
        parent=parent_node
    )
    
    # Create artist criteria node (combines URL and listeners verification)
    artist_criteria_node = evaluator.add_parallel(
        id=f"artist_{index}_criteria",
        desc=f"Core criteria verification for artist #{index+1}",
        parent=artist_node,
        critical=False  # Non-critical for partial credit
    )
    
    # Check if URL exists
    spotify_url_exists = evaluator.add_custom_node(
        result=bool(artist_info.spotify_url),
        id=f"artist_{index}_spotify_url_exists",
        desc=f"Spotify URL is provided for artist #{index + 1}",
        parent=artist_criteria_node,
        critical=True
    )
    
    # Verify the URL is valid
    spotify_url_valid_node = evaluator.add_leaf(
        id=f"artist_{index}_spotify_url_valid",
        desc=f"Spotify URL is valid for {artist_info.name if artist_info.name else 'Unknown'}",
        parent=artist_criteria_node,
        critical=True
    )
    
    await evaluator.verify(
        claim=f"This URL is a valid Spotify profile for artist {artist_info.name if artist_info.name else 'Unknown'}",
        node=spotify_url_valid_node,
        sources=artist_info.spotify_url,
        additional_instruction="Check that this is a valid Spotify artist profile page"
    )
    
    # Extract actual listeners from Spotify
    actual_listeners = None
    if artist_info.spotify_url:
        actual_listeners = await evaluator.extract(
            prompt=prompt_extract_monthly_listeners_from_spotify_url(),
            template_class=MonthlyListeners,
            extraction_name=f"actual_listeners_artist_{index}",
            source=artist_info.spotify_url
        )
    else:
        actual_listeners = MonthlyListeners()

    # Verify listeners are within range
    listeners_range_node = evaluator.add_custom_node(
        result=(actual_listeners.monthly_listeners is not None and 
                10 <= actual_listeners.monthly_listeners <= 20),
        id=f"artist_{index}_listeners_range",
        desc=f"Artist #{index + 1}'s {actual_listeners.monthly_listeners} million monthly listeners are between 10-20 million",
        parent=artist_criteria_node,
        critical=True
    )
    
    # Verify reported listeners match actual
    if artist_info.monthly_listeners is not None and actual_listeners.monthly_listeners is not None:
        difference = abs(artist_info.monthly_listeners - actual_listeners.monthly_listeners)
        is_accurate = difference <= MONTHLY_LISTENERS_MARGIN
    else:
        is_accurate = False
        
    listeners_accuracy_node = evaluator.add_custom_node(
        result=is_accurate,
        id=f"artist_{index}_listeners_accuracy",
        desc=f"Reported listeners ({artist_info.monthly_listeners} million) matches actual ({actual_listeners.monthly_listeners} million) within {MONTHLY_LISTENERS_MARGIN} million margin",
        parent=artist_criteria_node,
        critical=True
    )
    
    # Playlist verification parent
    playlist_parent = evaluator.add_parallel(
        id=f"artist_{index}_playlist",
        desc=f"Playlist verification for artist #{index+1}",
        parent=artist_node,
        critical=False # false to allow partial score
    )
    
    # Check if playlist info is provided
    playlist_info_exists = evaluator.add_custom_node(
        result=bool(artist_info.playlist_url and artist_info.song_title and artist_info.playlist_url.strip() != "" and artist_info.song_title.strip() != ""),
        id=f"artist_{index}_playlist_info_exists",
        desc=f"Playlist URL and song title are provided for artist #{index + 1}",
        parent=playlist_parent,
        critical=True
    )
    
    # Verify playlist is valid
    playlist_valid_node = evaluator.add_leaf(
        id=f"artist_{index}_playlist_valid",
        desc=f"Playlist is valid and not created by artist or Spotify Official",
        parent=playlist_parent,
        critical=True
    )
    
    await evaluator.verify(
        claim=f"This page is a Spotify playlist and was not created by {artist_info.name} or Spotify Official",
        node=playlist_valid_node,
        sources=artist_info.playlist_url,
        additional_instruction="Check the playlist creator information"
    )
    
    # Verify song in playlist
    song_in_playlist_node = evaluator.add_leaf(
        id=f"artist_{index}_song_in_playlist",
        desc=f"Playlist contains '{artist_info.song_title}' by {artist_info.name}",
        parent=playlist_parent,
        critical=True
    )
    
    await evaluator.verify(
        claim=f"This playlist contains the song '{artist_info.song_title}' by {artist_info.name}",
        node=song_in_playlist_node,
        sources=artist_info.playlist_url,
        additional_instruction="Look for the specific song title in the playlist"
    )
    
    # Verify main performer
    main_performer_node = evaluator.add_leaf(
        id=f"artist_{index}_main_performer",
        desc=f"{artist_info.name} is the main performer of '{artist_info.song_title}'",
        parent=playlist_parent,
        critical=True
    )
    
    await evaluator.verify(
        claim=f"In this playlist, {artist_info.name} is the main performer of '{artist_info.song_title}', not just a featured artist. The song should be credited primarily to {artist_info.name}, not in the format 'Other Artist feat. {artist_info.name}'",
        node=main_performer_node,
        sources=artist_info.playlist_url,
        additional_instruction="Check the artist credit format for the song"
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
    """
    Evaluate an answer to the Spotify artists task.
    
    The evaluation verifies:
    1. That five artists are identified
    2. Each artist has a valid Spotify profile URL
    3. Each artist's monthly listeners are between 10-20 million
    4. The reported monthly listeners match the actual number (within margin)
    5. Each artist has a valid playlist that:
       - Contains a song where they are the main performer
       - Was not created by the artist or Spotify Official
       - Includes the correct song title
    """
    # Initialize evaluator
    evaluator = Evaluator()
    evaluator.initialize(
        task_id=TASK_ID,
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
    
    # Extract all artists from the answer
    extracted_artists = await evaluator.extract(
        prompt=prompt_extract_artists(),
        template_class=ExtractedArtists,
        extraction_name="extracted_artists"
    )
    
    # Pad artists list to ensure we have exactly 5 entries
    artists_to_check = list(extracted_artists.artists[:5])
    while len(artists_to_check) < 5:
        artists_to_check.append(ArtistInfo())  # Empty artist info
    
    # Verify each of the 5 artists
    for i, artist in enumerate(artists_to_check):
        await verify_single_artist(
            evaluator=evaluator,
            parent_node=evaluator.root,
            artist_info=artist,
            index=i
        )
    
    # Return structured result
    return evaluator.get_summary()