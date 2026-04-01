import asyncio
import logging
from typing import Optional, List, Dict

import openai
from pydantic import BaseModel, Field

from mind2web2.llm_client.base_client import LLMClient
from mind2web2.utils.cache_filesys import CacheFileSys
from mind2web2.evaluator import Evaluator

# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "cyberpunk_music"
TASK_DESCRIPTION = """
In Cyberpunk 2077, there are multiple in-game radio stations featuring a diverse soundtrack. Please find five songs from these radio stations that meet the following requirements:
- Each song must be by a different real-life artist or band.
- The artist must appear in the game using their real-world name or stage name (not a fictional alias or a name created specifically for the game).
- Each song must be available on Spotify.

For each song, provide the title, the name(s) of all artist(s), a link to the song on Spotify, link(s) to each artist's Spotify page, and the name of the in-game radio station where the song is played.
"""


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class SongTitles(BaseModel):
    titles: Optional[List[str]] = Field(default_factory=list)


class SongDetails(BaseModel):
    artists: Optional[List[str]] = Field(default_factory=list)
    spotify_song_url: Optional[str] = None
    spotify_artist_urls: Optional[List[str]] = Field(default_factory=list)
    radio_station: Optional[str] = None


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_song_titles() -> str:
    return """
    Extract the titles of all songs mentioned in the answer. Return them as a list of strings in the order they appear in the answer.
    Only extract song titles, not artist names or other information.
    """


def prompt_extract_song_details(title: str) -> str:
    return f"""
    For the song titled "{title}", extract the following information:
    - artists: List of all artist names for this specific song
    - spotify_song_url: The Spotify URL for this song
    - spotify_artist_urls: List of Spotify URLs for each artist of this song
    - radio_station: The name of the in-game radio station where this song plays

    Extract exactly as mentioned in the answer text for this specific song.
    """


# --------------------------------------------------------------------------- #
# Verification functions                                                      #
# --------------------------------------------------------------------------- #
async def verify_individual_song(
        evaluator: Evaluator,
        parent_node,
        song_details: SongDetails,
        song_index: int,
        title: str,
) -> None:
    """
    Verify all requirements for an individual song.
    """
    song_node = evaluator.add_parallel(
        id=f"song_{song_index}",
        desc=f"Song {song_index} '{title}' meets all requirements",
        parent=parent_node,
        critical=False,
    )

    # 1. Check completeness (critical existence check)
    has_artists = (song_details.artists is not None and
                   len(song_details.artists) > 0 and
                   all(artist.strip() != "" for artist in song_details.artists))
    has_song_url = (song_details.spotify_song_url is not None and
                    song_details.spotify_song_url.strip() != "")
    has_artist_urls = (song_details.spotify_artist_urls is not None and
                       len(song_details.spotify_artist_urls) > 0 and
                       all(url.strip() != "" for url in song_details.spotify_artist_urls))
    has_radio_station = (song_details.radio_station is not None and
                         song_details.radio_station.strip() != "")

    is_complete = has_artists and has_song_url and has_artist_urls and has_radio_station

    evaluator.add_custom_node(
        result=is_complete,
        id=f"song_{song_index}_completeness",
        desc=f"Song {song_index} '{title}' has all required information",
        parent=song_node,
        critical=True
    )

    # 2. Verify Spotify song match
    spotify_match_node = evaluator.add_leaf(
        id=f"song_{song_index}_spotify_match",
        desc=f"Song '{title}' exists on Spotify with matching artists",
        parent=song_node,
        critical=True,
    )

    artists_list = ", ".join(song_details.artists) if song_details.artists else ""
    claim = f"This is a Spotify page, and it shows the song '{title}' performed by all of these artists: {artists_list}. The artist list should match completely (allowing for minor formatting differences and ignoring order)."

    await evaluator.verify(
        claim=claim,
        node=spotify_match_node,
        sources=song_details.spotify_song_url,
        additional_instruction="Check that all the mentioned artists appear on this Spotify song page. Minor formatting differences (like 'ft.' vs 'feat.' or capitalization) are acceptable. The order of artists doesn't matter."
    )

    # 3. Verify radio station
    radio_node = evaluator.add_leaf(
        id=f"song_{song_index}_radio_station",
        desc=f"Song '{title}' plays on radio station '{song_details.radio_station}'",
        parent=song_node,
        critical=True,
    )

    radio_claim = f"'{song_details.radio_station}' is a in-game radio station in Cyberpunk 2077 and the song '{title}' plays on this station."

    await evaluator.verify(
        claim=radio_claim,
        node=radio_node,
        sources=None,
        additional_instruction="Use your knowledge of Cyberpunk 2077 radio stations to verify this claim. Check both that the radio station exists in the game and that this specific song plays on it."
    )

    # 4. Verify artists (up to 5 artists max)
    if song_details.artists and song_details.spotify_artist_urls:
        artists_node = evaluator.add_parallel(
            id=f"song_{song_index}_artists",
            desc=f"All artists for song '{title}' are real-world artists",
            parent=song_node,
            critical=True,
        )

        max_artists = min(len(song_details.artists), len(song_details.spotify_artist_urls), 5)
        for i in range(max_artists):
            artist_node = evaluator.add_leaf(
                id=f"song_{song_index}_artist_{i + 1}",
                desc=f"Artist '{song_details.artists[i]}' is a real-world artist",
                parent=artists_node,
                critical=True,
            )

            artist_claim = f"This is a spotify page. And, this Spotify page is for the artist '{song_details.artists[i]}' and shows evidence that they are a real-world artist by having at least one of: non-default profile photo, verified artist status, or description/about section."

            await evaluator.verify(
                claim=artist_claim,
                node=artist_node,
                sources=song_details.spotify_artist_urls[i],
                additional_instruction="To decide whether it should pass, for simplicity, plz merely check: 1) The page is for the correct artist name, 2) At least one of these criteria is met: a) non-default profile photo (not the generic Spotify placeholder), b) verified artist badge/checkmark, c) an 'About' section with biographical information about the artist."
            )
    else:
        artist_node = evaluator.add_custom_node(
            result=False,
            id=f"song_{song_index}_artists",
            desc=f"All artists for song '{title}' are real-world artists - failed due to the missing of artists",
            parent=song_node,
            critical=True
        )


async def verify_unique_artists(
        evaluator: Evaluator,
        parent_node,
        song_titles: List[str],
        answer_text: str,
) -> None:
    """
    Verify that all mentioned artists are different using simple_verify.
    """
    unique_artists_node = evaluator.add_leaf(
        id="unique_artists",
        desc="All artists are different (no artist appears more than once)",
        parent=parent_node,
        critical=True,
    )

    # If 0 or 1 songs, automatically pass
    if len(song_titles) <= 1:
        unique_artists_node.score = 1.0
        unique_artists_node.status = "passed"
        return

    claim = f"All artists for each song mentioned across all the songs in this answer text are different from each other, with no artist or band appearing more than once: {answer_text}"

    await evaluator.verify(
        claim=claim,
        node=unique_artists_node,
        sources=None,
        additional_instruction="Carefully examine all artist names mentioned in the answer and determine if each artist/band is unique. Consider that the same artist might be listed under slightly different name formatting."
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
) -> Dict:
    """
    Evaluate a single answer and return a structured result dictionary.
    """
    # -------- 1. Initialize evaluator ------------------------------------- #
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
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

    # -------- 2. Extract song titles first -------------------------------- #
    extracted_titles = await evaluator.extract(
        prompt=prompt_extract_song_titles(),
        template_class=SongTitles,
        extraction_name="song_titles",
    )

    song_titles = extracted_titles.titles if extracted_titles.titles else []

    # -------- 3. Extract details for each song ---------------------------- #
    song_details_list = []
    for title in song_titles:
        details = await evaluator.extract(
            prompt=prompt_extract_song_details(title),
            template_class=SongDetails,
            extraction_name=f"song_details_{title}",
        )
        song_details_list.append(details)

    # Pad to 5 songs with empty details if needed
    while len(song_details_list) < 5:
        song_details_list.append(SongDetails())
        song_titles.append(None)

    # -------- 4. Build verification tree ---------------------------------- #
    
    # Verify unique artists (critical)
    await verify_unique_artists(evaluator, root, song_titles, answer_text=answer)

    # Create individual song verification nodes for all 5 songs
    for i in range(5):
        await verify_individual_song(
            evaluator,
            root,
            song_details_list[i],
            i + 1,
            song_titles[i],
        )

    # -------- 5. Get final result ----------------------------------------- #
    
    # Add custom info about extraction results
    evaluator.add_custom_info({
        "num_songs_found": len(song_titles),
        "songs_evaluated": min(len(song_titles), 5)
    }, info_type="extraction_stats")

    return evaluator.get_summary()