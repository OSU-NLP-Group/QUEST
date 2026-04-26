import asyncio
import logging
import re
from typing import Any, Optional, List, Dict

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "grammy_2026_albums"
TASK_DESCRIPTION = """
Identify 4 albums from the 2026 Grammy Awards (68th Annual Grammy Awards held on February 1, 2026) that meet the following criteria:

Album 1 must have:
- Won the Album of the Year award at the 2026 Grammys
- Been released on a record label based outside the United States
- Been released in January 2025

Album 2 must have:
- Been nominated for Album of the Year at the 2026 Grammys
- Been released by an artist who won Best New Artist at the 2026 Grammys
- Been released in the third quarter of 2025 (July-September)

Album 3 must have:
- Won Best Pop Vocal Album at the 2026 Grammys
- A title that is a single word
- Received at least 6 Grammy nominations at the 2026 Grammys

Album 4 must have:
- Been nominated for Album of the Year at the 2026 Grammys
- Been released by an artist who won Record of the Year at the 2026 Grammys (for any song)
- Been released in the fourth quarter of 2024 (October-December)
- Been released on a record label that includes 'Interscope' in its name

For each album, provide the album title, artist name, record label, release date, and relevant Grammy wins/nominations with supporting reference URLs.
""".strip()


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class AlbumEntry(BaseModel):
    album_title: Optional[str] = None
    artist_name: Optional[str] = None
    record_label: Optional[str] = None
    release_date: Optional[str] = None  # Keep as free-form string to maximize compatibility
    reference_urls: List[str] = Field(default_factory=list)


class AlbumsExtraction(BaseModel):
    album1: Optional[AlbumEntry] = None
    album2: Optional[AlbumEntry] = None
    album3: Optional[AlbumEntry] = None
    album4: Optional[AlbumEntry] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_albums() -> str:
    return """
Extract exactly four album entries as presented in the answer text, mapping them to album1, album2, album3, and album4 according to the ordering or labeling implied by the answer. For each album, extract the following fields strictly from the answer:

- album_title: The album title as written in the answer.
- artist_name: The (lead) artist name as written in the answer.
- record_label: The record label name as written in the answer (include imprint + parent if the answer lists them).
- release_date: The album’s release date as written in the answer (any format provided; do not normalize).
- reference_urls: A list of all URLs explicitly cited in the answer that relate to this album, its artist, its label, or Grammy wins/nominations for this album/artist. Include all relevant URLs mentioned near this album. If none are cited, return an empty list.

Rules:
- Do not invent or infer any values. Only extract what the answer explicitly states.
- If a required field is missing for an album, set that field to null.
- For URLs: extract only valid, complete URLs explicitly mentioned in the answer text (including inside markdown links). If no URLs are given, return an empty list.
- Return a JSON object with keys: album1, album2, album3, album4; each value is an object with the above fields (or null if the album is missing entirely).
""".strip()


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _nonempty(s: Optional[str]) -> bool:
    return bool(s and str(s).strip())


def _is_single_word(title: Optional[str]) -> bool:
    if not _nonempty(title):
        return False
    # Count alphanumeric tokens (allowing internal apostrophes/hyphens as part of a token)
    tokens = re.findall(r"[A-Za-z0-9]+(?:['-][A-Za-z0-9]+)?", title.strip())
    return len(tokens) == 1


def _label_includes_interscope(label: Optional[str]) -> bool:
    return _nonempty(label) and ("interscope" in label.lower())


# --------------------------------------------------------------------------- #
# Verification helpers                                                        #
# --------------------------------------------------------------------------- #
async def _add_and_verify(
    evaluator: Evaluator,
    parent,
    node_id: str,
    desc: str,
    claim: str,
    sources: Optional[List[str]],
    critical: bool = True,
    add_ins: str = ""
) -> None:
    leaf = evaluator.add_leaf(
        id=node_id,
        desc=desc,
        parent=parent,
        critical=critical
    )
    await evaluator.verify(
        claim=claim,
        node=leaf,
        sources=sources or [],
        additional_instruction=add_ins
    )


def _add_presence_nodes_for_album_common(evaluator: Evaluator, parent, album_prefix: str, album: Optional[AlbumEntry]) -> None:
    evaluator.add_custom_node(
        result=_nonempty(album.album_title) if album else False,
        id=f"{album_prefix}_Provides_Album_Title",
        desc=f"Response provides {album_prefix.replace('_', ' ')} album title.".replace("Album ", "Album "),
        parent=parent,
        critical=True
    )
    evaluator.add_custom_node(
        result=_nonempty(album.artist_name) if album else False,
        id=f"{album_prefix}_Provides_Artist_Name",
        desc=f"Response provides {album_prefix.replace('_', ' ')} artist name.".replace("Album ", "Album "),
        parent=parent,
        critical=True
    )
    evaluator.add_custom_node(
        result=_nonempty(album.record_label) if album else False,
        id=f"{album_prefix}_Provides_Record_Label",
        desc=f"Response provides {album_prefix.replace('_', ' ')} record label name.".replace("Album ", "Album "),
        parent=parent,
        critical=True
    )


def _add_supporting_urls_node(evaluator: Evaluator, parent, album_prefix: str, album: Optional[AlbumEntry], desc: str) -> None:
    evaluator.add_custom_node(
        result=bool(album and album.reference_urls and len(album.reference_urls) > 0),
        id=f"{album_prefix}_Supporting_Reference_URLs",
        desc=desc,
        parent=parent,
        critical=True
    )


# --------------------------------------------------------------------------- #
# Album-specific verification                                                 #
# --------------------------------------------------------------------------- #
async def verify_album_1(evaluator: Evaluator, parent, album: Optional[AlbumEntry]) -> None:
    node = evaluator.add_parallel(
        id="Album_1",
        desc="Album 1: AOTY winner + label based outside US + Jan 2025 release; include required fields and supporting sources.",
        parent=parent,
        critical=False
    )

    _add_presence_nodes_for_album_common(evaluator, node, "Album_1", album)
    _add_supporting_urls_node(
        evaluator, node, "Album_1", album,
        "Response provides one or more reference URLs for Album 1 that collectively support the required Album 1 claims (AOTY win, label, label location, and release date)."
    )

    title = album.album_title if album else ""
    artist = album.artist_name if album else ""
    label = album.record_label if album else ""
    urls = album.reference_urls if album else []

    # Won Album of the Year (2026)
    await _add_and_verify(
        evaluator, node,
        "Album_1_Won_AOTY_2026_Grammys",
        "Response correctly states that Album 1 won Album of the Year at the 2026 Grammys.",
        f"The album '{title}' by {artist} won 'Album of the Year' at the 2026 Grammy Awards.",
        urls,
        critical=True,
        add_ins="Focus on the 2026 Grammy Awards results page(s). The claim should be supported by winners lists. Allow minor title/artist formatting variants."
    )

    # Label based outside the US
    await _add_and_verify(
        evaluator, node,
        "Album_1_Label_Based_Outside_US",
        "Response correctly states (and is correct) that Album 1's record label is based outside the United States.",
        f"The record label '{label}' is headquartered or primarily based outside the United States.",
        urls,
        critical=True,
        add_ins="Verify the label's headquarters/base country. If multiple companies are listed, use the primary releasing label. If any subsidiary/label imprint is outside the U.S., and that is the releasing label, consider it outside the U.S."
    )

    # Released in January 2025
    await _add_and_verify(
        evaluator, node,
        "Album_1_Released_In_January_2025",
        "Response provides Album 1 release date and it is in January 2025.",
        f"The album '{title}' by {artist} was released in January 2025 (any date within 2025-01).",
        urls,
        critical=True,
        add_ins="Accept region-specific release dates if at least one primary/initial release date falls in January 2025."
    )


async def verify_album_2(evaluator: Evaluator, parent, album: Optional[AlbumEntry]) -> None:
    node = evaluator.add_parallel(
        id="Album_2",
        desc="Album 2: AOTY nomination + artist is Best New Artist winner + Q3 2025 release; include required fields and supporting sources.",
        parent=parent,
        critical=False
    )

    _add_presence_nodes_for_album_common(evaluator, node, "Album_2", album)
    _add_supporting_urls_node(
        evaluator, node, "Album_2", album,
        "Response provides one or more reference URLs for Album 2 that collectively support the required Album 2 claims (AOTY nomination, artist Best New Artist win, label, and release date)."
    )

    title = album.album_title if album else ""
    artist = album.artist_name if album else ""
    urls = album.reference_urls if album else []

    # Nominated for Album of the Year (2026)
    await _add_and_verify(
        evaluator, node,
        "Album_2_Nominated_For_AOTY_2026_Grammys",
        "Response correctly states that Album 2 was nominated for Album of the Year at the 2026 Grammys.",
        f"The album '{title}' by {artist} was nominated for 'Album of the Year' at the 2026 Grammy Awards.",
        urls,
        critical=True,
        add_ins="Look for official nominee lists for the 2026 Grammy Awards. Allow name/title variants."
    )

    # Artist won Best New Artist (2026)
    await _add_and_verify(
        evaluator, node,
        "Album_2_Artist_Won_Best_New_Artist_2026",
        "Response correctly states that Album 2's artist won Best New Artist at the 2026 Grammys.",
        f"The artist {artist} won 'Best New Artist' at the 2026 Grammy Awards.",
        urls,
        critical=True,
        add_ins="This claim is about the artist (regardless of album). Confirm the 2026 Best New Artist winner."
    )

    # Released in Q3 2025 (July–September)
    await _add_and_verify(
        evaluator, node,
        "Album_2_Released_In_Q3_2025",
        "Response provides Album 2 release date and it is in Q3 2025 (July–September).",
        f"The album '{title}' by {artist} was released during July–September 2025 (Q3 2025).",
        urls,
        critical=True,
        add_ins="Accept any primary/initial release dates in Jul/Aug/Sep 2025."
    )


async def verify_album_3(evaluator: Evaluator, parent, album: Optional[AlbumEntry]) -> None:
    node = evaluator.add_parallel(
        id="Album_3",
        desc="Album 3: Best Pop Vocal Album winner + single-word title + ≥6 nominations; include required fields and supporting sources.",
        parent=parent,
        critical=False
    )

    _add_presence_nodes_for_album_common(evaluator, node, "Album_3", album)

    # Explicit presence of release date (as per rubric)
    evaluator.add_custom_node(
        result=_nonempty(album.release_date) if album else False,
        id="Album_3_Provides_Release_Date",
        desc="Response provides Album 3 release date.",
        parent=node,
        critical=True
    )

    _add_supporting_urls_node(
        evaluator, node, "Album_3", album,
        "Response provides one or more reference URLs for Album 3 that collectively support the required Album 3 claims (Best Pop Vocal Album win, nomination count, label, and release date)."
    )

    title = album.album_title if album else ""
    artist = album.artist_name if album else ""
    urls = album.reference_urls if album else []

    # Won Best Pop Vocal Album (2026)
    await _add_and_verify(
        evaluator, node,
        "Album_3_Won_Best_Pop_Vocal_Album_2026",
        "Response correctly states that Album 3 won Best Pop Vocal Album at the 2026 Grammys.",
        f"The album '{title}' by {artist} won 'Best Pop Vocal Album' at the 2026 Grammy Awards.",
        urls,
        critical=True,
        add_ins="Confirm from official winners lists or reputable sources for the 2026 ceremony."
    )

    # Title is a single word (custom check)
    evaluator.add_custom_node(
        result=_is_single_word(title),
        id="Album_3_Title_Is_Single_Word",
        desc="Album 3 title provided in the response is a single word.",
        parent=node,
        critical=True
    )

    # At least 6 nominations (2026)
    await _add_and_verify(
        evaluator, node,
        "Album_3_At_Least_6_Nominations_2026",
        "Response correctly states that Album 3 received at least 6 Grammy nominations at the 2026 Grammys.",
        f"The album '{title}' by {artist} received at least 6 nominations at the 2026 Grammy Awards.",
        urls,
        critical=True,
        add_ins="Count only 2026 Grammy nominations for this album/artist. If a reputable source states a total count ≥ 6 for 2026, it is sufficient."
    )


async def verify_album_4(evaluator: Evaluator, parent, album: Optional[AlbumEntry]) -> None:
    node = evaluator.add_parallel(
        id="Album_4",
        desc="Album 4: AOTY nomination + artist is Record of the Year winner + Q4 2024 release + label contains 'Interscope'; include required fields and supporting sources.",
        parent=parent,
        critical=False
    )

    _add_presence_nodes_for_album_common(evaluator, node, "Album_4", album)
    _add_supporting_urls_node(
        evaluator, node, "Album_4", album,
        "Response provides one or more reference URLs for Album 4 that collectively support the required Album 4 claims (AOTY nomination, artist Record of the Year win, label containing 'Interscope', and release date)."
    )

    title = album.album_title if album else ""
    artist = album.artist_name if album else ""
    label = album.record_label if album else ""
    urls = album.reference_urls if album else []

    # Nominated for Album of the Year (2026)
    await _add_and_verify(
        evaluator, node,
        "Album_4_Nominated_For_AOTY_2026_Grammys",
        "Response correctly states that Album 4 was nominated for Album of the Year at the 2026 Grammys.",
        f"The album '{title}' by {artist} was nominated for 'Album of the Year' at the 2026 Grammy Awards.",
        urls,
        critical=True,
        add_ins="Check official nominee lists for the 2026 Grammys."
    )

    # Artist won Record of the Year (2026) for any song
    await _add_and_verify(
        evaluator, node,
        "Album_4_Artist_Won_Record_Of_The_Year_2026",
        "Response correctly states that Album 4's lead artist won Record of the Year at the 2026 Grammys (for any song).",
        f"The lead artist {artist} won 'Record of the Year' at the 2026 Grammy Awards (for any song).",
        urls,
        critical=True,
        add_ins="This is about the artist's ROTY win in 2026 (any song). It does not need to be a track from the selected album."
    )

    # Released in Q4 2024 (Oct–Dec)
    await _add_and_verify(
        evaluator, node,
        "Album_4_Released_In_Q4_2024",
        "Response provides Album 4 release date and it is in Q4 2024 (October–December).",
        f"The album '{title}' by {artist} was released during October–December 2024 (Q4 2024).",
        urls,
        critical=True,
        add_ins="Accept a primary/initial release date in Oct/Nov/Dec 2024."
    )

    # Label includes 'Interscope' (custom check)
    evaluator.add_custom_node(
        result=_label_includes_interscope(label),
        id="Album_4_Label_Includes_Interscope",
        desc="Album 4 record label name provided in the response includes the substring 'Interscope'.",
        parent=node,
        critical=True
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
    Evaluate an answer for the '2026 Grammys albums' task.
    """
    # Initialize evaluator with a neutral root; we will add our rubric root as a child node.
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

    # Add rubric root under framework root to mirror the rubric structure
    rubric_root = evaluator.add_parallel(
        id="Find_4_Grammy_Nominated_Albums",
        desc="Provide 4 albums that satisfy the album-specific Grammy/metadata constraints and include the requested fields and supporting sources.",
        parent=root,
        critical=False
    )

    # Extract structured album info
    albums = await evaluator.extract(
        prompt=prompt_extract_albums(),
        template_class=AlbumsExtraction,
        extraction_name="albums_extraction"
    )

    # Verify each album sub-tree
    await verify_album_1(evaluator, rubric_root, albums.album1 if albums else None)
    await verify_album_2(evaluator, rubric_root, albums.album2 if albums else None)
    await verify_album_3(evaluator, rubric_root, albums.album3 if albums else None)
    await verify_album_4(evaluator, rubric_root, albums.album4 if albums else None)

    return evaluator.get_summary()