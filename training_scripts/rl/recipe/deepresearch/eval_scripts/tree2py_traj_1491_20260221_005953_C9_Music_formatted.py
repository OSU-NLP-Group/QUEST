import asyncio
import logging
from typing import Optional, List, Dict, Any, Tuple

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy

# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "grammy_2025_albums_2024_us_tours_2025"
TASK_DESCRIPTION = (
    "Identify albums released in 2024 that won at least one Grammy award at the 67th Annual Grammy Awards ceremony "
    "(held on February 2, 2025, in Los Angeles) and whose artists performed on concert tours in the United States "
    "during 2025. For each qualifying album, provide the following information: (1) Artist name, (2) Album title, "
    "(3) Release date, (4) Record label, (5) All Grammy award categories won at the 2025 ceremony, (6) At least two "
    "specific US tour venues with their city and state locations, and (7) Reference URLs supporting the information. "
    "You should identify at least three such albums with complete documentation."
)

CEREMONY_NO = 67
CEREMONY_YEAR = 2025
CEREMONY_DATE = "February 2, 2025"
REQUIRED_RELEASE_YEAR = 2024
MIN_REQUIRED_ALBUMS = 3
MAX_ALLOWED_ALBUMS = 5


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class Venue(BaseModel):
    name: Optional[str] = None
    city: Optional[str] = None
    state: Optional[str] = None


class AlbumEntry(BaseModel):
    artist: Optional[str] = None
    album_title: Optional[str] = None
    release_date: Optional[str] = None
    record_label: Optional[str] = None
    grammy_wins: List[str] = Field(default_factory=list)
    release_sources: List[str] = Field(default_factory=list)
    grammy_sources: List[str] = Field(default_factory=list)
    tour_sources: List[str] = Field(default_factory=list)
    venues: List[Venue] = Field(default_factory=list)


class AlbumsExtraction(BaseModel):
    albums: List[AlbumEntry] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_albums() -> str:
    return f"""
Extract up to {MAX_ALLOWED_ALBUMS} album entries mentioned in the answer that the answer claims meet the following:
- Album was released in {REQUIRED_RELEASE_YEAR}.
- The album won at least one Grammy at the {CEREMONY_NO}th Annual Grammy Awards ({CEREMONY_DATE}, Los Angeles).
- The artist performed on a U.S. concert tour during {CEREMONY_YEAR} (at least two specific U.S. venues with city and state).

For each album, extract the following fields exactly as presented in the answer text:
- artist: Artist name (string).
- album_title: Album title (string).
- release_date: The release date as written in the answer (string). If multiple dates are given, use the primary release date.
- record_label: Record label (string).
- grammy_wins: Array of all Grammy category names the album won at the {CEREMONY_NO}th Annual Grammys (use names as written in the answer).
- release_sources: Array of URLs specifically supporting the release date and/or record label. Only include URLs explicitly present in the answer.
- grammy_sources: Array of URLs supporting the Grammy win(s) at the {CEREMONY_NO}th ceremony. Only include URLs explicitly present in the answer.
- tour_sources: Array of URLs supporting the 2025 U.S. tour and venues. Only include URLs explicitly present in the answer.
- venues: Array of at least two venue objects, each with:
    - name: Venue name (string)
    - city: City (string)
    - state: U.S. state (string)
  If more than two venues are provided, include at least two.

General rules:
- Do NOT invent any URLs or facts; extract only from the provided answer text.
- If a field is missing in the answer, return null (for strings) or an empty array (for lists).
- List albums in order of appearance in the answer and limit to {MAX_ALLOWED_ALBUMS}.
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _nonempty_str(s: Optional[str]) -> bool:
    return bool(s and isinstance(s, str) and s.strip() != "")


def _merge_urls(*lists: List[str]) -> List[str]:
    seen = set()
    merged: List[str] = []
    for lst in lists:
        for url in lst:
            u = (url or "").strip()
            if not u:
                continue
            if u not in seen:
                seen.add(u)
                merged.append(u)
    return merged


def _normalize_ident(artist: Optional[str], title: Optional[str]) -> str:
    a = (artist or "").strip().lower()
    t = (title or "").strip().lower()
    return f"{a}||{t}"


# --------------------------------------------------------------------------- #
# Verification for a single album                                             #
# --------------------------------------------------------------------------- #
async def verify_one_album(
    evaluator: Evaluator,
    parent_node,
    album: AlbumEntry,
    idx: int,
) -> Dict[str, Any]:
    """
    Build verification subtree for a single album and run checks.
    Returns a dict with references to critical nodes for later aggregation.
    """
    ai = idx + 1  # Human-friendly index starting at 1
    album_node = evaluator.add_parallel(
        id=f"album_{ai}",
        desc=f"Album entry #{ai} (counts toward the ≥3 minimum if it satisfies all critical checks below).",
        parent=parent_node,
        critical=False
    )

    # 1) Artist name provided (existence check)
    artist_node = evaluator.add_custom_node(
        result=_nonempty_str(album.artist),
        id=f"album_{ai}_artist_name",
        desc="Artist name is provided.",
        parent=album_node,
        critical=True
    )

    # 2) Album title provided (existence check)
    title_node = evaluator.add_custom_node(
        result=_nonempty_str(album.album_title),
        id=f"album_{ai}_album_title",
        desc="Album title is provided.",
        parent=album_node,
        critical=True
    )

    # 3) Release date provided and indicates 2024 (evidence-backed)
    release_leaf = evaluator.add_leaf(
        id=f"album_{ai}_release_date_in_2024",
        desc="Release date is provided and indicates the album was released in 2024.",
        parent=album_node,
        critical=True
    )
    release_claim_date_part = f" with release date '{album.release_date}'" if _nonempty_str(album.release_date) else ""
    release_claim = (
        f"The album '{album.album_title}' by {album.artist} was released in {REQUIRED_RELEASE_YEAR}{release_claim_date_part}."
    )
    release_additional = (
        "Use only the provided release_sources URLs. Confirm that the album was released in 2024. "
        "If there are no release_sources or the extracted release_date is missing, mark as NOT SUPPORTED."
    )
    await evaluator.verify(
        claim=release_claim,
        node=release_leaf,
        sources=album.release_sources if album.release_sources else None,
        additional_instruction=release_additional
    )

    # 4) Record label provided (existence check)
    label_node = evaluator.add_custom_node(
        result=_nonempty_str(album.record_label),
        id=f"album_{ai}_record_label",
        desc="Record label is provided.",
        parent=album_node,
        critical=True
    )

    # 5) Grammy wins listed and official (evidence-backed)
    grammy_leaf = evaluator.add_leaf(
        id=f"album_{ai}_grammy_wins_official",
        desc=f"At least one Grammy win at the {CEREMONY_NO}th Annual Grammy Awards is supported, and all categories are official.",
        parent=album_node,
        critical=True
    )
    categories_str = ", ".join(album.grammy_wins) if album.grammy_wins else ""
    grammy_claim = (
        f"At the {CEREMONY_NO}th Annual Grammy Awards on {CEREMONY_DATE}, the album '{album.album_title}' by {album.artist} "
        f"won at least one category, specifically: {categories_str}."
    )
    grammy_additional = (
        f"Verify wins at the {CEREMONY_NO}th Grammys ({CEREMONY_YEAR}). "
        "Ensure the named categories correspond to official Recording Academy category names (allow minor wording variants). "
        "If no grammy_sources are provided or grammy_wins list is empty, mark as NOT SUPPORTED."
    )
    await evaluator.verify(
        claim=grammy_claim,
        node=grammy_leaf,
        sources=album.grammy_sources if album.grammy_sources else None,
        additional_instruction=grammy_additional
    )

    # 6) US Tour 2025 with at least two venues (evidence-backed)
    tour_leaf = evaluator.add_leaf(
        id=f"album_{ai}_us_tour_2025_with_2_venues",
        desc="Artist toured in the United States during 2025 and at least two specific US venues are provided (venue + city + state).",
        parent=album_node,
        critical=True
    )
    venues = [v for v in (album.venues or []) if _nonempty_str(v.name) and _nonempty_str(v.city) and _nonempty_str(v.state)]
    # Build claim with up to two venues
    v_parts = []
    for v in venues[:2]:
        v_parts.append(f"{v.name} in {v.city}, {v.state}")
    venues_str = "; ".join(v_parts)
    tour_claim = (
        f"In {CEREMONY_YEAR}, the artist {album.artist} performed in the United States at least at two venues: {venues_str}."
    )
    tour_additional = (
        f"Confirm that the performances occurred in {CEREMONY_YEAR} and at U.S. venues (venue in a U.S. city/state). "
        "Use only the provided tour_sources URLs. If fewer than two valid venues are extracted or no tour_sources provided, mark as NOT SUPPORTED."
    )
    await evaluator.verify(
        claim=tour_claim,
        node=tour_leaf,
        sources=album.tour_sources if album.tour_sources else None,
        additional_instruction=tour_additional
    )

    # 7) Reference URLs collectively support all claims (evidence-backed)
    refs_leaf = evaluator.add_leaf(
        id=f"album_{ai}_reference_urls_support_all_claims",
        desc="Reference URL(s) are provided that support the album’s release info, Grammy win(s)/categories, and 2025 US tour/venues.",
        parent=album_node,
        critical=True
    )
    all_sources = _merge_urls(album.release_sources, album.grammy_sources, album.tour_sources)
    refs_claim = (
        "The provided sources collectively support all of the following for this album: "
        f"(a) release in {REQUIRED_RELEASE_YEAR} and the stated record label; "
        f"(b) the {CEREMONY_NO}th Annual Grammy Awards ({CEREMONY_YEAR}) win(s) with the listed categories; and "
        "(c) the 2025 U.S. tour with the named venues (venue + city + state)."
    )
    refs_additional = (
        "Evaluate the combined set of URLs. If any of the three aspects (release+label, Grammy wins, 2025 U.S. tour venues) "
        "lacks adequate URL support in the provided lists, mark as NOT SUPPORTED."
    )
    await evaluator.verify(
        claim=refs_claim,
        node=refs_leaf,
        sources=all_sources if all_sources else None,
        additional_instruction=refs_additional
    )

    # Return references to critical nodes to compute qualification later
    return {
        "artist_node": artist_node,
        "title_node": title_node,
        "release_leaf": release_leaf,
        "label_node": label_node,
        "grammy_leaf": grammy_leaf,
        "tour_leaf": tour_leaf,
        "refs_leaf": refs_leaf,
        "album_ident": _normalize_ident(album.artist, album.album_title),
    }


# --------------------------------------------------------------------------- #
# Main evaluation                                                             #
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
    Evaluate an answer for the '2024 albums with 2025 Grammys wins and 2025 U.S. tours' task.
    """
    # Initialize evaluator (root as non-critical to avoid child criticality constraint)
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

    # Extract up to MAX_ALLOWED_ALBUMS album entries from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_albums(),
        template_class=AlbumsExtraction,
        extraction_name="albums_extraction"
    )

    # Keep at most MAX_ALLOWED_ALBUMS in order of appearance
    albums: List[AlbumEntry] = (extracted.albums or [])[:MAX_ALLOWED_ALBUMS]

    # Build per-album verification subtrees
    album_results: List[Dict[str, Any]] = []
    for i, album in enumerate(albums):
        res = await verify_one_album(evaluator, root, album, i)
        album_results.append(res)

    # Compute minimum qualifying album count and distinctness
    qualified_indices: List[int] = []
    unique_idents: set = set()

    for i, res in enumerate(album_results):
        critical_nodes = [
            res["artist_node"],
            res["title_node"],
            res["release_leaf"],
            res["label_node"],
            res["grammy_leaf"],
            res["tour_leaf"],
            res["refs_leaf"],
        ]
        # Determine if album qualifies: all critical checks passed
        if all(n.status == "passed" for n in critical_nodes):
            ident = res["album_ident"]
            if ident and ident not in unique_idents:
                unique_idents.add(ident)
                qualified_indices.append(i)

    min_req_met = len(unique_idents) >= MIN_REQUIRED_ALBUMS

    # Add the critical minimum-count/distinctness node at root
    evaluator.add_custom_node(
        result=min_req_met,
        id="meets_minimum_qualifying_album_count_and_distinctness",
        desc="At least three of the provided album entries are qualifying (meet all per-album critical constraints) and are distinct albums (not duplicates/alternate editions reused).",
        parent=root,
        critical=True
    )

    # Add custom info for debugging
    evaluator.add_custom_info(
        {
            "min_required": MIN_REQUIRED_ALBUMS,
            "qualified_count": len(qualified_indices),
            "distinct_qualified_idents": list(unique_idents),
            "qualified_indices_0_based": qualified_indices
        },
        info_type="qualification_stats",
        info_name="qualification_summary"
    )

    return evaluator.get_summary()