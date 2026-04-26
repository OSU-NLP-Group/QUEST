import asyncio
import logging
import re
from datetime import datetime
from typing import Optional, List, Dict, Any, Tuple

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "albums_2025_riaa_platinum"
TASK_DESCRIPTION = """
Identify 3 different studio albums that were released between January 1, 2025 and December 31, 2025, and achieved RIAA Platinum certification or higher during the calendar year 2025. Across these 3 albums, they must collectively satisfy the following distribution requirements: (1) At least one album was released in Q1 2025 (January 1 - March 31, 2025), (2) At least one album was released in Q3 or Q4 2025 (July 1 - December 31, 2025), and (3) At least one album's artist was a headlining performer at either the Coachella Valley Music and Arts Festival 2025 or Lollapalooza Chicago 2025. For each of the 3 albums, provide: album title, artist name, exact release date (in MM/DD/YYYY format), RIAA certification level achieved in 2025 with the exact certification date, and a reference URL confirming this information.
"""


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class AlbumItem(BaseModel):
    album_title: Optional[str] = None
    artist_name: Optional[str] = None
    release_date: Optional[str] = None  # Keep as string; we'll parse heuristically
    riaa_certification_level: Optional[str] = None  # e.g., "Platinum", "2x Platinum", "Diamond"
    riaa_certification_date: Optional[str] = None   # expected in 2025, string format in answer
    reference_urls: List[str] = Field(default_factory=list)  # primary sources cited for this album

    # Optional festival info (to support distribution requirement #3)
    festival_headliner_event: Optional[str] = None  # "Coachella 2025" or "Lollapalooza Chicago 2025"
    festival_headliner_urls: List[str] = Field(default_factory=list)


class AlbumsExtraction(BaseModel):
    albums: List[AlbumItem] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_albums() -> str:
    return """
    Extract up to the first three distinct studio albums mentioned in the answer (in the same order they appear).
    For each album, extract the following fields exactly as presented in the answer text:
    - album_title: The album's title (string).
    - artist_name: The performing artist or group (string).
    - release_date: The exact release date string as provided in the answer (prefer MM/DD/YYYY if shown; otherwise keep the answer's textual date like 'January 5, 2025').
    - riaa_certification_level: The RIAA certification level achieved in calendar year 2025 (e.g., 'Platinum', '2x Platinum', '3x Platinum', 'Diamond'); extract exactly as written in the answer.
    - riaa_certification_date: The exact RIAA certification date string (prefer MM/DD/YYYY if shown; otherwise keep the answer's textual date) that is explicitly in 2025.
    - reference_urls: A list of all URLs the answer cites to support this album’s details (include RIAA page if provided; include any official or reputable pages for release date/title/artist).
    - festival_headliner_event: If the answer states that the album’s artist was a headlining performer at 'Coachella 2025' or 'Lollapalooza Chicago 2025', return the corresponding string ('Coachella 2025' or 'Lollapalooza Chicago 2025'); otherwise null.
    - festival_headliner_urls: A list of URLs (if any) that the answer cites to support the headliner claim for this artist at Coachella 2025 or Lollapalooza Chicago 2025.

    Rules:
    - Only include studio albums whose release dates fall within calendar year 2025 according to the answer.
    - Only include items where the answer claims an RIAA certification of Platinum or higher achieved during 2025.
    - If a field is missing in the answer, set it to null for strings and [] for URL lists.
    - For URLs, include the actual URLs (including from markdown links). Ignore clearly invalid URLs.
    - Return the result as an object with an 'albums' array of up to 3 AlbumItem objects.
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def is_valid_url(url: Optional[str]) -> bool:
    if not url or not isinstance(url, str):
        return False
    return bool(re.match(r"^https?://", url.strip(), re.IGNORECASE))


def normalize_urls(urls: Optional[List[str]]) -> List[str]:
    if not urls:
        return []
    valid = []
    for u in urls:
        if isinstance(u, str) and u.strip():
            # Add protocol if missing
            if not re.match(r"^https?://", u.strip(), re.IGNORECASE):
                u = "http://" + u.strip()
            if is_valid_url(u.strip()):
                valid.append(u.strip())
    # Deduplicate while preserving order
    seen = set()
    deduped = []
    for u in valid:
        if u not in seen:
            seen.add(u)
            deduped.append(u)
    return deduped


_DATE_FORMATS = [
    "%m/%d/%Y",
    "%m/%d/%y",
    "%Y-%m-%d",
    "%d-%m-%Y",
    "%B %d, %Y",
    "%b %d, %Y",
    "%d %B %Y",
    "%d %b %Y",
    "%Y.%m.%d",
    "%m.%d.%Y",
    "%Y/%m/%d",
]


def parse_date_str(date_str: Optional[str]) -> Optional[datetime]:
    if not date_str or not isinstance(date_str, str):
        return None
    s = date_str.strip()
    for fmt in _DATE_FORMATS:
        try:
            return datetime.strptime(s, fmt)
        except Exception:
            continue
    # Try to normalize ordinal suffixes (e.g., "January 1st, 2025")
    s2 = re.sub(r"(\d+)(st|nd|rd|th)", r"\1", s, flags=re.IGNORECASE)
    if s2 != s:
        for fmt in _DATE_FORMATS:
            try:
                return datetime.strptime(s2, fmt)
            except Exception:
                continue
    return None


def year_is_2025(dt: Optional[datetime]) -> bool:
    return bool(dt and dt.year == 2025)


def is_q1_2025(dt: Optional[datetime]) -> bool:
    return bool(dt and dt.year == 2025 and 1 <= dt.month <= 3)


def is_q3_or_q4_2025(dt: Optional[datetime]) -> bool:
    return bool(dt and dt.year == 2025 and 7 <= dt.month <= 12)


def contains_platinum_or_higher(level: Optional[str]) -> bool:
    if not level or not isinstance(level, str):
        return False
    s = level.lower()
    # Accept "platinum", "multi-platinum", "2x platinum", "diamond"
    return ("platinum" in s) or ("diamond" in s)


def pick_festival_candidate(albums: List[AlbumItem]) -> Tuple[Optional[int], Optional[str], List[str]]:
    """
    Returns (album_index, normalized_event, urls) for the first album that appears
    to have headliner info. If not explicitly provided, try to infer via URLs that
    contain festival names.
    """
    # Priority 1: explicit festival info with dedicated URLs
    for idx, a in enumerate(albums):
        ev = (a.festival_headliner_event or "").strip().lower()
        urls = normalize_urls(a.festival_headliner_urls)
        if ev and urls:
            norm_event = "Coachella 2025" if "coachella" in ev else ("Lollapalooza Chicago 2025" if "lollapalooza" in ev else a.festival_headliner_event or "Festival 2025")
            return idx, norm_event, urls

    # Priority 2: inferred by scanning general reference URLs for festival domains/keywords
    for idx, a in enumerate(albums):
        urls = normalize_urls(a.reference_urls)
        fest_urls = [u for u in urls if any(k in u.lower() for k in ["coachella", "lollapalooza"])]
        if fest_urls:
            # Attempt to label the event from URL content
            if any("coachella" in u.lower() for u in fest_urls):
                return idx, "Coachella 2025", fest_urls
            if any("lollapalooza" in u.lower() for u in fest_urls):
                return idx, "Lollapalooza Chicago 2025", fest_urls

    return None, None, []


# --------------------------------------------------------------------------- #
# Verification builders                                                       #
# --------------------------------------------------------------------------- #
async def verify_album(
    evaluator: Evaluator,
    parent_node,
    album: AlbumItem,
    album_idx: int,
) -> None:
    """
    Build verification subtree for a single album under `parent_node`.
    All children here are critical because the parent (Album_Set_Provided) is critical.
    """
    # Album parallel group
    album_node = evaluator.add_parallel(
        id=f"Album_{album_idx+1}",
        desc=f"{['First','Second','Third'][album_idx]} album meets individual album criteria",
        parent=parent_node,
        critical=True
    )

    # Normalize URLs (include both reference and festival URLs to maximize coverage when relevant)
    refs = normalize_urls(album.reference_urls)
    fest_refs = normalize_urls(album.festival_headliner_urls)
    all_urls = normalize_urls(refs + fest_refs)

    title = album.album_title or ""
    artist = album.artist_name or ""
    rel_date = album.release_date or ""
    cert_level = album.riaa_certification_level or ""
    cert_date = album.riaa_certification_date or ""

    # 1) Title leaf
    title_leaf = evaluator.add_leaf(
        id=f"Album_{album_idx+1}_Title",
        desc="Album title is provided",
        parent=album_node,
        critical=True
    )
    title_claim = f"The referenced page is about an album titled '{title}'."
    await evaluator.verify(
        claim=title_claim,
        node=title_leaf,
        sources=all_urls,
        additional_instruction="Accept reasonable variants or stylization of the album title. The page should clearly be about this album."
    )

    # 2) Artist leaf
    artist_leaf = evaluator.add_leaf(
        id=f"Album_{album_idx+1}_Artist",
        desc="Artist name is provided",
        parent=album_node,
        critical=True
    )
    artist_claim = f"The album titled '{title}' is by the artist '{artist}'."
    await evaluator.verify(
        claim=artist_claim,
        node=artist_leaf,
        sources=all_urls,
        additional_instruction="Allow minor name variations (case, middle names, '&' vs 'and'). If a collaboration, accept if the named artist is clearly primary or co-primary."
    )

    # 3) Release date leaf
    release_leaf = evaluator.add_leaf(
        id=f"Album_{album_idx+1}_Release_Date",
        desc="Exact release date in 2025 is provided in MM/DD/YYYY format",
        parent=album_node,
        critical=True
    )
    release_claim = f"The album '{title}' by '{artist}' was released on {rel_date} in calendar year 2025."
    await evaluator.verify(
        claim=release_claim,
        node=release_leaf,
        sources=all_urls,
        additional_instruction=(
            "Verify that the page explicitly states a release date matching the provided date string "
            "or an obviously equivalent representation (e.g., 'January 5, 2025' vs '01/05/2025'). "
            "If multiple regional/format dates are listed, accept if one matches. The date must be in 2025."
        )
    )

    # 4) Certification leaf (RIAA Platinum or higher in 2025, with exact date)
    cert_leaf = evaluator.add_leaf(
        id=f"Album_{album_idx+1}_Certification",
        desc="RIAA certification level (Platinum or higher) achieved in 2025 is provided with exact certification date",
        parent=album_node,
        critical=True
    )
    cert_claim = (
        f"The album '{title}' by '{artist}' achieved RIAA {cert_level} certification on {cert_date} "
        f"during calendar year 2025 (Platinum or higher)."
    )
    await evaluator.verify(
        claim=cert_claim,
        node=cert_leaf,
        sources=all_urls,
        additional_instruction=(
            "Prefer the official RIAA Gold & Platinum database page if available. "
            "Accept 'Platinum' or any higher tier (e.g., Multi-Platinum, Diamond). "
            "The certification date must be in 2025 and match (or be equivalent to) the provided date string."
        )
    )

    # 5) Reference URL existence leaf (custom, critical)
    has_valid_ref = len(refs) > 0
    ref_leaf = evaluator.add_custom_node(
        result=has_valid_ref,
        id=f"Album_{album_idx+1}_Reference_URL",
        desc="Valid reference URL supporting the album information is provided",
        parent=album_node,
        critical=True
    )


async def verify_distribution_requirements(
    evaluator: Evaluator,
    parent_node,
    albums: List[AlbumItem],
) -> None:
    """
    Build verification subtree for distribution requirements.
    - Q1 release present (custom boolean check)
    - Q3 or Q4 release present (custom boolean check)
    - Festival headliner present (LLM verify with URLs if provided/inferred)
    """
    dist_node = evaluator.add_parallel(
        id="Distribution_Requirements_Met",
        desc="The set of 3 albums collectively satisfies all distribution requirements",
        parent=parent_node,
        critical=True
    )

    # Parse release dates
    parsed_dates: List[Optional[datetime]] = [parse_date_str(a.release_date) for a in albums]

    q1_present = any(is_q1_2025(d) for d in parsed_dates)
    q34_present = any(is_q3_or_q4_2025(d) for d in parsed_dates)

    # Q1 present
    evaluator.add_custom_node(
        result=q1_present,
        id="Q1_Release_Present",
        desc="At least one album was released in Q1 2025 (January 1 - March 31, 2025)",
        parent=dist_node,
        critical=True
    )

    # Q3 or Q4 present
    evaluator.add_custom_node(
        result=q34_present,
        id="Q3_or_Q4_Release_Present",
        desc="At least one album was released in Q3 or Q4 2025 (July 1 - December 31, 2025)",
        parent=dist_node,
        critical=True
    )

    # Festival headliner present: pick a candidate album + sources
    cand_idx, norm_event, fest_urls = pick_festival_candidate(albums)
    # Build the leaf
    fest_leaf = evaluator.add_leaf(
        id="Festival_Headliner_Present",
        desc=("At least one album's artist was a headlining performer at Coachella Valley Music and Arts Festival 2025 "
              "(April 11-13 or 18-20, 2025) OR Lollapalooza Chicago 2025 (July 31 - August 3, 2025)"),
        parent=dist_node,
        critical=True
    )

    # Construct claim
    if cand_idx is not None and norm_event:
        artist = albums[cand_idx].artist_name or "the identified artist"
        fest_claim = f"Artist '{artist}' was a headlining performer at {norm_event}."
        await evaluator.verify(
            claim=fest_claim,
            node=fest_leaf,
            sources=fest_urls,
            additional_instruction=(
                "Verify explicitly that the artist is listed as a HEADLINER (top-billed) for the 2025 edition of the named festival. "
                "Accept official festival lineup pages or reputable news sources. "
                "For Coachella, either Weekend 1 or Weekend 2 is acceptable. "
                "For Lollapalooza Chicago 2025, confirm the official 2025 headliner list."
            )
        )
    else:
        # No candidate with credible URLs: verify will likely fail without sources
        # Still attempt a conservative generic claim with no sources; this should fail per evidence requirement.
        await evaluator.verify(
            claim="At least one of the three album artists was a headlining performer at Coachella 2025 or Lollapalooza Chicago 2025.",
            node=fest_leaf,
            sources=None,
            additional_instruction=(
                "No festival headliner URLs were provided or inferred. Without explicit evidence, this should not be considered supported."
            )
        )


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
    model: str = "o4-mini",
) -> Dict:
    """
    Evaluate an answer for the 2025 RIAA Platinum albums task and return a structured result dict.
    """
    # Initialize evaluator (root is always non-critical)
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

    # Extract album set (first up to 3)
    extracted = await evaluator.extract(
        prompt=prompt_extract_albums(),
        template_class=AlbumsExtraction,
        extraction_name="albums_extraction"
    )

    # Normalize and limit to 3 items; pad if fewer
    albums: List[AlbumItem] = list(extracted.albums or [])[:3]
    while len(albums) < 3:
        albums.append(AlbumItem())

    # Add top-level "Task_Completion" node (critical)
    task_node = evaluator.add_parallel(
        id="Task_Completion",
        desc="Successfully identify 3 studio albums meeting all specified criteria",
        parent=root,
        critical=True
    )

    # Album set provided (critical) -> contains Album_1..3 (each critical)
    album_set_node = evaluator.add_parallel(
        id="Album_Set_Provided",
        desc="Three distinct albums are identified with complete information",
        parent=task_node,
        critical=True
    )

    # Build per-album verification subtrees
    for idx in range(3):
        await verify_album(evaluator, album_set_node, albums[idx], idx)

    # Distribution requirements (critical)
    await verify_distribution_requirements(evaluator, task_node, albums)

    # Optionally record some computed info for transparency
    parsed_dates = [parse_date_str(a.release_date) for a in albums]
    info_summary = {
        "parsed_release_dates": [d.strftime("%Y-%m-%d") if d else None for d in parsed_dates],
        "q1_present": any(is_q1_2025(d) for d in parsed_dates),
        "q3_or_q4_present": any(is_q3_or_q4_2025(d) for d in parsed_dates),
        "albums": [
            {
                "title": a.album_title,
                "artist": a.artist_name,
                "release_date_str": a.release_date,
                "cert_level": a.riaa_certification_level,
                "cert_date_str": a.riaa_certification_date,
                "reference_urls": normalize_urls(a.reference_urls),
                "festival_headliner_event": a.festival_headliner_event,
                "festival_headliner_urls": normalize_urls(a.festival_headliner_urls),
            }
            for a in albums
        ]
    }
    evaluator.add_custom_info(info_summary, info_type="parsed_info", info_name="computed_fields")

    return evaluator.get_summary()