import asyncio
import logging
from typing import Optional, List, Dict, Any, Tuple
from urllib.parse import urlparse

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy

# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "streaming_originals_2025_q1_q3"
TASK_DESCRIPTION = """
Identify four streaming original series that premiered between January 1, 2025 and November 30, 2025. For each series, provide the following information: official series title, streaming platform (must be Netflix, Apple TV+, HBO Max, or Amazon Prime Video), exact premiere date, number of episodes in the first season (must be between 4 and 15 episodes), primary genre(s), creator(s) or showrunner(s), and a reference URL (official platform page or IMDb page). Additionally, ensure that the four series represent at least three different streaming platforms (no single platform can have more than two series) and the four series represent at least three different primary genre categories.
""".strip()

ALLOWED_PLATFORMS = {"Netflix", "Apple TV+", "HBO Max", "Amazon Prime Video"}
DATE_RANGE_START = "2025-01-01"
DATE_RANGE_END = "2025-11-30"

# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class SeriesItem(BaseModel):
    title: Optional[str] = None
    platform: Optional[str] = None
    premiere_date: Optional[str] = None  # keep as free-form string to be robust
    season1_episode_count: Optional[str] = None  # keep as free-form string
    primary_genres: List[str] = Field(default_factory=list)
    creators: List[str] = Field(default_factory=list)
    reference_url: Optional[str] = None


class SeriesExtraction(BaseModel):
    series: List[SeriesItem] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_series() -> str:
    return """
    Extract up to four streaming original series mentioned in the answer (use the first four if more are present).
    For each series, return an object with the following fields strictly as they appear in the answer:
    - title: official series title (string; null if missing)
    - platform: the streaming platform name as stated (string; null if missing). Prefer one of: Netflix, Apple TV+, HBO Max / Max, Amazon Prime Video / Prime Video.
    - premiere_date: the exact premiere/initial release date as written (string; null if missing). Do not infer or reformat.
    - season1_episode_count: the number of episodes in the first season as stated (string; null if missing). Keep any units/words (e.g., "8", "8 episodes").
    - primary_genres: array of primary genre strings. If the answer lists combined genres like "drama/thriller", split into separate items ["drama","thriller"]. Empty array if not provided.
    - creators: array of credited creator(s) or showrunner(s) names. Empty array if not provided.
    - reference_url: a single URL that is either the official platform page for the series or the IMDb title page for the series. 
                     If multiple URLs are given, choose the most authoritative (prefer official platform page over IMDb; prefer IMDb over press/news). 
                     If none are provided, set to null.

    Return a JSON object with:
    { "series": [ {title, platform, premiere_date, season1_episode_count, primary_genres, creators, reference_url}, ... ] }

    Do NOT invent any values that are not explicitly present in the answer text. If a field is missing, use null (or [] for arrays).
    """.strip()


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def _strip_or_none(s: Optional[str]) -> Optional[str]:
    return s.strip() if isinstance(s, str) and s.strip() != "" else None


def canonicalize_platform(name: Optional[str]) -> Optional[str]:
    if not name:
        return None
    s = name.strip().lower()
    # Normalize common synonyms
    if s in {"netflix"}:
        return "Netflix"
    if s in {"apple tv+", "appletv+", "apple tv plus", "apple tv plus+", "apple tv"}:
        return "Apple TV+"
    if s in {"hbo max", "max"}:
        return "HBO Max"  # Treat Max as HBO Max for rubric compatibility
    if s in {"prime video", "amazon prime video", "amazon prime", "amazon video"}:
        return "Amazon Prime Video"
    # Try partial contains
    if "netflix" in s:
        return "Netflix"
    if "apple" in s and ("tv" in s or "+" in s):
        return "Apple TV+"
    if "hbo" in s or "max" in s:
        return "HBO Max"
    if "prime" in s or "amazon" in s:
        return "Amazon Prime Video"
    return None


def is_allowed_reference_url(url: Optional[str]) -> bool:
    if not url:
        return False
    try:
        p = urlparse(url.strip())
        host = (p.hostname or "").lower()
        path = (p.path or "").lower()
        if not host:
            return False
        # IMDb title pages
        if "imdb.com" in host:
            return path.startswith("/title/")
        # Official platforms
        if "netflix.com" in host:
            return True
        if "tv.apple.com" in host or "apple.com" in host:
            return True
        if "max.com" in host or "hbo.com" in host:
            return True
        if "primevideo.com" in host:
            return True
        if "amazon.com" in host:
            # Accept common Prime Video paths
            if any(seg in path for seg in ["/primevideo", "/gp/video", "/gp/product", "/dp/"]):
                return True
            # Otherwise still accept as official series landing (be lenient)
            return True
        return False
    except Exception:
        return False


def categorize_genre(raw: Optional[str]) -> Optional[str]:
    if not raw:
        return None
    s = raw.strip().lower()
    # Map to broad categories
    if "drama" in s:
        return "Drama"
    if "comedy" in s:
        return "Comedy"
    if "thriller" in s:
        return "Thriller"
    if "action" in s:
        return "Action"
    if "sci" in s or "science fiction" in s or "science-fiction" in s:
        return "Science Fiction"
    if "fantasy" in s:
        return "Fantasy"
    if "horror" in s:
        return "Horror"
    if "document" in s:
        return "Documentary"
    if "reality" in s or "unscripted" in s:
        return "Reality"
    if "crime" in s:
        return "Crime"
    if "mystery" in s:
        return "Mystery"
    if "romance" in s or "rom-com" in s:
        return "Romance"
    if "animation" in s or "animated" in s:
        return "Animation"
    if "family" in s:
        return "Family"
    if "adventure" in s:
        return "Adventure"
    if "biograph" in s:
        return "Biography"
    if "music" in s or "musical" in s:
        return "Music"
    if "history" in s or "histor" in s:
        return "History"
    if "war" in s:
        return "War"
    if "western" in s:
        return "Western"
    if "sport" in s:
        return "Sport"
    if "talk" in s:
        return "Talk Show"
    if "game show" in s or "gameshow" in s:
        return "Game Show"
    if "news" in s:
        return "News"
    return "Other"


def primary_genre_category(genres: List[str]) -> Optional[str]:
    for g in genres:
        cat = categorize_genre(g)
        if cat:
            return cat
    return None


# --------------------------------------------------------------------------- #
# Verification subroutines                                                    #
# --------------------------------------------------------------------------- #
async def verify_single_series(
    evaluator: Evaluator,
    parent_node,
    series: SeriesItem,
    index: int
) -> None:
    """
    Build and run verification nodes for a single series.
    Node IDs follow the rubric naming (Series_1_..., Series_2_..., etc.).
    """
    sid = index + 1
    series_node = evaluator.add_parallel(
        id=f"Series_{sid}",
        desc=f"{['First','Second','Third','Fourth'][index]} identified series meeting all requirements",
        parent=parent_node,
        critical=False
    )

    # 1) Reference URL existence and validity (critical, custom node)
    ref_ok = is_allowed_reference_url(series.reference_url)
    evaluator.add_custom_node(
        result=ref_ok,
        id=f"Series_{sid}_Reference_URL",
        desc=f"A reference URL (official platform page or IMDb page) is provided for Series {sid}",
        parent=series_node,
        critical=True
    )

    # 2) Title provided (critical, custom node for existence in answer)
    title_ok = _strip_or_none(series.title) is not None
    evaluator.add_custom_node(
        result=title_ok,
        id=f"Series_{sid}_Title",
        desc=f"Official series title is provided for Series {sid}",
        parent=series_node,
        critical=True
    )

    # Prepare claims for URL-backed verifications (they will auto-skip if critical siblings failed)
    platform_node = evaluator.add_leaf(
        id=f"Series_{sid}_Platform",
        desc=f"Series {sid} is an original series on Netflix, Apple TV+, HBO Max, or Amazon Prime Video",
        parent=series_node,
        critical=True
    )
    premiere_node = evaluator.add_leaf(
        id=f"Series_{sid}_Premiere_Date",
        desc=f"Series {sid} premiered between January 1, 2025 and November 30, 2025",
        parent=series_node,
        critical=True
    )
    episode_node = evaluator.add_leaf(
        id=f"Series_{sid}_Episode_Count",
        desc=f"Series {sid} first season has between 4 and 15 episodes (inclusive)",
        parent=series_node,
        critical=True
    )
    genre_node = evaluator.add_leaf(
        id=f"Series_{sid}_Genre",
        desc=f"Series {sid} has at least one clearly defined primary genre",
        parent=series_node,
        critical=True
    )
    creator_node = evaluator.add_leaf(
        id=f"Series_{sid}_Creator",
        desc=f"Series {sid} has clearly credited creator(s) or showrunner(s)",
        parent=series_node,
        critical=True
    )

    title_text = series.title or ""
    platform_text = series.platform or ""
    premiere_text = series.premiere_date or ""
    ep_text = series.season1_episode_count or ""
    genres_text = ", ".join(series.primary_genres) if series.primary_genres else ""
    creators_text = ", ".join(series.creators) if series.creators else ""
    src = series.reference_url if ref_ok else None  # helps ensure grounding only when URL is valid

    # Build claims
    platform_claim = (
        f"The series titled '{title_text}' is a streaming original on {platform_text}, "
        f"which is one of: Netflix, Apple TV+, HBO Max (aka Max), or Amazon Prime Video (aka Prime Video)."
    )
    premiere_claim = (
        f"The series titled '{title_text}' premiered on {premiere_text}, "
        f"and that premiere date falls between {DATE_RANGE_START} and {DATE_RANGE_END} inclusive."
    )
    episodes_claim = (
        f"The first season of '{title_text}' consists of {ep_text} episodes, "
        f"and this count is between 4 and 15 inclusive."
    )
    genre_claim = (
        f"The primary genre(s) of '{title_text}' include: {genres_text}."
    )
    creator_claim = (
        f"The creator(s) or showrunner(s) credited for '{title_text}' include: {creators_text}."
    )

    # Batch verify URL-backed claims (auto-routing by evaluator.verify)
    claims_and_sources: List[Tuple[str, Optional[str], Any, Optional[str]]] = [
        (
            platform_claim,
            src,
            platform_node,
            "Confirm the platform is the official streaming home and that it is an 'Original' (e.g., 'Netflix Original', 'Apple TV+ Original', 'Max Original', or 'Amazon/Prime Video Original'). "
            "Accept synonyms: 'Max' = 'HBO Max'; 'Prime Video' = 'Amazon Prime Video'; letter casing variants OK. "
            "If the page clearly indicates a different platform, mark as not supported."
        ),
        (
            premiere_claim,
            src,
            premiere_node,
            "Look for 'Premiere', 'Original release', 'First aired', or similar. Verify the stated date matches the page and falls between 2025-01-01 and 2025-11-30 inclusive. "
            "Be tolerant of date formatting differences (e.g., 'Jan 5, 2025')."
        ),
        (
            episodes_claim,
            src,
            episode_node,
            "Verify the total number of episodes in Season 1 (not the entire series). If the page clearly lists Season 1 with an episode count, check that it matches and lies between 4 and 15 inclusive. "
            "If the page reports a different count or outside the range, mark as not supported."
        ),
        (
            genre_claim,
            src,
            genre_node,
            "Verify primary/top-level genres as indicated on the page. Synonyms and minor naming variations (e.g., 'Sci-Fi' vs 'Science Fiction') are acceptable."
        ),
        (
            creator_claim,
            src,
            creator_node,
            "Verify credited creator(s) or showrunner(s). Accept 'Created by', 'Developed by', or explicit showrunner credit lines."
        ),
    ]

    await evaluator.batch_verify(claims_and_sources)


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
    Evaluate an answer for the 2025 streaming originals task.
    """
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

    # 1) Extract structured info
    extracted = await evaluator.extract(
        prompt=prompt_extract_series(),
        template_class=SeriesExtraction,
        extraction_name="series_extraction",
    )

    # Ensure exactly 4 series entries (pad with empty if needed)
    items: List[SeriesItem] = list(extracted.series[:4])
    while len(items) < 4:
        items.append(SeriesItem())

    # 2) Build per-series verification subtrees
    for idx in range(4):
        await verify_single_series(evaluator, root, items[idx], idx)

    # 3) Cross-series diversity requirements (critical)
    diversity_node = evaluator.add_parallel(
        id="Diversity_Requirements",
        desc="Cross-series diversity requirements for platform and genre distribution",
        parent=root,
        critical=True  # Parent critical -> children must be critical per framework constraint
    )

    # Compute platform and genre categories
    canonical_platforms: List[Optional[str]] = [canonicalize_platform(_strip_or_none(s.platform)) for s in items]
    primary_genre_cats: List[Optional[str]] = [primary_genre_category(s.primary_genres) if s.primary_genres else None for s in items]

    # Platform diversity: all four must be recognized allowed platforms; at least 3 different; no platform count > 2
    platform_counts: Dict[str, int] = {}
    platforms_valid = True
    for p in canonical_platforms:
        if p is None or p not in ALLOWED_PLATFORMS:
            platforms_valid = False
            break
        platform_counts[p] = platform_counts.get(p, 0) + 1
    unique_platforms = len(platform_counts)
    max_platform_occurrence = max(platform_counts.values()) if platform_counts else 0
    platform_diversity_ok = platforms_valid and unique_platforms >= 3 and max_platform_occurrence <= 2

    evaluator.add_custom_node(
        result=platform_diversity_ok,
        id="Platform_Diversity",
        desc="The four series represent at least three different streaming platforms, with no single platform having more than two series",
        parent=diversity_node,
        critical=True
    )

    # Genre diversity: require at least 3 different primary genre categories across the four
    genre_counts: Dict[str, int] = {}
    genres_valid = True
    for g in primary_genre_cats:
        if g is None:
            genres_valid = False
            break
        genre_counts[g] = genre_counts.get(g, 0) + 1
    unique_genres = len(genre_counts)
    genre_diversity_ok = genres_valid and unique_genres >= 3

    evaluator.add_custom_node(
        result=genre_diversity_ok,
        id="Genre_Diversity",
        desc="The four series represent at least three different primary genre categories",
        parent=diversity_node,
        critical=True
    )

    # 4) Record some helpful custom info
    evaluator.add_custom_info(
        info={
            "canonical_platforms": canonical_platforms,
            "platform_counts": platform_counts,
            "unique_platforms": unique_platforms,
            "max_platform_occurrence": max_platform_occurrence,
            "primary_genre_categories": primary_genre_cats,
            "genre_counts": genre_counts,
            "unique_genre_categories": unique_genres
        },
        info_type="diversity_analysis"
    )

    # 5) Return standardized summary
    return evaluator.get_summary()