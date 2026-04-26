import asyncio
import logging
from datetime import datetime
from typing import Optional, List, Dict, Any
from urllib.parse import urlparse

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy

# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "streaming_series_2025_2026"
TASK_DESCRIPTION = """
Identify four streaming television series that premiered (or are scheduled to premiere) between December 1, 2025, and February 28, 2026. The series must collectively be available across at least three different streaming platforms, selected from: Netflix, Apple TV+, HBO Max, Disney+, Prime Video, Peacock, Hulu, or Paramount+. For each of the four series, provide the following information: (1) The exact title of the series, (2) The streaming platform where it is available, (3) The exact premiere date (month and day), (4) The total number of episodes in the season, (5) A reference URL that verifies the series information from an official streaming platform page or a major entertainment industry source (such as Rotten Tomatoes, The Hollywood Reporter, Deadline, Variety, or IMDb). Ensure that the four series you identify are distributed across at least three different streaming platforms.
"""

ALLOWED_PLATFORMS_CANONICAL = [
    "Netflix",
    "Apple TV+",
    "HBO Max",  # Accepts "Max" synonym
    "Disney+",
    "Prime Video",
    "Peacock",
    "Hulu",
    "Paramount+",
]

# Official platform domains (accept subdomains)
ALLOWED_PLATFORM_DOMAINS = {
    "netflix.com",
    "tv.apple.com",
    "apple.com",
    "max.com",
    "hbo.com",
    "disneyplus.com",
    "disney.com",
    "primevideo.com",
    "amazon.com",
    "peacocktv.com",
    "hulu.com",
    "paramountplus.com",
}
# Major entertainment industry source domains
ALLOWED_INDUSTRY_DOMAINS = {
    "rottentomatoes.com",
    "hollywoodreporter.com",
    "deadline.com",
    "variety.com",
    "imdb.com",
}

DATE_RANGE_START = datetime(2025, 12, 1)
DATE_RANGE_END = datetime(2026, 2, 28)

# --------------------------------------------------------------------------- #
# Helper normalization and validation utilities                               #
# --------------------------------------------------------------------------- #
_PLATFORM_SYNONYMS = {
    "netflix": "Netflix",
    "apple tv+": "Apple TV+",
    "apple tv plus": "Apple TV+",
    "apple tv": "Apple TV+",
    "hbo max": "HBO Max",  # Canonicalize to "HBO Max" per rubric
    "max": "HBO Max",      # Treat "Max" as "HBO Max"
    "disney+": "Disney+",
    "disney plus": "Disney+",
    "prime video": "Prime Video",
    "amazon prime video": "Prime Video",
    "amazon video": "Prime Video",
    "amazon prime": "Prime Video",
    "peacock": "Peacock",
    "peacock tv": "Peacock",
    "peacocktv": "Peacock",
    "hulu": "Hulu",
    "paramount+": "Paramount+",
    "paramount plus": "Paramount+",
    "paramountplus": "Paramount+",
}

_GENRE_SYNONYMS = {
    "docuseries": "documentary",
    "documentary series": "documentary",
    "docs": "documentary",
    "animation": "animated",
    "anime": "animated",
    "dramedy": "comedy",
    "romcom": "comedy",
    "crime drama": "drama",
    "thriller series": "thriller",
    "limited series": "limited",
    "miniseries": "limited",
}

def normalize_platform_name(name: Optional[str]) -> Optional[str]:
    if not name:
        return None
    s = name.strip().lower()
    return _PLATFORM_SYNONYMS.get(s, name.strip())

def endswith_any_domain(host: str, domains: set) -> bool:
    host = host.lower()
    return any(host == d or host.endswith("." + d) for d in domains)

def is_allowed_reference_url(url: Optional[str]) -> bool:
    if not url or not isinstance(url, str):
        return False
    try:
        parsed = urlparse(url)
        if parsed.scheme not in ("http", "https"):
            return False
        host = parsed.netloc
        return endswith_any_domain(host, ALLOWED_PLATFORM_DOMAINS) or endswith_any_domain(host, ALLOWED_INDUSTRY_DOMAINS)
    except Exception:
        return False

def normalize_genre(g: Optional[str]) -> Optional[str]:
    if not g:
        return None
    s = g.strip().lower()
    base = _GENRE_SYNONYMS.get(s, s)
    # Keep a simple set of canonical forms
    if base in {"drama", "comedy", "thriller", "documentary", "animated", "limited"}:
        return base
    # If unknown, return the lowercased string (still counts toward diversity)
    return base

# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class SeriesItem(BaseModel):
    title: Optional[str] = None
    platform: Optional[str] = None
    premiere_date: Optional[str] = None
    episode_count: Optional[str] = None
    reference_url: Optional[str] = None
    release_format: Optional[str] = None  # "weekly", "all-at-once", "multi-part", or free text
    season_label: Optional[str] = None    # e.g., "Season 1" or "Limited series"
    genre: Optional[str] = None           # e.g., drama, thriller, comedy, documentary, animated

class SeriesExtraction(BaseModel):
    series: List[SeriesItem] = Field(default_factory=list)

# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_series() -> str:
    return """
    Extract up to FOUR streaming television series listed in the answer.
    For each series, return a JSON object with the following fields exactly:

    - title: The exact title of the series as written in the answer.
    - platform: The streaming platform as written (e.g., Netflix, Apple TV+, HBO Max/Max, Disney+, Prime Video, Peacock, Hulu, Paramount+).
    - premiere_date: The exact premiere date as given in the answer, including month and day (and year if present).
    - episode_count: The total number of episodes in the season, exactly as presented (e.g., "8", "10", "8 episodes"). If not provided, return null.
    - reference_url: A single URL that the answer cites for that series (official platform page or one of: Rotten Tomatoes, The Hollywood Reporter, Deadline, Variety, IMDb). If the answer provides multiple URLs for a series, choose the most authoritative one. If no URL is given, return null.
    - release_format: The release format if stated (e.g., "weekly", "all-at-once", "multi-part", or a descriptive phrase). If not mentioned, return null.
    - season_label: If the series is a limited series or a specific season is indicated (e.g., "Season 1"), return that label. Otherwise, return null.
    - genre: The genre or format (e.g., drama, thriller, comedy, documentary, animated, limited). If not given, return null.

    Rules:
    - Extract only what is explicitly present in the answer. Do not invent values.
    - If any field is not in the answer for a series, return null for that field.
    - If the answer lists more than four series, extract the FIRST FOUR only.
    - Return a JSON object with a single key "series" that is an array of up to four series objects.
    """

# --------------------------------------------------------------------------- #
# Series verification logic                                                   #
# --------------------------------------------------------------------------- #
async def verify_one_series(
    evaluator: Evaluator,
    parent_node,
    idx: int,
    item: SeriesItem,
) -> None:
    """
    Build verification nodes for a single series and perform verifications.
    """
    series_node = evaluator.add_parallel(
        id=f"series_{idx+1}",
        desc=f"Evaluation of the {'first' if idx==0 else ('second' if idx==1 else ('third' if idx==2 else 'fourth'))} identified series",
        parent=parent_node,
        critical=False
    )

    # Gate: required key fields presence (title, platform, premiere_date, episode_count, reference_url)
    has_required = bool(item.title and item.platform and item.premiere_date and item.episode_count and item.reference_url)
    evaluator.add_custom_node(
        result=has_required,
        id=f"series_{idx+1}_required_fields",
        desc="All required fields (title, platform, premiere date, episode count, reference URL) are present in the answer",
        parent=series_node,
        critical=True
    )

    # Gate: reference URL domain check
    evaluator.add_custom_node(
        result=is_allowed_reference_url(item.reference_url),
        id=f"series_{idx+1}_reference_url",
        desc="A reference URL is provided that verifies the series information from an official streaming platform page or major entertainment industry source",
        parent=series_node,
        critical=True
    )

    # Platform verification
    platform_leaf = evaluator.add_leaf(
        id=f"series_{idx+1}_platform",
        desc="The series is available on one of the specified streaming platforms: Netflix, Apple TV+, HBO Max, Disney+, Prime Video, Peacock, Hulu, or Paramount+",
        parent=series_node,
        critical=True
    )
    normalized_platform = normalize_platform_name(item.platform) or (item.platform or "")
    platform_claim = (
        f"The series '{item.title}' is available on the streaming platform '{normalized_platform}', "
        f"and that platform is among the allowed list: {ALLOWED_PLATFORMS_CANONICAL}."
    )
    await evaluator.verify(
        claim=platform_claim,
        node=platform_leaf,
        sources=item.reference_url,
        additional_instruction=(
            "Use the provided source page to confirm the platform association of the series. "
            "Accept 'Max' as equivalent to 'HBO Max'. If the confirmed platform is NOT one of the allowed list, mark as not supported."
        )
    )

    # Premiere date within range
    premiere_leaf = evaluator.add_leaf(
        id=f"series_{idx+1}_premiere_date",
        desc="The series premiered (or is scheduled to premiere) between December 1, 2025, and February 28, 2026",
        parent=series_node,
        critical=True
    )
    premiere_claim = (
        f"The series '{item.title}' has a premiere date '{item.premiere_date}' shown on the source page, "
        f"and that date falls within the inclusive range from {DATE_RANGE_START.strftime('%B %d, %Y')} "
        f"to {DATE_RANGE_END.strftime('%B %d, %Y')}."
    )
    await evaluator.verify(
        claim=premiere_claim,
        node=premiere_leaf,
        sources=item.reference_url,
        additional_instruction=(
            "Confirm the premiere date on the page and then check if it is within the given range. "
            "Require at minimum month and day; year may also be present. If only month/year are shown but the month/day clearly imply the range, it is acceptable. "
            "If the page shows a different or out-of-range date, mark as not supported."
        )
    )

    # Episode count verification
    episodes_leaf = evaluator.add_leaf(
        id=f"series_{idx+1}_episode_count",
        desc="A verifiable total episode count for the series season is provided",
        parent=series_node,
        critical=True
    )
    episodes_claim = (
        f"The season of '{item.title}' has a total of '{item.episode_count}' episodes, as supported by the source page."
    )
    await evaluator.verify(
        claim=episodes_claim,
        node=episodes_leaf,
        sources=item.reference_url,
        additional_instruction=(
            "Verify the total number of episodes using the page. If the page indicates TBD/unknown, or only partial info, then it is not supported."
        )
    )

    # Original content verification
    original_leaf = evaluator.add_leaf(
        id=f"series_{idx+1}_original_content",
        desc="The series is original content produced or commissioned by the streaming platform, not acquired content that previously aired elsewhere",
        parent=series_node,
        critical=True
    )
    platform_for_original = normalized_platform if normalized_platform else (item.platform or "")
    original_claim = (
        f"The series '{item.title}' is an original production by {platform_for_original} (e.g., '{platform_for_original} Original') "
        f"and is not previously aired acquisition content."
    )
    await evaluator.verify(
        claim=original_claim,
        node=original_leaf,
        sources=item.reference_url,
        additional_instruction=(
            "Confirm any 'Original' branding or language such as 'Netflix Original', 'Apple Original', 'Hulu Original', "
            "'Paramount+ Original', 'Peacock Original', 'Max Original' (HBO Max). "
            "If the source indicates that the series previously aired on another network or is acquired, mark as not supported."
        )
    )

    # Season specification verification
    season_leaf = evaluator.add_leaf(
        id=f"series_{idx+1}_season_specification",
        desc="The series is either explicitly labeled as a limited series or has a specific season number designation (e.g., Season 1, Season 2)",
        parent=series_node,
        critical=True
    )
    season_label = item.season_label or "a season designation (e.g., Season 1) or a 'Limited series' label"
    season_claim = (
        f"The source page indicates that '{item.title}' has {season_label}."
    )
    await evaluator.verify(
        claim=season_claim,
        node=season_leaf,
        sources=item.reference_url,
        additional_instruction=(
            "Accept 'Limited series', 'Miniseries', or explicit season identifiers like 'Season 1', 'Season 2'. "
            "If none of these are present on the page, mark as not supported."
        )
    )

    # Release format verification
    release_leaf = evaluator.add_leaf(
        id=f"series_{idx+1}_release_format",
        desc="The release format (all-at-once, weekly, or multi-part) is verifiable for the series",
        parent=series_node,
        critical=True
    )
    release_fmt = item.release_format or "a clear release format (weekly, all-at-once, or multi-part)"
    release_claim = (
        f"The source page indicates that '{item.title}' has {release_fmt} release format."
    )
    await evaluator.verify(
        claim=release_claim,
        node=release_leaf,
        sources=item.reference_url,
        additional_instruction=(
            "Look for scheduling patterns: all episodes dropped on one date (all-at-once), weekly releases (e.g., 'new episodes every Friday'), "
            "or multi-part (e.g., two-episode premiere followed by weekly). If unclear or absent, mark as not supported."
        )
    )

# --------------------------------------------------------------------------- #
# Root-level meta constraints                                                 #
# --------------------------------------------------------------------------- #
def compute_platform_diversity(series_items: List[SeriesItem]) -> int:
    platforms = []
    for s in series_items[:4]:
        norm = normalize_platform_name(s.platform)
        if norm:
            platforms.append(norm)
    return len(set(platforms))

def compute_genre_diversity(series_items: List[SeriesItem]) -> int:
    genres = []
    for s in series_items[:4]:
        g = normalize_genre(s.genre)
        if g:
            genres.append(g)
    return len(set(genres))

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
) -> Dict[str, Any]:
    """
    Evaluate the answer to the streaming series identification task.
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
        default_model=model
    )

    # 1) Extract up to 4 series from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_series(),
        template_class=SeriesExtraction,
        extraction_name="series_extraction"
    )

    # Ensure exactly 4 items (pad with empty if fewer)
    items: List[SeriesItem] = list(extracted.series[:4])
    while len(items) < 4:
        items.append(SeriesItem())

    # 2) Create per-series verification nodes
    for i in range(4):
        await verify_one_series(evaluator, root, i, items[i])

    # 3) Root-level critical constraints: platform diversity and genre diversity
    platform_diversity_count = compute_platform_diversity(items)
    genre_diversity_count = compute_genre_diversity(items)

    evaluator.add_custom_node(
        result=platform_diversity_count >= 3,
        id="platform_diversity",
        desc="The four identified series collectively represent at least three different streaming platforms",
        parent=root,
        critical=True
    )

    evaluator.add_custom_node(
        result=genre_diversity_count >= 2,
        id="genre_diversity",
        desc="The four identified series include at least two different genres or formats (e.g., drama, thriller, comedy, documentary, animated)",
        parent=root,
        critical=True
    )

    # 4) Add helpful custom info for transparency
    evaluator.add_custom_info(
        {
            "allowed_platforms": ALLOWED_PLATFORMS_CANONICAL,
            "allowed_platform_domains": sorted(list(ALLOWED_PLATFORM_DOMAINS)),
            "allowed_industry_domains": sorted(list(ALLOWED_INDUSTRY_DOMAINS)),
            "platform_diversity_count": platform_diversity_count,
            "genre_diversity_count": genre_diversity_count,
            "date_range": {
                "start": DATE_RANGE_START.strftime("%Y-%m-%d"),
                "end": DATE_RANGE_END.strftime("%Y-%m-%d")
            }
        },
        info_type="meta",
        info_name="evaluation_parameters"
    )

    return evaluator.get_summary()