import asyncio
import logging
from typing import Any, List, Optional, Dict, Tuple

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "tv_series_q1_2026_highly_rated_streaming"
TASK_DESCRIPTION = (
    "Identify four TV series meeting constraints for Q1 2026 (Jan 1–Mar 31, 2026): "
    "premiere timing, availability on specified US streaming platforms, high ratings "
    "(Rotten Tomatoes Tomatometer ≥ 80% OR IMDb ≥ 8.0), new/returning season status, "
    "≥ 6 episodes for the season, English availability, documented genre, "
    "documented monthly subscription price (as of March 2026), and a reference URL "
    "to the series page on Rotten Tomatoes, IMDb, or the streaming platform. "
    "Ensure the four collectively include at least one comedy, at least one drama, "
    "and at least one from any other genre."
)

ALLOWED_PLATFORMS_CANONICAL = [
    "Netflix",
    "HBO Max",       # Treat "Max" as equivalent for evaluation
    "Paramount+",
    "Disney+",
    "Hulu",
    "Prime Video",   # Treat "Amazon Prime Video" as equivalent
    "Apple TV+",
]
# Accept common synonyms/branding variants for verification and simple checks
PLATFORM_SYNONYMS = {
    "HBO Max": ["Max", "HBO MAX", "MAX"],
    "Prime Video": ["Amazon Prime Video", "Amazon Prime", "Amazon Video", "PrimeVideo"],
    "Disney+": ["Disney Plus", "Disney Plus (Disney+)", "DisneyPlus"],
    "Paramount+": ["Paramount Plus", "ParamountPlus"],
    "Apple TV+": ["Apple TV Plus", "AppleTV+", "AppleTV Plus"],
    "Netflix": ["NETFLIX"],
    "Hulu": ["HULU"],
}

DATE_WINDOW_TEXT = "between January 1, 2026 and March 31, 2026 (inclusive)"


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class SeriesItem(BaseModel):
    title: Optional[str] = None
    platform: Optional[str] = None
    premiere_date: Optional[str] = None
    genre: Optional[str] = None
    season_number: Optional[str] = None
    episode_count: Optional[str] = None
    rating_value: Optional[str] = None  # e.g., "92%", "8.4/10", "8.4"
    rating_source: Optional[str] = None  # e.g., "Rotten Tomatoes", "IMDb"
    subscription_price: Optional[str] = None  # string form, e.g., "$9.99", "$19.99 (ad-free)"
    reference_url: Optional[str] = None  # RT/IMDb/Platform series page
    rating_url: Optional[str] = None     # Prefer RT/IMDb URL for rating verification
    platform_url: Optional[str] = None   # Streaming platform's series page
    price_url: Optional[str] = None      # Platform pricing page (if provided)
    additional_urls: List[str] = Field(default_factory=list)  # Any other supporting links


class SeriesExtraction(BaseModel):
    series: List[SeriesItem] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_series_items() -> str:
    allowed_text = ", ".join(ALLOWED_PLATFORMS_CANONICAL)
    return f"""
Extract up to 6 candidate TV series entries mentioned in the answer. For each, extract the following fields exactly as stated in the answer (use strings; do not coerce to numbers):

- title: The series title.
- platform: The streaming platform name (e.g., Netflix, HBO Max/Max, Paramount+, Disney+, Hulu, Prime Video/Amazon Prime Video, Apple TV+).
- premiere_date: The premiere date (or scheduled premiere date) for the current/new season, as stated.
- genre: The genre classification as stated (e.g., drama, comedy, sci-fi thriller). Preserve multiple-genre labels if present.
- season_number: The season number (e.g., "Season 1", "S2", "2", "Season Two") if provided.
- episode_count: The number of episodes for the current/new season (planned or released), as stated (e.g., "8", "10 episodes").
- rating_value: A stated Rotten Tomatoes Tomatometer percentage (e.g., "84%") OR an IMDb rating (e.g., "8.3" or "8.3/10") for the series or current season.
- rating_source: "Rotten Tomatoes" or "IMDb" corresponding to rating_value.
- subscription_price: The stated monthly subscription price for the platform as of March 2026 (string, e.g., "$9.99", "$19.99 (ad-free)").
- reference_url: A single URL to the series page on Rotten Tomatoes, IMDb, or the streaming platform (prefer the most relevant one).
- rating_url: A URL that directly shows the rating (prefer the corresponding Rotten Tomatoes or IMDb page). If not provided, set null.
- platform_url: The streaming platform's official page for the series, if provided; otherwise null.
- price_url: A URL that documents the platform's monthly subscription price as of March 2026, if provided; otherwise null.
- additional_urls: Any other URLs mentioned that substantiate details about the series (e.g., press releases, official announcements). If none, return an empty list.

Important rules:
- Extract ONLY URLs explicitly present in the answer. Do NOT invent URLs.
- If both Rotten Tomatoes and IMDb ratings are given, choose one rating_value + rating_source and set rating_url accordingly (prefer Rotten Tomatoes if both are present).
- If any field is missing, set it to null (or empty list for additional_urls).
- If more than 6 series are mentioned, extract the first 6 in order of appearance.
- The allowed platforms for platform field are: {allowed_text}. You may still extract other names if the answer used synonyms; do not normalize here.
Return a JSON object with a 'series' array of objects with the above fields.
    """.strip()


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _collect_sources(si: SeriesItem) -> List[str]:
    urls: List[str] = []
    for u in [si.rating_url, si.reference_url, si.platform_url, si.price_url]:
        if u and isinstance(u, str) and u.strip():
            urls.append(u.strip())
    for u in si.additional_urls or []:
        if u and isinstance(u, str) and u.strip():
            urls.append(u.strip())
    # Deduplicate preserving order
    seen = set()
    unique = []
    for u in urls:
        if u not in seen:
            seen.add(u)
            unique.append(u)
    return unique


def _allowed_platforms_instruction() -> str:
    synonym_lines = []
    for canon, syns in PLATFORM_SYNONYMS.items():
        if syns:
            synonym_lines.append(f"- {canon}: also accept {', '.join(syns)}")
    return (
        "The allowed US streaming platforms are: Netflix, HBO Max, Paramount+, Disney+, Hulu, Prime Video, Apple TV+. "
        "Treat common branding variants/synonyms as equivalent. Synonym guidance:\n" +
        ("\n".join(synonym_lines) if synonym_lines else "None") +
        "\nIf the page clearly indicates the series is on one of the above platforms (or an accepted synonym), consider it valid."
    )


def _rating_instruction(si: SeriesItem) -> Tuple[str, str]:
    """
    Returns (rating_kind_text, extra_instruction) tailored for Rotten Tomatoes vs IMDb.
    """
    source = (si.rating_source or "").strip().lower()
    rating_kind = "Rotten Tomatoes Tomatometer" if "rotten" in source else ("IMDb" if "imdb" in source else "rating")
    if "rotten" in source:
        threshold_text = "Tomatometer is at least 80% (80 or higher)."
        extra = (
            "Verify the Rotten Tomatoes Tomatometer percentage on the provided page. "
            "Treat audience score as different; focus on Tomatometer (critics). "
            "Allow minor rounding differences. Pass if the Tomatometer is ≥ 80%."
        )
    elif "imdb" in source:
        threshold_text = "IMDb rating is at least 8.0 (8.0 or higher)."
        extra = (
            "Verify the IMDb user rating on the provided page. "
            "Treat 8.0 exactly as passing; allow typical formatting like '8.0/10'. "
            "If multiple ratings appear, use the main series/season rating."
        )
    else:
        threshold_text = "Rating meets the stated threshold (Tomatometer ≥ 80% or IMDb ≥ 8.0)."
        extra = (
            "If the page shows either a Rotten Tomatoes Tomatometer ≥ 80% or an IMDb rating ≥ 8.0, consider it passing. "
            "Allow minor formatting or rounding differences."
        )
    return rating_kind, f"{extra} If both sites are shown on the page, either passing rating suffices."


def _language_instruction() -> str:
    return (
        "Confirm the series is available in English, either as the original language or with English audio. "
        "If the page indicates English language, English audio, or the production is in English, consider it satisfied. "
        "If only subtitles are mentioned without English audio, do not count as English audio availability."
    )


def _premiere_timing_instruction() -> str:
    return (
        f"Confirm the cited premiere date falls {DATE_WINDOW_TEXT}. "
        "Accept official announcements or authoritative listings showing the date (including scheduled future premieres). "
        "Allow minor timezone/date format variations but ensure the calendar date falls within the window."
    )


def _episode_count_instruction() -> str:
    return (
        "Confirm the current/new season has at least 6 episodes. "
        "If not fully released yet, accept official sources that state a planned or ordered episode count of ≥ 6."
    )


def _series_status_instruction() -> str:
    return (
        "Confirm that this is either a brand new original series (Season 1) or a returning series with a new season premiering in the specified window. "
        "Use the page's text (e.g., 'Season 1', 'Season 2 premieres...', 'new season', or similar) as evidence."
    )


def _genre_instruction() -> str:
    return (
        "Confirm the series' genre as stated (e.g., comedy, drama, sci-fi, thriller). "
        "Allow multi-genre labels and common variants (e.g., 'sci-fi' vs 'science fiction')."
    )


def _subscription_price_instruction() -> str:
    return (
        "Confirm the monthly subscription price for the named platform as of March 2026. "
        "Accept official platform pricing pages or authoritative summaries reflecting that timeframe. "
        "Allow minor currency formatting differences or taxes/add-on notes if the base monthly price matches."
    )


def _reference_url_instruction() -> str:
    return (
        "Verify that the provided reference URL is a valid page about the series on Rotten Tomatoes, IMDb, or the official streaming platform. "
        "Accept query strings or anchors. Minor title formatting differences are okay if it is clearly the same series."
    )


def _norm_text(x: Optional[str]) -> str:
    return (x or "").strip()


def _is_comedy(genre_text: Optional[str]) -> bool:
    g = _norm_text(genre_text).lower()
    return "comedy" in g


def _is_drama(genre_text: Optional[str]) -> bool:
    g = _norm_text(genre_text).lower()
    return "drama" in g


def _is_other_genre(genre_text: Optional[str]) -> bool:
    g = _norm_text(genre_text).lower()
    if not g:
        return False
    # If it contains comedy or drama only, it's not "other"
    # Consider it "other" if it includes any of these common other-genre tokens
    other_tokens = [
        "thriller", "sci-fi", "science fiction", "scifi", "fantasy", "action",
        "mystery", "crime", "horror", "animation", "animated", "documentary",
        "docuseries", "adventure", "romance", "superhero", "historical",
        "biopic", "war", "western"
    ]
    if any(tok in g for tok in other_tokens):
        return True
    # Also consider "other" if there are multiple genres and none are 'comedy'/'drama'
    if "comedy" not in g and "drama" not in g and any(ch in g for ch in ["/", ",", "-", "&"]):
        return True
    return False


def _first_n_series(items: List[SeriesItem], n: int = 4) -> List[SeriesItem]:
    res = (items or [])[:n]
    while len(res) < n:
        res.append(SeriesItem())
    return res


# --------------------------------------------------------------------------- #
# Per-series verification                                                     #
# --------------------------------------------------------------------------- #
async def verify_one_series(
    evaluator: Evaluator,
    parent_node,
    si: SeriesItem,
    series_idx: int,
) -> None:
    """
    Build the verification sub-tree for a single series and execute verifications.
    All leaf nodes under a series are critical to ensure the series passes only if all constraints pass.
    """
    idx_human = series_idx + 1
    series_node = evaluator.add_parallel(
        id=f"series_{idx_human}",
        desc=f"Series #{idx_human} verification - must satisfy all specified criteria",
        parent=parent_node,
        critical=False  # Parent is soft-aggregated at root; children under are critical
    )

    # Prepare leaf nodes
    # 1) Premiere Timing
    leaf_premiere = evaluator.add_leaf(
        id=f"series_{idx_human}_premiere_timing",
        desc="Series premiered or is scheduled within Jan 1–Mar 31, 2026",
        parent=series_node,
        critical=True
    )
    # 2) Platform Availability
    leaf_platform = evaluator.add_leaf(
        id=f"series_{idx_human}_platform_availability",
        desc="Series is available on a specified major US platform",
        parent=series_node,
        critical=True
    )
    # 3) Quality Rating
    leaf_quality = evaluator.add_leaf(
        id=f"series_{idx_human}_quality_rating",
        desc="Series has high rating (Tomatometer ≥80% OR IMDb ≥8.0) with source",
        parent=series_node,
        critical=True
    )
    # 4) Series Status (new S1 or returning with new season in window)
    leaf_status = evaluator.add_leaf(
        id=f"series_{idx_human}_series_status",
        desc="Series is Season 1 or a returning series with a new season premiering in the window",
        parent=series_node,
        critical=True
    )
    # 5) Episode Count ≥ 6
    leaf_episodes = evaluator.add_leaf(
        id=f"series_{idx_human}_episode_count",
        desc="Current/new season has at least 6 episodes",
        parent=series_node,
        critical=True
    )
    # 6) Language Availability (English)
    leaf_language = evaluator.add_leaf(
        id=f"series_{idx_human}_language_availability",
        desc="Series available in English (original or English audio)",
        parent=series_node,
        critical=True
    )
    # 7) Genre documented
    leaf_genre = evaluator.add_leaf(
        id=f"series_{idx_human}_genre",
        desc="Genre is documented and supported",
        parent=series_node,
        critical=True
    )
    # 8) Subscription Price documented (as of March 2026)
    leaf_price = evaluator.add_leaf(
        id=f"series_{idx_human}_subscription_price",
        desc="Platform monthly subscription price (as of March 2026) is documented",
        parent=series_node,
        critical=True
    )
    # 9) Reference URL validity
    leaf_ref = evaluator.add_leaf(
        id=f"series_{idx_human}_reference_url",
        desc="Reference URL is a valid series page (RT/IMDb/Platform)",
        parent=series_node,
        critical=True
    )

    # Build claims and sources
    title = _norm_text(si.title) or "the series"
    platform = _norm_text(si.platform) or "the stated platform"
    premiere_date = _norm_text(si.premiere_date) or "an unspecified date"
    season_num = _norm_text(si.season_number) or "a new season"
    episode_ct = _norm_text(si.episode_count) or "an unspecified number"
    genre_text = _norm_text(si.genre) or "an unspecified genre"
    rating_val = _norm_text(si.rating_value) or "an unspecified rating"
    rating_kind, rating_extra = _rating_instruction(si)
    price_text = _norm_text(si.subscription_price) or "an unspecified monthly price"

    all_sources = _collect_sources(si)
    rating_sources = []
    if si.rating_url and si.rating_url.strip():
        rating_sources.append(si.rating_url.strip())
    # Also allow the reference/platform pages as fallback
    for u in [si.reference_url, si.platform_url]:
        if u and u.strip() and u.strip() not in rating_sources:
            rating_sources.append(u.strip())
    if not rating_sources:
        rating_sources = all_sources[:]

    # Prepare batch verifications
    batch_items: List[Tuple[str, List[str] or str or None, Any, Optional[str]]] = []

    # 1) Premiere Timing
    claim_premiere = (
        f"The series '{title}' premiered or is scheduled to premiere on {premiere_date}, "
        f"which falls {DATE_WINDOW_TEXT}."
    )
    batch_items.append((
        claim_premiere,
        all_sources if all_sources else None,
        leaf_premiere,
        _premiere_timing_instruction()
    ))

    # 2) Platform Availability (and allowed platform set)
    claim_platform = (
        f"The series '{title}' is available on {platform}, which is one of the specified major US "
        f"platforms (Netflix, HBO Max, Paramount+, Disney+, Hulu, Prime Video, or Apple TV+)."
    )
    batch_items.append((
        claim_platform,
        all_sources if all_sources else None,
        leaf_platform,
        _allowed_platforms_instruction()
    ))

    # 3) Quality Rating
    claim_quality = (
        f"The {rating_kind} for '{title}' is '{rating_val}', and it meets the required threshold "
        f"(Rotten Tomatoes ≥ 80% or IMDb ≥ 8.0)."
    )
    batch_items.append((
        claim_quality,
        rating_sources if rating_sources else None,
        leaf_quality,
        rating_extra
    ))

    # 4) Series Status
    claim_status = (
        f"The series '{title}' is either a brand new original series (Season 1) or a returning "
        f"series with a new season (noted as '{season_num}') premiering {DATE_WINDOW_TEXT}."
    )
    batch_items.append((
        claim_status,
        all_sources if all_sources else None,
        leaf_status,
        _series_status_instruction()
    ))

    # 5) Episode Count ≥ 6
    claim_episodes = (
        f"The current/new season of '{title}' has at least 6 episodes; the stated episode count is '{episode_ct}'."
    )
    batch_items.append((
        claim_episodes,
        all_sources if all_sources else None,
        leaf_episodes,
        _episode_count_instruction()
    ))

    # 6) Language Availability (English)
    claim_language = (
        f"The series '{title}' is available in English (original language or with English audio)."
    )
    batch_items.append((
        claim_language,
        all_sources if all_sources else None,
        leaf_language,
        _language_instruction()
    ))

    # 7) Genre documented
    claim_genre = (
        f"The series '{title}' is classified with the genre: '{genre_text}'."
    )
    batch_items.append((
        claim_genre,
        all_sources if all_sources else None,
        leaf_genre,
        _genre_instruction()
    ))

    # 8) Subscription Price documented (as of March 2026)
    claim_price = (
        f"As of March 2026, the monthly subscription price for {platform} is '{price_text}'."
    )
    price_sources = []
    if si.price_url and si.price_url.strip():
        price_sources.append(si.price_url.strip())
    # Also allow platform main URL and any other supporting links
    for u in [si.platform_url, si.reference_url]:
        if u and u.strip() and u.strip() not in price_sources:
            price_sources.append(u.strip())
    if not price_sources:
        price_sources = all_sources[:]

    batch_items.append((
        claim_price,
        price_sources if price_sources else None,
        leaf_price,
        _subscription_price_instruction()
    ))

    # 9) Reference URL validity (must be RT/IMDb/Platform series page)
    ref_url = si.reference_url.strip() if si.reference_url else None
    claim_ref = (
        f"The provided reference URL is a valid page about the series '{title}' on "
        f"Rotten Tomatoes, IMDb, or the official streaming platform."
    )
    batch_items.append((
        claim_ref,
        ref_url if ref_url else None,
        leaf_ref,
        _reference_url_instruction()
    ))

    # Execute verifications in parallel for this series
    await evaluator.batch_verify(batch_items)


# --------------------------------------------------------------------------- #
# Genre diversity verification                                                #
# --------------------------------------------------------------------------- #
def add_genre_diversity_checks(
    evaluator: Evaluator,
    parent_node,
    series_items: List[SeriesItem],
) -> None:
    """
    Add a critical parent node that enforces genre diversity across the four series:
    - At least one comedy
    - At least one drama
    - At least one from a genre other than comedy or drama
    """
    node = evaluator.add_parallel(
        id="genre_diversity",
        desc="The four series collectively represent genre diversity: at least one comedy, at least one drama, and at least one other genre",
        parent=parent_node,
        critical=True
    )

    has_comedy = any(_is_comedy(si.genre) for si in series_items)
    has_drama = any(_is_drama(si.genre) for si in series_items)
    has_other = any(_is_other_genre(si.genre) for si in series_items)

    evaluator.add_custom_node(
        result=has_comedy,
        id="comedy_present",
        desc="At least one of the four series is classified as a comedy",
        parent=node,
        critical=True
    )

    evaluator.add_custom_node(
        result=has_drama,
        id="drama_present",
        desc="At least one of the four series is classified as a drama",
        parent=node,
        critical=True
    )

    evaluator.add_custom_node(
        result=has_other,
        id="other_genre_present",
        desc="At least one of the four series is from a genre other than comedy or drama",
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
    model: str = "o4-mini",
) -> Dict:
    """
    Evaluate the answer for the Q1 2026 highly-rated streaming TV series task.
    Returns a standardized evaluation summary dict.
    """
    # Initialize evaluator (root is non-critical by design to allow flexible aggregation)
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Root aggregates children in parallel
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

    # Extract structured series info from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_series_items(),
        template_class=SeriesExtraction,
        extraction_name="series_extraction",
    )

    # Select exactly four series (pad with empty entries if fewer provided)
    four_series = _first_n_series(extracted.series, 4)

    # Build series nodes (non-critical children under root)
    for i, si in enumerate(four_series):
        await verify_one_series(evaluator, root, si, i)

    # Add cross-series genre diversity critical checks
    add_genre_diversity_checks(evaluator, root, four_series)

    # Record custom info for transparency
    evaluator.add_custom_info(
        info={
            "allowed_platforms": ALLOWED_PLATFORMS_CANONICAL,
            "platform_synonyms": PLATFORM_SYNONYMS,
            "premiere_window": DATE_WINDOW_TEXT,
            "rating_thresholds": {
                "Rotten Tomatoes Tomatometer": "≥ 80%",
                "IMDb": "≥ 8.0"
            }
        },
        info_type="policy",
        info_name="evaluation_policy"
    )

    return evaluator.get_summary()