import asyncio
import logging
from datetime import datetime
from typing import Any, List, Optional, Dict, Tuple, Set

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy

# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "streaming_series_2025_2026"
TASK_DESCRIPTION = (
    "Identify 4 scripted drama or comedy streaming original series that released new episodes or seasons "
    "between November 1, 2025 and March 31, 2026. For each series, provide: (1) The exact premiere date of the new "
    "season/episodes, (2) The streaming platform where it is available, (3) The release format (whether all episodes "
    "were released at once, released weekly, or released in multiple batches), (4) The total number of episodes in the "
    "new season, (5) Whether the series received any Emmy nominations in 2025 (specify the category if nominated), "
    "and (6) Whether the series was among the top 10 most-viewed titles on its platform during the second half of 2025 "
    "(specify viewership numbers or ranking if available). Additionally, ensure that: at least 3 different streaming "
    "platforms are represented across the 4 series, and at least one of the series used a multi-part release strategy. "
    "Provide verifiable reference URLs for each piece of information."
)

ALLOWED_PLATFORMS = {
    "Netflix",
    "Disney+",
    "Apple TV+",
    "HBO Max",  # legacy naming
    "Max",      # current naming
    "Hulu",
    "Prime Video",
    "Amazon Prime Video",  # synonym
    "Peacock",
}

TIMEFRAME_START = datetime(2025, 11, 1)
TIMEFRAME_END = datetime(2026, 3, 31)

# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class SeriesSources(BaseModel):
    premiere_urls: List[str] = Field(default_factory=list)
    platform_urls: List[str] = Field(default_factory=list)
    release_format_urls: List[str] = Field(default_factory=list)
    episode_count_urls: List[str] = Field(default_factory=list)
    genre_urls: List[str] = Field(default_factory=list)
    emmy_urls: List[str] = Field(default_factory=list)
    viewership_urls: List[str] = Field(default_factory=list)
    all_urls: List[str] = Field(default_factory=list)


class SeriesInfo(BaseModel):
    title: Optional[str] = None
    premiere_date: Optional[str] = None
    platform: Optional[str] = None
    release_format: Optional[str] = None
    episode_count: Optional[str] = None
    genre: Optional[str] = None
    emmy_nominated_2025: Optional[bool] = None
    emmy_category_2025: Optional[str] = None
    viewership_top10_h2_2025: Optional[bool] = None
    viewership_detail: Optional[str] = None
    sources: SeriesSources = Field(default_factory=SeriesSources)


class SeriesExtraction(BaseModel):
    series: List[SeriesInfo] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_series() -> str:
    return """
    Extract up to 4 scripted drama or comedy streaming original series presented in the answer that released new episodes
    or a new season between Nov 1, 2025 and Mar 31, 2026. For each series, extract the following fields exactly as stated
    in the answer:

    - title: The series title.
    - premiere_date: The exact premiere date of the new season/episodes (string, keep the format provided in the answer).
    - platform: The streaming platform (e.g., Netflix, Disney+, Apple TV+, Max/HBO Max, Hulu, Prime Video, or Peacock).
    - release_format: One of "all at once", "weekly", or "multi-part" (if the answer uses synonyms like batch releases,
      split season, two-part, multiple drops, map them to "multi-part"; if it says binge drop/day-one all episodes,
      map to "all at once"; weekly/one-per-week -> "weekly").
    - episode_count: The total number of episodes in the new season (string; if unclear in the answer, return the text the
      answer provides, or null if missing).
    - genre: The genre label reported (e.g., "drama", "comedy"); return null if missing.
    - emmy_nominated_2025: true/false/null depending on whether the answer claims 2025 Emmy nominations.
    - emmy_category_2025: If nominated, include the category text from the answer; else null.
    - viewership_top10_h2_2025: true/false/null depending on whether the answer claims the series was top 10 most-viewed on
      its platform during the second half of 2025.
    - viewership_detail: If claimed top 10, include ranking or numbers as presented in the answer; else include any detail text or null.

    Also extract verifiable reference URLs. Prefer per-field URL lists if the answer distinguishes sources; otherwise collect all:
    - sources.premiere_urls: URLs that support the premiere date and timeframe.
    - sources.platform_urls: URLs that support the platform availability.
    - sources.release_format_urls: URLs that support the release format (all at once vs weekly vs multi-part).
    - sources.episode_count_urls: URLs that support the episode count.
    - sources.genre_urls: URLs that support that it is a scripted drama or comedy (not reality/doc/sports).
    - sources.emmy_urls: URLs that support the Emmy status/nomination info (2025).
    - sources.viewership_urls: URLs that support the top 10 viewership claim in H2 2025.
    - sources.all_urls: Any additional URLs cited for this series (deduplicate; only real URLs).

    IMPORTANT:
    - Only extract what is explicitly present in the answer. Do not invent any values or URLs.
    - If a requested field is missing, set it to null.
    - For URLs, capture valid full URLs (include http/https). If the answer uses markdown links, extract the actual destination URLs.
    - Return at most 4 series. If the answer includes more than 4, keep the first 4.
    """


# --------------------------------------------------------------------------- #
# Helpers                                                                     #
# --------------------------------------------------------------------------- #
def clean_urls(urls: List[str]) -> List[str]:
    seen = set()
    cleaned: List[str] = []
    for u in urls:
        if not u or not isinstance(u, str):
            continue
        uu = u.strip()
        if not uu:
            continue
        if uu not in seen:
            seen.add(uu)
            cleaned.append(uu)
    return cleaned


def combine_all_sources(s: SeriesSources) -> List[str]:
    combined = (
        s.all_urls
        + s.premiere_urls
        + s.platform_urls
        + s.release_format_urls
        + s.episode_count_urls
        + s.genre_urls
        + s.emmy_urls
        + s.viewership_urls
    )
    return clean_urls(combined)


def pick_field_sources(s: SeriesSources, field: str) -> List[str]:
    field_map = {
        "premiere": s.premiere_urls,
        "platform": s.platform_urls,
        "format": s.release_format_urls,
        "episodes": s.episode_count_urls,
        "genre": s.genre_urls,
        "emmy": s.emmy_urls,
        "viewership": s.viewership_urls,
    }
    specific = field_map.get(field, [])
    specific = clean_urls(specific)
    if specific:
        return specific
    return combine_all_sources(s)


def canonicalize_platform(platform: Optional[str]) -> Optional[str]:
    if not platform:
        return None
    p = platform.strip().lower()
    # Normalize common synonyms and stylings
    if "netflix" in p:
        return "Netflix"
    if "disney" in p:
        return "Disney+"
    if "apple" in p and "tv" in p:
        return "Apple TV+"
    if p in ("hbo max", "max") or "hbo max" in p or p == "max":
        return "Max"
    if "hulu" in p:
        return "Hulu"
    if "prime" in p or "amazon prime" in p:
        return "Prime Video"
    if "peacock" in p:
        return "Peacock"
    return platform.strip()


def normalize_release_format(fmt: Optional[str]) -> Optional[str]:
    if not fmt:
        return None
    f = fmt.strip().lower()
    # Multi-part indicators
    multi_markers = ["multi-part", "multi part", "multi", "split season", "split", "two-part", "two part", "batches", "batch", "multiple drops", "multi-drop", "parts"]
    if any(m in f for m in multi_markers):
        return "multi-part"
    # Weekly indicators
    weekly_markers = ["weekly", "each week", "one per week", "every week", "drops weekly"]
    if any(m in f for m in weekly_markers):
        return "weekly"
    # All-at-once indicators
    binge_markers = ["all at once", "full season drop", "binge", "entire season", "whole season", "day-one all"]
    if any(m in f for m in binge_markers):
        return "all at once"
    # Fallback: if the provided value exactly matches allowed labels, keep it
    if f in ("all at once", "weekly", "multi-part"):
        return f
    return fmt.strip()


# --------------------------------------------------------------------------- #
# Verification functions                                                      #
# --------------------------------------------------------------------------- #
async def verify_one_series(
    evaluator: Evaluator,
    parent_node,
    series: SeriesInfo,
    idx: int
) -> None:
    # Create the series-level parallel node (non-critical to allow partial credit per series)
    series_id = f"series_{idx + 1}"
    series_desc_map = {
        0: "First qualifying series meets all requirements",
        1: "Second qualifying series meets all requirements",
        2: "Third qualifying series meets all requirements",
        3: "Fourth qualifying series meets all requirements",
    }
    series_node = evaluator.add_parallel(
        id=series_id,
        desc=series_desc_map.get(idx, f"Series #{idx + 1} verification"),
        parent=parent_node,
        critical=False,
    )

    # Reference URLs must be provided (critical)
    all_sources = combine_all_sources(series.sources)
    ref_node = evaluator.add_custom_node(
        result=len(all_sources) > 0,
        id=f"{series_id}_reference_urls",
        desc="Verifiable reference URLs are provided for the series information",
        parent=series_node,
        critical=True
    )

    # Helper meta for claims
    title = (series.title or f"Series #{idx + 1}").strip()

    # 1) Release date within timeframe (critical)
    rd_node = evaluator.add_leaf(
        id=f"{series_id}_release_date",
        desc="The series has a verifiable premiere date for new episodes/season between November 1, 2025 and March 31, 2026",
        parent=series_node,
        critical=True
    )
    rd_claim = (
        f"The series '{title}' released new episodes or a new season on {series.premiere_date}. "
        f"This premiere date falls between November 1, 2025 and March 31, 2026 (inclusive)."
    )
    await evaluator.verify(
        claim=rd_claim,
        node=rd_node,
        sources=pick_field_sources(series.sources, "premiere"),
        additional_instruction=(
            "Verify the exact premiere date for the new season/episodes, and confirm that the date is within the inclusive "
            "range Nov 1, 2025 to Mar 31, 2026. If the provided date is outside this range or cannot be confirmed by the "
            "sources, mark as incorrect. Be strict about the date window."
        ),
        extra_prerequisites=[ref_node],
    )

    # 2) Platform (critical)
    plat_node = evaluator.add_leaf(
        id=f"{series_id}_platform",
        desc="The streaming platform is correctly identified as one of: Netflix, Disney+, Apple TV+, HBO Max, Hulu, Prime Video, or Peacock",
        parent=series_node,
        critical=True
    )
    plat_claim = (
        f"The series '{title}' is a streaming original available on {series.platform}. "
        f"{series.platform} is one of the allowed platforms."
    )
    await evaluator.verify(
        claim=plat_claim,
        node=plat_node,
        sources=pick_field_sources(series.sources, "platform"),
        additional_instruction=(
            "Confirm platform availability for this series using the provided URLs. "
            "Allowed platforms are: Netflix, Disney+, Apple TV+, Max (HBO Max), Hulu, Prime Video (Amazon Prime Video), and Peacock. "
            "Treat 'Max' and 'HBO Max' interchangeably, and 'Amazon Prime Video' as 'Prime Video'. "
            "If the series is not on an allowed platform or the URLs do not support the claimed platform, mark incorrect."
        ),
        extra_prerequisites=[ref_node],
    )

    # 3) Release format (critical)
    fmt_node = evaluator.add_leaf(
        id=f"{series_id}_release_format",
        desc="The release format is correctly specified (all at once, weekly, or multi-part)",
        parent=series_node,
        critical=True
    )
    fmt_claim = (
        f"The new season of '{title}' used the '{series.release_format}' release format."
    )
    await evaluator.verify(
        claim=fmt_claim,
        node=fmt_node,
        sources=pick_field_sources(series.sources, "format"),
        additional_instruction=(
            "Determine the release strategy from credible sources: "
            "'all at once' means the full season was dropped on the premiere date; "
            "'weekly' means episodes were released on a weekly cadence; "
            "'multi-part' means multiple batches/drops/split season (e.g., Part 1 and Part 2). "
            "If sources conflict or do not support the claimed format, mark incorrect."
        ),
        extra_prerequisites=[ref_node],
    )

    # 4) Episode count (critical)
    ec_node = evaluator.add_leaf(
        id=f"{series_id}_episode_count",
        desc="The total number of episodes in the new season is provided",
        parent=series_node,
        critical=True
    )
    ec_claim = (
        f"The new season of '{title}' has a total of {series.episode_count} episodes."
    )
    await evaluator.verify(
        claim=ec_claim,
        node=ec_node,
        sources=pick_field_sources(series.sources, "episodes"),
        additional_instruction=(
            "Verify the total number of episodes for the new season. If season is ongoing at premiere and a source clearly "
            "states the total count, accept it. If sources do not confirm a total episode count, mark incorrect."
        ),
        extra_prerequisites=[ref_node],
    )

    # 5) Genre must be scripted drama or comedy (critical)
    gen_node = evaluator.add_leaf(
        id=f"{series_id}_genre",
        desc="The series is confirmed to be a scripted drama or comedy (not reality, documentary, or sports content)",
        parent=series_node,
        critical=True
    )
    gen_claim = (
        f"The series '{title}' is a scripted {series.genre} series (drama or comedy), not reality, documentary, or sports."
    )
    await evaluator.verify(
        claim=gen_claim,
        node=gen_node,
        sources=pick_field_sources(series.sources, "genre"),
        additional_instruction=(
            "Confirm the series is a scripted drama or scripted comedy using the sources. "
            "If the series is reality, documentary, or sports content, mark incorrect."
        ),
        extra_prerequisites=[ref_node],
    )

    # 6) Emmy status (non-critical)
    emmy_node = evaluator.add_leaf(
        id=f"{series_id}_emmy_status",
        desc="Emmy nomination status for 2025 is correctly reported (including specific category if nominated, or confirmation of no nominations)",
        parent=series_node,
        critical=False
    )
    if series.emmy_nominated_2025 is True:
        emmy_claim = (
            f"The series '{title}' received Emmy nomination(s) in 2025, including category '{series.emmy_category_2025}'."
        )
    elif series.emmy_nominated_2025 is False:
        emmy_claim = (
            f"The series '{title}' did not receive any Emmy nominations in 2025."
        )
    else:
        # Unknown claim; this will likely fail verification but we still run it
        emmy_claim = (
            f"The series '{title}' Emmy nomination status in 2025 is as claimed in the answer."
        )
    await evaluator.verify(
        claim=emmy_claim,
        node=emmy_node,
        sources=pick_field_sources(series.sources, "emmy"),
        additional_instruction=(
            "Use official Emmy listings or credible trade publications to verify the 2025 Emmy nomination status. "
            "If nominated, ensure the category matches; if not nominated, ensure sources indicate absence of nominations."
        ),
        extra_prerequisites=[ref_node],
    )

    # 7) Viewership top 10 in H2 2025 (non-critical)
    view_node = evaluator.add_leaf(
        id=f"{series_id}_viewership",
        desc="Top 10 viewership status on the platform for second half of 2025 is correctly reported (including specific ranking/numbers if applicable, or confirmation it was not in top 10)",
        parent=series_node,
        critical=False
    )
    platform_for_view = series.platform or "the platform"
    if series.viewership_top10_h2_2025 is True:
        view_claim = (
            f"The series '{title}' was among the top 10 most-viewed titles on {platform_for_view} during the second half of 2025 "
            f"(detail: {series.viewership_detail})."
        )
    elif series.viewership_top10_h2_2025 is False:
        view_claim = (
            f"The series '{title}' was not among the top 10 most-viewed titles on {platform_for_view} during the second half of 2025."
        )
    else:
        view_claim = (
            f"The series '{title}' viewership ranking on {platform_for_view} during the second half of 2025 is as claimed in the answer."
        )
    await evaluator.verify(
        claim=view_claim,
        node=view_node,
        sources=pick_field_sources(series.sources, "viewership"),
        additional_instruction=(
            "Verify H2 2025 (Jul–Dec 2025) top-10 viewership status using credible sources (platform releases, Nielsen/streamer charts, "
            "trusted trade press). If ranking or numbers are provided, confirm them; otherwise confirm the top-10 inclusion/exclusion. "
            "If sources do not substantiate the claim, mark incorrect."
        ),
        extra_prerequisites=[ref_node],
    )


# --------------------------------------------------------------------------- #
# Root-level constraint checks                                                #
# --------------------------------------------------------------------------- #
def compute_platform_diversity(series_list: List[SeriesInfo]) -> Tuple[bool, Set[str]]:
    platforms = set()
    for s in series_list[:4]:
        canon = canonicalize_platform(s.platform)
        if canon in {"Netflix", "Disney+", "Apple TV+", "Max", "Hulu", "Prime Video", "Peacock"}:
            platforms.add(canon)
    return (len(platforms) >= 3), platforms


def has_multi_part_release(series_list: List[SeriesInfo]) -> Tuple[bool, List[str]]:
    matched_titles = []
    for s in series_list[:4]:
        fmt = normalize_release_format(s.release_format)
        if fmt == "multi-part":
            matched_titles.append(s.title or "Unknown")
    return (len(matched_titles) >= 1), matched_titles


# --------------------------------------------------------------------------- #
# Main evaluation function                                                    #
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
    # Initialize evaluator (root node non-critical with parallel strategy; critical constraints added as children)
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

    # 1) Extract series information from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_series(),
        template_class=SeriesExtraction,
        extraction_name="series_extraction"
    )

    # Ensure we have exactly 4 entries (pad with empty ones if necessary; trim if more)
    series_items: List[SeriesInfo] = list(extracted.series[:4])
    while len(series_items) < 4:
        series_items.append(SeriesInfo())

    # 2) Build verification subtrees for each of the 4 series
    for i, s in enumerate(series_items):
        await verify_one_series(evaluator, root, s, i)

    # 3) Root-level critical constraints: platform diversity and multi-part release requirement
    # Platform diversity: at least 3 different platforms
    platform_ok, platforms_used = compute_platform_diversity(series_items)
    evaluator.add_custom_node(
        result=platform_ok,
        id="platform_diversity",
        desc="At least 3 different streaming platforms are represented across all 4 series",
        parent=root,
        critical=True
    )

    # Multi-part release requirement: at least one series used multi-part strategy
    multi_ok, multi_titles = has_multi_part_release(series_items)
    evaluator.add_custom_node(
        result=multi_ok,
        id="multi_part_release_requirement",
        desc="At least one of the identified series used a multi-part release strategy (episodes released in multiple batches)",
        parent=root,
        critical=True
    )

    # Record custom info to aid analysis
    evaluator.add_custom_info(
        info={"platforms_used": sorted(list(platforms_used)), "multi_part_titles": multi_titles},
        info_type="computed_constraints",
        info_name="constraint_computation"
    )

    # 4) Return standard evaluation summary
    return evaluator.get_summary()