import asyncio
import logging
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy

# --------------------------------------------------------------------------- #
# Task constants                                                              #
# --------------------------------------------------------------------------- #
TASK_ID = "streaming_series_2024"
TASK_DESCRIPTION = (
    "Find four distinct streaming TV series that premiered or were released in 2024 and meet all of the following criteria: "
    "(1) The series must consist of exactly 8 episodes, "
    "(2) The series must have an IMDb rating of 8.0 or higher, "
    "(3) The series must have been filmed primarily in the United States, "
    "(4) The series must be available on at least one of these major streaming platforms: Netflix, Hulu, Max (HBO Max), Prime Video, Apple TV+, or Disney+, "
    "(5) The series must have between 6-10 series regular cast members. "
    "For each of the four series, provide: the official series title, confirmation it premiered/was released in 2024, the exact number of episodes (must be 8), "
    "the current IMDb rating (must be 8.0 or higher), the primary filming location(s) within the United States, the streaming platform(s) where it is available, "
    "the number of series regular cast members (must be 6-10), and reference URLs that verify each piece of information."
)

ALLOWED_PLATFORMS = ["Netflix", "Hulu", "Max", "HBO Max", "Prime Video", "Amazon Prime Video", "Apple TV+", "Disney+"]


# --------------------------------------------------------------------------- #
# Data models                                                                 #
# --------------------------------------------------------------------------- #
class SeriesItem(BaseModel):
    title: Optional[str] = None
    release_year_text: Optional[str] = None
    episode_count_text: Optional[str] = None
    imdb_rating_text: Optional[str] = None
    filming_locations: List[str] = Field(default_factory=list)
    streaming_platforms: List[str] = Field(default_factory=list)
    cast_regulars_count_text: Optional[str] = None

    imdb_url: Optional[str] = None
    platform_urls: List[str] = Field(default_factory=list)
    other_urls: List[str] = Field(default_factory=list)


class SeriesExtraction(BaseModel):
    series: List[SeriesItem] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_series() -> str:
    return """
    Extract up to FOUR streaming TV series from the answer. Return a JSON object with a 'series' array (max length 4).
    For each series, extract the following fields exactly as written in the answer (use strings for numbers; leave null if missing):
    - title: Official series title.
    - release_year_text: Text indicating the premiere/release year (e.g., "2024", "Premiered in April 2024").
    - episode_count_text: Text indicating the total number of episodes (e.g., "8 episodes").
    - imdb_rating_text: Text indicating the IMDb rating (e.g., "8.2/10", "8.0").
    - filming_locations: List of location strings indicating primary filming locations in the United States (e.g., ["Los Angeles, California", "Atlanta, Georgia"]). If only "United States" is mentioned, include that.
    - streaming_platforms: List of platform names mentioned (e.g., ["Prime Video", "Netflix"]).
    - cast_regulars_count_text: Text indicating the number of series regular cast members (e.g., "8 series regulars", "main cast of 7").
    - imdb_url: The URL to the IMDb title page for the series (if provided).
    - platform_urls: List of URLs to official platform pages where the show is available (e.g., Netflix, Hulu, Max, Prime Video, Apple TV+, Disney+). Include all that are provided.
    - other_urls: List of any other reference URLs cited for this series (e.g., Wikipedia, official site, press articles).

    Rules:
    - Do not invent information. Only extract what is explicitly in the answer text.
    - Include all URLs provided in the answer for each series in the appropriate fields.
    - If more than four series are mentioned, include only the first four as they appear.
    - If a field is missing in the answer for a series, set it to null or an empty list as appropriate.
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def ordinal(n: int) -> str:
    return ["First", "Second", "Third", "Fourth"][n] if 0 <= n < 4 else f"#{n+1}"


def collect_sources(item: SeriesItem) -> List[str]:
    urls: List[str] = []
    if item.imdb_url:
        urls.append(item.imdb_url)
    urls.extend(item.platform_urls or [])
    urls.extend(item.other_urls or [])
    # Deduplicate while preserving order
    seen = set()
    deduped = []
    for u in urls:
        if not u:
            continue
        if u not in seen:
            seen.add(u)
            deduped.append(u)
    return deduped


def has_any_url(item: SeriesItem) -> bool:
    return bool(
        (item.imdb_url and item.imdb_url.strip())
        or (item.platform_urls and any(u.strip() for u in item.platform_urls))
        or (item.other_urls and any(u.strip() for u in item.other_urls))
    )


def build_additional_instruction(base: str, has_sources: bool) -> str:
    suffix = " If no valid URL sources are provided or the webpages are irrelevant/inaccessible, you must mark the claim as not supported."
    return (base + suffix) if not has_sources else base


def join_list_readable(items: List[str]) -> str:
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    return ", ".join(items[:-1]) + " and " + items[-1]


# --------------------------------------------------------------------------- #
# Verification for one series                                                 #
# --------------------------------------------------------------------------- #
async def verify_one_series(
    evaluator: Evaluator,
    parent_node,
    item: SeriesItem,
    index: int,
) -> None:
    # Create series node (parallel as per rubric)
    series_node = evaluator.add_parallel(
        id=f"series_{index+1}",
        desc=f"{ordinal(index)} qualifying series meeting all constraints",
        parent=parent_node,
        critical=False
    )

    sources_all = collect_sources(item)
    has_sources_flag = len(sources_all) > 0

    # Leaf: Title (Critical)
    title_leaf = evaluator.add_leaf(
        id=f"series_{index+1}_title",
        desc=f"Provide the official title of the {ordinal(index).lower()} series",
        parent=series_node,
        critical=True
    )
    title_val = item.title or ""
    title_claim = f"This page shows the official title of the TV series as '{title_val}'. Allow minor formatting or punctuation differences."
    await evaluator.verify(
        claim=title_claim,
        node=title_leaf,
        sources=item.imdb_url or sources_all,
        additional_instruction=build_additional_instruction(
            "Verify the primary title displayed on the page matches the claimed title (case-insensitive, allow small stylistic variations).",
            bool(item.imdb_url or sources_all),
        ),
    )

    # Leaf: Release Year 2024 (Critical)
    release_leaf = evaluator.add_leaf(
        id=f"series_{index+1}_release_year",
        desc="Verify the series premiered or was released in 2024",
        parent=series_node,
        critical=True
    )
    release_claim = "This series premiered (or was first released) in the calendar year 2024."
    await evaluator.verify(
        claim=release_claim,
        node=release_leaf,
        sources=item.imdb_url or sources_all,
        additional_instruction=build_additional_instruction(
            "Accept phrases like '2024–', 'Premiered in 2024', or a first-release date in 2024 as valid evidence.",
            bool(item.imdb_url or sources_all),
        ),
    )

    # Leaf: Episode Count = 8 (Critical)
    episodes_leaf = evaluator.add_leaf(
        id=f"series_{index+1}_episode_count",
        desc="Verify the series has exactly 8 episodes",
        parent=series_node,
        critical=True
    )
    episodes_claim = "This series has exactly 8 total episodes."
    await evaluator.verify(
        claim=episodes_claim,
        node=episodes_leaf,
        sources=item.imdb_url or sources_all,
        additional_instruction=build_additional_instruction(
            "Look for 'Episodes' count or season/episode listings showing a total of 8. Minor formatting differences are okay; the total must be 8.",
            bool(item.imdb_url or sources_all),
        ),
    )

    # Leaf: IMDb rating >= 8.0 (Critical)
    rating_leaf = evaluator.add_leaf(
        id=f"series_{index+1}_imdb_rating",
        desc="Verify the series has an IMDb rating of 8.0 or higher",
        parent=series_node,
        critical=True
    )
    rating_claim = "The current IMDb user rating for this series is at least 8.0 out of 10."
    await evaluator.verify(
        claim=rating_claim,
        node=rating_leaf,
        sources=item.imdb_url or sources_all,
        additional_instruction=build_additional_instruction(
            "Check the IMDb title page rating. Allow rounding (e.g., 8.0 qualifies; 7.9 does not).",
            bool(item.imdb_url or sources_all),
        ),
    )

    # Leaf: Filmed primarily in the United States (Critical)
    filming_leaf = evaluator.add_leaf(
        id=f"series_{index+1}_filming_location",
        desc="Verify the series was filmed primarily in the United States and provide specific location(s)",
        parent=series_node,
        critical=True
    )
    locs_str = join_list_readable(item.filming_locations)
    filming_claim = (
        "This series was primarily filmed in the United States."
        + (f" Specific locations mentioned include: {locs_str}." if locs_str else "")
    )
    await evaluator.verify(
        claim=filming_claim,
        node=filming_leaf,
        sources=sources_all,
        additional_instruction=build_additional_instruction(
            "Accept evidence such as 'Country of origin: United States' or a Filming locations section dominated by U.S. locations. "
            "If non-U.S. locations dominate, the claim should be rejected.",
            has_sources_flag,
        ),
    )

    # Leaf: Streaming platform availability on major services (Critical)
    platform_leaf = evaluator.add_leaf(
        id=f"series_{index+1}_streaming_platform",
        desc="Verify the series is available on at least one major streaming platform (Netflix, Hulu, Max, Prime Video, Apple TV+, or Disney+)",
        parent=series_node,
        critical=True
    )
    claimed_platforms = join_list_readable(item.streaming_platforms) if item.streaming_platforms else "one major platform"
    platform_claim = (
        f"The series is available on at least one of these major platforms: Netflix, Hulu, Max (HBO Max), Prime Video, Apple TV+, or Disney+. "
        f"According to the provided pages, it is available on {claimed_platforms}."
    )
    await evaluator.verify(
        claim=platform_claim,
        node=platform_leaf,
        sources=(item.platform_urls or sources_all),
        additional_instruction=build_additional_instruction(
            "Prefer checking official platform pages (e.g., netflix.com, hulu.com, max.com, primevideo.com, tv.apple.com, disneyplus.com). "
            "If a provided page confirms availability on one of these, accept.",
            bool(item.platform_urls or sources_all),
        ),
    )

    # Leaf: Cast size between 6–10 series regulars (Critical)
    cast_leaf = evaluator.add_leaf(
        id=f"series_{index+1}_cast_size",
        desc="Verify the series has between 6-10 series regular cast members",
        parent=series_node,
        critical=True
    )
    cast_claim = (
        "The series has between 6 and 10 series-regular (main) cast members. "
        "Count 'Starring' or 'Main cast' as series regulars when clearly indicated."
    )
    await evaluator.verify(
        claim=cast_claim,
        node=cast_leaf,
        sources=sources_all,
        additional_instruction=build_additional_instruction(
            "Use sections labeled 'Starring', 'Main cast', or 'Series regulars'. If multiple sources disagree, prefer official or IMDb/Wikipedia pages with clear labeling.",
            has_sources_flag,
        ),
    )

    # Leaf: URL references provided (Critical)
    refs_leaf = evaluator.add_custom_node(
        result=has_sources_flag,
        id=f"series_{index+1}_url_references",
        desc="Provide reference URLs supporting the information for this series",
        parent=series_node,
        critical=True
    )
    # refs_leaf is a custom node and already scored; nothing to verify here.


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

    # Extract up to 4 series from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_series(),
        template_class=SeriesExtraction,
        extraction_name="series_extraction"
    )

    # Normalize to exactly 4 items (pad with empty if fewer)
    series_list: List[SeriesItem] = (extracted.series or [])[:4]
    while len(series_list) < 4:
        series_list.append(SeriesItem())

    # Build per-series nodes and verifications (parallel under root)
    tasks = []
    for idx in range(4):
        tasks.append(verify_one_series(evaluator, root, series_list[idx], idx))
    for t in tasks:
        await t

    return evaluator.get_summary()