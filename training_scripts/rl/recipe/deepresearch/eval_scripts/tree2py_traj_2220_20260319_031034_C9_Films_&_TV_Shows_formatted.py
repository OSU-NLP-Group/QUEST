import asyncio
import logging
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys

# --------------------------------------------------------------------------- #
# Task constants                                                              #
# --------------------------------------------------------------------------- #
TASK_ID = "emmy_drama_series_2024_2025"
TASK_DESCRIPTION = (
    "Identify four drama television series that each satisfy all of the following criteria:\n\n"
    "1. Emmy Recognition: The series was nominated for Outstanding Drama Series at either the 76th Primetime Emmy Awards (2024) or the 77th Primetime Emmy Awards (2025).\n"
    "2. Streaming Platform: The series is an exclusive original production on one of the following major streaming platforms: Netflix, HBO Max (Max), Hulu, Apple TV+, Disney+, or Amazon Prime Video.\n"
    "3. Series Specifications:\n"
    "   - The series has an IMDb rating of 8.0 or higher\n"
    "   - At least one season of the series contains between 8-10 episodes (inclusive)\n"
    "4. Production Information: The series has a documented creator or showrunner whose name is publicly credited.\n\n"
    "For each series, provide the title, the streaming platform, the creator/showrunner name, and supporting URLs "
    "(Television Academy nomination page, official platform page, IMDb rating page, episodes count page, creator/showrunner credit page). "
    "Each of the four series must be distinct."
)

ALLOWED_PLATFORMS = ["Netflix", "HBO Max", "Max", "Hulu", "Apple TV+", "Disney+", "Amazon Prime Video"]
ALLOWED_PLATFORM_DOMAINS = [
    "netflix.com",
    "max.com",
    "hbomax.com",
    "hulu.com",
    "tv.apple.com",
    "apple.com/tv",
    "apple.com/apple-tv-plus",
    "disneyplus.com",
    "primevideo.com",
    "amazon.com",
]
TV_ACADEMY_DOMAINS = ["televisionacademy.com", "emmys.com"]
IMDB_DOMAIN = "imdb.com"
RATING_THRESHOLD = 8.0
EPISODES_MIN = 8
EPISODES_MAX = 10
ALLOWED_EMMY_YEARS = [2024, 2025]
ALLOWED_EMMY_NUMBERS = [76, 77]
ALLOWED_EMMY_CATEGORY = "Outstanding Drama Series"


# --------------------------------------------------------------------------- #
# Data models                                                                 #
# --------------------------------------------------------------------------- #
class SeriesItem(BaseModel):
    title: Optional[str] = None
    platform: Optional[str] = None
    creator_or_showrunner: Optional[str] = None
    emmy_url: Optional[str] = None
    platform_url: Optional[str] = None
    imdb_url: Optional[str] = None
    episodes_url: Optional[str] = None
    creator_url: Optional[str] = None


class SeriesExtraction(BaseModel):
    series: List[SeriesItem] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_series_items() -> str:
    return """
    Extract up to the first four distinct drama television series listed in the answer, in the same order they appear.
    For each series, extract the following fields exactly as provided in the answer:
    - title: the series title
    - platform: the named streaming platform (e.g., Netflix, Max, Hulu, Apple TV+, Disney+, Amazon Prime Video). Use the wording as written in the answer.
    - creator_or_showrunner: the credited creator or showrunner name(s) as stated in the answer
    - emmy_url: the Television Academy URL that shows the series' nomination (should be from televisionacademy.com or emmys.com)
    - platform_url: the official platform URL for the series (e.g., netflix.com, max.com, hulu.com, tv.apple.com, disneyplus.com, primevideo.com, or a page under amazon.com for Prime Video)
    - imdb_url: the IMDb URL that shows the series rating (typically an imdb.com/title/... page)
    - episodes_url: a URL (IMDb or the official platform) that shows the episode counts by season
    - creator_url: a URL (IMDb or the official platform) that shows the creator/showrunner credit
    
    Rules:
    - Only extract information explicitly present in the answer. Do not invent or infer missing information.
    - If any field is missing for a series, set its value to null.
    - Ensure each extracted series is distinct by title; if duplicates occur in the answer, keep only the first occurrence and skip subsequent duplicates.
    - Return a JSON object with a single key 'series' mapping to an array of at most four objects, each with the fields above.
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _nonempty(s: Optional[str]) -> bool:
    return isinstance(s, str) and len(s.strip()) > 0


def _domain_of(url: str) -> str:
    try:
        d = urlparse(url).netloc.lower()
        # strip potential leading 'www.'
        if d.startswith("www."):
            d = d[4:]
        return d
    except Exception:
        return ""


def _url_is_from_domains(url: Optional[str], domains: List[str]) -> bool:
    if not _nonempty(url):
        return False
    host = _domain_of(url or "")
    if not host:
        return False
    for d in domains:
        d_l = d.lower().lstrip(".")
        if host == d_l or host.endswith("." + d_l) or d_l in host:
            return True
    return False


def _distinct_nonempty_titles(items: List[SeriesItem]) -> bool:
    seen = set()
    for it in items:
        if not _nonempty(it.title):
            return False  # Require all four to have titles to assert distinctness
        key = it.title.strip().lower()
        if key in seen:
            return False
        seen.add(key)
    return len(seen) == len(items)


# --------------------------------------------------------------------------- #
# Verification for one series                                                 #
# --------------------------------------------------------------------------- #
async def verify_one_series(
    evaluator: Evaluator,
    parent_node,
    item: SeriesItem,
    index_one_based: int,
) -> None:
    sid = index_one_based  # 1..4
    series_title = item.title or "the series"
    platform_name = item.platform or "the stated platform"

    # Series container (parallel; not critical to allow partial credit per series)
    series_node = evaluator.add_parallel(
        id=f"series_{sid}",
        desc=f"Series #{sid} verification",
        parent=parent_node,
        critical=False,
    )

    # Optional: title presence (non-critical informational)
    evaluator.add_custom_node(
        result=_nonempty(item.title),
        id=f"series_{sid}_title_present",
        desc=f"Series #{sid} title is provided",
        parent=series_node,
        critical=False,
    )

    # 1) Emmy nomination cluster (critical)
    emmy_node = evaluator.add_parallel(
        id=f"series_{sid}_emmy_nomination",
        desc=f"The series was nominated for Outstanding Drama Series at either the 76th (2024) or 77th (2025) Primetime Emmy Awards",
        parent=series_node,
        critical=True,
    )

    # 1.a) Emmy URL existence + domain check (critical)
    emmy_url_ok = _url_is_from_domains(item.emmy_url, TV_ACADEMY_DOMAINS)
    evaluator.add_custom_node(
        result=emmy_url_ok,
        id=f"series_{sid}_emmy_verification_url",
        desc="Provide a URL from the Television Academy (televisionacademy.com or emmys.com) showing the nomination",
        parent=emmy_node,
        critical=True,
    )

    # 1.b) Emmy nomination supported by the page (critical; gated by 1.a automatically)
    emmy_supported_leaf = evaluator.add_leaf(
        id=f"series_{sid}_emmy_supported",
        desc="Emmy nomination for Outstanding Drama Series at 76th (2024) or 77th (2025) is supported by the Television Academy page",
        parent=emmy_node,
        critical=True,
    )
    emmy_claim = (
        f"The Television Academy page confirms that '{series_title}' was nominated for '{ALLOWED_EMMY_CATEGORY}' "
        f"at either the {ALLOWED_EMMY_NUMBERS[0]}th (2024) or {ALLOWED_EMMY_NUMBERS[1]}th (2025) Primetime Emmy Awards."
    )
    await evaluator.verify(
        claim=emmy_claim,
        node=emmy_supported_leaf,
        sources=item.emmy_url,
        additional_instruction=(
            "Verify on the page that the series is listed as a nominee (or winner) specifically in the "
            f"'{ALLOWED_EMMY_CATEGORY}' category for the years {ALLOWED_EMMY_YEARS}. "
            "Accept pages that list nominees/winners for those years if the series appears under that category."
        ),
    )

    # 2) Platform/originality cluster (critical)
    platform_node = evaluator.add_parallel(
        id=f"series_{sid}_platform",
        desc="The series is an exclusive original on a major streaming platform (Netflix, HBO Max/Max, Hulu, Apple TV+, Disney+, or Amazon Prime Video)",
        parent=series_node,
        critical=True,
    )

    # 2.a) Official platform URL present (critical)
    platform_url_ok = _url_is_from_domains(item.platform_url, ALLOWED_PLATFORM_DOMAINS)
    evaluator.add_custom_node(
        result=platform_url_ok,
        id=f"series_{sid}_platform_url",
        desc="Provide the official streaming platform URL for the series",
        parent=platform_node,
        critical=True,
    )

    # 2.b) Original/exclusive status supported by platform page (critical)
    platform_original_leaf = evaluator.add_leaf(
        id=f"series_{sid}_platform_original",
        desc="The platform page indicates the series is an original/exclusive for the named platform",
        parent=platform_node,
        critical=True,
    )
    platform_claim = (
        f"The provided platform page shows that '{series_title}' is an original/exclusive series for {platform_name} "
        "(e.g., labeled as 'Netflix Series', 'Max Original', 'Hulu Original', 'Apple TV+ Original', "
        "'Disney+ Original', or 'Prime Video/Amazon Original')."
    )
    await evaluator.verify(
        claim=platform_claim,
        node=platform_original_leaf,
        sources=item.platform_url,
        additional_instruction=(
            "Check for labels like 'Netflix Series', 'A Netflix Series', 'Max Original', 'Hulu Original', "
            "'Apple TV+ Original', 'Disney+ Original', 'Amazon Original' or 'Prime Video Original'. "
            "If the platform's official page clearly presents the series as its own original, consider this supported."
        ),
    )

    # 3) Specifications cluster (critical)
    specs_node = evaluator.add_parallel(
        id=f"series_{sid}_specifications",
        desc="Series specifications meet the defined criteria",
        parent=series_node,
        critical=True,
    )

    # 3.a) IMDb rating >= 8.0
    imdb_rating_parent = evaluator.add_parallel(
        id=f"series_{sid}_imdb_rating_parent",
        desc="IMDb rating check cluster",
        parent=specs_node,
        critical=True,
    )

    imdb_url_ok = _url_is_from_domains(item.imdb_url, [IMDB_DOMAIN])
    evaluator.add_custom_node(
        result=imdb_url_ok,
        id=f"series_{sid}_imdb_url",
        desc="Provide the IMDb URL showing the series rating",
        parent=imdb_rating_parent,
        critical=True,
    )

    imdb_rating_leaf = evaluator.add_leaf(
        id=f"series_{sid}_imdb_rating",
        desc=f"The series has an IMDb rating of {RATING_THRESHOLD} or higher",
        parent=imdb_rating_parent,
        critical=True,
    )
    imdb_claim = (
        f"The IMDb page for '{series_title}' shows an aggregate user rating of at least {RATING_THRESHOLD:.1f} out of 10."
    )
    await evaluator.verify(
        claim=imdb_claim,
        node=imdb_rating_leaf,
        sources=item.imdb_url,
        additional_instruction=(
            f"Use the rating displayed on the IMDb title page. Consider {RATING_THRESHOLD:.1f} or higher as a pass; "
            "reasonable rounding is acceptable."
        ),
    )

    # 3.b) Episode structure: at least one season has 8-10 episodes inclusive
    episodes_parent = evaluator.add_parallel(
        id=f"series_{sid}_episode_structure_parent",
        desc="Episode count per season check cluster",
        parent=specs_node,
        critical=True,
    )

    episodes_url_ok = _url_is_from_domains(item.episodes_url, [IMDB_DOMAIN] + ALLOWED_PLATFORM_DOMAINS)
    evaluator.add_custom_node(
        result=episodes_url_ok,
        id=f"series_{sid}_episode_verification_url",
        desc="Provide a URL (IMDb or official platform) showing episode counts by season",
        parent=episodes_parent,
        critical=True,
    )

    episodes_leaf = evaluator.add_leaf(
        id=f"series_{sid}_episode_structure",
        desc=f"At least one season of the series contains between {EPISODES_MIN}-{EPISODES_MAX} episodes",
        parent=episodes_parent,
        critical=True,
    )
    episodes_claim = (
        f"At least one season of '{series_title}' has between {EPISODES_MIN} and {EPISODES_MAX} episodes inclusive."
    )
    await evaluator.verify(
        claim=episodes_claim,
        node=episodes_leaf,
        sources=item.episodes_url,
        additional_instruction=(
            "Check the season-by-season episode counts (on IMDb 'Episodes' tab/pages or the official platform page). "
            f"Pass if any one season has {EPISODES_MIN}–{EPISODES_MAX} episodes inclusive."
        ),
    )

    # 4) Creator/showrunner cluster (critical)
    creator_node = evaluator.add_parallel(
        id=f"series_{sid}_creator",
        desc="The series has a documented creator/showrunner whose name is publicly available",
        parent=series_node,
        critical=True,
    )

    creator_url_ok = _url_is_from_domains(item.creator_url, [IMDB_DOMAIN] + ALLOWED_PLATFORM_DOMAINS)
    evaluator.add_custom_node(
        result=creator_url_ok,
        id=f"series_{sid}_creator_url",
        desc="Provide a URL showing the creator/showrunner credit (from IMDb or official platform)",
        parent=creator_node,
        critical=True,
    )

    creator_credit_leaf = evaluator.add_leaf(
        id=f"series_{sid}_creator_credit",
        desc="The provided creator/showrunner is credited on the given page",
        parent=creator_node,
        critical=True,
    )
    creator_name = item.creator_or_showrunner or "the stated creator/showrunner"
    creator_claim = (
        f"The page credits {creator_name} as a creator, co-creator, showrunner, or developed-by credit for '{series_title}'."
    )
    await evaluator.verify(
        claim=creator_claim,
        node=creator_credit_leaf,
        sources=item.creator_url,
        additional_instruction=(
            "Accept common credit labels such as 'Created by', 'Co-Created by', 'Developed by', 'Showrunner', or "
            "'Executive Producer/Showrunner' in the context of showrunning. Minor name variations are acceptable."
        ),
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
    # Initialize evaluator with a parallel root (non-critical to allow partial credit overall)
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

    # Record constraints as ground truth/context
    evaluator.add_ground_truth(
        {
            "allowed_platforms": ALLOWED_PLATFORMS,
            "allowed_platform_domains": ALLOWED_PLATFORM_DOMAINS,
            "emmy_category": ALLOWED_EMMY_CATEGORY,
            "emmy_years": ALLOWED_EMMY_YEARS,
            "emmy_numbers": ALLOWED_EMMY_NUMBERS,
            "rating_threshold": RATING_THRESHOLD,
            "episodes_range_inclusive": [EPISODES_MIN, EPISODES_MAX],
            "television_academy_domains": TV_ACADEMY_DOMAINS,
            "imdb_domain": IMDB_DOMAIN,
        },
        gt_type="constraints",
    )

    # Extract up to four series from the answer
    extracted: SeriesExtraction = await evaluator.extract(
        prompt=prompt_extract_series_items(),
        template_class=SeriesExtraction,
        extraction_name="series_extraction",
    )

    # Normalize to exactly four items (truncate or pad with empty)
    series_items: List[SeriesItem] = list(extracted.series[:4])
    while len(series_items) < 4:
        series_items.append(SeriesItem())

    # Distinctness check across the four series (critical for the whole task)
    distinct_node = evaluator.add_custom_node(
        result=_distinct_nonempty_titles(series_items),
        id="all_series_distinct",
        desc="All four series have non-empty, distinct titles",
        parent=root,
        critical=True,
    )

    # Build per-series verification subtrees
    for idx, item in enumerate(series_items, start=1):
        await verify_one_series(evaluator, root, item, idx)

    # Return standardized summary
    return evaluator.get_summary()