import asyncio
import logging
import re
import unicodedata
from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys


# --------------------------------------------------------------------------- #
# Task metadata                                                               #
# --------------------------------------------------------------------------- #
TASK_ID = "emmy_2024_outstanding_drama_series"
TASK_DESCRIPTION = (
    "What television series won the 2024 Emmy Award for Outstanding Drama Series? "
    "Provide the following information about this series: the series title, the total number of episodes in its first season, "
    "the primary streaming platform where it is available in the United States, the production company that produced it, "
    "the television network that originally aired it, the exact premiere date of its first season, and the total number of Emmy Awards it won in 2024."
)

# --------------------------------------------------------------------------- #
# Expected ground-truth (used for deterministic checks)                       #
# --------------------------------------------------------------------------- #
EXPECTED_TITLE = "Shōgun"
EXPECTED_EPISODES_S1 = 10
EXPECTED_STREAMING_PLATFORM_US = "FX on Hulu"
EXPECTED_PRODUCTION_COMPANY = "FX Productions"
EXPECTED_ORIGINAL_NETWORK = "FX"
EXPECTED_PREMIERE_DATE = datetime(2024, 2, 27)
EXPECTED_TOTAL_EMMY_WINS_2024 = 18


# --------------------------------------------------------------------------- #
# Data models                                                                 #
# --------------------------------------------------------------------------- #
class SeriesExtraction(BaseModel):
    series_title: Optional[str] = None
    first_season_episode_count: Optional[str] = None
    streaming_platform_us: Optional[str] = None
    production_company: Optional[str] = None
    original_network: Optional[str] = None
    premiere_date_first_season: Optional[str] = None
    total_emmy_wins_2024: Optional[str] = None
    source_urls: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_series_info() -> str:
    return """
    Extract the requested fields about the TV series that the answer claims won the 2024 Emmy Award for Outstanding Drama Series.

    Return a JSON object with the following fields:
    - series_title: The series title exactly as written in the answer.
    - first_season_episode_count: The first season's total episode count exactly as written (raw text, do not transform; e.g., "10", "10 episodes", "ten").
    - streaming_platform_us: The primary streaming platform in the United States exactly as written (e.g., "FX on Hulu", "Hulu").
    - production_company: The production company exactly as written (e.g., "FX Productions").
    - original_network: The original airing network exactly as written (e.g., "FX").
    - premiere_date_first_season: The premiere date for the first season exactly as written (keep raw textual date, e.g., "February 27, 2024" or "2024-02-27" or "Feb. 27, 2024").
    - total_emmy_wins_2024: The total number of Emmy Awards the series won in 2024 exactly as written (raw text, e.g., "18", "18 awards").
    - source_urls: An array capturing all explicit URLs present in the answer that appear to support any of the above claims. 
                   Only include URLs that actually appear in the answer (plain URLs or markdown links). If none, return an empty array.

    Rules:
    1) Do not invent or infer information not explicitly present in the answer text.
    2) If a field is missing in the answer, return null for that field.
    3) For source_urls, only include valid URLs explicitly present in the answer text; if the answer mentions a source without a URL, do not include it.
    """


# --------------------------------------------------------------------------- #
# Helper normalization and parsing utilities                                  #
# --------------------------------------------------------------------------- #
def _is_nonempty(text: Optional[str]) -> bool:
    return bool(text and str(text).strip())


def _strip_accents(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn")


def _norm_text(s: str) -> str:
    s = s.strip().lower()
    s = " ".join(s.split())
    return s


def _alpha_num_space(s: str) -> str:
    return re.sub(r"[^a-z0-9\s]+", " ", s.lower()).strip()


def _title_matches_expected(title: Optional[str]) -> bool:
    if not _is_nonempty(title):
        return False
    t = _strip_accents(title or "")
    t = re.sub(r"\(.*?\)", "", t, flags=re.IGNORECASE)  # remove parenthetical suffixes like (TV series)
    t = _alpha_num_space(t)
    t = " ".join(t.split())
    return t == _alpha_num_space(_strip_accents(EXPECTED_TITLE))


def _parse_int_from_text(text: Optional[str]) -> Optional[int]:
    if not _is_nonempty(text):
        return None
    m = re.search(r"\d+", text)
    if not m:
        return None
    try:
        return int(m.group(0))
    except Exception:
        return None


def _platform_matches_fx_on_hulu(platform: Optional[str]) -> bool:
    if not _is_nonempty(platform):
        return False
    p = _alpha_num_space(platform or "")
    # Accept either exact phrase or clear mention of both FX and Hulu
    return ("fx on hulu" in p) or ("fx" in p and "hulu" in p)


def _production_company_matches_fx_productions(pc: Optional[str]) -> bool:
    if not _is_nonempty(pc):
        return False
    p = _alpha_num_space(pc or "")
    return "fx productions" in p or p == "fxp"


def _original_network_matches_fx(net: Optional[str]) -> bool:
    if not _is_nonempty(net):
        return False
    n = _alpha_num_space(net or "")
    return n == "fx" or "fx network" in n


def _normalize_month_abbrev(s: str) -> str:
    # Handle common dotted abbreviations like "Feb." -> "Feb"
    return re.sub(r"\b(Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)\.\b", r"\1", s)


def _parse_date(text: Optional[str]) -> Optional[datetime]:
    if not _is_nonempty(text):
        return None
    raw = _normalize_month_abbrev(text.strip())
    fmts = [
        "%B %d, %Y",   # February 27, 2024
        "%b %d, %Y",   # Feb 27, 2024
        "%Y-%m-%d",    # 2024-02-27
        "%m/%d/%Y",    # 02/27/2024
        "%d %B %Y",    # 27 February 2024
        "%d %b %Y",    # 27 Feb 2024
    ]
    for fmt in fmts:
        try:
            return datetime.strptime(raw, fmt)
        except Exception:
            continue
    return None


def _premiere_date_matches_expected(dtext: Optional[str]) -> bool:
    d = _parse_date(dtext)
    if not d:
        return False
    return d.date() == EXPECTED_PREMIERE_DATE.date()


def _wins_matches_expected(wtext: Optional[str]) -> bool:
    wins = _parse_int_from_text(wtext)
    return wins == EXPECTED_TOTAL_EMMY_WINS_2024


# --------------------------------------------------------------------------- #
# Verification tree construction                                              #
# --------------------------------------------------------------------------- #
async def build_verification_tree(evaluator: Evaluator, extracted: SeriesExtraction) -> None:
    # Add a critical parent node that mirrors the rubric's top-level requirement
    parent = evaluator.add_parallel(
        id="Emmy_Winner_Information",
        desc="Answer identifies the 2024 Emmy Award winner for Outstanding Drama Series and provides all required constrained details.",
        parent=evaluator.root,
        critical=True
    )

    # Ground truth info for transparency
    evaluator.add_ground_truth({
        "expected_series_title": EXPECTED_TITLE,
        "expected_first_season_episode_count": EXPECTED_EPISODES_S1,
        "expected_streaming_platform_us": EXPECTED_STREAMING_PLATFORM_US,
        "expected_production_company": EXPECTED_PRODUCTION_COMPANY,
        "expected_original_network": EXPECTED_ORIGINAL_NETWORK,
        "expected_premiere_date": "February 27, 2024",
        "expected_total_emmy_wins_2024": EXPECTED_TOTAL_EMMY_WINS_2024
    })

    # Helper for sources handling
    sources_list = extracted.source_urls if extracted.source_urls else []
    winner_sources = sources_list if len(sources_list) > 0 else None

    # 1) Winning Series Title (split into existence + title match + winner claim support)
    title_node = evaluator.add_sequential(
        id="Winning_Series_Title",
        desc="Provides the series title, and the named series is the winner of the 2024 Emmy Award for Outstanding Drama Series.",
        parent=parent,
        critical=True
    )
    evaluator.add_custom_node(
        result=_is_nonempty(extracted.series_title),
        id="title_provided",
        desc="Series title is provided in the answer.",
        parent=title_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=_title_matches_expected(extracted.series_title),
        id="title_matches_expected",
        desc=f"The provided series title matches the expected winner '{EXPECTED_TITLE}'.",
        parent=title_node,
        critical=True
    )
    # Winner claim supported (prefer sources if any)
    title_winner_leaf = evaluator.add_leaf(
        id="title_is_winner_2024",
        desc=f"The series '{EXPECTED_TITLE}' is the winner of the 2024 Emmy Award for Outstanding Drama Series (supported by cited sources if available).",
        parent=title_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The television series '{EXPECTED_TITLE}' won the 2024 Emmy Award for Outstanding Drama Series.",
        node=title_winner_leaf,
        sources=winner_sources,
        additional_instruction=(
            "If sources are provided, confirm the page clearly states that Shōgun won the 2024 Emmy Award for Outstanding Drama Series. "
            "Look for explicit phrasing like 'Outstanding Drama Series: Shōgun' or equivalent. "
            "If no sources are available, judge based on the provided answer content."
        ),
    )

    # 2) Episode Count (existence + equals 10)
    ep_node = evaluator.add_sequential(
        id="Episode_Count_First_Season",
        desc="States the first-season episode count and it is exactly 10 episodes.",
        parent=parent,
        critical=True
    )
    evaluator.add_custom_node(
        result=_is_nonempty(extracted.first_season_episode_count),
        id="episodes_provided",
        desc="First-season episode count is provided.",
        parent=ep_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=_parse_int_from_text(extracted.first_season_episode_count) == EXPECTED_EPISODES_S1,
        id="episodes_value_correct",
        desc=f"First-season episode count equals {EXPECTED_EPISODES_S1}.",
        parent=ep_node,
        critical=True
    )

    # 3) Streaming Platform US (existence + matches FX on Hulu)
    platform_node = evaluator.add_sequential(
        id="Streaming_Platform_US",
        desc="States the primary US streaming platform and it is FX on Hulu.",
        parent=parent,
        critical=True
    )
    evaluator.add_custom_node(
        result=_is_nonempty(extracted.streaming_platform_us),
        id="platform_provided",
        desc="Primary US streaming platform is provided.",
        parent=platform_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=_platform_matches_fx_on_hulu(extracted.streaming_platform_us),
        id="platform_value_correct",
        desc=f"Primary US streaming platform matches '{EXPECTED_STREAMING_PLATFORM_US}'.",
        parent=platform_node,
        critical=True
    )

    # 4) Production Company (existence + matches FX Productions)
    prod_node = evaluator.add_sequential(
        id="Production_Company",
        desc="States the production company and it is FX Productions.",
        parent=parent,
        critical=True
    )
    evaluator.add_custom_node(
        result=_is_nonempty(extracted.production_company),
        id="production_company_provided",
        desc="Production company is provided.",
        parent=prod_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=_production_company_matches_fx_productions(extracted.production_company),
        id="production_company_value_correct",
        desc=f"Production company matches '{EXPECTED_PRODUCTION_COMPANY}'.",
        parent=prod_node,
        critical=True
    )

    # 5) Original Network (existence + matches FX)
    net_node = evaluator.add_sequential(
        id="Original_Network",
        desc="States the original airing network and it is FX.",
        parent=parent,
        critical=True
    )
    evaluator.add_custom_node(
        result=_is_nonempty(extracted.original_network),
        id="original_network_provided",
        desc="Original airing network is provided.",
        parent=net_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=_original_network_matches_fx(extracted.original_network),
        id="original_network_value_correct",
        desc=f"Original network matches '{EXPECTED_ORIGINAL_NETWORK}'.",
        parent=net_node,
        critical=True
    )

    # 6) Premiere Date (existence + exact date match)
    date_node = evaluator.add_sequential(
        id="Premiere_Date_First_Season",
        desc="States the exact premiere date of the first season and it is February 27, 2024.",
        parent=parent,
        critical=True
    )
    evaluator.add_custom_node(
        result=_is_nonempty(extracted.premiere_date_first_season),
        id="premiere_date_provided",
        desc="Premiere date for the first season is provided.",
        parent=date_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=_premiere_date_matches_expected(extracted.premiere_date_first_season),
        id="premiere_date_value_correct",
        desc="Premiere date equals February 27, 2024.",
        parent=date_node,
        critical=True
    )

    # 7) Total Emmy Wins 2024 (existence + equals 18)
    wins_node = evaluator.add_sequential(
        id="Total_Emmy_Wins_2024",
        desc="States the total number of Emmy Awards the series won in 2024 and it is 18.",
        parent=parent,
        critical=True
    )
    evaluator.add_custom_node(
        result=_is_nonempty(extracted.total_emmy_wins_2024),
        id="wins_2024_provided",
        desc="Total number of Emmy Awards won in 2024 is provided.",
        parent=wins_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=_wins_matches_expected(extracted.total_emmy_wins_2024),
        id="wins_2024_value_correct",
        desc=f"Total number of Emmy Awards won in 2024 equals {EXPECTED_TOTAL_EMMY_WINS_2024}.",
        parent=wins_node,
        critical=True
    )


# --------------------------------------------------------------------------- #
# Main evaluation entry                                                       #
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
    # Initialize evaluator (root is non-critical; we will add a critical child for the main rubric node)
    evaluator = Evaluator()
    evaluator.initialize(
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

    # 1) Extract structured info from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_series_info(),
        template_class=SeriesExtraction,
        extraction_name="series_info",
    )

    # 2) Build verification tree and run checks
    await build_verification_tree(evaluator, extracted)

    # 3) Return summary
    return evaluator.get_summary()