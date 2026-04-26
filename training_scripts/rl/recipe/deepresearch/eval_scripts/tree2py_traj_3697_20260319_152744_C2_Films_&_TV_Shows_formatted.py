import asyncio
import logging
from typing import Any, List, Optional, Dict

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys

# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "abc_fall_2025_premieres_and_orders"
TASK_DESCRIPTION = (
    "I want to catch up on two long-running ABC comedy and drama series that returned with new seasons in October 2025: "
    "Abbott Elementary (Season 5) and Grey's Anatomy (Season 22). For each show, please provide: "
    "(1) The exact premiere date (month, day, and year) when the season began airing on ABC, and "
    "(2) The total number of episodes ordered for that season. Please include reference URLs for each piece of information."
)

# Ground truth constraints per rubric
ABBOTT_EXPECTED_PREMIERE = "October 1, 2025"
ABBOTT_EXPECTED_EPISODES = 15

GREYS_EXPECTED_PREMIERE = "October 9, 2025"
GREYS_EXPECTED_EPISODES = 18


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class ShowSeasonInfo(BaseModel):
    show_name: Optional[str] = None
    season_number: Optional[str] = None

    premiere_date: Optional[str] = None
    premiere_sources: List[str] = Field(default_factory=list)

    episode_count: Optional[str] = None
    episode_sources: List[str] = Field(default_factory=list)


class SeriesExtraction(BaseModel):
    abbott_elementary_s5: Optional[ShowSeasonInfo] = None
    greys_anatomy_s22: Optional[ShowSeasonInfo] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_series_info() -> str:
    return """
    Extract the requested season information EXACTLY as stated in the answer for the following two items:
    1) Abbott Elementary — Season 5
    2) Grey's Anatomy — Season 22

    For EACH show/season, extract the following fields:
    - show_name: the show title as written in the answer (e.g., "Abbott Elementary", "Grey's Anatomy")
    - season_number: the season identifier mentioned (e.g., "Season 5", "Season 22")
    - premiere_date: the exact ABC U.S. broadcast premiere date for that season, in the format the answer used (e.g., "October 1, 2025" or "Oct. 1, 2025" or "10/1/2025"). This should be the day the season began airing on ABC in the U.S., not a streaming release date or international airing.
    - premiere_sources: an array of all URLs that the answer associates with supporting the premiere date for that season. Extract only explicit URLs present in the answer text. If none are present, return an empty array.
    - episode_count: the total number of episodes ordered for that specific season, as stated in the answer (e.g., "15", "15 episodes", or "fifteen").
    - episode_sources: an array of all URLs that the answer associates with supporting the stated episode count for that season. Extract only explicit URLs present in the answer text. If none are present, return an empty array.

    Return a JSON object with these two top-level fields:
    - abbott_elementary_s5: object with the above fields for Abbott Elementary Season 5 (or null if not found in the answer)
    - greys_anatomy_s22: object with the above fields for Grey's Anatomy Season 22 (or null if not found in the answer)

    IMPORTANT:
    - Do NOT invent or infer any missing information. If a required field is not present in the answer, set it to null (or [] for URL arrays).
    - For URLs, extract only valid, explicit URLs found in the answer (including those inside markdown links). Include full URLs with protocol.
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def _filter_urls(urls: Optional[List[str]]) -> List[str]:
    if not urls:
        return []
    cleaned = []
    for u in urls:
        if isinstance(u, str):
            s = u.strip()
            if s.startswith("http://") or s.startswith("https://"):
                cleaned.append(s)
    return cleaned


def _safe_str(x: Optional[str]) -> str:
    return x if isinstance(x, str) else ""


# --------------------------------------------------------------------------- #
# Verification routines                                                       #
# --------------------------------------------------------------------------- #
async def verify_show_block(
    evaluator: Evaluator,
    parent_node,
    show_key: str,
    show_label: str,
    season_num: int,
    extracted: Optional[ShowSeasonInfo],
    expected_premiere: str,
    expected_episodes: int,
    wiki_location_requirement: Optional[str] = None,  # "infobox" or "article_text" for episode count source policy
) -> None:
    """
    Build and run verification for a single show-season block.
    """
    # Build show-level node (critical as per rubric)
    show_node = evaluator.add_parallel(
        id=show_key,
        desc=f"{show_label} Season {season_num}: required premiere date and episode order count, each with supporting URL(s).",
        parent=parent_node,
        critical=True,
    )

    # Extract fields safely
    premiere_date = _safe_str(extracted.premiere_date) if extracted else ""
    premiere_sources = _filter_urls(extracted.premiere_sources if extracted else [])
    episode_count = _safe_str(extracted.episode_count) if extracted else ""
    episode_sources = _filter_urls(extracted.episode_sources if extracted else [])

    # 1) Premiere Date Value (critical)
    premiere_value_node = evaluator.add_leaf(
        id=f"{show_key.split('_')[0]}_Premiere_Date_Value" if "Abbott" in show_key or "Greys" in show_key else f"{show_key}_Premiere_Date_Value",
        desc=f"States the exact ABC premiere date (month, day, year) for {show_label} Season {season_num}; must be {expected_premiere} (per constraints).",
        parent=show_node,
        critical=True,
    )
    # Compare extracted date to expected date (robust match via LLM)
    premiere_value_claim = (
        f"The extracted ABC premiere date for {show_label} Season {season_num} is '{premiere_date}', "
        f"which denotes the same calendar date as {expected_premiere}."
    )
    await evaluator.verify(
        claim=premiere_value_claim,
        node=premiere_value_node,
        additional_instruction=(
            "Judge whether the two date expressions refer to the SAME day. "
            "Allow reasonable formatting variants: 'Oct.' vs 'October', numeric '10/1/2025', inclusion/exclusion of weekday, commas, or leading zeros. "
            "If the extracted value is missing/blank or clearly refers to a different date, mark as incorrect."
        ),
    )

    # 2) Premiere Date Citation Allowed (critical)
    premiere_cite_node = evaluator.add_leaf(
        id=f"{show_key.split('_')[0]}_Premiere_Date_Citation_Allowed",
        desc=f"Provides at least one reference URL from an allowed/verifiable source type that supports the stated {show_label} Season {season_num} premiere date.",
        parent=show_node,
        critical=True,
    )
    # If no sources, fail immediately (source-grounding policy)
    if not premiere_sources:
        premiere_cite_node.score = 0.0
        premiere_cite_node.status = "failed"
    else:
        premiere_cite_claim = (
            f"{show_label} Season {season_num} premiered (or is scheduled to premiere) on ABC on {expected_premiere}."
        )
        await evaluator.verify(
            claim=premiere_cite_claim,
            node=premiere_cite_node,
            sources=premiere_sources,
            additional_instruction=(
                "Only mark as supported if the specific page explicitly states the ABC U.S. broadcast season premiere date as given. "
                "Allowed/verifiable source types include: Wikipedia (wikipedia.org), ABC official pages (abc.com), Disney/ABC press/PR pages "
                "(e.g., press.disneyabc.com, dgepress.com), and reputable entertainment trades (variety.com, hollywoodreporter.com, deadline.com, "
                "tvline.com, ew.com). If the URL is not one of these types or does not explicitly provide the ABC premiere date for that season, "
                "mark as not supported."
            ),
        )

    # 3) Episode Count Value (critical)
    episode_value_node = evaluator.add_leaf(
        id=f"{show_key.split('_')[0]}_Episode_Count_Value",
        desc=f"States the total number of episodes ordered for {show_label} Season {season_num}; must be {expected_episodes} (per constraints).",
        parent=show_node,
        critical=True,
    )
    episode_value_claim = (
        f"The extracted total number of episodes ordered for {show_label} Season {season_num} is '{episode_count}', "
        f"which equals {expected_episodes}."
    )
    await evaluator.verify(
        claim=episode_value_claim,
        node=episode_value_node,
        additional_instruction=(
            "Determine if the extracted value unambiguously indicates the same count as the target integer. "
            "Accept reasonable textual variants like '15 episodes', 'a 15-episode season', or number words like 'fifteen'. "
            "If the extracted value is blank/missing or clearly indicates a different number, mark as incorrect."
        ),
    )

    # 4) Episode Count Citation with Wikipedia location constraint (critical)
    # Node id/desc per rubric specifics differ by show
    if (show_label == "Abbott Elementary"):
        ep_cite_id = f"{show_key.split('_')[0]}_Episode_Count_Citation_Wikipedia_Infobox"
        ep_cite_desc = (
            "Provides a Wikipedia reference URL supporting the Abbott Elementary Season 5 episode count, "
            "with the evidence present in the Wikipedia infobox (per constraints)."
        )
        location_requirement_text = (
            "Only mark as supported if the evidence is on a Wikipedia article (wikipedia.org) AND is clearly present in the infobox/summary table "
            "(typically the right-hand column) — look for fields like 'No. of episodes' or 'Episodes' in the infobox. "
            "If the information only appears in the body text or if the page is not Wikipedia, mark as NOT supported."
        )
    else:
        ep_cite_id = f"{show_key.split('_')[0]}_Episode_Count_Citation_Wikipedia_Article_Text"
        ep_cite_desc = (
            "Provides a Wikipedia reference URL supporting the Grey's Anatomy Season 22 episode count, "
            "with the evidence present in the Wikipedia article text (per constraints)."
        )
        location_requirement_text = (
            "Only mark as supported if the evidence is on a Wikipedia article (wikipedia.org) AND appears in the main article body text "
            "(NOT just in the infobox). If the episode count appears solely in the infobox or the page is not Wikipedia, mark as NOT supported."
        )

    episode_cite_node = evaluator.add_leaf(
        id=ep_cite_id,
        desc=ep_cite_desc,
        parent=show_node,
        critical=True,
    )
    if not episode_sources:
        episode_cite_node.score = 0.0
        episode_cite_node.status = "failed"
    else:
        episode_cite_claim = (
            f"The total number of episodes ordered for {show_label} Season {season_num} is {expected_episodes}."
        )
        await evaluator.verify(
            claim=episode_cite_claim,
            node=episode_cite_node,
            sources=episode_sources,
            additional_instruction=(
                f"{location_requirement_text} "
                "When judging, ensure the page explicitly states the episode count for the specified season, not for the series overall or a different season."
            ),
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
    Evaluate an answer for: ABC October 2025 season premieres and episode orders for
    Abbott Elementary (S5) and Grey's Anatomy (S22).
    """
    # Initialize evaluator (root is non-critical by design)
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

    # Extraction
    extracted_series = await evaluator.extract(
        prompt=prompt_extract_series_info(),
        template_class=SeriesExtraction,
        extraction_name="series_extraction",
    )

    # Record ground truth expectations
    evaluator.add_ground_truth({
        "Abbott Elementary S5": {
            "expected_premiere": ABBOTT_EXPECTED_PREMIERE,
            "expected_episode_count": ABBOTT_EXPECTED_EPISODES
        },
        "Grey's Anatomy S22": {
            "expected_premiere": GREYS_EXPECTED_PREMIERE,
            "expected_episode_count": GREYS_EXPECTED_EPISODES
        }
    }, gt_type="expected_values")

    # Top-level rubric node (critical, parallel over two shows)
    rubric_root = evaluator.add_parallel(
        id="Renewed_TV_Series_Information",
        desc="For BOTH specified series/seasons, provide the exact ABC season premiere date (month/day/year) and total episodes ordered, each supported by a reference URL that is an allowed/verifiable source.",
        parent=root,
        critical=True,
    )

    # Verify Abbott Elementary Season 5
    await verify_show_block(
        evaluator=evaluator,
        parent_node=rubric_root,
        show_key="Abbott_Elementary_Season_5",
        show_label="Abbott Elementary",
        season_num=5,
        extracted=extracted_series.abbott_elementary_s5,
        expected_premiere=ABBOTT_EXPECTED_PREMIERE,
        expected_episodes=ABBOTT_EXPECTED_EPISODES,
        wiki_location_requirement="infobox",
    )

    # Verify Grey's Anatomy Season 22
    await verify_show_block(
        evaluator=evaluator,
        parent_node=rubric_root,
        show_key="Greys_Anatomy_Season_22",
        show_label="Grey's Anatomy",
        season_num=22,
        extracted=extracted_series.greys_anatomy_s22,
        expected_premiere=GREYS_EXPECTED_PREMIERE,
        expected_episodes=GREYS_EXPECTED_EPISODES,
        wiki_location_requirement="article_text",
    )

    # Return final structured summary
    return evaluator.get_summary()