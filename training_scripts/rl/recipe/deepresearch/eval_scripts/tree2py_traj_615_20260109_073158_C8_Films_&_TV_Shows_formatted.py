import asyncio
import logging
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys

# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "netflix_bnw_2024"
TASK_DESCRIPTION = """Identify the 2024 Netflix original limited series that meets ALL of the following criteria:
- The series is presented entirely in black and white cinematography
- The series was nominated for Outstanding Limited or Anthology Series at the 2024 Emmy Awards
- The series won BOTH the Emmy Award for Outstanding Directing for a Limited or Anthology Series or Movie AND the Emmy Award for Outstanding Cinematography for a Limited or Anthology Series or Movie in 2024
- The series consists of exactly 8 episodes
- All episodes were directed by the same person
- All episodes were shot by the same cinematographer
- The series premiered between January 1, 2024 and April 30, 2024

Provide the following information:
1. The title of the series
2. The name of the director who directed all episodes
3. The name of the cinematographer who shot all episodes
4. The name of at least one lead actor from the series
5. The exact premiere/release date of the series on Netflix
"""

# --------------------------------------------------------------------------- #
# Data models                                                                 #
# --------------------------------------------------------------------------- #
class SeriesInfo(BaseModel):
    """Structured extraction of the identified series and associated details from the answer."""
    title: Optional[str] = None
    director: Optional[str] = None  # Director who directed all episodes
    cinematographer: Optional[str] = None  # Cinematographer who shot all episodes
    lead_actor: Optional[str] = None
    premiere_date: Optional[str] = None  # Exact Netflix release date
    episodes_count: Optional[str] = None  # Keep as string to be robust to formats like "8"
    netflix_original: Optional[str] = None  # e.g., "yes"/"true"/"Netflix Original"
    limited_series: Optional[str] = None  # e.g., "limited series", "miniseries"
    black_and_white: Optional[str] = None  # e.g., "entirely black-and-white"
    emmy_nomination_limited_or_anthology: Optional[str] = None
    emmy_win_directing: Optional[str] = None
    emmy_win_cinematography: Optional[str] = None
    same_director_all_episodes: Optional[str] = None
    same_cinematographer_all_episodes: Optional[str] = None
    production_companies: List[str] = Field(default_factory=list)
    netflix_page_url: Optional[str] = None
    source_urls: List[str] = Field(default_factory=list)  # All other URLs cited in the answer


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_series_info() -> str:
    return """
    Extract the identified Netflix original limited series and all requested fields from the answer.
    Return a single JSON object using the following keys:
    - title: The exact title of the series mentioned.
    - director: The name of the director who is claimed to have directed all episodes.
    - cinematographer: The name of the cinematographer who is claimed to have shot all episodes.
    - lead_actor: The name of at least one lead actor mentioned from the series.
    - premiere_date: The exact premiere/release date of the series on Netflix (as written in the answer). Keep the original format/string.
    - episodes_count: The number of episodes (keep as a string, e.g., "8" or "eight").
    - netflix_original: A short phrase indicating the answer claims Netflix original status (e.g., "Netflix original", "original", "yes"). If not mentioned, return null.
    - limited_series: A short phrase indicating the answer claims it is a limited series/miniseries (e.g., "limited series", "miniseries"). If not mentioned, return null.
    - black_and_white: A short phrase indicating the answer claims the series is entirely in black-and-white (e.g., "black and white", "B&W"). If not mentioned, return null.
    - emmy_nomination_limited_or_anthology: A short phrase or sentence stating the answer's claim about nomination for Outstanding Limited or Anthology Series at the 2024 Emmys. If not present, return null.
    - emmy_win_directing: A short phrase or sentence stating the answer's claim about winning the Emmy for Outstanding Directing for a Limited or Anthology Series or Movie in 2024. If not present, return null.
    - emmy_win_cinematography: A short phrase or sentence stating the answer's claim about winning the Emmy for Outstanding Cinematography for a Limited or Anthology Series or Movie in 2024. If not present, return null.
    - same_director_all_episodes: A short phrase/sentence indicating the answer's claim that all episodes were directed by the same person. If not present, return null.
    - same_cinematographer_all_episodes: A short phrase/sentence indicating the answer's claim that all episodes were shot by the same cinematographer. If not present, return null.
    - production_companies: List all production company names mentioned in the answer for the series. If none are mentioned, return an empty list.
    - netflix_page_url: Extract the official Netflix series page URL if present in the answer (full URL including protocol). If absent, return null.
    - source_urls: Extract all other URLs cited in the answer (including news, Wikipedia, Emmys, etc.). Return as a list of URLs (full URLs with protocol). If none are cited, return an empty list.

    Do not invent or infer information. If a field is missing in the answer, use null or empty list as specified.
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def assemble_sources(info: SeriesInfo) -> List[str]:
    """Combine Netflix page and all cited source URLs into a single list, de-duplicated."""
    combined: List[str] = []
    if info.netflix_page_url and info.netflix_page_url.strip():
        combined.append(info.netflix_page_url.strip())
    for u in (info.source_urls or []):
        if isinstance(u, str) and u.strip():
            combined.append(u.strip())
    # De-duplicate while preserving order
    seen = set()
    deduped = []
    for s in combined:
        if s not in seen:
            seen.add(s)
            deduped.append(s)
    return deduped


def safe_title(info: SeriesInfo) -> str:
    return info.title.strip() if info.title else "the series"


# --------------------------------------------------------------------------- #
# Verification builders                                                       #
# --------------------------------------------------------------------------- #
async def build_series_eligibility_checks(evaluator: Evaluator, parent_node, info: SeriesInfo) -> None:
    """Create and execute all eligibility verification leaf nodes under a critical parallel aggregator."""
    criteria_node = evaluator.add_parallel(
        id="Series_Eligibility_Criteria",
        desc="Verify the series meets all eligibility constraints stated in the question/constraints",
        parent=parent_node,
        critical=True
    )

    title_for_claims = safe_title(info)
    all_sources = assemble_sources(info)

    # Netflix Original
    netflix_original_node = evaluator.add_leaf(
        id="Netflix_Original",
        desc="The series must be a Netflix original release",
        parent=criteria_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The series {title_for_claims} is a Netflix original release.",
        node=netflix_original_node,
        sources=all_sources,
        additional_instruction="Verify that the series is labeled or credited as a Netflix Original on official Netflix or reputable sources."
    )

    # Limited Series Format
    limited_series_node = evaluator.add_leaf(
        id="Limited_Series_Format",
        desc="The series must be a limited series format (not an ongoing series)",
        parent=criteria_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The series {title_for_claims} is categorized as a limited series (miniseries) and is not ongoing.",
        node=limited_series_node,
        sources=all_sources,
        additional_instruction="Check whether the series is described as a limited series/miniseries. Avoid ongoing shows."
    )

    # Black-and-White Cinematography
    bnw_node = evaluator.add_leaf(
        id="Black_and_White_Cinematography",
        desc="The series must be presented entirely in black and white cinematography",
        parent=criteria_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The series {title_for_claims} is presented entirely in black-and-white.",
        node=bnw_node,
        sources=all_sources,
        additional_instruction="Verify that all episodes are presented in black-and-white; no episodes in color."
    )

    # Emmy Nomination: Outstanding Limited or Anthology Series (2024)
    emmy_nomination_node = evaluator.add_leaf(
        id="Emmy_Nomination_Limited_or_Anthology",
        desc="The series must have been nominated for Outstanding Limited or Anthology Series at the 2024 Emmy Awards",
        parent=criteria_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The series {title_for_claims} was nominated for Outstanding Limited or Anthology Series at the 2024 Emmy Awards.",
        node=emmy_nomination_node,
        sources=all_sources,
        additional_instruction="Check official Emmys or reputable sources listing 2024 nominees for Outstanding Limited or Anthology Series."
    )

    # Emmy Win: Directing (2024)
    emmy_win_directing_node = evaluator.add_leaf(
        id="Emmy_Win_Directing",
        desc="The series must have won the Emmy Award for Outstanding Directing for a Limited or Anthology Series or Movie in 2024",
        parent=criteria_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The series {title_for_claims} won the 2024 Emmy Award for Outstanding Directing for a Limited or Anthology Series or Movie.",
        node=emmy_win_directing_node,
        sources=all_sources,
        additional_instruction="Confirm the series is listed as the winner (not just nominated) in 2024 for the specified directing category."
    )

    # Emmy Win: Cinematography (2024)
    emmy_win_cinema_node = evaluator.add_leaf(
        id="Emmy_Win_Cinematography",
        desc="The series must have won the Emmy Award for Outstanding Cinematography for a Limited or Anthology Series or Movie in 2024",
        parent=criteria_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The series {title_for_claims} won the 2024 Emmy Award for Outstanding Cinematography for a Limited or Anthology Series or Movie.",
        node=emmy_win_cinema_node,
        sources=all_sources,
        additional_instruction="Confirm the series is listed as the winner (not just nominated) in 2024 for the specified cinematography category."
    )

    # Exactly 8 Episodes
    episodes_node = evaluator.add_leaf(
        id="Exactly_8_Episodes",
        desc="The series consists of exactly 8 episodes",
        parent=criteria_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The series {title_for_claims} consists of exactly 8 episodes.",
        node=episodes_node,
        sources=all_sources,
        additional_instruction="Verify the episode count is exactly 8."
    )

    # Same Director Across All Episodes
    same_dir_node = evaluator.add_leaf(
        id="Same_Director_All_Episodes",
        desc="All episodes were directed by the same person",
        parent=criteria_node,
        critical=True
    )
    dir_claim = (
        f"All episodes of {title_for_claims} were directed by {info.director}."
        if info.director and info.director.strip()
        else f"All episodes of {title_for_claims} were directed by the same person."
    )
    await evaluator.verify(
        claim=dir_claim,
        node=same_dir_node,
        sources=all_sources,
        additional_instruction="Confirm that a single director handled all episodes; if a name is provided, verify that specific person."
    )

    # Same Cinematographer Across All Episodes
    same_dp_node = evaluator.add_leaf(
        id="Same_Cinematographer_All_Episodes",
        desc="All episodes were shot by the same cinematographer",
        parent=criteria_node,
        critical=True
    )
    dp_claim = (
        f"All episodes of {title_for_claims} were shot by {info.cinematographer}."
        if info.cinematographer and info.cinematographer.strip()
        else f"All episodes of {title_for_claims} were shot by the same cinematographer."
    )
    await evaluator.verify(
        claim=dp_claim,
        node=same_dp_node,
        sources=all_sources,
        additional_instruction="Confirm that a single cinematographer handled all episodes; if a name is provided, verify that specific person."
    )

    # Premiere Date In Window (Jan 1, 2024 to Apr 30, 2024 inclusive)
    premiere_window_node = evaluator.add_leaf(
        id="Premiere_Date_In_Window",
        desc="The series premiered/released on Netflix between January 1, 2024 and April 30, 2024 (inclusive)",
        parent=criteria_node,
        critical=True
    )
    if info.premiere_date and info.premiere_date.strip():
        prem_claim = (
            f"The series {title_for_claims} premiered on Netflix on {info.premiere_date}, "
            f"which falls between January 1, 2024 and April 30, 2024 (inclusive)."
        )
    else:
        prem_claim = (
            f"The series {title_for_claims} premiered on Netflix between January 1, 2024 and April 30, 2024 (inclusive)."
        )
    await evaluator.verify(
        claim=prem_claim,
        node=premiere_window_node,
        sources=all_sources,
        additional_instruction="Check the Netflix release date and judge whether it lies within the specified window."
    )

    # Production Company Credits Publicly Available
    prod_companies_str = ", ".join(info.production_companies) if info.production_companies else "production companies"
    prod_comp_node = evaluator.add_leaf(
        id="Production_Company_Credits_Publicly_Available",
        desc="The series must have identifiable production company credits publicly available",
        parent=criteria_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The series {title_for_claims} has identifiable production company credits publicly listed (e.g., {prod_companies_str}).",
        node=prod_comp_node,
        sources=all_sources,
        additional_instruction="Verify that production company credits are publicly available on reputable sources (official Netflix, studio pages, Wikipedia, etc.)."
    )


async def build_required_output_checks(evaluator: Evaluator, parent_node, info: SeriesInfo) -> None:
    """Create and execute leaf nodes ensuring the answer provides all requested fields."""
    output_node = evaluator.add_parallel(
        id="Required_Output_Fields",
        desc="Verify the response provides all requested information fields",
        parent=parent_node,
        critical=True
    )

    # Title provided
    evaluator.add_custom_node(
        result=bool(info.title and info.title.strip()),
        id="Provide_Series_Title",
        desc="Provide the title of the series",
        parent=output_node,
        critical=True
    )

    # Director name provided
    evaluator.add_custom_node(
        result=bool(info.director and info.director.strip()),
        id="Provide_Director_Name",
        desc="Provide the name of the director who directed all episodes",
        parent=output_node,
        critical=True
    )

    # Cinematographer name provided
    evaluator.add_custom_node(
        result=bool(info.cinematographer and info.cinematographer.strip()),
        id="Provide_Cinematographer_Name",
        desc="Provide the name of the cinematographer who shot all episodes",
        parent=output_node,
        critical=True
    )

    # Lead actor provided
    evaluator.add_custom_node(
        result=bool(info.lead_actor and info.lead_actor.strip()),
        id="Provide_Lead_Actor",
        desc="Provide the name of at least one lead actor from the series",
        parent=output_node,
        critical=True
    )

    # Exact premiere date provided
    evaluator.add_custom_node(
        result=bool(info.premiere_date and info.premiere_date.strip()),
        id="Provide_Exact_Premiere_Date",
        desc="Provide the exact premiere/release date of the series on Netflix",
        parent=output_node,
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
    model: str = "o4-mini"
) -> Dict:
    """
    Evaluate the answer for the Netflix black-and-white 2024 limited series task.
    Builds a critical verification tree ensuring all constraints and required fields are met.
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

    # Extract series info from the answer
    series_info = await evaluator.extract(
        prompt=prompt_extract_series_info(),
        template_class=SeriesInfo,
        extraction_name="series_info"
    )

    # Create a critical "task root" under the evaluator root to reflect rubric's critical root
    task_root = evaluator.add_parallel(
        id="Task_Root",
        desc="Evaluate whether the identified Netflix original limited series satisfies all required criteria and whether all requested information is provided",
        parent=root,
        critical=True
    )

    # Build verification subtrees
    await build_series_eligibility_checks(evaluator, task_root, series_info)
    await build_required_output_checks(evaluator, task_root, series_info)

    # Return structured summary
    return evaluator.get_summary()