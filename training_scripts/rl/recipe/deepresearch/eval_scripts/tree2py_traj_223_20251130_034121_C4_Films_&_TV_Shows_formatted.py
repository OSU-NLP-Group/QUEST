import asyncio
import logging
import re
from typing import Any, Optional, List, Dict

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy

# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "charlie_heaton_dci_banks_2015"
TASK_DESCRIPTION = (
    "The actor who plays Jonathan Byers in Stranger Things had a guest role in a British crime drama television series in 2015, "
    "appearing in two consecutive episodes during Season 4 of that series. Provide the following information about this appearance: "
    "(1) the name of the television series, (2) the broadcasting network, (3) the character name he played, (4) the season number, "
    "(5) the episode number of the first episode, (6) the title of the first episode, (7) the episode number of the second episode, "
    "(8) the title of the second episode."
)

# Ground-truth constraints expected by rubric
GROUND_TRUTH = {
    "actor_name": "Charlie Heaton",
    "series_name": "DCI Banks",
    "broadcast_network": "ITV",
    "character_name": "Gary McCready",
    "season_number": "4",
    "episode_1_number": "1",
    "episode_1_title": "What Will Survive: Part 1",
    "episode_2_number": "2",
    "episode_2_title": "What Will Survive: Part 2",
}

# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class AppearanceExtraction(BaseModel):
    """
    Structured extraction of the appearance details from the agent's answer.
    All fields are optional strings; URLs are collected for evidence verification.
    """
    actor_name: Optional[str] = None
    series_name: Optional[str] = None
    broadcasting_network: Optional[str] = None
    character_name: Optional[str] = None
    season_number: Optional[str] = None
    episode_1_number: Optional[str] = None
    episode_1_title: Optional[str] = None
    episode_2_number: Optional[str] = None
    episode_2_title: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_appearance() -> str:
    return """
    Extract the appearance information for the actor who plays Jonathan Byers in Stranger Things, focusing on the British crime drama guest role in 2015 across two consecutive episodes in Season 4.

    Extract the following fields exactly as presented in the answer:
    - actor_name: The name of the actor (the person who plays Jonathan Byers in Stranger Things)
    - series_name: The television series name for the guest appearance
    - broadcasting_network: The broadcasting network of that series (e.g., ITV)
    - character_name: The character name he played
    - season_number: The season number (e.g., 4, or 'Season 4'/'Series 4'; return the number part if possible)
    - episode_1_number: The episode number of the first episode (e.g., 1)
    - episode_1_title: The title of the first episode (e.g., 'What Will Survive: Part 1')
    - episode_2_number: The episode number of the second episode (e.g., 2)
    - episode_2_title: The title of the second episode (e.g., 'What Will Survive: Part 2')

    Also extract:
    - sources: All URLs explicitly mentioned in the answer that are relevant to this appearance (e.g., Wikipedia, IMDb, or official pages for DCI Banks, the season, or the episodes). Collect them into a list. Only include valid URLs that appear in the answer; do not invent any.

    Notes:
    - If any required field is missing from the answer, set it to null.
    - For season, if the answer uses 'Series 4' instead of 'Season 4', you may still return '4' in season_number.
    - Preserve episode titles exactly as written, including punctuation, but it's okay if the answer uses minor punctuation variations (hyphen vs dash).
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def _safe_int(text: Optional[str]) -> Optional[int]:
    if text is None:
        return None
    m = re.search(r"\d+", str(text))
    return int(m.group(0)) if m else None


# --------------------------------------------------------------------------- #
# Verification functions                                                      #
# --------------------------------------------------------------------------- #
async def build_and_verify_actor_identity(
    evaluator: Evaluator,
    parent_node,
    extracted: AppearanceExtraction,
) -> None:
    """
    Create and verify the critical actor identity leaf.
    """
    actor_leaf = evaluator.add_leaf(
        id="actor_identity",
        desc="Identifies the actor who plays Jonathan Byers in Stranger Things as Charlie Heaton.",
        parent=parent_node,
        critical=True,
    )

    # Use the extracted actor name if available to phrase a robust match claim.
    extracted_name = extracted.actor_name or ""
    claim = (
        f"The actor who plays Jonathan Byers in Stranger Things is Charlie Heaton. "
        f"The extracted name '{extracted_name}' should correspond to Charlie Heaton."
    )
    await evaluator.verify(
        claim=claim,
        node=actor_leaf,
        additional_instruction=(
            "Judge whether the answer correctly identifies the actor as Charlie Heaton. "
            "Allow minor formatting/casing differences or inclusion/omission of middle names. "
            "If the extracted name differs but clearly refers to the same person, consider it correct."
        ),
    )


async def build_and_verify_appearance_details(
    evaluator: Evaluator,
    parent_node,
    extracted: AppearanceExtraction,
) -> None:
    """
    Build the 'appearance_details' critical parallel node with eight critical leaf checks,
    each verified (preferably) against the cited sources.
    """
    details_node = evaluator.add_parallel(
        id="appearance_details",
        desc="Correctly identifies the specific series/role/season/episodes/titles for the appearance, matching the constraints.",
        parent=parent_node,
        critical=True,
    )

    sources_list = extracted.sources if extracted and extracted.sources else None

    # 1. Series name
    series_leaf = evaluator.add_leaf(
        id="series_name",
        desc="Television series name is DCI Banks.",
        parent=details_node,
        critical=True,
    )
    await evaluator.verify(
        claim="The British crime drama television series for Charlie Heaton's 2015 guest appearance (two consecutive episodes in Season 4) is 'DCI Banks'.",
        node=series_leaf,
        sources=sources_list,
        additional_instruction=(
            "Use the cited sources (e.g., Wikipedia/IMDb/official pages) to confirm that Charlie Heaton appeared on DCI Banks. "
            "Focus on validating the series title itself."
        ),
    )

    # 2. Broadcast network
    network_leaf = evaluator.add_leaf(
        id="broadcast_network",
        desc="Broadcasting network is ITV.",
        parent=details_node,
        critical=True,
    )
    await evaluator.verify(
        claim="The series DCI Banks was broadcast on ITV (UK).",
        node=network_leaf,
        sources=sources_list,
        additional_instruction=(
            "Confirm the broadcaster for DCI Banks is ITV. Accept variants like 'ITV1' or references to 'ITV network' as equivalent."
        ),
    )

    # 3. Character name
    character_leaf = evaluator.add_leaf(
        id="character_name",
        desc="Character name played is Gary McCready.",
        parent=details_node,
        critical=True,
    )
    await evaluator.verify(
        claim="Charlie Heaton played the character 'Gary McCready' in DCI Banks.",
        node=character_leaf,
        sources=sources_list,
        additional_instruction=(
            "Verify cast info that lists Charlie Heaton as portraying 'Gary McCready'. Allow minor punctuation or spacing variants."
        ),
    )

    # 4. Season number
    season_leaf = evaluator.add_leaf(
        id="season_number",
        desc="Season number is 4.",
        parent=details_node,
        critical=True,
    )
    await evaluator.verify(
        claim="Charlie Heaton's guest appearance in DCI Banks took place in Season 4 (also called Series 4).",
        node=season_leaf,
        sources=sources_list,
        additional_instruction=(
            "UK sources may label seasons as 'Series'. Treat 'Series 4' as equivalent to 'Season 4'."
        ),
    )

    # 5. Episode 1 number
    ep1_num_leaf = evaluator.add_leaf(
        id="episode_1_number",
        desc="First episode number is 1.",
        parent=details_node,
        critical=True,
    )
    await evaluator.verify(
        claim="The first episode of Charlie Heaton's appearance in DCI Banks Season 4 is Episode 1.",
        node=ep1_num_leaf,
        sources=sources_list,
        additional_instruction=(
            "Check episode listings for Season/Series 4 to confirm that 'What Will Survive: Part 1' is Episode 1."
        ),
    )

    # 6. Episode 1 title
    ep1_title_leaf = evaluator.add_leaf(
        id="episode_1_title",
        desc="First episode title is 'What Will Survive: Part 1'.",
        parent=details_node,
        critical=True,
    )
    await evaluator.verify(
        claim="The title of the first episode is 'What Will Survive: Part 1'.",
        node=ep1_title_leaf,
        sources=sources_list,
        additional_instruction=(
            "Allow minor punctuation variations (hyphen vs en dash, smart quotes). The core title must match 'What Will Survive: Part 1'."
        ),
    )

    # 7. Episode 2 number
    ep2_num_leaf = evaluator.add_leaf(
        id="episode_2_number",
        desc="Second episode number is 2.",
        parent=details_node,
        critical=True,
    )
    await evaluator.verify(
        claim="The second episode of Charlie Heaton's appearance in DCI Banks Season 4 is Episode 2.",
        node=ep2_num_leaf,
        sources=sources_list,
        additional_instruction=(
            "Check episode listings for Season/Series 4 to confirm that 'What Will Survive: Part 2' is Episode 2."
        ),
    )

    # 8. Episode 2 title
    ep2_title_leaf = evaluator.add_leaf(
        id="episode_2_title",
        desc="Second episode title is 'What Will Survive: Part 2'.",
        parent=details_node,
        critical=True,
    )
    await evaluator.verify(
        claim="The title of the second episode is 'What Will Survive: Part 2'.",
        node=ep2_title_leaf,
        sources=sources_list,
        additional_instruction=(
            "Allow minor punctuation variations (hyphen vs en dash, smart quotes). The core title must match 'What Will Survive: Part 2'."
        ),
    )


async def add_consecutive_episodes_check(
    evaluator: Evaluator,
    parent_node,
    extracted: AppearanceExtraction,
) -> None:
    """
    Non-critical custom check to ensure the two episodes are consecutive within the same season.
    """
    ep1_num = _safe_int(extracted.episode_1_number)
    ep2_num = _safe_int(extracted.episode_2_number)
    # If season_number is present, ensure it points to same season context; here both episodes
    # are from the same provided season, so we just ensure ep2 == ep1 + 1.
    consecutive = (ep1_num is not None and ep2_num is not None and (ep2_num == ep1_num + 1))

    evaluator.add_custom_node(
        result=bool(consecutive),
        id="consecutive_episodes",
        desc="The response indicates (or the cited episodes imply) that the appearance spans two consecutive episodes (Episode 1 then Episode 2 of the same season).",
        parent=parent_node,
        critical=False,
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
    Evaluate the agent's answer for Charlie Heaton's 2015 DCI Banks guest appearance details.
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

    # Extract structured info from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_appearance(),
        template_class=AppearanceExtraction,
        extraction_name="appearance_extraction",
    )

    # Record ground truth info for transparency
    evaluator.add_ground_truth(
        {
            "expected": GROUND_TRUTH,
            "notes": "Expected values per rubric for the 2015 DCI Banks guest appearance.",
        },
        gt_type="ground_truth",
    )

    # Build critical actor identity leaf
    await build_and_verify_actor_identity(evaluator, root, extracted)

    # Build critical appearance detail checks
    await build_and_verify_appearance_details(evaluator, root, extracted)

    # Add non-critical consecutive episodes check (computed)
    await add_consecutive_episodes_check(evaluator, root, extracted)

    # Return evaluation summary
    return evaluator.get_summary()