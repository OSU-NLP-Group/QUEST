import asyncio
import logging
from typing import Any, Optional, List, Dict

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy

# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "st_s5e3_credits"
TASK_DESCRIPTION = "Who directed and who wrote the third episode of Stranger Things Season 5, titled 'Chapter Three: The Turnbow Trap'?"

# Expected identity and credits based on rubric
EXPECTED_SHOW = "Stranger Things"
EXPECTED_SEASON = "5"
EXPECTED_EPISODE_NUMBER = "3"
EXPECTED_TITLE = "Chapter Three: The Turnbow Trap"
EXPECTED_PREMIERE_DATE = "November 26, 2025"
EXPECTED_DIRECTOR = "Frank Darabont"
EXPECTED_WRITER = "Caitlin Schneiderhan"


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class EpisodeIdentityExtraction(BaseModel):
    """Information identifying the specific episode as stated in the answer."""
    show_name: Optional[str] = None
    season_number: Optional[str] = None
    episode_number: Optional[str] = None
    episode_title: Optional[str] = None
    premiere_date: Optional[str] = None
    source_urls: List[str] = Field(default_factory=list)


class CreditsExtraction(BaseModel):
    """Director and writer for the episode as stated in the answer."""
    director_name: Optional[str] = None
    writer_name: Optional[str] = None
    source_urls: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_episode_identity() -> str:
    return (
        "Extract the episode identity details for the Stranger Things episode referenced in the answer. "
        "Return the following fields:\n"
        "1. show_name: The series name (e.g., 'Stranger Things').\n"
        "2. season_number: The season number as a string (e.g., '5').\n"
        "3. episode_number: The episode number as a string (e.g., '3').\n"
        "4. episode_title: The full episode title as written in the answer (e.g., 'Chapter Three: The Turnbow Trap').\n"
        "5. premiere_date: The premiere date for this episode if mentioned (e.g., 'November 26, 2025'). Keep the original formatting.\n"
        "6. source_urls: All URLs provided in the answer that are specifically cited for this episode (including plain URLs or markdown links). "
        "If no URLs are present, return an empty array.\n"
        "If any field is not mentioned in the answer, set it to null."
    )


def prompt_extract_credits() -> str:
    return (
        "Extract the director and writer credits for the Stranger Things episode referenced in the answer. "
        "Return the following fields:\n"
        "1. director_name: The name of the director as stated in the answer.\n"
        "2. writer_name: The name of the writer as stated in the answer.\n"
        "3. source_urls: All URLs provided in the answer that are cited for the director/writer credits of this episode "
        "(including plain URLs or markdown links). If none are present, return an empty array.\n"
        "If a field is not mentioned, set it to null."
    )


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def merge_sources(*lists: List[str]) -> List[str]:
    """Merge and deduplicate source URLs."""
    seen = set()
    merged: List[str] = []
    for lst in lists:
        for url in lst:
            if not url:
                continue
            if url not in seen:
                seen.add(url)
                merged.append(url)
    return merged


# --------------------------------------------------------------------------- #
# Verification functions                                                      #
# --------------------------------------------------------------------------- #
async def build_and_verify_episode_tree(
    evaluator: Evaluator,
    parent_node,
    id_info: EpisodeIdentityExtraction,
    credits_info: CreditsExtraction,
) -> None:
    """
    Build the verification tree following the rubric and perform verifications.
    """

    # Create the top-level node for this evaluation (critical, sequential)
    episode_credits_node = evaluator.add_sequential(
        id="episode_credits",
        desc="Identify the director and writer for the specified Stranger Things Season 5 Episode 3 ('Chapter Three: The Turnbow Trap').",
        parent=parent_node,
        critical=True,
    )

    # -------- Episode Identity (Parallel, Critical) -------- #
    identity_node = evaluator.add_parallel(
        id="episode_identity",
        desc="Answer corresponds to Stranger Things Season 5, Episode 3, titled 'Chapter Three: The Turnbow Trap', premiering Nov 26, 2025.",
        parent=episode_credits_node,
        critical=True,
    )

    identity_sources = id_info.source_urls or []
    # season_check
    season_leaf = evaluator.add_leaf(
        id="season_check",
        desc="Episode is from Stranger Things Season 5.",
        parent=identity_node,
        critical=True,
    )
    season_claim = (
        f"The episode titled '{EXPECTED_TITLE}' is part of {EXPECTED_SHOW} Season {EXPECTED_SEASON}."
    )
    await evaluator.verify(
        claim=season_claim,
        node=season_leaf,
        sources=identity_sources,
        additional_instruction=(
            "Verify using the provided URLs that this specific episode belongs to Season 5 of 'Stranger Things'. "
            "Allow minor name variants (e.g., casing or punctuation)."
        ),
    )

    # episode_number_check
    epnum_leaf = evaluator.add_leaf(
        id="episode_number_check",
        desc="Episode is Episode 3.",
        parent=identity_node,
        critical=True,
    )
    epnum_claim = (
        f"The episode titled '{EXPECTED_TITLE}' is Episode {EXPECTED_EPISODE_NUMBER} of {EXPECTED_SHOW} Season {EXPECTED_SEASON}."
    )
    await evaluator.verify(
        claim=epnum_claim,
        node=epnum_leaf,
        sources=identity_sources,
        additional_instruction=(
            "Confirm from the webpages that the specified episode is numbered 3 within Season 5."
        ),
    )

    # title_check
    title_leaf = evaluator.add_leaf(
        id="title_check",
        desc="Episode title is 'Chapter Three: The Turnbow Trap'.",
        parent=identity_node,
        critical=True,
    )
    title_claim = (
        f"The title of {EXPECTED_SHOW} Season {EXPECTED_SEASON} Episode {EXPECTED_EPISODE_NUMBER} is '{EXPECTED_TITLE}'."
    )
    await evaluator.verify(
        claim=title_claim,
        node=title_leaf,
        sources=identity_sources,
        additional_instruction=(
            "Check the episode listing or credits page to verify the exact title. "
            "Treat minor punctuation or quotation mark variations as acceptable."
        ),
    )

    # premiere_date_check
    premiere_leaf = evaluator.add_leaf(
        id="premiere_date_check",
        desc="Episode premiered on November 26, 2025 (as part of Volume 1).",
        parent=identity_node,
        critical=True,
    )
    premiere_claim = (
        f"{EXPECTED_SHOW} Season {EXPECTED_SEASON} Episode {EXPECTED_EPISODE_NUMBER} ('{EXPECTED_TITLE}') premiered on {EXPECTED_PREMIERE_DATE}."
    )
    await evaluator.verify(
        claim=premiere_claim,
        node=premiere_leaf,
        sources=identity_sources,
        additional_instruction=(
            "Validate the public Netflix release/premiere date from the webpage(s). Allow reasonable date format variants "
            "(e.g., 'Nov. 26, 2025', '26 November 2025')."
        ),
    )

    # -------- Credits (Parallel, Critical) -------- #
    credits_node = evaluator.add_parallel(
        id="credits",
        desc="Provide the correct director and writer for the specified episode.",
        parent=episode_credits_node,
        critical=True,
    )

    credits_sources = credits_info.source_urls or []
    combined_sources = merge_sources(identity_sources, credits_sources)

    # director_identification
    director_leaf = evaluator.add_leaf(
        id="director_identification",
        desc="Director is correctly identified as Frank Darabont.",
        parent=credits_node,
        critical=True,
    )
    director_claim = (
        f"{EXPECTED_SHOW} Season {EXPECTED_SEASON} Episode {EXPECTED_EPISODE_NUMBER} ('{EXPECTED_TITLE}') was directed by {EXPECTED_DIRECTOR}."
    )
    await evaluator.verify(
        claim=director_claim,
        node=director_leaf,
        sources=combined_sources,
        additional_instruction=(
            "Confirm the director credit from the provided URLs. The page must explicitly list the director for this specific episode."
        ),
    )

    # writer_identification
    writer_leaf = evaluator.add_leaf(
        id="writer_identification",
        desc="Writer is correctly identified as Caitlin Schneiderhan.",
        parent=credits_node,
        critical=True,
    )
    writer_claim = (
        f"{EXPECTED_SHOW} Season {EXPECTED_SEASON} Episode {EXPECTED_EPISODE_NUMBER} ('{EXPECTED_TITLE}') was written by {EXPECTED_WRITER}."
    )
    await evaluator.verify(
        claim=writer_claim,
        node=writer_leaf,
        sources=combined_sources,
        additional_instruction=(
            "Confirm the writer credit from the provided URLs. The page must explicitly list the writer for this specific episode."
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
    Evaluate the answer for Stranger Things S5E3 director and writer identification.
    """
    # Initialize evaluator
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.SEQUENTIAL,  # Root strategy; single child below
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

    # Extract identity and credits info from the answer
    id_info = await evaluator.extract(
        prompt=prompt_extract_episode_identity(),
        template_class=EpisodeIdentityExtraction,
        extraction_name="episode_identity",
    )
    credits_info = await evaluator.extract(
        prompt=prompt_extract_credits(),
        template_class=CreditsExtraction,
        extraction_name="credits_info",
    )

    # Add ground truth expectations for transparency
    evaluator.add_ground_truth({
        "expected_show": EXPECTED_SHOW,
        "expected_season": EXPECTED_SEASON,
        "expected_episode_number": EXPECTED_EPISODE_NUMBER,
        "expected_title": EXPECTED_TITLE,
        "expected_premiere_date": EXPECTED_PREMIERE_DATE,
        "expected_director": EXPECTED_DIRECTOR,
        "expected_writer": EXPECTED_WRITER,
    }, gt_type="expected_episode_credits")

    # Build tree and perform verifications according to rubric
    await build_and_verify_episode_tree(evaluator, root, id_info, credits_info)

    # Return structured summary
    return evaluator.get_summary()