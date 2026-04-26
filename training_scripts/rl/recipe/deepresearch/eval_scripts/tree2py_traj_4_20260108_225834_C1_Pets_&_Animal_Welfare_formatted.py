import asyncio
import logging
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy

# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "ca_pet_adoption_min_age"
TASK_DESCRIPTION = "What is the minimum age requirement for adopting a pet from an animal shelter in California?"


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class AgeSourceExtraction(BaseModel):
    """
    Extract from the agent's answer:
    - minimum_age_text: the age requirement as stated (e.g., '18 years old', '18+', 'at least 18')
    - minimum_age_number: normalized number string (e.g., '18') if clearly indicated; null otherwise
    - source_urls: URLs cited as sources/references (California shelter or official state/local sources if present)
    """
    minimum_age_text: Optional[str] = None
    minimum_age_number: Optional[str] = None
    source_urls: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_age_and_sources() -> str:
    return """
    Extract the minimum age requirement for adopting a pet from an animal shelter in California as stated in the answer, and collect any cited URLs.

    Return the following fields:
    - minimum_age_text: The exact phrase in the answer that states the minimum age requirement for adoption in California (e.g., '18 years old', 'at least 18', '18+'). If not stated, return null.
    - minimum_age_number: The normalized numeric value for the minimum age if explicitly stated (e.g., '18'). If ambiguous or not provided, return null.
    - source_urls: A list of all URLs mentioned as sources or references in the answer (including markdown links). Only include actual URLs; if sources are mentioned without a URL, do not include them.

    Notes:
    - If multiple age numbers are mentioned, choose the one clearly tied to California's adoption requirement.
    - Do not invent URLs. Extract only URLs explicitly present in the answer.
    """


# --------------------------------------------------------------------------- #
# Verification subtree builder                                                #
# --------------------------------------------------------------------------- #
async def build_minimum_age_requirement_subtree(
    evaluator: Evaluator,
    parent_node,
    extraction: AgeSourceExtraction,
) -> None:
    """
    Build the verification subtree according to the rubric:
    - Minimum_Age_Requirement (critical, parallel)
      - Age_Value (critical, leaf)
      - California_Source (critical, leaf)
    """
    # Parent node representing the rubric's main criterion
    min_node = evaluator.add_parallel(
        id="Minimum_Age_Requirement",
        desc="The minimum age requirement for adopting a pet from an animal shelter in California is correctly identified.",
        parent=parent_node,
        critical=True,
    )

    # Leaf: Age_Value — verify the answer explicitly states "18 years old" (or equivalent)
    age_leaf = evaluator.add_leaf(
        id="Age_Value",
        desc="The minimum age is correctly specified as 18 years old.",
        parent=min_node,
        critical=True,
    )

    # We verify against the answer text (simple verify). Accept common phrasings like "18", "18 years old", "at least 18", "18+"
    age_claim = (
        "The provided answer explicitly states that the minimum age requirement for adopting a pet from an animal "
        "shelter in California is 18 years old (phrases like '18', '18 years old', 'at least 18', or '18+')."
    )
    await evaluator.verify(
        claim=age_claim,
        node=age_leaf,
        additional_instruction=(
            "Only PASS if the answer text clearly indicates the age is 18. Minor phrasing variants are acceptable. "
            "If the answer states a different number (e.g., 21) or does not explicitly specify the age, mark as incorrect."
        ),
    )

    # Leaf: California_Source — verify that at least one cited URL is a California animal shelter or official California source
    ca_src_leaf = evaluator.add_leaf(
        id="California_Source",
        desc="A reference to a California animal shelter or official California source is provided.",
        parent=min_node,
        critical=True,
    )

    # Use all extracted URLs; verification succeeds if any URL is a CA shelter or official CA government page.
    urls = extraction.source_urls or []

    ca_src_claim = (
        "This page belongs to an official California government website (e.g., domain ending in .ca.gov or .ca.us) "
        "or a California animal shelter (city/county animal services or a California SPCA/humane society)."
    )
    await evaluator.verify(
        claim=ca_src_claim,
        node=ca_src_leaf,
        sources=urls,
        additional_instruction=(
            "PASS if at least one of the provided URLs is clearly a California government site or a California animal "
            "shelter (municipal animal services or well-known CA SPCA/humane society organizations). Evidence can be "
            "the domain (e.g., .ca.gov, .ca.us) or on-page address/branding indicating a California location. "
            "The page does not need to state the age explicitly; it only needs to be a California official/shelter source. "
            "If no URLs are provided or none qualify as California sources, mark as incorrect."
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
    Evaluate an answer for the California pet adoption minimum age task.
    """
    # Initialize evaluator
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

    # Extract age and sources from the answer
    extraction = await evaluator.extract(
        prompt=prompt_extract_age_and_sources(),
        template_class=AgeSourceExtraction,
        extraction_name="age_and_sources",
    )

    # Add ground truth info for reference
    evaluator.add_ground_truth(
        {
            "expected_minimum_age_years": "18",
            "notes": "Most California municipal shelters require adopters to be at least 18 years old.",
        }
    )

    # Build verification subtree and run checks
    await build_minimum_age_requirement_subtree(evaluator, root, extraction)

    # Return structured summary
    return evaluator.get_summary()