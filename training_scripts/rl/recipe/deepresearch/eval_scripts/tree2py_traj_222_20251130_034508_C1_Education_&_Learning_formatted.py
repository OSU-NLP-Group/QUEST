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
TASK_ID = "ncaa_head_coach_degree"
TASK_DESCRIPTION = "What is the minimum educational degree typically expected for a head football coaching position at NCAA Division I universities?"


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class RequirementExtraction(BaseModel):
    """
    Extracts what the answer claims is the minimum expected educational degree,
    and any URLs provided as supporting sources.
    """
    min_degree: Optional[str] = None
    supporting_sources: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_requirements() -> str:
    return """
    Extract the following information exactly as presented in the answer:
    1) min_degree: The minimum educational degree that the answer claims is typically expected for head football coaching positions at NCAA Division I universities. Return the exact phrase from the answer (e.g., "bachelor's degree", "BA/BS", "undergraduate degree", "four-year degree", "baccalaureate", etc.). If multiple degrees are mentioned (e.g., bachelor's required, master's preferred), return the minimum expected degree that the answer claims is typically expected. If the answer does not clearly state a minimum expected degree, return null.
    2) supporting_sources: An array of all URLs explicitly cited in the answer that are intended to support the educational requirement claim. Only include actual URLs (plain or in markdown). If no URLs are provided, return an empty array.
    """


# --------------------------------------------------------------------------- #
# Verification builder                                                        #
# --------------------------------------------------------------------------- #
async def build_verification_tree(
    evaluator: Evaluator,
    extraction: RequirementExtraction,
) -> None:
    """
    Build the verification tree according to the rubric:
    - Root (critical, parallel)
      - Degree_Identification (critical leaf): Checks the answer claims bachelor's degree is the minimum.
      - Source_Reference (critical leaf): Verifies at least one provided URL supports the bachelor's-minimum claim.
    """
    # Create a critical parent node to mirror the rubric "Root"
    overall = evaluator.add_parallel(
        id="root_task",
        desc="Correctly identifies the minimum educational degree typically expected for a head football coaching position at NCAA Division I universities",
        parent=evaluator.root,
        critical=True,
    )

    # Leaf 1: Degree Identification (critical)
    degree_node = evaluator.add_leaf(
        id="degree_identification",
        desc="States that a bachelor's degree is the minimum educational degree typically expected",
        parent=overall,
        critical=True,
    )

    # This simple verification checks the answer content itself (no external sources)
    # It should pass only if the answer clearly indicates a bachelor's degree (or equivalent BA/BS/undergraduate/baccalaureate/four-year degree) is the minimum typically expected.
    degree_claim = (
        "The answer explicitly states that a bachelor's degree (or equivalent BA/BS, undergraduate degree, "
        "baccalaureate, or four-year degree) is the minimum educational degree typically expected for "
        "NCAA Division I head football coaching positions."
    )
    await evaluator.verify(
        claim=degree_claim,
        node=degree_node,
        additional_instruction=(
            "Judge ONLY what the answer claims. Accept synonyms or equivalent wording for bachelor's degree "
            "(e.g., BA, BS, baccalaureate, four-year degree, undergraduate degree). "
            "Mentions like 'master's preferred' are fine as long as bachelor's is presented as the minimum expected. "
            "If the answer does not clearly state bachelor's (or equivalent) as the minimum, mark as incorrect."
        ),
    )

    # Leaf 2: Source Reference (critical)
    # If there are no URLs, directly fail this leaf via a custom node
    if not extraction.supporting_sources:
        evaluator.add_custom_node(
            result=False,
            id="source_reference",
            desc="Provides a verifiable source or reference supporting the educational requirement",
            parent=overall,
            critical=True,
        )
    else:
        source_node = evaluator.add_leaf(
            id="source_reference",
            desc="Provides a verifiable source or reference supporting the educational requirement",
            parent=overall,
            critical=True,
        )

        # Verify by the provided URLs that at least one supports the bachelor's-minimum expectation
        source_claim = (
            "A bachelor's degree is typically the minimum educational degree expected for NCAA Division I head football coach positions."
        )

        await evaluator.verify(
            claim=source_claim,
            node=source_node,
            sources=extraction.supporting_sources,
            additional_instruction=(
                "Pass if ANY provided URL clearly supports that a bachelor's degree is the minimum expected requirement. "
                "Accept job postings, HR pages, or official university/athletics postings for NCAA Division I head football coach roles "
                "that state 'Bachelor's degree required' (or similar phrasing like BA/BS required, undergraduate degree required). "
                "It is acceptable if the page also says 'master's preferred'; the key is that bachelor's is the minimum. "
                "If the link is irrelevant (different role/sport/level) or does not support the claim, it should not count."
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
    Entry point for evaluating an answer to the NCAA Division I head football coach minimum degree question.
    """
    # Initialize evaluator with a parallel root (rubric root is parallel)
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

    # Extract structured information from the answer
    extraction = await evaluator.extract(
        prompt=prompt_extract_requirements(),
        template_class=RequirementExtraction,
        extraction_name="requirements_extraction",
    )

    # Optionally record expected ground truth notion (for reference only)
    evaluator.add_ground_truth(
        {
            "expected_minimum_degree": "Bachelor's degree (BA/BS, undergraduate degree, baccalaureate, or equivalent) is typically the minimum for NCAA Division I head football coach positions.",
        },
        gt_type="ground_truth",
    )

    # Build and run verifications
    await build_verification_tree(evaluator, extraction)

    # Return summary
    return evaluator.get_summary()