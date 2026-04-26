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
TASK_ID = "eclipse_2026_03_03"
TASK_DESCRIPTION = (
    "On March 3, 2026, a total lunar eclipse will occur. What is the duration of the totality phase of this eclipse "
    "(in minutes), and which two continents will NOT be able to see this eclipse?"
)


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class EclipseExtraction(BaseModel):
    """
    Information explicitly stated in the answer about the March 3, 2026 total lunar eclipse.
    """
    totality_duration_minutes: Optional[str] = None
    not_visible_continents: List[str] = Field(default_factory=list)

    # Any URLs cited in the answer (overall or claim-specific if differentiated)
    sources_overall: List[str] = Field(default_factory=list)
    sources_duration: List[str] = Field(default_factory=list)
    sources_visibility: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_eclipse_info() -> str:
    return """
    Extract from the answer only what is explicitly stated about the March 3, 2026 total lunar eclipse.

    Fields to extract:
    1) totality_duration_minutes:
       - The duration of the TOTALITY phase, in minutes, as explicitly given in the answer.
       - If the answer gives formats like "59 min", "59 mins", "≈59 minutes", "~59 minutes", or "00:59", convert and return "59".
       - If the answer states something approximate like "about an hour" without a precise 59 minutes, return null.
       - If multiple durations are present, pick the one that is explicitly labeled for totality.

    2) not_visible_continents:
       - The continents the ANSWER claims will NOT be able to see this eclipse. Return a list.
       - Canonicalize continent names to one of:
         ["Africa", "Europe", "Asia", "North America", "South America", "Australia/Oceania", "Antarctica"].
       - Accept synonyms (e.g., "Oceania" -> "Australia/Oceania") during extraction.
       - If the answer implies non-visibility for parts of a continent but not the whole continent, do NOT include it.
       - If none are explicitly claimed as not visible, return an empty list.

    3) sources_overall:
       - All URLs the answer cites (any general source list).

    4) sources_duration:
       - URLs (if any) that the ANSWER explicitly associates with the totality duration claim.

    5) sources_visibility:
       - URLs (if any) that the ANSWER explicitly associates with the visibility (non-visibility) claim.

    Do not invent information. If a field is missing, return null (for a single value) or [] (for lists).
    """


# --------------------------------------------------------------------------- #
# Verification helpers                                                        #
# --------------------------------------------------------------------------- #
def build_duration_claim_for_answer() -> str:
    """
    Formulate a claim that checks whether the answer itself states the correct totality duration.
    """
    return (
        "Within the provided answer text, the duration of the TOTALITY phase for the March 3, 2026 total lunar eclipse "
        "is stated as 59 minutes. Minor textual variants like '59 min', '59 mins', '≈59 minutes', '~59 minutes', "
        "or a time format like '00:59' should be considered equivalent to 59 minutes."
    )


def build_visibility_claim_for_answer() -> str:
    """
    Formulate a claim that checks whether the answer itself lists exactly the two continents:
    Africa and Europe, as not able to see the eclipse.
    """
    return (
        "Within the provided answer text, the two continents that are stated to NOT be able to see the "
        "March 3, 2026 total lunar eclipse are Africa and Europe, and no additional continents are claimed to be "
        "completely not visible. Treat the comparison as order-insensitive. Allow minor wording variants like "
        "'not visible from Africa and Europe' or 'Europe and Africa won't see it'."
    )


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
    model: str = "o4-mini",
) -> Dict:
    """
    Evaluate an answer for the March 3, 2026 total lunar eclipse task.
    Rubric mapping:
      - Totality_Duration (critical): The duration is correctly stated as 59 minutes in the answer.
      - Non_Visibility_Regions (critical): The continents not able to see the eclipse are correctly identified as Africa and Europe.
    """
    # Initialize evaluator (root is a non-critical container by design)
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

    # Extract structured info from the answer (for logging/transparency; verification reads the answer directly)
    extracted = await evaluator.extract(
        prompt=prompt_extract_eclipse_info(),
        template_class=EclipseExtraction,
        extraction_name="eclipse_answer_extraction",
    )

    # Record ground truth (for transparency)
    evaluator.add_ground_truth(
        {
            "expected_totality_duration_minutes": "59",
            "expected_not_visible_continents": ["Africa", "Europe"],
            "eclipse_date": "2026-03-03",
        },
        gt_type="ground_truth",
    )

    # Leaf 1: Totality Duration (critical)
    duration_node = evaluator.add_leaf(
        id="Totality_Duration",
        desc="The duration of the totality phase is correctly stated as 59 minutes",
        parent=root,
        critical=True,
    )
    await evaluator.verify(
        claim=build_duration_claim_for_answer(),
        node=duration_node,
        # Simple logical check against the answer text; do not rely on web evidence here.
        additional_instruction=(
            "Judge ONLY by the text of the provided answer (context supplied). "
            "Confirm that the answer explicitly communicates 59 minutes for TOTALITY. "
            "Accept small textual variants such as '59 min', '59 mins', '≈59 minutes', '~59 minutes', or '00:59'. "
            "Do NOT accept vague statements like 'about an hour' unless 59 is explicitly present or unmistakably implied "
            "as the totality duration. If multiple durations appear, consider the one explicitly tied to TOTALITY."
        ),
    )

    # Leaf 2: Non-Visibility Regions (critical)
    regions_node = evaluator.add_leaf(
        id="Non_Visibility_Regions",
        desc="The regions where the eclipse is NOT visible (Africa and Europe) are correctly identified",
        parent=root,
        critical=True,
    )
    await evaluator.verify(
        claim=build_visibility_claim_for_answer(),
        node=regions_node,
        # Simple logical check against the answer text; do not rely on web evidence here.
        additional_instruction=(
            "Judge ONLY by the text of the provided answer (context supplied). "
            "Confirm that the answer specifies exactly the two continents Africa and Europe as NOT able to see the eclipse. "
            "Treat the check as order-insensitive and accept minor wording variants (e.g., 'not visible from Africa and Europe'). "
            "If additional continents are claimed to be completely not visible, or if either Africa or Europe is missing, mark incorrect. "
            "If the answer mentions partial visibility for some regions but still clearly states that Africa and Europe as continents "
            "will not see it, accept."
        ),
    )

    # Return the evaluation summary
    return evaluator.get_summary()