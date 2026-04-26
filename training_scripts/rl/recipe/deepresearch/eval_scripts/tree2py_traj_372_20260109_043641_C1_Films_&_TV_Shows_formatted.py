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
TASK_ID = "highest_grossing_film_2024"
TASK_DESCRIPTION = "What film was the highest-grossing movie worldwide in 2024? Provide its theatrical release date and the production companies responsible for the film."

EXPECTED_FILM = "Inside Out 2"
EXPECTED_RELEASE_DATE = "June 14, 2024"
EXPECTED_PRODUCTION_COMPANIES = ["Disney", "Pixar"]


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class FilmExtraction(BaseModel):
    film_name: Optional[str] = None
    theatrical_release_date: Optional[str] = None
    production_companies: List[str] = Field(default_factory=list)
    source_urls: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_film_info() -> str:
    return (
        "Extract the specific information the answer provides for the question about the highest-grossing movie worldwide in 2024. "
        "Return a JSON object with the following fields:\n"
        "1) film_name: The film the answer identifies as the highest-grossing movie worldwide in 2024.\n"
        "2) theatrical_release_date: The theatrical release date stated in the answer for that film. Keep the original formatting as written in the answer.\n"
        "3) production_companies: A list of the production companies named in the answer for that film. Keep each as a separate string. Examples of valid names include 'Pixar', 'Pixar Animation Studios', 'Disney', 'Walt Disney Pictures'.\n"
        "4) source_urls: Extract all URLs explicitly mentioned in the answer as sources or references for this information (including plain URLs or markdown links). If none are provided, return an empty list.\n"
        "If any of the fields are missing in the answer, set them to null (for single value fields) or an empty list (for array fields). Do not invent any values."
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
    # Initialize evaluator (root is non-critical by framework design)
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

    # Extract structured information from the answer
    extracted: FilmExtraction = await evaluator.extract(
        prompt=prompt_extract_film_info(),
        template_class=FilmExtraction,
        extraction_name="film_info",
    )

    # Add ground truth information for clarity in the summary
    evaluator.add_ground_truth(
        {
            "expected_film": EXPECTED_FILM,
            "expected_theatrical_release_date": EXPECTED_RELEASE_DATE,
            "expected_production_companies": EXPECTED_PRODUCTION_COMPANIES,
        },
        gt_type="expected_values",
    )

    # Build the critical verification node as per rubric
    main_node = evaluator.add_parallel(
        id="highest_grossing_film_2024",
        desc="Correctly identifies the highest-grossing film worldwide in 2024 with accurate release date and production companies",
        parent=root,
        critical=True,
    )

    # Leaf 1: Film name is Inside Out 2
    film_name_node = evaluator.add_leaf(
        id="film_name",
        desc="The film identified is Inside Out 2",
        parent=main_node,
        critical=True,
    )
    # Verify based on the answer text (simple check that the answer identifies the film as Inside Out 2)
    film_claim = "The answer identifies 'Inside Out 2' as the highest-grossing movie worldwide in 2024."
    await evaluator.verify(
        claim=film_claim,
        node=film_name_node,
        sources=None,
        additional_instruction="Focus on whether the answer explicitly names 'Inside Out 2' as the highest-grossing film worldwide in 2024. Allow minor variants such as 'Inside Out II' or 'Inside Out 2 (2024 film)'.",
    )

    # Leaf 2: Theatrical release date is June 14, 2024
    release_date_node = evaluator.add_leaf(
        id="release_date",
        desc="The release date provided is June 14, 2024",
        parent=main_node,
        critical=True,
    )
    release_claim = "The theatrical release date of Inside Out 2 is June 14, 2024."
    # Prefer verifying with any source URLs provided by the answer; fallback to simple verification if none
    release_sources = extracted.source_urls if extracted.source_urls else None
    await evaluator.verify(
        claim=release_claim,
        node=release_date_node,
        sources=release_sources,
        additional_instruction="Accept formatting variants like 'June 14 2024' or '14 June 2024'. If multiple dates appear on a page (premieres, international releases), consider the US wide theatrical release date for Inside Out 2.",
    )

    # Leaf 3: Production companies are Disney and Pixar
    production_companies_node = evaluator.add_leaf(
        id="production_companies",
        desc="The production companies identified are Disney and Pixar",
        parent=main_node,
        critical=True,
    )
    production_claim = "Inside Out 2 was produced by Disney and Pixar."
    production_sources = extracted.source_urls if extracted.source_urls else None
    await evaluator.verify(
        claim=production_claim,
        node=production_companies_node,
        sources=production_sources,
        additional_instruction="Allow common variants and formal names such as 'Walt Disney Pictures', 'Walt Disney Studios Motion Pictures' (distribution), and 'Pixar Animation Studios'. Treat 'Disney/Pixar' or 'Disney and Pixar' as equivalent.",
    )

    # Return structured evaluation summary
    return evaluator.get_summary()