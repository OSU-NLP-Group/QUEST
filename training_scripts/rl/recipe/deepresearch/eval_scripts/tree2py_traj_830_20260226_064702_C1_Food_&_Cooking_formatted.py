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
TASK_ID = "campbells_green_bean_casserole_creation"
TASK_DESCRIPTION = "Who created Campbell's green bean casserole, in what year, and in which city?"

# Optional reference information (not used for scoring)
REFERENCE_INFO = {
    "creator": "Dorcas Reilly",
    "year": "1955",
    "city": "Camden, New Jersey"
}

# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class CreationExtraction(BaseModel):
    creator: Optional[str] = None
    year: Optional[str] = None
    city: Optional[str] = None
    source_urls: List[str] = Field(default_factory=list)

# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_creation_info() -> str:
    return """
    Extract from the answer the following information about Campbell's green bean casserole, exactly as it appears in the answer:
    1) creator: The person credited with creating (inventing/developing) Campbell's green bean casserole.
    2) year: The year the recipe was created (use the exact year string as written in the answer; do not infer or convert).
    3) city: The city where the recipe was created (if the answer gives "City, State", include the full string as presented).
    4) source_urls: A list of all URLs cited in the answer that are used to support any of the above information. Extract only actual URLs explicitly present in the answer (including markdown links). If none are provided, return an empty list.

    If any field is missing in the answer, return null for that field (or an empty list for source_urls).
    """

# --------------------------------------------------------------------------- #
# Helper                                                                      #
# --------------------------------------------------------------------------- #
def _safe(value: Optional[str], placeholder: str = "<missing>") -> str:
    return value.strip() if value else placeholder

# --------------------------------------------------------------------------- #
# Verification tree construction and checks                                   #
# --------------------------------------------------------------------------- #
async def verify_creation_info(
    evaluator: Evaluator,
    parent_node,
    info: CreationExtraction
) -> None:
    # Top-level critical node (parallel aggregation of sub-claims)
    top_node = evaluator.add_parallel(
        id="Green_Bean_Casserole_Creation_Information",
        desc="Verifies that the answer correctly identifies the creator, year, and city where Campbell's green bean casserole was created",
        parent=parent_node,
        critical=True
    )

    urls = info.source_urls if info and info.source_urls else None

    # Creator verification (critical)
    creator_leaf = evaluator.add_leaf(
        id="Creator_Identification",
        desc="The answer correctly identifies the creator of Campbell's green bean casserole",
        parent=top_node,
        critical=True
    )
    creator_val = _safe(info.creator)
    creator_claim = (
        f"The cited webpage explicitly attributes the creation (or invention/development) of "
        f"Campbell's green bean casserole to '{creator_val}'."
    )
    await evaluator.verify(
        claim=creator_claim,
        node=creator_leaf,
        sources=urls,
        additional_instruction=(
            "Verify that at least one provided URL clearly states that this person is the creator/inventor/developer of "
            "Campbell's green bean casserole. Allow minor name variations (e.g., middle initials). "
            "If the extracted name is '<missing>' or blank, treat the claim as not supported."
        )
    )

    # Year verification (critical)
    year_leaf = evaluator.add_leaf(
        id="Creation_Year",
        desc="The answer correctly identifies the year when the recipe was created",
        parent=top_node,
        critical=True
    )
    year_val = _safe(info.year)
    year_claim = (
        f"The cited webpage explicitly states that Campbell's green bean casserole was created in the year '{year_val}'."
    )
    await evaluator.verify(
        claim=year_claim,
        node=year_leaf,
        sources=urls,
        additional_instruction=(
            "Confirm the page mentions the creation occurred in this year (e.g., 'in 1955'). "
            "Accept phrasings like 'in 1955' or 'circa 1955' only if it clearly refers to the creation date, not a revision or later publication. "
            "If the extracted year is '<missing>' or blank, treat the claim as not supported."
        )
    )

    # City verification (critical)
    city_leaf = evaluator.add_leaf(
        id="Creation_City",
        desc="The answer correctly identifies the city where the recipe was created",
        parent=top_node,
        critical=True
    )
    city_val = _safe(info.city)
    city_claim = (
        f"The cited webpage explicitly states that Campbell's green bean casserole was created in '{city_val}'."
    )
    await evaluator.verify(
        claim=city_claim,
        node=city_leaf,
        sources=urls,
        additional_instruction=(
            "Verify that the page identifies the city where the recipe was created. "
            "Treat 'Camden, NJ' and 'Camden, New Jersey' as equivalent. "
            "If the extracted city is '<missing>' or blank, treat the claim as not supported."
        )
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
    model: str = "o4-mini"
) -> Dict:
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

    extracted = await evaluator.extract(
        prompt=prompt_extract_creation_info(),
        template_class=CreationExtraction,
        extraction_name="creation_info"
    )

    evaluator.add_ground_truth(
        {"reference": REFERENCE_INFO, "note": "Reference only; not used for scoring."},
        gt_type="reference_info"
    )

    await verify_creation_info(evaluator, root, extracted)

    return evaluator.get_summary()