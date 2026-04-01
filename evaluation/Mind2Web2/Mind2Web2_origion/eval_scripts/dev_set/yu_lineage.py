import asyncio
import logging
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from mind2web2.utils.cache_filesys import CacheFileSys
from mind2web2.evaluator import Evaluator
from mind2web2.verification_tree import AggregationStrategy

# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "yu_lineage"
TASK_DESCRIPTION = """
Identify the academic lineage of Prof. Yu Su at The Ohio State University by tracing his doctoral advisor and continuing upward through advisor-advisee relationships for five generations (excluding Yu Su himself).
"""

# Ground truth for academic lineage
GROUND_TRUTH_LINEAGE = [
    "Xifeng Yan",  # Generation 1: Yu Su's advisor
    "Jiawei Han",  # Generation 2: Xifeng Yan's advisor
    "Larry E. Travis",  # Generation 3: Jiawei Han's advisor
    "Abraham Robinson",  # Generation 4: Larry E. Travis's advisor
    "Paul Dienes"  # Generation 5: Abraham Robinson's advisor
]


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class AdvisorInfo(BaseModel):
    """Model to represent a single advisor"""
    name: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class AdvisorsExtraction(BaseModel):
    """Model for the extracted academic lineage"""
    generation1: Optional[AdvisorInfo] = None  # Yu Su's advisor
    generation2: Optional[AdvisorInfo] = None  # Advisor's advisor
    generation3: Optional[AdvisorInfo] = None  # Advisor's advisor's advisor
    generation4: Optional[AdvisorInfo] = None  # 4th generation
    generation5: Optional[AdvisorInfo] = None  # 5th generation


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_advisors() -> str:
    return """
    Extract the academic lineage of Prof. Yu Su as presented in the answer. The task requires identifying 5 generations of advisors (excluding Yu Su himself).

    For each generation, extract:
    1. name: The full name of the advisor
    2. sources: URLs or references cited that directly or potentially support this advisor-advisee relationship. For each advisor-advisee relationship and generation, for example, Yu Su to Xifeng Yan, the sources could be urls related to both the advisor (Xifeng Yan) and the advisee (Yu Su), or a page that explicitly states the advisor-advisee relationship. Do not omit any sources that are mentioned in the answer, even if they seem weak or indirect.

    Organize the extraction by generation:
    - generation1: Yu Su's direct doctoral advisor
    - generation2: The advisor of Yu Su's advisor
    - generation3: The advisor of generation2
    - generation4: The advisor of generation3
    - generation5: The advisor of generation4

    For any generation not mentioned in the answer, return null.
    If sources are not provided for a particular advisor, return an empty list for sources.
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def get_advisee_name(generation: int, extracted_info: AdvisorsExtraction) -> str:
    """
    Get the advisee name for a given generation.
    """
    if generation == 1:
        return "Yu Su"
    else:
        prev_gen_attr = f"generation{generation - 1}"
        prev_advisor_info = getattr(extracted_info, prev_gen_attr)
        if prev_advisor_info and prev_advisor_info.name:
            return prev_advisor_info.name
        else:
            return f"Generation {generation - 1} advisor"


# --------------------------------------------------------------------------- #
# Verification functions                                                      #
# --------------------------------------------------------------------------- #
async def verify_generation(
        evaluator: Evaluator,
        parent_node,
        generation: int,
        expected_advisor: str,
        extracted_info: AdvisorsExtraction,
) -> None:
    """
    Verify the advisor for a specific generation.
    Creates all verification nodes regardless of whether the generation was provided.
    """
    gen_attr = f"generation{generation}"
    advisor_info = getattr(extracted_info, gen_attr)

    # Create generation node (non-critical to allow partial scoring between generations)
    gen_node = evaluator.add_parallel(
        id=f"generation_{generation}",
        desc=f"Generation {generation}: The advisor is correctly identified as {expected_advisor} with supporting evidence",
        parent=parent_node,
        critical=False,  # Allow partial scoring between generations
    )

    # Check if advisor exists and has sources
    advisor_exists = advisor_info is not None and advisor_info.name is not None and advisor_info.name.strip()
    sources_exist = advisor_exists and advisor_info.sources and len(advisor_info.sources) > 0
    
    # Step 1: Add existence check as critical node
    existence_check = evaluator.add_custom_node(
        result=advisor_exists and sources_exist,
        id=f"generation_{generation}_exists",
        desc=f"Generation {generation}: Advisor is identified with sources",
        parent=gen_node,
        critical=True  # Critical to gate subsequent verifications
    )

    # Step 2: Verify advisor matches ground truth
    match_node = evaluator.add_leaf(
        id=f"generation_{generation}_match",
        desc=f"Generation {generation}: The identified advisor matches the expected advisor '{expected_advisor}'",
        parent=gen_node,
        critical=True,
    )

    # Always call verify regardless of data existence
    advisor_name = advisor_info.name if advisor_info else ""
    match_claim = f"The name '{advisor_name}' and '{expected_advisor}' refer to the same person."
    
    await evaluator.verify(
        claim=match_claim,
        node=match_node,
        additional_instruction="Consider the names equivalent if they refer to the same person, even if there are minor variations in formatting, middle initials, or spelling. The comparison should be case-insensitive.",
    )

    # Step 3: Verify sources support the advisor-advisee relationship
    source_node = evaluator.add_leaf(
        id=f"generation_{generation}_source",
        desc=f"Generation {generation}: The advisor-advisee relationship is supported by cited sources",
        parent=gen_node,
        critical=True,
    )

    # Get sources list (may be empty)
    sources_list = advisor_info.sources if advisor_info else []
    advisee_name = get_advisee_name(generation, extracted_info)
    relationship_claim = f"{advisor_name} is the doctoral advisor of {advisee_name}."

    # Always call verify with sources
    await evaluator.verify(
        claim=relationship_claim,
        node=source_node,
        sources=sources_list,  # Pass list directly, even if empty
        additional_instruction=f"Verify whether the source confirms that {advisor_name} was the doctoral advisor (or PhD advisor/supervisor) of {advisee_name}. The source should explicitly state or clearly imply this advisor-advisee relationship.",
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
    Evaluate an answer for the Yu Su academic lineage task.
    """
    # Initialize evaluator
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.SEQUENTIAL,  # Sequential to enforce lineage dependency
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

    # Extract advisors from the answer
    extracted_advisors = await evaluator.extract(
        prompt=prompt_extract_advisors(),
        template_class=AdvisorsExtraction,
        extraction_name="academic_lineage",
    )

    # Add ground truth information
    evaluator.add_ground_truth({
        "expected_lineage": GROUND_TRUTH_LINEAGE,
        "task": "Identify 5 generations of academic advisors for Prof. Yu Su"
    })

    # Verify all 5 generations - always create nodes for all generations
    # The sequential strategy will automatically handle dependencies
    for gen in range(1, 6):
        await verify_generation(
            evaluator=evaluator,
            parent_node=root,
            generation=gen,
            expected_advisor=GROUND_TRUTH_LINEAGE[gen - 1],
            extracted_info=extracted_advisors,
        )

    # Return structured result using the evaluator's summary
    return evaluator.get_summary()