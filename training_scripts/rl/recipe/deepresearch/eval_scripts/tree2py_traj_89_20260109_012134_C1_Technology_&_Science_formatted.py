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
TASK_ID = "east_acronym_expansion_task"
TASK_DESCRIPTION = (
    'In January 2026, a Chinese fusion reactor made headlines by breaking through a long-standing plasma density barrier, '
    'entering what scientists call a "density-free regime." This breakthrough was published in Science Advances. '
    'What is the full name that the acronym EAST represents for this tokamak?'
)

EXPECTED_EAST_EXPANSION = "Experimental Advanced Superconducting Tokamak"

# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class EASTInfo(BaseModel):
    """
    Extracted information from the answer related to EAST and the described breakthrough context.
    """
    east_full_name: Optional[str] = None
    china_reference: Optional[str] = None
    january_2026_reference: Optional[str] = None
    density_barrier_reference: Optional[str] = None
    density_free_regime_reference: Optional[str] = None
    science_advances_reference: Optional[str] = None
    source_urls: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_east_info() -> str:
    return """
    You must extract the specific pieces of information from the answer text related to the EAST tokamak and the described breakthrough.

    Extract the following fields:
    1. east_full_name: The explicit full name that the acronym "EAST" expands to (e.g., "Experimental Advanced Superconducting Tokamak"). 
       Do not return the acronym itself ("EAST"); return the full expanded phrase if it is present in the answer. If not present, return null.
    2. china_reference: The exact phrase from the answer that indicates the reactor/tokamak is in China or is Chinese (e.g., "Chinese", "in Hefei, China"). If not present, return null.
    3. january_2026_reference: The exact phrase from the answer that ties the breakthrough/publication to January 2026 
       (e.g., "January 2026", "Jan. 2026"). If not present, return null.
    4. density_barrier_reference: The exact phrase indicating the breakthrough involved surpassing a plasma density barrier/limit 
       (e.g., "surpassed the plasma density limit", "broke the density barrier", "density limit"). If not present, return null.
    5. density_free_regime_reference: The exact phrase indicating entry into a "density-free regime". If not present, return null.
    6. science_advances_reference: The exact phrase indicating that the research was published in Science Advances 
       (e.g., "published in Science Advances"). If not present, return null.
    7. source_urls: Extract all explicit URLs mentioned in the answer text (including markdown links). If none, return an empty list.

    Return a JSON object containing these fields. Do not invent information; only extract what is explicitly present in the answer.
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def _is_full_expansion(text: Optional[str]) -> bool:
    """
    Determine whether the given text appears to be a full expansion of the acronym EAST,
    rather than just the acronym itself.
    """
    if text is None:
        return False
    s = text.strip()
    if not s:
        return False
    # Reject pure acronym or trivial forms
    if s.lower() == "east":
        return False
    # Heuristic: a proper expansion should contain multiple words
    if len(s.split()) >= 2:
        return True
    return False


# --------------------------------------------------------------------------- #
# Verification tree construction                                              #
# --------------------------------------------------------------------------- #
async def build_verification_tree(evaluator: Evaluator, extracted: EASTInfo) -> None:
    """
    Construct and execute the verification checks according to the rubric tree.
    """
    # Create the main critical parallel node to encapsulate all required checks
    task_node = evaluator.add_parallel(
        id="EAST_Acronym_Expansion_Task",
        desc="Evaluate whether the response correctly expands the acronym EAST for the tokamak described and satisfies all stated constraints.",
        parent=evaluator.root,
        critical=True
    )

    # 1) Provides_EAST_Full_Name (critical) - existence/format check using custom node
    evaluator.add_custom_node(
        result=_is_full_expansion(extracted.east_full_name),
        id="Provides_EAST_Full_Name",
        desc="The answer provides the complete full name that the acronym EAST represents (i.e., an explicit expansion of EAST, not just the acronym).",
        parent=task_node,
        critical=True
    )

    # 2) Reactor_Located_in_China (critical) - simple verify against answer text
    node_china = evaluator.add_leaf(
        id="Reactor_Located_in_China",
        desc="The answer indicates the reactor/tokamak is located in China.",
        parent=task_node,
        critical=True
    )
    await evaluator.verify(
        claim="The answer indicates that the reactor/tokamak is located in China or clearly refers to it as Chinese.",
        node=node_china,
        additional_instruction="Accept mentions such as 'Chinese', 'in China', 'Hefei, China', or similar clear indicators."
    )

    # 3) Breakthrough_Timing_January_2026 (critical)
    node_jan2026 = evaluator.add_leaf(
        id="Breakthrough_Timing_January_2026",
        desc="The answer states or clearly ties the referenced breakthrough/publication to January 2026.",
        parent=task_node,
        critical=True
    )
    await evaluator.verify(
        claim="The answer explicitly ties the breakthrough/publication timing to January 2026.",
        node=node_jan2026,
        additional_instruction="Accept reasonable variants such as 'January 2026', 'Jan. 2026', or equivalent phrasing."
    )

    # 4) Surpassed_Plasma_Density_Barrier (critical)
    node_density_barrier = evaluator.add_leaf(
        id="Surpassed_Plasma_Density_Barrier",
        desc="The answer indicates the breakthrough involved surpassing a plasma density barrier/limit.",
        parent=task_node,
        critical=True
    )
    await evaluator.verify(
        claim="The answer indicates that a plasma density barrier/limit was surpassed.",
        node=node_density_barrier,
        additional_instruction="Accept variants like 'density limit', 'density barrier', 'Greenwald limit', or equivalent wording."
    )

    # 5) Entered_Density_Free_Regime (critical)
    node_density_free = evaluator.add_leaf(
        id="Entered_Density_Free_Regime",
        desc="The answer indicates the reactor achieved entry into a 'density-free regime.'",
        parent=task_node,
        critical=True
    )
    await evaluator.verify(
        claim="The answer indicates that the reactor achieved entry into a 'density-free regime'.",
        node=node_density_free,
        additional_instruction="Accept the exact phrase 'density-free regime' or clearly equivalent phrasing indicating such a regime."
    )

    # 6) Published_in_Science_Advances (critical)
    node_science_advances = evaluator.add_leaf(
        id="Published_in_Science_Advances",
        desc="The answer indicates the research was published in the journal Science Advances.",
        parent=task_node,
        critical=True
    )
    await evaluator.verify(
        claim="The answer indicates that the research was published in Science Advances.",
        node=node_science_advances,
        additional_instruction="Explicit mention of 'Science Advances' is sufficient; variants like 'AAAS Science Advances' also acceptable."
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
    Evaluate an answer for the EAST acronym expansion task and required contextual mentions.
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
        default_model=model
    )

    # Extract information from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_east_info(),
        template_class=EASTInfo,
        extraction_name="east_answer_extraction"
    )

    # Record expected expansion as ground truth info (for reference only)
    evaluator.add_ground_truth({
        "expected_east_expansion": EXPECTED_EAST_EXPANSION,
        "notes": "Canonical expansion widely used in literature for EAST."
    }, gt_type="ground_truth_east")

    # Build and run verification tree
    await build_verification_tree(evaluator, extracted)

    # Return structured summary
    return evaluator.get_summary()