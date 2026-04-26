import asyncio
import logging
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.llm_client.base_client import LLMClient
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy

# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "paris2024_athletics_capacity"
TASK_DESCRIPTION = """
What was the official seating capacity for athletics events at the stadium that hosted track and field competitions during the Paris 2024 Summer Olympics?
"""


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class CapacityExtraction(BaseModel):
    """
    Information we expect from the agent's answer:
    - stadium_name: The stadium identified as hosting Paris 2024 athletics (track & field).
    - athletics_capacity: The stated official seating capacity (as written in the answer; keep formatting like commas).
    - capacity_context: A short phrase from the answer clarifying the configuration (e.g., "athletics", "track and field").
    - sources: All URLs cited in the answer to support the claim(s).
    """
    stadium_name: Optional[str] = None
    athletics_capacity: Optional[str] = None
    capacity_context: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_capacity_info() -> str:
    return """
    Extract the key facts the answer provides about the Paris 2024 Olympic athletics stadium and its capacity.

    Required fields:
    1) stadium_name: The name of the stadium that the answer claims hosted the Paris 2024 track & field (athletics) competitions. If not stated, return null.
    2) athletics_capacity: The official seating capacity value the answer states specifically for athletics events (track & field) at that stadium. 
       - Return the main figure exactly as written in the answer (keep commas or units like "69,000", "70,000", "69k", etc.).
       - If the answer gives a range or approximation, return it as-is (e.g., "about 70,000", "~69,000"). If no capacity is stated, return null.
    3) capacity_context: A short phrase the answer uses to clarify this is the athletics/track-and-field configuration (e.g., "athletics", "track & field", "Olympic athletics"). 
       - If the answer does not clearly state that the capacity is for athletics specifically, return null.
    4) sources: An array of all URLs provided in the answer that are meant to support the stadium identification and/or the capacity figure. 
       - Extract only actual URLs present in the answer (plain links or markdown links). If none are provided, return an empty array.

    Return a single JSON object for these fields. Do not invent information not present in the answer.
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def _clean_str(s: Optional[str]) -> str:
    return (s or "").strip()


def _sources_or_none(sources: Optional[List[str]]) -> Optional[List[str]]:
    if not sources:
        return None
    # basic normalization: strip whitespace; keep only non-empty
    cleaned = [u.strip() for u in sources if isinstance(u, str) and u.strip()]
    return cleaned if cleaned else None


# --------------------------------------------------------------------------- #
# Verification tree construction                                              #
# --------------------------------------------------------------------------- #
async def build_verification_tree(
    evaluator: Evaluator,
    extraction: CapacityExtraction,
) -> None:
    """
    Build and execute the verification tree according to the rubric:
    Paris_2024_Athletics_Stadium_Capacity (critical, parallel)
      ├─ Stadium_Identification (critical, leaf)
      └─ Athletics_Capacity (critical, parallel)
           ├─ Correct_Configuration (critical, leaf)
           └─ Capacity_Value_Correctness (critical, leaf)
    """
    # Create the rubric root (critical, parallel) under the global evaluator root
    rubric_root = evaluator.add_parallel(
        id="p24_capacity_root",
        desc="Provide the official seating capacity for athletics events at the stadium that hosted track and field competitions during the Paris 2024 Summer Olympics",
        parent=evaluator.root,
        critical=True
    )

    # Unpack extracted fields
    stadium_name = _clean_str(extraction.stadium_name)
    athletics_capacity = _clean_str(extraction.athletics_capacity)
    capacity_context = _clean_str(extraction.capacity_context)
    sources = _sources_or_none(extraction.sources)

    # 1) Stadium Identification (critical leaf)
    stadium_leaf = evaluator.add_leaf(
        id="p24_stadium_identification",
        desc="Correctly identifies the stadium that hosted Paris 2024 Olympic track & field competitions (athletics)",
        parent=rubric_root,
        critical=True
    )
    # Construct claim using the extracted stadium name
    # If missing, this will likely fail, which is desired because the answer didn't provide the stadium.
    stadium_claim = (
        f"The stadium that hosted track and field (athletics) competitions during the Paris 2024 Summer Olympics "
        f"was '{stadium_name}'." if stadium_name else
        "The answer does not provide a stadium name, so this claim should be considered not supported."
    )
    await evaluator.verify(
        claim=stadium_claim,
        node=stadium_leaf,
        sources=sources,
        additional_instruction=(
            "Only mark as supported if the webpage(s) explicitly state that the specified stadium hosted "
            "track & field (athletics) events for the Paris 2024 Olympics. If the page is about a different "
            "sport or a different venue, or if the claim lacks a concrete stadium name, mark as not supported."
        )
    )

    # 2) Athletics Capacity (critical, parallel parent)
    capacity_node = evaluator.add_parallel(
        id="p24_athletics_capacity",
        desc="Provides the official seating capacity for athletics events at that stadium",
        parent=rubric_root,
        critical=True
    )

    # 2.a) Correct_Configuration (critical leaf)
    correct_config_leaf = evaluator.add_leaf(
        id="p24_correct_configuration",
        desc="Capacity is explicitly for the athletics/track-and-field configuration (not a different configuration such as football, rugby, or concerts)",
        parent=capacity_node,
        critical=True
    )
    # Build a claim ensuring the capacity is for athletics/track & field, not another configuration
    # If capacity or stadium name missing, assert unsupported.
    if stadium_name and athletics_capacity:
        config_claim = (
            f"The seating capacity figure '{athletics_capacity}' mentioned in the answer refers specifically "
            f"to the athletics/track-and-field configuration at {stadium_name} (not football, rugby, or concerts), "
            f"for the Paris 2024 Olympics."
        )
    else:
        config_claim = (
            "The answer does not provide both a concrete stadium name and a concrete capacity, "
            "so the claim about an athletics-specific capacity is not supported."
        )

    await evaluator.verify(
        claim=config_claim,
        node=correct_config_leaf,
        sources=sources,
        additional_instruction=(
            "Confirm from the cited page(s) that the capacity number corresponds to the athletics or track-and-field "
            "configuration. If the page only mentions capacities for other configurations (e.g., football/rugby ~80,000) "
            "without clearly stating an athletics capacity, mark this as not supported."
        )
    )

    # 2.b) Capacity_Value_Correctness (critical leaf)
    capacity_value_leaf = evaluator.add_leaf(
        id="p24_capacity_value_correctness",
        desc="States the correct official athletics-event seating capacity value for that stadium (numerical figure is correct)",
        parent=capacity_node,
        critical=True
    )
    if stadium_name and athletics_capacity:
        value_claim = (
            f"The official seating capacity for athletics events at {stadium_name} during the Paris 2024 Summer Olympics "
            f"was {athletics_capacity}."
        )
    else:
        value_claim = (
            "The answer does not provide both a concrete stadium name and a concrete athletics capacity value; "
            "therefore the stated capacity cannot be verified against sources."
        )

    await evaluator.verify(
        claim=value_claim,
        node=capacity_value_leaf,
        sources=sources,
        additional_instruction=(
            "Verify that the numeric capacity figure matches what authoritative or official sources state for the "
            "athletics/track-and-field configuration (Paris 2024). Allow minor formatting/rounding variants "
            "(e.g., '69,000' vs '69k' or very close rounded values), but reject numbers that clearly refer to "
            "other configurations (e.g., football 80,000) or that contradict the cited sources."
        )
    )


# --------------------------------------------------------------------------- #
# Main evaluation entry point                                                 #
# --------------------------------------------------------------------------- #
async def evaluate_answer(
    client: LLMClient,
    answer: str,
    agent_name: str,
    answer_name: str,
    cache: CacheFileSys,
    semaphore: asyncio.Semaphore,
    logger: logging.Logger,
    model: str = "o4-mini"
) -> Dict[str, Any]:
    """
    Evaluate an answer for the Paris 2024 athletics stadium capacity task.
    """
    # Initialize evaluator (global wrapper root)
    evaluator = Evaluator()
    evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Wrapper aggregation; actual rubric root is added as a critical child
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
    extraction = await evaluator.extract(
        prompt=prompt_extract_capacity_info(),
        template_class=CapacityExtraction,
        extraction_name="capacity_extraction"
    )

    # Build the rubric tree and run verifications
    await build_verification_tree(evaluator, extraction)

    # Return evaluation summary
    return evaluator.get_summary()