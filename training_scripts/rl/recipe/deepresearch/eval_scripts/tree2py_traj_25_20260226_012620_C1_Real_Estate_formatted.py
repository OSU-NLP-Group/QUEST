import asyncio
import logging
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field
from obj_task_eval.llm_client.base_client import LLMClient

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy

# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "property_tax_extremes_2026"
TASK_DESCRIPTION = "According to 2026 data, which U.S. state has the lowest effective real-estate property tax rate, and which state has the highest? Provide the specific effective tax rate percentage for each state."

# Ground truth expectations
GROUND_TRUTH = {
    "lowest_state": "Hawaii",
    "lowest_rate": "0.27%",
    "highest_state": "New Jersey",
    "highest_rate": "2.23%"
}

# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class PropertyTaxExtremesExtraction(BaseModel):
    """
    Information extracted from the agent's answer about 2026 property tax extremes.
    All fields should be extracted exactly as presented in the answer text.
    """
    lowest_state: Optional[str] = None
    lowest_rate: Optional[str] = None
    lowest_sources: List[str] = Field(default_factory=list)

    highest_state: Optional[str] = None
    highest_rate: Optional[str] = None
    highest_sources: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_property_tax_extremes() -> str:
    return """
    From the answer, extract the following six fields about 2026 U.S. effective real-estate property tax extremes:

    1) lowest_state: The U.S. state identified as having the lowest effective real-estate property tax rate.
    2) lowest_rate: The specific effective tax rate percentage for the lowest state (include the percent sign if present, e.g., "0.27%"; extract exactly as written).
    3) lowest_sources: An array of URLs explicitly cited in the answer that support the lowest state's rate or ranking. Only include actual URLs mentioned; if none are provided, return an empty array.

    4) highest_state: The U.S. state identified as having the highest effective real-estate property tax rate.
    5) highest_rate: The specific effective tax rate percentage for the highest state (include the percent sign if present; extract exactly as written).
    6) highest_sources: An array of URLs explicitly cited in the answer that support the highest state's rate or ranking. Only include actual URLs mentioned; if none are provided, return an empty array.

    Rules:
    - Extract exactly what the answer states. Do not infer or convert units.
    - If multiple candidates are mentioned, pick the first one the answer commits to as the lowest/highest.
    - If any field is missing, set it to null (for strings) or an empty array (for sources).
    - For URLs, return full URLs; handle markdown links appropriately by extracting the URL portion only.
    """


# --------------------------------------------------------------------------- #
# Verification helpers                                                        #
# --------------------------------------------------------------------------- #
async def build_and_verify_lowest(
    evaluator: Evaluator,
    parent_node,
    ext: PropertyTaxExtremesExtraction,
) -> None:
    """
    Build the verification subtree for the lowest property tax state and perform checks.
    """
    lowest_node = evaluator.add_parallel(
        id="Lowest_Property_Tax_State",
        desc="Correctly identify the lowest 2026 effective real-estate property tax state and rate",
        parent=parent_node,
        critical=False  # allow partial credit across subchecks
    )

    # Existence of core info (state + rate) is essential
    has_info_lowest = evaluator.add_custom_node(
        result=bool(ext.lowest_state and ext.lowest_state.strip()) and bool(ext.lowest_rate and ext.lowest_rate.strip()),
        id="lowest_has_info",
        desc="Lowest: Answer provides both the state name and the specific effective tax rate",
        parent=lowest_node,
        critical=True
    )

    # Name match to ground truth (critical)
    match_lowest_state = evaluator.add_leaf(
        id="lowest_state_match",
        desc=f"Lowest: Identified state matches expected '{GROUND_TRUTH['lowest_state']}'",
        parent=lowest_node,
        critical=True
    )
    claim_state_match = f"The identified lowest property tax state '{ext.lowest_state}' refers to the same state as '{GROUND_TRUTH['lowest_state']}'."
    await evaluator.verify(
        claim=claim_state_match,
        node=match_lowest_state,
        additional_instruction="Allow minor naming variations and case differences. Focus on whether the two names denote the same U.S. state."
    )

    # Rate match to ground truth (soft for partial credit)
    match_lowest_rate = evaluator.add_leaf(
        id="lowest_rate_match",
        desc=f"Lowest: Stated effective rate equals expected '{GROUND_TRUTH['lowest_rate']}'",
        parent=lowest_node,
        critical=False
    )
    claim_rate_match = f"The stated lowest effective real-estate property tax rate '{ext.lowest_rate}' equals {GROUND_TRUTH['lowest_rate']} (considering reasonable rounding or formatting)."
    await evaluator.verify(
        claim=claim_rate_match,
        node=match_lowest_rate,
        additional_instruction="Treat values as equivalent if differences are due to minor rounding (e.g., 0.27% vs 0.270%)."
    )

    # Sources presence (soft; used to gate source-grounded verifications)
    has_sources_lowest = evaluator.add_custom_node(
        result=bool(ext.lowest_sources and len(ext.lowest_sources) > 0),
        id="lowest_has_sources",
        desc="Lowest: Answer provides at least one source URL",
        parent=lowest_node,
        critical=False
    )

    # Source-grounded verification: the stated lowest state claim is supported by cited sources
    lowest_state_supported = evaluator.add_leaf(
        id="lowest_state_supported_by_sources",
        desc="Lowest: Cited sources support the claimed lowest state in 2026",
        parent=lowest_node,
        critical=False
    )
    claim_lowest_supported = f"According to the cited sources, {ext.lowest_state} has the lowest effective real-estate property tax rate among U.S. states in 2026."
    await evaluator.verify(
        claim=claim_lowest_supported,
        node=lowest_state_supported,
        sources=ext.lowest_sources,
        additional_instruction="Verify the 2026 context explicitly or implicitly. If sources are for different years or do not support 'lowest', judge as not supported.",
        extra_prerequisites=[has_sources_lowest]
    )

    # Source-grounded verification: the stated rate value is supported by the cited sources
    lowest_rate_supported = evaluator.add_leaf(
        id="lowest_rate_supported_by_sources",
        desc="Lowest: Cited sources support the claimed effective rate value for the lowest state in 2026",
        parent=lowest_node,
        critical=False
    )
    claim_lowest_rate_supported = f"According to the cited sources, the effective real-estate property tax rate for {ext.lowest_state} is {ext.lowest_rate} based on 2026 data."
    await evaluator.verify(
        claim=claim_lowest_rate_supported,
        node=lowest_rate_supported,
        sources=ext.lowest_sources,
        additional_instruction="Confirm the rate value and ensure the data corresponds to 2026 (or explicitly stated for 2026). If the source's year differs, judge as not supported.",
        extra_prerequisites=[has_sources_lowest]
    )


async def build_and_verify_highest(
    evaluator: Evaluator,
    parent_node,
    ext: PropertyTaxExtremesExtraction,
) -> None:
    """
    Build the verification subtree for the highest property tax state and perform checks.
    """
    highest_node = evaluator.add_parallel(
        id="Highest_Property_Tax_State",
        desc="Correctly identify the highest 2026 effective real-estate property tax state and rate",
        parent=parent_node,
        critical=False  # allow partial credit across subchecks
    )

    # Existence of core info (state + rate) is essential
    has_info_highest = evaluator.add_custom_node(
        result=bool(ext.highest_state and ext.highest_state.strip()) and bool(ext.highest_rate and ext.highest_rate.strip()),
        id="highest_has_info",
        desc="Highest: Answer provides both the state name and the specific effective tax rate",
        parent=highest_node,
        critical=True
    )

    # Name match to ground truth (critical)
    match_highest_state = evaluator.add_leaf(
        id="highest_state_match",
        desc=f"Highest: Identified state matches expected '{GROUND_TRUTH['highest_state']}'",
        parent=highest_node,
        critical=True
    )
    claim_state_match = f"The identified highest property tax state '{ext.highest_state}' refers to the same state as '{GROUND_TRUTH['highest_state']}'."
    await evaluator.verify(
        claim=claim_state_match,
        node=match_highest_state,
        additional_instruction="Allow minor naming variations and case differences. Focus on whether the two names denote the same U.S. state."
    )

    # Rate match to ground truth (soft for partial credit)
    match_highest_rate = evaluator.add_leaf(
        id="highest_rate_match",
        desc=f"Highest: Stated effective rate equals expected '{GROUND_TRUTH['highest_rate']}'",
        parent=highest_node,
        critical=False
    )
    claim_rate_match = f"The stated highest effective real-estate property tax rate '{ext.highest_rate}' equals {GROUND_TRUTH['highest_rate']} (considering reasonable rounding or formatting)."
    await evaluator.verify(
        claim=claim_rate_match,
        node=match_highest_rate,
        additional_instruction="Treat values as equivalent if differences are due to minor rounding (e.g., 2.23% vs 2.230%)."
    )

    # Sources presence (soft; used to gate source-grounded verifications)
    has_sources_highest = evaluator.add_custom_node(
        result=bool(ext.highest_sources and len(ext.highest_sources) > 0),
        id="highest_has_sources",
        desc="Highest: Answer provides at least one source URL",
        parent=highest_node,
        critical=False
    )

    # Source-grounded verification: the stated highest state claim is supported by cited sources
    highest_state_supported = evaluator.add_leaf(
        id="highest_state_supported_by_sources",
        desc="Highest: Cited sources support the claimed highest state in 2026",
        parent=highest_node,
        critical=False
    )
    claim_highest_supported = f"According to the cited sources, {ext.highest_state} has the highest effective real-estate property tax rate among U.S. states in 2026."
    await evaluator.verify(
        claim=claim_highest_supported,
        node=highest_state_supported,
        sources=ext.highest_sources,
        additional_instruction="Verify the 2026 context explicitly or implicitly. If sources are for different years or do not support 'highest', judge as not supported.",
        extra_prerequisites=[has_sources_highest]
    )

    # Source-grounded verification: the stated rate value is supported by the cited sources
    highest_rate_supported = evaluator.add_leaf(
        id="highest_rate_supported_by_sources",
        desc="Highest: Cited sources support the claimed effective rate value for the highest state in 2026",
        parent=highest_node,
        critical=False
    )
    claim_highest_rate_supported = f"According to the cited sources, the effective real-estate property tax rate for {ext.highest_state} is {ext.highest_rate} based on 2026 data."
    await evaluator.verify(
        claim=claim_highest_rate_supported,
        node=highest_rate_supported,
        sources=ext.highest_sources,
        additional_instruction="Confirm the rate value and ensure the data corresponds to 2026 (or explicitly stated for 2026). If the source's year differs, judge as not supported.",
        extra_prerequisites=[has_sources_highest]
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
    Evaluate an answer for 2026 U.S. property tax extremes:
    - Lowest effective real-estate property tax state and its rate.
    - Highest effective real-estate property tax state and its rate.
    Returns a standardized summary including the verification tree and score.
    """
    # Initialize evaluator with a parallel root (we add our task-specific root child next)
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

    # Add task-level node under root to reflect rubric hierarchy
    task_root = evaluator.add_parallel(
        id="Property_Tax_Extremes",
        desc="Identify both the U.S. state with the lowest property tax rate and the state with the highest property tax rate in 2026, along with their respective effective real-estate tax rates",
        parent=root,
        critical=False
    )

    # Extract structured info from the answer
    ext = await evaluator.extract(
        prompt=prompt_extract_property_tax_extremes(),
        template_class=PropertyTaxExtremesExtraction,
        extraction_name="property_tax_extremes"
    )

    # Record ground truth info
    evaluator.add_ground_truth(
        {
            "expected_lowest_state": GROUND_TRUTH["lowest_state"],
            "expected_lowest_rate": GROUND_TRUTH["lowest_rate"],
            "expected_highest_state": GROUND_TRUTH["highest_state"],
            "expected_highest_rate": GROUND_TRUTH["highest_rate"],
            "year": 2026
        },
        gt_type="ground_truth"
    )

    # Build and verify subtrees
    await build_and_verify_lowest(evaluator, task_root, ext)
    await build_and_verify_highest(evaluator, task_root, ext)

    # Return evaluation summary
    return evaluator.get_summary()