import asyncio
import logging
from typing import Optional, List, Dict

import openai
from pydantic import BaseModel, Field

from mind2web2.evaluator import Evaluator, AggregationStrategy
from mind2web2.utils.cache_filesys import CacheFileSys

# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "car_insurance"
TASK_DESCRIPTION = """
Identify the top five U.S. states with the highest total number of automobile registrations. For each of these states, research and summarize the minimum car insurance requirements, specifically including:
- Liability coverage: Bodily Injury Liability and Property Damage Liability.
- Personal Injury Protection (PIP): Indicate whether PIP coverage is mandatory or optional in each state. If mandatory, specify the minimum required coverage.
- Uninsured Motorist (UM) Coverage: Indicate whether UM coverage is mandatory or optional in each state. If mandatory, specify the minimum required coverage.
**Note:** Some states may use slightly different terminology for certain types of coverage. Please ensure that functionally equivalent coverage is identified even if the exact term differs.
"""

JUDGE_MODEL = "o4-mini"


# --------------------------------------------------------------------------- #
# Data models for extracted information                                        #
# --------------------------------------------------------------------------- #
class RankedState(BaseModel):
    rank: int
    state_name: str


class StatesRankingInfo(BaseModel):
    ranked_states: List[RankedState] = Field(default_factory=list)
    source_urls: List[str] = Field(default_factory=list)


class StateDetailedInfo(BaseModel):
    liability_coverage: Optional[str]
    pip_coverage: Optional[str]
    um_coverage: Optional[str]
    relevant_urls: List[str] = Field(default_factory=list)  # URLs specific to this state's insurance requirements


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_states_ranking() -> str:
    return """
    Extract the following information about the top 5 U.S. states with highest automobile registrations:
    
    1. ranked_states: Extract states mentioned as having the highest automobile registrations.
       - If explicit rankings are given (e.g., "#1 California", "Texas ranks 2nd"), extract both rank and state_name
       - If states are listed in order without explicit rankings (e.g., "the top five states are California, Texas, Florida..."), 
         assign ranks based on the order presented (first mentioned = rank 1, second = rank 2, etc.)
       - For each state create an entry with:
         * rank: The numerical ranking position (1 for highest, 2 for second highest, etc.)
         * state_name: The name of the state
    
    2. source_description: Extract any description of the source or statistics used to determine these top 5 states.
    
    3. source_urls: Extract any URLs that are cited as sources for the automobile registration statistics/ranking.
    
    IMPORTANT: 
    - If the answer says "the five states with highest registrations are X, Y, Z, A, B" in that order, treat X as rank 1, Y as rank 2, etc.
    - Common patterns to recognize: "top five states are...", "states with highest registrations are...", "ranked from highest to lowest...", etc.
    """


def prompt_extract_state_details(state_name: str) -> str:
    return f"""
    Extract detailed insurance requirement information for {state_name} from the answer.

    For liability_coverage: Extract the complete description including both bodily injury liability and property damage liability requirements with specific amounts, exactly as described in the answer for {state_name}.

    For pip_coverage: Extract the complete description of Personal Injury Protection (PIP) coverage for {state_name}, clearly indicating whether it is mandatory or optional. If mandatory, include the minimum required coverage amounts. Extract exactly as presented in the answer.

    For um_coverage: Extract the complete description of Uninsured Motorist (UM) coverage for {state_name}, clearly indicating whether it is mandatory or optional. If mandatory, include the minimum required coverage amounts. Extract exactly as presented in the answer.

    For relevant_urls: Extract all URLs that are specifically cited for {state_name}'s insurance requirements in the answer.

    Extract the information exactly as presented in the answer, preserving all details about coverage amounts, mandatory/optional status, and specific requirements for {state_name}.
    """


# --------------------------------------------------------------------------- #
# Verification functions                                                      #
# --------------------------------------------------------------------------- #
async def verify_states_ranking(
        evaluator: Evaluator,
        parent_node,
        states_ranking: StatesRankingInfo,
) -> None:
    """
    Step 1: Verify that the answer provides a source for the automobile registration statistics.
    """
    # Check if source information exists
    source_exists = evaluator.add_custom_node(
        result=bool(states_ranking.source_urls),
        id="ranking_source_exists",
        desc="Check if source information for automobile registration statistics was provided",
        parent=parent_node,
        critical=True
    )

    # Verify source credibility if URLs provided
    source_node = evaluator.add_leaf(
        id="ranking_source_verification",
        desc="Verification that the source for automobile registration statistics is credible and supports the ranking",
        parent=parent_node,
        critical=True
    )

    # Build a dictionary of rank -> state for easier lookup
    rank_to_state = {rs.rank: rs.state_name for rs in states_ranking.ranked_states}
    
    # Build the ranked list showing what was provided
    ranked_states = []
    for i in range(1, 6):
        if i in rank_to_state:
            ranked_states.append(f"#{i}: {rank_to_state[i]}")
        else:
            ranked_states.append(f"#{i}: [not provided in answer]")
    
    source_claim = f"The provided source contains automobile registration statistics by U.S. state and confirms the following ranking: {', '.join(ranked_states)}"
    
    await evaluator.verify(
        claim=source_claim,
        node=source_node,
        sources=states_ranking.source_urls,
        additional_instruction="Verify that the source confirms the ranking for the states that were provided. Note that some ranking positions may not have been provided in the answer - only verify the positions that were explicitly stated."
    )


async def verify_single_state(
        evaluator: Evaluator,
        parent_node,
        rank: int,
        state_name: Optional[str],
) -> None:
    """
    Verify insurance requirements for a single state at a specific rank.
    """
    # Create state node (always create for all ranks)
    state_id = f"state_rank_{rank}"
    if state_name:
        state_display_name = state_name.replace(" ", "_").lower()
        state_id += f"_{state_display_name}"
        state_desc = f"Verification of insurance requirements for {state_name} (rank #{rank})"
    else:
        state_desc = f"Verification of insurance requirements for state at rank #{rank} (not provided)"

    state_node = evaluator.add_parallel(
        id=state_id,
        desc=state_desc,
        parent=parent_node,
        critical=False
    )

    # Critical existence check for state name
    state_exists = evaluator.add_custom_node(
        result=bool(state_name),
        id=f"{state_id}_exists",
        desc=f"Check if state at rank #{rank} was provided in the answer",
        parent=state_node,
        critical=True
    )

    # Extract detailed information for this specific state
    # This will only execute if state_exists passes due to short-circuit logic
    state_details = await evaluator.extract(
        prompt=prompt_extract_state_details(state_name if state_name else ""),
        template_class=StateDetailedInfo,
        extraction_name=f"state_rank_{rank}_details"
    )

    # 1. Liability coverage verification (non-critical intermediate node)
    liability_coverage_node = evaluator.add_parallel(
        id=f"{state_id}_liability_coverage",
        desc=f"Liability coverage verification for {state_name}",
        parent=state_node,
        critical=False  # Non-critical as requested
    )

    liability_exists = evaluator.add_custom_node(
        result=bool(state_details.liability_coverage) and bool(state_details.relevant_urls),
        id=f"{state_id}_liability_exists",
        desc="Check if liability coverage information and URLs were provided",
        parent=liability_coverage_node,
        critical=True
    )

    liability_node = evaluator.add_leaf(
        id=f"{state_id}_liability_verification",
        desc=f"Liability coverage information for {state_name} is accurate and includes both bodily injury and property damage requirements",
        parent=liability_coverage_node,
        critical=True
    )

    liability_claim = f"For {state_name}, the liability coverage requirements are: {state_details.liability_coverage}"
    await evaluator.verify(
        claim=liability_claim,
        node=liability_node,
        sources=state_details.relevant_urls,
        additional_instruction="Verify that the liability coverage information matches the official requirements and includes both bodily injury liability and property damage liability. Note: Some states may use slightly different terminology for certain types of coverage. Please ensure that functionally equivalent coverage is identified even if the exact term differs."
    )

    # 2. PIP coverage verification (non-critical intermediate node)
    pip_coverage_node = evaluator.add_parallel(
        id=f"{state_id}_pip_coverage",
        desc=f"PIP coverage verification for {state_name}",
        parent=state_node,
        critical=False  # Non-critical as requested
    )

    pip_exists = evaluator.add_custom_node(
        result=bool(state_details.pip_coverage) and bool(state_details.relevant_urls),
        id=f"{state_id}_pip_exists",
        desc="Check if PIP coverage information and URLs were provided",
        parent=pip_coverage_node,
        critical=True
    )

    pip_node = evaluator.add_leaf(
        id=f"{state_id}_pip_verification",
        desc=f"PIP coverage information for {state_name} accurately indicates mandatory/optional status and minimum coverage amounts if applicable",
        parent=pip_coverage_node,
        critical=True
    )

    pip_claim = f"For {state_name}, the Personal Injury Protection (PIP) coverage requirements are: {state_details.pip_coverage}"
    await evaluator.verify(
        claim=pip_claim,
        node=pip_node,
        sources=state_details.relevant_urls,
        additional_instruction="Verify that the PIP coverage information is accurate, clearly indicating whether it is mandatory or optional. If mandatory, verify that minimum required coverage amounts are specified. Note: Some states may use slightly different terminology for certain types of coverage. Please ensure that functionally equivalent coverage is identified even if the exact term differs. For PIP, ensure clarity on whether it is mandatory or optional, and if mandatory, the minimum required coverage amounts."
    )

    # 3. UM coverage verification (non-critical intermediate node)
    um_coverage_node = evaluator.add_parallel(
        id=f"{state_id}_um_coverage",
        desc=f"UM coverage verification for {state_name}",
        parent=state_node,
        critical=False  # Non-critical as requested
    )

    um_exists = evaluator.add_custom_node(
        result=bool(state_details.um_coverage) and bool(state_details.relevant_urls),
        id=f"{state_id}_um_exists",
        desc="Check if UM coverage information and URLs were provided",
        parent=um_coverage_node,
        critical=True
    )

    um_node = evaluator.add_leaf(
        id=f"{state_id}_um_verification",
        desc=f"UM coverage information for {state_name} accurately indicates mandatory/optional status and minimum coverage amounts if applicable",
        parent=um_coverage_node,
        critical=True
    )

    um_claim = f"For {state_name}, the Uninsured Motorist (UM) coverage requirements are: {state_details.um_coverage}"
    await evaluator.verify(
        claim=um_claim,
        node=um_node,
        sources=state_details.relevant_urls,
        additional_instruction="Verify that the UM coverage information is accurate, clearly indicating whether it is mandatory or optional. If mandatory, verify that minimum required coverage amounts are specified. Note: Some states may use slightly different terminology for certain types of coverage. Please ensure that functionally equivalent coverage is identified even if the exact term differs. For UM coverage, ensure clarity on whether it is mandatory or optional, and if mandatory, the minimum required coverage amounts."
    )


# --------------------------------------------------------------------------- #
# Main evaluation entry point                                                 #
# --------------------------------------------------------------------------- #
async def evaluate_answer(
        client: openai.AsyncAzureOpenAI,
        answer: str,
        agent_name: str,
        answer_name: str,
        cache: CacheFileSys,
        semaphore: asyncio.Semaphore,
        logger: logging.Logger,
        model: str = "o4-mini"
) -> Dict:
    """
    Evaluate a single answer and return a structured result dictionary.
    """
    # -------- 1. Set up evaluator ---------------------------------------- #
    evaluator = Evaluator()
    
    # Initialize evaluator with sequential strategy for root (Step 1 then Step 2)
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.SEQUENTIAL,
        agent_name=agent_name,
        answer_name=answer_name,
        client=client,
        task_description=TASK_DESCRIPTION,
        answer=answer,
        global_cache=cache,
        global_semaphore=semaphore,
        logger=logger,
        default_model=model if model else JUDGE_MODEL
    )

    # -------- 2. Step 1: Extract and verify states ranking source -------- #
    states_ranking = await evaluator.extract(
        prompt=prompt_extract_states_ranking(),
        template_class=StatesRankingInfo,
        extraction_name="states_ranking_info"
    )

    # Create Step 1 node for source verification
    step1_node = evaluator.add_parallel(
        id="step1_source_verification",
        desc="Step 1: Verification of automobile registration statistics source",
        parent=root,
        critical=False
    )

    await verify_states_ranking(evaluator, step1_node, states_ranking)

    # -------- 3. Step 2: Verify insurance requirements for each state ---- #
    step2_node = evaluator.add_parallel(
        id="step2_insurance_requirements",
        desc="Step 2: Verification of insurance requirements for top 5 states",
        parent=root,
        critical=False
    )

    # Build a dictionary of rank -> state
    rank_to_state = {rs.rank: rs.state_name for rs in states_ranking.ranked_states}

    # Always create nodes for ranks 1-5
    for rank in range(1, 6):
        state_name = rank_to_state.get(rank)
        await verify_single_state(
            evaluator,
            step2_node,
            rank,
            state_name
        )

    # -------- 4. Return structured result -------------------------------- #
    return evaluator.get_summary()