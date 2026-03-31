import asyncio
import logging
from typing import Optional, List, Dict, Any

import openai
from pydantic import BaseModel, Field

from mind2web2.evaluator import Evaluator
from mind2web2.verification_tree import VerificationNode, AggregationStrategy
from mind2web2.utils.cache_filesys import CacheFileSys

# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "marijuana_caregiver"
TASK_DESCRIPTION = """
Your task is to research and compile information about medical marijuana caregiver laws across any 10 different U.S. states. For each state, find the following specific information:

1. The maximum number of patients a single caregiver can legally serve
2. The minimum age requirement to become a caregiver
3. Whether the caregiver must be a resident of the state
4. Whether caregivers are allowed to grow marijuana plants on behalf of patients
5. If growing is allowed, the maximum number of plants per patient
"""

# Expected number of states
EXPECTED_NUM_STATES = 10

# --------------------------------------------------------------------------- #
# Data models for extraction                                                 #
# --------------------------------------------------------------------------- #
class StateInfo(BaseModel):
    """Model for storing information about a single state's marijuana caregiver laws."""
    state_name: Optional[str] = None
    max_patients: Optional[str] = None
    min_age: Optional[str] = None
    residency_required: Optional[str] = None
    growing_allowed: Optional[str] = None
    max_plants: Optional[str] = None

class StateNames(BaseModel):
    """Model for extracting the list of state names."""
    states: List[str] = Field(default_factory=list)

class ProvLinks(BaseModel):
    """Model for extracting URLs/links."""
    links: List[str] = Field(default_factory=list)

# --------------------------------------------------------------------------- #
# Extraction prompts                                                         #
# --------------------------------------------------------------------------- #
def prompt_extract_state_names() -> str:
    """Prompt to extract just the state names mentioned in the answer."""
    return """
    Extract a list of all U.S. state names mentioned in the answer that have information about medical marijuana caregiver laws.
    
    Return a list of state names as they appear in the text. Do not include territories, districts, or non-state entities.
    If the same state is mentioned multiple times, include it only once.
    
    The state names should be extracted exactly as written in the answer.
    """

def prompt_extract_state_info(state_name: str) -> str:
    """Prompt to extract detailed information about a specific state."""
    return f"""
    Extract specific information about medical marijuana caregiver laws for {state_name} from the answer.
    
    For {state_name}, extract the following fields:
    
    1. max_patients: The maximum number of patients a single caregiver can legally serve in {state_name}
    2. min_age: The minimum age requirement to become a caregiver in {state_name}
    3. residency_required: Whether the caregiver must be a resident of {state_name} (extract as "Yes", "No", or similar clear indication)
    4. growing_allowed: Whether caregivers are allowed to grow marijuana plants on behalf of patients in {state_name} (extract as "Yes", "No", or similar clear indication)
    5. max_plants: If growing is allowed, the maximum number of plants per patient in {state_name}
    
    If any information is not explicitly mentioned in the answer, return null for that field.
    """

def prompt_extract_urls_for_state_field(state_name: str, field_name: str, field_value: str) -> str:
    """Prompt to extract URLs cited for a specific field of a state."""
    return f"""
    Extract all URLs/links that are specifically cited or referenced in the answer when discussing the {field_name} for {state_name}.
    
    The answer states that for {state_name}, the {field_name} is: {field_value}
    
    Return all URLs that potentially support this specific claim. If there are no URLs specifically associated with this claim, return an empty list.
    """


async def verify_field_with_provenance(
        evaluator: Evaluator,
        parent_node: VerificationNode,
        state_name: str,
        field_name: str,
        field_value: Optional[str],
        field_display_name: str,
) -> None:
    """
    Verify a specific field for a state with proper provenance checking.
    """
    # Create parallel wrapper for field verification
    field_wrapper = evaluator.add_parallel(
        id=f"{field_name}_{state_name}_wrapper",
        desc=f"Verification of {field_display_name.lower()} for {state_name}",
        parent=parent_node,
        critical=False
    )
    
    # Add existence check
    existence_check = evaluator.add_custom_node(
        result=bool(field_value) and bool(field_value.strip() != ""),
        id=f"{field_name}_{state_name}_exists",
        desc=f"{field_display_name} information exists for {state_name}",
        parent=field_wrapper,
        critical=True
    )
    
    # Create verification node
    verification_node = evaluator.add_leaf(
        id=f"{field_name}_{state_name}_verify",
        desc=f"{field_display_name} for {state_name}: {field_value or 'None'}",
        parent=field_wrapper,
        critical=True
    )
    
    # Create human-readable claim for verification
    claim = f"In {state_name}, the {field_display_name.lower()} is {field_value}."
    
    # Extract URLs that support this specific claim
    urls_extraction = await evaluator.extract(
        prompt=prompt_extract_urls_for_state_field(state_name, field_display_name.lower(), field_value or ""),
        template_class=ProvLinks,
        extraction_name=f"urls_{field_name}_{state_name}"
    )
    
    # Always verify, passing the URLs directly
    if urls_extraction.links and len(urls_extraction.links) > 0:
        await evaluator.verify(
            claim=claim,
            node=verification_node,
            sources=urls_extraction.links if urls_extraction and hasattr(urls_extraction, 'links') else [],
            additional_instruction=f"Verify the {field_display_name.lower()} for {state_name}"
        )
    else:
        # no url to verify this
        verification_node.status = "failed"
        verification_node.score = 0.0


async def verify_state_information(
        evaluator: Evaluator,
        parent_node: VerificationNode,
        state_info: StateInfo,
        index: int
) -> None:
    """
    Verify all required information fields for a single state.
    """
    # Create parent node for this state
    state_node = evaluator.add_parallel(
        id=f"state_{index}",
        desc=f"Information about medical marijuana caregiver laws in {state_info.state_name or f'State #{index+1}'}",
        parent=parent_node,
        critical=False
    )
    
    # Verify each field with proper provenance
    await verify_field_with_provenance(
        evaluator=evaluator,
        parent_node=state_node,
        state_name=state_info.state_name or f"State #{index+1}",
        field_name="max_patients",
        field_value=state_info.max_patients,
        field_display_name="Maximum number of patients a caregiver can serve",
    )
    
    await verify_field_with_provenance(
        evaluator=evaluator,
        parent_node=state_node,
        state_name=state_info.state_name or f"State #{index+1}",
        field_name="min_age",
        field_value=state_info.min_age,
        field_display_name="Minimum age requirement to become a caregiver",
    )
    
    await verify_field_with_provenance(
        evaluator=evaluator,
        parent_node=state_node,
        state_name=state_info.state_name or f"State #{index+1}",
        field_name="residency_required",
        field_value=state_info.residency_required,
        field_display_name="Residency requirement for caregivers",
    )
    
    await verify_field_with_provenance(
        evaluator=evaluator,
        parent_node=state_node,
        state_name=state_info.state_name or f"State #{index+1}",
        field_name="growing_allowed",
        field_value=state_info.growing_allowed,
        field_display_name="Whether caregivers can grow marijuana plants",
    )
    
    # If growing is not allowed, max_plants is not applicable
    growing_allowed_lower = state_info.growing_allowed.lower() if state_info.growing_allowed else ""
    if growing_allowed_lower and ("no" in growing_allowed_lower or "not" in growing_allowed_lower):
        # Create a placeholder verification node for max_plants that automatically passes
        field_name = "max_plants"
        state_name = state_info.state_name or f"State #{index+1}"
        field_display_name="Maximum plants a caregiver can grow per patient"
        field_wrapper = evaluator.add_parallel(
            id=f"{field_name}_{state_name}_wrapper",
            desc=f"Verification of {field_display_name.lower()} for {state_name}",
            parent=state_node,
            critical=False
        )
        existence_check = evaluator.add_custom_node(
            result=True,
            id=f"{field_name}_{state_name}_exists",
            desc=f"{field_display_name} is not applicable for {state_name} (Automatically pass)",
            parent=field_wrapper,
            critical=True
        )
        verification_node = evaluator.add_custom_node(
            result=True,
            id=f"{field_name}_{state_name}_verify",
            desc=f"{field_display_name} for {state_name} is not applicable (Automatically pass)",
            parent=field_wrapper,
            critical=True
        )
    else:
        # Verify max_plants normally
        await verify_field_with_provenance(
            evaluator=evaluator,
            parent_node=state_node,
            state_name=state_info.state_name or f"State #{index+1}",
            field_name="max_plants",
            field_value=state_info.max_plants,
            field_display_name="Maximum plants a caregiver can grow per patient",
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
    Evaluate a single answer for the marijuana caregiver laws task and return a structured result dictionary.
    """
    # -------- 1. Initialize evaluator ----------------------------------- #
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
    
    # -------- 2. Extract state names first ------------------------------ #
    state_names_extraction = await evaluator.extract(
        prompt=prompt_extract_state_names(),
        template_class=StateNames,
        extraction_name="state_names"
    )
    
    # Get the list of state names and limit to expected number
    state_names = state_names_extraction.states if hasattr(state_names_extraction, 'states') else []
    
    # Limit to the first 10 states if more are provided
    if len(state_names) > EXPECTED_NUM_STATES:
        state_names = state_names[:EXPECTED_NUM_STATES]
    
    # -------- 3. Extract detailed info for each state ------------------- #
    all_states_info = []
    
    for state_name in state_names:
        logger.info(f"Extracting information for {state_name}")
        # Extract state-specific information
        state_info = await evaluator.extract(
            prompt=prompt_extract_state_info(state_name),
            template_class=StateInfo,
            extraction_name=f"state_info_{state_name}"
        )
        
        # Ensure state name is set
        if not state_info.state_name:
            state_info.state_name = state_name
        
        all_states_info.append(state_info)
    
    # Pad missing states with empty StateInfo objects
    while len(all_states_info) < EXPECTED_NUM_STATES:
        all_states_info.append(StateInfo())
    
    # -------- 4. Build verification tree -------------------------------- #
    # Verify each state's information (including empty states)
    for i, state_info in enumerate(all_states_info):
        await verify_state_information(
            evaluator=evaluator,
            parent_node=root,
            state_info=state_info,
            index=i
        )
    
    # -------- 5. Aggregate score and return result --------------------- #
    final_score = evaluator.score()
    logger.info(f"Final score: {final_score}")
    
    # Add custom info about states found
    evaluator.add_custom_info(
        {
            "states_found": len(state_names),
            "expected_states": EXPECTED_NUM_STATES,
            "state_names": state_names
        },
        "state_summary"
    )
    
    return evaluator.get_summary()