import asyncio
import logging
from typing import Optional, List, Dict

from pydantic import BaseModel, Field
from obj_task_eval.llm_client.base_client import LLMClient

from obj_task_eval.evaluator import Evaluator, AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys

# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "lowest_property_tax_2024"
TASK_DESCRIPTION = """
Which U.S. state had the lowest median annual property tax bill in 2024? Provide the name of the state, the median annual property tax bill amount in dollars, and a supporting URL reference from a credible source.
"""

# Current date context: 2026-01-11 (2024 data is historical)

# Valid U.S. states (50 states, excluding territories and D.C.)
VALID_US_STATES = [
    "Alabama", "Alaska", "Arizona", "Arkansas", "California", "Colorado", "Connecticut",
    "Delaware", "Florida", "Georgia", "Hawaii", "Idaho", "Illinois", "Indiana", "Iowa",
    "Kansas", "Kentucky", "Louisiana", "Maine", "Maryland", "Massachusetts", "Michigan",
    "Minnesota", "Mississippi", "Missouri", "Montana", "Nebraska", "Nevada", "New Hampshire",
    "New Jersey", "New Mexico", "New York", "North Carolina", "North Dakota", "Ohio",
    "Oklahoma", "Oregon", "Pennsylvania", "Rhode Island", "South Carolina", "South Dakota",
    "Tennessee", "Texas", "Utah", "Vermont", "Virginia", "Washington", "West Virginia",
    "Wisconsin", "Wyoming"
]

# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class PropertyTaxInfo(BaseModel):
    """Model for extracted property tax information."""
    state_name: Optional[str] = None
    median_annual_bill: Optional[str] = None  # String to handle various formats (e.g., "$600", "600-700")
    source_urls: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_property_tax_info() -> str:
    return """
    Extract the following information from the answer:
    1. state_name: The name of the U.S. state identified as having the lowest median annual property tax bill in 2024
    2. median_annual_bill: The median annual property tax bill amount in dollars (extract as provided, e.g., "$600", "600", etc.)
    3. source_urls: All URL references provided to support the claim about the state and the tax amount
    
    If any information is missing, set it to null or return an empty list for source_urls.
    Extract exactly as stated in the answer.
    """


# --------------------------------------------------------------------------- #
# Main evaluation function                                                    #
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
) -> Dict:
    """
    Evaluate an answer for the lowest property tax state task.
    """
    # -------- 1. Initialize evaluator ----------------------------------- #
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # All criteria evaluated independently
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

    # -------- 2. Extract structured information ------------------------- #
    extracted_info = await evaluator.extract(
        prompt=prompt_extract_property_tax_info(),
        template_class=PropertyTaxInfo,
        extraction_name="property_tax_info",
    )

    # Add ground truth context
    evaluator.add_ground_truth({
        "task": "Identify U.S. state with lowest median annual property tax bill in 2024",
        "required_year": 2024,
        "required_metric": "median annual property tax bill (dollar amount, not percentage)",
        "valid_states_count": 50,
    })

    # -------- 3. Build verification tree -------------------------------- #
    
    # Get extracted values for convenience
    state_name = extracted_info.state_name or "None"
    bill_amount = extracted_info.median_annual_bill or "None"
    source_urls = extracted_info.source_urls

    # Criterion 1: Identifies a valid U.S. state
    valid_state_node = evaluator.add_leaf(
        id="identifies_valid_us_state",
        desc="Identifies an entity that is one of the 50 U.S. states (not a city, territory, or D.C.)",
        parent=root,
        critical=True,
    )

    valid_states_list = ", ".join(VALID_US_STATES)
    claim = f"The entity '{state_name}' is one of the 50 U.S. states (not a city, territory like Puerto Rico, or Washington D.C.)."

    await evaluator.verify(
        claim=claim,
        node=valid_state_node,
        additional_instruction=f"Verify that '{state_name}' is one of these 50 U.S. states: {valid_states_list}. Allow for minor spelling variations or abbreviations (e.g., 'N.Y.' for 'New York'). It must NOT be a city, U.S. territory, or Washington D.C.",
    )

    # Criterion 2: Uses 2024 data
    uses_2024_data_node = evaluator.add_leaf(
        id="uses_2024_data",
        desc="The referenced median property tax bill data is explicitly for the year 2024",
        parent=root,
        critical=True,
    )

    claim = f"The median annual property tax bill data for {state_name} is explicitly for the year 2024."

    await evaluator.verify(
        claim=claim,
        node=uses_2024_data_node,
        sources=source_urls,
        additional_instruction="Verify that the source explicitly states the data is for 2024. The year 2024 should be clearly mentioned in relation to the property tax bill data presented. Do not accept data from other years.",
    )

    # Criterion 3: Uses correct metric
    correct_metric_node = evaluator.add_leaf(
        id="uses_correct_metric",
        desc="Uses the metric 'median annual property tax bill' as a dollar amount (not an effective tax rate percentage)",
        parent=root,
        critical=True,
    )

    claim = f"The data presented for {state_name} is 'median annual property tax bill' expressed as a dollar amount, not as a percentage or effective tax rate."

    await evaluator.verify(
        claim=claim,
        node=correct_metric_node,
        sources=source_urls,
        additional_instruction="Verify that the source provides 'median annual property tax bill' (or equivalent phrasing like 'median property tax' or 'median annual property tax') as a dollar amount (e.g., $600, $1,200), NOT as an effective tax rate percentage (e.g., 0.5%, 1.2%). The metric must be an absolute dollar figure.",
    )

    # Criterion 4: State is the lowest among all 50 states
    lowest_state_node = evaluator.add_leaf(
        id="state_is_lowest_among_all_states",
        desc="For the specified year and metric, the identified state has the lowest value among all 50 U.S. states",
        parent=root,
        critical=True,
    )

    claim = f"For 2024 median annual property tax bill data, {state_name} has the lowest value among all 50 U.S. states."

    await evaluator.verify(
        claim=claim,
        node=lowest_state_node,
        sources=source_urls,
        additional_instruction=f"Verify that {state_name} is explicitly identified as having the LOWEST median annual property tax bill among all 50 U.S. states for 2024. The source should either show comparative rankings/data across states or explicitly state that this state has the lowest amount. Exclude territories and D.C. from comparison.",
    )

    # Criterion 5: Provides median annual bill amount
    amount_provided_node = evaluator.add_leaf(
        id="provides_median_annual_bill_amount",
        desc="Provides the median annual property tax bill amount for the identified state as a numeric dollar figure (USD)",
        parent=root,
        critical=True,
    )

    claim = f"The median annual property tax bill amount for {state_name} in 2024 is {bill_amount}."

    await evaluator.verify(
        claim=claim,
        node=amount_provided_node,
        sources=source_urls,
        additional_instruction=f"Verify that the source provides the median annual property tax bill amount of approximately {bill_amount} (or very close to this amount) for {state_name} in 2024. Allow for minor rounding differences (e.g., $599 vs $600). The amount should be explicitly stated in the source.",
    )

    # Criterion 6: Supporting credible URL
    credible_url_node = evaluator.add_leaf(
        id="supporting_credible_url",
        desc="Includes at least one verifiable, credible source URL that supports both the identified state and the stated 2024 median annual property tax bill amount",
        parent=root,
        critical=True,
    )

    claim = f"The source is credible and verifiable, and it supports the claim that {state_name} had the lowest median annual property tax bill (approximately {bill_amount}) among all 50 U.S. states in 2024."

    await evaluator.verify(
        claim=claim,
        node=credible_url_node,
        sources=source_urls,
        additional_instruction="Verify that: (1) The source is credible - it should be from a government website, established research organization, reputable news outlet, real estate data provider (e.g., Zillow, Realtor.com), or authoritative tax research organization. (2) The source supports BOTH the state identification AND the bill amount for 2024. The page should be relevant and not a broken link.",
    )

    # -------- 4. Return structured result ------------------------------- #
    return evaluator.get_summary()