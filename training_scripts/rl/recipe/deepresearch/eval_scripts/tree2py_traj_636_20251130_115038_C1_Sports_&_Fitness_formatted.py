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
TASK_ID = "jevon_holland_contract"
TASK_DESCRIPTION = """
Jevon Holland signed a contract with the New York Giants in March 2025. What is the total value of his contract, how much money is guaranteed, and what is the seating capacity of his team's home stadium?
"""


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class HollandContractStadiumExtraction(BaseModel):
    """
    Extracted information from the agent's answer about:
    - Player and team context
    - Contract details
    - Stadium details
    - Any cited URLs supporting contract or stadium info
    """
    player_name: Optional[str] = None
    player_position: Optional[str] = None
    team_name: Optional[str] = None

    signing_date_text: Optional[str] = None          # e.g., "March 2025", "Mar 2025", "March 10, 2025"
    contract_length_text: Optional[str] = None       # e.g., "3-year", "three years", "three-year deal"
    total_value_text: Optional[str] = None           # e.g., "$44 million", "$44M", "44,000,000"
    guaranteed_money_text: Optional[str] = None      # e.g., "$24 million guaranteed", "$24M guaranteed"

    contract_source_urls: List[str] = Field(default_factory=list)

    stadium_name: Optional[str] = None               # e.g., "MetLife Stadium"
    stadium_location: Optional[str] = None           # e.g., "East Rutherford, New Jersey"
    stadium_capacity_text: Optional[str] = None      # e.g., "82,500", "82500", "around 82k"

    stadium_source_urls: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_contract_stadium() -> str:
    return """
    Extract the specific information stated in the answer related to:
    1) Player and team context:
       - player_name: The player's full name (e.g., "Jevon Holland"), if stated.
       - player_position: The player's position (e.g., "safety"), if stated.
       - team_name: The team he signed with (e.g., "New York Giants"), if stated.

    2) Contract details:
       - signing_date_text: The stated signing time (month and year is sufficient; e.g., "March 2025").
       - contract_length_text: The stated contract length (e.g., "3-year", "three-year deal", "3 years").
       - total_value_text: The stated total value of the contract (keep it as written, e.g., "$44 million", "$44M", "44,000,000").
       - guaranteed_money_text: The stated guaranteed amount (keep it as written, e.g., "$24 million", "$24M").

    3) Stadium details:
       - stadium_name: The New York Giants' home stadium name (e.g., "MetLife Stadium"), if stated.
       - stadium_location: The stadium location (e.g., "East Rutherford, New Jersey"), if stated.
       - stadium_capacity_text: The stated seating capacity (keep it as written, e.g., "82,500", "82500", "around 82k").

    4) Source URLs explicitly cited in the answer:
       - contract_source_urls: All URLs the answer cites that support the contract facts (signing, length, total value, guaranteed money). Extract actual URLs only.
       - stadium_source_urls: All URLs the answer cites that support stadium facts (name, location, capacity). Extract actual URLs only.

    IMPORTANT:
    - Extract values exactly as they appear in the answer. Do not invent or normalize beyond basic cleaning.
    - If a field is not present in the answer, return null for that field (or an empty array for URLs).
    - For URL fields, include only valid, complete URLs explicitly present in the answer (plain URL or markdown link).
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def _has_content(text: Optional[str]) -> bool:
    return bool(text) and bool(str(text).strip())

def _urls_or_none(urls: Optional[List[str]]) -> Optional[List[str]]:
    if not urls:
        return None
    return urls


# --------------------------------------------------------------------------- #
# Verification subtrees                                                       #
# --------------------------------------------------------------------------- #
async def verify_player_and_team_context(
    evaluator: Evaluator,
    parent_node,
    ext: HollandContractStadiumExtraction,
) -> None:
    """
    Build and verify the 'player_and_team_context' subtree:
    - signed_with_giants
    - player_position_safety
    """
    ctx_node = evaluator.add_parallel(
        id="player_and_team_context",
        desc="Correctly identifies the player and team context from the constraints.",
        parent=parent_node,
        critical=True
    )

    # Leaf: signed_with_giants
    signed_node = evaluator.add_leaf(
        id="signed_with_giants",
        desc="States that Jevon Holland signed with the New York Giants (per constraints).",
        parent=ctx_node,
        critical=True
    )
    signed_claim = "Jevon Holland signed with the New York Giants."
    await evaluator.verify(
        claim=signed_claim,
        node=signed_node,
        sources=_urls_or_none(ext.contract_source_urls),
        additional_instruction=(
            "Verify that the statement is explicitly supported either by the answer or by the provided sources. "
            "Allow wording variations like 'agreed to terms', 'inked a deal', or 'joined' the New York Giants."
        ),
    )

    # Leaf: player_position_safety
    safety_node = evaluator.add_leaf(
        id="player_position_safety",
        desc="States that Jevon Holland is a safety (per constraints).",
        parent=ctx_node,
        critical=True
    )
    safety_claim = "Jevon Holland is a safety."
    await evaluator.verify(
        claim=safety_claim,
        node=safety_node,
        sources=_urls_or_none(ext.contract_source_urls),
        additional_instruction=(
            "Confirm the player's position is safety. Minor wording variations are acceptable "
            "(e.g., 'a safety', 'plays safety')."
        ),
    )


async def verify_contract_details(
    evaluator: Evaluator,
    parent_node,
    ext: HollandContractStadiumExtraction,
) -> None:
    """
    Build and verify the 'contract_details' subtree:
    - contract_signed_march_2025
    - contract_length_3_years
    - contract_total_value
    - guaranteed_money
    """
    contract_node = evaluator.add_parallel(
        id="contract_details",
        desc="Correctly states the constrained details of the March 2025 contract.",
        parent=parent_node,
        critical=True
    )

    # Leaf: contract_signed_march_2025
    signed_time_node = evaluator.add_leaf(
        id="contract_signed_march_2025",
        desc="States that the contract was signed in March 2025 (per constraints).",
        parent=contract_node,
        critical=True
    )
    signed_time_claim = "The contract was signed in March 2025."
    await evaluator.verify(
        claim=signed_time_claim,
        node=signed_time_node,
        sources=_urls_or_none(ext.contract_source_urls),
        additional_instruction=(
            "Focus on the month and year (March 2025). Exact day is not required. "
            "Accept short forms like 'Mar 2025'."
        ),
    )

    # Leaf: contract_length_3_years
    length_node = evaluator.add_leaf(
        id="contract_length_3_years",
        desc="States that the contract is a 3-year deal (per constraints).",
        parent=contract_node,
        critical=True
    )
    length_claim = "The contract length is 3 years."
    await evaluator.verify(
        claim=length_claim,
        node=length_node,
        sources=_urls_or_none(ext.contract_source_urls),
        additional_instruction=(
            "Accept variations like 'three-year deal', '3-year contract', or '3 yr'."
        ),
    )

    # Leaf: contract_total_value
    total_node = evaluator.add_leaf(
        id="contract_total_value",
        desc="Correctly states the total value of Jevon Holland's contract (per constraints).",
        parent=contract_node,
        critical=True
    )
    if _has_content(ext.total_value_text):
        total_claim = f"The total value of Jevon Holland's contract is {ext.total_value_text}."
    else:
        # Fallback: check that the answer provides some total value statement
        total_claim = "The answer specifies the total value of Jevon Holland's contract."
    await evaluator.verify(
        claim=total_claim,
        node=total_node,
        sources=_urls_or_none(ext.contract_source_urls),
        additional_instruction=(
            "If a numeric value is present, verify that the total value matches the sources (if any). "
            "Allow common formatting variants such as '$44 million', '$44M', or '44,000,000'. "
            "If the answer only asserts that a total value is provided without a specific number, "
            "verify the presence of such a statement in the answer."
        ),
    )

    # Leaf: guaranteed_money
    guaranteed_node = evaluator.add_leaf(
        id="guaranteed_money",
        desc="Correctly states the amount of guaranteed money in the contract (per constraints).",
        parent=contract_node,
        critical=True
    )
    if _has_content(ext.guaranteed_money_text):
        guaranteed_claim = f"The guaranteed money in the contract is {ext.guaranteed_money_text}."
    else:
        guaranteed_claim = "The answer specifies the amount of guaranteed money in the contract."
    await evaluator.verify(
        claim=guaranteed_claim,
        node=guaranteed_node,
        sources=_urls_or_none(ext.contract_source_urls),
        additional_instruction=(
            "If a numeric value is present, verify that the guaranteed amount matches the sources (if any). "
            "Allow formatting variants like '$24 million', '$24M', etc. "
            "If the answer only asserts that a guaranteed amount is provided without a number, "
            "verify the presence of such a statement in the answer."
        ),
    )


async def verify_stadium_details(
    evaluator: Evaluator,
    parent_node,
    ext: HollandContractStadiumExtraction,
) -> None:
    """
    Build and verify the 'stadium_details' subtree:
    - home_stadium_metlife
    - stadium_location
    - stadium_capacity
    """
    stadium_node = evaluator.add_parallel(
        id="stadium_details",
        desc="Correctly states the constrained details of the New York Giants' home stadium.",
        parent=parent_node,
        critical=True
    )

    # Leaf: home_stadium_metlife
    home_node = evaluator.add_leaf(
        id="home_stadium_metlife",
        desc="States that the New York Giants play at MetLife Stadium (per constraints).",
        parent=stadium_node,
        critical=True
    )
    home_claim = "The New York Giants play their home games at MetLife Stadium."
    await evaluator.verify(
        claim=home_claim,
        node=home_node,
        sources=_urls_or_none(ext.stadium_source_urls),
        additional_instruction=(
            "Confirm that the New York Giants' home stadium is MetLife Stadium. "
            "Minor wording variations are acceptable."
        ),
    )

    # Leaf: stadium_location
    location_node = evaluator.add_leaf(
        id="stadium_location",
        desc="States that MetLife Stadium is located in East Rutherford, New Jersey (per constraints).",
        parent=stadium_node,
        critical=True
    )
    location_claim = "MetLife Stadium is located in East Rutherford, New Jersey."
    await evaluator.verify(
        claim=location_claim,
        node=location_node,
        sources=_urls_or_none(ext.stadium_source_urls),
        additional_instruction=(
            "Accept 'East Rutherford, NJ' as equivalent to 'East Rutherford, New Jersey'."
        ),
    )

    # Leaf: stadium_capacity
    capacity_node = evaluator.add_leaf(
        id="stadium_capacity",
        desc="Correctly states the seating capacity of MetLife Stadium (per constraints).",
        parent=stadium_node,
        critical=True
    )
    if _has_content(ext.stadium_capacity_text):
        capacity_claim = f"The seating capacity of MetLife Stadium is {ext.stadium_capacity_text}."
    else:
        capacity_claim = "The answer specifies the seating capacity of MetLife Stadium."
    await evaluator.verify(
        claim=capacity_claim,
        node=capacity_node,
        sources=_urls_or_none(ext.stadium_source_urls),
        additional_instruction=(
            "Verify the stated capacity or the presence of a capacity statement in the answer. "
            "Allow minor formatting differences (e.g., '82,500' vs '82500') and phrasing like 'around 82k' "
            "when clearly indicating the official seating capacity."
        ),
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
    model: str = "o4-mini",
) -> Dict:
    """
    Evaluate an answer for the Jevon Holland contract and stadium task.
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
        default_model=model,
    )

    # Extract structured information from the answer
    ext = await evaluator.extract(
        prompt=prompt_extract_contract_stadium(),
        template_class=HollandContractStadiumExtraction,
        extraction_name="contract_stadium_extraction",
    )

    # Add ground truth constraints context (as provided in the task)
    evaluator.add_ground_truth({
        "player": "Jevon Holland",
        "team": "New York Giants",
        "expected_signing_time": "March 2025",
        "expected_contract_length": "3 years",
        "stadium": "MetLife Stadium",
        "stadium_location": "East Rutherford, New Jersey",
        "questions": [
            "Total contract value?",
            "Guaranteed money amount?",
            "Home stadium seating capacity?"
        ]
    }, gt_type="constraints_context")

    # Build top-level critical aggregation node per rubric
    top_node = evaluator.add_parallel(
        id="jevon_holland_information",
        desc="Provides information consistent with the given constraints about Jevon Holland's contract with the New York Giants and the Giants' home stadium.",
        parent=root,
        critical=True
    )

    # Build and verify subtrees
    await verify_player_and_team_context(evaluator, top_node, ext)
    await verify_contract_details(evaluator, top_node, ext)
    await verify_stadium_details(evaluator, top_node, ext)

    # Return summary
    return evaluator.get_summary()