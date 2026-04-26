import asyncio
import logging
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field
from obj_task_eval.llm_client.base_client import LLMClient

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "largest_broadway_theater_nyc"
TASK_DESCRIPTION = "What is the largest Broadway theater in New York City by seating capacity, and what is its approximate total number of seats?"

EXPECTED_THEATER_NAME = "Gershwin Theatre"
EXPECTED_APPROX_CAPACITY_TEXT = "about 1,933 seats"


# --------------------------------------------------------------------------- #
# Data models for structured extraction                                       #
# --------------------------------------------------------------------------- #
class LargestBroadwayExtraction(BaseModel):
    """
    Structured extraction from the agent's answer.
    """
    theater_name: Optional[str] = None
    approx_total_seats_text: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_largest_broadway() -> str:
    return """
    From the answer, extract the following fields related to the largest Broadway theater in NYC:
    1) theater_name: The theater that the answer identifies as the largest Broadway theater by seating capacity.
    2) approx_total_seats_text: The exact text the answer gives for the theater's approximate total number of seats
       (e.g., "about 1,933", "~1,933", "approximately 1,933 seats", "around 1.9k", "1,930–1,935", etc.).
       Do not normalize; copy it as written in the answer. If missing, return null.
    3) sources: A list of all URLs in the answer that support the identification and/or capacity claims. Extract only
       URLs explicitly present in the answer (including markdown links). If none, return an empty list.
    """


# --------------------------------------------------------------------------- #
# Verification sub-tree construction                                          #
# --------------------------------------------------------------------------- #
async def verify_largest_broadway_theater(
    evaluator: Evaluator,
    parent_node,
    extracted: LargestBroadwayExtraction
) -> None:
    """
    Builds and verifies the rubric tree for the Largest Broadway Theater task.
    """
    # Create the critical parallel node per rubric
    largest_node = evaluator.add_parallel(
        id="Largest_Broadway_Theater",
        desc="Identifies the largest Broadway theater in New York City by seating capacity and provides its approximate total seat count, satisfying all stated constraints.",
        parent=parent_node,
        critical=True
    )

    sources_list: List[str] = extracted.sources if extracted and extracted.sources else []

    # Leaf nodes (all critical under a critical parent)
    node_name = evaluator.add_leaf(
        id="Theater_Name",
        desc="The theater is identified as the Gershwin Theatre.",
        parent=largest_node,
        critical=True
    )
    node_capacity_total = evaluator.add_leaf(
        id="Seating_Capacity_Total",
        desc="Provides the approximate total seating capacity as about 1,933 seats.",
        parent=largest_node,
        critical=True
    )
    node_500_plus = evaluator.add_leaf(
        id="Broadway_Definition_500_Plus",
        desc="Answer is consistent with the constraint that Broadway theaters have 500+ seats (i.e., the chosen theater meets/exceeds 500 seats).",
        parent=largest_node,
        critical=True
    )
    node_operating_nyc = evaluator.add_leaf(
        id="NYC_Operating_Broadway_Venue",
        desc="Answer is consistent with the constraint that the theater is currently operating as a Broadway venue in New York City.",
        parent=largest_node,
        critical=True
    )
    node_total_levels = evaluator.add_leaf(
        id="Capacity_Across_All_Levels",
        desc="Seat count is presented as the total across all seating levels (e.g., orchestra + mezzanine), not a partial-level count.",
        parent=largest_node,
        critical=True
    )

    # Build claims
    # 1) Theater name must be Gershwin Theatre (judge using the answer content; do not pass URLs)
    claim_name = (
        "In the answer, the largest Broadway theater by seating capacity is identified as the Gershwin Theatre "
        "(allow minor naming variations such as 'The Gershwin Theatre' or 'Gershwin Theater')."
    )
    add_ins_name = (
        "Judge based on the answer text only. Pass if the answer clearly names the Gershwin Theatre as the largest "
        "Broadway theater by seating capacity. Allow minor spelling/casing variations (e.g., 'The Gershwin Theatre', "
        "'Gershwin Theater'). Fail if a different theater is named or if no theater is identified."
    )

    # 2) Capacity text should be approximately 1,933 seats (judge using the answer content; do not pass URLs)
    claim_capacity = (
        "The answer provides an approximate total seating capacity around 1,933 seats for the theater "
        "(e.g., 'about 1,933', '~1,933', 'approximately 1.9k', or any wording within roughly ±50 of 1,933)."
    )
    add_ins_capacity = (
        "Judge using the answer text only. Accept phrasing like 'about 1,933', '~1,933', 'around 1.9k', "
        "'approximately 1,930–1,935', '≈1,933', etc. The number does not have to be exactly 1,933 if it clearly "
        "conveys an approximate value near 1,933 (within roughly ±50). Fail if no seat count is given or if the "
        "count is clearly inconsistent (e.g., ~1,600 or ~2,500)."
    )

    # 3) 500+ seats check (prefer using sources if available; otherwise simple)
    # Use the known theater identity (Gershwin Theatre) to verify >= 500 seats
    claim_500_plus = (
        "The Gershwin Theatre has at least 500 seats (i.e., 500+), satisfying the Broadway theater 500-seat criterion."
    )
    add_ins_500_plus = (
        "If URLs are provided, verify using the webpage text or screenshot that the Gershwin Theatre's seating capacity "
        "meets or exceeds 500 seats. If no URLs are provided, rely on the general claim and the answer context. "
        "Minor numerical variations or approximations are acceptable as long as the capacity is clearly ≥500."
    )

    # 4) Currently operating Broadway venue in NYC (prefer using sources if available)
    claim_operating_nyc = (
        "The Gershwin Theatre is a currently operating Broadway theater located in New York City."
    )
    add_ins_operating_nyc = (
        "If URLs are available, check that the page indicates the Gershwin Theatre is a Broadway theater in NYC and "
        "that it is currently operating (i.e., not permanently closed). If no URLs are provided, use the answer context "
        "and common-sense interpretation of the claim."
    )

    # 5) Capacity refers to total across all levels (prefer using sources if available)
    # Use the answer's capacity phrasing plus standard definition of 'seating capacity'
    cap_text = extracted.approx_total_seats_text if extracted and extracted.approx_total_seats_text else ""
    claim_total_levels = (
        f"The seating capacity mentioned for the Gershwin Theatre (e.g., '{cap_text}' if provided in the answer) "
        "refers to the total number of seats across all seating levels (e.g., orchestra + mezzanine), not just a "
        "single section."
    )
    add_ins_total_levels = (
        "If URLs are available, verify that the commonly cited 'seating capacity' for the theater represents the total "
        "theatre capacity across all levels (not a partial orchestra-only or mezzanine-only figure). Many official or "
        "reference pages list 'capacity' as a total; accept that as sufficient. If no URLs are provided, judge whether "
        "the answer presents the number as the total theatre capacity (e.g., uses phrasing like 'seating capacity' or "
        "'total seats' rather than referencing a single section)."
    )

    # Execute verifications; use URLs only where appropriate
    claims_and_sources = [
        (claim_name, None, node_name, add_ins_name),
        (claim_capacity, None, node_capacity_total, add_ins_capacity),
        (claim_500_plus, sources_list if sources_list else None, node_500_plus, add_ins_500_plus),
        (claim_operating_nyc, sources_list if sources_list else None, node_operating_nyc, add_ins_operating_nyc),
        (claim_total_levels, sources_list if sources_list else None, node_total_levels, add_ins_total_levels),
    ]

    await evaluator.batch_verify(claims_and_sources)


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
    Entry point for evaluating an answer against the 'Largest Broadway Theater in NYC' rubric.
    """
    # Initialize evaluator and root
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

    # Extract structured info
    extracted: LargestBroadwayExtraction = await evaluator.extract(
        prompt=prompt_extract_largest_broadway(),
        template_class=LargestBroadwayExtraction,
        extraction_name="largest_broadway_extraction"
    )

    # Add ground truth info (for transparency of expectations)
    evaluator.add_ground_truth(
        {
            "expected_theater_name": EXPECTED_THEATER_NAME,
            "expected_approx_capacity": EXPECTED_APPROX_CAPACITY_TEXT
        },
        gt_type="ground_truth_expectations"
    )

    # Build and run verification sub-tree
    await verify_largest_broadway_theater(evaluator, root, extracted)

    # Return standard summary
    return evaluator.get_summary()