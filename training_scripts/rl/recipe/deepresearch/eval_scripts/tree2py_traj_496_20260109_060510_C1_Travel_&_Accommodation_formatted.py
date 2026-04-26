import asyncio
import logging
from datetime import date
from calendar import monthrange
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "travel_readiness_france_20260315_20260325"
TASK_DESCRIPTION = (
    "A US citizen is planning a trip to France from March 15, 2026, to March 25, 2026. "
    "Their US passport expires on July 1, 2026. They plan to use their Priority Pass membership "
    "to access an airport lounge on their departure day and will arrive at their Paris hotel at 1:00 PM on March 15. "
    "Based on current travel requirements, answer the following: "
    "1. Does their passport meet the Schengen Area validity requirement for this trip? "
    "2. What two documents must they present to access a Priority Pass lounge? "
    "3. Given that their hotel arrival time is 1:00 PM, would this be considered early check-in at a standard hotel, "
    "and what is the typical standard check-in time range?"
)

DEPARTURE_DATE = date(2026, 3, 25)
PASSPORT_EXPIRY_DATE = date(2026, 7, 1)


# --------------------------------------------------------------------------- #
# Utility functions                                                           #
# --------------------------------------------------------------------------- #
def add_months(d: date, months: int) -> date:
    """Add months to a date, handling month length boundaries."""
    m = d.month - 1 + months
    y = d.year + m // 12
    m = m % 12 + 1
    last_day = monthrange(y, m)[1]
    return date(y, m, min(d.day, last_day))


def compute_schengen_validity_meets(departure: date, expiry: date, months_required: int = 3) -> bool:
    """Compute whether passport expiry meets the Schengen validity rule."""
    threshold = add_months(departure, months_required)
    return expiry >= threshold


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class TravelAnswerExtraction(BaseModel):
    """
    Structured extraction from the agent's answer.
    """
    passport_validity: Optional[str] = None  # Expected canonical forms: "yes" or "no" if explicitly stated
    passport_reasoning: Optional[str] = None

    priority_pass_documents: List[str] = Field(default_factory=list)  # Documents listed for lounge access

    hotel_standard_checkin_range: Optional[str] = None  # e.g., "3:00 PM – 4:00 PM"
    hotel_early_checkin_conclusion: Optional[str] = None  # "yes" if 1:00 PM considered early; "no" otherwise


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_travel_info() -> str:
    return (
        "Extract the following information exactly as presented in the answer:\n"
        "1) passport_validity: If the answer explicitly states whether the passport meets the Schengen Area validity requirement, "
        "return 'yes' or 'no' (lowercase). If not explicitly stated, return null.\n"
        "2) passport_reasoning: The reasoning the answer provides for its conclusion about Schengen validity; if no reasoning, return null.\n"
        "3) priority_pass_documents: The list of documents the answer says are required to access a Priority Pass lounge. "
        "Include each document as a separate string exactly as stated (e.g., 'Priority Pass membership card', 'digital membership card', 'same-day boarding pass'). "
        "If none are stated, return an empty array.\n"
        "4) hotel_standard_checkin_range: The standard hotel check-in time range provided in the answer (e.g., '3:00 PM to 4:00 PM'); if not stated, return null.\n"
        "5) hotel_early_checkin_conclusion: If the answer explicitly concludes whether a 1:00 PM arrival is considered early check-in, "
        "return 'yes' or 'no' (lowercase). If not explicitly stated, return null.\n"
        "Only extract what is explicitly in the answer; do not invent missing information."
    )


# --------------------------------------------------------------------------- #
# Verification subroutines                                                    #
# --------------------------------------------------------------------------- #
async def verify_passport_validity(
    evaluator: Evaluator,
    parent_node,
    extracted: TravelAnswerExtraction,
) -> None:
    """
    Passport validity check group:
    - Existence of conclusion in the answer
    - Ground truth computation (3 months beyond departure)
    - Answer correctness vs ground truth
    - Reasoning provided (existence)
    """
    node = evaluator.add_parallel(
        id="Passport_Validity",
        desc=(
            "Confirm that the passport expiration date satisfies the Schengen Area requirement of being valid for at least "
            "3 months beyond the planned departure date from the EU (March 25, 2026), and provide clear reasoning"
        ),
        parent=parent_node,
        critical=True,
    )

    # Ground truth computation
    meets_rule_truth = compute_schengen_validity_meets(DEPARTURE_DATE, PASSPORT_EXPIRY_DATE, months_required=3)
    departure_plus_3 = add_months(DEPARTURE_DATE, 3)

    # Existence check: conclusion provided
    evaluator.add_custom_node(
        result=(extracted.passport_validity is not None and extracted.passport_validity.strip() != ""),
        id="Passport_Validity_Conclusion_Provided",
        desc="The answer explicitly states whether the passport meets the Schengen validity requirement",
        parent=node,
        critical=True,
    )

    # Ground truth node (critical)
    evaluator.add_custom_node(
        result=meets_rule_truth,
        id="Passport_Validity_GroundTruth",
        desc=(
            f"Ground truth: With departure on {DEPARTURE_DATE.isoformat()} and expiry on {PASSPORT_EXPIRY_DATE.isoformat()}, "
            f"the passport meets the '3 months beyond departure' rule (threshold {departure_plus_3.isoformat()})"
        ),
        parent=node,
        critical=True,
    )

    # Answer correctness vs ground truth
    correctness_leaf = evaluator.add_leaf(
        id="Passport_Validity_Answer_Correct",
        desc="The answer's conclusion about Schengen passport validity matches the correct ground truth",
        parent=node,
        critical=True,
    )
    correct_conclusion_text = "meets" if meets_rule_truth else "does not meet"
    claim = (
        f"The answer's conclusion about Schengen passport validity is correct. "
        f"Given Schengen requires at least 3 months validity beyond the departure date from the area "
        f"(departure {DEPARTURE_DATE.isoformat()}, threshold {departure_plus_3.isoformat()}), "
        f"and the passport expiry is {PASSPORT_EXPIRY_DATE.isoformat()}, the correct conclusion is that it {correct_conclusion_text} the requirement."
    )
    await evaluator.verify(
        claim=claim,
        node=correctness_leaf,
        additional_instruction="Focus solely on whether the answer's stated conclusion aligns with the scenario and the described 3-month rule.",
    )

    # Reasoning provided check
    evaluator.add_custom_node(
        result=(extracted.passport_reasoning is not None and extracted.passport_reasoning.strip() != ""),
        id="Passport_Validity_Reasoning_Provided",
        desc="The answer provides clear reasoning for the Schengen validity conclusion",
        parent=node,
        critical=True,
    )


async def verify_priority_pass_access(
    evaluator: Evaluator,
    parent_node,
    extracted: TravelAnswerExtraction,
) -> None:
    """
    Priority Pass lounge access check group:
    - Existence: documents provided
    - Must include Priority Pass membership proof
    - Must include same-day boarding pass
    """
    node = evaluator.add_parallel(
        id="Priority_Pass_Lounge_Access",
        desc=(
            "Identify both required documents needed to access a Priority Pass lounge: "
            "a valid same-day boarding pass and a Priority Pass membership card (physical or digital)"
        ),
        parent=parent_node,
        critical=True,
    )

    # Existence check: any documents listed
    evaluator.add_custom_node(
        result=(bool(extracted.priority_pass_documents) and len(extracted.priority_pass_documents) >= 1),
        id="Priority_Pass_Documents_Listed",
        desc="The answer lists required document(s) for Priority Pass lounge access",
        parent=node,
        critical=True,
    )

    # Membership card presence
    membership_leaf = evaluator.add_leaf(
        id="Priority_Pass_Membership_Included",
        desc="The answer includes a Priority Pass membership card (physical or digital) as a required document",
        parent=node,
        critical=True,
    )
    membership_claim = (
        "The answer lists a Priority Pass membership credential (e.g., membership card, digital membership card, "
        "or app-based digital membership) as a required document for lounge access."
    )
    await evaluator.verify(
        claim=membership_claim,
        node=membership_leaf,
        additional_instruction=(
            "Accept reasonable synonyms indicating membership proof (e.g., 'membership card', 'digital membership card', "
            "'app membership QR code'). Focus on whether the answer includes this requirement."
        ),
    )

    # Same-day boarding pass presence
    boarding_leaf = evaluator.add_leaf(
        id="Priority_Pass_BoardingPass_Included",
        desc="The answer includes a valid same-day boarding pass as a required document",
        parent=node,
        critical=True,
    )
    boarding_claim = "The answer lists a valid same-day boarding pass as a required document for Priority Pass lounge access."
    await evaluator.verify(
        claim=boarding_claim,
        node=boarding_leaf,
        additional_instruction="Focus on whether the answer includes a boarding pass requirement; minor wording variations are acceptable.",
    )


async def verify_hotel_checkin(
    evaluator: Evaluator,
    parent_node,
    extracted: TravelAnswerExtraction,
) -> None:
    """
    Hotel check-in time check group:
    - Existence: standard check-in range provided
    - Range correctness: 3:00 PM – 4:00 PM typical
    - Ground truth: 1:00 PM is early (relative to standard check-in)
    - Answer states early check-in
    """
    node = evaluator.add_parallel(
        id="Hotel_Check_in_Time",
        desc=(
            "Provide the standard hotel check-in time range (between 3:00 PM and 4:00 PM) and correctly determine whether "
            "the 1:00 PM arrival time would require early check-in"
        ),
        parent=parent_node,
        critical=True,
    )

    # Existence check: range provided
    evaluator.add_custom_node(
        result=(extracted.hotel_standard_checkin_range is not None and extracted.hotel_standard_checkin_range.strip() != ""),
        id="Hotel_CheckIn_Range_Provided",
        desc="The answer provides a standard hotel check-in time range",
        parent=node,
        critical=True,
    )

    # Range correctness (verify the answer states 3–4 PM typical)
    range_leaf = evaluator.add_leaf(
        id="Hotel_CheckIn_Range_Correct",
        desc="The answer specifies a typical standard hotel check-in time range of between 3:00 PM and 4:00 PM",
        parent=node,
        critical=True,
    )
    range_claim = (
        "The answer states that the standard hotel check-in time range is between 3:00 PM and 4:00 PM (around 3–4 PM)."
    )
    await evaluator.verify(
        claim=range_claim,
        node=range_leaf,
        additional_instruction="Allow minor variations like 'around 3 pm' or 'typically 3 pm, sometimes 4 pm'.",
    )

    # Ground truth: 1:00 PM is early relative to standard 3–4 PM
    evaluator.add_custom_node(
        result=True,  # 1:00 PM is earlier than 3:00 PM, so it is early check-in
        id="Hotel_CheckIn_Early_GroundTruth",
        desc="Ground truth: Arriving at 1:00 PM is earlier than typical 3–4 PM check-in, thus considered early check-in",
        parent=node,
        critical=True,
    )

    # Answer states early check-in
    early_leaf = evaluator.add_leaf(
        id="Hotel_CheckIn_Early_Answer",
        desc="The answer correctly indicates that arriving at 1:00 PM would be considered early check-in",
        parent=node,
        critical=True,
    )
    early_claim = "The answer indicates that a 1:00 PM arrival is considered early check-in relative to the standard check-in time."
    await evaluator.verify(
        claim=early_claim,
        node=early_leaf,
        additional_instruction="Focus on whether the answer labels 1:00 PM as early check-in; minor phrasing differences are acceptable.",
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
    Evaluate the travel readiness scenario answer.
    """
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
    extracted = await evaluator.extract(
        prompt=prompt_extract_travel_info(),
        template_class=TravelAnswerExtraction,
        extraction_name="travel_answer_extraction",
    )

    # Add ground truth info for transparency
    schengen_threshold = add_months(DEPARTURE_DATE, 3)
    evaluator.add_ground_truth({
        "schengen_validity_rule_months": 3,
        "departure_date": DEPARTURE_DATE.isoformat(),
        "passport_expiry_date": PASSPORT_EXPIRY_DATE.isoformat(),
        "departure_plus_3_months": schengen_threshold.isoformat(),
        "meets_requirement": compute_schengen_validity_meets(DEPARTURE_DATE, PASSPORT_EXPIRY_DATE, 3),
        "priority_pass_required_docs_expected": [
            "Priority Pass membership card (physical or digital)",
            "valid same-day boarding pass"
        ],
        "hotel_standard_checkin_expected_range": "3:00 PM – 4:00 PM",
        "arrival_time": "1:00 PM",
        "arrival_is_early": True
    })

    # Build the top-level verification node (critical, parallel)
    travel_node = evaluator.add_parallel(
        id="Travel_Readiness_Verification",
        desc="Verify that all travel requirements are met for the specified European trip scenario",
        parent=root,
        critical=True,
    )

    # Run sub-verifications
    await verify_passport_validity(evaluator, travel_node, extracted)
    await verify_priority_pass_access(evaluator, travel_node, extracted)
    await verify_hotel_checkin(evaluator, travel_node, extracted)

    # Return the summary
    return evaluator.get_summary()