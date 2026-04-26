import asyncio
import logging
from typing import Any, Dict, Optional

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "grand_teton_backcountry_trip_planning_2026"
TASK_DESCRIPTION = (
    "I am planning a backcountry camping trip in Grand Teton National Park for a group of 4 people for 3 consecutive nights, "
    "with the first night starting on July 15, 2026. Please provide the following information:\n\n"
    "1. When (specific date and time) and through which platform can I make an advance reservation for this trip?\n"
    "2. What is the total permit cost for this trip, and what portion of this cost is refundable if I need to cancel?\n"
    "3. Provide the official National Park Service webpage URL that contains detailed information about Grand Teton backcountry camping permits."
)

# Known constraints and fee structure (expected ground truth for evaluation)
EXPECTED_OPENING_DT = "January 7, 2026 at 8:00 AM MST"  # Mountain Time
EXPECTED_PLATFORM = "Recreation.gov"

BASE_PERMIT_FEE_USD = 20  # per permit
NIGHTLY_FEE_PER_PERSON_USD = 7  # per person per night
GROUP_SIZE = 4
NUM_NIGHTS = 3

EXPECTED_TOTAL_COST_USD = BASE_PERMIT_FEE_USD + NIGHTLY_FEE_PER_PERSON_USD * GROUP_SIZE * NUM_NIGHTS  # 20 + 7*4*3 = 104
EXPECTED_REFUNDABLE_USD = NIGHTLY_FEE_PER_PERSON_USD * GROUP_SIZE * NUM_NIGHTS  # 7*4*3 = 84


# --------------------------------------------------------------------------- #
# Extraction models                                                           #
# --------------------------------------------------------------------------- #
class TripReservationExtraction(BaseModel):
    """
    Structured extraction of the answer's key fields. Strings are preferred for robustness.
    """
    reservation_opening_datetime_text: Optional[str] = None
    reservation_platform: Optional[str] = None
    total_permit_cost: Optional[str] = None
    refundable_amount: Optional[str] = None
    refund_policy_text: Optional[str] = None
    official_nps_url: Optional[str] = None


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_trip_details() -> str:
    return """
    From the provided answer, extract the following fields exactly as they appear:

    1. reservation_opening_datetime_text:
       - The specific date and time when advance reservations open for Grand Teton backcountry camping permits.
       - This should be a single string containing both date and time (and any time zone text), taken verbatim from the answer.

    2. reservation_platform:
       - The platform or website used to make advance reservations (e.g., "Recreation.gov"), taken verbatim from the answer.

    3. total_permit_cost:
       - The total stated permit cost for the specified trip (4 people, 3 nights), taken verbatim from the answer (e.g., "$104", "USD 104").

    4. refundable_amount:
       - The portion of the permit cost stated as refundable if canceled, taken verbatim from the answer (e.g., "$84", "84 dollars").

    5. refund_policy_text:
       - A short verbatim snippet from the answer that describes the refund policy (e.g., "base fee is non-refundable; nightly per-person fees refundable if canceled at least 5 days before start date").
       - If no policy is mentioned, return null.

    6. official_nps_url:
       - Extract the official National Park Service URL that contains detailed information about Grand Teton backcountry camping permits.
       - Prefer a URL on the nps.gov/grte domain (Grand Teton National Park).
       - If multiple NPS URLs are present, choose the one most clearly about backcountry camping permits.
       - Return the full URL including protocol.
       - If the answer does not include any such URL, return null.

    Return these fields in a JSON object with the specified keys. If any required information is missing, use null.
    """


# --------------------------------------------------------------------------- #
# Verification functions                                                      #
# --------------------------------------------------------------------------- #
async def verify_reservation_details(evaluator: Evaluator, parent_node, extraction: TripReservationExtraction) -> None:
    """
    Build and verify the Reservation_Details subtree:
    - Opening_Date_and_Time (critical)
    - Reservation_Platform (critical)
    """
    reservation_node = evaluator.add_parallel(
        id="Reservation_Details",
        desc="State when advance reservations open and which platform is used.",
        parent=parent_node,
        critical=True,
    )

    # Leaf: Opening_Date_and_Time
    open_dt_leaf = evaluator.add_leaf(
        id="Opening_Date_and_Time",
        desc="State the advance reservation opening date and time as given in constraints (January 7, 2026 at 8:00 AM MST / Mountain Time).",
        parent=reservation_node,
        critical=True,
    )
    open_dt_claim = (
        "The answer explicitly states that advance reservations open on January 7, 2026 at 8:00 AM MST (Mountain Time)."
    )
    await evaluator.verify(
        claim=open_dt_claim,
        node=open_dt_leaf,
        additional_instruction=(
            "Verify that the answer contains this exact opening schedule or an equivalent phrasing. "
            "Minor formatting differences (e.g., '8 AM', '8:00 a.m.', 'Mountain Time', 'MST') are acceptable as long as the date/time are correct."
        ),
    )

    # Leaf: Reservation_Platform
    platform_leaf = evaluator.add_leaf(
        id="Reservation_Platform",
        desc="Identify Recreation.gov as the reservation platform (per constraints).",
        parent=reservation_node,
        critical=True,
    )
    platform_claim = (
        "The answer identifies Recreation.gov as the platform used to make advance reservations for Grand Teton backcountry camping permits."
    )
    await evaluator.verify(
        claim=platform_claim,
        node=platform_leaf,
        additional_instruction=(
            "Confirm that the answer clearly names 'Recreation.gov' as the reservation platform. "
            "Accept minor casing variations; synonyms or other platforms should not be accepted."
        ),
    )


async def verify_cost_information(evaluator: Evaluator, parent_node, extraction: TripReservationExtraction) -> None:
    """
    Build and verify the Cost_Information subtree:
    - Total_Cost_Computation (critical)
    - Refundable_Portion (critical)
    """
    cost_node = evaluator.add_parallel(
        id="Cost_Information",
        desc="Provide the total permit cost and identify what portion is refundable under the stated policy.",
        parent=parent_node,
        critical=True,
    )

    # Leaf: Total_Cost_Computation
    total_cost_leaf = evaluator.add_leaf(
        id="Total_Cost_Computation",
        desc="Compute the total permit cost correctly using the provided fee structure ($20 per permit + $7 per person per night) for 4 people and 3 nights.",
        parent=cost_node,
        critical=True,
    )
    total_cost_claim = (
        f"For a group of {GROUP_SIZE} people staying {NUM_NIGHTS} consecutive nights, using the fee structure "
        f"of ${BASE_PERMIT_FEE_USD} per permit plus ${NIGHTLY_FEE_PER_PERSON_USD} per person per night, "
        f"the correct total permit cost is ${EXPECTED_TOTAL_COST_USD}. The answer's stated total matches ${EXPECTED_TOTAL_COST_USD}."
    )
    await evaluator.verify(
        claim=total_cost_claim,
        node=total_cost_leaf,
        additional_instruction=(
            "Focus on verifying that the answer's stated total equals the computed value. "
            "Allow currency symbols or formats like 'USD 104'. Do not penalize minor formatting differences."
        ),
    )

    # Leaf: Refundable_Portion
    refundable_leaf = evaluator.add_leaf(
        id="Refundable_Portion",
        desc=(
            "Correctly identify that the base permit fee is non-refundable and that the nightly per-person fees are "
            "refundable if canceled at least 5 days before the permit start date; provide the refundable amount "
            "consistent with the computed nightly fees for 4 people × 3 nights."
        ),
        parent=cost_node,
        critical=True,
    )
    refundable_claim = (
        "The answer correctly states that the $20 base permit fee is non-refundable and the $7 per person per night fees "
        f"are refundable if canceled at least 5 days before the permit start date. For 4 people across 3 nights, "
        f"the refundable portion is ${EXPECTED_REFUNDABLE_USD}, and the answer provides ${EXPECTED_REFUNDABLE_USD} as the refundable amount."
    )
    await evaluator.verify(
        claim=refundable_claim,
        node=refundable_leaf,
        additional_instruction=(
            "Verify both the policy (base fee non-refundable; nightly per-person fees refundable if canceled ≥5 days before start) "
            f"and the computed refundable amount (${EXPECTED_REFUNDABLE_USD}). Accept minor currency formatting differences."
        ),
    )


async def verify_official_source_url(evaluator: Evaluator, parent_node, extraction: TripReservationExtraction) -> None:
    """
    Build and verify the Official_Source_URL leaf:
    - Must be an official NPS page on nps.gov/grte
    - Must contain detailed information about Grand Teton backcountry camping permits
    """
    official_url_leaf = evaluator.add_leaf(
        id="Official_Source_URL",
        desc="Provide an official National Park Service URL on the nps.gov/grte domain that contains detailed information about Grand Teton backcountry camping permits.",
        parent=parent_node,
        critical=True,
    )

    url_to_check = extraction.official_nps_url if extraction and extraction.official_nps_url else None

    official_url_claim = (
        "This webpage is an official National Park Service page on the nps.gov/grte domain and contains detailed information "
        "about Grand Teton backcountry camping permits (including reservations, fees, and cancellation/refund policies)."
    )
    await evaluator.verify(
        claim=official_url_claim,
        node=official_url_leaf,
        sources=url_to_check,
        additional_instruction=(
            "Verify the domain is an official NPS page for Grand Teton (nps.gov/grte) and the page clearly covers backcountry camping permits. "
            "Check the web text and screenshot for terms like 'Backcountry', 'Permits', 'Grand Teton', 'Reservations', 'Fees', and 'Refund'. "
            "If the URL is missing, irrelevant, or not on nps.gov/grte, mark as not supported."
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
    Evaluate an answer for Grand Teton backcountry trip planning reservations, costs, and official source URL.
    """
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Independent criteria; allow partial credit
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

    # Extract key fields from the answer
    extraction = await evaluator.extract(
        prompt=prompt_extract_trip_details(),
        template_class=TripReservationExtraction,
        extraction_name="trip_details_extraction",
    )

    # Add ground truth / expected info for transparency
    evaluator.add_ground_truth(
        {
            "expected_opening_datetime": EXPECTED_OPENING_DT,
            "expected_platform": EXPECTED_PLATFORM,
            "fee_structure": {
                "base_permit_fee_usd": BASE_PERMIT_FEE_USD,
                "nightly_fee_per_person_usd": NIGHTLY_FEE_PER_PERSON_USD,
            },
            "group_size": GROUP_SIZE,
            "num_nights": NUM_NIGHTS,
            "expected_total_cost_usd": EXPECTED_TOTAL_COST_USD,
            "expected_refundable_usd": EXPECTED_REFUNDABLE_USD,
            "refund_policy": "Base $20 fee non-refundable; $7/person/night refundable if canceled ≥5 days before start date.",
        },
        gt_type="expected_constraints",
    )

    # Build and verify subtrees according to rubric
    await verify_reservation_details(evaluator, root, extraction)
    await verify_cost_information(evaluator, root, extraction)
    await verify_official_source_url(evaluator, root, extraction)

    # Return standardized summary
    return evaluator.get_summary()