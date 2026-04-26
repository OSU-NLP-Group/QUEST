import asyncio
import logging
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy

# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "yellowstone_winter_shuttle_2025"
TASK_DESCRIPTION = (
    "A family of four is planning a winter trip to Yellowstone National Park and will be flying into "
    "Bozeman-Yellowstone International Airport (BZN). They have already booked lodging at Mammoth Hot Springs Hotel "
    "through Xanterra for their stay from December 20-23, 2025. The family consists of two adults, one 8-year-old child, "
    "and one 2-year-old child.\n\nTheir flight is scheduled to arrive at Bozeman Airport at 12:45 pm on December 20, 2025. "
    "They are considering using the Yellowstone National Park Lodges winter airport shuttle service to travel from the airport "
    "to Mammoth Hot Springs Hotel.\n\nBased on the shuttle service's booking restrictions and pricing structure, answer the following:\n\n"
    "1. Can this family book the same-day airport shuttle for their arrival on December 20, 2025? Explain why or why not based on the shuttle's booking requirements.\n\n"
    "2. What is the total one-way shuttle cost from Bozeman Airport to Mammoth Hot Springs Hotel for this entire family (including all taxes and fees that are part of the shuttle fare)?"
)

# Known policy parameters from the task description/rubric
OPERATING_START = "December 15, 2025"
OPERATING_END = "March 1, 2026"
ARRIVAL_DATE = "December 20, 2025"
ARRIVAL_TIME = "12:45 pm"
CUTOFF_TIME = "1:30 pm"

# Fares (one-way) as specified
ADULT_FARE = 107.78
CHILD_FARE_3_11 = 53.89

EXPECTED_TOTAL_ONE_WAY = round(2 * ADULT_FARE + 1 * CHILD_FARE_3_11, 2)  # 269.45


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class PassengerAssignment(BaseModel):
    adults_count: Optional[int] = None
    child_8_category: Optional[str] = None  # e.g., "child fare (3–11)"
    child_2_category: Optional[str] = None  # e.g., "free", "infant", "under 3 free"


class ShuttlePlanExtraction(BaseModel):
    # Eligibility statement and reasoning
    can_book_same_day: Optional[str] = None  # e.g., "yes", "no", "can", "cannot"
    eligibility_reason: Optional[str] = None

    # Any policy/fare URLs explicitly cited in the answer (for verification by sources)
    policy_urls: List[str] = Field(default_factory=list)

    # Fares used in the answer (as strings to maximize compatibility)
    adult_fare_used: Optional[str] = None
    child_fare_used: Optional[str] = None
    under_3_free_stated: Optional[bool] = None
    fares_include_taxes_fees_stated: Optional[bool] = None

    # Total one-way cost stated in the answer
    total_one_way_cost_stated: Optional[str] = None

    # Passenger fare assignment as stated/used in the answer
    passengers: Optional[PassengerAssignment] = None

    # Optional note about NPS entrance fee not being included
    nps_fee_note_present: Optional[bool] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_shuttle_plan() -> str:
    return """
    Extract the shuttle planning details explicitly stated in the answer. Only extract what is actually present in the answer text.

    Required fields:
    1. can_book_same_day: Does the answer explicitly state whether the family can book the same-day shuttle? Return a short string such as "yes", "no", "can", or "cannot". If not stated, return null.
    2. eligibility_reason: A brief excerpt of the explanation about eligibility (e.g., references to the 1:30 pm arrival cutoff and the Xanterra lodging prerequisite). If not provided, return null.
    3. policy_urls: Extract all URLs (including markdown or plain URLs) that the answer cites for shuttle policies or fares. If none are cited, return an empty list.

    Fare-related fields:
    4. adult_fare_used: The adult one-way fare amount used in the answer (e.g., "$107.78"). If not stated, return null.
    5. child_fare_used: The child (ages 3–11) one-way fare amount used in the answer (e.g., "$53.89"). If not stated, return null.
    6. under_3_free_stated: Does the answer state that under age 3 rides free? Return true/false; if not mentioned, return null.
    7. fares_include_taxes_fees_stated: Does the answer state that the fares already include taxes/fees? Return true/false; if not mentioned, return null.

    Total cost field:
    8. total_one_way_cost_stated: The total one-way cost stated in the answer for the entire family. Extract as a string exactly as it appears (e.g., "$269.45"). If not stated, return null.

    Passenger fare assignment:
    9. passengers: A nested object capturing:
       - adults_count: The number of adults counted in the fare (should be 2 if stated).
       - child_8_category: How the 8-year-old child was categorized for fare (e.g., "child fare (3–11)"). If not stated, return null.
       - child_2_category: How the 2-year-old was categorized (e.g., "free", "under 3 free"). If not stated, return null.

    Optional note:
    10. nps_fee_note_present: Does the answer mention that the National Park Service (NPS) entrance fee is not included in the shuttle price? Return true/false; if not mentioned, return null.
    """


# --------------------------------------------------------------------------- #
# Verification helper functions                                               #
# --------------------------------------------------------------------------- #
async def build_eligibility_tree(
    evaluator: Evaluator,
    parent_node,
    extraction: ShuttlePlanExtraction,
) -> None:
    """
    Build and verify the Shuttle_Eligibility_Determination subtree.
    All checks here are critical.
    """
    elig_node = evaluator.add_parallel(
        id="Shuttle_Eligibility_Determination",
        desc="Determine whether the family can book the same-day shuttle for the stated arrival and justify using the booking requirements.",
        parent=parent_node,
        critical=True,
    )

    # Leaf: Operating_Dates_Check
    op_dates_leaf = evaluator.add_leaf(
        id="Operating_Dates_Check",
        desc=f"Confirm that {ARRIVAL_DATE} falls within the stated shuttle operating window for arrivals ({OPERATING_START}–{OPERATING_END}).",
        parent=elig_node,
        critical=True,
    )
    op_dates_claim = (
        f"{ARRIVAL_DATE} falls within the winter shuttle arrival operating window of {OPERATING_START} through {OPERATING_END} (inclusive)."
    )

    # Leaf: Arrival_Time_Cutoff_Check
    cutoff_leaf = evaluator.add_leaf(
        id="Arrival_Time_Cutoff_Check",
        desc=f"Confirm the flight arrival time ({ARRIVAL_TIME}) is not after the {CUTOFF_TIME} cutoff/check-in time.",
        parent=elig_node,
        critical=True,
    )
    cutoff_claim = (
        f"The flight arrival time of {ARRIVAL_TIME} is not after the {CUTOFF_TIME} cutoff; therefore it satisfies the rule 'do not book if arriving after {CUTOFF_TIME}'."
    )

    # Leaf: Xanterra_Eligibility_Check
    xanterra_leaf = evaluator.add_leaf(
        id="Xanterra_Eligibility_Check",
        desc="Confirm the family meets the shuttle-use prerequisite of having Xanterra lodging or a Xanterra tour booked (they have Xanterra lodging).",
        parent=elig_node,
        critical=True,
    )
    xanterra_claim = (
        "The family meets the prerequisite because they have Xanterra lodging booked at Mammoth Hot Springs Hotel."
    )

    # Leaf: Eligibility_Answer_Stated_With_Why
    stated_leaf = evaluator.add_leaf(
        id="Eligibility_Answer_Stated_With_Why",
        desc="Explicitly state whether the family can book the same-day shuttle and provide a brief explanation grounded in the applicable requirements (time cutoff and Xanterra requirement).",
        parent=elig_node,
        critical=True,
    )

    decision_text = extraction.can_book_same_day or "the answer states eligibility (can/cannot)"
    reason_text = extraction.eligibility_reason or ""
    stated_claim = (
        f"The answer explicitly states whether the family {decision_text} book the same-day shuttle and briefly explains why, "
        f"referencing both the {CUTOFF_TIME} arrival cutoff and their Xanterra lodging."
    )

    await evaluator.batch_verify(
        [
            (
                op_dates_claim,
                None,
                op_dates_leaf,
                "Treat the operating window boundaries as inclusive. Rely on the task description; do not introduce external assumptions."
            ),
            (
                cutoff_claim,
                None,
                cutoff_leaf,
                "Compare the stated arrival time to the cutoff time. Minor time format variations are acceptable; reason about 'not after' logically."
            ),
            (
                xanterra_claim,
                None,
                xanterra_leaf,
                "Use the task description to confirm they have Xanterra lodging and thus meet the prerequisite."
            ),
            (
                stated_claim,
                None,
                stated_leaf,
                f"Check the answer explicitly includes both the decision and a brief reason. Helpful context: Extracted explanation snippet: '{reason_text}'."
            ),
        ]
    )


async def build_cost_tree(
    evaluator: Evaluator,
    parent_node,
    extraction: ShuttlePlanExtraction,
) -> None:
    """
    Build and verify the Shuttle_Cost_Calculation subtree.
    Parent is non-critical to allow an optional non-critical note leaf.
    Critical children ensure correct fare assignment and total.
    """
    cost_node = evaluator.add_parallel(
        id="Shuttle_Cost_Calculation",
        desc="Compute the correct total one-way shuttle fare for the whole family, using the given fare rules and ensuring taxes/fees included in the fare are accounted for.",
        parent=parent_node,
        critical=False,
    )

    # Leaf: Passenger_Fare_Category_Assignment (Critical)
    assign_leaf = evaluator.add_leaf(
        id="Passenger_Fare_Category_Assignment",
        desc="Correctly assign fare categories: 2 adults at adult fare; the 8-year-old as child fare (ages 3–11); the 2-year-old free (under 3).",
        parent=cost_node,
        critical=True,
    )
    assign_claim = (
        "The correct assignment is: 2 adults at the adult fare; the 8‑year‑old at the child fare (ages 3–11); and the 2‑year‑old rides free (under age 3). "
        "Verify the answer aligns with this assignment."
    )

    # Leaf: Correct_Fares_Used (Critical) — verify official fares (using sources if provided)
    fares_leaf = evaluator.add_leaf(
        id="Correct_Fares_Used",
        desc="Use the provided one-way fares: adult $107.78 and child (3–11) $53.89, and apply the 'under 3 rides free' rule (fares include taxes/fees).",
        parent=cost_node,
        critical=True,
    )
    fares_claim = (
        "The official winter airport shuttle one‑way fares are adult $107.78 and child (ages 3–11) $53.89, with children under 3 riding free; "
        "the listed fares include taxes and fees."
    )
    policy_sources = extraction.policy_urls if extraction.policy_urls else []

    # Leaf: Total_One_Way_Sum_Correct (Critical)
    total_leaf = evaluator.add_leaf(
        id="Total_One_Way_Sum_Correct",
        desc="Provide a numeric total one-way cost that correctly sums the applicable fares for all family members.",
        parent=cost_node,
        critical=True,
    )
    expected_total_str = f"${EXPECTED_TOTAL_ONE_WAY:0.2f}"
    total_claim = (
        f"The correct total one‑way shuttle cost for this family (2 adults at ${ADULT_FARE:0.2f}, 1 child at ${CHILD_FARE_3_11:0.2f}, "
        f"and 1 under‑3 child free) is {expected_total_str}."
    )

    # Leaf: NPS_Entrance_Fee_Not_Included_Note (Non-critical)
    nps_leaf = evaluator.add_leaf(
        id="NPS_Entrance_Fee_Not_Included_Note",
        desc="Optionally note that the NPS entrance fee is not included in the shuttle price.",
        parent=cost_node,
        critical=False,
    )
    nps_claim = "The answer notes that the National Park Service (NPS) entrance fee is not included in the shuttle price."

    await evaluator.batch_verify(
        [
            (
                assign_claim,
                None,
                assign_leaf,
                "Confirm the answer assigns fares exactly as specified: adults=2, child age 8 uses child fare (3–11), child age 2 rides free."
            ),
            (
                fares_claim,
                policy_sources,
                fares_leaf,
                "Verify via the cited source(s), if provided, that adult=$107.78, child(3–11)=$53.89, under‑3 rides free, and fares include taxes/fees."
            ),
            (
                total_claim,
                None,
                total_leaf,
                f"Check the answer provides the correct numeric total: {expected_total_str}. Allow minor formatting variations, "
                "but the numeric value must match to the cent."
            ),
            (
                nps_claim,
                None,
                nps_leaf,
                "If the answer mentions NPS entrance fees are separate/not included, pass; otherwise fail."
            ),
        ]
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
    Evaluate an answer for the Yellowstone winter shuttle planning task.
    """
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Parallel aggregation for top-level checks
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
        prompt=prompt_extract_shuttle_plan(),
        template_class=ShuttlePlanExtraction,
        extraction_name="shuttle_plan_extraction",
    )

    # Add ground truth info for transparency
    evaluator.add_ground_truth({
        "operating_window": f"{OPERATING_START} – {OPERATING_END}",
        "arrival": {"date": ARRIVAL_DATE, "time": ARRIVAL_TIME},
        "cutoff_rule": f"Do not book if arriving after {CUTOFF_TIME}",
        "lodging_requirement": "Must have Xanterra lodging or a Xanterra tour booked to use shuttle",
        "fares_one_way": {"adult": "$107.78", "child_3_11": "$53.89", "under_3": "free", "includes_taxes_fees": True},
        "family_composition": {"adults": 2, "child_8": "child fare (3–11)", "child_2": "free (under 3)"},
        "expected_total_one_way": f"${EXPECTED_TOTAL_ONE_WAY:0.2f}"
    })

    # Optionally record extracted snippets to help debugging
    evaluator.add_custom_info({
        "can_book_same_day": extraction.can_book_same_day,
        "eligibility_reason_excerpt": extraction.eligibility_reason,
        "policy_urls": extraction.policy_urls,
        "adult_fare_used": extraction.adult_fare_used,
        "child_fare_used": extraction.child_fare_used,
        "under_3_free_stated": extraction.under_3_free_stated,
        "fares_include_taxes_fees_stated": extraction.fares_include_taxes_fees_stated,
        "total_one_way_cost_stated": extraction.total_one_way_cost_stated,
        "passengers": extraction.passengers.dict() if extraction.passengers else None,
        "nps_fee_note_present": extraction.nps_fee_note_present
    }, info_type="extraction_debug")

    # Build and verify subtrees
    await build_eligibility_tree(evaluator, root, extraction)
    await build_cost_tree(evaluator, root, extraction)

    # Return evaluator summary
    return evaluator.get_summary()