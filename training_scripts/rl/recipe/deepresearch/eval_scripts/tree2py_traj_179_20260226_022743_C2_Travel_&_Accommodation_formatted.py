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
TASK_ID = "atl_frontier_parking_personal_item_2025"
TASK_DESCRIPTION = (
    "A solo traveler is flying Frontier Airlines from Hartsfield-Jackson Atlanta International Airport to Orlando for a Disney World vacation. "
    "They will park their car at the airport on Monday (Day 1) and retrieve it on Friday (Day 5), requiring parking for 5 calendar days. "
    "They have a personal backpack with dimensions of 13.5\"H × 17\"W × 7.5\"D (including all handles and straps).\n\n"
    "Based on Atlanta Airport's parking rates effective May 1, 2025:\n"
    "- Economy: $20 per day\n"
    "- Domestic Park-Ride: $15 per day\n"
    "- Daily: $30 per day\n"
    "- ATL West Deck: $30 per day\n\n"
    "Answer the following:\n"
    "1. Which parking option is the most economical for this 5-day duration, and what is the total parking cost?\n"
    "2. Does the traveler's backpack qualify as a free personal item under Frontier Airlines' policy (maximum dimensions: 14\"H × 18\"W × 8\"D, including handles, wheels, and straps)?"
)

# Ground truth and constants from the task
ATL_PARKING_RATES = {
    "Economy": 20,
    "Domestic Park-Ride": 15,
    "Daily": 30,
    "ATL West Deck": 30,
}
PARKING_OPTIONS_LISTED = ["Economy", "Domestic Park-Ride", "Daily", "ATL West Deck"]
PARKING_DURATION_DAYS = 5

BACKPACK_DIMENSIONS = {
    "height_in": 13.5,
    "width_in": 17.0,
    "depth_in": 7.5,
}
FRONTIER_MAX_DIMENSIONS = {
    "height_in": 14.0,
    "width_in": 18.0,
    "depth_in": 8.0,
}

# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class ParkingExtraction(BaseModel):
    """
    Extract the chosen most economical parking option and any cost details stated in the answer.
    """
    selected_option: Optional[str] = None
    stated_daily_rate: Optional[str] = None
    total_cost: Optional[str] = None
    source_urls: List[str] = Field(default_factory=list)


class PersonalItemExtraction(BaseModel):
    """
    Extract the answer's conclusion regarding Frontier personal item compliance, if stated.
    """
    personal_item_conclusion: Optional[str] = None
    justification: Optional[str] = None


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_parking() -> str:
    return (
        "Identify the most economical on-airport parking option selected in the answer and the cost details.\n"
        "Extract the following fields:\n"
        "1) selected_option: the chosen parking option name (one of: Economy, Domestic Park-Ride, Daily, ATL West Deck), as stated in the answer.\n"
        "2) stated_daily_rate: the per-day rate used in the answer for the selected option (include currency symbol if present), or null if not explicitly stated.\n"
        "3) total_cost: the total parking cost for the 5-day duration as stated in the answer (include currency symbol if present), or null if not explicitly stated.\n"
        "4) source_urls: any URLs cited in the answer specific to parking information; return an empty list if none.\n"
        "Do not invent values. If the answer does not specify a field, return null for that field."
    )


def prompt_extract_personal_item() -> str:
    return (
        "Extract the answer's conclusion regarding whether the backpack qualifies as a free personal item under Frontier's policy.\n"
        "Fields:\n"
        "1) personal_item_conclusion: one of 'yes', 'no', or null if not explicitly concluded.\n"
        "2) justification: a brief phrase or sentence summarizing the reasoning as stated in the answer, or null if not provided."
    )


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def canonicalize_option(option_raw: Optional[str]) -> Optional[str]:
    """
    Map various textual variants to canonical parking option names.
    Returns one of: 'Economy', 'Domestic Park-Ride', 'Daily', 'ATL West Deck', or None if unknown.
    """
    if not option_raw:
        return None
    s = option_raw.strip().lower()
    s = s.replace("-", " ").replace("/", " ").replace("_", " ")
    s = " ".join(s.split())  # collapse whitespace

    if "economy" in s:
        return "Economy"
    if "park" in s and "ride" in s:
        # Accept 'park ride', 'park-ride', 'domestic park ride', etc.
        return "Domestic Park-Ride"
    if "daily" in s:
        return "Daily"
    if "west" in s and "deck" in s:
        return "ATL West Deck"
    # Sometimes 'atl west deck' might be shortened
    if "atl" in s and "deck" in s:
        return "ATL West Deck"
    return None


def format_money(amount: Optional[float]) -> Optional[str]:
    if amount is None:
        return None
    try:
        return f"${amount:,.2f}".replace(".00", "")
    except Exception:
        return None


# --------------------------------------------------------------------------- #
# Verification functions                                                      #
# --------------------------------------------------------------------------- #
async def build_and_verify_parking_selection(
    evaluator: Evaluator,
    parent_node,
    parking_info: ParkingExtraction,
) -> None:
    """
    Build the Parking_Selection sequential node and perform verifications:
    - Lowest_Rate_Option: Selected option has the lowest daily rate among listed on-airport options.
    - Duration_Calculation: Total cost equals 5 × daily rate for the selected option.
    """
    parking_node = evaluator.add_sequential(
        id="Parking_Selection",
        desc="Verifies that the most economical parking option at Hartsfield-Jackson Atlanta International Airport is correctly identified for the trip duration",
        parent=parent_node,
        critical=True,
    )

    # Leaf 1: Lowest_Rate_Option
    lowest_leaf = evaluator.add_leaf(
        id="Lowest_Rate_Option",
        desc="The selected parking option has the lowest daily rate among all available on-airport parking options (Economy, Park-Ride, Daily, and ATL West Deck)",
        parent=parking_node,
        critical=True,
    )

    selected_option_raw = parking_info.selected_option or ""
    selected_option_canonical = canonicalize_option(selected_option_raw)
    # Build claim: pure logical check using the provided rates
    rates_list_str = (
        "Economy $20/day; Domestic Park-Ride $15/day; Daily $30/day; ATL West Deck $30/day"
    )
    claim_lowest = (
        f"Given the on-airport parking rates effective May 1, 2025 — {rates_list_str} — "
        f"the option selected in the answer is '{selected_option_raw}'. "
        f"This selected option has the lowest daily rate among all listed options."
    )
    add_ins_lowest = (
        "Judge based solely on the provided rates and the four listed on-airport options. "
        "Treat 'Park-Ride' variants (e.g., 'Park Ride', 'Domestic Park-Ride') as the same option. "
        "Ignore any off-airport options. If the selected option is 'Domestic Park-Ride', it is indeed the lowest."
    )
    await evaluator.verify(
        claim=claim_lowest,
        node=lowest_leaf,
        additional_instruction=add_ins_lowest,
    )

    # Leaf 2: Duration_Calculation
    duration_leaf = evaluator.add_leaf(
        id="Duration_Calculation",
        desc="The calculated total parking cost equals the product of the 5-day parking duration and the daily rate of the selected parking option",
        parent=parking_node,
        critical=True,
    )

    # Try to compute expected total for the canonical option; otherwise, use a generic formula claim.
    expected_total: Optional[float] = None
    selected_rate: Optional[float] = None
    if selected_option_canonical and selected_option_canonical in ATL_PARKING_RATES:
        selected_rate = ATL_PARKING_RATES[selected_option_canonical]
        expected_total = selected_rate * PARKING_DURATION_DAYS

    stated_daily_rate_str = parking_info.stated_daily_rate or ""
    total_cost_str = parking_info.total_cost or ""

    if expected_total is not None and selected_rate is not None:
        claim_duration = (
            f"For a 5-day stay using the selected option '{selected_option_raw}' at ${selected_rate}/day, "
            f"the correct total is {format_money(expected_total)}. "
            f"The answer states the total cost as '{total_cost_str}'. These should match."
        )
        add_ins_duration = (
            "Focus on per-day pricing with simple multiplication (5 × daily rate). "
            "Allow for minor formatting differences (e.g., presence/absence of commas or cents). "
            "Do not consider taxes or extra fees unless explicitly included in the answer."
        )
    else:
        # Fallback generic claim if selected option not recognized
        claim_duration = (
            f"The total parking cost stated in the answer ('{total_cost_str}') should equal "
            f"5 times the daily rate used ('{stated_daily_rate_str}')."
        )
        add_ins_duration = (
            "Verify the arithmetic based on what the answer claims as the daily rate and total cost. "
            "If the daily rate is missing, you may refer to the rates in the task description."
        )

    await evaluator.verify(
        claim=claim_duration,
        node=duration_leaf,
        additional_instruction=add_ins_duration,
    )


async def build_and_verify_personal_item(
    evaluator: Evaluator,
    parent_node,
) -> None:
    """
    Build the Personal_Item_Compliance parallel node and verify each dimension
    against Frontier's free personal item limits.
    """
    personal_node = evaluator.add_parallel(
        id="Personal_Item_Compliance",
        desc="Verifies that the backpack dimensions comply with Frontier Airlines' free personal item size requirements",
        parent=parent_node,
        critical=True,
    )

    # Height compliance
    height_leaf = evaluator.add_leaf(
        id="Height_Compliance",
        desc="The backpack height does not exceed the maximum allowed height of 14 inches",
        parent=personal_node,
        critical=True,
    )
    claim_height = (
        f"The backpack height is {BACKPACK_DIMENSIONS['height_in']} inches, "
        f"which does not exceed Frontier's maximum of {FRONTIER_MAX_DIMENSIONS['height_in']} inches."
    )
    add_ins_height = (
        "Dimensions include handles and straps. Treat equality as compliant. Allow minor rounding (e.g., 13.5 ≤ 14)."
    )
    await evaluator.verify(
        claim=claim_height,
        node=height_leaf,
        additional_instruction=add_ins_height,
    )

    # Width compliance
    width_leaf = evaluator.add_leaf(
        id="Width_Compliance",
        desc="The backpack width does not exceed the maximum allowed width of 18 inches",
        parent=personal_node,
        critical=True,
    )
    claim_width = (
        f"The backpack width is {BACKPACK_DIMENSIONS['width_in']} inches, "
        f"which does not exceed Frontier's maximum of {FRONTIER_MAX_DIMENSIONS['width_in']} inches."
    )
    add_ins_width = (
        "Dimensions include handles and straps. Treat equality as compliant. Allow minor rounding."
    )
    await evaluator.verify(
        claim=claim_width,
        node=width_leaf,
        additional_instruction=add_ins_width,
    )

    # Depth compliance
    depth_leaf = evaluator.add_leaf(
        id="Depth_Compliance",
        desc="The backpack depth does not exceed the maximum allowed depth of 8 inches",
        parent=personal_node,
        critical=True,
    )
    claim_depth = (
        f"The backpack depth is {BACKPACK_DIMENSIONS['depth_in']} inches, "
        f"which does not exceed Frontier's maximum of {FRONTIER_MAX_DIMENSIONS['depth_in']} inches."
    )
    add_ins_depth = (
        "Dimensions include handles and straps. Treat equality as compliant. Allow minor rounding."
    )
    await evaluator.verify(
        claim=claim_depth,
        node=depth_leaf,
        additional_instruction=add_ins_depth,
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
    Evaluate an answer for the Atlanta airport parking and Frontier personal item compliance task.
    """
    # Initialize evaluator
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # The top-level rubric is parallel
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

    # Record ground truth/task constants for transparency
    evaluator.add_ground_truth({
        "atl_parking_rates_may_1_2025": ATL_PARKING_RATES,
        "parking_duration_days": PARKING_DURATION_DAYS,
        "backpack_dimensions_in": BACKPACK_DIMENSIONS,
        "frontier_max_personal_item_in": FRONTIER_MAX_DIMENSIONS,
    }, gt_type="task_constants")

    # Extract structured information from the answer
    parking_info = await evaluator.extract(
        prompt=prompt_extract_parking(),
        template_class=ParkingExtraction,
        extraction_name="parking_selection",
    )
    personal_info = await evaluator.extract(
        prompt=prompt_extract_personal_item(),
        template_class=PersonalItemExtraction,
        extraction_name="personal_item_conclusion",
    )

    # Build Trip_Planning_Compliance node
    trip_node = evaluator.add_parallel(
        id="Trip_Planning_Compliance",
        desc="Evaluates whether the traveler's planning decisions comply with all requirements and identify the most economical options",
        parent=root,
        critical=False,
    )

    # Build and verify subtrees
    await build_and_verify_parking_selection(evaluator, trip_node, parking_info)
    await build_and_verify_personal_item(evaluator, trip_node)

    # Return structured summary
    return evaluator.get_summary()