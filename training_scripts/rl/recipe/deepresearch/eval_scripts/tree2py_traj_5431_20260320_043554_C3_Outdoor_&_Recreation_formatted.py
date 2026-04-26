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
TASK_ID = "grand_canyon_bac_lottery_2026"
TASK_DESCRIPTION = (
    "You are planning a backpacking trip to Grand Canyon National Park's Bright Angel Campground for June 2026 with a group of 4 people. "
    "Using the Recreation.gov lottery system: (1) What is the earliest date you can submit your lottery application? "
    "(2) What is the total cost to enter the lottery for your group? "
    "(3) Where must you pick up the permit if your lottery application is selected? "
    "(4) After winning the lottery but before your trip date, can you modify your camping itinerary if needed, and under what condition?"
)

GROUND_TRUTH = {
    "expected_lottery_month_for_june_2026": "February 2026",
    "expected_opening_date_for_june_2026": "February 16, 2026",
    "expected_lottery_fee_total_for_group_of_4": "$10 (per application/permit, not per person)",
    "expected_pickup_location": "Backcountry Information Center (in person)",
    "modification_allowed": "Allowed after winning but before trip date",
    "modification_conditions": "Changes must be made in subsequent months (not during the lottery month) and space must be available",
}


# --------------------------------------------------------------------------- #
# Data models                                                                 #
# --------------------------------------------------------------------------- #
class LotteryExtraction(BaseModel):
    # Application timing
    lottery_month_for_trip: Optional[str] = None  # e.g., "February 2026" or "February"
    opening_date_earliest: Optional[str] = None   # e.g., "February 16, 2026" or "Feb 16, 2026"
    earliest_sources: List[str] = Field(default_factory=list)

    # Cost
    lottery_cost_total: Optional[str] = None      # e.g., "$10", "10 USD", "$40 total", etc.
    cost_sources: List[str] = Field(default_factory=list)

    # Permit pickup
    pickup_location: Optional[str] = None         # e.g., "Backcountry Information Center"
    pickup_sources: List[str] = Field(default_factory=list)

    # Modification policy
    modification_allowed: Optional[str] = None    # e.g., "yes", "no", "allowed", "not allowed"
    modification_conditions: Optional[str] = None # free text, ideally including both constraints
    modification_sources: List[str] = Field(default_factory=list)

    # General sources (if the answer provides a single shared sources section)
    all_sources: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_lottery_info() -> str:
    return """
Extract the user's final stated answers (exactly as written) to the four sub-questions and any URLs they cite as sources.

Required fields:
1) lottery_month_for_trip: The lottery month explicitly tied to a June 2026 trip (e.g., "February 2026" or "February"). If not explicitly given, return null.
2) opening_date_earliest: The earliest calendar date the application can be submitted for that June 2026 trip (e.g., "February 16, 2026"). If not explicitly given, return null.
3) earliest_sources: All URLs the answer cites that directly support the opening/lottery timing. Return [] if none.

4) lottery_cost_total: The total cost the answer claims is required to enter the lottery for a group of 4. Capture exactly what the answer states (e.g., "$10", "$40", etc.). If not given, return null.
5) cost_sources: All URLs the answer cites to support the fee. Return [] if none.

6) pickup_location: Where the answer says the permit must be picked up if selected. If not given, return null.
7) pickup_sources: All URLs cited for the pickup policy. Return [] if none.

8) modification_allowed: The answer's statement on whether itinerary changes can be made after winning but before the trip date (e.g., "yes", "allowed", "no", "not allowed"). If not clearly stated, return null.
9) modification_conditions: The answer's stated condition(s) (free text). If not provided, return null.
10) modification_sources: All URLs cited for modification policy. Return [] if none.

11) all_sources: Any general/source URLs the answer includes that apply to multiple statements. Return [] if none.

Rules:
- Do not infer or invent information; capture exactly what the answer states.
- For any URL fields, extract only valid URLs explicitly present in the answer (including markdown links).
- Return null for any missing field and [] for any missing URL arrays.
"""


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _merge_sources(*lists: Optional[List[str]]) -> List[str]:
    merged: List[str] = []
    for lst in lists:
        if not lst:
            continue
        for u in lst:
            if isinstance(u, str):
                u = u.strip()
            if u and u not in merged:
                merged.append(u)
    return merged


def _bool_from_text(s: Optional[str]) -> Optional[bool]:
    if s is None:
        return None
    t = s.strip().lower()
    # Heuristics for yes/no
    yes_tokens = {"yes", "y", "allowed", "can", "permitted", "true"}
    no_tokens = {"no", "n", "not allowed", "cannot", "can't", "forbidden", "false"}
    # Normalize phrases
    if any(tok in t for tok in yes_tokens):
        return True
    if any(tok in t for tok in no_tokens):
        return False
    return None


# --------------------------------------------------------------------------- #
# Verification subtrees                                                       #
# --------------------------------------------------------------------------- #
async def verify_lottery_application_date(evaluator: Evaluator, parent_node, info: LotteryExtraction) -> None:
    # Parent node for application date (sequential as per rubric)
    lot_app_node = evaluator.add_sequential(
        id="Lottery_Application_Date",
        desc="Correctly determines the earliest date to submit lottery application",
        parent=parent_node,
        critical=False,
    )

    # Timeline calculation (sequential)
    timeline_node = evaluator.add_sequential(
        id="Timeline_Calculation",
        desc="Understands that lottery applications are for trips four months after the lottery month (June 2026 trip = February 2026 lottery)",
        parent=lot_app_node,
        critical=False,
    )

    # 1) Existence: Did the answer explicitly state a lottery month for June 2026?
    timeline_provided = evaluator.add_custom_node(
        result=bool(info.lottery_month_for_trip and info.lottery_month_for_trip.strip()),
        id="Timeline_Provided",
        desc="Answer explicitly states the lottery month for a June 2026 trip",
        parent=timeline_node,
        critical=True,  # Gate subsequent checks in this sequential branch
    )

    # 2) Correctness: Is their stated lottery month actually February 2026?
    timeline_correct_leaf = evaluator.add_leaf(
        id="Timeline_Correct",
        desc="June 2026 trip corresponds to the February 2026 lottery month",
        parent=timeline_node,
        critical=True,
    )
    claim_timeline = (
        f"For a June 2026 Bright Angel Campground trip, the lottery month is stated as '{info.lottery_month_for_trip}'. "
        f"The correct lottery month should be February 2026 (four months earlier)."
    )
    await evaluator.verify(
        claim=claim_timeline,
        node=timeline_correct_leaf,
        sources=_merge_sources(info.earliest_sources, info.all_sources),
        additional_instruction="Mark as supported only if the webpage indicates that the lottery month for June trips is February (i.e., the lottery runs four months earlier).",
    )

    # 3) Opening date specification (critical leaf beneath timeline calculation)
    opening_leaf = evaluator.add_leaf(
        id="Opening_Date_Specification",
        desc="Correctly identifies February 16, 2026 as the lottery opening date (16th of the lottery month)",
        parent=timeline_node,
        critical=True,
    )
    # If the answer provided a specific opening date, verify that stated date
    stated_date = info.opening_date_earliest or ""
    claim_opening = (
        f"The earliest date to submit the lottery application for a June 2026 trip is {stated_date}. "
        f"The correct earliest date should be February 16, 2026 (the 16th day of the lottery month)."
    )
    await evaluator.verify(
        claim=claim_opening,
        node=opening_leaf,
        sources=_merge_sources(info.earliest_sources, info.all_sources),
        additional_instruction="Only the calendar date matters here; time/time zones may vary on the page. Consider 'Feb 16, 2026' equivalent to 'February 16, 2026'.",
    )


async def verify_lottery_cost(evaluator: Evaluator, parent_node, info: LotteryExtraction) -> None:
    # Single leaf for cost verification
    cost_leaf = evaluator.add_leaf(
        id="Lottery_Cost",
        desc="Correctly states the total cost to enter the lottery for a group of 4 people is $10 (fee is per permit, not per person)",
        parent=parent_node,
        critical=False,
    )
    stated_cost = info.lottery_cost_total or ""
    claim_cost = (
        f"The answer states the total cost to enter the lottery for a group of 4 is {stated_cost}. "
        f"The correct policy is that the lottery entry fee is $10 per application/permit (not per person), "
        f"so the total should be $10 for the group."
    )
    await evaluator.verify(
        claim=claim_cost,
        node=cost_leaf,
        sources=_merge_sources(info.cost_sources, info.all_sources),
        additional_instruction="Verify that the source says the lottery/early access application fee is $10 per application/permit (non‑refundable), not per person.",
    )


async def verify_permit_pickup_location(evaluator: Evaluator, parent_node, info: LotteryExtraction) -> None:
    pickup_leaf = evaluator.add_leaf(
        id="Permit_Pickup_Location",
        desc="Identifies that permits must be picked up in person at the Backcountry Information Center",
        parent=parent_node,
        critical=False,
    )
    stated_loc = info.pickup_location or ""
    claim_pickup = (
        f"The answer states the permit pickup location as '{stated_loc}'. "
        f"The correct policy is that permits must be picked up in person at the Backcountry Information Center."
    )
    await evaluator.verify(
        claim=claim_pickup,
        node=pickup_leaf,
        sources=_merge_sources(info.pickup_sources, info.all_sources),
        additional_instruction="Accept reasonable naming variants of 'Backcountry Information Center' (e.g., 'Backcountry Permit Office') if clearly the same office. Reject generic 'visitor center' with no indication it's the Backcountry office.",
    )


async def verify_itinerary_modification(evaluator: Evaluator, parent_node, info: LotteryExtraction) -> None:
    mod_node = evaluator.add_parallel(
        id="Itinerary_Modification",
        desc="Correctly explains the itinerary modification policy",
        parent=parent_node,
        critical=False,
    )

    # Optional presence check (non-critical) to encourage citing/mentioning
    mod_presence = evaluator.add_custom_node(
        result=bool((info.modification_allowed and info.modification_allowed.strip()) or (info.modification_conditions and info.modification_conditions.strip())),
        id="Modification_Info_Provided",
        desc="Answer provides a statement about itinerary modification policy",
        parent=mod_node,
        critical=False,
    )

    # Allowed?
    mod_allowed_leaf = evaluator.add_leaf(
        id="Modification_Allowed",
        desc="States that itinerary changes can be made after winning the lottery but before the trip",
        parent=mod_node,
        critical=True,
    )
    ans_allowed_bool = _bool_from_text(info.modification_allowed)
    if ans_allowed_bool is True:
        claim_allowed = "Itinerary changes can be made after winning the lottery but before the trip date."
    elif ans_allowed_bool is False:
        claim_allowed = "Itinerary changes cannot be made after winning the lottery but before the trip date."
    else:
        # If not clearly stated, assert the correct policy; the presence node captures omission separately
        claim_allowed = "Itinerary changes can be made after winning the lottery but before the trip date."
    await evaluator.verify(
        claim=claim_allowed,
        node=mod_allowed_leaf,
        sources=_merge_sources(info.modification_sources, info.all_sources),
        additional_instruction="Mark as supported only if the webpage indicates that itinerary modifications are allowed after selection and before the trip.",
    )

    # Conditions (must include BOTH: (1) subsequent months (not during lottery month) AND (2) space availability)
    mod_cond_leaf = evaluator.add_leaf(
        id="Modification_Conditions",
        desc="Specifies the conditions: changes must be made in subsequent months (not during lottery month) and space must be available",
        parent=mod_node,
        critical=True,
    )
    stated_cond = info.modification_conditions or ""
    claim_conditions = (
        f"The answer's stated modification conditions are: '{stated_cond}'. "
        f"These are correct only if BOTH of the following are true per the source: "
        f"(1) changes must be made in subsequent months (not during the lottery month), and "
        f"(2) space must be available."
    )
    await evaluator.verify(
        claim=claim_conditions,
        node=mod_cond_leaf,
        sources=_merge_sources(info.modification_sources, info.all_sources),
        additional_instruction="Mark as supported only if the source explicitly supports BOTH constraints: subsequent-months-only (not during lottery month) AND space-available requirement. If either is missing or contradicted, mark as not supported.",
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
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # root is non-critical to allow partial credit across sub-questions
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

    # Record ground truth for transparency
    evaluator.add_ground_truth(GROUND_TRUTH, gt_type="ground_truth")

    # Extract structured information from the answer
    extracted: LotteryExtraction = await evaluator.extract(
        prompt=prompt_extract_lottery_info(),
        template_class=LotteryExtraction,
        extraction_name="lottery_extraction",
    )

    # Build the rubric tree under the root
    gc_node = evaluator.add_parallel(
        id="Grand_Canyon_Permit_Requirements",
        desc="Evaluates understanding of Grand Canyon backcountry permit lottery system for a June 2026 trip",
        parent=root,
        critical=False,  # Make parent non-critical to avoid forcing all children to be critical
    )

    # Subtrees
    await verify_lottery_application_date(evaluator, gc_node, extracted)
    await verify_lottery_cost(evaluator, gc_node, extracted)
    await verify_permit_pickup_location(evaluator, gc_node, extracted)
    await verify_itinerary_modification(evaluator, gc_node, extracted)

    return evaluator.get_summary()