import asyncio
import logging
from typing import Any, Optional, List, Dict

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy

# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "budget_road_trip_2026_nps_enterprise"
TASK_DESCRIPTION = (
    "An 18-year-old US resident is planning a budget road trip in early 2026 to visit national parks. "
    "They want to minimize costs by visiting on a fee-free day and need to rent a car from Enterprise. "
    "Which US states allow Enterprise Rent-A-Car to rent vehicles to 18-year-olds, and what is the first fee-free day in February 2026 "
    "when US residents can enter national parks without paying entrance fees?"
)

# Ground truth expectations used for context in the evaluation (not strict enforcement beyond rubric)
GROUND_TRUTH = {
    "enterprise_18yo_states_expected": ["Michigan", "New York"],
    "first_feb_fee_free_day_expected": "February 16, 2026",
    "first_feb_fee_free_day_holiday": "Presidents Day",
}

# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class EnterpriseEligibility(BaseModel):
    states_allowing_18yo: List[str] = Field(default_factory=list)
    mentions_other_states_21_plus: Optional[bool] = None
    sources: List[str] = Field(default_factory=list)


class FeeFreeDayInfo(BaseModel):
    first_fee_free_day_feb_2026: Optional[str] = None
    holiday_name: Optional[str] = None
    applies_to_us_residents: Optional[bool] = None
    sources: List[str] = Field(default_factory=list)


class TripExtraction(BaseModel):
    enterprise: EnterpriseEligibility = EnterpriseEligibility()
    fee_free: FeeFreeDayInfo = FeeFreeDayInfo()


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_trip_info() -> str:
    return """
    Extract the key facts the answer provides for:
    (A) Enterprise Rent-A-Car state eligibility for 18-year-old renters, and 
    (B) the first fee-free national park day in February 2026.

    Return a JSON object with two top-level fields: "enterprise" and "fee_free".

    Under "enterprise", extract:
    - states_allowing_18yo: an array of US states (as they appear in the answer) that the answer claims allow Enterprise to rent to 18-year-olds.
      • Include state names or abbreviations exactly as written in the answer (e.g., "Michigan", "MI", "New York", "NY").
      • Do not infer or add states that are not explicitly claimed in the answer.
    - mentions_other_states_21_plus: a boolean indicating if the answer explicitly says that in all other US states the minimum age is at least 21 (true/false). If not clearly stated, set to null.
    - sources: an array of any URLs the answer cites specifically for Enterprise rental age policy.

    Under "fee_free", extract:
    - first_fee_free_day_feb_2026: the date string for the first fee-free day in February 2026, exactly as stated in the answer (e.g., "February 16, 2026" or "Feb 16, 2026"). If missing, set to null.
    - holiday_name: the holiday name if the answer mentions one (e.g., "Presidents Day"). If not given, set to null.
    - applies_to_us_residents: a boolean indicating if the answer frames this fee-free day as applying to US residents. If unclear or not mentioned, set to null.
    - sources: an array of any URLs the answer cites regarding national park fee-free days (e.g., NPS pages).

    Only extract what the answer explicitly states. Do not invent or infer missing information.
    """


# --------------------------------------------------------------------------- #
# Verification subroutines                                                    #
# --------------------------------------------------------------------------- #
async def verify_enterprise_eligibility(
    evaluator: Evaluator,
    parent_node,
    ent: EnterpriseEligibility
) -> None:
    """
    Build and verify the 'enterprise_18yo_state_eligibility' sub-tree.
    Note on criticality: To satisfy the framework constraint (critical parent cannot have non-critical children),
    we set this aggregation node to non-critical, while keeping its key leaf critical.
    """
    enterprise_node = evaluator.add_parallel(
        id="enterprise_18yo_state_eligibility",
        desc="Identify which US states allow Enterprise Rent-A-Car rentals by 18-year-olds.",
        parent=parent_node,
        critical=False  # Allows inclusion of a non-critical child leaf without violating framework constraints
    )

    # Leaf: states_exactly_mi_ny (Critical)
    states_leaf = evaluator.add_leaf(
        id="states_exactly_mi_ny",
        desc="The answer identifies exactly Michigan and New York (and no other states) as the states where Enterprise allows 18-year-olds to rent.",
        parent=enterprise_node,
        critical=True
    )

    states_list_str = ", ".join(ent.states_allowing_18yo) if ent.states_allowing_18yo else "(none extracted)"
    states_claim = (
        "From the answer text, the listed US states where Enterprise Rent-A-Car rents to 18-year-olds are: "
        f"{states_list_str}. This list contains exactly Michigan and New York and no other states."
    )
    await evaluator.verify(
        claim=states_claim,
        node=states_leaf,
        additional_instruction=(
            "Judge strictly based on the answer content. Treat 'MI' equivalent to 'Michigan' and 'NY' equivalent to 'New York'. "
            "Ignore case and punctuation. If any additional US state beyond Michigan and New York is listed (including non-state regions), "
            "then this claim should be considered incorrect."
        ),
    )

    # Leaf: optionally_mentions_other_states_21_plus (Non-critical)
    other_states_leaf = evaluator.add_leaf(
        id="optionally_mentions_other_states_21_plus",
        desc="The answer notes that in all other US states the minimum rental age is at least 21.",
        parent=enterprise_node,
        critical=False
    )

    other_states_claim = (
        "The answer explicitly states that in all other US states (besides the exceptions), the minimum rental age for Enterprise is at least 21."
    )
    await evaluator.verify(
        claim=other_states_claim,
        node=other_states_leaf,
        additional_instruction=(
            "Look for clear phrasings like 'all other states are 21+' or 'in other states you must be at least 21'. "
            "Minor wording variations are acceptable if the meaning is the same. If the answer is silent on this point, mark as incorrect."
        ),
    )


async def verify_fee_free_day(
    evaluator: Evaluator,
    parent_node,
    fee: FeeFreeDayInfo
) -> None:
    """
    Build and verify the 'first_fee_free_day_feb_2026' sub-tree.
    This node and both leaves are critical.
    """
    fee_node = evaluator.add_parallel(
        id="first_fee_free_day_feb_2026",
        desc="Identify the first fee-free day in February 2026 for entering national parks without entrance fees.",
        parent=parent_node,
        critical=True
    )

    # Leaf: first_fee_free_day_is_feb_16_2026 (Critical)
    date_leaf = evaluator.add_leaf(
        id="first_fee_free_day_is_feb_16_2026",
        desc="The answer states that the first fee-free day in February 2026 is February 16, 2026 (Presidents Day).",
        parent=fee_node,
        critical=True
    )

    extracted_date = fee.first_fee_free_day_feb_2026 or "(none extracted)"
    extracted_holiday = fee.holiday_name or "(none)"
    date_claim = (
        "The answer states that the first fee-free day in February 2026 is February 16, 2026 (Presidents Day). "
        f"The extracted date from the answer is '{extracted_date}' and the holiday name is '{extracted_holiday}'."
    )
    await evaluator.verify(
        claim=date_claim,
        node=date_leaf,
        sources=fee.sources if fee.sources else None,
        additional_instruction=(
            "We are verifying what the answer claims and, if URLs are provided, whether those URLs support that "
            "February 16, 2026 (Presidents Day) is a fee-free day. Accept reasonable date formatting variants like 'Feb 16, 2026'."
        ),
    )

    # Leaf: fee_free_day_applies_to_us_residents (Critical)
    resident_leaf = evaluator.add_leaf(
        id="fee_free_day_applies_to_us_residents",
        desc="The answer indicates the fee-free day benefit is for US residents (consistent with the task framing).",
        parent=fee_node,
        critical=True
    )

    resident_claim = (
        "The answer frames the fee-free day as applicable to US residents (i.e., it explicitly ties the benefit to US residents)."
    )
    await evaluator.verify(
        claim=resident_claim,
        node=resident_leaf,
        additional_instruction=(
            "Focus on the answer text. Pass if the answer clearly states or implies that US residents can take advantage of this fee-free day. "
            "Do not over-interpret external policy; judge only the answer's wording."
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
    model: str = "o4-mini"
) -> Dict:
    """
    Evaluate an answer for the Enterprise + National Park fee-free day planning task.
    """
    # Initialize evaluator (root is always non-critical by framework design)
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # High-level categories are independent
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
    extraction: TripExtraction = await evaluator.extract(
        prompt=prompt_extract_trip_info(),
        template_class=TripExtraction,
        extraction_name="trip_planning_extraction"
    )

    # Add ground-truth info (for context in final report; not used for automatic gating beyond rubric)
    evaluator.add_ground_truth(
        {
            "expected_enterprise_states": GROUND_TRUTH["enterprise_18yo_states_expected"],
            "expected_first_feb_2026_fee_free_day": GROUND_TRUTH["first_feb_fee_free_day_expected"],
            "expected_holiday": GROUND_TRUTH["first_feb_fee_free_day_holiday"],
        },
        gt_type="ground_truth_expectations"
    )

    # Build top-level task node. Note:
    # The provided rubric marks this node as critical, but because one of its grandchildren must be non-critical,
    # we relax this node to non-critical to satisfy the framework's constraint that a critical parent cannot have non-critical children.
    budget_node = evaluator.add_parallel(
        id="budget_road_trip_planning",
        desc="Evaluate whether the answer provides the required rental eligibility info and the first February 2026 national-park fee-free day for a US resident.",
        parent=root,
        critical=False
    )

    # Sub-verifications
    await verify_enterprise_eligibility(evaluator, budget_node, extraction.enterprise)
    await verify_fee_free_day(evaluator, budget_node, extraction.fee_free)

    # Return structured evaluation summary
    return evaluator.get_summary()