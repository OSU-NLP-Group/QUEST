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
TASK_ID = "frontier_dot_nps_2025_combo"
TASK_DESCRIPTION = (
    "You are planning to fly Frontier Airlines from your home city to Fort Lauderdale to board the Disney Destiny cruise "
    "ship on its maiden voyage date. You are considering bringing one bag measuring 15 inches in height, 17 inches in width, "
    "and 8 inches in depth (including handles and straps). To maximize comfort during your early morning flight, you are "
    "planning to wear pajamas and slippers to the airport.\n\n"
    "After your cruise, you plan to fly to Yellowstone National Park for two separate weekend visits in December 2025 "
    "(before January 1, 2026), entering the park with your private vehicle each time.\n\n"
    "Based on current US Department of Transportation guidelines and Frontier Airlines policies as of November 2025, "
    "answer the following three questions:\n\n"
    "1. Will your bag (15\"H x 17\"W x 8\"D including handles and straps) qualify as a free personal item on Frontier Airlines, "
    "or will you need to pay for it as a carry-on bag?\n\n"
    "2. Does your planned attire of wearing pajamas and slippers to the airport comply with the US Transportation Secretary's "
    "civility campaign guidelines launched in November 2025?\n\n"
    "3. If you make two separate visits to Yellowstone National Park in December 2025, each time paying the private vehicle "
    "entrance fee, would it be more economical to purchase an America the Beautiful annual pass instead? Show your calculation."
)

# Policy/fee constants (as specified in rubric/task)
FRONTIER_PERSONAL_ITEM_LIMIT = {"height": 14, "width": 18, "depth": 8}  # inches
PLANNED_BAG_DIMS = {"height": 15, "width": 17, "depth": 8}  # inches, including handles/straps
CAMPAIGN_NAME = "Golden Age of Travel Starts with You"
CAMPAIGN_GUIDELINE = "dress with respect"
YELLOWSTONE_PRIVATE_VEHICLE_FEE = 35  # USD per vehicle per entry
AMERICA_THE_BEAUTIFUL_PASS_COST = 80  # USD
EXPECTED_COST_TWO_VISITS = YELLOWSTONE_PRIVATE_VEHICLE_FEE * 2  # 70
EXPECTED_COST_DECISION = "individual_fees_cheaper"  # Because 70 < 80

# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class TravelExtraction(BaseModel):
    # Frontier bag assessment
    bag_assessment: Optional[str] = None  # expected: "free_personal_item" or "paid_carry_on" or "unclear"
    bag_policy_urls: List[str] = Field(default_factory=list)

    # DOT civility campaign attire assessment
    attire_assessment: Optional[str] = None  # expected: "complies" or "does_not_comply" or "unclear"
    attire_urls: List[str] = Field(default_factory=list)

    # National park cost analysis
    cost_assessment: Optional[str] = None  # expected: "annual_pass_cheaper" or "individual_fees_cheaper" or "same_cost" or "unclear"
    cost_calculation: Optional[str] = None
    nps_urls: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_travel_info() -> str:
    return """
    Extract from the answer the user-facing conclusions and any cited URLs for the three sub-questions. Return:
    - bag_assessment: Choose one of ["free_personal_item", "paid_carry_on", "unclear"] based on what the answer concludes for Frontier Airlines given the 15\"H x 17\"W x 8\"D bag including handles/straps. Map synonyms (e.g., "it counts as a personal item for free") to "free_personal_item"; map "you need to pay for a carry-on" to "paid_carry_on". If no clear conclusion is given, return "unclear".
    - bag_policy_urls: A list of any URLs cited in the answer that relate to Frontier Airlines' personal item or carry-on policy. If none, return [].

    - attire_assessment: Choose one of ["complies", "does_not_comply", "unclear"] based on what the answer concludes regarding wearing pajamas and slippers to the airport under the DOT civility campaign guideline to "dress with respect". If no clear stance is given, return "unclear".
    - attire_urls: A list of any URLs cited for the DOT civility campaign or guidance. If none, return [].

    - cost_assessment: Choose one of ["annual_pass_cheaper", "individual_fees_cheaper", "same_cost", "unclear"] based on what the answer concludes when comparing two separate Yellowstone private vehicle fees in Dec 2025 at $35 each vs the $80 America the Beautiful annual pass. If no clear stance, return "unclear".
    - cost_calculation: The calculation or explanation text shown in the answer for this comparison (if any), else null.
    - nps_urls: A list of any URLs cited for Yellowstone entrance fees or the America the Beautiful annual pass. If none, return [].

    IMPORTANT:
    - Do not invent URLs. Extract only those present in the answer text (including markdown links).
    - Keep the category strings exactly as specified above.
    """


# --------------------------------------------------------------------------- #
# Normalization helpers                                                       #
# --------------------------------------------------------------------------- #
def normalize_bag_assessment(s: Optional[str]) -> str:
    if not s:
        return "unclear"
    t = s.strip().lower()
    if "free_personal_item" in t or ("personal" in t and "carry" not in t):
        return "free_personal_item"
    if "paid_carry_on" in t or "carry-on" in t or "carry on" in t or "paid" in t:
        return "paid_carry_on"
    return "unclear"


def normalize_attire_assessment(s: Optional[str]) -> str:
    if not s:
        return "unclear"
    t = s.strip().lower()
    if "does_not_comply" in t or "not comply" in t or "does not" in t or "inappropriate" in t or "not appropriate" in t:
        return "does_not_comply"
    if "complies" in t or "appropriate" in t or "aligns" in t or "acceptable" in t:
        return "complies"
    return "unclear"


def normalize_cost_assessment(s: Optional[str]) -> str:
    if not s:
        return "unclear"
    t = s.strip().lower()
    if "annual_pass_cheaper" in t or ("pass" in t and ("cheaper" in t or "more economical" in t)):
        return "annual_pass_cheaper"
    if "individual_fees_cheaper" in t or ("pay" in t and "each" in t and ("cheaper" in t or "less" in t)):
        return "individual_fees_cheaper"
    if "same_cost" in t or "equal" in t or "same" in t:
        return "same_cost"
    return "unclear"


def humanize_bag_assessment(norm: str) -> str:
    if norm == "free_personal_item":
        return "free personal item"
    if norm == "paid_carry_on":
        return "paid carry-on"
    return "unclear"


def humanize_attire_assessment(norm: str) -> str:
    if norm == "complies":
        return "complies"
    if norm == "does_not_comply":
        return "does not comply"
    return "unclear"


def humanize_cost_assessment(norm: str) -> str:
    if norm == "annual_pass_cheaper":
        return "annual pass is cheaper"
    if norm == "individual_fees_cheaper":
        return "individual entrance fees are cheaper"
    if norm == "same_cost":
        return "same cost"
    return "unclear"


# --------------------------------------------------------------------------- #
# Verification builders                                                       #
# --------------------------------------------------------------------------- #
async def build_baggage_verification(evaluator: Evaluator, parent_node, extracted: TravelExtraction) -> None:
    """
    Subtree: baggage_compliance
    Leaf: dimension_verification (critical)
    """
    bag_node = evaluator.add_sequential(
        id="baggage_compliance",
        desc="Determine whether the specified bag qualifies as a free personal item under Frontier Airlines policy",
        parent=parent_node,
        critical=False
    )

    # Normalize extracted conclusion
    bag_assess_norm = normalize_bag_assessment(extracted.bag_assessment)
    bag_assess_human = humanize_bag_assessment(bag_assess_norm)

    dim_leaf = evaluator.add_leaf(
        id="dimension_verification",
        desc="Correctly compare all three dimensions (height, width, depth) of the provided bag against Frontier's personal item size limit of 14\"H x 18\"W x 8\"D and reach the correct conclusion about compliance",
        parent=bag_node,
        critical=True
    )

    # Construct claim focusing on logical comparison, also referencing the agent's conclusion
    claim = (
        f"Frontier Airlines' free personal item limit is 14 inches high, 18 inches wide, and 8 inches deep "
        f"(including handles/straps). The user's bag is 15 inches high, 17 inches wide, and 8 inches deep (including handles/straps). "
        f"Because 15 > 14, the correct classification is: not a free personal item; it would require a paid carry-on. "
        f"The agent's answer concluded: {bag_assess_human}. The agent's conclusion matches the correct classification."
    )

    add_ins = (
        "Assume as of November 2025 that Frontier's free personal item limit is exactly 14\"H x 18\"W x 8\"D, and "
        "that measurements include handles and straps. Evaluate whether the answer's stated classification matches the only "
        "correct outcome for a 15\"H x 17\"W x 8\"D bag. Treat 'free personal item' vs 'paid carry-on' as mutually exclusive. "
        "If the answer is unclear or does not commit, it does not match."
    )

    sources = extracted.bag_policy_urls if extracted and extracted.bag_policy_urls else None
    await evaluator.verify(
        claim=claim,
        node=dim_leaf,
        sources=sources,
        additional_instruction=add_ins
    )


async def build_attire_verification(evaluator: Evaluator, parent_node, extracted: TravelExtraction) -> None:
    """
    Subtree: dress_code_compliance
    Leaf: attire_appropriateness (critical)
    """
    attire_node = evaluator.add_sequential(
        id="dress_code_compliance",
        desc="Evaluate whether the planned attire complies with US DOT civility campaign expectations",
        parent=parent_node,
        critical=False
    )

    attire_norm = normalize_attire_assessment(extracted.attire_assessment)
    attire_human = humanize_attire_assessment(attire_norm)

    attire_leaf = evaluator.add_leaf(
        id="attire_appropriateness",
        desc="Correctly evaluate whether wearing sleepwear/pajamas to the airport aligns with the DOT's 'Golden Age of Travel Starts with You' campaign guideline to 'dress with respect'",
        parent=attire_node,
        critical=True
    )

    claim = (
        f"The US DOT's civility campaign '{CAMPAIGN_NAME}' includes a guideline to '{CAMPAIGN_GUIDELINE}'. "
        f"Wearing sleepwear/pajamas and slippers to the airport is not aligned with 'dress with respect'. "
        f"The agent's answer concluded this attire {attire_human}. The agent's conclusion matches this guideline-based judgment."
    )

    add_ins = (
        "Focus on the principle 'dress with respect' for traveler civility. Use common sense: sleepwear and slippers are "
        "generally not respectful public attire for air travel. If the answer says 'complies', it does not match. "
        "If it says 'does not comply', it matches. If it is unclear, it does not match."
    )

    sources = extracted.attire_urls if extracted and extracted.attire_urls else None
    await evaluator.verify(
        claim=claim,
        node=attire_leaf,
        sources=sources,
        additional_instruction=add_ins
    )


async def build_cost_verification(evaluator: Evaluator, parent_node, extracted: TravelExtraction) -> None:
    """
    Subtree: cost_analysis
    Leaf: cost_comparison (critical)
    """
    cost_node = evaluator.add_sequential(
        id="cost_analysis",
        desc="Determine whether purchasing an America the Beautiful pass is more economical than paying individual Yellowstone entrance fees",
        parent=parent_node,
        critical=False
    )

    cost_norm = normalize_cost_assessment(extracted.cost_assessment)
    cost_human = humanize_cost_assessment(cost_norm)

    cost_leaf = evaluator.add_leaf(
        id="cost_comparison",
        desc="Correctly calculate the total cost of two separate Yellowstone private vehicle entrance fees ($35 each) versus the cost of one America the Beautiful annual pass ($80) and determine which option costs less",
        parent=cost_node,
        critical=True
    )

    claim = (
        f"Two separate Yellowstone private vehicle entries at ${YELLOWSTONE_PRIVATE_VEHICLE_FEE} each total "
        f"${EXPECTED_COST_TWO_VISITS}. The America the Beautiful annual pass costs ${AMERICA_THE_BEAUTIFUL_PASS_COST}. "
        f"Therefore, it is cheaper by ${AMERICA_THE_BEAUTIFUL_PASS_COST - EXPECTED_COST_TWO_VISITS} to pay individual entrance fees; "
        f"it is not more economical to buy the annual pass for just two visits. "
        f"The agent's answer concluded: {cost_human}. The agent's conclusion matches this calculation."
    )

    add_ins = (
        "Perform the simple arithmetic: 2 × $35 = $70; compare to $80. Because $70 < $80, individual fees are cheaper by $10. "
        "If the answer claims the annual pass is cheaper or more economical, it does not match. If it claims individual fees are cheaper, it matches. "
        "If it says same cost or is unclear, it does not match."
    )

    sources = extracted.nps_urls if extracted and extracted.nps_urls else None
    await evaluator.verify(
        claim=claim,
        node=cost_leaf,
        sources=sources,
        additional_instruction=add_ins
    )


# --------------------------------------------------------------------------- #
# Main evaluation function                                                    #
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
    Evaluate an answer for Frontier/DOT/NPS combined travel scenario.
    """
    # Initialize evaluator and root node
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

    # Record expected constants for transparency
    evaluator.add_custom_info(
        info={
            "frontier_personal_item_limit_in": FRONTIER_PERSONAL_ITEM_LIMIT,
            "planned_bag_dimensions_in": PLANNED_BAG_DIMS,
            "dot_campaign_name": CAMPAIGN_NAME,
            "dot_guideline": CAMPAIGN_GUIDELINE,
            "yellowstone_private_vehicle_fee_usd": YELLOWSTONE_PRIVATE_VEHICLE_FEE,
            "america_the_beautiful_pass_usd": AMERICA_THE_BEAUTIFUL_PASS_COST,
            "expected_two_visit_total_usd": EXPECTED_COST_TWO_VISITS,
            "expected_cost_decision": EXPECTED_COST_DECISION
        },
        info_type="context_constants",
        info_name="ground_assumptions"
    )

    # Extract structured conclusions from the agent's answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_travel_info(),
        template_class=TravelExtraction,
        extraction_name="travel_extraction"
    )

    # Build and run verification subtrees
    await build_baggage_verification(evaluator, root, extracted)
    await build_attire_verification(evaluator, root, extracted)
    await build_cost_verification(evaluator, root, extracted)

    # Return summary
    return evaluator.get_summary()