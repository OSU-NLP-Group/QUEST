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
TASK_ID = "thanksgiving_planning_2025"
TASK_DESCRIPTION = (
    "A family is planning their Thanksgiving 2025 preparations and needs to identify four specific businesses based on their holiday schedules and offerings. "
    "On Thanksgiving Day (November 27, 2025), they need to complete an early morning grocery run, so they require a grocery store chain that is open on Thanksgiving Day but closes before 2:00 PM. "
    "They also want to know which national fast-food chain they should avoid visiting on Thanksgiving Day because it will be closed. "
    "Additionally, they are interested in ordering a Cajun-style turkey from a restaurant chain for their Thanksgiving meal—they need to know which chain offers this product, what the weight range is, and what the in-store pickup price is. "
    "Finally, on Thanksgiving Eve (November 26, 2025), they want to take advantage of a Buy-One-Get-One (BOGO) free entrée promotion at a restaurant chain that starts at or after 4:00 PM and is valid for in-restaurant dining only. "
    "Identify all four required items with their complete details."
)


# --------------------------------------------------------------------------- #
# Data models for structured extraction                                       #
# --------------------------------------------------------------------------- #
class GroceryStoreInfo(BaseModel):
    chain_name: Optional[str] = None
    thanksgiving_date: Optional[str] = None  # e.g., "November 27, 2025", "Thanksgiving Day 2025"
    closing_time: Optional[str] = None       # e.g., "1 PM", "Noon", "12:00 PM"
    notes: Optional[str] = None              # any extra text from the answer about hours
    sources: List[str] = Field(default_factory=list)


class FastFoodClosureInfo(BaseModel):
    chain_name: Optional[str] = None
    closure_note: Optional[str] = None       # text like "closed on Thanksgiving"
    sources: List[str] = Field(default_factory=list)


class CajunTurkeyInfo(BaseModel):
    restaurant_chain: Optional[str] = None
    product_name: Optional[str] = None       # e.g., "Cajun-Style Turkey"
    weight_range: Optional[str] = None       # e.g., "13–16 lbs", "10-14 pounds"
    pickup_price: Optional[str] = None       # e.g., "$59.99", "$XX (in-store pickup)"
    sources: List[str] = Field(default_factory=list)


class BogoPromotionInfo(BaseModel):
    restaurant_chain: Optional[str] = None
    date: Optional[str] = None               # e.g., "November 26, 2025"
    start_time: Optional[str] = None         # e.g., "4:00 PM"
    dine_in_only_text: Optional[str] = None  # text indicating in-restaurant only
    sources: List[str] = Field(default_factory=list)


class ThanksgivingPlanExtraction(BaseModel):
    early_closing_grocery_store: Optional[GroceryStoreInfo] = None
    closed_fast_food_chain: Optional[FastFoodClosureInfo] = None
    turkey_product_restaurant: Optional[CajunTurkeyInfo] = None
    thanksgiving_eve_bogo_promotion: Optional[BogoPromotionInfo] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_thanksgiving_plan() -> str:
    return """
    Extract exactly one item for each of the four required categories from the answer text. If a category is missing, set it to null. 
    Only extract information that is explicitly present in the answer. For all URL fields, extract only the actual URLs explicitly shown in the answer.

    1) early_closing_grocery_store:
       - chain_name: the grocery store chain name
       - thanksgiving_date: the date text for Thanksgiving, if mentioned (e.g., "November 27, 2025")
       - closing_time: the specific closing time on Thanksgiving Day (e.g., "1 PM", "12:30 PM", "Noon")
       - notes: any textual note about hours (e.g., "open in the morning", "hours vary by location")
       - sources: a list of all URLs cited to support the store's Thanksgiving hours

    2) closed_fast_food_chain:
       - chain_name: the national fast-food chain name
       - closure_note: text indicating closure on Thanksgiving Day
       - sources: a list of all URLs cited to support that the chain is closed on Thanksgiving

    3) turkey_product_restaurant:
       - restaurant_chain: the restaurant chain selling a Cajun-style turkey for Thanksgiving 2025
       - product_name: the product name if provided (e.g., "Cajun-Style Turkey")
       - weight_range: the weight range string exactly as stated (e.g., "13–16 lbs", "10-14 pounds")
       - pickup_price: the in-store pickup price string exactly as stated (e.g., "$59.99", "from $59.99", "$x (in-store pickup)")
       - sources: a list of all URLs cited to support the Cajun turkey product details

    4) thanksgiving_eve_bogo_promotion:
       - restaurant_chain: the restaurant chain running a BOGO entrée promotion
       - date: the date text for the promotion (e.g., "November 26, 2025")
       - start_time: the time the promotion starts (e.g., "4:00 PM", "after 4 PM")
       - dine_in_only_text: text indicating the promotion is for in-restaurant dining only
       - sources: a list of all URLs cited to support the promotion details

    Rules:
    - For all string fields, return the exact text from the answer where possible; do not normalize or infer.
    - For sources, extract only valid, full URLs explicitly present in the answer (including markdown links).
    - If a field is not present in the answer, set it to null (or [] for sources).
    """


# --------------------------------------------------------------------------- #
# Verification helpers                                                        #
# --------------------------------------------------------------------------- #
async def verify_grocery_store(
    evaluator: Evaluator,
    parent_node,
    info: Optional[GroceryStoreInfo],
) -> None:
    group = evaluator.add_parallel(
        id="early_closing_grocery_store",
        desc="A grocery store chain that is open on Thanksgiving Day 2025 but closes before 2:00 PM",
        parent=parent_node,
        critical=False,
    )

    # Reference URL existence (critical)
    sources = (info.sources if info else []) if info else []
    ref_exists = evaluator.add_custom_node(
        result=bool(sources),
        id="store_reference_url",
        desc="A reference URL is provided supporting the store's Thanksgiving hours",
        parent=group,
        critical=True,
    )

    # Store open on Thanksgiving (critical)
    open_node = evaluator.add_leaf(
        id="store_open_thanksgiving",
        desc="The store is confirmed to be open on Thanksgiving Day (November 27, 2025)",
        parent=group,
        critical=True,
    )
    store_name = (info.chain_name if info and info.chain_name else "the identified grocery chain")
    open_claim = f"{store_name} is open on Thanksgiving Day (November 27, 2025)."
    await evaluator.verify(
        claim=open_claim,
        node=open_node,
        sources=sources,
        additional_instruction=(
            "Verify from the provided URL(s) that this grocery chain is open on Thanksgiving Day 2025. "
            "Wording like 'open with limited hours' or 'open until 1 PM' qualifies as open. "
            "If the page indicates closure or does not cover 2025 Thanksgiving at all, mark as not supported."
        ),
    )

    # Closes before 2:00 PM (critical)
    close_node = evaluator.add_leaf(
        id="store_closes_before_2pm",
        desc="The store closes before 2:00 PM on Thanksgiving Day",
        parent=group,
        critical=True,
    )
    close_claim = f"On Thanksgiving Day (November 27, 2025), {store_name} closes before 2:00 PM local time."
    await evaluator.verify(
        claim=close_claim,
        node=close_node,
        sources=sources,
        additional_instruction=(
            "Look for explicit Thanksgiving Day 2025 hours stating a closing time strictly before 2:00 PM "
            "(e.g., 1:00 PM, 12:00 PM/Noon, 1:30 PM). If hours say 'closes at 2 PM' or later, this claim is NOT supported. "
            "If the page only states generic 'hours vary by location' without a clear before-2PM closing time for Thanksgiving 2025, mark as not supported."
        ),
    )


async def verify_fast_food_closure(
    evaluator: Evaluator,
    parent_node,
    info: Optional[FastFoodClosureInfo],
) -> None:
    group = evaluator.add_parallel(
        id="closed_fast_food_chain",
        desc="A national fast-food chain that is closed on Thanksgiving Day 2025",
        parent=parent_node,
        critical=False,
    )

    # Reference URL existence (critical)
    sources = (info.sources if info else []) if info else []
    evaluator.add_custom_node(
        result=bool(sources),
        id="chain_reference_url",
        desc="A reference URL is provided supporting the chain's closure on Thanksgiving",
        parent=group,
        critical=True,
    )

    # Closed on Thanksgiving (critical)
    closed_node = evaluator.add_leaf(
        id="chain_closed_thanksgiving",
        desc="The chain is confirmed to be closed on Thanksgiving Day (November 27, 2025)",
        parent=group,
        critical=True,
    )
    chain_name = (info.chain_name if info and info.chain_name else "the identified chain")
    closed_claim = f"{chain_name} is closed on Thanksgiving Day (November 27, 2025)."
    await evaluator.verify(
        claim=closed_claim,
        node=closed_node,
        sources=sources,
        additional_instruction=(
            "Use the provided source(s) to verify closure on Thanksgiving Day 2025. "
            "If the source indicates 'open' or is unrelated/non-2025, the claim is not supported."
        ),
    )

    # Is a fast-food chain (critical)
    ff_node = evaluator.add_leaf(
        id="chain_is_fast_food",
        desc="The identified chain is a fast-food restaurant chain",
        parent=group,
        critical=True,
    )
    ff_claim = f"{chain_name} is a national fast-food (quick-service) restaurant chain in the United States."
    await evaluator.verify(
        claim=ff_claim,
        node=ff_node,
        sources=sources,
        additional_instruction=(
            "Verify that the chain is commonly classified as a fast-food (quick-service) restaurant chain "
            "with national presence (locations across multiple U.S. states). "
            "Wikipedia or the official site stating quick-service/fast-food is acceptable."
        ),
    )


async def verify_cajun_turkey(
    evaluator: Evaluator,
    parent_node,
    info: Optional[CajunTurkeyInfo],
) -> None:
    group = evaluator.add_parallel(
        id="turkey_product_restaurant",
        desc="A restaurant chain offering a Cajun-style turkey product for Thanksgiving 2025",
        parent=parent_node,
        critical=False,
    )

    # Reference URL existence (critical)
    sources = (info.sources if info else []) if info else []
    evaluator.add_custom_node(
        result=bool(sources),
        id="turkey_reference_url",
        desc="A reference URL is provided supporting the turkey product details",
        parent=group,
        critical=True,
    )

    chain = (info.restaurant_chain if info and info.restaurant_chain else "the identified restaurant chain")
    product = (info.product_name if info and info.product_name else "Cajun-style turkey")

    # Offers Cajun turkey (critical)
    offer_node = evaluator.add_leaf(
        id="offers_cajun_turkey",
        desc="The restaurant offers a Cajun-style turkey for Thanksgiving 2025",
        parent=group,
        critical=True,
    )
    offer_claim = f"{chain} offers a {product} for Thanksgiving 2025 (order/pickup available)."
    await evaluator.verify(
        claim=offer_claim,
        node=offer_node,
        sources=sources,
        additional_instruction=(
            "Confirm from the source(s) that this chain sells a Cajun-style whole turkey specifically for Thanksgiving 2025. "
            "Announcements, product pages, or press confirming 2025 availability qualify."
        ),
    )

    # Weight range (critical)
    weight_text = (info.weight_range if info and info.weight_range else "")
    weight_node = evaluator.add_leaf(
        id="turkey_weight_range",
        desc="The turkey's weight is specified within a defined range",
        parent=group,
        critical=True,
    )
    weight_claim = f"The {product} sold by {chain} has a stated weight range of {weight_text}."
    await evaluator.verify(
        claim=weight_claim,
        node=weight_node,
        sources=sources,
        additional_instruction=(
            "Verify the stated weight range (e.g., '13–16 lbs', '10–14 pounds') from the source(s). "
            "If the range is not given or differs, mark as not supported."
        ),
    )

    # Pickup price (critical)
    price_text = (info.pickup_price if info and info.pickup_price else "")
    price_node = evaluator.add_leaf(
        id="pickup_price_specified",
        desc="The in-store pickup price for the turkey is specified",
        parent=group,
        critical=True,
    )
    price_claim = f"The in-store pickup price for the {product} at {chain} is {price_text}."
    await evaluator.verify(
        claim=price_claim,
        node=price_node,
        sources=sources,
        additional_instruction=(
            "Verify the in-store pickup price from the source(s). If the source only lists non-pickup pricing or omits price, mark as not supported. "
            "Minor formatting variations (e.g., '$59.99' vs '59.99 USD') are acceptable as equivalent."
        ),
    )


async def verify_bogo_promotion(
    evaluator: Evaluator,
    parent_node,
    info: Optional[BogoPromotionInfo],
) -> None:
    group = evaluator.add_parallel(
        id="thanksgiving_eve_bogo_promotion",
        desc="A restaurant chain with a BOGO (Buy-One-Get-One) promotion on Thanksgiving Eve 2025 starting at or after 4:00 PM",
        parent=parent_node,
        critical=False,
    )

    # Reference URL existence (critical)
    sources = (info.sources if info else []) if info else []
    evaluator.add_custom_node(
        result=bool(sources),
        id="promotion_reference_url",
        desc="A reference URL is provided supporting the promotional details",
        parent=group,
        critical=True,
    )

    chain = (info.restaurant_chain if info and info.restaurant_chain else "the identified restaurant chain")

    # Occurs on Nov 26, 2025 (critical)
    date_node = evaluator.add_leaf(
        id="bogo_on_nov_26",
        desc="The promotion occurs on November 26, 2025 (Thanksgiving Eve)",
        parent=group,
        critical=True,
    )
    date_claim = f"{chain} has a BOGO entrée promotion on November 26, 2025 (Thanksgiving Eve)."
    await evaluator.verify(
        claim=date_claim,
        node=date_node,
        sources=sources,
        additional_instruction=(
            "Verify that the promotion date is specifically November 26, 2025 (Thanksgiving Eve). "
            "If the page shows a different date or no 2025 date, mark as not supported."
        ),
    )

    # Starts at or after 4pm (critical)
    start_node = evaluator.add_leaf(
        id="starts_at_or_after_4pm",
        desc="The promotion starts at 4:00 PM or later",
        parent=group,
        critical=True,
    )
    start_claim = f"{chain}'s BOGO entrée promotion on November 26, 2025 starts at or after 4:00 PM local time."
    await evaluator.verify(
        claim=start_claim,
        node=start_node,
        sources=sources,
        additional_instruction=(
            "Verify that the start time is 4:00 PM or later (e.g., 'starts at 4 PM', 'after 4 PM'). "
            "If it starts earlier than 4 PM, or no time is given, mark as not supported."
        ),
    )

    # In-restaurant only (critical)
    dine_node = evaluator.add_leaf(
        id="in_restaurant_only",
        desc="The promotion is valid for in-restaurant redemption only",
        parent=group,
        critical=True,
    )
    dine_claim = (
        f"{chain}'s BOGO entrée promotion on November 26, 2025 is valid for in-restaurant (dine-in) redemption only "
        "and not valid for delivery or online orders."
    )
    await evaluator.verify(
        claim=dine_claim,
        node=dine_node,
        sources=sources,
        additional_instruction=(
            "Confirm the promotion requires in-restaurant (dine-in) redemption only. "
            "If the offer is available for online orders, delivery, or takeout, this claim is not supported."
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
    Evaluate an answer for the Thanksgiving 2025 planning task and return a structured result dictionary.
    """
    # Initialize evaluator with a parallel root as per rubric
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

    # Extraction
    extracted: ThanksgivingPlanExtraction = await evaluator.extract(
        prompt=prompt_extract_thanksgiving_plan(),
        template_class=ThanksgivingPlanExtraction,
        extraction_name="thanksgiving_plan_extraction",
    )

    # Build subtrees and verify each required item
    await verify_grocery_store(evaluator, root, extracted.early_closing_grocery_store)
    await verify_fast_food_closure(evaluator, root, extracted.closed_fast_food_chain)
    await verify_cajun_turkey(evaluator, root, extracted.turkey_product_restaurant)
    await verify_bogo_promotion(evaluator, root, extracted.thanksgiving_eve_bogo_promotion)

    # Return structured evaluation summary
    return evaluator.get_summary()