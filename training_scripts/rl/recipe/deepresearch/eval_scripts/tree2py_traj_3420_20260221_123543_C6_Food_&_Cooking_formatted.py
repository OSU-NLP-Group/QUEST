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
TASK_ID = "thanksgiving_2025_chains"
TASK_DESCRIPTION = (
    "Identify four distinct food service or retail chain establishments operating in the United States on "
    "Thanksgiving Day 2025 (November 27, 2025), meeting category-specific requirements and provide sources."
)

HOLIDAY_DATE_TEXT = "Thanksgiving Day 2025 (November 27, 2025)"
BLACK_FRIDAY_DATE_TEXT = "Black Friday 2025 (November 28, 2025)"


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class GroceryChainInfo(BaseModel):
    chain_name: Optional[str] = None
    thanksgiving_open_statement: Optional[str] = None
    closing_time_window: Optional[str] = None  # e.g., "3 PM - 5 PM", "closes at 4 PM"
    thanksgiving_hours_urls: List[str] = Field(default_factory=list)

    pharmacy_status_statement: Optional[str] = None  # e.g., "Pharmacy is closed"
    pharmacy_urls: List[str] = Field(default_factory=list)

    black_friday_opening_time: Optional[str] = None  # e.g., "6 AM", "7:00 a.m."
    black_friday_urls: List[str] = Field(default_factory=list)


class CoffeeChainInfo(BaseModel):
    chain_name: Optional[str] = None
    thanksgiving_open_statement: Optional[str] = None
    closing_time_window: Optional[str] = None  # e.g., "most locations close 12 PM - 2 PM"
    thanksgiving_hours_urls: List[str] = Field(default_factory=list)

    typical_opening_time_range: Optional[str] = None  # e.g., "5 AM - 7 AM"
    opening_range_urls: List[str] = Field(default_factory=list)

    hours_vary_by_location_statement: Optional[str] = None  # e.g., "hours vary by store/location"
    franchise_variation_urls: List[str] = Field(default_factory=list)


class RestaurantChainInfo(BaseModel):
    chain_name: Optional[str] = None
    regular_hours_statement: Optional[str] = None  # e.g., "open 6:00 AM to 10:00 PM"
    thanksgiving_hours_urls: List[str] = Field(default_factory=list)

    dine_in_available_statement: Optional[str] = None
    dine_in_urls: List[str] = Field(default_factory=list)

    takeout_available_statement: Optional[str] = None
    takeout_urls: List[str] = Field(default_factory=list)


class ConvenienceStoreChainInfo(BaseModel):
    chain_name: Optional[str] = None
    operates_24_7_statement: Optional[str] = None  # e.g., "open 24/7 including holidays"
    hours_urls: List[str] = Field(default_factory=list)

    food_service_statement: Optional[str] = None  # e.g., "offers food/snacks"
    food_urls: List[str] = Field(default_factory=list)

    fuel_service_statement: Optional[str] = None  # e.g., "offers fuel/gas"
    fuel_urls: List[str] = Field(default_factory=list)


class ThanksgivingChainsExtraction(BaseModel):
    grocery: Optional[GroceryChainInfo] = None
    coffee: Optional[CoffeeChainInfo] = None
    restaurant: Optional[RestaurantChainInfo] = None
    convenience: Optional[ConvenienceStoreChainInfo] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_thanksgiving_chains() -> str:
    return """
    Extract structured information for four distinct chains presented in the answer, one for each category:
    Category 1: Grocery Store Chain
    Category 2: Coffee Shop Chain
    Category 3: Full-Service Restaurant Chain
    Category 4: Convenience Store Chain

    For each category, extract exactly the fields specified below. If a field is not explicitly provided in the answer, set it to null (for single values) or an empty array (for URL lists). Extract only URLs actually visible in the answer (plain or markdown links).

    For the Grocery Store Chain, extract:
    - chain_name: the chain’s name
    - thanksgiving_open_statement: a sentence or phrase from the answer asserting it is open on Thanksgiving Day 2025
    - closing_time_window: the claimed Thanksgiving closing time window (e.g., "closes at 4 PM" or "3 PM - 5 PM")
    - thanksgiving_hours_urls: URLs that support Thanksgiving hours claims
    - pharmacy_status_statement: a sentence/phrase asserting pharmacy is closed on Thanksgiving
    - pharmacy_urls: URLs that support the pharmacy status claim
    - black_friday_opening_time: the claimed opening time on Black Friday (Nov 28, 2025)
    - black_friday_urls: URLs that support the Black Friday opening time claim

    For the Coffee Shop Chain, extract:
    - chain_name
    - thanksgiving_open_statement: a sentence asserting it is open on Thanksgiving Day 2025
    - closing_time_window: a claim that most locations close between 12 PM and 2 PM on Thanksgiving
    - thanksgiving_hours_urls: URLs supporting Thanksgiving hours claims
    - typical_opening_time_range: typical opening time range on Thanksgiving (e.g., "5 AM - 7 AM")
    - opening_range_urls: URLs supporting the opening time range
    - hours_vary_by_location_statement: a statement that hours vary by location or franchise
    - franchise_variation_urls: URLs supporting hours variation by location/franchise

    For the Full-Service Restaurant Chain, extract:
    - chain_name
    - regular_hours_statement: a statement asserting they are open during regular hours (6:00 AM to 10:00 PM) on Thanksgiving
    - thanksgiving_hours_urls: URLs supporting Thanksgiving hours claims
    - dine_in_available_statement: a statement asserting dine-in is available on Thanksgiving
    - dine_in_urls: URLs supporting dine-in availability
    - takeout_available_statement: a statement asserting takeout/to-go is available
    - takeout_urls: URLs supporting takeout availability

    For the Convenience Store Chain, extract:
    - chain_name
    - operates_24_7_statement: a statement asserting 24/7 operation including Thanksgiving Day 2025
    - hours_urls: URLs supporting 24/7/holiday operation claims
    - food_service_statement: a statement asserting food/snacks are offered
    - food_urls: URLs supporting food/snacks availability
    - fuel_service_statement: a statement asserting fuel/gas services are typically offered
    - fuel_urls: URLs supporting fuel/gas services

    Return a JSON object with keys: grocery, coffee, restaurant, convenience.
    Each key maps to an object with the corresponding fields. If the answer lacks a category, set that category to null.
    """


# --------------------------------------------------------------------------- #
# Helper: Safe name                                                           #
# --------------------------------------------------------------------------- #
def _safe_chain(chain_name: Optional[str]) -> str:
    return chain_name.strip() if chain_name else "the chain"


# --------------------------------------------------------------------------- #
# Verification functions per category                                         #
# --------------------------------------------------------------------------- #
async def verify_grocery(evaluator: Evaluator, parent_node, info: Optional[GroceryChainInfo]) -> None:
    node = evaluator.add_sequential(
        id="grocery_store",
        desc="Identify a grocery store chain meeting all specified criteria",
        parent=parent_node,
        critical=False
    )

    # Chain selection (open on Thanksgiving, closing time between 3 PM and 5 PM)
    select_node = evaluator.add_parallel(
        id="grocery_chain_selection",
        desc="Select a grocery chain that is open on Thanksgiving 2025 with closing time between 3 PM and 5 PM",
        parent=node,
        critical=True
    )

    chain_name = _safe_chain(info.chain_name if info else None)
    tg_urls = info.thanksgiving_hours_urls if info else []

    open_leaf = evaluator.add_leaf(
        id="grocery_open_thanksgiving",
        desc="Grocery chain is open on Thanksgiving 2025",
        parent=select_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"{chain_name} is open on {HOLIDAY_DATE_TEXT}.",
        node=open_leaf,
        sources=tg_urls,
        additional_instruction="Confirm the page indicates the chain operates or is open on Thanksgiving Day 2025."
    )

    closing_leaf = evaluator.add_leaf(
        id="grocery_closing_3_to_5",
        desc="Grocery chain closes between 3:00 PM and 5:00 PM local time on Thanksgiving 2025",
        parent=select_node,
        critical=True
    )
    claimed_window = info.closing_time_window if info and info.closing_time_window else "between 3 PM and 5 PM"
    await evaluator.verify(
        claim=f"On {HOLIDAY_DATE_TEXT}, {chain_name} closes {claimed_window}, and this is between 3:00 PM and 5:00 PM local time.",
        node=closing_leaf,
        sources=tg_urls,
        additional_instruction="If the page lists a specific closing time (e.g., 4 PM), consider it within the 3–5 PM window."
    )

    # Reference URLs present for Thanksgiving hours
    ref_urls_leaf = evaluator.add_custom_node(
        result=bool(info and info.thanksgiving_hours_urls and len(info.thanksgiving_hours_urls) > 0),
        id="grocery_reference_urls",
        desc="Provide reference URL(s) confirming the grocery chain's Thanksgiving hours",
        parent=node,
        critical=True
    )

    # Attributes group: pharmacy closed, Black Friday opening time
    attrs_node = evaluator.add_parallel(
        id="grocery_attributes",
        desc="Verify additional attributes of the selected grocery chain",
        parent=node,
        critical=True
    )

    # Pharmacy status
    pharm_leaf = evaluator.add_leaf(
        id="pharmacy_status",
        desc="Confirm whether the pharmacy department is closed on Thanksgiving",
        parent=attrs_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"On {HOLIDAY_DATE_TEXT}, the pharmacy department at {chain_name} is closed.",
        node=pharm_leaf,
        sources=(info.pharmacy_urls if info else []),
        additional_instruction="Confirm the pharmacy is closed on Thanksgiving; store may be open with pharmacy closed."
    )

    # Pharmacy reference URL existence
    pharm_ref_leaf = evaluator.add_custom_node(
        result=bool(info and info.pharmacy_urls and len(info.pharmacy_urls) > 0),
        id="pharmacy_reference_url",
        desc="Provide reference URL confirming pharmacy department status",
        parent=attrs_node,
        critical=True
    )

    # Black Friday opening time
    bf_leaf = evaluator.add_leaf(
        id="black_friday_hours",
        desc="Provide the Black Friday (Nov 28, 2025) opening time",
        parent=attrs_node,
        critical=True
    )
    bf_time = info.black_friday_opening_time if info and info.black_friday_opening_time else "an early morning time"
    await evaluator.verify(
        claim=f"On {BLACK_FRIDAY_DATE_TEXT}, {chain_name} opens at {bf_time}.",
        node=bf_leaf,
        sources=(info.black_friday_urls if info else []),
        additional_instruction="Verify the opening time for Black Friday 2025; approximate textual variants (e.g., 6 AM vs 6:00 a.m.) are acceptable."
    )

    # Black Friday reference URL existence
    bf_ref_leaf = evaluator.add_custom_node(
        result=bool(info and info.black_friday_urls and len(info.black_friday_urls) > 0),
        id="black_friday_reference_url",
        desc="Provide reference URL for Black Friday hours",
        parent=attrs_node,
        critical=True
    )


async def verify_coffee(evaluator: Evaluator, parent_node, info: Optional[CoffeeChainInfo]) -> None:
    node = evaluator.add_sequential(
        id="coffee_shop",
        desc="Identify a coffee shop chain meeting all specified criteria",
        parent=parent_node,
        critical=False
    )

    # Chain selection (open on Thanksgiving, most locations closing 12–2 PM)
    select_node = evaluator.add_parallel(
        id="coffee_chain_selection",
        desc="Select a coffee chain that is open on Thanksgiving 2025 with most locations closing between 12 PM and 2 PM",
        parent=node,
        critical=True
    )

    chain_name = _safe_chain(info.chain_name if info else None)
    tg_urls = info.thanksgiving_hours_urls if info else []

    open_leaf = evaluator.add_leaf(
        id="coffee_open_thanksgiving",
        desc="Coffee chain is open on Thanksgiving 2025",
        parent=select_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"{chain_name} is open on {HOLIDAY_DATE_TEXT}.",
        node=open_leaf,
        sources=tg_urls,
        additional_instruction="Confirm the page indicates the chain operates or is open on Thanksgiving Day 2025."
    )

    closing_leaf = evaluator.add_leaf(
        id="coffee_closing_12_to_2",
        desc="Most locations close between 12 PM and 2 PM on Thanksgiving 2025",
        parent=select_node,
        critical=True
    )
    claimed_window = info.closing_time_window if info and info.closing_time_window else "between 12 PM and 2 PM"
    await evaluator.verify(
        claim=f"On {HOLIDAY_DATE_TEXT}, most {chain_name} locations close {claimed_window}, i.e., between 12:00 PM and 2:00 PM.",
        node=closing_leaf,
        sources=tg_urls,
        additional_instruction="Statements like 'most stores close early around midday' or explicit 12–2 PM ranges should count."
    )

    # Reference URLs present for Thanksgiving hours
    ref_urls_leaf = evaluator.add_custom_node(
        result=bool(info and info.thanksgiving_hours_urls and len(info.thanksgiving_hours_urls) > 0),
        id="coffee_reference_urls",
        desc="Provide reference URL(s) confirming the coffee chain's Thanksgiving hours",
        parent=node,
        critical=True
    )

    # Attributes: typical opening time range; hours vary by location/franchise
    attrs_node = evaluator.add_parallel(
        id="coffee_attributes",
        desc="Verify additional attributes of the selected coffee chain",
        parent=node,
        critical=True
    )

    # Typical opening time range
    open_range_leaf = evaluator.add_leaf(
        id="typical_opening_time",
        desc="Provide the typical opening time range on Thanksgiving",
        parent=attrs_node,
        critical=True
    )
    opening_range = info.typical_opening_time_range if info and info.typical_opening_time_range else "an early morning range"
    await evaluator.verify(
        claim=f"On {HOLIDAY_DATE_TEXT}, typical {chain_name} opening times fall within {opening_range}.",
        node=open_range_leaf,
        sources=(info.opening_range_urls if info else []),
        additional_instruction="Accept reasonable textual variants indicating a morning opening range on Thanksgiving."
    )

    # Opening reference URL existence
    open_ref_leaf = evaluator.add_custom_node(
        result=bool(info and info.opening_range_urls and len(info.opening_range_urls) > 0),
        id="opening_reference_url",
        desc="Provide reference URL for opening times",
        parent=attrs_node,
        critical=True
    )

    # Franchise/location variation
    variation_leaf = evaluator.add_leaf(
        id="franchise_variation",
        desc="Confirm that hours vary by individual location/franchise",
        parent=attrs_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"For {chain_name}, hours vary by individual location or franchise on {HOLIDAY_DATE_TEXT}.",
        node=variation_leaf,
        sources=(info.franchise_variation_urls if info else []),
        additional_instruction="Look for explicit statements like 'hours vary by location' or 'check with your local store'."
    )

    # Franchise variation reference URL existence
    variation_ref_leaf = evaluator.add_custom_node(
        result=bool(info and info.franchise_variation_urls and len(info.franchise_variation_urls) > 0),
        id="franchise_reference_url",
        desc="Provide reference URL confirming franchise-based hour variations",
        parent=attrs_node,
        critical=True
    )


async def verify_restaurant(evaluator: Evaluator, parent_node, info: Optional[RestaurantChainInfo]) -> None:
    node = evaluator.add_sequential(
        id="full_service_restaurant",
        desc="Identify a full-service restaurant chain meeting all specified criteria",
        parent=parent_node,
        critical=False
    )

    # Chain selection (open during regular hours 6 AM–10 PM)
    select_leaf = evaluator.add_leaf(
        id="restaurant_chain_selection",
        desc="Select a full-service restaurant chain that is open during regular hours (6 AM to 10 PM) on Thanksgiving 2025",
        parent=node,
        critical=True
    )

    chain_name = _safe_chain(info.chain_name if info else None)
    tg_urls = info.thanksgiving_hours_urls if info else []
    regular_stmt = info.regular_hours_statement if info and info.regular_hours_statement else "open 6:00 AM to 10:00 PM"

    await evaluator.verify(
        claim=f"On {HOLIDAY_DATE_TEXT}, {chain_name} is open during regular hours ({regular_stmt}).",
        node=select_leaf,
        sources=tg_urls,
        additional_instruction="Confirm the page indicates regular hours apply on Thanksgiving (approximately 6 AM–10 PM)."
    )

    # Reference URLs present
    ref_urls_leaf = evaluator.add_custom_node(
        result=bool(info and info.thanksgiving_hours_urls and len(info.thanksgiving_hours_urls) > 0),
        id="restaurant_reference_urls",
        desc="Provide reference URL(s) confirming the restaurant's Thanksgiving hours",
        parent=node,
        critical=True
    )

    # Attributes: dine-in, takeout
    attrs_node = evaluator.add_parallel(
        id="restaurant_attributes",
        desc="Verify additional attributes of the selected restaurant chain",
        parent=node,
        critical=True
    )

    dine_leaf = evaluator.add_leaf(
        id="dine_in_available",
        desc="Confirm that dine-in service is available on Thanksgiving",
        parent=attrs_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"On {HOLIDAY_DATE_TEXT}, dine-in service is available at {chain_name}.",
        node=dine_leaf,
        sources=(info.dine_in_urls if info else []),
        additional_instruction="Look for explicit 'dine-in available' statements for Thanksgiving; reservation or wait details are acceptable context."
    )

    dine_ref_leaf = evaluator.add_custom_node(
        result=bool(info and info.dine_in_urls and len(info.dine_in_urls) > 0),
        id="dine_in_reference_url",
        desc="Provide reference URL confirming dine-in availability",
        parent=attrs_node,
        critical=True
    )

    takeout_leaf = evaluator.add_leaf(
        id="takeout_available",
        desc="Confirm that takeout/to-go options are available",
        parent=attrs_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"On {HOLIDAY_DATE_TEXT}, takeout or to-go options are available at {chain_name}.",
        node=takeout_leaf,
        sources=(info.takeout_urls if info else []),
        additional_instruction="Look for statements such as 'takeout available', 'to-go', 'pickup' for Thanksgiving."
    )

    takeout_ref_leaf = evaluator.add_custom_node(
        result=bool(info and info.takeout_urls and len(info.takeout_urls) > 0),
        id="takeout_reference_url",
        desc="Provide reference URL confirming takeout availability",
        parent=attrs_node,
        critical=True
    )


async def verify_convenience(evaluator: Evaluator, parent_node, info: Optional[ConvenienceStoreChainInfo]) -> None:
    node = evaluator.add_sequential(
        id="convenience_store",
        desc="Identify a convenience store chain meeting all specified criteria",
        parent=parent_node,
        critical=False
    )

    # Chain selection (24/7 including Thanksgiving)
    select_leaf = evaluator.add_leaf(
        id="convenience_chain_selection",
        desc="Select a convenience store chain that operates 24/7 including Thanksgiving 2025",
        parent=node,
        critical=True
    )
    chain_name = _safe_chain(info.chain_name if info else None)
    hours_urls = info.hours_urls if info else []
    await evaluator.verify(
        claim=f"{chain_name} operates 24/7, including on {HOLIDAY_DATE_TEXT}.",
        node=select_leaf,
        sources=hours_urls,
        additional_instruction="Confirm explicit 24/7 operation and that holidays (including Thanksgiving) are covered."
    )

    # Reference URLs present
    ref_urls_leaf = evaluator.add_custom_node(
        result=bool(info and info.hours_urls and len(info.hours_urls) > 0),
        id="convenience_reference_urls",
        desc="Provide reference URL(s) confirming 24/7 operation on Thanksgiving",
        parent=node,
        critical=True
    )

    # Attributes: food/snacks and fuel/gas
    attrs_node = evaluator.add_parallel(
        id="convenience_attributes",
        desc="Verify additional attributes of the selected convenience store chain",
        parent=node,
        critical=True
    )

    food_leaf = evaluator.add_leaf(
        id="food_service",
        desc="Confirm that the store offers food/snacks",
        parent=attrs_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"{chain_name} offers food and snacks.",
        node=food_leaf,
        sources=(info.food_urls if info else []),
        additional_instruction="Confirm availability of food/snack items (hot foods, packaged snacks, etc.)."
    )

    food_ref_leaf = evaluator.add_custom_node(
        result=bool(info and info.food_urls and len(info.food_urls) > 0),
        id="food_reference_url",
        desc="Provide reference URL confirming food availability",
        parent=attrs_node,
        critical=True
    )

    fuel_leaf = evaluator.add_leaf(
        id="fuel_service",
        desc="Confirm that the chain typically offers fuel/gas services",
        parent=attrs_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"{chain_name} typically offers fuel/gas services at its locations.",
        node=fuel_leaf,
        sources=(info.fuel_urls if info else []),
        additional_instruction="Confirm the presence of fuel/gas stations as a typical service at the chain's locations."
    )

    fuel_ref_leaf = evaluator.add_custom_node(
        result=bool(info and info.fuel_urls and len(info.fuel_urls) > 0),
        id="fuel_reference_url",
        desc="Provide reference URL confirming fuel services",
        parent=attrs_node,
        critical=True
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
    Evaluate an answer for the Thanksgiving 2025 chains task using the Mind2Web2 framework.
    """
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Parallel root: categories evaluated independently
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
    extraction = await evaluator.extract(
        prompt=prompt_extract_thanksgiving_chains(),
        template_class=ThanksgivingChainsExtraction,
        extraction_name="thanksgiving_chains_extraction",
    )

    # Optional contextual info
    evaluator.add_custom_info(
        {"holiday_date": HOLIDAY_DATE_TEXT, "black_friday_date": BLACK_FRIDAY_DATE_TEXT},
        info_type="holiday_context",
    )

    # Build verification tree and run checks per category
    await verify_grocery(evaluator, root, extraction.grocery)
    await verify_coffee(evaluator, root, extraction.coffee)
    await verify_restaurant(evaluator, root, extraction.restaurant)
    await verify_convenience(evaluator, root, extraction.convenience)

    # Return aggregated summary
    return evaluator.get_summary()