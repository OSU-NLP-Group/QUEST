import asyncio
import logging
import re
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "black_friday_chicago_2025_two_stores"
TASK_DESCRIPTION = (
    "You are planning a Black Friday 2025 shopping trip in Chicago, Illinois on November 29, 2025. You need to visit two different retail stores from different retail chains. "
    "The first store must sell electronics (such as smartphones, laptops, tablets, smartwatches, or similar electronic devices). "
    "The second store must sell home goods or seasonal decor (such as furniture, holiday decorations, advent calendars, home accessories, or similar items). "
    "For each store, provide: (1) the retail chain name, (2) the specific Chicago neighborhood or area and the complete street address, "
    "(3) the exact Black Friday opening time with AM/PM specified, (4) a reference URL for the store location, and (5) a reference URL confirming the Black Friday 2025 hours. "
    "The two stores must be from different retail chains (e.g., if one is Target, the other cannot be Target). "
    "Finally, provide a suggested starting address in Chicago for the shopping trip, recommend which store to visit first, and estimate the total travel time between the two stores."
)


# --------------------------------------------------------------------------- #
# Extraction models                                                           #
# --------------------------------------------------------------------------- #
class StoreExtract(BaseModel):
    chain_name: Optional[str] = None
    neighborhood_or_area: Optional[str] = None
    address: Optional[str] = None
    opening_time_bf2025: Optional[str] = None
    location_url: Optional[str] = None
    hours_url: Optional[str] = None


class RouteExtract(BaseModel):
    starting_address: Optional[str] = None
    recommended_first_store: Optional[str] = None
    travel_time_estimate: Optional[str] = None


class ShoppingPlanExtraction(BaseModel):
    store1: Optional[StoreExtract] = None  # Expected: electronics store
    store2: Optional[StoreExtract] = None  # Expected: home goods or seasonal decor store
    route: Optional[RouteExtract] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_shopping_plan() -> str:
    return """
    Extract the Black Friday 2025 shopping plan details for two stores in Chicago from the provided answer.

    Map the two stores as follows:
    - store1: The store that sells electronics (e.g., smartphones, laptops, tablets, smartwatches, or similar electronic devices).
    - store2: The store that sells home goods or seasonal decor (e.g., furniture, holiday decorations, advent calendars, home accessories, or similar items).

    For each store, extract the following fields exactly as mentioned in the answer:
    - chain_name: The retail chain name (e.g., Best Buy, Target, Walmart).
    - neighborhood_or_area: The specific Chicago neighborhood or area (e.g., The Loop, River North, Lincoln Park). If unclear, use the area description used by the answer.
    - address: The complete street address including city and state if provided.
    - opening_time_bf2025: The exact Black Friday 2025 opening time with AM/PM specified (e.g., "5:00 AM", "6 AM"). If the answer provides multiple times or a range, extract the one claimed as the Black Friday opening time.
    - location_url: A URL reference for the store location (prefer an official store page or store locator page).
    - hours_url: A URL reference confirming Black Friday 2025 hours for the specific location or chain.

    Also extract the routing details:
    - route.starting_address: The starting point address in Chicago for the shopping trip.
    - route.recommended_first_store: The recommendation of which store to visit first (can be a chain name or a specific store identifier used in the answer, e.g., "Store 1", "Best Buy").
    - route.travel_time_estimate: The estimated total travel time between the two stores (e.g., "15 minutes", "about 20–25 minutes").

    Rules:
    - Do not invent information. If any field is missing in the answer, set it to null.
    - For URLs, only extract actual URLs explicitly present in the answer. If a URL is in markdown format, extract the actual link target.
    - Preserve the exact strings as written in the answer.
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _exists_non_empty(s: Optional[str]) -> bool:
    return bool(s and isinstance(s, str) and s.strip())


def _looks_like_full_address(addr: Optional[str]) -> bool:
    if not _exists_non_empty(addr):
        return False
    # Heuristic: address should contain a number and some letters and likely a city/state marker
    has_number = bool(re.search(r"\d", addr))
    has_street_word = bool(re.search(r"[A-Za-z]", addr))
    return has_number and has_street_word


def _has_am_pm(time_str: Optional[str]) -> bool:
    if not _exists_non_empty(time_str):
        return False
    s = time_str.strip().lower()
    # Accept am/pm in forms like '5 am', '5:00am', '5:00 a.m.', '6 PM', etc.
    return "am" in s or "pm" in s


def _url_list(*urls: Optional[str]) -> List[str]:
    return [u for u in urls if _exists_non_empty(u)]


# --------------------------------------------------------------------------- #
# Verification subroutines                                                    #
# --------------------------------------------------------------------------- #
async def verify_store(
    evaluator: Evaluator,
    parent_node,
    store: StoreExtract,
    store_id: str,
    category_kind: str
) -> None:
    """
    Build and verify the subtree for a single store.
    category_kind: "electronics" for store1, "home_goods_or_decor" for store2
    """

    if category_kind == "electronics":
        store_desc = "Store 1 meets electronics requirement and includes all required store details"
        cat_confirm_desc = "Store 1 is confirmed to sell electronics (e.g., smartphones, laptops, tablets, smartwatches, or similar)"
        node_id = "Store_1_Electronics"
        chain_leaf_id = "Store_1_Chain_Name"
        addr_leaf_id = "Store_1_Address_and_Area"
        chicago_leaf_id = "Store_1_In_Chicago_IL"
        loc_ref_leaf_id = "Store_1_Location_Reference_URL"
        open_time_leaf_id = "Store_1_Black_Friday_2025_Opening_Time"
        hours_ref_leaf_id = "Store_1_Black_Friday_2025_Hours_Reference_URL"
        cat_conf_leaf_id = "Store_1_Electronics_Confirmation"
    else:
        store_desc = "Store 2 meets home goods/seasonal decor requirement and includes all required store details"
        cat_confirm_desc = "Store 2 is confirmed to sell home goods or seasonal decor (e.g., furniture, holiday decorations, advent calendars, home accessories, or similar)"
        node_id = "Store_2_Home_Goods_or_Seasonal_Decor"
        chain_leaf_id = "Store_2_Chain_Name"
        addr_leaf_id = "Store_2_Address_and_Area"
        chicago_leaf_id = "Store_2_In_Chicago_IL"
        loc_ref_leaf_id = "Store_2_Location_Reference_URL"
        open_time_leaf_id = "Store_2_Black_Friday_2025_Opening_Time"
        hours_ref_leaf_id = "Store_2_Black_Friday_2025_Hours_Reference_URL"
        cat_conf_leaf_id = "Store_2_Home_Goods_or_Decor_Confirmation"

    # Parent store node (critical parallel)
    store_node = evaluator.add_parallel(
        id=node_id,
        desc=store_desc,
        parent=parent_node,
        critical=True
    )

    # 1) Retail chain name provided (existence check)
    evaluator.add_custom_node(
        result=_exists_non_empty(store.chain_name),
        id=chain_leaf_id,
        desc="Retail chain name is provided for " + ("Store 1" if category_kind == "electronics" else "Store 2"),
        parent=store_node,
        critical=True
    )

    # 2) Address + neighborhood/area provided (existence + simple format)
    addr_ok = _exists_non_empty(store.neighborhood_or_area) and _looks_like_full_address(store.address)
    evaluator.add_custom_node(
        result=addr_ok,
        id=addr_leaf_id,
        desc="Store " + ("1" if category_kind == "electronics" else "2") + " includes a specific Chicago neighborhood/area AND a complete street address",
        parent=store_node,
        critical=True
    )

    # 3) Located in Chicago, IL (verify claim; use location_url to support if available)
    chicago_leaf = evaluator.add_leaf(
        id=chicago_leaf_id,
        desc="Store " + ("1" if category_kind == "electronics" else "2") + " is located in Chicago, Illinois (address indicates Chicago, IL)",
        parent=store_node,
        critical=True
    )
    claim_chicago = f"The address '{store.address}' is located in Chicago, IL."
    await evaluator.verify(
        claim=claim_chicago,
        node=chicago_leaf,
        sources=_url_list(store.location_url),
        additional_instruction="If a location URL is provided, confirm the address includes 'Chicago, IL' (or a Chicago ZIP). Minor formatting variations are acceptable."
    )

    # 4) Location reference URL is valid (verify the page is a store location/locator and shows the address)
    loc_ref_leaf = evaluator.add_leaf(
        id=loc_ref_leaf_id,
        desc="A valid reference URL is provided for Store " + ("1" if category_kind == "electronics" else "2") + " location (e.g., official store locator/page)",
        parent=store_node,
        critical=True
    )
    claim_loc_ref = (
        f"This webpage is an official store page or store locator entry for {store.chain_name} and lists the location at or matching the address '{store.address}' in Chicago."
    )
    await evaluator.verify(
        claim=claim_loc_ref,
        node=loc_ref_leaf,
        sources=_url_list(store.location_url),
        additional_instruction="Verify that the page is about the specific store location (or store locator entry) in Chicago and shows an address that matches or closely matches the answer."
    )

    # 5) Black Friday 2025 opening time provided with AM/PM (format/existence)
    evaluator.add_custom_node(
        result=_has_am_pm(store.opening_time_bf2025),
        id=open_time_leaf_id,
        desc="Exact Black Friday 2025 opening time for Store " + ("1" if category_kind == "electronics" else "2") + " is provided with AM/PM",
        parent=store_node,
        critical=True
    )

    # 6) Hours reference URL confirms Black Friday 2025 hours (verify by URL)
    hours_ref_leaf = evaluator.add_leaf(
        id=hours_ref_leaf_id,
        desc="A valid reference URL confirming Store " + ("1" if category_kind == "electronics" else "2") + " Black Friday 2025 hours is provided",
        parent=store_node,
        critical=True
    )
    if _exists_non_empty(store.opening_time_bf2025):
        claim_hours = (
            f"This webpage states the Black Friday 2025 opening time for the {store.chain_name} location at '{store.address}' is {store.opening_time_bf2025}."
        )
    else:
        claim_hours = (
            f"This webpage states the Black Friday 2025 opening time for the {store.chain_name} location at '{store.address}'."
        )
    await evaluator.verify(
        claim=claim_hours,
        node=hours_ref_leaf,
        sources=_url_list(store.hours_url),
        additional_instruction=(
            "Confirm the Black Friday 2025 opening time on the page. Accept phrasing like 'Black Friday hours', 'Doors open at', or "
            "'Holiday hours' that explicitly include Black Friday 2025 for this location or chain (if chain-wide hours clearly apply to this location)."
        )
    )

    # 7) Category confirmation (electronics vs home goods/seasonal decor)
    cat_leaf = evaluator.add_leaf(
        id=cat_conf_leaf_id,
        desc=cat_confirm_desc,
        parent=store_node,
        critical=True
    )
    if category_kind == "electronics":
        claim_cat = (
            f"The store (chain {store.chain_name}) sells electronics such as smartphones, laptops, tablets, or smartwatches."
        )
        add_ins = (
            "Use the provided location URL or chain page if available to confirm the chain/store sells electronics. "
            "It is acceptable to confirm at the chain level if the location page does not list categories explicitly."
        )
    else:
        claim_cat = (
            f"The store (chain {store.chain_name}) sells home goods or seasonal decor such as furniture, holiday decorations, advent calendars, or home accessories."
        )
        add_ins = (
            "Use the provided location URL or chain page if available to confirm offerings related to home goods or seasonal decor. "
            "Chain-level confirmation is acceptable if location pages do not list categories explicitly."
        )
    await evaluator.verify(
        claim=claim_cat,
        node=cat_leaf,
        sources=_url_list(store.location_url, store.hours_url),
        additional_instruction=add_ins
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
    Evaluate an answer for the Black Friday 2025 Chicago shopping plan task.
    """
    # Initialize evaluator
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

    # Extraction
    extraction: ShoppingPlanExtraction = await evaluator.extract(
        prompt=prompt_extract_shopping_plan(),
        template_class=ShoppingPlanExtraction,
        extraction_name="shopping_plan"
    )

    # Build top-level critical node
    plan_node = evaluator.add_parallel(
        id="Black_Friday_Shopping_Plan",
        desc="Complete Black Friday 2025 shopping plan in Chicago with two qualifying stores and required route details",
        parent=root,
        critical=True
    )

    # Store 1 subtree (electronics)
    store1 = extraction.store1 or StoreExtract()
    await verify_store(
        evaluator=evaluator,
        parent_node=plan_node,
        store=store1,
        store_id="Store_1_Electronics",
        category_kind="electronics"
    )

    # Store 2 subtree (home goods / seasonal decor)
    store2 = extraction.store2 or StoreExtract()
    await verify_store(
        evaluator=evaluator,
        parent_node=plan_node,
        store=store2,
        store_id="Store_2_Home_Goods_or_Seasonal_Decor",
        category_kind="home_goods_or_decor"
    )

    # Cross-store constraint
    cross_node = evaluator.add_parallel(
        id="Cross_Store_Constraint",
        desc="Constraints that relate Store 1 and Store 2",
        parent=plan_node,
        critical=True
    )
    # Different retail chains (case-insensitive)
    s1 = (store1.chain_name or "").strip().lower()
    s2 = (store2.chain_name or "").strip().lower()
    evaluator.add_custom_node(
        result=(bool(s1) and bool(s2) and s1 != s2),
        id="Different_Retail_Chains",
        desc="Store 1 and Store 2 are from different retail chains",
        parent=cross_node,
        critical=True
    )

    # Route planning
    route_node = evaluator.add_parallel(
        id="Route_Planning",
        desc="Required route logistics for the shopping trip",
        parent=plan_node,
        critical=True
    )
    route = extraction.route or RouteExtract()

    # Starting address in Chicago (leaf verification)
    start_leaf = evaluator.add_leaf(
        id="Starting_Address_in_Chicago",
        desc="A starting point address in Chicago is provided",
        parent=route_node,
        critical=True
    )
    claim_start = f"The plan provides a starting address in Chicago: '{route.starting_address}'."
    await evaluator.verify(
        claim=claim_start,
        node=start_leaf,
        additional_instruction="Confirm that a starting address is provided and it is located in Chicago (e.g., includes 'Chicago' or 'IL')."
    )

    # Recommended visit order (leaf verification)
    order_leaf = evaluator.add_leaf(
        id="Recommended_Visit_Order",
        desc="Recommendation of which store to visit first and second is provided",
        parent=route_node,
        critical=True
    )
    claim_order = f"The plan recommends visiting '{route.recommended_first_store}' first (i.e., it clearly specifies which store to start with)."
    await evaluator.verify(
        claim=claim_order,
        node=order_leaf,
        additional_instruction="Look for explicit language indicating which store to visit first (e.g., 'start at', 'visit X first', 'Store 1 first')."
    )

    # Total travel time estimate (leaf verification)
    travel_leaf = evaluator.add_leaf(
        id="Total_Travel_Time_Estimate",
        desc="An estimated total travel time between the two stores is provided",
        parent=route_node,
        critical=True
    )
    claim_travel = f"The plan includes an estimated total travel time between the two stores: '{route.travel_time_estimate}'."
    await evaluator.verify(
        claim=claim_travel,
        node=travel_leaf,
        additional_instruction="Confirm that an estimated time is provided (e.g., '15 minutes', 'about 20–25 minutes')."
    )

    return evaluator.get_summary()