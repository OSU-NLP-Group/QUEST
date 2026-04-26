import asyncio
import logging
from typing import List, Optional, Dict, Any

from pydantic import BaseModel, Field
from obj_task_eval.llm_client.base_client import LLMClient

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys

# --------------------------------------------------------------------------- #
# Task Constants                                                              #
# --------------------------------------------------------------------------- #
TASK_ID = "holiday_retail_2025_hours_returns"
TASK_DESCRIPTION = """Identify exactly four major U.S. national retail chains that meet ALL of the following criteria for the 2025 holiday shopping season:

1. Opens at 5:00 AM or earlier on Black Friday (November 28, 2025)
2. Has extended operating hours (open until at least 10:00 PM) on at least one day between December 20-23, 2025
3. Is open on Christmas Eve (December 24, 2025) and closes at or after 6:00 PM local time
4. Offers an extended holiday return policy where purchases made during November-December 2025 can be returned in January 2026 or later, with a return window of at least 30 days from the purchase date
5. Sells at least one of the following product categories: clothing/apparel, electronics, or home goods

For each of the four stores, provide:
- The store name
- Black Friday opening time on November 28, 2025, with a URL reference
- Specific date(s) between December 20-23, 2025 when the store has extended hours (open until at least 10:00 PM), with a URL reference
- Christmas Eve closing time on December 24, 2025, with a URL reference
- Extended holiday return policy details (purchase window and return deadline), with a URL reference
- Product category(ies) sold, with a URL reference
"""

BF_DATE = "November 28, 2025"
DEC_WINDOW = "December 20–23, 2025"
XMAS_EVE_DATE = "December 24, 2025"

# --------------------------------------------------------------------------- #
# Extraction Models                                                           #
# --------------------------------------------------------------------------- #
class StoreInfo(BaseModel):
    name: Optional[str] = None

    bf_open_time: Optional[str] = None
    bf_url: Optional[str] = None

    extended_hours_dates: List[str] = Field(default_factory=list)
    extended_hours_url: Optional[str] = None

    christmas_eve_close_time: Optional[str] = None
    christmas_eve_url: Optional[str] = None

    return_purchase_window: Optional[str] = None
    return_deadline: Optional[str] = None
    return_minimum_window_days: Optional[str] = None
    return_policy_url: Optional[str] = None

    product_categories: List[str] = Field(default_factory=list)
    product_url: Optional[str] = None


class StoresExtraction(BaseModel):
    stores: List[StoreInfo] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction Prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_stores() -> str:
    return """
    Extract up to four (4) major U.S. national retail chains and the requested holiday information from the provided answer text.

    For each store, extract the following fields exactly as written in the answer:
    - name: The store name.
    - bf_open_time: The Black Friday opening time (for November 28, 2025). Use the exact time string (e.g., "5:00 AM", "4am", "Doors open at 5 a.m.").
    - bf_url: The URL confirming Black Friday hours. If multiple URLs are given, extract the most relevant one for Black Friday hours; otherwise, the first one mentioned.
    - extended_hours_dates: A list of date strings (e.g., "Dec 20, 2025", "Dec 21, 2025") that fall between December 20–23, 2025 and for which the store is open until at least 10:00 PM. If none are specified, return an empty list.
    - extended_hours_url: The URL confirming extended December hours (between Dec 20–23). If multiple are given, extract one primary URL; else null if none.
    - christmas_eve_close_time: The Christmas Eve (Dec 24, 2025) closing time string exactly as presented (e.g., "6 PM", "7:00 p.m.", "6pm").
    - christmas_eve_url: The URL confirming Christmas Eve hours (open/close). If multiple are given, extract the most relevant; else null if none.
    - return_purchase_window: The purchase window for holiday returns (e.g., "Nov–Dec 2025", "Purchases made from Nov 1 through Dec 31, 2025").
    - return_deadline: The final return deadline (e.g., "Jan 31, 2026", "Returns accepted until 2/15/2026").
    - return_minimum_window_days: If the answer explicitly states a days-based window (e.g., "30 days"), extract that phrase; otherwise null.
    - return_policy_url: The URL confirming the extended holiday return policy.
    - product_categories: A list of product categories the store sells, as explicitly stated in the answer (e.g., ["electronics", "home goods", "apparel"]). Include only categories mentioned in the answer.
    - product_url: A URL confirming the product categories (e.g., a category page or store homepage).

    Rules:
    - Extract only from the answer text provided; do not invent any missing info. If a field is missing, return null (or empty list for array fields).
    - For URLs, extract the actual URL strings; include full protocol if present, otherwise prepend "http://".
    - If the answer provides more than four stores, extract only the first four mentioned. If fewer than four, return however many are present.
    """


# --------------------------------------------------------------------------- #
# Verification Helpers                                                        #
# --------------------------------------------------------------------------- #
def _first_or_empty(items: List[str]) -> str:
    return items[0] if items else ""


# --------------------------------------------------------------------------- #
# Verification per Store                                                      #
# --------------------------------------------------------------------------- #
async def verify_store(evaluator: Evaluator, parent_node, store: StoreInfo, idx: int) -> None:
    store_num = idx + 1
    store_node = evaluator.add_parallel(
        id=f"Store_{store_num}",
        desc=f"{['First','Second','Third','Fourth'][idx]} qualifying store with complete information",
        parent=parent_node,
        critical=True
    )

    # Identity
    identity_node = evaluator.add_leaf(
        id=f"Store_{store_num}_Identity",
        desc=f"Store {store_num} is identified as a major national retail chain",
        parent=store_node,
        critical=True
    )
    identity_claim = f"{store.name or ''} is a major U.S. national retail chain."
    await evaluator.verify(
        claim=identity_claim,
        node=identity_node,
        additional_instruction=(
            "Judge whether the named retailer is a widely recognized U.S. national chain "
            "(e.g., operates nationwide with many stores like Walmart, Target, Best Buy, Kohl's, Macy's, Home Depot, Lowe's, Costco, Sam's Club, Nordstrom, JCPenney, DICK'S Sporting Goods, etc.). "
            "If the store name is missing/empty or appears to be a local or niche shop, mark Incorrect."
        )
    )

    # Black Friday Hours
    bf_group = evaluator.add_parallel(
        id=f"Store_{store_num}_Black_Friday_Hours",
        desc=f"Store {store_num}'s Black Friday opening time and verification",
        parent=store_node,
        critical=True
    )

    bf_open_leaf = evaluator.add_leaf(
        id=f"Store_{store_num}_BF_Opens_5AM_Or_Earlier",
        desc=f"Store {store_num} opens at 5:00 AM or earlier on Black Friday (November 28, 2025)",
        parent=bf_group,
        critical=True
    )
    bf_claim = (
        f"On Black Friday ({BF_DATE}), {store.name or ''} opens at {store.bf_open_time or 'UNKNOWN'}, "
        f"which is at or before 5:00 AM local time."
    )
    await evaluator.verify(
        claim=bf_claim,
        node=bf_open_leaf,
        sources=store.bf_url,
        additional_instruction=(
            "Use the provided page to confirm Black Friday opening time. "
            "Accept variants like 'doors open at 5 a.m.' or earlier (e.g., 4:30 AM). "
            "If the page indicates opening strictly after 5:00 AM or does not specify an opening time, mark Incorrect."
        )
    )

    bf_url_exists = evaluator.add_custom_node(
        result=bool(store.bf_url),
        id=f"Store_{store_num}_BF_URL_Reference",
        desc=f"Provides valid URL reference confirming Store {store_num}'s Black Friday hours",
        parent=bf_group,
        critical=True
    )

    # Extended December Hours
    ext_group = evaluator.add_parallel(
        id=f"Store_{store_num}_Extended_December_Hours",
        desc=f"Store {store_num}'s extended December operating hours and verification",
        parent=store_node,
        critical=True
    )

    ext_leaf = evaluator.add_leaf(
        id=f"Store_{store_num}_Extended_Hours_Dec_20_23",
        desc=f"Store {store_num} has extended hours (open until at least 10:00 PM) on at least one day between December 20-23, 2025",
        parent=ext_group,
        critical=True
    )
    date_to_check = _first_or_empty(store.extended_hours_dates)
    ext_claim = (
        f"On {date_to_check or 'a date between Dec 20–23, 2025'}, {store.name or ''} is open until at least 10:00 PM."
    )
    await evaluator.verify(
        claim=ext_claim,
        node=ext_leaf,
        sources=store.extended_hours_url,
        additional_instruction=(
            "Confirm the store is open until 10:00 PM or later on at least one day between Dec 20 and Dec 23, 2025. "
            "If the page shows closing earlier than 10 PM on all dates within that range or provides no info, mark Incorrect."
        )
    )

    ext_url_exists = evaluator.add_custom_node(
        result=bool(store.extended_hours_url),
        id=f"Store_{store_num}_Extended_Hours_URL_Reference",
        desc=f"Provides valid URL reference confirming Store {store_num}'s extended December hours",
        parent=ext_group,
        critical=True
    )

    # Christmas Eve Hours
    xmas_group = evaluator.add_parallel(
        id=f"Store_{store_num}_Christmas_Eve_Hours",
        desc=f"Store {store_num}'s Christmas Eve hours and verification",
        parent=store_node,
        critical=True
    )

    xmas_factual = evaluator.add_sequential(
        id=f"Store_{store_num}_Christmas_Eve_Factual",
        desc=f"Store {store_num}'s Christmas Eve opening and closing time requirements",
        parent=xmas_group,
        critical=True
    )

    xmas_open_leaf = evaluator.add_leaf(
        id=f"Store_{store_num}_Open_Christmas_Eve",
        desc=f"Store {store_num} is open on Christmas Eve (December 24, 2025)",
        parent=xmas_factual,
        critical=True
    )
    xmas_open_claim = f"{store.name or ''} is open on Christmas Eve ({XMAS_EVE_DATE})."
    await evaluator.verify(
        claim=xmas_open_claim,
        node=xmas_open_leaf,
        sources=store.christmas_eve_url,
        additional_instruction=(
            "Verify the store lists operating hours on Dec 24, 2025. If the page indicates closed or provides no info, mark Incorrect."
        )
    )

    xmas_close_leaf = evaluator.add_leaf(
        id=f"Store_{store_num}_Christmas_Eve_Closes_6PM_Or_Later",
        desc=f"Store {store_num} closes at or after 6:00 PM on Christmas Eve",
        parent=xmas_factual,
        critical=True
    )
    xmas_close_claim = (
        f"On Christmas Eve ({XMAS_EVE_DATE}), {store.name or ''} closes at {store.christmas_eve_close_time or 'UNKNOWN'}, "
        f"which is at or after 6:00 PM local time."
    )
    await evaluator.verify(
        claim=xmas_close_claim,
        node=xmas_close_leaf,
        sources=store.christmas_eve_url,
        additional_instruction=(
            "Confirm the Christmas Eve closing time is 6:00 PM or later. "
            "If the page indicates closing earlier than 6 PM or provides no closing time, mark Incorrect."
        )
    )

    xmas_url_exists = evaluator.add_custom_node(
        result=bool(store.christmas_eve_url),
        id=f"Store_{store_num}_Christmas_Eve_URL_Reference",
        desc=f"Provides valid URL reference confirming Store {store_num}'s Christmas Eve hours",
        parent=xmas_group,
        critical=True
    )

    # Holiday Return Policy
    ret_group = evaluator.add_parallel(
        id=f"Store_{store_num}_Holiday_Return_Policy",
        desc=f"Store {store_num}'s extended holiday return policy and verification",
        parent=store_node,
        critical=True
    )

    ret_factual = evaluator.add_sequential(
        id=f"Store_{store_num}_Return_Policy_Factual",
        desc=f"Store {store_num}'s return policy factual requirements",
        parent=ret_group,
        critical=True
    )

    ret_ext_leaf = evaluator.add_leaf(
        id=f"Store_{store_num}_Extended_Return_Window",
        desc=f"Store {store_num} offers extended holiday returns where purchases made in November-December 2025 can be returned in January 2026 or later",
        parent=ret_factual,
        critical=True
    )
    ret_ext_claim = (
        f"{store.name or ''}'s holiday return policy allows purchases made during November–December 2025 "
        f"to be returned in January 2026 or later (deadline: {store.return_deadline or 'UNKNOWN'})."
    )
    await evaluator.verify(
        claim=ret_ext_claim,
        node=ret_ext_leaf,
        sources=store.return_policy_url,
        additional_instruction=(
            "Verify the policy explicitly permits returns of Nov–Dec 2025 purchases in January 2026 or later. "
            "If only standard (e.g., 30-day) window without explicit holiday extension to January 2026+, mark Incorrect."
        )
    )

    ret_30d_leaf = evaluator.add_leaf(
        id=f"Store_{store_num}_Return_Window_At_Least_30_Days",
        desc=f"Store {store_num}'s extended return window is at least 30 days from purchase date",
        parent=ret_factual,
        critical=True
    )
    ret_30d_claim = (
        f"{store.name or ''}'s extended holiday return window is at least 30 days from the purchase date "
        f"(stated window: {store.return_minimum_window_days or 'UNKNOWN'}; purchase window: {store.return_purchase_window or 'UNKNOWN'}; deadline: {store.return_deadline or 'UNKNOWN'})."
    )
    await evaluator.verify(
        claim=ret_30d_claim,
        node=ret_30d_leaf,
        sources=store.return_policy_url,
        additional_instruction=(
            "Confirm the holiday policy grants a minimum of 30 days from purchase. "
            "Acceptance criteria: explicit '30 days' or more; OR a final deadline sufficiently far that late-December purchases have ≥30 days. "
            "If the general holiday policy is shorter than 30 days, mark Incorrect."
        )
    )

    ret_url_exists = evaluator.add_custom_node(
        result=bool(store.return_policy_url),
        id=f"Store_{store_num}_Return_Policy_URL_Reference",
        desc=f"Provides valid URL reference confirming Store {store_num}'s holiday return policy",
        parent=ret_group,
        critical=True
    )

    # Product Category
    prod_group = evaluator.add_parallel(
        id=f"Store_{store_num}_Product_Category",
        desc=f"Store {store_num} sells qualifying product categories",
        parent=store_node,
        critical=True
    )

    prod_leaf = evaluator.add_leaf(
        id=f"Store_{store_num}_Sells_Qualifying_Products",
        desc=f"Store {store_num} sells at least one of: clothing/apparel, electronics, or home goods",
        parent=prod_group,
        critical=True
    )
    cats_str = ", ".join(store.product_categories) if store.product_categories else "UNKNOWN"
    prod_claim = (
        f"{store.name or ''} sells at least one of the following categories: clothing/apparel, electronics, or home goods. "
        f"Extracted categories: {cats_str}."
    )
    await evaluator.verify(
        claim=prod_claim,
        node=prod_leaf,
        sources=store.product_url,
        additional_instruction=(
            "Check the provided page for evidence that the retailer sells clothing/apparel (fashion), electronics (devices, TVs, computers, phones), "
            "or home goods (furniture, bedding, kitchen, decor). Synonyms count. If none are evident, mark Incorrect."
        )
    )

    prod_url_exists = evaluator.add_custom_node(
        result=bool(store.product_url),
        id=f"Store_{store_num}_Product_URL_Reference",
        desc=f"Provides valid URL reference confirming Store {store_num}'s product categories",
        parent=prod_group,
        critical=True
    )


# --------------------------------------------------------------------------- #
# Main Evaluation Entry                                                       #
# --------------------------------------------------------------------------- #
async def evaluate_answer(
    client: LLMClient,
    answer: str,
    agent_name: str,
    answer_name: str,
    cache: CacheFileSys,
    semaphore: asyncio.Semaphore,
    logger: logging.Logger,
    model: str = "o4-mini"
) -> Dict[str, Any]:
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

    # Real root aggregator for rubric (critical)
    rubric_root = evaluator.add_parallel(
        id="Four_Qualifying_Stores",
        desc="Identifies exactly four major U.S. retail stores that meet all specified Black Friday, extended December hours, Christmas Eve, and holiday return policy criteria",
        parent=root,
        critical=True
    )

    # Extract structured info
    extracted = await evaluator.extract(
        prompt=prompt_extract_stores(),
        template_class=StoresExtraction,
        extraction_name="stores_extraction"
    )

    # Ensure exactly 4 items for evaluation (pad or trim)
    stores = list(extracted.stores[:4])
    while len(stores) < 4:
        stores.append(StoreInfo())

    # Build verifications for each store
    for i in range(4):
        await verify_store(evaluator, rubric_root, stores[i], i)

    return evaluator.get_summary()