import asyncio
import logging
from typing import List, Optional, Dict, Any

from pydantic import BaseModel, Field
from obj_task_eval.llm_client.base_client import LLMClient

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys

# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "holiday_hours_2025"
TASK_DESCRIPTION = (
    "Identify three major retail stores in the United States that all met the following criteria during the 2025 holiday shopping season: "
    "(1) Opened at 6:00 a.m. or earlier on Black Friday 2025 (November 28, 2025), "
    "(2) Closed at 6:00 p.m. or earlier on Christmas Eve 2025 (December 24, 2025), "
    "(3) Were closed on Christmas Day 2025 (December 25, 2025), and "
    "(4) Each store must represent a different retail category (e.g., department store, home improvement, discount retailer, electronics, etc.). "
    "For each store, provide the store name, its Black Friday 2025 opening time, its Christmas Eve 2025 closing time, its retail category, "
    "and a reference URL from 2025 that confirms these holiday hours."
)

# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class HolidayStore(BaseModel):
    name: Optional[str] = None
    black_friday_opening_time: Optional[str] = None
    christmas_eve_closing_time: Optional[str] = None
    christmas_day_closed: Optional[str] = None  # e.g., "closed", "open", "unknown"
    category: Optional[str] = None
    reference_urls: List[str] = Field(default_factory=list)


class HolidayStoresExtraction(BaseModel):
    stores: List[HolidayStore] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_holiday_stores() -> str:
    return (
        "Extract up to the first three retail stores listed in the answer that claim to meet the 2025 holiday hours criteria. "
        "For each store, return an object with fields:\n"
        "1) name: the store name exactly as written in the answer (string)\n"
        "2) black_friday_opening_time: the opening time for Black Friday 2025 (November 28, 2025) as written (string, e.g., '5:00 a.m.', '6 AM', 'midnight')\n"
        "3) christmas_eve_closing_time: the closing time for Christmas Eve 2025 (December 24, 2025) as written (string, e.g., '6:00 p.m.', '5 PM')\n"
        "4) christmas_day_closed: the Christmas Day 2025 status as written (string; use 'closed' if the answer states closed, 'open' if states open, otherwise 'unknown')\n"
        "5) category: the retail category as written (string, e.g., 'department store', 'home improvement', 'discount retailer', 'electronics')\n"
        "6) reference_urls: an array of URLs explicitly provided in the answer that are intended to confirm the 2025 holiday hours for this store. "
        "Only include valid URLs that appear in the answer text; exclude non-URL mentions.\n"
        "If any field is not present in the answer for a store, set it to null (or empty array for reference_urls). "
        "Return a JSON object with a single 'stores' array containing these store objects in the order they appear in the answer."
    )


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def _safe_first_url(urls: List[str]) -> Optional[str]:
    return urls[0] if urls else None


def _pad_to_three(stores: List[HolidayStore]) -> List[HolidayStore]:
    padded = list(stores[:3])
    while len(padded) < 3:
        padded.append(HolidayStore())
    return padded


# --------------------------------------------------------------------------- #
# Verification for a single store                                             #
# --------------------------------------------------------------------------- #
async def verify_store(
    evaluator: Evaluator,
    parent_node,
    store: HolidayStore,
    idx: int,
    all_categories: List[Optional[str]],
) -> None:
    # Create store node
    store_node = evaluator.add_parallel(
        id=f"store_{idx+1}",
        desc=(
            "First retail store meeting all specified holiday hours criteria"
            if idx == 0 else
            ("Second retail store meeting all specified holiday hours criteria" if idx == 1
             else "Third retail store meeting all specified holiday hours criteria")
        ),
        parent=parent_node,
        critical=False
    )

    # Existence checks for required fields (each as a separate critical leaf)
    name_provided = evaluator.add_custom_node(
        result=bool(store.name and store.name.strip()),
        id=f"store_{idx+1}_name_provided",
        desc="Store name is provided",
        parent=store_node,
        critical=True
    )
    bf_opening_provided = evaluator.add_custom_node(
        result=bool(store.black_friday_opening_time and store.black_friday_opening_time.strip()),
        id=f"store_{idx+1}_black_friday_opening_provided",
        desc="Black Friday 2025 opening time is provided",
        parent=store_node,
        critical=True
    )
    cve_closing_provided = evaluator.add_custom_node(
        result=bool(store.christmas_eve_closing_time and store.christmas_eve_closing_time.strip()),
        id=f"store_{idx+1}_christmas_eve_closing_provided",
        desc="Christmas Eve 2025 closing time is provided",
        parent=store_node,
        critical=True
    )
    category_provided = evaluator.add_custom_node(
        result=bool(store.category and store.category.strip()),
        id=f"store_{idx+1}_category_provided",
        desc="Retail category is provided",
        parent=store_node,
        critical=True
    )
    url_provided = evaluator.add_custom_node(
        result=bool(store.reference_urls),
        id=f"store_{idx+1}_reference_url_provided",
        desc="At least one reference URL is provided",
        parent=store_node,
        critical=True
    )

    # Reference URL verification (critical per rubric)
    ref_leaf = evaluator.add_leaf(
        id=f"store_{idx+1}_reference_url",
        desc="A verifiable URL from 2025 confirming the store's holiday hours",
        parent=store_node,
        critical=True
    )
    first_url = _safe_first_url(store.reference_urls)
    ref_claim = (
        f"This webpage is a 2025 source and provides holiday hours information for {store.name or 'the store'}, "
        "including Black Friday 2025 and/or Christmas Eve/Christmas Day 2025."
    )
    await evaluator.verify(
        claim=ref_claim,
        node=ref_leaf,
        sources=first_url,
        additional_instruction=(
            "Confirm the page is relevant to 2025 (e.g., explicitly mentions '2025', 'Black Friday 2025', 'Christmas 2025', or date strings corresponding to 2025) "
            "and that it contains holiday hours information (opening/closing times) for this store. "
            "If the page appears to be from a different year or lacks holiday-specific hours, mark as not supported."
        ),
    )

    # Black Friday hours verification
    bf_leaf = evaluator.add_leaf(
        id=f"store_{idx+1}_black_friday_hours",
        desc="Store opened at 6:00 a.m. or earlier on Black Friday 2025 (November 28, 2025)",
        parent=store_node,
        critical=True
    )
    bf_claim = (
        f"According to the cited source, {store.name or 'the store'} opens at {store.black_friday_opening_time or 'an opening time'} "
        "on Black Friday 2025 (Friday, November 28, 2025), and that time is at or before 6:00 a.m. local time."
    )
    await evaluator.verify(
        claim=bf_claim,
        node=bf_leaf,
        sources=first_url,
        extra_prerequisites=[name_provided, bf_opening_provided, url_provided],
        additional_instruction=(
            "Read the holiday hours on the page and verify the Black Friday 2025 opening time. "
            "Pass if the opening time is earlier than or equal to 6:00 a.m. (e.g., 5:00 a.m., 6:00 a.m., midnight). "
            "If the page indicates 7:00 a.m. or later, or lacks Black Friday 2025 specifics, fail."
        ),
    )

    # Christmas Eve closing verification
    cve_leaf = evaluator.add_leaf(
        id=f"store_{idx+1}_christmas_eve_hours",
        desc="Store closed at 6:00 p.m. or earlier on Christmas Eve 2025 (December 24, 2025)",
        parent=store_node,
        critical=True
    )
    cve_claim = (
        f"According to the cited source, {store.name or 'the store'} closes at {store.christmas_eve_closing_time or 'a closing time'} "
        "on Christmas Eve 2025 (Wednesday, December 24, 2025), and that time is at or before 6:00 p.m. local time."
    )
    await evaluator.verify(
        claim=cve_claim,
        node=cve_leaf,
        sources=first_url,
        extra_prerequisites=[name_provided, cve_closing_provided, url_provided],
        additional_instruction=(
            "Verify the Christmas Eve 2025 closing time on the page. Pass if the closing time is 6:00 p.m. or earlier "
            "(e.g., 4:00 p.m., 5:00 p.m., 6:00 p.m.). If the page indicates later than 6:00 p.m. or lacks 2025 specifics, fail."
        ),
    )

    # Christmas Day closure verification
    xmas_day_leaf = evaluator.add_leaf(
        id=f"store_{idx+1}_christmas_day_closure",
        desc="Store was closed on Christmas Day 2025 (December 25, 2025)",
        parent=store_node,
        critical=True
    )
    xmas_day_claim = (
        f"According to the cited source, {store.name or 'the store'} is closed on Christmas Day 2025 (Thursday, December 25, 2025)."
    )
    await evaluator.verify(
        claim=xmas_day_claim,
        node=xmas_day_leaf,
        sources=first_url,
        extra_prerequisites=[name_provided, url_provided],
        additional_instruction=(
            "Look for explicit statements such as 'Closed on Christmas Day' or equivalent for 2025. "
            "If the page suggests open or lacks 2025 Christmas Day info, mark as not supported."
        ),
    )

    # Category distinction verification
    cat_leaf = evaluator.add_leaf(
        id=f"store_{idx+1}_category_distinction",
        desc="Store represents a distinct retail category not used by other stores in the answer",
        parent=store_node,
        critical=True
    )
    other_categories = [c for j, c in enumerate(all_categories) if j != idx and c]
    cat_claim = (
        f"The retail category '{(store.category or '').strip()}' for {store.name or 'the store'} is not used by the other stores: {other_categories}."
    )
    await evaluator.verify(
        claim=cat_claim,
        node=cat_leaf,
        extra_prerequisites=[category_provided],
        additional_instruction=(
            "Judge distinctness at the broad category level (e.g., 'department store' vs 'electronics' vs 'home improvement' vs 'discount retailer'). "
            "Treat close synonyms as the same category. Pass only if this store's category is fundamentally different from the other stores provided."
        ),
    )


# --------------------------------------------------------------------------- #
# Main evaluation entry                                                       #
# --------------------------------------------------------------------------- #
async def evaluate_answer(
    client: LLMClient,
    answer: str,
    agent_name: str,
    answer_name: str,
    cache: CacheFileSys,
    semaphore: asyncio.Semaphore,
    logger: logging.Logger,
    model: str = "o4-mini",
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
        default_model=model,
    )

    extracted = await evaluator.extract(
        prompt=prompt_extract_holiday_stores(),
        template_class=HolidayStoresExtraction,
        extraction_name="holiday_stores_2025",
    )

    stores = _pad_to_three(extracted.stores)
    categories_list: List[Optional[str]] = [s.category for s in stores]

    # Build verification subtrees for three stores
    for i in range(3):
        await verify_store(
            evaluator=evaluator,
            parent_node=root,
            store=stores[i],
            idx=i,
            all_categories=categories_list,
        )

    return evaluator.get_summary()