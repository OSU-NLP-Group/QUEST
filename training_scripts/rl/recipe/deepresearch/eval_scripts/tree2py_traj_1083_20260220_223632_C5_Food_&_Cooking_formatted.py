import asyncio
import logging
import re
from typing import Any, List, Optional, Dict

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "holiday_hours_2025_2026"
TASK_DESCRIPTION = (
    "Identify four different food retail establishments meeting the specified 2025–2026 holiday-hours criteria, "
    "each with a confirming reference URL."
)

THANKSGIVING_2025 = "November 27, 2025"
CHRISTMAS_EVE_2025 = "December 24, 2025"
CHRISTMAS_DAY_2025 = "December 25, 2025"
NEW_YEARS_DAY_2026 = "January 1, 2026"


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class HolidayEstablishment(BaseModel):
    """An establishment and the URLs the answer cites for it."""
    name: Optional[str] = None
    urls: List[str] = Field(default_factory=list)


class HolidayEstablishmentsExtraction(BaseModel):
    """Container for up to four establishments as presented in the answer."""
    items: List[HolidayEstablishment] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_establishments() -> str:
    return """
    Extract up to four (4) food retail establishments from the answer, preserving the answer's order.
    For each establishment, return:
    - name: The establishment name as written in the answer (string). If missing, set to null.
    - urls: An array of all URLs explicitly cited in the answer for this establishment (only real URLs; include protocol).
    Return a JSON object with a single field:
    {
      "items": [
        { "name": "...", "urls": ["...", "..."] },
        ...
      ]
    }
    If the answer lists more than four establishments, include ONLY the first four.
    If fewer than four are listed, include what is present.
    Apply the SPECIAL RULES FOR URL SOURCES EXTRACTION (only extract explicit URLs shown in the answer).
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def normalize_name(name: str) -> str:
    """Normalize establishment names for distinctness checks."""
    return re.sub(r'[^a-z0-9]', '', name.lower()) if name else ""


def pad_to_four(items: List[HolidayEstablishment]) -> List[HolidayEstablishment]:
    """Ensure we have exactly four items by padding with empty placeholders."""
    items = items[:4]
    while len(items) < 4:
        items.append(HolidayEstablishment())
    return items


# --------------------------------------------------------------------------- #
# Verification builders                                                       #
# --------------------------------------------------------------------------- #
async def build_item_1_thanksgiving_grocery_chain(
    evaluator: Evaluator,
    parent_node,
    item: HolidayEstablishment,
    idx_label: str = "item_1"
) -> None:
    """
    Item 1: Major grocery chain open on Thanksgiving Day 2025 and remains open until at least 4:00 PM local time.
    """
    node = evaluator.add_parallel(
        id=idx_label,
        desc=f"Major grocery chain open on Thanksgiving Day {THANKSGIVING_2025} and open until at least 4:00 PM",
        parent=parent_node,
        critical=False
    )

    # Existence checks (critical siblings to gate downstream verifications)
    name_ok = bool(item.name and item.name.strip())
    urls_ok = bool(item.urls)
    evaluator.add_custom_node(
        result=name_ok,
        id=f"{idx_label}_establishment_name_provided",
        desc="An establishment name is provided for item 1",
        parent=node,
        critical=True
    )
    evaluator.add_custom_node(
        result=urls_ok,
        id=f"{idx_label}_reference_url_provided",
        desc="At least one reference URL is provided for item 1 that supports the Thanksgiving-hours claim(s)",
        parent=node,
        critical=True
    )

    # Type check: major grocery chain (critical leaf)
    type_leaf = evaluator.add_leaf(
        id=f"{idx_label}_is_major_grocery_chain",
        desc="The establishment for item 1 is a major grocery chain",
        parent=node,
        critical=True
    )
    claim_type = f"{item.name} is a major grocery chain or supermarket chain."
    await evaluator.verify(
        claim=claim_type,
        node=type_leaf,
        sources=item.urls,
        additional_instruction=(
            "Use the provided URLs (e.g., official site, Wikipedia, reputable news) to confirm the chain is a major "
            "grocery/supermarket chain (large, multi-state, widely recognized)."
        ),
    )

    # Open on Thanksgiving Day (critical leaf)
    open_leaf = evaluator.add_leaf(
        id=f"{idx_label}_open_on_thanksgiving_2025",
        desc=f"Provided source confirms the chain is open on Thanksgiving Day ({THANKSGIVING_2025})",
        parent=node,
        critical=True
    )
    claim_open = f"{item.name} is open on Thanksgiving Day ({THANKSGIVING_2025})."
    await evaluator.verify(
        claim=claim_open,
        node=open_leaf,
        sources=item.urls,
        additional_instruction=(
            f"Confirm the source explicitly states that {item.name} (chain or stores) are open on {THANKSGIVING_2025}. "
            "Store-level pages count if representative for the chain."
        ),
    )

    # Open until at least 4:00 PM local time (critical leaf)
    until_leaf = evaluator.add_leaf(
        id=f"{idx_label}_open_until_at_least_4pm_thanksgiving",
        desc=f"Provided source confirms the chain remains open until at least 4:00 PM local time on Thanksgiving Day 2025",
        parent=node,
        critical=True
    )
    claim_until = (
        f"On Thanksgiving Day ({THANKSGIVING_2025}), {item.name} remains open until at least 4:00 PM local time."
    )
    await evaluator.verify(
        claim=claim_until,
        node=until_leaf,
        sources=item.urls,
        additional_instruction=(
            "Pass if the source shows a closing time at or after 4:00 PM (e.g., 4 PM, 5 PM). "
            "If hours vary, evidence must show at least representative locations open until 4 PM or later."
        ),
    )


async def build_item_2_christmas_day_coffee_chain(
    evaluator: Evaluator,
    parent_node,
    item: HolidayEstablishment,
    idx_label: str = "item_2"
) -> None:
    """
    Item 2: Coffee chain where most locations are open on Christmas Day 2025.
    """
    node = evaluator.add_parallel(
        id=idx_label,
        desc=f"Coffee chain where most locations are open on Christmas Day ({CHRISTMAS_DAY_2025})",
        parent=parent_node,
        critical=False
    )

    # Existence checks
    name_ok = bool(item.name and item.name.strip())
    urls_ok = bool(item.urls)
    evaluator.add_custom_node(
        result=name_ok,
        id=f"{idx_label}_establishment_name_provided",
        desc="An establishment name is provided for item 2",
        parent=node,
        critical=True
    )
    evaluator.add_custom_node(
        result=urls_ok,
        id=f"{idx_label}_reference_url_provided",
        desc="At least one reference URL is provided for item 2 that supports the Christmas Day open-status claim",
        parent=node,
        critical=True
    )

    # Type check: coffee chain
    type_leaf = evaluator.add_leaf(
        id=f"{idx_label}_is_coffee_chain",
        desc="The establishment for item 2 is primarily a coffee chain",
        parent=node,
        critical=True
    )
    claim_type = f"{item.name} is a coffee chain (primarily serving coffee)."
    await evaluator.verify(
        claim=claim_type,
        node=type_leaf,
        sources=item.urls,
        additional_instruction=(
            "Use the URLs to confirm it is a coffee chain (e.g., Wikipedia identifies as a coffeehouse chain; "
            "official site describes the brand as coffee-focused)."
        ),
    )

    # Most locations open on Christmas Day
    open_leaf = evaluator.add_leaf(
        id=f"{idx_label}_most_locations_open_christmas_day_2025",
        desc=f"Provided source confirms most locations are open on Christmas Day ({CHRISTMAS_DAY_2025})",
        parent=node,
        critical=True
    )
    claim_open = f"Most locations of {item.name} are open on Christmas Day ({CHRISTMAS_DAY_2025})."
    await evaluator.verify(
        claim=claim_open,
        node=open_leaf,
        sources=item.urls,
        additional_instruction=(
            "The source should indicate that most or the majority of locations/stores are open on Christmas Day 2025. "
            "Phrasing such as 'many', 'most', or 'majority of locations' qualifies; 'some' or 'select' alone does not."
        ),
    )


async def build_item_3_christmas_eve_early_close_grocery_chain(
    evaluator: Evaluator,
    parent_node,
    item: HolidayEstablishment,
    idx_label: str = "item_3"
) -> None:
    """
    Item 3: Major grocery chain open on Christmas Eve 2025 with a specific announced closing time of 5:00 PM or earlier.
    """
    node = evaluator.add_parallel(
        id=idx_label,
        desc=f"Major grocery chain open on Christmas Eve ({CHRISTMAS_EVE_2025}) with announced early close ≤ 5:00 PM",
        parent=parent_node,
        critical=False
    )

    # Existence checks
    name_ok = bool(item.name and item.name.strip())
    urls_ok = bool(item.urls)
    evaluator.add_custom_node(
        result=name_ok,
        id=f"{idx_label}_establishment_name_provided",
        desc="An establishment name is provided for item 3",
        parent=node,
        critical=True
    )
    evaluator.add_custom_node(
        result=urls_ok,
        id=f"{idx_label}_reference_url_provided",
        desc="At least one reference URL is provided for item 3 that supports the Christmas Eve hours claim(s)",
        parent=node,
        critical=True
    )

    # Type check: major grocery chain
    type_leaf = evaluator.add_leaf(
        id=f"{idx_label}_is_major_grocery_chain",
        desc="The establishment for item 3 is a major grocery chain",
        parent=node,
        critical=True
    )
    claim_type = f"{item.name} is a major grocery chain or supermarket chain."
    await evaluator.verify(
        claim=claim_type,
        node=type_leaf,
        sources=item.urls,
        additional_instruction=(
            "Use the URLs to confirm the chain is a major grocery/supermarket chain."
        ),
    )

    # Open on Christmas Eve
    open_leaf = evaluator.add_leaf(
        id=f"{idx_label}_open_on_christmas_eve_2025",
        desc=f"Provided source confirms the chain is open on Christmas Eve ({CHRISTMAS_EVE_2025})",
        parent=node,
        critical=True
    )
    claim_open = f"{item.name} is open on Christmas Eve ({CHRISTMAS_EVE_2025})."
    await evaluator.verify(
        claim=claim_open,
        node=open_leaf,
        sources=item.urls,
        additional_instruction=(
            f"Confirm the source explicitly states being open on {CHRISTMAS_EVE_2025}."
        ),
    )

    # Announced close time ≤ 5:00 PM
    close_leaf = evaluator.add_leaf(
        id=f"{idx_label}_announced_close_time_le_5pm_christmas_eve",
        desc="Provided source states a specific announced closing time on Christmas Eve 2025 that is 5:00 PM or earlier",
        parent=node,
        critical=True
    )
    claim_close = f"On Christmas Eve ({CHRISTMAS_EVE_2025}), {item.name} has an announced closing time at or before 5:00 PM."
    await evaluator.verify(
        claim=claim_close,
        node=close_leaf,
        sources=item.urls,
        additional_instruction=(
            "Pass if the source shows a specific closing time ≤ 5:00 PM (e.g., 4 PM, 5 PM). "
            "Location-level pages are acceptable if representative of the chain's announced early close."
        ),
    )


async def build_item_4_closed_thanksgiving_and_christmas_open_new_years(
    evaluator: Evaluator,
    parent_node,
    item: HolidayEstablishment,
    idx_label: str = "item_4"
) -> None:
    """
    Item 4: Major food retailer closed on Thanksgiving Day and Christmas Day 2025, but open on New Year's Day 2026.
    """
    node = evaluator.add_parallel(
        id=idx_label,
        desc=(
            f"Major food retailer closed on Thanksgiving Day ({THANKSGIVING_2025}) and Christmas Day ({CHRISTMAS_DAY_2025}), "
            f"but open on New Year's Day ({NEW_YEARS_DAY_2026})"
        ),
        parent=parent_node,
        critical=False
    )

    # Existence checks
    name_ok = bool(item.name and item.name.strip())
    urls_ok = bool(item.urls)
    evaluator.add_custom_node(
        result=name_ok,
        id=f"{idx_label}_establishment_name_provided",
        desc="An establishment name is provided for item 4",
        parent=node,
        critical=True
    )
    evaluator.add_custom_node(
        result=urls_ok,
        id=f"{idx_label}_reference_url_provided",
        desc="At least one reference URL is provided for item 4 that supports the holiday open/closed claim(s)",
        parent=node,
        critical=True
    )

    # Type check: major food retailer (grocery/supermarket)
    type_leaf = evaluator.add_leaf(
        id=f"{idx_label}_is_major_food_retailer",
        desc="The establishment for item 4 is a major food retailer (grocery store or supermarket)",
        parent=node,
        critical=True
    )
    claim_type = f"{item.name} is a major food retailer (grocery store/supermarket chain)."
    await evaluator.verify(
        claim=claim_type,
        node=type_leaf,
        sources=item.urls,
        additional_instruction=(
            "Use the URLs to confirm the retailer is a major grocery/supermarket chain."
        ),
    )

    # Closed on Thanksgiving Day 2025
    closed_thanks_leaf = evaluator.add_leaf(
        id=f"{idx_label}_closed_on_thanksgiving_2025",
        desc=f"Provided source confirms the retailer is closed on Thanksgiving Day 2025 ({THANKSGIVING_2025})",
        parent=node,
        critical=True
    )
    claim_closed_thanks = f"{item.name} is closed on Thanksgiving Day ({THANKSGIVING_2025})."
    await evaluator.verify(
        claim=claim_closed_thanks,
        node=closed_thanks_leaf,
        sources=item.urls,
        additional_instruction=(
            f"Confirm an explicit closure statement for {THANKSGIVING_2025}."
        ),
    )

    # Closed on Christmas Day 2025
    closed_christmas_leaf = evaluator.add_leaf(
        id=f"{idx_label}_closed_on_christmas_day_2025",
        desc=f"Provided source confirms the retailer is closed on Christmas Day 2025 ({CHRISTMAS_DAY_2025})",
        parent=node,
        critical=True
    )
    claim_closed_christmas = f"{item.name} is closed on Christmas Day ({CHRISTMAS_DAY_2025})."
    await evaluator.verify(
        claim=claim_closed_christmas,
        node=closed_christmas_leaf,
        sources=item.urls,
        additional_instruction=(
            f"Confirm an explicit closure statement for {CHRISTMAS_DAY_2025}."
        ),
    )

    # Open on New Year's Day 2026
    open_new_year_leaf = evaluator.add_leaf(
        id=f"{idx_label}_open_on_new_years_day_2026",
        desc=f"Provided source confirms the retailer is open on New Year's Day 2026 ({NEW_YEARS_DAY_2026})",
        parent=node,
        critical=True
    )
    claim_open_new_year = f"{item.name} is open on New Year's Day ({NEW_YEARS_DAY_2026})."
    await evaluator.verify(
        claim=claim_open_new_year,
        node=open_new_year_leaf,
        sources=item.urls,
        additional_instruction=(
            f"Confirm an explicit open/on-hours statement for {NEW_YEARS_DAY_2026}."
        ),
    )


# --------------------------------------------------------------------------- #
# Main evaluation entry                                                       #
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
    Evaluate an answer against the 2025–2026 holiday-hours criteria for four establishments.
    """
    # Initialize evaluator (root is non-critical to allow partial credit across items)
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

    # Extract establishments
    extraction = await evaluator.extract(
        prompt=prompt_extract_establishments(),
        template_class=HolidayEstablishmentsExtraction,
        extraction_name="holiday_establishments"
    )

    # Normalize to exactly 4 items (pad if needed)
    items = pad_to_four(extraction.items)

    # Root-level global checks (critical leaves)
    # 1) All four item entries present (names provided)
    four_present = all(bool(it.name and it.name.strip()) for it in items)
    evaluator.add_custom_node(
        result=four_present,
        id="all_four_item_entries_present",
        desc="Response provides four establishment entries corresponding to criteria/items 1–4 (one per criterion)",
        parent=root,
        critical=True
    )

    # 2) All four establishments are distinct (by normalized name)
    names_norm = [normalize_name(it.name or "") for it in items]
    distinct = len(set(n for n in names_norm if n)) == 4 if four_present else False
    evaluator.add_custom_node(
        result=distinct,
        id="all_four_establishments_distinct",
        desc="The four identified establishments are all different (no establishment is reused across items 1–4)",
        parent=root,
        critical=True
    )

    # Build per-item verification subtrees (parallel children under root)
    await build_item_1_thanksgiving_grocery_chain(evaluator, root, items[0], "item_1")
    await build_item_2_christmas_day_coffee_chain(evaluator, root, items[1], "item_2")
    await build_item_3_christmas_eve_early_close_grocery_chain(evaluator, root, items[2], "item_3")
    await build_item_4_closed_thanksgiving_and_christmas_open_new_years(evaluator, root, items[3], "item_4")

    # Optional: Add holiday date anchors for reference
    evaluator.add_custom_info(
        {
            "THANKSGIVING_2025": THANKSGIVING_2025,
            "CHRISTMAS_EVE_2025": CHRISTMAS_EVE_2025,
            "CHRISTMAS_DAY_2025": CHRISTMAS_DAY_2025,
            "NEW_YEARS_DAY_2026": NEW_YEARS_DAY_2026
        },
        info_type="date_anchors",
        info_name="holiday_dates"
    )

    return evaluator.get_summary()