import asyncio
import logging
import re
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "holiday_shopping_ca_2025"
TASK_DESCRIPTION = (
    "You are planning last-minute holiday shopping in California for Christmas 2025. "
    "To help shoppers find convenient options, identify 4 major retail chain stores that each meet ALL of the following requirements:\n"
    "1. Have at least 50 store locations in California\n"
    "2. Close at 8:00 PM or earlier on Christmas Eve (December 24, 2025)\n"
    "3. Are completely closed on Christmas Day (December 25, 2025)\n"
    "4. Offer same-day pickup, curbside pickup, or in-store pickup services\n\n"
    "For each of the 4 retailers, provide:\n"
    "- The retailer's name\n"
    "- The exact closing time on Christmas Eve 2025\n"
    "- The number of store locations in California\n"
    "- A description of the pickup service(s) offered\n"
    "- A reference URL confirming the Christmas Eve 2025 hours\n"
    "- A reference URL confirming the pickup service availability"
)

CHRISTMAS_EVE_DATE_STR = "December 24, 2025"
CHRISTMAS_DAY_DATE_STR = "December 25, 2025"


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class RetailerEntry(BaseModel):
    name: Optional[str] = None
    ca_location_count: Optional[str] = None
    christmas_eve_closing_time: Optional[str] = None
    christmas_day_closed_statement: Optional[str] = None
    pickup_services_description: Optional[str] = None
    hours_reference_url: Optional[str] = None
    pickup_reference_url: Optional[str] = None


class RetailersExtraction(BaseModel):
    retailers: List[RetailerEntry] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_retailers(k: int = 4) -> str:
    return f"""
Extract up to {k} retailers exactly as stated in the answer text. Preserve the original order from the answer. 
For each retailer mentioned, extract the following fields:
- name: The retailer's name (string).
- ca_location_count: The number of store locations in California as explicitly stated in the answer text (e.g., "120", "over 100", "at least 60"). If not explicitly stated in the answer, return null.
- christmas_eve_closing_time: The exact closing time on Christmas Eve 2025 (e.g., "7:00 PM", "6 PM"). If not explicitly stated in the answer, return null.
- christmas_day_closed_statement: The answer's statement about Christmas Day 2025 closure (e.g., "Closed", "Closed on Christmas Day"). If not explicitly stated, return null.
- pickup_services_description: The answer's description of pickup services offered (e.g., "same-day pickup and curbside pickup", "in-store pickup"). If not explicitly stated, return null.
- hours_reference_url: The specific URL in the answer that is intended to confirm Christmas Eve 2025 hours (or 'Christmas Eve' holiday hours). If not provided, return null.
- pickup_reference_url: The specific URL in the answer that is intended to confirm pickup service availability. If not provided, return null.

IMPORTANT:
- Only extract information explicitly found in the answer text. Do not infer or invent any value.
- For ca_location_count, extract the string exactly as written (do not normalize).
- For times, keep the exact formatting provided in the answer text (e.g., "8 PM", "8:00 p.m.").
- URLs can appear as plain links or markdown links; always extract the actual URL.
- If fewer than {k} retailers are mentioned, return fewer. If more are mentioned, include all; the evaluator will only use the first {k}.
    """.strip()


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
_INT_PATTERN = re.compile(r"\b(\d{1,5})\b")
_TIME_PATTERN = re.compile(
    r"\b(?:(1[0-2]|0?[1-9])(?::([0-5]\d))?\s*(?:a\.?m\.?|p\.?m\.?|am|pm))\b",
    flags=re.IGNORECASE
)
_NOON_PATTERN = re.compile(r"\bnoon\b", flags=re.IGNORECASE)
_MIDNIGHT_PATTERN = re.compile(r"\bmidnight\b", flags=re.IGNORECASE)


def first_int(text: Optional[str]) -> Optional[int]:
    if not text:
        return None
    m = _INT_PATTERN.search(text)
    if not m:
        return None
    try:
        return int(m.group(1))
    except Exception:
        return None


def has_time_string(text: Optional[str]) -> bool:
    if not text:
        return False
    if _TIME_PATTERN.search(text):
        return True
    if _NOON_PATTERN.search(text) or _MIDNIGHT_PATTERN.search(text):
        return True
    return False


def non_empty(s: Optional[str]) -> bool:
    return bool(s and str(s).strip())


# --------------------------------------------------------------------------- #
# Verification for a single retailer                                          #
# --------------------------------------------------------------------------- #
async def verify_single_retailer(
    evaluator: Evaluator,
    parent,
    retailer: RetailerEntry,
    index: int,
) -> None:
    """
    Build verification sub-tree for a single retailer following the rubric.
    """
    rid = index + 1
    retailer_name = retailer.name or f"Retailer #{rid}"

    # Parent node for this retailer (parallel aggregation, non-critical to allow partial credit across retailers)
    rnode = evaluator.add_parallel(
        id=f"retailer_{rid}",
        desc=f"{['First','Second','Third','Fourth'][index]} qualifying retail chain",
        parent=parent,
        critical=False
    )

    # 1) Location_Count_Stated (Critical) - existence and explicitness of the count in the answer
    count_str = retailer.ca_location_count
    count_num = first_int(count_str)
    evaluator.add_custom_node(
        result=(count_num is not None),
        id=f"retailer_{rid}_location_count_stated",
        desc="The specific number of California locations is explicitly stated",
        parent=rnode,
        critical=True
    )

    # 2) California_Presence (Critical) - numeric threshold ≥ 50 based on the stated count
    evaluator.add_custom_node(
        result=(count_num is not None and count_num >= 50),
        id=f"retailer_{rid}_california_presence",
        desc="Retailer has at least 50 store locations in California",
        parent=rnode,
        critical=True
    )

    # 3) Reference_Christmas_Eve_Hours (Critical) - validate the hours reference URL by content
    hours_url = retailer.hours_reference_url if non_empty(retailer.hours_reference_url) else None
    if not hours_url:
        evaluator.add_leaf(
            id=f"retailer_{rid}_ref_christmas_eve_hours",
            desc="Valid reference URL provided for Christmas Eve 2025 hours",
            parent=rnode,
            critical=True,
            score=0.0,
            status="failed"
        )
    else:
        ref_hours_leaf = evaluator.add_leaf(
            id=f"retailer_{rid}_ref_christmas_eve_hours",
            desc="Valid reference URL provided for Christmas Eve 2025 hours",
            parent=rnode,
            critical=True
        )
        ref_hours_claim = (
            f"This webpage is an official or reputable source for {retailer_name}'s store hours, "
            f"and explicitly mentions Christmas Eve or the date {CHRISTMAS_EVE_DATE_STR}, with a closing time or "
            f"a holiday schedule that applies to the 2025 holiday season."
        )
        await evaluator.verify(
            claim=ref_hours_claim,
            node=ref_hours_leaf,
            sources=hours_url,
            additional_instruction=(
                "Accept pages that state 'Christmas Eve hours', 'Dec 24' or explicitly show hours for the date. "
                "A location-specific store-hours page is acceptable. Reject pages unrelated to hours."
            )
        )

    # 4) Exact_Closing_Time_Stated (Critical) - ensure the exact time is given in the answer
    evaluator.add_custom_node(
        result=has_time_string(retailer.christmas_eve_closing_time),
        id=f"retailer_{rid}_exact_closing_time_stated",
        desc="The exact Christmas Eve closing time is explicitly stated",
        parent=rnode,
        critical=True
    )

    # 5) Christmas_Eve_Closing (Critical) - verify threshold (≤ 8:00 PM) against the hours URL
    ce_leaf = evaluator.add_leaf(
        id=f"retailer_{rid}_christmas_eve_closing",
        desc="Retailer closes at 8:00 PM or earlier on Christmas Eve 2025 (December 24)",
        parent=rnode,
        critical=True
    )
    ce_time_str = retailer.christmas_eve_closing_time or ""
    ce_claim = (
        f"According to the provided page, on {CHRISTMAS_EVE_DATE_STR}, {retailer_name} "
        f"stores close at {ce_time_str} local time, which is at or before 8:00 PM."
    )
    await evaluator.verify(
        claim=ce_claim,
        node=ce_leaf,
        sources=hours_url,  # May be None; if the reference URL failed, this leaf will be skipped by preconditions
        additional_instruction=(
            "Verify the closing time for Christmas Eve (Dec 24, 2025). "
            "If multiple locations are listed, it is acceptable if the general policy or example locations "
            "show a closing time not later than 8:00 PM local time. "
            "If the page indicates a later time (e.g., 9 PM) or contradicts the claim, mark as incorrect."
        )
    )

    # 6) Christmas_Day_Closure (Critical) - verify closure on Christmas Day against the hours URL
    cd_leaf = evaluator.add_leaf(
        id=f"retailer_{rid}_christmas_day_closure",
        desc="Retailer is closed on Christmas Day 2025 (December 25)",
        parent=rnode,
        critical=True
    )
    cd_claim = (
        f"On {CHRISTMAS_DAY_DATE_STR}, {retailer_name} is closed (no operating hours). "
        f"Accept equivalent statements like 'Closed on Christmas Day'."
    )
    await evaluator.verify(
        claim=cd_claim,
        node=cd_leaf,
        sources=hours_url,
        additional_instruction=(
            "Confirm the page shows 'Closed' for Dec 25, 2025 or an equivalent statement for Christmas Day. "
            "Generic 'Closed on Christmas Day' without a year is acceptable if the page is about holiday hours."
        )
    )

    # 7) Reference_Pickup_Service (Critical) - validate the pickup service reference URL by content
    pickup_url = retailer.pickup_reference_url if non_empty(retailer.pickup_reference_url) else None
    if not pickup_url:
        evaluator.add_leaf(
            id=f"retailer_{rid}_ref_pickup_service",
            desc="Valid reference URL provided for pickup service availability",
            parent=rnode,
            critical=True,
            score=0.0,
            status="failed"
        )
    else:
        ref_pickup_leaf = evaluator.add_leaf(
            id=f"retailer_{rid}_ref_pickup_service",
            desc="Valid reference URL provided for pickup service availability",
            parent=rnode,
            critical=True
        )
        ref_pickup_claim = (
            f"This webpage from or about {retailer_name} documents the availability of a pickup service, "
            f"such as same-day pickup, curbside pickup, or in-store pickup (BOPIS/Order Pickup/Drive Up)."
        )
        await evaluator.verify(
            claim=ref_pickup_claim,
            node=ref_pickup_leaf,
            sources=pickup_url,
            additional_instruction=(
                "The page should explicitly mention a pickup modality: 'same-day pickup', 'curbside pickup', "
                "'in-store pickup', 'BOPIS', 'Order Pickup', 'Drive Up', or clear equivalents."
            )
        )

    # 8) Pickup_Description_Provided (Critical) - ensure the description is in the answer
    evaluator.add_custom_node(
        result=non_empty(retailer.pickup_services_description),
        id=f"retailer_{rid}_pickup_description_provided",
        desc="A description of the pickup service(s) offered is provided",
        parent=rnode,
        critical=True
    )

    # 9) Pickup_Service (Critical) - verify the retailer offers at least one of the eligible pickup types
    pickup_leaf = evaluator.add_leaf(
        id=f"retailer_{rid}_pickup_service",
        desc="Retailer offers same-day pickup, curbside pickup, or in-store pickup service",
        parent=rnode,
        critical=True
    )
    pickup_claim = (
        f"{retailer_name} offers at least one of the following services: same-day pickup, curbside pickup, or in-store pickup "
        f"(including synonyms such as BOPIS, Order Pickup, or Drive Up)."
    )
    await evaluator.verify(
        claim=pickup_claim,
        node=pickup_leaf,
        sources=pickup_url,  # May be None; if the reference URL failed, this leaf will be skipped by preconditions
        additional_instruction=(
            "Accept synonyms and branded names for pickup modalities (e.g., 'Order Pickup', 'Drive Up', 'BOPIS'). "
            "The page should clearly indicate that customers can place orders online and pick them up the same day, curbside, or inside the store."
        )
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
    Evaluate an answer for the California holiday shopping options (Christmas 2025) task.
    """
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,
        agent_name=agent_name,
        answer_name=answer_name,
        client=client,
        task_description="Identify 4 major retail chains in California meeting all specified criteria for Christmas 2025 shopping",
        answer=answer,
        global_cache=cache,
        global_semaphore=semaphore,
        logger=logger,
        default_model=model
    )

    # Extract retailers from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_retailers(k=4),
        template_class=RetailersExtraction,
        extraction_name="retailers_extraction"
    )

    # Prepare exactly 4 retailer entries (pad or truncate)
    retailers: List[RetailerEntry] = list(extracted.retailers or [])
    retailers = retailers[:4]
    while len(retailers) < 4:
        retailers.append(RetailerEntry())

    # Build and verify subtrees for each retailer
    tasks = []
    for i in range(4):
        tasks.append(verify_single_retailer(evaluator, root, retailers[i], i))
    # Execute sequentially to maintain stable dependency logging (can also be awaited concurrently)
    for t in tasks:
        await t

    # Return structured evaluation summary
    return evaluator.get_summary()