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
TASK_ID = "holiday_ops_2025_2026"
TASK_DESCRIPTION = (
    "You are planning holiday shopping and dining across the Christmas and New Year period (December 24, 2025 through January 1, 2026) in the United States. "
    "You need to identify four different national retail and food service chains that meet specific holiday operating criteria:\n\n"
    "1. 24/7/365 Restaurant: Identify a national restaurant chain that operates 24 hours a day, 365 days a year, including Christmas Eve, Christmas Day, New Year's Eve, and New Year's Day. "
    "This chain must never close for any holiday (Thanksgiving, Easter, Memorial Day, Independence Day, Labor Day, Christmas, or New Year's Day) and this policy must apply to all or nearly all locations nationwide.\n\n"
    "2. Christmas Eve Early-Closing Retailer: Identify a major national retail chain (grocery store or big-box retailer) that closes at exactly 6:00 p.m. local time on Christmas Eve (December 24, 2025) and remains closed on Christmas Day (December 25, 2025). "
    "This closing time must apply nationwide or to most locations.\n\n"
    "3. Christmas Day Pharmacy: Identify a national pharmacy chain that keeps its stores open on Christmas Day (December 25, 2025), though pharmacy hours may vary by location. The chain should explicitly state that stores remain open on Christmas.\n\n"
    "4. Seven-Holiday Warehouse Club: Identify a warehouse club chain that closes for exactly seven (7) federal holidays per year. These seven holidays must be: New Year's Day, Easter Sunday, Memorial Day, Independence Day (July 4), Labor Day, Thanksgiving Day, and Christmas Day. "
    "The chain must be open on all other days of the year.\n\n"
    "For each chain, provide the chain's name, confirmation that it meets all specified criteria, and at least one reference URL supporting your answer. "
    "All answers must be based on official company policies or reliable news sources from 2025-2026."
)


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class ChainEntry(BaseModel):
    name: Optional[str] = None
    urls: List[str] = Field(default_factory=list)


class HolidayOpsExtraction(BaseModel):
    always_open_restaurant: Optional[ChainEntry] = None
    early_close_retailer: Optional[ChainEntry] = None
    christmas_day_pharmacy: Optional[ChainEntry] = None
    seven_holiday_warehouse: Optional[ChainEntry] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_holiday_ops() -> str:
    return """
    Extract the four chains and their source URLs from the answer. For each category, extract:
    - name: the chain’s name as written in the answer (string)
    - urls: all reference URLs (array of URLs) explicitly cited for that category

    The categories are:
    1) always_open_restaurant
       Meaning: a national restaurant chain that operates 24/7/365 and never closes, including Christmas and New Year holidays.
    2) early_close_retailer
       Meaning: a major national retail or grocery chain that closes at exactly 6:00 p.m. local time on Christmas Eve (Dec 24, 2025) and is closed on Christmas Day (Dec 25, 2025).
    3) christmas_day_pharmacy
       Meaning: a national pharmacy chain that keeps stores open on Christmas Day (Dec 25, 2025), with pharmacy hours possibly varying by location.
    4) seven_holiday_warehouse
       Meaning: a warehouse club that closes for exactly seven holidays: New Year's Day, Easter Sunday, Memorial Day, Independence Day (July 4), Labor Day, Thanksgiving Day, and Christmas Day, and is open on all other days.

    Rules:
    - Only extract URLs that are explicitly present in the answer (plain or Markdown).
    - Include all URLs cited for each category; if none provided, return an empty array.
    - If a category is missing from the answer, set it to null.
    - If a chain name is missing, set the name to null.
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def _normalize_urls(urls: List[str]) -> List[str]:
    out: List[str] = []
    for u in urls:
        if not u:
            continue
        u = u.strip().strip(".,);]")
        if not u:
            continue
        if not (u.startswith("http://") or u.startswith("https://")):
            if u.startswith("www."):
                u = "http://" + u
            else:
                # skip malformed
                continue
        out.append(u)
    # de-duplicate preserving order
    seen = set()
    deduped: List[str] = []
    for u in out:
        if u not in seen:
            deduped.append(u)
            seen.add(u)
    return deduped


def _has_nonempty_name(entry: Optional[ChainEntry]) -> bool:
    return bool(entry and entry.name and entry.name.strip())


def _has_valid_urls(entry: Optional[ChainEntry]) -> bool:
    return bool(entry and len(_normalize_urls(entry.urls)) > 0)


# --------------------------------------------------------------------------- #
# Verification builders                                                       #
# --------------------------------------------------------------------------- #
async def build_always_open_restaurant_checks(evaluator: Evaluator, parent, entry: Optional[ChainEntry]) -> None:
    node = evaluator.add_parallel(
        id="always_open_restaurant",
        desc="Identify a restaurant chain that never closes, operating 24/7/365",
        parent=parent,
        critical=False
    )

    name = (entry.name or "").strip() if entry else ""
    urls = _normalize_urls(entry.urls if entry else [])

    # Existence checks (critical)
    evaluator.add_custom_node(
        result=_has_nonempty_name(entry),
        id="always_open_restaurant_chain_name",
        desc="Provide the name of the restaurant chain",
        parent=node,
        critical=True
    )
    evaluator.add_custom_node(
        result=_has_valid_urls(entry),
        id="always_open_restaurant_reference_url",
        desc="Provide a reference URL documenting the 24/7/365 operations policy",
        parent=node,
        critical=True
    )

    # Helper to create and verify a leaf
    async def _verify_leaf(leaf_id: str, desc: str, claim: str, critical: bool = True, add_ins: Optional[str] = None):
        lf = evaluator.add_leaf(
            id=leaf_id,
            desc=desc,
            parent=node,
            critical=critical
        )
        await evaluator.verify(
            claim=claim,
            node=lf,
            sources=urls,
            additional_instruction=add_ins or (
                "Use official company policy pages or reliable 2025–2026 news references. "
                "If the source states 24/7/365 or open every day including holidays, that supports the claim for specific dates. "
                "Allow 'all or nearly all locations' as meaning chain-wide policy with rare exceptions."
            )
        )

    # Holiday-specific 24-hour claims (critical)
    await _verify_leaf(
        "always_open_restaurant_christmas_eve_24h",
        "Operates 24 hours on Christmas Eve (December 24, 2025)",
        f"{name} operates 24 hours on December 24, 2025 (Christmas Eve)."
    )
    await _verify_leaf(
        "always_open_restaurant_christmas_day_24h",
        "Operates 24 hours on Christmas Day (December 25, 2025)",
        f"{name} operates 24 hours on December 25, 2025 (Christmas Day)."
    )
    await _verify_leaf(
        "always_open_restaurant_new_years_eve_24h",
        "Operates 24 hours on New Year's Eve (December 31, 2025)",
        f"{name} operates 24 hours on December 31, 2025 (New Year's Eve)."
    )
    await _verify_leaf(
        "always_open_restaurant_new_years_day_24h",
        "Operates 24 hours on New Year's Day (January 1, 2026)",
        f"{name} operates 24 hours on January 1, 2026 (New Year's Day)."
    )
    await _verify_leaf(
        "always_open_restaurant_thanksgiving_24h",
        "Operates 24 hours on Thanksgiving Day",
        f"{name} operates 24 hours on Thanksgiving Day."
    )
    await _verify_leaf(
        "always_open_restaurant_easter_24h",
        "Operates 24 hours on Easter Sunday",
        f"{name} operates 24 hours on Easter Sunday."
    )
    await _verify_leaf(
        "always_open_restaurant_memorial_day_24h",
        "Operates 24 hours on Memorial Day",
        f"{name} operates 24 hours on Memorial Day."
    )
    await _verify_leaf(
        "always_open_restaurant_independence_day_24h",
        "Operates 24 hours on Independence Day",
        f"{name} operates 24 hours on Independence Day (July 4)."
    )
    await _verify_leaf(
        "always_open_restaurant_labor_day_24h",
        "Operates 24 hours on Labor Day",
        f"{name} operates 24 hours on Labor Day."
    )

    # Policy-wide leaves (critical)
    await _verify_leaf(
        "always_open_restaurant_never_closes_policy",
        "Chain explicitly states it operates 365 days a year without closing",
        f"{name} operates 24 hours a day, 7 days a week, 365 days a year (never closes)."
    )
    await _verify_leaf(
        "always_open_restaurant_all_locations_policy",
        "This 24/7/365 policy applies to all or nearly all locations nationwide",
        f"For {name}, the 24/7/365 'never closes' policy applies to all or nearly all locations nationwide."
    )

    # Non-critical recognition/reputation
    await _verify_leaf(
        "always_open_restaurant_known_for_reliability",
        "Chain is recognized or famous for never closing, even during emergencies",
        f"{name} is widely recognized for being open 24/7 and rarely or never closing even during emergencies or severe weather.",
        critical=False
    )


async def build_early_close_store_checks(evaluator: Evaluator, parent, entry: Optional[ChainEntry]) -> None:
    node = evaluator.add_parallel(
        id="christmas_eve_early_close_store",
        desc="Identify a major retailer that closes at 6 p.m. on Christmas Eve",
        parent=parent,
        critical=False
    )

    name = (entry.name or "").strip() if entry else ""
    urls = _normalize_urls(entry.urls if entry else [])

    evaluator.add_custom_node(
        result=_has_nonempty_name(entry),
        id="early_close_store_name",
        desc="Provide the name of the major retail chain",
        parent=node,
        critical=True
    )
    evaluator.add_custom_node(
        result=_has_valid_urls(entry),
        id="early_close_store_reference_url",
        desc="Provide a reference URL supporting the Christmas Eve 6 p.m. closing time",
        parent=node,
        critical=True
    )

    async def _verify_leaf(leaf_id: str, desc: str, claim: str, add_ins: Optional[str] = None):
        lf = evaluator.add_leaf(
            id=leaf_id,
            desc=desc,
            parent=node,
            critical=True
        )
        await evaluator.verify(
            claim=claim,
            node=lf,
            sources=urls,
            additional_instruction=add_ins or (
                "Prefer official 2025 holiday-hours pages or credible 2025–2026 news. "
                "Allow phrasing like '6 pm'/'6 p.m.' and 'local time'. "
                "If the source says 'most stores close at 6 p.m.' that counts as nationwide or near-nationwide."
            )
        )

    await _verify_leaf(
        "early_close_store_christmas_eve_6pm",
        "Closes at 6:00 p.m. local time on Christmas Eve (December 24, 2025)",
        f"{name} closes at exactly 6:00 p.m. local time on December 24, 2025 (Christmas Eve)."
    )
    await _verify_leaf(
        "early_close_store_christmas_day_closed",
        "Closed on Christmas Day (December 25, 2025)",
        f"{name} stores are closed on December 25, 2025 (Christmas Day)."
    )
    await _verify_leaf(
        "early_close_store_nationwide_policy",
        "This closing time applies nationwide or to most locations",
        f"The 6:00 p.m. Christmas Eve closing time applies nationwide (or to most locations) for {name}."
    )


async def build_pharmacy_christmas_checks(evaluator: Evaluator, parent, entry: Optional[ChainEntry]) -> None:
    node = evaluator.add_parallel(
        id="pharmacy_christmas_operations",
        desc="Identify a pharmacy chain open on Christmas Day with varying hours",
        parent=parent,
        critical=False
    )

    name = (entry.name or "").strip() if entry else ""
    urls = _normalize_urls(entry.urls if entry else [])

    evaluator.add_custom_node(
        result=_has_nonempty_name(entry),
        id="pharmacy_chain_name",
        desc="Provide the name of the pharmacy chain",
        parent=node,
        critical=True
    )
    evaluator.add_custom_node(
        result=_has_valid_urls(entry),
        id="pharmacy_reference_url",
        desc="Provide a reference URL documenting Christmas Day operations",
        parent=node,
        critical=True
    )

    async def _verify_leaf(leaf_id: str, desc: str, claim: str, critical: bool = True, add_ins: Optional[str] = None):
        lf = evaluator.add_leaf(
            id=leaf_id,
            desc=desc,
            parent=node,
            critical=critical
        )
        await evaluator.verify(
            claim=claim,
            node=lf,
            sources=urls,
            additional_instruction=add_ins or (
                "Prefer official company pages or reliable 2025 news. "
                "The chain should explicitly state stores are open on Christmas Day; hours may vary by location."
            )
        )

    await _verify_leaf(
        "pharmacy_christmas_day_open",
        "Stores are open on Christmas Day (December 25, 2025)",
        f"{name} stores are open on December 25, 2025 (Christmas Day)."
    )
    await _verify_leaf(
        "pharmacy_hours_vary",
        "Pharmacy hours may vary by location on Christmas Day",
        f"At {name}, pharmacy hours may vary by location on Christmas Day."
    )
    await _verify_leaf(
        "pharmacy_some_reduced_hours",
        "Some stores or pharmacies may have reduced hours on Christmas",
        f"Some {name} stores or their pharmacies may have reduced hours on Christmas Day.",
        critical=False
    )


async def build_warehouse_seven_holiday_checks(evaluator: Evaluator, parent, entry: Optional[ChainEntry]) -> None:
    node = evaluator.add_parallel(
        id="seven_holiday_closure_warehouse",
        desc="Identify a warehouse club that closes for exactly 7 federal holidays",
        parent=parent,
        critical=False
    )

    name = (entry.name or "").strip() if entry else ""
    urls = _normalize_urls(entry.urls if entry else [])

    evaluator.add_custom_node(
        result=_has_nonempty_name(entry),
        id="warehouse_club_name",
        desc="Provide the name of the warehouse club chain",
        parent=node,
        critical=True
    )
    evaluator.add_custom_node(
        result=_has_valid_urls(entry),
        id="warehouse_reference_url",
        desc="Provide a reference URL listing all 7 holiday closures",
        parent=node,
        critical=True
    )

    async def _verify_leaf(leaf_id: str, desc: str, claim: str, add_ins: Optional[str] = None):
        lf = evaluator.add_leaf(
            id=leaf_id,
            desc=desc,
            parent=node,
            critical=True
        )
        await evaluator.verify(
            claim=claim,
            node=lf,
            sources=urls,
            additional_instruction=add_ins or (
                "Use an official holiday schedule page or a reliable 2025–2026 summary. "
                "Confirm the chain closes on exactly these seven holidays and is open on all other days."
            )
        )

    # Exactly seven closures (overall)
    await _verify_leaf(
        "warehouse_exactly_seven_closures",
        "Closes for exactly 7 holidays per year",
        f"{name} closes for exactly seven holidays per year."
    )

    # Individual holiday closures (all critical)
    await _verify_leaf(
        "warehouse_new_years_day_closed",
        "Closed on New Year's Day",
        f"{name} is closed on New Year's Day."
    )
    await _verify_leaf(
        "warehouse_easter_sunday_closed",
        "Closed on Easter Sunday",
        f"{name} is closed on Easter Sunday."
    )
    await _verify_leaf(
        "warehouse_memorial_day_closed",
        "Closed on Memorial Day",
        f"{name} is closed on Memorial Day."
    )
    await _verify_leaf(
        "warehouse_independence_day_closed",
        "Closed on Independence Day (July 4)",
        f"{name} is closed on Independence Day (July 4)."
    )
    await _verify_leaf(
        "warehouse_labor_day_closed",
        "Closed on Labor Day",
        f"{name} is closed on Labor Day."
    )
    await _verify_leaf(
        "warehouse_thanksgiving_closed",
        "Closed on Thanksgiving Day",
        f"{name} is closed on Thanksgiving Day."
    )
    await _verify_leaf(
        "warehouse_christmas_closed",
        "Closed on Christmas Day",
        f"{name} is closed on Christmas Day."
    )

    # Only these seven holidays
    lf_only_seven = evaluator.add_leaf(
        id="warehouse_only_these_seven",
        desc="These are the only 7 holidays for which the warehouse closes",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim=f"Apart from those seven holidays, {name} is open on all other days of the year.",
        node=lf_only_seven,
        sources=urls,
        additional_instruction="Confirm the policy indicates these are the only closures and the club is open on all other days."
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
    Evaluate a single answer for the Holiday Operations Plan task and return a structured result dictionary.
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

    # Extract all four chains and their URLs
    extracted = await evaluator.extract(
        prompt=prompt_extract_holiday_ops(),
        template_class=HolidayOpsExtraction,
        extraction_name="holiday_ops_extraction"
    )

    # Top-level plan node (non-critical to allow partial scoring)
    plan_node = evaluator.add_parallel(
        id="holiday_operations_plan",
        desc="Comprehensive holiday operations plan identifying stores and restaurants with specific holiday hour requirements",
        parent=root,
        critical=False
    )

    # Build checks for each category
    await build_always_open_restaurant_checks(evaluator, plan_node, extracted.always_open_restaurant)
    await build_early_close_store_checks(evaluator, plan_node, extracted.early_close_retailer)
    await build_pharmacy_christmas_checks(evaluator, plan_node, extracted.christmas_day_pharmacy)
    await build_warehouse_seven_holiday_checks(evaluator, plan_node, extracted.seven_holiday_warehouse)

    # Return summary
    return evaluator.get_summary()