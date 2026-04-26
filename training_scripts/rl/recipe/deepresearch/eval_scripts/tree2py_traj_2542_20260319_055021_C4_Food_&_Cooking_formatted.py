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
TASK_ID = "ma_thanksgiving_2025_food_hours"
TASK_DESCRIPTION = (
    "On Thanksgiving Day 2025 (Thursday, November 27, 2025), identify four food establishments operating in "
    "Massachusetts: two grocery store chains and two restaurant chains. The establishments must meet the following "
    "specific criteria: (1) One grocery store chain that is open on Thanksgiving Day but closes by 1:00 PM or earlier; "
    "(2) One grocery store chain that remains open on Thanksgiving Day until 5:00 PM or later; "
    "(3) One restaurant chain that operates continuously for 24 hours on Thanksgiving Day; "
    "(4) One fast-food restaurant chain that is open on Thanksgiving Day with operating hours during the daytime. "
    "For each establishment, provide the chain name and confirm its operating status on Thanksgiving Day 2025 in Massachusetts."
)

THANKSGIVING_2025_DATE_STR = "Thursday, November 27, 2025"
THANKSGIVING_2025_SHORT = "Nov 27, 2025"


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class Establishment(BaseModel):
    chain_name: Optional[str] = None
    sources: List[str] = Field(default_factory=list)
    thanksgiving_hours_text: Optional[str] = None
    category_hint: Optional[str] = None  # e.g., "grocery", "restaurant", "fast food"
    location_notes: Optional[str] = None  # any MA-specific notes if present


class ThanksgivingEstablishmentsExtraction(BaseModel):
    grocery_close_early: Optional[Establishment] = None
    grocery_open_late: Optional[Establishment] = None
    restaurant_24h: Optional[Establishment] = None
    fast_food_daytime: Optional[Establishment] = None


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_establishments() -> str:
    return f"""
    Extract four specific establishments from the answer, each as a chain operating in Massachusetts, with URL sources
    explicitly cited in the answer. For each item, extract:
      - chain_name: The chain name as written in the answer.
      - sources: An array of the URL(s) explicitly cited for this item in the answer (do not invent).
      - thanksgiving_hours_text: Any quoted or paraphrased statement about Thanksgiving Day hours from the answer.
      - category_hint: If the answer indicates the type (e.g., "grocery", "restaurant", "fast food"), capture it; else null.
      - location_notes: Any Massachusetts-specific note if present, else null.

    Map them to:
      - grocery_close_early: A grocery store chain open on Thanksgiving Day 2025, but closing by 1:00 PM or earlier.
      - grocery_open_late: A grocery store chain open on Thanksgiving Day 2025 until 5:00 PM or later.
      - restaurant_24h: A restaurant chain operating 24 hours continuously on Thanksgiving Day 2025.
      - fast_food_daytime: A fast-food restaurant chain that is open on Thanksgiving Day 2025 with daytime hours.

    Rules:
    - Only extract URLs that are explicitly present in the provided answer (plain or markdown links).
    - If an item is not provided in the answer, set it to null.
    - If sources are not present for an item, set sources to an empty array.
    - Prefer Massachusetts-specific sources if multiple are provided; still include all cited URLs.
    - The date of interest is {THANKSGIVING_2025_DATE_STR}.
    """


# --------------------------------------------------------------------------- #
# Verification helpers                                                        #
# --------------------------------------------------------------------------- #
def _has_minimal_evidence(est: Optional[Establishment]) -> bool:
    return bool(est and est.chain_name and est.chain_name.strip() and est.sources and len(est.sources) > 0)


# --------------------------------------------------------------------------- #
# Verification functions for each category                                    #
# --------------------------------------------------------------------------- #
async def verify_grocery_close_early(evaluator: Evaluator, parent_node, est: Optional[Establishment]) -> None:
    node = evaluator.add_sequential(
        id="grocery_store_1",
        desc="Identify a grocery store chain in Massachusetts that is open on Thanksgiving Day 2025 and closes by 1:00 PM or earlier.",
        parent=parent_node,
        critical=False
    )

    exists = evaluator.add_custom_node(
        result=_has_minimal_evidence(est),
        id="g1_exists",
        desc="Grocery #1 has chain name and source URLs cited in the answer",
        parent=node,
        critical=True
    )

    # Category check: grocery store chain
    cat_leaf = evaluator.add_leaf(
        id="g1_is_grocery_chain",
        desc="Chain is a grocery store chain",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim=f"{(est.chain_name if est else '')} is a grocery store chain (e.g., supermarket) in the United States.",
        node=cat_leaf,
        sources=(est.sources if est else []),
        additional_instruction="Consider synonyms like 'supermarket' or 'grocery market' as grocery store chains. "
                               "Rely on the provided URLs only."
    )

    # Operates in Massachusetts
    ma_leaf = evaluator.add_leaf(
        id="g1_operates_in_ma",
        desc="Chain operates in Massachusetts (has MA locations)",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim=f"{(est.chain_name if est else '')} has locations or operates in Massachusetts.",
        node=ma_leaf,
        sources=(est.sources if est else []),
        additional_instruction="Prefer evidence such as a store locator page showing Massachusetts locations, "
                               "or an MA-specific announcement."
    )

    # Open on Thanksgiving Day 2025 in MA
    open_leaf = evaluator.add_leaf(
        id="g1_open_thx_2025",
        desc="Open on Thanksgiving Day 2025 in Massachusetts",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim=f"{(est.chain_name if est else '')} is open on Thanksgiving Day 2025 ({THANKSGIVING_2025_DATE_STR}) in Massachusetts.",
        node=open_leaf,
        sources=(est.sources if est else []),
        additional_instruction="The evidence must clearly refer to Thanksgiving for year 2025, or a 2025 holiday-hours "
                               "page/policy that applies to Massachusetts. If only a different year is shown, treat as not supported."
    )

    # Closes by 1:00 PM or earlier on that day
    close_leaf = evaluator.add_leaf(
        id="g1_closes_by_1pm",
        desc="Closes by 1:00 PM or earlier on Thanksgiving 2025 in MA",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim=f"On {THANKSGIVING_2025_DATE_STR}, {(est.chain_name if est else '')} locations in Massachusetts close by 1:00 PM or earlier.",
        node=close_leaf,
        sources=(est.sources if est else []),
        additional_instruction="Look for explicit Thanksgiving 2025 hours stating close at or before 1:00 PM local time "
                               "(e.g., 12 PM, 1 PM). Chain-wide announcements applying to MA are acceptable."
    )


async def verify_grocery_open_late(evaluator: Evaluator, parent_node, est: Optional[Establishment]) -> None:
    node = evaluator.add_sequential(
        id="grocery_store_2",
        desc="Identify a grocery store chain in Massachusetts that remains open on Thanksgiving Day 2025 until 5:00 PM or later.",
        parent=parent_node,
        critical=False
    )

    exists = evaluator.add_custom_node(
        result=_has_minimal_evidence(est),
        id="g2_exists",
        desc="Grocery #2 has chain name and source URLs cited in the answer",
        parent=node,
        critical=True
    )

    # Category check: grocery store chain
    cat_leaf = evaluator.add_leaf(
        id="g2_is_grocery_chain",
        desc="Chain is a grocery store chain",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim=f"{(est.chain_name if est else '')} is a grocery store chain (e.g., supermarket) in the United States.",
        node=cat_leaf,
        sources=(est.sources if est else []),
        additional_instruction="Consider synonyms like 'supermarket' or 'grocery market' as grocery store chains. "
    )

    # Operates in Massachusetts
    ma_leaf = evaluator.add_leaf(
        id="g2_operates_in_ma",
        desc="Chain operates in Massachusetts (has MA locations)",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim=f"{(est.chain_name if est else '')} has locations or operates in Massachusetts.",
        node=ma_leaf,
        sources=(est.sources if est else []),
        additional_instruction="Evidence may include a store locator page listing Massachusetts."
    )

    # Open on Thanksgiving Day 2025 in MA
    open_leaf = evaluator.add_leaf(
        id="g2_open_thx_2025",
        desc="Open on Thanksgiving Day 2025 in Massachusetts",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim=f"{(est.chain_name if est else '')} is open on Thanksgiving Day 2025 ({THANKSGIVING_2025_DATE_STR}) in Massachusetts.",
        node=open_leaf,
        sources=(est.sources if est else []),
        additional_instruction="The page should refer to 2025 Thanksgiving hours or a 2025 holiday schedule applying to MA."
    )

    # Open until 5:00 PM or later on that day
    late_leaf = evaluator.add_leaf(
        id="g2_open_until_5pm_or_later",
        desc="Remains open until 5:00 PM or later on Thanksgiving 2025 in MA",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim=f"On {THANKSGIVING_2025_DATE_STR}, {(est.chain_name if est else '')} locations in Massachusetts remain open until at least 5:00 PM local time.",
        node=late_leaf,
        sources=(est.sources if est else []),
        additional_instruction="Look for explicit hours indicating closing time at 5 PM or later (e.g., 6 PM, 7 PM)."
    )


async def verify_restaurant_24h(evaluator: Evaluator, parent_node, est: Optional[Establishment]) -> None:
    node = evaluator.add_sequential(
        id="restaurant_1",
        desc="Identify a restaurant chain in Massachusetts that operates 24 hours continuously on Thanksgiving Day 2025.",
        parent=parent_node,
        critical=False
    )

    exists = evaluator.add_custom_node(
        result=_has_minimal_evidence(est),
        id="r1_exists",
        desc="Restaurant #1 has chain name and source URLs cited in the answer",
        parent=node,
        critical=True
    )

    # Category check: restaurant chain
    cat_leaf = evaluator.add_leaf(
        id="r1_is_restaurant_chain",
        desc="Chain is a restaurant chain",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim=f"{(est.chain_name if est else '')} is a restaurant chain in the United States.",
        node=cat_leaf,
        sources=(est.sources if est else []),
        additional_instruction="It does not need to be fast-food; any sit-down or diner-style chain qualifies as a restaurant chain."
    )

    # Operates in Massachusetts
    ma_leaf = evaluator.add_leaf(
        id="r1_operates_in_ma",
        desc="Chain operates in Massachusetts (has MA locations)",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim=f"{(est.chain_name if est else '')} has locations or operates in Massachusetts.",
        node=ma_leaf,
        sources=(est.sources if est else []),
        additional_instruction="Store locator or MA-specific pages are acceptable."
    )

    # 24-hour continuous operation on Thanksgiving Day 2025 in MA
    thx_24h_leaf = evaluator.add_leaf(
        id="r1_24h_thx_2025",
        desc="Operates 24 hours continuously on Thanksgiving Day 2025 in Massachusetts",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim=f"On {THANKSGIVING_2025_DATE_STR}, {(est.chain_name if est else '')} operates 24 hours (open all day and night) in Massachusetts.",
        node=thx_24h_leaf,
        sources=(est.sources if est else []),
        additional_instruction="Accept evidence that indicates 24-hour service specifically on Thanksgiving Day 2025 and "
                               "that applies to at least some Massachusetts locations."
    )


async def verify_fast_food_daytime(evaluator: Evaluator, parent_node, est: Optional[Establishment]) -> None:
    node = evaluator.add_sequential(
        id="restaurant_2",
        desc="Identify a fast-food restaurant chain in Massachusetts that is open on Thanksgiving Day 2025 with operating hours during the daytime.",
        parent=parent_node,
        critical=False
    )

    exists = evaluator.add_custom_node(
        result=_has_minimal_evidence(est),
        id="r2_exists",
        desc="Restaurant #2 has chain name and source URLs cited in the answer",
        parent=node,
        critical=True
    )

    # Category check: fast-food restaurant chain
    cat_leaf = evaluator.add_leaf(
        id="r2_is_fast_food_chain",
        desc="Chain is a fast-food restaurant chain",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim=f"{(est.chain_name if est else '')} is a fast-food restaurant chain.",
        node=cat_leaf,
        sources=(est.sources if est else []),
        additional_instruction="Fast-food typically includes quick-service brands with counter service; rely only on the provided URLs."
    )

    # Operates in Massachusetts
    ma_leaf = evaluator.add_leaf(
        id="r2_operates_in_ma",
        desc="Chain operates in Massachusetts (has MA locations)",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim=f"{(est.chain_name if est else '')} has locations or operates in Massachusetts.",
        node=ma_leaf,
        sources=(est.sources if est else []),
        additional_instruction="Store locator or MA-specific evidence is acceptable."
    )

    # Open during daytime (e.g., at noon) on Thanksgiving Day 2025 in MA
    daytime_leaf = evaluator.add_leaf(
        id="r2_open_daytime_thx_2025",
        desc="Open during daytime on Thanksgiving Day 2025 in Massachusetts",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim=f"On {THANKSGIVING_2025_DATE_STR}, {(est.chain_name if est else '')} is open during daytime hours in Massachusetts (e.g., open at 12:00 PM local time).",
        node=daytime_leaf,
        sources=(est.sources if est else []),
        additional_instruction="Look for Thanksgiving 2025 hours indicating lunchtime or midday opening (e.g., 10 AM–4 PM, "
                               "open at noon). Chain-wide announcements applying to MA are acceptable."
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
    Evaluate an answer for the Massachusetts Thanksgiving 2025 operating-hours task.
    """
    # 1) Initialize evaluator and root node
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Top-level criteria are independent
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

    # 2) Extract structured info from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_establishments(),
        template_class=ThanksgivingEstablishmentsExtraction,
        extraction_name="thanksgiving_establishments"
    )

    # 3) Add a brief ground-truth-style target criteria for transparency
    evaluator.add_ground_truth({
        "date": THANKSGIVING_2025_DATE_STR,
        "requirements": [
            "Grocery #1: Open on Thanksgiving 2025; closes by 1:00 PM or earlier (MA).",
            "Grocery #2: Open on Thanksgiving 2025; open until 5:00 PM or later (MA).",
            "Restaurant #1: Operates 24 hours on Thanksgiving 2025 (MA).",
            "Restaurant #2: Fast-food; open with daytime hours on Thanksgiving 2025 (MA)."
        ]
    })

    # 4) Build verification subtrees corresponding to rubric items
    # Grocery store 1
    await verify_grocery_close_early(evaluator, root, extracted.grocery_close_early)

    # Grocery store 2
    await verify_grocery_open_late(evaluator, root, extracted.grocery_open_late)

    # Restaurant 1 (24 hours)
    await verify_restaurant_24h(evaluator, root, extracted.restaurant_24h)

    # Restaurant 2 (fast-food daytime)
    await verify_fast_food_daytime(evaluator, root, extracted.fast_food_daytime)

    # 5) Return structured result
    return evaluator.get_summary()