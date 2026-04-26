import asyncio
import logging
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy

# -----------------------------------------------------------------------------
# Task constants
# -----------------------------------------------------------------------------
TASK_ID = "holiday_hours_2025"
TASK_DESCRIPTION = """
You are helping someone plan their dining and shopping options during the 2025 holiday season. Identify four different national chain establishments that meet the following criteria:

1. A fast-food or quick-service restaurant chain that is open on Thanksgiving Day (November 27, 2025) and closes at or after 3 PM local time on that day. Provide the chain name and a reference URL confirming its Thanksgiving hours.

2. A coffee shop or bakery chain that is open on Thanksgiving Day but closes at or before noon local time. Provide the chain name and a reference URL confirming its Thanksgiving hours.

3. A major retail store chain that closes at 8 PM local time on Christmas Eve (December 24, 2025). Provide the chain name and a reference URL confirming its Christmas Eve closing time.

4. A restaurant chain that operates 24 hours on Christmas Day (December 25, 2025). Provide the chain name and a reference URL confirming its 24-hour Christmas Day operation.

For each establishment, provide:
- The name of the national chain
- A direct reference URL from the chain's official website, a major news outlet, or a reputable source that confirms the specific holiday hours mentioned
""".strip()

THANKSGIVING_2025 = "November 27, 2025"
CHRISTMAS_EVE_2025 = "December 24, 2025"
CHRISTMAS_DAY_2025 = "December 25, 2025"


# -----------------------------------------------------------------------------
# Extraction models
# -----------------------------------------------------------------------------
class Establishment(BaseModel):
    chain_name: Optional[str] = None
    category: Optional[str] = None
    reference_urls: List[str] = Field(default_factory=list)
    hours_note: Optional[str] = None


class HolidayPlanExtraction(BaseModel):
    establishment_1: Optional[Establishment] = None
    establishment_2: Optional[Establishment] = None
    establishment_3: Optional[Establishment] = None
    establishment_4: Optional[Establishment] = None


# -----------------------------------------------------------------------------
# Extraction prompt
# -----------------------------------------------------------------------------
def prompt_extract_establishments() -> str:
    return f"""
Extract four establishments from the answer, one per required scenario. For each establishment, return:
- chain_name: the specific national chain name
- category: a short label indicating the type (e.g., "fast_food_quick_service", "coffee_or_bakery", "major_retailer", "restaurant")
- reference_urls: a list of the explicit URLs cited in the answer that support the stated holiday hours (only include actual URLs present in the answer text; do not invent)
- hours_note: the snippet or sentence from the answer describing the holiday hours for the relevant date

Map them as follows (pick the first matching example for each if multiple are given):
- establishment_1: fast-food or quick-service restaurant chain that is open on Thanksgiving Day ({THANKSGIVING_2025}) AND closes at or after 3 PM local time that day.
- establishment_2: coffee shop or bakery chain that is open on Thanksgiving Day ({THANKSGIVING_2025}) but closes at or before noon local time that day.
- establishment_3: major retail store chain that closes at 8 PM local time on Christmas Eve ({CHRISTMAS_EVE_2025}).
- establishment_4: restaurant chain that operates 24 hours on Christmas Day ({CHRISTMAS_DAY_2025}).

If any field is missing, set it to null (or [] for reference_urls).
Only extract URLs explicitly present in the answer text (including markdown links). Do not infer or fabricate URLs.
""".strip()


# -----------------------------------------------------------------------------
# Helper utilities
# -----------------------------------------------------------------------------
def _safe_item(item: Optional[Establishment]) -> Establishment:
    return item or Establishment()


def _has_valid_url(urls: List[str]) -> bool:
    if not urls:
        return False
    for u in urls:
        if isinstance(u, str) and u.strip().lower().startswith(("http://", "https://")):
            return True
    return False


# -----------------------------------------------------------------------------
# Verification subroutines per establishment
# -----------------------------------------------------------------------------
async def verify_establishment_1(evaluator: Evaluator, parent_node, item: Establishment) -> None:
    """
    Establishment 1:
    - Fast-food or quick-service restaurant chain
    - Open on Thanksgiving Day (Nov 27, 2025)
    - Closes at or after 3 PM local time that day
    - National chain identification
    - Valid reference URL(s)
    """
    node = evaluator.add_parallel(
        id="establishment_1",
        desc="Identify a fast-food or quick-service restaurant chain that is open on Thanksgiving Day 2025 and closes at or after 3 PM local time on that day",
        parent=parent_node,
        critical=False,
    )

    name = item.chain_name or ""

    # est1_type (critical)
    leaf_type = evaluator.add_leaf(
        id="est1_type",
        desc="The establishment is a fast-food or quick-service restaurant chain",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"The establishment '{name}' is a fast-food or quick-service restaurant chain.",
        node=leaf_type,
        additional_instruction="Judge based on common usage of 'fast-food' or 'quick-service' for national restaurant chains. Minor variations in classification are acceptable if it is clearly a quick-service chain."
    )

    # est1_open_thanksgiving (critical, URL-grounded)
    leaf_open_tg = evaluator.add_leaf(
        id="est1_open_thanksgiving",
        desc=f"The restaurant chain is confirmed to be open on Thanksgiving Day ({THANKSGIVING_2025})",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"The chain '{name}' is open on Thanksgiving Day ({THANKSGIVING_2025}).",
        node=leaf_open_tg,
        sources=item.reference_urls,
        additional_instruction="Look for explicit statements for Thanksgiving Day 2025. If the page only says 'hours may vary' without stating 'open', consider not supported unless 'open' is clearly implied for that date."
    )

    # est1_closing_time (critical, URL-grounded)
    leaf_close_tg = evaluator.add_leaf(
        id="est1_closing_time",
        desc="The restaurant chain closes at or after 3 PM local time on Thanksgiving Day",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"On Thanksgiving Day ({THANKSGIVING_2025}), the chain '{name}' closes at or after 3:00 PM local time.",
        node=leaf_close_tg,
        sources=item.reference_urls,
        additional_instruction="The claim is satisfied if the source indicates a closing time at or after 3:00 PM (e.g., 'closes at 4 PM', 'regular hours until 10 PM'). If only 'hours vary by location' is stated without a clear closing time, consider not supported."
    )

    # est1_chain_identification (critical)
    leaf_chain = evaluator.add_leaf(
        id="est1_chain_identification",
        desc="The restaurant is identified as a specific national chain (not a local establishment)",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"The establishment '{name}' is a national chain (not a local-only business).",
        node=leaf_chain,
        additional_instruction="Consider common knowledge and brand presence; it should be recognized as a chain with multiple locations nationwide."
    )

    # est1_reference (critical) - basic URL presence/format check
    evaluator.add_custom_node(
        result=_has_valid_url(item.reference_urls),
        id="est1_reference",
        desc="A valid reference URL is provided confirming the Thanksgiving hours",
        parent=node,
        critical=True,
    )


async def verify_establishment_2(evaluator: Evaluator, parent_node, item: Establishment) -> None:
    """
    Establishment 2:
    - Coffee shop or bakery chain
    - Open on Thanksgiving Day (Nov 27, 2025)
    - Closes at or before noon local time that day
    - National chain identification
    - Valid reference URL(s)
    """
    node = evaluator.add_parallel(
        id="establishment_2",
        desc="Identify a coffee shop or bakery chain that is open on Thanksgiving Day 2025 but closes at or before noon local time",
        parent=parent_node,
        critical=False,
    )

    name = item.chain_name or ""

    # est2_type (critical)
    leaf_type = evaluator.add_leaf(
        id="est2_type",
        desc="The establishment is a coffee shop or bakery chain",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"The establishment '{name}' is a coffee shop or bakery chain.",
        node=leaf_type,
        additional_instruction="Classify based on common understanding; examples include national coffee chains and bakery chains."
    )

    # est2_open_thanksgiving (critical, URL-grounded)
    leaf_open_tg = evaluator.add_leaf(
        id="est2_open_thanksgiving",
        desc=f"The establishment is confirmed to be open on Thanksgiving Day ({THANKSGIVING_2025})",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"The chain '{name}' is open on Thanksgiving Day ({THANKSGIVING_2025}).",
        node=leaf_open_tg,
        sources=item.reference_urls,
        additional_instruction="Look for explicit statements for Thanksgiving 2025 that indicate it is open, even if for limited morning hours."
    )

    # est2_closing_time (critical, URL-grounded)
    leaf_close_tg = evaluator.add_leaf(
        id="est2_closing_time",
        desc="The establishment closes at or before noon local time on Thanksgiving Day",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"On Thanksgiving Day ({THANKSGIVING_2025}), the chain '{name}' closes at or before 12:00 PM (noon) local time.",
        node=leaf_close_tg,
        sources=item.reference_urls,
        additional_instruction="The claim is satisfied if the source indicates a closing time of 12:00 PM or earlier (e.g., 11 AM, noon). If only 'hours vary' is stated without a clear closing time, consider not supported."
    )

    # est2_chain_identification (critical)
    leaf_chain = evaluator.add_leaf(
        id="est2_chain_identification",
        desc="The establishment is identified as a specific national chain",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"The establishment '{name}' is a national chain (not local-only).",
        node=leaf_chain,
        additional_instruction="Use common knowledge of national chains; the brand should have broad U.S. presence."
    )

    # est2_reference (critical)
    evaluator.add_custom_node(
        result=_has_valid_url(item.reference_urls),
        id="est2_reference",
        desc="A valid reference URL is provided confirming the Thanksgiving hours",
        parent=node,
        critical=True,
    )


async def verify_establishment_3(evaluator: Evaluator, parent_node, item: Establishment) -> None:
    """
    Establishment 3:
    - Major retail store chain
    - Closes at 8 PM local time on Christmas Eve (Dec 24, 2025)
    - National chain identification (major retailer)
    - Valid reference URL(s)
    """
    node = evaluator.add_parallel(
        id="establishment_3",
        desc="Identify a major retail store chain that closes at 8 PM local time on Christmas Eve 2025",
        parent=parent_node,
        critical=False,
    )

    name = item.chain_name or ""

    # est3_closes_8pm (critical, URL-grounded)
    leaf_close_ce = evaluator.add_leaf(
        id="est3_closes_8pm",
        desc=f"The store closes at 8 PM local time on Christmas Eve ({CHRISTMAS_EVE_2025})",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"On Christmas Eve ({CHRISTMAS_EVE_2025}), the chain '{name}' closes at 8:00 PM local time.",
        node=leaf_close_ce,
        sources=item.reference_urls,
        additional_instruction="Look for explicit Christmas Eve 2025 closing time of 8:00 PM. Statements like 'stores close at 8 PM' satisfy this."
    )

    # est3_major_retailer (critical)
    leaf_major = evaluator.add_leaf(
        id="est3_major_retailer",
        desc="The store is identified as a major national retail chain",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"'{name}' is a major national retail chain.",
        node=leaf_major,
        additional_instruction="Consider well-known national retailers with large footprints (e.g., Walmart, Target, Best Buy, etc.)."
    )

    # est3_reference (critical)
    evaluator.add_custom_node(
        result=_has_valid_url(item.reference_urls),
        id="est3_reference",
        desc="A valid reference URL is provided confirming the Christmas Eve closing time",
        parent=node,
        critical=True,
    )


async def verify_establishment_4(evaluator: Evaluator, parent_node, item: Establishment) -> None:
    """
    Establishment 4:
    - Restaurant chain
    - Operates 24 hours on Christmas Day (Dec 25, 2025)
    - National chain identification
    - Valid reference URL(s)
    """
    node = evaluator.add_parallel(
        id="establishment_4",
        desc="Identify a restaurant chain that operates 24 hours on Christmas Day 2025",
        parent=parent_node,
        critical=False,
    )

    name = item.chain_name or ""

    # est4_24hour_christmas (critical, URL-grounded)
    leaf_24h = evaluator.add_leaf(
        id="est4_24hour_christmas",
        desc=f"The restaurant operates 24 hours on Christmas Day ({CHRISTMAS_DAY_2025})",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"On Christmas Day ({CHRISTMAS_DAY_2025}), the chain '{name}' is open 24 hours (open all day).",
        node=leaf_24h,
        sources=item.reference_urls,
        additional_instruction="Accept 'open 24 hours', 'open 24/7', or equivalent wording clearly tied to Christmas Day 2025."
    )

    # est4_chain_identification (critical)
    leaf_chain = evaluator.add_leaf(
        id="est4_chain_identification",
        desc="The restaurant is identified as a specific national chain",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"The establishment '{name}' is a national restaurant chain.",
        node=leaf_chain,
        additional_instruction="Use common recognition of national restaurant chains; multiple locations nationwide."
    )

    # est4_reference (critical)
    evaluator.add_custom_node(
        result=_has_valid_url(item.reference_urls),
        id="est4_reference",
        desc="A valid reference URL is provided confirming the 24-hour Christmas Day operation",
        parent=node,
        critical=True,
    )


# -----------------------------------------------------------------------------
# Main evaluation entry point
# -----------------------------------------------------------------------------
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
    Evaluate an answer for the 2025 holiday hours planning task.
    """
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

    # Extract structured information for the four establishments
    extraction = await evaluator.extract(
        prompt=prompt_extract_establishments(),
        template_class=HolidayPlanExtraction,
        extraction_name="holiday_plan_extraction",
    )

    # Record supporting context info
    evaluator.add_custom_info(
        info={
            "thanksgiving_2025": THANKSGIVING_2025,
            "christmas_eve_2025": CHRISTMAS_EVE_2025,
            "christmas_day_2025": CHRISTMAS_DAY_2025,
            "criteria_summary": {
                "establishment_1": "Fast-food/quick-service; open Thanksgiving 2025; closes >= 3 PM",
                "establishment_2": "Coffee/bakery; open Thanksgiving 2025; closes <= 12 PM",
                "establishment_3": "Major retailer; closes 8 PM on Christmas Eve 2025",
                "establishment_4": "Restaurant chain; open 24 hours on Christmas Day 2025",
            },
        },
        info_type="context",
        info_name="holiday_criteria_context",
    )

    # Build verification subtrees for each establishment
    est1 = _safe_item(extraction.establishment_1)
    est2 = _safe_item(extraction.establishment_2)
    est3 = _safe_item(extraction.establishment_3)
    est4 = _safe_item(extraction.establishment_4)

    await verify_establishment_1(evaluator, root, est1)
    await verify_establishment_2(evaluator, root, est2)
    await verify_establishment_3(evaluator, root, est3)
    await verify_establishment_4(evaluator, root, est4)

    return evaluator.get_summary()