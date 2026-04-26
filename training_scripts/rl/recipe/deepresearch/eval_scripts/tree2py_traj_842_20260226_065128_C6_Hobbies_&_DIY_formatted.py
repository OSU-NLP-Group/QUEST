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
TASK_ID = "thanksgiving_2025_diy_decor_plan"
TASK_DESCRIPTION = """
You are planning to host a Thanksgiving dinner in 2025 and want to create handmade table decorations for 8 guests. You prefer to shop at major craft stores like Hobby Lobby or Michaels for your materials.

Your decoration plan must include three DIY projects:
1. A burlap table runner
2. Decorative napkin rings (one for each guest)
3. A table centerpiece

For your planning, you need to:
- Determine the exact date of Thanksgiving 2025
- Identify the last day you can shop at craft stores before Thanksgiving (considering store holiday closures)
- Specify the complete list of materials needed for all three projects, including:
  - The quantity of burlap fabric needed for the table runner
  - The number of napkin rings required
  - All materials and components for each project
- Confirm that all specified materials are available at Hobby Lobby or Michaels
- Provide an estimated total budget for all materials

Provide a comprehensive DIY decoration plan that includes the shopping timeline, complete materials list with quantities for each project, store availability confirmation, and budget estimation with supporting reference URLs.
"""

EXPECTED_THANKSGIVING_2025_DATE = "Thursday, November 27, 2025"
EXPECTED_NAPKIN_RING_COUNT = 8
EXPECTED_LAST_SHOPPING_DAY_NOTE = "on or before November 26, 2025"
BURLAP_PRICE_MIN = 4.19
BURLAP_PRICE_MAX = 11.99

# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class MaterialItem(BaseModel):
    name: Optional[str] = None
    quantity: Optional[str] = None  # keep as string to allow "about 2 yards", "8 pcs", etc.


class PlanExtraction(BaseModel):
    # Timeline
    thanksgiving_date: Optional[str] = None
    last_shopping_day: Optional[str] = None
    store_closed_hobby_lobby_on_thanksgiving: Optional[bool] = None
    store_closed_michaels_on_thanksgiving: Optional[bool] = None

    # Projects included
    burlap_runner_included: Optional[bool] = None
    napkin_rings_included: Optional[bool] = None
    centerpiece_included: Optional[bool] = None

    # Materials by project
    runner_burlap_yardage_str: Optional[str] = None
    runner_burlap_yards_number: Optional[float] = None
    runner_materials: List[MaterialItem] = Field(default_factory=list)

    napkin_rings_count: Optional[int] = None
    napkin_rings_materials: List[MaterialItem] = Field(default_factory=list)

    centerpiece_materials: List[MaterialItem] = Field(default_factory=list)

    # Store availability confirmation
    availability_statement_present: Optional[bool] = None

    # Budget and pricing
    total_budget_amount: Optional[float] = None  # numeric total; if a range provided, midpoint
    total_budget_currency: Optional[str] = None  # e.g., "USD", "$"
    burlap_price_per_yard: Optional[float] = None  # numeric per-yard price if explicitly stated/used

    # Support URLs
    reference_urls: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_plan() -> str:
    return """
Extract the following structured information from the answer exactly as written (do not invent or infer facts not present in the answer). If an item is not present in the answer, return null for scalars or [] for lists.

TIMELINE
- thanksgiving_date: The stated date for Thanksgiving 2025 (e.g., "Thursday, November 27, 2025"). If the answer gives an equivalent phrasing (like "Nov 27, 2025" or "Thursday 11/27/2025"), extract that exact string.
- last_shopping_day: The last day to shop at craft stores before Thanksgiving according to the plan (e.g., "Wednesday, November 26, 2025" or "Nov 26, 2025").
- store_closed_hobby_lobby_on_thanksgiving: true/false if the answer explicitly states Hobby Lobby is closed on Thanksgiving; null if not specified.
- store_closed_michaels_on_thanksgiving: true/false if the answer explicitly states Michaels is closed on Thanksgiving; null if not specified.

PROJECTS INCLUDED (booleans)
- burlap_runner_included: true/false if the answer includes a DIY burlap table runner project.
- napkin_rings_included: true/false if the answer includes a DIY decorative napkin rings project.
- centerpiece_included: true/false if the answer includes at least one DIY table centerpiece project.

MATERIALS (by project)
For each project, extract a materials list, capturing each item and any quantity text exactly as stated.

TABLE RUNNER
- runner_burlap_yardage_str: The exact text describing the burlap amount for the runner (e.g., "about 2 yards", "2 yards", "1.8–2.2 yards").
- runner_burlap_yards_number: A numeric value only if an explicit numeric yard amount is given; choose the single most salient numeric (if a range, choose the midpoint); otherwise null.
- runner_materials: array of {name, quantity} from the answer for the table runner.

NAPKIN RINGS
- napkin_rings_count: Numeric count of napkin rings to make/buy (e.g., 8). If the answer states "one per guest" and the plan is for 8 guests, set to 8. If unclear, set null.
- napkin_rings_materials: array of {name, quantity} for the napkin rings.

CENTERPIECE
- centerpiece_materials: array of {name, quantity} for at least one centerpiece described.

STORE AVAILABILITY
- availability_statement_present: true/false if the answer explicitly confirms that the specified materials can be purchased at Hobby Lobby or Michaels (e.g., "All materials are available at Hobby Lobby or Michaels").

BUDGET AND PRICING
- total_budget_amount: A single numeric total budget covering all required projects. If a range is provided (e.g., $50–$70), return the midpoint as a float. If not stated, null.
- total_budget_currency: the currency symbol or code if present (e.g., "$", "USD"); else null.
- burlap_price_per_yard: A numeric per-yard price for burlap only if explicitly stated/used (e.g., 6.99). If multiple prices are given, pick the one tied to per-yard burlap pricing. Otherwise null.

SUPPORTING URLS
- reference_urls: Extract all URLs present in the answer (including store pages, product pages, calendar pages, or any supporting references). Return them as an array of full URLs. If none, return [].
"""


# --------------------------------------------------------------------------- #
# Helper functions for building verification tree                             #
# --------------------------------------------------------------------------- #
async def add_timeline_checks(evaluator: Evaluator, parent_node, plan: PlanExtraction) -> None:
    # Parent node for timeline (critical, parallel)
    timeline_node = evaluator.add_parallel(
        id="Timeline_and_Shopping_Deadline",
        desc="Verify Thanksgiving date and last feasible shopping day given store closures.",
        parent=parent_node,
        critical=True
    )

    urls = plan.reference_urls if plan.reference_urls else None

    # Thanksgiving date verification
    thx_date_node = evaluator.add_leaf(
        id="Thanksgiving_Date_2025",
        desc="States Thanksgiving 2025 is Thursday, November 27, 2025.",
        parent=timeline_node,
        critical=True
    )
    thx_claim = "In the United States, Thanksgiving in 2025 falls on Thursday, November 27, 2025."
    await evaluator.verify(
        claim=thx_claim,
        node=thx_date_node,
        sources=urls,
        additional_instruction="Prefer URLs that explicitly list 2025 dates or confirm that Thanksgiving is the 4th Thursday of November and that Nov 27, 2025 is the 4th Thursday. Ignore non‑US Thanksgiving."
    )

    # Store closure verification
    closure_node = evaluator.add_leaf(
        id="Store_Closure_on_Thanksgiving",
        desc="States Hobby Lobby and Michaels are closed on Thanksgiving Day.",
        parent=timeline_node,
        critical=True
    )
    closure_claim = "Hobby Lobby and Michaels are closed on Thanksgiving Day."
    await evaluator.verify(
        claim=closure_claim,
        node=closure_node,
        sources=urls,
        additional_instruction="Prefer official store pages or trustworthy sources indicating Thanksgiving Day closure. If both stores are shown as closed on Thanksgiving, mark as supported."
    )

    # Last shopping day verification (logical, simple check)
    last_shop_node = evaluator.add_leaf(
        id="Last_Shopping_Day",
        desc="Identifies the last day to shop as on or before November 26, 2025 (the day before Thanksgiving).",
        parent=timeline_node,
        critical=True
    )
    last_shop_claim = "The last day to shop at these craft stores before Thanksgiving 2025 is on or before Wednesday, November 26, 2025 (the day before Thanksgiving)."
    await evaluator.verify(
        claim=last_shop_claim,
        node=last_shop_node,
        additional_instruction="If the plan confirms both stores are closed on Thanksgiving Day and Thanksgiving 2025 is Nov 27, then the last feasible pre‑Thanksgiving shopping day is Nov 26, 2025."
    )


async def add_project_inclusion_checks(evaluator: Evaluator, parent_node, plan: PlanExtraction) -> None:
    projects_node = evaluator.add_parallel(
        id="DIY_Projects_Included",
        desc="Verify the plan includes the three required DIY projects.",
        parent=parent_node,
        critical=True
    )

    # Burlap runner included
    runner_included = evaluator.add_leaf(
        id="Burlap_Table_Runner_Included",
        desc="Includes a DIY burlap table runner project.",
        parent=projects_node,
        critical=True
    )
    await evaluator.verify(
        claim="The plan includes a DIY burlap table runner project.",
        node=runner_included
    )

    # Napkin rings included
    rings_included = evaluator.add_leaf(
        id="Napkin_Rings_Included",
        desc="Includes a DIY napkin rings project.",
        parent=projects_node,
        critical=True
    )
    await evaluator.verify(
        claim="The plan includes a DIY decorative napkin rings project.",
        node=rings_included
    )

    # Centerpiece included
    centerpiece_included = evaluator.add_leaf(
        id="Centerpiece_Included",
        desc="Includes at least one table centerpiece project.",
        parent=projects_node,
        critical=True
    )
    await evaluator.verify(
        claim="The plan includes at least one DIY table centerpiece project.",
        node=centerpiece_included
    )


async def add_materials_checks(evaluator: Evaluator, parent_node, plan: PlanExtraction) -> None:
    mats_node = evaluator.add_parallel(
        id="Materials_and_Quantities_By_Project",
        desc="Verify each required project has a materials list with quantities, including required quantities from constraints.",
        parent=parent_node,
        critical=True
    )

    # Table runner: materials + approximately 2 yards burlap
    runner_mats_node = evaluator.add_leaf(
        id="Table_Runner_Materials_and_Quantity",
        desc="Provides a materials list for the burlap table runner and specifies approximately 2 yards of burlap fabric.",
        parent=mats_node,
        critical=True
    )
    runner_claim = "The plan provides a materials list for the burlap table runner and specifies approximately 2 yards of burlap fabric (e.g., 'about 2 yards', 'around two yards', or a small range near 2 yards)."
    await evaluator.verify(
        claim=runner_claim,
        node=runner_mats_node,
        additional_instruction="Accept 'about/approx./~ 2 yards' or ranges roughly between 1.8–2.2 yards as 'approximately 2 yards'."
    )

    # Napkin rings: materials + exactly 8 rings total
    rings_mats_node = evaluator.add_leaf(
        id="Napkin_Rings_Materials_and_Quantity",
        desc="Provides a materials list for napkin rings and specifies exactly 8 napkin rings total (one for each of 8 guests).",
        parent=mats_node,
        critical=True
    )
    rings_claim = "The plan provides a materials list for the napkin rings and specifies exactly 8 napkin rings total (one per guest for 8 guests)."
    await evaluator.verify(
        claim=rings_claim,
        node=rings_mats_node,
        additional_instruction="Expressions like 'one per guest' with 8 guests count as exactly 8."
    )

    # Centerpiece: materials list (quantities where applicable)
    centerpiece_mats_node = evaluator.add_leaf(
        id="Centerpiece_Materials",
        desc="Provides a materials list (with quantities where applicable) for at least one centerpiece.",
        parent=mats_node,
        critical=True
    )
    centerpiece_claim = "The plan provides a materials list for at least one centerpiece, including quantities where applicable (not every item must have a quantity)."
    await evaluator.verify(
        claim=centerpiece_claim,
        node=centerpiece_mats_node,
        additional_instruction="Look for a clear list of items needed to build the centerpiece. Some items may be listed without quantities; that's acceptable."
    )


async def add_store_availability_checks(evaluator: Evaluator, parent_node, plan: PlanExtraction) -> None:
    avail_node = evaluator.add_parallel(
        id="Store_Availability_Confirmation",
        desc="Verify the response confirms that all listed materials are available at Hobby Lobby or Michaels (as required).",
        parent=parent_node,
        critical=True
    )

    availability_leaf = evaluator.add_leaf(
        id="Availability_Statement",
        desc="Explicitly confirms that the specified materials can be purchased at either Hobby Lobby or Michaels.",
        parent=avail_node,
        critical=True
    )
    await evaluator.verify(
        claim="The plan explicitly confirms that the specified materials can be purchased at either Hobby Lobby or Michaels.",
        node=availability_leaf,
        additional_instruction="Look for explicit language like 'All materials are available at Hobby Lobby or Michaels' or equivalent wording. Product/store URLs to those sites further support the statement."
    )


async def add_budget_checks(evaluator: Evaluator, parent_node, plan: PlanExtraction) -> None:
    budget_node = evaluator.add_parallel(
        id="Budget_Estimate",
        desc="Verify a total budget estimate is provided for all materials across the required projects.",
        parent=parent_node,
        critical=True
    )

    # Total budget provided (numeric)
    total_budget_provided = (
        plan.total_budget_amount is not None and isinstance(plan.total_budget_amount, (int, float)) and plan.total_budget_amount > 0
    )
    evaluator.add_custom_node(
        result=total_budget_provided,
        id="Total_Budget_Provided",
        desc="Provides a numeric estimated total budget covering materials for all three required projects.",
        parent=budget_node,
        critical=True
    )

    # Burlap per-yard price range consistency (pass if not stated)
    burlap_ok = (
        plan.burlap_price_per_yard is None or
        (isinstance(plan.burlap_price_per_yard, (int, float)) and BURLAP_PRICE_MIN <= float(plan.burlap_price_per_yard) <= BURLAP_PRICE_MAX)
    )
    evaluator.add_custom_node(
        result=burlap_ok,
        id="Burlap_Price_Range_Consistency",
        desc=f"If a burlap per-yard price is stated/used, it falls within ${BURLAP_PRICE_MIN:.2f}–${BURLAP_PRICE_MAX:.2f} per yard.",
        parent=budget_node,
        critical=True
    )


async def add_supporting_urls_check(evaluator: Evaluator, parent_node, plan: PlanExtraction) -> None:
    # Single critical leaf under root per rubric
    has_urls = bool(plan.reference_urls and len(plan.reference_urls) > 0)
    evaluator.add_custom_node(
        result=has_urls,
        id="Supporting_Reference_URLs",
        desc="Provides supporting reference URL(s) relevant to key factual claims in the plan (e.g., Thanksgiving date, store closures, and/or material availability/pricing).",
        parent=parent_node,
        critical=True
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
    Evaluate an answer for the Thanksgiving 2025 DIY table decoration plan task.
    """
    # Initialize evaluator (root is a wrapper; we add a critical child as the true root per rubric)
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

    # Extract plan details
    plan: PlanExtraction = await evaluator.extract(
        prompt=prompt_extract_plan(),
        template_class=PlanExtraction,
        extraction_name="plan_extraction",
    )

    # Add ground-truth references for context (not enforced, just recorded)
    evaluator.add_ground_truth({
        "expected_thanksgiving_2025_date": EXPECTED_THANKSGIVING_2025_DATE,
        "expected_last_shopping_day": EXPECTED_LAST_SHOPPING_DAY_NOTE,
        "expected_napkin_ring_count": EXPECTED_NAPKIN_RING_COUNT,
        "burlap_price_range_usd_per_yard": [BURLAP_PRICE_MIN, BURLAP_PRICE_MAX]
    }, gt_type="expected_constraints")

    # Top-level critical plan node
    plan_node = evaluator.add_parallel(
        id="Complete_Thanksgiving_DIY_Decoration_Plan",
        desc="Evaluate whether the response provides the required Thanksgiving 2025 DIY table decoration plan for 8 guests, including timeline, the three required projects, materials with quantities, store availability, budget, and supporting reference URLs.",
        parent=root,
        critical=True
    )

    # Build verification subtrees
    await add_timeline_checks(evaluator, plan_node, plan)
    await add_project_inclusion_checks(evaluator, plan_node, plan)
    await add_materials_checks(evaluator, plan_node, plan)
    await add_store_availability_checks(evaluator, plan_node, plan)
    await add_budget_checks(evaluator, plan_node, plan)
    await add_supporting_urls_check(evaluator, plan_node, plan)

    # Optional: record some custom info for debugging/reporting
    evaluator.add_custom_info(
        info={
            "extracted_thanksgiving_date": plan.thanksgiving_date,
            "extracted_last_shopping_day": plan.last_shopping_day,
            "burlap_runner_included": plan.burlap_runner_included,
            "napkin_rings_included": plan.napkin_rings_included,
            "centerpiece_included": plan.centerpiece_included,
            "runner_burlap_yardage_str": plan.runner_burlap_yardage_str,
            "napkin_rings_count": plan.napkin_rings_count,
            "availability_statement_present": plan.availability_statement_present,
            "total_budget_amount": plan.total_budget_amount,
            "burlap_price_per_yard": plan.burlap_price_per_yard,
            "reference_urls_count": len(plan.reference_urls) if plan.reference_urls else 0
        },
        info_type="extraction_summary"
    )

    # Return structured evaluation summary
    return evaluator.get_summary()