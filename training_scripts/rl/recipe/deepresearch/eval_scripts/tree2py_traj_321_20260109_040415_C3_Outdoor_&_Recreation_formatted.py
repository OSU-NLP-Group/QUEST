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
TASK_ID = "jmt_southbound_permit_planning_aug_2026"
TASK_DESCRIPTION = """
You are planning a southbound thru-hike of the John Muir Trail (JMT) from Yosemite National Park to Whitney Portal in August 2026 for a group of 3 people. You intend to complete the entire 211-mile trail from Yosemite Valley to the summit of Mt. Whitney, then descend to Whitney Portal to finish your hike.

For this trip, provide the following information:

1. Permit Trailhead: Which specific Yosemite wilderness permit trailhead must you apply for to be eligible to exit Yosemite via Donohue Pass and continue southbound on the JMT? Provide the exact trailhead name as it appears on Recreation.gov.

2. Total Permit Cost: Calculate the total cost for all required permits for your group of 3 people, assuming you are awarded a permit through the Yosemite lottery. Include:
   - The Yosemite wilderness permit application and per-person fees
   - The Whitney Zone exit fees (since you are exiting at Whitney Portal after entering from Yosemite)

3. First-Night Camping: Your selected permit has specific first-night camping requirements. Describe the minimum distance requirement from Little Yosemite Valley (LYV) for your first night's campsite to maintain Donohue Pass eligibility.

4. Bear Canister: Confirm whether bear-resistant food storage containers are required for your entire JMT route, and specify what items must be stored in the canister.

5. Campfire Regulations: On Day 12 of your hike, you plan to camp at a site located at 10,200 feet elevation in the John Muir Wilderness, south of the Glacier Divide. Are campfires permitted at this elevation? If not, what alternative cooking method is allowed?

Provide reference URLs from your research to support each answer.
"""


# --------------------------------------------------------------------------- #
# Data models for structured extraction                                       #
# --------------------------------------------------------------------------- #
class TrailheadInfo(BaseModel):
    trailhead_name: Optional[str] = None
    urls: List[str] = Field(default_factory=list)


class CostInfo(BaseModel):
    yosemite_application_fee: Optional[str] = None
    yosemite_per_person_fee: Optional[str] = None
    group_size: Optional[int] = None
    yosemite_subtotal: Optional[str] = None
    whitney_fee_applicable: Optional[bool] = None
    whitney_per_person_fee: Optional[str] = None
    whitney_subtotal: Optional[str] = None
    total_cost: Optional[str] = None
    urls: List[str] = Field(default_factory=list)


class FirstNightInfo(BaseModel):
    rule_description: Optional[str] = None
    distance_miles_minimum: Optional[str] = None  # e.g., "2 miles"
    urls: List[str] = Field(default_factory=list)


class BearCanInfo(BaseModel):
    required_entire_route: Optional[bool] = None
    items_to_store: List[str] = Field(default_factory=list)  # e.g., ["food", "toiletries", "scented items"]
    urls: List[str] = Field(default_factory=list)


class CampfireInfo(BaseModel):
    fire_permitted: Optional[bool] = None
    elevation_ft: Optional[str] = None  # e.g., "10,200"
    location_desc: Optional[str] = None  # e.g., "John Muir Wilderness south of Glacier Divide"
    alternative_cooking_method: Optional[str] = None  # e.g., "portable camp stove"
    urls: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_trailhead() -> str:
    return """
    Extract the Yosemite wilderness permit trailhead information for Donohue Pass eligibility, as stated in the answer.

    Return:
    - trailhead_name: The exact trailhead name as it appears on Recreation.gov that is Donohue Pass eligible for a southbound JMT starting from Yosemite Valley (e.g., the Happy Isles to Past Little Yosemite Valley variant).
    - urls: All reference URLs the answer cites that support this trailhead's eligibility/name (Recreation.gov listing and/or Yosemite official info).
    If the answer does not provide URLs, return an empty list for urls.
    """


def prompt_extract_costs() -> str:
    return """
    Extract all permit cost details for a group of 3 people as stated in the answer.

    Return a JSON object with:
    - yosemite_application_fee: The Yosemite wilderness permit application fee (string, include currency if present).
    - yosemite_per_person_fee: The Yosemite per-person fee charged if awarded (string).
    - group_size: The group size stated or implied (integer). If not stated, set to 3.
    - yosemite_subtotal: The stated Yosemite subtotal for the group (string). If not explicitly stated, return null.
    - whitney_fee_applicable: Whether Whitney Zone exit fees apply for hikers exiting at Whitney Portal after entering from Yosemite (boolean).
    - whitney_per_person_fee: The per-person Whitney fee (string). If none or not applicable, set to null.
    - whitney_subtotal: The stated Whitney subtotal for the group (string). If not explicitly stated, return null.
    - total_cost: The stated total permit cost for the group (string).
    - urls: Reference URLs supporting the Yosemite fee structure and Whitney Zone fee applicability/value. If none provided, return an empty list.
    """


def prompt_extract_first_night() -> str:
    return """
    Extract the first-night camping requirement associated with the selected Donohue Pass-eligible permit.

    Return:
    - rule_description: A concise description of the rule (e.g., must camp ≥2 miles beyond Little Yosemite Valley on night one).
    - distance_miles_minimum: The minimum distance requirement expressed as a string (e.g., "2 miles"). If not stated, return null.
    - urls: Reference URL(s) supporting the first-night camping requirement. If none provided, return an empty list.
    """


def prompt_extract_bear_canister() -> str:
    return """
    Extract the bear canister requirement details for the JMT route.

    Return:
    - required_entire_route: Boolean stating whether bear-resistant food storage containers are required along the entire JMT route.
    - items_to_store: A list of item categories required to be stored in the canister (e.g., food, toiletries, other scented items). If the answer lists fewer items, extract what is present.
    - urls: Reference URL(s) supporting the requirement and items to store. If none provided, return an empty list.
    """


def prompt_extract_campfire() -> str:
    return """
    Extract the campfire regulation determination for an elevation of 10,200 ft in the John Muir Wilderness south of the Glacier Divide.

    Return:
    - fire_permitted: Boolean indicating whether campfires are permitted at 10,200 ft south of Glacier Divide.
    - elevation_ft: The elevation stated (string).
    - location_desc: The location description stated (string).
    - alternative_cooking_method: If the answer states campfires are not permitted, extract the allowed alternative cooking method (e.g., portable camp stove). Otherwise, return null.
    - urls: Reference URL(s) supporting the elevation-based campfire rule for the John Muir Wilderness and the specified location. If none provided, return an empty list.
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def _ensure_list(urls: Optional[List[str]]) -> List[str]:
    if not urls:
        return []
    return [u for u in urls if isinstance(u, str) and u.strip()]


def _parse_amount_to_float(value: Optional[str]) -> Optional[float]:
    if value is None:
        return None
    try:
        # Extract first numeric like 10, 10.00 from strings like "$10", "USD 15", "15 dollars"
        m = re.search(r"(\d+(?:\.\d+)?)", value.replace(",", ""))
        if not m:
            return None
        return float(m.group(1))
    except Exception:
        return None


def _equal_money(a: Optional[float], b: Optional[float], tol: float = 0.01) -> bool:
    if a is None or b is None:
        return False
    return abs(a - b) <= tol


def _items_cover_required_categories(items: List[str]) -> bool:
    # Expect coverage of food, toiletries, and other scented items.
    low_items = [s.lower() for s in items]
    has_food = any("food" in s for s in low_items)
    has_toiletries = any(("toiletries" in s) or ("toothpaste" in s) or ("soap" in s) for s in low_items)
    has_scented = any(("scent" in s) or ("scented" in s) or ("odor" in s) for s in low_items)
    return has_food and has_toiletries and has_scented


# --------------------------------------------------------------------------- #
# Verification subtrees                                                       #
# --------------------------------------------------------------------------- #
async def verify_permit_trailhead(evaluator: Evaluator, parent_node, th: TrailheadInfo) -> None:
    node = evaluator.add_parallel(
        id="Permit_Trailhead",
        desc="Identify the correct Yosemite wilderness permit trailhead to be Donohue Pass eligible and provide supporting URL(s).",
        parent=parent_node,
        critical=True,
    )

    # Leaf: Trailhead_Reference_URL (existence)
    evaluator.add_custom_node(
        result=len(_ensure_list(th.urls)) > 0,
        id="Trailhead_Reference_URL",
        desc="Includes at least one reference URL supporting the trailhead’s Donohue Pass eligibility and/or Recreation.gov listing.",
        parent=node,
        critical=True,
    )

    # Leaf: Trailhead_Name_Correct_And_Exact (verify by URLs)
    name_leaf = evaluator.add_leaf(
        id="Trailhead_Name_Correct_And_Exact",
        desc="Trailhead name is the exact Recreation.gov listing and is Donohue Pass eligible from Yosemite Valley.",
        parent=node,
        critical=True,
    )
    trailhead_name = th.trailhead_name or ""
    claim = (
        f"The exact Recreation.gov trailhead name for Donohue Pass eligibility from Yosemite Valley is '{trailhead_name}', "
        f"and this trailhead is explicitly marked Donohue Pass eligible."
    )
    await evaluator.verify(
        claim=claim,
        node=name_leaf,
        sources=_ensure_list(th.urls),
        additional_instruction="Verify both the exact listing name string and Donohue Pass eligibility on the cited page(s). Allow minor hyphenation or casing variations that clearly refer to the same official listing.",
    )


async def verify_total_cost(evaluator: Evaluator, parent_node, ci: CostInfo) -> None:
    node = evaluator.add_parallel(
        id="Total_Permit_Cost",
        desc="Compute total required permit cost for group of 3 including Yosemite fees and Whitney Zone fee applicability, with URLs.",
        parent=parent_node,
        critical=True,
    )

    # Parse values
    app_fee = _parse_amount_to_float(ci.yosemite_application_fee)
    per_person_fee = _parse_amount_to_float(ci.yosemite_per_person_fee)
    yosemite_subtotal_val = _parse_amount_to_float(ci.yosemite_subtotal)
    whitney_pp_fee = _parse_amount_to_float(ci.whitney_per_person_fee)
    whitney_subtotal_val = _parse_amount_to_float(ci.whitney_subtotal)
    total_cost_val = _parse_amount_to_float(ci.total_cost)
    group_size = ci.group_size if isinstance(ci.group_size, int) and ci.group_size > 0 else 3
    urls = _ensure_list(ci.urls)

    # Leaf: Yosemite_Fee_Calculation (custom arithmetic check using provided numbers)
    yosemite_expected = None
    if app_fee is not None and per_person_fee is not None and isinstance(group_size, int):
        yosemite_expected = app_fee + per_person_fee * group_size
    yosemite_calc_ok = yosemite_expected is not None and _equal_money(yosemite_expected, yosemite_subtotal_val)
    evaluator.add_custom_node(
        result=yosemite_calc_ok,
        id="Yosemite_Fee_Calculation",
        desc=f"Correctly computes Yosemite fees as application + per-person*group_size for {group_size} people.",
        parent=node,
        critical=True,
    )

    # Leaf: Whitney_Fee_Applicability (verify by URLs)
    whitney_leaf = evaluator.add_leaf(
        id="Whitney_Fee_Applicability",
        desc="States correctly whether Whitney Zone exit fees are required for hikers starting in Yosemite and exiting via Whitney Portal.",
        parent=node,
        critical=True,
    )
    applies_text = "apply" if ci.whitney_fee_applicable else "do not apply"
    claim_whitney = (
        f"For hikers starting in Yosemite and exiting at Whitney Portal, Whitney Zone exit fees {applies_text}."
    )
    await evaluator.verify(
        claim=claim_whitney,
        node=whitney_leaf,
        sources=urls,
        additional_instruction="Check Inyo NF/Whitney Zone rules for exit via Whitney Portal attached to non-Inyo originating permits (e.g., Yosemite-originating JMT).",
    )

    # Leaf: Total_Cost_Stated (custom arithmetic consistency with components)
    total_expected = None
    if yosemite_expected is not None:
        if ci.whitney_fee_applicable and whitney_pp_fee is not None:
            total_expected = yosemite_expected + whitney_pp_fee * group_size
        elif ci.whitney_fee_applicable and whitney_subtotal_val is not None:
            total_expected = yosemite_expected + whitney_subtotal_val
        else:
            total_expected = yosemite_expected
    total_ok = total_expected is not None and _equal_money(total_expected, total_cost_val)
    evaluator.add_custom_node(
        result=total_ok,
        id="Total_Cost_Stated",
        desc="States a single total permit cost consistent with the component fees.",
        parent=node,
        critical=True,
    )

    # Leaf: Cost_Reference_URLs (existence)
    evaluator.add_custom_node(
        result=len(urls) > 0,
        id="Cost_Reference_URLs",
        desc="Provides reference URL(s) supporting Yosemite fee structure and Whitney fee applicability.",
        parent=node,
        critical=True,
    )


async def verify_first_night(evaluator: Evaluator, parent_node, fn: FirstNightInfo) -> None:
    node = evaluator.add_parallel(
        id="First_Night_Camping",
        desc="State the first-night camping distance rule (past Little Yosemite Valley) and provide a supporting URL.",
        parent=parent_node,
        critical=True,
    )

    # Leaf: Distance_Rule_From_LYV (verify by URLs)
    dist_leaf = evaluator.add_leaf(
        id="Distance_Rule_From_LYV",
        desc="Correctly describes minimum first-night distance requirement relative to LYV for Donohue eligibility.",
        parent=node,
        critical=True,
    )
    rule_text = fn.rule_description or ""
    dist_text = fn.distance_miles_minimum or "2 miles"
    claim = (
        f"The first-night requirement for the Donohue Pass-eligible Yosemite Valley start is to camp at least {dist_text} "
        f"beyond Little Yosemite Valley (LYV). The answer states: '{rule_text}'."
    )
    await evaluator.verify(
        claim=claim,
        node=dist_leaf,
        sources=_ensure_list(fn.urls),
        additional_instruction="Verify Yosemite's Donohue Pass eligibility condition requiring first-night camping ≥2 miles past LYV for the Happy Isles to Past LYV permit.",
    )

    # Leaf: First_Night_Reference_URL (existence)
    evaluator.add_custom_node(
        result=len(_ensure_list(fn.urls)) > 0,
        id="First_Night_Reference_URL",
        desc="Provides at least one reference URL supporting the first-night camping rule.",
        parent=node,
        critical=True,
    )


async def verify_bear_canister(evaluator: Evaluator, parent_node, bc: BearCanInfo) -> None:
    node = evaluator.add_parallel(
        id="Bear_Canister",
        desc="Confirm bear canister requirement along the route, specify what must be stored, and provide a supporting URL.",
        parent=parent_node,
        critical=True,
    )

    # Leaf: Canister_Required_Entire_Route (verify by URLs)
    can_req_leaf = evaluator.add_leaf(
        id="Canister_Required_Entire_Route",
        desc="States that bear-resistant food storage containers are required along the entire JMT route.",
        parent=node,
        critical=True,
    )
    claim_req = (
        "Bear-resistant food storage containers (bear canisters) are required along the entire John Muir Trail route."
    )
    await evaluator.verify(
        claim=claim_req,
        node=can_req_leaf,
        sources=_ensure_list(bc.urls),
        additional_instruction="Confirm across Yosemite, Inyo NF/John Muir Wilderness, and Sequoia-Kings areas the canister requirement applies on the JMT.",
    )

    # Leaf: Items_To_Store_Listed (custom coverage check)
    evaluator.add_custom_node(
        result=_items_cover_required_categories(bc.items_to_store),
        id="Items_To_Store_Listed",
        desc="Specifies required categories to store in the canister (food, toiletries, other scented items).",
        parent=node,
        critical=True,
    )

    # Leaf: Bear_Canister_Reference_URL (existence)
    evaluator.add_custom_node(
        result=len(_ensure_list(bc.urls)) > 0,
        id="Bear_Canister_Reference_URL",
        desc="Provides at least one reference URL supporting canister requirements and what must be stored.",
        parent=node,
        critical=True,
    )


async def verify_campfire(evaluator: Evaluator, parent_node, cf: CampfireInfo) -> None:
    node = evaluator.add_parallel(
        id="Campfire_Regulations",
        desc="Determine campfire permission at 10,200 ft south of Glacier Divide and address conditional alternative method, with a URL.",
        parent=parent_node,
        critical=True,
    )

    urls = _ensure_list(cf.urls)
    elevation_txt = cf.elevation_ft or "10,200 ft"
    loc_txt = cf.location_desc or "John Muir Wilderness south of Glacier Divide"

    # Leaf: Campfire_Permitted_Determination (verify by URLs)
    permitted_leaf = evaluator.add_leaf(
        id="Campfire_Permitted_Determination",
        desc="Correctly determines whether campfires are permitted at 10,200 ft south of Glacier Divide considering the 10,400 ft threshold.",
        parent=node,
        critical=True,
    )
    claim_perm = (
        f"Campfires are permitted at {elevation_txt} in the {loc_txt}, given that campfires are prohibited only above 10,400 ft south of the Glacier Divide."
    )
    await evaluator.verify(
        claim=claim_perm,
        node=permitted_leaf,
        sources=urls,
        additional_instruction="Check official USFS/Inyo NF guidance for John Muir Wilderness: south of Glacier Divide campfires prohibited above 10,400 ft; below that, permitted subject to seasonal restrictions.",
    )

    # Leaf: Alternative_Cooking_Method_If_Prohibited (conditional satisfaction as critical, but logical OR)
    cond_satisfied = (cf.fire_permitted is True) or (
        cf.fire_permitted is False and isinstance(cf.alternative_cooking_method, str) and cf.alternative_cooking_method.strip() != ""
    )
    evaluator.add_custom_node(
        result=cond_satisfied,
        id="Alternative_Cooking_Method_If_Prohibited",
        desc="Conditional alternative cooking method satisfied: either fires permitted OR a portable stove (or equivalent) is named when fires are not permitted.",
        parent=node,
        critical=True,
    )

    # Leaf: Campfire_Reference_URL (existence)
    evaluator.add_custom_node(
        result=len(urls) > 0,
        id="Campfire_Reference_URL",
        desc="Provides at least one reference URL supporting the campfire elevation rule for the specified area.",
        parent=node,
        critical=True,
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
    model: str = "o4-mini",
) -> Dict:
    """
    Evaluate an answer for the JMT southbound permit planning task.
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

    # Perform extractions (can be parallelized)
    trailhead_task = evaluator.extract(
        prompt=prompt_extract_trailhead(),
        template_class=TrailheadInfo,
        extraction_name="permit_trailhead",
    )
    costs_task = evaluator.extract(
        prompt=prompt_extract_costs(),
        template_class=CostInfo,
        extraction_name="permit_costs",
    )
    first_night_task = evaluator.extract(
        prompt=prompt_extract_first_night(),
        template_class=FirstNightInfo,
        extraction_name="first_night_rule",
    )
    bear_can_task = evaluator.extract(
        prompt=prompt_extract_bear_canister(),
        template_class=BearCanInfo,
        extraction_name="bear_canister",
    )
    campfire_task = evaluator.extract(
        prompt=prompt_extract_campfire(),
        template_class=CampfireInfo,
        extraction_name="campfire_regulations",
    )

    trailhead_info, cost_info, first_night_info, bear_can_info, campfire_info = await asyncio.gather(
        trailhead_task, costs_task, first_night_task, bear_can_task, campfire_task
    )

    # Build verification tree according to rubric
    await verify_permit_trailhead(evaluator, root, trailhead_info)
    await verify_total_cost(evaluator, root, cost_info)
    await verify_first_night(evaluator, root, first_night_info)
    await verify_bear_canister(evaluator, root, bear_can_info)
    await verify_campfire(evaluator, root, campfire_info)

    # Return summary
    return evaluator.get_summary()