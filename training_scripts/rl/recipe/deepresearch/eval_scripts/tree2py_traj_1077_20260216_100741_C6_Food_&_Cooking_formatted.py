import asyncio
import logging
import re
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "denver_christmas_breakfast"
TASK_DESCRIPTION = """
Identify four breakfast or brunch restaurants in the Denver, Colorado metropolitan area that are open and serving breakfast on Christmas Day morning (between 7:00 AM and 11:00 AM). For each restaurant, provide the restaurant name, complete physical address, and a reference URL that confirms the restaurant's existence and location.

Additionally, verify and document the following for each restaurant:
- Confirmation that the restaurant is open on December 25th (Christmas Day)
- Confirmation that the operating hours include the 7:00 AM to 11:00 AM time window on Christmas Day
- Confirmation that breakfast or brunch service is available during these morning hours
- Classification of the service type as either: sit-down dine-in, drive-thru, or 24-hour service
- Reference URLs supporting the Christmas Day hours, breakfast service availability, and service type

The four restaurants you identify must collectively satisfy these additional requirements:
- At least two of the four restaurants must be sit-down dine-in establishments
- At least one of the four restaurants must offer 24-hour service on Christmas Day
""".strip()


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class RestaurantItem(BaseModel):
    name: Optional[str] = None
    address: Optional[str] = None
    location_reference_urls: List[str] = Field(default_factory=list)
    hours_reference_urls: List[str] = Field(default_factory=list)
    menu_reference_urls: List[str] = Field(default_factory=list)
    service_reference_urls: List[str] = Field(default_factory=list)
    service_type: Optional[str] = None  # expected one of: "sit-down dine-in", "drive-thru", "24-hour service"


class RestaurantsExtraction(BaseModel):
    restaurants: List[RestaurantItem] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_restaurants() -> str:
    return """
Extract up to four restaurants described in the answer that are candidates for breakfast/brunch on Christmas Day morning in the Denver, Colorado metropolitan area.

For each restaurant, extract the following fields exactly as presented in the answer:
- name: The restaurant's specific name (string).
- address: The complete physical address as provided in the answer (string). Do not infer or add anything.
- location_reference_urls: An array of URL(s) that confirm the restaurant's existence and location (e.g., official site, Google Maps, Yelp). Extract only URLs explicitly present in the answer.
- hours_reference_urls: An array of URL(s) that support being open on December 25 and the operating hours for that day. Extract only URLs explicitly present in the answer. If none are given, return an empty array.
- menu_reference_urls: An array of URL(s) that support breakfast or brunch availability (e.g., menus, breakfast pages). Extract only URLs explicitly present in the answer. If none are given, return an empty array.
- service_reference_urls: An array of URL(s) that support the service type classification (e.g., dine-in/drive-thru/24-hour). Extract only URLs explicitly present in the answer. If none are given, return an empty array.
- service_type: A single label chosen from exactly one of the following, as explicitly stated in the answer:
  "sit-down dine-in", "drive-thru", or "24-hour service".
  If the answer implies 24/7 or 24 hours, set to "24-hour service". If the answer implies dine-in seating, set to "sit-down dine-in".
  If the answer does not explicitly state the service type, return null.

Rules:
- Do not invent URLs. Only include those explicitly present in the answer. If missing, return an empty array.
- Normalize any URL missing a protocol by prepending "http://".
- If fewer than four restaurants are present, return all that are mentioned.
- Preserve the exact strings for name and address as written in the answer (no normalization).
""".strip()


# --------------------------------------------------------------------------- #
# Utility helpers                                                             #
# --------------------------------------------------------------------------- #
DENVER_METRO_HINT = (
    "Treat 'Denver metropolitan area' as including Denver and nearby cities such as Aurora, Lakewood, Arvada, "
    "Westminster, Thornton, Centennial, Commerce City, Littleton, Englewood, Northglenn, Brighton, Golden, "
    "Wheat Ridge, Parker, Lone Tree, etc."
)

def _normalize_service_label(label: Optional[str]) -> Optional[str]:
    if not label:
        return None
    s = label.strip().lower()
    # normalize common synonyms/variants
    if "24" in s or "24/7" in s or "24-hour" in s or "24 hour" in s or "open 24" in s:
        return "24-hour service"
    if "dine" in s or "sit" in s:  # sit-down, dine-in
        return "sit-down dine-in"
    if "drive" in s and "thru" in s or "drive-thru" in s or "drive thru" in s:
        return "drive-thru"
    # Fallback to original sanitized, but we prefer strict set
    if s in {"sit-down dine-in", "drive-thru", "24-hour service"}:
        return s
    return s

def _count_service_types(restaurants: List[RestaurantItem]) -> Dict[str, int]:
    counts = {"sit-down dine-in": 0, "drive-thru": 0, "24-hour service": 0}
    for r in restaurants:
        norm = _normalize_service_label(r.service_type)
        if norm in counts:
            counts[norm] += 1
    return counts

def _has_nonempty_urls(urls: Optional[List[str]]) -> bool:
    return bool(urls and any(u and u.strip() for u in urls))

async def _verify_with_urls_or_fail(
    evaluator: Evaluator,
    node,
    claim: str,
    urls: Optional[List[str]],
    additional_instruction: str = "None",
) -> bool:
    """
    Helper: if urls exist and non-empty, run URL-based verify; otherwise mark node failed.
    """
    if _has_nonempty_urls(urls):
        return await evaluator.verify(
            claim=claim,
            node=node,
            sources=urls,
            additional_instruction=additional_instruction
        )
    else:
        node.score = 0.0
        node.status = "failed"
        return False


# --------------------------------------------------------------------------- #
# Verification for a single restaurant                                        #
# --------------------------------------------------------------------------- #
async def verify_restaurant(
    evaluator: Evaluator,
    parent_node,
    rest: RestaurantItem,
    index: int
) -> None:
    """
    Build and verify the subtree for one restaurant (index 1..4).
    """
    rid = f"restaurant_{index}"
    display_num = f"#{index}"

    # Top node for this restaurant (non-critical to allow partial credit per restaurant)
    rest_node = evaluator.add_parallel(
        id=rid,
        desc=f"{['First','Second','Third','Fourth'][index-1]} restaurant meeting all specified criteria",
        parent=parent_node,
        critical=False
    )

    # -------------------- Basic Info (critical) -------------------------- #
    basic_node = evaluator.add_parallel(
        id=f"{rid}_basic_info",
        desc=f"Basic identification information for the {['first','second','third','fourth'][index-1]} restaurant",
        parent=rest_node,
        critical=True
    )

    # Name provided
    evaluator.add_custom_node(
        result=bool(rest.name and rest.name.strip()),
        id=f"{rid}_name",
        desc="Specific restaurant name is provided",
        parent=basic_node,
        critical=True
    )

    # Address provided
    evaluator.add_custom_node(
        result=bool(rest.address and rest.address.strip()),
        id=f"{rid}_address",
        desc="Complete physical address in Denver metropolitan area is provided",
        parent=basic_node,
        critical=True
    )

    # Location reference URL is provided (presence check)
    evaluator.add_custom_node(
        result=_has_nonempty_urls(rest.location_reference_urls),
        id=f"{rid}_location_reference",
        desc="Valid reference URL confirming restaurant existence and location is provided",
        parent=basic_node,
        critical=True
    )

    # Verify location & existence via URL evidence
    # This adds a concrete web-grounded verification to ensure the URL actually supports the name/address.
    loc_supported_node = evaluator.add_leaf(
        id=f"{rid}_location_supported",
        desc="The provided location reference supports the restaurant name and address (Denver metro)",
        parent=basic_node,
        critical=True
    )
    name_for_claim = rest.name or ""
    addr_for_claim = rest.address or ""
    loc_claim = (
        f"The provided webpage confirms a restaurant named '{name_for_claim}' located at '{addr_for_claim}', "
        f"which is in the Denver, Colorado metropolitan area."
    )
    await _verify_with_urls_or_fail(
        evaluator,
        loc_supported_node,
        loc_claim,
        rest.location_reference_urls,
        additional_instruction=f"Confirm that the page shows the restaurant name and address (or clearly the same location). "
                               f"{DENVER_METRO_HINT} Minor formatting/name variations are acceptable."
    )

    # -------------------- Verification of ops/service (critical) --------- #
    verify_node = evaluator.add_parallel(
        id=f"{rid}_verification",
        desc=f"Verification of Christmas Day operations and service requirements for the {['first','second','third','fourth'][index-1]} restaurant",
        parent=rest_node,
        critical=True
    )

    # ---- Christmas operations (critical) ----
    xmas_node = evaluator.add_parallel(
        id=f"{rid}_christmas_ops",
        desc="Christmas Day operating status and hours verification",
        parent=verify_node,
        critical=True
    )

    # Presence of hours reference URL(s)
    evaluator.add_custom_node(
        result=_has_nonempty_urls(rest.hours_reference_urls),
        id=f"{rid}_hours_reference",
        desc="Valid reference URL supporting the Christmas Day hours is provided",
        parent=xmas_node,
        critical=True
    )

    # Open on Christmas Day
    open_node = evaluator.add_leaf(
        id=f"{rid}_open_christmas",
        desc="Evidence confirms restaurant is open on December 25th",
        parent=xmas_node,
        critical=True
    )
    await _verify_with_urls_or_fail(
        evaluator,
        open_node,
        "The restaurant is open on December 25 (Christmas Day).",
        rest.hours_reference_urls,
        additional_instruction="Look for special hours or holiday announcements indicating Christmas Day is open. "
                               "If the page clearly states 'open 24 hours' or '24/7', this satisfies being open on December 25."
    )

    # Morning window 7:00–11:00 AM on Christmas Day
    morning_node = evaluator.add_leaf(
        id=f"{rid}_morning_hours",
        desc="Operating hours include the 7:00 AM to 11:00 AM window on Christmas Day",
        parent=xmas_node,
        critical=True
    )
    await _verify_with_urls_or_fail(
        evaluator,
        morning_node,
        "On December 25 (Christmas Day), the operating hours include the morning window between 7:00 AM and 11:00 AM (inclusive). "
        "If the restaurant is open 24 hours on that date, this condition is satisfied.",
        rest.hours_reference_urls,
        additional_instruction="Verify that Christmas Day hours show either 'open 24 hours' or an opening time at or before 7:00 AM "
                               "and a closing time at or after 11:00 AM."
    )

    # ---- Breakfast availability (critical) ----
    breakfast_node = evaluator.add_parallel(
        id=f"{rid}_breakfast",
        desc="Breakfast service availability verification",
        parent=verify_node,
        critical=True
    )

    # Presence of breakfast/menu reference URL(s)
    evaluator.add_custom_node(
        result=_has_nonempty_urls(rest.menu_reference_urls),
        id=f"{rid}_menu_reference",
        desc="Valid reference URL supporting breakfast/brunch service availability is provided",
        parent=breakfast_node,
        critical=True
    )

    # Breakfast served during morning hours (general)
    bfast_avail_node = evaluator.add_leaf(
        id=f"{rid}_breakfast_available",
        desc="Evidence confirms breakfast or brunch menu is served during morning hours",
        parent=breakfast_node,
        critical=True
    )
    await _verify_with_urls_or_fail(
        evaluator,
        bfast_avail_node,
        "The restaurant serves breakfast or brunch during morning hours (e.g., typically early hours such as 6–11 AM).",
        rest.menu_reference_urls,
        additional_instruction="Look for menu pages or descriptions that explicitly mention breakfast or brunch service "
                               "and the time window in the morning. Minor variations in time range are acceptable."
    )

    # ---- Service type classification (critical) ----
    service_node = evaluator.add_parallel(
        id=f"{rid}_service",
        desc="Service type classification verification",
        parent=verify_node,
        critical=True
    )

    # Presence of service type reference URL(s)
    evaluator.add_custom_node(
        result=_has_nonempty_urls(rest.service_reference_urls),
        id=f"{rid}_service_reference",
        desc="Valid reference URL confirming the service type is provided",
        parent=service_node,
        critical=True
    )

    # Service type classification is correct and supported
    svc_type_node = evaluator.add_leaf(
        id=f"{rid}_service_type",
        desc="Restaurant is classified as either sit-down dine-in, drive-thru, or 24-hour service",
        parent=service_node,
        critical=True
    )
    service_label = _normalize_service_label(rest.service_type) or (rest.service_type or "")
    await _verify_with_urls_or_fail(
        evaluator,
        svc_type_node,
        f"The restaurant's service type is '{service_label}'. Allowed labels are 'sit-down dine-in', 'drive-thru', or '24-hour service'. "
        f"'Open 24 hours' qualifies for '24-hour service'.",
        rest.service_reference_urls,
        additional_instruction="Allow reasonable synonyms: e.g., 'dine-in' or 'table service' for sit-down; 'drive thru' for drive-thru; "
                               "'open 24 hours' or '24/7' for 24-hour service."
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
    Evaluate an answer for the Denver Christmas breakfast/brunch task using the Mind2Web2 evaluation framework.
    """
    # Initialize evaluator; root should be NON-CRITICAL to allow non-critical children per framework constraints
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,
        agent_name=agent_name,
        answer_name=answer_name,
        client=client,
        task_description="Identify four breakfast/brunch restaurants in Denver metro open and serving breakfast on Christmas morning (7–11 AM), "
                         "with at least two sit-down dine-in and at least one 24-hour service.",
        answer=answer,
        global_cache=cache,
        global_semaphore=semaphore,
        logger=logger,
        default_model=model
    )
    # Override root description to match JSON description
    root.desc = "Identify four breakfast or brunch restaurants in the Denver, Colorado metropolitan area that are open and serving breakfast on Christmas Day morning (7:00 AM to 11:00 AM), with at least two being sit-down dine-in restaurants and at least one offering 24-hour service."
    root.critical = False  # Important: keep root non-critical to satisfy framework rule

    # Extract structured restaurant information
    extracted = await evaluator.extract(
        prompt=prompt_extract_restaurants(),
        template_class=RestaurantsExtraction,
        extraction_name="restaurants_extraction"
    )

    # Keep exactly four restaurants (pad with empty items if fewer)
    restaurants: List[RestaurantItem] = (extracted.restaurants or [])[:4]
    while len(restaurants) < 4:
        restaurants.append(RestaurantItem())

    # Build/verify per-restaurant subtrees
    for i, rest in enumerate(restaurants, start=1):
        await verify_restaurant(evaluator, root, rest, i)

    # Aggregate requirements across four restaurants
    agg_node = evaluator.add_parallel(
        id="aggregate_requirements",
        desc="The collective set of four restaurants meets the specified service type distribution requirements",
        parent=root,
        critical=True
    )

    counts = _count_service_types(restaurants)
    dine_in_ok = counts.get("sit-down dine-in", 0) >= 2
    twentyfour_ok = counts.get("24-hour service", 0) >= 1

    evaluator.add_custom_node(
        result=dine_in_ok,
        id="minimum_dine_in",
        desc="At least two of the four restaurants are sit-down dine-in establishments",
        parent=agg_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=twentyfour_ok,
        id="minimum_24hour",
        desc="At least one of the four restaurants offers 24-hour service on Christmas Day",
        parent=agg_node,
        critical=True
    )

    # Record some custom info for debugging
    evaluator.add_custom_info(
        info={
            "service_type_counts": counts,
            "restaurants_extracted": len(restaurants)
        },
        info_type="debug_info",
        info_name="aggregation_stats"
    )

    # Return evaluation summary
    return evaluator.get_summary()