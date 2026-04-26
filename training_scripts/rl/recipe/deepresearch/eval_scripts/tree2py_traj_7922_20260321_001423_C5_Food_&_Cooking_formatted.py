import asyncio
import logging
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy

# --------------------------------------------------------------------------- #
# Task constants                                                              #
# --------------------------------------------------------------------------- #
TASK_ID = "atl_christmas_breakfast_2025"
TASK_DESCRIPTION = """
I am planning to host family members from out of town during the Christmas holiday in Atlanta, Georgia, and I want to take them out for breakfast on Christmas Day morning. I need to identify at least two restaurants that meet the following requirements:

1. The restaurant must be located within the Atlanta metropolitan area, specifically within a 15-mile radius of downtown Atlanta (coordinates: 33.7490° N, 84.3880° W).

2. The restaurant must be confirmed open on Christmas Day (December 25, 2025).

3. The restaurant must serve breakfast and must open before 10:00 a.m. on Christmas Day.

4. The restaurant must offer dine-in service on Christmas Day (not limited to takeout or delivery only).

5. At least one of the restaurants you identify must be a location of a nationally-recognized restaurant chain, defined as a chain with 50 or more locations across multiple U.S. states.

For each restaurant, please provide:
- The restaurant name
- The complete physical address
- The specific Christmas Day operating hours (opening and closing times)
- At least one authoritative reference URL that confirms the restaurant's Christmas Day hours and/or operational status (this can be the official restaurant website, official social media page, a reputable news article, or an established restaurant directory)
"""

ATL_DOWNTOWN_COORDS = (33.7490, -84.3880)
RADIUS_MILES = 15
CHRISTMAS_DATE_TEXT = "December 25, 2025"
OPEN_CUTOFF = "10:00 a.m."


# --------------------------------------------------------------------------- #
# Extraction models                                                           #
# --------------------------------------------------------------------------- #
class RestaurantItem(BaseModel):
    name: Optional[str] = None
    address: Optional[str] = None

    # Hours as presented in the answer for Christmas Day (verbatim text, e.g., "7:00 AM – 2:00 PM")
    christmas_hours: Optional[str] = None
    # If the answer provides a parsed opening/closing time for Christmas Day, capture them (optional)
    opening_time: Optional[str] = None
    closing_time: Optional[str] = None

    # Optional textual statements captured from the answer (if any)
    dine_in_text: Optional[str] = None

    # URLs explicitly mentioned in the answer that support hours/status, or the location
    urls: List[str] = Field(default_factory=list)
    hours_url: Optional[str] = None       # A page specifically about hours for this location (if present)
    map_url: Optional[str] = None         # A Google Maps/place page or equivalent (if present)

    # Chain identification (optional)
    chain_name: Optional[str] = None
    chain_info_urls: List[str] = Field(default_factory=list)  # URLs that establish chain scale/status (if provided)


class RestaurantsExtraction(BaseModel):
    restaurants: List[RestaurantItem] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_restaurants() -> str:
    return f"""
Extract up to five restaurants that the answer proposes for breakfast on Christmas Day in the Atlanta area. For each restaurant, return a JSON object with these fields:

- name: The restaurant name exactly as provided.
- address: The complete physical address as provided (street, city, state, ZIP if present).
- christmas_hours: The specific Christmas Day operating hours text (opening and closing time). Return verbatim text if present (e.g., "7:00 AM – 2:00 PM"). If not present, return null.
- opening_time: If the answer explicitly states a parsed Christmas Day opening time (e.g., "7:00 AM"), extract it. Else null.
- closing_time: If the answer explicitly states a parsed Christmas Day closing time (e.g., "2:00 PM"), extract it. Else null.
- dine_in_text: Any statement in the answer indicating dine-in availability on Christmas Day, if present. Else null.
- urls: An array of authoritative URLs explicitly mentioned in the answer that support Christmas Day hours and/or operational status. These can include official restaurant sites, official social media, reputable news, or established directories (Yelp, Google Maps place page, OpenTable, etc.). Include only valid URLs.
- hours_url: If one of the URLs is specifically a page about hours for this exact location (e.g., store hours page, holiday hours page), also provide it separately here. Else null.
- map_url: If a Google Maps (or similar) link for the specific location is included in the answer, provide it here. Else null.
- chain_name: If the restaurant is a branch/location of a larger chain and this is mentioned in the answer, provide the chain brand name (e.g., "Waffle House"). Else null.
- chain_info_urls: If the answer includes any URLs that establish the chain's national presence/scale (e.g., Wikipedia or corporate "About" pages mentioning number of locations and states), include them here. Else [].

Rules:
- Extract only from the provided answer. Do not invent or infer information not explicitly present.
- If any field is missing, set it to null (or [] where appropriate).
- Return a JSON object: {{ "restaurants": [ ... up to 5 items ... ] }}
"""


# --------------------------------------------------------------------------- #
# Utilities                                                                   #
# --------------------------------------------------------------------------- #
def _is_nonempty(s: Optional[str]) -> bool:
    return bool(s and str(s).strip())


def _dedup_preserve_order(items: List[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for x in items:
        if not x:
            continue
        x = x.strip()
        if not x:
            continue
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def _collect_sources(r: RestaurantItem) -> List[str]:
    urls: List[str] = []
    if _is_nonempty(r.hours_url):
        urls.append(r.hours_url)  # prioritize hours page if present
    if _is_nonempty(r.map_url):
        urls.append(r.map_url)
    urls.extend(r.urls or [])
    return _dedup_preserve_order(urls)


# --------------------------------------------------------------------------- #
# Per-restaurant verification                                                 #
# --------------------------------------------------------------------------- #
async def verify_one_restaurant(
    evaluator: Evaluator,
    parent_node,
    r: RestaurantItem,
    index: int,
) -> Dict[str, Any]:
    """
    Build the per-restaurant subtree and run verifications.

    Returns a dict carrying:
      - "parent": the Restaurant_i node
      - "nodes": mapping of critical child node IDs to node objects
      - "restaurant": the RestaurantItem
    """
    ridx = index + 1
    rest_node = evaluator.add_parallel(
        id=f"Restaurant_{ridx}",
        desc=f"Validation of candidate restaurant #{ridx}.",
        parent=parent_node,
        critical=False,  # Per JSON this container is non-critical; criticality applies to children
    )

    # Existence/required info checks (critical custom nodes)
    name_exists = _is_nonempty(r.name)
    addr_exists = _is_nonempty(r.address)
    hours_text_exists = _is_nonempty(r.christmas_hours)
    auth_urls = _collect_sources(r)
    has_auth_url = len(auth_urls) > 0

    node_name = evaluator.add_custom_node(
        result=name_exists,
        id=f"Provided_Name_R{ridx}",
        desc="Restaurant name is provided.",
        parent=rest_node,
        critical=True,
    )

    node_addr = evaluator.add_custom_node(
        result=addr_exists,
        id=f"Provided_Address_R{ridx}",
        desc="Complete physical address is provided.",
        parent=rest_node,
        critical=True,
    )

    node_hours_text = evaluator.add_custom_node(
        result=hours_text_exists,
        id=f"Provided_Christmas_Hours_R{ridx}",
        desc="Specific Christmas Day operating hours (opening and closing times) are provided.",
        parent=rest_node,
        critical=True,
    )

    node_has_url = evaluator.add_custom_node(
        result=has_auth_url,
        id=f"Authoritative_URL_R{ridx}",
        desc="At least one authoritative reference URL is provided that supports Christmas Day hours and/or open status.",
        parent=rest_node,
        critical=True,
    )

    # Verification leaves (critical)
    node_loc = evaluator.add_leaf(
        id=f"Location_Within_15mi_R{ridx}",
        desc=f"Restaurant location is within a {RADIUS_MILES}-mile radius of downtown Atlanta ({ATL_DOWNTOWN_COORDS[0]:.4f}° N, {ATL_DOWNTOWN_COORDS[1]:.4f}° W).",
        parent=rest_node,
        critical=True,
    )

    node_open = evaluator.add_leaf(
        id=f"Open_On_Christmas_Day_2025_R{ridx}",
        desc=f"Restaurant is confirmed open on {CHRISTMAS_DATE_TEXT}.",
        parent=rest_node,
        critical=True,
    )

    node_breakfast_open = evaluator.add_leaf(
        id=f"Breakfast_And_Open_Before_10am_R{ridx}",
        desc=f"Restaurant serves breakfast on Christmas Day AND opens before {OPEN_CUTOFF} local time on Christmas Day.",
        parent=rest_node,
        critical=True,
    )

    node_dinein = evaluator.add_leaf(
        id=f"DineIn_Service_R{ridx}",
        desc="Restaurant offers dine-in service on Christmas Day (not takeout/delivery only).",
        parent=rest_node,
        critical=True,
    )

    # Prepare claims
    addr_display = r.address or ""
    location_claim = (
        f"The restaurant located at '{addr_display}' is within a {RADIUS_MILES}-mile radius of downtown Atlanta "
        f"(coordinates {ATL_DOWNTOWN_COORDS[0]:.4f}, {ATL_DOWNTOWN_COORDS[1]:.4f})."
    )

    open_claim = (
        f"This restaurant is open on {CHRISTMAS_DATE_TEXT} (Christmas Day)."
    )

    breakfast_open_claim = (
        f"On {CHRISTMAS_DATE_TEXT}, the restaurant serves breakfast and opens before 10:00 a.m. local time."
    )

    dinein_claim = (
        f"The restaurant offers dine-in service on {CHRISTMAS_DATE_TEXT} (not limited to takeout or delivery only)."
    )

    # Batch verify the 4 factual leaves (URL-grounded)
    claims = [
        (
            location_claim,
            auth_urls,
            node_loc,
            (
                "Use the provided address and any map/location page to determine approximate distance from downtown Atlanta "
                f"({ATL_DOWNTOWN_COORDS[0]:.4f}, {ATL_DOWNTOWN_COORDS[1]:.4f}). If a Google Maps page is provided, you may use the map/screenshot context. "
                "If the page clearly shows the restaurant is well within the Atlanta core or inner suburbs (roughly the I-285 perimeter) and plausibly "
                f"within about {RADIUS_MILES} miles, consider it within range. If it's clearly far outside that radius, mark as not within 15 miles."
            ),
        ),
        (
            open_claim,
            auth_urls,
            node_open,
            (
                "Confirm that the evidence clearly indicates the location is open on December 25, 2025. "
                "Accept explicit Christmas Day hours, 'Open 24/7', or an official announcement that the location is open on Christmas Day 2025. "
                "If the page only lists generic hours with no holiday exceptions or is irrelevant, mark as not supported."
            ),
        ),
        (
            breakfast_open_claim,
            auth_urls,
            node_breakfast_open,
            (
                "Confirm that on December 25, 2025, the location is serving breakfast and opens before 10:00 a.m. local time. "
                "Evidence can include a breakfast menu with holiday service, a posted opening time before 10:00 a.m. on Christmas Day, "
                "or a Christmas brunch that clearly starts before 10:00 a.m. If unclear or opening time is at/after 10:00 a.m., mark as not supported."
            ),
        ),
        (
            dinein_claim,
            auth_urls,
            node_dinein,
            (
                "Confirm that dine-in is available on Christmas Day at this location. "
                "Look for indications such as 'dine-in available', table reservations, or seating. "
                "If evidence only confirms takeout/delivery with no dine-in, mark as not supported."
            ),
        ),
    ]

    # Execute verifications (will auto‑skip if critical prerequisites above failed)
    await evaluator.batch_verify(claims)

    return {
        "parent": rest_node,
        "nodes": {
            f"Provided_Name_R{ridx}": node_name,
            f"Provided_Address_R{ridx}": node_addr,
            f"Provided_Christmas_Hours_R{ridx}": node_hours_text,
            f"Authoritative_URL_R{ridx}": node_has_url,
            f"Location_Within_15mi_R{ridx}": node_loc,
            f"Open_On_Christmas_Day_2025_R{ridx}": node_open,
            f"Breakfast_And_Open_Before_10am_R{ridx}": node_breakfast_open,
            f"DineIn_Service_R{ridx}": node_dinein,
        },
        "restaurant": r,
    }


def _restaurant_is_valid(node_map: Dict[str, Any]) -> bool:
    """
    A restaurant is valid if ALL of its per-restaurant critical checks passed.
    """
    for n in node_map.values():
        # Node must be a leaf/custom with status 'passed'
        if getattr(n, "status", None) != "passed":
            return False
    return True


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
    model: str = "o4-mini",
) -> Dict:
    """
    Evaluate the answer for the Atlanta Christmas breakfast task.
    """
    evaluator = Evaluator()
    # IMPORTANT: Set root as non-critical to comply with critical-child constraint in the framework.
    # We'll enforce criticality within the "Global_Requirements" subtree.
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Root aggregates main sections in parallel
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

    # Extract structured restaurant info from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_restaurants(),
        template_class=RestaurantsExtraction,
        extraction_name="restaurants_extraction",
    )

    # Keep at most 5 items, pad nothing; Evaluation requires at least 2 valid ones anyway
    restaurants = (extracted.restaurants or [])[:5]

    # Build the "Candidate_Restaurants_Evaluation" subtree
    candidates_node = evaluator.add_parallel(
        id="Candidate_Restaurants_Evaluation",
        desc="Evaluate each restaurant provided in the answer against per-restaurant constraints.",
        parent=root,
        critical=False,  # Non-critical container
    )

    # Per-restaurant verification (parallelizable)
    per_restaurant_results: List[Dict[str, Any]] = []
    for idx, r in enumerate(restaurants):
        res = await verify_one_restaurant(evaluator, candidates_node, r, idx)
        per_restaurant_results.append(res)

    # Compute which restaurants meet ALL per-restaurant critical checks
    valid_indices: List[int] = []
    unique_name_addr_pairs = set()
    for idx, res in enumerate(per_restaurant_results):
        node_map: Dict[str, Any] = res["nodes"]
        if _restaurant_is_valid(node_map):
            valid_indices.append(idx)
            r: RestaurantItem = res["restaurant"]
            key = ((r.name or "").strip().lower(), (r.address or "").strip().lower())
            unique_name_addr_pairs.add(key)

    # Build "Global_Requirements" (critical) subtree
    global_reqs = evaluator.add_parallel(
        id="Global_Requirements",
        desc="Check global constraints across the set of candidate restaurants that passed all per-restaurant critical checks.",
        parent=root,
        critical=True,  # Critical section; all its children must be critical
    )

    # 1) Minimum_Valid_Restaurants (critical custom)
    min_valid = evaluator.add_custom_node(
        result=(len(valid_indices) >= 2),
        id="Minimum_Valid_Restaurants",
        desc="At least two candidate restaurants pass all per-restaurant critical requirements.",
        parent=global_reqs,
        critical=True,
    )

    # 2) Different_Restaurants (critical custom)
    # Count unique (name, address) among valid ones
    # Only meaningful if there are at least two valid
    unique_valid_count = 0
    if len(valid_indices) >= 2:
        # Recompute unique among VALID only
        unique_pairs_valid = set()
        for idx in valid_indices:
            r: RestaurantItem = per_restaurant_results[idx]["restaurant"]
            key = ((r.name or "").strip().lower(), (r.address or "").strip().lower())
            unique_pairs_valid.add(key)
        unique_valid_count = len(unique_pairs_valid)
    diff_valid = evaluator.add_custom_node(
        result=(unique_valid_count >= 2),
        id="Different_Restaurants",
        desc="The restaurants counted toward the minimum are different restaurants/locations (no duplicates).",
        parent=global_reqs,
        critical=True,
    )

    # 3) At_Least_One_National_Chain (critical leaf with verification)
    # Build claim over the VALID restaurants only
    valid_names = [ (per_restaurant_results[i]["restaurant"].name or f"Restaurant #{i+1}") for i in valid_indices ]
    chain_leaf = evaluator.add_leaf(
        id="At_Least_One_National_Chain",
        desc="Among the valid restaurants, at least one is a nationally-recognized restaurant chain (≥50 US locations across multiple states).",
        parent=global_reqs,
        critical=True,
    )

    # Collect supporting chain URLs from valid restaurants (prefer chain_info_urls, then fall back to general URLs)
    chain_urls: List[str] = []
    for i in valid_indices:
        r = per_restaurant_results[i]["restaurant"]
        if r.chain_info_urls:
            chain_urls.extend(r.chain_info_urls)
        else:
            # Fallback: general URLs (store locator pages or Wikipedia for the brand may already be present in 'urls')
            chain_urls.extend(_collect_sources(r))
    chain_urls = _dedup_preserve_order(chain_urls)

    chain_claim = (
        "Among the following valid restaurants: "
        + (", ".join(valid_names) if valid_names else "[]")
        + ", at least one is a location of a nationally-recognized restaurant chain with at least 50 locations across multiple U.S. states."
    )

    await evaluator.verify(
        claim=chain_claim,
        node=chain_leaf,
        sources=chain_urls if chain_urls else None,  # Will be skipped if prerequisites fail; else try URL-based verify
        additional_instruction=(
            "Use the provided URLs to verify that at least one restaurant is part of a national chain with ≥50 U.S. locations across multiple states. "
            "Good evidence includes Wikipedia or corporate 'About' pages that explicitly state the number of locations and multi-state presence, "
            "or reputable news sources citing the same. It's sufficient to verify any one qualifying chain among the list."
        ),
    )

    # Add useful custom info to the summary
    evaluator.add_custom_info(
        info={
            "total_candidates_parsed": len(restaurants),
            "valid_candidates_count": len(valid_indices),
            "valid_candidate_indices": valid_indices,
            "valid_candidate_names": valid_names,
        },
        info_type="diagnostics",
        info_name="evaluation_diagnostics",
    )

    return evaluator.get_summary()