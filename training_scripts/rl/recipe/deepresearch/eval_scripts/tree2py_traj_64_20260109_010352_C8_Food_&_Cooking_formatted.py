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
TASK_ID = "culinary_awards_2024_restaurants"
TASK_DESCRIPTION = """
Identify 4 restaurants that won culinary awards in 2024, where each restaurant satisfies ALL of the following criteria:

1. The restaurant won either a 2024 James Beard Award in the "Best Chef" category (any regional category such as Best Chef: California, Best Chef: Great Lakes, etc.) OR the restaurant received a new Michelin star (One Star, Two Stars, or Three Stars) in the 2024 Michelin Guide for any U.S. city.

2. Each of the 4 restaurants must be located in a different U.S. state.

3. For each restaurant, provide:
   - Restaurant name
   - Chef name (the chef or chef-owner associated with the award)
   - Cuisine type
   - Complete location (city and state)
   - Full street address
   - A reference URL documenting the award

Ensure that all 4 restaurants are in different states and that all information is accurate and verifiable.
"""


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class RestaurantItem(BaseModel):
    """Structured information for a single restaurant entry in the answer."""
    name: Optional[str] = None
    chef: Optional[str] = None
    cuisine: Optional[str] = None
    city: Optional[str] = None
    state: Optional[str] = None
    address: Optional[str] = None

    # Award-related fields
    award_type: Optional[str] = None  # e.g., "James Beard Best Chef" or "Michelin Star"
    award_detail: Optional[str] = None  # e.g., "Best Chef: California" or "New One Star – Chicago"
    award_year: Optional[str] = None  # should be '2024'
    award_city_or_region: Optional[str] = None  # For Michelin: city; For JBF: region
    star_level: Optional[str] = None  # "One Star", "Two Stars", "Three Stars" (Michelin)
    award_text: Optional[str] = None  # Free-form description as in the answer

    # Sources
    reference_url: Optional[str] = None  # Award documentation URL (required)
    extra_info_urls: List[str] = Field(default_factory=list)  # Additional URLs supporting chef/cuisine/location/address


class RestaurantsExtraction(BaseModel):
    """Top-level extraction model capturing multiple restaurants."""
    restaurants: List[RestaurantItem] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_restaurants() -> str:
    return """
    Extract up to the first 6 restaurant entries mentioned in the answer that claim a qualifying 2024 culinary award.
    For each restaurant, return a JSON object with the following fields:

    1) name: The restaurant name exactly as stated.
    2) chef: The chef or chef-owner associated with the award (if provided).
    3) cuisine: The cuisine type (e.g., Italian, Contemporary American, Vietnamese).
    4) city: The city.
    5) state: The U.S. state (use the full state name if available; otherwise, use the abbreviation).
    6) address: The full street address (include street number, street name, city, state, and ZIP if present).
    7) reference_url: A single URL explicitly documenting the award (prefer official sources, e.g., James Beard Foundation website for winners or Michelin Guide official pages for 2024 awards). If multiple award URLs are present, choose the most authoritative one.
    8) award_type: One of ["James Beard Best Chef", "Michelin Star"] based on the answer text. If unclear, set to null.
    9) award_detail: The specific detail string, e.g., "Best Chef: California" or "New One Star – Chicago".
    10) award_year: The year of the award (should be 2024 if the answer claims so). If not explicitly present, set to null.
    11) award_city_or_region: For Michelin, the city (e.g., "Chicago"); For JBF Best Chef, the region name (e.g., "California", "Great Lakes").
    12) star_level: For Michelin only, one of ["One Star", "Two Stars", "Three Stars"]. If not applicable or not provided, set to null.
    13) award_text: A free-form textual description of the award as stated in the answer (verbatim or closely paraphrased).
    14) extra_info_urls: An array of any additional URLs the answer provides to support chef, cuisine, location, or address facts (e.g., restaurant website, Michelin page listing cuisine/address, local press, etc.). Exclude duplicates of reference_url.

    RULES:
    - Extract only what appears explicitly in the answer; do not invent information.
    - If any required field is missing or not present in the answer, set it to null (for strings) or [] (for arrays).
    - For URLs, extract actual valid URLs (plain or markdown). If multiple URLs are given, include only one for `reference_url` and put the rest in `extra_info_urls` when relevant.
    - Preserve name spellings and diacritics as given. Do not normalize or change names.
    - Return a JSON object with a single field "restaurants" that is an array of these per-restaurant objects.
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def _collect_sources(item: RestaurantItem) -> List[str]:
    """Collect all available URLs for verification for a restaurant entry."""
    urls: List[str] = []
    if item.reference_url and item.reference_url.strip():
        urls.append(item.reference_url.strip())
    for u in item.extra_info_urls or []:
        if isinstance(u, str) and u.strip():
            urls.append(u.strip())
    # Deduplicate while preserving order
    seen = set()
    unique_urls = []
    for u in urls:
        if u not in seen:
            unique_urls.append(u)
            seen.add(u)
    return unique_urls


def _classify_award(item: RestaurantItem) -> str:
    """Classify award type based on extracted fields and text; returns 'JBF', 'MICHELIN', or 'UNKNOWN'."""
    atxt = (item.award_type or "") + " " + (item.award_detail or "") + " " + (item.award_text or "")
    lower = atxt.lower()
    if "james beard" in lower or "best chef" in lower:
        return "JBF"
    if "michelin" in lower or "star" in lower:
        return "MICHELIN"
    return "UNKNOWN"


def _safe_str(x: Optional[str]) -> str:
    return (x or "").strip()


# --------------------------------------------------------------------------- #
# Verification functions                                                      #
# --------------------------------------------------------------------------- #
async def verify_single_restaurant(
        evaluator: Evaluator,
        parent_node,
        item: RestaurantItem,
        idx: int,
) -> None:
    """
    Build verification sub-tree and run checks for one restaurant.
    """
    # Create per-restaurant parallel node (non-critical to allow partial scoring across restaurants)
    rnode = evaluator.add_parallel(
        id=f"restaurant_{idx + 1}",
        desc=f"{idx + 1}st restaurant (must satisfy all per-restaurant requirements)" if idx == 0 else
             (f"{idx + 1}nd restaurant (must satisfy all per-restaurant requirements)" if idx == 1 else
              (f"{idx + 1}rd restaurant (must satisfy all per-restaurant requirements)" if idx == 2 else
               f"{idx + 1}th restaurant (must satisfy all per-restaurant requirements)")),
        parent=parent_node,
        critical=False
    )

    # 0) Reference URL existence (critical)
    ref_present = bool(item.reference_url and item.reference_url.strip())
    ref_node = evaluator.add_custom_node(
        result=ref_present,
        id=f"reference_url_{idx + 1}",
        desc="A reference URL from an official or otherwise reliable source documenting the award is provided",
        parent=rnode,
        critical=True
    )

    # Collect sources (include extra info urls if provided)
    sources = _collect_sources(item)

    # 1) Restaurant name verification (critical)
    name_leaf = evaluator.add_leaf(
        id=f"restaurant_name_{idx + 1}",
        desc="Restaurant name is provided",
        parent=rnode,
        critical=True
    )
    name_claim = f"The provided page(s) show the restaurant named '{_safe_str(item.name)}'."
    await evaluator.verify(
        claim=name_claim,
        node=name_leaf,
        sources=sources,
        additional_instruction="Verify that the page explicitly includes the restaurant name. Reject if the name is blank or does not match.",
        extra_prerequisites=[ref_node]
    )

    # 2) Award verification (critical)
    award_leaf = evaluator.add_leaf(
        id=f"award_verification_{idx + 1}",
        desc=("Restaurant won either (a) a 2024 James Beard Award in a Best Chef regional category "
              "OR (b) received a new Michelin star (1/2/3) in a 2024 Michelin Guide for a U.S. city"),
        parent=rnode,
        critical=True
    )
    award_kind = _classify_award(item)
    if award_kind == "JBF":
        # Specific JBF claim
        jbf_claim = (
            f"In 2024, chef '{_safe_str(item.chef)}' associated with '{_safe_str(item.name)}' "
            f"won a James Beard Award in a Best Chef regional category"
            + (f" ('{_safe_str(item.award_city_or_region)}')." if _safe_str(item.award_city_or_region) else ".")
        )
        award_instruction = (
            "Confirm the page is an official or reliable source that explicitly lists a 2024 James Beard "
            "Best Chef (regional) WINNER (not semifinalist/finalist). The chef should be clearly associated "
            "with the restaurant. Minor name variants are acceptable."
        )
    elif award_kind == "MICHELIN":
        # Specific Michelin claim
        michelin_claim = (
            f"In the 2024 Michelin Guide for the U.S."
            + (f" ({_safe_str(item.award_city_or_region)})" if _safe_str(item.award_city_or_region) else "")
            + f", '{_safe_str(item.name)}' received a NEW Michelin {_safe_str(item.star_level) or 'star'}."
        )
        award_instruction = (
            "Confirm the page explicitly indicates that the restaurant received a NEW Michelin star in the 2024 "
            "Michelin Guide (not merely retained from prior years). If the page does not clearly indicate 'new' for 2024, reject."
        )
    else:
        # Generic claim fallback if classification is unclear
        generic_claim = (
            f"The provided page(s) explicitly document that in 2024, '{_safe_str(item.name)}' or its chef "
            f"('{_safe_str(item.chef)}') achieved a qualifying award: either a James Beard Best Chef (regional) WINNER "
            f"or a NEW Michelin star in the 2024 Michelin Guide."
        )
        award_instruction = (
            "Verify that ONE of the following is clearly true on the page: "
            "(1) A 2024 James Beard Best Chef (regional) WINNER tied to the restaurant/chef; OR "
            "(2) A NEW Michelin star awarded in the 2024 Michelin Guide to the restaurant. "
            "Reject if the page indicates finalist/nominee/semifinalist instead of winner, or retained star instead of new."
        )

    await evaluator.verify(
        claim=(jbf_claim if award_kind == "JBF" else (michelin_claim if award_kind == "MICHELIN" else generic_claim)),
        node=award_leaf,
        sources=sources,
        additional_instruction=award_instruction,
        extra_prerequisites=[ref_node]
    )

    # 3) Chef identification (critical)
    chef_leaf = evaluator.add_leaf(
        id=f"chef_identification_{idx + 1}",
        desc="Chef or chef-owner associated with the award is correctly identified",
        parent=rnode,
        critical=True
    )
    chef_claim = (
        f"The page(s) indicate that the award is associated with chef '{_safe_str(item.chef)}' "
        f"for '{_safe_str(item.name)}'."
    )
    await evaluator.verify(
        claim=chef_claim,
        node=chef_leaf,
        sources=sources,
        additional_instruction="Confirm that the chef named is directly tied to the award and the restaurant on the page(s). Reject if the chef is not mentioned or association is unclear.",
        extra_prerequisites=[ref_node]
    )

    # 4) Cuisine type (critical)
    cuisine_leaf = evaluator.add_leaf(
        id=f"cuisine_type_{idx + 1}",
        desc="Cuisine type is identified",
        parent=rnode,
        critical=True
    )
    cuisine_claim = f"The cuisine type of '{_safe_str(item.name)}' is '{_safe_str(item.cuisine)}'."
    await evaluator.verify(
        claim=cuisine_claim,
        node=cuisine_leaf,
        sources=sources,
        additional_instruction="Verify that the page(s) explicitly show or indicate the cuisine (e.g., Italian, Contemporary American). Accept equivalent wording. Reject if missing.",
        extra_prerequisites=[ref_node]
    )

    # 5) Location details (critical)
    location_leaf = evaluator.add_leaf(
        id=f"location_details_{idx + 1}",
        desc="City and state are provided",
        parent=rnode,
        critical=True
    )
    loc_claim = f"The restaurant '{_safe_str(item.name)}' is located in {_safe_str(item.city)}, {_safe_str(item.state)}."
    await evaluator.verify(
        claim=loc_claim,
        node=location_leaf,
        sources=sources,
        additional_instruction="Verify that the city and state on the page(s) match the claim. Reject if either is missing or does not match.",
        extra_prerequisites=[ref_node]
    )

    # 6) Full street address (critical)
    address_leaf = evaluator.add_leaf(
        id=f"address_{idx + 1}",
        desc="Full street address is provided",
        parent=rnode,
        critical=True
    )
    addr_claim = f"The full street address of '{_safe_str(item.name)}' is '{_safe_str(item.address)}'."
    await evaluator.verify(
        claim=addr_claim,
        node=address_leaf,
        sources=sources,
        additional_instruction="Verify that the page(s) contain the same full street address. Accept minor formatting variation; reject if missing or clearly mismatched.",
        extra_prerequisites=[ref_node]
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
    Evaluate an answer for the 2024 culinary awards restaurants task.
    """
    # Initialize evaluator (root parallel aggregation; set non-critical to allow partial scoring)
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

    # Extract restaurants from the answer
    extraction = await evaluator.extract(
        prompt=prompt_extract_restaurants(),
        template_class=RestaurantsExtraction,
        extraction_name="restaurants_extraction",
    )

    # Prepare exactly 4 restaurants (pad with empty if fewer, take first 4 if more)
    items: List[RestaurantItem] = list(extraction.restaurants[:4])
    while len(items) < 4:
        items.append(RestaurantItem())

    # Build verification subtrees for each restaurant
    for idx in range(4):
        await verify_single_restaurant(
            evaluator=evaluator,
            parent_node=root,
            item=items[idx],
            idx=idx
        )

    # Geographic diversity check (critical)
    states = [(_safe_str(it.state)) for it in items]
    # Fail if any missing or not 4 unique
    states_nonempty = all(bool(s) for s in states)
    unique_states = len(set(s.upper() for s in states if s)) == 4
    geo_diverse = states_nonempty and unique_states

    evaluator.add_custom_node(
        result=geo_diverse,
        id="geographic_diversity",
        desc="All 4 restaurants are located in different U.S. states",
        parent=root,
        critical=True
    )

    # Optional: record custom info
    evaluator.add_custom_info(
        info={"states": states, "unique_states": len(set(s.upper() for s in states if s))},
        info_type="geography_stats",
        info_name="geo_diversity_check"
    )

    # Return unified summary
    return evaluator.get_summary()