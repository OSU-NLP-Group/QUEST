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
TASK_ID = "chicago_group_dining"
TASK_DESCRIPTION = (
    "I'm planning a group dinner in Chicago for approximately 10 people. Several guests have celiac disease and require strict gluten-free options with no risk of cross-contamination, while others follow a vegan diet. "
    "I'm looking for restaurants that can provide outdoor seating for our gathering. Find two restaurants in Chicago that meet ALL of the following requirements: "
    "(1) Located in Chicago, Illinois, "
    "(2) Provide celiac-safe gluten-free dining options (either as a 100% dedicated gluten-free facility OR with documented gluten-free certification/procedures), "
    "(3) Offer substantial vegan menu options with multiple vegan entrees (not just salads), "
    "(4) Have outdoor seating or patio dining available, "
    "(5) Can accommodate group dining for at least 8-10 people. "
    "For each restaurant, provide the name, address, a brief description of how it meets the dietary requirements, and a reference URL that verifies these details."
)

# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class RestaurantItem(BaseModel):
    """One restaurant entry extracted from the answer."""
    name: Optional[str] = None
    address: Optional[str] = None
    gf_safety: Optional[str] = None  # Evidence/description for celiac-safe GF
    vegan_options: Optional[str] = None  # Evidence/description for multiple vegan entrees
    outdoor_seating: Optional[str] = None  # Evidence/description for outdoor/patio availability
    group_capacity: Optional[str] = None  # Evidence/description for accommodating groups 8-10+
    requirement_summary: Optional[str] = None  # Brief summary how it meets dietary needs
    reference_urls: List[str] = Field(default_factory=list)  # URLs verifying details


class RestaurantsExtraction(BaseModel):
    """Extraction of restaurants from the answer."""
    restaurants: List[RestaurantItem] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_restaurants() -> str:
    return """
    Extract restaurants mentioned in the answer that are proposed for the Chicago group dinner.
    For EACH restaurant, extract the following fields exactly from the answer:
    - name: The restaurant name
    - address: Street address as provided (if any). If only city/neighborhood is provided, include that text.
    - gf_safety: A short text snippet explaining the gluten-free safety (e.g., '100% gluten-free', 'gluten-free certification', 'dedicated GF kitchen', 'celiac-safe procedures', 'strict cross-contamination protocols'), as described in the answer.
    - vegan_options: A short text snippet indicating substantial vegan options (ideally multiple vegan entrees beyond salads), as described in the answer.
    - outdoor_seating: A short text snippet indicating outdoor seating/patio availability (e.g., 'outdoor patio', 'seasonal outdoor seating').
    - group_capacity: A short text snippet indicating the ability to accommodate groups of at least 8-10 people (e.g., 'private dining', 'large party reservations', 'group seating for 10').
    - requirement_summary: A brief summary sentence from the answer of how the restaurant meets the dietary requirements for celiac-safe gluten-free and vegan diners.
    - reference_urls: An array of URLs explicitly provided in the answer that can verify the restaurant's details. Include only valid URLs. Do not invent URLs.

    Rules:
    - Return all restaurants mentioned. If more than two are present, include them all; downstream will select the first two.
    - If a field is not present in the answer, set it to null (or empty array for reference_urls).
    - Prefer full URLs; if a URL is missing protocol, prepend http://.
    """


# --------------------------------------------------------------------------- #
# Verification logic                                                          #
# --------------------------------------------------------------------------- #
async def verify_restaurant(
    evaluator: Evaluator,
    parent_node,
    restaurant: RestaurantItem,
    index: int,
) -> None:
    """
    Build verification subtree for one restaurant and run checks.
    index: 0 for Restaurant_1, 1 for Restaurant_2
    """
    idx = index + 1
    node_id = f"Restaurant_{idx}"
    node_desc = "First restaurant meeting all requirements" if idx == 1 else "Second restaurant meeting all requirements"

    # Parent node for this restaurant (non-critical to allow partial scoring across restaurants)
    rest_node = evaluator.add_parallel(
        id=node_id,
        desc=node_desc,
        parent=parent_node,
        critical=False
    )

    # Convenience
    name = (restaurant.name or "").strip()
    addr = (restaurant.address or "").strip()
    urls = restaurant.reference_urls or []

    # 1) Existence checks (Critical)
    evaluator.add_custom_node(
        result=bool(name),
        id=f"Restaurant_Name_{idx}",
        desc="Restaurant name is provided",
        parent=rest_node,
        critical=True
    )

    evaluator.add_custom_node(
        result=bool(addr),
        id=f"Restaurant_Address_{idx}",
        desc="Restaurant address is provided",
        parent=rest_node,
        critical=True
    )

    evaluator.add_custom_node(
        result=bool(restaurant.requirement_summary and restaurant.requirement_summary.strip()),
        id=f"Dietary_Requirements_Description_{idx}",
        desc="A brief description of how the restaurant meets the dietary requirements is provided",
        parent=rest_node,
        critical=True
    )

    evaluator.add_custom_node(
        result=bool(urls),
        id=f"Reference_URL_{idx}",
        desc="A reference URL that verifies the restaurant's details is provided",
        parent=rest_node,
        critical=True
    )

    # 2) Chicago location (Critical, verify via URLs)
    chicago_node = evaluator.add_leaf(
        id=f"Chicago_Location_{idx}",
        desc="Restaurant is located in Chicago, Illinois",
        parent=rest_node,
        critical=True
    )
    chicago_claim = f"'{name}' is located in Chicago, Illinois. The address listed is '{addr}'."
    await evaluator.verify(
        claim=chicago_claim,
        node=chicago_node,
        sources=urls,
        additional_instruction=(
            "Verify that the restaurant is in Chicago, IL. Address lines showing 'Chicago, IL' or neighborhoods within Chicago "
            "count as being in Chicago. Any Chicago ZIP (e.g., 606xx) or explicit city mention is acceptable."
        ),
    )

    # 3) Gluten-free safety for celiac (Critical, verify via URLs)
    gf_node = evaluator.add_leaf(
        id=f"Gluten_Free_Safety_{idx}",
        desc="Restaurant is either a dedicated gluten-free facility OR has documented gluten-free certification/procedures for celiac safety",
        parent=rest_node,
        critical=True
    )
    gf_claim = (
        f"'{name}' is celiac-safe: either 100% dedicated gluten-free OR has documented gluten-free certification/procedures "
        f"(such as dedicated GF kitchen/space, cross-contamination controls, celiac training, or formal certification). "
        f"Answer evidence: '{restaurant.gf_safety or ''}'."
    )
    await evaluator.verify(
        claim=gf_claim,
        node=gf_node,
        sources=urls,
        additional_instruction=(
            "Confirm explicit evidence beyond merely 'gluten-free options'. Look for phrases like '100% gluten-free', "
            "'certified gluten-free', 'dedicated GF kitchen', 'celiac-safe protocols', or clear cross-contamination controls."
        ),
    )

    # 4) Vegan menu options (Critical, verify via URLs)
    vegan_node = evaluator.add_leaf(
        id=f"Vegan_Menu_Options_{idx}",
        desc="Restaurant offers substantial vegan menu options including multiple vegan entrees (not just salads)",
        parent=rest_node,
        critical=True
    )
    vegan_claim = (
        f"'{name}' offers substantial vegan options with multiple vegan entrees beyond salads. "
        f"Answer evidence: '{restaurant.vegan_options or ''}'."
    )
    await evaluator.verify(
        claim=vegan_claim,
        node=vegan_node,
        sources=urls,
        additional_instruction=(
            "Look for menu or About pages indicating multiple vegan mains/entrees. Vegan salads alone are insufficient; "
            "there should be at least two substantive vegan mains (bowls, sandwiches, plates, etc.)."
        ),
    )

    # 5) Outdoor seating (Critical, verify via URLs)
    outdoor_node = evaluator.add_leaf(
        id=f"Outdoor_Seating_{idx}",
        desc="Restaurant provides outdoor seating or patio dining",
        parent=rest_node,
        critical=True
    )
    outdoor_claim = (
        f"'{name}' provides outdoor seating or patio dining. "
        f"Answer evidence: '{restaurant.outdoor_seating or ''}'."
    )
    await evaluator.verify(
        claim=outdoor_claim,
        node=outdoor_node,
        sources=urls,
        additional_instruction=(
            "Confirm outdoor seating/patio/veranda availability. Seasonal patio or rooftop counts as outdoor seating."
        ),
    )

    # 6) Group dining capacity (Critical, verify via URLs)
    group_node = evaluator.add_leaf(
        id=f"Group_Dining_Capacity_{idx}",
        desc="Restaurant can accommodate groups of at least 8-10 people",
        parent=rest_node,
        critical=True
    )
    group_claim = (
        f"'{name}' can accommodate group dining for at least 8–10 people. "
        f"Answer evidence: '{restaurant.group_capacity or ''}'."
    )
    await evaluator.verify(
        claim=group_claim,
        node=group_node,
        sources=urls,
        additional_instruction=(
            "Look for statements about large party reservations, private/group dining, banquet menus, or explicit capacity. "
            "If the page indicates accepting large parties, private rooms, or tables/seating for 8+, it qualifies."
        ),
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
    Evaluate an answer for the Chicago group dining task.
    """
    # Initialize evaluator with root parallel node
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,
        agent_name=agent_name,
        answer_name=answer_name,
        client=client,
        task_description="Find two Chicago restaurants that meet all specified dietary safety and dining requirements",
        answer=answer,
        global_cache=cache,
        global_semaphore=semaphore,
        logger=logger,
        default_model=model,
    )

    # Extract restaurants
    extracted = await evaluator.extract(
        prompt=prompt_extract_restaurants(),
        template_class=RestaurantsExtraction,
        extraction_name="restaurants_extraction",
    )

    # Keep only the first two restaurants; pad if fewer
    restaurants: List[RestaurantItem] = list(extracted.restaurants[:2])
    while len(restaurants) < 2:
        restaurants.append(RestaurantItem())

    # Build verification subtrees for the two restaurants
    for i in range(2):
        await verify_restaurant(
            evaluator=evaluator,
            parent_node=root,
            restaurant=restaurants[i],
            index=i,
        )

    # Optional: add custom info about how many restaurants were detected
    evaluator.add_custom_info(
        info={"extracted_count": len(extracted.restaurants), "used_count": 2},
        info_type="extraction_stats",
    )

    # Return structured result
    return evaluator.get_summary()