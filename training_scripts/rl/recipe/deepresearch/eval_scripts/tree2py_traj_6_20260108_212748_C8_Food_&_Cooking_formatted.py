import asyncio
import logging
from typing import List, Optional, Dict, Any

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys

# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "jbf_2024_best_chef_restaurants"
TASK_DESCRIPTION = (
    "Identify three restaurants in the United States, where each restaurant is operated by a chef "
    "who won the 2024 James Beard Award for Best Chef in one of the following regions: California, "
    "Southwest (AZ, NM, NV, OK), or Texas. Each restaurant must be located in a different U.S. state. "
    "For each of the three restaurants, provide the following information: "
    "(1) The chef's full name, (2) The restaurant's name, (3) The city where the restaurant is located, "
    "(4) The U.S. state where the restaurant is located, (5) The primary cuisine type or culinary style "
    "of the restaurant, (6) A reference URL from the James Beard Foundation's official announcement or the "
    "restaurant's official website verifying this information. The chefs must have been announced as winners "
    "(not semifinalists or finalists) at the 2024 James Beard Awards ceremony held on June 10, 2024."
)

ALLOWED_REGIONS = ["California", "Southwest", "Texas"]
CEREMONY_DATE = "June 10, 2024"


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class RestaurantItem(BaseModel):
    chef_full_name: Optional[str] = None
    restaurant_name: Optional[str] = None
    city: Optional[str] = None
    state: Optional[str] = None
    cuisine_type: Optional[str] = None
    verification_urls: List[str] = Field(default_factory=list)
    # Optional helper field if the answer mentions the specific region label
    award_region_mentioned: Optional[str] = None


class RestaurantsExtraction(BaseModel):
    restaurants: List[RestaurantItem] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_restaurants() -> str:
    return """
    Extract up to three restaurant entries as presented in the answer text. For each entry, extract the following fields:
    - chef_full_name: The chef's full name as stated in the answer.
    - restaurant_name: The restaurant's name as stated in the answer.
    - city: The city where the restaurant is located.
    - state: The U.S. state where the restaurant is located. Accept either full state name or common postal abbreviation.
    - cuisine_type: The primary cuisine type or culinary style of the restaurant (as described in the answer).
    - verification_urls: A list of explicit URLs mentioned in the answer that serve as references. Accept URLs from:
        • The James Beard Foundation's official winners announcement pages, and/or
        • The restaurant's official website.
      Ignore URLs from unrelated third-party pages for this extraction.
    - award_region_mentioned: If the answer text explicitly mentions the James Beard Best Chef region (e.g., "California", "Southwest", or "Texas") for this chef, extract that region text; otherwise null.

    Rules:
    - Extract only what is explicitly stated in the answer; do not fabricate missing fields.
    - If the answer lists more than three qualifying entries, extract them all; the evaluator will consider the first three.
    - If any field is missing for an entry, set it to null (or an empty list for verification_urls).
    - For URLs, include the full URL with protocol if available. If protocol is missing, prepend http://.
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def nonempty(s: Optional[str]) -> bool:
    return bool(s and str(s).strip())


def normalize_state(s: Optional[str]) -> Optional[str]:
    if s is None:
        return None
    return s.strip().lower()


# --------------------------------------------------------------------------- #
# Verification functions                                                      #
# --------------------------------------------------------------------------- #
async def verify_restaurant(
    evaluator: Evaluator,
    parent_node,
    item: RestaurantItem,
    index: int,
) -> None:
    """
    Build the verification subtree for one restaurant item.
    This follows the rubric's Restaurant_i subtree with fine-grained leaf checks.
    """
    ridx = index + 1
    rest_node = evaluator.add_parallel(
        id=f"Restaurant_{ridx}",
        desc=f"{['First','Second','Third'][index]} qualifying restaurant and its required fields.",
        parent=parent_node,
        critical=False
    )

    # Existence checks (critical under restaurant node)
    evaluator.add_custom_node(
        result=nonempty(item.chef_full_name),
        id=f"R{ridx}_Chef_Full_Name",
        desc="Provides the chef's full name.",
        parent=rest_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=nonempty(item.restaurant_name),
        id=f"R{ridx}_Restaurant_Name",
        desc="Provides the restaurant's name.",
        parent=rest_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=nonempty(item.city),
        id=f"R{ridx}_City",
        desc="Provides the city where the restaurant is located.",
        parent=rest_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=nonempty(item.state),
        id=f"R{ridx}_State",
        desc="Provides the U.S. state where the restaurant is located.",
        parent=rest_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=nonempty(item.cuisine_type),
        id=f"R{ridx}_Cuisine_Type",
        desc="Provides the primary cuisine type or culinary style of the restaurant.",
        parent=rest_node,
        critical=True
    )

    # Verification URLs existence (critical requirement per rubric)
    urls_exist = bool(item.verification_urls and len(item.verification_urls) > 0)
    evaluator.add_custom_node(
        result=urls_exist,
        id=f"R{ridx}_Verification_URLs",
        desc=("Provides at least one URL from (a) the James Beard Foundation official winners announcement and/or "
              "(b) the restaurant's official website; collectively, the URL(s) corroborate the chef's winner status/region "
              "and the listed restaurant details (name, city/state, cuisine/style)."),
        parent=rest_node,
        critical=True
    )

    # Award eligibility verification (critical)
    # Verify the person is an announced winner (not semi/finalist) at 2024 ceremony in ALLOWED_REGIONS.
    award_node = evaluator.add_leaf(
        id=f"R{ridx}_Award_Eligibility",
        desc=("Chef is an announced WINNER (not semifinalist/finalist) of the 2024 James Beard Award for Best Chef "
              "in one of the allowed regions (California, Southwest, or Texas), at the June 10, 2024 awards ceremony."),
        parent=rest_node,
        critical=True
    )
    chef_name = item.chef_full_name or ""
    region_hint = item.award_region_mentioned or "one of: California, Southwest, or Texas"
    award_claim = (
        f"{chef_name} is an announced WINNER (not a semifinalist or finalist) of the 2024 James Beard Award for "
        f"Best Chef in {region_hint}. The winners were announced at the ceremony on {CEREMONY_DATE}."
    )
    await evaluator.verify(
        claim=award_claim,
        node=award_node,
        sources=item.verification_urls,
        additional_instruction=(
            "Prefer the official James Beard Foundation winners announcement page for verification, but the restaurant's "
            "official website is acceptable only if it clearly states that the chef won the 2024 James Beard Best Chef award "
            f"in one of these regions: {', '.join(ALLOWED_REGIONS)}. If the page indicates only semifinalist or finalist status, "
            "or a different year/award, the claim should be marked unsupported."
        )
    )

    # Restaurant details verification (critical)
    # Verify that the provided URLs support the restaurant details.
    details_node = evaluator.add_leaf(
        id=f"R{ridx}_Details_Supported",
        desc="Provided URL(s) corroborate restaurant details (name, city/state, cuisine/style).",
        parent=rest_node,
        critical=True
    )
    rname = item.restaurant_name or ""
    city = item.city or ""
    state = item.state or ""
    cuisine = item.cuisine_type or ""
    details_claim = (
        f"The restaurant named '{rname}' is located in {city}, {state}, United States, and its primary cuisine or culinary "
        f"style is described as '{cuisine}'."
    )
    await evaluator.verify(
        claim=details_claim,
        node=details_node,
        sources=item.verification_urls,
        additional_instruction=(
            "Accept verification from either the restaurant's official site or the James Beard winners announcement if it "
            "explicitly lists the restaurant and location. For cuisine or culinary style, the restaurant's official website "
            "is typically the authoritative source. Minor phrasing differences are acceptable if they clearly refer to the "
            "same restaurant and cuisine concept."
        )
    )


def compute_distinct_states(items: List[RestaurantItem]) -> bool:
    """
    Returns True iff all three states are present and pairwise distinct.
    """
    states = [normalize_state(it.state) for it in items]
    if any(s is None or s.strip() == "" for s in states):
        return False
    return len(set(states)) == len(states) == 3


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
    Evaluate an answer for the James Beard 2024 Best Chef restaurants task.
    """
    # Initialize evaluator (root as non-critical to allow partial credit across restaurants; gate by critical children)
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

    # Extraction
    extraction: RestaurantsExtraction = await evaluator.extract(
        prompt=prompt_extract_restaurants(),
        template_class=RestaurantsExtraction,
        extraction_name="restaurants_extraction"
    )

    # Normalize to exactly 3 items: take first 3 if more; pad with empty items if fewer
    items: List[RestaurantItem] = list(extraction.restaurants or [])
    if len(items) > 3:
        items = items[:3]
    while len(items) < 3:
        items.append(RestaurantItem())

    # Build three restaurant verification subtrees
    for i in range(3):
        await verify_restaurant(evaluator, root, items[i], i)

    # Global distinct state constraint (critical)
    evaluator.add_custom_node(
        result=compute_distinct_states(items),
        id="Global_Distinct_States",
        desc="The three restaurants are located in three different U.S. states (no state repeated across the set).",
        parent=root,
        critical=True
    )

    # Return the evaluation summary
    return evaluator.get_summary()