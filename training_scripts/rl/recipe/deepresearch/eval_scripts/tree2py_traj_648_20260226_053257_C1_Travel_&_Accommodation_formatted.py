import asyncio
import logging
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys

# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "universal_orlando_premier_express_2026"
TASK_DESCRIPTION = """
Which three Universal Orlando Resort hotels include complimentary Universal Express Unlimited passes for guests staying at the hotel in 2026?
"""

# Ground truth set of qualifying hotels (Premier category with Express Unlimited benefit)
EXPECTED_HOTELS = [
    "Loews Portofino Bay Hotel",
    "Hard Rock Hotel",
    "Loews Royal Pacific Resort",
]


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class HotelItem(BaseModel):
    name: Optional[str] = None
    urls: List[str] = Field(default_factory=list)


class HotelsExtraction(BaseModel):
    hotels: List[HotelItem] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_hotels() -> str:
    return """
    Extract the hotels that the answer claims include complimentary “Universal Express Unlimited” ride access with a Universal Orlando Resort hotel stay (for 2026 stays). 
    Return up to the first 5 such hotels mentioned in the answer, in the order they appear.

    For each hotel, extract:
    - name: the hotel name exactly as written in the answer (do not normalize).
    - urls: a list of URL(s) that the answer cites for this hotel or for the Express Unlimited benefit. 
      If the answer provides a general sources section that applies to the benefit (not tied to a specific hotel), 
      include those URLs in each hotel's urls field as well since they are cited as supporting evidence for the claim.

    Rules:
    - Do not invent any hotel names or URLs not present in the answer.
    - If no URL is cited at all in the answer for a hotel/benefit, return an empty list for urls.
    - If a hotel is mentioned but not clearly tied to the Express Unlimited benefit, exclude it.
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def pick_top_unique_hotels(items: List[HotelItem], k: int = 3) -> List[HotelItem]:
    seen = set()
    unique: List[HotelItem] = []
    for it in items:
        if not it or not it.name or not it.name.strip():
            continue
        key = it.name.strip().lower()
        if key in seen:
            continue
        seen.add(key)
        unique.append(it)
        if len(unique) >= k:
            break
    # Pad to k with empty items if needed
    while len(unique) < k:
        unique.append(HotelItem())
    return unique


def hotel_set_str() -> str:
    return "; ".join(EXPECTED_HOTELS)


# --------------------------------------------------------------------------- #
# Verification logic per hotel                                                #
# --------------------------------------------------------------------------- #
async def verify_one_hotel(
    evaluator: Evaluator,
    parent_node,
    hotel: HotelItem,
    index_zero_based: int,
    prior_hotels: List[HotelItem],
    prior_exist_nodes: List,  # List[VerificationNode]
) -> None:
    i = index_zero_based + 1
    # Create a container node corresponding to rubric child
    if i == 1:
        desc = "One of the three hotels that includes complimentary Universal Express Unlimited is correctly identified (Loews Portofino Bay Hotel, Hard Rock Hotel, or Loews Royal Pacific Resort)"
    elif i == 2:
        desc = "A second distinct hotel from the three hotels that includes complimentary Universal Express Unlimited is correctly identified (Loews Portofino Bay Hotel, Hard Rock Hotel, or Loews Royal Pacific Resort)"
    else:
        desc = "The third distinct hotel from the three hotels that includes complimentary Universal Express Unlimited is correctly identified (Loews Portofino Bay Hotel, Hard Rock Hotel, or Loews Royal Pacific Resort)"

    container = evaluator.add_parallel(
        id=f"Hotel_{i}_Identified",
        desc=desc,
        parent=parent_node,
        critical=False,  # non-critical group to allow partial credit
    )

    # Existence check (critical within this group)
    exists = hotel is not None and hotel.name is not None and hotel.name.strip() != ""
    exist_node = evaluator.add_custom_node(
        result=exists,
        id=f"hotel_{i}_exists",
        desc=f"Hotel #{i} name is provided in the answer",
        parent=container,
        critical=True,
    )

    # Membership check: is this name one of the 3 expected hotels?
    membership_leaf = evaluator.add_leaf(
        id=f"hotel_{i}_membership",
        desc=f"Hotel #{i} is one of the three qualifying Premier hotels (Portofino Bay, Hard Rock, or Royal Pacific)",
        parent=container,
        critical=False,  # soft to allow partial credit even if sources missing
    )

    hotel_name = hotel.name or ""
    membership_claim = (
        f"The hotel named '{hotel_name}' refers to one of the following three Universal Orlando Premier hotels that "
        f"include complimentary Universal Express Unlimited with a hotel stay: {hotel_set_str()}."
    )
    await evaluator.verify(
        claim=membership_claim,
        node=membership_leaf,
        additional_instruction=(
            "Treat reasonable naming variants as equivalent (e.g., adding/removing “Universal’s”, "
            "“at Universal Orlando”, minor punctuation, or abbreviations). Focus on whether the given hotel "
            "name denotes one of the three listed properties."
        ),
    )

    # Distinctness checks against prior hotels (critical within this group)
    # These depend on the current existence node plus the prior existence nodes
    for j, prev_hotel in enumerate(prior_hotels):
        prev_name = prev_hotel.name or ""
        distinct_leaf = evaluator.add_leaf(
            id=f"hotel_{i}_distinct_from_{j+1}",
            desc=f"Hotel #{i} is distinct from Hotel #{j+1}",
            parent=container,
            critical=True,
        )
        distinct_claim = (
            f"The hotel '{hotel_name}' is a different property than '{prev_name}'. "
            f"Do NOT count alternate phrasings or abbreviations of the same property as distinct."
        )
        await evaluator.verify(
            claim=distinct_claim,
            node=distinct_leaf,
            additional_instruction=(
                "Consider equivalences such as adding/removing 'Universal’s', 'at Universal Orlando', minor punctuation, "
                "or obvious abbreviations as the same property. Only judge 'distinct' if they clearly refer to different hotels."
            ),
            extra_prerequisites=[exist_node] + prior_exist_nodes,  # ensure both exist before comparing
        )

    # Source support for the Express Unlimited benefit (soft)
    # If sources available, verify by URLs; otherwise, add a failed soft node to reflect missing citations.
    if hotel.urls:
        source_leaf = evaluator.add_leaf(
            id=f"hotel_{i}_source_supports_express",
            desc=f"Sources support that Hotel #{i} includes complimentary Universal Express Unlimited with a stay",
            parent=container,
            critical=False,
        )
        support_claim = (
            f"The webpage confirms that guests staying at {hotel_name} receive complimentary Universal Express Unlimited "
            f"ride access (with valid theme park admission)."
        )
        await evaluator.verify(
            claim=support_claim,
            node=source_leaf,
            sources=hotel.urls,
            additional_instruction=(
                "Look for clear mention of 'Universal Express Unlimited' being included/complimentary for hotel guests "
                "when staying at this property. Minor wording differences are acceptable (e.g., 'included with stay', "
                "'complimentary Express Unlimited')."
            ),
        )
    else:
        # No sources provided in the answer – treat as a soft failure to reflect missing grounding
        evaluator.add_custom_node(
            result=False,
            id=f"hotel_{i}_source_supports_express",
            desc=f"No source URL provided in the answer to support that Hotel #{i} includes Express Unlimited",
            parent=container,
            critical=False,
        )

    return


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
    # Initialize evaluator and root
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

    # Add ground truth reference
    evaluator.add_ground_truth(
        {
            "expected_hotels": EXPECTED_HOTELS,
            "task": "Identify the three Universal Orlando Premier hotels that include complimentary Universal Express Unlimited with a stay.",
        },
        gt_type="ground_truth",
    )

    # Extract hotels and associated URLs from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_hotels(),
        template_class=HotelsExtraction,
        extraction_name="hotels_extraction",
    )

    # Prepare up to 3 unique hotel entries
    top3 = pick_top_unique_hotels(extracted.hotels, k=3)

    # Build verification subtrees
    prior_hotels: List[HotelItem] = []
    prior_exist_nodes = []
    for idx, hotel in enumerate(top3):
        # For each rubric child, create a structured verification group
        # We also capture the existence node to use as a precondition for later distinctness checks
        # To capture the existence node, we first create the group with a temporary call in verify_one_hotel, but
        # we need the actual node returned. We'll re-create existence node here in a controlled way:
        # Instead, we implement verify_one_hotel to create existence node and return nothing,
        # but we still need references to prior existence nodes. To handle that, we'll create the container and existence node here,
        # then pass them into the verifier helper via closures? 
        # Simpler: do it in two phases – create container + existence node here, then call a small inline function for the rest.

        # Create container for this hotel
        if idx == 0:
            desc = "One of the three hotels that includes complimentary Universal Express Unlimited is correctly identified (Loews Portofino Bay Hotel, Hard Rock Hotel, or Loews Royal Pacific Resort)"
        elif idx == 1:
            desc = "A second distinct hotel from the three hotels that includes complimentary Universal Express Unlimited is correctly identified (Loews Portofino Bay Hotel, Hard Rock Hotel, or Loews Royal Pacific Resort)"
        else:
            desc = "The third distinct hotel from the three hotels that includes complimentary Universal Express Unlimited is correctly identified (Loews Portofino Bay Hotel, Hard Rock Hotel, or Loews Royal Pacific Resort)"

        container = evaluator.add_parallel(
            id=f"Hotel_{idx+1}_Identified",
            desc=desc,
            parent=root,
            critical=False,
        )

        # Existence node (critical)
        exists = hotel is not None and hotel.name is not None and hotel.name.strip() != ""
        exist_node = evaluator.add_custom_node(
            result=exists,
            id=f"hotel_{idx+1}_exists",
            desc=f"Hotel #{idx+1} name is provided in the answer",
            parent=container,
            critical=True,
        )

        # Membership leaf (soft)
        membership_leaf = evaluator.add_leaf(
            id=f"hotel_{idx+1}_membership",
            desc=f"Hotel #{idx+1} is one of the three qualifying Premier hotels (Portofino Bay, Hard Rock, or Royal Pacific)",
            parent=container,
            critical=False,
        )
        hotel_name = hotel.name or ""
        membership_claim = (
            f"The hotel named '{hotel_name}' refers to one of the following three Universal Orlando Premier hotels that "
            f"include complimentary Universal Express Unlimited with a hotel stay: {hotel_set_str()}."
        )
        await evaluator.verify(
            claim=membership_claim,
            node=membership_leaf,
            additional_instruction=(
                "Treat reasonable naming variants as equivalent (e.g., adding/removing “Universal’s”, "
                "“at Universal Orlando”, minor punctuation, or abbreviations). Focus on whether the given hotel "
                "name denotes one of the three listed properties."
            ),
        )

        # Distinctness leaves (critical)
        for j, prev_h in enumerate(prior_hotels):
            prev_name = prev_h.name or ""
            distinct_leaf = evaluator.add_leaf(
                id=f"hotel_{idx+1}_distinct_from_{j+1}",
                desc=f"Hotel #{idx+1} is distinct from Hotel #{j+1}",
                parent=container,
                critical=True,
            )
            distinct_claim = (
                f"The hotel '{hotel_name}' is a different property than '{prev_name}'. "
                f"Do NOT count alternate phrasings or abbreviations of the same property as distinct."
            )
            await evaluator.verify(
                claim=distinct_claim,
                node=distinct_leaf,
                additional_instruction=(
                    "Consider equivalences such as adding/removing 'Universal’s', 'at Universal Orlando', minor punctuation, "
                    "or obvious abbreviations as the same property. Only judge 'distinct' if they clearly refer to different hotels."
                ),
                extra_prerequisites=[exist_node] + prior_exist_nodes,
            )

        # Source support (soft)
        if hotel.urls:
            source_leaf = evaluator.add_leaf(
                id=f"hotel_{idx+1}_source_supports_express",
                desc=f"Sources support that Hotel #{idx+1} includes complimentary Universal Express Unlimited with a stay",
                parent=container,
                critical=False,
            )
            support_claim = (
                f"The webpage confirms that guests staying at {hotel_name} receive complimentary Universal Express Unlimited "
                f"ride access (with valid theme park admission)."
            )
            await evaluator.verify(
                claim=support_claim,
                node=source_leaf,
                sources=hotel.urls,
                additional_instruction=(
                    "Look for clear mention of 'Universal Express Unlimited' being included/complimentary for hotel guests "
                    "when staying at this property. Minor wording differences are acceptable (e.g., 'included with stay', "
                    "'complimentary Express Unlimited')."
                ),
            )
        else:
            evaluator.add_custom_node(
                result=False,
                id=f"hotel_{idx+1}_source_supports_express",
                desc=f"No source URL provided in the answer to support that Hotel #{idx+1} includes Express Unlimited",
                parent=container,
                critical=False,
            )

        # Append for next rounds' distinctness checks
        prior_hotels.append(hotel)
        prior_exist_nodes.append(exist_node)

    # Return summary
    return evaluator.get_summary()