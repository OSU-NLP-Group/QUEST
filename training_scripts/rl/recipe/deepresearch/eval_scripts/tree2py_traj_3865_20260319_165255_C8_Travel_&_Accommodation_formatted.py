import asyncio
import logging
from typing import List, Optional, Dict, Any

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "cancun_hotel_zone_all_inclusive_family"
TASK_DESCRIPTION = """
Identify three all-inclusive resorts located in Cancun's Hotel Zone that meet the following requirements for a multi-generational family vacation:

Mandatory Requirements:
- Located in Cancun Hotel Zone (Zona Hotelera)
- Offers all-inclusive vacation packages
- Provides direct beach access
- Has at least 2 swimming pools
- Includes spa facilities on property
- Offers supervised kids club for children
- Features at least 3 on-site restaurants
- Provides family suites or rooms that accommodate 4 or more guests

Preferred Amenities (not mandatory but beneficial):
- 24-hour room service
- Fitness center
- Water sports activities
- Live entertainment
- WiFi access
- Multiple bars or lounges
- Water park, slides, or splash pad facilities

For each resort, provide:
1. The resort name
2. A reference URL confirming the resort's location and amenities
3. Verification that all mandatory requirements are met
"""


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class ResortCandidate(BaseModel):
    name: Optional[str] = None
    reference_urls: List[str] = Field(default_factory=list)


class ResortsExtraction(BaseModel):
    resorts: List[ResortCandidate] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_resorts() -> str:
    return """
    Extract up to the first five resort entries mentioned in the answer. For each resort, extract:
    - name: the resort name as stated in the answer (string; if missing, set to null)
    - reference_urls: an array of all URLs (HTTP/HTTPS) explicitly provided in the answer that refer to that resort (these can be official sites, hotel pages, or reputable travel pages). 
      If a URL is missing a protocol, prepend http://. If no URLs are provided for a resort, return an empty array.

    Preserve the original order that the resorts appear in the answer text. Do not invent any URLs.
    Return the results under a top-level field "resorts".
    """


# --------------------------------------------------------------------------- #
# Helper strings for additional verification instructions                     #
# --------------------------------------------------------------------------- #
ADD_INS_LOCATION = (
    "Confirm that the page clearly states the resort is in the Cancun Hotel Zone (Zona Hotelera). "
    "Accept synonyms like 'Hotel Zone', 'Zona Hotelera', or references to Blvd. Kukulcán (km markers) as proof of Hotel Zone. "
    "Ensure the page is about the specified resort."
)

ADD_INS_ALL_INCLUSIVE = (
    "Look for the phrases 'all inclusive', 'all‑inclusive', 'all inclusive plan', 'AI plan', "
    "'unlimited luxury', or equivalent indicating meals and drinks included."
)

ADD_INS_BEACH_ACCESS = (
    "Accept 'beachfront', 'on the beach', 'direct beach access', or 'private beach' as confirmation. "
    "A shuttle to a distant beach or 'near the beach' without direct access does not satisfy."
)

ADD_INS_MULTIPLE_POOLS = (
    "Confirm that there are at least two swimming pools. "
    "Phrases like 'two pools', 'multiple pools', 'pool complex', 'main pool and kids pool' count. "
    "Private plunge pools in individual suites do not alone satisfy unless there are 2+ shared pools."
)

ADD_INS_SPA = (
    "Look for 'spa', 'wellness center', 'spa services', 'hydrotherapy', or similar on-site spa facilities."
)

ADD_INS_KIDS_CLUB = (
    "Specifically require a supervised kids club/children's club (e.g., 'Kids Club', 'Explorer’s Club'). "
    "A playground or babysitting alone without a supervised club is insufficient."
)

ADD_INS_RESTAURANTS = (
    "Count only restaurants/dining venues. Buffet and à la carte restaurants count; bars alone do not. "
    "Confirm that there are at least 3 on-site restaurants."
)

ADD_INS_FAMILY_ROOMS = (
    "Confirm that the resort offers room or suite types that accommodate 4 or more guests. "
    "Phrases like 'sleeps up to 4', 'maximum occupancy 4', or '2 adults + 2 children' qualify. "
    "Connecting rooms alone do not qualify unless a single room type accommodates 4+."
)

# Preferred amenities
ADD_INS_ROOM_SERVICE = "Look for '24‑hour room service' or 'in‑room dining available 24 hours'."
ADD_INS_FITNESS = "Look for 'fitness center', 'gym', 'fitness room'."
ADD_INS_WATER_SPORTS = "Look for non‑motorized or motorized water sports offered by the resort (e.g., kayaks, paddleboards)."
ADD_INS_ENTERTAINMENT = "Look for live entertainment, nightly shows, or similar entertainment programs."
ADD_INS_WIFI = "Look for Wi‑Fi or internet access for guests (in rooms or public areas)."
ADD_INS_BARS = "Confirm that the resort has multiple bars or lounges (plural). Swim‑up bars also count."
ADD_INS_WATER_PARK = "Look for a water park, water slides, or a splash/splash pad facility."


def _clean_urls(urls: List[str]) -> List[str]:
    """Filter out empty strings and normalize minimal formatting expectations."""
    return [u.strip() for u in urls if isinstance(u, str) and u.strip()]


# --------------------------------------------------------------------------- #
# Resort verification helper                                                  #
# --------------------------------------------------------------------------- #
async def verify_resort(
    evaluator: Evaluator,
    parent_node,
    resort: ResortCandidate,
    resort_index: int,
) -> None:
    """
    Build the verification subtree for a single resort and run all checks.
    """
    rid = resort_index  # 1-based
    rnode = evaluator.add_parallel(
        id=f"resort_{rid}",
        desc=f"{['First','Second','Third'][rid-1]} qualifying resort meeting all requirements",
        parent=parent_node,
        critical=False
    )

    urls = _clean_urls(resort.reference_urls)

    # -------------------- Basic Info (critical group) -------------------- #
    basic = evaluator.add_parallel(
        id=f"resort_{rid}_basic_info",
        desc="Basic resort identification and location",
        parent=rnode,
        critical=True
    )

    # Name existence (critical custom)
    evaluator.add_custom_node(
        result=bool(resort.name and resort.name.strip()),
        id=f"resort_{rid}_name",
        desc="Provide the resort name",
        parent=basic,
        critical=True
    )

    # Reference URL existence (critical custom)
    evaluator.add_custom_node(
        result=len(urls) >= 1,
        id=f"resort_{rid}_reference",
        desc="Provide valid reference URL",
        parent=basic,
        critical=True
    )

    # Location in Cancun Hotel Zone (critical, URL-verified)
    loc_leaf = evaluator.add_leaf(
        id=f"resort_{rid}_location",
        desc="Resort is located in Cancun Hotel Zone",
        parent=basic,
        critical=True
    )
    loc_claim = (
        f"The resort named '{resort.name or 'the resort'}' is located in the Cancun Hotel Zone (Zona Hotelera) in Cancun, Mexico."
    )
    await evaluator.verify(
        claim=loc_claim,
        node=loc_leaf,
        sources=urls,
        additional_instruction=ADD_INS_LOCATION
    )

    # All-inclusive offering (critical, URL-verified)
    ai_leaf = evaluator.add_leaf(
        id=f"resort_{rid}_all_inclusive",
        desc="Resort offers all-inclusive packages",
        parent=basic,
        critical=True
    )
    ai_claim = (
        f"The resort named '{resort.name or 'the resort'}' offers all-inclusive packages or an all-inclusive plan (meals and drinks included)."
    )
    await evaluator.verify(
        claim=ai_claim,
        node=ai_leaf,
        sources=urls,
        additional_instruction=ADD_INS_ALL_INCLUSIVE
    )

    # -------------------- Mandatory amenities (critical) ----------------- #
    # Prepare leaf nodes
    beach_leaf = evaluator.add_leaf(
        id=f"resort_{rid}_beach_access",
        desc="Resort provides direct beach access",
        parent=rnode,
        critical=True
    )
    pools_leaf = evaluator.add_leaf(
        id=f"resort_{rid}_multiple_pools",
        desc="Resort has 2 or more swimming pools",
        parent=rnode,
        critical=True
    )
    spa_leaf = evaluator.add_leaf(
        id=f"resort_{rid}_spa",
        desc="Resort includes spa facilities",
        parent=rnode,
        critical=True
    )
    kids_leaf = evaluator.add_leaf(
        id=f"resort_{rid}_kids_club",
        desc="Resort offers supervised kids club",
        parent=rnode,
        critical=True
    )
    rest_leaf = evaluator.add_leaf(
        id=f"resort_{rid}_restaurants",
        desc="Resort has 3 or more on-site restaurants",
        parent=rnode,
        critical=True
    )
    family_leaf = evaluator.add_leaf(
        id=f"resort_{rid}_family_rooms",
        desc="Resort offers family suites or rooms accommodating 4+ guests",
        parent=rnode,
        critical=True
    )

    # Build claims
    beach_claim = f"The webpage confirms that '{resort.name or 'the resort'}' has direct beach access (is beachfront or on the beach)."
    pools_claim = f"The webpage confirms that '{resort.name or 'the resort'}' has at least two swimming pools."
    spa_claim = f"The webpage confirms that '{resort.name or 'the resort'}' has on-site spa facilities."
    kids_claim = f"The webpage confirms that '{resort.name or 'the resort'}' offers a supervised kids club or children's club."
    restaurants_claim = f"The webpage confirms that '{resort.name or 'the resort'}' has at least three on-site restaurants."
    family_claim = f"The webpage confirms that '{resort.name or 'the resort'}' offers family suites or rooms that accommodate four or more guests."

    # Batch verify mandatory amenities
    await evaluator.batch_verify([
        (beach_claim, urls, beach_leaf, ADD_INS_BEACH_ACCESS),
        (pools_claim, urls, pools_leaf, ADD_INS_MULTIPLE_POOLS),
        (spa_claim, urls, spa_leaf, ADD_INS_SPA),
        (kids_claim, urls, kids_leaf, ADD_INS_KIDS_CLUB),
        (restaurants_claim, urls, rest_leaf, ADD_INS_RESTAURANTS),
        (family_claim, urls, family_leaf, ADD_INS_FAMILY_ROOMS),
    ])

    # -------------------- Preferred (non-critical) ----------------------- #
    room_service_leaf = evaluator.add_leaf(
        id=f"resort_{rid}_room_service",
        desc="Resort provides 24-hour room service",
        parent=rnode,
        critical=False
    )
    fitness_leaf = evaluator.add_leaf(
        id=f"resort_{rid}_fitness",
        desc="Resort includes fitness center",
        parent=rnode,
        critical=False
    )
    water_sports_leaf = evaluator.add_leaf(
        id=f"resort_{rid}_water_sports",
        desc="Resort offers water sports activities",
        parent=rnode,
        critical=False
    )
    entertainment_leaf = evaluator.add_leaf(
        id=f"resort_{rid}_entertainment",
        desc="Resort provides live entertainment",
        parent=rnode,
        critical=False
    )
    wifi_leaf = evaluator.add_leaf(
        id=f"resort_{rid}_wifi",
        desc="Resort offers WiFi access",
        parent=rnode,
        critical=False
    )
    bars_leaf = evaluator.add_leaf(
        id=f"resort_{rid}_bars",
        desc="Resort has multiple bars or lounges",
        parent=rnode,
        critical=False
    )
    water_park_leaf = evaluator.add_leaf(
        id=f"resort_{rid}_water_park",
        desc="Resort features water park, slides, or splash pad",
        parent=rnode,
        critical=False
    )

    room_service_claim = f"The webpage confirms that '{resort.name or 'the resort'}' provides 24-hour room service or in-room dining."
    fitness_claim = f"The webpage confirms that '{resort.name or 'the resort'}' has a fitness center or gym."
    water_sports_claim = f"The webpage confirms that '{resort.name or 'the resort'}' offers water sports activities."
    entertainment_claim = f"The webpage confirms that '{resort.name or 'the resort'}' provides live entertainment or nightly shows."
    wifi_claim = f"The webpage confirms that '{resort.name or 'the resort'}' offers Wi‑Fi access for guests."
    bars_claim = f"The webpage confirms that '{resort.name or 'the resort'}' has multiple bars or lounges."
    water_park_claim = f"The webpage confirms that '{resort.name or 'the resort'}' features a water park, water slides, or a splash pad."

    await evaluator.batch_verify([
        (room_service_claim, urls, room_service_leaf, ADD_INS_ROOM_SERVICE),
        (fitness_claim, urls, fitness_leaf, ADD_INS_FITNESS),
        (water_sports_claim, urls, water_sports_leaf, ADD_INS_WATER_SPORTS),
        (entertainment_claim, urls, entertainment_leaf, ADD_INS_ENTERTAINMENT),
        (wifi_claim, urls, wifi_leaf, ADD_INS_WIFI),
        (bars_claim, urls, bars_leaf, ADD_INS_BARS),
        (water_park_claim, urls, water_park_leaf, ADD_INS_WATER_PARK),
    ])


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
    model: str = "o4-mini"
) -> Dict:
    """
    Evaluate an answer for the Cancun Hotel Zone all-inclusive family resorts task.
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

    # Extract resorts list from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_resorts(),
        template_class=ResortsExtraction,
        extraction_name="resort_list",
    )

    resorts = list(extracted.resorts or [])
    # Keep only the first three; pad with empty items if fewer provided
    resorts = resorts[:3]
    while len(resorts) < 3:
        resorts.append(ResortCandidate())

    # Build and verify tree per resort (1..3)
    for idx in range(1, 4):
        await verify_resort(
            evaluator=evaluator,
            parent_node=root,
            resort=resorts[idx - 1],
            resort_index=idx
        )

    return evaluator.get_summary()