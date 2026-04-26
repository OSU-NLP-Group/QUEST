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
TASK_ID = "multi_trip_accommodations"
TASK_DESCRIPTION = """I'm planning a multi-destination trip and need to book four different accommodations. Please help me find suitable options that meet the following specific requirements:

1. Charlotte Airport Layover Hotel: I have a long layover at Charlotte Douglas International Airport (CLT) and need a hotel for one night. The hotel must be located within 2 miles of the airport, provide free 24-hour shuttle service to and from the airport, and include complimentary breakfast. I'd also prefer it to have a swimming pool (indoor or outdoor).

2. Malta Luxury Hotel: For my Malta vacation, I want a five-star or luxury hotel located in Valletta or within 1 kilometer of Valletta's city center. The hotel must have on-site spa and wellness facilities. I'd also prefer rooms with Mediterranean Sea views.

3. Bali Ubud Resort: For my Bali stay, I'm looking for a resort in the Ubud area that offers rice terrace views or tropical garden views. The resort must provide villa accommodations with private pools. Having on-site spa facilities would be a bonus.

4. Disney Cruise Stateroom: I want to book a stateroom on a Disney Cruise Line ship that departs from Port Canaveral, Florida. The stateroom must be in the Verandah (balcony) category or higher, and the ship must have a passenger capacity of 4,000 or more guests. I'd like to confirm that Concierge-level staterooms are available as an option on this ship.

For each accommodation, please provide the name, a brief description of how it meets the requirements, and reference URLs supporting your findings.
"""

# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class AccommodationItem(BaseModel):
    name: Optional[str] = None
    brief_description: Optional[str] = None
    urls: List[str] = Field(default_factory=list)


class DisneyCruiseItem(BaseModel):
    ship_name: Optional[str] = None
    stateroom_category: Optional[str] = None
    urls: List[str] = Field(default_factory=list)


class TripAccommodationsExtraction(BaseModel):
    clt_hotel: Optional[AccommodationItem] = None
    malta_hotel: Optional[AccommodationItem] = None
    bali_resort: Optional[AccommodationItem] = None
    disney_cruise: Optional[DisneyCruiseItem] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_trip_accommodations() -> str:
    return """
Extract from the answer the structured information for the four requested accommodations. For each item, return exactly the fields described below. Extract only what is explicitly present in the answer text.

- clt_hotel:
  - name: the hotel name for the Charlotte airport layover (or null if missing)
  - brief_description: the user's brief justification/summary from the answer (or null)
  - urls: an array of all URLs cited for this hotel

- malta_hotel:
  - name: the hotel name for the Malta stay (or null)
  - brief_description: brief justification/summary (or null)
  - urls: an array of all URLs cited for this hotel

- bali_resort:
  - name: the resort name for the Ubud stay (or null)
  - brief_description: brief justification/summary (or null)
  - urls: an array of all URLs cited for this resort

- disney_cruise:
  - ship_name: the Disney ship name if stated (e.g., Disney Wish, Disney Dream) (or null)
  - stateroom_category: the stateroom type/category the answer claims (e.g., Verandah, Concierge) (or null)
  - urls: an array of all URLs cited for the cruise/ship/stateroom/itinerary

IMPORTANT:
- urls must be actual URLs explicitly present in the answer text (plain or markdown links). Do not invent URLs.
- If any section is missing in the answer, set the object to null.
"""


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _name_or_label(name: Optional[str], fallback: str) -> str:
    if name and name.strip():
        return name.strip()
    return fallback


def _sources_from(item_urls: Optional[List[str]]) -> List[str]:
    return item_urls if item_urls else []


# --------------------------------------------------------------------------- #
# Verification builders                                                       #
# --------------------------------------------------------------------------- #
async def build_charlotte_hotel_checks(evaluator: Evaluator, parent, item: Optional[AccommodationItem]) -> None:
    node = evaluator.add_parallel(
        id="Charlotte_Airport_Hotel",
        desc="Verify the Charlotte airport layover hotel meets all requirements",
        parent=parent,
        critical=False
    )
    hotel_name = _name_or_label(item.name if item else None, "the Charlotte layover hotel")
    sources = _sources_from(item.urls if item else None)

    # Leaves
    dist_leaf = evaluator.add_leaf(
        id="CLT_Distance_Requirement",
        desc="Hotel must be located within 2 miles of Charlotte Douglas International Airport (CLT)",
        parent=node,
        critical=True
    )
    shuttle_leaf = evaluator.add_leaf(
        id="Airport_Shuttle_Service",
        desc="Hotel must provide free 24-hour shuttle service to/from CLT airport",
        parent=node,
        critical=True
    )
    breakfast_leaf = evaluator.add_leaf(
        id="Complimentary_Breakfast",
        desc="Hotel must include free breakfast in the room rate",
        parent=node,
        critical=True
    )
    pool_leaf = evaluator.add_leaf(
        id="Swimming_Pool_Facility",
        desc="Hotel must have an indoor or outdoor swimming pool",
        parent=node,
        critical=False
    )

    claims = [
        (
            f"The hotel {hotel_name} is located within 2 miles (approximately 3.2 km) of Charlotte Douglas International Airport (CLT).",
            sources,
            dist_leaf,
            "Accept if the page explicitly states a distance ≤ 2 miles (or ≤ 3.2 km) to CLT. If the page reliably states the hotel is ~2 miles or closer, consider it supported."
        ),
        (
            f"The hotel {hotel_name} provides a free 24-hour airport shuttle to and from CLT (complimentary 24/7 airport shuttle).",
            sources,
            shuttle_leaf,
            "Look for phrases like 'free airport shuttle', 'complimentary airport shuttle', and '24-hour' or '24/7'. It must be free and run 24 hours."
        ),
        (
            f"The hotel {hotel_name} includes complimentary breakfast in the room rate.",
            sources,
            breakfast_leaf,
            "Accept 'free breakfast', 'complimentary breakfast', or similar wording that indicates breakfast is included."
        ),
        (
            f"The hotel {hotel_name} has a swimming pool on-site (indoor or outdoor).",
            sources,
            pool_leaf,
            "Look for 'pool', 'indoor pool', or 'outdoor pool' on the official/good-quality page."
        )
    ]

    await evaluator.batch_verify(claims)


async def build_malta_hotel_checks(evaluator: Evaluator, parent, item: Optional[AccommodationItem]) -> None:
    node = evaluator.add_parallel(
        id="Malta_Luxury_Hotel",
        desc="Verify the Malta vacation hotel meets all requirements",
        parent=parent,
        critical=False
    )
    hotel_name = _name_or_label(item.name if item else None, "the Malta hotel")
    sources = _sources_from(item.urls if item else None)

    five_star_leaf = evaluator.add_leaf(
        id="Five_Star_Rating",
        desc="Hotel must have a five-star rating or be classified as a luxury hotel",
        parent=node,
        critical=True
    )
    valletta_loc_leaf = evaluator.add_leaf(
        id="Valletta_Location",
        desc="Hotel must be located in Valletta or within 1 kilometer of Valletta's city center",
        parent=node,
        critical=True
    )
    spa_leaf = evaluator.add_leaf(
        id="Spa_Wellness_Center",
        desc="Hotel must have on-site spa and wellness facilities",
        parent=node,
        critical=True
    )
    sea_view_leaf = evaluator.add_leaf(
        id="Mediterranean_Views",
        desc="Hotel must offer rooms with Mediterranean Sea views",
        parent=node,
        critical=False
    )

    claims = [
        (
            f"The hotel {hotel_name} is a five-star (5-star) or explicitly described as a luxury hotel.",
            sources,
            five_star_leaf,
            "Accept explicit '5-star' or authoritative 'luxury hotel' classification (e.g., official site, credible OTA page)."
        ),
        (
            f"The hotel {hotel_name} is located in Valletta or within 1 kilometer (≈0.62 miles) of Valletta's city center.",
            sources,
            valletta_loc_leaf,
            "If the address is in Valletta, accept as meeting the ≤1 km requirement. Alternatively, accept explicit statements of ≤1 km to the Valletta city center/City Gate."
        ),
        (
            f"The hotel {hotel_name} has on-site spa and wellness facilities.",
            sources,
            spa_leaf,
            "Look for 'spa', 'wellness center', 'spa & wellness', or similar on the hotel's official/credible page indicating on-site facilities."
        ),
        (
            f"Rooms at {hotel_name} offer Mediterranean Sea views.",
            sources,
            sea_view_leaf,
            "Accept if at least some room types explicitly offer sea/harbor/Mediterranean views."
        )
    ]

    await evaluator.batch_verify(claims)


async def build_bali_resort_checks(evaluator: Evaluator, parent, item: Optional[AccommodationItem]) -> None:
    node = evaluator.add_parallel(
        id="Bali_Ubud_Resort",
        desc="Verify the Bali Ubud resort meets all requirements",
        parent=parent,
        critical=False
    )
    resort_name = _name_or_label(item.name if item else None, "the Ubud resort")
    sources = _sources_from(item.urls if item else None)

    ubud_loc_leaf = evaluator.add_leaf(
        id="Ubud_Location",
        desc="Resort must be located in the Ubud area of Bali",
        parent=node,
        critical=True
    )
    scenic_leaf = evaluator.add_leaf(
        id="Scenic_Views",
        desc="Resort must offer rice terrace views or tropical garden views",
        parent=node,
        critical=True
    )
    private_pool_leaf = evaluator.add_leaf(
        id="Private_Pool_Villa",
        desc="Resort must offer villa accommodations with private pools",
        parent=node,
        critical=True
    )
    spa_leaf = evaluator.add_leaf(
        id="Resort_Spa_Facilities",
        desc="Resort must have on-site spa facilities",
        parent=node,
        critical=False
    )

    claims = [
        (
            f"The resort {resort_name} is located in the Ubud area of Bali (Ubud town or immediate surroundings within Gianyar Regency, e.g., Sayan, Kedewatan, Tegallalang, Payangan).",
            sources,
            ubud_loc_leaf,
            "Accept addresses in Ubud or well-known Ubud environs in Gianyar Regency typically marketed as Ubud area."
        ),
        (
            f"The resort {resort_name} offers rice terrace views or lush tropical garden views.",
            sources,
            scenic_leaf,
            "Look for 'rice terrace views', 'rice paddies', 'jungle/garden views', or similar wording."
        ),
        (
            f"The resort {resort_name} offers villa accommodations that include private pools.",
            sources,
            private_pool_leaf,
            "Accept 'private pool villa', 'pool villa', or equivalent wording indicating a private pool within villa accommodation."
        ),
        (
            f"The resort {resort_name} has on-site spa facilities.",
            sources,
            spa_leaf,
            "Look for 'spa', 'wellness', or similar facilities available on the property."
        )
    ]

    await evaluator.batch_verify(claims)


async def build_disney_cruise_checks(evaluator: Evaluator, parent, item: Optional[DisneyCruiseItem]) -> None:
    node = evaluator.add_parallel(
        id="Disney_Cruise_Stateroom",
        desc="Verify the Disney Cruise Line stateroom booking meets all requirements",
        parent=parent,
        critical=False
    )
    ship_name = item.ship_name if item else None
    stateroom_category = item.stateroom_category if item else None
    sources = _sources_from(item.urls if item else None)

    port_leaf = evaluator.add_leaf(
        id="Port_Canaveral_Departure",
        desc="Cruise ship must depart from Port Canaveral, Florida",
        parent=node,
        critical=True
    )
    verandah_leaf = evaluator.add_leaf(
        id="Verandah_Stateroom",
        desc="Stateroom must be a Verandah (balcony) category or higher",
        parent=node,
        critical=True
    )
    capacity_leaf = evaluator.add_leaf(
        id="Large_Ship_Capacity",
        desc="Ship must have a passenger capacity of 4,000 or more guests",
        parent=node,
        critical=True
    )
    concierge_leaf = evaluator.add_leaf(
        id="Concierge_Available",
        desc="Ship must offer Concierge-level staterooms as an option",
        parent=node,
        critical=False
    )

    # Claims
    port_claim = (
        f"The itinerary indicates a Disney Cruise Line departure from Port Canaveral, Florida."
        if not ship_name else
        f"The Disney Cruise Line itinerary for the ship {ship_name} departs from Port Canaveral, Florida."
    )
    if stateroom_category and stateroom_category.strip():
        verandah_claim = (
            f"The selected stateroom category is '{stateroom_category}', which is a Verandah (balcony) category or higher (e.g., Verandah, Family Verandah, Concierge, or Suite)."
        )
    else:
        verandah_claim = (
            "The selected stateroom is a Verandah (balcony) category or higher (including Verandah, Concierge, or Suite)."
        )

    capacity_claim = (
        f"The Disney ship {ship_name} has a passenger capacity of 4,000 or more guests."
        if ship_name else
        "The Disney Cruise Line ship for this booking has a passenger capacity of 4,000 or more guests."
    )
    concierge_claim = (
        f"The Disney ship {ship_name} offers Concierge-level staterooms."
        if ship_name else
        "The Disney Cruise Line ship offers Concierge-level staterooms."
    )

    claims = [
        (
            port_claim,
            sources,
            port_leaf,
            "Look for 'Port Canaveral' as the embarkation/departure port on itinerary or booking pages."
        ),
        (
            verandah_claim,
            sources,
            verandah_leaf,
            "Accept synonyms 'verandah', 'veranda', or 'balcony'. Also accept 'Concierge' or 'Suite' as higher than Verandah."
        ),
        (
            capacity_claim,
            sources,
            capacity_leaf,
            "Accept official/credible sources indicating passenger capacity ≥ 4,000. Slight variations in phrasing (e.g., '4,000 passengers') are acceptable."
        ),
        (
            concierge_claim,
            sources,
            concierge_leaf,
            "Verify that the ship offers 'Concierge' staterooms/classes (e.g., Concierge Family Oceanview with Verandah)."
        )
    ]

    await evaluator.batch_verify(claims)


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

    extracted = await evaluator.extract(
        prompt=prompt_extract_trip_accommodations(),
        template_class=TripAccommodationsExtraction,
        extraction_name="trip_accommodations"
    )

    # Build checks for each accommodation group
    await build_charlotte_hotel_checks(evaluator, root, extracted.clt_hotel)
    await build_malta_hotel_checks(evaluator, root, extracted.malta_hotel)
    await build_bali_resort_checks(evaluator, root, extracted.bali_resort)
    await build_disney_cruise_checks(evaluator, root, extracted.disney_cruise)

    return evaluator.get_summary()