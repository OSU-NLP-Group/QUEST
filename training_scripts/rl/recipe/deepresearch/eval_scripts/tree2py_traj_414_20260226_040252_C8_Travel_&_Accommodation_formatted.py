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
TASK_ID = "pet_friendly_hotels_socal_4"
TASK_DESCRIPTION = (
    "I am planning a trip to Southern California with my dog and need to identify four different pet-friendly hotels "
    "where I can stay. Each hotel must meet all of the following requirements:\n\n"
    "1. Location: The hotel must be located in Los Angeles County, Orange County, or San Diego County, California.\n\n"
    "2. Pet Policy: The hotel must explicitly allow dogs and must publicly state a specific pet fee (either as a per-night "
    "charge or a per-stay charge).\n\n"
    "3. Parking: The hotel must offer on-site parking (either self-parking or valet parking) and must publicly state the "
    "daily or nightly parking fee.\n\n"
    "4. Verification: For each hotel, provide a URL to either the hotel's official website or a page on a major hotel booking "
    "platform (such as Marriott.com, Hilton.com, Booking.com, Hotels.com, Expedia, BringFido.com, etc.) that shows the hotel's information.\n\n"
    "Please identify four distinct hotels that each satisfy all four requirements listed above. For each hotel, provide the hotel name, "
    "the specific pet fee, the specific parking fee, and a reference URL."
)

ALLOWED_COUNTIES = ["Los Angeles County", "Orange County", "San Diego County"]

CITY_HINTS: Dict[str, List[str]] = {
    "Los Angeles County": [
        "Los Angeles", "Santa Monica", "Pasadena", "Long Beach", "Burbank",
        "Glendale", "West Hollywood", "Torrance", "Inglewood", "Beverly Hills",
        "Manhattan Beach", "Redondo Beach", "Culver City", "El Segundo", "Malibu"
    ],
    "Orange County": [
        "Anaheim", "Irvine", "Santa Ana", "Costa Mesa", "Newport Beach",
        "Huntington Beach", "Fullerton", "Garden Grove", "Orange", "Laguna Beach",
        "Dana Point", "Tustin", "Mission Viejo", "Lake Forest", "San Clemente"
    ],
    "San Diego County": [
        "San Diego", "La Jolla", "Carlsbad", "Chula Vista", "Oceanside",
        "Escondido", "Del Mar", "Coronado", "La Mesa", "El Cajon",
        "Encinitas", "Solana Beach", "San Marcos", "Poway", "National City"
    ],
}

MAJOR_PLATFORMS = [
    "marriott.com", "hilton.com", "hyatt.com", "ihg.com", "booking.com",
    "hotels.com", "expedia.com", "bringfido.com", "choicehotels.com", "wyndhamhotels.com",
    "bestwestern.com", "fourseasons.com", "kimptonhotels.com", "accor.com", "fairmont.com"
]


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class HotelItem(BaseModel):
    name: Optional[str] = None
    city: Optional[str] = None
    county: Optional[str] = None
    state: Optional[str] = None
    pet_fee: Optional[str] = None
    pet_fee_basis: Optional[str] = None  # e.g., "per night", "per stay", "one-time"
    parking_fee: Optional[str] = None
    parking_basis: Optional[str] = None  # e.g., "per night", "per day", "overnight"
    parking_type: Optional[str] = None   # e.g., "self-parking", "valet", "self and valet"
    url: Optional[str] = None


class HotelsExtraction(BaseModel):
    hotels: List[HotelItem] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_hotels() -> str:
    return (
        "Extract up to four hotels mentioned in the answer. For each hotel, extract the following fields strictly from the "
        "answer text (do not invent anything):\n"
        "- name: the hotel's proper name as provided\n"
        "- city: the city shown in the answer (if provided)\n"
        "- county: the county shown in the answer (if provided), e.g., 'Los Angeles County', 'Orange County', or 'San Diego County'\n"
        "- state: the state abbreviation or name, e.g., 'CA' or 'California' (if provided)\n"
        "- pet_fee: the specific pet fee amount text stated in the answer (e.g., '$75 per stay', '$50 per night'); include the currency symbol and units if present\n"
        "- pet_fee_basis: if the answer states per-night, per-stay, one-time, etc., extract that basis (if provided)\n"
        "- parking_fee: the specific parking fee amount text stated in the answer (e.g., '$45 per night', '$30 per day')\n"
        "- parking_basis: if the answer states per-day, per-night, overnight, etc., extract that basis (if provided)\n"
        "- parking_type: whether the parking is self-parking, valet, or both (if provided)\n"
        "- url: a single explicit URL for the hotel's official site OR a major booking platform page (must be an actual URL in the answer)\n\n"
        "Rules:\n"
        "1) Only extract values explicitly present in the answer. If a value is missing for a hotel, set it to null.\n"
        "2) For 'url', extract a valid, complete URL exactly as written in the answer. If the answer includes markdown links, return the actual URL.\n"
        "3) Return a JSON object with a 'hotels' array containing up to four hotel objects."
    )


# --------------------------------------------------------------------------- #
# Additional instructions for verification                                    #
# --------------------------------------------------------------------------- #
def loc_additional_instruction() -> str:
    # Provide helpful city hints by county and emphasize CA
    parts = [
        "Acceptable counties: Los Angeles County, Orange County, San Diego County (California).",
        "You must verify from the webpage that the property's address/city is in California and falls within one of these counties.",
        "If the county name is not explicitly shown, infer from the city/neighborhood when reasonable.",
        "Common cities by county (not exhaustive):"
    ]
    for county, cities in CITY_HINTS.items():
        parts.append(f"- {county}: {', '.join(cities)}")
    parts.append(
        "If the webpage indicates a city that clearly belongs to one of these counties, consider the county requirement satisfied. "
        "Do not use your own external browsing; rely on the page content and reasonable geographic knowledge."
    )
    return "\n".join(parts)


def pet_policy_additional_instruction() -> str:
    return (
        "Confirm BOTH of the following on the webpage:\n"
        "1) Dogs are allowed (look for terms like 'dog', 'dogs', 'pet-friendly', 'pets allowed' and ensure dogs are included; restrictions are okay).\n"
        "2) A specific pet fee amount is publicly stated (examples: '$75 per stay', '$50 per night', 'one-time $150'). "
        "If a basis is mentioned, note per-night, per-stay, or one-time. Minor wording differences are acceptable."
    )


def parking_additional_instruction() -> str:
    return (
        "Confirm BOTH of the following on the webpage:\n"
        "1) The hotel offers on-site parking (self-parking and/or valet parking). Keywords: 'on-site', 'on property', 'parking on site'.\n"
        "2) A specific daily or nightly fee is publicly stated for parking (e.g., '$45 per night', '$30 per day', 'overnight parking $55'). "
        "Street parking or third-party offsite parking does NOT satisfy 'on-site'. Minor wording/formatting differences are acceptable."
    )


def url_additional_instruction() -> str:
    return (
        "This must be either the hotel's official website or a major booking platform page that displays the hotel's information. "
        "Major platforms include (non-exhaustive): "
        + ", ".join(MAJOR_PLATFORMS)
        + ". The page should clearly present hotel details such as name, address/location, amenities/policies, etc."
    )


# --------------------------------------------------------------------------- #
# Verification helpers                                                        #
# --------------------------------------------------------------------------- #
async def verify_hotel(
    evaluator: Evaluator,
    parent_node,
    hotel: HotelItem,
    hotel_index: int,
) -> None:
    """
    Build verification sub-tree for a single hotel with the following critical checks:
    - URL presence (custom, gates others)
    - URL qualifies (official or major platform)
    - Location in LA/Orange/San Diego County, CA
    - Pet policy: dogs allowed + specific pet fee stated
    - Parking: on-site + specific parking fee stated

    Additionally, enforce that the answer provided pet_fee and parking_fee strings (custom nodes) to satisfy
    the 'provide the specific fee' requirement.
    """
    # Create hotel node as a parallel aggregator (non-critical to allow partial credit across hotels)
    hotel_node = evaluator.add_parallel(
        id=f"hotel_{hotel_index}",
        desc=(
            "First qualifying pet-friendly hotel" if hotel_index == 1 else
            "Second qualifying pet-friendly hotel (must be a different property from Hotel 1)" if hotel_index == 2 else
            "Third qualifying pet-friendly hotel (must be a different property from Hotels 1 and 2)" if hotel_index == 3 else
            "Fourth qualifying pet-friendly hotel (must be a different property from Hotels 1, 2, and 3)"
        ),
        parent=parent_node,
        critical=False
    )

    # Basic presence checks to gate URL-based verifications
    url_present = bool(hotel and hotel.url and hotel.url.strip().startswith(("http://", "https://")))
    evaluator.add_custom_node(
        result=url_present,
        id=f"hotel_{hotel_index}_url_present",
        desc=f"Hotel #{hotel_index}: URL is provided and looks valid",
        parent=hotel_node,
        critical=True
    )

    # Enforce that the answer actually provided fee values (as specified by the task)
    pet_fee_provided = bool(hotel and hotel.pet_fee and str(hotel.pet_fee).strip())
    evaluator.add_custom_node(
        result=pet_fee_provided,
        id=f"hotel_{hotel_index}_pet_fee_provided",
        desc=f"Hotel #{hotel_index}: Answer provides a specific pet fee string",
        parent=hotel_node,
        critical=True
    )

    parking_fee_provided = bool(hotel and hotel.parking_fee and str(hotel.parking_fee).strip())
    evaluator.add_custom_node(
        result=parking_fee_provided,
        id=f"hotel_{hotel_index}_parking_fee_provided",
        desc=f"Hotel #{hotel_index}: Answer provides a specific parking fee string",
        parent=hotel_node,
        critical=True
    )

    # 1) URL qualification check (Critical)
    url_node = evaluator.add_leaf(
        id=f"hotel_{hotel_index}_url",
        desc="Provide a valid URL to the hotel's official website or a major booking platform page showing the hotel information",
        parent=hotel_node,
        critical=True
    )
    await evaluator.verify(
        claim="This webpage is either the hotel's official website or a major booking platform page that shows the hotel's information.",
        node=url_node,
        sources=hotel.url if hotel and hotel.url else None,
        additional_instruction=url_additional_instruction()
    )

    # 2) Location check (Critical)
    loc_node = evaluator.add_leaf(
        id=f"hotel_{hotel_index}_location",
        desc="Hotel must be located in Los Angeles County, Orange County, or San Diego County, California",
        parent=hotel_node,
        critical=True
    )
    # Build a robust claim that allows the judge to use the city/address on the page to determine county membership
    city_part = f"in {hotel.city}, " if hotel and hotel.city else ""
    county_part = ""
    if hotel and hotel.county:
        county_part = f"in {hotel.county}, "
    claim_location = (
        f"The hotel's webpage indicates the property is {city_part}{county_part}California, and the location is within one of "
        f"Los Angeles County, Orange County, or San Diego County."
    )
    await evaluator.verify(
        claim=claim_location,
        node=loc_node,
        sources=hotel.url if hotel and hotel.url else None,
        additional_instruction=loc_additional_instruction()
    )

    # 3) Pet policy check (Critical)
    pet_node = evaluator.add_leaf(
        id=f"hotel_{hotel_index}_pet_policy",
        desc="Hotel must explicitly allow dogs and state a pet fee (either per night or per stay)",
        parent=hotel_node,
        critical=True
    )
    if hotel and hotel.pet_fee:
        basis_text = f" charged {hotel.pet_fee_basis}" if hotel.pet_fee_basis else ""
        claim_pet = (
            f"The webpage shows that dogs are allowed and publicly states a specific pet fee of '{hotel.pet_fee}'{basis_text}."
        )
    else:
        claim_pet = (
            "The webpage shows that dogs are allowed and publicly states a specific pet fee amount (either per night or per stay)."
        )
    await evaluator.verify(
        claim=claim_pet,
        node=pet_node,
        sources=hotel.url if hotel and hotel.url else None,
        additional_instruction=pet_policy_additional_instruction()
    )

    # 4) Parking check (Critical)
    park_node = evaluator.add_leaf(
        id=f"hotel_{hotel_index}_parking",
        desc="Hotel must offer on-site parking (self-parking or valet) with stated daily or nightly fee",
        parent=hotel_node,
        critical=True
    )
    parking_type_text = f"{hotel.parking_type} " if hotel and hotel.parking_type else ""
    if hotel and hotel.parking_fee:
        basis_text = f" {hotel.parking_basis}" if hotel.parking_basis else ""
        claim_parking = (
            f"The webpage shows the hotel offers on-site {parking_type_text}parking and publicly states a parking fee of "
            f"'{hotel.parking_fee}'{basis_text} (daily or nightly)."
        )
    else:
        claim_parking = (
            f"The webpage shows the hotel offers on-site {parking_type_text}parking and publicly states a daily or nightly parking fee."
        )
    await evaluator.verify(
        claim=claim_parking,
        node=park_node,
        sources=hotel.url if hotel and hotel.url else None,
        additional_instruction=parking_additional_instruction()
    )


# --------------------------------------------------------------------------- #
# Main evaluation function                                                    #
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
    Evaluate an answer for the Southern California pet-friendly hotels task.
    """
    # Initialize evaluator with a parallel root node (hotels are independent)
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,
        agent_name=agent_name,
        answer_name=answer_name,
        client=client,
        task_description="Find four pet-friendly hotels in Southern California (Los Angeles, Orange, or San Diego counties) that meet all specified requirements",
        answer=answer,
        global_cache=cache,
        global_semaphore=semaphore,
        logger=logger,
        default_model=model
    )

    # Record useful reference info
    evaluator.add_custom_info(
        info={"allowed_counties": ALLOWED_COUNTIES, "major_platform_examples": MAJOR_PLATFORMS},
        info_type="settings",
        info_name="evaluation_settings"
    )

    # Extract structured hotel info from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_hotels(),
        template_class=HotelsExtraction,
        extraction_name="hotels_extraction"
    )

    # Normalize to exactly 4 items (pad with empty if fewer; truncate if more)
    hotels: List[HotelItem] = list(extracted.hotels)[:4]
    while len(hotels) < 4:
        hotels.append(HotelItem())

    # Build verification subtrees for each of the 4 hotels (parallel under root)
    # We keep them in order as hotel_1 .. hotel_4 to align with rubric
    for idx in range(4):
        await verify_hotel(evaluator, root, hotels[idx], idx + 1)

    # Return evaluation summary
    return evaluator.get_summary()