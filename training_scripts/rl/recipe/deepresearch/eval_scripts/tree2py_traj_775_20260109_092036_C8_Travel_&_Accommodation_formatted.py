import asyncio
import logging
from typing import Optional, List, Dict, Any, Tuple

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy

# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "california_hotels_brand_requirements"
TASK_DESCRIPTION = """
Identify four different hotels in California, United States, each from a different hotel chain, that meet the following specific requirements:

Hotel 1: Find a Kimpton Hotels property in California that charges no pet fees and has no size or weight restrictions for pets.

Hotel 2: Find a Red Roof Inn or Red Roof PLUS+ property in California where the first pet stays free.

Hotel 3: Find a Hyatt Place property in California that offers free breakfast and has a 24-hour fitness center as standard amenities.

Hotel 4: Find a La Quinta Inn & Suites by Wyndham property in California that is located within 15 miles of a major California airport (LAX, SFO, SAN, ONT, SJC, or SMF).

For each hotel, provide:
- The hotel's full name
- The city where it is located
- A brief description confirming it meets the specified requirements
- A reference URL from the hotel's official website or a recognized booking platform
"""

ALLOWED_AIRPORTS: List[Dict[str, str]] = [
    {"code": "LAX", "name": "Los Angeles International Airport"},
    {"code": "SFO", "name": "San Francisco International Airport"},
    {"code": "SAN", "name": "San Diego International Airport"},
    {"code": "ONT", "name": "Ontario International Airport"},
    {"code": "SJC", "name": "San Jose International Airport"},
    {"code": "SMF", "name": "Sacramento International Airport"},
]


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class HotelItem(BaseModel):
    """Information for one hotel item extracted from the answer."""
    name: Optional[str] = None
    city: Optional[str] = None
    brand: Optional[str] = None
    description: Optional[str] = None
    url: Optional[str] = None
    near_airport: Optional[str] = None  # Especially for Hotel 4, e.g., "LAX" or full airport name


class HotelsExtraction(BaseModel):
    """Model for the extracted hotels list from the answer."""
    hotels: List[HotelItem] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_hotels() -> str:
    return """
    Extract up to four hotel entries from the answer (ideally one for each of the four specified hotel categories).
    For each extracted hotel, return the following fields exactly as stated in the answer:
    - name: full property name
    - city: the city in California (if provided)
    - brand: the hotel chain or brand (e.g., "Kimpton", "Red Roof Inn", "Hyatt Place", "La Quinta Inn & Suites by Wyndham")
    - description: a brief sentence or phrase from the answer confirming it meets the specific requirement for its category
    - url: a reference URL from the hotel's official website or a recognized booking platform
    - near_airport: ONLY for the La Quinta hotel (Hotel 4 category), the airport code (LAX, SFO, SAN, ONT, SJC, or SMF) or airport name mentioned in the answer; otherwise null

    Return a JSON object with a 'hotels' array of objects including these fields.
    If any field is missing in the answer for a given hotel, set it to null.
    Do not fabricate or infer information that isn't explicitly in the answer.
    Extract all hotels mentioned, then include up to the first four relevant entries.
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def normalize_brand(brand: Optional[str]) -> Optional[str]:
    """Normalize brand names into canonical labels for uniqueness checks."""
    if not brand:
        return None
    s = brand.strip().lower()
    # Kimpton
    if "kimpton" in s:
        return "kimpton"
    # Red Roof / Red Roof PLUS+
    if "red roof" in s:
        return "red roof"
    # Hyatt Place (ensure not confusing with other Hyatt brands)
    if "hyatt place" in s:
        return "hyatt place"
    # La Quinta Inn & Suites by Wyndham / La Quinta by Wyndham / La Quinta
    if "la quinta" in s:
        return "la quinta"
    return s


def is_valid_url(url: Optional[str]) -> bool:
    if not url:
        return False
    u = url.strip().lower()
    return u.startswith("http://") or u.startswith("https://")


def match_airport_code_or_name(text: Optional[str]) -> Optional[Tuple[str, str]]:
    """Return (code, name) for allowed airports if text matches either code or name."""
    if not text:
        return None
    t = text.strip().upper()
    for ap in ALLOWED_AIRPORTS:
        if t == ap["code"] or t.lower() == ap["name"].lower():
            return ap["code"], ap["name"]
        # Partial contains check for names (e.g., "Los Angeles International Airport")
        if ap["name"].lower() in t.lower():
            return ap["code"], ap["name"]
    return None


def select_hotels_by_category(extracted: HotelsExtraction) -> Dict[str, Optional[HotelItem]]:
    """
    Select first matching hotel for each required category by brand:
      - hotel_1: kimpton
      - hotel_2: red roof
      - hotel_3: hyatt place
      - hotel_4: la quinta
    If multiple are present, pick the first seen in the extracted list.
    """
    selection = {
        "hotel_1": None,
        "hotel_2": None,
        "hotel_3": None,
        "hotel_4": None,
    }
    for item in extracted.hotels:
        nb = normalize_brand(item.brand)
        if nb == "kimpton" and selection["hotel_1"] is None:
            selection["hotel_1"] = item
        elif nb == "red roof" and selection["hotel_2"] is None:
            selection["hotel_2"] = item
        elif nb == "hyatt place" and selection["hotel_3"] is None:
            selection["hotel_3"] = item
        elif nb == "la quinta" and selection["hotel_4"] is None:
            selection["hotel_4"] = item
        # Stop early if we have all four
        if all(selection.values()):
            break
    return selection


def hotels_provided(selection: Dict[str, Optional[HotelItem]]) -> bool:
    """Check that all four category hotels are present with a name and a URL."""
    for key in ["hotel_1", "hotel_2", "hotel_3", "hotel_4"]:
        h = selection.get(key)
        if not h or not h.name or not is_valid_url(h.url):
            return False
    return True


def brands_are_distinct(selection: Dict[str, Optional[HotelItem]]) -> bool:
    """Ensure all four selected hotels are from distinct brands/chains."""
    brands = []
    for key in ["hotel_1", "hotel_2", "hotel_3", "hotel_4"]:
        h = selection.get(key)
        nb = normalize_brand(h.brand if h else None)
        if not nb:
            return False
        brands.append(nb)
    return len(set(brands)) == 4


def properties_are_distinct(selection: Dict[str, Optional[HotelItem]]) -> bool:
    """Ensure all four selected hotels have distinct property names."""
    names = []
    for key in ["hotel_1", "hotel_2", "hotel_3", "hotel_4"]:
        h = selection.get(key)
        if not h or not h.name:
            return False
        names.append(h.name.strip().lower())
    return len(set(names)) == 4


# --------------------------------------------------------------------------- #
# Verification functions for each hotel category                              #
# --------------------------------------------------------------------------- #
async def verify_hotel_1(evaluator: Evaluator, parent, hotel: Optional[HotelItem]) -> None:
    """
    Hotel 1: Kimpton in California; charges no pet fees; no size/weight restrictions.
    Required fields and URL checks included.
    """
    node = evaluator.add_parallel(
        id="hotel_1",
        desc="Hotel 1 requirements (Kimpton in California; pet policy criteria; required fields and URL)",
        parent=parent,
        critical=False
    )

    # Existence/required fields (critical custom nodes)
    evaluator.add_custom_node(
        result=bool(hotel and hotel.name and hotel.name.strip()),
        id="hotel_1_full_name_provided",
        desc="Hotel's full name is provided",
        parent=node,
        critical=True
    )
    evaluator.add_custom_node(
        result=bool(hotel and hotel.city and hotel.city.strip()),
        id="hotel_1_city_provided",
        desc="Hotel's city is provided",
        parent=node,
        critical=True
    )
    evaluator.add_custom_node(
        result=bool(hotel and is_valid_url(hotel.url)),
        id="hotel_1_reference_url",
        desc="A verifiable reference URL from the hotel's official website or a recognized booking platform is provided",
        parent=node,
        critical=True
    )
    evaluator.add_custom_node(
        result=bool(hotel and hotel.description and hotel.description.strip()),
        id="hotel_1_confirming_description",
        desc="Brief description is provided that explicitly confirms the hotel meets the Hotel 1 requirements",
        parent=node,
        critical=True
    )

    # Location: California
    loc_node = evaluator.add_leaf(
        id="hotel_1_location_california",
        desc="Hotel is located in California, United States",
        parent=node,
        critical=True
    )
    loc_claim = f"The hotel named '{hotel.name if hotel and hotel.name else ''}' is located in California, United States."
    await evaluator.verify(
        claim=loc_claim,
        node=loc_node,
        sources=(hotel.url if hotel else None),
        additional_instruction="Check the address or city/state on the provided hotel page. The property must be in California (CA)."
    )

    # Brand: Kimpton
    brand_node = evaluator.add_leaf(
        id="hotel_1_brand_kimpton",
        desc="Hotel is a Kimpton Hotels property",
        parent=node,
        critical=True
    )
    brand_claim = f"'{hotel.name if hotel and hotel.name else ''}' is a Kimpton Hotels property."
    await evaluator.verify(
        claim=brand_claim,
        node=brand_node,
        sources=(hotel.url if hotel else None),
        additional_instruction="Verify the brand on the page (Kimpton Hotels), including branding, logos, or site domain indicating Kimpton."
    )

    # Pet policy: no pet fees
    no_fee_node = evaluator.add_leaf(
        id="hotel_1_no_pet_fees",
        desc="Hotel charges no pet fees",
        parent=node,
        critical=True
    )
    no_fee_claim = "The hotel's pet policy states that there are no pet fees."
    await evaluator.verify(
        claim=no_fee_claim,
        node=no_fee_node,
        sources=(hotel.url if hotel else None),
        additional_instruction="Look for phrases like 'no pet fee', 'no additional fee', or 'pets stay free'. If any pet fee is mentioned, this should be judged as not supported."
    )

    # Pet policy: no size/weight restrictions
    no_restr_node = evaluator.add_leaf(
        id="hotel_1_no_size_weight_restrictions",
        desc="Hotel has no size or weight restrictions for pets",
        parent=node,
        critical=True
    )
    no_restr_claim = "The hotel's pet policy states there are no size or weight restrictions for pets."
    await evaluator.verify(
        claim=no_restr_claim,
        node=no_restr_node,
        sources=(hotel.url if hotel else None),
        additional_instruction="Look for statements like 'no size restrictions' or 'no weight limits'. If any weight/size cap is mentioned, judge as not supported."
    )


async def verify_hotel_2(evaluator: Evaluator, parent, hotel: Optional[HotelItem]) -> None:
    """
    Hotel 2: Red Roof Inn or Red Roof PLUS+ in California; first pet stays free.
    """
    node = evaluator.add_parallel(
        id="hotel_2",
        desc="Hotel 2 requirements (Red Roof in California; first pet stays free; required fields and URL)",
        parent=parent,
        critical=False
    )

    evaluator.add_custom_node(
        result=bool(hotel and hotel.name and hotel.name.strip()),
        id="hotel_2_full_name_provided",
        desc="Hotel's full name is provided",
        parent=node,
        critical=True
    )
    evaluator.add_custom_node(
        result=bool(hotel and hotel.city and hotel.city.strip()),
        id="hotel_2_city_provided",
        desc="Hotel's city is provided",
        parent=node,
        critical=True
    )
    evaluator.add_custom_node(
        result=bool(hotel and is_valid_url(hotel.url)),
        id="hotel_2_reference_url",
        desc="A verifiable reference URL from the hotel's official website or a recognized booking platform is provided",
        parent=node,
        critical=True
    )
    evaluator.add_custom_node(
        result=bool(hotel and hotel.description and hotel.description.strip()),
        id="hotel_2_confirming_description",
        desc="Brief description is provided that explicitly confirms the hotel meets the Hotel 2 requirements",
        parent=node,
        critical=True
    )

    # Location: California
    loc_node = evaluator.add_leaf(
        id="hotel_2_location_california",
        desc="Hotel is located in California, United States",
        parent=node,
        critical=True
    )
    loc_claim = f"The hotel named '{hotel.name if hotel and hotel.name else ''}' is located in California, United States."
    await evaluator.verify(
        claim=loc_claim,
        node=loc_node,
        sources=(hotel.url if hotel else None),
        additional_instruction="Check the address or city/state on the provided hotel page. The property must be in California (CA)."
    )

    # Brand: Red Roof Inn or Red Roof PLUS+
    brand_node = evaluator.add_leaf(
        id="hotel_2_brand_red_roof",
        desc="Hotel is a Red Roof Inn or Red Roof PLUS+ property",
        parent=node,
        critical=True
    )
    brand_claim = f"'{hotel.name if hotel and hotel.name else ''}' is a Red Roof Inn or Red Roof PLUS+ property."
    await evaluator.verify(
        claim=brand_claim,
        node=brand_node,
        sources=(hotel.url if hotel else None),
        additional_instruction="Verify Red Roof branding on the page (Red Roof Inn or Red Roof PLUS+)."
    )

    # Pet policy: first pet stays free
    pet_free_node = evaluator.add_leaf(
        id="hotel_2_first_pet_free",
        desc="Hotel's pet policy states that the first pet stays free",
        parent=node,
        critical=True
    )
    pet_free_claim = "The hotel's pet policy states that the first pet stays free."
    await evaluator.verify(
        claim=pet_free_claim,
        node=pet_free_node,
        sources=(hotel.url if hotel else None),
        additional_instruction="Look for statements like 'first pet stays free' or 'one pet is free'. If all pets incur a fee, judge as not supported."
    )


async def verify_hotel_3(evaluator: Evaluator, parent, hotel: Optional[HotelItem]) -> None:
    """
    Hotel 3: Hyatt Place in California; offers free breakfast; has a 24-hour fitness center.
    """
    node = evaluator.add_parallel(
        id="hotel_3",
        desc="Hotel 3 requirements (Hyatt Place in California; free breakfast; 24-hour fitness; required fields and URL)",
        parent=parent,
        critical=False
    )

    evaluator.add_custom_node(
        result=bool(hotel and hotel.name and hotel.name.strip()),
        id="hotel_3_full_name_provided",
        desc="Hotel's full name is provided",
        parent=node,
        critical=True
    )
    evaluator.add_custom_node(
        result=bool(hotel and hotel.city and hotel.city.strip()),
        id="hotel_3_city_provided",
        desc="Hotel's city is provided",
        parent=node,
        critical=True
    )
    evaluator.add_custom_node(
        result=bool(hotel and is_valid_url(hotel.url)),
        id="hotel_3_reference_url",
        desc="A verifiable reference URL from the hotel's official website or a recognized booking platform is provided",
        parent=node,
        critical=True
    )
    evaluator.add_custom_node(
        result=bool(hotel and hotel.description and hotel.description.strip()),
        id="hotel_3_confirming_description",
        desc="Brief description is provided that explicitly confirms the hotel meets the Hotel 3 requirements",
        parent=node,
        critical=True
    )

    # Location: California
    loc_node = evaluator.add_leaf(
        id="hotel_3_location_california",
        desc="Hotel is located in California, United States",
        parent=node,
        critical=True
    )
    loc_claim = f"The hotel named '{hotel.name if hotel and hotel.name else ''}' is located in California, United States."
    await evaluator.verify(
        claim=loc_claim,
        node=loc_node,
        sources=(hotel.url if hotel else None),
        additional_instruction="Check the hotel's address or city/state on the provided page. It must be in California (CA)."
    )

    # Brand: Hyatt Place
    brand_node = evaluator.add_leaf(
        id="hotel_3_brand_hyatt_place",
        desc="Hotel is a Hyatt Place property",
        parent=node,
        critical=True
    )
    brand_claim = f"'{hotel.name if hotel and hotel.name else ''}' is a Hyatt Place property."
    await evaluator.verify(
        claim=brand_claim,
        node=brand_node,
        sources=(hotel.url if hotel else None),
        additional_instruction="Verify Hyatt Place branding and property identification."
    )

    # Amenity: free breakfast
    breakfast_node = evaluator.add_leaf(
        id="hotel_3_free_breakfast",
        desc="Hotel offers free breakfast as a standard amenity",
        parent=node,
        critical=True
    )
    breakfast_claim = "The hotel offers free (complimentary) breakfast as a standard amenity."
    await evaluator.verify(
        claim=breakfast_claim,
        node=breakfast_node,
        sources=(hotel.url if hotel else None),
        additional_instruction="Check the amenities or features page for 'free breakfast' or 'complimentary breakfast'. If it's only for certain members or paid, judge as not supported."
    )

    # Amenity: 24-hour fitness center
    fitness_node = evaluator.add_leaf(
        id="hotel_3_24h_fitness",
        desc="Hotel has a 24-hour fitness center",
        parent=node,
        critical=True
    )
    fitness_claim = "The hotel has a fitness center that is open 24 hours."
    await evaluator.verify(
        claim=fitness_claim,
        node=fitness_node,
        sources=(hotel.url if hotel else None),
        additional_instruction="Look for '24-hour fitness' or 'fitness center open 24/7'."
    )


async def verify_hotel_4(evaluator: Evaluator, parent, hotel: Optional[HotelItem]) -> None:
    """
    Hotel 4: La Quinta Inn & Suites by Wyndham in California; within 15 miles of one of LAX/SFO/SAN/ONT/SJC/SMF.
    """
    node = evaluator.add_parallel(
        id="hotel_4",
        desc="Hotel 4 requirements (La Quinta in California; within 15 miles of specified airport; required fields and URL)",
        parent=parent,
        critical=False
    )

    evaluator.add_custom_node(
        result=bool(hotel and hotel.name and hotel.name.strip()),
        id="hotel_4_full_name_provided",
        desc="Hotel's full name is provided",
        parent=node,
        critical=True
    )
    evaluator.add_custom_node(
        result=bool(hotel and hotel.city and hotel.city.strip()),
        id="hotel_4_city_provided",
        desc="Hotel's city is provided",
        parent=node,
        critical=True
    )
    evaluator.add_custom_node(
        result=bool(hotel and is_valid_url(hotel.url)),
        id="hotel_4_reference_url",
        desc="A verifiable reference URL from the hotel's official website or a recognized booking platform is provided",
        parent=node,
        critical=True
    )
    evaluator.add_custom_node(
        result=bool(hotel and hotel.description and hotel.description.strip()),
        id="hotel_4_confirming_description",
        desc="Brief description is provided that explicitly confirms the hotel meets the Hotel 4 requirements",
        parent=node,
        critical=True
    )

    # Location: California
    loc_node = evaluator.add_leaf(
        id="hotel_4_location_california",
        desc="Hotel is located in California, United States",
        parent=node,
        critical=True
    )
    loc_claim = f"The hotel named '{hotel.name if hotel and hotel.name else ''}' is located in California, United States."
    await evaluator.verify(
        claim=loc_claim,
        node=loc_node,
        sources=(hotel.url if hotel else None),
        additional_instruction="Check the hotel's address or city/state on the page. It must be in California (CA)."
    )

    # Brand: La Quinta Inn & Suites by Wyndham
    brand_node = evaluator.add_leaf(
        id="hotel_4_brand_la_quinta",
        desc="Hotel is a La Quinta Inn & Suites by Wyndham property",
        parent=node,
        critical=True
    )
    brand_claim = f"'{hotel.name if hotel and hotel.name else ''}' is a La Quinta Inn & Suites by Wyndham property."
    await evaluator.verify(
        claim=brand_claim,
        node=brand_node,
        sources=(hotel.url if hotel else None),
        additional_instruction="Verify La Quinta by Wyndham branding and property identification."
    )

    # Airport proximity: within 15 miles of specified airport
    airport_info = match_airport_code_or_name(hotel.near_airport if hotel else None)
    code, name = (airport_info if airport_info else ("one of LAX/SFO/SAN/ONT/SJC/SMF", "one of the specified airports"))
    proximity_node = evaluator.add_leaf(
        id="hotel_4_within_15_miles_airport",
        desc="Hotel is located within 15 miles of a major California airport (LAX, SFO, SAN, ONT, SJC, or SMF)",
        parent=node,
        critical=True
    )
    proximity_claim = f"The hotel is located within 15 miles of {name} ({code})."
    await evaluator.verify(
        claim=proximity_claim,
        node=proximity_node,
        sources=(hotel.url if hotel else None),
        additional_instruction="Look for an explicit distance (e.g., 'X miles from the airport') on the provided page. Accept if the stated distance is ≤ 15 miles. If no distance info is available, judge as not supported."
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
) -> Dict[str, Any]:
    """
    Evaluate an answer for the California hotels brand-specific requirements task.
    """
    # Initialize evaluator (root is non-critical and parallel by default)
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

    # Extract hotels from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_hotels(),
        template_class=HotelsExtraction,
        extraction_name="extracted_hotels",
    )

    # Select hotels by required categories (Kimpton, Red Roof, Hyatt Place, La Quinta)
    selection = select_hotels_by_category(extracted)

    # Record selection diagnostic info
    evaluator.add_custom_info(
        info={
            "selected_hotel_1": (selection["hotel_1"].dict() if selection["hotel_1"] else None),
            "selected_hotel_2": (selection["hotel_2"].dict() if selection["hotel_2"] else None),
            "selected_hotel_3": (selection["hotel_3"].dict() if selection["hotel_3"] else None),
            "selected_hotel_4": (selection["hotel_4"].dict() if selection["hotel_4"] else None),
        },
        info_type="selection_debug",
        info_name="selected_hotels_by_category"
    )

    # Global requirements (critical parallel node)
    global_node = evaluator.add_parallel(
        id="global_requirements",
        desc="Global response requirements across all four hotels",
        parent=root,
        critical=True
    )

    # Leaf: provides four hotels (category coverage)
    evaluator.add_custom_node(
        result=hotels_provided(selection),
        id="provides_four_hotels",
        desc="Response includes four hotels corresponding to Hotel 1–Hotel 4",
        parent=global_node,
        critical=True
    )

    # Leaf: distinct brands
    evaluator.add_custom_node(
        result=brands_are_distinct(selection),
        id="distinct_brands",
        desc="All four hotels are from different hotel chains/brands",
        parent=global_node,
        critical=True
    )

    # Leaf: distinct properties
    evaluator.add_custom_node(
        result=properties_are_distinct(selection),
        id="distinct_properties",
        desc="All four hotels are different properties (not the same hotel repeated)",
        parent=global_node,
        critical=True
    )

    # Per-hotel verification
    await verify_hotel_1(evaluator, root, selection["hotel_1"])
    await verify_hotel_2(evaluator, root, selection["hotel_2"])
    await verify_hotel_3(evaluator, root, selection["hotel_3"])
    await verify_hotel_4(evaluator, root, selection["hotel_4"])

    # Return structured summary
    return evaluator.get_summary()