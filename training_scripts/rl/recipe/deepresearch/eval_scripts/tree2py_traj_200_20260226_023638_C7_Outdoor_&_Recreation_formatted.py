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
TASK_ID = "us_airport_amenities"
TASK_DESCRIPTION = (
    "Identify a commercial passenger airport in the United States that offers all of the following amenities for families "
    "traveling for outdoor recreation activities: designated pet relief areas, mother's rooms (nursing or family rooms), "
    "free WiFi throughout the terminal, electronic device charging stations, food and beverage options, 24-hour operations "
    "or services, accessible parking facilities, ground transportation options (such as taxi, ride-share, or shuttle services), "
    "TSA security screening, and restroom facilities. Provide the airport name, its airport code, and reference URLs confirming "
    "each required amenity."
)

REQUIRED_AMENITIES_FIELDS = [
    "pet_relief_urls",
    "mothers_room_urls",
    "free_wifi_urls",
    "charging_stations_urls",
    "food_beverage_urls",
    "operating_hours_urls",
    "parking_urls",
    "ground_transportation_urls",
    "tsa_urls",
    "restroom_urls",
]


# --------------------------------------------------------------------------- #
# Data models for structured extraction                                       #
# --------------------------------------------------------------------------- #
class AirportAmenitySources(BaseModel):
    us_location_urls: List[str] = Field(default_factory=list)
    commercial_service_urls: List[str] = Field(default_factory=list)
    pet_relief_urls: List[str] = Field(default_factory=list)
    mothers_room_urls: List[str] = Field(default_factory=list)
    free_wifi_urls: List[str] = Field(default_factory=list)
    charging_stations_urls: List[str] = Field(default_factory=list)
    food_beverage_urls: List[str] = Field(default_factory=list)
    operating_hours_urls: List[str] = Field(default_factory=list)
    parking_urls: List[str] = Field(default_factory=list)
    ground_transportation_urls: List[str] = Field(default_factory=list)
    tsa_urls: List[str] = Field(default_factory=list)
    restroom_urls: List[str] = Field(default_factory=list)
    general_urls: List[str] = Field(default_factory=list)


class AirportSubmission(BaseModel):
    airport_name: Optional[str] = None
    airport_code: Optional[str] = None  # Prefer the 3-letter IATA code (e.g., CLT, BGR)
    airport_website_url: Optional[str] = None
    sources: AirportAmenitySources = Field(default_factory=AirportAmenitySources)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_airport_submission() -> str:
    return """
    You must extract a single airport submission from the answer. If multiple airports are mentioned, choose the first U.S. commercial passenger airport that the answer uses to satisfy the requirements. Extract:
    - airport_name: The full airport name as provided (e.g., "Charlotte Douglas International Airport")
    - airport_code: The 3-letter IATA airport code if provided (e.g., "CLT"). If multiple codes are given, choose the IATA 3-letter code.
    - airport_website_url: If an official airport website URL is provided or clearly implied.
    - sources: A nested object containing URL arrays. For each field, extract ALL URLs explicitly mentioned in the answer text (plain URLs or within markdown). Do not invent. If none are given for a field, return an empty array.
      sources.us_location_urls: URLs that show the airport is in the United States.
      sources.commercial_service_urls: URLs that show the airport has commercial passenger airline service (scheduled flights).
      sources.pet_relief_urls: URLs that show designated pet/animal relief areas exist at the airport.
      sources.mothers_room_urls: URLs that show "mother's rooms", nursing rooms, lactation rooms/pods, or family rooms for feeding.
      sources.free_wifi_urls: URLs that show free Wi-Fi is available in the terminal(s).
      sources.charging_stations_urls: URLs that show device charging stations/outlets/USB charging in terminal(s).
      sources.food_beverage_urls: URLs that show food and beverage options (dining/restaurants) in terminal(s).
      sources.operating_hours_urls: URLs that show 24-hour operations or services (e.g., "Open 24 hours", "24/7", or a specific service operating 24/7).
      sources.parking_urls: URLs that show accessible parking facilities for passengers (short-term/long-term/garage, etc.).
      sources.ground_transportation_urls: URLs that show ground transport options such as taxi, ride-share (Uber/Lyft), shuttles, or public transit serving the airport.
      sources.tsa_urls: URLs that show TSA security screening at the airport (e.g., security checkpoints, TSA PreCheck).
      sources.restroom_urls: URLs that show restroom facilities in the terminal(s).
      sources.general_urls: Any other URLs the answer cites and that may support multiple items above.
    Return a single JSON object matching the specified schema. If any field is missing, set to null (for strings) or [] (for URL arrays).
    """


# --------------------------------------------------------------------------- #
# Helpers                                                                     #
# --------------------------------------------------------------------------- #
def ensure_urls(urls: Optional[List[str]]) -> List[str]:
    if not urls:
        return []
    return [u.strip() for u in urls if isinstance(u, str) and u.strip()]


def pick_sources(primary: List[str], fallback: List[str]) -> List[str]:
    p = ensure_urls(primary)
    if p:
        return p
    return ensure_urls(fallback)


# --------------------------------------------------------------------------- #
# Verification logic                                                          #
# --------------------------------------------------------------------------- #
async def verify_airport(evaluator: Evaluator, parent_node, submission: AirportSubmission) -> None:
    # Create a container node mirroring the rubric root
    amenities_root = evaluator.add_parallel(
        id="US_Airport_Amenities",
        desc="Evaluate whether the identified airport is a US commercial passenger airport with all required amenities and provided information",
        parent=parent_node,
        critical=False
    )

    name_val = (submission.airport_name or "").strip()
    code_val = (submission.airport_code or "").strip()

    # 1) Existence checks (critical)
    evaluator.add_custom_node(
        result=bool(name_val),
        id="Airport_Name_Provided",
        desc="The solution must provide the name of the airport",
        parent=amenities_root,
        critical=True
    )

    evaluator.add_custom_node(
        result=bool(code_val),
        id="Airport_Code_Provided",
        desc="The solution must provide the airport code (e.g., CLT, BGR)",
        parent=amenities_root,
        critical=True
    )

    # Reference URLs must be provided for each required amenity
    s = submission.sources or AirportAmenitySources()
    # For this existence check, allow general_urls to serve as fallback for each amenity
    all_ok = True
    for field in REQUIRED_AMENITIES_FIELDS:
        amenity_urls = getattr(s, field, [])
        # consider general_urls as fallback presence
        if not ensure_urls(amenity_urls) and not ensure_urls(s.general_urls):
            all_ok = False
            break

    evaluator.add_custom_node(
        result=all_ok,
        id="Reference_URLs_Provided",
        desc="The solution must provide reference URLs confirming the required amenities",
        parent=amenities_root,
        critical=True
    )

    # Helper: build friendly airport reference for claims
    airport_ref = f"{name_val} ({code_val})" if name_val and code_val else (name_val or code_val or "the airport")

    # 2) Factual verifications (all critical) — use specific URLs with fallback to general_urls
    # 2.1 US Location
    node_us = evaluator.add_leaf(
        id="US_Location",
        desc="The airport must be located within the United States",
        parent=amenities_root,
        critical=True
    )
    claim_us = f"The airport {airport_ref} is located in the United States of America."
    await evaluator.verify(
        claim=claim_us,
        node=node_us,
        sources=pick_sources(s.us_location_urls, s.general_urls),
        additional_instruction="Confirm explicitly that this airport is in the USA (U.S., United States)."
    )

    # 2.2 Commercial passenger airport with scheduled service
    node_comm = evaluator.add_leaf(
        id="Commercial_Passenger_Airport",
        desc="The airport must be a commercial passenger airport with regular scheduled flights",
        parent=amenities_root,
        critical=True
    )
    claim_comm = f"The airport {airport_ref} is a commercial passenger airport with scheduled airline service."
    await evaluator.verify(
        claim=claim_comm,
        node=node_comm,
        sources=pick_sources(s.commercial_service_urls, s.general_urls),
        additional_instruction="Accept phrases like 'commercial service airport', 'primary commercial service airport', 'airline passenger service', or 'scheduled airline flights'."
    )

    # 3) Required amenities (all critical)
    # Pet Relief Areas
    node_pet = evaluator.add_leaf(
        id="Pet_Relief_Area",
        desc="The airport must have designated pet relief area facilities",
        parent=amenities_root,
        critical=True
    )
    claim_pet = f"The airport {airport_ref} provides designated pet/animal relief areas for passengers' pets/service animals."
    await evaluator.verify(
        claim=claim_pet,
        node=node_pet,
        sources=pick_sources(s.pet_relief_urls, s.general_urls),
        additional_instruction="Accept synonyms like 'pet relief area', 'animal relief area', 'SARA', 'pet relief station'."
    )

    # Mother's Room / Nursing / Family Room (lactation)
    node_mom = evaluator.add_leaf(
        id="Mothers_Room",
        desc="The airport must have mother's rooms (nursing rooms or family rooms)",
        parent=amenities_root,
        critical=True
    )
    claim_mom = f"The airport {airport_ref} provides mother's rooms such as nursing rooms, family rooms, lactation rooms, or lactation pods."
    await evaluator.verify(
        claim=claim_mom,
        node=node_mom,
        sources=pick_sources(s.mothers_room_urls, s.general_urls),
        additional_instruction="Accept 'lactation room', 'nursing room', 'Mamava pod', 'family room' explicitly intended for nursing/pumping."
    )

    # Free WiFi
    node_wifi = evaluator.add_leaf(
        id="Free_WiFi",
        desc="The airport must provide free WiFi access throughout the terminal",
        parent=amenities_root,
        critical=True
    )
    claim_wifi = f"The airport {airport_ref} offers free Wi‑Fi for passengers in the terminal(s)."
    await evaluator.verify(
        claim=claim_wifi,
        node=node_wifi,
        sources=pick_sources(s.free_wifi_urls, s.general_urls),
        additional_instruction="Accept terms like 'free Wi‑Fi', 'complimentary Wi‑Fi', or 'free internet access' in terminals or concourses."
    )

    # Charging Stations
    node_chg = evaluator.add_leaf(
        id="Charging_Stations",
        desc="The airport must have charging stations for electronic devices",
        parent=amenities_root,
        critical=True
    )
    claim_chg = f"The airport {airport_ref} provides charging stations, power outlets, or USB charging for electronic devices in terminal areas."
    await evaluator.verify(
        claim=claim_chg,
        node=node_chg,
        sources=pick_sources(s.charging_stations_urls, s.general_urls),
        additional_instruction="Accept 'charging station', 'power outlet', 'USB charging', 'charging ports' available to passengers."
    )

    # Food & Beverage
    node_fnb = evaluator.add_leaf(
        id="Food_Beverage",
        desc="The airport must have food and beverage options available",
        parent=amenities_root,
        critical=True
    )
    claim_fnb = f"The airport {airport_ref} has food and beverage (dining/restaurant) options available for passengers."
    await evaluator.verify(
        claim=claim_fnb,
        node=node_fnb,
        sources=pick_sources(s.food_beverage_urls, s.general_urls),
        additional_instruction="Accept dining, restaurants, cafes, concessions listings indicating food and beverage availability."
    )

    # 24-hour operations or services
    node_24h = evaluator.add_leaf(
        id="Operating_Hours",
        desc="The airport must offer 24-hour operations or services",
        parent=amenities_root,
        critical=True
    )
    claim_24h = f"The airport {airport_ref} has 24-hour operations or services (e.g., open 24 hours or specific services available 24/7)."
    await evaluator.verify(
        claim=claim_24h,
        node=node_24h,
        sources=pick_sources(s.operating_hours_urls, s.general_urls),
        additional_instruction="Accept explicit phrases like 'open 24 hours', '24/7'. It can be the airport generally or a core passenger service operating 24/7 (e.g., terminal access, parking)."
    )

    # Parking Facilities
    node_parking = evaluator.add_leaf(
        id="Parking_Facilities",
        desc="The airport must have accessible parking facilities",
        parent=amenities_root,
        critical=True
    )
    claim_parking = f"The airport {airport_ref} offers passenger parking facilities (e.g., short-term, long-term, garage), with accessible options."
    await evaluator.verify(
        claim=claim_parking,
        node=node_parking,
        sources=pick_sources(s.parking_urls, s.general_urls),
        additional_instruction="Accept official parking pages showing available passenger parking; accessible/ADA parking counts as supportive evidence."
    )

    # Ground Transportation
    node_gt = evaluator.add_leaf(
        id="Ground_Transportation",
        desc="The airport must have ground transportation options (taxi, ride-share, or shuttle services)",
        parent=amenities_root,
        critical=True
    )
    claim_gt = f"The airport {airport_ref} provides ground transportation options such as taxis, ride-share (Uber/Lyft), shuttles, or public transit."
    await evaluator.verify(
        claim=claim_gt,
        node=node_gt,
        sources=pick_sources(s.ground_transportation_urls, s.general_urls),
        additional_instruction="Accept any page that explicitly lists or describes taxi, rideshare, shuttle, or similar services serving the airport."
    )

    # TSA Security
    node_tsa = evaluator.add_leaf(
        id="TSA_Security",
        desc="The airport must have TSA security screening",
        parent=amenities_root,
        critical=True
    )
    claim_tsa = f"The airport {airport_ref} has TSA passenger security screening (e.g., security checkpoints, TSA PreCheck)."
    await evaluator.verify(
        claim=claim_tsa,
        node=node_tsa,
        sources=pick_sources(s.tsa_urls, s.general_urls),
        additional_instruction="Accept evidence of TSA checkpoints or TSA PreCheck operating at the airport."
    )

    # Restroom Facilities
    node_rr = evaluator.add_leaf(
        id="Restroom_Facilities",
        desc="The airport must have restroom facilities",
        parent=amenities_root,
        critical=True
    )
    claim_rr = f"The airport {airport_ref} provides restroom facilities in terminal areas."
    await evaluator.verify(
        claim=claim_rr,
        node=node_rr,
        sources=pick_sources(s.restroom_urls, s.general_urls),
        additional_instruction="Accept mention of 'restrooms' or 'toilets' in terminal maps, amenities, or facility pages."
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
    # Initialize evaluator with parallel aggregation (per rubric root)
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

    # Record requirement info (for transparency)
    evaluator.add_ground_truth({
        "required_amenities": [
            "Pet relief areas",
            "Mother's rooms / nursing / lactation rooms or pods",
            "Free Wi‑Fi in terminals",
            "Charging stations / outlets / USB charging",
            "Food & beverage options",
            "24‑hour operations or services",
            "Parking facilities (including accessible options)",
            "Ground transportation (taxi / ride‑share / shuttle / transit)",
            "TSA security screening",
            "Restroom facilities"
        ],
        "also_required": [
            "Airport name",
            "Airport code (IATA 3‑letter preferred)",
            "US location",
            "Commercial passenger service (scheduled flights)"
        ]
    }, gt_type="requirements")

    # Extract the airport submission
    submission = await evaluator.extract(
        prompt=prompt_extract_airport_submission(),
        template_class=AirportSubmission,
        extraction_name="airport_submission"
    )

    # Build verification tree and run checks
    await verify_airport(evaluator, root, submission)

    # Return aggregated summary
    return evaluator.get_summary()