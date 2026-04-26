import asyncio
import logging
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.llm_client.base_client import LLMClient


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "hotel_all_criteria_march_2026"
TASK_DESCRIPTION = """
Identify a hotel that meets ALL of the following criteria as of March 2026:

1. The hotel must belong to one of these major international hotel chains: Marriott, Hilton, IHG (InterContinental Hotels Group), Hyatt, Accor, Wyndham, Best Western, or Choice Hotels
2. The hotel must have a 4-star or higher rating
3. The hotel must be located in a city that has direct ITA Airways flight service from Rome, Italy
4. The hotel must be within 10 miles of the city's international airport
5. The hotel must have a swimming pool (indoor or outdoor)
6. The hotel must have a fitness center or gym facility
7. The hotel must have at least one on-site restaurant
8. The hotel must offer complimentary WiFi to guests
9. The hotel must have business center facilities
10. The hotel must offer spa or wellness services
11. The hotel must offer suite accommodations
12. The hotel must be wheelchair accessible with appropriate accessibility features
13. The hotel must provide parking facilities (either paid or complimentary)
14. The hotel must have a clearly stated pet policy
15. The hotel must offer a cancellation policy that allows guests to cancel their reservation at least 24 hours before check-in without penalty

Provide the hotel name, its specific location, and the hotel chain it belongs to.
"""

ALLOWED_CHAINS = [
    "Marriott",
    "Hilton",
    "IHG",
    "InterContinental Hotels Group",
    "Hyatt",
    "Accor",
    "Wyndham",
    "Best Western",
    "Choice Hotels",
]

# --------------------------------------------------------------------------- #
# Data models for structured extraction                                       #
# --------------------------------------------------------------------------- #
class HotelCore(BaseModel):
    hotel_name: Optional[str] = None
    location: Optional[str] = None  # Address or locality info as provided in the answer
    city: Optional[str] = None
    country_or_region: Optional[str] = None
    chain: Optional[str] = None
    airport_name: Optional[str] = None  # The city's international airport if mentioned
    rating_text: Optional[str] = None  # e.g., "4-star", "5 star", etc.
    explicitly_as_of_march_2026: Optional[bool] = None  # Whether the answer explicitly frames info "as of March 2026"


class HotelSources(BaseModel):
    official_site: Optional[str] = None
    chain_page_urls: List[str] = Field(default_factory=list)
    general_urls: List[str] = Field(default_factory=list)

    rating_urls: List[str] = Field(default_factory=list)
    ita_route_urls: List[str] = Field(default_factory=list)
    airport_proximity_urls: List[str] = Field(default_factory=list)

    pool_urls: List[str] = Field(default_factory=list)
    fitness_urls: List[str] = Field(default_factory=list)
    restaurant_urls: List[str] = Field(default_factory=list)
    wifi_urls: List[str] = Field(default_factory=list)
    business_center_urls: List[str] = Field(default_factory=list)
    spa_urls: List[str] = Field(default_factory=list)
    suites_urls: List[str] = Field(default_factory=list)
    accessibility_urls: List[str] = Field(default_factory=list)
    parking_urls: List[str] = Field(default_factory=list)
    pet_policy_urls: List[str] = Field(default_factory=list)
    cancellation_urls: List[str] = Field(default_factory=list)


class HotelAnswerExtraction(BaseModel):
    hotel: Optional[HotelCore] = None
    sources: Optional[HotelSources] = None


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_hotel_info() -> str:
    return """
Extract the SINGLE hotel that the answer ultimately recommends/identifies (choose the first if multiple are listed).
Return a JSON object with two top-level fields: "hotel" and "sources".

Under "hotel", extract:
- hotel_name: The full hotel name as written in the answer (do not add or change wording).
- location: The hotel's specific location as provided (address or sufficiently specific locality).
- city: The city where the hotel is located (if stated or can be unambiguously inferred from the answer's text).
- country_or_region: The country/region (if present).
- chain: The hotel chain/brand affiliation as named in the answer (e.g., "Marriott", "Hilton", "IHG", "Hyatt", "Accor", "Wyndham", "Best Western", "Choice Hotels" or a sub-brand like "Hilton Garden Inn", "Hyatt Regency", etc., but still capture the chain string the answer states).
- airport_name: The primary international airport serving the city (ONLY if explicitly named in the answer, otherwise null).
- rating_text: Any star-rating statement in the answer (e.g., "4-star", "5 star") exactly as presented; if absent, null.
- explicitly_as_of_march_2026: true if the answer explicitly frames or states the hotel meets the criteria "as of March 2026" (or equivalent phrase clearly indicating March 2026); otherwise false.

Under "sources", extract URL lists that the answer cites for each specific criterion:
- official_site: The hotel's official website URL (single string if provided; if none, null).
- chain_page_urls: URLs that show the hotel's chain/brand affiliation (e.g., the brand page, corporate site, or the hotel's page on the chain's domain).
- general_urls: Any general or catch-all URLs the answer cites for the hotel (e.g., booking/OTA pages, Wikipedia, etc.).
- rating_urls: URLs specifically supporting the star rating claim.
- ita_route_urls: URLs that support that ITA Airways operates DIRECT (non-stop) flights from Rome (FCO) to the hotel's city (e.g., ITA route map, schedule, or authoritative timetable).
- airport_proximity_urls: URLs that support the hotel's distance to the international airport (e.g., hotel's location page, map/directions, Google Maps links).
- pool_urls: URLs supporting that the hotel has a swimming pool.
- fitness_urls: URLs supporting that the hotel has a fitness center or gym.
- restaurant_urls: URLs supporting that the hotel has at least one on-site restaurant (on premises).
- wifi_urls: URLs supporting that the hotel offers complimentary WiFi to guests.
- business_center_urls: URLs supporting that the hotel has business center facilities.
- spa_urls: URLs supporting that the hotel offers spa or wellness services.
- suites_urls: URLs supporting that the hotel offers suite accommodations.
- accessibility_urls: URLs supporting wheelchair accessibility or accessibility features (e.g., ADA rooms, step-free access).
- parking_urls: URLs supporting that the hotel provides parking (paid or complimentary).
- pet_policy_urls: URLs showing a clearly stated pet policy (including "no pets allowed" policies).
- cancellation_urls: URLs supporting a cancellation policy that allows free cancellation at least 24 hours before check-in (note: rate plans may vary; capture typical/flexible policy pages).

General rules:
- Only extract URLs explicitly present in the answer's text. If a URL is missing a protocol, prepend "http://".
- If a specific category has no URL cited in the answer, return an empty list for that category (or null for 'official_site').
- Do not invent information; if the answer does not provide something, set it to null or an empty array as instructed.
"""


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _merge_urls(*url_groups: Optional[List[str] | str | None]) -> List[str]:
    """Merge multiple URL groups into a unique list, preserving order and skipping None/empty."""
    seen = set()
    merged: List[str] = []
    for group in url_groups:
        if not group:
            continue
        if isinstance(group, str):
            candidates = [group]
        else:
            candidates = list(group)
        for u in candidates:
            if not u:
                continue
            u = u.strip()
            if not u:
                continue
            if u not in seen:
                seen.add(u)
                merged.append(u)
    return merged


def _safe(val: Optional[str]) -> str:
    return val or ""


# --------------------------------------------------------------------------- #
# Verification logic                                                          #
# --------------------------------------------------------------------------- #
async def build_required_output_section(evaluator: Evaluator, root_node, ex: HotelAnswerExtraction) -> None:
    """
    Build the 'required_output_fields' parallel critical section with three leaf checks:
    - Provide the hotel name
    - Provide the hotel's specific location
    - Provide the hotel chain
    """
    node = evaluator.add_parallel(
        id="required_output_fields",
        desc="Response includes the required hotel identification details",
        parent=root_node,
        critical=True
    )

    hotel = ex.hotel or HotelCore()

    evaluator.add_custom_node(
        result=bool(hotel.hotel_name and hotel.hotel_name.strip()),
        id="provide_hotel_name",
        desc="Provide the hotel name",
        parent=node,
        critical=True
    )

    evaluator.add_custom_node(
        result=bool(hotel.location and hotel.location.strip()),
        id="provide_specific_location",
        desc="Provide the hotel's specific location (e.g., address or sufficiently specific locality information)",
        parent=node,
        critical=True
    )

    evaluator.add_custom_node(
        result=bool(hotel.chain and hotel.chain.strip()),
        id="provide_hotel_chain",
        desc="Provide the hotel chain the hotel belongs to",
        parent=node,
        critical=True
    )


async def build_eligibility_section(evaluator: Evaluator, root_node, ex: HotelAnswerExtraction) -> None:
    """
    Build the 'eligibility_criteria' parallel critical section with all leaf checks.
    Uses verify() with URL grounding wherever possible.
    """
    node = evaluator.add_parallel(
        id="eligibility_criteria",
        desc="The identified hotel meets all specified criteria as of March 2026",
        parent=root_node,
        critical=True
    )

    hotel = ex.hotel or HotelCore()
    src = ex.sources or HotelSources()

    # Common URL pools to reuse
    base_urls = _merge_urls(src.official_site, src.general_urls)
    chain_urls = _merge_urls(src.official_site, src.chain_page_urls, src.general_urls)
    rating_urls = _merge_urls(src.rating_urls, base_urls)
    ita_urls = _merge_urls(src.ita_route_urls)
    proximity_urls = _merge_urls(src.airport_proximity_urls, base_urls)

    pool_urls = _merge_urls(src.pool_urls, base_urls)
    fitness_urls = _merge_urls(src.fitness_urls, base_urls)
    restaurant_urls = _merge_urls(src.restaurant_urls, base_urls)
    wifi_urls = _merge_urls(src.wifi_urls, base_urls)
    business_urls = _merge_urls(src.business_center_urls, base_urls)
    spa_urls = _merge_urls(src.spa_urls, base_urls)
    suites_urls = _merge_urls(src.suites_urls, base_urls)
    accessibility_urls = _merge_urls(src.accessibility_urls, base_urls)
    parking_urls = _merge_urls(src.parking_urls, base_urls)
    pet_urls = _merge_urls(src.pet_policy_urls, base_urls)
    cancellation_urls = _merge_urls(src.cancellation_urls, base_urls)

    # 1) As-of timeframe explicitness (simple verify from the answer text)
    asof_node = evaluator.add_leaf(
        id="as_of_march_2026",
        desc='The response explicitly frames the hotel as meeting the listed criteria "as of March 2026" (i.e., it addresses the required timeframe rather than leaving it unspecified)',
        parent=node,
        critical=True
    )
    asof_claim = "The answer explicitly indicates that the hotel meets the listed criteria as of March 2026 (e.g., uses wording like 'as of March 2026')."
    await evaluator.verify(
        claim=asof_claim,
        node=asof_node,
        additional_instruction="Judge only based on whether the answer text itself clearly states 'as of March 2026' or an equivalent explicit timeframe mention for March 2026."
    )

    # 2) Chain membership within allowed set
    chain_node = evaluator.add_leaf(
        id="chain_membership",
        desc="The hotel belongs to one of: Marriott, Hilton, IHG, Hyatt, Accor, Wyndham, Best Western, or Choice Hotels",
        parent=node,
        critical=True
    )
    chain_claim = (
        f"The hotel named '{_safe(hotel.hotel_name)}' belongs to the hotel chain/brand '{_safe(hotel.chain)}', "
        f"and that chain is one of the following: {', '.join(ALLOWED_CHAINS)}."
    )
    await evaluator.verify(
        claim=chain_claim,
        node=chain_node,
        sources=chain_urls,
        additional_instruction=(
            "Pass only if the provided sources clearly show the hotel's brand/chain affiliation, "
            "and the chain is among the allowed set. Allowed set (case-insensitive, accept common synonyms): "
            f"{', '.join(ALLOWED_CHAINS)}. For IHG, any IHG brand (e.g., InterContinental, Holiday Inn, Crowne Plaza) "
            "counts as IHG; for Marriott, any Marriott brand (e.g., JW Marriott, Courtyard) counts; similar logic "
            "applies to Hilton, Hyatt, Accor, Wyndham, Best Western, and Choice."
        )
    )

    # 3) 4-star or higher rating
    star_node = evaluator.add_leaf(
        id="star_rating",
        desc="The hotel has a 4-star or higher rating",
        parent=node,
        critical=True
    )
    star_claim = (
        f"The hotel '{_safe(hotel.hotel_name)}' is rated 4-star or higher (e.g., 4-star or 5-star). "
        f"If the rating_text is provided in the answer, it is '{_safe(hotel.rating_text)}'."
    )
    await evaluator.verify(
        claim=star_claim,
        node=star_node,
        sources=rating_urls,
        additional_instruction="Accept official ratings or reputable sources (e.g., chain site, government/tourism-star system, or established OTA/wikilike listings) explicitly indicating 4-star or 5-star."
    )

    # 4) City has direct ITA Airways service from Rome
    ita_node = evaluator.add_leaf(
        id="ita_city",
        desc="The hotel is located in a city that has direct ITA Airways flight service from Rome, Italy",
        parent=node,
        critical=True
    )
    ita_claim = (
        f"ITA Airways operates DIRECT (non-stop) flights from Rome Fiumicino (FCO) to the city '{_safe(hotel.city)}' "
        "as of March 2026."
    )
    await evaluator.verify(
        claim=ita_claim,
        node=ita_node,
        sources=ita_urls,
        additional_instruction="Only pass if a route map, timetable, or authoritative page shows non-stop Rome (FCO) service to this city by ITA Airways as of March 2026 (seasonal OK if active in March 2026)."
    )

    # 5) Within 10 miles of the city's international airport
    prox_node = evaluator.add_leaf(
        id="airport_proximity",
        desc="The hotel is within 10 miles of the city's international airport",
        parent=node,
        critical=True
    )
    prox_claim = (
        f"The hotel '{_safe(hotel.hotel_name)}' is within 10 miles (≈16 km) of the international airport "
        f"'{_safe(hotel.airport_name)}' serving '{_safe(hotel.city)}'."
    )
    await evaluator.verify(
        claim=prox_claim,
        node=prox_node,
        sources=proximity_urls,
        additional_instruction="Accept pages that directly state the distance or a map/screenshot that clearly shows the hotel is within 10 miles (~16 km) driving/straight-line distance."
    )

    # 6) Pool
    pool_node = evaluator.add_leaf(
        id="pool",
        desc="The hotel has a swimming pool (indoor or outdoor)",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The hotel '{_safe(hotel.hotel_name)}' has a swimming pool (indoor or outdoor).",
        node=pool_node,
        sources=pool_urls,
        additional_instruction="Accept explicit mentions of a pool on amenities or property pages. Either indoor or outdoor qualifies."
    )

    # 7) Fitness center
    gym_node = evaluator.add_leaf(
        id="fitness_center",
        desc="The hotel has a fitness center or gym facility",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The hotel '{_safe(hotel.hotel_name)}' has a fitness center or gym facility.",
        node=gym_node,
        sources=fitness_urls,
        additional_instruction="Accept mentions like 'fitness center', 'gym', or similar on amenities/property pages."
    )

    # 8) On-site restaurant
    rest_node = evaluator.add_leaf(
        id="restaurant",
        desc="The hotel has at least one on-site restaurant",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The hotel '{_safe(hotel.hotel_name)}' has at least one on-site restaurant.",
        node=rest_node,
        sources=restaurant_urls,
        additional_instruction="On-site dining/restaurant counts; bars without food do not suffice unless they also serve meals."
    )

    # 9) Complimentary WiFi
    wifi_node = evaluator.add_leaf(
        id="free_wifi",
        desc="The hotel offers complimentary WiFi to guests",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The hotel '{_safe(hotel.hotel_name)}' offers complimentary WiFi to guests.",
        node=wifi_node,
        sources=wifi_urls,
        additional_instruction="Free WiFi qualifies; paid-only WiFi does not. If both free and paid tiers exist, free basic WiFi to guests qualifies."
    )

    # 10) Business center
    biz_node = evaluator.add_leaf(
        id="business_center",
        desc="The hotel has business center facilities",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The hotel '{_safe(hotel.hotel_name)}' has a business center (or business services area).",
        node=biz_node,
        sources=business_urls,
        additional_instruction="Accept 'business center', dedicated computer/print services area, or explicitly stated business facilities."
    )

    # 11) Spa or wellness
    spa_node = evaluator.add_leaf(
        id="spa_wellness",
        desc="The hotel offers spa or wellness services",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The hotel '{_safe(hotel.hotel_name)}' offers spa or wellness services.",
        node=spa_node,
        sources=spa_urls,
        additional_instruction="Accept on-site spa, wellness center, sauna/steam with treatments, or comparable wellness services."
    )

    # 12) Suites
    suites_node = evaluator.add_leaf(
        id="suites",
        desc="The hotel offers suite accommodations",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The hotel '{_safe(hotel.hotel_name)}' offers suite accommodations (rooms categorized as suites).",
        node=suites_node,
        sources=suites_urls,
        additional_instruction="Accept explicit room types named 'Suite' (e.g., Junior Suite, Executive Suite, etc.)."
    )

    # 13) Accessibility features
    access_node = evaluator.add_leaf(
        id="accessible",
        desc="The hotel is wheelchair accessible with appropriate accessibility features",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The hotel '{_safe(hotel.hotel_name)}' is wheelchair accessible and provides appropriate accessibility features.",
        node=access_node,
        sources=accessibility_urls,
        additional_instruction="Accept mentions of wheelchair-accessible rooms/entrances, ADA compliance, step-free access, accessible bathrooms, etc."
    )

    # 14) Parking
    park_node = evaluator.add_leaf(
        id="parking",
        desc="The hotel provides parking facilities (paid or complimentary)",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The hotel '{_safe(hotel.hotel_name)}' provides parking facilities (either paid or complimentary).",
        node=park_node,
        sources=parking_urls,
        additional_instruction="Accept valet or self-parking (garage or lot)."
    )

    # 15) Pet policy
    pet_node = evaluator.add_leaf(
        id="pet_policy",
        desc="The hotel has a clearly stated pet policy",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The hotel '{_safe(hotel.hotel_name)}' has a clearly stated pet policy.",
        node=pet_node,
        sources=pet_urls,
        additional_instruction="Any explicit pet policy counts (pet-friendly with fees/limits, or 'no pets allowed'). The key is that a clear policy exists."
    )

    # 16) Cancellation policy (>= 24 hours)
    cancel_node = evaluator.add_leaf(
        id="cancellation",
        desc="The hotel offers a cancellation policy allowing cancellation at least 24 hours before check-in without penalty",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The hotel '{_safe(hotel.hotel_name)}' offers a flexible cancellation policy that allows guests to cancel at least 24 hours before check-in without penalty.",
        node=cancel_node,
        sources=cancellation_urls,
        additional_instruction="Accept typical 'free cancellation until 24 hours before arrival' (or earlier). Policies may vary by rate; a standard/flexible rate policy that allows >=24h no-penalty cancellation qualifies."
    )


# --------------------------------------------------------------------------- #
# Main evaluation entry point                                                 #
# --------------------------------------------------------------------------- #
async def evaluate_answer(
    client: LLMClient,
    answer: str,
    agent_name: str,
    answer_name: str,
    cache: CacheFileSys,
    semaphore: asyncio.Semaphore,
    logger: logging.Logger,
    model: str = "o4-mini",
) -> Dict[str, Any]:
    """
    Evaluate an answer for the hotel criteria task (as of March 2026).
    """
    # Initialize evaluator (root is sequential and CRITICAL per rubric)
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.SEQUENTIAL,
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

    # Record allowed chains in the summary for transparency
    evaluator.add_custom_info(
        info={"allowed_chains": ALLOWED_CHAINS},
        info_type="constraints",
        info_name="allowed_hotel_chains"
    )

    # Extraction
    extraction: HotelAnswerExtraction = await evaluator.extract(
        prompt=prompt_extract_hotel_info(),
        template_class=HotelAnswerExtraction,
        extraction_name="hotel_extraction"
    )

    # Build verification tree according to rubric
    # 1) Required output fields (parallel, critical)
    await build_required_output_section(evaluator, root, extraction)

    # 2) Eligibility criteria (parallel, critical)
    await build_eligibility_section(evaluator, root, extraction)

    # Return structured evaluation summary
    return evaluator.get_summary()