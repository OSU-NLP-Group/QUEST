import asyncio
import logging
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy

# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "airport_hotels_three_cities"
TASK_DESCRIPTION = """
A corporate travel manager is planning accommodations for executive team meetings in three US cities that recently completed major airport terminal expansions or renovations. The hotels must meet specific corporate travel policy requirements.

Find one hotel in each of the following three cities that satisfies ALL the criteria below:

Cities (all three required):
1. Nashville, Tennessee - near Nashville International Airport (BNA), which opened its Concourse D Extension on July 8, 2025, adding 5 gates and 22,000 square feet of concessions space
2. Charlotte, North Carolina - near Charlotte Douglas International Airport (CLT), which completed its $608 million Terminal Lobby Expansion in September 2025, adding 175,000 square feet of new lobby space
3. Houston, Texas - near George Bush Intercontinental Airport (IAH), which opened its Terminal D-West Pier on October 22, 2024, adding 160,000 square feet and 6 new gates

Hotel Requirements (each hotel must meet ALL of these):
- Must belong to one of these four major international hotel chains: Marriott International, Hilton, IHG Hotels & Resorts, or Hyatt Hotels Corporation
- Must be classified as an upscale or luxury tier brand within its parent chain's portfolio (not midscale or budget brands)
- Must qualify as an airport hotel by either: (a) being located within 5 miles of the airport terminal, OR (b) providing complimentary shuttle service to the airport
- Must have on-site meeting or conference facilities available for business events
- Must have a minimum of 200 guest rooms
- Must participate in its parent chain's loyalty rewards program that offers tiered elite status benefits

For each hotel, provide:
- Hotel name and specific brand
- Parent hotel chain
- Brand tier classification (upscale or luxury)
- Exact distance from airport terminal OR confirmation of complimentary shuttle service
- Confirmation of meeting/conference facilities
- Total number of guest rooms
- Name of loyalty program
"""

ALLOWED_CHAINS = [
    "Marriott International",
    "Hilton",
    "IHG Hotels & Resorts",
    "Hyatt Hotels Corporation",
]

CITY_SPECS = {
    "nashville": {
        "city_display": "Nashville",
        "state_display": "Tennessee",
        "airport_code": "BNA",
        "airport_name": "Nashville International Airport",
        "city_node_desc": "Identify one hotel in Nashville, Tennessee near Nashville International Airport (BNA) that meets all requirements",
        "leaf_desc": {
            "location_airport": "Hotel is located in Nashville, TN, and is near Nashville International Airport (BNA), which opened the Concourse D Extension on July 8, 2025, adding 5 gates and 22,000 square feet of concessions space",
            "chain_brand": "Hotel belongs to one of the four major international chains (Marriott International, Hilton, IHG Hotels & Resorts, or Hyatt Hotels Corporation) and is classified as an upscale or luxury tier brand within that chain's portfolio",
            "proximity_transport": "Hotel qualifies as an airport hotel by being located within 5 miles of BNA terminal or by providing complimentary shuttle service to the airport",
            "meeting_facilities": "Hotel has on-site meeting or conference facilities available for business events",
            "room_count": "Hotel has a minimum of 200 guest rooms",
            "loyalty_program": "Hotel participates in its parent chain's loyalty rewards program that offers tiered elite status benefits",
            "operational_status": "Hotel is currently operational and accepting reservations",
        },
        "node_id": "hotel_nashville",
        "leaf_ids": {
            "location_airport": "nashville_location_airport",
            "chain_brand": "nashville_chain_brand",
            "proximity_transport": "nashville_proximity_transport",
            "meeting_facilities": "nashville_meeting_facilities",
            "room_count": "nashville_room_count",
            "loyalty_program": "nashville_loyalty_program",
            "operational_status": "nashville_operational_status",
        }
    },
    "charlotte": {
        "city_display": "Charlotte",
        "state_display": "North Carolina",
        "airport_code": "CLT",
        "airport_name": "Charlotte Douglas International Airport",
        "city_node_desc": "Identify one hotel in Charlotte, North Carolina near Charlotte Douglas International Airport (CLT) that meets all requirements",
        "leaf_desc": {
            "location_airport": "Hotel is located in Charlotte, NC, and is near Charlotte Douglas International Airport (CLT), which completed the $608 million Terminal Lobby Expansion in September 2025, adding 175,000 square feet of new lobby space",
            "chain_brand": "Hotel belongs to one of the four major international chains (Marriott International, Hilton, IHG Hotels & Resorts, or Hyatt Hotels Corporation) and is classified as an upscale or luxury tier brand within that chain's portfolio",
            "proximity_transport": "Hotel qualifies as an airport hotel by being located within 5 miles of CLT terminal or by providing complimentary shuttle service to the airport",
            "meeting_facilities": "Hotel has on-site meeting or conference facilities available for business events",
            "room_count": "Hotel has a minimum of 200 guest rooms",
            "loyalty_program": "Hotel participates in its parent chain's loyalty rewards program that offers tiered elite status benefits",
            "operational_status": "Hotel is currently operational and accepting reservations",
        },
        "node_id": "hotel_charlotte",
        "leaf_ids": {
            "location_airport": "charlotte_location_airport",
            "chain_brand": "charlotte_chain_brand",
            "proximity_transport": "charlotte_proximity_transport",
            "meeting_facilities": "charlotte_meeting_facilities",
            "room_count": "charlotte_room_count",
            "loyalty_program": "charlotte_loyalty_program",
            "operational_status": "charlotte_operational_status",
        }
    },
    "houston": {
        "city_display": "Houston",
        "state_display": "Texas",
        "airport_code": "IAH",
        "airport_name": "George Bush Intercontinental Airport",
        "city_node_desc": "Identify one hotel in Houston, Texas near George Bush Intercontinental Airport (IAH) that meets all requirements",
        "leaf_desc": {
            "location_airport": "Hotel is located in Houston, TX, and is near George Bush Intercontinental Airport (IAH), which opened the Terminal D-West Pier on October 22, 2024, adding 160,000 square feet and 6 new gates",
            "chain_brand": "Hotel belongs to one of the four major international chains (Marriott International, Hilton, IHG Hotels & Resorts, or Hyatt Hotels Corporation) and is classified as an upscale or luxury tier brand within that chain's portfolio",
            "proximity_transport": "Hotel qualifies as an airport hotel by being located within 5 miles of IAH terminal or by providing complimentary shuttle service to the airport",
            "meeting_facilities": "Hotel has on-site meeting or conference facilities available for business events",
            "room_count": "Hotel has a minimum of 200 guest rooms",
            "loyalty_program": "Hotel participates in its parent chain's loyalty rewards program that offers tiered elite status benefits",
            "operational_status": "Hotel is currently operational and accepting reservations",
        },
        "node_id": "hotel_houston",
        "leaf_ids": {
            "location_airport": "houston_location_airport",
            "chain_brand": "houston_chain_brand",
            "proximity_transport": "houston_proximity_transport",
            "meeting_facilities": "houston_meeting_facilities",
            "room_count": "houston_room_count",
            "loyalty_program": "houston_loyalty_program",
            "operational_status": "houston_operational_status",
        }
    },
}


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class HotelItem(BaseModel):
    # Identity and classification
    hotel_name: Optional[str] = None
    brand: Optional[str] = None
    parent_chain: Optional[str] = None
    brand_tier: Optional[str] = None  # expected values: "upscale" or "luxury" (allow synonyms like upper-upscale mapped to upscale)

    # Location / proximity
    city: Optional[str] = None
    state: Optional[str] = None
    airport_code: Optional[str] = None
    airport_name: Optional[str] = None
    distance_miles: Optional[str] = None  # free-form string, e.g., "2.5 miles", "3 mi"
    complimentary_shuttle: Optional[str] = None  # textual confirmation if free/complimentary shuttle is offered

    # Facilities and capacity
    meeting_facilities: Optional[str] = None  # textual confirmation of meeting/conference facilities
    room_count: Optional[str] = None  # free-form string with number

    # Loyalty
    loyalty_program: Optional[str] = None

    # Operational status
    operational_status: Optional[str] = None  # textual confirmation that booking/reservations are open

    # Source URLs attributed in the answer (explicit URLs only)
    sources_main: List[str] = Field(default_factory=list)        # hotel homepage or official page
    sources_brand: List[str] = Field(default_factory=list)       # brand/chain info pages supporting chain/tier
    sources_proximity: List[str] = Field(default_factory=list)   # map links or shuttle pages
    sources_meeting: List[str] = Field(default_factory=list)     # meetings/events/fact sheet pages
    sources_rooms: List[str] = Field(default_factory=list)       # room count support (fact sheet/datasheet)
    sources_loyalty: List[str] = Field(default_factory=list)     # loyalty program pages
    sources_status: List[str] = Field(default_factory=list)      # booking page or status info


class TravelHotelsExtraction(BaseModel):
    nashville: Optional[HotelItem] = None
    charlotte: Optional[HotelItem] = None
    houston: Optional[HotelItem] = None


# --------------------------------------------------------------------------- #
# Extraction prompt helpers                                                   #
# --------------------------------------------------------------------------- #
def prompt_extract_hotels() -> str:
    return """
Extract exactly one qualifying hotel for each of the three required cities: Nashville (Tennessee), Charlotte (North Carolina), and Houston (Texas). If the answer mentions multiple candidates for a city, pick the first one that best meets all requirements. If a city has no valid hotel mentioned, return null for that city.

For each city's selected hotel, extract the following fields strictly from the answer text:

Identity and classification:
- hotel_name: the hotel's official name as written in the answer
- brand: the specific brand (e.g., "Hyatt Regency", "Hilton", "JW Marriott")
- parent_chain: the parent hotel chain ("Marriott International", "Hilton", "IHG Hotels & Resorts", or "Hyatt Hotels Corporation")
- brand_tier: classify the brand as "upscale" or "luxury" (map synonyms like "upper-upscale", "upper upscale", "premium" to "upscale"; "luxury", "ultra-luxury" to "luxury"). If unclear, use the answer's stated tier.

Location / proximity:
- city: city name of the hotel
- state: state name (full name, not abbreviation)
- airport_code: airport IATA code (BNA, CLT, or IAH) associated with the city if explicitly referenced
- airport_name: airport full name if referenced
- distance_miles: the stated distance to the airport terminal in miles, if provided in the answer (free-form string, keep as-is)
- complimentary_shuttle: textual confirmation if the hotel provides complimentary (free) airport shuttle service, if mentioned

Facilities and capacity:
- meeting_facilities: textual confirmation the hotel has on-site meeting or conference facilities
- room_count: total number of guest rooms (as presented in the answer, keep as text)

Loyalty:
- loyalty_program: the name of the chain's loyalty rewards program

Operational:
- operational_status: textual confirmation that the hotel is open and accepting reservations

Source URLs (EXTRACT ONLY EXPLICIT URLS PRESENT IN THE ANSWER; do not invent):
- sources_main: official hotel homepage(s) or property page(s)
- sources_brand: chain/brand info pages supporting chain affiliation and tier classification
- sources_proximity: map links (e.g., Google Maps) showing distance or official pages stating complimentary shuttle
- sources_meeting: pages that show meeting/conference facilities (meetings/events/fact sheet)
- sources_rooms: pages that show room count (fact sheet/datasheet/press)
- sources_loyalty: loyalty program page that shows tiered elite benefits
- sources_status: booking/reservations page indicating availability

Return a JSON object with keys: "nashville", "charlotte", "houston". Each key maps to a HotelItem object as defined, or null if not provided in the answer.
"""


# --------------------------------------------------------------------------- #
# Utility helpers                                                             #
# --------------------------------------------------------------------------- #
def _safe_name(item: Optional[HotelItem], fallback: str) -> str:
    if item and item.hotel_name:
        return item.hotel_name
    return fallback


def _join_sources(*args: List[str]) -> List[str]:
    out: List[str] = []
    for lst in args:
        for u in lst or []:
            if isinstance(u, str):
                u = u.strip()
                if u and u not in out:
                    out.append(u)
    return out


def _source_required_instruction(sources: List[str]) -> str:
    if sources:
        return "Use only the provided URL evidence to judge the claim."
    return (
        "Source-grounding policy: No URLs were extracted for this check. "
        "You must judge this claim as NOT SUPPORTED/INCORRECT unless explicit evidence is visible in the provided webpages. "
        "Do not accept claims based solely on the answer text."
    )


# --------------------------------------------------------------------------- #
# Verification per-city                                                       #
# --------------------------------------------------------------------------- #
async def verify_city_hotel(
    evaluator: Evaluator,
    parent_node,
    city_key: str,
    extracted: TravelHotelsExtraction,
) -> None:
    spec = CITY_SPECS[city_key]
    city_node = evaluator.add_parallel(
        id=spec["node_id"],
        desc=spec["city_node_desc"],
        parent=parent_node,
        critical=True  # Each city is essential to satisfy the task
    )

    item: Optional[HotelItem] = getattr(extracted, city_key)
    name_for_text = _safe_name(item, f"the selected hotel for {spec['city_display']}")

    # Prepare common source pools
    main_sources = _join_sources(*(item.sources_main if item else []))
    brand_sources = _join_sources(*(item.sources_brand if item else []))
    prox_sources = _join_sources(*(item.sources_proximity if item else []))
    meeting_sources = _join_sources(*(item.sources_meeting if item else []))
    rooms_sources = _join_sources(*(item.sources_rooms if item else []))
    loyalty_sources = _join_sources(*(item.sources_loyalty if item else []))
    status_sources = _join_sources(*(item.sources_status if item else []))

    # 1) Location & airport association
    node_loc = evaluator.add_leaf(
        id=spec["leaf_ids"]["location_airport"],
        desc=spec["leaf_desc"]["location_airport"],
        parent=city_node,
        critical=True
    )
    claim_loc = (
        f"The hotel '{name_for_text}' is located in {spec['city_display']}, {spec['state_display']}, "
        f"and is associated with or serves {spec['airport_name']} ({spec['airport_code']})."
    )
    sources_loc = _join_sources(main_sources, prox_sources)
    await evaluator.verify(
        claim=claim_loc,
        node=node_loc,
        sources=sources_loc if sources_loc else None,
        additional_instruction=(
            "Confirm the city is correct and the hotel is an airport-area property for the specified airport. "
            "Ignore the airport expansion details; they are context only. "
            + _source_required_instruction(sources_loc)
        ),
    )

    # 2) Chain & brand tier (allowed chains + upscale/luxury)
    node_chain = evaluator.add_leaf(
        id=spec["leaf_ids"]["chain_brand"],
        desc=spec["leaf_desc"]["chain_brand"],
        parent=city_node,
        critical=True
    )
    chain_txt = item.parent_chain if (item and item.parent_chain) else "its parent chain"
    brand_txt = item.brand if (item and item.brand) else "the stated brand"
    tier_txt = item.brand_tier if (item and item.brand_tier) else "upscale or luxury"
    claim_chain = (
        f"The hotel '{name_for_text}' is a {brand_txt} property under {chain_txt}, "
        f"and this brand is classified as an {tier_txt} tier (treat 'upper-upscale' as 'upscale'). "
        f"The parent chain must be one of: {', '.join(ALLOWED_CHAINS)}."
    )
    sources_chain = _join_sources(brand_sources, main_sources)
    await evaluator.verify(
        claim=claim_chain,
        node=node_chain,
        sources=sources_chain if sources_chain else None,
        additional_instruction=(
            "Verify chain affiliation and that the brand's tier is upscale or luxury. "
            "Accept synonyms: 'upper-upscale' or 'upper upscale' count as 'upscale'. "
            f"The parent chain must be exactly one of: {', '.join(ALLOWED_CHAINS)}. "
            + _source_required_instruction(sources_chain)
        ),
    )

    # 3) Airport proximity OR complimentary shuttle
    node_prox = evaluator.add_leaf(
        id=spec["leaf_ids"]["proximity_transport"],
        desc=spec["leaf_desc"]["proximity_transport"],
        parent=city_node,
        critical=True
    )
    claim_prox = (
        f"For '{name_for_text}', at least one is true: "
        f"(a) it is located within 5 miles of {spec['airport_code']} terminals, "
        f"OR (b) it provides complimentary (free) shuttle service to/from {spec['airport_code']}."
    )
    sources_prox = _join_sources(prox_sources, main_sources)
    await evaluator.verify(
        claim=claim_prox,
        node=node_prox,
        sources=sources_prox if sources_prox else None,
        additional_instruction=(
            "It suffices to find clear evidence for either condition. "
            "Distance may be shown via an official site statement or a map link. "
            "Shuttle must be complimentary/free and airport-specific. "
            + _source_required_instruction(sources_prox)
        ),
    )

    # 4) Meeting/conference facilities
    node_meet = evaluator.add_leaf(
        id=spec["leaf_ids"]["meeting_facilities"],
        desc=spec["leaf_desc"]["meeting_facilities"],
        parent=city_node,
        critical=True
    )
    claim_meet = f"The hotel '{name_for_text}' has on-site meeting or conference facilities suitable for business events."
    sources_meet = _join_sources(meeting_sources, main_sources)
    await evaluator.verify(
        claim=claim_meet,
        node=node_meet,
        sources=sources_meet if sources_meet else None,
        additional_instruction=(
            "Look for 'Meetings', 'Events', 'Conference', or similar pages/fact sheets that confirm on-site facilities. "
            + _source_required_instruction(sources_meet)
        ),
    )

    # 5) Room count >= 200
    node_rooms = evaluator.add_leaf(
        id=spec["leaf_ids"]["room_count"],
        desc=spec["leaf_desc"]["room_count"],
        parent=city_node,
        critical=True
    )
    claim_rooms = f"The hotel '{name_for_text}' has at least 200 guest rooms (rooms >= 200)."
    sources_rooms_all = _join_sources(rooms_sources, main_sources)
    await evaluator.verify(
        claim=claim_rooms,
        node=node_rooms,
        sources=sources_rooms_all if sources_rooms_all else None,
        additional_instruction=(
            "Accept if the source indicates a room count of 200 or more (e.g., 250 rooms). "
            + _source_required_instruction(sources_rooms_all)
        ),
    )

    # 6) Loyalty program with tiered elite benefits
    node_loyal = evaluator.add_leaf(
        id=spec["leaf_ids"]["loyalty_program"],
        desc=spec["leaf_desc"]["loyalty_program"],
        parent=city_node,
        critical=True
    )
    loyalty_name = item.loyalty_program if (item and item.loyalty_program) else "the chain's loyalty program"
    claim_loyal = (
        f"The hotel '{name_for_text}' participates in {loyalty_name}, the chain's loyalty program, "
        "which offers tiered elite status benefits."
    )
    sources_loyal_all = _join_sources(loyalty_sources, brand_sources, main_sources)
    await evaluator.verify(
        claim=claim_loyal,
        node=node_loyal,
        sources=sources_loyal_all if sources_loyal_all else None,
        additional_instruction=(
            "Look for official loyalty program pages outlining elite tiers/benefits and the brand's participation. "
            + _source_required_instruction(sources_loyal_all)
        ),
    )

    # 7) Operational status (accepting reservations)
    node_oper = evaluator.add_leaf(
        id=spec["leaf_ids"]["operational_status"],
        desc=spec["leaf_desc"]["operational_status"],
        parent=city_node,
        critical=True
    )
    claim_oper = f"The hotel '{name_for_text}' is currently open and accepting reservations for upcoming dates."
    sources_oper_all = _join_sources(status_sources, main_sources)
    await evaluator.verify(
        claim=claim_oper,
        node=node_oper,
        sources=sources_oper_all if sources_oper_all else None,
        additional_instruction=(
            "Booking/reservations page showing availability counts as confirmation. "
            + _source_required_instruction(sources_oper_all)
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
    model: str = "o4-mini"
) -> Dict:
    """
    Evaluate an answer for the three-city airport hotel task.
    """
    # Initialize evaluator (root is non-critical by default; city nodes are set critical)
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Cities evaluated in parallel; each city node is critical
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

    # Extraction
    extracted_hotels = await evaluator.extract(
        prompt=prompt_extract_hotels(),
        template_class=TravelHotelsExtraction,
        extraction_name="selected_hotels_by_city"
    )

    # Optional GT/context info
    evaluator.add_ground_truth({
        "required_cities": ["Nashville (BNA)", "Charlotte (CLT)", "Houston (IAH)"],
        "allowed_chains": ALLOWED_CHAINS,
        "policy_requirements": [
            "Upscale or luxury tier",
            "Airport hotel (<=5 miles or complimentary shuttle)",
            "On-site meeting facilities",
            ">= 200 rooms",
            "Loyalty program with tiered elite benefits",
            "Operational (accepting reservations)"
        ]
    })

    # Per-city verification (each city node is critical to satisfy the task)
    await verify_city_hotel(evaluator, root, "nashville", extracted_hotels)
    await verify_city_hotel(evaluator, root, "charlotte", extracted_hotels)
    await verify_city_hotel(evaluator, root, "houston", extracted_hotels)

    # Return summary
    return evaluator.get_summary()