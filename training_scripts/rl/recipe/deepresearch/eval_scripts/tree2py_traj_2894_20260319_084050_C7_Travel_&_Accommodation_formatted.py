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
TASK_ID = "aaa_four_diamond_florida_hotel"
TASK_DESCRIPTION = """
Identify one AAA Four Diamond hotel in Florida that meets as many as possible of the following amenity and service requirements. For the hotel you identify, provide: (1) The hotel's official name, (2) The city where it is located, (3) The hotel's official website URL, and (4) Verification details for each amenity criterion met. Required amenity and service criteria to evaluate: 
1. The hotel must hold an AAA Four Diamond designation as documented in the official AAA Four Diamond Hotels publication for 2025 or 2026. 
2. The hotel must have a stated pet policy (whether pets are allowed or not, with any associated fees clearly documented). 
3. The hotel must offer on-site parking with clearly stated daily parking fees (either self-parking or valet parking). 
4. The hotel must have a fitness center with operating hours clearly stated. 
5. The hotel must have information about breakfast availability (whether complimentary, for purchase, or through on-site restaurants). 
6. The hotel must have a business center or business services available to guests. 
7. The hotel must have a swimming pool facility (indoor or outdoor) with operating information. 
8. The hotel must provide information about wheelchair accessible rooms (ADA-compliant accommodations). 
9. The hotel must have information about laundry services (self-service, full-service, or valet laundry). 
10. The hotel must provide information about shuttle services or transportation options. 
11. The hotel must have meeting facilities or event space with stated capacity information. 
12. The hotel must have clearly stated check-in and check-out times on its website. 
13. The hotel must have a cancellation policy documented on its website or booking platform. 
14. The hotel must provide information about concierge services or guest services available.
"""


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class AmenitySupport(BaseModel):
    details: Optional[str] = None
    urls: List[str] = Field(default_factory=list)


class HotelExtraction(BaseModel):
    hotel_name: Optional[str] = None
    city: Optional[str] = None
    official_website_url: Optional[str] = None

    # AAA sources (should be official AAA publication pages for 2025 or 2026)
    aaa_sources: List[str] = Field(default_factory=list)

    # Amenities and services
    pet_policy: AmenitySupport = Field(default_factory=AmenitySupport)
    parking: AmenitySupport = Field(default_factory=AmenitySupport)
    fitness_center: AmenitySupport = Field(default_factory=AmenitySupport)
    breakfast: AmenitySupport = Field(default_factory=AmenitySupport)
    business_center: AmenitySupport = Field(default_factory=AmenitySupport)
    swimming_pool: AmenitySupport = Field(default_factory=AmenitySupport)
    accessible_rooms: AmenitySupport = Field(default_factory=AmenitySupport)
    laundry_services: AmenitySupport = Field(default_factory=AmenitySupport)
    shuttle_transportation: AmenitySupport = Field(default_factory=AmenitySupport)
    meeting_facilities: AmenitySupport = Field(default_factory=AmenitySupport)
    check_in_out_times: AmenitySupport = Field(default_factory=AmenitySupport)
    cancellation_policy: AmenitySupport = Field(default_factory=AmenitySupport)
    concierge_services: AmenitySupport = Field(default_factory=AmenitySupport)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_hotel_info() -> str:
    return """
    Extract the single Florida hotel identified in the answer along with verification sources for each amenity criterion.
    Return a JSON object matching the following schema:

    - hotel_name: The official hotel name as given in the answer (string or null if missing)
    - city: The city in Florida where the hotel is located, as given in the answer (string or null if missing)
    - official_website_url: The official hotel website URL provided in the answer (string URL or null if missing)

    - aaa_sources: An array of URLs cited in the answer that specifically support the hotel's AAA Four Diamond designation.
      IMPORTANT: Extract only official AAA publications/pages for 2025 or 2026 if present in the answer. If the answer cites non-AAA pages for AAA designation, still include them, but do not invent URLs.

    For each amenity below, extract:
      - details: A short phrase or sentence summarizing what the answer claims (copy from the answer; do not invent).
      - urls: All URLs cited in the answer that support this amenity; include sub-pages of the hotel's site or booking pages if explicitly cited.

    Amenity fields to extract:
      - pet_policy
      - parking
      - fitness_center
      - breakfast
      - business_center
      - swimming_pool
      - accessible_rooms
      - laundry_services
      - shuttle_transportation
      - meeting_facilities
      - check_in_out_times
      - cancellation_policy
      - concierge_services

    RULES:
    - Extract only information explicitly present in the answer text. Do not infer or add missing URLs.
    - For URL fields, include only valid URLs exactly as written (plain URLs or markdown links).
    - If no URL is provided for an amenity, return an empty array for that amenity's urls.
    - If the amenity itself is not mentioned, set its details to null and urls to an empty array.

    Output strictly as JSON according to the schema.
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def _unique_urls(urls: List[str]) -> List[str]:
    seen = set()
    result = []
    for u in urls:
        if not u:
            continue
        if u not in seen:
            seen.add(u)
            result.append(u)
    return result


def collect_sources_with_fallback(amenity: AmenitySupport, website_url: Optional[str]) -> List[str]:
    """
    Collect sources for an amenity, using amenity.urls primarily and falling back to the official hotel website if provided.
    """
    urls = list(amenity.urls or [])
    if website_url:
        urls.append(website_url)
    return _unique_urls(urls)


# Map amenity node IDs to extraction attributes
AMENITY_MAP = {
    "Pet_Policy": ("pet_policy", "Verify that the page presents a clear pet policy stating whether pets are allowed or not, and any applicable fees. 'No pets allowed' also qualifies as a stated policy."),
    "Parking_Availability": ("parking", "Verify that on-site parking is offered AND that daily parking fee(s) are clearly stated (self-parking or valet). Complimentary parking qualifies as a clearly stated fee of $0 if explicitly said."),
    "Fitness_Center": ("fitness_center", "Verify that a fitness center/gym exists and its operating hours are clearly stated. '24-hour' counts as hours."),
    "Breakfast_Information": ("breakfast", "Verify that the page provides information about breakfast availability (complimentary, for purchase, or via on-site restaurants)."),
    "Business_Center": ("business_center", "Verify that a business center or guest business services (e.g., printing) are available."),
    "Swimming_Pool": ("swimming_pool", "Verify that a pool exists and operating information is provided (hours or seasonal availability)."),
    "Wheelchair_Accessible_Rooms": ("accessible_rooms", "Verify that ADA-compliant/wheelchair accessible rooms are available."),
    "Laundry_Services": ("laundry_services", "Verify that laundry options are available (self-service guest laundry, valet laundry, or dry cleaning)."),
    "Shuttle_Transportation": ("Shuttle_Transportation", "Verify that the hotel provides shuttle services or transportation options (e.g., airport shuttle). Generic public transit suggestions alone are insufficient."),
    "Meeting_Facilities": ("meeting_facilities", "Verify that meeting/event space exists AND at least one capacity figure (people or sq ft/m²) is stated."),
    "Check_In_Out_Times": ("check_in_out_times", "Verify that both check-in and check-out times are clearly stated."),
    "Cancellation_Policy": ("cancellation_policy", "Verify that a cancellation policy is documented (property or brand policy; booking engine pages are acceptable)."),
    "Concierge_Services": ("concierge_services", "Verify that concierge or guest services are offered (e.g., concierge desk, guest services)."),
}


def build_amenity_claim(amenity_id: str, hotel_name: Optional[str], city: Optional[str], details: Optional[str]) -> str:
    context = []
    if hotel_name:
        context.append(hotel_name)
    if city:
        context.append(city)
    where = " in ".join(context) if context else "the hotel"

    base_texts = {
        "Pet_Policy": f"This page provides a clear pet policy for {where}.",
        "Parking_Availability": f"This page shows that {where} offers on-site parking with clearly stated daily fee(s) (self or valet).",
        "Fitness_Center": f"This page shows that {where} has a fitness center with stated operating hours (including '24-hour' if applicable).",
        "Breakfast_Information": f"This page shows information about breakfast availability for {where} (complimentary or paid).",
        "Business_Center": f"This page shows that {where} provides a business center or business services for guests.",
        "Swimming_Pool": f"This page shows that {where} has a swimming pool and provides operating information (hours or seasonal).",
        "Wheelchair_Accessible_Rooms": f"This page shows that {where} offers ADA-compliant/wheelchair accessible rooms.",
        "Laundry_Services": f"This page shows that {where} offers laundry services or facilities (self-service, valet, or dry cleaning).",
        "Shuttle_Transportation": f"This page shows that {where} provides shuttle services or transportation options.",
        "Meeting_Facilities": f"This page shows that {where} has meeting/event space with at least one capacity figure stated.",
        "Check_In_Out_Times": f"This page shows clearly stated check-in and check-out times for {where}.",
        "Cancellation_Policy": f"This page shows a documented cancellation policy for {where}.",
        "Concierge_Services": f"This page shows that {where} provides concierge or guest services.",
    }
    base = base_texts.get(amenity_id, f"This page supports the amenity for {where}.")
    if details:
        return f"{base} Claimed details: {details}"
    return base


# --------------------------------------------------------------------------- #
# Verification builder                                                        #
# --------------------------------------------------------------------------- #
async def build_and_verify_nodes(evaluator: Evaluator, root_node, ex: HotelExtraction) -> None:
    """
    Build the verification tree nodes according to the rubric and run verifications.
    """
    # Create a container node to mirror rubric's "Hotel_Selection"
    hotel_node = evaluator.add_parallel(
        id="Hotel_Selection",
        desc="Evaluate whether the solution identifies a valid AAA Four Diamond hotel in Florida and provides required identification information and amenity details",
        parent=root_node,
        critical=False
    )

    # Critical presence checks
    evaluator.add_custom_node(
        result=bool(ex.hotel_name and ex.hotel_name.strip()),
        id="Hotel_Name_Provided",
        desc="The solution provides the hotel's official name",
        parent=hotel_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=bool(ex.city and ex.city.strip()),
        id="City_Location_Provided",
        desc="The solution provides the city where the hotel is located in Florida",
        parent=hotel_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=bool(ex.official_website_url and ex.official_website_url.strip()),
        id="Official_Website_URL_Provided",
        desc="The solution provides the hotel's official website URL",
        parent=hotel_node,
        critical=True
    )

    # AAA Four Diamond verification (Critical)
    aaa_node = evaluator.add_leaf(
        id="AAA_Four_Diamond",
        desc="The hotel holds an AAA Four Diamond designation as documented in official AAA publications for 2025 or 2026",
        parent=hotel_node,
        critical=True
    )
    aaa_claim = (
        f"This source is an official AAA publication/page for 2025 or 2026 and it lists "
        f"{ex.hotel_name or 'the hotel'} as an AAA Four Diamond hotel."
    )
    # Do not add fallback to non-AAA domains here; enforce official AAA evidence
    await evaluator.verify(
        claim=aaa_claim,
        node=aaa_node,
        sources=ex.aaa_sources,
        additional_instruction=(
            "Accept only if the page is clearly an official AAA source (e.g., on an AAA-owned domain such as aaa.com or newsroom.aaa.com, "
            "or an official AAA PDF/publication) and explicitly indicates a Four Diamond designation for 2025 or 2026. "
            "Mentions for other years do NOT satisfy this requirement."
        )
    )

    # Prepare amenity verifications (non-critical leaves)
    amenity_verifications: List[tuple[str, List[str] | str | None, Any, Optional[str]]] = []

    # Helper to add one amenity leaf and queued verification
    def queue_amenity(node_id: str, desc: str, amenity_attr: str, add_instr: str):
        node = evaluator.add_leaf(
            id=node_id,
            desc=desc,
            parent=hotel_node,
            critical=False
        )
        amenity_obj: AmenitySupport = getattr(ex, amenity_attr, AmenitySupport())
        claim = build_amenity_claim(node_id, ex.hotel_name, ex.city, amenity_obj.details)
        sources = collect_sources_with_fallback(amenity_obj, ex.official_website_url)
        # If no sources at all, leave as [] so verification will likely fail (no evidence)
        amenity_verifications.append((
            claim,
            sources if sources else None,
            node,
            add_instr
        ))

    # Queue all amenity checks according to rubric
    for node_id, (attr, add_instr) in AMENITY_MAP.items():
        queue_amenity(
            node_id=node_id,
            desc={
                "Pet_Policy": "The hotel has a stated pet policy with clearly documented information available on the hotel's official website or booking information",
                "Parking_Availability": "The hotel offers on-site parking with clearly stated daily fees documented on the hotel's official website",
                "Fitness_Center": "The hotel has a fitness center with stated operating hours documented on the hotel's official website or amenities list",
                "Breakfast_Information": "The hotel provides information about breakfast availability documented on the hotel's official website or booking information",
                "Business_Center": "The hotel has a business center with services documented on the hotel's official website or amenities list",
                "Swimming_Pool": "The hotel has a swimming pool facility with stated operating hours or seasonal availability documented on the hotel's official website",
                "Wheelchair_Accessible_Rooms": "The hotel offers ADA-compliant wheelchair accessible rooms as documented on the hotel's official website or accessibility information",
                "Laundry_Services": "The hotel provides laundry facilities or services documented on the hotel's official website or amenities list",
                "Shuttle_Transportation": "The hotel provides information about shuttle services or transportation options documented on the hotel's official website",
                "Meeting_Facilities": "The hotel has meeting facilities with stated capacity information documented on the hotel's official website or meeting facilities page",
                "Check_In_Out_Times": "The hotel has clearly stated check-in and check-out times documented on the hotel's official website or policies",
                "Cancellation_Policy": "The hotel has a cancellation policy documented on the hotel's official website or booking platform",
                "Concierge_Services": "The hotel provides information about concierge or guest services documented on the hotel's official website or services list",
            }[node_id],
            amenity_attr=attr,
            add_instr=add_instr
        )

    # Execute amenity verifications in parallel
    if amenity_verifications:
        await evaluator.batch_verify(amenity_verifications)


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
    model: str = "o4-mini",
) -> Dict:
    """
    Evaluate an answer for the AAA Four Diamond Florida hotel task.
    """
    # Initialize evaluator
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

    # Extract structured information from the answer
    extracted: HotelExtraction = await evaluator.extract(
        prompt=prompt_extract_hotel_info(),
        template_class=HotelExtraction,
        extraction_name="hotel_extraction",
    )

    # Build verification nodes and run checks
    await build_and_verify_nodes(evaluator, root, extracted)

    # Return summary
    return evaluator.get_summary()