import asyncio
import logging
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


# -----------------------------------------------------------------------------
# Task constants
# -----------------------------------------------------------------------------
TASK_ID = "destin_hotel_search_2026"
TASK_DESCRIPTION = """
I am planning a family vacation to Destin, Florida for June 2026 and need to find a hotel that accommodates our specific needs. Find one hotel that meets ALL of the following requirements: (1) The hotel must be located in Destin, Florida; (2) The hotel must be beachfront or provide direct beach access; (3) The hotel must be available for booking in June 2026; (4) The hotel must allow dogs (pet-friendly policy); (5) The pet fee (if any) must be clearly disclosed; (6) The hotel must offer complimentary breakfast or breakfast included in the room rate; (7) The hotel must provide free parking or parking for $25 or less per day; (8) The hotel must have at least one outdoor swimming pool; (9) The standard check-in time must be 4:00 PM or earlier; (10) The standard check-out time must be 11:00 AM or later; (11) The hotel must provide free WiFi to guests; (12) The hotel must have an on-site restaurant or dining service available (beyond breakfast); (13) The hotel must have a fitness center or gym available to guests; (14) The base nightly rate for a standard room must be under $400 (before taxes and fees); (15) The accommodation must be a traditional hotel property (not a vacation rental or privately-owned condo). For the hotel you identify, provide the hotel's full name, the complete physical address, a link to the hotel's official website or official booking page, and confirmation of how each of the 15 requirements is satisfied with specific details.
"""


# -----------------------------------------------------------------------------
# Data models for extraction
# -----------------------------------------------------------------------------
class CriterionExtraction(BaseModel):
    # A generic container for a single requirement's extracted details
    detail: Optional[str] = None         # e.g., a short sentence the answer claimed
    observed: Optional[str] = None       # e.g., "Check-in: 4 PM", "$20/day parking", "$350 base rate", "Oceanfront"
    sources: List[str] = Field(default_factory=list)  # URLs explicitly cited in the answer for this criterion


class HotelExtraction(BaseModel):
    # High-level hotel info
    hotel_name: Optional[str] = None
    address: Optional[str] = None
    website_url: Optional[str] = None           # Official site if provided in the answer
    booking_url: Optional[str] = None           # Official booking engine or official booking page if provided
    overall_sources: List[str] = Field(default_factory=list)  # Any other URLs cited globally in the answer

    # Per-criterion evidence (names aligned to rubric)
    location_in_destin: Optional[CriterionExtraction] = None
    beach_access: Optional[CriterionExtraction] = None
    june_availability: Optional[CriterionExtraction] = None
    pet_friendly: Optional[CriterionExtraction] = None
    pet_fee_disclosure: Optional[CriterionExtraction] = None
    breakfast_included: Optional[CriterionExtraction] = None
    affordable_parking: Optional[CriterionExtraction] = None
    outdoor_pool: Optional[CriterionExtraction] = None
    checkin_time: Optional[CriterionExtraction] = None
    checkout_time: Optional[CriterionExtraction] = None
    free_wifi: Optional[CriterionExtraction] = None
    dining_options: Optional[CriterionExtraction] = None
    fitness_center: Optional[CriterionExtraction] = None
    room_rate_under_400: Optional[CriterionExtraction] = None
    accommodation_type_hotel: Optional[CriterionExtraction] = None


# -----------------------------------------------------------------------------
# Extraction prompt
# -----------------------------------------------------------------------------
def prompt_extract_destin_hotel() -> str:
    return """
Extract structured information about exactly one hotel proposed in the answer for the Destin, FL June 2026 trip. Follow all rules below strictly.

Return a JSON object with the following fields:
- hotel_name: The hotel's full official name exactly as written in the answer
- address: The complete physical address for the hotel as given in the answer (if present)
- website_url: The official hotel website URL cited in the answer (if any)
- booking_url: The hotel's official booking engine or official booking page URL cited in the answer (if any)
- overall_sources: An array of all URLs mentioned anywhere in the answer that relate to this hotel (policies, amenities, booking, etc.)

For each of the 15 requirements below, also return an object with:
- detail: A concise sentence exactly reflecting what the answer claims for this requirement
- observed: A key value or phrase supporting the claim (e.g., a time like "3 PM", a price like "$20/day", a phrase like "beachfront", a rate like "$349")
- sources: An array of URLs cited in the answer that support this specific requirement (only URLs explicitly present in the answer)

The 15 requirement objects to return (exact field names):
- location_in_destin
- beach_access
- june_availability
- pet_friendly
- pet_fee_disclosure
- breakfast_included
- affordable_parking
- outdoor_pool
- checkin_time
- checkout_time
- free_wifi
- dining_options
- fitness_center
- room_rate_under_400
- accommodation_type_hotel

IMPORTANT URL RULES:
- Only include URLs explicitly present in the answer. Do not invent or infer any URLs.
- Accept plain URLs or markdown links (extract the actual URL).
- If a URL lacks protocol, prepend 'http://'.

If any field is missing in the answer, set it to null (for strings/objects) or [] (for arrays). Do NOT fabricate values.
"""


# -----------------------------------------------------------------------------
# Helper utilities
# -----------------------------------------------------------------------------
def _crit(extracted: HotelExtraction, attr: str) -> CriterionExtraction:
    """Safely get a CriterionExtraction for a given attribute name; return an empty container if absent."""
    obj = getattr(extracted, attr, None)
    return obj if isinstance(obj, CriterionExtraction) else CriterionExtraction()


def _dedup(seq: List[str]) -> List[str]:
    """Stable de-duplication for URLs."""
    seen = set()
    out: List[str] = []
    for s in seq:
        if not s:
            continue
        if s not in seen:
            out.append(s)
            seen.add(s)
    return out


def gather_sources_for(extracted: HotelExtraction, criterion_attr: str) -> List[str]:
    """Combine per-criterion sources with global/official URLs for robust verification."""
    c = _crit(extracted, criterion_attr)
    urls: List[str] = []
    # Per-criterion
    urls.extend(c.sources or [])
    # Official/overall
    if extracted.website_url:
        urls.append(extracted.website_url)
    if extracted.booking_url:
        urls.append(extracted.booking_url)
    urls.extend(extracted.overall_sources or [])
    return _dedup(urls)


def hotel_brief(extracted: HotelExtraction) -> str:
    name = extracted.hotel_name or "Unknown Hotel"
    addr = extracted.address or "Unknown address"
    return f"Hotel: {name}; Address: {addr}."


# -----------------------------------------------------------------------------
# Verification builders
# -----------------------------------------------------------------------------
async def verify_requirements(evaluator: Evaluator, parent_node, extracted: HotelExtraction) -> None:
    """
    Build and run all leaf verifications under the critical parallel node 'DestinHotelSearch'.
    Each leaf corresponds to one rubric requirement and is verified with URL evidence when available.
    """
    context_note = hotel_brief(extracted)

    # 1) Location in Destin, FL
    node_1 = evaluator.add_leaf(
        id="LocationInDestin",
        desc="Hotel must be located in Destin, Florida",
        parent=parent_node,
        critical=True,
    )
    claim_1 = f"The hotel's physical location is in Destin, Florida. {context_note}"
    await evaluator.verify(
        claim=claim_1,
        node=node_1,
        sources=gather_sources_for(extracted, "location_in_destin"),
        additional_instruction="Confirm that the hotel's city is 'Destin' (not Miramar Beach, Fort Walton Beach, Okaloosa Island, or neighboring towns). Prefer the official website's address/policy pages."
    )

    # 2) Beachfront or direct beach access
    node_2 = evaluator.add_leaf(
        id="BeachAccess",
        desc="Hotel must be beachfront or provide direct beach access",
        parent=parent_node,
        critical=True,
    )
    c2 = _crit(extracted, "beach_access")
    claim_2 = f"The hotel is beachfront or provides direct beach access. Stated: {c2.observed or c2.detail or 'unspecified'}. {context_note}"
    await evaluator.verify(
        claim=claim_2,
        node=node_2,
        sources=gather_sources_for(extracted, "beach_access"),
        additional_instruction="Look for phrases like 'beachfront', 'on the beach', 'direct beach access', or an on-property private beach. Proximity without access does not satisfy."
    )

    # 3) Availability in June 2026
    node_3 = evaluator.add_leaf(
        id="JuneAvailability",
        desc="Hotel must be available for booking in June 2026",
        parent=parent_node,
        critical=True,
    )
    c3 = _crit(extracted, "june_availability")
    claim_3 = f"The hotel has availability for at least some dates in June 2026 (any continuous stay length is acceptable). Example noted: {c3.observed or c3.detail or 'unspecified'}."
    await evaluator.verify(
        claim=claim_3,
        node=node_3,
        sources=gather_sources_for(extracted, "june_availability"),
        additional_instruction="Verify via the official booking engine or a clearly authoritative booking page that some dates in June 2026 can be booked (rooms not sold out)."
    )

    # 4) Pet-friendly (allows dogs)
    node_4 = evaluator.add_leaf(
        id="PetFriendly",
        desc="Hotel must allow dogs (pet-friendly policy)",
        parent=parent_node,
        critical=True,
    )
    c4 = _crit(extracted, "pet_friendly")
    claim_4 = f"The hotel allows dogs per its pet policy. Stated: {c4.observed or c4.detail or 'unspecified'}."
    await evaluator.verify(
        claim=claim_4,
        node=node_4,
        sources=gather_sources_for(extracted, "pet_friendly"),
        additional_instruction="Confirm an official pet policy indicating dogs are allowed. If only cats or service animals are allowed, it does not satisfy."
    )

    # 5) Pet fee disclosure
    node_5 = evaluator.add_leaf(
        id="PetFeeDisclosure",
        desc="Pet fee amount must be disclosed if applicable",
        parent=parent_node,
        critical=True,
    )
    c5 = _crit(extracted, "pet_fee_disclosure")
    claim_5 = f"The hotel's pet fee is clearly disclosed (or explicitly $0/no fee). Stated: {c5.observed or c5.detail or 'unspecified'}."
    await evaluator.verify(
        claim=claim_5,
        node=node_5,
        sources=gather_sources_for(extracted, "pet_fee_disclosure"),
        additional_instruction="Accept if the policy clearly states a specific amount (e.g., '$100 per stay' or '$50 per night per pet') or explicitly says no fee."
    )

    # 6) Breakfast included/complimentary
    node_6 = evaluator.add_leaf(
        id="BreakfastIncluded",
        desc="Hotel must offer complimentary breakfast or breakfast included in room rate",
        parent=parent_node,
        critical=True,
    )
    c6 = _crit(extracted, "breakfast_included")
    claim_6 = f"The hotel offers complimentary breakfast or breakfast included in the room rate. Stated: {c6.observed or c6.detail or 'unspecified'}."
    await evaluator.verify(
        claim=claim_6,
        node=node_6,
        sources=gather_sources_for(extracted, "breakfast_included"),
        additional_instruction="Look for 'complimentary breakfast', 'free breakfast', or 'breakfast included'. A paid breakfast add-on does not satisfy."
    )

    # 7) Parking <= $25/day or free
    node_7 = evaluator.add_leaf(
        id="AffordableParking",
        desc="Hotel must provide free parking or parking for $25 or less per day",
        parent=parent_node,
        critical=True,
    )
    c7 = _crit(extracted, "affordable_parking")
    claim_7 = f"The hotel provides free parking or charges $25/day or less for guest parking. Actual: {c7.observed or c7.detail or 'unspecified'}."
    await evaluator.verify(
        claim=claim_7,
        node=node_7,
        sources=gather_sources_for(extracted, "affordable_parking"),
        additional_instruction="If multiple parking options exist (self/valet), at least one guest-eligible option must be free or ≤ $25/day. Ignore unrelated offsite public parking prices."
    )

    # 8) Outdoor pool
    node_8 = evaluator.add_leaf(
        id="OutdoorPool",
        desc="Hotel must have at least one outdoor swimming pool",
        parent=parent_node,
        critical=True,
    )
    c8 = _crit(extracted, "outdoor_pool")
    claim_8 = f"The hotel has at least one outdoor swimming pool. Stated: {c8.observed or c8.detail or 'unspecified'}."
    await evaluator.verify(
        claim=claim_8,
        node=node_8,
        sources=gather_sources_for(extracted, "outdoor_pool"),
        additional_instruction="Confirm it's an outdoor pool on-site; an indoor-only pool does not satisfy."
    )

    # 9) Check-in time <= 4:00 PM
    node_9 = evaluator.add_leaf(
        id="CheckInTime",
        desc="Standard check-in time must be 4:00 PM or earlier",
        parent=parent_node,
        critical=True,
    )
    c9 = _crit(extracted, "checkin_time")
    claim_9 = f"The hotel's standard check-in time is 4:00 PM or earlier. Actual stated time: {c9.observed or c9.detail or 'unspecified'}."
    await evaluator.verify(
        claim=claim_9,
        node=node_9,
        sources=gather_sources_for(extracted, "checkin_time"),
        additional_instruction="Evaluate the standard check-in time (not early check-in privileges). Times such as 3 PM, 2 PM satisfy; 5 PM does not."
    )

    # 10) Check-out time >= 11:00 AM
    node_10 = evaluator.add_leaf(
        id="CheckOutTime",
        desc="Standard check-out time must be 11:00 AM or later",
        parent=parent_node,
        critical=True,
    )
    c10 = _crit(extracted, "checkout_time")
    claim_10 = f"The hotel's standard check-out time is 11:00 AM or later. Actual stated time: {c10.observed or c10.detail or 'unspecified'}."
    await evaluator.verify(
        claim=claim_10,
        node=node_10,
        sources=gather_sources_for(extracted, "checkout_time"),
        additional_instruction="Evaluate the standard check-out time (not late checkout by request). Times like 11 AM, 12 PM satisfy; 10 AM does not."
    )

    # 11) Free WiFi
    node_11 = evaluator.add_leaf(
        id="FreeWiFi",
        desc="Hotel must provide free WiFi to guests",
        parent=parent_node,
        critical=True,
    )
    c11 = _crit(extracted, "free_wifi")
    claim_11 = f"The hotel provides free WiFi to guests. Stated: {c11.observed or c11.detail or 'unspecified'}."
    await evaluator.verify(
        claim=claim_11,
        node=node_11,
        sources=gather_sources_for(extracted, "free_wifi"),
        additional_instruction="Confirm WiFi is complimentary for hotel guests (in-room or property-wide). Paid-only WiFi does not satisfy."
    )

    # 12) On-site dining options (beyond breakfast)
    node_12 = evaluator.add_leaf(
        id="DiningOptions",
        desc="Hotel must have on-site restaurant or dining service available",
        parent=parent_node,
        critical=True,
    )
    c12 = _crit(extracted, "dining_options")
    claim_12 = f"The hotel has an on-site restaurant or dining service available beyond breakfast. Stated: {c12.observed or c12.detail or 'unspecified'}."
    await evaluator.verify(
        claim=claim_12,
        node=node_12,
        sources=gather_sources_for(extracted, "dining_options"),
        additional_instruction="Accept a restaurant, bar with food menu, room service, or onsite grill that serves lunch/dinner. Breakfast-only venues do not satisfy."
    )

    # 13) Fitness center/gym
    node_13 = evaluator.add_leaf(
        id="FitnessCenter",
        desc="Hotel must have a fitness center or gym available to guests",
        parent=parent_node,
        critical=True,
    )
    c13 = _crit(extracted, "fitness_center")
    claim_13 = f"The hotel has a fitness center or gym available to guests. Stated: {c13.observed or c13.detail or 'unspecified'}."
    await evaluator.verify(
        claim=claim_13,
        node=node_13,
        sources=gather_sources_for(extracted, "fitness_center"),
        additional_instruction="Confirm an on-site fitness center/gym. Access to an offsite gym without being an on-property facility does not satisfy."
    )

    # 14) Base nightly rate under $400 before taxes/fees (standard room; June 2026)
    node_14 = evaluator.add_leaf(
        id="RoomRate",
        desc="Base nightly rate for standard room must be under $400 before taxes and fees",
        parent=parent_node,
        critical=True,
    )
    c14 = _crit(extracted, "room_rate_under_400")
    claim_14 = f"For some June 2026 date(s), the base nightly rate for a standard room is under $400 before taxes/fees. Example: {c14.observed or c14.detail or 'unspecified'}."
    await evaluator.verify(
        claim=claim_14,
        node=node_14,
        sources=gather_sources_for(extracted, "room_rate_under_400"),
        additional_instruction="Assess the pre-tax, pre-fee base rate on an official booking engine page (or authoritative booking page). If only total-after-tax is shown, infer base if it explicitly indicates it is under $400 before taxes/fees; otherwise, do not accept."
    )

    # 15) Accommodation type is a traditional hotel (not vacation rental/condo)
    node_15 = evaluator.add_leaf(
        id="AccommodationType",
        desc="Must be a traditional hotel property, not a vacation rental or condo",
        parent=parent_node,
        critical=True,
    )
    c15 = _crit(extracted, "accommodation_type_hotel")
    claim_15 = f"The accommodation is a traditional hotel property (not a vacation rental or individually-owned condo). Stated: {c15.observed or c15.detail or 'unspecified'}."
    await evaluator.verify(
        claim=claim_15,
        node=node_15,
        sources=gather_sources_for(extracted, "accommodation_type_hotel"),
        additional_instruction="Accept 'hotel', 'resort', 'inn' as hotel-type properties. Reject condo-hotels with individually-owned units, vacation rentals, apartments, or home rentals."
    )


# -----------------------------------------------------------------------------
# Main evaluation entry point
# -----------------------------------------------------------------------------
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
    Evaluate an answer for the Destin hotel search with 15 strict requirements.
    """
    # Initialize evaluator/root
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

    # Extract structured info from the answer
    extracted: HotelExtraction = await evaluator.extract(
        prompt=prompt_extract_destin_hotel(),
        template_class=HotelExtraction,
        extraction_name="destin_hotel_extraction",
    )

    # Record a concise summary for debugging/tracing
    evaluator.add_custom_info(
        info={
            "hotel_name": extracted.hotel_name,
            "address": extracted.address,
            "website_url": extracted.website_url,
            "booking_url": extracted.booking_url,
            "overall_sources_count": len(extracted.overall_sources or []),
        },
        info_type="extracted_overview",
        info_name="extracted_hotel_overview",
    )

    # Build rubric root node "DestinHotelSearch" (critical parallel)
    destin_node = evaluator.add_parallel(
        id="DestinHotelSearch",
        desc="Find a hotel in Destin, Florida that meets all specified requirements for a summer vacation",
        parent=root,
        critical=True,  # All children must be critical too
    )

    # Add and run all requirement verifications (15 critical leaves)
    await verify_requirements(evaluator, destin_node, extracted)

    # Return evaluation summary
    return evaluator.get_summary()