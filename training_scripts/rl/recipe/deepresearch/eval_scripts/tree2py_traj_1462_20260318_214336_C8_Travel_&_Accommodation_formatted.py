import asyncio
import logging
from typing import Any, List, Optional, Dict

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "port_canaveral_pre_cruise_hotel"
TASK_DESCRIPTION = """A couple is planning a 7-night cruise departing from Port Canaveral, Florida in Spring 2026. They will be flying into Orlando International Airport (MCO) the day before their cruise and need a hotel for a one-night pre-cruise stay. Find a hotel that meets all of the following requirements: (1) Location: The hotel must be located within 5 miles of Port Canaveral cruise terminals. (2) Shuttle Service: The hotel must provide shuttle transportation service to Port Canaveral cruise terminals. (3) Complimentary Breakfast: The hotel must include complimentary breakfast for guests. (4) Cruise Parking Package: The hotel must offer a cruise parking package (stay-and-cruise or park-and-cruise package) that allows guests to leave their vehicle parked at the hotel during their cruise. (5) Shuttle Timing: The shuttle service must operate at times that accommodate cruise embarkation (with operating hours specified). (6) Distance Specification: The specific distance or travel time from the hotel to Port Canaveral must be clearly stated. (7) Verification: Each of the above requirements must be supported by reference URLs. Provide the hotel name and brand, specific location/area, distance to Port Canaveral, shuttle service details and timing, breakfast details, cruise parking package information, standard checkout time, airport accessibility information, additional amenities, and reference URLs for verification."""


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class HotelExtraction(BaseModel):
    # Identity and documentation
    hotel_name: Optional[str] = None
    hotel_brand: Optional[str] = None
    area: Optional[str] = None  # e.g., Cape Canaveral, Cocoa Beach
    full_address: Optional[str] = None
    contact_methods: List[str] = Field(default_factory=list)  # phone, emails, booking links, etc.
    official_urls: List[str] = Field(default_factory=list)  # official site or brand/property page
    existence_urls: List[str] = Field(default_factory=list)  # any reputable listing/official site

    # Location and distance support
    location_distance_text: Optional[str] = None  # e.g., "2.1 miles"
    location_travel_time_text: Optional[str] = None  # e.g., "8 minutes"
    location_support_urls: List[str] = Field(default_factory=list)

    # Shuttle to cruise terminals
    shuttle_offered_text: Optional[str] = None  # e.g., "Shuttle to Port Canaveral available"
    shuttle_operating_times: Optional[str] = None  # hours or specific departure times
    shuttle_requirements: Optional[str] = None  # reservation, fee, limitations
    shuttle_urls: List[str] = Field(default_factory=list)

    # Complimentary breakfast
    breakfast_text: Optional[str] = None  # should indicate included/complimentary
    breakfast_urls: List[str] = Field(default_factory=list)

    # Cruise parking package
    parking_package_name: Optional[str] = None  # stay-and-cruise / park-and-cruise
    parking_package_details: Optional[str] = None  # explicit permission to leave vehicle during cruise, etc.
    parking_package_urls: List[str] = Field(default_factory=list)
    parking_package_price: Optional[str] = None  # optional
    price_comparison_text: Optional[str] = None  # optional narrative comparing to $20/day port parking

    # Required extra info
    checkout_time: Optional[str] = None
    mco_accessibility_info: Optional[str] = None  # drive time/distance or transport options from MCO
    extra_amenities: List[str] = Field(default_factory=list)
    extra_info_urls: List[str] = Field(default_factory=list)

    # Catch-all references, if provided
    all_reference_urls: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_hotel() -> str:
    return """
    Extract details for exactly one selected hotel (the primary recommendation) presented in the answer for a one-night pre-cruise stay near Port Canaveral. If multiple hotels are mentioned, extract the first one that the answer positions as meeting the constraints.

    For the selected hotel, extract the following fields exactly as they appear in the answer:
    1) hotel_name: The hotel's name.
    2) hotel_brand: The brand affiliation (e.g., Hilton, Marriott), if stated; otherwise null.
    3) area: The locality/area descriptor (e.g., Cape Canaveral, Cocoa Beach), if given; otherwise null.
    4) full_address: Full street address (street, city, state, ZIP) if present; otherwise null.
    5) contact_methods: An array of contact or booking methods (e.g., official website URL, booking link URL, phone number).
    6) official_urls: URLs for the hotel's official site or brand property page (array; may be empty).
    7) existence_urls: URLs evidencing the property exists and is operating (official site or reputable listing such as the brand website, TripAdvisor, Booking.com, Expedia) (array; may be empty).

    Location and distance support:
    8) location_distance_text: The exact distance string to Port Canaveral if stated (e.g., "2.1 miles"), else null.
    9) location_travel_time_text: The exact travel time string if stated (e.g., "8 minutes"), else null.
    10) location_support_urls: URLs that support the stated distance or time to Port Canaveral (array; may be empty).

    Shuttle to cruise terminals:
    11) shuttle_offered_text: Text indicating shuttle service to Port Canaveral is provided (verbatim from answer), else null.
    12) shuttle_operating_times: Any stated operating hours or scheduled departure times (verbatim), else null.
    13) shuttle_requirements: Any stated requirements (reservation needed, fees, limitations) (verbatim), else null.
    14) shuttle_urls: URLs documenting shuttle availability and timing/requirements (array; may be empty).

    Complimentary breakfast:
    15) breakfast_text: Text indicating that breakfast is complimentary/included, else null.
    16) breakfast_urls: URLs supporting complimentary breakfast (array; may be empty).

    Cruise parking package:
    17) parking_package_name: Name/label of the package (e.g., "Park and Cruise"), else null.
    18) parking_package_details: Text stating guests can leave their vehicle parked during the cruise (verbatim), else null.
    19) parking_package_urls: URLs documenting the cruise parking package and terms (array; may be empty).
    20) parking_package_price: If a price or clear pricing method is stated, extract it verbatim; else null.
    21) price_comparison_text: If a comparison is provided to Port Canaveral parking rates (e.g., "$20/day") for a 7-night cruise, extract the text verbatim; else null.

    Required extra info:
    22) checkout_time: The standard checkout time if stated; else null.
    23) mco_accessibility_info: Accessibility info from MCO (drive time/distance or transport options) if stated; else null.
    24) extra_amenities: Array of additional amenities beyond the core constraints (e.g., pool, Wi-Fi).
    25) extra_info_urls: URLs supporting checkout time, amenities, or MCO accessibility (array; may be empty).

    26) all_reference_urls: Every URL mentioned anywhere in the answer relating to this hotel (array; may be empty).

    SPECIAL RULES FOR URL SOURCES EXTRACTION:
    - Extract only URLs explicitly present in the answer text. Do not invent URLs.
    - Accept plain URLs or markdown links; extract the actual URL.
    - If a URL is missing a protocol (http/https), prepend http://.
    - Deduplicate URLs where possible.

    If any field is not present in the answer, set it to null (or empty array for list fields).
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _merge_urls(*url_lists: Optional[List[str]]) -> List[str]:
    seen = set()
    merged: List[str] = []
    for lst in url_lists:
        if not lst:
            continue
        for u in lst:
            if not u:
                continue
            if u not in seen:
                seen.add(u)
                merged.append(u)
    return merged


# --------------------------------------------------------------------------- #
# Verification tree construction                                              #
# --------------------------------------------------------------------------- #
async def verify_hotel_requirements(evaluator: Evaluator, root, info: HotelExtraction) -> None:
    # 1) Hotel Identity and Documentation (critical)
    identity_node = evaluator.add_parallel(
        id="hotel_identity_and_documentation",
        desc="Confirm the solution provides required identifying/booking information for the selected hotel and that it is a real, operating property.",
        parent=root,
        critical=True,
    )

    evaluator.add_custom_node(
        result=bool(info.hotel_name and info.hotel_name.strip()),
        id="hotel_name_and_brand_provided",
        desc="Hotel name and brand affiliation (if applicable) are stated.",
        parent=identity_node,
        critical=True,
    )

    evaluator.add_custom_node(
        result=bool(info.area and info.area.strip()),
        id="specific_location_or_area_provided",
        desc="The specific location/area is stated (e.g., Cape Canaveral/Cocoa Beach area or equivalent locality descriptor).",
        parent=identity_node,
        critical=True,
    )

    evaluator.add_custom_node(
        result=bool(info.full_address and info.full_address.strip()),
        id="full_address_provided",
        desc="The hotel's full address is provided.",
        parent=identity_node,
        critical=True,
    )

    evaluator.add_custom_node(
        result=bool((info.contact_methods and len(info.contact_methods) > 0) or (info.official_urls and len(info.official_urls) > 0)),
        id="contact_or_booking_info_provided",
        desc="At least one contact/booking method is provided (e.g., official website, phone number, or booking link).",
        parent=identity_node,
        critical=True,
    )

    # Real operating property verified via URL
    real_prop_leaf = evaluator.add_leaf(
        id="real_operating_property_verified_with_url",
        desc="At least one reference URL evidences the property exists and is currently operating (official site or reputable listing).",
        parent=identity_node,
        critical=True,
    )
    existence_sources = _merge_urls(info.official_urls, info.existence_urls, info.all_reference_urls)
    await evaluator.verify(
        claim=f"The referenced page(s) show that the hotel '{info.hotel_name or 'the selected hotel'}' is an existing, currently operating property with live booking or contact details.",
        node=real_prop_leaf,
        sources=existence_sources,
        additional_instruction="Accept brand property pages, the official hotel website, or reputable OTA/listing pages that clearly indicate the property exists and is operating.",
    )

    # 2) Location within 5 miles (critical)
    location_node = evaluator.add_parallel(
        id="location_within_5_miles",
        desc="Verify the hotel is within 5 miles of Port Canaveral cruise terminals and the distance/travel time is clearly stated and sourced.",
        parent=root,
        critical=True,
    )

    distance_or_time_exists = evaluator.add_custom_node(
        result=bool((info.location_distance_text and info.location_distance_text.strip()) or
                    (info.location_travel_time_text and info.location_travel_time_text.strip())),
        id="distance_or_travel_time_stated",
        desc="A specific distance (miles) or travel time (minutes) from the hotel to Port Canaveral is explicitly stated.",
        parent=location_node,
        critical=True,
    )

    within_5_leaf = evaluator.add_leaf(
        id="within_5_miles_requirement_met",
        desc="The stated/verified distance places the hotel within 5 miles of Port Canaveral cruise terminals.",
        parent=location_node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"The hotel's location relative to Port Canaveral is within 5 miles (the answer cites {info.location_distance_text or info.location_travel_time_text or 'a specific short distance/time'}).",
        node=within_5_leaf,
        sources=_merge_urls(info.location_support_urls, info.all_reference_urls),
        additional_instruction="Use the page evidence: if it states a distance < 5 miles or a very short typical drive time consistent with under 5 miles, consider this satisfied.",
    )

    loc_support_leaf = evaluator.add_leaf(
        id="location_claim_supported_by_url",
        desc="A reference URL is provided supporting the stated distance/proximity to Port Canaveral.",
        parent=location_node,
        critical=True,
    )
    stated_piece = info.location_distance_text or info.location_travel_time_text or "the specific proximity"
    await evaluator.verify(
        claim=f"The referenced page(s) explicitly state {stated_piece} from the hotel to Port Canaveral (or equivalent).",
        node=loc_support_leaf,
        sources=_merge_urls(info.location_support_urls, info.all_reference_urls),
        additional_instruction="Look for text on the page indicating distance in miles or minutes' drive to Port Canaveral.",
    )

    # 3) Shuttle to cruise terminals (critical)
    shuttle_node = evaluator.add_parallel(
        id="shuttle_to_cruise_terminals",
        desc="Confirm the hotel provides shuttle transportation to Port Canaveral cruise terminals, with timing/requirements specified and sourced.",
        parent=root,
        critical=True,
    )

    shuttle_confirmed_leaf = evaluator.add_leaf(
        id="shuttle_service_confirmed",
        desc="The hotel explicitly offers shuttle transportation to Port Canaveral cruise terminals.",
        parent=shuttle_node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"The hotel '{info.hotel_name or 'selected hotel'}' offers shuttle transportation to Port Canaveral cruise terminals.",
        node=shuttle_confirmed_leaf,
        sources=_merge_urls(info.shuttle_urls, info.all_reference_urls),
        additional_instruction="The page should clearly indicate shuttle service to Port Canaveral; minor variations in phrasing are acceptable.",
    )

    shuttle_times_leaf = evaluator.add_leaf(
        id="shuttle_operating_times_specified",
        desc="Shuttle operating hours and/or scheduled departure times are specified.",
        parent=shuttle_node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"The page specifies shuttle operating hours or departure times (e.g., '{info.shuttle_operating_times or 'stated schedule'}').",
        node=shuttle_times_leaf,
        sources=_merge_urls(info.shuttle_urls, info.all_reference_urls),
        additional_instruction="Confirm that hours or a schedule are listed; they may be approximate or in time windows.",
    )

    shuttle_requirements_leaf = evaluator.add_leaf(
        id="shuttle_requirements_specified",
        desc="Any stated shuttle requirements are specified (e.g., reservation/advance notice, fees, limitations).",
        parent=shuttle_node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"The page specifies shuttle requirements such as reservations, fees, or limitations (e.g., '{info.shuttle_requirements or 'stated requirements'}').",
        node=shuttle_requirements_leaf,
        sources=_merge_urls(info.shuttle_urls, info.all_reference_urls),
        additional_instruction="Look for mentions like reservation required, per-person fee, time windows, capacity limits, etc.",
    )

    shuttle_compatible_leaf = evaluator.add_leaf(
        id="shuttle_timing_compatible_with_embarkation",
        desc="Provided shuttle timing accommodates embarkation needs (arrive at port at least ~2 hours before typical 4:00–4:30 PM departure).",
        parent=shuttle_node,
        critical=True,
    )
    await evaluator.verify(
        claim="Based on the stated shuttle schedule/hours, the shuttle can get guests to Port Canaveral by roughly 2:00 PM or earlier on embarkation day.",
        node=shuttle_compatible_leaf,
        sources=_merge_urls(info.shuttle_urls, info.all_reference_urls),
        additional_instruction="Use the shuttle times listed on the page. Assume typical cruise departures are ~4:00–4:30 PM and guests should arrive at least ~2 hours prior; if schedule includes mid‑morning to early‑afternoon departures, consider it compatible.",
    )

    shuttle_docs_leaf = evaluator.add_leaf(
        id="shuttle_details_supported_by_url",
        desc="A reference URL documents shuttle availability and the timing/requirements used for these checks.",
        parent=shuttle_node,
        critical=True,
    )
    await evaluator.verify(
        claim="The referenced page(s) document both shuttle availability and the timing/requirements cited.",
        node=shuttle_docs_leaf,
        sources=_merge_urls(info.shuttle_urls, info.all_reference_urls),
        additional_instruction="The same page may include all details; otherwise, multiple referenced URLs together should cover availability, times, and requirements.",
    )

    # 4) Complimentary breakfast (critical)
    breakfast_node = evaluator.add_parallel(
        id="complimentary_breakfast",
        desc="Verify the hotel includes complimentary breakfast for guests and that it is sourced.",
        parent=root,
        critical=True,
    )

    breakfast_included_leaf = evaluator.add_leaf(
        id="breakfast_is_complimentary",
        desc="Breakfast is stated as included/complimentary for guests.",
        parent=breakfast_node,
        critical=True,
    )
    await evaluator.verify(
        claim="The hotel's breakfast is complimentary/included for guests.",
        node=breakfast_included_leaf,
        sources=_merge_urls(info.breakfast_urls, info.official_urls, info.all_reference_urls),
        additional_instruction="Look for wording like 'free breakfast', 'complimentary breakfast', or 'breakfast included'.",
    )

    breakfast_url_leaf = evaluator.add_leaf(
        id="breakfast_claim_supported_by_url",
        desc="A reference URL is provided confirming complimentary breakfast.",
        parent=breakfast_node,
        critical=True,
    )
    await evaluator.verify(
        claim="The referenced page(s) explicitly confirm complimentary or included breakfast.",
        node=breakfast_url_leaf,
        sources=_merge_urls(info.breakfast_urls, info.official_urls, info.all_reference_urls),
        additional_instruction="The statement may appear on the hotel amenities page, package page, or a reputable listing.",
    )

    # 5) Cruise Parking Package — REQUIRED components (critical)
    parking_required_node = evaluator.add_parallel(
        id="cruise_parking_package_required",
        desc="Confirm the hotel offers a cruise parking package that permits leaving the vehicle during the cruise, with documentation URLs.",
        parent=root,
        critical=True,
    )

    package_confirmed_leaf = evaluator.add_leaf(
        id="parking_package_confirmed",
        desc="A stay-and-cruise / park-and-cruise package (or equivalent) is offered that permits parking during the cruise.",
        parent=parking_required_node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"The hotel offers a '{info.parking_package_name or 'Park and Cruise'}' (or equivalent) package that allows guests to leave their vehicle parked during the cruise.",
        node=package_confirmed_leaf,
        sources=_merge_urls(info.parking_package_urls, info.official_urls, info.all_reference_urls),
        additional_instruction="The page should indicate parking during the cruise is included or permitted as part of the package.",
    )

    package_docs_leaf = evaluator.add_leaf(
        id="parking_package_details_supported_by_url",
        desc="A reference URL is provided documenting the cruise parking package (and any stated terms).",
        parent=parking_required_node,
        critical=True,
    )
    await evaluator.verify(
        claim="The referenced page(s) document the cruise parking package and its terms (e.g., parking duration during cruise, shuttle included or fee, etc.).",
        node=package_docs_leaf,
        sources=_merge_urls(info.parking_package_urls, info.official_urls, info.all_reference_urls),
        additional_instruction="Look for explicit description of the stay/park-and-cruise package and included terms.",
    )

    # 5b) Cruise Parking Package — OPTIONAL economics (non-critical)
    parking_optional_node = evaluator.add_parallel(
        id="cruise_parking_package_optional",
        desc="Optional economic details about the parking package (price and comparison).",
        parent=root,
        critical=False,
    )

    evaluator.add_custom_node(
        result=bool(info.parking_package_price and info.parking_package_price.strip()),
        id="parking_package_price_stated",
        desc="If available, the cruise parking package price (or a clearly stated way it is priced) is provided.",
        parent=parking_optional_node,
        critical=False,
    )

    econ_compare_leaf = evaluator.add_leaf(
        id="economic_advantage_compared_to_port_parking",
        desc="If pricing is given, the solution may compare the package cost to official Port Canaveral parking using the stated $20/day reference and an assumed duration.",
        parent=parking_optional_node,
        critical=False,
    )
    await evaluator.verify(
        claim=f"The answer's stated economics indicate the hotel's cruise parking package is financially advantageous versus Port Canaveral parking at $20/day for an assumed ~8 days (7-night cruise).",
        node=econ_compare_leaf,
        sources=_merge_urls(info.parking_package_urls, info.official_urls, info.all_reference_urls),
        additional_instruction="Accept this only if the answer provides a reasonable price/comparison narrative; exact math precision is not required. If the answer provides no pricing or comparison, mark as Incorrect.",
    )

    # 6) Required extra info from question (critical)
    extra_info_node = evaluator.add_parallel(
        id="required_extra_info_from_question",
        desc="Confirm the solution includes additional information explicitly requested in the question.",
        parent=root,
        critical=True,
    )

    checkout_leaf = evaluator.add_leaf(
        id="checkout_time_stated",
        desc="The hotel's standard checkout time is stated.",
        parent=extra_info_node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"The standard checkout time is stated (e.g., '{info.checkout_time or 'stated checkout time'}').",
        node=checkout_leaf,
        sources=_merge_urls(info.extra_info_urls, info.official_urls, info.all_reference_urls),
        additional_instruction="Look for 'check-out' or 'checkout' on the hotel's official page or reputable listing.",
    )

    mco_access_leaf = evaluator.add_leaf(
        id="mco_airport_accessibility_info_provided",
        desc="Airport accessibility information from Orlando International Airport (MCO) is provided (driving distance/time and/or transport options).",
        parent=extra_info_node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"The answer provides accessibility info from MCO to the hotel (e.g., '{info.mco_accessibility_info or 'stated drive time/distance or options'}').",
        node=mco_access_leaf,
        sources=_merge_urls(info.extra_info_urls, info.all_reference_urls),
        additional_instruction="A Google Maps link or transport description suffices if it clearly connects MCO to the hotel.",
    )

    amenities_leaf = evaluator.add_leaf(
        id="additional_amenities_listed",
        desc="Additional amenities (beyond the core requirements) are listed.",
        parent=extra_info_node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"The following additional amenities are listed for the hotel: {info.extra_amenities if info.extra_amenities else 'amenities stated in the answer'}",
        node=amenities_leaf,
        sources=_merge_urls(info.official_urls, info.extra_info_urls, info.all_reference_urls),
        additional_instruction="Confirm that at least one amenity beyond the required constraints appears on the referenced page(s).",
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
    model: str = "o4-mini",
) -> Dict:
    """
    Evaluate an answer for the Port Canaveral pre-cruise hotel selection task.
    """
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # independent requirement groups
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

    # Extract structured hotel info from the answer
    hotel_info: HotelExtraction = await evaluator.extract(
        prompt=prompt_extract_hotel(),
        template_class=HotelExtraction,
        extraction_name="hotel_extraction",
    )

    # Record ground truth-style expectation summary (for traceability)
    evaluator.add_ground_truth({
        "requirements": [
            "Within 5 miles of Port Canaveral",
            "Shuttle to Port Canaveral with operating hours/timing compatible with embarkation",
            "Complimentary breakfast",
            "Cruise parking package allowing vehicle to remain during cruise",
            "Distance or travel time explicitly stated",
            "All claims supported by reference URLs where applicable",
            "Provide checkout time, MCO accessibility, and extra amenities"
        ],
        "cruise_length_assumption_days": 7,
        "port_parking_rate_reference": "$20/day",
    })

    # Build verification tree per rubric (with minor criticality adjustment for optional economics)
    await verify_hotel_requirements(evaluator, root, hotel_info)

    return evaluator.get_summary()