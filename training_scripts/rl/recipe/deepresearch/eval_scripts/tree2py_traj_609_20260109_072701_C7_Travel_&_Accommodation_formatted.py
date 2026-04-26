import asyncio
import logging
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "airport_amenities_hub_us"
TASK_DESCRIPTION = (
    "You are planning to book a long layover at a major US airport and want to ensure it has comprehensive amenities "
    "for your family's comfort and convenience. Identify one major US hub airport (defined as a primary hub for "
    "American Airlines, Delta Air Lines, or United Airlines) that provides ALL of the following amenities and services:\n\n"
    "1. An on-site hotel directly attached to or located within the terminal (accessible without shuttle service)\n"
    "2. Priority Pass lounge access\n"
    "3. TSA PreCheck enrollment center on-site\n"
    "4. Pet relief areas located post-security (inside the terminal)\n"
    "5. Dedicated nursing mothers rooms or lactation spaces\n"
    "6. Children's play areas or playground facilities\n"
    "7. Free WiFi throughout all terminal areas\n"
    "8. Designated accessible parking spaces\n"
    "9. Currency exchange services in the terminal\n"
    "10. ATM machines in the terminal\n"
    "11. Rental car center or on-site rental car facilities\n"
    "12. Medical services (clinic, first aid station, or pharmacy)\n"
    "13. Baggage wrapping services\n"
    "14. Spa or massage services available\n\n"
    "Provide the airport's three-letter IATA code, full name, and a brief description of how it meets each requirement, "
    "with reference URLs supporting each amenity."
)


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class AirportPrimaryInfo(BaseModel):
    name: Optional[str] = None
    iata_code: Optional[str] = None  # Expect a 3-letter IATA code


class AirportCandidate(BaseModel):
    name: Optional[str] = None
    iata_code: Optional[str] = None


class HubInfo(BaseModel):
    airline: Optional[str] = None  # Airline name as written in the answer (e.g., "United Airlines")
    urls: List[str] = Field(default_factory=list)


class AmenityEvidence(BaseModel):
    description: Optional[str] = None  # Short statement from the answer describing the amenity
    urls: List[str] = Field(default_factory=list)  # All URLs cited for this amenity


class AirportAmenitiesExtraction(BaseModel):
    # Which airport(s) the answer presents as candidates
    primary_airport: Optional[AirportPrimaryInfo] = None
    alternate_airports: List[AirportCandidate] = Field(default_factory=list)

    # Hub status evidence and airline (if specified)
    hub_info: Optional[HubInfo] = None

    # Amenity evidences (1–14)
    on_site_hotel: AmenityEvidence = Field(default_factory=AmenityEvidence)
    priority_pass_access: AmenityEvidence = Field(default_factory=AmenityEvidence)
    tsa_precheck_enrollment: AmenityEvidence = Field(default_factory=AmenityEvidence)
    post_security_pet_relief: AmenityEvidence = Field(default_factory=AmenityEvidence)
    nursing_mothers_rooms: AmenityEvidence = Field(default_factory=AmenityEvidence)
    children_play_areas: AmenityEvidence = Field(default_factory=AmenityEvidence)
    free_wifi: AmenityEvidence = Field(default_factory=AmenityEvidence)
    accessible_parking: AmenityEvidence = Field(default_factory=AmenityEvidence)
    currency_exchange: AmenityEvidence = Field(default_factory=AmenityEvidence)
    atm_services: AmenityEvidence = Field(default_factory=AmenityEvidence)
    rental_car_facilities: AmenityEvidence = Field(default_factory=AmenityEvidence)
    medical_services: AmenityEvidence = Field(default_factory=AmenityEvidence)
    baggage_wrapping: AmenityEvidence = Field(default_factory=AmenityEvidence)
    spa_services: AmenityEvidence = Field(default_factory=AmenityEvidence)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_airport_amenities() -> str:
    return """
Extract the airport and amenity details explicitly presented in the answer. Follow all rules exactly.

You must extract:
1) primary_airport: The single airport that the answer clearly presents as the recommended/qualifying airport.
   - name: Full official airport name (e.g., "Hartsfield–Jackson Atlanta International Airport")
   - iata_code: Three-letter IATA code (e.g., "ATL")
2) alternate_airports: Any other airports that the answer also presents as additional/alternative candidates that meet (or may meet) the criteria. 
   Only include airports if they are clearly presented as other answers, not merely mentioned in passing or as examples.
   - array of objects: { name, iata_code }
3) hub_info: If the answer states hub status, extract:
   - airline: The airline name exactly as written (e.g., "Delta Air Lines", "American Airlines", or "United Airlines")
   - urls: All URLs cited that support the hub status claim

For each of the following 14 amenity requirements, extract an AmenityEvidence object with:
- description: A brief quote or paraphrase from the answer describing how the airport satisfies this amenity
- urls: All URL(s) provided in the answer that support this amenity. Extract only valid URLs explicitly present in the answer (including markdown links). Do NOT invent URLs.
If the answer provides no URL(s) for an amenity, return an empty array for that amenity's urls.

Required amenities to extract:
- on_site_hotel
- priority_pass_access
- tsa_precheck_enrollment
- post_security_pet_relief
- nursing_mothers_rooms
- children_play_areas
- free_wifi
- accessible_parking
- currency_exchange
- atm_services
- rental_car_facilities
- medical_services
- baggage_wrapping
- spa_services

IMPORTANT URL rules:
- Only extract URLs explicitly provided in the answer. If none are provided for an item, return an empty list.
- Include full URLs; if protocol is missing, prepend "http://".
- Do not deduplicate; include all URLs provided.

Return a single JSON object strictly conforming to the specified schema.
    """.strip()


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _airport_display(primary: Optional[AirportPrimaryInfo]) -> str:
    if not primary:
        return "the selected airport"
    name = primary.name or "the selected airport"
    code = primary.iata_code or ""
    if code:
        return f"{name} ({code})"
    return name


def _is_valid_iata(code: Optional[str]) -> bool:
    if not code or len(code.strip()) != 3:
        return False
    c = code.strip()
    return c.isalpha()


def _has_any_urls(ev: AmenityEvidence) -> bool:
    return bool(ev and isinstance(ev.urls, list) and len(ev.urls) > 0)


def _collect_all_amenity_evidences(ext: AirportAmenitiesExtraction) -> List[Tuple[str, AmenityEvidence]]:
    return [
        ("On_Site_Hotel", ext.on_site_hotel),
        ("Priority_Pass_Access", ext.priority_pass_access),
        ("TSA_PreCheck_Enrollment", ext.tsa_precheck_enrollment),
        ("Post_Security_Pet_Relief", ext.post_security_pet_relief),
        ("Nursing_Mothers_Rooms", ext.nursing_mothers_rooms),
        ("Children_Play_Areas", ext.children_play_areas),
        ("Free_WiFi", ext.free_wifi),
        ("Accessible_Parking", ext.accessible_parking),
        ("Currency_Exchange", ext.currency_exchange),
        ("ATM_Services", ext.atm_services),
        ("Rental_Car_Facilities", ext.rental_car_facilities),
        ("Medical_Services", ext.medical_services),
        ("Baggage_Wrapping", ext.baggage_wrapping),
        ("Spa_Services", ext.spa_services),
    ]


def _gather_all_urls_for_search(ext: AirportAmenitiesExtraction) -> List[str]:
    urls: List[str] = []
    for _, ev in _collect_all_amenity_evidences(ext):
        if ev and ev.urls:
            urls.extend([u for u in ev.urls if isinstance(u, str) and u.strip()])
    if ext.hub_info and ext.hub_info.urls:
        urls.extend([u for u in ext.hub_info.urls if isinstance(u, str) and u.strip()])
    # Remove obvious duplicates while preserving order
    seen = set()
    deduped = []
    for u in urls:
        if u not in seen:
            seen.add(u)
            deduped.append(u)
    return deduped


# --------------------------------------------------------------------------- #
# Verification helpers                                                        #
# --------------------------------------------------------------------------- #
async def _verify_hub_status(
    evaluator: Evaluator,
    parent,
    extraction: AirportAmenitiesExtraction,
) -> None:
    # Node for hub status (critical)
    hub_node = evaluator.add_leaf(
        id="Hub_Status_Primary_Hub_AA_DL_UA",
        desc="The airport is a primary hub for at least one of: American Airlines, Delta Air Lines, or United Airlines (as defined in the question).",
        parent=parent,
        critical=True,
    )

    airport_str = _airport_display(extraction.primary_airport)
    # Build sources: prefer hub_info.urls; if missing, fall back to all extracted amenity URLs to try to find mention
    sources: List[str] = []
    if extraction.hub_info and extraction.hub_info.urls:
        sources = extraction.hub_info.urls
    if not sources:
        sources = _gather_all_urls_for_search(extraction)

    claim = (
        f"{airport_str} is a primary or major hub for at least one of these airlines: American Airlines, Delta Air Lines, or United Airlines."
    )
    add_ins = (
        "Verify using the provided webpages whether the airport is a 'hub' (primary/major hub) of American Airlines, "
        "Delta Air Lines, or United Airlines. Do NOT count 'focus city', 'crew base', or 'operating base' without explicit 'hub' designation. "
        "Pass only if the page(s) clearly denote 'hub' for one of these three airlines."
    )
    await evaluator.verify(claim=claim, node=hub_node, sources=sources, additional_instruction=add_ins)


async def _verify_single_airport(
    evaluator: Evaluator,
    parent,
    extraction: AirportAmenitiesExtraction,
) -> None:
    # Exactly one airport should be presented as the answer
    primary_present = extraction.primary_airport is not None and (
        (extraction.primary_airport.name and extraction.primary_airport.name.strip())
        or (extraction.primary_airport.iata_code and extraction.primary_airport.iata_code.strip())
    )
    alternates_count = len(extraction.alternate_airports or [])
    exactly_one = primary_present and alternates_count == 0

    evaluator.add_custom_node(
        result=exactly_one,
        id="Single_Airport_Provided",
        desc="The response identifies exactly one airport as the answer (not multiple airports).",
        parent=parent,
        critical=True,
    )


async def _verify_airport_identification(
    evaluator: Evaluator,
    parent,
    extraction: AirportAmenitiesExtraction,
) -> None:
    # Must include IATA code (3 letters) and full name (non-empty)
    name_ok = bool(extraction.primary_airport and extraction.primary_airport.name and extraction.primary_airport.name.strip())
    code_ok = bool(extraction.primary_airport and _is_valid_iata(extraction.primary_airport.iata_code))
    evaluator.add_custom_node(
        result=name_ok and code_ok,
        id="Airport_Identification",
        desc="The response provides the airport's three-letter IATA code and full official name.",
        parent=parent,
        critical=True,
    )


async def _verify_reference_documentation(
    evaluator: Evaluator,
    parent,
    extraction: AirportAmenitiesExtraction,
) -> None:
    # Ensure each of the 14 amenities has at least one supporting URL
    amenities = _collect_all_amenity_evidences(extraction)
    all_have_urls = all(_has_any_urls(ev) for _, ev in amenities)

    evaluator.add_custom_node(
        result=all_have_urls,
        id="Reference_Documentation",
        desc="The response includes reference URL(s) that support each required amenity/service claim (i.e., evidence is provided for all listed requirements).",
        parent=parent,
        critical=True,
    )


async def _verify_amenity(
    evaluator: Evaluator,
    parent,
    node_id: str,
    node_desc: str,
    claim: str,
    sources: List[str],
    add_ins: str,
) -> None:
    leaf = evaluator.add_leaf(
        id=node_id,
        desc=node_desc,
        parent=parent,
        critical=True,
    )
    await evaluator.verify(
        claim=claim,
        node=leaf,
        sources=sources,
        additional_instruction=add_ins,
    )


async def _verify_all_amenities(
    evaluator: Evaluator,
    parent,
    extraction: AirportAmenitiesExtraction,
) -> None:
    airport_str = _airport_display(extraction.primary_airport)

    # Prepare amenity verifications (id, desc, claim, sources, add_ins)
    items: List[Tuple[str, str, str, List[str], str]] = []

    # 1. On-site hotel (attached or within terminal; no shuttle required)
    items.append((
        "On_Site_Hotel",
        "Airport has an on-site hotel directly attached to or located within the terminal, accessible without requiring shuttle service.",
        f"There is an on-site hotel directly attached to or located within the terminal at {airport_str}, accessible without using a shuttle.",
        extraction.on_site_hotel.urls,
        "Pass if the hotel is inside the terminal or directly connected (e.g., skybridge/walkway) such that guests do not need a shuttle to reach it. "
        "If only off-site hotels or shuttle-required hotels are available, fail."
    ))

    # 2. Priority Pass lounge access
    items.append((
        "Priority_Pass_Access",
        "Airport offers Priority Pass lounge access.",
        f"At least one lounge, suite, or facility at {airport_str} grants entry via Priority Pass membership.",
        extraction.priority_pass_access.urls,
        "Look for Priority Pass-affiliated lounges or facilities (e.g., The Club, Minute Suites credit, select contract lounges). "
        "If only non-Priority Pass lounges are present, fail."
    ))

    # 3. TSA PreCheck enrollment center on-site
    items.append((
        "TSA_PreCheck_Enrollment",
        "Airport has a TSA PreCheck enrollment center available on-site.",
        f"There is a TSA PreCheck enrollment center located on the premises of {airport_str}.",
        extraction.tsa_precheck_enrollment.urls,
        "Accept if an official TSA or airport page lists an enrollment center at the airport or on its property."
    ))

    # 4. Post-security pet relief areas
    items.append((
        "Post_Security_Pet_Relief",
        "Airport has pet relief areas located post-security (inside terminal after security checkpoint).",
        f"{airport_str} has at least one post-security (airside) pet relief area inside the terminal.",
        extraction.post_security_pet_relief.urls,
        "The evidence must clearly indicate 'post-security', 'airside', or 'inside the terminal beyond security'. "
        "Pre-security only is not sufficient."
    ))

    # 5. Nursing mothers rooms or lactation spaces
    items.append((
        "Nursing_Mothers_Rooms",
        "Airport provides dedicated nursing mothers rooms or lactation spaces.",
        f"Dedicated nursing mothers rooms or lactation spaces are available at {airport_str}.",
        extraction.nursing_mothers_rooms.urls,
        "Look for dedicated lactation rooms, Mamava pods, or nursing suites."
    ))

    # 6. Children's play areas
    items.append((
        "Children_Play_Areas",
        "Airport has children's play areas or playground facilities available.",
        f"{airport_str} provides a children's play area or playground facility inside the terminal.",
        extraction.children_play_areas.urls,
        "Accept indoor play areas, themed play spaces, or designated kids' play zones."
    ))

    # 7. Free WiFi throughout terminal
    items.append((
        "Free_WiFi",
        "Airport provides free WiFi service throughout all terminal areas.",
        f"Free Wi-Fi is available throughout all terminal areas at {airport_str}.",
        extraction.free_wifi.urls,
        "Accept statements indicating free Wi-Fi is available airport-wide or throughout the terminals."
    ))

    # 8. Accessible parking
    items.append((
        "Accessible_Parking",
        "Airport has designated accessible parking spaces for travelers with disabilities.",
        f"Designated accessible (ADA) parking spaces are available at {airport_str}'s parking facilities.",
        extraction.accessible_parking.urls,
        "Look for mentions of ADA/accessible parking, handicap parking spaces, locations, or accessibility accommodations."
    ))

    # 9. Currency exchange
    items.append((
        "Currency_Exchange",
        "Airport has currency exchange services available in the terminal.",
        f"Foreign currency exchange services are available inside the terminal at {airport_str}.",
        extraction.currency_exchange.urls,
        "If the page indicates currency exchange counters or services (e.g., Travelex) operating in the terminal, pass. "
        "If services are permanently closed or unavailable, fail."
    ))

    # 10. ATM services
    items.append((
        "ATM_Services",
        "Airport has ATM machines located in the terminal.",
        f"ATMs are available inside the terminals at {airport_str}.",
        extraction.atm_services.urls,
        "Look for mentions of ATM machines and their terminal locations."
    ))

    # 11. Rental car facilities
    items.append((
        "Rental_Car_Facilities",
        "Airport has a rental car center or on-site rental car pickup facilities.",
        f"{airport_str} has a rental car center or on-site rental car pickup facilities available on the airport property.",
        extraction.rental_car_facilities.urls,
        "Accept a consolidated Rental Car Center (RCC) or on-site counters; train/people-mover access is acceptable if on airport property."
    ))

    # 12. Medical services
    items.append((
        "Medical_Services",
        "Airport has medical services available (medical clinic, first aid station, or pharmacy).",
        f"Medical services such as a clinic, first aid station, or a pharmacy are available at {airport_str}.",
        extraction.medical_services.urls,
        "Any in-terminal medical clinic, first aid office, or a clearly identified pharmacy qualifies."
    ))

    # 13. Baggage wrapping services
    items.append((
        "Baggage_Wrapping",
        "Airport offers baggage wrapping services.",
        f"Baggage wrapping services are available at {airport_str}.",
        extraction.baggage_wrapping.urls,
        "Look for dedicated baggage wrapping counters or services mentioned on official or operator websites."
    ))

    # 14. Spa or massage services
    items.append((
        "Spa_Services",
        "Airport has spa services or massage facilities available.",
        f"Spa or massage services are available inside the terminal at {airport_str}.",
        extraction.spa_services.urls,
        "Accept spa/massage brands (e.g., XpresSpa, Massage Bar) or similar in-terminal services."
    ))

    # Issue verifications sequentially (precondition gating handled by framework)
    for node_id, node_desc, claim, sources, add_ins in items:
        await _verify_amenity(
            evaluator=evaluator,
            parent=parent,
            node_id=node_id,
            node_desc=node_desc,
            claim=claim,
            sources=sources,
            add_ins=add_ins,
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
    # Initialize evaluator
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Overall checks are independent; critical node will gate pass/fail
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

    # Extract structured details from the answer
    extraction = await evaluator.extract(
        prompt=prompt_extract_airport_amenities(),
        template_class=AirportAmenitiesExtraction,
        extraction_name="airport_amenities_extraction",
    )

    # Build the critical qualification node (all children must be critical)
    airport_qualification = evaluator.add_parallel(
        id="Airport_Qualification",
        desc="The response identifies exactly one qualifying major US hub airport and demonstrates it meets all required amenities, with supporting references.",
        parent=root,
        critical=True,
    )

    # 1) Single airport check
    await _verify_single_airport(evaluator, airport_qualification, extraction)

    # 2) Airport identification (IATA + full name)
    await _verify_airport_identification(evaluator, airport_qualification, extraction)

    # 3) Reference URLs exist for all amenities (required to avoid unsupported claims)
    await _verify_reference_documentation(evaluator, airport_qualification, extraction)

    # 4) Hub status (AA / DL / UA)
    await _verify_hub_status(evaluator, airport_qualification, extraction)

    # 5) The 14 amenity verifications
    await _verify_all_amenities(evaluator, airport_qualification, extraction)

    # Return evaluation summary
    return evaluator.get_summary()