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
TASK_ID = "wi_to_co_ski_trip_2026"
TASK_DESCRIPTION = """
A family from Appleton, Wisconsin is planning a ski vacation to Colorado in February 2026. The family includes two adults and two children who are first-time skiers. They want to minimize costs while ensuring a beginner-friendly experience.

Please provide a complete trip plan that includes:

1. Flight Information: Identify a budget airline that offers nonstop flights from Appleton/Green Bay, Wisconsin (ATW) to an airport providing access to Colorado ski resorts. Include the airline's carry-on baggage fee and first checked baggage fee when paid at the time of booking.

2. Ground Transportation: Identify a shuttle service company that provides transportation from Denver International Airport to Colorado ski resort areas, and describe the type of service they offer.

3. Ski Resort Selection: Recommend a Colorado ski resort near Denver that is specifically recognized as beginner-friendly or having excellent terrain for first-time skiers. Provide evidence of why this resort is suitable for beginners, and confirm its operating status for the 2025-26 ski season.

4. Accommodation: Confirm that lodging is available at or near your selected resort, and indicate whether vacation packages that bundle lodging with lift tickets are offered.

5. Return Activity (Optional): After returning to Wisconsin, the family would like to visit a state park near Wisconsin Dells for hiking. Identify a state park in that area and its approximate proximity to Wisconsin Dells.

For each component of your answer, provide reference URLs from authoritative sources to support your recommendations.
""".strip()


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class FlightInfo(BaseModel):
    airline: Optional[str] = None
    origin_airport: Optional[str] = None  # e.g., ATW or GRB (allow free text)
    destination_airport: Optional[str] = None  # e.g., DEN, EGE, ASE (allow free text)
    carry_on_fee_at_booking: Optional[str] = None  # e.g., "$39"
    first_checked_bag_fee_at_booking: Optional[str] = None  # e.g., "$45"
    route_urls: List[str] = Field(default_factory=list)  # URLs confirming routes/nonstop
    fee_urls: List[str] = Field(default_factory=list)  # URLs confirming baggage fees


class GroundTransport(BaseModel):
    company: Optional[str] = None
    service_type: Optional[str] = None  # e.g., shared ride, private, scheduled
    urls: List[str] = Field(default_factory=list)


class ResortInfo(BaseModel):
    resort_name: Optional[str] = None
    beginner_evidence: Optional[str] = None  # quote or summary from the answer
    season_status: Optional[str] = None  # e.g., "Operating 2025–26"
    urls: List[str] = Field(default_factory=list)


class AccommodationInfo(BaseModel):
    lodging_available: Optional[str] = None  # free text "On-site lodging available" or "Yes"
    package_offered: Optional[str] = None  # free text "Yes" or description of bundles
    urls: List[str] = Field(default_factory=list)


class ReturnActivity(BaseModel):
    park_name: Optional[str] = None
    proximity: Optional[str] = None  # e.g., "10 miles from Wisconsin Dells", or "15-minute drive"
    urls: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_flight() -> str:
    return """
    Extract the flight information presented in the answer for a budget airline and nonstop route from Appleton/Green Bay (ATW or GRB) to a Colorado-accessible airport (e.g., DEN/EGE/ASE). Return:
    - airline: the airline name chosen
    - origin_airport: the origin airport as written (e.g., "ATW", "Appleton (ATW)", or "Green Bay (GRB)")
    - destination_airport: the destination airport as written (e.g., "DEN", "Eagle County (EGE)")
    - carry_on_fee_at_booking: the carry-on baggage fee when paid during booking, exactly as written (include currency sign if present)
    - first_checked_bag_fee_at_booking: the first checked baggage fee when paid during booking, exactly as written
    - route_urls: an array of all URLs cited that support the route or nonstop service claim
    - fee_urls: an array of all URLs cited that support baggage fee details
    If any field is missing, set it to null. Only extract URLs explicitly present in the answer.
    """.strip()


def prompt_extract_ground_transport() -> str:
    return """
    Extract the ground transportation/shuttle information from Denver International Airport (DEN) to ski resort areas. Return:
    - company: shuttle provider/company name
    - service_type: the type of service described (e.g., shared ride, private shuttle, scheduled service)
    - urls: an array of URLs that support the service operating between DEN and ski resort areas
    If any field is missing, set it to null (or an empty array for URLs).
    """.strip()


def prompt_extract_resort() -> str:
    return """
    Extract the selected ski resort and beginner suitability details. Return:
    - resort_name: the chosen Colorado ski resort near Denver
    - beginner_evidence: the evidence or direct quote from the answer showing beginner-friendly terrain or suitability for first-time skiers
    - season_status: any mention of operating/opening status for the 2025–26 ski season (e.g., "Open for 2025/26", "Operating Winter 2025–26"); if not mentioned, set null
    - urls: an array of URLs that support beginner-friendly attributes and/or the resort identification (and season status if provided)
    If any field is missing, set it to null (or an empty array for URLs).
    """.strip()


def prompt_extract_accommodation() -> str:
    return """
    Extract the lodging and package details for stays at or near the selected resort. Return:
    - lodging_available: a statement confirming lodging availability (e.g., "On-site lodging available", "Yes")
    - package_offered: whether lodging + lift ticket bundles are offered; include phrasing as written (e.g., "Yes, bundle available", "Package with lift tickets")
    - urls: an array of URLs that support lodging availability and/or package offerings
    If any field is missing, set it to null (or an empty array for URLs).
    """.strip()


def prompt_extract_return_activity() -> str:
    return """
    Extract the optional return activity near Wisconsin Dells. Return:
    - park_name: the identified state park for hiking near Wisconsin Dells
    - proximity: the approximate distance or proximity to Wisconsin Dells (e.g., "5 miles", "10-minute drive")
    - urls: an array of URLs that confirm the park's location and hiking opportunities
    If any field is missing, set it to null (or an empty array for URLs).
    """.strip()


# --------------------------------------------------------------------------- #
# Verification helpers                                                        #
# --------------------------------------------------------------------------- #
async def verify_flight_section(evaluator: Evaluator, parent_node, flight: FlightInfo) -> None:
    flight_node = evaluator.add_parallel(
        id="flight_selection",
        desc="Identify appropriate airline service from Wisconsin to Denver area with correct baggage fee information",
        parent=parent_node,
        critical=False
    )

    has_required = (
        (flight.airline is not None and flight.airline.strip() != "") and
        (flight.origin_airport is not None and flight.origin_airport.strip() != "") and
        (flight.destination_airport is not None and flight.destination_airport.strip() != "") and
        (flight.carry_on_fee_at_booking is not None and flight.carry_on_fee_at_booking.strip() != "") and
        (flight.first_checked_bag_fee_at_booking is not None and flight.first_checked_bag_fee_at_booking.strip() != "") and
        (len(flight.route_urls) > 0) and
        (len(flight.fee_urls) > 0)
    )

    evaluator.add_custom_node(
        result=has_required,
        id="flight_required_info",
        desc="Flight info includes airline, nonstop route endpoints, baggage fees, and source URLs",
        parent=flight_node,
        critical=True
    )

    airline_ident_leaf = evaluator.add_leaf(
        id="airline_identification",
        desc="Identify a budget airline that operates nonstop flights from Appleton/Green Bay, Wisconsin (ATW/GRB) to a Colorado-accessible airport",
        parent=flight_node,
        critical=True
    )
    route_claim = f"{flight.airline} operates nonstop flights from {flight.origin_airport} to {flight.destination_airport}."
    await evaluator.verify(
        claim=route_claim,
        node=airline_ident_leaf,
        sources=flight.route_urls,
        additional_instruction=(
            "Verify the nonstop route using airline route maps, booking results, schedules, airport or airline pages, "
            "or credible transportation sources. Seasonal nonstop service is acceptable if indicated."
        )
    )

    carry_on_leaf = evaluator.add_leaf(
        id="carry_on_baggage_fee",
        desc="Provide the correct carry-on baggage fee when paid at booking for the identified airline",
        parent=flight_node,
        critical=True
    )
    carry_on_claim = f"The carry-on baggage fee when paid during booking for {flight.airline} is {flight.carry_on_fee_at_booking}."
    await evaluator.verify(
        claim=carry_on_claim,
        node=carry_on_leaf,
        sources=flight.fee_urls,
        additional_instruction=(
            "Confirm the fee specifically for a carry-on bag when purchased at the time of booking. "
            "If the page lists multiple prices (e.g., at booking vs airport), ensure the extracted value corresponds to 'at booking'."
        )
    )

    checked_leaf = evaluator.add_leaf(
        id="checked_baggage_fee",
        desc="Provide the correct first checked baggage fee when paid at booking for the identified airline",
        parent=flight_node,
        critical=True
    )
    checked_claim = f"The first checked baggage fee when paid during booking for {flight.airline} is {flight.first_checked_bag_fee_at_booking}."
    await evaluator.verify(
        claim=checked_claim,
        node=checked_leaf,
        sources=flight.fee_urls,
        additional_instruction=(
            "Confirm the fee specifically for the first checked bag when purchased at the time of booking. "
            "Some airlines have dynamic or route-based pricing; accept the stated amount only if clearly supported."
        )
    )

    flight_ref_leaf = evaluator.add_leaf(
        id="flight_reference_url",
        desc="Provide a valid reference URL confirming the airline's route and baggage fee information",
        parent=flight_node,
        critical=True
    )
    combo_sources = (flight.route_urls or []) + (flight.fee_urls or [])
    flight_ref_claim = (
        f"At least one of the provided pages is an authoritative or credible source (e.g., airline or airport site) "
        f"that confirms either the {flight.airline} nonstop route {flight.origin_airport}–{flight.destination_airport} "
        f"and/or the specified baggage fees when paid at booking."
    )
    await evaluator.verify(
        claim=flight_ref_claim,
        node=flight_ref_leaf,
        sources=combo_sources,
        additional_instruction="Judge relevance and authority of the sources; airline/airport pages are preferred, credible travel pages are acceptable."
    )


async def verify_ground_transport_section(evaluator: Evaluator, parent_node, gt: GroundTransport) -> None:
    gt_node = evaluator.add_parallel(
        id="ground_transportation",
        desc="Identify appropriate ground transportation option from Denver Airport to ski resorts",
        parent=parent_node,
        critical=False
    )

    has_required = (gt.company is not None and gt.company.strip() != "" and len(gt.urls) > 0)
    evaluator.add_custom_node(
        result=has_required,
        id="transport_required_info",
        desc="Ground transportation info includes company name and supporting URL(s)",
        parent=gt_node,
        critical=True
    )

    shuttle_leaf = evaluator.add_leaf(
        id="shuttle_service_identification",
        desc="Identify at least one mountain carrier shuttle service between DEN and Colorado ski resort areas",
        parent=gt_node,
        critical=True
    )
    shuttle_claim = f"{gt.company} operates shuttle services between Denver International Airport (DEN) and Colorado ski resort areas."
    await evaluator.verify(
        claim=shuttle_claim,
        node=shuttle_leaf,
        sources=gt.urls,
        additional_instruction=(
            "Confirm that the company provides service from DEN to mountain towns or ski resorts (e.g., Summit County, Vail, Aspen, etc.). "
            "Service may be shared, private, or scheduled; multiple destination areas acceptable."
        )
    )

    service_type_leaf = evaluator.add_leaf(
        id="service_type_description",
        desc="Describe the type of service provided (e.g., shared ride, private shuttle, scheduled service)",
        parent=gt_node,
        critical=False
    )
    type_claim = f"The service offered by {gt.company} is described as {gt.service_type}."
    await evaluator.verify(
        claim=type_claim,
        node=service_type_leaf,
        sources=gt.urls,
        additional_instruction="Verify that the description (shared, private, scheduled, etc.) matches the wording on the provider’s site."
    )

    transport_ref_leaf = evaluator.add_leaf(
        id="transportation_reference_url",
        desc="Provide a valid reference URL confirming the shuttle service operates between DEN and ski resorts",
        parent=gt_node,
        critical=True
    )
    transport_ref_claim = (
        f"At least one of these pages confirms that {gt.company} runs transportation connecting DEN and the ski resort areas."
    )
    await evaluator.verify(
        claim=transport_ref_claim,
        node=transport_ref_leaf,
        sources=gt.urls,
        additional_instruction="Prefer official company pages; regional transportation authority/tourism pages are acceptable if clear."
    )


async def verify_resort_section(evaluator: Evaluator, parent_node, resort: ResortInfo) -> None:
    resort_node = evaluator.add_parallel(
        id="ski_resort_selection",
        desc="Select an appropriate ski resort for beginner skiers based on terrain and accessibility",
        parent=parent_node,
        critical=False
    )

    has_required = (resort.resort_name is not None and resort.resort_name.strip() != "" and len(resort.urls) > 0)
    evaluator.add_custom_node(
        result=has_required,
        id="resort_required_info",
        desc="Resort info includes a resort name and supporting URL(s)",
        parent=resort_node,
        critical=True
    )

    resort_ident_leaf = evaluator.add_leaf(
        id="resort_identification",
        desc="Identify a Colorado ski resort near Denver that is described as beginner-friendly",
        parent=resort_node,
        critical=True
    )
    resort_ident_claim = (
        f"{resort.resort_name} is a Colorado ski resort near Denver (within a reasonable drive) and is recognized as suitable for beginners."
    )
    await evaluator.verify(
        claim=resort_ident_claim,
        node=resort_ident_leaf,
        sources=resort.urls,
        additional_instruction=(
            "It is sufficient if the page (or one of the pages) lists the resort among 'near Denver' options or clearly indicates accessibility from Denver."
        )
    )

    beginner_evidence_leaf = evaluator.add_leaf(
        id="beginner_terrain_evidence",
        desc="Provide evidence that the selected resort is beginner-friendly (dedicated beginner areas or gentle terrain)",
        parent=resort_node,
        critical=True
    )
    beginner_claim = (
        f"{resort.resort_name} is beginner-friendly and offers terrain or programs suitable for first-time skiers. "
        f"For example: \"{resort.beginner_evidence or ''}\""
    )
    await evaluator.verify(
        claim=beginner_claim,
        node=beginner_evidence_leaf,
        sources=resort.urls,
        additional_instruction=(
            "Look for explicit mentions of beginner terrain, green runs, learning areas, ski school, or similar beginner-friendly features. "
            "Minor wording variations are acceptable."
        )
    )

    season_status_leaf = evaluator.add_leaf(
        id="season_status",
        desc="Confirm the resort's opening status or opening date for the 2025-26 ski season",
        parent=resort_node,
        critical=False
    )
    season_claim = (
        f"{resort.resort_name} is operating or scheduled to operate during the 2025–26 ski season."
    )
    await evaluator.verify(
        claim=season_claim,
        node=season_status_leaf,
        sources=resort.urls,
        additional_instruction=(
            "Confirm with resort operations, announcements, lift status, season dates, or pass information indicating 2025/26 season activity."
        )
    )

    resort_ref_leaf = evaluator.add_leaf(
        id="resort_reference_url",
        desc="Provide a valid reference URL confirming the resort's beginner-friendly characteristics",
        parent=resort_node,
        critical=True
    )
    resort_ref_claim = (
        f"At least one of these pages explicitly supports that {resort.resort_name} is beginner-friendly."
    )
    await evaluator.verify(
        claim=resort_ref_claim,
        node=resort_ref_leaf,
        sources=resort.urls,
        additional_instruction="Prefer official resort pages or reputable guides; the support must be explicit or clearly implied."
    )


async def verify_accommodation_section(evaluator: Evaluator, parent_node, accom: AccommodationInfo, resort_name: Optional[str]) -> None:
    accom_node = evaluator.add_parallel(
        id="accommodation",
        desc="Identify suitable lodging options at or near the selected ski resort",
        parent=parent_node,
        critical=False
    )

    has_required = (len(accom.urls) > 0)
    evaluator.add_custom_node(
        result=has_required,
        id="accommodation_required_info",
        desc="Accommodation info has at least one supporting URL",
        parent=accom_node,
        critical=True
    )

    lodging_leaf = evaluator.add_leaf(
        id="lodging_availability",
        desc="Confirm that lodging options are available at or near the selected ski resort",
        parent=accom_node,
        critical=True
    )
    resort_label = resort_name or "the selected resort"
    lodging_claim = f"Lodging is available at or near {resort_label}."
    await evaluator.verify(
        claim=lodging_claim,
        node=lodging_leaf,
        sources=accom.urls,
        additional_instruction="This can be on-mountain lodging or nearby accommodations within a short drive or shuttle access."
    )

    package_leaf = evaluator.add_leaf(
        id="package_options",
        desc="Identify whether the resort or area offers vacation packages that bundle lodging with lift tickets",
        parent=accom_node,
        critical=False
    )
    package_claim = f"Vacation packages bundling lodging with lift tickets are offered for stays at or near {resort_label}."
    await evaluator.verify(
        claim=package_claim,
        node=package_leaf,
        sources=accom.urls,
        additional_instruction="Look for 'lodging + lift ticket' deals or 'ski & stay' packages on the resort or lodging partners."
    )

    accom_ref_leaf = evaluator.add_leaf(
        id="accommodation_reference_url",
        desc="Provide a valid reference URL confirming lodging availability and package options",
        parent=accom_node,
        critical=True
    )
    accom_ref_claim = (
        "At least one of these pages confirms lodging availability and/or bundled lodging-with-lift-ticket packages for the resort area."
    )
    await evaluator.verify(
        claim=accom_ref_claim,
        node=accom_ref_leaf,
        sources=accom.urls,
        additional_instruction="Prefer official resort lodging pages or well-known lodging partners; regional tourism pages are acceptable."
    )


async def verify_return_activity_section(evaluator: Evaluator, parent_node, ret: ReturnActivity) -> None:
    return_node = evaluator.add_parallel(
        id="return_activity",
        desc="Identify an outdoor recreation activity or location in Wisconsin accessible after the ski trip",
        parent=parent_node,
        critical=False
    )

    park_leaf = evaluator.add_leaf(
        id="state_park_identification",
        desc="Identify a state park near Wisconsin Dells that offers hiking trails",
        parent=return_node,
        critical=False
    )
    park_claim = f"{ret.park_name} is a state park near Wisconsin Dells that offers hiking trails."
    await evaluator.verify(
        claim=park_claim,
        node=park_leaf,
        sources=ret.urls,
        additional_instruction="Prefer official Wisconsin DNR pages or official park pages; local tourism pages acceptable if clear."
    )

    proximity_leaf = evaluator.add_leaf(
        id="park_proximity",
        desc="Confirm the state park's approximate distance or proximity to Wisconsin Dells",
        parent=return_node,
        critical=False
    )
    prox_claim = f"{ret.park_name} is approximately {ret.proximity} from Wisconsin Dells."
    await evaluator.verify(
        claim=prox_claim,
        node=proximity_leaf,
        sources=ret.urls,
        additional_instruction="Distance or drive-time approximations are acceptable."
    )

    wi_ref_leaf = evaluator.add_leaf(
        id="wisconsin_reference_url",
        desc="Provide a valid reference URL confirming the state park's location and hiking opportunities",
        parent=return_node,
        critical=False
    )
    wi_ref_claim = (
        f"At least one of these pages confirms {ret.park_name}'s location near Wisconsin Dells and its hiking trails."
    )
    await evaluator.verify(
        claim=wi_ref_claim,
        node=wi_ref_leaf,
        sources=ret.urls,
        additional_instruction="Official DNR or authoritative tourism sources preferred."
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

    # Parallelize extractions
    flight_task = evaluator.extract(
        prompt=prompt_extract_flight(),
        template_class=FlightInfo,
        extraction_name="flight_info"
    )
    ground_task = evaluator.extract(
        prompt=prompt_extract_ground_transport(),
        template_class=GroundTransport,
        extraction_name="ground_transport"
    )
    resort_task = evaluator.extract(
        prompt=prompt_extract_resort(),
        template_class=ResortInfo,
        extraction_name="resort_info"
    )
    accom_task = evaluator.extract(
        prompt=prompt_extract_accommodation(),
        template_class=AccommodationInfo,
        extraction_name="accommodation_info"
    )
    return_task = evaluator.extract(
        prompt=prompt_extract_return_activity(),
        template_class=ReturnActivity,
        extraction_name="return_activity"
    )

    flight_info, ground_info, resort_info, accom_info, return_info = await asyncio.gather(
        flight_task, ground_task, resort_task, accom_task, return_task
    )

    # Build verification tree and verify sections
    await verify_flight_section(evaluator, root, flight_info)
    await verify_ground_transport_section(evaluator, root, ground_info)
    await verify_resort_section(evaluator, root, resort_info)
    await verify_accommodation_section(evaluator, root, accom_info, resort_info.resort_name)
    await verify_return_activity_section(evaluator, root, return_info)

    return evaluator.get_summary()