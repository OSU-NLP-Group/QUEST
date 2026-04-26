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
TASK_ID = "outdoor_recreation_trip_2026"
TASK_DESCRIPTION = (
    "A family is planning a comprehensive outdoor recreation trip for 2026 and needs detailed logistical information. "
    "Please provide the following information:\n\n"
    "1. Flight Information: Identify which airline offers direct flights from Atlanta (ATL) to Turks and Caicos "
    "(Providenciales), and specify which concourse this airline operates from at Atlanta's Hartsfield-Jackson "
    "International Airport.\n\n"
    "2. Resort Activity Details: For kayak rentals at Disney's Fort Wilderness Resort and Campground:\n"
    "- What is the specific facility name and recreation area where kayak rentals are available?\n"
    "- What are the daily operating hours?\n"
    "- What is the rental cost and time increment?\n"
    "- What is the minimum age to rent independently with ID, and what is the minimum age for youth to rent with "
    "parent/guardian signature?\n\n"
    "3. Wilderness Camping Permit Information: For Everglades National Park wilderness camping permits:\n"
    "- How many days in advance can permits be reserved?\n"
    "- At what time (including time zone) do permit reservations open each day?\n"
    "- What is the non-refundable reservation fee and the per-person per-night recreation fee?\n"
    "- What is the name of the online platform used for reservations?\n\n"
    "4. Airline Industry Update: Regarding the recent merger between Allegiant Air and Sun Country Airlines:\n"
    "- On what date was this merger announced?\n"
    "- Which airline is acquiring the other?\n"
    "- How many international routes to Mexico, Central America, Caribbean, and Canada did the acquired airline "
    "operate that the acquiring airline will gain access to?\n\n"
    "For each piece of information, provide a supporting reference URL."
)


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class FlightInfo(BaseModel):
    airline_name: Optional[str] = None
    flight_service_urls: List[str] = Field(default_factory=list)  # URLs supporting direct ATL-PLS service
    atl_concourse: Optional[str] = None
    concourse_urls: List[str] = Field(default_factory=list)       # URLs supporting ATL concourse


class ResortKayakInfo(BaseModel):
    facility_and_area: Optional[str] = None
    location_urls: List[str] = Field(default_factory=list)

    daily_operating_hours: Optional[str] = None
    rental_cost_structure: Optional[str] = None
    operations_urls: List[str] = Field(default_factory=list)

    min_age_independent: Optional[str] = None        # e.g., "18+" or "18 years"
    min_age_with_guardian: Optional[str] = None      # e.g., "12+" or "12 years"
    id_types_accepted: Optional[str] = None          # optional detail
    requirements_urls: List[str] = Field(default_factory=list)


class EvergladesPermitInfo(BaseModel):
    advance_reservation_period: Optional[str] = None   # e.g., "90 days (rolling daily basis)"
    booking_window_urls: List[str] = Field(default_factory=list)

    daily_release_time_and_zone: Optional[str] = None  # e.g., "10:00 AM Eastern Time"
    release_time_urls: List[str] = Field(default_factory=list)

    permit_fees: Optional[str] = None                  # e.g., "$21 non-refundable reservation fee; $2 per person per night"
    entrance_fee_note: Optional[str] = None            # optional: note about separate park entrance fee
    fee_structure_urls: List[str] = Field(default_factory=list)

    platform_name: Optional[str] = None                # e.g., "Recreation.gov"
    platform_urls: List[str] = Field(default_factory=list)


class MergerInfo(BaseModel):
    merger_announcement_date: Optional[str] = None     # e.g., "January 11, 2026"
    acquiring_airline: Optional[str] = None            # e.g., "Allegiant Air"
    acquired_airline: Optional[str] = None             # e.g., "Sun Country Airlines"
    announcement_urls: List[str] = Field(default_factory=list)

    international_route_count: Optional[str] = None    # e.g., "18"
    destination_regions: Optional[str] = None          # e.g., "Mexico, Caribbean, Central America, Canada"
    route_expansion_urls: List[str] = Field(default_factory=list)


class TripExtraction(BaseModel):
    flight: Optional[FlightInfo] = None
    resort: Optional[ResortKayakInfo] = None
    permit: Optional[EvergladesPermitInfo] = None
    merger: Optional[MergerInfo] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_trip_info() -> str:
    return """
Extract the specific information requested below strictly from the provided answer text. For each item that asks for a URL reference, extract all explicit URLs mentioned for that item.

1) Flight Information (ATL → Turks and Caicos (Providenciales, PLS)):
- airline_name: Name of the airline claimed to offer direct/nonstop ATL–PLS service.
- flight_service_urls: All URLs that support the claim of direct/nonstop ATL–PLS service.
- atl_concourse: Specific concourse (e.g., “Concourse F”) where this airline operates at ATL.
- concourse_urls: All URLs that support the concourse/terminal assignment at ATL.

2) Disney’s Fort Wilderness Resort & Campground — Kayak Rentals:
- facility_and_area: The named facility and recreation area where kayak rentals occur (e.g., “Bike Barn at the Meadow Recreation Area”).
- location_urls: All URLs that support where the kayak rentals are located.
- daily_operating_hours: The daily operating hours string as presented (e.g., “9:00 AM – 5:00 PM”).
- rental_cost_structure: The fee and time increment as presented (e.g., “$13 per 30 minutes”).
- operations_urls: All URLs that support the hours and pricing.
- min_age_independent: Minimum age to rent independently with valid ID (e.g., “18+” or “18 years”).
- min_age_with_guardian: Minimum age for a youth to rent with parent/guardian signature (e.g., “12+”).
- id_types_accepted: If present, the types of ID accepted (free text).
- requirements_urls: All URLs that support age/ID requirements.

3) Everglades National Park — Wilderness Camping Permits:
- advance_reservation_period: How far in advance permits can be reserved, as stated (e.g., “90 days (rolling daily basis)”).
- booking_window_urls: All URLs supporting the advance reservation period.
- daily_release_time_and_zone: The daily release time and the time zone (e.g., “10:00 AM Eastern Time”).
- release_time_urls: All URLs supporting the daily release time.
- permit_fees: Fee structure as stated, including non-refundable reservation fee and per-person per-night recreation fee (e.g., “$21 reservation fee; $2 per person per night”).
- entrance_fee_note: If present, note that a separate park entrance fee may apply.
- fee_structure_urls: All URLs supporting the permit fee structure.
- platform_name: Name of the online reservation platform (e.g., “Recreation.gov”).
- platform_urls: All URLs supporting the platform name.

4) Airline Industry Update — Allegiant Air & Sun Country Airlines:
- merger_announcement_date: The announcement date as presented (e.g., “January 11, 2026”).
- acquiring_airline: Which airline is acquiring the other.
- acquired_airline: Which airline is being acquired.
- announcement_urls: All URLs supporting announcement details (date and acquiring/acquired relationship).
- international_route_count: The number of international routes to Mexico, Central America, Caribbean, and Canada that will be gained (e.g., “18”).
- destination_regions: If stated, the list of regions (e.g., “Mexico, Caribbean, Central America, Canada”).
- route_expansion_urls: All URLs supporting the international route count and regions.

Rules:
- Extract only what appears in the answer text; do not invent or infer information.
- If an item is not mentioned, set it to null (or an empty list for URL fields).
- For URL fields, include every explicit URL shown (plain or markdown). Do not infer URLs.
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _nonempty(s: Optional[str]) -> bool:
    return bool(s and str(s).strip())


def _has_urls(urls: Optional[List[str]]) -> bool:
    return bool(urls and len(urls) > 0)


def _fail_leaf_due_to_missing_sources(node):
    node.score = 0.0
    node.status = "failed"


# --------------------------------------------------------------------------- #
# Verification builders                                                       #
# --------------------------------------------------------------------------- #
async def verify_flight_information(evaluator: Evaluator, parent, flight: Optional[FlightInfo]) -> None:
    node_air = evaluator.add_sequential(
        id="airline_flight_information",
        desc="Information about airline service from Atlanta to a Caribbean beach destination",
        parent=parent,
        critical=True  # All children under this will be critical
    )

    # Child 1: Flight service identification
    flight_ident = evaluator.add_parallel(
        id="flight_service_identification",
        desc="Identify the airline offering direct flights from Atlanta to Turks and Caicos with URL reference",
        parent=node_air,
        critical=True
    )

    # Leaf: airline name existence
    airline_exists = evaluator.add_custom_node(
        result=_nonempty(flight.airline_name) if flight else False,
        id="airline_name",
        desc="Name of the airline providing direct service from ATL to Turks and Caicos",
        parent=flight_ident,
        critical=True
    )

    # Leaf: direct service supported by URL(s)
    direct_ref = evaluator.add_leaf(
        id="flight_service_reference",
        desc="URL reference confirming the airline's direct flight service from Atlanta to Turks and Caicos",
        parent=flight_ident,
        critical=True
    )
    airline_name = (flight.airline_name or "").strip() if flight else ""
    direct_urls = flight.flight_service_urls if (flight and flight.flight_service_urls) else []
    if not _has_urls(direct_urls):
        _fail_leaf_due_to_missing_sources(direct_ref)
    else:
        claim_direct = (
            f"{airline_name} operates direct (nonstop) flights from Atlanta (ATL) to Providenciales (PLS), "
            f"Turks and Caicos."
        )
        await evaluator.verify(
            claim=claim_direct,
            node=direct_ref,
            sources=direct_urls,
            additional_instruction=(
                "Verify that the page explicitly shows a nonstop (direct) route between Atlanta (ATL) and "
                "Providenciales (PLS) or lists ATL–PLS as a direct/nonstop service. Seasonal or route map confirmation "
                "is acceptable. If the page is irrelevant or does not clearly support this, mark as not supported."
            )
        )

    # Child 2: Airport terminal/concourse details (sequential after flight identification)
    terminal = evaluator.add_parallel(
        id="airport_terminal_details",
        desc="Terminal/concourse information for the identified airline at Atlanta airport with URL reference",
        parent=node_air,
        critical=True
    )

    # Leaf: concourse identification existence
    concourse_exists = evaluator.add_custom_node(
        result=_nonempty(flight.atl_concourse) if flight else False,
        id="concourse_identification",
        desc="Specific concourse where the airline operates at Atlanta airport",
        parent=terminal,
        critical=True
    )

    # Leaf: concourse verification by URL(s)
    concourse_ref = evaluator.add_leaf(
        id="terminal_reference",
        desc="URL reference confirming the airline's terminal/concourse at Atlanta airport",
        parent=terminal,
        critical=True
    )
    concourse_name = (flight.atl_concourse or "").strip() if flight else ""
    concourse_urls = flight.concourse_urls if (flight and flight.concourse_urls) else []
    if not _has_urls(concourse_urls):
        _fail_leaf_due_to_missing_sources(concourse_ref)
    else:
        claim_concourse = (
            f"At Hartsfield-Jackson Atlanta International Airport (ATL), {airline_name} operates from {concourse_name}."
        )
        await evaluator.verify(
            claim=claim_concourse,
            node=concourse_ref,
            sources=concourse_urls,
            additional_instruction=(
                "Confirm the airline's ATL terminal/concourse assignment. Minor naming variations like "
                "'Concourse F (International Terminal)' vs 'Concourse F' should be considered a match."
            )
        )


async def verify_resort_activity(evaluator: Evaluator, parent, resort: Optional[ResortKayakInfo]) -> None:
    node_resort = evaluator.add_parallel(
        id="resort_activity_details",
        desc="Specific details about kayak rental operations at Disney's Fort Wilderness Resort",
        parent=parent,
        critical=False  # Mixed criticality within children
    )

    # 1) Rental location info
    loc = evaluator.add_parallel(
        id="rental_location_info",
        desc="Facility name, recreation area location, and URL reference for kayak rentals",
        parent=node_resort,
        critical=True
    )
    # existence
    evaluator.add_custom_node(
        result=_nonempty(resort.facility_and_area) if resort else False,
        id="facility_and_area",
        desc="Name of the facility and recreation area where kayak rentals are available",
        parent=loc,
        critical=True
    )
    # verification by URL
    loc_ref = evaluator.add_leaf(
        id="location_reference",
        desc="URL reference confirming the kayak rental location details",
        parent=loc,
        critical=True
    )
    facility_area = (resort.facility_and_area or "").strip() if resort else ""
    loc_urls = resort.location_urls if (resort and resort.location_urls) else []
    if not _has_urls(loc_urls):
        _fail_leaf_due_to_missing_sources(loc_ref)
    else:
        claim_loc = f"At Disney's Fort Wilderness Resort & Campground, kayak rentals are available at {facility_area}."
        await evaluator.verify(
            claim=claim_loc,
            node=loc_ref,
            sources=loc_urls,
            additional_instruction=(
                "Verify that the page shows kayak rentals occurring at this named facility/area. "
                "Accept reasonable naming variants like 'Bike Barn' and 'Meadow Recreation Area'."
            )
        )

    # 2) Operating hours and pricing
    ops = evaluator.add_parallel(
        id="operating_hours_and_pricing",
        desc="Operating hours and pricing information with URL reference",
        parent=node_resort,
        critical=True
    )
    evaluator.add_custom_node(
        result=_nonempty(resort.daily_operating_hours) if resort else False,
        id="daily_operating_hours",
        desc="Daily operating hours (opening and closing times) for kayak rentals",
        parent=ops,
        critical=True
    )
    evaluator.add_custom_node(
        result=_nonempty(resort.rental_cost_structure) if resort else False,
        id="rental_cost_structure",
        desc="Rental fee amount and time increment",
        parent=ops,
        critical=True
    )
    ops_ref = evaluator.add_leaf(
        id="operations_reference",
        desc="URL reference confirming hours and pricing information",
        parent=ops,
        critical=True
    )
    hours = (resort.daily_operating_hours or "").strip() if resort else ""
    pricing = (resort.rental_cost_structure or "").strip() if resort else ""
    ops_urls = resort.operations_urls if (resort and resort.operations_urls) else []
    if not _has_urls(ops_urls):
        _fail_leaf_due_to_missing_sources(ops_ref)
    else:
        claim_ops = (
            f"For kayak rentals at Disney's Fort Wilderness Resort & Campground, the posted daily operating hours are "
            f"'{hours}', and the rental pricing/time increment is '{pricing}'."
        )
        await evaluator.verify(
            claim=claim_ops,
            node=ops_ref,
            sources=ops_urls,
            additional_instruction=(
                "Confirm both the hours and the pricing/time increment from the page. Allow minor formatting differences "
                "like '9am' vs '9:00 AM', or commas vs semicolons. If the page lists seasonal variations, the claimed "
                "values must be clearly supported within that context."
            )
        )

    # 3) Rental requirements info
    req = evaluator.add_parallel(
        id="rental_requirements_info",
        desc="Age and identification requirements with URL reference",
        parent=node_resort,
        critical=False  # Contains a non-critical child
    )
    evaluator.add_custom_node(
        result=(_nonempty(resort.min_age_independent) and _nonempty(resort.min_age_with_guardian)) if resort else False,
        id="age_requirements",
        desc="Minimum age to rent independently (18+) and minimum age for youth with parent/guardian permission (12+)",
        parent=req,
        critical=True
    )
    evaluator.add_custom_node(
        result=_nonempty(resort.id_types_accepted) if resort else False,
        id="id_types_accepted",
        desc="Types of valid identification accepted for rental",
        parent=req,
        critical=False
    )
    req_ref = evaluator.add_leaf(
        id="requirements_reference",
        desc="URL reference confirming age and ID requirements",
        parent=req,
        critical=True
    )
    age_ind = (resort.min_age_independent or "").strip() if resort else ""
    age_guard = (resort.min_age_with_guardian or "").strip() if resort else ""
    req_urls = resort.requirements_urls if (resort and resort.requirements_urls) else []
    if not _has_urls(req_urls):
        _fail_leaf_due_to_missing_sources(req_ref)
    else:
        claim_req = (
            f"For Fort Wilderness kayak rentals, the minimum age to rent independently with valid ID is {age_ind}, "
            f"and the minimum age to rent with a parent/guardian signature is {age_guard}."
        )
        await evaluator.verify(
            claim=claim_req,
            node=req_ref,
            sources=req_urls,
            additional_instruction=(
                "Verify both age thresholds. Accept equivalent phrasing like '18 years (with ID)' and '12 years (with "
                "parent/guardian consent)'. Focus on age requirements; ID type details are optional."
            )
        )


async def verify_permit_info(evaluator: Evaluator, parent, permit: Optional[EvergladesPermitInfo]) -> None:
    node_permit = evaluator.add_parallel(
        id="wilderness_permit_booking",
        desc="Booking procedures and requirements for Everglades National Park wilderness camping permits",
        parent=parent,
        critical=False  # Contains a child section with non-critical note
    )

    # 1) Booking window info
    bw = evaluator.add_parallel(
        id="booking_window_info",
        desc="Advance booking window with URL reference",
        parent=node_permit,
        critical=True
    )
    evaluator.add_custom_node(
        result=_nonempty(permit.advance_reservation_period) if permit else False,
        id="advance_reservation_period",
        desc="Number of days in advance (90 days) and confirmation of rolling daily basis",
        parent=bw,
        critical=True
    )
    bw_ref = evaluator.add_leaf(
        id="booking_window_reference",
        desc="URL reference confirming the advance booking window",
        parent=bw,
        critical=True
    )
    adv = (permit.advance_reservation_period or "").strip() if permit else ""
    bw_urls = permit.booking_window_urls if (permit and permit.booking_window_urls) else []
    if not _has_urls(bw_urls):
        _fail_leaf_due_to_missing_sources(bw_ref)
    else:
        claim_bw = (
            f"Everglades wilderness permits can be reserved up to {adv} in advance on a rolling daily basis."
        )
        await evaluator.verify(
            claim=claim_bw,
            node=bw_ref,
            sources=bw_urls,
            additional_instruction=(
                "Confirm that the policy states 90 days in advance on a rolling basis (or as claimed). "
                "The page must explicitly support both the maximum advance window and the rolling daily release."
            )
        )

    # 2) Daily release time
    rt = evaluator.add_parallel(
        id="release_time_info",
        desc="Daily release time with URL reference",
        parent=node_permit,
        critical=True
    )
    evaluator.add_custom_node(
        result=_nonempty(permit.daily_release_time_and_zone) if permit else False,
        id="daily_release_time_and_zone",
        desc="Time of day when reservations open and time zone (10:00 AM Eastern Time)",
        parent=rt,
        critical=True
    )
    rt_ref = evaluator.add_leaf(
        id="release_time_reference",
        desc="URL reference confirming the daily release time",
        parent=rt,
        critical=True
    )
    rel = (permit.daily_release_time_and_zone or "").strip() if permit else ""
    rt_urls = permit.release_time_urls if (permit and permit.release_time_urls) else []
    if not _has_urls(rt_urls):
        _fail_leaf_due_to_missing_sources(rt_ref)
    else:
        claim_rt = f"Permit reservations open each day at {rel}."
        await evaluator.verify(
            claim=claim_rt,
            node=rt_ref,
            sources=rt_urls,
            additional_instruction=(
                "Confirm the daily release time and time zone (e.g., '10:00 AM ET' or '10 am Eastern Time'). "
                "Accept minor formatting differences."
            )
        )

    # 3) Fee information
    fees = evaluator.add_parallel(
        id="fee_information",
        desc="Permit fee structure with URL reference",
        parent=node_permit,
        critical=False  # Contains a non-critical child
    )
    evaluator.add_custom_node(
        result=_nonempty(permit.permit_fees) if permit else False,
        id="permit_fees",
        desc="Non-refundable reservation fee ($21) and per-person per-night recreation fee ($2)",
        parent=fees,
        critical=True
    )
    evaluator.add_custom_node(
        result=_nonempty(permit.entrance_fee_note) if permit else False,
        id="separate_entrance_fee_note",
        desc="Note about separate park entrance fee requirement",
        parent=fees,
        critical=False
    )
    fees_ref = evaluator.add_leaf(
        id="fee_structure_reference",
        desc="URL reference confirming the fee structure",
        parent=fees,
        critical=True
    )
    fees_str = (permit.permit_fees or "").strip() if permit else ""
    fee_urls = permit.fee_structure_urls if (permit and permit.fee_structure_urls) else []
    if not _has_urls(fee_urls):
        _fail_leaf_due_to_missing_sources(fees_ref)
    else:
        claim_fees = (
            f"The Everglades wilderness permit fees are as stated: {fees_str}."
        )
        await evaluator.verify(
            claim=claim_fees,
            node=fees_ref,
            sources=fee_urls,
            additional_instruction=(
                "Verify both the non-refundable reservation fee (e.g., $21) and the per-person per-night recreation fee "
                "(e.g., $2). Minor formatting differences are acceptable; amounts must match."
            )
        )

    # 4) Reservation platform
    plat = evaluator.add_parallel(
        id="reservation_platform_info",
        desc="Online booking platform with URL reference",
        parent=node_permit,
        critical=True
    )
    evaluator.add_custom_node(
        result=_nonempty(permit.platform_name) if permit else False,
        id="platform_name",
        desc="Name of the reservation system/website (Recreation.gov)",
        parent=plat,
        critical=True
    )
    plat_ref = evaluator.add_leaf(
        id="platform_reference",
        desc="URL reference confirming the booking platform",
        parent=plat,
        critical=True
    )
    platform_name = (permit.platform_name or "").strip() if permit else ""
    plat_urls = permit.platform_urls if (permit and permit.platform_urls) else []
    if not _has_urls(plat_urls):
        _fail_leaf_due_to_missing_sources(plat_ref)
    else:
        claim_plat = f"Everglades wilderness permit reservations are made on {platform_name}."
        await evaluator.verify(
            claim=claim_plat,
            node=plat_ref,
            sources=plat_urls,
            additional_instruction=(
                "Confirm explicitly that the reservation platform is Recreation.gov (or as claimed)."
            )
        )


async def verify_merger_info(evaluator: Evaluator, parent, merger: Optional[MergerInfo]) -> None:
    node_merger = evaluator.add_parallel(
        id="airline_merger_information",
        desc="Information about recent airline merger affecting leisure travel routes",
        parent=parent,
        critical=False  # Contains a child section with a non-critical leaf
    )

    # 1) Merger details
    md = evaluator.add_parallel(
        id="merger_details",
        desc="Merger announcement details with URL reference",
        parent=node_merger,
        critical=True
    )
    evaluator.add_custom_node(
        result=(
            _nonempty(merger.merger_announcement_date) and
            _nonempty(merger.acquiring_airline) and
            _nonempty(merger.acquired_airline)
        ) if merger else False,
        id="merger_announcement_facts",
        desc="Announcement date (e.g., January 11, 2026), acquiring airline, and acquired airline",
        parent=md,
        critical=True
    )
    md_ref = evaluator.add_leaf(
        id="merger_announcement_reference",
        desc="URL reference confirming the merger announcement details",
        parent=md,
        critical=True
    )
    ann_date = (merger.merger_announcement_date or "").strip() if merger else ""
    acquirer = (merger.acquiring_airline or "").strip() if merger else ""
    acquiree = (merger.acquired_airline or "").strip() if merger else ""
    md_urls = merger.announcement_urls if (merger and merger.announcement_urls) else []
    if not _has_urls(md_urls):
        _fail_leaf_due_to_missing_sources(md_ref)
    else:
        claim_md = (
            f"On {ann_date}, {acquirer} announced that it would acquire {acquiree} (i.e., a merger/acquisition deal)."
        )
        await evaluator.verify(
            claim=claim_md,
            node=md_ref,
            sources=md_urls,
            additional_instruction=(
                "Confirm both the announcement date and that the acquiring airline is acquiring the other. "
                "Accept phrasing like 'announced a merger' or 'agreed to acquire'."
            )
        )

    # 2) Route expansion details
    rexp = evaluator.add_parallel(
        id="route_expansion_details",
        desc="International route expansion with URL reference",
        parent=node_merger,
        critical=False
    )
    evaluator.add_custom_node(
        result=_nonempty(merger.international_route_count) if merger else False,
        id="international_route_count",
        desc="Number of international routes that the acquirer will gain access to",
        parent=rexp,
        critical=True
    )
    evaluator.add_custom_node(
        result=_nonempty(merger.destination_regions) if merger else False,
        id="destination_regions",
        desc="Geographic regions covered by the routes (Mexico, Caribbean, Central America, Canada)",
        parent=rexp,
        critical=False
    )
    rexp_ref = evaluator.add_leaf(
        id="route_expansion_reference",
        desc="URL reference confirming the international route details",
        parent=rexp,
        critical=True
    )
    route_count = (merger.international_route_count or "").strip() if merger else ""
    regions = (merger.destination_regions or "").strip() if merger else ""
    rexp_urls = merger.route_expansion_urls if (merger and merger.route_expansion_urls) else []
    if not _has_urls(rexp_urls):
        _fail_leaf_due_to_missing_sources(rexp_ref)
    else:
        claim_routes = (
            f"{acquirer} will gain access to {route_count} international routes previously operated by {acquiree}."
        )
        await evaluator.verify(
            claim=claim_routes,
            node=rexp_ref,
            sources=rexp_urls,
            additional_instruction=(
                "Verify the international route count. If the page also references regions (e.g., Mexico, Caribbean, "
                "Central America, Canada), consider that context supportive but focus on the numeric count."
            )
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
    Evaluate an answer for the comprehensive outdoor recreation trip planning task.
    """
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

    # Extraction
    trip_info = await evaluator.extract(
        prompt=prompt_extract_trip_info(),
        template_class=TripExtraction,
        extraction_name="trip_info"
    )

    # Build and verify subtrees
    await verify_flight_information(evaluator, root, trip_info.flight)
    await verify_resort_activity(evaluator, root, trip_info.resort)
    await verify_permit_info(evaluator, root, trip_info.permit)
    await verify_merger_info(evaluator, root, trip_info.merger)

    return evaluator.get_summary()