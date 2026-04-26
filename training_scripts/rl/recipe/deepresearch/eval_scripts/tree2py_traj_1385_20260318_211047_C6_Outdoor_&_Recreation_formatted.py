import asyncio
import logging
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "eu_accessible_venues"
TASK_DESCRIPTION = (
    "I'm planning accessible family trips to multiple recreational venues in Europe and need detailed visitor "
    "information for two specific types of facilities.\n\n"
    "Please identify:\n\n"
    "Venue 1: A zoo in the United Kingdom that meets ALL of the following requirements:\n"
    "- Offers manual wheelchair rental with a refundable deposit system (provide the exact deposit amount in GBP)\n"
    "- Is accessible by at least two specific numbered public bus routes (provide the route numbers and confirm buses "
    "stop at or outside the zoo entrance)\n"
    "- Has on-site parking for visitors (provide the daily parking fee in GBP for regular visitors)\n"
    "- Provides specific operating hours for March (provide both opening and closing times, and confirm the last entry "
    "policy relative to closing time)\n"
    "- Has a defined annual closure policy (confirm whether it closes on December 25, Christmas Day)\n\n"
    "Venue 2: A major international airport in Germany that meets ALL of the following requirements:\n"
    "- Is served by S-Bahn train service to the city center with at least two line numbers (provide the line numbers "
    "and the service frequency interval in minutes)\n"
    "- Operates a shuttle bus service between terminals (provide the frequency during peak hours 07:00-17:00 and "
    "outside peak hours)\n"
    "- Has designated parking garages for Terminal 2 (identify at least two parking garage codes, such as P20, P26, etc.)\n"
    "- Has at least one passenger lounge facility in Terminal 1 (provide the lounge name and operating hours)\n"
    "- Is located in a specific public transport fare zone (provide the zone number)\n\n"
    "For each venue, provide the facility name, and for each requirement, include a reference URL that supports the information."
)


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class ZooWheelchairInfo(BaseModel):
    availability_statement: Optional[str] = None  # e.g., "Manual wheelchairs available to hire"
    deposit_gbp: Optional[str] = None             # e.g., "£20", "20 GBP"
    urls: List[str] = Field(default_factory=list) # URLs supporting wheelchair rental info


class ZooTransportInfo(BaseModel):
    bus_routes: List[str] = Field(default_factory=list)  # e.g., ["12", "34", "X5"]
    stop_location_statement: Optional[str] = None        # e.g., "Buses stop outside the main entrance"
    urls: List[str] = Field(default_factory=list)        # URLs supporting bus info


class ZooParkingInfo(BaseModel):
    availability_statement: Optional[str] = None  # e.g., "On-site parking available"
    daily_fee_gbp: Optional[str] = None           # e.g., "£7", "7 GBP"
    urls: List[str] = Field(default_factory=list) # URLs supporting parking info


class ZooHoursInfo(BaseModel):
    march_opening_time: Optional[str] = None      # e.g., "10:00"
    march_closing_time: Optional[str] = None      # e.g., "17:00"
    last_entry_policy: Optional[str] = None       # e.g., "Last entry 1 hour before closing"
    urls: List[str] = Field(default_factory=list) # URLs supporting hours info


class ZooClosureInfo(BaseModel):
    christmas_day_closed: Optional[str] = None    # e.g., "Closed on December 25"
    urls: List[str] = Field(default_factory=list) # URLs supporting closure info


class ZooVenue(BaseModel):
    name: Optional[str] = None
    wheelchair: ZooWheelchairInfo = ZooWheelchairInfo()
    transport: ZooTransportInfo = ZooTransportInfo()
    parking: ZooParkingInfo = ZooParkingInfo()
    hours: ZooHoursInfo = ZooHoursInfo()
    closure: ZooClosureInfo = ZooClosureInfo()


class AirportPublicTransportInfo(BaseModel):
    sbahn_lines: List[str] = Field(default_factory=list)   # e.g., ["S1", "S8"]
    frequency_minutes: Optional[str] = None                # e.g., "10", "10-20"
    urls: List[str] = Field(default_factory=list)          # URLs supporting S-Bahn info


class AirportShuttleInfo(BaseModel):
    availability_statement: Optional[str] = None           # e.g., "Shuttle between terminals"
    peak_frequency: Optional[str] = None                   # e.g., "5-10 minutes"
    offpeak_frequency: Optional[str] = None                # e.g., "10-20 minutes"
    urls: List[str] = Field(default_factory=list)          # URLs supporting shuttle info


class AirportParkingInfo(BaseModel):
    terminal2_parking_codes: List[str] = Field(default_factory=list)  # e.g., ["P20", "P26"]
    urls: List[str] = Field(default_factory=list)                     # URLs supporting parking designation


class AirportLoungeInfo(BaseModel):
    t1_lounge_name: Optional[str] = None
    t1_lounge_hours: Optional[str] = None
    urls: List[str] = Field(default_factory=list)                     # URLs supporting lounge info


class AirportZoneInfo(BaseModel):
    zone_number: Optional[str] = None                                 # e.g., "M" or "1"
    urls: List[str] = Field(default_factory=list)                     # URLs supporting zone info


class AirportVenue(BaseModel):
    name: Optional[str] = None
    public_transport: AirportPublicTransportInfo = AirportPublicTransportInfo()
    shuttle: AirportShuttleInfo = AirportShuttleInfo()
    parking: AirportParkingInfo = AirportParkingInfo()
    lounge: AirportLoungeInfo = AirportLoungeInfo()
    zone: AirportZoneInfo = AirportZoneInfo()


class VenuesExtraction(BaseModel):
    zoo: ZooVenue = ZooVenue()
    airport: AirportVenue = AirportVenue()


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_venues() -> str:
    return """
    Extract structured information for two venues described in the answer.

    VENUE 1: UK Zoo
    - name: The zoo's name.
    - wheelchair:
        - availability_statement: Exact sentence/phrase indicating manual wheelchair rental is available (hire/loan acceptable).
        - deposit_gbp: The refundable deposit amount as written (preserve currency formatting, e.g., "£20" or "20 GBP").
        - urls: All URLs cited for wheelchair rental details.
    - transport:
        - bus_routes: A list of at least two specific public bus route numbers serving the zoo (as strings; include all mentioned).
        - stop_location_statement: Exact sentence/phrase confirming buses stop at or outside the zoo entrance.
        - urls: All URLs cited for public transport info.
    - parking:
        - availability_statement: Exact sentence/phrase confirming on-site parking for visitors.
        - daily_fee_gbp: The daily parking fee for regular visitors in GBP (preserve formatting).
        - urls: All URLs cited for parking info.
    - hours:
        - march_opening_time: Opening time in March (string as written, e.g., "10:00").
        - march_closing_time: Closing time in March (string as written, e.g., "17:00").
        - last_entry_policy: Exact sentence/phrase describing last entry relative to closing time (e.g., "Last entry 1 hour before close").
        - urls: All URLs cited for operating hours info.
    - closure:
        - christmas_day_closed: Exact sentence/phrase that confirms whether the zoo is closed on December 25 (Christmas Day).
        - urls: All URLs cited for closure info.

    VENUE 2: German International Airport
    - name: The airport's full name.
    - public_transport:
        - sbahn_lines: At least two S-Bahn line numbers serving the airport to the city center (e.g., ["S1", "S8"]).
        - frequency_minutes: The service frequency interval in minutes (string as written, e.g., "10", "10-20").
        - urls: All URLs cited for S-Bahn info.
    - shuttle:
        - availability_statement: Exact sentence/phrase confirming a shuttle bus service exists between terminals.
        - peak_frequency: Shuttle frequency during 07:00–17:00 (string as written).
        - offpeak_frequency: Shuttle frequency outside those hours (string as written).
        - urls: All URLs cited for shuttle info.
    - parking:
        - terminal2_parking_codes: At least two parking garage codes designated for Terminal 2 (e.g., ["P20", "P26"]).
        - urls: All URLs cited for parking designation info.
    - lounge:
        - t1_lounge_name: Name of at least one passenger lounge in Terminal 1.
        - t1_lounge_hours: Operating hours for that lounge (string as written).
        - urls: All URLs cited for lounge info.
    - zone:
        - zone_number: The public transport fare zone number (or code) for the airport.
        - urls: All URLs cited for zone info.

    RULES:
    - Extract only what is explicitly stated in the answer.
    - Preserve original formatting for times and currency (e.g., "£", "GBP", "10:00").
    - If an item is not mentioned, set it to null or an empty list as appropriate.
    - For all URL lists, include only valid URLs explicitly present in the answer text (or markdown links). If none, return an empty list.
    """


# --------------------------------------------------------------------------- #
# Helper functions for verification                                           #
# --------------------------------------------------------------------------- #
def _normalize_urls(urls: Optional[List[str]]) -> List[str]:
    if not urls:
        return []
    # Deduplicate while preserving order
    seen = set()
    out = []
    for u in urls:
        if not u:
            continue
        if u not in seen:
            seen.add(u)
            out.append(u)
    return out


def _join_comma(items: List[str]) -> str:
    return ", ".join([s.strip() for s in items if s and s.strip()])


async def verify_with_urls_or_fail(
    evaluator: Evaluator,
    *,
    node_id: str,
    desc: str,
    parent,
    claim: Optional[str],
    sources: Optional[List[str]],
    additional_instruction: str,
    critical: bool = True,
) -> None:
    """
    If claim and sources exist, create a verifying leaf and run URL-grounded verification.
    Otherwise, create a failing custom node with the same ID.
    """
    src = _normalize_urls(sources)
    if claim and src:
        node = evaluator.add_leaf(
            id=node_id,
            desc=desc,
            parent=parent,
            critical=critical,
        )
        await evaluator.verify(
            claim=claim,
            node=node,
            sources=src,
            additional_instruction=additional_instruction,
        )
    else:
        evaluator.add_custom_node(
            result=False,
            id=node_id,
            desc=desc,
            parent=parent,
            critical=critical,
        )


def add_reference_presence_check(
    evaluator: Evaluator,
    *,
    node_id: str,
    desc: str,
    parent,
    sources: Optional[List[str]],
    critical: bool = True,
) -> None:
    src = _normalize_urls(sources)
    evaluator.add_custom_node(
        result=len(src) > 0,
        id=node_id,
        desc=desc,
        parent=parent,
        critical=critical,
    )


# --------------------------------------------------------------------------- #
# Venue-specific verification subtrees                                        #
# --------------------------------------------------------------------------- #
async def verify_uk_zoo(evaluator: Evaluator, parent, zoo: ZooVenue) -> None:
    """
    Build and verify the subtree for the UK zoo with all required checks.
    """
    zoo_node = evaluator.add_parallel(
        id="venue_1_zoo",
        desc="Identify a zoo in the UK that meets accessibility and service requirements",
        parent=parent,
        critical=False,
    )
    zoo_name = zoo.name or "the zoo"

    # 1) Wheelchair rental with refundable deposit
    wc_node = evaluator.add_parallel(
        id="zoo_wheelchair_rental",
        desc="Verify the zoo offers wheelchair rental with refundable deposit",
        parent=zoo_node,
        critical=True,
    )

    # 1.a Availability
    await verify_with_urls_or_fail(
        evaluator,
        node_id="wheelchair_availability",
        desc="Confirm manual wheelchairs are available for rental",
        parent=wc_node,
        claim=f"The page(s) state that manual wheelchairs are available to rent (hire/loan) at {zoo_name}.",
        sources=zoo.wheelchair.urls,
        additional_instruction=(
            "Look for explicit mentions of manual wheelchair rental/loan/hire. Synonyms like 'hire' or 'loan' are acceptable."
        ),
        critical=True,
    )

    # 1.b Deposit amount (GBP)
    if zoo.wheelchair.deposit_gbp and zoo.wheelchair.deposit_gbp.strip():
        deposit_text = zoo.wheelchair.deposit_gbp.strip()
        await verify_with_urls_or_fail(
            evaluator,
            node_id="wheelchair_deposit_amount",
            desc="Provide the refundable deposit amount for manual wheelchair rental",
            parent=wc_node,
            claim=(
                f"The refundable deposit for renting a manual wheelchair at {zoo_name} is {deposit_text}."
            ),
            sources=zoo.wheelchair.urls,
            additional_instruction=(
                "Verify that the page clearly lists a refundable deposit amount for manual wheelchair rental. "
                "Allow '£' or 'GBP' prefixes/suffixes and minor formatting variants."
            ),
            critical=True,
        )
    else:
        evaluator.add_custom_node(
            result=False,
            id="wheelchair_deposit_amount",
            desc="Provide the refundable deposit amount for manual wheelchair rental",
            parent=wc_node,
            critical=True,
        )

    # 1.c Reference presence
    add_reference_presence_check(
        evaluator,
        node_id="wheelchair_rental_reference",
        desc="URL reference for wheelchair rental information",
        parent=wc_node,
        sources=zoo.wheelchair.urls,
        critical=True,
    )

    # 2) Public transport (bus routes + stop location)
    pt_node = evaluator.add_parallel(
        id="zoo_public_transport",
        desc="Verify public bus access to the zoo",
        parent=zoo_node,
        critical=True,
    )

    # 2.a At least two bus routes that serve the zoo
    if len(zoo.transport.bus_routes) >= 2:
        routes_text = _join_comma(zoo.transport.bus_routes)
        await verify_with_urls_or_fail(
            evaluator,
            node_id="bus_routes_available",
            desc="Identify at least two specific bus route numbers that stop at the zoo",
            parent=pt_node,
            claim=f"The following public bus route numbers serve {zoo_name}: {routes_text}.",
            sources=zoo.transport.urls,
            additional_instruction=(
                "Verify that each listed route number is shown as serving the zoo or its entrance/adjacent stop."
            ),
            critical=True,
        )
    else:
        evaluator.add_custom_node(
            result=False,
            id="bus_routes_available",
            desc="Identify at least two specific bus route numbers that stop at the zoo",
            parent=pt_node,
            critical=True,
        )

    # 2.b Stop location at/outside entrance
    await verify_with_urls_or_fail(
        evaluator,
        node_id="bus_stop_location",
        desc="Confirm bus stops are located outside or at the zoo entrance",
        parent=pt_node,
        claim=f"Buses serving {zoo_name} stop at or directly outside the main entrance.",
        sources=zoo.transport.urls,
        additional_instruction=(
            "Look for phrasing like 'stops outside the zoo', 'at the entrance', or an official bus stop located at the zoo gate."
        ),
        critical=True,
    )

    # 2.c Reference presence
    add_reference_presence_check(
        evaluator,
        node_id="public_transport_reference",
        desc="URL reference for public transportation information",
        parent=pt_node,
        sources=zoo.transport.urls,
        critical=True,
    )

    # 3) Parking availability and daily fee
    pk_node = evaluator.add_parallel(
        id="zoo_parking",
        desc="Verify parking facilities and costs",
        parent=zoo_node,
        critical=True,
    )

    # 3.a Parking availability
    await verify_with_urls_or_fail(
        evaluator,
        node_id="parking_availability",
        desc="Confirm on-site parking is available",
        parent=pk_node,
        claim=f"{zoo_name} provides on-site parking for visitors.",
        sources=zoo.parking.urls,
        additional_instruction="Verify that parking is available on the zoo premises (on-site visitor car park).",
        critical=True,
    )

    # 3.b Daily parking fee in GBP
    if zoo.parking.daily_fee_gbp and zoo.parking.daily_fee_gbp.strip():
        fee_text = zoo.parking.daily_fee_gbp.strip()
        await verify_with_urls_or_fail(
            evaluator,
            node_id="parking_fee_visitors",
            desc="Provide the daily parking fee for regular visitors",
            parent=pk_node,
            claim=f"The daily parking fee for regular visitors at {zoo_name} is {fee_text}.",
            sources=zoo.parking.urls,
            additional_instruction="Verify the standard daily parking price for regular visitors. Allow '£' or 'GBP' formatting.",
            critical=True,
        )
    else:
        evaluator.add_custom_node(
            result=False,
            id="parking_fee_visitors",
            desc="Provide the daily parking fee for regular visitors",
            parent=pk_node,
            critical=True,
        )

    # 3.c Reference presence
    add_reference_presence_check(
        evaluator,
        node_id="parking_reference",
        desc="URL reference for parking information",
        parent=pk_node,
        sources=zoo.parking.urls,
        critical=True,
    )

    # 4) Operating hours in March (open, close, last entry policy)
    oh_node = evaluator.add_parallel(
        id="zoo_operating_hours",
        desc="Verify operating hours for a specific month",
        parent=zoo_node,
        critical=True,
    )

    # 4.a March opening time
    if zoo.hours.march_opening_time and zoo.hours.march_opening_time.strip():
        open_time = zoo.hours.march_opening_time.strip()
        await verify_with_urls_or_fail(
            evaluator,
            node_id="march_opening_time",
            desc="Provide the opening time for March",
            parent=oh_node,
            claim=f"In March, {zoo_name} opens at {open_time}.",
            sources=zoo.hours.urls,
            additional_instruction="Verify the March opening time. Accept 24-hour or am/pm format and minor variants.",
            critical=True,
        )
    else:
        evaluator.add_custom_node(
            result=False,
            id="march_opening_time",
            desc="Provide the opening time for March",
            parent=oh_node,
            critical=True,
        )

    # 4.b March closing time
    if zoo.hours.march_closing_time and zoo.hours.march_closing_time.strip():
        close_time = zoo.hours.march_closing_time.strip()
        await verify_with_urls_or_fail(
            evaluator,
            node_id="march_closing_time",
            desc="Provide the closing time for March",
            parent=oh_node,
            claim=f"In March, {zoo_name} closes at {close_time}.",
            sources=zoo.hours.urls,
            additional_instruction="Verify the March closing time. Accept 24-hour or am/pm format and minor variants.",
            critical=True,
        )
    else:
        evaluator.add_custom_node(
            result=False,
            id="march_closing_time",
            desc="Provide the closing time for March",
            parent=oh_node,
            critical=True,
        )

    # 4.c Last entry policy relative to closing
    if zoo.hours.last_entry_policy and zoo.hours.last_entry_policy.strip():
        lep = zoo.hours.last_entry_policy.strip()
        await verify_with_urls_or_fail(
            evaluator,
            node_id="last_entry_policy",
            desc="Confirm the last entry time policy relative to closing time",
            parent=oh_node,
            claim=f"The last entry policy for {zoo_name} in March is: {lep}.",
            sources=zoo.hours.urls,
            additional_instruction=(
                "Verify that the page describes last admission relative to closing time "
                "(e.g., 'last entry 1 hour before close')."
            ),
            critical=True,
        )
    else:
        evaluator.add_custom_node(
            result=False,
            id="last_entry_policy",
            desc="Confirm the last entry time policy relative to closing time",
            parent=oh_node,
            critical=True,
        )

    # 4.d Reference presence
    add_reference_presence_check(
        evaluator,
        node_id="operating_hours_reference",
        desc="URL reference for operating hours information",
        parent=oh_node,
        sources=zoo.hours.urls,
        critical=True,
    )

    # 5) Closure days: Christmas Day
    cl_node = evaluator.add_parallel(
        id="zoo_closure_days",
        desc="Identify annual closure days",
        parent=zoo_node,
        critical=True,
    )

    await verify_with_urls_or_fail(
        evaluator,
        node_id="christmas_closure",
        desc="Confirm whether the zoo is closed on December 25 (Christmas Day)",
        parent=cl_node,
        claim=f"{zoo_name} is closed on December 25 (Christmas Day).",
        sources=zoo.closure.urls,
        additional_instruction=(
            "Verify explicit mention of Christmas Day (25 December). If the source states 'open 364 days' or "
            "'closed Christmas Day', interpret accordingly."
        ),
        critical=True,
    )

    add_reference_presence_check(
        evaluator,
        node_id="closure_reference",
        desc="URL reference for closure information",
        parent=cl_node,
        sources=zoo.closure.urls,
        critical=True,
    )


async def verify_german_airport(evaluator: Evaluator, parent, airport: AirportVenue) -> None:
    """
    Build and verify the subtree for the German international airport with all required checks.
    """
    ap_node = evaluator.add_parallel(
        id="venue_2_airport",
        desc="Identify a European international airport with specific passenger services",
        parent=parent,
        critical=False,
    )
    ap_name = airport.name or "the airport"

    # 1) Public transport (S-Bahn to city center: lines + frequency)
    sb_node = evaluator.add_parallel(
        id="airport_public_transport",
        desc="Verify S-Bahn/train service to city center",
        parent=ap_node,
        critical=True,
    )

    # 1.a At least two S-Bahn lines
    if len(airport.public_transport.sbahn_lines) >= 2:
        lines_text = _join_comma(airport.public_transport.sbahn_lines)
        await verify_with_urls_or_fail(
            evaluator,
            node_id="sbahn_lines_available",
            desc="Identify at least two S-Bahn line numbers serving the airport",
            parent=sb_node,
            claim=f"S-Bahn lines {lines_text} serve {ap_name} and connect to the city center.",
            sources=airport.public_transport.urls,
            additional_instruction=(
                "Verify that each listed S-Bahn line directly serves the airport and runs to the central city area "
                "(allow common city-center stations)."
            ),
            critical=True,
        )
    else:
        evaluator.add_custom_node(
            result=False,
            id="sbahn_lines_available",
            desc="Identify at least two S-Bahn line numbers serving the airport",
            parent=sb_node,
            critical=True,
        )

    # 1.b Frequency (minutes)
    if airport.public_transport.frequency_minutes and airport.public_transport.frequency_minutes.strip():
        freq_text = airport.public_transport.frequency_minutes.strip()
        await verify_with_urls_or_fail(
            evaluator,
            node_id="sbahn_frequency",
            desc="Provide the service frequency interval (in minutes) for S-Bahn trains",
            parent=sb_node,
            claim=f"The S-Bahn service between {ap_name} and the city center runs every {freq_text} minutes.",
            sources=airport.public_transport.urls,
            additional_instruction="Verify the typical headway/interval in minutes; allow ranges (e.g., '10–20').",
            critical=True,
        )
    else:
        evaluator.add_custom_node(
            result=False,
            id="sbahn_frequency",
            desc="Provide the service frequency interval (in minutes) for S-Bahn trains",
            parent=sb_node,
            critical=True,
        )

    # 1.c Reference presence
    add_reference_presence_check(
        evaluator,
        node_id="sbahn_reference",
        desc="URL reference for S-Bahn service information",
        parent=sb_node,
        sources=airport.public_transport.urls,
        critical=True,
    )

    # 2) Inter-terminal shuttle (availability + frequencies)
    sh_node = evaluator.add_parallel(
        id="airport_terminal_shuttle",
        desc="Verify inter-terminal shuttle service",
        parent=ap_node,
        critical=True,
    )

    # 2.a Shuttle availability
    await verify_with_urls_or_fail(
        evaluator,
        node_id="shuttle_availability",
        desc="Confirm shuttle bus service exists between terminals",
        parent=sh_node,
        claim=f"There is a shuttle bus service that connects terminals at {ap_name}.",
        sources=airport.shuttle.urls,
        additional_instruction="Verify an inter-terminal shuttle bus exists (any branded/numbered service is acceptable).",
        critical=True,
    )

    # 2.b Peak frequency (07:00–17:00)
    if airport.shuttle.peak_frequency and airport.shuttle.peak_frequency.strip():
        peak_text = airport.shuttle.peak_frequency.strip()
        await verify_with_urls_or_fail(
            evaluator,
            node_id="shuttle_peak_frequency",
            desc="Provide the shuttle frequency during peak hours (07:00-17:00)",
            parent=sh_node,
            claim=f"Between 07:00 and 17:00, the inter-terminal shuttle at {ap_name} runs every {peak_text}.",
            sources=airport.shuttle.urls,
            additional_instruction="Verify stated peak frequency; allow ranges (e.g., '5–10 minutes').",
            critical=True,
        )
    else:
        evaluator.add_custom_node(
            result=False,
            id="shuttle_peak_frequency",
            desc="Provide the shuttle frequency during peak hours (07:00-17:00)",
            parent=sh_node,
            critical=True,
        )

    # 2.c Off-peak frequency
    if airport.shuttle.offpeak_frequency and airport.shuttle.offpeak_frequency.strip():
        off_text = airport.shuttle.offpeak_frequency.strip()
        await verify_with_urls_or_fail(
            evaluator,
            node_id="shuttle_offpeak_frequency",
            desc="Provide the shuttle frequency outside peak hours",
            parent=sh_node,
            claim=f"Outside 07:00–17:00, the inter-terminal shuttle at {ap_name} runs every {off_text}.",
            sources=airport.shuttle.urls,
            additional_instruction="Verify stated off-peak frequency; allow ranges.",
            critical=True,
        )
    else:
        evaluator.add_custom_node(
            result=False,
            id="shuttle_offpeak_frequency",
            desc="Provide the shuttle frequency outside peak hours",
            parent=sh_node,
            critical=True,
        )

    # 2.d Reference presence
    add_reference_presence_check(
        evaluator,
        node_id="shuttle_reference",
        desc="URL reference for shuttle service information",
        parent=sh_node,
        sources=airport.shuttle.urls,
        critical=True,
    )

    # 3) Terminal 2 parking designations (at least two codes)
    prk_node = evaluator.add_parallel(
        id="airport_parking",
        desc="Verify parking facilities for specific terminal",
        parent=ap_node,
        critical=True,
    )

    if len(airport.parking.terminal2_parking_codes) >= 2:
        codes_text = _join_comma(airport.parking.terminal2_parking_codes)
        await verify_with_urls_or_fail(
            evaluator,
            node_id="terminal2_parking_designation",
            desc="Identify at least two parking garage codes designated for Terminal 2",
            parent=prk_node,
            claim=f"Terminal 2 parking garages at {ap_name} include: {codes_text}.",
            sources=airport.parking.urls,
            additional_instruction="Verify that the listed codes are designated for Terminal 2 (e.g., P20, P26).",
            critical=True,
        )
    else:
        evaluator.add_custom_node(
            result=False,
            id="terminal2_parking_designation",
            desc="Identify at least two parking garage codes designated for Terminal 2",
            parent=prk_node,
            critical=True,
        )

    add_reference_presence_check(
        evaluator,
        node_id="parking_designation_reference",
        desc="URL reference for parking designation information",
        parent=prk_node,
        sources=airport.parking.urls,
        critical=True,
    )

    # 4) Terminal 1 lounge: existence + name + hours
    lg_node = evaluator.add_parallel(
        id="airport_lounge",
        desc="Verify passenger lounge facilities in Terminal 1",
        parent=ap_node,
        critical=True,
    )

    # 4.a Lounge existence (at least one)
    await verify_with_urls_or_fail(
        evaluator,
        node_id="lounge_existence",
        desc="Confirm at least one passenger lounge exists in Terminal 1",
        parent=lg_node,
        claim=f"Terminal 1 at {ap_name} has at least one passenger lounge.",
        sources=airport.lounge.urls,
        additional_instruction="Verify that Terminal 1 lists at least one lounge available to passengers.",
        critical=True,
    )

    # 4.b Lounge name
    if airport.lounge.t1_lounge_name and airport.lounge.t1_lounge_name.strip():
        lname = airport.lounge.t1_lounge_name.strip()
        await verify_with_urls_or_fail(
            evaluator,
            node_id="lounge_name",
            desc="Provide the name of at least one lounge in Terminal 1",
            parent=lg_node,
            claim=f"One passenger lounge in Terminal 1 at {ap_name} is '{lname}'.",
            sources=airport.lounge.urls,
            additional_instruction="Verify the lounge name is associated with Terminal 1.",
            critical=True,
        )
    else:
        evaluator.add_custom_node(
            result=False,
            id="lounge_name",
            desc="Provide the name of at least one lounge in Terminal 1",
            parent=lg_node,
            critical=True,
        )

    # 4.c Lounge operating hours
    if airport.lounge.t1_lounge_hours and airport.lounge.t1_lounge_hours.strip():
        lhrs = airport.lounge.t1_lounge_hours.strip()
        await verify_with_urls_or_fail(
            evaluator,
            node_id="lounge_opening_hours",
            desc="Provide the operating hours for the identified lounge",
            parent=lg_node,
            claim=f"The operating hours for this Terminal 1 lounge at {ap_name} are: {lhrs}.",
            sources=airport.lounge.urls,
            additional_instruction="Verify the hours for the named lounge; accept time ranges and weekday variations if stated.",
            critical=True,
        )
    else:
        evaluator.add_custom_node(
            result=False,
            id="lounge_opening_hours",
            desc="Provide the operating hours for the identified lounge",
            parent=lg_node,
            critical=True,
        )

    add_reference_presence_check(
        evaluator,
        node_id="lounge_reference",
        desc="URL reference for lounge information",
        parent=lg_node,
        sources=airport.lounge.urls,
        critical=True,
    )

    # 5) Public transport fare zone
    zn_node = evaluator.add_parallel(
        id="airport_zone_info",
        desc="Verify public transport fare zone designation",
        parent=ap_node,
        critical=True,
    )

    if airport.zone.zone_number and airport.zone.zone_number.strip():
        zone_text = airport.zone.zone_number.strip()
        await verify_with_urls_or_fail(
            evaluator,
            node_id="zone_number",
            desc="Provide the public transport zone number for the airport",
            parent=zn_node,
            claim=f"{ap_name} is located in public transport fare zone {zone_text}.",
            sources=airport.zone.urls,
            additional_instruction="Verify the official fare zone designation (number or code) covering the airport.",
            critical=True,
        )
    else:
        evaluator.add_custom_node(
            result=False,
            id="zone_number",
            desc="Provide the public transport zone number for the airport",
            parent=zn_node,
            critical=True,
        )

    add_reference_presence_check(
        evaluator,
        node_id="zone_reference",
        desc="URL reference for zone information",
        parent=zn_node,
        sources=airport.zone.urls,
        critical=True,
    )


# --------------------------------------------------------------------------- #
# Main evaluation entry                                                       #
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
    Evaluate an answer for the 'eu_accessible_venues' task and return a structured summary.
    """
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,     # Root aggregates two independent venues
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

    # IMPORTANT: Keep root non-critical to avoid forcing all descendants to be critical
    root.critical = False

    # 1) Extract structured info from the answer
    extraction = await evaluator.extract(
        prompt=prompt_extract_venues(),
        template_class=VenuesExtraction,
        extraction_name="venues_extraction",
    )

    # 2) Build verification trees for each venue and run checks
    await verify_uk_zoo(evaluator, root, extraction.zoo)
    await verify_german_airport(evaluator, root, extraction.airport)

    # 3) Return the aggregated summary
    return evaluator.get_summary()