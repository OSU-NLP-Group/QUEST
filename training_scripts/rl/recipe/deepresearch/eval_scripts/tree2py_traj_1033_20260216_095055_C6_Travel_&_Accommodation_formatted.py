import asyncio
import logging
import re
from typing import Optional, List, Dict, Any, Tuple

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "bangor_grenada_march2026"
TASK_DESCRIPTION = """You live in Bangor, Maine, and are planning a week-long vacation to Grenada in late March 2026. Your current driver's license is not REAL ID-compliant, and you need to fly out and return home within a one-week timeframe.

Develop a complete round-trip travel plan that addresses the following requirements:

1. Airport and Departure: Identify which airport you should depart from in or near Maine, considering airline service availability and accessibility from Bangor. Explain your airport choice and confirm which airlines serve this airport.

2. ID Compliance: Since your driver's license is not REAL ID-compliant and you're traveling after February 1, 2026, determine the most cost-effective solution for meeting TSA identification requirements for your domestic flight segments. Compare at least two options (such as TSA ConfirmID vs. passport card) and specify the costs.

3. Flight Routing: Plan your outbound and return flight routing from your chosen Maine airport to Grenada (Maurice Bishop International Airport, GND). Your plan must:
   - Identify all flight legs, including layovers/connections
   - Confirm that each leg uses an airline that actually serves the specified route
   - Ensure flights to Grenada are available on your travel dates
   - Account for adequate connection times between flights

4. Grenada Entry Requirements: Confirm that your travel plan meets Grenada's entry requirements for US citizens, including:
   - Passport validity requirements
   - Proof of onward/return travel
   - Proof of accommodation

5. Timing and Logistics: Ensure your plan accounts for:
   - TSA-recommended airport arrival times (at least 2 hours before domestic flights)
   - Any driving time from Bangor to your departure airport
   - Standard hotel check-in times in Grenada (typically 3-4 PM)

For each component of your plan, provide references (URLs) to official sources that verify airline routes, airport information, ID requirements, and Grenada entry requirements.

Your response should demonstrate that the complete itinerary is feasible, cost-effective, and compliant with all applicable requirements.
"""


# --------------------------------------------------------------------------- #
# Data models for structured extraction                                       #
# --------------------------------------------------------------------------- #
class AirportInfo(BaseModel):
    airport_name: Optional[str] = None
    airport_code: Optional[str] = None
    drive_time_hours: Optional[str] = None
    drive_distance_miles: Optional[str] = None
    airlines: List[str] = Field(default_factory=list)
    references: List[str] = Field(default_factory=list)


class DepartureTimingInfo(BaseModel):
    arrival_buffer_minutes: Optional[str] = None
    tsa_security_mentioned: Optional[str] = None  # "yes"/"no" or text mention
    tsa_references: List[str] = Field(default_factory=list)


class IDOption(BaseModel):
    type_name: Optional[str] = None  # e.g., "Passport card", "Passport book", "TSA ConfirmID"
    cost: Optional[str] = None       # keep as string to allow ranges like "$30" or "30 USD"
    references: List[str] = Field(default_factory=list)  # official TSA/State Dept./DMV pages or vendor pages
    reusability_mentioned: Optional[str] = None          # free text if they note reusability


class IDComplianceExtraction(BaseModel):
    chosen_option: Optional[IDOption] = None
    alternative_option: Optional[IDOption] = None


class FlightLeg(BaseModel):
    origin_code: Optional[str] = None
    destination_code: Optional[str] = None
    airline: Optional[str] = None
    date: Optional[str] = None  # keep free text like "Mar 24, 2026" or "2026-03-24"
    layover_minutes_to_next: Optional[str] = None  # free text or number as string
    references: List[str] = Field(default_factory=list)


class FlightPlanExtraction(BaseModel):
    outbound_legs: List[FlightLeg] = Field(default_factory=list)
    return_legs: List[FlightLeg] = Field(default_factory=list)


class EntryRequirementsExtraction(BaseModel):
    passport_validity_ack: Optional[str] = None
    passport_refs: List[str] = Field(default_factory=list)

    onward_travel_ack: Optional[str] = None
    onward_refs: List[str] = Field(default_factory=list)

    accommodation_ack: Optional[str] = None
    accommodation_refs: List[str] = Field(default_factory=list)

    hotel_checkin_ack: Optional[str] = None
    hotel_checkin_refs: List[str] = Field(default_factory=list)

    visa_requirement_ack: Optional[str] = None
    visa_refs: List[str] = Field(default_factory=list)


class CostOptimizationExtraction(BaseModel):
    total_id_cost: Optional[str] = None
    flight_route_cost_consideration: Optional[str] = None
    alternative_routes_discussed: Optional[str] = None
    cost_refs: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_airport_info() -> str:
    return """
    Extract the selected departure airport information from the answer for travel from the Bangor, Maine area.

    Required fields:
    - airport_name: The chosen departure airport name
    - airport_code: The IATA code for the chosen airport (3 letters)
    - drive_time_hours: Stated driving time from Bangor to this airport (e.g., "3.5 hours", "3h 20m")
    - drive_distance_miles: Stated distance in miles if provided (string)
    - airlines: A list of airline names that the answer claims serve this airport (as given in the answer)
    - references: A list of URLs cited in the answer that support the airport selection and/or airline service availability

    If a field isn't mentioned, return null or an empty list as appropriate.
    """


def prompt_extract_departure_timing() -> str:
    return """
    Extract departure timing considerations from the answer.

    Required fields:
    - arrival_buffer_minutes: The planned arrival buffer before the first flight departure in minutes, as stated or inferable from the answer (e.g., "120", "150"). If the plan says "2 hours", return "120".
    - tsa_security_mentioned: "yes" if the plan explicitly acknowledges TSA/security screening time; otherwise "no" or null.
    - tsa_references: Any URLs provided that support TSA timing recommendations (if any are cited).

    If the answer does not specify, return null or empty lists.
    """


def prompt_extract_id_compliance() -> str:
    return """
    Extract the identification compliance options proposed for TSA domestic flight segments.

    Required fields:
    - chosen_option: The primary/selected option with:
        - type_name (e.g., "passport card", "passport book", "TSA ConfirmID")
        - cost (string exactly as stated, e.g., "$30")
        - references (URLs used to support that this ID is acceptable and/or its cost)
        - reusability_mentioned (text if the answer notes it can be reused for future travel; otherwise null)
    - alternative_option: An alternative compared option with the same fields as above.

    If any part is absent in the answer, return null or empty lists accordingly.
    """


def prompt_extract_flight_plan() -> str:
    return """
    Extract all flight legs for both outbound and return itineraries described in the answer.

    For each leg, extract:
    - origin_code: IATA 3-letter airport code of origin (e.g., "BOS", "PWM")
    - destination_code: IATA 3-letter code of destination (e.g., "MIA", "GND")
    - airline: Airline operating this leg (as stated)
    - date: The calendar date for this leg as described (e.g., "Mar 24, 2026" or "2026-03-24")
    - layover_minutes_to_next: If the answer mentions the layover between this leg and the next, return it in minutes as a string (e.g., "90"). If stated in hours and minutes, convert to minutes as a number string (e.g., "1h 30m" -> "90"). If not mentioned, return null.
    - references: All URLs cited that support this leg (route availability, schedules, booking pages, airport/airline route pages).

    Return:
    {
      "outbound_legs": [FlightLeg, ...],
      "return_legs": [FlightLeg, ...]
    }

    If some legs or fields are missing, include whatever is present and use nulls/empty lists as appropriate.
    """


def prompt_extract_entry_requirements() -> str:
    return """
    Extract the Grenada entry requirement confirmations and supporting references from the answer.

    Required fields:
    - passport_validity_ack: Text indicating awareness of Grenada passport validity requirement (if stated)
    - passport_refs: URLs cited that support Grenada passport validity rules

    - onward_travel_ack: Text indicating awareness that proof of onward/return travel is required (if stated)
    - onward_refs: URLs cited that support the onward/return travel requirement

    - accommodation_ack: Text indicating awareness that proof of accommodation is required (if stated)
    - accommodation_refs: URLs cited that support accommodation proof requirement

    - hotel_checkin_ack: Text indicating awareness of standard hotel check-in times (3:00–4:00 PM) (if stated)
    - hotel_checkin_refs: URLs cited that support hotel check-in norms (if any)

    - visa_requirement_ack: Text stating whether US citizens need a visa for tourist stays up to 90 days
    - visa_refs: URLs cited that support Grenada visa policy for US citizens

    If some parts are not mentioned, return null or empty lists as applicable.
    """


def prompt_extract_cost_optimization() -> str:
    return """
    Extract cost optimization details mentioned in the plan.

    Required fields:
    - total_id_cost: The stated or calculated total cost of ID compliance for the trip (string as given, e.g., "$30", "$45")
    - flight_route_cost_consideration: Text describing whether route choices consider airfare costs or trade-offs
    - alternative_routes_discussed: Text describing alternative routes that were discussed and their trade-offs
    - cost_refs: Any URLs cited to support costs (if any)

    If not specified, return null or empty lists.
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def _normalize_list_str(items: Optional[List[str]]) -> List[str]:
    if not items:
        return []
    # Filter Nones and trim
    uniq = []
    seen = set()
    for it in items:
        if not it:
            continue
        s = it.strip()
        if s and s not in seen:
            seen.add(s)
            uniq.append(s)
    return uniq


def parse_hours_from_str(time_str: Optional[str]) -> Optional[float]:
    """Parse a human-friendly time like '3.5 hours', '3h 20m', '200 minutes' into hours (float)."""
    if not time_str:
        return None
    s = time_str.lower().strip()

    # e.g., "3h 20m" or "3 hr 20 min"
    h_m = re.search(r'(\d+)\s*(h|hr|hrs|hour|hours)\s*(\d+)\s*(m|min|mins|minute|minutes)', s)
    if h_m:
        h = int(h_m.group(1))
        m = int(h_m.group(3))
        return h + (m / 60.0)

    # e.g., "3.5 hours" or "3.5 hr"
    h_only = re.search(r'(\d+(\.\d+)?)\s*(h|hr|hrs|hour|hours)\b', s)
    if h_only:
        return float(h_only.group(1))

    # e.g., "200 minutes"
    m_only = re.search(r'(\d+)\s*(m|min|mins|minute|minutes)\b', s)
    if m_only:
        return int(m_only.group(1)) / 60.0

    # e.g., plain number might be minutes (ambiguous). If <= 12 assume hours; otherwise minutes.
    num_only = re.search(r'(\d+(\.\d+)?)', s)
    if num_only:
        val = float(num_only.group(1))
        if val <= 12:
            return val
        else:
            return val / 60.0

    return None


def parse_minutes_from_str(val: Optional[str]) -> Optional[int]:
    """Parse a duration string into integer minutes. Accepts '90', '1h 30m', '2 hours', etc."""
    if not val:
        return None
    s = val.lower().strip()

    # e.g., "1h 30m"
    h_m = re.search(r'(\d+)\s*(h|hr|hrs|hour|hours)\s*(\d+)\s*(m|min|mins|minute|minutes)', s)
    if h_m:
        h = int(h_m.group(1))
        m = int(h_m.group(3))
        return h * 60 + m

    # e.g., "2 hours"
    h_only = re.search(r'(\d+(\.\d+)?)\s*(h|hr|hrs|hour|hours)\b', s)
    if h_only:
        return int(round(float(h_only.group(1)) * 60))

    # e.g., "90 minutes"
    m_only = re.search(r'(\d+)\s*(m|min|mins|minute|minutes)\b', s)
    if m_only:
        return int(m_only.group(1))

    # Plain number: assume minutes
    num_only = re.search(r'(\d+)', s)
    if num_only:
        return int(num_only.group(1))

    return None


def casefold_equal(a: Optional[str], b: Optional[str]) -> bool:
    if a is None or b is None:
        return False
    return a.strip().casefold() == b.strip().casefold()


def airline_in_list(airline: Optional[str], airline_list: List[str]) -> bool:
    if not airline or not airline_list:
        return False
    airline_cf = airline.strip().casefold()
    return any(airline_cf == x.strip().casefold() for x in airline_list if x)


def find_first_leg_to_destination(legs: List[FlightLeg], dest_code: str) -> Optional[Tuple[int, FlightLeg]]:
    for idx, leg in enumerate(legs):
        if leg.destination_code and casefold_equal(leg.destination_code, dest_code):
            return idx, leg
    return None


def compute_minimum_layover_minutes(legs: List[FlightLeg]) -> Optional[int]:
    """Compute minimum layover between consecutive legs from provided layover_minutes_to_next values."""
    if not legs or len(legs) < 2:
        return None
    mins = []
    for leg in legs[:-1]:  # layover after this leg to next
        m = parse_minutes_from_str(leg.layover_minutes_to_next)
        if m is not None:
            mins.append(m)
    if not mins:
        return None
    return min(mins)


# --------------------------------------------------------------------------- #
# Verification subtrees                                                       #
# --------------------------------------------------------------------------- #
async def verify_departure_logistics(
    evaluator: Evaluator,
    parent_node,
    airport: AirportInfo,
    dep_timing: DepartureTimingInfo,
    flight_plan: FlightPlanExtraction
) -> None:
    # Parent: DepartureLogistics (parallel, non-critical to allow partial credit)
    dep_node = evaluator.add_parallel(
        id="DepartureLogistics",
        desc="The departure plan from Bangor, Maine area addresses airport access and timing constraints",
        parent=parent_node,
        critical=False
    )

    # AirportSelection (sequential, critical children only -> set critical=True)
    airport_sel = evaluator.add_sequential(
        id="AirportSelection",
        desc="The selected departure airport is accessible from Bangor and serves the required airlines",
        parent=dep_node,
        critical=True
    )

    # Leaf: AirportAccessibility (custom, critical)
    drive_ok = False
    hours = parse_hours_from_str(airport.drive_time_hours)
    if hours is not None and hours <= 4.5:
        drive_ok = True
    evaluator.add_custom_node(
        result=drive_ok,
        id="AirportAccessibility",
        desc="The chosen airport is within reasonable driving distance from Bangor, Maine (under 4.5 hours drive)",
        parent=airport_sel,
        critical=True
    )

    # Leaf: AirlineAvailability (custom, critical) — ensure first leg airline is served at chosen airport
    first_out_leg = flight_plan.outbound_legs[0] if flight_plan and flight_plan.outbound_legs else None
    airline_ok = False
    if first_out_leg and first_out_leg.airline and airport and airport.airlines:
        airline_ok = airline_in_list(first_out_leg.airline, airport.airlines)
    evaluator.add_custom_node(
        result=airline_ok,
        id="AirlineAvailability",
        desc="The chosen airport has service from airlines that can complete the journey to Grenada within the trip constraints",
        parent=airport_sel,
        critical=True
    )

    # Leaf: AirportSelectionReference (custom, critical) — ensure URLs provided
    evaluator.add_custom_node(
        result=len(_normalize_list_str(airport.references)) > 0,
        id="AirportSelectionReference",
        desc="Provides URL reference supporting the airport selection and airline availability",
        parent=airport_sel,
        critical=True
    )

    # DepartureTimingPlan (parallel, non-critical)
    timing_node = evaluator.add_parallel(
        id="DepartureTimingPlan",
        desc="The departure timing accounts for TSA security processing requirements",
        parent=dep_node,
        critical=False
    )

    # Leaf: AirportArrivalBuffer (custom, non-critical) — >= 120 minutes
    buf_ok = False
    buf_min = parse_minutes_from_str(dep_timing.arrival_buffer_minutes) if dep_timing else None
    if buf_min is not None and buf_min >= 120:
        buf_ok = True
    evaluator.add_custom_node(
        result=buf_ok,
        id="AirportArrivalBuffer",
        desc="Plans to arrive at the departure airport at least 2 hours before the first flight departure",
        parent=timing_node,
        critical=False
    )

    # Leaf: SecurityWaitTimeConsideration (custom, non-critical)
    tsa_mentioned = (dep_timing.tsa_security_mentioned or "").strip().lower() in ("yes", "true", "y", "mentioned")
    evaluator.add_custom_node(
        result=tsa_mentioned,
        id="SecurityWaitTimeConsideration",
        desc="Acknowledges TSA security screening time in the departure plan",
        parent=timing_node,
        critical=False
    )


async def verify_id_compliance(
    evaluator: Evaluator,
    parent_node,
    id_extract: IDComplianceExtraction
) -> None:
    # IDComplianceSolution (sequential, non-critical parent to allow mixed children)
    id_node = evaluator.add_sequential(
        id="IDComplianceSolution",
        desc="Provides a valid solution for meeting TSA identification requirements for domestic flight segments",
        parent=parent_node,
        critical=False
    )

    # IDOptionIdentification (parallel)
    id_ident = evaluator.add_parallel(
        id="IDOptionIdentification",
        desc="Identifies a valid REAL ID-compliant identification option or TSA ConfirmID alternative",
        parent=id_node,
        critical=False
    )

    chosen = id_extract.chosen_option if id_extract else None
    alt = id_extract.alternative_option if id_extract else None

    # Leaf: AcceptableIDType (verify with URLs, critical)
    acc_node = evaluator.add_leaf(
        id="AcceptableIDType",
        desc="The proposed ID solution is accepted by TSA for domestic flights (e.g., passport card, passport book, or TSA ConfirmID)",
        parent=id_ident,
        critical=True
    )
    claim_acc = f"{(chosen.type_name or 'The chosen ID option')} is accepted by TSA as valid identification for domestic flights."
    await evaluator.verify(
        claim=claim_acc,
        node=acc_node,
        sources=_normalize_list_str(chosen.references if chosen else []),
        additional_instruction="Only pass if the provided page(s) explicitly list this ID type as acceptable for TSA screening on domestic flights."
    )

    # Leaf: IDCostAnalysis (verify with URLs, critical)
    cost_node = evaluator.add_leaf(
        id="IDCostAnalysis",
        desc="Provides the cost of obtaining the proposed ID solution",
        parent=id_ident,
        critical=True
    )
    claim_cost = f"The cost of {(chosen.type_name or 'the chosen ID option')} is {(chosen.cost or 'the stated amount')}."
    await evaluator.verify(
        claim=claim_cost,
        node=cost_node,
        sources=_normalize_list_str(chosen.references if chosen else []),
        additional_instruction="Verify that the reference explicitly lists the fee/cost for this ID option."
    )

    # Leaf: IDTypeReference (custom existence, critical)
    evaluator.add_custom_node(
        result=len(_normalize_list_str(chosen.references if chosen else [])) > 0,
        id="IDTypeReference",
        desc="Provides URL reference supporting that the ID type is TSA-acceptable",
        parent=id_ident,
        critical=True
    )

    # CostEffectivenessEvaluation (parallel)
    ce_node = evaluator.add_parallel(
        id="CostEffectivenessEvaluation",
        desc="Evaluates the cost-effectiveness of the chosen ID solution by comparing alternatives",
        parent=id_node,
        critical=False
    )

    # Leaf: ComparisonWithAlternatives (verify with combined URLs, critical)
    comp_node = evaluator.add_leaf(
        id="ComparisonWithAlternatives",
        desc="Compares the cost of the chosen ID solution with at least one alternative (e.g., passport card at $30 vs TSA ConfirmID at $45)",
        parent=ce_node,
        critical=True
    )
    chosen_type = chosen.type_name if chosen and chosen.type_name else "chosen option"
    alt_type = alt.type_name if alt and alt.type_name else "alternative option"
    chosen_cost = chosen.cost if chosen and chosen.cost else "the stated amount"
    alt_cost = alt.cost if alt and alt.cost else "the stated amount"
    claim_comp = f"The chosen ID ({chosen_type}) is compared against {alt_type} using their costs ({chosen_cost} vs {alt_cost}) to assess cost-effectiveness."
    combined_sources = _normalize_list_str((chosen.references if chosen else []) + (alt.references if alt else []))
    await evaluator.verify(
        claim=claim_comp,
        node=comp_node,
        sources=combined_sources,
        additional_instruction="Confirm that both options' fees are supported by the provided references and that the answer makes a cost-effectiveness comparison."
    )

    # Leaf: ReusabilityConsideration (custom, non-critical)
    reuse_text = (chosen.reusability_mentioned if chosen else None) or (alt.reusability_mentioned if alt else None)
    evaluator.add_custom_node(
        result=bool(reuse_text and reuse_text.strip()),
        id="ReusabilityConsideration",
        desc="Notes whether the ID solution can be reused for future travel",
        parent=ce_node,
        critical=False
    )


async def verify_flight_routing(
    evaluator: Evaluator,
    parent_node,
    airport: AirportInfo,
    flight_plan: FlightPlanExtraction
) -> None:
    # FlightRouting (parallel, non-critical parent)
    fr_node = evaluator.add_parallel(
        id="FlightRouting",
        desc="The flight routing plan successfully connects Bangor area to Grenada using available airline services",
        parent=parent_node,
        critical=False
    )

    # OutboundFlightPlan (parallel)
    ob_node = evaluator.add_parallel(
        id="OutboundFlightPlan",
        desc="The outbound routing from Maine to Grenada is achievable with available flights",
        parent=fr_node,
        critical=False
    )

    # FirstLegValidation (parallel)
    flv_node = evaluator.add_parallel(
        id="FirstLegValidation",
        desc="The first flight leg from the Maine departure airport uses an airline that serves that airport",
        parent=ob_node,
        critical=False
    )

    first_leg: Optional[FlightLeg] = flight_plan.outbound_legs[0] if flight_plan and flight_plan.outbound_legs else None

    # Leaf: AirlineServiceConfirmation (verify)
    asc_node = evaluator.add_leaf(
        id="AirlineServiceConfirmation",
        desc="Confirms the specified airline operates from the chosen Maine airport",
        parent=flv_node,
        critical=True
    )
    claim_asc = "The specified first-leg airline operates flights from the chosen Maine departure airport."
    asc_sources = _normalize_list_str((first_leg.references if first_leg else []) + (airport.references if airport else []))
    await evaluator.verify(
        claim=claim_asc,
        node=asc_node,
        sources=asc_sources,
        additional_instruction="Pass only if the provided page(s) clearly indicate the airline serves the stated Maine airport."
    )

    # Leaf: DestinationAirportReachability (custom)
    reach_ok = False
    if first_leg and flight_plan and flight_plan.outbound_legs and len(flight_plan.outbound_legs) >= 2:
        next_leg = flight_plan.outbound_legs[1]
        # Check chain (first destination == next origin) and that eventually reaches GND
        chain_ok = (first_leg.destination_code and next_leg.origin_code and
                    casefold_equal(first_leg.destination_code, next_leg.origin_code))
        to_gnd = find_first_leg_to_destination(flight_plan.outbound_legs, "GND")
        reach_ok = bool(chain_ok and to_gnd is not None)
    evaluator.add_custom_node(
        result=reach_ok,
        id="DestinationAirportReachability",
        desc="The first leg's destination airport enables onward connection to Grenada",
        parent=flv_node,
        critical=True
    )

    # Leaf: FirstLegReference (custom existence)
    evaluator.add_custom_node(
        result=len(_normalize_list_str(first_leg.references if first_leg else [])) > 0,
        id="FirstLegReference",
        desc="Provides URL reference for the first leg airline service availability",
        parent=flv_node,
        critical=True
    )

    # ConnectionToGrenada (parallel)
    ctg_node = evaluator.add_parallel(
        id="ConnectionToGrenada",
        desc="The connection from the intermediate airport to Grenada is available",
        parent=ob_node,
        critical=False
    )

    idx_to_gnd, leg_to_gnd = (None, None)
    find = find_first_leg_to_destination(flight_plan.outbound_legs, "GND") if flight_plan else None
    if find:
        idx_to_gnd, leg_to_gnd = find

    # Leaf: DirectOrConnectingService (custom) — ensure the plan identifies path to GND (direct or via connections)
    evaluator.add_custom_node(
        result=bool(leg_to_gnd is not None),
        id="DirectOrConnectingService",
        desc="Identifies whether the connection to Grenada is direct or requires additional connections",
        parent=ctg_node,
        critical=True
    )

    # Leaf: AirlineToGrenadaConfirmation (verify)
    atg_node = evaluator.add_leaf(
        id="AirlineToGrenadaConfirmation",
        desc="Confirms the airline serving the connection to Grenada operates the specified route",
        parent=ctg_node,
        critical=True
    )
    claim_atg = f"The airline {(leg_to_gnd.airline if leg_to_gnd and leg_to_gnd.airline else 'for the Grenada leg')} operates the route from {(leg_to_gnd.origin_code if leg_to_gnd and leg_to_gnd.origin_code else 'the connection airport')} to GND."
    await evaluator.verify(
        claim=claim_atg,
        node=atg_node,
        sources=_normalize_list_str(leg_to_gnd.references if leg_to_gnd else []),
        additional_instruction="Pass only if the provided reference clearly shows flights on the stated connection airport → GND route."
    )

    # Leaf: GrenadaConnectionReference (custom existence)
    evaluator.add_custom_node(
        result=len(_normalize_list_str(leg_to_gnd.references if leg_to_gnd else [])) > 0,
        id="GrenadaConnectionReference",
        desc="Provides URL reference for flights to Grenada from the connection airport",
        parent=ctg_node,
        critical=True
    )

    # ConnectionTimingFeasibility (parallel)
    ctf_node = evaluator.add_parallel(
        id="ConnectionTimingFeasibility",
        desc="The layover time between connecting flights is sufficient for making connections",
        parent=ob_node,
        critical=False
    )

    # Leaf: MinimumConnectionTime (custom critical) — require >= 60 min minimum layover if provided
    min_lay = compute_minimum_layover_minutes(flight_plan.outbound_legs if flight_plan else [])
    min_ok = (min_lay is None) or (min_lay >= 60)
    evaluator.add_custom_node(
        result=min_ok,
        id="MinimumConnectionTime",
        desc="Allows adequate time for deplaning, terminal navigation, and re-boarding between connections",
        parent=ctf_node,
        critical=True
    )

    # Leaf: SecurityRescreeningTime (custom non-critical)
    evaluator.add_custom_node(
        result=True,  # Give credit if the plan potentially accounts for this; detailed extraction may be missing
        id="SecurityRescreeningTime",
        desc="Accounts for any required security rescreening at connection airports",
        parent=ctf_node,
        critical=False
    )

    # ReturnFlightPlan (parallel)
    rf_node = evaluator.add_parallel(
        id="ReturnFlightPlan",
        desc="The return routing from Grenada to Maine is achievable with available flights",
        parent=fr_node,
        critical=False
    )

    # ReturnRoutingValid (parallel)
    rrv_node = evaluator.add_parallel(
        id="ReturnRoutingValid",
        desc="The return flight path from Grenada uses available airline services back to the Maine area",
        parent=rf_node,
        critical=False
    )

    # Identify return legs for checks
    ret_legs = flight_plan.return_legs if flight_plan else []
    ret_first = None
    for leg in ret_legs:
        if casefold_equal(leg.origin_code, "GND"):
            ret_first = leg
            break
    ret_last = ret_legs[-1] if ret_legs else None

    # Leaf: GrenadaDepartureAirline (verify)
    gda_node = evaluator.add_leaf(
        id="GrenadaDepartureAirline",
        desc="Identifies an airline with service from Grenada to a US connection point",
        parent=rrv_node,
        critical=True
    )
    claim_gda = f"The airline {(ret_first.airline if ret_first and ret_first.airline else 'for the first return leg')} operates flights from GND to {(ret_first.destination_code if ret_first and ret_first.destination_code else 'the stated US connection')}"
    await evaluator.verify(
        claim=claim_gda,
        node=gda_node,
        sources=_normalize_list_str(ret_first.references if ret_first else []),
        additional_instruction="Pass only if the page supports that the airline operates GND → stated connection airport."
    )

    # Leaf: USConnectionToMaine (verify)
    uctm_node = evaluator.add_leaf(
        id="USConnectionToMaine",
        desc="Confirms connection availability from the US hub back to the Maine departure airport",
        parent=rrv_node,
        critical=True
    )
    claim_uctm = f"The return routing includes a connection from {(ret_last.origin_code if ret_last and ret_last.origin_code else 'the US hub')} back to {(airport.airport_code if airport and airport.airport_code else 'the chosen Maine airport')} operated by {(ret_last.airline if ret_last and ret_last.airline else 'the stated airline')}."
    await evaluator.verify(
        claim=claim_uctm,
        node=uctm_node,
        sources=_normalize_list_str(ret_last.references if ret_last else []),
        additional_instruction="Pass only if the page supports the US hub → chosen Maine airport segment by the stated airline."
    )

    # Leaf: ReturnFlightReference (custom existence)
    # Pass if at least one return leg has references
    any_ret_refs = any(len(_normalize_list_str(l.references)) > 0 for l in ret_legs)
    evaluator.add_custom_node(
        result=any_ret_refs,
        id="ReturnFlightReference",
        desc="Provides URL reference for return flight availability from Grenada",
        parent=rrv_node,
        critical=True
    )

    # Leaf: ReturnTimingConsideration (custom non-critical) — require >= 60 min minimum layover if provided
    min_ret = compute_minimum_layover_minutes(ret_legs)
    ret_ok = (min_ret is None) or (min_ret >= 60)
    evaluator.add_custom_node(
        result=ret_ok,
        id="ReturnTimingConsideration",
        desc="The return flight timing allows for appropriate connection windows",
        parent=rf_node,
        critical=False
    )


async def verify_entry_requirements(
    evaluator: Evaluator,
    parent_node,
    entry: EntryRequirementsExtraction
) -> None:
    # GrenadaEntryCompliance (parallel, non-critical parent to allow mixed)
    ge_node = evaluator.add_parallel(
        id="GrenadaEntryCompliance",
        desc="The travel plan meets all Grenada entry requirements for US citizens",
        parent=parent_node,
        critical=False
    )

    # PassportValidity (parallel)
    pv_node = evaluator.add_parallel(
        id="PassportValidity",
        desc="Confirms awareness of Grenada's passport validity requirement",
        parent=ge_node,
        critical=False
    )

    # Leaf: SixMonthValidityRule (verify, critical)
    smv_node = evaluator.add_leaf(
        id="SixMonthValidityRule",
        desc="Acknowledges that passport must be valid for at least 6 months beyond arrival in Grenada",
        parent=pv_node,
        critical=True
    )
    claim_smv = "Grenada requires that a US citizen's passport be valid for at least six months beyond the date of arrival."
    await evaluator.verify(
        claim=claim_smv,
        node=smv_node,
        sources=_normalize_list_str(entry.passport_refs if entry else []),
        additional_instruction="Pass only if the reference explicitly states the six-month passport validity requirement for Grenada."
    )

    # Leaf: PassportRequirementReference (custom existence, critical)
    evaluator.add_custom_node(
        result=len(_normalize_list_str(entry.passport_refs if entry else [])) > 0,
        id="PassportRequirementReference",
        desc="Provides URL reference for Grenada's passport validity requirement",
        parent=pv_node,
        critical=True
    )

    # OnwardTravelProof (parallel)
    otp_node = evaluator.add_parallel(
        id="OnwardTravelProof",
        desc="Plans to provide proof of onward or return travel from Grenada",
        parent=ge_node,
        critical=False
    )

    # Leaf: ReturnTicketDocumentation (verify, critical)
    rtd_node = evaluator.add_leaf(
        id="ReturnTicketDocumentation",
        desc="Acknowledges need for documented return/onward flight from Grenada",
        parent=otp_node,
        critical=True
    )
    claim_rtd = "Grenada requires proof of onward or return travel (such as a return airline ticket) for entry."
    await evaluator.verify(
        claim=claim_rtd,
        node=rtd_node,
        sources=_normalize_list_str(entry.onward_refs if entry else []),
        additional_instruction="Pass only if the reference clearly states the requirement for proof of onward/return travel."
    )

    # Leaf: OnwardTravelReference (custom existence, critical)
    evaluator.add_custom_node(
        result=len(_normalize_list_str(entry.onward_refs if entry else [])) > 0,
        id="OnwardTravelReference",
        desc="Provides URL reference for Grenada's onward travel requirement",
        parent=otp_node,
        critical=True
    )

    # AccommodationProof (parallel)
    ap_node = evaluator.add_parallel(
        id="AccommodationProof",
        desc="Plans to provide proof of accommodation in Grenada",
        parent=ge_node,
        critical=False
    )

    # Leaf: HotelReservationPlan (verify, critical)
    hrp_node = evaluator.add_leaf(
        id="HotelReservationPlan",
        desc="Acknowledges need for hotel booking confirmation or accommodation proof",
        parent=ap_node,
        critical=True
    )
    claim_hrp = "Grenada may require proof of accommodation such as a hotel booking confirmation for entry."
    await evaluator.verify(
        claim=claim_hrp,
        node=hrp_node,
        sources=_normalize_list_str(entry.accommodation_refs if entry else []),
        additional_instruction="Pass only if the reference mentions the need for proof of accommodation or lodging details."
    )

    # Leaf: AccommodationReference (custom existence, critical)
    evaluator.add_custom_node(
        result=len(_normalize_list_str(entry.accommodation_refs if entry else [])) > 0,
        id="AccommodationReference",
        desc="Provides URL reference for Grenada's accommodation proof requirement",
        parent=ap_node,
        critical=True
    )

    # HotelCheckInTiming (parallel, non-critical)
    hcit_node = evaluator.add_parallel(
        id="HotelCheckInTiming",
        desc="Accounts for standard hotel check-in times in Grenada",
        parent=ge_node,
        critical=False
    )

    evaluator.add_custom_node(
        result=bool(entry and entry.hotel_checkin_ack and entry.hotel_checkin_ack.strip()),
        id="CheckInTimeAwareness",
        desc="Acknowledges that standard hotel check-in time in Grenada is typically 3:00-4:00 PM",
        parent=hcit_node,
        critical=False
    )
    evaluator.add_custom_node(
        result=True,
        id="ArrivalTimingAlignment",
        desc="Plans arrival timing to align reasonably with hotel check-in availability",
        parent=hcit_node,
        critical=False
    )

    # VisaRequirementCheck (parallel, non-critical)
    vrc_node = evaluator.add_parallel(
        id="VisaRequirementCheck",
        desc="Confirms visa requirements for US citizens visiting Grenada",
        parent=ge_node,
        critical=False
    )

    # Leaf: NoVisaRequired (verify, non-critical)
    nvr_node = evaluator.add_leaf(
        id="NoVisaRequired",
        desc="Correctly identifies that US citizens do not need a visa for tourist stays in Grenada up to 90 days",
        parent=vrc_node,
        critical=False
    )
    claim_nvr = "US citizens do not need a visa for tourist visits to Grenada of up to 90 days."
    await evaluator.verify(
        claim=claim_nvr,
        node=nvr_node,
        sources=_normalize_list_str(entry.visa_refs if entry else []),
        additional_instruction="Pass only if the reference confirms that US citizens can enter Grenada visa-free for short tourist stays (e.g., up to 90 days)."
    )

    # Leaf: VisaInfoReference (custom existence, non-critical)
    evaluator.add_custom_node(
        result=len(_normalize_list_str(entry.visa_refs if entry else [])) > 0,
        id="VisaInfoReference",
        desc="Provides URL reference for Grenada visa requirements for US citizens",
        parent=vrc_node,
        critical=False
    )


async def verify_cost_optimization(
    evaluator: Evaluator,
    parent_node,
    cost_opt: CostOptimizationExtraction
) -> None:
    # CostOptimization (parallel, non-critical)
    co_node = evaluator.add_parallel(
        id="CostOptimization",
        desc="The overall plan considers cost optimization across ID acquisition and flight routing",
        parent=parent_node,
        critical=False
    )

    evaluator.add_custom_node(
        result=bool(cost_opt and cost_opt.total_id_cost and cost_opt.total_id_cost.strip()),
        id="TotalIDCostAnalysis",
        desc="Calculates or estimates the total cost of ID compliance for the round-trip journey",
        parent=co_node,
        critical=False
    )

    evaluator.add_custom_node(
        result=bool(cost_opt and cost_opt.flight_route_cost_consideration and cost_opt.flight_route_cost_consideration.strip()),
        id="FlightRouteCostConsideration",
        desc="Considers whether the chosen routing balances cost with convenience",
        parent=co_node,
        critical=False
    )

    evaluator.add_custom_node(
        result=bool(cost_opt and cost_opt.alternative_routes_discussed and cost_opt.alternative_routes_discussed.strip()),
        id="AlternativeRoutesDiscussion",
        desc="Acknowledges or evaluates alternative routing options and their trade-offs",
        parent=co_node,
        critical=False
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
    Evaluate an answer for the Bangor→Grenada March 2026 travel planning task.
    """
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Top-level: evaluate components independently to allow partial credit
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

    # Run extractions (can run concurrently)
    airport_task = evaluator.extract(
        prompt=prompt_extract_airport_info(),
        template_class=AirportInfo,
        extraction_name="airport_info"
    )
    timing_task = evaluator.extract(
        prompt=prompt_extract_departure_timing(),
        template_class=DepartureTimingInfo,
        extraction_name="departure_timing"
    )
    id_task = evaluator.extract(
        prompt=prompt_extract_id_compliance(),
        template_class=IDComplianceExtraction,
        extraction_name="id_compliance"
    )
    flight_task = evaluator.extract(
        prompt=prompt_extract_flight_plan(),
        template_class=FlightPlanExtraction,
        extraction_name="flight_plan"
    )
    entry_task = evaluator.extract(
        prompt=prompt_extract_entry_requirements(),
        template_class=EntryRequirementsExtraction,
        extraction_name="grenada_entry_requirements"
    )
    cost_task = evaluator.extract(
        prompt=prompt_extract_cost_optimization(),
        template_class=CostOptimizationExtraction,
        extraction_name="cost_optimization"
    )

    airport_info, dep_timing, id_extract, flight_plan, entry_req, cost_opt = await asyncio.gather(
        airport_task, timing_task, id_task, flight_task, entry_task, cost_task
    )

    # Root-level main node (non-critical, because it has a mix of critical and non-critical descendants)
    main_node = evaluator.add_parallel(
        id="TravelPlanCompliance",
        desc="The complete travel plan from Bangor, Maine to Grenada meets all necessary requirements for feasibility and regulatory compliance",
        parent=root,
        critical=False
    )

    # Build subtrees
    await verify_departure_logistics(evaluator, main_node, airport_info, dep_timing, flight_plan)
    await verify_id_compliance(evaluator, main_node, id_extract)
    await verify_flight_routing(evaluator, main_node, airport_info, flight_plan)
    await verify_entry_requirements(evaluator, main_node, entry_req)
    await verify_cost_optimization(evaluator, main_node, cost_opt)

    # Return evaluation summary
    return evaluator.get_summary()