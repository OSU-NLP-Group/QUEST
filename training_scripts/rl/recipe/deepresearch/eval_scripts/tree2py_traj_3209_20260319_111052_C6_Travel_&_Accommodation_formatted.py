import asyncio
import logging
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator, AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.llm_client.base_client import LLMClient


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "intl_journey_2026_breeze_cdg_tnr_angola"
TASK_DESCRIPTION = (
    "A US citizen is planning a multi-destination international trip in January 2026, departing from Charleston, "
    "South Carolina. They want to take advantage of Breeze Airways' newly launched international service to visit "
    "the Caribbean, then continue to Madagascar via a European hub, with potential return routing through Luanda, Angola.\n\n"
    "For this journey, provide the following information:\n"
    "1) Charleston to Caribbean Leg (Breeze's first international route from Charleston): destination city + IATA code, launch date, operating schedule, first available Saturday after launch;\n"
    "2) Caribbean to Madagascar Routing via European hub: hub city + IATA, airline operating direct Europe–TNR, approx weekly frequency, approx departure and arrival times;\n"
    "3) Travel Documentation: which of the countries (Caribbean destination, Madagascar, Angola) require US passports valid ≥6 months beyond stay;\n"
    "4) Angola Airport Information: IATA code and name of the new Luanda international airport replacing LAD, and the date Qatar Airways moved operations to the new airport;\n"
    "5) Timeline Verification: MLK Day observed date in 2026; dates Bangor International Airport (BGR) was closed due to an aircraft accident in Jan 2026.\n\n"
    "Each item should have supporting reference URL(s)."
)

# Ground-truth-like expectations embedded in rubric for judging
EXPECTED = {
    "breeze_destination_city": "Cancun",
    "breeze_destination_iata": "CUN",
    "breeze_operating_airline": "Breeze Airways",
    "breeze_launch_date": "January 17, 2026",
    "breeze_operating_days": "Saturdays only (seasonal)",  # wording-flexible
    "first_saturday_after_launch": "January 24, 2026",
    "hub_city": "Paris",
    "hub_iata": "CDG",
    "paris_tnr_airline": "Air France",
    "tnr_airport_name": "Ivato International Airport",
    "tnr_iata": "TNR",
    "paris_tnr_weekly_frequency": "≈3 per week",  # approximate
    "paris_departure_time_local": "around 10:25",
    "tnr_arrival_time_local": "around 23:15",
    "angola_new_airport_name": "Dr. António Agostinho Neto International Airport",
    "angola_new_airport_iata": "NBJ",
    "angola_old_airport_iata": "LAD",
    "qatar_transition_date": "December 5, 2025",
    "mlk_2026_observed": "Monday, January 19, 2026",
    "bgr_closure_dates": "January 25–29, 2026",
}


# --------------------------------------------------------------------------- #
# Data models                                                                 #
# --------------------------------------------------------------------------- #
class BreezeRouteExtraction(BaseModel):
    destination_city: Optional[str] = None
    destination_iata: Optional[str] = None
    operating_airline: Optional[str] = None
    launch_date: Optional[str] = None
    operating_days: Optional[str] = None
    first_saturday_after_launch: Optional[str] = None
    route_urls: List[str] = Field(default_factory=list)
    schedule_urls: List[str] = Field(default_factory=list)


class HubExtraction(BaseModel):
    hub_city: Optional[str] = None
    hub_iata: Optional[str] = None
    hub_urls: List[str] = Field(default_factory=list)


class ParisMadagascarExtraction(BaseModel):
    operating_airline: Optional[str] = None
    destination_airport_name: Optional[str] = None
    destination_iata: Optional[str] = None
    weekly_frequency: Optional[str] = None
    departure_time_local: Optional[str] = None
    arrival_time_local: Optional[str] = None
    airline_urls: List[str] = Field(default_factory=list)
    schedule_urls: List[str] = Field(default_factory=list)


class PassportExtraction(BaseModel):
    mexico_requires_six_months: Optional[str] = None  # Expect "yes" or "no" (case-insensitive) if provided
    madagascar_requires_six_months: Optional[str] = None  # Expect "yes"
    angola_requires_six_months: Optional[str] = None  # Expect "yes"
    passport_rule_urls: List[str] = Field(default_factory=list)


class AngolaExtraction(BaseModel):
    new_airport_name: Optional[str] = None
    new_airport_iata: Optional[str] = None
    old_airport_iata: Optional[str] = None
    airport_info_urls: List[str] = Field(default_factory=list)
    qatar_transition_date: Optional[str] = None
    qatar_transition_urls: List[str] = Field(default_factory=list)


class TimelineExtraction(BaseModel):
    mlk_day_2026: Optional[str] = None
    bgr_closure_dates: Optional[str] = None
    timeline_urls: List[str] = Field(default_factory=list)


class CombinedExtraction(BaseModel):
    breeze_route: Optional[BreezeRouteExtraction] = None
    europe_hub: Optional[HubExtraction] = None
    paris_tnr: Optional[ParisMadagascarExtraction] = None
    passport: Optional[PassportExtraction] = None
    angola: Optional[AngolaExtraction] = None
    timeline: Optional[TimelineExtraction] = None


# --------------------------------------------------------------------------- #
# Extraction Prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_combined() -> str:
    return """
    Extract the following structured information from the answer. Use the exact values and wordings as stated in the answer where possible. If a field is not mentioned, return null for that field or an empty list for URL arrays. For booleans like six-month passport validity, return 'yes' or 'no' strings.

    1) breeze_route:
       - destination_city: Caribbean destination city served by Breeze from Charleston (CHS) on its first international route (e.g., "Cancun").
       - destination_iata: IATA airport code (e.g., "CUN").
       - operating_airline: Airline operating the route (e.g., "Breeze Airways").
       - launch_date: Launch date of CHS→Caribbean service (e.g., "January 17, 2026").
       - operating_days: Operating schedule (e.g., "Saturdays only (seasonal)").
       - first_saturday_after_launch: The first available Saturday departure after the launch date in the user's plan (e.g., "January 24, 2026").
       - route_urls: All URLs cited that confirm the route (press releases/news/pages).
       - schedule_urls: All URLs cited that confirm the schedule/launch/specific operating days.

    2) europe_hub:
       - hub_city: European hub city for connections to Madagascar (e.g., "Paris").
       - hub_iata: IATA code (e.g., "CDG").
       - hub_urls: URLs supporting this hub/connection choice.

    3) paris_tnr:
       - operating_airline: Airline operating direct service from the European hub to Antananarivo (e.g., "Air France").
       - destination_airport_name: The Antananarivo airport name (e.g., "Ivato International Airport").
       - destination_iata: IATA code "TNR".
       - weekly_frequency: Approximate weekly frequency for the Europe–TNR segment (e.g., "3 per week").
       - departure_time_local: Approximate local departure time from Europe (e.g., "10:25").
       - arrival_time_local: Approximate local arrival time into TNR (e.g., "23:15").
       - airline_urls: URLs cited that confirm airline/route details.
       - schedule_urls: URLs cited that confirm schedule/times/frequencies.

    4) passport:
       - mexico_requires_six_months: "yes" if the answer states that Mexico (Cancun) requires US passport validity ≥ 6 months beyond stay; "no" if it states this is not required; null if unspecified.
       - madagascar_requires_six_months: "yes" or "no" (as stated).
       - angola_requires_six_months: "yes" or "no" (as stated).
       - passport_rule_urls: URLs cited for these passport validity requirements.

    5) angola:
       - new_airport_name: Name of Luanda’s new international airport replacing Quatro de Fevereiro.
       - new_airport_iata: The new airport’s IATA code.
       - old_airport_iata: The old airport’s IATA code (LAD).
       - airport_info_urls: URLs confirming new airport name and IATA code.
       - qatar_transition_date: The date Qatar Airways moved operations to the new airport.
       - qatar_transition_urls: URLs confirming the Qatar Airways transition date.

    6) timeline:
       - mlk_day_2026: The observed date for Martin Luther King Jr. Day in 2026 (e.g., "Monday, January 19, 2026").
       - bgr_closure_dates: The dates Bangor International Airport (BGR) was closed due to an aircraft accident in January 2026 (e.g., "January 25–29, 2026").
       - timeline_urls: URLs cited to support these dates.

    Return a single JSON object with keys: breeze_route, europe_hub, paris_tnr, passport, angola, timeline.
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def _merge_urls(*url_lists: Optional[List[str]]) -> List[str]:
    combined: List[str] = []
    for lst in url_lists:
        if not lst:
            continue
        for u in lst:
            if isinstance(u, str):
                u = u.strip()
                if u and u not in combined:
                    combined.append(u)
    return combined


def _yn_to_bool(val: Optional[str]) -> Optional[bool]:
    if val is None:
        return None
    s = val.strip().lower()
    if s in {"yes", "y", "true"}:
        return True
    if s in {"no", "n", "false"}:
        return False
    return None


# --------------------------------------------------------------------------- #
# Verification builders                                                       #
# --------------------------------------------------------------------------- #
async def build_outbound_initial_leg(evaluator: Evaluator, parent, data: Optional[BreezeRouteExtraction]) -> None:
    node_outbound = evaluator.add_parallel(
        id="Outbound_Initial_Leg",
        desc="Identify and verify the Charleston to Cancun flight segment using the newly launched Breeze Airways international service.",
        parent=parent,
        critical=False,
    )

    # Sub-node: Route Identification Details
    node_route = evaluator.add_parallel(
        id="Route_Identification_Details",
        desc="Identify the specific route and airline operating the Charleston to Caribbean service.",
        parent=node_outbound,
        critical=False,
    )

    route_urls = data.route_urls if data else []
    schedule_urls = data.schedule_urls if data else []
    route_related_urls = _merge_urls(route_urls, schedule_urls)

    # Destination_City_and_Code (Critical)
    leaf_dest = evaluator.add_leaf(
        id="Destination_City_and_Code",
        desc="Identify that the destination is Cancun International Airport (CUN).",
        parent=node_route,
        critical=True,
    )
    await evaluator.verify(
        claim="Breeze Airways' first international route from Charleston (CHS) serves Cancun International Airport (CUN) in Cancun, Mexico.",
        node=leaf_dest,
        sources=route_related_urls,
        additional_instruction="Verify that the cited page(s) explicitly mention CHS–CUN or Charleston–Cancun as Breeze's inaugural/first international route from Charleston, and that the destination IATA code is CUN.",
    )

    # Operating_Airline (Critical)
    leaf_airline = evaluator.add_leaf(
        id="Breeze_Operating_Airline",
        desc="Identify that Breeze Airways operates this route as its first international service.",
        parent=node_route,
        critical=True,
    )
    await evaluator.verify(
        claim="Breeze Airways operates the Charleston (CHS) to Cancun (CUN) route and this was Breeze's first international service from Charleston.",
        node=leaf_airline,
        sources=route_related_urls,
        additional_instruction="Look for press releases or news confirming that the CHS–CUN flight is part of Breeze's first international service from Charleston.",
    )

    # Route_Reference_URL (Critical) - ensure at least one route-related URL is provided
    evaluator.add_custom_node(
        result=len(route_urls) > 0,
        id="Route_Reference_URL",
        desc="Provide URL reference confirming Breeze Airways Charleston-Cancun route details.",
        parent=node_route,
        critical=True,
    )

    # Sub-node: Service Schedule Details
    node_sched = evaluator.add_parallel(
        id="Service_Schedule_Details",
        desc="Verify the launch date and operating schedule for the Charleston-Cancun service.",
        parent=node_outbound,
        critical=False,
    )

    # Service_Launch_Date (Critical)
    leaf_launch = evaluator.add_leaf(
        id="Service_Launch_Date",
        desc="Verify that the Charleston-Cancun service launched on January 17, 2026.",
        parent=node_sched,
        critical=True,
    )
    await evaluator.verify(
        claim="The Charleston (CHS) to Cancun (CUN) service launched on January 17, 2026.",
        node=leaf_launch,
        sources=route_related_urls,
        additional_instruction="Confirm that the cited source(s) explicitly indicate Jan 17, 2026 as the launch date for the Breeze CHS–CUN service.",
    )

    # Operating_Days (Critical)
    leaf_days = evaluator.add_leaf(
        id="Operating_Days",
        desc="Confirm that the service operates on Saturdays only on a seasonal basis.",
        parent=node_sched,
        critical=True,
    )
    await evaluator.verify(
        claim="The Breeze Charleston (CHS) to Cancun (CUN) service operates on Saturdays only on a seasonal basis.",
        node=leaf_days,
        sources=schedule_urls if schedule_urls else route_related_urls,
        additional_instruction="Accept wording variations indicating Saturday-only seasonal operation.",
    )

    # First_Available_Saturday (Non-critical; simple logical check)
    leaf_first_sat = evaluator.add_leaf(
        id="First_Available_Saturday",
        desc="Identify that the first available Saturday after the January 17, 2026 launch is January 24, 2026.",
        parent=node_sched,
        critical=False,
    )
    await evaluator.verify(
        claim="The first Saturday after January 17, 2026 is January 24, 2026.",
        node=leaf_first_sat,
        sources=None,
        additional_instruction="Do a simple calendar check. January 17, 2026 is a Saturday; the next Saturday is January 24, 2026.",
    )

    # Schedule_Reference_URL (Critical) - ensure at least one schedule URL is provided
    evaluator.add_custom_node(
        result=len(schedule_urls) > 0,
        id="Schedule_Reference_URL",
        desc="Provide URL reference confirming the service schedule and launch date.",
        parent=node_sched,
        critical=True,
    )


async def build_connection_to_europe(evaluator: Evaluator, parent, hub: Optional[HubExtraction], paris_tnr: Optional[ParisMadagascarExtraction]) -> None:
    node_conn = evaluator.add_parallel(
        id="Connection_to_Europe",
        desc="Specify the European hub for connecting from Cancun to Madagascar.",
        parent=parent,
        critical=False,
    )

    node_hub = evaluator.add_parallel(
        id="Hub_Identification",
        desc="Identify the European hub airport for Madagascar connections.",
        parent=node_conn,
        critical=False,
    )

    hub_urls = hub.hub_urls if hub else []
    # In case the answer used airline/schedule pages to justify the hub, allow those too.
    extra_urls = _merge_urls(paris_tnr.airline_urls if paris_tnr else [], paris_tnr.schedule_urls if paris_tnr else [])
    hub_supporting_urls = _merge_urls(hub_urls, extra_urls)

    # Hub_City_and_Code (Critical)
    leaf_hub = evaluator.add_leaf(
        id="Hub_City_and_Code",
        desc="Identify Paris Charles de Gaulle (CDG) as the European hub for connections to Madagascar.",
        parent=node_hub,
        critical=True,
    )
    await evaluator.verify(
        claim="Paris Charles de Gaulle (CDG) is an appropriate European hub offering connections/direct service to Antananarivo (TNR), Madagascar.",
        node=leaf_hub,
        sources=hub_supporting_urls,
        additional_instruction="Verify that the provided URLs support using CDG for connecting to TNR (e.g., Air France CDG–TNR service).",
    )

    # Hub_Reference_URL (Critical) - ensure at least one hub-supporting URL provided
    evaluator.add_custom_node(
        result=len(hub_supporting_urls) > 0,
        id="Hub_Reference_URL",
        desc="Provide URL reference supporting the hub selection and Madagascar connectivity.",
        parent=node_hub,
        critical=True,
    )


async def build_paris_to_madagascar(evaluator: Evaluator, parent, data: Optional[ParisMadagascarExtraction]) -> None:
    node_paris_tnr = evaluator.add_parallel(
        id="Paris_to_Madagascar_Segment",
        desc="Verify the Paris to Antananarivo, Madagascar flight details and schedule.",
        parent=parent,
        critical=False,
    )

    airline_urls = data.airline_urls if data else []
    schedule_urls = data.schedule_urls if data else []
    support_urls = _merge_urls(airline_urls, schedule_urls)

    # Airline_and_Route_Info
    node_airline_info = evaluator.add_parallel(
        id="Airline_and_Route_Info",
        desc="Identify the airline and route details for Paris to Madagascar service.",
        parent=node_paris_tnr,
        critical=False,
    )

    # Operating_Airline (Critical)
    leaf_op_airline = evaluator.add_leaf(
        id="ParisTNR_Operating_Airline",
        desc="Identify Air France as the airline operating direct service from Paris (CDG) to Antananarivo (TNR).",
        parent=node_airline_info,
        critical=True,
    )
    await evaluator.verify(
        claim="Air France operates direct service from Paris (CDG) to Antananarivo (TNR).",
        node=leaf_op_airline,
        sources=support_urls,
        additional_instruction="Check airline route pages or schedules confirming non-stop CDG–TNR by Air France.",
    )

    # Destination_Airport (Critical)
    leaf_dest_ap = evaluator.add_leaf(
        id="ParisTNR_Destination_Airport",
        desc="Verify that the destination airport in Madagascar is Ivato International Airport (TNR) in Antananarivo.",
        parent=node_airline_info,
        critical=True,
    )
    await evaluator.verify(
        claim="The destination airport in Antananarivo is Ivato International Airport (IATA: TNR).",
        node=leaf_dest_ap,
        sources=support_urls,
        additional_instruction="Allow variants such as 'Antananarivo Ivato' or 'TNR'.",
    )

    # Airline_Route_Reference_URL (Critical) - ensure route reference URL(s)
    evaluator.add_custom_node(
        result=len(support_urls) > 0,
        id="Airline_Route_Reference_URL",
        desc="Provide URL reference confirming Air France Paris-Madagascar service.",
        parent=node_airline_info,
        critical=True,
    )

    # Flight_Schedule_Info
    node_sched_info = evaluator.add_parallel(
        id="Flight_Schedule_Info",
        desc="Verify the flight frequency and timing for the Paris-Madagascar segment.",
        parent=node_paris_tnr,
        critical=False,
    )

    # Weekly_Frequency (Non-critical)
    leaf_freq = evaluator.add_leaf(
        id="Weekly_Frequency",
        desc="Verify that Air France operates approximately 3 flights per week on this route.",
        parent=node_sched_info,
        critical=False,
    )
    await evaluator.verify(
        claim="Air France operates approximately 3 flights per week between Paris (CDG) and Antananarivo (TNR).",
        node=leaf_freq,
        sources=schedule_urls if schedule_urls else support_urls,
        additional_instruction="Allow seasonal variability; 'approximately 3 per week' is acceptable if schedules indicate about three weekly frequencies.",
    )

    # Departure_Time (Non-critical)
    leaf_dep = evaluator.add_leaf(
        id="Departure_Time",
        desc="Confirm that flights depart Paris around 10:25 local time.",
        parent=node_sched_info,
        critical=False,
    )
    await evaluator.verify(
        claim="Flights from Paris (CDG) to Antananarivo (TNR) depart around 10:25 local time.",
        node=leaf_dep,
        sources=schedule_urls if schedule_urls else support_urls,
        additional_instruction="Accept reasonable variations (±30 minutes) and seasonal shifts.",
    )

    # Arrival_Time (Non-critical)
    leaf_arr = evaluator.add_leaf(
        id="Arrival_Time",
        desc="Confirm that flights arrive in Antananarivo around 23:15 local time.",
        parent=node_sched_info,
        critical=False,
    )
    await evaluator.verify(
        claim="Flights arrive in Antananarivo (TNR) around 23:15 local time.",
        node=leaf_arr,
        sources=schedule_urls if schedule_urls else support_urls,
        additional_instruction="Accept reasonable variations (±45 minutes) and seasonal shifts.",
    )

    # Schedule_Reference_URL (Critical)
    evaluator.add_custom_node(
        result=len(schedule_urls) > 0 or len(support_urls) > 0,
        id="ParisTNR_Schedule_Reference_URL",
        desc="Provide URL reference confirming the flight schedule details.",
        parent=node_sched_info,
        critical=True,
    )


async def build_passport_requirements(evaluator: Evaluator, parent, data: Optional[PassportExtraction]) -> None:
    node_docs = evaluator.add_parallel(
        id="Travel_Documentation_Requirements",
        desc="Identify which destination countries require US passport holders to have passports valid for at least 6 months beyond their intended stay.",
        parent=parent,
        critical=False,
    )

    node_validity = evaluator.add_parallel(
        id="Passport_Validity_by_Country",
        desc="Verify the six-month passport validity requirement for each destination country.",
        parent=node_docs,
        critical=False,
    )

    urls = data.passport_rule_urls if data else []

    # Mexico_Requirement (Non-critical, dynamic)
    if data and data.mexico_requires_six_months is not None:
        mx_bool = _yn_to_bool(data.mexico_requires_six_months)
        mx_leaf = evaluator.add_leaf(
            id="Mexico_Requirement",
            desc="Determine whether Mexico (Cancun) requires the six-month validity rule for US passport holders.",
            parent=node_validity,
            critical=False,
        )
        if mx_bool is True:
            claim = "Mexico requires U.S. passport holders to have at least six months of validity remaining beyond their intended stay."
        else:
            claim = "Mexico does not require a full six months of passport validity for U.S. citizens; typically the passport must be valid for the duration of stay."
        await evaluator.verify(
            claim=claim,
            node=mx_leaf,
            sources=urls,
            additional_instruction="Use authoritative sources (e.g., U.S. State Department or Mexican government/embassy sites). Allow policy wording variations.",
        )
    else:
        evaluator.add_custom_node(
            result=False,
            id="Mexico_Requirement",
            desc="Determine whether Mexico (Cancun) requires the six-month validity rule for US passport holders.",
            parent=node_validity,
            critical=False,
        )

    # Madagascar_Requirement (Critical)
    mg_leaf = evaluator.add_leaf(
        id="Madagascar_Requirement",
        desc="Confirm that Madagascar requires US passport holders to have passports valid for at least 6 months beyond the intended stay.",
        parent=node_validity,
        critical=True,
    )
    await evaluator.verify(
        claim="Madagascar requires U.S. passport holders to have at least six months of validity remaining beyond their intended stay.",
        node=mg_leaf,
        sources=urls,
        additional_instruction="Prefer U.S. State Department or Madagascar government/embassy sources. Accept clear statements of ≥6 months validity requirement.",
    )

    # Angola_Requirement (Critical)
    ao_leaf = evaluator.add_leaf(
        id="Angola_Requirement",
        desc="Confirm that Angola requires US passport holders to have passports valid for at least 6 months beyond the intended stay.",
        parent=node_validity,
        critical=True,
    )
    await evaluator.verify(
        claim="Angola requires U.S. passport holders to have at least six months of validity remaining beyond their intended stay.",
        node=ao_leaf,
        sources=urls,
        additional_instruction="Prefer U.S. State Department or Angola government/embassy sources. Accept clear statements of ≥6 months validity requirement.",
    )

    # Passport_Rule_Reference_URL (Critical) - ensure citation(s)
    evaluator.add_custom_node(
        result=len(urls) > 0,
        id="Passport_Rule_Reference_URL",
        desc="Provide URL reference documenting the six-month passport validity rule for the applicable countries.",
        parent=node_validity,
        critical=True,
    )


async def build_return_via_angola(evaluator: Evaluator, parent, data: Optional[AngolaExtraction]) -> None:
    node_angola = evaluator.add_parallel(
        id="Return_Routing_via_Angola",
        desc="Provide information about Angola's new international airport for potential return routing through Luanda.",
        parent=parent,
        critical=False,
    )

    node_airport = evaluator.add_parallel(
        id="New_Airport_Identification",
        desc="Identify the new international airport in Luanda, Angola.",
        parent=node_angola,
        critical=False,
    )

    airport_urls = data.airport_info_urls if data else []
    qatar_urls = data.qatar_transition_urls if data else []

    # Airport_Name (Critical)
    leaf_name = evaluator.add_leaf(
        id="Airport_Name",
        desc="Identify that the new airport is named Dr. António Agostinho Neto International Airport.",
        parent=node_airport,
        critical=True,
    )
    await evaluator.verify(
        claim="Luanda's new international airport is named Dr. António Agostinho Neto International Airport.",
        node=leaf_name,
        sources=airport_urls,
        additional_instruction="Check authoritative or aviation sources confirming the new Luanda airport name.",
    )

    # New_Airport_IATA_Code (Critical)
    leaf_new_iata = evaluator.add_leaf(
        id="New_Airport_IATA_Code",
        desc="Verify that the IATA code for the new Luanda airport is NBJ.",
        parent=node_airport,
        critical=True,
    )
    await evaluator.verify(
        claim="The IATA code for Luanda's new international airport (Dr. António Agostinho Neto International Airport) is NBJ.",
        node=leaf_new_iata,
        sources=airport_urls,
        additional_instruction="Verify that the page explicitly lists NBJ as the IATA code for the new airport.",
    )

    # Old_Airport_IATA_Code (Non-critical)
    leaf_old_iata = evaluator.add_leaf(
        id="Old_Airport_IATA_Code",
        desc="Verify that the old Luanda airport (Quatro de Fevereiro) has IATA code LAD.",
        parent=node_airport,
        critical=False,
    )
    await evaluator.verify(
        claim="Luanda's old Quatro de Fevereiro Airport has IATA code LAD.",
        node=leaf_old_iata,
        sources=airport_urls,
        additional_instruction="Any credible aviation source or airport profile stating LAD is acceptable.",
    )

    # Airport_ID_Reference_URL (Critical) - ensure airport info URL(s)
    evaluator.add_custom_node(
        result=len(airport_urls) > 0,
        id="Airport_ID_Reference_URL",
        desc="Provide URL reference confirming the new airport name and IATA code.",
        parent=node_airport,
        critical=True,
    )

    # Operational_Transition_Info
    node_transition = evaluator.add_parallel(
        id="Operational_Transition_Info",
        desc="Verify the transition timeline for airline operations moving to the new airport.",
        parent=node_angola,
        critical=False,
    )

    # Qatar_Airways_Transition_Date (Critical)
    leaf_qatar = evaluator.add_leaf(
        id="Qatar_Airways_Transition_Date",
        desc="Confirm that Qatar Airways moved its Luanda operations from LAD to NBJ effective December 5, 2025.",
        parent=node_transition,
        critical=True,
    )
    await evaluator.verify(
        claim="Qatar Airways moved its Luanda operations from LAD to NBJ effective December 5, 2025.",
        node=leaf_qatar,
        sources=qatar_urls if qatar_urls else airport_urls,
        additional_instruction="Prefer airline or airport notices and reputable aviation news confirming the operational move on Dec 5, 2025.",
    )

    # Transition_Reference_URL (Critical)
    evaluator.add_custom_node(
        result=len(qatar_urls) > 0,
        id="Transition_Reference_URL",
        desc="Provide URL reference confirming the Qatar Airways transition date and details.",
        parent=node_transition,
        critical=True,
    )


async def build_timeline_verification(evaluator: Evaluator, parent, data: Optional[TimelineExtraction]) -> None:
    node_timeline = evaluator.add_parallel(
        id="Timeline_Verification",
        desc="Verify additional timeline information relevant to January 2026 travel planning.",
        parent=parent,
        critical=False,
    )

    node_events = evaluator.add_parallel(
        id="Holiday_and_Events_Timeline",
        desc="Confirm specific dates for holidays and airport operational events in January 2026.",
        parent=node_timeline,
        critical=False,
    )

    urls = data.timeline_urls if data else []

    # MLK_Holiday_Date (Non-critical)
    leaf_mlk = evaluator.add_leaf(
        id="MLK_Holiday_Date",
        desc="Verify that MLK Day in 2026 was observed on Monday, January 19, 2026.",
        parent=node_events,
        critical=False,
    )
    await evaluator.verify(
        claim="Martin Luther King Jr. Day in 2026 was observed on Monday, January 19, 2026.",
        node=leaf_mlk,
        sources=urls if urls else None,
        additional_instruction="Use authoritative calendar references (e.g., U.S. federal holiday calendars).",
    )

    # Bangor_Airport_Closure_Dates (Non-critical)
    leaf_bgr = evaluator.add_leaf(
        id="Bangor_Airport_Closure_Dates",
        desc="Confirm that Bangor International Airport (BGR) was closed from January 25-29, 2026 due to an aircraft accident.",
        parent=node_events,
        critical=False,
    )
    await evaluator.verify(
        claim="Bangor International Airport (BGR) was closed from January 25 to 29, 2026 due to an aircraft accident.",
        node=leaf_bgr,
        sources=urls if urls else None,
        additional_instruction="Accept credible local news, airport notices, or NOTAM summaries confirming the closure dates and cause.",
    )

    # Timeline_Reference_URL (Non-critical)
    evaluator.add_custom_node(
        result=len(urls) > 0,
        id="Timeline_Reference_URL",
        desc="Provide URL reference for January 2026 timeline events.",
        parent=node_events,
        critical=False,
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
        default_model=model,
    )

    # Extract combined structured information from the answer
    extracted: CombinedExtraction = await evaluator.extract(
        prompt=prompt_extract_combined(),
        template_class=CombinedExtraction,
        extraction_name="combined_extraction",
    )

    # Build verification tree according to rubric
    # Root node (non-critical, parallel) already created by initialize()

    # 1) Outbound initial leg (Breeze CHS → Cancun)
    await build_outbound_initial_leg(evaluator, root, extracted.breeze_route or BreezeRouteExtraction())

    # 2) Connection to Europe (CDG hub)
    await build_connection_to_europe(evaluator, root, extracted.europe_hub or HubExtraction(), extracted.paris_tnr or ParisMadagascarExtraction())

    # 3) Paris to Madagascar segment (Air France CDG → TNR)
    await build_paris_to_madagascar(evaluator, root, extracted.paris_tnr or ParisMadagascarExtraction())

    # 4) Travel documentation requirements
    await build_passport_requirements(evaluator, root, extracted.passport or PassportExtraction())

    # 5) Return routing via Angola (new airport, Qatar move date)
    await build_return_via_angola(evaluator, root, extracted.angola or AngolaExtraction())

    # 6) Timeline verification (MLK day and BGR closures)
    await build_timeline_verification(evaluator, root, extracted.timeline or TimelineExtraction())

    # Record ground-truth-style expectations (from rubric) to help interpretation
    evaluator.add_ground_truth({
        "expected": EXPECTED
    }, gt_type="rubric_expectations")

    return evaluator.get_summary()