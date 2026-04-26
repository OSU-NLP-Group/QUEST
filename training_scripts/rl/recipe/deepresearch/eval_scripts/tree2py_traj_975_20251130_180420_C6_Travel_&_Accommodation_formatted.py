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
TASK_ID = "avelo_epic_gcnp_2025_plan"
TASK_DESCRIPTION = (
    "A family living in coastal North Carolina wants to plan a budget-friendly vacation in 2025 using Avelo Airlines, "
    "which operates crew bases in several U.S. cities. They plan to visit Universal's newly opened Epic Universe theme park "
    "in the Orlando area, followed by a trip to Grand Canyon National Park's South Rim.\n\n"
    "Please provide a comprehensive travel plan that includes:\n\n"
    "1. Departure Airport: Identify which Avelo Airlines base airport serves the coastal North Carolina region. Provide the airport code, "
    "confirm it is an Avelo crew base and base of operations, and specify its geographic location.\n\n"
    "2. Orlando Flight Route: Determine the nonstop Avelo Airlines route(s) from the identified North Carolina base to the Orlando area. "
    "Specify the destination airport code(s), confirm nonstop service availability, and document the aircraft type used on this route.\n\n"
    "3. Epic Universe Admission: Document the admission requirements for Universal Epic Universe, including the official opening date, the location (city and resort), "
    "whether a separate dedicated admission ticket is required, and whether standard Park-to-Park tickets provide access to Epic Universe in 2025.\n\n"
    "4. Grand Canyon Lodging: Identify at least two different lodging properties available at Grand Canyon South Rim. For each property, provide the official property name, "
    "the total number of rooms or lodging units, key characteristics (such as historic designation, location, or room types), and confirm that South Rim lodging is available year-round.\n\n"
    "5. Avelo Network Size: State the total number of nonstop destinations served by Avelo from the identified North Carolina base airport.\n\n"
    "All information must be supported with valid reference URLs from your research."
)

STRICT_URL_ONLY = (
    "IMPORTANT: Judge strictly based on the provided URL webpage(s). If no valid URL(s) are provided "
    "or if the pages are irrelevant/inaccessible, you must respond 'Incorrect' (not supported)."
)

# --------------------------------------------------------------------------- #
# Extraction data models                                                      #
# --------------------------------------------------------------------------- #
class DepartureAirportExtraction(BaseModel):
    airport_name: Optional[str] = None
    airport_code: Optional[str] = None
    is_crew_base: Optional[str] = None  # 'yes' or 'no'
    base_confirmation_text: Optional[str] = None
    location_city: Optional[str] = None
    location_state: Optional[str] = None
    region_note: Optional[str] = None  # e.g., statement about serving coastal NC
    sources: List[str] = Field(default_factory=list)


class OrlandoRouteItem(BaseModel):
    destination_airport_code: Optional[str] = None  # e.g., MCO, SFB
    nonstop: Optional[str] = None  # 'yes' or 'no'
    aircraft_type: Optional[str] = None  # e.g., "Boeing 737-800"
    sources: List[str] = Field(default_factory=list)


class OrlandoFlightExtraction(BaseModel):
    routes: List[OrlandoRouteItem] = Field(default_factory=list)
    sources: List[str] = Field(default_factory=list)  # general route references if provided


class EpicUniverseAdmissionExtraction(BaseModel):
    official_opening_date: Optional[str] = None
    location_city: Optional[str] = None
    resort_complex: Optional[str] = None
    dedicated_ticket_required: Optional[str] = None  # 'yes' or 'no'
    park_to_park_access_2025: Optional[str] = None  # 'yes' or 'no' (whether standard Park-to-Park tickets include Epic)
    sources: List[str] = Field(default_factory=list)


class LodgingPropertyExtraction(BaseModel):
    property_name: Optional[str] = None
    rooms_or_units_count: Optional[str] = None  # Keep as string to be flexible (e.g., ranges)
    key_characteristics: List[str] = Field(default_factory=list)
    sources: List[str] = Field(default_factory=list)


class GrandCanyonLodgingExtraction(BaseModel):
    property_1: Optional[LodgingPropertyExtraction] = None
    property_2: Optional[LodgingPropertyExtraction] = None
    year_round_availability: Optional[str] = None  # 'yes' or 'no' for South Rim lodging being year-round
    year_round_sources: List[str] = Field(default_factory=list)


class AveloNetworkSizeExtraction(BaseModel):
    nonstop_destination_count: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_departure_airport() -> str:
    return """
    From the answer, extract the Avelo Airlines base airport that serves coastal North Carolina.
    Return:
    - airport_name: official name of the departure airport (e.g., "Wilmington International Airport").
    - airport_code: IATA code (e.g., "ILM").
    - is_crew_base: "yes" if the answer says it is an Avelo crew base/base of operations; otherwise "no".
    - base_confirmation_text: the exact phrasing used in the answer that confirms crew base/base of operations.
    - location_city: the city of the airport.
    - location_state: the state of the airport.
    - region_note: any statement from the answer that it serves the coastal North Carolina region.
    - sources: all URLs the answer cites for this departure/base claim.
    """


def prompt_extract_orlando_flight_route() -> str:
    return """
    From the answer, extract the nonstop Avelo Airlines route(s) from the identified NC base to the Orlando area.
    Return:
    - routes: an array with objects:
        - destination_airport_code: e.g., "MCO" or "SFB".
        - nonstop: "yes" if the answer explicitly claims nonstop service, else "no".
        - aircraft_type: the aircraft type claimed for this route (e.g., "Boeing 737-800", "Boeing 737").
        - sources: URLs supporting this route and/or aircraft type claim.
    - sources: any additional Orlando-route-related URLs from the answer (beyond per-route sources).
    """


def prompt_extract_epic_universe() -> str:
    return """
    From the answer, extract Epic Universe admission details for 2025:
    - official_opening_date: the specific opening date.
    - location_city: city where Epic Universe is located (e.g., Orlando).
    - resort_complex: the resort/complex it is part of (e.g., Universal Orlando Resort).
    - dedicated_ticket_required: "yes" if a separate dedicated ticket is required to enter Epic Universe; else "no".
    - park_to_park_access_2025: "yes" if standard Park-to-Park tickets provide access to Epic Universe in 2025; else "no".
    - sources: all URLs the answer cites for Epic Universe opening, location, and ticketing/access rules.
    """


def prompt_extract_grand_canyon_lodging() -> str:
    return """
    From the answer, extract details for at least two lodging properties at Grand Canyon National Park South Rim:
    - property_1:
        - property_name
        - rooms_or_units_count (keep as stated; can be a number or a range)
        - key_characteristics: array of at least one characteristic (historic designation, location, room/cabin types, etc.)
        - sources: URLs supporting property details
    - property_2:
        - property_name
        - rooms_or_units_count
        - key_characteristics
        - sources
    - year_round_availability: "yes" if the answer claims South Rim lodging is available year-round; else "no".
    - year_round_sources: URLs supporting the year-round availability claim.
    """


def prompt_extract_avelo_network_size() -> str:
    return """
    From the answer, extract the total number of nonstop destinations that Avelo serves from the identified North Carolina base:
    - nonstop_destination_count: the count as stated in the answer (string as-is).
    - sources: URLs supporting the destination count claim.
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _is_yes(val: Optional[str]) -> bool:
    return isinstance(val, str) and val.strip().lower() in {"yes", "true", "y"}


def _has_any_url(urls: Optional[List[str]]) -> bool:
    return bool(urls) and any(isinstance(u, str) and u.strip().startswith(("http://", "https://")) for u in urls)


def _first_non_empty(lst: List[str]) -> Optional[str]:
    for x in lst:
        if isinstance(x, str) and x.strip():
            return x
    return None


def _merge_sources(*source_lists: Optional[List[str]]) -> List[str]:
    result: List[str] = []
    seen: set = set()
    for lst in source_lists:
        if not lst:
            continue
        for u in lst:
            if isinstance(u, str) and u.strip() and u not in seen:
                result.append(u)
                seen.add(u)
    return result


# --------------------------------------------------------------------------- #
# Verification subroutines                                                    #
# --------------------------------------------------------------------------- #
async def verify_departure_airport(
    evaluator: Evaluator,
    root,
    dep: DepartureAirportExtraction,
) -> None:
    node = evaluator.add_parallel(
        id="departure_airport",
        desc="Identify the Avelo Airlines base airport that serves coastal North Carolina and provide required details.",
        parent=root,
        critical=True,
    )

    # 1) Airport identification (existence of name + code)
    ident_ok = bool(dep and dep.airport_name and dep.airport_code)
    evaluator.add_custom_node(
        result=ident_ok,
        id="airport_identification",
        desc="Provides the departure airport name and airport code for the Avelo base serving coastal North Carolina.",
        parent=node,
        critical=True,
    )

    # 2) Base confirmation (verify via sources)
    base_leaf = evaluator.add_leaf(
        id="base_confirmation",
        desc="Explicitly confirms the airport is an Avelo crew base and/or base of operations (as asked).",
        parent=node,
        critical=True,
    )
    base_claim = f"{dep.airport_name or 'The identified airport'} ({dep.airport_code or 'N/A'}) is an Avelo Airlines crew base (base of operations)."
    await evaluator.verify(
        claim=base_claim,
        node=base_leaf,
        sources=dep.sources,
        additional_instruction=STRICT_URL_ONLY + " Accept synonyms like 'crew base', 'base of operations', or 'operating base' that clearly indicate an Avelo crew base.",
    )

    # 3) Geographic location (verify via sources)
    geo_leaf = evaluator.add_leaf(
        id="geographic_location",
        desc="States the airport’s geographic location (at least city and state) and indicates it serves the coastal North Carolina region.",
        parent=node,
        critical=True,
    )
    city = dep.location_city or "the stated city"
    state = dep.location_state or "the stated state"
    geo_claim = f"{dep.airport_name or 'The identified airport'} ({dep.airport_code or 'N/A'}) is located in {city}, {state}."
    await evaluator.verify(
        claim=geo_claim,
        node=geo_leaf,
        sources=dep.sources,
        additional_instruction=STRICT_URL_ONLY + " Focus on verifying the city and state location for the airport.",
    )


async def verify_orlando_flight_route(
    evaluator: Evaluator,
    root,
    orl: OrlandoFlightExtraction,
    dep: DepartureAirportExtraction,
) -> None:
    node = evaluator.add_parallel(
        id="orlando_flight_route",
        desc="Determine the nonstop Avelo route(s) from the identified NC base to the Orlando area and provide required route details.",
        parent=root,
        critical=True,
    )

    # Choose first declared route for verification claims
    route0 = orl.routes[0] if orl and orl.routes else None
    base_code = dep.airport_code or "the identified base airport"
    # Consolidated sources for route verification
    route_sources = _merge_sources(orl.sources, *(r.sources for r in orl.routes) if orl and orl.routes else [])

    # 1) Nonstop routes listed -> verify a concrete nonstop statement for the first route (if available)
    nonstop_leaf = evaluator.add_leaf(
        id="nonstop_routes_listed",
        desc="Lists the nonstop route(s) from the identified base airport to the Orlando area and explicitly indicates nonstop service.",
        parent=node,
        critical=True,
    )
    if route0 and route0.destination_airport_code:
        nonstop_claim = f"Avelo operates nonstop service from {base_code} to {route0.destination_airport_code}."
    else:
        nonstop_claim = f"Avelo operates at least one nonstop route from {base_code} to the Orlando area."
    await evaluator.verify(
        claim=nonstop_claim,
        node=nonstop_leaf,
        sources=route_sources,
        additional_instruction=STRICT_URL_ONLY + " Verify that the route is nonstop. Orlando-area airports include Orlando International (MCO) or Orlando Sanford (SFB).",
    )

    # 2) Destination airport codes -> ensure agent provided code(s)
    has_dest_code = bool(route0 and route0.destination_airport_code and route0.destination_airport_code.strip())
    evaluator.add_custom_node(
        result=has_dest_code,
        id="destination_airport_codes",
        desc="Provides destination airport code(s) for the Orlando-area airport(s) served by the nonstop route(s).",
        parent=node,
        critical=True,
    )

    # 3) Aircraft type documented -> verify the aircraft type statement against sources
    aircraft_leaf = evaluator.add_leaf(
        id="aircraft_type_documented",
        desc="Documents the aircraft type used on the identified route(s).",
        parent=node,
        critical=True,
    )
    aircraft_type = route0.aircraft_type if route0 and route0.aircraft_type else "the stated aircraft type"
    if route0 and route0.destination_airport_code:
        ac_claim = f"Avelo operates {aircraft_type} on the {base_code}–{route0.destination_airport_code} route."
    else:
        ac_claim = f"Avelo operates {aircraft_type} on the identified nonstop route from {base_code} to the Orlando area."
    await evaluator.verify(
        claim=ac_claim,
        node=aircraft_leaf,
        sources=route_sources,
        additional_instruction=STRICT_URL_ONLY + " If the specific route page does not list aircraft, use reliable official sources in the provided URLs that tie Avelo to the specified aircraft type.",
    )


async def verify_epic_universe_admission(
    evaluator: Evaluator,
    root,
    epic: EpicUniverseAdmissionExtraction,
) -> None:
    node = evaluator.add_parallel(
        id="epic_universe_admission",
        desc="Document Epic Universe admission requirements for 2025 (opening date, location, ticketing rules).",
        parent=root,
        critical=True,
    )

    # 1) Official opening date
    open_leaf = evaluator.add_leaf(
        id="official_opening_date",
        desc="States the official opening date of Universal Epic Universe.",
        parent=node,
        critical=True,
    )
    open_claim = f"Epic Universe's official opening date is {epic.official_opening_date or 'the stated opening date'}."
    await evaluator.verify(
        claim=open_claim,
        node=open_leaf,
        sources=epic.sources,
        additional_instruction=STRICT_URL_ONLY + " Accept date formats that clearly represent the same date.",
    )

    # 2) Location: city and resort
    loc_leaf = evaluator.add_leaf(
        id="location_city_and_resort",
        desc="States the location (city) and the resort/complex it is part of.",
        parent=node,
        critical=True,
    )
    city = epic.location_city or "the stated city"
    resort = epic.resort_complex or "the stated resort"
    loc_claim = f"Epic Universe is located in {city} and is part of {resort}."
    await evaluator.verify(
        claim=loc_claim,
        node=loc_leaf,
        sources=epic.sources,
        additional_instruction=STRICT_URL_ONLY + " Look for explicit statements on official or authoritative pages.",
    )

    # 3) Dedicated ticket requirement
    dt_leaf = evaluator.add_leaf(
        id="dedicated_ticket_requirement",
        desc="States whether Epic Universe requires a separate, dedicated admission ticket.",
        parent=node,
        critical=True,
    )
    dt_phrase = "requires a separate, dedicated ticket" if _is_yes(epic.dedicated_ticket_required) else "does not require a separate, dedicated ticket"
    dt_claim = f"In 2025, Epic Universe {dt_phrase}."
    await evaluator.verify(
        claim=dt_claim,
        node=dt_leaf,
        sources=epic.sources,
        additional_instruction=STRICT_URL_ONLY + " Confirm whether entry to Epic Universe needs its own dedicated admission product in 2025.",
    )

    # 4) Park-to-Park access in 2025
    p2p_leaf = evaluator.add_leaf(
        id="park_to_park_access_2025",
        desc="States whether standard Park-to-Park tickets provide access to Epic Universe in 2025.",
        parent=node,
        critical=True,
    )
    p2p_phrase = "provide access to Epic Universe in 2025" if _is_yes(epic.park_to_park_access_2025) else "do not provide access to Epic Universe in 2025"
    p2p_claim = f"In 2025, standard Park-to-Park tickets {p2p_phrase}."
    await evaluator.verify(
        claim=p2p_claim,
        node=p2p_leaf,
        sources=epic.sources,
        additional_instruction=STRICT_URL_ONLY + " Verify whether regular Park-to-Park products include Epic Universe in 2025; if separate ticket-type is required, this should be 'no'.",
    )


async def verify_grand_canyon_lodging(
    evaluator: Evaluator,
    root,
    gcnp: GrandCanyonLodgingExtraction,
) -> None:
    node = evaluator.add_parallel(
        id="grand_canyon_lodging",
        desc="Identify at least two different Grand Canyon National Park South Rim lodging properties and provide required details for each, plus year-round availability.",
        parent=root,
        critical=True,
    )

    # Helper to build property subtree
    async def _verify_property(prop: Optional[LodgingPropertyExtraction], prop_id: str, prop_desc: str):
        pnode = evaluator.add_parallel(
            id=prop_id,
            desc=prop_desc,
            parent=node,
            critical=True,
        )

        # a) Property name existence + verification that property exists at South Rim
        name_ok = bool(prop and prop.property_name)
        evaluator.add_custom_node(
            result=name_ok,
            id=f"{prop_id}_property_name",
            desc="Provides the official property name.",
            parent=pnode,
            critical=True,
        )

        # b) Room/unit count verification via sources
        count_leaf = evaluator.add_leaf(
            id=f"{prop_id}_room_or_unit_count",
            desc="Provides the total number of rooms or lodging units.",
            parent=pnode,
            critical=True,
        )
        count_claim = f"{prop.property_name or 'The property'} has {prop.rooms_or_units_count or 'the stated number of'} rooms or lodging units."
        await evaluator.verify(
            claim=count_claim,
            node=count_leaf,
            sources=(prop.sources if prop else []),
            additional_instruction=STRICT_URL_ONLY + " Accept approximations or phrasing that clearly represent the same count.",
        )

        # c) Key characteristics (existence) and verify one characteristic against sources
        has_char = bool(prop and prop.key_characteristics and _first_non_empty(prop.key_characteristics))
        evaluator.add_custom_node(
            result=has_char,
            id=f"{prop_id}_key_characteristics",
            desc="Provides at least one key characteristic (e.g., historic designation, location, or room/cabin types).",
            parent=pnode,
            critical=True,
        )

    # Property 1 and 2
    await _verify_property(gcnp.property_1, "property_1", "First South Rim lodging property with required details.")
    await _verify_property(gcnp.property_2, "property_2", "Second, different South Rim lodging property with required details.")

    # Year-round availability verification
    year_leaf = evaluator.add_leaf(
        id="year_round_availability",
        desc="Confirms that South Rim lodging is available year-round.",
        parent=node,
        critical=True,
    )
    year_claim = "Grand Canyon National Park South Rim lodging is available year-round."
    combined_sources = _merge_sources(gcnp.year_round_sources,
                                     gcnp.property_1.sources if gcnp.property_1 else [],
                                     gcnp.property_2.sources if gcnp.property_2 else [])
    await evaluator.verify(
        claim=year_claim,
        node=year_leaf,
        sources=combined_sources,
        additional_instruction=STRICT_URL_ONLY + " Support that at least some South Rim lodges operate year-round (not the North Rim).",
    )


async def verify_avelo_network_size(
    evaluator: Evaluator,
    root,
    net: AveloNetworkSizeExtraction,
    dep: DepartureAirportExtraction,
) -> None:
    node = evaluator.add_parallel(
        id="avelo_network_size",
        desc="State the total number of nonstop destinations served by Avelo from the identified NC base airport (for the relevant 2025 context).",
        parent=root,
        critical=True,
    )

    leaf = evaluator.add_leaf(
        id="nonstop_destination_count",
        desc="Provides the total count of nonstop destinations from the identified base airport.",
        parent=node,
        critical=True,
    )
    base_code = dep.airport_code or "the identified base airport"
    count_str = net.nonstop_destination_count or "the stated number"
    claim = f"Avelo serves {count_str} nonstop destinations from {base_code}."
    await evaluator.verify(
        claim=claim,
        node=leaf,
        sources=net.sources,
        additional_instruction=STRICT_URL_ONLY + " Verify the count strictly for the named base airport and nonstop destinations.",
    )


async def verify_references(
    evaluator: Evaluator,
    root,
    dep: DepartureAirportExtraction,
    orl: OrlandoFlightExtraction,
    epic: EpicUniverseAdmissionExtraction,
    gcnp: GrandCanyonLodgingExtraction,
    net: AveloNetworkSizeExtraction,
) -> None:
    node = evaluator.add_parallel(
        id="references",
        desc="Provide valid reference URLs supporting claims in each required section.",
        parent=root,
        critical=True,
    )

    # Departure airport references
    evaluator.add_custom_node(
        result=_has_any_url(dep.sources),
        id="departure_airport_references",
        desc="Includes at least one valid reference URL supporting the departure airport identification/base claim(s).",
        parent=node,
        critical=True,
    )

    # Orlando route references (combine overall + per-route)
    orl_src = _merge_sources(orl.sources, *(r.sources for r in orl.routes) if orl and orl.routes else [])
    evaluator.add_custom_node(
        result=_has_any_url(orl_src),
        id="orlando_route_references",
        desc="Includes at least one valid reference URL supporting the nonstop route(s) and aircraft type claim(s).",
        parent=node,
        critical=True,
    )

    # Epic Universe references
    evaluator.add_custom_node(
        result=_has_any_url(epic.sources),
        id="epic_universe_references",
        desc="Includes at least one valid reference URL supporting the Epic Universe opening date, location, and ticketing/access rules.",
        parent=node,
        critical=True,
    )

    # Grand Canyon lodging references (combine both properties + year-round)
    gcnp_src = _merge_sources(
        gcnp.year_round_sources,
        gcnp.property_1.sources if gcnp.property_1 else [],
        gcnp.property_2.sources if gcnp.property_2 else [],
    )
    evaluator.add_custom_node(
        result=_has_any_url(gcnp_src),
        id="grand_canyon_lodging_references",
        desc="Includes at least one valid reference URL supporting the South Rim lodging property details and year-round availability claim.",
        parent=node,
        critical=True,
    )

    # Avelo network size references
    evaluator.add_custom_node(
        result=_has_any_url(net.sources),
        id="avelo_network_size_references",
        desc="Includes at least one valid reference URL supporting the nonstop destination count from the identified base airport.",
        parent=node,
        critical=True,
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
    Evaluate an answer for the 2025 Avelo + Epic Universe + Grand Canyon South Rim travel plan task.
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
        default_model=model,
    )

    # Extract all sections in parallel
    dep_task = evaluator.extract(
        prompt=prompt_extract_departure_airport(),
        template_class=DepartureAirportExtraction,
        extraction_name="departure_airport_extraction",
    )
    orl_task = evaluator.extract(
        prompt=prompt_extract_orlando_flight_route(),
        template_class=OrlandoFlightExtraction,
        extraction_name="orlando_flight_extraction",
    )
    epic_task = evaluator.extract(
        prompt=prompt_extract_epic_universe(),
        template_class=EpicUniverseAdmissionExtraction,
        extraction_name="epic_universe_extraction",
    )
    gcnp_task = evaluator.extract(
        prompt=prompt_extract_grand_canyon_lodging(),
        template_class=GrandCanyonLodgingExtraction,
        extraction_name="grand_canyon_lodging_extraction",
    )
    net_task = evaluator.extract(
        prompt=prompt_extract_avelo_network_size(),
        template_class=AveloNetworkSizeExtraction,
        extraction_name="avelo_network_size_extraction",
    )

    dep, orl, epic, gcnp, net = await asyncio.gather(dep_task, orl_task, epic_task, gcnp_task, net_task)

    # Build verification tree according to rubric
    await verify_departure_airport(evaluator, root, dep)
    await verify_orlando_flight_route(evaluator, root, orl, dep)
    await verify_epic_universe_admission(evaluator, root, epic)
    await verify_grand_canyon_lodging(evaluator, root, gcnp)
    await verify_avelo_network_size(evaluator, root, net, dep)
    await verify_references(evaluator, root, dep, orl, epic, gcnp, net)

    # Return structured evaluation summary
    return evaluator.get_summary()