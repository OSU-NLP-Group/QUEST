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
TASK_ID = "vacation_sixflags_avelo_curacao_bangor_2026"
TASK_DESCRIPTION = (
    "You are planning a vacation that includes visiting Six Flags Magic Mountain in California and later traveling to Curacao. "
    "Please provide the following information:\n\n"
    "1. Identify two hotels located less than 1 mile from Six Flags Magic Mountain in Valencia, California. For each hotel, provide:\n"
    "   - The hotel name\n"
    "   - The exact distance from Six Flags Magic Mountain (must be less than 1 mile)\n"
    "   - The total number of rooms in the hotel\n"
    "   - The minimum number of accessible rooms required by ADA regulations based on the hotel's total room count\n\n"
    "2. Verify that Avelo Airlines operates flights from New Haven, CT (HVN) to a California destination, with service scheduled through at least mid-November 2026. Confirm which California destination(s) Avelo serves from New Haven.\n\n"
    "3. Provide the entry requirements for U.S. travelers visiting Curacao, including:\n"
    "   - Whether a Digital Immigration Card (DI card) is required\n"
    "   - The timeframe for completing the DI card before travel\n"
    "   - Passport validity requirements\n\n"
    "4. Provide information about the plane crash that occurred at Bangor International Airport in January 2026, including:\n"
    "   - The exact date of the incident\n"
    "   - The total number of fatalities\n"
    "   - The aircraft type involved in the crash\n\n"
    "For all information provided, include reference URLs from reliable sources to support your answers."
)


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class HotelItem(BaseModel):
    name: Optional[str] = None
    distance: Optional[str] = None
    distance_sources: List[str] = Field(default_factory=list)
    room_count: Optional[str] = None
    room_count_sources: List[str] = Field(default_factory=list)
    ada_min_accessible_rooms: Optional[str] = None
    ada_standard_sources: List[str] = Field(default_factory=list)


class HotelsExtraction(BaseModel):
    hotel1: Optional[HotelItem] = None
    hotel2: Optional[HotelItem] = None


class AveloExtraction(BaseModel):
    ca_destinations_from_hvn: List[str] = Field(default_factory=list)
    operates_sources: List[str] = Field(default_factory=list)
    destinations_sources: List[str] = Field(default_factory=list)
    schedule_through_date: Optional[str] = None
    schedule_sources: List[str] = Field(default_factory=list)


class CuracaoEntryExtraction(BaseModel):
    di_required: Optional[str] = None  # e.g., "yes" or "no"
    di_timeframe: Optional[str] = None  # e.g., "within 7 days before departure"
    passport_validity: Optional[str] = None  # e.g., "valid for duration of stay"
    sources: List[str] = Field(default_factory=list)


class BangorIncidentExtraction(BaseModel):
    incident_date: Optional[str] = None  # e.g., "January 12, 2026"
    fatalities: Optional[str] = None     # e.g., "3"
    aircraft_type: Optional[str] = None  # e.g., "Cessna 402"
    sources: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_hotels() -> str:
    return (
        "Extract details for two hotels mentioned in the answer that are near Six Flags Magic Mountain in Valencia, California. "
        "For each hotel, return the following fields:\n"
        "1) name: The hotel name as stated.\n"
        "2) distance: The exact distance to 'Six Flags Magic Mountain' as stated (e.g., '0.6 miles', '0.9 mi', '800 meters').\n"
        "3) distance_sources: An array of all URLs that specifically support the distance claim to Six Flags Magic Mountain.\n"
        "4) room_count: The total number of rooms in the hotel as stated (e.g., '245', '300 rooms').\n"
        "5) room_count_sources: An array of all URLs that specifically support the total room count claim.\n"
        "6) ada_min_accessible_rooms: The minimum number of accessible rooms required by ADA regulations for this hotel, "
        "   based on the stated room_count (as stated in the answer).\n"
        "7) ada_standard_sources: An array of authoritative ADA/DOJ standard URLs cited in the answer that justify the accessible-room minimum.\n\n"
        "Return a JSON with keys 'hotel1' and 'hotel2', each as an object with the above fields. "
        "If any field is missing in the answer for a hotel, set the field to null (for strings) or [] (for arrays). "
        "Only include URLs explicitly provided in the answer text."
    )


def prompt_extract_avelo() -> str:
    return (
        "Extract information about Avelo Airlines service from New Haven, CT (HVN) to California from the answer. "
        "Return:\n"
        "1) ca_destinations_from_hvn: Array of California destination names (airport/city) that Avelo serves directly from HVN as stated in the answer (e.g., 'Burbank (BUR)').\n"
        "2) operates_sources: Array of URLs that support the claim that Avelo operates flights out of HVN.\n"
        "3) destinations_sources: Array of URLs that support the claim about California destinations served from HVN.\n"
        "4) schedule_through_date: The stated date phrase indicating service is scheduled through at least mid-November 2026 (e.g., 'mid-November 2026', 'November 15, 2026').\n"
        "5) schedule_sources: Array of URLs that support the schedule-through claim.\n\n"
        "If any item is missing, set it to null (for string) or [] (for arrays). Extract only URLs that appear in the answer."
    )


def prompt_extract_curacao() -> str:
    return (
        "Extract Curacao entry requirements for U.S. travelers from the answer. Return:\n"
        "1) di_required: 'yes' or 'no' indicating whether a Digital Immigration Card (DI card) is required.\n"
        "2) di_timeframe: The timeframe for completing the DI card before travel as stated (e.g., '48-72 hours before departure').\n"
        "3) passport_validity: The passport validity requirement (e.g., 'valid for duration of stay', 'valid for 3 months beyond entry').\n"
        "4) sources: Array of URLs that support these Curacao entry requirements.\n\n"
        "If any field is missing, set it to null (for strings) or [] (for sources). Extract only URLs that appear in the answer."
    )


def prompt_extract_bangor() -> str:
    return (
        "Extract details of the plane crash incident at Bangor International Airport in January 2026 from the answer. Return:\n"
        "1) incident_date: The exact date of the incident (e.g., 'January 8, 2026').\n"
        "2) fatalities: The total number of fatalities (e.g., '2').\n"
        "3) aircraft_type: The aircraft type involved (e.g., 'Beechcraft King Air').\n"
        "4) sources: Array of URLs that support the incident date, fatalities, and aircraft type.\n\n"
        "If any field is missing, set it to null (for strings) or [] (for sources). Extract only URLs that appear in the answer."
    )


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def _normalize_name(name: Optional[str]) -> str:
    if not name:
        return ""
    return "".join(ch.lower() for ch in name.strip() if ch.isalnum())


def _union_urls(*lists: List[str]) -> List[str]:
    s = []
    seen = set()
    for lst in lists:
        for u in lst:
            if u and u not in seen:
                s.append(u)
                seen.add(u)
    return s


# --------------------------------------------------------------------------- #
# Verification functions                                                      #
# --------------------------------------------------------------------------- #
async def verify_hotels(evaluator: Evaluator, parent_node, hotels: HotelsExtraction) -> None:
    # Parent node for Hotels (critical)
    hotels_node = evaluator.add_parallel(
        id="hotels_near_sixflags",
        desc="Identify two distinct hotels located < 1 mile from Six Flags Magic Mountain and provide required attributes with sources.",
        parent=parent_node,
        critical=True
    )

    # Distinct hotels check
    h1_name = hotels.hotel1.name if hotels.hotel1 else None
    h2_name = hotels.hotel2.name if hotels.hotel2 else None
    distinct = bool(h1_name and h2_name and _normalize_name(h1_name) != _normalize_name(h2_name))
    evaluator.add_custom_node(
        result=distinct,
        id="two_distinct_hotels",
        desc="Provides two hotels that are distinct (not the same property repeated).",
        parent=hotels_node,
        critical=True
    )

    # Helper to verify a single hotel block
    async def _verify_one_hotel(h: Optional[HotelItem], idx: int) -> None:
        hotel_node = evaluator.add_parallel(
            id=f"hotel_{idx}",
            desc=f"Hotel #{idx} requirements.",
            parent=hotels_node,
            critical=True
        )

        # Name provided (existence)
        evaluator.add_custom_node(
            result=bool(h and h.name and h.name.strip()),
            id=f"hotel_{idx}_name",
            desc=f"Hotel #{idx} name is provided.",
            parent=hotel_node,
            critical=True
        )

        # Distance source presence
        dist_src_present = bool(h and h.distance_sources and len(h.distance_sources) > 0)
        dist_src_node = evaluator.add_custom_node(
            result=dist_src_present,
            id=f"hotel_{idx}_distance_source_url",
            desc=f"Provides a reliable reference URL supporting Hotel #{idx} distance claim.",
            parent=hotel_node,
            critical=True
        )

        # Distance claim: provided and < 1 mile
        dist_node = evaluator.add_leaf(
            id=f"hotel_{idx}_distance",
            desc=f"Hotel #{idx} exact distance from Six Flags Magic Mountain is provided and is < 1 mile.",
            parent=hotel_node,
            critical=True
        )
        distance_text = h.distance if h else ""
        await evaluator.verify(
            claim=f"The hotel's distance to Six Flags Magic Mountain is stated as '{distance_text}', and that distance is less than 1 mile.",
            node=dist_node,
            sources=(h.distance_sources if h else []),
            additional_instruction=(
                "Verify the page explicitly states the hotel's distance to 'Six Flags Magic Mountain' and that the value is strictly less than 1.0 mile. "
                "Allow minor rounding or unit conversions (e.g., feet/meters) only if it clearly indicates < 1 mile. "
                "Do not accept vague phrasing like 'next to the park' without a numeric distance."
            ),
        )

        # Room count source presence
        room_src_present = bool(h and h.room_count_sources and len(h.room_count_sources) > 0)
        room_src_node = evaluator.add_custom_node(
            result=room_src_present,
            id=f"hotel_{idx}_room_count_source_url",
            desc=f"Provides a reliable reference URL supporting Hotel #{idx} room-count claim.",
            parent=hotel_node,
            critical=True
        )

        # Room count claim
        room_node = evaluator.add_leaf(
            id=f"hotel_{idx}_room_count",
            desc=f"Hotel #{idx} total number of rooms is provided.",
            parent=hotel_node,
            critical=True
        )
        room_text = h.room_count if h else ""
        await evaluator.verify(
            claim=f"The hotel's total number of rooms is stated as '{room_text}'.",
            node=room_node,
            sources=(h.room_count_sources if h else []),
            additional_instruction=(
                "Confirm the total room count value appears on the cited page for this hotel. "
                "Accept reasonable formatting variants (e.g., '300 rooms', 'Total rooms: 300')."
            ),
        )

        # ADA standard source presence
        ada_src_present = bool(h and h.ada_standard_sources and len(h.ada_standard_sources) > 0)
        ada_src_node = evaluator.add_custom_node(
            result=ada_src_present,
            id=f"hotel_{idx}_ada_source_url",
            desc=f"Provides a reliable reference URL for the ADA/DOJ accessibility standard used for the accessible-room minimum calculation.",
            parent=hotel_node,
            critical=True
        )

        # ADA minimum accessible rooms claim
        ada_node = evaluator.add_leaf(
            id=f"hotel_{idx}_ada_min_accessible_rooms",
            desc=f"Hotel #{idx} minimum ADA-required accessible rooms is provided and correctly derived from the stated room count using an authoritative standard.",
            parent=hotel_node,
            critical=True
        )
        ada_min_text = h.ada_min_accessible_rooms if h else ""
        await evaluator.verify(
            claim=(
                f"For a hotel with total rooms '{room_text}', the minimum ADA-required number of accessible rooms is '{ada_min_text}' "
                f"according to the cited ADA/DOJ standard."
            ),
            node=ada_node,
            sources=(h.ada_standard_sources if h else []),
            additional_instruction=(
                "Use the cited ADA/DOJ standard page(s) to confirm the minimum number of accessible sleeping rooms required for transient lodging "
                "given the stated total room count. Allow reasonable matching if the standard provides ranges or tiered thresholds; "
                "the stated minimum must align with the standard."
            ),
        )

    await _verify_one_hotel(hotels.hotel1, 1)
    await _verify_one_hotel(hotels.hotel2, 2)


async def verify_avelo(evaluator: Evaluator, parent_node, avelo: AveloExtraction) -> None:
    avelo_node = evaluator.add_parallel(
        id="avelo_airlines_route",
        desc="Verify Avelo operates HVN→California service, destinations, and schedule through at least mid-Nov 2026.",
        parent=parent_node,
        critical=True
    )

    # Operates from HVN
    operates_node = evaluator.add_leaf(
        id="avelo_operates_from_hvn",
        desc="Confirms Avelo Airlines operates flights out of New Haven, CT (HVN).",
        parent=avelo_node,
        critical=True
    )
    await evaluator.verify(
        claim="Avelo Airlines operates flights from New Haven (HVN).",
        node=operates_node,
        sources=avelo.operates_sources,
        additional_instruction="Confirm via official Avelo or airport/route pages (or equivalent authoritative sources) that Avelo serves HVN."
    )

    # California destinations from HVN (at least one)
    dest_list_present = len(avelo.ca_destinations_from_hvn) > 0
    evaluator.add_custom_node(
        result=dest_list_present,
        id="avelo_ca_destination_list_presence",
        desc="At least one California destination is identified for Avelo from HVN.",
        parent=avelo_node,
        critical=True
    )

    dests_node = evaluator.add_leaf(
        id="avelo_ca_destination_list",
        desc="Identifies the California destination(s) Avelo serves from HVN (at least one).",
        parent=avelo_node,
        critical=True
    )
    dest_text = ", ".join(avelo.ca_destinations_from_hvn) if avelo.ca_destinations_from_hvn else ""
    await evaluator.verify(
        claim=f"From HVN, Avelo serves the following California destination(s): {dest_text}.",
        node=dests_node,
        sources=avelo.destinations_sources,
        additional_instruction=(
            "Check the cited route map, booking page, or announcement to confirm the California destinations listed originate from HVN."
        ),
    )

    # Schedule through mid-November 2026
    sched_node = evaluator.add_leaf(
        id="avelo_schedule_through_mid_nov_2026",
        desc="Confirms schedules/service availability through at least mid-November 2026.",
        parent=avelo_node,
        critical=True
    )
    sched_text = avelo.schedule_through_date or "mid-November 2026"
    await evaluator.verify(
        claim=f"Avelo has HVN→California service scheduled through at least {sched_text}.",
        node=sched_node,
        sources=avelo.schedule_sources,
        additional_instruction=(
            "Confirm that published schedules, booking availability, or official announcements indicate service on or after approximately November 15, 2026."
        ),
    )

    # Source URL presence (union over categories)
    union = _union_urls(avelo.operates_sources, avelo.destinations_sources, avelo.schedule_sources)
    evaluator.add_custom_node(
        result=bool(union),
        id="avelo_source_url",
        desc="Provides reliable reference URL(s) supporting HVN service, the CA destination(s), and the schedule-through date claim.",
        parent=avelo_node,
        critical=True
    )


async def verify_curacao(evaluator: Evaluator, parent_node, cura: CuracaoEntryExtraction) -> None:
    cura_node = evaluator.add_parallel(
        id="curacao_entry_requirements",
        desc="Provide Curacao entry requirements for U.S. travelers (DI card requirement, timing, passport validity), with sources.",
        parent=parent_node,
        critical=True
    )

    # DI Card requirement
    di_req_node = evaluator.add_leaf(
        id="curacao_di_card_requirement",
        desc="States whether a Digital Immigration Card (DI card) is required for U.S. travelers.",
        parent=cura_node,
        critical=True
    )
    di_req_text = (cura.di_required or "").strip()
    await evaluator.verify(
        claim=f"A Digital Immigration Card (DI card) is {'required' if di_req_text.lower() == 'yes' else 'not required'} for U.S. travelers to Curacao.",
        node=di_req_node,
        sources=cura.sources,
        additional_instruction="Verify on official Curacao government/immigration/tourism sources whether a DI card is required."
    )

    # DI Card timeframe
    di_time_node = evaluator.add_leaf(
        id="curacao_di_card_timeframe",
        desc="States the timeframe for when the DI card must/can be completed before travel.",
        parent=cura_node,
        critical=True
    )
    di_time_text = cura.di_timeframe or ""
    await evaluator.verify(
        claim=f"The DI card must/can be completed within the following timeframe: '{di_time_text}'.",
        node=di_time_node,
        sources=cura.sources,
        additional_instruction="Confirm the official guidance regarding when travelers must complete the DI card prior to travel."
    )

    # Passport validity requirements
    passport_node = evaluator.add_leaf(
        id="curacao_passport_validity_requirement",
        desc="States passport validity requirements for entry (as specified by authoritative guidance).",
        parent=cura_node,
        critical=True
    )
    passport_text = cura.passport_validity or ""
    await evaluator.verify(
        claim=f"Passport validity requirement for U.S. travelers entering Curacao is: '{passport_text}'.",
        node=passport_node,
        sources=cura.sources,
        additional_instruction="Confirm on official sources (government/immigration/tourism) the passport validity requirement."
    )

    # Source presence
    evaluator.add_custom_node(
        result=bool(cura.sources),
        id="curacao_source_url",
        desc="Provides reliable reference URL(s) supporting the stated Curacao entry requirements.",
        parent=cura_node,
        critical=True
    )


async def verify_bangor(evaluator: Evaluator, parent_node, bangor: BangorIncidentExtraction) -> None:
    bangor_node = evaluator.add_parallel(
        id="bangor_airport_incident",
        desc="Provide details of the Bangor International Airport plane crash in January 2026 (exact date, fatalities, aircraft type), supported by sources.",
        parent=parent_node,
        critical=True
    )

    # Incident exact date (and month-year check implicitly)
    date_node = evaluator.add_leaf(
        id="bangor_incident_exact_date",
        desc="Provides the exact date of the incident and it is in January 2026.",
        parent=bangor_node,
        critical=True
    )
    date_text = bangor.incident_date or ""
    await evaluator.verify(
        claim=f"The Bangor International Airport incident occurred on '{date_text}', which is in January 2026.",
        node=date_node,
        sources=bangor.sources,
        additional_instruction="Confirm the exact incident date and ensure it falls within January 2026."
    )

    # Fatalities
    fat_node = evaluator.add_leaf(
        id="bangor_fatalities",
        desc="Provides the total number of fatalities.",
        parent=bangor_node,
        critical=True
    )
    fat_text = bangor.fatalities or ""
    await evaluator.verify(
        claim=f"The total number of fatalities in the Bangor incident was '{fat_text}'.",
        node=fat_node,
        sources=bangor.sources,
        additional_instruction="Confirm the reported number of fatalities from authoritative news or official sources."
    )

    # Aircraft type
    ac_node = evaluator.add_leaf(
        id="bangor_aircraft_type",
        desc="Identifies the aircraft type involved.",
        parent=bangor_node,
        critical=True
    )
    ac_text = bangor.aircraft_type or ""
    await evaluator.verify(
        claim=f"The aircraft type involved in the Bangor incident was '{ac_text}'.",
        node=ac_node,
        sources=bangor.sources,
        additional_instruction="Confirm aircraft type details from authoritative reports."
    )

    # Source presence
    evaluator.add_custom_node(
        result=bool(bangor.sources),
        id="bangor_source_url",
        desc="Provides reliable reference URL(s) supporting the incident date, fatalities, and aircraft type.",
        parent=bangor_node,
        critical=True
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
    Evaluate the provided answer for hotels near Six Flags Magic Mountain (<1 mile), Avelo HVN→California service through mid-Nov 2026,
    Curacao entry requirements (DI card requirement, timeframe, passport validity), and Bangor Jan 2026 crash details. 
    All claims must be supported by cited URLs.
    """
    # Initialize evaluator with a parallel root
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

    # Concurrent extractions
    hotels_task = evaluator.extract(
        prompt=prompt_extract_hotels(),
        template_class=HotelsExtraction,
        extraction_name="hotels_near_sixflags"
    )
    avelo_task = evaluator.extract(
        prompt=prompt_extract_avelo(),
        template_class=AveloExtraction,
        extraction_name="avelo_hvn_ca_service"
    )
    cura_task = evaluator.extract(
        prompt=prompt_extract_curacao(),
        template_class=CuracaoEntryExtraction,
        extraction_name="curacao_entry_requirements"
    )
    bangor_task = evaluator.extract(
        prompt=prompt_extract_bangor(),
        template_class=BangorIncidentExtraction,
        extraction_name="bangor_incident"
    )

    hotels_ext, avelo_ext, cura_ext, bangor_ext = await asyncio.gather(
        hotels_task, avelo_task, cura_task, bangor_task
    )

    # Build verification tree and run checks
    await verify_hotels(evaluator, root, hotels_ext)
    await verify_avelo(evaluator, root, avelo_ext)
    await verify_curacao(evaluator, root, cura_ext)
    await verify_bangor(evaluator, root, bangor_ext)

    # Return structured summary
    return evaluator.get_summary()