import asyncio
import logging
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator, AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys


# --------------------------------------------------------------------------- #
# Task constants                                                              #
# --------------------------------------------------------------------------- #
TASK_ID = "dcl_wish_nov2025_and_travel_infra"
TASK_DESCRIPTION = """
Identify the Disney Cruise Line Wish-class cruise ship that had its maiden voyage in November 2025 and meets all of the following specifications. Additionally, provide verification of specific travel infrastructure details for travelers planning to use this ship.

Cruise Ship Requirements:
- Must be a Disney Cruise Line vessel of the Wish class
- Gross tonnage of approximately 144,000 GT
- Passenger capacity of approximately 4,000 guests
- Exactly 1,256 staterooms
- Exactly 3 rotational dining restaurants
- 10 pools and water play areas on its upper decks
- Sails year-round from Port Everglades in Fort Lauderdale, Florida as its home port

Travel Infrastructure Requirements:
Provide the following verified information about airports and parking facilities:

1. Charlotte Douglas International Airport (CLT):
   - Total number of concourses
   - Total number of gates
   - Completion status and timeframe of the terminal lobby expansion project in 2025

2. JFK Airport Terminal 4:
   - Number of concourses in Terminal 4
   - Number of gates in Concourse B

3. Parking Rates:
   - Port Everglades (Fort Lauderdale): Daily rates for both standard vehicles and oversized vehicles
   - Baltimore Cruise Terminal: Nightly rate for passenger vehicles and SUVs

Provide the cruise ship's name and reference URLs supporting each piece of information.
"""


# --------------------------------------------------------------------------- #
# Data models                                                                 #
# --------------------------------------------------------------------------- #
class SourceValue(BaseModel):
    value: Optional[str] = None
    urls: List[str] = Field(default_factory=list)


class ShipExtraction(BaseModel):
    name: Optional[str] = None
    wish_class: SourceValue = SourceValue()
    maiden_voyage: SourceValue = SourceValue()
    gross_tonnage: SourceValue = SourceValue()
    capacity: SourceValue = SourceValue()
    staterooms: SourceValue = SourceValue()
    rotational_dining: SourceValue = SourceValue()
    pools_play_areas: SourceValue = SourceValue()
    home_port_year_round: SourceValue = SourceValue()


class CLTInfo(BaseModel):
    total_concourses: SourceValue = SourceValue()
    total_gates: SourceValue = SourceValue()
    lobby_expansion_2025: SourceValue = SourceValue()


class JFKT4Info(BaseModel):
    num_concourses: SourceValue = SourceValue()
    concourse_b_gates: SourceValue = SourceValue()


class ParkingRatesInfo(BaseModel):
    port_everglades_standard_daily: SourceValue = SourceValue()
    port_everglades_oversize_daily: SourceValue = SourceValue()
    baltimore_nightly_passenger_suv: SourceValue = SourceValue()


class TravelInfrastructureExtraction(BaseModel):
    clt: CLTInfo = CLTInfo()
    jfk_t4: JFKT4Info = JFKT4Info()
    parking: ParkingRatesInfo = ParkingRatesInfo()


class FullExtraction(BaseModel):
    ship: ShipExtraction = ShipExtraction()
    travel: TravelInfrastructureExtraction = TravelInfrastructureExtraction()


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_all() -> str:
    return """
Extract the following structured information from the answer text. Return exactly the JSON with the schema described; do not invent values or URLs. For any value not explicitly stated in the answer, return null for 'value'. For any citation not explicitly present as a URL in the answer, return an empty list for 'urls'.

SHIP:
- ship.name: The cruise ship name explicitly provided in the answer.
- ship.wish_class: value (free text as claimed in the answer, e.g., "Wish class" or "Disney Wish-class"), and urls (all supporting URLs cited for this claim).
- ship.maiden_voyage: value (the stated month and year of the maiden voyage, e.g., "November 2025"), and urls (supporting URLs).
- ship.gross_tonnage: value (as written in the answer, e.g., "144,000 GT" or "about 144k GT"), and urls (supporting URLs).
- ship.capacity: value (as written, e.g., "4,000" or "about 4,000 guests"), and urls (supporting URLs).
- ship.staterooms: value (as written, e.g., "1,256"), and urls (supporting URLs).
- ship.rotational_dining: value (as written, e.g., "3"), and urls (supporting URLs).
- ship.pools_play_areas: value (as written, e.g., "10"), and urls (supporting URLs).
- ship.home_port_year_round: value (as written, e.g., "year-round from Port Everglades"), and urls (supporting URLs).

TRAVEL INFRASTRUCTURE:
- travel.clt.total_concourses: value (as written in the answer), and urls (all supporting URLs).
- travel.clt.total_gates: value (as written), and urls.
- travel.clt.lobby_expansion_2025: value (as written, directly describing the completion status/timeframe in 2025; keep the phrasing from the answer), and urls.

- travel.jfk_t4.num_concourses: value (as written), and urls.
- travel.jfk_t4.concourse_b_gates: value (as written), and urls.

- travel.parking.port_everglades_standard_daily: value (the daily rate as written for standard vehicles), and urls.
- travel.parking.port_everglades_oversize_daily: value (the daily rate as written for oversized vehicles), and urls.
- travel.parking.baltimore_nightly_passenger_suv: value (the nightly rate as written for passenger vehicles/SUVs), and urls.
"""


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _has_nonempty_urls(urls: Optional[List[str]]) -> bool:
    return bool(urls) and any((u or "").strip() for u in urls or [])


def _safe(s: Optional[str]) -> str:
    return (s or "").strip()


async def _verify_with_sources(
    evaluator: Evaluator,
    *,
    id: str,
    desc: str,
    parent,
    claim: str,
    urls: List[str],
    critical: bool = True,
    add_ins: str = "None",
    prereq_nodes: Optional[List] = None,
) -> None:
    leaf = evaluator.add_leaf(
        id=id,
        desc=desc,
        parent=parent,
        critical=critical,
    )
    await evaluator.verify(
        claim=claim,
        node=leaf,
        sources=urls,
        additional_instruction=add_ins,
        extra_prerequisites=prereq_nodes or [],
    )


def _add_url_presence_node(
    evaluator: Evaluator,
    *,
    id: str,
    desc: str,
    parent,
    urls: Optional[List[str]],
    critical: bool = True,
):
    return evaluator.add_custom_node(
        result=_has_nonempty_urls(urls),
        id=id,
        desc=desc,
        parent=parent,
        critical=critical,
    )


def _add_value_presence_node(
    evaluator: Evaluator,
    *,
    id: str,
    desc: str,
    parent,
    value: Optional[str],
    critical: bool = True,
):
    return evaluator.add_custom_node(
        result=bool(_safe(value)),
        id=id,
        desc=desc,
        parent=parent,
        critical=critical,
    )


# --------------------------------------------------------------------------- #
# Verification builders                                                       #
# --------------------------------------------------------------------------- #
async def build_cruise_ship_checks(evaluator: Evaluator, root, ship: ShipExtraction) -> None:
    cs_node = evaluator.add_parallel(
        id="Cruise_Ship",
        desc="Provide the ship name and verify it satisfies all stated cruise-ship requirements, with a supporting URL for each required specification.",
        parent=root,
        critical=True,
    )

    # Ship name provided (existence)
    evaluator.add_custom_node(
        result=bool(_safe(ship.name)),
        id="Ship_Name_Provided",
        desc="Provides the cruise ship's name.",
        parent=cs_node,
        critical=True,
    )

    ship_name = _safe(ship.name) or "the ship"

    # 1) Wish class with citation
    urls = ship.wish_class.urls
    url_check = _add_url_presence_node(
        evaluator,
        id="Ship_Is_DCL_Wish_Class_URLs_Provided",
        desc="Supporting URL(s) provided for Wish-class claim.",
        parent=cs_node,
        urls=urls,
        critical=True,
    )
    await _verify_with_sources(
        evaluator,
        id="Ship_Is_DCL_Wish_Class_With_Citation",
        desc="States the ship is a Disney Cruise Line vessel of the Wish class and provides a supporting reference URL.",
        parent=cs_node,
        claim=f"The ship '{ship_name}' is a Disney Cruise Line vessel of the Wish class.",
        urls=urls or [],
        add_ins="Treat 'Wish class', 'Wish-class', and 'Disney Wish-class' as equivalent phrasings. The source must clearly indicate the ship belongs to Disney Cruise Line's Wish class.",
        prereq_nodes=[url_check],
    )

    # 2) Maiden voyage in Nov 2025
    urls = ship.maiden_voyage.urls
    url_check = _add_url_presence_node(
        evaluator,
        id="Ship_Maiden_Voyage_Nov_2025_URLs_Provided",
        desc="Supporting URL(s) provided for maiden voyage (Nov 2025) claim.",
        parent=cs_node,
        urls=urls,
        critical=True,
    )
    await _verify_with_sources(
        evaluator,
        id="Ship_Maiden_Voyage_Nov_2025_With_Citation",
        desc="States the ship's maiden voyage occurred in November 2025 and provides a supporting reference URL.",
        parent=cs_node,
        claim=f"The ship '{ship_name}' had its maiden voyage in November 2025.",
        urls=urls or [],
        add_ins="Allow variants like 'first revenue sailing' or 'maiden season' in November 2025.",
        prereq_nodes=[url_check],
    )

    # 3) Gross tonnage approx 144,000 GT
    urls = ship.gross_tonnage.urls
    url_check = _add_url_presence_node(
        evaluator,
        id="Ship_Gross_Tonnage_URLs_Provided",
        desc="Supporting URL(s) provided for gross tonnage claim.",
        parent=cs_node,
        urls=urls,
        critical=True,
    )
    await _verify_with_sources(
        evaluator,
        id="Ship_Gross_Tonnage_Approx_144k_With_Citation",
        desc="States the ship's gross tonnage is approximately 144,000 GT and provides a supporting reference URL.",
        parent=cs_node,
        claim=f"The gross tonnage of '{ship_name}' is approximately 144,000 GT (exact '144,000 GT' is acceptable).",
        urls=urls or [],
        add_ins="Allow rounding expressions like 'about 144k GT' or 'approximately 144,000 GT'.",
        prereq_nodes=[url_check],
    )

    # 4) Capacity approx 4,000 guests
    urls = ship.capacity.urls
    url_check = _add_url_presence_node(
        evaluator,
        id="Ship_Capacity_URLs_Provided",
        desc="Supporting URL(s) provided for passenger capacity claim.",
        parent=cs_node,
        urls=urls,
        critical=True,
    )
    await _verify_with_sources(
        evaluator,
        id="Ship_Capacity_Approx_4000_With_Citation",
        desc="States the ship's passenger capacity is approximately 4,000 guests and provides a supporting reference URL.",
        parent=cs_node,
        claim=f"The passenger capacity of '{ship_name}' is approximately 4,000 guests.",
        urls=urls or [],
        add_ins="Allow rounding (e.g., 'around 4,000').",
        prereq_nodes=[url_check],
    )

    # 5) Staterooms exactly 1,256
    urls = ship.staterooms.urls
    url_check = _add_url_presence_node(
        evaluator,
        id="Ship_Staterooms_URLs_Provided",
        desc="Supporting URL(s) provided for staterooms claim.",
        parent=cs_node,
        urls=urls,
        critical=True,
    )
    await _verify_with_sources(
        evaluator,
        id="Ship_Staterooms_1256_With_Citation",
        desc="States the ship has exactly 1,256 staterooms and provides a supporting reference URL.",
        parent=cs_node,
        claim=f"'{ship_name}' has exactly 1,256 staterooms.",
        urls=urls or [],
        add_ins="This must be an exact match: 1,256 staterooms.",
        prereq_nodes=[url_check],
    )

    # 6) Rotational dining exactly 3
    urls = ship.rotational_dining.urls
    url_check = _add_url_presence_node(
        evaluator,
        id="Ship_Rotational_Dining_URLs_Provided",
        desc="Supporting URL(s) provided for rotational dining claim.",
        parent=cs_node,
        urls=urls,
        critical=True,
    )
    await _verify_with_sources(
        evaluator,
        id="Ship_Rotational_Dining_Exactly_3_With_Citation",
        desc="States the ship has exactly 3 rotational dining restaurants and provides a supporting reference URL.",
        parent=cs_node,
        claim=f"'{ship_name}' has exactly 3 rotational dining restaurants.",
        urls=urls or [],
        add_ins="Must be exactly three rotational dining venues.",
        prereq_nodes=[url_check],
    )

    # 7) 10 pools and water play areas on upper decks
    urls = ship.pools_play_areas.urls
    url_check = _add_url_presence_node(
        evaluator,
        id="Ship_Pools_Play_Areas_URLs_Provided",
        desc="Supporting URL(s) provided for pools and water play areas claim.",
        parent=cs_node,
        urls=urls,
        critical=True,
    )
    await _verify_with_sources(
        evaluator,
        id="Ship_Pools_And_Water_Play_Areas_10_With_Citation",
        desc="States the ship has 10 pools and water play areas on its upper decks and provides a supporting reference URL.",
        parent=cs_node,
        claim=f"'{ship_name}' has 10 pools and water play areas on its upper decks.",
        urls=urls or [],
        add_ins="Count should total 10 including pools and water play areas collectively on upper decks.",
        prereq_nodes=[url_check],
    )

    # 8) Home port year-round from Port Everglades
    urls = ship.home_port_year_round.urls
    url_check = _add_url_presence_node(
        evaluator,
        id="Ship_Home_Port_URLs_Provided",
        desc="Supporting URL(s) provided for home-port year-round claim.",
        parent=cs_node,
        urls=urls,
        critical=True,
    )
    await _verify_with_sources(
        evaluator,
        id="Ship_Home_Port_Port_Everglades_Year_Round_With_Citation",
        desc="States the ship sails year-round from Port Everglades (Fort Lauderdale, Florida) as its home port and provides a supporting reference URL.",
        parent=cs_node,
        claim=f"'{ship_name}' sails year-round from Port Everglades in Fort Lauderdale, Florida as its home port.",
        urls=urls or [],
        add_ins="The source should clearly indicate Port Everglades (Fort Lauderdale) as the year-round homeport for this ship.",
        prereq_nodes=[url_check],
    )


async def build_travel_infrastructure_checks(evaluator: Evaluator, root, travel: TravelInfrastructureExtraction) -> None:
    ti_node = evaluator.add_parallel(
        id="Travel_Infrastructure",
        desc="Provide the requested airport and parking details with supporting URLs (no specific values are pre-assumed).",
        parent=root,
        critical=True,
    )

    # -------- CLT --------
    clt_node = evaluator.add_parallel(
        id="CLT_Airport",
        desc="Charlotte Douglas International Airport (CLT) requested details with citations.",
        parent=ti_node,
        critical=True,
    )

    # CLT concourses
    val_check = _add_value_presence_node(
        evaluator,
        id="CLT_Total_Concourses_Value_Provided",
        desc="Provides a value for total concourses at CLT.",
        parent=clt_node,
        value=travel.clt.total_concourses.value,
        critical=True,
    )
    url_check = _add_url_presence_node(
        evaluator,
        id="CLT_Total_Concourses_URLs_Provided",
        desc="Provides supporting URL(s) for total concourses at CLT.",
        parent=clt_node,
        urls=travel.clt.total_concourses.urls,
        critical=True,
    )
    await _verify_with_sources(
        evaluator,
        id="CLT_Total_Concourses_With_Citation",
        desc="Provides the total number of concourses at CLT and a supporting reference URL.",
        parent=clt_node,
        claim=f"Charlotte Douglas International Airport (CLT) has a total of '{_safe(travel.clt.total_concourses.value)}' concourses.",
        urls=travel.clt.total_concourses.urls or [],
        add_ins="Confirm the total number of concourses at CLT as stated. Allow minor phrasing variations.",
        prereq_nodes=[val_check, url_check],
    )

    # CLT gates
    val_check = _add_value_presence_node(
        evaluator,
        id="CLT_Total_Gates_Value_Provided",
        desc="Provides a value for total gates at CLT.",
        parent=clt_node,
        value=travel.clt.total_gates.value,
        critical=True,
    )
    url_check = _add_url_presence_node(
        evaluator,
        id="CLT_Total_Gates_URLs_Provided",
        desc="Provides supporting URL(s) for total gates at CLT.",
        parent=clt_node,
        urls=travel.clt.total_gates.urls,
        critical=True,
    )
    await _verify_with_sources(
        evaluator,
        id="CLT_Total_Gates_With_Citation",
        desc="Provides the total number of gates at CLT and a supporting reference URL.",
        parent=clt_node,
        claim=f"Charlotte Douglas International Airport (CLT) has a total of '{_safe(travel.clt.total_gates.value)}' gates.",
        urls=travel.clt.total_gates.urls or [],
        add_ins="Use authoritative sources (airport or government/industry). Gates may include all concourses combined.",
        prereq_nodes=[val_check, url_check],
    )

    # CLT lobby expansion 2025 status/timeframe
    val_check = _add_value_presence_node(
        evaluator,
        id="CLT_Lobby_Expansion_2025_Value_Provided",
        desc="Provides a status/timeframe value for CLT terminal lobby expansion in 2025.",
        parent=clt_node,
        value=travel.clt.lobby_expansion_2025.value,
        critical=True,
    )
    url_check = _add_url_presence_node(
        evaluator,
        id="CLT_Lobby_Expansion_2025_URLs_Provided",
        desc="Provides supporting URL(s) for CLT terminal lobby expansion (2025) status/timeframe.",
        parent=clt_node,
        urls=travel.clt.lobby_expansion_2025.urls,
        critical=True,
    )
    await _verify_with_sources(
        evaluator,
        id="CLT_Terminal_Lobby_Expansion_2025_Status_Timeframe_With_Citation",
        desc="States the completion status and timeframe (in 2025) of the CLT terminal lobby expansion project and provides a supporting reference URL.",
        parent=clt_node,
        claim=f"The CLT terminal lobby expansion project's 2025 status/timeframe is: '{_safe(travel.clt.lobby_expansion_2025.value)}'.",
        urls=travel.clt.lobby_expansion_2025.urls or [],
        add_ins="Look for explicit mention of 2025 status/timeframe (e.g., completed in 2025, phase milestones in 2025).",
        prereq_nodes=[val_check, url_check],
    )

    # -------- JFK Terminal 4 --------
    jfk_node = evaluator.add_parallel(
        id="JFK_Terminal_4",
        desc="JFK Airport Terminal 4 requested details with citations.",
        parent=ti_node,
        critical=True,
    )

    # JFK T4 number of concourses
    val_check = _add_value_presence_node(
        evaluator,
        id="JFK_T4_Concourse_Count_Value_Provided",
        desc="Provides a value for number of concourses in JFK Terminal 4.",
        parent=jfk_node,
        value=travel.jfk_t4.num_concourses.value,
        critical=True,
    )
    url_check = _add_url_presence_node(
        evaluator,
        id="JFK_T4_Concourse_Count_URLs_Provided",
        desc="Provides supporting URL(s) for number of concourses in JFK Terminal 4.",
        parent=jfk_node,
        urls=travel.jfk_t4.num_concourses.urls,
        critical=True,
    )
    await _verify_with_sources(
        evaluator,
        id="JFK_T4_Number_Of_Concourses_With_Citation",
        desc="Provides the number of concourses in JFK Terminal 4 and a supporting reference URL.",
        parent=jfk_node,
        claim=f"JFK Terminal 4 has '{_safe(travel.jfk_t4.num_concourses.value)}' concourse(s).",
        urls=travel.jfk_t4.num_concourses.urls or [],
        add_ins="Terminal 4 typically has Concourse A and Concourse B; confirm the exact count as stated.",
        prereq_nodes=[val_check, url_check],
    )

    # JFK T4 Concourse B gates
    val_check = _add_value_presence_node(
        evaluator,
        id="JFK_T4_B_Gates_Value_Provided",
        desc="Provides a value for number of gates in JFK Terminal 4 Concourse B.",
        parent=jfk_node,
        value=travel.jfk_t4.concourse_b_gates.value,
        critical=True,
    )
    url_check = _add_url_presence_node(
        evaluator,
        id="JFK_T4_B_Gates_URLs_Provided",
        desc="Provides supporting URL(s) for number of gates in JFK Terminal 4 Concourse B.",
        parent=jfk_node,
        urls=travel.jfk_t4.concourse_b_gates.urls,
        critical=True,
    )
    await _verify_with_sources(
        evaluator,
        id="JFK_T4_Concourse_B_Number_Of_Gates_With_Citation",
        desc="Provides the number of gates in JFK Terminal 4 Concourse B and a supporting reference URL.",
        parent=jfk_node,
        claim=f"Concourse B at JFK Terminal 4 has '{_safe(travel.jfk_t4.concourse_b_gates.value)}' gates.",
        urls=travel.jfk_t4.concourse_b_gates.urls or [],
        add_ins="Focus on Concourse B only (not combined terminal totals).",
        prereq_nodes=[val_check, url_check],
    )

    # -------- Parking Rates --------
    pr_node = evaluator.add_parallel(
        id="Parking_Rates",
        desc="Requested parking rates with citations.",
        parent=ti_node,
        critical=True,
    )

    # Port Everglades standard daily
    val_check = _add_value_presence_node(
        evaluator,
        id="Port_Everglades_Standard_Rate_Value_Provided",
        desc="Provides a value for Port Everglades standard daily parking rate.",
        parent=pr_node,
        value=travel.parking.port_everglades_standard_daily.value,
        critical=True,
    )
    url_check = _add_url_presence_node(
        evaluator,
        id="Port_Everglades_Standard_Rate_URLs_Provided",
        desc="Provides supporting URL(s) for Port Everglades standard daily parking rate.",
        parent=pr_node,
        urls=travel.parking.port_everglades_standard_daily.urls,
        critical=True,
    )
    await _verify_with_sources(
        evaluator,
        id="Port_Everglades_Standard_Daily_Rate_With_Citation",
        desc="Provides Port Everglades daily parking rate for standard vehicles and a supporting reference URL.",
        parent=pr_node,
        claim=f"The daily parking rate for standard vehicles at Port Everglades is '{_safe(travel.parking.port_everglades_standard_daily.value)}'.",
        urls=travel.parking.port_everglades_standard_daily.urls or [],
        add_ins="Accept official port pages and cruise terminal operator pages. Rate may be 'per day' or 'per 24 hours'; taxes/fees can be excluded.",
        prereq_nodes=[val_check, url_check],
    )

    # Port Everglades oversized daily
    val_check = _add_value_presence_node(
        evaluator,
        id="Port_Everglades_Oversize_Rate_Value_Provided",
        desc="Provides a value for Port Everglades oversized daily parking rate.",
        parent=pr_node,
        value=travel.parking.port_everglades_oversize_daily.value,
        critical=True,
    )
    url_check = _add_url_presence_node(
        evaluator,
        id="Port_Everglades_Oversize_Rate_URLs_Provided",
        desc="Provides supporting URL(s) for Port Everglades oversized daily parking rate.",
        parent=pr_node,
        urls=travel.parking.port_everglades_oversize_daily.urls,
        critical=True,
    )
    await _verify_with_sources(
        evaluator,
        id="Port_Everglades_Oversized_Daily_Rate_With_Citation",
        desc="Provides Port Everglades daily parking rate for oversized vehicles and a supporting reference URL.",
        parent=pr_node,
        claim=f"The daily parking rate for oversized vehicles at Port Everglades is '{_safe(travel.parking.port_everglades_oversize_daily.value)}'.",
        urls=travel.parking.port_everglades_oversize_daily.urls or [],
        add_ins="Oversized typically includes RVs, large vans, buses; confirm the rate applies to oversized vehicles.",
        prereq_nodes=[val_check, url_check],
    )

    # Baltimore nightly rate for passenger vehicles/SUVs
    val_check = _add_value_presence_node(
        evaluator,
        id="Baltimore_Nightly_Rate_Value_Provided",
        desc="Provides a value for Baltimore Cruise Terminal nightly rate (passenger vehicles/SUVs).",
        parent=pr_node,
        value=travel.parking.baltimore_nightly_passenger_suv.value,
        critical=True,
    )
    url_check = _add_url_presence_node(
        evaluator,
        id="Baltimore_Nightly_Rate_URLs_Provided",
        desc="Provides supporting URL(s) for Baltimore Cruise Terminal nightly rate (passenger vehicles/SUVs).",
        parent=pr_node,
        urls=travel.parking.baltimore_nightly_passenger_suv.urls,
        critical=True,
    )
    await _verify_with_sources(
        evaluator,
        id="Baltimore_Cruise_Terminal_Nightly_Rate_With_Citation",
        desc="Provides Baltimore Cruise Terminal nightly parking rate for passenger vehicles and SUVs and a supporting reference URL.",
        parent=pr_node,
        claim=f"The nightly parking rate for passenger vehicles and SUVs at the Baltimore Cruise Terminal is '{_safe(travel.parking.baltimore_nightly_passenger_suv.value)}'.",
        urls=travel.parking.baltimore_nightly_passenger_suv.urls or [],
        add_ins="Prefer official Port of Baltimore / Maryland Port Administration or terminal operator pages.",
        prereq_nodes=[val_check, url_check],
    )


# --------------------------------------------------------------------------- #
# Main evaluation function                                                    #
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
        default_model=model,
    )

    # Extraction
    extracted = await evaluator.extract(
        prompt=prompt_extract_all(),
        template_class=FullExtraction,
        extraction_name="full_extraction",
    )

    # Build verification tree
    await build_cruise_ship_checks(evaluator, root, extracted.ship)
    await build_travel_infrastructure_checks(evaluator, root, extracted.travel)

    # Return summary
    return evaluator.get_summary()