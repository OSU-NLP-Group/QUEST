import asyncio
import logging
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy

# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "maine_airport_2026"
TASK_DESCRIPTION = (
    "Identify a commercial airport located in Maine that meets the following operational requirements: "
    "(1) has at least one runway measuring 11,000 feet or longer, "
    "(2) is currently served by all three major U.S. carriers - American Airlines, Delta Air Lines, and United Airlines, "
    "(3) has U.S. Customs and Border Protection facilities capable of handling international arrivals, "
    "(4) offers nonstop service to at least one Delta Air Lines hub city (either Atlanta or Detroit), and "
    "(5) was operational and accepting commercial flights as of February 2026."
)

# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class CarrierSources(BaseModel):
    aa_urls: List[str] = Field(default_factory=list)
    dl_urls: List[str] = Field(default_factory=list)
    ua_urls: List[str] = Field(default_factory=list)


class AirportExtraction(BaseModel):
    airport_name: Optional[str] = None
    airport_iata: Optional[str] = None

    # Evidence URLs for each criterion
    runway_sources: List[str] = Field(default_factory=list)
    location_sources: List[str] = Field(default_factory=list)
    customs_sources: List[str] = Field(default_factory=list)
    delta_hub_city: Optional[str] = None  # Expect one of: Atlanta, ATL, Detroit, DTW
    delta_hub_sources: List[str] = Field(default_factory=list)
    operational_sources: List[str] = Field(default_factory=list)

    carriers: Optional[CarrierSources] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_airport_info() -> str:
    return (
        "You must extract the single airport proposed in the answer as satisfying all conditions, along with the "
        "supporting URLs grouped by each requirement. Extract the following fields:\n"
        "1) airport_name: The full airport name as stated (e.g., 'Bangor International Airport').\n"
        "2) airport_iata: The IATA code if explicitly given (e.g., 'BGR'); otherwise null.\n"
        "3) runway_sources: An array of all URLs the answer cites to support that the airport has at least one runway "
        "   11,000 feet or longer.\n"
        "4) location_sources: An array of all URLs cited to confirm the airport is in the U.S. state of Maine.\n"
        "5) customs_sources: An array of all URLs cited to support that the airport has U.S. Customs and Border Protection "
        "   facilities capable of handling international arrivals (e.g., CBP presence, FIS facilities, port of entry).\n"
        "6) carriers: An object grouping URLs per airline:\n"
        "     - aa_urls: URLs supporting that American Airlines serves the airport.\n"
        "     - dl_urls: URLs supporting that Delta Air Lines serves the airport.\n"
        "     - ua_urls: URLs supporting that United Airlines serves the airport.\n"
        "   Only include URLs explicitly present in the answer. If a given airline is not supported by any URL, return an empty list for that airline.\n"
        "7) delta_hub_city: The specific Delta hub city that the answer claims has nonstop service from the airport, if stated. "
        "   Return a short value like 'Atlanta', 'ATL', 'Detroit', or 'DTW'. If not stated, return null.\n"
        "8) delta_hub_sources: An array of URLs supporting that there is nonstop service on Delta Air Lines to either Atlanta (ATL) or Detroit (DTW).\n"
        "9) operational_sources: An array of URLs supporting that the airport was operational and accepting commercial flights as of February 2026 "
        "   (e.g., airport or airline schedule pages, notices, or authoritative sources referencing operations around Feb 2026).\n\n"
        "Rules:\n"
        "- Extract only URLs explicitly present in the answer (including markdown links). If a URL is missing a protocol, prepend 'http://'.\n"
        "- Do not invent any URLs or details not found in the answer.\n"
        "- If a category lacks URLs in the answer, return an empty list for that category.\n"
        "- If the answer lists multiple airports, select the single main airport that is presented as meeting all criteria.\n"
    )


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def _normalize_hub_city(city: Optional[str]) -> Optional[str]:
    if not city:
        return None
    text = city.strip().lower()
    if text in {"atl", "atlanta"}:
        return "Atlanta (ATL)"
    if text in {"dtw", "detroit"}:
        return "Detroit (DTW)"
    return None


def _has_any_url(urls: Optional[List[str]]) -> bool:
    return bool(urls) and any(isinstance(u, str) and u.strip() for u in urls)


# --------------------------------------------------------------------------- #
# Verification builder                                                        #
# --------------------------------------------------------------------------- #
async def build_airport_verification(evaluator: Evaluator, root, extr: AirportExtraction) -> None:
    # Main node representing the JSON's 'Airport_Identification'
    airport_node = evaluator.add_parallel(
        id="Airport_Identification",
        desc="Identify a U.S. airport that meets all specified operational and technical criteria",
        parent=root,
        critical=True,
    )

    # Gate: Airport name must be provided
    evaluator.add_custom_node(
        result=bool(extr.airport_name and extr.airport_name.strip()),
        id="Airport_Name_Provided",
        desc="A specific airport name is provided in the answer",
        parent=airport_node,
        critical=True
    )

    airport_name = extr.airport_name or "the airport"

    # --------------------------- Runway_Length --------------------------------
    runway_node = evaluator.add_sequential(
        id="Runway_Length",
        desc="The airport has at least one runway measuring 11,000 feet or longer",
        parent=airport_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=_has_any_url(extr.runway_sources),
        id="Runway_Length_sources_provided",
        desc="Runway length evidence sources are provided",
        parent=runway_node,
        critical=True
    )
    runway_check = evaluator.add_leaf(
        id="Runway_Length_supported",
        desc="Runway length requirement is supported by sources",
        parent=runway_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The airport {airport_name} has at least one runway that is 11,000 feet or longer.",
        node=runway_check,
        sources=extr.runway_sources,
        additional_instruction="Verify using authoritative airport data (e.g., FAA/airport fact sheets, official airport pages, or trusted references). "
                               "Treat any runway listed at 11,000 ft or greater as satisfying the requirement. Minor formatting or unit differences are acceptable."
    )

    # ------------------------ Geographic_Location ------------------------------
    location_node = evaluator.add_sequential(
        id="Geographic_Location",
        desc="The airport is located in Maine",
        parent=airport_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=_has_any_url(extr.location_sources),
        id="Geographic_Location_sources_provided",
        desc="Location evidence sources are provided",
        parent=location_node,
        critical=True
    )
    location_check = evaluator.add_leaf(
        id="Geographic_Location_supported",
        desc="Maine location is supported by sources",
        parent=location_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The airport {airport_name} is located in the U.S. state of Maine.",
        node=location_check,
        sources=extr.location_sources,
        additional_instruction="Look for explicit mention that the airport is in Maine (e.g., city, state, address)."
    )

    # ---------------------- Major_Carrier_Service ------------------------------
    carriers_node = evaluator.add_parallel(
        id="Major_Carrier_Service",
        desc="The airport is served by American Airlines, Delta Air Lines, and United Airlines",
        parent=airport_node,
        critical=True
    )

    # Set up per-airline subsections (each sequential: sources -> verify)
    carriers = extr.carriers or CarrierSources()

    # American Airlines
    aa_node = evaluator.add_sequential(
        id="Major_Carrier_Service_AA",
        desc="American Airlines service verification",
        parent=carriers_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=_has_any_url(carriers.aa_urls),
        id="AA_Service_sources_provided",
        desc="American Airlines service sources are provided",
        parent=aa_node,
        critical=True
    )
    aa_leaf = evaluator.add_leaf(
        id="AA_Service_supported",
        desc="American Airlines serves the airport",
        parent=aa_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"American Airlines provides scheduled service to or from {airport_name}.",
        node=aa_leaf,
        sources=carriers.aa_urls,
        additional_instruction="Confirm that AA lists the airport as a destination or has scheduled flights to/from the airport. "
                               "Airline destination pages, schedules, or airport route maps are acceptable."
    )

    # Delta Air Lines
    dl_node = evaluator.add_sequential(
        id="Major_Carrier_Service_DL",
        desc="Delta Air Lines service verification",
        parent=carriers_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=_has_any_url(carriers.dl_urls),
        id="DL_Service_sources_provided",
        desc="Delta Air Lines service sources are provided",
        parent=dl_node,
        critical=True
    )
    dl_leaf = evaluator.add_leaf(
        id="DL_Service_supported",
        desc="Delta Air Lines serves the airport",
        parent=dl_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"Delta Air Lines provides scheduled service to or from {airport_name}.",
        node=dl_leaf,
        sources=carriers.dl_urls,
        additional_instruction="Confirm that Delta lists the airport as a destination or shows scheduled flights to/from the airport."
    )

    # United Airlines
    ua_node = evaluator.add_sequential(
        id="Major_Carrier_Service_UA",
        desc="United Airlines service verification",
        parent=carriers_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=_has_any_url(carriers.ua_urls),
        id="UA_Service_sources_provided",
        desc="United Airlines service sources are provided",
        parent=ua_node,
        critical=True
    )
    ua_leaf = evaluator.add_leaf(
        id="UA_Service_supported",
        desc="United Airlines serves the airport",
        parent=ua_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"United Airlines provides scheduled service to or from {airport_name}.",
        node=ua_leaf,
        sources=carriers.ua_urls,
        additional_instruction="Confirm that United lists the airport as a destination or shows scheduled flights to/from the airport."
    )

    # ------------------------- Customs_Facility --------------------------------
    customs_node = evaluator.add_sequential(
        id="Customs_Facility",
        desc="The airport has U.S. Customs and Border Protection facilities capable of handling international arrivals",
        parent=airport_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=_has_any_url(extr.customs_sources),
        id="Customs_Facility_sources_provided",
        desc="Customs facility sources are provided",
        parent=customs_node,
        critical=True
    )
    customs_leaf = evaluator.add_leaf(
        id="Customs_Facility_supported",
        desc="CBP international arrivals capability is supported by sources",
        parent=customs_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The airport {airport_name} has U.S. Customs and Border Protection facilities capable of handling international arrivals.",
        node=customs_leaf,
        sources=extr.customs_sources,
        additional_instruction="Look for explicit mention of CBP presence, Federal Inspection Services (FIS), or international arrivals processing at the airport."
    )

    # --------------------- Delta_Hub_Connectivity ------------------------------
    delta_node = evaluator.add_sequential(
        id="Delta_Hub_Connectivity",
        desc="The airport offers nonstop service to at least one Delta Air Lines hub city (Atlanta or Detroit)",
        parent=airport_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=_has_any_url(extr.delta_hub_sources),
        id="Delta_Hub_Connectivity_sources_provided",
        desc="Delta hub nonstop connectivity sources are provided",
        parent=delta_node,
        critical=True
    )
    hub_city_norm = _normalize_hub_city(extr.delta_hub_city)
    hub_phrase = hub_city_norm if hub_city_norm else "Atlanta (ATL) or Detroit (DTW)"
    delta_leaf = evaluator.add_leaf(
        id="Delta_Hub_Connectivity_supported",
        desc="Delta hub nonstop connectivity is supported by sources",
        parent=delta_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The airport {airport_name} offers nonstop service on Delta Air Lines to {hub_phrase}.",
        node=delta_leaf,
        sources=extr.delta_hub_sources,
        additional_instruction="Verify that there is nonstop service (no connections) on Delta Air Lines between the airport and either Atlanta (ATL) or Detroit (DTW). "
                               "Airline schedules, route maps, or airport flight listings are acceptable."
    )

    # ------------------------- Operational_Status ------------------------------
    ops_node = evaluator.add_sequential(
        id="Operational_Status",
        desc="The airport is currently operational and accepting commercial flights as of February 2026",
        parent=airport_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=_has_any_url(extr.operational_sources),
        id="Operational_Status_sources_provided",
        desc="Operational status sources are provided",
        parent=ops_node,
        critical=True
    )
    ops_leaf = evaluator.add_leaf(
        id="Operational_Status_supported",
        desc="Operational as of Feb 2026 is supported by sources",
        parent=ops_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"As of February 2026, the airport {airport_name} was operational and accepting commercial passenger flights.",
        node=ops_leaf,
        sources=extr.operational_sources,
        additional_instruction="Prefer authoritative sources such as the airport's official site, airline schedules, NOTAMs/alerts, or reputable news/industry sources indicating normal commercial operations around February 2026."
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
    model: str = "o4-mini"
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

    # Extraction
    extracted = await evaluator.extract(
        prompt=prompt_extract_airport_info(),
        template_class=AirportExtraction,
        extraction_name="airport_candidate_and_sources"
    )

    # Build and run verification tree
    await build_airport_verification(evaluator, root, extracted)

    # Return structured evaluation summary
    return evaluator.get_summary()