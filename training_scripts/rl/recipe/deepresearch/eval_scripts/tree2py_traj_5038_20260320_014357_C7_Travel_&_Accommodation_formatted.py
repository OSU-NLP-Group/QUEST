import asyncio
import logging
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.llm_client.base_client import LLMClient


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "us_airport_layover_amenities_v1"
TASK_DESCRIPTION = (
    "You are planning a transcontinental flight within the United States and have the option to choose your layover airport. "
    "You have an 8-hour daytime layover and want to maximize your comfort and productivity. You need an airport that offers ALL of the following amenities and services:\n\n"
    "1. At least 3 dedicated mother's rooms or private nursing facilities (you're traveling with an infant)\n"
    "2. Complimentary WiFi throughout all terminal areas (you need to work remotely)\n"
    "3. A hotel located inside or directly connected to a terminal building without requiring shuttle transportation\n"
    "4. At least 10 airline or credit card lounges where you could potentially access comfortable seating and refreshments\n"
    "5. Currency exchange services (you need to exchange some foreign currency from a previous trip)\n"
    "6. Designated relief areas for your service animal\n"
    "7. A chapel, meditation room, or prayer space (you observe midday prayer)\n"
    "8. Device charging stations (your laptop and phone need charging)\n"
    "9. Multiple ATM locations\n"
    "10. At least 4 separate concourses or terminal buildings (indicating a major hub with good connectivity options)\n"
    "11. At least 10 dining establishments in post-security areas (you want meal variety)\n"
    "12. At least 5 retail shops in post-security areas\n"
    "13. The airport must serve a major US metropolitan area with a population exceeding 1 million\n\n"
    "Identify one major US airport that meets ALL of these requirements. Provide the airport's three-letter IATA code and the city it serves."
)


# --------------------------------------------------------------------------- #
# Data Models                                                                 #
# --------------------------------------------------------------------------- #
class AirportPick(BaseModel):
    airport_name: Optional[str] = None
    iata_code: Optional[str] = None
    city: Optional[str] = None
    state: Optional[str] = None


class AmenityEvidence(BaseModel):
    general_urls: List[str] = Field(default_factory=list)

    mothers_rooms_urls: List[str] = Field(default_factory=list)
    wifi_urls: List[str] = Field(default_factory=list)
    hotel_urls: List[str] = Field(default_factory=list)
    lounges_urls: List[str] = Field(default_factory=list)
    currency_exchange_urls: List[str] = Field(default_factory=list)
    pet_relief_urls: List[str] = Field(default_factory=list)
    prayer_room_urls: List[str] = Field(default_factory=list)
    charging_urls: List[str] = Field(default_factory=list)
    atms_urls: List[str] = Field(default_factory=list)
    concourses_urls: List[str] = Field(default_factory=list)
    dining_urls: List[str] = Field(default_factory=list)
    shopping_urls: List[str] = Field(default_factory=list)
    metro_pop_urls: List[str] = Field(default_factory=list)


class AirportExtraction(BaseModel):
    airport: Optional[AirportPick] = None
    evidence: AmenityEvidence = Field(default_factory=AmenityEvidence)


# --------------------------------------------------------------------------- #
# Extraction Prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_airport_and_sources() -> str:
    return """
    Your task is to extract, from the provided answer text, a single US airport the answer proposes and all URLs cited as evidence, grouped by amenity category.

    Extract the following fields:
    1) airport:
       - airport_name: The full official airport name if present (e.g., "Hartsfield–Jackson Atlanta International Airport")
       - iata_code: The 3-letter IATA code (uppercase) if present (e.g., "ATL")
       - city: The primary city the airport serves (e.g., "Atlanta")
       - state: The US state if present (e.g., "GA" or "Georgia")
    2) evidence:
       For each amenity category below, collect all URLs explicitly present in the answer that are clearly associated with that amenity. If the answer uses markdown links, extract the actual URLs.
       - mothers_rooms_urls (dedicated lactation/nursing rooms)
       - wifi_urls (complimentary Wi‑Fi)
       - hotel_urls (on‑terminal or directly connected hotel)
       - lounges_urls (airline or credit card lounges)
       - currency_exchange_urls
       - pet_relief_urls (service animal relief areas)
       - prayer_room_urls (chapel/meditation/prayer space)
       - charging_urls (device charging points/stations)
       - atms_urls (ATM locations)
       - concourses_urls (terminal/concourse structure)
       - dining_urls (post‑security dining/restaurants)
       - shopping_urls (post‑security retail/shops)
       - metro_pop_urls (evidence that the metro area served > 1,000,000 population)
       - general_urls: any additional URLs cited that may broadly support multiple amenities but are not clearly tied to a single category above.

    Rules:
    - Only extract URLs that are explicitly present in the answer text (plain, markdown, or similar). Do not invent or infer any new URLs.
    - If the answer lists multiple airports, pick the first clearly recommended one and extract it as the 'airport'.
    - If a field is not mentioned, set it to null (for strings) or an empty array (for URL lists).
    - Do not deduplicate across categories; include the same URL in multiple categories if the answer explicitly ties it to multiple amenities.
    """


# --------------------------------------------------------------------------- #
# Helper Functions                                                            #
# --------------------------------------------------------------------------- #
def airport_label(ap: Optional[AirportPick]) -> str:
    """Return a readable label for the chosen airport."""
    if not ap:
        return "the identified airport"
    parts = []
    if ap.airport_name:
        parts.append(ap.airport_name)
    if ap.iata_code:
        parts.append(f"({ap.iata_code})")
    loc = []
    if ap.city:
        loc.append(ap.city)
    if ap.state:
        loc.append(ap.state)
    if loc:
        parts.append("in " + ", ".join(loc))
    return " ".join(parts) if parts else "the identified airport"


def choose_sources(primary: List[str], fallback: List[str]) -> Optional[List[str]]:
    """Prefer amenity-specific URLs; otherwise use general URLs; return None if both empty."""
    if primary:
        return primary
    if fallback:
        return fallback
    return None


# --------------------------------------------------------------------------- #
# Verification Builder                                                        #
# --------------------------------------------------------------------------- #
async def verify_airport_requirements(evaluator: Evaluator, parent_node, extraction: AirportExtraction) -> None:
    ap = extraction.airport or AirportPick()
    ev = extraction.evidence or AmenityEvidence()
    label = airport_label(ap)

    # Root critical parallel node: all 13 must pass
    root_node = evaluator.add_parallel(
        id="Airport_Meets_All_Layover_Amenity_Requirements",
        desc="The identified US airport meets all 13 specified amenity requirements for a comfortable long layover",
        parent=parent_node,
        critical=True,
    )

    # 1) Mother's rooms >= 3 (post-security)
    node1 = evaluator.add_leaf(
        id="Has_Sufficient_Mother_Rooms",
        desc="The airport has at least 3 dedicated mother's rooms or nursing facilities located in post-security areas",
        parent=root_node,
        critical=True,
    )
    claim1 = (
        f"{label} has at least three (≥3) dedicated mother's rooms / lactation or nursing rooms located airside (post‑security) "
        f"across its terminals or concourses."
    )
    await evaluator.verify(
        claim=claim1,
        node=node1,
        sources=choose_sources(ev.mothers_rooms_urls, ev.general_urls),
        additional_instruction=(
            "Accept synonyms like 'lactation room', 'nursing pod', 'Mamava pod', or 'mother's room'. "
            "They must be inside secured/airside areas or clearly available to departing passengers after security. "
            "Count distinct rooms/pods; at least three are required."
        ),
    )

    # 2) Complimentary WiFi throughout terminals
    node2 = evaluator.add_leaf(
        id="Provides_Free_WiFi",
        desc="The airport provides free WiFi access throughout all terminal areas without time restrictions or payment requirements",
        parent=root_node,
        critical=True,
    )
    claim2 = (
        f"{label} provides complimentary Wi‑Fi throughout terminal areas without time limits or paid access requirements."
    )
    await evaluator.verify(
        claim=claim2,
        node=node2,
        sources=choose_sources(ev.wifi_urls, ev.general_urls),
        additional_instruction="Look for terms like 'free Wi‑Fi', 'complimentary Wi‑Fi', and no paid/time‑limited restrictions.",
    )

    # 3) On-terminal or directly connected hotel (no shuttle)
    node3 = evaluator.add_leaf(
        id="Has_OnTerminal_Hotel",
        desc="The airport has at least one hotel located inside a terminal building or directly connected to a terminal without requiring shuttle bus service",
        parent=root_node,
        critical=True,
    )
    claim3 = (
        f"{label} has a hotel located inside or physically connected to a terminal building, reachable without using a shuttle bus."
    )
    await evaluator.verify(
        claim=claim3,
        node=node3,
        sources=choose_sources(ev.hotel_urls, ev.general_urls),
        additional_instruction="Evidence should indicate an in‑terminal or skybridge/connector‑attached hotel; shuttles are not allowed.",
    )

    # 4) Lounges >= 10
    node4 = evaluator.add_leaf(
        id="Has_Multiple_Lounges",
        desc="The airport has at least 10 airline or credit card lounges accessible to passengers",
        parent=root_node,
        critical=True,
    )
    claim4 = f"{label} has at least ten (≥10) passenger lounges (airline- or credit‑card‑affiliated)."
    await evaluator.verify(
        claim=claim4,
        node=node4,
        sources=choose_sources(ev.lounges_urls, ev.general_urls),
        additional_instruction=(
            "Count airline lounges (e.g., Delta Sky Club, United Club, Admirals Club), credit‑card lounges (e.g., Centurion, Capital One), "
            "and partner lounges. At least 10 distinct lounges total are required."
        ),
    )

    # 5) Currency exchange services
    node5 = evaluator.add_leaf(
        id="Has_Currency_Exchange",
        desc="The airport has currency exchange services available within the terminal area",
        parent=root_node,
        critical=True,
    )
    claim5 = f"Currency exchange services are available within terminal areas at {label}."
    await evaluator.verify(
        claim=claim5,
        node=node5,
        sources=choose_sources(ev.currency_exchange_urls, ev.general_urls),
        additional_instruction="Look for on‑site currency exchange counters/kiosks in terminals or concourses.",
    )

    # 6) Pet/service animal relief areas
    node6 = evaluator.add_leaf(
        id="Has_Pet_Relief_Areas",
        desc="The airport has designated pet or service animal relief areas accessible to travelers",
        parent=root_node,
        critical=True,
    )
    claim6 = f"{label} has designated service‑animal/pet relief areas accessible to passengers."
    await evaluator.verify(
        claim=claim6,
        node=node6,
        sources=choose_sources(ev.pet_relief_urls, ev.general_urls),
        additional_instruction="Airside locations (post‑security) preferred; landside acceptable if clearly available to travelers.",
    )

    # 7) Chapel/meditation/prayer space
    node7 = evaluator.add_leaf(
        id="Has_Religious_Meditation_Space",
        desc="The airport has a chapel, meditation room, or multi-faith prayer space available for passengers",
        parent=root_node,
        critical=True,
    )
    claim7 = f"{label} provides a chapel, meditation room, or multi‑faith/prayer space for passengers."
    await evaluator.verify(
        claim=claim7,
        node=node7,
        sources=choose_sources(ev.prayer_room_urls, ev.general_urls),
        additional_instruction="Accept 'interfaith chapel', 'meditation room', 'prayer room', or similar.",
    )

    # 8) Device charging stations
    node8 = evaluator.add_leaf(
        id="Has_Charging_Stations",
        desc="The airport has electric device charging stations available for passengers throughout the terminal",
        parent=root_node,
        critical=True,
    )
    claim8 = f"Device charging stations/ports are available throughout terminal areas at {label}."
    await evaluator.verify(
        claim=claim8,
        node=node8,
        sources=choose_sources(ev.charging_urls, ev.general_urls),
        additional_instruction="Look for charging stations/counters/outlets/USB ports in gates, seating, or common areas.",
    )

    # 9) Multiple ATMs
    node9 = evaluator.add_leaf(
        id="Has_Multiple_ATMs",
        desc="The airport has multiple ATM locations distributed throughout the terminal areas",
        parent=root_node,
        critical=True,
    )
    claim9 = f"There are multiple ATM locations distributed throughout terminal areas at {label}."
    await evaluator.verify(
        claim=claim9,
        node=node9,
        sources=choose_sources(ev.atms_urls, ev.general_urls),
        additional_instruction="Evidence should indicate more than one ATM location within the airport terminals.",
    )

    # 10) At least 4 concourses or terminal buildings
    node10 = evaluator.add_leaf(
        id="Has_Multiple_Concourses",
        desc="The airport has at least 4 separate concourses or terminal buildings",
        parent=root_node,
        critical=True,
    )
    claim10 = f"{label} has at least four (≥4) distinct concourses or terminal buildings."
    await evaluator.verify(
        claim=claim10,
        node=node10,
        sources=choose_sources(ev.concourses_urls, ev.general_urls),
        additional_instruction="Accept 'Concourse A/B/C/D...' or multiple named/lettered terminals that are distinct areas.",
    )

    # 11) Dining options >= 10 in post-security
    node11 = evaluator.add_leaf(
        id="Has_Sufficient_Dining_Options",
        desc="The airport has at least 10 dining establishments (restaurants, cafes, or food courts) located in post-security areas",
        parent=root_node,
        critical=True,
    )
    claim11 = f"{label} offers at least ten (≥10) dining establishments in post‑security (airside) areas."
    await evaluator.verify(
        claim=claim11,
        node=node11,
        sources=choose_sources(ev.dining_urls, ev.general_urls),
        additional_instruction=(
            "Count restaurants, cafes, bars, fast‑casual, and food court outlets inside security (airside). "
            "If a source lists many airside venues by concourse, that suffices."
        ),
    )

    # 12) Retail shops >= 5 in post-security
    node12 = evaluator.add_leaf(
        id="Has_Sufficient_Shopping",
        desc="The airport has at least 5 retail shops or stores located in post-security areas",
        parent=root_node,
        critical=True,
    )
    claim12 = f"{label} has at least five (≥5) retail shops/stores in post‑security (airside) areas."
    await evaluator.verify(
        claim=claim12,
        node=node12,
        sources=choose_sources(ev.shopping_urls, ev.general_urls),
        additional_instruction="Accept bookstores, convenience, apparel, electronics, specialty retail, and duty‑free if applicable.",
    )

    # 13) Serves major US metro > 1M population
    node13 = evaluator.add_leaf(
        id="Serves_Major_US_City",
        desc="The airport serves a major US metropolitan area with a population greater than 1 million people",
        parent=root_node,
        critical=True,
    )
    city_part = ap.city or "its primary metropolitan area"
    claim13 = (
        f"{label} serves a major US metropolitan area ({city_part}) with a population exceeding 1,000,000."
    )
    await evaluator.verify(
        claim=claim13,
        node=node13,
        sources=choose_sources(ev.metro_pop_urls, ev.general_urls),
        additional_instruction=(
            "Verify both: (1) the airport serves the named US metro/city; (2) that metro/city's metropolitan area "
            "population exceeds 1,000,000 (MSA/CMSA acceptable). Wikipedia or official stats pages are acceptable."
        ),
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
    """
    Evaluate an answer for the US airport layover amenities task.
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

    # 1) Extract structured airport selection and amenity evidence URLs
    extraction = await evaluator.extract(
        prompt=prompt_extract_airport_and_sources(),
        template_class=AirportExtraction,
        extraction_name="airport_and_amenity_sources",
    )

    # 2) Build verification tree and run checks
    await verify_airport_requirements(evaluator, root, extraction)

    # 3) Return summary
    return evaluator.get_summary()