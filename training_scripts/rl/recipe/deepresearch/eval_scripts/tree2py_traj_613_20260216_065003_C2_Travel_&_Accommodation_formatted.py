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
TASK_ID = "celebrity_apex_eclipse_aug12_2026"
TASK_DESCRIPTION = (
    "The Celebrity Apex is sailing a Mediterranean cruise specifically designed for viewing the August 12, 2026 total solar eclipse, "
    "departing from Southampton, England on August 1, 2026. For passengers planning to witness the eclipse from this cruise, identify: "
    "(1) The Spanish port city where the Celebrity Apex is scheduled to dock on August 12, 2026 (the day of the total solar eclipse), "
    "(2) The scheduled port call hours (arrival and departure times) at that port on August 12, and "
    "(3) The approximate local time when totality begins at that port city, and the duration of totality in minutes and seconds."
)

# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class CruiseEclipseExtraction(BaseModel):
    # Cruise identity and constraints
    operator: Optional[str] = None
    departure_port: Optional[str] = None
    departure_date: Optional[str] = None
    return_port: Optional[str] = None
    return_date: Optional[str] = None
    duration_nights: Optional[str] = None
    marketed_as_eclipse: Optional[str] = None

    # Ship identity & specs
    ship_name: Optional[str] = None
    ship_inauguration_date: Optional[str] = None
    ship_occupancy: Optional[str] = None

    # Aug 12 port & hours
    aug12_port_city: Optional[str] = None
    aug12_arrival_time: Optional[str] = None
    aug12_departure_time: Optional[str] = None

    # Eclipse timing at port city
    totality_start_local: Optional[str] = None
    totality_duration: Optional[str] = None

    # Sources (URLs explicitly provided in the answer)
    cruise_source_urls: List[str] = Field(default_factory=list)
    eclipse_source_urls: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_main() -> str:
    return """
Extract the following items exactly as stated in the answer text (do not infer). Use null for any missing item.

CRUISE IDENTITY & CONSTRAINTS
- operator: the cruise line/operator for this voyage (e.g., "Celebrity Cruises").
- departure_port: the embarkation port city and country for this voyage (e.g., "Southampton, England").
- departure_date: the embarkation date (e.g., "August 1, 2026" or "2026-08-01").
- return_port: the final disembarkation port for this voyage (e.g., "Southampton, England"), if mentioned.
- return_date: the final disembarkation date (e.g., "August 15, 2026" or "2026-08-15"), if mentioned.
- duration_nights: the cruise length in nights as presented in the answer (e.g., "14 nights" or "14-night").
- marketed_as_eclipse: the wording used in the answer to indicate it's marketed as a solar eclipse cruise for the Aug 12, 2026 total solar eclipse (e.g., "eclipse cruise"), if present; otherwise null.

SHIP IDENTITY & SPECS
- ship_name: the ship name as stated (e.g., "Celebrity Apex").
- ship_inauguration_date: the inauguration/launch/maiden voyage date as stated (e.g., "April 5, 2020").
- ship_occupancy: the occupancy/guest capacity as stated (e.g., "2,910" or "2910").

AUG 12 PORT & HOURS
- aug12_port_city: the Spanish port city where the ship is scheduled on August 12, 2026 (e.g., "La Coruña", "A Coruña", "La Coruna").
- aug12_arrival_time: the scheduled arrival time at that port on Aug 12 (as written, e.g., "8:00 AM", "08:00", "8am").
- aug12_departure_time: the scheduled departure time from that port on Aug 12 (as written, e.g., "4:00 PM", "16:00", "4pm").

ECLIPSE TIMING AT PORT CITY (LOCAL)
- totality_start_local: the approximate local time when totality begins at that city (as written, e.g., "8:27 PM CEST", "20:27").
- totality_duration: the length of totality (as written, e.g., "1 minute 16 seconds", "76 seconds").

URL SOURCES
- cruise_source_urls: array of all URLs in the answer that support the specific cruise identity/itinerary/port/times/ship details. Include itinerary pages, official line pages, or cruise news pages if provided. Only include URLs explicitly present in the answer.
- eclipse_source_urls: array of all URLs in the answer that support the eclipse totality start time and duration at the Aug 12 port city. Only include URLs explicitly present in the answer.

Return a single JSON object with these fields.
    """.strip()


# --------------------------------------------------------------------------- #
# Verification helpers                                                        #
# --------------------------------------------------------------------------- #
async def verify_cruise_identity(
    evaluator: Evaluator,
    parent_node,
    data: CruiseEclipseExtraction,
):
    """
    Build and verify the 'cruise_identity_constraints' subtree.
    Returns the 'cruise_sources_present' guard node for reuse as a prerequisite.
    """
    constraints_node = evaluator.add_parallel(
        id="cruise_identity_constraints",
        desc="Cruise identity matches the provided constraints (operator, ship, dates, duration, marketing, ship specs).",
        parent=parent_node,
        critical=True,
    )

    # Guard: ensure cruise sources exist (treat missing sources as a critical quality issue)
    cruise_sources_present = evaluator.add_custom_node(
        result=(len(data.cruise_source_urls) > 0),
        id="cruise_sources_present",
        desc="Cruise sources (itinerary/official) are provided in the answer.",
        parent=constraints_node,
        critical=True,
    )

    # Operator check
    operator_node = evaluator.add_leaf(
        id="operator_celebrity_cruises",
        desc="Cruise is operated by Celebrity Cruises.",
        parent=constraints_node,
        critical=True,
    )
    operator_claim = f"The cruise operator is '{data.operator}' for this voyage."
    await evaluator.verify(
        claim=operator_claim,
        node=operator_node,
        sources=data.cruise_source_urls,
        extra_prerequisites=[cruise_sources_present],
        additional_instruction="Verify on the cited itinerary/official page that the operator/brand for this voyage is Celebrity Cruises (minor naming variants like 'Celebrity' are acceptable).",
    )

    # Departure port & date check
    depart_node = evaluator.add_leaf(
        id="departure_southampton_aug1_2026",
        desc="Cruise departs from Southampton, England on August 1, 2026.",
        parent=constraints_node,
        critical=True,
    )
    depart_claim = (
        f"The cruise departs from {data.departure_port} on {data.departure_date}."
    )
    await evaluator.verify(
        claim=depart_claim,
        node=depart_node,
        sources=data.cruise_source_urls,
        extra_prerequisites=[cruise_sources_present],
        additional_instruction="Confirm the embarkation details on the itinerary page. Expect 'Southampton, England' on August 1, 2026 (date format variations are acceptable).",
    )

    # Duration and return-to-Southampton check
    duration_node = evaluator.add_leaf(
        id="duration_and_return",
        desc="Cruise is 14 nights and returns to Southampton on August 15, 2026.",
        parent=constraints_node,
        critical=True,
    )
    duration_claim = (
        f"The cruise duration is {data.duration_nights} and it returns to {data.return_port} on {data.return_date}."
    )
    await evaluator.verify(
        claim=duration_claim,
        node=duration_node,
        sources=data.cruise_source_urls,
        extra_prerequisites=[cruise_sources_present],
        additional_instruction="Verify the overall length (14 nights) and that the itinerary shows return to Southampton on Aug 15, 2026. Accept minor formatting differences.",
    )

    # Marketed as eclipse cruise check
    marketed_node = evaluator.add_leaf(
        id="marketed_as_eclipse_cruise",
        desc="Cruise is specifically marketed as a solar eclipse cruise for the August 12, 2026 total solar eclipse.",
        parent=constraints_node,
        critical=True,
    )
    marketed_phrase = data.marketed_as_eclipse if data.marketed_as_eclipse else ""
    marketed_claim = (
        "This voyage is explicitly marketed as a solar eclipse cruise for the August 12, 2026 total solar eclipse."
    )
    await evaluator.verify(
        claim=marketed_claim,
        node=marketed_node,
        sources=data.cruise_source_urls,
        extra_prerequisites=[cruise_sources_present],
        additional_instruction=f"Check the itinerary/marketing language for eclipse-related phrasing. The answer's phrasing was: '{marketed_phrase}'. Minor wording variations are fine as long as the eclipse focus is clear.",
    )

    # Ship identity & specs
    ship_specs_node = evaluator.add_parallel(
        id="ship_identity_and_specs",
        desc="Ship constraint is satisfied (Celebrity Apex and the provided inauguration/occupancy details).",
        parent=constraints_node,
        critical=True,
    )

    # Ship name
    ship_name_node = evaluator.add_leaf(
        id="ship_is_celebrity_apex",
        desc="The ship is Celebrity Apex.",
        parent=ship_specs_node,
        critical=True,
    )
    ship_name_claim = f"The ship for this voyage is '{data.ship_name}'."
    await evaluator.verify(
        claim=ship_name_claim,
        node=ship_name_node,
        sources=data.cruise_source_urls,
        extra_prerequisites=[cruise_sources_present],
        additional_instruction="Confirm that the itinerary identifies the vessel as 'Celebrity Apex'. Minor casing or prefix differences are acceptable.",
    )

    # Inauguration date
    ship_inaug_node = evaluator.add_leaf(
        id="ship_inaugurated_apr_5_2020",
        desc="Celebrity Apex inauguration date is April 5, 2020.",
        parent=ship_specs_node,
        critical=True,
    )
    ship_inaug_claim = f"Celebrity Apex inauguration/maiden voyage date is {data.ship_inauguration_date}."
    await evaluator.verify(
        claim=ship_inaug_claim,
        node=ship_inaug_node,
        sources=data.cruise_source_urls,
        extra_prerequisites=[cruise_sources_present],
        additional_instruction="Accept equivalent phrasing like 'Maiden Voyage' and date format variants. The expected date is April 5, 2020.",
    )

    # Occupancy
    ship_occ_node = evaluator.add_leaf(
        id="ship_occupancy_2910",
        desc="Celebrity Apex occupancy is 2,910 passengers.",
        parent=ship_specs_node,
        critical=True,
    )
    ship_occ_claim = f"Celebrity Apex occupancy/guest capacity is {data.ship_occupancy} passengers."
    await evaluator.verify(
        claim=ship_occ_claim,
        node=ship_occ_node,
        sources=data.cruise_source_urls,
        extra_prerequisites=[cruise_sources_present],
        additional_instruction="Accept synonyms like 'guest capacity' or 'double occupancy'. The expected figure is 2,910.",
    )

    # Return the guard node to reuse in other groups
    return cruise_sources_present


async def verify_aug12_details(
    evaluator: Evaluator,
    parent_node,
    data: CruiseEclipseExtraction,
    cruise_sources_guard,  # prerequisite from cruise identity group
):
    """
    Build and verify the 'aug12_port_and_eclipse_details' subtree.
    """
    details_node = evaluator.add_parallel(
        id="aug12_port_and_eclipse_details",
        desc="Provide the requested Aug 12, 2026 port call and eclipse totality timing/duration details.",
        parent=parent_node,
        critical=True,
    )

    # Port city on Aug 12
    port_city_node = evaluator.add_leaf(
        id="port_city_aug12",
        desc="Spanish port city on Aug 12, 2026 is La Coruña (A Coruña), Spain.",
        parent=details_node,
        critical=True,
    )
    port_city_claim = (
        f"On August 12, 2026, the ship is scheduled to be in port at {data.aug12_port_city}, Spain."
    )
    await evaluator.verify(
        claim=port_city_claim,
        node=port_city_node,
        sources=data.cruise_source_urls,
        extra_prerequisites=[cruise_sources_guard],
        additional_instruction="Confirm the Aug 12 port call city on the itinerary. Allow variants of the name A Coruña / La Coruña / La Coruna.",
    )

    # Port call hours (arrival and departure)
    hours_node = evaluator.add_parallel(
        id="port_call_hours_aug12",
        desc="Provide the scheduled port call hours (arrival and departure times) at the Aug 12 port.",
        parent=details_node,
        critical=True,
    )

    arrival_node = evaluator.add_leaf(
        id="arrival_time",
        desc="Arrival time at La Coruña on Aug 12, 2026 is 8:00 AM (or equivalent 08:00).",
        parent=hours_node,
        critical=True,
    )
    arrival_claim = (
        f"The scheduled arrival time at {data.aug12_port_city} on August 12, 2026 is {data.aug12_arrival_time} (local time)."
    )
    await evaluator.verify(
        claim=arrival_claim,
        node=arrival_node,
        sources=data.cruise_source_urls,
        extra_prerequisites=[cruise_sources_guard],
        additional_instruction="Verify the itinerary's listed arrival time for Aug 12 at the stated port. Accept formatting like '08:00', '8:00 AM', or '8am'.",
    )

    departure_node = evaluator.add_leaf(
        id="departure_time",
        desc="Departure time from La Coruña on Aug 12, 2026 is 4:00 PM (or equivalent 16:00).",
        parent=hours_node,
        critical=True,
    )
    departure_claim = (
        f"The scheduled departure time from {data.aug12_port_city} on August 12, 2026 is {data.aug12_departure_time} (local time)."
    )
    await evaluator.verify(
        claim=departure_claim,
        node=departure_node,
        sources=data.cruise_source_urls,
        extra_prerequisites=[cruise_sources_guard],
        additional_instruction="Verify the itinerary's listed departure time for Aug 12 at the stated port. Accept '16:00', '4:00 PM', or similar.",
    )

    # Eclipse timing & duration (use eclipse sources)
    eclipse_node = evaluator.add_parallel(
        id="eclipse_totality_timing_and_duration",
        desc="Provide the approximate local totality start time and totality duration at the Aug 12 port city.",
        parent=details_node,
        critical=True,
    )

    # Guard for eclipse sources
    eclipse_sources_present = evaluator.add_custom_node(
        result=(len(data.eclipse_source_urls) > 0),
        id="eclipse_sources_present",
        desc="Eclipse timing sources are provided in the answer.",
        parent=eclipse_node,
        critical=True,
    )

    totality_start_node = evaluator.add_leaf(
        id="totality_start_time",
        desc="Totality begins at 8:27 PM CEST (or equivalent 20:27 CEST) on Aug 12, 2026 at La Coruña.",
        parent=eclipse_node,
        critical=True,
    )
    totality_start_claim = (
        f"At {data.aug12_port_city}, Spain on August 12, 2026, totality begins at approximately {data.totality_start_local} (local time)."
    )
    await evaluator.verify(
        claim=totality_start_claim,
        node=totality_start_node,
        sources=data.eclipse_source_urls,
        extra_prerequisites=[eclipse_sources_present],
        additional_instruction=(
            "Confirm the local totality start time for the stated city on Aug 12, 2026. "
            "Allow approximate equivalence (e.g., 20:27 ≈ 8:27 PM), minor rounding, and timezone annotations (e.g., CEST). "
            "A tolerance of ±2 minutes is acceptable."
        ),
    )

    totality_duration_node = evaluator.add_leaf(
        id="totality_duration",
        desc="Totality duration at La Coruña is 1 minute 16 seconds (or equivalent 76 seconds).",
        parent=eclipse_node,
        critical=True,
    )
    totality_duration_claim = (
        f"The totality duration at {data.aug12_port_city}, Spain is approximately {data.totality_duration}."
    )
    await evaluator.verify(
        claim=totality_duration_claim,
        node=totality_duration_node,
        sources=data.eclipse_source_urls,
        extra_prerequisites=[eclipse_sources_present],
        additional_instruction=(
            "Confirm the total duration of totality for the stated city. "
            "Allow equivalent expressions (e.g., '1m 16s' ≈ '76 seconds'). "
            "A tolerance of ±10 seconds is acceptable."
        ),
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
    Evaluate an answer for the Celebrity Apex Aug 12, 2026 eclipse cruise task.
    """
    # Initialize evaluator (root is non-critical by framework design; children carry criticality)
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

    # Extract structured information from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_main(),
        template_class=CruiseEclipseExtraction,
        extraction_name="extracted_cruise_eclipse_info",
    )

    # Add ground-truth expectations (for transparency in the summary; not used for direct auto-judgment)
    evaluator.add_ground_truth(
        {
            "expected_operator": "Celebrity Cruises",
            "expected_departure": {"port": "Southampton, England", "date": "August 1, 2026"},
            "expected_duration_return": {"duration_nights": "14 nights", "return_port": "Southampton, England", "return_date": "August 15, 2026"},
            "expected_ship": {"name": "Celebrity Apex", "inauguration_date": "April 5, 2020", "occupancy": "2,910"},
            "expected_aug12": {
                "city": "A Coruña / La Coruña, Spain",
                "arrival_time": "08:00 (8:00 AM) local",
                "departure_time": "16:00 (4:00 PM) local",
                "totality_start": "20:27 CEST (~8:27 PM)",
                "totality_duration": "1 minute 16 seconds (~76 seconds)",
            },
        }
    )

    # Build and verify subtrees
    cruise_sources_guard = await verify_cruise_identity(evaluator, root, extracted)
    await verify_aug12_details(evaluator, root, extracted, cruise_sources_guard)

    # Return evaluation summary
    return evaluator.get_summary()