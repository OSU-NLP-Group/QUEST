import asyncio
import logging
from typing import Optional, List, Dict, Any, Union

from pydantic import BaseModel, Field
from obj_task_eval.llm_client.base_client import LLMClient

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "transatlantic_azores_2026"
TASK_DESCRIPTION = (
    "A traveler is planning a transatlantic cruise in 2026 and has specific requirements: "
    "the cruise must depart from Fort Lauderdale, Florida in April 2026, arrive in Barcelona, Spain, "
    "and include a stop at Ponta Delgada in the Azores islands. Identify the cruise that meets these "
    "requirements and provide the following information: 1. The name of the cruise ship, 2. The exact "
    "departure date, 3. The exact date when the ship stops at Ponta Delgada, Azores, 4. The name of the "
    "cruise terminal at Ponta Delgada, 5. The technical specifications of the Ponta Delgada cruise "
    "terminal's berth (length and depth alongside). Provide reference URLs for all information."
)

GROUND_TRUTH = {
    "ship_name": "Oosterdam",
    "departure_month_year": "April 2026",
    "departure_date": "April 8, 2026",
    "arrival_city": "Barcelona, Spain",
    "ponta_delgada_stop_date": "April 16, 2026",
    "terminal_name": "Portas do Mar",
    "berth_length_m": "380 meters",
    "berth_depth_m": "11 meters",
}


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class CruiseExtraction(BaseModel):
    # Core cruise identification and itinerary fields (strings preferred for robustness)
    ship_name: Optional[str] = None
    departure_port: Optional[str] = None  # e.g., "Port Everglades"
    departure_city_state: Optional[str] = None  # e.g., "Fort Lauderdale, Florida"
    departure_month_year: Optional[str] = None  # e.g., "April 2026"
    departure_date: Optional[str] = None  # e.g., "April 8, 2026"
    arrival_city: Optional[str] = None  # e.g., "Barcelona, Spain"
    arrival_port: Optional[str] = None  # optional
    ponta_delgada_stop_date: Optional[str] = None  # e.g., "April 16, 2026"

    # Source URLs for each claim (extracted exactly from the answer text)
    ship_name_urls: List[str] = Field(default_factory=list)
    depart_port_urls: List[str] = Field(default_factory=list)
    departs_april_2026_urls: List[str] = Field(default_factory=list)
    departure_date_urls: List[str] = Field(default_factory=list)
    arrival_urls: List[str] = Field(default_factory=list)
    pd_stop_in_itinerary_urls: List[str] = Field(default_factory=list)
    pd_stop_date_urls: List[str] = Field(default_factory=list)
    itinerary_urls: List[str] = Field(default_factory=list)


class TerminalExtraction(BaseModel):
    terminal_name: Optional[str] = None  # e.g., "Portas do Mar"
    euro_project_amount: Optional[str] = None  # e.g., "€50 million"
    amenities_summary: Optional[str] = None  # text description mentioning cafes/restaurants/shops
    berth_length: Optional[str] = None  # e.g., "380 m" or "380 meters"
    berth_depth: Optional[str] = None  # e.g., "11 m" or "11 meters"

    # Source URLs for terminal-related claims
    terminal_name_urls: List[str] = Field(default_factory=list)
    euro_project_urls: List[str] = Field(default_factory=list)
    amenities_urls: List[str] = Field(default_factory=list)
    berth_length_urls: List[str] = Field(default_factory=list)
    berth_depth_urls: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_cruise() -> str:
    return """
    Extract the specific cruise information presented in the answer that is intended to meet the following constraints:
    - Departs from Fort Lauderdale, Florida (Port Everglades)
    - Departs in April 2026
    - Arrives in Barcelona, Spain
    - Includes a stop at Ponta Delgada (Azores)

    Extract the following fields as strings (do not infer; use exactly what is stated in the answer):
    - ship_name
    - departure_port (e.g., "Port Everglades")
    - departure_city_state (e.g., "Fort Lauderdale, Florida")
    - departure_month_year (e.g., "April 2026")
    - departure_date (e.g., "April 8, 2026")
    - arrival_city (e.g., "Barcelona, Spain")
    - arrival_port (if provided)
    - ponta_delgada_stop_date (e.g., "April 16, 2026")

    Also extract source URLs explicitly mentioned in the answer for each claim. Return arrays with the following names:
    - ship_name_urls: URLs that support the ship name
    - depart_port_urls: URLs that support departure from Fort Lauderdale / Port Everglades
    - departs_april_2026_urls: URLs that support the departure month/year being April 2026
    - departure_date_urls: URLs that support the exact departure date
    - arrival_urls: URLs that support arrival in Barcelona, Spain
    - pd_stop_in_itinerary_urls: URLs that support that the itinerary includes Ponta Delgada (Azores)
    - pd_stop_date_urls: URLs that support the exact date of the Ponta Delgada stop
    - itinerary_urls: general itinerary URL(s) if provided

    Rules for URLs:
    - Extract only actual URLs that appear in the answer (including within markdown links).
    - Do not invent URLs.
    - Ignore clearly malformed URLs.
    - Include the full URL with protocol; if missing, prepend http://.
    - If multiple cruises are mentioned, focus on the one presented as meeting the constraints.
    - If any item is not mentioned, set its field to null and provide an empty array for its URLs.

    Return a single JSON object with these fields.
    """


def prompt_extract_terminal() -> str:
    return """
    Extract terminal information for the Ponta Delgada (Azores) cruise stop from the answer.

    Extract the following fields exactly as stated:
    - terminal_name (e.g., "Portas do Mar")
    - euro_project_amount (e.g., "€50 million")
    - amenities_summary (text indicating the complex includes cafes, restaurants, shops, and cruise passenger amenities)
    - berth_length (e.g., "380 m" or "380 meters")
    - berth_depth (e.g., "11 m" or "11 meters")

    Also extract supporting source URLs for each:
    - terminal_name_urls
    - euro_project_urls
    - amenities_urls
    - berth_length_urls
    - berth_depth_urls

    Rules for URLs:
    - Extract only URLs explicitly present in the answer, including markdown links.
    - Do not invent URLs.
    - Include full URLs with protocol; if missing, prepend http://.
    - If a field is not mentioned, set it to null and provide an empty array for its URLs.

    Return a single JSON object with these fields.
    """


# --------------------------------------------------------------------------- #
# Helper: verify a leaf with URL sources                                      #
# --------------------------------------------------------------------------- #
async def verify_leaf_with_sources(
    evaluator: Evaluator,
    parent_node,
    node_id: str,
    desc: str,
    claim: str,
    sources: Optional[Union[str, List[str]]],
    *,
    critical: bool = True,
    additional_instruction: Optional[str] = None,
) -> bool:
    """
    Add a leaf node and verify the claim against provided URL sources.
    If no sources are provided, mark the node as failed (source-grounding required).
    """
    leaf = evaluator.add_leaf(
        id=node_id,
        desc=desc,
        parent=parent_node,
        critical=critical,
    )

    # Enforce source-grounding: fail if no valid sources
    no_sources = False
    if sources is None:
        no_sources = True
    elif isinstance(sources, str):
        if sources.strip() == "":
            no_sources = True
    elif isinstance(sources, list):
        # filter invalid empties
        clean = [s for s in sources if isinstance(s, str) and s.strip() != ""]
        if len(clean) == 0:
            no_sources = True
        else:
            sources = clean

    if no_sources:
        leaf.score = 0.0
        leaf.status = "failed"
        return False

    return await evaluator.verify(
        claim=claim,
        node=leaf,
        sources=sources,
        additional_instruction=additional_instruction or "None",
    )


# --------------------------------------------------------------------------- #
# Verification subtrees                                                       #
# --------------------------------------------------------------------------- #
async def verify_cruise_route_and_timing_constraints(
    evaluator: Evaluator,
    root_node,
    cruise: CruiseExtraction,
) -> None:
    group = evaluator.add_parallel(
        id="CruiseRouteAndTimingConstraints",
        desc="Cruise satisfies the route/time constraints (with citations)",
        parent=root_node,
        critical=True,
    )

    # 1) Departs from Fort Lauderdale, Florida (Port Everglades)
    await verify_leaf_with_sources(
        evaluator,
        group,
        "DepartsFromPortEvergladesFortLauderdale_WithURL",
        "States that the cruise departs from Fort Lauderdale, Florida (Port Everglades) and provides a supporting URL",
        "This cruise departs from Fort Lauderdale, Florida (Port Everglades).",
        sources=(cruise.depart_port_urls or cruise.itinerary_urls),
        additional_instruction=(
            "Verify the departure port is Fort Lauderdale/Port Everglades. Accept reasonable synonyms "
            "and formatting such as 'Port Everglades (Fort Lauderdale)'. The statement must be explicitly supported by the page."
        ),
    )

    # 2) Departs in April 2026
    await verify_leaf_with_sources(
        evaluator,
        group,
        "DepartsInApril2026_WithURL",
        "States that the cruise departs in April 2026 and provides a supporting URL",
        "This cruise departs in April 2026.",
        sources=(cruise.departs_april_2026_urls or cruise.departure_date_urls or cruise.itinerary_urls),
        additional_instruction=(
            "Confirm the departure month and year are April 2026. Accept reasonable date formats, e.g., "
            "'Apr 2026', 'April 2026', or specific dates in April 2026."
        ),
    )

    # 3) Arrives in Barcelona, Spain
    await verify_leaf_with_sources(
        evaluator,
        group,
        "ArrivesInBarcelona_WithURL",
        "States that the cruise arrives in Barcelona, Spain and provides a supporting URL",
        "This cruise arrives in Barcelona, Spain.",
        sources=(cruise.arrival_urls or cruise.itinerary_urls),
        additional_instruction=(
            "Verify that the itinerary shows Barcelona, Spain as the arrival/destination port. "
            "Accept clear itinerary listings that indicate Barcelona."
        ),
    )

    # 4) Includes stop at Ponta Delgada (Azores)
    await verify_leaf_with_sources(
        evaluator,
        group,
        "IncludesStopAtPontaDelgadaAzores_WithURL",
        "States that the cruise itinerary includes a stop at Ponta Delgada (Azores) and provides a supporting URL",
        "This cruise itinerary includes a stop at Ponta Delgada (Azores).",
        sources=(cruise.pd_stop_in_itinerary_urls or cruise.itinerary_urls),
        additional_instruction=(
            "The page should explicitly list Ponta Delgada (Azores) as a port of call."
        ),
    )


async def verify_exact_cruise_identity_and_dates_constraints(
    evaluator: Evaluator,
    root_node,
    cruise: CruiseExtraction,
) -> None:
    group = evaluator.add_parallel(
        id="ExactCruiseIdentityAndDatesConstraints",
        desc="Cruise matches the exact constrained identity/dates (with citations)",
        parent=root_node,
        critical=True,
    )

    # Ship name is Oosterdam
    await verify_leaf_with_sources(
        evaluator,
        group,
        "ShipNameIsOosterdam_WithURL",
        "States the ship name is Holland America’s Oosterdam and provides a supporting URL",
        "The ship for this cruise is the Oosterdam (Holland America Line).",
        sources=(cruise.ship_name_urls or cruise.itinerary_urls),
        additional_instruction=(
            "Confirm the ship name is 'Oosterdam'. Accept minor variants like 'MS Oosterdam' or 'HAL Oosterdam'."
        ),
    )

    # Exact departure date is April 8, 2026
    await verify_leaf_with_sources(
        evaluator,
        group,
        "ExactDepartureDateIsApr8_2026_WithURL",
        "States the exact departure date is April 8, 2026 and provides a supporting URL",
        "The cruise departs on April 8, 2026.",
        sources=(cruise.departure_date_urls or cruise.itinerary_urls),
        additional_instruction=(
            "Verify the specific departure date is April 8, 2026. Accept reasonable date formatting variations."
        ),
    )

    # Exact Ponta Delgada stop date is April 16, 2026
    await verify_leaf_with_sources(
        evaluator,
        group,
        "ExactPontaDelgadaStopDateIsApr16_2026_WithURL",
        "States the exact Ponta Delgada stop date is April 16, 2026 and provides a supporting URL",
        "The cruise stops at Ponta Delgada (Azores) on April 16, 2026.",
        sources=(cruise.pd_stop_date_urls or cruise.pd_stop_in_itinerary_urls or cruise.itinerary_urls),
        additional_instruction=(
            "Verify the exact date when the ship is scheduled to be at Ponta Delgada is April 16, 2026."
        ),
    )


async def verify_ponta_delgada_terminal_constraints(
    evaluator: Evaluator,
    root_node,
    terminal: TerminalExtraction,
) -> None:
    group = evaluator.add_parallel(
        id="PontaDelgadaTerminalConstraints",
        desc="Terminal-related constrained facts are provided (with citations)",
        parent=root_node,
        critical=True,
    )

    # Terminal name is Portas do Mar
    await verify_leaf_with_sources(
        evaluator,
        group,
        "TerminalNameIsPortasDoMar_WithURL",
        "States the Ponta Delgada cruise terminal is called 'Portas do Mar' (allowing stated aliases) and provides a supporting URL",
        "The Ponta Delgada cruise terminal is called 'Portas do Mar'.",
        sources=terminal.terminal_name_urls,
        additional_instruction=(
            "Accept aliases such as 'Portas do Mar Cruise Terminal' or 'Portas do Mar Complex' if clearly referring to the cruise terminal."
        ),
    )

    # Terminal built as part of a €50 million project
    await verify_leaf_with_sources(
        evaluator,
        group,
        "TerminalBuiltAsPartOf50MEuroProject_WithURL",
        "States the terminal was built as part of a €50 million project and provides a supporting URL",
        "Portas do Mar was built as part of a €50 million project.",
        sources=terminal.euro_project_urls,
        additional_instruction=(
            "Confirm the page states the project cost around €50 million (accept '50 million euros' or similar phrasing)."
        ),
    )

    # Terminal complex includes cafes, restaurants, shops, and cruise passenger amenities
    await verify_leaf_with_sources(
        evaluator,
        group,
        "TerminalComplexIncludesPassengerAmenities_WithURL",
        "States the terminal complex includes cafes, restaurants, shops, and cruise passenger amenities and provides a supporting URL",
        "The Portas do Mar complex includes cafes, restaurants, shops, and cruise passenger amenities.",
        sources=terminal.amenities_urls,
        additional_instruction=(
            "Verify that the page explicitly mentions cafes, restaurants, retail/shops, and cruise passenger amenities as part of the complex."
        ),
    )


async def verify_ponta_delgada_berth_specs_constraints(
    evaluator: Evaluator,
    root_node,
    terminal: TerminalExtraction,
) -> None:
    group = evaluator.add_parallel(
        id="PontaDelgadaBerthTechnicalSpecsConstraints",
        desc="Berth technical specifications match the constrained values (with citations)",
        parent=root_node,
        critical=True,
    )

    # Berth length is 380 meters
    await verify_leaf_with_sources(
        evaluator,
        group,
        "BerthLengthIs380m_WithURL",
        "States the dedicated cruise berth length is 380 meters and provides a supporting URL",
        "The dedicated cruise berth length at Portas do Mar is 380 meters.",
        sources=terminal.berth_length_urls,
        additional_instruction=(
            "Confirm the berth length is specified as approximately 380 meters; accept minor variants like '380 m'."
        ),
    )

    # Berth depth alongside is 11 meters
    await verify_leaf_with_sources(
        evaluator,
        group,
        "BerthDepthAlongsideIs11m_WithURL",
        "States the berth depth alongside is 11 meters and provides a supporting URL",
        "The berth depth alongside at Portas do Mar is 11 meters.",
        sources=terminal.berth_depth_urls,
        additional_instruction=(
            "Confirm the alongside depth is specified as approximately 11 meters; accept '11 m' or '11.0 m'."
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
) -> Dict[str, Any]:
    """
    Evaluate an answer for the 2026 transatlantic cruise with Azores stop task.
    """
    # Initialize evaluator (base root is non-critical by design)
    evaluator = Evaluator()
    base_root = evaluator.initialize(
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

    # Create a critical "Root" node per rubric, under the base root
    rubric_root = evaluator.add_parallel(
        id="Root",
        desc="Identify a transatlantic cruise and verify it satisfies all stated constraints; provide all required details with supporting reference URLs",
        parent=base_root,
        critical=True,
    )

    # Perform extractions (can run concurrently)
    cruise_task = evaluator.extract(
        prompt=prompt_extract_cruise(),
        template_class=CruiseExtraction,
        extraction_name="cruise_info",
    )
    terminal_task = evaluator.extract(
        prompt=prompt_extract_terminal(),
        template_class=TerminalExtraction,
        extraction_name="terminal_info",
    )
    extracted_cruise, extracted_terminal = await asyncio.gather(cruise_task, terminal_task)

    # Add ground truth info for transparency
    evaluator.add_ground_truth(
        {
            "expected_values": GROUND_TRUTH,
            "notes": "These represent the specific cruise and terminal facts targeted by the rubric."
        },
        gt_type="ground_truth"
    )

    # Build and verify subtrees
    await verify_cruise_route_and_timing_constraints(evaluator, rubric_root, extracted_cruise)
    await verify_exact_cruise_identity_and_dates_constraints(evaluator, rubric_root, extracted_cruise)
    await verify_ponta_delgada_terminal_constraints(evaluator, rubric_root, extracted_terminal)
    await verify_ponta_delgada_berth_specs_constraints(evaluator, rubric_root, extracted_terminal)

    # Return unified summary with verification tree and scores
    return evaluator.get_summary()