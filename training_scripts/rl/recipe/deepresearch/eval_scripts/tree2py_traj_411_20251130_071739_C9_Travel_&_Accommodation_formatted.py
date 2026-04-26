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
TASK_ID = "us_travel_infra_apr_nov_2025"
TASK_DESCRIPTION = (
    "Research and compile detailed information about 4 major travel infrastructure developments in the United States "
    "that opened or launched between April 1 and November 30, 2025. Specifically, identify:\n\n"
    "1. An airport security checkpoint development: Find a US airport that opened a new security checkpoint during this period. "
    "Provide the airport name, the specific checkpoint designation (e.g., East Checkpoint, West Checkpoint), the exact opening date, "
    "the terminal level or location where it's situated, the number of screening lanes, and describe one advanced screening technology "
    "feature that was implemented.\n\n"
    "2. An airline base opening: Find a US airline that opened a new base of operations at a US airport during this period. "
    "Provide the airline name, the airport name where the base was established, the exact opening date, which numbered base this represents "
    "for that airline (e.g., 'eighth base'), the number of initial routes launched when the base opened, and list the initial destination "
    "airports served from this new base.\n\n"
    "3. A cruise ship launch: Find a major cruise ship that had its maiden voyage from a US home port during this period. "
    "Provide the ship name, the cruise line operating it, the exact maiden voyage date, the US home port, the ship's class designation, "
    "and the ship's gross tonnage.\n\n"
    "4. An airport terminal opening: Find a US airport that opened a new terminal building during this period. "
    "Provide the airport name, the exact opening date, the city and state location, the total project cost in billions of dollars, "
    "the number of security lanes in the new terminal, and describe one significant improvement or benefit for passengers compared "
    "to the old facility.\n\n"
    "For each development, provide reference URLs that verify the information."
)


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class SecurityCheckpointDev(BaseModel):
    airport_name: Optional[str] = None
    checkpoint_designation: Optional[str] = None
    opening_date: Optional[str] = None
    terminal_location: Optional[str] = None
    number_of_screening_lanes: Optional[str] = None
    advanced_screening_technology_feature: Optional[str] = None
    source_urls: List[str] = Field(default_factory=list)


class AirlineBaseDev(BaseModel):
    airline_name: Optional[str] = None
    airport_name: Optional[str] = None
    opening_date: Optional[str] = None
    numbered_base_ordinal: Optional[str] = None
    initial_route_count: Optional[str] = None
    initial_destination_airports: List[str] = Field(default_factory=list)
    source_urls: List[str] = Field(default_factory=list)


class CruiseShipDev(BaseModel):
    ship_name: Optional[str] = None
    cruise_line: Optional[str] = None
    maiden_voyage_date: Optional[str] = None
    us_home_port: Optional[str] = None
    ship_class: Optional[str] = None
    gross_tonnage: Optional[str] = None
    source_urls: List[str] = Field(default_factory=list)


class TerminalOpeningDev(BaseModel):
    airport_name: Optional[str] = None
    opening_date: Optional[str] = None
    city_and_state: Optional[str] = None
    project_cost_billions_usd: Optional[str] = None
    security_lane_count: Optional[str] = None
    key_passenger_benefit: Optional[str] = None
    source_urls: List[str] = Field(default_factory=list)


class AllDevelopments(BaseModel):
    airport_security_checkpoint: Optional[SecurityCheckpointDev] = None
    airline_base_opening: Optional[AirlineBaseDev] = None
    cruise_ship_launch: Optional[CruiseShipDev] = None
    airport_terminal_opening: Optional[TerminalOpeningDev] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_all() -> str:
    return """
Extract from the answer exactly one example for each of the following four categories (if multiple are provided, choose the first well-supported one). Return fields exactly as strings where applicable, and collect all reference URLs mentioned for that item into 'source_urls'. If any required information is missing in the answer, set it to null (or empty list for arrays).

Top-level JSON keys must be:
- airport_security_checkpoint: object or null
- airline_base_opening: object or null
- cruise_ship_launch: object or null
- airport_terminal_opening: object or null

For each category, extract the following fields:

1) airport_security_checkpoint:
- airport_name: string
- checkpoint_designation: string
- opening_date: string (as written, e.g., "April 15, 2025" or "2025-04-15")
- terminal_location: string
- number_of_screening_lanes: string (keep units/format if present, e.g., "10", "10 lanes")
- advanced_screening_technology_feature: string
- source_urls: array of URLs that support the above attributes

2) airline_base_opening:
- airline_name: string
- airport_name: string
- opening_date: string
- numbered_base_ordinal: string (e.g., "eighth base")
- initial_route_count: string (keep original form, e.g., "8", "eight")
- initial_destination_airports: array of strings (airport names or codes)
- source_urls: array of URLs that support the above attributes

3) cruise_ship_launch:
- ship_name: string
- cruise_line: string
- maiden_voyage_date: string
- us_home_port: string
- ship_class: string
- gross_tonnage: string (keep original format, e.g., "250,800 GT")
- source_urls: array of URLs that support the above attributes

4) airport_terminal_opening:
- airport_name: string
- opening_date: string
- city_and_state: string (e.g., "Kansas City, Missouri")
- project_cost_billions_usd: string (keep as written, e.g., "$1.5 billion", "1.5 billion USD")
- security_lane_count: string
- key_passenger_benefit: string (a succinct textual description)
- source_urls: array of URLs that support the above attributes

Rules for URLs extraction:
- Only include URLs explicitly present in the answer (plain or markdown).
- Do not invent URLs.
- Deduplicate if the same URL appears multiple times.
- If no URLs are provided for a category, return an empty list for source_urls.
"""


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def list_to_english(items: List[str]) -> str:
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    return ", ".join(items[:-1]) + " and " + items[-1]


def safe_sources(urls: Optional[List[str]]) -> List[str]:
    return [u for u in (urls or []) if isinstance(u, str) and u.strip()]


# --------------------------------------------------------------------------- #
# Verification builders per category                                          #
# --------------------------------------------------------------------------- #
async def verify_checkpoint_dev(evaluator: Evaluator, parent_node, data: Optional[SecurityCheckpointDev]) -> None:
    node = evaluator.add_parallel(
        id="Airport_Security_Checkpoint_Development",
        desc="A US airport that opened a new security checkpoint during the specified timeframe, with required details and sources.",
        parent=parent_node,
        critical=False
    )
    airport_name = data.airport_name if data else None
    checkpoint_designation = data.checkpoint_designation if data else None
    opening_date = data.opening_date if data else None
    terminal_location = data.terminal_location if data else None
    lanes = data.number_of_screening_lanes if data else None
    tech = data.advanced_screening_technology_feature if data else None
    refs = safe_sources(data.source_urls if data else [])

    # Reference URLs presence (critical gate)
    evaluator.add_custom_node(
        result=len(refs) > 0,
        id="checkpoint_reference_urls_present",
        desc="Reference URL(s) exist for the checkpoint development",
        parent=node,
        critical=True
    )

    # Airport Name
    leaf = evaluator.add_leaf(
        id="checkpoint_airport_name",
        desc="Name of the US airport.",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The new security checkpoint opened at {airport_name}.",
        node=leaf,
        sources=refs,
        additional_instruction="Verify that the referenced page(s) explicitly identify the airport where the new checkpoint opened. Allow minor name variants (e.g., full official airport name vs. commonly used name)."
    )

    # Checkpoint Designation
    leaf = evaluator.add_leaf(
        id="checkpoint_designation",
        desc="Specific checkpoint designation/name (e.g., East/West/terminal designation).",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The checkpoint designation/name is '{checkpoint_designation}'.",
        node=leaf,
        sources=refs,
        additional_instruction="Verify that the source explicitly names the checkpoint (e.g., East/West, Level 3 checkpoint, etc.). Allow minor formatting differences."
    )

    # Opening Date in Range (supported by sources)
    leaf = evaluator.add_leaf(
        id="checkpoint_opening_date_in_range",
        desc="Exact opening date, and it is between April 1 and November 30, 2025.",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The new security checkpoint opened on {opening_date}, and that date falls between April 1, 2025 and November 30, 2025 (inclusive).",
        node=leaf,
        sources=refs,
        additional_instruction="First, confirm the opening date from the source(s). Then confirm that this date is within 2025-04-01 to 2025-11-30 inclusive."
    )

    # Terminal Location
    leaf = evaluator.add_leaf(
        id="checkpoint_terminal_location",
        desc="Terminal level or specific location where the checkpoint is situated.",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The checkpoint is located at: {terminal_location}.",
        node=leaf,
        sources=refs,
        additional_instruction="Confirm the terminal/level/location phrasing from the source(s). Minor wording differences are acceptable if the meaning matches."
    )

    # Number of Screening Lanes
    leaf = evaluator.add_leaf(
        id="checkpoint_screening_lanes",
        desc="Number of screening lanes in the new checkpoint.",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The new checkpoint has {lanes} screening lanes.",
        node=leaf,
        sources=refs,
        additional_instruction="Look for an explicit count of lanes. Accept slight variations like 'up to 10 lanes' matching '10' if context indicates the same count."
    )

    # Advanced screening technology feature
    leaf = evaluator.add_leaf(
        id="checkpoint_tech_feature",
        desc="One advanced screening technology feature implemented.",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The checkpoint implements the advanced screening technology: {tech}.",
        node=leaf,
        sources=refs,
        additional_instruction="Verify that the feature (e.g., CT scanners, ASLs, credential authentication technology) is explicitly mentioned for this checkpoint."
    )


async def verify_airline_base_dev(evaluator: Evaluator, parent_node, data: Optional[AirlineBaseDev]) -> None:
    node = evaluator.add_parallel(
        id="Airline_Base_Opening_Development",
        desc="A US airline that opened a new base of operations at a US airport during the specified timeframe, with required details and sources.",
        parent=parent_node,
        critical=False
    )
    airline = data.airline_name if data else None
    airport = data.airport_name if data else None
    date = data.opening_date if data else None
    ordinal = data.numbered_base_ordinal if data else None
    route_count = data.initial_route_count if data else None
    initial_destinations = data.initial_destination_airports if data else []
    refs = safe_sources(data.source_urls if data else [])

    # Reference URLs presence (critical gate)
    evaluator.add_custom_node(
        result=len(refs) > 0,
        id="base_reference_urls_present",
        desc="Reference URL(s) exist for the airline base opening",
        parent=node,
        critical=True
    )

    # Airline Name
    leaf = evaluator.add_leaf(
        id="base_airline_name",
        desc="Name of the airline opening the new base.",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The airline that opened the base is {airline}.",
        node=leaf,
        sources=refs,
        additional_instruction="Confirm the airline name exactly or with minor variants (e.g., 'Southwest Airlines' vs 'Southwest')."
    )

    # Airport Name
    leaf = evaluator.add_leaf(
        id="base_airport_name",
        desc="Name of the US airport where the base was established.",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The base was established at {airport}.",
        node=leaf,
        sources=refs,
        additional_instruction="Confirm the airport specified is the location of the new base."
    )

    # Opening Date in Range (supported by sources)
    leaf = evaluator.add_leaf(
        id="base_opening_date_in_range",
        desc="Exact base opening date, and it is between April 1 and November 30, 2025.",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The base opened on {date}, and that date falls between April 1, 2025 and November 30, 2025 (inclusive).",
        node=leaf,
        sources=refs,
        additional_instruction="First, confirm the date from sources; then check that it is within the specified window."
    )

    # Numbered Base Ordinal
    leaf = evaluator.add_leaf(
        id="base_ordinal",
        desc="Which numbered base this represents for the airline (e.g., eighth base).",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim=f"This new base is the airline's {ordinal}.",
        node=leaf,
        sources=refs,
        additional_instruction="Look for phrases like 'eighth base' or '#8 base' and accept equivalent wording."
    )

    # Initial Route Count
    leaf = evaluator.add_leaf(
        id="base_initial_route_count",
        desc="Number of initial routes launched when the base opened.",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim=f"When the base opened, {route_count} routes were initially launched.",
        node=leaf,
        sources=refs,
        additional_instruction="Confirm the initial number of routes from source(s). Accept words or numerals representing the same count."
    )

    # Initial Destination Airports
    leaf = evaluator.add_leaf(
        id="base_initial_destinations",
        desc="List of initial destination airports served from the new base at opening.",
        parent=node,
        critical=True
    )
    destinations_text = list_to_english(initial_destinations)
    await evaluator.verify(
        claim=f"The initial destination airports from the new base included: {destinations_text}.",
        node=leaf,
        sources=refs,
        additional_instruction="Match the list of initial destinations from source(s). Minor name/code formatting differences are acceptable if they refer to the same airports."
    )


async def verify_cruise_ship_dev(evaluator: Evaluator, parent_node, data: Optional[CruiseShipDev]) -> None:
    node = evaluator.add_parallel(
        id="Cruise_Ship_Launch_Development",
        desc="A major cruise ship that had its maiden voyage from a US home port during the specified timeframe, with required details and sources.",
        parent=parent_node,
        critical=False
    )
    ship = data.ship_name if data else None
    line = data.cruise_line if data else None
    date = data.maiden_voyage_date if data else None
    home_port = data.us_home_port if data else None
    ship_class = data.ship_class if data else None
    tonnage = data.gross_tonnage if data else None
    refs = safe_sources(data.source_urls if data else [])

    # Reference URLs presence (critical gate)
    evaluator.add_custom_node(
        result=len(refs) > 0,
        id="cruise_reference_urls_present",
        desc="Reference URL(s) exist for the cruise ship launch",
        parent=node,
        critical=True
    )

    # Ship Name
    leaf = evaluator.add_leaf(
        id="cruise_ship_name",
        desc="Name of the cruise ship.",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The cruise ship's name is {ship}.",
        node=leaf,
        sources=refs,
        additional_instruction="Verify the specific vessel name from the source(s)."
    )

    # Cruise Line
    leaf = evaluator.add_leaf(
        id="cruise_line",
        desc="Cruise line operating the ship.",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The ship is operated by {line}.",
        node=leaf,
        sources=refs,
        additional_instruction="Confirm the cruise line operating the vessel."
    )

    # Maiden Voyage Date in Range
    leaf = evaluator.add_leaf(
        id="cruise_maiden_date_in_range",
        desc="Exact maiden voyage date, and it is between April 1 and November 30, 2025.",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The maiden voyage occurred on {date}, and that date falls between April 1, 2025 and November 30, 2025 (inclusive).",
        node=leaf,
        sources=refs,
        additional_instruction="Confirm the maiden voyage date from source(s) and verify it lies within the specified window."
    )

    # US Home Port
    leaf = evaluator.add_leaf(
        id="cruise_home_port",
        desc="US home port for the maiden voyage.",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The US home port for the maiden voyage was {home_port}.",
        node=leaf,
        sources=refs,
        additional_instruction="Confirm the home port in the US for the maiden voyage."
    )

    # Ship Class
    leaf = evaluator.add_leaf(
        id="cruise_ship_class",
        desc="Ship class designation.",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The ship's class is {ship_class}.",
        node=leaf,
        sources=refs,
        additional_instruction="Confirm the class designation of the vessel."
    )

    # Gross Tonnage
    leaf = evaluator.add_leaf(
        id="cruise_gross_tonnage",
        desc="Ship gross tonnage.",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The ship's gross tonnage is {tonnage}.",
        node=leaf,
        sources=refs,
        additional_instruction="Confirm the published gross tonnage. Accept minor formatting differences (e.g., inclusion of 'GT') if the value matches."
    )


async def verify_terminal_opening_dev(evaluator: Evaluator, parent_node, data: Optional[TerminalOpeningDev]) -> None:
    node = evaluator.add_parallel(
        id="Airport_Terminal_Opening_Development",
        desc="A US airport that opened a new terminal building during the specified timeframe, with required details and sources.",
        parent=parent_node,
        critical=False
    )
    airport = data.airport_name if data else None
    date = data.opening_date if data else None
    city_state = data.city_and_state if data else None
    project_cost = data.project_cost_billions_usd if data else None
    lanes = data.security_lane_count if data else None
    benefit = data.key_passenger_benefit if data else None
    refs = safe_sources(data.source_urls if data else [])

    # Reference URLs presence (critical gate)
    evaluator.add_custom_node(
        result=len(refs) > 0,
        id="terminal_reference_urls_present",
        desc="Reference URL(s) exist for the airport terminal opening",
        parent=node,
        critical=True
    )

    # Airport Name
    leaf = evaluator.add_leaf(
        id="terminal_airport_name",
        desc="Name of the US airport that opened a new terminal building.",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The airport that opened the new terminal is {airport}.",
        node=leaf,
        sources=refs,
        additional_instruction="Confirm the airport name from source(s)."
    )

    # Opening Date in Range
    leaf = evaluator.add_leaf(
        id="terminal_opening_date_in_range",
        desc="Exact terminal opening date, and it is between April 1 and November 30, 2025.",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The new terminal opened on {date}, and that date falls between April 1, 2025 and November 30, 2025 (inclusive).",
        node=leaf,
        sources=refs,
        additional_instruction="Confirm the terminal opening date from source(s) and verify it lies within the specified window."
    )

    # City and State
    leaf = evaluator.add_leaf(
        id="terminal_city_state",
        desc="City and state where the airport is located.",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The airport is located in {city_state}.",
        node=leaf,
        sources=refs,
        additional_instruction="Verify city and state from source(s). Allow minor variants (e.g., official names vs. common usage)."
    )

    # Project Cost in Billions USD
    leaf = evaluator.add_leaf(
        id="terminal_project_cost",
        desc="Total project cost expressed in billions of dollars (or clearly convertible to billions).",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The total project cost for the terminal is {project_cost}.",
        node=leaf,
        sources=refs,
        additional_instruction="Confirm the project cost. Accept equivalent expressions (e.g., '$1.5 billion', 'USD 1.5 billion'). Focus on the value being in billions."
    )

    # Security Lane Count
    leaf = evaluator.add_leaf(
        id="terminal_security_lanes",
        desc="Number of security lanes in the new terminal.",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The new terminal has {lanes} security lanes.",
        node=leaf,
        sources=refs,
        additional_instruction="Confirm the security screening lane count from sources."
    )

    # Key Passenger Benefit
    leaf = evaluator.add_leaf(
        id="terminal_key_benefit",
        desc="One significant passenger improvement/benefit compared to the old facility.",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim=f"A significant passenger improvement compared to the old facility is: {benefit}.",
        node=leaf,
        sources=refs,
        additional_instruction="Confirm an explicit improvement or benefit cited in sources (e.g., faster screening, more space, better amenities)."
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
    Evaluate an answer for the US Travel Infrastructure Developments (Apr-Nov 2025) task.
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

    # Extract structured data
    extracted: AllDevelopments = await evaluator.extract(
        prompt=prompt_extract_all(),
        template_class=AllDevelopments,
        extraction_name="developments_extraction"
    )

    # Build verification tree according to rubric (parallel across 4 developments)
    await verify_checkpoint_dev(evaluator, root, extracted.airport_security_checkpoint)
    await verify_airline_base_dev(evaluator, root, extracted.airline_base_opening)
    await verify_cruise_ship_dev(evaluator, root, extracted.cruise_ship_launch)
    await verify_terminal_opening_dev(evaluator, root, extracted.airport_terminal_opening)

    # Return final summary
    return evaluator.get_summary()