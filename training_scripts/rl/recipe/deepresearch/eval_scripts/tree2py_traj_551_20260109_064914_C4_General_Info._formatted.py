import asyncio
import logging
import re
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


# ------------------------------------------------------------------------------
# Task metadata
# ------------------------------------------------------------------------------
TASK_ID = "us_airports_2024_milestones"
TASK_DESCRIPTION = (
    "Identify five U.S. airports that achieved major construction or infrastructure milestones in 2024. "
    "For each airport, provide the following information: "
    "(1) An airport that opened a new Terminal 1 Parking Plaza in August 2024 providing exactly 2,834 parking spaces - "
    "provide the airport name and the specific opening date. "
    "(2) An airport that opened two new skybridges on July 31, 2024, that connect the terminal building to the Hourly Deck - "
    "provide the airport name. "
    "(3) An airport that broke ground in April 2024 on a $135 million Terminal Enhancement Project that will add 175,000 square feet of terminal space - "
    "provide the airport name. "
    "(4) An airport that broke ground in January 2024 on a terminal replacement project that is scheduled to open in October 2026 - "
    "provide the airport name. "
    "(5) An airport that broke ground in December 2024 on a new Airside terminal with 16 gates - provide the airport name."
)


# ------------------------------------------------------------------------------
# Utilities
# ------------------------------------------------------------------------------
def _nonempty(s: Optional[str]) -> bool:
    return bool(s and isinstance(s, str) and s.strip())


def _has_specific_date_strict(date_str: Optional[str]) -> bool:
    """
    Heuristic check that the provided date string is a specific calendar date (day + month + year).
    Accept common formats like "August 13, 2024", "Aug 13, 2024", "8/13/2024", "2024-08-13".
    """
    if not _nonempty(date_str):
        return False

    s = date_str.strip()

    # Month name + day + year (e.g., August 13, 2024 or Aug 13 2024)
    month_name_pattern = r"(?i)\b(jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*\s+\d{1,2}(?:st|nd|rd|th)?(?:,)?\s+\d{4}\b"
    # Numeric MM/DD/YYYY or M/D/YYYY
    numeric_slash_pattern = r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b"
    # ISO YYYY-MM-DD
    iso_pattern = r"\b\d{4}-\d{2}-\d{2}\b"

    return bool(
        re.search(month_name_pattern, s)
        or re.search(numeric_slash_pattern, s)
        or re.search(iso_pattern, s)
    )


# ------------------------------------------------------------------------------
# Extraction models
# ------------------------------------------------------------------------------
class Item1ParkingPlaza(BaseModel):
    airport_name: Optional[str] = None
    specific_opening_date: Optional[str] = None  # e.g., "August 13, 2024"
    spaces: Optional[str] = None  # e.g., "2,834" or "2834"
    sources: List[str] = Field(default_factory=list)


class Item2Skybridges(BaseModel):
    airport_name: Optional[str] = None
    date: Optional[str] = None  # Should be "July 31, 2024"
    sources: List[str] = Field(default_factory=list)


class Item3TerminalEnhancement(BaseModel):
    airport_name: Optional[str] = None
    groundbreaking_date: Optional[str] = None  # e.g., "April 2024" or a specific April date in 2024
    project_name: Optional[str] = None  # Expect "Terminal Enhancement Project" or equivalent
    budget: Optional[str] = None  # e.g., "$135 million"
    added_sqft: Optional[str] = None  # e.g., "175,000"
    sources: List[str] = Field(default_factory=list)


class Item4TerminalReplacement(BaseModel):
    airport_name: Optional[str] = None
    groundbreaking_date: Optional[str] = None  # e.g., "January 2024" or specific date
    project_type: Optional[str] = None  # Expect "terminal replacement project"
    scheduled_opening: Optional[str] = None  # e.g., "October 2026"
    sources: List[str] = Field(default_factory=list)


class Item5AirsideTerminal(BaseModel):
    airport_name: Optional[str] = None
    groundbreaking_date: Optional[str] = None  # e.g., "December 2024" or specific date
    project_type: Optional[str] = None  # e.g., "Airside terminal"
    gates: Optional[str] = None  # e.g., "16"
    sources: List[str] = Field(default_factory=list)


class AirportsMilestonesExtraction(BaseModel):
    item1: Optional[Item1ParkingPlaza] = None
    item2: Optional[Item2Skybridges] = None
    item3: Optional[Item3TerminalEnhancement] = None
    item4: Optional[Item4TerminalReplacement] = None
    item5: Optional[Item5AirsideTerminal] = None


# ------------------------------------------------------------------------------
# Extraction prompt
# ------------------------------------------------------------------------------
def prompt_extract_airports_milestones() -> str:
    return """
    Extract from the answer exactly one airport (and requested details) for each of the five milestone categories below.
    Return a single JSON object with fields: item1, item2, item3, item4, item5.
    For all URL fields, extract only valid URLs explicitly present in the answer (including in markdown link syntax).
    If multiple candidates are present for a category, pick the one that best matches the constraints. If a category is missing, set it to null.

    item1 (Terminal 1 Parking Plaza):
      - airport_name: string
      - specific_opening_date: string (the specific opening date, e.g., "August 13, 2024"; include a day, month, and year if provided)
      - spaces: string (e.g., "2,834" or "2834", as stated)
      - sources: array of URLs (all URLs cited for this item)

    item2 (Two skybridges on July 31, 2024 connecting to Hourly Deck):
      - airport_name: string
      - date: string (e.g., "July 31, 2024"; include the date format as written)
      - sources: array of URLs (all URLs cited for this item)

    item3 (April 2024 groundbreaking; $135M Terminal Enhancement Project; adds 175,000 sq ft):
      - airport_name: string
      - groundbreaking_date: string (e.g., "April 2024" or a specific April date in 2024)
      - project_name: string (e.g., "Terminal Enhancement Project")
      - budget: string (e.g., "$135 million")
      - added_sqft: string (e.g., "175,000")
      - sources: array of URLs

    item4 (January 2024 groundbreaking; terminal replacement project; scheduled to open October 2026):
      - airport_name: string
      - groundbreaking_date: string (e.g., "January 2024" or specific date)
      - project_type: string (e.g., "terminal replacement project")
      - scheduled_opening: string (e.g., "October 2026")
      - sources: array of URLs

    item5 (December 2024 groundbreaking; new Airside terminal with 16 gates):
      - airport_name: string
      - groundbreaking_date: string (e.g., "December 2024" or specific date)
      - project_type: string (e.g., "Airside terminal")
      - gates: string (e.g., "16")
      - sources: array of URLs

    General extraction rules:
    - Extract only what is explicitly stated in the answer.
    - Do not invent or infer information.
    - If a field is not provided, return null for that field or an empty array for sources.
    - Preserve wording for numbers/dates as they appear (e.g., keep commas in "2,834" if shown).
    """


# ------------------------------------------------------------------------------
# Verification helpers
# ------------------------------------------------------------------------------
async def verify_item_1(evaluator: Evaluator, parent_node, item: Optional[Item1ParkingPlaza]) -> None:
    node = evaluator.add_parallel(
        id="item_1_parking_plaza",
        desc="Milestone (1): Airport that opened a new Terminal 1 Parking Plaza in August 2024 with exactly 2,834 spaces; includes airport name and specific opening date",
        parent=parent_node,
        critical=False,
    )

    # provides_airport_name (critical - existence)
    evaluator.add_custom_node(
        result=_nonempty(item.airport_name) if item else False,
        id="item_1_provides_airport_name",
        desc="Provides the airport name",
        parent=node,
        critical=True
    )

    # airport_is_in_us (critical - verify by URLs)
    leaf = evaluator.add_leaf(
        id="item_1_airport_is_in_us",
        desc="The identified airport is a U.S. airport",
        parent=node,
        critical=True
    )
    airport_name = item.airport_name if item and item.airport_name else ""
    await evaluator.verify(
        claim=f"The airport named '{airport_name}' is located in the United States.",
        node=leaf,
        sources=item.sources if item else [],
        additional_instruction="Confirm using the provided sources that the airport is a U.S. airport (city/state in the U.S.)."
    )

    # opened_new_terminal_1_parking_plaza (critical)
    leaf = evaluator.add_leaf(
        id="item_1_opened_new_terminal_1_parking_plaza",
        desc="States the airport opened a new Terminal 1 Parking Plaza",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim=f"{airport_name} opened a new Terminal 1 Parking Plaza.",
        node=leaf,
        sources=item.sources if item else [],
        additional_instruction="Look for explicit language like 'Terminal 1 Parking Plaza' and that it was opened (opening announcement)."
    )

    # opened_in_august_2024 (critical)
    leaf = evaluator.add_leaf(
        id="item_1_opened_in_august_2024",
        desc="States the opening occurred in August 2024",
        parent=node,
        critical=True
    )
    date_text = item.specific_opening_date if item and item.specific_opening_date else ""
    await evaluator.verify(
        claim=f"The opening of the Terminal 1 Parking Plaza at {airport_name} occurred in August 2024.",
        node=leaf,
        sources=item.sources if item else [],
        additional_instruction="Check that the opening month/year is August 2024. If a specific date is provided, it should fall within August 2024."
    )

    # parking_spaces_exactly_2834 (critical)
    leaf = evaluator.add_leaf(
        id="item_1_parking_spaces_exactly_2834",
        desc="States the parking plaza provides exactly 2,834 parking spaces",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="The Terminal 1 Parking Plaza provides exactly 2,834 parking spaces.",
        node=leaf,
        sources=item.sources if item else [],
        additional_instruction="Be strict: it must explicitly indicate 2,834 spaces (allow '2834' without comma as equivalent)."
    )

    # provides_specific_opening_date (critical - existence and specificity)
    evaluator.add_custom_node(
        result=_has_specific_date_strict(item.specific_opening_date) if item else False,
        id="item_1_provides_specific_opening_date",
        desc="Provides the specific opening date (day/month/year or equivalent unambiguous date)",
        parent=node,
        critical=True
    )


async def verify_item_2(evaluator: Evaluator, parent_node, item: Optional[Item2Skybridges]) -> None:
    node = evaluator.add_parallel(
        id="item_2_skybridges",
        desc="Milestone (2): Airport that opened two new skybridges on July 31, 2024 connecting the terminal to the Hourly Deck; provide airport name",
        parent=parent_node,
        critical=False,
    )

    # provides_airport_name
    evaluator.add_custom_node(
        result=_nonempty(item.airport_name) if item else False,
        id="item_2_provides_airport_name",
        desc="Provides the airport name",
        parent=node,
        critical=True
    )

    # airport_is_in_us
    leaf = evaluator.add_leaf(
        id="item_2_airport_is_in_us",
        desc="The identified airport is a U.S. airport",
        parent=node,
        critical=True
    )
    name = item.airport_name if item and item.airport_name else ""
    await evaluator.verify(
        claim=f"The airport named '{name}' is located in the United States.",
        node=leaf,
        sources=item.sources if item else [],
        additional_instruction="Confirm location within the U.S. using the provided sources."
    )

    # opened_two_new_skybridges
    leaf = evaluator.add_leaf(
        id="item_2_opened_two_new_skybridges",
        desc="States the airport opened two new skybridges",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim=f"{name} opened two new skybridges.",
        node=leaf,
        sources=item.sources if item else [],
        additional_instruction="Accept synonyms like 'pedestrian bridges' or 'sky bridges' if clearly equivalent, and that there are two."
    )

    # date_is_july_31_2024
    leaf = evaluator.add_leaf(
        id="item_2_date_is_july_31_2024",
        desc="States the opening date is July 31, 2024",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The opening date for the two new skybridges at {name} is July 31, 2024.",
        node=leaf,
        sources=item.sources if item else [],
        additional_instruction="Check for the exact date 'July 31, 2024' or equivalent numeric format 7/31/2024."
    )

    # connects_terminal_to_hourly_deck
    leaf = evaluator.add_leaf(
        id="item_2_connects_terminal_to_hourly_deck",
        desc="States the skybridges connect the terminal building to the Hourly Deck",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="The two skybridges connect the terminal building to the Hourly Deck.",
        node=leaf,
        sources=item.sources if item else [],
        additional_instruction="Look for phrasing that the bridges connect the terminal to the Hourly Deck (or equivalent named parking deck)."
    )


async def verify_item_3(evaluator: Evaluator, parent_node, item: Optional[Item3TerminalEnhancement]) -> None:
    node = evaluator.add_parallel(
        id="item_3_terminal_enhancement_project",
        desc="Milestone (3): Broke ground in April 2024 on a $135M Terminal Enhancement Project adding 175,000 sq ft; provide airport name",
        parent=parent_node,
        critical=False,
    )

    # provides_airport_name
    evaluator.add_custom_node(
        result=_nonempty(item.airport_name) if item else False,
        id="item_3_provides_airport_name",
        desc="Provides the airport name",
        parent=node,
        critical=True
    )

    # airport_is_in_us
    leaf = evaluator.add_leaf(
        id="item_3_airport_is_in_us",
        desc="The identified airport is a U.S. airport",
        parent=node,
        critical=True
    )
    name = item.airport_name if item and item.airport_name else ""
    await evaluator.verify(
        claim=f"The airport named '{name}' is located in the United States.",
        node=leaf,
        sources=item.sources if item else [],
        additional_instruction="Confirm U.S. location from sources."
    )

    # broke_ground_in_april_2024
    leaf = evaluator.add_leaf(
        id="item_3_broke_ground_in_april_2024",
        desc="States the airport broke ground in April 2024",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim=f"{name} broke ground in April 2024 on the relevant project.",
        node=leaf,
        sources=item.sources if item else [],
        additional_instruction="The groundbreaking must be in April 2024."
    )

    # project_is_terminal_enhancement_project
    leaf = evaluator.add_leaf(
        id="item_3_project_is_terminal_enhancement_project",
        desc="Identifies the project as a Terminal Enhancement Project",
        parent=node,
        critical=True
    )
    project_name = item.project_name if item and item.project_name else "Terminal Enhancement Project"
    await evaluator.verify(
        claim=f"The project is a Terminal Enhancement Project at {name}.",
        node=leaf,
        sources=item.sources if item else [],
        additional_instruction="Look for 'Terminal Enhancement Project' or equivalent phrasing clearly indicating such a program."
    )

    # project_cost_135_million
    leaf = evaluator.add_leaf(
        id="item_3_project_cost_135_million",
        desc="States the project cost is $135 million",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="The project budget is $135 million.",
        node=leaf,
        sources=item.sources if item else [],
        additional_instruction="Confirm the stated budget equals $135 million."
    )

    # adds_175000_sq_ft
    leaf = evaluator.add_leaf(
        id="item_3_adds_175000_sq_ft",
        desc="States the project will add 175,000 square feet of terminal space",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="The project will add 175,000 square feet of terminal space.",
        node=leaf,
        sources=item.sources if item else [],
        additional_instruction="Confirm explicit reference to 175,000 square feet (allow minor formatting differences like '175,000 sq ft')."
    )


async def verify_item_4(evaluator: Evaluator, parent_node, item: Optional[Item4TerminalReplacement]) -> None:
    node = evaluator.add_parallel(
        id="item_4_terminal_replacement_project",
        desc="Milestone (4): Broke ground in January 2024 on a terminal replacement project scheduled to open in October 2026; provide airport name",
        parent=parent_node,
        critical=False,
    )

    # provides_airport_name
    evaluator.add_custom_node(
        result=_nonempty(item.airport_name) if item else False,
        id="item_4_provides_airport_name",
        desc="Provides the airport name",
        parent=node,
        critical=True
    )

    # airport_is_in_us
    leaf = evaluator.add_leaf(
        id="item_4_airport_is_in_us",
        desc="The identified airport is a U.S. airport",
        parent=node,
        critical=True
    )
    name = item.airport_name if item and item.airport_name else ""
    await evaluator.verify(
        claim=f"The airport named '{name}' is located in the United States.",
        node=leaf,
        sources=item.sources if item else [],
        additional_instruction="Confirm US location from sources."
    )

    # broke_ground_in_january_2024
    leaf = evaluator.add_leaf(
        id="item_4_broke_ground_in_january_2024",
        desc="States the airport broke ground in January 2024",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim=f"{name} broke ground in January 2024 on the terminal project.",
        node=leaf,
        sources=item.sources if item else [],
        additional_instruction="Groundbreaking must be in January 2024."
    )

    # project_is_terminal_replacement
    leaf = evaluator.add_leaf(
        id="item_4_project_is_terminal_replacement",
        desc="States the project is a terminal replacement project",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="The project is a terminal replacement project.",
        node=leaf,
        sources=item.sources if item else [],
        additional_instruction="Look for phrasing clearly indicating a terminal replacement (not just renovation or expansion)."
    )

    # scheduled_open_october_2026
    leaf = evaluator.add_leaf(
        id="item_4_scheduled_open_october_2026",
        desc="States the project is scheduled to open in October 2026",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="The project is scheduled to open in October 2026.",
        node=leaf,
        sources=item.sources if item else [],
        additional_instruction="Confirm explicit schedule for October 2026."
    )


async def verify_item_5(evaluator: Evaluator, parent_node, item: Optional[Item5AirsideTerminal]) -> None:
    node = evaluator.add_parallel(
        id="item_5_airside_terminal",
        desc="Milestone (5): Broke ground in December 2024 on a new Airside terminal with 16 gates; provide airport name",
        parent=parent_node,
        critical=False,
    )

    # provides_airport_name
    evaluator.add_custom_node(
        result=_nonempty(item.airport_name) if item else False,
        id="item_5_provides_airport_name",
        desc="Provides the airport name",
        parent=node,
        critical=True
    )

    # airport_is_in_us
    leaf = evaluator.add_leaf(
        id="item_5_airport_is_in_us",
        desc="The identified airport is a U.S. airport",
        parent=node,
        critical=True
    )
    name = item.airport_name if item and item.airport_name else ""
    await evaluator.verify(
        claim=f"The airport named '{name}' is located in the United States.",
        node=leaf,
        sources=item.sources if item else [],
        additional_instruction="Confirm US location from sources."
    )

    # broke_ground_in_december_2024
    leaf = evaluator.add_leaf(
        id="item_5_broke_ground_in_december_2024",
        desc="States the airport broke ground in December 2024",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim=f"{name} broke ground in December 2024 on the new terminal.",
        node=leaf,
        sources=item.sources if item else [],
        additional_instruction="Groundbreaking must be in December 2024."
    )

    # new_airside_terminal
    leaf = evaluator.add_leaf(
        id="item_5_new_airside_terminal",
        desc="States the project is a new Airside terminal",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="The project is a new Airside terminal.",
        node=leaf,
        sources=item.sources if item else [],
        additional_instruction="Look for explicit use of 'Airside terminal' for this project."
    )

    # has_16_gates
    leaf = evaluator.add_leaf(
        id="item_5_has_16_gates",
        desc="States the new Airside terminal has 16 gates",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="The new Airside terminal includes 16 gates.",
        node=leaf,
        sources=item.sources if item else [],
        additional_instruction="Confirm exactly 16 gates (allow 'sixteen' as equivalent)."
    )


# ------------------------------------------------------------------------------
# Main evaluation function
# ------------------------------------------------------------------------------
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
    Entry point for evaluating an answer for the U.S. airports 2024 milestones task.
    Builds the verification tree according to the rubric and returns the evaluation summary.
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
        default_model=model
    )

    # Extraction
    extracted: AirportsMilestonesExtraction = await evaluator.extract(
        prompt=prompt_extract_airports_milestones(),
        template_class=AirportsMilestonesExtraction,
        extraction_name="airports_milestones"
    )

    # Build and verify each item subtree according to rubric (parallel under root)
    await verify_item_1(evaluator, root, extracted.item1)
    await verify_item_2(evaluator, root, extracted.item2)
    await verify_item_3(evaluator, root, extracted.item3)
    await verify_item_4(evaluator, root, extracted.item4)
    await verify_item_5(evaluator, root, extracted.item5)

    # Return final summary
    return evaluator.get_summary()