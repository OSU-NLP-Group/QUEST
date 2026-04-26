import asyncio
import logging
from typing import Any, List, Optional, Dict

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "largest_ca_indoor_nba_arena"
TASK_DESCRIPTION = (
    "Identify the largest indoor arena in California by basketball seating capacity that currently serves as the home "
    "venue for at least one NBA team. Provide the following verified information about this arena:\n\n"
    "1. The current official name of the arena\n"
    "2. The city in California where it is located\n"
    "3. The exact basketball seating capacity\n"
    "4. The name(s) of the current NBA team(s) that use this arena as their home venue\n"
    "5. The year the arena originally opened\n"
    "6. Confirmation that the arena has a regulation-size professional basketball court (94 feet by 50 feet)\n"
    "7. Confirmation that this is indeed the largest indoor basketball arena in California by seating capacity\n\n"
    "For each piece of information provided, include a reference URL from a reliable source that verifies the claim."
)


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class FieldValue(BaseModel):
    value: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class ListFieldValue(BaseModel):
    values: List[str] = Field(default_factory=list)
    sources: List[str] = Field(default_factory=list)


class ArenaExtraction(BaseModel):
    """
    Structured extraction for the selected arena and all required verified items.
    """
    selected_arena_name: FieldValue = Field(default_factory=FieldValue)
    city: FieldValue = Field(default_factory=FieldValue)
    capacity: FieldValue = Field(default_factory=FieldValue)  # Prefer strings to accommodate formats like "19,079"
    nba_teams: ListFieldValue = Field(default_factory=ListFieldValue)
    opening_year: FieldValue = Field(default_factory=FieldValue)
    regulation_court_confirmation: FieldValue = Field(default_factory=FieldValue)
    largest_in_ca_confirmation: FieldValue = Field(default_factory=FieldValue)
    indoor_facility_confirmation: FieldValue = Field(default_factory=FieldValue)
    operational_2024_2025_confirmation: FieldValue = Field(default_factory=FieldValue)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_arena_details() -> str:
    return (
        "Extract the single arena that the answer identifies as the largest indoor arena in California by basketball "
        "seating capacity that currently serves as the home venue for at least one NBA team. For that one arena, "
        "extract the following fields exactly as stated in the answer, and list all verifying reference URLs provided "
        "in the answer for each field.\n\n"
        "Return JSON with these keys:\n"
        "- selected_arena_name: { value: string|null, sources: [url, ...] }  // The current official name\n"
        "- city: { value: string|null, sources: [url, ...] }                 // City in California\n"
        "- capacity: { value: string|null, sources: [url, ...] }             // Exact basketball seating capacity\n"
        "- nba_teams: { values: [string, ...], sources: [url, ...] }         // Current NBA home team(s)\n"
        "- opening_year: { value: string|null, sources: [url, ...] }         // Original opening year\n"
        "- regulation_court_confirmation: { value: string|null, sources: [url, ...] }  // Confirmation 94x50 court\n"
        "- largest_in_ca_confirmation: { value: string|null, sources: [url, ...] }     // Confirmation largest in CA by basketball capacity\n"
        "- indoor_facility_confirmation: { value: string|null, sources: [url, ...] }   // Confirmation permanent indoor facility\n"
        "- operational_2024_2025_confirmation: { value: string|null, sources: [url, ...] } // Confirmation currently operational in 2024–2025\n\n"
        "Rules:\n"
        "1) Extract only what the answer explicitly provides. Do not invent or infer missing values.\n"
        "2) For URLs, include only actual URLs present in the answer text (plain or markdown). If missing, use an empty array.\n"
        "3) Keep values as strings (e.g., capacity '19,079').\n"
        "4) If any value is not mentioned, set value to null; if teams are not listed, return an empty array.\n"
    )


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _safe_str(s: Optional[str]) -> str:
    return (s or "").strip()


def _has_value(s: Optional[str]) -> bool:
    return bool(_safe_str(s))


def _has_sources(srcs: Optional[List[str]]) -> bool:
    return bool(srcs) and len([u for u in srcs if _safe_str(u)]) > 0


def _join_list(values: List[str]) -> str:
    return ", ".join([v.strip() for v in values if v and v.strip()])


# --------------------------------------------------------------------------- #
# Verification helpers                                                        #
# --------------------------------------------------------------------------- #
async def add_value_source_verification(
    evaluator: Evaluator,
    parent_node,
    id_base: str,
    description: str,
    value_present: bool,
    sources_list: List[str],
    claim: str,
    add_ins: str,
) -> None:
    """
    Build a small sequential subtree:
      1) value is provided (critical)
      2) at least one source is provided (critical)
      3) verify the claim against the provided sources (critical leaf)
    """
    subnode = evaluator.add_sequential(
        id=id_base,
        desc=description,
        parent=parent_node,
        critical=True,
    )

    evaluator.add_custom_node(
        result=value_present,
        id=f"{id_base}_value_provided",
        desc=f"Value for '{id_base}' is provided in the answer",
        parent=subnode,
        critical=True,
    )

    evaluator.add_custom_node(
        result=_has_sources(sources_list),
        id=f"{id_base}_sources_provided",
        desc=f"At least one reference URL for '{id_base}' is provided",
        parent=subnode,
        critical=True,
    )

    leaf = evaluator.add_leaf(
        id=f"{id_base}_verify",
        desc=description,
        parent=subnode,
        critical=True,
    )

    await evaluator.verify(
        claim=claim,
        node=leaf,
        sources=sources_list,
        additional_instruction=add_ins,
    )


async def add_source_only_verification(
    evaluator: Evaluator,
    parent_node,
    id_base: str,
    description: str,
    sources_list: List[str],
    claim: str,
    add_ins: str,
    value_present: bool,
) -> None:
    """
    Similar to add_value_source_verification, but used for boolean/confirmation-style checks
    where the 'value' may just be a simple phrase/boolean. Still requires value present and sources.
    """
    subnode = evaluator.add_sequential(
        id=id_base,
        desc=description,
        parent=parent_node,
        critical=True,
    )

    evaluator.add_custom_node(
        result=value_present,
        id=f"{id_base}_value_provided",
        desc=f"Confirmation/assertion for '{id_base}' is provided in the answer",
        parent=subnode,
        critical=True,
    )

    evaluator.add_custom_node(
        result=_has_sources(sources_list),
        id=f"{id_base}_sources_provided",
        desc=f"At least one reference URL for '{id_base}' is provided",
        parent=subnode,
        critical=True,
    )

    leaf = evaluator.add_leaf(
        id=f"{id_base}_verify",
        desc=description,
        parent=subnode,
        critical=True,
    )

    await evaluator.verify(
        claim=claim,
        node=leaf,
        sources=sources_list,
        additional_instruction=add_ins,
    )


# --------------------------------------------------------------------------- #
# Subtree builders                                                            #
# --------------------------------------------------------------------------- #
async def build_additional_constraints(
    evaluator: Evaluator,
    parent_node,
    ex: ArenaExtraction,
) -> None:
    """
    Build 'Arena_Meets_Additional_Constraints' subtree:
      - Permanent_Indoor_Facility_With_Source
      - Operational_2024_2025_With_Source
    """
    constraints_node = evaluator.add_parallel(
        id="Arena_Meets_Additional_Constraints",
        desc="Checks constraints that are required but not fully covered by the requested fields.",
        parent=parent_node,
        critical=True,
    )

    arena_name = _safe_str(ex.selected_arena_name.value) or "the arena"

    # Permanent indoor facility
    await add_source_only_verification(
        evaluator=evaluator,
        parent_node=constraints_node,
        id_base="Permanent_Indoor_Facility_With_Source",
        description="Confirms the venue is a permanent indoor facility (not an outdoor stadium or temporary venue) and provides at least one supporting reference URL.",
        sources_list=ex.indoor_facility_confirmation.sources,
        claim=f"{arena_name} is a permanent indoor facility (fully enclosed indoor arena).",
        add_ins=(
            "Confirm from the cited page(s) that the venue is an indoor arena (roofed/enclosed). "
            "Terms like 'indoor arena', 'arena', or clear indications of a roof/enclosure are acceptable. "
            "Do not accept outdoor stadiums or temporary venues."
        ),
        value_present=_has_value(ex.indoor_facility_confirmation.value),
    )

    # Operational in 2024–2025
    await add_source_only_verification(
        evaluator=evaluator,
        parent_node=constraints_node,
        id_base="Operational_2024_2025_With_Source",
        description="Confirms the venue is currently operational as of 2024–2025 and provides at least one supporting reference URL.",
        sources_list=ex.operational_2024_2025_confirmation.sources,
        claim=f"As of the 2024–2025 timeframe, {arena_name} is operational and hosting events/games.",
        add_ins=(
            "Accept event calendars, schedule pages, or credible news indicating current operations in 2024–2025. "
            "Focus on evidence that the venue is open and actively hosting events."
        ),
        value_present=_has_value(ex.operational_2024_2025_confirmation.value),
    )


async def build_required_details(
    evaluator: Evaluator,
    parent_node,
    ex: ArenaExtraction,
) -> None:
    """
    Build 'Required_Arena_Details_With_Verification' subtree with all seven required items.
    """
    details_node = evaluator.add_parallel(
        id="Required_Arena_Details_With_Verification",
        desc="All required information items are provided and each includes at least one verifying reference URL.",
        parent=parent_node,
        critical=True,
    )

    arena_name = _safe_str(ex.selected_arena_name.value) or "the arena"

    # 1) Current official name
    await add_value_source_verification(
        evaluator=evaluator,
        parent_node=details_node,
        id_base="Current_Official_Name_With_Source",
        description="Provides the current official arena name (reflecting any recent naming rights changes) and includes a verifying reference URL.",
        value_present=_has_value(ex.selected_arena_name.value),
        sources_list=ex.selected_arena_name.sources,
        claim=f"The current official name of the arena is '{arena_name}'.",
        add_ins=(
            "Verify the present official name as of 2024–2025 (post any recent naming rights changes). "
            "Minor formatting differences (case, punctuation) are acceptable."
        ),
    )

    # 2) City in California
    city_val = _safe_str(ex.city.value)
    await add_value_source_verification(
        evaluator=evaluator,
        parent_node=details_node,
        id_base="City_In_California_With_Source",
        description="Provides the city in California where the arena is located and includes a verifying reference URL.",
        value_present=_has_value(ex.city.value),
        sources_list=ex.city.sources,
        claim=f"{arena_name} is located in {city_val}, California.",
        add_ins=(
            "Confirm the city and that it is in California. Accept common variants like 'Los Angeles, CA' or 'Inglewood, CA'."
        ),
    )

    # 3) Exact basketball seating capacity (official/authoritative) and >= 10,000
    capacity_val = _safe_str(ex.capacity.value)
    await add_value_source_verification(
        evaluator=evaluator,
        parent_node=details_node,
        id_base="Exact_Basketball_Seating_Capacity_Official_And_GTE_10000_With_Source",
        description="Provides the exact basketball seating capacity, supported by an official/authoritative source, and the stated capacity is at least 10,000 seats; includes a verifying reference URL.",
        value_present=_has_value(ex.capacity.value),
        sources_list=ex.capacity.sources,
        claim=f"The basketball seating capacity of {arena_name} is {capacity_val}.",
        add_ins=(
            "Verify that the capacity number refers specifically to the basketball configuration (not concerts/hockey). "
            "Treat minor formatting variations (commas, rounding) as acceptable. "
            "Also confirm that this capacity is at least 10,000 seats."
        ),
    )

    # 4) Current NBA home team(s)
    teams_str = _join_list(ex.nba_teams.values)
    await add_value_source_verification(
        evaluator=evaluator,
        parent_node=details_node,
        id_base="Current_NBA_Home_Team_Names_With_Source",
        description="Provides the name(s) of the current NBA team(s) that use the arena as their home venue and includes a verifying reference URL.",
        value_present=bool(ex.nba_teams.values) and len(ex.nba_teams.values) > 0,
        sources_list=ex.nba_teams.sources,
        claim=f"The current NBA team(s) that use {arena_name} as their home venue are: {teams_str}.",
        add_ins=(
            "Confirm that the listed team(s) currently use this arena as their home venue in the 2024–2025 season. "
            "Accept phrasing like 'home arena' or 'home court'."
        ),
    )

    # 5) Opening year
    opening_year_val = _safe_str(ex.opening_year.value)
    await add_value_source_verification(
        evaluator=evaluator,
        parent_node=details_node,
        id_base="Opening_Year_With_Source",
        description="Provides the year the arena originally opened and includes a verifying reference URL.",
        value_present=_has_value(ex.opening_year.value),
        sources_list=ex.opening_year.sources,
        claim=f"{arena_name} originally opened in {opening_year_val}.",
        add_ins=(
            "Confirm the original opening year (first opening to the public), not a major renovation year."
        ),
    )

    # 6) Regulation-size professional basketball court (94x50)
    await add_source_only_verification(
        evaluator=evaluator,
        parent_node=details_node,
        id_base="Regulation_Size_Court_94x50_Confirmation_With_Source",
        description="Confirms the arena has a regulation-size professional basketball court (94 ft by 50 ft) and includes a verifying reference URL.",
        sources_list=ex.regulation_court_confirmation.sources,
        claim=f"The basketball court at {arena_name} is regulation size: 94 feet by 50 feet.",
        add_ins=(
            "Prefer explicit confirmation of 94x50 dimensions. "
            "If the source credibly states this arena hosts NBA games, it implies a regulation 94x50 court per NBA standards."
        ),
        value_present=_has_value(ex.regulation_court_confirmation.value),
    )

    # 7) Largest in CA by basketball capacity
    await add_source_only_verification(
        evaluator=evaluator,
        parent_node=details_node,
        id_base="Largest_In_CA_By_Basketball_Capacity_Confirmation_With_Source",
        description="Confirms this arena has the highest basketball seating capacity among all indoor arenas in California and includes a verifying reference URL.",
        sources_list=ex.largest_in_ca_confirmation.sources,
        claim=f"Among indoor arenas in California, {arena_name} has the highest basketball seating capacity.",
        add_ins=(
            "Accept explicit statements declaring 'largest' by basketball seating capacity in California. "
            "Alternatively, if sources list capacities for multiple California indoor arenas and this arena's capacity "
            "is higher than all others, consider that sufficient."
        ),
        value_present=_has_value(ex.largest_in_ca_confirmation.value),
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
    Evaluate an answer for the largest CA indoor NBA arena task.
    """
    # Initialize evaluator with a non-critical root and then add a critical top-level node
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.SEQUENTIAL,
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

    # Extract structured arena details from the answer
    ex: ArenaExtraction = await evaluator.extract(
        prompt=prompt_extract_arena_details(),
        template_class=ArenaExtraction,
        extraction_name="arena_extraction",
    )

    # Create the critical top-level node as described by the rubric
    top = evaluator.add_sequential(
        id="Largest_CA_Indoor_NBA_Arena",
        desc="Evaluate whether the response correctly identifies California's largest indoor arena by basketball seating capacity that is a current NBA home venue, and provides all required verified details with sources.",
        parent=root,
        critical=True,
    )

    # Build additional constraints subtree (critical)
    await build_additional_constraints(evaluator, top, ex)

    # Build required details subtree (critical)
    await build_required_details(evaluator, top, ex)

    # Return evaluator summary
    return evaluator.get_summary()