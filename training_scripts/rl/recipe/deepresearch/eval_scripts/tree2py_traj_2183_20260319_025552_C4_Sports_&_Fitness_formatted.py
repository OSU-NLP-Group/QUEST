import asyncio
import logging
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator, AggregationStrategy

# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "nfl_stadiums_2026"
TASK_DESCRIPTION = """
As part of a comprehensive NFL infrastructure analysis for the 2026 season, identify and provide detailed information about five specific NFL stadiums:

1. The stadium that hosted the 2026 AFC Championship Game between the New England Patriots and Denver Broncos on January 25, 2026
2. The stadium that hosted Super Bowl LX on February 8, 2026
3. The NFL stadium with the largest seating capacity
4. The NFL stadium with the second-largest seating capacity
5. The NFL stadium with the third-largest seating capacity

For each stadium, provide:
- The official stadium name
- Seating capacity
- Location (city and state)
- Field surface type (natural grass or artificial turf type)
- Home team(s)

All information must be based on the 2025-2026 NFL season data and should reflect the standard/listed seating capacities (not expanded capacities for special events).
""".strip()


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class StadiumInfo(BaseModel):
    """Information block for a single stadium."""
    name: Optional[str] = None
    capacity: Optional[str] = None  # Keep as string to allow ranges or commas
    city: Optional[str] = None
    state: Optional[str] = None
    surface: Optional[str] = None
    home_teams: List[str] = Field(default_factory=list)
    sources: List[str] = Field(default_factory=list)


class StadiumsExtraction(BaseModel):
    """All five stadiums required by the task."""
    afc_championship: Optional[StadiumInfo] = None
    super_bowl_lx: Optional[StadiumInfo] = None
    largest_capacity: Optional[StadiumInfo] = None
    second_largest_capacity: Optional[StadiumInfo] = None
    third_largest_capacity: Optional[StadiumInfo] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_stadiums() -> str:
    return """
    Extract structured information for exactly five NFL stadiums referenced in the answer text, corresponding to the following categories:
    - afc_championship: The stadium that hosted the 2026 AFC Championship Game (New England Patriots vs Denver Broncos) on January 25, 2026.
    - super_bowl_lx: The stadium that hosted Super Bowl LX on February 8, 2026.
    - largest_capacity: The NFL stadium with the largest listed standard seating capacity (do not use expanded/standing-room capacities).
    - second_largest_capacity: The NFL stadium with the second-largest listed standard seating capacity.
    - third_largest_capacity: The NFL stadium with the third-largest listed standard seating capacity.

    For each of the five stadiums, extract the following fields from the answer text exactly as stated:
    - name: Official stadium name.
    - capacity: The standard/listed seating capacity number as presented (keep as text, do not parse to a number; keep commas if present).
    - city: The city or municipality (e.g., "Arlington").
    - state: The U.S. state (two-letter abbreviation or full name as presented).
    - surface: The playing surface used during the 2025–2026 NFL season (e.g., "natural grass", "Bermuda grass", "FieldTurf", "Matrix Turf", etc.).
    - home_teams: Array of the NFL home team names (e.g., ["Dallas Cowboys"] or ["New York Giants", "New York Jets"]).
    - sources: An array of all explicit URLs cited in the answer that support any of the above facts for this specific stadium. Include only URLs that actually appear in the answer (plain URLs or markdown links). If no URLs are present for the stadium, return an empty array.

    Important rules:
    - Only extract information explicitly present in the answer. Do not invent or infer missing fields.
    - The capacity must reflect the standard/listed capacity, not expanded/standing-room numbers for special events.
    - If a field is not mentioned for a stadium, set it to null (or an empty array for home_teams/sources).
    - Return a single JSON object with exactly five top-level keys: afc_championship, super_bowl_lx, largest_capacity, second_largest_capacity, third_largest_capacity.
    - Each value must be an object with the fields: name, capacity, city, state, surface, home_teams, sources.
    """.strip()


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def teams_to_str(teams: List[str]) -> str:
    if not teams:
        return ""
    return ", ".join(teams)


def location_to_str(city: Optional[str], state: Optional[str]) -> str:
    city = city or ""
    state = state or ""
    if city and state:
        return f"{city}, {state}"
    return city or state


def ensure_sources_present(info: Optional[StadiumInfo]) -> bool:
    return bool(info and isinstance(info.sources, list) and len(info.sources) > 0)


# --------------------------------------------------------------------------- #
# Verification builders                                                       #
# --------------------------------------------------------------------------- #
async def build_and_verify_afc_group(evaluator: Evaluator, parent, info: Optional[StadiumInfo]) -> None:
    group = evaluator.add_parallel(
        id="AFC_Championship_Host_Stadium",
        desc="Information about the stadium that hosted the 2026 AFC Championship Game on January 25, 2026",
        parent=parent,
        critical=False
    )

    # Critical gate: at least one supporting URL for this stadium's info
    evaluator.add_custom_node(
        result=ensure_sources_present(info),
        id="AFC_Sources_Provided",
        desc="AFC Championship stadium: at least one source URL is provided",
        parent=group,
        critical=True
    )

    name_node = evaluator.add_leaf(
        id="AFC_Stadium_Name",
        desc="Correctly identify the official name of the stadium that hosted the 2026 AFC Championship Game",
        parent=group,
        critical=True
    )
    capacity_node = evaluator.add_leaf(
        id="AFC_Stadium_Capacity",
        desc="Provide the correct seating capacity of the AFC Championship host stadium",
        parent=group,
        critical=True
    )
    location_node = evaluator.add_leaf(
        id="AFC_Stadium_Location",
        desc="Provide the correct location (city and state) of the AFC Championship host stadium",
        parent=group,
        critical=True
    )
    surface_node = evaluator.add_leaf(
        id="AFC_Stadium_Field_Surface",
        desc="Provide the correct field surface type of the AFC Championship host stadium",
        parent=group,
        critical=True
    )
    home_node = evaluator.add_leaf(
        id="AFC_Stadium_Home_Team",
        desc="Identify the home team(s) of the AFC Championship host stadium",
        parent=group,
        critical=True
    )

    # Prepare claims
    name_val = info.name if info else ""
    capacity_val = info.capacity if info else ""
    loc_val = location_to_str(info.city if info else None, info.state if info else None)
    surface_val = info.surface if info else ""
    teams_val = teams_to_str(info.home_teams if info else [])

    claims = [
        (
            f"The stadium that hosted the 2026 AFC Championship Game on January 25, 2026 (New England Patriots vs Denver Broncos) was '{name_val}'.",
            info.sources if info else [],
            name_node,
            "Verify with the cited sources that this specific game was hosted at the named stadium. Prioritize official NFL game summaries, credible news outlets, or the stadium/teams' official sites."
        ),
        (
            f"The listed standard seating capacity of '{name_val}' (not expanded/standing-room) is '{capacity_val}'.",
            info.sources if info else [],
            capacity_node,
            "Confirm the standard/listed seating capacity number for the 2025–2026 season; do not use expanded capacities posted for special events."
        ),
        (
            f"The stadium '{name_val}' is located in {loc_val}.",
            info.sources if info else [],
            location_node,
            "Accept reasonable format variants (e.g., municipality vs. larger metro area) but ensure the city and state match the stadium's official location."
        ),
        (
            f"During the 2025–2026 NFL season, the field surface at '{name_val}' was '{surface_val}'.",
            info.sources if info else [],
            surface_node,
            "Check the surface type used that season (e.g., natural grass or a specific turf brand). Allow minor naming variations."
        ),
        (
            f"During the 2025–2026 NFL season, the home NFL team(s) for '{name_val}' were: {teams_val}.",
            info.sources if info else [],
            home_node,
            "Verify the home NFL team(s) for the stadium; some stadiums host two NFL teams."
        ),
    ]
    await evaluator.batch_verify(claims)


async def build_and_verify_super_bowl_group(evaluator: Evaluator, parent, info: Optional[StadiumInfo]) -> None:
    group = evaluator.add_parallel(
        id="Super_Bowl_LX_Host_Stadium",
        desc="Information about the stadium that hosted Super Bowl LX on February 8, 2026",
        parent=parent,
        critical=False
    )

    evaluator.add_custom_node(
        result=ensure_sources_present(info),
        id="Super_Bowl_Sources_Provided",
        desc="Super Bowl LX stadium: at least one source URL is provided",
        parent=group,
        critical=True
    )

    name_node = evaluator.add_leaf(
        id="Super_Bowl_Stadium_Name",
        desc="Correctly identify the official name of the stadium that hosted Super Bowl LX",
        parent=group,
        critical=True
    )
    capacity_node = evaluator.add_leaf(
        id="Super_Bowl_Stadium_Capacity",
        desc="Provide the correct standard seating capacity of the Super Bowl LX host stadium (not expanded capacity)",
        parent=group,
        critical=True
    )
    location_node = evaluator.add_leaf(
        id="Super_Bowl_Stadium_Location",
        desc="Provide the correct location (city and state) of the Super Bowl LX host stadium",
        parent=group,
        critical=True
    )
    surface_node = evaluator.add_leaf(
        id="Super_Bowl_Stadium_Field_Surface",
        desc="Provide the correct field surface type of the Super Bowl LX host stadium",
        parent=group,
        critical=True
    )
    home_node = evaluator.add_leaf(
        id="Super_Bowl_Stadium_Home_Team",
        desc="Identify the home team(s) of the Super Bowl LX host stadium",
        parent=group,
        critical=True
    )

    name_val = info.name if info else ""
    capacity_val = info.capacity if info else ""
    loc_val = location_to_str(info.city if info else None, info.state if info else None)
    surface_val = info.surface if info else ""
    teams_val = teams_to_str(info.home_teams if info else [])

    claims = [
        (
            f"The stadium that hosted Super Bowl LX on February 8, 2026 was '{name_val}'.",
            info.sources if info else [],
            name_node,
            "Confirm via the cited sources that Super Bowl LX took place at the named stadium. Prefer official NFL sources, credible news outlets, or the stadium/teams' official sites."
        ),
        (
            f"The listed standard seating capacity of '{name_val}' (not expanded/standing-room) is '{capacity_val}'.",
            info.sources if info else [],
            capacity_node,
            "Confirm the standard/listed seating capacity; do not use expanded capacities for special events."
        ),
        (
            f"The stadium '{name_val}' is located in {loc_val}.",
            info.sources if info else [],
            location_node,
            "Ensure the city and state match the stadium's official location."
        ),
        (
            f"During the 2025–2026 NFL season, the field surface at '{name_val}' was '{surface_val}'.",
            info.sources if info else [],
            surface_node,
            "Check the surface type for that season. Allow minor naming variations."
        ),
        (
            f"During the 2025–2026 NFL season, the home NFL team(s) for '{name_val}' were: {teams_val}.",
            info.sources if info else [],
            home_node,
            "Verify the home NFL team(s) for the stadium."
        ),
    ]
    await evaluator.batch_verify(claims)


async def build_and_verify_largest_group(evaluator: Evaluator, parent, info: Optional[StadiumInfo]) -> None:
    group = evaluator.add_parallel(
        id="NFL_Largest_Capacity_Stadium",
        desc="Information about the NFL stadium with the highest listed seating capacity",
        parent=parent,
        critical=False
    )

    evaluator.add_custom_node(
        result=ensure_sources_present(info),
        id="Largest_Sources_Provided",
        desc="Largest-capacity stadium: at least one source URL is provided",
        parent=group,
        critical=True
    )

    name_node = evaluator.add_leaf(
        id="Largest_Stadium_Name",
        desc="Correctly identify the official name of the NFL stadium with the highest listed seating capacity",
        parent=group,
        critical=True
    )
    capacity_node = evaluator.add_leaf(
        id="Largest_Stadium_Capacity",
        desc="Provide the correct seating capacity number of the largest NFL stadium",
        parent=group,
        critical=True
    )
    location_node = evaluator.add_leaf(
        id="Largest_Stadium_Location",
        desc="Provide the correct location (city and state) of the largest NFL stadium",
        parent=group,
        critical=True
    )
    surface_node = evaluator.add_leaf(
        id="Largest_Stadium_Field_Surface",
        desc="Provide the correct field surface type of the largest NFL stadium",
        parent=group,
        critical=True
    )
    home_node = evaluator.add_leaf(
        id="Largest_Stadium_Home_Teams",
        desc="Identify the home team(s) of the largest NFL stadium",
        parent=group,
        critical=True
    )

    name_val = info.name if info else ""
    capacity_val = info.capacity if info else ""
    loc_val = location_to_str(info.city if info else None, info.state if info else None)
    surface_val = info.surface if info else ""
    teams_val = teams_to_str(info.home_teams if info else [])

    claims = [
        (
            f"The NFL stadium with the highest listed standard seating capacity (not expanded) is '{name_val}'.",
            info.sources if info else [],
            name_node,
            "Confirm using reputable lists or official sources that among NFL home stadiums, this stadium ranks first by standard listed capacity."
        ),
        (
            f"The listed standard seating capacity of '{name_val}' is '{capacity_val}'.",
            info.sources if info else [],
            capacity_node,
            "Verify the standard/listed capacity for the 2025–2026 season; ignore expanded/standing-room figures."
        ),
        (
            f"The stadium '{name_val}' is located in {loc_val}.",
            info.sources if info else [],
            location_node,
            "Verify city and state for the stadium."
        ),
        (
            f"During the 2025–2026 NFL season, the field surface at '{name_val}' was '{surface_val}'.",
            info.sources if info else [],
            surface_node,
            "Verify playing surface for that season."
        ),
        (
            f"During the 2025–2026 NFL season, the home NFL team(s) for '{name_val}' were: {teams_val}.",
            info.sources if info else [],
            home_node,
            "Verify home NFL team(s) for the stadium."
        ),
    ]
    await evaluator.batch_verify(claims)


async def build_and_verify_second_group(evaluator: Evaluator, parent, info: Optional[StadiumInfo]) -> None:
    group = evaluator.add_parallel(
        id="NFL_Second_Largest_Capacity_Stadium",
        desc="Information about the NFL stadium with the second-highest listed seating capacity",
        parent=parent,
        critical=False
    )

    evaluator.add_custom_node(
        result=ensure_sources_present(info),
        id="Second_Largest_Sources_Provided",
        desc="Second-largest-capacity stadium: at least one source URL is provided",
        parent=group,
        critical=True
    )

    name_node = evaluator.add_leaf(
        id="Second_Largest_Stadium_Name",
        desc="Correctly identify the official name of the NFL stadium with the second-highest listed seating capacity",
        parent=group,
        critical=True
    )
    capacity_node = evaluator.add_leaf(
        id="Second_Largest_Stadium_Capacity",
        desc="Provide the correct seating capacity number of the second-largest NFL stadium",
        parent=group,
        critical=True
    )
    location_node = evaluator.add_leaf(
        id="Second_Largest_Stadium_Location",
        desc="Provide the correct location (city and state) of the second-largest NFL stadium",
        parent=group,
        critical=True
    )
    surface_node = evaluator.add_leaf(
        id="Second_Largest_Stadium_Field_Surface",
        desc="Provide the correct field surface type of the second-largest NFL stadium",
        parent=group,
        critical=True
    )
    home_node = evaluator.add_leaf(
        id="Second_Largest_Stadium_Home_Team",
        desc="Identify the home team of the second-largest NFL stadium",
        parent=group,
        critical=True
    )

    name_val = info.name if info else ""
    capacity_val = info.capacity if info else ""
    loc_val = location_to_str(info.city if info else None, info.state if info else None)
    surface_val = info.surface if info else ""
    teams_val = teams_to_str(info.home_teams if info else [])

    claims = [
        (
            f"The NFL stadium with the second-highest listed standard seating capacity (not expanded) is '{name_val}'.",
            info.sources if info else [],
            name_node,
            "Confirm using reputable lists or official sources that among NFL home stadiums, this stadium ranks second by standard listed capacity."
        ),
        (
            f"The listed standard seating capacity of '{name_val}' is '{capacity_val}'.",
            info.sources if info else [],
            capacity_node,
            "Verify the standard/listed capacity for the 2025–2026 season; ignore expanded/standing-room figures."
        ),
        (
            f"The stadium '{name_val}' is located in {loc_val}.",
            info.sources if info else [],
            location_node,
            "Verify city and state for the stadium."
        ),
        (
            f"During the 2025–2026 NFL season, the field surface at '{name_val}' was '{surface_val}'.",
            info.sources if info else [],
            surface_node,
            "Verify playing surface for that season."
        ),
        (
            f"During the 2025–2026 NFL season, the home NFL team(s) for '{name_val}' were: {teams_val}.",
            info.sources if info else [],
            home_node,
            "Verify the NFL home team(s) for the stadium (usually one team)."
        ),
    ]
    await evaluator.batch_verify(claims)


async def build_and_verify_third_group(evaluator: Evaluator, parent, info: Optional[StadiumInfo]) -> None:
    group = evaluator.add_parallel(
        id="NFL_Third_Largest_Capacity_Stadium",
        desc="Information about the NFL stadium with the third-highest listed seating capacity",
        parent=parent,
        critical=False
    )

    evaluator.add_custom_node(
        result=ensure_sources_present(info),
        id="Third_Largest_Sources_Provided",
        desc="Third-largest-capacity stadium: at least one source URL is provided",
        parent=group,
        critical=True
    )

    name_node = evaluator.add_leaf(
        id="Third_Largest_Stadium_Name",
        desc="Correctly identify the official name of the NFL stadium with the third-highest listed seating capacity",
        parent=group,
        critical=True
    )
    capacity_node = evaluator.add_leaf(
        id="Third_Largest_Stadium_Capacity",
        desc="Provide the correct seating capacity number of the third-largest NFL stadium",
        parent=group,
        critical=True
    )
    location_node = evaluator.add_leaf(
        id="Third_Largest_Stadium_Location",
        desc="Provide the correct location (city and state) of the third-largest NFL stadium",
        parent=group,
        critical=True
    )
    surface_node = evaluator.add_leaf(
        id="Third_Largest_Stadium_Field_Surface",
        desc="Provide the correct field surface type of the third-largest NFL stadium",
        parent=group,
        critical=True
    )
    home_node = evaluator.add_leaf(
        id="Third_Largest_Stadium_Home_Team",
        desc="Identify the home team of the third-largest NFL stadium",
        parent=group,
        critical=True
    )

    name_val = info.name if info else ""
    capacity_val = info.capacity if info else ""
    loc_val = location_to_str(info.city if info else None, info.state if info else None)
    surface_val = info.surface if info else ""
    teams_val = teams_to_str(info.home_teams if info else [])

    claims = [
        (
            f"The NFL stadium with the third-highest listed standard seating capacity (not expanded) is '{name_val}'.",
            info.sources if info else [],
            name_node,
            "Confirm using reputable lists or official sources that among NFL home stadiums, this stadium ranks third by standard listed capacity."
        ),
        (
            f"The listed standard seating capacity of '{name_val}' is '{capacity_val}'.",
            info.sources if info else [],
            capacity_node,
            "Verify the standard/listed capacity for the 2025–2026 season; ignore expanded/standing-room figures."
        ),
        (
            f"The stadium '{name_val}' is located in {loc_val}.",
            info.sources if info else [],
            location_node,
            "Verify city and state for the stadium."
        ),
        (
            f"During the 2025–2026 NFL season, the field surface at '{name_val}' was '{surface_val}'.",
            info.sources if info else [],
            surface_node,
            "Verify playing surface for that season."
        ),
        (
            f"During the 2025–2026 NFL season, the home NFL team(s) for '{name_val}' were: {teams_val}.",
            info.sources if info else [],
            home_node,
            "Verify the NFL home team(s) for the stadium."
        ),
    ]
    await evaluator.batch_verify(claims)


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
    Evaluate an answer for the 2026 NFL stadiums task using the Mind2Web2 framework.
    """
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Overall, items are independent
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

    # Note: The provided JSON marks the overall node as critical; however,
    # Mind2Web2 enforces that a critical parent must have all critical children.
    # To allow partial credit across the five stadium blocks (as in the JSON),
    # we keep the root non-critical and set criticality at the leaf level.

    # Extract structured stadium info from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_stadiums(),
        template_class=StadiumsExtraction,
        extraction_name="stadiums_extraction"
    )

    # Build verification groups corresponding to the rubric tree
    await build_and_verify_afc_group(evaluator, root, extracted.afc_championship)
    await build_and_verify_super_bowl_group(evaluator, root, extracted.super_bowl_lx)
    await build_and_verify_largest_group(evaluator, root, extracted.largest_capacity)
    await build_and_verify_second_group(evaluator, root, extracted.second_largest_capacity)
    await build_and_verify_third_group(evaluator, root, extracted.third_largest_capacity)

    # Optional: record brief custom info for transparency
    evaluator.add_custom_info(
        info={"note": "Capacities must reflect standard/listed seating for the 2025–2026 NFL season (not expanded)."},
        info_type="policy_note",
        info_name="evaluation_policy"
    )

    return evaluator.get_summary()