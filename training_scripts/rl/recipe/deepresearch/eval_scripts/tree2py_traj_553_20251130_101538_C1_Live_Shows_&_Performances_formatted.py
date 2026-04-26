import asyncio
import logging
from typing import Any, List, Optional, Dict

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


TASK_ID = "nfl_thanksgiving_2025_halftime"
TASK_DESCRIPTION = """
Among the performers who appeared at the 2025 NFL Thanksgiving halftime shows, identify the one who was born in the same city as the home location of the team they performed for. Provide the name of the stadium where this performer's halftime show took place and the stadium's seating capacity.
"""


class PerformerVenue(BaseModel):
    performer_name: Optional[str] = None
    team_name: Optional[str] = None
    team_city: Optional[str] = None
    birth_city: Optional[str] = None

    halftime_sources: List[str] = Field(default_factory=list)
    birth_city_sources: List[str] = Field(default_factory=list)
    team_city_sources: List[str] = Field(default_factory=list)

    stadium_name: Optional[str] = None
    stadium_capacity: Optional[str] = None
    stadium_sources: List[str] = Field(default_factory=list)


class ThanksgivingHalftimeExtraction(BaseModel):
    candidates: List[PerformerVenue] = Field(default_factory=list)


def prompt_extract_thanksgiving_halftime() -> str:
    return """
    Extract all performers that the answer claims appeared at NFL Thanksgiving halftime shows in 2025.
    For each performer, return a JSON object with fields:

    - performer_name: The performer’s full name as provided.
    - team_name: The NFL team for whom they performed the Thanksgiving halftime show in 2025.
    - team_city: The city where this team plays its home games (the team’s home city).
    - birth_city: The performer’s birth city (city only; if a state or country is included, keep the full text as-is).
    - halftime_sources: All URLs cited that support the fact that this performer appeared at a 2025 NFL Thanksgiving halftime show (e.g., team press releases, NFL announcements, reputable news articles).
    - birth_city_sources: All URLs cited that support the performer’s birth city (e.g., Wikipedia, official bio).
    - team_city_sources: All URLs cited that support the team’s home city (e.g., team official site, Wikipedia).
    - stadium_name: The stadium name where that Thanksgiving halftime show took place (if provided).
    - stadium_capacity: The seating capacity of that stadium (as provided in the answer; keep it as a string, e.g., "80,000", "80,000–85,000", etc.).
    - stadium_sources: All URLs cited that support the stadium name and/or capacity (e.g., stadium official site, Wikipedia, team site, game recap).

    Return a JSON object with a single key "candidates" whose value is an array of such objects.
    If any field is missing in the answer for a given performer, set it to null (for strings) or [] (for lists).
    Only include URLs explicitly present in the answer. Do not invent any URLs.
    """


def _normalize_city_name(city: Optional[str]) -> Optional[str]:
    if city is None:
        return None
    s = city.strip().lower()
    # If formatted like "Detroit, Michigan", keep "Detroit" for equality check while preserving full text for claims
    if "," in s:
        s = s.split(",")[0].strip()
    # Remove common prefixes
    for prefix in ["city of ", "the city of "]:
        if s.startswith(prefix):
            s = s[len(prefix):].strip()
    return s


def select_matching_candidate(extraction: ThanksgivingHalftimeExtraction) -> Optional[PerformerVenue]:
    """
    Select the first candidate whose birth_city matches team_city (city-level, case-insensitive).
    If none matches, return the first candidate (to allow verification to proceed and fail gracefully).
    """
    for c in extraction.candidates:
        bc = _normalize_city_name(c.birth_city)
        tc = _normalize_city_name(c.team_city)
        if bc and tc and bc == tc:
            return c
    return extraction.candidates[0] if extraction.candidates else None


async def verify_selected_performer_and_venue(
    evaluator: Evaluator,
    root_node,
    selected: Optional[PerformerVenue],
) -> None:
    """
    Build the verification tree and run checks based on the selected candidate.
    """
    # Child 1: Identify correct performer (parallel, critical)
    identify_node = evaluator.add_parallel(
        id="identify_correct_performer",
        desc="Select a performer who appeared at a 2025 NFL Thanksgiving halftime show and who was born in the same city as the home location of the team they performed for",
        parent=root_node,
        critical=True,
    )

    # Leaf: performed_at_2025_thanksgiving_halftime
    performed_leaf = evaluator.add_leaf(
        id="performed_at_2025_thanksgiving_halftime",
        desc="The identified performer did perform at an NFL Thanksgiving halftime show in 2025",
        parent=identify_node,
        critical=True,
    )

    performer_name = selected.performer_name if selected else ""
    team_name = selected.team_name if selected else ""
    halftime_sources = (selected.halftime_sources if selected else []) or []

    performed_claim = (
        f"{performer_name} performed at a 2025 NFL Thanksgiving halftime show for the {team_name}."
        if performer_name and team_name
        else "The identified performer performed at a 2025 NFL Thanksgiving halftime show."
    )
    await evaluator.verify(
        claim=performed_claim,
        node=performed_leaf,
        sources=halftime_sources,
        additional_instruction=(
            "Verify that this performer appeared specifically at an NFL Thanksgiving (late November 2025) halftime show, "
            "not just any regular-season halftime. Use event announcements or game recaps when available."
        ),
    )

    # Leaf: birth_city_matches_team_home_city
    city_match_leaf = evaluator.add_leaf(
        id="birth_city_matches_team_home_city",
        desc="The identified performer’s birth city matches the city where the relevant team plays its home games (same city)",
        parent=identify_node,
        critical=True,
    )

    birth_city = selected.birth_city if selected else ""
    team_city = selected.team_city if selected else ""

    city_match_claim = (
        f"The birth city of {performer_name} and the home city of the {team_name} refer to the same city: "
        f"'{birth_city}' vs '{team_city}'."
    )
    await evaluator.verify(
        claim=city_match_claim,
        node=city_match_leaf,
        additional_instruction=(
            "Judge whether the two city names refer to the same city. Allow reasonable formatting differences such as "
            "including a state ('Detroit' vs 'Detroit, Michigan') or minor variations in phrasing. Consider them the same if "
            "they clearly denote the same city."
        ),
    )

    # Child 2: Venue details (parallel, critical)
    venue_node = evaluator.add_parallel(
        id="venue_details_for_that_halftime_show",
        desc="Provide venue details for the stadium where that identified performer’s halftime show took place",
        parent=root_node,
        critical=True,
    )

    # Leaf: stadium_name
    stadium_name_leaf = evaluator.add_leaf(
        id="stadium_name",
        desc="Provide the stadium name where the identified performer’s 2025 Thanksgiving halftime show took place",
        parent=venue_node,
        critical=True,
    )

    stadium_name = selected.stadium_name if selected else ""
    stadium_sources = (selected.stadium_sources if selected else []) or []
    combined_sources_for_stadium_name = stadium_sources + halftime_sources

    stadium_name_claim = (
        f"The 2025 NFL Thanksgiving halftime show featuring {performer_name} took place at {stadium_name}."
        if performer_name and stadium_name
        else f"The halftime show took place at {stadium_name}."
    )
    await evaluator.verify(
        claim=stadium_name_claim,
        node=stadium_name_leaf,
        sources=combined_sources_for_stadium_name if combined_sources_for_stadium_name else None,
        additional_instruction=(
            "Verify the game venue/stadium for the Thanksgiving game where the performer appeared at halftime. "
            "If an event recap or the team's official announcement states the venue, that suffices."
        ),
    )

    # Leaf: stadium_seating_capacity
    stadium_capacity_leaf = evaluator.add_leaf(
        id="stadium_seating_capacity",
        desc="Provide the seating capacity of that stadium",
        parent=venue_node,
        critical=True,
    )

    stadium_capacity = selected.stadium_capacity if selected else ""
    stadium_capacity_claim = (
        f"The seating capacity of {stadium_name} is {stadium_capacity}."
        if stadium_name and stadium_capacity
        else f"The stadium’s seating capacity is {stadium_capacity}."
    )
    await evaluator.verify(
        claim=stadium_capacity_claim,
        node=stadium_capacity_leaf,
        sources=stadium_sources if stadium_sources else None,
        additional_instruction=(
            "Verify the stated seating capacity figure for the stadium from reliable sources (e.g., stadium official site or Wikipedia). "
            "Allow reasonable variations due to configurable seating (e.g., ranges or approximations)."
        ),
    )


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
    Entry point to evaluate an answer for the 2025 NFL Thanksgiving halftime performer and venue details task.
    """
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

    extraction = await evaluator.extract(
        prompt=prompt_extract_thanksgiving_halftime(),
        template_class=ThanksgivingHalftimeExtraction,
        extraction_name="thanksgiving_halftime_candidates",
    )

    selected = select_matching_candidate(extraction)

    evaluator.add_custom_info(
        info={
            "selected_candidate": selected.dict() if selected else None,
            "total_candidates_extracted": len(extraction.candidates),
        },
        info_type="selection_summary",
    )

    await verify_selected_performer_and_venue(evaluator, root, selected)

    return evaluator.get_summary()