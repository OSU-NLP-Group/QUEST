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
TASK_ID = "ohsaa_d1_2025_occ_central"
TASK_DESCRIPTION = """
In the 2025 OHSAA Division I state football championship, one of the participating schools was a member of the Ohio Capital Conference Central Division. Identify this school and provide the following information about their championship season: (1) their final season record, (2) the opponent they faced in the championship game, (3) the final score of that championship game, (4) the city in Ohio where the championship game was played, (5) the name of the stadium where the game was held, and (6) the seating capacity of that stadium.
"""


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class ChampionshipInfo(BaseModel):
    # Core information extracted from the answer
    occ_central_school: Optional[str] = None
    final_record: Optional[str] = None
    opponent: Optional[str] = None
    final_score: Optional[str] = None
    game_city: Optional[str] = None
    stadium_name: Optional[str] = None
    stadium_capacity: Optional[str] = None

    # Sources: general and per-field (URLs only, as cited in the answer)
    sources: List[str] = Field(default_factory=list)
    occ_central_school_sources: List[str] = Field(default_factory=list)
    final_record_sources: List[str] = Field(default_factory=list)
    opponent_sources: List[str] = Field(default_factory=list)
    final_score_sources: List[str] = Field(default_factory=list)
    game_city_sources: List[str] = Field(default_factory=list)
    stadium_name_sources: List[str] = Field(default_factory=list)
    stadium_capacity_sources: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_championship_info() -> str:
    return """
    Extract from the answer the specific information about the 2025 OHSAA Division I state football championship, focusing on the school from the Ohio Capital Conference (OCC) Central Division and the requested game details.

    You must extract the following fields (return null if missing):
    - occ_central_school: The high school from the OCC Central Division identified by the answer as relevant to the 2025 Division I state championship (preferably the champion if the answer says so).
    - final_record: This school's final season record (e.g., "15-1" or similar string).
    - opponent: The team that this school faced in the 2025 Division I championship game.
    - final_score: The final score of that championship game (e.g., "28-14"; accept "28 to 14" style in the answer).
    - game_city: The Ohio city where the championship game was played (e.g., "Canton").
    - stadium_name: The name of the stadium where the game was held.
    - stadium_capacity: The seating capacity of that stadium (string; retain commas if present).

    Also extract URL sources explicitly cited in the answer:
    - sources: All URLs in the answer relevant to any of the above.
    - occ_central_school_sources: URLs supporting the identified school AND its OCC Central Division membership and/or championship result.
    - final_record_sources: URLs supporting the final season record.
    - opponent_sources: URLs supporting the opponent identification.
    - final_score_sources: URLs supporting the final score.
    - game_city_sources: URLs supporting the city where the game was played.
    - stadium_name_sources: URLs supporting the stadium name.
    - stadium_capacity_sources: URLs supporting the stadium's seating capacity.

    URL extraction rules:
    - Extract only URLs that appear in the answer (plain links or markdown links).
    - Do not invent or infer URLs.
    - If a URL is missing protocol, prepend "http://".
    - If no URLs are given for a category, return an empty array for that category.

    Ensure all fields are strings (or null) and all URL lists are arrays of URL strings.
    """


# --------------------------------------------------------------------------- #
# Helpers                                                                     #
# --------------------------------------------------------------------------- #
def combine_sources(*lists: Optional[List[str]]) -> List[str]:
    """Combine multiple lists of URLs, deduplicate, and keep order."""
    out: List[str] = []
    seen = set()
    for lst in lists:
        if not lst:
            continue
        for url in lst:
            if not isinstance(url, str):
                continue
            u = url.strip()
            if u and u not in seen:
                out.append(u)
                seen.add(u)
    return out


# --------------------------------------------------------------------------- #
# Verification builders                                                       #
# --------------------------------------------------------------------------- #
async def verify_school_identification(evaluator: Evaluator, parent, info: ChampionshipInfo) -> None:
    """
    Build verification nodes for identifying the OCC Central Division school that won the
    2025 OHSAA Division I championship (as the rubric specifies). This block is critical.
    """
    group = evaluator.add_sequential(
        id="Winning_School_from_OCC_Central_group",
        desc="School identification and OCC Central Division membership verification",
        parent=parent,
        critical=False  # Group non-critical; the key leaf inside is critical as per rubric
    )

    # Existence check (critical within the group)
    school_sources = combine_sources(info.occ_central_school_sources, info.sources)
    provided = bool(info.occ_central_school and info.occ_central_school.strip()) and len(school_sources) > 0
    evaluator.add_custom_node(
        result=provided,
        id="Winning_School_from_OCC_Central_provided",
        desc="School name and at least one supporting source are provided",
        parent=group,
        critical=True
    )

    # Main verification leaf (critical)
    school_leaf = evaluator.add_leaf(
        id="Winning_School_from_OCC_Central",
        desc="Correctly identifies the high school from the Ohio Capital Conference Central Division that won the 2025 OHSAA Division I state football championship",
        parent=group,
        critical=True
    )

    school_name = info.occ_central_school or ""
    claim = f"The 2025 OHSAA Division I state football champion that is a member of the Ohio Capital Conference Central Division is {school_name}."
    await evaluator.verify(
        claim=claim,
        node=school_leaf,
        sources=school_sources,
        additional_instruction=(
            "To pass, the evidence must show BOTH: "
            f"(a) that {school_name} won the 2025 OHSAA Division I state football championship, and "
            "(b) that the school is a member of the Ohio Capital Conference (OCC) Central Division. "
            "Allow minor naming variants (e.g., 'H.S.' for High School)."
        )
    )


async def verify_final_record(evaluator: Evaluator, parent, info: ChampionshipInfo) -> None:
    group = evaluator.add_sequential(
        id="Final_Season_Record_group",
        desc="Final season record verification",
        parent=parent,
        critical=False
    )
    record_sources = combine_sources(info.final_record_sources, info.occ_central_school_sources, info.sources)
    provided = bool(info.final_record and info.final_record.strip()) and len(record_sources) > 0
    evaluator.add_custom_node(
        result=provided,
        id="Final_Season_Record_provided",
        desc="Final season record and at least one supporting source are provided",
        parent=group,
        critical=True
    )

    leaf = evaluator.add_leaf(
        id="Final_Season_Record",
        desc="Provides the accurate final season record of the championship-winning school",
        parent=group,
        critical=False
    )

    school_name = info.occ_central_school or "the school"
    record_str = info.final_record or ""
    claim = f"The final season record for {school_name} in 2025 was {record_str}."
    await evaluator.verify(
        claim=claim,
        node=leaf,
        sources=record_sources,
        additional_instruction=(
            "Verify the overall final record for the season (including all games). "
            "Accept minor formatting variations such as different dashes or parentheses. "
            "If multiple records are shown (e.g., league vs overall), confirm the overall final record."
        )
    )


async def verify_opponent(evaluator: Evaluator, parent, info: ChampionshipInfo) -> None:
    group = evaluator.add_sequential(
        id="Championship_Opponent_group",
        desc="Championship opponent verification",
        parent=parent,
        critical=False
    )
    opp_sources = combine_sources(info.opponent_sources, info.occ_central_school_sources, info.sources)
    provided = bool(info.opponent and info.opponent.strip()) and len(opp_sources) > 0
    evaluator.add_custom_node(
        result=provided,
        id="Championship_Opponent_provided",
        desc="Opponent and at least one supporting source are provided",
        parent=group,
        critical=True
    )

    leaf = evaluator.add_leaf(
        id="Championship_Opponent",
        desc="Correctly identifies the opponent team that the winning school faced in the championship game",
        parent=group,
        critical=False
    )

    school_name = info.occ_central_school or "the school"
    opponent_name = info.opponent or ""
    claim = f"The opponent that {school_name} faced in the 2025 OHSAA Division I state championship game was {opponent_name}."
    await evaluator.verify(
        claim=claim,
        node=leaf,
        sources=opp_sources,
        additional_instruction=(
            "Verify the opponent in the 2025 Division I state championship game. "
            "Allow phrasing like 'vs', 'played', 'faced'."
        )
    )


async def verify_final_score(evaluator: Evaluator, parent, info: ChampionshipInfo) -> None:
    group = evaluator.add_sequential(
        id="Championship_Game_Score_group",
        desc="Championship game final score verification",
        parent=parent,
        critical=False
    )
    score_sources = combine_sources(info.final_score_sources, info.occ_central_school_sources, info.sources)
    provided = bool(info.final_score and info.final_score.strip()) and len(score_sources) > 0
    evaluator.add_custom_node(
        result=provided,
        id="Championship_Game_Score_provided",
        desc="Final score and at least one supporting source are provided",
        parent=group,
        critical=True
    )

    leaf = evaluator.add_leaf(
        id="Championship_Game_Score",
        desc="States the correct final score of the championship game",
        parent=group,
        critical=False
    )

    school_name = info.occ_central_school or "the school"
    opponent_name = info.opponent or "the opponent"
    score_str = info.final_score or ""
    claim = f"The final score of the 2025 OHSAA Division I state championship game between {school_name} and {opponent_name} was {score_str}."
    await evaluator.verify(
        claim=claim,
        node=leaf,
        sources=score_sources,
        additional_instruction=(
            "Verify the final score of the game. "
            "Allow score formatting variations such as '28-14' vs '28 to 14'. "
            "Consider either team-first ordering acceptable as long as the point totals match."
        )
    )


async def verify_game_city(evaluator: Evaluator, parent, info: ChampionshipInfo) -> None:
    group = evaluator.add_sequential(
        id="Game_Location_City_group",
        desc="Championship game city verification",
        parent=parent,
        critical=False
    )
    city_sources = combine_sources(info.game_city_sources, info.stadium_name_sources, info.sources)
    provided = bool(info.game_city and info.game_city.strip()) and len(city_sources) > 0
    evaluator.add_custom_node(
        result=provided,
        id="Game_Location_City_provided",
        desc="Game city and at least one supporting source are provided",
        parent=group,
        critical=True
    )

    leaf = evaluator.add_leaf(
        id="Game_Location_City",
        desc="Identifies the city in Ohio where the championship game was played",
        parent=group,
        critical=False
    )

    city = info.game_city or ""
    claim = f"The 2025 OHSAA Division I state championship game was played in {city}, Ohio."
    await evaluator.verify(
        claim=claim,
        node=leaf,
        sources=city_sources,
        additional_instruction=(
            "The city may be mentioned alongside the stadium (e.g., 'in Canton at Tom Benson Hall of Fame Stadium'). "
            "Accept if the source clearly indicates the game location city in Ohio."
        )
    )


async def verify_stadium_name(evaluator: Evaluator, parent, info: ChampionshipInfo) -> None:
    group = evaluator.add_sequential(
        id="Championship_Stadium_Name_group",
        desc="Championship stadium name verification",
        parent=parent,
        critical=False
    )
    stadium_sources = combine_sources(info.stadium_name_sources, info.game_city_sources, info.sources)
    provided = bool(info.stadium_name and info.stadium_name.strip()) and len(stadium_sources) > 0
    evaluator.add_custom_node(
        result=provided,
        id="Championship_Stadium_Name_provided",
        desc="Stadium name and at least one supporting source are provided",
        parent=group,
        critical=True
    )

    leaf = evaluator.add_leaf(
        id="Championship_Stadium_Name",
        desc="Provides the correct name of the stadium where the championship game was held",
        parent=group,
        critical=False
    )

    stadium = info.stadium_name or ""
    claim = f"The 2025 OHSAA Division I state championship game was held at {stadium}."
    await evaluator.verify(
        claim=claim,
        node=leaf,
        sources=stadium_sources,
        additional_instruction=(
            "Allow reasonable variants or sponsor names in the stadium title (e.g., short vs full official name). "
            "Pass if the evidence clearly shows the game took place at this stadium."
        )
    )


async def verify_stadium_capacity(evaluator: Evaluator, parent, info: ChampionshipInfo) -> None:
    group = evaluator.add_sequential(
        id="Stadium_Seating_Capacity_group",
        desc="Stadium seating capacity verification",
        parent=parent,
        critical=False
    )
    capacity_sources = combine_sources(info.stadium_capacity_sources, info.stadium_name_sources, info.sources)
    provided = bool(info.stadium_capacity and info.stadium_capacity.strip()) and len(capacity_sources) > 0
    evaluator.add_custom_node(
        result=provided,
        id="Stadium_Seating_Capacity_provided",
        desc="Stadium seating capacity and at least one supporting source are provided",
        parent=group,
        critical=True
    )

    leaf = evaluator.add_leaf(
        id="Stadium_Seating_Capacity",
        desc="States the seating capacity of the stadium where the championship game was played",
        parent=group,
        critical=False
    )

    stadium = info.stadium_name or "the stadium"
    capacity = info.stadium_capacity or ""
    claim = f"The seating capacity of {stadium} is {capacity}."
    await evaluator.verify(
        claim=claim,
        node=leaf,
        sources=capacity_sources,
        additional_instruction=(
            "Verify the stadium's seating capacity. "
            "If multiple numbers are shown (e.g., 'expandable' or ranges), use the standard football capacity. "
            "Allow minor rounding differences (approximately within ±5%)."
        )
    )


# --------------------------------------------------------------------------- #
# Main evaluation entry                                                       #
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
    Build the verification tree and evaluate the agent answer for the 2025 OHSAA Division I championship task.
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

    # Create a main container node mirroring the rubric root
    main_node = evaluator.add_parallel(
        id="2025_OHSAA_Division_I_Championship_Information",
        desc="Complete and accurate information about the 2025 OHSAA Division I state football championship, including the winning school from Ohio Capital Conference Central Division and game details",
        parent=root,
        critical=False
    )

    # Extract structured information from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_championship_info(),
        template_class=ChampionshipInfo,
        extraction_name="championship_info"
    )

    # Build verification subtrees
    await verify_school_identification(evaluator, main_node, extracted)
    await verify_final_record(evaluator, main_node, extracted)
    await verify_opponent(evaluator, main_node, extracted)
    await verify_final_score(evaluator, main_node, extracted)
    await verify_game_city(evaluator, main_node, extracted)
    await verify_stadium_name(evaluator, main_node, extracted)
    await verify_stadium_capacity(evaluator, main_node, extracted)

    return evaluator.get_summary()