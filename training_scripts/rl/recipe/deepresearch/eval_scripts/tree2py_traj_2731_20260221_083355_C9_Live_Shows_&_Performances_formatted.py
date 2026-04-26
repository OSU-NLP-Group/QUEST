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
TASK_ID = "us_concert_venues_4"
TASK_DESCRIPTION = (
    "Identify four major concert venues in the United States, each meeting all of the following specific criteria:\n\n"
    "Venue 1:\n"
    "- Located in Manhattan, New York City\n"
    "- Concert seating capacity between 19,000 and 21,000 seats\n"
    "- Current building opened between 1960 and 1970\n"
    "- Underwent at least one major renovation between 1990 and 2015\n"
    "- Recognized as one of the most famous arenas globally for hosting major concerts and sporting events\n\n"
    "Venue 2:\n"
    "- Located in Chicago, Illinois\n"
    "- Concert seating capacity between 22,000 and 24,500 seats\n"
    "- Among the largest arenas in the NBA by seating capacity\n"
    "- Hosts more than 200 events annually\n"
    "- Opened in the 1990s\n\n"
    "Venue 3:\n"
    "- Located in Atlanta, Georgia\n"
    "- Concert seating capacity between 15,000 and 17,500 seats\n"
    "- Ranked among the top 10 highest-grossing concert venues worldwide with 15,000+ capacity in 2024 by Pollstar or Billboard\n"
    "- Ranked among the top 7 venues in the United States for the 15,000+ capacity category in 2024\n"
    "- Serves as the home arena for an NBA team\n\n"
    "Venue 4:\n"
    "- Located in Manhattan, New York City\n"
    "- Seating capacity between 5,500 and 6,500 seats\n"
    "- Originally opened before 1950\n"
    "- Ranked #1 among venues in the 5,001-10,000 capacity category by Pollstar or Billboard in 2024\n"
    "- Recognized for its distinctive architectural style and historic significance\n\n"
    "For each venue, provide the venue name and reference URLs confirming each criterion."
)

RANKING_YEAR = 2024

# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class Venue1Info(BaseModel):
    name: Optional[str] = None
    capacity: Optional[str] = None
    capacity_sources: List[str] = Field(default_factory=list)
    location: Optional[str] = None
    location_sources: List[str] = Field(default_factory=list)
    opening_year: Optional[str] = None  # Current building opened year
    opening_sources: List[str] = Field(default_factory=list)
    renovation_years: List[str] = Field(default_factory=list)
    renovation_sources: List[str] = Field(default_factory=list)
    recognition: Optional[str] = None
    recognition_sources: List[str] = Field(default_factory=list)


class Venue2Info(BaseModel):
    name: Optional[str] = None
    capacity: Optional[str] = None
    capacity_sources: List[str] = Field(default_factory=list)
    location: Optional[str] = None
    location_sources: List[str] = Field(default_factory=list)
    size_ranking_claim: Optional[str] = None
    size_ranking_sources: List[str] = Field(default_factory=list)
    event_volume: Optional[str] = None
    event_volume_sources: List[str] = Field(default_factory=list)
    opening_year: Optional[str] = None
    opening_sources: List[str] = Field(default_factory=list)


class Venue3Info(BaseModel):
    name: Optional[str] = None
    capacity: Optional[str] = None
    capacity_sources: List[str] = Field(default_factory=list)
    location: Optional[str] = None
    location_sources: List[str] = Field(default_factory=list)
    ranking_global_2024_claim: Optional[str] = None
    ranking_global_2024_sources: List[str] = Field(default_factory=list)
    ranking_domestic_2024_claim: Optional[str] = None
    ranking_domestic_2024_sources: List[str] = Field(default_factory=list)
    home_team: Optional[str] = None
    home_team_sources: List[str] = Field(default_factory=list)


class Venue4Info(BaseModel):
    name: Optional[str] = None
    capacity: Optional[str] = None
    capacity_sources: List[str] = Field(default_factory=list)
    location: Optional[str] = None
    location_sources: List[str] = Field(default_factory=list)
    originally_opened_year: Optional[str] = None
    originally_opened_sources: List[str] = Field(default_factory=list)
    ranking_2024_category_claim: Optional[str] = None
    ranking_2024_category_sources: List[str] = Field(default_factory=list)
    architecture_recognition: Optional[str] = None
    architecture_sources: List[str] = Field(default_factory=list)


class VenuesExtraction(BaseModel):
    venue1: Optional[Venue1Info] = None
    venue2: Optional[Venue2Info] = None
    venue3: Optional[Venue3Info] = None
    venue4: Optional[Venue4Info] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_venues() -> str:
    return """
    Extract structured information for four venues described in the answer. For each numbered venue (Venue 1 to Venue 4), return the following fields exactly as they appear in the answer text. Also, extract the reference URLs that the answer provides to support each individual criterion. If a field or URL is missing for a venue, set it to null or an empty list as appropriate.

    Venue 1 fields:
    - name
    - capacity
    - capacity_sources (array of URLs)
    - location
    - location_sources (array of URLs)
    - opening_year (the year the current building opened)
    - opening_sources (array of URLs)
    - renovation_years (array of years or year ranges that indicate major renovations)
    - renovation_sources (array of URLs)
    - recognition (a sentence/phrase indicating recognition as a globally famous arena for concerts/sports)
    - recognition_sources (array of URLs)

    Venue 2 fields:
    - name
    - capacity
    - capacity_sources (array of URLs)
    - location
    - location_sources (array of URLs)
    - size_ranking_claim (sentence/phrase indicating it is among the largest NBA arenas)
    - size_ranking_sources (array of URLs)
    - event_volume (a phrase indicating it hosts more than 200 events annually)
    - event_volume_sources (array of URLs)
    - opening_year (the year it opened, should be in the 1990s)
    - opening_sources (array of URLs)

    Venue 3 fields:
    - name
    - capacity
    - capacity_sources (array of URLs)
    - location
    - location_sources (array of URLs)
    - ranking_global_2024_claim (phrase stating it was top 10 globally in 2024 in the 15,000+ capacity category by Pollstar or Billboard)
    - ranking_global_2024_sources (array of URLs)
    - ranking_domestic_2024_claim (phrase stating it was top 7 in the U.S. for 15,000+ capacity in 2024)
    - ranking_domestic_2024_sources (array of URLs)
    - home_team (NBA team name, if provided)
    - home_team_sources (array of URLs)

    Venue 4 fields:
    - name
    - capacity
    - capacity_sources (array of URLs)
    - location
    - location_sources (array of URLs)
    - originally_opened_year (year originally opened; should be before 1950)
    - originally_opened_sources (array of URLs)
    - ranking_2024_category_claim (phrase stating #1 ranking in 2024 for the 5,001–10,000 capacity category by Pollstar or Billboard)
    - ranking_2024_category_sources (array of URLs)
    - architecture_recognition (phrase noting distinctive architectural style and historic significance)
    - architecture_sources (array of URLs)

    IMPORTANT:
    - Only extract URLs explicitly present in the answer text. If the answer references a source without a URL, do not invent one—leave the corresponding sources list empty.
    - When extracting URLs, include the complete URL. Accept plain URLs or those embedded in markdown links; extract the actual URL.
    - Preserve the venue names and phrases exactly as written in the answer.

    Return the JSON object with the following top-level keys: venue1, venue2, venue3, venue4.
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _urls_non_empty(urls: Optional[List[str]]) -> bool:
    return bool(urls) and len([u for u in urls if isinstance(u, str) and u.strip()]) > 0


def _safe_name(name: Optional[str]) -> str:
    return name.strip() if isinstance(name, str) and name.strip() else "the venue"


def _add_sources_existence_node(
    evaluator: Evaluator,
    parent,
    node_id: str,
    desc: str,
    sources: Optional[List[str]],
) -> Any:
    """Add a non-critical existence check node for sources presence; used as a prerequisite for its paired leaf."""
    return evaluator.add_custom_node(
        result=_urls_non_empty(sources),
        id=node_id,
        desc=desc,
        parent=parent,
        critical=False  # keep non-critical to avoid global sibling gating; used as explicit prerequisite
    )


# --------------------------------------------------------------------------- #
# Verification functions per venue                                            #
# --------------------------------------------------------------------------- #
async def verify_venue_1(evaluator: Evaluator, parent_node: Any, v: Optional[Venue1Info]) -> None:
    venue_node = evaluator.add_parallel(
        id="venue_1",
        desc="Identify a major arena concert venue that meets all specified criteria for Venue 1",
        parent=parent_node,
        critical=False
    )
    name = _safe_name(v.name if v else None)

    # Capacity between 19,000 and 21,000
    cap_exist = _add_sources_existence_node(
        evaluator, venue_node, "venue_1_capacity_sources_exist",
        "Venue 1 capacity sources are provided", v.capacity_sources if v else []
    )
    cap_leaf = evaluator.add_leaf(
        id="venue_1_capacity",
        desc="The venue must have a concert seating capacity between 19,000 and 21,000 seats",
        parent=venue_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"{name} has a concert seating capacity between 19,000 and 21,000 seats.",
        node=cap_leaf,
        sources=(v.capacity_sources if v else []),
        additional_instruction="Confirm concert or arena seating capacity from the provided source(s). Accept reasonable variants; ensure the capacity falls within the stated range.",
        extra_prerequisites=[cap_exist]
    )

    # Location: Manhattan, NYC
    loc_exist = _add_sources_existence_node(
        evaluator, venue_node, "venue_1_location_sources_exist",
        "Venue 1 location sources are provided", v.location_sources if v else []
    )
    loc_leaf = evaluator.add_leaf(
        id="venue_1_location",
        desc="The venue must be located in New York City, specifically in Manhattan",
        parent=venue_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"{name} is located in Manhattan, New York City.",
        node=loc_leaf,
        sources=(v.location_sources if v else []),
        additional_instruction="Verify that the venue is in the Manhattan borough of NYC; accept minor address variations that clearly indicate Manhattan.",
        extra_prerequisites=[loc_exist]
    )

    # Current building opened between 1960 and 1970
    open_exist = _add_sources_existence_node(
        evaluator, venue_node, "venue_1_opening_sources_exist",
        "Venue 1 opening year sources are provided", v.opening_sources if v else []
    )
    open_leaf = evaluator.add_leaf(
        id="venue_1_historical",
        desc="The venue's current building must have opened between 1960 and 1970",
        parent=venue_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The current building of {name} opened between 1960 and 1970.",
        node=open_leaf,
        sources=(v.opening_sources if v else []),
        additional_instruction="Check the opening year of the current building iteration (not earlier versions) and confirm it lies between 1960 and 1970 inclusive.",
        extra_prerequisites=[open_exist]
    )

    # Renovation between 1990 and 2015
    ren_exist = _add_sources_existence_node(
        evaluator, venue_node, "venue_1_renovation_sources_exist",
        "Venue 1 renovation sources are provided", v.renovation_sources if v else []
    )
    ren_leaf = evaluator.add_leaf(
        id="venue_1_renovation",
        desc="The venue must have undergone at least one major renovation between 1990 and 2015",
        parent=venue_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"{name} underwent at least one major renovation between 1990 and 2015.",
        node=ren_leaf,
        sources=(v.renovation_sources if v else []),
        additional_instruction="Confirm a major renovation (e.g., significant modernization, reconfiguration) whose date falls in 1990–2015.",
        extra_prerequisites=[ren_exist]
    )

    # Recognized globally as a famous arena for concerts/sports
    rec_exist = _add_sources_existence_node(
        evaluator, venue_node, "venue_1_recognition_sources_exist",
        "Venue 1 recognition sources are provided", v.recognition_sources if v else []
    )
    rec_leaf = evaluator.add_leaf(
        id="venue_1_status",
        desc="The venue must be recognized as one of the most famous arenas globally for hosting major concerts and sporting events",
        parent=venue_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"{name} is recognized as one of the most famous arenas globally for major concerts and sporting events.",
        node=rec_leaf,
        sources=(v.recognition_sources if v else []),
        additional_instruction="Look for reputable sources (e.g., encyclopedic entries, major media) asserting global fame/prominence.",
        extra_prerequisites=[rec_exist]
    )


async def verify_venue_2(evaluator: Evaluator, parent_node: Any, v: Optional[Venue2Info]) -> None:
    venue_node = evaluator.add_parallel(
        id="venue_2",
        desc="Identify a major arena concert venue that meets all specified criteria for Venue 2",
        parent=parent_node,
        critical=False
    )
    name = _safe_name(v.name if v else None)

    # Capacity between 22,000 and 24,500 seats
    cap_exist = _add_sources_existence_node(
        evaluator, venue_node, "venue_2_capacity_sources_exist",
        "Venue 2 capacity sources are provided", v.capacity_sources if v else []
    )
    cap_leaf = evaluator.add_leaf(
        id="venue_2_capacity",
        desc="The venue must have a concert seating capacity between 22,000 and 24,500 seats",
        parent=venue_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"{name} has a concert seating capacity between 22,000 and 24,500 seats.",
        node=cap_leaf,
        sources=(v.capacity_sources if v else []),
        additional_instruction="Confirm concert or arena seating capacity; ensure it falls within the stated range.",
        extra_prerequisites=[cap_exist]
    )

    # Location: Chicago, Illinois
    loc_exist = _add_sources_existence_node(
        evaluator, venue_node, "venue_2_location_sources_exist",
        "Venue 2 location sources are provided", v.location_sources if v else []
    )
    loc_leaf = evaluator.add_leaf(
        id="venue_2_location",
        desc="The venue must be located in Chicago, Illinois",
        parent=venue_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"{name} is located in Chicago, Illinois.",
        node=loc_leaf,
        sources=(v.location_sources if v else []),
        additional_instruction="Verify city and state: Chicago, IL.",
        extra_prerequisites=[loc_exist]
    )

    # Among largest NBA arenas by seating capacity
    size_exist = _add_sources_existence_node(
        evaluator, venue_node, "venue_2_size_sources_exist",
        "Venue 2 NBA size ranking sources are provided", v.size_ranking_sources if v else []
    )
    size_leaf = evaluator.add_leaf(
        id="venue_2_size_ranking",
        desc="The venue must be among the largest arenas in its league (NBA) by seating capacity",
        parent=venue_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"{name} is among the largest NBA arenas by seating capacity.",
        node=size_leaf,
        sources=(v.size_ranking_sources if v else []),
        additional_instruction="Use authoritative lists (e.g., Wikipedia 'List of NBA arenas' or official sources) to confirm it's near the top by seating capacity.",
        extra_prerequisites=[size_exist]
    )

    # Hosts more than 200 events annually
    ev_exist = _add_sources_existence_node(
        evaluator, venue_node, "venue_2_event_sources_exist",
        "Venue 2 annual events sources are provided", v.event_volume_sources if v else []
    )
    ev_leaf = evaluator.add_leaf(
        id="venue_2_event_volume",
        desc="The venue must host more than 200 events annually",
        parent=venue_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"{name} hosts more than 200 events annually.",
        node=ev_leaf,
        sources=(v.event_volume_sources if v else []),
        additional_instruction="Look for official venue reports or credible media indicating >200 events per year.",
        extra_prerequisites=[ev_exist]
    )

    # Opened in the 1990s
    open_exist = _add_sources_existence_node(
        evaluator, venue_node, "venue_2_opening_sources_exist",
        "Venue 2 opening year sources are provided", v.opening_sources if v else []
    )
    open_leaf = evaluator.add_leaf(
        id="venue_2_opening",
        desc="The venue must have opened in the 1990s",
        parent=venue_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"{name} opened in the 1990s.",
        node=open_leaf,
        sources=(v.opening_sources if v else []),
        additional_instruction="Confirm the opening year is between 1990 and 1999 inclusive.",
        extra_prerequisites=[open_exist]
    )


async def verify_venue_3(evaluator: Evaluator, parent_node: Any, v: Optional[Venue3Info]) -> None:
    venue_node = evaluator.add_parallel(
        id="venue_3",
        desc="Identify a major arena concert venue that meets all specified criteria for Venue 3",
        parent=parent_node,
        critical=False
    )
    name = _safe_name(v.name if v else None)

    # Capacity between 15,000 and 17,500
    cap_exist = _add_sources_existence_node(
        evaluator, venue_node, "venue_3_capacity_sources_exist",
        "Venue 3 capacity sources are provided", v.capacity_sources if v else []
    )
    cap_leaf = evaluator.add_leaf(
        id="venue_3_capacity",
        desc="The venue must have a concert seating capacity between 15,000 and 17,500 seats",
        parent=venue_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"{name} has a concert seating capacity between 15,000 and 17,500 seats.",
        node=cap_leaf,
        sources=(v.capacity_sources if v else []),
        additional_instruction="Confirm concert/arena capacity falls within the specified range.",
        extra_prerequisites=[cap_exist]
    )

    # Location: Atlanta, Georgia
    loc_exist = _add_sources_existence_node(
        evaluator, venue_node, "venue_3_location_sources_exist",
        "Venue 3 location sources are provided", v.location_sources if v else []
    )
    loc_leaf = evaluator.add_leaf(
        id="venue_3_location",
        desc="The venue must be located in Atlanta, Georgia",
        parent=venue_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"{name} is located in Atlanta, Georgia.",
        node=loc_leaf,
        sources=(v.location_sources if v else []),
        additional_instruction="Verify city and state: Atlanta, GA.",
        extra_prerequisites=[loc_exist]
    )

    # 2024 global ranking top 10 in 15,000+ capacity category by Pollstar or Billboard
    rank_global_exist = _add_sources_existence_node(
        evaluator, venue_node, "venue_3_rank_global_sources_exist",
        "Venue 3 global 2024 ranking sources are provided", v.ranking_global_2024_sources if v else []
    )
    rank_global_leaf = evaluator.add_leaf(
        id="venue_3_2024_ranking",
        desc="The venue must have been ranked among the top 10 highest-grossing concert venues worldwide with 15,000+ capacity in 2024 by Pollstar or Billboard",
        parent=venue_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"In {RANKING_YEAR}, {name} was ranked among the top 10 highest-grossing concert venues worldwide in the 15,000+ capacity category by Pollstar or Billboard.",
        node=rank_global_leaf,
        sources=(v.ranking_global_2024_sources if v else []),
        additional_instruction=f"Use official Pollstar/Billboard 2024 lists for arenas (15,000+ capacity). Confirm top-10 worldwide placement. Accept small textual variants; ensure year {RANKING_YEAR} and capacity bracket are correct.",
        extra_prerequisites=[rank_global_exist]
    )

    # 2024 U.S. ranking top 7 for 15,000+ capacity
    rank_domestic_exist = _add_sources_existence_node(
        evaluator, venue_node, "venue_3_rank_domestic_sources_exist",
        "Venue 3 domestic 2024 ranking sources are provided", v.ranking_domestic_2024_sources if v else []
    )
    rank_domestic_leaf = evaluator.add_leaf(
        id="venue_3_domestic_ranking",
        desc="The venue must have been ranked among the top 7 venues in the United States for 15,000+ capacity category in 2024",
        parent=venue_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"In {RANKING_YEAR}, {name} was ranked among the top 7 venues in the United States for the 15,000+ capacity category.",
        node=rank_domestic_leaf,
        sources=(v.ranking_domestic_2024_sources if v else []),
        additional_instruction=f"Use Pollstar/Billboard 2024 U.S. venue rankings in the 15,000+ category to confirm top-7 status.",
        extra_prerequisites=[rank_domestic_exist]
    )

    # Home NBA team
    team_exist = _add_sources_existence_node(
        evaluator, venue_node, "venue_3_home_team_sources_exist",
        "Venue 3 home team sources are provided", v.home_team_sources if v else []
    )
    team_leaf = evaluator.add_leaf(
        id="venue_3_home_team",
        desc="The venue must serve as the home arena for an NBA team",
        parent=venue_node,
        critical=True
    )
    team_phrase = (
        f"{name} serves as the home arena of the {v.home_team} (NBA)." if (v and v.home_team and v.home_team.strip())
        else f"{name} serves as the home arena for an NBA team."
    )
    await evaluator.verify(
        claim=team_phrase,
        node=team_leaf,
        sources=(v.home_team_sources if v else []),
        additional_instruction="Confirm the venue is the home arena for an NBA franchise (e.g., team pages, NBA.com, official venue pages).",
        extra_prerequisites=[team_exist]
    )


async def verify_venue_4(evaluator: Evaluator, parent_node: Any, v: Optional[Venue4Info]) -> None:
    venue_node = evaluator.add_parallel(
        id="venue_4",
        desc="Identify a historic theater concert venue that meets all specified criteria for Venue 4",
        parent=parent_node,
        critical=False
    )
    name = _safe_name(v.name if v else None)

    # Capacity between 5,500 and 6,500
    cap_exist = _add_sources_existence_node(
        evaluator, venue_node, "venue_4_capacity_sources_exist",
        "Venue 4 capacity sources are provided", v.capacity_sources if v else []
    )
    cap_leaf = evaluator.add_leaf(
        id="venue_4_capacity",
        desc="The venue must have a seating capacity between 5,500 and 6,500 seats",
        parent=venue_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"{name} has a seating capacity between 5,500 and 6,500 seats.",
        node=cap_leaf,
        sources=(v.capacity_sources if v else []),
        additional_instruction="Confirm overall seating capacity; ensure it falls within the stated range.",
        extra_prerequisites=[cap_exist]
    )

    # Location: Manhattan, NYC
    loc_exist = _add_sources_existence_node(
        evaluator, venue_node, "venue_4_location_sources_exist",
        "Venue 4 location sources are provided", v.location_sources if v else []
    )
    loc_leaf = evaluator.add_leaf(
        id="venue_4_location",
        desc="The venue must be located in New York City, specifically in Manhattan",
        parent=venue_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"{name} is located in Manhattan, New York City.",
        node=loc_leaf,
        sources=(v.location_sources if v else []),
        additional_instruction="Verify location in the Manhattan borough of NYC.",
        extra_prerequisites=[loc_exist]
    )

    # Originally opened before 1950
    open_exist = _add_sources_existence_node(
        evaluator, venue_node, "venue_4_opening_sources_exist",
        "Venue 4 original opening sources are provided", v.originally_opened_sources if v else []
    )
    open_leaf = evaluator.add_leaf(
        id="venue_4_historical",
        desc="The venue must have originally opened before 1950",
        parent=venue_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"{name} originally opened before 1950.",
        node=open_leaf,
        sources=(v.originally_opened_sources if v else []),
        additional_instruction="Confirm the original opening year/date is earlier than 1950.",
        extra_prerequisites=[open_exist]
    )

    # 2024 ranking #1 in 5,001–10,000 category by Pollstar or Billboard
    rank_exist = _add_sources_existence_node(
        evaluator, venue_node, "venue_4_rank_sources_exist",
        "Venue 4 2024 category ranking sources are provided", v.ranking_2024_category_sources if v else []
    )
    rank_leaf = evaluator.add_leaf(
        id="venue_4_2024_ranking",
        desc="The venue must have been ranked #1 among venues in the 5,001-10,000 capacity category by Pollstar or Billboard in 2024",
        parent=venue_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"In {RANKING_YEAR}, {name} was ranked #1 among venues in the 5,001–10,000 capacity category by Pollstar or Billboard.",
        node=rank_leaf,
        sources=(v.ranking_2024_category_sources if v else []),
        additional_instruction=f"Use Pollstar/Billboard 2024 rankings by capacity category (5,001–10,000). Confirm #1 placement.",
        extra_prerequisites=[rank_exist]
    )

    # Recognized for distinctive architectural style and historic significance
    arch_exist = _add_sources_existence_node(
        evaluator, venue_node, "venue_4_arch_sources_exist",
        "Venue 4 architecture/historic recognition sources are provided", v.architecture_sources if v else []
    )
    arch_leaf = evaluator.add_leaf(
        id="venue_4_architecture",
        desc="The venue must be recognized for its distinctive architectural style and historic significance",
        parent=venue_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"{name} is recognized for its distinctive architectural style and historic significance.",
        node=arch_leaf,
        sources=(v.architecture_sources if v else []),
        additional_instruction="Look for credible coverage (e.g., official historic designation, architectural reviews) emphasizing distinctive style and historic importance.",
        extra_prerequisites=[arch_exist]
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
    Evaluate an answer for the four-venue concert task using the Mind2Web2 framework.
    """
    # Initialize evaluator
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # parallel aggregation for the overall task
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

    # Add an explicit task completion node to mirror rubric
    task_node = evaluator.add_parallel(
        id="task_completion",
        desc="Successfully identify all four major concert venues in the United States that meet the specified multi-dimensional criteria",
        parent=root,
        critical=False
    )

    # Extract venues info
    venues = await evaluator.extract(
        prompt=prompt_extract_venues(),
        template_class=VenuesExtraction,
        extraction_name="venues_extraction",
    )

    # Optionally log constraints as ground truth-style info for transparency
    evaluator.add_custom_info(
        info={
            "ranking_year": RANKING_YEAR,
            "capacity_requirements": {
                "venue_1": "19,000–21,000",
                "venue_2": "22,000–24,500",
                "venue_3": "15,000–17,500",
                "venue_4": "5,500–6,500"
            },
            "location_requirements": {
                "venue_1": "Manhattan, NYC",
                "venue_2": "Chicago, IL",
                "venue_3": "Atlanta, GA",
                "venue_4": "Manhattan, NYC"
            }
        },
        info_type="constraints_summary"
    )

    # Build verification subtrees
    await verify_venue_1(evaluator, task_node, venues.venue1)
    await verify_venue_2(evaluator, task_node, venues.venue2)
    await verify_venue_3(evaluator, task_node, venues.venue3)
    await verify_venue_4(evaluator, task_node, venues.venue4)

    # Return structured result
    return evaluator.get_summary()