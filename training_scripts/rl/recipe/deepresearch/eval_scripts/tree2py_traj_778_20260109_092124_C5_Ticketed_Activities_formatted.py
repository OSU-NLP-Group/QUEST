import asyncio
import logging
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field
from obj_task_eval.llm_client.base_client import LLMClient

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys

# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "chicago_venues_task"
TASK_DESCRIPTION = """
You are planning a comprehensive entertainment guide for visitors to Chicago, Illinois. Identify three distinct ticketed activity venues that meet the following criteria:

1. An outdoor amphitheater that:
   - Is located in or within 50 miles of Chicago, Illinois
   - Has a seating capacity between 15,000 and 25,000 people
   - Is currently operational and actively hosting concerts

2. An indoor arena that:
   - Is located in Chicago, Illinois
   - Has a capacity of at least 19,000 people for concerts
   - Serves as the home arena for at least one professional sports team (NBA or NHL)

3. A family-friendly cultural attraction (zoo, aquarium, or museum) that:
   - Is located in Chicago, Illinois
   - Offers either completely free admission to all visitors OR free admission for children under age 3
   - Is open to the public year-round (365 days per year)

For each venue, provide its name, relevant capacity information (where applicable), admission policy (for the cultural attraction), and a reference URL that confirms the required information.
"""

# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class AmphitheaterInfo(BaseModel):
    """Outdoor amphitheater info extracted from the answer."""
    name: Optional[str] = None
    city: Optional[str] = None
    state: Optional[str] = None
    capacity_info: Optional[str] = None
    operational_status: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class ArenaInfo(BaseModel):
    """Indoor arena info extracted from the answer."""
    name: Optional[str] = None
    city: Optional[str] = None
    state: Optional[str] = None
    capacity_info: Optional[str] = None
    home_teams: List[str] = Field(default_factory=list)  # e.g., ["Chicago Bulls (NBA)", "Chicago Blackhawks (NHL)"]
    sources: List[str] = Field(default_factory=list)


class AttractionInfo(BaseModel):
    """Family-friendly cultural attraction info."""
    name: Optional[str] = None
    type: Optional[str] = None  # zoo, aquarium, or museum
    city: Optional[str] = None
    state: Optional[str] = None
    admission_policy: Optional[str] = None  # e.g., "Free for children under 3"
    year_round_statement: Optional[str] = None  # e.g., "Open 365 days"
    sources: List[str] = Field(default_factory=list)


class ChicagoVenuesExtraction(BaseModel):
    """Top-level extraction model for the three venues."""
    amphitheater: Optional[AmphitheaterInfo] = None
    arena: Optional[ArenaInfo] = None
    attraction: Optional[AttractionInfo] = None


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_chicago_venues() -> str:
    return """
    Your task is to extract structured information for three distinct venues described in the answer:
    1) An outdoor amphitheater (concert venue within ~50 miles of Chicago),
    2) An indoor arena in Chicago,
    3) A family-friendly cultural attraction (zoo, aquarium, or museum) in Chicago.

    For each venue, extract the following fields:

    Amphitheater (outdoor):
    - name: The amphitheater's name exactly as given.
    - city: City where the venue is located.
    - state: State where the venue is located (e.g., "Illinois").
    - capacity_info: Any stated capacity number(s) or description (string form; do not convert to numbers).
    - operational_status: Any statement indicating current operation and hosting concerts (e.g., "upcoming concerts", "2025 season").
    - sources: All explicit URLs provided that reference this amphitheater. Include only valid URLs. If none are provided, return an empty list.

    Arena (indoor):
    - name: The arena's name exactly as given.
    - city: City where the venue is located.
    - state: State where the venue is located (should be Illinois).
    - capacity_info: Any stated capacity number(s) or description (string form).
    - home_teams: The professional sports team(s) (NBA/NHL) listed as home teams (e.g., "Chicago Bulls", "Chicago Blackhawks"). If none are mentioned, return an empty list.
    - sources: All explicit URLs provided that reference this arena. Include only valid URLs. If none are provided, return an empty list.

    Attraction (zoo/aquarium/museum):
    - name: The attraction's name exactly as given.
    - type: One of: "zoo", "aquarium", or "museum". Use lowercase single word; if unclear, use what's most appropriate from the answer.
    - city: City where the venue is located.
    - state: State where the venue is located (should be Illinois).
    - admission_policy: Any text about admission policy, especially whether it's free for all OR free for children under age 3.
    - year_round_statement: Any text indicating year-round or 365-days-per-year operation (string form).
    - sources: All explicit URLs provided that reference this attraction. Include only valid URLs. If none are provided, return an empty list.

    General rules:
    - Extract ONLY what is explicitly present in the answer; do not invent.
    - For URLs, include full URLs (with protocol). Accept plain URLs or markdown links, but extract the actual URL.
    - If any field is missing in the answer, return null for strings or empty array for lists.
    - Maintain exact strings as they appear; do not normalize numbers or rewrite policies.
    """


# --------------------------------------------------------------------------- #
# Verification helper functions                                               #
# --------------------------------------------------------------------------- #
async def verify_amphitheater(
    evaluator: Evaluator,
    parent_node,
    info: Optional[AmphitheaterInfo],
) -> None:
    """
    Build verification tree and run checks for the outdoor amphitheater venue.
    """
    # Parent node (parallel, non-critical)
    amph_node = evaluator.add_parallel(
        id="Outdoor_Amphitheater_Venue",
        desc="Identify an outdoor concert venue/amphitheater in the Chicago area",
        parent=parent_node,
        critical=False,
    )

    # Basic existence checks (critical gates)
    name_provided = bool(info and info.name and info.name.strip())
    sources_list = info.sources if info and info.sources else []

    evaluator.add_custom_node(
        result=name_provided,
        id="Amphitheater_Name_Provided",
        desc="Amphitheater name is provided",
        parent=amph_node,
        critical=True,
    )

    evaluator.add_custom_node(
        result=len(sources_list) > 0,
        id="Amphitheater_Reference_URL_Provided",
        desc="At least one reference URL is provided for the amphitheater",
        parent=amph_node,
        critical=True,
    )

    # Location and Type (split into two critical leaves under a critical parallel node)
    loc_type_node = evaluator.add_parallel(
        id="Location_and_Type",
        desc="The venue must be an outdoor amphitheater located in or within 50 miles of Chicago, Illinois",
        parent=amph_node,
        critical=True,
    )

    # Type check
    type_leaf = evaluator.add_leaf(
        id="Amphitheater_Type_Check",
        desc=f"The venue '{info.name if info and info.name else ''}' is an outdoor amphitheater",
        parent=loc_type_node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"The venue named '{info.name if info else ''}' is an outdoor amphitheater.",
        node=type_leaf,
        sources=sources_list,
        additional_instruction="Confirm the venue is an outdoor amphitheater (open-air concert venue). Accept synonyms like 'outdoor pavilion', 'outdoor music theatre'.",
    )

    # Location within 50 miles
    loc_leaf = evaluator.add_leaf(
        id="Amphitheater_Location_Within_50_Miles",
        desc=f"The venue '{info.name if info and info.name else ''}' is located in or within 50 miles of Chicago, Illinois",
        parent=loc_type_node,
        critical=True,
    )
    city_state = ", ".join([s for s in [info.city if info else None, info.state if info else None] if s])
    await evaluator.verify(
        claim=f"The venue '{info.name if info else ''}' is located in {city_state} and is in or within 50 miles of Chicago, Illinois.",
        node=loc_leaf,
        sources=sources_list,
        additional_instruction="Use the page to confirm the city/state. If distance isn't explicitly stated, use common geographic knowledge to judge whether the city is within ~50 miles of Chicago. Treat 'Chicago area', 'Chicagoland', or 'suburb of Chicago' as within 50 miles.",
    )

    # Capacity requirement (critical)
    cap_leaf = evaluator.add_leaf(
        id="Capacity_Requirement",
        desc="The venue must have a seating capacity between 15,000 and 25,000 people",
        parent=amph_node,
        critical=True,
    )
    await evaluator.verify(
        claim="The venue has a seating capacity between 15,000 and 25,000 people.",
        node=cap_leaf,
        sources=sources_list,
        additional_instruction="Look for capacity numbers (e.g., 20,000). If multiple capacities (pavilion + lawn) are listed, consider total capacity. Accept approximate wording like 'about 22,000'.",
    )

    # Operational status (critical)
    op_leaf = evaluator.add_leaf(
        id="Operational_Status",
        desc="The venue must be currently operational and actively hosting concerts",
        parent=amph_node,
        critical=True,
    )
    await evaluator.verify(
        claim="The venue is currently operational and actively hosting concerts.",
        node=op_leaf,
        sources=sources_list,
        additional_instruction="Look for recent or upcoming events, current season schedules, or ticket on-sale notices indicating active operation.",
    )

    # Reference URL relevance (critical)
    ref_leaf = evaluator.add_leaf(
        id="Reference_URL",
        desc="Provide a valid reference URL confirming the venue's capacity and operational status",
        parent=amph_node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"The provided webpage(s) are about the amphitheater '{info.name if info else ''}' and confirm capacity and operational/concert activity.",
        node=ref_leaf,
        sources=sources_list,
        additional_instruction="Confirm that at least one provided URL is directly about the named amphitheater and includes capacity and/or current event information.",
    )


async def verify_arena(
    evaluator: Evaluator,
    parent_node,
    info: Optional[ArenaInfo],
) -> None:
    """
    Build verification tree and run checks for the indoor arena venue.
    """
    arena_node = evaluator.add_parallel(
        id="Indoor_Arena_Venue",
        desc="Identify an indoor arena in Chicago that hosts concerts and sports events",
        parent=parent_node,
        critical=False,
    )

    name_provided = bool(info and info.name and info.name.strip())
    sources_list = info.sources if info and info.sources else []

    evaluator.add_custom_node(
        result=name_provided,
        id="Arena_Name_Provided",
        desc="Arena name is provided",
        parent=arena_node,
        critical=True,
    )

    evaluator.add_custom_node(
        result=len(sources_list) > 0,
        id="Arena_Reference_URL_Provided",
        desc="At least one reference URL is provided for the arena",
        parent=arena_node,
        critical=True,
    )

    # Location and type (critical parallel node)
    loc_type_node = evaluator.add_parallel(
        id="Arena_Location_and_Type",
        desc="The venue must be an indoor arena located in Chicago, Illinois",
        parent=arena_node,
        critical=True,
    )

    # Type check
    type_leaf = evaluator.add_leaf(
        id="Arena_Type_Check",
        desc=f"The venue '{info.name if info and info.name else ''}' is an indoor arena",
        parent=loc_type_node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"The venue named '{info.name if info else ''}' is an indoor arena.",
        node=type_leaf,
        sources=sources_list,
        additional_instruction="Confirm the venue is an indoor arena (enclosed multipurpose venue).",
    )

    # Located in Chicago, IL
    loc_leaf = evaluator.add_leaf(
        id="Arena_Located_in_Chicago",
        desc=f"The venue '{info.name if info and info.name else ''}' is located in Chicago, Illinois",
        parent=loc_type_node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"The venue '{info.name if info else ''}' is located in Chicago, Illinois.",
        node=loc_leaf,
        sources=sources_list,
        additional_instruction="Confirm the page states the city is Chicago and state is Illinois.",
    )

    # Capacity requirement (critical)
    cap_leaf = evaluator.add_leaf(
        id="Arena_Capacity_Requirement",
        desc="The venue must have a capacity of at least 19,000 people for concerts",
        parent=arena_node,
        critical=True,
    )
    await evaluator.verify(
        claim="The arena has a capacity of at least 19,000 people for concerts (or events).",
        node=cap_leaf,
        sources=sources_list,
        additional_instruction="Check capacity figures on the page. If sports/event max capacity is ≥19,000, consider the requirement satisfied for concerts.",
    )

    # Professional sports (critical)
    pro_leaf = evaluator.add_leaf(
        id="Professional_Sports",
        desc="The venue must serve as the home arena for at least one professional sports team (NBA or NHL)",
        parent=arena_node,
        critical=True,
    )
    teams_text = ", ".join(info.home_teams) if info and info.home_teams else ""
    await evaluator.verify(
        claim=f"The arena serves as the home arena for at least one professional NBA or NHL team (e.g., {teams_text}).",
        node=pro_leaf,
        sources=sources_list,
        additional_instruction="Confirm that the page indicates home teams (NBA/NHL). Examples: Chicago Bulls (NBA), Chicago Blackhawks (NHL).",
    )

    # Reference URL relevance (critical)
    ref_leaf = evaluator.add_leaf(
        id="Arena_Reference_URL",
        desc="Provide a valid reference URL confirming the venue's capacity and home team(s)",
        parent=arena_node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"The provided webpage(s) are about the arena '{info.name if info else ''}' and confirm capacity and home team(s).",
        node=ref_leaf,
        sources=sources_list,
        additional_instruction="Confirm that at least one provided URL is directly about the named arena and includes capacity and team affiliation details.",
    )


async def verify_attraction(
    evaluator: Evaluator,
    parent_node,
    info: Optional[AttractionInfo],
) -> None:
    """
    Build verification tree and run checks for the family-friendly cultural attraction (zoo/aquarium/museum).
    """
    attr_node = evaluator.add_parallel(
        id="Family_Friendly_Attraction",
        desc="Identify a zoo, aquarium, or museum in Chicago with free or child-friendly admission",
        parent=parent_node,
        critical=False,
    )

    name_provided = bool(info and info.name and info.name.strip())
    sources_list = info.sources if info and info.sources else []

    evaluator.add_custom_node(
        result=name_provided,
        id="Attraction_Name_Provided",
        desc="Attraction name is provided",
        parent=attr_node,
        critical=True,
    )

    evaluator.add_custom_node(
        result=len(sources_list) > 0,
        id="Attraction_Reference_URL_Provided",
        desc="At least one reference URL is provided for the attraction",
        parent=attr_node,
        critical=True,
    )

    # Location and type (critical parallel node)
    loc_type_node = evaluator.add_parallel(
        id="Attraction_Location_and_Type",
        desc="The venue must be a zoo, aquarium, or museum located in Chicago, Illinois",
        parent=attr_node,
        critical=True,
    )

    # Type is one of zoo/aquarium/museum
    type_leaf = evaluator.add_leaf(
        id="Attraction_Type_Check",
        desc=f"The venue '{info.name if info and info.name else ''}' is a zoo, aquarium, or museum",
        parent=loc_type_node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"The venue '{info.name if info else ''}' is a {info.type if info and info.type else 'zoo/aquarium/museum'}.",
        node=type_leaf,
        sources=sources_list,
        additional_instruction="Confirm the page identifies the venue type as zoo, aquarium, or museum. Accept synonyms like 'science museum', 'children's museum', etc.",
    )

    # Located in Chicago, IL
    loc_leaf = evaluator.add_leaf(
        id="Attraction_Located_in_Chicago",
        desc=f"The venue '{info.name if info and info.name else ''}' is located in Chicago, Illinois",
        parent=loc_type_node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"The venue '{info.name if info else ''}' is located in Chicago, Illinois.",
        node=loc_leaf,
        sources=sources_list,
        additional_instruction="Confirm the page states the city is Chicago and state is Illinois.",
    )

    # Admission policy (critical)
    adm_leaf = evaluator.add_leaf(
        id="Admission_Policy",
        desc="The venue must offer either completely free admission to all visitors OR free admission for children under age 3",
        parent=attr_node,
        critical=True,
    )
    await evaluator.verify(
        claim="The venue offers either completely free admission for all visitors OR free admission for children under age 3.",
        node=adm_leaf,
        sources=sources_list,
        additional_instruction="Check the admission policy. Accept phrasing like 'children under 3 are free', 'ages 0–2 free', or 'free general admission'.",
    )

    # Year-round operation (critical)
    yr_leaf = evaluator.add_leaf(
        id="Year_Round_Operation",
        desc="The venue must be open to the public year-round (365 days per year)",
        parent=attr_node,
        critical=True,
    )
    await evaluator.verify(
        claim="The venue is open to the public year-round (365 days per year).",
        node=yr_leaf,
        sources=sources_list,
        additional_instruction="Confirm wording like 'open 365 days a year', 'open daily year-round'. If the venue states it's 'open year-round' (non-seasonal), consider the requirement satisfied.",
    )

    # Reference URL relevance (critical)
    ref_leaf = evaluator.add_leaf(
        id="Attraction_Reference_URL",
        desc="Provide a valid reference URL confirming the admission policy and operating schedule",
        parent=attr_node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"The provided webpage(s) are about the attraction '{info.name if info else ''}' and confirm admission policy and operating schedule.",
        node=ref_leaf,
        sources=sources_list,
        additional_instruction="Confirm that at least one provided URL is directly about the named attraction and includes admission policy and operation schedule information.",
    )


# --------------------------------------------------------------------------- #
# Main evaluation function                                                    #
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
    Entry point for evaluating the Chicago venues task.
    Builds a verification tree following the rubric and returns a structured summary.
    """
    # Initialize evaluator with a parallel root (as per rubric)
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

    # Extract all venue information from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_chicago_venues(),
        template_class=ChicagoVenuesExtraction,
        extraction_name="chicago_venues_extraction",
    )

    # Build and run verification checks for each venue
    await verify_amphitheater(evaluator, root, extracted.amphitheater)
    await verify_arena(evaluator, root, extracted.arena)
    await verify_attraction(evaluator, root, extracted.attraction)

    # Return standard summary
    return evaluator.get_summary()