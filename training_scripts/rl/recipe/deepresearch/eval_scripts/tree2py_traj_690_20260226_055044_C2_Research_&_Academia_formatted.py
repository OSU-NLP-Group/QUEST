import asyncio
import logging
from typing import Any, List, Optional, Dict

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "space_events_2026"
TASK_DESCRIPTION = (
    "Identify two significant space-related events scheduled for 2026: (1) a total lunar eclipse and (2) a crewed space mission. "
    "For each event, provide the requested details, and verify them against authoritative or official sources."
)


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class EclipseInfo(BaseModel):
    date: Optional[str] = None  # e.g., "March 3, 2026"
    visibility_regions: List[str] = Field(default_factory=list)  # continents/regions; at least one
    totality_duration_minutes: Optional[str] = None  # keep as string to allow formats like "65", "~65", "65-66"
    reference_urls: List[str] = Field(default_factory=list)  # authoritative astronomy/space agency sources


class CrewMember(BaseModel):
    name: Optional[str] = None
    agency: Optional[str] = None  # e.g., "NASA", "ESA", "JAXA", "CSA"


class MissionInfo(BaseModel):
    launch_timeframe: Optional[str] = None  # e.g., "NET October 2026", "Q3 2026", "November 2026"
    crew: List[CrewMember] = Field(default_factory=list)
    destination: Optional[str] = None  # e.g., "International Space Station", "Lunar orbit", "Moon surface"
    reference_urls: List[str] = Field(default_factory=list)  # official space agency site URLs


class SpaceEventsExtraction(BaseModel):
    eclipse: Optional[EclipseInfo] = None
    mission: Optional[MissionInfo] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_space_events() -> str:
    return """
    Extract exactly one 2026 total lunar eclipse and exactly one 2026 crewed space mission from the answer. 
    If multiple are mentioned, select the most clearly documented one for each category. 
    Return all fields even if some are missing (use nulls or empty arrays accordingly).

    For the total lunar eclipse (2026):
    - date: The exact date (month day, year) of the TOTAL lunar eclipse in 2026 (e.g., "March 3, 2026"). 
            If only year or month-year are given, still return the best available string.
    - visibility_regions: An array of at least one major geographic region or continent (e.g., "North America", "Asia", "Europe", "South America", "Pacific").
    - totality_duration_minutes: The duration of the total phase ("totality") in minutes as a string. 
                                 Do not parse into a number; keep the exact string from the answer (e.g., "65", "~65", "65-66").
    - reference_urls: An array of URLs from authoritative astronomy or space agency sources (e.g., NASA/GSFC, ESA, timeanddate, national observatories) 
                      that document the total lunar eclipse.

    For the crewed space mission (2026):
    - launch_timeframe: The planned launch timeframe for 2026 (month and year, or a "NET" style date).
    - crew: An array of objects with:
        - name: crew member's full name as written in the answer
        - agency: the crew member's space agency (e.g., "NASA", "ESA", "CSA", "JAXA", "Roscosmos", "CNSA", "ISRO")
    - destination: The primary destination or objective (e.g., "International Space Station", "lunar orbit", "Moon surface").
    - reference_urls: An array of official space agency website URLs documenting this mission (e.g., nasa.gov, esa.int, jaxa.jp, csa-asc.gc.ca).

    IMPORTANT:
    - Extract only what is explicitly stated in the provided answer.
    - Do not invent or infer missing information.
    - If a field is missing, use null (for single fields) or [] (for arrays).
    - Return a JSON object with keys: eclipse, mission.
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def format_crew_list(crew: List[CrewMember]) -> str:
    if not crew:
        return ""
    items = []
    for member in crew:
        n = member.name or ""
        a = member.agency or ""
        if a:
            items.append(f"{n} ({a})")
        else:
            items.append(f"{n}")
    return "; ".join([s for s in items if s.strip() != ""])


def safe_first_region(regions: List[str]) -> str:
    for r in regions:
        if r and r.strip():
            return r.strip()
    return ""


def join_urls(urls: List[str]) -> List[str]:
    """Normalize URL list: remove empties and duplicates, preserve order."""
    seen = set()
    out = []
    for u in urls:
        if not u or not isinstance(u, str):
            continue
        v = u.strip()
        if not v:
            continue
        if v not in seen:
            seen.add(v)
            out.append(v)
    return out


# --------------------------------------------------------------------------- #
# Verification builders                                                       #
# --------------------------------------------------------------------------- #
async def verify_eclipse(evaluator: Evaluator, parent_node, eclipse: Optional[EclipseInfo]) -> None:
    # Parent node for eclipse
    eclipse_node = evaluator.add_parallel(
        id="Total_Lunar_Eclipse_2026",
        desc="Identifies the total lunar eclipse occurring in 2026 with complete details including date, visibility region, duration of totality, and reference URL.",
        parent=parent_node,
        critical=False
    )

    # Normalize urls
    eclipse_urls = join_urls(eclipse.reference_urls if eclipse else [])

    # Critical: reference URL(s) provided (existence gate)
    evaluator.add_custom_node(
        result=len(eclipse_urls) > 0,
        id="Eclipse_Reference_URL_Provided",
        desc="Provides a valid reference URL from an authoritative astronomy or space agency source that documents the total lunar eclipse.",
        parent=eclipse_node,
        critical=True
    )

    # Critical: Date specified and supported
    date_leaf = evaluator.add_leaf(
        id="Eclipse_Date_Specified",
        desc="Provides the specific date (month and day, year) when the total lunar eclipse occurs in 2026.",
        parent=eclipse_node,
        critical=True
    )
    date_text = eclipse.date if eclipse and eclipse.date else ""
    eclipse_date_claim = f"There is a total lunar eclipse on {date_text}."
    await evaluator.verify(
        claim=eclipse_date_claim,
        node=date_leaf,
        sources=eclipse_urls,
        additional_instruction=(
            "Verify that the webpage explicitly mentions a total lunar eclipse on the exact stated date in 2026. "
            "Allow for minor timezone/local date presentation differences. Ensure it's a TOTAL lunar eclipse (not partial or penumbral)."
        ),
    )

    # Parallel subgroup: Visibility and Duration
    visdur_node = evaluator.add_parallel(
        id="Visibility_and_Duration_Details",
        desc="Provides both the visibility region and the duration of totality for the eclipse.",
        parent=eclipse_node,
        critical=False
    )

    # Critical: Visibility region identified and supported
    visibility_leaf = evaluator.add_leaf(
        id="Visibility_Region_Identified",
        desc="Identifies at least one major geographic region or continent from which the total lunar eclipse will be visible.",
        parent=visdur_node,
        critical=True
    )
    region_text = safe_first_region(eclipse.visibility_regions if eclipse else [])
    # If multiple regions available, include all to increase chance of match
    regions_for_claim = ", ".join([r for r in (eclipse.visibility_regions if eclipse else []) if r and r.strip()])
    if not regions_for_claim and region_text:
        regions_for_claim = region_text
    vis_claim = (
        f"The total lunar eclipse on {date_text or 'the stated 2026 date'} will be visible from at least one of the following major regions: "
        f"{regions_for_claim}."
    )
    await evaluator.verify(
        claim=vis_claim,
        node=visibility_leaf,
        sources=eclipse_urls,
        additional_instruction=(
            "Pass if the page shows the total lunar eclipse visibility includes at least one of the listed regions/continents "
            "(e.g., North America, South America, Europe, Africa, Asia, Australia, Pacific). Minor naming variations are acceptable."
        ),
    )

    # Critical: Duration of totality stated and supported
    duration_leaf = evaluator.add_leaf(
        id="Duration_of_Totality_Stated",
        desc="States the duration of totality (the total phase) of the eclipse in minutes.",
        parent=visdur_node,
        critical=True
    )
    duration_text = eclipse.totality_duration_minutes if eclipse and eclipse.totality_duration_minutes else ""
    dur_claim = (
        f"The duration of totality (total phase) for this total lunar eclipse is {duration_text}."
    )
    await evaluator.verify(
        claim=dur_claim,
        node=duration_leaf,
        sources=eclipse_urls,
        additional_instruction=(
            "Confirm that the page states the duration of the total phase (totality). "
            "Accept equivalent expressions (e.g., '~65 minutes', 'about 65 min', '1h05m'). Allow reasonable rounding."
        ),
    )


async def verify_mission(evaluator: Evaluator, parent_node, mission: Optional[MissionInfo]) -> None:
    # Parent node for mission
    mission_node = evaluator.add_parallel(
        id="Crewed_Space_Mission_2026",
        desc="Identifies a crewed space mission scheduled to launch in 2026, with complete details including launch timeframe, crew members, mission destination, and reference URL.",
        parent=parent_node,
        critical=False
    )

    # Normalize urls
    mission_urls = join_urls(mission.reference_urls if mission else [])

    # Critical: reference URL(s) provided (existence gate)
    evaluator.add_custom_node(
        result=len(mission_urls) > 0,
        id="Mission_Reference_URL_Provided",
        desc="Provides a valid reference URL from an official space agency website that documents the crewed space mission.",
        parent=mission_node,
        critical=True
    )

    # Critical: Launch timeframe specified and supported
    launch_leaf = evaluator.add_leaf(
        id="Launch_Timeframe_Specified",
        desc="Specifies the planned launch timeframe for the crewed mission, including at minimum the month and year or a 'no earlier than' date.",
        parent=mission_node,
        critical=True
    )
    timeframe_text = mission.launch_timeframe if mission and mission.launch_timeframe else ""
    launch_claim = f"The mission has a planned launch timeframe of {timeframe_text}, and this timeframe refers to 2026."
    await evaluator.verify(
        claim=launch_claim,
        node=launch_leaf,
        sources=mission_urls,
        additional_instruction=(
            "Confirm the page indicates a crewed mission with a launch timeframe in 2026. "
            "Accept formats like 'NET 2026', 'Q2 2026', 'November 2026', or similar. "
            "If the page does not indicate 2026, mark as not supported."
        ),
    )

    # Parallel subgroup: Crew and Destination information
    crewdest_node = evaluator.add_parallel(
        id="Crew_and_Destination_Information",
        desc="Provides both the crew member details and the mission destination.",
        parent=mission_node,
        critical=False
    )

    # Critical: Crew members named (with agencies) and supported
    crew_leaf = evaluator.add_leaf(
        id="Crew_Members_Named",
        desc="Lists all crew members by name and identifies their respective space agencies (e.g., NASA, CSA, ESA).",
        parent=crewdest_node,
        critical=True
    )
    crew_formatted = format_crew_list(mission.crew if mission else [])
    crew_claim = (
        f"The mission's crew (names with agencies) are: {crew_formatted}."
    )
    await evaluator.verify(
        claim=crew_claim,
        node=crew_leaf,
        sources=mission_urls,
        additional_instruction=(
            "Verify the listed crew members and their agencies match the official page. "
            "Allow minor variations in name formatting (e.g., middle initials). Fail if crew names are missing or do not match."
        ),
    )

    # Critical: Mission destination/objective described and supported
    dest_leaf = evaluator.add_leaf(
        id="Mission_Destination_Described",
        desc="Specifies the primary destination or objective of the crewed mission (e.g., lunar orbit, International Space Station, etc.).",
        parent=crewdest_node,
        critical=True
    )
    dest_text = mission.destination if mission and mission.destination else ""
    dest_claim = f"The mission's primary destination or objective is {dest_text}."
    await evaluator.verify(
        claim=dest_claim,
        node=dest_leaf,
        sources=mission_urls,
        additional_instruction=(
            "Confirm that the official page states this primary destination or objective. "
            "Accept equivalent phrases (e.g., 'ISS' vs 'International Space Station', 'lunar orbit' vs 'orbit around the Moon')."
        ),
    )


# --------------------------------------------------------------------------- #
# Main evaluation function                                                    #
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
    Evaluate an answer for the 2026 space events task (total lunar eclipse and crewed mission).
    """
    # Initialize evaluator with a parallel root to allow partial credit across the two events
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

    # Extract both events in one pass
    extracted = await evaluator.extract(
        prompt=prompt_extract_space_events(),
        template_class=SpaceEventsExtraction,
        extraction_name="space_events_extraction",
    )

    # Top-level organizer node (optional, aligns with rubric tree label)
    events_node = evaluator.add_parallel(
        id="2026_Space_Events_Research",
        desc="Identifies two specific space-related events occurring in 2026: a total lunar eclipse and a crewed space mission, with complete details for each.",
        parent=root,
        critical=False
    )

    # Build and verify eclipse subtree
    await verify_eclipse(evaluator, events_node, extracted.eclipse)

    # Build and verify mission subtree
    await verify_mission(evaluator, events_node, extracted.mission)

    # Return summary with verification tree
    return evaluator.get_summary()