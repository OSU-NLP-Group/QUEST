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
TASK_ID = "space_milestones_2025_2026"
TASK_DESCRIPTION = (
    "Between July 2025 and February 2026, several significant space exploration milestones occurred. "
    "Identify the following five events with their complete details:\n\n"
    "1. The interstellar object discovered in July 2025: Provide its name, discovery date, discovery location "
    "(including telescope and specific site), classification type, and perihelion date.\n\n"
    "2. The mission that achieved the first private spacewalk: Provide the mission name, the date when the spacewalk occurred, "
    "and the name of the mission commander who performed it.\n\n"
    "3. The Mars rover that completed the first AI-planned autonomous drives: Provide the rover name, the two specific dates "
    "when these drives occurred, and the corresponding mission sols.\n\n"
    "4. The first full moon of 2026: Provide its traditional name, the exact date and time of peak illumination (in EST), "
    "and explain why it is classified as a supermoon.\n\n"
    "5. The SpaceX crew mission that launched to the International Space Station in February 2026: Provide the mission designation, "
    "launch date, and the expedition number it joined aboard the ISS.\n\n"
    "For each event, include reference URLs that support your information."
)

# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class InterstellarDiscoveryInfo(BaseModel):
    name: Optional[str] = None
    discovery_date: Optional[str] = None
    telescope: Optional[str] = None
    site: Optional[str] = None
    classification: Optional[str] = None
    perihelion_date: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class PrivateSpacewalkInfo(BaseModel):
    mission_name: Optional[str] = None
    spacewalk_date: Optional[str] = None
    commander_name: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class AIRoverMilestoneInfo(BaseModel):
    rover_name: Optional[str] = None
    drive_date_1: Optional[str] = None
    drive_sol_1: Optional[str] = None
    drive_date_2: Optional[str] = None
    drive_sol_2: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class SupermoonEventInfo(BaseModel):
    traditional_name: Optional[str] = None
    peak_datetime_est: Optional[str] = None
    supermoon_explanation: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class ISSCrewMissionInfo(BaseModel):
    mission_designation: Optional[str] = None
    launch_date: Optional[str] = None
    expedition_number: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class SpaceMilestonesExtraction(BaseModel):
    interstellar_discovery: Optional[InterstellarDiscoveryInfo] = None
    private_spacewalk: Optional[PrivateSpacewalkInfo] = None
    ai_rover_milestone: Optional[AIRoverMilestoneInfo] = None
    supermoon_event: Optional[SupermoonEventInfo] = None
    iss_crew_mission: Optional[ISSCrewMissionInfo] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_space_milestones() -> str:
    return """
    Extract the requested five events exactly as presented in the answer. For each event, extract ONLY explicit information stated in the answer text and the URLs explicitly provided.

    1) interstellar_discovery:
       - name: Name of the interstellar object discovered in July 2025.
       - discovery_date: Exact discovery date as stated.
       - telescope: Name of the discovering telescope or instrument.
       - site: Specific site/location of the discovery (e.g., observatory/site name).
       - classification: Classification type (e.g., 'interstellar comet', 'interstellar object', etc.).
       - perihelion_date: Perihelion date as stated.
       - sources: Array of URLs provided in the answer that support these details.

    2) private_spacewalk:
       - mission_name: The mission that achieved the first private spacewalk.
       - spacewalk_date: The exact date of the spacewalk.
       - commander_name: Name of the mission commander who performed it.
       - sources: Array of URLs provided in the answer that support these details.

    3) ai_rover_milestone:
       - rover_name: The Mars rover that completed the first AI-planned autonomous drives.
       - drive_date_1: Date of the first AI-planned drive as stated.
       - drive_sol_1: Mission sol for that first drive as stated.
       - drive_date_2: Date of the second AI-planned drive as stated.
       - drive_sol_2: Mission sol for that second drive as stated.
       - sources: Array of URLs provided in the answer that support these details.

    4) supermoon_event:
       - traditional_name: The traditional name of the first full moon of 2026.
       - peak_datetime_est: The exact date and time of peak illumination in EST as stated in the answer (keep the formatting used in the answer).
       - supermoon_explanation: The explanation provided for why it is classified as a supermoon (e.g., proximity to perigee with a threshold).
       - sources: Array of URLs provided in the answer that support these details.

    5) iss_crew_mission:
       - mission_designation: The SpaceX crew mission that launched to the ISS in February 2026.
       - launch_date: Exact launch date as stated.
       - expedition_number: The ISS expedition number the crew joined upon arrival.
       - sources: Array of URLs provided in the answer that support these details.

    IMPORTANT:
    - Only extract information explicitly present in the answer.
    - For all 'sources' fields, extract only actual URLs present in the answer (plain URLs or markdown links).
    - If a field is missing, set it to null. If no sources are provided for an event, return an empty array for sources.
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _non_empty(s: Optional[str]) -> bool:
    return bool(s and str(s).strip() != "")


def _safe_sources(urls: Optional[List[str]]) -> List[str]:
    if not urls:
        return []
    # Deduplicate while preserving order
    seen = set()
    out = []
    for u in urls:
        if not isinstance(u, str):
            continue
        uu = u.strip()
        if uu and uu not in seen:
            seen.add(uu)
            out.append(uu)
    return out


# --------------------------------------------------------------------------- #
# Verification builders                                                       #
# --------------------------------------------------------------------------- #
async def verify_interstellar_discovery(evaluator: Evaluator, parent_node, data: Optional[InterstellarDiscoveryInfo]) -> None:
    node = evaluator.add_parallel(
        id="interstellar_discovery",
        desc="Correctly identify the interstellar object discovered in July 2025 with full details and sources",
        parent=parent_node,
        critical=False
    )
    urls = _safe_sources(data.sources if data else [])

    # Existence (critical gate)
    exists = (
        data is not None
        and _non_empty(data.name)
        and _non_empty(data.discovery_date)
        and _non_empty(data.telescope)
        and _non_empty(data.site)
        and _non_empty(data.classification)
        and _non_empty(data.perihelion_date)
        and len(urls) > 0
    )
    evaluator.add_custom_node(
        result=exists,
        id="interstellar_discovery_exists",
        desc="Interstellar discovery has all required fields and at least one source URL",
        parent=node,
        critical=True
    )

    # Individual factual checks (soft/partial)
    # Name
    leaf = evaluator.add_leaf(
        id="interstellar_name",
        desc="The name of the interstellar object is correctly stated",
        parent=node,
        critical=False
    )
    await evaluator.verify(
        claim=f"The interstellar object discovered in July 2025 is named '{data.name}'.",
        node=leaf,
        sources=urls,
        additional_instruction="Verify the object's official name as stated on the sources."
    )

    # Discovery date (and that it is in July 2025)
    leaf = evaluator.add_leaf(
        id="interstellar_discovery_date",
        desc="The discovery date is correct and falls in July 2025",
        parent=node,
        critical=False
    )
    await evaluator.verify(
        claim=f"The discovery date of {data.name} is {data.discovery_date}, which is in July 2025.",
        node=leaf,
        sources=urls,
        additional_instruction="Confirm the exact discovery date and that it is in July 2025."
    )

    # Telescope
    leaf = evaluator.add_leaf(
        id="interstellar_telescope",
        desc="The discovery telescope is correctly identified",
        parent=node,
        critical=False
    )
    await evaluator.verify(
        claim=f"{data.name} was discovered using the '{data.telescope}'.",
        node=leaf,
        sources=urls,
        additional_instruction="The sources should explicitly mention the discovering telescope/instrument."
    )

    # Site
    leaf = evaluator.add_leaf(
        id="interstellar_site",
        desc="The specific discovery site is correctly identified",
        parent=node,
        critical=False
    )
    await evaluator.verify(
        claim=f"The specific discovery site/location for {data.name} was '{data.site}'.",
        node=leaf,
        sources=urls,
        additional_instruction="The sources should explicitly mention the specific observing site or facility."
    )

    # Classification
    leaf = evaluator.add_leaf(
        id="interstellar_classification",
        desc="The object is classified as an interstellar object with the stated type",
        parent=node,
        critical=False
    )
    await evaluator.verify(
        claim=f"The object {data.name} is classified as an interstellar object of type '{data.classification}'.",
        node=leaf,
        sources=urls,
        additional_instruction="Confirm that the sources describe the object as interstellar (originating from outside the Solar System) and the specific type."
    )

    # Perihelion date
    leaf = evaluator.add_leaf(
        id="interstellar_perihelion_date",
        desc="The perihelion date is correctly stated",
        parent=node,
        critical=False
    )
    await evaluator.verify(
        claim=f"The perihelion date of {data.name} is {data.perihelion_date}.",
        node=leaf,
        sources=urls,
        additional_instruction="Verify the perihelion date as stated on the sources."
    )


async def verify_private_spacewalk(evaluator: Evaluator, parent_node, data: Optional[PrivateSpacewalkInfo]) -> None:
    node = evaluator.add_parallel(
        id="private_spacewalk",
        desc="Correctly identify the mission and details for the first private spacewalk with sources",
        parent=parent_node,
        critical=False
    )
    urls = _safe_sources(data.sources if data else [])

    # Existence (critical)
    exists = (
        data is not None
        and _non_empty(data.mission_name)
        and _non_empty(data.spacewalk_date)
        and _non_empty(data.commander_name)
        and len(urls) > 0
    )
    evaluator.add_custom_node(
        result=exists,
        id="private_spacewalk_exists",
        desc="Private spacewalk has all required fields and at least one source URL",
        parent=node,
        critical=True
    )

    # Mission name
    leaf = evaluator.add_leaf(
        id="spacewalk_mission",
        desc="The mission name is correctly stated",
        parent=node,
        critical=False
    )
    await evaluator.verify(
        claim=f"The mission that achieved the first private spacewalk was '{data.mission_name}'.",
        node=leaf,
        sources=urls,
        additional_instruction="Confirm the mission name associated with the first private/commercially funded spacewalk (EVA)."
    )

    # Spacewalk date
    leaf = evaluator.add_leaf(
        id="spacewalk_date",
        desc="The spacewalk date is correctly stated",
        parent=node,
        critical=False
    )
    await evaluator.verify(
        claim=f"The first private spacewalk occurred on {data.spacewalk_date}.",
        node=leaf,
        sources=urls,
        additional_instruction="Confirm the exact calendar date of the EVA."
    )

    # Commander name
    leaf = evaluator.add_leaf(
        id="spacewalk_commander",
        desc="The commander who performed the spacewalk is correctly identified",
        parent=node,
        critical=False
    )
    await evaluator.verify(
        claim=f"The spacewalk was performed by mission commander {data.commander_name}.",
        node=leaf,
        sources=urls,
        additional_instruction="Confirm that the identified commander actually performed the EVA."
    )

    # First private spacewalk claim
    leaf = evaluator.add_leaf(
        id="spacewalk_first_private",
        desc="The event is indeed the first private spacewalk",
        parent=node,
        critical=False
    )
    await evaluator.verify(
        claim=f"The spacewalk on {data.spacewalk_date} during {data.mission_name} is described as the first private spacewalk.",
        node=leaf,
        sources=urls,
        additional_instruction="Look for phrasing like 'first private spacewalk', 'first commercial EVA', or equivalent."
    )


async def verify_ai_rover_milestone(evaluator: Evaluator, parent_node, data: Optional[AIRoverMilestoneInfo]) -> None:
    node = evaluator.add_parallel(
        id="ai_rover_milestone",
        desc="Correctly identify the Mars rover and dates/sols for the first AI-planned autonomous drives with sources",
        parent=parent_node,
        critical=False
    )
    urls = _safe_sources(data.sources if data else [])

    # Existence (critical)
    exists = (
        data is not None
        and _non_empty(data.rover_name)
        and _non_empty(data.drive_date_1)
        and _non_empty(data.drive_sol_1)
        and _non_empty(data.drive_date_2)
        and _non_empty(data.drive_sol_2)
        and len(urls) > 0
    )
    evaluator.add_custom_node(
        result=exists,
        id="ai_rover_milestone_exists",
        desc="AI rover milestone has all required fields and at least one source URL",
        parent=node,
        critical=True
    )

    # Rover name
    leaf = evaluator.add_leaf(
        id="ai_rover_name",
        desc="The rover name is correctly stated",
        parent=node,
        critical=False
    )
    await evaluator.verify(
        claim=f"The Mars rover that completed the first AI-planned autonomous drives was {data.rover_name}.",
        node=leaf,
        sources=urls,
        additional_instruction="Confirm that the specified rover is credited with AI-planned autonomous drives."
    )

    # Drive 1
    leaf = evaluator.add_leaf(
        id="ai_drive1",
        desc="The first AI-planned drive date and sol are correct",
        parent=node,
        critical=False
    )
    await evaluator.verify(
        claim=f"An AI-planned autonomous drive occurred on {data.drive_date_1}, which corresponds to mission sol {data.drive_sol_1}.",
        node=leaf,
        sources=urls,
        additional_instruction="Verify both the calendar date and the mission sol."
    )

    # Drive 2
    leaf = evaluator.add_leaf(
        id="ai_drive2",
        desc="The second AI-planned drive date and sol are correct",
        parent=node,
        critical=False
    )
    await evaluator.verify(
        claim=f"Another AI-planned autonomous drive occurred on {data.drive_date_2}, which corresponds to mission sol {data.drive_sol_2}.",
        node=leaf,
        sources=urls,
        additional_instruction="Verify both the calendar date and the mission sol."
    )

    # First AI-planned claim
    leaf = evaluator.add_leaf(
        id="ai_first_ai_planned",
        desc="These are recognized as the first AI-planned autonomous drives",
        parent=node,
        critical=False
    )
    await evaluator.verify(
        claim="These events are identified as the first AI-planned autonomous drives on Mars.",
        node=leaf,
        sources=urls,
        additional_instruction="Look for explicit 'first AI-planned' phrasing or equivalent milestone description."
    )


async def verify_supermoon_event(evaluator: Evaluator, parent_node, data: Optional[SupermoonEventInfo]) -> None:
    node = evaluator.add_parallel(
        id="supermoon_event",
        desc="Correctly identify the first full moon of 2026 with name, EST peak time, and supermoon explanation with sources",
        parent=parent_node,
        critical=False
    )
    urls = _safe_sources(data.sources if data else [])

    # Existence (critical)
    exists = (
        data is not None
        and _non_empty(data.traditional_name)
        and _non_empty(data.peak_datetime_est)
        and _non_empty(data.supermoon_explanation)
        and len(urls) > 0
    )
    evaluator.add_custom_node(
        result=exists,
        id="supermoon_event_exists",
        desc="Supermoon event has all required fields and at least one source URL",
        parent=node,
        critical=True
    )

    # Traditional name
    leaf = evaluator.add_leaf(
        id="supermoon_name",
        desc="The traditional name of the first full moon of 2026 is correct",
        parent=node,
        critical=False
    )
    await evaluator.verify(
        claim=f"The first full moon of 2026 is traditionally called the '{data.traditional_name}'.",
        node=leaf,
        sources=urls,
        additional_instruction="Common traditional names include 'Wolf Moon', etc. Verify the exact designation for the first full moon of 2026."
    )

    # Peak illumination time in EST
    leaf = evaluator.add_leaf(
        id="supermoon_peak_est",
        desc="The EST date/time of peak illumination is correct",
        parent=node,
        critical=False
    )
    await evaluator.verify(
        claim=f"The peak illumination of this full moon occurs at {data.peak_datetime_est} EST.",
        node=leaf,
        sources=urls,
        additional_instruction="Accept 'ET' (Eastern Time) if the source uses ET/EST interchangeably and the time corresponds to EST as stated. "
                               "If a source lists UTC, it should also provide an explicit conversion or an EST/ET time."
    )

    # Classified as a supermoon
    leaf = evaluator.add_leaf(
        id="supermoon_is_supermoon",
        desc="The full moon is classified as a supermoon",
        parent=node,
        critical=False
    )
    await evaluator.verify(
        claim="This first full moon of 2026 is classified as a supermoon.",
        node=leaf,
        sources=urls,
        additional_instruction="Look for explicit classification as a supermoon (e.g., proximity to perigee or threshold definition)."
    )

    # Explanation why it is a supermoon
    leaf = evaluator.add_leaf(
        id="supermoon_why_supermoon",
        desc="The provided explanation for the supermoon classification is valid",
        parent=node,
        critical=False
    )
    await evaluator.verify(
        claim=f"It is classified as a supermoon because {data.supermoon_explanation}.",
        node=leaf,
        sources=urls,
        additional_instruction="The explanation should refer to the Moon being near perigee/closer than average or within a stated threshold leading to apparent size/brightness increase."
    )


async def verify_iss_crew_mission(evaluator: Evaluator, parent_node, data: Optional[ISSCrewMissionInfo]) -> None:
    node = evaluator.add_parallel(
        id="iss_crew_mission",
        desc="Correctly identify the SpaceX crew mission to ISS in Feb 2026 with designation, launch date, and expedition number with sources",
        parent=parent_node,
        critical=False
    )
    urls = _safe_sources(data.sources if data else [])

    # Existence (critical)
    exists = (
        data is not None
        and _non_empty(data.mission_designation)
        and _non_empty(data.launch_date)
        and _non_empty(data.expedition_number)
        and len(urls) > 0
    )
    evaluator.add_custom_node(
        result=exists,
        id="iss_crew_mission_exists",
        desc="ISS crew mission has all required fields and at least one source URL",
        parent=node,
        critical=True
    )

    # Mission designation
    leaf = evaluator.add_leaf(
        id="iss_mission_designation",
        desc="The mission designation is correctly stated",
        parent=node,
        critical=False
    )
    await evaluator.verify(
        claim=f"The SpaceX crew mission that launched to the ISS in February 2026 was {data.mission_designation}.",
        node=leaf,
        sources=urls,
        additional_instruction="Confirm that the mission designation matches the flight that launched in February 2026."
    )

    # Launch date (must be in February 2026)
    leaf = evaluator.add_leaf(
        id="iss_launch_date",
        desc="The mission launch date is correctly stated and falls in February 2026",
        parent=node,
        critical=False
    )
    await evaluator.verify(
        claim=f"The mission {data.mission_designation} launched on {data.launch_date}, which is in February 2026.",
        node=leaf,
        sources=urls,
        additional_instruction="Confirm the exact launch date and that it falls in February 2026."
    )

    # Mission to ISS
    leaf = evaluator.add_leaf(
        id="iss_to_iss",
        desc="The mission indeed launched to the International Space Station",
        parent=node,
        critical=False
    )
    await evaluator.verify(
        claim=f"The mission {data.mission_designation} launched to the International Space Station (ISS).",
        node=leaf,
        sources=urls,
        additional_instruction="Verify that the mission was a crewed flight to the ISS."
    )

    # Expedition number joined
    leaf = evaluator.add_leaf(
        id="iss_expedition_number",
        desc="The expedition number joined aboard the ISS is correctly stated",
        parent=node,
        critical=False
    )
    await evaluator.verify(
        claim=f"Upon arrival, the crew of {data.mission_designation} joined Expedition {data.expedition_number} aboard the ISS.",
        node=leaf,
        sources=urls,
        additional_instruction="Confirm the Expedition number they joined on station."
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
    model: str = "o4-mini"
) -> Dict:
    """
    Evaluate an answer for the 'space_milestones_2025_2026' task.
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

    # Extract structured information from the answer
    extraction = await evaluator.extract(
        prompt=prompt_extract_space_milestones(),
        template_class=SpaceMilestonesExtraction,
        extraction_name="space_milestones"
    )

    # Build verification subtrees for each event
    await verify_interstellar_discovery(evaluator, root, extraction.interstellar_discovery)
    await verify_private_spacewalk(evaluator, root, extraction.private_spacewalk)
    await verify_ai_rover_milestone(evaluator, root, extraction.ai_rover_milestone)
    await verify_supermoon_event(evaluator, root, extraction.supermoon_event)
    await verify_iss_crew_mission(evaluator, root, extraction.iss_crew_mission)

    # Return summary
    return evaluator.get_summary()