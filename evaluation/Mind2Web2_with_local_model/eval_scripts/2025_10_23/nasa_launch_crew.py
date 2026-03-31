import asyncio
import logging
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from mind2web2 import CacheFileSys, Evaluator, VerificationNode, AggregationStrategy, LLMClient

TASK_ID = "nasa_launch_crew"
TASK_DESCRIPTION = """
Visit NASA's official website and locate the main page listing all upcoming launch events. For each scheduled event, clearly provide a direct link to its official NASA event page and explicitly indicate whether the mission is crewed.
For each crewed mission, if crew members have already been announced, clearly provide each crew member's full name, nationality, and a link to an informational page about that person, such as a biography, profile, or general introduction. If crew members have not yet been announced, explicitly state this.
"""

EVAL_NOTES = """
All launch schedule events can be found here: https://www.nasa.gov/event-type/launch-schedule/
Compare the list with all events provided in the answer.

The the official event page of each launch event clearly states if the mission is crewed, and the names of the crew members if they have been announced. Use the information on the official event page of each launch event to verify:
- whether the mission is crewed or not, and 
- whether the crew members have been announced
- the names of the crew members if they have been announced
"""

GROUND_TRUTH = {}


class CrewMember(BaseModel):
    """Information about a crew member"""
    name: Optional[str] = Field(default=None, description="Full name of the crew member")
    nationality: Optional[str] = Field(default=None, description="Nationality of the crew member")
    info_url: Optional[str] = Field(default=None, description="URL to crew member's informational page")


class LaunchEvent(BaseModel):
    """Information about a single launch event"""
    event_name: Optional[str] = Field(default=None, description="Name/title of the launch event")
    event_url: Optional[str] = Field(default=None, description="Direct link to official NASA event page")
    is_crewed: Optional[bool] = Field(default=None, description="Whether the mission is crewed")
    crew_announced: Optional[bool] = Field(default=None, description="Whether crew members have been announced")
    crew_members: List[CrewMember] = Field(default_factory=list, description="List of crew members if announced")
    crew_announcement_statement: Optional[str] = Field(default=None,
                                                       description="Statement about crew announcement status")


class NASALaunchInfo(BaseModel):
    """All NASA launch events extracted from the answer"""
    launch_events: List[LaunchEvent] = Field(default_factory=list, description="List of all launch events found")


class GroundTruthEvent(BaseModel):
    """Ground truth event from NASA website"""
    event_name: str = Field(description="Name/title of the launch event from NASA website")


class GroundTruthEvents(BaseModel):
    """All events from NASA launch schedule page"""
    events: List[GroundTruthEvent] = Field(default_factory=list, description="List of all upcoming launch events from NASA")


def prompt_extract_launch_info() -> str:
    """Extraction prompt for NASA launch information"""
    return """
    Extract information about NASA launch events from the answer.

    For each launch event mentioned, extract:
    - event_name: The name or title of the launch event
    - event_url: The direct link to the official NASA event page (must be a valid URL)
    - is_crewed: Whether the mission is crewed (true/false), or null if not mentioned
    - crew_announced: Whether crew members have been announced (true/false), or null if not mentioned
    - crew_announcement_statement: Any explicit statement about crew announcement status
    - crew_members: List of crew members with their details if announced

    For each crew member, extract:
    - name: Full name as provided
    - nationality: Nationality if mentioned
    - info_url: Link to their informational page (biography, profile, etc.)

    Extract ALL launch events mentioned in the answer, in the order they appear.
    Only extract information that is explicitly stated in the answer.
    """


def prompt_extract_ground_truth_events() -> str:
    """Extraction prompt for ground truth NASA events"""
    return """
    Extract all upcoming launch events from the NASA launch schedule page.
    
    For each launch event listed on the page, extract:
    - event_name: The name or title of the launch event as shown on the page
    
    Extract ALL events listed on the page as upcoming launches, in the order they appear.
    Include the full event name/title as displayed.
    """


async def find_matching_event(
    evaluator: Evaluator,
    gt_event: GroundTruthEvent, 
    answer_events: List[LaunchEvent] = None
) -> Optional[LaunchEvent]:
    """Find a matching event in the answer using LLM verification"""
    if not gt_event.event_name or not answer_events:
        return None
    
    # Try to find a match using LLM verification
    for answer_event in answer_events:
        if not answer_event.event_name:
            continue

        # Use LLM to verify if these are the same event
        is_match = await evaluator.verify(
            claim=f"The event '{gt_event.event_name}' from NASA's launch schedule refers to the same launch event as '{answer_event.event_name}' mentioned in the answer",
            node=None,
            sources=None,  # No URL needed for name matching
            additional_instruction="Consider these to be the same event if they refer to the same mission/launch, even if the names are worded slightly differently. For example, 'SpaceX Crew-8' and 'Crew-8 Mission' would be the same event."
        )
        
        if is_match:
            return answer_event
    
    return None


async def verify_launch_event(
        evaluator: Evaluator,
        parent_node: VerificationNode,
        event: LaunchEvent,
        gt_event: Optional[GroundTruthEvent],
        event_index: int,
) -> None:
    """Verify a single launch event"""

    # Create container node for this event
    event_name = event.event_name or f'Event {event_index + 1}'
    event_node = evaluator.add_parallel(
        id=f"event_{event_index}",
        desc=f"Launch event: {event_name}",
        parent=parent_node,
        critical=False,  # Non-critical to allow partial credit
    )

    # Check if event was found in answer (exists check)
    event_found = evaluator.add_custom_node(
        result=bool(event.event_name and event.event_url and event.is_crewed is not None),  # If event has a name, it was found in answer
        id=f"event_{event_index}_found",
        desc=f"Event '{event_name}' was found in answer with necessary information",
        parent=event_node,
        critical=True,  # Critical - if not found, nothing else to verify
    )

    # Verify the event URL is valid and from NASA
    url_valid_node = evaluator.add_leaf(
        id=f"event_{event_index}_url_valid",
        desc="Event URL is a valid NASA event page",
        parent=event_node,
        critical=True,
    )

    await evaluator.verify(
        claim=f"The URL {event.event_url} is a valid NASA event page for {event_name}",
        node=url_valid_node,
        sources=event.event_url if event.event_url else None,
        additional_instruction="Verify this is an official NASA page for a launch event. The URL should be from nasa.gov domain and contain information about a specific launch event."
    )

    # Verify crewed status against the event page
    crewed_status_node = evaluator.add_leaf(
        id=f"event_{event_index}_crewed_status_correct",
        desc=f"Crewed status ({event.is_crewed}) is correct",
        parent=event_node,
        critical=True,
    )

    await evaluator.verify(
        claim=f"The mission {event_name} is {'crewed' if event.is_crewed else 'not crewed'}",
        node=crewed_status_node,
        sources=event.event_url if event.event_url else None,
        additional_instruction="Check the event page to verify whether this mission is crewed or uncrewed. Look for explicit mentions of crew, astronauts, or statements about the mission being crewed/uncrewed."
    )

    # If mission is crewed, verify crew information
    if event.is_crewed:
        await verify_crew_information(evaluator, event_node, event, event_index)
    else:
        evaluator.add_custom_node(
            result=True,
            id=f"event_{event_index}_crew",
            desc="This event is not crewed",
            parent=event_node,
            critical=False
        )


async def verify_crew_information(
        evaluator: Evaluator,
        parent_node: VerificationNode,
        event: LaunchEvent,
        event_index: int,
) -> None:
    """Verify crew information for a crewed mission"""

    crew_node = evaluator.add_sequential(
        id=f"event_{event_index}_crew",
        desc="Crew information",
        parent=parent_node,
        critical=False,  # Non-critical to allow partial credit
    )

    # Check if crew announcement status was provided
    announcement_exists = evaluator.add_custom_node(
        result=(event.crew_announced is not None or bool(event.crew_announcement_statement)),
        id=f"event_{event_index}_crew_announcement_exists",
        desc="Crew announcement status is provided",
        parent=crew_node,
        critical=True,  # Critical - required for crewed missions
    )

    # Verify crew announcement status
    announcement_node = evaluator.add_leaf(
        id=f"event_{event_index}_crew_announcement_correct",
        desc="Crew announcement status is correct",
        parent=crew_node,
        critical=True,
    )

    announcement_claim = ""
    if event.crew_announced is True:
        announcement_claim = f"Crew members have been announced for {event.event_name or 'this mission'}"
    elif event.crew_announced is False:
        announcement_claim = f"Crew members have NOT been announced for {event.event_name or 'this mission'}"
    elif event.crew_announcement_statement:
        announcement_claim = f"The crew announcement status is: {event.crew_announcement_statement}"

    await evaluator.verify(
        claim=announcement_claim,
        node=announcement_node,
        sources=event.event_url if event.event_url else None,
        additional_instruction="Check the event page for information about whether crew members have been announced. Look for crew names or explicit statements about crew announcement status."
    )

    # If crew announced, verify crew member details
    if event.crew_announced is True and event.crew_members:
        await verify_crew_members(evaluator, crew_node, event, event_index)
    else:
        evaluator.add_custom_node(
            result=True,
            id=f"event_{event_index}_crew_members",
            desc="This event has not announced crew members",
            parent=crew_node,
            critical=False
        )


async def verify_crew_members(
        evaluator: Evaluator,
        parent_node: VerificationNode,
        event: LaunchEvent,
        event_index: int,
) -> None:
    """Verify individual crew members"""

    members_node = evaluator.add_parallel(
        id=f"event_{event_index}_crew_members",
        desc="Individual crew members",
        parent=parent_node,
        critical=False,  # Non-critical to allow partial credit
    )

    for member_index, member in enumerate(event.crew_members):
        if not member.name:
            continue

        member_node = evaluator.add_parallel(
            id=f"event_{event_index}_member_{member_index}",
            desc=f"Crew member: {member.name or f'Member {member_index + 1}'}",
            parent=members_node,
            critical=False,  # Non-critical for partial credit
        )

        # Check if name exists
        name_exists = evaluator.add_custom_node(
            result=bool(member.name and member.nationality and member.info_url),
            id=f"event_{event_index}_member_{member_index}_info_exists",
            desc="Necessary info is provided",
            parent=member_node,
            critical=True,
        )

        # Verify name against event page
        name_node = evaluator.add_leaf(
            id=f"event_{event_index}_member_{member_index}_name_correct",
            desc="Crew member name is correct",
            parent=member_node,
            critical=True,
        )

        await evaluator.verify(
            claim=f"There is a crew member named {member.name} for this mission",
            node=name_node,
            sources=event.event_url if event.event_url else None,
            additional_instruction="Verify that this person is listed as a crew member for this specific mission on the event page."
        )

        # Check nationality
        nationality_node = evaluator.add_leaf(
            id=f"event_{event_index}_member_{member_index}_nationality",
            desc=f"Nationality ({member.nationality}) is correct",
            parent=member_node,
            critical=True,
        )

        urls_for_nationality = []
        if event.event_url:
            urls_for_nationality.append(event.event_url)
        if member.info_url:
            urls_for_nationality.append(member.info_url)

        await evaluator.verify(
            claim=f"The crew member {member.name} has nationality: {member.nationality}",
            node=nationality_node,
            sources=urls_for_nationality if urls_for_nationality else None,
            additional_instruction="Check if the nationality information is mentioned on the event page or crew member's info page."
        )

        # Check info URL
        info_url_node = evaluator.add_leaf(
            id=f"event_{event_index}_member_{member_index}_info_url",
            desc="Info page URL is valid and about the crew member",
            parent=member_node,
            critical=True,
        )

        await evaluator.verify(
            claim=f"The URL {member.info_url} is an informational page about {member.name}",
            node=info_url_node,
            sources=member.info_url,
            additional_instruction=f"The page should contain information about {member.name}, such as a biography, profile, or general introduction."
        )


async def evaluate_answer(
        client: LLMClient,
        answer: str,
        agent_name: str,
        answer_name: str,
        cache: CacheFileSys,
        semaphore: asyncio.Semaphore,
        logger: logging.Logger,
        model: str = "o4-mini"
) -> Dict[str, Any]:
    """
    Main evaluation function for NASA launch crew task.

    Evaluates whether the answer correctly identifies all NASA launch events,
    their crewed status, and crew member information where applicable.
    """

    # Initialize evaluator
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,
        agent_name=agent_name,
        answer_name=answer_name,
        # Evaluator creation parameters
        client=client,
        task_description=TASK_DESCRIPTION,
        answer=answer,
        global_cache=cache,
        global_semaphore=semaphore,
        logger=logger,
        default_model=model,
    )

    # Extract ground truth events from NASA website
    nasa_schedule_url = "https://www.nasa.gov/event-type/launch-schedule/"
    gt_events = await evaluator.extract(
        prompt=prompt_extract_ground_truth_events(),
        template_class=GroundTruthEvents,
        extraction_name="ground_truth_events",
        source=nasa_schedule_url,
        additional_instruction="Extract ALL upcoming launch events listed on this NASA launch schedule page. Get the complete event names/titles as they appear on the page."
    )

    # Extract launch information from answer
    launch_info = await evaluator.extract(
        prompt=prompt_extract_launch_info(),
        template_class=NASALaunchInfo,
        extraction_name="nasa_launch_info",
    )

    # Add evaluation notes as custom info
    evaluator.add_custom_info(
        {"evaluation_notes": EVAL_NOTES},
        "evaluation_context"
    )

    # Add statistics
    evaluator.add_custom_info({
        "total_events_in_ground_truth": len(gt_events.events),
        "total_events_in_answer": len(launch_info.launch_events),
        "ground_truth_source": nasa_schedule_url
    }, "event_statistics")

    # Create container for all events
    all_events_node = evaluator.add_parallel(
        id="all_events",
        desc="All launch events",
        parent=root,
        critical=False,  # Non-critical to allow partial credit
    )

    # Create a unified list of events to verify
    # Match ground truth events with answer events
    event_index = 0
    
    # First, process all ground truth events
    for gt_event in gt_events.events:
        matching_answer_event = await find_matching_event(evaluator, gt_event, launch_info.launch_events)
        
        if matching_answer_event:
            await verify_launch_event(evaluator, all_events_node, matching_answer_event, gt_event, event_index)
        else:
            # Create empty event for missing ground truth event
            empty_event = LaunchEvent(
                event_name=None,  # Will trigger "not found" in verify_launch_event
                event_url=None,
                is_crewed=None,
                crew_announced=None,
                crew_members=[]
            )
            await verify_launch_event(evaluator, all_events_node, empty_event, gt_event, event_index)
        
        event_index += 1

    # Return evaluation results
    return evaluator.get_summary()