import asyncio
import logging
import datetime
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field

from mind2web2.utils.cache_filesys import CacheFileSys
from mind2web2.evaluator import (
    Evaluator,
    VerificationNode,
    AggregationStrategy,
)
from mind2web2.llm_client.base_client import LLMClient
from mind2web2.api_tools import tool_googlemap

# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "nyc_sport_event"
TASK_DESCRIPTION = """
Find two upcoming sports or music events in New York City that are scheduled within the next two months. For each event, include the following information:

- Location: The venue where the event will take place and its address.
- Date and Time: When the event is scheduled to occur.
- Ticketing Information: A link to where tickets can be purchased.
- Travel Time: An estimate of the travel time to the venue from 485 Marin Blvd, Jersey City, NJ, via public transportation.
"""

# Constants for evaluation
ORIGIN_ADDRESS = "485 Marin Blvd, Jersey City, NJ"
CURRENT_DATE = datetime.datetime.now()
TWO_MONTHS_LATER = CURRENT_DATE + datetime.timedelta(days=60)


# --------------------------------------------------------------------------- #
# Data Models for Extraction                                                  #
# --------------------------------------------------------------------------- #
class EventNames(BaseModel):
    """Structure to hold names of events mentioned in the answer."""
    names: List[str] = Field(default_factory=list)

class EventVenue(BaseModel):
    """Structure to hold venue information for an event."""
    name: Optional[str] = None
    address: Optional[str] = None

class EventGroup1(BaseModel):
    """Event name, type, venue, and source URLs"""
    name: Optional[str] = None
    type: Optional[str] = None  # "sports" or "music" or other
    venue: Optional[EventVenue] = None
    source_urls: List[str] = Field(default_factory=list)

class EventGroup2(BaseModel):
    """Event name, date, time, and source URLs"""
    name: Optional[str] = None
    date: Optional[str] = None
    time: Optional[str] = None
    source_urls: List[str] = Field(default_factory=list)

class EventGroup3(BaseModel):
    """Event name, ticket link, travel time, and source URLs"""
    name: Optional[str] = None
    ticket_link: Optional[str] = None
    travel_time: Optional[str] = None
    source_urls: List[str] = Field(default_factory=list)

class EventInfo(BaseModel):
    """Structure to represent a single event with all required details."""
    name: Optional[str] = None
    type: Optional[str] = None
    venue: Optional[EventVenue] = None
    date: Optional[str] = None
    time: Optional[str] = None
    ticket_link: Optional[str] = None
    travel_time: Optional[str] = None
    source_urls: List[str] = Field(default_factory=list)

class SourceURLs(BaseModel):
    """Structure to hold extracted URLs from the answer."""
    urls: List[str] = Field(default_factory=list)

# --------------------------------------------------------------------------- #
# Extraction Prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_event_names() -> str:
    return """
    Extract the names of all sports or music events mentioned in the answer that are scheduled to occur in New York City.
    
    Return a list of all event names found in the answer. If no event names are mentioned, return an empty list.
    """

def prompt_extract_event_group1(event_name: str) -> str:
    return f"""
    For the event named "{event_name}", extract the following information:
    
    1. name: Confirm the name of the event (should be "{event_name}")
    2. type: Whether it's a "sports" or "music" event (if specified)
    3. venue: A nested object with:
       - name: The name of the venue
       - address: The full address of the venue (if provided)
    4. source_urls: Any URLs explicitly cited as sources for this event's information
    
    If any information is missing, use null for that field.
    """

def prompt_extract_event_group2(event_name: str) -> str:
    return f"""
    For the event named "{event_name}", extract the following information:
    
    1. name: Confirm the name of the event (should be "{event_name}")
    2. date: The date when the event will occur (in any format mentioned)
    3. time: The time when the event will occur (in any format mentioned)
    4. source_urls: Any URLs explicitly cited as sources for this event's information
    
    If any information is missing, use null for that field.
    """

def prompt_extract_event_group3(event_name: str) -> str:
    return f"""
    For the event named "{event_name}", extract the following information:
    
    1. name: Confirm the name of the event (should be "{event_name}")
    2. ticket_link: The URL where tickets can be purchased
    3. travel_time: The estimated travel time from 485 Marin Blvd, Jersey City, NJ to the venue via public transportation
    4. source_urls: Any URLs explicitly cited as sources for this event's information
    
    If any information is missing, use null for that field.
    """
# --------------------------------------------------------------------------- #
# Helper functions for verification                                           #
# --------------------------------------------------------------------------- #
def merge_event_info(event_name: str, group1: EventGroup1, group2: EventGroup2, group3: EventGroup3) -> EventInfo:
    """Merge the three event information groups into a single EventInfo object."""
    # Combine source URLs from all groups, removing duplicates
    all_source_urls = list(set(
        group1.source_urls + 
        group2.source_urls + 
        group3.source_urls
    ))
    
    return EventInfo(
        name=event_name,
        type=group1.type,
        venue=group1.venue,
        date=group2.date,
        time=group2.time,
        ticket_link=group3.ticket_link,
        travel_time=group3.travel_time,
        source_urls=all_source_urls
    )

# --------------------------------------------------------------------------- #
# Verification functions for each aspect of events                            #
# --------------------------------------------------------------------------- #
async def verify_event(
    evaluator: Evaluator,
    parent_node: VerificationNode,
    event: EventInfo,
    index: int,
) -> None:
    """
    Verify all aspects of a single event using the new pattern.
    Each event_node is parallel with one critical node for essential checks 
    and four non-critical nodes for location, time, ticket, and travel info.
    """
    # Create a parallel parent for this event
    event_node = evaluator.add_sequential(
        id=f"event_{index}",
        desc=f"Event #{index+1}: {event.name or 'Unnamed event'} - All required information is correct and substantiated",
        parent=parent_node,
        critical=False,
    )
    
    # Step 1: find the eligible event - essential event requirements (all must pass)
    essential_node = evaluator.add_parallel(
        id=f"event_{index}_essential",
        desc=f"Event #{index+1} meets all essential requirements",
        parent=event_node,
        critical=False,
    )
    
    # Check 1: Event has necessary info (name and sources)
    evaluator.add_custom_node(
        result=bool(event.name and event.source_urls),
        id=f"event_{index}_has_info",
        desc="Event has necessary information (name and source URLs)",
        parent=essential_node,
        critical=True
    )
    
    # Check 2: Event is a sports or music event
    event_urls = event.source_urls if event.source_urls else []
    
    is_sports_music_node = evaluator.add_leaf(
        id=f"event_{index}_is_sports_music",
        desc=f"Event is a sports or music event",
        parent=essential_node,
        critical=True
    )
    
    await evaluator.verify(
        claim=f"The event '{event.name}' is specifically a sports or music event.",
        node=is_sports_music_node,
        sources=event_urls,
    )
    
    # Check 3: Event is within two months
    within_two_months_node = evaluator.add_leaf(
        id=f"event_{index}_within_two_months",
        desc=f"Event is within the next two months",
        parent=essential_node,
        critical=True
    )
    
    await evaluator.verify(
        claim=f"The event '{event.name}' is scheduled to occur between {CURRENT_DATE.strftime('%Y-%m-%d')} and {TWO_MONTHS_LATER.strftime('%Y-%m-%d')}.",
        node=within_two_months_node,
        sources=event_urls
    )
    
    # Check 4: Event is in NYC
    in_nyc_node = evaluator.add_leaf(
        id=f"event_{index}_in_nyc",
        desc=f"Event is in New York City",
        parent=essential_node,
        critical=True
    )
    
    await evaluator.verify(
        claim=f"The event '{event.name}' is taking place in New York City.",
        node=in_nyc_node,
        sources=event_urls,
    )

    # Step 2: provide additional required information
    additional_info_node = evaluator.add_parallel(
        id=f"event_{index}_additional",
        desc=f"Event #{index+1} is provided with required additional infomation",
        parent=event_node,
        critical=False,
    )
    
    # Non-critical node 1: Location information
    location_node = evaluator.add_parallel(
        id=f"event_{index}_location",
        desc=f"Location information is provided and accurate",
        parent=additional_info_node,
        critical=False,
    )
    
    evaluator.add_custom_node(
        result=bool(event.venue and event.venue.name and event.venue.address),
        id=f"event_{index}_location_exists",
        desc="Venue information is provided",
        parent=location_node,
        critical=True
    )
    
    location_accuracy_node = evaluator.add_leaf(
        id=f"event_{index}_location_accurate",
        desc=f"Venue name and address are accurate",
        parent=location_node,
        critical=True
    )
    
    venue_desc = event.venue.name if event.venue else ""
    if event.venue and event.venue.address:
        venue_desc += f" located at {event.venue.address}"
    
    await evaluator.verify(
        claim=f"The event '{event.name}' will take place at {venue_desc}.",
        node=location_accuracy_node,
        sources=event_urls,
    )
    
    # Non-critical node 2: Time information
    time_node = evaluator.add_parallel(
        id=f"event_{index}_time",
        desc=f"Time information is provided and accurate",
        parent=additional_info_node,
        critical=False,
    )
    
    evaluator.add_custom_node(
        result=bool(event.time and event.date),
        id=f"event_{index}_time_exists",
        desc="Time information is provided",
        parent=time_node,
        critical=True
    )
    
    time_accuracy_node = evaluator.add_leaf(
        id=f"event_{index}_time_accurate",
        desc=f"Time is accurate: {event.time or 'Not provided'}",
        parent=time_node,
        critical=True
    )
    
    await evaluator.verify(
        claim=f"The event '{event.name}' is scheduled for the time: {event.time} on {event.date}.",
        node=time_accuracy_node,
        sources=event_urls,
    )
    
    # Non-critical node 3: Ticket information
    ticket_node = evaluator.add_parallel(
        id=f"event_{index}_ticket",
        desc=f"Ticket information is provided and accurate",
        parent=additional_info_node,
        critical=False,
    )
    
    evaluator.add_custom_node(
        result=bool(event.ticket_link),
        id=f"event_{index}_ticket_exists",
        desc="Ticket link is provided",
        parent=ticket_node,
        critical=True
    )
    
    ticket_validity_node = evaluator.add_leaf(
        id=f"event_{index}_ticket_valid",
        desc=f"Ticket link is valid",
        parent=ticket_node,
        critical=True
    )
    
    await evaluator.verify(
        claim=f"The webpage is a valid link where tickets for the event '{event.name}' can be purchased.",
        node=ticket_validity_node,
        sources=[event.ticket_link] if event.ticket_link else [],
    )
    
    # Non-critical node 4: Travel time information
    travel_node = evaluator.add_parallel(
        id=f"event_{index}_travel",
        desc=f"Travel time information is provided and accurate",
        parent=additional_info_node,
        critical=False,
    )
    
    evaluator.add_custom_node(
        result=bool(event.travel_time),
        id=f"event_{index}_travel_exists",
        desc="Travel time estimate is provided",
        parent=travel_node,
        critical=True
    )
    
    # Use Google Maps API to verify travel time
    if event.venue and (event.venue.address or event.venue.name):
        # Initialize Google Maps tool
        gmaps = tool_googlemap.GoogleMapsTool()
        
        # Determine destination address
        destination = event.venue.address if event.venue.address else event.venue.name

        travel_accuracy_node = evaluator.add_leaf(
            id=f"event_{index}_travel_accurate",
            desc=f"Travel time is reasonable: {event.travel_time or 'Not provided'}",
            parent=travel_node,
            critical=True
        )
        
        try:
            # Calculate actual travel time using public transit
            actual_travel_seconds = await gmaps.calculate_travel_time(
                address1=ORIGIN_ADDRESS,
                address2=destination,
                mode="transit"
            )
            
            if isinstance(actual_travel_seconds, int):
                # Convert seconds to minutes for comparison
                actual_travel_minutes = actual_travel_seconds // 60
                
                # Create a claim that compares the stated time with the actual time
                claim = f"The stated travel time of '{event.travel_time}' from {ORIGIN_ADDRESS} to {destination} via public transportation is reasonably accurate compared to the actual travel time of approximately {actual_travel_minutes} minutes."
                additional_instruction = f"The Google Maps API shows the actual travel time is {actual_travel_minutes} minutes. Consider the stated time '{event.travel_time}' reasonable if it's within a 10% margin of the actual time, accounting for variations in transit schedules and routes."
                await evaluator.verify(
                    claim=claim,
                    node=travel_accuracy_node,
                    additional_instruction=additional_instruction
                )
            else:
                # If API call failed, fall back to general verification
                claim = f"The travel time from {ORIGIN_ADDRESS} to {destination} via public transportation is approximately '{event.travel_time}'."
                additional_instruction = f"Consider the stated time '{event.travel_time}' reasonable if it's within a 10% margin of the actual time, accounting for variations in transit schedules and routes."
                await evaluator.verify(
                    claim=claim,
                    node=travel_accuracy_node,
                    sources=event_urls if event_urls else None,
                    additional_instruction=additional_instruction
                )
        except Exception as e:
            # If API call fails, use fallback verification
            evaluator.verifier.logger.warning(f"Google Maps API call failed: {e}")
            claim = f"The travel time from {ORIGIN_ADDRESS} to {destination} via public transportation is approximately '{event.travel_time}'."
            additional_instruction = f"Consider the stated time '{event.travel_time}' reasonable if it's within a 10% margin of the actual time, accounting for variations in transit schedules and routes."
            await evaluator.verify(
                claim=claim,
                node=travel_accuracy_node,
                sources=event_urls if event_urls else None,
                additional_instruction=additional_instruction
            )
    else:
        # No venue information available, directly failed
        evaluator.add_custom_node(
            result=False,
            id=f"event_{index}_travel_accurate",
            desc=f"Travel time cannot be valided due the missing of address",
            parent=travel_node,
            critical=True
        )

# --------------------------------------------------------------------------- #
# Main evaluation entry point                                                 #
# --------------------------------------------------------------------------- #
async def evaluate_answer(
    client: LLMClient,
    answer: str,
    agent_name: str,
    answer_name: str,
    cache: CacheFileSys,
    semaphore: asyncio.Semaphore,
    logger: logging.Logger,
    model: str = "o4-mini"
) -> Dict:
    """
    Evaluate a single answer and return a structured result dictionary.
    """
    # -------- 1. Initialize evaluator ------------------------------------ #
    evaluator = Evaluator()
    evaluator.initialize(
        task_id=TASK_ID,
        task_description=TASK_DESCRIPTION,
        strategy=AggregationStrategy.PARALLEL,
        agent_name=agent_name,
        answer_name=answer_name,
        client=client,
        answer=answer,
        global_cache=cache,
        global_semaphore=semaphore,
        logger=logger,
        default_model=model
    )
    
    # -------- 2. Extract all event names first --------------------------- #
    event_names_result = await evaluator.extract(
        prompt=prompt_extract_event_names(),
        template_class=EventNames,
        extraction_name="event_names"
    )
    
    # Limit to the first 2 event names if more were extracted
    event_names = event_names_result.names[:2]
    
    # -------- 3. Extract details for each event in groups --------------- #
    events = []
    for i, event_name in enumerate(event_names):
        # Group 1: Name, type, venue, source_urls
        group1 = await evaluator.extract(
            prompt=prompt_extract_event_group1(event_name),
            template_class=EventGroup1,
            extraction_name=f"event_{i}_group1"
        )
        
        # Group 2: Name, date, time, source_urls
        group2 = await evaluator.extract(
            prompt=prompt_extract_event_group2(event_name),
            template_class=EventGroup2,
            extraction_name=f"event_{i}_group2"
        )
        
        # Group 3: Name, ticket_link, travel_time, source_urls
        group3 = await evaluator.extract(
            prompt=prompt_extract_event_group3(event_name),
            template_class=EventGroup3,
            extraction_name=f"event_{i}_group3"
        )
        
        # Merge all groups into a single EventInfo
        event_info = merge_event_info(event_name, group1, group2, group3)
        events.append(event_info)
    
    # Pad missing events with empty EventInfo objects
    while len(events) < 2:
        events.append(EventInfo())
    
    # -------- 4. Verify each event --------------------------------------- #
    for i, event in enumerate(events[:2]):  # Ensure we only process 2 events
        await verify_event(evaluator, evaluator.root, event, i)
    
    # -------- 5. Return structured result -------------------------------- #
    return evaluator.get_summary()