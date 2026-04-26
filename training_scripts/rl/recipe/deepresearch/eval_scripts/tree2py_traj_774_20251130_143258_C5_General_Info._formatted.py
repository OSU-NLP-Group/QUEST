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
TASK_ID = "events_nov2024_east_us"
TASK_DESCRIPTION = """
You are planning to attend two major events that took place in November 2024 in the eastern United States. For each event, provide the following information:

Event 1 - Raleigh Christmas Parade 2024:
1. The exact date (day of week and full date) the parade took place
2. The parade start time
3. The city and state where it took place
4. The name of the person who served as Grand Marshal
5. Confirmation that the Grand Marshal is a Grammy-nominated musician
6. The name of a country music artist who performed at the parade
7. The approximate length of the parade route in miles
8. The name of the street where the parade route started

Event 2 - National Dog Show 2024:
1. The dates (day of week and full dates) when the in-person show took place
2. The name of the venue where the show was held
3. The city and state where the venue is located
4. The complete street address of the venue
5. The name of the dog that won Best in Show in 2024
6. The breed of the winning dog
7. The date when the show was broadcast on television
8. The TV network that broadcast the show
9. The time the broadcast started (include time zone)

For each piece of information, provide a reference URL from a reliable source.
"""

# Expected canonical facts (used to frame claims)
EVENT1_LABEL = "Raleigh Christmas Parade 2024"
EVENT2_LABEL = "National Dog Show 2024"

EVENT1_EXPECTED = {
    "date": "Saturday, November 23, 2024",
    "start_time": "9:30 AM",
    "city_state": "Downtown Raleigh, North Carolina",
    "grand_marshal": "Marcus King",
    "grand_marshal_grammy": "Marcus King is a Grammy-nominated musician",
    "performer": "George Birge",
    "route_length": "approximately 1.5 miles",
    "route_start": "Hillsborough Street at the intersection with St. Mary's Street",
}

EVENT2_EXPECTED = {
    "in_person_dates": "Saturday and Sunday, November 16–17, 2024",
    "venue_name": "Greater Philadelphia Expo Center",
    "venue_city_state": "Oaks, Pennsylvania",
    "venue_address": "100 Station Avenue, Oaks, PA 19456",
    "best_in_show_name": "Vito",
    "best_in_show_breed": "Pug",
    "tv_broadcast_date": "Thursday, November 28, 2024",
    "tv_network": "NBC",
    "broadcast_start_time_tz": "12:00 PM ET",
}


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class FieldWithCitations(BaseModel):
    """A field value plus supporting citation URLs explicitly mentioned in the answer."""
    value: Optional[str] = None
    urls: List[str] = Field(default_factory=list)


class Event1Extraction(BaseModel):
    parade_date: Optional[FieldWithCitations] = None
    parade_start_time: Optional[FieldWithCitations] = None
    parade_city_state: Optional[FieldWithCitations] = None
    grand_marshal_name: Optional[FieldWithCitations] = None
    grand_marshal_grammy_confirmation: Optional[FieldWithCitations] = None
    country_music_artist_performed: Optional[FieldWithCitations] = None
    route_length_miles: Optional[FieldWithCitations] = None
    route_start_location: Optional[FieldWithCitations] = None


class Event2Extraction(BaseModel):
    in_person_show_dates: Optional[FieldWithCitations] = None
    venue_name: Optional[FieldWithCitations] = None
    venue_city_state: Optional[FieldWithCitations] = None
    venue_full_street_address: Optional[FieldWithCitations] = None
    best_in_show_winner_name: Optional[FieldWithCitations] = None
    best_in_show_winner_breed: Optional[FieldWithCitations] = None
    tv_broadcast_date: Optional[FieldWithCitations] = None
    tv_network: Optional[FieldWithCitations] = None
    broadcast_start_time_with_time_zone: Optional[FieldWithCitations] = None


class EventsExtraction(BaseModel):
    """Complete extraction for both events."""
    event1: Optional[Event1Extraction] = None
    event2: Optional[Event2Extraction] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_events() -> str:
    return """
    Extract the requested information for the two specified events from the provided answer text. For each requested field, return:
    - value: the exact value stated in the answer text (preserve formatting; include day-of-week for dates where requested; include time zone where requested).
    - urls: an array of http(s) URLs that the answer cites specifically for this field. Extract only URLs explicitly provided in the answer (plain URLs or within markdown links), not inferred.

    Structure the JSON with two top-level objects: 'event1' (Raleigh Christmas Parade 2024) and 'event2' (National Dog Show 2024).
    Under event1, extract:
      - parade_date {value, urls}
      - parade_start_time {value, urls}
      - parade_city_state {value, urls}
      - grand_marshal_name {value, urls}
      - grand_marshal_grammy_confirmation {value, urls}
      - country_music_artist_performed {value, urls}
      - route_length_miles {value, urls}
      - route_start_location {value, urls}

    Under event2, extract:
      - in_person_show_dates {value, urls}
      - venue_name {value, urls}
      - venue_city_state {value, urls}
      - venue_full_street_address {value, urls}
      - best_in_show_winner_name {value, urls}
      - best_in_show_winner_breed {value, urls}
      - tv_broadcast_date {value, urls}
      - tv_network {value, urls}
      - broadcast_start_time_with_time_zone {value, urls}

    Rules:
    - If the answer does not provide a value for a field, set 'value' to null.
    - If the answer does not provide any URLs for a field, return an empty 'urls' array.
    - Do not invent, infer, or normalize values — extract exactly as written in the answer.
    - For URLs, include only valid http(s) links explicitly present in the answer.
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _safe_urls(field: Optional[FieldWithCitations]) -> List[str]:
    return field.urls if (field and field.urls) else []


def _value_or_empty(field: Optional[FieldWithCitations]) -> str:
    return field.value or ""


# --------------------------------------------------------------------------- #
# Generic field verification builder                                          #
# --------------------------------------------------------------------------- #
async def add_field_checks(
    evaluator: Evaluator,
    parent_node,
    field_group_id: str,
    group_desc: str,
    expected_claim: str,
    urls: List[str],
    value_check_desc: str,
    citation_check_desc: str,
    value_additional_instruction: str,
    url_additional_instruction: str,
) -> None:
    """
    Create a parallel group with two critical leaf checks:
      - value_correct: Verify the answer states the expected value.
      - citation_url_provided: Verify the claim is supported by at least one cited URL.
    """
    group_node = evaluator.add_parallel(
        id=field_group_id,
        desc=group_desc,
        parent=parent_node,
        critical=True,
    )

    # Leaf 1: value_correct (simple verification against the answer content)
    value_node = evaluator.add_leaf(
        id=f"{field_group_id}_value_correct",
        desc=value_check_desc,
        parent=group_node,
        critical=True,
    )
    await evaluator.verify(
        claim=expected_claim,
        node=value_node,
        additional_instruction=value_additional_instruction,
    )

    # Leaf 2: citation_url_provided (multi-URL evidence verification)
    citation_node = evaluator.add_leaf(
        id=f"{field_group_id}_citation_url_provided",
        desc=citation_check_desc,
        parent=group_node,
        critical=True,
    )
    await evaluator.verify(
        claim=expected_claim,
        node=citation_node,
        sources=urls,
        additional_instruction=url_additional_instruction,
    )


# --------------------------------------------------------------------------- #
# Event-specific verification functions                                       #
# --------------------------------------------------------------------------- #
async def verify_event_1(
    evaluator: Evaluator,
    parent_node,
    ex: Optional[Event1Extraction],
) -> None:
    """Build verification sub-tree for Raleigh Christmas Parade 2024."""
    event_node = evaluator.add_parallel(
        id="event_1_raleigh_christmas_parade_2024",
        desc="Raleigh Christmas Parade 2024: provide each requested field and a supporting URL citation for each.",
        parent=parent_node,
        critical=True,
    )

    # 1) Parade date
    await add_field_checks(
        evaluator=evaluator,
        parent_node=event_node,
        field_group_id="parade_date_day_and_full_date",
        group_desc="Provide the parade date (day of week and full date) and a supporting URL citation.",
        expected_claim=f"The {EVENT1_LABEL} took place on {EVENT1_EXPECTED['date']}.",
        urls=_safe_urls(ex.parade_date if ex else None),
        value_check_desc="States the parade took place on Saturday, November 23, 2024.",
        citation_check_desc="Provides at least one http(s) URL citation that supports the parade date.",
        value_additional_instruction="Judge whether the answer explicitly states the parade date exactly as specified. Do not use external knowledge.",
        url_additional_instruction="Verify that at least one of the cited URLs explicitly states the parade date as Saturday, November 23, 2024 for the Raleigh Christmas Parade 2024.",
    )

    # 2) Parade start time
    await add_field_checks(
        evaluator=evaluator,
        parent_node=event_node,
        field_group_id="parade_start_time",
        group_desc="Provide the parade start time and a supporting URL citation.",
        expected_claim=f"The {EVENT1_LABEL} started at {EVENT1_EXPECTED['start_time']}.",
        urls=_safe_urls(ex.parade_start_time if ex else None),
        value_check_desc="States the parade started at 9:30 AM.",
        citation_check_desc="Provides at least one http(s) URL citation that supports the parade start time.",
        value_additional_instruction="Only pass if the answer clearly states 9:30 AM as the start time. Allow minor variations in punctuation (e.g., a.m.).",
        url_additional_instruction="Verify that at least one cited URL clearly states the parade start time as 9:30 AM for the 2024 Raleigh Christmas Parade.",
    )

    # 3) Parade city & state
    await add_field_checks(
        evaluator=evaluator,
        parent_node=event_node,
        field_group_id="parade_city_state",
        group_desc="Provide the city and state where the parade took place and a supporting URL citation.",
        expected_claim=f"The {EVENT1_LABEL} took place in {EVENT1_EXPECTED['city_state']}.",
        urls=_safe_urls(ex.parade_city_state if ex else None),
        value_check_desc="States the parade took place in Downtown Raleigh, North Carolina (Raleigh, NC).",
        citation_check_desc="Provides at least one http(s) URL citation that supports the parade location.",
        value_additional_instruction="Pass only if the answer explicitly indicates Raleigh, North Carolina (downtown wording is acceptable).",
        url_additional_instruction="Verify from the cited URLs that the parade location is Raleigh, NC (downtown references acceptable).",
    )

    # 4) Grand Marshal name
    await add_field_checks(
        evaluator=evaluator,
        parent_node=event_node,
        field_group_id="grand_marshal_name",
        group_desc="Provide the Grand Marshal's name and a supporting URL citation.",
        expected_claim=f"Marcus King served as Grand Marshal of the {EVENT1_LABEL}.",
        urls=_safe_urls(ex.grand_marshal_name if ex else None),
        value_check_desc="Identifies Marcus King as the Grand Marshal.",
        citation_check_desc="Provides at least one http(s) URL citation that supports the Grand Marshal identity.",
        value_additional_instruction="Only pass if the answer explicitly names Marcus King as Grand Marshal.",
        url_additional_instruction="Verify that at least one cited URL explicitly states Marcus King as the Grand Marshal for the 2024 Raleigh Christmas Parade.",
    )

    # 5) Grammy-nominated confirmation
    await add_field_checks(
        evaluator=evaluator,
        parent_node=event_node,
        field_group_id="grand_marshal_grammy_nominated_confirmation",
        group_desc="Confirm the Grand Marshal is a Grammy-nominated musician and provide a supporting URL citation.",
        expected_claim="Marcus King is a Grammy-nominated musician.",
        urls=_safe_urls(ex.grand_marshal_grammy_confirmation if ex else None),
        value_check_desc="Confirms Marcus King is a Grammy-nominated musician.",
        citation_check_desc="Provides at least one http(s) URL citation that supports the Grammy-nominated confirmation.",
        value_additional_instruction="Pass only if the answer explicitly claims Marcus King is Grammy-nominated.",
        url_additional_instruction="Verify from cited URLs that Marcus King has been nominated for a Grammy (any year is acceptable, but the nomination must be explicit).",
    )

    # 6) Country artist performed
    await add_field_checks(
        evaluator=evaluator,
        parent_node=event_node,
        field_group_id="country_music_artist_performed",
        group_desc="Provide the name of a country music artist who performed at the parade and a supporting URL citation.",
        expected_claim=f"George Birge performed at the {EVENT1_LABEL}.",
        urls=_safe_urls(ex.country_music_artist_performed if ex else None),
        value_check_desc="Identifies George Birge as a country music artist who performed at the parade.",
        citation_check_desc="Provides at least one http(s) URL citation that supports the performer claim.",
        value_additional_instruction="Pass only if the answer explicitly states George Birge performed at the parade.",
        url_additional_instruction="Verify that at least one cited URL explicitly states George Birge performed at the 2024 Raleigh Christmas Parade.",
    )

    # 7) Route length (miles)
    await add_field_checks(
        evaluator=evaluator,
        parent_node=event_node,
        field_group_id="route_length_miles",
        group_desc="Provide the approximate parade route length in miles and a supporting URL citation.",
        expected_claim=f"The {EVENT1_LABEL} route was approximately 1.5 miles long.",
        urls=_safe_urls(ex.route_length_miles if ex else None),
        value_check_desc="States the route was approximately 1.5 miles long.",
        citation_check_desc="Provides at least one http(s) URL citation that supports the route length.",
        value_additional_instruction="Pass if the answer states approximately 1.5 miles (minor phrasing variations allowed).",
        url_additional_instruction="Verify from cited URLs that the route length is approximately 1.5 miles for the 2024 parade.",
    )

    # 8) Route start location (street & intersection)
    await add_field_checks(
        evaluator=evaluator,
        parent_node=event_node,
        field_group_id="route_start_street_and_intersection",
        group_desc="Provide where the route started (street; intersection as specified) and a supporting URL citation.",
        expected_claim=f"The {EVENT1_LABEL} route started on Hillsborough Street at the intersection with St. Mary's Street.",
        urls=_safe_urls(ex.route_start_location if ex else None),
        value_check_desc="States the route started on Hillsborough Street at the intersection with St. Mary's Street.",
        citation_check_desc="Provides at least one http(s) URL citation that supports the route start location.",
        value_additional_instruction="Pass only if the answer clearly gives 'Hillsborough Street at St. Mary's Street' as the start location.",
        url_additional_instruction="Verify from cited URLs that the route start was Hillsborough Street at St. Mary's Street for the 2024 Raleigh Christmas Parade.",
    )


async def verify_event_2(
    evaluator: Evaluator,
    parent_node,
    ex: Optional[Event2Extraction],
) -> None:
    """Build verification sub-tree for National Dog Show 2024."""
    event_node = evaluator.add_parallel(
        id="event_2_national_dog_show_2024",
        desc="National Dog Show 2024: provide each requested field and a supporting URL citation for each.",
        parent=parent_node,
        critical=True,
    )

    # 1) In-person show dates
    await add_field_checks(
        evaluator=evaluator,
        parent_node=event_node,
        field_group_id="in_person_show_dates_day_and_full_dates",
        group_desc="Provide the in-person show dates (day(s) of week and full dates) and a supporting URL citation.",
        expected_claim=f"The in-person {EVENT2_LABEL} took place on {EVENT2_EXPECTED['in_person_dates']}.",
        urls=_safe_urls(ex.in_person_show_dates if ex else None),
        value_check_desc="States the in-person event took place on Saturday and Sunday, November 16–17, 2024.",
        citation_check_desc="Provides at least one http(s) URL citation that supports the in-person show dates.",
        value_additional_instruction="Pass only if the answer explicitly includes both the days of week and the dates (Nov 16–17, 2024).",
        url_additional_instruction="Verify from cited URLs that the in-person show occurred on Nov 16–17, 2024 (Saturday and Sunday).",
    )

    # 2) Venue name
    await add_field_checks(
        evaluator=evaluator,
        parent_node=event_node,
        field_group_id="venue_name",
        group_desc="Provide the venue name and a supporting URL citation.",
        expected_claim=f"The {EVENT2_LABEL} was held at the Greater Philadelphia Expo Center.",
        urls=_safe_urls(ex.venue_name if ex else None),
        value_check_desc="Identifies the venue as the Greater Philadelphia Expo Center.",
        citation_check_desc="Provides at least one http(s) URL citation that supports the venue name.",
        value_additional_instruction="Pass only if the answer explicitly states Greater Philadelphia Expo Center.",
        url_additional_instruction="Verify from cited URLs that the venue name is Greater Philadelphia Expo Center.",
    )

    # 3) Venue city & state
    await add_field_checks(
        evaluator=evaluator,
        parent_node=event_node,
        field_group_id="venue_city_state",
        group_desc="Provide the venue city and state and a supporting URL citation.",
        expected_claim=f"The {EVENT2_LABEL} venue is located in Oaks, Pennsylvania.",
        urls=_safe_urls(ex.venue_city_state if ex else None),
        value_check_desc="States the venue is located in Oaks, Pennsylvania.",
        citation_check_desc="Provides at least one http(s) URL citation that supports the venue city/state.",
        value_additional_instruction="Pass only if the answer explicitly gives Oaks, Pennsylvania.",
        url_additional_instruction="Verify from cited URLs that the venue is in Oaks, Pennsylvania.",
    )

    # 4) Venue full street address
    await add_field_checks(
        evaluator=evaluator,
        parent_node=event_node,
        field_group_id="venue_full_street_address",
        group_desc="Provide the complete street address of the venue and a supporting URL citation.",
        expected_claim="The Greater Philadelphia Expo Center address is 100 Station Avenue, Oaks, PA 19456.",
        urls=_safe_urls(ex.venue_full_street_address if ex else None),
        value_check_desc="Provides the complete address: 100 Station Avenue, Oaks, PA 19456.",
        citation_check_desc="Provides at least one http(s) URL citation that supports the venue address.",
        value_additional_instruction="Pass if the answer includes the full address exactly or equivalently formatted.",
        url_additional_instruction="Verify from cited URLs that the full address is 100 Station Avenue, Oaks, PA 19456.",
    )

    # 5) Best in Show winner name
    await add_field_checks(
        evaluator=evaluator,
        parent_node=event_node,
        field_group_id="best_in_show_winner_name",
        group_desc="Provide the Best in Show winning dog's name and a supporting URL citation.",
        expected_claim=f"The 2024 {EVENT2_LABEL} Best in Show winner was Vito.",
        urls=_safe_urls(ex.best_in_show_winner_name if ex else None),
        value_check_desc="Identifies Vito as the 2024 Best in Show winner name.",
        citation_check_desc="Provides at least one http(s) URL citation that supports the Best in Show winner name.",
        value_additional_instruction="Pass only if the answer clearly states the winner's name is Vito.",
        url_additional_instruction="Verify from cited URLs that Vito won Best in Show at the 2024 National Dog Show.",
    )

    # 6) Best in Show winner breed
    await add_field_checks(
        evaluator=evaluator,
        parent_node=event_node,
        field_group_id="best_in_show_winner_breed",
        group_desc="Provide the breed of the winning dog and a supporting URL citation.",
        expected_claim="The 2024 National Dog Show Best in Show winner (Vito) is a Pug.",
        urls=_safe_urls(ex.best_in_show_winner_breed if ex else None),
        value_check_desc="Identifies the winning dog (Vito) as a Pug.",
        citation_check_desc="Provides at least one http(s) URL citation that supports the winner breed.",
        value_additional_instruction="Pass only if the answer explicitly states Pug for the winner's breed.",
        url_additional_instruction="Verify from cited URLs that the 2024 Best in Show winner Vito is a Pug.",
    )

    # 7) TV broadcast date
    await add_field_checks(
        evaluator=evaluator,
        parent_node=event_node,
        field_group_id="tv_broadcast_date_day_and_full_date",
        group_desc="Provide the TV broadcast date (day of week and full date) and a supporting URL citation.",
        expected_claim=f"The {EVENT2_LABEL} broadcast aired on {EVENT2_EXPECTED['tv_broadcast_date']}.",
        urls=_safe_urls(ex.tv_broadcast_date if ex else None),
        value_check_desc="States the show was broadcast on Thursday, November 28, 2024.",
        citation_check_desc="Provides at least one http(s) URL citation that supports the broadcast date.",
        value_additional_instruction="Pass only if the answer includes the day of week and date for the broadcast (Thursday, Nov 28, 2024).",
        url_additional_instruction="Verify from cited URLs that the TV broadcast date was Thursday, Nov 28, 2024.",
    )

    # 8) TV network
    await add_field_checks(
        evaluator=evaluator,
        parent_node=event_node,
        field_group_id="tv_network",
        group_desc="Provide the TV network that broadcast the show and a supporting URL citation.",
        expected_claim=f"The {EVENT2_LABEL} was broadcast on {EVENT2_EXPECTED['tv_network']}.",
        urls=_safe_urls(ex.tv_network if ex else None),
        value_check_desc="Identifies NBC as the broadcast network.",
        citation_check_desc="Provides at least one http(s) URL citation that supports the broadcast network.",
        value_additional_instruction="Pass only if the answer clearly states NBC as the network.",
        url_additional_instruction="Verify from cited URLs that NBC broadcast the 2024 National Dog Show.",
    )

    # 9) Broadcast start time (with time zone)
    await add_field_checks(
        evaluator=evaluator,
        parent_node=event_node,
        field_group_id="broadcast_start_time_with_time_zone",
        group_desc="Provide the broadcast start time (including time zone) and a supporting URL citation.",
        expected_claim=f"The {EVENT2_LABEL} broadcast started at {EVENT2_EXPECTED['broadcast_start_time_tz']}.",
        urls=_safe_urls(ex.broadcast_start_time_with_time_zone if ex else None),
        value_check_desc="States the broadcast started at 12:00 PM ET.",
        citation_check_desc="Provides at least one http(s) URL citation that supports the broadcast start time and time zone.",
        value_additional_instruction="Pass only if the answer explicitly includes both the time and ET time zone.",
        url_additional_instruction="Verify from cited URLs that the broadcast start time was 12:00 PM ET.",
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
    Evaluate an answer for the November 2024 events task (Raleigh Christmas Parade & National Dog Show).
    """
    # Initialize evaluator
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

    # Create a top-level critical aggregator to enforce overall criticality
    main_node = evaluator.add_parallel(
        id="events_main_critical",
        desc="Provide complete and accurate information about the two specified November 2024 events, including at least one supporting URL citation for each requested field.",
        parent=root,
        critical=True,
    )

    # Extract structured information from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_events(),
        template_class=EventsExtraction,
        extraction_name="events_extraction",
    )

    # Add ground truth expectations to summary for transparency (not used for direct verification)
    evaluator.add_ground_truth({
        "event_1_expected": EVENT1_EXPECTED,
        "event_2_expected": EVENT2_EXPECTED,
        "note": "These expected facts are used to frame verification claims; the actual correctness is determined via cited sources."
    })

    # Build verification subtrees
    await verify_event_1(evaluator, main_node, extracted.event1 if extracted else None)
    await verify_event_2(evaluator, main_node, extracted.event2 if extracted else None)

    # Return evaluation summary
    return evaluator.get_summary()