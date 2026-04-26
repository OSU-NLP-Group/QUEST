import asyncio
import logging
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.llm_client.base_client import LLMClient
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "az_speaker_series_journalist_march_2026"
TASK_DESCRIPTION = (
    "Identify the ticketed speaking engagement in the Arizona Speaker Series featuring a broadcast journalist who "
    'hosted NBC\'s "Today" show from 1976 to 1989, later became the third anchor of CBS Sunday Morning in 2016, and '
    "received a Lifetime Achievement Emmy Award from the National Academy of Television Arts and Sciences (NATAS) "
    "in 2024. The event must take place in Phoenix, Arizona, in March 2026. Provide the following information: "
    "(1) The speaker's full name, (2) The exact date and start time of the event (format: Day of week, Month DD, YYYY, and time), "
    "(3) The venue name, (4) The venue's seating capacity, and (5) Reference URLs confirming these details."
)

EXPECTED_SPEAKER_NAME = "Jane Pauley"
EXPECTED_EVENT_DATE = "Wednesday, March 4, 2026"
EXPECTED_EVENT_TIME = "7:30 PM"
EXPECTED_VENUE_NAME = "Arizona Financial Theatre"
EXPECTED_VENUE_CAPACITY = "5,000"
EXPECTED_CITY = "Phoenix"
EXPECTED_STATE = "Arizona"
EXPECTED_SERIES_NAME = "Arizona Speaker Series"


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class SpeakerInfo(BaseModel):
    full_name: Optional[str] = None
    urls: List[str] = Field(default_factory=list)


class EventInfo(BaseModel):
    series_name: Optional[str] = None
    date: Optional[str] = None  # Expected format: "Day of week, Month DD, YYYY" e.g., "Wednesday, March 4, 2026"
    start_time: Optional[str] = None  # Expected format: "7:30 PM"
    city: Optional[str] = None
    state: Optional[str] = None
    urls: List[str] = Field(default_factory=list)  # Event listing URLs confirming date/time/venue/series/ticketing


class VenueInfo(BaseModel):
    name: Optional[str] = None
    capacity: Optional[str] = None  # Keep as string to allow "5,000" style formatting
    urls: List[str] = Field(default_factory=list)  # URLs confirming capacity (e.g., official venue or authoritative source)


class ExtractedEventPackage(BaseModel):
    speaker: Optional[SpeakerInfo] = None
    event: Optional[EventInfo] = None
    venue: Optional[VenueInfo] = None


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_event_package() -> str:
    return """
    Extract the requested event information as it appears in the provided answer text. Return a JSON object with the following structure:

    {
      "speaker": {
        "full_name": string | null,
        "urls": string[]  // URLs that specifically substantiate the speaker's identity or credentials.
      },
      "event": {
        "series_name": string | null,       // e.g., "Arizona Speaker Series"
        "date": string | null,              // e.g., "Wednesday, March 4, 2026" (include day of week)
        "start_time": string | null,        // e.g., "7:30 PM"
        "city": string | null,              // e.g., "Phoenix"
        "state": string | null,             // e.g., "Arizona"
        "urls": string[]                    // URLs that list or describe this event (date/time/venue/ticketing/series).
      },
      "venue": {
        "name": string | null,              // e.g., "Arizona Financial Theatre"
        "capacity": string | null,          // e.g., "5,000"
        "urls": string[]                    // URLs that substantiate the venue's capacity (authoritative sources).
      }
    }

    Rules:
    - Only extract values explicitly mentioned in the answer text.
    - If a required item is not present in the answer, return null for the field; if URLs are not provided, return an empty array.
    - For URLs, extract actual, valid URLs (plain or markdown links). Do not invent or infer missing URLs.
    - Preserve the exact formatting provided in the answer (e.g., keep commas in "5,000" and include day of week in dates when present).
    - Do not add information not found in the answer.
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
async def _verify_or_fail_due_to_missing_sources(
    evaluator: Evaluator,
    claim: str,
    node,
    sources: Optional[List[str] | str],
    additional_instruction: str
) -> bool:
    """
    Verify the claim against sources; if sources are missing or empty, mark node as failed immediately.
    """
    # Normalize to list length
    if sources is None or (isinstance(sources, list) and len(sources) == 0) or (isinstance(sources, str) and sources.strip() == ""):
        # Explicitly fail due to missing sources (since evidence-backed verification is required)
        node.score = 0.0
        node.status = "failed"
        try:
            evaluator.verifier.logger.info(
                f"Node {node.id} failed: No sources provided for evidence-backed verification.",
                extra={"id": node.id, "node_desc": node.desc}
            )
        except Exception:
            pass
        return False

    return await evaluator.verify(
        claim=claim,
        node=node,
        sources=sources,
        additional_instruction=additional_instruction
    )


# --------------------------------------------------------------------------- #
# Verification logic                                                          #
# --------------------------------------------------------------------------- #
async def build_verification_tree(evaluator: Evaluator, extracted: ExtractedEventPackage) -> None:
    """
    Build the verification tree and run checks according to the rubric.
    """
    # Create top-level critical node under evaluator's non-critical root
    event_ident_node = evaluator.add_parallel(
        id="Event_Identification",
        desc="Identify the qualifying Arizona Speaker Series ticketed speaking engagement and provide the required event details with supporting URLs.",
        parent=evaluator.root,
        critical=True
    )

    # --------------------- Speaker_Details_And_Eligibility ---------------------
    speaker_node = evaluator.add_parallel(
        id="Speaker_Details_And_Eligibility",
        desc="Speaker is identified and satisfies all specified career/award constraints.",
        parent=event_ident_node,
        critical=True
    )

    speaker_full_name = extracted.speaker.full_name if extracted.speaker else None
    speaker_urls = extracted.speaker.urls if extracted.speaker else []

    # Speaker full name provided (existence)
    evaluator.add_custom_node(
        result=bool(speaker_full_name and speaker_full_name.strip()),
        id="Speaker_Full_Name_Provided",
        desc="Provide the speaker's full name.",
        parent=speaker_node,
        critical=True
    )

    # Hosted NBC Today 1976–1989
    hosted_today_leaf = evaluator.add_leaf(
        id="Hosted_Today_1976_1989",
        desc="Speaker hosted NBC's 'Today' show from 1976 to 1989.",
        parent=speaker_node,
        critical=True
    )
    await _verify_or_fail_due_to_missing_sources(
        evaluator=evaluator,
        claim=f"{speaker_full_name or 'The speaker'} hosted NBC's 'Today' show from 1976 to 1989. "
              "Co-hosting counts as hosting.",
        node=hosted_today_leaf,
        sources=speaker_urls,
        additional_instruction="Use the provided webpage(s) to confirm the timeframe 1976–1989 for hosting/co-hosting Today."
    )

    # Became the third anchor of CBS Sunday Morning in 2016
    cbs_anchor_leaf = evaluator.add_leaf(
        id="CBS_Sunday_Morning_Third_Anchor_2016",
        desc="Speaker became the third anchor of CBS Sunday Morning in 2016.",
        parent=speaker_node,
        critical=True
    )
    await _verify_or_fail_due_to_missing_sources(
        evaluator=evaluator,
        claim=f"{speaker_full_name or 'The speaker'} became the third anchor (host) of CBS Sunday Morning in 2016.",
        node=cbs_anchor_leaf,
        sources=speaker_urls,
        additional_instruction="Allow synonyms like 'host'/'anchor' and verify the year 2016 for assuming the role."
    )

    # NATAS Lifetime Achievement Emmy Award in 2024
    natas_award_leaf = evaluator.add_leaf(
        id="NATAS_Lifetime_Achievement_Emmy_2024",
        desc="Speaker received a NATAS Lifetime Achievement Emmy Award in 2024.",
        parent=speaker_node,
        critical=True
    )
    await _verify_or_fail_due_to_missing_sources(
        evaluator=evaluator,
        claim=f"{speaker_full_name or 'The speaker'} received a Lifetime Achievement Emmy Award from NATAS in 2024.",
        node=natas_award_leaf,
        sources=speaker_urls,
        additional_instruction="Verify explicitly that the awarding body is NATAS and the year is 2024."
    )

    # --------------------- Event_Details_And_Constraints ----------------------
    event_node = evaluator.add_parallel(
        id="Event_Details_And_Constraints",
        desc="Event satisfies all specified series/type/location/time/venue/capacity constraints and those details are provided.",
        parent=event_ident_node,
        critical=True
    )

    event_urls = extracted.event.urls if extracted.event else []
    venue_urls = extracted.venue.urls if extracted.venue else []

    # Ticketed speaking engagement
    ticketed_leaf = evaluator.add_leaf(
        id="Ticketed_Speaking_Engagement",
        desc="Event is a ticketed speaking engagement.",
        parent=event_node,
        critical=True
    )
    await _verify_or_fail_due_to_missing_sources(
        evaluator=evaluator,
        claim="This event is a ticketed speaking engagement (tickets are required or offered for sale to attend).",
        node=ticketed_leaf,
        sources=event_urls,
        additional_instruction="Confirm the event page shows ticketing info such as 'Buy Tickets', pricing, or ticket policy."
    )

    # Part of Arizona Speaker Series
    series_leaf = evaluator.add_leaf(
        id="Part_Of_Arizona_Speaker_Series",
        desc="Event is part of the Arizona Speaker Series.",
        parent=event_node,
        critical=True
    )
    await _verify_or_fail_due_to_missing_sources(
        evaluator=evaluator,
        claim="This event is part of the Arizona Speaker Series.",
        node=series_leaf,
        sources=event_urls,
        additional_instruction="The page should explicitly mention 'Arizona Speaker Series' (or an official series page URL)."
    )

    # Location Phoenix, Arizona
    location_leaf = evaluator.add_leaf(
        id="Location_Phoenix_Arizona",
        desc="Event takes place in Phoenix, Arizona.",
        parent=event_node,
        critical=True
    )
    await _verify_or_fail_due_to_missing_sources(
        evaluator=evaluator,
        claim="The event takes place in Phoenix, Arizona.",
        node=location_leaf,
        sources=event_urls,
        additional_instruction="Explicitly confirm the city 'Phoenix' and state 'Arizona' on the event page."
    )

    # Event date: Wednesday, March 4, 2026
    date_leaf = evaluator.add_leaf(
        id="Event_Date_Wed_March_4_2026",
        desc="Event date is Wednesday, March 4, 2026.",
        parent=event_node,
        critical=True
    )
    await _verify_or_fail_due_to_missing_sources(
        evaluator=evaluator,
        claim=f"The event date is {EXPECTED_EVENT_DATE}.",
        node=date_leaf,
        sources=event_urls,
        additional_instruction="Confirm exact date including day of the week, month, day, and year."
    )

    # Event start time: 7:30 PM
    time_leaf = evaluator.add_leaf(
        id="Event_Start_Time_730PM",
        desc="Event start time is 7:30 PM.",
        parent=event_node,
        critical=True
    )
    await _verify_or_fail_due_to_missing_sources(
        evaluator=evaluator,
        claim=f"The event start time is {EXPECTED_EVENT_TIME}.",
        node=time_leaf,
        sources=event_urls,
        additional_instruction="Allow minor formatting variants (e.g., '7:30 p.m.'); verify equivalence."
    )

    # Venue name: Arizona Financial Theatre
    venue_name_leaf = evaluator.add_leaf(
        id="Venue_Name_Arizona_Financial_Theatre",
        desc="Venue name is Arizona Financial Theatre.",
        parent=event_node,
        critical=True
    )
    await _verify_or_fail_due_to_missing_sources(
        evaluator=evaluator,
        claim=f"The venue name is {EXPECTED_VENUE_NAME}.",
        node=venue_name_leaf,
        sources=event_urls,
        additional_instruction="The event page should list the venue as 'Arizona Financial Theatre' (allow capitalization variants)."
    )

    # Venue capacity: 5,000
    capacity_leaf = evaluator.add_leaf(
        id="Venue_Capacity_5000",
        desc="Venue seating capacity is 5,000.",
        parent=event_node,
        critical=True
    )
    await _verify_or_fail_due_to_missing_sources(
        evaluator=evaluator,
        claim=f"The venue seating capacity is {EXPECTED_VENUE_CAPACITY}.",
        node=capacity_leaf,
        sources=venue_urls,
        additional_instruction="Confirm the seating capacity on an authoritative page (official venue site or credible sources)."
    )

    # --------------------- Reference_URLs -------------------------------------
    refs_node = evaluator.add_parallel(
        id="Reference_URLs",
        desc="Provide reference URLs that substantiate the speaker and event details.",
        parent=event_ident_node,
        critical=True
    )

    # Include at least one reference URL for speaker credentials (existence check)
    evaluator.add_custom_node(
        result=bool(speaker_urls and len(speaker_urls) > 0),
        id="URLs_Confirm_Speaker_Credentials",
        desc="Include at least one reference URL that supports the speaker identity and stated career/award credentials.",
        parent=refs_node,
        critical=True
    )

    # Include at least one reference URL confirming event listing details (verify combined claim against event URLs)
    urls_event_details_leaf = evaluator.add_leaf(
        id="URLs_Confirm_Event_Details",
        desc="Include at least one reference URL that supports the event listing details (date, start time, and venue).",
        parent=refs_node,
        critical=True
    )
    await _verify_or_fail_due_to_missing_sources(
        evaluator=evaluator,
        claim=f"The referenced page confirms the event date '{EXPECTED_EVENT_DATE}', start time '{EXPECTED_EVENT_TIME}', "
              f"and venue '{EXPECTED_VENUE_NAME}'.",
        node=urls_event_details_leaf,
        sources=event_urls,
        additional_instruction="Verify that a single event page (or any one provided event URL) explicitly lists date, start time, and venue."
    )

    # Include at least one reference URL confirming venue capacity (verify against venue URLs)
    urls_venue_capacity_leaf = evaluator.add_leaf(
        id="URLs_Confirm_Venue_Capacity",
        desc="Include at least one reference URL that supports the venue seating capacity.",
        parent=refs_node,
        critical=True
    )
    await _verify_or_fail_due_to_missing_sources(
        evaluator=evaluator,
        claim=f"The referenced page confirms that the venue seating capacity is {EXPECTED_VENUE_CAPACITY}.",
        node=urls_venue_capacity_leaf,
        sources=venue_urls,
        additional_instruction="Use a credible venue capacity source; match numeric value allowing commas or formatting variants."
    )

    # Optional: record ground truth for clarity
    evaluator.add_ground_truth({
        "expected_speaker": EXPECTED_SPEAKER_NAME,
        "expected_series": EXPECTED_SERIES_NAME,
        "expected_city": EXPECTED_CITY,
        "expected_state": EXPECTED_STATE,
        "expected_date": EXPECTED_EVENT_DATE,
        "expected_time": EXPECTED_EVENT_TIME,
        "expected_venue": EXPECTED_VENUE_NAME,
        "expected_capacity": EXPECTED_VENUE_CAPACITY
    }, gt_type="expected_values")


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
) -> Dict[str, Any]:
    """
    Evaluate an answer for the Arizona Speaker Series journalist event task.
    """
    # Initialize evaluator
    evaluator = Evaluator()
    evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Root container, non-critical
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

    # Extract structured event package from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_event_package(),
        template_class=ExtractedEventPackage,
        extraction_name="extracted_event_package"
    )

    # Build verification tree and run checks
    await build_verification_tree(evaluator, extracted)

    # Return summary
    return evaluator.get_summary()