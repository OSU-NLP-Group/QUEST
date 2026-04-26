import asyncio
import logging
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "boston_gaming_convention_2026"
TASK_DESCRIPTION = (
    "Identify the major gaming convention taking place in Boston, Massachusetts during late March 2026. "
    "Provide the following information: the event name, the exact start and end dates, the venue name and "
    "location, and a reference URL to an official or authoritative source."
)


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class EventExtraction(BaseModel):
    event_name: Optional[str] = None
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    venue_name: Optional[str] = None
    venue_location: Optional[str] = None  # e.g., "Boston, MA" or an address; extract as written
    reference_urls: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt helpers                                                   #
# --------------------------------------------------------------------------- #
def prompt_extract_event() -> str:
    return """
    Extract the event details for a major gaming convention in Boston, MA during late March 2026, as presented in the answer.
    Extract the following fields exactly as they appear in the answer text:

    - event_name: The explicit event name (e.g., "PAX East 2026")
    - start_date: The exact start date provided (e.g., "March 26, 2026" or "2026-03-26")
    - end_date: The exact end date provided
    - venue_name: The venue name (e.g., "Boston Convention and Exhibition Center" or "Hynes Convention Center")
    - venue_location: The venue location as written in the answer (at least city/state; e.g., "Boston, MA")
    - reference_urls: A list of all URLs cited in the answer as sources for this event (official websites or reputable outlets).
      Include the full URLs. If none are given, return an empty list.

    Rules:
    - Do not infer or invent any values. Only extract information that explicitly appears in the answer.
    - If a field is not present in the answer, set it to null (or an empty list for reference_urls).
    - Keep the original formatting of dates and names as they appear.
    """


# --------------------------------------------------------------------------- #
# Verification helpers                                                        #
# --------------------------------------------------------------------------- #
async def build_required_output_fields(
    evaluator: Evaluator,
    parent_node,
    info: EventExtraction
) -> None:
    """
    Build and verify the 'Required_Output_Fields' subtree first to ensure gating for constraints.
    """
    required_node = evaluator.add_parallel(
        id="Required_Output_Fields",
        desc="Answer provides all requested fields: event name, exact start/end dates, venue name and location, and an authoritative reference URL.",
        parent=parent_node,
        critical=True
    )

    # Event name provided
    evaluator.add_custom_node(
        result=(info.event_name is not None and str(info.event_name).strip() != ""),
        id="Event_Name_Provided",
        desc="Event name is explicitly provided.",
        parent=required_node,
        critical=True
    )

    # Start date provided
    evaluator.add_custom_node(
        result=(info.start_date is not None and str(info.start_date).strip() != ""),
        id="Start_Date_Provided",
        desc="Exact start date is explicitly provided.",
        parent=required_node,
        critical=True
    )

    # End date provided
    evaluator.add_custom_node(
        result=(info.end_date is not None and str(info.end_date).strip() != ""),
        id="End_Date_Provided",
        desc="Exact end date is explicitly provided.",
        parent=required_node,
        critical=True
    )

    # Venue name and location provided
    evaluator.add_custom_node(
        result=(
            info.venue_name is not None and str(info.venue_name).strip() != "" and
            info.venue_location is not None and str(info.venue_location).strip() != ""
        ),
        id="Venue_Name_And_Location_Provided",
        desc="Venue name and venue location (at least city/state) are explicitly provided.",
        parent=required_node,
        critical=True
    )

    # Authoritative reference URL provided and supports event details
    # If no URLs are provided, mark as failed directly; otherwise verify with URLs
    if info.reference_urls and len(info.reference_urls) > 0:
        auth_leaf = evaluator.add_leaf(
            id="Authoritative_Reference_URL_Provided",
            desc="At least one authoritative reference URL is provided (official event website or reputable major outlet) that supports the event details.",
            parent=required_node,
            critical=True
        )
        event_ref = info.event_name or "the event"
        claim = (
            f"At least one of the provided URLs is an official event website or a reputable major outlet page, "
            f"and it provides the key details (name and/or dates and/or venue/location) for {event_ref} in Boston, Massachusetts in March 2026."
        )
        await evaluator.verify(
            claim=claim,
            node=auth_leaf,
            sources=info.reference_urls,
            additional_instruction=(
                "Accept official domains (e.g., the event's own site) or reputable major outlets (e.g., major gaming "
                "publications or the venue's official site). The page should clearly present event details (at least "
                "name and dates or venue/location). If none of the provided pages are official or reputable, or if they "
                "do not include event details, mark as not supported."
            )
        )
    else:
        evaluator.add_custom_node(
            result=False,
            id="Authoritative_Reference_URL_Provided",
            desc="At least one authoritative reference URL is provided (official event website or reputable major outlet) that supports the event details.",
            parent=required_node,
            critical=True
        )


async def build_event_matches_constraints(
    evaluator: Evaluator,
    parent_node,
    info: EventExtraction
) -> None:
    """
    Build and verify the 'Event_Matches_Constraints' subtree.
    This is evaluated after Required_Output_Fields so that missing/invalid sources can gate these checks.
    """
    constraints_node = evaluator.add_parallel(
        id="Event_Matches_Constraints",
        desc="The identified event satisfies the question constraints (major gaming convention, Boston MA, late March 2026).",
        parent=parent_node,
        critical=True
    )

    # Major_Gaming_Convention
    major_leaf = evaluator.add_leaf(
        id="Major_Gaming_Convention",
        desc="Event is a major gaming convention (not a small local meetup/single-day minor event); evidence indicates notable scale/prominence.",
        parent=constraints_node,
        critical=True
    )
    event_ref = info.event_name or "the event"
    claim_major = (
        f"The referenced page(s) indicate that {event_ref} is a major gaming convention or expo of notable scale or prominence "
        f"(e.g., widely recognized, large attendance, or substantial industry/consumer presence)."
    )
    await evaluator.verify(
        claim=claim_major,
        node=major_leaf,
        sources=info.reference_urls if info.reference_urls else None,
        additional_instruction=(
            "Assess whether the page portrays the event as a significant gaming convention (not a local/minor meetup). "
            "Signals include descriptors like 'major', 'large', 'annual convention/expo', attendance estimates in the thousands, "
            "or coverage by well-known outlets. Focus only on what the provided page(s) state."
        )
    )

    # Location_Boston_MA
    location_leaf = evaluator.add_leaf(
        id="Location_Boston_MA",
        desc="Event takes place in Boston, Massachusetts (venue/city/state match).",
        parent=constraints_node,
        critical=True
    )
    venue_clause = f" at the venue '{info.venue_name}'" if info.venue_name else ""
    claim_location = (
        f"The event {event_ref}{venue_clause} takes place in Boston, Massachusetts, United States."
    )
    await evaluator.verify(
        claim=claim_location,
        node=location_leaf,
        sources=info.reference_urls if info.reference_urls else None,
        additional_instruction=(
            "Look for explicit location indicators on the page such as 'Boston, MA' or 'Boston, Massachusetts'. "
            "Venue names like 'Boston Convention and Exhibition Center (BCEC)' or 'Hynes Convention Center' are valid Boston venues."
        )
    )

    # Timing_Late_March_2026
    timing_leaf = evaluator.add_leaf(
        id="Timing_Late_March_2026",
        desc="Event start and end dates occur during late March 2026 (i.e., the latter part of March 2026, consistent with the question wording).",
        parent=constraints_node,
        critical=True
    )
    claim_timing = (
        "The event's schedule on the referenced page shows that the start and end dates both fall within late March 2026, "
        "defined as March 20–31, 2026 (inclusive)."
    )
    await evaluator.verify(
        claim=claim_timing,
        node=timing_leaf,
        sources=info.reference_urls if info.reference_urls else None,
        additional_instruction=(
            "Check the dates listed on the page. Mark as supported only if BOTH the start date and end date are within "
            "March 20–31, 2026 (inclusive). If either date lies outside this range, it does not satisfy 'late March 2026'."
        )
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
    Evaluate an answer for the Boston late-March 2026 major gaming convention task.
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

    # Extract event information from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_event(),
        template_class=EventExtraction,
        extraction_name="event_extraction"
    )

    # Optional: add custom info for transparency (definition used for "late March")
    evaluator.add_custom_info(
        info={
            "late_march_definition": "March 20–31, 2026 (inclusive)",
            "notes": "Timing_Late_March_2026 requires both start and end dates within this window."
        },
        info_type="policy",
        info_name="timing_policy"
    )

    # Build top-level critical node
    gaming_node = evaluator.add_parallel(
        id="Gaming_Event_Identification",
        desc="Identify a major gaming convention in Boston, MA occurring in late March 2026 and provide the required details with an authoritative reference URL.",
        parent=root,
        critical=True
    )

    # Build and verify 'Required_Output_Fields' first to gate constraint checks
    await build_required_output_fields(evaluator, gaming_node, extracted)

    # Build and verify 'Event_Matches_Constraints'
    await build_event_matches_constraints(evaluator, gaming_node, extracted)

    # Return evaluation summary
    return evaluator.get_summary()