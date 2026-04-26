import asyncio
import logging
import re
from typing import Any, Optional, List, Dict

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "latin_grammy_2026_nfpa_eval"
TASK_DESCRIPTION = """
Identify the venue that hosted the 26th Annual Latin Grammy Awards in 2026 and determine its concert seating capacity. According to NFPA regulations, venues with an occupant load exceeding a specific threshold require a mandatory life safety evaluation. What is this threshold, and does the identified venue's capacity exceed it, thereby requiring such an evaluation? Additionally, confirm the specific date when the 26th Annual Latin Grammy Awards took place at this venue.
"""


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class VenueInfo(BaseModel):
    name: Optional[str] = None
    city: Optional[str] = None
    state: Optional[str] = None
    venue_url: Optional[str] = None


class CapacityInfo(BaseModel):
    capacity_value: Optional[str] = None
    capacity_url: Optional[str] = None


class RegulationInfo(BaseModel):
    threshold_value: Optional[str] = None
    regulation_url: Optional[str] = None


class ComplianceInfo(BaseModel):
    exceeds_threshold: Optional[str] = None  # "Yes" or "No"
    requirement_applies: Optional[str] = None  # "Yes" or "No"


class EventInfo(BaseModel):
    event_date: Optional[str] = None
    date_url: Optional[str] = None


class AnswerExtraction(BaseModel):
    venue: Optional[VenueInfo] = None
    capacity: Optional[CapacityInfo] = None
    regulation: Optional[RegulationInfo] = None
    compliance: Optional[ComplianceInfo] = None
    event: Optional[EventInfo] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_all_fields() -> str:
    return """
    Extract the following information exactly as presented in the answer. Do not invent or infer any missing data. If a field is missing, return null for that field. Extract only URLs that are explicitly present in the answer.

    Return a single JSON object with the following nested structure:

    {
      "venue": {
        "name": string | null,
        "city": string | null,
        "state": string | null,
        "venue_url": string | null
      },
      "capacity": {
        "capacity_value": string | null,
        "capacity_url": string | null
      },
      "regulation": {
        "threshold_value": string | null,
        "regulation_url": string | null
      },
      "compliance": {
        "exceeds_threshold": "Yes" | "No" | null,
        "requirement_applies": "Yes" | "No" | null
      },
      "event": {
        "event_date": string | null,
        "date_url": string | null
      }
    }

    Field-by-field instructions:
    - venue.name: The venue that hosted the 26th Annual Latin Grammy Awards in 2026.
    - venue.city / venue.state: The venue's city and state.
    - venue.venue_url: A single URL that explicitly confirms this venue hosted the 26th Annual Latin Grammy Awards in 2026.
    - capacity.capacity_value: The concert/event seating capacity at the identified venue (use the number as shown, including commas or descriptors).
    - capacity.capacity_url: A single URL that explicitly states the venue's concert/event seating capacity.
    - regulation.threshold_value: The NFPA occupant load threshold (number of people) that triggers a mandatory life safety evaluation requirement for assembly occupancies. Extract the number exactly as stated.
    - regulation.regulation_url: A single authoritative URL (e.g., NFPA or recognized compliance documentation) that supports the threshold requirement.
    - compliance.exceeds_threshold: "Yes" if the venue capacity exceeds the NFPA threshold; "No" otherwise. If not explicitly stated in the answer, return null.
    - compliance.requirement_applies: "Yes" if a life safety evaluation is required based on the capacity-to-threshold comparison; "No" otherwise. If not explicitly stated in the answer, return null.
    - event.event_date: The specific date (month, day, year) when the 26th Annual Latin Grammy Awards occurred at the venue.
    - event.date_url: A single URL that confirms the event date.

    URL extraction rules:
    - Extract only valid URLs present in the answer text (plain URLs or markdown links).
    - If a URL lacks a protocol, prepend http://.
    - Do not fabricate URLs or convert non-URL references (e.g., "according to NFPA") into URLs.
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def sanitize(s: Optional[str]) -> str:
    return (s or "").strip()


def extract_numeric_int(value: Optional[str]) -> Optional[int]:
    """
    Attempt to parse a human-written number into an integer.
    Handles commas, "k" (thousands), and plain digit sequences.
    Returns None if parsing fails.
    """
    if not value:
        return None
    text = value.lower().strip()

    # Handle "20k", "20 k"
    m = re.search(r'(\d+(?:\.\d+)?)\s*k\b', text)
    if m:
        try:
            return int(round(float(m.group(1)) * 1000))
        except Exception:
            pass

    # Handle "1m", "1 million"
    m = re.search(r'(\d+(?:\.\d+)?)\s*m(?:illion)?\b', text)
    if m:
        try:
            return int(round(float(m.group(1)) * 1_000_000))
        except Exception:
            pass

    # Plain digits (first sequence)
    digits = re.findall(r'\d+', text.replace(",", ""))
    if digits:
        try:
            return int(digits[0])
        except Exception:
            return None

    return None


# --------------------------------------------------------------------------- #
# Verification subtrees                                                       #
# --------------------------------------------------------------------------- #
async def verify_venue_identification(evaluator: Evaluator, parent_node, ext: AnswerExtraction) -> None:
    venue = ext.venue or VenueInfo()
    venue_node = evaluator.add_parallel(
        id="venue_identification",
        desc="Identify the specific venue that hosted the 26th Annual Latin Grammy Awards in 2026",
        parent=parent_node,
        critical=True
    )

    # Existence check for venue URL (source-grounding gate)
    evaluator.add_custom_node(
        result=bool(sanitize(venue.venue_url)),
        id="venue_url_present",
        desc="Venue confirmation URL is provided",
        parent=venue_node,
        critical=True
    )

    # Leaf: Venue name (verify via venue URL)
    name_leaf = evaluator.add_leaf(
        id="venue_name",
        desc="Provide the correct name of the venue",
        parent=venue_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The venue's name is '{sanitize(venue.name)}'.",
        node=name_leaf,
        sources=sanitize(venue.venue_url),
        additional_instruction="Verify on the provided page that the venue hosting the 26th Annual Latin Grammy Awards in 2026 is named as claimed. Allow minor naming variations or language differences."
    )

    # Leaf: Venue location (verify via venue URL)
    location_leaf = evaluator.add_leaf(
        id="venue_location",
        desc="Provide the correct city and state where the venue is located",
        parent=venue_node,
        critical=True
    )
    city_state = ", ".join([p for p in [sanitize(venue.city), sanitize(venue.state)] if p])
    await evaluator.verify(
        claim=f"The venue is located in {city_state}.",
        node=location_leaf,
        sources=sanitize(venue.venue_url),
        additional_instruction="Verify that the page confirms the venue's city and state. Allow minor formatting differences and abbreviations."
    )

    # Leaf: Venue URL confirms hosting the event
    url_ref_leaf = evaluator.add_leaf(
        id="venue_url_reference",
        desc="Provide a URL that confirms this venue hosted the 26th Annual Latin Grammy Awards in 2026",
        parent=venue_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The 26th Annual Latin Grammy Awards in 2026 took place at {sanitize(venue.name)}.",
        node=url_ref_leaf,
        sources=sanitize(venue.venue_url),
        additional_instruction="Verify that the provided URL clearly states the venue and that it hosted the 26th Annual Latin Grammy Awards in 2026."
    )


async def verify_capacity_verification(evaluator: Evaluator, parent_node, ext: AnswerExtraction) -> None:
    capacity = ext.capacity or CapacityInfo()
    cap_node = evaluator.add_parallel(
        id="capacity_verification",
        desc="Determine the concert seating capacity of the identified venue",
        parent=parent_node,
        critical=True
    )

    # Existence check for capacity URL
    evaluator.add_custom_node(
        result=bool(sanitize(capacity.capacity_url)),
        id="capacity_url_present",
        desc="Capacity confirmation URL is provided",
        parent=cap_node,
        critical=True
    )

    # Leaf: Capacity value supported by URL
    cap_val_leaf = evaluator.add_leaf(
        id="capacity_value",
        desc="Provide the accurate seating capacity number for concerts/events at this venue",
        parent=cap_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The concert/event seating capacity at the venue is '{sanitize(capacity.capacity_value)}'.",
        node=cap_val_leaf,
        sources=sanitize(capacity.capacity_url),
        additional_instruction="Verify that the provided page explicitly states the venue's concert/event seating capacity (not a different configuration). Allow reasonable wording variations."
    )

    # Leaf: Capacity URL supports the capacity claim
    cap_url_leaf = evaluator.add_leaf(
        id="capacity_url_reference",
        desc="Provide a URL that confirms the seating capacity of this venue",
        parent=cap_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"This webpage explicitly confirms the venue's concert/event seating capacity as '{sanitize(capacity.capacity_value)}'.",
        node=cap_url_leaf,
        sources=sanitize(capacity.capacity_url),
        additional_instruction="Confirm that the source clearly states the capacity figure and it pertains to the venue's concert/event configuration."
    )


async def verify_regulation_threshold(evaluator: Evaluator, parent_node, ext: AnswerExtraction) -> None:
    regulation = ext.regulation or RegulationInfo()
    reg_node = evaluator.add_parallel(
        id="regulation_threshold",
        desc="State the occupant load threshold that triggers life safety evaluation requirements according to NFPA",
        parent=parent_node,
        critical=True
    )

    # Existence check for regulation URL
    evaluator.add_custom_node(
        result=bool(sanitize(regulation.regulation_url)),
        id="regulation_url_present",
        desc="NFPA regulation URL is provided",
        parent=reg_node,
        critical=True
    )

    # Leaf: Threshold value supported by NFPA URL
    thr_leaf = evaluator.add_leaf(
        id="threshold_value",
        desc="Provide the correct occupant load number (in people) that triggers mandatory life safety evaluation according to NFPA regulations",
        parent=reg_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"According to NFPA, the occupant load threshold that mandates a life safety evaluation is '{sanitize(regulation.threshold_value)}' people.",
        node=thr_leaf,
        sources=sanitize(regulation.regulation_url),
        additional_instruction="Verify on the provided page that NFPA requires a life safety evaluation for assembly occupancies at or above the specified threshold. Accept authoritative summaries if directly from NFPA or official adoptions."
    )

    # Leaf: Regulation URL confirms threshold requirement
    reg_url_leaf = evaluator.add_leaf(
        id="regulation_url_reference",
        desc="Provide a URL that confirms this NFPA life safety evaluation threshold requirement",
        parent=reg_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"This webpage confirms that NFPA mandates a life safety evaluation at the occupant load threshold '{sanitize(regulation.threshold_value)}'.",
        node=reg_url_leaf,
        sources=sanitize(regulation.regulation_url),
        additional_instruction="Focus on the life safety evaluation requirement and the specific occupant load threshold mentioned."
    )


async def verify_compliance_determination(evaluator: Evaluator, parent_node, ext: AnswerExtraction) -> None:
    capacity = ext.capacity or CapacityInfo()
    regulation = ext.regulation or RegulationInfo()
    compliance = ext.compliance or ComplianceInfo()

    comp_node = evaluator.add_parallel(
        id="compliance_determination",
        desc="Determine whether the venue's capacity exceeds the stated NFPA threshold, thereby requiring life safety evaluation",
        parent=parent_node,
        critical=True
    )

    # Leaf: Exceeds threshold (logical verification)
    exceeds_leaf = evaluator.add_leaf(
        id="exceeds_threshold",
        desc="Correctly state whether the venue's capacity exceeds the NFPA threshold identified above (Yes/No)",
        parent=comp_node,
        critical=True
    )
    # Build a purely logical claim that references both numbers; judge must evaluate correctness
    cap_val_str = sanitize(capacity.capacity_value)
    thr_val_str = sanitize(regulation.threshold_value)
    exceeds_claim = (
        f"With a capacity of '{cap_val_str}' and an NFPA threshold of '{thr_val_str}', "
        f"the capacity exceeds the threshold: {sanitize(compliance.exceeds_threshold)}."
    )
    await evaluator.verify(
        claim=exceeds_claim,
        node=exceeds_leaf,
        additional_instruction=(
            "Determine if the statement is logically correct by comparing the numeric values in the strings. "
            "Treat 'exceeds' as strictly greater than (equal to is not exceeding). "
            "Ignore commas and units; interpret 'k' as thousands when present."
        )
    )

    # Leaf: Requirement applies (logical conclusion)
    req_leaf = evaluator.add_leaf(
        id="requirement_applies",
        desc="Correctly conclude whether life safety evaluation is required for this venue based on the capacity-to-threshold comparison",
        parent=comp_node,
        critical=True
    )
    req_claim = (
        f"Based on whether the venue's capacity exceeds the NFPA threshold, "
        f"a mandatory life safety evaluation is required for this venue: {sanitize(compliance.requirement_applies)}."
    )
    await evaluator.verify(
        claim=req_claim,
        node=req_leaf,
        additional_instruction=(
            "If capacity exceeds the NFPA threshold, the correct conclusion is 'Yes'; "
            "if capacity is less than or equal to the threshold, the correct conclusion is 'No'."
        )
    )


async def verify_event_date_confirmation(evaluator: Evaluator, parent_node, ext: AnswerExtraction) -> None:
    event = ext.event or EventInfo()

    date_node = evaluator.add_parallel(
        id="event_date_confirmation",
        desc="Confirm the specific date when the 26th Annual Latin Grammy Awards took place at this venue",
        parent=parent_node,
        critical=False  # Non-critical to allow partial credit if core items pass
    )

    # Existence check for date URL (gate)
    evaluator.add_custom_node(
        result=bool(sanitize(event.date_url)),
        id="date_url_present",
        desc="Event date confirmation URL is provided",
        parent=date_node,
        critical=True
    )

    # Leaf: Event date value (verified by URL)
    date_leaf = evaluator.add_leaf(
        id="event_date",
        desc="Provide the correct date (month, day, and year) of the 26th Annual Latin Grammy Awards",
        parent=date_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The 26th Annual Latin Grammy Awards took place on {sanitize(event.event_date)}.",
        node=date_leaf,
        sources=sanitize(event.date_url),
        additional_instruction="Verify that the page clearly states the exact date (month, day, year) of the 26th Annual Latin Grammy Awards."
    )

    # Leaf: Date URL confirms the date
    date_url_leaf = evaluator.add_leaf(
        id="date_url_reference",
        desc="Provide a URL that confirms the date of the 26th Annual Latin Grammy Awards",
        parent=date_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"This webpage confirms the date of the 26th Annual Latin Grammy Awards as {sanitize(event.event_date)}.",
        node=date_url_leaf,
        sources=sanitize(event.date_url),
        additional_instruction="Confirm that the source explicitly states the event date, focusing on the 26th edition."
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
    Evaluate an agent's answer for the Latin Grammy 2026 NFPA threshold and venue capacity task.
    """
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.SEQUENTIAL,  # Enforce task order and skip later checks if earlier fail
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

    # Root should be non-critical to allow partial credit (date section is non-critical)
    root.critical = False

    # Extract structured information once
    extraction = await evaluator.extract(
        prompt=prompt_extract_all_fields(),
        template_class=AnswerExtraction,
        extraction_name="answer_extraction"
    )

    # Build and verify the tree following rubric order (sequential at root)
    await verify_venue_identification(evaluator, root, extraction)
    await verify_capacity_verification(evaluator, root, extraction)
    await verify_regulation_threshold(evaluator, root, extraction)
    await verify_compliance_determination(evaluator, root, extraction)
    await verify_event_date_confirmation(evaluator, root, extraction)

    return evaluator.get_summary()