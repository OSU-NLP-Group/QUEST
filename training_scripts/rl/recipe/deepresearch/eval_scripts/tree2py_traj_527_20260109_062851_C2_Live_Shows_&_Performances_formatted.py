import asyncio
import logging
import re
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "nyc_concert_venue_ada_capacity"
TASK_DESCRIPTION = (
    "Identify a concert venue located in New York City, New York, that has a seating capacity "
    "between 5,000 and 7,000 seats for concerts and provides wheelchair-accessible seating. "
    "For your answer, provide: (1) The name of the venue, (2) The venue's concert seating capacity, "
    "(3) A reference URL from the venue's official website or a recognized venue information database that confirms the seating capacity, "
    "(4) Confirmation that the venue provides wheelchair-accessible seating, and (5) A reference URL from the venue's official website or a recognized information source "
    "that documents the wheelchair-accessible seating availability."
)


# --------------------------------------------------------------------------- #
# Data models for structured extraction                                       #
# --------------------------------------------------------------------------- #
class VenueExtraction(BaseModel):
    """
    Structured extraction for the requested venue information from the answer.
    All fields should be extracted exactly as presented in the answer text.
    """
    venue_name: Optional[str] = None

    # Location as presented in the answer (e.g., "New York, NY", "New York City, New York", a borough, etc.)
    location_text: Optional[str] = None

    # Concert seating capacity information as presented in the answer
    # capacity_value: single numeric string if the answer provides a specific number (e.g., "6015")
    capacity_value: Optional[str] = None
    # capacity_text: the exact textual snippet from the answer describing capacity (e.g., "about 6,000", "6,015 seats")
    capacity_text: Optional[str] = None

    # Capacity source URLs explicitly provided in the answer that are claimed to confirm capacity
    capacity_source_urls: List[str] = Field(default_factory=list)

    # Accessibility statements (as presented in the answer)
    wheelchair_accessible_text: Optional[str] = None
    ada_compliance_text: Optional[str] = None

    # Accessibility source URLs explicitly provided in the answer that document accessible seating and/or ADA compliance
    accessibility_source_urls: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_venue_info() -> str:
    return """
    Extract the following information exactly as presented in the answer text:

    1) venue_name: The venue's name.
    2) location_text: Any location text provided for the venue (e.g., "New York City, New York", "NYC", "New York, NY", or a borough within NYC).
    3) capacity_value: If the answer explicitly provides a single numeric concert seating capacity (e.g., "6015"), extract that number only as a string without commas. If the answer does not clearly provide a single numeric value, return null for this field.
    4) capacity_text: The exact text from the answer that describes the concert seating capacity (e.g., "6,015 seats", "about 6,000", "5,600 to 5,800"). Always return this if any capacity description is present; otherwise return null.
    5) capacity_source_urls: All URLs explicitly provided in the answer that are claimed to confirm the seating capacity. Extract actual URLs only.
    6) wheelchair_accessible_text: The exact text confirming wheelchair-accessible seating (e.g., "wheelchair-accessible seating is provided"). If not explicitly stated, return null.
    7) ada_compliance_text: The exact text indicating ADA compliance (e.g., "ADA-compliant", "compliant with ADA requirements"). If not explicitly stated, return null.
    8) accessibility_source_urls: All URLs explicitly provided in the answer that document wheelchair-accessible seating availability and/or ADA accessibility. Extract actual URLs only.

    IMPORTANT RULES FOR URL EXTRACTION:
    - Extract only actual URLs explicitly shown in the answer (plain URLs or in markdown link format).
    - Ensure each extracted URL includes the protocol (http:// or https://). If a URL is missing a protocol, prepend http://.
    - Ignore any non-URL citations or general references that lack a concrete URL.
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def _is_valid_url(url: Optional[str]) -> bool:
    if not url or not isinstance(url, str):
        return False
    return url.strip().lower().startswith(("http://", "https://"))


def _any_valid_urls(urls: List[str]) -> bool:
    return any(_is_valid_url(u) for u in urls)


def _parse_first_int(text: Optional[str]) -> Optional[int]:
    """
    Parse the first integer found in a string, handling thousand separators.
    Returns None if no integer-like token is found.
    """
    if not text:
        return None
    match = re.search(r"(\d{1,3}(?:,\d{3})+|\d+)", text)
    if not match:
        return None
    token = match.group(1).replace(",", "")
    try:
        return int(token)
    except Exception:
        return None


def _resolve_capacity_number(extracted: VenueExtraction) -> Optional[int]:
    """
    Resolve a capacity integer from extracted fields.
    Prefer 'capacity_value' if present, otherwise parse the first int from 'capacity_text'.
    """
    if extracted.capacity_value:
        try:
            return int(extracted.capacity_value)
        except Exception:
            pass
    return _parse_first_int(extracted.capacity_text)


def _nyc_location_claim(venue_name: Optional[str]) -> str:
    """
    Construct a location claim for verification.
    """
    if venue_name and venue_name.strip():
        return f"The venue '{venue_name.strip()}' is located in New York City, New York (NYC)."
    return "The selected venue is located in New York City, New York (NYC)."


def _capacity_exact_claim(venue_name: Optional[str], capacity: int) -> str:
    """
    Construct an exact capacity claim for verification.
    """
    if venue_name and venue_name.strip():
        return f"The concert seating capacity of {venue_name.strip()} is {capacity} seats."
    return f"The venue's concert seating capacity is {capacity} seats."


def _capacity_range_claim(venue_name: Optional[str]) -> str:
    """
    Construct a range-based capacity claim for verification when an exact number is unavailable.
    """
    if venue_name and venue_name.strip():
        return f"The concert seating capacity of {venue_name.strip()} is between 5,000 and 7,000 seats."
    return "The venue's concert seating capacity is between 5,000 and 7,000 seats."


def _wheelchair_accessible_claim(venue_name: Optional[str]) -> str:
    if venue_name and venue_name.strip():
        return f"{venue_name.strip()} provides wheelchair-accessible seating."
    return "The venue provides wheelchair-accessible seating."


def _ada_compliance_claim(venue_name: Optional[str]) -> str:
    if venue_name and venue_name.strip():
        return f"{venue_name.strip()} indicates ADA-compliant accessibility (compliant with ADA requirements)."
    return "The venue indicates ADA-compliant accessibility (compliant with ADA requirements)."


# --------------------------------------------------------------------------- #
# Verification subtrees                                                       #
# --------------------------------------------------------------------------- #
async def build_and_verify_capacity_requirement(
    evaluator: Evaluator,
    parent_node,
    extracted: VenueExtraction,
) -> None:
    """
    Build the CapacityRequirement subtree and run verifications.
    """
    cap_node = evaluator.add_parallel(
        id="CapacityRequirement",
        desc="The venue's concert seating capacity must be between 5,000 and 7,000 seats and be supported by an appropriate reference URL.",
        parent=parent_node,
        critical=True,
    )

    # CapacityValueInRange (custom binary check)
    capacity_int = _resolve_capacity_number(extracted)
    in_range = capacity_int is not None and 5000 <= capacity_int <= 7000

    evaluator.add_custom_node(
        result=in_range,
        id="CapacityValueInRange",
        desc="The provided concert seating capacity value is between 5,000 and 7,000 seats (inclusive).",
        parent=cap_node,
        critical=True,
    )

    # CapacitySourceURLProvided (custom binary check)
    evaluator.add_custom_node(
        result=_any_valid_urls(extracted.capacity_source_urls),
        id="CapacitySourceURLProvided",
        desc="Provide a reference URL from the venue's official website or a recognized venue information database that confirms the capacity.",
        parent=cap_node,
        critical=True,
    )

    # CapacitySourceSupportsClaim (leaf verification against cited URLs)
    # Although not explicitly listed in the rubric leaves, this is the concrete verification that the cited URL confirms the capacity.
    cap_support_node = evaluator.add_leaf(
        id="CapacitySourceSupportsClaim",
        desc="The capacity source URL(s) confirm the venue's concert seating capacity.",
        parent=cap_node,
        critical=True,
    )

    claim = (
        _capacity_exact_claim(extracted.venue_name, capacity_int)
        if capacity_int is not None
        else _capacity_range_claim(extracted.venue_name)
    )

    await evaluator.verify(
        claim=claim,
        node=cap_support_node,
        sources=extracted.capacity_source_urls,
        additional_instruction=(
            "Verify that the cited source explicitly confirms the concert seating capacity. "
            "If multiple capacities are mentioned on the page (e.g., for sports vs. concerts), "
            "use the concert seating capacity. Allow minor rounding differences (e.g., 6,000 vs 6,015). "
            "If the source is irrelevant or does not confirm capacity, mark as not supported."
        ),
    )


async def build_and_verify_accessibility_requirement(
    evaluator: Evaluator,
    parent_node,
    extracted: VenueExtraction,
) -> None:
    """
    Build the AccessibilityRequirement subtree and run verifications.
    """
    acc_node = evaluator.add_parallel(
        id="AccessibilityRequirement",
        desc="The venue must provide ADA-compliant wheelchair-accessible seating and be supported by an appropriate reference URL.",
        parent=parent_node,
        critical=True,
    )

    # AccessibilitySourceURLProvided (custom binary check)
    evaluator.add_custom_node(
        result=_any_valid_urls(extracted.accessibility_source_urls),
        id="AccessibilitySourceURLProvided",
        desc="Provide a reference URL from the venue's official website or a recognized accessibility information source that documents wheelchair-accessible seating availability (and/or ADA accessibility).",
        parent=acc_node,
        critical=True,
    )

    # WheelchairAccessibleSeatingProvided (leaf verification against cited URLs)
    wh_node = evaluator.add_leaf(
        id="WheelchairAccessibleSeatingProvided",
        desc="Confirm the venue provides wheelchair-accessible seating.",
        parent=acc_node,
        critical=True,
    )
    await evaluator.verify(
        claim=_wheelchair_accessible_claim(extracted.venue_name),
        node=wh_node,
        sources=extracted.accessibility_source_urls,
        additional_instruction=(
            "Determine whether the cited source explicitly indicates the availability of wheelchair-accessible seating, "
            "including synonyms such as 'accessible seating', 'wheelchair seating', or 'ADA seating'. "
            "If the source is irrelevant, inaccessible, or does not document accessible seating, mark as not supported."
        ),
    )

    # ADAComplianceIndicated (leaf verification against cited URLs)
    ada_node = evaluator.add_leaf(
        id="ADAComplianceIndicated",
        desc="Confirm the venue's wheelchair-accessible seating/accessibility is indicated as ADA-compliant (or compliant with ADA requirements).",
        parent=acc_node,
        critical=True,
    )
    await evaluator.verify(
        claim=_ada_compliance_claim(extracted.venue_name),
        node=ada_node,
        sources=extracted.accessibility_source_urls,
        additional_instruction=(
            "Verify that the cited source indicates ADA compliance or explicitly references ADA requirements for accessibility. "
            "Accept reasonable phrasing variants such as 'ADA compliant', 'ADA accessibility', or 'compliant with ADA'. "
            "If the source lacks any ADA indication, mark as not supported."
        ),
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
    Evaluate an answer for the NYC concert venue ADA capacity task.
    Returns a structured summary including the verification tree and final score.
    """
    # Initialize evaluator (root is a non-critical container; we add a critical child for the rubric root)
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

    # Extract structured information from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_venue_info(),
        template_class=VenueExtraction,
        extraction_name="venue_extraction",
    )

    # Build the rubric root (critical)
    rubric_root = evaluator.add_parallel(
        id="VenueIdentification",
        desc="Identify a concert venue in New York City with concert seating capacity between 5,000 and 7,000 that provides ADA-compliant wheelchair-accessible seating, and provide required supporting URLs.",
        parent=root,
        critical=True,
    )

    # VenueNameProvided (custom binary check)
    evaluator.add_custom_node(
        result=bool(extracted.venue_name and extracted.venue_name.strip()),
        id="VenueNameProvided",
        desc="Provide the name of the venue.",
        parent=rubric_root,
        critical=True,
    )

    # GeographicLocation (leaf verification; use answer context)
    geo_node = evaluator.add_leaf(
        id="GeographicLocation",
        desc="The venue must be located in New York City, New York.",
        parent=rubric_root,
        critical=True,
    )
    await evaluator.verify(
        claim=_nyc_location_claim(extracted.venue_name),
        node=geo_node,
        additional_instruction=(
            "Verify based on the answer content whether the venue is located in New York City, New York. "
            "Allow common variants like 'NYC', 'New York, NY', or boroughs within NYC (Manhattan, Brooklyn, Queens, The Bronx, Staten Island). "
            "If the answer does not indicate NYC location, mark as incorrect."
        ),
    )

    # CapacityRequirement subtree
    await build_and_verify_capacity_requirement(evaluator, rubric_root, extracted)

    # AccessibilityRequirement subtree
    await build_and_verify_accessibility_requirement(evaluator, rubric_root, extracted)

    # Return structured evaluation summary
    return evaluator.get_summary()