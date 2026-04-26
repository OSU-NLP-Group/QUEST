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
TASK_ID = "outdoor_amphitheaters_us"
TASK_DESCRIPTION = (
    "Identify four outdoor amphitheaters in the United States that meet the following criteria:\n"
    "- Each venue must have a seating capacity between 5,000 and 20,000 people\n"
    "- Each venue must be recognized for its natural acoustic properties or naturally-formed geographic features that contribute to sound quality\n"
    "- The four venues must be located in four different U.S. states\n"
    "- Each venue must be currently operational\n\n"
    "For each venue, provide:\n"
    "- The official venue name\n"
    "- The complete street address (including street number, street name, city, state, and ZIP code)\n"
    "- The exact seating capacity\n"
    "- The U.S. state where it is located\n"
    "- A reference URL from the official venue website or a verified source"
)

CURRENT_YEAR = 2026

# --------------------------------------------------------------------------- #
# Extraction models                                                           #
# --------------------------------------------------------------------------- #
class VenueItem(BaseModel):
    name: Optional[str] = None
    street_number_and_name: Optional[str] = None
    city: Optional[str] = None
    state: Optional[str] = None
    zip_code: Optional[str] = None
    full_address: Optional[str] = None
    capacity: Optional[str] = None
    type_desc: Optional[str] = None
    natural_acoustics_evidence: Optional[str] = None
    operational_status_desc: Optional[str] = None
    reference_urls: List[str] = Field(default_factory=list)


class VenuesExtraction(BaseModel):
    venues: List[VenueItem] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_venues() -> str:
    return (
        "Extract up to four venues from the answer that best match the request for outdoor amphitheaters in the U.S. "
        "For each venue, return a JSON object with the following fields:\n"
        "- name: Official venue name.\n"
        "- street_number_and_name: The street number and street name.\n"
        "- city: City.\n"
        "- state: U.S. state.\n"
        "- zip_code: ZIP code.\n"
        "- full_address: If the answer provides a single complete address string, include it here; otherwise, leave null.\n"
        "- capacity: The seating capacity string as written in the answer (do not convert).\n"
        "- type_desc: Any description from the answer indicating the venue is an outdoor amphitheater designed for live music performances.\n"
        "- natural_acoustics_evidence: Any text from the answer that indicates the venue is recognized for natural acoustic properties or naturally-formed geographic features contributing to sound quality.\n"
        "- operational_status_desc: Any text from the answer indicating the venue is currently operational/hosting concerts.\n"
        "- reference_urls: An array of URLs cited for this venue (official website or verified sources). If the answer includes markdown links, extract the raw URLs.\n\n"
        "Rules:\n"
        "1) Only extract URLs explicitly present in the answer. Do not invent URLs.\n"
        "2) If the answer includes more than four venues, return only the first four.\n"
        "3) If any field is missing in the answer, set it to null (or empty array for reference_urls).\n"
        "4) Do not transform the capacity value into a number; keep it as a string exactly as written.\n"
    )


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def compose_address(v: VenueItem) -> str:
    if v.full_address and v.full_address.strip():
        return v.full_address.strip()
    parts = [
        (v.street_number_and_name or "").strip(),
        (v.city or "").strip(),
        (v.state or "").strip(),
        (v.zip_code or "").strip(),
    ]
    # Join intelligently, skipping empties
    formatted = ", ".join([p for p in parts[:-1] if p])  # street, city, state
    if parts[-1]:
        if formatted:
            formatted = f"{formatted} {parts[-1]}"
        else:
            formatted = parts[-1]
    return formatted


def first_url(urls: List[str]) -> Optional[str]:
    for u in urls:
        if isinstance(u, str) and u.strip():
            return u.strip()
    return None


def normalize_state(s: Optional[str]) -> Optional[str]:
    if not s:
        return None
    return s.strip().upper()


# --------------------------------------------------------------------------- #
# Verification per-venue                                                      #
# --------------------------------------------------------------------------- #
async def verify_venue(
    evaluator: Evaluator,
    parent_node,
    venue: VenueItem,
    index: int,
) -> None:
    vid = index + 1
    venue_node = evaluator.add_parallel(
        id=f"Venue_{vid}",
        desc=f"{['First','Second','Third','Fourth'][index]} outdoor amphitheater meeting all specified criteria",
        parent=parent_node,
        critical=False,  # Allow partial credit per venue
    )

    # Name provided (non-critical existence)
    name_exists = bool(venue.name and venue.name.strip())
    evaluator.add_custom_node(
        result=name_exists,
        id=f"Venue_{vid}_Name",
        desc=f"Official name of the {['first','second','third','fourth'][index]} venue is provided",
        parent=venue_node,
        critical=False
    )

    # Reference URL validity (critical, and used as a prerequisite by other checks via auto preconditions)
    ref_leaf = evaluator.add_leaf(
        id=f"Venue_{vid}_Reference_URL",
        desc=f"Valid reference URL from official venue website or verified source is provided for the {['first','second','third','fourth'][index]} venue",
        parent=venue_node,
        critical=True
    )
    ref_url = first_url(venue.reference_urls)
    ref_claim_name = venue.name or f"venue #{vid}"
    await evaluator.verify(
        claim=f"This webpage is the official site of {ref_claim_name} or a verified source that provides authoritative information about it.",
        node=ref_leaf,
        sources=ref_url,
        additional_instruction=(
            "Assess whether this URL appears to be the official venue website (domain branding, About/Contact pages, Tickets/Events) "
            "or a verified authoritative source (e.g., Wikipedia with citations, city/government site, Ticketmaster listing). "
            "If no URL is provided, this check should fail."
        ),
    )

    # Type verification (critical)
    type_leaf = evaluator.add_leaf(
        id=f"Venue_{vid}_Type",
        desc=f"The {['first','second','third','fourth'][index]} venue is an outdoor amphitheater designed for live music performances",
        parent=venue_node,
        critical=True
    )
    await evaluator.verify(
        claim="This venue is an outdoor amphitheater designed for live music performances.",
        node=type_leaf,
        sources=venue.reference_urls,
        additional_instruction=(
            "Look for language on the referenced page(s) indicating 'outdoor amphitheater', 'open-air', "
            "'amphitheatre', 'live music venue', or similar. Minor variations and synonyms are acceptable."
        )
    )

    # Address verification (critical)
    address_leaf = evaluator.add_leaf(
        id=f"Venue_{vid}_Address",
        desc=f"Complete street address (street number, street name, city, state, ZIP code) for the {['first','second','third','fourth'][index]} venue is provided",
        parent=venue_node,
        critical=True
    )
    address_text = compose_address(venue).strip()
    await evaluator.verify(
        claim=f"The official street address of {ref_claim_name} is '{address_text}'.",
        node=address_leaf,
        sources=venue.reference_urls,
        additional_instruction=(
            "Verify the address (street number and name, city, state, ZIP) matches what's shown on the referenced page(s). "
            "Allow minor formatting differences (e.g., abbreviations like 'St.' vs 'Street'). "
            "If the provided address is incomplete or missing, this check should fail."
        ),
    )

    # Capacity range verification (critical)
    capacity_leaf = evaluator.add_leaf(
        id=f"Venue_{vid}_Capacity",
        desc=f"Seating capacity of the {['first','second','third','fourth'][index]} venue is between 5,000 and 20,000 people (inclusive)",
        parent=venue_node,
        critical=True
    )
    await evaluator.verify(
        claim="This venue’s seating capacity is between 5,000 and 20,000 people, inclusive.",
        node=capacity_leaf,
        sources=venue.reference_urls,
        additional_instruction=(
            "Check the capacity stated on the referenced page(s); if a specific number is given, verify it lies within 5,000–20,000. "
            "If there are multiple capacities (e.g., lawn + reserved), use the typical stated total capacity. "
            "If capacity is not stated or clearly outside range, this should fail."
        ),
    )

    # State location verification (critical)
    state_leaf = evaluator.add_leaf(
        id=f"Venue_{vid}_State",
        desc=f"U.S. state location of the {['first','second','third','fourth'][index]} venue is identified",
        parent=venue_node,
        critical=True
    )
    state_text = (venue.state or "").strip()
    await evaluator.verify(
        claim=f"This venue is located in the U.S. state of {state_text}.",
        node=state_leaf,
        sources=venue.reference_urls,
        additional_instruction=(
            "Use the address on the referenced page(s) to confirm the U.S. state. "
            "Allow abbreviations (e.g., 'CO' for Colorado) and minor variations."
        ),
    )

    # Natural acoustics recognition verification (critical)
    acoustics_leaf = evaluator.add_leaf(
        id=f"Venue_{vid}_Natural_Acoustics",
        desc=f"The {['first','second','third','fourth'][index]} venue is recognized for its natural acoustic properties or naturally-formed geographic features contributing to sound quality",
        parent=venue_node,
        critical=True
    )
    await evaluator.verify(
        claim="This venue is recognized for natural acoustic properties or naturally-formed geographic features that contribute to sound quality.",
        node=acoustics_leaf,
        sources=venue.reference_urls,
        additional_instruction=(
            "Look for mentions of rock formations, canyons, cliffs, hillsides, natural amphitheater shapes, limestone/granite outcrops, "
            "or explicit statements about natural acoustics. The referenced page(s) should clearly support the claim."
        ),
    )

    # Operational status verification (critical)
    operational_leaf = evaluator.add_leaf(
        id=f"Venue_{vid}_Operational_Status",
        desc=f"The {['first','second','third','fourth'][index]} venue is currently operational and actively hosting concerts as of {CURRENT_YEAR}",
        parent=venue_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"This venue is currently operational and actively hosting concerts as of {CURRENT_YEAR}.",
        node=operational_leaf,
        sources=venue.reference_urls,
        additional_instruction=(
            "Check for upcoming events, season schedules, or ticket pages indicating current activity in the present year. "
            "If the page indicates closure, hiatus, or solely historical info with no upcoming shows, this should fail."
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
    model: str = "o4-mini"
) -> Dict:
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Parallel aggregation for task completion
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

    # Root node modeled after "Task_Completion". Set as non-critical due to framework constraint
    task_node = evaluator.add_parallel(
        id="Task_Completion",
        desc="Identify four outdoor amphitheaters in the United States, each located in a different state, with seating capacity between 5,000 and 20,000, and recognized for natural acoustic properties. Provide complete information for each venue.",
        parent=root,
        critical=False
    )

    # Extract venues from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_venues(),
        template_class=VenuesExtraction,
        extraction_name="venues_extraction",
    )

    # Normalize: take first four venues and pad if fewer
    venues: List[VenueItem] = list(extracted.venues[:4])
    while len(venues) < 4:
        venues.append(VenueItem())

    # Verify each venue sequentially to ensure URL precondition checks occur before other criteria
    for i in range(4):
        await verify_venue(evaluator, task_node, venues[i], i)

    # Geographic diversity check (critical): four different U.S. states among the identified venues
    states = [normalize_state(v.state) for v in venues]
    unique_states = set([s for s in states if s])
    geo_ok = len(unique_states) == 4
    evaluator.add_custom_node(
        result=geo_ok,
        id="Geographic_Diversity",
        desc="All four identified venues are located in four different U.S. states",
        parent=task_node,
        critical=True
    )

    return evaluator.get_summary()