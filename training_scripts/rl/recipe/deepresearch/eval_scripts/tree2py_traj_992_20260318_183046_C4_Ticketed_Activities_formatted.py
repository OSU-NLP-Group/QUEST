import asyncio
import logging
import math
import re
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "la_venues_ada_2026"
TASK_DESCRIPTION = """
Identify 3 entertainment venues in Los Angeles that are scheduled to host ticketed events between March and May 2026 and meet ADA wheelchair accessibility standards. For each venue, provide: (1) the venue name and complete street address, (2) the total seating capacity of the venue, and (3) verification of ADA-compliant wheelchair-accessible seating, including the specific number of wheelchair spaces available at the venue. Ensure that all three venues are distinct locations within the Los Angeles area and that each hosts ticketed entertainment events (such as concerts, theater performances, or sporting events) during the specified timeframe.
"""

START_DATE = "2026-03-01"
END_DATE = "2026-05-31"


# --------------------------------------------------------------------------- #
# Extraction Models                                                           #
# --------------------------------------------------------------------------- #
class VenueItem(BaseModel):
    name: Optional[str] = None
    address: Optional[str] = None  # Expect a complete street address if available
    capacity: Optional[str] = None  # Keep as free text to be robust; we'll parse when needed
    wheelchair_spaces: Optional[str] = None  # Specific number of wheelchair spaces (string as stated)
    venue_sources: List[str] = Field(default_factory=list)  # Official site, maps, city pages for address/location
    event_urls: List[str] = Field(default_factory=list)  # URLs to ticket/event pages explicitly between Mar–May 2026
    capacity_sources: List[str] = Field(default_factory=list)  # Sources that state seating capacity
    accessibility_sources: List[str] = Field(default_factory=list)  # Sources that state wheelchair spaces/ADA seating


class VenuesExtraction(BaseModel):
    venues: List[VenueItem] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction Prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_venues() -> str:
    return f"""
    Extract up to 3 distinct entertainment venues mentioned in the answer that are within the Los Angeles area (City of Los Angeles or any city in Los Angeles County).
    For each venue, extract the following fields exactly as they appear in the answer:
    - name: The venue's full name.
    - address: The complete street address string as presented (include street number, street name, city, state abbreviation, and ZIP code if available).
    - capacity: The venue's total seating capacity as a string (e.g., "18,000", "about 1,200").
    - wheelchair_spaces: The specific number of wheelchair-accessible seating spaces as a string (e.g., "at least 20", "24").
    - venue_sources: A list of URLs explicitly cited in the answer that substantiate the venue identification and address/location (official page, map listing, city page, etc.).
    - event_urls: A list of URLs explicitly cited in the answer that show ticketed entertainment events (concerts, theater, sports) scheduled to occur between {START_DATE} and {END_DATE} (inclusive) at this venue. Include only URLs that the answer actually provided.
    - capacity_sources: A list of URLs explicitly cited in the answer that substantiate the total seating capacity for this venue.
    - accessibility_sources: A list of URLs explicitly cited in the answer that substantiate ADA-compliant wheelchair-accessible seating and the specific number of wheelchair spaces.
    
    Important:
    - Only extract URLs that are explicitly present in the answer text (including markdown links). Do not invent URLs.
    - If a field is missing from the answer, set it to null (for strings) or an empty list (for lists).
    - Preserve strings exactly as written in the answer (do not normalize numbers).
    - Return a JSON object with a single key "venues" that is an array of up to 3 objects with the schema above, in the same order as they appear in the answer.
    """


# --------------------------------------------------------------------------- #
# Helpers                                                                     #
# --------------------------------------------------------------------------- #
def _parse_int_from_text(text: Optional[str]) -> Optional[int]:
    """
    Try to parse an integer from a free-form text like "about 1,200", "1.5k", "20–30", "at least 24".
    Returns None if no reasonable numeric value can be derived.
    """
    if not text:
        return None
    s = text.strip().lower()

    # Handle "1.5k" style
    m_k = re.search(r"(\d+(?:\.\d+)?)\s*k\b", s, flags=re.I)
    if m_k:
        try:
            return int(round(float(m_k.group(1)) * 1000))
        except Exception:
            pass

    # Handle ranges like "20-30" or "20–30": take the lower bound
    m_range = re.search(r"(\d[\d,\.]*)\s*[-–]\s*(\d[\d,\.]*)", s)
    if m_range:
        val = m_range.group(1).replace(",", "")
        try:
            return int(round(float(val)))
        except Exception:
            pass

    # General large-number with commas, or plain digits
    m_num = re.search(r"\d{1,3}(?:,\d{3})+|\d+", s)
    if m_num:
        try:
            return int(m_num.group(0).replace(",", ""))
        except Exception:
            pass

    return None


def _meets_min_one_percent(capacity_str: Optional[str], wheelchair_str: Optional[str]) -> Optional[bool]:
    """
    Check if wheelchair spaces >= ceil(1% of total capacity).
    Returns:
      True/False if both numbers parsed,
      None if cannot parse numbers.
    """
    cap = _parse_int_from_text(capacity_str)
    wc = _parse_int_from_text(wheelchair_str)
    if cap is None or wc is None or cap <= 0:
        return None
    required = max(1, math.ceil(cap * 0.01))
    return wc >= required


# --------------------------------------------------------------------------- #
# Verification for a single venue                                             #
# --------------------------------------------------------------------------- #
async def verify_single_venue(
    evaluator: Evaluator,
    parent_node,
    venue: VenueItem,
    idx: int,
    previous_venues: List[VenueItem],
) -> None:
    """
    Build and run verifications for one venue as a sequential, fully critical pipeline.
    """
    # Group node for this venue (critical, sequential)
    group = evaluator.add_sequential(
        id=f"venue_{idx}_group",
        desc=f"Venue #{idx + 1}: end-to-end verification",
        parent=parent_node,
        critical=True,
    )

    # 1) Identification & Location
    ident = evaluator.add_sequential(
        id=f"venue_{idx}_ident_loc",
        desc=f"Venue #{idx + 1}: identification, address completeness, and LA area location",
        parent=group,
        critical=True,
    )

    # 1.a) Existence of name/address + at least one venue source
    evaluator.add_custom_node(
        result=(bool(venue.name) and bool(venue.address) and len(venue.venue_sources) > 0),
        id=f"venue_{idx}_ident_provided",
        desc=f"Venue #{idx + 1}: name and complete street address provided with source URLs",
        parent=ident,
        critical=True,
    )

    # 1.b) Address looks complete (format/logical check, no URL evidence required)
    addr_complete_leaf = evaluator.add_leaf(
        id=f"venue_{idx}_address_complete",
        desc=f"Venue #{idx + 1}: provided address appears to be a complete US street address",
        parent=ident,
        critical=True,
    )
    addr_claim = (
        f"The provided address for venue #{idx + 1} is '{venue.address}'. "
        f"This appears to be a complete US street address including street number and name, city, state abbreviation, "
        f"and ideally a ZIP code."
    )
    await evaluator.verify(
        claim=addr_claim,
        node=addr_complete_leaf,
        additional_instruction="Judge completeness logically based on formatting and components; minor omissions (e.g., missing ZIP) may still count as complete if the rest is clearly a full street address."
    )

    # 1.c) Located in Los Angeles area (LA city or any LA County city), supported by URLs
    loc_leaf = evaluator.add_leaf(
        id=f"venue_{idx}_location_la_area",
        desc=f"Venue #{idx + 1}: location is within the Los Angeles area at the stated address",
        parent=ident,
        critical=True,
    )
    loc_claim = (
        f"The venue '{venue.name}' is located in the Los Angeles area (City of Los Angeles or elsewhere within "
        f"Los Angeles County, California) at the provided address: '{venue.address}'."
    )
    await evaluator.verify(
        claim=loc_claim,
        node=loc_leaf,
        sources=venue.venue_sources,
        additional_instruction="Accept any city within Los Angeles County (e.g., Los Angeles, Inglewood, Pasadena, Glendale, Burbank, etc.) as 'Los Angeles area'."
    )

    # 1.d) Distinctness from previously listed venues (for venue #2 and #3)
    if previous_venues:
        prev_summaries = "; ".join(
            [f"'{pv.name}' at '{pv.address}'" for pv in previous_venues if pv.name or pv.address]
        )
        distinct_leaf = evaluator.add_leaf(
            id=f"venue_{idx}_distinct_from_previous",
            desc=f"Venue #{idx + 1}: distinct location from previously listed venue(s)",
            parent=ident,
            critical=True,
        )
        distinct_claim = (
            f"The venue '{venue.name}' at '{venue.address}' is a different, distinct location from the previously "
            f"listed venue(s): {prev_summaries}."
        )
        await evaluator.verify(
            claim=distinct_claim,
            node=distinct_leaf,
            additional_instruction="Judge by names and addresses; allow that names can be similar, but addresses must be different physical locations."
        )

    # 2) Event scheduling between March–May 2026
    ev = evaluator.add_sequential(
        id=f"venue_{idx}_events",
        desc=f"Venue #{idx + 1}: has at least one ticketed entertainment event scheduled between March and May 2026",
        parent=group,
        critical=True,
    )

    evaluator.add_custom_node(
        result=(len(venue.event_urls) > 0),
        id=f"venue_{idx}_event_urls_provided",
        desc=f"Venue #{idx + 1}: event URL(s) provided in the answer",
        parent=ev,
        critical=True,
    )

    ev_leaf = evaluator.add_leaf(
        id=f"venue_{idx}_event_in_window",
        desc=f"Venue #{idx + 1}: at least one ticketed entertainment event scheduled in Mar–May 2026",
        parent=ev,
        critical=True,
    )
    ev_claim = (
        f"At least one of the provided pages shows a ticketed entertainment event (concert, theater, or sporting event) "
        f"scheduled to occur at '{venue.name}' between {START_DATE} and {END_DATE} (inclusive)."
    )
    await evaluator.verify(
        claim=ev_claim,
        node=ev_leaf,
        sources=venue.event_urls,
        additional_instruction="Confirm the page is for this venue, shows a specific event date within the date window, and indicates tickets are for sale (e.g., 'Buy Tickets', ticketing links). Allow official venue event pages, ticketing platforms (Ticketmaster, AXS, SeatGeek), or promoter pages."
    )

    # 3) Capacity verification
    cap = evaluator.add_sequential(
        id=f"venue_{idx}_capacity",
        desc=f"Venue #{idx + 1}: total seating capacity provided and supported",
        parent=group,
        critical=True,
    )

    evaluator.add_custom_node(
        result=(bool(venue.capacity) and len(venue.capacity_sources) > 0),
        id=f"venue_{idx}_capacity_provided",
        desc=f"Venue #{idx + 1}: seating capacity value and source URL(s) provided",
        parent=cap,
        critical=True,
    )

    cap_leaf = evaluator.add_leaf(
        id=f"venue_{idx}_capacity_supported",
        desc=f"Venue #{idx + 1}: capacity value supported by cited sources",
        parent=cap,
        critical=True,
    )
    cap_claim = f"The total seating capacity of '{venue.name}' is '{venue.capacity}'."
    await evaluator.verify(
        claim=cap_claim,
        node=cap_leaf,
        sources=venue.capacity_sources,
        additional_instruction="Look for explicit statements of venue capacity; allow close/rounded numbers as equivalent when clearly intended."
    )

    # 4) Wheelchair accessibility (ADA) verification
    ada = evaluator.add_sequential(
        id=f"venue_{idx}_ada",
        desc=f"Venue #{idx + 1}: ADA wheelchair-accessible seating verified, including number of wheelchair spaces",
        parent=group,
        critical=True,
    )

    evaluator.add_custom_node(
        result=(bool(venue.wheelchair_spaces) and len(venue.accessibility_sources) > 0),
        id=f"venue_{idx}_ada_provided",
        desc=f"Venue #{idx + 1}: wheelchair spaces count and accessibility source URL(s) provided",
        parent=ada,
        critical=True,
    )

    # 4.a) Specific number of wheelchair spaces supported
    ada_spaces_leaf = evaluator.add_leaf(
        id=f"venue_{idx}_wheelchair_spaces_supported",
        desc=f"Venue #{idx + 1}: specific number of wheelchair spaces supported by sources",
        parent=ada,
        critical=True,
    )
    ada_spaces_claim = (
        f"The venue '{venue.name}' provides ADA-compliant wheelchair-accessible seating and has "
        f"'{venue.wheelchair_spaces}' wheelchair spaces available for wheelchair users (companion seats aside)."
    )
    await evaluator.verify(
        claim=ada_spaces_claim,
        node=ada_spaces_leaf,
        sources=venue.accessibility_sources,
        additional_instruction="Verify that the source explicitly references wheelchair-accessible seating/areas and the quantity of wheelchair spaces (or an equivalent phrase like 'wheelchair positions'). Allow 'at least N' style statements to match the extracted text."
    )

    # 4.b) ADA compliance mention (qualitative)
    ada_compliance_leaf = evaluator.add_leaf(
        id=f"venue_{idx}_ada_compliance_supported",
        desc=f"Venue #{idx + 1}: ADA-compliant accessible seating is explicitly indicated",
        parent=ada,
        critical=True,
    )
    ada_compliance_claim = (
        f"The cited sources indicate that '{venue.name}' offers ADA-compliant wheelchair-accessible seating "
        f"(meeting ADA requirements such as appropriate dimensions and lines-of-sight)."
    )
    await evaluator.verify(
        claim=ada_compliance_claim,
        node=ada_compliance_leaf,
        sources=venue.accessibility_sources,
        additional_instruction="Treat explicit mentions of ADA-compliant accessible seating, accessibility sections, or official statements about ADA seating as sufficient. If only generic accessibility without seating is stated, do not count."
    )

    # 4.c) Minimum 1% rule (computed logical check from extracted numbers)
    meets_min = _meets_min_one_percent(venue.capacity, venue.wheelchair_spaces)
    evaluator.add_custom_node(
        result=(meets_min is True),
        id=f"venue_{idx}_ada_min_one_percent",
        desc=f"Venue #{idx + 1}: wheelchair spaces meet or exceed 1% of total capacity",
        parent=ada,
        critical=True,
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
    Evaluate an answer for the LA venues ADA task (Mar–May 2026).
    """
    # Initialize evaluator
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Top-level: evaluate 3 venues independently
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

    # Extract venue information from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_venues(),
        template_class=VenuesExtraction,
        extraction_name="venues_extraction",
    )

    # Normalize to exactly 3 venues (pad with empty items if fewer; truncate if more)
    venues: List[VenueItem] = list(extracted.venues or [])
    if len(venues) < 3:
        venues.extend([VenueItem() for _ in range(3 - len(venues))])
    else:
        venues = venues[:3]

    # Add GT/context info for transparency
    evaluator.add_ground_truth({
        "required_venues": 3,
        "time_window_inclusive": {"start": START_DATE, "end": END_DATE},
        "location_scope": "Los Angeles area (City of Los Angeles or anywhere in Los Angeles County)",
        "ada_min_ratio_rule": "wheelchair spaces >= ceil(1% of total capacity)",
    })

    # Build three critical venue groups under root
    venue_groups = []
    for i in range(3):
        venue_group_parent = evaluator.add_sequential(
            id=f"venue_{i}",
            desc=f"Venue #{i + 1} verification (critical)",
            parent=root,
            critical=True,
        )
        venue_groups.append(venue_group_parent)

    # Verify each venue with dependencies/sequence inside each group
    for i in range(3):
        prev = venues[:i]
        await verify_single_venue(
            evaluator=evaluator,
            parent_node=venue_groups[i],
            venue=venues[i],
            idx=i,
            previous_venues=prev,
        )

    # Return evaluation summary
    return evaluator.get_summary()