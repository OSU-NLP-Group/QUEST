import asyncio
import logging
import re
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy

# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "us_indoor_concert_venues_3"
TASK_DESCRIPTION = (
    "Identify three major indoor concert venues in the United States that meet ALL of the following criteria:\n\n"
    "1. Concert Capacity: Each venue must have a concert seating capacity between 19,000 and 24,000 (inclusive).\n"
    "2. Geographic Distribution: The three venues must be located in three different U.S. states.\n"
    "3. Opening or Renovation Timeline: Each venue must have either opened OR undergone a major renovation between "
    "January 1, 1990, and December 31, 2020 (inclusive). For venues with renovations, the renovation must have been "
    "substantial (involving significant upgrades to seating, suites, or major facility systems).\n"
    "4. Luxury Suites: Each venue must have at least 50 luxury suites or executive suites.\n"
    "5. Facility Standards: Each venue should meet standard accessibility requirements for major concert venues "
    "(including ADA-compliant wheelchair seating and adequate restroom facilities).\n\n"
    "For each venue, provide:\n"
    "- The venue name\n"
    "- The specific city and state location\n"
    "- The exact concert seating capacity\n"
    "- The exact number of luxury suites\n"
    "- The specific opening date OR major renovation completion date (with the year clearly identified)\n"
    "- Valid reference URL(s) that confirm these specifications"
)

# --------------------------------------------------------------------------- #
# US states normalization helpers                                             #
# --------------------------------------------------------------------------- #
STATE_NAME_TO_CODE = {
    "alabama": "AL", "alaska": "AK", "arizona": "AZ", "arkansas": "AR", "california": "CA",
    "colorado": "CO", "connecticut": "CT", "delaware": "DE", "florida": "FL", "georgia": "GA",
    "hawaii": "HI", "idaho": "ID", "illinois": "IL", "indiana": "IN", "iowa": "IA",
    "kansas": "KS", "kentucky": "KY", "louisiana": "LA", "maine": "ME", "maryland": "MD",
    "massachusetts": "MA", "michigan": "MI", "minnesota": "MN", "mississippi": "MS", "missouri": "MO",
    "montana": "MT", "nebraska": "NE", "nevada": "NV", "new hampshire": "NH", "new jersey": "NJ",
    "new mexico": "NM", "new york": "NY", "north carolina": "NC", "north dakota": "ND", "ohio": "OH",
    "oklahoma": "OK", "oregon": "OR", "pennsylvania": "PA", "rhode island": "RI", "south carolina": "SC",
    "south dakota": "SD", "tennessee": "TN", "texas": "TX", "utah": "UT", "vermont": "VT",
    "virginia": "VA", "washington": "WA", "west virginia": "WV", "wisconsin": "WI", "wyoming": "WY",
    "district of columbia": "DC", "washington dc": "DC", "washington, dc": "DC", "dc": "DC"
}
STATE_CODES = {code: code for code in STATE_NAME_TO_CODE.values()}


def normalize_state_code(state_str: Optional[str]) -> Optional[str]:
    if not state_str:
        return None
    s = state_str.strip()
    if not s:
        return None
    up = s.upper()
    if up in STATE_CODES:
        return up
    low = s.lower()
    if low in STATE_NAME_TO_CODE:
        return STATE_NAME_TO_CODE[low]
    # Try stripping periods and spaces (e.g., "D.C.")
    compact = re.sub(r"[.\s]", "", low)
    if compact in STATE_NAME_TO_CODE:
        return STATE_NAME_TO_CODE[compact]
    return None


# --------------------------------------------------------------------------- #
# Parsing helpers                                                             #
# --------------------------------------------------------------------------- #
def parse_int_from_text(text: Optional[str]) -> Optional[int]:
    if not text:
        return None
    match = re.search(r"(\d[\d,]*)", text)
    if not match:
        return None
    digits = match.group(1).replace(",", "")
    try:
        return int(digits)
    except Exception:
        return None


def parse_year_from_text(text: Optional[str]) -> Optional[int]:
    if not text:
        return None
    m = re.search(r"(19|20)\d{2}", text)
    if not m:
        return None
    try:
        return int(m.group(0))
    except Exception:
        return None


def year_in_range(year: Optional[int], start: int = 1990, end: int = 2020) -> bool:
    if year is None:
        return False
    return start <= year <= end


def build_timeline_claim(name: str, event_type: Optional[str], event_date: Optional[str]) -> str:
    et = (event_type or "").strip().lower()
    date_text = (event_date or "").strip()
    yr = parse_year_from_text(date_text)
    # If we only have a year, use "in YEAR"; if more detail appears present, use "on <date string>"
    if yr and len(date_text) <= 4:
        date_clause = f"in {yr}"
    elif date_text:
        date_clause = f"on {date_text}"
    else:
        date_clause = "on the provided date"
    if et == "renovation":
        return f"{name} underwent a major renovation completed {date_clause}."
    else:
        # Default to opening if unknown
        return f"{name} opened {date_clause}."


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class VenueItem(BaseModel):
    name: Optional[str] = None
    city: Optional[str] = None
    state: Optional[str] = None
    concert_capacity: Optional[str] = None
    luxury_suites: Optional[str] = None
    event_type: Optional[str] = None  # "opening" or "renovation"
    event_date: Optional[str] = None  # Date string including a year
    sources: List[str] = Field(default_factory=list)


class VenuesExtraction(BaseModel):
    venues: List[VenueItem] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_venues() -> str:
    return """
    Extract up to the first three distinct venues listed in the answer that the author intends to present as satisfying the requirements. For each venue, extract:
    - name: venue name exactly as written
    - city: the city (not including the state)
    - state: the U.S. state (either full name like "California" or two-letter code like "CA")
    - concert_capacity: the exact concert seating capacity number as stated (use digits; include commas if present as in the answer)
    - luxury_suites: the exact number of luxury or executive suites as stated (use digits; include commas if present as in the answer)
    - event_type: either "opening" or "renovation" depending on what the answer uses for the timeline item
    - event_date: the date string for the opening or the renovation completion (must include a year)
    - sources: all URLs cited in the answer that are used as references for this venue. Collect every URL mentioned for the venue. Include valid complete URLs only.
    
    Rules:
    - Only extract information that explicitly appears in the answer.
    - If the answer contains more than three venues, include only the first three.
    - If the answer contains fewer than three venues, extract as many as present.
    - If any field is missing for a venue, set it to null (or an empty list for sources).
    - For sources, include only URLs explicitly present in the answer text (plain URLs or markdown links). Do not invent or infer any URL.
    """


# --------------------------------------------------------------------------- #
# Verification helpers                                                        #
# --------------------------------------------------------------------------- #
async def verify_venue(
    evaluator: Evaluator,
    parent_node,
    v: VenueItem,
    idx: int
) -> None:
    venue_node = evaluator.add_parallel(
        id=f"venue_{idx}",
        desc=f"Venue {idx} details and constraints compliance",
        parent=parent_node,
        critical=False
    )

    # vX_name
    evaluator.add_custom_node(
        result=bool(v.name and v.name.strip()),
        id=f"v{idx}_name",
        desc=f"Venue name is provided",
        parent=venue_node,
        critical=True
    )

    # vX_location group
    loc_group = evaluator.add_parallel(
        id=f"v{idx}_location",
        desc=f"Venue location is provided and is in the United States",
        parent=venue_node,
        critical=True
    )

    evaluator.add_custom_node(
        result=bool(v.city and v.city.strip()) and bool(v.state and v.state.strip()),
        id=f"v{idx}_city_state",
        desc=f"Specific city and state are provided",
        parent=loc_group,
        critical=True
    )

    state_code = normalize_state_code(v.state)
    evaluator.add_custom_node(
        result=state_code is not None,
        id=f"v{idx}_us",
        desc=f"Venue is located in the United States",
        parent=loc_group,
        critical=True
    )

    # Location supported by references
    loc_ref_node = evaluator.add_leaf(
        id=f"v{idx}_location_reference",
        desc=f"Valid reference URL(s) support the stated location",
        parent=loc_group,
        critical=True
    )
    loc_claim = f"The venue {v.name or 'the venue'} is located in {v.city or 'the stated city'}, {v.state or 'the stated state'}."
    await evaluator.verify(
        claim=loc_claim,
        node=loc_ref_node,
        sources=v.sources,
        additional_instruction="Check the webpage(s) to confirm the venue's city and U.S. state."
    )

    # vX_indoor group
    indoor_group = evaluator.add_parallel(
        id=f"v{idx}_indoor",
        desc=f"Venue is an indoor concert venue (e.g., described as an indoor arena/indoor venue)",
        parent=venue_node,
        critical=True
    )

    indoor_claim_node = evaluator.add_leaf(
        id=f"v{idx}_indoor_claim",
        desc=f"Answer states or clearly implies the venue is indoor",
        parent=indoor_group,
        critical=True
    )
    claim_indoor_from_answer = f"The answer states or clearly implies that {v.name or 'the venue'} is an indoor arena or indoor venue."
    await evaluator.verify(
        claim=claim_indoor_from_answer,
        node=indoor_claim_node,
        additional_instruction="Judge based only on the provided answer text whether it indicates the venue is indoor."
    )

    indoor_ref_node = evaluator.add_leaf(
        id=f"v{idx}_indoor_reference",
        desc=f"Valid reference URL(s) support that the venue is indoor",
        parent=indoor_group,
        critical=True
    )
    claim_indoor_reference = f"{v.name or 'The venue'} is an indoor arena or indoor concert venue."
    await evaluator.verify(
        claim=claim_indoor_reference,
        node=indoor_ref_node,
        sources=v.sources,
        additional_instruction="Look for terms like 'indoor arena', 'indoor venue', 'domed', etc."
    )

    # vX_capacity group
    cap_group = evaluator.add_parallel(
        id=f"v{idx}_capacity",
        desc=f"Concert seating capacity is provided, in-range, and referenced",
        parent=venue_node,
        critical=True
    )
    cap_val = parse_int_from_text(v.concert_capacity)
    evaluator.add_custom_node(
        result=cap_val is not None,
        id=f"v{idx}_capacity_value",
        desc=f"Exact concert seating capacity number is provided",
        parent=cap_group,
        critical=True
    )
    evaluator.add_custom_node(
        result=(cap_val is not None and 19000 <= cap_val <= 24000),
        id=f"v{idx}_capacity_range",
        desc=f"Concert seating capacity is between 19,000 and 24,000 inclusive",
        parent=cap_group,
        critical=True
    )
    cap_ref_node = evaluator.add_leaf(
        id=f"v{idx}_capacity_reference",
        desc=f"Valid reference URL(s) support the stated concert capacity",
        parent=cap_group,
        critical=True
    )
    cap_claim = f"The concert configuration seating capacity of {v.name or 'the venue'} is {cap_val if cap_val is not None else (v.concert_capacity or 'the stated number')}."
    await evaluator.verify(
        claim=cap_claim,
        node=cap_ref_node,
        sources=v.sources,
        additional_instruction=(
            "Confirm the 'concert' seating capacity (or equivalent phrasing like 'for concerts', "
            "'end-stage configuration', or 'maximum seating for concerts'). Allow minor rounding differences."
        )
    )

    # vX_suites group
    suites_group = evaluator.add_parallel(
        id=f"v{idx}_suites",
        desc=f"Luxury/executive suite count is provided, meets minimum, and referenced",
        parent=venue_node,
        critical=True
    )
    suites_val = parse_int_from_text(v.luxury_suites)
    evaluator.add_custom_node(
        result=suites_val is not None,
        id=f"v{idx}_suite_count",
        desc=f"Exact number of luxury suites or executive suites is provided",
        parent=suites_group,
        critical=True
    )
    evaluator.add_custom_node(
        result=(suites_val is not None and suites_val >= 50),
        id=f"v{idx}_suite_minimum",
        desc=f"Venue has at least 50 luxury/executive suites",
        parent=suites_group,
        critical=True
    )
    suite_ref_node = evaluator.add_leaf(
        id=f"v{idx}_suite_reference",
        desc=f"Valid reference URL(s) support the stated suite count",
        parent=suites_group,
        critical=True
    )
    suites_claim = f"{v.name or 'The venue'} has {suites_val if suites_val is not None else (v.luxury_suites or 'the stated number')} luxury suites (also referred to as suites or executive suites)."
    await evaluator.verify(
        claim=suites_claim,
        node=suite_ref_node,
        sources=v.sources,
        additional_instruction="Confirm the count of luxury suites/executive suites; treat 'suites' and 'luxury suites' as equivalent if the context is luxury/executive suites."
    )

    # vX_timeline group
    timeline_group = evaluator.add_parallel(
        id=f"v{idx}_timeline",
        desc=f"Opening OR major renovation timeline is provided, in-range, and referenced; renovation must be substantial if used",
        parent=venue_node,
        critical=True
    )
    event_type_norm = (v.event_type or "").strip().lower()
    evt_year = parse_year_from_text(v.event_date)

    evaluator.add_custom_node(
        result=(event_type_norm in ("opening", "renovation") and bool(v.event_date and evt_year is not None)),
        id=f"v{idx}_event_type_and_date",
        desc=f"Answer specifies whether the provided date is an opening date or a major renovation completion date, and provides the specific date (year clearly identified)",
        parent=timeline_group,
        critical=True
    )
    evaluator.add_custom_node(
        result=year_in_range(evt_year, 1990, 2020),
        id=f"v{idx}_date_range",
        desc=f"The opening/renovation date is between Jan 1, 1990 and Dec 31, 2020 inclusive",
        parent=timeline_group,
        critical=True
    )

    # Substantial renovation if applicable
    if event_type_norm == "renovation":
        subst_node = evaluator.add_leaf(
            id=f"v{idx}_substantial_renovation_if_applicable",
            desc=f"If a renovation (not opening) is cited, the answer indicates it was substantial (significant upgrades to seating, suites, or major facility systems)",
            parent=timeline_group,
            critical=True
        )
        subst_claim = (
            f"The renovation of {v.name or 'the venue'} was substantial, involving significant upgrades to seating, suites, or major facility systems."
        )
        await evaluator.verify(
            claim=subst_claim,
            node=subst_node,
            sources=v.sources,
            additional_instruction="Look for words/phrases such as 'major renovation', 'extensive upgrades', 'significant improvements', 'seating upgrades', 'suite upgrades', or major building systems upgrades."
        )
    else:
        evaluator.add_custom_node(
            result=True,
            id=f"v{idx}_substantial_renovation_if_applicable",
            desc=f"If a renovation (not opening) is cited, the answer indicates it was substantial (N/A since opening is used)",
            parent=timeline_group,
            critical=True
        )

    time_ref_node = evaluator.add_leaf(
        id=f"v{idx}_timeline_reference",
        desc=f"Valid reference URL(s) support the opening/renovation date and (if renovation) substantiveness",
        parent=timeline_group,
        critical=True
    )
    timeline_claim = build_timeline_claim(v.name or "the venue", event_type_norm, v.event_date)
    await evaluator.verify(
        claim=timeline_claim,
        node=time_ref_node,
        sources=v.sources,
        additional_instruction="Confirm both the date (year must match) and the event type (opening vs major renovation). Allow month/day variations so long as the year and event align."
    )

    # vX_accessibility_facilities (non-critical)
    access_group = evaluator.add_parallel(
        id=f"v{idx}_accessibility_facilities",
        desc=f"Accessibility and restroom facility standards (should-have items)",
        parent=venue_node,
        critical=False
    )
    ada_node = evaluator.add_leaf(
        id=f"v{idx}_ada_wheelchair",
        desc=f"Answer states the venue provides ADA-compliant wheelchair-accessible seating (incl. any stated minimum such as 1% of capacity, if claimed)",
        parent=access_group,
        critical=False
    )
    ada_claim = f"The answer states that {v.name or 'the venue'} provides ADA-compliant wheelchair-accessible seating."
    await evaluator.verify(
        claim=ada_claim,
        node=ada_node,
        additional_instruction="Judge based only on the answer text; do not require external sources for this should-have item."
    )

    rest_node = evaluator.add_leaf(
        id=f"v{idx}_restrooms",
        desc=f"Answer states the venue provides adequate restroom facilities meeting building codes for large assembly occupancies",
        parent=access_group,
        critical=False
    )
    rest_claim = f"The answer states that {v.name or 'the venue'} provides adequate restroom facilities for large assembly occupancies."
    await evaluator.verify(
        claim=rest_claim,
        node=rest_node,
        additional_instruction="Judge based only on the answer text; do not require external sources for this should-have item."
    )

    # vX_references_present
    evaluator.add_custom_node(
        result=bool(v.sources and len(v.sources) > 0),
        id=f"v{idx}_references_present",
        desc=f"At least one valid reference URL is provided for the venue",
        parent=venue_node,
        critical=True
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
    Evaluate an answer for the 'three major indoor concert venues' task.
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

    # Extract venues from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_venues(),
        template_class=VenuesExtraction,
        extraction_name="venues_extraction"
    )

    # Select exactly first 3 venues for evaluation (padding if fewer)
    venues = (extracted.venues or [])[:3]
    while len(venues) < 3:
        venues.append(VenueItem())

    # Global critical checks at root
    # "Exactly three venues are provided" -> follow evaluation guidance: at least 3 in the answer
    evaluator.add_custom_node(
        result=(len(extracted.venues) >= 3),
        id="venue_count",
        desc="Exactly three venues are provided",
        parent=root,
        critical=True
    )

    # Distinct states across the three venues
    state_codes = [normalize_state_code(v.state) for v in venues]
    distinct_states_ok = all(code is not None for code in state_codes) and len(set(state_codes)) == 3
    evaluator.add_custom_node(
        result=distinct_states_ok,
        id="distinct_states",
        desc="The three venues are located in three different U.S. states (all states are distinct)",
        parent=root,
        critical=True
    )

    # Per-venue verification
    for i, v in enumerate(venues, start=1):
        await verify_venue(evaluator, root, v, i)

    return evaluator.get_summary()