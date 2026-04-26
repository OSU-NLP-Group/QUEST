import asyncio
import logging
import re
from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys


# --------------------------------------------------------------------------- #
# Task constants                                                              #
# --------------------------------------------------------------------------- #
TASK_ID = "touring_broadway_spring_2026_se_midatl"
TASK_DESCRIPTION = (
    "Identify two touring Broadway productions that are scheduled to perform in the Southeastern or Mid-Atlantic "
    "United States during Spring 2026 (March 1 - May 31, 2026). For each production, provide: "
    "1) show name; 2) venue name and complete physical address (street, city, state); "
    "3) venue seating capacity (>= 2,000 seats); 4) engagement date range at that venue (>= one week); "
    "5) a URL that verifies the show's tour schedule and venue information. "
    "The two productions must be different Broadway shows at different venues."
)

SPRING_START = datetime(2026, 3, 1)
SPRING_END = datetime(2026, 5, 31)

# --------------------------------------------------------------------------- #
# Region mapping and helpers                                                  #
# --------------------------------------------------------------------------- #
US_STATE_TO_ABBR = {
    "alabama": "AL",
    "alaska": "AK",
    "arizona": "AZ",
    "arkansas": "AR",
    "california": "CA",
    "colorado": "CO",
    "connecticut": "CT",
    "delaware": "DE",
    "district of columbia": "DC",
    "washington dc": "DC",
    "dc": "DC",
    "florida": "FL",
    "georgia": "GA",
    "hawaii": "HI",
    "idaho": "ID",
    "illinois": "IL",
    "indiana": "IN",
    "iowa": "IA",
    "kansas": "KS",
    "kentucky": "KY",
    "louisiana": "LA",
    "maine": "ME",
    "maryland": "MD",
    "massachusetts": "MA",
    "michigan": "MI",
    "minnesota": "MN",
    "mississippi": "MS",
    "missouri": "MO",
    "montana": "MT",
    "nebraska": "NE",
    "nevada": "NV",
    "new hampshire": "NH",
    "new jersey": "NJ",
    "new mexico": "NM",
    "new york": "NY",
    "north carolina": "NC",
    "north dakota": "ND",
    "ohio": "OH",
    "oklahoma": "OK",
    "oregon": "OR",
    "pennsylvania": "PA",
    "rhode island": "RI",
    "south carolina": "SC",
    "south dakota": "SD",
    "tennessee": "TN",
    "texas": "TX",
    "utah": "UT",
    "vermont": "VT",
    "virginia": "VA",
    "washington": "WA",
    "west virginia": "WV",
    "wisconsin": "WI",
    "wyoming": "WY",
}

ABBR_SET = set(US_STATE_TO_ABBR.values())

# Define accepted regions
SE_STATES = {
    "AL", "AR", "FL", "GA", "KY", "LA", "MS", "NC", "SC", "TN", "VA", "WV"
}
MIDATL_STATES = {
    "DC", "DE", "MD", "NJ", "NY", "PA", "VA", "WV"
}
ACCEPTED_STATES = SE_STATES | MIDATL_STATES

def normalize_state_to_abbr(s: Optional[str]) -> Optional[str]:
    if not s:
        return None
    s_clean = re.sub(r"[^\w\s]", "", s).strip().lower()
    if len(s_clean) == 2 and s_clean.upper() in ABBR_SET:
        return s_clean.upper()
    return US_STATE_TO_ABBR.get(s_clean)

def is_se_or_midatlantic_state(state_str: Optional[str]) -> bool:
    abbr = normalize_state_to_abbr(state_str)
    return bool(abbr and abbr in ACCEPTED_STATES)

def normalize_text_for_compare(s: Optional[str]) -> str:
    if not s:
        return ""
    return re.sub(r"[^a-z0-9]", "", s.lower())

def filter_valid_urls(urls: Optional[List[str]]) -> List[str]:
    if not urls:
        return []
    out = []
    for u in urls:
        if not u:
            continue
        u = u.strip()
        if re.match(r"^https?://", u, flags=re.I):
            out.append(u)
    # Deduplicate
    seen = set()
    uniq = []
    for u in out:
        if u not in seen:
            uniq.append(u)
            seen.add(u)
    return uniq


# --------------------------------------------------------------------------- #
# Extraction models                                                           #
# --------------------------------------------------------------------------- #
class ShowItem(BaseModel):
    show_name: Optional[str] = None
    venue_name: Optional[str] = None
    street_address: Optional[str] = None
    city: Optional[str] = None
    state: Optional[str] = None
    seating_capacity: Optional[str] = None
    date_start: Optional[str] = None
    date_end: Optional[str] = None
    source_urls: List[str] = Field(default_factory=list)


class ShowsExtraction(BaseModel):
    shows: List[ShowItem] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_shows() -> str:
    return """
    Extract up to the first 4 touring Broadway productions mentioned in the answer, each with:
    - show_name: the name of the touring Broadway show
    - venue_name: the performance venue name
    - street_address: the street address of the venue (e.g., "123 Main St")
    - city: the city of the venue
    - state: the state of the venue (accept full name like "Virginia" or abbreviation "VA")
    - seating_capacity: the stated seating capacity for the venue (as written, including commas or text)
    - date_start: the start date of the engagement at that venue (as written in the answer)
    - date_end: the end date of the engagement at that venue (as written in the answer)
    - source_urls: all URLs provided in the answer that support this show's schedule and venue info (include any venue page, tour page, ticketing page, Broadway Across America page, press release, etc.). Only include URLs explicitly present in the answer text. Return as a list; include both http and https links.
    Important:
    - Do not invent information not present in the answer.
    - For any missing field, return null (or empty list for source_urls).
    - If multiple URLs are provided, include them all in source_urls.
    - The address fields must be split into street_address, city, and state separately if possible.
    Return a JSON with a single field "shows" which is an array of these objects.
    """


# --------------------------------------------------------------------------- #
# Verification helpers                                                        #
# --------------------------------------------------------------------------- #
def build_date_range_claim(item: ShowItem) -> str:
    s = item.date_start or ""
    e = item.date_end or ""
    return (
        f"For the venue engagement of '{item.show_name}' at '{item.venue_name}', the run is from '{s}' to '{e}', "
        f"and this entire date range falls between March 1, 2026 and May 31, 2026 (inclusive). "
        f"The engagement lasts at least 7 days."
    )

def build_show_name_claim(item: ShowItem) -> str:
    title = item.show_name or ""
    return f"The show title on the referenced page(s) is '{title}' (allowing minor variations like possessives or subtitles)."

def build_venue_identification_claim(item: ShowItem) -> str:
    venue = item.venue_name or ""
    addr = f"{item.street_address or ''}, {item.city or ''}, {item.state or ''}".strip(", ").replace(" ,", ",")
    return (
        f"The referenced page(s) indicate that '{item.show_name}' is scheduled at the venue '{venue}', "
        f"whose physical address is '{addr}'. The address must include street address, city, and state."
    )

def build_show_type_claim(item: ShowItem) -> str:
    return (
        f"The referenced page(s) confirm that '{item.show_name}' is a touring Broadway production (i.e., part of a national "
        f"tour or equivalent), not a local/regional resident production."
    )

def build_capacity_claim(item: ShowItem) -> str:
    venue = item.venue_name or "the venue"
    return (
        f"The referenced page(s) indicate that {venue} has a seating capacity of at least 2,000 seats "
        f"(accept reasonable formulations like 'capacity: 2,100', 'over 2,000 seats', or named auditorium capacity)."
    )

def additional_instruction_show_name() -> str:
    return (
        "Verify that the page is about the same show and the title matches, allowing reasonable variants such as capitalization, "
        "presence/absence of articles, possessives, and franchise qualifiers like 'Disney's The Lion King'."
    )

def additional_instruction_venue_identification() -> str:
    return (
        "Confirm both: (1) the specific venue name for this show engagement; and (2) the full physical address including "
        "street, city, and state. The address may appear on the venue's contact/about pages linked from the main page."
    )

def additional_instruction_show_type() -> str:
    return (
        "Look for phrases like 'national tour', 'Broadway tour', 'North American tour', 'on tour', "
        "'Broadway in [City]' subscription, or official tour pages listing multiple cities/venues."
    )

def additional_instruction_capacity() -> str:
    return (
        "The claim passes if the venue capacity stated or clearly implied on the referenced page(s) is >= 2,000. "
        "Venues with multiple configurations are acceptable if any standard/theatrical configuration for the listed auditorium is >= 2,000."
    )

def additional_instruction_dates() -> str:
    return (
        "Validate that the engagement dates at this venue start no earlier than March 1, 2026 and end no later than May 31, 2026, "
        "and that the span covers at least 7 consecutive days (inclusive). If only month/day are shown, assume the year is 2026 "
        "if implied by the page; otherwise treat as not supported."
    )


# --------------------------------------------------------------------------- #
# Verification for a single show                                              #
# --------------------------------------------------------------------------- #
async def verify_show(
    evaluator: Evaluator,
    parent_node,
    item: ShowItem,
    idx: int,
) -> None:
    """
    Build and run verification for one show (parallel aggregation).
    """
    show_node = evaluator.add_parallel(
        id=f"show_{idx+1}",
        desc=f"{'First' if idx == 0 else 'Second'} touring Broadway production meeting all requirements",
        parent=parent_node,
        critical=False,
    )

    # Clean and normalize URLs once
    valid_urls = filter_valid_urls(item.source_urls)

    # 1) URL reference provided (critical; existence/valid form)
    url_ref_node = evaluator.add_custom_node(
        result=len(valid_urls) > 0,
        id=f"show_{idx+1}_url_reference",
        desc="A valid URL reference supporting the show's tour schedule and venue information is provided",
        parent=show_node,
        critical=True
    )

    # 2) Show type is touring Broadway (critical; evidence-grounded)
    show_type_node = evaluator.add_leaf(
        id=f"show_{idx+1}_show_type",
        desc="The production is verified as a touring Broadway production (not a local or regional theater production)",
        parent=show_node,
        critical=True
    )

    # 3) Show name correct (critical; evidence-grounded)
    show_name_node = evaluator.add_leaf(
        id=f"show_{idx+1}_show_name",
        desc="The name of the touring Broadway production is correctly identified",
        parent=show_node,
        critical=True
    )

    # 4) Venue identification: name + full address (critical; evidence-grounded)
    venue_id_node = evaluator.add_leaf(
        id=f"show_{idx+1}_venue_identification",
        desc="The venue name and complete physical address (including street address, city, and state) are correctly provided",
        parent=show_node,
        critical=True
    )

    # 5) Venue location in SE or Mid-Atlantic (critical; deterministic region check)
    venue_loc_ok = is_se_or_midatlantic_state(item.state)
    venue_loc_node = evaluator.add_custom_node(
        result=venue_loc_ok,
        id=f"show_{idx+1}_venue_location",
        desc="The venue is located in the Southeastern or Mid-Atlantic United States",
        parent=show_node,
        critical=True
    )

    # 6) Venue capacity >= 2,000 (critical; evidence-grounded)
    capacity_node = evaluator.add_leaf(
        id=f"show_{idx+1}_venue_capacity",
        desc="The venue has a seating capacity of at least 2,000 seats",
        parent=show_node,
        critical=True
    )

    # 7) Performance dates within Spring 2026 and at least one week (critical; evidence-grounded)
    perf_dates_node = evaluator.add_leaf(
        id=f"show_{idx+1}_performance_dates",
        desc="The performance date range (start and end dates) falls within Spring 2026 (March 1 - May 31, 2026) and represents at least one week of performances",
        parent=show_node,
        critical=True
    )

    # Batch verify evidence-grounded leaves (url_ref_node already evaluated; venue_loc_node is deterministic)
    claims_and_sources = [
        (
            build_show_type_claim(item),
            valid_urls,
            show_type_node,
            additional_instruction_show_type()
        ),
        (
            build_show_name_claim(item),
            valid_urls,
            show_name_node,
            additional_instruction_show_name()
        ),
        (
            build_venue_identification_claim(item),
            valid_urls,
            venue_id_node,
            additional_instruction_venue_identification()
        ),
        (
            build_capacity_claim(item),
            valid_urls,
            capacity_node,
            additional_instruction_capacity()
        ),
        (
            build_date_range_claim(item),
            valid_urls,
            perf_dates_node,
            additional_instruction_dates()
        ),
    ]
    await evaluator.batch_verify(claims_and_sources)


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
    Evaluate an answer for the touring Broadway Spring 2026 task.
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
        default_model=model,
    )

    # Extract structured show info
    extracted = await evaluator.extract(
        prompt=prompt_extract_shows(),
        template_class=ShowsExtraction,
        extraction_name="shows_extraction",
    )

    # Select the first two shows (pad if fewer)
    shows: List[ShowItem] = list(extracted.shows[:2])
    while len(shows) < 2:
        shows.append(ShowItem())

    # Build verification subtrees for each show
    await verify_show(evaluator, root, shows[0], idx=0)
    await verify_show(evaluator, root, shows[1], idx=1)

    # Uniqueness checks (critical)
    # - Different shows
    name1 = normalize_text_for_compare(shows[0].show_name)
    name2 = normalize_text_for_compare(shows[1].show_name)
    diff_shows = bool(name1 and name2 and name1 != name2)
    evaluator.add_custom_node(
        result=diff_shows,
        id="uniqueness_shows",
        desc="The two productions are different Broadway shows (not the same show listed twice)",
        parent=root,
        critical=True
    )

    # - Different venues (use venue name + city + state)
    venue_key_1 = normalize_text_for_compare(
        f"{shows[0].venue_name or ''} {shows[0].city or ''} {normalize_state_to_abbr(shows[0].state) or ''}"
    )
    venue_key_2 = normalize_text_for_compare(
        f"{shows[1].venue_name or ''} {shows[1].city or ''} {normalize_state_to_abbr(shows[1].state) or ''}"
    )
    diff_venues = bool(venue_key_1 and venue_key_2 and venue_key_1 != venue_key_2)
    evaluator.add_custom_node(
        result=diff_venues,
        id="uniqueness_venues",
        desc="The two productions are performing at different venues (not the same venue listed twice)",
        parent=root,
        critical=True
    )

    return evaluator.get_summary()