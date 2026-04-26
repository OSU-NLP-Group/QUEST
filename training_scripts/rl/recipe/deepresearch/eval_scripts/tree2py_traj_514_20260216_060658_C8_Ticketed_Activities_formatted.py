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
TASK_ID = "coachella_2026_info"
TASK_DESCRIPTION = (
    "I'm planning to attend the Coachella Valley Music and Arts Festival in 2026 and need comprehensive planning "
    "information. Please provide the following details: (1) The official event name, year, and location (city and state), "
    "(2) The complete schedule for Weekend 1, including start date, end date, and the headlining performers for Friday, "
    "Saturday, and Sunday, (3) The complete schedule for Weekend 2, including start date, end date, and the headlining "
    "performers for Friday, Saturday, and Sunday, (4) The venue name, its location, and its capacity for festival events, "
    "(5) The starting ticket prices for General Admission, GA+, and VIP passes. Please include reference URLs for all "
    "information provided."
)

# Ground-truth expectations that will be checked against cited sources
FESTIVAL_OFFICIAL_NAME = "Coachella Valley Music and Arts Festival"
FESTIVAL_SHORT_NAME = "Coachella"
HOST_CITY = "Indio"
HOST_STATE = "California"
YEAR_EXPECTED = "2026"

W1_START = "April 10, 2026"
W1_END = "April 12, 2026"
W2_START = "April 17, 2026"
W2_END = "April 19, 2026"

WEEKEND_HEADLINERS = {
    "friday": "Sabrina Carpenter",
    "saturday": "Justin Bieber",
    "sunday": "Karol G",
}

VENUE_NAME = "Empire Polo Club"
VENUE_CITY = "Indio"
VENUE_STATE = "California"
FESTIVAL_CAPACITY_APPROX = "90,000"

START_PRICE_GA = "339"
START_PRICE_GA_PLUS = "449"
START_PRICE_VIP = "899"

# --------------------------------------------------------------------------- #
# Data models                                                                 #
# --------------------------------------------------------------------------- #
class BasicEventDetails(BaseModel):
    event_name: Optional[str] = None
    year: Optional[str] = None
    city: Optional[str] = None
    state: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class WeekendSchedule(BaseModel):
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    friday_headliner: Optional[str] = None
    saturday_headliner: Optional[str] = None
    sunday_headliner: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class VenueInfo(BaseModel):
    venue_name: Optional[str] = None
    location_city: Optional[str] = None
    location_state: Optional[str] = None
    capacity: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class TicketPricing(BaseModel):
    ga: Optional[str] = None
    ga_plus: Optional[str] = None
    vip: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class Coachella2026Extraction(BaseModel):
    basic: Optional[BasicEventDetails] = None
    weekend1: Optional[WeekendSchedule] = None
    weekend2: Optional[WeekendSchedule] = None
    venue: Optional[VenueInfo] = None
    tickets: Optional[TicketPricing] = None
    all_sources: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_coachella_2026() -> str:
    return """
    Extract structured information about the Coachella Valley Music and Arts Festival 2026 from the provided answer.

    Return a single JSON object with the following structure and fields (use null for any missing field). 
    Include only information explicitly present in the answer text. For all 'sources' fields, extract the URLs the answer cites for that section.

    {
      "basic": {
        "event_name": string|null,          // The festival name as stated in the answer (e.g., "Coachella", "Coachella Valley Music and Arts Festival")
        "year": string|null,                 // The year as stated in the answer (e.g., "2026")
        "city": string|null,                 // The city name as stated (e.g., "Indio")
        "state": string|null,                // The state name as stated (e.g., "California")
        "sources": string[]                  // URLs cited for these basic details (from the answer)
      },
      "weekend1": {
        "start_date": string|null,           // Start date of Weekend 1 as stated (any format is fine, keep as-is)
        "end_date": string|null,             // End date of Weekend 1 as stated
        "friday_headliner": string|null,     // The headliner named for Friday of Weekend 1
        "saturday_headliner": string|null,   // The headliner named for Saturday of Weekend 1
        "sunday_headliner": string|null,     // The headliner named for Sunday of Weekend 1
        "sources": string[]                  // URLs cited for Weekend 1 information (from the answer)
      },
      "weekend2": {
        "start_date": string|null,           
        "end_date": string|null,
        "friday_headliner": string|null,
        "saturday_headliner": string|null,
        "sunday_headliner": string|null,
        "sources": string[]                  // URLs cited for Weekend 2 information (from the answer)
      },
      "venue": {
        "venue_name": string|null,           // The venue name as stated
        "location_city": string|null,        // City for venue location as stated
        "location_state": string|null,       // State for venue location as stated
        "capacity": string|null,             // Capacity as stated in the answer (keep formatting as-is, e.g., "90,000")
        "sources": string[]                  // URLs cited for venue information (from the answer)
      },
      "tickets": {
        "ga": string|null,                   // Stated starting price for General Admission (keep as-is, e.g., "$339", "339 USD", "around 339")
        "ga_plus": string|null,              // Stated starting price for GA+ (keep as-is)
        "vip": string|null,                  // Stated starting price for VIP (keep as-is)
        "sources": string[]                  // URLs cited for ticket/pricing information (from the answer)
      },
      "all_sources": string[]                // All URLs mentioned anywhere in the answer (deduplicated if possible)
    }

    Rules:
    - Do not invent or infer any information. Extract only what the answer states.
    - For any URLs missing a protocol, prepend "http://".
    - If multiple URLs are provided, include all of them in the corresponding 'sources' array; do not summarize.
    - Dates can be any format as presented (e.g., "April 10, 2026", "Apr 10, 2026"); do not normalize.
    - If the answer gives a range for prices or uses "starting at", keep the text as-is in the extracted string.
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _unique_urls(urls: List[str]) -> List[str]:
    seen = set()
    uniq: List[str] = []
    for u in urls or []:
        if not u:
            continue
        u = u.strip()
        if not u:
            continue
        if u not in seen:
            seen.add(u)
            uniq.append(u)
    return uniq


def _collect_global_sources(extracted: Coachella2026Extraction) -> List[str]:
    all_urls: List[str] = []
    if extracted and extracted.all_sources:
        all_urls.extend(extracted.all_sources)
    if extracted and extracted.basic and extracted.basic.sources:
        all_urls.extend(extracted.basic.sources)
    if extracted and extracted.weekend1 and extracted.weekend1.sources:
        all_urls.extend(extracted.weekend1.sources)
    if extracted and extracted.weekend2 and extracted.weekend2.sources:
        all_urls.extend(extracted.weekend2.sources)
    if extracted and extracted.venue and extracted.venue.sources:
        all_urls.extend(extracted.venue.sources)
    if extracted and extracted.tickets and extracted.tickets.sources:
        all_urls.extend(extracted.tickets.sources)
    return _unique_urls(all_urls)


def _choose_sources(primary: Optional[List[str]], global_sources: List[str]) -> List[str]:
    p = _unique_urls(primary or [])
    if p:
        return p
    return _unique_urls(global_sources or [])


def _no_source_guard_instruction(base_instruction: str, sources: List[str]) -> str:
    if sources and len(sources) > 0:
        return base_instruction
    # If no sources present for a factual check, guide the judge to mark it unsupported
    return (
        base_instruction.strip() +
        "\nIMPORTANT: The answer did not provide any URL(s) to support this specific claim. "
        "Per the evaluation policy, treat the claim as NOT SUPPORTED and mark it Incorrect."
    )


# --------------------------------------------------------------------------- #
# Verification builders                                                       #
# --------------------------------------------------------------------------- #
async def build_basic_event_details(
    evaluator: Evaluator,
    parent,
    extracted: Coachella2026Extraction,
    global_sources: List[str],
):
    node = evaluator.add_parallel(
        id="Basic_Event_Details",
        desc="Fundamental event identification information",
        parent=parent,
        critical=True,
    )
    basic = extracted.basic or BasicEventDetails()

    # Event Name
    leaf = evaluator.add_leaf(
        id="Event_Name",
        desc="The festival is correctly identified as Coachella Valley Music and Arts Festival or Coachella",
        parent=node,
        critical=True,
    )
    src = _choose_sources(basic.sources, global_sources)
    claim = (
        f"The festival's official name is '{FESTIVAL_OFFICIAL_NAME}', which is commonly referred to as '{FESTIVAL_SHORT_NAME}'."
    )
    add_ins = _no_source_guard_instruction(
        "Use official festival or organizer sources when possible. Treat 'Coachella' as an acceptable shorthand "
        "for 'Coachella Valley Music and Arts Festival'.",
        src
    )
    await evaluator.verify(claim=claim, node=leaf, sources=src, additional_instruction=add_ins)

    # Event Year = 2026
    leaf = evaluator.add_leaf(
        id="Event_Year",
        desc="The year 2026 is correctly specified",
        parent=node,
        critical=True,
    )
    src = _choose_sources(basic.sources, global_sources)
    claim = "The Coachella festival edition in question is for the year 2026."
    add_ins = _no_source_guard_instruction(
        "Verify the edition is specifically the 2026 festival. Prefer official date/schedule pages or press releases.",
        src
    )
    await evaluator.verify(claim=claim, node=leaf, sources=src, additional_instruction=add_ins)

    # Location City = Indio
    leaf = evaluator.add_leaf(
        id="Location_City",
        desc="Indio is identified as the host city",
        parent=node,
        critical=True,
    )
    src = _choose_sources(basic.sources, global_sources)
    claim = "The Coachella festival takes place in Indio."
    add_ins = _no_source_guard_instruction(
        "Minor formatting like 'Indio, CA' should be accepted as indicating Indio, California.",
        src
    )
    await evaluator.verify(claim=claim, node=leaf, sources=src, additional_instruction=add_ins)

    # Location State = California
    leaf = evaluator.add_leaf(
        id="Location_State",
        desc="California is identified as the host state",
        parent=node,
        critical=True,
    )
    src = _choose_sources(basic.sources, global_sources)
    claim = "The Coachella festival takes place in California."
    add_ins = _no_source_guard_instruction(
        "If the source shows 'Indio, CA', that implies the state is California.",
        src
    )
    await evaluator.verify(claim=claim, node=leaf, sources=src, additional_instruction=add_ins)


async def build_weekend_1(
    evaluator: Evaluator,
    parent,
    extracted: Coachella2026Extraction,
    global_sources: List[str],
):
    node = evaluator.add_parallel(
        id="Weekend_1_Schedule",
        desc="Complete schedule information for Weekend 1",
        parent=parent,
        critical=True,
    )
    w1 = extracted.weekend1 or WeekendSchedule()
    src = _choose_sources(w1.sources, global_sources)

    # Start Date
    leaf = evaluator.add_leaf(
        id="W1_Start_Date",
        desc="Weekend 1 start date is April 10, 2026 (Friday)",
        parent=node,
        critical=True,
    )
    claim = "Weekend 1 of Coachella 2026 starts on April 10, 2026 (Friday)."
    add_ins = _no_source_guard_instruction(
        "Allow minor date format variations (e.g., 'Apr 10, 2026'). Ensure the date corresponds to Friday.",
        src
    )
    await evaluator.verify(claim=claim, node=leaf, sources=src, additional_instruction=add_ins)

    # End Date
    leaf = evaluator.add_leaf(
        id="W1_End_Date",
        desc="Weekend 1 end date is April 12, 2026 (Sunday)",
        parent=node,
        critical=True,
    )
    claim = "Weekend 1 of Coachella 2026 ends on April 12, 2026 (Sunday)."
    add_ins = _no_source_guard_instruction(
        "Allow minor date format variations. Ensure the date corresponds to Sunday.",
        src
    )
    await evaluator.verify(claim=claim, node=leaf, sources=src, additional_instruction=add_ins)

    # Friday Headliner
    leaf = evaluator.add_leaf(
        id="W1_Friday_Headliner",
        desc="Sabrina Carpenter is identified as the Friday headliner",
        parent=node,
        critical=True,
    )
    claim = "Sabrina Carpenter is the Friday headliner for Weekend 1 of Coachella 2026."
    add_ins = _no_source_guard_instruction(
        "Confirm that the performer listed is the headliner (top-billed act) for Friday of Weekend 1.",
        src
    )
    await evaluator.verify(claim=claim, node=leaf, sources=src, additional_instruction=add_ins)

    # Saturday Headliner
    leaf = evaluator.add_leaf(
        id="W1_Saturday_Headliner",
        desc="Justin Bieber is identified as the Saturday headliner",
        parent=node,
        critical=True,
    )
    claim = "Justin Bieber is the Saturday headliner for Weekend 1 of Coachella 2026."
    add_ins = _no_source_guard_instruction(
        "Confirm that the performer listed is the headliner for Saturday of Weekend 1.",
        src
    )
    await evaluator.verify(claim=claim, node=leaf, sources=src, additional_instruction=add_ins)

    # Sunday Headliner
    leaf = evaluator.add_leaf(
        id="W1_Sunday_Headliner",
        desc="Karol G is identified as the Sunday headliner",
        parent=node,
        critical=True,
    )
    claim = "Karol G is the Sunday headliner for Weekend 1 of Coachella 2026."
    add_ins = _no_source_guard_instruction(
        "Confirm that the performer listed is the headliner for Sunday of Weekend 1.",
        src
    )
    await evaluator.verify(claim=claim, node=leaf, sources=src, additional_instruction=add_ins)


async def build_weekend_2(
    evaluator: Evaluator,
    parent,
    extracted: Coachella2026Extraction,
    global_sources: List[str],
):
    node = evaluator.add_parallel(
        id="Weekend_2_Schedule",
        desc="Complete schedule information for Weekend 2",
        parent=parent,
        critical=True,
    )
    w2 = extracted.weekend2 or WeekendSchedule()
    src = _choose_sources(w2.sources, global_sources)

    # Start Date
    leaf = evaluator.add_leaf(
        id="W2_Start_Date",
        desc="Weekend 2 start date is April 17, 2026 (Friday)",
        parent=node,
        critical=True,
    )
    claim = "Weekend 2 of Coachella 2026 starts on April 17, 2026 (Friday)."
    add_ins = _no_source_guard_instruction(
        "Allow minor date format variations (e.g., 'Apr 17, 2026'). Ensure the date corresponds to Friday.",
        src
    )
    await evaluator.verify(claim=claim, node=leaf, sources=src, additional_instruction=add_ins)

    # End Date
    leaf = evaluator.add_leaf(
        id="W2_End_Date",
        desc="Weekend 2 end date is April 19, 2026 (Sunday)",
        parent=node,
        critical=True,
    )
    claim = "Weekend 2 of Coachella 2026 ends on April 19, 2026 (Sunday)."
    add_ins = _no_source_guard_instruction(
        "Allow minor date format variations. Ensure the date corresponds to Sunday.",
        src
    )
    await evaluator.verify(claim=claim, node=leaf, sources=src, additional_instruction=add_ins)

    # Friday Headliner
    leaf = evaluator.add_leaf(
        id="W2_Friday_Headliner",
        desc="Sabrina Carpenter is identified as the Friday headliner",
        parent=node,
        critical=True,
    )
    claim = "Sabrina Carpenter is the Friday headliner for Weekend 2 of Coachella 2026."
    add_ins = _no_source_guard_instruction(
        "Confirm that the performer listed is the headliner for Friday of Weekend 2.",
        src
    )
    await evaluator.verify(claim=claim, node=leaf, sources=src, additional_instruction=add_ins)

    # Saturday Headliner
    leaf = evaluator.add_leaf(
        id="W2_Saturday_Headliner",
        desc="Justin Bieber is identified as the Saturday headliner",
        parent=node,
        critical=True,
    )
    claim = "Justin Bieber is the Saturday headliner for Weekend 2 of Coachella 2026."
    add_ins = _no_source_guard_instruction(
        "Confirm that the performer listed is the headliner for Saturday of Weekend 2.",
        src
    )
    await evaluator.verify(claim=claim, node=leaf, sources=src, additional_instruction=add_ins)

    # Sunday Headliner
    leaf = evaluator.add_leaf(
        id="W2_Sunday_Headliner",
        desc="Karol G is identified as the Sunday headliner",
        parent=node,
        critical=True,
    )
    claim = "Karol G is the Sunday headliner for Weekend 2 of Coachella 2026."
    add_ins = _no_source_guard_instruction(
        "Confirm that the performer listed is the headliner for Sunday of Weekend 2.",
        src
    )
    await evaluator.verify(claim=claim, node=leaf, sources=src, additional_instruction=add_ins)


async def build_venue_information(
    evaluator: Evaluator,
    parent,
    extracted: Coachella2026Extraction,
    global_sources: List[str],
):
    node = evaluator.add_parallel(
        id="Venue_Information",
        desc="Venue details and specifications",
        parent=parent,
        critical=True,
    )
    venue = extracted.venue or VenueInfo()
    src = _choose_sources(venue.sources, global_sources)

    # Venue Name
    leaf = evaluator.add_leaf(
        id="Venue_Name",
        desc="Empire Polo Club is identified as the venue",
        parent=node,
        critical=True,
    )
    claim = "The Coachella 2026 festival is held at the Empire Polo Club."
    add_ins = _no_source_guard_instruction(
        "Prefer official festival pages or authoritative references that specify the venue.",
        src
    )
    await evaluator.verify(claim=claim, node=leaf, sources=src, additional_instruction=add_ins)

    # Venue Location
    leaf = evaluator.add_leaf(
        id="Venue_Location",
        desc="The venue is located in Indio, California",
        parent=node,
        critical=True,
    )
    claim = "The Empire Polo Club is located in Indio, California."
    add_ins = _no_source_guard_instruction(
        "Minor variants like 'Indio, CA' are acceptable as indicating Indio, California.",
        src
    )
    await evaluator.verify(claim=claim, node=leaf, sources=src, additional_instruction=add_ins)

    # Festival Capacity
    leaf = evaluator.add_leaf(
        id="Festival_Capacity",
        desc="The venue can accommodate up to 90,000 attendees for festivals",
        parent=node,
        critical=True,
    )
    claim = "The Coachella festival can accommodate approximately 90,000 attendees."
    add_ins = _no_source_guard_instruction(
        "Allow approximate phrasing such as 'about 90,000', 'around 90k', or 'up to 90,000'. "
        "Focus on the typical stated capacity for the festival at Empire Polo Club.",
        src
    )
    await evaluator.verify(claim=claim, node=leaf, sources=src, additional_instruction=add_ins)


async def build_ticket_pricing(
    evaluator: Evaluator,
    parent,
    extracted: Coachella2026Extraction,
    global_sources: List[str],
):
    node = evaluator.add_parallel(
        id="Ticket_Pricing",
        desc="Ticket pricing information for different pass types",
        parent=parent,
        critical=False,
    )
    tickets = extracted.tickets or TicketPricing()
    src = _choose_sources(tickets.sources, global_sources)

    # GA start price
    leaf = evaluator.add_leaf(
        id="GA_Ticket_Price",
        desc="General Admission tickets start at approximately $339",
        parent=node,
        critical=False,
    )
    claim = "General Admission (GA) passes start at approximately $339 (before fees)."
    add_ins = _no_source_guard_instruction(
        "Focus on the starting (lowest-tier) price. Accept variants like 'from $339' or minor formatting/currency variations. "
        "Ignore taxes/fees; check base price.",
        src
    )
    await evaluator.verify(claim=claim, node=leaf, sources=src, additional_instruction=add_ins)

    # GA+ start price
    leaf = evaluator.add_leaf(
        id="GA_Plus_Price",
        desc="GA+ tickets start at approximately $449",
        parent=node,
        critical=False,
    )
    claim = "GA+ passes start at approximately $449 (before fees)."
    add_ins = _no_source_guard_instruction(
        "Focus on the starting (lowest-tier) GA+ price. Accept 'from $449' or minor formatting/currency variations. Ignore taxes/fees.",
        src
    )
    await evaluator.verify(claim=claim, node=leaf, sources=src, additional_instruction=add_ins)

    # VIP start price
    leaf = evaluator.add_leaf(
        id="VIP_Price",
        desc="VIP tickets start at approximately $899",
        parent=node,
        critical=False,
    )
    claim = "VIP passes start at approximately $899 (before fees)."
    add_ins = _no_source_guard_instruction(
        "Focus on the starting VIP price. Accept 'from $899' or minor formatting/currency variations. Ignore taxes/fees.",
        src
    )
    await evaluator.verify(claim=claim, node=leaf, sources=src, additional_instruction=add_ins)


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

    # Extraction
    extracted: Coachella2026Extraction = await evaluator.extract(
        prompt=prompt_extract_coachella_2026(),
        template_class=Coachella2026Extraction,
        extraction_name="coachella_2026_extraction",
    )

    # Add ground-truth expectations for reference in the summary
    evaluator.add_ground_truth({
        "festival_official_name": FESTIVAL_OFFICIAL_NAME,
        "festival_short_name": FESTIVAL_SHORT_NAME,
        "year": YEAR_EXPECTED,
        "host_city": HOST_CITY,
        "host_state": HOST_STATE,
        "weekend_1": {"start": W1_START, "end": W1_END, "headliners": WEEKEND_HEADLINERS},
        "weekend_2": {"start": W2_START, "end": W2_END, "headliners": WEEKEND_HEADLINERS},
        "venue": {"name": VENUE_NAME, "city": VENUE_CITY, "state": VENUE_STATE, "capacity_approx": FESTIVAL_CAPACITY_APPROX},
        "starting_prices": {"GA": START_PRICE_GA, "GA+": START_PRICE_GA_PLUS, "VIP": START_PRICE_VIP}
    }, gt_type="expected_facts")

    # Build verification tree according to rubric
    await build_basic_event_details(evaluator, root, extracted, _collect_global_sources(extracted))
    await build_weekend_1(evaluator, root, extracted, _collect_global_sources(extracted))
    await build_weekend_2(evaluator, root, extracted, _collect_global_sources(extracted))
    await build_venue_information(evaluator, root, extracted, _collect_global_sources(extracted))
    await build_ticket_pricing(evaluator, root, extracted, _collect_global_sources(extracted))

    return evaluator.get_summary()