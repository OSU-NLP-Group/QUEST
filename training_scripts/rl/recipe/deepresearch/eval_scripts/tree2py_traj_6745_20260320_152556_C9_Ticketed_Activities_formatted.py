import asyncio
import logging
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "summer_2026_country_amphitheaters"
TASK_DESCRIPTION = """
I am planning to attend country music concerts at outdoor amphitheaters during summer 2026. Please identify three outdoor amphitheater concerts between June 1 and August 31, 2026, where each concert must meet all of the following requirements:

1. The concert must be held at an outdoor amphitheater venue with a documented seating capacity between 10,000 and 25,000 people.

2. The headlining performer must be a country music artist or band.

3. The concert must be scheduled to begin at 6:00 PM or later in the evening.

4. Tickets must be available in at least three distinct price tiers (such as Pit, Floor, and Grandstand, or equivalent reserved and lawn sections).

5. The highest-priced ticket tier must cost at least $75.

6. Each of the three concerts must be located in a different U.S. state.

For each concert, please provide:
- The name of the venue
- The city and state where the venue is located
- A URL reference to the official venue information or capacity documentation
- The name of the headlining country music artist or band
- A URL reference to the artist's official website or tour page
- The specific date and start time of the concert
- A URL reference to the official concert schedule
- The ticket price tiers and their respective prices
- A URL reference to the official ticket pricing information
"""

SUMMER_START_TEXT = "June 1, 2026"
SUMMER_END_TEXT = "August 31, 2026"


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class PriceTier(BaseModel):
    name: Optional[str] = None
    price: Optional[str] = None  # Keep as string (e.g., "$89.50", "From $75")


class ConcertItem(BaseModel):
    # Venue
    venue_name: Optional[str] = None
    venue_city: Optional[str] = None
    venue_state: Optional[str] = None  # Accept "CA" or "California"
    venue_urls: List[str] = Field(default_factory=list)  # Official venue/capacity docs
    claimed_capacity: Optional[str] = None  # Optional textual capacity if present in answer

    # Artist
    artist_name: Optional[str] = None
    artist_url: Optional[str] = None  # Official website or tour page

    # Schedule
    date: Optional[str] = None  # Free text date, e.g., "June 15, 2026"
    start_time: Optional[str] = None  # e.g., "7:00 PM", "19:30"
    schedule_url: Optional[str] = None  # Official concert schedule URL

    # Tickets
    ticket_tiers: List[PriceTier] = Field(default_factory=list)  # Named tiers with prices
    pricing_url: Optional[str] = None  # Official ticket pricing URL


class ConcertsExtraction(BaseModel):
    concerts: List[ConcertItem] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_concerts() -> str:
    return """
Extract up to three distinct concerts from the answer, preserving the exact wording used in the answer for all fields. Return a JSON object with field "concerts": an array of at most 3 objects. For each concert object, extract the following fields exactly as they appear:

- venue_name: The name of the venue.
- venue_city: The city of the venue.
- venue_state: The U.S. state of the venue (either the two-letter code or full state name).
- venue_urls: An array of one or more URLs referencing official venue information and/or capacity documentation. Include only explicit URLs shown in the answer.
- claimed_capacity: The textual capacity mentioned in the answer, if any (e.g., "20,000" or "20,000 (pavilion) + lawn"); otherwise null.

- artist_name: The headlining artist or band name.
- artist_url: The official website or official tour page URL for the artist/band (if provided).

- date: The concert date text (e.g., "June 15, 2026").
- start_time: The concert start time text (e.g., "7:00 PM", "19:30").
- schedule_url: The URL to the official concert schedule or event listing page.

- ticket_tiers: An array of objects for ticket tiers with:
  - name: The tier/section name (e.g., "Pit", "Lawn", "Reserved", "VIP").
  - price: The textual price for that tier (e.g., "$89.50", "From $75", "$75+").
- pricing_url: The URL pointing to the official ticket pricing information page.

GENERAL RULES:
1) Do not invent any information. Only extract what is explicitly present in the provided answer.
2) If any field is missing in the answer, set it to null (or [] for arrays).
3) For URLs, extract the actual URLs only (plain or in markdown).
4) Keep date and time as strings exactly as presented; do not normalize formats.
5) Extract at most three concerts. If more are present, include only the first three in the order they appear.
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
US_STATES = {
    "AL": "Alabama", "AK": "Alaska", "AZ": "Arizona", "AR": "Arkansas", "CA": "California",
    "CO": "Colorado", "CT": "Connecticut", "DE": "Delaware", "FL": "Florida", "GA": "Georgia",
    "HI": "Hawaii", "ID": "Idaho", "IL": "Illinois", "IN": "Indiana", "IA": "Iowa",
    "KS": "Kansas", "KY": "Kentucky", "LA": "Louisiana", "ME": "Maine", "MD": "Maryland",
    "MA": "Massachusetts", "MI": "Michigan", "MN": "Minnesota", "MS": "Mississippi", "MO": "Missouri",
    "MT": "Montana", "NE": "Nebraska", "NV": "Nevada", "NH": "New Hampshire", "NJ": "New Jersey",
    "NM": "New Mexico", "NY": "New York", "NC": "North Carolina", "ND": "North Dakota", "OH": "Ohio",
    "OK": "Oklahoma", "OR": "Oregon", "PA": "Pennsylvania", "RI": "Rhode Island", "SC": "South Carolina",
    "SD": "South Dakota", "TN": "Tennessee", "TX": "Texas", "UT": "Utah", "VT": "Vermont",
    "VA": "Virginia", "WA": "Washington", "WV": "West Virginia", "WI": "Wisconsin", "WY": "Wyoming",
    "DC": "District of Columbia",
}


def normalize_state(state: Optional[str]) -> Optional[str]:
    if not state:
        return None
    s = state.strip()
    if not s:
        return None
    # Check 2-letter code
    s_up = s.upper()
    if s_up in US_STATES:
        return s_up
    # Try full name mapping
    # Normalize capitalization to title-case for matching
    s_title = s.lower().strip().replace(".", "").replace(",", "")
    # Special handling for DC variants
    if s_title in {"washington dc", "district of columbia", "washington d c", "washington, dc"}:
        return "DC"
    for code, fullname in US_STATES.items():
        if s_title == fullname.lower():
            return code
    return s_up  # Fallback to upper string (may still be okay for equality)


def valid_urls(urls: Optional[List[str]]) -> List[str]:
    if not urls:
        return []
    out = []
    for u in urls:
        if isinstance(u, str) and u.strip():
            uu = u.strip()
            if uu.startswith("http://") or uu.startswith("https://"):
                out.append(uu)
            else:
                # allow missing protocol by prefixing http:// as per extraction special rules
                out.append("http://" + uu)
    return out


def format_tiers_for_claim(tiers: List[PriceTier]) -> str:
    if not tiers:
        return "none"
    parts = []
    for t in tiers:
        n = (t.name or "").strip()
        p = (t.price or "").strip()
        if n and p:
            parts.append(f"{n}: {p}")
        elif n:
            parts.append(f"{n}: [no price]")
        elif p:
            parts.append(f"[unnamed]: {p}")
    return "; ".join(parts) if parts else "none"


# --------------------------------------------------------------------------- #
# Verification for a single concert                                           #
# --------------------------------------------------------------------------- #
async def verify_concert(
    evaluator: Evaluator,
    parent_node,
    concert: ConcertItem,
    concert_index: int,
    all_states_normalized: List[Optional[str]],
) -> None:
    idx = concert_index + 1
    concert_node = evaluator.add_parallel(
        id=f"concert_{concert_index+1}",
        desc=f"{['First', 'Second', 'Third'][concert_index]} summer amphitheater concert meeting all specified criteria",
        parent=parent_node,
        critical=False,
    )

    # ---------------- Venue Information (Critical, Parallel) ----------------
    venue_node = evaluator.add_parallel(
        id=f"c{idx}_venue_info",
        desc="Complete venue information is provided and verified",
        parent=concert_node,
        critical=True,
    )

    # Existence: Venue name, city, state provided
    has_venue_core = all([
        bool((concert.venue_name or "").strip()),
        bool((concert.venue_city or "").strip()),
        bool((concert.venue_state or "").strip()),
    ])
    evaluator.add_custom_node(
        result=has_venue_core,
        id=f"c{idx}_venue_name_city_state",
        desc="Venue name, city, and state are all provided",
        parent=venue_node,
        critical=True,
    )

    # Existence: Venue documentation URLs present
    venue_urls_list = valid_urls(concert.venue_urls)
    evaluator.add_custom_node(
        result=len(venue_urls_list) > 0,
        id=f"c{idx}_venue_docs_urls",
        desc="URL references to official venue information and capacity documentation are provided",
        parent=venue_node,
        critical=True,
    )

    # Verify: Venue is an outdoor amphitheater
    amph_node = evaluator.add_leaf(
        id=f"c{idx}_outdoor_amphitheater",
        desc="Venue is documented as an outdoor amphitheater",
        parent=venue_node,
        critical=True,
    )
    amph_claim = f"The venue '{concert.venue_name or 'the venue'}' is documented as an outdoor (open-air) amphitheater."
    await evaluator.verify(
        claim=amph_claim,
        node=amph_node,
        sources=venue_urls_list,
        additional_instruction="Confirm that the referenced venue page(s) explicitly indicate it is an outdoor/open-air amphitheater; acceptable synonyms include 'outdoor amphitheatre', 'open-air amphitheater', 'outdoor concert amphitheater'.",
    )

    # Verify: Capacity in 10,000–25,000 range
    cap_node = evaluator.add_leaf(
        id=f"c{idx}_capacity_range",
        desc="Venue capacity is documented as between 10,000 and 25,000 people",
        parent=venue_node,
        critical=True,
    )
    cap_claim = (
        f"The seating capacity of '{concert.venue_name or 'the venue'}' is between 10,000 and 25,000 people."
    )
    await evaluator.verify(
        claim=cap_claim,
        node=cap_node,
        sources=venue_urls_list,
        additional_instruction="Verify that the venue capacity documented on the provided venue/capacity source(s) lies within 10,000 to 25,000 inclusive. "
                               "Accept either 'seating capacity' or an explicitly documented 'total capacity' if it's unambiguous from the page.",
    )

    # ---------------- Artist Information (Critical, Parallel) ---------------
    artist_node = evaluator.add_parallel(
        id=f"c{idx}_artist_info",
        desc="Headlining artist information is provided and verified",
        parent=concert_node,
        critical=True,
    )

    has_artist_name = bool((concert.artist_name or "").strip())
    evaluator.add_custom_node(
        result=has_artist_name,
        id=f"c{idx}_artist_name_provided",
        desc="Name of headlining artist or band is provided",
        parent=artist_node,
        critical=True,
    )

    artist_url_single = (concert.artist_url or "").strip()
    artist_url_valid = valid_urls([artist_url_single]) if artist_url_single else []
    evaluator.add_custom_node(
        result=len(artist_url_valid) > 0,
        id=f"c{idx}_artist_ref_url",
        desc="URL reference to artist's official website or tour page is provided",
        parent=artist_node,
        critical=True,
    )

    genre_node = evaluator.add_leaf(
        id=f"c{idx}_artist_country_genre",
        desc="Artist is documented as a country music performer",
        parent=artist_node,
        critical=True,
    )
    genre_claim = f"The headlining artist '{concert.artist_name or 'the artist'}' is a country music performer or band (including country sub-genres)."
    await evaluator.verify(
        claim=genre_claim,
        node=genre_node,
        sources=artist_url_valid[0] if len(artist_url_valid) == 1 else artist_url_valid,
        additional_instruction="Check the official website or tour page to confirm the act is a country music performer (or a recognized country sub-genre such as country pop, country rock, Americana/country, etc.).",
    )

    # ---------------- Date & Time (Critical, Parallel) ----------------------
    dt_node = evaluator.add_parallel(
        id=f"c{idx}_date_time_info",
        desc="Concert date and time meet specified requirements",
        parent=concert_node,
        critical=True,
    )

    schedule_url_single = (concert.schedule_url or "").strip()
    schedule_urls = valid_urls([schedule_url_single]) if schedule_url_single else []
    evaluator.add_custom_node(
        result=len(schedule_urls) > 0,
        id=f"c{idx}_schedule_ref_url",
        desc="URL reference to official concert schedule is provided",
        parent=dt_node,
        critical=True,
    )

    # Verify: Date in summer 2026
    date_node = evaluator.add_leaf(
        id=f"c{idx}_date_in_summer_2026",
        desc="Concert date is between June 1 and August 31, 2026 (inclusive)",
        parent=dt_node,
        critical=True,
    )
    date_claim = (
        f"The concert is scheduled on {concert.date or '[date not provided]'}, which must fall between {SUMMER_START_TEXT} and {SUMMER_END_TEXT} (inclusive)."
    )
    await evaluator.verify(
        claim=date_claim,
        node=date_node,
        sources=schedule_urls,
        additional_instruction="From the official schedule/event page, verify that the event date is within the inclusive range June 1, 2026 to August 31, 2026.",
    )

    # Verify: Evening start time >= 6:00 PM
    time_node = evaluator.add_leaf(
        id=f"c{idx}_evening_start_time",
        desc="Concert start time is 6:00 PM or later",
        parent=dt_node,
        critical=True,
    )
    time_claim = (
        f"The concert start time is {concert.start_time or '[time not provided]'}, and it is at or after 6:00 PM local time."
    )
    await evaluator.verify(
        claim=time_claim,
        node=time_node,
        sources=schedule_urls,
        additional_instruction="Use the official schedule/event page to confirm the show/performance start time (ignore 'doors' or 'gates' times). "
                               "Accept formats like '7 PM', '7:30PM', '19:00'. The time must be >= 6:00 PM local time.",
    )

    # ---------------- Ticket Pricing (Critical, Parallel) -------------------
    price_node = evaluator.add_parallel(
        id=f"c{idx}_ticket_pricing_info",
        desc="Complete ticket pricing information meets requirements",
        parent=concert_node,
        critical=True,
    )

    pricing_url_single = (concert.pricing_url or "").strip()
    pricing_urls = valid_urls([pricing_url_single]) if pricing_url_single else []
    evaluator.add_custom_node(
        result=len(pricing_urls) > 0,
        id=f"c{idx}_pricing_ref_url",
        desc="URL reference to official ticket pricing information is provided",
        parent=price_node,
        critical=True,
    )

    tiers_text = format_tiers_for_claim(concert.ticket_tiers)

    # Verify: At least three distinct tiers, each with prices
    tiers_node = evaluator.add_leaf(
        id=f"c{idx}_three_distinct_tiers",
        desc="Concert offers at least three distinct ticket price tiers with prices provided for each tier",
        parent=price_node,
        critical=True,
    )
    tiers_claim = (
        f"According to the official ticket/pricing page, this concert offers at least three distinct ticket tiers with prices. "
        f"Claimed tiers include: {tiers_text}."
    )
    await evaluator.verify(
        claim=tiers_claim,
        node=tiers_node,
        sources=pricing_urls,
        additional_instruction="Verify the page shows at least three distinct named sections/tier categories (e.g., Pit, Reserved, Lawn, VIP) and that each has an associated price (or 'from $X'). "
                               "Dynamic ranges or 'from' pricing are acceptable as long as specific numeric prices are shown per tier.",
    )

    # Verify: Highest-priced tier >= $75
    high_node = evaluator.add_leaf(
        id=f"c{idx}_highest_tier_at_least_75",
        desc="The highest-priced ticket tier costs at least $75",
        parent=price_node,
        critical=True,
    )
    high_claim = (
        f"Among the ticket tiers listed ({tiers_text}), the highest-priced ticket tier is at least $75."
    )
    await evaluator.verify(
        claim=high_claim,
        node=high_node,
        sources=pricing_urls,
        additional_instruction="Confirm from the official pricing/ticketing page that the maximum listed price for any available tier is >= $75. "
                               "Base price (before fees) is acceptable; if only total price with fees is shown, that is also acceptable.",
    )

    # ---------------- State Uniqueness (Critical) --------------------------
    # All three concerts must be in different U.S. states
    states_present = [s for s in all_states_normalized if s]
    all_three_present = len(states_present) == 3
    all_distinct = len(set(states_present)) == 3 if all_three_present else False
    this_state_norm = normalize_state(concert.venue_state)

    evaluator.add_custom_node(
        result=bool(this_state_norm) and all_distinct,
        id=f"c{idx}_state_uniqueness",
        desc="Concert is in a different US state than the other two concerts",
        parent=concert_node,
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
    extracted = await evaluator.extract(
        prompt=prompt_extract_concerts(),
        template_class=ConcertsExtraction,
        extraction_name="concerts_extraction",
    )

    # Normalize to exactly 3 concerts (pad with empty if fewer)
    concerts: List[ConcertItem] = list(extracted.concerts[:3])
    while len(concerts) < 3:
        concerts.append(ConcertItem())

    # Prepare normalized states list for uniqueness checks
    normalized_states = [normalize_state(c.venue_state) for c in concerts]

    # Build main node and verify each concert in parallel branches
    main_node = evaluator.add_parallel(
        id="three_summer_amphitheater_concerts",
        desc="Identification of three outdoor amphitheater concerts in summer 2026, each in a different US state",
        parent=root,
        critical=False,
    )

    # Verify each concert subtree
    for i in range(3):
        await verify_concert(
            evaluator=evaluator,
            parent_node=main_node,
            concert=concerts[i],
            concert_index=i,
            all_states_normalized=normalized_states,
        )

    # Optionally, record custom info for debugging/traceability
    evaluator.add_custom_info(
        {
            "extracted_states_raw": [c.venue_state for c in concerts],
            "extracted_states_normalized": normalized_states,
        },
        info_type="debug_info",
        info_name="state_normalization",
    )

    return evaluator.get_summary()