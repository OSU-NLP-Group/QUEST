import asyncio
import logging
import re
from typing import List, Optional, Dict, Any

from pydantic import BaseModel, Field
from obj_task_eval.llm_client.base_client import LLMClient

from obj_task_eval.evaluator import Evaluator, AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "concert_venues_2025_2026"
TASK_DESCRIPTION = (
    "Identify three major concert venues in the United States where high-profile artists are performing in late 2025 and 2026. "
    "Specifically, provide detailed information for: (1) The venue where Billie Eilish is performing in Miami, Florida during "
    "October 2025, (2) The venue where Billie Eilish is performing in San Francisco, California during November 2025, and (3) "
    "The venue where Yungblud is performing in New York City, New York during June 2026. For each of the three venues, provide "
    "the following information: official venue name, city and state location, exact performance date (or the first date if there "
    "are multiple consecutive performances at that venue), and venue seating capacity for concerts. Your answer must include "
    "reference URLs from official venue websites, tour schedule pages, or verified ticketing platforms to support the information provided."
)

# Ground truth constraints used for verification logic
MIAMI_EXPECTED = {
    "artist": "Billie Eilish",
    "venue": "Kaseya Center",
    "city": "Miami",
    "state": "Florida",
    "date": "October 9, 2025",
}
SF_EXPECTED = {
    "artist": "Billie Eilish",
    "venue": "Chase Center",
    "city": "San Francisco",
    "state": "California",
    "date": "November 22, 2025",
    "capacity_low": 18000,   # Approximate acceptable range for concerts
    "capacity_high": 20500,
}
NYC_EXPECTED = {
    "artist": "Yungblud",
    "venue": "Radio City Music Hall",
    "city": "New York City",
    "state": "New York",
    "date": "June 10, 2026",
    "capacity_low": 5900,    # Around ~5,960–6,000
    "capacity_high": 6100,
}


# --------------------------------------------------------------------------- #
# Data models for structured extraction                                       #
# --------------------------------------------------------------------------- #
class EventInfo(BaseModel):
    artist: Optional[str] = None
    venue_name: Optional[str] = None
    city: Optional[str] = None
    state: Optional[str] = None
    performance_date: Optional[str] = None
    concert_capacity: Optional[str] = None
    urls: List[str] = Field(default_factory=list)


class ConcertExtraction(BaseModel):
    miami: Optional[EventInfo] = None
    san_francisco: Optional[EventInfo] = None
    new_york: Optional[EventInfo] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_concert_info() -> str:
    return """
Extract the required details for the three specified U.S. performances as they appear in the provided answer text. Do not infer or invent anything.

For each of the following, extract the fields exactly as stated in the answer:
1) Billie Eilish in Miami, Florida during October 2025 → assign to 'miami'
2) Billie Eilish in San Francisco, California during November 2025 → assign to 'san_francisco'
   - If the answer mentions multiple consecutive dates at the same venue, return the first date in November 2025 for this entry.
3) Yungblud in New York City, New York during June 2026 → assign to 'new_york'

For each entry, extract the following fields:
- artist: The artist name for this stop.
- venue_name: The official name of the venue as written in the answer.
- city: The city for the venue (e.g., "Miami", "San Francisco", "New York City").
- state: The U.S. state for the venue (e.g., "Florida", "California", "New York"). If the answer uses an abbreviation (e.g., FL, CA, NY), extract as written.
- performance_date: The exact performance date for that venue stop. If multiple consecutive dates are listed for the same venue, extract the first date for that venue (e.g., "November 22, 2025" for the San Francisco entry).
- concert_capacity: A concrete concert seating capacity figure or a clearly stated range as written (e.g., "19,500 for concerts", "5,960–6,000").
- urls: All URLs mentioned in the answer that directly support this venue/date information (official venue websites, the artist’s official tour/schedule pages, or verified ticketing platforms such as Ticketmaster, Live Nation, AXS, SeatGeek, Eventim, Tickets.com). Extract the actual URLs only.

If any field is not specified in the answer for a required entry, set it to null (or an empty list for 'urls').
"""


# --------------------------------------------------------------------------- #
# Helper utilities for capacity checks                                        #
# --------------------------------------------------------------------------- #
def _extract_numeric_tokens(capacity_text: Optional[str]) -> List[int]:
    if not capacity_text:
        return []
    s = capacity_text

    # Capture numbers like 19,500 or 5960
    nums = [int(x.replace(",", "")) for x in re.findall(r"\d[\d,]*", s)]

    # Capture patterns like 19.5k or 20k
    k_matches = re.findall(r"(\d+(?:\.\d+)?)\s*[kK]\b", s)
    for km in k_matches:
        try:
            val = float(km) * 1000.0
            nums.append(int(round(val)))
        except Exception:
            pass

    # Deduplicate while preserving order
    seen = set()
    unique_nums = []
    for n in nums:
        if n not in seen:
            seen.add(n)
            unique_nums.append(n)
    return unique_nums


def capacity_provided(capacity_text: Optional[str]) -> bool:
    return len(_extract_numeric_tokens(capacity_text)) > 0


def capacity_in_range(capacity_text: Optional[str], low: int, high: int) -> bool:
    vals = _extract_numeric_tokens(capacity_text)
    if not vals:
        return False
    # Accept if any mentioned capacity falls within the expected range
    return any(low <= v <= high for v in vals)


# --------------------------------------------------------------------------- #
# Verification subroutines for each city                                      #
# --------------------------------------------------------------------------- #
async def verify_miami(evaluator: Evaluator, parent_node, miami: Optional[EventInfo]) -> None:
    node = evaluator.add_parallel(
        id="miami_venue_billie_eilish_oct_2025",
        desc="Venue details for Billie Eilish in Miami, Florida in October 2025.",
        parent=parent_node,
        critical=False
    )

    artist = (miami.artist or "").strip()
    venue = (miami.venue_name or "").strip()
    city = (miami.city or "").strip()
    state = (miami.state or "").strip()
    date = (miami.performance_date or "").strip()
    capacity_txt = (miami.concert_capacity or "").strip()
    urls = miami.urls if (miami and miami.urls) else []

    # Miami_Correct_Venue_Identified
    leaf = evaluator.add_leaf(
        id="Miami_Correct_Venue_Identified",
        desc="Identifies the correct venue for this performance (matches constraints: Kaseya Center).",
        parent=node,
        critical=True
    )
    claim = f"The extracted venue '{venue}' and the expected venue '{MIAMI_EXPECTED['venue']}' refer to the same venue."
    await evaluator.verify(
        claim=claim,
        node=leaf,
        additional_instruction="Allow minor naming variations (e.g., punctuation, suffixes). Focus on whether both strings denote the same venue."
    )

    # Miami_City_State_Provided
    leaf = evaluator.add_leaf(
        id="Miami_City_State_Provided",
        desc="Provides the city and state location (Miami, Florida).",
        parent=node,
        critical=True
    )
    loc_claim = f"The extracted location '{city}, {state}' matches the expected 'Miami, Florida'."
    await evaluator.verify(
        claim=loc_claim,
        node=leaf,
        additional_instruction="Consider 'FL' equivalent to 'Florida'. Be tolerant of casing and minor formatting differences."
    )

    # Miami_Correct_Performance_Date
    leaf = evaluator.add_leaf(
        id="Miami_Correct_Performance_Date",
        desc="Provides the exact performance date for this venue stop (matches constraints: October 9, 2025).",
        parent=node,
        critical=True
    )
    date_claim = f"The extracted performance date '{date}' equals '{MIAMI_EXPECTED['date']}'."
    await evaluator.verify(
        claim=date_claim,
        node=leaf,
        additional_instruction="Treat common shorthand (e.g., 'Oct 9, 2025') as equivalent to the full form 'October 9, 2025'."
    )

    # Miami_Concert_Capacity_Provided (concrete figure or range)
    evaluator.add_custom_node(
        result=capacity_provided(capacity_txt),
        id="Miami_Concert_Capacity_Provided",
        desc="Provides a concrete venue seating capacity for concerts (a specific figure or clearly stated range).",
        parent=node,
        critical=True
    )

    # Miami_Reference_URLs_Valid
    leaf = evaluator.add_leaf(
        id="Miami_Reference_URLs_Valid",
        desc="Includes reference URL(s) from official venue websites, official tour/artist pages, or verified ticketing platforms that support the provided venue/date/capacity information.",
        parent=node,
        critical=True
    )
    ref_claim = (
        f"This page confirms that {MIAMI_EXPECTED['artist']} will perform at {MIAMI_EXPECTED['venue']} in "
        f"{MIAMI_EXPECTED['city']}, {MIAMI_EXPECTED['state']} on {MIAMI_EXPECTED['date']}."
    )
    await evaluator.verify(
        claim=ref_claim,
        node=leaf,
        sources=urls,
        additional_instruction=(
            "The page must be from an official venue website, the artist's official tour/schedule page, "
            "or a widely recognized/verified ticketing platform (e.g., Ticketmaster, Live Nation, AXS, SeatGeek, Eventim, Tickets.com). "
            "If the URL is irrelevant, unofficial, or does not explicitly confirm the event details, mark as not supported."
        )
    )


async def verify_sf(evaluator: Evaluator, parent_node, sf: Optional[EventInfo]) -> None:
    node = evaluator.add_parallel(
        id="san_francisco_venue_billie_eilish_nov_2025",
        desc="Venue details for Billie Eilish in San Francisco, California in November 2025.",
        parent=parent_node,
        critical=False
    )

    artist = (sf.artist or "").strip()
    venue = (sf.venue_name or "").strip()
    city = (sf.city or "").strip()
    state = (sf.state or "").strip()
    date = (sf.performance_date or "").strip()
    capacity_txt = (sf.concert_capacity or "").strip()
    urls = sf.urls if (sf and sf.urls) else []

    # SF_Correct_Venue_Identified
    leaf = evaluator.add_leaf(
        id="SF_Correct_Venue_Identified",
        desc="Identifies the correct venue for this performance (matches constraints: Chase Center).",
        parent=node,
        critical=True
    )
    claim = f"The extracted venue '{venue}' and the expected venue '{SF_EXPECTED['venue']}' refer to the same venue."
    await evaluator.verify(
        claim=claim,
        node=leaf,
        additional_instruction="Allow minor naming variations and punctuation; judge whether both strings refer to the same venue."
    )

    # SF_City_State_Provided
    leaf = evaluator.add_leaf(
        id="SF_City_State_Provided",
        desc="Provides the city and state location (San Francisco, California).",
        parent=node,
        critical=True
    )
    loc_claim = f"The extracted location '{city}, {state}' matches the expected 'San Francisco, California'."
    await evaluator.verify(
        claim=loc_claim,
        node=leaf,
        additional_instruction="Consider 'CA' equivalent to 'California'. Be tolerant of minor formatting and casing differences."
    )

    # SF_Correct_Performance_Date (first date if multiple in November 2025)
    leaf = evaluator.add_leaf(
        id="SF_Correct_Performance_Date",
        desc="Provides the exact performance date, using the first date if multiple consecutive performances occur (matches constraints: first date November 22, 2025).",
        parent=node,
        critical=True
    )
    date_claim = f"The extracted performance date '{date}' equals '{SF_EXPECTED['date']}'."
    await evaluator.verify(
        claim=date_claim,
        node=leaf,
        additional_instruction="If multiple November 2025 dates at the same venue are in the answer, the first date should be used. Accept common date abbreviations."
    )

    # SF_Concert_Capacity_Consistent_With_Constraints (up to ~19,500 for concerts)
    evaluator.add_custom_node(
        result=capacity_in_range(capacity_txt, SF_EXPECTED["capacity_low"], SF_EXPECTED["capacity_high"]),
        id="SF_Concert_Capacity_Consistent_With_Constraints",
        desc="Provides the venue seating capacity for concerts and it is consistent with the constraints (up to ~19,500 for concerts).",
        parent=node,
        critical=True
    )

    # SF_Reference_URLs_Valid
    leaf = evaluator.add_leaf(
        id="SF_Reference_URLs_Valid",
        desc="Includes reference URL(s) from official venue websites, official tour/artist pages, or verified ticketing platforms that support the provided venue/date/capacity information.",
        parent=node,
        critical=True
    )
    ref_claim = (
        f"This page confirms that {SF_EXPECTED['artist']} will perform at {SF_EXPECTED['venue']} in "
        f"{SF_EXPECTED['city']}, {SF_EXPECTED['state']} on {SF_EXPECTED['date']}."
    )
    await evaluator.verify(
        claim=ref_claim,
        node=leaf,
        sources=urls,
        additional_instruction=(
            "Only accept the claim as supported if the URL is an official venue website, the artist's official tour/schedule page, "
            "or a recognized/verified ticketing platform (e.g., Ticketmaster, Live Nation, AXS, SeatGeek, Eventim, Tickets.com) and it explicitly confirms the event details."
        )
    )


async def verify_nyc(evaluator: Evaluator, parent_node, nyc: Optional[EventInfo]) -> None:
    node = evaluator.add_parallel(
        id="new_york_venue_yungblud_jun_2026",
        desc="Venue details for Yungblud in New York City, New York in June 2026.",
        parent=parent_node,
        critical=False
    )

    artist = (nyc.artist or "").strip()
    venue = (nyc.venue_name or "").strip()
    city = (nyc.city or "").strip()
    state = (nyc.state or "").strip()
    date = (nyc.performance_date or "").strip()
    capacity_txt = (nyc.concert_capacity or "").strip()
    urls = nyc.urls if (nyc and nyc.urls) else []

    # NYC_Correct_Venue_Identified
    leaf = evaluator.add_leaf(
        id="NYC_Correct_Venue_Identified",
        desc="Identifies the correct venue for this performance (matches constraints: Radio City Music Hall).",
        parent=node,
        critical=True
    )
    claim = f"The extracted venue '{venue}' and the expected venue '{NYC_EXPECTED['venue']}' refer to the same venue."
    await evaluator.verify(
        claim=claim,
        node=leaf,
        additional_instruction="Allow minor naming variations; judge whether both strings refer to the same venue."
    )

    # NYC_City_State_Provided
    leaf = evaluator.add_leaf(
        id="NYC_City_State_Provided",
        desc="Provides the city and state location (New York City, New York).",
        parent=node,
        critical=True
    )
    loc_claim = f"The extracted location '{city}, {state}' matches the expected 'New York City, New York'."
    await evaluator.verify(
        claim=loc_claim,
        node=leaf,
        additional_instruction="Treat 'NYC' as equivalent to 'New York City' and 'NY' equivalent to 'New York'. Allow minor formatting differences."
    )

    # NYC_Correct_Performance_Date
    leaf = evaluator.add_leaf(
        id="NYC_Correct_Performance_Date",
        desc="Provides the exact performance date (matches constraints: June 10, 2026).",
        parent=node,
        critical=True
    )
    date_claim = f"The extracted performance date '{date}' equals '{NYC_EXPECTED['date']}'."
    await evaluator.verify(
        claim=date_claim,
        node=leaf,
        additional_instruction="Accept common date abbreviations (e.g., 'Jun 10, 2026')."
    )

    # NYC_Concert_Capacity_Consistent_With_Constraints (~5,960–6,000)
    evaluator.add_custom_node(
        result=capacity_in_range(capacity_txt, NYC_EXPECTED["capacity_low"], NYC_EXPECTED["capacity_high"]),
        id="NYC_Concert_Capacity_Consistent_With_Constraints",
        desc="Provides the venue seating capacity for concerts and it is consistent with the constraints (~5,960–6,000).",
        parent=node,
        critical=True
    )

    # NYC_Reference_URLs_Valid
    leaf = evaluator.add_leaf(
        id="NYC_Reference_URLs_Valid",
        desc="Includes reference URL(s) from official venue websites, official tour/artist pages, or verified ticketing platforms that support the provided venue/date/capacity information.",
        parent=node,
        critical=True
    )
    ref_claim = (
        f"This page confirms that {NYC_EXPECTED['artist']} will perform at {NYC_EXPECTED['venue']} in "
        f"{NYC_EXPECTED['city']}, {NYC_EXPECTED['state']} on {NYC_EXPECTED['date']}."
    )
    await evaluator.verify(
        claim=ref_claim,
        node=leaf,
        sources=urls,
        additional_instruction=(
            "Only accept the claim as supported if the URL is an official venue website, the artist's official tour/schedule page, "
            "or a recognized/verified ticketing platform (e.g., Ticketmaster, Live Nation, AXS, SeatGeek, Eventim, Tickets.com) and it explicitly confirms the event details."
        )
    )


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
    Evaluate an answer for the 'concert_venues_2025_2026' task and return a structured result dictionary.
    """
    # Initialize evaluator
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

    # Add a top-level aggregator to mirror rubric's "Concert_Venue_Information"
    top = evaluator.add_parallel(
        id="concert_venue_information",
        desc="Provide required details for the three specified US performances and support them with appropriate reference URLs.",
        parent=root,
        critical=False
    )

    # Extract structured info from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_concert_info(),
        template_class=ConcertExtraction,
        extraction_name="concert_venue_extraction"
    )

    # Add GT/constraints info
    evaluator.add_ground_truth({
        "miami_expected": MIAMI_EXPECTED,
        "san_francisco_expected": {
            "venue": SF_EXPECTED["venue"],
            "city": SF_EXPECTED["city"],
            "state": SF_EXPECTED["state"],
            "date": SF_EXPECTED["date"],
            "capacity_range_hint": [SF_EXPECTED["capacity_low"], SF_EXPECTED["capacity_high"]]
        },
        "new_york_expected": {
            "venue": NYC_EXPECTED["venue"],
            "city": NYC_EXPECTED["city"],
            "state": NYC_EXPECTED["state"],
            "date": NYC_EXPECTED["date"],
            "capacity_range_hint": [NYC_EXPECTED["capacity_low"], NYC_EXPECTED["capacity_high"]]
        }
    }, gt_type="constraints")

    # Build verification subtrees
    await verify_miami(evaluator, top, extracted.miami)
    await verify_sf(evaluator, top, extracted.san_francisco)
    await verify_nyc(evaluator, top, extracted.new_york)

    # Return evaluation summary
    return evaluator.get_summary()