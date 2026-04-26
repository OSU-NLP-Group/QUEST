import asyncio
import logging
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "mccartney_got_back_2025_consecutive_venues"
TASK_DESCRIPTION = (
    "Paul McCartney's Got Back 2025 North America tour includes performances at multiple major venues across the United States and Canada. "
    "Identify all venues where Paul McCartney is scheduled to perform on consecutive nights at the same location. For each venue, provide the following information: "
    "the official venue name, complete street address, city, state or province, both specific performance dates, the venue's seating capacity, and a reference URL confirming the tour information."
)


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class VenueDetails(BaseModel):
    venue_name: Optional[str] = None
    street_address: Optional[str] = None
    city: Optional[str] = None
    state_province: Optional[str] = None
    date_1: Optional[str] = None
    date_2: Optional[str] = None
    seating_capacity: Optional[str] = None
    reference_urls: List[str] = Field(default_factory=list)


class TourVenuesExtraction(BaseModel):
    """
    Extract details for these consecutive-night venues if they appear in the answer.
    If an entry is not present in the answer, return null for that entry.
    """
    state_farm_arena: Optional[VenueDetails] = None     # Atlanta, GA
    bell_centre: Optional[VenueDetails] = None          # Montreal, QC
    united_center: Optional[VenueDetails] = None        # Chicago, IL


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_consecutive_venues() -> str:
    return """
    From the answer, extract the details for Paul McCartney's 2025 Got Back tour venues where he performs on consecutive nights at the same location for the following venues (if they appear in the answer):
    - State Farm Arena (Atlanta, GA, USA)
    - Bell Centre / Centre Bell (Montreal, QC, Canada)
    - United Center (Chicago, IL, USA)

    For each venue, extract the following fields exactly as they appear in the answer text:
    - venue_name: the official venue name as written (e.g., "State Farm Arena", "Bell Centre" or "Centre Bell", "United Center")
    - street_address: the street-line portion only (e.g., "1 State Farm Drive", "1909 Av. des Canadiens-de-Montréal", "1901 W Madison Street")
                     Do not include city, state/province, postal code in this field.
    - city: the city name (e.g., "Atlanta", "Montreal" or "Montréal", "Chicago")
    - state_province: the state or province (e.g., "Georgia" or "GA"; "Quebec" or "QC"; "Illinois" or "IL")
    - date_1: the earlier of the two consecutive performance dates for this venue, exactly as shown in the answer (e.g., "November 2, 2025")
    - date_2: the later of the two consecutive performance dates, exactly as shown in the answer (e.g., "November 3, 2025")
    - seating_capacity: the capacity information as written (e.g., "17,500", "about 21,000", "20,000–21,000", "up to 23,500", "17k–17.5k")
    - reference_urls: an array of all explicit URLs provided in the answer that confirm the tour information (venue and dates) for that specific venue.
                      URLs can be plain or in markdown. Extract only valid URLs explicitly present in the answer.

    Only extract information for a venue if the answer indicates there are two consecutive nights at the same location.
    If a particular venue is not present in the answer, set that venue entry to null.
    If any individual field is missing for a present venue, set that field to null.
    Return a JSON object with keys: state_farm_arena, bell_centre, united_center, each mapping to a VenueDetails object or null.
    """


# --------------------------------------------------------------------------- #
# Utilities: capacity parsing & checking                                      #
# --------------------------------------------------------------------------- #
def _extract_k_numbers(text: str) -> Tuple[List[float], str]:
    """Extract numbers like '17k', '17.5k' as thousands and remove them from text."""
    nums: List[float] = []
    def repl(m: re.Match) -> str:
        val = float(m.group(1))
        nums.append(val * 1000.0)
        return " "  # remove token from text
    new_text = re.sub(r'(\d+(?:\.\d+)?)\s*[kK]\b', repl, text)
    return nums, new_text


def _extract_plain_numbers(text: str) -> List[float]:
    """Extract plain integers with optional thousands separators and decimals."""
    results: List[float] = []
    for m in re.finditer(r'(\d{1,3}(?:,\d{3})+|\d{4,6})(?:\.\d+)?', text):
        token = m.group(0)
        token = token.replace(",", "")
        try:
            results.append(float(token))
        except Exception:
            continue
    # Also catch simple 2-3 digit numbers occasionally used with denominators like '21k' already handled
    return results


def parse_capacity_range(capacity_text: Optional[str]) -> Optional[Tuple[float, float]]:
    """
    Parse a seating capacity description into an approximate (min, max) tuple.
    - Accepts forms like '17,000–17,500', '17k–17.5k', 'about 21,000', 'up to 23,500', '20,000-21,000', '23500'.
    - Returns None if parsing fails.
    """
    if not capacity_text or not isinstance(capacity_text, str):
        return None

    text = capacity_text.strip()
    # Normalize dashes
    text_norm = text.replace("–", "-").replace("—", "-").lower()

    k_nums, remainder = _extract_k_numbers(text_norm)
    nums = k_nums + _extract_plain_numbers(remainder)

    if not nums:
        return None

    # If phrase 'up to' present and only one number, interpret as (0, n]
    if "up to" in text_norm and len(nums) == 1:
        n = nums[0]
        return (0.0, n)

    # If there's an explicit range token like a-b, prefer interpreting as a range
    # Otherwise, use min-max of all numbers found
    if "-" in text_norm and len(nums) >= 2:
        nums_sorted = sorted(nums)
        return (nums_sorted[0], nums_sorted[-1])

    # Single number: treat as a narrow range +- 0 (we will apply tolerance later)
    if len(nums) == 1:
        n = nums[0]
        return (n, n)

    # Multiple numbers but no dash: use min-max
    return (min(nums), max(nums))


def ranges_intersect(a: Tuple[float, float], b: Tuple[float, float]) -> bool:
    """Check if two numeric ranges intersect or touch."""
    return not (a[1] < b[0] or b[1] < a[0])


def capacity_matches_state_farm(cap_str: Optional[str]) -> bool:
    """
    Expected approximately 17,000–17,500 (allowing modest tolerance).
    We'll accept overlap with [16,700, 17,800] to cover modest approximations.
    """
    rng = parse_capacity_range(cap_str)
    if not rng:
        return False
    expected = (16700.0, 17800.0)
    return ranges_intersect(rng, expected)


def capacity_matches_bell_centre(cap_str: Optional[str]) -> bool:
    """
    Expected approximately 21,000.
    We'll accept any overlap with [20,000, 22,000] to allow approx phrasing.
    """
    rng = parse_capacity_range(cap_str)
    if not rng:
        return False
    expected = (20000.0, 22000.0)
    return ranges_intersect(rng, expected)


def capacity_matches_united_center(cap_str: Optional[str]) -> bool:
    """
    Expected consistent with approx 20,000–21,000 (sports) and/or up to 23,500 (concerts).
    We'll accept overlap with union of:
      - [19,500, 21,500]  (sports approx)
      - [22,800, 23,700]  (concerts 'up to ~23,500' with modest tolerance)
    If either intersects, accept.
    """
    rng = parse_capacity_range(cap_str)
    if not rng:
        return False
    sports = (19500.0, 21500.0)
    concerts = (22800.0, 23700.0)
    return ranges_intersect(rng, sports) or ranges_intersect(rng, concerts)


# --------------------------------------------------------------------------- #
# Verification helpers                                                        #
# --------------------------------------------------------------------------- #
@dataclass
class VenueExpectation:
    entry_id: str
    display_name: str  # For claims
    expected_name: str
    address_expected: str
    city_expected: str
    state_options: List[str]
    date1_expected: str
    date2_expected: str
    capacity_checker: Any  # Callable[[Optional[str]], bool]


async def verify_venue_entry(
    evaluator: Evaluator,
    parent_node,
    details: Optional[VenueDetails],
    exp: VenueExpectation,
    location_context_for_claim: str,
) -> None:
    """
    Build the verification subtree for one venue entry.
    All child nodes are critical under the entry node (which itself is non-critical).
    """
    # Parent node for this entry (parallel aggregation)
    entry_node = evaluator.add_parallel(
        id=exp.entry_id,
        desc=f"Provide the required details for the {exp.display_name} consecutive-night stop.",
        parent=parent_node,
        critical=False,
    )

    # Extracted fields (may be None)
    venue_name = details.venue_name if details else None
    street_address = details.street_address if details else None
    city = details.city if details else None
    state_province = details.state_province if details else None
    date_1 = details.date_1 if details else None
    date_2 = details.date_2 if details else None
    capacity = details.seating_capacity if details else None
    urls = details.reference_urls if details else []

    # Venue Name
    node_name = evaluator.add_leaf(
        id=f"{exp.entry_id}_Venue_Name_Matches",
        desc=f"Official venue name is provided and is '{exp.expected_name}'.",
        parent=entry_node,
        critical=True,
    )
    claim = (
        f"The provided official venue name is '{venue_name}'. It matches the expected '{exp.expected_name}'. "
        f"Allow minor variants like adding the city or qualifiers, but the core venue name should be the same."
    )
    await evaluator.verify(
        claim=claim,
        node=node_name,
        additional_instruction="Treat small formatting differences as equivalent (e.g., casing, a trailing city qualifier)."
    )

    # Street Address
    node_addr = evaluator.add_leaf(
        id=f"{exp.entry_id}_Street_Address_Matches",
        desc=f"Complete street address is provided and is '{exp.address_expected}'.",
        parent=entry_node,
        critical=True,
    )
    claim = (
        f"The street address for {exp.display_name} is '{street_address}' and it matches '{exp.address_expected}'. "
        f"Accept minor variants or abbreviations such as 'Drive' vs 'Dr', 'Street' vs 'St', 'Avenue' vs 'Ave/Av', "
        f"and allow presence or absence of directionals like 'W'/'West' or punctuation."
    )
    await evaluator.verify(
        claim=claim,
        node=node_addr,
        additional_instruction="Judge if the provided street-line is equivalent to the expected one, ignoring city/state/postal code."
    )

    # City
    node_city = evaluator.add_leaf(
        id=f"{exp.entry_id}_City_Matches",
        desc=f"City is provided and is '{exp.city_expected}'.",
        parent=entry_node,
        critical=True,
    )
    claim = (
        f"The city for {exp.display_name} is '{city}' and it matches '{exp.city_expected}'. "
        f"Treat 'Montréal' and 'Montreal' as equivalent spellings if applicable."
    )
    await evaluator.verify(
        claim=claim,
        node=node_city,
        additional_instruction="Allow accent/non-accent variants (e.g., Montréal vs Montreal)."
    )

    # State/Province
    node_state = evaluator.add_leaf(
        id=f"{exp.entry_id}_State_Province_Matches",
        desc=f"State/province is provided and is '{' or '.join(exp.state_options)}'.",
        parent=entry_node,
        critical=True,
    )
    claim = (
        f"The state or province provided is '{state_province}'. It matches one of the acceptable forms "
        f"{exp.state_options}."
    )
    await evaluator.verify(
        claim=claim,
        node=node_state,
        additional_instruction="Consider full names and standard postal abbreviations as equivalent (e.g., Georgia/GA, Quebec/QC, Illinois/IL)."
    )

    # Date 1
    node_d1 = evaluator.add_leaf(
        id=f"{exp.entry_id}_Performance_Date_1_Matches",
        desc=f"First performance date is provided and is '{exp.date1_expected}'.",
        parent=entry_node,
        critical=True,
    )
    claim = (
        f"The first (earlier) performance date for {exp.display_name} is '{date_1}', and it matches '{exp.date1_expected}'. "
        f"Accept equivalent date formats such as 'Nov 2, 2025', '2025-11-02', or '2 November 2025'."
    )
    await evaluator.verify(
        claim=claim,
        node=node_d1,
        additional_instruction="Verify equivalence allowing common date formats; focus on the same calendar date."
    )

    # Date 2
    node_d2 = evaluator.add_leaf(
        id=f"{exp.entry_id}_Performance_Date_2_Matches",
        desc=f"Second performance date is provided and is '{exp.date2_expected}'.",
        parent=entry_node,
        critical=True,
    )
    claim = (
        f"The second (later) performance date for {exp.display_name} is '{date_2}', and it matches '{exp.date2_expected}'. "
        f"Accept equivalent date formats such as 'Nov 3, 2025', '2025-11-03', or '3 November 2025'."
    )
    await evaluator.verify(
        claim=claim,
        node=node_d2,
        additional_instruction="Verify equivalence allowing common date formats; focus on the same calendar date."
    )

    # Seating Capacity (custom check using numeric logic)
    cap_ok = exp.capacity_checker(capacity) if capacity is not None else False
    evaluator.add_custom_node(
        result=cap_ok,
        id=f"{exp.entry_id}_Seating_Capacity_Consistent",
        desc="Seating capacity is provided and matches the expected approximate range for this venue.",
        parent=entry_node,
        critical=True,
    )

    # Reference URL Provided (and supports the venue+both dates) - If no URL, fail directly
    if urls:
        node_ref = evaluator.add_leaf(
            id=f"{exp.entry_id}_Reference_URL_Provided",
            desc=f"A reference URL is provided that confirms this tour stop information for {exp.display_name} (venue and dates).",
            parent=entry_node,
            critical=True,
        )
        # Build a strong claim requiring both dates and the specific venue
        claim = (
            f"Paul McCartney will perform at {location_context_for_claim} on {exp.date1_expected} and {exp.date2_expected}. "
            f"The webpage must explicitly confirm both dates at this venue."
        )
        await evaluator.verify(
            claim=claim,
            node=node_ref,
            sources=urls,
            additional_instruction=(
                "The page should clearly show Paul McCartney's event listings or tour schedule indicating two consecutive nights "
                "at the specified venue on the exact dates. Accept official tour site, venue event pages, Ticketmaster/Event listings, "
                "or reputable media announcements. The URL is considered supportive only if BOTH dates at this venue are confirmed."
            )
        )
    else:
        evaluator.add_custom_node(
            result=False,
            id=f"{exp.entry_id}_Reference_URL_Provided",
            desc=f"A reference URL is provided that confirms this tour stop information for {exp.display_name} (venue and dates).",
            parent=entry_node,
            critical=True,
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
    Evaluate an answer for the 'Got Back 2025 consecutive-night venues' task and return a structured result.
    """
    # Initialize evaluator (root is non-critical parallel aggregation)
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

    # Extract structured venue details mentioned in the answer
    extraction: TourVenuesExtraction = await evaluator.extract(
        prompt=prompt_extract_consecutive_venues(),
        template_class=TourVenuesExtraction,
        extraction_name="venues_extraction",
    )

    # Add ground-truth expectations (for transparency/debugging; not used for scoring directly)
    evaluator.add_ground_truth({
        "expected_entries": {
            "State Farm Arena": {
                "address": "1 State Farm Drive",
                "city": "Atlanta",
                "state_or_province": ["Georgia", "GA"],
                "dates": ["November 2, 2025", "November 3, 2025"],
                "capacity": "approximately 17,000–17,500"
            },
            "Bell Centre": {
                "address": "1909 Av. des Canadiens-de-Montréal",
                "city": "Montreal",
                "state_or_province": ["Quebec", "QC"],
                "dates": ["November 17, 2025", "November 18, 2025"],
                "capacity": "approximately 21,000"
            },
            "United Center": {
                "address": "1901 W Madison Street",
                "city": "Chicago",
                "state_or_province": ["Illinois", "IL"],
                "dates": ["November 24, 2025", "November 25, 2025"],
                "capacity": "approximately 20,000–21,000 (sports) and/or up to ~23,500 (concerts)"
            }
        }
    }, gt_type="ground_truth_expectations")

    # Build expectations
    exp_state_farm = VenueExpectation(
        entry_id="State_Farm_Arena_Entry",
        display_name="State Farm Arena consecutive-night stop",
        expected_name="State Farm Arena",
        address_expected="1 State Farm Drive",
        city_expected="Atlanta",
        state_options=["Georgia", "GA"],
        date1_expected="November 2, 2025",
        date2_expected="November 3, 2025",
        capacity_checker=capacity_matches_state_farm
    )
    exp_bell = VenueExpectation(
        entry_id="Bell_Centre_Entry",
        display_name="Bell Centre consecutive-night stop",
        expected_name="Bell Centre",
        address_expected="1909 Av. des Canadiens-de-Montréal",
        city_expected="Montreal",
        state_options=["Quebec", "QC"],
        date1_expected="November 17, 2025",
        date2_expected="November 18, 2025",
        capacity_checker=capacity_matches_bell_centre
    )
    exp_united = VenueExpectation(
        entry_id="United_Center_Entry",
        display_name="United Center consecutive-night stop",
        expected_name="United Center",
        address_expected="1901 W Madison Street",
        city_expected="Chicago",
        state_options=["Illinois", "IL"],
        date1_expected="November 24, 2025",
        date2_expected="November 25, 2025",
        capacity_checker=capacity_matches_united_center
    )

    # Verify each venue entry subtree
    await verify_venue_entry(
        evaluator=evaluator,
        parent_node=root,
        details=extraction.state_farm_arena,
        exp=exp_state_farm,
        location_context_for_claim="State Farm Arena (Atlanta, Georgia)"
    )

    await verify_venue_entry(
        evaluator=evaluator,
        parent_node=root,
        details=extraction.bell_centre,
        exp=exp_bell,
        location_context_for_claim="Bell Centre (Montreal, Quebec)"
    )

    await verify_venue_entry(
        evaluator=evaluator,
        parent_node=root,
        details=extraction.united_center,
        exp=exp_united,
        location_context_for_claim="United Center (Chicago, Illinois)"
    )

    # Return structured summary
    return evaluator.get_summary()