import asyncio
import logging
from typing import Optional, List, Dict, Any, Tuple

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "springsteen_2026_venues"
TASK_DESCRIPTION = """
Bruce Springsteen and The E Street Band are conducting their "Land of Hope and Dreams" American Tour in spring 2026, running from late March through late May. For this tour, identify four specific venues with the following characteristics:

1. The venue that hosted the tour's opening show on March 31, 2026, located in Minneapolis, MN
2. The venue that hosted the show on April 3, 2026, located in Portland, OR
3. The venue that hosted two consecutive shows on April 7 and April 9, 2026, located in Inglewood, CA
4. The New York City venue that hosted two shows on May 11 and May 16, 2026

For each venue, provide:
- The venue name
- The city and state location
- The approximate concert seating capacity
- A reference URL confirming the venue information
"""

# Ground-truth expectations for identification checks
EXPECTED_VENUES = {
    "opening": {
        "expected_name": "Target Center",
        "city": "Minneapolis",
        "state": "MN",
        "capacity_range": (19000, 20500),
        "date_info": ["March 31, 2026"],
        "group_id": "Opening_Venue",
        "group_desc": "Identify the venue that hosted the tour's opening show on March 31, 2026 in Minneapolis, MN",
        "id_leaf_ident": "Opening_Venue_Identification",
        "id_leaf_ref": "Opening_Venue_Reference",
        "ident_desc": "The provided venue name is Target Center, the location is Minneapolis, MN, and the stated concert capacity is approximately 19,000-20,500",
        "ref_desc": "A valid URL reference is provided that confirms the venue information",
    },
    "portland": {
        "expected_name": "Moda Center",
        "city": "Portland",
        "state": "OR",
        "capacity_range": (19000, 20000),
        "date_info": ["April 3, 2026"],
        "group_id": "Portland_Venue",
        "group_desc": "Identify the venue that hosted the show on April 3, 2026 in Portland, OR",
        "id_leaf_ident": "Portland_Venue_Identification",
        "id_leaf_ref": "Portland_Venue_Reference",
        "ident_desc": "The provided venue name is Moda Center, the location is Portland, OR, and the stated concert capacity is approximately 19,000-20,000",
        "ref_desc": "A valid URL reference is provided that confirms the venue information",
    },
    "inglewood": {
        "expected_name": "Kia Forum",
        "city": "Inglewood",
        "state": "CA",
        "capacity_range": (16500, 18500),  # approx 17,500
        "date_info": ["April 7, 2026", "April 9, 2026"],
        "group_id": "Inglewood_Venue",
        "group_desc": "Identify the venue that hosted two consecutive shows on April 7 and April 9, 2026 in Inglewood, CA",
        "id_leaf_ident": "Inglewood_Venue_Identification",
        "id_leaf_ref": "Inglewood_Venue_Reference",
        "ident_desc": "The provided venue name is Kia Forum, the location is Inglewood, CA, and the stated concert capacity is approximately 17,500",
        "ref_desc": "A valid URL reference is provided that confirms the venue information",
    },
    "nyc": {
        "expected_name": "Madison Square Garden",
        "city": "New York",
        "state": "NY",
        "capacity_range": (19500, 20000),
        "date_info": ["May 11, 2026", "May 16, 2026"],
        "group_id": "NYC_Venue_Two_Shows",
        "group_desc": "Identify the New York City venue that hosted two shows on May 11 and May 16, 2026",
        "id_leaf_ident": "NYC_Venue_Identification",
        "id_leaf_ref": "NYC_Venue_Reference",
        "ident_desc": "The provided venue name is Madison Square Garden, the location is New York, NY, and the stated concert capacity is approximately 19,500-20,000",
        "ref_desc": "A valid URL reference is provided that confirms the venue information",
    },
}


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class VenueInfo(BaseModel):
    name: Optional[str] = None
    city: Optional[str] = None
    state: Optional[str] = None
    capacity: Optional[str] = None  # Keep as string to allow ranges like "19,000–20,000"
    url: Optional[str] = None       # Single best supporting URL explicitly present in the answer


class TourVenuesExtraction(BaseModel):
    opening: Optional[VenueInfo] = None
    portland: Optional[VenueInfo] = None
    inglewood: Optional[VenueInfo] = None
    nyc: Optional[VenueInfo] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_tour_venues() -> str:
    return """
    Extract structured information for four specific venues from the provided answer text about Bruce Springsteen and The E Street Band's "Land of Hope and Dreams" American Tour (spring 2026). Map the information into these four slots:

    Fields to extract for each slot:
    - name: the venue name exactly as stated in the answer
    - city: the city name exactly as stated (e.g., "Minneapolis" or "New York")
    - state: the two-letter state abbreviation if available (e.g., "MN", "OR", "CA", "NY"); if spelled out, keep it as stated
    - capacity: the approximate concert seating capacity string as presented (e.g., "19,500", "about 20,000", "19,000–20,500")
    - url: ONE URL explicitly mentioned in the answer that best supports the venue information (prefer a page that confirms the show date(s) at that venue; if multiple URLs are given, pick the single best one)

    Map to these four keys exactly:
    - opening: the Minneapolis, MN show on March 31, 2026
    - portland: the Portland, OR show on April 3, 2026
    - inglewood: the Inglewood, CA shows on April 7 and April 9, 2026
    - nyc: the New York, NY shows on May 11 and May 16, 2026

    Important rules:
    - Extract ONLY what the answer explicitly states. Do not infer or add new information.
    - If any field for a slot is missing in the answer, set it to null.
    - For url, return null if no URL is explicitly present for that slot.
    - Keep capacities as free-form strings (numbers, ranges, "about ~", etc.) exactly as in the answer.
    """


# --------------------------------------------------------------------------- #
# Helper: build identification claim strings                                  #
# --------------------------------------------------------------------------- #
def build_identification_claim(
    slot_label: str,
    extracted: Optional[VenueInfo],
    expected_name: str,
    expected_city: str,
    expected_state: str,
    capacity_range: Tuple[int, int],
    ident_desc: str
) -> str:
    provided_name = extracted.name if extracted and extracted.name else "NULL"
    provided_city = extracted.city if extracted and extracted.city else "NULL"
    provided_state = extracted.state if extracted and extracted.state else "NULL"
    provided_capacity = extracted.capacity if extracted and extracted.capacity else "NULL"

    cap_min, cap_max = capacity_range
    return (
        f"For the {slot_label} slot, the answer provides venue '{provided_name}' in "
        f"'{provided_city}, {provided_state}' with stated capacity '{provided_capacity}'. "
        f"Judge whether this satisfies the requirement: venue should be '{expected_name}' in "
        f"'{expected_city}, {expected_state}', and the stated concert seating capacity should be "
        f"approximately within {cap_min} to {cap_max} seats. "
        f"Allow minor formatting differences (e.g., dashes, commas) and reasonable approximations. "
        f"If the answer omits any of name/city/state/capacity, consider this identification incorrect. "
        f"Requirement restated: {ident_desc}"
    )


def build_reference_claim(
    expected_name: str,
    expected_city: str,
    expected_state: str,
    date_info: List[str]
) -> str:
    # Build a readable date requirement
    if len(date_info) == 1:
        dates_phrase = f"on {date_info[0]}"
    else:
        dates_phrase = "on both of the specified dates: " + ", ".join(date_info)

    return (
        f"This webpage confirms that Bruce Springsteen and The E Street Band performed at "
        f"{expected_name} in {expected_city}, {expected_state} {dates_phrase}. "
        f"The page should clearly indicate the artist and the specified date(s) at the stated venue/location. "
        f"Minor naming variations are acceptable (e.g., 'Bruce Springsteen & The E Street Band', 'MSG' for Madison Square Garden, "
        f"'The Forum' for Kia Forum, etc.). If the page is irrelevant, inaccessible, or does not clearly support the event at the "
        f"stated venue on the specified date(s), judge it as not supported."
    )


# --------------------------------------------------------------------------- #
# Verification per-venue block                                                #
# --------------------------------------------------------------------------- #
async def verify_venue_block(
    evaluator: Evaluator,
    parent_node,
    group_id: str,
    group_desc: str,
    id_leaf_ident: str,
    ident_desc: str,
    id_leaf_ref: str,
    ref_desc: str,
    extracted: Optional[VenueInfo],
    expected_name: str,
    expected_city: str,
    expected_state: str,
    capacity_range: Tuple[int, int],
    date_info: List[str],
) -> None:
    # Create the group node (parallel, non-critical)
    group_node = evaluator.add_parallel(
        id=group_id,
        desc=group_desc,
        parent=parent_node,
        critical=False
    )

    # Identification leaf (critical)
    ident_leaf = evaluator.add_leaf(
        id=id_leaf_ident,
        desc=ident_desc,
        parent=group_node,
        critical=True
    )

    ident_claim = build_identification_claim(
        slot_label=group_id,
        extracted=extracted,
        expected_name=expected_name,
        expected_city=expected_city,
        expected_state=expected_state,
        capacity_range=capacity_range,
        ident_desc=ident_desc
    )
    await evaluator.verify(
        claim=ident_claim,
        node=ident_leaf,
        additional_instruction=(
            "You are checking whether the ANSWER's provided venue info matches the expected venue and capacity range. "
            "Be tolerant to minor variants (e.g., 'NY' vs 'New York, NY'), hyphens, commas, approximate words like 'about'. "
            "However, if name or city/state do not match the expected, or the capacity is clearly outside the expected range, judge incorrect."
        )
    )

    # Reference leaf (critical) with URL verification
    ref_leaf = evaluator.add_leaf(
        id=id_leaf_ref,
        desc=ref_desc,
        parent=group_node,
        critical=True
    )

    ref_claim = build_reference_claim(
        expected_name=expected_name,
        expected_city=expected_city,
        expected_state=expected_state,
        date_info=date_info
    )
    url = extracted.url if extracted else None

    await evaluator.verify(
        claim=ref_claim,
        node=ref_leaf,
        sources=url,  # if None or invalid, the verifier should fail
        additional_instruction=(
            "If no URL is provided, or the URL is invalid/irrelevant, judge as not supported. "
            "The page should indicate Bruce Springsteen (with the E Street Band) at the specified venue and date(s). "
            "Calendar/event list pages are acceptable if they clearly show the performer and the required date(s). "
            "Accept reasonable naming variants (e.g., 'Bruce Springsteen & The E Street Band', 'MSG' for Madison Square Garden, "
            "'The Forum' for Kia Forum)."
        )
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

    # Extract structured venue info from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_tour_venues(),
        template_class=TourVenuesExtraction,
        extraction_name="tour_venues_extraction"
    )

    # Optional: add ground truth info to the summary for transparency
    evaluator.add_ground_truth({
        "expected": {
            "opening": {
                "venue": EXPECTED_VENUES["opening"]["expected_name"],
                "city": EXPECTED_VENUES["opening"]["city"],
                "state": EXPECTED_VENUES["opening"]["state"],
                "capacity_range": EXPECTED_VENUES["opening"]["capacity_range"],
                "dates": EXPECTED_VENUES["opening"]["date_info"],
            },
            "portland": {
                "venue": EXPECTED_VENUES["portland"]["expected_name"],
                "city": EXPECTED_VENUES["portland"]["city"],
                "state": EXPECTED_VENUES["portland"]["state"],
                "capacity_range": EXPECTED_VENUES["portland"]["capacity_range"],
                "dates": EXPECTED_VENUES["portland"]["date_info"],
            },
            "inglewood": {
                "venue": EXPECTED_VENUES["inglewood"]["expected_name"],
                "city": EXPECTED_VENUES["inglewood"]["city"],
                "state": EXPECTED_VENUES["inglewood"]["state"],
                "capacity_range": EXPECTED_VENUES["inglewood"]["capacity_range"],
                "dates": EXPECTED_VENUES["inglewood"]["date_info"],
            },
            "nyc": {
                "venue": EXPECTED_VENUES["nyc"]["expected_name"],
                "city": EXPECTED_VENUES["nyc"]["city"],
                "state": EXPECTED_VENUES["nyc"]["state"],
                "capacity_range": EXPECTED_VENUES["nyc"]["capacity_range"],
                "dates": EXPECTED_VENUES["nyc"]["date_info"],
            }
        }
    })

    # Verify each venue block according to rubric
    await verify_venue_block(
        evaluator=evaluator,
        parent_node=root,
        group_id=EXPECTED_VENUES["opening"]["group_id"],
        group_desc=EXPECTED_VENUES["opening"]["group_desc"],
        id_leaf_ident=EXPECTED_VENUES["opening"]["id_leaf_ident"],
        ident_desc=EXPECTED_VENUES["opening"]["ident_desc"],
        id_leaf_ref=EXPECTED_VENUES["opening"]["id_leaf_ref"],
        ref_desc=EXPECTED_VENUES["opening"]["ref_desc"],
        extracted=extracted.opening if extracted else None,
        expected_name=EXPECTED_VENUES["opening"]["expected_name"],
        expected_city=EXPECTED_VENUES["opening"]["city"],
        expected_state=EXPECTED_VENUES["opening"]["state"],
        capacity_range=EXPECTED_VENUES["opening"]["capacity_range"],
        date_info=EXPECTED_VENUES["opening"]["date_info"]
    )

    await verify_venue_block(
        evaluator=evaluator,
        parent_node=root,
        group_id=EXPECTED_VENUES["portland"]["group_id"],
        group_desc=EXPECTED_VENUES["portland"]["group_desc"],
        id_leaf_ident=EXPECTED_VENUES["portland"]["id_leaf_ident"],
        ident_desc=EXPECTED_VENUES["portland"]["ident_desc"],
        id_leaf_ref=EXPECTED_VENUES["portland"]["id_leaf_ref"],
        ref_desc=EXPECTED_VENUES["portland"]["ref_desc"],
        extracted=extracted.portland if extracted else None,
        expected_name=EXPECTED_VENUES["portland"]["expected_name"],
        expected_city=EXPECTED_VENUES["portland"]["city"],
        expected_state=EXPECTED_VENUES["portland"]["state"],
        capacity_range=EXPECTED_VENUES["portland"]["capacity_range"],
        date_info=EXPECTED_VENUES["portland"]["date_info"]
    )

    await verify_venue_block(
        evaluator=evaluator,
        parent_node=root,
        group_id=EXPECTED_VENUES["inglewood"]["group_id"],
        group_desc=EXPECTED_VENUES["inglewood"]["group_desc"],
        id_leaf_ident=EXPECTED_VENUES["inglewood"]["id_leaf_ident"],
        ident_desc=EXPECTED_VENUES["inglewood"]["ident_desc"],
        id_leaf_ref=EXPECTED_VENUES["inglewood"]["id_leaf_ref"],
        ref_desc=EXPECTED_VENUES["inglewood"]["ref_desc"],
        extracted=extracted.inglewood if extracted else None,
        expected_name=EXPECTED_VENUES["inglewood"]["expected_name"],
        expected_city=EXPECTED_VENUES["inglewood"]["city"],
        expected_state=EXPECTED_VENUES["inglewood"]["state"],
        capacity_range=EXPECTED_VENUES["inglewood"]["capacity_range"],
        date_info=EXPECTED_VENUES["inglewood"]["date_info"]
    )

    await verify_venue_block(
        evaluator=evaluator,
        parent_node=root,
        group_id=EXPECTED_VENUES["nyc"]["group_id"],
        group_desc=EXPECTED_VENUES["nyc"]["group_desc"],
        id_leaf_ident=EXPECTED_VENUES["nyc"]["id_leaf_ident"],
        ident_desc=EXPECTED_VENUES["nyc"]["ident_desc"],
        id_leaf_ref=EXPECTED_VENUES["nyc"]["id_leaf_ref"],
        ref_desc=EXPECTED_VENUES["nyc"]["ref_desc"],
        extracted=extracted.nyc if extracted else None,
        expected_name=EXPECTED_VENUES["nyc"]["expected_name"],
        expected_city=EXPECTED_VENUES["nyc"]["city"],
        expected_state=EXPECTED_VENUES["nyc"]["state"],
        capacity_range=EXPECTED_VENUES["nyc"]["capacity_range"],
        date_info=EXPECTED_VENUES["nyc"]["date_info"]
    )

    return evaluator.get_summary()