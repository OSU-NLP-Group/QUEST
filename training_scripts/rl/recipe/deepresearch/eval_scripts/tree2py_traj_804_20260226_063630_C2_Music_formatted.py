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
TASK_ID = "grammy_capacity_cowboy_carter_2025"
TASK_DESCRIPTION = (
    "In February 2025, Beyoncé won the Grammy Award for Album of the Year for her album 'Cowboy Carter'. "
    "Determine whether the concert seating capacity of the venue where this ceremony took place meets or exceeds "
    "40 times the minimum seating requirement for a Broadway theatre (which is 500 seats). Additionally, name one producer "
    "who was credited on at least 5 different tracks of the 'Cowboy Carter' album."
)

BROADWAY_MINIMUM_SEATS = 500
MULTIPLIER = 40
THRESHOLD_SEATS = BROADWAY_MINIMUM_SEATS * MULTIPLIER  # 20,000

# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class EventInfo(BaseModel):
    event_name: Optional[str] = None  # e.g., "67th Annual Grammy Awards"
    event_date: Optional[str] = None  # e.g., "February 2, 2025"
    album_title: Optional[str] = None  # e.g., "Cowboy Carter"
    ceremony_sources: List[str] = Field(default_factory=list)  # URLs cited for event/winner/venue details


class VenueInfo(BaseModel):
    venue_name: Optional[str] = None  # e.g., "Crypto.com Arena"
    venue_location: Optional[str] = None  # e.g., "Los Angeles, California"
    capacity_value: Optional[str] = None  # e.g., "20,000", "about 20,000", "approximately 20k"
    comparison_result: Optional[str] = None  # e.g., "meets", "exceeds", "does not meet", "yes", "no"
    venue_sources: List[str] = Field(default_factory=list)  # URLs cited for venue/capacity


class ProducerInfo(BaseModel):
    producer_name: Optional[str] = None  # A producer said to have >=5 track credits
    producer_sources: List[str] = Field(default_factory=list)  # URLs cited for credits


class MainExtraction(BaseModel):
    event: Optional[EventInfo] = None
    venue: Optional[VenueInfo] = None
    producer: Optional[ProducerInfo] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_main() -> str:
    return """
Extract the following information exactly as stated in the answer text and the URLs the answer explicitly provides.

1) Event (the Grammy ceremony where Beyoncé won AOTY for 'Cowboy Carter'):
   - event_name: the ceremony name as written (e.g., "67th Annual Grammy Awards").
   - event_date: the date as written (e.g., "February 2, 2025").
   - album_title: the album title associated with Beyoncé's Album of the Year win (should be "Cowboy Carter" if claimed).
   - ceremony_sources: all URLs the answer cites for the event/winner/ceremony (extract actual URLs only).

2) Venue and capacity comparison:
   - venue_name: the venue where the ceremony took place (e.g., "Crypto.com Arena").
   - venue_location: the city/location text (e.g., "Los Angeles, California").
   - capacity_value: the concert seating capacity value the answer used or cited (string as presented).
   - comparison_result: the answer's conclusion about whether the concert capacity meets or exceeds 20,000 (40 × 500). Use a simple label:
       - "meets" or "exceeds" or "yes" if they conclude it meets or exceeds,
       - "does not meet" or "no" if they conclude it does not meet,
       - null if not explicitly stated.
   - venue_sources: all URLs the answer cites specifically for the venue or capacity.

3) Producer with >=5 track credits on 'Cowboy Carter':
   - producer_name: a producer named by the answer that it claims has at least 5 different track credits on 'Cowboy Carter'.
   - producer_sources: all URLs the answer cites to support this producer credit count (e.g., official credits pages, reputable discography credits).

Rules:
- Extract only what the answer states and only the URLs explicitly present in the answer (plain URLs or markdown links).
- Do not invent URLs.
- If a field is missing in the answer, return null for that field or an empty list for URLs.
- Preserve the original wording/spelling of names and titles.
"""


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def normalize_yes_no(s: Optional[str]) -> Optional[bool]:
    if not s:
        return None
    t = s.strip().lower()
    # Handle common variants
    if any(k in t for k in ["does not", "not meet", "no", "doesn't"]):
        return False
    if any(k in t for k in ["meet", "meets", "exceed", "exceeds", ">= ", "≥", "yes"]):
        return True
    return None


def safe_venue_name(extracted: MainExtraction) -> str:
    if extracted and extracted.venue and extracted.venue.venue_name:
        return extracted.venue.venue_name
    # Fallback to the expected venue to avoid empty claims
    return "Crypto.com Arena"


def safe_capacity_value(extracted: MainExtraction) -> Optional[str]:
    if extracted and extracted.venue and extracted.venue.capacity_value:
        return extracted.venue.capacity_value
    return None


def get_event_sources(extracted: MainExtraction) -> List[str]:
    if extracted and extracted.event and extracted.event.ceremony_sources:
        return extracted.event.ceremony_sources
    return []


def get_venue_sources(extracted: MainExtraction) -> List[str]:
    if extracted and extracted.venue and extracted.venue.venue_sources:
        return extracted.venue.venue_sources
    return []


def get_producer_sources(extracted: MainExtraction) -> List[str]:
    if extracted and extracted.producer and extracted.producer.producer_sources:
        return extracted.producer.producer_sources
    return []


# --------------------------------------------------------------------------- #
# Verification subroutines                                                    #
# --------------------------------------------------------------------------- #
async def build_capacity_comparison_subtree(
    evaluator: Evaluator,
    parent_node,
    extracted: MainExtraction,
) -> None:
    """
    Build and verify the 'Capacity Comparison' subtree:
    - Venue Identification (parallel, critical)
        * Grammy Ceremony (leaf)
        * Venue (leaf)
    - Comparison Result (leaf)
    """
    # Capacity Comparison (sequential, critical)
    capacity_node = evaluator.add_sequential(
        id="capacity_comparison",
        desc="The answer correctly determines whether the venue's concert capacity meets or exceeds 40 times the Broadway theatre minimum (500 seats).",
        parent=parent_node,
        critical=True,
    )

    # Venue Identification (parallel, critical)
    venue_ident_node = evaluator.add_parallel(
        id="venue_identification",
        desc="The Grammy ceremony and venue are correctly identified.",
        parent=capacity_node,
        critical=True,
    )

    # Leaf: Grammy Ceremony identification
    grammy_leaf = evaluator.add_leaf(
        id="grammy_ceremony",
        desc="The 67th Annual Grammy Awards ceremony on February 2, 2025, is identified as the event where Beyoncé won Album of the Year for 'Cowboy Carter'.",
        parent=venue_ident_node,
        critical=True,
    )
    ceremony_sources = get_event_sources(extracted)
    grammy_claim = (
        "The 67th Annual Grammy Awards ceremony took place on February 2, 2025, and at that event Beyoncé won "
        "the Grammy Award for Album of the Year for 'Cowboy Carter'."
    )
    await evaluator.verify(
        claim=grammy_claim,
        node=grammy_leaf,
        sources=ceremony_sources,
        additional_instruction=(
            "Verify that the source explicitly supports: (1) the ceremony is the 67th Annual Grammy Awards, "
            "(2) the date is February 2, 2025 (allow reasonable date formatting variants), and "
            "(3) Beyoncé won Album of the Year for 'Cowboy Carter' at that ceremony. "
            "Accept minor name/formatting variants."
        ),
    )

    # Leaf: Venue identification
    venue_leaf = evaluator.add_leaf(
        id="venue_identification_venue",
        desc="Crypto.com Arena in Los Angeles is identified as the venue where the ceremony took place.",
        parent=venue_ident_node,
        critical=True,
    )
    venue_claim = (
        "The 67th Annual Grammy Awards ceremony took place at Crypto.com Arena in Los Angeles, California."
    )
    await evaluator.verify(
        claim=venue_claim,
        node=venue_leaf,
        sources=ceremony_sources,
        additional_instruction=(
            "Confirm that the page indicates the ceremony venue was Crypto.com Arena in Los Angeles "
            "(also acceptable if listed as 'Downtown Los Angeles' or references former name 'Staples Center')."
        ),
    )

    # Leaf: Comparison Result
    comparison_leaf = evaluator.add_leaf(
        id="comparison_result",
        desc=(
            "The answer correctly states whether the venue's concert capacity meets or exceeds 40 times the Broadway "
            "minimum of 500 seats (i.e., 20,000 seats). The venue's actual concert capacity is approximately 20,000 seats."
        ),
        parent=capacity_node,
        critical=True,
    )

    # Compose claim using extracted info when available
    venue_name = safe_venue_name(extracted)
    capacity_str = safe_capacity_value(extracted)
    user_meets = normalize_yes_no(extracted.venue.comparison_result) if (extracted and extracted.venue) else None
    # Default to asserting the true-world check (meets/exceeds) if user did not explicitly state
    meets_phrase = "meets or exceeds"
    if user_meets is False:
        meets_phrase = "does not meet"
    elif user_meets is True:
        meets_phrase = "meets or exceeds"

    if capacity_str:
        comp_claim = (
            f"The concert seating capacity of {venue_name} is approximately {capacity_str} seats, "
            f"which {meets_phrase} {THRESHOLD_SEATS} seats (40 × {BROADWAY_MINIMUM_SEATS})."
        )
    else:
        comp_claim = (
            f"The concert seating capacity of {venue_name} {meets_phrase} {THRESHOLD_SEATS} seats "
            f"(40 × {BROADWAY_MINIMUM_SEATS})."
        )

    await evaluator.verify(
        claim=comp_claim,
        node=comparison_leaf,
        sources=get_venue_sources(extracted),
        additional_instruction=(
            "First, confirm the concert seating capacity figure for the venue from the provided source(s). "
            "If multiple capacities are listed (e.g., basketball/hockey/concert), use the 'concert' capacity. "
            f"Then, determine whether that capacity is >= {THRESHOLD_SEATS} (40 × {BROADWAY_MINIMUM_SEATS}). "
            "Allow approximate wording (e.g., 'about 20,000'). If the page provides a range, take the relevant "
            "concert figure for this comparison. The final judgment should reflect whether the capacity meets or "
            "exceeds the threshold."
        ),
    )


async def build_producer_leaf(
    evaluator: Evaluator,
    parent_node,
    extracted: MainExtraction,
) -> None:
    """
    Build and verify the 'Producer Identification' leaf.
    """
    producer_leaf = evaluator.add_leaf(
        id="producer_identification",
        desc=(
            "A producer who was credited on at least 5 different tracks of the 'Cowboy Carter' album is correctly named. "
            "Valid answers include: Beyoncé (all 27 tracks), The-Dream (5 tracks), Dave Hamelin (5 tracks), or Khirye Tyler (multiple tracks)."
        ),
        parent=parent_node,
        critical=True,
    )

    producer_name = extracted.producer.producer_name if (extracted and extracted.producer) else None
    name_for_claim = producer_name if producer_name else ""
    producer_claim = (
        f"The producer named '{name_for_claim}' is credited on at least 5 different tracks on Beyoncé's album 'Cowboy Carter'."
    )

    await evaluator.verify(
        claim=producer_claim,
        node=producer_leaf,
        sources=get_producer_sources(extracted),
        additional_instruction=(
            "Verify from the provided source(s) that the specified individual has production credits "
            "on at least 5 distinct tracks on the 'Cowboy Carter' album. Count track-level production roles "
            "such as producer/co-producer/additional producer as indicated by the credits page. "
            "If the producer is credited on all tracks, that trivially satisfies the condition."
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
    Evaluate an answer for the Grammy venue capacity comparison and producer identification task.
    """
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # "Complete Answer" aggregates two parallel criteria
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
        prompt=prompt_extract_main(),
        template_class=MainExtraction,
        extraction_name="extracted_answer_fields",
    )

    # Add helpful ground-truth context (non-scoring) for logs
    evaluator.add_ground_truth(
        {
            "threshold_seats": THRESHOLD_SEATS,
            "broadway_minimum": BROADWAY_MINIMUM_SEATS,
            "multiplier": MULTIPLIER,
            "expected_event_name": "67th Annual Grammy Awards",
            "expected_venue": "Crypto.com Arena, Los Angeles",
            "album": "Cowboy Carter",
        },
        gt_type="reference_info",
    )

    # Build and verify Capacity Comparison subtree
    await build_capacity_comparison_subtree(evaluator, root, extracted)

    # Build and verify Producer Identification leaf
    await build_producer_leaf(evaluator, root, extracted)

    # Return evaluation summary
    return evaluator.get_summary()