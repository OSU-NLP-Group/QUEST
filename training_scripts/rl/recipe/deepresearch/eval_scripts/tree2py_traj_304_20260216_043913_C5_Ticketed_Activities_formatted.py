import asyncio
import logging
import math
import re
from typing import Any, List, Optional, Dict

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "concert_venues_2026"
TASK_DESCRIPTION = (
    "During 2026, two major artists—David Byrne and Renee Rapp—are conducting concert tours across different continents. "
    "Your task is to identify three specific venues from their tour schedules based on the following criteria:\n\n"
    "Venue 1: Identify the smallest capacity venue where David Byrne performs during his North American tour dates in April and May 2026. "
    "Provide venue name and city, the exact seating capacity, the minimum number of wheelchair accessible seats required (1% rounded up), "
    "and a reference URL confirming the venue, tour date, and capacity.\n\n"
    "Venue 2: Identify the largest capacity venue where Renee Rapp performs during her European BITE ME tour in March 2026. "
    "Provide venue name and city, the exact seating capacity, the minimum number of wheelchair accessible seats required (1% rounded up), "
    "and a reference URL confirming the venue, tour date, and capacity.\n\n"
    "Venue 3: Identify one venue where both David Byrne and Renee Rapp perform during their respective 2026 tours (different dates). "
    "Provide venue name and city and reference URLs confirming both artists perform at this venue during 2026. "
    "All information must be verifiable via official tour announcements, venue websites, or reputable ticketing platforms."
)


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class ComparableVenue(BaseModel):
    """A competitor venue used to justify smallest/largest capacity claims."""
    venue_name: Optional[str] = None
    city: Optional[str] = None
    capacity: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class VenuePick(BaseModel):
    """A single venue pick with details and sources."""
    venue_name: Optional[str] = None
    city: Optional[str] = None
    capacity: Optional[str] = None
    wheelchair_min_provided: Optional[str] = None
    performance_date: Optional[str] = None

    # General sources for this venue (schedule, tickets, venue page, etc.)
    sources: List[str] = Field(default_factory=list)

    # Capacity-specific sources (if separately provided)
    capacity_sources: List[str] = Field(default_factory=list)

    # Artist/tour/date-specific sources (if separately provided)
    artist_tour_sources: List[str] = Field(default_factory=list)


class TourVenuesExtraction(BaseModel):
    """Full extraction for the three venues required by the task."""
    # Venue 1: David Byrne smallest capacity NA April-May 2026
    venue1: Optional[VenuePick] = None
    venue1_competitors: List[ComparableVenue] = Field(default_factory=list)

    # Venue 2: Renee Rapp largest capacity EU March 2026
    venue2: Optional[VenuePick] = None
    venue2_competitors: List[ComparableVenue] = Field(default_factory=list)

    # Venue 3: Shared venue both artists perform (2026, different dates)
    venue3_name: Optional[str] = None
    venue3_city: Optional[str] = None
    # Optionally included general sources for the venue
    venue3_sources: List[str] = Field(default_factory=list)
    # Distinct sources per artist
    venue3_david_sources: List[str] = Field(default_factory=list)
    venue3_david_date: Optional[str] = None
    venue3_renee_sources: List[str] = Field(default_factory=list)
    venue3_renee_date: Optional[str] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_venues() -> str:
    return """
    Extract structured information from the answer for three venues as specified.

    GENERAL RULES:
    - Extract exactly what is present in the answer. Do not invent any missing details.
    - For any required field not explicitly provided, return null.
    - For URLs, extract actual URLs only (plain or markdown). If a URL is missing or malformed, omit it.

    VENUE 1 (David Byrne — smallest capacity in North America during April–May 2026):
    Return object `venue1` with:
      - venue_name: The venue name
      - city: The venue city
      - capacity: The exact seating capacity provided in the answer (string as written)
      - wheelchair_min_provided: The minimum wheelchair accessible seats number, if the answer provided it (string as written)
      - performance_date: The specific date mentioned for this venue (if provided), else null
      - sources: All URLs that generally confirm the venue, artist, and tour date
      - capacity_sources: URLs specifically confirming capacity (if separately provided)
      - artist_tour_sources: URLs specifically confirming the artist/tour/date (if separately provided)
    Also return `venue1_competitors`: an array of other David Byrne North American April–May 2026 venues mentioned in the answer (if any), each with:
      - venue_name
      - city
      - capacity
      - sources (URLs for that competitor)

    VENUE 2 (Renee Rapp — largest capacity in Europe during March 2026 BITE ME tour):
    Return object `venue2` with the same fields as venue1, adapted for Renee Rapp.
    Also return `venue2_competitors`: an array of other Renee Rapp European March 2026 venues mentioned in the answer (if any), each with:
      - venue_name
      - city
      - capacity
      - sources

    VENUE 3 (Shared venue both artists perform in 2026, different dates):
    Return:
      - venue3_name: The shared venue name
      - venue3_city: The shared venue city
      - venue3_sources: Any general URLs for the venue (optional)
      - venue3_david_sources: URLs confirming David Byrne performs at this venue in 2026 (schedule, tickets, official)
      - venue3_david_date: If a specific date for David Byrne is provided, extract it; else null
      - venue3_renee_sources: URLs confirming Renee Rapp performs at this venue in 2026 (schedule, tickets, official)
      - venue3_renee_date: If a specific date for Renee Rapp is provided, extract it; else null
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _combine_sources(*lists: List[str]) -> List[str]:
    """Combine multiple source lists and deduplicate while preserving order."""
    seen = set()
    combined = []
    for lst in lists:
        for url in lst:
            if not url or not isinstance(url, str):
                continue
            u = url.strip()
            if u and u not in seen:
                seen.add(u)
                combined.append(u)
    return combined


def _parse_capacity_to_int(capacity_str: Optional[str]) -> Optional[int]:
    """Parse a human-written capacity string into an integer if possible."""
    if not capacity_str:
        return None
    s = capacity_str.strip().lower()

    # Replace commas and spaces
    s = s.replace(",", "").replace("~", "").replace("approx.", "").strip()

    # Handle 'k' suffix (e.g., '2k' or '2.5k')
    k_match = re.findall(r"(\d+(?:\.\d+)?)\s*k\b", s)
    if k_match:
        try:
            # Use the first occurrence
            val = float(k_match[0]) * 1000.0
            return int(round(val))
        except Exception:
            pass

    # Extract all integer numbers present
    nums = re.findall(r"\d{1,7}", s)
    if nums:
        try:
            # Use the largest number as a heuristic for capacity
            int_nums = [int(n) for n in nums]
            return max(int_nums) if int_nums else None
        except Exception:
            return None

    return None


def _compute_wheelchair_min(capacity_int: Optional[int]) -> Optional[int]:
    """Compute minimum accessible seats as ceil(1% of capacity)."""
    if capacity_int is None:
        return None
    return math.ceil(capacity_int * 0.01)


# --------------------------------------------------------------------------- #
# Verification subtrees                                                       #
# --------------------------------------------------------------------------- #
async def verify_venue_1(evaluator: Evaluator, parent_node, ext: TourVenuesExtraction) -> None:
    """Verify Venue 1: David Byrne smallest capacity NA Apr–May 2026."""
    v = ext.venue1 or VenuePick()

    # Group node
    v1_node = evaluator.add_parallel(
        id="venue_1_smallest_david_byrne_na",
        desc="The smallest capacity venue where David Byrne performs in North America during April–May 2026",
        parent=parent_node,
        critical=False,
    )

    # Venue identification (existence)
    ident_ok = bool(v.venue_name) and bool(v.city)
    evaluator.add_custom_node(
        result=ident_ok,
        id="venue_1_identification",
        desc="The venue name and city are provided",
        parent=v1_node,
        critical=True,
    )

    # Artist and tour confirmation
    artist_leaf = evaluator.add_leaf(
        id="venue_1_artist_and_tour",
        desc="The venue is confirmed to host David Byrne during his 2026 North American tour (April–May 2026)",
        parent=v1_node,
        critical=True,
    )
    artist_sources = _combine_sources(v.sources, v.artist_tour_sources)
    artist_claim = (
        f"David Byrne performs at {v.venue_name or ''} in {v.city or ''} during April or May 2026 "
        f"as part of his North American tour."
    )
    await evaluator.verify(
        claim=artist_claim,
        node=artist_leaf,
        sources=artist_sources,
        additional_instruction="Accept reputable schedule or ticket pages. The page should clearly indicate David Byrne, "
                              "the venue, and a date in April or May 2026. Minor naming variations are acceptable.",
    )

    # Capacity stated and verified
    capacity_leaf = evaluator.add_leaf(
        id="venue_1_capacity_stated",
        desc="The exact seating capacity is provided",
        parent=v1_node,
        critical=True,
    )
    cap_sources = _combine_sources(v.capacity_sources, v.sources)
    cap_claim = f"The seating capacity of {v.venue_name or ''} is {v.capacity or ''}."
    await evaluator.verify(
        claim=cap_claim,
        node=capacity_leaf,
        sources=cap_sources,
        additional_instruction="Verify the capacity figure on venue pages or trusted sources. If multiple sources exist, "
                              "prefer venue/official pages.",
    )

    # Smallest capacity among NA April–May 2026
    smallest_leaf = evaluator.add_leaf(
        id="venue_1_smallest_capacity",
        desc="Among all David Byrne's North American venues in April–May 2026, this venue has the smallest seating capacity",
        parent=v1_node,
        critical=True,
    )
    competitor_sources = []
    for c in ext.venue1_competitors:
        competitor_sources.extend(c.sources)
    smallest_sources = _combine_sources(cap_sources, artist_sources, competitor_sources)
    smallest_claim = (
        f"Among David Byrne's North American venues during April and May 2026, "
        f"{v.venue_name or ''} in {v.city or ''} has the smallest seating capacity."
    )
    await evaluator.verify(
        claim=smallest_claim,
        node=smallest_leaf,
        sources=smallest_sources,
        additional_instruction="Use the provided sources to compare capacities. If the available sources cannot establish "
                              "the comparative claim, judge as not supported.",
    )

    # Wheelchair accessibility calculation (non-critical, pure computation check)
    wc_leaf = evaluator.add_leaf(
        id="venue_1_wheelchair_accessibility",
        desc="Calculate the minimum number of wheelchair accessible seats required based on 1% of total capacity",
        parent=v1_node,
        critical=False,
    )
    cap_int = _parse_capacity_to_int(v.capacity)
    wc_min = _compute_wheelchair_min(cap_int)
    # If the answer provided a number, verify it matches the calculation; otherwise, verify the calculation statement itself.
    if v.wheelchair_min_provided and wc_min is not None:
        wc_claim = (
            f"The minimum number of wheelchair accessible seats required at {v.venue_name or ''} "
            f"(capacity {cap_int}) is {v.wheelchair_min_provided}, calculated as 1% rounded up."
        )
        await evaluator.verify(
            claim=wc_claim,
            node=wc_leaf,
            additional_instruction=f"Check the arithmetic: ceil(1% of {cap_int}) should equal {wc_min}. "
                                   f"Pass if the provided number equals {wc_min}.",
        )
    elif wc_min is not None:
        wc_claim = (
            f"For a capacity of {cap_int}, the minimum number of wheelchair accessible seats required is {wc_min}, "
            f"calculated as 1% rounded up."
        )
        await evaluator.verify(
            claim=wc_claim,
            node=wc_leaf,
            additional_instruction="This is a pure arithmetic check. Confirm the calculation is correct.",
        )
    else:
        # If we cannot parse capacity, mark this check as failed explicitly
        # Convert leaf to failed state
        wc_leaf.score = 0.0
        wc_leaf.status = "failed"

    # URL reference existence (critical: at least one reference provided)
    has_any_url = len(artist_sources) > 0 or len(cap_sources) > 0
    evaluator.add_custom_node(
        result=has_any_url,
        id="venue_1_url_reference",
        desc="Provide a valid URL confirming the venue, artist, tour date, and capacity information",
        parent=v1_node,
        critical=True,
    )


async def verify_venue_2(evaluator: Evaluator, parent_node, ext: TourVenuesExtraction) -> None:
    """Verify Venue 2: Renee Rapp largest capacity EU March 2026."""
    v = ext.venue2 or VenuePick()

    # Group node
    v2_node = evaluator.add_parallel(
        id="venue_2_largest_renee_rapp_eu",
        desc="The largest capacity venue where Renee Rapp performs in Europe during March 2026 BITE ME tour",
        parent=parent_node,
        critical=False,
    )

    # Venue identification (existence)
    ident_ok = bool(v.venue_name) and bool(v.city)
    evaluator.add_custom_node(
        result=ident_ok,
        id="venue_2_identification",
        desc="The venue name and city are provided",
        parent=v2_node,
        critical=True,
    )

    # Artist and tour confirmation
    artist_leaf = evaluator.add_leaf(
        id="venue_2_artist_and_tour",
        desc="The venue is confirmed to host Renee Rapp during her March 2026 European BITE ME tour",
        parent=v2_node,
        critical=True,
    )
    artist_sources = _combine_sources(v.sources, v.artist_tour_sources)
    artist_claim = (
        f"Renee Rapp performs at {v.venue_name or ''} in {v.city or ''} during March 2026 as part of her European BITE ME tour."
    )
    await evaluator.verify(
        claim=artist_claim,
        node=artist_leaf,
        sources=artist_sources,
        additional_instruction="Accept reputable schedule or ticket pages. The page should clearly indicate Renee Rapp, "
                              "the venue, and a date in March 2026 for Europe.",
    )

    # Capacity stated and verified
    capacity_leaf = evaluator.add_leaf(
        id="venue_2_capacity_stated",
        desc="The exact seating capacity is provided",
        parent=v2_node,
        critical=True,
    )
    cap_sources = _combine_sources(v.capacity_sources, v.sources)
    cap_claim = f"The seating capacity of {v.venue_name or ''} is {v.capacity or ''}."
    await evaluator.verify(
        claim=cap_claim,
        node=capacity_leaf,
        sources=cap_sources,
        additional_instruction="Verify the capacity figure on venue or trusted pages.",
    )

    # Largest capacity among EU March 2026
    largest_leaf = evaluator.add_leaf(
        id="venue_2_largest_capacity",
        desc="Among all Renee Rapp's European venues in March 2026, this venue has the largest seating capacity",
        parent=v2_node,
        critical=True,
    )
    competitor_sources = []
    for c in ext.venue2_competitors:
        competitor_sources.extend(c.sources)
    largest_sources = _combine_sources(cap_sources, artist_sources, competitor_sources)
    largest_claim = (
        f"Among Renee Rapp's European venues during March 2026, "
        f"{v.venue_name or ''} in {v.city or ''} has the largest seating capacity."
    )
    await evaluator.verify(
        claim=largest_claim,
        node=largest_leaf,
        sources=largest_sources,
        additional_instruction="Use the provided sources to compare capacities. If the available sources cannot establish "
                              "the comparative claim, judge as not supported.",
    )

    # Wheelchair accessibility calculation (non-critical)
    wc_leaf = evaluator.add_leaf(
        id="venue_2_wheelchair_accessibility",
        desc="Calculate the minimum number of wheelchair accessible seats required based on 1% of total capacity",
        parent=v2_node,
        critical=False,
    )
    cap_int = _parse_capacity_to_int(v.capacity)
    wc_min = _compute_wheelchair_min(cap_int)
    if v.wheelchair_min_provided and wc_min is not None:
        wc_claim = (
            f"The minimum number of wheelchair accessible seats required at {v.venue_name or ''} "
            f"(capacity {cap_int}) is {v.wheelchair_min_provided}, calculated as 1% rounded up."
        )
        await evaluator.verify(
            claim=wc_claim,
            node=wc_leaf,
            additional_instruction=f"Check the arithmetic: ceil(1% of {cap_int}) should equal {wc_min}. "
                                   f"Pass if the provided number equals {wc_min}.",
        )
    elif wc_min is not None:
        wc_claim = (
            f"For a capacity of {cap_int}, the minimum number of wheelchair accessible seats required is {wc_min}, "
            f"calculated as 1% rounded up."
        )
        await evaluator.verify(
            claim=wc_claim,
            node=wc_leaf,
            additional_instruction="This is a pure arithmetic check. Confirm the calculation is correct.",
        )
    else:
        wc_leaf.score = 0.0
        wc_leaf.status = "failed"

    # URL reference existence (critical)
    has_any_url = len(artist_sources) > 0 or len(cap_sources) > 0
    evaluator.add_custom_node(
        result=has_any_url,
        id="venue_2_url_reference",
        desc="Provide a valid URL confirming the venue, artist, tour date, and capacity information",
        parent=v2_node,
        critical=True,
    )


async def verify_venue_3(evaluator: Evaluator, parent_node, ext: TourVenuesExtraction) -> None:
    """Verify Venue 3: Shared venue where both artists perform in 2026."""
    name = ext.venue3_name or ""
    city = ext.venue3_city or ""

    # Group node
    v3_node = evaluator.add_parallel(
        id="venue_3_shared",
        desc="A venue where both David Byrne and Renee Rapp perform during their respective 2026 tours",
        parent=parent_node,
        critical=False,
    )

    # Venue identification (existence)
    ident_ok = bool(ext.venue3_name) and bool(ext.venue3_city)
    evaluator.add_custom_node(
        result=ident_ok,
        id="venue_3_identification",
        desc="The venue name and city are provided",
        parent=v3_node,
        critical=True,
    )

    # Both artists perform (critical) -> split into two critical leaves to avoid multi-fact in one leaf
    both_node = evaluator.add_parallel(
        id="venue_3_both_artists_main",
        desc="The venue hosts both David Byrne (in 2026) and Renee Rapp (in 2026) as confirmed by their respective tour schedules",
        parent=v3_node,
        critical=True,
    )

    # David Byrne at venue (2026)
    db_leaf = evaluator.add_leaf(
        id="venue_3_david_byrne_performs",
        desc=f"David Byrne performs at {name} in {city} during 2026",
        parent=both_node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"David Byrne performs at {name} in {city} during 2026.",
        node=db_leaf,
        sources=_combine_sources(ext.venue3_david_sources, ext.venue3_sources),
        additional_instruction="Accept reputable tour schedules or ticketing pages confirming the artist, venue, and a 2026 date.",
    )

    # Renee Rapp at venue (2026)
    rr_leaf = evaluator.add_leaf(
        id="venue_3_renee_rapp_performs",
        desc=f"Renee Rapp performs at {name} in {city} during 2026",
        parent=both_node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"Renee Rapp performs at {name} in {city} during 2026.",
        node=rr_leaf,
        sources=_combine_sources(ext.venue3_renee_sources, ext.venue3_sources),
        additional_instruction="Accept reputable tour schedules or ticketing pages confirming the artist, venue, and a 2026 date.",
    )

    # URL references existence (critical – both artists should have at least one confirming URL)
    has_db = len(ext.venue3_david_sources) > 0
    has_rr = len(ext.venue3_renee_sources) > 0
    evaluator.add_custom_node(
        result=(has_db and has_rr),
        id="venue_3_url_reference",
        desc="Provide valid URLs confirming both artists perform at this venue during their 2026 tours",
        parent=v3_node,
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
    Evaluate an agent's answer for the 2026 concert venues identification task and return a structured summary.
    """
    # Initialize evaluator with parallel root to allow partial credit across venues
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

    # Extract structured information from the answer
    extraction = await evaluator.extract(
        prompt=prompt_extract_venues(),
        template_class=TourVenuesExtraction,
        extraction_name="venues_extraction",
    )

    # Build verification subtrees
    await verify_venue_1(evaluator, root, extraction)
    await verify_venue_2(evaluator, root, extraction)
    await verify_venue_3(evaluator, root, extraction)

    # Return evaluation summary
    return evaluator.get_summary()