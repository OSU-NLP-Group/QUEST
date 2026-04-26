import asyncio
import logging
from typing import Any, List, Optional, Dict, Tuple

from pydantic import BaseModel, Field

from obj_task_eval.llm_client.base_client import LLMClient
from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


TASK_ID = "bruno_romantic_tour_top3_stadiums_2026_us"
TASK_DESCRIPTION = (
    "Identify the 3 largest-capacity stadium venues (by standard concert capacity) where Bruno Mars will perform "
    "during the United States leg of \"The Romantic Tour\" in 2026. For each of these three venues, provide the following information: "
    "(1) Official venue name, (2) City and state location, (3) Standard concert capacity, "
    "(4) All specific dates (month and day) that Bruno Mars is scheduled to perform at that venue in 2026, "
    "(5) The opening acts scheduled to perform at those shows, "
    "(6) A reference URL to verify the tour information. "
    "Additionally, calculate and provide the total number of shows Bruno Mars will perform across these three venues combined."
)


class VenueItem(BaseModel):
    venue_name: Optional[str] = None
    city: Optional[str] = None
    state: Optional[str] = None
    standard_concert_capacity: Optional[str] = None
    dates_2026: List[str] = Field(default_factory=list)
    opening_acts: List[str] = Field(default_factory=list)
    tour_reference_urls: List[str] = Field(default_factory=list)
    capacity_reference_urls: List[str] = Field(default_factory=list)


class VenuesExtraction(BaseModel):
    venues: List[VenueItem] = Field(default_factory=list)
    total_shows_text: Optional[str] = None


def prompt_extract_venues() -> str:
    return (
        "Extract up to three venues listed in the answer in the order they are presented. For each venue, extract:\n"
        "1) venue_name: The official venue name exactly as written in the answer.\n"
        "2) city: The city of the venue as written.\n"
        "3) state: The U.S. state of the venue as written (e.g., CA, California).\n"
        "4) standard_concert_capacity: The standard concert capacity number/text as given (do not normalize; copy exactly).\n"
        "5) dates_2026: A list of all 2026 performance dates (month and day) explicitly listed for this venue. Preserve formatting.\n"
        "6) opening_acts: The opening acts listed for the show(s) at this venue. If none are listed, return an empty list.\n"
        "7) tour_reference_urls: A list of URL(s) that support the tour-stop information for this venue (venue and dates; and if asserted, opening acts). Extract only explicit URLs in the answer.\n"
        "8) capacity_reference_urls: A list of URL(s) that support the venue's stated standard concert capacity. Extract only explicit URLs in the answer.\n"
        "Also extract total_shows_text: The total number of shows across the three venues as stated in the answer; copy the exact text (e.g., '6', 'six', 'a total of six shows'). If not provided, return null.\n"
        "If the answer lists more than three venues, only extract the first three. If fewer than three are listed, extract what is available and return empty placeholders for missing ones."
    )


def _normalize_name(name: Optional[str]) -> str:
    return (name or "").strip().lower()


def _is_valid_url(u: Optional[str]) -> bool:
    if not u:
        return False
    return u.strip().lower().startswith("http://") or u.strip().lower().startswith("https://")


def _clean_urls(urls: List[str]) -> List[str]:
    seen = set()
    cleaned = []
    for u in urls:
        if _is_valid_url(u):
            key = u.strip()
            if key not in seen:
                seen.add(key)
                cleaned.append(key)
    return cleaned


def _flatten_urls(groups: List[List[str]]) -> List[str]:
    allu = []
    for g in groups:
        allu.extend(g or [])
    return _clean_urls(allu)


def _parse_int_from_text(text: Optional[str]) -> Optional[int]:
    if not text:
        return None
    raw = text.strip().lower()
    digits = "".join(ch for ch in raw if ch.isdigit())
    if digits:
        try:
            return int(digits)
        except Exception:
            pass
    words_map = {
        "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
        "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
        "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14,
        "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18,
        "nineteen": 19, "twenty": 20
    }
    for word, val in words_map.items():
        if word in raw:
            return val
    return None


def _capacity_to_int(cap: Optional[str]) -> Optional[int]:
    if not cap:
        return None
    s = cap.lower()
    s = s.replace(",", "")
    digits = [ch for ch in s if ch.isdigit()]
    if not digits:
        return None
    try:
        num = int("".join(digits))
        return num
    except Exception:
        return None


def _format_list_for_claim(items: List[str]) -> str:
    if not items:
        return "None"
    return "; ".join(items)


async def _verify_venue(
    evaluator: Evaluator,
    parent_node,
    venue: VenueItem,
    rank_index: int
) -> None:
    vn = evaluator.add_parallel(
        id=f"venue_{rank_index}",
        desc=f"Venue ranked #{rank_index} by standard concert capacity among eligible US stadium tour stops",
        parent=parent_node,
        critical=False
    )

    tour_urls = _clean_urls(venue.tour_reference_urls)
    cap_urls = _clean_urls(venue.capacity_reference_urls)
    all_urls = _flatten_urls([tour_urls, cap_urls])

    # Tour reference URL presence (critical)
    evaluator.add_custom_node(
        result=len(tour_urls) > 0,
        id=f"venue_{rank_index}_tour_reference_url",
        desc="Provides at least one valid, verifiable reference URL that supports the tour-stop information (venue and dates; and if asserted, opening acts)",
        parent=vn,
        critical=True
    )

    # Capacity reference URL presence (critical)
    evaluator.add_custom_node(
        result=len(cap_urls) > 0,
        id=f"venue_{rank_index}_capacity_reference_url",
        desc="Provides at least one valid, verifiable reference URL that supports the stated standard concert capacity",
        parent=vn,
        critical=True
    )

    # Official venue name (critical)
    name_node = evaluator.add_leaf(
        id=f"venue_{rank_index}_name",
        desc="Official venue name is provided (correctly spelled and identifiable)",
        parent=vn,
        critical=True
    )
    name_claim = (
        f"The official venue listed for Bruno Mars' 2026 'The Romantic Tour' at this stop is '{venue.venue_name}'. "
        f"Verify the venue name on the provided tour reference URL(s)."
    )
    await evaluator.verify(
        claim=name_claim,
        node=name_node,
        sources=tour_urls,
        additional_instruction="Match the venue name exactly or allow minor formatting variations. Use the tour page or official announcement."
    )

    # Location (critical)
    location_node = evaluator.add_leaf(
        id=f"venue_{rank_index}_location",
        desc="City and state are provided and accurate",
        parent=vn,
        critical=True
    )
    loc_claim = (
        f"The venue is located in {venue.city}, {venue.state}."
    )
    await evaluator.verify(
        claim=loc_claim,
        node=location_node,
        sources=all_urls,
        additional_instruction="Confirm the venue's city and state from the tour page and/or venue page."
    )

    # Scope eligibility (critical)
    scope_node = evaluator.add_leaf(
        id=f"venue_{rank_index}_scope_eligibility",
        desc="Venue is a stadium-type facility and is an official stop on the US leg of the tour per brunomars.com/tour",
        parent=vn,
        critical=True
    )
    scope_claim = (
        "This venue is a stadium-type facility and is listed as an official stop on the United States leg of Bruno Mars' 2026 'The Romantic Tour'."
    )
    await evaluator.verify(
        claim=scope_claim,
        node=scope_node,
        sources=all_urls,
        additional_instruction="Use the venue page to confirm stadium classification and the brunomars.com/tour or official tour announcement to confirm the US leg stop."
    )

    # Capacity value (critical)
    capacity_node = evaluator.add_leaf(
        id=f"venue_{rank_index}_capacity",
        desc="Standard concert capacity is provided and verifiable (fits within documented concert configuration information for the venue)",
        parent=vn,
        critical=True
    )
    cap_claim = (
        f"The venue's standard concert capacity (concert configuration) is approximately '{venue.standard_concert_capacity}'."
    )
    await evaluator.verify(
        claim=cap_claim,
        node=capacity_node,
        sources=cap_urls,
        additional_instruction="Verify the stated capacity against reliable sources (official venue documentation or reputable references). Allow minor rounding or typical concert configuration variance."
    )

    # Dates (critical)
    dates_node = evaluator.add_leaf(
        id=f"venue_{rank_index}_dates",
        desc="All 2026 performance dates (month and day) at this venue are provided and match the official tour schedule",
        parent=vn,
        critical=True
    )
    dates_str = _format_list_for_claim(venue.dates_2026)
    dates_claim = (
        f"Bruno Mars is scheduled to perform at this venue on the following 2026 dates: {dates_str}. "
        f"Verify that all listed dates match the official tour schedule."
    )
    await evaluator.verify(
        claim=dates_claim,
        node=dates_node,
        sources=tour_urls,
        additional_instruction="Check the official tour page(s) or announcement for the exact dates at this venue. All dates must match; missing or extra dates should fail."
    )

    # Opening acts (critical)
    opening_node = evaluator.add_leaf(
        id=f"venue_{rank_index}_opening_acts",
        desc="Opening acts for the show(s) at this venue are provided and verified by official tour announcements/sources",
        parent=vn,
        critical=True
    )
    openers_str = _format_list_for_claim(venue.opening_acts)
    opening_claim = (
        f"The opening acts scheduled to perform at the shows at this venue are: {openers_str}. "
        f"Verify these opening acts via official tour announcements or the tour page."
    )
    await evaluator.verify(
        claim=opening_claim,
        node=opening_node,
        sources=tour_urls,
        additional_instruction="If the provided sources do not list opening acts or indicate TBA, the claim should be judged incorrect."
    )


async def evaluate_answer(
    client: LLMClient,
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
        default_model=model
    )

    extracted = await evaluator.extract(
        prompt=prompt_extract_venues(),
        template_class=VenuesExtraction,
        extraction_name="venues_extraction"
    )

    venues: List[VenueItem] = list(extracted.venues or [])
    if len(venues) < 3:
        # Pad with empty VenueItem to ensure 3 slots
        venues = venues + [VenueItem() for _ in range(3 - len(venues))]
    else:
        venues = venues[:3]

    # Build global constraints node (critical parallel)
    global_node = evaluator.add_parallel(
        id="global_constraints",
        desc="Set-level constraints that must hold for the selected set of three venues",
        parent=root,
        critical=True
    )

    # exactly_three_venues (critical)
    unique_names = set()
    names_ok = True
    for v in venues:
        nm = _normalize_name(v.venue_name)
        if not nm:
            names_ok = False
            break
        if nm in unique_names:
            names_ok = False
            break
        unique_names.add(nm)
    evaluator.add_custom_node(
        result=(len(venues) == 3 and names_ok),
        id="exactly_three_venues",
        desc="Provides exactly 3 distinct venues (no duplicates)",
        parent=global_node,
        critical=True
    )

    # top3_and_ordering_by_capacity (critical)
    top3_node = evaluator.add_leaf(
        id="top3_and_ordering_by_capacity",
        desc="The 3 venues are the top 3 eligible US stadium tour stops by standard concert capacity and are presented in descending order by that capacity",
        parent=global_node,
        critical=True
    )
    caps_for_claim = []
    for idx, v in enumerate(venues, start=1):
        caps_for_claim.append(f"#{idx}: {v.venue_name} — capacity '{v.standard_concert_capacity}'")
    cap_order_claim_details = "; ".join(caps_for_claim) if caps_for_claim else "No capacity details provided."
    all_tour_urls = _flatten_urls([v.tour_reference_urls for v in venues])
    all_capacity_urls = _flatten_urls([v.capacity_reference_urls for v in venues])
    combined_sources = _flatten_urls([all_tour_urls, all_capacity_urls])
    top3_claim = (
        f"These three venues are the top three eligible UNITED STATES stadium tour stops by standard concert capacity "
        f"for Bruno Mars' 2026 'The Romantic Tour', and they are presented in descending order of capacity: {cap_order_claim_details}. "
        f"Verify using the provided tour schedule and capacity references; if any other US stop during this leg has an equal or greater "
        f"standard concert capacity than one listed, the claim should be judged incorrect."
    )
    await evaluator.verify(
        claim=top3_claim,
        node=top3_node,
        sources=combined_sources,
        additional_instruction="Confirm both the capacities and the relative ordering. Use reliable capacity references and the official tour schedule."
    )

    # tour_scope_and_timeframe (critical)
    timeframe_node = evaluator.add_leaf(
        id="tour_scope_and_timeframe",
        desc="All selected venues and listed show dates are on the United States leg of the tour in 2026 and fall within the stated tour period (April 10, 2026 through October 20, 2026)",
        parent=global_node,
        critical=True
    )
    # Summarize dates and locations for claim context
    venue_dates_summary = []
    for v in venues:
        venue_dates_summary.append(
            f"{v.venue_name} in {v.city}, {v.state}: dates {', '.join(v.dates_2026) if v.dates_2026 else 'None'}"
        )
    timeframe_claim = (
        "All selected venues and listed show dates are within the UNITED STATES leg of Bruno Mars' 2026 'The Romantic Tour'. "
        "All dates fall in 2026 within the period April 10, 2026 through October 20, 2026. "
        "Summary: " + "; ".join(venue_dates_summary)
    )
    await evaluator.verify(
        claim=timeframe_claim,
        node=timeframe_node,
        sources=all_tour_urls,
        additional_instruction="Use the official tour page(s) and announcements to confirm each venue is a US stop and every listed date falls within the specified 2026 timeframe."
    )

    # Venue nodes
    for idx, v in enumerate(venues, start=1):
        await _verify_venue(evaluator, root, v, idx)

    # total_shows (critical)
    total_shows_node = evaluator.add_leaf(
        id="total_shows",
        desc="Total number of shows across the three identified venues is correctly calculated as the sum of all listed performance dates across those venues",
        parent=root,
        critical=True
    )
    sum_of_dates = sum(len(v.dates_2026 or []) for v in venues)
    reported_total = _parse_int_from_text(extracted.total_shows_text)
    if reported_total is not None:
        total_claim = (
            f"The answer explicitly provides a total number of shows across the three venues as {reported_total}, "
            f"and the sum of all listed performance dates across those venues equals {sum_of_dates}."
        )
    else:
        total_claim = (
            f"The answer correctly calculates and provides the total number of shows across the three venues, "
            f"which should equal the sum of all listed performance dates across those venues: {sum_of_dates}."
        )
    await evaluator.verify(
        claim=total_claim,
        node=total_shows_node,
        additional_instruction="Check the answer text to ensure it explicitly provides the total shows count and that it equals the sum of the listed dates across the three venues."
    )

    return evaluator.get_summary()