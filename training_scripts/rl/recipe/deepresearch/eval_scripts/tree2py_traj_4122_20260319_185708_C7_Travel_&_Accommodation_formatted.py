import asyncio
import logging
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field
from obj_task_eval.llm_client.base_client import LLMClient

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy, VerificationNode
from obj_task_eval.utils.cache_filesys import CacheFileSys


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "airport_hotels_multi_city_march_2026"
TASK_DESCRIPTION = (
    "You are planning a multi-city business trip that will require overnight layovers at four major "
    "international airports in different countries. To minimize transit time and costs, identify one suitable "
    "airport hotel at each of the following locations: (1) Singapore Changi Airport (Singapore), "
    "(2) Los Angeles International Airport - LAX (United States), (3) Tokyo Narita International Airport (Japan), "
    "and (4) Dubai International Airport - DXB (United Arab Emirates). For each airport, identify one hotel that "
    "meets ALL of the following criteria: located within 10 minutes travel time from the airport terminal "
    "(either by free shuttle service or within walking distance), offers free airport shuttle service to and from "
    "the airport, has a guest rating of at least 4.0 out of 5.0 (or equivalent rating of 8.0 out of 10.0 on other "
    "rating scales), and must be currently operational and accepting bookings as of March 2026. For each of the four "
    "airports, provide: the hotel name, verification of its proximity to the airport, verification of free shuttle "
    "service availability, the hotel's guest rating and source, and a reference URL from an official hotel website "
    "or major booking platform."
)

CURRENT_TIMEPOINT_LABEL = "March 2026"


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class HotelItem(BaseModel):
    # Core identification
    hotel_name: Optional[str] = None

    # Sources (URLs)
    reference_url: Optional[str] = None           # Official site or major booking platform (primary reference)
    rating_source_url: Optional[str] = None       # URL where the rating is shown (can equal reference_url)
    booking_url: Optional[str] = None             # URL demonstrating booking availability (can equal reference_url)
    additional_urls: List[str] = Field(default_factory=list)  # Any other URLs cited for this hotel

    # Claimed details from the answer (free-form text)
    proximity_text: Optional[str] = None          # e.g., "5 minutes by free shuttle" or "connected to Terminal 3"
    shuttle_text: Optional[str] = None            # e.g., "Complimentary shuttle to/from LAX"
    rating_text: Optional[str] = None             # e.g., "4.3/5" or "8.8/10"


class HotelsExtraction(BaseModel):
    singapore: Optional[HotelItem] = None
    los_angeles: Optional[HotelItem] = None
    tokyo: Optional[HotelItem] = None
    dubai: Optional[HotelItem] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_hotels() -> str:
    return """
    Extract one candidate hotel per location from the answer. If multiple hotels are listed for a location, extract only the first one.
    For each location, extract the following fields exactly as presented in the answer:
    - hotel_name: Name of the hotel.
    - reference_url: A single URL (official hotel site or a major booking platform) referenced for the hotel. If multiple URLs are present, prefer the official hotel website; otherwise, choose a well-known booking platform (e.g., booking.com, hotels.com, expedia.com, agoda.com, marriott.com, hilton.com, hyatt.com, ihg.com, accor.com, radissonhotels.com).
    - rating_source_url: The URL where the guest rating is displayed. If the rating appears on the same page as the reference_url, set it equal to reference_url. If the answer does not cite a URL for the rating, return null.
    - booking_url: A URL that shows booking availability for the hotel (e.g., "Book now", availability calendar). If the reference_url already shows booking availability, set booking_url equal to reference_url. If not provided, return null.
    - additional_urls: Any other URLs referenced for this hotel (list all remaining URLs that are not duplicates of the above).
    - proximity_text: The proximity or travel time statement given in the answer (e.g., "5 minutes by free shuttle", "inside terminal 3", "connected via walkway").
    - shuttle_text: The precise text from the answer about free airport shuttle service (e.g., "complimentary shuttle every 20 minutes").
    - rating_text: The rating value and scale as quoted in the answer (e.g., "4.2/5", "8.6/10", "4.5 out of 5").

    Structure the JSON object with the following top-level keys (one object per location):
    - singapore
    - los_angeles
    - tokyo
    - dubai

    Each location key maps to a HotelItem object with the fields listed above. If the answer provides no hotel for a location, set that location to null. If a particular field is missing, set it to null; for additional_urls, set to an empty list if none.
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _unique_non_empty(urls: List[Optional[str]]) -> List[str]:
    seen = set()
    out: List[str] = []
    for u in urls:
        if not u:
            continue
        uu = u.strip()
        if not uu:
            continue
        if uu not in seen:
            seen.add(uu)
            out.append(uu)
    return out


def _sources_for_proximity(item: HotelItem) -> List[str]:
    return _unique_non_empty([item.reference_url] + item.additional_urls)


def _sources_for_shuttle(item: HotelItem) -> List[str]:
    return _unique_non_empty([item.reference_url] + item.additional_urls)


def _sources_for_rating(item: HotelItem) -> List[str]:
    if item.rating_source_url:
        return _unique_non_empty([item.rating_source_url])
    # Fallback to reference/additional if rating source not explicitly provided
    return _unique_non_empty([item.reference_url] + item.additional_urls)


def _sources_for_operational(item: HotelItem) -> List[str]:
    if item.booking_url:
        return _unique_non_empty([item.booking_url])
    return _unique_non_empty([item.reference_url] + item.additional_urls)


def _airport_label(tag: str) -> str:
    return {
        "singapore": "Singapore Changi Airport (SIN)",
        "los_angeles": "Los Angeles International Airport (LAX)",
        "tokyo": "Tokyo Narita International Airport (NRT)",
        "dubai": "Dubai International Airport (DXB)",
    }.get(tag, tag)


# --------------------------------------------------------------------------- #
# Verification builder per airport                                            #
# --------------------------------------------------------------------------- #
async def verify_airport_hotel(
    evaluator: Evaluator,
    parent: VerificationNode,
    tag: str,
    item: Optional[HotelItem],
) -> None:
    """
    Build the verification subtree for one airport location.
    Each required criterion is wrapped in a critical sub-group to enforce "all-or-nothing" at the location level.
    """
    airport = _airport_label(tag)

    # Parent node per airport (parallel; not critical at root level to allow partial credit across airports)
    loc_node = evaluator.add_parallel(
        id=f"{tag}_hotel",
        desc=f"Identify and verify a qualifying hotel at {airport}",
        parent=parent,
        critical=False,
    )

    # Convenience values
    name = item.hotel_name if item else None

    # 1) Hotel name provided (critical leaf as existence check)
    evaluator.add_custom_node(
        result=bool(name and name.strip()),
        id=f"{tag}_hotel_name",
        desc=f"Provide the name of a hotel at {airport}",
        parent=loc_node,
        critical=True
    )

    # 2) Reference URL validity group (critical)
    ref_group = evaluator.add_parallel(
        id=f"{tag}_reference_group",
        desc="Reference URL validity check",
        parent=loc_node,
        critical=True
    )
    # 2.1) Reference URL provided (critical)
    evaluator.add_custom_node(
        result=bool(item and item.reference_url and item.reference_url.strip()),
        id=f"{tag}_reference_url_present",
        desc="Provide a reference URL from an official hotel website or major booking platform (presence check)",
        parent=ref_group,
        critical=True
    )
    # 2.2) Reference URL is official hotel website or major booking platform (critical)
    ref_valid = evaluator.add_leaf(
        id=f"{tag}_reference_url_valid",
        desc="Reference URL is an official hotel website or a major booking platform",
        parent=ref_group,
        critical=True
    )
    await evaluator.verify(
        claim="This URL is an official hotel website or a major hotel booking platform.",
        node=ref_valid,
        sources=(item.reference_url if item else None),
        additional_instruction=(
            "Judge based on the page content and domain reputation. Acceptable platforms include well-known brands "
            "such as booking.com, hotels.com, expedia.com, agoda.com, marriott.com, hilton.com, hyatt.com, ihg.com, "
            "accor.com, radissonhotels.com, trip.com, priceline.com, orbitz.com, travelocity.com. "
            "An official hotel website typically has brand/chain domain and booking/rooms sections."
        )
    )

    # 3) Proximity group (critical)
    prox_group = evaluator.add_parallel(
        id=f"{tag}_proximity_group",
        desc=f"Verify hotel is within 10 minutes from {airport} terminal by free shuttle or walking distance",
        parent=loc_node,
        critical=True
    )
    prox_sources = _sources_for_proximity(item) if item else []
    evaluator.add_custom_node(
        result=len(prox_sources) > 0,
        id=f"{tag}_proximity_sources_present",
        desc="At least one proximity-related source URL is provided",
        parent=prox_group,
        critical=True
    )
    prox_leaf = evaluator.add_leaf(
        id=f"{tag}_proximity_verification",
        desc=f"Hotel is within 10 minutes from {airport} terminal (by free shuttle or walking distance)",
        parent=prox_group,
        critical=True
    )
    await evaluator.verify(
        claim=(
            f"The hotel '{name}' is within 10 minutes travel time of the terminal(s) at {airport}, "
            "either by free shuttle or within walking distance."
        ),
        node=prox_leaf,
        sources=prox_sources,
        additional_instruction=(
            "Look for explicit mentions such as 'inside terminal', 'connected to terminal', 'on-airport', "
            "'adjacent to terminal', 'walking distance', or a stated shuttle time <= 10 minutes. "
            "If the page clearly indicates it is terminal-connected/inside/on-airport, treat it as within 10 minutes. "
            "If only distance is given, assume a typical walking speed (approx. 80m/min). "
            "If evidence implies longer than 10 minutes, mark as not supported."
        )
    )

    # 4) Free shuttle service group (critical)
    shuttle_group = evaluator.add_parallel(
        id=f"{tag}_shuttle_group",
        desc=f"Verify the hotel offers free airport shuttle service to/from {airport}",
        parent=loc_node,
        critical=True
    )
    shuttle_sources = _sources_for_shuttle(item) if item else []
    evaluator.add_custom_node(
        result=len(shuttle_sources) > 0,
        id=f"{tag}_shuttle_sources_present",
        desc="At least one shuttle-related source URL is provided",
        parent=shuttle_group,
        critical=True
    )
    shuttle_leaf = evaluator.add_leaf(
        id=f"{tag}_free_shuttle_verification",
        desc="Hotel offers free (complimentary) airport shuttle service to and from the airport",
        parent=shuttle_group,
        critical=True
    )
    await evaluator.verify(
        claim=(
            f"The hotel '{name}' provides free (complimentary) airport shuttle service to and from {airport}."
        ),
        node=shuttle_leaf,
        sources=shuttle_sources,
        additional_instruction=(
            "Confirm 'free' or 'complimentary' shuttle (both directions). Accept synonyms like 'complimentary airport "
            "transfers' or 'free airport bus'. Reject if the shuttle is paid, third-party only, or not available."
        )
    )

    # 5) Rating group (critical)
    rating_group = evaluator.add_parallel(
        id=f"{tag}_rating_group",
        desc="Verify the hotel's guest rating and threshold with source",
        parent=loc_node,
        critical=True
    )
    rating_sources = _sources_for_rating(item) if item else []
    evaluator.add_custom_node(
        result=len(rating_sources) > 0,
        id=f"{tag}_rating_source_present",
        desc="A rating source URL is provided (can be the same as the reference URL)",
        parent=rating_group,
        critical=True
    )
    rating_supported = evaluator.add_leaf(
        id=f"{tag}_rating_value_supported",
        desc="The stated guest rating is shown on the cited page",
        parent=rating_group,
        critical=True
    )
    rating_text_clause = (
        f" is {item.rating_text}" if (item and item.rating_text and item.rating_text.strip()) else " is shown"
    )
    await evaluator.verify(
        claim=f"According to this page, the guest rating for hotel '{name}'{rating_text_clause}.",
        node=rating_supported,
        sources=rating_sources,
        additional_instruction=(
            "Verify the page displays a numeric guest rating or a clear star/score indicator. "
            "Allow rounding differences (e.g., 4.0 vs 4.03)."
        )
    )
    rating_threshold = evaluator.add_leaf(
        id=f"{tag}_rating_threshold_met",
        desc="Guest rating meets the threshold (≥4.0/5.0 or ≥8.0/10.0)",
        parent=rating_group,
        critical=True
    )
    await evaluator.verify(
        claim="The guest rating shown on this page is at least 4.0 out of 5.0 OR at least 8.0 out of 10.0.",
        node=rating_threshold,
        sources=rating_sources,
        additional_instruction=(
            "Parse the rating scale on the page. Accept if the rating is ≥4.0 on a 5-point scale or ≥8.0 on a 10-point "
            "scale. If the page uses a different scale (e.g., 100-point), convert proportionally: 80/100 equals 8/10."
        )
    )

    # 6) Operational / accepting bookings as of March 2026 (critical)
    op_group = evaluator.add_parallel(
        id=f"{tag}_operational_group",
        desc=f"Verify hotel is currently operational and accepting bookings as of {CURRENT_TIMEPOINT_LABEL}",
        parent=loc_node,
        critical=True
    )
    op_sources = _sources_for_operational(item) if item else []
    evaluator.add_custom_node(
        result=len(op_sources) > 0,
        id=f"{tag}_operational_sources_present",
        desc="At least one booking/operational source URL is provided",
        parent=op_group,
        critical=True
    )
    op_leaf = evaluator.add_leaf(
        id=f"{tag}_operational_booking_verification",
        desc=f"Hotel is operational and accepting bookings as of {CURRENT_TIMEPOINT_LABEL}",
        parent=op_group,
        critical=True
    )
    await evaluator.verify(
        claim=f"As of {CURRENT_TIMEPOINT_LABEL}, the hotel '{name}' is operational and accepting bookings.",
        node=op_leaf,
        sources=op_sources,
        additional_instruction=(
            f"Look for a booking engine, 'Book now', 'Check availability', or availability calendar indicating open "
            f"inventory around {CURRENT_TIMEPOINT_LABEL}. If the page signals permanent closure, renovation closure, "
            f"or no availability without explanation, mark as not supported."
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
    Evaluate an answer for the multi-airport hotel selection and verification task.
    """
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,   # Root kept non-critical to allow partial credit across airports
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

    # 1) Extract structured info from the answer
    extracted: HotelsExtraction = await evaluator.extract(
        prompt=prompt_extract_hotels(),
        template_class=HotelsExtraction,
        extraction_name="airport_hotels_extraction"
    )

    # 2) Build verification subtrees per airport
    await verify_airport_hotel(evaluator, root, "singapore", extracted.singapore or HotelItem())
    await verify_airport_hotel(evaluator, root, "los_angeles", extracted.los_angeles or HotelItem())
    await verify_airport_hotel(evaluator, root, "tokyo", extracted.tokyo or HotelItem())
    await verify_airport_hotel(evaluator, root, "dubai", extracted.dubai or HotelItem())

    # 3) Return structured summary
    return evaluator.get_summary()