import asyncio
import logging
from typing import Any, List, Optional
from urllib.parse import urlparse

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "seatac_tukwila_hotels_amenities"
TASK_DESCRIPTION = (
    "Identify three hotels located in the SeaTac or Tukwila area near Seattle-Tacoma International Airport that each "
    "provide all of the following amenities: complimentary airport shuttle service to and from SEA, on-site meeting or "
    "conference facilities, and an on-site fitness center or gym. For each hotel, provide the hotel's name, a direct link "
    "to the hotel's official website or a major booking platform page (such as the hotel's page on Hilton.com, Marriott.com, "
    "IHG.com, or similar official sources), and a brief confirmation that all three required amenities are available."
)


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class HotelEntry(BaseModel):
    name: Optional[str] = None
    url: Optional[str] = None
    additional_urls: List[str] = Field(default_factory=list)


class HotelsExtraction(BaseModel):
    hotels: List[HotelEntry] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_hotels() -> str:
    return """
    From the provided answer, extract up to the first five distinct hotels mentioned.
    For each hotel, extract:
    - name: The hotel's name as written in the answer (string).
    - url: A single direct link to the hotel's official website or a major booking platform page for this property.
           This should be the most direct/primary URL given for the hotel. If multiple are provided, pick the main one.
    - additional_urls: Any other URLs (array) mentioned in the answer that are clearly associated with this same hotel
                       (e.g., a separate amenities page, meeting/banquet page, fitness page, or a secondary official/booking URL).
    
    IMPORTANT URL RULES:
    - Extract only URLs explicitly present in the answer text. Do not invent URLs.
    - Extract valid URLs. If a URL is missing a protocol, prepend http://
    - Do not include generic search result links that are not specific to the hotel or its official/booking page.

    Return a JSON object of the form:
    {
      "hotels": [
        {"name": string or null, "url": string or null, "additional_urls": [string, ...]},
        ...
      ]
    }
    If any field is missing, set it to null (for strings) or [] for lists.
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
ALLOWED_OFFICIAL_DOMAINS = {
    # Multi-brand official domains (cover subdomains by suffix matching)
    "hilton.com", "marriott.com", "ihg.com", "hyatt.com", "choicehotels.com", "bestwestern.com",
    "wyndhamhotels.com", "sonesta.com", "radissonhotels.com", "accor.com", "omnihotels.com", "druryhotels.com",

    # Common brand vanity/legacy second-level domains (some redirect but seen in answers)
    "doubletree.com", "embassysuites.com", "hampton.com", "homewoodsuites.com", "home2suites.com",
    "holidayinn.com", "crowneplaza.com", "intercontinental.com", "staybridge.com", "hotelindigo.com",
    "kimptonhotels.com", "parkhyatt.com", "hyattregency.com", "andaz.com", "grandhyatt.com",
    "courtyard.com", "residenceinn.marriott.com", "springhillsuites.com", "towneplaceSuites.com",
    "aloft-hotels.com", "westin.com", "sheraton.com", "fourpoints.com", "lemeridien.com", "renaissancehotels.com",
    "novotel.com", "ibis.com", "mercure.com", "pullmanhotels.com", "sofitel.com", "laquinta.com",
    "ramada.com", "daysinn.com", "super8.com", "hawthorn.com", "tryphotel.com"
}

ALLOWED_MAJOR_BOOKING_DOMAINS = {
    "booking.com", "expedia.com", "hotels.com", "priceline.com", "orbitz.com",
    "travelocity.com", "agoda.com", "kayak.com", "trivago.com",
    # Google Hotels pages are sometimes provided
    "google.com",
    # TripAdvisor is a major travel platform that often lists amenities
    "tripadvisor.com"
}


def _netloc(url: str) -> str:
    try:
        parsed = urlparse(url if "://" in url else "http://" + url)
        return parsed.netloc.lower().lstrip()
    except Exception:
        return ""


def url_matches_allowed_domains(url: Optional[str]) -> bool:
    if not url or not isinstance(url, str):
        return False
    host = _netloc(url)
    if not host:
        return False
    host = host.replace("www.", "")
    return any(host.endswith(d) for d in (ALLOWED_OFFICIAL_DOMAINS | ALLOWED_MAJOR_BOOKING_DOMAINS))


def collect_sources(entry: HotelEntry) -> List[str]:
    urls: List[str] = []
    if entry.url:
        urls.append(entry.url)
    if entry.additional_urls:
        urls.extend([u for u in entry.additional_urls if isinstance(u, str) and u.strip()])
    # Deduplicate while preserving order
    seen = set()
    unique_urls = []
    for u in urls:
        key = u.strip()
        if key and key not in seen:
            seen.add(key)
            unique_urls.append(key)
    return unique_urls


def ordinal(idx: int) -> str:
    mapping = {0: "First", 1: "Second", 2: "Third"}
    return mapping.get(idx, f"#{idx + 1}")


# --------------------------------------------------------------------------- #
# Verification logic                                                          #
# --------------------------------------------------------------------------- #
async def verify_hotel(
    evaluator: Evaluator,
    parent_node,
    hotel: HotelEntry,
    index: int,
) -> None:
    """
    Build verification subtree and run checks for a single hotel.
    """
    hotel_node = evaluator.add_parallel(
        id=f"hotel_{index + 1}",
        desc=f"{ordinal(index)} hotel entry.",
        parent=parent_node,
        critical=False
    )

    # Name provided (existence)
    name_ok = bool(hotel.name and isinstance(hotel.name, str) and hotel.name.strip())
    name_node = evaluator.add_custom_node(
        result=name_ok,
        id=f"hotel_{index + 1}_name",
        desc="Provides the hotel's name.",
        parent=hotel_node,
        critical=True
    )

    # Reference URL provided and is an official or major booking platform page
    url_ok = url_matches_allowed_domains(hotel.url)
    ref_url_node = evaluator.add_custom_node(
        result=url_ok,
        id=f"hotel_{index + 1}_reference_url",
        desc="Provides a direct link to the hotel's official website or a major booking platform page that supports verifying the hotel and its amenities.",
        parent=hotel_node,
        critical=True
    )

    # Prepare sources for subsequent checks (gate by ref_url_node)
    sources = collect_sources(hotel)

    # Location check
    loc_node = evaluator.add_leaf(
        id=f"hotel_{index + 1}_location",
        desc="Hotel is located in the SeaTac or Tukwila area near Seattle-Tacoma International Airport (SEA).",
        parent=hotel_node,
        critical=True
    )
    hotel_name_for_claim = hotel.name if name_ok else "the hotel"
    loc_claim = (
        f"The official/booking page for {hotel_name_for_claim} indicates the property is located in SeaTac, Washington "
        f"or Tukwila, Washington (cities near Seattle-Tacoma International Airport, SEA)."
    )
    await evaluator.verify(
        claim=loc_claim,
        node=loc_node,
        sources=sources,
        additional_instruction=(
            "Focus on the address or city shown on the page. Accept these forms as valid indicators: 'SeaTac', 'Seatac', "
            "'Sea-Tac', or 'Tukwila' (with state WA/Washington). If the page clearly shows the address in SeaTac or Tukwila, "
            "consider it correct. If the page only says 'Seattle' without SeaTac/Tukwila and does not indicate airport-area "
            "location, do not consider it valid."
        ),
    )

    # Amenities group (critical parent)
    amenities_node = evaluator.add_parallel(
        id=f"hotel_{index + 1}_amenities",
        desc="Amenity requirements for this hotel.",
        parent=hotel_node,
        critical=True
    )

    # Complimentary airport shuttle to/from SEA
    shuttle_node = evaluator.add_leaf(
        id=f"hotel_{index + 1}_airport_shuttle",
        desc="Offers complimentary shuttle service to and from Seattle-Tacoma International Airport (SEA).",
        parent=amenities_node,
        critical=True
    )
    shuttle_claim = (
        f"The official/booking page for {hotel_name_for_claim} confirms a complimentary (free) airport shuttle service "
        f"to and from Seattle-Tacoma International Airport (SEA)."
    )
    await evaluator.verify(
        claim=shuttle_claim,
        node=shuttle_node,
        sources=sources,
        additional_instruction=(
            "Look for phrases like 'complimentary airport shuttle', 'free airport shuttle', 'courtesy shuttle', "
            "'complimentary shuttle to/from the airport', or similar. It must clearly indicate no charge (complimentary/free) "
            "and that the shuttle is for the airport. If only a paid shuttle is mentioned, or shuttle is unrelated to the airport, fail."
        ),
    )

    # On-site meeting or conference facilities
    meeting_node = evaluator.add_leaf(
        id=f"hotel_{index + 1}_meeting_space",
        desc="Has on-site meeting rooms or conference facilities.",
        parent=amenities_node,
        critical=True
    )
    meeting_claim = (
        f"The official/booking page for {hotel_name_for_claim} confirms the property has on-site meeting rooms or "
        f"conference/event facilities."
    )
    await evaluator.verify(
        claim=meeting_claim,
        node=meeting_node,
        sources=sources,
        additional_instruction=(
            "Accept terms like 'meeting room(s)', 'conference room(s)', 'event space', 'banquet space', 'ballroom', "
            "or a dedicated meetings/events section on the official site. Generic statements without on-site meeting "
            "space should not pass."
        ),
    )

    # On-site fitness center or gym
    fitness_node = evaluator.add_leaf(
        id=f"hotel_{index + 1}_fitness_center",
        desc="Has an on-site fitness center or gym.",
        parent=amenities_node,
        critical=True
    )
    fitness_claim = (
        f"The official/booking page for {hotel_name_for_claim} confirms the property has an on-site fitness center or gym."
    )
    await evaluator.verify(
        claim=fitness_claim,
        node=fitness_node,
        sources=sources,
        additional_instruction=(
            "Look for 'fitness center', 'gym', 'exercise room', or similar. The facility should be on-site."
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
) -> dict:
    # Initialize evaluator (root is non-critical to allow partial scoring across hotels)
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Hotels evaluated independently
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

    # Extract hotel entries from the answer
    extracted_hotels = await evaluator.extract(
        prompt=prompt_extract_hotels(),
        template_class=HotelsExtraction,
        extraction_name="hotels_extraction",
    )

    # Use up to 3 hotels; pad if fewer
    hotels: List[HotelEntry] = list(extracted_hotels.hotels[:3])
    while len(hotels) < 3:
        hotels.append(HotelEntry())

    # Build the main container node for the task (parallel aggregation)
    find_three_node = evaluator.add_parallel(
        id="find_three_hotels",
        desc="Identify three hotels in the SeaTac or Tukwila area near SEA that each meet the required amenities and provide a valid reference link.",
        parent=root,
        critical=False
    )

    # Add custom info for debugging/traceability
    evaluator.add_custom_info(
        info={
            "allowed_official_domains": sorted(list(ALLOWED_OFFICIAL_DOMAINS)),
            "allowed_major_booking_domains": sorted(list(ALLOWED_MAJOR_BOOKING_DOMAINS))
        },
        info_type="domain_whitelist",
        info_name="reference_url_domain_whitelist"
    )

    # Verify each hotel
    for idx in range(3):
        await verify_hotel(evaluator, find_three_node, hotels[idx], idx)

    # Return structured result
    return evaluator.get_summary()