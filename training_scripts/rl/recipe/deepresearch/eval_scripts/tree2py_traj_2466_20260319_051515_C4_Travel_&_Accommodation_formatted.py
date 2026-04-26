import asyncio
import logging
from typing import Optional, List, Dict, Any, Tuple
from urllib.parse import urlparse

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator, AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys


# --------------------------------------------------------------------------- #
# Task metadata                                                               #
# --------------------------------------------------------------------------- #
TASK_ID = "atl_airport_hotels_amenities"
TASK_DESCRIPTION = (
    "I'm planning a trip to Atlanta in late March 2026 and need to book a hotel near Hartsfield-Jackson Atlanta "
    "International Airport for a one-night stay during a layover. Due to my travel circumstances, I need a hotel that "
    "offers maximum convenience and flexibility. Please identify three hotels near ATL airport that meet all of the "
    "following requirements:\n\n"
    "1. Free 24-hour airport shuttle service (available at all hours of the day and night)\n"
    "2. Complimentary breakfast included with the room\n"
    "3. On-site fitness center\n"
    "4. Swimming pool (either indoor or outdoor)\n"
    "5. Free Wi-Fi access\n"
    "6. 24-hour front desk service\n"
    "7. Free cancellation policy (allowing cancellation up to at least 24 hours before check-in without penalty)\n"
    "8. Located in close proximity to Hartsfield-Jackson Atlanta International Airport\n\n"
    "For each hotel, please provide the hotel name and the official website URL where these amenities are confirmed."
)

# Some well-known third-party/OTA/aggregator domains (not official property/brand sites)
AGGREGATOR_DOMAINS = {
    "booking.com",
    "expedia.com",
    "hotels.com",
    "orbitz.com",
    "travelocity.com",
    "priceline.com",
    "agoda.com",
    "tripadvisor.com",
    "trivago.com",
    "kayak.com",
    "google.com",
    "maps.google.com",
    "bing.com",
    "yelp.com",
    "airbnb.com",
    "facebook.com",
    "instagram.com",
    "twitter.com",
    "x.com",
    "hotwire.com",
}


# --------------------------------------------------------------------------- #
# Extraction models                                                           #
# --------------------------------------------------------------------------- #
class HotelItem(BaseModel):
    name: Optional[str] = None
    official_url: Optional[str] = None
    extra_official_urls: List[str] = Field(default_factory=list)


class HotelsExtraction(BaseModel):
    hotels: List[HotelItem] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_hotels() -> str:
    return """
    From the answer, extract up to three hotel recommendations intended to satisfy the user's request.
    For each hotel, extract:
    - name: the hotel's name as written in the answer (string)
    - official_url: a single official hotel or brand website URL that represents the hotel and where amenities/policies can be verified (string)
    - extra_official_urls: any additional official, same-brand/property URLs from the answer that may directly mention amenities/policies (array of strings)
    
    Rules:
    - Only include URLs that are official brand/property pages (e.g., marriott.com, hilton.com, hyatt.com, ihg.com, choicehotels.com, wyndhamhotels.com, bestwestern.com, sonesta.com, druryhotels.com, laquinta.com, etc.).
    - Do NOT include third-party aggregators or OTAs (e.g., booking.com, expedia.com, hotels.com, tripadvisor.com, trivago.com, kayak.com).
    - If the answer includes more than 3 hotels, extract only the first three in order.
    - If a field is missing, set it to null (or [] for the array).
    - Do not invent any URLs or hotel names.
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def _is_http_url(url: Optional[str]) -> bool:
    if not url or not isinstance(url, str):
        return False
    try:
        p = urlparse(url.strip())
    except Exception:
        return False
    return p.scheme in ("http", "https") and bool(p.netloc)


def _domain_is_blocked(netloc: str, blocked: set) -> bool:
    net = netloc.lower().lstrip()
    if net.startswith("www."):
        net = net[4:]
    for dom in blocked:
        d = dom.lower()
        if net == d or net.endswith("." + d):
            return True
    return False


def is_official_url(url: Optional[str]) -> bool:
    if not _is_http_url(url):
        return False
    netloc = urlparse(url).netloc.lower()
    if netloc.startswith("www."):
        netloc = netloc[4:]
    # Disallow well-known aggregator/OTA/social/map domains
    if _domain_is_blocked(netloc, AGGREGATOR_DOMAINS):
        return False
    # Otherwise treat as official (covers brand domains like marriott.com, hilton.com, etc.)
    return True


def filter_official_urls(urls: List[str]) -> List[str]:
    kept = []
    for u in urls:
        if is_official_url(u):
            kept.append(u)
    # De-duplicate while preserving order
    seen = set()
    unique = []
    for u in kept:
        if u not in seen:
            unique.append(u)
            seen.add(u)
    return unique


def ensure_first_three(hotels: List[HotelItem]) -> List[HotelItem]:
    # Guarantee we only use first 3 items (pad with empty if fewer)
    first3 = hotels[:3]
    while len(first3) < 3:
        first3.append(HotelItem())
    return first3


def names_are_three_distinct(hotels_first3: List[HotelItem]) -> bool:
    # Passes if the first 3 extracted hotels have 3 nonempty, distinct names (case-insensitive)
    names = [h.name.strip() for h in hotels_first3 if h.name and isinstance(h.name, str) and h.name.strip()]
    if len(names) < 3:
        return False
    normalized = [n.lower() for n in names[:3]]
    return len(set(normalized)) == 3


def hotel_sources(h: HotelItem) -> List[str]:
    seeds = []
    if h.official_url:
        seeds.append(h.official_url)
    if h.extra_official_urls:
        seeds.extend(h.extra_official_urls)
    return filter_official_urls(seeds)


# --------------------------------------------------------------------------- #
# Verification builders                                                       #
# --------------------------------------------------------------------------- #
async def verify_single_hotel(evaluator: Evaluator, parent_node, h: HotelItem, idx: int) -> None:
    """
    Build the verification subtree for one hotel with all required amenity checks.
    """
    hotel_num = idx + 1
    hotel_node = evaluator.add_parallel(
        id=f"Hotel_{hotel_num}",
        desc=f"Hotel {hotel_num} satisfies all requirements and includes required identifying/reference information.",
        parent=parent_node,
        critical=False
    )

    # Name provided (critical)
    name_ok = bool(h.name and isinstance(h.name, str) and h.name.strip())
    evaluator.add_custom_node(
        result=name_ok,
        id=f"Hotel_{hotel_num}_Name_Provided",
        desc="The hotel name is provided.",
        parent=hotel_node,
        critical=True
    )

    # Official website provided and is official (critical)
    official_ok = bool(h.official_url and is_official_url(h.official_url))
    evaluator.add_custom_node(
        result=official_ok,
        id=f"Hotel_{hotel_num}_Official_Website_URL_Provided",
        desc="An official hotel website URL is provided (not a third-party aggregator) where amenities/policies can be verified.",
        parent=hotel_node,
        critical=True
    )

    # Prepare sources (will be empty if official_url missing; amenity leaves will be auto-skipped due to critical sibling failure)
    srcs = hotel_sources(h)

    # Create amenity leaves
    leaf_nodes_and_claims: List[Tuple[str, Any, Any, str]] = []

    # Free 24-hour airport shuttle
    shuttle_node = evaluator.add_leaf(
        id=f"Hotel_{hotel_num}_Free_Airport_Shuttle_24h",
        desc="The hotel offers a free airport shuttle that operates 24 hours.",
        parent=hotel_node,
        critical=True
    )
    shuttle_claim = (
        "The hotel's official website confirms a complimentary (free) airport shuttle that operates 24 hours a day "
        "(24/7, around the clock) to/from Hartsfield-Jackson Atlanta International Airport (ATL)."
    )
    shuttle_ins = (
        "Treat 'complimentary', 'free', or 'no charge' as equivalent to free. Accept expressions such as '24-hour', "
        "'24/7', 'around the clock'. If the shuttle has limited hours or doesn't clearly run overnight, this should FAIL."
    )
    leaf_nodes_and_claims.append((shuttle_claim, srcs, shuttle_node, shuttle_ins))

    # Complimentary breakfast included
    breakfast_node = evaluator.add_leaf(
        id=f"Hotel_{hotel_num}_Complimentary_Breakfast",
        desc="The hotel offers complimentary breakfast included with the room.",
        parent=hotel_node,
        critical=True
    )
    breakfast_claim = (
        "The hotel's official website states that complimentary breakfast is included with the room rate (e.g., 'free "
        "hot breakfast', 'complimentary breakfast')."
    )
    breakfast_ins = (
        "Confirm that breakfast is included without an extra fee. Phrases like 'free breakfast' or 'complimentary "
        "breakfast' qualify. If breakfast is only available for purchase, this should FAIL."
    )
    leaf_nodes_and_claims.append((breakfast_claim, srcs, breakfast_node, breakfast_ins))

    # Fitness center
    fitness_node = evaluator.add_leaf(
        id=f"Hotel_{hotel_num}_Fitness_Center",
        desc="The hotel has an on-site fitness center available to guests.",
        parent=hotel_node,
        critical=True
    )
    fitness_claim = "The hotel's official website lists an on-site fitness center (gym) available to guests."
    fitness_ins = "Accept synonyms like 'fitness center', 'gym', or 'health club' if on property."
    leaf_nodes_and_claims.append((fitness_claim, srcs, fitness_node, fitness_ins))

    # Swimming pool
    pool_node = evaluator.add_leaf(
        id=f"Hotel_{hotel_num}_Swimming_Pool",
        desc="The hotel has a swimming pool (indoor or outdoor).",
        parent=hotel_node,
        critical=True
    )
    pool_claim = "The hotel's official website lists a swimming pool (indoor or outdoor) on property."
    pool_ins = "Any on-property pool qualifies (indoor or outdoor)."
    leaf_nodes_and_claims.append((pool_claim, srcs, pool_node, pool_ins))

    # Free Wi-Fi
    wifi_node = evaluator.add_leaf(
        id=f"Hotel_{hotel_num}_Free_WiFi",
        desc="The hotel offers free Wi-Fi access.",
        parent=hotel_node,
        critical=True
    )
    wifi_claim = "The hotel's official website states that Wi‑Fi is complimentary (free) for guests."
    wifi_ins = (
        "Accept terms like 'complimentary WiFi' or 'free Wi-Fi'. If Wi-Fi is only paid/tiered without a free option for "
        "guests, this should FAIL."
    )
    leaf_nodes_and_claims.append((wifi_claim, srcs, wifi_node, wifi_ins))

    # 24-hour front desk
    frontdesk_node = evaluator.add_leaf(
        id=f"Hotel_{hotel_num}_24Hour_Front_Desk",
        desc="The hotel provides 24-hour front desk service.",
        parent=hotel_node,
        critical=True
    )
    frontdesk_claim = "The hotel's official website states that a 24‑hour front desk (reception) is available."
    frontdesk_ins = "Look for '24-hour front desk', '24/7 reception', or similar phrasing."
    leaf_nodes_and_claims.append((frontdesk_claim, srcs, frontdesk_node, frontdesk_ins))

    # Free cancellation up to at least 24h before check-in
    cancel_node = evaluator.add_leaf(
        id=f"Hotel_{hotel_num}_Free_Cancellation_24h",
        desc="The hotel allows free cancellation up to at least 24 hours before check-in without penalty.",
        parent=hotel_node,
        critical=True
    )
    cancel_claim = (
        "The hotel's official website describes a cancellation policy or commonly available flexible rate that allows "
        "free cancellation at least 24 hours before check‑in (no penalty)."
    )
    cancel_ins = (
        "Look for 'free cancellation', 'cancel by 6 PM day before arrival', '24 hours before arrival', or similar. "
        "If cancellation terms are variable across rates, it's acceptable as long as a standard/flexible rate offered "
        "by the hotel provides free cancellation ≥24 hours before arrival."
    )
    leaf_nodes_and_claims.append((cancel_claim, srcs, cancel_node, cancel_ins))

    # Airport proximity
    proximity_node = evaluator.add_leaf(
        id=f"Hotel_{hotel_num}_Airport_Proximity",
        desc="The hotel is located near Hartsfield-Jackson Atlanta International Airport (ATL).",
        parent=hotel_node,
        critical=True
    )
    proximity_claim = (
        "The hotel's official website indicates the property is near Hartsfield‑Jackson Atlanta International Airport "
        "(ATL), e.g., explicitly mentions 'near Atlanta Airport', 'near Hartsfield‑Jackson', or states a short distance/shuttle."
    )
    proximity_ins = (
        "Prefer explicit mentions of 'Atlanta Airport', 'Hartsfield‑Jackson', 'ATL', 'airport area', or stated distance/time "
        "to the airport. If no airport reference appears at all, this should FAIL."
    )
    leaf_nodes_and_claims.append((proximity_claim, srcs, proximity_node, proximity_ins))

    # Batch verify all amenity claims
    await evaluator.batch_verify(leaf_nodes_and_claims)


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
    # Initialize evaluator (root non-critical to allow partial scoring across hotels)
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

    # Extract hotels
    extracted = await evaluator.extract(
        prompt=prompt_extract_hotels(),
        template_class=HotelsExtraction,
        extraction_name="hotels_extraction",
    )

    hotels_first3 = ensure_first_three(extracted.hotels)

    # Custom info: URL quality diagnostics
    url_quality = []
    for i, h in enumerate(hotels_first3):
        url_quality.append({
            "index": i + 1,
            "name": h.name,
            "official_url": h.official_url,
            "official_url_is_official": bool(is_official_url(h.official_url)) if h.official_url else False,
            "extra_official_urls_kept": filter_official_urls(h.extra_official_urls),
        })
    evaluator.add_custom_info({"url_checks": url_quality}, info_type="diagnostics", info_name="url_quality")

    # Provide_Three_Distinct_Hotels (critical leaf under root)
    evaluator.add_custom_node(
        result=names_are_three_distinct(hotels_first3),
        id="Provide_Three_Distinct_Hotels",
        desc="The solution identifies exactly three distinct hotels (not duplicates/aliases of the same property).",
        parent=root,
        critical=True
    )

    # Build subtrees for each of the three hotels (non-critical groups; each group enforces its own critical leaves)
    for i in range(3):
        await verify_single_hotel(evaluator, root, hotels_first3[i], i)

    # Return evaluation summary
    return evaluator.get_summary()