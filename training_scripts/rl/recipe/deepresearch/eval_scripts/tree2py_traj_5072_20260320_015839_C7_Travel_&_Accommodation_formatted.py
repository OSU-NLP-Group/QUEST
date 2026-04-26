import asyncio
import logging
from datetime import datetime, timedelta, date
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys


# ----------------------------- Task Constants ----------------------------- #
TASK_ID = "caribbean_beach_hotel_jetblue"
TASK_DESCRIPTION = (
    "You are a Gen Z traveler (born between 1997-2012) planning a 3-night beach vacation to the Caribbean, "
    "departing from New York on JetBlue. You need to find a hotel that meets the following requirements:\n\n"
    "1. The hotel must be located in a Caribbean country or territory that JetBlue serves with direct flights from New York\n"
    "2. The destination must NOT have a Level 3 (\"Reconsider Travel\") or Level 4 (\"Do Not Travel\") US State Department travel advisory currently in effect\n"
    "3. The hotel must offer free WiFi to guests (essential for Gen Z travelers)\n"
    "4. The hotel must be beachfront or have direct beach access\n"
    "5. The hotel must have a minimum 4-star rating or equivalent quality classification\n"
    "6. The hotel must offer flexible cancellation allowing free cancellation at least 24-48 hours before check-in\n"
    "7. The total cost for a 3-night stay (room only, excluding taxes and fees) must be $600 or less\n"
    "8. The hotel must have available rooms for booking within the next 90 days (from March 20, 2026)\n"
    "9. All hotel information must be verifiable through official hotel websites or major booking platforms\n\n"
    "Additionally, preferred (non-required) features:\n"
    "- A recognized sustainability certification (Green Globe, EarthCheck, LEED, or similar)\n"
    "- Average guest rating of at least 8.0/10 or 4.0/5.0 on major booking platforms\n"
    "- At least one on-site restaurant or dining facility\n"
    "- A swimming pool (indoor or outdoor)\n"
    "- Location within 45 minutes driving time from the main international airport\n\n"
    "Identify a specific hotel that satisfies all required criteria, and provide: hotel name, destination, verification for each criterion with reference URLs, "
    "and information about any additional preferred features."
)

REFERENCE_DATE = date(2026, 3, 20)
NINETY_DAYS_DEADLINE = REFERENCE_DATE + timedelta(days=90)
NINETY_DAYS_DEADLINE_STR = NINETY_DAYS_DEADLINE.strftime("%B %-d, %Y") if hasattr(NINETY_DAYS_DEADLINE, "strftime") else "June 18, 2026"


# ----------------------------- Extraction Models ----------------------------- #
class HotelSelectionExtraction(BaseModel):
    # Core identification
    hotel_name: Optional[str] = None
    destination: Optional[str] = None

    # Required criteria evidence URLs
    route_urls: List[str] = Field(default_factory=list, description="URLs showing JetBlue direct service from NYC to destination.")
    state_advisory_urls: List[str] = Field(default_factory=list, description="URLs showing the US State Dept advisory level.")
    wifi_urls: List[str] = Field(default_factory=list, description="URLs showing free WiFi is offered.")
    beachfront_urls: List[str] = Field(default_factory=list, description="URLs showing beachfront or direct beach access.")
    rating_or_classification: Optional[str] = None
    rating_urls: List[str] = Field(default_factory=list, description="URLs showing 4-star or equivalent classification/ratings.")
    cancellation_policy: Optional[str] = None
    cancellation_urls: List[str] = Field(default_factory=list, description="URLs showing free cancellation ≥24–48h pre check-in for an eligible rate.")
    price_total_room_only_3n: Optional[str] = None
    price_urls: List[str] = Field(default_factory=list, description="URLs showing a qualifying 3-night room-only price quote (excl. taxes/fees).")
    availability_urls: List[str] = Field(default_factory=list, description="URLs demonstrating bookable availability for chosen dates.")
    example_checkin_date: Optional[str] = None
    example_checkout_date: Optional[str] = None
    room_type_or_rate_name: Optional[str] = None

    # Preferred features (optional claims + evidence)
    sustainability_certification: Optional[str] = None
    sustainability_urls: List[str] = Field(default_factory=list)

    guest_rating_value: Optional[str] = None
    guest_rating_urls: List[str] = Field(default_factory=list)

    restaurant_claim: Optional[str] = None
    restaurant_urls: List[str] = Field(default_factory=list)

    pool_claim: Optional[str] = None
    pool_urls: List[str] = Field(default_factory=list)

    airport_time_claim: Optional[str] = None
    airport_urls: List[str] = Field(default_factory=list)


# ----------------------------- Extraction Prompt ----------------------------- #
def prompt_extract_hotel_selection() -> str:
    return """
    Extract the specific hotel selection and all supporting information from the answer text. Follow these rules:
    - Do not invent any information. Extract exactly what the answer provides.
    - For each criterion, return every URL mentioned in the answer that supports that criterion.
    - For URLs, accept plain links or markdown links; return absolute URLs only.
    - If a field is not stated, set it to null or an empty list as appropriate.
    - If one URL supports multiple criteria, include it in every relevant list.

    Required fields to extract:
    1) hotel_name: The full hotel/resort name selected.
    2) destination: The Caribbean destination country/territory or island.
    3) route_urls: URLs that indicate JetBlue operates direct (nonstop) flights from New York (JFK/LGA/EWR) to the stated destination.
    4) state_advisory_urls: URLs (preferably travel.state.gov) showing that the destination is NOT Level 3 or 4.
    5) wifi_urls: URLs (hotel site or major booking platform) showing free WiFi for guests (complimentary wireless internet).
    6) beachfront_urls: URLs (hotel site or major booking platform) showing the property is beachfront or has direct beach access.
    7) rating_or_classification: A textual claim such as '4-star', '4.5/5', '8.5/10', or an equivalent classification.
    8) rating_urls: URLs supporting the 4-star (or equivalent) status (booking platform or reputable ratings source).
    9) cancellation_policy: The described flexible cancellation policy (e.g., 'free cancellation until 48 hours before check-in').
    10) cancellation_urls: URLs that show the flexible/free cancellation policy for an eligible rate.
    11) price_total_room_only_3n: The stated total for 3 nights (room-only, excluding taxes/fees), such as '$570 total for 3 nights'.
    12) price_urls: URLs (hotel or major booking platform) that show a qualifying price quote for 3 nights or nightly rate clearly leading to <= $600 total before taxes/fees.
    13) availability_urls: URLs that show the hotel is bookable (has availability) for a specific set of dates within the next 90 days from March 20, 2026.
    14) example_checkin_date: The check-in date used for the example availability/price (in any readable format if present).
    15) example_checkout_date: The check-out date used (in any readable format if present).
    16) room_type_or_rate_name: The exact room/rate label (if the answer specifies one).

    Preferred (optional) features (extract them only if the answer explicitly claims them):
    17) sustainability_certification: The named sustainability certification (e.g., 'Green Globe', 'EarthCheck', 'LEED'), else null.
    18) sustainability_urls: URLs supporting the certification claim, else empty list.
    19) guest_rating_value: A rating such as '8.4/10' or '4.3/5', else null.
    20) guest_rating_urls: URLs supporting the guest rating claim, else empty list.
    21) restaurant_claim: A claim that the property has an on-site restaurant/dining facility, else null.
    22) restaurant_urls: URLs supporting the restaurant claim, else empty list.
    23) pool_claim: A claim that the property has a swimming pool (indoor or outdoor), else null.
    24) pool_urls: URLs supporting the pool claim, else empty list.
    25) airport_time_claim: A claim that the property is within ~45 minutes' drive of the main international airport, else null.
    26) airport_urls: URLs supporting the airport time/distance claim, else empty list.
    """


# ----------------------------- Utility Functions ----------------------------- #
def _domain_from_url(url: str) -> str:
    try:
        netloc = urlparse(url).netloc.lower()
        if netloc.startswith("www."):
            netloc = netloc[4:]
        return netloc
    except Exception:
        return ""


def _path_from_url(url: str) -> str:
    try:
        return urlparse(url).path.lower()
    except Exception:
        return ""


def _is_recognized_booking_domain(domain: str, path: str = "") -> bool:
    known_bases = [
        # OTAs / Meta
        "booking.com",
        "expedia.com",
        "hotels.com",
        "agoda.com",
        "priceline.com",
        "orbitz.com",
        "travelocity.com",
        "trip.com",
        "tripadvisor.com",
        "kayak.com",
        "trivago.com",
        "google.com",  # Google Hotels
        # Major chains / groups
        "marriott.com",
        "hilton.com",
        "hyatt.com",
        "ihg.com",
        "accor.com",
        "wyndhamhotels.com",
        "radissonhotels.com",
        "bestwestern.com",
        "choicehotels.com",
        # Common Caribbean brands/groups
        "melia.com",
        "barcelo.com",
        "riu.com",
        "iberostar.com",
        "palladiumhotelgroup.com",
        "sandals.com",
        "beaches.com",
        "palaceresorts.com",
        "karismahotels.com",
        "playaresorts.com",
        "hardrockhotels.com",
        "hyattinclusivecollection.com",
        "fiestamericana.com",
        "occidentalhotels.com",
    ]
    if any(domain == base or domain.endswith("." + base) for base in known_bases):
        if domain == "google.com":
            # Accept only Google Hotels pages
            return path.startswith("/travel/hotels")
        return True
    return False


def _is_official_hotel_domain(domain: str, hotel_name: Optional[str]) -> bool:
    if not hotel_name or not domain:
        return False
    name = "".join(ch.lower() if ch.isalnum() or ch.isspace() else " " for ch in hotel_name)
    tokens = [t for t in name.split() if len(t) >= 3]
    # Consider it official if at least one significant token is present in domain
    return any(t in domain for t in tokens)


def _has_verifiable_source(urls: List[str], hotel_name: Optional[str]) -> bool:
    for u in urls:
        d = _domain_from_url(u)
        p = _path_from_url(u)
        if _is_recognized_booking_domain(d, p) or _is_official_hotel_domain(d, hotel_name):
            return True
    return False


def _joined_urls(urls: List[str]) -> List[str]:
    # Deduplicate while preserving order
    seen = set()
    out = []
    for u in urls:
        if u and u not in seen:
            seen.add(u)
            out.append(u)
    return out


# ----------------------------- Verification Builders ----------------------------- #
async def build_identification_nodes(evaluator: Evaluator, parent, info: HotelSelectionExtraction) -> None:
    node = evaluator.add_parallel(
        id="provide_hotel_identification",
        desc="Response identifies a specific hotel and its Caribbean destination.",
        parent=parent,
        critical=True,
    )

    evaluator.add_custom_node(
        result=bool(info.hotel_name and info.hotel_name.strip()),
        id="hotel_name_provided",
        desc="Hotel name is explicitly provided.",
        parent=node,
        critical=True,
    )

    evaluator.add_custom_node(
        result=bool(info.destination and info.destination.strip()),
        id="destination_provided",
        desc="Caribbean destination (country/territory) is explicitly provided.",
        parent=node,
        critical=True,
    )


async def build_required_criteria_nodes(evaluator: Evaluator, parent, info: HotelSelectionExtraction) -> None:
    node = evaluator.add_parallel(
        id="meets_required_criteria_with_verification",
        desc="Hotel satisfies each required criterion; each criterion includes at least one supporting reference URL appropriate to that criterion.",
        parent=parent,
        critical=True,
    )

    hotel = info.hotel_name or "the selected hotel"
    dest = info.destination or "the stated destination"

    # 1) JetBlue direct from NY
    n = evaluator.add_leaf(
        id="direct_jetblue_from_ny",
        desc="Destination is served by JetBlue with direct flights from New York (JFK/LGA/EWR), and at least one supporting URL is provided (e.g., JetBlue or a major flight/route listing).",
        parent=node,
        critical=True,
    )
    claim = f"JetBlue operates direct (nonstop) flights from New York City area airports (JFK, LGA, or EWR) to {dest}."
    await evaluator.verify(
        claim=claim,
        node=n,
        sources=_joined_urls(info.route_urls),
        additional_instruction=(
            "Use the provided URLs (e.g., JetBlue route map/schedule or reputable route listings). "
            "Confirm that the service is JetBlue (B6) and is direct/nonstop from a NYC airport (JFK/LGA/EWR) to the stated Caribbean destination. "
            "If no valid URL is provided, mark as not supported."
        ),
    )

    # 2) Not Level 3 or 4 advisory
    n = evaluator.add_leaf(
        id="no_level3_or_4_advisory",
        desc="Destination is NOT under a US State Department Level 3 or Level 4 travel advisory in the stated time context, with at least one supporting URL (preferably travel.state.gov).",
        parent=node,
        critical=True,
    )
    claim = f"As of March 20, 2026, the US State Department advisory for {dest} is NOT Level 3 or Level 4 (i.e., Level 1 or Level 2 is acceptable)."
    await evaluator.verify(
        claim=claim,
        node=n,
        sources=_joined_urls(info.state_advisory_urls),
        additional_instruction=(
            "Prefer travel.state.gov pages. If the destination is listed as Level 3 or Level 4, mark Incorrect. "
            "If the page shows Level 1 or Level 2, mark Correct. If URLs are missing or irrelevant, mark Incorrect."
        ),
    )

    # 3) Free WiFi
    n = evaluator.add_leaf(
        id="free_wifi",
        desc="Hotel offers free WiFi to guests, with at least one supporting URL from the official hotel site or a major booking platform.",
        parent=node,
        critical=True,
    )
    claim = f"{hotel} offers free WiFi (complimentary wireless internet) to guests."
    await evaluator.verify(
        claim=claim,
        node=n,
        sources=_joined_urls(info.wifi_urls),
        additional_instruction=(
            "Accept terms such as 'complimentary Wi-Fi', 'free wireless internet', or equivalent. "
            "Evidence should come from the official hotel website or a major booking platform. "
            "If URLs are missing or do not state free WiFi, mark Incorrect."
        ),
    )

    # 4) Beachfront or direct beach access
    n = evaluator.add_leaf(
        id="beachfront_or_beach_access",
        desc="Hotel is beachfront or has direct beach access, with at least one supporting URL from the official hotel site or a major booking platform.",
        parent=node,
        critical=True,
    )
    claim = f"{hotel} is beachfront or has direct beach access."
    await evaluator.verify(
        claim=claim,
        node=n,
        sources=_joined_urls(info.beachfront_urls),
        additional_instruction=(
            "Look for explicit phrases like 'beachfront', 'on the beach', or 'direct beach access'. "
            "Statements like 'near the beach' or 'short walk to beach' do not satisfy 'beachfront/direct access'. "
            "If URLs are missing or inconclusive, mark Incorrect."
        ),
    )

    # 5) Minimum 4-star or equivalent
    n = evaluator.add_leaf(
        id="min_4_star_or_equivalent",
        desc="Hotel has a minimum 4-star rating or equivalent quality classification, with at least one supporting URL from a major booking platform or reputable ratings source.",
        parent=node,
        critical=True,
    )
    claim = (
        f"{hotel} meets a minimum 4-star level or an equivalent quality classification. "
        f"Claimed rating/classification: {info.rating_or_classification or 'unspecified'}."
    )
    await evaluator.verify(
        claim=claim,
        node=n,
        sources=_joined_urls(info.rating_urls),
        additional_instruction=(
            "Accept any of the following as equivalent: an explicit 4-star (or higher) classification; an average rating ≥ 8.0/10; or ≥ 4.0/5.0 on a reputable platform. "
            "If the source shows below these thresholds or is not credible, mark Incorrect. If URLs are missing, mark Incorrect."
        ),
    )

    # 6) Flexible free cancellation (≥24–48h before check-in)
    n = evaluator.add_leaf(
        id="free_cancellation_24_48h",
        desc="Hotel offers flexible cancellation allowing free cancellation at least 24–48 hours before check-in, with at least one supporting URL showing the cancellation terms for an eligible rate.",
        parent=node,
        critical=True,
    )
    claim = (
        f"{hotel} offers a flexible rate with free cancellation at least 24–48 hours before check-in. "
        f"Claimed policy text: {info.cancellation_policy or 'unspecified'}."
    )
    await evaluator.verify(
        claim=claim,
        node=n,
        sources=_joined_urls(info.cancellation_urls),
        additional_instruction=(
            "Look for rate terms indicating free cancellation until a point ≥24 hours prior to check-in (48h also satisfies). "
            "If only non-refundable rates are shown, mark Incorrect. If URLs are missing, mark Incorrect."
        ),
    )

    # 7) Price for 3 nights ≤ $600 (room-only, excl. taxes/fees)
    n = evaluator.add_leaf(
        id="price_3_nights_600_or_less",
        desc="Total room-only price for 3 nights (excluding taxes/fees) is $600 or less, with at least one supporting URL showing a qualifying price quote for specific dates/rate.",
        parent=node,
        critical=True,
    )
    checkin = info.example_checkin_date or "the shown check-in date"
    checkout = info.example_checkout_date or "the shown check-out date"
    claim = (
        f"For dates {checkin} to {checkout}, the total room-only price (excluding taxes/fees) for 3 nights at {hotel} is $600 or less. "
        f"Stated total: {info.price_total_room_only_3n or 'unspecified'}."
    )
    await evaluator.verify(
        claim=claim,
        node=n,
        sources=_joined_urls(info.price_urls),
        additional_instruction=(
            "Confirm the subtotal/base rate across exactly 3 nights is ≤ $600 before taxes/fees. "
            "If only nightly rates are shown, multiply by 3 to check. If currency is USD on the page, use it; "
            "if not clearly USD or not enough info to confirm ≤ $600, mark Incorrect. If URLs are missing, mark Incorrect."
        ),
    )

    # 8) Availability within next 90 days of Mar 20, 2026
    n = evaluator.add_leaf(
        id="availability_next_90_days",
        desc="Hotel has bookable availability within the next 90 days relative to the stated reference date, with at least one supporting URL demonstrating availability.",
        parent=node,
        critical=True,
    )
    claim = (
        f"{hotel} has bookable availability for the example dates {checkin} to {checkout}, which are within 90 days of March 20, 2026 (i.e., no later than {NINETY_DAYS_DEADLINE_STR})."
    )
    await evaluator.verify(
        claim=claim,
        node=n,
        sources=_joined_urls(info.availability_urls),
        additional_instruction=(
            "Check that the booking engine or platform shows available rooms for the specified dates. "
            "Also ensure those dates fall within 90 days of March 20, 2026. If no availability is shown or URLs are missing, mark Incorrect."
        ),
    )

    # 9) Verifiable sources used (official hotel site or major booking platform per hotel-specific claim)
    # Check internally (domain-based) for the hotel-specific claims: wifi, beachfront, rating, cancellation, price, availability
    hotel_specific_groups = {
        "wifi": info.wifi_urls,
        "beachfront": info.beachfront_urls,
        "rating": info.rating_urls,
        "cancellation": info.cancellation_urls,
        "price": info.price_urls,
        "availability": info.availability_urls,
    }
    all_ok = True
    for _, urls in hotel_specific_groups.items():
        if not _has_verifiable_source(urls, info.hotel_name):
            all_ok = False
            break

    evaluator.add_custom_node(
        result=all_ok,
        id="verifiable_sources_used",
        desc="For hotel-specific claims (e.g., WiFi, beachfront access, rating/classification, cancellation, price, availability), at least one cited source per claim is an official hotel website or a major booking platform; additional URLs are allowed.",
        parent=node,
        critical=True,
    )


async def build_preferred_features_nodes(evaluator: Evaluator, parent, info: HotelSelectionExtraction) -> None:
    node = evaluator.add_parallel(
        id="preferred_features_reported",
        desc="Response reports any of the preferred (non-required) features the hotel offers, and provides evidence for any preferred feature it claims the hotel has.",
        parent=parent,
        critical=False,
    )
    hotel = info.hotel_name or "the selected hotel"
    dest = info.destination or "the destination"

    # Sustainability certification (if claimed)
    if info.sustainability_certification and info.sustainability_certification.strip() and info.sustainability_urls:
        leaf = evaluator.add_leaf(
            id="sustainability_certification_if_claimed",
            desc="If the response claims a sustainability certification, it provides a supporting URL; otherwise this check passes.",
            parent=node,
            critical=False,
        )
        claim = (
            f"{hotel} holds a recognized sustainability certification (e.g., Green Globe, EarthCheck, LEED or similar): "
            f"{info.sustainability_certification}."
        )
        await evaluator.verify(
            claim=claim,
            node=leaf,
            sources=_joined_urls(info.sustainability_urls),
            additional_instruction="Verify that the cited page explicitly indicates a formal sustainability certification for the property.",
        )
    else:
        evaluator.add_custom_node(
            result=True,
            id="sustainability_certification_if_claimed",
            desc="If the response claims a sustainability certification, it provides a supporting URL; otherwise this check passes.",
            parent=node,
            critical=False,
        )

    # Guest rating threshold/value (if claimed)
    rating_urls_all = _joined_urls((info.guest_rating_urls or []) + (info.rating_urls or []))
    if info.guest_rating_value and info.guest_rating_value.strip() and rating_urls_all:
        leaf = evaluator.add_leaf(
            id="guest_rating_if_claimed",
            desc="If the response claims an average guest rating threshold/value, it provides a supporting URL; otherwise this check passes.",
            parent=node,
            critical=False,
        )
        claim = (
            f"{hotel} has an average guest rating meeting or exceeding the stated threshold (e.g., ≥ 8.0/10 or ≥ 4.0/5.0). "
            f"Claimed value: {info.guest_rating_value}."
        )
        await evaluator.verify(
            claim=claim,
            node=leaf,
            sources=rating_urls_all,
            additional_instruction="Confirm the rating value/scale on the page and that it meets or exceeds 8.0/10 or 4.0/5.0.",
        )
    else:
        evaluator.add_custom_node(
            result=True,
            id="guest_rating_if_claimed",
            desc="If the response claims an average guest rating threshold/value, it provides a supporting URL; otherwise this check passes.",
            parent=node,
            critical=False,
        )

    # On-site restaurant (if claimed)
    if info.restaurant_claim and info.restaurant_claim.strip() and info.restaurant_urls:
        leaf = evaluator.add_leaf(
            id="on_site_restaurant_if_claimed",
            desc="If the response claims an on-site restaurant/dining facility, it provides a supporting URL; otherwise this check passes.",
            parent=node,
            critical=False,
        )
        claim = f"{hotel} has at least one on-site restaurant or dining facility."
        await evaluator.verify(
            claim=claim,
            node=leaf,
            sources=_joined_urls(info.restaurant_urls),
            additional_instruction="Verify the property's official page or major platform lists an on-site restaurant/dining option.",
        )
    else:
        evaluator.add_custom_node(
            result=True,
            id="on_site_restaurant_if_claimed",
            desc="If the response claims an on-site restaurant/dining facility, it provides a supporting URL; otherwise this check passes.",
            parent=node,
            critical=False,
        )

    # Swimming pool (if claimed)
    if info.pool_claim and info.pool_claim.strip() and info.pool_urls:
        leaf = evaluator.add_leaf(
            id="pool_if_claimed",
            desc="If the response claims a swimming pool, it provides a supporting URL; otherwise this check passes.",
            parent=node,
            critical=False,
        )
        claim = f"{hotel} has a swimming pool (indoor or outdoor)."
        await evaluator.verify(
            claim=claim,
            node=leaf,
            sources=_joined_urls(info.pool_urls),
            additional_instruction="Verify that the property lists/presents a pool amenity on the provided page(s).",
        )
    else:
        evaluator.add_custom_node(
            result=True,
            id="pool_if_claimed",
            desc="If the response claims a swimming pool, it provides a supporting URL; otherwise this check passes.",
            parent=node,
            critical=False,
        )

    # Airport within ~45 minutes (if claimed)
    if info.airport_time_claim and info.airport_time_claim.strip() and info.airport_urls:
        leaf = evaluator.add_leaf(
            id="airport_within_45_min_if_claimed",
            desc="If the response claims the hotel is within ~45 minutes' drive of the main international airport, it provides a supporting basis/URL; otherwise this check passes.",
            parent=node,
            critical=False,
        )
        claim = f"{hotel} is within approximately 45 minutes' drive from the main international airport serving {dest}."
        await evaluator.verify(
            claim=claim,
            node=leaf,
            sources=_joined_urls(info.airport_urls),
            additional_instruction="Check travel time/distance info (e.g., hotel page or reputable mapping/distance references shown in the provided URLs).",
        )
    else:
        evaluator.add_custom_node(
            result=True,
            id="airport_within_45_min_if_claimed",
            desc="If the response claims the hotel is within ~45 minutes' drive of the main international airport, it provides a supporting basis/URL; otherwise this check passes.",
            parent=node,
            critical=False,
        )


# ----------------------------- Main Evaluation ----------------------------- #
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
        strategy=AggregationStrategy.SEQUENTIAL,
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

    # IMPORTANT: Root must be non-critical to allow a non-critical preferred-features branch.
    root.critical = False

    # Extraction
    extracted = await evaluator.extract(
        prompt=prompt_extract_hotel_selection(),
        template_class=HotelSelectionExtraction,
        extraction_name="hotel_selection_extraction",
    )

    # Record ground-truth constraints for context
    evaluator.add_ground_truth(
        {
            "required_criteria": [
                "JetBlue direct from NYC to destination",
                "Destination NOT Level 3 or 4 (US State Dept)",
                "Free WiFi",
                "Beachfront/direct beach access",
                "≥ 4-star or equivalent classification",
                "Free cancellation ≥24–48h before check-in",
                "3-night room-only total ≤ $600 (excl. taxes/fees)",
                f"Availability within 90 days from {REFERENCE_DATE.isoformat()} (deadline ≈ {NINETY_DAYS_DEADLINE.isoformat()})",
                "Hotel info verifiable via official site or major booking platforms",
            ],
            "preferred_features": [
                "Sustainability certification",
                "Avg guest rating ≥ 8.0/10 or ≥ 4.0/5.0",
                "On-site restaurant",
                "Swimming pool",
                "≤ ~45 minutes from main international airport",
            ],
        },
        gt_type="rubric_requirements",
    )

    # Build verification tree
    await build_identification_nodes(evaluator, root, extracted)
    await build_required_criteria_nodes(evaluator, root, extracted)
    await build_preferred_features_nodes(evaluator, root, extracted)

    return evaluator.get_summary()