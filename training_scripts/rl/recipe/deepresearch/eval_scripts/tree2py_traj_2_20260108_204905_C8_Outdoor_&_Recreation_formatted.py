import asyncio
import logging
import re
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "ca_state_parks_camping_q2_2026"
TASK_DESCRIPTION = (
    "I'm planning four different camping trips to California State Parks between March and May 2026, each with different needs. "
    "Help me find four suitable campgrounds (one for each trip), providing complete reservation and facility information for each.\n\n"
    "Trip 1: I need an ADA-accessible campsite in Northern California (north of the San Francisco Bay Area). The campground must allow my service dog on leash and have standard camping amenities like picnic tables and fire rings.\n\n"
    "Trip 2: I'm traveling in a 35-foot RV along the Southern California coast and want to stay at a beachside state park. The campground must accommodate my RV size and allow my dog on leash in the campground area.\n\n"
    "Trip 3: I want a campground with modern conveniences—specifically electric hookups at the campsite, flush toilets, and hot showers available. My dog travels with me, and I need standard camping amenities. Any location in California is fine.\n\n"
    "Trip 4: I prefer a more primitive camping experience with hike-in or bike-in campsites (not standard drive-in sites). The park should allow dogs on leash in the campground and have basic camping facilities.\n\n"
    "For each campground, please provide:\n"
    "- The state park name and specific campground name\n"
    "- A direct URL to make reservations (via ReserveCalifornia.com or park contact)\n"
    "- Confirmation that it meets all my stated requirements\n"
    "- Key reservation details (advance booking window, reservation fees)\n"
)


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class CampgroundItem(BaseModel):
    """One campground entry, ideally mapped to Trip 1..4."""
    trip_id: Optional[int] = None
    park_name: Optional[str] = None
    campground_name: Optional[str] = None
    reservation_url: Optional[str] = None
    other_urls: List[str] = Field(default_factory=list)

    # Free-text fields extracted from the answer to support various checks.
    location_text: Optional[str] = None
    ada_features_text: Optional[str] = None
    dog_policy_text: Optional[str] = None
    amenities_text: Optional[str] = None
    rv_length_text: Optional[str] = None
    hookups_text: Optional[str] = None
    toilets_text: Optional[str] = None
    showers_text: Optional[str] = None
    hike_bike_text: Optional[str] = None

    reservation_instructions: Optional[str] = None
    booking_window_text: Optional[str] = None
    reservation_fee_text: Optional[str] = None


class CampgroundsExtraction(BaseModel):
    campgrounds: List[CampgroundItem] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_campgrounds() -> str:
    return """
    Extract up to four distinct California State Parks campground recommendations presented in the answer, one per trip (Trip 1–Trip 4). For each extracted campground, provide the following fields:

    - trip_id: The trip number (1, 2, 3, or 4). Infer from labels like "Trip 1", "Trip 2", etc. If not explicitly labeled, assign by order of appearance: the first campground is trip_id=1, second is 2, third is 3, fourth is 4.
    - park_name: The official California State Park name (e.g., "Humboldt Lagoons State Park").
    - campground_name: The specific campground name within that park (e.g., "Dry Lagoon Campground"). If not provided, use null.
    - reservation_url: A direct booking URL (prefer ReserveCalifornia.com). If the answer instead provides a clear official booking page or contact to reserve, include that URL here. If none is given, set to null.
    - other_urls: ALL other URLs cited for this campground (e.g., park pages on parks.ca.gov, campground info pages, dog policy pages). Only include URLs explicitly present in the answer.

    To help verification for constraints, also extract any relevant snippets from the answer (verbatim or close paraphrase) into the following fields when present:
    - location_text: Any statement about location or region (e.g., Northern California, Southern California coast, county, region).
    - ada_features_text: Any statement about ADA accessible sites.
    - dog_policy_text: Any statement about dog policy (e.g., dogs allowed on leash in campground).
    - amenities_text: Any statement about campsite standard amenities (e.g., picnic table, fire ring).
    - rv_length_text: Any statement about RV length limits or accommodation (e.g., "max 35 ft").
    - hookups_text: Any statement about electrical hookups available at campsites.
    - toilets_text: Any statement about toilets (e.g., flush toilets).
    - showers_text: Any statement about showers (e.g., hot showers, coin-operated showers).
    - hike_bike_text: Any statement about hike-in or bike-in sites distinct from standard drive-in sites.

    Reservation details to confirm inclusion in the answer:
    - reservation_instructions: If the answer describes how to reserve (e.g., "first-come-first-served", phone, office, ReserveCalifornia process), capture it here.
    - booking_window_text: Any text that mentions the advance booking window. We are specifically looking for the 6-month window for ReserveCalifornia. If present anywhere in the answer for this campground, copy it here.
    - reservation_fee_text: Any text that mentions the non-refundable reservation fee (we expect $8.25). If present, copy it here.

    Return a JSON with field "campgrounds": an array of at most 4 objects with the exact schema above. Do not invent any URLs or facts not present in the answer. If a field is not present for a campground, use null (or empty list for other_urls).
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def _collect_sources(item: CampgroundItem) -> List[str]:
    urls: List[str] = []
    if item.reservation_url and isinstance(item.reservation_url, str) and item.reservation_url.strip():
        urls.append(item.reservation_url.strip())
    for u in item.other_urls or []:
        if isinstance(u, str) and u.strip():
            if u.strip() not in urls:
                urls.append(u.strip())
    return urls


def _exists_names(item: CampgroundItem) -> bool:
    return bool(item and item.park_name and item.park_name.strip() and item.campground_name and item.campground_name.strip())


def _has_reservation_access(item: CampgroundItem) -> bool:
    return bool(
        (item and item.reservation_url and item.reservation_url.strip())
        or (item and item.reservation_instructions and item.reservation_instructions.strip())
    )


def _mentions_six_month(text: Optional[str]) -> bool:
    if not text:
        return False
    pattern = re.compile(r"\b(6|six)\s*[- ]?\s*month", re.IGNORECASE)
    return bool(pattern.search(text))


def _mentions_fee_825(text: Optional[str]) -> bool:
    if not text:
        return False
    # Accept $8.25 or 8.25 (with or without dollar sign), possibly with commas or spaces
    pattern = re.compile(r"\$?\s*8\.25\b")
    return bool(pattern.search(text))


def _select_item(extracted: CampgroundsExtraction, trip_id: int) -> CampgroundItem:
    # Try explicit match by trip_id
    for c in extracted.campgrounds:
        if c.trip_id == trip_id:
            return c
    # Fallback by position
    idx = trip_id - 1
    if 0 <= idx < len(extracted.campgrounds):
        return extracted.campgrounds[idx]
    return CampgroundItem()


# --------------------------------------------------------------------------- #
# Verification builders (one per trip)                                        #
# --------------------------------------------------------------------------- #
async def verify_trip1(evaluator: Evaluator, parent_node, item: CampgroundItem) -> None:
    """
    Trip 1: Northern California + ADA accessible + dogs on leash + standard amenities + reservation info (URL or instructions) + booking window + fee included.
    Also must be a California State Park and include park and campground names.
    """
    node = evaluator.add_parallel(
        id="Campground_1_Northern_CA_ADA",
        desc="Trip 1 campground meeting Northern CA + ADA + amenities + dog policy + reservation info requirements.",
        parent=parent_node,
        critical=False
    )

    # C1_Park_And_Campground_Names (existence gate)
    c1_names = evaluator.add_custom_node(
        result=_exists_names(item),
        id="C1_Park_And_Campground_Names",
        desc="Response includes the California State Park name and the specific campground name.",
        parent=node,
        critical=True
    )

    sources = _collect_sources(item)
    park_name = item.park_name or "the park"
    cg_name = item.campground_name or "the campground"

    # C1_Is_California_State_Park
    leaf = evaluator.add_leaf(
        id="C1_Is_California_State_Park",
        desc="Campground is part of the California State Parks system.",
        parent=node,
        critical=True
    )
    claim = f"The campground '{cg_name}' at '{park_name}' is within the California State Parks system."
    await evaluator.verify(
        claim=claim,
        node=leaf,
        sources=sources,
        additional_instruction="Use the provided official park or reservation pages to confirm it is a California State Park (parks.ca.gov or ReserveCalifornia are good indicators).",
        extra_prerequisites=[c1_names]
    )

    # C1_Northern_California_Location
    leaf = evaluator.add_leaf(
        id="C1_Northern_California_Location",
        desc="Campground is located in Northern California (north of the San Francisco Bay Area).",
        parent=node,
        critical=True
    )
    claim = "This park is located in Northern California (north of the San Francisco Bay Area counties)."
    await evaluator.verify(
        claim=claim,
        node=leaf,
        sources=sources,
        additional_instruction=("If the page indicates a county in far northern or north coast/interior regions (e.g., Mendocino, Humboldt, Del Norte, Shasta, etc.), "
                                "consider it Northern California. Do not count locations within Bay Area counties as 'north of the Bay Area'."),
        extra_prerequisites=[c1_names]
    )

    # C1_ADA_Accessible_Site
    leaf = evaluator.add_leaf(
        id="C1_ADA_Accessible_Site",
        desc="Campground has at least one ADA-accessible campsite available.",
        parent=node,
        critical=True
    )
    claim = "The campground offers at least one ADA-accessible (accessible) campsite."
    await evaluator.verify(
        claim=claim,
        node=leaf,
        sources=sources,
        additional_instruction="Confirm via official page or reservation listings mentioning accessible/ADA sites.",
        extra_prerequisites=[c1_names]
    )

    # C1_Dogs_Allowed_On_Leash
    leaf = evaluator.add_leaf(
        id="C1_Dogs_Allowed_On_Leash",
        desc="Dogs (including a service dog) are allowed on leash in the campground area.",
        parent=node,
        critical=True
    )
    claim = "Dogs are allowed on leash in the campground area."
    await evaluator.verify(
        claim=claim,
        node=leaf,
        sources=sources,
        additional_instruction="Look for dog policy statements on the park or campground page.",
        extra_prerequisites=[c1_names]
    )

    # C1_Standard_Amenities
    leaf = evaluator.add_leaf(
        id="C1_Standard_Amenities",
        desc="Campsites include standard amenities: picnic table and fire ring.",
        parent=node,
        critical=True
    )
    claim = "Campsites include a picnic table and a fire ring."
    await evaluator.verify(
        claim=claim,
        node=leaf,
        sources=sources,
        additional_instruction="Check a campsite amenities list or campground description for 'picnic table' and 'fire ring' (or fire pit).",
        extra_prerequisites=[c1_names]
    )

    # C1_Reservation_Access_Info (custom existence of booking URL or clear instructions)
    evaluator.add_custom_node(
        result=_has_reservation_access(item),
        id="C1_Reservation_Access_Info",
        desc="Provides a direct booking URL (ReserveCalifornia.com or official booking page) OR clear park contact/reservation instructions sufficient to reserve.",
        parent=node,
        critical=True
    )

    # C1_Booking_Window_Included (6 months mentioned in the answer)
    evaluator.add_custom_node(
        result=_mentions_six_month(item.booking_window_text),
        id="C1_Booking_Window_Included",
        desc="Reservation information includes the 6-month advance booking window.",
        parent=node,
        critical=True
    )

    # C1_Reservation_Fee_Included ($8.25 mentioned in the answer)
    evaluator.add_custom_node(
        result=_mentions_fee_825(item.reservation_fee_text),
        id="C1_Reservation_Fee_Included",
        desc="Reservation information includes the $8.25 non-refundable reservation fee.",
        parent=node,
        critical=True
    )


async def verify_trip2(evaluator: Evaluator, parent_node, item: CampgroundItem) -> None:
    """
    Trip 2: Southern California coastal state park + accommodates 35-foot RV + dogs on leash + standard amenities + reservation info + booking window + fee included.
    Also must be a California State Park and include names.
    """
    node = evaluator.add_parallel(
        id="Campground_2_SoCal_Coastal_RV",
        desc="Trip 2 campground meeting SoCal coast + RV length + amenities + dog policy + reservation info requirements.",
        parent=parent_node,
        critical=False
    )

    # C2_Park_And_Campground_Names
    c2_names = evaluator.add_custom_node(
        result=_exists_names(item),
        id="C2_Park_And_Campground_Names",
        desc="Response includes the California State Park name and the specific campground name.",
        parent=node,
        critical=True
    )

    sources = _collect_sources(item)
    park_name = item.park_name or "the park"
    cg_name = item.campground_name or "the campground"

    # C2_Is_California_State_Park
    leaf = evaluator.add_leaf(
        id="C2_Is_California_State_Park",
        desc="Campground is part of the California State Parks system.",
        parent=node,
        critical=True
    )
    claim = f"The campground '{cg_name}' at '{park_name}' is within the California State Parks system."
    await evaluator.verify(
        claim=claim,
        node=leaf,
        sources=sources,
        additional_instruction="Use the provided official park or reservation pages to confirm it is a California State Park (parks.ca.gov or ReserveCalifornia).",
        extra_prerequisites=[c2_names]
    )

    # C2_Southern_California_Coastal
    leaf = evaluator.add_leaf(
        id="C2_Southern_California_Coastal",
        desc="Campground is located in a Southern California coastal state park along the Pacific Ocean.",
        parent=node,
        critical=True
    )
    claim = "This park is on the Southern California coast along the Pacific Ocean (e.g., in Santa Barbara, Ventura, Los Angeles, Orange, or San Diego counties) and beachside."
    await evaluator.verify(
        claim=claim,
        node=leaf,
        sources=sources,
        additional_instruction=("Confirm it's a coastal/beachside California state park on the Pacific Ocean in Southern California. "
                                "Treat counties like Santa Barbara, Ventura, Los Angeles, Orange, and San Diego as SoCal."),
        extra_prerequisites=[c2_names]
    )

    # C2_RV_Length_Capacity (>= 35 ft)
    leaf = evaluator.add_leaf(
        id="C2_RV_Length_Capacity",
        desc="Campground accommodates RVs with a maximum length of at least 35 feet.",
        parent=node,
        critical=True
    )
    claim = "The campground accommodates RVs up to at least 35 feet in length."
    await evaluator.verify(
        claim=claim,
        node=leaf,
        sources=sources,
        additional_instruction="Look for RV length limits on the campground or reservation page; confirm 35 ft or greater is supported.",
        extra_prerequisites=[c2_names]
    )

    # C2_Dogs_Allowed_On_Leash
    leaf = evaluator.add_leaf(
        id="C2_Dogs_Allowed_On_Leash",
        desc="Dogs are allowed on leash in the campground area.",
        parent=node,
        critical=True
    )
    claim = "Dogs are allowed on leash in the campground area."
    await evaluator.verify(
        claim=claim,
        node=leaf,
        sources=sources,
        additional_instruction="Check the dog policy on the park or campground page.",
        extra_prerequisites=[c2_names]
    )

    # C2_Standard_Amenities
    leaf = evaluator.add_leaf(
        id="C2_Standard_Amenities",
        desc="Campsites include standard amenities: picnic table and fire ring.",
        parent=node,
        critical=True
    )
    claim = "Campsites include a picnic table and a fire ring."
    await evaluator.verify(
        claim=claim,
        node=leaf,
        sources=sources,
        additional_instruction="Confirm via campsite amenities list.",
        extra_prerequisites=[c2_names]
    )

    # C2_Reservation_Access_Info
    evaluator.add_custom_node(
        result=_has_reservation_access(item),
        id="C2_Reservation_Access_Info",
        desc="Provides a direct booking URL (ReserveCalifornia.com or official booking page) OR clear park contact/reservation instructions sufficient to reserve.",
        parent=node,
        critical=True
    )

    # C2_Booking_Window_Included
    evaluator.add_custom_node(
        result=_mentions_six_month(item.booking_window_text),
        id="C2_Booking_Window_Included",
        desc="Reservation information includes the 6-month advance booking window.",
        parent=node,
        critical=True
    )

    # C2_Reservation_Fee_Included
    evaluator.add_custom_node(
        result=_mentions_fee_825(item.reservation_fee_text),
        id="C2_Reservation_Fee_Included",
        desc="Reservation information includes the $8.25 non-refundable reservation fee.",
        parent=node,
        critical=True
    )


async def verify_trip3(evaluator: Evaluator, parent_node, item: CampgroundItem) -> None:
    """
    Trip 3: Electric hookups + flush toilets + hot/coin-operated showers + dogs on leash + standard amenities + reservation info + booking window + fee.
    Also must be California State Park and include names.
    """
    node = evaluator.add_parallel(
        id="Campground_3_Electric_Hookups",
        desc="Trip 3 campground meeting electric hookups + flush toilets + hot showers + amenities + dog policy + reservation info requirements.",
        parent=parent_node,
        critical=False
    )

    # C3_Park_And_Campground_Names
    c3_names = evaluator.add_custom_node(
        result=_exists_names(item),
        id="C3_Park_And_Campground_Names",
        desc="Response includes the California State Park name and the specific campground name.",
        parent=node,
        critical=True
    )

    sources = _collect_sources(item)
    park_name = item.park_name or "the park"
    cg_name = item.campground_name or "the campground"

    # C3_Is_California_State_Park
    leaf = evaluator.add_leaf(
        id="C3_Is_California_State_Park",
        desc="Campground is part of the California State Parks system.",
        parent=node,
        critical=True
    )
    claim = f"The campground '{cg_name}' at '{park_name}' is within the California State Parks system."
    await evaluator.verify(
        claim=claim,
        node=leaf,
        sources=sources,
        additional_instruction="Confirm via parks.ca.gov or ReserveCalifornia.",
        extra_prerequisites=[c3_names]
    )

    # C3_Electric_Hookups
    leaf = evaluator.add_leaf(
        id="C3_Electric_Hookups",
        desc="Campsites offer electric hookups.",
        parent=node,
        critical=True
    )
    claim = "Campsites at this campground offer electric hookups."
    await evaluator.verify(
        claim=claim,
        node=leaf,
        sources=sources,
        additional_instruction="Look for mentions of 'electric' or 'power hookups' at the campsite level.",
        extra_prerequisites=[c3_names]
    )

    # C3_Flush_Toilets
    leaf = evaluator.add_leaf(
        id="C3_Flush_Toilets",
        desc="Campground has flush toilets available.",
        parent=node,
        critical=True
    )
    claim = "The campground provides flush toilets."
    await evaluator.verify(
        claim=claim,
        node=leaf,
        sources=sources,
        additional_instruction="Facilities list or campground description should indicate flush toilets.",
        extra_prerequisites=[c3_names]
    )

    # C3_Pay_Showers
    leaf = evaluator.add_leaf(
        id="C3_Pay_Showers",
        desc="Campground has coin-operated or pay showers available.",
        parent=node,
        critical=True
    )
    claim = "The campground provides coin-operated or pay showers (hot showers)."
    await evaluator.verify(
        claim=claim,
        node=leaf,
        sources=sources,
        additional_instruction="Confirm showers are available and typically coin-operated or fee-based in CA State Parks.",
        extra_prerequisites=[c3_names]
    )

    # C3_Dogs_Allowed_On_Leash
    leaf = evaluator.add_leaf(
        id="C3_Dogs_Allowed_On_Leash",
        desc="Dogs are allowed on leash in the campground area.",
        parent=node,
        critical=True
    )
    claim = "Dogs are allowed on leash in the campground area."
    await evaluator.verify(
        claim=claim,
        node=leaf,
        sources=sources,
        additional_instruction="Check dog policy statements.",
        extra_prerequisites=[c3_names]
    )

    # C3_Standard_Amenities
    leaf = evaluator.add_leaf(
        id="C3_Standard_Amenities",
        desc="Campsites include standard amenities: picnic table and fire ring.",
        parent=node,
        critical=True
    )
    claim = "Campsites include a picnic table and a fire ring."
    await evaluator.verify(
        claim=claim,
        node=leaf,
        sources=sources,
        additional_instruction="Look for campsite amenities.",
        extra_prerequisites=[c3_names]
    )

    # C3_Reservation_Access_Info
    evaluator.add_custom_node(
        result=_has_reservation_access(item),
        id="C3_Reservation_Access_Info",
        desc="Provides a direct booking URL (ReserveCalifornia.com or official booking page) OR clear park contact/reservation instructions sufficient to reserve.",
        parent=node,
        critical=True
    )

    # C3_Booking_Window_Included
    evaluator.add_custom_node(
        result=_mentions_six_month(item.booking_window_text),
        id="C3_Booking_Window_Included",
        desc="Reservation information includes the 6-month advance booking window.",
        parent=node,
        critical=True
    )

    # C3_Reservation_Fee_Included
    evaluator.add_custom_node(
        result=_mentions_fee_825(item.reservation_fee_text),
        id="C3_Reservation_Fee_Included",
        desc="Reservation information includes the $8.25 non-refundable reservation fee.",
        parent=node,
        critical=True
    )


async def verify_trip4(evaluator: Evaluator, parent_node, item: CampgroundItem) -> None:
    """
    Trip 4: Hike-in or bike-in (not standard drive-in) + dogs on leash + basic facilities + reservation/access info + booking window + fee.
    Also must be a California State Park and include names.
    """
    node = evaluator.add_parallel(
        id="Campground_4_Hike_Bike_In",
        desc="Trip 4 campground meeting hike/bike-in requirement + dog policy + basic facilities + reservation/access info requirements.",
        parent=parent_node,
        critical=False
    )

    # C4_Park_And_Campground_Names
    c4_names = evaluator.add_custom_node(
        result=_exists_names(item),
        id="C4_Park_And_Campground_Names",
        desc="Response includes the California State Park name and the specific campground name.",
        parent=node,
        critical=True
    )

    sources = _collect_sources(item)
    park_name = item.park_name or "the park"
    cg_name = item.campground_name or "the campground"

    # C4_Is_California_State_Park
    leaf = evaluator.add_leaf(
        id="C4_Is_California_State_Park",
        desc="Campground is part of the California State Parks system.",
        parent=node,
        critical=True
    )
    claim = f"The campground '{cg_name}' at '{park_name}' is within the California State Parks system."
    await evaluator.verify(
        claim=claim,
        node=leaf,
        sources=sources,
        additional_instruction="Confirm via parks.ca.gov or ReserveCalifornia page.",
        extra_prerequisites=[c4_names]
    )

    # C4_Hike_Or_Bike_In_Distinct
    leaf = evaluator.add_leaf(
        id="C4_Hike_Or_Bike_In_Distinct",
        desc="Campground offers hike-in or bike-in campsites that are distinct from standard drive-in sites.",
        parent=node,
        critical=True
    )
    claim = "The campground offers hike-in or bike-in campsites (not standard drive-in sites)."
    await evaluator.verify(
        claim=claim,
        node=leaf,
        sources=sources,
        additional_instruction="Look for terms like 'hike-in', 'walk-in', 'bike-in', or environmental campsites that require walking/biking from parking.",
        extra_prerequisites=[c4_names]
    )

    # C4_Dogs_Allowed_On_Leash
    leaf = evaluator.add_leaf(
        id="C4_Dogs_Allowed_On_Leash",
        desc="Dogs are allowed on leash in the campground area.",
        parent=node,
        critical=True
    )
    claim = "Dogs are allowed on leash in the campground area."
    await evaluator.verify(
        claim=claim,
        node=leaf,
        sources=sources,
        additional_instruction="Verify dog policy on the official page.",
        extra_prerequisites=[c4_names]
    )

    # C4_Basic_Facilities
    leaf = evaluator.add_leaf(
        id="C4_Basic_Facilities",
        desc="Basic camping facilities are available.",
        parent=node,
        critical=True
    )
    claim = "The campground provides basic camping facilities (for example, toilets and water)."
    await evaluator.verify(
        claim=claim,
        node=leaf,
        sources=sources,
        additional_instruction="Check for at least basic facilities such as toilets (pit or flush) and water availability.",
        extra_prerequisites=[c4_names]
    )

    # C4_Reservation_Access_Info
    evaluator.add_custom_node(
        result=_has_reservation_access(item),
        id="C4_Reservation_Access_Info",
        desc="Provides a direct booking URL (ReserveCalifornia.com or official booking page) OR clear park contact/access instructions sufficient to reserve or legally obtain a site (e.g., first-come-first-served process).",
        parent=node,
        critical=True
    )

    # C4_Booking_Window_Included
    evaluator.add_custom_node(
        result=_mentions_six_month(item.booking_window_text),
        id="C4_Booking_Window_Included",
        desc="Reservation information includes the 6-month advance booking window.",
        parent=node,
        critical=True
    )

    # C4_Reservation_Fee_Included
    evaluator.add_custom_node(
        result=_mentions_fee_825(item.reservation_fee_text),
        id="C4_Reservation_Fee_Included",
        desc="Reservation information includes the $8.25 non-refundable reservation fee.",
        parent=node,
        critical=True
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
    Evaluate an answer for the California State Parks multi-trip campground selection task.
    """
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

    # Extract structured campground info from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_campgrounds(),
        template_class=CampgroundsExtraction,
        extraction_name="campgrounds_extraction"
    )

    # Build the root node for the four campgrounds (parallel aggregation)
    top = evaluator.add_parallel(
        id="Find_4_California_Campgrounds",
        desc="Find 4 campgrounds in California State Parks meeting the four trip-specific requirements and provide required reservation details.",
        parent=root,
        critical=False
    )

    # Select items for each trip (with fallback by order)
    trip1_item = _select_item(extracted, trip_id=1)
    trip2_item = _select_item(extracted, trip_id=2)
    trip3_item = _select_item(extracted, trip_id=3)
    trip4_item = _select_item(extracted, trip_id=4)

    # Verify each trip subtree
    await verify_trip1(evaluator, top, trip1_item)
    await verify_trip2(evaluator, top, trip2_item)
    await verify_trip3(evaluator, top, trip3_item)
    await verify_trip4(evaluator, top, trip4_item)

    # Return final summary
    return evaluator.get_summary()