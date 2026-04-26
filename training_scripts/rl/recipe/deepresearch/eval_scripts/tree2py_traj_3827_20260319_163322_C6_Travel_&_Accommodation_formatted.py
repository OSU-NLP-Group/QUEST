import asyncio
import logging
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.verification_tree import AggregationStrategy


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "boutique_hotels_suncountry_msp"
TASK_DESCRIPTION = (
    "I am planning a vacation from Minneapolis and prefer to fly nonstop on Sun Country Airlines to keep my travel "
    "simple and affordable. I'm looking for boutique hotel or resort properties (defined as having fewer than 100 "
    "guest rooms) that offer a luxury experience with 4-star or higher standards.\n\n"
    "Please identify three boutique properties in destinations served by Sun Country Airlines with nonstop flights "
    "from Minneapolis-St. Paul International Airport (MSP) that meet all of the following requirements:\n\n"
    "Property Classification:\n"
    "- Must be a boutique hotel or resort with fewer than 100 guest rooms\n"
    "- Must be rated 4-star or higher (or clearly meet 4-star standards)\n"
    "- Guest rooms must be at least 18 square meters (approximately 194 square feet)\n\n"
    "Amenities:\n"
    "- Must have at least 2 on-site restaurants\n"
    "- Must have an on-site spa or wellness center\n"
    "- Must have a swimming pool\n\n"
    "Booking Flexibility:\n"
    "- Must offer free cancellation at least 24 hours before check-in\n"
    "- Must offer an advance purchase discount of at least 10% when booking 7 or more days in advance\n\n"
    "For each property, please provide:\n"
    "1. Property name and destination city\n"
    "2. Number of guest rooms\n"
    "3. Star rating or classification\n"
    "4. Guest room size\n"
    "5. Names of at least 2 on-site restaurants\n"
    "6. Description of spa/wellness facilities\n"
    "7. Description of pool facilities\n"
    "8. Cancellation policy details\n"
    "9. Advance booking discount details\n"
    "10. URL reference to the property's official website or booking page\n"
    "11. URL reference to Sun Country Airlines route information confirming nonstop service from MSP to the destination"
)


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class PropertyItem(BaseModel):
    # Basic identity
    name: Optional[str] = None
    city: Optional[str] = None

    # Classification details
    number_of_rooms: Optional[str] = None
    star_rating_or_classification: Optional[str] = None
    min_guest_room_size: Optional[str] = None  # Keep as string; can be "200 sq ft" or "20 sqm"

    # Amenities
    restaurants: List[str] = Field(default_factory=list)  # Names of at least 2 on-site restaurants
    spa_description: Optional[str] = None
    pool_description: Optional[str] = None

    # Booking policies
    cancellation_policy_details: Optional[str] = None
    advance_discount_details: Optional[str] = None

    # URLs
    property_urls: List[str] = Field(default_factory=list)        # Official site or booking page(s)
    classification_urls: List[str] = Field(default_factory=list)  # Pages showing room count, stars, room size
    amenity_urls: List[str] = Field(default_factory=list)         # Pages showing restaurants/spa/pool
    policy_urls: List[str] = Field(default_factory=list)          # Pages showing cancellation/advance purchase
    sun_country_route_urls: List[str] = Field(default_factory=list)  # Route map/schedule/destination page


class PropertiesExtraction(BaseModel):
    properties: List[PropertyItem] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_properties() -> str:
    return """
Extract up to three boutique hotel or resort properties (return the first three mentioned) that the answer proposes.
For EACH property, extract the following fields exactly as presented in the answer:

1) name: The official property name.
2) city: The destination city served by Sun Country Airlines from MSP.
3) number_of_rooms: The stated number of guest rooms (e.g., "72", "85 rooms", "86 keys"); return as a string exactly as written.
4) star_rating_or_classification: A 4-star or higher rating, or a clearly stated equivalent luxury classification; return as a string exactly as written (e.g., "5-star", "4.5-star", "luxury boutique").
5) min_guest_room_size: The minimum guest room size or a typical baseline size (e.g., "200 sq ft", "20 sqm"); return exactly as written.
6) restaurants: An array of the names of on-site restaurants (extract all that are listed; must include at least two names if provided).
7) spa_description: A short phrase quoted or paraphrased from the answer describing the spa/wellness facilities; if not provided, set to null.
8) pool_description: A short phrase quoted or paraphrased from the answer describing the pool facilities; if not provided, set to null.
9) cancellation_policy_details: Details indicating free cancellation at least 24 hours before check-in (or stricter, e.g., 48/72 hours, which still qualifies as >=24h); if not provided, set to null.
10) advance_discount_details: Details indicating an advance purchase discount of at least 10% for bookings made 7+ days in advance; if not provided, set to null.
11) property_urls: Array of official website or booking page URLs for the property (one or more). Only include URLs explicitly present in the answer.
12) classification_urls: Array of URLs (official site or booking platform) that show room count, star rating, and/or room size info (can overlap with property_urls if applicable).
13) amenity_urls: Array of URLs (official site or booking platform) that show restaurants, spa/wellness, and/or pool details (can overlap with property_urls if applicable).
14) policy_urls: Array of URLs (official site or booking platform) that show cancellation policy AND advance purchase/early booking discount info (can overlap with other URLs if applicable).
15) sun_country_route_urls: Array of Sun Country Airlines URLs (route map, schedule, destination page) that explicitly confirm nonstop service from MSP to the property's destination city.

RULES:
- Do NOT invent any information; extract only what the answer explicitly provides.
- For URLs, extract the actual link targets; accept plain URLs or markdown links. If a URL lacks protocol, prepend "http://".
- If some fields are not provided in the answer, set them to null (for single fields) or empty array (for list fields).
- Return JSON with a top-level "properties" array of up to three PropertyItem objects in the same order as the answer.
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _merge_urls(*lists: List[str]) -> List[str]:
    seen = set()
    merged: List[str] = []
    for lst in lists:
        for url in lst or []:
            if url and url not in seen:
                seen.add(url)
                merged.append(url)
    return merged


# --------------------------------------------------------------------------- #
# Verification per property                                                   #
# --------------------------------------------------------------------------- #
async def verify_property(evaluator: Evaluator, parent_node, prop: PropertyItem, idx: int) -> None:
    """
    Build verification sub-tree and run verifications for a single property.
    """
    prop_id = f"property_{idx + 1}"

    # Container for this property (parallel, non-critical to allow partial credit across 3 properties)
    prop_node = evaluator.add_parallel(
        id=prop_id,
        desc=[
            "First boutique property identification and verification",
            "Second boutique property identification and verification",
            "Third boutique property identification and verification",
        ][idx],
        parent=parent_node,
        critical=False
    )

    # -------------------- Basic information -------------------- #
    basic_node = evaluator.add_parallel(
        id=f"{prop_id}_basic_information",
        desc="Verify basic property information is provided",
        parent=prop_node,
        critical=True
    )

    # Name + city presence
    name_city_present = bool(prop.name and prop.name.strip()) and bool(prop.city and prop.city.strip())
    evaluator.add_custom_node(
        result=name_city_present,
        id=f"{prop_id}_name_city",
        desc="Property name and destination city are provided",
        parent=basic_node,
        critical=True
    )

    # Property website/booking URL provided
    has_property_url = len(prop.property_urls) > 0
    evaluator.add_custom_node(
        result=has_property_url,
        id=f"{prop_id}_website_url",
        desc="URL reference to the property's official website or booking page is provided",
        parent=basic_node,
        critical=True
    )

    # -------------------- Flight accessibility (Sun Country nonstop MSP) -------------------- #
    flight_node = evaluator.add_sequential(
        id=f"{prop_id}_flight_accessibility",
        desc="Verify the destination is served by Sun Country Airlines nonstop from MSP",
        parent=prop_node,
        critical=True
    )

    route_ver_node = evaluator.add_parallel(
        id=f"{prop_id}_route_verification",
        desc="Verify Sun Country Airlines route exists",
        parent=flight_node,
        critical=True
    )

    # Route reference URL presence (critical sibling, created first to gate the claim)
    has_route_ref = len(prop.sun_country_route_urls) > 0
    evaluator.add_custom_node(
        result=has_route_ref,
        id=f"{prop_id}_route_reference",
        desc="URL reference from Sun Country Airlines route map or schedule confirming the nonstop route is provided",
        parent=route_ver_node,
        critical=True
    )

    # Verify the nonstop MSP->city service on Sun Country
    route_claim_node = evaluator.add_leaf(
        id=f"{prop_id}_sun_country_route",
        desc="The property's destination city has nonstop Sun Country Airlines service from Minneapolis-St. Paul International Airport (MSP)",
        parent=route_ver_node,
        critical=True
    )
    dest_city = prop.city or "the destination city"
    route_claim = (
        f"Sun Country Airlines operates nonstop (direct) flights from Minneapolis–St. Paul (MSP) to {dest_city}. "
        f"Seasonal or limited-service nonstop still qualifies."
    )
    await evaluator.verify(
        claim=route_claim,
        node=route_claim_node,
        sources=prop.sun_country_route_urls,
        additional_instruction="Verify using the Sun Country official site (route map, destination page, or schedule). "
                               "Accept 'nonstop' or 'direct'. Ignore codeshares; it must be Sun Country metal."
    )

    # -------------------- Classification: boutique + stars + room size -------------------- #
    class_node = evaluator.add_parallel(
        id=f"{prop_id}_classification",
        desc="Verify the property meets boutique and star rating standards",
        parent=prop_node,
        critical=True
    )

    class_details_node = evaluator.add_parallel(
        id=f"{prop_id}_size_rating_details",
        desc="Property classification details are verified",
        parent=class_node,
        critical=True
    )

    # Classification references presence (critical sibling to gate checks)
    class_sources = _merge_urls(prop.classification_urls, prop.property_urls)
    evaluator.add_custom_node(
        result=len(class_sources) > 0,
        id=f"{prop_id}_classification_reference",
        desc="URL reference from the property's official website or booking platform showing room count, star rating, and room size information is provided",
        parent=class_node,
        critical=True
    )

    # Boutique size: < 100 guestrooms
    size_leaf = evaluator.add_leaf(
        id=f"{prop_id}_boutique_size",
        desc="The property has fewer than 100 guest rooms, meeting the industry standard definition of a boutique hotel, and the number of guest rooms is provided",
        parent=class_details_node,
        critical=True
    )
    size_claim = (
        f"The property has fewer than 100 guest rooms. The answer cites: '{prop.number_of_rooms}'. "
        f"Confirm on the cited page(s) that the total keys/rooms/suites are under 100."
    )
    await evaluator.verify(
        claim=size_claim,
        node=size_leaf,
        sources=class_sources,
        additional_instruction="Treat 'keys' or 'suites' as room count equivalents if explicitly referring to total inventory."
    )

    # Star rating: >= 4-star or clearly meets 4-star standards
    star_leaf = evaluator.add_leaf(
        id=f"{prop_id}_star_rating",
        desc="The property is rated 4-star or higher according to a recognized rating system or clearly meets 4-star standards, and the star rating or classification is provided",
        parent=class_details_node,
        critical=True
    )
    star_claim = (
        f"The property is 4-star or higher (or clearly meets 4-star luxury standards). The answer cites: "
        f"'{prop.star_rating_or_classification}'. Confirm on the cited page(s)."
    )
    await evaluator.verify(
        claim=star_claim,
        node=star_leaf,
        sources=class_sources,
        additional_instruction="Accept explicit star ratings (e.g., 4-star, 4.5-star, 5-star) from booking platforms or "
                               "official descriptors that unambiguously indicate 4-star-level service standard."
    )

    # Room size: >= 18 sqm (~194 sq ft)
    roomsize_leaf = evaluator.add_leaf(
        id=f"{prop_id}_room_size",
        desc="Guest rooms are at least 18 square meters (approximately 194 square feet), meeting the minimum size requirement for 4-star properties, and the guest room size is provided",
        parent=class_details_node,
        critical=True
    )
    roomsize_claim = (
        f"Guest rooms are at least 18 square meters (≈194 sq ft). The answer cites: '{prop.min_guest_room_size}'. "
        f"Confirm the smallest listed room type is ≥ 18 sqm (or ≥ 194 sq ft)."
    )
    await evaluator.verify(
        claim=roomsize_claim,
        node=roomsize_leaf,
        sources=class_sources,
        additional_instruction="If sizes are shown in sq ft, convert: 194 sq ft ≈ 18 sqm. "
                               "Use the smallest/entry-level room size when multiple categories are listed."
    )

    # -------------------- Dining & wellness amenities -------------------- #
    amenities_node = evaluator.add_parallel(
        id=f"{prop_id}_dining_wellness",
        desc="Verify the property has required dining and wellness amenities",
        parent=prop_node,
        critical=True
    )

    amenities_details_node = evaluator.add_parallel(
        id=f"{prop_id}_amenities_details",
        desc="Property amenities are verified and described",
        parent=amenities_node,
        critical=True
    )

    amenity_sources = _merge_urls(prop.amenity_urls, prop.property_urls)

    # Amenity references presence (critical sibling to gate checks)
    evaluator.add_custom_node(
        result=len(amenity_sources) > 0,
        id=f"{prop_id}_amenities_reference",
        desc="URL reference showing the property's restaurants, spa/wellness center, and pool facilities is provided",
        parent=amenities_node,
        critical=True
    )

    # Restaurants: at least 2 on-site, with names
    restaurants_leaf = evaluator.add_leaf(
        id=f"{prop_id}_restaurants",
        desc="The property has at least 2 on-site restaurants, meeting the dining standard for 4-star+ properties, and the names of at least 2 on-site restaurants are provided",
        parent=amenities_details_node,
        critical=True
    )
    rest_names = ", ".join(prop.restaurants[:5]) if prop.restaurants else "None listed"
    restaurants_claim = (
        f"The property has at least two on-site restaurants. Examples provided: {rest_names}."
    )
    await evaluator.verify(
        claim=restaurants_claim,
        node=restaurants_leaf,
        sources=amenity_sources,
        additional_instruction="Verify that there are at least two distinct on-site dining outlets (restaurant/brasserie/bistro). "
                               "Bars or lounges count only if clearly positioned as dining venues."
    )

    # Spa/wellness present
    spa_leaf = evaluator.add_leaf(
        id=f"{prop_id}_spa",
        desc="The property has an on-site spa or wellness center, and a description of spa/wellness facilities is provided",
        parent=amenities_details_node,
        critical=True
    )
    spa_claim = (
        f"The property has an on-site spa or wellness center. The answer describes: '{prop.spa_description}'."
    )
    await evaluator.verify(
        claim=spa_claim,
        node=spa_leaf,
        sources=amenity_sources,
        additional_instruction="Accept synonyms such as 'spa', 'wellness center', 'treatment rooms', 'massage spa', "
                               "'thermal area', or similar facilities clearly on-site."
    )

    # Pool present
    pool_leaf = evaluator.add_leaf(
        id=f"{prop_id}_pool",
        desc="The property has a swimming pool facility, and a description of pool facilities is provided",
        parent=amenities_details_node,
        critical=True
    )
    pool_claim = (
        f"The property has a swimming pool (e.g., indoor/outdoor/rooftop/plunge). The answer describes: '{prop.pool_description}'."
    )
    await evaluator.verify(
        claim=pool_claim,
        node=pool_leaf,
        sources=amenity_sources,
        additional_instruction="Confirm any on-site pool; rooftop or outdoor pools count. Whirlpool-only without a pool does not count."
    )

    # -------------------- Booking policies: cancellation + advance purchase -------------------- #
    policies_node = evaluator.add_parallel(
        id=f"{prop_id}_booking_policies",
        desc="Verify the property offers required booking flexibility",
        parent=prop_node,
        critical=True
    )

    policy_details_node = evaluator.add_parallel(
        id=f"{prop_id}_policy_details",
        desc="Booking policy details are verified and provided",
        parent=policies_node,
        critical=True
    )

    policy_sources = _merge_urls(prop.policy_urls, prop.property_urls)

    # Policies reference presence (critical sibling to gate checks)
    evaluator.add_custom_node(
        result=len(policy_sources) > 0,
        id=f"{prop_id}_policies_reference",
        desc="URL reference from the property's website or booking platform showing cancellation policy and advance purchase discount details is provided",
        parent=policies_node,
        critical=True
    )

    # Free cancellation >= 24 hours before check-in
    cancel_leaf = evaluator.add_leaf(
        id=f"{prop_id}_cancellation",
        desc="The property offers free cancellation at least 24 hours before check-in for standard bookings, and cancellation policy details are provided",
        parent=policy_details_node,
        critical=True
    )
    cancel_claim = (
        f"The property's standard flexible rate allows free cancellation at least 24 hours before check-in. "
        f"The answer cites: '{prop.cancellation_policy_details}'. More restrictive windows (e.g., 48/72 hours) still satisfy 'at least 24 hours'."
    )
    await evaluator.verify(
        claim=cancel_claim,
        node=cancel_leaf,
        sources=policy_sources,
        additional_instruction="Focus on flexible/standard rates (not non-refundable/advance purchase). "
                               "If policy states 24h+, 48h, 72h, or 'day before by 4 PM', treat as meeting the ≥24h requirement."
    )

    # Advance purchase discount >= 10% when booking 7+ days ahead
    adv_leaf = evaluator.add_leaf(
        id=f"{prop_id}_advance_discount",
        desc="The property offers an advance purchase discount of at least 10% when booking 7 or more days ahead, and advance booking discount details are provided",
        parent=policy_details_node,
        critical=True
    )
    adv_claim = (
        f"The property offers an advance purchase/early booking discount of at least 10% for bookings made 7 or more days in advance. "
        f"The answer cites: '{prop.advance_discount_details}'."
    )
    await evaluator.verify(
        claim=adv_claim,
        node=adv_leaf,
        sources=policy_sources,
        additional_instruction="Accept phrases like 'advance purchase', 'book early and save', 'advance saver', "
                               "'pay now and save' so long as discount ≥ 10% and lead time ≥ 7 days."
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
    """
    Evaluate an answer for the Sun Country MSP boutique hotels task.
    """
    # Initialize evaluator (root is non-critical by framework design; parallel across 3 properties)
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

    # Extract structured properties info
    extracted: PropertiesExtraction = await evaluator.extract(
        prompt=prompt_extract_properties(),
        template_class=PropertiesExtraction,
        extraction_name="properties_extraction",
    )

    # Ensure exactly 3 properties (pad with empty objects if fewer)
    props: List[PropertyItem] = list(extracted.properties[:3])
    while len(props) < 3:
        props.append(PropertyItem())

    # Build verification subtrees for 3 properties
    for i in range(3):
        await verify_property(evaluator, root, props[i], i)

    # Return the aggregated evaluation summary
    return evaluator.get_summary()