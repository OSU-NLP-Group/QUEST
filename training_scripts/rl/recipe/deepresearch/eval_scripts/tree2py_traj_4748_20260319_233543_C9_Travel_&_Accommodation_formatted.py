import asyncio
import logging
from typing import Any, List, Optional, Dict

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "aruba_all_inclusive_resorts"
TASK_DESCRIPTION = (
    "I'm planning a luxury vacation to Aruba and want to find three all-inclusive beachfront resorts that offer "
    "comprehensive amenities. For each resort, the answer should include: Basic information (resort name, official "
    "website URL, specific beach location, confirmation of true all-inclusive), Accommodation details (suites, private "
    "balconies/terraces, direct beach access), Dining & AI Package (three meals included, >=3 restaurants, unlimited "
    "beverages), Amenities & Facilities (>=2 outdoor pools, complimentary non-motorized water sports, on-site spa with "
    "treatment rooms), Booking Policies (minimum check-in age, deposit amount, cancellation policy with timeframes/fees). "
    "Each piece of information must be supported with URL(s)."
)


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class ResortBasic(BaseModel):
    name: Optional[str] = None
    official_website_url: Optional[str] = None
    basic_info_urls: List[str] = Field(default_factory=list)


class ResortLocation(BaseModel):
    beach: Optional[str] = None
    location_urls: List[str] = Field(default_factory=list)


class ResortAllInclusive(BaseModel):
    confirmation_text: Optional[str] = None
    ai_urls: List[str] = Field(default_factory=list)


class ResortAccommodation(BaseModel):
    suite_available_text: Optional[str] = None
    suite_urls: List[str] = Field(default_factory=list)
    balcony_terrace_text: Optional[str] = None
    room_features_urls: List[str] = Field(default_factory=list)
    direct_beach_access_text: Optional[str] = None
    beach_access_urls: List[str] = Field(default_factory=list)


class ResortDining(BaseModel):
    meals_included_text: Optional[str] = None
    meals_urls: List[str] = Field(default_factory=list)
    restaurant_count_text: Optional[str] = None
    restaurants_urls: List[str] = Field(default_factory=list)
    beverages_included_text: Optional[str] = None
    beverages_urls: List[str] = Field(default_factory=list)


class ResortAmenities(BaseModel):
    pools_count_text: Optional[str] = None
    pools_urls: List[str] = Field(default_factory=list)
    water_sports_text: Optional[str] = None
    water_sports_urls: List[str] = Field(default_factory=list)
    spa_facility_text: Optional[str] = None
    spa_urls: List[str] = Field(default_factory=list)


class ResortBooking(BaseModel):
    check_in_age_text: Optional[str] = None
    age_urls: List[str] = Field(default_factory=list)
    deposit_text: Optional[str] = None
    deposit_urls: List[str] = Field(default_factory=list)
    cancellation_text: Optional[str] = None
    cancellation_urls: List[str] = Field(default_factory=list)


class ResortItem(BaseModel):
    basic: ResortBasic = Field(default_factory=ResortBasic)
    location: ResortLocation = Field(default_factory=ResortLocation)
    all_inclusive: ResortAllInclusive = Field(default_factory=ResortAllInclusive)
    accommodation: ResortAccommodation = Field(default_factory=ResortAccommodation)
    dining: ResortDining = Field(default_factory=ResortDining)
    amenities: ResortAmenities = Field(default_factory=ResortAmenities)
    booking: ResortBooking = Field(default_factory=ResortBooking)


class ResortsExtraction(BaseModel):
    resorts: List[ResortItem] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_resorts() -> str:
    return """
You will extract up to three Aruba all-inclusive beachfront resorts from the answer text. Only extract the first three resorts mentioned; if fewer than three are present, extract the available ones. For each resort, return a JSON object capturing the requested fields and the URL(s) cited in the answer for each field.

Return a JSON object:
{
  "resorts": [
    {
      "basic": {
        "name": string or null,
        "official_website_url": string or null,  // the brand/resort's official site for direct booking
        "basic_info_urls": [array of URLs]       // any URLs cited supporting the name/official site
      },
      "location": {
        "beach": string or null,                 // e.g., "Palm Beach", "Eagle Beach"
        "location_urls": [array of URLs]
      },
      "all_inclusive": {
        "confirmation_text": string or null,     // verbatim or paraphrase that it's truly all-inclusive
        "ai_urls": [array of URLs]
      },
      "accommodation": {
        "suite_available_text": string or null,  // indicates suites beyond standard rooms
        "suite_urls": [array of URLs],
        "balcony_terrace_text": string or null,  // indicates private balcony or terrace in rooms/suites
        "room_features_urls": [array of URLs],
        "direct_beach_access_text": string or null, // indicates direct beach access from property
        "beach_access_urls": [array of URLs]
      },
      "dining": {
        "meals_included_text": string or null,   // indicates breakfast + lunch + dinner included
        "meals_urls": [array of URLs],
        "restaurant_count_text": string or null, // any phrasing indicating total venues; do NOT infer a number
        "restaurants_urls": [array of URLs],
        "beverages_included_text": string or null, // indicates unlimited alcoholic & non-alcoholic drinks
        "beverages_urls": [array of URLs]
      },
      "amenities": {
        "pools_count_text": string or null,         // any phrasing indicating >= 2 outdoor pools
        "pools_urls": [array of URLs],
        "water_sports_text": string or null,        // indicates complimentary non-motorized water sports included
        "water_sports_urls": [array of URLs],
        "spa_facility_text": string or null,        // indicates on-site spa with treatment rooms
        "spa_urls": [array of URLs]
      },
      "booking": {
        "check_in_age_text": string or null,        // minimum age requirement wording
        "age_urls": [array of URLs],
        "deposit_text": string or null,             // deposit requirement wording/amount
        "deposit_urls": [array of URLs],
        "cancellation_text": string or null,        // cancellation policy wording with timeframes/fees
        "cancellation_urls": [array of URLs]
      }
    }
  ]
}

Rules:
- Extract only what is explicitly present in the answer.
- For any field missing from the answer, set it to null (for strings) or [] (for URL arrays).
- For all URL arrays, include every URL explicitly cited for that item in the answer. If none are cited, return [].
- If a URL is missing the protocol, prepend http://
- Do not invent data, do not summarize beyond what is needed to capture each field.
"""


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def _nonempty(s: Optional[str]) -> bool:
    return bool(s and str(s).strip())


def _valid_url(s: Optional[str]) -> bool:
    return _nonempty(s) and (s.strip().lower().startswith("http://") or s.strip().lower().startswith("https://"))


def _build_sources(priority_urls: List[str], fallback_url: Optional[str]) -> List[str]:
    urls = [u for u in priority_urls if _nonempty(u)]
    if not urls and _valid_url(fallback_url):
        urls = [fallback_url]  # fallback to official website if available
    # Deduplicate while preserving order
    seen = set()
    deduped = []
    for u in urls:
        if u not in seen:
            seen.add(u)
            deduped.append(u)
    return deduped


# --------------------------------------------------------------------------- #
# Verification builder for one resort                                         #
# --------------------------------------------------------------------------- #
async def verify_resort(evaluator: Evaluator, parent_node, resort: ResortItem, idx: int) -> None:
    rnum = idx + 1
    resort_node = evaluator.add_parallel(
        id=f"Resort_{rnum}",
        desc=f"Evaluation of resort #{rnum}",
        parent=parent_node,
        critical=False
    )

    # --------------------------- Basic Information --------------------------- #
    basic_node = evaluator.add_parallel(
        id=f"Resort_{rnum}_Basic_Information",
        desc=f"Verify basic information for resort #{rnum}",
        parent=resort_node,
        critical=True
    )

    # Resort name and website
    rnw_node = evaluator.add_parallel(
        id=f"Resort_{rnum}_Resort_Name_and_Website",
        desc=f"Verify resort name and official website for resort #{rnum}",
        parent=basic_node,
        critical=True
    )

    # Existence: Resort name
    evaluator.add_custom_node(
        result=_nonempty(resort.basic.name),
        id=f"Resort_{rnum}_Resort_Name_Provided",
        desc="Resort name is clearly stated",
        parent=rnw_node,
        critical=True
    )

    # Existence: Official website URL
    evaluator.add_custom_node(
        result=_valid_url(resort.basic.official_website_url),
        id=f"Resort_{rnum}_Official_Website_URL",
        desc="Official website URL for direct booking is provided",
        parent=rnw_node,
        critical=True
    )

    # URL verification for name/website (use official site)
    basic_info_verify_leaf = evaluator.add_leaf(
        id=f"Resort_{rnum}_Basic_Info_URL_Verification",
        desc="Valid URL supports resort name and official website information",
        parent=rnw_node,
        critical=True
    )
    name_for_claim = resort.basic.name or "the resort"
    await evaluator.verify(
        claim=f"This page is the official website for {name_for_claim} in Aruba.",
        node=basic_info_verify_leaf,
        sources=resort.basic.official_website_url or resort.basic.basic_info_urls,
        additional_instruction="Confirm the page clearly represents the resort (brand-owned domain or the resort's own domain) in Aruba."
    )

    # Location details
    loc_node = evaluator.add_parallel(
        id=f"Resort_{rnum}_Location_Details",
        desc=f"Verify specific beach location in Aruba for resort #{rnum}",
        parent=basic_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=_nonempty(resort.location.beach),
        id=f"Resort_{rnum}_Beach_Location_Specified",
        desc="The specific beach location in Aruba is identified",
        parent=loc_node,
        critical=True
    )
    loc_verify_leaf = evaluator.add_leaf(
        id=f"Resort_{rnum}_Location_URL_Verification",
        desc="Valid URL supports the beach location information",
        parent=loc_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The resort {name_for_claim} is located on {resort.location.beach or 'a named beach'} in Aruba.",
        node=loc_verify_leaf,
        sources=_build_sources(resort.location.location_urls, resort.basic.official_website_url),
        additional_instruction="Verify the page explicitly states the resort is on the specified Aruba beach (e.g., Palm Beach, Eagle Beach)."
    )

    # All-inclusive confirmation
    ai_node = evaluator.add_parallel(
        id=f"Resort_{rnum}_All_Inclusive_Confirmation",
        desc=f"Verify that the resort offers a true all-inclusive package for resort #{rnum}",
        parent=basic_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=_nonempty(resort.all_inclusive.confirmation_text),
        id=f"Resort_{rnum}_All_Inclusive_Package_Confirmed",
        desc="Confirmation that the resort offers a true all-inclusive package (not European Plan or room-only)",
        parent=ai_node,
        critical=True
    )
    ai_verify_leaf = evaluator.add_leaf(
        id=f"Resort_{rnum}_All_Inclusive_URL_Verification",
        desc="Valid URL supports the all-inclusive package information",
        parent=ai_node,
        critical=True
    )
    await evaluator.verify(
        claim="The resort offers a true all-inclusive plan that includes meals and beverages (not just room-only or European Plan).",
        node=ai_verify_leaf,
        sources=_build_sources(resort.all_inclusive.ai_urls, resort.basic.official_website_url),
        additional_instruction="Look for explicit 'all-inclusive' language including meals and drinks. If the page indicates EP/RO is standard, this claim is not supported."
    )

    # ------------------------ Accommodation Details ------------------------- #
    accom_node = evaluator.add_parallel(
        id=f"Resort_{rnum}_Accommodation_Details",
        desc=f"Verify accommodation features for resort #{rnum}",
        parent=resort_node,
        critical=True
    )

    # Suite accommodations
    suite_node = evaluator.add_parallel(
        id=f"Resort_{rnum}_Suite_Accommodations",
        desc="Verify availability of suite-level accommodations",
        parent=accom_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=_nonempty(resort.accommodation.suite_available_text),
        id=f"Resort_{rnum}_Suite_Availability",
        desc="Resort offers suite-level accommodations beyond standard rooms",
        parent=suite_node,
        critical=True
    )
    suite_verify_leaf = evaluator.add_leaf(
        id=f"Resort_{rnum}_Suite_URL_Verification",
        desc="Valid URL supports suite availability information",
        parent=suite_node,
        critical=True
    )
    await evaluator.verify(
        claim="The resort offers suites or suite-level accommodation categories.",
        node=suite_verify_leaf,
        sources=_build_sources(resort.accommodation.suite_urls, resort.basic.official_website_url),
        additional_instruction="Confirm presence of room categories explicitly labeled as 'suite' or equivalent (e.g., Junior Suite, One-Bedroom Suite)."
    )

    # Room features (balcony/terrace)
    room_feat_node = evaluator.add_parallel(
        id=f"Resort_{rnum}_Room_Features",
        desc="Verify room features including balconies/terraces",
        parent=accom_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=_nonempty(resort.accommodation.balcony_terrace_text),
        id=f"Resort_{rnum}_Balcony_or_Terrace",
        desc="Rooms or suites include private balconies or terraces",
        parent=room_feat_node,
        critical=True
    )
    room_feat_verify_leaf = evaluator.add_leaf(
        id=f"Resort_{rnum}_Room_Features_URL_Verification",
        desc="Valid URL supports room features information",
        parent=room_feat_node,
        critical=True
    )
    await evaluator.verify(
        claim="Rooms or suites at the resort include private balconies or terraces.",
        node=room_feat_verify_leaf,
        sources=_build_sources(resort.accommodation.room_features_urls, resort.basic.official_website_url),
        additional_instruction="Confirm phrasing like 'private balcony', 'private terrace', or similar in room descriptions."
    )

    # Direct beach access
    beach_access_node = evaluator.add_parallel(
        id=f"Resort_{rnum}_Beach_Access",
        desc="Verify direct beach access from resort property",
        parent=accom_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=_nonempty(resort.accommodation.direct_beach_access_text),
        id=f"Resort_{rnum}_Direct_Beach_Access",
        desc="Resort has direct beach access from the property",
        parent=beach_access_node,
        critical=True
    )
    beach_access_verify_leaf = evaluator.add_leaf(
        id=f"Resort_{rnum}_Beach_Access_URL_Verification",
        desc="Valid URL supports beach access information",
        parent=beach_access_node,
        critical=True
    )
    await evaluator.verify(
        claim="The resort provides direct access to the beach from the property (i.e., it is beachfront).",
        node=beach_access_verify_leaf,
        sources=_build_sources(resort.accommodation.beach_access_urls, resort.basic.official_website_url),
        additional_instruction="Look for explicit statements like 'direct beach access', 'on the beach', or 'beachfront' with clear indication of access."
    )

    # --------------------- Dining & All-Inclusive Package ------------------- #
    dining_node = evaluator.add_parallel(
        id=f"Resort_{rnum}_Dining_and_All_Inclusive",
        desc=f"Verify dining and all-inclusive package details for resort #{rnum}",
        parent=resort_node,
        critical=True
    )

    # Meal inclusions (3 meals)
    meals_node = evaluator.add_parallel(
        id=f"Resort_{rnum}_Meal_Inclusions",
        desc="Verify that three meals per day are included",
        parent=dining_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=_nonempty(resort.dining.meals_included_text),
        id=f"Resort_{rnum}_Three_Meals_Included",
        desc="All-inclusive package includes breakfast, lunch, and dinner",
        parent=meals_node,
        critical=True
    )
    meals_verify_leaf = evaluator.add_leaf(
        id=f"Resort_{rnum}_Meals_URL_Verification",
        desc="Valid URL supports meal inclusions information",
        parent=meals_node,
        critical=True
    )
    await evaluator.verify(
        claim="The all-inclusive package includes three meals per day: breakfast, lunch, and dinner.",
        node=meals_verify_leaf,
        sources=_build_sources(resort.dining.meals_urls, resort.basic.official_website_url),
        additional_instruction="Confirm phrasing that clearly includes all three meals; 'meals included' alone is insufficient unless it enumerates B/L/D or clearly implies all."
    )

    # Restaurant count (>=3)
    rest_node = evaluator.add_parallel(
        id=f"Resort_{rnum}_Restaurant_Count",
        desc="Verify minimum number of restaurants",
        parent=dining_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=_nonempty(resort.dining.restaurant_count_text),
        id=f"Resort_{rnum}_Minimum_Three_Restaurants",
        desc="Resort has at least 3 on-site restaurants or dining venues",
        parent=rest_node,
        critical=True
    )
    rest_verify_leaf = evaluator.add_leaf(
        id=f"Resort_{rnum}_Restaurants_URL_Verification",
        desc="Valid URL supports restaurant count information",
        parent=rest_node,
        critical=True
    )
    await evaluator.verify(
        claim="The resort has at least 3 on-site restaurants or dining venues.",
        node=rest_verify_leaf,
        sources=_build_sources(resort.dining.restaurants_urls, resort.basic.official_website_url),
        additional_instruction="Validate that there are 3 or more distinct on-site dining venues serving meals (buffet/à la carte). Bars without meal service should not be counted."
    )

    # Beverages (unlimited alcoholic & non-alcoholic)
    bev_node = evaluator.add_parallel(
        id=f"Resort_{rnum}_Beverage_Inclusions",
        desc="Verify unlimited beverage inclusions",
        parent=dining_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=_nonempty(resort.dining.beverages_included_text),
        id=f"Resort_{rnum}_Unlimited_Beverages",
        desc="All-inclusive package includes unlimited alcoholic and non-alcoholic beverages",
        parent=bev_node,
        critical=True
    )
    bev_verify_leaf = evaluator.add_leaf(
        id=f"Resort_{rnum}_Beverages_URL_Verification",
        desc="Valid URL supports beverage inclusions information",
        parent=bev_node,
        critical=True
    )
    await evaluator.verify(
        claim="Unlimited alcoholic and non-alcoholic beverages are included in the all-inclusive plan.",
        node=bev_verify_leaf,
        sources=_build_sources(resort.dining.beverages_urls, resort.basic.official_website_url),
        additional_instruction="Confirm explicit mention of unlimited drinks (alcoholic and non-alcoholic) as part of the plan."
    )

    # ----------------------- Amenities & Facilities ------------------------- #
    amenities_node = evaluator.add_parallel(
        id=f"Resort_{rnum}_Amenities_and_Facilities",
        desc=f"Verify amenities and facilities for resort #{rnum}",
        parent=resort_node,
        critical=True
    )

    # Pools (>=2 outdoor)
    pools_node = evaluator.add_parallel(
        id=f"Resort_{rnum}_Pool_Facilities",
        desc="Verify minimum number of outdoor pools",
        parent=amenities_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=_nonempty(resort.amenities.pools_count_text),
        id=f"Resort_{rnum}_Minimum_Two_Pools",
        desc="Resort has at least 2 outdoor swimming pools",
        parent=pools_node,
        critical=True
    )
    pools_verify_leaf = evaluator.add_leaf(
        id=f"Resort_{rnum}_Pools_URL_Verification",
        desc="Valid URL supports pool facilities information",
        parent=pools_node,
        critical=True
    )
    await evaluator.verify(
        claim="The resort has at least 2 outdoor swimming pools.",
        node=pools_verify_leaf,
        sources=_build_sources(resort.amenities.pools_urls, resort.basic.official_website_url),
        additional_instruction="Verify mention of two or more distinct outdoor pools; kids' pools count if separate."
    )

    # Complimentary non-motorized water sports
    ws_node = evaluator.add_parallel(
        id=f"Resort_{rnum}_Water_Sports",
        desc="Verify complimentary water sports availability",
        parent=amenities_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=_nonempty(resort.amenities.water_sports_text),
        id=f"Resort_{rnum}_Complimentary_Water_Sports",
        desc="Complimentary non-motorized water sports are included in the all-inclusive package",
        parent=ws_node,
        critical=True
    )
    ws_verify_leaf = evaluator.add_leaf(
        id=f"Resort_{rnum}_Water_Sports_URL_Verification",
        desc="Valid URL supports water sports information",
        parent=ws_node,
        critical=True
    )
    await evaluator.verify(
        claim="Complimentary non-motorized water sports (e.g., kayaks, paddleboards, snorkeling gear) are included.",
        node=ws_verify_leaf,
        sources=_build_sources(resort.amenities.water_sports_urls, resort.basic.official_website_url),
        additional_instruction="Confirm inclusion as part of the all-inclusive package (no extra fee)."
    )

    # On-site spa with treatment rooms
    spa_node = evaluator.add_parallel(
        id=f"Resort_{rnum}_Spa_Facility",
        desc="Verify on-site spa with treatment rooms",
        parent=amenities_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=_nonempty(resort.amenities.spa_facility_text),
        id=f"Resort_{rnum}_On_Site_Spa",
        desc="Resort has an on-site spa facility with treatment rooms",
        parent=spa_node,
        critical=True
    )
    spa_verify_leaf = evaluator.add_leaf(
        id=f"Resort_{rnum}_Spa_URL_Verification",
        desc="Valid URL supports spa facility information",
        parent=spa_node,
        critical=True
    )
    await evaluator.verify(
        claim="The resort has an on-site spa facility with treatment rooms.",
        node=spa_verify_leaf,
        sources=_build_sources(resort.amenities.spa_urls, resort.basic.official_website_url),
        additional_instruction="Look for 'spa', 'treatment rooms', or equivalent on the official or brand site."
    )

    # --------------------------- Booking Policies --------------------------- #
    booking_node = evaluator.add_parallel(
        id=f"Resort_{rnum}_Booking_Policies",
        desc=f"Verify booking policies for resort #{rnum}",
        parent=resort_node,
        critical=True
    )

    # Check-in age
    age_node = evaluator.add_parallel(
        id=f"Resort_{rnum}_Check_In_Requirements",
        desc="Verify minimum check-in age requirement",
        parent=booking_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=_nonempty(resort.booking.check_in_age_text),
        id=f"Resort_{rnum}_Check_In_Age_Stated",
        desc="Minimum check-in age requirement is clearly stated",
        parent=age_node,
        critical=True
    )
    age_verify_leaf = evaluator.add_leaf(
        id=f"Resort_{rnum}_Age_URL_Verification",
        desc="Valid URL supports check-in age information",
        parent=age_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The resort specifies a minimum check-in age requirement. Details: {resort.booking.check_in_age_text or 'N/A'}.",
        node=age_verify_leaf,
        sources=_build_sources(resort.booking.age_urls, resort.basic.official_website_url),
        additional_instruction="Verify that a minimum age is stated for check-in (e.g., 18+ or 21+)."
    )

    # Deposit requirement
    dep_node = evaluator.add_parallel(
        id=f"Resort_{rnum}_Deposit_Requirements",
        desc="Verify deposit amount for reservations",
        parent=booking_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=_nonempty(resort.booking.deposit_text),
        id=f"Resort_{rnum}_Deposit_Requirement_Stated",
        desc="Deposit amount required for reservations is clearly stated",
        parent=dep_node,
        critical=True
    )
    dep_verify_leaf = evaluator.add_leaf(
        id=f"Resort_{rnum}_Deposit_URL_Verification",
        desc="Valid URL supports deposit requirement information",
        parent=dep_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The resort requires a deposit for reservations. Details: {resort.booking.deposit_text or 'N/A'}.",
        node=dep_verify_leaf,
        sources=_build_sources(resort.booking.deposit_urls, resort.basic.official_website_url),
        additional_instruction="Confirm deposit requirement and amount/timing if available (e.g., 1 night, percentage)."
    )

    # Cancellation terms
    canc_node = evaluator.add_parallel(
        id=f"Resort_{rnum}_Cancellation_Terms",
        desc="Verify cancellation policy details",
        parent=booking_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=_nonempty(resort.booking.cancellation_text),
        id=f"Resort_{rnum}_Cancellation_Policy_Provided",
        desc="Cancellation policy with specific timeframes and fees is provided",
        parent=canc_node,
        critical=True
    )
    canc_verify_leaf = evaluator.add_leaf(
        id=f"Resort_{rnum}_Cancellation_URL_Verification",
        desc="Valid URL supports cancellation policy information",
        parent=canc_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The resort provides a cancellation policy with specific timeframes and/or fees. Details: {resort.booking.cancellation_text or 'N/A'}.",
        node=canc_verify_leaf,
        sources=_build_sources(resort.booking.cancellation_urls, resort.basic.official_website_url),
        additional_instruction="Confirm existence of a policy with a timeframe (e.g., 72 hours) and any penalties/fees."
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
    # Initialize evaluator (root is non-critical to allow partial scoring)
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

    # Extract structured resorts info from the answer
    extraction = await evaluator.extract(
        prompt=prompt_extract_resorts(),
        template_class=ResortsExtraction,
        extraction_name="resorts_structured"
    )

    # Normalize to exactly 3 items (truncate or pad with empty)
    resorts: List[ResortItem] = list(extraction.resorts[:3])
    while len(resorts) < 3:
        resorts.append(ResortItem())

    # Add a wrapper node reflecting the rubric's top-level description (non-critical)
    task_node = evaluator.add_parallel(
        id="Find_Three_Luxury_All_Inclusive_Resorts_in_Aruba",
        desc="Evaluate whether the answer provides up to three luxury all-inclusive resorts in Aruba that meet the specified criteria",
        parent=root,
        critical=False
    )

    # Build verification subtrees for the three resorts
    for i in range(3):
        await verify_resort(evaluator, task_node, resorts[i], i)

    # Return the evaluation summary
    return evaluator.get_summary()