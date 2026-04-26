import asyncio
import logging
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "southern_ca_accessible_hotel"
TASK_DESCRIPTION = (
    "Identify a hotel located in Southern California's coastal region (specifically in San Diego County or Orange County) "
    "that satisfies all of the following requirements:\n\n"
    "Location Requirements:\n"
    "- Must be located within 2 miles of the Pacific Ocean coastline\n"
    "- Must be in either San Diego County or Orange County, California\n"
    "- Must be within reasonable proximity to wheelchair-accessible beach facilities\n\n"
    "Accessibility Requirements:\n"
    "- Must offer ADA-compliant wheelchair accessible guest rooms with mobility features\n"
    "- Accessible rooms must include doorways with minimum 32-inch clear width\n"
    "- Accessible rooms must feature either roll-in showers or accessible bathtubs with grab bars\n"
    "- Accessible rooms must provide at least 400 square feet of floor space\n"
    "- Must have grab bars properly installed in bathroom areas of accessible rooms\n\n"
    "Quality and Amenities:\n"
    "- Must have achieved at least an AAA Three Diamond rating\n"
    "- Must offer at least one accessible room option that includes ocean or bay views\n"
    "- Must be pet-friendly, accepting dogs weighing up to 50 pounds\n\n"
    "For your answer, provide the hotel name, complete address, and verifiable reference URLs that confirm each requirement is met."
)


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class HotelSources(BaseModel):
    identification_urls: List[str] = Field(default_factory=list, description="URLs confirming hotel name/address")
    county_urls: List[str] = Field(default_factory=list, description="URLs that establish the hotel's county")
    ocean_proximity_urls: List[str] = Field(default_factory=list, description="URLs showing proximity to the Pacific ocean within 2 miles, or beachfront")
    beach_access_urls: List[str] = Field(default_factory=list, description="URLs showing wheelchair-accessible beach facilities near the hotel")

    ada_rooms_urls: List[str] = Field(default_factory=list, description="URLs confirming ADA mobility accessible guestrooms")
    doorway_width_urls: List[str] = Field(default_factory=list, description="URLs confirming 32-inch minimum clear doorway width in accessible rooms")
    bath_fixture_urls: List[str] = Field(default_factory=list, description="URLs confirming roll-in shower or accessible bathtub in accessible rooms")
    grab_bars_urls: List[str] = Field(default_factory=list, description="URLs confirming bathroom grab bars in accessible rooms")
    room_size_urls: List[str] = Field(default_factory=list, description="URLs confirming accessible room floor space at least 400 sq ft")

    aaa_rating_urls: List[str] = Field(default_factory=list, description="URLs confirming AAA Three Diamond rating or higher")
    accessible_view_urls: List[str] = Field(default_factory=list, description="URLs confirming at least one accessible room offers ocean or bay views")
    pet_friendly_urls: List[str] = Field(default_factory=list, description="URLs confirming pet friendly policy accepting dogs up to 50 lbs")


class HotelExtraction(BaseModel):
    name: Optional[str] = None
    address: Optional[str] = None
    sources: HotelSources = Field(default_factory=HotelSources)

    # Optional textual fields extracted verbatim from the answer (if present)
    doorway_width_text: Optional[str] = None
    room_size_text: Optional[str] = None
    bath_fixture_text: Optional[str] = None
    accessible_view_text: Optional[str] = None
    pet_policy_text: Optional[str] = None
    aaa_rating_text: Optional[str] = None


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_hotel() -> str:
    return """
    Extract the primary hotel and all supporting URLs grouped by requirement from the answer. If multiple hotels are mentioned, pick the first one as the primary hotel.

    Return a JSON object with fields:
    - name: The hotel name exactly as given in the answer (string or null if missing).
    - address: The complete address exactly as presented in the answer (string or null if missing). Include street, city, state, and ZIP if present; otherwise return what is provided.
    - doorway_width_text: If the answer mentions a specific doorway width for accessible rooms (e.g., "32 inches"), extract that phrase; else null.
    - room_size_text: If the answer mentions accessible room size (e.g., "400 sq ft"), extract that phrase; else null.
    - bath_fixture_text: If the answer mentions roll-in showers or accessible bathtubs with grab bars, extract that phrase; else null.
    - accessible_view_text: If the answer mentions accessible rooms with an ocean or bay view, extract the phrase; else null.
    - pet_policy_text: If the answer mentions dogs or weight limits for pets, extract the phrase; else null.
    - aaa_rating_text: If the answer mentions AAA rating (e.g., "AAA Three Diamond"), extract the phrase; else null.

    - sources: An object with the following URL arrays. Include only explicit URLs that appear in the answer. If a category has no URLs, return an empty array.
        • identification_urls: URLs that confirm hotel identity/address (e.g., official site contact/location page, Google Maps link).
        • county_urls: URLs that help establish the hotel's county (e.g., an official site page specifying county, or a page that clearly indicates the city and county).
        • ocean_proximity_urls: URLs that show the hotel is within 2 miles of the Pacific Ocean coastline; accept pages that clearly indicate beachfront/oceanfront or show distances/maps.
        • beach_access_urls: URLs showing nearby wheelchair-accessible beach facilities (e.g., official city/county beach accessibility pages, access maps).
        • ada_rooms_urls: URLs confirming ADA mobility accessible guestrooms exist.
        • doorway_width_urls: URLs explicitly mentioning accessible room doorways with at least 32-inch clear width.
        • bath_fixture_urls: URLs confirming roll-in shower or accessible bathtub for accessible rooms.
        • grab_bars_urls: URLs confirming bathroom grab bars for accessible rooms.
        • room_size_urls: URLs indicating accessible room floor area meets or exceeds 400 square feet.
        • aaa_rating_urls: URLs confirming AAA Three Diamond (or higher) rating.
        • accessible_view_urls: URLs confirming at least one accessible room option has ocean or bay views.
        • pet_friendly_urls: URLs confirming pet policy that allows dogs up to 50 pounds.

    SPECIAL RULES FOR URL EXTRACTION:
    - Extract only URLs explicitly present in the answer. Do not invent or infer any URLs.
    - Accept URLs in plain form or markdown link form. Return the actual URL strings.
    - If no URL for a category is provided in the answer, return an empty array for that category.
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def _dedup_urls(urls: List[str]) -> List[str]:
    seen = set()
    deduped: List[str] = []
    for u in urls:
        if not isinstance(u, str):
            continue
        v = u.strip()
        if not v:
            continue
        if v not in seen:
            seen.add(v)
            deduped.append(v)
    return deduped


def _merge_urls(*url_lists: List[str]) -> List[str]:
    merged: List[str] = []
    for lst in url_lists:
        merged.extend(lst or [])
    return _dedup_urls(merged)


def _all_requirement_url_lists_non_empty(h: HotelExtraction) -> bool:
    s = h.sources or HotelSources()
    required_lists = [
        s.identification_urls,
        s.county_urls,
        s.ocean_proximity_urls,
        s.beach_access_urls,
        s.ada_rooms_urls,
        s.doorway_width_urls,
        s.bath_fixture_urls,
        s.grab_bars_urls,
        s.room_size_urls,
        s.aaa_rating_urls,
        s.accessible_view_urls,
        s.pet_friendly_urls,
    ]
    return all(isinstance(lst, list) and len(lst) > 0 for lst in required_lists)


# --------------------------------------------------------------------------- #
# Verification subroutines                                                    #
# --------------------------------------------------------------------------- #
async def build_required_output_fields(evaluator: Evaluator, parent_node, hotel: HotelExtraction) -> None:
    required_node = evaluator.add_parallel(
        id="Required_Output_Fields",
        desc="Answer includes required identifying information",
        parent=parent_node,
        critical=True,
    )

    evaluator.add_custom_node(
        result=bool(hotel.name and str(hotel.name).strip()),
        id="Hotel_Name_Provided",
        desc="Answer provides the hotel name",
        parent=required_node,
        critical=True,
    )

    evaluator.add_custom_node(
        result=bool(hotel.address and str(hotel.address).strip()),
        id="Complete_Address_Provided",
        desc="Answer provides the complete address",
        parent=required_node,
        critical=True,
    )


async def build_geographic_location(evaluator: Evaluator, parent_node, hotel: HotelExtraction) -> None:
    geo_node = evaluator.add_parallel(
        id="Geographic_Location",
        desc="Verify hotel location meets geographic requirements",
        parent=parent_node,
        critical=True,
    )

    # County requirement
    county_node = evaluator.add_leaf(
        id="County_Requirement",
        desc="Hotel is located in San Diego County or Orange County, California",
        parent=geo_node,
        critical=True,
    )
    county_claim = (
        f"The hotel '{hotel.name or ''}' at address '{hotel.address or ''}' is located in San Diego County "
        f"or Orange County, California."
    )
    county_sources = _merge_urls(hotel.sources.identification_urls, hotel.sources.county_urls)
    await evaluator.verify(
        claim=county_claim,
        node=county_node,
        sources=county_sources,
        additional_instruction="Use the evidence on the page(s) to confirm the county. It is acceptable if the page explicitly states 'San Diego County' or 'Orange County', or clearly shows a city that belongs to those counties. Do not rely on the assistant's external knowledge beyond what is shown on the provided page(s).",
    )

    # Ocean proximity (<= 2 miles)
    ocean_node = evaluator.add_leaf(
        id="Ocean_Proximity",
        desc="Hotel is within 2 miles of the Pacific Ocean coastline",
        parent=geo_node,
        critical=True,
    )
    ocean_claim = "The hotel is within 2 miles of the Pacific Ocean coastline."
    ocean_sources = _merge_urls(hotel.sources.ocean_proximity_urls, hotel.sources.identification_urls)
    await evaluator.verify(
        claim=ocean_claim,
        node=ocean_node,
        sources=ocean_sources,
        additional_instruction="Confirm via the page content (e.g., beachfront/oceanfront description, stated distance to the beach, or map evidence) that the hotel is no more than 2 miles from the Pacific coast.",
    )

    # Wheelchair-accessible beach facilities nearby
    beach_node = evaluator.add_leaf(
        id="Beach_Access",
        desc="Hotel is near wheelchair-accessible beach facilities",
        parent=geo_node,
        critical=True,
    )
    beach_claim = "There are wheelchair-accessible beach facilities near the hotel."
    beach_sources = _merge_urls(hotel.sources.beach_access_urls, hotel.sources.identification_urls)
    await evaluator.verify(
        claim=beach_claim,
        node=beach_node,
        sources=beach_sources,
        additional_instruction="It suffices if a referenced page clearly indicates accessible beach facilities in close proximity to the hotel's location (e.g., beach access maps, official city/county accessibility pages).",
    )


async def build_accessibility_compliance(evaluator: Evaluator, parent_node, hotel: HotelExtraction) -> None:
    acc_node = evaluator.add_parallel(
        id="Accessibility_Compliance",
        desc="Verify hotel meets ADA accessibility requirements for guest rooms",
        parent=parent_node,
        critical=True,
    )

    # ADA-compliant accessible guest rooms
    ada_rooms_leaf = evaluator.add_leaf(
        id="ADA_Rooms_Available",
        desc="Hotel offers ADA-compliant accessible guest rooms with mobility features",
        parent=acc_node,
        critical=True,
    )
    ada_rooms_claim = "The hotel offers ADA-compliant wheelchair accessible guest rooms with mobility features."
    await evaluator.verify(
        claim=ada_rooms_claim,
        node=ada_rooms_leaf,
        sources=hotel.sources.ada_rooms_urls,
        additional_instruction="Look for 'ADA', 'mobility accessible', 'wheelchair accessible', 'accessible room' or similar language indicating compliance.",
    )

    # Doorway 32-inch minimum clear width
    door_leaf = evaluator.add_leaf(
        id="Doorway_Width",
        desc="Accessible rooms have doorways with minimum 32-inch clear width",
        parent=acc_node,
        critical=True,
    )
    door_claim = "Accessible guest room doorways provide a minimum clear width of at least 32 inches."
    await evaluator.verify(
        claim=door_claim,
        node=door_leaf,
        sources=hotel.sources.doorway_width_urls,
        additional_instruction="Verify the page explicitly mentions door width for accessible rooms being 32 inches or greater (allow variants like 32 in, 32-inch).",
    )

    # Roll-in shower or accessible bathtub
    bath_leaf = evaluator.add_leaf(
        id="Accessible_Bath_Fixture",
        desc="Accessible rooms include either roll-in showers or accessible bathtubs",
        parent=acc_node,
        critical=True,
    )
    bath_claim = "Accessible rooms include either a roll-in shower or an accessible bathtub (with grab bars)."
    await evaluator.verify(
        claim=bath_claim,
        node=bath_leaf,
        sources=hotel.sources.bath_fixture_urls,
        additional_instruction="Accept if the page clearly states roll-in shower OR accessible tub/bathtub availability for the accessible rooms.",
    )

    # Grab bars in bathroom
    grab_leaf = evaluator.add_leaf(
        id="Grab_Bars",
        desc="Accessible rooms have grab bars in bathroom areas (properly installed as required)",
        parent=acc_node,
        critical=True,
    )
    grab_claim = "Accessible guest rooms include bathroom grab bars."
    await evaluator.verify(
        claim=grab_claim,
        node=grab_leaf,
        sources=hotel.sources.grab_bars_urls,
        additional_instruction="Look for mention of grab bars in bath/shower/toilet areas of accessible rooms.",
    )

    # Room size >= 400 sq ft
    size_leaf = evaluator.add_leaf(
        id="Room_Size",
        desc="Accessible rooms provide at least 400 square feet of floor space",
        parent=acc_node,
        critical=True,
    )
    size_claim = "At least one accessible room provides a minimum of 400 square feet of floor space."
    await evaluator.verify(
        claim=size_claim,
        node=size_leaf,
        sources=hotel.sources.room_size_urls,
        additional_instruction="Verify the page explicitly mentions square footage for an accessible room that is 400 sq ft or greater.",
    )


async def build_quality_standard(evaluator: Evaluator, parent_node, hotel: HotelExtraction) -> None:
    qual_node = evaluator.add_parallel(
        id="Quality_Standard",
        desc="Verify hotel meets quality rating requirement",
        parent=parent_node,
        critical=True,
    )

    aaa_leaf = evaluator.add_leaf(
        id="AAA_Diamond_Rating",
        desc="Hotel has achieved at least an AAA Three Diamond rating",
        parent=qual_node,
        critical=True,
    )
    aaa_claim = "The hotel has an AAA Three Diamond rating or higher."
    await evaluator.verify(
        claim=aaa_claim,
        node=aaa_leaf,
        sources=hotel.sources.aaa_rating_urls,
        additional_instruction="Confirm via AAA or other credible source page that the hotel is rated AAA Three Diamond or above.",
    )


async def build_amenities(evaluator: Evaluator, parent_node, hotel: HotelExtraction) -> None:
    amen_node = evaluator.add_parallel(
        id="Amenities",
        desc="Verify hotel offers required amenities",
        parent=parent_node,
        critical=True,
    )

    view_leaf = evaluator.add_leaf(
        id="Ocean_Or_Bay_View_Accessible",
        desc="Hotel offers at least one accessible room option with ocean or bay views",
        parent=amen_node,
        critical=True,
    )
    view_claim = "At least one accessible room option includes ocean or bay views."
    await evaluator.verify(
        claim=view_claim,
        node=view_leaf,
        sources=hotel.sources.accessible_view_urls,
        additional_instruction="The page should indicate that an accessible room type offers an ocean view or bay view.",
    )

    pet_leaf = evaluator.add_leaf(
        id="Pet_Friendly",
        desc="Hotel is pet-friendly and accepts dogs up to 50 pounds",
        parent=amen_node,
        critical=True,
    )
    pet_claim = "The hotel is pet-friendly and accepts dogs up to 50 pounds."
    await evaluator.verify(
        claim=pet_claim,
        node=pet_leaf,
        sources=hotel.sources.pet_friendly_urls,
        additional_instruction="Look for pet policy details explicitly stating dogs are allowed with a maximum weight of 50 lbs (or higher).",
    )


def build_reference_urls(evaluator: Evaluator, parent_node, hotel: HotelExtraction) -> None:
    ref_node = evaluator.add_parallel(
        id="Reference_URLs",
        desc="Answer provides verifiable reference URLs supporting each requirement",
        parent=parent_node,
        critical=True,
    )

    # URLs provided at all
    all_urls = _merge_urls(
        hotel.sources.identification_urls,
        hotel.sources.county_urls,
        hotel.sources.ocean_proximity_urls,
        hotel.sources.beach_access_urls,
        hotel.sources.ada_rooms_urls,
        hotel.sources.doorway_width_urls,
        hotel.sources.bath_fixture_urls,
        hotel.sources.grab_bars_urls,
        hotel.sources.room_size_urls,
        hotel.sources.aaa_rating_urls,
        hotel.sources.accessible_view_urls,
        hotel.sources.pet_friendly_urls,
    )
    evaluator.add_custom_node(
        result=len(all_urls) > 0,
        id="URLs_Provided",
        desc="Answer includes reference URLs",
        parent=ref_node,
        critical=True,
    )

    # URLs cover each requirement (presence-based check over categories)
    cover_all = _all_requirement_url_lists_non_empty(hotel)
    evaluator.add_custom_node(
        result=cover_all,
        id="URLs_Cover_All_Requirements",
        desc="Provided URLs collectively support/confirm each stated requirement and the provided hotel details",
        parent=ref_node,
        critical=True,
    )

    # Optionally, record some counts
    evaluator.add_custom_info(
        info={
            "total_urls": len(all_urls),
            "by_category_counts": {
                "identification": len(hotel.sources.identification_urls),
                "county": len(hotel.sources.county_urls),
                "ocean_proximity": len(hotel.sources.ocean_proximity_urls),
                "beach_access": len(hotel.sources.beach_access_urls),
                "ada_rooms": len(hotel.sources.ada_rooms_urls),
                "doorway_width": len(hotel.sources.doorway_width_urls),
                "bath_fixture": len(hotel.sources.bath_fixture_urls),
                "grab_bars": len(hotel.sources.grab_bars_urls),
                "room_size": len(hotel.sources.room_size_urls),
                "aaa_rating": len(hotel.sources.aaa_rating_urls),
                "accessible_view": len(hotel.sources.accessible_view_urls),
                "pet_friendly": len(hotel.sources.pet_friendly_urls),
            },
        },
        info_type="debug",
        info_name="url_statistics",
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
    Evaluate an answer for the Southern California accessible hotel task.
    """
    # Initialize evaluator with a parallel root
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

    # Extract structured hotel info and requirement-specific URLs
    hotel_info = await evaluator.extract(
        prompt=prompt_extract_hotel(),
        template_class=HotelExtraction,
        extraction_name="hotel_extraction",
    )

    # Top-level node mirroring the rubric
    top_node = evaluator.add_parallel(
        id="Hotel_Identification",
        desc="Identify a hotel meeting all specified requirements in Southern California coastal region and provide required details and references",
        parent=root,
        critical=True,
    )

    # 1) Required output fields (presence checks)
    await build_required_output_fields(evaluator, top_node, hotel_info)

    # 2) Geographic location checks
    await build_geographic_location(evaluator, top_node, hotel_info)

    # 3) Accessibility compliance checks
    await build_accessibility_compliance(evaluator, top_node, hotel_info)

    # 4) Quality standard (AAA)
    await build_quality_standard(evaluator, top_node, hotel_info)

    # 5) Amenities checks
    await build_amenities(evaluator, top_node, hotel_info)

    # 6) Reference URLs presence/coverage (computed last to avoid gating other verifies prematurely)
    build_reference_urls(evaluator, top_node, hotel_info)

    # Return structured evaluation summary
    return evaluator.get_summary()