import asyncio
import logging
from typing import Any, Optional, List, Dict

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy

# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "recreational_facilities_2025_factbook"
TASK_DESCRIPTION = (
    "A travel research company is compiling a factbook about record-breaking and unique recreational facilities "
    "in the United States for their 2025 publication. They need verified information about three specific facilities:\n\n"
    "1. The hotel with exclusive entrance to Epic Universe: Identify the hotel at Universal Orlando Resort that offers guests "
    "a dedicated, exclusive entrance directly into the Epic Universe theme park (an entrance that only guests of this specific hotel can use). "
    "Provide the hotel's total room count and its opening date.\n\n"
    "2. The highest elevation visitor center in the National Park Service: Identify the visitor center that holds the record for highest elevation "
    "in the entire National Park Service system. Provide its exact elevation in feet, the specific mountain pass where it is located, "
    "the national park it belongs to, the road that provides access to it, and information about its seasonal operation.\n\n"
    "3. The world's tallest and longest single-rail roller coaster: Identify the roller coaster that holds both world records for being the tallest "
    "AND longest single-rail coaster. Provide the theme park where it is located, its height in feet, its track length in feet, its top speed in mph, "
    "and the minimum height requirement for riders in inches.\n\n"
    "For each facility and each specification, provide a reference URL from your research that verifies the information."
)

# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class HotelEpicEntrance(BaseModel):
    """Facility 1: Universal Orlando hotel with exclusive Epic Universe entrance."""
    hotel_name: Optional[str] = None
    exclusive_entrance_urls: List[str] = Field(default_factory=list)

    total_room_count: Optional[str] = None
    room_count_urls: List[str] = Field(default_factory=list)

    opening_date: Optional[str] = None
    opening_date_urls: List[str] = Field(default_factory=list)


class NPSHighestVisitorCenter(BaseModel):
    """Facility 2: Highest-elevation NPS visitor center."""
    visitor_center_name: Optional[str] = None
    highest_record_urls: List[str] = Field(default_factory=list)

    elevation_feet: Optional[str] = None
    elevation_urls: List[str] = Field(default_factory=list)

    mountain_pass: Optional[str] = None
    mountain_pass_urls: List[str] = Field(default_factory=list)

    national_park: Optional[str] = None
    national_park_urls: List[str] = Field(default_factory=list)

    access_road: Optional[str] = None
    access_road_urls: List[str] = Field(default_factory=list)

    seasonal_operation: Optional[str] = None
    seasonal_urls: List[str] = Field(default_factory=list)


class SingleRailCoasterRecord(BaseModel):
    """Facility 3: World's tallest and longest single-rail roller coaster."""
    coaster_name: Optional[str] = None
    record_urls: List[str] = Field(default_factory=list)

    theme_park: Optional[str] = None
    theme_park_urls: List[str] = Field(default_factory=list)

    height_feet: Optional[str] = None
    height_urls: List[str] = Field(default_factory=list)

    track_length_feet: Optional[str] = None
    track_length_urls: List[str] = Field(default_factory=list)

    top_speed_mph: Optional[str] = None
    top_speed_urls: List[str] = Field(default_factory=list)

    min_rider_height_inches: Optional[str] = None
    min_height_urls: List[str] = Field(default_factory=list)


class FacilitiesExtraction(BaseModel):
    """Top-level extraction for all three facilities."""
    hotel: Optional[HotelEpicEntrance] = None
    visitor_center: Optional[NPSHighestVisitorCenter] = None
    coaster: Optional[SingleRailCoasterRecord] = None


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_facilities() -> str:
    return """
    Extract structured information for three facilities from the answer. Keep values exactly as stated in the answer (use strings for numbers/dates). 
    For each subfield that requires verification, extract the associated reference URLs explicitly mentioned in the answer. 
    If a field is missing, set it to null. If URLs are not provided for an attribute, return an empty array for that attribute's URLs.

    Return a JSON object with this shape:

    {
      "hotel": {
        "hotel_name": string|null,
        "exclusive_entrance_urls": string[],

        "total_room_count": string|null,
        "room_count_urls": string[],

        "opening_date": string|null,
        "opening_date_urls": string[]
      },
      "visitor_center": {
        "visitor_center_name": string|null,
        "highest_record_urls": string[],

        "elevation_feet": string|null,
        "elevation_urls": string[],

        "mountain_pass": string|null,
        "mountain_pass_urls": string[],

        "national_park": string|null,
        "national_park_urls": string[],

        "access_road": string|null,
        "access_road_urls": string[],

        "seasonal_operation": string|null,
        "seasonal_urls": string[]
      },
      "coaster": {
        "coaster_name": string|null,
        "record_urls": string[],

        "theme_park": string|null,
        "theme_park_urls": string[],

        "height_feet": string|null,
        "height_urls": string[],

        "track_length_feet": string|null,
        "track_length_urls": string[],

        "top_speed_mph": string|null,
        "top_speed_urls": string[],

        "min_rider_height_inches": string|null,
        "min_height_urls": string[]
      }
    }

    Clarifications:
    - exclusive_entrance_urls: URLs that explicitly state the hotel offers an exclusive/dedicated entrance into Epic Universe, only for its guests.
    - highest_record_urls: URLs that explicitly state the visitor center holds the highest elevation record in the National Park Service system.
    - elevation_urls/mountain_pass_urls/national_park_urls/access_road_urls/seasonal_urls: URLs that verify each respective attribute.
    - record_urls for the coaster: URLs that explicitly state the coaster holds BOTH records (tallest and longest) for single-rail coasters.
    - For numeric fields (e.g., heights, lengths, speeds), keep them as strings exactly as written in the answer (e.g., "130 ft", "3,000 feet", "60 mph").
    """


# --------------------------------------------------------------------------- #
# Verification helper functions                                               #
# --------------------------------------------------------------------------- #
async def verify_facility_1(
    evaluator: Evaluator,
    parent_node,
    hotel: HotelEpicEntrance,
) -> None:
    """
    Facility 1: Hotel at Universal Orlando Resort with dedicated/exclusive entrance to Epic Universe.
    Includes room count and opening date, each verified by a URL.
    """
    f_node = evaluator.add_sequential(
        id="Facility_1_Epic_Universe_Exclusive_Entrance_Hotel",
        desc="Hotel at Universal Orlando Resort with a dedicated/exclusive entrance into Epic Universe; include room count and opening date, each verified by a reference URL.",
        parent=parent_node,
        critical=False
    )

    # Identify qualifying hotel (critical)
    identify_node = evaluator.add_parallel(
        id="Identify_Qualifying_Hotel",
        desc="Correctly identifies a hotel that meets the exclusive/dedicated Epic Universe entrance criterion, with verification.",
        parent=f_node,
        critical=True
    )

    # Hotel name provided
    evaluator.add_custom_node(
        result=(hotel is not None and hotel.hotel_name is not None and hotel.hotel_name.strip() != ""),
        id="Hotel_Name_Provided",
        desc="Provides the hotel name (unambiguously identifiable).",
        parent=identify_node,
        critical=True
    )

    # Exclusive entrance URL presence (explicitly ensure URL exists)
    evaluator.add_custom_node(
        result=(hotel is not None and len(hotel.exclusive_entrance_urls) > 0),
        id="Exclusive_Entrance_URLs_Provided",
        desc="Provides at least one reference URL for the exclusive/dedicated entrance claim.",
        parent=identify_node,
        critical=True
    )

    # Verify exclusive entrance claim using URLs
    exclusive_leaf = evaluator.add_leaf(
        id="Exclusive_Entrance_Verified_With_URL",
        desc="Provides a reference URL that supports the claim that this hotel offers a dedicated/exclusive entrance directly into Epic Universe (usable only by guests of that hotel).",
        parent=identify_node,
        critical=True
    )
    claim_exclusive = (
        f"The hotel '{hotel.hotel_name}' offers an exclusive/dedicated entrance directly into Epic Universe that only its own hotel guests may use."
    )
    await evaluator.verify(
        claim=claim_exclusive,
        node=exclusive_leaf,
        sources=hotel.exclusive_entrance_urls,
        additional_instruction=(
            "Confirm that the referenced page explicitly indicates a 'private', 'exclusive', or 'dedicated' entrance "
            "from the named hotel into Epic Universe, restricted to that hotel's guests. Proximity or general early entry "
            "does NOT count. The evidence must clearly support exclusivity of the entrance."
        ),
    )

    # Required hotel details (critical)
    details_node = evaluator.add_parallel(
        id="Hotel_Required_Details",
        desc="Provides the required hotel attributes, each with a verifying reference URL.",
        parent=f_node,
        critical=True
    )

    # Room count URL presence
    evaluator.add_custom_node(
        result=(hotel is not None and hotel.total_room_count is not None and hotel.total_room_count.strip() != "" and len(hotel.room_count_urls) > 0),
        id="Total_Room_Count_URLs_Provided",
        desc="Provides the hotel's total room/guest-room count and includes at least one reference URL for it.",
        parent=details_node,
        critical=True
    )
    # Room count verification
    room_count_leaf = evaluator.add_leaf(
        id="Total_Room_Count_With_URL",
        desc="Provides the hotel's total room/guest-room count and a reference URL that verifies it.",
        parent=details_node,
        critical=True
    )
    claim_room_count = f"The total room/guest-room count for '{hotel.hotel_name}' is {hotel.total_room_count}."
    await evaluator.verify(
        claim=claim_room_count,
        node=room_count_leaf,
        sources=hotel.room_count_urls,
        additional_instruction="Verify the total guest-room count (rooms). Accept reasonable synonyms such as 'guest rooms'."
    )

    # Opening date URL presence
    evaluator.add_custom_node(
        result=(hotel is not None and hotel.opening_date is not None and hotel.opening_date.strip() != "" and len(hotel.opening_date_urls) > 0),
        id="Opening_Date_URLs_Provided",
        desc="Provides the hotel's opening date and includes at least one reference URL for it.",
        parent=details_node,
        critical=True
    )
    # Opening date verification
    opening_date_leaf = evaluator.add_leaf(
        id="Opening_Date_With_URL",
        desc="Provides the hotel's opening date and a reference URL that verifies it.",
        parent=details_node,
        critical=True
    )
    claim_opening = f"The opening date of '{hotel.hotel_name}' is {hotel.opening_date}."
    await evaluator.verify(
        claim=claim_opening,
        node=opening_date_leaf,
        sources=hotel.opening_date_urls,
        additional_instruction="Verify the hotel's opening date. Minor formatting differences (e.g., 'July 2025' vs 'July 15, 2025') can be acceptable if clearly equivalent."
    )


async def verify_facility_2(
    evaluator: Evaluator,
    parent_node,
    vc: NPSHighestVisitorCenter,
) -> None:
    """
    Facility 2: Highest-elevation visitor center in the NPS system.
    Includes elevation (ft), mountain pass, national park, access road, and seasonal operation info—each verified by a URL.
    """
    f_node = evaluator.add_sequential(
        id="Facility_2_Highest_Elevation_NPS_Visitor_Center",
        desc="Highest-elevation visitor center in the National Park Service; include elevation (ft), mountain pass, national park, access road, and seasonal operation info—each verified by a reference URL.",
        parent=parent_node,
        critical=False
    )

    # Identify and verify record status (critical)
    identify_node = evaluator.add_parallel(
        id="Identify_Visitor_Center_Record",
        desc="Correctly identifies the visitor center and verifies that it holds the highest-elevation record in NPS.",
        parent=f_node,
        critical=True
    )

    evaluator.add_custom_node(
        result=(vc is not None and vc.visitor_center_name is not None and vc.visitor_center_name.strip() != ""),
        id="Visitor_Center_Name_Provided",
        desc="Provides the visitor center name (unambiguously identifiable).",
        parent=identify_node,
        critical=True
    )

    evaluator.add_custom_node(
        result=(vc is not None and len(vc.highest_record_urls) > 0),
        id="Highest_Elevation_Record_URLs_Provided",
        desc="Provides at least one reference URL verifying the highest-elevation visitor center record status.",
        parent=identify_node,
        critical=True
    )

    record_leaf = evaluator.add_leaf(
        id="Highest_Elevation_Status_Verified_With_URL",
        desc="Provides a reference URL that verifies the visitor center is the highest-elevation visitor center in the NPS system.",
        parent=identify_node,
        critical=True
    )
    claim_record = f"'{vc.visitor_center_name}' is the highest-elevation visitor center in the National Park Service system."
    await evaluator.verify(
        claim=claim_record,
        node=record_leaf,
        sources=vc.highest_record_urls,
        additional_instruction=(
            "Confirm the URL explicitly states this visitor center holds the highest elevation among ALL NPS visitor centers. "
            "Mentions limited to a single park or area are insufficient."
        ),
    )

    # Required details (critical)
    details_node = evaluator.add_parallel(
        id="Visitor_Center_Required_Details",
        desc="Provides required visitor center attributes, each with a verifying reference URL.",
        parent=f_node,
        critical=True
    )

    # Elevation
    evaluator.add_custom_node(
        result=(vc is not None and vc.elevation_feet is not None and vc.elevation_feet.strip() != "" and len(vc.elevation_urls) > 0),
        id="Elevation_URLs_Provided",
        desc="Provides elevation in feet and includes at least one reference URL for it.",
        parent=details_node,
        critical=True
    )
    elevation_leaf = evaluator.add_leaf(
        id="Elevation_Feet_With_URL",
        desc="Provides exact elevation in feet and a reference URL verifying it.",
        parent=details_node,
        critical=True
    )
    claim_elev = f"The elevation of '{vc.visitor_center_name}' is {vc.elevation_feet}."
    await evaluator.verify(
        claim=claim_elev,
        node=elevation_leaf,
        sources=vc.elevation_urls,
        additional_instruction="Verify the elevation (in feet). Small rounding differences are acceptable if clearly equivalent."
    )

    # Mountain pass
    evaluator.add_custom_node(
        result=(vc is not None and vc.mountain_pass is not None and vc.mountain_pass.strip() != "" and len(vc.mountain_pass_urls) > 0),
        id="Mountain_Pass_URLs_Provided",
        desc="Provides the specific mountain pass and includes at least one reference URL for it.",
        parent=details_node,
        critical=True
    )
    pass_leaf = evaluator.add_leaf(
        id="Mountain_Pass_With_URL",
        desc="Provides the specific mountain pass location and a reference URL verifying it.",
        parent=details_node,
        critical=True
    )
    claim_pass = f"'{vc.visitor_center_name}' is located at {vc.mountain_pass}."
    await evaluator.verify(
        claim=claim_pass,
        node=pass_leaf,
        sources=vc.mountain_pass_urls,
        additional_instruction="Verify the specific mountain pass name associated with the visitor center."
    )

    # National park
    evaluator.add_custom_node(
        result=(vc is not None and vc.national_park is not None and vc.national_park.strip() != "" and len(vc.national_park_urls) > 0),
        id="National_Park_URLs_Provided",
        desc="Provides the national park name and includes at least one reference URL for it.",
        parent=details_node,
        critical=True
    )
    park_leaf = evaluator.add_leaf(
        id="National_Park_With_URL",
        desc="Provides the national park it belongs to and a reference URL verifying it.",
        parent=details_node,
        critical=True
    )
    claim_park = f"'{vc.visitor_center_name}' belongs to {vc.national_park} National Park."
    await evaluator.verify(
        claim=claim_park,
        node=park_leaf,
        sources=vc.national_park_urls,
        additional_instruction="Verify the national park that this visitor center is part of."
    )

    # Access road
    evaluator.add_custom_node(
        result=(vc is not None and vc.access_road is not None and vc.access_road.strip() != "" and len(vc.access_road_urls) > 0),
        id="Access_Road_URLs_Provided",
        desc="Provides the road used to access it and includes at least one reference URL for it.",
        parent=details_node,
        critical=True
    )
    road_leaf = evaluator.add_leaf(
        id="Access_Road_With_URL",
        desc="Provides the road used to access it and a reference URL verifying it.",
        parent=details_node,
        critical=True
    )
    claim_road = f"The visitor center '{vc.visitor_center_name}' is accessed via {vc.access_road}."
    await evaluator.verify(
        claim=claim_road,
        node=road_leaf,
        sources=vc.access_road_urls,
        additional_instruction="Verify the specific road (e.g., highway or named road) that provides access to the visitor center."
    )

    # Seasonal operation
    evaluator.add_custom_node(
        result=(vc is not None and vc.seasonal_operation is not None and vc.seasonal_operation.strip() != "" and len(vc.seasonal_urls) > 0),
        id="Seasonal_Operation_URLs_Provided",
        desc="Provides seasonal operation information and includes at least one reference URL for it.",
        parent=details_node,
        critical=True
    )
    seasonal_leaf = evaluator.add_leaf(
        id="Seasonal_Operation_With_URL",
        desc="Provides seasonal operation information (e.g., typical open/close season or months) and a reference URL verifying it.",
        parent=details_node,
        critical=True
    )
    claim_seasonal = (
        f"Seasonal operation for '{vc.visitor_center_name}': {vc.seasonal_operation}."
    )
    await evaluator.verify(
        claim=claim_seasonal,
        node=seasonal_leaf,
        sources=vc.seasonal_urls,
        additional_instruction=(
            "Verify typical seasonal operations (e.g., summer-only, months open/closed). "
            "Minor phrasing or month range variations are acceptable if they clearly match the claimed seasonality."
        )
    )


async def verify_facility_3(
    evaluator: Evaluator,
    parent_node,
    coaster: SingleRailCoasterRecord,
) -> None:
    """
    Facility 3: World's tallest and longest single-rail roller coaster.
    Includes park, height (ft), track length (ft), top speed (mph), and minimum rider height (in), each verified by a URL.
    """
    f_node = evaluator.add_sequential(
        id="Facility_3_World_Tallest_And_Longest_Single_Rail_Coaster",
        desc="Roller coaster holding world records for both tallest and longest single-rail coaster; include park, height (ft), track length (ft), top speed (mph), and minimum rider height (in), each verified by a reference URL.",
        parent=parent_node,
        critical=False
    )

    # Identify and record verification (critical)
    identify_node = evaluator.add_parallel(
        id="Identify_Coaster_Record",
        desc="Correctly identifies the coaster and verifies it holds both the tallest and longest single-rail coaster records.",
        parent=f_node,
        critical=True
    )

    evaluator.add_custom_node(
        result=(coaster is not None and coaster.coaster_name is not None and coaster.coaster_name.strip() != ""),
        id="Coaster_Name_Provided",
        desc="Provides the coaster name (unambiguously identifiable).",
        parent=identify_node,
        critical=True
    )

    evaluator.add_custom_node(
        result=(coaster is not None and len(coaster.record_urls) > 0),
        id="Tallest_And_Longest_Record_URLs_Provided",
        desc="Provides at least one reference URL verifying BOTH records (tallest and longest) for single-rail coasters.",
        parent=identify_node,
        critical=True
    )

    record_leaf = evaluator.add_leaf(
        id="Tallest_And_Longest_Record_Verified_With_URL",
        desc="Provides a reference URL verifying the coaster holds BOTH records (tallest and longest) for single-rail coasters.",
        parent=identify_node,
        critical=True
    )
    claim_records = (
        f"'{coaster.coaster_name}' holds BOTH world records for tallest AND longest single-rail coaster."
    )
    await evaluator.verify(
        claim=claim_records,
        node=record_leaf,
        sources=coaster.record_urls,
        additional_instruction=(
            "Verify that the page explicitly claims BOTH records—tallest AND longest—for single-rail coasters (e.g., RMC Raptor track). "
            "A claim of only one of the two records is insufficient."
        ),
    )

    # Required coaster details (critical)
    details_node = evaluator.add_parallel(
        id="Coaster_Required_Details",
        desc="Provides required coaster attributes, each with a verifying reference URL.",
        parent=f_node,
        critical=True
    )

    # Theme park
    evaluator.add_custom_node(
        result=(coaster is not None and coaster.theme_park is not None and coaster.theme_park.strip() != "" and len(coaster.theme_park_urls) > 0),
        id="Theme_Park_URLs_Provided",
        desc="Provides the theme park and includes at least one reference URL for it.",
        parent=details_node,
        critical=True
    )
    park_leaf = evaluator.add_leaf(
        id="Theme_Park_With_URL",
        desc="Provides the theme park where the coaster is located and a reference URL verifying it.",
        parent=details_node,
        critical=True
    )
    claim_park = f"'{coaster.coaster_name}' is located at {coaster.theme_park}."
    await evaluator.verify(
        claim=claim_park,
        node=park_leaf,
        sources=coaster.theme_park_urls,
        additional_instruction="Verify the theme park location of the coaster."
    )

    # Height
    evaluator.add_custom_node(
        result=(coaster is not None and coaster.height_feet is not None and coaster.height_feet.strip() != "" and len(coaster.height_urls) > 0),
        id="Height_Feet_URLs_Provided",
        desc="Provides height in feet and includes at least one reference URL for it.",
        parent=details_node,
        critical=True
    )
    height_leaf = evaluator.add_leaf(
        id="Height_Feet_With_URL",
        desc="Provides height in feet and a reference URL verifying it.",
        parent=details_node,
        critical=True
    )
    claim_height = f"The height of '{coaster.coaster_name}' is {coaster.height_feet}."
    await evaluator.verify(
        claim=claim_height,
        node=height_leaf,
        sources=coaster.height_urls,
        additional_instruction="Verify the coaster's height (in feet). Minor rounding/formatting variants are acceptable."
    )

    # Track length
    evaluator.add_custom_node(
        result=(coaster is not None and coaster.track_length_feet is not None and coaster.track_length_feet.strip() != "" and len(coaster.track_length_urls) > 0),
        id="Track_Length_Feet_URLs_Provided",
        desc="Provides track length in feet and includes at least one reference URL for it.",
        parent=details_node,
        critical=True
    )
    length_leaf = evaluator.add_leaf(
        id="Track_Length_Feet_With_URL",
        desc="Provides track length in feet and a reference URL verifying it.",
        parent=details_node,
        critical=True
    )
    claim_length = f"The track length of '{coaster.coaster_name}' is {coaster.track_length_feet}."
    await evaluator.verify(
        claim=claim_length,
        node=length_leaf,
        sources=coaster.track_length_urls,
        additional_instruction="Verify the coaster's track length (in feet). Minor rounding/formatting variants are acceptable."
    )

    # Top speed
    evaluator.add_custom_node(
        result=(coaster is not None and coaster.top_speed_mph is not None and coaster.top_speed_mph.strip() != "" and len(coaster.top_speed_urls) > 0),
        id="Top_Speed_Mph_URLs_Provided",
        desc="Provides top speed in mph and includes at least one reference URL for it.",
        parent=details_node,
        critical=True
    )
    speed_leaf = evaluator.add_leaf(
        id="Top_Speed_Mph_With_URL",
        desc="Provides top speed in mph and a reference URL verifying it.",
        parent=details_node,
        critical=True
    )
    claim_speed = f"The top speed of '{coaster.coaster_name}' is {coaster.top_speed_mph}."
    await evaluator.verify(
        claim=claim_speed,
        node=speed_leaf,
        sources=coaster.top_speed_urls,
        additional_instruction="Verify the coaster's top speed (in mph). Minor rounding/formatting variants are acceptable."
    )

    # Minimum rider height
    evaluator.add_custom_node(
        result=(coaster is not None and coaster.min_rider_height_inches is not None and coaster.min_rider_height_inches.strip() != "" and len(coaster.min_height_urls) > 0),
        id="Minimum_Rider_Height_Inches_URLs_Provided",
        desc="Provides minimum rider height (in inches) and includes at least one reference URL for it.",
        parent=details_node,
        critical=True
    )
    min_height_leaf = evaluator.add_leaf(
        id="Minimum_Rider_Height_Inches_With_URL",
        desc="Provides minimum height requirement in inches and a reference URL verifying it.",
        parent=details_node,
        critical=True
    )
    claim_min_height = f"The minimum rider height requirement for '{coaster.coaster_name}' is {coaster.min_rider_height_inches}."
    await evaluator.verify(
        claim=claim_min_height,
        node=min_height_leaf,
        sources=coaster.min_height_urls,
        additional_instruction="Verify the rider height requirement in inches."
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
    Evaluate an answer for the Recreational Facilities 2025 Factbook task.
    Builds a hierarchical verification tree and returns a structured summary.
    """
    # Initialize evaluator - root is non-critical by framework design
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Facilities can be evaluated independently
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

    # Create top-level node (non-critical to allow partial credit across facilities)
    main_node = evaluator.add_parallel(
        id="Recreational_Facilities_Research",
        desc="Provide verified information (with supporting reference URLs) for three U.S. recreational facilities matching the criteria in the question.",
        parent=root,
        critical=False
    )

    # Extract structured information from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_facilities(),
        template_class=FacilitiesExtraction,
        extraction_name="facilities_extraction"
    )

    # Verify Facility 1
    await verify_facility_1(
        evaluator=evaluator,
        parent_node=main_node,
        hotel=extracted.hotel or HotelEpicEntrance()
    )

    # Verify Facility 2
    await verify_facility_2(
        evaluator=evaluator,
        parent_node=main_node,
        vc=extracted.visitor_center or NPSHighestVisitorCenter()
    )

    # Verify Facility 3
    await verify_facility_3(
        evaluator=evaluator,
        parent_node=main_node,
        coaster=extracted.coaster or SingleRailCoasterRecord()
    )

    # Return structured result
    return evaluator.get_summary()