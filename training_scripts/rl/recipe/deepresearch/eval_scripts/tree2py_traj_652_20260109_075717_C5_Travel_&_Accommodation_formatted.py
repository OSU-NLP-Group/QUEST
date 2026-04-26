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
TASK_ID = "hotel_evv_training"
TASK_DESCRIPTION = """
I am organizing a corporate training event for 25 employees near Evansville Regional Airport (EVV) in Indiana. I need to find a suitable hotel that meets the following requirements:

1. Meeting Space: The hotel must have a conference room that can accommodate all 25 attendees in conference-style seating.

2. Accessibility: The hotel must have ADA-compliant accessible rooms with roll-in showers. Specifically, the roll-in showers must be at least 30 inches by 60 inches, with grab bars on the back wall and the side wall opposite the shower seat, as one of our attendees requires mobility-accessible accommodations.

3. Airport Transportation: The hotel must provide free airport shuttle service to Evansville Regional Airport (EVV) for convenient attendee arrival and departure.

4. Hotel Size: The hotel must have at least 100 guest rooms to ensure adequate capacity and appropriate service levels for corporate groups.

Please recommend a specific hotel that meets all these requirements. Provide the hotel's complete name and full address, and include reference URLs that verify each requirement (meeting space capacity, accessibility features, shuttle service, and total room count).
"""


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class HotelInfo(BaseModel):
    name: Optional[str] = None
    street: Optional[str] = None
    city: Optional[str] = None
    state: Optional[str] = None
    zip: Optional[str] = None
    full_address: Optional[str] = None
    location_urls: List[str] = Field(default_factory=list)


class MeetingSpaceInfo(BaseModel):
    capacity_conference: Optional[str] = None
    square_footage: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class AccessibilityInfo(BaseModel):
    roll_in_shower_dimensions: Optional[str] = None
    grab_bars_description: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class ShuttleInfo(BaseModel):
    shuttle_description: Optional[str] = None  # e.g., "Complimentary EVV airport shuttle"
    sources: List[str] = Field(default_factory=list)


class SizeInfo(BaseModel):
    room_count: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class HotelRecommendationExtraction(BaseModel):
    hotel: Optional[HotelInfo] = None
    meeting_space: Optional[MeetingSpaceInfo] = None
    accessibility: Optional[AccessibilityInfo] = None
    shuttle: Optional[ShuttleInfo] = None
    size: Optional[SizeInfo] = None


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_hotel_recommendation() -> str:
    return """
    Extract the single recommended hotel and all required verification information as presented in the answer.

    Return a JSON object with the following fields:
    - hotel:
        - name: The hotel's full official name.
        - street: Street address line (e.g., "123 Main St").
        - city: City (e.g., "Evansville").
        - state: State (e.g., "IN").
        - zip: ZIP code (e.g., "47711").
        - full_address: The complete address string if provided (street, city, state, ZIP).
        - location_urls: An array of URLs provided in the answer that help verify the hotel's proximity to EVV (e.g., hotel location page, map link, distance-to-airport info). If none are provided, return an empty array.

    - meeting_space:
        - capacity_conference: The stated capacity for conference-style seating if mentioned (e.g., "25", "30", or "Up to 40 conference-style").
        - square_footage: The meeting room square footage if mentioned (e.g., "900 sq ft", "1200 square feet").
        - sources: An array of URLs that support the meeting space capacity or dimensions. If none, return an empty array.

    - accessibility:
        - roll_in_shower_dimensions: The described dimensions for roll-in showers if mentioned (e.g., "30 x 60 inches", "at least 30\" by 60\"").
        - grab_bars_description: Description of grab bar placement if mentioned (e.g., "grab bars on back wall and opposite the seat").
        - sources: An array of URLs verifying ADA accessibility and roll-in shower features. If none, return an empty array.

    - shuttle:
        - shuttle_description: Description indicating complimentary/free airport shuttle to EVV if provided (e.g., "free EVV airport shuttle").
        - sources: An array of URLs verifying the shuttle service to Evansville Regional Airport (EVV). If none, return an empty array.

    - size:
        - room_count: The hotel's total room count as stated (e.g., "120 rooms", "150 guest rooms").
        - sources: An array of URLs verifying the total guest room count. If none, return an empty array.

    IMPORTANT:
    - Extract only what is explicitly in the answer. If any field is missing, set it to null (or empty array for URLs).
    - For URLs, include full valid URLs; do not invent any.
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def _has_valid_urls(urls: List[str]) -> bool:
    if not urls:
        return False
    return any(isinstance(u, str) and u.strip().lower().startswith(("http://", "https://")) for u in urls)


def _combine_sources(*url_lists: Optional[List[str]]) -> List[str]:
    seen = set()
    combined: List[str] = []
    for lst in url_lists:
        if not lst:
            continue
        for u in lst:
            if isinstance(u, str):
                key = u.strip()
                if key and key not in seen:
                    seen.add(key)
                    combined.append(key)
    return combined


# --------------------------------------------------------------------------- #
# Verification subtrees                                                       #
# --------------------------------------------------------------------------- #
async def verify_hotel_identification_and_location(
    evaluator: Evaluator,
    parent_node,
    ext: HotelRecommendationExtraction
) -> None:
    node = evaluator.add_parallel(
        id="Hotel_Identification_and_Location",
        desc="Verify the hotel is clearly identified and located near Evansville Regional Airport (EVV) in Indiana.",
        parent=parent_node,
        critical=True
    )

    hotel = ext.hotel or HotelInfo()

    name_ok = bool(hotel.name and hotel.name.strip())
    address_ok = bool(
        hotel.full_address and hotel.full_address.strip()
        and hotel.street and hotel.city and hotel.state and hotel.zip
    )

    evaluator.add_custom_node(
        result=(name_ok and address_ok),
        id="Hotel_Name_and_Full_Address",
        desc="Provide the hotel's complete name and full address (street, city, state, ZIP).",
        parent=node,
        critical=True
    )

    # Location verification
    near_node = evaluator.add_leaf(
        id="Located_Near_EVV",
        desc="Hotel is located near Evansville Regional Airport (EVV) in Indiana (reasonable proximity for airport access).",
        parent=node,
        critical=True
    )

    claim_parts = []
    if hotel.name:
        claim_parts.append(f"The hotel '{hotel.name}'")
    else:
        claim_parts.append("The hotel")
    if hotel.full_address:
        claim_parts.append(f"located at {hotel.full_address}")
    if hotel.city or hotel.state:
        city_state = ", ".join([p for p in [hotel.city, hotel.state] if p])
        if city_state:
            claim_parts.append(f"in {city_state}")
    claim_parts.append("is reasonably near Evansville Regional Airport (EVV) in Indiana.")
    claim = " ".join(claim_parts)

    # Use any provided location URLs; if none, use shuttle sources as secondary evidence
    location_sources = _combine_sources(hotel.location_urls, (ext.shuttle.sources if ext.shuttle else []))

    await evaluator.verify(
        claim=claim,
        node=near_node,
        sources=location_sources if location_sources else None,
        additional_instruction=(
            "Consider 'near' to mean within roughly 10 miles or otherwise clearly convenient for airport access. "
            "If sources indicate the address is in Evansville, IN, or explicitly state proximity/short distance to EVV, "
            "that should be considered near. Focus only on the claim using provided sources (if any)."
        ),
    )


async def verify_meeting_space(
    evaluator: Evaluator,
    parent_node,
    ext: HotelRecommendationExtraction
) -> None:
    node = evaluator.add_parallel(
        id="Meeting_Space",
        desc="Verify the hotel has meeting space adequate for 25 attendees in conference-style seating.",
        parent=parent_node,
        critical=True
    )

    meeting = ext.meeting_space or MeetingSpaceInfo()

    # Source existence check
    src_exists = evaluator.add_custom_node(
        result=_has_valid_urls(meeting.sources),
        id="Meeting_Space_Source_URL",
        desc="Provide at least one valid reference URL that supports the meeting space/capacity evidence.",
        parent=node,
        critical=True
    )

    adequate_node = evaluator.add_leaf(
        id="Meeting_Space_Adequate_For_25",
        desc="Evidence shows meeting space can accommodate 25 attendees in conference-style seating (or roughly ≥875 sq ft).",
        parent=node,
        critical=True
    )

    capacity_text = meeting.capacity_conference or ""
    sqft_text = meeting.square_footage or ""
    claim = (
        f"The hotel's meeting space can accommodate at least 25 attendees in conference-style seating "
        f"({capacity_text if capacity_text else 'capacity not explicitly stated'}), "
        f"or provides sufficient square footage (e.g., roughly ≥875 sq ft; provided: {sqft_text if sqft_text else 'not explicitly stated'}) "
        f"consistent with ~35–40 sq ft per person."
    )

    await evaluator.verify(
        claim=claim,
        node=adequate_node,
        sources=meeting.sources if meeting.sources else None,
        extra_prerequisites=[src_exists],
        additional_instruction=(
            "Support the claim if the source explicitly states conference-style capacity ≥25 OR the room size is roughly ≥875 sq ft. "
            "If multiple rooms can be combined to reach this capacity and the source indicates divisibility, that is acceptable."
        ),
    )


async def verify_accessibility(
    evaluator: Evaluator,
    parent_node,
    ext: HotelRecommendationExtraction
) -> None:
    node = evaluator.add_parallel(
        id="Accessibility",
        desc="Verify the hotel has ADA-compliant accessible rooms with roll-in showers meeting the specified requirements.",
        parent=parent_node,
        critical=True
    )

    acc = ext.accessibility or AccessibilityInfo()

    # Source existence check
    src_exists = evaluator.add_custom_node(
        result=_has_valid_urls(acc.sources),
        id="Accessibility_Source_URL",
        desc="Provide at least one valid reference URL verifying the accessibility/roll-in shower features.",
        parent=node,
        critical=True
    )

    # Dimensions check
    dims_node = evaluator.add_leaf(
        id="Roll_In_Shower_Dimensions_30x60_Min",
        desc="Accessible rooms include roll-in showers with minimum dimensions of 30 inches by 60 inches.",
        parent=node,
        critical=True
    )
    claim_dims = (
        "The hotel's mobility accessible rooms include roll-in showers with minimum dimensions of at least 30 inches by 60 inches."
    )
    await evaluator.verify(
        claim=claim_dims,
        node=dims_node,
        sources=acc.sources if acc.sources else None,
        extra_prerequisites=[src_exists],
        additional_instruction=(
            "Confirm explicit dimensions such as '30 x 60 inches' or equivalent phrasing indicating minimum dimensions ≥30\" by ≥60\". "
            "If the source lists ADA specs that unambiguously meet or exceed these dimensions, support the claim."
        ),
    )

    # Grab bars placement check
    bars_node = evaluator.add_leaf(
        id="Grab_Bars_Required_Placement",
        desc="Roll-in showers have grab bars on the back wall and on the side wall opposite the shower seat.",
        parent=node,
        critical=True
    )
    claim_bars = (
        "The hotel's roll-in showers have grab bars on the back wall and on the side wall opposite the shower seat."
    )
    await evaluator.verify(
        claim=claim_bars,
        node=bars_node,
        sources=acc.sources if acc.sources else None,
        extra_prerequisites=[src_exists],
        additional_instruction=(
            "Verify explicit mentions that grab bars exist on the back wall and on the side wall opposite the shower seat, "
            "consistent with ADA requirements for roll-in showers."
        ),
    )


async def verify_shuttle(
    evaluator: Evaluator,
    parent_node,
    ext: HotelRecommendationExtraction
) -> None:
    node = evaluator.add_parallel(
        id="Airport_Shuttle",
        desc="Verify the hotel provides free airport shuttle service to EVV and provides a verifying source.",
        parent=parent_node,
        critical=True
    )

    shuttle = ext.shuttle or ShuttleInfo()

    # Source existence check
    src_exists = evaluator.add_custom_node(
        result=_has_valid_urls(shuttle.sources),
        id="Shuttle_Source_URL",
        desc="Provide at least one valid reference URL verifying the complimentary EVV airport shuttle service.",
        parent=node,
        critical=True
    )

    shuttle_node = evaluator.add_leaf(
        id="Complimentary_Airport_Shuttle_To_EVV",
        desc="Hotel provides complimentary airport shuttle service to Evansville Regional Airport (EVV).",
        parent=node,
        critical=True
    )

    claim = (
        "The hotel provides complimentary (free) airport shuttle service to and from Evansville Regional Airport (EVV)."
    )
    await evaluator.verify(
        claim=claim,
        node=shuttle_node,
        sources=shuttle.sources if shuttle.sources else None,
        extra_prerequisites=[src_exists],
        additional_instruction=(
            "Confirm that the shuttle is complimentary/free and explicitly serves Evansville Regional Airport (EVV). "
            "Mentions of paid or third-party shuttle do not qualify."
        ),
    )


async def verify_size(
    evaluator: Evaluator,
    parent_node,
    ext: HotelRecommendationExtraction
) -> None:
    node = evaluator.add_parallel(
        id="Hotel_Size",
        desc="Verify the hotel has at least 100 guest rooms and provide a verifying source.",
        parent=parent_node,
        critical=True
    )

    size = ext.size or SizeInfo()

    # Source existence check
    src_exists = evaluator.add_custom_node(
        result=_has_valid_urls(size.sources),
        id="Room_Count_Source_URL",
        desc="Provide at least one valid reference URL verifying the total guest room count.",
        parent=node,
        critical=True
    )

    rooms_node = evaluator.add_leaf(
        id="At_Least_100_Guest_Rooms",
        desc="Hotel has at least 100 guest rooms.",
        parent=node,
        critical=True
    )

    claim = "The hotel has at least 100 total guest rooms."
    await evaluator.verify(
        claim=claim,
        node=rooms_node,
        sources=size.sources if size.sources else None,
        extra_prerequisites=[src_exists],
        additional_instruction=(
            "Verify the total room count is ≥100. Accept equivalent phrasing such as '100+ rooms', 'over 100 rooms', or 'at least 100 keys'."
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
    model: str = "o4-mini"
) -> Dict:
    """
    Evaluate whether the recommended hotel near EVV meets meeting space, accessibility, shuttle, and size requirements,
    with proper verification by cited sources for each requirement.
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

    # Extract structured info from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_hotel_recommendation(),
        template_class=HotelRecommendationExtraction,
        extraction_name="hotel_recommendation_extraction"
    )

    # Add top-level critical node representing hotel recommendation
    hotel_rec_node = evaluator.add_parallel(
        id="Hotel_Recommendation",
        desc="Evaluate whether the recommended single hotel near Evansville Regional Airport (EVV) meets the meeting space, accessibility, shuttle, and size requirements, and provides sources verifying each requirement.",
        parent=root,
        critical=True
    )

    # Build subtrees
    await verify_hotel_identification_and_location(evaluator, hotel_rec_node, extracted)
    await verify_meeting_space(evaluator, hotel_rec_node, extracted)
    await verify_accessibility(evaluator, hotel_rec_node, extracted)
    await verify_shuttle(evaluator, hotel_rec_node, extracted)
    await verify_size(evaluator, hotel_rec_node, extracted)

    # Optional: record requirement summary
    evaluator.add_custom_info(
        {
            "requirements": [
                "Meeting space for 25 attendees (conference-style or ≥875 sq ft)",
                "ADA rooms with roll-in showers ≥30x60 inches and grab bars (back wall + opposite seat side wall)",
                "Complimentary airport shuttle to EVV",
                "At least 100 guest rooms",
                "Provide URLs verifying each requirement"
            ],
            "airport": "Evansville Regional Airport (EVV), Indiana"
        },
        info_type="requirements_summary"
    )

    return evaluator.get_summary()