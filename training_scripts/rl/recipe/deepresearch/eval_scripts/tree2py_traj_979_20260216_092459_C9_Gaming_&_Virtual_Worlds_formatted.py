import asyncio
import logging
from typing import Any, List, Optional, Dict

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "il_gaming_venues_2025"
TASK_DESCRIPTION = """
You are planning a comprehensive gaming and esports event series across Illinois in 2025 and need to identify four distinct venues, each suitable for different event types and scales. For each venue, you must provide complete details including name, location, capacity specifications, and scheduled events.

Identify the following four venues:

Venue 1 - Large Convention Center:
Find a convention center in the Chicago metropolitan area that:
- Has at least 800,000 square feet of flexible exhibition space
- Can accommodate at least 3,000 exhibition booths
- Has ceiling heights of at least 16 feet
- Provides fiber optic Internet access
- Is hosting or available for a major gaming convention in August 2025

Venue 2 - Fighting Game Festival Venue:
Find a convention center in the Chicago metropolitan area that:
- Has a maximum venue capacity of at least 8,000 attendees
- Has at least 140,000 square feet of meeting space
- Has a largest single room of at least 90,000 square feet
- Can accommodate at least 6,000 people in theater-style seating
- Is hosting a fighting game festival in late May 2025 with at least 20 official tournaments
- The event must have a stated registrant cap

Venue 3 - Regional Gaming Convention Facility:
Find a venue in Illinois (outside the Chicago metropolitan area) that:
- Has at least two separate enclosed buildings
- Has combined usable space of at least 40,000 square feet in its enclosed buildings
- Is hosting a gaming convention in June 2025 with at least 100 vendor booths

Venue 4 - Esports Gaming Center:
Find an esports gaming center in the Chicago suburbs (not Chicago city proper) that:
- Has at least 50 gaming stations
- Is at least 5,000 square feet in size
- Features custom gaming PCs or high-performance gaming computers
- Is hosting a tournament or LAN event in March 2025

For each venue, provide:
- Venue name
- Complete address and city
- All relevant capacity/space specifications that satisfy the requirements
- Event name and exact dates (where applicable)
- Reference URL(s) supporting your answer
"""


# --------------------------------------------------------------------------- #
# Extraction models                                                           #
# --------------------------------------------------------------------------- #
class Venue1Extraction(BaseModel):
    venue_name: Optional[str] = None
    address: Optional[str] = None
    city: Optional[str] = None
    location_urls: List[str] = Field(default_factory=list)

    exhibition_space_sqft: Optional[str] = None
    booth_capacity: Optional[str] = None
    ceiling_height_ft: Optional[str] = None
    fiber_optic_internet: Optional[str] = None
    physical_urls: List[str] = Field(default_factory=list)

    event_name: Optional[str] = None
    event_dates: Optional[str] = None
    event_urls: List[str] = Field(default_factory=list)


class Venue2Extraction(BaseModel):
    venue_name: Optional[str] = None
    address: Optional[str] = None
    city: Optional[str] = None
    location_urls: List[str] = Field(default_factory=list)

    overall_capacity: Optional[str] = None
    meeting_space_sqft: Optional[str] = None
    largest_room_sqft: Optional[str] = None
    theater_seating_capacity: Optional[str] = None
    physical_urls: List[str] = Field(default_factory=list)

    event_name: Optional[str] = None
    event_dates: Optional[str] = None
    tournament_count: Optional[str] = None
    registrant_cap: Optional[str] = None
    event_urls: List[str] = Field(default_factory=list)


class Venue3Extraction(BaseModel):
    venue_name: Optional[str] = None
    city: Optional[str] = None
    location_urls: List[str] = Field(default_factory=list)

    building_1_size_sqft: Optional[str] = None
    building_2_size_sqft: Optional[str] = None
    combined_space_sqft: Optional[str] = None
    physical_urls: List[str] = Field(default_factory=list)

    event_name: Optional[str] = None
    event_dates: Optional[str] = None
    vendor_booth_count: Optional[str] = None
    event_urls: List[str] = Field(default_factory=list)


class Venue4Extraction(BaseModel):
    facility_name: Optional[str] = None
    address: Optional[str] = None
    city: Optional[str] = None
    location_urls: List[str] = Field(default_factory=list)

    gaming_station_count: Optional[str] = None
    floor_area_sqft: Optional[str] = None
    pc_equipment_desc: Optional[str] = None
    physical_urls: List[str] = Field(default_factory=list)

    event_name: Optional[str] = None
    event_dates: Optional[str] = None
    event_urls: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_venue1() -> str:
    return """
    Extract details for Venue 1 (Large Convention Center) from the answer.

    Required fields:
    - venue_name: The official name of the venue
    - address: Complete street address
    - city: City name
    - location_urls: All URLs that support the venue location (can include venue page or city/metro references)

    Physical specifications (return numbers as strings; if ranges or approximate values, extract the text as-is):
    - exhibition_space_sqft: The stated flexible exhibition space square footage
    - booth_capacity: The maximum number of exhibition booths the venue can accommodate
    - ceiling_height_ft: The ceiling height specification (minimum or range)
    - fiber_optic_internet: Text indicating fiber optic Internet access availability (e.g., "fiber optic available")
    - physical_urls: URLs that support the above physical specs

    Event details:
    - event_name: Name of the major gaming convention (if stated)
    - event_dates: Exact dates in August 2025 (if stated; otherwise the textual date description)
    - event_urls: URLs that support the event details

    If a field is missing, set it to null. Always include any URLs cited in the answer related to each section.
    """


def prompt_extract_venue2() -> str:
    return """
    Extract details for Venue 2 (Fighting Game Festival Venue) from the answer.

    Required fields:
    - venue_name
    - address
    - city
    - location_urls: URLs supporting the venue location in the Chicago metropolitan area

    Physical/capacity specs (return as strings):
    - overall_capacity: Maximum attendee capacity (e.g., "8,000")
    - meeting_space_sqft: Total meeting space square footage
    - largest_room_sqft: Largest single room square footage
    - theater_seating_capacity: Theater-style seating capacity
    - physical_urls: URLs supporting the above specs

    Event details (late May 2025 fighting game festival):
    - event_name
    - event_dates: exact date(s) in May 2025
    - tournament_count: number of official tournaments (e.g., "20", "22")
    - registrant_cap: stated registrant cap text or value (e.g., "cap 3000", "limited to 2500")
    - event_urls: URLs supporting the event details

    If a field is missing, set it to null. Extract all URLs referenced in the answer.
    """


def prompt_extract_venue3() -> str:
    return """
    Extract details for Venue 3 (Regional Gaming Convention Facility outside Chicago metro) from the answer.

    Required fields:
    - venue_name
    - city
    - location_urls: URLs supporting that the venue is in Illinois and outside Chicago metropolitan area

    Building/space specs (return as strings):
    - building_1_size_sqft: square footage of the first enclosed building
    - building_2_size_sqft: square footage of the second enclosed building
    - combined_space_sqft: combined usable enclosed space square footage (if stated)
    - physical_urls: URLs supporting the building configuration and sizes

    Event details (June 2025):
    - event_name
    - event_dates: exact dates in June 2025
    - vendor_booth_count: number of vendor booths (e.g., "100", "120")
    - event_urls: URLs supporting the event details

    If a field is missing, set it to null. Extract all URLs cited in the answer.
    """


def prompt_extract_venue4() -> str:
    return """
    Extract details for Venue 4 (Esports Gaming Center in Chicago suburbs) from the answer.

    Required fields:
    - facility_name
    - address
    - city
    - location_urls: URLs supporting that the facility is in Chicago suburbs (not Chicago city proper)

    Physical/equipment specs (return as strings):
    - gaming_station_count: number of gaming stations
    - floor_area_sqft: facility size square footage
    - pc_equipment_desc: text describing custom/high-performance gaming PCs
    - physical_urls: URLs supporting the above specs

    Event details (March 2025 tournament or LAN):
    - event_name
    - event_dates: exact dates in March 2025
    - event_urls: URLs supporting the event details

    If a field is missing, set it to null. Extract all URLs cited in the answer.
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _combine_sources(*lists: List[str]) -> List[str]:
    combined: List[str] = []
    for lst in lists:
        if lst:
            for u in lst:
                if u and u not in combined:
                    combined.append(u)
    return combined


# --------------------------------------------------------------------------- #
# Verification builders                                                       #
# --------------------------------------------------------------------------- #
async def build_venue1_tree(evaluator: Evaluator, root, v1: Venue1Extraction) -> None:
    venue_node = evaluator.add_parallel(
        id="Venue_1_Large_Convention",
        desc="Identify a large convention center venue capable of hosting a major gaming convention in August 2025",
        parent=root,
        critical=False
    )

    # Physical Requirements
    phys_node = evaluator.add_parallel(
        id="V1_Physical_Requirements",
        desc="Venue must meet all physical space and infrastructure requirements",
        parent=venue_node,
        critical=True
    )

    # Exhibition Space
    ex_node = evaluator.add_sequential(
        id="V1_Exhibition_Space",
        desc="Venue must have at least 800,000 square feet of flexible exhibition space",
        parent=phys_node,
        critical=True
    )
    leaf = evaluator.add_leaf(
        id="V1_Space_Threshold_Met",
        desc="Verify that the venue's exhibition space meets or exceeds 800,000 square feet",
        parent=ex_node,
        critical=True
    )
    await evaluator.verify(
        claim="The venue has at least 800,000 square feet of flexible exhibition space.",
        node=leaf,
        sources=v1.physical_urls,
        additional_instruction="Confirm text stating total flexible exhibition space. Allow phrasing like 'over 800,000 sq ft' or 'approximately 800,000 sq ft'."
    )
    leaf = evaluator.add_leaf(
        id="V1_Space_Documentation",
        desc="Document the exact square footage with URL reference",
        parent=ex_node,
        critical=True
    )
    exact_space = v1.exhibition_space_sqft or ""
    await evaluator.verify(
        claim=f"The venue's exhibition space is stated as {exact_space} square feet (or equivalent wording).",
        node=leaf,
        sources=v1.physical_urls,
        additional_instruction="Verify that the provided pages explicitly mention the quantitative exhibition space figure."
    )

    # Booth Capacity
    booth_node = evaluator.add_sequential(
        id="V1_Booth_Capacity",
        desc="Venue must accommodate at least 3,000 exhibition booths",
        parent=phys_node,
        critical=True
    )
    leaf = evaluator.add_leaf(
        id="V1_Booth_Threshold_Met",
        desc="Verify that the venue can accommodate at least 3,000 booths",
        parent=booth_node,
        critical=True
    )
    await evaluator.verify(
        claim="The venue can accommodate at least 3,000 exhibition booths.",
        node=leaf,
        sources=v1.physical_urls,
        additional_instruction="Check capabilities/performance specs or floor plans indicating booth capacity."
    )
    leaf = evaluator.add_leaf(
        id="V1_Booth_Documentation",
        desc="Document the maximum booth capacity with URL reference",
        parent=booth_node,
        critical=True
    )
    booth_cap = v1.booth_capacity or ""
    await evaluator.verify(
        claim=f"The venue's maximum booth capacity is stated as {booth_cap}.",
        node=leaf,
        sources=v1.physical_urls,
        additional_instruction="Verify that the page mentions an explicit booth capacity number."
    )

    # Ceiling Height
    ceil_node = evaluator.add_sequential(
        id="V1_Ceiling_Height",
        desc="Venue must have ceiling heights of at least 16 feet",
        parent=phys_node,
        critical=True
    )
    leaf = evaluator.add_leaf(
        id="V1_Height_Threshold_Met",
        desc="Verify that ceiling heights meet or exceed 16 feet",
        parent=ceil_node,
        critical=True
    )
    await evaluator.verify(
        claim="The venue has ceiling heights of at least 16 feet.",
        node=leaf,
        sources=v1.physical_urls,
        additional_instruction="Check specification pages or floor plans for ceiling height values."
    )
    leaf = evaluator.add_leaf(
        id="V1_Height_Documentation",
        desc="Document the ceiling height specifications with URL reference",
        parent=ceil_node,
        critical=True
    )
    height_ft = v1.ceiling_height_ft or ""
    await evaluator.verify(
        claim=f"The documented ceiling height specification includes {height_ft} feet (or an equivalent value ≥ 16 ft).",
        node=leaf,
        sources=v1.physical_urls,
        additional_instruction="Verify that the cited page mentions ceiling height. Minor wording variations acceptable."
    )

    # Technical Infrastructure
    tech_node = evaluator.add_sequential(
        id="V1_Technical_Infrastructure",
        desc="Venue must provide fiber optic Internet access",
        parent=phys_node,
        critical=True
    )
    leaf = evaluator.add_leaf(
        id="V1_Internet_Available",
        desc="Verify that fiber optic Internet access is available",
        parent=tech_node,
        critical=True
    )
    await evaluator.verify(
        claim="Fiber optic Internet access is available at the venue for events.",
        node=leaf,
        sources=v1.physical_urls,
        additional_instruction="Accept explicit mentions of 'fiber optic' or equivalent high-bandwidth fiber internet services."
    )
    leaf = evaluator.add_leaf(
        id="V1_Internet_Documentation",
        desc="Document the Internet specifications with URL reference",
        parent=tech_node,
        critical=True
    )
    fiber_text = v1.fiber_optic_internet or ""
    await evaluator.verify(
        claim=f"The venue's documentation explicitly mentions fiber optic Internet (e.g., '{fiber_text}').",
        node=leaf,
        sources=v1.physical_urls,
        additional_instruction="Verify text snippets that demonstrate fiber optic availability."
    )

    # Location Requirements
    loc_node = evaluator.add_parallel(
        id="V1_Location_Requirements",
        desc="Venue must be located in the Chicago metropolitan area",
        parent=venue_node,
        critical=True
    )
    leaf = evaluator.add_leaf(
        id="V1_Location_Verification",
        desc="Verify that the venue is within the Chicago metropolitan area",
        parent=loc_node,
        critical=True
    )
    await evaluator.verify(
        claim="This venue is located within the Chicago metropolitan area.",
        node=leaf,
        sources=_combine_sources(v1.location_urls, v1.physical_urls),
        additional_instruction="Use location pages or credible references. Accept suburban municipalities widely recognized as part of Chicago metro."
    )
    id_node = evaluator.add_parallel(
        id="V1_Venue_Identification",
        desc="Provide complete venue identification information",
        parent=loc_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=bool(v1.venue_name and v1.venue_name.strip()),
        id="V1_Venue_Name",
        desc="Provide the specific name of the venue",
        parent=id_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=bool(v1.address and v1.address.strip()),
        id="V1_Street_Address",
        desc="Provide the complete street address",
        parent=id_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=bool(v1.city and v1.city.strip()),
        id="V1_City_Name",
        desc="Provide the city where the venue is located",
        parent=id_node,
        critical=True
    )

    # Event Requirements
    evt_node = evaluator.add_sequential(
        id="V1_Event_Requirements",
        desc="Venue must be hosting or available for a gaming convention in August 2025",
        parent=venue_node,
        critical=True
    )
    leaf = evaluator.add_leaf(
        id="V1_August_Availability",
        desc="Verify that the venue is available/hosting an event in August 2025",
        parent=evt_node,
        critical=True
    )
    await evaluator.verify(
        claim="The venue is hosting or available for a gaming convention in August 2025.",
        node=leaf,
        sources=v1.event_urls,
        additional_instruction="Confirm event listings, calendars, or announcements explicitly mentioning August 2025 for a gaming convention."
    )
    leaf = evaluator.add_leaf(
        id="V1_Event_Documentation",
        desc="Document the event details with URL reference",
        parent=evt_node,
        critical=True
    )
    evt_name = v1.event_name or ""
    evt_dates = v1.event_dates or ""
    await evaluator.verify(
        claim=f"The gaming convention '{evt_name}' has stated dates '{evt_dates}' in August 2025.",
        node=leaf,
        sources=v1.event_urls,
        additional_instruction="Check the event page or venue calendar for the event name and exact August 2025 dates."
    )


async def build_venue2_tree(evaluator: Evaluator, root, v2: Venue2Extraction) -> None:
    venue_node = evaluator.add_parallel(
        id="Venue_2_Fighting_Game_Festival",
        desc="Identify a convention center venue hosting a major fighting game festival in May 2025",
        parent=root,
        critical=False
    )

    # Physical Requirements
    phys_node = evaluator.add_parallel(
        id="V2_Physical_Requirements",
        desc="Venue must meet all capacity and space requirements",
        parent=venue_node,
        critical=True
    )

    # Overall Capacity
    cap_node = evaluator.add_sequential(
        id="V2_Overall_Capacity",
        desc="Venue must have maximum capacity of at least 8,000 attendees",
        parent=phys_node,
        critical=True
    )
    leaf = evaluator.add_leaf(
        id="V2_Capacity_Threshold_Met",
        desc="Verify that maximum venue capacity is at least 8,000",
        parent=cap_node,
        critical=True
    )
    await evaluator.verify(
        claim="The venue's maximum capacity is at least 8,000 attendees.",
        node=leaf,
        sources=v2.physical_urls,
        additional_instruction="Confirm with venue specs or fire code capacities listed on official sources."
    )
    leaf = evaluator.add_leaf(
        id="V2_Capacity_Documentation",
        desc="Document the maximum capacity with URL reference",
        parent=cap_node,
        critical=True
    )
    cap_text = v2.overall_capacity or ""
    await evaluator.verify(
        claim=f"The documented maximum capacity is {cap_text} attendees.",
        node=leaf,
        sources=v2.physical_urls,
        additional_instruction="Verify the explicit capacity number on the cited page."
    )

    # Meeting Space
    meet_node = evaluator.add_sequential(
        id="V2_Meeting_Space",
        desc="Venue must have at least 140,000 square feet of meeting space",
        parent=phys_node,
        critical=True
    )
    leaf = evaluator.add_leaf(
        id="V2_Meeting_Space_Threshold_Met",
        desc="Verify that total meeting space is at least 140,000 square feet",
        parent=meet_node,
        critical=True
    )
    await evaluator.verify(
        claim="The venue provides at least 140,000 square feet of meeting space.",
        node=leaf,
        sources=v2.physical_urls,
        additional_instruction="Confirm the total meeting space figure on venue factsheets or official pages."
    )
    leaf = evaluator.add_leaf(
        id="V2_Meeting_Space_Documentation",
        desc="Document the total meeting space with URL reference",
        parent=meet_node,
        critical=True
    )
    meet_text = v2.meeting_space_sqft or ""
    await evaluator.verify(
        claim=f"The documented total meeting space is {meet_text} square feet.",
        node=leaf,
        sources=v2.physical_urls,
        additional_instruction="Verify the explicit number stated."
    )

    # Largest Room
    large_node = evaluator.add_sequential(
        id="V2_Largest_Room",
        desc="Venue's largest single room must be at least 90,000 square feet",
        parent=phys_node,
        critical=True
    )
    leaf = evaluator.add_leaf(
        id="V2_Largest_Room_Threshold_Met",
        desc="Verify that the largest room is at least 90,000 square feet",
        parent=large_node,
        critical=True
    )
    await evaluator.verify(
        claim="The venue's largest single room is at least 90,000 square feet.",
        node=leaf,
        sources=v2.physical_urls,
        additional_instruction="Check hall specs or largest space descriptions."
    )
    leaf = evaluator.add_leaf(
        id="V2_Largest_Room_Documentation",
        desc="Document the largest room size with URL reference",
        parent=large_node,
        critical=True
    )
    largest_text = v2.largest_room_sqft or ""
    await evaluator.verify(
        claim=f"The documented largest single room size is {largest_text} square feet.",
        node=leaf,
        sources=v2.physical_urls,
        additional_instruction="Verify the explicit figure."
    )

    # Theater Seating
    theater_node = evaluator.add_sequential(
        id="V2_Theater_Seating",
        desc="Venue must accommodate at least 6,000 people in theater-style seating",
        parent=phys_node,
        critical=True
    )
    leaf = evaluator.add_leaf(
        id="V2_Theater_Threshold_Met",
        desc="Verify that theater-style capacity is at least 6,000",
        parent=theater_node,
        critical=True
    )
    await evaluator.verify(
        claim="The venue can accommodate at least 6,000 people in theater-style seating.",
        node=leaf,
        sources=v2.physical_urls,
        additional_instruction="Confirm theater seating capacities for main halls."
    )
    leaf = evaluator.add_leaf(
        id="V2_Theater_Documentation",
        desc="Document the theater-style capacity with URL reference",
        parent=theater_node,
        critical=True
    )
    theater_text = v2.theater_seating_capacity or ""
    await evaluator.verify(
        claim=f"The documented theater-style seating capacity is {theater_text}.",
        node=leaf,
        sources=v2.physical_urls,
        additional_instruction="Verify the explicit capacity number."
    )

    # Location Requirements
    loc_node = evaluator.add_parallel(
        id="V2_Location_Requirements",
        desc="Venue must be located in the Chicago metropolitan area",
        parent=venue_node,
        critical=True
    )
    leaf = evaluator.add_leaf(
        id="V2_Location_Verification",
        desc="Verify that the venue is within the Chicago metropolitan area",
        parent=loc_node,
        critical=True
    )
    await evaluator.verify(
        claim="This venue is within the Chicago metropolitan area.",
        node=leaf,
        sources=_combine_sources(v2.location_urls, v2.physical_urls),
        additional_instruction="Accept suburban municipalities recognized as part of Chicago metro."
    )
    id_node = evaluator.add_parallel(
        id="V2_Venue_Identification",
        desc="Provide complete venue identification information",
        parent=loc_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=bool(v2.venue_name and v2.venue_name.strip()),
        id="V2_Venue_Name",
        desc="Provide the specific name of the venue",
        parent=id_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=bool(v2.address and v2.address.strip()),
        id="V2_Address_City",
        desc="Provide the complete address and city",
        parent=id_node,
        critical=True
    )

    # Event Requirements
    evt_node = evaluator.add_parallel(
        id="V2_Event_Requirements",
        desc="Venue must be hosting a fighting game festival in late May 2025 with specific characteristics",
        parent=venue_node,
        critical=True
    )
    may_node = evaluator.add_sequential(
        id="V2_May_Event",
        desc="Event must be scheduled in late May 2025",
        parent=evt_node,
        critical=True
    )
    leaf = evaluator.add_leaf(
        id="V2_May_Timing_Verified",
        desc="Verify that the event is in late May 2025",
        parent=may_node,
        critical=True
    )
    await evaluator.verify(
        claim="The fighting game festival is scheduled in late May 2025 (approximately May 20–31).",
        node=leaf,
        sources=v2.event_urls,
        additional_instruction="Confirm the event dates fall within the last third of May 2025."
    )
    leaf = evaluator.add_leaf(
        id="V2_Event_Dates_Documented",
        desc="Document the exact event dates with URL reference",
        parent=may_node,
        critical=True
    )
    evt_dates = v2.event_dates or ""
    await evaluator.verify(
        claim=f"The event dates are explicitly stated as '{evt_dates}'.",
        node=leaf,
        sources=v2.event_urls,
        additional_instruction="Verify exact dates listed on the official event/venue page."
    )

    tourn_node = evaluator.add_sequential(
        id="V2_Tournament_Count",
        desc="Event must feature at least 20 official tournaments",
        parent=evt_node,
        critical=True
    )
    leaf = evaluator.add_leaf(
        id="V2_Tournament_Threshold_Met",
        desc="Verify that at least 20 official tournaments are featured",
        parent=tourn_node,
        critical=True
    )
    await evaluator.verify(
        claim="The event features at least 20 official tournaments.",
        node=leaf,
        sources=v2.event_urls,
        additional_instruction="Confirm tournament listings or counts on the event page."
    )
    leaf = evaluator.add_leaf(
        id="V2_Tournament_Documentation",
        desc="Document the number of tournaments with URL reference",
        parent=tourn_node,
        critical=True
    )
    tcount = v2.tournament_count or ""
    await evaluator.verify(
        claim=f"The documented number of official tournaments is {tcount}.",
        node=leaf,
        sources=v2.event_urls,
        additional_instruction="Verify the explicit tournament count."
    )

    cap_node = evaluator.add_sequential(
        id="V2_Registrant_Cap",
        desc="Event must have a stated registrant cap",
        parent=evt_node,
        critical=True
    )
    leaf = evaluator.add_leaf(
        id="V2_Cap_Exists",
        desc="Verify that a registrant cap is stated",
        parent=cap_node,
        critical=True
    )
    await evaluator.verify(
        claim="The event explicitly states a registrant cap.",
        node=leaf,
        sources=v2.event_urls,
        additional_instruction="Look for phrases such as 'cap', 'limited to', 'maximum registrants', etc."
    )
    leaf = evaluator.add_leaf(
        id="V2_Cap_Documentation",
        desc="Document the registrant cap value with URL reference",
        parent=cap_node,
        critical=True
    )
    cap_val = v2.registrant_cap or ""
    await evaluator.verify(
        claim=f"The registrant cap value is stated as '{cap_val}'.",
        node=leaf,
        sources=v2.event_urls,
        additional_instruction="Verify the explicit cap value or phrasing indicating the cap amount."
    )

    leaf = evaluator.add_leaf(
        id="V2_Event_Name",
        desc="Provide the name of the fighting game festival event",
        parent=evt_node,
        critical=True
    )
    evt_name = v2.event_name or ""
    await evaluator.verify(
        claim=f"The event name is '{evt_name}'.",
        node=leaf,
        sources=v2.event_urls,
        additional_instruction="Verify the event branding/name on the official page."
    )


async def build_venue3_tree(evaluator: Evaluator, root, v3: Venue3Extraction) -> None:
    venue_node = evaluator.add_parallel(
        id="Venue_3_Regional_Gaming_Con",
        desc="Identify a regional facility hosting a gaming convention in June 2025 outside the Chicago area",
        parent=root,
        critical=False
    )

    # Physical Requirements
    phys_node = evaluator.add_parallel(
        id="V3_Physical_Requirements",
        desc="Venue must meet building and space requirements",
        parent=venue_node,
        critical=True
    )

    # Building Configuration
    bcfg_node = evaluator.add_sequential(
        id="V3_Building_Configuration",
        desc="Venue must have at least two separate enclosed buildings",
        parent=phys_node,
        critical=True
    )
    leaf = evaluator.add_leaf(
        id="V3_Two_Buildings_Verified",
        desc="Verify that venue has at least two separate enclosed buildings",
        parent=bcfg_node,
        critical=True
    )
    await evaluator.verify(
        claim="The venue has at least two separate enclosed buildings.",
        node=leaf,
        sources=v3.physical_urls,
        additional_instruction="Confirm site maps, facility descriptions, or specs describing multiple enclosed buildings."
    )
    details_node = evaluator.add_parallel(
        id="V3_Building_Details",
        desc="Document the size of each enclosed building with URL reference",
        parent=bcfg_node,
        critical=True
    )
    leaf1 = evaluator.add_leaf(
        id="V3_Building_1_Size",
        desc="Provide the square footage of the first enclosed building",
        parent=details_node,
        critical=True
    )
    b1 = v3.building_1_size_sqft or ""
    await evaluator.verify(
        claim=f"The first enclosed building is {b1} square feet.",
        node=leaf1,
        sources=v3.physical_urls,
        additional_instruction="Verify the explicit square footage for building 1."
    )
    leaf2 = evaluator.add_leaf(
        id="V3_Building_2_Size",
        desc="Provide the square footage of the second enclosed building",
        parent=details_node,
        critical=True
    )
    b2 = v3.building_2_size_sqft or ""
    await evaluator.verify(
        claim=f"The second enclosed building is {b2} square feet.",
        node=leaf2,
        sources=v3.physical_urls,
        additional_instruction="Verify the explicit square footage for building 2."
    )

    # Combined Space
    comb_node = evaluator.add_sequential(
        id="V3_Combined_Space",
        desc="Combined usable space of enclosed buildings must be at least 40,000 square feet",
        parent=phys_node,
        critical=True
    )
    leaf = evaluator.add_leaf(
        id="V3_Space_Threshold_Met",
        desc="Verify that combined enclosed space is at least 40,000 square feet",
        parent=comb_node,
        critical=True
    )
    await evaluator.verify(
        claim="The combined usable space of the enclosed buildings is at least 40,000 square feet.",
        node=leaf,
        sources=v3.physical_urls,
        additional_instruction="Verify any statements or calculations on official sources indicating combined usable space."
    )
    leaf = evaluator.add_leaf(
        id="V3_Space_Documentation",
        desc="Document the combined space calculation with URL reference",
        parent=comb_node,
        critical=True
    )
    comb_text = v3.combined_space_sqft or ""
    await evaluator.verify(
        claim=f"The combined usable enclosed space is documented as {comb_text} square feet.",
        node=leaf,
        sources=v3.physical_urls,
        additional_instruction="Verify the explicit combined number or calculation shown."
    )

    # Location Requirements
    loc_node = evaluator.add_parallel(
        id="V3_Location_Requirements",
        desc="Venue must be in Illinois but outside Chicago metropolitan area",
        parent=venue_node,
        critical=True
    )
    leaf = evaluator.add_leaf(
        id="V3_Location_Verification",
        desc="Verify that venue is in Illinois but outside Chicago metro area",
        parent=loc_node,
        critical=True
    )
    await evaluator.verify(
        claim="This venue is located in Illinois and outside the Chicago metropolitan area.",
        node=leaf,
        sources=v3.location_urls,
        additional_instruction="Use city/region references to confirm non-Chicago-metro status while being in Illinois."
    )
    id_node = evaluator.add_parallel(
        id="V3_Venue_Identification",
        desc="Provide complete venue identification information",
        parent=loc_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=bool(v3.venue_name and v3.venue_name.strip()),
        id="V3_Venue_Name",
        desc="Provide the specific name of the venue facility",
        parent=id_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=bool(v3.city and v3.city.strip()),
        id="V3_City_Name",
        desc="Provide the city where the venue is located",
        parent=id_node,
        critical=True
    )

    # Event Requirements
    evt_node = evaluator.add_parallel(
        id="V3_Event_Requirements",
        desc="Venue must be hosting a gaming convention in June 2025 with at least 100 vendor booths",
        parent=venue_node,
        critical=True
    )
    june_node = evaluator.add_sequential(
        id="V3_June_Convention",
        desc="Gaming convention must be scheduled in June 2025",
        parent=evt_node,
        critical=True
    )
    leaf = evaluator.add_leaf(
        id="V3_June_Timing_Verified",
        desc="Verify that the gaming convention is in June 2025",
        parent=june_node,
        critical=True
    )
    await evaluator.verify(
        claim="The gaming convention is scheduled in June 2025.",
        node=leaf,
        sources=v3.event_urls,
        additional_instruction="Confirm event dates are in June 2025."
    )
    edet_node = evaluator.add_parallel(
        id="V3_Event_Details",
        desc="Document the event name and dates with URL reference",
        parent=june_node,
        critical=True
    )
    leaf = evaluator.add_leaf(
        id="V3_Event_Name",
        desc="Provide the name of the gaming convention",
        parent=edet_node,
        critical=True
    )
    v3_evt_name = v3.event_name or ""
    await evaluator.verify(
        claim=f"The convention name is '{v3_evt_name}'.",
        node=leaf,
        sources=v3.event_urls,
        additional_instruction="Verify the event name on the official page."
    )
    leaf = evaluator.add_leaf(
        id="V3_Event_Dates",
        desc="Provide the exact dates in June 2025",
        parent=edet_node,
        critical=True
    )
    v3_evt_dates = v3.event_dates or ""
    await evaluator.verify(
        claim=f"The event dates are '{v3_evt_dates}' in June 2025.",
        node=leaf,
        sources=v3.event_urls,
        additional_instruction="Verify the exact event dates."
    )

    # Vendor Booths
    booth_node = evaluator.add_sequential(
        id="V3_Vendor_Booths",
        desc="Convention must feature at least 100 vendor booths",
        parent=evt_node,
        critical=True
    )
    leaf = evaluator.add_leaf(
        id="V3_Booth_Threshold_Met",
        desc="Verify that at least 100 vendor booths are featured",
        parent=booth_node,
        critical=True
    )
    await evaluator.verify(
        claim="The convention features at least 100 vendor booths.",
        node=leaf,
        sources=v3.event_urls,
        additional_instruction="Confirm exhibitor or vendor booth counts listed."
    )
    leaf = evaluator.add_leaf(
        id="V3_Booth_Documentation",
        desc="Document the vendor booth count with URL reference",
        parent=booth_node,
        critical=True
    )
    vcount = v3.vendor_booth_count or ""
    await evaluator.verify(
        claim=f"The documented vendor booth count is {vcount}.",
        node=leaf,
        sources=v3.event_urls,
        additional_instruction="Verify the explicit count."
    )


async def build_venue4_tree(evaluator: Evaluator, root, v4: Venue4Extraction) -> None:
    venue_node = evaluator.add_parallel(
        id="Venue_4_Esports_Gaming_Center",
        desc="Identify an esports gaming center in the Chicago suburbs hosting a tournament in March 2025",
        parent=root,
        critical=False
    )

    # Physical Requirements
    phys_node = evaluator.add_parallel(
        id="V4_Physical_Requirements",
        desc="Facility must meet size and equipment requirements",
        parent=venue_node,
        critical=True
    )

    # Gaming Stations
    gs_node = evaluator.add_sequential(
        id="V4_Gaming_Stations",
        desc="Facility must have at least 50 gaming stations",
        parent=phys_node,
        critical=True
    )
    leaf = evaluator.add_leaf(
        id="V4_Station_Threshold_Met",
        desc="Verify that facility has at least 50 gaming stations",
        parent=gs_node,
        critical=True
    )
    await evaluator.verify(
        claim="The facility has at least 50 gaming stations.",
        node=leaf,
        sources=v4.physical_urls,
        additional_instruction="Confirm the number of stations on official pages or credible listings."
    )
    leaf = evaluator.add_leaf(
        id="V4_Station_Documentation",
        desc="Document the exact number of gaming stations with URL reference",
        parent=gs_node,
        critical=True
    )
    station_text = v4.gaming_station_count or ""
    await evaluator.verify(
        claim=f"The documented count of gaming stations is {station_text}.",
        node=leaf,
        sources=v4.physical_urls,
        additional_instruction="Verify the explicit number in the cited sources."
    )

    # Facility Size
    size_node = evaluator.add_sequential(
        id="V4_Facility_Size",
        desc="Facility must be at least 5,000 square feet",
        parent=phys_node,
        critical=True
    )
    leaf = evaluator.add_leaf(
        id="V4_Size_Threshold_Met",
        desc="Verify that facility size is at least 5,000 square feet",
        parent=size_node,
        critical=True
    )
    await evaluator.verify(
        claim="The facility is at least 5,000 square feet in size.",
        node=leaf,
        sources=v4.physical_urls,
        additional_instruction="Confirm the facility's square footage."
    )
    leaf = evaluator.add_leaf(
        id="V4_Size_Documentation",
        desc="Document the facility size with URL reference",
        parent=size_node,
        critical=True
    )
    size_text = v4.floor_area_sqft or ""
    await evaluator.verify(
        claim=f"The documented facility size is {size_text} square feet.",
        node=leaf,
        sources=v4.physical_urls,
        additional_instruction="Verify the explicit square footage."
    )

    # PC Equipment
    pc_node = evaluator.add_sequential(
        id="V4_PC_Equipment",
        desc="Facility must feature custom gaming PCs or high-performance gaming computers",
        parent=phys_node,
        critical=True
    )
    leaf = evaluator.add_leaf(
        id="V4_Equipment_Verified",
        desc="Verify that facility features custom/high-performance gaming PCs",
        parent=pc_node,
        critical=True
    )
    await evaluator.verify(
        claim="The facility features custom gaming PCs or high-performance gaming computers.",
        node=leaf,
        sources=v4.physical_urls,
        additional_instruction="Confirm mentions of custom-built PCs, high-end GPUs/CPUs, or equivalent phrasing."
    )
    leaf = evaluator.add_leaf(
        id="V4_Equipment_Documentation",
        desc="Document the equipment specifications with URL reference",
        parent=pc_node,
        critical=True
    )
    pc_text = v4.pc_equipment_desc or ""
    await evaluator.verify(
        claim=f"The equipment description indicates custom/high-performance gaming PCs (e.g., '{pc_text}').",
        node=leaf,
        sources=v4.physical_urls,
        additional_instruction="Verify specs or descriptive text indicating high-performance PCs."
    )

    # Location Requirements
    loc_node = evaluator.add_parallel(
        id="V4_Location_Requirements",
        desc="Facility must be in Chicago suburbs (not Chicago city proper)",
        parent=venue_node,
        critical=True
    )
    leaf = evaluator.add_leaf(
        id="V4_Location_Verification",
        desc="Verify that facility is in Chicago suburbs, not Chicago proper",
        parent=loc_node,
        critical=True
    )
    await evaluator.verify(
        claim="This facility is located in the Chicago suburbs and not within Chicago city proper.",
        node=leaf,
        sources=v4.location_urls,
        additional_instruction="Confirm suburban municipality location distinct from Chicago city limits."
    )
    id_node = evaluator.add_parallel(
        id="V4_Facility_Identification",
        desc="Provide complete facility identification information",
        parent=loc_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=bool(v4.facility_name and v4.facility_name.strip()),
        id="V4_Facility_Name",
        desc="Provide the specific name of the gaming center",
        parent=id_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=bool(v4.address and v4.address.strip()),
        id="V4_Street_Address",
        desc="Provide the complete street address",
        parent=id_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=bool(v4.city and v4.city.strip()),
        id="V4_City_Name",
        desc="Provide the city where the facility is located",
        parent=id_node,
        critical=True
    )

    # Event Requirements
    evt_node = evaluator.add_sequential(
        id="V4_Event_Requirements",
        desc="Facility must be hosting a tournament or LAN event in March 2025",
        parent=venue_node,
        critical=True
    )
    march_node = evaluator.add_sequential(
        id="V4_March_Event",
        desc="Tournament or LAN event must be scheduled in March 2025",
        parent=evt_node,
        critical=True
    )
    leaf = evaluator.add_leaf(
        id="V4_March_Timing_Verified",
        desc="Verify that tournament/LAN event is in March 2025",
        parent=march_node,
        critical=True
    )
    await evaluator.verify(
        claim="The tournament or LAN event is scheduled in March 2025.",
        node=leaf,
        sources=v4.event_urls,
        additional_instruction="Confirm the event dates fall in March 2025."
    )
    edet_node = evaluator.add_parallel(
        id="V4_Event_Details",
        desc="Document the event details with URL reference",
        parent=march_node,
        critical=True
    )
    leaf = evaluator.add_leaf(
        id="V4_Event_Name",
        desc="Provide the name of the tournament or LAN event",
        parent=edet_node,
        critical=True
    )
    v4_evt_name = v4.event_name or ""
    await evaluator.verify(
        claim=f"The event name is '{v4_evt_name}'.",
        node=leaf,
        sources=v4.event_urls,
        additional_instruction="Verify the event name on official sources."
    )
    leaf = evaluator.add_leaf(
        id="V4_Event_Dates",
        desc="Provide the exact dates in March 2025",
        parent=edet_node,
        critical=True
    )
    v4_evt_dates = v4.event_dates or ""
    await evaluator.verify(
        claim=f"The event dates are '{v4_evt_dates}' in March 2025.",
        node=leaf,
        sources=v4.event_urls,
        additional_instruction="Verify the exact event dates."
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
    Evaluate an answer for the Illinois gaming/esports venues 2025 task.
    """
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Root parallel aggregation
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

    # Extract structured info for each venue
    v1, v2, v3, v4 = await asyncio.gather(
        evaluator.extract(
            prompt=prompt_extract_venue1(),
            template_class=Venue1Extraction,
            extraction_name="venue_1_extraction",
        ),
        evaluator.extract(
            prompt=prompt_extract_venue2(),
            template_class=Venue2Extraction,
            extraction_name="venue_2_extraction",
        ),
        evaluator.extract(
            prompt=prompt_extract_venue3(),
            template_class=Venue3Extraction,
            extraction_name="venue_3_extraction",
        ),
        evaluator.extract(
            prompt=prompt_extract_venue4(),
            template_class=Venue4Extraction,
            extraction_name="venue_4_extraction",
        ),
    )

    # Build verification trees for each venue
    await build_venue1_tree(evaluator, root, v1)
    await build_venue2_tree(evaluator, root, v2)
    await build_venue3_tree(evaluator, root, v3)
    await build_venue4_tree(evaluator, root, v4)

    # Return summary
    return evaluator.get_summary()