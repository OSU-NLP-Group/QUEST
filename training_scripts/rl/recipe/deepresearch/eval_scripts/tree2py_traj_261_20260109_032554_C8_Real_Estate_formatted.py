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
TASK_ID = "boston_coworking_building"
TASK_DESCRIPTION = """
A company is planning to establish a coworking space operation in Boston, Massachusetts, and needs to identify a suitable commercial office property. Find and describe a commercial office building in Boston that meets the following comprehensive requirements:

Location Requirements:
- Must be located within 0.5 miles (half-mile) walking distance of a major MBTA transit station (subway, commuter rail, or light rail station)

Building Classification and Certification:
- Must be classified as a Class A office building
- Must hold at least LEED Gold certification (60-79 points) or higher (LEED Platinum: 80+ points)

Space Requirements:
- Must have at least 10,000 square feet of available contiguous office space
- Office layout should be suitable for standard coworking density (150-200 square feet per person)

ADA Accessibility Compliance:
Entrance Requirements:
- At least 60% of public building entrances must be ADA-accessible
- ADA-accessible entrance doors must provide a 32-inch minimum clear width opening

Parking Requirements:
- Must provide van-accessible parking spaces with 132-inch minimum width
- Accessible parking spaces must have adjacent access aisles of 60-inch minimum width
- Vehicular routes serving van parking must provide 98-inch minimum vertical clearance

Interior Requirements:
- Interior accessible routes must provide 36-inch minimum clear width
- If multi-story, the building must have ADA-compliant passenger elevators
- Building design must allow for at least 5% of work surfaces to meet ADA accessibility standards

Additional Requirements:
- Must provide parking at a ratio of at least 4 parking spaces per 1,000 square feet of office space
- Should have or accommodate high-speed internet infrastructure suitable for coworking
- Should include or accommodate private meeting and conference rooms with audiovisual capabilities
- Should include or accommodate kitchen and coffee preparation areas
- Should provide or accommodate printing, scanning, and mail handling services
- Should have security systems including controlled access and surveillance capabilities

For your answer, provide:
1. The building name and complete address
2. Documentation (with reference URLs) demonstrating how the property meets each of the above requirements
"""


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class BuildingExtraction(BaseModel):
    # Identification
    building_name: Optional[str] = None
    full_address: Optional[str] = None
    property_main_urls: List[str] = Field(default_factory=list)

    # Transit proximity
    transit_station_name: Optional[str] = None
    transit_station_type: Optional[str] = None  # subway / commuter rail / light rail
    walking_distance_miles: Optional[str] = None  # e.g., "0.4 mi"
    walking_distance_minutes: Optional[str] = None  # e.g., "8 min"
    transit_urls: List[str] = Field(default_factory=list)  # MBTA station page or equivalent
    maps_walking_url: Optional[str] = None  # Google Maps or similar walking route URL

    # Classification & certification
    building_class_is_class_a: Optional[str] = None  # "yes"/"no"/"unknown"
    building_class_urls: List[str] = Field(default_factory=list)
    leed_cert_level: Optional[str] = None  # e.g., "Gold", "Platinum", "None"
    leed_urls: List[str] = Field(default_factory=list)

    # Space and density
    contiguous_available_space_sqft: Optional[str] = None  # e.g., "12,000", "15,000 SF"
    space_urls: List[str] = Field(default_factory=list)
    coworking_density_support_desc: Optional[str] = None  # rationale or statement
    density_urls: List[str] = Field(default_factory=list)

    # ADA - Entrance
    ada_entrance_accessible_percentage: Optional[str] = None  # e.g., "60%", ">=60%"
    ada_entrance_door_clear_width_inches: Optional[str] = None  # e.g., "32 in"
    ada_entrance_urls: List[str] = Field(default_factory=list)

    # ADA - Parking
    ada_parking_van_space_width_inches: Optional[str] = None  # e.g., "132 in"
    ada_parking_access_aisle_width_inches: Optional[str] = None  # e.g., "60 in"
    ada_parking_vertical_clearance_inches: Optional[str] = None  # e.g., "98 in"
    ada_parking_urls: List[str] = Field(default_factory=list)

    # ADA - Interior
    ada_interior_route_width_inches: Optional[str] = None  # e.g., "36 in"
    ada_has_ada_compliant_elevators: Optional[str] = None  # "yes"/"no"/"n/a"
    ada_interior_urls: List[str] = Field(default_factory=list)

    # ADA - Work surfaces
    ada_work_surfaces_percent: Optional[str] = None  # e.g., ">=5%"
    ada_work_surface_urls: List[str] = Field(default_factory=list)

    # Parking ratio
    parking_ratio_spaces_per_1000: Optional[str] = None  # e.g., "4/1,000 SF"
    parking_ratio_urls: List[str] = Field(default_factory=list)

    # Operational features
    internet_infrastructure_desc: Optional[str] = None
    internet_urls: List[str] = Field(default_factory=list)

    meeting_rooms_av_desc: Optional[str] = None
    meeting_urls: List[str] = Field(default_factory=list)

    kitchen_coffee_desc: Optional[str] = None
    kitchen_urls: List[str] = Field(default_factory=list)

    printing_scanning_mail_desc: Optional[str] = None
    print_mail_urls: List[str] = Field(default_factory=list)

    security_control_access_surveillance_desc: Optional[str] = None
    security_urls: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_building_info() -> str:
    return """
    Extract the information about ONE commercial office building in Boston, Massachusetts that the answer proposes for a coworking space. Extract ONLY what is explicitly provided in the answer text and its cited sources. If a field is not mentioned, return null or an empty list as appropriate.

    Required fields:

    Identification:
    - building_name: The official or commonly used building name.
    - full_address: Complete street address; must be in Boston, MA (Massachusetts).
    - property_main_urls: An array of any primary property/brochure/listing URLs cited.

    Transit Proximity:
    - transit_station_name: Name of the nearest major MBTA station (subway / commuter rail / light rail).
    - transit_station_type: One of ["subway", "commuter rail", "light rail"], if provided.
    - walking_distance_miles: The walking distance expressed in miles (e.g., "0.4 mi"), if provided.
    - walking_distance_minutes: The walking time (e.g., "8 min"), if provided.
    - transit_urls: Array of station or MBTA-related page URLs cited.
    - maps_walking_url: A Google Maps (or similar) walking route URL from the building to the station, if provided.

    Classification & Certification:
    - building_class_is_class_a: "yes" if the answer claims Class A; otherwise "no" or "unknown".
    - building_class_urls: Array of URLs supporting Class A classification.
    - leed_cert_level: The LEED certification level (e.g., "Gold", "Platinum") if claimed.
    - leed_urls: Array of URLs supporting the LEED certification claim (USGBC directory, building-specific sustainability page, etc.).

    Space & Density:
    - contiguous_available_space_sqft: The contiguous available office space figure (string, as given).
    - space_urls: Array of URLs supporting the available space claim.
    - coworking_density_support_desc: A description or statement indicating layout supports 150–200 sq ft/person, if claimed.
    - density_urls: Array of URLs supporting the coworking density suitability (e.g., floor plans, lease brochure).

    ADA – Entrance:
    - ada_entrance_accessible_percentage: Percentage of public entrances that are ADA-accessible (string), if claimed.
    - ada_entrance_door_clear_width_inches: Clear width of ADA-accessible entrance doors (string inches), if claimed.
    - ada_entrance_urls: Array of URLs supporting entrance ADA claims.

    ADA – Parking:
    - ada_parking_van_space_width_inches: Van-accessible parking space width (string inches), if claimed.
    - ada_parking_access_aisle_width_inches: Access aisle width (string inches), if claimed.
    - ada_parking_vertical_clearance_inches: Vehicular route vertical clearance (string inches), if claimed.
    - ada_parking_urls: Array of URLs supporting parking ADA claims.

    ADA – Interior:
    - ada_interior_route_width_inches: Interior accessible route width (string inches), if claimed.
    - ada_has_ada_compliant_elevators: "yes"/"no"/"n/a" as claimed.
    - ada_interior_urls: Array of URLs supporting interior ADA claims.

    ADA – Work Surfaces:
    - ada_work_surfaces_percent: Percentage of work surfaces that meet ADA accessibility (string), if claimed.
    - ada_work_surface_urls: Array of URLs supporting this claim.

    Parking Ratio:
    - parking_ratio_spaces_per_1000: The parking ratio (string like "4/1,000 SF") if claimed.
    - parking_ratio_urls: Array of URLs supporting parking ratio.

    Operational Features:
    - internet_infrastructure_desc: Statement indicating high-speed internet infrastructure suitability, if claimed.
    - internet_urls: Array of URLs supporting the internet infrastructure claim.
    - meeting_rooms_av_desc: Statement indicating meeting/conference rooms with AV, if claimed.
    - meeting_urls: Array of URLs supporting meeting room/AV claim.
    - kitchen_coffee_desc: Statement indicating kitchen/coffee prep areas, if claimed.
    - kitchen_urls: Array of URLs supporting kitchen/coffee claim.
    - printing_scanning_mail_desc: Statement indicating printing/scanning/mail handling, if claimed.
    - print_mail_urls: Array of URLs supporting the printing/scanning/mail claim.
    - security_control_access_surveillance_desc: Statement indicating controlled access and surveillance systems, if claimed.
    - security_urls: Array of URLs supporting the security claim.

    IMPORTANT:
    - Extract ONLY URLs explicitly present in the answer. If the answer references a site without a direct URL, leave the corresponding URL field empty or null.
    - Prefer full URLs (include http/https). If a URL is missing a protocol, prepend "http://".
    - If any requirement is not documented in the answer, return null/empty values for that requirement.
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _list_nonempty(lst: Optional[List[str]]) -> bool:
    return bool(lst) and len(lst) > 0


def _combine_sources(*args: Optional[List[str]]) -> List[str]:
    out: List[str] = []
    for a in args:
        if a:
            out.extend([u for u in a if isinstance(u, str) and u.strip()])
    return out


def _references_provided_for_all_requirements(ex: BuildingExtraction) -> bool:
    categories_presence = [
        _list_nonempty(ex.transit_urls) or bool(ex.maps_walking_url),
        _list_nonempty(ex.building_class_urls),
        _list_nonempty(ex.leed_urls),
        _list_nonempty(ex.space_urls),
        _list_nonempty(ex.density_urls),
        _list_nonempty(ex.ada_entrance_urls),
        _list_nonempty(ex.ada_parking_urls),
        _list_nonempty(ex.ada_interior_urls),
        _list_nonempty(ex.ada_work_surface_urls),
        _list_nonempty(ex.parking_ratio_urls),
        _list_nonempty(ex.internet_urls),
        _list_nonempty(ex.meeting_urls),
        _list_nonempty(ex.kitchen_urls),
        _list_nonempty(ex.print_mail_urls),
        _list_nonempty(ex.security_urls),
    ]
    return all(categories_presence)


# --------------------------------------------------------------------------- #
# Verification tree builder                                                   #
# --------------------------------------------------------------------------- #
async def build_verification_tree(evaluator: Evaluator, root, ex: BuildingExtraction) -> None:
    # 1) Response Identification and Documentation (critical)
    resp_node = evaluator.add_parallel(
        id="Response_Identification_and_Documentation",
        desc="Response includes required identification details and supporting references",
        parent=root,
        critical=True
    )

    # 1.1 Building Name & Address (critical leaf) – existence check
    exists_building_info = (
        (ex.building_name is not None and ex.building_name.strip() != "") and
        (ex.full_address is not None and ex.full_address.strip() != "") and
        ("Boston" in ex.full_address if ex.full_address else False)
    )
    evaluator.add_custom_node(
        result=exists_building_info,
        id="Building_Name_and_Full_Address",
        desc="Provides the building name and complete address (Boston, Massachusetts)",
        parent=resp_node,
        critical=True
    )

    # 1.2 References Provided (critical leaf) – must provide URLs for each requirement
    evaluator.add_custom_node(
        result=_references_provided_for_all_requirements(ex),
        id="References_Provided",
        desc="Provides reference URL(s) supporting the claims for each requirement (documentation demonstrating compliance)",
        parent=resp_node,
        critical=True
    )

    # 2) Location Requirement (critical)
    loc_node = evaluator.add_parallel(
        id="Location_Requirement",
        desc="Meets location proximity requirement to transit",
        parent=root,
        critical=True
    )
    transit_leaf = evaluator.add_leaf(
        id="Transit_Proximity",
        desc="Property is within 0.5 miles walking distance of a major MBTA transit station (subway, commuter rail, or light rail)",
        parent=loc_node,
        critical=True
    )
    transit_sources: List[str] = []
    if ex.maps_walking_url and ex.maps_walking_url.strip():
        transit_sources.append(ex.maps_walking_url.strip())
    transit_sources = _combine_sources(transit_sources, ex.transit_urls)
    transit_claim = f"Walking distance from '{ex.full_address}' to MBTA station '{ex.transit_station_name}' is 0.5 miles or less."
    await evaluator.verify(
        claim=transit_claim,
        node=transit_leaf,
        sources=transit_sources if len(transit_sources) > 1 else (transit_sources[0] if transit_sources else None),
        additional_instruction="Verify that the referenced station is an MBTA station (subway, commuter rail, or light rail) and that the walking distance displayed is 0.5 miles or less. Use the map route page and/or station page. Allow minor rounding."
    )

    # 3) Building Classification and Certification (critical)
    cls_node = evaluator.add_parallel(
        id="Building_Classification_and_Certification",
        desc="Meets building class and LEED certification requirements",
        parent=root,
        critical=True
    )

    # 3.1 Class A
    class_leaf = evaluator.add_leaf(
        id="Building_Classification",
        desc="Property is classified as a Class A office building",
        parent=cls_node,
        critical=True
    )
    class_claim = "This property is classified as a Class A office building."
    await evaluator.verify(
        claim=class_claim,
        node=class_leaf,
        sources=ex.building_class_urls if len(ex.building_class_urls) > 1 else (ex.building_class_urls[0] if _list_nonempty(ex.building_class_urls) else None),
        additional_instruction="Check listing pages, broker brochures, or official building information to confirm Class A classification."
    )

    # 3.2 LEED Certification
    leed_leaf = evaluator.add_leaf(
        id="LEED_Certification",
        desc="Building holds at least LEED Gold certification (or higher, e.g., Platinum)",
        parent=cls_node,
        critical=True
    )
    leed_level_text = ex.leed_cert_level or "Gold/Platinum"
    leed_claim = f"The building has LEED {leed_level_text} certification (Gold or higher)."
    await evaluator.verify(
        claim=leed_claim,
        node=leed_leaf,
        sources=ex.leed_urls if len(ex.leed_urls) > 1 else (ex.leed_urls[0] if _list_nonempty(ex.leed_urls) else None),
        additional_instruction="Verify certification via USGBC directory or official sustainability pages. Passing requires evidence of LEED Gold (≥60 points) or Platinum."
    )

    # 4) Space Requirements (critical)
    space_node = evaluator.add_parallel(
        id="Space_Requirements",
        desc="Meets minimum space and coworking density requirements",
        parent=root,
        critical=True
    )

    # 4.1 Contiguous space
    space_leaf = evaluator.add_leaf(
        id="Contiguous_Available_Space",
        desc="Has at least 10,000 square feet of available contiguous office space",
        parent=space_node,
        critical=True
    )
    space_claim = "The property has at least 10,000 square feet of contiguous office space available."
    await evaluator.verify(
        claim=space_claim,
        node=space_leaf,
        sources=ex.space_urls if len(ex.space_urls) > 1 else (ex.space_urls[0] if _list_nonempty(ex.space_urls) else None),
        additional_instruction="Check the listing or brochure for an available contiguous office space figure ≥ 10,000 SF."
    )

    # 4.2 Coworking density suitability
    density_leaf = evaluator.add_leaf(
        id="Coworking_Density_Suitability",
        desc="Office layout supports standard coworking density of 150–200 square feet per person",
        parent=space_node,
        critical=True
    )
    density_claim = "The office layout can reasonably support standard coworking density of 150–200 square feet per person."
    await evaluator.verify(
        claim=density_claim,
        node=density_leaf,
        sources=ex.density_urls if len(ex.density_urls) > 1 else (ex.density_urls[0] if _list_nonempty(ex.density_urls) else None),
        additional_instruction="Use floor plans, space descriptions, or planning notes to infer suitability for 150–200 SF/person. Allow reasonable wording and plans indicating open layouts."
    )

    # 5) ADA Accessibility Compliance (critical)
    ada_node = evaluator.add_parallel(
        id="ADA_Accessibility_Compliance",
        desc="Meets ADA-related entrance, parking, interior route/elevator, and work-surface accessibility requirements",
        parent=root,
        critical=True
    )

    # 5.1 Entrance Compliance (critical group)
    ada_ent_node = evaluator.add_parallel(
        id="ADA_Entrance_Compliance",
        desc="Building entrance accessibility meets stated ADA requirements",
        parent=ada_node,
        critical=True
    )

    # 5.1.1 Entrance accessibility percentage
    ent_pct_leaf = evaluator.add_leaf(
        id="Entrance_Accessibility_Percentage",
        desc="At least 60% of public building entrances are ADA-accessible",
        parent=ada_ent_node,
        critical=True
    )
    await evaluator.verify(
        claim="At least 60% of public building entrances are ADA-accessible at this property.",
        node=ent_pct_leaf,
        sources=ex.ada_entrance_urls if len(ex.ada_entrance_urls) > 1 else (ex.ada_entrance_urls[0] if _list_nonempty(ex.ada_entrance_urls) else None),
        additional_instruction="Look for ADA compliance documents, building specs, or official statements indicating accessible entrances ≥ 60%."
    )

    # 5.1.2 Entrance door clear width
    ent_width_leaf = evaluator.add_leaf(
        id="Entrance_Door_Clear_Width",
        desc="ADA-accessible entrance doors provide a 32-inch minimum clear width opening",
        parent=ada_ent_node,
        critical=True
    )
    await evaluator.verify(
        claim="ADA-accessible entrance doors provide a minimum clear width of 32 inches.",
        node=ent_width_leaf,
        sources=ex.ada_entrance_urls if len(ex.ada_entrance_urls) > 1 else (ex.ada_entrance_urls[0] if _list_nonempty(ex.ada_entrance_urls) else None),
        additional_instruction="Confirm any specification or compliance statement showing entrance door clear width ≥ 32 inches."
    )

    # 5.2 Parking Compliance (critical group)
    ada_park_node = evaluator.add_parallel(
        id="ADA_Parking_Compliance",
        desc="Parking facilities meet stated ADA requirements for van-accessible parking",
        parent=ada_node,
        critical=True
    )

    # 5.2.1 Van-accessible space width
    van_width_leaf = evaluator.add_leaf(
        id="Van_Accessible_Space_Width",
        desc="Provides van-accessible parking spaces with 132-inch minimum width",
        parent=ada_park_node,
        critical=True
    )
    await evaluator.verify(
        claim="Van-accessible parking spaces are at least 132 inches wide.",
        node=van_width_leaf,
        sources=ex.ada_parking_urls if len(ex.ada_parking_urls) > 1 else (ex.ada_parking_urls[0] if _list_nonempty(ex.ada_parking_urls) else None),
        additional_instruction="Check parking specifications or ADA compliance docs showing van space width ≥ 132 inches."
    )

    # 5.2.2 Access aisle width
    aisle_leaf = evaluator.add_leaf(
        id="Access_Aisle_Width",
        desc="Accessible parking spaces have adjacent access aisles of 60-inch minimum width",
        parent=ada_park_node,
        critical=True
    )
    await evaluator.verify(
        claim="Accessible parking spaces have adjacent access aisles of at least 60 inches in width.",
        node=aisle_leaf,
        sources=ex.ada_parking_urls if len(ex.ada_parking_urls) > 1 else (ex.ada_parking_urls[0] if _list_nonempty(ex.ada_parking_urls) else None),
        additional_instruction="Confirm access aisle width ≥ 60 inches using parking specs or compliance documentation."
    )

    # 5.2.3 Vertical clearance
    clearance_leaf = evaluator.add_leaf(
        id="Vertical_Clearance",
        desc="Vehicular routes serving van parking provide 98-inch minimum vertical clearance",
        parent=ada_park_node,
        critical=True
    )
    await evaluator.verify(
        claim="Vehicular routes serving van-accessible parking provide a vertical clearance of at least 98 inches.",
        node=clearance_leaf,
        sources=ex.ada_parking_urls if len(ex.ada_parking_urls) > 1 else (ex.ada_parking_urls[0] if _list_nonempty(ex.ada_parking_urls) else None),
        additional_instruction="Look for garage or route clearance specs ≥ 98 inches for van-accessible areas."
    )

    # 5.3 Interior Compliance (critical group)
    ada_int_node = evaluator.add_parallel(
        id="ADA_Interior_Compliance",
        desc="Interior accessibility meets stated ADA requirements",
        parent=ada_node,
        critical=True
    )

    # 5.3.1 Interior route width
    route_leaf = evaluator.add_leaf(
        id="Interior_Route_Width",
        desc="Interior accessible routes provide 36-inch minimum clear width",
        parent=ada_int_node,
        critical=True
    )
    await evaluator.verify(
        claim="Interior accessible routes provide a minimum clear width of 36 inches.",
        node=route_leaf,
        sources=ex.ada_interior_urls if len(ex.ada_interior_urls) > 1 else (ex.ada_interior_urls[0] if _list_nonempty(ex.ada_interior_urls) else None),
        additional_instruction="Check interior accessibility specs showing clear route width ≥ 36 inches."
    )

    # 5.3.2 Elevators if multi-story
    elevator_leaf = evaluator.add_leaf(
        id="Passenger_Elevators_If_Multi_Story",
        desc="If the building is multi-story, it has ADA-compliant passenger elevators",
        parent=ada_int_node,
        critical=True
    )
    await evaluator.verify(
        claim="If the building is multi-story, it has ADA-compliant passenger elevators.",
        node=elevator_leaf,
        sources=ex.ada_interior_urls if len(ex.ada_interior_urls) > 1 else (ex.ada_interior_urls[0] if _list_nonempty(ex.ada_interior_urls) else None),
        additional_instruction="Check for elevator specifications meeting ADA. If the building is single-story, this requirement can be considered not applicable; otherwise, evidence of ADA-compliant elevators is required."
    )

    # 5.4 Accessible Work Surfaces (critical leaf)
    work_surface_leaf = evaluator.add_leaf(
        id="Accessible_Work_Surfaces",
        desc="Building design allows at least 5% of work surfaces to meet ADA accessibility standards",
        parent=ada_node,
        critical=True
    )
    await evaluator.verify(
        claim="The building design allows at least 5% of work surfaces to meet ADA accessibility standards.",
        node=work_surface_leaf,
        sources=ex.ada_work_surface_urls if len(ex.ada_work_surface_urls) > 1 else (ex.ada_work_surface_urls[0] if _list_nonempty(ex.ada_work_surface_urls) else None),
        additional_instruction="Look for design standards or fit-out specifications indicating at least 5% of work surfaces are ADA-compliant."
    )

    # 6) Parking Ratio (critical leaf)
    ratio_leaf = evaluator.add_leaf(
        id="Parking_Ratio",
        desc="Provides parking at a ratio of at least 4 spaces per 1,000 square feet of office space",
        parent=root,
        critical=True
    )
    await evaluator.verify(
        claim="The property provides a parking ratio of at least 4 spaces per 1,000 square feet of office space.",
        node=ratio_leaf,
        sources=ex.parking_ratio_urls if len(ex.parking_ratio_urls) > 1 else (ex.parking_ratio_urls[0] if _list_nonempty(ex.parking_ratio_urls) else None),
        additional_instruction="Confirm parking ratio specifications on property listing or brochure, requiring ≥ 4/1,000 SF."
    )

    # 7) Coworking Operational Features (critical group)
    ops_node = evaluator.add_parallel(
        id="Coworking_Operational_Features",
        desc="Meets coworking operational amenity requirements",
        parent=root,
        critical=True
    )

    # 7.1 High-speed internet
    internet_leaf = evaluator.add_leaf(
        id="High_Speed_Internet",
        desc="Equipped with (or can support) high-speed internet infrastructure suitable for coworking",
        parent=ops_node,
        critical=True
    )
    await evaluator.verify(
        claim="The building is equipped with or can support high-speed internet infrastructure suitable for coworking.",
        node=internet_leaf,
        sources=ex.internet_urls if len(ex.internet_urls) > 1 else (ex.internet_urls[0] if _list_nonempty(ex.internet_urls) else None),
        additional_instruction="Look for fiber availability, high-speed ISP readiness, or building IT infrastructure statements."
    )

    # 7.2 Meeting and conference rooms with AV
    meeting_leaf = evaluator.add_leaf(
        id="Meeting_and_Conference_Rooms_AV",
        desc="Includes or can accommodate private meeting/conference rooms with audiovisual capabilities",
        parent=ops_node,
        critical=True
    )
    await evaluator.verify(
        claim="The building includes or can accommodate private meeting and conference rooms with audiovisual capabilities.",
        node=meeting_leaf,
        sources=ex.meeting_urls if len(ex.meeting_urls) > 1 else (ex.meeting_urls[0] if _list_nonempty(ex.meeting_urls) else None),
        additional_instruction="Verify floor plans or amenity lists indicating meeting/conference rooms with AV provisions."
    )

    # 7.3 Kitchen and coffee preparation areas
    kitchen_leaf = evaluator.add_leaf(
        id="Kitchen_and_Coffee",
        desc="Includes or can accommodate kitchen and coffee preparation areas",
        parent=ops_node,
        critical=True
    )
    await evaluator.verify(
        claim="The building includes or can accommodate kitchen and coffee preparation areas.",
        node=kitchen_leaf,
        sources=ex.kitchen_urls if len(ex.kitchen_urls) > 1 else (ex.kitchen_urls[0] if _list_nonempty(ex.kitchen_urls) else None),
        additional_instruction="Check amenities or fit-out readiness indicating kitchen/pantry and coffee preparation areas."
    )

    # 7.4 Printing, scanning, mail handling
    print_leaf = evaluator.add_leaf(
        id="Printing_Scanning_Mail",
        desc="Provides or can accommodate printing, scanning, and mail handling services",
        parent=ops_node,
        critical=True
    )
    await evaluator.verify(
        claim="The building provides or can accommodate printing, scanning, and mail handling services.",
        node=print_leaf,
        sources=ex.print_mail_urls if len(ex.print_mail_urls) > 1 else (ex.print_mail_urls[0] if _list_nonempty(ex.print_mail_urls) else None),
        additional_instruction="Look for amenities or building services indicating print/scan capabilities and mail handling."
    )

    # 7.5 Security: controlled access and surveillance
    security_leaf = evaluator.add_leaf(
        id="Security_Controlled_Access_and_Surveillance",
        desc="Has security systems including controlled access and surveillance capabilities",
        parent=ops_node,
        critical=True
    )
    await evaluator.verify(
        claim="The building has security systems including controlled access and surveillance capabilities.",
        node=security_leaf,
        sources=ex.security_urls if len(ex.security_urls) > 1 else (ex.security_urls[0] if _list_nonempty(ex.security_urls) else None),
        additional_instruction="Verify building features mention access control (badging/turnstiles) and surveillance (CCTV) or equivalent."
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
    Evaluate an answer for the Boston coworking building requirements task.
    """
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Root per rubric
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

    # 1) Extract structured building information from the answer
    ex = await evaluator.extract(
        prompt=prompt_extract_building_info(),
        template_class=BuildingExtraction,
        extraction_name="building_extraction"
    )

    # 2) Build and verify tree (ensure identification/doc first to act as gate)
    await build_verification_tree(evaluator, root, ex)

    # 3) Return summary
    return evaluator.get_summary()