import asyncio
import logging
from datetime import datetime
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy

TASK_ID = "class_a_office_hubs"
TASK_DESCRIPTION = """I am the Director of Real Estate for a rapidly expanding technology company, and we need to establish regional office hubs across the United States to support our distributed workforce. To maintain our premium corporate image and attract top talent, we require Class A office buildings that meet our company's high standards for quality, accessibility, and employee amenities.

Please identify four distinct Class A office spaces currently available for lease in major U.S. metropolitan areas that meet all of the following requirements:

**Building Classification & Quality:**
- Must be designated as Class A office building (highest quality, premium location, professional management)
- Building must be less than 15 years old
- Must demonstrate high-quality finishes and premium features typical of Class A properties

**Space Requirements:**
- Each office space must provide between 8,000 and 15,000 rentable square feet (RSF)
- Minimum ceiling height of 9 feet
- Must accommodate 50-150 employees (based on standard allocation of 100-150 square feet per person)

**Parking Requirements:**
- Must provide at least 4 parking spaces per 1,000 square feet of office space (minimum 32-60 spaces depending on size)
- At least 2% of total parking spaces must be ADA-accessible, with a minimum of 1 accessible space
- Parking must be either on-site or within 2 blocks of the building

**Essential Amenities (all required):**
- High-speed internet or fiber connectivity
- Conference rooms or meeting facilities

**Premium Amenities (at least 2 required from this list):**
- Fitness center or gym facility
- Café or on-site food service
- Shared common areas or lounges
- Smart building technology
- Outdoor spaces or terraces

**ADA Compliance:**
- Building must have accessible entrance(s) meeting ADA standards
- Elevator access for multi-story buildings
- ADA-compliant restroom facilities

**Listing & Availability:**
- Must be actively listed on at least one major commercial real estate platform (LoopNet, CoStar, Crexi, or CBRE)
- Space must be currently available for lease
- Must provide verifiable leasing contact information (phone number and/or email address)

For each of the four office spaces, please provide:
1. Official property/building name
2. Complete street address (including city and state)
3. Confirmation that it is located in a major U.S. metropolitan area
4. Verification of Class A status and building age
5. Square footage (RSF), ceiling height, and employee capacity
6. Parking specifications and accessibility features
7. List of essential amenities present
8. List of at least 2 premium amenities (specifying which ones)
9. ADA compliance features
10. Name of the commercial real estate platform where it is listed
11. Availability status
12. Leasing contact information
13. Direct URL to the active listing

All information must be verified with URLs from reputable commercial real estate listing platforms or official building/property management websites.
"""

CURRENT_YEAR = datetime.utcnow().year

# ------------------------------- Data Models ------------------------------- #
class OfficeSpace(BaseModel):
    # Basic information
    property_name: Optional[str] = None
    full_address: Optional[str] = None
    city: Optional[str] = None
    state: Optional[str] = None
    metro_area_note: Optional[str] = None  # e.g., "Chicago metro", optional descriptor
    basic_info_sources: List[str] = Field(default_factory=list)

    # Classification & Quality
    class_a_status: Optional[str] = None  # e.g., "Class A" or description
    year_built: Optional[str] = None
    building_quality_features: List[str] = Field(default_factory=list)
    classification_sources: List[str] = Field(default_factory=list)

    # Space specifications
    rsf: Optional[str] = None
    ceiling_height_ft: Optional[str] = None
    employee_capacity: Optional[str] = None  # if the answer provides an explicit capacity
    specs_sources: List[str] = Field(default_factory=list)

    # Parking
    parking_ratio_per_1000: Optional[str] = None  # e.g., "4/1000"
    total_parking_spaces: Optional[str] = None
    accessible_parking_spaces: Optional[str] = None  # e.g., "2%" or "4 spaces"
    parking_location: Optional[str] = None  # "on-site" or "within 2 blocks"
    parking_sources: List[str] = Field(default_factory=list)

    # Essential amenities
    has_high_speed_internet: Optional[str] = None
    has_meeting_facilities: Optional[str] = None
    essential_amenities_sources: List[str] = Field(default_factory=list)

    # Premium amenities
    premium_amenities: List[str] = Field(default_factory=list)
    premium_amenities_sources: List[str] = Field(default_factory=list)

    # ADA compliance
    ada_accessible_entrance: Optional[str] = None
    elevator_access: Optional[str] = None
    ada_compliant_restrooms: Optional[str] = None
    ada_sources: List[str] = Field(default_factory=list)

    # Listing & availability
    listing_platform: Optional[str] = None  # LoopNet, CoStar, Crexi, or CBRE
    availability_status: Optional[str] = None  # "Available", etc.
    contact_name: Optional[str] = None
    contact_phone: Optional[str] = None
    contact_email: Optional[str] = None
    listing_url: Optional[str] = None
    listing_sources: List[str] = Field(default_factory=list)


class OfficeSpacesExtraction(BaseModel):
    spaces: List[OfficeSpace] = Field(default_factory=list)


# ---------------------------- Extraction Prompt ---------------------------- #
def prompt_extract_office_spaces() -> str:
    return """
    Identify up to four distinct Class A office spaces described in the answer. For each office space, extract the following fields EXACTLY as stated in the answer:

    Basic Information:
    - property_name: Official property/building name
    - full_address: Complete street address including city and state
    - city: City name
    - state: State abbreviation or full name
    - metro_area_note: Any statement confirming it is in a major U.S. metropolitan area (if present)
    - basic_info_sources: URLs that verify property name and address (listing page or official site)

    Building Classification & Quality:
    - class_a_status: Text claiming/designating Class A (e.g., "Class A")
    - year_built: Year built or explicit building age statement
    - building_quality_features: List of phrases indicating premium finishes/management typical of Class A
    - classification_sources: URLs supporting Class A and age claims

    Space Specifications:
    - rsf: Rentable square footage; prefer a single suite or a stated range
    - ceiling_height_ft: Ceiling height (feet)
    - employee_capacity: If provided in the answer; otherwise leave null
    - specs_sources: URLs showing RSF and ceiling height

    Parking:
    - parking_ratio_per_1000: Parking ratio statement (e.g., "4 per 1000")
    - total_parking_spaces: Total spaces (if provided)
    - accessible_parking_spaces: ADA-accessible count or percent (if provided)
    - parking_location: "on-site" or indicate proximity (e.g., "within 2 blocks")
    - parking_sources: URLs supporting parking specifications

    Essential Amenities:
    - has_high_speed_internet: Statement confirming high-speed internet or fiber
    - has_meeting_facilities: Statement confirming conference/meeting rooms
    - essential_amenities_sources: URLs supporting essential amenities

    Premium Amenities:
    - premium_amenities: List of at least two premium amenities (fitness center, café/on-site food service, shared lounges, smart building tech, outdoor spaces/terraces)
    - premium_amenities_sources: URLs supporting premium amenities

    ADA Compliance:
    - ada_accessible_entrance: Statement confirming accessible entrance(s)
    - elevator_access: Statement confirming elevator access (for multi-story)
    - ada_compliant_restrooms: Statement confirming ADA-compliant restrooms
    - ada_sources: URLs supporting ADA features

    Listing & Availability:
    - listing_platform: LoopNet, CoStar, Crexi, or CBRE (explicitly mentioned)
    - availability_status: Statement confirming currently available for lease
    - contact_name: Leasing contact name (if provided)
    - contact_phone: Leasing contact phone (if provided)
    - contact_email: Leasing contact email (if provided)
    - listing_url: Direct URL to the active listing for the property
    - listing_sources: Any additional listing URLs (if provided)

    RULES:
    - Extract only what appears in the answer. Do not invent or infer any values.
    - If a field is missing, set it to null; if a URLs list is missing, set it to an empty array.
    - Return a JSON object with 'spaces' as an array of up to four OfficeSpace objects.
    """


# ------------------------------- Helpers ----------------------------------- #
def _merge_sources(*groups: Optional[List[str]], listing_url: Optional[str] = None) -> List[str]:
    urls: List[str] = []
    for group in groups:
        if group:
            for u in group:
                if isinstance(u, str) and u.strip():
                    urls.append(u.strip())
    if listing_url and listing_url.strip():
        urls.append(listing_url.strip())
    # Deduplicate while preserving order
    seen = set()
    deduped = []
    for u in urls:
        if u not in seen:
            deduped.append(u)
            seen.add(u)
    return deduped


def _parse_int(text: Optional[str]) -> Optional[int]:
    if not text:
        return None
    try:
        s = "".join(ch for ch in text if (ch.isdigit()))
        if not s:
            return None
        return int(s)
    except Exception:
        return None


def _parse_float(text: Optional[str]) -> Optional[float]:
    if not text:
        return None
    try:
        cleaned = text.replace(",", "")
        # Keep digits, one dot
        out = []
        dot_used = False
        for ch in cleaned:
            if ch.isdigit():
                out.append(ch)
            elif ch == "." and not dot_used:
                out.append(ch)
                dot_used = True
        s = "".join(out)
        if not s:
            return None
        return float(s)
    except Exception:
        return None


def _capacity_range_from_rsf(rsf_text: Optional[str]) -> Optional[str]:
    rsf = _parse_float(rsf_text)
    if rsf and rsf > 0:
        cap_min = int(rsf / 150.0)
        cap_max = int(rsf / 100.0)
        return f"{cap_min}-{cap_max}"
    return None


# ------------------------ Verification per Office Space --------------------- #
async def verify_office_space(
    evaluator: Evaluator,
    parent_node,
    space: OfficeSpace,
    index: int
) -> None:
    # Create node for this office space under critical root
    office_node = evaluator.add_parallel(
        id=f"office_space_{index+1}",
        desc=f"Class A office space #{index+1} meeting all requirements",
        parent=parent_node,
        critical=True  # Root is critical; framework requires children also critical
    )

    # ---------------- Basic Information ----------------
    basic_node = evaluator.add_parallel(
        id=f"space_{index+1}_basic_information",
        desc=f"Basic identifying information for Office Space {index+1}",
        parent=office_node,
        critical=True
    )
    # Reference existence first (gating)
    evaluator.add_custom_node(
        result=bool(space.listing_url) or bool(space.basic_info_sources),
        id=f"space_{index+1}_basic_info_reference",
        desc=f"Provide URL reference verifying the property name and address",
        parent=basic_node,
        critical=True
    )

    # Property Name
    pn_node = evaluator.add_leaf(
        id=f"space_{index+1}_property_name",
        desc=f"Provide the official property/building name",
        parent=basic_node,
        critical=True
    )
    pn_claim = f"The official property/building name is '{space.property_name or ''}'."
    await evaluator.verify(
        claim=pn_claim,
        node=pn_node,
        sources=_merge_sources(space.basic_info_sources, listing_url=space.listing_url),
        additional_instruction="Verify that the page shows the building's official name; allow minor naming variants (e.g., tower vs building) but it should clearly match."
    )

    # Full Address
    addr_node = evaluator.add_leaf(
        id=f"space_{index+1}_full_address",
        desc=f"Provide complete street address including city and state",
        parent=basic_node,
        critical=True
    )
    addr_claim = f"The complete street address is '{space.full_address or ''}' located in {space.city or ''}, {space.state or ''}."
    await evaluator.verify(
        claim=addr_claim,
        node=addr_node,
        sources=_merge_sources(space.basic_info_sources, listing_url=space.listing_url),
        additional_instruction="Verify the full address components; allow minor formatting (abbreviations, commas)."
    )

    # Metropolitan Area confirmation
    metro_node = evaluator.add_leaf(
        id=f"space_{index+1}_metropolitan_area",
        desc=f"Confirm location is in a major U.S. metropolitan area",
        parent=basic_node,
        critical=True
    )
    city_state = f"{space.city or ''}, {space.state or ''}".strip(", ")
    metro_claim = f"The property is located in {city_state}, which is a major U.S. metropolitan area."
    await evaluator.verify(
        claim=metro_claim,
        node=metro_node,
        sources=_merge_sources(space.basic_info_sources, listing_url=space.listing_url),
        additional_instruction="Use the listing page to confirm the city/state. Treat widely recognized large cities (e.g., New York, Los Angeles, Chicago, San Francisco Bay Area, Seattle, Boston, Houston, Dallas, Austin, Washington DC, Miami, Atlanta, Denver, Phoenix, Philadelphia, San Diego, Minneapolis, Charlotte, San Jose) as major metros."
    )

    # ---------------- Building Classification ----------------
    class_node = evaluator.add_parallel(
        id=f"space_{index+1}_building_classification",
        desc=f"Verification that the building meets Class A office building standards",
        parent=office_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=bool(space.classification_sources) or bool(space.listing_url),
        id=f"space_{index+1}_classification_reference",
        desc=f"Provide URL reference supporting the building's Class A status and age",
        parent=class_node,
        critical=True
    )

    # Class A Status
    classa_node = evaluator.add_leaf(
        id=f"space_{index+1}_class_a_status",
        desc=f"Confirm the building is designated or described as Class A quality with premium features",
        parent=class_node,
        critical=True
    )
    classa_claim = f"The building is designated or described as Class A with premium features."
    await evaluator.verify(
        claim=classa_claim,
        node=classa_node,
        sources=_merge_sources(space.classification_sources, listing_url=space.listing_url),
        additional_instruction="Look for explicit 'Class A' designation or equivalent descriptions on the page."
    )

    # Building Age < 15 years
    age_node = evaluator.add_leaf(
        id=f"space_{index+1}_building_age",
        desc=f"Verify the building is less than 15 years old",
        parent=class_node,
        critical=True
    )
    year_txt = space.year_built or ""
    age_claim = f"The building was built in {year_txt}, which is less than 15 years old as of {CURRENT_YEAR}."
    await evaluator.verify(
        claim=age_claim,
        node=age_node,
        sources=_merge_sources(space.classification_sources, listing_url=space.listing_url),
        additional_instruction=f"Confirm the year built on the page and compute age relative to {CURRENT_YEAR}."
    )

    # Building Quality
    quality_node = evaluator.add_leaf(
        id=f"space_{index+1}_building_quality",
        desc=f"Confirm high-quality finishes and professional management standards typical of Class A properties",
        parent=class_node,
        critical=True
    )
    quality_features = ", ".join(space.building_quality_features) if space.building_quality_features else ""
    quality_claim = f"The building exhibits high-quality finishes and professional management typical of Class A properties. Features: {quality_features}."
    await evaluator.verify(
        claim=quality_claim,
        node=quality_node,
        sources=_merge_sources(space.classification_sources, listing_url=space.listing_url),
        additional_instruction="Look for mentions of premium finishes, Class A amenities, or professional management."
    )

    # ---------------- Space Specifications ----------------
    specs_node = evaluator.add_parallel(
        id=f"space_{index+1}_space_specifications",
        desc=f"Physical space requirements and measurements",
        parent=office_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=bool(space.specs_sources) or bool(space.listing_url),
        id=f"space_{index+1}_specifications_reference",
        desc=f"Provide URL reference verifying square footage and ceiling height specifications",
        parent=specs_node,
        critical=True
    )

    # Square Footage
    rsf_node = evaluator.add_leaf(
        id=f"space_{index+1}_square_footage",
        desc=f"Verify the available space is between 8,000 and 15,000 rentable square feet (RSF)",
        parent=specs_node,
        critical=True
    )
    rsf_claim = f"The available office space is between 8,000 and 15,000 RSF; the stated RSF is '{space.rsf or ''}'."
    await evaluator.verify(
        claim=rsf_claim,
        node=rsf_node,
        sources=_merge_sources(space.specs_sources, listing_url=space.listing_url),
        additional_instruction="Check RSF on the page. If multiple suites are listed, confirm at least one is within 8,000–15,000 RSF. Allow minor rounding."
    )

    # Ceiling Height
    ch_node = evaluator.add_leaf(
        id=f"space_{index+1}_ceiling_height",
        desc=f"Confirm minimum ceiling height of 9 feet or greater",
        parent=specs_node,
        critical=True
    )
    ch_claim = f"The ceiling height is at least 9 feet; the stated ceiling height is '{space.ceiling_height_ft or ''}'."
    await evaluator.verify(
        claim=ch_claim,
        node=ch_node,
        sources=_merge_sources(space.specs_sources, listing_url=space.listing_url),
        additional_instruction="Confirm ceiling height; accept formats like 9', 9 ft, or ranges (e.g., 9–12 ft)."
    )

    # Employee Capacity
    ec_node = evaluator.add_leaf(
        id=f"space_{index+1}_employee_capacity",
        desc=f"Confirm the space can accommodate 50-150 employees based on 100-150 sq ft per person standard",
        parent=specs_node,
        critical=True
    )
    cap_range = _capacity_range_from_rsf(space.rsf)
    if cap_range:
        ec_claim = f"Based on a standard 100–150 sq ft per person and RSF '{space.rsf}', the capacity range is {cap_range}, which includes 50–150 employees."
    else:
        ec_claim = f"The space can accommodate 50–150 employees based on a standard allocation of 100–150 sq ft per person."
    await evaluator.verify(
        claim=ec_claim,
        node=ec_node,
        sources=_merge_sources(space.specs_sources, listing_url=space.listing_url),
        additional_instruction="Use the RSF on the page and the 100–150 sq ft per person standard to determine capacity; confirm it covers 50–150 employees."
    )

    # ---------------- Parking Requirements ----------------
    parking_node = evaluator.add_parallel(
        id=f"space_{index+1}_parking_requirements",
        desc=f"Parking availability and accessibility compliance",
        parent=office_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=bool(space.parking_sources) or bool(space.listing_url),
        id=f"space_{index+1}_parking_reference",
        desc=f"Provide URL reference verifying parking availability and specifications",
        parent=parking_node,
        critical=True
    )

    # Parking Ratio
    pr_node = evaluator.add_leaf(
        id=f"space_{index+1}_total_parking",
        desc=f"Verify at least 4 parking spaces per 1,000 sq ft (minimum 32–60 spaces depending on size)",
        parent=parking_node,
        critical=True
    )
    pr_claim = f"The property provides at least 4 parking spaces per 1,000 sq ft; stated ratio: '{space.parking_ratio_per_1000 or ''}'."
    await evaluator.verify(
        claim=pr_claim,
        node=pr_node,
        sources=_merge_sources(space.parking_sources, listing_url=space.listing_url),
        additional_instruction="Look for 'parking ratio' on the page. If only total spaces and RSF are provided, infer whether total meets ≥4/1000 requirement."
    )

    # Accessible Parking
    ap_node = evaluator.add_leaf(
        id=f"space_{index+1}_accessible_parking",
        desc=f"Confirm at least 2% of parking spaces are ADA-accessible with minimum of 1 accessible space",
        parent=parking_node,
        critical=True
    )
    ap_claim = f"The property provides ≥2% ADA-accessible parking spaces (minimum 1). Stated accessible parking: '{space.accessible_parking_spaces or ''}'."
    await evaluator.verify(
        claim=ap_claim,
        node=ap_node,
        sources=_merge_sources(space.parking_sources, listing_url=space.listing_url),
        additional_instruction="Verify page mentions ADA accessible parking count or percentage meeting ≥2% and at least 1 space."
    )

    # Parking Location
    pl_node = evaluator.add_leaf(
        id=f"space_{index+1}_parking_location",
        desc=f"Verify parking is on-site or within 2 blocks of the building",
        parent=parking_node,
        critical=True
    )
    pl_claim = f"Parking is {space.parking_location or ''}, and is on-site or within 2 blocks of the building."
    await evaluator.verify(
        claim=pl_claim,
        node=pl_node,
        sources=_merge_sources(space.parking_sources, listing_url=space.listing_url),
        additional_instruction="Confirm on the page that parking is on-site or specify a nearby distance within 2 blocks."
    )

    # ---------------- Essential Amenities ----------------
    essential_node = evaluator.add_parallel(
        id=f"space_{index+1}_essential_amenities",
        desc=f"Required amenities that must be present",
        parent=office_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=bool(space.essential_amenities_sources) or bool(space.listing_url),
        id=f"space_{index+1}_essential_amenities_reference",
        desc=f"Provide URL reference verifying essential amenities",
        parent=essential_node,
        critical=True
    )

    # High-speed Internet
    hi_node = evaluator.add_leaf(
        id=f"space_{index+1}_internet_connectivity",
        desc=f"Confirm high-speed internet or fiber connectivity is available",
        parent=essential_node,
        critical=True
    )
    hi_claim = f"High-speed internet or fiber connectivity is available for the property."
    await evaluator.verify(
        claim=hi_claim,
        node=hi_node,
        sources=_merge_sources(space.essential_amenities_sources, listing_url=space.listing_url),
        additional_instruction="Look for 'fiber', 'gigabit', or 'high-speed internet' offerings on the page."
    )

    # Meeting Facilities
    mf_node = evaluator.add_leaf(
        id=f"space_{index+1}_meeting_facilities",
        desc=f"Verify conference rooms or meeting facilities are available",
        parent=essential_node,
        critical=True
    )
    mf_claim = f"The property offers conference rooms or meeting facilities."
    await evaluator.verify(
        claim=mf_claim,
        node=mf_node,
        sources=_merge_sources(space.essential_amenities_sources, listing_url=space.listing_url),
        additional_instruction="Verify the page lists conference rooms, meeting rooms, huddle rooms, or similar."
    )

    # ---------------- Premium Amenities ----------------
    premium_node = evaluator.add_parallel(
        id=f"space_{index+1}_premium_amenities",
        desc=f"At least 2 premium amenities from the specified list",
        parent=office_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=(len(space.premium_amenities) >= 2),
        id=f"space_{index+1}_premium_amenities_reference",
        desc=f"Provide URL reference verifying the premium amenities",
        parent=premium_node,
        critical=True
    )

    # Premium Amenity 1
    pa1_node = evaluator.add_leaf(
        id=f"space_{index+1}_premium_amenity_1",
        desc=f"Identify and verify the first premium amenity",
        parent=premium_node,
        critical=True
    )
    amenity1 = (space.premium_amenities[0] if space.premium_amenities else "") or ""
    pa1_claim = f"Premium amenity present: {amenity1}."
    await evaluator.verify(
        claim=pa1_claim,
        node=pa1_node,
        sources=_merge_sources(space.premium_amenities_sources, listing_url=space.listing_url),
        additional_instruction="Verify one premium amenity from the list (fitness center, café/on-site food, shared lounges, smart building tech, outdoor spaces/terraces)."
    )

    # Premium Amenity 2 (must be different)
    pa2_node = evaluator.add_leaf(
        id=f"space_{index+1}_premium_amenity_2",
        desc=f"Identify and verify the second premium amenity (must be different from the first)",
        parent=premium_node,
        critical=True
    )
    amenity2 = (space.premium_amenities[1] if len(space.premium_amenities) > 1 else "") or ""
    pa2_claim = f"Another distinct premium amenity present (different from the first): {amenity2}."
    await evaluator.verify(
        claim=pa2_claim,
        node=pa2_node,
        sources=_merge_sources(space.premium_amenities_sources, listing_url=space.listing_url),
        additional_instruction=f"The second premium amenity must be different from the first ('{amenity1}'). If it duplicates the first, judge incorrect."
    )

    # ---------------- ADA Compliance ----------------
    ada_node = evaluator.add_parallel(
        id=f"space_{index+1}_ada_compliance",
        desc=f"Building accessibility compliance with ADA standards",
        parent=office_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=bool(space.ada_sources) or bool(space.listing_url),
        id=f"space_{index+1}_ada_reference",
        desc=f"Provide URL reference verifying ADA compliance features",
        parent=ada_node,
        critical=True
    )

    # Accessible Entrance
    ae_node = evaluator.add_leaf(
        id=f"space_{index+1}_accessible_entrance",
        desc=f"Verify the building has accessible entrance(s) complying with ADA standards",
        parent=ada_node,
        critical=True
    )
    ae_claim = f"The building provides accessible entrance(s) complying with ADA standards."
    await evaluator.verify(
        claim=ae_claim,
        node=ae_node,
        sources=_merge_sources(space.ada_sources, listing_url=space.listing_url),
        additional_instruction="Look for mentions of ADA-accessible entry, ramps, or compliant entrances on the page."
    )

    # Elevator Access
    el_node = evaluator.add_leaf(
        id=f"space_{index+1}_elevator_access",
        desc=f"For multi-story buildings, confirm elevator access is available",
        parent=ada_node,
        critical=True
    )
    el_claim = f"Elevator access is available for the property."
    await evaluator.verify(
        claim=el_claim,
        node=el_node,
        sources=_merge_sources(space.ada_sources, listing_url=space.listing_url),
        additional_instruction="Verify the page lists elevators; if the building is multi-story, elevator access must be present."
    )

    # ADA Restrooms
    ar_node = evaluator.add_leaf(
        id=f"space_{index+1}_ada_restrooms",
        desc=f"Verify ADA-compliant restroom facilities are available",
        parent=ada_node,
        critical=True
    )
    ar_claim = f"ADA-compliant restroom facilities are available."
    await evaluator.verify(
        claim=ar_claim,
        node=ar_node,
        sources=_merge_sources(space.ada_sources, listing_url=space.listing_url),
        additional_instruction="Look for ADA-compliant restroom mentions or accessible facilities on the page."
    )

    # ---------------- Listing & Availability ----------------
    list_node = evaluator.add_parallel(
        id=f"space_{index+1}_listing_details",
        desc=f"Active listing verification and contact information",
        parent=office_node,
        critical=True
    )

    # Platform Listing
    plat_node = evaluator.add_leaf(
        id=f"space_{index+1}_platform_listing",
        desc=f"Verify the property is actively listed on LoopNet, CoStar, Crexi, or CBRE",
        parent=list_node,
        critical=True
    )
    plat_claim = f"The property is actively listed on {space.listing_platform or ''}."
    await evaluator.verify(
        claim=plat_claim,
        node=plat_node,
        sources=space.listing_url,
        additional_instruction="Confirm that the listing URL domain corresponds to the stated platform (loopnet.com, costar.com, crexi.com, cbre.com) and the page is an active listing."
    )

    # Availability Status
    avail_node = evaluator.add_leaf(
        id=f"space_{index+1}_availability_status",
        desc=f"Confirm the space is available for immediate lease",
        parent=list_node,
        critical=True
    )
    avail_claim = f"The space is currently available for lease."
    await evaluator.verify(
        claim=avail_claim,
        node=avail_node,
        sources=space.listing_url,
        additional_instruction="Verify 'available' or 'for lease' status on the listing page."
    )

    # Contact Information
    contact_node = evaluator.add_leaf(
        id=f"space_{index+1}_contact_information",
        desc=f"Provide verifiable leasing contact information (phone number and/or email address)",
        parent=list_node,
        critical=True
    )
    contact_text = f"Phone: {space.contact_phone or ''}; Email: {space.contact_email or ''}; Contact: {space.contact_name or ''}"
    contact_claim = f"Verifiable leasing contact information is provided. {contact_text}"
    await evaluator.verify(
        claim=contact_claim,
        node=contact_node,
        sources=space.listing_url,
        additional_instruction="Confirm that the listing page shows a contact phone and/or email for leasing; names/titles also acceptable."
    )

    # Listing Reference (direct URL)
    listref_node = evaluator.add_leaf(
        id=f"space_{index+1}_listing_reference",
        desc=f"Provide the direct URL to the active listing on the commercial real estate platform",
        parent=list_node,
        critical=True
    )
    listref_claim = f"This is the direct URL to the active listing: {space.listing_url or ''}."
    await evaluator.verify(
        claim=listref_claim,
        node=listref_node,
        sources=space.listing_url,
        additional_instruction="Verify that the URL loads a listing page for the property and is currently active."
    )


# ------------------------------- Main Entry -------------------------------- #
async def evaluate_answer(
    client: Any,
    answer: str,
    agent_name: str,
    answer_name: str,
    cache: CacheFileSys,
    semaphore: asyncio.Semaphore,
    logger: logging.Logger,
    model: str = "o4-mini"
) -> Dict[str, Any]:
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Root aggregates four independent office verifications
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

    # Root is critical per rubric; framework requires children of a critical parent also critical
    root.critical = True

    extracted = await evaluator.extract(
        prompt=prompt_extract_office_spaces(),
        template_class=OfficeSpacesExtraction,
        extraction_name="office_spaces_extraction"
    )

    # Use only the first four office spaces; pad with empties if fewer provided
    spaces = list(extracted.spaces[:4])
    while len(spaces) < 4:
        spaces.append(OfficeSpace())

    # Build verification tree for each office space
    for i, space in enumerate(spaces):
        await verify_office_space(evaluator, root, space, i)

    return evaluator.get_summary()