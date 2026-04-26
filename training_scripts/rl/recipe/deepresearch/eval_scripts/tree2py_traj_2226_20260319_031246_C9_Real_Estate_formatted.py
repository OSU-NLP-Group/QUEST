import asyncio
import logging
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys


# -----------------------------------------------------------------------------
# Task metadata
# -----------------------------------------------------------------------------
TASK_ID = "seattle_class_a_office_selection"
TASK_DESCRIPTION = (
    "A mid-sized technology company is planning to consolidate its Seattle-area operations and is seeking to identify "
    "potential Class A office buildings that could serve as regional headquarters or major satellite offices. The company "
    "requires at least 3 qualifying properties to compare and evaluate.\n\n"
    "Identify 3 Class A office buildings in the Seattle metropolitan area (including Seattle proper, Bellevue, or immediate "
    "suburbs) that meet ALL of the following requirements:\n\n"
    "1. Building Classification: Must be designated as Class A office buildings\n"
    "2. Size Requirement: Each building must contain at least 500,000 square feet of total office space\n"
    "3. Environmental Certification: Must have achieved LEED certification at Silver level or higher\n"
    "4. Market Data Availability: Current vacancy rate or occupancy information with recent data (2025 or 2026)\n"
    "5. Lease Rate Transparency: Published asking lease rates in $30-$50 per square foot per year range\n"
    "6. Parking Infrastructure: On-site or adjacent parking facilities available\n"
    "7. Established Tenant Base: Currently houses or recently (past 2 years) housed at least one major tech/corporate tenant\n"
    "8. Building Condition: Newly constructed 2010+ OR major renovation since 2011\n"
    "9. Transit Connectivity: Within 0.5 miles of public transportation\n"
    "10. Code Compliance: Meets Seattle seismic code (assume any building constructed or renovated after 2000 qualifies)\n\n"
    "For each building, provide: name and street address, total square footage, LEED certification level with URL, "
    "current vacancy or recent occupancy data, asking lease rate, at least one major tenant with URL, year of construction "
    "or last major renovation, parking description, confirmation of transit accessibility, and notable Class A amenities. "
    "All information must be supported by verifiable sources with URL references provided where applicable."
)


# -----------------------------------------------------------------------------
# Extraction data models
# -----------------------------------------------------------------------------
class BuildingEntry(BaseModel):
    name: Optional[str] = None
    address: Optional[str] = None
    city: Optional[str] = None

    total_square_footage: Optional[str] = None
    class_designation: Optional[str] = None

    leed_level: Optional[str] = None
    leed_sources: List[str] = Field(default_factory=list)

    vacancy_info: Optional[str] = None
    vacancy_year: Optional[str] = None
    vacancy_sources: List[str] = Field(default_factory=list)

    lease_rate: Optional[str] = None  # e.g., "$42/SF/yr FS" or "$38 per square foot per year triple-net"
    lease_rate_sources: List[str] = Field(default_factory=list)

    parking_description: Optional[str] = None
    parking_sources: List[str] = Field(default_factory=list)

    major_tenant: Optional[str] = None
    tenant_sources: List[str] = Field(default_factory=list)

    construction_year: Optional[str] = None
    renovation_year: Optional[str] = None
    year_sources: List[str] = Field(default_factory=list)

    transit_access_description: Optional[str] = None
    transit_sources: List[str] = Field(default_factory=list)

    amenities: List[str] = Field(default_factory=list)
    amenities_sources: List[str] = Field(default_factory=list)

    property_management: Optional[str] = None
    property_management_sources: List[str] = Field(default_factory=list)

    floor_plate: Optional[str] = None
    floor_plate_sources: List[str] = Field(default_factory=list)

    building_urls: List[str] = Field(default_factory=list)  # general reference pages for the building


class BuildingsExtraction(BaseModel):
    buildings: List[BuildingEntry] = Field(default_factory=list)


# -----------------------------------------------------------------------------
# Extraction prompt
# -----------------------------------------------------------------------------
def prompt_extract_buildings() -> str:
    return """
You must extract structured information about Class A office buildings listed in the answer. Extract up to the first 5 buildings mentioned. For each building, fill out all fields exactly from the answer text; do not invent any values. Include all explicit URLs that support each specific field. If a field is not mentioned, set it to null (for strings) or [] (for lists).

For each building, extract:
- name: Building name as written
- address: Complete street address (street + number; include suite if given)
- city: City name
- total_square_footage: Total building office/rentable square footage exactly as stated (e.g., "1,100,000 SF")
- class_designation: Classification text if specified (e.g., "Class A")

- leed_level: LEED certification level if mentioned (Silver, Gold, Platinum)
- leed_sources: URLs directly confirming LEED status/level

- vacancy_info: Current vacancy or occupancy info exactly as stated (e.g., "12% vacant", "88% occupied")
- vacancy_year: Year the vacancy/occupancy figure pertains to (prefer 2025 or 2026 if multiple)
- vacancy_sources: URLs supporting the vacancy/occupancy data

- lease_rate: Published asking lease rate text (e.g., "$42/SF/yr FS", "$38 per sf per year")
- lease_rate_sources: URLs for the lease rate

- parking_description: Text describing on-site or adjacent parking (e.g., "on-site garage with 900 stalls")
- parking_sources: URLs supporting the parking information

- major_tenant: Name of at least one major technology or corporate tenant currently or within the last 2 years
- tenant_sources: URLs confirming that tenant’s presence (press release, property page, news, leasing brochure, etc.)

- construction_year: Year built if provided
- renovation_year: Year of last major renovation if provided
- year_sources: URLs supporting construction/renovation years

- transit_access_description: Text confirming proximity to public transit (e.g., "5-minute walk to Link light rail", "adjacent to bus rapid transit")
- transit_sources: URLs supporting transit accessibility claim

- amenities: List of notable Class A amenities (e.g., "fitness center", "conference center", "food service", "rooftop terrace")
- amenities_sources: URLs supporting amenities

- property_management: Name of professional property management company if mentioned
- property_management_sources: URLs supporting management info

- floor_plate: Typical floor plate size or configuration text if mentioned
- floor_plate_sources: URLs supporting floor plate info

- building_urls: General building URLs (official site, brochure, loopnet, leasing page, Wikipedia, etc.) that are broadly relevant

Output a JSON object with a single field:
{
  "buildings": [ BuildingEntry, ... ]
}

Important:
- Only include URLs explicitly present in the answer. If none are provided for a field, leave the corresponding *sources* array empty.
- Preserve numbers and symbols exactly as written (e.g., keep "$" and "SF").
- Do not merge information across buildings; keep each building separate.
"""


# -----------------------------------------------------------------------------
# Helper utilities
# -----------------------------------------------------------------------------
def merge_sources(*lists: List[str], fallback: Optional[List[str]] = None) -> Optional[List[str]]:
    urls: List[str] = []
    for lst in lists:
        if lst:
            urls.extend([u for u in lst if isinstance(u, str) and u.strip()])
    # Deduplicate while preserving order
    seen = set()
    urls = [u for u in urls if not (u in seen or seen.add(u))]
    if not urls and fallback:
        fb = [u for u in fallback if isinstance(u, str) and u.strip()]
        seen_fb = set()
        fb = [u for u in fb if not (u in seen_fb or seen_fb.add(u))]
        urls = fb
    return urls if urls else None


def building_prefix(idx: int) -> str:
    return f"B{idx + 1}"


def allowed_seattle_metro_description() -> str:
    return (
        "Treat 'Seattle metropolitan area' as inclusive of Seattle proper and immediate suburbs such as Bellevue, "
        "Redmond, Kirkland, Mercer Island, Issaquah, Renton, Shoreline, Tukwila, Bothell, Lynnwood, and similar nearby cities."
    )


# -----------------------------------------------------------------------------
# Verification for a single building
# -----------------------------------------------------------------------------
async def verify_one_building(
    evaluator: Evaluator,
    parent_node,
    b: BuildingEntry,
    idx: int,
) -> None:
    pfx = building_prefix(idx)

    # Create the main building node (parallel; allow partial credit within a building)
    building_node = evaluator.add_parallel(
        id=f"Building_{idx + 1}",
        desc=f"{['First','Second','Third','Fourth','Fifth'][idx] if idx < 5 else f'#{idx+1}'} qualifying Class A office building identified with complete specifications",
        parent=parent_node,
        critical=False,
    )

    # 1) Geographic Location
    n_geo = evaluator.add_leaf(
        id=f"{pfx}_Geographic_Location",
        desc="Building is located within Seattle metropolitan area (Seattle proper, Bellevue, or immediate suburbs)",
        parent=building_node,
        critical=True,
    )
    claim_geo = (
        f"The building '{b.name or ''}' at address '{b.address or ''}', city '{b.city or ''}', is located within the "
        f"Seattle metropolitan area (Seattle proper or an immediate suburb)."
    )
    await evaluator.verify(
        claim=claim_geo,
        node=n_geo,
        sources=merge_sources(b.year_sources, b.tenant_sources, b.parking_sources, b.amenities_sources, fallback=b.building_urls),
        additional_instruction=allowed_seattle_metro_description(),
    )

    # 2) Building Classification (Class A)
    n_class = evaluator.add_leaf(
        id=f"{pfx}_Building_Classification",
        desc="Building is classified as Class A office building (highest-quality with prime location, premium amenities, professional management)",
        parent=building_node,
        critical=True,
    )
    claim_class = "This property is a Class A office building."
    await evaluator.verify(
        claim=claim_class,
        node=n_class,
        sources=merge_sources(b.amenities_sources, b.property_management_sources, fallback=b.building_urls),
        additional_instruction="Accept 'Class A' or equivalent phrasing shown on leasing pages, brokerage listings, or official building materials.",
    )

    # 3) Total Square Footage >= 500,000 SF
    n_sf = evaluator.add_leaf(
        id=f"{pfx}_Total_Square_Footage",
        desc="Building contains at least 500,000 square feet of total office space",
        parent=building_node,
        critical=True,
    )
    claim_sf = (
        f"The total office/rentable square footage for this building is at least 500,000 square feet. "
        f"Stated total: {b.total_square_footage or 'unknown'}."
    )
    await evaluator.verify(
        claim=claim_sf,
        node=n_sf,
        sources=merge_sources(b.amenities_sources, b.year_sources, fallback=b.building_urls),
        additional_instruction="Verify the total building office/rentable area. If multiple figures are shown, prefer the total rentable area. Confirm it is >= 500,000 SF.",
    )

    # 4) LEED Certification (Level + Reference)
    leed_parent = evaluator.add_parallel(
        id=f"{pfx}_LEED_Certification",
        desc="Building has achieved LEED certification at Silver level or higher (Silver, Gold, or Platinum)",
        parent=building_node,
        critical=True,
    )
    n_leed_level = evaluator.add_leaf(
        id=f"{pfx}_LEED_Level",
        desc="Specific LEED certification level is provided (Silver, Gold, or Platinum)",
        parent=leed_parent,
        critical=True,
    )
    claim_leed = f"The building has LEED certification at level {b.leed_level or 'unknown level'} (Silver or higher)."
    await evaluator.verify(
        claim=claim_leed,
        node=n_leed_level,
        sources=merge_sources(b.leed_sources, fallback=b.building_urls),
        additional_instruction="Confirm that the building is LEED Silver, Gold, or Platinum. Accept official LEED listings, property pages, or reputable brokers.",
    )
    n_leed_ref = evaluator.add_custom_node(
        result=bool(b.leed_sources),
        id=f"{pfx}_LEED_Reference",
        desc="URL reference confirming LEED certification is provided",
        parent=leed_parent,
        critical=True,
    )

    # 5) Current Vacancy Information (2025-2026)
    n_vac = evaluator.add_leaf(
        id=f"{pfx}_Current_Vacancy_Information",
        desc="Current vacancy rate or occupancy status is provided with recent data (2025-2026)",
        parent=building_node,
        critical=True,
    )
    claim_vac = (
        f"The building's current vacancy or occupancy information is from 2025 or 2026. "
        f"Stated info: {b.vacancy_info or 'unknown'}, year: {b.vacancy_year or 'unknown'}."
    )
    await evaluator.verify(
        claim=claim_vac,
        node=n_vac,
        sources=merge_sources(b.vacancy_sources, fallback=b.building_urls),
        additional_instruction="Confirm that the data year shown is 2025 or 2026. If multiple dates are shown, prefer the most recent within 2025-2026.",
    )

    # 6) Lease Rate Data: value provided + market compliance
    lease_parent = evaluator.add_parallel(
        id=f"{pfx}_Lease_Rate_Data",
        desc="Published asking lease rates are provided and fall within Seattle Class A office market range ($30-$50 PSF per year)",
        parent=building_node,
        critical=True,
    )
    n_lease_value = evaluator.add_custom_node(
        result=bool(b.lease_rate and b.lease_rate.strip()),
        id=f"{pfx}_Lease_Rate_Value",
        desc="Specific lease rate value is provided in dollars per square foot",
        parent=lease_parent,
        critical=True,
    )
    n_lease_range = evaluator.add_leaf(
        id=f"{pfx}_Lease_Rate_Market_Compliance",
        desc="Lease rate falls within the acceptable market range of $30-$50 PSF per year",
        parent=lease_parent,
        critical=True,
    )
    claim_lease_range = (
        "The published asking lease rate for this building is between $30 and $50 per square foot per year."
    )
    await evaluator.verify(
        claim=claim_lease_range,
        node=n_lease_range,
        sources=merge_sources(b.lease_rate_sources, fallback=b.building_urls),
        additional_instruction="Check the listed asking lease rate and confirm it falls within $30-$50 per SF per year (regardless of FSG, NNN, or similar, unless explicitly outside this range).",
    )

    # 7) Parking Facilities
    n_parking = evaluator.add_leaf(
        id=f"{pfx}_Parking_Facilities",
        desc="On-site or adjacent parking facilities are available",
        parent=building_node,
        critical=True,
    )
    claim_parking = f"The building has on-site or adjacent parking available. Details: {b.parking_description or 'N/A'}."
    await evaluator.verify(
        claim=claim_parking,
        node=n_parking,
        sources=merge_sources(b.parking_sources, fallback=b.building_urls),
        additional_instruction="Accept structured parking garages, underground parking, or adjacent dedicated lots as meeting the requirement.",
    )

    # 8) Major Corporate Tenant (name + reference)
    tenant_parent = evaluator.add_parallel(
        id=f"{pfx}_Major_Corporate_Tenant",
        desc="Building currently houses or has recently housed at least one major technology or corporate tenant",
        parent=building_node,
        critical=True,
    )
    n_tenant_name = evaluator.add_custom_node(
        result=bool(b.major_tenant and b.major_tenant.strip()),
        id=f"{pfx}_Tenant_Name",
        desc="Specific major tenant name is provided",
        parent=tenant_parent,
        critical=True,
    )
    n_tenant_ref = evaluator.add_leaf(
        id=f"{pfx}_Tenant_Reference",
        desc="URL reference confirming major tenant occupancy is provided",
        parent=tenant_parent,
        critical=True,
    )
    claim_tenant = (
        f"The building currently houses or has housed within the past 2 years at least one major technology or corporate tenant: {b.major_tenant or 'N/A'}."
    )
    await evaluator.verify(
        claim=claim_tenant,
        node=n_tenant_ref,
        sources=merge_sources(b.tenant_sources, fallback=b.building_urls),
        additional_instruction="Verify tenant presence in the building at present or within the last 2 years relative to 2026-03-22 (i.e., since 2024). Accept reputable news, press releases, property pages, or leasing materials.",
    )

    # 9) Building Condition (2010+ construction OR renovation since 2011)
    n_age = evaluator.add_leaf(
        id=f"{pfx}_Building_Age_Renovation",
        desc="Building is either newly constructed (post-2010) or has undergone major renovation in the last 15 years",
        parent=building_node,
        critical=True,
    )
    claim_age = (
        f"The building satisfies: constructed in 2010 or later OR has a major renovation since 2011. "
        f"Construction year: {b.construction_year or 'unknown'}, Renovation year: {b.renovation_year or 'unknown'}."
    )
    await evaluator.verify(
        claim=claim_age,
        node=n_age,
        sources=merge_sources(b.year_sources, fallback=b.building_urls),
        additional_instruction="Confirm that either the construction year is >= 2010, or the most recent major renovation year is >= 2011.",
    )

    # 10) Transit Connectivity (within 0.5 miles)
    n_transit = evaluator.add_leaf(
        id=f"{pfx}_Transit_Accessibility",
        desc="Building is located within 0.5 miles of public transportation options",
        parent=building_node,
        critical=True,
    )
    claim_transit = (
        f"The building is within 0.5 miles (walkable distance) of public transit. Details: {b.transit_access_description or 'N/A'}."
    )
    await evaluator.verify(
        claim=claim_transit,
        node=n_transit,
        sources=merge_sources(b.transit_sources, fallback=b.building_urls),
        additional_instruction="Accept proximity statements like '5-10 minute walk' to Link light rail stations or major bus lines as evidence of being within ~0.5 miles.",
    )

    # 11) Code Compliance (Seismic) - assumes post-2000 construction/renovation meets requirement
    n_seismic = evaluator.add_leaf(
        id=f"{pfx}_Seismic_Compliance",
        desc="Building meets current Seattle seismic building code requirements for commercial structures",
        parent=building_node,
        critical=True,
    )
    claim_seismic = (
        "Per the task assumption, any building constructed or renovated after 2000 meets Seattle seismic code requirements. "
        f"This building has construction year {b.construction_year or 'unknown'} and/or renovation year {b.renovation_year or 'unknown'}, "
        "so it meets the seismic compliance requirement if at least one of those years is >= 2000."
    )
    await evaluator.verify(
        claim=claim_seismic,
        node=n_seismic,
        additional_instruction="This is a logical check based on the provided assumption. If either construction_year >= 2000 or renovation_year >= 2000, treat as compliant.",
        extra_prerequisites=[n_age],  # depend on age/renovation verification
    )

    # 12) Building Name and Address (critical)
    n_name_addr = evaluator.add_leaf(
        id=f"{pfx}_Building_Name_Address",
        desc="Complete building name and street address are provided",
        parent=building_node,
        critical=True,
    )
    claim_name_addr = f"The building name is '{b.name or ''}' and the street address is '{b.address or ''}'."
    await evaluator.verify(
        claim=claim_name_addr,
        node=n_name_addr,
        sources=merge_sources(b.building_urls, b.property_management_sources, b.amenities_sources, b.year_sources, fallback=b.building_urls),
        additional_instruction="Verify that both the name and a street address are explicitly correct per the provided source(s).",
    )

    # 13) Amenities (non-critical)
    n_amen = evaluator.add_leaf(
        id=f"{pfx}_Building_Amenities",
        desc="Building offers modern Class A amenities (such as fitness center, conference facilities, food service, etc.)",
        parent=building_node,
        critical=False,
    )
    claim_amen = f"The building offers Class A amenities such as: {', '.join(b.amenities) if b.amenities else 'N/A'}."
    await evaluator.verify(
        claim=claim_amen,
        node=n_amen,
        sources=merge_sources(b.amenities_sources, fallback=b.building_urls),
        additional_instruction="Confirm presence of amenities typical of Class A buildings (fitness center, conference center, food service, rooftop spaces, etc.).",
    )

    # 14) Property Management (non-critical)
    n_mgmt = evaluator.add_leaf(
        id=f"{pfx}_Property_Management",
        desc="Professional property management company information is provided",
        parent=building_node,
        critical=False,
    )
    claim_mgmt = f"The property is professionally managed by '{b.property_management or 'N/A'}'."
    await evaluator.verify(
        claim=claim_mgmt,
        node=n_mgmt,
        sources=merge_sources(b.property_management_sources, fallback=b.building_urls),
        additional_instruction="Verify the stated management company on property pages or reputable broker/owner sites.",
    )

    # 15) Floor Plate Size (non-critical)
    n_floor = evaluator.add_leaf(
        id=f"{pfx}_Floor_Plate_Size",
        desc="Typical floor plate size or configuration information is provided",
        parent=building_node,
        critical=False,
    )
    claim_floor = f"The building has a typical floor plate size or configuration: {b.floor_plate or 'N/A'}."
    await evaluator.verify(
        claim=claim_floor,
        node=n_floor,
        sources=merge_sources(b.floor_plate_sources, fallback=b.building_urls),
        additional_instruction="Confirm typical floor plate size or equivalent configuration details from leasing materials.",
    )


# -----------------------------------------------------------------------------
# Main evaluation entry point
# -----------------------------------------------------------------------------
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
        strategy=AggregationStrategy.PARALLEL,  # Buildings evaluated independently
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

    # Extract building candidates from the answer
    extracted: BuildingsExtraction = await evaluator.extract(
        prompt=prompt_extract_buildings(),
        template_class=BuildingsExtraction,
        extraction_name="buildings_extraction",
    )

    # Create a top-level task node (non-critical here to allow partial credit if <3 qualify)
    task_node = evaluator.add_parallel(
        id="Task_Completion",
        desc="Successfully identify 3 Class A office buildings in the Seattle metropolitan area that meet all specified criteria",
        parent=root,
        critical=False,
    )

    # Normalize to exactly 3 entries (pad with empty entries if fewer)
    buildings = list(extracted.buildings[:3])
    while len(buildings) < 3:
        buildings.append(BuildingEntry())

    # Verify each of the first 3 buildings
    for i in range(3):
        await verify_one_building(evaluator, task_node, buildings[i], i)

    # Optionally, record some custom info
    evaluator.add_custom_info(
        info={"extracted_building_count": len(extracted.buildings)},
        info_type="stats",
        info_name="extraction_summary",
    )

    return evaluator.get_summary()