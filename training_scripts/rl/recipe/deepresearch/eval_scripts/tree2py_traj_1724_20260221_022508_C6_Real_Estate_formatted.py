import asyncio
import logging
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator, AggregationStrategy
from obj_task_eval.verification_tree import VerificationNode

# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "gas_station_bay_area_site_eval"
TASK_DESCRIPTION = (
    "You are a commercial real estate consultant helping a client identify a suitable property in the San Francisco Bay Area for developing a modern gas station with a convenience store. "
    "Your task is to find one specific commercial property that meets all of the following requirements:\n\n"
    "Location Requirements:\n"
    "- The property must be located in one of the 9 San Francisco Bay Area counties: Alameda, Contra Costa, Marin, Napa, San Francisco, San Mateo, Santa Clara, Solano, or Sonoma County.\n\n"
    "Property Size and Zoning:\n"
    "- The lot size must be at least 1 acre (43,560 square feet).\n"
    "- The property must be zoned for commercial use that permits gas station and fuel dispensing facility operations (such as C-2 General Commercial or equivalent zoning designation).\n\n"
    "Infrastructure Compliance:\n"
    "- The property must comply with California's underground storage tank regulations effective December 31, 2025, which require all UST systems to be double-walled. The property must either: "
    "(a) have no existing underground storage tanks, (b) have compliant double-walled UST systems already installed, or (c) have documentation showing that any single-walled tanks were permanently closed or upgraded by the deadline.\n\n"
    "Facility Specifications:\n"
    "- The property must be able to accommodate a convenience store building of at least 4,000 square feet.\n"
    "- The site layout must support at least 6 multiproduct fuel dispensers (MPDs).\n"
    "- The property must provide or accommodate parking for at least 40 vehicles.\n\n"
    "Your Deliverable:\n"
    "Provide the following information about the property you identify:\n"
    "1. Complete property address (street address, city, county)\n"
    "2. Lot size in square feet or acres\n"
    "3. Current zoning designation\n"
    "4. Underground storage tank system status and compliance\n"
    "5. Convenience store building size capacity (existing or planned)\n"
    "6. Number of fuel dispensers the site can accommodate\n"
    "7. Parking capacity\n"
    "8. For each piece of information above, provide a reference URL to a commercial real estate listing, property database, municipal zoning ordinance, or other official source that verifies the information."
)

ALLOWED_BAY_AREA_COUNTIES = {
    "alameda",
    "contra costa",
    "marin",
    "napa",
    "san francisco",
    "san mateo",
    "santa clara",
    "solano",
    "sonoma",
}

# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class PropertyAddress(BaseModel):
    street_address: Optional[str] = None
    city: Optional[str] = None
    county: Optional[str] = None
    state: Optional[str] = None


class PropertySources(BaseModel):
    address_sources: List[str] = Field(default_factory=list)
    lot_size_sources: List[str] = Field(default_factory=list)
    zoning_sources: List[str] = Field(default_factory=list)
    zoning_permission_sources: List[str] = Field(default_factory=list)
    ust_sources: List[str] = Field(default_factory=list)
    store_sources: List[str] = Field(default_factory=list)
    dispenser_sources: List[str] = Field(default_factory=list)
    parking_sources: List[str] = Field(default_factory=list)


class PropertyCore(BaseModel):
    address: PropertyAddress = PropertyAddress()
    lot_size: Optional[str] = None
    zoning_designation: Optional[str] = None
    ust_status: Optional[str] = None
    store_sqft: Optional[str] = None
    num_dispensers: Optional[str] = None
    parking_capacity: Optional[str] = None
    listing_urls: List[str] = Field(default_factory=list)
    sources: PropertySources = PropertySources()


class PropertyExtraction(BaseModel):
    primary_property: Optional[PropertyCore] = None
    other_properties: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_property() -> str:
    return """
    Extract information for a single specific commercial property proposed in the answer for a gas station development in the San Francisco Bay Area.

    You must:
    1) Identify the primary property (if multiple properties are mentioned, choose the main one the answer proposes; also list all additional distinct properties separately).
    2) Extract the following fields for the primary property:
       - address.street_address
       - address.city
       - address.county
       - address.state
       - lot_size (string, as written, e.g., "1.5 acres" or "65,000 sq ft")
       - zoning_designation (e.g., "C-2", "GC", etc.)
       - ust_status (e.g., "no USTs", "double-walled USTs", "single-walled closed by 12/31/2025", etc.)
       - store_sqft (string, as written, e.g., "4,500 sq ft" or "≥ 4,000 sq ft")
       - num_dispensers (string, as written, e.g., "6 MPDs" or "6 pumps / 12 fueling positions")
       - parking_capacity (string, as written, e.g., "40 spaces" or "≥ 40 stalls")
       - listing_urls (all general listing or property page URLs relevant to this property)
    3) For each of the following fields, also extract URLs that directly verify them (if provided in the answer):
       - sources.address_sources: URLs verifying the address (can be a listing or property database page)
       - sources.lot_size_sources: URLs where the lot size is shown
       - sources.zoning_sources: URLs showing the zoning designation for the property
       - sources.zoning_permission_sources: URLs to municipal code, ordinance, or official documents that show gas stations are permitted/allowable under the zoning
       - sources.ust_sources: URLs supporting the claimed UST status/compliance
       - sources.store_sources: URLs supporting the store size capacity
       - sources.dispenser_sources: URLs supporting the number of MPDs or fueling positions
       - sources.parking_sources: URLs supporting parking capacity

    Also extract:
       - other_properties: a list of any additional specific property addresses or listing URLs mentioned in the answer (excluding the primary property).

    Rules:
    - Return null when a required scalar field is not present in the answer.
    - For each sources.* field and listing_urls, return an array of explicit URLs only; if none are present, return an empty array.
    - Do not invent any URLs. Only include URLs explicitly present in the answer text (or linked in markdown format).
    - Keep all scalars as strings exactly as they appear in the answer (e.g., "1.2 acres", "50,000 SF", "C-2 General Commercial").
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def _normalize_county(county: Optional[str]) -> Optional[str]:
    if county is None:
        return None
    c = county.strip().lower()
    if c.endswith(" county"):
        c = c[:-7].strip()
    return c


def _has_nonempty(s: Optional[str]) -> bool:
    return bool(s and s.strip())


def _unique_nonempty(urls: List[str]) -> List[str]:
    # Normalize and keep unique while preserving order
    seen = set()
    out = []
    for u in urls:
        if not _has_nonempty(u):
            continue
        if u not in seen:
            seen.add(u)
            out.append(u)
    return out


def _select_sources(preferred: List[str], *fallback_groups: List[str]) -> List[str]:
    combined = _unique_nonempty(preferred)
    if not combined:
        for group in fallback_groups:
            combined = _unique_nonempty(group)
            if combined:
                break
    return combined


# --------------------------------------------------------------------------- #
# Verification builders                                                       #
# --------------------------------------------------------------------------- #
async def verify_property_identification(
    evaluator: Evaluator,
    parent: VerificationNode,
    data: PropertyExtraction
) -> None:
    # Parent node for Property Identification (critical)
    prop_ident = evaluator.add_parallel(
        id="Property_Identification",
        desc="Identify a single specific property and provide required location/address info.",
        parent=parent,
        critical=True
    )

    primary = data.primary_property or PropertyCore()
    other_props = data.other_properties or []

    # Exactly one specific property
    has_identifier = (_has_nonempty(primary.address.street_address) and _has_nonempty(primary.address.city)) or bool(primary.listing_urls)
    exactly_one = has_identifier and len(other_props) == 0
    evaluator.add_custom_node(
        result=exactly_one,
        id="Exactly_One_Specific_Property",
        desc="Identifies one (and only one) specific commercial property as the proposed site.",
        parent=prop_ident,
        critical=True
    )

    # Address block: split into source presence and source-backed verification
    addr_block = evaluator.add_parallel(
        id="Complete_Address_With_Source_Block",
        desc="Complete address present and verified by at least one URL.",
        parent=prop_ident,
        critical=True
    )

    # Address source presence
    addr_sources = _select_sources(primary.sources.address_sources, primary.listing_urls)
    addr_source_provided = evaluator.add_custom_node(
        result=len(addr_sources) > 0,
        id="Complete_Address_Source_Provided",
        desc="At least one URL is provided to verify the address.",
        parent=addr_block,
        critical=True
    )

    # Address verification via sources
    addr_leaf = evaluator.add_leaf(
        id="Complete_Address_With_Source",
        desc="Provides complete property address (street address, city, county) and includes at least one reference URL verifying the address.",
        parent=addr_block,
        critical=True
    )
    street = primary.address.street_address or ""
    city = primary.address.city or ""
    county = primary.address.county or ""
    claim_addr = f"The property's address is '{street}, {city}, {county} County' or an equivalent standard format for this address on the cited page(s)."
    await evaluator.verify(
        claim=claim_addr,
        node=addr_leaf,
        sources=addr_sources,
        additional_instruction="Verify that the cited page(s) show the same property address (minor formatting variations acceptable). The county may appear elsewhere on the page or be implied by the jurisdiction."
    )

    # Bay Area county eligibility (no source required; deterministic check)
    county_norm = _normalize_county(primary.address.county)
    in_bay_area = county_norm in ALLOWED_BAY_AREA_COUNTIES if county_norm else False
    evaluator.add_custom_node(
        result=in_bay_area,
        id="Bay_Area_County_Eligibility",
        desc="The county stated for the property is one of: Alameda, Contra Costa, Marin, Napa, San Francisco, San Mateo, Santa Clara, Solano, Sonoma.",
        parent=prop_ident,
        critical=True
    )


async def verify_constraints_and_required_fields(
    evaluator: Evaluator,
    parent: VerificationNode,
    data: PropertyExtraction
) -> None:
    primary = data.primary_property or PropertyCore()

    constraints = evaluator.add_parallel(
        id="Constraint_Compliance_and_Required_Sourced_Fields",
        desc="Check that the property meets all constraints and that required attributes are provided with sources where requested.",
        parent=parent,
        critical=True
    )

    # Lot size ≥ 1 acre with source (split presence + verification)
    lot_block = evaluator.add_parallel(
        id="Lot_Size_Block",
        desc="Lot size at least 1 acre and verified by at least one URL.",
        parent=constraints,
        critical=True
    )
    lot_sources = _select_sources(primary.sources.lot_size_sources, primary.listing_urls)
    evaluator.add_custom_node(
        result=len(lot_sources) > 0,
        id="Lot_Size_Source_Provided",
        desc="At least one URL provided verifying lot size.",
        parent=lot_block,
        critical=True
    )
    lot_leaf = evaluator.add_leaf(
        id="Lot_Size_At_Least_One_Acre_With_Source",
        desc="States lot size is ≥ 1 acre (≥ 43,560 sq ft) and includes at least one reference URL verifying the lot size.",
        parent=lot_block,
        critical=True
    )
    lot_size_str = primary.lot_size or ""
    claim_lot = f"The lot size for the property is at least 1 acre (>= 43,560 square feet). The cited page(s) show a lot size such as '{lot_size_str}' that meets or exceeds this requirement."
    await evaluator.verify(
        claim=claim_lot,
        node=lot_leaf,
        sources=lot_sources,
        additional_instruction="Accept reasonable equivalents like '1.0 acres', '1+ acres', '≥ 43,560 SF', or '50,000 SF'. If unit is in SF, ensure it is >= 43,560."
    )

    # Zoning permits gas station: split into zoning designation value and allowance evidence
    zoning_block = evaluator.add_parallel(
        id="Zoning_Block",
        desc="Zoning designation is stated and zoning allows gas station uses, each supported by sources.",
        parent=constraints,
        critical=True
    )

    # Zoning designation value with source
    zoning_value_sources = _select_sources(primary.sources.zoning_sources, primary.listing_urls)
    evaluator.add_custom_node(
        result=len(zoning_value_sources) > 0,
        id="Zoning_Value_Source_Provided",
        desc="At least one URL provided verifying the zoning designation.",
        parent=zoning_block,
        critical=True
    )
    zoning_value_leaf = evaluator.add_leaf(
        id="Zoning_Designation_With_Source",
        desc="Provides the current zoning designation with a verifying URL.",
        parent=zoning_block,
        critical=True
    )
    zoning = primary.zoning_designation or ""
    claim_zoning_value = f"The property's current zoning designation is '{zoning}' as shown on the cited page(s)."
    await evaluator.verify(
        claim=claim_zoning_value,
        node=zoning_value_leaf,
        sources=zoning_value_sources,
        additional_instruction="Allow minor formatting differences (e.g., 'C-2', 'C2', or 'C-2 General Commercial'). The page should clearly identify the zoning for this parcel/property."
    )

    # Zoning allows gas stations with ordinance/official source
    zoning_perm_sources = _select_sources(primary.sources.zoning_permission_sources, primary.sources.zoning_sources)
    evaluator.add_custom_node(
        result=len(zoning_perm_sources) > 0,
        id="Zoning_Allowance_Source_Provided",
        desc="At least one official/ordinance URL provided indicating gas stations are allowed under the zoning.",
        parent=zoning_block,
        critical=True
    )
    zoning_allow_leaf = evaluator.add_leaf(
        id="Zoning_Permits_Gas_Station_With_Source",
        desc="Zoning allows gas station/fuel dispensing for the stated zoning, supported by an ordinance or equivalent official source.",
        parent=zoning_block,
        critical=True
    )
    city = primary.address.city or ""
    county = primary.address.county or ""
    juris = city if _has_nonempty(city) else county
    claim_zoning_allow = (
        f"Under the '{zoning}' zoning in {juris}, gasoline service stations or fuel dispensing uses are permitted "
        f"(either by right or via conditional/special use), as stated in the cited ordinance or official zoning document."
    )
    await evaluator.verify(
        claim=claim_zoning_allow,
        node=zoning_allow_leaf,
        sources=zoning_perm_sources,
        additional_instruction="Look for terms like 'gasoline service station', 'service station', 'fueling station', 'automobile service station', 'motor vehicle fueling', permitted or conditional use lists, or use tables."
    )

    # UST status meets 2025 requirement with source (split presence + verification)
    ust_block = evaluator.add_parallel(
        id="UST_Status_Block",
        desc="UST status meets California 12/31/2025 double-wall requirement with at least one supporting URL.",
        parent=constraints,
        critical=True
    )
    ust_sources = _select_sources(primary.sources.ust_sources, primary.listing_urls)
    evaluator.add_custom_node(
        result=len(ust_sources) > 0,
        id="UST_Source_Provided",
        desc="At least one URL provided supporting the stated UST status/compliance.",
        parent=ust_block,
        critical=True
    )
    ust_leaf = evaluator.add_leaf(
        id="UST_Status_Meets_2025_Double_Wall_Requirement_With_Source",
        desc="UST status satisfies the 12/31/2025 double-wall requirement with supporting source(s).",
        parent=ust_block,
        critical=True
    )
    ust_status = primary.ust_status or ""
    claim_ust = (
        "The property's underground storage tank (UST) status meets California's 12/31/2025 double-wall requirement, "
        f"such that either (a) there are no existing USTs, (b) existing USTs are double-walled, or (c) any single-walled tanks were permanently closed or upgraded by the deadline. "
        f"The cited page(s) support the stated status: '{ust_status}'."
    )
    await evaluator.verify(
        claim=claim_ust,
        node=ust_leaf,
        sources=ust_sources,
        additional_instruction="Check for explicit statements about double-walled USTs, lack of USTs, or closure/upgrade documentation by 12/31/2025. Accept official records, environmental reports, or authoritative listings."
    )

    # Facility specifications: convenience store size, MPDs, parking, each with source presence + verification
    facility_block = evaluator.add_parallel(
        id="Facility_Specifications",
        desc="Check required facility specification constraints (store size, dispensers, parking) with sources.",
        parent=constraints,
        critical=True
    )

    # Store size ≥ 4,000 sq ft
    store_block = evaluator.add_parallel(
        id="Store_Size_Block",
        desc="Store size capacity at least 4,000 sq ft with supporting URL.",
        parent=facility_block,
        critical=True
    )
    store_sources = _select_sources(primary.sources.store_sources, primary.listing_urls)
    evaluator.add_custom_node(
        result=len(store_sources) > 0,
        id="Store_Size_Source_Provided",
        desc="At least one URL provided supporting the store size capacity.",
        parent=store_block,
        critical=True
    )
    store_leaf = evaluator.add_leaf(
        id="Convenience_Store_Size_At_Least_4000_With_Source",
        desc="States that the site can accommodate a convenience store building of at least 4,000 sq ft and includes at least one reference URL.",
        parent=store_block,
        critical=True
    )
    store_sqft = primary.store_sqft or ""
    claim_store = (
        "The site can accommodate a convenience store building of at least 4,000 square feet. "
        f"The cited page(s) indicate a store size capacity such as '{store_sqft}' that meets or exceeds this requirement."
    )
    await evaluator.verify(
        claim=claim_store,
        node=store_leaf,
        sources=store_sources,
        additional_instruction="Allow phrasing like '4,000+ sq ft', '≥ 4,000 SF', or a numeric value >= 4,000. The page may describe existing or planned store size."
    )

    # MPDs ≥ 6
    mpd_block = evaluator.add_parallel(
        id="MPD_Block",
        desc="Fuel dispensers (MPDs) at least 6 with supporting URL.",
        parent=facility_block,
        critical=True
    )
    mpd_sources = _select_sources(primary.sources.dispenser_sources, primary.listing_urls)
    evaluator.add_custom_node(
        result=len(mpd_sources) > 0,
        id="MPD_Source_Provided",
        desc="At least one URL provided supporting the number of MPDs/fueling positions.",
        parent=mpd_block,
        critical=True
    )
    mpd_leaf = evaluator.add_leaf(
        id="Fuel_Dispensers_At_Least_6_MPDs_With_Source",
        desc="States the site layout can accommodate at least 6 MPDs and includes at least one reference URL.",
        parent=mpd_block,
        critical=True
    )
    num_disp = primary.num_dispensers or ""
    claim_mpd = (
        "The site layout can accommodate at least six multiproduct dispensers (MPDs) or equivalent fueling positions/pumps. "
        f"The cited page(s) support a capacity such as '{num_disp}' that meets or exceeds six."
    )
    await evaluator.verify(
        claim=claim_mpd,
        node=mpd_leaf,
        sources=mpd_sources,
        additional_instruction="Accept synonyms like 'pumps' or 'fueling positions'. If positions are stated, ensure total fueling positions imply ≥6 MPDs or equivalent."
    )

    # Parking ≥ 40 vehicles
    parking_block = evaluator.add_parallel(
        id="Parking_Block",
        desc="Parking capacity at least 40 vehicles with supporting URL.",
        parent=facility_block,
        critical=True
    )
    parking_sources = _select_sources(primary.sources.parking_sources, primary.listing_urls)
    evaluator.add_custom_node(
        result=len(parking_sources) > 0,
        id="Parking_Source_Provided",
        desc="At least one URL provided supporting parking capacity.",
        parent=parking_block,
        critical=True
    )
    parking_leaf = evaluator.add_leaf(
        id="Parking_At_Least_40_With_Source",
        desc="States the site provides or can accommodate parking for at least 40 vehicles and includes at least one reference URL.",
        parent=parking_block,
        critical=True
    )
    parking_str = primary.parking_capacity or ""
    claim_parking = (
        "The property provides or can accommodate parking for at least 40 vehicles. "
        f"The cited page(s) show a capacity such as '{parking_str}' that meets or exceeds this requirement."
    )
    await evaluator.verify(
        claim=claim_parking,
        node=parking_leaf,
        sources=parking_sources,
        additional_instruction="Accept phrasing such as '40+ spaces', '≥ 40 stalls', or a numeric value >= 40."
    )

    # Additional infrastructure and safety constraints (statements in the answer; no extra sources required)
    additional_block = evaluator.add_parallel(
        id="Additional_Infrastructure_and_Safety_Constraints",
        desc="Address additional explicitly listed safety/infrastructure constraints (statements present in the answer).",
        parent=constraints,
        critical=True
    )

    # ADA compliance features
    ada_leaf = evaluator.add_leaf(
        id="ADA_Compliance_Features",
        desc="States the facility will include ADA-compliant features including accessible fueling stations, accessible restrooms, and accessible parking spaces as required by the ADA.",
        parent=additional_block,
        critical=True
    )
    claim_ada = (
        "The answer explicitly commits that the facility will include ADA-compliant features, including accessible fueling stations, accessible restrooms, and accessible parking spaces as required by the ADA."
    )
    await evaluator.verify(
        claim=claim_ada,
        node=ada_leaf,
        additional_instruction="Check the answer text for a clear commitment or statement covering the three ADA elements."
    )

    # Canopy clearance ≥ 14 ft
    canopy_leaf = evaluator.add_leaf(
        id="Canopy_Clearance_At_Least_14ft",
        desc="States the fuel dispensing area canopy has (or will have) a minimum clearance height of 14 feet.",
        parent=additional_block,
        critical=True
    )
    claim_canopy = "The answer explicitly states that the fuel dispensing area canopy will have a minimum clearance height of at least 14 feet."
    await evaluator.verify(
        claim=claim_canopy,
        node=canopy_leaf,
        additional_instruction="Minor wording variations acceptable (e.g., '14 ft minimum canopy clearance')."
    )

    # Automatic fire suppression
    afs_leaf = evaluator.add_leaf(
        id="Automatic_Fire_Suppression",
        desc="States the fuel dispensing area includes (or will include) automatic fire suppression systems as required by California fire safety regulations.",
        parent=additional_block,
        critical=True
    )
    claim_afs = "The answer explicitly states that the fuel dispensing area will include automatic fire suppression systems compliant with California fire safety regulations."
    await evaluator.verify(
        claim=claim_afs,
        node=afs_leaf,
        additional_instruction="Accept synonymous phrasing like 'automatic suppression system' or 'fire suppression under canopy.'"
    )

    # Certified vapor recovery system
    vapor_leaf = evaluator.add_leaf(
        id="Certified_Vapor_Recovery_System",
        desc="States the facility is (or will be) equipped with a certified vapor recovery system.",
        parent=additional_block,
        critical=True
    )
    claim_vapor = "The answer explicitly states that the facility will be equipped with a certified vapor recovery system."
    await evaluator.verify(
        claim=claim_vapor,
        node=vapor_leaf,
        additional_instruction="Accept 'Stage II vapor recovery' or similar compliant terminology."
    )

    # Air quality district permit
    aqd_leaf = evaluator.add_leaf(
        id="Air_Quality_District_Permit",
        desc="States the facility can obtain (or already has) an Air Quality District permit for fuel dispensing operations.",
        parent=additional_block,
        critical=True
    )
    claim_aqd = "The answer explicitly states that the facility can obtain or already has an Air Quality District permit for fuel dispensing operations."
    await evaluator.verify(
        claim=claim_aqd,
        node=aqd_leaf,
        additional_instruction="Accept mention of 'BAAQMD' or relevant local air district permitting as equivalent."
    )

    # Public restrooms per CA Health and Safety Code
    rest_leaf = evaluator.add_leaf(
        id="Public_Restrooms_CA_Health_Code",
        desc="States the convenience store includes (or will include) public restroom facilities meeting California Health and Safety Code requirements.",
        parent=additional_block,
        critical=True
    )
    claim_rest = "The answer explicitly commits that the convenience store will include public restrooms meeting California Health and Safety Code requirements."
    await evaluator.verify(
        claim=claim_rest,
        node=rest_leaf,
        additional_instruction="Accept clear equivalences like 'restrooms compliant with CA Health and Safety Code'."
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
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.SEQUENTIAL,
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

    # Extract structured info
    extracted: PropertyExtraction = await evaluator.extract(
        prompt=prompt_extract_property(),
        template_class=PropertyExtraction,
        extraction_name="property_extraction"
    )

    # Build top-level critical node to represent the rubric root
    top = evaluator.add_sequential(
        id="Gas_Station_Development_Site_Suitability",
        desc="Assess whether the identified single property satisfies all stated constraints and includes the required sourced deliverables.",
        parent=root,
        critical=True
    )

    # Subtrees
    await verify_property_identification(evaluator, top, extracted)
    await verify_constraints_and_required_fields(evaluator, top, extracted)

    return evaluator.get_summary()