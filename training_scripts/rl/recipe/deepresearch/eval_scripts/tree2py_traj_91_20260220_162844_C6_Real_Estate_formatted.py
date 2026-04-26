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
TASK_ID = "breeze_intuit_office_2026"
TASK_DESCRIPTION = """Identify one US city where both Breeze Airways operates direct flight service (as of February 2026) and Intuit maintains an office location. For the identified city, verify that the existing Intuit office space meets the following 2026 commercial office standards for accommodating a team of 50 employees:

1. Space Allocation: The office must provide 100-150 square feet per employee (total 5,000-7,500 sq ft required for 50 employees)
2. Parking Requirements: Parking availability should meet commercial office standards of 3-5 spaces per 1,000 square feet of office space
3. ADA Accessibility: The office must meet ADA compliance requirements, including:
   - Accessible routes and aisles with a minimum 36-inch clear width
   - Accessible common use circulation paths in work areas of 1,000 sq ft or larger
4. Ceiling Height: Office ceiling height must meet the minimum requirement of 7.5-8 feet
5. Building Classification: Identify whether the building is Class A commercial property (featuring professional management, prominent location, and top-tier HVAC and lighting systems)
6. Lease Terms: Note whether lease terms align with typical commercial office standards of 3-10 years (5-10 years for larger spaces)

Provide the city name, the Intuit office address, and verification that each requirement is met, with supporting URL references from official sources (Breeze Airways, Intuit, and relevant commercial real estate or building code sources)."""


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class PrimaryInfo(BaseModel):
    city: Optional[str] = None
    intuit_office_address: Optional[str] = None
    breeze_official_urls: List[str] = Field(default_factory=list)
    intuit_office_official_urls: List[str] = Field(default_factory=list)


class SpaceAllocationData(BaseModel):
    office_sqft: Optional[str] = None
    office_sqft_urls: List[str] = Field(default_factory=list)
    standard_urls: List[str] = Field(default_factory=list)


class ParkingData(BaseModel):
    parking_ratio_per_1000: Optional[str] = None
    parking_total_spaces: Optional[str] = None
    parking_office_urls: List[str] = Field(default_factory=list)
    standard_urls: List[str] = Field(default_factory=list)


class ADAData(BaseModel):
    ada_36in_statement: Optional[str] = None
    ada_common_paths_statement: Optional[str] = None
    office_ada_urls: List[str] = Field(default_factory=list)
    standard_urls: List[str] = Field(default_factory=list)


class CeilingData(BaseModel):
    ceiling_height: Optional[str] = None
    office_ceiling_urls: List[str] = Field(default_factory=list)
    standard_urls: List[str] = Field(default_factory=list)


class BuildingClassData(BaseModel):
    building_class: Optional[str] = None
    building_class_urls: List[str] = Field(default_factory=list)
    definition_urls: List[str] = Field(default_factory=list)


class LeaseTermsData(BaseModel):
    lease_term_years: Optional[str] = None
    lease_terms_urls: List[str] = Field(default_factory=list)
    standard_urls: List[str] = Field(default_factory=list)


class CityOfficeExtraction(BaseModel):
    primary: PrimaryInfo = PrimaryInfo()
    space_allocation: SpaceAllocationData = SpaceAllocationData()
    parking: ParkingData = ParkingData()
    ada: ADAData = ADAData()
    ceiling: CeilingData = CeilingData()
    building_class: BuildingClassData = BuildingClassData()
    lease_terms: LeaseTermsData = LeaseTermsData()


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_city_office() -> str:
    return """
    Extract from the answer the single selected city and the corresponding Intuit office address, plus the exact URLs cited to support Breeze service, Intuit office presence, and each standard check.

    Return JSON with this schema:
    {
      "primary": {
        "city": string or null,
        "intuit_office_address": string or null,
        "breeze_official_urls": string[] (official Breeze urls as listed in the answer),
        "intuit_office_official_urls": string[] (official Intuit urls from the answer that substantiate the office presence/address)
      },
      "space_allocation": {
        "office_sqft": string or null,  // e.g., "6,200 sq ft", "7000 RSF", or a range like "6,000–6,500 sq ft"
        "office_sqft_urls": string[],   // urls that substantiate the square footage for the identified office
        "standard_urls": string[]       // urls that substantiate the 100–150 sq ft per employee standard
      },
      "parking": {
        "parking_ratio_per_1000": string or null,  // e.g., "4/1,000 sq ft"
        "parking_total_spaces": string or null,    // e.g., "30 spaces"
        "parking_office_urls": string[],           // urls that substantiate the parking availability/ratio for this office/building
        "standard_urls": string[]                  // urls that substantiate the 3–5 spaces per 1,000 sq ft standard
      },
      "ada": {
        "ada_36in_statement": string or null,      // any phrasing used in the answer for 36-inch clear width routes/aisles compliance
        "ada_common_paths_statement": string or null, // any phrasing used in the answer for accessible common use circulation paths in 1,000+ sq ft work areas
        "office_ada_urls": string[],               // urls that substantiate ADA compliance for the office/building
        "standard_urls": string[]                  // urls that substantiate the ADA requirements (36-inch clear width and accessible common circulation paths)
      },
      "ceiling": {
        "ceiling_height": string or null,          // e.g., "8 ft", "9'", "8-9 feet"
        "office_ceiling_urls": string[],           // urls that substantiate the office/building ceiling height
        "standard_urls": string[]                  // urls that substantiate minimum office ceiling height of 7.5–8 ft
      },
      "building_class": {
        "building_class": string or null,          // e.g., "Class A", "Class B"
        "building_class_urls": string[],           // urls that substantiate the building's classification
        "definition_urls": string[]                // urls that define/explain Class A office criteria
      },
      "lease_terms": {
        "lease_term_years": string or null,        // e.g., "5-7 years", "7 years", "3-10 years"
        "lease_terms_urls": string[],              // urls that substantiate the lease term(s) for this office/building
        "standard_urls": string[]                  // urls that substantiate typical office lease terms of 3–10 years (5–10 for larger spaces)
      }
    }

    Requirements:
    - Only extract URLs that actually appear in the answer (plain or markdown). Do not invent or infer URLs.
    - If the answer gives multiple cities or addresses, pick the single main one the answer uses for verification.
    - Keep values as strings exactly as written where applicable (e.g., "6,200 sq ft", "4/1,000 sf").
    - If a value or URL set is missing, set to null (for scalar) or [] (for arrays).
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def _is_nonempty(s: Optional[str]) -> bool:
    return bool(s and str(s).strip())


def _safe_urls(urls: Optional[List[str]]) -> List[str]:
    return urls or []


# --------------------------------------------------------------------------- #
# Verification builders                                                       #
# --------------------------------------------------------------------------- #
async def build_identification_section(evaluator: Evaluator, parent, extr: CityOfficeExtraction) -> None:
    """
    Create 'Identify_City_and_Office_Address' section with two existence checks.
    """
    node = evaluator.add_parallel(
        id="Identify_City_and_Office_Address",
        desc="Provide (a) the selected US city name and (b) the Intuit office address in that city.",
        parent=parent,
        critical=True
    )

    evaluator.add_custom_node(
        result=_is_nonempty(extr.primary.city),
        id="City_Name_Provided",
        desc="City name is explicitly stated.",
        parent=node,
        critical=True
    )

    evaluator.add_custom_node(
        result=_is_nonempty(extr.primary.intuit_office_address),
        id="Intuit_Office_Address_Provided",
        desc="A specific Intuit office street address in the selected city is provided.",
        parent=node,
        critical=True
    )


async def build_breeze_service_check(evaluator: Evaluator, parent, extr: CityOfficeExtraction) -> None:
    """
    Verify Breeze Airways direct service for the selected city with official Breeze source(s).
    """
    node = evaluator.add_leaf(
        id="Breeze_Direct_Service_AsOf_Feb2026",
        desc="Verifies Breeze Airways operates direct flight service serving the selected city as of February 2026 and includes a supporting URL from an official Breeze Airways source.",
        parent=parent,
        critical=True
    )
    city = extr.primary.city or "the selected city"
    claim = f"Breeze Airways operates direct (nonstop) flight service serving {city} as of February 2026."
    await evaluator.verify(
        claim=claim,
        node=node,
        sources=_safe_urls(extr.primary.breeze_official_urls),
        additional_instruction=(
            "Verify on official Breeze sources (e.g., flybreeze.com destinations, route map, or announcements). "
            "Accept 'nonstop' as 'direct'. Check page update dates if available to ensure the information aligns with Feb 2026."
        )
    )


async def build_intuit_presence_check(evaluator: Evaluator, parent, extr: CityOfficeExtraction) -> None:
    """
    Verify Intuit office presence and address with official Intuit source(s).
    """
    node = evaluator.add_leaf(
        id="Intuit_Office_Presence",
        desc="Verifies Intuit maintains an office location in the selected city and supports the address with an official Intuit source.",
        parent=parent,
        critical=True
    )
    city = extr.primary.city or "the selected city"
    address = extr.primary.intuit_office_address or "the stated address"
    claim = f"Intuit maintains an office at {address} in {city}."
    await evaluator.verify(
        claim=claim,
        node=node,
        sources=_safe_urls(extr.primary.intuit_office_official_urls),
        additional_instruction=(
            "Verify from official Intuit web pages (e.g., intuit.com, Intuit careers/locations/contact pages) that this address "
            "is an Intuit office in the stated city."
        )
    )


async def build_space_allocation_checks(evaluator: Evaluator, parent, extr: CityOfficeExtraction) -> None:
    """
    Space Allocation standard: decompose into (a) building fact, (b) standard reference, (c) logical calculation check.
    """
    node = evaluator.add_parallel(
        id="Space_Allocation_100to150_SqFtPerEmployee",
        desc="Verify the office meets 100–150 sq ft per employee (5,000–7,500 sq ft for 50 employees), with standard + office evidence.",
        parent=parent,
        critical=True
    )

    # (a) Building square footage supported
    leaf_a = evaluator.add_leaf(
        id="Space_Allocation_Office_SqFt_Supported",
        desc="The office square footage for the identified address is supported by cited source(s).",
        parent=node,
        critical=True
    )
    address = extr.primary.intuit_office_address or "the identified address"
    sqft = extr.space_allocation.office_sqft or ""
    claim_a = f"The office at {address} has approximately {sqft} of office space."
    await evaluator.verify(
        claim=claim_a,
        node=leaf_a,
        sources=_safe_urls(extr.space_allocation.office_sqft_urls),
        additional_instruction="Accept synonyms like rentable/leasable/RSF and minor unit formatting. Reasonable ±10% tolerance is acceptable."
    )

    # (b) Standard reference supported
    leaf_b = evaluator.add_leaf(
        id="Space_Allocation_Standard_Supported",
        desc="The 100–150 square feet per employee planning standard is supported by cited source(s).",
        parent=node,
        critical=True
    )
    claim_b = "A commonly cited 2026 office planning standard is 100–150 square feet per employee."
    await evaluator.verify(
        claim=claim_b,
        node=leaf_b,
        sources=_safe_urls(extr.space_allocation.standard_urls),
        additional_instruction="Verify that the page explicitly states or strongly implies a range that covers 100–150 sq ft per employee."
    )

    # (c) Logical calculation check (no external source required)
    leaf_c = evaluator.add_leaf(
        id="Space_Allocation_Calculation_50_Employees",
        desc="Given the stated square footage, the per-employee allocation for 50 employees falls within 100–150 sq ft (or total within 5,000–7,500 sq ft).",
        parent=node,
        critical=True
    )
    claim_c = (
        f"Given a total office area of '{sqft}' for 50 employees, the per-employee allocation is within 100–150 sq ft "
        f"and the total is within 5,000–7,500 sq ft."
    )
    await evaluator.verify(
        claim=claim_c,
        node=leaf_c,
        additional_instruction="Parse numbers from the area string (e.g., '6,200 sq ft'). If a range is given, consider if any value in the range satisfies the requirement."
    )


async def build_parking_checks(evaluator: Evaluator, parent, extr: CityOfficeExtraction) -> None:
    """
    Parking standard: (a) building fact, (b) standard reference, (c) logical compliance computation.
    """
    node = evaluator.add_parallel(
        id="Parking_3to5_Per_1000SqFt",
        desc="Verify parking availability meets 3–5 spaces per 1,000 sq ft using standard + office evidence.",
        parent=parent,
        critical=True
    )

    # (a) Building parking info supported
    leaf_a = evaluator.add_leaf(
        id="Parking_Office_Fact_Supported",
        desc="Parking availability (ratio and/or total spaces) for the identified office is supported by cited source(s).",
        parent=node,
        critical=True
    )
    ratio = extr.parking.parking_ratio_per_1000 or ""
    spaces = extr.parking.parking_total_spaces or ""
    address = extr.primary.intuit_office_address or "the identified address"
    if _is_nonempty(ratio):
        claim_a = f"The building/office serving {address} provides parking at a ratio of {ratio}."
    elif _is_nonempty(spaces):
        claim_a = f"The building/office serving {address} provides approximately {spaces} of parking."
    else:
        claim_a = f"The building/office serving {address} provides on-site parking meeting a stated ratio or total spaces."
    await evaluator.verify(
        claim=claim_a,
        node=leaf_a,
        sources=_safe_urls(extr.parking.parking_office_urls),
        additional_instruction="Look for a stated parking ratio (e.g., '4/1,000 sf') or a total space count. Minor formatting variants are acceptable."
    )

    # (b) Standard reference supported
    leaf_b = evaluator.add_leaf(
        id="Parking_Standard_Supported",
        desc="The standard of 3–5 parking spaces per 1,000 sq ft is supported by cited source(s).",
        parent=node,
        critical=True
    )
    claim_b = "A commonly cited commercial office parking standard is 3–5 spaces per 1,000 square feet of office space."
    await evaluator.verify(
        claim=claim_b,
        node=leaf_b,
        sources=_safe_urls(extr.parking.standard_urls),
        additional_instruction="Verify that the page states or clearly implies the 3–5 per 1,000 sq ft standard."
    )

    # (c) Logical compliance computation
    leaf_c = evaluator.add_leaf(
        id="Parking_Calculation_Compliance",
        desc="Given the office area and the stated parking ratio/total, parking availability meets the 3–5 per 1,000 sq ft standard.",
        parent=node,
        critical=True
    )
    sqft = extr.space_allocation.office_sqft or ""
    if _is_nonempty(ratio):
        calc_claim = (
            f"Given a parking ratio of '{ratio}' and a total office area of '{sqft}', the provided parking satisfies the "
            f"standard of 3–5 spaces per 1,000 sq ft."
        )
    elif _is_nonempty(spaces):
        calc_claim = (
            f"Given total parking of '{spaces}' and a total office area of '{sqft}', the provided parking satisfies the "
            f"standard of 3–5 spaces per 1,000 sq ft."
        )
    else:
        calc_claim = (
            f"Given the stated office area '{sqft}', the available parking satisfies the standard of 3–5 spaces per 1,000 sq ft."
        )
    await evaluator.verify(
        claim=calc_claim,
        node=leaf_c,
        additional_instruction="Compute whether the ratio or total implies compliance. If a range is provided, see if any value within the range satisfies 3–5 per 1,000."
    )


async def build_ada_checks(evaluator: Evaluator, parent, extr: CityOfficeExtraction) -> None:
    """
    ADA standard: (a) standard reference, (b) 36-inch routes compliance at office, (c) common paths compliance at office.
    """
    node = evaluator.add_parallel(
        id="ADA_Accessibility_36in_And_CommonPaths",
        desc="Verify ADA 36-inch clear width routes and accessible common-use circulation paths in 1,000+ sq ft work areas via standard + office evidence.",
        parent=parent,
        critical=True
    )

    # (a) ADA standard reference
    leaf_a = evaluator.add_leaf(
        id="ADA_Standard_Supported",
        desc="ADA requires 36-inch minimum clear width for accessible routes/aisles and accessible common-use circulation paths for work areas ≥1,000 sq ft.",
        parent=node,
        critical=True
    )
    claim_a = (
        "ADA requirements include (1) accessible routes/aisles with at least 36 inches of clear width, and "
        "(2) accessible common-use circulation paths in work areas of 1,000 square feet or larger."
    )
    await evaluator.verify(
        claim=claim_a,
        node=leaf_a,
        sources=_safe_urls(extr.ada.standard_urls),
        additional_instruction="Verify both elements are present: 36-inch clear width for accessible routes/aisles and accessible common-use circulation paths for work areas ≥1,000 sq ft."
    )

    # (b) Office compliance: 36-inch routes/aisles
    leaf_b = evaluator.add_leaf(
        id="ADA_36in_Compliance_Supported",
        desc="The office provides accessible routes/aisles with a minimum 36-inch clear width, supported by cited source(s).",
        parent=node,
        critical=True
    )
    addr = extr.primary.intuit_office_address or "the identified address"
    stmt36 = extr.ada.ada_36in_statement or "accessible routes/aisles with at least 36 inches of clear width"
    claim_b = f"The office at {addr} provides {stmt36}."
    await evaluator.verify(
        claim=claim_b,
        node=leaf_b,
        sources=_safe_urls(extr.ada.office_ada_urls),
        additional_instruction="Look for statements about ADA-compliant accessible routes/aisles and minimum clear widths."
    )

    # (c) Office compliance: accessible common-use circulation paths (in >1,000 sq ft work areas)
    leaf_c = evaluator.add_leaf(
        id="ADA_Common_Paths_Compliance_Supported",
        desc="The office provides accessible common-use circulation paths in work areas of 1,000+ sq ft, supported by cited source(s).",
        parent=node,
        critical=True
    )
    stmtcp = extr.ada.ada_common_paths_statement or "accessible common-use circulation paths in work areas of 1,000+ sq ft"
    claim_c = f"The office at {addr} provides {stmtcp}."
    await evaluator.verify(
        claim=claim_c,
        node=leaf_c,
        sources=_safe_urls(extr.ada.office_ada_urls),
        additional_instruction="Look for statements about ADA-compliant circulation paths in common/work areas meeting the ≥1,000 sq ft criteria."
    )


async def build_ceiling_checks(evaluator: Evaluator, parent, extr: CityOfficeExtraction) -> None:
    """
    Ceiling height standard: (a) building fact, (b) standard reference, (c) logical compliance check.
    """
    node = evaluator.add_parallel(
        id="Ceiling_Height_7_5to8ft_Minimum",
        desc="Verify ceiling height meets ≥7.5–8 ft minimum by combining standard + office evidence.",
        parent=parent,
        critical=True
    )

    # (a) Building ceiling height supported
    leaf_a = evaluator.add_leaf(
        id="Ceiling_Height_Office_Fact_Supported",
        desc="Office/building ceiling height for the identified address is supported by cited source(s).",
        parent=node,
        critical=True
    )
    address = extr.primary.intuit_office_address or "the identified address"
    ch = extr.ceiling.ceiling_height or ""
    claim_a = f"The office at {address} has ceiling height '{ch}'."
    await evaluator.verify(
        claim=claim_a,
        node=leaf_a,
        sources=_safe_urls(extr.ceiling.office_ceiling_urls),
        additional_instruction="Accept units like ft or ' (feet), and minor phrasing variants. If a range is provided, consider the minimum stated height."
    )

    # (b) Standard reference supported
    leaf_b = evaluator.add_leaf(
        id="Ceiling_Height_Standard_Supported",
        desc="Minimum office ceiling height of 7.5–8 feet is supported by cited source(s).",
        parent=node,
        critical=True
    )
    claim_b = "A minimum office ceiling height requirement or guideline is around 7.5–8 feet."
    await evaluator.verify(
        claim=claim_b,
        node=leaf_b,
        sources=_safe_urls(extr.ceiling.standard_urls),
        additional_instruction="Verify that the page states or implies minimum office ceiling heights around 7.5–8 ft."
    )

    # (c) Logical compliance check
    leaf_c = evaluator.add_leaf(
        id="Ceiling_Height_Compliance_Check",
        desc="Given the stated ceiling height, the office meets the minimum of 7.5–8 ft.",
        parent=node,
        critical=True
    )
    claim_c = f"Given the ceiling height '{ch}', the office meets the minimum requirement of 7.5–8 feet."
    await evaluator.verify(
        claim=claim_c,
        node=leaf_c,
        additional_instruction="Parse numbers from the height string and check if the minimum stated height is ≥7.5 ft."
    )


async def build_building_class_checks(evaluator: Evaluator, parent, extr: CityOfficeExtraction) -> None:
    """
    Building Class standard: (a) building classification supported, (b) definition/criteria reference supported.
    """
    node = evaluator.add_parallel(
        id="Building_Classification_ClassA",
        desc="State whether the building is Class A and support with definition and building evidence.",
        parent=parent,
        critical=True
    )

    # (a) Building classification supported
    leaf_a = evaluator.add_leaf(
        id="Building_Classification_Office_Fact_Supported",
        desc="The identified building's classification (e.g., Class A) is supported by cited source(s).",
        parent=node,
        critical=True
    )
    bcls = extr.building_class.building_class or "a stated class"
    address = extr.primary.intuit_office_address or "the identified address"
    claim_a = f"The building at {address} is classified as {bcls}."
    await evaluator.verify(
        claim=claim_a,
        node=leaf_a,
        sources=_safe_urls(extr.building_class.building_class_urls),
        additional_instruction="Look for explicit mention of 'Class A', 'Class B', etc., on property listings, brokerage pages, or official building materials."
    )

    # (b) Class definition/criteria supported
    leaf_b = evaluator.add_leaf(
        id="Building_Classification_Definition_Supported",
        desc="The definition/criteria for Class A office is supported by cited source(s).",
        parent=node,
        critical=True
    )
    claim_b = "Class A office buildings are top-tier properties with professional management, prominent location, and high-quality systems (e.g., HVAC, lighting)."
    await evaluator.verify(
        claim=claim_b,
        node=leaf_b,
        sources=_safe_urls(extr.building_class.definition_urls),
        additional_instruction="Verify that the page defines 'Class A' with characteristics like top-tier quality, professional management, prime locations, and superior building systems."
    )


async def build_lease_terms_checks(evaluator: Evaluator, parent, extr: CityOfficeExtraction) -> None:
    """
    Lease terms standard: (a) typical range supported, (b) office lease terms supported, (c) logical alignment check.
    """
    node = evaluator.add_parallel(
        id="Lease_Terms_Align_3to10_Years",
        desc="Note whether lease terms align with the typical 3–10 years (5–10 years for larger spaces), supported by sources.",
        parent=parent,
        critical=True
    )

    # (a) Typical range standard supported
    leaf_a = evaluator.add_leaf(
        id="Lease_Terms_Standard_Supported",
        desc="Typical office lease terms of 3–10 years (5–10 for larger spaces) are supported by cited source(s).",
        parent=node,
        critical=True
    )
    claim_a = "Typical commercial office lease terms fall within 3–10 years, with 5–10 years common for larger spaces."
    await evaluator.verify(
        claim=claim_a,
        node=leaf_a,
        sources=_safe_urls(extr.lease_terms.standard_urls),
        additional_instruction="Verify that the page describes typical lease terms for office space as 3–10 years, and notes 5–10 years for larger spaces when applicable."
    )

    # (b) Office lease terms supported
    leaf_b = evaluator.add_leaf(
        id="Lease_Terms_Office_Fact_Supported",
        desc="The identified office's stated lease term(s) are supported by cited source(s).",
        parent=node,
        critical=True
    )
    lterm = extr.lease_terms.lease_term_years or ""
    address = extr.primary.intuit_office_address or "the identified address"
    claim_b = f"The office at {address} has lease term(s) stated as '{lterm}'."
    await evaluator.verify(
        claim=claim_b,
        node=leaf_b,
        sources=_safe_urls(extr.lease_terms.lease_terms_urls),
        additional_instruction="Look for stated lease term ranges/years on property listings or broker pages. Accept reasonable phrasing variants."
    )

    # (c) Logical alignment check
    leaf_c = evaluator.add_leaf(
        id="Lease_Terms_Alignment_Check",
        desc="Given the stated lease term(s), they align with the typical 3–10 year range (5–10 years for larger spaces).",
        parent=node,
        critical=True
    )
    claim_c = f"Given the stated lease term(s) '{lterm}', they align with the typical 3–10 year range for commercial office leases."
    await evaluator.verify(
        claim=claim_c,
        node=leaf_c,
        additional_instruction="Parse numeric years from the lease term string and check whether they fall within 3–10 years. If a range is given, consider if any value within the range satisfies alignment."
    )


async def build_standards_section(evaluator: Evaluator, parent, extr: CityOfficeExtraction) -> None:
    """
    Build the overall Office Standards section containing all six standard areas.
    """
    node = evaluator.add_parallel(
        id="Office_Standards_For_50_Employees_Verification",
        desc="Verify the identified Intuit office meets each specified 2026 commercial office standard for accommodating a team of 50 employees, with supporting URLs.",
        parent=parent,
        critical=True
    )

    await build_space_allocation_checks(evaluator, node, extr)
    await build_parking_checks(evaluator, node, extr)
    await build_ada_checks(evaluator, node, extr)
    await build_ceiling_checks(evaluator, node, extr)
    await build_building_class_checks(evaluator, node, extr)
    await build_lease_terms_checks(evaluator, node, extr)


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
    Evaluate an answer for the Breeze + Intuit 2026 office standards task.
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

    # Extract structured information from the answer
    extraction = await evaluator.extract(
        prompt=prompt_extract_city_office(),
        template_class=CityOfficeExtraction,
        extraction_name="parsed_city_office_and_standards"
    )

    # Build the main sequential critical verification flow under root
    main = evaluator.add_sequential(
        id="City_and_Intuit_Office_Standards_Verification",
        desc="Identify one US city served by Breeze direct flights (as of Feb 2026) where Intuit has an office, provide the city and Intuit office address, and verify the office meets all listed 2026 standards with supporting URLs.",
        parent=root,
        critical=True
    )

    # 1) Identify city and office address (existence checks)
    await build_identification_section(evaluator, main, extraction)

    # 2) Breeze direct service check
    await build_breeze_service_check(evaluator, main, extraction)

    # 3) Intuit office presence check
    await build_intuit_presence_check(evaluator, main, extraction)

    # 4) Office standards checks (parallel critical group)
    await build_standards_section(evaluator, main, extraction)

    return evaluator.get_summary()