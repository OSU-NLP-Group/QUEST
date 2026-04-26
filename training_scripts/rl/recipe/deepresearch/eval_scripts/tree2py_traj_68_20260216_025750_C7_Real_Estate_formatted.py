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
TASK_ID = "fl_class_a_office_verification"
TASK_DESCRIPTION = (
    "Identify one Class A commercial office building currently available for investment or lease in the Tampa, "
    "Orlando, or Miami metropolitan area in Florida that meets institutional investment standards. The building must "
    "be at least 50,000 square feet in total size, have a minimum of 3 stories, and have available leasable space or be "
    "available for purchase. Provide the building's name and complete street address.\n\n"
    "Additionally, provide the following information about the identified property:\n"
    "- Confirmation of its Class A classification or characteristics\n"
    "- Total building square footage\n"
    "- Number of stories\n"
    "- Available space details (amount available for lease or purchase status)\n"
    "- At least one credible reference URL from commercial real estate listing sites or professional sources\n\n"
    "If available, also include:\n"
    "- Parking information (number of spaces or parking ratio)\n"
    "- Year of construction or most recent major renovation\n"
    "- Energy efficiency certifications (ENERGY STAR or LEED) or green building features\n"
    "- Property management company name\n"
    "- Information about major tenants or tenant quality\n"
    "- Current asking lease rates with market context\n"
    "- ADA accessibility features (such as elevators, accessible entrances, and facilities)"
)


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class PropertyExtraction(BaseModel):
    # Core identification
    building_name: Optional[str] = None
    street_address: Optional[str] = None
    city: Optional[str] = None
    state: Optional[str] = None
    zip_code: Optional[str] = None
    metro_area: Optional[str] = None  # e.g., "Tampa", "Orlando", "Miami", or submarket within

    # Mandatory constraints & disclosures
    total_sqft: Optional[str] = None
    stories: Optional[str] = None
    class_a_explicit: Optional[bool] = None  # True if explicitly labeled Class A in the answer
    class_a_characteristics: List[str] = Field(default_factory=list)
    availability_status: Optional[str] = None  # e.g., "for lease", "for sale", "both"
    available_space_details: Optional[str] = None  # e.g., "20,000 SF", "Suite 300 10k SF", etc.
    reference_urls: List[str] = Field(default_factory=list)

    # Additional/optional info
    parking_info: Optional[str] = None
    parking_ratio: Optional[str] = None
    parking_spaces: Optional[str] = None
    management_company: Optional[str] = None

    construction_year: Optional[str] = None
    renovation_year: Optional[str] = None
    energy_certifications: List[str] = Field(default_factory=list)  # e.g., ["LEED Gold", "ENERGY STAR"]
    green_features: List[str] = Field(default_factory=list)

    tenants_info: Optional[str] = None
    asking_lease_rates: Optional[str] = None
    market_context: Optional[str] = None

    elevator_info: Optional[str] = None
    ada_features: List[str] = Field(default_factory=list)

    amenities: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_property() -> str:
    return """
    Extract details for exactly ONE primary property described in the answer that is a Class A commercial office building
    in Florida within the Tampa, Orlando, or Miami metropolitan area, and currently available for lease or purchase.

    If multiple properties are mentioned, choose the first one that appears to meet the constraints and treat it as the primary property.

    Return a JSON object with the following fields (use null when missing; for lists, use empty arrays):

    Core identification:
    - building_name: string
    - street_address: string
    - city: string
    - state: string (prefer "FL" or "Florida")
    - zip_code: string
    - metro_area: string (e.g., "Tampa", "Orlando", "Miami", or a common submarket within them like "Brickell", "Downtown Tampa")

    Mandatory constraints & disclosures:
    - total_sqft: string (e.g., "250,000 SF", "approx. 65,000")
    - stories: string (e.g., "10", "12 floors")
    - class_a_explicit: boolean (true if the answer explicitly states the building is Class A; false otherwise)
    - class_a_characteristics: array of strings (list features indicating Class A quality: prime location, high-quality construction, professional management, top-tier amenities, modern building systems, structured parking, etc.)
    - availability_status: string (e.g., "for lease", "for sale", "both", "available")
    - available_space_details: string (e.g., "20,000 SF available", "Suite 300: 10,500 SF", "asking $ X / SF", "for purchase")
    - reference_urls: array of strings (extract ALL URLs mentioned; include listing pages, broker sites, or credible sources)

    Optional details (include if present):
    - parking_info: string (e.g., "structured garage", "surface parking", "attached garage")
    - parking_ratio: string (e.g., "3.0/1,000 SF")
    - parking_spaces: string (e.g., "500 spaces")
    - management_company: string (property management company or owner/manager)

    - construction_year: string
    - renovation_year: string
    - energy_certifications: array of strings (e.g., ["LEED Gold", "ENERGY STAR"])
    - green_features: array of strings (e.g., "efficient HVAC", "solar panels")

    - tenants_info: string (major tenants or tenant quality)
    - asking_lease_rates: string (e.g., "$45/SF full service", "NNN rates")
    - market_context: string (comparisons or positioning vs market)

    - elevator_info: string (e.g., "4 elevators", "elevator access")
    - ada_features: array of strings (e.g., "accessible entrance", "accessible restrooms", "ramps", "ADA compliant")

    - amenities: array of strings (e.g., "fitness center", "conference center", "on-site cafe", "security", "covered parking")
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def format_address(info: PropertyExtraction) -> str:
    parts = []
    if info.street_address:
        parts.append(info.street_address.strip())
    city_state_zip = " ".join(
        p for p in [
            (info.city or "").strip(),
            (info.state or "").strip(),
            (info.zip_code or "").strip()
        ] if p
    ).strip()
    if city_state_zip:
        parts.append(city_state_zip)
    return ", ".join(parts)


def credible_source_guidance() -> str:
    return (
        "Treat commercial real estate listing sites or professional sources as credible, including: CBRE, JLL, Cushman & "
        "Wakefield, Colliers, Marcus & Millichap, Newmark, Transwestern, Hines, Skanska, LoopNet, CoStar, NAIOP, Yardi, "
        "Reonomy, CREXi, and official property/owner/broker websites. Verify that the page contains property details "
        "(name and/or address) and is clearly about the property."
    )


# --------------------------------------------------------------------------- #
# Verification subtrees                                                       #
# --------------------------------------------------------------------------- #
async def verify_property_identification(evaluator: Evaluator, parent_node, info: PropertyExtraction) -> None:
    pid_node = evaluator.add_parallel(
        id="Property_Identification",
        desc="The property is uniquely identified with name/address and correct metro location.",
        parent=parent_node,
        critical=True
    )

    # Building name provided (existence check)
    evaluator.add_custom_node(
        result=bool(info.building_name and info.building_name.strip()),
        id="Building_Name_Provided",
        desc="Building name is provided.",
        parent=pid_node,
        critical=True
    )

    # Complete street address provided (street + city + state)
    address_complete = bool(info.street_address and info.city and info.state)
    evaluator.add_custom_node(
        result=address_complete,
        id="Complete_Street_Address_Provided",
        desc="Street address + city + state are provided (ZIP optional).",
        parent=pid_node,
        critical=True
    )

    # Metro area constraint satisfied (verify by URLs)
    metro_leaf = evaluator.add_leaf(
        id="Metro_Area_Constraint_Satisfied",
        desc="Property is located in the Tampa, Orlando, or Miami metropolitan area in Florida.",
        parent=pid_node,
        critical=True
    )
    address_line = format_address(info)
    metro_claim = (
        f"The property '{(info.building_name or '').strip()}' at '{address_line}' is in Florida and is within either "
        f"the Tampa Bay (Tampa/St. Petersburg/Clearwater), Orlando-Kissimmee-Sanford, or Miami metro (including submarkets "
        f"like Brickell, Downtown Miami, Doral, Coral Gables, Miami Beach)."
    )
    await evaluator.verify(
        claim=metro_claim,
        node=metro_leaf,
        sources=info.reference_urls,
        additional_instruction=(
            "Confirm the city/state on the page. Accept common submarket names mapping to these metros. "
            "Florida must be the state. If the page clearly shows the address or city within these metros, pass."
        )
    )


async def verify_class_a_requirement(evaluator: Evaluator, parent_node, info: PropertyExtraction) -> None:
    classa_node = evaluator.add_sequential(
        id="Class_A_Requirement",
        desc="Response confirms the building is Class A by explicit labeling or at least TWO Class A characteristics.",
        parent=parent_node,
        critical=True
    )

    # Existence: either explicit or at least two characteristics listed
    has_classa_basis = bool(info.class_a_explicit) or (len(info.class_a_characteristics) >= 2)
    evaluator.add_custom_node(
        result=has_classa_basis,
        id="Class_A_Basis_Provided",
        desc="Class A is explicitly stated OR at least two Class A characteristics are provided.",
        parent=classa_node,
        critical=True
    )

    # Verification with sources
    classa_leaf = evaluator.add_leaf(
        id="Class_A_Supported_By_Sources",
        desc="Class A classification or features are supported by cited sources.",
        parent=classa_node,
        critical=True
    )

    characteristics_text = ", ".join(info.class_a_characteristics) if info.class_a_characteristics else "none listed"
    if info.class_a_explicit:
        classa_claim = "The property is explicitly labeled as 'Class A' office."
    else:
        classa_claim = (
            f"The property exhibits at least two Class A characteristics (e.g., prime location, high-quality construction,"
            f" professional management, top-tier amenities, modern building systems). The answer listed: {characteristics_text}."
        )

    await evaluator.verify(
        claim=classa_claim,
        node=classa_leaf,
        sources=info.reference_urls,
        additional_instruction=(
            "Pass if the page either explicitly says 'Class A' or clearly demonstrates multiple high-end characteristics "
            "expected of Class A office (e.g., premier location, modern building systems, structured parking, professional "
            "management, top-tier amenities like fitness/conference, recent major renovation)."
        )
    )


async def verify_minimum_size(evaluator: Evaluator, parent_node, info: PropertyExtraction) -> None:
    size_node = evaluator.add_sequential(
        id="Minimum_Size_And_Disclosed_SF",
        desc="Total building square footage is stated and is at least 50,000 sq ft.",
        parent=parent_node,
        critical=True
    )

    # Existence check
    evaluator.add_custom_node(
        result=bool(info.total_sqft and info.total_sqft.strip()),
        id="Total_SF_Stated",
        desc="Total building square footage is stated.",
        parent=size_node,
        critical=True
    )

    # Verification by source
    sf_leaf = evaluator.add_leaf(
        id="Total_SF_At_Least_50k_Supported",
        desc="Total building square footage is at least 50,000 sq ft, supported by sources.",
        parent=size_node,
        critical=True
    )

    sf_claim = (
        f"The total building size is '{(info.total_sqft or '').strip()}', and the page indicates the building is at least 50,000 square feet."
    )
    await evaluator.verify(
        claim=sf_claim,
        node=sf_leaf,
        sources=info.reference_urls,
        additional_instruction="Check the building's total size on the page; pass if it is >= 50,000 SF."
    )


async def verify_minimum_stories(evaluator: Evaluator, parent_node, info: PropertyExtraction) -> None:
    stories_node = evaluator.add_sequential(
        id="Minimum_Stories_And_Disclosed_Stories",
        desc="Number of stories is stated and is at least 3.",
        parent=parent_node,
        critical=True
    )

    # Existence
    evaluator.add_custom_node(
        result=bool(info.stories and info.stories.strip()),
        id="Stories_Stated",
        desc="Number of stories is stated.",
        parent=stories_node,
        critical=True
    )

    # Verification
    stories_leaf = evaluator.add_leaf(
        id="Stories_At_Least_3_Supported",
        desc="Number of stories is at least 3, supported by sources.",
        parent=stories_node,
        critical=True
    )

    stories_claim = (
        f"The building has '{(info.stories or '').strip()}' stories, and the page indicates the building has at least 3 stories."
    )
    await evaluator.verify(
        claim=stories_claim,
        node=stories_leaf,
        sources=info.reference_urls,
        additional_instruction="Confirm that the building has three or more stories based on the page."
    )


async def verify_availability(evaluator: Evaluator, parent_node, info: PropertyExtraction) -> None:
    avail_node = evaluator.add_sequential(
        id="Availability_Requirement",
        desc="The property is available for lease and/or purchase and provides an availability detail.",
        parent=parent_node,
        critical=True
    )

    # Existence
    has_avail_detail = bool((info.availability_status and info.availability_status.strip()) or
                            (info.available_space_details and info.available_space_details.strip()))
    evaluator.add_custom_node(
        result=has_avail_detail,
        id="Availability_Detail_Provided",
        desc="Availability status/details are provided.",
        parent=avail_node,
        critical=True
    )

    # Verification
    avail_leaf = evaluator.add_leaf(
        id="Availability_Supported_By_Sources",
        desc="Availability for lease and/or purchase is supported by sources.",
        parent=avail_node,
        critical=True
    )

    avail_text = (info.available_space_details or info.availability_status or "").strip()
    avail_claim = (
        f"The property is currently available for lease and/or purchase. Availability details: '{avail_text}'."
    )
    await evaluator.verify(
        claim=avail_claim,
        node=avail_leaf,
        sources=info.reference_urls,
        additional_instruction="Confirm the listing shows available space (SF or suites) or a clear for-sale status."
    )


async def verify_sources(evaluator: Evaluator, parent_node, info: PropertyExtraction) -> None:
    src_node = evaluator.add_parallel(
        id="Source_Documentation",
        desc="At least one credible reference URL from a CRE listing site or professional source is provided.",
        parent=parent_node,
        critical=True
    )

    evaluator.add_custom_node(
        result=bool(info.reference_urls and len(info.reference_urls) > 0),
        id="Source_URLs_Provided",
        desc="At least one reference URL is provided.",
        parent=src_node,
        critical=True
    )

    contain_leaf = evaluator.add_leaf(
        id="Source_Contains_Property_Info",
        desc="A provided URL contains property name and/or address indicating the page is about this property.",
        parent=src_node,
        critical=True
    )

    name_part = (info.building_name or "").strip()
    addr_part = format_address(info)
    contain_claim = (
        f"The provided source page contains the property name '{name_part}' or the address '{addr_part}', indicating it is the property's detail/listing page."
    )
    await evaluator.verify(
        claim=contain_claim,
        node=contain_leaf,
        sources=info.reference_urls,
        additional_instruction=credible_source_guidance()
    )


async def verify_parking(evaluator: Evaluator, parent_node, info: PropertyExtraction) -> None:
    park_node = evaluator.add_sequential(
        id="Parking_Constraint",
        desc="Response provides parking information (spaces, ratio, or clear statement of on-site/attached/structured parking).",
        parent=parent_node,
        critical=True
    )

    has_parking_info = bool((info.parking_info and info.parking_info.strip()) or
                            (info.parking_ratio and info.parking_ratio.strip()) or
                            (info.parking_spaces and info.parking_spaces.strip()))
    evaluator.add_custom_node(
        result=has_parking_info,
        id="Parking_Info_Provided",
        desc="Parking information is provided.",
        parent=park_node,
        critical=True
    )

    park_leaf = evaluator.add_leaf(
        id="Parking_Info_Supported_By_Sources",
        desc="Parking information is supported by sources.",
        parent=park_node,
        critical=True
    )

    parking_text = ", ".join(
        [t for t in [
            (info.parking_info or "").strip(),
            (info.parking_ratio or "").strip(),
            (info.parking_spaces or "").strip()
        ] if t]
    )
    parking_claim = f"Parking details for this property include: {parking_text}."
    await evaluator.verify(
        claim=parking_claim,
        node=park_leaf,
        sources=info.reference_urls,
        additional_instruction="Confirm the page mentions structured garage, on-site parking, parking ratio or number of spaces."
    )


async def verify_management(evaluator: Evaluator, parent_node, info: PropertyExtraction) -> None:
    mgmt_node = evaluator.add_sequential(
        id="Professional_Management_Constraint",
        desc="Response names the property management company or owner/manager entity.",
        parent=parent_node,
        critical=True
    )

    evaluator.add_custom_node(
        result=bool(info.management_company and info.management_company.strip()),
        id="Management_Company_Provided",
        desc="Property management company or owner/manager is provided.",
        parent=mgmt_node,
        critical=True
    )

    mgmt_leaf = evaluator.add_leaf(
        id="Management_Company_Supported_By_Sources",
        desc="Management company / manager is supported by sources.",
        parent=mgmt_node,
        critical=True
    )

    mgmt_claim = f"The property is managed by '{(info.management_company or '').strip()}' (or named owner/manager)."
    await evaluator.verify(
        claim=mgmt_claim,
        node=mgmt_leaf,
        sources=info.reference_urls,
        additional_instruction="Pass if the page states a management company or owner/manager entity."
    )


async def verify_ada(evaluator: Evaluator, parent_node, info: PropertyExtraction) -> None:
    ada_node = evaluator.add_sequential(
        id="ADA_Accessibility_Constraint",
        desc="Response describes ADA-related accessibility including elevators AND at least one additional accessibility feature.",
        parent=parent_node,
        critical=True
    )

    has_ada_basis = bool(info.elevator_info and info.elevator_info.strip()) and (len(info.ada_features) >= 1)
    evaluator.add_custom_node(
        result=has_ada_basis,
        id="ADA_Basis_Provided",
        desc="Elevators and at least one ADA feature are provided.",
        parent=ada_node,
        critical=True
    )

    ada_leaf = evaluator.add_leaf(
        id="ADA_Features_Supported_By_Sources",
        desc="ADA accessibility features (elevator + additional feature) are supported by sources.",
        parent=ada_node,
        critical=True
    )

    ada_text = ", ".join(info.ada_features) if info.ada_features else "none listed"
    ada_claim = (
        f"The property has elevators ('{(info.elevator_info or '').strip()}') and ADA features ({ada_text}), such as accessible entrances, restrooms, ramps or accessible paths."
    )
    await evaluator.verify(
        claim=ada_claim,
        node=ada_leaf,
        sources=info.reference_urls,
        additional_instruction="Confirm the page indicates elevator access and at least one ADA-related feature. Accept 'ADA compliant' statements."
    )


# Optional details subtree
async def verify_optional_details(evaluator: Evaluator, parent_node, info: PropertyExtraction) -> None:
    opt_root = evaluator.add_parallel(
        id="Optional_Details",
        desc="Optional property details (non-critical).",
        parent=parent_node,
        critical=False
    )

    # Construction vintage
    cons_node = evaluator.add_sequential(
        id="Construction_Vintage_Optional",
        desc="Year of construction or most recent major renovation (if available).",
        parent=opt_root,
        critical=False
    )
    has_cons = bool((info.construction_year and info.construction_year.strip()) or
                    (info.renovation_year and info.renovation_year.strip()))
    evaluator.add_custom_node(
        result=has_cons,
        id="Construction_Info_Provided",
        desc="Construction/renovation year provided.",
        parent=cons_node,
        critical=False
    )
    cons_leaf = evaluator.add_leaf(
        id="Construction_Info_Supported",
        desc="Construction/renovation year supported by sources.",
        parent=cons_node,
        critical=False
    )
    cons_text = ", ".join([t for t in [
        (info.construction_year or "").strip(),
        (info.renovation_year or "").strip()
    ] if t])
    await evaluator.verify(
        claim=f"Construction/renovation vintage: {cons_text}.",
        node=cons_leaf,
        sources=info.reference_urls,
        additional_instruction="Verify year(s) mentioned on the page."
    )

    # Green building
    green_node = evaluator.add_sequential(
        id="Green_Building_Optional",
        desc="Energy certifications or green features (if available).",
        parent=opt_root,
        critical=False
    )
    has_green = bool(info.energy_certifications or info.green_features)
    evaluator.add_custom_node(
        result=has_green,
        id="Green_Info_Provided",
        desc="Energy certifications or green features provided.",
        parent=green_node,
        critical=False
    )
    green_leaf = evaluator.add_leaf(
        id="Green_Info_Supported",
        desc="Energy certifications or green features supported by sources.",
        parent=green_node,
        critical=False
    )
    green_text = ", ".join(info.energy_certifications + info.green_features)
    await evaluator.verify(
        claim=f"Green/certification info: {green_text}.",
        node=green_leaf,
        sources=info.reference_urls,
        additional_instruction="Confirm LEED/ENERGY STAR or listed sustainability features on the page."
    )

    # Tenant information
    tenant_node = evaluator.add_sequential(
        id="Tenant_Information_Optional",
        desc="Information about major tenants or tenant quality (if available).",
        parent=opt_root,
        critical=False
    )
    has_tenants = bool(info.tenants_info and info.tenants_info.strip())
    evaluator.add_custom_node(
        result=has_tenants,
        id="Tenant_Info_Provided",
        desc="Tenant information provided.",
        parent=tenant_node,
        critical=False
    )
    tenant_leaf = evaluator.add_leaf(
        id="Tenant_Info_Supported",
        desc="Tenant information supported by sources.",
        parent=tenant_node,
        critical=False
    )
    await evaluator.verify(
        claim=f"Tenant information: {(info.tenants_info or '').strip()}",
        node=tenant_leaf,
        sources=info.reference_urls,
        additional_instruction="Pass if the page names notable tenants or describes tenant quality."
    )

    # Market rate positioning
    rate_node = evaluator.add_sequential(
        id="Market_Rate_Positioning_Optional",
        desc="Asking lease rates with market context (if available).",
        parent=opt_root,
        critical=False
    )
    has_rates = bool((info.asking_lease_rates and info.asking_lease_rates.strip()) or
                     (info.market_context and info.market_context.strip()))
    evaluator.add_custom_node(
        result=has_rates,
        id="Rates_Info_Provided",
        desc="Lease rates and/or market context provided.",
        parent=rate_node,
        critical=False
    )
    rate_leaf = evaluator.add_leaf(
        id="Rates_Info_Supported",
        desc="Lease rates/market context supported by sources.",
        parent=rate_node,
        critical=False
    )
    rate_text = ", ".join([t for t in [
        (info.asking_lease_rates or "").strip(),
        (info.market_context or "").strip()
    ] if t])
    await evaluator.verify(
        claim=f"Lease rates / market context: {rate_text}.",
        node=rate_leaf,
        sources=info.reference_urls,
        additional_instruction="Confirm rate figures or market positioning statements on the page."
    )

    # Building amenities
    amen_node = evaluator.add_sequential(
        id="Building_Amenities_Optional",
        desc="Building amenities and features described (if available).",
        parent=opt_root,
        critical=False
    )
    has_amen = bool(info.amenities)
    evaluator.add_custom_node(
        result=has_amen,
        id="Amenities_Info_Provided",
        desc="Amenities/features provided.",
        parent=amen_node,
        critical=False
    )
    amen_leaf = evaluator.add_leaf(
        id="Amenities_Info_Supported",
        desc="Amenities/features supported by sources.",
        parent=amen_node,
        critical=False
    )
    amenities_text = ", ".join(info.amenities)
    await evaluator.verify(
        claim=f"Amenities/features include: {amenities_text}.",
        node=amen_leaf,
        sources=info.reference_urls,
        additional_instruction="Verify listed amenities on the page (e.g., fitness center, conference rooms, cafe, security, structured parking)."
    )


# --------------------------------------------------------------------------- #
# Main evaluation function                                                    #
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
    Entry point for evaluating the agent's answer against the Class A office property rubric.
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

    # Extract structured property info
    prop_info = await evaluator.extract(
        prompt=prompt_extract_property(),
        template_class=PropertyExtraction,
        extraction_name="property_extraction"
    )

    # Build top-level aggregation: split mandatory vs optional to satisfy critical-child constraints
    top_node = evaluator.add_parallel(
        id="Investment_Grade_Office_Property_Analysis",
        desc="Verify one qualifying Class A office in Tampa/Orlando/Miami meets all mandatory constraints with credible sources; optional enrichments allowed.",
        parent=root,
        critical=False  # Non-critical wrapper to allow optional subtree
    )

    mandatory_node = evaluator.add_parallel(
        id="Mandatory_Requirements",
        desc="Mandatory requirements (critical): Identification, Class A, size, stories, availability, source documentation, parking, management, ADA.",
        parent=top_node,
        critical=True
    )

    # Mandatory verifications
    await verify_property_identification(evaluator, mandatory_node, prop_info)
    await verify_class_a_requirement(evaluator, mandatory_node, prop_info)
    await verify_minimum_size(evaluator, mandatory_node, prop_info)
    await verify_minimum_stories(evaluator, mandatory_node, prop_info)
    await verify_availability(evaluator, mandatory_node, prop_info)
    await verify_sources(evaluator, mandatory_node, prop_info)
    await verify_parking(evaluator, mandatory_node, prop_info)
    await verify_management(evaluator, mandatory_node, prop_info)
    await verify_ada(evaluator, mandatory_node, prop_info)

    # Optional verifications
    await verify_optional_details(evaluator, top_node, prop_info)

    # Return summary
    return evaluator.get_summary()