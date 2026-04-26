import asyncio
import logging
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


# --------------------------------------------------------------------------- #
# Task constants                                                              #
# --------------------------------------------------------------------------- #
TASK_ID = "mixed_use_development_project_compliance"
TASK_DESCRIPTION = (
    "Identify a mixed-use development project in the United States that meets all of the following requirements:\n"
    "1. The project must contain both commercial and residential components on the same property\n"
    "2. It must have achieved or be pursuing LEED Gold certification (60-79 points) or higher\n"
    "3. It must comply with 2025 Building Energy Efficiency Standards (effective January 1, 2026)\n"
    "4. The office space component must allocate between 150-200 square feet per employee\n"
    "5. The commercial office portion must provide a parking ratio between 4-6 spaces per 1,000 square feet of leasable area\n"
    "6. The residential component must comply with multifamily design and construction standards for 3+ dwelling units\n"
    "7. It must meet ADA Standards for Accessible Design\n"
    "8. It must have proper IBC occupancy classification for mixed-use buildings\n"
    "9. It must comply with applicable fire safety codes (NFPA standards)\n"
    "10. It must demonstrate positive Net Operating Income (NOI)\n"
    "11. It must have professional property management services for the residential component\n"
    "12. Provide the project name, complete address, total square footage (with commercial/residential breakdown), and "
    "at least three reference URLs from reputable sources documenting these requirements."
)


# --------------------------------------------------------------------------- #
# Extraction models                                                           #
# --------------------------------------------------------------------------- #
class MixedUseProjectExtraction(BaseModel):
    # Identification
    project_name: Optional[str] = None
    full_address: Optional[str] = None
    city: Optional[str] = None
    state: Optional[str] = None

    # Area and composition
    total_sqft: Optional[str] = None
    commercial_sqft: Optional[str] = None
    residential_sqft: Optional[str] = None

    # Core requirements and compliance statements
    mixed_use_composition_claim: Optional[str] = None
    leed_status: Optional[str] = None  # e.g., "LEED Gold", "LEED Platinum", "pursuing LEED Gold"
    energy_efficiency_compliance: Optional[str] = None  # statement about 2025 standards or equivalent
    office_sqft_per_employee: Optional[str] = None  # e.g., "175"
    parking_ratio_office_per_1000sf: Optional[str] = None  # e.g., "5/1000 sf"
    multifamily_units: Optional[str] = None  # e.g., "120"
    multifamily_standards_compliance: Optional[str] = None
    ada_compliance: Optional[str] = None
    ibc_occupancy_classification: Optional[str] = None  # e.g., "B/R-2 mixed-occupancy"
    fire_safety_compliance: Optional[str] = None  # e.g., "NFPA 13 sprinklered"
    noi_statement: Optional[str] = None  # e.g., "Projected NOI is positive"
    property_management_firm: Optional[str] = None  # e.g., "Greystar", "Managed by ..."

    # Additional context (non-critical)
    timeline: Optional[str] = None
    sustainable_features: List[str] = Field(default_factory=list)
    zoning_permits: Optional[str] = None
    residential_unit_density: Optional[str] = None
    tenant_mix: Optional[str] = None
    financing_structure: Optional[str] = None

    # References
    primary_reference_url: Optional[str] = None
    sustainability_reference_url: Optional[str] = None
    design_compliance_reference_url: Optional[str] = None
    other_reference_urls: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_project() -> str:
    return """
Extract structured details for a single mixed-use development project mentioned in the answer. Return exactly the following JSON fields from the answer text (do not invent). If something is not present, return null or an empty list as appropriate.

Required identification:
- project_name: official project name
- full_address: complete street address; if only city/state mentioned, include them here as given
- city: city name if present
- state: state (two-letter preferred, otherwise as given)

Area and composition:
- total_sqft: total building area (any format)
- commercial_sqft: commercial/office component area (any format)
- residential_sqft: residential component area (any format)

Core requirements (verbatim phrases as cited):
- mixed_use_composition_claim: the statement showing both commercial and residential on same property
- leed_status: text like "LEED Gold", "pursuing LEED Gold", "LEED Platinum"
- energy_efficiency_compliance: statement referencing 2025 Building Energy Efficiency Standards (effective Jan 1, 2026) or explicitly stated equivalent energy performance standard/code compliance (e.g., Title 24 2025)
- office_sqft_per_employee: numeric or textual figure for office space per employee (e.g., "175 sf/employee", "150-200")
- parking_ratio_office_per_1000sf: office parking ratio (e.g., "5 spaces per 1,000 sf")
- multifamily_units: number of dwelling units (as given)
- multifamily_standards_compliance: statement that the residential component complies with applicable multifamily design/construction standards
- ada_compliance: statement indicating ADA Standards for Accessible Design compliance
- ibc_occupancy_classification: IBC occupancy info for mixed-use (e.g., "Group B and R-2, separated/non-separated")
- fire_safety_compliance: statement citing NFPA standards or equivalent life-safety systems compliance
- noi_statement: statement showing positive (actual or projected) Net Operating Income (NOI)
- property_management_firm: professional property management firm for the residential component, or a statement confirming PM services

Additional (non-critical):
- timeline: project's development/construction timeline or completion date
- sustainable_features: list of specific green building features (e.g., PV, green roof, low-flow fixtures)
- zoning_permits: statement about zoning/building permit approvals for mixed-use
- residential_unit_density: units per acre as given (text)
- tenant_mix: description of commercial tenant mix (office/retail/restaurant/anchor tenants)
- financing_structure: description of financing approach incl. down payment % or lending sources if given

References (URLs exactly as in the answer; do not infer):
- primary_reference_url: the main authoritative project page (developer/official page or reputable article)
- sustainability_reference_url: a page documenting LEED status or energy/sustainability standards
- design_compliance_reference_url: a page with design/compliance details (e.g., space allocation, parking, occupancy, code info)
- other_reference_urls: array of any other URLs cited for this project
"""


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def dedup_urls(urls: List[Optional[str]]) -> List[str]:
    out: List[str] = []
    seen = set()
    for u in urls:
        if not u:
            continue
        u_str = str(u).strip()
        if not u_str:
            continue
        if u_str not in seen:
            out.append(u_str)
            seen.add(u_str)
    return out


def all_sources(extracted: MixedUseProjectExtraction) -> List[str]:
    return dedup_urls([
        extracted.primary_reference_url,
        extracted.sustainability_reference_url,
        extracted.design_compliance_reference_url,
        *(extracted.other_reference_urls or []),
    ])


def sources_pref(extracted: MixedUseProjectExtraction, prefer: List[str]) -> List[str]:
    buckets: Dict[str, Optional[str] | List[str]] = {
        "primary": extracted.primary_reference_url,
        "sustainability": extracted.sustainability_reference_url,
        "design": extracted.design_compliance_reference_url,
        "other": extracted.other_reference_urls or [],
        "all": all_sources(extracted),
    }
    collected: List[str] = []
    for key in prefer:
        val = buckets.get(key)
        if isinstance(val, list):
            collected.extend(val)
        elif isinstance(val, str) and val:
            collected.append(val)
    # Always fall back to all if specific preferences yield nothing
    if not collected:
        collected = buckets["all"] if isinstance(buckets["all"], list) else []
    return dedup_urls(collected)


def project_anchor(extracted: MixedUseProjectExtraction) -> str:
    name = extracted.project_name or "the project"
    where = None
    if extracted.full_address:
        where = extracted.full_address
    else:
        city_state = " ".join([p for p in [extracted.city, extracted.state] if p])
        where = city_state if city_state else "its stated location"
    return f"{name} located at {where}"


async def add_claim_leaf(
    evaluator: Evaluator,
    *,
    parent,
    node_id: str,
    desc: str,
    claim: str,
    sources: List[str],
    critical: bool = True,
    extra_prerequisites: Optional[List] = None,
    additional_instruction: str = "None",
) -> None:
    leaf = evaluator.add_leaf(
        id=node_id,
        desc=desc,
        parent=parent,
        critical=critical,
    )
    await evaluator.verify(
        claim=claim,
        node=leaf,
        sources=sources,
        additional_instruction=additional_instruction,
        extra_prerequisites=extra_prerequisites,
    )


# --------------------------------------------------------------------------- #
# Subtree builders                                                            #
# --------------------------------------------------------------------------- #
async def build_reference_documentation(
    evaluator: Evaluator,
    parent,
    extracted: MixedUseProjectExtraction,
):
    """
    Build the Project_Reference_Documentation subtree.
    Returns a dict of prerequisite nodes that other checks can depend on.
    """
    # Container node (non-critical to avoid global cross-sibling gating)
    ref_container = evaluator.add_parallel(
        id="Project_Reference_Documentation",
        desc="Provide at least three valid reference URLs documenting project specs and compliance",
        parent=parent,
        critical=False,
    )

    urls_all = all_sources(extracted)
    min_three_urls = evaluator.add_custom_node(
        result=(len(urls_all) >= 3),
        id="project_refs_min_three",
        desc="At least three reference URLs are provided",
        parent=ref_container,
        critical=True,
    )

    # Primary reference (sequential: existence -> support)
    primary_seq = evaluator.add_sequential(
        id="Primary_Project_Reference",
        desc="Primary reference URL documents basic information, mixed-use nature, and location",
        parent=ref_container,
        critical=True,
    )
    primary_exists = evaluator.add_custom_node(
        result=bool(extracted.primary_reference_url),
        id="primary_ref_exists",
        desc="Primary reference URL is provided",
        parent=primary_seq,
        critical=True,
    )
    await add_claim_leaf(
        evaluator,
        parent=primary_seq,
        node_id="primary_ref_supports_basics",
        desc="Primary URL supports project's name, mixed-use nature, and location",
        claim=f"The referenced page documents {project_anchor(extracted)}, including that it is a mixed-use development.",
        sources=sources_pref(extracted, ["primary"]),
        critical=True,
        extra_prerequisites=[primary_exists, min_three_urls],
        additional_instruction="Confirm the page clearly names the project and indicates commercial and residential uses at the stated location.",
    )

    # Sustainability/Certification reference
    sust_seq = evaluator.add_sequential(
        id="Sustainability_Certification_Reference",
        desc="Reference URL documents LEED or sustainability/energy standards",
        parent=ref_container,
        critical=True,
    )
    sust_exists = evaluator.add_custom_node(
        result=bool(extracted.sustainability_reference_url),
        id="sustainability_ref_exists",
        desc="Sustainability/certification URL is provided",
        parent=sust_seq,
        critical=True,
    )
    await add_claim_leaf(
        evaluator,
        parent=sust_seq,
        node_id="sustainability_ref_supports_cert",
        desc="Sustainability URL supports LEED status or energy/sustainability features",
        claim=f"The referenced page documents {project_anchor(extracted)} LEED certification status or equivalent sustainability/energy standards.",
        sources=sources_pref(extracted, ["sustainability"]),
        critical=True,
        extra_prerequisites=[sust_exists, min_three_urls],
        additional_instruction="Look for explicit LEED level or active pursuit (e.g., 'LEED Gold'), or explicit energy/sustainability compliance statements.",
    )

    # Design/Compliance reference
    design_seq = evaluator.add_sequential(
        id="Design_and_Compliance_Reference",
        desc="Reference URL documents design standards, space allocation, parking, occupancy or regulatory compliance",
        parent=ref_container,
        critical=True,
    )
    design_exists = evaluator.add_custom_node(
        result=bool(extracted.design_compliance_reference_url),
        id="design_ref_exists",
        desc="Design/compliance URL is provided",
        parent=design_seq,
        critical=True,
    )
    await add_claim_leaf(
        evaluator,
        parent=design_seq,
        node_id="design_ref_supports_compliance",
        desc="Design/compliance URL supports space/parking/occupancy/compliance details",
        claim=f"The referenced page documents design/compliance details for {project_anchor(extracted)} such as space allocation, parking ratios, occupancy, or other regulatory information.",
        sources=sources_pref(extracted, ["design"]),
        critical=True,
        extra_prerequisites=[design_exists, min_three_urls],
        additional_instruction="Accept pages that explicitly detail office area allocation, parking ratios, occupancy groups, or code compliance statements.",
    )

    return {
        "gate_min_three": min_three_urls,
        "primary_seq": primary_seq,
        "sust_seq": sust_seq,
        "design_seq": design_seq,
    }


async def verify_project_identification(
    evaluator: Evaluator,
    parent,
    extracted: MixedUseProjectExtraction,
    prereq_gate,
):
    container = evaluator.add_sequential(
        id="Project_Identification_and_Name",
        desc="Provide the official name and complete address of the project",
        parent=parent,
        critical=False,
    )
    exists = evaluator.add_custom_node(
        result=bool(extracted.project_name and (extracted.full_address or (extracted.city and extracted.state))),
        id="project_identification_exists",
        desc="Project name and a complete address (or city+state) are provided",
        parent=container,
        critical=True,
    )
    await add_claim_leaf(
        evaluator,
        parent=container,
        node_id="project_identification_supported",
        desc="Project name and address supported by source",
        claim=f"The referenced page shows the project's official name and its address/location: {project_anchor(extracted)}.",
        sources=sources_pref(extracted, ["primary", "all"]),
        critical=True,
        extra_prerequisites=[exists, prereq_gate],
        additional_instruction="Confirm the page explicitly lists the project name and its location/address.",
    )


async def verify_mixed_use(
    evaluator: Evaluator,
    parent,
    extracted: MixedUseProjectExtraction,
    prereq_gate,
):
    container = evaluator.add_parallel(
        id="Mixed_Use_Composition_Confirmed",
        desc="Verify the project contains both commercial and residential components on the same property",
        parent=parent,
        critical=False,
    )
    await add_claim_leaf(
        evaluator,
        parent=container,
        node_id="mixed_use_both_components",
        desc="Project has both commercial and residential components on same property",
        claim=f"{project_anchor(extracted)} includes both commercial (e.g., office/retail) and residential components on the same property.",
        sources=sources_pref(extracted, ["primary", "design", "all"]),
        critical=True,
        extra_prerequisites=[prereq_gate],
        additional_instruction="The page must clearly indicate the project is mixed-use with both commercial and residential on a single site.",
    )


async def verify_leed_status(
    evaluator: Evaluator,
    parent,
    extracted: MixedUseProjectExtraction,
    prereq_gate,
):
    container = evaluator.add_parallel(
        id="LEED_Gold_Certification_Status",
        desc="Confirm the project achieved or is pursuing LEED Gold (60-79) or higher (Platinum: 80+)",
        parent=parent,
        critical=False,
    )
    status_text = extracted.leed_status or "LEED Gold or higher"
    await add_claim_leaf(
        evaluator,
        parent=container,
        node_id="leed_gold_or_higher",
        desc="LEED Gold (or higher) achieved or being pursued",
        claim=f"{project_anchor(extracted)} has achieved or is pursuing {status_text}, which is LEED Gold (60-79 points) or higher.",
        sources=sources_pref(extracted, ["sustainability", "primary", "all"]),
        critical=True,
        extra_prerequisites=[prereq_gate],
        additional_instruction="Explicitly look for 'LEED Gold' (or 'Platinum'). If 'pursuing' is stated, this is acceptable.",
    )


async def verify_energy_efficiency(
    evaluator: Evaluator,
    parent,
    extracted: MixedUseProjectExtraction,
    prereq_gate,
):
    container = evaluator.add_parallel(
        id="Energy_Efficiency_Standards_Compliance",
        desc="Verify compliance with 2025 Building Energy Efficiency Standards (effective Jan 1, 2026) or equivalent",
        parent=parent,
        critical=False,
    )
    claim_text = (
        f"{project_anchor(extracted)} complies with the 2025 Building Energy Efficiency Standards "
        f"(effective January 1, 2026) or an explicitly stated equivalent energy performance/code standard."
    )
    await add_claim_leaf(
        evaluator,
        parent=container,
        node_id="energy_2025_or_equivalent",
        desc="Complies with 2025 energy standards or equivalent code",
        claim=claim_text,
        sources=sources_pref(extracted, ["sustainability", "design", "all"]),
        critical=True,
        extra_prerequisites=[prereq_gate],
        additional_instruction="Accept explicit references to 2025 Title 24 (or equivalent) or formal statements of compliance from credible/official sources.",
    )


async def verify_office_space_per_employee(
    evaluator: Evaluator,
    parent,
    extracted: MixedUseProjectExtraction,
    prereq_gate,
):
    container = evaluator.add_sequential(
        id="Office_Space_Per_Employee_Standard",
        desc="Office space allocation between 150-200 square feet per employee",
        parent=parent,
        critical=False,
    )
    exists = evaluator.add_custom_node(
        result=bool(extracted.office_sqft_per_employee),
        id="office_space_per_employee_value_exists",
        desc="Office space per employee figure is provided",
        parent=container,
        critical=True,
    )
    await add_claim_leaf(
        evaluator,
        parent=container,
        node_id="office_space_in_range",
        desc="Office space allocation is between 150-200 sf per employee",
        claim=(
            f"The office space allocation for {project_anchor(extracted)} is between 150 and 200 square feet "
            f"per employee (reported value: {extracted.office_sqft_per_employee or 'unspecified'})."
        ),
        sources=sources_pref(extracted, ["design", "primary", "all"]),
        critical=True,
        extra_prerequisites=[exists, prereq_gate],
        additional_instruction="Confirm any stated planning standard or program metric indicating ~150–200 sf/employee.",
    )


async def verify_parking_ratio(
    evaluator: Evaluator,
    parent,
    extracted: MixedUseProjectExtraction,
    prereq_gate,
):
    container = evaluator.add_sequential(
        id="Parking_Ratio_Requirements",
        desc="Office parking ratio between 4-6 spaces per 1,000 sf of leasable area",
        parent=parent,
        critical=False,
    )
    exists = evaluator.add_custom_node(
        result=bool(extracted.parking_ratio_office_per_1000sf),
        id="parking_ratio_value_exists",
        desc="Parking ratio figure is provided",
        parent=container,
        critical=True,
    )
    await add_claim_leaf(
        evaluator,
        parent=container,
        node_id="parking_ratio_in_range",
        desc="Parking ratio is between 4-6 spaces per 1,000 sf of office",
        claim=(
            f"The commercial office portion of {project_anchor(extracted)} provides a parking ratio between 4 and 6 "
            f"spaces per 1,000 square feet of leasable area (reported: {extracted.parking_ratio_office_per_1000sf or 'unspecified'})."
        ),
        sources=sources_pref(extracted, ["design", "primary", "all"]),
        critical=True,
        extra_prerequisites=[exists, prereq_gate],
        additional_instruction="Confirm a parking metric stated explicitly for the office component within the 4–6/1000 sf range.",
    )


async def verify_multifamily_standards(
    evaluator: Evaluator,
    parent,
    extracted: MixedUseProjectExtraction,
    prereq_gate,
):
    container = evaluator.add_parallel(
        id="Multifamily_Standards_Compliance",
        desc="Residential component complies with multifamily standards for 3+ units",
        parent=parent,
        critical=False,
    )
    await add_claim_leaf(
        evaluator,
        parent=container,
        node_id="multifamily_compliance_3plus",
        desc="Residential component: 3+ units and multifamily standards compliance",
        claim=(
            f"{project_anchor(extracted)} has a residential component of at least 3 dwelling units "
            f"and complies with applicable multifamily design and construction standards."
        ),
        sources=sources_pref(extracted, ["design", "primary", "all"]),
        critical=True,
        extra_prerequisites=[prereq_gate],
        additional_instruction="Look for unit counts (>=3) and explicit multifamily/building code compliance statements.",
    )


async def verify_ada(
    evaluator: Evaluator,
    parent,
    extracted: MixedUseProjectExtraction,
    prereq_gate,
):
    container = evaluator.add_parallel(
        id="ADA_Accessibility_Compliance",
        desc="Meets ADA Standards for Accessible Design",
        parent=parent,
        critical=False,
    )
    await add_claim_leaf(
        evaluator,
        parent=container,
        node_id="ada_standards_met",
        desc="Project meets ADA Standards for Accessible Design",
        claim=f"{project_anchor(extracted)} meets ADA Standards for Accessible Design.",
        sources=sources_pref(extracted, ["design", "primary", "all"]),
        critical=True,
        extra_prerequisites=[prereq_gate],
        additional_instruction="Accept explicit ADA compliance statements or code summaries indicating ADA adherence.",
    )


async def verify_ibc_occupancy(
    evaluator: Evaluator,
    parent,
    extracted: MixedUseProjectExtraction,
    prereq_gate,
):
    container = evaluator.add_parallel(
        id="IBC_Occupancy_Classification",
        desc="Proper IBC occupancy type classification for mixed-use buildings",
        parent=parent,
        critical=False,
    )
    await add_claim_leaf(
        evaluator,
        parent=container,
        node_id="ibc_mixed_use_occupancy",
        desc="IBC mixed-use occupancy properly classified",
        claim=(
            f"{project_anchor(extracted)} is assigned proper IBC occupancy classifications for a mixed-use building "
            f"(e.g., B, R-2, M) with appropriate separated or non-separated use approach."
        ),
        sources=sources_pref(extracted, ["design", "primary", "all"]),
        critical=True,
        extra_prerequisites=[prereq_gate],
        additional_instruction="Look for explicit occupancy groups or mixed-occupancy strategies per IBC.",
    )


async def verify_fire_safety(
    evaluator: Evaluator,
    parent,
    extracted: MixedUseProjectExtraction,
    prereq_gate,
):
    container = evaluator.add_parallel(
        id="Fire_Safety_Code_Requirements",
        desc="Compliance with applicable fire safety codes (NFPA) and life safety systems",
        parent=parent,
        critical=False,
    )
    await add_claim_leaf(
        evaluator,
        parent=container,
        node_id="nfpa_compliance",
        desc="Complies with NFPA fire safety codes/life safety systems",
        claim=f"{project_anchor(extracted)} complies with applicable fire safety codes (e.g., NFPA) and life-safety systems requirements.",
        sources=sources_pref(extracted, ["design", "primary", "all"]),
        critical=True,
        extra_prerequisites=[prereq_gate],
        additional_instruction="Accept explicit NFPA references (e.g., NFPA 13 sprinklers) or fire code compliance statements.",
    )


async def verify_noi(
    evaluator: Evaluator,
    parent,
    extracted: MixedUseProjectExtraction,
    prereq_gate,
):
    container = evaluator.add_parallel(
        id="Positive_Net_Operating_Income",
        desc="Evidence of positive projected or actual Net Operating Income (NOI)",
        parent=parent,
        critical=False,
    )
    await add_claim_leaf(
        evaluator,
        parent=container,
        node_id="noi_positive",
        desc="Positive NOI (projected or actual) documented",
        claim=f"{project_anchor(extracted)} demonstrates positive Net Operating Income (NOI), excluding debt service.",
        sources=sources_pref(extracted, ["all"]),
        critical=True,
        extra_prerequisites=[prereq_gate],
        additional_instruction="Prefer investor or pro forma docs; accept credible sources explicitly stating positive NOI/projections.",
    )


async def verify_property_management(
    evaluator: Evaluator,
    parent,
    extracted: MixedUseProjectExtraction,
    prereq_gate,
):
    container = evaluator.add_parallel(
        id="Property_Management_Services",
        desc="Professional property management for the residential component",
        parent=parent,
        critical=False,
    )
    firm_text = extracted.property_management_firm or "a professional property management firm"
    await add_claim_leaf(
        evaluator,
        parent=container,
        node_id="pm_services_residential",
        desc="Residential component has professional property management",
        claim=f"The residential component of {project_anchor(extracted)} is managed by {firm_text} (or equivalent professional PM services are contracted).",
        sources=sources_pref(extracted, ["primary", "all"]),
        critical=True,
        extra_prerequisites=[prereq_gate],
        additional_instruction="Look for explicit PM firm names or formal statements of contracted residential property management.",
    )


async def verify_sqft_breakdown(
    evaluator: Evaluator,
    parent,
    extracted: MixedUseProjectExtraction,
    prereq_gate,
):
    container = evaluator.add_sequential(
        id="Project_Total_Square_Footage",
        desc="Total square footage with commercial and residential breakdown",
        parent=parent,
        critical=False,
    )
    exists = evaluator.add_custom_node(
        result=bool(extracted.total_sqft and extracted.commercial_sqft and extracted.residential_sqft),
        id="sqft_breakdown_exists",
        desc="Total, commercial, and residential square footage values are provided",
        parent=container,
        critical=True,
    )
    await add_claim_leaf(
        evaluator,
        parent=container,
        node_id="sqft_breakdown_supported",
        desc="Source supports total and component square footage breakdown",
        claim=(
            f"The sources report for {project_anchor(extracted)} the total area "
            f"({extracted.total_sqft or 'total'}), with commercial ({extracted.commercial_sqft or 'commercial'}) "
            f"and residential ({extracted.residential_sqft or 'residential'}) breakdown."
        ),
        sources=sources_pref(extracted, ["primary", "design", "all"]),
        critical=True,
        extra_prerequisites=[exists, prereq_gate],
        additional_instruction="Allow minor rounding differences. The page must clearly list total and separate component areas.",
    )


async def verify_timeline(
    evaluator: Evaluator,
    parent,
    extracted: MixedUseProjectExtraction,
    prereq_gate,
):
    container = evaluator.add_parallel(
        id="Development_Timeline_Information",
        desc="Project development/construction timeline or completion date",
        parent=parent,
        critical=False,
    )
    await add_claim_leaf(
        evaluator,
        parent=container,
        node_id="timeline_available",
        desc="Timeline/Completion info documented",
        claim=f"The sources provide development/construction timeline or completion date for {project_anchor(extracted)}.",
        sources=sources_pref(extracted, ["primary", "all"]),
        critical=False,
        extra_prerequisites=[prereq_gate],
        additional_instruction="Look for a schedule, start/finish dates, or stated completion year.",
    )


async def verify_sustainable_features(
    evaluator: Evaluator,
    parent,
    extracted: MixedUseProjectExtraction,
    prereq_gate,
):
    container = evaluator.add_parallel(
        id="Sustainable_Building_Features",
        desc="Specific green building features identified",
        parent=parent,
        critical=False,
    )
    feature_list = ", ".join(extracted.sustainable_features) if extracted.sustainable_features else "sustainable features"
    await add_claim_leaf(
        evaluator,
        parent=container,
        node_id="sustainable_features_supported",
        desc="Sustainable features documented",
        claim=f"The sources identify specific sustainable building features for {project_anchor(extracted)} (e.g., {feature_list}).",
        sources=sources_pref(extracted, ["sustainability", "primary", "all"]),
        critical=False,
        extra_prerequisites=[prereq_gate],
        additional_instruction="Accept explicit mentions (e.g., PV, green roof, efficient fixtures, recycled materials).",
    )


async def verify_zoning_permits(
    evaluator: Evaluator,
    parent,
    extracted: MixedUseProjectExtraction,
    prereq_gate,
):
    container = evaluator.add_parallel(
        id="Zoning_and_Permit_Approvals",
        desc="Necessary zoning approvals and permits obtained",
        parent=parent,
        critical=False,
    )
    await add_claim_leaf(
        evaluator,
        parent=container,
        node_id="zoning_permits_obtained",
        desc="Zoning/permits documented",
        claim=f"The sources indicate that {project_anchor(extracted)} has obtained necessary zoning approvals and/or building permits for mixed-use.",
        sources=sources_pref(extracted, ["design", "primary", "all"]),
        critical=False,
        extra_prerequisites=[prereq_gate],
        additional_instruction="Look for planning commission/city approvals, permit issuance, or entitlement status.",
    )


async def verify_unit_count_density(
    evaluator: Evaluator,
    parent,
    extracted: MixedUseProjectExtraction,
    prereq_gate,
):
    container = evaluator.add_parallel(
        id="Residential_Unit_Count_and_Density",
        desc="Number of dwelling units and density documented",
        parent=parent,
        critical=False,
    )
    await add_claim_leaf(
        evaluator,
        parent=container,
        node_id="unit_count_density_supported",
        desc="Unit count and/or density documented",
        claim=f"The sources document residential unit count and/or density for {project_anchor(extracted)}.",
        sources=sources_pref(extracted, ["primary", "design", "all"]),
        critical=False,
        extra_prerequisites=[prereq_gate],
        additional_instruction="Look for explicit unit counts and, if available, units-per-acre density.",
    )


async def verify_tenant_mix(
    evaluator: Evaluator,
    parent,
    extracted: MixedUseProjectExtraction,
    prereq_gate,
):
    container = evaluator.add_parallel(
        id="Commercial_Tenant_Mix",
        desc="Commercial tenant mix and anchor tenants described",
        parent=parent,
        critical=False,
    )
    await add_claim_leaf(
        evaluator,
        parent=container,
        node_id="tenant_mix_documented",
        desc="Tenant mix documented",
        claim=f"The sources describe the commercial tenant mix for {project_anchor(extracted)} (office/retail/restaurant/anchor tenants).",
        sources=sources_pref(extracted, ["primary", "all"]),
        critical=False,
        extra_prerequisites=[prereq_gate],
        additional_instruction="Descriptions of business types or named anchor tenants are acceptable.",
    )


async def verify_financing_structure(
    evaluator: Evaluator,
    parent,
    extracted: MixedUseProjectExtraction,
    prereq_gate,
):
    container = evaluator.add_parallel(
        id="Financing_Structure_Documentation",
        desc="Financing approach documented incl. down payment % or lending sources",
        parent=parent,
        critical=False,
    )
    await add_claim_leaf(
        evaluator,
        parent=container,
        node_id="financing_structure_supported",
        desc="Financing approach documented",
        claim=f"The sources document the financing approach for {project_anchor(extracted)}, including down payment percentage or lending sources if stated.",
        sources=sources_pref(extracted, ["all"]),
        critical=False,
        extra_prerequisites=[prereq_gate],
        additional_instruction="Investor decks, credible articles, or official releases mentioning financing are acceptable.",
    )


# --------------------------------------------------------------------------- #
# Main evaluation entry                                                       #
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
    # Initialize evaluator
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

    # Extract structured info
    extracted = await evaluator.extract(
        prompt=prompt_extract_project(),
        template_class=MixedUseProjectExtraction,
        extraction_name="mixed_use_project_extraction",
    )

    # Build references subtree first and get a simple gate prerequisite
    refs_info = await build_reference_documentation(evaluator, root, extracted)
    prereq_gate = refs_info["gate_min_three"]

    # Build and verify all other rubric criteria as independent containers
    # (non-critical at root to avoid cross-container precondition skipping).
    await verify_project_identification(evaluator, root, extracted, prereq_gate)
    await verify_mixed_use(evaluator, root, extracted, prereq_gate)
    await verify_leed_status(evaluator, root, extracted, prereq_gate)
    await verify_energy_efficiency(evaluator, root, extracted, prereq_gate)
    await verify_office_space_per_employee(evaluator, root, extracted, prereq_gate)
    await verify_parking_ratio(evaluator, root, extracted, prereq_gate)
    await verify_multifamily_standards(evaluator, root, extracted, prereq_gate)
    await verify_ada(evaluator, root, extracted, prereq_gate)
    await verify_ibc_occupancy(evaluator, root, extracted, prereq_gate)
    await verify_fire_safety(evaluator, root, extracted, prereq_gate)
    await verify_noi(evaluator, root, extracted, prereq_gate)
    await verify_property_management(evaluator, root, extracted, prereq_gate)
    await verify_sqft_breakdown(evaluator, root, extracted, prereq_gate)

    # Non-critical informational checks
    await verify_timeline(evaluator, root, extracted, prereq_gate)
    await verify_sustainable_features(evaluator, root, extracted, prereq_gate)
    await verify_zoning_permits(evaluator, root, extracted, prereq_gate)
    await verify_unit_count_density(evaluator, root, extracted, prereq_gate)
    await verify_tenant_mix(evaluator, root, extracted, prereq_gate)
    await verify_financing_structure(evaluator, root, extracted, prereq_gate)

    # Return structured evaluation summary
    return evaluator.get_summary()