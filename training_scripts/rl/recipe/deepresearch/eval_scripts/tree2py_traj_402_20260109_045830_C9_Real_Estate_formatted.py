import asyncio
import logging
from typing import Any, List, Dict, Optional

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "ca_mixed_use_oz_leed_lihtc"
TASK_DESCRIPTION = (
    "Identify a mixed-use development project in California that satisfies ALL of the following requirements:\n\n"
    "Location Requirements:\n"
    "- Located within a federally designated Qualified Opportunity Zone census tract\n"
    "- Within 1/2 mile walking distance of an existing rail transit station OR within 1/4 mile of an existing bus rapid transit stop\n"
    "- Located in a census tract that qualifies as 'Severely Distressed' under New Markets Tax Credit criteria (poverty rate >30% OR median family income ≤60% of area median OR unemployment rate ≥1.5x national average)\n\n"
    "Certification and Program Requirements:\n"
    "- Achieved LEED Gold certification (60-79 points under the LEED rating system)\n"
    "- Participates in the Low-Income Housing Tax Credit (LIHTC) program\n"
    "- Meets the LIHTC 40/60 minimum set-aside test\n\n"
    "Physical and Design Requirements:\n"
    "- Contains at least 50 residential dwelling units\n"
    "- Total development size of at least 20,000 square feet\n"
    "- Residential uses comprise at least 65% of total square footage\n"
    "- At least 20% of residential units are affordable housing at ≤60% AMI\n"
    "- Commercial uses occupy at least 50% of ground floor street frontage\n\n"
    "Provide: (1) Project name and full address, (2) Developer name, (3) Completion year, "
    "(4) Reference URLs verifying location/OZ designation, LEED Gold status, and project specifications."
)


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class ReferenceURLs(BaseModel):
    """Categorized reference URLs explicitly provided in the answer."""
    location_oz_urls: List[str] = Field(default_factory=list)
    leed_gold_urls: List[str] = Field(default_factory=list)
    specs_urls: List[str] = Field(default_factory=list)


class ProjectFields(BaseModel):
    """Core requested fields from the answer."""
    project_name: Optional[str] = None
    full_address: Optional[str] = None
    developer_name: Optional[str] = None
    completion_year: Optional[str] = None
    city: Optional[str] = None
    state: Optional[str] = None


class LocationCriteria(BaseModel):
    """Location requirements and their supporting details as stated in the answer."""
    is_opportunity_zone: Optional[str] = None  # 'yes'/'no'/'unknown'
    oz_census_tract_id: Optional[str] = None
    transit_proximity_desc: Optional[str] = None
    transit_type: Optional[str] = None  # e.g., 'rail' or 'BRT'
    transit_distance: Optional[str] = None  # e.g., '0.4 miles', '0.2 mi'
    nmtc_severely_distressed_basis: Optional[str] = None  # e.g., 'poverty>30%' or 'MFI<=60%' or 'unemployment>=1.5x'
    nmtc_values: Optional[str] = None  # e.g., 'poverty 33%, MFI 58%, unemployment 1.7x'
    oz_acquisition_involved: Optional[str] = None  # 'yes'/'no'/'unknown'
    oz_substantial_improvement_statement: Optional[str] = None  # Evidence or statement


class ProgramCertification(BaseModel):
    """Certification and program participation details as stated in the answer."""
    leed_level: Optional[str] = None  # Expect 'LEED Gold'
    lihtc_participation: Optional[str] = None  # 'yes'/'no'/'unknown' or narrative
    lihtc_40_60_statement: Optional[str] = None  # narrative about meeting 40/60 test


class PhysicalDesign(BaseModel):
    """Physical and design specifications as stated in the answer."""
    unit_count: Optional[str] = None
    total_size_sqft: Optional[str] = None
    residential_sqft_share_percent: Optional[str] = None
    affordable_units_60ami_percent: Optional[str] = None
    ground_floor_frontage_commercial_percent: Optional[str] = None
    ada_compliance_statement: Optional[str] = None
    mixed_use_statement: Optional[str] = None  # should indicate both residential and commercial


class ProjectExtraction(BaseModel):
    """Complete extraction container for all needed fields and references."""
    fields: ProjectFields = ProjectFields()
    refs: ReferenceURLs = ReferenceURLs()
    location: LocationCriteria = LocationCriteria()
    program: ProgramCertification = ProgramCertification()
    physical: PhysicalDesign = PhysicalDesign()


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_project() -> str:
    return """
    Extract the following structured information from the answer text about a single mixed-use development project in California.
    IMPORTANT: Extract exactly what is stated in the answer. Do not invent or infer anything not explicitly present.
    Return strings for numeric values (e.g., '52', '21,000 sq ft', '66%') to maximize compatibility.

    Required fields:
    - fields.project_name: The project's name.
    - fields.full_address: The full street address (include city and state).
    - fields.developer_name: The developer or development company.
    - fields.completion_year: The year the project completed construction (or officially completed).
    - fields.city: City name (if present).
    - fields.state: State (e.g., 'California' or 'CA').

    Location & designation details (as stated in the answer):
    - location.is_opportunity_zone: 'yes'/'no'/'unknown' (explicit statement).
    - location.oz_census_tract_id: Census tract ID if provided (e.g., '06037201200').
    - location.transit_proximity_desc: Narrative regarding proximity to rail/BRT.
    - location.transit_type: 'rail' or 'BRT' (if specified).
    - location.transit_distance: e.g., '0.4 miles', '0.2 mi' (if stated).
    - location.nmtc_severely_distressed_basis: e.g., 'poverty>30%' or 'MFI<=60%' or 'unemployment>=1.5x' (as stated).
    - location.nmtc_values: Any explicit values (e.g., 'poverty 33%, MFI 58%, unemployment 1.7x').
    - location.oz_acquisition_involved: 'yes'/'no'/'unknown' if acquisition in OZ is mentioned.
    - location.oz_substantial_improvement_statement: Any explicit statement about substantial improvement or that condition being not applicable.

    Certifications and programs:
    - program.leed_level: e.g., 'LEED Gold'.
    - program.lihtc_participation: e.g., 'yes', 'participates', or narrative indicating LIHTC involvement.
    - program.lihtc_40_60_statement: Narrative or statement indicating the 40/60 set-aside is met.

    Physical & design specs:
    - physical.unit_count: Number of residential units (string).
    - physical.total_size_sqft: Total development size (string).
    - physical.residential_sqft_share_percent: Percent of residential square footage (string).
    - physical.affordable_units_60ami_percent: Percent of residential units at or below 60% AMI (string).
    - physical.ground_floor_frontage_commercial_percent: Percent of ground floor street frontage that is commercial (string).
    - physical.ada_compliance_statement: Narrative confirming public/common areas meet ADA standards.
    - physical.mixed_use_statement: Narrative confirming presence of both residential and commercial uses.

    Reference URLs (explicitly provided in the answer; do not invent):
    - refs.location_oz_urls: One or more URLs for location and Opportunity Zone tract designation (address/city + OZ evidence).
    - refs.leed_gold_urls: At least one URL supporting LEED Gold certification status.
    - refs.specs_urls: One or more URLs supporting project specifications (unit count, sizes, percentages, ADA, LIHTC, frontage, etc.).

    If a field is missing in the answer, return null for that field. If a URL category is missing, return an empty list for that array.
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def _non_empty(s: Optional[str]) -> bool:
    return bool(s and str(s).strip())


def _merge_urls(*lists: List[str]) -> List[str]:
    """Merge and deduplicate URLs, ignoring empty strings."""
    seen = set()
    merged: List[str] = []
    for lst in lists:
        for u in lst:
            u2 = (u or "").strip()
            if not u2:
                continue
            if u2 not in seen:
                seen.add(u2)
                merged.append(u2)
    return merged


# --------------------------------------------------------------------------- #
# Tree construction and verification                                          #
# --------------------------------------------------------------------------- #
async def build_verification_tree(evaluator: Evaluator, extraction: ProjectExtraction) -> None:
    # Root node (parallel aggregation). Note: Evaluator.initialize creates a non-critical root by design.
    root = evaluator.root

    # ------------------ Required response fields ------------------------- #
    req_fields_node = evaluator.add_parallel(
        id="required_response_fields",
        desc="Response includes all explicitly requested fields and references.",
        parent=root,
        critical=True
    )

    # Existence checks (custom nodes are binary leaf-equivalents)
    evaluator.add_custom_node(
        result=_non_empty(extraction.fields.project_name) and _non_empty(extraction.fields.full_address),
        id="project_name_and_full_address",
        desc="Project name and full address are provided.",
        parent=req_fields_node,
        critical=True
    )

    evaluator.add_custom_node(
        result=_non_empty(extraction.fields.developer_name),
        id="developer_name",
        desc="Developer (or development company) name is provided.",
        parent=req_fields_node,
        critical=True
    )

    evaluator.add_custom_node(
        result=_non_empty(extraction.fields.completion_year),
        id="completion_year",
        desc="Project completion (or construction completion) year is provided.",
        parent=req_fields_node,
        critical=True
    )

    # Reference URLs provided
    refs_node = evaluator.add_parallel(
        id="reference_urls_provided",
        desc="Reference URLs are provided that verify (a) location + Opportunity Zone designation, (b) LEED Gold status, and (c) project specifications.",
        parent=req_fields_node,
        critical=True
    )

    evaluator.add_custom_node(
        result=len(extraction.refs.location_oz_urls) > 0,
        id="urls_cover_location_and_oz",
        desc="One or more URLs are provided that collectively support the project’s location (address/city) and its Qualified Opportunity Zone tract designation.",
        parent=refs_node,
        critical=True
    )

    evaluator.add_custom_node(
        result=len(extraction.refs.leed_gold_urls) > 0,
        id="urls_leed_gold",
        desc="At least one URL is provided that supports the project’s LEED Gold certification status.",
        parent=refs_node,
        critical=True
    )

    evaluator.add_custom_node(
        result=len(extraction.refs.specs_urls) > 0,
        id="urls_project_specifications",
        desc="One or more URLs are provided that collectively support the project specifications needed to evaluate the stated physical/design and affordability/frontage requirements.",
        parent=refs_node,
        critical=True
    )

    # ------------------ Project is real and in scope --------------------- #
    scope_node = evaluator.add_parallel(
        id="project_is_real_and_in_scope",
        desc="Project is a real, verifiable mixed-use development located in California (residential + commercial).",
        parent=root,
        critical=True
    )

    # project_exists
    exists_leaf = evaluator.add_leaf(
        id="project_exists",
        desc="The project exists and is verifiable through documentation.",
        parent=scope_node,
        critical=True
    )
    claim_exists = (
        f"The project named '{extraction.fields.project_name or 'UNKNOWN'}' at address "
        f"'{extraction.fields.full_address or 'UNKNOWN'}' is a real mixed-use development documented by the provided URLs."
    )
    await evaluator.verify(
        claim=claim_exists,
        node=exists_leaf,
        sources=_merge_urls(extraction.refs.location_oz_urls, extraction.refs.specs_urls, extraction.refs.leed_gold_urls),
        additional_instruction="Verify that the named project exists and is documented by credible sources (developer page, news, city documents, LEED listings, etc.)."
    )

    # project_california
    ca_leaf = evaluator.add_leaf(
        id="project_california",
        desc="The project is located in California.",
        parent=scope_node,
        critical=True
    )
    claim_ca = "The project is located in California (CA)."
    await evaluator.verify(
        claim=claim_ca,
        node=ca_leaf,
        sources=_merge_urls(extraction.refs.location_oz_urls, extraction.refs.specs_urls),
        additional_instruction="Confirm the address/city/state clearly indicates the project is in California (CA)."
    )

    # project_mixed_use
    mixed_use_leaf = evaluator.add_leaf(
        id="project_mixed_use",
        desc="The project includes both residential and commercial components.",
        parent=scope_node,
        critical=True
    )
    claim_mixed = "This project includes both residential and commercial uses (i.e., it is mixed-use)."
    await evaluator.verify(
        claim=claim_mixed,
        node=mixed_use_leaf,
        sources=extraction.refs.specs_urls,
        additional_instruction="Look for explicit statements of residential and commercial components (e.g., housing + retail/office)."
    )

    # ------------------ Location requirements ---------------------------- #
    loc_node = evaluator.add_parallel(
        id="location_requirements",
        desc="Project meets all location-based requirements.",
        parent=root,
        critical=True
    )

    # opportunity_zone
    oz_leaf = evaluator.add_leaf(
        id="opportunity_zone",
        desc="Project is located within a federally designated Qualified Opportunity Zone census tract.",
        parent=loc_node,
        critical=True
    )
    claim_oz = "The project is located within a federally designated Qualified Opportunity Zone census tract."
    await evaluator.verify(
        claim=claim_oz,
        node=oz_leaf,
        sources=extraction.refs.location_oz_urls,
        additional_instruction="Confirm that the address/census tract is inside a Qualified Opportunity Zone using authoritative sources (e.g., official OZ maps, government resources, or credible listings)."
    )

    # transit_proximity
    transit_leaf = evaluator.add_leaf(
        id="transit_proximity",
        desc="Project is within 1/2 mile walking distance of an existing rail transit station OR within 1/4 mile of an existing bus rapid transit stop.",
        parent=loc_node,
        critical=True
    )
    claim_transit = (
        "The project is within 1/2 mile walking distance of an existing rail transit station OR within 1/4 mile of an existing bus rapid transit (BRT) stop."
    )
    await evaluator.verify(
        claim=claim_transit,
        node=transit_leaf,
        sources=_merge_urls(extraction.refs.specs_urls, extraction.refs.location_oz_urls),
        additional_instruction="Accept explicit proximity statements or clearly substantiated distances in credible sources. Rail includes metro/light rail/commuter rail; BRT refers to bus rapid transit stops."
    )

    # severely_distressed_nmtc
    nmtc_leaf = evaluator.add_leaf(
        id="severely_distressed_nmtc",
        desc="Project’s census tract qualifies as 'Severely Distressed' under NMTC criteria (poverty >30% OR median family income ≤60% of area median OR unemployment ≥1.5× national average).",
        parent=loc_node,
        critical=True
    )
    claim_nmtc = (
        "The project's census tract qualifies as 'Severely Distressed' under NMTC criteria "
        "(poverty >30% OR median family income ≤60% of area median OR unemployment ≥1.5× national average)."
    )
    await evaluator.verify(
        claim=claim_nmtc,
        node=nmtc_leaf,
        sources=_merge_urls(extraction.refs.location_oz_urls, extraction.refs.specs_urls),
        additional_instruction="Look for explicit NMTC severe distress qualification or tract metrics supporting one of the criteria."
    )

    # oz_substantial_improvement_if_acquired
    oz_si_leaf = evaluator.add_leaf(
        id="oz_substantial_improvement_if_acquired",
        desc="If the project involved acquisition of existing property in an Opportunity Zone, the improvements equal at least 100% of the acquisition cost within 30 months; otherwise this condition is not applicable and is treated as satisfied.",
        parent=loc_node,
        critical=True
    )
    # Build claim considering provided narrative
    if (extraction.location.oz_acquisition_involved or "").lower() == "yes":
        si_claim = (
            "For this Opportunity Zone project that involved acquisition of existing property, "
            "the improvements equaled at least 100% of the acquisition cost within 30 months."
        )
    else:
        si_claim = (
            "This project did not involve acquisition of existing property in an Opportunity Zone, "
            "so the substantial improvement requirement is not applicable and is treated as satisfied."
        )
    await evaluator.verify(
        claim=si_claim,
        node=oz_si_leaf,
        sources=_merge_urls(extraction.refs.specs_urls, extraction.refs.location_oz_urls),
        additional_instruction="If acquisition occurred, confirm improvements ≥100% of basis within 30 months; if not, treat the requirement as N/A and satisfied."
    )

    # ------------------ Certifications and programs ---------------------- #
    cert_node = evaluator.add_parallel(
        id="certification_and_program_requirements",
        desc="Project meets required certification and program participation requirements.",
        parent=root,
        critical=True
    )

    # LEED Gold
    leed_leaf = evaluator.add_leaf(
        id="leed_gold",
        desc="Project achieved LEED Gold certification (60–79 points under the LEED rating system).",
        parent=cert_node,
        critical=True
    )
    claim_leed = "The project achieved LEED Gold certification."
    await evaluator.verify(
        claim=claim_leed,
        node=leed_leaf,
        sources=extraction.refs.leed_gold_urls,
        additional_instruction="Accept official LEED listings, USGBC project pages, or credible sources explicitly stating 'LEED Gold'."
    )

    # LIHTC participation
    lihtc_leaf = evaluator.add_leaf(
        id="lihtc_participation",
        desc="Project participates in the Low-Income Housing Tax Credit (LIHTC) program.",
        parent=cert_node,
        critical=True
    )
    claim_lihtc = "The project participates in the Low-Income Housing Tax Credit (LIHTC) program."
    await evaluator.verify(
        claim=claim_lihtc,
        node=lihtc_leaf,
        sources=extraction.refs.specs_urls,
        additional_instruction="Confirm LIHTC participation via developer materials, housing authority listings, or credible publications."
    )

    # LIHTC 40/60 minimum set-aside
    lihtc4060_leaf = evaluator.add_leaf(
        id="lihtc_40_60_set_aside",
        desc="Project meets the LIHTC 40/60 minimum set-aside test (≥40% of residential units affordable to households at or below 60% AMI).",
        parent=cert_node,
        critical=True
    )
    claim_4060 = (
        "The project meets the LIHTC 40/60 minimum set-aside test (at least 40% of residential units at or below 60% AMI)."
    )
    await evaluator.verify(
        claim=claim_4060,
        node=lihtc4060_leaf,
        sources=extraction.refs.specs_urls,
        additional_instruction="Look for explicit statements of the 40/60 test being met, or clear unit-percentage counts matching LIHTC 40/60."
    )

    # ------------------ Physical and design requirements ----------------- #
    phys_node = evaluator.add_parallel(
        id="physical_and_design_requirements",
        desc="Project meets all physical/design requirements.",
        parent=root,
        critical=True
    )

    # unit_count_minimum
    units_leaf = evaluator.add_leaf(
        id="unit_count_minimum",
        desc="Residential component contains at least 50 dwelling units.",
        parent=phys_node,
        critical=True
    )
    unit_str = extraction.physical.unit_count or "UNKNOWN"
    claim_units = f"The project contains {unit_str} residential dwelling units and meets the minimum of at least 50 units."
    await evaluator.verify(
        claim=claim_units,
        node=units_leaf,
        sources=extraction.refs.specs_urls,
        additional_instruction="Confirm the residential unit count is 50 or greater."
    )

    # total_size_minimum
    size_leaf = evaluator.add_leaf(
        id="total_size_minimum",
        desc="Total development size is at least 20,000 square feet.",
        parent=phys_node,
        critical=True
    )
    size_str = extraction.physical.total_size_sqft or "UNKNOWN"
    claim_size = f"The total development size is {size_str} and is at least 20,000 square feet."
    await evaluator.verify(
        claim=claim_size,
        node=size_leaf,
        sources=extraction.refs.specs_urls,
        additional_instruction="Confirm total building/site development square footage ≥ 20,000 sq ft."
    )

    # residential_sqft_share
    res_share_leaf = evaluator.add_leaf(
        id="residential_sqft_share",
        desc="Residential uses comprise at least 65% of total square footage.",
        parent=phys_node,
        critical=True
    )
    res_share_str = extraction.physical.residential_sqft_share_percent or "UNKNOWN"
    claim_res_share = f"Residential uses comprise {res_share_str} of total square footage, which is at least 65%."
    await evaluator.verify(
        claim=claim_res_share,
        node=res_share_leaf,
        sources=extraction.refs.specs_urls,
        additional_instruction="Confirm residential square footage share ≥ 65% of total development."
    )

    # affordable_units_20pct_at_60ami
    aff_leaf = evaluator.add_leaf(
        id="affordable_units_20pct_at_60ami",
        desc="At least 20% of residential units are affordable housing at or below 60% of Area Median Income (AMI).",
        parent=phys_node,
        critical=True
    )
    aff_str = extraction.physical.affordable_units_60ami_percent or "UNKNOWN"
    claim_aff = f"At least 20% of residential units are affordable at ≤60% AMI (stated as {aff_str})."
    await evaluator.verify(
        claim=claim_aff,
        node=aff_leaf,
        sources=extraction.refs.specs_urls,
        additional_instruction="Confirm affordable housing share at ≤60% AMI is ≥ 20% of units."
    )

    # ground_floor_frontage_commercial
    frontage_leaf = evaluator.add_leaf(
        id="ground_floor_frontage_commercial",
        desc="Commercial uses occupy at least 50% of ground floor street frontage.",
        parent=phys_node,
        critical=True
    )
    frontage_str = extraction.physical.ground_floor_frontage_commercial_percent or "UNKNOWN"
    claim_frontage = f"Commercial uses occupy {frontage_str} of ground floor street frontage, which is at least 50%."
    await evaluator.verify(
        claim=claim_frontage,
        node=frontage_leaf,
        sources=extraction.refs.specs_urls,
        additional_instruction="Confirm the commercial frontage share on the ground floor is ≥ 50%."
    )

    # ada_accessibility_public_common_areas
    ada_leaf = evaluator.add_leaf(
        id="ada_accessibility_public_common_areas",
        desc="All public and common areas meet Americans with Disabilities Act (ADA) accessibility standards.",
        parent=phys_node,
        critical=True
    )
    ada_str = extraction.physical.ada_compliance_statement or "UNKNOWN"
    claim_ada = "All public and common areas of the project meet ADA accessibility standards."
    await evaluator.verify(
        claim=claim_ada,
        node=ada_leaf,
        sources=extraction.refs.specs_urls,
        additional_instruction="Look for compliance statements, certifications, or design documentation indicating ADA compliance in public/common areas."
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
    Evaluate an answer for the California mixed-use OZ/LEED/LIHTC task.
    """
    evaluator = Evaluator()
    evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Root as parallel aggregator
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

    # Extract structured information from the answer
    extraction: ProjectExtraction = await evaluator.extract(
        prompt=prompt_extract_project(),
        template_class=ProjectExtraction,
        extraction_name="project_extraction",
    )

    # Build verification tree and run checks
    await build_verification_tree(evaluator, extraction)

    # Return the evaluator's structured summary
    return evaluator.get_summary()