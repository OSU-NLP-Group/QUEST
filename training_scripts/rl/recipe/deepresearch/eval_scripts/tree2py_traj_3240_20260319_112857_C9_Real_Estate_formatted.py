import asyncio
import logging
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.llm_client.base_client import LLMClient
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy, VerificationNode


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "mixed_use_affordable_housing_ca_2024_2026"
TASK_DESCRIPTION = """
Identify a mixed-use affordable housing development project located in California that meets ALL of the following requirements: 
(1) Location and Type: The project must be in California and structured as a mixed-use development with at least 20% of ground floor area designated for commercial or retail use, and minimum 30 residential units, completed between 2024-2026. 
(2) Affordable Housing: The development must be 100% affordable housing targeting households at or below 60% Area Median Income (AMI), utilizing the 9% Low-Income Housing Tax Credit (LIHTC) program for new construction, serving people with disabilities and/or low-income households, and with at least 50% of units containing two or more bedrooms. 
(3) Sustainability: Must have achieved LEED Gold (60-79 points) or LEED Platinum (80+ points) certification after completing all mandatory prerequisites, and achieve Net Zero energy performance or demonstrate exceptional energy efficiency with renewable energy systems. 
(4) Compliance: Must meet Americans with Disabilities Act (ADA) accessibility standards with accessible dwelling units, comply with NFPA fire safety standards, obtain a Certificate of Occupancy, and comply with local mixed-use zoning requirements. 
(5) Financial Structure: Construction financing must meet a minimum Debt Service Coverage Ratio (DSCR) of 1.20 for affordable housing projects. 
Provide the project name, location (city), number of units, LEED certification level achieved, and documentation URLs that verify each major requirement category.
"""


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class ProjectExtraction(BaseModel):
    # Required identifying fields
    project_name: Optional[str] = None
    city: Optional[str] = None
    unit_count: Optional[str] = None
    leed_level: Optional[str] = None

    # Optional supporting statements (free text from answer; may be null)
    mixed_use_description: Optional[str] = None
    commercial_share_ground_floor: Optional[str] = None
    completion_year_or_date: Optional[str] = None

    affordability_statement: Optional[str] = None
    lihtc_program: Optional[str] = None
    populations_served: Optional[str] = None
    bedroom_mix_statement: Optional[str] = None

    energy_performance_statement: Optional[str] = None

    ada_compliance_statement: Optional[str] = None
    accessible_units_statement: Optional[str] = None
    nfpa_compliance_statement: Optional[str] = None
    certificate_of_occupancy_statement: Optional[str] = None
    zoning_compliance_statement: Optional[str] = None
    parking_requirements_statement: Optional[str] = None

    dscr_value_or_statement: Optional[str] = None

    # Documentation URLs by major requirement category
    location_and_type_urls: List[str] = Field(default_factory=list)
    affordable_housing_urls: List[str] = Field(default_factory=list)
    sustainability_urls: List[str] = Field(default_factory=list)
    compliance_urls: List[str] = Field(default_factory=list)
    financial_urls: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_project() -> str:
    return """
    Extract details for exactly ONE California mixed-use affordable housing project described in the answer that the answer claims meets the constraints.

    Return a JSON object with these fields:
    - project_name: string or null
    - city: string or null (California city)
    - unit_count: string or null (as written, do not convert; e.g., "45", "approximately 50", etc.)
    - leed_level: string or null (e.g., "LEED Gold", "LEED Platinum")

    Optional supportive statements (copy exact phrasing from the answer if present, else null):
    - mixed_use_description
    - commercial_share_ground_floor
    - completion_year_or_date
    - affordability_statement
    - lihtc_program
    - populations_served
    - bedroom_mix_statement
    - energy_performance_statement
    - ada_compliance_statement
    - accessible_units_statement
    - nfpa_compliance_statement
    - certificate_of_occupancy_statement
    - zoning_compliance_statement
    - parking_requirements_statement
    - dscr_value_or_statement

    Documentation URLs by category (MUST be actual URLs explicitly shown in the answer; if none, return an empty array):
    - location_and_type_urls: array of URLs supporting California location, mixed-use, >=20% ground-floor commercial share, >=30 units, completion 2024–2026.
    - affordable_housing_urls: array of URLs supporting 100% affordable at/under 60% AMI, 9% LIHTC new construction, serves disabilities/low-income, and >=50% two-bedroom units.
    - sustainability_urls: array of URLs supporting LEED Gold/Platinum certification and Net Zero or exceptional energy efficiency with renewables.
    - compliance_urls: array of URLs supporting ADA, accessible units, NFPA compliance, Certificate of Occupancy, local mixed-use zoning (and parking requirements if applicable).
    - financial_urls: array of URLs supporting construction financing DSCR >= 1.20 for the project.

    Extraction rules:
    - Only extract URLs that are explicitly present in the answer (including in markdown links). If the answer references a source without a URL, do not invent one.
    - If a field is not present in the answer, set it to null (for strings) or empty array (for URL lists).
    - If multiple projects are mentioned, extract the first one the answer explicitly presents as meeting all constraints.
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _has_any_url(urls: List[str]) -> bool:
    return bool(urls) and any(isinstance(u, str) and u.strip() for u in urls)


def _nonempty(s: Optional[str]) -> bool:
    return bool(s and isinstance(s, str) and s.strip())


# --------------------------------------------------------------------------- #
# Verification builders                                                       #
# --------------------------------------------------------------------------- #
async def build_required_output_fields(
    evaluator: Evaluator,
    parent: VerificationNode,
    ex: ProjectExtraction,
) -> VerificationNode:
    node = evaluator.add_parallel(
        id="Required_Output_Fields",
        desc="Answer includes the required identifying fields for the project.",
        parent=parent,
        critical=True,
    )

    evaluator.add_custom_node(
        result=_nonempty(ex.project_name),
        id="Project_Name_Provided",
        desc="Provides the project name.",
        parent=node,
        critical=True,
    )

    evaluator.add_custom_node(
        result=_nonempty(ex.city),
        id="City_Provided",
        desc="Provides the project location city.",
        parent=node,
        critical=True,
    )

    evaluator.add_custom_node(
        result=_nonempty(ex.unit_count),
        id="Unit_Count_Provided",
        desc="Provides the number of residential units.",
        parent=node,
        critical=True,
    )

    evaluator.add_custom_node(
        result=_nonempty(ex.leed_level),
        id="LEED_Level_Provided",
        desc="Provides the LEED certification level achieved.",
        parent=node,
        critical=True,
    )
    return node


async def build_documentation_urls(
    evaluator: Evaluator,
    parent: VerificationNode,
    ex: ProjectExtraction,
) -> Dict[str, VerificationNode]:
    node = evaluator.add_parallel(
        id="Documentation_URLs_By_Major_Category",
        desc="Provides documentation URLs that verify each major requirement category.",
        parent=parent,
        critical=True,
    )

    url_nodes = {}

    url_nodes["location"] = evaluator.add_custom_node(
        result=_has_any_url(ex.location_and_type_urls),
        id="URL_For_Location_And_Type",
        desc="Provides at least one URL supporting Location and Type requirements.",
        parent=node,
        critical=True,
    )

    url_nodes["affordable"] = evaluator.add_custom_node(
        result=_has_any_url(ex.affordable_housing_urls),
        id="URL_For_Affordable_Housing",
        desc="Provides at least one URL supporting Affordable Housing requirements.",
        parent=node,
        critical=True,
    )

    url_nodes["sustainability"] = evaluator.add_custom_node(
        result=_has_any_url(ex.sustainability_urls),
        id="URL_For_Sustainability",
        desc="Provides at least one URL supporting Sustainability requirements.",
        parent=node,
        critical=True,
    )

    url_nodes["compliance"] = evaluator.add_custom_node(
        result=_has_any_url(ex.compliance_urls),
        id="URL_For_Compliance",
        desc="Provides at least one URL supporting Compliance requirements.",
        parent=node,
        critical=True,
    )

    url_nodes["financial"] = evaluator.add_custom_node(
        result=_has_any_url(ex.financial_urls),
        id="URL_For_Financial",
        desc="Provides at least one URL supporting the DSCR/financial structure requirement.",
        parent=node,
        critical=True,
    )

    return url_nodes


async def build_location_and_type(
    evaluator: Evaluator,
    parent: VerificationNode,
    ex: ProjectExtraction,
    prereq_url_node: VerificationNode,
) -> VerificationNode:
    node = evaluator.add_parallel(
        id="Location_And_Type",
        desc="Meets California location, mixed-use, commercial share, unit minimum, and completion window constraints.",
        parent=parent,
        critical=True,
    )

    # Create leaves
    located_ca = evaluator.add_leaf(
        id="Located_In_California",
        desc="Project is located in California.",
        parent=node,
        critical=True,
    )
    mixed_use = evaluator.add_leaf(
        id="Mixed_Use_Development",
        desc="Project is a mixed-use development combining residential and commercial/retail uses.",
        parent=node,
        critical=True,
    )
    commercial_share = evaluator.add_leaf(
        id="Commercial_Share_Ground_Floor",
        desc="At least 20% of ground floor area is commercial/retail (non-residential).",
        parent=node,
        critical=True,
    )
    min_units = evaluator.add_leaf(
        id="Minimum_Residential_Units",
        desc="Includes at least 30 residential units.",
        parent=node,
        critical=True,
    )
    completion_window = evaluator.add_leaf(
        id="Completion_2024_2026",
        desc="Completed construction or achieved substantial completion between 2024–2026.",
        parent=node,
        critical=True,
    )

    project_label = ex.project_name or "the project"
    city_label = ex.city or "a city in California"

    claims = [
        (
            f"{project_label} is located in California (e.g., in {city_label}).",
            ex.location_and_type_urls,
            located_ca,
            "Confirm that the project is in California; any credible page indicating the site or city in CA is acceptable."
        ),
        (
            f"{project_label} is a mixed-use development that includes both residential housing and commercial/retail space.",
            ex.location_and_type_urls,
            mixed_use,
            "Look for descriptions like 'mixed-use', 'ground-floor retail', or combined residential/commercial program."
        ),
        (
            f"At least 20% of the ground floor area of {project_label} is dedicated to commercial or retail (non-residential) use.",
            ex.location_and_type_urls,
            commercial_share,
            "The evidence may state a percent or approximate ratio; accept clear statements meeting or exceeding 20% of ground floor commercial area."
        ),
        (
            f"{project_label} includes at least 30 residential units.",
            ex.location_and_type_urls,
            min_units,
            "Verify the unit count; accept language such as '30 units', '30 apartments', 'more than 30 units', etc."
        ),
        (
            f"{project_label} reached completion or substantial completion between 2024 and 2026.",
            ex.location_and_type_urls,
            completion_window,
            "Accept terms like 'completed', 'substantially completed', 'opened', or a Certificate of Occupancy in 2024–2026."
        ),
    ]

    await evaluator.batch_verify(
        claims_and_sources=claims,
        extra_prerequisites=[prereq_url_node],
    )

    return node


async def build_affordable_housing(
    evaluator: Evaluator,
    parent: VerificationNode,
    ex: ProjectExtraction,
    prereq_url_node: VerificationNode,
) -> VerificationNode:
    node = evaluator.add_parallel(
        id="Affordable_Housing",
        desc="Meets affordability, LIHTC, population served, and unit mix constraints.",
        parent=parent,
        critical=True,
    )

    all_100_aff = evaluator.add_leaf(
        id="All_Units_100pct_Affordable_AtOrBelow_60_AMI",
        desc="100% affordable housing targeting households at or below 60% AMI.",
        parent=node,
        critical=True,
    )
    uses_9pct = evaluator.add_leaf(
        id="Uses_9pct_LIHTC_New_Construction",
        desc="Utilizes the 9% LIHTC program for new construction.",
        parent=node,
        critical=True,
    )
    serves_pop = evaluator.add_leaf(
        id="Serves_Disabilities_AndOr_LowIncome",
        desc="Serves people with disabilities and/or low-income households.",
        parent=node,
        critical=True,
    )
    bedroom_mix = evaluator.add_leaf(
        id="Bedroom_Mix_Family_Sized",
        desc="At least 50% of units contain two or more bedrooms.",
        parent=node,
        critical=True,
    )

    project_label = ex.project_name or "the project"

    claims = [
        (
            f"100% of residential units at {project_label} are affordable to households at or below 60% AMI.",
            ex.affordable_housing_urls,
            all_100_aff,
            "Look for statements like '100% affordable' and income targeting at or below 60% AMI."
        ),
        (
            f"{project_label} utilizes 9% Low-Income Housing Tax Credits (LIHTC) for new construction.",
            ex.affordable_housing_urls,
            uses_9pct,
            "Accept mentions like '9% LIHTC', 'competitive tax credits', and that the project is new construction."
        ),
        (
            f"{project_label} serves people with disabilities and/or low-income households.",
            ex.affordable_housing_urls,
            serves_pop,
            "Evidence can include unit set-asides, supportive services, or priority populations."
        ),
        (
            f"At least 50% of the units at {project_label} have two or more bedrooms.",
            ex.affordable_housing_urls,
            bedroom_mix,
            "Look for explicit unit mix or summary stating that family-sized (2BR+) units are at least 50%."
        ),
    ]

    await evaluator.batch_verify(
        claims_and_sources=claims,
        extra_prerequisites=[prereq_url_node],
    )

    return node


async def build_sustainability(
    evaluator: Evaluator,
    parent: VerificationNode,
    ex: ProjectExtraction,
    prereq_url_node: VerificationNode,
) -> VerificationNode:
    node = evaluator.add_parallel(
        id="Sustainability",
        desc="Meets LEED and energy performance constraints.",
        parent=parent,
        critical=True,
    )

    leed = evaluator.add_leaf(
        id="LEED_Gold_Or_Platinum_Certified_With_Prerequisites",
        desc="Achieved LEED Gold (60–79 points) or LEED Platinum (80+ points) certification (with mandatory prerequisites completed).",
        parent=node,
        critical=True,
    )
    energy = evaluator.add_leaf(
        id="Energy_Performance_Meets_Requirement",
        desc="Documentation supports Net Zero energy or exceptional efficiency with renewables.",
        parent=node,
        critical=True,
    )

    project_label = ex.project_name or "the project"

    claims = [
        (
            f"{project_label} achieved LEED Gold or LEED Platinum certification.",
            ex.sustainability_urls,
            leed,
            "Accept any official LEED reference (USGBC, GBCI, project sheets) showing Gold or Platinum. Prerequisites are inherent to LEED certification."
        ),
        (
            f"{project_label} either achieves Net Zero energy performance or demonstrates exceptional energy efficiency with renewable energy systems.",
            ex.sustainability_urls,
            energy,
            "Accept terms like 'Net Zero', 'Zero Net Energy', 'all-electric with onsite solar and exceptional performance', or clear evidence of exceptional efficiency plus renewables."
        ),
    ]

    await evaluator.batch_verify(
        claims_and_sources=claims,
        extra_prerequisites=[prereq_url_node],
    )

    return node


async def build_compliance_and_financial(
    evaluator: Evaluator,
    parent: VerificationNode,
    ex: ProjectExtraction,
    prereq_compliance_node: VerificationNode,
    prereq_financial_node: VerificationNode,
) -> VerificationNode:
    node = evaluator.add_parallel(
        id="Compliance_And_Financial",
        desc="Meets ADA/accessibility, fire safety, occupancy, zoning/parking, and DSCR constraints.",
        parent=parent,
        critical=True,
    )

    # Compliance leaves
    ada = evaluator.add_leaf(
        id="ADA_Compliance",
        desc="Complies with ADA accessibility requirements.",
        parent=node,
        critical=True,
    )
    accessible = evaluator.add_leaf(
        id="Accessible_Dwelling_Units",
        desc="Includes accessible dwelling units designed for people with disabilities.",
        parent=node,
        critical=True,
    )
    nfpa = evaluator.add_leaf(
        id="NFPA_Fire_Safety_Compliance",
        desc="Meets NFPA fire safety standards, including sprinkler system requirements.",
        parent=node,
        critical=True,
    )
    coo = evaluator.add_leaf(
        id="Certificate_Of_Occupancy_Obtained",
        desc="Obtained a Certificate of Occupancy demonstrating code compliance.",
        parent=node,
        critical=True,
    )
    zoning = evaluator.add_leaf(
        id="Local_MixedUse_Zoning_Compliance",
        desc="Complies with local zoning requirements for mixed-use development.",
        parent=node,
        critical=True,
    )
    parking = evaluator.add_leaf(
        id="Parking_Requirements_Met",
        desc="Meets local parking requirements applicable to the mixed-use development (as stated in constraints).",
        parent=node,
        critical=True,
    )

    # Financial leaf
    dscr = evaluator.add_leaf(
        id="DSCR_At_Least_1_20",
        desc="Construction financing meets minimum DSCR of 1.20.",
        parent=node,
        critical=True,
    )

    project_label = ex.project_name or "the project"

    # Batch for compliance
    comp_claims = [
        (
            f"{project_label} complies with ADA accessibility requirements.",
            ex.compliance_urls,
            ada,
            "Look for references to ADA compliance, accessible design standards, or equivalent accessibility statements."
        ),
        (
            f"{project_label} includes accessible dwelling units designed for people with disabilities.",
            ex.compliance_urls,
            accessible,
            "Evidence may include unit set-asides, UFAS/ADA units, accessible features, or similar."
        ),
        (
            f"{project_label} meets NFPA fire safety standards (e.g., NFPA-compliant sprinklers or fire/life safety systems).",
            ex.compliance_urls,
            nfpa,
            "Accept clear references to NFPA standards or code-compliant sprinkler systems meeting NFPA."
        ),
        (
            f"{project_label} obtained a Certificate of Occupancy.",
            ex.compliance_urls,
            coo,
            "Look for explicit 'Certificate of Occupancy' or equivalent occupancy approval from the authority having jurisdiction."
        ),
        (
            f"{project_label} complies with local zoning requirements for mixed-use development.",
            ex.compliance_urls,
            zoning,
            "Look for planning approvals, zoning compliance statements, or entitlements indicating mixed-use zoning compliance."
        ),
        (
            f"{project_label} meets local parking requirements applicable to the mixed-use development.",
            ex.compliance_urls,
            parking,
            "Evidence can include approved parking counts, compliance statements, or entitlement conditions satisfied."
        ),
    ]
    await evaluator.batch_verify(
        claims_and_sources=comp_claims,
        extra_prerequisites=[prereq_compliance_node],
    )

    # Financial (separate prerequisite)
    await evaluator.verify(
        claim=f"The construction financing for {project_label} achieved a Debt Service Coverage Ratio (DSCR) of at least 1.20.",
        node=dscr,
        sources=ex.financial_urls,
        extra_prerequisites=[prereq_financial_node],
        additional_instruction="Accept any financing document or credible project source explicitly stating DSCR >= 1.20 for construction/permanent financing of this affordable housing project.",
    )

    return node


# --------------------------------------------------------------------------- #
# Main evaluation entry                                                       #
# --------------------------------------------------------------------------- #
async def evaluate_answer(
    client: LLMClient,
    answer: str,
    agent_name: str,
    answer_name: str,
    cache: CacheFileSys,
    semaphore: asyncio.Semaphore,
    logger: logging.Logger,
    model: str = "o4-mini",
) -> Dict:
    """
    Evaluate an answer for the California mixed-use affordable housing development task.
    """
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

    # Extract structured project information
    ex: ProjectExtraction = await evaluator.extract(
        prompt=prompt_extract_project(),
        template_class=ProjectExtraction,
        extraction_name="project_extraction",
    )

    # Add a top-level critical node for the project qualification rubric
    project_node = evaluator.add_parallel(
        id="Project_Qualification",
        desc="Identify ONE California mixed-use affordable housing development meeting all stated constraints, and provide required fields plus documentation URLs per major category.",
        parent=root,
        critical=True,
    )

    # Build Documentation URLs first so we can reference URL existence nodes as preconditions
    url_nodes = await build_documentation_urls(evaluator, project_node, ex)

    # Required output fields (presence checks)
    await build_required_output_fields(evaluator, project_node, ex)

    # Location and Type verification
    await build_location_and_type(
        evaluator,
        project_node,
        ex,
        prereq_url_node=url_nodes["location"],
    )

    # Affordable Housing verification
    await build_affordable_housing(
        evaluator,
        project_node,
        ex,
        prereq_url_node=url_nodes["affordable"],
    )

    # Sustainability verification
    await build_sustainability(
        evaluator,
        project_node,
        ex,
        prereq_url_node=url_nodes["sustainability"],
    )

    # Compliance and Financial verification
    await build_compliance_and_financial(
        evaluator,
        project_node,
        ex,
        prereq_compliance_node=url_nodes["compliance"],
        prereq_financial_node=url_nodes["financial"],
    )

    # Record small custom info summary (for debugging/traceability)
    evaluator.add_custom_info(
        info={
            "project_name": ex.project_name,
            "city": ex.city,
            "unit_count": ex.unit_count,
            "leed_level": ex.leed_level,
            "url_counts": {
                "location_and_type": len(ex.location_and_type_urls or []),
                "affordable_housing": len(ex.affordable_housing_urls or []),
                "sustainability": len(ex.sustainability_urls or []),
                "compliance": len(ex.compliance_urls or []),
                "financial": len(ex.financial_urls or []),
            },
        },
        info_type="extraction_summary",
    )

    # Return evaluation summary
    return evaluator.get_summary()