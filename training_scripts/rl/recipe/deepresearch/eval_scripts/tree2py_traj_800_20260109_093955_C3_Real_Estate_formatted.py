import asyncio
import logging
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field
from obj_task_eval.llm_client.base_client import LLMClient

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "real_estate_research_chain"
TASK_DESCRIPTION = """A real estate development company was founded in 1993 in Houston, Texas. The company's headquarters were later relocated to Charleston, South Carolina in 1998. The founder of this company holds a Bachelor's degree in Petroleum Engineering from the University of Oklahoma and an MBA from Harvard Business School.

This developer completed their largest-ever project in 2024. The project is a $650 million mixed-use development located in Santa Ana, California, featuring 1,100 residential units and 40,000 square feet of commercial space on 14.5 acres.

An architecture firm based in Orange, California designed this project. The architecture firm was co-founded by two individuals in 1974, both of whom were graduates of Orange High School.

Identify the university where one of the co-founders of the architecture firm earned their architecture degree.
"""


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class DeveloperInfo(BaseModel):
    company_name: Optional[str] = None
    founded_year: Optional[str] = None
    founded_location: Optional[str] = None
    hq_relocation_year: Optional[str] = None
    hq_relocation_to: Optional[str] = None
    founder_name: Optional[str] = None
    founder_bs_field: Optional[str] = None
    founder_bs_university: Optional[str] = None
    founder_mba_school: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class ProjectInfo(BaseModel):
    name: Optional[str] = None
    is_largest_ever: Optional[str] = None  # "yes"/"no"/null (keep as string for robustness)
    completed_year: Optional[str] = None
    location_city: Optional[str] = None
    location_state: Optional[str] = None
    value: Optional[str] = None
    project_type: Optional[str] = None
    residential_units: Optional[str] = None
    commercial_sqft: Optional[str] = None
    site_area_acres: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class FirmInfo(BaseModel):
    name: Optional[str] = None
    base_city: Optional[str] = None
    base_state: Optional[str] = None
    designed_project_name: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class CoFoundersInfo(BaseModel):
    cofounding_year: Optional[str] = None
    cofounders: List[str] = Field(default_factory=list)
    both_orange_hs_graduates: Optional[str] = None  # "yes"/"no"/null
    selected_cofounder: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class DegreeInfo(BaseModel):
    selected_cofounder_name: Optional[str] = None
    architecture_degree_university: Optional[str] = None
    architecture_degree_type: Optional[str] = None  # e.g., B.Arch, M.Arch
    sources: List[str] = Field(default_factory=list)


class RealEstateResearchExtraction(BaseModel):
    developer: DeveloperInfo = Field(default_factory=DeveloperInfo)
    project: ProjectInfo = Field(default_factory=ProjectInfo)
    firm: FirmInfo = Field(default_factory=FirmInfo)
    founders: CoFoundersInfo = Field(default_factory=CoFoundersInfo)
    degree: DegreeInfo = Field(default_factory=DegreeInfo)
    global_sources: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_all() -> str:
    return """
    Extract structured information from the answer according to the following schema. Only extract information explicitly present in the answer (do not invent). If a field is missing, set it to null. Extract URLs explicitly cited in the answer text for the corresponding 'sources' arrays.

    Required JSON fields:
    - developer:
        company_name: the developer company's name
        founded_year: the year the company was founded
        founded_location: the city and state where the company was founded (e.g., "Houston, Texas")
        hq_relocation_year: the year the HQ relocated
        hq_relocation_to: the city and state where HQ relocated (e.g., "Charleston, South Carolina")
        founder_name: the founder's name if mentioned
        founder_bs_field: the field/major of the founder's bachelor's degree (e.g., "Petroleum Engineering")
        founder_bs_university: the university awarding the founder's bachelor's degree
        founder_mba_school: the MBA school (e.g., "Harvard Business School")
        sources: array of URLs that support developer-related facts
    - project:
        name: the project name
        is_largest_ever: "yes" if stated as developer's largest-ever project, else "no" or null
        completed_year: the year completed/opened
        location_city: project city (e.g., "Santa Ana")
        location_state: project state (e.g., "California")
        value: the stated value (e.g., "$650 million" or "650M")
        project_type: the type (e.g., "mixed-use")
        residential_units: e.g., "1,100"
        commercial_sqft: e.g., "40,000 square feet"
        site_area_acres: e.g., "14.5 acres"
        sources: array of URLs that support project-related facts
    - firm:
        name: architecture firm name
        base_city: city where firm is based (e.g., "Orange")
        base_state: state (e.g., "California")
        designed_project_name: the project this firm designed (should match project.name if stated)
        sources: array of URLs that support firm-related facts
    - founders:
        cofounding_year: e.g., "1974"
        cofounders: array of cofounder names
        both_orange_hs_graduates: "yes"/"no"/null as stated
        selected_cofounder: one cofounder chosen in the answer (whose degree university will be reported)
        sources: array of URLs that support cofounder-related facts
    - degree:
        selected_cofounder_name: the same selected cofounder name
        architecture_degree_university: the university for the selected cofounder's architecture degree
        architecture_degree_type: e.g., "Bachelor of Architecture", "B.Arch", "M.Arch" if stated
        sources: array of URLs that support the degree facts
    - global_sources: array of any URLs cited anywhere in the answer
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def combine_sources(*lists: List[str]) -> List[str]:
    uniq = []
    seen = set()
    for lst in lists:
        for u in lst or []:
            if not u:
                continue
            s = u.strip()
            if s and s not in seen:
                seen.add(s)
                uniq.append(s)
    return uniq


# --------------------------------------------------------------------------- #
# Verification step builders                                                  #
# --------------------------------------------------------------------------- #
async def verify_step1_developer(evaluator: Evaluator, parent_node, info: DeveloperInfo) -> None:
    step_node = evaluator.add_parallel(
        id="Step1_Developer",
        desc="Identify the real estate developer described in the prompt and verify its founder’s education constraints",
        parent=parent_node,
        critical=False
    )

    dev_urls = combine_sources(info.sources)

    # Developer Identity
    node_identity = evaluator.add_leaf(
        id="Developer_Identity",
        desc="Provide the developer company name matching the described developer",
        parent=step_node,
        critical=True
    )
    claim_identity = f"The developer that matches the described constraints is '{info.company_name}'."
    # Founded Year 1993
    node_year = evaluator.add_leaf(
        id="Developer_Founded_Year_1993",
        desc="Verify the developer was founded in 1993",
        parent=step_node,
        critical=True
    )
    claim_year = f"The company '{info.company_name}' was founded in 1993."
    # Founded Location Houston, TX
    node_loc = evaluator.add_leaf(
        id="Developer_Founded_Location_Houston_TX",
        desc="Verify the developer was originally founded in Houston, Texas",
        parent=step_node,
        critical=True
    )
    claim_loc = f"The company '{info.company_name}' was founded in Houston, Texas."
    # HQ Relocation Charleston, 1998
    node_hq = evaluator.add_leaf(
        id="Developer_HQ_Relocation_Charleston_1998",
        desc="Verify the developer’s headquarters were relocated to Charleston, South Carolina in 1998",
        parent=step_node,
        critical=True
    )
    claim_hq = f"The headquarters of '{info.company_name}' were relocated to Charleston, South Carolina in 1998."
    # Founder BS Petroleum Eng @ University of Oklahoma
    node_bs = evaluator.add_leaf(
        id="Founder_BS_PetroleumEng_UOklahoma",
        desc="Verify the developer’s founder holds a Bachelor's degree in Petroleum Engineering from the University of Oklahoma",
        parent=step_node,
        critical=True
    )
    if info.founder_name:
        claim_bs = f"The founder of '{info.company_name}', {info.founder_name}, holds a Bachelor's degree in Petroleum Engineering from the University of Oklahoma."
    else:
        claim_bs = f"The founder of '{info.company_name}' holds a Bachelor's degree in Petroleum Engineering from the University of Oklahoma."
    # Founder MBA Harvard Business School
    node_mba = evaluator.add_leaf(
        id="Founder_MBA_Harvard",
        desc="Verify the developer’s founder holds an MBA from Harvard Business School",
        parent=step_node,
        critical=True
    )
    if info.founder_name:
        claim_mba = f"The founder of '{info.company_name}', {info.founder_name}, holds an MBA from Harvard Business School."
    else:
        claim_mba = f"The founder of '{info.company_name}' holds an MBA from Harvard Business School."

    claims_and_sources = [
        (claim_identity, dev_urls, node_identity, "Evaluate if the named company corresponds to the developer described; rely on the URLs for confirmation."),
        (claim_year, dev_urls, node_year, "Confirm the founding year is 1993."),
        (claim_loc, dev_urls, node_loc, "Confirm the founding location is Houston, Texas."),
        (claim_hq, dev_urls, node_hq, "Confirm the HQ relocation destination (Charleston, South Carolina) and year (1998)."),
        (claim_bs, dev_urls, node_bs, "Accept reasonable phrasing variations for 'Bachelor of Science', 'B.S.', 'Bachelors', and field 'Petroleum Engineering'."),
        (claim_mba, dev_urls, node_mba, "Accept phrasing variants; ensure the MBA is specifically from Harvard Business School."),
    ]
    await evaluator.batch_verify(claims_and_sources)


async def verify_step2_project(evaluator: Evaluator, parent_node, dev: DeveloperInfo, proj: ProjectInfo) -> None:
    step_node = evaluator.add_parallel(
        id="Step2_Project",
        desc="Identify the developer’s largest-ever project and verify all project constraints",
        parent=parent_node,
        critical=False
    )

    proj_urls = combine_sources(proj.sources, dev.sources)

    # Project Identity
    node_p_ident = evaluator.add_leaf(
        id="Project_Identity",
        desc="Provide the project name matching the described project",
        parent=step_node,
        critical=True
    )
    claim_p_ident = f"The project matching the described constraints is '{proj.name}'."

    # Largest-Ever
    node_p_largest = evaluator.add_leaf(
        id="Project_Largest_Ever",
        desc="Verify this project is the developer’s largest-ever project",
        parent=step_node,
        critical=True
    )
    claim_p_largest = f"The project '{proj.name}' is the largest-ever project undertaken by '{dev.company_name}'."

    # Completed 2024
    node_p_2024 = evaluator.add_leaf(
        id="Project_Completed_2024",
        desc="Verify the project was completed/opened in 2024",
        parent=step_node,
        critical=True
    )
    claim_p_2024 = f"The project '{proj.name}' was completed or opened in 2024."

    # Location Santa Ana, CA
    node_p_loc = evaluator.add_leaf(
        id="Project_Location_Santa_Ana_CA",
        desc="Verify the project is located in Santa Ana, California",
        parent=step_node,
        critical=True
    )
    claim_p_loc = f"The project '{proj.name}' is located in Santa Ana, California."

    # Value $650M
    node_p_val = evaluator.add_leaf(
        id="Project_Value_650M",
        desc="Verify the project is valued at $650 million",
        parent=step_node,
        critical=True
    )
    claim_p_val = f"The project '{proj.name}' is valued at approximately $650 million."

    # Mixed-use
    node_p_type = evaluator.add_leaf(
        id="Project_Type_Mixed_Use",
        desc="Verify the project is a mixed-use development",
        parent=step_node,
        critical=True
    )
    claim_p_type = f"The project '{proj.name}' is a mixed-use development."

    # Residential units 1,100
    node_p_units = evaluator.add_leaf(
        id="Project_Residential_Units_1100",
        desc="Verify the project includes 1,100 residential units",
        parent=step_node,
        critical=True
    )
    claim_p_units = f"The project '{proj.name}' includes around 1,100 residential units (accept reasonable formatting like 1,100 or 1100)."

    # Commercial 40,000 sqft
    node_p_sqft = evaluator.add_leaf(
        id="Project_Commercial_Space_40000_SqFt",
        desc="Verify the project includes 40,000 square feet of commercial space",
        parent=step_node,
        critical=True
    )
    claim_p_sqft = f"The project '{proj.name}' includes approximately 40,000 square feet of commercial space."

    # Site area 14.5 acres
    node_p_acres = evaluator.add_leaf(
        id="Project_Site_Area_14_5_Acres",
        desc="Verify the project is located on 14.5 acres",
        parent=step_node,
        critical=True
    )
    claim_p_acres = f"The project '{proj.name}' occupies a site of approximately 14.5 acres."

    claims_and_sources = [
        (claim_p_ident, proj_urls, node_p_ident, "Confirm the project name corresponds to the described constraints."),
        (claim_p_largest, proj_urls, node_p_largest, "Accept equivalent phrases like 'largest project to date' or 'largest-ever'."),
        (claim_p_2024, proj_urls, node_p_2024, "Verify completion/opening year is 2024."),
        (claim_p_loc, proj_urls, node_p_loc, "Verify the location city and state are Santa Ana, California."),
        (claim_p_val, proj_urls, node_p_val, "Accept '$650 million', '650M', or similar phrasing indicating $650,000,000."),
        (claim_p_type, proj_urls, node_p_type, "Mixed-use should clearly include multiple use types (e.g., residential + commercial)."),
        (claim_p_units, proj_urls, node_p_units, "Allow standard numeric formatting variations (1,100 vs 1100)."),
        (claim_p_sqft, proj_urls, node_p_sqft, "Allow 'sf', 'sq ft', 'square feet' equivalence; approximate phrasing acceptable."),
        (claim_p_acres, proj_urls, node_p_acres, "Allow slight rounding if stated as ~14.5 acres."),
    ]
    await evaluator.batch_verify(claims_and_sources)


async def verify_step3_firm(evaluator: Evaluator, parent_node, firm: FirmInfo, proj: ProjectInfo) -> None:
    step_node = evaluator.add_parallel(
        id="Step3_Architecture_Firm",
        desc="Identify the architecture firm that designed the project and verify firm constraints",
        parent=parent_node,
        critical=False
    )

    firm_urls = combine_sources(firm.sources, proj.sources)

    # Firm Identity
    node_f_ident = evaluator.add_leaf(
        id="Architecture_Firm_Identity",
        desc="Provide the architecture firm name that designed the identified project",
        parent=step_node,
        critical=True
    )
    claim_f_ident = f"The architecture firm that designed the project '{proj.name}' is '{firm.name}'."

    # Firm Based in Orange, CA
    node_f_base = evaluator.add_leaf(
        id="Architecture_Firm_Based_In_Orange_CA",
        desc="Verify the architecture firm is based in Orange, California",
        parent=step_node,
        critical=True
    )
    claim_f_base = f"The architecture firm '{firm.name}' is based in Orange, California."

    # Firm Designed Project
    node_f_designed = evaluator.add_leaf(
        id="Architecture_Firm_Designed_Project",
        desc="Verify the architecture firm designed the identified project",
        parent=step_node,
        critical=True
    )
    claim_f_designed = f"The architecture firm '{firm.name}' designed the project '{proj.name}'."

    claims_and_sources = [
        (claim_f_ident, firm_urls, node_f_ident, "Confirm the firm identity and association with the project."),
        (claim_f_base, firm_urls, node_f_base, "Verify city and state for the firm's base location."),
        (claim_f_designed, firm_urls, node_f_designed, "Verify firm attribution as the designer of the specified project."),
    ]
    await evaluator.batch_verify(claims_and_sources)


async def verify_step4_cofounders(evaluator: Evaluator, parent_node, firm: FirmInfo, founders: CoFoundersInfo) -> None:
    step_node = evaluator.add_parallel(
        id="Step4_CoFounders",
        desc="Verify co-founder constraints for the architecture firm and select a co-founder for the education query",
        parent=parent_node,
        critical=False
    )

    cf_urls = combine_sources(founders.sources, firm.sources)

    # Co-founded in 1974
    node_cf_year = evaluator.add_leaf(
        id="Firm_CoFounded_1974",
        desc="Verify the architecture firm was co-founded in 1974",
        parent=step_node,
        critical=True
    )
    claim_cf_year = f"The architecture firm '{firm.name}' was co-founded in 1974."

    # Two co-founders
    node_cf_two = evaluator.add_leaf(
        id="Firm_Has_Two_CoFounders",
        desc="Verify the architecture firm was co-founded by two individuals",
        parent=step_node,
        critical=True
    )
    claim_cf_two = f"The architecture firm '{firm.name}' was co-founded by two individuals."

    # Both graduates of Orange High School
    node_cf_ohs = evaluator.add_leaf(
        id="Both_CoFounders_Orange_HS_Graduates",
        desc="Verify both co-founders of the architecture firm were graduates of Orange High School",
        parent=step_node,
        critical=True
    )
    claim_cf_ohs = f"Both co-founders of '{firm.name}' graduated from Orange High School."

    # Selected co-founder identity
    node_sel_ident = evaluator.add_leaf(
        id="Selected_CoFounder_Identity",
        desc="Identify one of the co-founders (the one whose architecture degree university will be reported)",
        parent=step_node,
        critical=True
    )
    claim_sel_ident = f"The selected co-founder is '{founders.selected_cofounder}'."

    # Selected is co-founder
    node_sel_is_cf = evaluator.add_leaf(
        id="Selected_Is_CoFounder",
        desc="Verify the selected individual is one of the firm’s co-founders",
        parent=step_node,
        critical=True
    )
    claim_sel_is_cf = f"'{founders.selected_cofounder}' is one of the co-founders of '{firm.name}'."

    claims_and_sources = [
        (claim_cf_year, cf_urls, node_cf_year, "Confirm the co-founding year is 1974."),
        (claim_cf_two, cf_urls, node_cf_two, "Confirm the firm has exactly two co-founders."),
        (claim_cf_ohs, cf_urls, node_cf_ohs, "Confirm both co-founders graduated from Orange High School."),
        (claim_sel_ident, cf_urls, node_sel_ident, "Confirm the identity of the selected co-founder."),
        (claim_sel_is_cf, cf_urls, node_sel_is_cf, "Verify the selected individual is indeed a co-founder of the firm."),
    ]
    await evaluator.batch_verify(claims_and_sources)


async def verify_step5_education(evaluator: Evaluator, parent_node, founders: CoFoundersInfo, degree: DegreeInfo) -> None:
    step_node = evaluator.add_parallel(
        id="Step5_Education",
        desc="Provide the university where the selected co-founder earned their architecture degree",
        parent=parent_node,
        critical=True
    )

    deg_urls = combine_sources(degree.sources, founders.sources)

    # Architecture Degree University
    node_deg_uni = evaluator.add_leaf(
        id="Architecture_Degree_University",
        desc="State the university where the selected co-founder earned their architecture degree",
        parent=step_node,
        critical=True
    )
    claim_deg_uni = f"The selected co-founder '{degree.selected_cofounder_name}' earned an architecture degree from '{degree.architecture_degree_university}'."

    # Degree is architecture and belongs to selected
    node_deg_belongs = evaluator.add_leaf(
        id="Degree_Is_Architecture_And_Belongs_To_Selected",
        desc="Verify the stated university corresponds to an architecture degree earned by the selected co-founder (not a different degree/person)",
        parent=step_node,
        critical=True
    )
    if degree.architecture_degree_type:
        claim_deg_belongs = f"At '{degree.architecture_degree_university}', '{degree.selected_cofounder_name}' earned an architecture degree (e.g., {degree.architecture_degree_type})."
    else:
        claim_deg_belongs = f"At '{degree.architecture_degree_university}', '{degree.selected_cofounder_name}' earned an architecture degree."

    claims_and_sources = [
        (claim_deg_uni, deg_urls, node_deg_uni, "Accept common variants like 'Bachelor of Architecture (B.Arch)' or 'Master of Architecture (M.Arch)'. Ensure the degree pertains to architecture."),
        (claim_deg_belongs, deg_urls, node_deg_belongs, "Verify the degree subject is architecture and the recipient is the selected co-founder; avoid confusing with other individuals or degrees."),
    ]
    await evaluator.batch_verify(claims_and_sources)


# --------------------------------------------------------------------------- #
# Main evaluation entry point                                                 #
# --------------------------------------------------------------------------- #
async def evaluate_answer(
    client: LLMClient,
    answer: str,
    agent_name: str,
    answer_name: str,
    cache: CacheFileSys,
    semaphore: asyncio.Semaphore,
    logger: logging.Logger,
    model: str = "o4-mini"
) -> Dict[str, Any]:
    """
    Evaluate an agent's answer for the research chain task.
    Builds a sequential verification tree covering developer -> project -> firm -> co-founders -> education.
    """
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.SEQUENTIAL,  # Research chain proceeds step by step
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

    research_chain = evaluator.add_sequential(
        id="Research_Chain",
        desc="Identify the correct developer, its largest project, the designing architecture firm, a co-founder, and the university of that co-founder's architecture degree",
        parent=root,
        critical=False  # Keep non-critical so children can mix critical/soft checks
    )

    # 1) Extraction
    extracted: RealEstateResearchExtraction = await evaluator.extract(
        prompt=prompt_extract_all(),
        template_class=RealEstateResearchExtraction,
        extraction_name="structured_research_chain"
    )

    # 2) Verification steps in sequence
    await verify_step1_developer(evaluator, research_chain, extracted.developer)
    await verify_step2_project(evaluator, research_chain, extracted.developer, extracted.project)
    await verify_step3_firm(evaluator, research_chain, extracted.firm, extracted.project)
    await verify_step4_cofounders(evaluator, research_chain, extracted.firm, extracted.founders)
    await verify_step5_education(evaluator, research_chain, extracted.founders, extracted.degree)

    # 3) Return summary
    return evaluator.get_summary()