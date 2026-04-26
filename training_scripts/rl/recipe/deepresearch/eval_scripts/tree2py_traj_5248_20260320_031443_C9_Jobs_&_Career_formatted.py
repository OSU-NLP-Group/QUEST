import asyncio
import logging
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "educational_employment_research"
TASK_DESCRIPTION = """
Research and identify four different educational institutions across the United States that meet the following diversity requirements:
- Must be from at least three different U.S. states
- Must represent at least three different types of educational institutions (such as community college, K-12 school district, four-year public university, four-year private university)

For each of the four institutions, provide comprehensive employment and career information including:

Institution Identification:
- Institution name, type, and location (city and state)
- Direct link to the main employment/careers page

Employment System and Application Process:
- Name of the online application system used (e.g., AppliTrack, Workday, institution-specific portal)
- Direct link to the job search/listings page
- At least three major job categories available at the institution

Salary and Compensation:
- Evidence that salary information is publicly available (provide link to salary schedules, salary information page, or statement about salary transparency)
- Description of how salary information is organized (e.g., by position type, by experience level, by salary schedule)

Benefits Information:
- Direct link to benefits information page
- Confirmation of health insurance availability (medical, dental, vision)
- Confirmation of retirement plan availability

Contact and Support:
- HR department contact information (phone number or email address)
- Physical address of the institution or HR department

Diversity in Job Offerings:
- List at least four distinct job categories or position types currently advertised or typically available

Provide URL references for all information sections (employment main page, job listings, salary information, benefits, and contact details).
"""


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class Institution(BaseModel):
    # Identification
    name: Optional[str] = None
    type: Optional[str] = None
    city: Optional[str] = None
    state: Optional[str] = None
    main_career_url: Optional[str] = None

    # Employment system
    job_listings_url: Optional[str] = None
    application_system_name: Optional[str] = None
    job_categories: List[str] = Field(default_factory=list)          # >= 3
    diversity_categories: List[str] = Field(default_factory=list)    # >= 4

    # Salary
    salary_url: Optional[str] = None
    salary_availability: Optional[str] = None  # any textual confirmation the answer provides
    salary_organization: Optional[str] = None  # e.g., by schedule/step/position

    # Benefits
    benefits_url: Optional[str] = None
    health_insurance_confirmation: Optional[str] = None
    retirement_plan_confirmation: Optional[str] = None

    # Contact
    contact_url: Optional[str] = None
    hr_contact: Optional[str] = None  # phone or email
    physical_address: Optional[str] = None


class InstitutionsExtraction(BaseModel):
    institutions: List[Institution] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_institutions() -> str:
    return """
Extract details for up to four distinct U.S. educational institutions mentioned in the answer. Only extract information explicitly present in the answer. If more than four are provided, include only the first four. If fewer are provided, return what is available and set missing fields to null.

For each institution, extract the following fields exactly as they appear in the answer:

Identification
- name: Institution name
- type: Institution type (e.g., K-12 school district, community college, four-year public university, four-year private university, etc.)
- city: City
- state: State (two-letter abbreviation preferred if provided, otherwise full name)
- main_career_url: Direct link to the main employment/careers page

Employment system and categories
- job_listings_url: Direct link to the job search/listings page
- application_system_name: Name of the online application system used (e.g., Workday, PeopleAdmin, NeoEd, AppliTrack, or institution’s own portal)
- job_categories: List at least three major job categories available (e.g., faculty, administrative, classified, operations, student, etc.)
- diversity_categories: List at least four distinct job categories or position types (may overlap with job_categories if the answer lists four or more)

Salary and compensation
- salary_url: Direct link to salary schedules or salary information page
- salary_availability: Short text confirming salary info is public (if provided in the answer)
- salary_organization: How salary is organized (e.g., by position type, by experience/step/lane, by schedule)

Benefits
- benefits_url: Direct link to benefits information page
- health_insurance_confirmation: Text confirming availability of health insurance (medical/dental/vision)
- retirement_plan_confirmation: Text confirming availability of retirement plan(s)

Contact and support
- contact_url: URL that shows HR/contact information
- hr_contact: HR department contact info (phone number or email) as provided in the answer
- physical_address: Physical address of the institution or HR department as provided

Return a JSON object with:
{
  "institutions": [
    { ... up to 4 institutions with fields above ... }
  ]
}

Rules:
- Extract only URLs explicitly present in the answer. If missing a protocol, prepend http://
- If a field is not present, set it to null (or empty list for the two list fields).
- Do not invent or infer any content not present in the answer.
"""


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _nonempty(s: Optional[str]) -> bool:
    return bool(s and str(s).strip())


def _valid_url(u: Optional[str]) -> bool:
    if not _nonempty(u):
        return False
    u = u.strip()
    return u.startswith("http://") or u.startswith("https://")


def _merge_urls(*urls: Optional[str]) -> List[str]:
    merged: List[str] = []
    for u in urls:
        if _valid_url(u):
            if u not in merged:
                merged.append(u)
    return merged


def _fmt_list(items: List[str]) -> str:
    return ", ".join(items)


# --------------------------------------------------------------------------- #
# Verification for one institution                                            #
# --------------------------------------------------------------------------- #
async def verify_institution(evaluator: Evaluator, parent_node, inst: Institution, idx: int) -> None:
    inst_n = idx + 1
    inst_node = evaluator.add_parallel(
        id=f"Institution_{inst_n}",
        desc=f"Institution {inst_n}: educational institution with complete employment information",
        parent=parent_node,
        critical=False
    )

    # ---------------- Identification ----------------
    ident_node = evaluator.add_parallel(
        id=f"Institution_{inst_n}_Identification",
        desc=f"Institution {inst_n} basic identification and career page",
        parent=inst_node,
        critical=False
    )

    # Existence checks
    evaluator.add_custom_node(
        result=_valid_url(inst.main_career_url),
        id=f"Institution_{inst_n}_Main_Career_Page_URL_exists",
        desc="Main employment/careers page URL is provided",
        parent=ident_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=_nonempty(inst.name),
        id=f"Institution_{inst_n}_Name_exists",
        desc="Institution name is provided",
        parent=ident_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=_nonempty(inst.type),
        id=f"Institution_{inst_n}_Type_exists",
        desc="Institution type is provided",
        parent=ident_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=_nonempty(inst.city) and _nonempty(inst.state),
        id=f"Institution_{inst_n}_Location_exists",
        desc="Institution city and state are provided",
        parent=ident_node,
        critical=True
    )

    # Verify main career page URL content
    leaf = evaluator.add_leaf(
        id=f"Institution_{inst_n}_Main_Career_Page_URL",
        desc="Provide direct link to the main employment or careers page",
        parent=ident_node,
        critical=True
    )
    claim = (
        f"This URL is the main employment/careers page"
        + (f" for {inst.name}." if _nonempty(inst.name) else ".")
    )
    await evaluator.verify(
        claim=claim,
        node=leaf,
        sources=inst.main_career_url,
        additional_instruction="Accept pages clearly labeled 'Careers', 'Employment', 'Jobs', 'Work with us', or similar as the main careers page."
    )

    # Verify name against available sources (careers and/or contact)
    leaf = evaluator.add_leaf(
        id=f"Institution_{inst_n}_Name_Type_Location_name",
        desc="Institution name is supported by provided sources",
        parent=ident_node,
        critical=True
    )
    name_claim = f"The institution name is '{inst.name}'."
    await evaluator.verify(
        claim=name_claim,
        node=leaf,
        sources=_merge_urls(inst.main_career_url, inst.contact_url),
        additional_instruction="Allow minor formatting or legal suffix variations (e.g., 'USD', 'ISD', 'College', 'University')."
    )

    # Verify type
    leaf = evaluator.add_leaf(
        id=f"Institution_{inst_n}_Name_Type_Location_type",
        desc="Institution type is supported by provided sources",
        parent=ident_node,
        critical=True
    )
    type_claim = f"This institution is a {inst.type}."
    await evaluator.verify(
        claim=type_claim,
        node=leaf,
        sources=_merge_urls(inst.main_career_url, inst.contact_url),
        additional_instruction="Accept synonymous type descriptors (e.g., 'public university', 'state university', 'K-12 school district', 'community college'). Fuzzy match allowed."
    )

    # Verify location (city, state)
    leaf = evaluator.add_leaf(
        id=f"Institution_{inst_n}_Name_Type_Location_location",
        desc="Institution location (city, state) is supported by provided sources",
        parent=ident_node,
        critical=True
    )
    loc_claim = f"The institution is located in {inst.city}, {inst.state}."
    await evaluator.verify(
        claim=loc_claim,
        node=leaf,
        sources=_merge_urls(inst.contact_url, inst.main_career_url),
        additional_instruction="Look for address/contact footer, About or HR contact sections. Minor formatting differences are acceptable."
    )

    # ---------------- Employment System & Categories ----------------
    emp_node = evaluator.add_parallel(
        id=f"Institution_{inst_n}_Employment_System",
        desc=f"Institution {inst_n} application system and job categories",
        parent=inst_node,
        critical=False
    )

    evaluator.add_custom_node(
        result=_valid_url(inst.job_listings_url),
        id=f"Institution_{inst_n}_System_Reference_URL_exists",
        desc="Job search/listings page URL is provided",
        parent=emp_node,
        critical=True
    )
    leaf = evaluator.add_leaf(
        id=f"Institution_{inst_n}_System_Reference_URL",
        desc="Provide direct link to the job search or job listings page",
        parent=emp_node,
        critical=True
    )
    await evaluator.verify(
        claim="This URL displays current job listings or a search page for open positions.",
        node=leaf,
        sources=inst.job_listings_url,
        additional_instruction="Accept pages that clearly show job search, job listings, position postings, or talent/recruiting portal."
    )

    evaluator.add_custom_node(
        result=_nonempty(inst.application_system_name),
        id=f"Institution_{inst_n}_Application_System_Name_exists",
        desc="Application system name is provided",
        parent=emp_node,
        critical=True
    )
    leaf = evaluator.add_leaf(
        id=f"Institution_{inst_n}_Application_System_Name",
        desc="Identify the online application system used",
        parent=emp_node,
        critical=True
    )
    sys_claim = f"The online application system used is '{inst.application_system_name}'."
    await evaluator.verify(
        claim=sys_claim,
        node=leaf,
        sources=_merge_urls(inst.job_listings_url, inst.main_career_url),
        additional_instruction="Match common systems like Workday, PeopleAdmin, NeoEd, AppliTrack, PageUp, iCIMS, Oracle, or institution-specific portals. Allow branding or subdomain variants."
    )

    evaluator.add_custom_node(
        result=len(inst.job_categories) >= 3,
        id=f"Institution_{inst_n}_Job_Categories_count",
        desc="At least three major job categories are listed in the answer",
        parent=emp_node,
        critical=True
    )
    leaf = evaluator.add_leaf(
        id=f"Institution_{inst_n}_Job_Categories",
        desc="List at least three major job categories available at the institution",
        parent=emp_node,
        critical=True
    )
    jc_claim = f"The major job categories include: {_fmt_list(inst.job_categories)}."
    await evaluator.verify(
        claim=jc_claim,
        node=leaf,
        sources=inst.job_listings_url,
        additional_instruction="Verify that at least three categories (e.g., Faculty, Administrative/Professional, Classified/Support, Operations/Maintenance, Student/Temp, etc.) are present or implied on the listings page."
    )

    evaluator.add_custom_node(
        result=len(inst.diversity_categories) >= 4,
        id=f"Institution_{inst_n}_Diversity_of_Offerings_count",
        desc="At least four distinct job categories/position types are listed in the answer",
        parent=emp_node,
        critical=True
    )
    leaf = evaluator.add_leaf(
        id=f"Institution_{inst_n}_Diversity_of_Offerings",
        desc="List at least four distinct job categories or position types",
        parent=emp_node,
        critical=True
    )
    do_claim = f"At least four distinct job categories or position types include: {_fmt_list(inst.diversity_categories)}."
    await evaluator.verify(
        claim=do_claim,
        node=leaf,
        sources=inst.job_listings_url,
        additional_instruction="Confirm that these categories/position types are advertised or typically available; synonyms and family groupings are acceptable."
    )

    # ---------------- Salary Information ----------------
    sal_node = evaluator.add_parallel(
        id=f"Institution_{inst_n}_Salary_Information",
        desc=f"Institution {inst_n} salary and compensation details",
        parent=inst_node,
        critical=False
    )

    evaluator.add_custom_node(
        result=_valid_url(inst.salary_url),
        id=f"Institution_{inst_n}_Salary_Reference_URL_exists",
        desc="Salary information page URL is provided",
        parent=sal_node,
        critical=True
    )
    leaf = evaluator.add_leaf(
        id=f"Institution_{inst_n}_Salary_Reference_URL",
        desc="Provide link to salary schedules, salary information page, or documentation of salary availability",
        parent=sal_node,
        critical=True
    )
    await evaluator.verify(
        claim="This URL provides salary schedules or salary information for the institution.",
        node=leaf,
        sources=inst.salary_url,
        additional_instruction="Accept salary schedules, pay scales, pay grades, compensation pages, or HR documents that include salary/pay tables."
    )

    evaluator.add_custom_node(
        result=_nonempty(inst.salary_availability),
        id=f"Institution_{inst_n}_Salary_Availability_exists",
        desc="Salary availability confirmation text is provided in the answer",
        parent=sal_node,
        critical=True
    )
    leaf = evaluator.add_leaf(
        id=f"Institution_{inst_n}_Salary_Availability",
        desc="Confirm that salary information is publicly available",
        parent=sal_node,
        critical=True
    )
    await evaluator.verify(
        claim="Salary information is publicly available on the provided page.",
        node=leaf,
        sources=inst.salary_url,
        additional_instruction="Look for salary tables, schedules, pay bands, or explicit statements that salary information is published."
    )

    evaluator.add_custom_node(
        result=_nonempty(inst.salary_organization),
        id=f"Institution_{inst_n}_Salary_Organization_exists",
        desc="Salary organization description is provided in the answer",
        parent=sal_node,
        critical=True
    )
    leaf = evaluator.add_leaf(
        id=f"Institution_{inst_n}_Salary_Organization",
        desc="Describe how salary information is organized",
        parent=sal_node,
        critical=True
    )
    sal_org_claim = f"Salary information is organized by {inst.salary_organization}."
    await evaluator.verify(
        claim=sal_org_claim,
        node=leaf,
        sources=inst.salary_url,
        additional_instruction="Match organization such as 'by schedule/step/lane', 'by position type/classification', 'by experience level/years/education'."
    )

    # ---------------- Benefits ----------------
    ben_node = evaluator.add_parallel(
        id=f"Institution_{inst_n}_Benefits",
        desc=f"Institution {inst_n} benefits information",
        parent=inst_node,
        critical=False
    )

    evaluator.add_custom_node(
        result=_valid_url(inst.benefits_url),
        id=f"Institution_{inst_n}_Benefits_Reference_URL_exists",
        desc="Benefits information page URL is provided",
        parent=ben_node,
        critical=True
    )
    leaf = evaluator.add_leaf(
        id=f"Institution_{inst_n}_Benefits_Reference_URL",
        desc="Provide direct link to benefits information page",
        parent=ben_node,
        critical=True
    )
    await evaluator.verify(
        claim="This URL provides employee benefits information for the institution.",
        node=leaf,
        sources=inst.benefits_url,
        additional_instruction="Accept HR benefits overview, health & welfare, total rewards, or similar benefits pages."
    )

    evaluator.add_custom_node(
        result=_nonempty(inst.health_insurance_confirmation),
        id=f"Institution_{inst_n}_Health_Insurance_exists",
        desc="Health insurance confirmation text is provided in the answer",
        parent=ben_node,
        critical=True
    )
    leaf = evaluator.add_leaf(
        id=f"Institution_{inst_n}_Health_Insurance",
        desc="Confirm availability of health insurance benefits (medical, dental, vision)",
        parent=ben_node,
        critical=True
    )
    await evaluator.verify(
        claim="The benefits page indicates availability of medical, dental, and vision insurance coverage.",
        node=leaf,
        sources=inst.benefits_url,
        additional_instruction="Synonyms allowed (health/medical plan, dental coverage, vision care). Accept if all three are available, even if on subpages."
    )

    evaluator.add_custom_node(
        result=_nonempty(inst.retirement_plan_confirmation),
        id=f"Institution_{inst_n}_Retirement_Plan_exists",
        desc="Retirement plan confirmation text is provided in the answer",
        parent=ben_node,
        critical=True
    )
    leaf = evaluator.add_leaf(
        id=f"Institution_{inst_n}_Retirement_Plan",
        desc="Confirm availability of retirement plan benefits",
        parent=ben_node,
        critical=True
    )
    await evaluator.verify(
        claim="The benefits page indicates availability of employee retirement plan(s).",
        node=leaf,
        sources=inst.benefits_url,
        additional_instruction="Pension, 401(k), 403(b), 457, PERS, TRS, or state retirement systems are acceptable."
    )

    # ---------------- Contact & Support ----------------
    contact_node = evaluator.add_parallel(
        id=f"Institution_{inst_n}_Contact_Information",
        desc=f"Institution {inst_n} HR contact and address",
        parent=inst_node,
        critical=False
    )

    evaluator.add_custom_node(
        result=_valid_url(inst.contact_url),
        id=f"Institution_{inst_n}_Contact_Reference_URL_exists",
        desc="Contact/HR information page URL is provided",
        parent=contact_node,
        critical=True
    )
    leaf = evaluator.add_leaf(
        id=f"Institution_{inst_n}_Contact_Reference_URL",
        desc="URL reference showing contact information",
        parent=contact_node,
        critical=True
    )
    await evaluator.verify(
        claim="This URL shows HR or institutional contact information.",
        node=leaf,
        sources=inst.contact_url,
        additional_instruction="Accept HR contact page, general contact page that clearly indicates HR contacts, or a directory listing HR."
    )

    evaluator.add_custom_node(
        result=_nonempty(inst.hr_contact),
        id=f"Institution_{inst_n}_HR_Contact_exists",
        desc="HR contact (phone or email) is provided in the answer",
        parent=contact_node,
        critical=True
    )
    leaf = evaluator.add_leaf(
        id=f"Institution_{inst_n}_HR_Contact",
        desc="Provide HR department contact information (phone number or email address)",
        parent=contact_node,
        critical=True
    )
    hr_claim = f"The HR department contact information includes '{inst.hr_contact}'."
    await evaluator.verify(
        claim=hr_claim,
        node=leaf,
        sources=inst.contact_url,
        additional_instruction="Minor formatting/spacing differences acceptable. For phone numbers, allow punctuation variations."
    )

    evaluator.add_custom_node(
        result=_nonempty(inst.physical_address),
        id=f"Institution_{inst_n}_Physical_Address_exists",
        desc="Physical address is provided in the answer",
        parent=contact_node,
        critical=True
    )
    leaf = evaluator.add_leaf(
        id=f"Institution_{inst_n}_Physical_Address",
        desc="Provide physical address of the institution or HR department",
        parent=contact_node,
        critical=True
    )
    addr_claim = f"The physical address is '{inst.physical_address}'."
    await evaluator.verify(
        claim=addr_claim,
        node=leaf,
        sources=_merge_urls(inst.contact_url, inst.main_career_url),
        additional_instruction="Allow standard mailing address formatting differences; focus on matching street, city, state, and ZIP if available."
    )


# --------------------------------------------------------------------------- #
# Diversity verification (cross-institution)                                  #
# --------------------------------------------------------------------------- #
def compute_diversity_stats(institutions: List[Institution]) -> Dict[str, Any]:
    states = set()
    types = set()
    for inst in institutions:
        if _nonempty(inst.state):
            states.add(inst.state.strip())
        if _nonempty(inst.type):
            types.add(inst.type.strip().lower())
    return {
        "unique_states": sorted(states),
        "unique_types": sorted(types),
        "num_states": len(states),
        "num_types": len(types),
    }


async def add_diversity_checks(evaluator: Evaluator, parent_node, institutions: List[Institution]) -> None:
    div_node = evaluator.add_parallel(
        id="Diversity_Requirements",
        desc="Verify that the four institutions collectively satisfy geographic and institutional type diversity requirements",
        parent=parent_node,
        critical=False
    )

    stats = compute_diversity_stats(institutions)

    geo_ok = stats["num_states"] >= 3
    type_ok = stats["num_types"] >= 3

    evaluator.add_custom_node(
        result=geo_ok,
        id="Geographic_Diversity",
        desc="The four institutions are from at least three different U.S. states",
        parent=div_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=type_ok,
        id="Institution_Type_Diversity",
        desc="The four institutions include at least three different institution types",
        parent=div_node,
        critical=True
    )

    # Record stats for transparency
    evaluator.add_custom_info(
        info={
            "unique_states": stats["unique_states"],
            "unique_types": stats["unique_types"],
            "num_states": stats["num_states"],
            "num_types": stats["num_types"]
        },
        info_type="diversity_stats",
        info_name="diversity_statistics"
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
    # Initialize evaluator with a parallel root
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

    # Extract up to 4 institutions from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_institutions(),
        template_class=InstitutionsExtraction,
        extraction_name="institutions_extraction"
    )

    # Normalize to exactly 4 institutions (pad with empty if fewer)
    institutions: List[Institution] = list(extracted.institutions[:4])
    while len(institutions) < 4:
        institutions.append(Institution())

    # Add ground truth style meta info about the constraints
    evaluator.add_ground_truth({
        "requirements": {
            "min_unique_states": 3,
            "min_unique_types": 3,
            "institutions_required": 4
        }
    }, gt_type="task_requirements")

    # Diversity checks
    await add_diversity_checks(evaluator, root, institutions)

    # Build verification trees for each institution
    for idx, inst in enumerate(institutions):
        await verify_institution(evaluator, root, inst, idx)

    # Return structured summary
    return evaluator.get_summary()