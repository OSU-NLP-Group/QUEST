import asyncio
import logging
from typing import Any, Dict, List, Optional, Set

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "district_or_tx_hiring_2026"
TASK_DESCRIPTION = """
You are a newly certified elementary school teacher from California looking to relocate to either Oregon or Texas for the 2026-2027 school year. You want to find a district with transparent hiring processes and comprehensive information for out-of-state applicants.

Identify ONE public school district in Oregon or Texas that meets ALL of the following requirements:

1. Is a public school district located in Oregon or Texas
2. Has elementary teacher job openings posted for the 2026-2027 school year
3. Publicly provides starting salary information for first-year teachers
4. Offers an online application system for job seekers
5. Identifies the specific online application platform used (e.g., AppliTrack, Frontline Recruitment)
6. Clearly states teacher certification or licensure requirements for their state
7. Provides information about accepting or processing out-of-state teaching credentials
8. Conducted or has scheduled at least one job fair or recruitment event in 2026
9. Lists contact information (phone or email) for the Human Resources or employment office
10. Posts multiple categories of employment positions (such as certified teachers, administrators, and support staff)
11. Provides information about substitute teaching opportunities or requirements
12. Lists specific required qualifications for teachers (bachelor's degree, preparation program, certification exams)
13. Mentions background check and fingerprinting requirements for teaching candidates
14. Can be verified as meeting all criteria through official district websites or government sources

Provide the district name, state location, and specific evidence with reference URLs demonstrating how it satisfies each of the 14 criteria.
"""


# --------------------------------------------------------------------------- #
# Extraction models                                                           #
# --------------------------------------------------------------------------- #
class Evidence(BaseModel):
    urls: List[str] = Field(default_factory=list)
    text: Optional[str] = None


class PlatformEvidence(Evidence):
    platform_name: Optional[str] = None  # e.g., Frontline, AppliTrack


class CategoriesEvidence(Evidence):
    categories: List[str] = Field(default_factory=list)  # e.g., Teachers, Administrators, Classified


class QualificationsEvidence(Evidence):
    qualifications: List[str] = Field(default_factory=list)  # e.g., bachelor's degree, program, exams


class HRContactEvidence(Evidence):
    contact: Optional[str] = None  # email or phone string from the answer


class DistrictExtraction(BaseModel):
    district_name: Optional[str] = None
    state: Optional[str] = None  # "Oregon" / "Texas" or abbreviations like "OR"/"TX"

    c1: Evidence = Field(default_factory=Evidence)
    c2: Evidence = Field(default_factory=Evidence)
    c3: Evidence = Field(default_factory=Evidence)
    c4: Evidence = Field(default_factory=Evidence)
    c5: PlatformEvidence = Field(default_factory=PlatformEvidence)
    c6: Evidence = Field(default_factory=Evidence)
    c7: Evidence = Field(default_factory=Evidence)
    c8: Evidence = Field(default_factory=Evidence)
    c9: HRContactEvidence = Field(default_factory=HRContactEvidence)
    c10: CategoriesEvidence = Field(default_factory=CategoriesEvidence)
    c11: Evidence = Field(default_factory=Evidence)
    c12: QualificationsEvidence = Field(default_factory=QualificationsEvidence)
    c13: Evidence = Field(default_factory=Evidence)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_district() -> str:
    return """
Extract the following structured information from the answer. Return exactly and only the JSON specified by the schema. Use null for any missing field. For URLs, extract only explicit URLs present in the answer text (including markdown links).

Fields to extract:
- district_name: The single district named in the answer.
- state: The state the district is located in (Texas or Oregon). Accept abbreviations (TX/OR) if used in the answer.

For each criterion C1 to C13, extract:
- urls: Array of URLs the answer cites specifically as evidence for that criterion. If none, return an empty array.
- text: A short 1–2 sentence summary (from the answer) of the claim being made for that criterion. If not included, return null.

Additionally extract specialized fields for some criteria if present in the answer:
- c5.platform_name: If the answer names the online application platform (e.g., Frontline, AppliTrack), include the platform name; else null.
- c10.categories: If the answer lists multiple position categories (e.g., Teachers, Administration, Classified), include all categories mentioned; else an empty array.
- c12.qualifications: If the answer lists specific teacher qualifications (e.g., bachelor's degree, teacher prep program, exams), include all mentioned; else an empty array.
- c9.contact: If the answer specifies an HR contact email or phone number, capture it; else null.

Mapping of criteria:
- c1: District is a public school district in Oregon or Texas.
- c2: Elementary teacher openings for the 2026–2027 school year are posted.
- c3: Starting salary for first-year teachers is publicly provided.
- c4: Online application system exists.
- c5: The specific online application platform is identified (plus platform_name field).
- c6: State certification/licensure requirements are clearly stated.
- c7: Info about accepting/processing out-of-state credentials is provided.
- c8: At least one 2026 job fair/recruitment event (held or scheduled) is shown.
- c9: HR/employment office contact info (phone or email) is listed (plus contact field).
- c10: Multiple job position categories are posted (plus categories list).
- c11: Substitute teaching info (opportunities/requirements) is provided.
- c12: Specific required teacher qualifications are listed (plus qualifications list).
- c13: Background check and fingerprinting requirements are mentioned.

Important URL rules:
- Extract only valid URLs explicitly present in the answer. If missing protocol, prepend http://
- Do not invent URLs.

Return JSON matching the schema of the provided pydantic model DistrictExtraction.
"""


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def normalize_state(value: Optional[str]) -> Optional[str]:
    if not value:
        return None
    v = value.strip().lower()
    if v in {"or", "oregon"}:
        return "Oregon"
    if v in {"tx", "texas"}:
        return "Texas"
    return None


def nonempty_urls(urls: Optional[List[str]]) -> bool:
    return bool(urls and len([u for u in urls if isinstance(u, str) and u.strip()]) > 0)


def unique_urls(url_lists: List[List[str]]) -> List[str]:
    seen: Set[str] = set()
    result: List[str] = []
    for lst in url_lists:
        for u in lst or []:
            uu = (u or "").strip()
            if uu and uu not in seen:
                seen.add(uu)
                result.append(uu)
    return result


# --------------------------------------------------------------------------- #
# Verification builders                                                       #
# --------------------------------------------------------------------------- #
async def add_response_identity_checks(evaluator: Evaluator, parent, ex: DistrictExtraction) -> None:
    node = evaluator.add_parallel(
        id="Response_Identity",
        desc="Response includes the district name and the state (Oregon or Texas).",
        parent=parent,
        critical=True
    )

    # District name provided
    evaluator.add_custom_node(
        result=bool(ex.district_name and ex.district_name.strip()),
        id="District_Name_Provided",
        desc="District name is provided in the response.",
        parent=node,
        critical=True
    )

    # State provided and valid (OR/TX)
    norm_state = normalize_state(ex.state)
    evaluator.add_custom_node(
        result=bool(norm_state),
        id="State_Provided_Valid",
        desc="State is provided and is Oregon or Texas (OR/TX).",
        parent=node,
        critical=True
    )


async def add_criterion_with_urls(
    evaluator: Evaluator,
    parent,
    crit_id: str,
    meets_desc: str,
    urls: List[str],
    claim: str,
    additional_instruction: str
) -> None:
    """
    Build a parallel, critical criterion node with:
      - Meets requirement (verified by provided URLs)
      - URL evidence provided (existence check)
    """
    node = evaluator.add_parallel(
        id=crit_id,
        desc=meets_desc,
        parent=parent,
        critical=True
    )

    # URL evidence provided (critical gate)
    evaluator.add_custom_node(
        result=nonempty_urls(urls),
        id=f"{crit_id}_URL_Evidence_Provided",
        desc=f"Provides at least one URL as evidence supporting {crit_id.split('_')[0].upper()}",
        parent=node,
        critical=True
    )

    # Meets requirement (critical)
    leaf = evaluator.add_leaf(
        id=f"{crit_id}_Meets_Requirement",
        desc=meets_desc,
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim=claim,
        node=leaf,
        sources=urls,
        additional_instruction=additional_instruction
    )


def build_c14_instructions() -> str:
    return (
        "Determine whether the page is an OFFICIAL district website or a government/education-agency source. "
        "Strong signals include: .gov, state education agencies (e.g., TEA for Texas, ODE for Oregon), regional education service districts, or official district domains "
        "(often containing k12, isd, schooldistrict, or .org/.us domains clearly branded for the district). "
        "Vendor-hosted application portals directly used by the district (e.g., Frontline Recruiting & Hiring, AppliTrack) can be considered official for hiring if the page clearly represents the district's own portal. "
        "Do not consider third-party news, blogs, general job aggregators, or unofficial community sites as official."
    )


async def add_c14_official_sources_check(
    evaluator: Evaluator,
    parent,
    all_urls: List[str]
) -> None:
    """
    C14: All evidence URLs used to justify C1–C13 are official district or government sources.
    Implemented as a parallel critical node with one leaf per URL.
    """
    c14_node = evaluator.add_parallel(
        id="C14_Verifiable_Through_Official_Sources",
        desc="All provided evidence URLs used to justify C1–C13 are official district/government/education-agency sources.",
        parent=parent,
        critical=True
    )

    if not all_urls:
        evaluator.add_custom_node(
            result=False,
            id="C14_No_URLs_Found",
            desc="No evidence URLs were provided for any criterion (thus cannot verify official sources).",
            parent=c14_node,
            critical=True
        )
        return

    claims_and_sources = []
    for idx, url in enumerate(all_urls):
        leaf = evaluator.add_leaf(
            id=f"C14_URL_{idx+1}_Official",
            desc=f"URL is an official district or government/education-agency source",
            parent=c14_node,
            critical=True
        )
        claim = (
            "This page is an official district website or a government/education-agency source (not an unofficial third-party)."
        )
        claims_and_sources.append((claim, url, leaf, build_c14_instructions()))

    # Batch verify for efficiency
    await evaluator.batch_verify(claims_and_sources)


# --------------------------------------------------------------------------- #
# Main evaluation routine per-criterion                                       #
# --------------------------------------------------------------------------- #
async def build_verification_tree(evaluator: Evaluator, ex: DistrictExtraction) -> None:
    # Top-level critical node that aggregates all checks
    top = evaluator.add_parallel(
        id="District_Selection_and_Evidence",
        desc="Identify ONE public school district in Oregon or Texas that meets all requirements and provide URL evidence for each requirement; ensure evidence is from official district/government sources.",
        parent=evaluator.root,
        critical=True
    )

    # Response identity
    await add_response_identity_checks(evaluator, top, ex)

    # Normalize state for readable claims
    norm_state = normalize_state(ex.state) or (ex.state or "").strip()
    district_name = (ex.district_name or "").strip()

    # C1: Public district in OR/TX
    await add_criterion_with_urls(
        evaluator,
        top,
        "C1_Public_District_in_OR_or_TX",
        "District is a public school district located in Oregon or Texas.",
        ex.c1.urls,
        claim=f"{district_name} is a public school district located in {norm_state}.",
        additional_instruction="Accept 'Independent School District' (ISD) in Texas or 'School District' in Oregon as public. Verify location state explicitly."
    )

    # C2: Elementary openings 2026-2027
    await add_criterion_with_urls(
        evaluator,
        top,
        "C2_Elementary_Openings_2026_2027",
        "District has elementary teacher job openings posted for the 2026-2027 school year.",
        ex.c2.urls,
        claim="The district has posted elementary teacher job openings specifically for the 2026–2027 school year (allow variants like 2026-27, 26–27, SY 2026/27).",
        additional_instruction="Confirm 'Elementary' (or 'Elementary Teacher') appears with the 2026–2027 school year (e.g., '2026-27', '2026/27', 'SY 26-27')."
    )

    # C3: Starting salary public
    await add_criterion_with_urls(
        evaluator,
        top,
        "C3_Starting_Salary_Public",
        "District publicly provides starting salary information for first-year teachers.",
        ex.c3.urls,
        claim="The page publicly provides starting salary information for first-year teachers (explicit amount or salary schedule indicating starting step).",
        additional_instruction="Look for a salary schedule or explicit starting salary (e.g., Step 0) for teachers new to the district."
    )

    # C4: Online application system
    await add_criterion_with_urls(
        evaluator,
        top,
        "C4_Online_Application_System",
        "District offers an online application system for job seekers.",
        ex.c4.urls,
        claim="The district offers an online application system for job seekers to submit applications electronically.",
        additional_instruction="Verify there is an online system/portal for submitting job applications."
    )

    # C5: Application platform identified
    platform_name = (ex.c5.platform_name or "").strip()
    c5_claim = (
        f"The district uses the '{platform_name}' online application platform."
        if platform_name else
        "The district identifies the specific online application platform used for job applications (e.g., Frontline, AppliTrack)."
    )
    await add_criterion_with_urls(
        evaluator,
        top,
        "C5_Application_Platform_Identified",
        "District identifies the specific online application platform used (e.g., AppliTrack, Frontline Recruitment).",
        ex.c5.urls,
        claim=c5_claim,
        additional_instruction="Confirm the platform is explicitly named on the page. Accept synonyms/branding variants (e.g., 'Frontline Recruiting & Hiring')."
    )

    # C6: State certification/licensure requirements
    await add_criterion_with_urls(
        evaluator,
        top,
        "C6_State_Cert_or_Licensure_Requirements",
        "District clearly states teacher certification or licensure requirements for their state.",
        ex.c6.urls,
        claim=f"The district clearly states {norm_state} teacher certification/licensure requirements or links to the official state authority.",
        additional_instruction="Accept references/links to TEA (Texas) or TSPC/ODE (Oregon) as fulfilling 'clearly states' licensure requirements."
    )

    # C7: Out-of-state credentials info
    await add_criterion_with_urls(
        evaluator,
        top,
        "C7_Out_of_State_Credentials_Info",
        "District provides information about accepting or processing out-of-state teaching credentials.",
        ex.c7.urls,
        claim="The district provides information about accepting or processing out-of-state teaching credentials (e.g., reciprocity, evaluation, temporary permits).",
        additional_instruction="Look for guidance directed to out-of-state applicants or credential transfer instructions."
    )

    # C8: Job fair or recruitment event in 2026
    await add_criterion_with_urls(
        evaluator,
        top,
        "C8_Job_Fair_or_Recruitment_Event_2026",
        "District conducted or has scheduled at least one job fair or recruitment event in 2026.",
        ex.c8.urls,
        claim="There is at least one job fair or recruitment event in calendar year 2026 (held or scheduled).",
        additional_instruction="Verify a date in 2026 associated with a job fair/recruitment event (career fair, hiring fair, etc.)."
    )

    # C9: HR contact info listed
    await add_criterion_with_urls(
        evaluator,
        top,
        "C9_HR_Contact_Info",
        "District lists contact information (phone or email) for the Human Resources or employment office.",
        ex.c9.urls,
        claim="The page lists contact information (phone number or email) for the Human Resources or employment office.",
        additional_instruction="Accept either a phone OR an email (or both). It must be clearly associated with HR/employment."
    )

    # C10: Multiple position categories
    await add_criterion_with_urls(
        evaluator,
        top,
        "C10_Multiple_Position_Categories",
        "District posts multiple categories of employment positions (e.g., certified teachers, administrators, support staff).",
        ex.c10.urls,
        claim="The district's employment pages show multiple distinct job categories (for example: Teachers/Certified, Administrators, Classified/Support/Paraprofessional).",
        additional_instruction="Look for separate category labels/filters or listings that clearly indicate multiple categories."
    )

    # C11: Substitute teaching info
    await add_criterion_with_urls(
        evaluator,
        top,
        "C11_Substitute_Info",
        "District provides information about substitute teaching opportunities or requirements.",
        ex.c11.urls,
        claim="The page provides information about substitute teaching opportunities and/or requirements in the district.",
        additional_instruction="Any official page that explains sub application, pay, steps, or requirements is acceptable."
    )

    # C12: Specific required qualifications for teachers
    await add_criterion_with_urls(
        evaluator,
        top,
        "C12_Teacher_Qualifications_Listed",
        "District lists specific required qualifications for teachers (e.g., bachelor's degree, preparation program, certification exams).",
        ex.c12.urls,
        claim="The page lists specific teacher qualification requirements (e.g., bachelor's degree, approved preparation program, certification exams).",
        additional_instruction="Avoid vague statements—confirm concrete requirements are listed."
    )

    # C13: Background check and fingerprinting mentioned
    await add_criterion_with_urls(
        evaluator,
        top,
        "C13_Background_and_Fingerprinting",
        "District mentions background check and fingerprinting requirements for teaching candidates.",
        ex.c13.urls,
        claim="The page mentions background check and fingerprinting requirements for teaching candidates.",
        additional_instruction="Accept descriptions of required background screening and fingerprint submission for employment eligibility."
    )

    # C14: Official sources check across all provided URLs
    all_urls = unique_urls([
        ex.c1.urls, ex.c2.urls, ex.c3.urls, ex.c4.urls, ex.c5.urls, ex.c6.urls, ex.c7.urls,
        ex.c8.urls, ex.c9.urls, ex.c10.urls, ex.c11.urls, ex.c12.urls, ex.c13.urls
    ])
    await add_c14_official_sources_check(evaluator, top, all_urls)


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
    Evaluate the agent's answer for the Oregon/Texas district hiring transparency task.
    """
    # Initialize evaluator
    evaluator = Evaluator()
    evaluator.initialize(
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
    extraction: DistrictExtraction = await evaluator.extract(
        prompt=prompt_extract_district(),
        template_class=DistrictExtraction,
        extraction_name="district_extraction"
    )

    # Build verification tree and run checks
    await build_verification_tree(evaluator, extraction)

    # Return structured summary
    return evaluator.get_summary()