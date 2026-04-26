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
TASK_ID = "tx_bilingual_career_eval"
TASK_DESCRIPTION = (
    "You hold a bachelor's degree in Spanish (GPA 3.2) from an accredited university and are considering a career as a bilingual education teacher in Texas, with the long-term goal of becoming a school principal. "
    "You are evaluating two specific large school districts: Frisco ISD and Katy ISD.\n\n"
    "Research and provide comprehensive information about both districts to determine their suitability for your career goals. For each district (Frisco ISD and Katy ISD), provide:\n\n"
    "1. Current student enrollment number (verify if it meets the 40,000+ threshold)\n"
    "2. Availability of bilingual/ESL teaching positions\n"
    "3. Starting teacher salary for the 2025-2026 school year\n"
    "4. Confirmation that the salary meets Texas state minimum requirements\n"
    "5. Availability of health insurance benefits\n"
    "6. Whether the district accepts alternatively certified teachers\n"
    "7. Evidence of job fairs or recruitment activities scheduled for 2026\n"
    "8. Confirmation that salary schedules use experience-based pay steps\n"
    "9. Availability of an online job application system\n\n"
    "Additionally, verify the following general Texas education career information:\n\n"
    "10. Whether Bilingual/ESL is designated as a teacher shortage area in Texas for 2025-2026 (provide reference URL from Texas Education Agency)\n"
    "11. Typical requirements for becoming a principal in Texas (including master's degree requirements and years of teaching experience needed)\n"
    "12. Whether the career pathway from beginning teacher to principal is realistically achievable within 8-10 years in Texas\n\n"
    "For each piece of information, provide specific data points and reference URLs to support your findings."
)


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class DistrictInfo(BaseModel):
    name: Optional[str] = None

    enrollment: Optional[str] = None
    enrollment_urls: List[str] = Field(default_factory=list)

    bilingual_positions: Optional[str] = None
    bilingual_urls: List[str] = Field(default_factory=list)

    starting_salary_2025_2026: Optional[str] = None
    starting_salary_urls: List[str] = Field(default_factory=list)

    meets_state_minimum: Optional[str] = None
    meets_minimum_urls: List[str] = Field(default_factory=list)

    health_insurance: Optional[str] = None
    health_insurance_urls: List[str] = Field(default_factory=list)

    accepts_alt_cert: Optional[str] = None
    alt_cert_urls: List[str] = Field(default_factory=list)

    job_fairs_2026: Optional[str] = None
    job_fairs_urls: List[str] = Field(default_factory=list)

    salary_steps: Optional[str] = None
    salary_steps_urls: List[str] = Field(default_factory=list)

    online_application: Optional[str] = None
    online_application_urls: List[str] = Field(default_factory=list)


class DistrictsExtraction(BaseModel):
    frisco: Optional[DistrictInfo] = None
    katy: Optional[DistrictInfo] = None


class TexasGeneralExtraction(BaseModel):
    bilingual_esl_shortage_2025_2026: Optional[str] = None
    shortage_urls: List[str] = Field(default_factory=list)

    principal_masters_requirement: Optional[str] = None
    principal_masters_urls: List[str] = Field(default_factory=list)

    principal_teaching_experience_requirement: Optional[str] = None
    principal_experience_urls: List[str] = Field(default_factory=list)

    timeline_feasible_8_10_years: Optional[str] = None
    timeline_urls: List[str] = Field(default_factory=list)

    grad_programs_available: Optional[str] = None
    grad_programs_urls: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_districts() -> str:
    return """
Extract structured information for the following two Texas school districts ONLY: Frisco ISD and Katy ISD.

For EACH district, extract the following fields EXACTLY as they appear in the answer and include all URLs cited to support each item (do not invent URLs). Use strings for values (even for numbers), and return null when missing.

For each district object, include these keys:
- name: district name (e.g., "Frisco ISD", "Katy ISD")
- enrollment: current student enrollment figure as stated (string, e.g., "67,000", "about 60k", "approximately 90,000")
- enrollment_urls: array of URLs that support the enrollment claim
- bilingual_positions: whether bilingual/ESL teaching positions are available or applications accepted (e.g., "Yes - listed on jobs page", or "No/Not listed") 
- bilingual_urls: array of URLs supporting bilingual/ESL positions availability
- starting_salary_2025_2026: the starting teacher salary for the 2025-2026 school year as stated in the answer (string, e.g., "$61,000", "starting at 58,500")
- starting_salary_urls: array of URLs supporting the starting salary figure
- meets_state_minimum: whether the district's starting salary meets/exceeds Texas state minimum (string such as "yes", "no", or a short explanation)
- meets_minimum_urls: array of URLs used to justify meeting state minimum (e.g., TEA minimum schedule and/or district salary page)
- health_insurance: whether health insurance benefits are available to teachers (string)
- health_insurance_urls: array of URLs supporting the health insurance benefits
- accepts_alt_cert: whether the district accepts alternatively certified teachers (string)
- alt_cert_urls: array of URLs supporting alternative certification acceptance
- job_fairs_2026: whether job fairs or recruitment events are scheduled in 2026 (string; include brief detail if present)
- job_fairs_urls: array of URLs supporting the 2026 job fair/recruiting events
- salary_steps: whether the salary schedule uses experience-based pay steps (string)
- salary_steps_urls: array of URLs supporting experience-based steps
- online_application: whether there is an online job application system (string; may include the portal name)
- online_application_urls: array of URLs for the online application or HR portal

Important:
- Extract ONLY URLs explicitly present in the answer. If a URL is missing a protocol, prepend http://.
- Do not merge or infer values; extract exactly what the answer states.
- If anything is missing, set the value to null (and the URL list to an empty array).
- Return JSON with two top-level keys: "frisco" and "katy", each an object with the above keys.
    """


def prompt_extract_texas_general() -> str:
    return """
Extract the following general Texas education career items from the answer, with supporting URLs for each item. Use strings for values (e.g., "yes", "no", or short explanation). If an item is missing, set it to null and its URLs to an empty array.

- bilingual_esl_shortage_2025_2026: whether Bilingual/ESL is designated as a teacher shortage area in Texas for 2025-2026 (string like "yes", "no", or short note)
- shortage_urls: URLs that support the shortage designation; must include at least one Texas Education Agency (TEA) link if provided in the answer
- principal_masters_requirement: whether a master's degree in educational administration/leadership (or approved program) is typically required for principal certification in Texas (string)
- principal_masters_urls: URLs supporting master's degree requirement
- principal_teaching_experience_requirement: years of classroom teaching experience typically required before becoming a principal in Texas (string; e.g., "2 years minimum", "3-5 years typical")
- principal_experience_urls: URLs supporting teaching experience requirement
- timeline_feasible_8_10_years: whether a pathway from starting teacher to principal is realistically achievable within 8-10 years in Texas (string)
- timeline_urls: URLs supporting the feasibility timeline
- grad_programs_available: whether educational administration/leadership master's degree programs are accessible in Texas (string)
- grad_programs_urls: URLs to Texas universities or credible sources offering such programs

Important:
- Extract ONLY URLs explicitly present in the answer. If a URL is missing a protocol, prepend http://.
- Do not create or infer any URLs.
- If any item is missing, set its value to null and its URL list to an empty array.
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _non_empty_urls(urls: Optional[List[str]]) -> Optional[List[str]]:
    if not urls:
        return None
    cleaned = [u.strip() for u in urls if isinstance(u, str) and u.strip()]
    return cleaned if cleaned else None


def _no_source_fail_instruction(extra: Optional[str] = None) -> str:
    base = (
        "You must verify the claim using the provided webpage(s). If the provided URL list is empty or none of the URLs are valid/relevant, "
        "judge the claim as not supported (Incorrect). Prefer explicit statements on the page. Allow reasonable rounding for numbers."
    )
    if extra:
        return base + " " + extra
    return base


# --------------------------------------------------------------------------- #
# Verification builders                                                       #
# --------------------------------------------------------------------------- #
async def verify_district(
    evaluator: Evaluator,
    parent_node,
    district_label: str,
    info: Optional[DistrictInfo],
    id_prefix: str
) -> None:
    """
    Build verification leaves for a district under its evaluation node.
    district_label: human-readable (e.g., "Frisco ISD")
    id_prefix: "Frisco" or "Katy" to match rubric leaf IDs
    """
    # Enrollment (Critical)
    enrollment_leaf = evaluator.add_leaf(
        id=f"{id_prefix}_Enrollment",
        desc=f"Provide {district_label} enrollment figure and verify it meets minimum 40,000 students",
        parent=parent_node,
        critical=True
    )
    enrollment_val = info.enrollment if info else None
    claim_enrollment = (
        f"{district_label}'s current student enrollment is reported as '{enrollment_val}'. "
        f"This enrollment meets or exceeds the 40,000 threshold."
    )
    await evaluator.verify(
        claim=claim_enrollment,
        node=enrollment_leaf,
        sources=_non_empty_urls(info.enrollment_urls if info else None),
        additional_instruction=_no_source_fail_instruction(
            "Accept approximate phrases like 'about' or 'approximately'. If the page shows a number below 40,000 or does not provide a clear, current total, mark Incorrect."
        ),
    )

    # Bilingual/ESL positions available (Critical)
    bilingual_leaf = evaluator.add_leaf(
        id=f"{id_prefix}_Bilingual_Positions",
        desc=f"Confirm {district_label} has bilingual/ESL teaching positions available or accepts applications",
        parent=parent_node,
        critical=True
    )
    claim_bilingual = (
        f"{district_label} currently lists bilingual and/or ESL teaching positions or accepts applications for such roles."
    )
    await evaluator.verify(
        claim=claim_bilingual,
        node=bilingual_leaf,
        sources=_non_empty_urls(info.bilingual_urls if info else None),
        additional_instruction=_no_source_fail_instruction(
            "Look for job postings, job categories, or HR pages explicitly mentioning bilingual or ESL teacher roles."
        ),
    )

    # Starting salary for 2025-2026 (Critical)
    salary_leaf = evaluator.add_leaf(
        id=f"{id_prefix}_Starting_Salary",
        desc=f"Report {district_label} starting teacher salary for 2025-2026 school year",
        parent=parent_node,
        critical=True
    )
    salary_val = info.starting_salary_2025_2026 if info else None
    claim_salary = (
        f"The starting teacher salary for the 2025-2026 school year at {district_label} is '{salary_val}'."
    )
    await evaluator.verify(
        claim=claim_salary,
        node=salary_leaf,
        sources=_non_empty_urls(info.starting_salary_urls if info else None),
        additional_instruction=_no_source_fail_instruction(
            "The page should be a district salary schedule or official HR/board document clearly labeled 2025-2026 (or explicitly stating it applies to 2025-2026). "
            "Allow for pages that say '2025-26' or similar wording."
        ),
    )

    # Meets Texas state minimum (Critical)
    meets_min_leaf = evaluator.add_leaf(
        id=f"{id_prefix}_Meets_State_Minimum",
        desc=f"Confirm {district_label} starting salary meets or exceeds Texas state minimum",
        parent=parent_node,
        critical=True
    )
    # Use the salary value from the claim; verifier may compare to TEA min displayed on the page.
    claim_meets_min = (
        f"{district_label}'s starting teacher salary for 2025-2026 ('{salary_val}') meets or exceeds the Texas state minimum salary schedule for 2025-2026."
    )
    await evaluator.verify(
        claim=claim_meets_min,
        node=meets_min_leaf,
        sources=_non_empty_urls(info.meets_minimum_urls if info else None),
        additional_instruction=_no_source_fail_instruction(
            "This check can be satisfied by a TEA minimum salary schedule page for 2025-2026. "
            "Compare the numeric starting salary stated in the claim text to the TEA minimum shown on this page; if the claim's value is greater than or equal to the TEA minimum, mark Supported. "
            "If the page is not a TEA/official page or does not show the applicable minimum for 2025-2026, mark Incorrect."
        ),
    )

    # Health insurance benefits (Non-critical)
    health_leaf = evaluator.add_leaf(
        id=f"{id_prefix}_Health_Insurance",
        desc=f"Verify {district_label} offers health insurance benefits to teachers",
        parent=parent_node,
        critical=False
    )
    claim_health = f"{district_label} offers health insurance benefits to teachers."
    await evaluator.verify(
        claim=claim_health,
        node=health_leaf,
        sources=_non_empty_urls(info.health_insurance_urls if info else None),
        additional_instruction=_no_source_fail_instruction(
            "Accept pages outlining employee benefits or TRS-ActiveCare information that explicitly indicates health/medical insurance availability to teachers."
        ),
    )

    # Accepts alternative certification (Non-critical)
    alt_cert_leaf = evaluator.add_leaf(
        id=f"{id_prefix}_Alt_Cert",
        desc=f"Confirm {district_label} accepts teachers from alternative certification programs",
        parent=parent_node,
        critical=False
    )
    claim_alt_cert = f"{district_label} accepts alternatively certified teachers (Texas ACP or equivalent)."
    await evaluator.verify(
        claim=claim_alt_cert,
        node=alt_cert_leaf,
        sources=_non_empty_urls(info.alt_cert_urls if info else None),
        additional_instruction=_no_source_fail_instruction(
            "Look for HR employment requirements or job postings that mention acceptance of alternative certification (ACP) or 'intern' certificates."
        ),
    )

    # Job fairs/recruitment 2026 (Non-critical)
    job_fairs_leaf = evaluator.add_leaf(
        id=f"{id_prefix}_Job_Fairs",
        desc=f"Verify {district_label} has job fairs or recruitment events scheduled for 2026",
        parent=parent_node,
        critical=False
    )
    claim_job_fairs = f"{district_label} has job fairs or recruitment events scheduled during 2026."
    await evaluator.verify(
        claim=claim_job_fairs,
        node=job_fairs_leaf,
        sources=_non_empty_urls(info.job_fairs_urls if info else None),
        additional_instruction=_no_source_fail_instruction(
            "The page should indicate event(s) in calendar year 2026 (e.g., dates in 2026). Past-year events are not acceptable."
        ),
    )

    # Salary schedule uses experience-based steps (Non-critical)
    salary_steps_leaf = evaluator.add_leaf(
        id=f"{id_prefix}_Salary_Steps",
        desc=f"Confirm {district_label} salary schedule includes experience-based pay increases",
        parent=parent_node,
        critical=False
    )
    claim_salary_steps = f"{district_label}'s teacher salary schedule uses experience-based step increases."
    await evaluator.verify(
        claim=claim_salary_steps,
        node=salary_steps_leaf,
        sources=_non_empty_urls(info.salary_steps_urls if info else None),
        additional_instruction=_no_source_fail_instruction(
            "Accept if the salary schedule shows step increments by years of experience (e.g., Step 0, Step 1, etc.)."
        ),
    )

    # Online job application (Non-critical)
    online_app_leaf = evaluator.add_leaf(
        id=f"{id_prefix}_Online_Application",
        desc=f"Confirm {district_label} provides online job application system",
        parent=parent_node,
        critical=False
    )
    claim_online_app = f"{district_label} provides an online job application system/portal for applicants."
    await evaluator.verify(
        claim=claim_online_app,
        node=online_app_leaf,
        sources=_non_empty_urls(info.online_application_urls if info else None),
        additional_instruction=_no_source_fail_instruction(
            "Accept if the URL is a district HR application portal (e.g., TalentEd, Frontline, or a district-branded eRecruit site) where candidates can submit applications online."
        ),
    )


async def verify_texas_shortage_status(
    evaluator: Evaluator,
    parent_node
) -> None:
    """
    Build verification nodes for Texas bilingual/ESL shortage designation.
    """
    # Extracted general info should already be recorded; access later via closure in evaluate_answer
    pass  # Will be replaced in evaluate_answer where data is available


async def verify_principal_requirements(
    evaluator: Evaluator,
    parent_node
) -> None:
    pass  # Implemented in evaluate_answer with available extracted data


async def verify_career_timeline(
    evaluator: Evaluator,
    parent_node
) -> None:
    pass  # Implemented in evaluate_answer with available extracted data


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
    Evaluate an answer for the Texas bilingual teacher and principal pathway career opportunity task.
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
        default_model=model,
    )

    # Top-level career evaluation node as per rubric
    career_root = evaluator.add_parallel(
        id="Career_Opportunity_Evaluation",
        desc="Evaluate whether Frisco ISD and Katy ISD meet key qualifications for a bilingual teaching career with principal pathway",
        parent=root,
        critical=False
    )

    # Extract structured info
    districts_data, texas_general = await asyncio.gather(
        evaluator.extract(
            prompt=prompt_extract_districts(),
            template_class=DistrictsExtraction,
            extraction_name="districts_extraction",
        ),
        evaluator.extract(
            prompt=prompt_extract_texas_general(),
            template_class=TexasGeneralExtraction,
            extraction_name="texas_general_extraction",
        )
    )

    # Frisco ISD Evaluation subtree
    frisco_parent = evaluator.add_parallel(
        id="Frisco_ISD_Evaluation",
        desc="Comprehensive evaluation of Frisco ISD for bilingual teaching career suitability",
        parent=career_root,
        critical=False
    )
    await verify_district(
        evaluator=evaluator,
        parent_node=frisco_parent,
        district_label="Frisco ISD",
        info=districts_data.frisco if districts_data else None,
        id_prefix="Frisco"
    )

    # Katy ISD Evaluation subtree
    katy_parent = evaluator.add_parallel(
        id="Katy_ISD_Evaluation",
        desc="Comprehensive evaluation of Katy ISD for bilingual teaching career suitability",
        parent=career_root,
        critical=False
    )
    await verify_district(
        evaluator=evaluator,
        parent_node=katy_parent,
        district_label="Katy ISD",
        info=districts_data.katy if districts_data else None,
        id_prefix="Katy"
    )

    # Texas Bilingual Shortage Status subtree
    shortage_parent = evaluator.add_parallel(
        id="Texas_Bilingual_Shortage_Status",
        desc="Verify Bilingual/ESL is designated as a teacher shortage area in Texas for 2025-2026",
        parent=career_root,
        critical=False
    )

    shortage_confirm_leaf = evaluator.add_leaf(
        id="Shortage_Designation_Confirmed",
        desc="Confirm Bilingual/ESL appears on Texas Education Agency shortage area list for 2025-2026",
        parent=shortage_parent,
        critical=True
    )
    claim_shortage = "For the 2025-2026 school year in Texas, Bilingual/ESL is designated as a teacher shortage area."
    await evaluator.verify(
        claim=claim_shortage,
        node=shortage_confirm_leaf,
        sources=_non_empty_urls(texas_general.shortage_urls if texas_general else None),
        additional_instruction=_no_source_fail_instruction(
            "Prefer an official TEA source that explicitly names 'Bilingual' and/or 'ESL' as shortage areas for 2025-2026."
        ),
    )

    tea_ref_leaf = evaluator.add_leaf(
        id="TEA_Reference_URL",
        desc="Provide reference URL from Texas Education Agency confirming shortage designation",
        parent=shortage_parent,
        critical=True
    )
    claim_tea_ref = "At least one of the provided URLs is an official Texas Education Agency (TEA) webpage that explicitly confirms the Bilingual/ESL shortage designation for 2025-2026."
    await evaluator.verify(
        claim=claim_tea_ref,
        node=tea_ref_leaf,
        sources=_non_empty_urls(texas_general.shortage_urls if texas_general else None),
        additional_instruction=_no_source_fail_instruction(
            "Verify that the page domain is tea.texas.gov (or an official TEA subdomain) and that the page content confirms the shortage area for 2025-2026."
        ),
    )

    # Texas Principal Pathway Requirements subtree
    principal_parent = evaluator.add_parallel(
        id="Texas_Principal_Pathway_Requirements",
        desc="Verify typical requirements for becoming a principal in Texas",
        parent=career_root,
        critical=False
    )

    masters_leaf = evaluator.add_leaf(
        id="Masters_Degree_Requirement",
        desc="Confirm master's degree in educational administration is typically required for principals in Texas",
        parent=principal_parent,
        critical=True
    )
    claim_masters = (
        "In Texas, becoming a principal typically requires a master's degree in educational leadership/administration or an approved principal preparation program (in addition to certification requirements)."
    )
    await evaluator.verify(
        claim=claim_masters,
        node=masters_leaf,
        sources=_non_empty_urls(texas_general.principal_masters_urls if texas_general else None),
        additional_instruction=_no_source_fail_instruction(
            "Prefer TEA certification guidance or accredited university program/certification pages that state a master's degree is required or expected for principal certification."
        ),
    )

    experience_leaf = evaluator.add_leaf(
        id="Teaching_Experience_Requirement",
        desc="Confirm 3-5 years classroom teaching experience typically required before becoming principal",
        parent=principal_parent,
        critical=True
    )
    claim_experience = (
        "In Texas, a typical requirement before becoming a principal is approximately 3–5 years of classroom teaching experience (recognizing that TEA principal certification requires at least two years)."
    )
    await evaluator.verify(
        claim=claim_experience,
        node=experience_leaf,
        sources=_non_empty_urls(texas_general.principal_experience_urls if texas_general else None),
        additional_instruction=_no_source_fail_instruction(
            "Accept credible sources (TEA or Texas districts/universities) that indicate minimums and/or typical expectations (e.g., 2-year minimum for certification; many roles prefer 3–5 years)."
        ),
    )

    # Career Timeline Feasibility subtree
    timeline_parent = evaluator.add_parallel(
        id="Career_Timeline_Feasibility",
        desc="Verify the career pathway from beginning teacher to principal is realistically achievable within 8-10 years in Texas",
        parent=career_root,
        critical=False
    )

    timeline_leaf = evaluator.add_leaf(
        id="Timeline_Confirmation",
        desc="Confirm typical career timeline is 8-10 years from starting as teacher to becoming principal",
        parent=timeline_parent,
        critical=False
    )
    claim_timeline = (
        "In Texas, the pathway from starting as a teacher to becoming a principal is realistically achievable within approximately 8–10 years under typical circumstances."
    )
    await evaluator.verify(
        claim=claim_timeline,
        node=timeline_leaf,
        sources=_non_empty_urls(texas_general.timeline_urls if texas_general else None),
        additional_instruction=_no_source_fail_instruction(
            "Consider pages that outline principal certification requirements, program lengths (e.g., 1–3 years for a master's), and common experience expectations. "
            "If the provided sources reasonably support an 8–10 year pathway given these steps, mark Supported."
        ),
    )

    grad_programs_leaf = evaluator.add_leaf(
        id="Graduate_Programs_Available",
        desc="Confirm educational administration master's degree programs are accessible in Texas",
        parent=timeline_parent,
        critical=True
    )
    claim_grad_programs = "Educational administration/leadership master's degree programs (leading to principal certification) are accessible in Texas."
    await evaluator.verify(
        claim=claim_grad_programs,
        node=grad_programs_leaf,
        sources=_non_empty_urls(texas_general.grad_programs_urls if texas_general else None),
        additional_instruction=_no_source_fail_instruction(
            "Accept any credible Texas university page offering an M.Ed./M.S. in Educational Leadership/Administration or Principal Certification program."
        ),
    )

    return evaluator.get_summary()