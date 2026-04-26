import asyncio
import logging
from typing import Any, List, Optional, Dict

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy

# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "multi_sector_career_positions_eval"
TASK_DESCRIPTION = """
I am researching career advancement opportunities across multiple professional sectors for a career guidance publication. I need to identify four specific mid-to-senior level positions that meet the following criteria:

Position 1 - Restaurant Industry Leadership:
A multi-unit supervisory role in the restaurant industry that requires a minimum of 5 years of industry experience, typically oversees 5-8 locations, requires food safety certification, and offers an annual salary range between $69,000-$103,000.

Position 2 - Higher Education Athletics Administration:
An athletics leadership position at a college or university that requires a master's degree in sports management or a related field, requires professional certification from a national athletic administrators' association including completion of specific leadership training courses, and requires a minimum of 2 years of experience as an athletic administrator.

Position 3 - Broadcast Meteorology:
A television weather forecasting position that requires a bachelor's degree in meteorology or atmospheric science from an accredited institution, requires professional certification from the American Meteorological Society (AMS), and requires a minimum of 2 years of full-time on-air broadcast experience.

Position 4 - Higher Education Student Services:
A student services leadership position in higher education that focuses on career development, requires a master's degree in counseling or higher education administration, and requires 5-7 years of professional experience in a relevant field.

For each position, provide:
- The specific job title
- The educational degree requirement
- The required professional certification (if applicable)
- The minimum years of experience required
- The typical annual salary range (if specified for that position type)
- A reference URL supporting each requirement
"""

# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class PositionData(BaseModel):
    job_title: Optional[str] = None
    degree_requirement: Optional[str] = None
    required_certifications: List[str] = Field(default_factory=list)
    years_experience: Optional[str] = None
    salary_range: Optional[str] = None
    reference_urls: List[str] = Field(default_factory=list)


class PositionsExtraction(BaseModel):
    position_1: Optional[PositionData] = None
    position_2: Optional[PositionData] = None
    position_3: Optional[PositionData] = None
    position_4: Optional[PositionData] = None


# --------------------------------------------------------------------------- #
# Extraction prompt helpers                                                   #
# --------------------------------------------------------------------------- #
def prompt_extract_positions() -> str:
    return """
Extract exactly four positions from the answer, mapping them to the following categories:

- position_1: Restaurant industry leadership (multi-unit supervisory role)
- position_2: Higher education athletics administration (college/university athletics leadership)
- position_3: Broadcast meteorology (television/broadcast meteorologist)
- position_4: Higher education student services (career services leadership)

For each position, extract:
1) job_title: The specific job title as given in the answer.
2) degree_requirement: The required degree(s) and field(s), exactly as stated.
3) required_certifications: A list of the professional certifications explicitly required (e.g., "ServSafe Food Protection Manager", "NIAAA CAA", "AMS CBM"). If none are stated, return an empty list.
4) years_experience: The minimum years of experience requirement exactly as described (e.g., "5 years", "2+ years", "5-7 years").
5) salary_range: The typical annual salary range for that position as stated (e.g., "$69,000-$103,000"). If not explicitly provided, return null.
6) reference_urls: A list of all URLs present in the answer that specifically support this position’s requirements (degree, certifications, years of experience, scope/salary where applicable). Extract only valid URLs explicitly mentioned. Include multiple URLs if present.

Important:
- Do not invent or infer any values. Use only what appears in the answer.
- If a field is not mentioned, return null (for strings) or [] (for lists).
- The reference_urls must be URLs explicitly present in the answer text.
"""


# --------------------------------------------------------------------------- #
# Utility helpers                                                             #
# --------------------------------------------------------------------------- #
def _safe_sources(urls: Optional[List[str]]) -> List[str]:
    if not urls:
        return []
    # Filter out obviously invalid or empty strings
    return [u for u in urls if isinstance(u, str) and len(u.strip()) > 0]


# --------------------------------------------------------------------------- #
# Verification functions for each position                                    #
# --------------------------------------------------------------------------- #
async def verify_position_1_restaurant(evaluator: Evaluator, parent_node, data: Optional[PositionData]) -> None:
    """
    Position 1 - Restaurant Industry Leadership
    Checks:
      - Job title identifies a multi-unit restaurant management or district manager role
      - Requires minimum 5 years of restaurant industry experience
      - Typically oversees 5-8 locations
      - Requires food safety certification (e.g., ServSafe)
      - Salary range between $69,000-$103,000
      - Valid reference URL provided
    """
    d = data or PositionData()
    sources = _safe_sources(d.reference_urls)

    pos_node = evaluator.add_parallel(
        id="position_1_restaurant",
        desc="Restaurant industry multi-unit leadership position with all specified qualifications",
        parent=parent_node,
        critical=False
    )

    # 1) Job title classification
    n_job = evaluator.add_leaf(
        id="position_1_job_title",
        desc="Job title identifies a multi-unit restaurant management or district manager role",
        parent=pos_node,
        critical=True
    )
    job_title_text = d.job_title or "the specified role"
    claim_job = (
        f"Based on the cited source(s), the job titled '{job_title_text}' is a multi-unit supervisory role "
        f"in the restaurant industry (e.g., District/Area/Multi-Unit Manager) that oversees multiple restaurant locations."
    )

    # 2) Experience >= 5 years (restaurant industry)
    n_exp = evaluator.add_leaf(
        id="position_1_experience",
        desc="Position requires minimum 5 years of restaurant industry experience",
        parent=pos_node,
        critical=True
    )
    claim_exp = "This position requires at least 5 years of experience in the restaurant industry."

    # 3) Scope: oversees 5–8 locations
    n_scope = evaluator.add_leaf(
        id="position_1_scope",
        desc="Position typically oversees 5-8 restaurant locations",
        parent=pos_node,
        critical=True
    )
    claim_scope = "This position typically oversees 5 to 8 restaurant locations (units/stores)."

    # 4) Food safety certification required
    n_cert = evaluator.add_leaf(
        id="position_1_certification",
        desc="Position requires food safety certification such as ServSafe",
        parent=pos_node,
        critical=True
    )
    claim_cert = (
        "This position requires a food safety certification such as ServSafe (e.g., ServSafe Food Protection Manager)."
    )

    # 5) Salary range check
    n_salary = evaluator.add_leaf(
        id="position_1_salary",
        desc="Position offers annual salary range between $69,000-$103,000",
        parent=pos_node,
        critical=True
    )
    claim_salary = (
        "The typical annual base salary range for this position is between $69,000 and $103,000 USD."
    )

    # 6) Reference URLs existence (treat as existence check)
    evaluator.add_custom_node(
        result=len(sources) > 0,
        id="position_1_reference",
        desc="Valid reference URL provided supporting the position requirements",
        parent=pos_node,
        critical=True
    )

    claims = [
        (claim_job, sources, n_job,
         "Verify the role is clearly a multi-unit supervisory position in the restaurant industry. "
         "Accept common titles like District Manager, Area Manager, Area Coach, Multi-Unit Manager. "
         "Look for language like 'oversees multiple restaurants/units/locations'."),
        (claim_exp, sources, n_exp,
         "Confirm the minimum experience is at least 5 years in the restaurant/foodservice industry. "
         "Treat '5+' or 'five years' as meeting the requirement."),
        (claim_scope, sources, n_scope,
         "Confirm that the role's scope commonly spans about 5 to 8 locations. "
         "Allow equivalent phrasing such as 'units', 'stores', or 'restaurants'."),
        (claim_cert, sources, n_cert,
         "Look for explicit requirement of food safety certification (e.g., ServSafe Food Protection Manager) "
         "or equivalent food protection/food safety manager certification."),
        (claim_salary, sources, n_salary,
         "Check for a typical salary range that falls within or is substantively equivalent to $69,000–$103,000. "
         "Allow minor rounding or inclusive bounds if strongly consistent with the stated range.")
    ]
    await evaluator.batch_verify(claims)


async def verify_position_2_athletics(evaluator: Evaluator, parent_node, data: Optional[PositionData]) -> None:
    """
    Position 2 - Higher Education Athletics Administration
    Checks:
      - Job title identifies a college/university athletics director or athletics administrator
      - Requires a master's degree in sports management/education administration/related
      - Requires certification from NIAAA
      - NIAAA certification includes completion of LTC 501, 502, 503
      - Requires minimum 2 years of experience as an athletic administrator
      - Valid reference URL provided
    """
    d = data or PositionData()
    sources = _safe_sources(d.reference_urls)

    pos_node = evaluator.add_parallel(
        id="position_2_athletics",
        desc="Higher education athletics administration position with all specified qualifications",
        parent=parent_node,
        critical=False
    )

    # 1) Job title classification
    n_job = evaluator.add_leaf(
        id="position_2_job_title",
        desc="Job title identifies a college or university athletic director or athletics administrator role",
        parent=pos_node,
        critical=True
    )
    job_title_text = d.job_title or "the specified role"
    claim_job = (
        f"The job titled '{job_title_text}' is an athletics leadership role at a college or university "
        f"(e.g., Athletic Director, Associate/Assistant AD, or Athletics Administrator)."
    )

    # 2) Education requirement
    n_edu = evaluator.add_leaf(
        id="position_2_education",
        desc="Position requires a master's degree in sports management, education administration, or related field",
        parent=pos_node,
        critical=True
    )
    claim_edu = (
        "This position requires a master's degree in sports management, education administration, or a closely related field."
    )

    # 3) Certification body: NIAAA
    n_cert_body = evaluator.add_leaf(
        id="position_2_certification_body",
        desc="Position requires professional certification from the National Interscholastic Athletic Administrators Association (NIAAA)",
        parent=pos_node,
        critical=True
    )
    claim_cert_body = (
        "This position requires professional certification from the National Interscholastic Athletic Administrators Association (NIAAA)."
    )

    # 4) Certification courses: LTC 501/502/503
    n_cert_courses = evaluator.add_leaf(
        id="position_2_certification_courses",
        desc="NIAAA certification includes completion of Leadership Training Courses 501, 502, and 503",
        parent=pos_node,
        critical=True
    )
    claim_cert_courses = (
        "NIAAA certification includes completion of Leadership Training Courses (LTC) 501, 502, and 503."
    )

    # 5) Experience >= 2 years (athletic administrator)
    n_exp = evaluator.add_leaf(
        id="position_2_experience",
        desc="Position requires minimum 2 years of experience as an athletic administrator",
        parent=pos_node,
        critical=True
    )
    claim_exp = "This position requires a minimum of 2 years of experience as an athletic administrator."

    # 6) Reference URLs existence
    evaluator.add_custom_node(
        result=len(sources) > 0,
        id="position_2_reference",
        desc="Valid reference URL provided supporting the position requirements",
        parent=pos_node,
        critical=True
    )

    claims = [
        (claim_job, sources, n_job,
         "Verify the role is an athletics leadership/administration position at a college or university. "
         "Accept titles such as Athletic Director, Associate/Assistant Athletic Director, or Athletics Administrator."),
        (claim_edu, sources, n_edu,
         "Confirm the master's degree requirement is specifically in sports management, education administration, "
         "or a clearly related field."),
        (claim_cert_body, sources, n_cert_body,
         "Confirm the requirement is certification from NIAAA (National Interscholastic Athletic Administrators Association). "
         "If other certifications are listed but NIAAA is not required, mark as not supported."),
        (claim_cert_courses, sources, n_cert_courses,
         "Verify that NIAAA certification involves LTC 501, 502, and 503 courses; "
         "pages from NIAAA or official course descriptions are acceptable."),
        (claim_exp, sources, n_exp,
         "Confirm a minimum of 2 years of experience as an athletic administrator is explicitly required. "
         "Treat '2+' or 'two years' as meeting the requirement.")
    ]
    await evaluator.batch_verify(claims)


async def verify_position_3_meteorology(evaluator: Evaluator, parent_node, data: Optional[PositionData]) -> None:
    """
    Position 3 - Broadcast Meteorology
    Checks:
      - Job title identifies a television/broadcast meteorologist role
      - Requires bachelor's degree in meteorology or atmospheric science from an accredited institution
      - Requires AMS Certified Broadcast Meteorologist (CBM) certification
      - Requires minimum 2 years of full-time on-air broadcast meteorologist experience
      - Valid reference URL provided
    """
    d = data or PositionData()
    sources = _safe_sources(d.reference_urls)

    pos_node = evaluator.add_parallel(
        id="position_3_meteorology",
        desc="Television broadcast meteorology position with all specified qualifications",
        parent=parent_node,
        critical=False
    )

    # 1) Job title classification
    n_job = evaluator.add_leaf(
        id="position_3_job_title",
        desc="Job title identifies a television meteorologist or broadcast meteorologist role",
        parent=pos_node,
        critical=True
    )
    job_title_text = d.job_title or "the specified role"
    claim_job = (
        f"The job titled '{job_title_text}' is a television/broadcast meteorologist role."
    )

    # 2) Education requirement
    n_edu = evaluator.add_leaf(
        id="position_3_education",
        desc="Position requires a bachelor's degree in meteorology or atmospheric science from an accredited institution",
        parent=pos_node,
        critical=True
    )
    claim_edu = (
        "This position requires a bachelor's degree in meteorology or atmospheric science from an accredited institution."
    )

    # 3) Certification: AMS CBM
    n_cert_body = evaluator.add_leaf(
        id="position_3_certification_body",
        desc="Position requires American Meteorological Society (AMS) Certified Broadcast Meteorologist (CBM) certification",
        parent=pos_node,
        critical=True
    )
    claim_cert_body = (
        "This position requires the American Meteorological Society's Certified Broadcast Meteorologist (AMS CBM) certification."
    )

    # 4) Experience >= 2 years (full-time on-air)
    n_exp = evaluator.add_leaf(
        id="position_3_experience",
        desc="Position requires minimum 2 years of full-time on-air broadcast meteorologist experience",
        parent=pos_node,
        critical=True
    )
    claim_exp = (
        "This position requires a minimum of 2 years of full-time on-air broadcast meteorologist experience."
    )

    # 5) Reference URLs existence
    evaluator.add_custom_node(
        result=len(sources) > 0,
        id="position_3_reference",
        desc="Valid reference URL provided supporting the position requirements",
        parent=pos_node,
        critical=True
    )

    claims = [
        (claim_job, sources, n_job,
         "Verify the role is clearly a television/broadcast meteorologist (on-air weathercaster)."),
        (claim_edu, sources, n_edu,
         "Confirm that a bachelor's degree in meteorology or atmospheric science is required. "
         "If 'accredited' is not explicitly mentioned but the degree requirement is standard and credible for broadcast meteorology, "
         "consider it acceptable unless the page suggests a non-accredited program."),
        (claim_cert_body, sources, n_cert_body,
         "Verify that the role requires AMS CBM certification; accept 'AMS Certified Broadcast Meteorologist (CBM)'."),
        (claim_exp, sources, n_exp,
         "Confirm a minimum of 2 years of full-time on-air broadcast experience; accept synonyms like 'on-air', 'broadcast'.")
    ]
    await evaluator.batch_verify(claims)


async def verify_position_4_student_services(evaluator: Evaluator, parent_node, data: Optional[PositionData]) -> None:
    """
    Position 4 - Higher Education Student Services (Career Services Leadership)
    Checks:
      - Job title identifies a career services director or similar student services leadership role in higher education
      - Role focuses on career development and student career services
      - Requires a master's degree in counseling or higher education administration
      - Requires 5-7 years of professional experience in a relevant field
      - Valid reference URL provided
    """
    d = data or PositionData()
    sources = _safe_sources(d.reference_urls)

    pos_node = evaluator.add_parallel(
        id="position_4_student_services",
        desc="Higher education career services leadership position with all specified qualifications",
        parent=parent_node,
        critical=False
    )

    # 1) Job title classification
    n_job = evaluator.add_leaf(
        id="position_4_job_title",
        desc="Job title identifies a career services director or similar student services leadership role in higher education",
        parent=pos_node,
        critical=True
    )
    job_title_text = d.job_title or "the specified role"
    claim_job = (
        f"The job titled '{job_title_text}' is a higher education career services leadership role "
        f"(e.g., Director of Career Services, Career Center Director)."
    )

    # 2) Focus on career development & student services
    n_focus = evaluator.add_leaf(
        id="position_4_focus",
        desc="Position focuses on career development and student career services",
        parent=pos_node,
        critical=True
    )
    claim_focus = "This role focuses on career development and student career services in a higher education setting."

    # 3) Education requirement
    n_edu = evaluator.add_leaf(
        id="position_4_education",
        desc="Position requires a master's degree in counseling or higher education administration",
        parent=pos_node,
        critical=True
    )
    claim_edu = "This position requires a master's degree in counseling or higher education administration."

    # 4) Experience 5–7 years
    n_exp = evaluator.add_leaf(
        id="position_4_experience",
        desc="Position requires 5-7 years of professional experience in a relevant field",
        parent=pos_node,
        critical=True
    )
    claim_exp = "This position requires 5 to 7 years of relevant professional experience."

    # 5) Reference URLs existence
    evaluator.add_custom_node(
        result=len(sources) > 0,
        id="position_4_reference",
        desc="Valid reference URL provided supporting the position requirements",
        parent=pos_node,
        critical=True
    )

    claims = [
        (claim_job, sources, n_job,
         "Verify the role is a leadership position in higher education career services (e.g., Director of Career Services)."),
        (claim_focus, sources, n_focus,
         "Confirm that the position's responsibilities emphasize career development and services for students."),
        (claim_edu, sources, n_edu,
         "Verify a master's degree is required specifically in counseling or higher education administration."),
        (claim_exp, sources, n_exp,
         "Confirm the required experience is within the 5–7 years range; accept equivalent formats like '5-7', 'five to seven'.")
    ]
    await evaluator.batch_verify(claims)


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
    model: str = "o4-mini",
) -> Dict:
    """
    Evaluate an answer for the multi-sector career positions task.
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

    # Extraction
    extracted = await evaluator.extract(
        prompt=prompt_extract_positions(),
        template_class=PositionsExtraction,
        extraction_name="positions_extraction",
    )

    # Optionally record simple custom info for debugging
    evaluator.add_custom_info(
        info={
            "p1_urls_count": len(_safe_sources(getattr(extracted.position_1 or PositionData(), "reference_urls", []))),
            "p2_urls_count": len(_safe_sources(getattr(extracted.position_2 or PositionData(), "reference_urls", []))),
            "p3_urls_count": len(_safe_sources(getattr(extracted.position_3 or PositionData(), "reference_urls", []))),
            "p4_urls_count": len(_safe_sources(getattr(extracted.position_4 or PositionData(), "reference_urls", []))),
        },
        info_type="extraction_stats",
        info_name="extraction_statistics",
    )

    # Build verification tree per position (parallel under root)
    await verify_position_1_restaurant(evaluator, root, extracted.position_1)
    await verify_position_2_athletics(evaluator, root, extracted.position_2)
    await verify_position_3_meteorology(evaluator, root, extracted.position_3)
    await verify_position_4_student_services(evaluator, root, extracted.position_4)

    return evaluator.get_summary()