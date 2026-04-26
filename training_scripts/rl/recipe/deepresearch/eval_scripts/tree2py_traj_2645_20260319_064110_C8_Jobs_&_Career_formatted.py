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
TASK_ID = "principal_positions_2026"
TASK_DESCRIPTION = """
You are assisting a highly qualified educator who is seeking principal positions across multiple states for the 2026-2027 school year. This educator holds a master's degree in educational leadership and has over 5 years of teaching experience. They are interested in relocating and want to explore opportunities in four specific states: North Carolina, Maryland, Colorado, and Ohio.

Find four current principal job openings, one from each of the following school districts:
1. Charlotte-Mecklenburg Schools in North Carolina
2. Prince George's County Public Schools in Maryland
3. Jeffco Public Schools (Jefferson County Public Schools) in Colorado
4. Columbus City Schools in Ohio

For each position, provide:
- The specific job title and school/location (if specified in the posting)
- A direct URL to the job posting
- Salary information (if provided in the posting or district salary schedule)
- For the Colorado position specifically: Information about retirement benefits (PERA)
- For the Ohio position specifically: Information about principal licensing requirements mentioned in the job description

Requirements:
- All positions must be for principal roles (not assistant principal, vice principal, or other administrative positions)
- All job postings must be currently active and accepting applications as of March 2026
- Each position must be from the specifically named school district in its respective state
"""

AS_OF_MONTH_YEAR = "March 2026"

DISTRICT_SPEC = {
    "NC": {
        "district": "Charlotte-Mecklenburg Schools",
        "aliases": ["Charlotte-Mecklenburg Schools", "CMS"],
        "state": "North Carolina",
    },
    "MD": {
        "district": "Prince George's County Public Schools",
        "aliases": ["Prince George's County Public Schools", "PGCPS", "Prince Georges County Public Schools"],
        "state": "Maryland",
    },
    "CO": {
        "district": "Jeffco Public Schools",
        "aliases": ["Jeffco Public Schools", "Jefferson County Public Schools"],
        "state": "Colorado",
    },
    "OH": {
        "district": "Columbus City Schools",
        "aliases": ["Columbus City Schools", "CCS"],
        "state": "Ohio",
    },
}


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class JobPosting(BaseModel):
    district_name: Optional[str] = None
    state: Optional[str] = None
    job_title: Optional[str] = None
    school_or_location: Optional[str] = None
    job_url: Optional[str] = None

    # Evidence snippets and sources the answer cited
    salary_info: Optional[str] = None
    salary_source_urls: List[str] = Field(default_factory=list)

    benefits_info: Optional[str] = None            # For CO (PERA)
    benefits_source_urls: List[str] = Field(default_factory=list)

    licensing_info: Optional[str] = None           # For OH (licensure requirements)
    licensing_source_urls: List[str] = Field(default_factory=list)

    active_status_text: Optional[str] = None       # E.g., "Open", close date, "Apply" present text
    extra_source_urls: List[str] = Field(default_factory=list)


class PrincipalJobsExtraction(BaseModel):
    north_carolina: Optional[JobPosting] = None
    maryland: Optional[JobPosting] = None
    colorado: Optional[JobPosting] = None
    ohio: Optional[JobPosting] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_principal_jobs() -> str:
    return f"""
Extract exactly one principal position per the following four districts from the provided answer text. If multiple are mentioned for the same district, pick the first clearly identified one. If a required district has no position mentioned, return null for that district.

For each district-specific position, extract the following fields strictly from the answer text:
- district_name: The employing district as stated (e.g., "Charlotte-Mecklenburg Schools").
- state: The state (e.g., "North Carolina", "Maryland", "Colorado", "Ohio") if provided.
- job_title: The specific job title (e.g., "Elementary School Principal").
- school_or_location: The school name or location if mentioned (else null).
- job_url: A direct URL to the job posting itself (not a general careers landing page). Must be explicitly present in the answer.
- salary_info: Any salary detail or pay grade/range text quoted or summarized in the answer (else null).
- salary_source_urls: Any URLs in the answer that support the salary_info (district salary schedules, HR pages, or the job posting itself). Return an empty list if none beyond the job_url is cited.
- benefits_info: For the Colorado position only: any text referencing retirement benefits, especially PERA. For other states, set null.
- benefits_source_urls: For the Colorado position only: URLs cited that support the benefits_info (e.g., Jeffco HR/benefits/PERA pages). Else empty list.
- licensing_info: For the Ohio position only: any text referencing Ohio principal licensure requirements or administrator credentials. For other states, set null.
- licensing_source_urls: For the Ohio position only: URLs cited that support the licensing_info. Else empty list.
- active_status_text: Any explicit sign from the answer indicating the posting is current/accepting applications (e.g., "Apply" button mentioned, "Open until filled", "Closes 2026-04-01").

Map the results into this JSON structure:
{{
  "north_carolina": JobPosting | null,   # Must be from Charlotte-Mecklenburg Schools (CMS)
  "maryland": JobPosting | null,         # Must be from Prince George's County Public Schools (PGCPS)
  "colorado": JobPosting | null,         # Must be from Jeffco Public Schools (Jefferson County Public Schools)
  "ohio": JobPosting | null              # Must be from Columbus City Schools
}}

Rules:
- Only use URLs explicitly present in the answer; do not invent URLs.
- If a field is not present in the answer, set it to null (or an empty list for URL arrays).
- Prefer direct posting URLs (specific job detail pages), not general listings.
- Keep the extracted text faithful to the answer (do not paraphrase beyond minor normalization).
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def _unique_preserve_order(urls: List[Optional[str]]) -> List[str]:
    seen = set()
    out: List[str] = []
    for u in urls:
        if not u:
            continue
        if u not in seen:
            seen.add(u)
            out.append(u)
    return out


def _collect_salary_sources(job: Optional[JobPosting]) -> List[str]:
    if not job:
        return []
    return _unique_preserve_order([job.job_url] + (job.salary_source_urls or []))


def _collect_benefits_sources(job: Optional[JobPosting]) -> List[str]:
    if not job:
        return []
    return _unique_preserve_order([job.job_url] + (job.benefits_source_urls or []))


def _collect_licensing_sources(job: Optional[JobPosting]) -> List[str]:
    if not job:
        return []
    return _unique_preserve_order([job.job_url] + (job.licensing_source_urls or []))


# --------------------------------------------------------------------------- #
# Verification functions per state                                            #
# --------------------------------------------------------------------------- #
async def verify_nc(evaluator: Evaluator, parent_node, job: Optional[JobPosting]) -> None:
    """
    North Carolina: Charlotte-Mecklenburg Schools (CMS)
    Critical leaves: District match, Principal level, Active posting, URL provided
    Non-critical: Salary info supported
    """
    node = evaluator.add_parallel(
        id="North_Carolina_Position",
        desc="Principal job opening in Charlotte-Mecklenburg Schools, North Carolina",
        parent=parent_node,
        critical=False
    )

    # URL provided (critical existence check)
    url_ok = bool(job and job.job_url and job.job_url.strip())
    evaluator.add_custom_node(
        result=url_ok,
        id="NC_URL_Provided",
        desc="A direct URL to the job posting is provided",
        parent=node,
        critical=True
    )

    # District match (critical, verify by URL)
    district_leaf = evaluator.add_leaf(
        id="NC_District_Match",
        desc="The job posting is from Charlotte-Mecklenburg Schools in North Carolina",
        parent=node,
        critical=True
    )
    district_claim = (
        "Based on the job posting page, the employing district is Charlotte-Mecklenburg Schools "
        "(CMS) in North Carolina."
    )
    await evaluator.verify(
        claim=district_claim,
        node=district_leaf,
        sources=(job.job_url if job else None),
        additional_instruction=(
            "Confirm the employer/district shown is Charlotte-Mecklenburg Schools. Accept common alias 'CMS'. "
            "Ignore third-party hosting platforms; focus on the stated district/employer on the page."
        )
    )

    # Principal level (critical)
    principal_leaf = evaluator.add_leaf(
        id="NC_Principal_Level",
        desc="The position is for a principal (not assistant principal or other administrative role)",
        parent=node,
        critical=True
    )
    principal_claim = (
        "The position described on this page is a Principal role (e.g., 'Principal', 'School Principal') and "
        "NOT an Assistant/Associate/Vice Principal or other administrative role."
    )
    await evaluator.verify(
        claim=principal_claim,
        node=principal_leaf,
        sources=(job.job_url if job else None),
        additional_instruction=(
            "Treat titles like 'Assistant Principal', 'Associate Principal', 'Vice Principal', 'AP' as NOT acceptable. "
            "Accept variations like 'Elementary School Principal', 'High School Principal', etc."
        )
    )

    # Active posting (critical)
    active_leaf = evaluator.add_leaf(
        id="NC_Active_Posting",
        desc="The job posting is current and accepting applications",
        parent=node,
        critical=True
    )
    active_claim = (
        f"The posting is currently open and accepting applications as of {AS_OF_MONTH_YEAR} "
        "(e.g., shows an 'Apply' button, status 'Open', 'Accepting Applications', "
        "or a closing date later than March 1, 2026)."
    )
    await evaluator.verify(
        claim=active_claim,
        node=active_leaf,
        sources=(job.job_url if job else None),
        additional_instruction=(
            f"Use cues such as an active 'Apply' button, 'Open until filled', or a future close date relative to {AS_OF_MONTH_YEAR}. "
            "If the page clearly states the posting is closed/expired, mark as not active."
        )
    )

    # Salary info (non-critical)
    salary_leaf = evaluator.add_leaf(
        id="NC_Salary_Info",
        desc="Salary information is provided in the job posting or related district materials",
        parent=node,
        critical=False
    )
    salary_claim = (
        f"The provided salary information for this position is supported by the sources: '{(job.salary_info or '').strip()}'"
        if job and job.salary_info else
        "The provided sources include salary or pay grade information relevant to this principal position."
    )
    await evaluator.verify(
        claim=salary_claim,
        node=salary_leaf,
        sources=_collect_salary_sources(job),
        additional_instruction=(
            "Accept explicit salary ranges, pay grades/steps, or links to official salary schedules that apply to principals. "
            "Minor formatting differences are acceptable."
        )
    )


async def verify_md(evaluator: Evaluator, parent_node, job: Optional[JobPosting]) -> None:
    """
    Maryland: Prince George's County Public Schools (PGCPS)
    """
    node = evaluator.add_parallel(
        id="Maryland_Position",
        desc="Principal job opening in Prince George's County Public Schools, Maryland",
        parent=parent_node,
        critical=False
    )

    # URL provided (critical)
    url_ok = bool(job and job.job_url and job.job_url.strip())
    evaluator.add_custom_node(
        result=url_ok,
        id="MD_URL_Provided",
        desc="A direct URL to the job posting is provided",
        parent=node,
        critical=True
    )

    # District match (critical)
    district_leaf = evaluator.add_leaf(
        id="MD_District_Match",
        desc="The job posting is from Prince George's County Public Schools in Maryland",
        parent=node,
        critical=True
    )
    district_claim = (
        "Based on the job posting page, the employing district is Prince George's County Public Schools (PGCPS) in Maryland."
    )
    await evaluator.verify(
        claim=district_claim,
        node=district_leaf,
        sources=(job.job_url if job else None),
        additional_instruction=(
            "Confirm the employer is PGCPS. Accept the abbreviation 'PGCPS'. "
            "Disregard third-party hosting; focus on the stated district/employer."
        )
    )

    # Principal level (critical)
    principal_leaf = evaluator.add_leaf(
        id="MD_Principal_Level",
        desc="The position is for a principal (not assistant principal or other administrative role)",
        parent=node,
        critical=True
    )
    principal_claim = (
        "The position described on this page is a Principal role, not an Assistant/Associate/Vice Principal or other role."
    )
    await evaluator.verify(
        claim=principal_claim,
        node=principal_leaf,
        sources=(job.job_url if job else None),
        additional_instruction=(
            "Reject 'Assistant Principal', 'Associate Principal', 'Vice Principal', 'AP'. "
            "Accept variants like 'School Principal'."
        )
    )

    # Active posting (critical)
    active_leaf = evaluator.add_leaf(
        id="MD_Active_Posting",
        desc="The job posting is current and accepting applications",
        parent=node,
        critical=True
    )
    active_claim = (
        f"The posting is currently open and accepting applications as of {AS_OF_MONTH_YEAR} "
        "(Apply button/status open/future close date)."
    )
    await evaluator.verify(
        claim=active_claim,
        node=active_leaf,
        sources=(job.job_url if job else None),
        additional_instruction=(
            f"Look for 'Apply' actions, 'Open until filled', or a closing date after March 1, 2026. "
            "Mark as not active if the page states closed/expired."
        )
    )

    # Salary info (non-critical)
    salary_leaf = evaluator.add_leaf(
        id="MD_Salary_Info",
        desc="Salary information is provided in the job posting or related district materials",
        parent=node,
        critical=False
    )
    salary_claim = (
        f"The provided salary information for this position is supported by the sources: '{(job.salary_info or '').strip()}'"
        if job and job.salary_info else
        "The provided sources include salary or pay grade information relevant to this principal position."
    )
    await evaluator.verify(
        claim=salary_claim,
        node=salary_leaf,
        sources=_collect_salary_sources(job),
        additional_instruction=(
            "Salary can be verified from the job posting or official PGCPS salary schedules. "
            "Ranges/grades/steps are acceptable evidence."
        )
    )


async def verify_co(evaluator: Evaluator, parent_node, job: Optional[JobPosting]) -> None:
    """
    Colorado: Jeffco Public Schools (Jefferson County Public Schools)
    Additional non-critical: PERA retirement benefits referenced
    """
    node = evaluator.add_parallel(
        id="Colorado_Position",
        desc="Principal job opening in Jeffco Public Schools, Colorado",
        parent=parent_node,
        critical=False
    )

    # URL provided (critical)
    url_ok = bool(job and job.job_url and job.job_url.strip())
    evaluator.add_custom_node(
        result=url_ok,
        id="CO_URL_Provided",
        desc="A direct URL to the job posting is provided",
        parent=node,
        critical=True
    )

    # District match (critical)
    district_leaf = evaluator.add_leaf(
        id="CO_District_Match",
        desc="The job posting is from Jeffco Public Schools in Colorado",
        parent=node,
        critical=True
    )
    district_claim = (
        "Based on the job posting page, the employing district is Jeffco Public Schools "
        "(also known as Jefferson County Public Schools) in Colorado."
    )
    await evaluator.verify(
        claim=district_claim,
        node=district_leaf,
        sources=(job.job_url if job else None),
        additional_instruction=(
            "Accept 'Jeffco Public Schools' or 'Jefferson County Public Schools' as equivalent. "
            "Verify the page attributes the employer as Jeffco."
        )
    )

    # Principal level (critical)
    principal_leaf = evaluator.add_leaf(
        id="CO_Principal_Level",
        desc="The position is for a principal (not assistant principal or other administrative role)",
        parent=node,
        critical=True
    )
    principal_claim = "This page describes a Principal position, not Assistant/Associate/Vice Principal."
    await evaluator.verify(
        claim=principal_claim,
        node=principal_leaf,
        sources=(job.job_url if job else None),
        additional_instruction="Reject any 'Assistant/Associate/Vice Principal' roles."
    )

    # Active posting (critical)
    active_leaf = evaluator.add_leaf(
        id="CO_Active_Posting",
        desc="The job posting is current and accepting applications",
        parent=node,
        critical=True
    )
    active_claim = f"The posting is open and accepting applications as of {AS_OF_MONTH_YEAR}."
    await evaluator.verify(
        claim=active_claim,
        node=active_leaf,
        sources=(job.job_url if job else None),
        additional_instruction=(
            f"Consider 'Apply' button, 'Open until filled', or a close date after March 1, 2026 as evidence. "
            "Mark as not active if explicitly closed/expired."
        )
    )

    # Salary info (non-critical)
    salary_leaf = evaluator.add_leaf(
        id="CO_Salary_Info",
        desc="Salary information is provided in the job posting or related district materials",
        parent=node,
        critical=False
    )
    salary_claim = (
        f"The provided salary information for this position is supported by the sources: '{(job.salary_info or '').strip()}'"
        if job and job.salary_info else
        "The provided sources include salary or pay grade information relevant to this principal position."
    )
    await evaluator.verify(
        claim=salary_claim,
        node=salary_leaf,
        sources=_collect_salary_sources(job),
        additional_instruction="Accept salary range/grade evidence from the job posting or official Jeffco HR/salary schedules."
    )

    # PERA benefits info (non-critical)
    benefits_leaf = evaluator.add_leaf(
        id="CO_Benefits_Info",
        desc="Information about retirement benefits (specifically PERA) is provided or referenced",
        parent=node,
        critical=False
    )
    benefits_claim = (
        "The benefits information for this Jeffco principal position references PERA "
        "(Colorado Public Employees' Retirement Association) as part of retirement benefits."
    )
    await evaluator.verify(
        claim=benefits_claim,
        node=benefits_leaf,
        sources=_collect_benefits_sources(job),
        additional_instruction=(
            "Look for 'PERA', 'Colorado PERA', or explicit reference to the Colorado Public Employees' Retirement Association "
            "on the job posting or linked Jeffco benefits pages."
        )
    )


async def verify_oh(evaluator: Evaluator, parent_node, job: Optional[JobPosting]) -> None:
    """
    Ohio: Columbus City Schools
    Additional non-critical: Licensing info presence (Ohio principal license requirements)
    """
    node = evaluator.add_parallel(
        id="Ohio_Position",
        desc="Principal job opening in Columbus City Schools, Ohio",
        parent=parent_node,
        critical=False
    )

    # URL provided (critical)
    url_ok = bool(job and job.job_url and job.job_url.strip())
    evaluator.add_custom_node(
        result=url_ok,
        id="OH_URL_Provided",
        desc="A direct URL to the job posting is provided",
        parent=node,
        critical=True
    )

    # District match (critical)
    district_leaf = evaluator.add_leaf(
        id="OH_District_Match",
        desc="The job posting is from Columbus City Schools in Ohio",
        parent=node,
        critical=True
    )
    district_claim = "Based on the job posting page, the employing district is Columbus City Schools (CCS) in Ohio."
    await evaluator.verify(
        claim=district_claim,
        node=district_leaf,
        sources=(job.job_url if job else None),
        additional_instruction="Accept 'CCS' as a common abbreviation if the employer is clearly Columbus City Schools."
    )

    # Principal level (critical)
    principal_leaf = evaluator.add_leaf(
        id="OH_Principal_Level",
        desc="The position is for a principal (not assistant principal or other administrative role)",
        parent=node,
        critical=True
    )
    principal_claim = "This page describes a Principal position, not Assistant/Associate/Vice Principal."
    await evaluator.verify(
        claim=principal_claim,
        node=principal_leaf,
        sources=(job.job_url if job else None),
        additional_instruction="Reject 'Assistant/Associate/Vice Principal' or similar titles."
    )

    # Active posting (critical)
    active_leaf = evaluator.add_leaf(
        id="OH_Active_Posting",
        desc="The job posting is current and accepting applications",
        parent=node,
        critical=True
    )
    active_claim = f"The posting is currently open and accepting applications as of {AS_OF_MONTH_YEAR}."
    await evaluator.verify(
        claim=active_claim,
        node=active_leaf,
        sources=(job.job_url if job else None),
        additional_instruction=(
            f"Look for 'Apply' functionality, 'Open until filled', or a close date after March 1, 2026. "
            "If the page states closed/expired, treat as not active."
        )
    )

    # Licensing info (non-critical)
    licensing_leaf = evaluator.add_leaf(
        id="OH_Licensing_Info",
        desc="Job description references Ohio principal license requirements or educational leadership credentials",
        parent=node,
        critical=False
    )
    licensing_claim = (
        "The job description for this Columbus City Schools principal position references Ohio principal licensure "
        "requirements (e.g., a valid Ohio Principal license, ODE administrator credential) or equivalent credentials."
    )
    await evaluator.verify(
        claim=licensing_claim,
        node=licensing_leaf,
        sources=_collect_licensing_sources(job),
        additional_instruction=(
            "Accept phrasing like 'Valid Ohio Principal License', 'Ohio Department of Education (ODE) principal license', "
            "'appropriate Ohio administrative certificate', or similar."
        )
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
    Evaluate an answer for four principal positions across specified districts/states.
    """
    # Initialize evaluator (root is non-critical aggregator)
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

    # Extract structured positions from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_principal_jobs(),
        template_class=PrincipalJobsExtraction,
        extraction_name="principal_jobs_extraction"
    )

    # Add ground truth expectations (for context)
    evaluator.add_ground_truth({
        "expected_districts": {
            "NC": DISTRICT_SPEC["NC"]["district"],
            "MD": DISTRICT_SPEC["MD"]["district"],
            "CO": DISTRICT_SPEC["CO"]["district"],
            "OH": DISTRICT_SPEC["OH"]["district"],
        },
        "as_of": AS_OF_MONTH_YEAR
    })

    # Add a top-level node mirroring the rubric root
    top_node = evaluator.add_parallel(
        id="Four_Principal_Positions",
        desc="Find four principal job openings, one in each of four specified states and school districts",
        parent=root,
        critical=False
    )

    # Build verification subtrees, one per state/district
    await verify_nc(evaluator, top_node, extracted.north_carolina)
    await verify_md(evaluator, top_node, extracted.maryland)
    await verify_co(evaluator, top_node, extracted.colorado)
    await verify_oh(evaluator, top_node, extracted.ohio)

    # Return the evaluation summary
    return evaluator.get_summary()