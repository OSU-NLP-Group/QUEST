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
TASK_ID = "remote_data_analyst_one_company"
TASK_DESCRIPTION = (
    "Identify one technology company that has a permanent remote work policy and is currently hiring for an "
    "entry-level data analyst position. For your answer, provide: (1) the company name, (2) a direct link to the "
    "company's official page describing their permanent remote work policy, and (3) a direct link to a current job "
    "posting for a data analyst position that accepts candidates with 0-2 years of experience (or internship/academic "
    "experience), requires no more than a bachelor's degree as the minimum education, and lists at least two of the "
    "following core skills: SQL, Python, Excel, Tableau, or Power BI."
)

ALLOWED_SKILLS = ["SQL", "Python", "Excel", "Tableau", "Power BI"]
SALARY_MIN = 60000
SALARY_MAX = 136000


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class AnswerExtraction(BaseModel):
    company_name: Optional[str] = None
    remote_policy_url: Optional[str] = None
    job_posting_url: Optional[str] = None

    # Optional extras if the agent included them explicitly in the answer (not required for verification logic)
    job_title_in_answer: Optional[str] = None
    experience_requirement_text: Optional[str] = None
    minimum_education_text: Optional[str] = None
    skills_mentioned: List[str] = Field(default_factory=list)
    salary_text_in_answer: Optional[str] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_main() -> str:
    return f"""
Extract from the answer the following fields, strictly based on what is explicitly written:

- company_name: the name of the identified company (string).
- remote_policy_url: a direct URL to an official company page (e.g., company site, official blog, careers site, or ATS subdomain officially used by the company) that describes the company's remote work policy. If not present, return null.
- job_posting_url: a direct URL to the current/active job posting page for the data analyst role. This should be a single posting page (not a search results page). If not present, return null.

Additionally, if the answer explicitly states them, extract the following (otherwise set to null or empty):
- job_title_in_answer: the job title text as stated in the answer.
- experience_requirement_text: quoted/near-exact text in the answer that describes acceptable years of experience or mentions 0–2 years or internship/academic experience.
- minimum_education_text: quoted/near-exact text in the answer describing minimum education.
- skills_mentioned: list of strings of any skills from this set if explicitly written in the answer: {ALLOWED_SKILLS}.
- salary_text_in_answer: any salary/compensation text mentioned in the answer.

Return a JSON object following the AnswerExtraction schema. If any field is missing in the answer, return null (or [] for lists).
"""


# --------------------------------------------------------------------------- #
# Helper: Additional instructions for verifier                                #
# --------------------------------------------------------------------------- #
def addins_company_is_technology(company_name: str) -> str:
    return (
        "Determine if the company is a 'technology company' based on the provided page(s). "
        "Clues include being a software, hardware, cloud, data, platform, or IT services provider; "
        "self-descriptions like 'tech company', 'software company', 'technology firm', 'SaaS', "
        "'cloud-native', 'AI company', 'analytics platform', etc. Avoid using outside knowledge. "
        "Rely on the text/screenshot on the provided official pages. "
        f"Company to check: {company_name}. Minor paraphrasing is fine."
    )


def addins_policy_permanent_remote() -> str:
    return (
        "Verify that the page explicitly supports a permanent or fully-remote policy (e.g., 'remote-first', "
        "'fully distributed', 'permanent remote', 'remote by default'). Hybrid-only or temporary remote policies do "
        "NOT satisfy this. The page should be on an official company-controlled domain or official blog/careers/ATS "
        "page. If the page is a third-party news article or unrelated blog, reject."
    )


def addins_job_link_is_active() -> str:
    return (
        "Verify this URL is a current, active job posting page for a single role (not a general listing). "
        "Signs of active include 'Apply' buttons, 'Accepting applications', or recent posting dates; "
        "avoid pages clearly marked 'Closed', 'No longer accepting applications', or 'Archived'. "
        "Pages hosted on official company domains or official ATS platforms (e.g., Workday, Greenhouse, Lever) are okay."
    )


def addins_job_is_data_analyst() -> str:
    return (
        "Verify the posting is for a 'Data Analyst' role. Accept close variations such as 'Data Analyst I/II', "
        "'Junior Data Analyst', or 'Data Analytics Analyst'. Do NOT accept 'Business Analyst', 'Data Scientist', "
        "'Data Engineer', or unrelated roles."
    )


def addins_entry_level_experience() -> str:
    return (
        "Verify that the posting accepts candidates with 0–2 years of experience OR explicitly allows internship/academic "
        "experience OR labels the role as 'Entry Level' or 'New Grad'. Phrases like 'up to 2 years', '0-2 years', "
        "'internship experience acceptable', 'academic project experience acceptable' all qualify."
    )


def addins_education_max_bachelors() -> str:
    return (
        "Verify the minimum education requirement is at most a Bachelor's degree (or lower such as Associate's or High School). "
        "If a Master's is preferred (not required), it still qualifies. Reject if the minimum requirement is a Master's or PhD."
    )


def addins_two_required_skills() -> str:
    skills = ", ".join(ALLOWED_SKILLS)
    return (
        f"Verify the posting explicitly lists at least two of these skills in qualifications/responsibilities/requirements: {skills}. "
        "Count synonyms or common variants as matches (e.g., 'MS Excel' counts as Excel)."
    )


def addins_salary_in_range() -> str:
    return (
        f"Consider this claim satisfied (Correct) if the posting either: "
        f"(a) does NOT list any salary/compensation information, OR "
        f"(b) lists an annual salary (or clearly annualized total cash compensation, excluding equity-only) within "
        f"${SALARY_MIN:,}–${SALARY_MAX:,} USD. If an hourly rate is given, convert to annual with ~2080 hours/year. "
        "If multiple locations list ranges, evaluate the one applicable to remote U.S. candidates. "
        "If compensation is listed but clearly outside the range, mark Incorrect."
    )


# --------------------------------------------------------------------------- #
# Verification builder                                                        #
# --------------------------------------------------------------------------- #
async def build_and_run_verifications(evaluator: Evaluator, extraction: AnswerExtraction) -> None:
    """
    Build the verification tree according to the rubric and run checks.
    Adjusted criticalities:
    - Root is non-critical to allow partial scoring capture.
    - All rubric-required items are implemented as critical leaves so they gate correctness.
    - Salary is treated as critical in line with rubric (but passes by default when salary not listed).
    """
    root = evaluator.root

    # Basic existence checks for required fields (critical)
    company_present_node = evaluator.add_custom_node(
        result=bool(extraction.company_name and extraction.company_name.strip()),
        id="Answer_Provides_Company_Name",
        desc="Response provides the company name.",
        parent=root,
        critical=True
    )

    policy_url_present_node = evaluator.add_custom_node(
        result=bool(extraction.remote_policy_url and extraction.remote_policy_url.strip()),
        id="Answer_Provides_Remote_Policy_Link",
        desc="Response provides a direct URL link to an official company page describing the permanent/fully remote work policy.",
        parent=root,
        critical=True
    )

    job_url_present_node = evaluator.add_custom_node(
        result=bool(extraction.job_posting_url and extraction.job_posting_url.strip()),
        id="Answer_Provides_Job_Posting_Link",
        desc="Response provides a direct URL link to a current (active) job posting for the role.",
        parent=root,
        critical=True
    )

    # Build verification leaves that rely on URLs
    company_sources: List[str] = []
    if extraction.remote_policy_url:
        company_sources.append(extraction.remote_policy_url)
    if extraction.job_posting_url:
        company_sources.append(extraction.job_posting_url)

    # Company is technology company
    company_is_tech_node = evaluator.add_leaf(
        id="Company_Is_Technology_Company",
        desc="The identified company is a technology company.",
        parent=root,
        critical=True
    )
    company_name = extraction.company_name or "the company"
    company_is_tech_claim = f"{company_name} is a technology company."
    await evaluator.verify(
        claim=company_is_tech_claim,
        node=company_is_tech_node,
        sources=company_sources if company_sources else None,
        additional_instruction=addins_company_is_technology(company_name)
    )

    # Remote policy is permanent/fully remote
    policy_perm_node = evaluator.add_leaf(
        id="Company_Policy_Is_Permanent_Remote",
        desc="The remote-work policy is explicitly stated as permanent or fully remote (not hybrid/temporary).",
        parent=root,
        critical=True
    )
    await evaluator.verify(
        claim="The company's remote work policy is permanent or fully remote (remote-first/fully distributed).",
        node=policy_perm_node,
        sources=extraction.remote_policy_url if extraction.remote_policy_url else None,
        additional_instruction=addins_policy_permanent_remote()
    )

    # Job posting link is an active job posting page (not search page)
    job_active_node = evaluator.add_leaf(
        id="Job_Posting_Link_Is_Active_Verified",
        desc="The provided job posting URL is a current, active single-role posting page.",
        parent=root,
        critical=True
    )
    await evaluator.verify(
        claim="This URL is a current, active job posting page for a single role (not a general search list).",
        node=job_active_node,
        sources=extraction.job_posting_url if extraction.job_posting_url else None,
        additional_instruction=addins_job_link_is_active()
    )

    # Job: Title is Data Analyst
    job_title_node = evaluator.add_leaf(
        id="Job_Posting_Is_Data_Analyst_Title",
        desc="The job posting is specifically for a 'Data Analyst' role.",
        parent=root,
        critical=True
    )
    await evaluator.verify(
        claim="This job posting is for a 'Data Analyst' role (including variations like Data Analyst I/II or Junior Data Analyst), not Business Analyst or Data Scientist.",
        node=job_title_node,
        sources=extraction.job_posting_url if extraction.job_posting_url else None,
        additional_instruction=addins_job_is_data_analyst()
    )

    # Job: Entry-level experience
    job_entry_level_node = evaluator.add_leaf(
        id="Job_Is_Entry_Level_Experience",
        desc="The posting accepts candidates with 0–2 years experience OR allows internship/academic experience.",
        parent=root,
        critical=True
    )
    await evaluator.verify(
        claim="The posting accepts 0–2 years of experience OR internship/academic experience OR marks the role as Entry Level/New Grad.",
        node=job_entry_level_node,
        sources=extraction.job_posting_url if extraction.job_posting_url else None,
        additional_instruction=addins_entry_level_experience()
    )

    # Job: Education max Bachelor's
    job_edu_node = evaluator.add_leaf(
        id="Job_Education_Max_Bachelors",
        desc="The posting does not require more than a Bachelor's degree as the minimum education qualification.",
        parent=root,
        critical=True
    )
    await evaluator.verify(
        claim="The minimum education requirement is at most a Bachelor's degree (or lower); Master's/PhD not required.",
        node=job_edu_node,
        sources=extraction.job_posting_url if extraction.job_posting_url else None,
        additional_instruction=addins_education_max_bachelors()
    )

    # Job: At least two required skills among SQL, Python, Excel, Tableau, Power BI
    job_skills_node = evaluator.add_leaf(
        id="Job_Lists_Two_Required_Skills",
        desc="The posting lists at least two of: SQL, Python, Excel, Tableau, Power BI.",
        parent=root,
        critical=True
    )
    skills_list = ", ".join(ALLOWED_SKILLS)
    await evaluator.verify(
        claim=f"The posting lists at least two of the following skills: {skills_list}.",
        node=job_skills_node,
        sources=extraction.job_posting_url if extraction.job_posting_url else None,
        additional_instruction=addins_two_required_skills()
    )

    # Salary: If provided, in range
    salary_node = evaluator.add_leaf(
        id="Salary_If_Provided_In_Range",
        desc=f"If salary is provided, it falls within ${SALARY_MIN:,}–${SALARY_MAX:,} annually for remote U.S. data analyst positions.",
        parent=root,
        critical=True  # Honor rubric criticality but pass-by-default if salary absent (see instruction)
    )
    await evaluator.verify(
        claim=f"Either the posting lists no salary, or any listed annual salary is within ${SALARY_MIN:,}–${SALARY_MAX:,} USD.",
        node=salary_node,
        sources=extraction.job_posting_url if extraction.job_posting_url else None,
        additional_instruction=addins_salary_in_range()
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
    model: str = "o4-mini",
) -> Dict:
    """
    Evaluate an answer for identifying a single technology company with a permanent remote policy and
    a current entry-level Data Analyst job posting, with required links and constraints.
    """
    # Initialize evaluator (root is non-critical parallel to allow partial credit reporting)
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
        default_model=model,
    )

    # Extract structured info from the answer text
    extraction = await evaluator.extract(
        prompt=prompt_extract_main(),
        template_class=AnswerExtraction,
        extraction_name="answer_extraction_main",
    )

    # Record reference info for debugging
    evaluator.add_custom_info(
        {
            "allowed_skills": ALLOWED_SKILLS,
            "salary_range_usd": [SALARY_MIN, SALARY_MAX],
        },
        info_type="rubric_parameters",
        info_name="rubric_parameters",
    )

    # Build and execute verification checks
    await build_and_run_verifications(evaluator, extraction)

    # Return evaluator summary
    return evaluator.get_summary()