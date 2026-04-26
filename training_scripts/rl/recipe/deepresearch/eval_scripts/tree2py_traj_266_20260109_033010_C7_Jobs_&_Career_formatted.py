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
TASK_ID = "us_swe_job_benefits_eval"
TASK_DESCRIPTION = (
    "Identify a current software engineering job position in the United States that meets all of the following criteria: "
    "(1) offers a minimum annual base salary of $100,000, (2) provides H-1B visa sponsorship for international candidates, "
    "(3) allows fully remote work, (4) includes comprehensive health insurance coverage, (5) offers a 401(k) retirement plan "
    "with employer matching, (6) provides an annual professional development or learning budget of at least $1,000 per employee, "
    "(7) offers minimum 15 days of paid time off annually or an unlimited PTO policy, (8) includes annual performance bonus "
    "opportunities of at least 10% of base salary, (9) provides equity compensation through stock options or RSUs, "
    "(10) offers relocation assistance packages, (11) includes a signing bonus of at least $10,000, "
    "(12) requires a Bachelor's degree in Computer Science or related field OR accepts equivalent practical experience, "
    "(13) is suitable for candidates with 0-3 years of professional experience, and (14) is with a company that has at least 100 employees. "
    "Provide the job title, company name, and a reference URL to the job posting or company careers page that confirms these benefits."
)

# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class JobExtraction(BaseModel):
    # Required identifying fields
    job_title: Optional[str] = None
    company_name: Optional[str] = None
    reference_urls: List[str] = Field(default_factory=list)

    # Optional helpful context fields (strings preferred; can be null)
    location_text: Optional[str] = None
    remote_policy_text: Optional[str] = None

    base_salary_text: Optional[str] = None
    visa_policy_text: Optional[str] = None
    health_insurance_text: Optional[str] = None
    retirement_401k_text: Optional[str] = None
    professional_dev_budget_text: Optional[str] = None
    pto_text: Optional[str] = None
    bonus_text: Optional[str] = None
    equity_text: Optional[str] = None
    relocation_text: Optional[str] = None
    signing_bonus_text: Optional[str] = None
    education_requirements_text: Optional[str] = None
    experience_level_text: Optional[str] = None
    company_size_text: Optional[str] = None
    posting_status_text: Optional[str] = None
    role_type_text: Optional[str] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_job() -> str:
    return """
    Select exactly one job position referenced in the answer that best matches the task. If multiple jobs are listed, pick the first one that appears to meet the constraints; otherwise choose the first listed job. Extract the following fields from the answer text:

    1) job_title: The job title of the selected position.
    2) company_name: The company offering the position.
    3) reference_urls: An array of all URLs mentioned in the answer that are relevant to this chosen position (include the job posting, company careers page, and any benefits or company information pages the answer cites). Deduplicate and ensure valid URLs with http:// or https://.

    Also extract the following helper fields AS PLAIN TEXT EXCERPTS if present (or null if not mentioned):
    - location_text
    - remote_policy_text
    - base_salary_text
    - visa_policy_text
    - health_insurance_text
    - retirement_401k_text
    - professional_dev_budget_text
    - pto_text
    - bonus_text
    - equity_text
    - relocation_text
    - signing_bonus_text
    - education_requirements_text
    - experience_level_text
    - company_size_text
    - posting_status_text
    - role_type_text

    IMPORTANT:
    - Do not invent any information not present in the answer.
    - Return null for any field not explicitly supported by the answer text.
    - For reference_urls, capture every explicit URL (plain or markdown links) related to the selected job in the answer.
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _mk_job_label(ex: JobExtraction) -> str:
    jt = (ex.job_title or "the position").strip()
    co = (ex.company_name or "the company").strip()
    return f"{jt} at {co}"


async def _add_leaf_and_verify(
    evaluator: Evaluator,
    parent,
    node_id: str,
    desc: str,
    claim: str,
    sources: Optional[List[str]],
    additional_instruction: str,
    critical: bool = True,
):
    node = evaluator.add_leaf(
        id=node_id,
        desc=desc,
        parent=parent,
        critical=critical,
    )
    await evaluator.verify(
        claim=claim,
        node=node,
        sources=sources if (sources and len(sources) > 0) else None,
        additional_instruction=additional_instruction,
    )


# --------------------------------------------------------------------------- #
# Build verification tree                                                     #
# --------------------------------------------------------------------------- #
async def build_job_verification_tree(evaluator: Evaluator, root, ex: JobExtraction):
    # Top-level compliance node (critical)
    top = evaluator.add_parallel(
        id="job_position_compliance",
        desc="The response identifies one current US software engineering job and provides required fields; the job meets all specified compensation/benefit and eligibility criteria.",
        parent=root,
        critical=True
    )

    # 1) Required output fields (critical)
    req = evaluator.add_parallel(
        id="required_output_fields",
        desc="Response includes the required identifying fields for the job and a reference link.",
        parent=top,
        critical=True
    )
    evaluator.add_custom_node(
        result=bool(ex.job_title and ex.job_title.strip()),
        id="job_title_provided",
        desc="Provides a job title for the identified position.",
        parent=req,
        critical=True
    )
    evaluator.add_custom_node(
        result=bool(ex.company_name and ex.company_name.strip()),
        id="company_name_provided",
        desc="Provides the company name for the identified position.",
        parent=req,
        critical=True
    )
    evaluator.add_custom_node(
        result=bool(ex.reference_urls and len(ex.reference_urls) > 0),
        id="reference_url_provided",
        desc="Provides a reference URL to a job posting or company careers page.",
        parent=req,
        critical=True
    )

    job_label = _mk_job_label(ex)
    sources = ex.reference_urls if ex.reference_urls else None

    # 2) Position basics (critical)
    basics = evaluator.add_parallel(
        id="position_basics",
        desc="The identified role matches the basic scope requirements of the question.",
        parent=top,
        critical=True
    )

    await _add_leaf_and_verify(
        evaluator,
        basics,
        "us_location",
        "Job position is in the United States.",
        claim=f"The job posting for {job_label} indicates the role is in the United States, or it is fully remote but explicitly open to US-based candidates.",
        sources=sources,
        additional_instruction=(
            "Check the job page(s) to confirm US location eligibility. Accept phrases like 'US-remote', "
            "'United States', 'remote within the US', or a listed US city/state. If the role is worldwide-remote "
            "but explicitly open to US-based candidates, consider this supported."
        ),
        critical=True
    )

    await _add_leaf_and_verify(
        evaluator,
        basics,
        "current_job_position",
        "Job position is current/open (i.e., an active posting/role at time of answer).",
        claim=f"The job posting for {job_label} is currently open and accepting applications (not closed or archived).",
        sources=sources,
        additional_instruction=(
            "Look for signals such as an active 'Apply' button, an open/active status, recent posting date, or "
            "explicit wording that the company is hiring now. If the page clearly indicates the role is closed or archived, do not support."
        ),
        critical=True
    )

    await _add_leaf_and_verify(
        evaluator,
        basics,
        "software_engineering_role",
        "Position is a software engineering job position.",
        claim=f"The role {job_label} is a software engineering position (e.g., Software Engineer, SWE, SDE, or similar).",
        sources=sources,
        additional_instruction=(
            "Confirm from the page that the role is in software engineering or equivalent scope (e.g., backend engineer, "
            "frontend engineer, full-stack engineer, SDE). Roles like data analyst or IT support should not count unless clearly "
            "framed as software engineering."
        ),
        critical=True
    )

    # 3) Compensation, benefits, eligibility (critical)
    comp = evaluator.add_parallel(
        id="compensation_benefits_and_eligibility",
        desc="The job satisfies all compensation, benefits, and candidate requirement constraints listed.",
        parent=top,
        critical=True
    )

    await _add_leaf_and_verify(
        evaluator,
        comp,
        "minimum_base_salary",
        "Position offers annual base salary of at least $100,000.",
        claim=f"The job posting for {job_label} indicates an annual base salary of at least $100,000.",
        sources=sources,
        additional_instruction=(
            "Verify the base salary (not total compensation or OTE) is >= $100,000. Salary ranges qualify if the minimum or a stated base meets/exceeds $100,000."
        ),
        critical=True
    )

    await _add_leaf_and_verify(
        evaluator,
        comp,
        "visa_sponsorship",
        "Employer sponsors H-1B visas for international candidates.",
        claim=f"H-1B visa sponsorship is available for the role {job_label}.",
        sources=sources,
        additional_instruction=(
            "Look for explicit mention of 'H-1B' or clear visa sponsorship statements. Wording like 'no sponsorship' or 'must be authorized to work without sponsorship' disqualifies."
        ),
        critical=True
    )

    await _add_leaf_and_verify(
        evaluator,
        comp,
        "remote_work_policy",
        "Position allows fully remote work arrangement.",
        claim=f"The role {job_label} allows fully remote work.",
        sources=sources,
        additional_instruction=(
            "Accept 'fully remote', 'remote (US)', or equivalent. 'Hybrid' or 'on-site' does not qualify as fully remote."
        ),
        critical=True
    )

    await _add_leaf_and_verify(
        evaluator,
        comp,
        "health_insurance",
        "Comprehensive health insurance coverage is provided.",
        claim=f"The compensation/benefits for {job_label} include comprehensive health insurance coverage.",
        sources=sources,
        additional_instruction=(
            "Look for medical coverage and preferably mentions of dental/vision. General 'benefits' without mention of health/medical do not qualify."
        ),
        critical=True
    )

    await _add_leaf_and_verify(
        evaluator,
        comp,
        "retirement_benefits",
        "401(k) retirement plan with employer matching contributions is offered.",
        claim=f"The benefits for {job_label} include a 401(k) plan with employer matching.",
        sources=sources,
        additional_instruction=(
            "Must include both 401(k) and employer match. If only 401(k) is mentioned without matching, do not support."
        ),
        critical=True
    )

    await _add_leaf_and_verify(
        evaluator,
        comp,
        "professional_development_budget",
        "Annual professional development or learning budget of at least $1,000 per employee is provided.",
        claim=f"The benefits for {job_label} include an annual professional development/learning budget of at least $1,000 per employee.",
        sources=sources,
        additional_instruction=(
            "Look for explicit amounts like '$1,000 annual learning budget', 'education stipend', 'L&D budget' of $1,000 or more."
        ),
        critical=True
    )

    await _add_leaf_and_verify(
        evaluator,
        comp,
        "paid_time_off",
        "Minimum 15 days of paid time off annually, or unlimited PTO policy is offered.",
        claim=f"The benefits for {job_label} include at least 15 days of PTO annually or an unlimited PTO policy.",
        sources=sources,
        additional_instruction=(
            "Accept if PTO >= 15 days, or 'unlimited PTO' is indicated. Fewer than 15 days does not qualify."
        ),
        critical=True
    )

    await _add_leaf_and_verify(
        evaluator,
        comp,
        "performance_bonus",
        "Annual performance bonus opportunity of at least 10% of base salary is available.",
        claim=f"The compensation for {job_label} includes an annual performance bonus of at least 10% of base salary.",
        sources=sources,
        additional_instruction=(
            "Look for target/annual bonus >= 10% of base. If only 'bonus eligible' with no percentage or a percentage < 10%, do not support."
        ),
        critical=True
    )

    await _add_leaf_and_verify(
        evaluator,
        comp,
        "equity_compensation",
        "Stock options or RSU grants are included as part of the compensation package.",
        claim=f"The compensation for {job_label} includes equity (stock options or RSUs).",
        sources=sources,
        additional_instruction=(
            "Accept explicit mention of 'RSUs', 'stock options', or 'equity'."
        ),
        critical=True
    )

    await _add_leaf_and_verify(
        evaluator,
        comp,
        "relocation_assistance",
        "Relocation assistance package is available.",
        claim=f"Relocation assistance is offered for {job_label}.",
        sources=sources,
        additional_instruction=(
            "Look for 'relocation assistance' or similar language. If explicitly not offered, do not support."
        ),
        critical=True
    )

    await _add_leaf_and_verify(
        evaluator,
        comp,
        "signing_bonus",
        "One-time signing bonus of at least $10,000 is offered.",
        claim=f"A one-time signing bonus of at least $10,000 is offered for {job_label}.",
        sources=sources,
        additional_instruction=(
            "Must explicitly indicate a signing bonus of $10,000 or more. Generic 'sign-on bonus' without amount does not qualify."
        ),
        critical=True
    )

    await _add_leaf_and_verify(
        evaluator,
        comp,
        "educational_requirements",
        "Position requires a Bachelor's degree in Computer Science or related field, OR accepts equivalent practical experience.",
        claim=f"The {job_label} posting requires a Bachelor's degree in CS or related field OR explicitly accepts equivalent practical experience.",
        sources=sources,
        additional_instruction=(
            "Accept statements like 'BS in CS or related field' or 'or equivalent practical experience'. If strictly requires a degree without equivalency accepted, it still qualifies; equivalency is optional alternative."
        ),
        critical=True
    )

    await _add_leaf_and_verify(
        evaluator,
        comp,
        "experience_level",
        "Position is suitable for candidates with 0–3 years of professional software engineering experience.",
        claim=f"The {job_label} role is suitable for candidates with 0–3 years of professional experience (e.g., new grad, entry-level, or 1–3 years).",
        sources=sources,
        additional_instruction=(
            "Look for 'entry level', 'new grad', or requirements like '0–3 years' or '1–3 years'. If the posting requires 4+ years, do not support."
        ),
        critical=True
    )

    await _add_leaf_and_verify(
        evaluator,
        comp,
        "company_size",
        "Employing company has at least 100 employees.",
        claim=f"The company offering {job_label} has at least 100 employees.",
        sources=sources,
        additional_instruction=(
            "Verify company size from the provided job/careers/about sources. Accept statements like '100+ employees', 'over 100 employees', etc. "
            "If no size information is present in the provided sources, do not support."
        ),
        critical=True
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
        default_model=model
    )

    # Extraction
    extracted_job = await evaluator.extract(
        prompt=prompt_extract_job(),
        template_class=JobExtraction,
        extraction_name="selected_job"
    )

    # Optional: record some custom info
    evaluator.add_custom_info(
        info={
            "job_title": extracted_job.job_title,
            "company_name": extracted_job.company_name,
            "reference_urls_count": len(extracted_job.reference_urls),
        },
        info_type="extraction_summary",
        info_name="selected_job_overview"
    )

    # Build verification tree and run checks
    await build_job_verification_tree(evaluator, root, extracted_job)

    # Return structured result
    return evaluator.get_summary()