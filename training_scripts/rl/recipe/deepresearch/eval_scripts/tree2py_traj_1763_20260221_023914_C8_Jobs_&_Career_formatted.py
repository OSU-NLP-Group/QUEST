import asyncio
import logging
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field
from obj_task_eval.llm_client.base_client import LLMClient

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy, VerificationNode

# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "univ_benefits_comp"
TASK_DESCRIPTION = """
Among the universities New York University (NYU), University of Notre Dame, Case Western Reserve University, and Ferris State University, compare their employee benefits by providing for each institution: (1) the employer retirement contribution rate (as a percentage of salary or tiered structure), (2) the employee tuition benefit (number of credits per academic year or coverage description), (3) the dependent undergraduate tuition benefit (percentage coverage, credit transferability, or specific description), and (4) whether an on-site wellness or fitness center is available exclusively for employees and their families. For each benefit category, provide the specific value or description and include reference URLs from official university human resources or benefits pages.
"""

UNIVERSITY_META = {
    "nyu": {
        "name": "New York University (NYU)",
        "official_domains": ["nyu.edu"]
    },
    "notre_dame": {
        "name": "University of Notre Dame",
        "official_domains": ["nd.edu"]
    },
    "cwru": {
        "name": "Case Western Reserve University",
        "official_domains": ["case.edu", "cwru.edu"]
    },
    "ferris_state": {
        "name": "Ferris State University",
        "official_domains": ["ferris.edu", "ferrisstate.edu"]
    }
}

# --------------------------------------------------------------------------- #
# Data models                                                                 #
# --------------------------------------------------------------------------- #
class BenefitEntry(BaseModel):
    # Applicability to staff/administrative employees (not faculty-only)
    applicability_text: Optional[str] = None
    applicability_urls: List[str] = Field(default_factory=list)

    # Retirement – employer contribution portion (percent or tiered)
    retirement_employer_contribution: Optional[str] = None
    retirement_urls: List[str] = Field(default_factory=list)

    # Employee tuition benefit (credits/year or coverage description)
    employee_tuition_benefit: Optional[str] = None
    employee_tuition_urls: List[str] = Field(default_factory=list)

    # Dependent undergraduate tuition benefit (coverage %, transferability, description)
    dependent_undergrad_benefit: Optional[str] = None
    dependent_undergrad_urls: List[str] = Field(default_factory=list)

    # Wellness/Fitness center availability and exclusivity
    wellness_center_available: Optional[bool] = None
    wellness_exclusive_to_employees: Optional[bool] = None
    wellness_fitness_description: Optional[str] = None
    wellness_urls: List[str] = Field(default_factory=list)

    # Currentness/Plan year or effective date citation
    plan_year_or_effective_date: Optional[str] = None
    currentness_urls: List[str] = Field(default_factory=list)


class UniversitiesBenefitsExtraction(BaseModel):
    nyu: Optional[BenefitEntry] = None
    notre_dame: Optional[BenefitEntry] = None
    cwru: Optional[BenefitEntry] = None
    ferris_state: Optional[BenefitEntry] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_universities_benefits() -> str:
    return """
    Extract structured benefits information for the following four universities exactly as presented in the answer text. For EACH university, return the following fields and URLs the answer provides. Do NOT invent any values or URLs.

    Universities:
    - New York University (NYU)
    - University of Notre Dame
    - Case Western Reserve University
    - Ferris State University

    For each university, extract the following fields:

    1) applicability_text: A short sentence indicating that the benefits described apply to staff/administrative employees (not faculty-only). Extract only if the answer explicitly states applicability to staff/administrative or non-faculty employees.
    2) applicability_urls: All URLs cited in the answer that support applicability statements (e.g., HR/benefits pages that are clearly for staff/administrative employees).

    3) retirement_employer_contribution: The employer retirement contribution rate (percent of salary and/or tiered structure). This must explicitly be the employer portion (not employee elective deferrals). If tiered or conditional (e.g., match up to X%), extract the description verbatim from the answer.
    4) retirement_urls: URLs cited in the answer that support the retirement employer contribution. Prefer official HR/benefits pages on the university’s .edu domain.

    5) employee_tuition_benefit: The employee tuition benefit (number of credits per academic year OR coverage description). Ensure it applies to the employee (not dependents).
    6) employee_tuition_urls: URLs cited in the answer that support the employee tuition benefit.

    7) dependent_undergrad_benefit: The dependent undergraduate tuition benefit (coverage %, transferability/eligibility rules, discount structure, or description). Ensure it applies to dependents (not employees).
    8) dependent_undergrad_urls: URLs cited in the answer that support the dependent undergraduate tuition benefit.

    9) wellness_center_available: Boolean indicating whether an on-site (physically on campus) wellness/fitness center is available for employees. If ambiguous in the answer, return null.
    10) wellness_exclusive_to_employees: Boolean indicating whether access is exclusive to employees and their families (i.e., NOT a general campus facility). If ambiguous in the answer, return null.
    11) wellness_fitness_description: Brief description of the facility/access from the answer.
    12) wellness_urls: URLs cited in the answer that support the wellness/fitness facility and its access/exclusivity.

    13) plan_year_or_effective_date: The plan year or effective date mentioned (e.g., "2025–2026", "Effective July 1, 2025"). If the answer mentions a specific plan year/effective date, extract it verbatim. If none is stated, return null.
    14) currentness_urls: URLs cited in the answer that indicate a plan year or effective date or otherwise show the most recent documentation.

    URL extraction rules:
    - Extract only URLs explicitly present in the answer. Do not invent URLs.
    - Prefer official HR/benefits pages on the university’s .edu domain.
    - If no URL is cited for a category, return an empty array for that category.

    Return a JSON object with keys: nyu, notre_dame, cwru, ferris_state. Each key should map to the object containing the fields above for that university. If a university is not covered in the answer, set that key to null.
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def _is_official_url(url: str, uni_key: str) -> bool:
    if not url or not isinstance(url, str):
        return False
    domains = UNIVERSITY_META.get(uni_key, {}).get("official_domains", [])
    u = url.lower()
    if not u.startswith(("http://", "https://")):
        return False
    return any(d in u for d in domains)


def _has_official_url(urls: List[str], uni_key: str) -> bool:
    return any(_is_official_url(u, uni_key) for u in urls or [])


def _non_empty(s: Optional[str]) -> bool:
    return bool(s and isinstance(s, str) and s.strip() != "")


def _union_urls(entry: BenefitEntry) -> List[str]:
    """Union of all URL arrays for a university entry."""
    all_urls = []
    if entry is None:
        return all_urls
    for arr in [
        entry.applicability_urls,
        entry.retirement_urls,
        entry.employee_tuition_urls,
        entry.dependent_undergrad_urls,
        entry.wellness_urls,
        entry.currentness_urls,
    ]:
        for u in arr or []:
            if u and u not in all_urls:
                all_urls.append(u)
    return all_urls


# --------------------------------------------------------------------------- #
# Verification builders                                                       #
# --------------------------------------------------------------------------- #
async def verify_university_benefits(
    evaluator: Evaluator,
    parent: VerificationNode,
    uni_key: str,
    uni_node_id: str,
    uni_node_desc: str,
    entry: Optional[BenefitEntry],
) -> None:
    """
    Build and verify the sub-tree for a single university.
    All children are critical due to rubric requirement.
    """
    uni_name = UNIVERSITY_META.get(uni_key, {}).get("name", uni_key)

    uni_node = evaluator.add_parallel(
        id=uni_node_id,
        desc=uni_node_desc,
        parent=parent,
        critical=True  # Root child nodes are critical
    )

    # ---------------- Applicability ----------------
    app_exist = evaluator.add_custom_node(
        result=(entry is not None and _non_empty(entry.applicability_text) and _has_official_url(entry.applicability_urls, uni_key)),
        id=f"{uni_node_id}_Applicability_Exists",
        desc=f"{uni_name} applicability statement present and supported by official URL(s).",
        parent=uni_node,
        critical=True
    )

    app_leaf = evaluator.add_leaf(
        id=f"{uni_node_id.split('_')[0]}_Staff_Admin_Applicability",
        desc="States the benefits described apply to staff/administrative employees (not faculty-only).",
        parent=uni_node,
        critical=True
    )
    app_sources = entry.applicability_urls if entry else []
    app_claim = f"For {uni_name}, the benefits applicability statement in the answer is: '{entry.applicability_text if entry else ''}'. This applicability refers to staff/administrative employees (not faculty-only) and is supported by the cited official HR/benefits page(s)."
    await evaluator.verify(
        claim=app_claim,
        node=app_leaf,
        sources=app_sources,
        additional_instruction="Verify the page(s) explicitly indicate applicability to staff/administrative or non-faculty employees. If page is faculty-only or student-only, mark incorrect. Ensure URLs are on the university’s official .edu domain.",
        extra_prerequisites=[app_exist],
    )

    # ---------------- Retirement ----------------
    ret_exist = evaluator.add_custom_node(
        result=(entry is not None and _non_empty(entry.retirement_employer_contribution) and _has_official_url(entry.retirement_urls, uni_key)),
        id=f"{uni_node_id}_Retirement_Exists",
        desc=f"{uni_name} employer retirement contribution value present and official URL(s) provided.",
        parent=uni_node,
        critical=True
    )

    ret_leaf = evaluator.add_leaf(
        id=f"{uni_node_id.split('_')[0]}_Retirement_Employer_Contribution_With_URL",
        desc="Provides employer retirement contribution rate and includes at least one supporting official HR/benefits URL.",
        parent=uni_node,
        critical=True
    )
    ret_sources = entry.retirement_urls if entry else []
    ret_val = entry.retirement_employer_contribution if entry else ""
    ret_claim = f"For {uni_name}, the EMPLOYER retirement contribution component is: {ret_val}. This refers to the employer's contribution (e.g., fixed %, match, or tiered structure), not merely employee elective deferrals."
    await evaluator.verify(
        claim=ret_claim,
        node=ret_leaf,
        sources=ret_sources,
        additional_instruction="Confirm that the statement is about the employer's contribution portion (e.g., match or fixed %). If the cited content only mentions employee deferrals with no employer contribution, mark incorrect. Use the official HR/benefits page.",
        extra_prerequisites=[ret_exist],
    )

    # ---------------- Employee Tuition ----------------
    emp_tu_exist = evaluator.add_custom_node(
        result=(entry is not None and _non_empty(entry.employee_tuition_benefit) and _has_official_url(entry.employee_tuition_urls, uni_key)),
        id=f"{uni_node_id}_Employee_Tuition_Exists",
        desc=f"{uni_name} employee tuition benefit value present and official URL(s) provided.",
        parent=uni_node,
        critical=True
    )

    emp_tu_leaf = evaluator.add_leaf(
        id=f"{uni_node_id.split('_')[0]}_Employee_Tuition_Benefit_With_URL",
        desc="Provides employee tuition benefit and includes at least one supporting official HR/benefits URL.",
        parent=uni_node,
        critical=True
    )
    emp_tu_sources = entry.employee_tuition_urls if entry else []
    emp_tu_val = entry.employee_tuition_benefit if entry else ""
    emp_tu_claim = f"For {uni_name}, the EMPLOYEE tuition benefit (credits/year or coverage) is: {emp_tu_val}. This applies to the employee, not dependents."
    await evaluator.verify(
        claim=emp_tu_claim,
        node=emp_tu_leaf,
        sources=emp_tu_sources,
        additional_instruction="Verify that the benefit applies to employees (not dependents) and that credits/year or coverage description matches the page.",
        extra_prerequisites=[emp_tu_exist],
    )

    # ---------------- Dependent Undergrad Tuition ----------------
    dep_tu_exist = evaluator.add_custom_node(
        result=(entry is not None and _non_empty(entry.dependent_undergrad_benefit) and _has_official_url(entry.dependent_undergrad_urls, uni_key)),
        id=f"{uni_node_id}_Dependent_Tuition_Exists",
        desc=f"{uni_name} dependent undergraduate tuition benefit value present and official URL(s) provided.",
        parent=uni_node,
        critical=True
    )

    dep_tu_leaf = evaluator.add_leaf(
        id=f"{uni_node_id.split('_')[0]}_Dependent_Undergrad_Tuition_Benefit_With_URL",
        desc="Provides dependent undergraduate tuition benefit and includes at least one supporting official HR/benefits URL.",
        parent=uni_node,
        critical=True
    )
    dep_tu_sources = entry.dependent_undergrad_urls if entry else []
    dep_tu_val = entry.dependent_undergrad_benefit if entry else ""
    dep_tu_claim = f"For {uni_name}, the DEPENDENT undergraduate tuition benefit is: {dep_tu_val}."
    await evaluator.verify(
        claim=dep_tu_claim,
        node=dep_tu_leaf,
        sources=dep_tu_sources,
        additional_instruction="Confirm that the benefit applies to dependents (not employees) and that coverage %, transferability, or eligibility rules match the page.",
        extra_prerequisites=[dep_tu_exist],
    )

    # ---------------- Wellness/Fitness Exclusivity ----------------
    well_exist = evaluator.add_custom_node(
        result=(entry is not None and _non_empty(entry.wellness_fitness_description) and _has_official_url(entry.wellness_urls, uni_key)),
        id=f"{uni_node_id}_Wellness_Exists",
        desc=f"{uni_name} wellness/fitness description present and official URL(s) provided.",
        parent=uni_node,
        critical=True
    )

    well_leaf = evaluator.add_leaf(
        id=f"{uni_node_id.split('_')[0]}_Wellness_Fitness_Exclusivity_With_URL",
        desc="States on-site wellness/fitness availability and exclusivity, with at least one supporting official HR/benefits URL.",
        parent=uni_node,
        critical=True
    )
    well_sources = entry.wellness_urls if entry else []
    available = entry.wellness_center_available if entry else None
    exclusive = entry.wellness_exclusive_to_employees if entry else None
    well_desc = entry.wellness_fitness_description if entry else ""
    avail_str = "available" if available is True else ("not available" if available is False else "unspecified")
    excl_str = "exclusive to employees and their families" if exclusive is True else ("not exclusive (general campus facility)" if exclusive is False else "exclusivity unspecified")
    well_claim = f"For {uni_name}, the on-site wellness/fitness center is {avail_str}. Access is {excl_str}. Description from the answer: {well_desc}"
    await evaluator.verify(
        claim=well_claim,
        node=well_leaf,
        sources=well_sources,
        additional_instruction="Verify the facility is physically on campus and whether access is exclusive to employees/families. If the page describes a general campus facility (e.g., student recreation center) without exclusive employee access, mark accordingly.",
        extra_prerequisites=[well_exist],
    )

    # ---------------- Currentness (2025–2026 or most recent) ----------------
    curr_exist = evaluator.add_custom_node(
        result=(entry is not None and _non_empty(entry.plan_year_or_effective_date) and _has_official_url(entry.currentness_urls, uni_key)),
        id=f"{uni_node_id}_Currentness_Exists",
        desc=f"{uni_name} plan year/effective date present and official URL(s) provided.",
        parent=uni_node,
        critical=True
    )

    curr_leaf = evaluator.add_leaf(
        id=f"{uni_node_id.split('_')[0]}_Currentness_2025_2026",
        desc="Uses 2025–2026 benefit structures when available; otherwise identifies the most recent plan year/effective date shown in cited sources.",
        parent=uni_node,
        critical=True
    )
    curr_sources = entry.currentness_urls if entry else []
    curr_val = entry.plan_year_or_effective_date if entry else ""
    curr_claim = f"For {uni_name}, the plan year/effective date identified in the answer is: '{curr_val}'. This should match what is explicitly shown on the cited official sources; if 2025–2026 documentation is available, the answer uses that year's structure; otherwise, the answer cites the most recent plan year/effective date indicated."
    await evaluator.verify(
        claim=curr_claim,
        node=curr_leaf,
        sources=curr_sources,
        additional_instruction="Check that the cited official page(s) show the same plan year/effective date string or equivalent wording. If a 2025–2026 plan document is present, ensure the answer aligns to that; else ensure the answer explicitly names the most recent date shown.",
        extra_prerequisites=[curr_exist],
    )


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
    model: str = "o4-mini"
) -> Dict[str, Any]:
    """
    Evaluate the employee benefits comparison across four universities.
    """
    # Initialize evaluator (root is always non-critical; create a critical child node for rubric root)
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

    # Create rubric root as a critical node under evaluator root
    rubric_root = evaluator.add_parallel(
        id="University_Benefits_Comparison",
        desc="Compare the specified employee benefits across the four specified universities, covering all required categories with verifiable, official HR/benefits sources.",
        parent=root,
        critical=True
    )

    # Extract structured info from the answer
    extraction = await evaluator.extract(
        prompt=prompt_extract_universities_benefits(),
        template_class=UniversitiesBenefitsExtraction,
        extraction_name="benefits_extraction"
    )

    # Build per-university verification subtrees
    await verify_university_benefits(
        evaluator=evaluator,
        parent=rubric_root,
        uni_key="nyu",
        uni_node_id="NYU_Benefits",
        uni_node_desc="NYU staff/administrative employee benefits with required categories and official HR/benefits .edu sources.",
        entry=extraction.nyu
    )

    await verify_university_benefits(
        evaluator=evaluator,
        parent=rubric_root,
        uni_key="notre_dame",
        uni_node_id="Notre_Dame_Benefits",
        uni_node_desc="University of Notre Dame staff/administrative employee benefits with required categories and official HR/benefits .edu sources.",
        entry=extraction.notre_dame
    )

    await verify_university_benefits(
        evaluator=evaluator,
        parent=rubric_root,
        uni_key="cwru",
        uni_node_id="Case_Western_Benefits",
        uni_node_desc="Case Western Reserve University staff/administrative employee benefits with required categories and official HR/benefits .edu sources.",
        entry=extraction.cwru
    )

    await verify_university_benefits(
        evaluator=evaluator,
        parent=rubric_root,
        uni_key="ferris_state",
        uni_node_id="Ferris_State_Benefits",
        uni_node_desc="Ferris State University staff/administrative employee benefits with required categories and official HR/benefits .edu sources.",
        entry=extraction.ferris_state
    )

    # Return structured summary
    return evaluator.get_summary()