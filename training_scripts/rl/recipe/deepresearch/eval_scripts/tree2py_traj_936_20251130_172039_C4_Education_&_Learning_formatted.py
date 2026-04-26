import asyncio
import logging
from typing import Optional, List, Dict, Any
from urllib.parse import urlparse

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy

# --------------------------------------------------------------------------- #
# Task constants                                                              #
# --------------------------------------------------------------------------- #
TASK_ID = "hs_eligibility_requirements_by_state"
TASK_DESCRIPTION = (
    "You are preparing a comprehensive guide for high school student-athletes who may be relocating to different U.S. "
    "states and need to understand varying academic eligibility requirements for interscholastic athletic participation.\n\n"
    "Research and provide the following specific academic eligibility requirements as established by state high school athletic associations:\n\n"
    "1. Ohio (OHSAA): Minimum number of credit courses (or equivalent) a student must pass in the immediately preceding grading period; and whether OHSAA has a state-level minimum GPA requirement.\n"
    "2. Georgia (GHSA): Minimum number of Carnegie units that a student must pass in the previous semester (starting from the second semester of 9th grade).\n"
    "3. Louisiana (LHSAA) - First Semester: Minimum number of units earned from the previous school year required for first-semester eligibility.\n"
    "4. Louisiana (LHSAA) - Second Semester: Minimum number of half units (0.5 credit units) passed from the first semester required for second-semester eligibility.\n"
    "5. Louisiana (LHSAA) - GPA Requirement: Overall GPA students must maintain for the entire school year to remain eligible.\n"
    "6. Illinois (IHSA): Minimum number of credit hours of high school work per week a student must be passing to maintain eligibility.\n\n"
    "For each requirement, provide the specific numerical value or policy statement along with a reference URL to the official state athletic association source that confirms this requirement."
)

# --------------------------------------------------------------------------- #
# Official domains per association                                            #
# --------------------------------------------------------------------------- #
ASSOCIATION_DOMAINS = {
    "OHSAA": ["ohsaa.org"],
    "GHSA": ["ghsa.net"],
    "LHSAA": ["lhsaa.org"],
    "IHSA": ["ihsa.org"],
}

# --------------------------------------------------------------------------- #
# Descriptions from rubric                                                    #
# --------------------------------------------------------------------------- #
GROUP_DESCRIPTIONS = {
    "ohio_ohsaa": "Ohio (OHSAA) requirements are correctly reported and each is supported by an official OHSAA source URL.",
    "georgia_ghsa": "Georgia (GHSA) requirement is correctly reported and supported by an official GHSA source URL.",
    "louisiana_lhsaa": "Louisiana (LHSAA) requirements are correctly reported and each is supported by an official LHSAA source URL.",
    "illinois_ihsa": "Illinois (IHSA) requirement is correctly reported and supported by an official IHSA source URL.",
}

REQ_DESCRIPTIONS = {
    "ohio_credit_requirement_with_citation": "States the OHSAA minimum number of credit courses (or equivalent) that must be passed in the immediately preceding grading period, and provides an official OHSAA URL that supports it.",
    "ohio_gpa_policy_with_citation": "States whether OHSAA has a state-level minimum GPA requirement (policy statement), and provides an official OHSAA URL that supports it.",
    "georgia_carnegie_units_with_citation": "States the GHSA minimum Carnegie units that must be passed in the previous semester (starting from second semester of 9th grade), and provides an official GHSA URL that supports it.",
    "louisiana_first_semester_units_with_citation": "States the LHSAA minimum units earned from the previous school year required for first-semester eligibility, and provides an official LHSAA URL that supports it.",
    "louisiana_second_semester_half_units_with_citation": "States the LHSAA minimum half-units (0.5 credit units) passed from the first semester required for second-semester eligibility, and provides an official LHSAA URL that supports it.",
    "louisiana_overall_gpa_policy_with_citation": "States the LHSAA overall GPA/grade-average requirement for the entire school year, and provides an official LHSAA URL that supports it.",
    "illinois_weekly_credit_hours_with_citation": "States the IHSA minimum number of credit hours of high school work per week the student must be passing to maintain eligibility, and provides an official IHSA URL that supports it.",
}

# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class CitedField(BaseModel):
    value: Optional[str] = None
    urls: List[str] = Field(default_factory=list)


class RequirementsExtraction(BaseModel):
    ohio_credit_requirement: Optional[CitedField] = None
    ohio_gpa_policy: Optional[CitedField] = None
    georgia_carnegie_units: Optional[CitedField] = None
    louisiana_first_semester_units: Optional[CitedField] = None
    louisiana_second_semester_half_units: Optional[CitedField] = None
    louisiana_overall_gpa: Optional[CitedField] = None
    illinois_weekly_credit_hours: Optional[CitedField] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_requirements() -> str:
    return """
Extract the specific academic eligibility requirements and the corresponding official source URLs as stated in the answer. For each of the following items, return an object with:
- value: the exact numerical threshold or policy statement as written in the answer (do not infer or paraphrase).
- urls: a list of URLs explicitly provided in the answer that support the item. Extract URLs exactly as presented (plain or markdown). If none are provided, return an empty list.

Items to extract (all are independent):
1) ohio_credit_requirement: OHSAA minimum number of credit courses (or equivalent) that must be passed in the immediately preceding grading period to maintain eligibility.
2) ohio_gpa_policy: OHSAA state-level minimum GPA policy (e.g., whether there is a statewide minimum GPA requirement).
3) georgia_carnegie_units: GHSA minimum number of Carnegie units that must be passed in the previous semester (starting from second semester of 9th grade).
4) louisiana_first_semester_units: LHSAA minimum units earned from the previous school year required for first-semester eligibility.
5) louisiana_second_semester_half_units: LHSAA minimum number of half-units (0.5 credit units) that must be passed in the first semester for second-semester eligibility.
6) louisiana_overall_gpa: LHSAA overall GPA/grade-average requirement for the entire school year.
7) illinois_weekly_credit_hours: IHSA minimum number of credit hours of high school work per week that must be passing.

Rules:
- Extract only what is explicitly stated in the answer.
- Do not invent values or URLs. If missing, leave value as null and urls as [].
- Preserve the original phrasing for 'value' as it appears in the answer (including units like 'Carnegie units', 'credit hours', 'half-units', etc.).
"""


# --------------------------------------------------------------------------- #
# Utility functions                                                           #
# --------------------------------------------------------------------------- #
def _extract_netloc(u: str) -> str:
    try:
        netloc = urlparse(u).netloc.lower()
        if netloc.startswith("www."):
            netloc = netloc[4:]
        return netloc
    except Exception:
        return ""


def has_official_url(urls: List[str], allowed_domains: List[str]) -> bool:
    if not urls:
        return False
    for u in urls:
        netloc = _extract_netloc(u)
        for dom in allowed_domains:
            if netloc.endswith(dom):
                return True
    return False


def safe_list(urls: Optional[List[str]]) -> List[str]:
    return [u for u in (urls or []) if isinstance(u, str) and u.strip()]


def value_present(v: Optional[str]) -> bool:
    return bool(v and isinstance(v, str) and v.strip())


def compose_additional_instruction(association: str, nuance: str = "") -> str:
    base = (
        f"Verify that the webpage explicitly supports the stated eligibility requirement for {association}. "
        f"Focus on the exact numeric threshold or policy statement as provided in the claim. "
        f"Allow equivalent wording (e.g., 'courses' vs 'credit courses', '0.5 units' vs 'half-units'), "
        f"but the meaning and numeric threshold must match."
    )
    if nuance:
        base += f" Also consider this nuance: {nuance}"
    return base


# --------------------------------------------------------------------------- #
# Requirement builder                                                         #
# --------------------------------------------------------------------------- #
async def build_requirement_with_citation(
    evaluator: Evaluator,
    parent_node,
    req_id: str,
    req_desc: str,
    association: str,
    allowed_domains: List[str],
    value_text: Optional[str],
    urls: List[str],
    claim_prefix: str,
    nuance_instruction: str = "",
) -> None:
    """
    Build a critical requirement node with:
    - value presence check
    - source URL presence check
    - official-domain check
    - claim supported by provided URL(s)
    """
    # Container node (critical)
    req_node = evaluator.add_parallel(
        id=req_id,
        desc=req_desc,
        parent=parent_node,
        critical=True,
    )

    # 1) Value presence (critical)
    evaluator.add_custom_node(
        result=value_present(value_text),
        id=f"{req_id}_value_present",
        desc="Answer provides a specific value or policy statement for this requirement.",
        parent=req_node,
        critical=True,
    )

    # 2) URL presence (critical)
    evaluator.add_custom_node(
        result=len(urls) > 0,
        id=f"{req_id}_urls_present",
        desc="At least one source URL is provided for this requirement.",
        parent=req_node,
        critical=True,
    )

    # 3) Official domain check (critical)
    evaluator.add_custom_node(
        result=has_official_url(urls, allowed_domains),
        id=f"{req_id}_official_source",
        desc=f"At least one provided URL is an official {association} website (domains include: {', '.join(allowed_domains)}).",
        parent=req_node,
        critical=True,
    )

    # 4) Claim supported by the cited source(s) (critical)
    support_leaf = evaluator.add_leaf(
        id=f"{req_id}_claim_supported",
        desc="The requirement statement is supported by the cited source(s).",
        parent=req_node,
        critical=True,
    )

    claim_text = f"{claim_prefix}: '{value_text or ''}'."
    add_ins = compose_additional_instruction(association, nuance_instruction)

    await evaluator.verify(
        claim=claim_text,
        node=support_leaf,
        sources=urls,
        additional_instruction=add_ins,
    )


# --------------------------------------------------------------------------- #
# Group verifiers                                                             #
# --------------------------------------------------------------------------- #
async def verify_ohio_ohsaa(evaluator: Evaluator, root_node, data: RequirementsExtraction) -> None:
    group = evaluator.add_parallel(
        id="ohio_ohsaa",
        desc=GROUP_DESCRIPTIONS["ohio_ohsaa"],
        parent=root_node,
        critical=False,
    )

    # Ohio: Credit requirement (immediately preceding grading period)
    ohio_credit = data.ohio_credit_requirement or CitedField()
    await build_requirement_with_citation(
        evaluator=evaluator,
        parent_node=group,
        req_id="ohio_credit_requirement_with_citation",
        req_desc=REQ_DESCRIPTIONS["ohio_credit_requirement_with_citation"],
        association="OHSAA",
        allowed_domains=ASSOCIATION_DOMAINS["OHSAA"],
        value_text=ohio_credit.value,
        urls=safe_list(ohio_credit.urls),
        claim_prefix="OHSAA minimum courses passed in the immediately preceding grading period",
        nuance_instruction="This refers to the grading period immediately before the current one for athletic eligibility. Accept equivalent phrasing such as 'five (5) one-credit courses or the equivalent' if that matches what the page states.",
    )

    # Ohio: GPA policy (state-level)
    ohio_gpa = data.ohio_gpa_policy or CitedField()
    await build_requirement_with_citation(
        evaluator=evaluator,
        parent_node=group,
        req_id="ohio_gpa_policy_with_citation",
        req_desc=REQ_DESCRIPTIONS["ohio_gpa_policy_with_citation"],
        association="OHSAA",
        allowed_domains=ASSOCIATION_DOMAINS["OHSAA"],
        value_text=ohio_gpa.value,
        urls=safe_list(ohio_gpa.urls),
        claim_prefix="OHSAA statewide minimum GPA policy for athletic eligibility",
        nuance_instruction="If OHSAA does not set a statewide minimum GPA, the page should explicitly indicate that local boards may adopt their own GPA policies.",
    )


async def verify_georgia_ghsa(evaluator: Evaluator, root_node, data: RequirementsExtraction) -> None:
    group = evaluator.add_parallel(
        id="georgia_ghsa",
        desc=GROUP_DESCRIPTIONS["georgia_ghsa"],
        parent=root_node,
        critical=False,
    )

    ghsa_units = data.georgia_carnegie_units or CitedField()
    await build_requirement_with_citation(
        evaluator=evaluator,
        parent_node=group,
        req_id="georgia_carnegie_units_with_citation",
        req_desc=REQ_DESCRIPTIONS["georgia_carnegie_units_with_citation"],
        association="GHSA",
        allowed_domains=ASSOCIATION_DOMAINS["GHSA"],
        value_text=ghsa_units.value,
        urls=safe_list(ghsa_units.urls),
        claim_prefix="GHSA minimum Carnegie units that must be passed in the previous semester (from second semester of 9th grade onward)",
        nuance_instruction="Ensure the source specifies this rule applies beginning with the second semester of the 9th grade.",
    )


async def verify_louisiana_lhsaa(evaluator: Evaluator, root_node, data: RequirementsExtraction) -> None:
    group = evaluator.add_parallel(
        id="louisiana_lhsaa",
        desc=GROUP_DESCRIPTIONS["louisiana_lhsaa"],
        parent=root_node,
        critical=False,
    )

    # First semester eligibility: units earned from the previous year
    lhsaa_first = data.louisiana_first_semester_units or CitedField()
    await build_requirement_with_citation(
        evaluator=evaluator,
        parent_node=group,
        req_id="louisiana_first_semester_units_with_citation",
        req_desc=REQ_DESCRIPTIONS["louisiana_first_semester_units_with_citation"],
        association="LHSAA",
        allowed_domains=ASSOCIATION_DOMAINS["LHSAA"],
        value_text=lhsaa_first.value,
        urls=safe_list(lhsaa_first.urls),
        claim_prefix="LHSAA minimum units earned from the previous school year required for first-semester eligibility",
        nuance_instruction="This applies to eligibility at the start of the academic year; confirm the minimum total units earned from the prior year.",
    )

    # Second semester eligibility: half-units passed in first semester
    lhsaa_second = data.louisiana_second_semester_half_units or CitedField()
    await build_requirement_with_citation(
        evaluator=evaluator,
        parent_node=group,
        req_id="louisiana_second_semester_half_units_with_citation",
        req_desc=REQ_DESCRIPTIONS["louisiana_second_semester_half_units_with_citation"],
        association="LHSAA",
        allowed_domains=ASSOCIATION_DOMAINS["LHSAA"],
        value_text=lhsaa_second.value,
        urls=safe_list(lhsaa_second.urls),
        claim_prefix="LHSAA minimum number of half-units (0.5 credit units) that must be passed in the first semester for second-semester eligibility",
        nuance_instruction="Confirm that the requirement is expressed in half-units (0.5 credits) and pertains specifically to the first-semester performance for second-semester eligibility.",
    )

    # Overall GPA requirement for entire school year
    lhsaa_gpa = data.louisiana_overall_gpa or CitedField()
    await build_requirement_with_citation(
        evaluator=evaluator,
        parent_node=group,
        req_id="louisiana_overall_gpa_policy_with_citation",
        req_desc=REQ_DESCRIPTIONS["louisiana_overall_gpa_policy_with_citation"],
        association="LHSAA",
        allowed_domains=ASSOCIATION_DOMAINS["LHSAA"],
        value_text=lhsaa_gpa.value,
        urls=safe_list(lhsaa_gpa.urls),
        claim_prefix="LHSAA overall GPA/grade-average requirement for the entire school year",
        nuance_instruction="Ensure the source articulates an overall GPA or average applicable to the full school year (not just a single term).",
    )


async def verify_illinois_ihsa(evaluator: Evaluator, root_node, data: RequirementsExtraction) -> None:
    group = evaluator.add_parallel(
        id="illinois_ihsa",
        desc=GROUP_DESCRIPTIONS["illinois_ihsa"],
        parent=root_node,
        critical=False,
    )

    ihsa_hours = data.illinois_weekly_credit_hours or CitedField()
    await build_requirement_with_citation(
        evaluator=evaluator,
        parent_node=group,
        req_id="illinois_weekly_credit_hours_with_citation",
        req_desc=REQ_DESCRIPTIONS["illinois_weekly_credit_hours_with_citation"],
        association="IHSA",
        allowed_domains=ASSOCIATION_DOMAINS["IHSA"],
        value_text=ihsa_hours.value,
        urls=safe_list(ihsa_hours.urls),
        claim_prefix="IHSA minimum number of credit hours of high school work per week that a student must be passing",
        nuance_instruction="Confirm that the requirement is measured per week and pertains to 'credit hours of high school work' for eligibility.",
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
    Evaluate an answer for the state athletic associations' academic eligibility requirements task.
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

    # Record helpful reference info
    evaluator.add_custom_info(
        info={"association_domains": ASSOCIATION_DOMAINS},
        info_type="reference",
        info_name="official_association_domains"
    )

    # Extraction
    extracted = await evaluator.extract(
        prompt=prompt_extract_requirements(),
        template_class=RequirementsExtraction,
        extraction_name="requirements_extraction",
    )

    # Build verification tree per group
    await verify_ohio_ohsaa(evaluator, root, extracted)
    await verify_georgia_ghsa(evaluator, root, extracted)
    await verify_louisiana_lhsaa(evaluator, root, extracted)
    await verify_illinois_ihsa(evaluator, root, extracted)

    # Return summary
    return evaluator.get_summary()