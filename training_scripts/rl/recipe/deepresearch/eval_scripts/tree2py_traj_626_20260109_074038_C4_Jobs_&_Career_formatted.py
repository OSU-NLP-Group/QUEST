import asyncio
import logging
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys


# --------------------------------------------------------------------------- #
# Task constants                                                              #
# --------------------------------------------------------------------------- #
TASK_ID = "us_tech_benefits_2023"
TASK_DESCRIPTION = (
    "Identify a technology company headquartered in the United States that, as of December 31, 2023, offered all of the following employee benefits to its U.S.-based employees:\n\n"
    "1. An annual wellness reimbursement benefit of at least $500 per employee to cover fitness, mental health, or related wellness expenses\n"
    "2. At least 16 weeks of paid parental leave for at least one category of parents (primary caregiver, secondary caregiver, or both)\n"
    "3. A sabbatical benefit available to employees after 5 years of continuous service\n"
    "4. An annual education or professional development reimbursement budget of at least $5,000 per employee specifically for degree programs or professional certifications\n"
    "5. An Employee Stock Purchase Plan (ESPP) offering at least a 10% discount on company stock\n"
    "6. A 401(k) retirement plan with employer matching of at least 50% on employee contributions up to 6% of salary\n\n"
    "Provide the company name and reference URLs documenting each of these six benefits."
)

AS_OF_TARGET_DATE = "December 31, 2023"


# --------------------------------------------------------------------------- #
# Extraction models                                                           #
# --------------------------------------------------------------------------- #
class BenefitEvidence(BaseModel):
    details: Optional[str] = None  # free-form text quoted or summarized from answer (if present)
    urls: List[str] = Field(default_factory=list)  # reference URLs explicitly included in the answer


class CompanyBenefitsExtraction(BaseModel):
    company_name: Optional[str] = None
    company_industry_or_description: Optional[str] = None  # e.g., "cloud software company"
    headquarters_text: Optional[str] = None  # any HQ text mentioned in the answer (e.g., "San Francisco, CA, USA")
    as_of_statement: Optional[str] = None  # any temporal statement present in the answer (e.g., "as of Dec 31, 2023")

    wellness: Optional[BenefitEvidence] = None
    parental_leave: Optional[BenefitEvidence] = None
    sabbatical: Optional[BenefitEvidence] = None
    professional_development: Optional[BenefitEvidence] = None
    espp: Optional[BenefitEvidence] = None
    k401: Optional[BenefitEvidence] = None  # 401(k)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_company_benefits() -> str:
    return """
    Extract the information about the chosen company and the cited sources for each required benefit from the answer.

    Return a JSON object with the following fields:
    - company_name: The name of the company identified in the answer (single company).
    - company_industry_or_description: Short phrase describing the company's industry/domain as presented in the answer (if present).
    - headquarters_text: Any headquarters/location text mentioned in the answer (e.g., "San Francisco, CA, USA"). If not present, return null.
    - as_of_statement: Any explicit temporal statement indicating that the benefits are as of or before December 31, 2023 (e.g., "as of Dec 31, 2023"). If not present, return null.

    For each of the six required benefits, extract an object with:
    - details: The free-form benefit description as stated in the answer (if any).
    - urls: All URLs explicitly provided in the answer that document this benefit (benefit pages, policy pages, company benefits pages, etc.). Only include actual URLs present in the answer text. Do not invent or infer any URL.

    The six benefit objects and their expected keys:
    - wellness: Supporting the annual wellness reimbursement requirement
    - parental_leave: Supporting the paid parental leave requirement
    - sabbatical: Supporting the sabbatical-after-5-years requirement
    - professional_development: Supporting the >= $5,000/year education/professional development reimbursement for degrees/certifications
    - espp: Supporting the ESPP >= 10% discount requirement
    - k401: Supporting the 401(k) match >= 50% up to 6% of salary requirement

    IMPORTANT:
    - Only extract information that explicitly appears in the answer text.
    - For URLs, include only valid HTTP/HTTPS links actually present in the answer.
    - If any field is not mentioned, set it to null (or for urls, use an empty list).
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _safe_urls(ev: Optional[BenefitEvidence]) -> List[str]:
    return [] if (ev is None or ev.urls is None) else ev.urls


def _union_all_benefit_urls(extracted: CompanyBenefitsExtraction) -> List[str]:
    urls: List[str] = []
    urls.extend(_safe_urls(extracted.wellness))
    urls.extend(_safe_urls(extracted.parental_leave))
    urls.extend(_safe_urls(extracted.sabbatical))
    urls.extend(_safe_urls(extracted.professional_development))
    urls.extend(_safe_urls(extracted.espp))
    urls.extend(_safe_urls(extracted.k401))
    # Deduplicate while preserving order
    seen = set()
    unique_urls = []
    for u in urls:
        if u and u not in seen:
            seen.add(u)
            unique_urls.append(u)
    return unique_urls


# --------------------------------------------------------------------------- #
# Benefit verification builder                                                #
# --------------------------------------------------------------------------- #
async def add_benefit_check(
    evaluator: Evaluator,
    parent_node,
    *,
    benefit_node_id: str,
    benefit_node_desc: str,
    requirement_leaf_id: str,
    requirement_leaf_desc: str,
    requirement_claim: str,
    requirement_sources: List[str],
    requirement_additional_instruction: str,
    url_leaf_id: str,
    url_leaf_desc: str,
) -> None:
    """
    Create a benefit sub-tree with:
      - A critical parallel parent node (e.g., "Wellness_Reimbursement_Benefit")
      - A critical URL presence leaf (custom node)
      - A critical requirement verification leaf (verified against provided URLs)
    """
    benefit_node = evaluator.add_parallel(
        id=benefit_node_id,
        desc=benefit_node_desc,
        parent=parent_node,
        critical=True,
    )

    # URL presence check (critical)
    has_any_url = len(requirement_sources) > 0
    evaluator.add_custom_node(
        result=has_any_url,
        id=url_leaf_id,
        desc=url_leaf_desc,
        parent=benefit_node,
        critical=True,
    )

    # Requirement verification (critical)
    req_leaf = evaluator.add_leaf(
        id=requirement_leaf_id,
        desc=requirement_leaf_desc,
        parent=benefit_node,
        critical=True,
    )
    await evaluator.verify(
        claim=requirement_claim,
        node=req_leaf,
        sources=requirement_sources if requirement_sources else None,
        additional_instruction=requirement_additional_instruction,
    )


# --------------------------------------------------------------------------- #
# Main evaluation                                                             #
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
    Evaluate an answer for the U.S.-headquartered technology company with six specific benefits task.
    """
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
        default_model=model,
    )

    # Extract structured content from the answer
    extracted: CompanyBenefitsExtraction = await evaluator.extract(
        prompt=prompt_extract_company_benefits(),
        template_class=CompanyBenefitsExtraction,
        extraction_name="company_benefits_extraction",
    )

    # Add reference info
    evaluator.add_custom_info(
        {"as_of_target_date": AS_OF_TARGET_DATE},
        info_type="meta",
        info_name="as_of_requirement",
    )

    # Build the critical master node that aggregates all checks
    master = evaluator.add_parallel(
        id="Company_Meets_All_Requirements",
        desc="The response identifies a qualifying U.S.-headquartered technology company, confirms all required benefits as of Dec 31, 2023, and provides documentation URLs for each required benefit.",
        parent=root,
        critical=True,
    )

    company_name = (extracted.company_name or "").strip()
    hq_text = (extracted.headquarters_text or "").strip()

    # 1) Company name provided (critical)
    evaluator.add_custom_node(
        result=bool(company_name),
        id="Company_Name_Provided",
        desc="The company name is provided.",
        parent=master,
        critical=True,
    )

    # 2) Technology company requirement (critical)
    tech_leaf = evaluator.add_leaf(
        id="Technology_Company_Requirement",
        desc="The identified company is a technology company.",
        parent=master,
        critical=True,
    )
    tech_claim = (
        f"The company '{company_name}' is a technology company (e.g., builds/provides software, hardware, semiconductors, "
        f"internet services, or cloud technology)."
    )
    await evaluator.verify(
        claim=tech_claim,
        node=tech_leaf,
        additional_instruction=(
            "Judge based primarily on the answer's own description of the company's industry/domain. "
            "If the answer clearly indicates a technology domain (software, hardware, chips, cloud, internet, AI, etc.), consider it a technology company. "
            "If the answer does not indicate technology or indicates a non-technology primary industry, mark Incorrect."
        ),
    )

    # 3) U.S. headquarters requirement (critical)
    us_hq_leaf = evaluator.add_leaf(
        id="US_Headquarters_Requirement",
        desc="The company is headquartered in the United States.",
        parent=master,
        critical=True,
    )
    us_hq_claim = (
        f"The company '{company_name}' is headquartered in the United States."
    )
    await evaluator.verify(
        claim=us_hq_claim,
        node=us_hq_leaf,
        additional_instruction=(
            "Use only the information presented in the answer. If the answer explicitly states a U.S. city/state or 'United States' as headquarters, "
            "consider the requirement satisfied. If the answer does not indicate U.S. headquarters, mark Incorrect."
        ),
    )

    # 4) Temporal requirement (critical)
    temporal_leaf = evaluator.add_leaf(
        id="Temporal_Requirement",
        desc="The response supports that all specified benefits were offered as of December 31, 2023.",
        parent=master,
        critical=True,
    )
    temporal_claim = (
        f"As of {AS_OF_TARGET_DATE}, {company_name} offered all six specified benefits (wellness reimbursement >= $500, "
        f"paid parental leave >= 16 weeks for at least one parent category, sabbatical after 5 years, education/professional development "
        f"reimbursement >= $5,000 specifically for degrees or certifications, ESPP with >= 10% discount, and 401(k) matching >= 50% up to 6% of salary) "
        f"to its U.S.-based employees, as supported by the answer."
    )
    await evaluator.verify(
        claim=temporal_claim,
        node=temporal_leaf,
        additional_instruction=(
            "Check whether the answer itself clearly asserts or implies that these benefits were in effect as of December 31, 2023. "
            "Look for explicit time markers such as 'as of 2023' or similar statements. If the answer does not provide any time indication for the set of benefits, mark Incorrect."
        ),
    )

    # ------------------------------------------------------------------- #
    # Benefit-specific verification subtrees (all critical)               #
    # ------------------------------------------------------------------- #

    # Wellness
    wellness_urls = _safe_urls(extracted.wellness)
    await add_benefit_check(
        evaluator,
        master,
        benefit_node_id="Wellness_Reimbursement_Benefit",
        benefit_node_desc="Wellness reimbursement benefit meets the requirement and is documented.",
        requirement_leaf_id="Wellness_Reimbursement_Requirement",
        requirement_leaf_desc="Offers an annual wellness reimbursement of at least $500 per employee covering fitness/mental health/related wellness expenses.",
        requirement_claim=(
            f"{company_name} offers an annual wellness reimbursement of at least $500 per U.S.-based employee "
            f"to cover fitness, mental health, or related wellness expenses."
        ),
        requirement_sources=wellness_urls,
        requirement_additional_instruction=(
            "Accept equivalent terms such as 'wellness stipend', 'fitness reimbursement', or 'wellness benefit'. "
            "The minimum threshold is $500 per year per employee. If the amount is lower than $500 or not clearly annual, mark Incorrect."
        ),
        url_leaf_id="Wellness_Reimbursement_URL",
        url_leaf_desc="Provides at least one reference URL documenting the wellness reimbursement benefit.",
    )

    # Parental leave
    parental_urls = _safe_urls(extracted.parental_leave)
    await add_benefit_check(
        evaluator,
        master,
        benefit_node_id="Parental_Leave_Benefit",
        benefit_node_desc="Paid parental leave benefit meets the requirement and is documented.",
        requirement_leaf_id="Parental_Leave_Requirement",
        requirement_leaf_desc="Offers at least 16 weeks of paid parental leave for at least one category of parents (primary caregiver, secondary caregiver, or both).",
        requirement_claim=(
            f"{company_name} offers at least 16 weeks of paid parental leave for at least one category of parents "
            f"(such as primary caregiver or birthing parent) for U.S.-based employees."
        ),
        requirement_sources=parental_urls,
        requirement_additional_instruction=(
            "Allow common variants like 'primary caregiver 16 weeks' or 'birthing parent 16 weeks'. "
            "If only 12 weeks or less are offered to all categories, mark Incorrect. "
            "If 'up to 16 weeks' is stated, ensure at least one category receives a full 16 weeks of paid leave."
        ),
        url_leaf_id="Parental_Leave_URL",
        url_leaf_desc="Provides at least one reference URL documenting the paid parental leave benefit.",
    )

    # Sabbatical
    sabbatical_urls = _safe_urls(extracted.sabbatical)
    await add_benefit_check(
        evaluator,
        master,
        benefit_node_id="Sabbatical_Benefit",
        benefit_node_desc="Sabbatical benefit meets the requirement and is documented.",
        requirement_leaf_id="Sabbatical_Benefit_Requirement",
        requirement_leaf_desc="Offers a sabbatical benefit available after 5 years of continuous service.",
        requirement_claim=(
            f"{company_name} offers a sabbatical benefit that is available to U.S.-based employees after 5 years of continuous service."
        ),
        requirement_sources=sabbatical_urls,
        requirement_additional_instruction=(
            "Accept phrasings like 'every 5 years', 'after five years of service', or equivalent. "
            "The key criterion is eligibility beginning at 5 years of continuous service (the duration of the sabbatical itself may vary)."
        ),
        url_leaf_id="Sabbatical_Benefit_URL",
        url_leaf_desc="Provides at least one reference URL documenting the sabbatical benefit.",
    )

    # Professional development (education reimbursement for degrees/certifications)
    profdev_urls = _safe_urls(extracted.professional_development)
    await add_benefit_check(
        evaluator,
        master,
        benefit_node_id="Professional_Development_Benefit",
        benefit_node_desc="Education/professional development reimbursement meets the requirement and is documented.",
        requirement_leaf_id="Professional_Development_Budget_Requirement",
        requirement_leaf_desc="Provides an annual education/professional development reimbursement budget of at least $5,000 per employee specifically for degree programs or professional certifications.",
        requirement_claim=(
            f"{company_name} provides an annual education/professional development reimbursement budget of at least $5,000 per U.S.-based employee "
            f"specifically for degree programs or professional certifications."
        ),
        requirement_sources=profdev_urls,
        requirement_additional_instruction=(
            "Tuition assistance or education reimbursement that explicitly covers accredited degree programs or professional certifications qualifies. "
            "An annual cap of $5,000 or higher (e.g., $5,250) satisfies the threshold. "
            "If the program only covers conferences or trainings and does not cover degrees/certifications, mark Incorrect."
        ),
        url_leaf_id="Professional_Development_Budget_URL",
        url_leaf_desc="Provides at least one reference URL documenting the education/professional development reimbursement benefit.",
    )

    # ESPP
    espp_urls = _safe_urls(extracted.espp)
    await add_benefit_check(
        evaluator,
        master,
        benefit_node_id="ESPP_Benefit",
        benefit_node_desc="ESPP benefit meets the requirement and is documented.",
        requirement_leaf_id="ESPP_Discount_Requirement",
        requirement_leaf_desc="Offers an Employee Stock Purchase Plan (ESPP) with at least a 10% discount on company stock.",
        requirement_claim=(
            f"{company_name} offers an Employee Stock Purchase Plan (ESPP) with at least a 10% discount on company stock to U.S.-based employees."
        ),
        requirement_sources=espp_urls,
        requirement_additional_instruction=(
            "Accept '15% discount' or statements like 'stock purchased at 85% of fair market value' as satisfying the >= 10% discount requirement. "
            "The presence of a lookback feature is not required. "
            "If there is no ESPP or the discount is less than 10%, mark Incorrect."
        ),
        url_leaf_id="ESPP_Discount_URL",
        url_leaf_desc="Provides at least one reference URL documenting the ESPP discount benefit.",
    )

    # 401(k)
    k401_urls = _safe_urls(extracted.k401)
    await add_benefit_check(
        evaluator,
        master,
        benefit_node_id="401k_Benefit",
        benefit_node_desc="401(k) matching benefit meets the requirement and is documented.",
        requirement_leaf_id="401k_Matching_Requirement",
        requirement_leaf_desc="Provides a 401(k) plan with employer matching of at least 50% on employee contributions up to 6% of salary.",
        requirement_claim=(
            f"{company_name} provides a 401(k) retirement plan with employer matching of at least 50% on employee contributions up to 6% of salary for U.S.-based employees."
        ),
        requirement_sources=k401_urls,
        requirement_additional_instruction=(
            "Accept any match that is equal to or better than '50% up to 6%'. For example, '100% up to 6%' or '50% up to 8%' both qualify. "
            "'100% up to 4%' does NOT qualify because it does not cover up to 6%. "
            "If the plan is not a 401(k) (e.g., only a non-matching plan) or the match is below the specified threshold, mark Incorrect."
        ),
        url_leaf_id="401k_Matching_URL",
        url_leaf_desc="Provides at least one reference URL documenting the 401(k) matching benefit.",
    )

    # Return evaluation summary
    return evaluator.get_summary()