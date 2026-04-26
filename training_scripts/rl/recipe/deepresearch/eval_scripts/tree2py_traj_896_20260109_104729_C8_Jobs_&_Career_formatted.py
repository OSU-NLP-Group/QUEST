import asyncio
import logging
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.llm_client.base_client import LLMClient


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "tech_company_benefits_2024_2025"
TASK_DESCRIPTION = (
    "Among the major U.S. technology companies (Google, Meta, Microsoft, Amazon, Apple, and Salesforce), "
    "identify one company that offers a comprehensive employee benefits package meeting or exceeding ALL of the following "
    "minimum requirements as of 2024-2025. For each benefit category, provide the specific details and include reference URLs "
    "from official company benefits pages or verified third-party benefits platforms (such as Levels.fyi) to document your findings.\n\n"
    "Required Minimum Benefits:\n"
    "1. Company Eligibility: Must be one of the six specified companies (Google, Meta, Microsoft, Amazon, Apple, Salesforce)\n"
    "2. 401(k) Retirement Plan: Employer matching of at least 50% on employee contributions\n"
    "3. Maternity Leave: At least 16 weeks of paid leave for birthing parents\n"
    "4. Paternity Leave: At least 12 weeks of paid leave for non-birthing parents\n"
    "5. Health Insurance: Employee premium is $0 or minimal (fully covered or highly subsidized)\n"
    "6. Paid Time Off: At least 20 days of annual vacation/personal PTO, or unlimited PTO\n"
    "7. Tuition Reimbursement: Educational assistance or tuition reimbursement program\n"
    "8. Adoption Assistance: At least $10,000 per child in adoption support\n"
    "9. Wellness Reimbursement: At least $1,000 per year for gym membership or wellness expenses\n"
    "10. Meal Benefits: On-site or subsidized meal services (breakfast, lunch, or dinner)\n"
    "11. HSA Contribution: Employer contributes at least $1,000 per year to Health Savings Account\n"
    "12. Student Loan Support: Loan repayment matching, assistance, or refinancing support program\n"
    "13. Life Insurance: Coverage of at least 2x (twice) annual employee salary\n"
    "14. Phone Reimbursement: At least $50 per month for phone bill expenses\n"
    "15. Remote Work: Flexible remote work policy allowing at least some work-from-home options\n"
    "16. Fertility Benefits: Fertility treatment assistance or family planning benefits\n\n"
    "Your response must include:\n"
    "- The name of the company that meets all requirements\n"
    "- For each of the 16 benefit categories, provide: (a) the specific benefit details that meet or exceed the minimum requirement, "
    "and (b) a reference URL documenting that benefit"
)

ALLOWED_COMPANIES = ["Google", "Meta", "Microsoft", "Amazon", "Apple", "Salesforce"]

# Domain hints for acceptable official sources by company.
ACCEPTABLE_DOMAIN_HINTS: Dict[str, List[str]] = {
    "Google": ["google.com", "careers.google.com", "alphabet.com"],
    "Meta": ["meta.com", "about.meta.com", "facebook.com", "fb.com"],
    "Microsoft": ["microsoft.com", "careers.microsoft.com"],
    "Amazon": ["amazon.jobs", "aboutamazon.com", "amazon.com"],
    "Apple": ["apple.com"],
    "Salesforce": ["salesforce.com", "salesforcebenefits.com", "benefits.salesforce.com"],
}
THIRD_PARTY_OK = ["levels.fyi"]


# --------------------------------------------------------------------------- #
# Data models                                                                 #
# --------------------------------------------------------------------------- #
class BenefitEntry(BaseModel):
    details: Optional[str] = None
    urls: List[str] = Field(default_factory=list)


class BenefitsReport(BaseModel):
    # Company selection and timeframe framing
    selected_company: Optional[str] = None
    also_selected_companies: List[str] = Field(
        default_factory=list,
        description="Any other companies the answer explicitly claims also meet all requirements; should be empty if exactly one is chosen."
    )
    mentioned_companies: List[str] = Field(
        default_factory=list,
        description="Any of the six allowed companies that are mentioned anywhere in the answer text."
    )
    timeframe_statement: Optional[str] = Field(
        default=None,
        description="The exact phrasing from the answer indicating benefits are as of 2024–2025 or current in that timeframe."
    )

    # Benefit categories
    retirement_401k: Optional[BenefitEntry] = None
    maternity_leave: Optional[BenefitEntry] = None
    paternity_leave: Optional[BenefitEntry] = None
    health_insurance: Optional[BenefitEntry] = None
    pto: Optional[BenefitEntry] = None
    tuition_reimbursement: Optional[BenefitEntry] = None
    adoption_assistance: Optional[BenefitEntry] = None
    wellness_reimbursement: Optional[BenefitEntry] = None
    meal_benefits: Optional[BenefitEntry] = None
    hsa_contribution: Optional[BenefitEntry] = None
    student_loan_support: Optional[BenefitEntry] = None
    life_insurance: Optional[BenefitEntry] = None
    phone_reimbursement: Optional[BenefitEntry] = None
    remote_work: Optional[BenefitEntry] = None
    fertility_benefits: Optional[BenefitEntry] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_benefits() -> str:
    return """
Extract structured information from the answer as follows. Your extraction must reflect only what is explicitly stated in the answer text.

A) Company selection and timeframe framing:
- selected_company: The single company the answer claims meets ALL requirements (Google, Meta, Microsoft, Amazon, Apple, or Salesforce). If multiple are presented as fully meeting all requirements, choose the main one if explicitly designated; otherwise, leave null.
- also_selected_companies: If the answer explicitly claims that more than one of the six companies meets all requirements, list the others here (do not include the selected_company again).
- mentioned_companies: List any of these six companies mentioned anywhere in the answer (Google, Meta, Microsoft, Amazon, Apple, Salesforce).
- timeframe_statement: Copy the exact phrase that indicates the benefits are current as of 2024–2025 (e.g., “as of 2024”, “current in 2025”, “2024–2025 benefits”). If absent, set to null.

B) For the SELECTED company only, extract for each benefit category:
For each item, set:
- details: The specific detail string as given in the answer (e.g., “50% match up to 6%”, “16 weeks paid maternity leave”, “$1,500 annual wellness stipend”). If not stated, set to null.
- urls: All URLs the answer cites for that category (official company benefits pages or verified third-party like levels.fyi). If none, return an empty array.

Categories (use these exact keys):
- retirement_401k
- maternity_leave
- paternity_leave
- health_insurance
- pto
- tuition_reimbursement
- adoption_assistance
- wellness_reimbursement
- meal_benefits
- hsa_contribution
- student_loan_support
- life_insurance
- phone_reimbursement
- remote_work
- fertility_benefits

Return JSON with all fields exactly as specified by the schema. Do not invent URLs. If a URL is missing a protocol, prepend http:// as per rules.
"""


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _get_entry_urls(entry: Optional[BenefitEntry]) -> List[str]:
    if not entry or not entry.urls:
        return []
    # Deduplicate while preserving order
    seen = set()
    out: List[str] = []
    for u in entry.urls:
        if not u:
            continue
        if u not in seen:
            seen.add(u)
            out.append(u)
    return out


def _collect_all_urls(report: BenefitsReport) -> List[str]:
    fields = [
        report.retirement_401k,
        report.maternity_leave,
        report.paternity_leave,
        report.health_insurance,
        report.pto,
        report.tuition_reimbursement,
        report.adoption_assistance,
        report.wellness_reimbursement,
        report.meal_benefits,
        report.hsa_contribution,
        report.student_loan_support,
        report.life_insurance,
        report.phone_reimbursement,
        report.remote_work,
        report.fertility_benefits,
    ]
    all_urls: List[str] = []
    for e in fields:
        all_urls.extend(_get_entry_urls(e))
    # Deduplicate
    seen = set()
    uniq: List[str] = []
    for u in all_urls:
        if u not in seen:
            seen.add(u)
            uniq.append(u)
    return uniq


def _build_acceptable_source_instruction(company: Optional[str], category_label: str) -> str:
    # Compose guidance for acceptable sources
    company_name = (company or "the selected company").strip()
    hints = ACCEPTABLE_DOMAIN_HINTS.get(company or "", [])
    hint_str = ", ".join(hints) if hints else "the company's official domain (e.g., careers or benefits pages)"
    third_party = ", ".join(THIRD_PARTY_OK)

    return (
        f"Verify the specific {category_label} details against the provided URLs. Consider the claim 'Supported' only if at least one URL is acceptable AND directly supports the stated detail.\n"
        f"Acceptable sources include:\n"
        f"- Official {company_name} domains (such as: {hint_str}).\n"
        f"- Reputable benefits platforms (e.g., {third_party}).\n"
        f"Do NOT accept anonymous blogs, generic forums, or unrelated aggregator sites.\n"
        f"If multiple URLs are provided, a single acceptable URL that clearly supports the detail is sufficient.\n"
        f"Give extra weight to pages that are current in 2024–2025 or explicitly labeled as 2024/2025 benefits."
    )


def _build_recency_instruction() -> str:
    return (
        "Judge this ONLY by the page itself. Consider 'Supported' if the page explicitly shows a last updated date in 2024 or 2025, "
        "mentions 2024/2025 in the benefits context, or is clearly the current/live benefits page for that timeframe. "
        "If none of the provided URLs are from 2024–2025 or clearly current in that timeframe, return 'Incorrect'."
    )


# --------------------------------------------------------------------------- #
# Verification builders                                                       #
# --------------------------------------------------------------------------- #
async def build_company_selection_checks(evaluator: Evaluator, parent_node, report: BenefitsReport) -> None:
    # Company Selection (critical)
    node = evaluator.add_parallel(
        id="company_selection",
        desc="Check that the response names exactly one eligible company from the provided list.",
        parent=parent_node,
        critical=True
    )

    # Exactly One Company Named (critical leaf)
    exactly_one = evaluator.add_custom_node(
        result=(report.selected_company is not None and isinstance(report.selected_company, str)
                and report.selected_company.strip() != "" and len(report.also_selected_companies) == 0),
        id="exactly_one_company_named",
        desc="Response identifies exactly one company (not multiple) as the selected company.",
        parent=node,
        critical=True
    )

    # Company In Allowed Set (critical leaf; LLM check allows minor variants)
    in_allowed = evaluator.add_leaf(
        id="company_in_allowed_set",
        desc="Selected company is one of: Google, Meta, Microsoft, Amazon, Apple, Salesforce.",
        parent=node,
        critical=True
    )
    selected = report.selected_company or ""
    claim = (
        f"The selected company '{selected}' refers to one of the allowed companies: "
        f"{', '.join(ALLOWED_COMPANIES)}. Treat close variants or parent-company naming as acceptable (e.g., 'Alphabet' for Google)."
    )
    await evaluator.verify(
        claim=claim,
        node=in_allowed,
        additional_instruction="Allow reasonable synonyms, rebrandings, or parent-company names that unambiguously refer to the same company."
    )


async def build_recency_checks(evaluator: Evaluator, parent_node, report: BenefitsReport) -> None:
    node = evaluator.add_parallel(
        id="recency_2024_2025",
        desc="Benefit claims are presented as current for 2024–2025 (as requested by the question).",
        parent=parent_node,
        critical=True
    )

    # Claims Framed As 2024–2025 (critical leaf; verify against the answer text)
    framed_leaf = evaluator.add_leaf(
        id="claims_framed_as_2024_2025",
        desc="Response explicitly indicates benefits are as of 2024–2025 (or equivalent language indicating current policy in that timeframe).",
        parent=node,
        critical=True
    )
    timeframe_phrase = report.timeframe_statement or ""
    claim = (
        "The answer explicitly indicates that the benefits are current as of 2024–2025 (e.g., phrases like "
        f"'as of 2024', 'current in 2025', '2024–2025 benefits'). "
        f"Extracted indicative phrase (if any): '{timeframe_phrase}'."
    )
    await evaluator.verify(
        claim=claim,
        node=framed_leaf,
        additional_instruction="Search the full answer for any explicit 2024 or 2025 timeframe language. Minor paraphrases or equivalent timing phrases are acceptable."
    )

    # Sources Support Recency (critical leaf; verify by URLs)
    recency_leaf = evaluator.add_leaf(
        id="sources_support_recency",
        desc="Cited sources for benefits are reasonably attributable to 2024–2025 (e.g., dated/updated in 2024–2025 or clearly presented as current benefits information).",
        parent=node,
        critical=True
    )
    all_urls = _collect_all_urls(report)
    recency_claim = (
        "This page is reasonably attributable to the 2024–2025 timeframe for current benefits information (e.g., shows an update/publish date in 2024 or 2025, "
        "or clearly labels benefits as 2024/2025 or 'current' during that timeframe)."
    )
    await evaluator.verify(
        claim=recency_claim,
        node=recency_leaf,
        sources=all_urls if all_urls else None,
        additional_instruction=_build_recency_instruction()
    )


async def build_category_checks(
    evaluator: Evaluator,
    parent_node,
    company: Optional[str],
    report: BenefitsReport
) -> None:
    """
    Build all 16 benefit category checks under the critical assessment node.
    Each category node is critical with two critical leaves:
      - Meets Minimum (or Exists)
      - Details + Acceptable Documentation
    """

    async def _two_leaf_category(
        cat_id: str,
        cat_desc: str,
        entry: Optional[BenefitEntry],
        meets_id: str,
        meets_desc: str,
        meets_claim: str,
        details_id: str,
        details_desc: str,
        details_label: str
    ):
        node = evaluator.add_parallel(
            id=cat_id,
            desc=cat_desc,
            parent=parent_node,
            critical=True
        )

        # Leaf 1: Meets Minimum / Exists
        leaf_min = evaluator.add_leaf(
            id=meets_id,
            desc=meets_desc,
            parent=node,
            critical=True
        )
        await evaluator.verify(
            claim=meets_claim,
            node=leaf_min,
            sources=_get_entry_urls(entry),
            additional_instruction="Judge only by the provided URLs. Allow reasonable phrasing equivalence. If URLs are irrelevant or do not support the claim, return 'Incorrect'."
        )

        # Leaf 2: Details + Acceptable Documentation
        leaf_details = evaluator.add_leaf(
            id=details_id,
            desc=details_desc,
            parent=node,
            critical=True
        )
        details_text = (entry.details if entry and entry.details else "")  # use empty if missing
        await evaluator.verify(
            claim=f"The specific detail is accurate: {details_label}: {details_text}",
            node=leaf_details,
            sources=_get_entry_urls(entry),
            additional_instruction=_build_acceptable_source_instruction(company, details_label)
        )

    # 1) 401(k) Retirement Plan
    await _two_leaf_category(
        cat_id="cat_401k",
        cat_desc="401(k) matching meets the minimum and is documented with acceptable sources.",
        entry=report.retirement_401k,
        meets_id="meets_401k_match_minimum",
        meets_desc="Employer match is at least 50% on employee contributions.",
        meets_claim=f"For {company or 'the company'}, the employer 401(k) match is at least 50% on employee contributions.",
        details_id="k401_details_documentation",
        details_desc="Response provides specific matching details and at least one supporting URL from an official company benefits page or a verified third-party benefits platform.",
        details_label="401(k) matching details"
    )

    # 2) Maternity Leave
    await _two_leaf_category(
        cat_id="cat_maternity_leave",
        cat_desc="Paid maternity leave meets the minimum and is documented with acceptable sources.",
        entry=report.maternity_leave,
        meets_id="meets_maternity_leave_minimum",
        meets_desc="Paid maternity leave for birthing parents is at least 16 weeks.",
        meets_claim=f"For {company or 'the company'}, paid maternity leave (birthing parent) is at least 16 weeks.",
        details_id="maternity_leave_details_doc",
        details_desc="Response provides specific maternity leave details and at least one supporting URL from an official company benefits page or a verified third-party benefits platform.",
        details_label="Maternity leave detail"
    )

    # 3) Paternity Leave
    await _two_leaf_category(
        cat_id="cat_paternity_leave",
        cat_desc="Paid paternity leave meets the minimum and is documented with acceptable sources.",
        entry=report.paternity_leave,
        meets_id="meets_paternity_leave_minimum",
        meets_desc="Paid paternity leave for non-birthing parents is at least 12 weeks.",
        meets_claim=f"For {company or 'the company'}, paid paternity leave (non-birthing parent) is at least 12 weeks.",
        details_id="paternity_leave_details_doc",
        details_desc="Response provides specific paternity leave details and at least one supporting URL from an official company benefits page or a verified third-party benefits platform.",
        details_label="Paternity leave detail"
    )

    # 4) Health Insurance
    await _two_leaf_category(
        cat_id="cat_health_insurance",
        cat_desc="Health premium requirement is met and documented.",
        entry=report.health_insurance,
        meets_id="meets_health_premium_requirement",
        meets_desc="Employee premium is $0 or minimal (fully covered or highly subsidized).",
        meets_claim=f"For {company or 'the company'}, employee-only medical premiums are $0 or effectively fully covered/highly subsidized.",
        details_id="health_insurance_details_doc",
        details_desc="Response provides specific premium/cost details and at least one supporting URL from an official company benefits page or a verified third-party benefits platform.",
        details_label="Health insurance premium detail"
    )

    # 5) Paid Time Off (PTO)
    await _two_leaf_category(
        cat_id="cat_pto",
        cat_desc="PTO meets the minimum and is documented.",
        entry=report.pto,
        meets_id="meets_pto_minimum",
        meets_desc="At least 20 days annual vacation/personal PTO, or unlimited PTO.",
        meets_claim=f"For {company or 'the company'}, the vacation/personal PTO is at least 20 days per year or is unlimited.",
        details_id="pto_details_doc",
        details_desc="Response provides specific PTO details and at least one supporting URL from an official company benefits page or a verified third-party benefits platform.",
        details_label="PTO detail"
    )

    # 6) Tuition Reimbursement
    await _two_leaf_category(
        cat_id="cat_tuition",
        cat_desc="Tuition/education assistance exists and is documented.",
        entry=report.tuition_reimbursement,
        meets_id="tuition_program_exists",
        meets_desc="There is an educational assistance or tuition reimbursement program.",
        meets_claim=f"For {company or 'the company'}, an educational assistance or tuition reimbursement program exists.",
        details_id="tuition_details_doc",
        details_desc="Response provides specific program details and at least one supporting URL from an official company benefits page or a verified third-party benefits platform.",
        details_label="Tuition/education assistance detail"
    )

    # 7) Adoption Assistance
    await _two_leaf_category(
        cat_id="cat_adoption",
        cat_desc="Adoption support meets the minimum and is documented.",
        entry=report.adoption_assistance,
        meets_id="meets_adoption_minimum",
        meets_desc="At least $10,000 per child in adoption support.",
        meets_claim=f"For {company or 'the company'}, adoption assistance is at least $10,000 per child.",
        details_id="adoption_details_doc",
        details_desc="Response provides specific adoption assistance details and at least one supporting URL from an official company benefits page or a verified third-party benefits platform.",
        details_label="Adoption assistance detail"
    )

    # 8) Wellness Reimbursement
    await _two_leaf_category(
        cat_id="cat_wellness",
        cat_desc="Wellness/gym reimbursement meets the minimum and is documented.",
        entry=report.wellness_reimbursement,
        meets_id="meets_wellness_minimum",
        meets_desc="At least $1,000 per year for gym membership or wellness expenses.",
        meets_claim=f"For {company or 'the company'}, wellness/gym reimbursement is at least $1,000 per year.",
        details_id="wellness_details_doc",
        details_desc="Response provides specific wellness benefit details and at least one supporting URL from an official company benefits page or a verified third-party benefits platform.",
        details_label="Wellness reimbursement detail"
    )

    # 9) Meal Benefits
    await _two_leaf_category(
        cat_id="cat_meals",
        cat_desc="Meal benefit exists and is documented.",
        entry=report.meal_benefits,
        meets_id="meal_benefit_exists",
        meets_desc="On-site or subsidized meal services (breakfast, lunch, or dinner) are provided.",
        meets_claim=f"For {company or 'the company'}, meal benefits exist in the form of on-site or subsidized meals (breakfast/lunch/dinner).",
        details_id="meal_benefit_details_doc",
        details_desc="Response provides specific meal benefit details and at least one supporting URL from an official company benefits page or a verified third-party benefits platform.",
        details_label="Meal benefit detail"
    )

    # 10) HSA Contribution
    await _two_leaf_category(
        cat_id="cat_hsa",
        cat_desc="Employer HSA contribution meets the minimum and is documented.",
        entry=report.hsa_contribution,
        meets_id="meets_hsa_minimum",
        meets_desc="Employer contributes at least $1,000 per year to an HSA (for eligible plans).",
        meets_claim=f"For {company or 'the company'}, the employer contributes at least $1,000 per year to an HSA for eligible high-deductible plans.",
        details_id="hsa_details_doc",
        details_desc="Response provides specific HSA contribution details and at least one supporting URL from an official company benefits page or a verified third-party benefits platform.",
        details_label="HSA contribution detail"
    )

    # 11) Student Loan Support
    await _two_leaf_category(
        cat_id="cat_student_loan",
        cat_desc="Student loan support exists and is documented.",
        entry=report.student_loan_support,
        meets_id="student_loan_support_exists",
        meets_desc="Provides loan repayment matching, assistance, or refinancing support program.",
        meets_claim=f"For {company or 'the company'}, there is student loan support such as repayment matching, assistance, or refinancing.",
        details_id="student_loan_details_doc",
        details_desc="Response provides specific student loan benefit details and at least one supporting URL from an official company benefits page or a verified third-party benefits platform.",
        details_label="Student loan support detail"
    )

    # 12) Life Insurance
    await _two_leaf_category(
        cat_id="cat_life_insurance",
        cat_desc="Life insurance meets the minimum and is documented.",
        entry=report.life_insurance,
        meets_id="meets_life_insurance_minimum",
        meets_desc="Coverage of at least 2x annual employee salary.",
        meets_claim=f"For {company or 'the company'}, company-paid basic life insurance coverage is at least 2x the employee's annual salary.",
        details_id="life_insurance_details_doc",
        details_desc="Response provides specific life insurance details and at least one supporting URL from an official company benefits page or a verified third-party benefits platform.",
        details_label="Life insurance detail"
    )

    # 13) Phone Reimbursement
    await _two_leaf_category(
        cat_id="cat_phone",
        cat_desc="Phone reimbursement meets the minimum and is documented.",
        entry=report.phone_reimbursement,
        meets_id="meets_phone_minimum",
        meets_desc="At least $50 per month for phone bill expenses.",
        meets_claim=f"For {company or 'the company'}, the phone reimbursement or stipend is at least $50 per month.",
        details_id="phone_reimbursement_details_doc",
        details_desc="Response provides specific phone reimbursement details and at least one supporting URL from an official company benefits page or a verified third-party benefits platform.",
        details_label="Phone reimbursement detail"
    )

    # 14) Remote Work
    await _two_leaf_category(
        cat_id="cat_remote_work",
        cat_desc="Remote work flexibility exists and is documented.",
        entry=report.remote_work,
        meets_id="meets_remote_work_minimum",
        meets_desc="Policy allows at least some work-from-home options (flexible remote work).",
        meets_claim=f"For {company or 'the company'}, the policy allows at least some work-from-home options (flexible remote work).",
        details_id="remote_work_details_doc",
        details_desc="Response provides specific remote work policy details and at least one supporting URL from an official company benefits page or a verified third-party benefits platform.",
        details_label="Remote work policy detail"
    )

    # 15) Fertility Benefits
    await _two_leaf_category(
        cat_id="cat_fertility",
        cat_desc="Fertility/family planning benefits exist and are documented.",
        entry=report.fertility_benefits,
        meets_id="fertility_benefits_exist",
        meets_desc="Provides fertility treatment assistance or family planning benefits.",
        meets_claim=f"For {company or 'the company'}, fertility benefits exist (e.g., fertility treatment assistance or family planning coverage).",
        details_id="fertility_details_doc",
        details_desc="Response provides specific fertility benefit details and at least one supporting URL from an official company benefits page or a verified third-party benefits platform.",
        details_label="Fertility/family planning benefit detail"
    )


# --------------------------------------------------------------------------- #
# Main evaluation entry point                                                 #
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
    Evaluate an answer for the Tech Company Benefits Package Assessment (2024–2025).
    """

    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Top-level: independent checks aggregated under a critical assessment node
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
    report: BenefitsReport = await evaluator.extract(
        prompt=prompt_extract_benefits(),
        template_class=BenefitsReport,
        extraction_name="benefits_extraction"
    )

    # Add GT/Context info for reference
    evaluator.add_ground_truth({
        "allowed_companies": ALLOWED_COMPANIES,
        "required_minimums": {
            "401k": ">= 50% match",
            "maternity_weeks": ">= 16 weeks paid",
            "paternity_weeks": ">= 12 weeks paid",
            "health_premium": "$0 or minimal (fully covered/highly subsidized)",
            "pto": ">= 20 days per year or unlimited",
            "tuition_program": "exists",
            "adoption_assistance": ">= $10,000 per child",
            "wellness": ">= $1,000 per year",
            "meals": "on-site or subsidized meals",
            "hsa": ">= $1,000 per year",
            "student_loan": "support program exists",
            "life_insurance": ">= 2x annual salary",
            "phone": ">= $50 per month",
            "remote_work": "some WFH allowed",
            "fertility": "benefits exist"
        }
    })

    # Create the critical assessment node mirroring the rubric's root
    assessment = evaluator.add_parallel(
        id="assessment",
        desc="Determine whether exactly one of the specified companies is identified and whether it meets all minimum benefit requirements, with per-category details and acceptable documentation (current as of 2024–2025).",
        parent=root,
        critical=True
    )

    # Company selection checks
    await build_company_selection_checks(evaluator, assessment, report)

    # Recency checks
    await build_recency_checks(evaluator, assessment, report)

    # All category checks
    await build_category_checks(evaluator, assessment, report.selected_company, report)

    # Return summary
    return evaluator.get_summary()