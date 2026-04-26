import asyncio
import logging
import re
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "professional_dev_plans_4_employees_section127"
TASK_DESCRIPTION = (
    "A corporate HR department is evaluating a comprehensive professional development initiative to support four employees across different career paths and geographic locations. "
    "The company has established an educational assistance program under IRS Section 127, providing up to $5,250 per employee annually in tax-free benefits. "
    "You need to create a detailed professional development plan for each employee that ensures they meet all certification requirements, continuing education obligations, and state-specific mandates while maximizing the use of employer benefits.\n\n"
    "Employee 1 - Project Manager (Sarah): PMP certification within 12 months, 35 hours training prerequisite, PMP recert 60 PDUs/3y incl. min 18 Education PDUs, considering PMI membership.\n"
    "Employee 2 - Financial Advisor (Michael): CFP certification plan (6-hour, 170-question exam), CE 30 hrs/2y incl. 2 ethics, budget for exam fees, CFP Board-registered coursework, and CE.\n"
    "Employee 3 - HR Manager (Jennifer): SHRM-SCP recert, cycle ends in 14 months, needs total 60 PDCs (has 22), PDC categories (Education; Advancing Your Organization; Giving Back), considering SHRM Annual Conference.\n"
    "Employee 4 - Teacher (David): Texas standard teaching (150 CPE/5y) and administrative (200 CPE/5y) certificates; both expire in 18 months; CPE to be with TEA-approved providers; district PD plus additional hours.\n\n"
    "Your task: For each employee, provide a complete plan including: Requirements, Timeline & milestones, Section 127 utilization (up to $5,250/year; eligible: tuition/fees/books/supplies/equipment for courses at educational institutions; exam fees; required educational materials), PD activities, cost-benefit analyses where requested, and compliance verification (state rules, cert body rules, IRS §127, employer pre-approval)."
)

ALLOWED_SECTION127_KEYWORDS = [
    # Core categories per prompt
    "tuition", "course", "courses", "coursework", "university", "college", "educational institution",
    "fees", "exam fee", "exam fees", "examination fee", "examination fees",
    "books", "textbook", "textbooks",
    "supplies",
    "equipment",
    "required material", "required materials",
]

SECTION127_ANNUAL_LIMIT = 5250.0

# Ground truth numeric references (for validation-style checks)
GT_EMP3_TOTAL_REQUIRED_PDC = 60
GT_EMP3_EARNED_PDC = 22
GT_EMP3_REMAINING_PDC = GT_EMP3_TOTAL_REQUIRED_PDC - GT_EMP3_EARNED_PDC  # 38

GT_EMP4_TEACHING_TOTAL = 150
GT_EMP4_TEACHING_EARNED = 85
GT_EMP4_TEACHING_REMAINING = GT_EMP4_TEACHING_TOTAL - GT_EMP4_TEACHING_EARNED  # 65

GT_EMP4_ADMIN_TOTAL = 200
GT_EMP4_ADMIN_EARNED = 90
GT_EMP4_ADMIN_REMAINING = GT_EMP4_ADMIN_TOTAL - GT_EMP4_ADMIN_EARNED  # 110


# --------------------------------------------------------------------------- #
# Helper parsing utilities                                                    #
# --------------------------------------------------------------------------- #
def parse_amount_to_float(value: Optional[str]) -> Optional[float]:
    if value is None:
        return None
    # Extract first number-like token
    m = re.search(r"(-?\d[\d,]*(?:\.\d+)?)", value.replace("$", "").replace("USD", ""), flags=re.IGNORECASE)
    if not m:
        return None
    try:
        return float(m.group(1).replace(",", ""))
    except Exception:
        return None


def sum_expense_amounts(expenses: List["ExpenseItem"]) -> Optional[float]:
    total = 0.0
    seen_any = False
    for it in expenses:
        amt = parse_amount_to_float(it.amount if hasattr(it, "amount") else None)
        if amt is not None:
            total += amt
            seen_any = True
    return total if seen_any else None


def categories_all_eligible(expenses: List["ExpenseItem"]) -> bool:
    if not expenses:
        return False
    for it in expenses:
        cat = (it.category or "").lower()
        desc = (it.description or "").lower()
        text = f"{cat} {desc}".strip()
        if not any(kw in text for kw in ALLOWED_SECTION127_KEYWORDS):
            return False
    return True


def parse_int(value: Optional[str]) -> Optional[int]:
    if value is None:
        return None
    m = re.search(r"(-?\d+)", value)
    if not m:
        return None
    try:
        return int(m.group(1))
    except Exception:
        return None


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class ExpenseItem(BaseModel):
    description: Optional[str] = None
    category: Optional[str] = None
    amount: Optional[str] = None  # Keep as string for robust extraction


class Section127Utilization(BaseModel):
    total_planned_amount: Optional[str] = None  # string; we'll parse
    expenses: List[ExpenseItem] = Field(default_factory=list)


class GlobalSection127Extraction(BaseModel):
    mentions_written_plan: Optional[bool] = None
    mentions_nondiscrimination: Optional[bool] = None
    includes_employer_preapproval_step: Optional[bool] = None


class Emp1Extraction(BaseModel):
    pmp_experience_requirement_met: Optional[bool] = None
    includes_35_hours_training_before_exam: Optional[bool] = None
    provides_timeline_within_12_months: Optional[bool] = None
    timeline_orders_prereqs_correctly: Optional[bool] = None
    recert_identifies_60_pdu_and_18_education: Optional[bool] = None
    pdu_activity_types: List[str] = Field(default_factory=list)
    pmi_membership_costs_mentioned: Optional[bool] = None
    pmi_membership_benefits_mentioned: Optional[bool] = None
    pmi_membership_recommendation: Optional[str] = None
    section127: Section127Utilization = Field(default_factory=Section127Utilization)


class Emp2Extraction(BaseModel):
    bachelors_prereq_addressed: Optional[bool] = None
    experience_requirement_addressed: Optional[bool] = None
    cfp_registered_coursework_included: Optional[bool] = None
    exam_requirement_included_6hr_170q: Optional[bool] = None
    ce_requirement_identified_30hr_2yr_incl_2_ethics: Optional[bool] = None
    timeline_for_cert_steps: Optional[bool] = None
    ongoing_ce_plan_references_2yr_cycle: Optional[bool] = None
    ce_activity_types: List[str] = Field(default_factory=list)
    section127: Section127Utilization = Field(default_factory=Section127Utilization)
    budget_includes_exam_coursework_ce: Optional[bool] = None


class Emp3Extraction(BaseModel):
    shrm_recert_requirement_stated_60pdc_3yr: Optional[bool] = None
    remaining_pdc_calculated: Optional[str] = None  # expect "38" or phrase containing 38
    timeline_meets_14_month_deadline: Optional[bool] = None
    uses_three_categories: Optional[bool] = None  # Education; Advancing Your Organization; Giving Back
    pdc_activity_types: List[str] = Field(default_factory=list)
    shrm_conference_costs_mentioned: Optional[bool] = None
    shrm_conference_benefits_mentioned: Optional[bool] = None
    shrm_conference_recommendation: Optional[str] = None
    section127: Section127Utilization = Field(default_factory=Section127Utilization)


class Emp4Extraction(BaseModel):
    teaching_cpe_requirement_stated_150_5y: Optional[bool] = None
    admin_cpe_requirement_stated_200_5y: Optional[bool] = None
    remaining_teaching_cpe: Optional[str] = None
    remaining_admin_cpe: Optional[str] = None
    timeline_meets_18_month_deadline: Optional[bool] = None
    uses_tea_approved_providers: Optional[bool] = None
    activity_types_include_district_and_supplement: Optional[bool] = None
    section127: Section127Utilization = Field(default_factory=Section127Utilization)


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_global_section127() -> str:
    return """
    Extract whether the answer explicitly covers the following IRS Section 127 program-level compliance points (True/False):
    - mentions_written_plan: The Section 127 educational assistance program must be a written plan.
    - mentions_nondiscrimination: The program cannot discriminate in favor of highly compensated employees.
    - includes_employer_preapproval_step: The plan includes an employer pre-approval step before enrolling/spending.
    Set any item to false if not clearly stated in the answer.
    """


def prompt_extract_emp1() -> str:
    return """
    From the answer's plan for Employee 1 (Sarah, Project Manager pursuing PMP), extract the following fields (True/False unless otherwise specified):
    - pmp_experience_requirement_met: Confirms the PMP experience requirement is met for the bachelor's degree pathway (36 months) given 38 months experience.
    - includes_35_hours_training_before_exam: Includes 35 hours of project management education BEFORE taking the PMP exam.
    - provides_timeline_within_12_months: Timeline ensures PMP certification is completed within 12 months.
    - timeline_orders_prereqs_correctly: Timeline explicitly places the 35 hours training before the exam.
    - recert_identifies_60_pdu_and_18_education: Identifies PMP recertification as 60 PDUs per 3-year cycle incl. minimum 18 Education PDUs.
    - pdu_activity_types: List the concrete types of activities named to earn PDUs (e.g., courses, webinars, conferences, volunteering).
    - pmi_membership_costs_mentioned: Mentions the costs or dues of PMI membership.
    - pmi_membership_benefits_mentioned: Mentions benefits (discounts, networking, PDU opportunities).
    - pmi_membership_recommendation: The stated recommendation/decision (e.g., join now, defer, or similar). Use a short phrase or null.

    Also extract Section 127 allocation for Employee 1 (if any):
    - section127.total_planned_amount: total per-year amount planned under Section 127 (string, keep currency formatting if any).
    - section127.expenses: array where each item has:
        * description: short text of the expense
        * category: category such as tuition/course/fees/exam/books/supplies/equipment/required materials
        * amount: amount string (e.g., "$1,200"), if stated; else null
    """


def prompt_extract_emp2() -> str:
    return """
    From the answer's plan for Employee 2 (Michael, Financial Advisor pursuing CFP), extract the following (True/False unless otherwise noted):
    - bachelors_prereq_addressed: Confirms bachelor’s degree prerequisite is satisfied.
    - experience_requirement_addressed: Confirms that 5,200 hours meets the 4,000–6,000 hour CFP experience requirement.
    - cfp_registered_coursework_included: Includes CFP Board-registered coursework in the plan/budget.
    - exam_requirement_included_6hr_170q: Includes the CFP exam requirement, explicitly noted as a 6-hour, 170-question exam.
    - ce_requirement_identified_30hr_2yr_incl_2_ethics: Identifies CE as 30 hours every 2 years incl. 2 hours of ethics.
    - timeline_for_cert_steps: Provides a timeline/milestones covering coursework and exam steps (with dependencies).
    - ongoing_ce_plan_references_2yr_cycle: Ongoing CE plan references the 2-year CE reporting cycle.
    - ce_activity_types: List the activity types named to earn CE (e.g., approved CE courses, webinars).
    - budget_includes_exam_coursework_ce: The budget includes (at minimum) exam fees, registered coursework, and continuing education costs (can be separate line items or noted in text).

    Also extract Section 127 allocation for Employee 2 (if any):
    - section127.total_planned_amount: total per-year amount planned under Section 127 (string).
    - section127.expenses: array of {description, category, amount} as stated.
    """


def prompt_extract_emp3() -> str:
    return """
    From the answer's plan for Employee 3 (Jennifer, HR Manager with SHRM-SCP recert in 14 months), extract:
    - shrm_recert_requirement_stated_60pdc_3yr (True/False): States SHRM recert as 60 PDCs within a 3-year cycle.
    - remaining_pdc_calculated (string or number): The explicitly stated remaining PDCs needed (should be 38 given 60 required and 22 earned). Use just the number if possible; else include phrase.
    - timeline_meets_14_month_deadline (True/False): Milestones complete remaining PDCs before 14 months.
    - uses_three_categories (True/False): Plans PDC earning using the three categories (Education; Advancing Your Organization; Giving Back).
    - pdc_activity_types: List concrete activity types named to earn PDCs.
    - shrm_conference_costs_mentioned (True/False): Mentions SHRM Annual Conference costs.
    - shrm_conference_benefits_mentioned (True/False): Mentions benefits (PDCs, networking, learning).
    - shrm_conference_recommendation (string): Recommendation/decision about attending.

    Also extract Section 127 allocation for Employee 3:
    - section127.total_planned_amount (string)
    - section127.expenses: array of {description, category, amount}
    """


def prompt_extract_emp4() -> str:
    return """
    From the answer's plan for Employee 4 (David, Texas educator with both standard teaching and administrative certificates, both expire in 18 months), extract:
    - teaching_cpe_requirement_stated_150_5y (True/False): States 150 CPE hours every 5 years for the teaching certificate.
    - admin_cpe_requirement_stated_200_5y (True/False): States 200 CPE hours every 5 years for the administrative certificate.
    - remaining_teaching_cpe (string or number): The explicitly stated remaining hours needed for teaching (should be 65, given 150 required and 85 completed).
    - remaining_admin_cpe (string or number): The explicitly stated remaining hours needed for admin (should be 110, given 200 required and 90 completed).
    - timeline_meets_18_month_deadline (True/False): Plans to complete required hours before 18 months.
    - uses_tea_approved_providers (True/False): Ensures TEA-approved providers are used.
    - activity_types_include_district_and_supplement (True/False): Names district on-site PD plus additional TEA-approved options.

    Also extract Section 127 allocation for Employee 4:
    - section127.total_planned_amount (string)
    - section127.expenses: array of {description, category, amount}
    """


# --------------------------------------------------------------------------- #
# Verification helpers (tree builders)                                        #
# --------------------------------------------------------------------------- #
async def verify_global_section127_rules(evaluator: Evaluator, parent) -> None:
    node = evaluator.add_parallel(
        id="global_section127_program_rules",
        desc="Addresses IRS Section 127 program-level compliance requirements stated in constraints and includes employer pre-approval as a compliance step.",
        parent=parent,
        critical=True
    )

    extraction = await evaluator.extract(
        prompt=prompt_extract_global_section127(),
        template_class=GlobalSection127Extraction,
        extraction_name="global_section127_rules"
    )

    # 3 leaf checks (simple presence in the answer)
    written_node = evaluator.add_leaf(
        id="written_plan_requirement",
        desc="States that the Section 127 educational assistance program must be a written plan.",
        parent=node,
        critical=True
    )
    nondisc_node = evaluator.add_leaf(
        id="nondiscrimination_requirement",
        desc="States that the Section 127 program cannot discriminate in favor of highly compensated employees.",
        parent=node,
        critical=True
    )
    preapprove_node = evaluator.add_leaf(
        id="employer_preapproval_step",
        desc="Includes employer pre-approval as a required compliance step before enrolling/spending (per prompt/constraints).",
        parent=node,
        critical=True
    )

    claims_and_sources: List[Tuple[str, Optional[List[str] | str | None], Any, Optional[str]]] = [
        (
            "The response explicitly states that the Section 127 educational assistance program must be a written plan.",
            None,
            written_node,
            "Focus only on whether the answer states this requirement; accept synonyms or paraphrasing."
        ),
        (
            "The response explicitly states that the Section 127 program cannot discriminate in favor of highly compensated employees.",
            None,
            nondisc_node,
            "Focus only on whether the answer states this requirement; accept synonyms or paraphrasing."
        ),
        (
            "The response includes an employer pre‑approval step as a required compliance action before enrolling or spending any funds.",
            None,
            preapprove_node,
            "Judge only based on the answer content; accept phrases like 'manager/HR approval' as equivalent."
        ),
    ]
    await evaluator.batch_verify(claims_and_sources)


def add_section127_subtree(
    evaluator: Evaluator,
    parent,
    id_prefix: str,
    desc_prefix: str,
    sec127: Section127Utilization
) -> None:
    node = evaluator.add_parallel(
        id=f"{id_prefix}_section127_utilization",
        desc=f"Calculates/allocates Section 127 benefit usage for {desc_prefix} and maps claimed expenses to eligible categories listed in the prompt, while respecting the annual cap.",
        parent=parent,
        critical=True
    )

    # Compute annual limit check
    # Prefer explicit total_planned_amount if present; otherwise sum line items
    computed_total = None
    if sec127 and sec127.total_planned_amount:
        computed_total = parse_amount_to_float(sec127.total_planned_amount)
    if computed_total is None:
        computed_total = sum_expense_amounts(sec127.expenses if sec127 else [])

    within_limit = (computed_total is not None) and (computed_total <= SECTION127_ANNUAL_LIMIT)

    evaluator.add_custom_node(
        result=within_limit,
        id=f"{id_prefix}_section127_within_annual_limit",
        desc="Does not exceed the $5,250 per-employee annual tax-free limit.",
        parent=node,
        critical=True
    )

    # Category eligibility (all claimed categories must be within allowed set)
    eligible_only = categories_all_eligible(sec127.expenses if sec127 else [])
    evaluator.add_custom_node(
        result=eligible_only,
        id=f"{id_prefix}_section127_expense_eligibility",
        desc="Limits claimed Section 127 expenses to the eligible categories provided in the prompt (tuition/fees/books/supplies/equipment for courses at educational institutions; exam fees; required educational materials).",
        parent=node,
        critical=True
    )


async def verify_employee_1(evaluator: Evaluator, parent) -> None:
    emp_node = evaluator.add_parallel(
        id="employee_1_plan",
        desc="Employee 1 (Project Manager) plan covering PMP certification within 12 months and PMP recertification planning.",
        parent=parent,
        critical=True
    )

    extraction = await evaluator.extract(
        prompt=prompt_extract_emp1(),
        template_class=Emp1Extraction,
        extraction_name="emp1_extraction"
    )

    # Leaves
    exp_req_node = evaluator.add_leaf(
        id="emp1_pmp_experience_requirement_met",
        desc="Confirms the PMP experience requirement is satisfied for the 4-year degree pathway (36 months required) given the stated months of experience.",
        parent=emp_node,
        critical=True
    )
    hours35_node = evaluator.add_leaf(
        id="emp1_pmp_35_hours_training_included",
        desc="Includes completion of 35 hours of project management education/training before taking the PMP exam (per constraints).",
        parent=emp_node,
        critical=True
    )
    t12_node = evaluator.add_leaf(
        id="emp1_timeline_cert_within_12_months",
        desc="Provides milestones that complete PMP certification within the stated 12-month window.",
        parent=emp_node,
        critical=True
    )
    order_node = evaluator.add_leaf(
        id="emp1_timeline_prerequisite_ordering",
        desc="Orders prerequisites correctly in the timeline (35 training hours completed before the PMP exam).",
        parent=emp_node,
        critical=True
    )
    recert_node = evaluator.add_leaf(
        id="emp1_pmp_recert_requirement_identified",
        desc="Identifies PMP recertification requirement: 60 PDUs per 3-year cycle with a minimum of 18 Education PDUs (per prompt/constraints).",
        parent=emp_node,
        critical=True
    )
    pdu_types_node = evaluator.add_leaf(
        id="emp1_pdu_activity_types_specified",
        desc="Names specific activity types the employee can use to earn PDUs consistent with PMP recertification (e.g., courses, webinars, conferences, volunteering).",
        parent=emp_node,
        critical=True
    )

    cb_pmi_node = evaluator.add_leaf(
        id="emp1_cost_benefit_pmi_membership",
        desc="Provides a cost-benefit analysis of optional PMI membership that discusses costs and benefits (discounts/networking/PDU opportunities) and gives a recommendation/decision.",
        parent=emp_node,
        critical=True
    )

    claims = [
        (
            "For Employee 1, the plan confirms Sarah meets the PMP experience requirement for the bachelor's degree pathway (at least 36 months), given her 38 months of experience.",
            exp_req_node
        ),
        (
            "For Employee 1, the plan requires completing 35 contact hours of project management education before taking the PMP exam.",
            hours35_node
        ),
        (
            "For Employee 1, the timeline ensures PMP certification is completed within 12 months.",
            t12_node
        ),
        (
            "For Employee 1, the timeline explicitly places the 35 training hours before the PMP exam.",
            order_node
        ),
        (
            "For Employee 1, the plan states PMP recertification requires 60 PDUs every 3 years, including at least 18 Education PDUs.",
            recert_node
        ),
        (
            f"For Employee 1, the plan lists concrete PDU-earning activities (e.g., {', '.join(extraction.pdu_activity_types) if extraction.pdu_activity_types else 'named activities'}).",
            pdu_types_node
        ),
        (
            "For Employee 1, the plan includes a cost-benefit analysis of PMI membership that mentions costs and benefits (discounts, networking, PDU opportunities) and provides a clear recommendation or decision.",
            cb_pmi_node
        ),
    ]
    await evaluator.batch_verify([
        (c, None, n, "Judge only based on explicit content in the answer; accept paraphrases and synonyms.") for c, n in claims
    ])

    # Section 127 subtree
    add_section127_subtree(
        evaluator,
        emp_node,
        id_prefix="emp1",
        desc_prefix="Employee 1",
        sec127=extraction.section127
    )


async def verify_employee_2(evaluator: Evaluator, parent) -> None:
    emp_node = evaluator.add_parallel(
        id="employee_2_plan",
        desc="Employee 2 (Financial Advisor) plan covering CFP certification and CFP continuing education planning.",
        parent=parent,
        critical=True
    )

    extraction = await evaluator.extract(
        prompt=prompt_extract_emp2(),
        template_class=Emp2Extraction,
        extraction_name="emp2_extraction"
    )

    # Leaves
    bach_node = evaluator.add_leaf(
        id="emp2_bachelors_prereq_addressed",
        desc="Confirms the bachelor’s degree prerequisite is satisfied (as stated for Employee 2).",
        parent=emp_node,
        critical=True
    )
    exp_node = evaluator.add_leaf(
        id="emp2_experience_requirement_addressed",
        desc="Confirms the stated experience hours meet the 4,000–6,000 hour CFP experience requirement (per constraints).",
        parent=emp_node,
        critical=True
    )
    coursework_node = evaluator.add_leaf(
        id="emp2_cfp_registered_coursework_included",
        desc="Includes CFP Board-registered coursework as part of the certification plan/budget (per prompt).",
        parent=emp_node,
        critical=True
    )
    exam_node = evaluator.add_leaf(
        id="emp2_exam_requirement_included",
        desc="Includes the CFP exam requirement as stated (6-hour, 170-question exam).",
        parent=emp_node,
        critical=True
    )
    ce_req_node = evaluator.add_leaf(
        id="emp2_cfp_ce_requirement_identified",
        desc="Identifies CFP continuing education requirement: 30 hours every 2 years including 2 hours ethics (per prompt/constraints).",
        parent=emp_node,
        critical=True
    )
    timeline_node = evaluator.add_leaf(
        id="emp2_timeline_for_certification_steps",
        desc="Provides milestones covering CFP coursework and exam steps, including any prerequisite/sequential dependencies.",
        parent=emp_node,
        critical=True
    )
    ce_cycle_node = evaluator.add_leaf(
        id="emp2_ongoing_ce_plan_references_2_year_cycle",
        desc="Provides an ongoing CE plan that references the 2-year CE reporting cycle.",
        parent=emp_node,
        critical=True
    )
    ce_types_node = evaluator.add_leaf(
        id="emp2_ce_activity_types_specified",
        desc="Names specific activity types to earn required CE consistent with CFP requirements (e.g., approved CE courses/webinars).",
        parent=emp_node,
        critical=True
    )
    budget_inc_node = evaluator.add_leaf(
        id="emp2_budget_includes_required_cost_types",
        desc="Includes budgeting for exam fees, CFP Board-registered coursework, and continuing education costs (per prompt).",
        parent=emp_node,
        critical=True
    )

    claims = [
        ("For Employee 2, the plan confirms the bachelor’s degree prerequisite is satisfied.", bach_node),
        ("For Employee 2, the plan confirms that 5,200 hours meets the 4,000–6,000 hour CFP experience requirement.", exp_node),
        ("For Employee 2, the plan includes CFP Board‑registered coursework in the certification plan/budget.", coursework_node),
        ("For Employee 2, the plan includes the CFP exam requirement and states it is a 6‑hour, 170‑question exam.", exam_node),
        ("For Employee 2, the plan identifies CE as 30 hours every 2 years including 2 hours of ethics.", ce_req_node),
        ("For Employee 2, the plan provides a timeline covering coursework and exam steps with proper prerequisites.", timeline_node),
        ("For Employee 2, the ongoing CE plan references the 2‑year CE reporting cycle.", ce_cycle_node),
        (
            f"For Employee 2, the plan names specific CE activity types (e.g., {', '.join(extraction.ce_activity_types) if extraction.ce_activity_types else 'approved CE courses/webinars'}).",
            ce_types_node
        ),
        ("For Employee 2, the budget explicitly includes exam fees, registered coursework costs, and continuing education costs.", budget_inc_node),
    ]
    await evaluator.batch_verify([
        (c, None, n, "Judge only based on the answer; accept synonyms and paraphrases.") for c, n in claims
    ])

    # Section 127 subtree
    add_section127_subtree(
        evaluator,
        emp_node,
        id_prefix="emp2",
        desc_prefix="Employee 2",
        sec127=extraction.section127
    )


async def verify_employee_3(evaluator: Evaluator, parent) -> None:
    emp_node = evaluator.add_parallel(
        id="employee_3_plan",
        desc="Employee 3 (HR Manager) plan for SHRM-SCP recertification within remaining cycle time and optional SHRM conference evaluation.",
        parent=parent,
        critical=True
    )

    extraction = await evaluator.extract(
        prompt=prompt_extract_emp3(),
        template_class=Emp3Extraction,
        extraction_name="emp3_extraction"
    )

    # Leaves
    req_node = evaluator.add_leaf(
        id="emp3_shrm_recert_requirement_stated",
        desc="States SHRM recertification requirement: 60 PDCs within a 3-year recertification cycle (per prompt/constraints).",
        parent=emp_node,
        critical=True
    )
    timeline_node = evaluator.add_leaf(
        id="emp3_timeline_meets_14_month_deadline",
        desc="Provides milestones that complete the remaining PDCs before the stated cycle end (14 months).",
        parent=emp_node,
        critical=True
    )
    cats_node = evaluator.add_leaf(
        id="emp3_pdc_categories_used",
        desc="Plans PDC earning using SHRM’s three categories stated (Education; Advancing Your Organization; Giving Back).",
        parent=emp_node,
        critical=True
    )
    act_types_node = evaluator.add_leaf(
        id="emp3_activity_types_for_pdcs_specified",
        desc="Names specific activity types tied to earning PDCs (e.g., webinars/courses, organizational projects, volunteering).",
        parent=emp_node,
        critical=True
    )
    conf_cb_node = evaluator.add_leaf(
        id="emp3_cost_benefit_shrm_annual_conference",
        desc="Provides a cost-benefit analysis for optional SHRM Annual Conference attendance that discusses costs and benefits (PDCs/networking/learning) and gives a recommendation/decision.",
        parent=emp_node,
        critical=True
    )

    # Simple verification leaves
    claims = [
        ("For Employee 3, the plan states SHRM recertification requires 60 PDCs within a 3‑year cycle.", req_node),
        ("For Employee 3, the plan provides milestones to complete the remaining PDCs before the 14‑month deadline.", timeline_node),
        ("For Employee 3, the plan leverages SHRM’s three PDC categories: Education; Advancing Your Organization; and Giving Back.", cats_node),
        (
            f"For Employee 3, the plan lists specific PDC‑earning activity types (e.g., {', '.join(extraction.pdc_activity_types) if extraction.pdc_activity_types else 'webinars/courses, projects, volunteering'}).",
            act_types_node
        ),
        ("For Employee 3, the plan includes a cost‑benefit analysis of the SHRM Annual Conference, discussing costs and benefits and giving a recommendation.", conf_cb_node),
    ]
    await evaluator.batch_verify([
        (c, None, n, "Judge only based on the answer; accept synonyms and paraphrases.") for c, n in claims
    ])

    # Custom node for remaining PDC correctly calculated (should be 38)
    remaining_pdc_val = parse_int(extraction.remaining_pdc_calculated)
    evaluator.add_custom_node(
        result=(remaining_pdc_val == GT_EMP3_REMAINING_PDC),
        id="emp3_remaining_pdc_need_calculated_correctly",
        desc="Correctly calculates remaining PDCs needed as (required PDCs minus earned-to-date PDCs) using the stated earned progress.",
        parent=emp_node,
        critical=True
    )

    # Section 127 subtree
    add_section127_subtree(
        evaluator,
        emp_node,
        id_prefix="emp3",
        desc_prefix="Employee 3",
        sec127=extraction.section127
    )


async def verify_employee_4(evaluator: Evaluator, parent) -> None:
    emp_node = evaluator.add_parallel(
        id="employee_4_plan",
        desc="Employee 4 (Texas educator) plan for renewal of standard teaching and administrative certificates under Texas rules.",
        parent=parent,
        critical=True
    )

    extraction = await evaluator.extract(
        prompt=prompt_extract_emp4(),
        template_class=Emp4Extraction,
        extraction_name="emp4_extraction"
    )

    # Leaves for stated requirements and planning
    teach_req_node = evaluator.add_leaf(
        id="emp4_teaching_certificate_requirement_stated",
        desc="States teaching certificate renewal requirement: 150 CPE hours every 5 years (per prompt/constraints).",
        parent=emp_node,
        critical=True
    )
    admin_req_node = evaluator.add_leaf(
        id="emp4_admin_certificate_requirement_stated",
        desc="States administrative certificate renewal requirement: 200 CPE hours every 5 years (per prompt/constraints).",
        parent=emp_node,
        critical=True
    )
    timeline_node = evaluator.add_leaf(
        id="emp4_timeline_meets_18_month_deadline",
        desc="Plan completes required CPE hours before the stated expiration timing (both expire in 18 months).",
        parent=emp_node,
        critical=True
    )
    tea_prov_node = evaluator.add_leaf(
        id="emp4_tea_approved_providers_used",
        desc="Ensures CPE activities/providers used are TEA-approved (per prompt).",
        parent=emp_node,
        critical=True
    )
    act_types_node = evaluator.add_leaf(
        id="emp4_activity_types_include_district_and_supplement",
        desc="Names specific activity types including district on-site PD plus additional TEA-approved options to reach required hours.",
        parent=emp_node,
        critical=True
    )

    claims = [
        ("For Employee 4, the plan states that the teaching certificate requires 150 CPE hours every 5 years.", teach_req_node),
        ("For Employee 4, the plan states that the administrative certificate requires 200 CPE hours every 5 years.", admin_req_node),
        ("For Employee 4, the plan completes the required hours before the 18‑month expiration window.", timeline_node),
        ("For Employee 4, the plan ensures use of TEA‑approved providers for CPE activities.", tea_prov_node),
        ("For Employee 4, the plan includes district on‑site PD and additional TEA‑approved options to reach the hours.", act_types_node),
    ]
    await evaluator.batch_verify([
        (c, None, n, "Judge only based on the answer; accept synonyms and paraphrases.") for c, n in claims
    ])

    # Custom node: remaining CPE hours correctly calculated for both certs
    remain_teach = parse_int(extraction.remaining_teaching_cpe)
    remain_admin = parse_int(extraction.remaining_admin_cpe)
    evaluator.add_custom_node(
        result=(remain_teach == GT_EMP4_TEACHING_REMAINING and remain_admin == GT_EMP4_ADMIN_REMAINING),
        id="emp4_remaining_hours_calculated_correctly",
        desc="Correctly calculates remaining CPE hours needed for each certificate using the stated completed CPE totals.",
        parent=emp_node,
        critical=True
    )

    # Section 127 subtree
    add_section127_subtree(
        evaluator,
        emp_node,
        id_prefix="emp4",
        desc_prefix="Employee 4",
        sec127=extraction.section127
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

    # Record ground-truth numeric references for transparent judging context
    evaluator.add_ground_truth({
        "section127_annual_limit": SECTION127_ANNUAL_LIMIT,
        "emp3_required_pdc": GT_EMP3_TOTAL_REQUIRED_PDC,
        "emp3_earned_pdc": GT_EMP3_EARNED_PDC,
        "emp3_expected_remaining_pdc": GT_EMP3_REMAINING_PDC,
        "emp4_teaching_total": GT_EMP4_TEACHING_TOTAL,
        "emp4_teaching_earned": GT_EMP4_TEACHING_EARNED,
        "emp4_teaching_expected_remaining": GT_EMP4_TEACHING_REMAINING,
        "emp4_admin_total": GT_EMP4_ADMIN_TOTAL,
        "emp4_admin_earned": GT_EMP4_ADMIN_EARNED,
        "emp4_admin_expected_remaining": GT_EMP4_ADMIN_REMAINING,
    }, gt_type="ground_truth_numbers")

    # Build tree per rubric
    # Global Section 127 rules
    await verify_global_section127_rules(evaluator, root)

    # Employee subtrees
    await asyncio.gather(
        verify_employee_1(evaluator, root),
        verify_employee_2(evaluator, root),
        verify_employee_3(evaluator, root),
        verify_employee_4(evaluator, root),
    )

    return evaluator.get_summary()