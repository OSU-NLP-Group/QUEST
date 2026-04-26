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
TASK_ID = "obbb_2025_documentation"
TASK_DESCRIPTION = (
    "Provide comprehensive documentation of the One Big Beautiful Bill Act of 2025, including: "
    "(1) Bill Identification: The House bill number, Public Law number, and the date it was signed into law. "
    "(2) SNAP Work Requirement Changes: The new age range for work requirements, the effective date when changes took effect, "
    "the required monthly work hours, the change to parent eligibility based on children's ages, and the total SNAP budget reduction over 10 years. "
    "(3) Medicaid Program Changes: The amount of federal Medicaid spending reduction, the estimated impact on the number of uninsured Americans, "
    "and the elimination of federal matching fund increases from a previous act. "
    "(4) Tax Code Modifications: The bonus depreciation percentage for qualifying business property, the amount of the permanent child tax credit increase, "
    "the new SALT (state and local tax) deduction limit, the additional deduction amount for seniors aged 65 and older, and the tax treatment of tips and overtime pay. "
    "(5) Budget Impact: The overall fiscal direction of the legislation and whether it extends the TCJA (Tax Cuts and Jobs Act) individual provisions. "
    "For each major category, provide at least one authoritative URL reference to support the information."
)
ACT_NAME = "One Big Beautiful Bill Act of 2025"


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class BillIdentification(BaseModel):
    house_bill_number: Optional[str] = None
    public_law_number: Optional[str] = None
    signing_date: Optional[str] = None
    urls: List[str] = Field(default_factory=list)


class SNAPRequirements(BaseModel):
    age_range: Optional[str] = None
    effective_date: Optional[str] = None
    monthly_work_hours: Optional[str] = None
    parent_eligibility: Optional[str] = None
    budget_reduction_10yr: Optional[str] = None
    urls: List[str] = Field(default_factory=list)


class MedicaidChanges(BaseModel):
    spending_reduction: Optional[str] = None
    coverage_impact_uninsured: Optional[str] = None
    matching_fund_elimination: Optional[str] = None
    urls: List[str] = Field(default_factory=list)


class TaxModifications(BaseModel):
    bonus_depreciation: Optional[str] = None
    child_tax_credit_increase: Optional[str] = None
    salt_deduction_limit: Optional[str] = None
    salt_deduction_phaseout: Optional[str] = None
    senior_additional_deduction: Optional[str] = None
    tips_overtime_treatment: Optional[str] = None
    urls: List[str] = Field(default_factory=list)


class BudgetImpact(BaseModel):
    fiscal_direction: Optional[str] = None
    tcja_extension: Optional[str] = None
    urls: List[str] = Field(default_factory=list)


class OBBBExtraction(BaseModel):
    bill_identification: Optional[BillIdentification] = None
    snap_work_requirements: Optional[SNAPRequirements] = None
    medicaid_changes: Optional[MedicaidChanges] = None
    tax_modifications: Optional[TaxModifications] = None
    budget_impact: Optional[BudgetImpact] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_obbb() -> str:
    return """
Extract the following structured information exactly as stated in the provided answer about the One Big Beautiful Bill Act of 2025 (do not infer, do not normalize beyond minor whitespace and punctuation). If a field is missing, set it to null. Also extract at least one authoritative URL per major category if present in the answer text.

Return a JSON with this structure:

{
  "bill_identification": {
    "house_bill_number": string or null,
    "public_law_number": string or null,
    "signing_date": string or null,
    "urls": [list of URLs explicitly present in the answer for this category]
  },
  "snap_work_requirements": {
    "age_range": string or null,
    "effective_date": string or null,
    "monthly_work_hours": string or null,
    "parent_eligibility": string or null,
    "budget_reduction_10yr": string or null,
    "urls": [category URLs]
  },
  "medicaid_changes": {
    "spending_reduction": string or null,
    "coverage_impact_uninsured": string or null,
    "matching_fund_elimination": string or null,
    "urls": [category URLs]
  },
  "tax_modifications": {
    "bonus_depreciation": string or null,
    "child_tax_credit_increase": string or null,
    "salt_deduction_limit": string or null,
    "salt_deduction_phaseout": string or null,
    "senior_additional_deduction": string or null,
    "tips_overtime_treatment": string or null,
    "urls": [category URLs]
  },
  "budget_impact": {
    "fiscal_direction": string or null,
    "tcja_extension": string or null,
    "urls": [category URLs]
  }
}

URL extraction rules:
- Only extract URLs explicitly present in the answer. If none are given for a category, return an empty list.
- Accept URLs shown as plain text or within markdown links. Extract the actual URL targets.
- Do not deduplicate or alter URLs; include as-is (prepend http:// if missing a protocol).
    """


# --------------------------------------------------------------------------- #
# Helper                                                                      #
# --------------------------------------------------------------------------- #
def _safe_urls(urls: Optional[List[str]]) -> List[str]:
    return [u for u in (urls or []) if isinstance(u, str) and u.strip()]


# --------------------------------------------------------------------------- #
# Verification builders                                                       #
# --------------------------------------------------------------------------- #
async def verify_bill_identification(evaluator: Evaluator, parent, bill: Optional[BillIdentification]) -> None:
    cat = evaluator.add_parallel(
        id="Bill_Identification",
        desc="Accurate identification of the bill's official numbers and signing date",
        parent=parent,
        critical=False
    )
    urls = _safe_urls(bill.urls if bill else None)

    evaluator.add_custom_node(
        result=len(urls) > 0,
        id="Bill_ID_URL_Reference",
        desc="At least one authoritative URL reference supporting bill identification information (e.g., from Congress.gov, IRS, or official government sources)",
        parent=cat,
        critical=True
    )

    # House Bill Number
    n1 = evaluator.add_leaf(
        id="House_Bill_Number",
        desc="Correctly identifies the House bill number as H.R. 1 in the 119th Congress",
        parent=cat,
        critical=False
    )
    claim1 = f'The House bill number for "{ACT_NAME}" is "{(bill.house_bill_number if bill else None)}".'
    await evaluator.verify(
        claim=claim1,
        node=n1,
        sources=urls,
        additional_instruction="Verify that the cited page explicitly ties this bill number to the One Big Beautiful Bill Act of 2025. Allow minor formatting differences (e.g., H.R. 1 vs HR 1)."
    )

    # Public Law Number
    n2 = evaluator.add_leaf(
        id="Public_Law_Number",
        desc="Correctly identifies the Public Law number as Public Law 119-21",
        parent=cat,
        critical=False
    )
    claim2 = f'The Public Law number for "{ACT_NAME}" is "{(bill.public_law_number if bill else None)}".'
    await evaluator.verify(
        claim=claim2,
        node=n2,
        sources=urls,
        additional_instruction="Confirm that the page shows the Public Law number associated with this act. Accept minor formatting differences (e.g., Pub. L. 119-21)."
    )

    # Signing Date
    n3 = evaluator.add_leaf(
        id="Signing_Date",
        desc="Correctly identifies the signing date as July 4, 2025",
        parent=cat,
        critical=False
    )
    claim3 = f'"{ACT_NAME}" was signed into law on {(bill.signing_date if bill else None)}.'
    await evaluator.verify(
        claim=claim3,
        node=n3,
        sources=urls,
        additional_instruction="Accept reasonable date formatting variations (e.g., July 4, 2025 vs 2025-07-04). The page should clearly indicate the signing date."
    )


async def verify_snap_requirements(evaluator: Evaluator, parent, snap: Optional[SNAPRequirements]) -> None:
    cat = evaluator.add_parallel(
        id="SNAP_Work_Requirements",
        desc="Complete documentation of SNAP work requirement changes mandated by the act",
        parent=parent,
        critical=False
    )
    urls = _safe_urls(snap.urls if snap else None)

    evaluator.add_custom_node(
        result=len(urls) > 0,
        id="SNAP_URL_Reference",
        desc="At least one authoritative URL reference supporting SNAP work requirement information (e.g., from USDA FNS, state government sites, or policy analysis organizations)",
        parent=cat,
        critical=True
    )

    # Age Range
    n1 = evaluator.add_leaf(
        id="Age_Range",
        desc="Correctly identifies the new age range as 18-64 years old (expanded from previous 18-54)",
        parent=cat,
        critical=False
    )
    claim1 = f'The act sets the SNAP work-requirement age range to "{(snap.age_range if snap else None)}".'
    await evaluator.verify(
        claim=claim1,
        node=n1,
        sources=urls,
        additional_instruction="Treat ranges like '18–64', '18 to 64', or 'ages 18-64' as equivalent."
    )

    # Effective Date
    n2 = evaluator.add_leaf(
        id="Effective_Date",
        desc="Correctly identifies the effective date as November 1, 2025",
        parent=cat,
        critical=False
    )
    claim2 = f'The SNAP work-requirement changes took effect on "{(snap.effective_date if snap else None)}".'
    await evaluator.verify(
        claim=claim2,
        node=n2,
        sources=urls,
        additional_instruction="Confirm the effective date on the cited page. Accept common date format variations."
    )

    # Monthly Work Hours
    n3 = evaluator.add_leaf(
        id="Monthly_Work_Hours",
        desc="Correctly identifies the required monthly work hours as 80 hours per month (approximately 20 hours per week)",
        parent=cat,
        critical=False
    )
    claim3 = f'The SNAP work requirement under the act is "{(snap.monthly_work_hours if snap else None)}" per month (approximately 20 hours per week).'
    await evaluator.verify(
        claim=claim3,
        node=n3,
        sources=urls,
        additional_instruction="Consider '80 hours per month' equivalent to about '20 hours per week'."
    )

    # Parent Eligibility
    n4 = evaluator.add_leaf(
        id="Parent_Eligibility",
        desc="Correctly identifies that parents are only exempt if they have children under 14 years old (previously under 18)",
        parent=cat,
        critical=False
    )
    claim4 = f'Parent eligibility for exemptions is described as: "{(snap.parent_eligibility if snap else None)}".'
    await evaluator.verify(
        claim=claim4,
        node=n4,
        sources=urls,
        additional_instruction="Verify that the page supports the policy about parental exemptions, especially the child age threshold."
    )

    # SNAP Budget Reduction
    n5 = evaluator.add_leaf(
        id="SNAP_Budget_Reduction",
        desc="Correctly identifies the SNAP budget reduction as $186 billion over 10 years",
        parent=cat,
        critical=False
    )
    claim5 = f'The act reduces the SNAP budget by "{(snap.budget_reduction_10yr if snap else None)}" over 10 years.'
    await evaluator.verify(
        claim=claim5,
        node=n5,
        sources=urls,
        additional_instruction="If multiple estimates are present, prefer official/primary estimates cited on the page."
    )


async def verify_medicaid_changes(evaluator: Evaluator, parent, med: Optional[MedicaidChanges]) -> None:
    cat = evaluator.add_parallel(
        id="Medicaid_Changes",
        desc="Accurate documentation of Medicaid program modifications",
        parent=parent,
        critical=False
    )
    urls = _safe_urls(med.urls if med else None)

    evaluator.add_custom_node(
        result=len(urls) > 0,
        id="Medicaid_URL_Reference",
        desc="At least one authoritative URL reference supporting Medicaid changes information (e.g., from KFF, CBO, HHS, or health policy organizations)",
        parent=cat,
        critical=True
    )

    # Spending Reduction
    n1 = evaluator.add_leaf(
        id="Spending_Reduction",
        desc="Correctly identifies federal Medicaid spending reduction as over $1 trillion",
        parent=cat,
        critical=False
    )
    claim1 = f'The federal Medicaid spending reduction attributed to the act is "{(med.spending_reduction if med else None)}".'
    await evaluator.verify(
        claim=claim1,
        node=n1,
        sources=urls,
        additional_instruction="Confirm the cited spending reduction magnitude (e.g., billions vs trillions) per the page."
    )

    # Coverage Impact
    n2 = evaluator.add_leaf(
        id="Coverage_Impact",
        desc="Correctly identifies the estimated increase in uninsured Americans as 10 million",
        parent=cat,
        critical=False
    )
    claim2 = f'The act is estimated to increase the number of uninsured Americans by "{(med.coverage_impact_uninsured if med else None)}".'
    await evaluator.verify(
        claim=claim2,
        node=n2,
        sources=urls,
        additional_instruction="Verify the estimate on the cited page; accept rounded figures (e.g., 10 million vs 10,000,000)."
    )

    # Matching Fund Elimination
    n3 = evaluator.add_leaf(
        id="Matching_Fund_Elimination",
        desc="Correctly identifies the elimination of the 5% federal matching fund increase from the American Rescue Plan Act",
        parent=cat,
        critical=False
    )
    claim3 = f'Matching fund policy change under the act: "{(med.matching_fund_elimination if med else None)}".'
    await evaluator.verify(
        claim=claim3,
        node=n3,
        sources=urls,
        additional_instruction="Confirm whether the act eliminates the 5% enhanced federal matching funds from ARPA (American Rescue Plan Act)."
    )


async def verify_tax_modifications(evaluator: Evaluator, parent, tax: Optional[TaxModifications]) -> None:
    cat = evaluator.add_parallel(
        id="Tax_Modifications",
        desc="Complete documentation of tax code changes in the legislation",
        parent=parent,
        critical=False
    )
    urls = _safe_urls(tax.urls if tax else None)

    evaluator.add_custom_node(
        result=len(urls) > 0,
        id="Tax_URL_Reference",
        desc="At least one authoritative URL reference supporting tax modification information (e.g., from IRS, tax policy foundations, or Congressional Research Service)",
        parent=cat,
        critical=True
    )

    # Bonus Depreciation
    n1 = evaluator.add_leaf(
        id="Bonus_Depreciation",
        desc="Correctly identifies bonus depreciation as 100% for qualifying business property acquired after January 19, 2025",
        parent=cat,
        critical=False
    )
    claim1 = f'Bonus depreciation under the act is "{(tax.bonus_depreciation if tax else None)}" for qualifying business property (notably for acquisitions after Jan 19, 2025, if applicable).'
    await evaluator.verify(
        claim=claim1,
        node=n1,
        sources=urls,
        additional_instruction="Confirm the percentage and any acquisition-date conditions described on the page."
    )

    # Child Tax Credit Increase
    n2 = evaluator.add_leaf(
        id="Child_Tax_Credit",
        desc="Correctly identifies the permanent child tax credit increase as $200",
        parent=cat,
        critical=False
    )
    claim2 = f'The permanent child tax credit increase under the act is "{(tax.child_tax_credit_increase if tax else None)}".'
    await evaluator.verify(
        claim=claim2,
        node=n2,
        sources=urls,
        additional_instruction="Verify the amount and permanence of the CTC change per the cited page."
    )

    # SALT Deduction Limit
    n3 = evaluator.add_leaf(
        id="SALT_Deduction_Limit",
        desc="Correctly identifies the new SALT deduction limit as $40,000 (increased from $10,000)",
        parent=cat,
        critical=False
    )
    claim3 = f'The SALT deduction limit under the act is "{(tax.salt_deduction_limit if tax else None)}".'
    await evaluator.verify(
        claim=claim3,
        node=n3,
        sources=urls,
        additional_instruction="Confirm the new SALT cap on the cited page; accept reasonable formatting variations (e.g., with commas)."
    )

    # SALT Deduction Phaseout
    n3b = evaluator.add_leaf(
        id="SALT_Deduction_Phaseout",
        desc="Correctly identifies that the SALT deduction phases out for taxpayers earning over $500,000",
        parent=cat,
        critical=False
    )
    claim3b = f'The SALT deduction phaseout provision is described as: "{(tax.salt_deduction_phaseout if tax else None)}".'
    await evaluator.verify(
        claim=claim3b,
        node=n3b,
        sources=urls,
        additional_instruction="Verify that the page describes the SALT deduction phaseout threshold and mechanics."
    )

    # Senior Additional Deduction
    n4 = evaluator.add_leaf(
        id="Senior_Deduction",
        desc="Correctly identifies the additional deduction for seniors aged 65+ as $6,000 for tax years 2025-2028",
        parent=cat,
        critical=False
    )
    claim4 = f'The additional standard deduction for seniors aged 65+ is "{(tax.senior_additional_deduction if tax else None)}".'
    await evaluator.verify(
        claim=claim4,
        node=n4,
        sources=urls,
        additional_instruction="Confirm the amount and applicable tax years as described on the page."
    )

    # Tips and Overtime Treatment
    n5 = evaluator.add_leaf(
        id="Tips_Overtime_Treatment",
        desc="Correctly identifies that the bill eliminates federal taxes on tips and overtime pay",
        parent=cat,
        critical=False
    )
    claim5 = f'The act\'s tax treatment of tips and overtime pay is described as: "{(tax.tips_overtime_treatment if tax else None)}".'
    await evaluator.verify(
        claim=claim5,
        node=n5,
        sources=urls,
        additional_instruction="Confirm whether the act eliminates, reduces, or otherwise changes federal tax on tips and overtime; match the answer's statement."
    )


async def verify_budget_impact(evaluator: Evaluator, parent, budget: Optional[BudgetImpact]) -> None:
    cat = evaluator.add_parallel(
        id="Budget_Impact",
        desc="Accurate summary of the overall fiscal impact and TCJA extension status",
        parent=parent,
        critical=False
    )
    urls = _safe_urls(budget.urls if budget else None)

    evaluator.add_custom_node(
        result=len(urls) > 0,
        id="Budget_URL_Reference",
        desc="At least one authoritative URL reference supporting budget impact information (e.g., from CBO, budget policy organizations, or Congressional sources)",
        parent=cat,
        critical=True
    )

    # Fiscal Direction
    n1 = evaluator.add_leaf(
        id="Fiscal_Direction",
        desc="Correctly describes the overall fiscal direction as reducing taxes and modifying federal spending across multiple programs",
        parent=cat,
        critical=False
    )
    claim1 = f'The overall fiscal direction of "{ACT_NAME}" is: "{(budget.fiscal_direction if budget else None)}".'
    await evaluator.verify(
        claim=claim1,
        node=n1,
        sources=urls,
        additional_instruction="Check whether the page characterizes the bill as reducing taxes and modifying spending, consistent with the answer."
    )

    # TCJA Extension
    n2 = evaluator.add_leaf(
        id="TCJA_Extension",
        desc="Correctly identifies that the bill extends and expands TCJA individual tax provisions",
        parent=cat,
        critical=False
    )
    claim2 = f'The bill extends and/or expands the TCJA individual tax provisions: "{(budget.tcja_extension if budget else None)}".'
    await evaluator.verify(
        claim=claim2,
        node=n2,
        sources=urls,
        additional_instruction="Verify that the cited page indicates extension or expansion of TCJA individual provisions."
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
    # Initialize evaluator (root is non-critical to allow partial scoring per category)
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

    # Optional: Add high-level node reflecting the rubric's top-level label (non-critical to permit partials)
    top = evaluator.add_parallel(
        id="Comprehensive_OBBB_Documentation",
        desc="Complete and accurate documentation of the One Big Beautiful Bill Act of 2025 across all required categories",
        parent=root,
        critical=False
    )

    # Extract structured information from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_obbb(),
        template_class=OBBBExtraction,
        extraction_name="obbb_extraction"
    )

    # Add rubric expectations as GT info (for transparency; not used for gating)
    evaluator.add_ground_truth({
        "expected_fields": {
            "Bill_Identification": ["house_bill_number", "public_law_number", "signing_date", "urls"],
            "SNAP_Work_Requirements": ["age_range", "effective_date", "monthly_work_hours", "parent_eligibility", "budget_reduction_10yr", "urls"],
            "Medicaid_Changes": ["spending_reduction", "coverage_impact_uninsured", "matching_fund_elimination", "urls"],
            "Tax_Modifications": ["bonus_depreciation", "child_tax_credit_increase", "salt_deduction_limit", "salt_deduction_phaseout", "senior_additional_deduction", "tips_overtime_treatment", "urls"],
            "Budget_Impact": ["fiscal_direction", "tcja_extension", "urls"]
        }
    })

    # Build verification subtrees per category
    await verify_bill_identification(evaluator, top, extracted.bill_identification if extracted else None)
    await verify_snap_requirements(evaluator, top, extracted.snap_work_requirements if extracted else None)
    await verify_medicaid_changes(evaluator, top, extracted.medicaid_changes if extracted else None)
    await verify_tax_modifications(evaluator, top, extracted.tax_modifications if extracted else None)
    await verify_budget_impact(evaluator, top, extracted.budget_impact if extracted else None)

    # Return structured summary
    return evaluator.get_summary()