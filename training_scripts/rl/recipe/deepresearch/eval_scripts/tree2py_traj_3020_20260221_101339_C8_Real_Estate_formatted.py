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
TASK_ID = "mortgage_2026_comparison"
TASK_DESCRIPTION = """
I am a first-time homebuyer researching mortgage options for purchasing a property in the United States in 2026. I want to understand the four major types of mortgage loans available: FHA loans, Conventional loans, VA loans, and USDA loans.

For each of these four loan types, please provide the following information based on current 2026 requirements:

1. Minimum Credit Score Requirement: What is the minimum credit score needed to qualify?

2. Down Payment Requirement: What is the minimum down payment percentage or amount required?

3. Debt-to-Income (DTI) Ratio Maximum: What is the maximum debt-to-income ratio allowed?

4. Mortgage Insurance or Fee Requirements: What mortgage insurance premiums or loan fees are required (including upfront and ongoing costs)?

5. Special Eligibility Requirements: Are there any special eligibility criteria (such as for VA loans requiring veteran status, or USDA loans requiring property to be in designated rural areas, or income limits)?

For each loan type, please provide a reference URL from an official or reputable source (such as government websites, major lenders, or established financial institutions) that supports the information you provide.

Please organize your response clearly by loan type, ensuring all five informational categories are addressed for FHA, Conventional, VA, and USDA loans.
"""

# --------------------------------------------------------------------------- #
# Data models for structured extraction                                       #
# --------------------------------------------------------------------------- #
class FHADetails(BaseModel):
    credit_tier_580_plus_down_payment_percent: Optional[str] = None
    credit_tier_500_579_down_payment_percent: Optional[str] = None
    dti_max: Optional[str] = None
    mip_upfront_percent_or_text: Optional[str] = None
    mip_annual_requirement_text: Optional[str] = None
    special_primary_residence_text: Optional[str] = None
    reference_urls: List[str] = Field(default_factory=list)


class ConventionalDetails(BaseModel):
    credit_min: Optional[str] = None
    down_payment_min_percent: Optional[str] = None
    dti_max_primary_value: Optional[str] = None
    dti_max_with_comp_factors: Optional[str] = None
    pmi_required_under_20_text: Optional[str] = None
    pmi_ongoing_text: Optional[str] = None
    pmi_cancellation_80_ltv_text: Optional[str] = None
    special_eligibility_text: Optional[str] = None
    reference_urls: List[str] = Field(default_factory=list)


class VADetails(BaseModel):
    credit_min_lender_guidance: Optional[str] = None
    down_payment_requirement_text: Optional[str] = None
    dti_guideline_text: Optional[str] = None
    funding_fee_text: Optional[str] = None
    mortgage_insurance_monthly_text: Optional[str] = None
    eligibility_service_text: Optional[str] = None
    eligibility_primary_residence_text: Optional[str] = None
    reference_urls: List[str] = Field(default_factory=list)


class USDADetails(BaseModel):
    credit_score_requirement_text: Optional[str] = None
    down_payment_requirement_text: Optional[str] = None
    dti_max_text: Optional[str] = None
    guarantee_fee_upfront_text: Optional[str] = None
    guarantee_fee_annual_text: Optional[str] = None
    eligibility_rural_area_text: Optional[str] = None
    # Explicitly request AMI ranges to enforce rubric requirement
    usda_low_income_ami_range: Optional[str] = None   # e.g., "50–80% AMI"
    usda_moderate_income_ami_limit: Optional[str] = None  # e.g., "115% AMI"
    eligibility_primary_residence_text: Optional[str] = None
    reference_urls: List[str] = Field(default_factory=list)


class AllLoansExtraction(BaseModel):
    fha: Optional[FHADetails] = None
    conventional: Optional[ConventionalDetails] = None
    va: Optional[VADetails] = None
    usda: Optional[USDADetails] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_all_loans() -> str:
    return """
    Extract the mortgage information for the four loan types (FHA, Conventional, VA, USDA) as explicitly stated in the answer. For each loan type, fill the following fields using exact values or phrases from the answer. If a field is not stated, return null. Extract any and all reference URLs the answer cites for that loan type.

    For FHA (object name 'fha'):
    - credit_tier_580_plus_down_payment_percent: The minimum down payment percent for borrowers with credit scores 580 or higher (e.g., "3.5%").
    - credit_tier_500_579_down_payment_percent: The minimum down payment percent for borrowers with credit scores 500–579 (e.g., "10%").
    - dti_max: The maximum debt-to-income ratio stated for FHA (e.g., "43%").
    - mip_upfront_percent_or_text: The stated upfront MIP percent or description (e.g., "1.75% upfront").
    - mip_annual_requirement_text: The stated ongoing annual MIP requirement description (e.g., "annual MIP required").
    - special_primary_residence_text: The text indicating FHA requires the property be the borrower's primary residence (quote or phrase).
    - reference_urls: List all URLs cited in the answer for FHA (official or reputable sources).

    For Conventional (object name 'conventional'):
    - credit_min: The minimum credit score stated for conventional loans (e.g., "620").
    - down_payment_min_percent: The minimum down payment option stated (e.g., "3%").
    - dti_max_primary_value: The primary max DTI stated (e.g., "45%").
    - dti_max_with_comp_factors: The higher DTI allowed with compensating factors (e.g., "50%").
    - pmi_required_under_20_text: Phrase stating PMI is required when down payment < 20%.
    - pmi_ongoing_text: Phrase stating PMI is an ongoing premium/cost.
    - pmi_cancellation_80_ltv_text: Phrase stating PMI can be canceled around 80% LTV.
    - special_eligibility_text: Any special eligibility criteria text; or explicitly "none beyond standard underwriting" if stated.
    - reference_urls: List all URLs cited for Conventional.

    For VA (object name 'va'):
    - credit_min_lender_guidance: The lender minimum credit score guidance stated (e.g., "620").
    - down_payment_requirement_text: Phrase stating VA requires no down payment.
    - dti_guideline_text: The stated DTI rule or guideline (e.g., "no strict cap; lenders use ~41% guideline" or similar).
    - funding_fee_text: Phrase indicating VA funding fee is required (varies).
    - mortgage_insurance_monthly_text: Phrase indicating there is no monthly mortgage insurance (or equivalent).
    - eligibility_service_text: Phrase describing eligible veterans, service members, qualifying spouses.
    - eligibility_primary_residence_text: Phrase indicating property must be primary residence.
    - reference_urls: List all URLs cited for VA.

    For USDA (object name 'usda'):
    - credit_score_requirement_text: The stated minimum credit score requirement (program or typical lender/AUS threshold).
    - down_payment_requirement_text: Phrase stating USDA requires no down payment.
    - dti_max_text: The stated maximum DTI guideline for USDA.
    - guarantee_fee_upfront_text: Phrase indicating an upfront guarantee fee exists.
    - guarantee_fee_annual_text: Phrase indicating an annual fee exists.
    - eligibility_rural_area_text: Phrase indicating property must be in USDA-designated rural area.
    - usda_low_income_ami_range: The stated AMI range for low-income programs (e.g., "50–80% AMI"); if not stated, null.
    - usda_moderate_income_ami_limit: The stated AMI limit for moderate-income programs (e.g., "115% AMI"); if not stated, null.
    - eligibility_primary_residence_text: Phrase indicating property must be a primary residence.
    - reference_urls: List all URLs cited for USDA.

    Always extract the actual URLs mentioned. If any field is not present in the answer text, return null or an empty list as appropriate.
    """


# --------------------------------------------------------------------------- #
# Verification helpers                                                        #
# --------------------------------------------------------------------------- #
def _safe_text(val: Optional[str], placeholder: str = "<missing>") -> str:
    return val.strip() if isinstance(val, str) and val.strip() else placeholder


# --------------------------------------------------------------------------- #
# Verification functions per loan type                                        #
# --------------------------------------------------------------------------- #
async def verify_fha(evaluator: Evaluator, root_node, fha: Optional[FHADetails]) -> None:
    loan_node = evaluator.add_parallel(
        id="FHA_loan_information",
        desc="Information for FHA loans",
        parent=root_node,
        critical=False
    )

    urls = fha.reference_urls if (fha and fha.reference_urls) else []

    # Credit score and down payment tier mapping
    leaf1 = evaluator.add_leaf(
        id="FHA_credit_score_and_down_payment_requirements",
        desc="States FHA minimum credit score and minimum down payment requirements with correct tier association: 580+ -> 3.5% down; 500–579 -> 10% down",
        parent=loan_node,
        critical=True
    )
    claim1 = (
        f"For FHA loans, borrowers with credit scores of 580 or higher can qualify for a minimum down payment of "
        f"{_safe_text(fha.credit_tier_580_plus_down_payment_percent if fha else None)}, and borrowers with scores "
        f"between 500 and 579 must make at least a {_safe_text(fha.credit_tier_500_579_down_payment_percent if fha else None)} down payment."
    )
    await evaluator.verify(
        claim=claim1,
        node=leaf1,
        sources=urls,
        additional_instruction="Verify the FHA down payment rules by credit score tiers (580+ → 3.5%; 500–579 → 10%). If the claim values differ or are missing, mark unsupported."
    )

    # DTI maximum (expect 43%)
    leaf2 = evaluator.add_leaf(
        id="FHA_DTI_ratio_maximum",
        desc="States FHA maximum DTI ratio (43%)",
        parent=loan_node,
        critical=True
    )
    claim2 = f"The maximum debt-to-income (DTI) ratio allowed for FHA underwriting is {_safe_text(fha.dti_max if fha else None)}."
    await evaluator.verify(
        claim=claim2,
        node=leaf2,
        sources=urls,
        additional_instruction="Confirm HUD/FHA guidance that the standard max DTI is 43% unless compensating factors apply."
    )

    # MIP upfront and annual
    leaf3 = evaluator.add_leaf(
        id="FHA_mortgage_insurance_fees_upfront_and_ongoing",
        desc="States FHA mortgage insurance requirements including upfront and ongoing components (1.75% upfront MIP and an annual premium requirement)",
        parent=loan_node,
        critical=True
    )
    claim3 = (
        f"FHA loans require an upfront mortgage insurance premium of "
        f"{_safe_text(fha.mip_upfront_percent_or_text if fha else None)}, and they also require an ongoing annual mortgage insurance premium "
        f"({_safe_text(fha.mip_annual_requirement_text if fha else None)})."
    )
    await evaluator.verify(
        claim=claim3,
        node=leaf3,
        sources=urls,
        additional_instruction="Verify that FHA requires 1.75% upfront MIP and an annual MIP (amount varies by factors)."
    )

    # Special eligibility - primary residence
    leaf4 = evaluator.add_leaf(
        id="FHA_special_eligibility_primary_residence",
        desc="States FHA special eligibility requirement that the property must be the borrower's primary residence",
        parent=loan_node,
        critical=True
    )
    claim4 = (
        f"FHA financing requires the property to be the borrower's primary residence "
        f"({_safe_text(fha.special_primary_residence_text if fha else None)})."
    )
    await evaluator.verify(
        claim=claim4,
        node=leaf4,
        sources=urls,
        additional_instruction="Confirm HUD occupancy requirement: FHA loans are intended for primary residences."
    )

    # Reference URL existence
    evaluator.add_custom_node(
        result=bool(urls),
        id="FHA_reference_url",
        desc="Provides at least one reference URL from an official or reputable source supporting the FHA information",
        parent=loan_node,
        critical=True
    )


async def verify_conventional(evaluator: Evaluator, root_node, conv: Optional[ConventionalDetails]) -> None:
    loan_node = evaluator.add_parallel(
        id="Conventional_loan_information",
        desc="Information for Conventional loans",
        parent=root_node,
        critical=False
    )

    urls = conv.reference_urls if (conv and conv.reference_urls) else []

    # Credit score minimum (620)
    leaf1 = evaluator.add_leaf(
        id="Conventional_credit_score_minimum",
        desc="States the typical minimum credit score for Conventional loans (620)",
        parent=loan_node,
        critical=True
    )
    claim1 = f"Conventional loans typically require a minimum credit score of {_safe_text(conv.credit_min if conv else None)}."
    await evaluator.verify(
        claim=claim1,
        node=leaf1,
        sources=urls,
        additional_instruction="Confirm reputable guidance indicating a 620 minimum credit score for conventional loans."
    )

    # Down payment minimum (as low as 3%)
    leaf2 = evaluator.add_leaf(
        id="Conventional_down_payment_minimum",
        desc="States the minimum down payment option for Conventional loans (as low as 3%)",
        parent=loan_node,
        critical=True
    )
    claim2 = f"Conventional loans allow a minimum down payment as low as {_safe_text(conv.down_payment_min_percent if conv else None)}."
    await evaluator.verify(
        claim=claim2,
        node=leaf2,
        sources=urls,
        additional_instruction="Confirm that some conventional programs permit 3% down payments."
    )

    # DTI guidance (45%, up to 50% with compensating factors)
    leaf3 = evaluator.add_leaf(
        id="Conventional_DTI_ratio_maximum",
        desc="States Conventional maximum DTI guidance (45%, and up to 50% with compensating factors)",
        parent=loan_node,
        critical=True
    )
    claim3 = (
        f"Conventional loans generally cap DTI at {_safe_text(conv.dti_max_primary_value if conv else None)}, and with strong compensating factors may allow up to "
        f"{_safe_text(conv.dti_max_with_comp_factors if conv else None)}."
    )
    await evaluator.verify(
        claim=claim3,
        node=leaf3,
        sources=urls,
        additional_instruction="Verify mainstream lender/Fannie/Freddie guidance: ~45% DTI typical; up to 50% with compensating factors."
    )

    # PMI rules combined
    leaf4 = evaluator.add_leaf(
        id="Conventional_mortgage_insurance_costs_and_rules",
        desc="Addresses Conventional mortgage insurance requirements, including (a) PMI required when down payment is less than 20%, (b) PMI is an ongoing premium/cost, and (c) PMI removal/cancellation guidance consistent with the constraints (removable when LTV reaches 80%)",
        parent=loan_node,
        critical=True
    )
    claim4 = (
        f"For Conventional loans, private mortgage insurance (PMI) is required when the down payment is less than 20% "
        f"({_safe_text(conv.pmi_required_under_20_text if conv else None)}), PMI is an ongoing premium/cost "
        f"({_safe_text(conv.pmi_ongoing_text if conv else None)}), and it can be canceled when the loan-to-value reaches about 80% "
        f"({_safe_text(conv.pmi_cancellation_80_ltv_text if conv else None)})."
    )
    await evaluator.verify(
        claim=claim4,
        node=leaf4,
        sources=urls,
        additional_instruction="Confirm conventional PMI requirements: required <20% down, ongoing monthly cost, cancellable near 80% LTV."
    )

    # Special eligibility requirements
    leaf5 = evaluator.add_leaf(
        id="Conventional_special_eligibility_requirements",
        desc="Addresses the 'special eligibility requirements' category for Conventional loans (states any special eligibility criteria or explicitly notes none beyond standard underwriting)",
        parent=loan_node,
        critical=True
    )
    claim5 = (
        f"Conventional loans do not have special program eligibility such as veteran status or rural-location requirements; qualification follows standard underwriting "
        f"({_safe_text(conv.special_eligibility_text if conv else None)})."
    )
    await evaluator.verify(
        claim=claim5,
        node=leaf5,
        sources=urls,
        additional_instruction="Confirm that conventional loans are standard underwriting products without special eligibility like VA/USDA."
    )

    # Reference URL existence
    evaluator.add_custom_node(
        result=bool(urls),
        id="Conventional_reference_url",
        desc="Provides at least one reference URL from an official or reputable source supporting the Conventional information",
        parent=loan_node,
        critical=True
    )


async def verify_va(evaluator: Evaluator, root_node, va: Optional[VADetails]) -> None:
    loan_node = evaluator.add_parallel(
        id="VA_loan_information",
        desc="Information for VA loans",
        parent=root_node,
        critical=False
    )

    urls = va.reference_urls if (va and va.reference_urls) else []

    # Credit score minimum (lender guidance, e.g., 620)
    leaf1 = evaluator.add_leaf(
        id="VA_credit_score_minimum",
        desc="States typical lender minimum credit score guidance for VA loans (620 as lender requirement per constraints)",
        parent=loan_node,
        critical=True
    )
    claim1 = (
        f"Although the VA program itself does not set a minimum credit score, many VA lenders require a score around "
        f"{_safe_text(va.credit_min_lender_guidance if va else None)}."
    )
    await evaluator.verify(
        claim=claim1,
        node=leaf1,
        sources=urls,
        additional_instruction="Confirm lender overlays (e.g., ~620 minimum) referenced by reputable sources; VA does not set an official minimum."
    )

    # Down payment requirement (no down payment)
    leaf2 = evaluator.add_leaf(
        id="VA_down_payment_requirement",
        desc="States VA loans require no down payment",
        parent=loan_node,
        critical=True
    )
    claim2 = f"VA loans do not require any down payment ({_safe_text(va.down_payment_requirement_text if va else None)})."
    await evaluator.verify(
        claim=claim2,
        node=leaf2,
        sources=urls,
        additional_instruction="Confirm that VA-guaranteed loans commonly require $0 down for eligible borrowers."
    )

    # DTI guideline or cap
    leaf3 = evaluator.add_leaf(
        id="VA_DTI_ratio_maximum_or_guideline",
        desc="Addresses the maximum DTI ratio allowed/guideline for VA loans per current (2026) guidance, including any caveats if VA uses guidelines rather than a strict cap, supported by an official/reputable source",
        parent=loan_node,
        critical=True
    )
    claim3 = f"VA underwriting uses {_safe_text(va.dti_guideline_text if va else None)} for DTI evaluation."
    await evaluator.verify(
        claim=claim3,
        node=leaf3,
        sources=urls,
        additional_instruction="Confirm that VA emphasizes residual income and does not enforce a strict DTI cap; lenders may apply guideline thresholds (e.g., ~41%)."
    )

    # Fees and insurance (funding fee; no monthly MI)
    leaf4 = evaluator.add_leaf(
        id="VA_mortgage_fees_upfront_and_ongoing",
        desc="States VA fee/insurance information including the VA funding fee requirement (varies by service type and down payment) and addresses whether there are ongoing monthly mortgage insurance costs (e.g., none) with support from an official/reputable source",
        parent=loan_node,
        critical=True
    )
    claim4 = (
        f"VA loans require a funding fee ({_safe_text(va.funding_fee_text if va else None)}), and they do not have monthly mortgage insurance "
        f"({_safe_text(va.mortgage_insurance_monthly_text if va else None)})."
    )
    await evaluator.verify(
        claim=claim4,
        node=leaf4,
        sources=urls,
        additional_instruction="Confirm VA funding fee is required (varies by factors) and there is no monthly mortgage insurance."
    )

    # Special eligibility (service + primary residence)
    leaf5 = evaluator.add_leaf(
        id="VA_special_eligibility_requirements",
        desc="States VA special eligibility requirements: eligible veterans, active duty service members, or qualifying spouses; and that the property must be used as the primary residence",
        parent=loan_node,
        critical=True
    )
    claim5 = (
        f"VA loans are available to eligible veterans, active-duty service members, or qualifying surviving spouses "
        f"({_safe_text(va.eligibility_service_text if va else None)}), and the property must be used as the borrower's primary residence "
        f"({_safe_text(va.eligibility_primary_residence_text if va else None)})."
    )
    await evaluator.verify(
        claim=claim5,
        node=leaf5,
        sources=urls,
        additional_instruction="Confirm eligibility categories (veterans/service members/spouses) and primary residence occupancy requirement."
    )

    # Reference URL existence
    evaluator.add_custom_node(
        result=bool(urls),
        id="VA_reference_url",
        desc="Provides at least one reference URL from an official or reputable source supporting the VA information",
        parent=loan_node,
        critical=True
    )


async def verify_usda(evaluator: Evaluator, root_node, usda: Optional[USDADetails]) -> None:
    loan_node = evaluator.add_parallel(
        id="USDA_loan_information",
        desc="Information for USDA loans",
        parent=root_node,
        critical=False
    )

    urls = usda.reference_urls if (usda and usda.reference_urls) else []

    # Credit score requirement
    leaf1 = evaluator.add_leaf(
        id="USDA_credit_score_requirement",
        desc="States the minimum credit score requirement for USDA loans per current (2026) guidance (may be expressed as program minimum or typical lender/automated underwriting threshold) and supports it with an official/reputable source; do not require a specific value unless provided in constraints",
        parent=loan_node,
        critical=True
    )
    claim1 = (
        f"USDA lenders or automated underwriting typically require a minimum credit score threshold of "
        f"{_safe_text(usda.credit_score_requirement_text if usda else None)}."
    )
    await evaluator.verify(
        claim=claim1,
        node=leaf1,
        sources=urls,
        additional_instruction="Confirm USDA credit score expectations (e.g., lender/AUS thresholds) per reputable sources."
    )

    # Down payment requirement (no down payment)
    leaf2 = evaluator.add_leaf(
        id="USDA_down_payment_requirement",
        desc="States USDA loans require no down payment",
        parent=loan_node,
        critical=True
    )
    claim2 = f"USDA loans do not require a down payment ({_safe_text(usda.down_payment_requirement_text if usda else None)})."
    await evaluator.verify(
        claim=claim2,
        node=leaf2,
        sources=urls,
        additional_instruction="Confirm USDA guaranteed loans commonly permit 0% down for eligible borrowers."
    )

    # DTI maximum guideline
    leaf3 = evaluator.add_leaf(
        id="USDA_DTI_ratio_maximum",
        desc="States the maximum DTI ratio allowed for USDA loans per current (2026) guidance and supports it with a reputable/official source (do not require a specific value unless given in constraints)",
        parent=loan_node,
        critical=True
    )
    claim3 = f"USDA underwriting uses a maximum DTI guideline of {_safe_text(usda.dti_max_text if usda else None)}."
    await evaluator.verify(
        claim=claim3,
        node=leaf3,
        sources=urls,
        additional_instruction="Confirm USDA DTI guideline (commonly around 41–45% in practice)."
    )

    # Guarantee fees (upfront and annual)
    leaf4 = evaluator.add_leaf(
        id="USDA_guarantee_fee_upfront_and_ongoing",
        desc="States USDA guarantee fee requirements including upfront guarantee fee and annual fee",
        parent=loan_node,
        critical=True
    )
    claim4 = (
        f"USDA loans have a guarantee fee structure with an upfront guarantee fee "
        f"({_safe_text(usda.guarantee_fee_upfront_text if usda else None)}) and an annual fee "
        f"({_safe_text(usda.guarantee_fee_annual_text if usda else None)})."
    )
    await evaluator.verify(
        claim=claim4,
        node=leaf4,
        sources=urls,
        additional_instruction="Confirm that USDA guaranteed loans charge both an upfront guarantee fee and an annual fee (rates can vary by year)."
    )

    # Special eligibility: rural area, income limits (50–80% AMI low-income; up to 115% AMI moderate), primary residence
    leaf5 = evaluator.add_leaf(
        id="USDA_special_eligibility_requirements",
        desc="States USDA special eligibility requirements: property must be in a USDA-designated rural area; household income limits consistent with constraints (low-income 50–80% AMI; moderate-income up to 115% AMI); and property must be used as the primary residence",
        parent=loan_node,
        critical=True
    )
    claim5 = (
        f"USDA loans require the property to be in a USDA-designated rural area "
        f"({_safe_text(usda.eligibility_rural_area_text if usda else None)}), household income must meet program limits "
        f"(including low-income {_safe_text(usda.usda_low_income_ami_range if usda else None)} and moderate-income up to "
        f"{_safe_text(usda.usda_moderate_income_ami_limit if usda else None)} of area median income), and the home must be a primary residence "
        f"({_safe_text(usda.eligibility_primary_residence_text if usda else None)})."
    )
    await evaluator.verify(
        claim=claim5,
        node=leaf5,
        sources=urls,
        additional_instruction="Confirm USDA eligibility: rural area designation, income limits (low-income 50–80% AMI; moderate up to 115% AMI), and primary residence requirement."
    )

    # Reference URL existence
    evaluator.add_custom_node(
        result=bool(urls),
        id="USDA_reference_url",
        desc="Provides at least one reference URL from an official or reputable source supporting the USDA information",
        parent=loan_node,
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
    """
    Evaluate an answer for the 2026 mortgage loan comparison task.
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

    # Extract structured information for all four loan types
    extraction = await evaluator.extract(
        prompt=prompt_extract_all_loans(),
        template_class=AllLoansExtraction,
        extraction_name="mortgage_loans_2026"
    )

    # Build and execute verification for each loan type
    await verify_fha(evaluator, root, extraction.fha)
    await verify_conventional(evaluator, root, extraction.conventional)
    await verify_va(evaluator, root, extraction.va)
    await verify_usda(evaluator, root, extraction.usda)

    return evaluator.get_summary()