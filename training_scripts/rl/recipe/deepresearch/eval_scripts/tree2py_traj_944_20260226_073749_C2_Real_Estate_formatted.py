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
TASK_ID = "mortgage_eligibility_2026"
TASK_DESCRIPTION = """
A homebuyer has the following profile in 2026: credit score of 625, debt-to-income (DTI) ratio of 40%, honorably discharged veteran with a valid Certificate of Eligibility (COE), intends to use the property as their primary residence, purchasing a single-family home for $400,000 in a standard-cost area. Based on current 2026 mortgage lending standards, identify all mortgage loan types (FHA, VA, and/or Conventional) for which this borrower qualifies. For each qualifying loan type, state the minimum required down payment percentage.
"""


# --------------------------------------------------------------------------- #
# Data models for extracted info                                              #
# --------------------------------------------------------------------------- #
class MortgageAnswerExtraction(BaseModel):
    # Loan type qualification flags (as stated by the answer)
    qualifies_fha: Optional[bool] = None
    qualifies_va: Optional[bool] = None
    qualifies_conventional: Optional[bool] = None

    # Down payment percentages stated in the answer (strings, e.g., "3.5%")
    fha_down_payment_pct: Optional[str] = None
    va_down_payment_pct: Optional[str] = None
    conventional_down_payment_pct: Optional[str] = None

    # Source URLs the answer cites per-loan-type (explicit URLs only)
    fha_sources: List[str] = Field(default_factory=list)
    va_sources: List[str] = Field(default_factory=list)
    conventional_sources: List[str] = Field(default_factory=list)

    # Optional general sources if the answer does not attribute per loan type
    general_sources: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_mortgage_answer() -> str:
    return """
    Extract from the answer the following, based strictly on what is explicitly stated:

    1) Loan type qualification determinations (booleans):
       - qualifies_fha: Does the answer explicitly state that the borrower qualifies for an FHA loan?
       - qualifies_va: Does the answer explicitly state that the borrower qualifies for a VA loan?
       - qualifies_conventional: Does the answer explicitly state that the borrower qualifies for a Conventional loan?
       If the answer does not clearly state yes/no for a given loan type, set the corresponding field to null.

    2) Down payment percentages (strings):
       - fha_down_payment_pct: The minimum required down payment percentage stated in the answer for FHA (if any). Use the exact string from the answer, e.g., "3.5%".
       - va_down_payment_pct: The minimum required down payment percentage stated in the answer for VA (if any), e.g., "0%".
       - conventional_down_payment_pct: The minimum required down payment percentage stated in the answer for Conventional (if any), e.g., "3%".
       If the answer does not state a minimum down payment for a given loan type, set the field to null.

    3) Source URLs:
       - fha_sources: URLs explicitly cited to support FHA eligibility or down payment.
       - va_sources: URLs explicitly cited to support VA eligibility or down payment.
       - conventional_sources: URLs explicitly cited to support Conventional eligibility or down payment.
       - general_sources: Any other URLs cited that discuss mortgage rules or serve as general references (not specific to a single loan type).

    IMPORTANT:
    - Only extract actual URLs that appear in the answer text. Do not invent or infer URLs.
    - For booleans, return true/false only if the answer clearly asserts it. Otherwise return null.
    - Preserve down payment percentages exactly as written (e.g., include the % sign if present).
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
BORROWER_PROFILE_TEXT = (
    "Borrower profile: credit score 625; DTI 40%; honorably discharged veteran with a valid COE; "
    "primary residence; single-family home; purchase price $400,000; standard-cost area; year 2026."
)


def merge_sources(*lists: List[str]) -> List[str]:
    """Merge and deduplicate multiple URL lists while preserving order."""
    seen = set()
    merged: List[str] = []
    for lst in lists:
        for url in lst:
            if url and isinstance(url, str) and url not in seen:
                seen.add(url)
                merged.append(url)
    return merged


def boolean_to_text(value: Optional[bool]) -> str:
    if value is True:
        return "does"
    if value is False:
        return "does not"
    return "does not (not stated in the answer)"


# --------------------------------------------------------------------------- #
# Verification logic                                                          #
# --------------------------------------------------------------------------- #
async def build_and_verify_tree(evaluator: Evaluator, extraction: MortgageAnswerExtraction) -> None:
    """
    Build the verification tree per rubric and run verifications.
    JSON rubric (adapted to framework constraints):
      - Root: sequential, critical (children must also be critical due to framework rule)
          1) LoanTypeIdentification: parallel, critical
             - FHAEligibilityCheck (leaf, critical)
             - VAEligibilityCheck (leaf, critical)
             - ConventionalEligibilityCheck (leaf, critical)
          2) DownPaymentRequirement: parallel, critical
             - FHADownPayment (leaf, critical)
             - VADownPayment (leaf, critical)
             - ConventionalDownPayment (leaf, critical)
    """

    # ---------------- Root ----------------
    root = evaluator.add_sequential(
        id="Root",
        desc="Correctly identifies all qualifying mortgage loan types and their minimum down payment requirements for the given borrower profile",
        parent=evaluator.root,
        critical=True
    )

    # ---------------- LoanTypeIdentification (parallel, critical) ----------------
    loan_ident_node = evaluator.add_parallel(
        id="LoanTypeIdentification",
        desc="Correctly identifies which loan type(s) (FHA, VA, Conventional) the borrower qualifies for based on eligibility criteria",
        parent=root,
        critical=True
    )

    # FHA eligibility leaf
    fha_elig_leaf = evaluator.add_leaf(
        id="FHAEligibilityCheck",
        desc=("Correctly determines FHA loan eligibility: credit score meets minimum threshold "
              "(580 for 3.5% down or 500-579 for 10% down) and DTI ratio does not exceed ~43%"),
        parent=loan_ident_node,
        critical=True
    )
    # If the answer never stated eligibility for FHA, mark failed directly
    if extraction.qualifies_fha is None:
        fha_elig_leaf.score = 0.0
        fha_elig_leaf.status = "failed"
    else:
        fha_claim = (
            f"Based on {BORROWER_PROFILE_TEXT} the answer asserts the borrower {boolean_to_text(extraction.qualifies_fha)} "
            "qualify for an FHA loan. Verify whether that assertion is correct under FHA rules "
            "(e.g., >=580 credit score allows 3.5% down; 500-579 requires 10% down; typical DTI cap around 43%; "
            "owner-occupied primary residence; single-family)."
        )
        fha_sources = merge_sources(extraction.fha_sources, extraction.general_sources)
        await evaluator.verify(
            claim=fha_claim,
            node=fha_elig_leaf,
            sources=fha_sources if fha_sources else None,
            additional_instruction=(
                "Use the provided source(s) to confirm FHA qualification criteria and apply them to the stated borrower profile. "
                "Treat minor phrasing differences as acceptable; focus on whether the borrower would be eligible."
            )
        )

    # VA eligibility leaf
    va_elig_leaf = evaluator.add_leaf(
        id="VAEligibilityCheck",
        desc=("Correctly determines VA loan eligibility: borrower has military service eligibility "
              "(veteran/COE) and meets lender credit requirements"),
        parent=loan_ident_node,
        critical=True
    )
    if extraction.qualifies_va is None:
        va_elig_leaf.score = 0.0
        va_elig_leaf.status = "failed"
    else:
        va_claim = (
            f"Based on {BORROWER_PROFILE_TEXT} the answer asserts the borrower {boolean_to_text(extraction.qualifies_va)} "
            "qualify for a VA purchase loan. Verify whether that assertion is correct: valid COE, primary residence, "
            "and typical lender credit overlays (e.g., ~620 FICO) are satisfied."
        )
        va_sources = merge_sources(extraction.va_sources, extraction.general_sources)
        await evaluator.verify(
            claim=va_claim,
            node=va_elig_leaf,
            sources=va_sources if va_sources else None,
            additional_instruction=(
                "Use the provided source(s) to confirm VA purchase eligibility (COE, owner-occupancy) and common lender credit requirements. "
                "Apply them to the stated borrower profile to judge correctness of the answer's assertion."
            )
        )

    # Conventional eligibility leaf
    conv_elig_leaf = evaluator.add_leaf(
        id="ConventionalEligibilityCheck",
        desc=("Correctly determines Conventional loan eligibility: credit score is at least 620 and "
              "DTI does not exceed lender/AUS limits (typically 45-50%)"),
        parent=loan_ident_node,
        critical=True
    )
    if extraction.qualifies_conventional is None:
        conv_elig_leaf.score = 0.0
        conv_elig_leaf.status = "failed"
    else:
        conv_claim = (
            f"Based on {BORROWER_PROFILE_TEXT} the answer asserts the borrower {boolean_to_text(extraction.qualifies_conventional)} "
            "qualify for a Conventional conforming loan. Verify whether that assertion is correct given a 625 credit score "
            "and 40% DTI relative to typical AUS/lender caps (e.g., <=45-50%)."
        )
        conv_sources = merge_sources(extraction.conventional_sources, extraction.general_sources)
        await evaluator.verify(
            claim=conv_claim,
            node=conv_elig_leaf,
            sources=conv_sources if conv_sources else None,
            additional_instruction=(
                "Use the provided source(s) to confirm minimum credit score and general maximum DTI for conventional conforming loans, "
                "then apply to the stated borrower profile to judge correctness of the answer's assertion."
            )
        )

    # ---------------- DownPaymentRequirement (parallel, critical) ----------------
    downpay_node = evaluator.add_parallel(
        id="DownPaymentRequirement",
        desc="For each qualifying loan type identified, provides the correct minimum down payment percentage required",
        parent=root,
        critical=True
    )

    # FHA down payment leaf
    fha_dp_leaf = evaluator.add_leaf(
        id="FHADownPayment",
        desc="If FHA qualifies: states 3.5% down payment for credit score 580+ or 10% for credit score 500-579",
        parent=downpay_node,
        critical=True
    )
    if extraction.qualifies_fha:
        # If borrower qualifies FHA, a down payment percentage must be provided and correct
        if not extraction.fha_down_payment_pct or not isinstance(extraction.fha_down_payment_pct, str):
            fha_dp_leaf.score = 0.0
            fha_dp_leaf.status = "failed"
        else:
            fha_dp_claim = (
                f"The minimum required down payment for an FHA purchase for a borrower with a 625 credit score "
                f"is '{extraction.fha_down_payment_pct}'."
            )
            fha_sources = merge_sources(extraction.fha_sources, extraction.general_sources)
            await evaluator.verify(
                claim=fha_dp_claim,
                node=fha_dp_leaf,
                sources=fha_sources if fha_sources else None,
                additional_instruction=(
                    "Check FHA minimum down payment brackets: score >= 580 → 3.5%; score 500–579 → 10%. "
                    "Given the stated 625 score, the correct minimum should be 3.5%. "
                    "Mark PASS only if the stated percentage matches this bracket."
                )
            )
    else:
        # If FHA not claimed as qualifying (or not stated), this DP check is not applicable; treat as passed
        fha_dp_leaf.score = 1.0
        fha_dp_leaf.status = "passed"

    # VA down payment leaf
    va_dp_leaf = evaluator.add_leaf(
        id="VADownPayment",
        desc="If VA qualifies: states 0% down payment (no down payment required)",
        parent=downpay_node,
        critical=True
    )
    if extraction.qualifies_va:
        if not extraction.va_down_payment_pct or not isinstance(extraction.va_down_payment_pct, str):
            va_dp_leaf.score = 0.0
            va_dp_leaf.status = "failed"
        else:
            va_dp_claim = (
                f"The minimum required down payment for a VA purchase loan for an eligible borrower is "
                f"'{extraction.va_down_payment_pct}'."
            )
            va_sources = merge_sources(extraction.va_sources, extraction.general_sources)
            await evaluator.verify(
                claim=va_dp_claim,
                node=va_dp_leaf,
                sources=va_sources if va_sources else None,
                additional_instruction=(
                    "Confirm that VA purchase loans generally allow 0% down for eligible borrowers with sufficient entitlement, "
                    "especially for a $400,000 home in a standard-cost county. "
                    "Mark PASS only if the stated percentage is effectively 0%."
                )
            )
    else:
        va_dp_leaf.score = 1.0
        va_dp_leaf.status = "passed"

    # Conventional down payment leaf
    conv_dp_leaf = evaluator.add_leaf(
        id="ConventionalDownPayment",
        desc="If Conventional qualifies: states minimum 3% down payment for first-time homebuyers",
        parent=downpay_node,
        critical=True
    )
    if extraction.qualifies_conventional:
        if not extraction.conventional_down_payment_pct or not isinstance(extraction.conventional_down_payment_pct, str):
            conv_dp_leaf.score = 0.0
            conv_dp_leaf.status = "failed"
        else:
            conv_dp_claim = (
                f"The minimum required down payment for a conventional conforming purchase loan is "
                f"'{extraction.conventional_down_payment_pct}'."
            )
            conv_sources = merge_sources(extraction.conventional_sources, extraction.general_sources)
            await evaluator.verify(
                claim=conv_dp_claim,
                node=conv_dp_leaf,
                sources=conv_sources if conv_sources else None,
                additional_instruction=(
                    "Verify the minimum is 3% for first-time homebuyers (and certain programs such as HomeReady/Home Possible). "
                    "Given the prompt asks for the minimum required percentage, accept 3% as correct when stated as the minimum. "
                    "Mark PASS only if the stated percentage is consistent with this minimum."
                )
            )
    else:
        conv_dp_leaf.score = 1.0
        conv_dp_leaf.status = "passed"


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
    Evaluate an answer for the 2026 mortgage eligibility and minimum down payment task.
    Returns a standardized summary dictionary produced by the evaluator.
    """
    # Initialize evaluator
    evaluator = Evaluator()
    evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.SEQUENTIAL,  # Keep sequential per rubric: identification then down payments
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

    # Record borrower profile as custom info for clarity in logs
    evaluator.add_custom_info(
        info={
            "credit_score": "625",
            "dti_ratio": "40%",
            "veteran_with_coe": True,
            "occupancy": "primary residence",
            "property_type": "single-family",
            "purchase_price": "$400,000",
            "area_cost_tier": "standard-cost",
            "year": 2026,
        },
        info_type="borrower_profile",
        info_name="borrower_profile"
    )

    # Extract structured info from the answer
    extraction = await evaluator.extract(
        prompt=prompt_extract_mortgage_answer(),
        template_class=MortgageAnswerExtraction,
        extraction_name="mortgage_answer_extraction"
    )

    # Add Ground Truth context note (not enforcing a fixed ground truth here; rely on source-grounded verification)
    evaluator.add_ground_truth({
        "note": "Evaluation checks whether the answer's stated eligibility and minimum down payments are correct for the given borrower profile, grounded by the answer's cited sources when available."
    })

    # Build verification tree and run checks
    await build_and_verify_tree(evaluator, extraction)

    # Return summary
    return evaluator.get_summary()