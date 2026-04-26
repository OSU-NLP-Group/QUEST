import asyncio
import logging
from typing import Any, List, Dict, Optional

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "California_CPA_Renewal_CPE_Requirements"
TASK_DESCRIPTION = (
    "A Certified Public Accountant holds an active CPA license in California and is preparing to renew their license. "
    "Their current two-year renewal period runs from February 1, 2024, to January 31, 2026. The CPA was originally "
    "licensed in California on March 15, 2018. To ensure compliance with all California Board of Accountancy "
    "requirements for license renewal, please provide: 1. The minimum total number of CPE hours required for the "
    "two-year renewal period; 2. The minimum number of CPE hours required in each individual year of the renewal "
    "period; 3. The minimum total number of technical subject CPE hours required for the two-year renewal period; "
    "4. The minimum number of technical subject CPE hours required in each individual year; 5. The required number "
    "of ethics CPE hours for the two-year renewal period; 6. Whether a Regulatory Review course is required for this "
    "renewal (and if so, how many hours), considering the CPA's license date; 7. The official California Board of "
    "Accountancy URL that confirms these CPE requirements."
)

LICENSE_DATE = "March 15, 2018"
RENEWAL_PERIOD = "February 1, 2024 – January 31, 2026"

# Ground truth expectations used for logging and context
GROUND_TRUTH_REQUIREMENTS = {
    "total_cpe_two_year_min": "80",
    "annual_cpe_per_year_min": "20",
    "technical_total_two_year_min": "40",
    "technical_annual_per_year_min": "12",
    "ethics_total_required": "4",
    "reg_review_every_6_years_hours": "2",
}


# --------------------------------------------------------------------------- #
# Extraction models                                                           #
# --------------------------------------------------------------------------- #
class RegulatoryReviewInfo(BaseModel):
    every_6_years_statement: Optional[str] = None
    first_renewal_rule_statement: Optional[str] = None
    conclusion_for_this_renewal_period: Optional[str] = None
    hours_if_required: Optional[str] = None


class CPERenewalExtraction(BaseModel):
    total_cpe_two_year_min: Optional[str] = None
    annual_cpe_per_year_min: Optional[str] = None
    technical_total_two_year_min: Optional[str] = None
    technical_annual_per_year_min: Optional[str] = None
    ethics_total_required: Optional[str] = None
    regulatory_review: Optional[RegulatoryReviewInfo] = None
    official_urls: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_cpa_requirements() -> str:
    return (
        "Extract the specific California CPA CPE requirements stated in the answer text. "
        "Return a JSON object with the following fields (use strings for numbers, exactly as stated in the answer; "
        "return null when missing):\n"
        "1) total_cpe_two_year_min: The minimum total CPE hours required for the two-year renewal period (e.g., '80').\n"
        "2) annual_cpe_per_year_min: The minimum number of CPE hours required in each individual year (e.g., '20').\n"
        "3) technical_total_two_year_min: The minimum total technical subject hours required over the two-year period "
        "(e.g., '40').\n"
        "4) technical_annual_per_year_min: The minimum technical subject hours required in each individual year "
        "(e.g., '12').\n"
        "5) ethics_total_required: The required number of ethics hours during the two-year period (e.g., '4').\n"
        "6) regulatory_review: An object with the following fields capturing what the answer states:\n"
        "   - every_6_years_statement: The wording that indicates Regulatory Review is required every six years and "
        "     how many hours (e.g., '2 hours every 6 years').\n"
        "   - first_renewal_rule_statement: Any mention of the special rule for CPAs licensed on or after July 1, 2024 "
        "     (e.g., 'first renewal must include Regulatory Review'), and whether it applies here.\n"
        "   - conclusion_for_this_renewal_period: The answer’s conclusion about whether Regulatory Review is required "
        "     during this renewal period given the license date March 15, 2018.\n"
        "   - hours_if_required: The hours stated for Regulatory Review in this renewal period, if applicable "
        "(e.g., '2').\n"
        "7) official_urls: An array of official URL(s) explicitly provided in the answer that corroborate these "
        "requirements—only include URLs from official California government domains (e.g., cba.ca.gov, dca.ca.gov). "
        "Do not invent URLs; extract only those explicitly present in the answer (including markdown links). "
        "If none are provided, return an empty array.\n"
        "Important: Do not infer or calculate values. Only extract what the answer explicitly states."
    )


# --------------------------------------------------------------------------- #
# Verification helpers                                                        #
# --------------------------------------------------------------------------- #
async def build_and_verify_main_requirements(
    evaluator: Evaluator,
    main_node,
) -> None:
    # Total CPE 80 (two-year minimum)
    node_total_80 = evaluator.add_leaf(
        id="Total_CPE_Hours_Required_80",
        desc="Correctly states that the minimum total CPE required for the two-year renewal period is 80 hours.",
        parent=main_node,
        critical=True,
    )
    await evaluator.verify(
        claim="The answer explicitly states that the minimum total CPE required for the two-year renewal period is 80 hours.",
        node=node_total_80,
        additional_instruction="Allow phrasing such as 'minimum of 80 hours' or 'at least 80 hours'. Minor wording variations are acceptable as long as 80 hours minimum is clearly conveyed."
    )

    # Annual minimum 20 per year
    node_annual_20 = evaluator.add_leaf(
        id="Annual_CPE_Hours_Required_20_Per_Year",
        desc="Correctly states that the minimum CPE required in each individual year of the renewal period is 20 hours per year.",
        parent=main_node,
        critical=True,
    )
    await evaluator.verify(
        claim="The answer explicitly states that the minimum CPE required in each individual year of the renewal period is 20 hours per year.",
        node=node_annual_20,
        additional_instruction="Accept equivalent wording indicating 'at least 20 hours each year' or 'minimum 20 per year'."
    )

    # Technical total 40 over two years
    node_technical_40 = evaluator.add_leaf(
        id="Total_Technical_CPE_Required_40",
        desc="Correctly states that at least 40 of the 80 total CPE hours must be in technical subjects over the two-year period.",
        parent=main_node,
        critical=True,
    )
    await evaluator.verify(
        claim="The answer explicitly states that at least 40 of the 80 total CPE hours must be in technical subjects over the two-year period.",
        node=node_technical_40,
        additional_instruction="Allow phrasing like '40+ technical hours' or 'minimum 40 technical hours across the two-year cycle'."
    )

    # Technical annual minimum 12 per year
    node_technical_annual_12 = evaluator.add_leaf(
        id="Annual_Technical_CPE_Required_12_Per_Year",
        desc="Correctly states that a minimum of 12 technical-subject CPE hours must be completed in each individual year.",
        parent=main_node,
        critical=True,
    )
    await evaluator.verify(
        claim="The answer explicitly states that a minimum of 12 technical-subject CPE hours must be completed in each individual year.",
        node=node_technical_annual_12,
        additional_instruction="Accept equivalent language; must clearly indicate 12 technical hours each year."
    )

    # Ethics total 4 over two years
    node_ethics_4 = evaluator.add_leaf(
        id="Ethics_CPE_Required_4_Total",
        desc="Correctly states that exactly 4 hours of ethics CPE are required during the two-year renewal cycle.",
        parent=main_node,
        critical=True,
    )
    await evaluator.verify(
        claim="The answer explicitly states that exactly 4 hours of ethics CPE are required during the two-year renewal cycle.",
        node=node_ethics_4,
        additional_instruction="Allow minor wording variations but the quantity must be 4 hours for ethics over the two-year period."
    )


async def build_and_verify_reg_review(
    evaluator: Evaluator,
    parent_node,
) -> None:
    reg_node = evaluator.add_parallel(
        id="Regulatory_Review_Course_Requirement",
        desc="Correctly addresses whether a Board-approved Regulatory Review course is required for this renewal, and if so, the required hours, using the provided regulatory-review constraints and the CPA's license date.",
        parent=parent_node,
        critical=True,
    )

    # 2-hour every 6 years statement
    node_rr_6yr_2hr = evaluator.add_leaf(
        id="Reg_Review_Every_6_Years_2_Hours",
        desc="States that a 2-hour Board-approved Regulatory Review course must be completed every 6 years.",
        parent=reg_node,
        critical=True,
    )
    await evaluator.verify(
        claim="The answer states that a Board-approved Regulatory Review course of 2 hours must be completed every six (6) years.",
        node=node_rr_6yr_2hr,
        additional_instruction="Minor phrasing variations are acceptable; it must clearly indicate frequency (every 6 years) and duration (2 hours)."
    )

    # First-renewal rule applicability (post 7/1/2024) correctly not applicable
    node_rr_first_rule = evaluator.add_leaf(
        id="Reg_Review_First_Renewal_Rule_After_2024_Not_Applicable",
        desc="Correctly notes that the special rule 'CPAs licensed on or after July 1, 2024 must complete Regulatory Review for their first renewal' does not apply to a CPA licensed on March 15, 2018.",
        parent=reg_node,
        critical=True,
    )
    await evaluator.verify(
        claim="The answer correctly indicates that the special rule requiring Regulatory Review at first renewal applies only to CPAs licensed on or after July 1, 2024 and therefore does not apply to someone licensed on March 15, 2018.",
        node=node_rr_first_rule,
        additional_instruction="Ensure the answer explicitly ties non-applicability to the March 15, 2018 license date."
    )

    # Conclusion for this renewal period (2024–2026) given 2018 license date
    node_rr_conclusion = evaluator.add_leaf(
        id="Reg_Review_Conclusion_For_This_Renewal_Period",
        desc="Provides a correct conclusion about applicability for this renewal period consistent with the constraints and available facts (e.g., identifies whether it is required or conditionally required, and includes the 2-hour amount if required).",
        parent=reg_node,
        critical=True,
    )
    await evaluator.verify(
        claim=(
            "Given the license date of March 15, 2018 and the requirement to complete a 2-hour Regulatory Review course every six years, "
            "the answer correctly concludes that the Regulatory Review course is required during the renewal period February 1, 2024 – January 31, 2026, "
            "and specifies that it is 2 hours."
        ),
        node=node_rr_conclusion,
        additional_instruction=(
            "A correct conclusion should recognize the six-year milestone occurs in 2024 (2018 + 6), which is within the stated renewal period. "
            "If the answer states it is required during this renewal period and includes '2 hours', mark as correct."
        )
    )


async def build_and_verify_official_url(
    evaluator: Evaluator,
    parent_node,
    official_urls: List[str],
) -> None:
    node_official_url = evaluator.add_leaf(
        id="Official_CBA_Source_URL",
        desc="Provides at least one official California Board of Accountancy (or official California government) URL that corroborates the stated CPE requirements.",
        parent=parent_node,
        critical=True,
    )

    claim = (
        "The provided URL(s) are official California Board of Accountancy or California government pages that "
        "corroborate California CPA CPE renewal requirements, including total hours, annual minimums, technical hours, ethics hours, "
        "and the Regulatory Review schedule."
    )
    await evaluator.verify(
        claim=claim,
        node=node_official_url,
        sources=official_urls if official_urls else None,
        additional_instruction=(
            "Verify that at least one URL is from an official domain (e.g., cba.ca.gov, dca.ca.gov) and the page content "
            "covers California CPA CPE requirements relevant to renewal."
        )
    )


# --------------------------------------------------------------------------- #
# Main evaluation function                                                    #
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

    # Extract structured data from the answer
    extraction = await evaluator.extract(
        prompt=prompt_extract_cpa_requirements(),
        template_class=CPERenewalExtraction,
        extraction_name="cpa_cpe_extraction",
    )

    # Add contextual ground truth info (for summary)
    evaluator.add_ground_truth({
        "license_date": LICENSE_DATE,
        "renewal_period": RENEWAL_PERIOD,
        "expected_requirements": GROUND_TRUTH_REQUIREMENTS,
    }, gt_type="expected_rules")

    # Add a critical top-level node matching rubric root
    main_node = evaluator.add_parallel(
        id="California_CPA_Renewal_Requirements_Information",
        desc="Evaluates whether the response correctly provides all California Board of Accountancy CPE renewal requirements for the specified renewal scenario, consistent with the given constraints.",
        parent=root,
        critical=True,
    )

    # Build and verify requirement leaves
    await build_and_verify_main_requirements(evaluator, main_node)

    # Regulatory Review subtree
    await build_and_verify_reg_review(evaluator, main_node)

    # Official URL verification
    await build_and_verify_official_url(
        evaluator,
        main_node,
        extraction.official_urls if extraction and extraction.official_urls else []
    )

    # Return standard summary
    return evaluator.get_summary()