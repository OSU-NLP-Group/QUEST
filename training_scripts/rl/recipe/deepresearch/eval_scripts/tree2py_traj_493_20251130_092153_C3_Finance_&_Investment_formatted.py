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
TASK_ID = "largest_public_pension_13f_q3_2024"
TASK_DESCRIPTION = (
    "What was the total fair market value (in U.S. dollars) of all Section 13(f) securities holdings reported by the largest public pension fund in the United States "
    "in its Form 13F filing for the quarter ending September 30, 2024? Provide the value as it appears in the 'Form 13F Information Table Value Total' field on the Summary Page, "
    "formatted according to SEC requirements effective January 3, 2023 (rounded to the nearest dollar, not thousand dollars). Also provide the SEC EDGAR URL for the filing."
)

# Ground truth context for constraints (not used to judge, only for info)
GROUND_TRUTH_CONTEXT = {
    "expected_institution_example": "California Public Employees' Retirement System (CalPERS)",
    "expected_state": "California",
    "reporting_period": "2024-09-30",
    "deadline_within_45_days": "2024-11-14",
    "form_type": "13F-HR"
}

# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class FilingExtraction(BaseModel):
    """Structured extraction of answer content relevant to this task."""
    institution_name: Optional[str] = None  # e.g., "California Public Employees' Retirement System (CalPERS)"
    institution_location: Optional[str] = None  # e.g., "California"
    value_total_usd: Optional[str] = None  # e.g., "123,456,789"
    edgar_url: Optional[str] = None  # SEC EDGAR URL to the specific 13F-HR filing
    reporting_period: Optional[str] = None  # e.g., "2024-09-30"
    filing_date: Optional[str] = None  # e.g., "2024-11-12"
    institution_support_urls: List[str] = Field(default_factory=list)  # URLs cited to support institution constraints


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_filing_info() -> str:
    return (
        "Extract the following fields from the answer:\n"
        "1) institution_name: The name of the institutional investor identified as the largest U.S. public pension fund.\n"
        "2) institution_location: The location (state) of the institution if stated; otherwise return null.\n"
        "3) value_total_usd: The total fair market value as shown in the 'Form 13F Information Table Value Total' field on the filing's Summary Page, "
        "exactly as presented in the answer (keep commas and any currency symbol if present in the answer).\n"
        "4) edgar_url: The SEC EDGAR URL for the specific Form 13F-HR filing used. If multiple URLs are given, choose the one that corresponds to the quarter ending 2024-09-30.\n"
        "5) reporting_period: The 'Period of Report' date mentioned in the answer, if any (e.g., 2024-09-30); otherwise null.\n"
        "6) filing_date: The filing date mentioned in the answer (e.g., 2024-11-12); otherwise null.\n"
        "7) institution_support_urls: All URLs provided in the answer that support claims about the institution (e.g., largest by AUM, located in California). "
        "Include any non-EDGAR URLs that substantiate those constraints.\n\n"
        "Special rules:\n"
        "- Extract only URLs explicitly present in the answer (plain, markdown, etc.). Ensure full URLs; if protocol missing, prepend http://.\n"
        "- Do not invent any values. If a field is not in the answer, return null.\n"
    )


# --------------------------------------------------------------------------- #
# Verification helpers                                                        #
# --------------------------------------------------------------------------- #
async def verify_institution_constraints(
    evaluator: Evaluator,
    parent_node,
    ex: FilingExtraction,
) -> None:
    """
    Build and verify the 'Identify_Institution_Meeting_Constraints' subtree.
    All nodes are critical and evaluated in parallel.
    """
    node = evaluator.add_parallel(
        id="Identify_Institution_Meeting_Constraints",
        desc="Identify the institutional investor that satisfies the investor constraints.",
        parent=parent_node,
        critical=True
    )

    # Existence of institution name (critical gate to help subsequent checks)
    evaluator.add_custom_node(
        result=bool(ex.institution_name and ex.institution_name.strip()),
        id="Institution_Name_Provided",
        desc="An institution name is identified in the answer.",
        parent=node,
        critical=True
    )

    # Largest US public pension by AUM
    n1 = evaluator.add_leaf(
        id="Is_Largest_US_Public_Pension_Fund_By_AUM",
        desc="The identified institution is the largest public pension fund in the United States by assets under management.",
        parent=node,
        critical=True
    )
    claim1 = (
        f"The institution '{ex.institution_name or ''}' is the largest public pension fund in the United States by assets under management."
    )
    sources1 = ex.institution_support_urls if ex.institution_support_urls else None
    await evaluator.verify(
        claim=claim1,
        node=n1,
        sources=sources1,
        additional_instruction=(
            "Verify using the provided sources whether the named institution is the largest U.S. public pension fund by AUM. "
            "Allow reasonable naming variants (e.g., 'CalPERS' vs full name)."
        )
    )

    # Located in California
    n2 = evaluator.add_leaf(
        id="Located_In_California",
        desc="The identified institution is located in California.",
        parent=node,
        critical=True
    )
    claim2 = (
        f"The institution '{ex.institution_name or ''}' is located in California."
    )
    sources2 = ex.institution_support_urls if ex.institution_support_urls else None
    await evaluator.verify(
        claim=claim2,
        node=n2,
        sources=sources2,
        additional_instruction=(
            "Confirm that the institution's location (headquarters or principal office) is in California using the provided sources."
        )
    )

    # Is 13F quarterly filer (institutional investment manager submits 13F filings)
    n3 = evaluator.add_leaf(
        id="Is_13F_Quarterly_Filer",
        desc="The identified institution files Form 13F quarterly with the SEC (i.e., is an institutional investment manager that submits 13F filings).",
        parent=node,
        critical=True
    )
    claim3 = (
        f"The institution '{ex.institution_name or ''}' submits Form 13F filings to the SEC (is a 13F institutional investment manager)."
    )
    # We can verify this using the specific EDGAR 13F-HR filing URL if provided
    sources3 = ex.edgar_url if ex.edgar_url else (ex.institution_support_urls if ex.institution_support_urls else None)
    await evaluator.verify(
        claim=claim3,
        node=n3,
        sources=sources3,
        additional_instruction=(
            "Use the EDGAR filing page (if provided) or other sources in the answer to confirm the institution files Form 13F."
        )
    )


async def verify_locate_correct_filing(
    evaluator: Evaluator,
    parent_node,
    ex: FilingExtraction,
) -> None:
    """
    Build and verify the 'Locate_Correct_13F_Filing' subtree.
    All nodes are critical and evaluated in parallel.
    """
    node = evaluator.add_parallel(
        id="Locate_Correct_13F_Filing",
        desc="Locate the institution's Form 13F-HR filing corresponding to the quarter ending September 30, 2024 on SEC EDGAR.",
        parent=parent_node,
        critical=True
    )

    # Existence gate: EDGAR URL provided
    evaluator.add_custom_node(
        result=bool(ex.edgar_url and ex.edgar_url.strip()),
        id="EDGAR_URL_Provided",
        desc="The SEC EDGAR URL for the specific Form 13F-HR filing is provided in the answer.",
        parent=node,
        critical=True
    )

    # Correct reporting period: 2024-09-30
    n1 = evaluator.add_leaf(
        id="Correct_Reporting_Period",
        desc="The filing’s reporting period is for the quarter ending September 30, 2024 (Period of Report = 2024-09-30).",
        parent=node,
        critical=True
    )
    claim1 = "The 'Period of Report' shown on this EDGAR filing page is 2024-09-30."
    await evaluator.verify(
        claim=claim1,
        node=n1,
        sources=ex.edgar_url,
        additional_instruction=(
            "Check the EDGAR filing's Summary Page for the 'Period of Report' field. It must be exactly '2024-09-30'."
        )
    )

    # Filing is Form 13F-HR on EDGAR
    n2 = evaluator.add_leaf(
        id="Filing_Is_Form_13F_HR_On_EDGAR",
        desc="The source is the SEC EDGAR record for the specific Form 13F-HR filing (electronically submitted via EDGAR).",
        parent=node,
        critical=True
    )
    claim2 = "This EDGAR record is for a 'Form 13F-HR' filing."
    await evaluator.verify(
        claim=claim2,
        node=n2,
        sources=ex.edgar_url,
        additional_instruction=(
            "Confirm the filing type on the EDGAR page is '13F-HR' (not '13F-NT' or other)."
        )
    )

    # Filed within 45 days after quarter end (on or before 2024-11-14)
    n3 = evaluator.add_leaf(
        id="Filed_Within_45_Days",
        desc="The filing date is within 45 days after the quarter end (i.e., on or before November 14, 2024 for a 2024-09-30 quarter end).",
        parent=node,
        critical=True
    )
    claim3 = "The filing date shown on the EDGAR page is on or before 2024-11-14."
    await evaluator.verify(
        claim=claim3,
        node=n3,
        sources=ex.edgar_url,
        additional_instruction=(
            "Locate the filing date on the EDGAR page (e.g., 'Filed on' or 'Filing Date') and judge whether it is on or before 2024-11-14."
        )
    )


async def verify_extract_value_and_url(
    evaluator: Evaluator,
    parent_node,
    ex: FilingExtraction,
) -> None:
    """
    Build and verify the 'Extract_And_Report_Value_Total_And_URL' subtree.
    All nodes are critical and evaluated in parallel.
    """
    node = evaluator.add_parallel(
        id="Extract_And_Report_Value_Total_And_URL",
        desc="Extract the required total value from the specified Summary Page field and report it with the EDGAR URL.",
        parent=parent_node,
        critical=True
    )

    # Value from Summary Page 'Form 13F Information Table Value Total'
    n1 = evaluator.add_leaf(
        id="Value_From_Summary_Page_Field",
        desc="The reported total fair market value is taken from the filing Summary Page field labeled “Form 13F Information Table Value Total.”",
        parent=node,
        critical=True
    )
    claim1 = (
        f"The 'Form 13F Information Table Value Total' on the filing's Summary Page equals '{ex.value_total_usd or ''}'."
    )
    await evaluator.verify(
        claim=claim1,
        node=n1,
        sources=ex.edgar_url,
        additional_instruction=(
            "Inspect the EDGAR filing's Summary Page. Find the field labeled 'Form 13F Information Table Value Total' and verify that its displayed numeric value "
            f"matches the value extracted from the answer ('{ex.value_total_usd or ''}'). Allow minor formatting differences (e.g., currency symbol or commas)."
        )
    )

    # Value format per 2023 rules (dollars rounded to nearest dollar, not thousands)
    n2 = evaluator.add_leaf(
        id="Value_Format_Per_2023_Rules",
        desc="The value is expressed in U.S. dollars rounded to the nearest dollar (not thousands), consistent with SEC requirements effective January 3, 2023.",
        parent=node,
        critical=True
    )
    claim2 = (
        "The Form 13F Information Table Value Total on the Summary Page is expressed in U.S. dollars rounded to the nearest dollar, not thousands."
    )
    await evaluator.verify(
        claim=claim2,
        node=n2,
        sources=ex.edgar_url,
        additional_instruction=(
            "Use the EDGAR page to determine the unit convention for the Information Table Value Total. "
            "Confirm that it is in dollars (no 'in thousands') per SEC rules effective January 3, 2023."
        )
    )

    # Provide EDGAR filing URL (valid SEC EDGAR page for the specific 13F-HR filing)
    n3 = evaluator.add_leaf(
        id="Provide_EDGAR_Filing_URL",
        desc="Provide the SEC EDGAR URL for the specific Form 13F-HR filing used as the reference source.",
        parent=node,
        critical=True
    )
    claim3 = "The provided URL is a valid SEC EDGAR page for the specific Form 13F-HR filing."
    await evaluator.verify(
        claim=claim3,
        node=n3,
        sources=ex.edgar_url,
        additional_instruction=(
            "Check that the URL domain is sec.gov and that the page corresponds to the specific Form 13F-HR filing referenced."
        )
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
    Evaluate an answer for the largest U.S. public pension fund Form 13F Q3 2024 task.
    """
    # Initialize evaluator with a sequential root strategy to respect task ordering
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.SEQUENTIAL,
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
    ex = await evaluator.extract(
        prompt=prompt_extract_filing_info(),
        template_class=FilingExtraction,
        extraction_name="filing_extraction"
    )

    # Record ground truth context for transparency (not used to judge)
    evaluator.add_ground_truth(GROUND_TRUTH_CONTEXT, gt_type="context_expectations")

    # Build critical sequential "Complete_Task" node as per rubric (all children must be critical)
    complete_task = evaluator.add_sequential(
        id="Complete_Task",
        desc="Provide the total fair market value (USD) of all Section 13(f) securities holdings from the Form 13F filing for the quarter ending September 30, 2024, for the largest U.S. public pension fund (by AUM) located in California, and provide the SEC EDGAR URL for the filing.",
        parent=root,
        critical=True
    )

    # 1) Identify institution meeting constraints
    await verify_institution_constraints(evaluator, complete_task, ex)

    # 2) Locate correct 13F filing
    await verify_locate_correct_filing(evaluator, complete_task, ex)

    # 3) Extract and report the required value and EDGAR URL
    await verify_extract_value_and_url(evaluator, complete_task, ex)

    # Return evaluation summary
    return evaluator.get_summary()