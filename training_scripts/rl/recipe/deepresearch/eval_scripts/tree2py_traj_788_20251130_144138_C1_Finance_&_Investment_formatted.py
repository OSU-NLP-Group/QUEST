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
TASK_ID = "fdic_hysa_selection"
TASK_DESCRIPTION = """
Identify an FDIC-insured online high-yield savings account currently available in the United States that meets all of the following criteria: (1) offers an Annual Percentage Yield (APY) of at least 4.00%, (2) charges no monthly maintenance fee, and (3) requires no minimum deposit to open the account. Provide the name of the financial institution, the specific APY offered, and a reference URL to the bank's official page for this account.
"""


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class SavingsAccountExtraction(BaseModel):
    institution_name: Optional[str] = None
    account_name: Optional[str] = None
    apy: Optional[str] = None
    official_bank_url: Optional[str] = None
    extra_urls: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_savings_account() -> str:
    return """
    Extract the single savings account referenced in the answer that the responder proposes as meeting the task.
    Return a JSON object with the following fields:

    - institution_name: The bank/financial institution name offering the account (string).
    - account_name: The account/product name if provided (e.g., "High-Yield Savings") (string or null).
    - apy: The specific APY stated in the answer (e.g., "4.50%" or "4.5% APY") exactly as written in the answer (string or null).
    - official_bank_url: The URL to the institution's official page for this exact account (string or null). Do NOT use 3rd-party aggregator/review sites.
    - extra_urls: Any additional URLs mentioned in the answer relevant to the account (array of strings). Exclude the official_bank_url from this list.

    Rules:
    - Extract only what is explicitly present in the answer text.
    - If any field is missing, set it to null (or [] for extra_urls).
    - For URLs, prefer full https:// links. If a URL is missing a protocol, prepend http://
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def _has_numeric_rate(rate_text: Optional[str]) -> bool:
    if not rate_text:
        return False
    # Consider a concrete numeric rate if it contains at least one digit
    return any(ch.isdigit() for ch in rate_text)


# --------------------------------------------------------------------------- #
# Verification subtree builders                                               #
# --------------------------------------------------------------------------- #
async def build_required_response_fields(
    evaluator: Evaluator,
    parent_node,
    extracted: SavingsAccountExtraction,
) -> None:
    """
    Build and evaluate the 'Required_Response_Fields' subtree:
    - Institution_Name_Provided
    - Specific_APY_Provided
    - Official_Bank_URL_Provided
    All are critical existence checks.
    """
    req_node = evaluator.add_parallel(
        id="Required_Response_Fields",
        desc="The response includes all required pieces of information requested in the prompt.",
        parent=parent_node,
        critical=True
    )

    evaluator.add_custom_node(
        result=bool(extracted.institution_name and extracted.institution_name.strip()),
        id="Institution_Name_Provided",
        desc="Provides the name of the financial institution offering the account.",
        parent=req_node,
        critical=True
    )

    evaluator.add_custom_node(
        result=_has_numeric_rate(extracted.apy),
        id="Specific_APY_Provided",
        desc="States the specific APY offered for the account (a concrete numeric rate).",
        parent=req_node,
        critical=True
    )

    evaluator.add_custom_node(
        result=bool(extracted.official_bank_url and extracted.official_bank_url.strip()),
        id="Official_Bank_URL_Provided",
        desc="Provides a reference URL to the bank’s official page for this specific account (not a third-party aggregator page).",
        parent=req_node,
        critical=True
    )


async def build_meets_account_criteria(
    evaluator: Evaluator,
    parent_node,
    extracted: SavingsAccountExtraction,
) -> None:
    """
    Build and evaluate the 'Meets_Account_Criteria' subtree with evidence-based checks
    against the official bank URL.
    """
    criteria_node = evaluator.add_parallel(
        id="Meets_Account_Criteria",
        desc="The selected account satisfies all eligibility/feature constraints from the prompt.",
        parent=parent_node,
        critical=True
    )

    official_url = extracted.official_bank_url

    # FDIC Insured
    fdic_node = evaluator.add_leaf(
        id="FDIC_Insured",
        desc="The account/institution is FDIC-insured (deposit insurance up to $250,000 per depositor, per insured bank, per ownership category).",
        parent=criteria_node,
        critical=True
    )
    await evaluator.verify(
        claim=(
            "The institution and this savings account are FDIC-insured (e.g., 'Member FDIC', 'FDIC insured'). "
            "Do NOT count NCUA/credit union insurance as FDIC."
        ),
        node=fdic_node,
        sources=official_url,
        additional_instruction=(
            "Look for explicit text such as 'Member FDIC', 'FDIC Insured', or standard FDIC disclosure language on the page. "
            "If the page indicates credit union insurance (NCUA) or says deposits are NOT FDIC insured, mark incorrect."
        ),
    )

    # US Availability
    us_node = evaluator.add_leaf(
        id="US_Availability",
        desc="The account is currently available in the United States.",
        parent=criteria_node,
        critical=True
    )
    await evaluator.verify(
        claim=(
            "This savings account is currently available to customers in the United States (offered by a US bank/institution)."
        ),
        node=us_node,
        sources=official_url,
        additional_instruction=(
            "Confirm that the institution is a US bank and the product page indicates it is offered to US consumers. "
            "Signals may include US regulatory disclosures, US-centric terms, or an apply/open account flow targeted at US customers. "
            "If the page indicates it is restricted to non-US regions, mark incorrect."
        ),
    )

    # Online Account
    online_node = evaluator.add_leaf(
        id="Online_Account",
        desc="The account is an online high-yield savings account (as characterized in the prompt).",
        parent=criteria_node,
        critical=True
    )
    await evaluator.verify(
        claim=(
            "This product is a savings account that can be opened and managed online and is marketed or positioned as 'high-yield' (or equivalent phrasing)."
        ),
        node=online_node,
        sources=official_url,
        additional_instruction=(
            "Check that it is a savings account (not a checking/CD/MM account) and that the page shows an online application or management flow. "
            "Accept common variants like 'high-yield', 'high yield', or equivalent phrasing indicating above-average APY."
        ),
    )

    # APY Requirement (>= 4.00%)
    apy_req_node = evaluator.add_leaf(
        id="APY_Requirement",
        desc="The account offers an APY of at least 4.00%.",
        parent=criteria_node,
        critical=True
    )
    await evaluator.verify(
        claim="The APY shown for this account is at least 4.00%.",
        node=apy_req_node,
        sources=official_url,
        additional_instruction=(
            "Locate the APY on the official page. If multiple APYs are shown, focus on the standard/base APY. "
            "Allow minor rounding (e.g., 3.995% ≈ 4.00% should be treated as 4.00%). If the APY is below 4.00%, mark incorrect."
        ),
    )

    # No Monthly Maintenance Fee
    no_fee_node = evaluator.add_leaf(
        id="No_Monthly_Fee",
        desc="The account charges no monthly maintenance fee ($0 monthly fee).",
        parent=criteria_node,
        critical=True
    )
    await evaluator.verify(
        claim="This savings account charges no monthly maintenance fee.",
        node=no_fee_node,
        sources=official_url,
        additional_instruction=(
            "Look for phrases like 'no monthly maintenance fee', '$0 monthly fee', or equivalent. "
            "If monthly maintenance fees apply, mark incorrect."
        ),
    )

    # No Minimum Deposit to Open
    no_min_dep_node = evaluator.add_leaf(
        id="No_Minimum_Deposit",
        desc="The account requires no minimum deposit to open ($0 minimum opening deposit).",
        parent=criteria_node,
        critical=True
    )
    await evaluator.verify(
        claim="There is no minimum opening deposit required ($0 minimum) for this savings account.",
        node=no_min_dep_node,
        sources=official_url,
        additional_instruction=(
            "Verify the minimum opening deposit requirement. Accept phrasing like '$0 minimum opening deposit' or 'no minimum deposit'. "
            "If any minimum deposit greater than $0 is required, mark incorrect."
        ),
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
    Evaluate an answer for the FDIC-insured online high-yield savings account selection task.
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

    # Extract proposed account information from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_savings_account(),
        template_class=SavingsAccountExtraction,
        extraction_name="savings_account_selection",
    )

    # Build the main critical node
    main_node = evaluator.add_parallel(
        id="Savings_Account_Selection",
        desc="Identify an FDIC-insured online high-yield savings account currently available in the United States that meets all stated criteria, and provide the required details and official URL.",
        parent=root,
        critical=True,
    )

    # Build required response fields first (to act as preconditions for criteria verification)
    await build_required_response_fields(evaluator, main_node, extracted)

    # Build criteria verification subtree
    await build_meets_account_criteria(evaluator, main_node, extracted)

    # Optional: record constraints in summary for transparency
    evaluator.add_ground_truth({
        "required_constraints": [
            "FDIC-insured",
            "Available in the United States",
            "Online high-yield savings account",
            "APY >= 4.00%",
            "No monthly maintenance fee",
            "No minimum opening deposit"
        ],
        "required_response_fields": [
            "institution_name",
            "apy (specific numeric rate)",
            "official_bank_url (official page)"
        ]
    }, gt_type="rubric_requirements")

    # Return evaluation summary
    return evaluator.get_summary()