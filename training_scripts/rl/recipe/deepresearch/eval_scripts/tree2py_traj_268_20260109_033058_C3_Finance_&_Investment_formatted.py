import asyncio
import logging
from typing import List, Optional, Dict, Any

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.llm_client.base_client import LLMClient


# ----------------------------------------------------------------------------- #
# Task constants                                                                #
# ----------------------------------------------------------------------------- #
TASK_ID = "bitcoin_etf_multi_custody_criteria_v1"
TASK_DESCRIPTION = (
    "Among the 11 spot Bitcoin ETFs approved by the U.S. Securities and Exchange Commission on January 10, 2024, "
    "identify one ETF that satisfies ALL of the following criteria:\n\n"
    "1. The ETF must use a multi-custodian model (employing more than one custodian for its Bitcoin holdings)\n"
    "2. Coinbase Custody Trust Company must be one of the custodians used by the ETF\n"
    "3. At least one of the ETF's custodians must be a qualified custodian under the Investment Advisers Act of 1940\n"
    "4. At least one of the ETF's custodians must be a fiduciary under New York state law\n"
    "5. At least one of the ETF's custodians must be regulated by the New York Department of Financial Services (NYDFS)\n"
    "6. The ETF's expense ratio must be 0.25% or lower as of December 5, 2025\n"
    "7. The ETF must have added additional custodians after its initial launch in January 2024\n\n"
    "Provide the ETF's full name, ticker symbol, and reference URLs supporting each criterion."
)


# ----------------------------------------------------------------------------- #
# Extraction models                                                             #
# ----------------------------------------------------------------------------- #
class ETFExtraction(BaseModel):
    # Identity
    name: Optional[str] = None
    ticker: Optional[str] = None
    # Custodians mentioned in the answer (names only, as text)
    custodian_names: List[str] = Field(default_factory=list)

    # URL groups directly cited in the answer for each verification aspect
    identity_approval_urls: List[str] = Field(default_factory=list)   # Supports identity + approval among 11 on Jan 10, 2024
    custodian_model_urls: List[str] = Field(default_factory=list)     # Supports multi-custodian + Coinbase inclusion + custodian identities
    regulatory_urls: List[str] = Field(default_factory=list)          # Supports qualified custodian, fiduciary status, NYDFS regulation
    cold_storage_urls: List[str] = Field(default_factory=list)        # Supports institutional-grade cold storage custody arrangement
    fee_urls: List[str] = Field(default_factory=list)                 # Supports fee/expense ratio as of Dec 5, 2025
    post_launch_urls: List[str] = Field(default_factory=list)         # Supports post-launch custodian additions

    # Optional textual claims from the answer (if present)
    expense_ratio: Optional[str] = None           # e.g., "0.25%" or "0.19%"
    expense_ratio_asof: Optional[str] = None      # a date string if explicitly stated


# ----------------------------------------------------------------------------- #
# Extraction prompt                                                             #
# ----------------------------------------------------------------------------- #
def prompt_extract_etf() -> str:
    return """
Extract exactly one ETF that the answer claims satisfies all criteria. If multiple ETFs are mentioned, select the first ETF that the answer uses to fulfill all the requirements. Extract ONLY information explicitly present in the answer text.

Return a single JSON object with the following fields:
- name: the ETF's full name (string or null)
- ticker: the ETF's ticker symbol (string or null)
- custodian_names: array of custodian names mentioned for this ETF ([] if none)
- identity_approval_urls: array of URLs that support the ETF's identity (name/ticker) AND its SEC approval eligibility as one of the 11 spot Bitcoin ETFs approved on Jan 10, 2024 ([] if none)
- custodian_model_urls: array of URLs that support the ETF using a multi-custodian model and that name the custodians (including Coinbase Custody Trust Company) ([] if none)
- regulatory_urls: array of URLs that support (any of): a custodian being a Qualified Custodian under the Investment Advisers Act of 1940; a custodian being a fiduciary under New York law; a custodian being regulated/chartered/supervised by NYDFS ([] if none)
- cold_storage_urls: array of URLs that support institutional-grade custody using cold storage ([] if none)
- fee_urls: array of URLs that document the ETF's expense ratio as of (or updated no earlier than) December 5, 2025 ([] if none)
- post_launch_urls: array of URLs that support that the ETF added additional custodians after its initial January 2024 launch ([] if none)
- expense_ratio: the expense ratio string as stated in the answer (e.g., "0.25%"), if present, else null
- expense_ratio_asof: the "as of" date string for the expense ratio if present in the answer (e.g., "December 5, 2025"), else null

Important:
- Extract ONLY URLs explicitly present in the answer text (plain URLs or inside markdown links).
- Do not fabricate or infer URLs.
- If a field is not found in the answer, return null (for string fields) or [] (for URL lists).
"""


# ----------------------------------------------------------------------------- #
# Helper utilities                                                              #
# ----------------------------------------------------------------------------- #
def _list_to_english(items: List[str]) -> str:
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    return ", ".join(items[:-1]) + " and " + items[-1]


async def _verify_with_urls_required(
    evaluator: Evaluator,
    claim: str,
    node_id: str,
    node_desc: str,
    parent,
    urls: Optional[List[str]],
    critical: bool = True,
    additional_instruction: str = "None"
):
    """
    Create a leaf node and verify the claim against given URLs.
    If no URLs are provided, the leaf immediately fails (enforcing the 'Provide reference URL(s)' requirement).
    """
    leaf = evaluator.add_leaf(
        id=node_id,
        desc=node_desc,
        parent=parent,
        critical=critical
    )

    # Enforce that URL-backed claims actually have URLs
    if not urls or len(urls) == 0:
        leaf.score = 0.0
        leaf.status = "failed"
        return False

    return await evaluator.verify(
        claim=claim,
        node=leaf,
        sources=urls,
        additional_instruction=additional_instruction
    )


# ----------------------------------------------------------------------------- #
# Verification subtrees                                                         #
# ----------------------------------------------------------------------------- #
async def verify_identification_and_eligibility(evaluator: Evaluator, root, ex: ETFExtraction):
    node = evaluator.add_parallel(
        id="ETF_Identification_and_Eligibility",
        desc="Verify the ETF is eligible (SEC-approved spot Bitcoin ETF on Jan 10, 2024) and is correctly identified, with supporting evidence",
        parent=root,
        critical=True
    )

    # 1) Approved_ETF_Status (must verify via URL; fail if no URLs)
    name_part = ex.name or "the ETF"
    ticker_part = f" (ticker {ex.ticker})" if ex.ticker else ""
    claim_approved = (
        f"{name_part}{ticker_part} is one of the 11 spot Bitcoin ETFs approved by the U.S. Securities and Exchange Commission on January 10, 2024."
    )
    await _verify_with_urls_required(
        evaluator=evaluator,
        claim=claim_approved,
        node_id="Approved_ETF_Status",
        node_desc="The identified ETF must be one of the 11 spot Bitcoin ETFs approved by the SEC on January 10, 2024",
        parent=node,
        urls=ex.identity_approval_urls,
        critical=True,
        additional_instruction=(
            "Confirm that the page explicitly lists or confirms the ETF was approved as part of the 11 spot Bitcoin ETFs on Jan 10, 2024. "
            "Accept official SEC approvals, reputable lists, or press releases that clearly indicate this ETF was among those 11."
        )
    )

    # 2) ETF_Name_Ticker_Provided (existence check)
    evaluator.add_custom_node(
        result=bool(ex.name) and bool(ex.ticker),
        id="ETF_Name_Ticker_Provided",
        desc="The solution must provide the ETF's full name and ticker symbol",
        parent=node,
        critical=True
    )

    # 3) URL_Reference_ETF_Identity_and_Approval (must verify via URL; fail if no URLs)
    claim_identity_urls = (
        f"The provided reference page(s) confirm the ETF's identity (full name '{ex.name}' and ticker '{ex.ticker}') "
        f"and that it was approved on January 10, 2024 as one of the 11 spot Bitcoin ETFs."
    )
    await _verify_with_urls_required(
        evaluator=evaluator,
        claim=claim_identity_urls,
        node_id="URL_Reference_ETF_Identity_and_Approval",
        node_desc="Provide reference URL(s) supporting the ETF’s identity (name/ticker) and its SEC approval eligibility (one of the 11 approved on Jan 10, 2024)",
        parent=node,
        urls=ex.identity_approval_urls,
        critical=True,
        additional_instruction=(
            "The page(s) should clearly show the ETF's full name and ticker, and explicitly indicate approval on Jan 10, 2024 as part of the 11 spot Bitcoin ETFs."
        )
    )


async def verify_custodians_and_model(evaluator: Evaluator, root, ex: ETFExtraction):
    node = evaluator.add_parallel(
        id="Custodians_and_Custody_Model",
        desc="Verify the ETF uses a multi-custodian model and includes Coinbase Custody Trust Company, with supporting evidence",
        parent=root,
        critical=True
    )

    # 1) Multiple_Custodians_Used (must verify via URL)
    await _verify_with_urls_required(
        evaluator=evaluator,
        claim="The ETF uses more than one custodian for its Bitcoin holdings (i.e., a multi-custodian model).",
        node_id="Multiple_Custodians_Used",
        node_desc="The ETF must use more than one custodian for its Bitcoin holdings (multi-custodian model)",
        parent=node,
        urls=ex.custodian_model_urls,
        critical=True,
        additional_instruction=(
            "Look for explicit statements of 'multiple custodians', 'co-custodians', or listings of two or more custodian entities responsible for Bitcoin custody."
        )
    )

    # 2) Coinbase_Inclusion (must verify via URL)
    await _verify_with_urls_required(
        evaluator=evaluator,
        claim="Coinbase Custody Trust Company is one of the ETF's custodians.",
        node_id="Coinbase_Inclusion",
        node_desc="Coinbase Custody Trust Company must be one of the custodians used by the ETF",
        parent=node,
        urls=ex.custodian_model_urls,
        critical=True,
        additional_instruction=(
            "The evidence should explicitly name 'Coinbase Custody Trust Company' as a custodian for the ETF."
        )
    )

    # 3) URL_Reference_Custodian_Model_and_Identities (must verify via URL)
    custodian_list_str = _list_to_english(ex.custodian_names)
    claim_custodian_refs = (
        f"The provided reference page(s) identify the ETF's custodians (e.g., {custodian_list_str if custodian_list_str else 'named custodians'}) "
        f"and confirm that the ETF uses a multi-custodian model including Coinbase Custody Trust Company."
    )
    await _verify_with_urls_required(
        evaluator=evaluator,
        claim=claim_custodian_refs,
        node_id="URL_Reference_Custodian_Model_and_Identities",
        node_desc="Provide reference URL(s) supporting the ETF’s multi-custodian model and naming the custodians (including Coinbase Custody Trust Company)",
        parent=node,
        urls=ex.custodian_model_urls,
        critical=True,
        additional_instruction=(
            "The pages should both (1) evidence the use of multiple custodians and (2) explicitly name those custodians, including Coinbase Custody Trust Company."
        )
    )


async def verify_custodian_regulatory_attributes(evaluator: Evaluator, root, ex: ETFExtraction):
    node = evaluator.add_parallel(
        id="Custodian_Regulatory_Attributes",
        desc="Verify custodianship regulatory attributes are satisfied, with supporting evidence",
        parent=root,
        critical=True
    )

    # Using the list of custodian names for context in the claim
    custodian_list_str = _list_to_english(ex.custodian_names) if ex.custodian_names else "the ETF's custodian(s)"

    # 1) Qualified_Custodian_Status (must verify via URL)
    await _verify_with_urls_required(
        evaluator=evaluator,
        claim=f"At least one of {custodian_list_str} is a 'qualified custodian' under the Investment Advisers Act of 1940.",
        node_id="Qualified_Custodian_Status",
        node_desc="At least one of the ETF's custodians must be a qualified custodian under the Investment Advisers Act of 1940",
        parent=node,
        urls=ex.regulatory_urls,
        critical=True,
        additional_instruction=(
            "Accept explicit references to 'Qualified Custodian' under the Advisers Act or equivalent authoritative designation. "
            "State-chartered trust companies that are explicitly described as qualified custodians also qualify."
        )
    )

    # 2) Fiduciary_Status (must verify via URL)
    await _verify_with_urls_required(
        evaluator=evaluator,
        claim=f"At least one of {custodian_list_str} is a fiduciary under New York state law.",
        node_id="Fiduciary_Status",
        node_desc="At least one of the ETF's custodians must be a fiduciary under New York state law",
        parent=node,
        urls=ex.regulatory_urls,
        critical=True,
        additional_instruction=(
            "Look for explicit statements that the custodian acts as a fiduciary under New York law (e.g., as a New York limited purpose trust company with fiduciary responsibilities)."
        )
    )

    # 3) NYDFS_Regulation (must verify via URL)
    await _verify_with_urls_required(
        evaluator=evaluator,
        claim=f"At least one of {custodian_list_str} is regulated by the New York Department of Financial Services (NYDFS).",
        node_id="NYDFS_Regulation",
        node_desc="At least one of the ETF's custodians must be regulated by the New York Department of Financial Services (NYDFS)",
        parent=node,
        urls=ex.regulatory_urls,
        critical=True,
        additional_instruction=(
            "Accept explicit references to being 'regulated', 'chartered', or 'supervised' by NYDFS, including holding a NY trust charter or BitLicense when applicable."
        )
    )

    # 4) URL_Reference_Custodian_Regulatory_Qualifications (must verify via URL)
    claim_reg_refs = (
        "The provided reference page(s) collectively support that: "
        "(1) at least one custodian is a Qualified Custodian under the Investment Advisers Act of 1940; "
        "(2) at least one custodian is a fiduciary under New York law; and "
        "(3) at least one custodian is regulated by NYDFS."
    )
    await _verify_with_urls_required(
        evaluator=evaluator,
        claim=claim_reg_refs,
        node_id="URL_Reference_Custodian_Regulatory_Qualifications",
        node_desc="Provide reference URL(s) supporting the qualified-custodian, NY fiduciary, and NYDFS-regulated status claims (as applicable)",
        parent=node,
        urls=ex.regulatory_urls,
        critical=True,
        additional_instruction=(
            "Ensure that the cited pages explicitly provide evidence for all three attributes across one or more custodians."
        )
    )


async def verify_institutional_grade_cold_storage(evaluator: Evaluator, root, ex: ETFExtraction):
    node = evaluator.add_parallel(
        id="Institutional_Grade_Cold_Storage",
        desc="Verify the institutional-grade cold storage custody constraint (from constraints list), with supporting evidence",
        parent=root,
        critical=True
    )

    # 1) Institutional_Grade_Custody_Cold_Storage_Claim (must verify via URL)
    await _verify_with_urls_required(
        evaluator=evaluator,
        claim="The ETF employs institutional-grade custodial arrangements that provide cold storage for Bitcoin holdings.",
        node_id="Institutional_Grade_Custody_Cold_Storage_Claim",
        node_desc="The ETF must have institutional-grade custodial arrangements that provide cold storage for Bitcoin holdings",
        parent=node,
        urls=ex.cold_storage_urls,
        critical=True,
        additional_instruction=(
            "The page(s) should explicitly mention cold storage and indicate institutional-grade/enterprise-grade security or custody."
        )
    )

    # 2) URL_Reference_Cold_Storage (must verify via URL)
    await _verify_with_urls_required(
        evaluator=evaluator,
        claim="The provided reference page(s) explicitly support the institutional-grade cold storage custody arrangement claim for the ETF.",
        node_id="URL_Reference_Cold_Storage",
        node_desc="Provide reference URL(s) supporting the institutional-grade cold storage custody arrangement claim",
        parent=node,
        urls=ex.cold_storage_urls,
        critical=True,
        additional_instruction="The page(s) should explicitly reference cold storage and institutional-grade custody for this ETF."
    )


async def verify_expense_ratio_requirement(evaluator: Evaluator, root, ex: ETFExtraction):
    node = evaluator.add_parallel(
        id="Expense_Ratio_Requirement",
        desc="Verify the expense ratio constraint as of December 5, 2025, with supporting evidence",
        parent=root,
        critical=True
    )

    # 1) Expense_Ratio_Limit (must verify via URL)
    ratio_txt = ex.expense_ratio or "0.25% or lower"
    claim_fee = (
        f"The ETF's expense ratio is 0.25% or lower as of December 5, 2025 (e.g., stated as {ratio_txt} on or after that date)."
    )
    await _verify_with_urls_required(
        evaluator=evaluator,
        claim=claim_fee,
        node_id="Expense_Ratio_Limit",
        node_desc="The ETF's expense ratio must be 0.25% or lower as of December 5, 2025",
        parent=node,
        urls=ex.fee_urls,
        critical=True,
        additional_instruction=(
            "Confirm the expense ratio displayed on the page is ≤ 0.25%. Prefer evidence that is dated 'as of' Dec 5, 2025 or updated on/after that date. "
            "If the page shows a current fee on/after that date that is ≤ 0.25%, consider it acceptable."
        )
    )

    # 2) URL_Reference_Fee (must verify via URL)
    claim_fee_urls = (
        "The provided reference page(s) document the ETF's expense ratio and indicate that the value is current as of December 5, 2025 "
        "or was updated on/after that date."
    )
    await _verify_with_urls_required(
        evaluator=evaluator,
        claim=claim_fee_urls,
        node_id="URL_Reference_Fee",
        node_desc="Provide reference URL(s) documenting the expense ratio as of December 5, 2025",
        parent=node,
        urls=ex.fee_urls,
        critical=True,
        additional_instruction=(
            "Look for explicit 'as of' dates or page update timestamps; pass if the page clearly supports the ratio on/after Dec 5, 2025."
        )
    )


async def verify_post_launch_custodian_addition(evaluator: Evaluator, root, ex: ETFExtraction):
    node = evaluator.add_parallel(
        id="Post_Launch_Custodian_Addition",
        desc="Verify the ETF added additional custodians after its initial January 2024 launch, with supporting evidence",
        parent=root,
        critical=True
    )

    # 1) Added_Custodians_After_Launch (must verify via URL)
    await _verify_with_urls_required(
        evaluator=evaluator,
        claim="The ETF added additional custodians after its initial launch in January 2024.",
        node_id="Added_Custodians_After_Launch",
        node_desc="The ETF must have added additional custodians after its initial launch in January 2024",
        parent=node,
        urls=ex.post_launch_urls,
        critical=True,
        additional_instruction=(
            "Evidence should clearly indicate the addition of new custodian(s) at a date after January 2024. "
            "Press releases, sponsor announcements, or updated filings are acceptable."
        )
    )

    # 2) URL_Reference_Post_Launch_Custodian_Addition (must verify via URL)
    await _verify_with_urls_required(
        evaluator=evaluator,
        claim="The provided reference page(s) explicitly state that additional custodians were added after the ETF's January 2024 launch.",
        node_id="URL_Reference_Post_Launch_Custodian_Addition",
        node_desc="Provide reference URL(s) supporting that additional custodians were added after the initial January 2024 launch",
        parent=node,
        urls=ex.post_launch_urls,
        critical=True,
        additional_instruction=(
            "The page(s) must clearly show that additional custodian(s) were added after the initial launch in January 2024, including dates/timelines when possible."
        )
    )


# ----------------------------------------------------------------------------- #
# Main evaluation entry point                                                   #
# ----------------------------------------------------------------------------- #
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
    Evaluate an agent's answer for the Bitcoin ETF multi-custodian and regulatory criteria task.
    """
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Root is parallel, all children critical; any failure will fail overall.
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

    # Set root critical status to align with rubric (root critical -> all children critical)
    root.critical = True
    root.desc = "Evaluate whether the identified Bitcoin ETF meets all specified eligibility, custodial, regulatory, operational, and evidence (URL) requirements"

    # Extract structured information from the answer
    ex: ETFExtraction = await evaluator.extract(
        prompt=prompt_extract_etf(),
        template_class=ETFExtraction,
        extraction_name="etf_extraction"
    )

    # Build verification subtrees according to rubric (all critical under a critical root)
    await verify_identification_and_eligibility(evaluator, root, ex)
    await verify_custodians_and_model(evaluator, root, ex)
    await verify_custodian_regulatory_attributes(evaluator, root, ex)
    await verify_institutional_grade_cold_storage(evaluator, root, ex)
    await verify_expense_ratio_requirement(evaluator, root, ex)
    await verify_post_launch_custodian_addition(evaluator, root, ex)

    # Return summary with verification tree and final score
    return evaluator.get_summary()