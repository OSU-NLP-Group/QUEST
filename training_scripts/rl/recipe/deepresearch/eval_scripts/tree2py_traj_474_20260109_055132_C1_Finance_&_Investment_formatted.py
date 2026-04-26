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
TASK_ID = "sec_bitcoin_etf_lowest_expense_ratio"
TASK_DESCRIPTION = (
    "On January 10, 2024, the U.S. Securities and Exchange Commission (SEC) approved 11 spot Bitcoin exchange-traded products (ETPs) for listing and trading. "
    "Which of these 11 approved ETFs has the lowest standard expense ratio for long-term investors?"
)


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class ETFItem(BaseModel):
    """One ETF entry extracted from the answer."""
    name: Optional[str] = None
    ticker: Optional[str] = None
    expense_ratio_standard: Optional[str] = None
    expense_ratio_promotional: Optional[str] = None
    promotional_notes: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class ETFExtraction(BaseModel):
    """Structured extraction for the identified ETF and referenced list."""
    identified_etf_name: Optional[str] = None
    identified_etf_ticker: Optional[str] = None
    identified_etf_expense_ratio_standard: Optional[str] = None
    identified_etf_expense_ratio_promotional: Optional[str] = None
    identified_etf_sources: List[str] = Field(default_factory=list)

    # Any ETFs and fee info the answer provided (ideally the full set of 11)
    approved_etfs: List[ETFItem] = Field(default_factory=list)

    # General sources that mention the SEC approval and/or list of the 11 ETFs
    general_sources: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_etf_info() -> str:
    return """
    The answer should identify which spot Bitcoin ETF, among the 11 SEC-approved products on January 10, 2024, has the lowest standard (non-promotional) expense ratio for long-term investors.

    Extract the following fields exactly as presented in the answer:

    1) identified_etf_name: The fund name of the ETF the answer claims has the lowest standard expense ratio.
    2) identified_etf_ticker: The ticker symbol for the identified ETF (if mentioned); otherwise null.
    3) identified_etf_expense_ratio_standard: The standard/base expense ratio (exclude temporary/promotional fee waivers). If the answer does not explicitly provide the standard ratio, return null.
    4) identified_etf_expense_ratio_promotional: If the answer mentions a temporary or promotional expense ratio (e.g., fee waivers for an initial period), extract it; otherwise null.
    5) identified_etf_sources: All URLs the answer cites specifically to support the identification of the lowest-expense ETF (e.g., comparison articles, official fact sheets). Extract actual URLs only.

    Also extract any ETF fee details the answer provides for the SEC-approved set:
    6) approved_etfs: An array of objects. For each ETF entry the answer mentions:
       - name: Fund name
       - ticker: Ticker symbol, if available
       - expense_ratio_standard: The standard/base expense ratio (exclude temporary/promotional waivers), if provided
       - expense_ratio_promotional: Promotional/temporary fee ratio, if provided
       - promotional_notes: Any notes indicating waivers, limited-time discounts, or expiration info
       - sources: All URLs associated with that ETF's fee information (e.g., fact sheets, prospectuses, reputable news summaries)

    Finally:
    7) general_sources: Any URLs the answer cites that reference the SEC approval event and/or list the 11 approved ETFs (e.g., SEC press releases, reputable media articles). Extract only actual URLs.

    Important rules:
    - Extract only information explicitly present in the answer; do not infer or invent.
    - For URLs, include valid full URLs (markdown links are acceptable; extract the actual href).
    - If a field is missing in the answer, return null (or empty list for URLs).
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def _dedup_preserve_order(urls: List[str]) -> List[str]:
    seen = set()
    out = []
    for u in urls:
        if not u:
            continue
        if u not in seen:
            seen.add(u)
            out.append(u)
    return out


def collect_all_sources(extraction: ETFExtraction) -> List[str]:
    """Union of all relevant URLs cited in the answer."""
    urls: List[str] = []
    urls.extend(extraction.identified_etf_sources or [])
    urls.extend(extraction.general_sources or [])
    for etf in extraction.approved_etfs or []:
        urls.extend(etf.sources or [])
    return _dedup_preserve_order(urls)


# --------------------------------------------------------------------------- #
# Verification sub-tree                                                       #
# --------------------------------------------------------------------------- #
async def verify_etf_identification(evaluator: Evaluator, parent_node, extraction: ETFExtraction) -> None:
    """
    Build and run verification for ETF Identification as per rubric:
    - Approved_List_Verification: Identified ETF is one of the 11 SEC-approved spot Bitcoin ETFs on Jan 10, 2024.
    - Lowest_Expense_Ratio: Identified ETF has the lowest standard (non-promotional) expense ratio among those 11.
    """
    etf_node = evaluator.add_parallel(
        id="ETF_Identification",
        desc="Identifies the spot Bitcoin ETF with the lowest standard expense ratio from the SEC-approved list",
        parent=parent_node,
        critical=True
    )

    # Optional existence gate to avoid meaningless downstream checks
    has_identified_name = bool(extraction.identified_etf_name and extraction.identified_etf_name.strip())
    evaluator.add_custom_node(
        result=has_identified_name,
        id="ETF_Name_Present",
        desc="The answer identifies a specific ETF candidate by name",
        parent=etf_node,
        critical=True
    )

    # Leaf 1: Approved_List_Verification
    approved_leaf = evaluator.add_leaf(
        id="Approved_List_Verification",
        desc="The identified ETF is one of the 11 spot Bitcoin ETFs approved by the SEC on January 10, 2024",
        parent=etf_node,
        critical=True
    )

    etf_name = extraction.identified_etf_name or ""
    etf_ticker = extraction.identified_etf_ticker or ""
    approved_claim = (
        f"The ETF '{etf_name}'"
        + (f" (ticker '{etf_ticker}')" if etf_ticker else "")
        + " is one of the 11 spot Bitcoin ETFs approved by the U.S. Securities and Exchange Commission on January 10, 2024 for listing and trading."
    )

    approved_sources = _dedup_preserve_order((extraction.identified_etf_sources or []) + (extraction.general_sources or []))

    await evaluator.verify(
        claim=approved_claim,
        node=approved_leaf,
        sources=approved_sources if approved_sources else None,
        additional_instruction=(
            "Use the provided URLs to confirm that the identified ETF appears in the SEC approval lists or credible reports of the 11 spot Bitcoin ETFs approved on Jan 10, 2024. "
            "Allow minor naming or ticker variations. If the URLs are invalid, irrelevant, or do not support the claim, judge it as not supported."
        )
    )

    # Leaf 2: Lowest_Expense_Ratio
    lowest_leaf = evaluator.add_leaf(
        id="Lowest_Expense_Ratio",
        desc="The identified ETF has the lowest standard (non-promotional) expense ratio among all 11 approved ETFs",
        parent=etf_node,
        critical=True
    )

    std_ratio_str = extraction.identified_etf_expense_ratio_standard or ""
    lowest_claim = (
        f"Among the 11 SEC-approved spot Bitcoin ETFs, '{etf_name}'"
        + (f" (ticker '{etf_ticker}')" if etf_ticker else "")
        + (f" has the lowest standard (non-promotional) expense ratio" + (f" of {std_ratio_str}" if std_ratio_str else "") + " for long-term investors.")
    )

    all_sources = collect_all_sources(extraction)

    await evaluator.verify(
        claim=lowest_claim,
        node=lowest_leaf,
        sources=all_sources if all_sources else None,
        additional_instruction=(
            "Verify the comparative fee structure across the 11 SEC-approved spot Bitcoin ETFs. "
            "Focus on the standard/base expense ratio and ignore temporary or promotional fee waivers. "
            "If there is a tie for the lowest standard expense ratio, consider the claim correct as long as the identified ETF is among the tied lowest. "
            "Reject the claim if any competitor has a strictly lower standard ratio, or if the claim relies solely on temporary promotional fees."
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
) -> Dict[str, Any]:
    """
    Evaluate the agent's answer for the SEC spot Bitcoin ETFs lowest expense ratio question.
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
        default_model=model
    )

    # Extract structured info from the answer
    extraction = await evaluator.extract(
        prompt=prompt_extract_etf_info(),
        template_class=ETFExtraction,
        extraction_name="etf_lowest_expense_ratio_extraction"
    )

    # Build verification subtree and run checks
    await verify_etf_identification(evaluator, root, extraction)

    # Return standardized evaluation summary
    return evaluator.get_summary()