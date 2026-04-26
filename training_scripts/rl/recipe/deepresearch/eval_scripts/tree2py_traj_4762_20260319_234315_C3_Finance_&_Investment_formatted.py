import asyncio
import logging
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys

# ----------------------------- Task Constants ----------------------------- #
TASK_ID = "sec_generic_crypto_etf_first_launch"
TASK_DESCRIPTION = """
On September 17, 2025, the SEC approved generic listing standards for commodity-based trust shares, including cryptocurrency ETFs. This approval streamlined the process for launching new cryptocurrency spot ETFs. Prior to this approval, only Bitcoin and Ethereum had spot ETF approvals in the United States.

Identify the following:
1. The date when the SEC approved these generic listing standards
2. Which cryptocurrency (other than Bitcoin and Ethereum) had its first spot ETF launch under these new generic listing standards
3. The asset management company that launched this very first spot ETF for that cryptocurrency
4. The exact date this ETF began trading
5. The exchange where it was listed and its ticker symbol

Provide supporting URL references for your answer.
"""

# Expected constraints (as specified by the rubric)
EXPECTED_SEC_APPROVAL_DATE = "September 17, 2025"
EXPECTED_CRYPTO = "XRP"
EXPECTED_ASSET_MANAGER = "Canary Capital"
EXPECTED_TRADING_START_DATE = "November 13, 2025"
EXPECTED_EXCHANGE = "Nasdaq"  # Allow "NASDAQ" variant
EXPECTED_TICKER = "XRPC"


# ----------------------------- Data Models -------------------------------- #
class ETFFirstLaunchExtraction(BaseModel):
    """Structured extraction from the agent's answer."""
    sec_approval_date: Optional[str] = None
    cryptocurrency: Optional[str] = None
    asset_manager: Optional[str] = None
    trading_start_date: Optional[str] = None
    exchange: Optional[str] = None
    ticker: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


# --------------------------- Extraction Prompt ---------------------------- #
def prompt_extract_first_launch_info() -> str:
    return """
    Extract the following fields exactly as stated in the answer text:

    - sec_approval_date: The date the SEC approved generic listing standards for commodity-based trust shares (including cryptocurrency ETFs).
    - cryptocurrency: The cryptocurrency (other than Bitcoin and Ethereum) whose first U.S. spot ETF launched under these new generic listing standards.
    - asset_manager: The asset management company that launched the very first spot ETF for that cryptocurrency.
    - trading_start_date: The exact date this first ETF began trading.
    - exchange: The stock exchange where this ETF was listed (e.g., Nasdaq).
    - ticker: The ticker symbol of this ETF.
    - sources: All URLs explicitly cited in the answer text as supporting evidence. Include only valid URLs that actually appear in the answer text. Normalize to include protocol (prepend http:// if missing).

    Notes:
    - Do not invent any information. If any item is missing in the answer, set it to null (for strings) or [] (for the list).
    - For 'sources', extract every URL mentioned anywhere in the answer (including a "Sources" section or inline markdown links).
    """


# --------------------------- Helper Functions ----------------------------- #
def _norm(s: Optional[str]) -> str:
    return (s or "").strip()


def _urls_or_empty(extracted: ETFFirstLaunchExtraction) -> List[str]:
    return extracted.sources if extracted and extracted.sources else []


# -------------------------- Verification Blocks --------------------------- #
async def verify_sec_approval_date(evaluator: Evaluator, parent_node, data: ETFFirstLaunchExtraction):
    """Node: SEC_Approval_Date (critical)"""
    sec_node = evaluator.add_parallel(
        id="SEC_Approval_Date",
        desc="Provide the date the SEC approved generic listing standards for commodity-based trust shares (including cryptocurrency ETFs).",
        parent=parent_node,
        critical=True
    )

    # Presence check
    presence = evaluator.add_custom_node(
        result=bool(_norm(data.sec_approval_date)),
        id="SEC_Approval_Date_present",
        desc="SEC approval date is provided in the answer",
        parent=sec_node,
        critical=True
    )

    # Match expected constraint
    match_node = evaluator.add_leaf(
        id="SEC_Approval_Date_match_expected",
        desc=f"Provided SEC approval date matches expected '{EXPECTED_SEC_APPROVAL_DATE}'",
        parent=sec_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The stated SEC approval date '{_norm(data.sec_approval_date)}' equals '{EXPECTED_SEC_APPROVAL_DATE}' allowing only formatting variations (e.g., Sept. 17, 2025 vs September 17, 2025).",
        node=match_node,
        additional_instruction="Judge equality leniently for date formats; they must denote the same calendar date."
    )

    # Supported by cited sources
    support_node = evaluator.add_leaf(
        id="SEC_Approval_Date_supported",
        desc="SEC approval date is supported by the cited sources",
        parent=sec_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"On {_norm(data.sec_approval_date)}, the U.S. SEC approved generic listing standards for commodity-based trust shares that include cryptocurrency ETFs.",
        node=support_node,
        sources=_urls_or_empty(data),
        additional_instruction="Confirm the webpage states the SEC approval of generic listing standards (for commodity-based trust shares, including crypto ETFs) occurred on the given date."
    )


async def verify_first_cryptocurrency(evaluator: Evaluator, parent_node, data: ETFFirstLaunchExtraction):
    """Node: First_Cryptocurrency (critical)"""
    crypto_node = evaluator.add_parallel(
        id="First_Cryptocurrency",
        desc="Provide the cryptocurrency (other than Bitcoin and Ethereum) whose first spot ETF launched under the new generic listing standards.",
        parent=parent_node,
        critical=True
    )

    # Presence check
    presence = evaluator.add_custom_node(
        result=bool(_norm(data.cryptocurrency)),
        id="First_Cryptocurrency_present",
        desc="Cryptocurrency is provided in the answer",
        parent=crypto_node,
        critical=True
    )

    # Match expected constraint (XRP)
    match_node = evaluator.add_leaf(
        id="First_Cryptocurrency_match_expected",
        desc=f"Provided cryptocurrency matches expected '{EXPECTED_CRYPTO}'",
        parent=crypto_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The stated cryptocurrency '{_norm(data.cryptocurrency)}' is equivalent to '{EXPECTED_CRYPTO}' (allow case-insensitive variations and the synonym 'Ripple' for XRP).",
        node=match_node,
        additional_instruction="Accept 'XRP' or 'Ripple' as equivalent; judge case-insensitively."
    )

    # Supported by cited sources
    support_node = evaluator.add_leaf(
        id="First_Cryptocurrency_supported",
        desc="The identified 'first under new standards' cryptocurrency is supported by cited sources",
        parent=crypto_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"Under the SEC's new generic listing standards, the first cryptocurrency (other than Bitcoin and Ethereum) to have a spot ETF launched was {_norm(data.cryptocurrency)}.",
        node=support_node,
        sources=_urls_or_empty(data),
        additional_instruction="The page should explicitly support that this cryptocurrency had the first spot ETF launch under the new generic listing standards."
    )


async def verify_asset_manager(evaluator: Evaluator, parent_node, data: ETFFirstLaunchExtraction):
    """Node: Asset_Manager (critical)"""
    am_node = evaluator.add_parallel(
        id="Asset_Manager",
        desc="Provide the asset management company that launched the first spot ETF for that cryptocurrency.",
        parent=parent_node,
        critical=True
    )

    # Presence check
    presence = evaluator.add_custom_node(
        result=bool(_norm(data.asset_manager)),
        id="Asset_Manager_present",
        desc="Asset manager/company is provided in the answer",
        parent=am_node,
        critical=True
    )

    # Match expected constraint (Canary Capital)
    match_node = evaluator.add_leaf(
        id="Asset_Manager_match_expected",
        desc=f"Provided asset manager matches expected '{EXPECTED_ASSET_MANAGER}'",
        parent=am_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The stated asset manager '{_norm(data.asset_manager)}' equals '{EXPECTED_ASSET_MANAGER}' (allow minor variations like 'Canary Capital Management').",
        node=match_node,
        additional_instruction="Judge equivalence leniently for naming variants such as 'Canary Capital Management'."
    )

    # Supported by sources
    support_node = evaluator.add_leaf(
        id="Asset_Manager_supported",
        desc="Asset manager attribution is supported by cited sources",
        parent=am_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The first {_norm(data.cryptocurrency)} spot ETF under the SEC's new generic listing standards was launched by {_norm(data.asset_manager)}.",
        node=support_node,
        sources=_urls_or_empty(data),
        additional_instruction="The source should clearly attribute the launch/issuer/sponsor to the stated asset manager for the first ETF under the new standards."
    )


async def verify_trading_start_date(evaluator: Evaluator, parent_node, data: ETFFirstLaunchExtraction):
    """Node: ETF_Began_Trading_Date (critical)"""
    t_node = evaluator.add_parallel(
        id="ETF_Began_Trading_Date",
        desc="Provide the exact date the ETF began trading.",
        parent=parent_node,
        critical=True
    )

    # Presence
    presence = evaluator.add_custom_node(
        result=bool(_norm(data.trading_start_date)),
        id="ETF_Began_Trading_Date_present",
        desc="Trading start date is provided in the answer",
        parent=t_node,
        critical=True
    )

    # Match expected constraint
    match_node = evaluator.add_leaf(
        id="ETF_Began_Trading_Date_match_expected",
        desc=f"Trading start date matches expected '{EXPECTED_TRADING_START_DATE}'",
        parent=t_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The stated trading start date '{_norm(data.trading_start_date)}' equals '{EXPECTED_TRADING_START_DATE}' allowing common date formatting differences.",
        node=match_node,
        additional_instruction="Accept common format variants that denote the identical calendar date."
    )

    # Supported by sources
    support_node = evaluator.add_leaf(
        id="ETF_Began_Trading_Date_supported",
        desc="Trading start date is supported by cited sources",
        parent=t_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The first {_norm(data.cryptocurrency)} spot ETF under the SEC's new generic listing standards began trading on {_norm(data.trading_start_date)}.",
        node=support_node,
        sources=_urls_or_empty(data),
        additional_instruction="The page should clearly state the first-trading date for the first ETF under the new generic standards."
    )


async def verify_exchange_and_ticker(evaluator: Evaluator, parent_node, data: ETFFirstLaunchExtraction):
    """Node: Exchange_and_Ticker (critical)"""
    et_node = evaluator.add_parallel(
        id="Exchange_and_Ticker",
        desc="Provide the exchange where the ETF was listed and its ticker symbol.",
        parent=parent_node,
        critical=True
    )

    # Exchange presence
    exch_presence = evaluator.add_custom_node(
        result=bool(_norm(data.exchange)),
        id="Exchange_present",
        desc="Exchange is provided in the answer",
        parent=et_node,
        critical=True
    )

    # Ticker presence
    ticker_presence = evaluator.add_custom_node(
        result=bool(_norm(data.ticker)),
        id="Ticker_present",
        desc="Ticker is provided in the answer",
        parent=et_node,
        critical=True
    )

    # Exchange match expected
    exch_match = evaluator.add_leaf(
        id="Exchange_match_expected",
        desc=f"Exchange matches expected '{EXPECTED_EXCHANGE}'",
        parent=et_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The stated exchange '{_norm(data.exchange)}' is equivalent to '{EXPECTED_EXCHANGE}' (accept 'NASDAQ' as equivalent to 'Nasdaq').",
        node=exch_match,
        additional_instruction="Treat 'Nasdaq' and 'NASDAQ' as equivalent; be lenient with capitalization."
    )

    # Ticker match expected
    ticker_match = evaluator.add_leaf(
        id="Ticker_match_expected",
        desc=f"Ticker matches expected '{EXPECTED_TICKER}'",
        parent=et_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The stated ticker '{_norm(data.ticker)}' equals '{EXPECTED_TICKER}' (case-insensitive).",
        node=ticker_match,
        additional_instruction="Judge ticker equality case-insensitively."
    )

    # Exchange supported by sources
    exch_supported = evaluator.add_leaf(
        id="Exchange_supported",
        desc="Exchange listing is supported by cited sources",
        parent=et_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The first {_norm(data.cryptocurrency)} spot ETF under the SEC's new generic listing standards was listed on {_norm(data.exchange)}.",
        node=exch_supported,
        sources=_urls_or_empty(data),
        additional_instruction="Source should explicitly state the exchange where the ETF lists."
    )

    # Ticker supported by sources
    ticker_supported = evaluator.add_leaf(
        id="Ticker_supported",
        desc="Ticker symbol is supported by cited sources",
        parent=et_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The ticker symbol of the first {_norm(data.cryptocurrency)} spot ETF under the SEC's new generic listing standards is {_norm(data.ticker)}.",
        node=ticker_supported,
        sources=_urls_or_empty(data),
        additional_instruction="Source should explicitly show the ETF ticker."
    )


async def verify_supporting_urls(evaluator: Evaluator, parent_node, data: ETFFirstLaunchExtraction):
    """Node: Supporting_Evidence_URLs (critical)"""
    evaluator.add_custom_node(
        result=len(_urls_or_empty(data)) > 0,
        id="Supporting_Evidence_URLs",
        desc="At least one supporting URL is provided in the answer",
        parent=parent_node,
        critical=True
    )


# ------------------------------ Main Entry -------------------------------- #
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
    Evaluate an answer for the 'first crypto spot ETF under SEC generic listing standards' task.
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

    # Extract structured fields from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_first_launch_info(),
        template_class=ETFFirstLaunchExtraction,
        extraction_name="etf_first_launch_extraction",
    )

    # Record ground-truth/expected constraints for transparency
    evaluator.add_ground_truth(
        {
            "expected_sec_approval_date": EXPECTED_SEC_APPROVAL_DATE,
            "expected_crypto": EXPECTED_CRYPTO,
            "expected_asset_manager": EXPECTED_ASSET_MANAGER,
            "expected_trading_start_date": EXPECTED_TRADING_START_DATE,
            "expected_exchange": EXPECTED_EXCHANGE,
            "expected_ticker": EXPECTED_TICKER,
        },
        gt_type="expected_constraints",
    )

    # Build rubric root node (critical, parallel)
    complete_task = evaluator.add_parallel(
        id="Complete_Task",
        desc="Identify all required information about the first cryptocurrency spot ETF launched under the SEC's new generic listing standards and provide supporting URL references.",
        parent=root,
        critical=True
    )

    # Create and verify each sub-requirement
    await verify_sec_approval_date(evaluator, complete_task, extracted)
    await verify_first_cryptocurrency(evaluator, complete_task, extracted)
    await verify_asset_manager(evaluator, complete_task, extracted)
    await verify_trading_start_date(evaluator, complete_task, extracted)
    await verify_exchange_and_ticker(evaluator, complete_task, extracted)
    await verify_supporting_urls(evaluator, complete_task, extracted)

    return evaluator.get_summary()