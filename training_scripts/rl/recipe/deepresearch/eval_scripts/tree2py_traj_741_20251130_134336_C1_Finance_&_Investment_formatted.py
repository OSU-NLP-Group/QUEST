import asyncio
import logging
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "franklin_xrp_etf"
TASK_DESCRIPTION = """
What is the ticker symbol and primary listing exchange for Franklin Templeton's XRP ETF that launched in November 2025?
"""


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class ETFInfoExtraction(BaseModel):
    sponsor_name: Optional[str] = None
    product_name: Optional[str] = None
    asset: Optional[str] = None
    launch_month_year: Optional[str] = None
    ticker: Optional[str] = None
    primary_exchange: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_etf_info() -> str:
    return """
    Extract the specific ETF information mentioned in the answer that pertains to Franklin Templeton's XRP ETF launched in November 2025.
    Return a JSON object with the following fields:
    - sponsor_name: The issuer/sponsor name (e.g., "Franklin Templeton")
    - product_name: The ETF product name as stated in the answer (if provided)
    - asset: The underlying asset (should be XRP if this is the XRP ETF)
    - launch_month_year: The stated launch month and year (e.g., "November 2025")
    - ticker: The ticker symbol of the ETF (e.g., "FXRP" or similar, as presented in the answer)
    - primary_exchange: The primary listing exchange for the ETF (e.g., "NASDAQ", "NYSE Arca", "Cboe BZX")
    - sources: An array of URLs explicitly cited in the answer that support the identification, ticker, and exchange details for this ETF.
    
    IMPORTANT:
    - Extract only what appears in the answer. If a field is not mentioned, set it to null (or an empty array for sources).
    - For sources, include only valid, complete URLs explicitly present in the answer (plain URLs or Markdown links).
    - Do not infer or invent values; be faithful to the text.
    """


# --------------------------------------------------------------------------- #
# Verification logic                                                          #
# --------------------------------------------------------------------------- #
async def verify_ft_xrp_etf(
    evaluator: Evaluator,
    parent_node,
    etf_info: ETFInfoExtraction,
) -> None:
    """
    Build and execute verification for the Franklin Templeton XRP ETF info (ticker and primary exchange),
    ensuring we are referring to the correct product launched in November 2025.
    """
    # Create critical parent node (parallel aggregation)
    ft_node = evaluator.add_parallel(
        id="Franklin_Templeton_XRP_ETF_Information",
        desc="Verify the ticker symbol and primary listing exchange for Franklin Templeton's XRP ETF launched in November 2025, with correctness as of November 2025.",
        parent=parent_node,
        critical=True,
    )

    # Prepare common information
    sponsor = etf_info.sponsor_name or ""
    product_name = etf_info.product_name or ""
    asset = etf_info.asset or ""
    launch_m_y = etf_info.launch_month_year or ""
    ticker = etf_info.ticker or ""
    primary_ex = etf_info.primary_exchange or ""
    sources = etf_info.sources if etf_info.sources else []

    # Leaf 1: ETF Identification (Critical)
    id_node = evaluator.add_leaf(
        id="ETF_Identification",
        desc="The answer clearly pertains to Franklin Templeton's XRP ETF that launched in November 2025 (not a different product).",
        parent=ft_node,
        critical=True,
    )
    identification_claim = (
        "These sources show that there is a Franklin Templeton XRP ETF and the answer refers to that specific ETF. "
        "It is an exchange-traded fund sponsored by Franklin Templeton, associated with XRP (Ripple), and its launch occurred in November 2025."
    )
    id_instruction = (
        "Verify that the provided sources explicitly reference a Franklin Templeton ETF tied to XRP and confirm "
        "the launch timing in November 2025. Reject if the sources refer to a different issuer, product category "
        "(e.g., trust or ETP not explicitly the ETF), a different asset, or a different launch time."
    )

    # Leaf 2: Ticker Symbol (Critical)
    ticker_node = evaluator.add_leaf(
        id="Ticker_Symbol_As_Of_Nov_2025",
        desc="The provided ticker symbol matches the officially registered ticker for that ETF as of November 2025.",
        parent=ft_node,
        critical=True,
    )
    ticker_claim = (
        f"The official ticker symbol for Franklin Templeton’s XRP ETF is '{ticker}' as of November 2025."
    )
    ticker_instruction = (
        "Check the sources for a line such as 'Ticker' or explicit mentions of the ETF symbol. "
        "Minor formatting differences are acceptable (e.g., surrounding quotes or capitalization), "
        "but the core symbol must match exactly. If no reliable source shows the ticker or the sources contradict the claim, judge Incorrect."
    )

    # Leaf 3: Primary Listing Exchange (Critical)
    exchange_node = evaluator.add_leaf(
        id="Primary_Listing_Exchange_As_Of_Nov_2025",
        desc="The provided exchange matches the ETF's primary listing exchange as of November 2025.",
        parent=ft_node,
        critical=True,
    )
    exchange_claim = (
        f"The primary listing exchange for Franklin Templeton’s XRP ETF is '{primary_ex}' as of November 2025."
    )
    exchange_instruction = (
        "Verify in the sources which exchange the ETF is primarily listed on (e.g., NASDAQ, NYSE Arca, Cboe BZX). "
        "Allow minor naming variants (e.g., 'The Nasdaq Stock Market' vs 'NASDAQ'). "
        "If no clear statement exists or the sources conflict, judge Incorrect."
    )

    # Perform verifications (can run in parallel)
    await evaluator.batch_verify(
        [
            (identification_claim, sources, id_node, id_instruction),
            (ticker_claim, sources, ticker_node, ticker_instruction),
            (exchange_claim, sources, exchange_node, exchange_instruction),
        ]
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
    Evaluate an answer for Franklin Templeton's XRP ETF ticker and primary exchange as of November 2025.
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

    # Extract ETF info from the answer
    etf_info = await evaluator.extract(
        prompt=prompt_extract_etf_info(),
        template_class=ETFInfoExtraction,
        extraction_name="extracted_etf_info",
    )

    # Add custom info snapshot to summary for debugging
    evaluator.add_custom_info(
        info={
            "sponsor_name": etf_info.sponsor_name,
            "product_name": etf_info.product_name,
            "asset": etf_info.asset,
            "launch_month_year": etf_info.launch_month_year,
            "ticker": etf_info.ticker,
            "primary_exchange": etf_info.primary_exchange,
            "sources_count": len(etf_info.sources),
            "sources": etf_info.sources,
        },
        info_type="extraction_snapshot",
    )

    # Build and run verification tree
    await verify_ft_xrp_etf(evaluator, root, etf_info)

    # Return evaluation summary
    return evaluator.get_summary()