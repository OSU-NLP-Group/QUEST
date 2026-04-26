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
TASK_ID = "sp500_dividend_stocks_3_sectors"
TASK_DESCRIPTION = """
Identify three dividend-paying stocks from the S&P 500 index, with each stock coming from a different sector among Healthcare, Financials, and Consumer Staples. For each stock, provide the following information: (1) Company name and ticker symbol, (2) Sector classification (must be Healthcare, Financials, or Consumer Staples), (3) Market capitalization (must be at least $10 billion), (4) Dividend yield (Healthcare and Consumer Staples stocks must have yield above 2.0%; Financials stocks must have yield above 1.5%), (5) Debt-to-equity ratio (must be at or below 2.0), (6) Current ratio (must be at or above 1.0), (7) Reference URL (provide a link to a reputable financial data website showing the stock's key financial metrics). All three stocks must meet their respective sector-specific criteria, and each must be from a different sector. Ensure that all financial data is current as of February 2026.
"""

REQUIRED_SECTORS_ORDERED = ["Healthcare", "Financials", "Consumer Staples"]
DIVIDEND_THRESHOLDS = {
    "Healthcare": 2.0,
    "Financials": 1.5,
    "Consumer Staples": 2.0,
}


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class StockItem(BaseModel):
    company_name: Optional[str] = None
    ticker: Optional[str] = None
    sector: Optional[str] = None
    market_cap: Optional[str] = None
    dividend_yield: Optional[str] = None
    debt_to_equity: Optional[str] = None
    current_ratio: Optional[str] = None
    reference_url: Optional[str] = None


class StocksExtraction(BaseModel):
    stocks: List[StockItem] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_stocks() -> str:
    return """
    Extract up to the first three stocks described in the answer. For each stock, extract the following fields exactly as presented:
    - company_name: The company name.
    - ticker: The stock ticker symbol (without exchange prefix if present; e.g., use "JNJ", not "NYSE:JNJ").
    - sector: The sector classification stated in the answer (e.g., "Healthcare", "Financials", "Consumer Staples"). If the answer uses synonymous labels commonly used by finance sites (e.g., "Health Care" for "Healthcare", "Consumer Defensive" for "Consumer Staples", "Financial Services" for "Financials"), extract them as-is in sector.
    - market_cap: The market capitalization value string as given (e.g., "$370B", "$12.3B", "USD 370 billion").
    - dividend_yield: The dividend yield string as given (e.g., "2.7%", "1.8% (forward)").
    - debt_to_equity: The debt-to-equity ratio string as given (e.g., "0.45", "1.8x", "D/E 1.2").
    - current_ratio: The current ratio string as given (e.g., "1.2", "1.05x").
    - reference_url: A single URL to a reputable financial data webpage for this company (e.g., Yahoo Finance, Morningstar, MarketWatch, GuruFocus, companiesmarketcap.com, etc.). Extract the actual URL; if none is provided, set to null.

    Return a JSON object:
    {
      "stocks": [
        { "company_name": ..., "ticker": ..., "sector": ..., "market_cap": ..., "dividend_yield": ..., "debt_to_equity": ..., "current_ratio": ..., "reference_url": ... },
        ...
      ]
    }

    Rules:
    - Only extract information explicitly present in the answer. Do not invent any values.
    - If a field is missing for a stock, set it to null.
    - If more than three stocks are listed, include only the first three as they appear.
    """


# --------------------------------------------------------------------------- #
# Helper: Build additional instructions per check                             #
# --------------------------------------------------------------------------- #
def addins_sector(required_sector: str) -> str:
    # Allow common synonyms that many sites use
    synonyms = {
        "Healthcare": ["Healthcare", "Health Care"],
        "Financials": ["Financials", "Financial", "Financial Services"],
        "Consumer Staples": ["Consumer Staples", "Consumer Defensive"],
    }
    allowed = ", ".join(synonyms.get(required_sector, [required_sector]))
    return (
        f"Verify that the page indicates the company's sector matches the required sector '{required_sector}'. "
        f"Treat common synonymous labels as equivalent: {allowed} are equivalent to '{required_sector}'. "
        f"Focus on the sector classification on this page. Use only this page’s content."
    )


def addins_market_cap() -> str:
    return (
        "Verify the company's market capitalization on this page is at least $10 billion. "
        "Treat formatting variants equivalently (e.g., '$10B', '$10 billion', 'USD 10B'). "
        "Use the current value shown on the page (as-of February 2026 or latest available on the page)."
    )


def addins_dividend(threshold: float) -> str:
    return (
        f"Verify the page shows that the company pays dividends and that the dividend yield is strictly above {threshold}%. "
        "Accept either trailing or forward dividend yield if presented; if multiple are shown, any one above the threshold suffices. "
        "If yield is absent or listed as 0%, the claim is not supported."
    )


def addins_debt_to_equity() -> str:
    return (
        "Verify the page shows a debt-to-equity (D/E) ratio, and it is at or below 2.0. "
        "Treat variations like '1.8', '1.8x', or 'D/E 1.8' equivalently. "
        "If D/E is not available or exceeds 2.0, the claim is not supported."
    )


def addins_current_ratio() -> str:
    return (
        "Verify the page shows a current ratio (current assets / current liabilities) at or above 1.0. "
        "Treat values like '1.0', '1.05', or '1.05x' equivalently. "
        "If the current ratio is missing or below 1.0, the claim is not supported."
    )


# --------------------------------------------------------------------------- #
# Verification per stock                                                      #
# --------------------------------------------------------------------------- #
async def verify_single_stock(
    evaluator: Evaluator,
    parent_node,
    stock: StockItem,
    required_sector: str,
    stock_node_id: str,
    node_prefix: str,
) -> None:
    """
    Build and verify the subtree for a single stock based on the rubric.
    - parent_node: parent parallel node for this stock.
    - stock_node_id: e.g., "Stock_1_Healthcare"
    - node_prefix: e.g., "Stock_1"
    """

    # Create the stock group node (parallel aggregation, non-critical as per JSON)
    stock_group = evaluator.add_parallel(
        id=stock_node_id,
        desc=f"{stock_node_id.replace('_', ' ')} verification group",
        parent=parent_node,
        critical=False
    )

    # Existence: Identification (company name + ticker)
    identification_ok = bool(stock and stock.company_name and stock.company_name.strip()
                             and stock.ticker and stock.ticker.strip())
    evaluator.add_custom_node(
        result=identification_ok,
        id=f"{node_prefix}_Identification",
        desc=f"Company name and ticker symbol are provided for {node_prefix.replace('_', ' ')}",
        parent=stock_group,
        critical=True
    )

    # Existence: Reference URL
    ref_ok = bool(stock and stock.reference_url and stock.reference_url.strip())
    evaluator.add_custom_node(
        result=ref_ok,
        id=f"{node_prefix}_Reference",
        desc=f"Valid reference URL provided for {node_prefix.replace('_', ' ')} financial data",
        parent=stock_group,
        critical=True
    )

    # Sector check (critical)
    sector_node = evaluator.add_leaf(
        id=f"{node_prefix}_Sector",
        desc=f"Stock is from the {required_sector} sector",
        parent=stock_group,
        critical=True
    )
    comp = stock.company_name or "the company"
    tick = stock.ticker or "[ticker missing]"
    sector_claim = (
        f"{comp} (ticker {tick}) is classified in the {required_sector} sector."
    )
    await evaluator.verify(
        claim=sector_claim,
        node=sector_node,
        sources=stock.reference_url,
        additional_instruction=addins_sector(required_sector)
    )

    # Market Cap >= $10B (critical)
    mcap_node = evaluator.add_leaf(
        id=f"{node_prefix}_Market_Cap",
        desc="Stock has market capitalization of at least $10 billion",
        parent=stock_group,
        critical=True
    )
    mcap_claim = (
        f"{comp} (ticker {tick}) has a market capitalization of at least $10 billion."
    )
    await evaluator.verify(
        claim=mcap_claim,
        node=mcap_node,
        sources=stock.reference_url,
        additional_instruction=addins_market_cap()
    )

    # Dividend yield above threshold (critical)
    required_yield = DIVIDEND_THRESHOLDS.get(required_sector, 0.0)
    div_node = evaluator.add_leaf(
        id=f"{node_prefix}_Dividend",
        desc=f"Stock pays dividends with yield above {required_yield}%",
        parent=stock_group,
        critical=True
    )
    div_claim = (
        f"{comp} (ticker {tick}) pays dividends and its dividend yield is above {required_yield}%."
    )
    await evaluator.verify(
        claim=div_claim,
        node=div_node,
        sources=stock.reference_url,
        additional_instruction=addins_dividend(required_yield)
    )

    # Debt-to-equity <= 2.0 (critical)
    de_node = evaluator.add_leaf(
        id=f"{node_prefix}_Debt_Equity",
        desc="Stock has debt-to-equity ratio at or below 2.0",
        parent=stock_group,
        critical=True
    )
    de_claim = (
        f"{comp} (ticker {tick}) has a debt-to-equity ratio at or below 2.0."
    )
    await evaluator.verify(
        claim=de_claim,
        node=de_node,
        sources=stock.reference_url,
        additional_instruction=addins_debt_to_equity()
    )

    # Current ratio >= 1.0 (critical)
    cr_node = evaluator.add_leaf(
        id=f"{node_prefix}_Current_Ratio",
        desc="Stock has current ratio at or above 1.0",
        parent=stock_group,
        critical=True
    )
    cr_claim = (
        f"{comp} (ticker {tick}) has a current ratio at or above 1.0."
    )
    await evaluator.verify(
        claim=cr_claim,
        node=cr_node,
        sources=stock.reference_url,
        additional_instruction=addins_current_ratio()
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
    Evaluate an answer for the S&P 500 dividend-paying stocks across three sectors (Healthcare, Financials, Consumer Staples).
    """
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,   # Root: parallel aggregation
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

    # IMPORTANT: Make root non-critical to allow partial credit even if some stocks fail.
    root.critical = False

    # Extract up to 3 stocks from the answer (first three as presented)
    extracted = await evaluator.extract(
        prompt=prompt_extract_stocks(),
        template_class=StocksExtraction,
        extraction_name="stocks_extraction"
    )

    # Normalize to exactly 3 items (pad with empty if needed)
    stocks_list: List[StockItem] = list(extracted.stocks[:3])
    while len(stocks_list) < 3:
        stocks_list.append(StockItem())

    # Add a brief requirements summary to the report
    evaluator.add_custom_info(
        info={
            "required_sectors": REQUIRED_SECTORS_ORDERED,
            "market_cap_min": "$10B",
            "dividend_yield_thresholds": DIVIDEND_THRESHOLDS,
            "de_max": 2.0,
            "current_ratio_min": 1.0,
            "as_of": "February 2026",
            "note": "Only the first three stocks in the answer are evaluated."
        },
        info_type="requirements",
        info_name="evaluation_requirements"
    )

    # Build verification subtrees for each required sector in fixed order
    # Stock 1 -> Healthcare, Stock 2 -> Financials, Stock 3 -> Consumer Staples
    sector_assignments = [
        ("Stock_1_Healthcare", "Stock_1", "Healthcare"),
        ("Stock_2_Financials", "Stock_2", "Financials"),
        ("Stock_3_Consumer_Staples", "Stock_3", "Consumer Staples"),
    ]

    for idx, (stock_node_id, node_prefix, required_sector) in enumerate(sector_assignments):
        stock_item = stocks_list[idx] if idx < len(stocks_list) else StockItem()
        await verify_single_stock(
            evaluator=evaluator,
            parent_node=root,
            stock=stock_item,
            required_sector=required_sector,
            stock_node_id=stock_node_id,
            node_prefix=node_prefix
        )

    return evaluator.get_summary()