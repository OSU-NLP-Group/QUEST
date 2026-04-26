import asyncio
import logging
from typing import Any, List, Optional, Dict

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "sp500_dividend_four_stocks"
TASK_DESCRIPTION = """
Identify four stocks currently in the S&P 500 index that meet the following criteria:
(1) each stock must have a current dividend yield above 4.0%, significantly exceeding the S&P 500 average dividend yield of 1.191% as of March 2026,
(2) each company must have a demonstrated track record of at least 20 consecutive years of annual dividend increases, and
(3) the four stocks must collectively represent at least two different GICS (Global Industry Classification Standard) sectors.
For each stock, provide its ticker symbol, current dividend yield (as a percentage), the number of consecutive years of dividend increases, its GICS sector, and a reference URL that verifies the dividend information.
"""

SP500_AVG_YIELD_MAR_2026 = "1.191%"
DIVIDEND_YIELD_THRESHOLD = 4.0  # percent


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class StockItem(BaseModel):
    ticker: Optional[str] = None
    company_name: Optional[str] = None
    dividend_yield_percent: Optional[str] = None  # keep as string for flexibility (e.g., "4.3%" or "4.3")
    consecutive_years_increase: Optional[str] = None  # keep as string (e.g., "25", "25+")
    gics_sector: Optional[str] = None
    source_urls: List[str] = Field(default_factory=list)


class StocksExtraction(BaseModel):
    stocks: List[StockItem] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_stocks() -> str:
    return """
    From the answer, extract details for up to four stocks (in the original order they appear).
    For each stock, extract the following fields exactly as stated in the answer:
    - ticker: the stock ticker symbol (e.g., "T", "MO", "XOM")
    - company_name: the company name if provided (else null)
    - dividend_yield_percent: the stated current dividend yield percentage value (keep any % sign or decimals if present)
    - consecutive_years_increase: the number of consecutive years of dividend increases (e.g., "20", "25+", "50"), as a string
    - gics_sector: the GICS sector as stated (e.g., "Utilities", "Energy", "Consumer Staples"); keep the text as given
    - source_urls: list all URLs explicitly mentioned for this stock (include any link for dividend info, streak/years evidence, sector, or S&P 500 membership if present). Extract actual URLs from plain text or markdown links.
    
    Rules:
    - Do not infer or create data that is not explicitly in the answer text.
    - If a field is missing for a stock, set it to null (except source_urls which should be an empty list if none).
    - Return at most 4 stocks in the 'stocks' array.
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def display_name(stock: StockItem) -> str:
    if stock.company_name and stock.company_name.strip():
        return f"{stock.company_name} ({stock.ticker})" if stock.ticker else stock.company_name
    return f"ticker {stock.ticker}" if stock.ticker else "the company"

def normalize_sector_name(sector: Optional[str]) -> Optional[str]:
    if not sector:
        return None
    s = sector.strip().lower()
    mapping = {
        "communication services": "Communication Services",
        "communications services": "Communication Services",
        "comm services": "Communication Services",
        "telecommunication services": "Communication Services",  # legacy
        "consumer discretionary": "Consumer Discretionary",
        "consumer cyclical": "Consumer Discretionary",
        "consumer staples": "Consumer Staples",
        "consumer defensive": "Consumer Staples",
        "information technology": "Information Technology",
        "technology": "Information Technology",
        "tech": "Information Technology",
        "health care": "Health Care",
        "healthcare": "Health Care",
        "financials": "Financials",
        "financial": "Financials",
        "industrials": "Industrials",
        "industrial": "Industrials",
        "energy": "Energy",
        "materials": "Materials",
        "utilities": "Utilities",
        "real estate": "Real Estate",
        "reit": "Real Estate",
        "reits": "Real Estate",
        "real estate investment trusts": "Real Estate",
    }
    return mapping.get(s, sector.strip())


# --------------------------------------------------------------------------- #
# Verification logic per stock                                                #
# --------------------------------------------------------------------------- #
async def verify_single_stock(
    evaluator: Evaluator,
    parent_node,
    stock: StockItem,
    stock_index: int,
) -> None:
    """
    Build verification subtree for a single stock.

    Structure:
    stock_{i} (sequential)
      ├─ stock_{i}_required_info (custom, critical)
      └─ stock_{i}_constraints (parallel, critical)
           ├─ stock_{i}_sp500_member (leaf, critical)
           ├─ stock_{i}_ticker_supported (leaf, critical)
           ├─ stock_{i}_yield_value_supported (leaf, critical)
           ├─ stock_{i}_yield_above_4 (leaf, critical)
           ├─ stock_{i}_streak_20plus (leaf, critical)
           └─ stock_{i}_sector_supported (leaf, critical)
    """
    i = stock_index
    stock_node = evaluator.add_sequential(
        id=f"stock_{i}",
        desc=(
            f"Stock #{i+1}: Must be an S&P 500 member, dividend yield > {DIVIDEND_YIELD_THRESHOLD}%, "
            f"≥20 consecutive years of increases, with GICS sector provided and a reference URL."
        ),
        parent=parent_node,
        critical=False,  # each stock contributes to overall, but not all-or-nothing for the whole task
    )

    # Required info presence check (critical)
    has_ticker = bool(stock.ticker and stock.ticker.strip())
    has_yield = bool(stock.dividend_yield_percent and stock.dividend_yield_percent.strip())
    has_years = bool(stock.consecutive_years_increase and stock.consecutive_years_increase.strip())
    has_sector = bool(stock.gics_sector and stock.gics_sector.strip())
    has_any_url = bool(stock.source_urls and len(stock.source_urls) > 0)

    evaluator.add_custom_node(
        result=(has_ticker and has_yield and has_years and has_sector and has_any_url),
        id=f"stock_{i}_required_info",
        desc=f"Stock #{i+1} has required fields (ticker, yield%, years, sector) and at least one reference URL",
        parent=stock_node,
        critical=True,
    )

    # Constraints node, all children under it must be critical (per rubric)
    constraints_node = evaluator.add_parallel(
        id=f"stock_{i}_constraints",
        desc=f"Stock #{i+1} must satisfy: S&P 500 member, yield > {DIVIDEND_YIELD_THRESHOLD}%, ≥20-yr streak, and correct sector/ticker",
        parent=stock_node,
        critical=True,  # this makes all of its children also required to be critical
    )

    urls = stock.source_urls if stock.source_urls else []
    ticker = stock.ticker or ""
    comp_disp = display_name(stock)
    yield_str = stock.dividend_yield_percent or ""
    years_str = stock.consecutive_years_increase or ""
    sector_str = stock.gics_sector or ""

    # Create leaf nodes
    n_sp500 = evaluator.add_leaf(
        id=f"stock_{i}_sp500_member",
        desc=f"Stock #{i+1}: Is a current S&P 500 constituent",
        parent=constraints_node,
        critical=True,
    )
    n_ticker = evaluator.add_leaf(
        id=f"stock_{i}_ticker_supported",
        desc=f"Stock #{i+1}: Ticker symbol is correctly identified as {ticker}",
        parent=constraints_node,
        critical=True,
    )
    n_yield_val = evaluator.add_leaf(
        id=f"stock_{i}_yield_value_supported",
        desc=f"Stock #{i+1}: Dividend yield value is supported by the cited source(s)",
        parent=constraints_node,
        critical=True,
    )
    n_yield_thr = evaluator.add_leaf(
        id=f"stock_{i}_yield_above_4",
        desc=f"Stock #{i+1}: Dividend yield exceeds {DIVIDEND_YIELD_THRESHOLD}%",
        parent=constraints_node,
        critical=True,
    )
    n_streak = evaluator.add_leaf(
        id=f"stock_{i}_streak_20plus",
        desc=f"Stock #{i+1}: At least 20 consecutive years of dividend increases are supported",
        parent=constraints_node,
        critical=True,
    )
    n_sector = evaluator.add_leaf(
        id=f"stock_{i}_sector_supported",
        desc=f"Stock #{i+1}: GICS sector '{sector_str}' is supported by the cited source(s)",
        parent=constraints_node,
        critical=True,
    )

    # Build verification tasks (batch)
    verify_tasks: List[tuple] = []

    claim_sp500 = (
        f"The company corresponding to ticker '{ticker}' is a current constituent of the S&P 500 index."
    )
    add_ins_sp500 = (
        "Look for explicit indications on the provided page(s), such as 'S&P 500', "
        "'S&P 500 constituent/component', or inclusion in lists like 'S&P 500 Dividend Aristocrats'. "
        "Treat 'S&P 500 Dividend Aristocrats' as confirming S&P 500 membership. "
        "If the provided source(s) do not mention S&P 500 membership at all, consider the claim not supported."
    )
    verify_tasks.append((claim_sp500, urls, n_sp500, add_ins_sp500))

    claim_ticker = (
        f"The stock's ticker symbol for {comp_disp} is '{ticker}', allowing for minor formatting variants "
        "(e.g., 'BRK.B' vs 'BRK-B', or prefixes like 'NYSE:'/'NASDAQ:')."
    )
    add_ins_ticker = (
        "Check the page header or primary identification area for the ticker. "
        "Allow insignificant formatting differences (e.g., '.' vs '-' in class tickers; exchange prefixes)."
    )
    verify_tasks.append((claim_ticker, urls, n_ticker, add_ins_ticker))

    claim_yield_val = (
        f"The current dividend yield for {comp_disp} is approximately {yield_str}."
    )
    add_ins_yield_val = (
        "Verify the page shows a current dividend yield approximately matching the stated value. "
        "Allow rounding differences up to about ±0.2 percentage points and slight day-to-day variation."
    )
    verify_tasks.append((claim_yield_val, urls, n_yield_val, add_ins_yield_val))

    claim_yield_thr = (
        f"The current dividend yield for {comp_disp} exceeds {DIVIDEND_YIELD_THRESHOLD}% "
        f"and therefore is well above the S&P 500 average of {SP500_AVG_YIELD_MAR_2026} (Mar 2026)."
    )
    add_ins_yield_thr = (
        f"Check the yield on the page and judge whether it is > {DIVIDEND_YIELD_THRESHOLD}%. "
        f"You do not need the page to mention the S&P 500 average; just confirm the yield exceeds the threshold."
    )
    verify_tasks.append((claim_yield_thr, urls, n_yield_thr, add_ins_yield_thr))

    claim_streak = (
        f"{comp_disp} has at least 20 consecutive years of annual dividend increases."
    )
    add_ins_streak = (
        "Accept evidence such as explicit '20+ years', '25+ years', 'Dividend Aristocrat' (≥25 years) "
        "or 'Dividend King' (≥50 years). The page must clearly indicate the streak length or status implying ≥20 years."
    )
    verify_tasks.append((claim_streak, urls, n_streak, add_ins_streak))

    claim_sector = (
        f"The GICS sector for {comp_disp} is '{sector_str}', allowing for standard GICS naming variations."
    )
    add_ins_sector = (
        "Verify sector classification on the provided source(s). Allow common synonyms/mappings, e.g., "
        "'Consumer Defensive' -> 'Consumer Staples', 'Consumer Cyclical' -> 'Consumer Discretionary', "
        "'Technology' -> 'Information Technology', 'REITs' -> 'Real Estate', 'Healthcare' -> 'Health Care', "
        "'Communication/Communications Services' -> 'Communication Services'."
    )
    verify_tasks.append((claim_sector, urls, n_sector, add_ins_sector))

    # Execute all verifications (they will auto-skip if the required_info node fails due to sequential gating)
    await evaluator.batch_verify(verify_tasks)


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
    Evaluate an answer for the S&P 500 high-dividend stock task.
    """
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Root aggregates four independent stocks + diversification check
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

    # Record contextual ground truth information (for transparency only)
    evaluator.add_ground_truth({
        "sp500_average_yield_march_2026": SP500_AVG_YIELD_MAR_2026,
        "required_min_yield_percent": f"{DIVIDEND_YIELD_THRESHOLD}%",
        "required_min_consecutive_years_increase": "20",
        "min_unique_sectors": 2,
        "required_stock_count": 4
    })

    # Extract structured stock info from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_stocks(),
        template_class=StocksExtraction,
        extraction_name="stocks_extraction",
    )

    # Ensure we consider exactly the first 4 stocks (pad with empty if fewer)
    stocks: List[StockItem] = list(extracted.stocks[:4])
    while len(stocks) < 4:
        stocks.append(StockItem())

    # Build per-stock verification subtrees
    for idx, stock in enumerate(stocks):
        await verify_single_stock(evaluator, root, stock, idx)

    # Sector diversification (critical): at least two different sectors across the four provided stocks
    sectors_norm = [
        normalize_sector_name(s.gics_sector) for s in stocks if s.gics_sector and s.gics_sector.strip()
    ]
    unique_sectors = set(filter(None, sectors_norm))
    evaluator.add_custom_node(
        result=(len(unique_sectors) >= 2),
        id="sector_diversification",
        desc="The four stocks collectively represent at least two different GICS sectors",
        parent=root,
        critical=True,  # Critical as per rubric
    )

    # Return evaluation summary
    return evaluator.get_summary()