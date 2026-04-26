import asyncio
import logging
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys

# -----------------------------------------------------------------------------
# Task constants
# -----------------------------------------------------------------------------
TASK_ID = "sp500_dividend_aristocrats_duo_2026"
TASK_DESCRIPTION = """
Identify two distinct S&P 500 Dividend Aristocrat stocks that meet ALL of the following criteria as of March 19, 2026:

1. S&P 500 Membership: The stock must be a current member of the S&P 500 index.
2. Dividend Aristocrat Status: The company must have increased its dividend payments for at least 25 consecutive years.
3. Dividend Yield: The current dividend yield must be at least 2.5%.
4. Payment Frequency: The company must pay dividends on a quarterly basis (four times per year).
5. Upcoming Ex-Dividend Date: The next ex-dividend date must fall within the next 90 days (by June 17, 2026).
6. Institutional Ownership: At least one institutional investor holding must be verifiable through publicly available Form 13F filings or institutional holdings databases.
7. Recent Earnings Report: The company must have reported quarterly earnings within the last 60 days (since January 18, 2026).
8. Analyst Consensus Rating: The stock must have a consensus analyst rating of "Buy" or better (numerical rating of 2.0 or lower on a standard 1-5 scale, where 1 is Strong Buy and 5 is Strong Sell).
9. Market Capitalization: The company must have a market capitalization of at least $50 billion.
10. Year-to-Date Performance: The stock must show a positive year-to-date price return as of March 19, 2026.

For each stock, provide:
- Stock ticker symbol and company name
- Current dividend yield with source URL
- Date of the company's most recent earnings report with source URL
- Verification that it is a Dividend Aristocrat with 25+ years of consecutive dividend increases, with source URL
- Next ex-dividend date with source URL
- Evidence of institutional ownership with source URL
- Current market capitalization with source URL
- Year-to-date performance percentage with source URL
- Analyst consensus rating with source URL
"""

# Reference dates for temporal checks
AS_OF_DATE = "March 19, 2026"
EXDIV_DEADLINE = "June 17, 2026"   # 90 days from March 19, 2026
EARNINGS_WINDOW_START = "January 18, 2026"  # 60 days prior to March 19, 2026

# -----------------------------------------------------------------------------
# Extraction data models
# -----------------------------------------------------------------------------
class StockItem(BaseModel):
    ticker: Optional[str] = None
    company_name: Optional[str] = None

    sp500_membership_urls: List[str] = Field(default_factory=list)
    aristocrat_status_urls: List[str] = Field(default_factory=list)

    dividend_yield: Optional[str] = None
    dividend_yield_urls: List[str] = Field(default_factory=list)

    payment_frequency_text: Optional[str] = None
    payment_frequency_urls: List[str] = Field(default_factory=list)

    next_ex_dividend_date: Optional[str] = None
    ex_dividend_urls: List[str] = Field(default_factory=list)

    institutional_holders: List[str] = Field(default_factory=list)
    institutional_urls: List[str] = Field(default_factory=list)

    recent_earnings_date: Optional[str] = None
    earnings_urls: List[str] = Field(default_factory=list)

    analyst_rating_value: Optional[str] = None
    analyst_rating_urls: List[str] = Field(default_factory=list)

    market_cap_value: Optional[str] = None
    market_cap_urls: List[str] = Field(default_factory=list)

    ytd_return_value: Optional[str] = None
    ytd_urls: List[str] = Field(default_factory=list)


class TwoStocksExtraction(BaseModel):
    stocks: List[StockItem] = Field(default_factory=list)


# -----------------------------------------------------------------------------
# Extraction prompt
# -----------------------------------------------------------------------------
def prompt_extract_two_stocks() -> str:
    return f"""
You must extract structured information for up to two distinct stocks mentioned in the answer that the agent claims satisfy all criteria as of {AS_OF_DATE}.
Return a JSON object with a top-level array field named "stocks". Include at most the first two distinct stocks the answer presents.

For each stock, extract the following fields (use strings for values; URLs must come from the answer text):
- ticker: Stock ticker symbol (string)
- company_name: Full company name (string)

- sp500_membership_urls: Array of URL(s) the answer cites as evidence that the company is currently in the S&P 500
- aristocrat_status_urls: Array of URL(s) that verify 25+ consecutive years of dividend increases (Dividend Aristocrat/Champion/etc.)

- dividend_yield: The current dividend yield as presented (string, e.g., "2.7%" or "2.70%")
- dividend_yield_urls: Array of URL(s) for the dividend yield

- payment_frequency_text: The dividend payment frequency as presented (e.g., "quarterly")
- payment_frequency_urls: Array of URL(s) that show the dividend payment frequency

- next_ex_dividend_date: The next ex-dividend date as presented (string, e.g., "April 15, 2026")
- ex_dividend_urls: Array of URL(s) for the next ex-dividend date

- institutional_holders: Array of at least one named institutional holder if provided (e.g., "Vanguard", "BlackRock"); leave empty if not explicitly named
- institutional_urls: Array of URL(s) that show institutional holdings or Form 13F evidence

- recent_earnings_date: The date of the most recent quarterly earnings report as presented (string)
- earnings_urls: Array of URL(s) for the earnings report date

- analyst_rating_value: The consensus analyst rating as presented (string; accept either a numeric 1–5 score or a text like "Buy", "Strong Buy")
- analyst_rating_urls: Array of URL(s) that show the analyst consensus

- market_cap_value: The current market capitalization as presented (string, e.g., "$180B", "$0.20 trillion")
- market_cap_urls: Array of URL(s) for market capitalization

- ytd_return_value: The year-to-date price return as presented (string, e.g., "+8.2%", "5.1%")
- ytd_urls: Array of URL(s) that show YTD performance

RULES:
- Extract only what is explicitly present in the answer. Do not invent any values or URLs.
- Each URL must be a valid URL string and must be directly present in the answer (including markdown links). If none are provided for a field, return an empty array for that field.
- If a value (like a date or percentage) isn’t provided in the answer, set that field to null.
- Prefer official or reputable sources if multiple URLs are listed in the answer (but still extract all explicitly cited URLs).
- Ensure stocks are distinct by ticker if the answer lists more than one; keep their order of appearance.
"""


# -----------------------------------------------------------------------------
# Helper utilities
# -----------------------------------------------------------------------------
def _has_urls(urls: Optional[List[str]]) -> bool:
    return bool(urls) and len([u for u in urls if isinstance(u, str) and u.strip()]) > 0


def _nonnull_str(s: Optional[str]) -> bool:
    return isinstance(s, str) and s.strip() != ""


def _label(i: int) -> str:
    return "First_Stock" if i == 0 else "Second_Stock"


def _stock_display(stock: StockItem) -> str:
    t = stock.ticker or ""
    n = stock.company_name or ""
    if t and n:
        return f"{n} ({t})"
    return t or n or "the company"


# -----------------------------------------------------------------------------
# Verification logic per stock (build subtree and run checks)
# -----------------------------------------------------------------------------
async def verify_one_stock(
    evaluator: Evaluator,
    parent_node,
    stock: StockItem,
    stock_index: int,
) -> None:
    """
    Build the verification subtree for one stock and execute all verifications.
    Mirrors the rubric hierarchy with explicit binary leaf checks.
    """
    stock_node = evaluator.add_parallel(
        id=_label(stock_index),
        desc=("First qualifying S&P 500 Dividend Aristocrat stock" if stock_index == 0
              else "Second qualifying S&P 500 Dividend Aristocrat stock (distinct from the first)"),
        parent=parent_node,
        critical=False,
    )

    # -------------------- Basic_Eligibility (critical) --------------------
    basic_node = evaluator.add_parallel(
        id=f"stock{stock_index}_Basic_Eligibility",
        desc="Stock meets fundamental index membership and dividend aristocrat requirements",
        parent=stock_node,
        critical=True,
    )

    # SP500_Membership subgroup
    sp500_node = evaluator.add_parallel(
        id=f"stock{stock_index}_SP500_Membership",
        desc="The stock is currently a member of the S&P 500 index",
        parent=basic_node,
        critical=True,
    )
    sp500_src_present = evaluator.add_custom_node(
        result=_has_urls(stock.sp500_membership_urls),
        id=f"stock{stock_index}_SP500_Membership_Source_Present",
        desc="At least one S&P 500 membership source URL is provided",
        parent=sp500_node,
        critical=True,
    )
    sp500_verify_leaf = evaluator.add_leaf(
        id=f"stock{stock_index}_SP500_Membership_Reference",
        desc="URL reference verifying S&P 500 membership",
        parent=sp500_node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"{_stock_display(stock)} is a current member of the S&P 500 index.",
        node=sp500_verify_leaf,
        sources=stock.sp500_membership_urls,
        additional_instruction="Accept official S&P Dow Jones Indices, S&P 500 company list (e.g., Wikipedia 'List of S&P 500 companies'), or other reputable sources that explicitly indicate S&P 500 membership as of the present context.",
    )

    # Dividend_Aristocrat_Status subgroup
    arist_node = evaluator.add_parallel(
        id=f"stock{stock_index}_Dividend_Aristocrat_Status",
        desc="The company has increased its dividend for at least 25 consecutive years",
        parent=basic_node,
        critical=True,
    )
    arist_src_present = evaluator.add_custom_node(
        result=_has_urls(stock.aristocrat_status_urls),
        id=f"stock{stock_index}_Dividend_Aristocrat_Source_Present",
        desc="At least one Dividend Aristocrat/25+ years source URL is provided",
        parent=arist_node,
        critical=True,
    )
    arist_verify_leaf = evaluator.add_leaf(
        id=f"stock{stock_index}_Dividend_History_Reference",
        desc="URL reference verifying 25+ years of consecutive dividend increases",
        parent=arist_node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"{_stock_display(stock)} has increased its dividend for at least 25 consecutive years (Dividend Aristocrat).",
        node=arist_verify_leaf,
        sources=stock.aristocrat_status_urls,
        additional_instruction="Look for explicit mentions of 'Dividend Aristocrat', '25+ years of dividend increases', or similar. It's acceptable if the source indicates 50+ years (Dividend King), as that implies ≥25.",
    )

    # -------------------- Dividend_Requirements (critical) ----------------
    div_node = evaluator.add_parallel(
        id=f"stock{stock_index}_Dividend_Requirements",
        desc="Stock meets all dividend-related criteria",
        parent=stock_node,
        critical=True,
    )

    # Dividend_Yield
    dy_node = evaluator.add_parallel(
        id=f"stock{stock_index}_Dividend_Yield",
        desc="Current dividend yield is at least 2.5%",
        parent=div_node,
        critical=True,
    )
    dy_src_present = evaluator.add_custom_node(
        result=_has_urls(stock.dividend_yield_urls),
        id=f"stock{stock_index}_Dividend_Yield_Source_Present",
        desc="URL reference for current dividend yield is provided",
        parent=dy_node,
        critical=True,
    )
    dy_verify_leaf = evaluator.add_leaf(
        id=f"stock{stock_index}_Dividend_Yield_Reference",
        desc="URL reference showing current dividend yield",
        parent=dy_node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"The current dividend yield for {_stock_display(stock)} is '{stock.dividend_yield}', and it is at least 2.5%.",
        node=dy_verify_leaf,
        sources=stock.dividend_yield_urls,
        additional_instruction="Confirm the page shows a dividend yield value ≥ 2.5%. Allow small rounding differences (e.g., 2.49% vs 2.5%). Prefer the 'current' yield shown.",
    )

    # Payment_Frequency
    pf_node = evaluator.add_parallel(
        id=f"stock{stock_index}_Payment_Frequency",
        desc="Company pays dividends quarterly (four times per year)",
        parent=div_node,
        critical=True,
    )
    pf_src_present = evaluator.add_custom_node(
        result=_has_urls(stock.payment_frequency_urls),
        id=f"stock{stock_index}_Payment_Frequency_Source_Present",
        desc="URL reference for dividend payment frequency is provided",
        parent=pf_node,
        critical=True,
    )
    pf_verify_leaf = evaluator.add_leaf(
        id=f"stock{stock_index}_Payment_Frequency_Reference",
        desc="URL reference showing dividend payment frequency",
        parent=pf_node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"{_stock_display(stock)} pays dividends quarterly (four times per year).",
        node=pf_verify_leaf,
        sources=stock.payment_frequency_urls,
        additional_instruction="Confirm the source shows quarterly payments (or 'four times per year'). Pages like investor relations dividend policies or dividend history calendars are suitable.",
    )

    # Next_ExDividend_Date
    exd_node = evaluator.add_parallel(
        id=f"stock{stock_index}_Next_ExDividend_Date",
        desc=f"Next ex-dividend date is within 90 days from {AS_OF_DATE}",
        parent=div_node,
        critical=True,
    )
    exd_src_present = evaluator.add_custom_node(
        result=_has_urls(stock.ex_dividend_urls),
        id=f"stock{stock_index}_ExDividend_Date_Source_Present",
        desc="URL reference for the next ex-dividend date is provided",
        parent=exd_node,
        critical=True,
    )
    exd_verify_leaf = evaluator.add_leaf(
        id=f"stock{stock_index}_ExDividend_Date_Reference",
        desc="URL reference showing the next ex-dividend date",
        parent=exd_node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"The next ex-dividend date for {_stock_display(stock)} is '{stock.next_ex_dividend_date}', and it falls between {AS_OF_DATE} and {EXDIV_DEADLINE} (i.e., within the next 90 days).",
        node=exd_verify_leaf,
        sources=stock.ex_dividend_urls,
        additional_instruction=f"Check the upcoming ex-dividend date and ensure it lies within the window [{AS_OF_DATE}, {EXDIV_DEADLINE}].",
    )

    # -------------------- Institutional_Ownership (critical) --------------
    inst_node = evaluator.add_parallel(
        id=f"stock{stock_index}_Institutional_Ownership",
        desc="At least one institutional investor holding is verifiable through 13F filings",
        parent=stock_node,
        critical=True,
    )
    inst_src_present = evaluator.add_custom_node(
        result=_has_urls(stock.institutional_urls),
        id=f"stock{stock_index}_Institutional_Ownership_Source_Present",
        desc="URL reference showing institutional ownership data is provided",
        parent=inst_node,
        critical=True,
    )
    inst_verify_leaf = evaluator.add_leaf(
        id=f"stock{stock_index}_Institutional_Holding_Reference",
        desc="URL reference showing institutional ownership data",
        parent=inst_node,
        critical=True,
    )
    inst_holder_example = stock.institutional_holders[0] if stock.institutional_holders else "an institutional investor"
    await evaluator.verify(
        claim=f"There is at least one institutional investor (e.g., {inst_holder_example}) reported to hold shares of {_stock_display(stock)} via 13F filings or a reputable institutional holdings database.",
        node=inst_verify_leaf,
        sources=stock.institutional_urls,
        additional_instruction="Accept SEC 13F sources or reputable holdings aggregators (e.g., Nasdaq, WhaleWisdom, Holdings Channel, etc.) that explicitly list institutional holders and positions.",
    )

    # -------------------- Recent_Financial_Activity (critical) ------------
    rfa_node = evaluator.add_parallel(
        id=f"stock{stock_index}_Recent_Financial_Activity",
        desc="Stock has recent earnings reporting activity",
        parent=stock_node,
        critical=True,
    )

    # Recent_Earnings
    earn_node = evaluator.add_parallel(
        id=f"stock{stock_index}_Recent_Earnings",
        desc=f"Company reported earnings within the last 60 days from {AS_OF_DATE}",
        parent=rfa_node,
        critical=True,
    )
    earn_src_present = evaluator.add_custom_node(
        result=_has_urls(stock.earnings_urls),
        id=f"stock{stock_index}_Earnings_Source_Present",
        desc="URL reference for recent earnings report date is provided",
        parent=earn_node,
        critical=True,
    )
    earn_verify_leaf = evaluator.add_leaf(
        id=f"stock{stock_index}_Earnings_Date_Reference",
        desc="URL reference showing recent earnings report date",
        parent=earn_node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"The most recent quarterly earnings report for {_stock_display(stock)} occurred on '{stock.recent_earnings_date}', which is within 60 days prior to {AS_OF_DATE} (i.e., on or after {EARNINGS_WINDOW_START}).",
        node=earn_verify_leaf,
        sources=stock.earnings_urls,
        additional_instruction="Verify the reported earnings date and ensure it falls within the stated 60-day window.",
    )

    # Analyst_Rating
    rating_node = evaluator.add_parallel(
        id=f"stock{stock_index}_Analyst_Rating",
        desc="Stock has a consensus analyst rating of Buy or better (≤2.0 on 1-5 scale)",
        parent=rfa_node,
        critical=True,
    )
    rating_src_present = evaluator.add_custom_node(
        result=_has_urls(stock.analyst_rating_urls),
        id=f"stock{stock_index}_Analyst_Rating_Source_Present",
        desc="URL reference showing analyst consensus rating is provided",
        parent=rating_node,
        critical=True,
    )
    rating_verify_leaf = evaluator.add_leaf(
        id=f"stock{stock_index}_Analyst_Rating_Reference",
        desc="URL reference showing analyst consensus rating",
        parent=rating_node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"The consensus analyst rating for {_stock_display(stock)} is '{stock.analyst_rating_value}', which is 'Buy' or better (equivalent to a numeric score of 2.0 or lower on a 1–5 scale where 1 is Strong Buy and 5 is Strong Sell).",
        node=rating_verify_leaf,
        sources=stock.analyst_rating_urls,
        additional_instruction="Accept either explicit text ('Buy', 'Strong Buy') or a numeric consensus ≤ 2.0 as satisfying the criterion.",
    )

    # -------------------- Market_Valuation (critical) ---------------------
    mcap_node = evaluator.add_parallel(
        id=f"stock{stock_index}_Market_Valuation",
        desc="Market capitalization is at least $50 billion",
        parent=stock_node,
        critical=True,
    )
    mcap_src_present = evaluator.add_custom_node(
        result=_has_urls(stock.market_cap_urls),
        id=f"stock{stock_index}_Market_Cap_Source_Present",
        desc="URL reference showing current market capitalization is provided",
        parent=mcap_node,
        critical=True,
    )
    mcap_verify_leaf = evaluator.add_leaf(
        id=f"stock{stock_index}_Market_Cap_Reference",
        desc="URL reference showing current market capitalization",
        parent=mcap_node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"The current market capitalization of {_stock_display(stock)} is '{stock.market_cap_value}', which is at least $50 billion.",
        node=mcap_verify_leaf,
        sources=stock.market_cap_urls,
        additional_instruction="Confirm the market cap is ≥ $50B. Treat '$0.20 trillion' as $200B, etc. Allow minor real-time fluctuations.",
    )

    # -------------------- Stock_Performance (critical) --------------------
    perf_node = evaluator.add_parallel(
        id=f"stock{stock_index}_Stock_Performance",
        desc=f"Stock shows positive year-to-date price return as of {AS_OF_DATE}",
        parent=stock_node,
        critical=True,
    )
    ytd_src_present = evaluator.add_custom_node(
        result=_has_urls(stock.ytd_urls),
        id=f"stock{stock_index}_YTD_Performance_Source_Present",
        desc="URL reference showing year-to-date performance is provided",
        parent=perf_node,
        critical=True,
    )
    ytd_verify_leaf = evaluator.add_leaf(
        id=f"stock{stock_index}_YTD_Performance_Reference",
        desc="URL reference showing year-to-date performance",
        parent=perf_node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"The year-to-date price return for {_stock_display(stock)} as of {AS_OF_DATE} is '{stock.ytd_return_value}', which is positive (above 0%).",
        node=ytd_verify_leaf,
        sources=stock.ytd_urls,
        additional_instruction="Confirm that YTD return is positive (> 0%). Allow reasonable rounding.",
    )


# -----------------------------------------------------------------------------
# Main evaluation entry point
# -----------------------------------------------------------------------------
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
    Evaluate an answer for selecting two S&P 500 Dividend Aristocrat stocks meeting all criteria.
    """
    # Initialize unified evaluator
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

    # Extract structured info for up to 2 stocks
    extracted = await evaluator.extract(
        prompt=prompt_extract_two_stocks(),
        template_class=TwoStocksExtraction,
        extraction_name="two_stocks_extraction",
    )

    # Keep at most two; pad to exactly two items to build a stable tree
    stocks: List[StockItem] = list(extracted.stocks[:2])
    while len(stocks) < 2:
        stocks.append(StockItem())

    # Optional distinctness check for tickers (non-leaf helper node under Second_Stock)
    # We'll add a small, dedicated critical custom node under the Second_Stock cluster to ensure they are distinct if both tickers are provided.
    # Build the two stock branches
    await verify_one_stock(evaluator, root, stocks[0], 0)
    await verify_one_stock(evaluator, root, stocks[1], 1)

    # Distinctness check (critical) under Second_Stock
    second_stock_node = evaluator.find_node("Second_Stock") or root
    tickers_distinct = (
        _nonnull_str(stocks[0].ticker) and
        _nonnull_str(stocks[1].ticker) and
        stocks[0].ticker.strip().upper() != stocks[1].ticker.strip().upper()
    )
    evaluator.add_custom_node(
        result=tickers_distinct,
        id="stock1_2_distinct_tickers",
        desc="Second stock is distinct from the first by ticker symbol",
        parent=second_stock_node,
        critical=True,
    )

    # Record helpful reference info
    evaluator.add_custom_info(
        {
            "as_of_date": AS_OF_DATE,
            "ex_dividend_deadline": EXDIV_DEADLINE,
            "earnings_window_start": EARNINGS_WINDOW_START,
            "required_min_yield": "2.5%",
            "required_min_market_cap": "$50B",
            "analyst_rating_threshold": "Buy or better (<= 2.0 on 1–5 scale)"
        },
        info_type="reference_context",
        info_name="evaluation_context",
    )

    return evaluator.get_summary()