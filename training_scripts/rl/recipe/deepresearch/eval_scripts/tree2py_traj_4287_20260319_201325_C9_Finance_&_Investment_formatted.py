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
TASK_ID = "djia_aristocrats_portfolio_2026"
TASK_DESCRIPTION = (
    "As of March 2026, identify exactly 3 stocks that simultaneously meet all of the following investment criteria "
    "for a diversified income-focused portfolio:\n\n"
    "Index Requirements:\n"
    "- Current member of the S&P 500 Dividend Aristocrats index (25+ consecutive years of dividend increases)\n"
    "- One of the 30 components of the Dow Jones Industrial Average\n\n"
    "Dividend Requirements:\n"
    "- Pays dividends on a quarterly schedule (four times per year)\n"
    "- Provide the current dividend yield (Annual Dividends per Share / Current Share Price × 100)\n\n"
    "Investment Profile Requirements:\n"
    "- Large-cap market capitalization (≥ $10 billion)\n"
    "- Included in at least one of VOO, IVV, or SPY\n\n"
    "Regulatory Requirements:\n"
    "- Classified as a traditional equity security (not a cryptocurrency or digital commodity under SEC March 2026 guidance)\n"
    "- Qualifies for long-term capital gains treatment when held for more than 12 months\n\n"
    "For each stock, provide: company name, ticker, confirmations for both indices, current dividend yield, current market cap, "
    "which of VOO/IVV/SPY hold it, and URL references supporting each claim."
)


# --------------------------------------------------------------------------- #
# Data Models for Extraction                                                  #
# --------------------------------------------------------------------------- #
class StockItem(BaseModel):
    company_name: Optional[str] = None
    ticker: Optional[str] = None

    aristocrat_sources: List[str] = Field(default_factory=list)
    djia_sources: List[str] = Field(default_factory=list)

    dividend_payment_frequency: Optional[str] = None
    dividend_payment_sources: List[str] = Field(default_factory=list)

    dividend_yield: Optional[str] = None
    dividend_yield_sources: List[str] = Field(default_factory=list)

    market_cap: Optional[str] = None
    market_cap_sources: List[str] = Field(default_factory=list)

    included_etfs: List[str] = Field(default_factory=list)  # Expected values subset of ["VOO", "IVV", "SPY"]
    etf_sources: List[str] = Field(default_factory=list)

    security_type: Optional[str] = None  # e.g., "Common Stock"
    security_type_sources: List[str] = Field(default_factory=list)

    ltcg_eligible: Optional[str] = None  # e.g., "yes", "no", "unknown"
    ltcg_sources: List[str] = Field(default_factory=list)


class PortfolioExtraction(BaseModel):
    stocks: List[StockItem] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction Prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_portfolio() -> str:
    return """
    Extract all distinct U.S. stocks (company + ticker) mentioned in the answer that are proposed for the requested portfolio.
    For each stock, extract the following fields exactly as presented in the answer. Only include URLs that are explicitly present in the answer (plain URL or inside markdown links). Do not invent URLs.

    For each stock, extract:
    - company_name: Full company name as written in the answer
    - ticker: Ticker symbol as written in the answer (uppercase if readily clear)
    - aristocrat_sources: URLs cited to support that the stock is a current member of the S&P 500 Dividend Aristocrats index
    - djia_sources: URLs cited to support that the stock is a current Dow Jones Industrial Average component
    - dividend_payment_frequency: The dividend payment frequency text (e.g., "quarterly", "four times per year") if mentioned
    - dividend_payment_sources: URLs cited to support quarterly dividend payments
    - dividend_yield: The current dividend yield value as written (e.g., "3.2%", "approximately 3%"). Keep it as a string.
    - dividend_yield_sources: URLs cited that either state the current dividend yield or provide values needed to compute it (current price and annual dividend per share)
    - market_cap: The current market capitalization text (e.g., "$350B", "$12.4 billion")
    - market_cap_sources: URLs cited that show current market capitalization
    - included_etfs: A list naming any of these ETFs that the answer claims hold the stock: VOO, IVV, SPY. Use the exact tickers only ("VOO", "IVV", "SPY"). If others are mentioned, ignore them.
    - etf_sources: URLs cited that show the ETF holdings page(s) including this stock (e.g., the ETF provider's official holdings page, daily holdings CSV, or fact sheet)
    - security_type: The asset/security type text if given (e.g., "common stock", "equity")
    - security_type_sources: URLs cited that confirm the stock is a traditional corporate equity (e.g., exchange listing page, company investor relations)
    - ltcg_eligible: "yes" if the answer explicitly asserts the stock qualifies for U.S. long-term capital gains tax treatment when held >12 months; else "no" or "unknown"
    - ltcg_sources: URLs cited supporting the long-term capital gains eligibility claim (e.g., IRS guidance + stock classification evidence)

    Important:
    - Do not infer any information that isn't explicitly stated in the answer.
    - Only include valid URLs that appear in the answer.
    - Return all stocks the answer lists (not just 3); the evaluator may select the first 3 if more are provided.
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _safe_list(xs: Optional[List[str]]) -> List[str]:
    return xs if isinstance(xs, list) else []


def _name_ticker_str(stock: StockItem, index_one_based: int) -> str:
    n = stock.company_name or f"Stock #{index_one_based}"
    t = stock.ticker or "UNKNOWN"
    return f"{n} ({t})"


# --------------------------------------------------------------------------- #
# Verification for a single stock                                             #
# --------------------------------------------------------------------------- #
async def verify_single_stock(
    evaluator: Evaluator,
    parent_node,
    stock: StockItem,
    idx: int,
) -> None:
    """
    Build the verification subtree for one stock according to the rubric.
    This function adds nodes and performs verifications.
    """
    stock_num = idx + 1
    stock_node = evaluator.add_parallel(
        id=f"stock_{stock_num}",
        desc=f"Validation of the stock #{stock_num} in the portfolio",
        parent=parent_node,
        critical=False,  # The overall portfolio may allow partial credit per stock
    )

    # 1) Stock Identification
    ident_node = evaluator.add_parallel(
        id=f"stock_{stock_num}_identification",
        desc="Verify the stock is properly identified with company name and ticker symbol",
        parent=stock_node,
        critical=True,
    )
    evaluator.add_custom_node(
        result=bool(stock.company_name and stock.company_name.strip()),
        id=f"stock_{stock_num}_company_name",
        desc="Confirm the company name is provided",
        parent=ident_node,
        critical=True,
    )
    evaluator.add_custom_node(
        result=bool(stock.ticker and stock.ticker.strip()),
        id=f"stock_{stock_num}_ticker_symbol",
        desc="Confirm the ticker symbol is provided",
        parent=ident_node,
        critical=True,
    )

    # 2) Index Membership
    index_node = evaluator.add_parallel(
        id=f"stock_{stock_num}_index_membership",
        desc="Verify the stock's membership in required indices",
        parent=stock_node,
        critical=True,
    )
    # 2a) Dividend Aristocrats
    arist_node = evaluator.add_parallel(
        id=f"stock_{stock_num}_aristocrat_status",
        desc="Verify S&P 500 Dividend Aristocrats membership (25+ yrs dividend increases)",
        parent=index_node,
        critical=True,
    )
    arist_ref = evaluator.add_custom_node(
        result=len(_safe_list(stock.aristocrat_sources)) > 0,
        id=f"stock_{stock_num}_aristocrat_reference",
        desc="Provide URL reference to Dividend Aristocrats listing",
        parent=arist_node,
        critical=True,
    )
    arist_verify = evaluator.add_leaf(
        id=f"stock_{stock_num}_aristocrat_verification",
        desc="Confirm the stock appears on a Dividend Aristocrats list (as of March 2026)",
        parent=arist_node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"As of March 2026, {_name_ticker_str(stock, stock_num)} is a current member of the S&P 500 Dividend Aristocrats index.",
        node=arist_verify,
        sources=stock.aristocrat_sources,
        additional_instruction=(
            "Accept official index provider pages, S&P Global index facts, ProShares NOBL official holdings, or other "
            "credible sources explicitly listing current S&P 500 Dividend Aristocrats constituents. "
            "If the provided page lists past constituents only, consider it not supported."
        ),
    )

    # 2b) Dow Jones Industrial Average membership
    djia_node = evaluator.add_parallel(
        id=f"stock_{stock_num}_djia_status",
        desc="Verify Dow Jones Industrial Average (30 components) membership",
        parent=index_node,
        critical=True,
    )
    djia_ref = evaluator.add_custom_node(
        result=len(_safe_list(stock.djia_sources)) > 0,
        id=f"stock_{stock_num}_djia_reference",
        desc="Provide URL reference to official/current Dow Jones components listing",
        parent=djia_node,
        critical=True,
    )
    djia_verify = evaluator.add_leaf(
        id=f"stock_{stock_num}_djia_verification",
        desc="Confirm the stock is a current Dow Jones 30 component (as of March 2026)",
        parent=djia_node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"As of March 2026, {_name_ticker_str(stock, stock_num)} is one of the 30 components of the Dow Jones Industrial Average.",
        node=djia_verify,
        sources=stock.djia_sources,
        additional_instruction="Prefer official index or exchange/operator pages listing the 30 DJIA components.",
    )

    # 3) Dividend Characteristics
    div_node = evaluator.add_parallel(
        id=f"stock_{stock_num}_dividend_characteristics",
        desc="Verify the stock's dividend payment structure and metrics",
        parent=stock_node,
        critical=True,
    )

    # 3a) Payment Frequency = Quarterly
    pay_node = evaluator.add_parallel(
        id=f"stock_{stock_num}_payment_frequency",
        desc="Verify the stock pays dividends on a quarterly schedule",
        parent=div_node,
        critical=True,
    )
    pay_ref = evaluator.add_custom_node(
        result=len(_safe_list(stock.dividend_payment_sources)) > 0,
        id=f"stock_{stock_num}_payment_reference",
        desc="Provide URL reference documenting quarterly dividend payment schedule",
        parent=pay_node,
        critical=True,
    )
    pay_verify = evaluator.add_leaf(
        id=f"stock_{stock_num}_quarterly_confirmation",
        desc="Confirm the stock pays dividends four times per year",
        parent=pay_node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"{_name_ticker_str(stock, stock_num)} pays dividends on a quarterly schedule (four times per year).",
        node=pay_verify,
        sources=stock.dividend_payment_sources,
        additional_instruction="Accept official IR pages, dividend policy pages, or credible finance portals that explicitly state quarterly payments.",
    )

    # 3b) Dividend Yield (value provided; verification with evidence)
    yield_node = evaluator.add_parallel(
        id=f"stock_{stock_num}_dividend_yield",
        desc="Verify calculation and provision of the stock's dividend yield",
        parent=div_node,
        critical=True,
    )
    yield_ref = evaluator.add_custom_node(
        result=len(_safe_list(stock.dividend_yield_sources)) > 0,
        id=f"stock_{stock_num}_yield_reference",
        desc="Provide URL reference with current dividend yield or the components needed for calculation",
        parent=yield_node,
        critical=True,
    )
    yield_verify = evaluator.add_leaf(
        id=f"stock_{stock_num}_yield_calculation",
        desc="Provide the dividend yield calculated as (Annual Dividends per Share / Current Share Price) × 100",
        parent=yield_node,
        critical=True,
    )
    human_yield = (stock.dividend_yield or "").strip()
    await evaluator.verify(
        claim=f"The current dividend yield for {_name_ticker_str(stock, stock_num)} is approximately {human_yield}.",
        node=yield_verify,
        sources=stock.dividend_yield_sources,
        additional_instruction=(
            "Verify the yield value directly from the source if shown, or that the page provides current price and annual dividend per share "
            "making the stated yield plausible. Allow minor rounding differences (e.g., 3.04% ~ 3.0%)."
        ),
    )

    # 4) Investment Profile
    invest_node = evaluator.add_parallel(
        id=f"stock_{stock_num}_investment_profile",
        desc="Verify the stock's investment characteristics and availability",
        parent=stock_node,
        critical=True,
    )

    # 4a) Market Capitalization ≥ $10B (Large-cap)
    mcap_node = evaluator.add_parallel(
        id=f"stock_{stock_num}_market_cap",
        desc="Verify the stock has large-cap classification (market cap ≥ $10 billion)",
        parent=invest_node,
        critical=True,
    )
    mcap_ref = evaluator.add_custom_node(
        result=len(_safe_list(stock.market_cap_sources)) > 0,
        id=f"stock_{stock_num}_market_cap_reference",
        desc="Provide URL reference showing current market capitalization",
        parent=mcap_node,
        critical=True,
    )
    mcap_verify = evaluator.add_leaf(
        id=f"stock_{stock_num}_market_cap_verification",
        desc="Confirm the stock's market capitalization is at least $10 billion",
        parent=mcap_node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"The market capitalization of {_name_ticker_str(stock, stock_num)} is at least $10 billion (large-cap).",
        node=mcap_verify,
        sources=stock.market_cap_sources,
        additional_instruction=(
            "If the market cap is displayed in billions (e.g., $350B, $12.4B), confirm it is ≥ $10B. "
            "If shown in millions, convert mentally (e.g., $12,400M = $12.4B). Allow minor intraday fluctuations."
        ),
    )

    # 4b) ETF Inclusion (VOO, IVV, or SPY)
    etf_node = evaluator.add_parallel(
        id=f"stock_{stock_num}_etf_inclusion",
        desc="Verify the stock is included in at least one major S&P 500 ETF (VOO, IVV, or SPY)",
        parent=invest_node,
        critical=True,
    )
    etf_ref = evaluator.add_custom_node(
        result=len(_safe_list(stock.etf_sources)) > 0,
        id=f"stock_{stock_num}_etf_reference",
        desc="Provide URL reference to ETF holdings page showing the stock's inclusion",
        parent=etf_node,
        critical=True,
    )
    etf_verify = evaluator.add_leaf(
        id=f"stock_{stock_num}_etf_holdings_verification",
        desc="Confirm the stock appears in holdings of VOO, IVV, or SPY",
        parent=etf_node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"{_name_ticker_str(stock, stock_num)} is included in the holdings of at least one of these ETFs: VOO, IVV, or SPY.",
        node=etf_verify,
        sources=stock.etf_sources,
        additional_instruction=(
            "Use official provider holdings pages (Vanguard for VOO, BlackRock for IVV, State Street for SPY), their downloadable holdings files, "
            "or credible databases mirroring official holdings. Any one of VOO/IVV/SPY suffices."
        ),
    )

    # 5) Regulatory Compliance
    # The rubric marks regulatory checks, but one sub-check (LT capital gains eligibility) is non-critical.
    reg_node = evaluator.add_parallel(
        id=f"stock_{stock_num}_regulatory_compliance",
        desc="Verify the stock meets regulatory and classification requirements",
        parent=stock_node,
        critical=False,  # keep non-critical to allow the non-critical LT cap gains leaf below
    )

    # 5a) Asset Classification = Traditional Equity (critical within this subgroup)
    class_node = evaluator.add_parallel(
        id=f"stock_{stock_num}_asset_classification",
        desc="Verify the stock is classified as a traditional equity security",
        parent=reg_node,
        critical=True,
    )
    class_ref = evaluator.add_custom_node(
        result=len(_safe_list(stock.security_type_sources)) > 0,
        id=f"stock_{stock_num}_classification_reference",
        desc="Provide URL reference confirming the stock's classification as traditional equity",
        parent=class_node,
        critical=True,
    )
    class_verify = evaluator.add_leaf(
        id=f"stock_{stock_num}_security_type_verification",
        desc="Confirm the stock is a traditional corporate equity, not a crypto asset",
        parent=class_node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"{_name_ticker_str(stock, stock_num)} represents a traditional corporate equity security (e.g., common stock), not a cryptocurrency or digital commodity.",
        node=class_verify,
        sources=stock.security_type_sources,
        additional_instruction="Accept exchange listing pages (e.g., NYSE/Nasdaq), company investor relations, or similar authoritative sources indicating common stock/equity.",
    )

    # 5b) Long-term capital gains eligibility (non-critical leaf)
    ltcg_leaf = evaluator.add_leaf(
        id=f"stock_{stock_num}_tax_treatment_eligibility",
        desc="Verify the stock qualifies for long-term capital gains treatment when held more than 12 months",
        parent=reg_node,
        critical=False,  # explicitly non-critical
    )
    await evaluator.verify(
        claim=(
            "Under U.S. tax rules, equity securities (stocks) held for more than 12 months qualify for long-term capital gains tax treatment. "
            f"{_name_ticker_str(stock, stock_num)} is a stock and therefore qualifies when held > 12 months."
        ),
        node=ltcg_leaf,
        sources=stock.ltcg_sources,
        additional_instruction=(
            "Accept a combination of IRS guidance on capital gains and any source confirming the asset is a stock. "
            "If sources clearly establish both the general rule and that the asset is a stock, mark as supported."
        ),
    )


# --------------------------------------------------------------------------- #
# Main Evaluation Entry Point                                                 #
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
    Evaluate an answer for the DJIA + Dividend Aristocrats portfolio task (as of March 2026).
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

    # Extract portfolio information
    extracted_portfolio: PortfolioExtraction = await evaluator.extract(
        prompt=prompt_extract_portfolio(),
        template_class=PortfolioExtraction,
        extraction_name="portfolio_extraction",
    )

    all_stocks = extracted_portfolio.stocks or []
    total_reported = len(all_stocks)

    # Select the first 3 stocks (ignore extras; pad with placeholders if fewer)
    selected: List[StockItem] = list(all_stocks[:3])
    while len(selected) < 3:
        selected.append(StockItem())

    # Record ground-truth style meta expectations
    evaluator.add_ground_truth({
        "as_of_date": "March 2026",
        "required_indices": ["S&P 500 Dividend Aristocrats", "Dow Jones Industrial Average (DJIA)"],
        "required_etfs_any_of": ["VOO", "IVV", "SPY"],
        "market_cap_threshold": ">= $10 billion",
        "dividend_schedule": "Quarterly (four times per year)",
        "portfolio_size_requirement": "Provide 3 stocks (we evaluate the first 3 if more are given)"
    })

    # Portfolio validation node
    portfolio_node = evaluator.add_parallel(
        id="portfolio_validation",
        desc="Verify that the provided portfolio contains 3 evaluable stocks (excess items ignored)",
        parent=root,
        critical=False,  # keep non-critical to allow detailed scoring below
    )

    # Portfolio size sufficiency (critical gate at portfolio level)
    evaluator.add_custom_node(
        result=total_reported >= 3,
        id="portfolio_size_at_least_3",
        desc="At least 3 stocks are provided in the answer (excess items ignored for scoring)",
        parent=portfolio_node,
        critical=True,
    )

    evaluator.add_custom_info(
        info={
            "total_reported_stocks": total_reported,
            "evaluated_stocks": min(3, total_reported),
            "ignored_extra_stocks": max(total_reported - 3, 0),
        },
        info_type="portfolio_stats",
        info_name="portfolio_statistics"
    )

    # Build verification subtrees for each of the 3 stocks
    for i, stock in enumerate(selected):
        await verify_single_stock(evaluator, portfolio_node, stock, i)

    # Return structured summary
    return evaluator.get_summary()