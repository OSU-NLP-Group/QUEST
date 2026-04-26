import asyncio
import logging
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys

# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "dividend_aristocrats_4_sectors"
TASK_DESCRIPTION = """Identify four S&P 500 Dividend Aristocrat stocks, each from a different sector among Financials, Utilities, Consumer Staples, or Industrials. Each stock must meet all of the following criteria:

1. Must be confirmed as an S&P 500 Dividend Aristocrat with at least 25 consecutive years of dividend increases
2. Must have a current dividend yield of at least 3.0%
3. Must have significant institutional ownership from at least three major institutional investors (such as Vanguard, BlackRock, State Street, or Fidelity)
4. Must have analyst coverage with a consensus rating of Buy or Hold (not Sell)

For each stock, provide:
- Stock ticker symbol and company name
- Current sector classification
- Current dividend yield (as a percentage)
- List of at least three major institutional investors holding the stock
- Current analyst consensus rating
- A reference URL from a reputable financial source (such as Yahoo Finance, Morningstar, Bloomberg, or the company's investor relations page) that confirms the stock's Dividend Aristocrat status and key financial metrics

Ensure that all four stocks are from different sectors within the specified list."""

ALLOWED_SECTORS = ["Financials", "Utilities", "Consumer Staples", "Industrials"]
MAJOR_INSTITUTIONS = ["Vanguard", "BlackRock", "State Street", "Fidelity"]

ORDINAL = ["First", "Second", "Third", "Fourth"]


# --------------------------------------------------------------------------- #
# Data models for structured extraction                                       #
# --------------------------------------------------------------------------- #
class StockItem(BaseModel):
    ticker: Optional[str] = None
    company_name: Optional[str] = None
    sector: Optional[str] = None
    dividend_yield: Optional[str] = None
    institutional_investors: List[str] = Field(default_factory=list)
    analyst_consensus: Optional[str] = None
    reference_urls: List[str] = Field(default_factory=list)


class StocksExtraction(BaseModel):
    stocks: List[StockItem] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_stocks() -> str:
    return """
    Extract up to four stocks listed in the answer. For each stock, extract the following fields exactly as presented:

    - ticker: Stock ticker symbol (string, no extra text)
    - company_name: Company name (string)
    - sector: Sector classification (string)
    - dividend_yield: Current dividend yield as presented (string, e.g., "3.4%" or "3.40")
    - institutional_investors: A list of institutional investor names explicitly mentioned for this stock (extract exactly as written; include names like "The Vanguard Group, Inc.", "BlackRock", "State Street", "Fidelity", etc.)
    - analyst_consensus: Analyst consensus rating as presented (string; examples: "Buy", "Hold", "Overweight", "Equal Weight", "2.3/5", "3.0", etc.)
    - reference_urls: All URLs explicitly mentioned in the answer that support this stock’s information. Include all relevant URLs such as Yahoo Finance, Morningstar, Bloomberg, S&P Global, company investor relations, SEC filings, or other reputable sources. Extract actual URLs (resolve markdown links).

    Rules:
    - Do not invent data. If a field is missing, set it to null (or empty list for lists).
    - Only include at most four stocks. If the answer lists more than four, take the first four in order of appearance.
    - For URLs, include complete valid URLs; if protocol is missing, prepend http://.
    - Keep the raw strings as-is; do not normalize or interpret values (e.g., keep "%", rating words, etc.).

    Return a JSON object:
    {
      "stocks": [
        { "ticker": ..., "company_name": ..., "sector": ..., "dividend_yield": ..., "institutional_investors": [...], "analyst_consensus": ..., "reference_urls": [...] },
        ...
      ]
    }
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def first_n_or_pad(items: List[StockItem], n: int) -> List[StockItem]:
    out = items[:n]
    while len(out) < n:
        out.append(StockItem())
    return out


def list_to_english(items: List[str]) -> str:
    arr = [s for s in items if s]
    if not arr:
        return ""
    if len(arr) == 1:
        return arr[0]
    return ", ".join(arr[:-1]) + " and " + arr[-1]


# --------------------------------------------------------------------------- #
# Per-stock verification logic                                                #
# --------------------------------------------------------------------------- #
async def verify_single_stock(
    evaluator: Evaluator,
    parent_node,
    stock: StockItem,
    stock_idx: int,
    prev_sectors: List[Optional[str]],
) -> Optional[str]:
    """
    Build the verification subtree for a single stock.
    Returns the (possibly None) sector extracted for this stock for cross-item distinctness checks.
    """
    ordinal = ORDINAL[stock_idx] if stock_idx < len(ORDINAL) else f"#{stock_idx + 1}"

    stock_node = evaluator.add_parallel(
        id=f"stock_{stock_idx + 1}",
        desc=f"{ordinal} qualifying stock meeting all specified criteria",
        parent=parent_node,
        critical=False,
    )

    # Basic existence / minimum info (critical gate)
    has_min_info = bool(stock and stock.ticker and stock.company_name and stock.reference_urls)
    evaluator.add_custom_node(
        result=has_min_info,
        id=f"stock_{stock_idx + 1}_required_info",
        desc=f"{ordinal} stock has minimally required info (ticker, company name, and at least one reference URL)",
        parent=stock_node,
        critical=True
    )

    # Dividend Aristocrat Status (critical group)
    aristocrat_group = evaluator.add_parallel(
        id=f"stock_{stock_idx + 1}_dividend_aristocrat_status",
        desc="Stock must be confirmed as an S&P 500 Dividend Aristocrat with 25+ consecutive years of dividend increases",
        parent=stock_node,
        critical=True
    )

    # 1) Consecutive years >= 25
    node_years = evaluator.add_leaf(
        id=f"stock_{stock_idx + 1}_consecutive_years_verification",
        desc="Verify the stock has increased dividends for at least 25 consecutive years",
        parent=aristocrat_group,
        critical=True
    )
    claim_years = f"{stock.company_name or 'This company'} (ticker: {stock.ticker or ''}) has increased its dividend for at least 25 consecutive years."
    await evaluator.verify(
        claim=claim_years,
        node=node_years,
        sources=stock.reference_urls,
        additional_instruction="Treat 'Dividend Aristocrat' or '25+ years of consecutive dividend increases' as confirming evidence. Accept reasonable synonyms like 'increased annual dividend for 25 consecutive years'."
    )

    # 2) S&P 500 membership
    node_sp = evaluator.add_leaf(
        id=f"stock_{stock_idx + 1}_sp500_membership",
        desc="Verify the stock is currently a member of the S&P 500 index",
        parent=aristocrat_group,
        critical=True
    )
    claim_sp = f"{stock.company_name or 'This company'} (ticker: {stock.ticker or ''}) is currently a member of the S&P 500 Index."
    await evaluator.verify(
        claim=claim_sp,
        node=node_sp,
        sources=stock.reference_urls,
        additional_instruction="If the page confirms Dividend Aristocrat status (which requires S&P 500 membership), that is acceptable support for S&P 500 membership."
    )

    # Sector classification (critical)
    if stock_idx == 0:
        # Single check for allowed sector
        node_sector_allowed = evaluator.add_leaf(
            id=f"stock_{stock_idx + 1}_sector_classification_allowed",
            desc="Sector is one of Financials, Utilities, Consumer Staples, or Industrials",
            parent=stock_node,
            critical=True
        )
        claim_sector_allowed = f"{stock.company_name or 'This company'} (ticker: {stock.ticker or ''}) is classified in the '{stock.sector or ''}' sector, which is among {list_to_english(ALLOWED_SECTORS)}."
        await evaluator.verify(
            claim=claim_sector_allowed,
            node=node_sector_allowed,
            sources=stock.reference_urls,
            additional_instruction="Allow reasonable synonyms (e.g., 'Consumer Defensive' ~= 'Consumer Staples'). Verify via a reputable profile or IR page that clearly lists the sector."
        )
    else:
        # Parallel: allowed + distinct from previous
        sector_group = evaluator.add_parallel(
            id=f"stock_{stock_idx + 1}_sector_classification",
            desc="Sector must be allowed and distinct from all previously selected stocks",
            parent=stock_node,
            critical=True
        )
        # Allowed
        node_sector_allowed = evaluator.add_leaf(
            id=f"stock_{stock_idx + 1}_sector_allowed",
            desc="Sector is one of Financials, Utilities, Consumer Staples, or Industrials",
            parent=sector_group,
            critical=True
        )
        claim_sector_allowed = f"{stock.company_name or 'This company'} (ticker: {stock.ticker or ''}) is classified in the '{stock.sector or ''}' sector, which is among {list_to_english(ALLOWED_SECTORS)}."
        await evaluator.verify(
            claim=claim_sector_allowed,
            node=node_sector_allowed,
            sources=stock.reference_urls,
            additional_instruction="Allow reasonable synonyms (e.g., 'Consumer Defensive' ~= 'Consumer Staples'). Verify via a reputable profile or IR page that clearly lists the sector."
        )
        # Distinctness from previous sectors (simple verify)
        prev_clean = [s for s in prev_sectors if s]
        node_sector_unique = evaluator.add_leaf(
            id=f"stock_{stock_idx + 1}_sector_distinct",
            desc="Sector is different from previously selected stocks' sectors",
            parent=sector_group,
            critical=True
        )
        claim_sector_unique = f"The sector '{stock.sector or ''}' for {stock.ticker or 'this stock'} is different from the previously selected sector(s): {list_to_english(prev_clean) if prev_clean else '(none)'}."
        await evaluator.verify(
            claim=claim_sector_unique,
            node=node_sector_unique,
            additional_instruction="Treat commonly-accepted synonyms as the same sector (e.g., 'Consumer Defensive' equals 'Consumer Staples'). Consider letter casing and trivial formatting differences as the same."
        )

    # Dividend yield >= 3.0% (critical)
    node_yield = evaluator.add_leaf(
        id=f"stock_{stock_idx + 1}_dividend_yield",
        desc="Stock must have a current dividend yield of at least 3.0%",
        parent=stock_node,
        critical=True
    )
    claim_yield = f"The current dividend yield for {stock.company_name or 'this company'} (ticker: {stock.ticker or ''}) is at least 3.0%."
    await evaluator.verify(
        claim=claim_yield,
        node=node_yield,
        sources=stock.reference_urls,
        additional_instruction="Use the most recent yield shown on the page. Allow small timing/rounding fluctuations (e.g., 2.95–2.99% may reasonably be considered 3.0%)."
    )

    # Institutional ownership by at least 3 major institutions (critical)
    node_inst = evaluator.add_leaf(
        id=f"stock_{stock_idx + 1}_institutional_ownership",
        desc="At least three of Vanguard, BlackRock, State Street, Fidelity hold the stock",
        parent=stock_node,
        critical=True
    )
    claim_inst = f"At least three of the following are institutional investors in {stock.company_name or 'this company'} (ticker: {stock.ticker or ''}): {', '.join(MAJOR_INSTITUTIONS)}."
    await evaluator.verify(
        claim=claim_inst,
        node=node_inst,
        sources=stock.reference_urls,
        additional_instruction="Allow common variants such as 'The Vanguard Group, Inc.', 'BlackRock Fund Advisors', 'State Street Global Advisors', 'Fidelity Management & Research'. Verify on holders/ownership pages."
    )

    # Analyst consensus Buy or Hold (critical)
    node_rating = evaluator.add_leaf(
        id=f"stock_{stock_idx + 1}_analyst_consensus",
        desc="Consensus analyst rating is Buy or Hold (not Sell)",
        parent=stock_node,
        critical=True
    )
    claim_rating = f"The consensus analyst rating for {stock.company_name or 'this company'} (ticker: {stock.ticker or ''}) is Buy or Hold (i.e., not Sell)."
    await evaluator.verify(
        claim=claim_rating,
        node=node_rating,
        sources=stock.reference_urls,
        additional_instruction=(
            "Accept synonymous language or scales mapping to Buy/Hold (e.g., 'Strong Buy', 'Outperform', 'Overweight' ≈ Buy; "
            "'Equal Weight', 'Market Perform', 'Neutral' ≈ Hold). For numeric 1–5 scales, treat 1–2 as Buy, 3 as Hold, 4–5 as Sell."
        )
    )

    # Reference URL validity and coverage (critical)
    node_ref = evaluator.add_leaf(
        id=f"stock_{stock_idx + 1}_reference_url",
        desc="At least one provided URL is reputable and confirms Dividend Aristocrat status and key metrics",
        parent=stock_node,
        critical=True
    )
    claim_ref = (
        f"At least one of the provided URLs is from a reputable financial source (e.g., Yahoo Finance, Morningstar, Bloomberg, S&P Global, or the company's investor relations) "
        f"and explicitly confirms that {stock.company_name or 'this company'} (ticker: {stock.ticker or ''}) is a Dividend Aristocrat "
        f"(or has 25+ consecutive years of dividend increases) and also provides current dividend yield and/or analyst rating information."
    )
    await evaluator.verify(
        claim=claim_ref,
        node=node_ref,
        sources=stock.reference_urls,
        additional_instruction="Reputable sources include finance.yahoo.com, morningstar.com, bloomberg.com, spglobal.com, and official company investor relations pages."
    )

    # Return sector for distinctness checks
    return stock.sector


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
    Evaluate an answer for the Dividend Aristocrats multi-sector task.
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

    # Extraction
    extracted = await evaluator.extract(
        prompt=prompt_extract_stocks(),
        template_class=StocksExtraction,
        extraction_name="stocks_extraction",
    )

    # Keep first 4 items, pad if fewer
    stocks = first_n_or_pad(extracted.stocks, 4)

    # Optional: record allowed sectors & major institutions for transparency
    evaluator.add_custom_info(
        info={"allowed_sectors": ALLOWED_SECTORS, "major_institutions": MAJOR_INSTITUTIONS},
        info_type="config",
        info_name="evaluation_constraints"
    )

    # Build per-stock verification subtrees (in parallel aggregation under root)
    prev_sectors: List[Optional[str]] = []
    for i in range(4):
        sec = await verify_single_stock(
            evaluator=evaluator,
            parent_node=root,
            stock=stocks[i],
            stock_idx=i,
            prev_sectors=prev_sectors,
        )
        prev_sectors.append(sec)

    # Return evaluation summary
    return evaluator.get_summary()