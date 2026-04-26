import asyncio
import logging
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.verification_tree import AggregationStrategy

# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "dividend_aristocrats_4"
TASK_DESCRIPTION = """
Identify four Dividend Aristocrat stocks from the S&P 500 index that meet all of the following criteria:

1. The stock must belong to one of these three sectors: Consumer Staples, Utilities, or Materials
2. The stock must have a current forward dividend yield of at least 2.5% (as of March 2026)
3. The stock must pay dividends quarterly (four times per year)
4. The stock must have increased its dividend for at least 40 consecutive years
5. The stock must meet all standard Dividend Aristocrat requirements (S&P 500 membership, minimum $3 billion market cap, at least $5 million average daily trading volume)

For each stock, provide:
- Company name and ticker symbol
- Sector classification
- Current forward dividend yield
- Number of consecutive years of dividend increases
- Most recent quarterly dividend amount per share
- Reference URLs from official investor relations pages or major financial data providers confirming: (a) Dividend Aristocrat status, (b) sector classification, (c) current dividend yield, (d) payment frequency, and (e) consecutive years of dividend increases
"""

ALLOWED_SECTORS = {"Consumer Staples", "Utilities", "Materials"}

# --------------------------------------------------------------------------- #
# Data models for structured extraction                                       #
# --------------------------------------------------------------------------- #
class StockExtraction(BaseModel):
    company_name: Optional[str] = None
    ticker: Optional[str] = None
    sector: Optional[str] = None

    forward_dividend_yield: Optional[str] = None  # e.g., "2.7%" or "2.7% forward"
    dividend_payment_frequency: Optional[str] = None  # e.g., "Quarterly"
    consecutive_increase_years: Optional[str] = None  # e.g., "48", "48 years"
    recent_quarterly_dividend: Optional[str] = None  # e.g., "$0.59"

    # References
    aristocrat_refs: List[str] = Field(default_factory=list)       # URLs explicitly confirming Dividend Aristocrat status
    sp500_refs: List[str] = Field(default_factory=list)            # URLs confirming S&P 500 membership
    years_refs: List[str] = Field(default_factory=list)            # URLs confirming consecutive increase years
    sector_refs: List[str] = Field(default_factory=list)           # URLs confirming sector classification
    yield_refs: List[str] = Field(default_factory=list)            # URLs confirming current forward dividend yield
    frequency_refs: List[str] = Field(default_factory=list)        # URLs confirming payment frequency (quarterly)
    market_cap_refs: List[str] = Field(default_factory=list)       # URLs confirming market capitalization >= $3B
    trading_volume_refs: List[str] = Field(default_factory=list)   # URLs confirming average daily trading value >= $5M
    id_refs: List[str] = Field(default_factory=list)               # URLs confirming company name and ticker
    dividend_amount_refs: List[str] = Field(default_factory=list)  # URLs showing most recent dividend amount per share


class StocksExtraction(BaseModel):
    stocks: List[StockExtraction] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_stocks() -> str:
    return """
    Extract up to four (4) Dividend Aristocrat stocks presented in the answer, in the same order as they appear.
    For each stock, extract the following fields exactly as stated in the answer text:

    - company_name: Full company name
    - ticker: Stock ticker symbol (e.g., "PG")
    - sector: Sector classification string
    - forward_dividend_yield: The current forward dividend yield percentage as written (e.g., "2.7%")
    - dividend_payment_frequency: The stated payment frequency (e.g., "Quarterly")
    - consecutive_increase_years: The number of consecutive years of dividend increases as written (e.g., "48", "48 years")
    - recent_quarterly_dividend: The most recent quarterly dividend amount per share as written (e.g., "$0.59")

    Also extract dedicated URL lists (only include URLs explicitly present in the answer):
    - aristocrat_refs: URLs explicitly confirming Dividend Aristocrat status
    - sp500_refs: URLs confirming S&P 500 membership
    - years_refs: URLs confirming consecutive increase years
    - sector_refs: URLs confirming sector classification
    - yield_refs: URLs confirming current forward dividend yield
    - frequency_refs: URLs confirming payment frequency (quarterly)
    - market_cap_refs: URLs confirming market capitalization (>= $3B)
    - trading_volume_refs: URLs confirming average daily trading value (>= $5M)
    - id_refs: URLs confirming company name and ticker (IR or reputable financial provider page)
    - dividend_amount_refs: URLs showing most recent dividend amount per share

    Return a JSON object:
    {
      "stocks": [
        {
          "company_name": ...,
          "ticker": ...,
          "sector": ...,
          "forward_dividend_yield": ...,
          "dividend_payment_frequency": ...,
          "consecutive_increase_years": ...,
          "recent_quarterly_dividend": ...,
          "aristocrat_refs": [ ... ],
          "sp500_refs": [ ... ],
          "years_refs": [ ... ],
          "sector_refs": [ ... ],
          "yield_refs": [ ... ],
          "frequency_refs": [ ... ],
          "market_cap_refs": [ ... ],
          "trading_volume_refs": [ ... ],
          "id_refs": [ ... ],
          "dividend_amount_refs": [ ... ]
        },
        ...
      ]
    }

    GENERAL RULES:
    - Do not invent any fields or URLs; include only what appears in the answer.
    - If a field is missing, set it to null.
    - If a URL list is missing, return an empty array for that list.
    - Normalize markdown links to raw URLs.
    - Extract at most four stocks (ignore extras beyond the first four).
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _combine_sources(*url_lists: List[str]) -> List[str]:
    """Combine multiple URL lists, deduplicate while preserving order."""
    combined: List[str] = []
    seen = set()
    for urls in url_lists:
        for u in urls or []:
            if not isinstance(u, str):
                continue
            uu = u.strip()
            if not uu:
                continue
            if uu not in seen:
                seen.add(uu)
                combined.append(uu)
    return combined


def _nz(s: Optional[str], fallback: str = "") -> str:
    return s.strip() if isinstance(s, str) else fallback


# --------------------------------------------------------------------------- #
# Verification for one stock                                                  #
# --------------------------------------------------------------------------- #
async def verify_one_stock(
    evaluator: Evaluator,
    parent_node,
    stock: StockExtraction,
    stock_index_1based: int,
) -> None:
    """Build verification subtree and run checks for a single stock."""

    name = _nz(stock.company_name, "the company")
    ticker = _nz(stock.ticker, "").upper()
    sector = _nz(stock.sector, "")
    yield_str = _nz(stock.forward_dividend_yield, "")
    freq = _nz(stock.dividend_payment_frequency, "")
    years_str = _nz(stock.consecutive_increase_years, "")
    recent_div = _nz(stock.recent_quarterly_dividend, "")

    # Parent node for this stock (non-critical to allow partial credit across different stocks)
    stock_node = evaluator.add_parallel(
        id=f"stock_{stock_index_1based}",
        desc=[
            "First qualifying Dividend Aristocrat stock",
            "Second qualifying Dividend Aristocrat stock",
            "Third qualifying Dividend Aristocrat stock",
            "Fourth qualifying Dividend Aristocrat stock",
        ][stock_index_1based - 1],
        parent=parent_node,
        critical=False,
    )

    # ---------------- Aristocrat status group (critical) ---------------- #
    arist_node = evaluator.add_parallel(
        id=f"stock_{stock_index_1based}_aristocrat_status",
        desc="Stock meets Dividend Aristocrat requirements",
        parent=stock_node,
        critical=True,
    )

    # Create leaves
    sp500_leaf = evaluator.add_leaf(
        id=f"stock_{stock_index_1based}_sp500_member",
        desc="Stock is a component of the S&P 500 index",
        parent=arist_node,
        critical=True,
    )
    min25_leaf = evaluator.add_leaf(
        id=f"stock_{stock_index_1based}_min_increase_years",
        desc="Stock has increased dividends for at least 25 consecutive years",
        parent=arist_node,
        critical=True,
    )
    mktcap_leaf = evaluator.add_leaf(
        id=f"stock_{stock_index_1based}_market_cap_requirement",
        desc="Stock has minimum market capitalization of $3 billion",
        parent=arist_node,
        critical=True,
    )
    adv_leaf = evaluator.add_leaf(
        id=f"stock_{stock_index_1based}_trading_volume",
        desc="Stock has average daily trading volume of at least $5 million",
        parent=arist_node,
        critical=True,
    )
    arist_ref_leaf = evaluator.add_leaf(
        id=f"stock_{stock_index_1based}_aristocrat_reference",
        desc="Provide URL confirming Dividend Aristocrat status",
        parent=arist_node,
        critical=True,
    )

    sp500_sources = _combine_sources(stock.sp500_refs, stock.aristocrat_refs)
    min25_sources = _combine_sources(stock.years_refs, stock.aristocrat_refs)
    mcap_sources = _combine_sources(stock.market_cap_refs, stock.aristocrat_refs, stock.id_refs)
    adv_sources = _combine_sources(stock.trading_volume_refs, stock.aristocrat_refs)
    arist_sources = _combine_sources(stock.aristocrat_refs)

    # Prepare claims
    sp500_claim = f"{name} ({ticker}) is a current constituent of the S&P 500 index."
    min25_claim = f"{name} ({ticker}) has increased its dividend for at least 25 consecutive years."
    mcap_claim = f"{name} ({ticker}) has a market capitalization of at least $3 billion (USD)."
    adv_claim = f"{name} ({ticker}) has an average daily trading value of at least $5 million (USD)."
    arist_claim = f"{name} ({ticker}) is a member of the S&P 500 Dividend Aristocrats."

    # Batch verify
    await evaluator.batch_verify(
        [
            (
                sp500_claim,
                sp500_sources if sp500_sources else None,
                sp500_leaf,
                "Judge based only on the provided URLs. If no valid URL is provided, mark this claim as not supported. Accept official S&P pages or reputable financial sources that explicitly confirm S&P 500 index membership."
            ),
            (
                min25_claim,
                min25_sources if min25_sources else None,
                min25_leaf,
                "Judge based only on the provided URLs. If no valid URL is provided, mark not supported. The page must explicitly indicate ≥25 consecutive years of dividend increases (synonyms like 'annual dividend growth streak' are acceptable)."
            ),
            (
                mcap_claim,
                mcap_sources if mcap_sources else None,
                mktcap_leaf,
                "Judge based only on the provided URLs. If no valid URL is provided, mark not supported. Use clearly labeled market capitalization; rounding within a few percent is acceptable if clearly ≥ $3B."
            ),
            (
                adv_claim,
                adv_sources if adv_sources else None,
                adv_leaf,
                "Judge based only on the provided URLs. If no valid URL is provided, mark not supported. ADTV is measured in dollar value (not just shares). You may accept S&P's official Dividend Aristocrats constituency (which implies meeting this screen) if explicitly stated."
            ),
            (
                arist_claim,
                arist_sources if arist_sources else None,
                arist_ref_leaf,
                "Judge based only on the provided URLs. If no valid URL is provided, mark not supported. The page must explicitly identify the company as an 'S&P 500 Dividend Aristocrat' (or an official list entry)."
            ),
        ]
    )

    # ---------------- Sector requirement (critical) --------------------- #
    sector_node = evaluator.add_parallel(
        id=f"stock_{stock_index_1based}_sector_requirement",
        desc="Stock belongs to Consumer Staples, Utilities, or Materials sector",
        parent=stock_node,
        critical=True,
    )

    sector_ident_leaf = evaluator.add_leaf(
        id=f"stock_{stock_index_1based}_sector_identification",
        desc="Correctly identify which of the three allowed sectors the stock belongs to",
        parent=sector_node,
        critical=True,
    )
    sector_ref_leaf = evaluator.add_leaf(
        id=f"stock_{stock_index_1based}_sector_reference",
        desc="Provide URL confirming sector classification",
        parent=sector_node,
        critical=True,
    )

    # Verify sector identification (simple logic check)
    sector_ident_claim = f"The sector value '{sector}' is one of: Consumer Staples, Utilities, or Materials."
    await evaluator.verify(
        claim=sector_ident_claim,
        node=sector_ident_leaf,
        additional_instruction="Be lenient to common synonyms: treat 'Consumer Defensive' as 'Consumer Staples'. Case-insensitive comparison."
    )

    # Verify sector classification via URLs
    sector_ref_claim = f"{name} ({ticker}) is classified in the {sector} sector."
    sector_sources = _combine_sources(stock.sector_refs)
    await evaluator.verify(
        claim=sector_ref_claim,
        node=sector_ref_leaf,
        sources=sector_sources if sector_sources else None,
        additional_instruction="Judge based only on the provided URLs. If no valid URL is provided, mark not supported. Accept classification from GICS/IR pages or major financial data providers."
    )

    # ---------------- Yield requirement (critical) ---------------------- #
    yield_node = evaluator.add_parallel(
        id=f"stock_{stock_index_1based}_yield_requirement",
        desc="Stock has forward dividend yield of at least 2.5%",
        parent=stock_node,
        critical=True,
    )

    yield_value_leaf = evaluator.add_leaf(
        id=f"stock_{stock_index_1based}_yield_value",
        desc="Provide current forward dividend yield percentage",
        parent=yield_node,
        critical=True,
    )
    yield_ref_leaf = evaluator.add_leaf(
        id=f"stock_{stock_index_1based}_yield_reference",
        desc="Provide URL confirming current dividend yield",
        parent=yield_node,
        critical=True,
    )

    yield_sources = _combine_sources(stock.yield_refs)

    # Verify threshold (>= 2.5%) as of March 2026 (or nearest available)
    yield_threshold_claim = f"The forward dividend yield for {name} ({ticker}) is at least 2.5% as of March 2026 (or the most recent data very close to that date)."
    await evaluator.verify(
        claim=yield_threshold_claim,
        node=yield_value_leaf,
        sources=yield_sources if yield_sources else None,
        additional_instruction="Judge based only on the provided URLs. If no valid URL is provided, mark not supported. Prefer 'Forward Dividend & Yield' figures; minor rounding differences are acceptable as long as it is clearly ≥ 2.5%."
    )

    # Verify the stated numeric yield value itself (best-effort exact/approx match)
    if yield_str:
        yield_ref_claim = f"The current forward dividend yield for {name} ({ticker}) is reported as approximately {yield_str}."
    else:
        yield_ref_claim = f"The current forward dividend yield for {name} ({ticker}) is clearly reported on the provided page(s)."
    await evaluator.verify(
        claim=yield_ref_claim,
        node=yield_ref_leaf,
        sources=yield_sources if yield_sources else None,
        additional_instruction="Judge based only on the provided URLs. If no valid URL is provided, mark not supported. Allow minor rounding and format variation; ensure the page shows a forward dividend yield figure matching the claim."
    )

    # ---------------- Payment frequency (critical) ---------------------- #
    freq_node = evaluator.add_parallel(
        id=f"stock_{stock_index_1based}_payment_frequency",
        desc="Stock pays dividends quarterly (four times per year)",
        parent=stock_node,
        critical=True,
    )

    freq_confirm_leaf = evaluator.add_leaf(
        id=f"stock_{stock_index_1based}_frequency_confirmation",
        desc="Confirm quarterly payment schedule",
        parent=freq_node,
        critical=True,
    )
    freq_ref_leaf = evaluator.add_leaf(
        id=f"stock_{stock_index_1based}_frequency_reference",
        desc="Provide URL confirming payment frequency",
        parent=freq_node,
        critical=True,
    )

    freq_sources = _combine_sources(stock.frequency_refs, stock.dividend_amount_refs)
    freq_claim = f"{name} ({ticker}) pays dividends quarterly (four times per year)."
    await evaluator.verify(
        claim=freq_claim,
        node=freq_confirm_leaf,
        sources=freq_sources if freq_sources else None,
        additional_instruction="Judge based only on the provided URLs. If no valid URL is provided, mark not supported. Accept wording variants like 'quarterly', 'every quarter', or 'four times per year'."
    )
    await evaluator.verify(
        claim="The provided webpage(s) explicitly indicate a quarterly dividend payment schedule for the company.",
        node=freq_ref_leaf,
        sources=freq_sources if freq_sources else None,
        additional_instruction="Judge based only on the provided URLs. If no valid URL is provided, mark not supported. Look for a frequency field or a recent history that clearly indicates quarterly cadence."
    )

    # ---------------- Extended increase requirement (critical) ---------- #
    years_node = evaluator.add_parallel(
        id=f"stock_{stock_index_1based}_extended_increase_requirement",
        desc="Stock has increased dividends for at least 40 consecutive years",
        parent=stock_node,
        critical=True,
    )

    years_count_leaf = evaluator.add_leaf(
        id=f"stock_{stock_index_1based}_years_count",
        desc="Provide exact number of consecutive years of dividend increases",
        parent=years_node,
        critical=True,
    )
    years_ref_leaf = evaluator.add_leaf(
        id=f"stock_{stock_index_1based}_years_reference",
        desc="Provide URL confirming consecutive years of increases",
        parent=years_node,
        critical=True,
    )

    years_sources = _combine_sources(stock.years_refs, stock.aristocrat_refs)
    years_threshold_claim = f"{name} ({ticker}) has increased its dividend for at least 40 consecutive years (reported as: {years_str if years_str else 'N/A'})."
    await evaluator.verify(
        claim=years_threshold_claim,
        node=years_count_leaf,
        sources=years_sources if years_sources else None,
        additional_instruction="Judge based only on the provided URLs. If no valid URL is provided, mark not supported. The page should explicitly indicate ≥40 consecutive years; if an exact number is given, that is even better."
    )

    years_exact_claim = (
        f"The exact count of consecutive years of dividend increases for {name} ({ticker}) is {years_str}."
        if years_str else
        f"The exact count of consecutive years of dividend increases for {name} ({ticker}) is clearly stated on the provided page(s)."
    )
    await evaluator.verify(
        claim=years_exact_claim,
        node=years_ref_leaf,
        sources=years_sources if years_sources else None,
        additional_instruction="Judge based only on the provided URLs. If no valid URL is provided, mark not supported. Verify that the stated number of consecutive increase years matches the page."
    )

    # ---------------- Stock identification (critical) ------------------- #
    ident_node = evaluator.add_parallel(
        id=f"stock_{stock_index_1based}_stock_identification",
        desc="Provide complete stock identification information",
        parent=stock_node,
        critical=True,
    )

    company_leaf = evaluator.add_leaf(
        id=f"stock_{stock_index_1based}_company_name",
        desc="Provide full company name",
        parent=ident_node,
        critical=True,
    )
    ticker_leaf = evaluator.add_leaf(
        id=f"stock_{stock_index_1based}_ticker_symbol",
        desc="Provide stock ticker symbol",
        parent=ident_node,
        critical=True,
    )
    recent_div_leaf = evaluator.add_leaf(
        id=f"stock_{stock_index_1based}_recent_dividend",
        desc="Provide most recent quarterly dividend amount per share",
        parent=ident_node,
        critical=True,
    )

    id_sources = _combine_sources(
        stock.id_refs,
        stock.aristocrat_refs,
        stock.sector_refs,
        stock.yield_refs,
        stock.frequency_refs,
        stock.years_refs,
        stock.market_cap_refs,
        stock.sp500_refs,
        stock.dividend_amount_refs,
    )

    # Company name
    comp_claim = f"The company's full name is '{name}'." if name else "The company's full name is clearly shown on the provided page(s)."
    await evaluator.verify(
        claim=comp_claim,
        node=company_leaf,
        sources=id_sources if id_sources else None,
        additional_instruction="Judge based only on the provided URLs. If no valid URL is provided, mark not supported. Allow minor suffix/abbreviation variants (e.g., 'Inc.' vs 'Incorporated')."
    )

    # Ticker symbol
    ticker_claim = f"The stock ticker symbol for {name} is '{ticker}'." if ticker else "The stock ticker symbol is clearly shown on the provided page(s)."
    await evaluator.verify(
        claim=ticker_claim,
        node=ticker_leaf,
        sources=id_sources if id_sources else None,
        additional_instruction="Judge based only on the provided URLs. If no valid URL is provided, mark not supported. Allow minor exchange suffix variants if applicable."
    )

    # Most recent quarterly dividend amount per share
    if recent_div:
        recent_div_claim = f"The most recent quarterly dividend amount per share for {name} ({ticker}) is {recent_div}."
    else:
        recent_div_claim = f"The most recent quarterly dividend amount per share for {name} ({ticker}) is clearly provided on the page(s)."
    await evaluator.verify(
        claim=recent_div_claim,
        node=recent_div_leaf,
        sources=_combine_sources(stock.dividend_amount_refs, stock.frequency_refs, stock.id_refs) or None,
        additional_instruction="Judge based only on the provided URLs. If no valid URL is provided, mark not supported. Accept minor rounding differences; ensure it is quarterly amount per share."
    )


# --------------------------------------------------------------------------- #
# Main evaluation function                                                    #
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
    Evaluate an answer for the Dividend Aristocrats task and return a structured result dict.
    """
    # Initialize evaluator
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,   # Parallel: each stock evaluated independently
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

    # IMPORTANT: Root must be non-critical to allow partial scoring across stocks
    # (JSON had root critical=true which conflicts with allowing partial credit)
    root.critical = False

    # Extract structured information (up to 4 stocks)
    extracted = await evaluator.extract(
        prompt=prompt_extract_stocks(),
        template_class=StocksExtraction,
        extraction_name="stocks_extraction",
    )

    # Prepare up to four stocks (pad with empty entries if fewer than four provided)
    stocks: List[StockExtraction] = list(extracted.stocks[:4])
    while len(stocks) < 4:
        stocks.append(StockExtraction())

    # Verify each stock
    for idx, stock in enumerate(stocks, start=1):
        await verify_one_stock(evaluator, root, stock, idx)

    # Add custom info
    evaluator.add_custom_info(
        {
            "extracted_stock_count": len(extracted.stocks),
            "evaluated_stock_count": 4,
            "allowed_sectors": sorted(list(ALLOWED_SECTORS)),
        },
        info_type="meta",
        info_name="evaluation_meta",
    )

    # Return summary
    return evaluator.get_summary()