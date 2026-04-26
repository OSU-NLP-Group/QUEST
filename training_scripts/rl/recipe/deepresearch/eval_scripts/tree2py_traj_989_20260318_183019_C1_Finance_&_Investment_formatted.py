import asyncio
import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "utility_dividend_stock_qtrly_1"
TASK_DESCRIPTION = """
Identify one publicly traded utility company listed on either the New York Stock Exchange (NYSE) or NASDAQ that pays quarterly dividends and has made at least one dividend payment within the last 6 months (from March 18, 2026). Provide the company's stock ticker symbol, the exchange where it is listed, and the amount of the most recent quarterly dividend payment per share in US dollars.
"""

REFERENCE_DATE = datetime(2026, 3, 18).date()
# Use ~6 months as 183 days to be robust (allow 1-day tolerance by LLM)
CUTOFF_DATE_6M = (REFERENCE_DATE - timedelta(days=183)).isoformat()


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class UtilityStockExtraction(BaseModel):
    company_name: Optional[str] = None
    ticker: Optional[str] = None
    exchange: Optional[str] = None  # e.g., NYSE, NASDAQ, NasdaqGS, NasdaqGM, New York Stock Exchange
    dividend_frequency: Optional[str] = None  # e.g., Quarterly
    most_recent_dividend_amount_usd: Optional[str] = None  # e.g., $0.44 or 0.44
    most_recent_dividend_pay_date: Optional[str] = None  # optional; e.g., 2026-03-15

    # URL sources explicitly cited in the answer
    sources_overall: List[str] = Field(default_factory=list)
    sources_company: List[str] = Field(default_factory=list)
    sources_dividend: List[str] = Field(default_factory=list)
    sources_exchange: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_utility_stock_info() -> str:
    return """
    Extract the utility stock information that the answer provides. Return exactly the fields below:

    - company_name: The full company name as written in the answer.
    - ticker: The stock ticker symbol as written (do not invent). Prefer the primary common shares ticker.
    - exchange: The exchange name as written (e.g., NYSE, NASDAQ, NasdaqGS, NasdaqGM, NasdaqCM, or New York Stock Exchange). Do not infer; use what the answer states.
    - dividend_frequency: The dividend payment frequency as explicitly stated (e.g., Quarterly, Monthly, etc.). If not stated, return null.
    - most_recent_dividend_amount_usd: The most recent quarterly dividend per share amount in USD as provided in the answer. Keep any $ sign if present. If not explicitly provided, return null.
    - most_recent_dividend_pay_date: The date of the most recent dividend payment or ex-dividend date if a specific pay date is given in the answer. If not provided, return null.

    Also extract any URLs explicitly cited in the answer. Categorize them if possible:
    - sources_overall: All URLs cited anywhere in the answer.
    - sources_company: URLs that help confirm the ticker-company mapping or sector/industry.
    - sources_dividend: URLs that show dividend history, dividend amount, ex-dividend or payment dates.
    - sources_exchange: URLs that show where the stock is listed (NYSE/NASDAQ).

    SPECIAL RULES FOR URL EXTRACTION:
    - Only include URLs that are explicitly present in the answer (plain links or markdown links).
    - Do not fabricate URLs. If none are provided for a category, return an empty list for that category.

    If a field is missing in the answer, set it to null. Do not infer missing values.
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _collect_sources(ex: UtilityStockExtraction, fields: List[str]) -> List[str]:
    urls: List[str] = []
    for f in fields:
        v = getattr(ex, f, None)
        if isinstance(v, list):
            urls.extend(v)
        elif isinstance(v, str) and v.strip():
            urls.append(v)
    # Deduplicate while preserving order
    seen = set()
    deduped = []
    for u in urls:
        if u and u not in seen:
            seen.add(u)
            deduped.append(u)
    return deduped


# --------------------------------------------------------------------------- #
# Verification subtrees                                                       #
# --------------------------------------------------------------------------- #
async def verify_company_identification(evaluator: Evaluator, parent_node, ex: UtilityStockExtraction) -> None:
    """
    Verify that the provided ticker symbol corresponds to a publicly traded utility company.
    """
    node = evaluator.add_parallel(
        id="company_identification",
        desc="Verify that the provided ticker symbol corresponds to a publicly traded utility company",
        parent=parent_node,
        critical=True
    )

    company_sources = _collect_sources(ex, ["sources_company", "sources_overall", "sources_exchange"])

    # Leaf 1: Existence check (ticker + company + at least one source)
    has_min_info = (
        (ex.ticker is not None and ex.ticker.strip() != "") and
        (ex.company_name is not None and ex.company_name.strip() != "") and
        (len(company_sources) > 0)
    )
    evaluator.add_custom_node(
        result=has_min_info,
        id="company_info_and_sources_provided",
        desc="Company name and ticker are provided, with at least one cited source URL",
        parent=node,
        critical=True
    )

    # Leaf 2: Ticker-company match (source-grounded)
    leaf_match = evaluator.add_leaf(
        id="ticker_company_match",
        desc="Ticker corresponds to the stated publicly traded company",
        parent=node,
        critical=True
    )
    match_claim = f"The stock ticker '{ex.ticker or ''}' corresponds to the publicly traded company '{ex.company_name or ''}'."
    await evaluator.verify(
        claim=match_claim,
        node=leaf_match,
        sources=company_sources,
        additional_instruction=(
            "Confirm from the provided URLs (e.g., NYSE/Nasdaq listing page, Yahoo/Google Finance, "
            "company IR) that the ticker maps to this company name. Allow minor naming variants "
            "like Inc., Corp., Co., punctuation, or capitalization differences."
        )
    )

    # Leaf 3: Utility sector/industry (source-grounded)
    leaf_util = evaluator.add_leaf(
        id="is_utility_company",
        desc="Company operates in the Utilities sector (e.g., Electric, Gas, Water, Multi-Utilities, or IP&RE)",
        parent=node,
        critical=True
    )
    util_claim = (
        f"'{ex.company_name or 'The company'}' (ticker {ex.ticker or ''}) is a utility company, "
        "i.e., classified in the Utilities sector (such as Electric Utilities, Gas Utilities, Water Utilities, "
        "Multi-Utilities, or Independent Power and Renewable Electricity Producers)."
    )
    await evaluator.verify(
        claim=util_claim,
        node=leaf_util,
        sources=company_sources,
        additional_instruction=(
            "Rely on sector/industry classifications shown on the provided pages (GICS/ICB/NAICS). "
            "Accept 'Utilities' and subindustries like Electric, Gas, Water, Multi-Utilities, or Independent Power "
            "and Renewable Electricity Producers. Do NOT accept classifications like 'Energy', 'Oil & Gas', "
            "or 'Energy Equipment & Services' as Utilities."
        )
    )


async def verify_dividend_payment_information(evaluator: Evaluator, parent_node, ex: UtilityStockExtraction) -> None:
    """
    Verify that the company pays dividends quarterly and has made at least one dividend payment within
    the last 6 months (from 2026-03-18), and that the most recent quarterly dividend amount is provided and correct.
    """
    node = evaluator.add_sequential(
        id="dividend_payment_information",
        desc="Verify dividend frequency is quarterly, recent payment within last 6 months, and most recent amount provided and correct",
        parent=parent_node,
        critical=True
    )

    dividend_sources = _collect_sources(ex, ["sources_dividend", "sources_overall"])

    # Leaf 1: Existence check (amount + frequency + at least one source)
    has_essentials = (
        (ex.most_recent_dividend_amount_usd is not None and ex.most_recent_dividend_amount_usd.strip() != "") and
        (ex.dividend_frequency is not None and ex.dividend_frequency.strip() != "") and
        (len(dividend_sources) > 0)
    )
    evaluator.add_custom_node(
        result=has_essentials,
        id="dividend_info_and_sources_provided",
        desc="Dividend frequency and most recent per-share amount are provided, with at least one cited source URL",
        parent=node,
        critical=True
    )

    # Leaf 2: Pays quarterly (source-grounded)
    leaf_quarterly = evaluator.add_leaf(
        id="pays_quarterly",
        desc="Company pays dividends quarterly (4x per year)",
        parent=node,
        critical=True
    )
    q_claim = "This company pays dividends quarterly (i.e., four times per year)."
    await evaluator.verify(
        claim=q_claim,
        node=leaf_quarterly,
        sources=dividend_sources,
        additional_instruction=(
            "Use the dividend history/summary on the provided URLs to confirm the payment frequency is quarterly. "
            "Accept synonyms like 'quarterly dividend' or a clear cadence of approximately every 3 months."
        )
    )

    # Leaf 3: Recent payment within last 6 months from 2026-03-18 (source-grounded)
    leaf_recent = evaluator.add_leaf(
        id="dividend_within_last_6_months",
        desc=f"At least one dividend payment occurred on or after {CUTOFF_DATE_6M}",
        parent=node,
        critical=True
    )
    recent_claim = f"The company made at least one cash dividend payment on or after {CUTOFF_DATE_6M}."
    await evaluator.verify(
        claim=recent_claim,
        node=leaf_recent,
        sources=dividend_sources,
        additional_instruction=(
            f"Check dividend history tables for a Pay Date or Ex-Dividend Date on/after {CUTOFF_DATE_6M}. "
            "Either a pay date or ex-dividend date within that window is acceptable evidence of a recent quarterly payment."
        )
    )

    # Leaf 4: Most recent quarterly dividend amount is correct (source-grounded)
    leaf_amount = evaluator.add_leaf(
        id="most_recent_dividend_amount_correct",
        desc="Most recent quarterly dividend per share amount matches the provided amount",
        parent=node,
        critical=True
    )
    amt = ex.most_recent_dividend_amount_usd or ""
    amount_claim = f"The most recent quarterly cash dividend per share is USD {amt}."
    await evaluator.verify(
        claim=amount_claim,
        node=leaf_amount,
        sources=dividend_sources,
        additional_instruction=(
            "From the dividend history or summary, identify the latest (most recent) quarterly cash dividend amount and "
            "check it equals the provided amount. Do not use annualized/dividend rate figures. Allow trivial formatting "
            "differences, currency symbols, or rounding to two decimals when appropriate."
        )
    )


async def verify_exchange_listing(evaluator: Evaluator, parent_node, ex: UtilityStockExtraction) -> None:
    """
    Verify that the stock is listed on either NYSE or NASDAQ.
    """
    node = evaluator.add_parallel(
        id="exchange_listing",
        desc="Verify that the stock is listed on either NYSE or NASDAQ",
        parent=parent_node,
        critical=True
    )

    exchange_sources = _collect_sources(ex, ["sources_exchange", "sources_overall", "sources_company"])

    # Leaf 1: Existence of exchange value and at least one source URL
    has_exchange_and_source = (
        (ex.exchange is not None and ex.exchange.strip() != "") and
        (len(exchange_sources) > 0)
    )
    evaluator.add_custom_node(
        result=has_exchange_and_source,
        id="exchange_and_source_provided",
        desc="Exchange name is provided with at least one cited source URL",
        parent=node,
        critical=True
    )

    # Leaf 2: Exchange is a major US exchange name (simple logical check)
    leaf_is_major = evaluator.add_leaf(
        id="exchange_is_major_name_check",
        desc="Provided exchange is one of NYSE (New York Stock Exchange) or NASDAQ",
        parent=node,
        critical=True
    )
    ex_name = ex.exchange or ""
    major_claim = f"The provided exchange name '{ex_name}' refers to either NYSE (New York Stock Exchange) or NASDAQ."
    await evaluator.verify(
        claim=major_claim,
        node=leaf_is_major,
        additional_instruction=(
            "Allow common variants like 'Nasdaq', 'NasdaqGS', 'NasdaqGM', 'NasdaqCM' for NASDAQ, and "
            "'NYSE' or 'New York Stock Exchange' for NYSE."
        )
    )

    # Leaf 3: Source-grounded listing verification
    leaf_listed = evaluator.add_leaf(
        id="exchange_listing_supported",
        desc="Ticker is listed on the stated exchange according to cited sources",
        parent=node,
        critical=True
    )
    listed_claim = f"The stock ticker '{ex.ticker or ''}' is listed on {ex.exchange or 'the stated exchange'}."
    await evaluator.verify(
        claim=listed_claim,
        node=leaf_listed,
        sources=exchange_sources,
        additional_instruction=(
            "Confirm from the provided URLs (preferably official exchange listing pages or trusted finance portals) "
            "that this ticker is listed on the stated exchange. Accept reasonable naming variants (e.g., NasdaqGS)."
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
    model: str = "o4-mini",
) -> Dict:
    """
    Evaluate an answer for the utility dividend stock task and return a structured result dictionary.
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

    # Extract structured info from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_utility_stock_info(),
        template_class=UtilityStockExtraction,
        extraction_name="utility_stock_info",
    )

    # Record evaluation window
    evaluator.add_custom_info(
        {
            "reference_date": REFERENCE_DATE.isoformat(),
            "cutoff_date_for_6_months": CUTOFF_DATE_6M,
        },
        info_type="time_window",
        info_name="dividend_time_window",
    )

    # Build and run verifications per rubric
    await verify_company_identification(evaluator, root, extracted)
    await verify_dividend_payment_information(evaluator, root, extracted)
    await verify_exchange_listing(evaluator, root, extracted)

    # Return the evaluation summary including the verification tree
    return evaluator.get_summary()