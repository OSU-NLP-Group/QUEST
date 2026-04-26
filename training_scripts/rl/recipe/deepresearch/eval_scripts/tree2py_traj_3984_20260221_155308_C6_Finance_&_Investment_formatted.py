import asyncio
import logging
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


TASK_ID = "dividend_util_bank_yield_2026"
TASK_DESCRIPTION = (
    "An income-focused investor is researching dividend-paying companies in the utilities and financial sectors for a diversified portfolio. "
    "The investor has the following specific requirements:\n\n"
    "Part 1 - Utility Sector Analysis:\n"
    "Identify ONE natural gas utility company that:\n"
    "- Is headquartered in Texas\n"
    "- Currently qualifies as an S&P 500 Dividend Aristocrat (25+ consecutive years of dividend increases)\n"
    "- Pays quarterly dividends\n\n"
    "Provide: name & ticker; the exact number of consecutive years (as of 2026), the current annual dividend per share (most recently declared rate for 2026), and the most recent quarterly dividend amount per share.\n\n"
    "Part 2 - Regional Bank Analysis:\n"
    "Identify ONE regional bank headquartered in Ohio, Indiana, or Illinois; trading on NYSE or NASDAQ; pays quarterly dividends; market cap between $5B and $15B.\n"
    "Provide: name & ticker; HQ state; current quarterly dividend per share; current market cap (billions, rounded to one decimal place).\n\n"
    "Part 3 - Comparative Analysis:\n"
    "Compute and provide dividend yields for both companies and identify which is higher. All financial data should reflect most current info as of Feb 2026. Provide reference URLs for all key financial metrics."
)


# ----------------------------- Data Models ----------------------------- #
class UtilityInfo(BaseModel):
    name: Optional[str] = None
    ticker: Optional[str] = None
    hq_state: Optional[str] = None
    classification: Optional[str] = None  # e.g., "natural gas utility"
    consecutive_years: Optional[str] = None  # as-of 2026
    annual_dividend_per_share_2026: Optional[str] = None
    most_recent_quarterly_dividend_per_share: Optional[str] = None
    pays_quarterly: Optional[str] = None  # e.g., "yes"/"no"/"quarterly"

    profile_urls: List[str] = Field(default_factory=list)  # general company profile / IR pages for HQ & classification
    aristocrat_or_streak_urls: List[str] = Field(default_factory=list)
    annual_dividend_urls: List[str] = Field(default_factory=list)
    quarterly_dividend_urls: List[str] = Field(default_factory=list)


class BankInfo(BaseModel):
    name: Optional[str] = None
    ticker: Optional[str] = None
    hq_state: Optional[str] = None
    exchange: Optional[str] = None  # NYSE or NASDAQ
    current_quarterly_dividend_per_share: Optional[str] = None
    market_cap_billions_rounded_1_decimal: Optional[str] = None
    pays_quarterly: Optional[str] = None  # e.g., "yes"/"no"/"quarterly"

    profile_urls: List[str] = Field(default_factory=list)  # for HQ state, exchange
    quarterly_dividend_urls: List[str] = Field(default_factory=list)
    market_cap_urls: List[str] = Field(default_factory=list)


class YieldInfo(BaseModel):
    utility_yield_percent: Optional[str] = None  # e.g., "2.7%"
    utility_stock_price: Optional[str] = None
    price_utility_urls: List[str] = Field(default_factory=list)

    bank_yield_percent: Optional[str] = None
    bank_stock_price: Optional[str] = None
    price_bank_urls: List[str] = Field(default_factory=list)

    higher_yield_company: Optional[str] = None  # name or ticker
    recency_note: Optional[str] = None


class FullExtraction(BaseModel):
    utility: Optional[UtilityInfo] = None
    bank: Optional[BankInfo] = None
    yields: Optional[YieldInfo] = None


# -------------------------- Extraction Prompt -------------------------- #
def prompt_extract_all() -> str:
    return """
    Extract structured information from the answer for three parts: a utility company, a regional bank, and dividend yield comparison. Return a JSON object with sections: utility, bank, yields. Extract only what is explicitly present in the answer. Include all cited URLs.

    utility:
      - name: Company name
      - ticker: Stock ticker symbol
      - hq_state: The headquarters state
      - classification: A short phrase that indicates the company is a natural gas utility (e.g., "natural gas utility", "gas distribution utility")
      - consecutive_years: Exact number of consecutive years of dividend increases, as of 2026
      - annual_dividend_per_share_2026: Current annual dividend per share (most recently declared rate for 2026)
      - most_recent_quarterly_dividend_per_share: Most recent quarterly dividend amount per share
      - pays_quarterly: State if the company pays quarterly dividends ("yes"/"no"/"quarterly")
      - profile_urls: URLs in the answer that support general company info (HQ state, business type). Include any official IR pages, company profile pages, or reputable sources
      - aristocrat_or_streak_urls: URLs that support Dividend Aristocrat status and/or the dividend-increase streak figure
      - annual_dividend_urls: URLs that support the annual dividend rate used
      - quarterly_dividend_urls: URLs that support the most recent quarterly dividend amount
    bank:
      - name: Bank company name
      - ticker: Stock ticker symbol
      - hq_state: Headquarters state
      - exchange: Trading exchange (NYSE or NASDAQ)
      - current_quarterly_dividend_per_share: Current quarterly dividend amount per share (most recently declared)
      - market_cap_billions_rounded_1_decimal: Current market capitalization in billions, rounded to one decimal place (as presented in the answer)
      - pays_quarterly: State if the bank pays quarterly dividends ("yes"/"no"/"quarterly")
      - profile_urls: URLs that support HQ state and exchange information (official IR, exchange listing page, or reputable sources)
      - quarterly_dividend_urls: URLs that support the most recent quarterly dividend amount
      - market_cap_urls: URLs that support the market capitalization value used
    yields:
      - utility_yield_percent: Utility dividend yield reported (as a percentage string, e.g., "2.7%")
      - utility_stock_price: The utility's current stock price used for the yield calculation
      - price_utility_urls: URLs that support the current stock price used
      - bank_yield_percent: Bank dividend yield reported (as a percentage string)
      - bank_stock_price: The bank's current stock price used for the yield calculation
      - price_bank_urls: URLs that support the current stock price used
      - higher_yield_company: Which company has the higher dividend yield (state the company name or ticker from the answer)
      - recency_note: Any explicit statement about data recency as of February 2026 (optional)

    URL extraction rules:
    - Extract only full, valid URLs explicitly present in the answer (including markdown links).
    - If no URL is present for a field, return an empty list.
    - Do not invent URLs.

    For any missing field, return null (or [] for lists).
    """


# ----------------------------- Helpers ----------------------------- #
def _merge_sources(*lists: Optional[List[str]]) -> List[str]:
    seen = set()
    merged: List[str] = []
    for lst in lists:
        if not lst:
            continue
        for url in lst:
            if url and url not in seen:
                seen.add(url)
                merged.append(url)
    return merged


def _non_empty(s: Optional[str]) -> bool:
    return bool(s and str(s).strip())


def _safe(s: Optional[str]) -> str:
    return s or ""


# ------------------------- Verification Logic ------------------------- #
async def verify_part1_utility(evaluator: Evaluator, parent_node, util: Optional[UtilityInfo]) -> None:
    part1 = evaluator.add_parallel(
        id="part1_utility_sector_analysis",
        desc="Identify 1 qualifying Texas natural gas utility Dividend Aristocrat that pays quarterly dividends, and provide metrics with sources.",
        parent=parent_node,
        critical=True,
    )

    # Identity
    identity = evaluator.add_parallel(
        id="utility_identity",
        desc="Provide the utility company name and ticker symbol.",
        parent=part1,
        critical=True,
    )
    evaluator.add_custom_node(
        result=_non_empty(util.name) if util else False,
        id="utility_company_name",
        desc="Utility company name is provided.",
        parent=identity,
        critical=True,
    )
    evaluator.add_custom_node(
        result=_non_empty(util.ticker) if util else False,
        id="utility_ticker_symbol",
        desc="Utility stock ticker symbol is provided.",
        parent=identity,
        critical=True,
    )

    # Qualifications
    quals = evaluator.add_parallel(
        id="utility_qualifications",
        desc="Utility meets all stated qualification criteria from Part 1.",
        parent=part1,
        critical=True,
    )

    # Natural gas utility
    ng_leaf = evaluator.add_leaf(
        id="utility_is_natural_gas_utility",
        desc="Company is a natural gas utility company.",
        parent=quals,
        critical=True,
    )
    ng_sources = _merge_sources(util.profile_urls if util else None, util.aristocrat_or_streak_urls if util else None)
    await evaluator.verify(
        claim=f"The company {_safe(util.name)} is a natural gas utility company.",
        node=ng_leaf,
        sources=ng_sources,
        additional_instruction="Verify on the provided company profile or reputable sources that the primary business is natural gas utility/distribution.",
    )

    # HQ in Texas
    hq_leaf = evaluator.add_leaf(
        id="utility_headquartered_in_texas",
        desc="Company is headquartered in Texas.",
        parent=quals,
        critical=True,
    )
    hq_sources = _merge_sources(util.profile_urls if util else None)
    await evaluator.verify(
        claim=f"The company {_safe(util.name)} is headquartered in Texas (reported state: {_safe(util.hq_state)}).",
        node=hq_leaf,
        sources=hq_sources,
        additional_instruction="Confirm the HQ state is Texas on official IR or reputable profiles. Minor formatting variations are acceptable.",
    )

    # Dividend Aristocrat (25+ years)
    arist_leaf = evaluator.add_leaf(
        id="utility_is_dividend_aristocrat_25_plus_years",
        desc="Company currently qualifies as an S&P 500 Dividend Aristocrat (25+ consecutive years of dividend increases).",
        parent=quals,
        critical=True,
    )
    await evaluator.verify(
        claim=f"The company {_safe(util.name)} is an S&P 500 Dividend Aristocrat with at least 25 consecutive years of dividend increases as of 2026.",
        node=arist_leaf,
        sources=(util.aristocrat_or_streak_urls if util else []),
        additional_instruction="Use the aristocrats list or authoritative sources to confirm 25+ years and current qualification.",
    )

    # Pays quarterly dividends
    freq_leaf = evaluator.add_leaf(
        id="utility_pays_quarterly_dividends",
        desc="Company pays quarterly dividends.",
        parent=quals,
        critical=True,
    )
    freq_sources = _merge_sources(util.quarterly_dividend_urls if util else None, util.annual_dividend_urls if util else None, util.profile_urls if util else None)
    await evaluator.verify(
        claim=f"The company {_safe(util.name)} pays dividends on a quarterly schedule.",
        node=freq_leaf,
        sources=freq_sources,
        additional_instruction="Confirm payout frequency as quarterly via dividend history/IR pages.",
    )

    # Required metrics
    metrics = evaluator.add_parallel(
        id="utility_required_metrics",
        desc="Provide all required utility metrics requested in Part 1.",
        parent=part1,
        critical=True,
    )

    # Consecutive years exact number
    years_leaf = evaluator.add_leaf(
        id="utility_consecutive_years_exact_number",
        desc="Exact number of consecutive years of dividend increases (as of 2026) is provided.",
        parent=metrics,
        critical=True,
    )
    await evaluator.verify(
        claim=f"As of 2026, {_safe(util.name)} has {_safe(util.consecutive_years)} consecutive years of dividend increases.",
        node=years_leaf,
        sources=(util.aristocrat_or_streak_urls if util else []),
        additional_instruction="Verify the exact streak count; minor rounding or off-by-one due to recent increases should be treated carefully.",
    )

    # Current annual dividend per share (2026)
    annual_div_leaf = evaluator.add_leaf(
        id="utility_current_annual_dividend_per_share_2026",
        desc="Current annual dividend per share (most recently declared rate for 2026) is provided.",
        parent=metrics,
        critical=True,
    )
    await evaluator.verify(
        claim=f"The current annual dividend per share for {_safe(util.name)} (2026 most recently declared rate) is {_safe(util.annual_dividend_per_share_2026)}.",
        node=annual_div_leaf,
        sources=(util.annual_dividend_urls if util else []),
        additional_instruction="Check IR/dividend pages for the annual rate used for 2026. Accept small rounding.",
    )

    # Most recent quarterly dividend per share
    qtr_div_leaf = evaluator.add_leaf(
        id="utility_most_recent_quarterly_dividend_per_share",
        desc="Most recent quarterly dividend amount per share is provided.",
        parent=metrics,
        critical=True,
    )
    await evaluator.verify(
        claim=f"The most recent quarterly dividend amount per share for {_safe(util.name)} is {_safe(util.most_recent_quarterly_dividend_per_share)}.",
        node=qtr_div_leaf,
        sources=(util.quarterly_dividend_urls if util else []),
        additional_instruction="Confirm the most recent quarterly amount on dividend history/IR pages.",
    )

    # Sources existence checks
    srcs = evaluator.add_parallel(
        id="utility_sources",
        desc="Provide reference URLs supporting the key utility metrics/status used.",
        parent=part1,
        critical=True,
    )
    evaluator.add_custom_node(
        result=bool(util and util.aristocrat_or_streak_urls and len(util.aristocrat_or_streak_urls) > 0),
        id="source_utility_aristocrat_or_streak",
        desc="A reference URL is provided supporting the Dividend Aristocrat qualification and/or dividend-increase streak figure.",
        parent=srcs,
        critical=True,
    )
    evaluator.add_custom_node(
        result=bool(util and util.annual_dividend_urls and len(util.annual_dividend_urls) > 0),
        id="source_utility_annual_dividend",
        desc="A reference URL is provided supporting the annual dividend per share rate used (most recently declared for 2026).",
        parent=srcs,
        critical=True,
    )
    evaluator.add_custom_node(
        result=bool(util and util.quarterly_dividend_urls and len(util.quarterly_dividend_urls) > 0),
        id="source_utility_quarterly_dividend",
        desc="A reference URL is provided supporting the most recent quarterly dividend amount.",
        parent=srcs,
        critical=True,
    )


async def verify_part2_bank(evaluator: Evaluator, parent_node, bank: Optional[BankInfo]) -> None:
    part2 = evaluator.add_parallel(
        id="part2_regional_bank_analysis",
        desc="Identify 1 qualifying regional bank (HQ in OH/IN/IL; NYSE/NASDAQ; quarterly dividends; $5B–$15B market cap) and provide metrics with sources.",
        parent=parent_node,
        critical=True,
    )

    # Identity
    identity = evaluator.add_parallel(
        id="bank_identity",
        desc="Provide the bank company name and ticker symbol.",
        parent=part2,
        critical=True,
    )
    evaluator.add_custom_node(
        result=_non_empty(bank.name) if bank else False,
        id="bank_company_name",
        desc="Bank company name is provided.",
        parent=identity,
        critical=True,
    )
    evaluator.add_custom_node(
        result=_non_empty(bank.ticker) if bank else False,
        id="bank_ticker_symbol",
        desc="Bank stock ticker symbol is provided.",
        parent=identity,
        critical=True,
    )

    # HQ State report + allowed
    hq_group = evaluator.add_parallel(
        id="bank_headquarters_state",
        desc="Report the bank headquarters state and ensure it is one of the allowed states.",
        parent=part2,
        critical=True,
    )
    evaluator.add_custom_node(
        result=_non_empty(bank.hq_state) if bank else False,
        id="bank_hq_state_reported",
        desc="The specific state where the bank is headquartered is stated.",
        parent=hq_group,
        critical=True,
    )
    allowed_leaf = evaluator.add_leaf(
        id="bank_hq_state_allowed",
        desc="The headquarters state is Ohio, Indiana, or Illinois.",
        parent=hq_group,
        critical=True,
    )
    await evaluator.verify(
        claim=f"The reported headquarters state '{_safe(bank.hq_state)}' is one of Ohio, Indiana, or Illinois.",
        node=allowed_leaf,
        additional_instruction="Pure logic check: Pass if the reported state exactly matches Ohio, Indiana, or Illinois (case-insensitive).",
    )

    # Qualifications
    quals = evaluator.add_parallel(
        id="bank_qualifications",
        desc="Bank meets the remaining stated qualification criteria from Part 2.",
        parent=part2,
        critical=True,
    )

    # Exchange
    exch_leaf = evaluator.add_leaf(
        id="bank_trades_on_nyse_or_nasdaq",
        desc="Bank trades on NYSE or NASDAQ.",
        parent=quals,
        critical=True,
    )
    await evaluator.verify(
        claim=f"The bank {_safe(bank.name)} trades on {_safe(bank.exchange)}, which is a major U.S. stock exchange (NYSE or NASDAQ).",
        node=exch_leaf,
        sources=(bank.profile_urls if bank else []),
        additional_instruction="Confirm exchange listing (NYSE or NASDAQ) via company profile, exchange listing page, or reputable sources.",
    )

    # Pays quarterly dividends
    pays_leaf = evaluator.add_leaf(
        id="bank_pays_quarterly_dividends",
        desc="Bank currently pays quarterly dividends.",
        parent=quals,
        critical=True,
    )
    await evaluator.verify(
        claim=f"The bank {_safe(bank.name)} pays dividends on a quarterly schedule.",
        node=pays_leaf,
        sources=(bank.quarterly_dividend_urls if bank else []),
        additional_instruction="Confirm the payout frequency as quarterly via dividend history/IR pages.",
    )

    # Market cap in range
    mc_range_leaf = evaluator.add_leaf(
        id="bank_market_cap_in_range_5_to_15",
        desc="Bank market capitalization is between $5B and $15B (as of Feb 2026).",
        parent=quals,
        critical=True,
    )
    await evaluator.verify(
        claim=f"As of February 2026, {_safe(bank.name)} has a market capitalization between $5B and $15B (reported: {_safe(bank.market_cap_billions_rounded_1_decimal)} billion).",
        node=mc_range_leaf,
        sources=(bank.market_cap_urls if bank else []),
        additional_instruction="Check market cap from reputable financial sources. Allow rounding and small daily fluctuations around threshold; pass if clearly within 5–15B.",
    )

    # Required metrics
    metrics = evaluator.add_parallel(
        id="bank_required_metrics",
        desc="Provide all required bank metrics requested in Part 2.",
        parent=part2,
        critical=True,
    )

    # Quarterly dividend amount
    bank_q_leaf = evaluator.add_leaf(
        id="bank_current_quarterly_dividend_per_share",
        desc="Current quarterly dividend amount per share (most recently declared) is provided.",
        parent=metrics,
        critical=True,
    )
    await evaluator.verify(
        claim=f"The current quarterly dividend amount per share for {_safe(bank.name)} is {_safe(bank.current_quarterly_dividend_per_share)}.",
        node=bank_q_leaf,
        sources=(bank.quarterly_dividend_urls if bank else []),
        additional_instruction="Confirm the most recently declared quarterly dividend amount.",
    )

    # Market cap (billions, one decimal)
    bank_mc_leaf = evaluator.add_leaf(
        id="bank_market_cap_billions_rounded_1_decimal",
        desc="Current market capitalization is provided in billions, rounded to one decimal place.",
        parent=metrics,
        critical=True,
    )
    await evaluator.verify(
        claim=f"The current market capitalization for {_safe(bank.name)} is {_safe(bank.market_cap_billions_rounded_1_decimal)} billion (rounded to one decimal place).",
        node=bank_mc_leaf,
        sources=(bank.market_cap_urls if bank else []),
        additional_instruction="Verify the market cap value and accept rounding to one decimal place.",
    )

    # Sources existence
    srcs = evaluator.add_parallel(
        id="bank_sources",
        desc="Provide reference URLs supporting the key bank metrics used.",
        parent=part2,
        critical=True,
    )
    evaluator.add_custom_node(
        result=bool(bank and bank.quarterly_dividend_urls and len(bank.quarterly_dividend_urls) > 0),
        id="source_bank_quarterly_dividend",
        desc="A reference URL is provided supporting the most recently declared quarterly dividend amount.",
        parent=srcs,
        critical=True,
    )
    evaluator.add_custom_node(
        result=bool(bank and bank.market_cap_urls and len(bank.market_cap_urls) > 0),
        id="source_bank_market_cap",
        desc="A reference URL is provided supporting the market capitalization value used.",
        parent=srcs,
        critical=True,
    )


async def verify_part3_yields(evaluator: Evaluator, parent_node, util: Optional[UtilityInfo], bank: Optional[BankInfo], yi: Optional[YieldInfo]) -> None:
    part3 = evaluator.add_parallel(
        id="part3_comparative_dividend_yield_analysis",
        desc="Compute and report dividend yields for both selected companies (as of Feb 2026) and identify which is higher, with reference URLs for key inputs.",
        parent=parent_node,
        critical=True,
    )

    # Utility yield group
    u_group = evaluator.add_parallel(
        id="utility_dividend_yield",
        desc="Provide the utility dividend yield calculation and result.",
        parent=part3,
        critical=True,
    )

    u_yield_leaf = evaluator.add_leaf(
        id="utility_yield_percentage_reported",
        desc="Utility dividend yield is reported as a percentage.",
        parent=u_group,
        critical=True,
    )
    u_yield_sources = _merge_sources(util.annual_dividend_urls if util else None, yi.price_utility_urls if yi else None)
    await evaluator.verify(
        claim=f"The dividend yield for {_safe(util.name)} is reported as {_safe(yi.utility_yield_percent)}.",
        node=u_yield_leaf,
        sources=u_yield_sources,
        additional_instruction=(
            "Verify consistency using Yield = Annual Dividend per Share / Current Stock Price. "
            "Use the provided annual dividend (2026 rate) and the provided current price. Accept minor rounding."
        ),
    )

    u_inputs_leaf = evaluator.add_custom_node(
        result=bool(util and _non_empty(util.annual_dividend_per_share_2026) and yi and _non_empty(yi.utility_stock_price)),
        id="utility_yield_inputs_provided",
        desc="Inputs used for the utility yield calculation are provided (annual dividend per share and current stock price).",
        parent=u_group,
        critical=True,
    )

    u_input_src_leaf = evaluator.add_custom_node(
        result=bool(util and util.annual_dividend_urls and len(util.annual_dividend_urls) > 0 and yi and yi.price_utility_urls and len(yi.price_utility_urls) > 0),
        id="utility_yield_sources_for_inputs",
        desc="Reference URL(s) are provided for the utility annual dividend per share and the utility current stock price used.",
        parent=u_group,
        critical=True,
    )

    # Bank yield group
    b_group = evaluator.add_parallel(
        id="bank_dividend_yield",
        desc="Provide the bank dividend yield calculation and result.",
        parent=part3,
        critical=True,
    )

    b_yield_leaf = evaluator.add_leaf(
        id="bank_yield_percentage_reported",
        desc="Bank dividend yield is reported as a percentage.",
        parent=b_group,
        critical=True,
    )
    b_yield_sources = _merge_sources(bank.quarterly_dividend_urls if bank else None, yi.price_bank_urls if yi else None)
    await evaluator.verify(
        claim=f"The dividend yield for {_safe(bank.name)} is reported as {_safe(yi.bank_yield_percent)}.",
        node=b_yield_leaf,
        sources=b_yield_sources,
        additional_instruction=(
            "Verify consistency using Yield = Annual Dividend per Share / Current Stock Price. "
            "If only quarterly dividend is provided, annualize by multiplying by 4. Accept minor rounding."
        ),
    )

    b_inputs_leaf = evaluator.add_custom_node(
        result=bool(bank and _non_empty(bank.current_quarterly_dividend_per_share) and yi and _non_empty(yi.bank_stock_price)),
        id="bank_yield_inputs_provided",
        desc="Inputs used for the bank yield calculation are provided (annual dividend per share and current stock price).",
        parent=b_group,
        critical=True,
    )

    b_input_src_leaf = evaluator.add_custom_node(
        result=bool(bank and bank.quarterly_dividend_urls and len(bank.quarterly_dividend_urls) > 0 and yi and yi.price_bank_urls and len(yi.price_bank_urls) > 0),
        id="bank_yield_sources_for_inputs",
        desc="Reference URL(s) are provided for the bank annual dividend per share (or derivation) and the bank current stock price used.",
        parent=b_group,
        critical=True,
    )

    # Higher yield identified
    higher_leaf = evaluator.add_leaf(
        id="higher_yield_identified",
        desc="Correctly identify which of the two companies has the higher dividend yield based on the reported yields.",
        parent=part3,
        critical=True,
    )
    await evaluator.verify(
        claim=(
            f"Between utility {_safe(util.name or util.ticker)} with yield {_safe(yi.utility_yield_percent)} "
            f"and bank {_safe(bank.name or bank.ticker)} with yield {_safe(yi.bank_yield_percent)}, "
            f"the higher yield is {_safe(yi.higher_yield_company)}."
        ),
        node=higher_leaf,
        additional_instruction="Pure logic check using the reported yield percentages; pass if the stated higher-yield company matches the comparison.",
    )

    # Recency check as of Feb 2026
    recency_leaf = evaluator.add_leaf(
        id="recency_as_of_feb_2026",
        desc="Dividend yield inputs/results are consistent with being current as of February 2026 (per the question’s recency requirement).",
        parent=part3,
        critical=True,
    )
    recency_sources = _merge_sources(
        util.annual_dividend_urls if util else None,
        yi.price_utility_urls if yi else None,
        bank.quarterly_dividend_urls if bank else None,
        yi.price_bank_urls if yi else None,
    )
    await evaluator.verify(
        claim=(
            f"The dividend inputs (annual/quarterly dividends and prices) used are current as of around February 2026. "
            f"Note: {_safe(yi.recency_note)}"
        ),
        node=recency_leaf,
        sources=recency_sources,
        additional_instruction="Use dates/timestamps or context on the pages to judge recency (2026 timeframe). Accept 'most recent' if context aligns with early 2026.",
    )


# ------------------------- Main Evaluation ------------------------- #
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

    extracted = await evaluator.extract(
        prompt=prompt_extract_all(),
        template_class=FullExtraction,
        extraction_name="structured_extraction",
    )

    # Build verification tree for all three parts
    await verify_part1_utility(evaluator, root, extracted.utility or UtilityInfo())
    await verify_part2_bank(evaluator, root, extracted.bank or BankInfo())
    await verify_part3_yields(evaluator, root, extracted.utility or UtilityInfo(), extracted.bank or BankInfo(), extracted.yields or YieldInfo())

    # Optional: add custom info about evaluation context
    evaluator.add_custom_info(
        {"as_of": "February 2026", "notes": "All financial metrics intended to reflect latest info as of the stated timeframe."},
        info_type="context",
        info_name="recency_context",
    )

    return evaluator.get_summary()