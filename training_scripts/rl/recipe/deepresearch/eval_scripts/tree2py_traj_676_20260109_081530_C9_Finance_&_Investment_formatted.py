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
TASK_ID = "portfolio_etf_selection"
TASK_DESCRIPTION = """You are building a diversified investment portfolio and need to identify three Exchange-Traded Funds (ETFs) that meet specific criteria. Find the following:

1. A U.S. Large-Cap Equity ETF that:
   - Is issued by one of the major providers: BlackRock (iShares), Vanguard, State Street (SPDR), Invesco, or Fidelity
   - Primarily invests in U.S. large-cap stocks (companies with market capitalization of $10 billion or more)
   - Has an expense ratio below 0.50%
   - Has at least $100 million in assets under management (AUM)
   - Was launched at least 3 years ago (inception date before January 2023)
   - Provides daily holdings disclosure on its website
   - Has publicly available 1-year, 3-year, and 5-year performance returns
   - Discloses risk metrics including beta and standard deviation

2. A U.S. Aggregate Bond ETF that:
   - Is issued by one of the major providers: BlackRock (iShares), Vanguard, State Street (SPDR), Invesco, or Fidelity
   - Provides broad exposure to the U.S. investment-grade bond market
   - Has an expense ratio below 0.50%
   - Has at least $100 million in assets under management (AUM)
   - Was launched at least 3 years ago (inception date before January 2023)
   - Provides daily holdings disclosure on its website
   - Discloses the 30-day SEC yield, weighted average maturity, average credit quality, and effective duration

3. A U.S. Technology Sector ETF that:
   - Is issued by one of the major providers: BlackRock (iShares), Vanguard, State Street (SPDR), Invesco, or Fidelity
   - Primarily invests in U.S. technology sector companies
   - Has an expense ratio below 0.50%
   - Has at least $100 million in assets under management (AUM)
   - Was launched at least 3 years ago (inception date before January 2023)
   - Provides daily holdings disclosure on its website
   - Has publicly available 1-year, 3-year, and 5-year performance returns
   - Discloses risk metrics including beta and standard deviation

For each ETF, provide:
- The ticker symbol and full name
- The issuer
- Inception date
- Expense ratio and AUM
- Relevant performance metrics (returns for equity/tech ETFs; bond-specific metrics for bond ETF)
- Risk metrics (beta and standard deviation where applicable)
- Holdings information (top holdings, concentration percentages, sector/subsector allocations)
- Distribution frequency and current yield
- URL references to verify all provided information

All information must be current and verifiable through official ETF provider websites or reputable financial data sources.
"""

ALLOWED_ISSUER_HINT = "Allowed issuers: BlackRock (iShares), Vanguard, State Street (SPDR), Invesco, Fidelity."
CUTOFF_INCEPTION_TEXT = "January 2023"  # Fixed textual cutoff used in claims


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class Holding(BaseModel):
    name: Optional[str] = None
    weight_percent: Optional[str] = None  # Keep as string to allow formats like "6.1%"


class Allocation(BaseModel):
    label: Optional[str] = None  # sector/subsector/composition bucket label
    percent: Optional[str] = None


class EquityLikeETF(BaseModel):
    ticker: Optional[str] = None
    name: Optional[str] = None
    issuer: Optional[str] = None
    inception_date: Optional[str] = None
    expense_ratio: Optional[str] = None
    aum: Optional[str] = None
    performance_1y: Optional[str] = None
    performance_3y: Optional[str] = None
    performance_5y: Optional[str] = None
    beta: Optional[str] = None
    standard_deviation: Optional[str] = None
    top10_holdings: List[Holding] = Field(default_factory=list)
    top10_concentration: Optional[str] = None
    sector_allocations: List[Allocation] = Field(default_factory=list)  # sector/subsector breakdown
    distribution_frequency: Optional[str] = None
    current_yield: Optional[str] = None
    source_urls: List[str] = Field(default_factory=list)


class BondETF(BaseModel):
    ticker: Optional[str] = None
    name: Optional[str] = None
    issuer: Optional[str] = None
    inception_date: Optional[str] = None
    expense_ratio: Optional[str] = None
    aum: Optional[str] = None
    sec_yield_30d: Optional[str] = None
    weighted_average_maturity: Optional[str] = None
    average_credit_quality: Optional[str] = None
    effective_duration: Optional[str] = None
    top_holdings: List[Holding] = Field(default_factory=list)
    composition_allocations: List[Allocation] = Field(default_factory=list)  # sector/quality/maturity buckets, etc.
    distribution_frequency: Optional[str] = None
    current_yield: Optional[str] = None
    source_urls: List[str] = Field(default_factory=list)


class PortfolioETFExtraction(BaseModel):
    large_cap_equity: Optional[EquityLikeETF] = None
    aggregate_bond: Optional[BondETF] = None
    technology: Optional[EquityLikeETF] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_portfolio_etfs() -> str:
    return """
    Extract, from the answer text, exactly one ETF for each category:
    1) U.S. Large-Cap Equity ETF
    2) U.S. Aggregate Bond ETF
    3) U.S. Technology Sector ETF

    Rules:
    - If multiple candidates are present in the answer for a category, extract only the first one mentioned for that category.
    - Extract values exactly as stated in the answer (do not infer).
    - Extract source URLs exactly as they appear in the answer. Do not invent URLs.
    - Keep numeric values as strings (e.g., "0.03%", "$500B", "8.6 years").
    - For holdings/allocations, extract as many as clearly provided. If none, leave as empty arrays.

    For the two equity-like ETFs (large-cap equity and technology), extract into the corresponding fields:
    - ticker
    - name
    - issuer
    - inception_date
    - expense_ratio
    - aum
    - performance_1y
    - performance_3y
    - performance_5y
    - beta
    - standard_deviation
    - top10_holdings: array of {name, weight_percent}
    - top10_concentration
    - sector_allocations: array of {label, percent} (sectors or subsectors)
    - distribution_frequency
    - current_yield
    - source_urls: list of URLs included in the answer that relate to this ETF (fund page, factsheet, holdings, index, reputable data sites, etc.)

    For the U.S. Aggregate Bond ETF, extract:
    - ticker
    - name
    - issuer
    - inception_date
    - expense_ratio
    - aum
    - sec_yield_30d
    - weighted_average_maturity
    - average_credit_quality
    - effective_duration
    - top_holdings: array of {name, weight_percent}
    - composition_allocations: array of {label, percent} (e.g., sector/quality/maturity allocations as provided)
    - distribution_frequency
    - current_yield
    - source_urls: list of URLs included in the answer that relate to this ETF

    Return a JSON object with the following top-level fields:
    - large_cap_equity: object as defined above (or null if not provided in the answer)
    - aggregate_bond: object as defined above (or null if not provided in the answer)
    - technology: object as defined above (or null if not provided in the answer)
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def nonempty(s: Optional[str]) -> bool:
    return bool(s and str(s).strip())


def has_any_urls(urls: Optional[List[str]]) -> bool:
    return bool(urls and len(urls) > 0)


def truncate_list_str(items: List[str], max_len: int = 8) -> str:
    if not items:
        return ""
    if len(items) <= max_len:
        return ", ".join(items)
    return ", ".join(items[:max_len]) + f", ... (+{len(items) - max_len} more)"


def combine_sources(item_urls: Optional[List[str]]) -> List[str]:
    return [u for u in (item_urls or []) if nonempty(u)]


# --------------------------------------------------------------------------- #
# Verification builders                                                       #
# --------------------------------------------------------------------------- #
async def verify_equity_like_etf(
    evaluator: Evaluator,
    parent_node,
    item: EquityLikeETF,
    etf_id_prefix: str,
    category_desc: str,
) -> None:
    """
    Verify one equity-like ETF (U.S. Large-Cap Equity or U.S. Technology Sector).
    Follows the rubric structure with critical gating and parallel aggregation.
    """
    etf_node = evaluator.add_parallel(
        id=f"{etf_id_prefix}",
        desc=f"One {category_desc} ETF meeting all stated requirements, with required fields and sources provided.",
        parent=parent_node,
        critical=True,  # Parent ('Portfolio_ETF_Selection') is critical; all children must be critical
    )

    # ---------------- Eligibility Constraints (critical parallel) ----------------
    elig_node = evaluator.add_parallel(
        id=f"{etf_id_prefix}_eligibility",
        desc=f"ETF meets all eligibility constraints for the {category_desc} category.",
        parent=etf_node,
        critical=True,
    )

    # Issuer from allowed providers (simple check)
    issuer_leaf = evaluator.add_leaf(
        id=f"{etf_id_prefix}_issuer_allowed",
        desc="Issuer is one of: BlackRock (iShares), Vanguard, State Street (SPDR), Invesco, Fidelity.",
        parent=elig_node,
        critical=True,
    )
    issuer_claim = (
        f"The issuer '{item.issuer}' is one of the allowed providers: BlackRock (iShares), Vanguard, State Street (SPDR), Invesco, or Fidelity."
    )
    await evaluator.verify(
        claim=issuer_claim,
        node=issuer_leaf,
        additional_instruction="Consider common brand relationships as equivalent (e.g., iShares=BlackRock; SPDR=State Street). If issuer is missing or not clearly allowed, mark as Incorrect."
    )

    # Category focus (equity-like variant-specific)
    focus_leaf = evaluator.add_leaf(
        id=f"{etf_id_prefix}_category_focus",
        desc=f"Primarily invests in U.S. {('large-cap stocks' if 'Large-Cap' in category_desc else 'technology sector companies')}.",
        parent=elig_node,
        critical=True,
    )
    focus_claim = (
        f"The ETF {item.name or '(selected ETF)'} ({item.ticker or 'ticker unavailable'}) primarily invests in "
        f"{'U.S. large-cap stocks (companies with market capitalization of $10B or more)' if 'Large-Cap' in category_desc else 'U.S. technology sector companies'}."
    )
    await evaluator.verify(
        claim=focus_claim,
        node=focus_leaf,
        sources=combine_sources(item.source_urls),
        additional_instruction="Use the fund page/factsheet/index methodology or sector exposure to determine primary focus."
    )

    # Expense ratio under 0.50%
    expense_leaf = evaluator.add_leaf(
        id=f"{etf_id_prefix}_expense_under_050",
        desc="Expense ratio is below 0.50%.",
        parent=elig_node,
        critical=True,
    )
    if nonempty(item.expense_ratio):
        expense_claim = f"The ETF's expense ratio is {item.expense_ratio}, and it is below 0.50%."
    else:
        expense_claim = "The ETF's expense ratio is below 0.50%."
    await evaluator.verify(
        claim=expense_claim,
        node=expense_leaf,
        sources=combine_sources(item.source_urls),
        additional_instruction="Verify the stated expense ratio on the fund page or factsheet."
    )

    # AUM at least $100M
    aum_leaf = evaluator.add_leaf(
        id=f"{etf_id_prefix}_aum_over_100m",
        desc="Assets under management (AUM) is at least $100 million.",
        parent=elig_node,
        critical=True,
    )
    if nonempty(item.aum):
        aum_claim = f"The ETF's assets under management (AUM) is {item.aum}, which is at least $100 million."
    else:
        aum_claim = "The ETF's assets under management (AUM) is at least $100 million."
    await evaluator.verify(
        claim=aum_claim,
        node=aum_leaf,
        sources=combine_sources(item.source_urls),
        additional_instruction="If the AUM figure is not clear or appears below $100M, mark as Incorrect."
    )

    # Inception before Jan 2023
    inception_leaf = evaluator.add_leaf(
        id=f"{etf_id_prefix}_inception_before_2023",
        desc="Inception date is before January 2023 (launched at least 3 years ago).",
        parent=elig_node,
        critical=True,
    )
    if nonempty(item.inception_date):
        inception_claim = f"The ETF's inception date is {item.inception_date}, which is before {CUTOFF_INCEPTION_TEXT}."
    else:
        inception_claim = f"The ETF's inception date is before {CUTOFF_INCEPTION_TEXT}."
    await evaluator.verify(
        claim=inception_claim,
        node=inception_leaf,
        sources=combine_sources(item.source_urls),
        additional_instruction=f"Confirm inception date from fund page/factsheet. If inception is {CUTOFF_INCEPTION_TEXT} or later, mark as Incorrect."
    )

    # Daily holdings disclosure available
    holdings_disclosure_leaf = evaluator.add_leaf(
        id=f"{etf_id_prefix}_daily_holdings",
        desc="Provider website offers daily holdings disclosure.",
        parent=elig_node,
        critical=True,
    )
    holdings_disclosure_claim = (
        f"The ETF {item.name or '(selected ETF)'} provides daily holdings disclosure on the provider's website (a daily-updated holdings file/list)."
    )
    await evaluator.verify(
        claim=holdings_disclosure_claim,
        node=holdings_disclosure_leaf,
        sources=combine_sources(item.source_urls),
        additional_instruction="Look for a holdings page or file indicating daily update frequency."
    )

    # ---------------- Required Reported Fields (critical parallel) ----------------
    req_node = evaluator.add_parallel(
        id=f"{etf_id_prefix}_required_fields",
        desc=f"All required output fields for the {category_desc} ETF are provided.",
        parent=etf_node,
        critical=True,
    )

    # Presence checks (custom nodes)
    core_info_ok = all([
        nonempty(item.ticker),
        nonempty(item.name),
        nonempty(item.issuer),
        nonempty(item.inception_date),
        nonempty(item.expense_ratio),
        nonempty(item.aum),
    ])
    evaluator.add_custom_node(
        result=core_info_ok,
        id=f"{etf_id_prefix}_core_fields_present",
        desc="Provide ticker symbol, full name, issuer, inception date, expense ratio, and AUM.",
        parent=req_node,
        critical=True
    )

    perf_ok = all([
        nonempty(item.performance_1y),
        nonempty(item.performance_3y),
        nonempty(item.performance_5y),
    ])
    evaluator.add_custom_node(
        result=perf_ok,
        id=f"{etf_id_prefix}_performance_present",
        desc="Provide publicly available 1-year, 3-year, and 5-year performance returns.",
        parent=req_node,
        critical=True
    )

    risk_ok = all([
        nonempty(item.beta),
        nonempty(item.standard_deviation),
    ])
    evaluator.add_custom_node(
        result=risk_ok,
        id=f"{etf_id_prefix}_risk_present",
        desc="Provide risk metrics including beta and standard deviation.",
        parent=req_node,
        critical=True
    )

    holdings_ok = (len(item.top10_holdings) > 0) and nonempty(item.top10_concentration)
    evaluator.add_custom_node(
        result=holdings_ok,
        id=f"{etf_id_prefix}_holdings_top10_conc_present",
        desc="Provide top 10 holdings with weights AND the concentration percentage for the top 10 holdings.",
        parent=req_node,
        critical=True
    )

    sectors_ok = (len(item.sector_allocations) > 0)
    evaluator.add_custom_node(
        result=sectors_ok,
        id=f"{etf_id_prefix}_sector_allocations_present",
        desc="Provide sector/subsector allocation percentages.",
        parent=req_node,
        critical=True
    )

    dist_yield_ok = all([
        nonempty(item.distribution_frequency),
        nonempty(item.current_yield),
    ])
    evaluator.add_custom_node(
        result=dist_yield_ok,
        id=f"{etf_id_prefix}_distribution_yield_present",
        desc="Provide distribution frequency and current yield.",
        parent=req_node,
        critical=True
    )

    # ---------------- Sources (critical parallel with single leaf) ----------------
    sources_node = evaluator.add_parallel(
        id=f"{etf_id_prefix}_sources",
        desc="Provide URL references that verify the provided information, using official ETF provider sites and/or reputable financial data sources.",
        parent=etf_node,
        critical=True,
    )

    urls_leaf = evaluator.add_leaf(
        id=f"{etf_id_prefix}_urls_verify_all",
        desc="URLs are provided and collectively support the eligibility constraints and all reported fields above.",
        parent=sources_node,
        critical=True,
    )

    # Build a compound claim summarizing the key facts the URLs should support
    top10_names = [h.name for h in item.top10_holdings if nonempty(h.name)]
    sectors = [a.label for a in item.sector_allocations if nonempty(a.label)]

    compound_claim = (
        f"The provided URLs collectively support the details for ETF {item.name or '(selected ETF)'} "
        f"({item.ticker or 'ticker unavailable'}): issuer {item.issuer or 'N/A'}; expense ratio {item.expense_ratio or 'N/A'} "
        f"(below 0.50%); AUM {item.aum or 'N/A'} (≥ $100M); inception date {item.inception_date or 'N/A'} (before {CUTOFF_INCEPTION_TEXT}); "
        f"daily holdings disclosure exists; 1Y {item.performance_1y or 'N/A'}, 3Y {item.performance_3y or 'N/A'}, 5Y {item.performance_5y or 'N/A'} returns; "
        f"risk metrics beta {item.beta or 'N/A'} and standard deviation {item.standard_deviation or 'N/A'}; "
        f"top holdings (e.g., {truncate_list_str([t for t in top10_names if t])}); top-10 concentration {item.top10_concentration or 'N/A'}; "
        f"sector/subsector allocations (e.g., {truncate_list_str([s for s in sectors if s])}); "
        f"distribution frequency {item.distribution_frequency or 'N/A'} and current yield {item.current_yield or 'N/A'}."
    )
    await evaluator.verify(
        claim=compound_claim,
        node=urls_leaf,
        sources=combine_sources(item.source_urls),
        additional_instruction="Judge as Incorrect if the URLs are missing or do not substantiate the enumerated details. Prefer official provider pages (fund page, factsheet, holdings) and reputable data sources (e.g., Morningstar)."
    )


async def verify_bond_etf(
    evaluator: Evaluator,
    parent_node,
    item: BondETF,
    etf_id_prefix: str,
) -> None:
    """
    Verify the U.S. Aggregate Bond ETF according to the rubric.
    """
    etf_node = evaluator.add_parallel(
        id=f"{etf_id_prefix}",
        desc="One U.S. aggregate bond ETF meeting all stated requirements, with required fields and sources provided.",
        parent=parent_node,
        critical=True,
    )

    # ---------------- Eligibility Constraints ----------------
    elig_node = evaluator.add_parallel(
        id=f"{etf_id_prefix}_eligibility",
        desc="ETF meets all eligibility constraints for the U.S. aggregate bond category.",
        parent=etf_node,
        critical=True,
    )

    issuer_leaf = evaluator.add_leaf(
        id=f"{etf_id_prefix}_issuer_allowed",
        desc="Issuer is one of: BlackRock (iShares), Vanguard, State Street (SPDR), Invesco, Fidelity.",
        parent=elig_node,
        critical=True,
    )
    issuer_claim = (
        f"The issuer '{item.issuer}' is one of the allowed providers: BlackRock (iShares), Vanguard, State Street (SPDR), Invesco, or Fidelity."
    )
    await evaluator.verify(
        claim=issuer_claim,
        node=issuer_leaf,
        additional_instruction="Consider iShares=BlackRock and SPDR=State Street equivalences. If issuer missing or not allowed, mark Incorrect."
    )

    exposure_leaf = evaluator.add_leaf(
        id=f"{etf_id_prefix}_broad_investment_grade",
        desc="Provides broad exposure to the U.S. investment-grade bond market.",
        parent=elig_node,
        critical=True,
    )
    exposure_claim = (
        f"The ETF {item.name or '(selected ETF)'} ({item.ticker or 'ticker unavailable'}) provides broad exposure to the U.S. investment-grade bond market."
    )
    await evaluator.verify(
        claim=exposure_claim,
        node=exposure_leaf,
        sources=combine_sources(item.source_urls),
        additional_instruction="Use the fund page/factsheet/index summary to confirm aggregate, investment-grade U.S. exposure."
    )

    expense_leaf = evaluator.add_leaf(
        id=f"{etf_id_prefix}_expense_under_050",
        desc="Expense ratio is below 0.50%.",
        parent=elig_node,
        critical=True,
    )
    if nonempty(item.expense_ratio):
        expense_claim = f"The ETF's expense ratio is {item.expense_ratio}, and it is below 0.50%."
    else:
        expense_claim = "The ETF's expense ratio is below 0.50%."
    await evaluator.verify(
        claim=expense_claim,
        node=expense_leaf,
        sources=combine_sources(item.source_urls),
        additional_instruction="Confirm fee on fund page/factsheet."
    )

    aum_leaf = evaluator.add_leaf(
        id=f"{etf_id_prefix}_aum_over_100m",
        desc="Assets under management (AUM) is at least $100 million.",
        parent=elig_node,
        critical=True,
    )
    if nonempty(item.aum):
        aum_claim = f"The ETF's assets under management (AUM) is {item.aum}, which is at least $100 million."
    else:
        aum_claim = "The ETF's assets under management (AUM) is at least $100 million."
    await evaluator.verify(
        claim=aum_claim,
        node=aum_leaf,
        sources=combine_sources(item.source_urls),
        additional_instruction="If AUM appears < $100M or unclear, mark Incorrect."
    )

    inception_leaf = evaluator.add_leaf(
        id=f"{etf_id_prefix}_inception_before_2023",
        desc="Inception date is before January 2023 (launched at least 3 years ago).",
        parent=elig_node,
        critical=True,
    )
    if nonempty(item.inception_date):
        inception_claim = f"The ETF's inception date is {item.inception_date}, which is before {CUTOFF_INCEPTION_TEXT}."
    else:
        inception_claim = f"The ETF's inception date is before {CUTOFF_INCEPTION_TEXT}."
    await evaluator.verify(
        claim=inception_claim,
        node=inception_leaf,
        sources=combine_sources(item.source_urls),
        additional_instruction=f"Verify inception date. If it is {CUTOFF_INCEPTION_TEXT} or later, mark Incorrect."
    )

    holdings_disclosure_leaf = evaluator.add_leaf(
        id=f"{etf_id_prefix}_daily_holdings",
        desc="Provider website offers daily holdings disclosure.",
        parent=elig_node,
        critical=True,
    )
    holdings_disclosure_claim = (
        f"The ETF {item.name or '(selected ETF)'} provides daily holdings disclosure on the provider website (daily holdings page/file)."
    )
    await evaluator.verify(
        claim=holdings_disclosure_claim,
        node=holdings_disclosure_leaf,
        sources=combine_sources(item.source_urls),
        additional_instruction="Look for a holdings page/file indicating daily updates."
    )

    # ---------------- Required Reported Fields ----------------
    req_node = evaluator.add_parallel(
        id=f"{etf_id_prefix}_required_fields",
        desc="All required output fields for the bond ETF are provided.",
        parent=etf_node,
        critical=True,
    )

    # Ticker, name, issuer, inception, expense, AUM
    core_info_ok = all([
        nonempty(item.ticker),
        nonempty(item.name),
        nonempty(item.issuer),
        nonempty(item.inception_date),
        nonempty(item.expense_ratio),
        nonempty(item.aum),
    ])
    evaluator.add_custom_node(
        result=core_info_ok,
        id=f"{etf_id_prefix}_core_fields_present",
        desc="Provide ticker symbol, full name, issuer, inception date, expense ratio, and AUM.",
        parent=req_node,
        critical=True
    )

    # Bond-specific metrics
    bond_metrics_ok = all([
        nonempty(item.sec_yield_30d),
        nonempty(item.weighted_average_maturity),
        nonempty(item.average_credit_quality),
        nonempty(item.effective_duration),
    ])
    evaluator.add_custom_node(
        result=bond_metrics_ok,
        id=f"{etf_id_prefix}_bond_metrics_present",
        desc="Provide 30-day SEC yield, weighted average maturity, average credit quality, and effective duration.",
        parent=req_node,
        critical=True
    )

    # Holdings information (top holdings/weights and composition/allocation percentages if presented)
    holdings_info_ok = (len(item.top_holdings) > 0) or (len(item.composition_allocations) > 0)
    evaluator.add_custom_node(
        result=holdings_info_ok,
        id=f"{etf_id_prefix}_holdings_info_present",
        desc="Provide holdings information (top holdings/weights and applicable composition/allocation percentages).",
        parent=req_node,
        critical=True
    )

    # Distribution frequency and current yield
    # Accept either current_yield or sec_yield_30d to count as "current yield" if the answer framed SEC yield as current yield
    dist_yield_ok = nonempty(item.distribution_frequency) and (nonempty(item.current_yield) or nonempty(item.sec_yield_30d))
    evaluator.add_custom_node(
        result=dist_yield_ok,
        id=f"{etf_id_prefix}_distribution_yield_present",
        desc="Provide distribution frequency and current yield.",
        parent=req_node,
        critical=True
    )

    # ---------------- Sources ----------------
    sources_node = evaluator.add_parallel(
        id=f"{etf_id_prefix}_sources",
        desc="Provide URL references that verify the provided information, using official ETF provider sites and/or reputable financial data sources.",
        parent=etf_node,
        critical=True,
    )

    urls_leaf = evaluator.add_leaf(
        id=f"{etf_id_prefix}_urls_verify_all",
        desc="URLs are provided and collectively support the eligibility constraints and all reported fields above.",
        parent=sources_node,
        critical=True,
    )

    top_names = [h.name for h in item.top_holdings if nonempty(h.name)]
    comp_labels = [a.label for a in item.composition_allocations if nonempty(a.label)]

    compound_claim = (
        f"The provided URLs collectively support the details for ETF {item.name or '(selected ETF)'} "
        f"({item.ticker or 'ticker unavailable'}): issuer {item.issuer or 'N/A'}; expense ratio {item.expense_ratio or 'N/A'} (<0.50%); "
        f"AUM {item.aum or 'N/A'} (≥ $100M); inception {item.inception_date or 'N/A'} (before {CUTOFF_INCEPTION_TEXT}); daily holdings disclosure; "
        f"broad exposure to U.S. investment-grade bonds; 30-day SEC yield {item.sec_yield_30d or 'N/A'}, weighted average maturity {item.weighted_average_maturity or 'N/A'}, "
        f"average credit quality {item.average_credit_quality or 'N/A'}, effective duration {item.effective_duration or 'N/A'}; "
        f"holdings (e.g., {truncate_list_str([t for t in top_names if t])}); composition/allocations (e.g., {truncate_list_str([c for c in comp_labels if c])}); "
        f"distribution frequency {item.distribution_frequency or 'N/A'} and current yield {item.current_yield or 'N/A'}."
    )
    await evaluator.verify(
        claim=compound_claim,
        node=urls_leaf,
        sources=combine_sources(item.source_urls),
        additional_instruction="If the URLs are missing or do not substantiate these facts, judge as Incorrect. Prefer official fund sources and reputable financial data sites."
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
    Evaluate an answer for the diversified ETF portfolio selection task using a hierarchical verification tree.
    """
    # Initialize evaluator (root is non-critical by design in the framework)
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

    # Ground truth policy/constraints information (for transparency in summary)
    evaluator.add_ground_truth({
        "allowed_issuers": ["BlackRock (iShares)", "Vanguard", "State Street (SPDR)", "Invesco", "Fidelity"],
        "large_cap_equity_constraints": [
            "US large-cap focus", "expense ratio < 0.50%", "AUM >= $100M",
            f"inception before {CUTOFF_INCEPTION_TEXT}", "daily holdings disclosure",
            "1Y/3Y/5Y returns", "beta & standard deviation"
        ],
        "aggregate_bond_constraints": [
            "broad US investment-grade exposure", "expense ratio < 0.50%", "AUM >= $100M",
            f"inception before {CUTOFF_INCEPTION_TEXT}", "daily holdings disclosure",
            "30-day SEC yield, WAM, avg credit quality, effective duration"
        ],
        "technology_sector_constraints": [
            "US technology sector focus", "expense ratio < 0.50%", "AUM >= $100M",
            f"inception before {CUTOFF_INCEPTION_TEXT}", "daily holdings disclosure",
            "1Y/3Y/5Y returns", "beta & standard deviation"
        ],
    }, gt_type="constraints")

    # Extract structured info from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_portfolio_etfs(),
        template_class=PortfolioETFExtraction,
        extraction_name="etf_extraction",
    )

    # Build the top-level critical node (mirrors JSON root critical=true)
    portfolio_node = evaluator.add_parallel(
        id="Portfolio_ETF_Selection",
        desc="Identify three ETFs (U.S. large-cap equity, U.S. aggregate bond, U.S. technology sector) that each satisfy the stated constraints and provide the required disclosures/metrics with verifiable sources.",
        parent=root,
        critical=True,
    )

    # Create and verify each ETF subtree (all critical under the portfolio node)
    # 1) U.S. Large-Cap Equity
    await verify_equity_like_etf(
        evaluator=evaluator,
        parent_node=portfolio_node,
        item=extracted.large_cap_equity or EquityLikeETF(),
        etf_id_prefix="US_Large_Cap_Equity_ETF",
        category_desc="U.S. Large-Cap Equity",
    )

    # 2) U.S. Aggregate Bond
    await verify_bond_etf(
        evaluator=evaluator,
        parent_node=portfolio_node,
        item=extracted.aggregate_bond or BondETF(),
        etf_id_prefix="US_Aggregate_Bond_ETF",
    )

    # 3) U.S. Technology Sector
    await verify_equity_like_etf(
        evaluator=evaluator,
        parent_node=portfolio_node,
        item=extracted.technology or EquityLikeETF(),
        etf_id_prefix="Technology_Sector_ETF",
        category_desc="U.S. Technology Sector",
    )

    # Return evaluation summary
    return evaluator.get_summary()