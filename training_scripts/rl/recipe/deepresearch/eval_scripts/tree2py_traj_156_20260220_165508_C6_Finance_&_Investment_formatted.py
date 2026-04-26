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
TASK_ID = "pharma_constraints_2026"
TASK_DESCRIPTION = """Identify the publicly traded pharmaceutical company that simultaneously meets all of the following criteria as of February 2026:

1. The company must be classified in the pharmaceutical or biotechnology sector
2. The company must be a member of the S&P 500 Dividend Aristocrats, having increased its dividend payment for at least 25 consecutive years
3. The company must currently be included in the S&P 500 index
4. The company's market capitalization must exceed $500 billion USD
5. Institutional investors must own more than 65% of the company's outstanding shares
6. At least one of the company's top three institutional shareholders must be either Vanguard Group Inc., BlackRock Inc., or another major institutional investor holding at least 7% of the company's shares
7. The company's current dividend yield must fall between 0.5% and 1.5%
8. The company must allocate at least 15% of its annual revenue to research and development expenditures
9. The company must have publicly reported its quarterly earnings results for Q4 2025 (covering the period October-December 2025)
10. The company must have demonstrated positive year-over-year revenue growth in fiscal year 2024 compared to fiscal year 2023
11. The company's stock must trade on either the New York Stock Exchange (NYSE) or NASDAQ
12. The company must have active analyst coverage with published ratings and recommendations

Provide the company's name and stock ticker symbol, along with supporting evidence for each criterion.
"""

# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class HolderInfo(BaseModel):
    name: Optional[str] = None
    percent: Optional[str] = None  # Keep as free-form text (e.g., "7.2%")


class CompanyEvidence(BaseModel):
    # Company identity
    company_name: Optional[str] = None
    ticker: Optional[str] = None
    company_sources: List[str] = Field(default_factory=list)

    # Sector classification
    sector: Optional[str] = None
    sector_sources: List[str] = Field(default_factory=list)

    # Dividend Aristocrats membership
    dividend_aristocrats_member: Optional[bool] = None
    dividend_aristocrats_sources: List[str] = Field(default_factory=list)

    # S&P 500 inclusion
    sp500_member: Optional[bool] = None
    sp500_sources: List[str] = Field(default_factory=list)

    # Market capitalization
    market_cap: Optional[str] = None
    market_cap_sources: List[str] = Field(default_factory=list)

    # Institutional ownership
    institutional_ownership_pct: Optional[str] = None
    institutional_ownership_sources: List[str] = Field(default_factory=list)

    # Top 3 institutional holders
    top_institutional_holders: List[HolderInfo] = Field(default_factory=list)
    top_holders_sources: List[str] = Field(default_factory=list)

    # Dividend yield
    dividend_yield: Optional[str] = None
    dividend_yield_sources: List[str] = Field(default_factory=list)

    # R&D spending
    rd_spend_pct_revenue: Optional[str] = None
    rd_sources: List[str] = Field(default_factory=list)

    # Q4 2025 earnings reported
    q4_2025_reported: Optional[bool] = None
    q4_2025_sources: List[str] = Field(default_factory=list)

    # FY2024 YoY revenue growth
    fy2024_yoy_revenue_growth_positive: Optional[bool] = None
    yoy_sources: List[str] = Field(default_factory=list)

    # Exchange listing
    exchange: Optional[str] = None  # e.g., "NYSE" or "NASDAQ"
    exchange_sources: List[str] = Field(default_factory=list)

    # Analyst coverage
    analyst_coverage: Optional[bool] = None
    analyst_sources: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_company_evidence() -> str:
    return """
    Identify the single company selected in the answer and extract the following structured information exactly as presented in the answer text. For each criterion, also extract all cited source URLs supporting the claim. If a field is not explicitly stated in the answer, set it to null (or empty list for URLs). Do NOT invent values or URLs.

    Required JSON fields:
    - company_name: The company's name presented by the answer (string or null)
    - ticker: The stock ticker symbol presented by the answer (string or null)
    - company_sources: List of URLs that support the company name/ticker identity (list, can be empty)

    - sector: The sector classification text (e.g., "Pharmaceuticals", "Biotechnology") (string or null)
    - sector_sources: List of URLs cited to support the sector classification (list)

    - dividend_aristocrats_member: true/false if explicitly stated; otherwise null
    - dividend_aristocrats_sources: List of URLs cited to support Dividend Aristocrats membership (list)

    - sp500_member: true/false if explicitly stated; otherwise null
    - sp500_sources: List of URLs cited to support S&P 500 inclusion (list)

    - market_cap: The market capitalization value or description extracted from the answer (string, keep formatting like "$512B") or null
    - market_cap_sources: List of URLs cited for market cap evidence (list)

    - institutional_ownership_pct: The institutional ownership percentage extracted (string like "68%") or null
    - institutional_ownership_sources: List of URLs cited (list)

    - top_institutional_holders: Array of up to the top three holders mentioned, each with:
        • name: Holder name (string or null)
        • percent: Ownership percent (string like "7.4%") or null
    - top_holders_sources: List of URLs cited (list)

    - dividend_yield: The current dividend yield extracted (string like "0.9%") or null
    - dividend_yield_sources: List of URLs cited (list)

    - rd_spend_pct_revenue: The R&D spending as % of revenue (string like "15%") or null
    - rd_sources: List of URLs cited (list)

    - q4_2025_reported: true/false if the answer explicitly states the company reported Q4 2025 earnings (Oct–Dec 2025). Otherwise null.
    - q4_2025_sources: List of URLs cited (list)

    - fy2024_yoy_revenue_growth_positive: true/false if explicitly stated; otherwise null
    - yoy_sources: List of URLs cited (list)

    - exchange: The exchange name if stated (e.g., "NYSE" or "NASDAQ") or null
    - exchange_sources: List of URLs cited (list)

    - analyst_coverage: true/false if explicitly stated that analyst coverage with ratings exists; otherwise null
    - analyst_sources: List of URLs cited (list)

    SPECIAL NOTES:
    - Extract only URLs explicitly present in the answer (including markdown links). Do not infer URLs.
    - Keep numeric values as strings; do not convert or normalize. We will evaluate thresholds separately.
    - If multiple URLs are cited for a single criterion, include all of them.
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def safe_sources(*lists: Optional[List[str]]) -> List[str]:
    """Merge multiple source lists into a unique flattened list, removing empties."""
    merged: List[str] = []
    for lst in lists:
        if not lst:
            continue
        for u in lst:
            if u and isinstance(u, str) and u.strip() and u not in merged:
                merged.append(u.strip())
    return merged


def has_nonempty_sources(sources: Optional[List[str]]) -> bool:
    return bool(sources) and len([s for s in sources if isinstance(s, str) and s.strip()]) > 0


# --------------------------------------------------------------------------- #
# Verification construction                                                   #
# --------------------------------------------------------------------------- #
async def build_and_verify_criteria(
    evaluator: Evaluator,
    root_node,
    ev: CompanyEvidence
) -> None:
    """
    Build the verification tree according to the rubric and launch URL-grounded checks.
    We create one sequential sub-node per rubric criterion:
      - First child: existence check (custom, critical) to require sources/values
      - Second child: factual leaf verification grounded by URLs (critical)
    """

    claims_batch: List[tuple[str, List[str] | str | None, Any, Optional[str]]] = []

    # Company Identification
    company_node = evaluator.add_sequential(
        id="Company_Identification",
        desc="Provides the company’s name AND stock ticker symbol.",
        parent=root_node,
        critical=True
    )
    ci_sources = safe_sources(ev.company_sources, ev.exchange_sources)
    evaluator.add_custom_node(
        result=(bool(ev.company_name) and bool(ev.ticker) and has_nonempty_sources(ci_sources)),
        id="Company_Identification_exists",
        desc="Company name and ticker provided with at least one source URL",
        parent=company_node,
        critical=True
    )
    ci_leaf = evaluator.add_leaf(
        id="Company_Identification_verify",
        desc="Ticker corresponds to the named company (identity supported by sources)",
        parent=company_node,
        critical=True
    )
    ci_claim = f"The company's stock ticker symbol '{ev.ticker or ''}' corresponds to '{ev.company_name or ''}'."
    claims_batch.append((
        ci_claim,
        ci_sources,
        ci_leaf,
        "Verify that the cited source(s) explicitly show the company's name and its ticker symbol matching each other (e.g., exchange listing page)."
    ))

    # Sector Classification
    sector_node = evaluator.add_sequential(
        id="Sector_Classification",
        desc="Provides evidence/citation that the company is classified in the pharmaceutical or biotechnology sector.",
        parent=root_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=(bool(ev.sector) and has_nonempty_sources(ev.sector_sources)),
        id="Sector_Classification_exists",
        desc="Sector classification and at least one source URL provided",
        parent=sector_node,
        critical=True
    )
    sector_leaf = evaluator.add_leaf(
        id="Sector_Classification_verify",
        desc="Company is classified in pharmaceutical or biotechnology sector (supported by sources)",
        parent=sector_node,
        critical=True
    )
    sector_claim = "The company is classified in the pharmaceutical or biotechnology sector."
    claims_batch.append((
        sector_claim,
        ev.sector_sources,
        sector_leaf,
        "Accept reasonable synonyms (e.g., 'Pharmaceuticals', 'Biotech', 'Health Care: Pharmaceuticals & Biotechnology')."
    ))

    # Dividend Aristocrats Membership
    arist_node = evaluator.add_sequential(
        id="Dividend_Aristocrats_Membership",
        desc="Provides evidence/citation that the company is a member of the S&P 500 Dividend Aristocrats (i.e., 25+ consecutive years of dividend increases).",
        parent=root_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=has_nonempty_sources(ev.dividend_aristocrats_sources),
        id="Dividend_Aristocrats_Membership_exists",
        desc="At least one source URL provided for Dividend Aristocrats membership",
        parent=arist_node,
        critical=True
    )
    arist_leaf = evaluator.add_leaf(
        id="Dividend_Aristocrats_Membership_verify",
        desc="Company is an S&P 500 Dividend Aristocrat (supported by sources)",
        parent=arist_node,
        critical=True
    )
    arist_claim = "The company is a member of the S&P 500 Dividend Aristocrats and has increased its dividend for at least 25 consecutive years."
    claims_batch.append((
        arist_claim,
        ev.dividend_aristocrats_sources,
        arist_leaf,
        "Use official index provider pages, reputable fund fact sheets (e.g., NOBL), or credible references that explicitly list the company among Dividend Aristocrats."
    ))

    # S&P 500 Index Inclusion
    sp500_node = evaluator.add_sequential(
        id="SP500_Index_Inclusion",
        desc="Provides evidence/citation that the company is included in the S&P 500 index as of Feb 2026.",
        parent=root_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=has_nonempty_sources(ev.sp500_sources),
        id="SP500_Index_Inclusion_exists",
        desc="At least one source URL provided for S&P 500 inclusion",
        parent=sp500_node,
        critical=True
    )
    sp500_leaf = evaluator.add_leaf(
        id="SP500_Index_Inclusion_verify",
        desc="Company inclusion in the S&P 500 (supported by sources)",
        parent=sp500_node,
        critical=True
    )
    sp500_claim = "The company is currently included in the S&P 500 index."
    claims_batch.append((
        sp500_claim,
        ev.sp500_sources,
        sp500_leaf,
        "Confirm inclusion using S&P Dow Jones Indices pages, exchange/official listings, or reputable financial data pages that explicitly show S&P 500 membership."
    ))

    # Market Capitalization > $500B
    mcap_node = evaluator.add_sequential(
        id="Market_Capitalization",
        desc="Provides evidence/citation that the company’s market capitalization exceeds $500B USD as of Feb 2026.",
        parent=root_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=has_nonempty_sources(ev.market_cap_sources),
        id="Market_Capitalization_exists",
        desc="At least one source URL provided for market capitalization",
        parent=mcap_node,
        critical=True
    )
    mcap_leaf = evaluator.add_leaf(
        id="Market_Capitalization_verify",
        desc="Market capitalization exceeds $500B (supported by sources)",
        parent=mcap_node,
        critical=True
    )
    mcap_claim = "The company's market capitalization exceeds $500 billion USD."
    claims_batch.append((
        mcap_claim,
        ev.market_cap_sources,
        mcap_leaf,
        "Use the cited page(s) to verify market cap > $500B. Allow reasonable rounding/estimation and currency formatting. As-of date should be around late 2025 or early 2026."
    ))

    # Institutional Ownership > 65%
    instown_node = evaluator.add_sequential(
        id="Institutional_Ownership_Percentage",
        desc="Provides evidence/citation that institutional investors own more than 65% of the company’s outstanding shares.",
        parent=root_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=has_nonempty_sources(ev.institutional_ownership_sources),
        id="Institutional_Ownership_Percentage_exists",
        desc="At least one source URL provided for institutional ownership percentage",
        parent=instown_node,
        critical=True
    )
    instown_leaf = evaluator.add_leaf(
        id="Institutional_Ownership_Percentage_verify",
        desc="Institutional ownership exceeds 65% (supported by sources)",
        parent=instown_node,
        critical=True
    )
    instown_claim = "Institutional investors own more than 65% of the company's outstanding shares."
    claims_batch.append((
        instown_claim,
        ev.institutional_ownership_sources,
        instown_leaf,
        "Confirm using credible ownership data pages (e.g., exchange, NASDAQ/NYSE profile, FactSet, Morningstar) that explicitly show >65% institutional ownership."
    ))

    # Top 3 Institutional Holder Requirement
    top3_node = evaluator.add_sequential(
        id="Top_3_Institutional_Holder_Requirement",
        desc="Provides evidence/citation that at least one of the company’s top three institutional shareholders is Vanguard Group Inc., BlackRock Inc., or another major investor holding at least 7%.",
        parent=root_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=has_nonempty_sources(ev.top_holders_sources),
        id="Top_3_Institutional_Holder_Requirement_exists",
        desc="At least one source URL provided for top institutional holders",
        parent=top3_node,
        critical=True
    )
    top3_leaf = evaluator.add_leaf(
        id="Top_3_Institutional_Holder_Requirement_verify",
        desc="Top-3 includes Vanguard/BlackRock or another ≥7% holder (supported by sources)",
        parent=top3_node,
        critical=True
    )
    top3_claim = ("Among the company's top three institutional shareholders, at least one is Vanguard Group Inc. or BlackRock Inc., "
                  "or another institutional investor holding at least 7% of the company's shares.")
    claims_batch.append((
        top3_claim,
        ev.top_holders_sources,
        top3_leaf,
        "Verify the top three institutional holders and their percentages; accept minor naming variants (e.g., 'The Vanguard Group, Inc.', 'BlackRock Fund Advisors'). Threshold is ≥7%."
    ))

    # Dividend Yield between 0.5% and 1.5%
    dy_node = evaluator.add_sequential(
        id="Dividend_Yield",
        desc="Provides evidence/citation that the company’s current dividend yield is between 0.5% and 1.5%.",
        parent=root_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=has_nonempty_sources(ev.dividend_yield_sources),
        id="Dividend_Yield_exists",
        desc="At least one source URL provided for dividend yield",
        parent=dy_node,
        critical=True
    )
    dy_leaf = evaluator.add_leaf(
        id="Dividend_Yield_verify",
        desc="Dividend yield in [0.5%, 1.5%] (supported by sources)",
        parent=dy_node,
        critical=True
    )
    dy_claim = "The company's current dividend yield falls between 0.5% and 1.5%."
    claims_batch.append((
        dy_claim,
        ev.dividend_yield_sources,
        dy_leaf,
        "Use the cited financial data page to confirm the current dividend yield is within [0.5%, 1.5%]. Allow reasonable rounding."
    ))

    # R&D Spending ≥ 15% of revenue
    rd_node = evaluator.add_sequential(
        id="R_and_D_Spending",
        desc="Provides evidence/citation that the company allocates at least 15% of annual revenue to R&D expenditures.",
        parent=root_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=has_nonempty_sources(ev.rd_sources),
        id="R_and_D_Spending_exists",
        desc="At least one source URL provided for R&D spending",
        parent=rd_node,
        critical=True
    )
    rd_leaf = evaluator.add_leaf(
        id="R_and_D_Spending_verify",
        desc="R&D spending ≥ 15% of annual revenue (supported by sources)",
        parent=rd_node,
        critical=True
    )
    rd_claim = "The company allocates at least 15% of annual revenue to research and development expenditures."
    claims_batch.append((
        rd_claim,
        ev.rd_sources,
        rd_leaf,
        "Confirm with company filings, annual reports, or credible financial analysis that shows R&D expense as a percentage of revenue ≥ 15%."
    ))

    # Q4 2025 Earnings Reported
    q4_node = evaluator.add_sequential(
        id="Q4_2025_Earnings_Reported",
        desc="Provides evidence/citation that the company publicly reported quarterly earnings results for Q4 2025 (covering Oct–Dec 2025).",
        parent=root_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=has_nonempty_sources(ev.q4_2025_sources),
        id="Q4_2025_Earnings_Reported_exists",
        desc="At least one source URL provided for Q4 2025 earnings report",
        parent=q4_node,
        critical=True
    )
    q4_leaf = evaluator.add_leaf(
        id="Q4_2025_Earnings_Reported_verify",
        desc="Q4 2025 earnings were publicly reported (supported by sources)",
        parent=q4_node,
        critical=True
    )
    q4_claim = "The company publicly reported quarterly earnings results for Q4 2025 (covering October–December 2025)."
    claims_batch.append((
        q4_claim,
        ev.q4_2025_sources,
        q4_leaf,
        "Accept fiscal calendars that map Q4 to Oct–Dec 2025. Verify press releases, 10-Q/8-K filings, or credible news that explicitly states Q4 2025 results."
    ))

    # FY2024 Revenue Growth Positive
    yoy_node = evaluator.add_sequential(
        id="FY2024_Revenue_Growth",
        desc="Provides evidence/citation that fiscal year 2024 revenue is higher than fiscal year 2023 revenue (positive YoY revenue growth).",
        parent=root_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=has_nonempty_sources(ev.yoy_sources),
        id="FY2024_Revenue_Growth_exists",
        desc="At least one source URL provided for FY2024 YoY revenue growth",
        parent=yoy_node,
        critical=True
    )
    yoy_leaf = evaluator.add_leaf(
        id="FY2024_Revenue_Growth_verify",
        desc="FY2024 revenue > FY2023 revenue (supported by sources)",
        parent=yoy_node,
        critical=True
    )
    yoy_claim = "The company demonstrated positive year-over-year revenue growth in fiscal year 2024 compared to fiscal year 2023."
    claims_batch.append((
        yoy_claim,
        ev.yoy_sources,
        yoy_leaf,
        "Verify using company filings, annual reports, or credible financial data pages that explicitly compare FY2024 vs. FY2023 revenue."
    ))

    # Exchange Listing (NYSE or NASDAQ)
    exch_node = evaluator.add_sequential(
        id="Exchange_Listing",
        desc="Provides evidence/citation that the stock trades on NYSE or NASDAQ.",
        parent=root_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=has_nonempty_sources(ev.exchange_sources),
        id="Exchange_Listing_exists",
        desc="At least one source URL provided for exchange listing",
        parent=exch_node,
        critical=True
    )
    exch_leaf = evaluator.add_leaf(
        id="Exchange_Listing_verify",
        desc="Stock trades on NYSE or NASDAQ (supported by sources)",
        parent=exch_node,
        critical=True
    )
    exch_claim = "The company's stock trades on either NYSE or NASDAQ."
    claims_batch.append((
        exch_claim,
        ev.exchange_sources,
        exch_leaf,
        "Confirm exchange listing using official exchange pages or authoritative finance sites showing 'NYSE: TICKER' or 'NASDAQ: TICKER'."
    ))

    # Analyst Coverage (ratings/recommendations)
    analyst_node = evaluator.add_sequential(
        id="Analyst_Coverage",
        desc="Provides evidence/citation that the company has active analyst coverage with published ratings/recommendations.",
        parent=root_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=has_nonempty_sources(ev.analyst_sources),
        id="Analyst_Coverage_exists",
        desc="At least one source URL provided for analyst coverage/ratings",
        parent=analyst_node,
        critical=True
    )
    analyst_leaf = evaluator.add_leaf(
        id="Analyst_Coverage_verify",
        desc="Active analyst coverage with ratings/recommendations (supported by sources)",
        parent=analyst_node,
        critical=True
    )
    analyst_claim = "The company has active analyst coverage with published ratings and recommendations."
    claims_batch.append((
        analyst_claim,
        ev.analyst_sources,
        analyst_leaf,
        "Use credible finance sources or broker research summaries that explicitly show analyst ratings/recommendations for the company."
    ))

    # Execute all verifications in parallel to avoid cross-sibling precondition skipping
    await evaluator.batch_verify(claims_batch)


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
    Evaluate an answer for the pharmaceutical constraints task (as of Feb 2026).
    """
    # Initialize evaluator: root is non-critical parallel to compute all checks independently
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
        default_model=model
    )

    # Extract structured evidence from the answer
    ev = await evaluator.extract(
        prompt=prompt_extract_company_evidence(),
        template_class=CompanyEvidence,
        extraction_name="company_evidence"
    )

    # Build critical verification subtrees for each rubric criterion and run checks
    await build_and_verify_criteria(evaluator, root, ev)

    # Return structured summary
    return evaluator.get_summary()