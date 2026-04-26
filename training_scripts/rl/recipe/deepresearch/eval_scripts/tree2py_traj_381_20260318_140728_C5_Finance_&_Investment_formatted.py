import asyncio
import logging
import re
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.llm_client.base_client import LLMClient


# -----------------------------------------------------------------------------
# Task constants
# -----------------------------------------------------------------------------
TASK_ID = "us_healthcare_largecap_criteria_2024"
TASK_DESCRIPTION = (
    "Identify a publicly traded U.S. healthcare company that meets ALL of the following criteria: "
    "(1) Fiscal year ended December 31, 2024, "
    "(2) Reported total annual revenue exceeding $100 billion for fiscal year 2024, "
    "(3) Is classified as a large accelerated filer by the SEC, "
    "(4) Pays regular quarterly cash dividends to shareholders, "
    "(5) Demonstrated positive year-over-year revenue growth from fiscal year 2023 to fiscal year 2024, and "
    "(6) Common stock trades on the New York Stock Exchange (NYSE). "
    "Provide: the company name, its NYSE stock ticker symbol, the exact total revenue amount for fiscal year 2024 (in billions of USD), "
    "the exact total revenue amount for fiscal year 2023 (in billions of USD), the calculated year-over-year revenue growth rate "
    "(as a percentage, rounded to two decimal places), the quarterly dividend amount per share (in USD), and the Form 10-K filing deadline "
    "date for fiscal year 2024 (based on large accelerated filer status)."
)

# Expected regulatory rule for the 10-K deadline of large accelerated filers
EXPECTED_10K_DEADLINE_RULE = "Large accelerated filers must file Form 10-K within 60 days after fiscal year end."
# FY2024 year-end and expected 60-day deadline (no weekend/holiday adjustment per rubric wording)
FY2024_END_ISO = "2024-12-31"
EXPECTED_10K_DEADLINE_DATE = "March 1, 2025"  # 60 days after December 31, 2024


# -----------------------------------------------------------------------------
# Utility helpers
# -----------------------------------------------------------------------------
def unique_urls(*lists: Optional[List[str]]) -> List[str]:
    seen = set()
    result: List[str] = []
    for lst in lists:
        if not lst:
            continue
        for u in lst:
            if not isinstance(u, str):
                continue
            url = u.strip()
            if not url:
                continue
            if not (url.startswith("http://") or url.startswith("https://")):
                continue
            if url not in seen:
                seen.add(url)
                result.append(url)
    return result


def parse_billions(val: Optional[str]) -> Optional[float]:
    """
    Parse a textual revenue string into a float number in billions USD.
    Accepts formats like:
      - 324.5
      - $324.5B, 324.5B, 324.5 billion, USD 324.5 billion
      - 324,500,000,000 (treated as absolute dollars -> /1e9)
    Returns None if cannot parse.
    """
    if val is None:
        return None
    s = val.strip().lower()
    if not s:
        return None

    # Remove symbols/words
    s = s.replace("usd", " ").replace("$", " ")
    s = s.replace("us$", " ").replace("u.s.", " ")
    s = s.replace(",", " ").replace("\u00a0", " ")
    s = re.sub(r"\s+", " ", s).strip()

    # Trillion -> billions * 1000
    if "trillion" in s or re.search(r"\b(\d+(\.\d+)?)\s*t\b", s):
        m = re.search(r"(-?\d+(\.\d+)?)", s)
        if m:
            return float(m.group(1)) * 1000.0

    # Billion or suffix b
    if "billion" in s or re.search(r"\b(-?\d+(\.\d+)?)\s*b\b", s):
        m = re.search(r"(-?\d+(\.\d+)?)", s)
        if m:
            return float(m.group(1))

    # Plain numeric -> assume billions (per extraction instruction)
    s_plain = s.replace(" ", "")
    if re.fullmatch(r"-?\d+(\.\d+)?", s_plain):
        try:
            return float(s_plain)
        except Exception:
            pass

    # Very large absolute dollars -> convert to billions
    digits = re.sub(r"[^\d]", "", s_plain)
    if digits and len(digits) >= 11:
        try:
            return float(digits) / 1e9
        except Exception:
            pass

    return None


def parse_percent(val: Optional[str]) -> Optional[float]:
    """Parse a percentage string to float (without the percent sign)."""
    if val is None:
        return None
    s = val.strip().lower()
    if not s:
        return None
    m = re.search(r"-?\d+(\.\d+)?", s)
    if not m:
        return None
    try:
        return float(m.group(0))
    except Exception:
        return None


def datestr_60_days_after_yyyy_mm_dd(yyyy_mm_dd: str) -> Optional[str]:
    try:
        base = datetime.strptime(yyyy_mm_dd, "%Y-%m-%d").date()
        d = base + timedelta(days=60)
        # Format like "March 1, 2025"
        return d.strftime("%B %-d, %Y") if hasattr(d, "strftime") else None
    except Exception:
        # On some systems, %-d may not be supported (Windows). Fallback to no dash:
        try:
            base = datetime.strptime(yyyy_mm_dd, "%Y-%m-%d").date()
            d = base + timedelta(days=60)
            # Use day without leading zero via int conversion
            return f"{d.strftime('%B')} {int(d.strftime('%d'))}, {d.strftime('%Y')}"
        except Exception:
            return None


def text_matches_expected_date(answer_date: Optional[str], expected_date: str) -> bool:
    """
    Weak comparison that allows common variants:
      - March 1, 2025
      - Mar 1, 2025
      - 2025-03-01
      - 03/01/2025
    """
    if not answer_date:
        return False
    a = answer_date.strip().lower()
    if not a:
        return False
    exp = expected_date.strip().lower()

    # Direct substring or equality
    if a == exp or exp in a or a in exp:
        return True

    # Normalize to ISO if the answer gives common date formats; check if ISO equals 2025-03-01
    # Try to parse several patterns
    for fmt in ["%B %d, %Y", "%b %d, %Y", "%Y-%m-%d", "%m/%d/%Y", "%Y/%m/%d"]:
        try:
            dt = datetime.strptime(answer_date.strip(), fmt).date()
            if dt.strftime("%Y-%m-%d") == "2025-03-01":
                return True
        except Exception:
            continue
    return False


# -----------------------------------------------------------------------------
# Extraction models
# -----------------------------------------------------------------------------
class CompanyExtraction(BaseModel):
    # Basic
    company_name: Optional[str] = None
    nyse_ticker: Optional[str] = None

    # Sector / listing / public status
    sector_or_industry: Optional[str] = None
    sector_sources: List[str] = Field(default_factory=list)
    nyse_sources: List[str] = Field(default_factory=list)
    us_publicly_traded_sources: List[str] = Field(default_factory=list)
    company_profile_sources: List[str] = Field(default_factory=list)

    # Fiscal year / filer status
    fiscal_year_end_date: Optional[str] = None  # expected "December 31, 2024"
    fiscal_year_end_sources: List[str] = Field(default_factory=list)

    filer_status: Optional[str] = None  # e.g., "Large accelerated filer"
    filer_status_sources: List[str] = Field(default_factory=list)

    # Revenues (billions)
    revenue_2024_billion: Optional[str] = None
    revenue_2024_sources: List[str] = Field(default_factory=list)
    revenue_2023_billion: Optional[str] = None
    revenue_2023_sources: List[str] = Field(default_factory=list)

    yoy_revenue_growth_percent: Optional[str] = None  # e.g., "7.85%"

    # Dividends
    dividend_quarterly_amount_usd: Optional[str] = None  # numeric string preferred, e.g., "2.00"
    dividend_sources: List[str] = Field(default_factory=list)

    # 10-K rule / deadline
    form_10k_deadline_date: Optional[str] = None  # date string provided in the answer
    form_10k_rule_sources: List[str] = Field(default_factory=list)


# -----------------------------------------------------------------------------
# Extraction prompt
# -----------------------------------------------------------------------------
def prompt_extract_company() -> str:
    return """
You will extract a single company's information from the answer. The company must be a U.S. healthcare company listed on the NYSE that satisfies all criteria in the task.
Return a JSON object following these rules:

1) company_name: The exact company name mentioned in the answer (string).
2) nyse_ticker: The NYSE stock ticker symbol provided in the answer (string, uppercase if present).
3) sector_or_industry: The sector or industry description as stated (string).
4) sector_sources: All URLs in the answer that support that the company operates in the healthcare sector (list of URLs).
5) nyse_sources: All URLs that support the company's listing/trading on the NYSE or show "NYSE: TICKER" (list of URLs).
6) us_publicly_traded_sources: Any URLs in the answer that support the company is publicly traded in the U.S. (e.g., SEC filings, NYSE page) (list of URLs).
7) company_profile_sources: Any general profile/about/overview URLs cited for the company (list of URLs).

8) fiscal_year_end_date: The fiscal year end date for FY2024 as stated in the answer (string; e.g., "December 31, 2024").
9) fiscal_year_end_sources: URLs that support the stated fiscal year end date (e.g., Form 10-K or annual report) (list of URLs).

10) filer_status: The SEC filer status provided in the answer (string; e.g., "Large accelerated filer").
11) filer_status_sources: URLs that support the stated filer status (e.g., 10-K cover page) (list of URLs).

12) revenue_2024_billion: The total revenue for fiscal year 2024 (string number in billions USD; do NOT include $ or words; examples: "327.50", "400.0").
13) revenue_2024_sources: URLs that support the FY2024 revenue figure (list of URLs).
14) revenue_2023_billion: The total revenue for fiscal year 2023 (string number in billions USD; do NOT include $ or words).
15) revenue_2023_sources: URLs that support the FY2023 revenue figure (list of URLs).

16) yoy_revenue_growth_percent: The provided year-over-year revenue growth rate from FY2023 to FY2024 as a percentage, rounded to two decimals, without the % sign (string like "7.85").

17) dividend_quarterly_amount_usd: The regular quarterly cash dividend amount per share in USD provided in the answer (string number only; e.g., "2.00", "1.65").
18) dividend_sources: URLs that support the quarterly dividend payment and amount (e.g., dividend history/press releases/exchange page) (list of URLs).

19) form_10k_deadline_date: The Form 10-K deadline date for FY2024 provided in the answer (string; any reasonable date format allowed).
20) form_10k_rule_sources: URLs that support the regulatory rule that large accelerated filers must file Form 10-K within 60 days after fiscal year end (list of URLs).

General rules:
- Extract ONLY what appears explicitly in the answer.
- For any missing value, return null (for strings) or [] (for lists).
- Ensure all URLs are valid and complete, including http/https.
- For numeric fields, return only the numeric string (no symbols, no units).
"""


# -----------------------------------------------------------------------------
# Verification builders
# -----------------------------------------------------------------------------
async def build_company_basic_info_nodes(evaluator: Evaluator, parent, data: CompanyExtraction):
    node = evaluator.add_parallel(
        id="company_basic_information",
        desc="Basic identifying information about the company",
        parent=parent,
        critical=True,
    )

    # 1) Company name provided (existence)
    evaluator.add_custom_node(
        result=bool(data.company_name and data.company_name.strip()),
        id="company_name_provided",
        desc="The answer provides a specific company name",
        parent=node,
        critical=True,
    )

    # Prepare sources
    sector_srcs = unique_urls(data.sector_sources, data.company_profile_sources, data.nyse_sources)
    public_srcs = unique_urls(data.us_publicly_traded_sources, data.nyse_sources, data.filer_status_sources)
    nyse_srcs = unique_urls(data.nyse_sources, data.company_profile_sources)

    # 2) Healthcare sector verification
    healthcare_leaf = evaluator.add_leaf(
        id="healthcare_sector",
        desc="The company operates in the healthcare sector (such as health insurance, managed care, healthcare services, or related healthcare businesses)",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"The company '{data.company_name or ''}' operates in the healthcare sector (e.g., health insurance, managed care, healthcare services, or related healthcare businesses).",
        node=healthcare_leaf,
        sources=sector_srcs if sector_srcs else None,
        additional_instruction="Use the provided sources to confirm that the company's sector/industry is healthcare-related. Accept synonyms like 'managed care', 'health insurance', 'healthcare services', 'pharmacy benefits', etc.",
    )

    # 3) US publicly traded
    public_leaf = evaluator.add_leaf(
        id="us_publicly_traded",
        desc="The company is publicly traded in the United States",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"The company '{data.company_name or ''}' is publicly traded in the United States.",
        node=public_leaf,
        sources=public_srcs if public_srcs else None,
        additional_instruction="Look for evidence such as SEC 10-K filings or exchange listing pages that indicate U.S. public trading status.",
    )

    # 4) NYSE listing
    nyse_leaf = evaluator.add_leaf(
        id="nyse_listing",
        desc="The company's common stock trades on the New York Stock Exchange (NYSE)",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"The company's common stock trades on the New York Stock Exchange (NYSE).",
        node=nyse_leaf,
        sources=nyse_srcs if nyse_srcs else None,
        additional_instruction="Verify that the sources explicitly indicate listing/trading on NYSE.",
    )

    # 5) Ticker symbol (correct NYSE ticker)
    ticker_leaf = evaluator.add_leaf(
        id="ticker_symbol",
        desc="The answer provides the correct NYSE stock ticker symbol for the identified company",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"The NYSE stock ticker symbol for {data.company_name or ''} is '{(data.nyse_ticker or '').upper()}'.",
        node=ticker_leaf,
        sources=nyse_srcs if nyse_srcs else None,
        additional_instruction="Confirm the ticker symbol exactly as shown on official NYSE or investor relations pages. Allow case-insensitive comparison.",
    )


async def build_fiscal_and_filing_nodes(evaluator: Evaluator, parent, data: CompanyExtraction):
    node = evaluator.add_parallel(
        id="fiscal_year_and_filing_requirements",
        desc="Fiscal year timing and SEC filing requirements",
        parent=parent,
        critical=True,
    )

    # Sources
    fy_srcs = unique_urls(data.fiscal_year_end_sources)
    filer_srcs = unique_urls(data.filer_status_sources)
    rule_srcs = unique_urls(data.form_10k_rule_sources)

    # Fiscal year end: must be Dec 31, 2024
    fy_leaf = evaluator.add_leaf(
        id="fiscal_year_end_date",
        desc="The company's fiscal year ended on December 31, 2024",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim="The company's fiscal year ended on December 31, 2024.",
        node=fy_leaf,
        sources=fy_srcs if fy_srcs else None,
        additional_instruction="Confirm from FY2024 Form 10-K or authoritative filings. The year-end must be exactly December 31, 2024.",
    )

    # Large accelerated filer status
    filer_leaf = evaluator.add_leaf(
        id="large_accelerated_filer_status",
        desc="The company is classified as a large accelerated filer by the SEC",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"The company '{data.company_name or ''}' is classified as a large accelerated filer.",
        node=filer_leaf,
        sources=filer_srcs if filer_srcs else None,
        additional_instruction="Check the Form 10-K cover page or similar section explicitly indicating 'Large accelerated filer'.",
    )

    # 10-K deadline: verify that the provided date matches exactly 60 days after 2024-12-31 (i.e., March 1, 2025)
    expected_calc = datestr_60_days_after_yyyy_mm_dd(FY2024_END_ISO) or EXPECTED_10K_DEADLINE_DATE
    deadline_ok = text_matches_expected_date(data.form_10k_deadline_date, EXPECTED_10K_DEADLINE_DATE)
    evaluator.add_custom_node(
        result=deadline_ok,
        id="form_10k_deadline",
        desc="The answer correctly calculates the Form 10-K filing deadline as 60 days after December 31, 2024",
        parent=node,
        critical=True,
    )

    # 10-K deadline rule references (must support the 60-day rule)
    rule_leaf = evaluator.add_leaf(
        id="form_10k_deadline_reference",
        desc="The answer provides URL references supporting the 60-day filing deadline requirement for large accelerated filers",
        parent=node,
        critical=True,
    )
    # If no rule sources at all, mark as failed directly via custom node; else verify by URLs
    if not rule_srcs:
        # Replace leaf with a failed custom node for "no sources"
        evaluator.add_custom_node(
            result=False,
            id="form_10k_deadline_reference_no_sources",
            desc="No URL references provided to support the 60-day filing deadline requirement",
            parent=node,
            critical=True,
        )
    else:
        await evaluator.verify(
            claim=EXPECTED_10K_DEADLINE_RULE,
            node=rule_leaf,
            sources=rule_srcs,
            additional_instruction="Verify from SEC or authoritative sources that large accelerated filers must file Form 10-K within 60 days after fiscal year end.",
        )


async def build_revenue_nodes(evaluator: Evaluator, parent, data: CompanyExtraction):
    node = evaluator.add_parallel(
        id="revenue_information",
        desc="Annual revenue data and growth metrics",
        parent=parent,
        critical=True,
    )

    # Presence checks (the rubric says 'The answer provides ...')
    evaluator.add_custom_node(
        result=bool(data.revenue_2024_billion and data.revenue_2024_billion.strip()),
        id="fiscal_2024_revenue",
        desc="The answer provides the total annual revenue for fiscal year 2024",
        parent=node,
        critical=True,
    )
    evaluator.add_custom_node(
        result=bool(data.revenue_2023_billion and data.revenue_2023_billion.strip()),
        id="fiscal_2023_revenue",
        desc="The answer provides the total annual revenue for fiscal year 2023",
        parent=node,
        critical=True,
    )

    # Threshold: FY2024 > $100B
    rev24 = parse_billions(data.revenue_2024_billion)
    exceeds_100b = (rev24 is not None) and (rev24 > 100.0)
    evaluator.add_custom_node(
        result=bool(exceeds_100b),
        id="fiscal_2024_revenue_exceeds_100b",
        desc="The reported fiscal year 2024 revenue exceeds $100 billion USD",
        parent=node,
        critical=True,
    )

    # Reference verifications (by URL)
    rev24_srcs = unique_urls(data.revenue_2024_sources)
    rev23_srcs = unique_urls(data.revenue_2023_sources)

    # FY2024 reference
    if not rev24_srcs:
        evaluator.add_custom_node(
            result=False,
            id="fiscal_2024_revenue_reference",
            desc="The answer provides URL references supporting the fiscal year 2024 revenue figure",
            parent=node,
            critical=True,
        )
    else:
        rev24_leaf = evaluator.add_leaf(
            id="fiscal_2024_revenue_reference",
            desc="The answer provides URL references supporting the fiscal year 2024 revenue figure",
            parent=node,
            critical=True,
        )
        await evaluator.verify(
            claim=f"The company's total revenue for fiscal year 2024 was {data.revenue_2024_billion or ''} billion USD.",
            node=rev24_leaf,
            sources=rev24_srcs,
            additional_instruction="Confirm the FY2024 total (consolidated) revenue. Allow minor rounding differences.",
        )

    # FY2023 reference
    if not rev23_srcs:
        evaluator.add_custom_node(
            result=False,
            id="fiscal_2023_revenue_reference",
            desc="The answer provides URL references supporting the fiscal year 2023 revenue figure",
            parent=node,
            critical=True,
        )
    else:
        rev23_leaf = evaluator.add_leaf(
            id="fiscal_2023_revenue_reference",
            desc="The answer provides URL references supporting the fiscal year 2023 revenue figure",
            parent=node,
            critical=True,
        )
        await evaluator.verify(
            claim=f"The company's total revenue for fiscal year 2023 was {data.revenue_2023_billion or ''} billion USD.",
            node=rev23_leaf,
            sources=rev23_srcs,
            additional_instruction="Confirm the FY2023 total (consolidated) revenue. Allow minor rounding differences.",
        )

    # Positive YoY growth (FY2024 > FY2023)
    rev23 = parse_billions(data.revenue_2023_billion)
    positive_growth = (rev24 is not None) and (rev23 is not None) and (rev24 > rev23)
    evaluator.add_custom_node(
        result=bool(positive_growth),
        id="positive_yoy_growth",
        desc="The company demonstrated positive year-over-year revenue growth from fiscal year 2023 to fiscal year 2024 (FY2024 revenue > FY2023 revenue)",
        parent=node,
        critical=True,
    )

    # Growth rate calculation correctness:
    # Expect yoy% = ((rev24 - rev23) / rev23) * 100, rounded to two decimals; compare against provided yoy_revenue_growth_percent
    def compute_yoy_percent(_r24: Optional[float], _r23: Optional[float]) -> Optional[float]:
        if _r24 is None or _r23 is None or _r23 == 0:
            return None
        return round(((_r24 - _r23) / _r23) * 100.0, 2)

    expected_yoy = compute_yoy_percent(rev24, rev23)
    provided_yoy = parse_percent(data.yoy_revenue_growth_percent)
    # Tolerance for rounding/representation issues
    yoy_ok = (expected_yoy is not None) and (provided_yoy is not None) and (abs(expected_yoy - provided_yoy) <= 0.2)

    evaluator.add_custom_node(
        result=bool(yoy_ok),
        id="growth_rate_calculation",
        desc="The answer provides the calculated year-over-year revenue growth rate as a percentage, correctly calculated using the formula: ((FY2024 Revenue - FY2023 Revenue) / FY2023 Revenue) × 100%, rounded to two decimal places",
        parent=node,
        critical=True,
    )

    # Also record the values we parsed/computed for transparency
    evaluator.add_custom_info(
        info={
            "rev24_billion_parsed": rev24,
            "rev23_billion_parsed": rev23,
            "expected_yoy_percent": expected_yoy,
            "provided_yoy_percent": provided_yoy,
        },
        info_type="computed_metrics",
        info_name="revenue_computation_details",
    )


async def build_dividend_nodes(evaluator: Evaluator, parent, data: CompanyExtraction):
    node = evaluator.add_parallel(
        id="dividend_information",
        desc="Dividend payment information",
        parent=parent,
        critical=True,
    )

    div_srcs = unique_urls(data.dividend_sources)

    # Quarterly dividend payment (policy)
    dividend_policy_leaf = evaluator.add_leaf(
        id="quarterly_dividend_payment",
        desc="The company pays regular quarterly cash dividends to shareholders",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim="The company pays regular quarterly cash dividends to shareholders.",
        node=dividend_policy_leaf,
        sources=div_srcs if div_srcs else None,
        additional_instruction="Look for dividend policy statements, dividend history pages, or press releases confirming regular quarterly cash dividends.",
    )

    # Dividend amount verification
    dividend_amount_leaf = evaluator.add_leaf(
        id="dividend_amount",
        desc="The answer provides the quarterly dividend amount per share in USD",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"The company's quarterly cash dividend per share is ${data.dividend_quarterly_amount_usd or ''}.",
        node=dividend_amount_leaf,
        sources=div_srcs if div_srcs else None,
        additional_instruction="Verify that the amount stated is a regular quarterly cash dividend (not a special dividend). Allow minor timing differences if the amount has recently changed.",
    )

    # Dividend references (presence)
    evaluator.add_custom_node(
        result=bool(div_srcs),
        id="dividend_reference",
        desc="The answer provides URL references supporting the dividend payment information",
        parent=node,
        critical=True,
    )


async def build_company_identification_tree(evaluator: Evaluator, root, data: CompanyExtraction):
    # Top-level critical node aggregating all requirements in parallel
    top = evaluator.add_parallel(
        id="company_identification",
        desc="Identification of a publicly traded U.S. healthcare company meeting all specified criteria",
        parent=root,
        critical=True,
    )

    await build_company_basic_info_nodes(evaluator, top, data)
    await build_fiscal_and_filing_nodes(evaluator, top, data)
    await build_revenue_nodes(evaluator, top, data)
    await build_dividend_nodes(evaluator, top, data)


# -----------------------------------------------------------------------------
# Main evaluation function
# -----------------------------------------------------------------------------
async def evaluate_answer(
    client: LLMClient,
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
        strategy=AggregationStrategy.PARALLEL,  # Overall independent checks aggregate here
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
        prompt=prompt_extract_company(),
        template_class=CompanyExtraction,
        extraction_name="company_extraction",
    )

    # Add ground-truth rule info for context/debugging
    evaluator.add_ground_truth(
        {
            "expected_10k_deadline_rule": EXPECTED_10K_DEADLINE_RULE,
            "fy2024_end_date": "December 31, 2024",
            "expected_10k_deadline_date": EXPECTED_10K_DEADLINE_DATE,
        },
        gt_type="regulatory_expectations",
    )

    # Build verification tree
    await build_company_identification_tree(evaluator, root, extracted)

    # Return the final summary
    return evaluator.get_summary()