import asyncio
import logging
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "chevron_dividend_eval_2026"
TASK_DESCRIPTION = """An investor is evaluating Chevron Corporation (CVX) for inclusion in their dividend-focused portfolio and needs to understand the company's dividend characteristics and tax implications.

Please provide a comprehensive analysis that addresses the following:

1. Dividend Aristocrat Status: Verify whether Chevron qualifies as an S&P 500 Dividend Aristocrat by confirming it meets the minimum requirement of 25 consecutive years of dividend increases. Specify the actual number of consecutive years Chevron has increased its dividend and confirm whether Chevron increased its dividend in 2026.

2. Q1 2026 Dividend Payment Details: For Chevron's Q1 2026 quarterly dividend, provide:
   - The dividend amount per share (in dollars)
   - The ex-dividend date (in MM/DD/YYYY format)
   - The payment date (in MM/DD/YYYY format)
   - The approximate dividend yield (as a percentage or percentage range)

3. Tax Analysis: For a single filer with a taxable income of $75,000 in 2026:
   - Determine the applicable qualified dividend tax rate
   - Calculate the after-tax dividend amount per share for the Q1 2026 dividend

4. Holding Period Requirements: To qualify for qualified dividend tax treatment on Chevron's Q1 2026 dividend:
   - Specify the minimum number of days the investor must hold the stock during the 121-day period
   - Calculate the latest date the investor can purchase Chevron stock to be eligible to receive the Q1 2026 dividend

Provide all information with supporting reference URLs from reliable sources."""


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class AristocratExtraction(BaseModel):
    s_and_p_500_member: Optional[str] = None  # yes/no/claim text
    consecutive_years: Optional[str] = None   # e.g., "37", "37+"
    meets_25_years: Optional[str] = None      # yes/no/claim text
    increased_in_2026: Optional[str] = None   # yes/no/claim text
    sources: List[str] = Field(default_factory=list)


class DividendQ1Extraction(BaseModel):
    amount_per_share_usd: Optional[str] = None        # e.g., "$1.63"
    ex_dividend_date: Optional[str] = None            # target MM/DD/YYYY if available
    payment_date: Optional[str] = None                # target MM/DD/YYYY if available
    approx_yield: Optional[str] = None                # e.g., "4.2%" or "4%–5%"
    sources: List[str] = Field(default_factory=list)


class TaxExtraction(BaseModel):
    qualified_dividend_tax_rate: Optional[str] = None  # e.g., "15%"
    after_tax_dividend_per_share: Optional[str] = None # e.g., "$1.39"
    sources: List[str] = Field(default_factory=list)


class HoldingExtraction(BaseModel):
    min_holding_days: Optional[str] = None              # e.g., "more than 60 days", "61"
    latest_purchase_date: Optional[str] = None          # e.g., "01/10/2026"
    sources: List[str] = Field(default_factory=list)


class ChevronDividendAnalysisExtract(BaseModel):
    aristocrat: Optional[AristocratExtraction] = None
    q1_2026: Optional[DividendQ1Extraction] = None
    tax_2026_single_75k: Optional[TaxExtraction] = None
    holding_period: Optional[HoldingExtraction] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_main() -> str:
    return """
    Extract the requested fields from the answer. Return exactly the JSON schema described. If something is missing in the answer, return null for that field and an empty list for URLs.

    Structure:
    {
      "aristocrat": {
        "s_and_p_500_member": string or null,     // e.g., "yes", "no", or a short claim text extracted verbatim
        "consecutive_years": string or null,      // number of consecutive years of dividend increases (keep as string as written, e.g., "37", "37+", "at least 25")
        "meets_25_years": string or null,         // "yes"/"no" or short phrase, extracted verbatim
        "increased_in_2026": string or null,      // "yes"/"no" or short phrase, extracted verbatim
        "sources": [urls...]                      // all URLs cited for Aristocrat/membership/years/2026 increase
      },
      "q1_2026": {
        "amount_per_share_usd": string or null,   // the Q1 2026 per-share dividend amount (e.g., "$1.63" or "1.63")
        "ex_dividend_date": string or null,       // ex-dividend date; prefer MM/DD/YYYY if the answer provides it; otherwise extract exactly as written
        "payment_date": string or null,           // payment date; prefer MM/DD/YYYY if provided; otherwise extract as written
        "approx_yield": string or null,           // approximate yield (e.g., "4.2%" or "4%–5%")
        "sources": [urls...]                      // all URLs for Q1 2026 amount/dates/yield
      },
      "tax_2026_single_75k": {
        "qualified_dividend_tax_rate": string or null,    // e.g., "15%"
        "after_tax_dividend_per_share": string or null,   // e.g., "$1.39"
        "sources": [urls...]                               // URLs supporting the 2026 qualified dividend rate/brackets
      },
      "holding_period": {
        "min_holding_days": string or null,               // e.g., "more than 60 days" or "61"
        "latest_purchase_date": string or null,           // the answer's stated latest purchase date to receive Q1 2026 dividend (prefer MM/DD/YYYY if provided)
        "sources": [urls...]                               // URLs supporting the holding-period rule
      }
    }

    Rules:
    - Extract only what is explicitly present in the answer.
    - For URLs: include every URL cited in the answer for each category; accept plain URLs or markdown links.
    - Do not invent values. If not present, return null.
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def _normalize_money_to_float(val: Optional[str]) -> Optional[float]:
    if not val:
        return None
    s = val.strip()
    # Remove $ and commas and spaces
    s = s.replace("$", "").replace(",", "").strip()
    # If trailing text like "per share", split
    for token in [" per share", "/share", "USD", "usd"]:
        if token in s:
            s = s.replace(token, "").strip()
    try:
        return float(s)
    except Exception:
        return None


def _normalize_percent_to_float(val: Optional[str]) -> Optional[float]:
    if not val:
        return None
    s = val.strip()
    s = s.replace("%", "").replace("percent", "").strip()
    # handle ranges like "4-5" or "4–5"
    for sep in ["–", "-", "—", " to "]:
        if sep in s:
            parts = [p.strip() for p in s.split(sep) if p.strip()]
            try:
                nums = [float(p) for p in parts]
                if len(nums) >= 2:
                    return sum(nums[:2]) / 2.0
            except Exception:
                pass
    try:
        return float(s)
    except Exception:
        return None


def _try_parse_date(dt_str: Optional[str]) -> Optional[datetime]:
    if not dt_str:
        return None
    s = dt_str.strip()
    fmts = [
        "%m/%d/%Y",
        "%Y-%m-%d",
        "%b %d, %Y",
        "%B %d, %Y",
        "%m-%d-%Y",
        "%m.%d.%Y",
    ]
    for fmt in fmts:
        try:
            return datetime.strptime(s, fmt)
        except Exception:
            continue
    return None


def _to_mmddyyyy(d: datetime) -> str:
    return d.strftime("%m/%d/%Y")


def _previous_weekday(d: datetime) -> datetime:
    d2 = d - timedelta(days=1)
    while d2.weekday() >= 5:  # 5=Sat, 6=Sun
        d2 = d2 - timedelta(days=1)
    return d2


def _dedup_urls(urls: List[str]) -> List[str]:
    seen = set()
    out = []
    for u in urls:
        u = (u or "").strip()
        if not u:
            continue
        if u not in seen:
            seen.add(u)
            out.append(u)
    return out


def _merge_url_lists(*args: List[str]) -> List[str]:
    all_urls: List[str] = []
    for lst in args:
        all_urls.extend(lst or [])
    return _dedup_urls(all_urls)


# --------------------------------------------------------------------------- #
# Verification subroutines                                                    #
# --------------------------------------------------------------------------- #
async def verify_aristocrat_block(
    evaluator: Evaluator,
    parent,
    ext: ChevronDividendAnalysisExtract,
):
    node = evaluator.add_parallel(
        id="Dividend_Aristocrat_Status",
        desc="Verify whether Chevron qualifies as an S&P 500 Dividend Aristocrat and address the question’s requested details.",
        parent=parent,
        critical=True,
    )

    arist = ext.aristocrat or AristocratExtraction()

    # Leaf 1: S&P 500 membership
    sp_leaf = evaluator.add_leaf(
        id="S&P500_Membership",
        desc="Confirm whether Chevron is a member of the S&P 500 Index (required to be an 'S&P 500 Dividend Aristocrat').",
        parent=node,
        critical=True,
    )
    sp_claim = "Chevron Corporation (CVX) is a member (constituent) of the S&P 500 Index."
    await evaluator.verify(
        claim=sp_claim,
        node=sp_leaf,
        sources=arist.sources,
        additional_instruction="If a source is an official S&P page or a credible list of S&P 500 Dividend Aristocrats (which requires S&P 500 membership), that supports membership. Allow minor naming variants.",
    )

    # Leaf 2: Consecutive dividend increase years and ≥25 threshold
    years_leaf = evaluator.add_leaf(
        id="Consecutive_Dividend_Increase_Years_And_Threshold",
        desc="State the actual number of consecutive years Chevron has increased its dividend and confirm it meets the ≥25-year Dividend Aristocrat requirement.",
        parent=node,
        critical=True,
    )
    years_text = arist.consecutive_years or "an indicated number of"
    claim_years = (
        f"Chevron has increased its dividend for {years_text} consecutive years, "
        f"and this meets or exceeds the 25-year minimum required for 'S&P 500 Dividend Aristocrats'."
    )
    await evaluator.verify(
        claim=claim_years,
        node=years_leaf,
        sources=arist.sources,
        additional_instruction="Confirm both the consecutive-years count and that it meets or exceeds 25 years. Allow small wording variations; prefer explicit statements.",
    )

    # Leaf 3: Dividend increase in 2026
    inc2026_leaf = evaluator.add_leaf(
        id="Dividend_Increase_In_2026",
        desc="Confirm whether Chevron increased its dividend in 2026.",
        parent=node,
        critical=True,
    )
    inc_flag = arist.increased_in_2026 or "yes"
    claim_inc2026 = "Chevron increased its dividend in the year 2026."
    await evaluator.verify(
        claim=claim_inc2026,
        node=inc2026_leaf,
        sources=_merge_url_lists(arist.sources, (ext.q1_2026.sources if ext.q1_2026 else [])),
        additional_instruction="Look for a 2026 press release, investor relations page, or trusted financial news confirming a 2026 increase vs. the prior dividend. If the sources clearly show a 2026 raise, mark as supported.",
    )

    return node


async def verify_q1_block(
    evaluator: Evaluator,
    parent,
    ext: ChevronDividendAnalysisExtract,
):
    node = evaluator.add_parallel(
        id="Q1_2026_Dividend_Payment_Details",
        desc="Provide the requested Q1 2026 dividend payment details for Chevron.",
        parent=parent,
        critical=True,
    )

    q1 = ext.q1_2026 or DividendQ1Extraction()
    urls = q1.sources

    # Dividend amount per share
    amt_leaf = evaluator.add_leaf(
        id="Dividend_Amount_Per_Share_USD",
        desc="Provide the Q1 2026 quarterly dividend amount per share (USD).",
        parent=node,
        critical=True,
    )
    claim_amt = f"Chevron's Q1 2026 quarterly dividend amount per share was {q1.amount_per_share_usd}."
    await evaluator.verify(
        claim=claim_amt,
        node=amt_leaf,
        sources=urls,
        additional_instruction="Confirm the per-share dividend amount from the source. Accept $-prefixed or plain numeric formats as equivalent.",
    )

    # Ex-dividend date
    exd_leaf = evaluator.add_leaf(
        id="Ex_Dividend_Date_MMDDYYYY",
        desc="Provide the Q1 2026 ex-dividend date in MM/DD/YYYY format.",
        parent=node,
        critical=True,
    )
    claim_exd = f"The ex-dividend date for Chevron's Q1 2026 dividend was {q1.ex_dividend_date}."
    await evaluator.verify(
        claim=claim_exd,
        node=exd_leaf,
        sources=urls,
        additional_instruction="Verify the ex-dividend date. Allow equivalence between MM/DD/YYYY and textual formats (e.g., Jan 15, 2026), but ensure the same calendar date.",
    )

    # Payment date
    pay_leaf = evaluator.add_leaf(
        id="Payment_Date_MMDDYYYY",
        desc="Provide the Q1 2026 payment date in MM/DD/YYYY format.",
        parent=node,
        critical=True,
    )
    claim_pay = f"The payment date for Chevron's Q1 2026 dividend was {q1.payment_date}."
    await evaluator.verify(
        claim=claim_pay,
        node=pay_leaf,
        sources=urls,
        additional_instruction="Verify the payment date. Allow equivalence between formats as long as the same calendar date is indicated.",
    )

    # Approximate dividend yield
    yld_leaf = evaluator.add_leaf(
        id="Approximate_Dividend_Yield",
        desc="Provide the approximate dividend yield as a percentage or percentage range (and indicate what price basis/timeframe it corresponds to if needed).",
        parent=node,
        critical=True,
    )
    claim_yld = f"The approximate dividend yield stated for the relevant timeframe was {q1.approx_yield}."
    await evaluator.verify(
        claim=claim_yld,
        node=yld_leaf,
        sources=urls,
        additional_instruction="Confirm that the source supports an approximate dividend yield near the stated value or range. Allow reasonable variance (about ±0.3 percentage points) due to price fluctuations over the cited timeframe.",
    )

    # Return leaves we may need as prerequisites elsewhere
    return node, amt_leaf, exd_leaf


async def verify_tax_block(
    evaluator: Evaluator,
    parent,
    ext: ChevronDividendAnalysisExtract,
    amt_leaf_for_dep,  # prerequisite from Q1 block
):
    node = evaluator.add_sequential(
        id="Tax_Analysis_2026_Single_75000",
        desc="Compute qualified-dividend tax rate and after-tax per-share dividend for a single filer with $75,000 taxable income in 2026.",
        parent=parent,
        critical=True,
    )

    tax = ext.tax_2026_single_75k or TaxExtraction()
    q1 = ext.q1_2026 or DividendQ1Extraction()

    # Qualified dividend tax rate (from sources)
    rate_leaf = evaluator.add_leaf(
        id="Qualified_Dividend_Tax_Rate",
        desc="Determine the applicable qualified dividend tax rate for a single filer with $75,000 taxable income in 2026 using the provided brackets.",
        parent=node,
        critical=True,
    )
    claim_rate = "For a single filer with $75,000 of taxable income in 2026, the applicable qualified dividend tax rate is " + str(tax.qualified_dividend_tax_rate) + "."
    await evaluator.verify(
        claim=claim_rate,
        node=rate_leaf,
        sources=tax.sources,
        additional_instruction="Use credible sources that state the qualified dividend (long-term capital gains) tax brackets/rates for 2026. Ensure the stated rate corresponds to the given income and filing status.",
    )

    # After-tax dividend per share (logical/mathematical check)
    after_leaf = evaluator.add_leaf(
        id="After_Tax_Dividend_Per_Share",
        desc="Calculate the after-tax dividend amount per share for the Q1 2026 dividend using the determined rate and the Q1 2026 dividend per share.",
        parent=node,
        critical=True,
    )
    # Build a verification claim that instructs the judge to compute
    claim_after = (
        f"Given Chevron's Q1 2026 per-share dividend {q1.amount_per_share_usd} "
        f"and a qualified dividend tax rate of {tax.qualified_dividend_tax_rate}, "
        f"the after-tax per-share dividend equals {tax.after_tax_dividend_per_share} "
        f"when rounded to the nearest cent."
    )
    await evaluator.verify(
        claim=claim_after,
        node=after_leaf,
        # No sources needed; this is a math check based on already-verified inputs
        sources=None,
        extra_prerequisites=[rate_leaf, amt_leaf_for_dep],
        additional_instruction="Compute after_tax = amount * (1 - rate). Treat '%' carefully (e.g., 15% = 0.15). Round to nearest cent. If the provided amount or rate is missing/null, consider the claim incorrect.",
    )

    return node


async def verify_holding_block(
    evaluator: Evaluator,
    parent,
    ext: ChevronDividendAnalysisExtract,
    exd_leaf_for_dep,  # prerequisite from Q1 block
):
    node = evaluator.add_parallel(
        id="Qualified_Dividend_Holding_Period_Eligibility",
        desc="Provide holding-period requirements for qualified dividend treatment and determine the latest purchase date for eligibility to receive the Q1 2026 dividend.",
        parent=parent,
        critical=True,
    )

    hp = ext.holding_period or HoldingExtraction()
    q1 = ext.q1_2026 or DividendQ1Extraction()

    # Leaf 1: Minimum holding days rule (verify by URLs)
    min_hold_leaf = evaluator.add_leaf(
        id="Minimum_Holding_Days_In_121_Day_Window",
        desc="Specify the minimum number of days the investor must hold the stock during the 121-day period (more than 60 days) to qualify for qualified dividend treatment.",
        parent=node,
        critical=True,
    )
    # Build an inclusive claim around "more than 60 days" a.k.a. at least 61 days
    claim_hold = (
        "To qualify dividends as 'qualified' for U.S. tax purposes, an investor must hold the stock for more than 60 days "
        "within the 121-day period that begins 60 days before the ex-dividend date (effectively at least 61 days, excluding the ex-dividend date itself)."
    )
    await evaluator.verify(
        claim=claim_hold,
        node=min_hold_leaf,
        sources=hp.sources,
        additional_instruction="Look for IRS or other authoritative guidance. Wording variations are acceptable as long as the rule is 'more than 60 days' in a 121-day window around the ex-dividend date.",
    )

    # Leaf 2: Latest purchase date to receive Q1 2026 dividend (logical check)
    latest_leaf_result = False
    exd = _try_parse_date(q1.ex_dividend_date)
    expected_latest_str = None
    if exd:
        prev_trading = _previous_weekday(exd)
        expected_latest_str = _to_mmddyyyy(prev_trading)
        if hp.latest_purchase_date:
            # Normalize both strings to comparable dates if possible
            ans_latest_dt = _try_parse_date(hp.latest_purchase_date)
            if ans_latest_dt:
                ans_latest_str = _to_mmddyyyy(ans_latest_dt)
            else:
                ans_latest_str = (hp.latest_purchase_date or "").strip()
            latest_leaf_result = (ans_latest_str == expected_latest_str)
        else:
            latest_leaf_result = False
    else:
        latest_leaf_result = False

    latest_leaf = evaluator.add_custom_node(
        result=latest_leaf_result,
        id="Latest_Purchase_Date_For_Dividend_Eligibility",
        desc=f"Calculate the latest date the investor can purchase Chevron stock to be eligible to receive the Q1 2026 dividend (must own shares before ex-dividend date). Expected latest purchase date based on extracted ex-date: {expected_latest_str if expected_latest_str else 'unknown'}.",
        parent=node,
        critical=True,
    )
    # Even though this is a custom check (pure logic), we establish a dependency on ex-dividend verification
    # by ensuring that, if ex-dividend leaf failed/skipped, this logical check would also fail by design due to missing/invalid date.

    return node


async def verify_supporting_urls_block(
    evaluator: Evaluator,
    parent,
    ext: ChevronDividendAnalysisExtract,
):
    node = evaluator.add_parallel(
        id="Supporting_Reference_URLs",
        desc="Include supporting reference URLs from reliable sources for the key claims/data requested.",
        parent=parent,
        critical=True,
    )

    arist_urls = (ext.aristocrat.sources if ext.aristocrat else []) or []
    q1_urls = (ext.q1_2026.sources if ext.q1_2026 else []) or []
    tax_urls = (ext.tax_2026_single_75k.sources if ext.tax_2026_single_75k else []) or []
    hold_urls = (ext.holding_period.sources if ext.holding_period else []) or []

    evaluator.add_custom_node(
        result=len(_dedup_urls(arist_urls)) >= 1,
        id="Sources_For_Dividend_Aristocrat_Status",
        desc="Provide at least one reliable reference URL supporting the Dividend Aristocrat-related claims (criteria and/or Chevron’s status/history).",
        parent=node,
        critical=True,
    )

    evaluator.add_custom_node(
        result=len(_dedup_urls(q1_urls)) >= 1,
        id="Sources_For_Q1_2026_Dividend_Details",
        desc="Provide at least one reliable reference URL supporting the Q1 2026 dividend amount and dates (ex-dividend and payment date).",
        parent=node,
        critical=True,
    )

    evaluator.add_custom_node(
        result=len(_dedup_urls(tax_urls)) >= 1,
        id="Sources_For_Qualified_Dividend_Tax_Rate",
        desc="Provide at least one reliable reference URL supporting the qualified dividend tax brackets/rates used for 2026.",
        parent=node,
        critical=True,
    )

    evaluator.add_custom_node(
        result=len(_dedup_urls(hold_urls)) >= 1,
        id="Sources_For_Holding_Period_Rule",
        desc="Provide at least one reliable reference URL supporting the qualified dividend holding-period rule (>60 days in the 121-day window).",
        parent=node,
        critical=True,
    )

    return node


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
    # Initialize evaluator with a parallel root
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
        prompt=prompt_extract_main(),
        template_class=ChevronDividendAnalysisExtract,
        extraction_name="extracted_chevron_dividend_analysis",
    )

    # Build the top-level critical analysis node as in rubric
    top_node = evaluator.add_parallel(
        id="Chevron_Dividend_Investment_Analysis",
        desc="Provide the requested analysis of Chevron’s dividend characteristics, Q1 2026 dividend details, 2026 qualified-dividend tax implications, holding-period eligibility, and supporting URLs.",
        parent=root,
        critical=True,
    )

    # Verify each block
    aristocrat_node = await verify_aristocrat_block(evaluator, top_node, extracted)
    q1_node, amount_leaf, exd_leaf = await verify_q1_block(evaluator, top_node, extracted)
    tax_node = await verify_tax_block(evaluator, top_node, extracted, amount_leaf)
    holding_node = await verify_holding_block(evaluator, top_node, extracted, exd_leaf)
    sources_node = await verify_supporting_urls_block(evaluator, top_node, extracted)

    # Optional: add some custom info for debugging computations
    q1 = extracted.q1_2026 or DividendQ1Extraction()
    tax = extracted.tax_2026_single_75k or TaxExtraction()
    hp = extracted.holding_period or HoldingExtraction()

    evaluator.add_custom_info(
        info={
            "extracted_amount_per_share": q1.amount_per_share_usd,
            "extracted_ex_dividend_date": q1.ex_dividend_date,
            "extracted_payment_date": q1.payment_date,
            "extracted_approx_yield": q1.approx_yield,
            "extracted_qualified_rate": tax.qualified_dividend_tax_rate,
            "extracted_after_tax_div_per_share": tax.after_tax_dividend_per_share,
            "extracted_min_holding_days": hp.min_holding_days,
            "extracted_latest_purchase_date": hp.latest_purchase_date,
        },
        info_type="extraction_debug",
        info_name="parsed_values_summary"
    )

    return evaluator.get_summary()