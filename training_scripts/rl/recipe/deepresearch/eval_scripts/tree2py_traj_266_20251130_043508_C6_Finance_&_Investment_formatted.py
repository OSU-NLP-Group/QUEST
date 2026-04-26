import asyncio
import logging
from typing import Any, List, Optional, Dict

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "finance_markets_2024"
TASK_DESCRIPTION = (
    "Research the following five aspects of US financial markets and monetary policy in late 2024: "
    "(1) What were the NYSE and NASDAQ trading hours on Black Friday 2024, specifically what time did the equity markets close early? "
    "(2) What is the Federal Reserve's federal funds rate target range as of the December 2024 FOMC meeting, and by how much was the rate cut? "
    "(3) In which months do the S&P 500's quarterly rebalances occur, and on which day of the month (e.g., first Friday, third Friday) do they take effect? "
    "(4) What is the current position limit for options on BlackRock's iShares Bitcoin Trust ETF (IBIT) on Nasdaq ISE, and what new limit has Nasdaq ISE proposed to the SEC? "
    "(5) What was the average 30-year fixed mortgage rate according to Freddie Mac as of late November 2024? "
    "For each answer, provide the specific numerical values, dates, and include the reference URL from which the information was obtained."
)

# Ground truth reference values to verify against sources
GROUND_TRUTH = {
    "black_friday_date": "November 29, 2024",
    "equity_early_close_et": "1:00 PM ET",
    "eligible_options_early_close_et": "1:15 PM ET",
    "applies_to_both_exchanges": "applies to both NYSE and Nasdaq",
    "fomc_decision_date": "December 18, 2024",
    "fed_target_range": "4.25% to 4.5%",
    "fed_rate_cut_amount": "25 basis points",
    "sp500_rebalance_months": ["March", "June", "September", "December"],
    "sp500_day_of_month_rule": "third Friday",
    "ibit_current_limit": "250,000 contracts",
    "ibit_proposed_limit": "1,000,000 contracts",
    "ibit_proposal_timing": "late November 2024",
    "mortgage_rate_value": "6.23%",
    "mortgage_rate_as_of_date": "November 26, 2024",
    "mortgage_rate_source_label": "Freddie Mac PMMS",
}


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class TradingHoursInfo(BaseModel):
    black_friday_date: Optional[str] = None
    equity_early_close_et: Optional[str] = None
    eligible_options_early_close_et: Optional[str] = None
    applies_to_both_exchanges: Optional[str] = None  # e.g., "applies to both NYSE and Nasdaq"
    reference_urls: List[str] = Field(default_factory=list)


class FedFundsInfo(BaseModel):
    fomc_decision_date: Optional[str] = None
    target_range: Optional[str] = None
    rate_cut_amount: Optional[str] = None
    reference_urls: List[str] = Field(default_factory=list)


class SP500RebalanceInfo(BaseModel):
    rebalance_months: List[str] = Field(default_factory=list)
    day_of_month_rule: Optional[str] = None
    reference_urls: List[str] = Field(default_factory=list)


class IBITOptionsLimitsInfo(BaseModel):
    current_limit: Optional[str] = None
    proposed_limit: Optional[str] = None
    proposal_timing: Optional[str] = None
    reference_urls: List[str] = Field(default_factory=list)


class MortgageRateInfo(BaseModel):
    rate_value: Optional[str] = None
    as_of_date: Optional[str] = None
    source_label: Optional[str] = None  # e.g., "Freddie Mac PMMS"
    reference_urls: List[str] = Field(default_factory=list)


class CombinedExtraction(BaseModel):
    trading_hours: Optional[TradingHoursInfo] = None
    fed_funds: Optional[FedFundsInfo] = None
    sp500_rebalance: Optional[SP500RebalanceInfo] = None
    ibit_limits: Optional[IBITOptionsLimitsInfo] = None
    mortgage_rate: Optional[MortgageRateInfo] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_combined() -> str:
    return """
    Extract the five requested aspects exactly as they appear in the answer text. For each aspect, include any reference URLs cited in the answer (only actual URLs).
    
    1) Trading hours on Black Friday 2024:
       - black_friday_date: the date stated for Black Friday 2024 (e.g., "November 29, 2024")
       - equity_early_close_et: the early close time for equity markets in ET (e.g., "1:00 PM ET")
       - eligible_options_early_close_et: the early close time for eligible options in ET (e.g., "1:15 PM ET")
       - applies_to_both_exchanges: a short textual confirmation about scope, if present (e.g., "applies to both NYSE and Nasdaq")
       - reference_urls: list all URLs cited for trading hours/holiday schedule
        
    2) Federal funds rate at December 2024 FOMC:
       - fomc_decision_date: the announcement date stated (e.g., "December 18, 2024")
       - target_range: the post‑decision target range stated (e.g., "4.25% to 4.5%")
       - rate_cut_amount: the size of the cut (e.g., "25 bps" or "0.25 percentage points")
       - reference_urls: list URLs cited for FOMC decision
        
    3) S&P 500 quarterly rebalance schedule:
       - rebalance_months: list of months stated (e.g., ["March","June","September","December"])
       - day_of_month_rule: when changes take effect (e.g., "third Friday")
       - reference_urls: list URLs supporting schedule
        
    4) IBIT options position limits on Nasdaq ISE:
       - current_limit: the current limit stated (e.g., "250,000 contracts")
       - proposed_limit: the proposed limit stated (e.g., "1,000,000 contracts")
       - proposal_timing: timing stated for the filing (e.g., "late November 2024")
       - reference_urls: list URLs cited (e.g., Nasdaq ISE/SEC rule filing)
        
    5) Mortgage rate (Freddie Mac) late November 2024:
       - rate_value: the average 30‑year fixed rate stated (e.g., "6.23%")
       - as_of_date: the as‑of date stated (e.g., "November 26, 2024")
       - source_label: any source statement provided (e.g., "Freddie Mac PMMS")
       - reference_urls: list URLs cited for mortgage rate
        
    Rules:
    - Extract only what is explicitly present in the answer. If missing, set null or empty list.
    - For URLs, include only valid URLs that appear in the answer (plain or markdown links).
    """


# --------------------------------------------------------------------------- #
# Verification helper builders                                                #
# --------------------------------------------------------------------------- #
async def verify_trading_hours(
    evaluator: Evaluator,
    parent_node,
    info: Optional[TradingHoursInfo],
) -> None:
    node = evaluator.add_parallel(
        id="trading_hours_black_friday",
        desc="Provide NYSE and NASDAQ trading hours on Black Friday 2024, including early equity-market close time, relevant date/timing, and citation",
        parent=parent_node,
        critical=True,
    )

    urls = info.reference_urls if info else []

    # Reference URL existence (critical)
    ref_node = evaluator.add_custom_node(
        result=bool(urls),
        id="reference_url_trading_hours",
        desc="Provides a reference URL for the trading-hours/holiday-schedule information",
        parent=node,
        critical=True,
    )

    # Black Friday date (critical)
    date_leaf = evaluator.add_leaf(
        id="black_friday_date_provided",
        desc="States the Black Friday 2024 date (should be Nov 29, 2024)",
        parent=node,
        critical=True,
    )
    date_claim = f"Black Friday 2024 is {GROUND_TRUTH['black_friday_date']}."
    await evaluator.verify(
        claim=date_claim,
        node=date_leaf,
        sources=urls,
        additional_instruction="Validate using official exchange holiday calendars or trusted schedules. Minor formatting differences (e.g., Nov vs. November) are acceptable.",
        extra_prerequisites=[ref_node],
    )

    # Equity markets early close time (critical)
    equity_leaf = evaluator.add_leaf(
        id="equity_early_close_time_provided",
        desc="Provides the early close time for the equity markets (should be 1:00 PM ET)",
        parent=node,
        critical=True,
    )
    equity_claim = (
        f"On Black Friday 2024, NYSE and Nasdaq U.S. equity markets closed early at {GROUND_TRUTH['equity_early_close_et']}."
    )
    await evaluator.verify(
        claim=equity_claim,
        node=equity_leaf,
        sources=urls,
        additional_instruction="Confirm early close in ET for equities; allow minor wording variations like '1 pm ET'.",
        extra_prerequisites=[ref_node],
    )

    # Eligible options early close time (critical)
    options_leaf = evaluator.add_leaf(
        id="eligible_options_close_time_provided",
        desc="Provides the early close time for eligible options (should be 1:15 PM ET)",
        parent=node,
        critical=True,
    )
    options_claim = (
        f"On Black Friday 2024, eligible options closed early at {GROUND_TRUTH['eligible_options_early_close_et']}."
    )
    await evaluator.verify(
        claim=options_claim,
        node=options_leaf,
        sources=urls,
        additional_instruction="Confirm eligible options early close time on the same day; allow minor variations in punctuation and casing.",
        extra_prerequisites=[ref_node],
    )

    # Applies to both exchanges (critical)
    applies_leaf = evaluator.add_leaf(
        id="applies_to_both_exchanges",
        desc="Confirms the early-close schedule applies to both NYSE and NASDAQ",
        parent=node,
        critical=True,
    )
    applies_claim = (
        "The early-close schedule (equities at 1:00 PM ET and eligible options at 1:15 PM ET) applied to both NYSE and Nasdaq U.S. markets on Black Friday 2024."
    )
    await evaluator.verify(
        claim=applies_claim,
        node=applies_leaf,
        sources=urls,
        additional_instruction="Verify that both NYSE and Nasdaq U.S. markets followed the early-close schedule.",
        extra_prerequisites=[ref_node],
    )


async def verify_fed_funds(
    evaluator: Evaluator,
    parent_node,
    info: Optional[FedFundsInfo],
) -> None:
    node = evaluator.add_parallel(
        id="fed_funds_rate_december_2024",
        desc="Provide the federal funds rate target range as of the December 2024 FOMC decision, the cut amount, the decision timing/date, and citation",
        parent=parent_node,
        critical=True,
    )

    urls = info.reference_urls if info else []

    # Reference URL existence (critical)
    ref_node = evaluator.add_custom_node(
        result=bool(urls),
        id="reference_url_fed",
        desc="Provides a reference URL for the Federal Reserve/FOMC rate decision",
        parent=node,
        critical=True,
    )

    # FOMC decision date (critical)
    date_leaf = evaluator.add_leaf(
        id="fomc_decision_date_provided",
        desc="Provides the December 2024 FOMC decision announcement date (should be Dec 18, 2024)",
        parent=node,
        critical=True,
    )
    date_claim = f"The December 2024 FOMC decision was announced on {GROUND_TRUTH['fomc_decision_date']}."
    await evaluator.verify(
        claim=date_claim,
        node=date_leaf,
        sources=urls,
        additional_instruction="Prefer the official Federal Reserve press release or FOMC statement.",
        extra_prerequisites=[ref_node],
    )

    # Target range (critical)
    range_leaf = evaluator.add_leaf(
        id="target_range_provided",
        desc="Provides the federal funds target range as of the December 2024 decision (should be 4.25%–4.5%)",
        parent=node,
        critical=True,
    )
    range_claim = f"Following the December 2024 meeting, the target range for the federal funds rate was {GROUND_TRUTH['fed_target_range']}."
    await evaluator.verify(
        claim=range_claim,
        node=range_leaf,
        sources=urls,
        additional_instruction="Confirm the post-decision target range; allow typographic variations like en dash vs. 'to'.",
        extra_prerequisites=[ref_node],
    )

    # Rate cut amount (critical)
    cut_leaf = evaluator.add_leaf(
        id="rate_cut_amount_provided",
        desc="Provides the size of the rate cut at that meeting (should be 25 bps / 0.25 percentage points)",
        parent=node,
        critical=True,
    )
    cut_claim = (
        f"At the December 2024 meeting, the FOMC cut rates by {GROUND_TRUTH['fed_rate_cut_amount']} (0.25 percentage points)."
    )
    await evaluator.verify(
        claim=cut_claim,
        node=cut_leaf,
        sources=urls,
        additional_instruction="Confirm the cut magnitude; allow wording variations (bps vs. basis points).",
        extra_prerequisites=[ref_node],
    )


async def verify_sp500_rebalance(
    evaluator: Evaluator,
    parent_node,
    info: Optional[SP500RebalanceInfo],
) -> None:
    node = evaluator.add_parallel(
        id="sp500_rebalance_schedule",
        desc="Provide the months of S&P 500 quarterly rebalances and the day-of-month pattern when they take effect, with citation",
        parent=parent_node,
        critical=True,
    )

    urls = info.reference_urls if info else []

    # Reference URL existence (critical)
    ref_node = evaluator.add_custom_node(
        result=bool(urls),
        id="reference_url_sp500",
        desc="Provides a reference URL supporting the S&P 500 rebalance schedule claim",
        parent=node,
        critical=True,
    )

    # Rebalance months (critical)
    months_leaf = evaluator.add_leaf(
        id="rebalance_months_provided",
        desc="Provides the rebalance months (should be March, June, September, December)",
        parent=node,
        critical=True,
    )
    months_str = ", ".join(GROUND_TRUTH["sp500_rebalance_months"])
    months_claim = f"The S&P 500 quarterly rebalances occur in {months_str}."
    await evaluator.verify(
        claim=months_claim,
        node=months_leaf,
        sources=urls,
        additional_instruction="Use S&P Dow Jones Indices methodology or official documentation; allow minor formatting variations.",
        extra_prerequisites=[ref_node],
    )

    # Day-of-month rule (critical)
    day_leaf = evaluator.add_leaf(
        id="day_of_month_provided",
        desc="Provides the day-of-month rule for when changes take effect (should be the third Friday of the rebalance month)",
        parent=node,
        critical=True,
    )
    day_claim = f"Index changes take effect on the {GROUND_TRUTH['sp500_day_of_month_rule']} of the rebalance month."
    await evaluator.verify(
        claim=day_claim,
        node=day_leaf,
        sources=urls,
        additional_instruction="Confirm effective timing (third Friday) for quarterly rebalances; allow wording variants.",
        extra_prerequisites=[ref_node],
    )


async def verify_ibit_limits(
    evaluator: Evaluator,
    parent_node,
    info: Optional[IBITOptionsLimitsInfo],
) -> None:
    node = evaluator.add_parallel(
        id="ibit_options_position_limits",
        desc="Provide the current and proposed IBIT options position limits on Nasdaq ISE, include proposal timing, and citation",
        parent=parent_node,
        critical=True,
    )

    urls = info.reference_urls if info else []

    # Reference URL existence (critical)
    ref_node = evaluator.add_custom_node(
        result=bool(urls),
        id="reference_url_ibit",
        desc="Provides a reference URL supporting the IBIT options position-limit current and proposed values",
        parent=node,
        critical=True,
    )

    # Current limit (critical)
    current_leaf = evaluator.add_leaf(
        id="current_limit_provided",
        desc="Provides the current IBIT options position limit (should be 250,000 contracts)",
        parent=node,
        critical=True,
    )
    current_claim = f"The current position limit for options on IBIT on Nasdaq ISE is {GROUND_TRUTH['ibit_current_limit']}."
    await evaluator.verify(
        claim=current_claim,
        node=current_leaf,
        sources=urls,
        additional_instruction="Validate via Nasdaq ISE rulebook/filings or SEC notices; allow minor numeric formatting (commas).",
        extra_prerequisites=[ref_node],
    )

    # Proposed limit (critical)
    proposed_leaf = evaluator.add_leaf(
        id="proposed_limit_provided",
        desc="Provides the proposed new IBIT options position limit (should be 1,000,000 contracts)",
        parent=node,
        critical=True,
    )
    proposed_claim = f"Nasdaq ISE proposed increasing the IBIT options position limit to {GROUND_TRUTH['ibit_proposed_limit']}."
    await evaluator.verify(
        claim=proposed_claim,
        node=proposed_leaf,
        sources=urls,
        additional_instruction="Confirm proposal details in Nasdaq ISE or SEC rule filing documents.",
        extra_prerequisites=[ref_node],
    )

    # Proposal timing (critical)
    timing_leaf = evaluator.add_leaf(
        id="proposal_timing_provided",
        desc="States when the proposal was filed (should be late November 2024)",
        parent=node,
        critical=True,
    )
    timing_claim = f"The proposal was filed in {GROUND_TRUTH['ibit_proposal_timing']}."
    await evaluator.verify(
        claim=timing_claim,
        node=timing_leaf,
        sources=urls,
        additional_instruction="Confirm filing date or timing; allow phrasing like 'filed November 26, 2024'.",
        extra_prerequisites=[ref_node],
    )


async def verify_mortgage_rate(
    evaluator: Evaluator,
    parent_node,
    info: Optional[MortgageRateInfo],
) -> None:
    node = evaluator.add_parallel(
        id="mortgage_rate_november_2024",
        desc="Provide the average 30-year fixed mortgage rate per Freddie Mac as of late Nov 2024, including the as-of date and citation",
        parent=parent_node,
        critical=True,
    )

    urls = info.reference_urls if info else []

    # Reference URL existence (critical)
    ref_node = evaluator.add_custom_node(
        result=bool(urls),
        id="reference_url_mortgage",
        desc="Provides a reference URL for the Freddie Mac mortgage-rate data",
        parent=node,
        critical=True,
    )

    # Rate value (critical)
    rate_leaf = evaluator.add_leaf(
        id="rate_value_provided",
        desc="Provides the average 30-year fixed mortgage rate value (should be 6.23%)",
        parent=node,
        critical=True,
    )
    rate_claim = (
        f"As of {GROUND_TRUTH['mortgage_rate_as_of_date']}, the average 30-year fixed mortgage rate was {GROUND_TRUTH['mortgage_rate_value']}."
    )
    await evaluator.verify(
        claim=rate_claim,
        node=rate_leaf,
        sources=urls,
        additional_instruction="Prefer Freddie Mac PMMS page; allow minor formatting variants.",
        extra_prerequisites=[ref_node],
    )

    # As-of date (critical)
    date_leaf = evaluator.add_leaf(
        id="as_of_date_provided",
        desc="Provides the as-of date for that rate (should be Nov 26, 2024)",
        parent=node,
        critical=True,
    )
    date_claim = f"The as-of date for the reported mortgage rate is {GROUND_TRUTH['mortgage_rate_as_of_date']}."
    await evaluator.verify(
        claim=date_claim,
        node=date_leaf,
        sources=urls,
        additional_instruction="Confirm exact as-of date; formatting variations (Nov vs. November) acceptable.",
        extra_prerequisites=[ref_node],
    )

    # Source confirmation (critical)
    source_leaf = evaluator.add_leaf(
        id="freddie_mac_source_confirmed",
        desc="Confirms the rate is sourced from Freddie Mac (PMMS)",
        parent=node,
        critical=True,
    )
    source_claim = "The rate is sourced from Freddie Mac's Primary Mortgage Market Survey (PMMS)."
    await evaluator.verify(
        claim=source_claim,
        node=source_leaf,
        sources=urls,
        additional_instruction="Verify that the page attributes the rate to Freddie Mac PMMS.",
        extra_prerequisites=[ref_node],
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
    Evaluate the answer for the five specified aspects of US financial markets and monetary policy in late 2024.
    """
    # Initialize evaluator with a critical root (all aspects must be correct)
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
    # Make root critical as per rubric: must satisfy all aspects
    root.critical = True

    # Extract combined information from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_combined(),
        template_class=CombinedExtraction,
        extraction_name="combined_extraction",
    )

    # Record ground truth for transparency
    evaluator.add_ground_truth(
        {
            "expected": GROUND_TRUTH,
            "notes": "All five aspects must match these values; verification uses cited URLs from the answer.",
        },
        gt_type="ground_truth",
    )

    # Build and verify each aspect
    await verify_trading_hours(evaluator, root, extracted.trading_hours)
    await verify_fed_funds(evaluator, root, extracted.fed_funds)
    await verify_sp500_rebalance(evaluator, root, extracted.sp500_rebalance)
    await verify_ibit_limits(evaluator, root, extracted.ibit_limits)
    await verify_mortgage_rate(evaluator, root, extracted.mortgage_rate)

    # Return the evaluation summary
    return evaluator.get_summary()