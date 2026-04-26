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
TASK_ID = "staking_etf_2026"
TASK_DESCRIPTION = (
    "For a portfolio allocation strategy in 2026, identify a cryptocurrency staking ETF that meets the following "
    "requirements: (1) The ETF must be staking-enabled in compliance with U.S. regulatory guidance (SEC and IRS) issued in 2025; "
    "(2) The ETF must have launched on or after October 28, 2025; (3) The ETF must be listed on a major U.S. exchange (NYSE, NYSE Arca, or Nasdaq); "
    "(4) The ETF must have a management/expense fee ratio of 0.25% or lower; (5) The ETF must use a qualified custodian explicitly named in the ETF documentation; "
    "(6) The ETF must stake at least 70% of its cryptocurrency holdings to generate rewards; (7) The underlying cryptocurrency must use a Proof-of-Stake (PoS) consensus mechanism; "
    "(8) The expected annual staking yield must be at least 3.0%; (9) Staking rewards must be distributed to investors; "
    "(10) The ETF must be tradeable on February 17, 2026 (after Presidents Day), April 6, 2026 (after Good Friday), and July 6, 2026 (after Independence Day observance). "
    "Provide the ETF name, ticker symbol, and supporting reference URLs for each requirement."
)

LAUNCH_CUTOFF_DATE = "2025-10-28"
SEC_GUIDANCE_DATE = "2025-05-29"
IRS_SAFE_HARBOR_DATE = "2025-11-11"


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class ETFInfo(BaseModel):
    name: Optional[str] = None
    ticker: Optional[str] = None


class RegulatoryExtraction(BaseModel):
    launch_date: Optional[str] = None
    launch_source_urls: List[str] = Field(default_factory=list)
    exchange_listing: Optional[str] = None  # e.g., "NYSE", "NYSE Arca", or "Nasdaq"
    exchange_source_urls: List[str] = Field(default_factory=list)
    staking_compliance_statement: Optional[str] = None
    staking_compliance_source_urls: List[str] = Field(default_factory=list)


class FeesExtraction(BaseModel):
    expense_ratio: Optional[str] = None
    fees_source_urls: List[str] = Field(default_factory=list)


class StakingExtraction(BaseModel):
    underlying_crypto_name: Optional[str] = None
    pos_consensus_statement: Optional[str] = None
    pos_source_urls: List[str] = Field(default_factory=list)
    staking_percentage: Optional[str] = None  # use string to be robust; example: "75%" or "≥70%"
    staking_percentage_source_urls: List[str] = Field(default_factory=list)
    expected_yield: Optional[str] = None      # e.g., "3.5%" or "3-4%"
    yield_source_urls: List[str] = Field(default_factory=list)
    reward_distribution_statement: Optional[str] = None
    rewards_source_urls: List[str] = Field(default_factory=list)


class CustodianExtraction(BaseModel):
    custodian_name: Optional[str] = None
    custodian_source_urls: List[str] = Field(default_factory=list)


class TradingExtraction(BaseModel):
    trading_source_urls: List[str] = Field(default_factory=list)       # ETF listing/price/volume pages
    trading_calendar_urls: List[str] = Field(default_factory=list)     # Exchange calendars or market holiday schedules


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_etf_info() -> str:
    return """
    Extract the identified ETF's basic information from the answer.

    Required fields:
    - name: The exact ETF name as stated in the answer.
    - ticker: The ETF ticker symbol as stated in the answer.

    If any field is missing, return null for that field.
    """


def prompt_extract_regulatory() -> str:
    return f"""
    Extract regulatory and listing information for the identified ETF.

    Required fields:
    - launch_date: The ETF's official launch/listing date (prefer ISO format YYYY-MM-DD if available, else copy verbatim).
    - launch_source_urls: All URLs in the answer that directly support the launch date (prospectus, press release, exchange listing page).
    - exchange_listing: The major U.S. exchange name where the ETF is listed (one of "NYSE", "NYSE Arca", or "Nasdaq").
    - exchange_source_urls: All URLs that confirm the ETF is listed on that exchange (official listing page, exchange profile, or issuer page).
    - staking_compliance_statement: The statement from ETF documentation asserting staking-enabled status compliant with SEC guidance ({SEC_GUIDANCE_DATE}) and IRS safe harbor ({IRS_SAFE_HARBOR_DATE}).
    - staking_compliance_source_urls: All URLs cited that explicitly discuss staking compliance status.

    Rules:
    - Extract only URLs explicitly present in the answer.
    - Include both plain URLs and URLs inside markdown links.
    - Do not invent URLs.
    """


def prompt_extract_fees() -> str:
    return """
    Extract the ETF fee information.

    Required fields:
    - expense_ratio: The management/expense fee ratio (e.g., "0.25%" or "0.20%").
    - fees_source_urls: All URLs cited for fee information (prospectus, fact sheet, issuer page).

    If any field is missing, set it to null or empty list accordingly.
    """


def prompt_extract_staking() -> str:
    return """
    Extract staking mechanics and yield information.

    Required fields:
    - underlying_crypto_name: The name of the underlying cryptocurrency held or tracked by the ETF.
    - pos_consensus_statement: The statement confirming the underlying cryptocurrency uses Proof-of-Stake (PoS).
    - pos_source_urls: URLs supporting PoS consensus.
    - staking_percentage: The percentage of ETF-held cryptocurrency that is staked (e.g., "70%", "≥75%").
    - staking_percentage_source_urls: URLs supporting the staking percentage.
    - expected_yield: The expected annual staking yield stated (e.g., "3.0%", "3-4%").
    - yield_source_urls: URLs supporting the yield expectation.
    - reward_distribution_statement: Statement confirming staking rewards are distributed to investors (not retained solely by fund/service providers).
    - rewards_source_urls: URLs supporting the reward distribution policy.
    """


def prompt_extract_custodian() -> str:
    return """
    Extract custodian information.

    Required fields:
    - custodian_name: The explicitly named qualified custodian in ETF documentation.
    - custodian_source_urls: URLs supporting the custodian's role and explicit naming.

    If any field is missing, set it to null or empty list accordingly.
    """


def prompt_extract_trading() -> str:
    return """
    Extract trading accessibility references.

    Required fields:
    - trading_source_urls: URLs confirming the ETF is actively listed and tradeable (exchange page, issuer page, market data page).
    - trading_calendar_urls: URLs for U.S. market/exchange trading calendars or holiday schedules.

    Include all URLs provided in the answer that support trading status and calendar information.
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def _valid_urls(urls: List[str]) -> List[str]:
    """Return only plausible http(s) URLs."""
    return [u for u in (urls or []) if isinstance(u, str) and u.strip().lower().startswith(("http://", "https://"))]


def _combine_sources(*lists: List[str]) -> List[str]:
    """Combine URL lists and de-duplicate while keeping order."""
    seen = set()
    combined: List[str] = []
    for lst in lists:
        for u in _valid_urls(lst):
            if u not in seen:
                combined.append(u)
                seen.add(u)
    return combined


# --------------------------------------------------------------------------- #
# Verification builders                                                       #
# --------------------------------------------------------------------------- #
async def build_etf_identification(evaluator: Evaluator, parent) -> None:
    """
    Critical gating: Ensure ETF name and ticker are provided.
    """
    # Extract ETF info
    etf_info = await evaluator.extract(
        prompt=prompt_extract_etf_info(),
        template_class=ETFInfo,
        extraction_name="etf_info"
    )

    # Existence checks (critical)
    evaluator.add_custom_node(
        result=bool(etf_info.name and etf_info.name.strip()),
        id="ETF_Name_Provided",
        desc="ETF name is provided",
        parent=parent,
        critical=True
    )
    evaluator.add_custom_node(
        result=bool(etf_info.ticker and etf_info.ticker.strip()),
        id="ETF_Ticker_Provided",
        desc="ETF ticker symbol is provided",
        parent=parent,
        critical=True
    )

    # Save for downstream use
    evaluator.add_custom_info(
        {"etf_name": etf_info.name, "etf_ticker": etf_info.ticker},
        info_type="extraction_snapshot",
        info_name="identified_etf"
    )


async def build_regulatory_checks(evaluator: Evaluator, parent) -> None:
    reg = await evaluator.extract(
        prompt=prompt_extract_regulatory(),
        template_class=RegulatoryExtraction,
        extraction_name="regulatory_listing"
    )

    # Post-SEC Guidance Launch
    node_launch = evaluator.add_leaf(
        id="Post_SEC_Guidance_Launch",
        desc="ETF launched on or after October 28, 2025",
        parent=parent,
        critical=True
    )
    claim_launch = (
        f"The ETF launched on {reg.launch_date}, and this launch date is on or after {LAUNCH_CUTOFF_DATE}."
    )
    await evaluator.verify(
        claim=claim_launch,
        node=node_launch,
        sources=_valid_urls(reg.launch_source_urls),
        additional_instruction="Verify the stated launch/listing date on the provided URL(s). Confirm that the date is on or after 2025-10-28."
    )

    # US exchange listing
    node_exchange = evaluator.add_leaf(
        id="US_Exchange_Listed",
        desc="ETF is listed on a major U.S. exchange (NYSE, NYSE Arca, or Nasdaq)",
        parent=parent,
        critical=True
    )
    claim_exchange = (
        f"The ETF is listed on {reg.exchange_listing}, which is one of NYSE, NYSE Arca, or Nasdaq."
    )
    await evaluator.verify(
        claim=claim_exchange,
        node=node_exchange,
        sources=_valid_urls(reg.exchange_source_urls),
        additional_instruction="Check the exchange listing page or issuer page to confirm the ETF is listed on NYSE, NYSE Arca, or Nasdaq."
    )

    # Staking-enabled compliance with SEC/IRS guidance
    node_staking_enabled = evaluator.add_leaf(
        id="Staking_Enabled_Status",
        desc=f"ETF staking-enabled in compliance with SEC guidance ({SEC_GUIDANCE_DATE}) and IRS safe harbor ({IRS_SAFE_HARBOR_DATE})",
        parent=parent,
        critical=True
    )
    compliance_stmt = reg.staking_compliance_statement or ""
    claim_compliance = (
        f"The ETF explicitly states it is staking-enabled in compliance with SEC guidance issued {SEC_GUIDANCE_DATE} "
        f"and IRS safe harbor guidance issued {IRS_SAFE_HARBOR_DATE}. Statement: {compliance_stmt}"
    )
    await evaluator.verify(
        claim=claim_compliance,
        node=node_staking_enabled,
        sources=_valid_urls(reg.staking_compliance_source_urls),
        additional_instruction="Confirm the ETF documentation states staking-enabled compliance with the referenced SEC and IRS guidance dates."
    )

    # Reference URL presence (critical to satisfy parent critical requirement and source-grounding)
    reg_urls_present = bool(_valid_urls(reg.launch_source_urls) or _valid_urls(reg.staking_compliance_source_urls))
    evaluator.add_custom_node(
        result=reg_urls_present,
        id="Reference_URL_Regulatory",
        desc="At least one valid URL confirms regulatory compliance and launch date",
        parent=parent,
        critical=True
    )

    # Snapshot for debugging
    evaluator.add_custom_info(
        {"regulatory": reg.dict()},
        info_type="extraction_snapshot",
        info_name="regulatory_extraction"
    )


async def build_fee_checks(evaluator: Evaluator, parent) -> None:
    fees = await evaluator.extract(
        prompt=prompt_extract_fees(),
        template_class=FeesExtraction,
        extraction_name="fees"
    )

    # Management fee threshold
    node_fee = evaluator.add_leaf(
        id="Management_Fee_Threshold",
        desc="ETF management/expense fee ratio is 0.25% or lower",
        parent=parent,
        critical=True
    )
    claim_fee = (
        f"The ETF's management/expense fee ratio is {fees.expense_ratio}, which is less than or equal to 0.25%."
    )
    await evaluator.verify(
        claim=claim_fee,
        node=node_fee,
        sources=_valid_urls(fees.fees_source_urls),
        additional_instruction="Check the fee ratio on the provided source(s). Allow minor rounding; confirm it does not exceed 0.25%."
    )

    # Reference URL presence (critical)
    fees_urls_present = bool(_valid_urls(fees.fees_source_urls))
    evaluator.add_custom_node(
        result=fees_urls_present,
        id="Reference_URL_Fees",
        desc="Valid URL reference confirming fee structure is provided",
        parent=parent,
        critical=True
    )

    evaluator.add_custom_info(
        {"fees": fees.dict()},
        info_type="extraction_snapshot",
        info_name="fees_extraction"
    )


async def build_staking_checks(evaluator: Evaluator, parent) -> None:
    staking = await evaluator.extract(
        prompt=prompt_extract_staking(),
        template_class=StakingExtraction,
        extraction_name="staking_mechanics"
    )

    # Proof-of-Stake mechanism
    node_pos = evaluator.add_leaf(
        id="Proof_of_Stake_Mechanism",
        desc="Underlying cryptocurrency uses Proof-of-Stake (PoS) consensus mechanism",
        parent=parent,
        critical=True
    )
    crypto_name = staking.underlying_crypto_name or "the underlying cryptocurrency"
    pos_stmt = staking.pos_consensus_statement or ""
    claim_pos = f"{crypto_name} uses a Proof-of-Stake (PoS) consensus mechanism. Statement: {pos_stmt}"
    await evaluator.verify(
        claim=claim_pos,
        node=node_pos,
        sources=_valid_urls(staking.pos_source_urls),
        additional_instruction="Confirm the consensus mechanism from the provided sources; allow reasonable synonyms for PoS."
    )

    # Minimum staking percentage ≥ 70%
    node_stake_pct = evaluator.add_leaf(
        id="Minimum_Staking_Percentage",
        desc="ETF stakes at least 70% of its cryptocurrency holdings",
        parent=parent,
        critical=True
    )
    spct = staking.staking_percentage or ""
    claim_spct = f"The ETF stakes {spct} of its cryptocurrency holdings, which is at least 70%."
    await evaluator.verify(
        claim=claim_spct,
        node=node_stake_pct,
        sources=_valid_urls(staking.staking_percentage_source_urls),
        additional_instruction="Confirm that the stated staking percentage meets or exceeds 70%."
    )

    # Minimum expected yield ≥ 3.0%
    node_yield = evaluator.add_leaf(
        id="Minimum_Expected_Yield",
        desc="Expected annual staking yield is at least 3.0%",
        parent=parent,
        critical=True
    )
    yexp = staking.expected_yield or ""
    claim_yexp = f"The expected annual staking yield is {yexp}, which is at least 3.0%."
    await evaluator.verify(
        claim=claim_yexp,
        node=node_yield,
        sources=_valid_urls(staking.yield_source_urls),
        additional_instruction="Confirm the expected yield from provided sources; allow reasonable ranges that include ≥3.0%."
    )

    # Investor reward distribution
    node_rewards = evaluator.add_leaf(
        id="Investor_Reward_Distribution",
        desc="Staking rewards are distributed to investors",
        parent=parent,
        critical=True
    )
    rstmt = staking.reward_distribution_statement or ""
    claim_rewards = (
        f"Staking rewards generated are distributed to investors. Policy statement: {rstmt}"
    )
    await evaluator.verify(
        claim=claim_rewards,
        node=node_rewards,
        sources=_valid_urls(staking.rewards_source_urls),
        additional_instruction="Confirm that staking rewards are paid out or accrued to ETF investors (not solely retained by the fund or service providers)."
    )

    # Reference URL presence (critical)
    staking_urls_present = bool(
        _valid_urls(staking.pos_source_urls)
        or _valid_urls(staking.staking_percentage_source_urls)
        or _valid_urls(staking.yield_source_urls)
        or _valid_urls(staking.rewards_source_urls)
    )
    evaluator.add_custom_node(
        result=staking_urls_present,
        id="Reference_URL_Staking",
        desc="Valid URL references confirm staking mechanics and yield information",
        parent=parent,
        critical=True
    )

    evaluator.add_custom_info(
        {"staking": staking.dict()},
        info_type="extraction_snapshot",
        info_name="staking_extraction"
    )


async def build_custodian_checks(evaluator: Evaluator, parent) -> None:
    cust = await evaluator.extract(
        prompt=prompt_extract_custodian(),
        template_class=CustodianExtraction,
        extraction_name="custodian"
    )

    # Qualified custodian explicitly named
    node_custodian = evaluator.add_leaf(
        id="Qualified_Custodian",
        desc="ETF uses a qualified custodian explicitly named in ETF documentation",
        parent=parent,
        critical=True
    )
    cname = cust.custodian_name or ""
    claim_custodian = f"The ETF uses a qualified custodian named '{cname}', explicitly named in ETF documentation."
    await evaluator.verify(
        claim=claim_custodian,
        node=node_custodian,
        sources=_valid_urls(cust.custodian_source_urls),
        additional_instruction="Verify custodian naming and qualified status from the ETF prospectus or issuer page."
    )

    # Reference URL presence (critical)
    cust_urls_present = bool(_valid_urls(cust.custodian_source_urls))
    evaluator.add_custom_node(
        result=cust_urls_present,
        id="Reference_URL_Custodian",
        desc="Valid URL reference confirming custodian information is provided",
        parent=parent,
        critical=True
    )

    evaluator.add_custom_info(
        {"custodian": cust.dict()},
        info_type="extraction_snapshot",
        info_name="custodian_extraction"
    )


async def build_trading_checks(evaluator: Evaluator, parent) -> None:
    trading = await evaluator.extract(
        prompt=prompt_extract_trading(),
        template_class=TradingExtraction,
        extraction_name="trading_accessibility"
    )

    # Helper: combined trading sources
    combined_trading_sources = _combine_sources(trading.trading_source_urls, trading.trading_calendar_urls)

    # February 17, 2026
    node_feb = evaluator.add_leaf(
        id="Available_February_17_2026",
        desc="ETF is tradeable on Tuesday, February 17, 2026",
        parent=parent,
        critical=True
    )
    claim_feb = (
        "The ETF is tradeable on Tuesday, February 17, 2026 (the day after Presidents Day closure on Feb 16, 2026). "
        "The provided sources confirm the ETF's active listing/trading status and the U.S. market calendar."
    )
    await evaluator.verify(
        claim=claim_feb,
        node=node_feb,
        sources=combined_trading_sources,
        additional_instruction="Use exchange/market data pages to confirm listing and tradability; use trading calendar URL(s) to confirm market open date."
    )

    # April 6, 2026
    node_apr = evaluator.add_leaf(
        id="Available_April_6_2026",
        desc="ETF is tradeable on Monday, April 6, 2026",
        parent=parent,
        critical=True
    )
    claim_apr = (
        "The ETF is tradeable on Monday, April 6, 2026 (first trading day after Good Friday closure on April 3, 2026). "
        "The provided sources confirm the ETF's active listing/trading status and the U.S. market calendar."
    )
    await evaluator.verify(
        claim=claim_apr,
        node=node_apr,
        sources=combined_trading_sources,
        additional_instruction="Use exchange/market data pages to confirm listing and tradability; use trading calendar URL(s) to confirm market open date."
    )

    # July 6, 2026
    node_jul = evaluator.add_leaf(
        id="Available_July_6_2026",
        desc="ETF is tradeable on Monday, July 6, 2026",
        parent=parent,
        critical=True
    )
    claim_jul = (
        "The ETF is tradeable on Monday, July 6, 2026 (first trading day after Independence Day observance closure on July 3, 2026). "
        "The provided sources confirm the ETF's active listing/trading status and the U.S. market calendar."
    )
    await evaluator.verify(
        claim=claim_jul,
        node=node_jul,
        sources=combined_trading_sources,
        additional_instruction="Use exchange/market data pages to confirm listing and tradability; use trading calendar URL(s) to confirm market open date."
    )

    # Reference URL presence (critical)
    trading_urls_present = bool(_valid_urls(trading.trading_source_urls))
    evaluator.add_custom_node(
        result=trading_urls_present,
        id="Reference_URL_Trading",
        desc="Valid URL reference confirming ETF is actively traded and listing status",
        parent=parent,
        critical=True
    )

    evaluator.add_custom_info(
        {"trading": trading.dict()},
        info_type="extraction_snapshot",
        info_name="trading_extraction"
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
    model: str = "o4-mini"
) -> Dict:
    """
    Evaluate the answer for the 2026 staking ETF portfolio criteria.

    Returns:
        A structured evaluation summary dict containing the verification tree and final score.
    """
    # Initialize evaluator (root is non-critical by framework design)
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

    # Create a critical wrapper so failing any essential group fails overall
    all_criteria = evaluator.add_parallel(
        id="All_Criteria",
        desc="All essential criteria must be satisfied by the identified ETF",
        parent=root,
        critical=True
    )

    # 0) Critical gating: ETF identification (sequential)
    etf_ident = evaluator.add_sequential(
        id="ETF_Identification",
        desc="ETF name and ticker must be provided",
        parent=all_criteria,
        critical=True
    )
    await build_etf_identification(evaluator, etf_ident)

    # 1) Regulatory compliance and launch (parallel, critical)
    regulatory_node = evaluator.add_parallel(
        id="Regulatory_Compliance_and_Launch",
        desc="Verify regulatory requirements and launch timeline",
        parent=all_criteria,
        critical=True
    )
    await build_regulatory_checks(evaluator, regulatory_node)

    # 2) Fee structure (parallel, critical)
    fees_node = evaluator.add_parallel(
        id="Fee_Structure",
        desc="Verify ETF fee requirements (<= 0.25%)",
        parent=all_criteria,
        critical=True
    )
    await build_fee_checks(evaluator, fees_node)

    # 3) Staking mechanics and yield (parallel, critical)
    staking_node = evaluator.add_parallel(
        id="Staking_Mechanics_and_Yield",
        desc="Verify staking structure, percentages, PoS, yield, and reward distribution",
        parent=all_criteria,
        critical=True
    )
    await build_staking_checks(evaluator, staking_node)

    # 4) Custodian and structure (parallel, critical)
    custodian_node = evaluator.add_parallel(
        id="Custodian_and_Structure",
        desc="Verify qualified custodian explicitly named in ETF documentation",
        parent=all_criteria,
        critical=True
    )
    await build_custodian_checks(evaluator, custodian_node)

    # 5) Trading accessibility 2026 (parallel, critical)
    trading_node = evaluator.add_parallel(
        id="Trading_Accessibility_2026",
        desc="Verify ETF tradability on specified 2026 dates",
        parent=all_criteria,
        critical=True
    )
    await build_trading_checks(evaluator, trading_node)

    # Final structured result
    return evaluator.get_summary()