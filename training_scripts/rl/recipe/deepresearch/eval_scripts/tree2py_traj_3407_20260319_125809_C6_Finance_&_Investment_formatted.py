import asyncio
import logging
import re
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "sol_spot_etf_staking_portfolio"
TASK_DESCRIPTION = (
    "I am building a diversified cryptocurrency investment portfolio and want to focus on Solana-based exchange-traded funds (ETFs) that offer both cost efficiency and yield enhancement through staking. "
    "Identify four (4) different spot Solana ETFs that are currently trading on U.S. exchanges and meet all of the following criteria: "
    "(1) Each ETF must have an expense ratio of 0.40% or lower, "
    "(2) Each ETF must offer staking features (i.e., the fund stakes SOL holdings and passes staking rewards to investors), "
    "(3) Each ETF must have at least $100 million in assets under management (AUM), "
    "(4) Each ETF must be a spot ETF (not a futures-based or leveraged ETF). "
    "For each of the four ETFs you identify, provide: the ETF's full name and ticker symbol, the expense ratio (as a percentage), a brief description of its staking approach (e.g., what percentage of holdings are staked, how rewards are distributed), the current or most recent reported AUM, and a direct URL to the ETF's official product page or prospectus on the issuer's website."
)


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class ETFItem(BaseModel):
    name: Optional[str] = None
    ticker: Optional[str] = None
    expense_ratio: Optional[str] = None  # Prefer string (e.g., "0.29%", "40 bps")
    staking_description: Optional[str] = None  # Brief description from the answer
    aum: Optional[str] = None  # e.g., "$350 million", "$1.2B", "USD 0.13 billion"
    product_url: Optional[str] = None  # Official issuer product page or prospectus (URL)
    extra_sources: List[str] = Field(default_factory=list)  # Additional URLs cited for this ETF


class ETFExtraction(BaseModel):
    etfs: List[ETFItem] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_etfs() -> str:
    return """
    Extract up to four (4) Solana ETFs described in the answer. For each ETF, extract ONLY what is explicitly present in the answer text.
    For each ETF, return the following fields:
    - name: the ETF’s full name (string)
    - ticker: the ETF’s ticker symbol (string)
    - expense_ratio: the stated expense ratio as presented (keep the original percentage or bps text, e.g., "0.25%" or "25 bps")
    - staking_description: a brief description of the staking approach if provided (string). If not provided, return null.
    - aum: the current or most recent reported assets under management as presented (e.g., "$350 million", "$1.2B"). If missing, return null.
    - product_url: a direct URL to the official issuer’s product page or prospectus if the answer includes it as a URL. If not explicitly present as a URL, return null.
    - extra_sources: an array of any other URLs cited for this ETF in the answer (e.g., factsheets, press releases). Include only URLs explicitly present in the answer.

    IMPORTANT:
    - Do NOT invent any URLs. Only include URLs explicitly present in the answer.
    - If a URL is provided in markdown format ([text](url)), extract the actual URL.
    - If the same ETF appears multiple times, include it only once.
    - If the answer mentions more than 4 ETFs, include the first four as they appear.

    Return a JSON object with an array field "etfs".
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def _canonical_ticker(ticker: Optional[str]) -> Optional[str]:
    if not ticker:
        return None
    t = ticker.strip().upper()
    t = re.sub(r"[^A-Z0-9]", "", t)
    return t or None


def _has_percentage_notation(text: Optional[str]) -> bool:
    if not text:
        return False
    t = text.lower()
    return "%" in t or "percent" in t or "bps" in t or "basis point" in t


def _parse_percentage_value(text: Optional[str]) -> Optional[float]:
    """
    Parse a percentage string to a numeric percent value (e.g., "0.35%" -> 0.35; "35 bps" -> 0.35).
    Returns None if parsing fails.
    """
    if not text:
        return None
    t = text.lower().replace(",", " ").strip()
    # Percent pattern
    m = re.search(r"([0-9]+(?:\.[0-9]+)?)\s*%", t)
    if m:
        try:
            return float(m.group(1))
        except Exception:
            return None
    # "percent" word pattern
    m = re.search(r"([0-9]+(?:\.[0-9]+)?)\s*percent", t)
    if m:
        try:
            return float(m.group(1))
        except Exception:
            return None
    # Basis points (bps)
    m = re.search(r"([0-9]+(?:\.[0-9]+)?)\s*(bps|basis\s*points?)", t)
    if m:
        try:
            bps_val = float(m.group(1))
            return bps_val / 100.0  # 1 bp = 0.01%
        except Exception:
            return None
    return None


def _parse_aum_to_millions(text: Optional[str]) -> Optional[float]:
    """
    Parse AUM textual value to millions of USD if possible.
    Examples:
      "$350 million" -> 350.0
      "$1.2B" / "$1.2 billion" -> 1200.0
      "USD 0.15 billion" -> 150.0
      "$800M" -> 800.0
    Returns None if cannot parse.
    """
    if not text:
        return None
    t = text.lower().replace(",", "").strip()

    # Extract the leading numeric portion
    m = re.search(r"([0-9]+(?:\.[0-9]+)?)", t)
    if not m:
        return None
    try:
        value = float(m.group(1))
    except Exception:
        return None

    # Determine unit multiplier
    if re.search(r"\b(billion|bn|b)\b", t):
        return value * 1000.0
    if re.search(r"\b(million|mm|mn|m)\b", t):
        return value
    # Sometimes shown as raw dollars like "$120000000"
    if re.search(r"\$", t) and "million" not in t and "billion" not in t and "m" not in t and "b" not in t and "bn" not in t:
        # Convert bare dollars to millions
        return value / 1_000_000.0
    # If unit not specified but looks like a typical "X M" shorthand (e.g., "120m")
    if re.search(r"[0-9.]+m\b", t):
        return value

    return None


def _ordinal(idx: int) -> str:
    mapping = {0: "first", 1: "second", 2: "third", 3: "fourth"}
    return mapping.get(idx, f"#{idx + 1}")


def _all_sources(etf: ETFItem) -> List[str]:
    urls: List[str] = []
    if etf.product_url and isinstance(etf.product_url, str) and etf.product_url.strip():
        urls.append(etf.product_url.strip())
    for u in (etf.extra_sources or []):
        if isinstance(u, str) and u.strip():
            urls.append(u.strip())
    return urls


# --------------------------------------------------------------------------- #
# Verification for one ETF                                                    #
# --------------------------------------------------------------------------- #
async def verify_single_etf(
    evaluator: Evaluator,
    parent_node,
    etf: ETFItem,
    idx: int,
    seen_tickers: List[str],
) -> None:
    """
    Build verification sub-tree for a single ETF following the rubric.
    """
    ordinal = _ordinal(idx)
    etf_node = evaluator.add_parallel(
        id=f"ETF_{idx + 1}",
        desc=f"{ordinal.capitalize()} Solana ETF identification and verification",
        parent=parent_node,
        critical=False
    )

    # Create the "Official URL" requirement FIRST so we can use it as a precondition
    official_url_present = bool(etf.product_url and etf.product_url.strip().lower().startswith(("http://", "https://")))

    official_url_node = evaluator.add_custom_node(
        result=official_url_present,
        id=f"ETF_{idx + 1}_Official_URL",
        desc=f"A direct URL to the official product page or prospectus is provided for the {ordinal} ETF",
        parent=etf_node,
        critical=True
    )

    # Basic Info (parallel, critical)
    basic_info_node = evaluator.add_parallel(
        id=f"ETF_{idx + 1}_Basic_Info",
        desc="Verify basic ETF identification information",
        parent=etf_node,
        critical=True
    )

    # Name and Ticker provided (critical leaf -> implement as custom)
    name_and_ticker_ok = bool((etf.name and etf.name.strip()) and (etf.ticker and etf.ticker.strip()))
    evaluator.add_custom_node(
        result=name_and_ticker_ok,
        id=f"ETF_{idx + 1}_Name_And_Ticker",
        desc=f"The full name and ticker symbol of the {ordinal} ETF are provided",
        parent=basic_info_node,
        critical=True
    )

    # Is Spot ETF (critical leaf -> verify by URLs)
    is_spot_node = evaluator.add_leaf(
        id=f"ETF_{idx + 1}_Is_Spot_ETF",
        desc=f"The {ordinal} ETF is confirmed to be a spot Solana ETF (not futures-based or leveraged)",
        parent=basic_info_node,
        critical=True
    )
    name_part = etf.name or "the ETF"
    ticker_part = f" ({etf.ticker})" if etf.ticker else ""
    claim_spot = (
        f"The ETF {name_part}{ticker_part} is a spot Solana ETF that holds SOL directly (physically backed), "
        f"and it is not a futures-based or leveraged product."
    )
    await evaluator.verify(
        claim=claim_spot,
        node=is_spot_node,
        sources=_all_sources(etf),
        additional_instruction="Check the issuer's page and any cited official source for language like 'spot', 'physically backed', 'holds SOL directly', or similar. "
                               "It should explicitly not be a futures-based or leveraged ETF.",
        extra_prerequisites=[official_url_node]
    )

    # Trading Status on U.S. exchanges (critical leaf -> verify by URLs)
    trading_node = evaluator.add_leaf(
        id=f"ETF_{idx + 1}_Trading_Status",
        desc=f"The {ordinal} ETF is confirmed to be trading on U.S. exchanges",
        parent=basic_info_node,
        critical=True
    )
    claim_trading = (
        f"The ETF {name_part}{ticker_part} is listed and currently trading on a U.S. exchange such as NYSE Arca, Nasdaq, or Cboe BZX."
    )
    await evaluator.verify(
        claim=claim_trading,
        node=trading_node,
        sources=_all_sources(etf),
        additional_instruction="Look for explicit exchange listing such as 'Exchange: NYSE Arca', 'NASDAQ', or 'Cboe BZX' on the issuer page or official documents.",
        extra_prerequisites=[official_url_node]
    )

    # Uniqueness constraints for ETF #2, #3, #4 (critical under Basic Info)
    if idx >= 1:
        this_ticker = _canonical_ticker(etf.ticker)
        distinct = bool(this_ticker) and (this_ticker not in seen_tickers)
        evaluator.add_custom_node(
            result=distinct,
            id=f"ETF_{idx + 1}_Uniqueness",
            desc=(
                "The second ETF is different from the first ETF" if idx == 1 else
                ("The third ETF is different from the first and second ETFs" if idx == 2 else
                 "The fourth ETF is different from the first, second, and third ETFs")
            ),
            parent=basic_info_node,
            critical=True
        )

    # Cost Criteria (sequential, critical)
    cost_node = evaluator.add_sequential(
        id=f"ETF_{idx + 1}_Cost_Criteria",
        desc=f"Verify the {ordinal} ETF meets the expense ratio requirement",
        parent=etf_node,
        critical=True
    )

    # Expense ratio stated (custom existence/format)
    expense_stated = _has_percentage_notation(etf.expense_ratio)
    evaluator.add_custom_node(
        result=expense_stated,
        id=f"ETF_{idx + 1}_Expense_Ratio_Stated",
        desc=f"The expense ratio is stated as a percentage for the {ordinal} ETF",
        parent=cost_node,
        critical=True
    )

    # Expense ratio <= 0.40% (custom threshold)
    exp_pct = _parse_percentage_value(etf.expense_ratio)
    expense_threshold_ok = (exp_pct is not None) and (exp_pct <= 0.40)
    evaluator.add_custom_node(
        result=expense_threshold_ok,
        id=f"ETF_{idx + 1}_Expense_Ratio_Threshold",
        desc=f"The stated expense ratio is 0.40% or lower for the {ordinal} ETF",
        parent=cost_node,
        critical=True
    )

    # Staking Criteria (sequential, critical)
    staking_node = evaluator.add_sequential(
        id=f"ETF_{idx + 1}_Staking_Criteria",
        desc=f"Verify the {ordinal} ETF offers staking features",
        parent=etf_node,
        critical=True
    )

    # Staking feature present (verify by URLs)
    staking_feature_node = evaluator.add_leaf(
        id=f"ETF_{idx + 1}_Staking_Feature",
        desc=f"The {ordinal} ETF offers staking features (stakes SOL holdings)",
        parent=staking_node,
        critical=True
    )
    claim_staking_feature = (
        f"The ETF {name_part}{ticker_part} stakes a portion of its SOL holdings (staking enabled)."
    )
    await evaluator.verify(
        claim=claim_staking_feature,
        node=staking_feature_node,
        sources=_all_sources(etf),
        additional_instruction="Look for explicit mentions of 'staking', 'validator', 'staking program', or similar on the issuer page or official docs.",
        extra_prerequisites=[official_url_node]
    )

    # Staking description provided (custom existence)
    staking_desc_ok = bool(etf.staking_description and etf.staking_description.strip())
    evaluator.add_custom_node(
        result=staking_desc_ok,
        id=f"ETF_{idx + 1}_Staking_Description",
        desc=f"A description of the staking approach is provided for the {ordinal} ETF",
        parent=staking_node,
        critical=True
    )

    # Staking rewards are passed to investors (verify by URLs)
    staking_rewards_node = evaluator.add_leaf(
        id=f"ETF_{idx + 1}_Staking_Rewards_Passed",
        desc=f"The {ordinal} ETF passes staking rewards to investors",
        parent=staking_node,
        critical=True
    )
    claim_rewards = (
        f"The ETF {name_part}{ticker_part} passes staking rewards to investors/shareholders (e.g., via distributions or NAV accrual)."
    )
    await evaluator.verify(
        claim=claim_rewards,
        node=staking_rewards_node,
        sources=_all_sources(etf),
        additional_instruction="Look for language indicating staking rewards are passed through to fund shareholders (e.g., as income distributions or NAV additions).",
        extra_prerequisites=[official_url_node]
    )

    # AUM Criteria (sequential, critical)
    aum_node = evaluator.add_sequential(
        id=f"ETF_{idx + 1}_AUM_Criteria",
        desc=f"Verify the {ordinal} ETF meets the AUM requirement",
        parent=etf_node,
        critical=True
    )

    # AUM stated (custom existence)
    aum_stated_ok = bool(etf.aum and etf.aum.strip())
    evaluator.add_custom_node(
        result=aum_stated_ok,
        id=f"ETF_{idx + 1}_AUM_Stated",
        desc=f"The AUM is stated for the {ordinal} ETF",
        parent=aum_node,
        critical=True
    )

    # AUM >= $100 million (custom threshold from stated value)
    aum_millions = _parse_aum_to_millions(etf.aum)
    aum_threshold_ok = (aum_millions is not None) and (aum_millions >= 100.0)
    evaluator.add_custom_node(
        result=aum_threshold_ok,
        id=f"ETF_{idx + 1}_AUM_Threshold",
        desc=f"The stated AUM is at least $100 million for the {ordinal} ETF",
        parent=aum_node,
        critical=True
    )

    # Update seen tickers after processing this ETF
    ct = _canonical_ticker(etf.ticker)
    if ct:
        seen_tickers.append(ct)


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
    Evaluate an answer for the Solana spot ETF with staking portfolio construction task.
    """
    # Initialize evaluator with a parallel root (we'll add our own top-level node according to rubric)
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

    # Extract ETFs from the answer
    extraction = await evaluator.extract(
        prompt=prompt_extract_etfs(),
        template_class=ETFExtraction,
        extraction_name="sol_spot_etfs_extraction",
    )

    # Add top-level portfolio construction node (parallel aggregator)
    portfolio_node = evaluator.add_parallel(
        id="Portfolio_Construction",
        desc="Evaluate whether the solution correctly identifies four different spot Solana ETFs that meet all specified investment criteria",
        parent=root,
        critical=False  # Set to non-critical to allow partial scoring across ETFs
    )

    # Prepare up to 4 ETFs (pad with empty placeholders if fewer)
    etfs: List[ETFItem] = list(extraction.etfs[:4]) if extraction and extraction.etfs else []
    while len(etfs) < 4:
        etfs.append(ETFItem())

    # Verify each ETF per rubric
    seen_tickers: List[str] = []
    for i in range(4):
        await verify_single_etf(
            evaluator=evaluator,
            parent_node=portfolio_node,
            etf=etfs[i],
            idx=i,
            seen_tickers=seen_tickers,
        )

    # Return structured evaluation summary
    return evaluator.get_summary()