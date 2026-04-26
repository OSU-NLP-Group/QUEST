import asyncio
import logging
import re
from typing import Optional, List, Dict, Any, Tuple

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


TASK_ID = "crypto_etf_portfolio"
TASK_DESCRIPTION = """As a financial advisor building a diversified cryptocurrency investment portfolio for clients, you need to identify three spot cryptocurrency ETFs that meet specific institutional criteria. Select one ETF each for Bitcoin, Ethereum, and one other major cryptocurrency (such as XRP), ensuring that: (1) Each ETF has an expense ratio of 0.35% or lower; (2) The Bitcoin ETF has assets under management (AUM) of at least $500 million; (3) The Ethereum ETF has AUM of at least $500 million (or is among the largest available); (4) The third cryptocurrency ETF has AUM of at least $200 million; (5) For each cryptocurrency, you select one of the lowest-fee options that meets the AUM requirement; (6) No more than two ETFs in your portfolio come from the same issuer; (7) All ETFs are spot ETFs (not futures-based or leveraged); (8) All ETFs are currently live and trading (not pending approval). For each selected ETF, provide: the ETF ticker symbol, the expense ratio, the current AUM, the issuer/provider name, the cryptocurrency it tracks, and a reference URL supporting your selection. Your selection should optimize for the lowest fees while maintaining diversification across both cryptocurrencies and issuers."""


# ----------------------- Data Models ----------------------- #
class ETFItem(BaseModel):
    ticker: Optional[str] = None
    expense_ratio: Optional[str] = None
    aum: Optional[str] = None
    issuer: Optional[str] = None
    cryptocurrency: Optional[str] = None
    reference_urls: List[str] = Field(default_factory=list)
    # Optional soft fields, if present in answer
    spot_type_mentioned: Optional[bool] = None
    trading_status: Optional[str] = None
    tax_reporting_form: Optional[str] = None


class PortfolioExtraction(BaseModel):
    etfs: List[ETFItem] = Field(default_factory=list)


# ----------------------- Extraction Prompt ----------------------- #
def prompt_extract_portfolio() -> str:
    return """
    Extract the three selected cryptocurrency ETFs from the answer. Only include the ETFs that the answer presents as the final portfolio selection (ignore any non-selected comparisons).
    For each selected ETF, extract:
    - ticker: The ETF ticker symbol
    - expense_ratio: The stated expense ratio exactly as written (e.g., "0.25%", "25 bps")
    - aum: The current assets under management exactly as written (e.g., "$600M", "$0.5B", "$500,000,000")
    - issuer: The issuer/provider/fund company name
    - cryptocurrency: The cryptocurrency tracked (e.g., "Bitcoin", "BTC", "Ethereum", "ETH", "XRP", "Solana")
    - reference_urls: A list of all URLs referenced to support the ETF details (expense ratio and AUM at minimum). Include official provider pages or reputable finance sources.
    - spot_type_mentioned: If the answer explicitly says it is "spot", set true; if it says "futures", "leveraged", "inverse", set false; else null.
    - trading_status: If the answer states the ETF is "live", "trading", "pending"/"approved", capture that phrase; else null.
    - tax_reporting_form: If the answer mentions tax reporting form (e.g., "Form 1099-B" or "no K-1"), extract it; else null.

    Return as: {"etfs": [ ... up to 3 items ... ]}
    If the answer provides more than three ETFs, include only the first three presented as part of the selected portfolio.
    If any field is missing, set it to null. Ensure reference_urls are actual URLs (http/https).
    """


# ----------------------- Helpers ----------------------- #
def is_valid_url(url: Optional[str]) -> bool:
    if not url:
        return False
    return bool(re.match(r'^(https?://)', url.strip()))


def normalize_crypto(name: Optional[str]) -> Optional[str]:
    if not name:
        return None
    s = name.strip().lower()
    if s in {"btc", "bitcoin", "xbt"}:
        return "BTC"
    if s in {"eth", "ethereum"}:
        return "ETH"
    # Normalize common majors
    if s in {"xrp", "ripple"}:
        return "XRP"
    if s in {"sol", "solana"}:
        return "SOL"
    if s in {"ada", "cardano"}:
        return "ADA"
    if s in {"ltc", "litecoin"}:
        return "LTC"
    # Fallback: uppercase symbol-ish
    return s.upper()


def parse_percent(pct_str: Optional[str]) -> Optional[float]:
    """
    Parse an expense ratio string to a floating percentage value (e.g., "0.25%" -> 0.25 ; "25 bps" -> 0.25).
    Returns None if parsing fails.
    """
    if not pct_str:
        return None
    s = pct_str.strip().lower()
    # Basis points
    m_bps = re.search(r'([0-9]+(?:\.[0-9]+)?)\s*bps', s)
    if m_bps:
        try:
            val = float(m_bps.group(1))
            return val / 100.0
        except:
            return None
    # Percent with %
    m_pct = re.search(r'([0-9]+(?:\.[0-9]+)?)\s*%', s)
    if m_pct:
        try:
            return float(m_pct.group(1))
        except:
            return None
    # Plain number (assume percent)
    m_num = re.search(r'([0-9]+(?:\.[0-9]+)?)', s)
    if m_num:
        try:
            val = float(m_num.group(1))
            # If looks like 0.0025 might be fraction; treat <= 1 as percent already
            return val
        except:
            return None
    return None


def parse_aum_usd(aum_str: Optional[str]) -> Optional[float]:
    """
    Parse AUM strings to USD dollars (float). Handles formats:
    - "$600M", "$0.5B", "$500,000,000", "USD 750 million", "US$ 1.2 billion"
    Returns None if cannot parse.
    """
    if not aum_str:
        return None
    s = aum_str.strip().lower()
    s_clean = re.sub(r'[\$,]', '', s)
    # Determine multiplier
    mult = 1.0
    if 'b' in s or 'billion' in s_clean or 'bn' in s_clean or 'bln' in s_clean:
        mult = 1_000_000_000.0
    elif 'm' in s or 'million' in s_clean:
        mult = 1_000_000.0
    elif 'k' in s or 'thousand' in s_clean:
        mult = 1_000.0
    # Extract number
    m = re.search(r'([0-9]+(?:\.[0-9]+)?)', s_clean)
    if not m:
        return None
    try:
        val = float(m.group(1))
        # If no unit hints and value seems very large, assume dollars already
        # Otherwise, apply multiplier
        # Heuristic: if 'million/billion' present we used multiplier; else assume raw dollars
        if mult != 1.0:
            return val * mult
        # If raw integer with commas removed, treat as dollars
        return val
    except:
        return None


def first_three_selected(etfs: List[ETFItem]) -> List[ETFItem]:
    return etfs[:3] if len(etfs) >= 3 else etfs + [ETFItem() for _ in range(3 - len(etfs))]


def count_issuer_occurrences(etfs: List[ETFItem]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for e in etfs:
        issuer = (e.issuer or "").strip().lower()
        if not issuer:
            # Treat missing issuer as unique bucket to not penalize diversity check
            issuer = f"__missing__{id(e)}"
        counts[issuer] = counts.get(issuer, 0) + 1
    return counts


def etf_urls(etf: ETFItem) -> List[str]:
    return [u for u in (etf.reference_urls or []) if is_valid_url(u)]


def build_lowest_fee_claim(etf: ETFItem, crypto_code: str, aum_requirement_text: str) -> str:
    return (
        f"The selected spot {crypto_code} ETF {etf.ticker or '[unknown ticker]'} has an expense ratio of "
        f"{etf.expense_ratio or '[unknown expense ratio]'}, and it is one of the lowest-fee options among spot "
        f"{crypto_code} ETFs that meet the {aum_requirement_text} requirement."
    )


# ----------------------- Verification Builders ----------------------- #
async def verify_portfolio_composition(
    evaluator: Evaluator,
    parent_node,
    selected: List[ETFItem]
) -> None:
    node = evaluator.add_parallel(
        id="Portfolio_Composition",
        desc="Portfolio includes exactly 3 ETFs covering Bitcoin, Ethereum, and exactly one other major cryptocurrency (not Bitcoin or Ethereum).",
        parent=parent_node,
        critical=True
    )
    # Count checks
    count_ok = len([e for e in selected if e.ticker or e.cryptocurrency or e.issuer]) == 3
    evaluator.add_custom_node(
        result=count_ok,
        id="Portfolio_Composition_Count",
        desc="Exactly three ETFs are selected in the portfolio.",
        parent=node,
        critical=True
    )

    # Crypto coverage checks within selected
    norm_cryptos = [normalize_crypto(e.cryptocurrency) for e in selected]
    has_btc = any(c == "BTC" for c in norm_cryptos)
    has_eth = any(c == "ETH" for c in norm_cryptos)
    other_count = sum(1 for c in norm_cryptos if c and c not in {"BTC", "ETH"})

    evaluator.add_custom_node(
        result=has_btc,
        id="Portfolio_Composition_Has_BTC",
        desc="Portfolio contains exactly one Bitcoin ETF.",
        parent=node,
        critical=True
    )
    evaluator.add_custom_node(
        result=has_eth,
        id="Portfolio_Composition_Has_ETH",
        desc="Portfolio contains exactly one Ethereum ETF.",
        parent=node,
        critical=True
    )
    evaluator.add_custom_node(
        result=(other_count == 1),
        id="Portfolio_Composition_Has_One_Other",
        desc="Portfolio contains exactly one ETF tracking a non-BTC, non-ETH major cryptocurrency.",
        parent=node,
        critical=True
    )


async def verify_portfolio_issuer_diversity(
    evaluator: Evaluator,
    parent_node,
    selected: List[ETFItem]
) -> None:
    node = evaluator.add_leaf(
        id="Portfolio_Issuer_Diversity",
        desc="No more than two of the three ETFs are from the same issuer/provider.",
        parent=parent_node,
        critical=True
    )
    counts = count_issuer_occurrences(selected)
    violates = any(cnt >= 3 for cnt in counts.values())
    claim = "In this portfolio, no issuer appears three times; at most two ETFs are from the same issuer."
    await evaluator.verify(
        claim=claim,
        node=node,
        additional_instruction="Use the extracted issuer names for the three selected ETFs to check if any issuer occurs 3 times."
    )


async def verify_portfolio_trading_status(
    evaluator: Evaluator,
    parent_node,
    selected: List[ETFItem]
) -> None:
    node = evaluator.add_parallel(
        id="Portfolio_Trading_Status",
        desc="All selected ETFs are currently live and actively trading (not pending approval).",
        parent=parent_node,
        critical=True
    )
    for idx, etf in enumerate(selected):
        leaf = evaluator.add_leaf(
            id=f"ETF_{idx}_Live_Trading",
            desc=f"ETF {etf.ticker or '[unknown]'} is live and actively trading.",
            parent=node,
            critical=True
        )
        urls = etf_urls(etf)
        claim = f"The ETF {etf.ticker or '[unknown ticker]'} is live and actively trading (not pending)."
        await evaluator.verify(
            claim=claim,
            node=leaf,
            sources=urls,
            additional_instruction="Verify on the provided official or finance source pages that the ETF is currently trading/listed (not pending approval)."
        )


async def verify_portfolio_tax_reporting(
    evaluator: Evaluator,
    parent_node,
    selected: List[ETFItem]
) -> None:
    node = evaluator.add_parallel(
        id="Portfolio_Tax_Reporting",
        desc="All selected ETFs use Form 1099-B for tax reporting.",
        parent=parent_node,
        critical=True
    )
    for idx, etf in enumerate(selected):
        leaf = evaluator.add_leaf(
            id=f"ETF_{idx}_Uses_1099B",
            desc=f"ETF {etf.ticker or '[unknown]'} uses Form 1099-B for tax reporting.",
            parent=node,
            critical=True
        )
        urls = etf_urls(etf)
        claim = f"The ETF {etf.ticker or '[unknown ticker]'} uses IRS Form 1099-B for tax reporting (no K-1)."
        await evaluator.verify(
            claim=claim,
            node=leaf,
            sources=urls,
            additional_instruction="Look for tax reporting disclosures (e.g., '1099', 'no K‑1') on issuer pages or reputable finance sources. If not mentioned at all, consider it not supported."
        )


async def verify_single_etf_selection(
    evaluator: Evaluator,
    parent_node,
    etf: ETFItem,
    selection_id: str,
    selection_desc: str,
    expected_crypto: Optional[str],  # "BTC", "ETH", or None for 'Other'
    aum_requirement_text: str,
    aum_threshold_usd: Optional[float],  # e.g., 500_000_000 for BTC; None for ETH flexible criterion; 200_000_000 for Other
    allow_eth_largest_available: bool = False
) -> None:
    sel_node = evaluator.add_parallel(
        id=selection_id,
        desc=selection_desc,
        parent=parent_node,
        critical=True
    )

    # Existence nodes
    evaluator.add_custom_node(
        result=bool(etf.ticker and etf.ticker.strip()),
        id=f"{selection_id}_Ticker_Provided",
        desc="Ticker symbol is provided.",
        parent=sel_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=bool(etf.issuer and etf.issuer.strip()),
        id=f"{selection_id}_Issuer_Provided",
        desc="Issuer/provider name is provided.",
        parent=sel_node,
        critical=True
    )

    # Reference URL presence
    has_url = len(etf_urls(etf)) > 0
    evaluator.add_custom_node(
        result=has_url,
        id=f"{selection_id}_Reference_URL_Provided",
        desc="At least one valid reference URL from a reliable financial source is provided supporting the ETF details (at minimum expense ratio and AUM).",
        parent=sel_node,
        critical=True
    )

    # Spot type verification
    spot_leaf = evaluator.add_leaf(
        id=f"{selection_id}_Spot_Type",
        desc="ETF is a spot cryptocurrency ETF (not futures-based, leveraged, or inverse).",
        parent=sel_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The ETF {etf.ticker or '[unknown ticker]'} is a spot ETF (not futures-based, not leveraged/inverse).",
        node=spot_leaf,
        sources=etf_urls(etf),
        additional_instruction="Confirm via the ETF's official page or credible finance sources that it is structured as a spot ETF (holds or directly tracks the crypto), not futures/leveraged/inverse."
    )

    # Tracks crypto verification
    tracks_leaf = evaluator.add_leaf(
        id=f"{selection_id}_Tracks_Crypto",
        desc=f"ETF tracks {expected_crypto or 'a major cryptocurrency other than BTC and ETH'}.",
        parent=sel_node,
        critical=True
    )
    if expected_crypto in {"BTC", "ETH"}:
        claim_track = f"The ETF {etf.ticker or '[unknown ticker]'} tracks {expected_crypto}."
    else:
        # Other crypto: must be non-BTC, non-ETH, explicitly stated
        normalized = normalize_crypto(etf.cryptocurrency)
        claim_track = (
            f"The ETF {etf.ticker or '[unknown ticker]'} tracks {normalized or '[unknown crypto]'}, "
            f"which is neither Bitcoin nor Ethereum."
        )
    await evaluator.verify(
        claim=claim_track,
        node=tracks_leaf,
        sources=etf_urls(etf),
        additional_instruction="Verify the tracked asset on the ETF page or finance sources. For 'Other', ensure the crypto is explicitly stated and not BTC/ETH."
    )

    # Expense ratio provided AND threshold
    pct_val = parse_percent(etf.expense_ratio)
    exp_provided_and_thresh = (pct_val is not None) and (pct_val <= 0.35)
    evaluator.add_custom_node(
        result=exp_provided_and_thresh,
        id=f"{selection_id}_Expense_Ratio_Provided_And_Threshold",
        desc="Expense ratio is provided and is ≤ 0.35%.",
        parent=sel_node,
        critical=True
    )
    # Verify the expense ratio against sources
    exp_leaf = evaluator.add_leaf(
        id=f"{selection_id}_Expense_Ratio_Supported",
        desc="Provided expense ratio is supported by the cited sources.",
        parent=sel_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The expense ratio for {etf.ticker or '[unknown ticker]'} is {etf.expense_ratio or '[unknown expense ratio]'}.",
        node=exp_leaf,
        sources=etf_urls(etf),
        additional_instruction="Check the fee/expense section. Allow minor wording variations but focus on the numerical ratio."
    )

    # AUM checks
    aum_val = parse_aum_usd(etf.aum)
    aum_provided_leaf = evaluator.add_custom_node(
        result=aum_val is not None,
        id=f"{selection_id}_AUM_Provided",
        desc="Current AUM is provided.",
        parent=sel_node,
        critical=True
    )

    if expected_crypto == "BTC":
        aum_leaf = evaluator.add_leaf(
            id=f"{selection_id}_AUM_Threshold",
            desc="Current AUM is provided and is ≥ $500 million.",
            parent=sel_node,
            critical=True
        )
        await evaluator.verify(
            claim=f"The ETF {etf.ticker or '[unknown]'} has AUM of at least $500 million.",
            node=aum_leaf,
            sources=etf_urls(etf),
            additional_instruction="Confirm net assets/AUM on the cited sources. Allow approximate rounding but ensure ≥ $500M."
        )
    elif expected_crypto == "ETH" and allow_eth_largest_available:
        # Flexible criterion: ≥ $500M OR among largest available
        eth_aum_leaf = evaluator.add_leaf(
            id=f"{selection_id}_AUM_Criterion",
            desc="Current AUM meets criterion: ≥ $500 million OR among the largest available spot Ethereum ETFs.",
            parent=sel_node,
            critical=True
        )
        claim = (
            f"The ETF {etf.ticker or '[unknown]'} has AUM {etf.aum or '[unknown]'} and meets the AUM "
            f"criterion: either at least $500 million or among the largest available spot Ethereum ETFs."
        )
        await evaluator.verify(
            claim=claim,
            node=eth_aum_leaf,
            sources=etf_urls(etf),
            additional_instruction="Verify AUM on sources. If < $500M, check whether reputable sources indicate this ETF is among the largest spot ETH ETFs."
        )
    else:
        # Other crypto threshold (e.g., ≥ $200M)
        thr = aum_threshold_usd or 200_000_000.0
        other_aum_leaf = evaluator.add_leaf(
            id=f"{selection_id}_AUM_Threshold",
            desc=f"Current AUM is provided and is ≥ ${int(thr):,}.",
            parent=sel_node,
            critical=True
        )
        await evaluator.verify(
            claim=f"The ETF {etf.ticker or '[unknown]'} has AUM of at least ${int(thr):,}.",
            node=other_aum_leaf,
            sources=etf_urls(etf),
            additional_instruction="Confirm net assets/AUM on the cited sources. Allow approximate rounding."
        )

    # Lowest-fee among eligible that meet AUM requirement
    lowest_fee_leaf = evaluator.add_leaf(
        id=f"{selection_id}_Lowest_Fee_Among_Eligible",
        desc="Selected ETF is one of the lowest-fee options among spot ETFs for that cryptocurrency that meet the AUM requirement.",
        parent=sel_node,
        critical=True
    )
    claim_lowest = build_lowest_fee_claim(etf, expected_crypto or (normalize_crypto(etf.cryptocurrency) or "OTHER"), aum_requirement_text)
    await evaluator.verify(
        claim=claim_lowest,
        node=lowest_fee_leaf,
        # Comparative claims often rely on broad context; still include URLs if available
        sources=etf_urls(etf),
        additional_instruction=(
            "Use the provided answer context and known fee levels to judge whether this ETF's fee is among the lowest "
            "for spot ETFs tracking that crypto that meet the stated AUM criterion. Allow ties for lowest."
        )
    )


# ----------------------- Main Evaluate Function ----------------------- #
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

    # Extract portfolio selection
    extraction = await evaluator.extract(
        prompt=prompt_extract_portfolio(),
        template_class=PortfolioExtraction,
        extraction_name="portfolio_selection"
    )

    # Prepare selected ETFs (first three)
    selected = first_three_selected(extraction.etfs)

    # Create critical portfolio-level node
    portfolio_node = evaluator.add_parallel(
        id="Cryptocurrency_ETF_Portfolio_Selection",
        desc="Evaluate a portfolio of exactly three live, spot cryptocurrency ETFs (Bitcoin, Ethereum, and one other major cryptocurrency) against fee/AUM constraints, lowest-fee selection intent, issuer diversification, required outputs, and sourcing.",
        parent=root,
        critical=True
    )

    # Portfolio composition checks
    await verify_portfolio_composition(evaluator, portfolio_node, selected)

    # Issuer diversity check
    await verify_portfolio_issuer_diversity(evaluator, portfolio_node, selected)

    # Portfolio trading status check (all 3)
    await verify_portfolio_trading_status(evaluator, portfolio_node, selected)

    # Portfolio tax reporting check (all 3 use 1099-B)
    await verify_portfolio_tax_reporting(evaluator, portfolio_node, selected)

    # Map selected to categories (within the three)
    norm_cryptos = [normalize_crypto(e.cryptocurrency) for e in selected]
    # Identify BTC, ETH, OTHER
    btc_idx = next((i for i, c in enumerate(norm_cryptos) if c == "BTC"), None)
    eth_idx = next((i for i, c in enumerate(norm_cryptos) if c == "ETH"), None)
    other_idx = next((i for i, c in enumerate(norm_cryptos) if c not in {"BTC", "ETH"} and c is not None), None)

    # BTC selection verification
    btc_etf = selected[btc_idx] if btc_idx is not None else ETFItem()
    await verify_single_etf_selection(
        evaluator=evaluator,
        parent_node=portfolio_node,
        etf=btc_etf,
        selection_id="Bitcoin_ETF_Selection",
        selection_desc="Evaluate the selected Bitcoin spot ETF against all required criteria and required reported fields.",
        expected_crypto="BTC",
        aum_requirement_text="≥ $500 million",
        aum_threshold_usd=500_000_000,
        allow_eth_largest_available=False
    )

    # ETH selection verification (flexible AUM criterion)
    eth_etf = selected[eth_idx] if eth_idx is not None else ETFItem()
    await verify_single_etf_selection(
        evaluator=evaluator,
        parent_node=portfolio_node,
        etf=eth_etf,
        selection_id="Ethereum_ETF_Selection",
        selection_desc="Evaluate the selected Ethereum spot ETF against all required criteria and required reported fields.",
        expected_crypto="ETH",
        aum_requirement_text="≥ $500 million OR among the largest available",
        aum_threshold_usd=None,
        allow_eth_largest_available=True
    )

    # Other crypto selection verification (AUM ≥ $200M)
    other_etf = selected[other_idx] if other_idx is not None else ETFItem()
    await verify_single_etf_selection(
        evaluator=evaluator,
        parent_node=portfolio_node,
        etf=other_etf,
        selection_id="Other_Crypto_ETF_Selection",
        selection_desc="Evaluate the selected third (non-BTC, non-ETH) major cryptocurrency spot ETF against all required criteria and required reported fields.",
        expected_crypto=None,
        aum_requirement_text="≥ $200 million",
        aum_threshold_usd=200_000_000,
        allow_eth_largest_available=False
    )

    return evaluator.get_summary()