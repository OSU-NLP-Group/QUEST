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
TASK_ID = "crypto_etf_portfolio_low_cost"
TASK_DESCRIPTION = (
    "A financial advisor is building a low-cost cryptocurrency ETF portfolio for a client who wants diversified exposure "
    "to major digital assets while minimizing fees. The portfolio must include exactly four ETFs that collectively meet "
    "the following requirements:\n\n"
    "1. Exactly one ETF must track XRP with an expense ratio of 0.25% or lower\n"
    "2. Exactly one ETF must track Solana with an expense ratio of 0.25% or lower\n"
    "3. Exactly one ETF must track Bitcoin with an expense ratio of 0.25% or lower\n"
    "4. Exactly one ETF must track Ethereum with an expense ratio of 0.35% or lower\n"
    "5. All four ETFs must have been launched before December 1, 2025\n"
    "6. At least three of the four ETFs must offer fee waivers that remain active through at least May 31, 2026\n"
    "7. The four ETFs must represent at least three different issuers/sponsors\n\n"
    "For each of the four selected ETFs, provide:\n"
    "- Official fund name\n"
    "- Ticker symbol\n"
    "- Issuer/sponsor name\n"
    "- Exact expense ratio (stated annual fee)\n"
    "- Fee waiver details (coverage amount and expiration date, if applicable)\n"
    "- Launch date\n"
    "- Reference URL to the official fund page or regulatory filing"
)

LAUNCH_DEADLINE_DATE_TEXT = "December 1, 2025"
FEE_WAIVER_DEADLINE_DATE_TEXT = "May 31, 2026"

# Numeric thresholds for verification
EXPENSE_THRESHOLDS = {
    "XRP": 0.25,
    "SOL": 0.25,
    "BTC": 0.25,
    "ETH": 0.35,
}

# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class ETFItem(BaseModel):
    fund_name: Optional[str] = None
    ticker: Optional[str] = None
    issuer: Optional[str] = None
    asset: Optional[str] = None  # Canonical values expected: XRP, SOL, BTC, ETH
    expense_ratio: Optional[str] = None  # e.g., "0.25%" or "0.24%"
    fee_waiver_details: Optional[str] = None  # e.g., "Waived to 0.00% through 2026-06-30"
    launch_date: Optional[str] = None  # e.g., "2024-11-15" or "Nov 15, 2024"
    reference_url: Optional[str] = None  # Official fund page or regulatory filing URL


class ETFExtraction(BaseModel):
    etfs: List[ETFItem] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_etfs() -> str:
    return (
        "Extract all cryptocurrency ETFs mentioned in the answer. For each ETF, return these fields:\n"
        "- fund_name: The official fund name as stated\n"
        "- ticker: The ETF ticker symbol\n"
        "- issuer: The issuer/sponsor company name\n"
        "- asset: The underlying asset tracked; map to canonical one of [XRP, SOL, BTC, ETH]. "
        "If the answer uses variants (e.g., Ripple -> XRP, Solana -> SOL, Ether -> ETH, Bitcoin -> BTC), normalize to the canonical symbol.\n"
        "- expense_ratio: The stated annual expense ratio as a percentage string (e.g., '0.25%')\n"
        "- fee_waiver_details: Any fee waiver statement including the waiver amount and expiration date, if available (otherwise null)\n"
        "- launch_date: The launch/inception date as stated in the answer (string)\n"
        "- reference_url: A URL to the official fund page OR an official regulatory filing page (e.g., prospectus, SEC filing). "
        "If multiple URLs are provided, select the most official/specific fund page.\n\n"
        "Return a JSON object with a single key 'etfs' containing an array of ETF objects. If any field is not present in the answer, set it to null. "
        "Extract all ETFs explicitly present in the answer text."
    )


# --------------------------------------------------------------------------- #
# Helper selection and normalization                                          #
# --------------------------------------------------------------------------- #
def _normalize_asset(asset: Optional[str]) -> Optional[str]:
    if not asset:
        return None
    a = asset.strip().upper()
    # Map common variants to canonical
    if a in {"XRP", "RIPPLE"}:
        return "XRP"
    if a in {"SOL", "SOLANA"}:
        return "SOL"
    if a in {"BTC", "BITCOIN"}:
        return "BTC"
    if a in {"ETH", "ETHEREUM", "ETHER"}:
        return "ETH"
    return a  # fallback


def _matches_asset(etf: ETFItem, target: str) -> bool:
    """Check if the ETF matches the target asset (canonical)."""
    target = target.upper()
    asset_norm = _normalize_asset(etf.asset)
    if asset_norm == target:
        return True
    # Fallback heuristics using fund_name/ticker
    name = (etf.fund_name or "").lower()
    tick = (etf.ticker or "").lower()
    if target == "XRP":
        return ("xrp" in name) or ("xrp" in tick) or ("ripple" in name)
    if target == "SOL":
        return ("solana" in name) or ("sol" in tick)
    if target == "BTC":
        return ("bitcoin" in name) or ("btc" in tick)
    if target == "ETH":
        return ("ethereum" in name) or ("eth" in tick) or ("ether" in name)
    return False


def select_one_by_asset(etfs: List[ETFItem], target_asset: str) -> ETFItem:
    """Select the first ETF that matches the target asset; return empty placeholder if none."""
    for etf in etfs:
        if _matches_asset(etf, target_asset):
            return etf
    return ETFItem()  # placeholder with nulls


# --------------------------------------------------------------------------- #
# Verification for a single ETF group                                         #
# --------------------------------------------------------------------------- #
async def verify_single_etf(
    evaluator: Evaluator,
    parent_node,
    etf: ETFItem,
    group_id: str,
    required_asset: str,
    max_expense_pct: float,
) -> Dict[str, Any]:
    """
    Build verification nodes and run checks for a single ETF grouped by asset.
    Returns a dict with useful info for portfolio-level constraints:
      { 'fee_waiver_leaf': VerificationNode, 'issuer': str }
    """
    # Create ETF group node (parallel aggregation; allow partial credit)
    etf_node = evaluator.add_parallel(
        id=group_id,
        desc=f"Select one {required_asset} ETF meeting all specified requirements",
        parent=parent_node,
        critical=False,
    )

    # Presence checks (non-critical custom nodes)
    fund_name_node = evaluator.add_custom_node(
        result=bool(etf.fund_name and etf.fund_name.strip()),
        id=f"{group_id}_Fund_Name",
        desc="The official fund name is provided",
        parent=etf_node,
        critical=False
    )

    ticker_node = evaluator.add_custom_node(
        result=bool(etf.ticker and etf.ticker.strip()),
        id=f"{group_id}_Ticker",
        desc="The ticker symbol is provided",
        parent=etf_node,
        critical=False
    )

    issuer_node = evaluator.add_custom_node(
        result=bool(etf.issuer and etf.issuer.strip()),
        id=f"{group_id}_Issuer",
        desc="The issuer/sponsor name is provided",
        parent=etf_node,
        critical=False
    )

    ref_url_node = evaluator.add_custom_node(
        result=bool(etf.reference_url and etf.reference_url.strip()),
        id=f"{group_id}_Reference_URL",
        desc="A reference URL to the official fund page or regulatory filing is provided",
        parent=etf_node,
        critical=False
    )

    # Critical checks with source verification; use reference_url as prerequisite
    asset_leaf = evaluator.add_leaf(
        id=f"{group_id}_Asset_Type",
        desc=f"The ETF tracks {required_asset} as its underlying asset",
        parent=etf_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"This ETF tracks {required_asset} as its underlying digital asset.",
        node=asset_leaf,
        sources=etf.reference_url,
        additional_instruction=(
            "Check the official fund page/regulatory filing to confirm the ETF's underlying asset. "
            "Allow reasonable naming variants (e.g., Ripple -> XRP, Solana -> SOL, Bitcoin -> BTC, Ethereum/Ether -> ETH)."
        ),
        extra_prerequisites=[ref_url_node]
    )

    expense_leaf = evaluator.add_leaf(
        id=f"{group_id}_Expense_Ratio",
        desc=f"The ETF has an expense ratio of {max_expense_pct:.2f}% or lower",
        parent=etf_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The ETF's stated annual expense ratio is less than or equal to {max_expense_pct:.2f}%.",
        node=expense_leaf,
        sources=etf.reference_url,
        additional_instruction=(
            "Verify the expense ratio on the official page; if the page shows a percentage <= the threshold, pass. "
            "Minor rounding differences are acceptable."
        ),
        extra_prerequisites=[ref_url_node]
    )

    launch_leaf = evaluator.add_leaf(
        id=f"{group_id}_Launch_Date",
        desc=f"The ETF was launched before {LAUNCH_DEADLINE_DATE_TEXT}",
        parent=etf_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The ETF's launch/inception date is before {LAUNCH_DEADLINE_DATE_TEXT}.",
        node=launch_leaf,
        sources=etf.reference_url,
        additional_instruction=(
            "Use the 'launch', 'inception', or 'listing' date as applicable. "
            "If the date is strictly before December 1, 2025, pass."
        ),
        extra_prerequisites=[ref_url_node]
    )

    fee_waiver_leaf = evaluator.add_leaf(
        id=f"{group_id}_Fee_Waiver",
        desc=f"The ETF offers a fee waiver that remains active through at least {FEE_WAIVER_DEADLINE_DATE_TEXT}",
        parent=etf_node,
        critical=False
    )
    await evaluator.verify(
        claim=f"The ETF has an active fee waiver with an expiration date on or after {FEE_WAIVER_DEADLINE_DATE_TEXT}.",
        node=fee_waiver_leaf,
        sources=etf.reference_url,
        additional_instruction=(
            "Check waiver disclosures (prospectus, supplements, footnotes) for waiver end dates. "
            "Pass only if the waiver remains active through at least May 31, 2026."
        ),
        extra_prerequisites=[ref_url_node]
    )

    return {
        "fee_waiver_leaf": fee_waiver_leaf,
        "issuer": (etf.issuer or "").strip()
    }


# --------------------------------------------------------------------------- #
# Portfolio-level constraints                                                 #
# --------------------------------------------------------------------------- #
def compute_min_three_fee_waivers(evaluator: Evaluator, parent_node, wafer_leaves: List[Any]) -> None:
    """Add a critical custom node ensuring at least three fee waivers passed."""
    passed_count = sum(1 for leaf in wafer_leaves if getattr(leaf, "status", "") == "passed")
    evaluator.add_custom_node(
        result=(passed_count >= 3),
        id="Min_Three_Fee_Waivers",
        desc=f"At least three of the four selected ETFs offer fee waivers active through {FEE_WAIVER_DEADLINE_DATE_TEXT}",
        parent=parent_node,
        critical=True
    )


def compute_min_three_issuers(evaluator: Evaluator, parent_node, issuers: List[str]) -> None:
    """Add a critical custom node ensuring at least three distinct issuers."""
    normalized = [i.lower().strip() for i in issuers if i and i.strip()]
    distinct_count = len(set(normalized))
    evaluator.add_custom_node(
        result=(distinct_count >= 3),
        id="Min_Three_Issuers",
        desc="The four selected ETFs represent at least three different issuers/sponsors",
        parent=parent_node,
        critical=True
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
    Evaluate the answer for the low-cost diversified cryptocurrency ETF portfolio task.
    """
    # Initialize evaluator with root parallel strategy
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

    # Extract ETF list from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_etfs(),
        template_class=ETFExtraction,
        extraction_name="etf_extraction"
    )

    # Select exactly one ETF per required asset category
    etf_xrp = select_one_by_asset(extracted.etfs, "XRP")
    etf_sol = select_one_by_asset(extracted.etfs, "SOL")
    etf_btc = select_one_by_asset(extracted.etfs, "BTC")
    etf_eth = select_one_by_asset(extracted.etfs, "ETH")

    # Build ETF group nodes under root
    xrp_info = await verify_single_etf(
        evaluator=evaluator,
        parent_node=root,
        etf=etf_xrp,
        group_id="ETF_1_XRP",
        required_asset="XRP",
        max_expense_pct=EXPENSE_THRESHOLDS["XRP"]
    )
    sol_info = await verify_single_etf(
        evaluator=evaluator,
        parent_node=root,
        etf=etf_sol,
        group_id="ETF_2_Solana",
        required_asset="SOL",
        max_expense_pct=EXPENSE_THRESHOLDS["SOL"]
    )
    btc_info = await verify_single_etf(
        evaluator=evaluator,
        parent_node=root,
        etf=etf_btc,
        group_id="ETF_3_Bitcoin",
        required_asset="BTC",
        max_expense_pct=EXPENSE_THRESHOLDS["BTC"]
    )
    eth_info = await verify_single_etf(
        evaluator=evaluator,
        parent_node=root,
        etf=etf_eth,
        group_id="ETF_4_Ethereum",
        required_asset="ETH",
        max_expense_pct=EXPENSE_THRESHOLDS["ETH"]
    )

    # Portfolio-level constraints (critical)
    portfolio_node = evaluator.add_parallel(
        id="Portfolio_Constraints",
        desc="Verify that the portfolio as a whole meets cross-ETF requirements",
        parent=root,
        critical=True
    )

    # Constraint 1: At least three fee waivers active through May 31, 2026
    fee_waiver_leaves = [
        xrp_info["fee_waiver_leaf"],
        sol_info["fee_waiver_leaf"],
        btc_info["fee_waiver_leaf"],
        eth_info["fee_waiver_leaf"],
    ]
    compute_min_three_fee_waivers(evaluator, portfolio_node, fee_waiver_leaves)

    # Constraint 2: At least three different issuers
    issuer_list = [
        xrp_info["issuer"],
        sol_info["issuer"],
        btc_info["issuer"],
        eth_info["issuer"],
    ]
    compute_min_three_issuers(evaluator, portfolio_node, issuer_list)

    # Add custom info about thresholds for transparency
    evaluator.add_custom_info(
        {
            "expense_ratio_thresholds": EXPENSE_THRESHOLDS,
            "launch_deadline": LAUNCH_DEADLINE_DATE_TEXT,
            "fee_waiver_deadline": FEE_WAIVER_DEADLINE_DATE_TEXT
        },
        info_type="constraints",
        info_name="portfolio_requirements"
    )

    # Return structured result summary
    return evaluator.get_summary()