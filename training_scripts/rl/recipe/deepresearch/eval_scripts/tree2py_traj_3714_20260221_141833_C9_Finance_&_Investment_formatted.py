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
TASK_ID = "crypto_etf_portfolio_v1"
TASK_DESCRIPTION = (
    "An institutional investment committee is seeking to establish diversified exposure to the cryptocurrency market "
    "through exchange-traded funds (ETFs). They require a portfolio of four distinct spot cryptocurrency ETFs that collectively "
    "meet the following institutional investment criteria: (1) Asset Class Coverage: Each ETF must provide exposure to a "
    "cryptocurrency that ranks within the top 5 digital assets by current market capitalization. (2) Cost Efficiency: Each ETF "
    "must have a standard (non-promotional) expense ratio at or below 0.35% to ensure long-term cost competitiveness for "
    "institutional-scale holdings. (3) Yield Optimization: For cryptocurrencies that support staking mechanisms (such as Ethereum "
    "or Solana), the selected ETF must implement staking functionality to maximize total returns. For cryptocurrencies that do not "
    "support staking (such as Bitcoin), this requirement does not apply. (4) Regulatory Compliance: Each ETF must be a spot ETF "
    "(directly holding the underlying cryptocurrency rather than futures contracts) and must be listed on a major U.S. exchange "
    "(NYSE Arca, Nasdaq, or Cboe). (5) Recent Launch: Each ETF must have launched after October 1, 2025, to reflect the most "
    "current regulatory environment following the SEC's September 2025 approval of generic listing standards for cryptocurrency ETFs. "
    "(6) Institutional Liquidity: Each ETF must have achieved at least $100 million in assets under management (AUM) to demonstrate "
    "sufficient market adoption, trading liquidity, and operational scale for institutional investors. (7) Operational Transparency: "
    "For each ETF, identify the digital asset custodian(s) used to secure the cryptocurrency holdings, as custody arrangements are a "
    "material operational and security consideration. For each of the four qualifying ETFs, provide: (a) the official ETF name and "
    "ticker symbol, (b) the issuer/sponsor, (c) the launch date, (d) the primary listing exchange, (e) the underlying cryptocurrency, "
    "(f) verification that the cryptocurrency ranks in the top 5 by market cap, (g) the standard expense ratio, (h) staking "
    "implementation details (if applicable to the cryptocurrency), (i) the digital asset custodian(s), (j) current AUM, and (k) "
    "reference URLs from official issuer websites or reputable financial data providers confirming each key detail."
)


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class ETFItem(BaseModel):
    # Core identification
    name: Optional[str] = None
    ticker: Optional[str] = None
    issuer: Optional[str] = None
    launch_date: Optional[str] = None
    exchange: Optional[str] = None

    # Asset class and structure
    cryptocurrency: Optional[str] = None

    # Cost structure
    expense_ratio: Optional[str] = None
    fee_waiver: Optional[str] = None  # terms and expiration if disclosed

    # Yield / staking
    staking_offered: Optional[str] = None  # "yes"/"no" or descriptive
    staking_percentage: Optional[str] = None
    staking_rewards_rate: Optional[str] = None

    # Regulatory & operational
    custodians: List[str] = Field(default_factory=list)

    # Liquidity
    aum: Optional[str] = None

    # References by category (URLs explicitly present in the answer)
    id_urls: List[str] = Field(default_factory=list)          # issuer page or major fin data provider for identity
    asset_urls: List[str] = Field(default_factory=list)       # market cap ranking + spot structure
    cost_urls: List[str] = Field(default_factory=list)        # expense ratio / prospectus / fact sheet
    staking_urls: List[str] = Field(default_factory=list)     # staking implementation & metrics
    regulatory_urls: List[str] = Field(default_factory=list)  # SEC approval + custodian info
    liquidity_urls: List[str] = Field(default_factory=list)   # AUM figures


class ETFPortfolioExtraction(BaseModel):
    etfs: List[ETFItem] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_etf_portfolio() -> str:
    return """
    Extract details for up to four spot cryptocurrency ETFs described in the answer. For each ETF, extract:

    1) name: Official ETF name (string)
    2) ticker: Ticker symbol (string)
    3) issuer: Issuer/Sponsor (string)
    4) launch_date: Launch or inception date (string as presented)
    5) exchange: Primary listing exchange (e.g., NYSE Arca, Nasdaq, Cboe) (string)
    6) cryptocurrency: Underlying cryptocurrency (string, e.g., Bitcoin, Ethereum, Solana, XRP)
    7) expense_ratio: Standard (non-promotional) expense ratio (string as presented, e.g., "0.25%" or "25 bps")
    8) fee_waiver: If a promotional fee waiver exists, disclose its terms and expiration (string; otherwise null)
    9) staking_offered: If staking is implemented by the ETF, indicate "yes" or "no" (or a short description)
    10) staking_percentage: Percentage of holdings staked if disclosed (string; otherwise null)
    11) staking_rewards_rate: Staking rewards rate if disclosed (string; otherwise null)
    12) custodians: List the digital asset custodian(s) (array of strings)
    13) aum: Current assets under management (AUM) as presented (string)

    Also extract categorized reference URL lists explicitly present in the answer text:
    - id_urls: URLs confirming name, ticker, issuer, launch date, exchange (issuer website or reputable data provider)
    - asset_urls: URLs confirming underlying crypto top-5 market cap ranking AND spot ETF structure
    - cost_urls: URLs confirming standard expense ratio (prospectus, fact sheet, issuer page)
    - staking_urls: URLs confirming staking implementation and any related metrics
    - regulatory_urls: URLs confirming SEC approval/trading status and custodian information
    - liquidity_urls: URLs confirming current AUM

    Rules:
    - Only extract information explicitly present in the answer. Do not invent or infer data.
    - For URLs, only extract actual URLs that appear in the answer (including markdown links). If not present, leave the list empty.
    - If any field is missing, set it to null (for strings) or [] (for lists).
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def _safe_list(x: Optional[List[str]]) -> List[str]:
    return x if isinstance(x, list) else []


def _supports_staking(crypto_name: Optional[str]) -> bool:
    if not crypto_name:
        return False
    name = crypto_name.strip().lower()
    # Common staking-enabled networks (non-exhaustive but covers major ones)
    staking_names = {
        "ethereum", "eth", "solana", "sol", "cardano", "ada", "polkadot", "dot",
        "avalanche", "avax", "tron", "trx", "cosmos", "atom", "near"
    }
    # Common non-staking (PoW or otherwise)
    non_staking = {"bitcoin", "btc", "xrp", "dogecoin", "doge", "litecoin", "ltc"}
    if name in staking_names:
        return True
    if name in non_staking:
        return False
    # Default conservative: treat unknown as non-staking unless explicitly known
    return False


# --------------------------------------------------------------------------- #
# Verification for a single ETF subtree                                       #
# --------------------------------------------------------------------------- #
async def verify_single_etf(
    evaluator: Evaluator,
    etf_parent_node,
    etf: ETFItem,
    etf_index: int
) -> None:
    etf_node = evaluator.add_parallel(
        id=f"ETF_{etf_index+1}",
        desc=f"{['First','Second','Third','Fourth'][etf_index]} qualifying cryptocurrency ETF",
        parent=etf_parent_node,
        critical=False  # Non-critical at ETF level; inner groups handle criticality
    )

    # ---------------- Identification (Critical, Parallel) ----------------
    id_node = evaluator.add_parallel(
        id=f"ETF_{etf_index+1}_Identification",
        desc="ETF identification information",
        parent=etf_node,
        critical=True
    )

    id_urls = _safe_list(etf.id_urls)

    # Name & Ticker
    name_ticker_leaf = evaluator.add_leaf(
        id=f"ETF_{etf_index+1}_Name_Ticker",
        desc="Provide the official ETF name and ticker symbol",
        parent=id_node,
        critical=True
    )
    claim_name_ticker = f"The ETF is officially named '{etf.name}' and its ticker symbol is '{etf.ticker}'."
    await evaluator.verify(
        claim=claim_name_ticker,
        node=name_ticker_leaf,
        sources=id_urls,
        additional_instruction=(
            "Confirm on issuer or reputable data provider pages. Allow minor naming suffix variants "
            "like 'Trust' vs 'ETF Trust' if clearly the same product."
        )
    )

    # Issuer
    issuer_leaf = evaluator.add_leaf(
        id=f"ETF_{etf_index+1}_Issuer",
        desc="Provide the ETF issuer/sponsor name",
        parent=id_node,
        critical=True
    )
    claim_issuer = f"The ETF's issuer/sponsor is '{etf.issuer}'."
    await evaluator.verify(
        claim=claim_issuer,
        node=issuer_leaf,
        sources=id_urls,
        additional_instruction="Confirm the sponsor/issuer is correctly cited."
    )

    # Launch Date (includes threshold after 2025-10-01)
    launch_leaf = evaluator.add_leaf(
        id=f"ETF_{etf_index+1}_Launch_Date",
        desc="Provide the ETF launch date (must be after October 1, 2025)",
        parent=id_node,
        critical=True
    )
    claim_launch = (
        f"The ETF launched on {etf.launch_date}, and this launch date is after October 1, 2025."
    )
    await evaluator.verify(
        claim=claim_launch,
        node=launch_leaf,
        sources=id_urls,
        additional_instruction=(
            "Verify the stated launch/inception date and ensure it is after 2025-10-01."
        )
    )

    # Exchange
    exchange_leaf = evaluator.add_leaf(
        id=f"ETF_{etf_index+1}_Exchange",
        desc="Provide the primary listing exchange (must be NYSE Arca, Nasdaq, or Cboe)",
        parent=id_node,
        critical=True
    )
    claim_exchange = (
        f"The ETF's primary listing exchange is {etf.exchange}, which is one of NYSE Arca, Nasdaq, or Cboe."
    )
    await evaluator.verify(
        claim=claim_exchange,
        node=exchange_leaf,
        sources=id_urls,
        additional_instruction=(
            "Accept exchange names like 'Cboe BZX' (Cboe), 'NYSE Arca', or 'Nasdaq'."
        )
    )

    # Identification Reference URL(s)
    id_ref_leaf = evaluator.add_leaf(
        id=f"ETF_{etf_index+1}_Reference_URL",
        desc="Provide a reference URL from the issuer's official website or major financial data provider confirming the ETF details",
        parent=id_node,
        critical=True
    )
    claim_id_ref = (
        "At least one of the provided identification references is an official issuer page or a reputable "
        "financial data provider that confirms the ETF's name, ticker, issuer, launch date, and exchange."
    )
    await evaluator.verify(
        claim=claim_id_ref,
        node=id_ref_leaf,
        sources=id_urls,
        additional_instruction=(
            "Issuer pages (e.g., prospectus/fact sheet) or major providers (e.g., Nasdaq listings, NYSE Arca, Cboe, or well-known "
            "fund databases) qualify."
        )
    )

    # ---------------- Asset Class (Critical, Parallel) -------------------
    asset_node = evaluator.add_parallel(
        id=f"ETF_{etf_index+1}_Asset_Class",
        desc="Underlying cryptocurrency asset verification",
        parent=etf_node,
        critical=True
    )

    asset_urls = _safe_list(etf.asset_urls)
    # Cryptocurrency
    crypto_leaf = evaluator.add_leaf(
        id=f"ETF_{etf_index+1}_Cryptocurrency",
        desc="Identify the specific cryptocurrency (must be Bitcoin, Ethereum, XRP, Solana, or another top-5 digital asset by market cap)",
        parent=asset_node,
        critical=True
    )
    claim_crypto = f"The ETF provides exposure to the cryptocurrency '{etf.cryptocurrency}'."
    await evaluator.verify(
        claim=claim_crypto,
        node=crypto_leaf,
        sources=(asset_urls or id_urls),
        additional_instruction="Confirm that the ETF directly holds the stated cryptocurrency."
    )

    # Market cap ranking (Top 5)
    ranking_leaf = evaluator.add_leaf(
        id=f"ETF_{etf_index+1}_Market_Cap_Ranking",
        desc="Verify the cryptocurrency ranks in the top 5 by market capitalization",
        parent=asset_node,
        critical=True
    )
    claim_ranking = (
        f"The cryptocurrency '{etf.cryptocurrency}' ranks within the top 5 digital assets by market capitalization."
    )
    await evaluator.verify(
        claim=claim_ranking,
        node=ranking_leaf,
        sources=asset_urls,
        additional_instruction=(
            "Use the provided market data source (e.g., CoinMarketCap, CoinGecko, Bloomberg) to confirm top-5 status."
        )
    )

    # Spot ETF confirmation
    spot_leaf = evaluator.add_leaf(
        id=f"ETF_{etf_index+1}_Spot_Type",
        desc="Confirm the ETF is a spot ETF (directly holds the cryptocurrency, not futures contracts)",
        parent=asset_node,
        critical=True
    )
    claim_spot = (
        f"The ETF is a spot ETF that directly holds {etf.cryptocurrency}, not futures contracts."
    )
    await evaluator.verify(
        claim=claim_spot,
        node=spot_leaf,
        sources=(asset_urls or id_urls or _safe_list(etf.regulatory_urls)),
        additional_instruction=(
            "Look for wording like 'spot', 'physically backed', 'directly holds', 'in-kind creations', "
            "or portfolio holdings referencing the underlying cryptocurrency."
        )
    )

    # Asset reference URL(s)
    asset_ref_leaf = evaluator.add_leaf(
        id=f"ETF_{etf_index+1}_Asset_Reference_URL",
        desc="Provide a reference URL confirming the cryptocurrency's market cap ranking and the ETF's spot structure",
        parent=asset_node,
        critical=True
    )
    claim_asset_ref = (
        "The provided asset references explicitly support the cryptocurrency's top-5 market cap status and the ETF's spot structure."
    )
    await evaluator.verify(
        claim=claim_asset_ref,
        node=asset_ref_leaf,
        sources=asset_urls,
        additional_instruction="At least one reference should clearly support each of the two points."
    )

    # ---------------- Cost Structure (Non-Critical, Parallel) ------------
    # Note: Parent set non-critical to allow a non-critical Fee Waiver child.
    cost_node = evaluator.add_parallel(
        id=f"ETF_{etf_index+1}_Cost_Structure",
        desc="ETF expense ratio and fee analysis",
        parent=etf_node,
        critical=False
    )

    cost_urls = _safe_list(etf.cost_urls)

    # Expense ratio (value)
    expense_leaf = evaluator.add_leaf(
        id=f"ETF_{etf_index+1}_Expense_Ratio",
        desc="Provide the standard (non-promotional) expense ratio",
        parent=cost_node,
        critical=True
    )
    claim_expense = f"The ETF's standard (non-promotional) expense ratio is {etf.expense_ratio}."
    await evaluator.verify(
        claim=claim_expense,
        node=expense_leaf,
        sources=cost_urls,
        additional_instruction=(
            "Verify the standard net expense ratio (not a temporary or promotional waiver rate)."
        )
    )

    # Expense threshold <= 0.35%
    expense_threshold_leaf = evaluator.add_leaf(
        id=f"ETF_{etf_index+1}_Expense_Threshold",
        desc="Verify the standard expense ratio is at or below 0.35%",
        parent=cost_node,
        critical=True
    )
    claim_expense_thresh = "The ETF's standard (non-promotional) expense ratio is at or below 0.35%."
    await evaluator.verify(
        claim=claim_expense_thresh,
        node=expense_threshold_leaf,
        sources=cost_urls,
        additional_instruction=(
            "Ignore temporary promotional waivers; judge the standard expense ratio only."
        )
    )

    # Fee waiver disclosure (Non-critical)
    if etf.fee_waiver and etf.fee_waiver.strip():
        fee_waiver_leaf = evaluator.add_leaf(
            id=f"ETF_{etf_index+1}_Fee_Waiver_Disclosure",
            desc="If a promotional fee waiver exists, disclose its terms and expiration date",
            parent=cost_node,
            critical=False
        )
        claim_fee_waiver = f"A promotional fee waiver is disclosed with terms: {etf.fee_waiver}."
        await evaluator.verify(
            claim=claim_fee_waiver,
            node=fee_waiver_leaf,
            sources=cost_urls,
            additional_instruction="Confirm stated waiver terms and any expiration."
        )
    else:
        evaluator.add_custom_node(
            result=True,
            id=f"ETF_{etf_index+1}_Fee_Waiver_Disclosure",
            desc="No promotional fee waiver exists or is claimed; disclosure requirement not applicable",
            parent=cost_node,
            critical=False
        )

    # Cost reference URL(s)
    cost_ref_leaf = evaluator.add_leaf(
        id=f"ETF_{etf_index+1}_Cost_Reference_URL",
        desc="Provide a reference URL from the ETF prospectus or fact sheet confirming the expense ratio",
        parent=cost_node,
        critical=True
    )
    claim_cost_ref = "The provided cost references confirm the ETF's standard expense ratio."
    await evaluator.verify(
        claim=claim_cost_ref,
        node=cost_ref_leaf,
        sources=cost_urls,
        additional_instruction="Issuer prospectus/fact sheet preferred; reputable providers acceptable."
    )

    # ---------------- Yield Features (Non-Critical, Parallel) ------------
    # We treat the detailed staking disclosures as non-critical informational metrics.
    yield_node = evaluator.add_parallel(
        id=f"ETF_{etf_index+1}_Yield_Features",
        desc="Staking and yield generation capabilities",
        parent=etf_node,
        critical=False
    )

    staking_applicable = _supports_staking(etf.cryptocurrency)

    # Staking applicability informational node
    evaluator.add_custom_node(
        result=True,
        id=f"ETF_{etf_index+1}_Staking_Applicability",
        desc=f"Determine if the underlying cryptocurrency supports staking: {staking_applicable}",
        parent=yield_node,
        critical=False
    )

    # Staking implementation group (Non-critical, Parallel)
    staking_impl_node = evaluator.add_parallel(
        id=f"ETF_{etf_index+1}_Staking_Implementation",
        desc="If staking is applicable, verify the ETF implements staking functionality; if not applicable, automatically satisfied",
        parent=yield_node,
        critical=False
    )

    staking_urls = _safe_list(etf.staking_urls)
    reg_urls = _safe_list(etf.regulatory_urls)

    if staking_applicable:
        # Confirmation (Critical under this sub-group, but parent is non-critical overall)
        staking_confirm_leaf = evaluator.add_leaf(
            id=f"ETF_{etf_index+1}_Staking_Confirmation",
            desc="Confirm the ETF offers staking (if applicable to the cryptocurrency)",
            parent=staking_impl_node,
            critical=True
        )
        claim_staking_confirm = (
            f"The ETF implements staking for its {etf.cryptocurrency} holdings."
        )
        await evaluator.verify(
            claim=claim_staking_confirm,
            node=staking_confirm_leaf,
            sources=(staking_urls or reg_urls or id_urls),
            additional_instruction="Look for explicit mention of staking in prospectus, fact sheet, or issuer updates."
        )

        # Percentage (Non-critical)
        if etf.staking_percentage and etf.staking_percentage.strip():
            staking_pct_leaf = evaluator.add_leaf(
                id=f"ETF_{etf_index+1}_Staking_Percentage",
                desc="Provide the percentage of holdings that are staked (if staking is offered)",
                parent=staking_impl_node,
                critical=False
            )
            claim_pct = f"The percentage of holdings that are staked is {etf.staking_percentage}."
            await evaluator.verify(
                claim=claim_pct,
                node=staking_pct_leaf,
                sources=staking_urls,
                additional_instruction="Confirm if a percentage disclosure is provided."
            )
        else:
            evaluator.add_custom_node(
                result=True,
                id=f"ETF_{etf_index+1}_Staking_Percentage",
                desc="Percentage of holdings staked is not disclosed; non-critical",
                parent=staking_impl_node,
                critical=False
            )

        # Rewards rate (Non-critical)
        if etf.staking_rewards_rate and etf.staking_rewards_rate.strip():
            staking_rate_leaf = evaluator.add_leaf(
                id=f"ETF_{etf_index+1}_Staking_Rewards_Rate",
                desc="Provide the gross or net staking rewards rate (if disclosed)",
                parent=staking_impl_node,
                critical=False
            )
            claim_rate = f"The staking rewards rate is {etf.staking_rewards_rate}."
            await evaluator.verify(
                claim=claim_rate,
                node=staking_rate_leaf,
                sources=staking_urls,
                additional_instruction="Confirm any disclosed staking yield metrics."
            )
        else:
            evaluator.add_custom_node(
                result=True,
                id=f"ETF_{etf_index+1}_Staking_Rewards_Rate",
                desc="Staking rewards rate not disclosed; non-critical",
                parent=staking_impl_node,
                critical=False
            )

        # Staking reference URL(s) (Critical under this sub-group)
        staking_ref_leaf = evaluator.add_leaf(
            id=f"ETF_{etf_index+1}_Staking_Reference_URL",
            desc="Provide a reference URL confirming staking implementation and related metrics",
            parent=staking_impl_node,
            critical=True
        )
        claim_staking_ref = (
            "The provided staking references confirm that the ETF implements staking and any related metrics disclosed."
        )
        await evaluator.verify(
            claim=claim_staking_ref,
            node=staking_ref_leaf,
            sources=staking_urls,
            additional_instruction="Issuer pages preferred; reputable data providers acceptable."
        )
    else:
        # Not applicable: auto-satisfy by custom nodes
        evaluator.add_custom_node(
            result=True,
            id=f"ETF_{etf_index+1}_Staking_Confirmation",
            desc=f"Staking not applicable for {etf.cryptocurrency}; criterion satisfied by definition",
            parent=staking_impl_node,
            critical=False
        )
        evaluator.add_custom_node(
            result=True,
            id=f"ETF_{etf_index+1}_Staking_Reference_URL",
            desc=f"No staking references required as staking is not applicable for {etf.cryptocurrency}",
            parent=staking_impl_node,
            critical=False
        )
        evaluator.add_custom_node(
            result=True,
            id=f"ETF_{etf_index+1}_Staking_Percentage",
            desc="Not applicable (no staking)",
            parent=staking_impl_node,
            critical=False
        )
        evaluator.add_custom_node(
            result=True,
            id=f"ETF_{etf_index+1}_Staking_Rewards_Rate",
            desc="Not applicable (no staking)",
            parent=staking_impl_node,
            critical=False
        )

    # ---------------- Regulatory Compliance (Critical, Parallel) ---------
    reg_node = evaluator.add_parallel(
        id=f"ETF_{etf_index+1}_Regulatory_Compliance",
        desc="Regulatory structure and operational requirements",
        parent=etf_node,
        critical=True
    )

    # SEC approval and active trading
    sec_leaf = evaluator.add_leaf(
        id=f"ETF_{etf_index+1}_SEC_Approval",
        desc="Verify the ETF has SEC approval and is actively trading",
        parent=reg_node,
        critical=True
    )
    claim_sec = (
        f"The ETF is SEC-approved and is actively trading on {etf.exchange}."
    )
    await evaluator.verify(
        claim=claim_sec,
        node=sec_leaf,
        sources=(reg_urls or id_urls),
        additional_instruction="Evidence can include issuer announcements, SEC filings, or listing exchange pages."
    )

    # Custodian(s)
    custodian_leaf = evaluator.add_leaf(
        id=f"ETF_{etf_index+1}_Custodian",
        desc="Identify the digital asset custodian(s) used by the ETF",
        parent=reg_node,
        critical=True
    )
    custodian_list_str = ", ".join(etf.custodians) if etf.custodians else ""
    claim_custodian = f"The ETF uses the following digital asset custodian(s): {custodian_list_str}."
    await evaluator.verify(
        claim=claim_custodian,
        node=custodian_leaf,
        sources=reg_urls,
        additional_instruction="Accept reasonable naming variants (e.g., 'Coinbase Custody' vs 'Coinbase Custody Trust Company')."
    )

    # Regulatory reference URL(s)
    reg_ref_leaf = evaluator.add_leaf(
        id=f"ETF_{etf_index+1}_Regulatory_Reference_URL",
        desc="Provide a reference URL confirming regulatory approval status and custodian information",
        parent=reg_node,
        critical=True
    )
    claim_reg_ref = "The provided references confirm the ETF's regulatory approval/trading status and its custodian(s)."
    await evaluator.verify(
        claim=claim_reg_ref,
        node=reg_ref_leaf,
        sources=reg_urls,
        additional_instruction="Issuer, SEC, or listing exchange pages preferred; reputable providers acceptable."
    )

    # ---------------- Liquidity (Critical, Parallel) ---------------------
    liq_node = evaluator.add_parallel(
        id=f"ETF_{etf_index+1}_Liquidity",
        desc="Market adoption and liquidity metrics",
        parent=etf_node,
        critical=True
    )

    liquidity_urls = _safe_list(etf.liquidity_urls)

    # AUM value
    aum_leaf = evaluator.add_leaf(
        id=f"ETF_{etf_index+1}_AUM",
        desc="Provide the current assets under management (AUM)",
        parent=liq_node,
        critical=True
    )
    claim_aum = f"The ETF's current assets under management (AUM) is {etf.aum}."
    await evaluator.verify(
        claim=claim_aum,
        node=aum_leaf,
        sources=liquidity_urls,
        additional_instruction="Confirm the AUM figure from issuer or reputable data provider."
    )

    # AUM threshold >= $100M
    aum_thresh_leaf = evaluator.add_leaf(
        id=f"ETF_{etf_index+1}_AUM_Threshold",
        desc="Verify AUM is at least $100 million",
        parent=liq_node,
        critical=True
    )
    claim_aum_thresh = "The ETF's current AUM is at least $100 million."
    await evaluator.verify(
        claim=claim_aum_thresh,
        node=aum_thresh_leaf,
        sources=liquidity_urls,
        additional_instruction="Accept approximate statements indicating AUM exceeds $100 million."
    )

    # Liquidity reference URL(s)
    liq_ref_leaf = evaluator.add_leaf(
        id=f"ETF_{etf_index+1}_Liquidity_Reference_URL",
        desc="Provide a reference URL confirming current AUM figures",
        parent=liq_node,
        critical=True
    )
    claim_liq_ref = "The provided references confirm the ETF's current AUM."
    await evaluator.verify(
        claim=claim_liq_ref,
        node=liq_ref_leaf,
        sources=liquidity_urls,
        additional_instruction="Issuer or reputable data provider pages acceptable."
    )


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
    model: str = "o4-mini"
) -> Dict:
    # Initialize evaluator (Root as non-critical parallel to avoid critical-child constraint conflicts)
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

    # Extract portfolio info (up to 4 ETFs)
    extraction = await evaluator.extract(
        prompt=prompt_extract_etf_portfolio(),
        template_class=ETFPortfolioExtraction,
        extraction_name="etf_portfolio_extraction"
    )

    # Ensure exactly four items (pad with empty ones if needed)
    etfs: List[ETFItem] = list(extraction.etfs[:4])
    while len(etfs) < 4:
        etfs.append(ETFItem())

    # Add a small non-critical portfolio info node for uniqueness/overview (optional)
    portfolio_info_node = evaluator.add_parallel(
        id="Portfolio_Info",
        desc="Portfolio-level informational checks",
        parent=root,
        critical=False
    )
    extracted_tickers = [e.ticker for e in etfs if e.ticker]
    extracted_cryptos = [e.cryptocurrency for e in etfs if e.cryptocurrency]
    # Distinct tickers informational node
    evaluator.add_custom_node(
        result=len(set(extracted_tickers)) == len(extracted_tickers) if extracted_tickers else False,
        id="Distinct_Tickers",
        desc=f"Distinct ETF tickers across portfolio: {extracted_tickers}",
        parent=portfolio_info_node,
        critical=False
    )
    # Record extracted overview as custom info
    evaluator.add_custom_info(
        info={
            "tickers": extracted_tickers,
            "cryptocurrencies": extracted_cryptos
        },
        info_type="portfolio_overview"
    )

    # Build four ETF subtrees in parallel
    for i in range(4):
        await verify_single_etf(evaluator, root, etfs[i], i)

    return evaluator.get_summary()