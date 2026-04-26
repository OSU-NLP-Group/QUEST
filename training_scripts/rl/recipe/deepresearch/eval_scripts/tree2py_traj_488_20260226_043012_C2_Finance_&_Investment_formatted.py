import asyncio
import logging
from typing import Optional, List, Dict, Any
from urllib.parse import urlparse

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.llm_client.base_client import LLMClient
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy, VerificationNode

# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "sol_spot_etf_zero_fee_nov2025"
TASK_DESCRIPTION = (
    "An investor is researching spot Solana ETFs that launched in the United States during November 2025 and wants to take "
    "advantage of promotional zero-fee periods before they expire. Identify the spot Solana ETF that meets these criteria and "
    "has its zero-fee promotional period expiring in February 2026. Provide the following information: the ETF's ticker symbol, "
    "its exact launch date, the U.S. exchange where it trades, and the exact expiration date of the zero-fee promotional period."
)

# Domains considered "official" for ETF verification (issuers, exchanges, SEC, etc.)
OFFICIAL_DOMAIN_KEYWORDS = [
    "sec.gov",
    "cboe",
    "nyse",
    "nasdaq",
    "vaneck",
    "fidelity",
    "invesco",
    "blackrock",
    "ishares",
    "franklintempleton",
    "wisdomtree",
    "globalx",
    "bitwise",
    "ark",
    "hashdex",
    "21shares",
    "proshares",
    "grayscale",
    "coinbase",  # sometimes custody or partner announcements
]


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class ETFDetails(BaseModel):
    """Structured details for the selected ETF"""
    name: Optional[str] = None
    ticker: Optional[str] = None
    exchange: Optional[str] = None  # e.g., "NYSE Arca", "Nasdaq", "Cboe BZX"
    launch_date: Optional[str] = None  # exact date as in answer text, e.g., "November 15, 2025"
    etf_structure: Optional[str] = None  # expected values: "spot", "futures", or null
    underlying_asset: Optional[str] = None  # expected: "Solana"
    zero_fee_promo: Optional[bool] = None  # whether a zero-fee promotional period is offered
    fee_waiver_expiration_date: Optional[str] = None  # exact expiration date string, e.g., "February 28, 2026"
    issuer: Optional[str] = None  # ETF sponsor/issuer if present
    sources: List[str] = Field(default_factory=list)  # list of URLs cited for this ETF


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_etf_details() -> str:
    return (
        "From the answer, select the single ETF that best matches all of the following criteria:\n"
        "- It is a spot Solana ETF (not futures-based).\n"
        "- It launched (began trading/listed) on a U.S. exchange in November 2025.\n"
        "- It offers a promotional zero-fee period (waived sponsor/management/advisor fees) that expires in February 2026.\n"
        "Extract the following fields for that ETF exactly as stated in the answer:\n"
        "1. name: The ETF's name (if provided; null if not).\n"
        "2. ticker: The ETF's ticker symbol.\n"
        "3. exchange: The name of the U.S. exchange where it trades (e.g., NYSE Arca, Nasdaq, Cboe BZX).\n"
        "4. launch_date: The ETF's exact launch date (listing/trading start date), as a textual date string.\n"
        "5. etf_structure: The ETF structure ('spot' or 'futures'); use 'spot' for spot-based ETFs.\n"
        "6. underlying_asset: The underlying crypto asset (should be 'Solana').\n"
        "7. zero_fee_promo: Whether a zero-fee promotional period exists (true/false).\n"
        "8. fee_waiver_expiration_date: The exact expiration date of the zero-fee period, as a textual date string.\n"
        "9. issuer: The ETF issuer/sponsor if mentioned (null if not).\n"
        "10. sources: An array of URLs cited in the answer that support the ETF's details.\n"
        "Rules:\n"
        "- Do not invent or infer information. Only extract what the answer explicitly states.\n"
        "- If multiple ETFs are mentioned, choose the one that fully meets the criteria. If none fully meet, choose the best candidate and extract its details exactly; fields not stated should be null.\n"
        "- Only include URLs actually present in the answer in 'sources'.\n"
    )


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def has_official_source(urls: List[str]) -> bool:
    """Return True if at least one URL appears to be from an 'official' domain."""
    for u in urls:
        try:
            hostname = urlparse(u).hostname or ""
        except Exception:
            hostname = ""
        host_lower = hostname.lower()
        if any(key in host_lower for key in OFFICIAL_DOMAIN_KEYWORDS):
            return True
    return False


# --------------------------------------------------------------------------- #
# Verification sub-tree builder                                               #
# --------------------------------------------------------------------------- #
async def build_verification_tree(evaluator: Evaluator, parent: VerificationNode, etf: ETFDetails) -> None:
    """
    Build and execute the verification tree for the selected ETF, following the rubric.
    All verification leaves are binary and will be evaluated with proper source grounding whenever possible.
    """
    # Top-level critical sequential node per rubric
    main = evaluator.add_sequential(
        id="Solana_ETF_Investment_Research",
        desc="Identify and provide complete information about a spot Solana ETF meeting specific launch timing and fee waiver criteria",
        parent=parent,
        critical=True
    )

    # 1) ETF Identification Criteria (critical sequential)
    ident = evaluator.add_sequential(
        id="ETF_Identification_Criteria",
        desc="The identified ETF must meet all specified selection criteria",
        parent=main,
        critical=True
    )

    # 1.1) Asset Type and Launch Timing (critical parallel)
    asset_launch = evaluator.add_parallel(
        id="Asset_Type_and_Launch_Timing",
        desc="The ETF must be a spot Solana ETF (not futures-based) that launched on a U.S. exchange in November 2025",
        parent=ident,
        critical=True
    )

    # Existence check for minimal required fields to proceed with identification checks
    evaluator.add_custom_node(
        result=bool(etf.ticker) and bool(etf.sources),
        id="selected_etf_provided",
        desc="Selected ETF details and sources are provided in the answer",
        parent=asset_launch,
        critical=True
    )

    # Leaf: spot vs futures
    leaf_spot = evaluator.add_leaf(
        id="is_spot_structure",
        desc="ETF is a spot (not futures-based) product",
        parent=asset_launch,
        critical=True
    )
    claim_spot = "This ETF is a spot Solana ETF and is not a futures-based product."
    await evaluator.verify(
        claim=claim_spot,
        node=leaf_spot,
        sources=etf.sources,
        additional_instruction="Confirm that the product is spot-based (physically backed or spot exposure), not futures-based."
    )

    # Leaf: underlying asset Solana
    leaf_solana = evaluator.add_leaf(
        id="underlying_asset_is_solana",
        desc="ETF provides spot exposure to Solana",
        parent=asset_launch,
        critical=True
    )
    claim_solana = "This ETF provides spot exposure to the Solana (SOL) cryptocurrency."
    await evaluator.verify(
        claim=claim_solana,
        node=leaf_solana,
        sources=etf.sources,
        additional_instruction="Verify that the ETF's underlying asset is Solana (SOL)."
    )

    # Leaf: launch in November 2025
    leaf_nov_launch = evaluator.add_leaf(
        id="launched_in_november_2025",
        desc="ETF launched (began trading/listing) in November 2025",
        parent=asset_launch,
        critical=True
    )
    launch_date_text = etf.launch_date or ""
    claim_launch_month = f"The ETF launched on {launch_date_text}, and that date is in November 2025."
    await evaluator.verify(
        claim=claim_launch_month,
        node=leaf_nov_launch,
        sources=etf.sources,
        additional_instruction="Check the listing/trading start date specified on official sources to confirm it falls in November 2025."
    )

    # Leaf: trades on a U.S. exchange
    leaf_us_exch = evaluator.add_leaf(
        id="trades_on_us_exchange",
        desc="ETF trades on a U.S. exchange (e.g., NYSE Arca, Nasdaq, Cboe)",
        parent=asset_launch,
        critical=True
    )
    exch_text = etf.exchange or ""
    claim_us_exch = f"The ETF trades on the U.S. exchange '{exch_text}'."
    await evaluator.verify(
        claim=claim_us_exch,
        node=leaf_us_exch,
        sources=etf.sources,
        additional_instruction="Confirm the exchange is a U.S. venue such as NYSE Arca, Nasdaq, or Cboe."
    )

    # 1.2) Fee Structure Requirements (critical sequential)
    fee_req = evaluator.add_sequential(
        id="Fee_Structure_Requirements",
        desc="The ETF must offer a zero-fee promotional period (waived sponsor fees) that expires in February 2026",
        parent=ident,
        critical=True
    )

    # Leaf: zero fee promo present
    leaf_zero_fee = evaluator.add_leaf(
        id="has_zero_fee_promo",
        desc="ETF offers a waived sponsor/management/advisor fee (zero-fee) promotional period",
        parent=fee_req,
        critical=True
    )
    claim_zero_fee = "The ETF offers a zero-fee promotional period where sponsor/management/advisor fees are waived."
    await evaluator.verify(
        claim=claim_zero_fee,
        node=leaf_zero_fee,
        sources=etf.sources,
        additional_instruction="Verify the existence of a fee waiver or promotional zero-fee period as stated by official sources."
    )

    # Leaf: promo expiration in February 2026 with exact date
    leaf_fee_expiry = evaluator.add_leaf(
        id="zero_fee_expires_feb_2026",
        desc="ETF's zero-fee promotional period expires in February 2026 (exact date provided)",
        parent=fee_req,
        critical=True
    )
    expiry_text = etf.fee_waiver_expiration_date or ""
    claim_fee_expiry = f"The ETF's promotional zero-fee period expires on {expiry_text}, which is in February 2026."
    await evaluator.verify(
        claim=claim_fee_expiry,
        node=leaf_fee_expiry,
        sources=etf.sources,
        additional_instruction="Confirm the exact expiration date for the zero-fee period and ensure it is a date in February 2026."
    )

    # 2) Essential Information Package (critical parallel)
    info_pkg = evaluator.add_parallel(
        id="Essential_Information_Package",
        desc="All required details about the identified ETF must be provided accurately",
        parent=main,
        critical=True
    )

    # 2.1) Trading Information (critical parallel group with separate leaf checks)
    trading_info = evaluator.add_parallel(
        id="Trading_Information",
        desc="Provide the ETF's ticker symbol and the name of the U.S. exchange where it trades",
        parent=info_pkg,
        critical=True
    )

    # Existence checks for ticker/exchange
    evaluator.add_custom_node(
        result=bool(etf.ticker),
        id="ticker_provided",
        desc="ETF ticker is provided",
        parent=trading_info,
        critical=True
    )
    evaluator.add_custom_node(
        result=bool(etf.exchange),
        id="exchange_provided",
        desc="ETF exchange is provided",
        parent=trading_info,
        critical=True
    )

    # Leaf: ticker verified by sources
    leaf_ticker = evaluator.add_leaf(
        id="ticker_accurate",
        desc="ETF ticker matches what is shown on official sources",
        parent=trading_info,
        critical=True
    )
    claim_ticker = f"The ETF's ticker is '{etf.ticker or ''}'."
    await evaluator.verify(
        claim=claim_ticker,
        node=leaf_ticker,
        sources=etf.sources,
        additional_instruction="Verify that the official page lists the same ticker."
    )

    # Leaf: exchange verified by sources
    leaf_exchange = evaluator.add_leaf(
        id="exchange_accurate",
        desc="The named U.S. exchange matches what is shown on official sources",
        parent=trading_info,
        critical=True
    )
    claim_exchange = f"The ETF trades on '{etf.exchange or ''}'."
    await evaluator.verify(
        claim=claim_exchange,
        node=leaf_exchange,
        sources=etf.sources,
        additional_instruction="Verify that the official page states the exchange (e.g., NYSE Arca, Nasdaq, Cboe)."
    )

    # 2.2) Timing Information (critical parallel group)
    timing_info = evaluator.add_parallel(
        id="Timing_Information",
        desc="Provide the exact launch date and the exact expiration date of the fee waiver promotional period",
        parent=info_pkg,
        critical=True
    )

    # Existence checks for dates
    evaluator.add_custom_node(
        result=bool(etf.launch_date),
        id="launch_date_provided",
        desc="ETF launch date is provided",
        parent=timing_info,
        critical=True
    )
    evaluator.add_custom_node(
        result=bool(etf.fee_waiver_expiration_date),
        id="fee_expiration_date_provided",
        desc="Zero-fee expiration date is provided",
        parent=timing_info,
        critical=True
    )

    # Leaf: launch date verified by sources
    leaf_launch_date = evaluator.add_leaf(
        id="launch_date_accurate",
        desc="ETF launch date matches official sources",
        parent=timing_info,
        critical=True
    )
    claim_launch_date = f"The ETF launched on {etf.launch_date or ''}."
    await evaluator.verify(
        claim=claim_launch_date,
        node=leaf_launch_date,
        sources=etf.sources,
        additional_instruction="Confirm that the official page shows the same launch/listing/trading start date."
    )

    # Leaf: fee waiver expiration date verified by sources
    leaf_fee_date = evaluator.add_leaf(
        id="fee_waiver_expiration_accurate",
        desc="Zero-fee promotional period expiration date matches official sources",
        parent=timing_info,
        critical=True
    )
    claim_fee_date = f"The zero-fee promotional period expires on {etf.fee_waiver_expiration_date or ''}."
    await evaluator.verify(
        claim=claim_fee_date,
        node=leaf_fee_date,
        sources=etf.sources,
        additional_instruction="Confirm the exact fee waiver expiration date on official sources."
    )

    # 2.3) Source Verification (critical parallel group)
    source_ver = evaluator.add_parallel(
        id="Source_Verification",
        desc="Provide at least one official URL reference that verifies the ETF's details",
        parent=info_pkg,
        critical=True
    )

    # Leaf: sources provided
    evaluator.add_custom_node(
        result=bool(etf.sources),
        id="sources_provided",
        desc="At least one source URL is provided in the answer",
        parent=source_ver,
        critical=True
    )

    # Leaf: at least one official source present
    evaluator.add_custom_node(
        result=has_official_source(etf.sources),
        id="official_source_present",
        desc="At least one provided URL appears to be an official source (issuer/exchange/SEC)",
        parent=source_ver,
        critical=True
    )


# --------------------------------------------------------------------------- #
# Main evaluation function                                                    #
# --------------------------------------------------------------------------- #
async def evaluate_answer(
    client: LLMClient,
    answer: str,
    agent_name: str,
    answer_name: str,
    cache: CacheFileSys,
    semaphore: asyncio.Semaphore,
    logger: logging.Logger,
    model: str = "o4-mini"
) -> Dict[str, Any]:
    """
    Entry point for evaluating an answer against the Solana spot ETF rubric.
    """
    # Initialize evaluator with a sequential root (will host the critical main node)
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.SEQUENTIAL,
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

    # Extract selected ETF details from the answer
    etf_details = await evaluator.extract(
        prompt=prompt_extract_etf_details(),
        template_class=ETFDetails,
        extraction_name="selected_spot_solana_etf"
    )

    # Add optional custom info about extracted details for debugging/traceability
    evaluator.add_custom_info(
        info={
            "ticker": etf_details.ticker,
            "exchange": etf_details.exchange,
            "launch_date": etf_details.launch_date,
            "etf_structure": etf_details.etf_structure,
            "underlying_asset": etf_details.underlying_asset,
            "zero_fee_promo": etf_details.zero_fee_promo,
            "fee_waiver_expiration_date": etf_details.fee_waiver_expiration_date,
            "issuer": etf_details.issuer,
            "sources_count": len(etf_details.sources),
        },
        info_type="extraction_summary",
        info_name="extracted_etf_details_overview"
    )

    # Build verification tree and run checks
    await build_verification_tree(evaluator, root, etf_details)

    # Return unified summary including verification tree and scores
    return evaluator.get_summary()