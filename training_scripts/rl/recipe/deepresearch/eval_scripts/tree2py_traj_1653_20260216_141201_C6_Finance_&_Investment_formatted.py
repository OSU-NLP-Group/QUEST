import asyncio
import logging
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


TASK_ID = "xrp_spot_etf_nov2025"
TASK_DESCRIPTION = """In November 2025, several asset management companies launched spot XRP exchange-traded funds (ETFs) in the United States. Identify THREE distinct spot XRP ETFs that satisfy ALL of the following criteria:

1. Launch Timeframe: The ETF began trading in November 2025 (between November 1-30, 2025) in the United States market.
2. Product Type: The ETF is a spot XRP ETF, meaning it directly holds XRP as the underlying asset (not futures-based, not leveraged, not inverse).
3. Fee Structure: The ETF's annual expense ratio is 0.40% or lower.
4. Issuer Experience: The asset management company that issued the ETF has also filed or launched at least one other cryptocurrency spot ETF covering Bitcoin (BTC) or Ethereum (ETH), either before or at the same time as this XRP ETF.
5. Exchange Listing: The ETF is listed and trades on one of the following major U.S. exchanges: NYSE Arca, Nasdaq, or Cboe.
6. Custodian Disclosure: The ETF's prospectus, Form S-1, or official product documentation publicly discloses the name of at least one qualified custodian responsible for holding the ETF's XRP holdings.
7. Seed Investment Disclosure: The ETF's regulatory filings (such as Form S-1 or amendments) disclose both the initial seed capital investment amount and the name of the initial authorized participant or seed investor who provided that capital.

For each of the three ETFs you identify, provide:
- The full official name of the ETF
- The ticker symbol
- The issuer/sponsor company name
- The expense ratio
- The launch date
- The exchange where it is listed
- The custodian name(s)
- The initial seed investment amount and seed investor name
- Reference URL(s) supporting each piece of information
"""


# =========================
# Data Models for Extraction
# =========================

class ETFItem(BaseModel):
    name: Optional[str] = None
    ticker: Optional[str] = None
    issuer: Optional[str] = None
    expense_ratio: Optional[str] = None
    launch_date: Optional[str] = None
    exchange: Optional[str] = None
    custodian_names: List[str] = Field(default_factory=list)
    seed_investment_amount: Optional[str] = None
    seed_investor_name: Optional[str] = None

    # URL references (categorical + fallback)
    basic_refs: List[str] = Field(default_factory=list)                # For name/ticker/issuer
    regulatory_refs: List[str] = Field(default_factory=list)           # For launch date/market/spot/exchange/fee
    issuer_experience_refs: List[str] = Field(default_factory=list)    # For issuer BTC/ETH spot ETF(s)
    operational_refs: List[str] = Field(default_factory=list)          # For custodian + seed disclosures
    all_refs: List[str] = Field(default_factory=list)                  # Fallback if categories aren’t specified

    # Optional info about issuer's other spot products (names or tickers)
    issuer_other_spot_products: List[str] = Field(default_factory=list)


class ETFsExtraction(BaseModel):
    etfs: List[ETFItem] = Field(default_factory=list)


# =========================
# Extraction Prompt
# =========================

def prompt_extract_etfs() -> str:
    return """
Extract up to all XRP ETF entries mentioned in the answer. For each ETF, extract the following fields exactly as stated in the answer text:

- name: Full official fund name
- ticker: The ETF's ticker symbol
- issuer: The issuer/sponsor company name
- expense_ratio: Annual expense ratio as written (e.g., "0.25%" or "0.30%")
- launch_date: Trading launch date as stated (e.g., "November 12, 2025")
- exchange: The U.S. exchange where the ETF is listed (e.g., "NYSE Arca", "Nasdaq", "Cboe", "Cboe BZX")
- custodian_names: An array of custodian names disclosed in prospectus/official docs (e.g., ["Coinbase Custody"])
- seed_investment_amount: The initial seed capital amount as disclosed (e.g., "$10,000,000")
- seed_investor_name: The name of the initial authorized participant or seed investor (e.g., "Jane Street Capital LLC")

- basic_refs: URLs supporting the ETF's name, ticker, and issuer (prefer official product pages, exchange listings, or SEC filings)
- regulatory_refs: URLs supporting launch date, US market, spot (physically backed) nature, exchange listing, and fee
- issuer_experience_refs: URLs showing the issuer has filed or launched a Bitcoin or Ethereum spot ETF (e.g., SEC filings or press releases)
- operational_refs: URLs that disclose custodian(s) and seed investment details (SEC S-1, amendments, or official documentation)
- all_refs: Include all other URLs mentioned for this ETF that may be relevant; if you cannot categorize references, put them here.

- issuer_other_spot_products: Names or tickers of the issuer’s other spot products (BTC/ETH), if explicitly stated

Rules:
1) Only extract information that is explicitly present in the answer text. Do not invent or infer.
2) For any missing field, return null (or an empty array for list fields).
3) For URLs, extract valid full URLs (plain or from markdown links). If none are given for a category but there are general references, put them into all_refs.
4) Do not normalize numeric formats; keep the exact strings (e.g., "0.25%", "$10 million", "November 12, 2025").
5) Return a JSON object with a top-level "etfs" array of ETF objects.
"""


# =========================
# Utility Helpers
# =========================

def _normalize_text(s: Optional[str]) -> str:
    return s.strip() if isinstance(s, str) else ""


def _lower(s: Optional[str]) -> str:
    return _normalize_text(s).lower()


def _dedupe(urls: List[str]) -> List[str]:
    seen = set()
    out = []
    for u in urls:
        if not u:
            continue
        if u not in seen:
            seen.add(u)
            out.append(u)
    return out


def _collect_urls(item: ETFItem, category: str) -> List[str]:
    """
    Get URLs for a given category with fallback to all_refs, then union of all sets.
    category in {"basic", "regulatory", "issuer_experience", "operational"}
    """
    if category == "basic":
        urls = item.basic_refs
    elif category == "regulatory":
        urls = item.regulatory_refs
    elif category == "issuer_experience":
        urls = item.issuer_experience_refs
    elif category == "operational":
        urls = item.operational_refs
    else:
        urls = []

    urls = urls or item.all_refs
    if not urls:
        # ultimate fallback: union all categories
        all_cat = (item.basic_refs or []) + (item.regulatory_refs or []) + (item.issuer_experience_refs or []) + (item.operational_refs or []) + (item.all_refs or [])
        urls = all_cat
    return _dedupe(urls or [])


async def _verify_with_urls_or_fail(
    evaluator: Evaluator,
    claim: str,
    node,
    urls: List[str],
    additional_instruction: str = "None"
) -> bool:
    """
    Run URL-grounded verification if URLs exist; otherwise mark the node failed explicitly.
    This adheres to the policy that factual leaves should be web-grounded whenever possible.
    """
    if urls and len(urls) > 0:
        return await evaluator.verify(
            claim=claim,
            node=node,
            sources=urls,
            additional_instruction=additional_instruction
        )
    else:
        # No sources provided → fail the leaf explicitly
        node.score = 0.0
        node.status = "failed"
        return False


def _allowed_us_exchanges() -> List[str]:
    return ["nyse arca", "nasdaq", "cboe", "cboe bzx", "cboe bzx exchange"]


# =========================
# Verification for One ETF
# =========================

async def verify_single_etf(
    evaluator: Evaluator,
    etf_parent,
    etf: ETFItem,
    idx: int,
    ordinal_name: str
) -> None:
    """
    Build and verify the tree for a single ETF.
    All sub-criteria nodes mirror the rubric structure. Critical grouping nodes have critical=True
    and all their children are marked critical as required by the framework.
    """
    # ---------- Basic Identifiers ----------
    basic_node = evaluator.add_parallel(
        id=f"etf_{idx}_basic_identifiers",
        desc=f"{ordinal_name} ETF: Basic identifying information",
        parent=etf_parent,
        critical=True
    )

    # Name & Ticker existence (custom existence check)
    name_ticker_exists = evaluator.add_custom_node(
        result=bool(_normalize_text(etf.name)) and bool(_normalize_text(etf.ticker)),
        id=f"etf_{idx}_name_ticker_provided",
        desc=f"{ordinal_name} ETF's official name and ticker symbol are provided",
        parent=basic_node,
        critical=True
    )

    # Issuer existence
    issuer_exists = evaluator.add_custom_node(
        result=bool(_normalize_text(etf.issuer)),
        id=f"etf_{idx}_issuer_provided",
        desc=f"{ordinal_name} ETF issuer/sponsor company name is provided",
        parent=basic_node,
        critical=True
    )

    # Reference confirms name, ticker, issuer
    basic_ref_leaf = evaluator.add_leaf(
        id=f"etf_{idx}_basic_reference",
        desc=f"Reference URL confirms {ordinal_name.lower()} ETF name, ticker, and issuer",
        parent=basic_node,
        critical=True
    )
    basic_urls = _collect_urls(etf, "basic")
    basic_claim = f"The ETF named '{_normalize_text(etf.name)}' with ticker '{_normalize_text(etf.ticker)}' is issued by '{_normalize_text(etf.issuer)}'."
    await _verify_with_urls_or_fail(
        evaluator,
        claim=basic_claim,
        node=basic_ref_leaf,
        urls=basic_urls,
        additional_instruction="Verify that at least one provided URL clearly shows the fund name, ticker, and issuer/sponsor. Allow minor formatting differences."
    )

    # ---------- Regulatory & Market Requirements ----------
    regulatory_node = evaluator.add_parallel(
        id=f"etf_{idx}_regulatory",
        desc=f"{ordinal_name} ETF: Regulatory and market requirements",
        parent=etf_parent,
        critical=True
    )

    # Launch Requirements (date within Nov 2025 and US market)
    launch_req_node = evaluator.add_parallel(
        id=f"etf_{idx}_launch",
        desc=f"{ordinal_name} ETF: Launch date and market verification",
        parent=regulatory_node,
        critical=True
    )

    # Launch Date within Nov 2025
    launch_date_leaf = evaluator.add_leaf(
        id=f"etf_{idx}_launch_date",
        desc=f"{ordinal_name} ETF launch date is within November 2025 (Nov 1–30, 2025)",
        parent=launch_req_node,
        critical=True
    )
    reg_urls = _collect_urls(etf, "regulatory")
    if _normalize_text(etf.launch_date):
        ld_claim = f"This ETF began trading in November 2025; the stated launch date is '{_normalize_text(etf.launch_date)}', which is in November 2025."
    else:
        ld_claim = "This ETF began trading in November 2025 (between November 1–30, 2025)."
    await _verify_with_urls_or_fail(
        evaluator,
        claim=ld_claim,
        node=launch_date_leaf,
        urls=reg_urls,
        additional_instruction="Confirm that trading commenced in November 2025 per exchange listing, press release, or SEC filing. Accept language like 'began trading', 'launch', 'commenced trading'."
    )

    # Launch Market (United States)
    launch_market_leaf = evaluator.add_leaf(
        id=f"etf_{idx}_launch_market",
        desc=f"{ordinal_name} ETF launched in the United States market",
        parent=launch_req_node,
        critical=True
    )
    lm_claim = "This ETF launched in the United States market."
    await _verify_with_urls_or_fail(
        evaluator,
        claim=lm_claim,
        node=launch_market_leaf,
        urls=reg_urls,
        additional_instruction="Confirm that the ETF trades on a U.S. exchange (NYSE Arca, Nasdaq, or Cboe). Exchange listing evidence suffices to establish U.S. market."
    )

    # Product Structure: spot and listed on allowed exchange
    product_node = evaluator.add_parallel(
        id=f"etf_{idx}_product",
        desc=f"{ordinal_name} ETF: Product type and exchange listing verification",
        parent=regulatory_node,
        critical=True
    )

    # Spot (physically backed) type
    spot_type_leaf = evaluator.add_leaf(
        id=f"etf_{idx}_spot_type",
        desc=f"{ordinal_name} ETF is a spot XRP ETF (directly holds XRP), not futures/leveraged/inverse",
        parent=product_node,
        critical=True
    )
    spot_claim = "This ETF is a spot XRP ETF that directly holds XRP (physically backed), and is not futures-based, leveraged, or inverse."
    await _verify_with_urls_or_fail(
        evaluator,
        claim=spot_claim,
        node=spot_type_leaf,
        urls=reg_urls,
        additional_instruction="Look for language such as 'spot', 'physically backed', 'holds XRP in custody', and absence of 'futures', 'leveraged', or 'inverse'. SEC filings or official pages should state this."
    )

    # Exchange listing: NYSE Arca, Nasdaq, or Cboe
    exchange_leaf = evaluator.add_leaf(
        id=f"etf_{idx}_exchange_listing",
        desc=f"{ordinal_name} ETF is listed on a major U.S. exchange (NYSE Arca, Nasdaq, or Cboe)",
        parent=product_node,
        critical=True
    )
    exch_text = _normalize_text(etf.exchange)
    exch_claim = f"The ETF is listed on '{exch_text}', which is one of NYSE Arca, Nasdaq, or Cboe."
    await _verify_with_urls_or_fail(
        evaluator,
        claim=exch_claim,
        node=exchange_leaf,
        urls=reg_urls,
        additional_instruction=f"Accept synonyms like 'Cboe BZX' for Cboe. Valid exchanges: {', '.join(_allowed_us_exchanges())}. Minor naming variants are acceptable."
    )

    # Fee Structure (<= 0.40%)
    fee_node = evaluator.add_parallel(
        id=f"etf_{idx}_fee",
        desc=f"{ordinal_name} ETF: Fee requirement verification",
        parent=regulatory_node,
        critical=True
    )
    fee_leaf = evaluator.add_leaf(
        id=f"etf_{idx}_fee_requirement",
        desc=f"{ordinal_name} ETF's expense ratio is 0.40% or lower",
        parent=fee_node,
        critical=True
    )
    fee_text = _normalize_text(etf.expense_ratio)
    if fee_text:
        fee_claim = f"The ETF's annual expense ratio is '{fee_text}' and is 0.40% or lower."
    else:
        fee_claim = "The ETF's annual expense ratio is 0.40% or lower."
    await _verify_with_urls_or_fail(
        evaluator,
        claim=fee_claim,
        node=fee_leaf,
        urls=reg_urls,
        additional_instruction="Confirm the expense ratio posted in official sources. If multiple share classes exist, match the share class implied by the ticker. Accept reasonable formatting differences (e.g., 0.25% vs 0.25)."
    )

    # Regulatory Reference confirms set of requirements
    reg_ref_leaf = evaluator.add_leaf(
        id=f"etf_{idx}_reg_reference",
        desc=f"Reference URL(s) confirm {ordinal_name.lower()} ETF launch timing, US market, spot type, exchange, and fee",
        parent=regulatory_node,
        critical=True
    )
    reg_bundle_claim = "The provided references confirm the ETF's November 2025 launch in the U.S., spot XRP structure (physically backed), listing exchange, and the stated fee."
    await _verify_with_urls_or_fail(
        evaluator,
        claim=reg_bundle_claim,
        node=reg_ref_leaf,
        urls=reg_urls,
        additional_instruction="A single authoritative source may confirm multiple items; otherwise, multiple links together suffice."
    )

    # ---------- Issuer Experience ----------
    issuer_node = evaluator.add_parallel(
        id=f"etf_{idx}_issuer_verification",
        desc=f"{ordinal_name} ETF: Issuer experience with other cryptocurrency spot ETFs",
        parent=etf_parent,
        critical=True
    )

    issuer_credentials_node = evaluator.add_parallel(
        id=f"etf_{idx}_issuer_credentials",
        desc=f"{ordinal_name} ETF: Issuer's other cryptocurrency ETF products",
        parent=issuer_node,
        critical=True
    )

    other_crypto_leaf = evaluator.add_leaf(
        id=f"etf_{idx}_issuer_other_crypto",
        desc=f"Issuer of the {ordinal_name.lower()} ETF has filed or launched at least one BTC/ETH spot ETF prior to or concurrent with this XRP ETF",
        parent=issuer_credentials_node,
        critical=True
    )
    issuer_urls = _collect_urls(etf, "issuer_experience")
    issuer_name_txt = _normalize_text(etf.issuer)
    other_products_txt = ", ".join(etf.issuer_other_spot_products) if etf.issuer_other_spot_products else "at least one BTC/ETH spot ETF"
    issuer_claim = f"The issuer '{issuer_name_txt}' has filed or launched {other_products_txt} (Bitcoin or Ethereum spot ETF) prior to or at the time of this XRP ETF."
    await _verify_with_urls_or_fail(
        evaluator,
        claim=issuer_claim,
        node=other_crypto_leaf,
        urls=issuer_urls,
        additional_instruction="Accept SEC S-1/424B filings or credible press releases indicating spot BTC/ETH ETF filings/launches. Futures-only products are not sufficient."
    )

    issuer_ref_leaf = evaluator.add_leaf(
        id=f"etf_{idx}_issuer_reference",
        desc=f"Reference URL(s) confirm issuer's other cryptocurrency spot ETF products",
        parent=issuer_node,
        critical=True
    )
    issuer_bundle_claim = "The provided references confirm that the issuer has at least one Bitcoin or Ethereum spot ETF filed or launched."
    await _verify_with_urls_or_fail(
        evaluator,
        claim=issuer_bundle_claim,
        node=issuer_ref_leaf,
        urls=issuer_urls,
        additional_instruction="If multiple references collectively support this, that is acceptable."
    )

    # ---------- Operational Disclosures ----------
    operational_node = evaluator.add_parallel(
        id=f"etf_{idx}_operational",
        desc=f"{ordinal_name} ETF: Custodian and seed investment disclosures",
        parent=etf_parent,
        critical=True
    )

    custodian_node = evaluator.add_parallel(
        id=f"etf_{idx}_custodian_info",
        desc=f"{ordinal_name} ETF: Custodian disclosure verification",
        parent=operational_node,
        critical=True
    )

    custodian_leaf = evaluator.add_leaf(
        id=f"etf_{idx}_custodian_disclosure",
        desc=f"{ordinal_name} ETF filings disclose qualified custodian(s) for XRP",
        parent=custodian_node,
        critical=True
    )
    op_urls = _collect_urls(etf, "operational")
    custodian_list_txt = ", ".join(etf.custodian_names) if etf.custodian_names else ""
    if custodian_list_txt:
        custodian_claim = f"The ETF's official documentation discloses the qualified custodian(s): {custodian_list_txt}."
    else:
        custodian_claim = "The ETF's official documentation discloses the qualified custodian(s) responsible for holding the ETF's XRP."
    await _verify_with_urls_or_fail(
        evaluator,
        claim=custodian_claim,
        node=custodian_leaf,
        urls=op_urls,
        additional_instruction="Look for custodian names in SEC filings (e.g., S-1, S-1/A) or official product pages. Accept multiple custodians."
    )

    seed_node = evaluator.add_parallel(
        id=f"etf_{idx}_seed_info",
        desc=f"{ordinal_name} ETF: Seed investment disclosure verification",
        parent=operational_node,
        critical=True
    )

    seed_leaf = evaluator.add_leaf(
        id=f"etf_{idx}_seed_investment",
        desc=f"{ordinal_name} ETF filings disclose both seed capital amount and initial AP/seed investor name",
        parent=seed_node,
        critical=True
    )
    seed_amt_txt = _normalize_text(etf.seed_investment_amount)
    seed_inv_txt = _normalize_text(etf.seed_investor_name)
    if seed_amt_txt and seed_inv_txt:
        seed_claim = f"The ETF's filings disclose the initial seed capital amount '{seed_amt_txt}' and the initial authorized participant or seed investor '{seed_inv_txt}'."
    else:
        seed_claim = "The ETF's filings disclose both the initial seed capital amount and the name of the initial authorized participant or seed investor."
    await _verify_with_urls_or_fail(
        evaluator,
        claim=seed_claim,
        node=seed_leaf,
        urls=op_urls,
        additional_instruction="Confirm both the amount and the seed investor/AP are explicitly named in filings (e.g., S-1, amendments). Both must be present."
    )

    operational_ref_leaf = evaluator.add_leaf(
        id=f"etf_{idx}_operational_reference",
        desc=f"Reference URL(s) confirm custodian and seed investment details for the {ordinal_name.lower()} ETF",
        parent=operational_node,
        critical=True
    )
    op_bundle_claim = "The provided references confirm the ETF's custodian(s) and the initial seed capital details including the seed investor/AP."
    await _verify_with_urls_or_fail(
        evaluator,
        claim=op_bundle_claim,
        node=operational_ref_leaf,
        urls=op_urls,
        additional_instruction="A single authoritative filing may confirm both items; otherwise multiple references are acceptable."
    )


# =========================
# Main Evaluation Function
# =========================

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

    # Extract ETF items
    extraction = await evaluator.extract(
        prompt=prompt_extract_etfs(),
        template_class=ETFsExtraction,
        extraction_name="xrp_etf_extraction"
    )

    all_etfs: List[ETFItem] = extraction.etfs if extraction and extraction.etfs else []

    # Select up to first 3 for evaluation; pad with placeholders if fewer found
    selected: List[ETFItem] = all_etfs[:3]
    while len(selected) < 3:
        selected.append(ETFItem())

    # Add some diagnostic info
    evaluator.add_custom_info(
        info={
            "total_etfs_extracted": len(all_etfs),
            "using_first_three": len(selected)
        },
        info_type="diagnostics",
        info_name="extraction_stats"
    )

    # ---------- Completeness & Distinctness (Critical) ----------
    completeness_node = evaluator.add_parallel(
        id="etf_set_completeness",
        desc="Verification that at least three distinct XRP spot ETFs are identified for evaluation",
        parent=root,
        critical=True
    )

    # Count how many provided ETFs have minimally name & ticker & issuer present
    valid_count = sum(
        1 for it in selected if _normalize_text(it.name) and _normalize_text(it.ticker) and _normalize_text(it.issuer)
    )
    three_provided = evaluator.add_custom_node(
        result=(valid_count >= 3),
        id="three_etfs_provided",
        desc="Exactly three ETFs are provided in the answer (treated as satisfied if at least three valid ETFs are present; only the first three are evaluated)",
        parent=completeness_node,
        critical=True
    )

    # Distinctness across the three selected ETFs (names, tickers, issuers)
    names = [_lower(it.name) for it in selected if _normalize_text(it.name)]
    tickers = [_lower(it.ticker) for it in selected if _normalize_text(it.ticker)]
    issuers = [_lower(it.issuer) for it in selected if _normalize_text(it.issuer)]
    distinct = evaluator.add_custom_node(
        result=(len(names) == 3 and len(set(names)) == 3 and
                len(tickers) == 3 and len(set(tickers)) == 3 and
                len(issuers) == 3 and len(set(issuers)) == 3),
        id="etfs_are_distinct",
        desc="The three ETFs have different names, tickers, and issuers (no duplicates among the evaluated three)",
        parent=completeness_node,
        critical=True
    )

    # ---------- Per-ETF Verification (Non-critical container, critical sub-criteria inside) ----------
    ordinals = ["First", "Second", "Third"]
    for i in range(3):
        etf_container = evaluator.add_parallel(
            id=f"etf_{i}",
            desc=f"{ordinals[i]} XRP spot ETF meeting all criteria",
            parent=root,
            critical=False
        )
        await verify_single_etf(
            evaluator=evaluator,
            etf_parent=etf_container,
            etf=selected[i],
            idx=i,
            ordinal_name=ordinals[i]
        )

    return evaluator.get_summary()