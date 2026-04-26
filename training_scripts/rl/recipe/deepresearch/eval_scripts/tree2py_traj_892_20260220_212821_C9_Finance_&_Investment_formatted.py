import asyncio
import logging
from typing import Optional, List, Dict, Any, Tuple

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "xrp_spot_etf_2025_11"
TASK_DESCRIPTION = """
Identify 4 distinct spot XRP exchange-traded funds (ETFs) that launched in the United States in November 2025. For each ETF, provide:
1) Official fund name, ticker symbol, sponsoring firm
2) Expense ratio
3) Exchange listing (NYSE Arca, Nasdaq, or Cboe BZX) and confirmation of November 2025 launch
4) Custodian and confirmation of spot exposure (physically-backed, not futures)
5) A reference URL for each category above.
Ensure all four ETFs are distinct products from different fund families, represent at least three different sponsoring firms collectively, and list on at least two different U.S. exchanges collectively. All must be spot ETFs that hold actual XRP.
"""


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class ETFItem(BaseModel):
    fund_name: Optional[str] = None
    ticker: Optional[str] = None
    sponsor: Optional[str] = None
    identification_url: Optional[str] = None

    expense_ratio: Optional[str] = None
    fee_url: Optional[str] = None

    exchange: Optional[str] = None
    launch_date: Optional[str] = None
    exchange_url: Optional[str] = None

    custodian: Optional[str] = None
    spot_exposure: Optional[str] = None
    custody_url: Optional[str] = None


class ETFExtraction(BaseModel):
    etfs: List[ETFItem] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_xrp_etfs() -> str:
    return """
    Extract up to FOUR spot XRP ETFs as they are presented in the answer text. For each ETF, return an object with the following string fields (use null if missing):
    - fund_name: Official fund name
    - ticker: Trading ticker symbol
    - sponsor: Sponsoring asset management firm (issuer/manager)
    - identification_url: A URL in the answer that confirms fund_name/ticker/sponsor
    - expense_ratio: Expense ratio (management fee) as presented (e.g., "0.25%" or "25 bps")
    - fee_url: A URL in the answer that confirms the expense ratio
    - exchange: U.S. listing exchange (e.g., "NYSE Arca", "Nasdaq", or "Cboe BZX")
    - launch_date: The launch date as given in the answer (any string form)
    - exchange_url: A URL in the answer that confirms exchange listing and launch timing
    - custodian: The qualified custodian holding the ETF’s XRP
    - spot_exposure: The answer’s wording confirming spot/physically-backed exposure (e.g., "spot, physically backed", "holds actual XRP", "not futures")
    - custody_url: A URL in the answer that confirms custodian and spot exposure
    
    Rules:
    - Only extract information explicitly present in the answer.
    - For each URL field, extract a single representative URL if multiple are present (prefer an official issuer page, prospectus, or exchange listing; otherwise take the first).
    - Do NOT invent or infer any value or URL.
    - Preserve text exactly as written for fields like expense_ratio and launch_date.
    - Return the array as {"etfs": [ ... up to 4 items ... ]}.
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def is_valid_url(url: Optional[str]) -> bool:
    if not url:
        return False
    u = url.strip()
    return u.startswith("http://") or u.startswith("https://")


def normalize_text(s: Optional[str]) -> str:
    return (s or "").strip().lower()


def normalize_ticker(t: Optional[str]) -> str:
    return (t or "").strip().upper()


def normalize_exchange_name(x: Optional[str]) -> str:
    if not x:
        return ""
    s = x.strip().lower()
    if "arca" in s and "nyse" in s:
        return "NYSE Arca"
    if "nasdaq" in s:
        return "Nasdaq"
    if "cboe" in s and "bzx" in s:
        return "Cboe BZX"
    if "bzx" in s:
        return "Cboe BZX"
    # Fallback: title case the input
    return x.strip()


def allowed_exchange(normalized: str) -> bool:
    return normalized in {"NYSE Arca", "Nasdaq", "Cboe BZX"}


def compute_distinctness_flags(items: List[ETFItem]) -> List[bool]:
    """
    For each ETF, return True if:
      - ticker is unique among the four
      - sponsor is unique among the four (different fund family)
      - fund_name is unique among the four
    """
    tickers = [normalize_ticker(i.ticker) for i in items]
    sponsors = [normalize_text(i.sponsor) for i in items]
    fund_names = [normalize_text(i.fund_name) for i in items]

    def counts(lst: List[str]) -> Dict[str, int]:
        c: Dict[str, int] = {}
        for v in lst:
            if v:
                c[v] = c.get(v, 0) + 1
        return c

    t_counts = counts(tickers)
    s_counts = counts(sponsors)
    n_counts = counts(fund_names)

    flags = []
    for i in range(len(items)):
        t = tickers[i]
        s = sponsors[i]
        n = fund_names[i]
        t_unique = (t != "") and t_counts.get(t, 0) == 1
        s_unique = (s != "") and s_counts.get(s, 0) == 1
        n_unique = (n != "") and n_counts.get(n, 0) == 1
        flags.append(t_unique and s_unique and n_unique)
    return flags


def compute_diversity(items: List[ETFItem]) -> Tuple[int, int]:
    sponsors = set(normalize_text(i.sponsor) for i in items if normalize_text(i.sponsor))
    exchanges = set(normalize_exchange_name(i.exchange) for i in items if normalize_exchange_name(i.exchange))
    return len(sponsors), len(exchanges)


# --------------------------------------------------------------------------- #
# Verification builder for a single ETF                                       #
# --------------------------------------------------------------------------- #
async def build_etf_verification(
    evaluator: Evaluator,
    parent_node,
    etf: ETFItem,
    index_1_based: int,
    distinct_ok: bool
) -> None:
    """
    Build verification sub-tree for a single ETF under parent_node.
    """
    etf_node = evaluator.add_parallel(
        id=f"etf_{index_1_based}",
        desc=f"{['First','Second','Third','Fourth'][index_1_based-1]} XRP spot ETF identification and verification",
        parent=parent_node,
        critical=False
    )

    # ---------------- Identification ----------------
    identification_node = evaluator.add_parallel(
        id=f"etf_{index_1_based}_identification",
        desc=f"Basic identification information for the {['first','second','third','fourth'][index_1_based-1]} ETF",
        parent=etf_node,
        critical=True
    )
    # URL presence check
    id_url_present = evaluator.add_custom_node(
        result=is_valid_url(etf.identification_url),
        id=f"etf_{index_1_based}_identification_url",
        desc="Provide a reference URL confirming the fund name, ticker, and sponsor",
        parent=identification_node,
        critical=True
    )
    # Fund name
    fn_leaf = evaluator.add_leaf(
        id=f"etf_{index_1_based}_fund_name",
        desc="Provide the official fund name of a spot XRP ETF that launched in November 2025",
        parent=identification_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The official fund name is '{etf.fund_name}'.",
        node=fn_leaf,
        sources=etf.identification_url,
        additional_instruction="Confirm the official fund name on the cited page. Allow minor formatting variants (e.g., 'Trust' vs 'ETF'), but it must clearly refer to the same product.",
        extra_prerequisites=[id_url_present]
    )
    # Ticker
    ticker_leaf = evaluator.add_leaf(
        id=f"etf_{index_1_based}_ticker",
        desc="Provide the ticker symbol used for trading this ETF",
        parent=identification_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The ETF's ticker symbol is '{etf.ticker}'.",
        node=ticker_leaf,
        sources=etf.identification_url,
        additional_instruction="Verify the trading symbol exactly (case-insensitive).",
        extra_prerequisites=[id_url_present]
    )
    # Sponsor
    sponsor_leaf = evaluator.add_leaf(
        id=f"etf_{index_1_based}_sponsor",
        desc="Identify the asset management firm that sponsors this ETF",
        parent=identification_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The sponsoring asset management firm (issuer/manager) is '{etf.sponsor}'.",
        node=sponsor_leaf,
        sources=etf.identification_url,
        additional_instruction="Treat 'issuer', 'sponsor', 'manager', or 'advisor' as acceptable identifiers for the sponsoring firm if the page clearly indicates the responsible firm.",
        extra_prerequisites=[id_url_present]
    )

    # ---------------- Cost Structure ----------------
    cost_node = evaluator.add_parallel(
        id=f"etf_{index_1_based}_cost_structure",
        desc=f"Cost and fee information for the {['first','second','third','fourth'][index_1_based-1]} ETF",
        parent=etf_node,
        critical=True
    )
    fee_url_present = evaluator.add_custom_node(
        result=is_valid_url(etf.fee_url),
        id=f"etf_{index_1_based}_fee_url",
        desc="Provide a reference URL confirming the expense ratio",
        parent=cost_node,
        critical=True
    )
    expense_leaf = evaluator.add_leaf(
        id=f"etf_{index_1_based}_expense_ratio",
        desc="Specify the expense ratio (management fee) charged by this ETF",
        parent=cost_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The expense ratio (management fee) is '{etf.expense_ratio}'.",
        node=expense_leaf,
        sources=etf.fee_url,
        additional_instruction="Verify the management fee. If both gross and net (waived/promotional) fees are shown, accept the value as stated in the answer if it appears on the page in any official fee representation.",
        extra_prerequisites=[fee_url_present]
    )

    # ---------------- Exchange & Launch ----------------
    exch_node = evaluator.add_parallel(
        id=f"etf_{index_1_based}_exchange_info",
        desc=f"Exchange listing and trading information for the {['first','second','third','fourth'][index_1_based-1]} ETF",
        parent=etf_node,
        critical=True
    )
    exch_url_present = evaluator.add_custom_node(
        result=is_valid_url(etf.exchange_url),
        id=f"etf_{index_1_based}_exchange_url",
        desc="Provide a reference URL confirming the exchange listing and launch date",
        parent=exch_node,
        critical=True
    )
    norm_exch = normalize_exchange_name(etf.exchange)
    exch_leaf = evaluator.add_leaf(
        id=f"etf_{index_1_based}_exchange",
        desc="Identify the U.S. exchange where this ETF is listed (must be NYSE Arca, Nasdaq, or CBOE BZX)",
        parent=exch_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The ETF is listed on '{norm_exch}', which is one of NYSE Arca, Nasdaq, or Cboe BZX.",
        node=exch_leaf,
        sources=etf.exchange_url,
        additional_instruction="Confirm the listing exchange on the cited page. Allow naming variants like 'NYSE Arca, Inc.', 'Nasdaq Stock Market', or 'Cboe BZX Exchange'.",
        extra_prerequisites=[exch_url_present]
    )
    launch_leaf = evaluator.add_leaf(
        id=f"etf_{index_1_based}_launch_date",
        desc="Confirm the ETF launched in November 2025",
        parent=exch_node,
        critical=True
    )
    await evaluator.verify(
        claim="This ETF launched (began trading) in November 2025.",
        node=launch_leaf,
        sources=etf.exchange_url,
        additional_instruction="Accept explicit phrasing like 'launched in Nov. 2025', 'began trading in November 2025', or specific November 2025 dates.",
        extra_prerequisites=[exch_url_present]
    )

    # ---------------- Custody & Spot Exposure ----------------
    custody_node = evaluator.add_parallel(
        id=f"etf_{index_1_based}_custody",
        desc=f"Custody and operational details for the {['first','second','third','fourth'][index_1_based-1]} ETF",
        parent=etf_node,
        critical=True
    )
    custody_url_present = evaluator.add_custom_node(
        result=is_valid_url(etf.custody_url),
        id=f"etf_{index_1_based}_custody_url",
        desc="Provide a reference URL confirming custodian and spot exposure structure",
        parent=custody_node,
        critical=True
    )
    custodian_leaf = evaluator.add_leaf(
        id=f"etf_{index_1_based}_custodian",
        desc="Identify the qualified custodian holding the ETF's XRP assets",
        parent=custody_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The ETF's XRP custodian is '{etf.custodian}'.",
        node=custodian_leaf,
        sources=etf.custody_url,
        additional_instruction="Confirm the qualified custodian (e.g., 'Custodian', 'Digital assets custodian', 'Qualified custodian').",
        extra_prerequisites=[custody_url_present]
    )
    spot_leaf = evaluator.add_leaf(
        id=f"etf_{index_1_based}_spot_exposure",
        desc="Confirm the ETF provides spot exposure with physical XRP backing (not futures-based)",
        parent=custody_node,
        critical=True
    )
    await evaluator.verify(
        claim="The ETF is a spot XRP ETF that holds actual XRP (physically backed), not futures-based.",
        node=spot_leaf,
        sources=etf.custody_url,
        additional_instruction="Verify the page states the ETF holds actual XRP (physically-backed, spot exposure); reject if exposure is futures-based or synthetic.",
        extra_prerequisites=[custody_url_present]
    )

    # ---------------- Distinctness ----------------
    evaluator.add_custom_node(
        result=distinct_ok,
        id=f"etf_{index_1_based}_distinctness",
        desc="Verify this ETF is distinct from the other three identified ETFs",
        parent=etf_node,
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
    Evaluate an answer for the XRP spot ETF November 2025 task.
    """
    # Initialize evaluator (root is non-critical to allow partial scoring; inner critical nodes enforce requirements)
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

    # Extract ETF entries from answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_xrp_etfs(),
        template_class=ETFExtraction,
        extraction_name="extracted_etfs"
    )

    # Keep exactly 4 ETFs (pad with empty entries if fewer)
    etfs: List[ETFItem] = list(extracted.etfs[:4])
    while len(etfs) < 4:
        etfs.append(ETFItem())

    # Compute distinctness and diversity
    distinct_flags = compute_distinctness_flags(etfs)
    sponsor_count, exchange_count = compute_diversity(etfs)

    # Build per-ETF verification trees
    for idx, etf in enumerate(etfs, start=1):
        await build_etf_verification(
            evaluator=evaluator,
            parent_node=root,
            etf=etf,
            index_1_based=idx,
            distinct_ok=distinct_flags[idx - 1]
        )

    # Diversity requirements across all four ETFs
    diversity_node = evaluator.add_parallel(
        id="diversity_requirements",
        desc="Verify diversity across the four identified ETFs",
        parent=root,
        critical=True
    )

    evaluator.add_custom_node(
        result=(sponsor_count >= 3),
        id="sponsor_diversity",
        desc="Verify that the four ETFs are issued by at least three different asset management firms",
        parent=diversity_node,
        critical=True
    )

    evaluator.add_custom_node(
        result=(exchange_count >= 2),
        id="exchange_diversity",
        desc="Verify that the four ETFs collectively list on at least two different U.S. exchanges",
        parent=diversity_node,
        critical=True
    )

    # Record some custom info for transparency
    normalized_exchanges = [normalize_exchange_name(e.exchange) for e in etfs]
    evaluator.add_custom_info(
        {
            "unique_sponsor_count": sponsor_count,
            "unique_exchange_count": exchange_count,
            "normalized_exchanges": normalized_exchanges,
            "tickers": [normalize_ticker(e.ticker) for e in etfs],
            "sponsors": [e.sponsor for e in etfs],
        },
        info_type="computed_aggregates",
        info_name="diversity_metrics"
    )

    # Return evaluation summary
    return evaluator.get_summary()