import asyncio
import logging
import re
from typing import Optional, List, Dict, Any, Tuple

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "spot_btc_etf_us_eval"
TASK_DESCRIPTION = (
    "I am conducting comparative research on spot Bitcoin ETFs available in the U.S. market. Please identify 5 distinct "
    "spot Bitcoin ETFs that collectively meet the following criteria:\n\n"
    "1. One ETF with an expense ratio of 0.20% or lower\n"
    "2. One ETF with an expense ratio of 0.25% or higher\n"
    "3. One ETF that ranks among the top 5 by assets under management\n"
    "4. One ETF that employs a multi-custodian model (uses more than one custodian)\n"
    "5. One ETF that does NOT use Coinbase as its sole custodian\n\n"
    "All 5 ETFs must be issued by different financial institutions.\n\n"
    "For each ETF, provide the following information:\n"
    "- Ticker symbol\n"
    "- Issuer name\n"
    "- Current expense ratio (%)\n"
    "- Assets under management (in billions USD)\n"
    "- Custodian model description\n"
    "- A reference URL to an official source (such as the ETF's page on the issuer's website, ETF database page, or financial news source)"
)


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class ETFItem(BaseModel):
    """One ETF entry as provided in the agent's answer."""
    ticker: Optional[str] = None
    issuer: Optional[str] = None
    expense_ratio: Optional[str] = None  # Keep as string (e.g., "0.19%", "19 bps")
    aum_billion_usd: Optional[str] = None  # Keep as string (e.g., "3.5", "$3.5B", "3.5 billion")
    custodian_model: Optional[str] = None  # Free-text description of custodians
    reference_urls: List[str] = Field(default_factory=list)  # One or multiple URLs for this ETF


class ETFListExtraction(BaseModel):
    """Container for up to 5 ETFs extracted from the answer."""
    etfs: List[ETFItem] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_etfs() -> str:
    return """
    Extract up to the first five (5) distinct U.S.-traded spot Bitcoin ETFs listed in the answer. For each ETF, extract:
    - ticker: the ticker symbol (string)
    - issuer: the issuer or sponsor name (string)
    - expense_ratio: the stated expense ratio as a text exactly as shown (e.g., "0.19%", "0.25%", "19 bps"). Do not convert to numeric.
    - aum_billion_usd: the assets under management in billions USD as text, if provided (e.g., "3.5", "$3.5B", "3.5 billion USD"). Do not convert to numeric; extract verbatim.
    - custodian_model: brief description of the custody arrangement as described in the answer (e.g., "Coinbase as sole custodian", "multi-custodian: Coinbase and Gemini", "BNY Mellon", etc.).
    - reference_urls: an array of URLs explicitly cited in the answer that refer to this ETF (issuer official page, reputable ETF database page like etfdb/etf.com, or credible financial news/articles). Extract only actual URLs shown (plain or markdown links). If none are provided, return an empty array.

    Requirements:
    - Only extract information explicitly present in the answer.
    - If the answer lists more than 5 ETFs, return only the first 5.
    - If any field for an ETF is missing in the answer, set it to null (or an empty array for reference_urls).
    - Ensure that 'reference_urls' contains only valid-looking URLs. If the answer mentions a source without a URL, do not fabricate one; just leave the array empty.

    Return a JSON object with:
    {
      "etfs": [ { ... up to 5 items ... } ]
    }
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _normalize_ticker(t: Optional[str]) -> Optional[str]:
    if t is None:
        return None
    return re.sub(r"[^A-Za-z0-9]", "", t).upper()


def _normalize_issuer(issuer: Optional[str]) -> Optional[str]:
    if issuer is None:
        return None
    s = issuer.strip().lower()
    s = re.sub(r"[\s&/\-.,]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s or None


def parse_expense_ratio_to_percent(value: Optional[str]) -> Optional[float]:
    """
    Parse an expense ratio string to a percent float (e.g., "0.19%" -> 0.19, "19 bps" -> 0.19).
    Returns None if cannot parse.
    """
    if not value:
        return None
    s = value.strip().lower()
    # Look for basis points
    m_bps = re.search(r"(\d+(\.\d+)?)\s*bp[s]?", s)
    if m_bps:
        try:
            bps = float(m_bps.group(1))
            return bps / 100.0
        except:
            pass
    # Look for percent value
    m_pct = re.search(r"(\d+(\.\d+)?)\s*%$", s)
    if m_pct:
        try:
            return float(m_pct.group(1))
        except:
            pass
    # If just a number like "0.25" assume it's percent figure already
    m_plain = re.fullmatch(r"\d+(\.\d+)?", s)
    if m_plain:
        try:
            return float(s)
        except:
            pass
    # Try to find first number and percent later in string
    m_any = re.search(r"(\d+(\.\d+)?)", s)
    if m_any and "%" in s:
        try:
            return float(m_any.group(1))
        except:
            pass
    return None


def parse_aum_to_billions(value: Optional[str]) -> Optional[float]:
    """
    Parse an AUM text to a float in billions USD if possible.
    Accepts formats like "3.5", "$3.5B", "3.5 billion", "$800M".
    """
    if not value:
        return None
    s = value.strip().lower().replace(",", "")
    # Extract number
    m = re.search(r"(\d+(\.\d+)?)", s)
    if not m:
        return None
    try:
        num = float(m.group(1))
    except:
        return None

    # Unit detection
    if "b" in s or "billion" in s or "bn" in s:
        return num
    if "m" in s or "million" in s:
        return num / 1000.0
    # If unit not specified, assume already billions if seems reasonable
    return num


def collect_all_sources(items: List[ETFItem]) -> List[str]:
    urls: List[str] = []
    for etf in items:
        for u in etf.reference_urls or []:
            if isinstance(u, str) and u.strip():
                urls.append(u.strip())
    # Deduplicate while preserving order
    seen = set()
    deduped = []
    for u in urls:
        if u not in seen:
            deduped.append(u)
            seen.add(u)
    return deduped


# --------------------------------------------------------------------------- #
# Verification node builders                                                  #
# --------------------------------------------------------------------------- #
async def add_count_and_distinctness_node(evaluator: Evaluator, parent, etfs: List[ETFItem]) -> None:
    """
    Adds a critical custom node to ensure exactly 5 ETFs and distinct tickers.
    """
    tickers = [_normalize_ticker(etf.ticker) for etf in etfs if etf.ticker]
    distinct_tickers = len(tickers) == len(set(tickers)) == 5
    result = (len(etfs) == 5) and distinct_tickers

    evaluator.add_custom_node(
        result=result,
        id="ETF_Count_And_Distinctness",
        desc="Response lists exactly 5 ETFs and they are distinct (no duplicate funds/tickers)",
        parent=parent,
        critical=True
    )


async def add_distinct_issuers_node(evaluator: Evaluator, parent, etfs: List[ETFItem]) -> None:
    """
    Adds a critical custom node to ensure all 5 issuers are distinct.
    """
    issuers = [_normalize_issuer(etf.issuer) for etf in etfs if etf.issuer]
    result = (len(issuers) == 5) and (len(set(issuers)) == 5)
    evaluator.add_custom_node(
        result=result,
        id="Distinct_Issuers",
        desc="All 5 identified ETFs are issued by different financial institutions",
        parent=parent,
        critical=True
    )


async def add_complete_information_nodes(evaluator: Evaluator, parent, etfs: List[ETFItem]) -> List[Any]:
    """
    Adds a critical parallel node with 5 critical children to check completeness per ETF.
    Returns the list of per-ETF completeness nodes for use as prerequisites.
    """
    comp_parent = evaluator.add_parallel(
        id="Complete_Information_For_Each_ETF",
        desc=("For each of the 5 ETFs, provide: ticker symbol, issuer name, current expense ratio (%), "
              "AUM (in billions USD), custodian model description, and at least one reference URL"),
        parent=parent,
        critical=True
    )

    comp_nodes = []
    for i, etf in enumerate(etfs):
        has_all = (
            (etf.ticker is not None and str(etf.ticker).strip() != "") and
            (etf.issuer is not None and str(etf.issuer).strip() != "") and
            (etf.expense_ratio is not None and str(etf.expense_ratio).strip() != "") and
            (etf.aum_billion_usd is not None and str(etf.aum_billion_usd).strip() != "") and
            (etf.custodian_model is not None and str(etf.custodian_model).strip() != "") and
            (isinstance(etf.reference_urls, list) and len([u for u in etf.reference_urls if isinstance(u, str) and u.strip()]) > 0)
        )
        node = evaluator.add_custom_node(
            result=has_all,
            id=f"Complete_Info_ETF_{i}",
            desc=f"ETF #{i+1} has all required fields and at least one reference URL",
            parent=comp_parent,
            critical=True
        )
        comp_nodes.append(node)
    return comp_nodes


async def add_spot_bitcoin_validity_nodes(
    evaluator: Evaluator,
    parent,
    etfs: List[ETFItem],
    comp_nodes: List[Any]
) -> None:
    """
    Adds a critical parallel node with 5 critical leaves to verify each ETF is a spot Bitcoin ETF.
    """
    spot_parent = evaluator.add_parallel(
        id="Spot_Bitcoin_ETF_Validity",
        desc="Each identified fund is a spot Bitcoin ETF",
        parent=parent,
        critical=True
    )

    batch: List[Tuple[str, List[str], Any, Optional[str]]] = []
    for i, etf in enumerate(etfs):
        leaf = evaluator.add_leaf(
            id=f"ETF_{i}_Is_Spot_Bitcoin",
            desc=f"ETF #{i+1} ({etf.ticker or 'Unknown ticker'}) is a spot Bitcoin ETF (physically backed, not futures-based)",
            parent=spot_parent,
            critical=True
        )
        claim = (
            f"The ETF {etf.ticker or 'Unknown'} is a spot Bitcoin ETF that holds Bitcoin directly (physically-backed), "
            f"not a futures-based product."
        )
        urls = etf.reference_urls or []
        add_ins = (
            "Check the provided official/credible sources for explicit wording like 'spot Bitcoin ETF', "
            "'physically backed', 'holds bitcoin'. Do not rely on the answer text; rely on the URLs."
        )
        batch.append((claim, urls, leaf, add_ins))

    await evaluator.batch_verify(batch)


async def add_sec_approved_us_trading_nodes(
    evaluator: Evaluator,
    parent,
    etfs: List[ETFItem],
    comp_nodes: List[Any]
) -> None:
    """
    Adds a critical parallel node with 5 critical leaves to verify each ETF is SEC-approved and trading on US exchanges.
    """
    sec_parent = evaluator.add_parallel(
        id="SEC_Approved_And_US_Trading",
        desc="Each identified ETF is SEC-approved and currently trading on U.S. exchanges",
        parent=parent,
        critical=True
    )

    batch: List[Tuple[str, List[str], Any, Optional[str]]] = []
    for i, etf in enumerate(etfs):
        leaf = evaluator.add_leaf(
            id=f"ETF_{i}_SEC_US",
            desc=f"ETF #{i+1} ({etf.ticker or 'Unknown ticker'}) is SEC-approved and trading on a U.S. exchange",
            parent=sec_parent,
            critical=True
        )
        claim = (
            f"The ETF {etf.ticker or 'Unknown'} is approved by the U.S. SEC and is currently listed/trading on a U.S. "
            f"exchange (e.g., NYSE Arca, Nasdaq, Cboe BZX)."
        )
        urls = etf.reference_urls or []
        add_ins = (
            "Look for explicit mentions of SEC approval and the U.S. listing exchange name. "
            "If pages are irrelevant or do not support the claim, return not supported."
        )
        batch.append((claim, urls, leaf, add_ins))

    await evaluator.batch_verify(batch)


async def add_fee_criteria_nodes(evaluator: Evaluator, parent, etfs: List[ETFItem]) -> None:
    """
    Adds two critical leaves verifying the 'at least one' fee thresholds using all provided sources.
    """
    all_sources = collect_all_sources(etfs)

    # Low fee: <= 0.20%
    low_leaf = evaluator.add_leaf(
        id="Low_Fee_Criterion",
        desc="At least one of the 5 ETFs has an expense ratio of 0.20% or lower",
        parent=parent,
        critical=True
    )
    low_claim = (
        "Among the five ETFs listed in the answer, at least one has an expense ratio of 0.20% or lower."
    )
    await evaluator.verify(
        claim=low_claim,
        node=low_leaf,
        sources=all_sources,
        additional_instruction=(
            "Check the expense ratio values from the provided URLs for each ETF. "
            "Treat '20 bps' as 0.20%. If any ETF meets 0.20% or below, mark supported; otherwise not supported."
        )
    )

    # High fee: >= 0.25%
    high_leaf = evaluator.add_leaf(
        id="High_Fee_Criterion",
        desc="At least one of the 5 ETFs has an expense ratio of 0.25% or higher",
        parent=parent,
        critical=True
    )
    high_claim = (
        "Among the five ETFs listed in the answer, at least one has an expense ratio of 0.25% or higher."
    )
    await evaluator.verify(
        claim=high_claim,
        node=high_leaf,
        sources=all_sources,
        additional_instruction=(
            "Check the expense ratio values from the provided URLs for each ETF. "
            "Treat '25 bps' as 0.25%. If any ETF meets 0.25% or higher, mark supported; otherwise not supported."
        )
    )


async def add_top5_aum_criterion_node(evaluator: Evaluator, parent, etfs: List[ETFItem]) -> None:
    """
    Adds a single critical leaf verifying that at least one of the ETFs is in the top 5 by AUM.
    Uses all sources combined to allow ranking pages or news sources to satisfy the claim.
    """
    all_sources = collect_all_sources(etfs)
    tickers_str = ", ".join([_normalize_ticker(e.ticker) or "Unknown" for e in etfs])

    top5_leaf = evaluator.add_leaf(
        id="Top5_AUM_Criterion",
        desc="At least one of the 5 ETFs ranks among the top 5 by assets under management",
        parent=parent,
        critical=True
    )
    top5_claim = (
        f"Among the five ETFs listed ({tickers_str}), at least one ranks within the top 5 by assets under management "
        f"among U.S. spot Bitcoin ETFs (as of the dates of the cited sources)."
    )
    await evaluator.verify(
        claim=top5_claim,
        node=top5_leaf,
        sources=all_sources,
        additional_instruction=(
            "Use any credible ranking or comparative AUM article/page from the provided URLs. "
            "Look for explicit 'top 5' status or ranking tables showing the ETF is within top five by AUM. "
            "If no such evidence is present, mark not supported."
        )
    )


async def add_multi_custodian_criterion_node(evaluator: Evaluator, parent, etfs: List[ETFItem]) -> None:
    """
    Adds a single critical leaf verifying at least one ETF uses more than one custodian.
    """
    all_sources = collect_all_sources(etfs)
    multi_leaf = evaluator.add_leaf(
        id="Multi_Custodian_Criterion",
        desc="At least one of the 5 ETFs employs a multi-custodian model (uses more than one custodian)",
        parent=parent,
        critical=True
    )
    multi_claim = (
        "At least one of the five ETFs uses more than one custodian (multi-custodian model), e.g., 'co-custodians', "
        "'two custodians', or listing two or more distinct custodian names."
    )
    await evaluator.verify(
        claim=multi_claim,
        node=multi_leaf,
        sources=all_sources,
        additional_instruction=(
            "From the provided URLs, confirm language indicating multiple custodians (e.g., 'custodians: A and B', "
            "'co-custodians', 'multi-custodian', or any explicit listing of 2+ custodian names). "
            "If no clear evidence, mark not supported."
        )
    )


async def add_not_coinbase_sole_custodian_node(evaluator: Evaluator, parent, etfs: List[ETFItem]) -> None:
    """
    Adds a single critical leaf verifying at least one ETF does NOT use Coinbase as its sole custodian.
    """
    all_sources = collect_all_sources(etfs)
    not_coinbase_leaf = evaluator.add_leaf(
        id="Not_Coinbase_Sole_Custodian_Criterion",
        desc="At least one of the 5 ETFs does NOT use Coinbase as its sole custodian",
        parent=parent,
        critical=True
    )
    not_coinbase_claim = (
        "At least one of the five ETFs does not use Coinbase as its sole custodian (i.e., either uses a different sole "
        "custodian or uses Coinbase along with at least one additional custodian)."
    )
    await evaluator.verify(
        claim=not_coinbase_claim,
        node=not_coinbase_leaf,
        sources=all_sources,
        additional_instruction=(
            "Check custodial information from the provided URLs. If any ETF shows a sole custodian other than Coinbase, "
            "or lists Coinbase plus one or more additional custodians, the claim is supported."
        )
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
    Evaluate an answer for the U.S. spot Bitcoin ETFs criteria task using the obj_task_eval framework.
    """
    # Initialize evaluator with a critical parallel root as specified
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
    # IMPORTANT: Root is non-critical by default in Evaluator.initialize; We need to override by creating a critical wrapper node
    # However, VerificationNode enforces child critical consistency only when parent is critical. To adhere to rubric's "Root critical",
    # we add a critical wrapper node under the default root to act as the true root for criteria aggregation.
    critical_root = evaluator.add_parallel(
        id="Root",
        desc="Identify 5 distinct U.S.-traded, SEC-approved spot Bitcoin ETFs that collectively meet all specified criteria and provide all required fields with sources",
        parent=root,
        critical=True
    )

    # 1) Extract structured ETF list from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_etfs(),
        template_class=ETFListExtraction,
        extraction_name="etf_list_extraction"
    )

    # 2) Normalize to exactly first 5 ETFs (do not fabricate)
    etfs: List[ETFItem] = list(extracted.etfs[:5])
    # If fewer than 5 were provided, we still proceed (some criteria will fail accordingly)

    # Record a custom info snapshot
    evaluator.add_custom_info(
        {
            "extracted_count": len(etfs),
            "tickers": [e.ticker for e in etfs],
            "issuers": [e.issuer for e in etfs]
        },
        info_type="extraction_overview",
        info_name="extraction_overview"
    )

    # 3) Build verification nodes according to rubric

    # 3.1 Count and distinctness
    await add_count_and_distinctness_node(evaluator, critical_root, etfs)

    # 3.2 Completeness per ETF (we will reuse these nodes as preconditions conceptually for per-ETF checks)
    complete_nodes = await add_complete_information_nodes(evaluator, critical_root, etfs)

    # 3.3 Spot Bitcoin ETF validity (per ETF, critical)
    await add_spot_bitcoin_validity_nodes(evaluator, critical_root, etfs, complete_nodes)

    # 3.4 SEC-approved and US-trading (per ETF, critical)
    await add_sec_approved_us_trading_nodes(evaluator, critical_root, etfs, complete_nodes)

    # 3.5 Distinct issuers (all 5 different)
    await add_distinct_issuers_node(evaluator, critical_root, etfs)

    # 3.6 Low and High fee criteria (aggregated across sources)
    await add_fee_criteria_nodes(evaluator, critical_root, etfs)

    # 3.7 Top 5 by AUM criterion (aggregated check)
    await add_top5_aum_criterion_node(evaluator, critical_root, etfs)

    # 3.8 Multi-custodian criterion (aggregated check)
    await add_multi_custodian_criterion_node(evaluator, critical_root, etfs)

    # 3.9 Not Coinbase as sole custodian criterion (aggregated check)
    await add_not_coinbase_sole_custodian_node(evaluator, critical_root, etfs)

    # 4) Return evaluation summary
    return evaluator.get_summary()