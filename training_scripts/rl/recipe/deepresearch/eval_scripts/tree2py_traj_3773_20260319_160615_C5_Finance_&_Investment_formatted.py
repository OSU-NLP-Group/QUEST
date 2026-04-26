import asyncio
import logging
import re
from datetime import datetime, date
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy, VerificationNode
from obj_task_eval.utils.cache_filesys import CacheFileSys


# -----------------------------------------------------------------------------
# Task constants
# -----------------------------------------------------------------------------
TASK_ID = "franklin_crypto_etfs"
TASK_DESCRIPTION = (
    "Franklin Templeton, a major global asset management firm, has launched several spot cryptocurrency "
    "exchange-traded funds (ETFs) between January 2024 and November 2025. Identify at least three of these spot "
    "cryptocurrency ETFs. For each ETF, provide the following information: the official ETF name, the ticker symbol, "
    "the underlying cryptocurrency asset, the launch date, the U.S. exchange where it trades, and a direct link to "
    "Franklin Templeton's official website (either a press release or product page) that confirms the ETF details. "
    "Note: Focus only on spot ETFs (providing direct exposure to the cryptocurrency) that were launched during the "
    "specified timeframe, not futures-based or other derivative products."
)

DATE_RANGE_START = date(2024, 1, 1)
DATE_RANGE_END = date(2025, 11, 30)
ALLOWED_US_EXCHANGES_HINT = ["Cboe BZX", "NYSE Arca"]


# -----------------------------------------------------------------------------
# Extraction models
# -----------------------------------------------------------------------------
class ETFEntry(BaseModel):
    official_name: Optional[str] = None
    ticker: Optional[str] = None
    underlying: Optional[str] = None
    launch_date: Optional[str] = None
    exchange: Optional[str] = None
    ft_links: List[str] = Field(default_factory=list)


class ETFExtraction(BaseModel):
    etfs: List[ETFEntry] = Field(default_factory=list)


# -----------------------------------------------------------------------------
# Extraction prompt
# -----------------------------------------------------------------------------
def prompt_extract_etfs() -> str:
    return """
Extract up to five (5) Franklin Templeton spot cryptocurrency ETFs exactly as presented in the answer.
For each ETF, extract the following fields from the answer text:
- official_name: The official ETF name as written in the answer (string).
- ticker: The ETF ticker symbol (string).
- underlying: The single underlying cryptocurrency (e.g., Bitcoin, Ethereum) (string).
- launch_date: The ETF's launch date as written in the answer (any human-readable date string).
- exchange: The U.S. exchange where it trades (string), such as "Cboe BZX" or "NYSE Arca".
- ft_links: A list of direct URLs to Franklin Templeton's official website (press release or product page) that corroborate the ETF details.
  Only include URLs that are clearly from Franklin Templeton's official site (e.g., domains containing "franklintempleton.").

Rules:
- Do NOT invent any information; extract exactly what the answer provides.
- If a field is missing for an ETF, set it to null (for strings) or [] for ft_links.
- If the answer lists more than five ETFs, extract only the first five mentioned.
- If the answer includes links that are not Franklin Templeton official pages, exclude them from ft_links.

Return a JSON object with a single property:
{
  "etfs": [ { official_name, ticker, underlying, launch_date, exchange, ft_links }, ... ]
}
    """.strip()


# -----------------------------------------------------------------------------
# Helper utilities
# -----------------------------------------------------------------------------
MONTHS_MAP = {
    "jan": 1, "january": 1,
    "feb": 2, "february": 2,
    "mar": 3, "march": 3,
    "apr": 4, "april": 4,
    "may": 5,
    "jun": 6, "june": 6,
    "jul": 7, "july": 7,
    "aug": 8, "august": 8,
    "sep": 9, "sept": 9, "september": 9,
    "oct": 10, "october": 10,
    "nov": 11, "november": 11,
    "dec": 12, "december": 12,
}


def _try_parse_int(s: str) -> Optional[int]:
    try:
        return int(s)
    except Exception:
        return None


def parse_date_loose(s: Optional[str]) -> Optional[date]:
    """
    Parse a variety of common human-readable date formats into a date object.
    If only year-month is available, default day=15.
    """
    if not s:
        return None
    text = s.strip()

    # Direct ISO-like formats
    for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%m/%d/%Y"):
        try:
            return datetime.strptime(text, fmt).date()
        except Exception:
            pass

    # Handle "Month DD, YYYY" or "Mon DD, YYYY"
    m = re.match(r"^\s*([A-Za-z]+)\s+(\d{1,2}),\s*(\d{4})\s*$", text)
    if m:
        mon = MONTHS_MAP.get(m.group(1).lower())
        day = _try_parse_int(m.group(2))
        yr = _try_parse_int(m.group(3))
        if mon and day and yr:
            try:
                return date(yr, mon, day)
            except Exception:
                return None

    # Handle "Month YYYY" or "Mon YYYY"
    m = re.match(r"^\s*([A-Za-z]+)\s+(\d{4})\s*$", text)
    if m:
        mon = MONTHS_MAP.get(m.group(1).lower())
        yr = _try_parse_int(m.group(2))
        if mon and yr:
            try:
                return date(yr, mon, 15)
            except Exception:
                return None

    # Handle "YYYY-MM" or "YYYY/MM"
    m = re.match(r"^\s*(\d{4})[-/](\d{1,2})\s*$", text)
    if m:
        yr = _try_parse_int(m.group(1))
        mon = _try_parse_int(m.group(2))
        if yr and mon and 1 <= mon <= 12:
            try:
                return date(yr, mon, 15)
            except Exception:
                return None

    # Fallback: Try "Mon DD YYYY" without comma
    m = re.match(r"^\s*([A-Za-z]+)\s+(\d{1,2})\s+(\d{4})\s*$", text)
    if m:
        mon = MONTHS_MAP.get(m.group(1).lower())
        day = _try_parse_int(m.group(2))
        yr = _try_parse_int(m.group(3))
        if mon and day and yr:
            try:
                return date(yr, mon, day)
            except Exception:
                return None

    # Final: Try just year
    m = re.match(r"^\s*(\d{4})\s*$", text)
    if m:
        yr = _try_parse_int(m.group(1))
        if yr:
            return date(yr, 6, 15)

    return None


def is_in_range(d: Optional[date], start: date, end: date) -> bool:
    if d is None:
        return False
    return start <= d <= end


def is_single_crypto(underlying: Optional[str]) -> bool:
    if not underlying:
        return False
    s = underlying.strip().lower()
    if not s:
        return False
    # Reject obvious multi-asset phrases
    multi_markers = [",", " and ", "/", "&", "basket", "index", "indices", "multiple", "various"]
    return not any(m in s for m in multi_markers)


def exchange_allowed(exchange: Optional[str]) -> bool:
    if not exchange:
        return False
    s = exchange.strip().lower()
    if not s:
        return False
    # Accept common variants
    if "cboe" in s or "bzx" in s:
        return True
    if "nyse arca" in s or "arca" in s:
        return True
    return False


def filter_ft_urls(urls: List[str]) -> List[str]:
    valid = []
    for u in urls:
        if isinstance(u, str) and "franklintempleton." in u.lower():
            valid.append(u)
    return valid


def safe_str(v: Optional[str]) -> str:
    return v if isinstance(v, str) else ""


# -----------------------------------------------------------------------------
# Verification builders
# -----------------------------------------------------------------------------
async def verify_one_etf(
    evaluator: Evaluator,
    parent: VerificationNode,
    etf: ETFEntry,
    idx: int,
) -> VerificationNode:
    """
    Build and run verification for a single ETF entry.
    Returns the entry node.
    """
    entry_num = idx + 1
    entry_node = evaluator.add_parallel(
        id=f"ETF_Entry_{entry_num}",
        desc=f"{['First','Second','Third','Fourth','Fifth'][idx] if idx < 5 else f'#{entry_num}th'} ETF entry is a fully qualifying Franklin Templeton spot crypto ETF with complete required details.",
        parent=parent,
        critical=False
    )

    name = safe_str(etf.official_name)
    ticker = safe_str(etf.ticker)
    underlying = safe_str(etf.underlying)
    launch_date_str = safe_str(etf.launch_date)
    exchange_str = safe_str(etf.exchange)

    # Prepare FT links
    ft_urls_all = etf.ft_links or []
    ft_urls = filter_ft_urls(ft_urls_all)

    # 1) FT_Official_Link_Provided_And_Relevant (critical)
    if not ft_urls:
        evaluator.add_custom_node(
            result=False,
            id=f"ETF_{entry_num}_FT_Official_Link_Provided_And_Relevant",
            desc="A direct Franklin Templeton official website link (press release or product page) is provided and corroborates key ETF details",
            parent=entry_node,
            critical=True
        )
        ft_link_leaf = evaluator.find_node(f"ETF_{entry_num}_FT_Official_Link_Provided_And_Relevant")
    else:
        ft_link_leaf = evaluator.add_leaf(
            id=f"ETF_{entry_num}_FT_Official_Link_Provided_And_Relevant",
            desc="A direct Franklin Templeton official website link (press release or product page) is provided and corroborates key ETF details",
            parent=entry_node,
            critical=True
        )

        # Build a flexible claim: confirm at least name/ticker, and that it is a spot crypto ETF tied to the stated underlying if provided.
        parts = []
        if name:
            parts.append(f"the ETF name is '{name}'")
        if ticker:
            parts.append(f"the ticker is '{ticker}'")
        if underlying:
            parts.append(f"it provides spot (direct) exposure to {underlying}")
        parts_text = "; ".join(parts) if parts else "it is the correct ETF product page"
        claim = (
            f"This is an official Franklin Templeton page for the ETF. The page confirms that {parts_text}. "
            f"It is acceptable if the page is either a dedicated product page or a Franklin Templeton press release."
        )

        await evaluator.verify(
            claim=claim,
            node=ft_link_leaf,
            sources=ft_urls,
            additional_instruction=(
                "Treat name/ticker matches leniently (minor punctuation/casing/spacing differences are okay). "
                "For the underlying, confirm that the ETF provides spot/direct exposure to the specified cryptocurrency "
                "rather than futures exposure. If multiple FT pages are provided, any one that clearly establishes these "
                "details is sufficient."
            ),
        )

    # 2) Issuer_Is_Franklin_Templeton (critical, relies on FT link)
    issuer_leaf = evaluator.add_leaf(
        id=f"ETF_{entry_num}_Issuer_Is_Franklin_Templeton",
        desc="ETF is issued/managed by Franklin Templeton (as specified in constraints).",
        parent=entry_node,
        critical=True
    )
    await evaluator.verify(
        claim="This ETF is issued or managed by Franklin Templeton.",
        node=issuer_leaf,
        sources=ft_urls if ft_urls else None,
        additional_instruction="Verify directly on the provided Franklin Templeton page that Franklin Templeton is the issuer/manager.",
        extra_prerequisites=[ft_link_leaf] if ft_link_leaf else None,
    )

    # 3) Spot_Not_Futures (critical, relies on FT link)
    spot_leaf = evaluator.add_leaf(
        id=f"ETF_{entry_num}_Spot_Not_Futures",
        desc="ETF is spot (direct exposure) and not futures-based or another derivative product.",
        parent=entry_node,
        critical=True
    )
    spot_claim = (
        f"This ETF is a spot cryptocurrency ETF that provides direct exposure"
        f"{f' to {underlying}' if underlying else ''}, not a futures-based product."
    )
    await evaluator.verify(
        claim=spot_claim,
        node=spot_leaf,
        sources=ft_urls if ft_urls else None,
        additional_instruction="Look for explicit wording such as 'spot', 'physically backed', or 'provides direct exposure', and the absence of futures-based structure.",
        extra_prerequisites=[ft_link_leaf] if ft_link_leaf else None,
    )

    # 4) SEC_Approved_And_US_Listed (critical, relies on FT link)
    sec_leaf = evaluator.add_leaf(
        id=f"ETF_{entry_num}_SEC_Approved_And_US_Listed",
        desc="ETF has received SEC approval and is listed for trading on a U.S. exchange.",
        parent=entry_node,
        critical=True
    )
    await evaluator.verify(
        claim="This ETF received the necessary SEC approval and is listed for trading on a U.S. exchange (e.g., Cboe BZX or NYSE Arca).",
        node=sec_leaf,
        sources=ft_urls if ft_urls else None,
        additional_instruction="Accept wording that clearly indicates regulatory approval for U.S. listing/trading, or explicit mention that it trades on a U.S. exchange.",
        extra_prerequisites=[ft_link_leaf] if ft_link_leaf else None,
    )

    # 5) Launch_Date_In_Range (critical, local check for range; details corroborated by FT link in previous step)
    parsed = parse_date_loose(launch_date_str)
    in_range = is_in_range(parsed, DATE_RANGE_START, DATE_RANGE_END)
    evaluator.add_custom_node(
        result=(parsed is not None and in_range),
        id=f"ETF_{entry_num}_Launch_Date_In_Range",
        desc="Launch date is provided and falls between January 2024 and November 2025 (inclusive).",
        parent=entry_node,
        critical=True
    )

    # 6) US_Exchange_Provided_And_Allowed (critical, local allowed-set check)
    evaluator.add_custom_node(
        result=exchange_allowed(exchange_str),
        id=f"ETF_{entry_num}_US_Exchange_Provided_And_Allowed",
        desc="Trading exchange is provided and is an allowed U.S. exchange per constraints (Cboe BZX Exchange or NYSE Arca).",
        parent=entry_node,
        critical=True
    )

    # 7) Official_Name_Provided (critical, existence)
    evaluator.add_custom_node(
        result=(bool(name.strip())),
        id=f"ETF_{entry_num}_Official_Name_Provided",
        desc="Official ETF name is provided.",
        parent=entry_node,
        critical=True
    )

    # 8) Ticker_Provided (critical, existence)
    evaluator.add_custom_node(
        result=(bool(ticker.strip())),
        id=f"ETF_{entry_num}_Ticker_Provided",
        desc="Official ticker symbol is provided.",
        parent=entry_node,
        critical=True
    )

    # 9) Single_Crypto_Underlying (critical, local sanity)
    evaluator.add_custom_node(
        result=is_single_crypto(underlying),
        id=f"ETF_{entry_num}_Single_Crypto_Underlying",
        desc="Underlying cryptocurrency exposure is identified and is exactly one specific cryptocurrency (not a basket).",
        parent=entry_node,
        critical=True
    )

    return entry_node


# -----------------------------------------------------------------------------
# Main evaluation entry point
# -----------------------------------------------------------------------------
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
    Evaluate an answer for the Franklin Templeton spot crypto ETFs task.
    """
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
        extraction_name="extracted_etfs",
    )

    # Ensure exactly up to 5 entries for evaluation (pad if fewer)
    etfs = (extraction.etfs or [])[:5]
    while len(etfs) < 3:  # We need at least 3 entries attempted; pad with empty for structured feedback
        etfs.append(ETFEntry())
    # If 4th/5th are missing, it's fine; we evaluate as many as present (up to 5)
    while len(etfs) < 5:
        etfs.append(ETFEntry())

    # Build top-level rubric node (non-critical to satisfy framework constraints)
    main_node = evaluator.add_parallel(
        id="Franklin_Templeton_Crypto_ETFs_Identification",
        desc=("Provide at least three Franklin Templeton spot cryptocurrency ETFs launched between Jan 2024 and Nov 2025, "
              "each with all required verifiable details and an official Franklin Templeton link corroborating them."),
        parent=root,
        critical=False
    )

    # Verify each ETF entry
    entry_nodes: List[VerificationNode] = []
    for i in range(5):
        # Create nodes only for indices that correspond to provided items (placeholders will mostly fail)
        node = await verify_one_etf(evaluator, main_node, etfs[i], i)
        entry_nodes.append(node)

    # Force compute to update current statuses before counting
    if evaluator.root:
        evaluator.root.compute_score(mutate=True)

    # Count qualifying entries: those whose entry node has all critical children passed → aggregated_score == 1.0
    qualifying_count = sum(1 for n in entry_nodes if n.aggregated_score == 1.0)

    # Add the "At least 3" critical gate
    evaluator.add_custom_node(
        result=(qualifying_count >= 3),
        id="At_Least_3_Qualifying_ETF_Entries",
        desc="At least three of the ETF_Entry nodes included in the response pass all their critical subchecks (fully qualifying).",
        parent=main_node,
        critical=True
    )

    # Add task context info
    evaluator.add_custom_info(
        {
            "allowed_us_exchanges": ALLOWED_US_EXCHANGES_HINT,
            "date_range_inclusive": {
                "start": DATE_RANGE_START.isoformat(),
                "end": DATE_RANGE_END.isoformat(),
            },
            "qualifying_entries_detected": qualifying_count,
        },
        info_type="constraints",
        info_name="task_constraints_and_stats"
    )

    return evaluator.get_summary()