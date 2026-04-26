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
TASK_ID = "sol_etf_lowest_coinbase_2025"
TASK_DESCRIPTION = (
    "Among the Solana exchange-traded funds (ETFs) that began trading on October 28, 2025, "
    "identify the one with the lowest expense ratio (after the promotional period ends) that uses "
    "Coinbase Custody Trust Company, LLC as its cryptocurrency custodian. Provide the following "
    "information: (1) The fund's ticker symbol, (2) The standard expense ratio (applicable after "
    "promotional waiver expires), and (3) The specific duration or conditions of the promotional fee waiver."
)


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class ETFInfo(BaseModel):
    fund_name: Optional[str] = None
    ticker: Optional[str] = None
    launch_date: Optional[str] = None  # e.g., "October 28, 2025" or "10/28/2025"
    custodian: Optional[str] = None  # e.g., "Coinbase Custody Trust Company, LLC"
    standard_expense_ratio: Optional[str] = None  # Post-promo/standard ER, keep as string (e.g., "0.19%")
    fee_waiver_details: Optional[str] = None  # Free text description of promo duration/conditions
    sources: List[str] = Field(default_factory=list)  # URLs directly cited for this ETF


class SolanaETFSelection(BaseModel):
    selected: Optional[ETFInfo] = None  # The ETF the answer claims is the lowest-cost qualifying fund
    other_etfs: List[ETFInfo] = Field(default_factory=list)  # Other Solana ETFs mentioned
    all_urls: List[str] = Field(default_factory=list)  # All URLs mentioned anywhere in the answer


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_selection() -> str:
    return """
    From the answer, extract the Solana ETF that the answer identifies as the one with the lowest expense ratio
    (after any promotional/temporary waiver ends) among Solana ETFs that began trading on October 28, 2025 and
    that use Coinbase Custody Trust Company, LLC as the cryptocurrency custodian. Call this the 'selected' ETF.

    If multiple funds are discussed, set 'selected' to the fund that the answer explicitly claims is the lowest-cost
    among those meeting the stated criteria. If the answer does not clearly declare a single lowest-cost pick but lists
    multiple candidates, choose the first one that is presented by the answer as meeting the criteria. If no ETF seems
    to be selected, set 'selected' to null.

    For the 'selected' ETF and for each of the 'other_etfs' mentioned in the answer, extract:
    - fund_name: The ETF's name as stated
    - ticker: The ticker symbol
    - launch_date: The launch/inception/listing date as stated
    - custodian: The fund’s cryptocurrency custodian as stated
    - standard_expense_ratio: The standard/post-promo expense ratio (i.e., the rate that applies after any temporary or promotional waivers end)
    - fee_waiver_details: The descriptive text of the promotional fee waiver period and/or any conditions, as stated in the answer
    - sources: All URLs in the answer that are directly about this specific ETF (official website, prospectus, press release, factsheet, SEC filing, news coverage, etc.)

    Additionally, extract:
    - all_urls: A list of every URL mentioned anywhere in the answer (including those already used in 'sources').

    Important:
    - Do not invent information. Only extract what is explicitly stated in the answer.
    - If any field is missing, set it to null (or an empty list for sources).
    - For URLs, include only valid, complete URLs (prepend http:// if protocol missing).
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _dedup_urls(urls: List[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for u in urls:
        if not u:
            continue
        v = u.strip()
        if not v:
            continue
        if not (v.startswith("http://") or v.startswith("https://")):
            v = "http://" + v
        if v not in seen:
            seen.add(v)
            out.append(v)
    return out


def _fmt_selected_ident(etf: Optional[ETFInfo]) -> str:
    if not etf:
        return "the selected ETF"
    if etf.fund_name and etf.ticker:
        return f"{etf.fund_name} ({etf.ticker})"
    if etf.ticker:
        return f"ticker {etf.ticker}"
    if etf.fund_name:
        return etf.fund_name
    return "the selected ETF"


def _collect_selected_sources(selection: SolanaETFSelection) -> List[str]:
    sel_sources = selection.selected.sources if selection.selected and selection.selected.sources else []
    return _dedup_urls(sel_sources)


def _collect_all_sources(selection: SolanaETFSelection) -> List[str]:
    urls: List[str] = []
    if selection.selected and selection.selected.sources:
        urls.extend(selection.selected.sources)
    for e in selection.other_etfs:
        urls.extend(e.sources or [])
    urls.extend(selection.all_urls or [])
    return _dedup_urls(urls)


# --------------------------------------------------------------------------- #
# Verification builder                                                        #
# --------------------------------------------------------------------------- #
async def build_and_verify(evaluator: Evaluator, root_node, selection: SolanaETFSelection) -> None:
    selected = selection.selected or ETFInfo()

    # Prepare sources
    selected_sources = _collect_selected_sources(selection)
    all_sources = _collect_all_sources(selection)

    # 1) Launch date verification
    node_launch = evaluator.add_leaf(
        id="launch_date_verification",
        desc="The identified ETF launched on October 28, 2025 (the first day spot Solana ETFs became available for trading)",
        parent=root_node,
        critical=True,
    )
    claim_launch = f"{_fmt_selected_ident(selected)} launched (or has inception/listing date) on October 28, 2025."
    await evaluator.verify(
        claim=claim_launch,
        node=node_launch,
        sources=selected_sources if selected_sources else all_sources,
        additional_instruction=(
            "Treat 'inception date', 'launch date', or 'listing date' as acceptable evidence of first trading day. "
            "Judge as not supported if the page does not clearly indicate October 28, 2025."
        ),
    )

    # 2) Custodian verification
    node_custody = evaluator.add_leaf(
        id="custodian_verification",
        desc="The identified ETF uses Coinbase Custody Trust Company, LLC as its cryptocurrency custodian",
        parent=root_node,
        critical=True,
    )
    claim_custody = f"{_fmt_selected_ident(selected)} uses Coinbase Custody Trust Company, LLC as its cryptocurrency custodian."
    await evaluator.verify(
        claim=claim_custody,
        node=node_custody,
        sources=selected_sources if selected_sources else all_sources,
        additional_instruction=(
            "Accept minor name variants like 'Coinbase Custody Trust Co., LLC' or 'Coinbase Custody'. "
            "The page should explicitly indicate Coinbase Custody Trust Company, LLC (or a clear equivalent) as the crypto custodian."
        ),
    )

    # 3) Expense ratio accuracy (post-promo)
    node_expense = evaluator.add_leaf(
        id="expense_ratio_accuracy",
        desc="The provided expense ratio correctly reflects the fund's expense ratio after the promotional fee waiver period ends",
        parent=root_node,
        critical=True,
    )
    er_text = selected.standard_expense_ratio or "UNKNOWN"
    claim_expense = (
        f"The standard (post-promo) expense ratio for {_fmt_selected_ident(selected)} is {er_text}."
    )
    await evaluator.verify(
        claim=claim_expense,
        node=node_expense,
        sources=selected_sources if selected_sources else all_sources,
        additional_instruction=(
            "Verify the expense ratio that applies AFTER any temporary/promotional waiver ends. "
            "If a page shows both a temporarily waived 'net' fee and a higher 'gross' or 'standard' fee, the claim refers to the latter. "
            "Allow minor formatting or rounding differences (e.g., 0.10% vs 0.1%). If the post-promo/standard rate is not clearly shown, mark as not supported."
        ),
    )

    # 4) Ticker symbol accuracy
    node_ticker = evaluator.add_leaf(
        id="ticker_symbol_accuracy",
        desc="The provided ticker symbol correctly identifies the fund",
        parent=root_node,
        critical=True,
    )
    ticker_text = selected.ticker or "UNKNOWN"
    claim_ticker = f"The ticker symbol for {_fmt_selected_ident(selected)} is {ticker_text}."
    await evaluator.verify(
        claim=claim_ticker,
        node=node_ticker,
        sources=selected_sources if selected_sources else all_sources,
        additional_instruction=(
            "Confirm the ETF's ticker symbol on the official fund page, factsheet, prospectus, or trusted listings. "
            "If the ticker cannot be found or differs, mark as not supported."
        ),
    )

    # 5) Promotional fee waiver details
    node_waiver = evaluator.add_leaf(
        id="fee_waiver_details",
        desc="The answer accurately describes the promotional fee waiver duration and/or conditions (e.g., time period or AUM threshold)",
        parent=root_node,
        critical=True,
    )
    waiver_text = selected.fee_waiver_details or "UNKNOWN"
    claim_waiver = f"The promotional fee waiver for {_fmt_selected_ident(selected)} is described as: '{waiver_text}'."
    await evaluator.verify(
        claim=claim_waiver,
        node=node_waiver,
        sources=selected_sources if selected_sources else all_sources,
        additional_instruction=(
            "Check footnotes or disclosures for exact waiver details (dates, duration, AUM caps, or other conditions). "
            "Paraphrasing is acceptable if it conveys the same constraints. If waiver details are missing or materially different, mark as not supported."
        ),
    )

    # 6) Lowest-cost verification among qualifying ETFs
    node_lowest = evaluator.add_leaf(
        id="lowest_cost_verification",
        desc="The identified ETF has the lowest expense ratio among all Solana ETFs meeting the specified criteria (Oct 28 launch date and Coinbase custody)",
        parent=root_node,
        critical=True,
    )
    # Build claim referring to the explicit conditions
    lowest_target = selected.ticker or selected.fund_name or "the selected ETF"
    claim_lowest = (
        f"Among Solana ETFs that began trading on October 28, 2025 and use Coinbase Custody Trust Company, LLC as custodian, "
        f"{lowest_target} has the lowest expense ratio that applies after the promotional waiver ends."
    )
    await evaluator.verify(
        claim=claim_lowest,
        node=node_lowest,
        sources=all_sources,  # Use all sources, including competitor pages or comparative news/articles
        additional_instruction=(
            "Prefer a source that explicitly states this 'lowest expense ratio' comparison across qualifying Solana ETFs. "
            "If no single page makes this comparative claim clearly, mark as not supported. "
            "Do not infer by comparing numbers across multiple pages unless the page explicitly summarizes the comparison."
        ),
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
    model: str = "o4-mini",
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
        default_model=model,
    )

    # Extraction
    selection: SolanaETFSelection = await evaluator.extract(
        prompt=prompt_extract_selection(),
        template_class=SolanaETFSelection,
        extraction_name="solana_etf_selection",
    )

    # Optional: record a concise summary for debugging
    selected_summary = {
        "fund_name": selection.selected.fund_name if selection.selected else None,
        "ticker": selection.selected.ticker if selection.selected else None,
        "launch_date": selection.selected.launch_date if selection.selected else None,
        "custodian": selection.selected.custodian if selection.selected else None,
        "standard_expense_ratio": selection.selected.standard_expense_ratio if selection.selected else None,
        "fee_waiver_details": selection.selected.fee_waiver_details if selection.selected else None,
        "num_selected_sources": len(selection.selected.sources) if selection.selected and selection.selected.sources else 0,
        "num_other_etfs": len(selection.other_etfs or []),
        "num_all_urls": len(selection.all_urls or []),
    }
    evaluator.add_custom_info(selected_summary, info_type="extraction_summary", info_name="selected_etf_summary")

    # Build verification tree per rubric and run checks
    await build_and_verify(evaluator, root, selection)

    return evaluator.get_summary()