import asyncio
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.verification_tree import AggregationStrategy


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "xrp_etf_selection"
TASK_DESCRIPTION = (
    "You are an investment analyst preparing a recommendation report for clients interested in gaining exposure to XRP "
    "through SEC-approved exchange-traded funds (ETFs). Your clients have specified the following investment criteria: "
    "(1) The ETF must be approved by the U.S. Securities and Exchange Commission (SEC) and actively trading on a major "
    "U.S. stock exchange; (2) The stated annual expense ratio (not including temporary promotional waivers) must be "
    "0.50% or lower; (3) The ETF must currently offer a fee waiver or promotional reduced-fee period that extends beyond "
    "March 31, 2026; (4) The ETF must be listed on either the New York Stock Exchange (NYSE) or NASDAQ. Identify at least "
    "two distinct XRP ETFs that meet all of these criteria. For each qualifying ETF, provide: the complete official name "
    "of the ETF, the issuing asset management company, the official ticker symbol, the exchange where it is listed (NYSE or "
    "NASDAQ), the stated annual expense ratio, the terms and end date of any current fee waiver or promotional period, and "
    "verifiable URLs documenting the ETF's SEC approval status, exchange listing, and cost structure."
)

WAIVER_DEADLINE_STR = "March 31, 2026"


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class ETFSourceDocs(BaseModel):
    sec_approval_urls: List[str] = Field(default_factory=list)
    exchange_listing_urls: List[str] = Field(default_factory=list)
    cost_structure_urls: List[str] = Field(default_factory=list)


class FeeWaiverInfo(BaseModel):
    terms: Optional[str] = None
    end_date: Optional[str] = None
    waiver_urls: List[str] = Field(default_factory=list)


class ETFExtraction(BaseModel):
    official_name: Optional[str] = None
    issuer: Optional[str] = None
    ticker: Optional[str] = None
    exchange: Optional[str] = None  # e.g., "NYSE", "NYSE Arca", "NASDAQ"
    stated_expense_ratio: Optional[str] = None  # keep as string (e.g., "0.49%")
    fee_waiver: Optional[FeeWaiverInfo] = None
    sources: Optional[ETFSourceDocs] = None


class ETFListExtraction(BaseModel):
    etfs: List[ETFExtraction] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_etfs() -> str:
    return """
    Extract up to the first 4 XRP ETFs mentioned in the answer that are intended to meet the client's criteria.
    For each ETF, extract these fields exactly as stated in the answer text:

    - official_name: The complete official fund name.
    - issuer: The issuing asset management company (sponsor).
    - ticker: The official trading ticker symbol (letters only).
    - exchange: The named exchange where it is listed (acceptable examples include "NYSE", "NYSE Arca", or "NASDAQ").
    - stated_expense_ratio: The stated annual expense ratio excluding any temporary promotional waivers (keep as a string, e.g., "0.49%").
    - fee_waiver:
        - terms: The described fee waiver or promotional reduced-fee terms (e.g., "Sponsor fee waived to 0.00%").
        - end_date: The end date for the current fee waiver or promotional reduced-fee period, as written in the answer.
        - waiver_urls: All URLs in the answer that specifically document the waiver or promotional fee terms and/or end date.
    - sources:
        - sec_approval_urls: All URLs in the answer that document SEC approval or effectiveness for the ETF.
        - exchange_listing_urls: All URLs in the answer that show exchange listing and ticker (exchange website or equivalent authoritative listing page).
        - cost_structure_urls: All URLs in the answer that document the stated expense ratio or fee table (e.g., prospectus, summary prospectus, fact sheet).

    Strict URL extraction rules:
    - Extract only URLs explicitly present in the answer text (including markdown links). Do not invent or infer URLs.
    - Include full URLs. If protocol is missing, prepend http://.
    - If a field is missing, return null (for scalars) or an empty array (for URL lists).
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _dedup_urls(urls: List[str]) -> List[str]:
    seen = set()
    out = []
    for u in urls:
        if not u:
            continue
        if u not in seen:
            seen.add(u)
            out.append(u)
    return out


def _combined_sources(etf: ETFExtraction) -> List[str]:
    urls: List[str] = []
    if etf.sources:
        urls += etf.sources.sec_approval_urls or []
        urls += etf.sources.exchange_listing_urls or []
        urls += etf.sources.cost_structure_urls or []
    if etf.fee_waiver:
        urls += etf.fee_waiver.waiver_urls or []
    return _dedup_urls(urls)


def _listing_sources(etf: ETFExtraction) -> List[str]:
    urls: List[str] = []
    if etf.sources:
        urls += etf.sources.exchange_listing_urls or []
    return _dedup_urls(urls) or _combined_sources(etf)


def _sec_sources(etf: ETFExtraction) -> List[str]:
    urls: List[str] = []
    if etf.sources:
        urls += etf.sources.sec_approval_urls or []
    return _dedup_urls(urls) or _combined_sources(etf)


def _cost_sources(etf: ETFExtraction) -> List[str]:
    urls: List[str] = []
    if etf.sources:
        urls += etf.sources.cost_structure_urls or []
    if etf.fee_waiver:
        urls += etf.fee_waiver.waiver_urls or []
    return _dedup_urls(urls) or _combined_sources(etf)


def _main_exchange_label(exchange_value: Optional[str]) -> Optional[str]:
    if not exchange_value:
        return None
    s = exchange_value.strip().lower()
    if "nasdaq" in s:
        return "NASDAQ"
    if "nyse" in s:
        return "NYSE"
    return exchange_value.strip()


# --------------------------------------------------------------------------- #
# Verification for a single ETF                                               #
# --------------------------------------------------------------------------- #
async def verify_single_etf(
    evaluator: Evaluator,
    parent_node,
    etf: ETFExtraction,
    idx: int,
) -> None:
    etf_idx_human = idx + 1

    # Top-level node for this ETF (parallel; partial credit allowed across big sections)
    etf_node = evaluator.add_parallel(
        id=f"etf_{idx}",
        desc=f"ETF #{etf_idx_human}: Verify the ETF satisfies all specified investment criteria",
        parent=parent_node,
        critical=False,
    )

    # ---------------- Identification ----------------
    ident_node = evaluator.add_parallel(
        id=f"etf_{idx}_identification",
        desc=f"ETF #{etf_idx_human}: Provide the complete official name and issuer",
        parent=etf_node,
        critical=True,
    )

    # Full Name
    full_name_leaf = evaluator.add_leaf(
        id=f"etf_{idx}_full_name",
        desc="State the complete official name of the ETF",
        parent=ident_node,
        critical=True,
    )
    if etf.official_name and _combined_sources(etf):
        await evaluator.verify(
            claim=f"The ETF's complete official name is '{etf.official_name}'.",
            node=full_name_leaf,
            sources=_combined_sources(etf),
            additional_instruction="Verify the exact official fund name from authoritative sources (prospectus, fact sheet, exchange listing, or issuer site). Allow minor punctuation or case variations.",
        )
    else:
        full_name_leaf.score = 0.0
        full_name_leaf.status = "failed"

    # Issuer
    issuer_leaf = evaluator.add_leaf(
        id=f"etf_{idx}_issuer",
        desc="Identify the asset management company (issuer) of the ETF",
        parent=ident_node,
        critical=True,
    )
    if etf.issuer and _combined_sources(etf):
        await evaluator.verify(
            claim=f"The ETF's issuer (sponsor) is '{etf.issuer}'.",
            node=issuer_leaf,
            sources=_combined_sources(etf),
            additional_instruction="Confirm the sponsor/issuer name from official materials (prospectus, issuer homepage, or exchange listing). Allow minor naming variants (LLC/Inc.) if clearly the same company.",
        )
    else:
        issuer_leaf.score = 0.0
        issuer_leaf.status = "failed"

    # ---------------- Regulatory Compliance ----------------
    reg_node = evaluator.add_parallel(
        id=f"etf_{idx}_regulatory",
        desc=f"ETF #{etf_idx_human}: Verify SEC approval and exchange listing status with ticker",
        parent=etf_node,
        critical=True,
    )

    # SEC approval (focus on approval/effectiveness; active trading verified via exchange listing)
    sec_leaf = evaluator.add_leaf(
        id=f"etf_{idx}_sec_approval",
        desc="Confirm the ETF is SEC-approved (registration effective and/or rule approval)",
        parent=reg_node,
        critical=True,
    )
    sec_src = _sec_sources(etf)
    if etf and sec_src:
        await evaluator.verify(
            claim="This ETF has been approved/effectively authorized by the U.S. Securities and Exchange Commission (e.g., an approved 19b-4 order and/or an effective registration statement).",
            node=sec_leaf,
            sources=sec_src,
            additional_instruction=(
                "Accept SEC approval evidence including: (a) an SEC order approving the exchange rule filing (19b-4) "
                "and/or (b) an effective registration statement (e.g., S-1/S-3/N-1A). "
                "Issuer or exchange announcements quoting the SEC order are weaker; prefer direct SEC/EDGAR pages. "
                "If no reliable page clearly indicates SEC approval/effectiveness, mark as not supported."
            ),
        )
    else:
        sec_leaf.score = 0.0
        sec_leaf.status = "failed"

    # Exchange listing (parallel: exchange name + ticker)
    listing_node = evaluator.add_parallel(
        id=f"etf_{idx}_exchange_listing",
        desc="Verify the ETF is listed on NYSE or NASDAQ with an official ticker symbol",
        parent=reg_node,
        critical=True,
    )

    # Exchange Name
    exch_leaf = evaluator.add_leaf(
        id=f"etf_{idx}_exchange_name",
        desc="Specify the exchange where the ETF is listed (NYSE or NASDAQ)",
        parent=listing_node,
        critical=True,
    )
    exch_src = _listing_sources(etf)
    main_exch = _main_exchange_label(etf.exchange)
    if main_exch and exch_src:
        await evaluator.verify(
            claim=f"This ETF is listed on the {main_exch}.",
            node=exch_leaf,
            sources=exch_src,
            additional_instruction=(
                "Verify that the ETF is listed on NYSE (including NYSE Arca as part of the NYSE family) or NASDAQ. "
                "If the page shows 'NYSE Arca', treat that as NYSE. "
                "Accept exchange or issuer pages that clearly show the listing venue."
            ),
        )
    else:
        exch_leaf.score = 0.0
        exch_leaf.status = "failed"

    # Ticker Symbol
    ticker_leaf = evaluator.add_leaf(
        id=f"etf_{idx}_ticker",
        desc="Provide the official trading ticker symbol",
        parent=listing_node,
        critical=True,
    )
    if etf.ticker and exch_src:
        await evaluator.verify(
            claim=f"The ETF's official ticker symbol is '{etf.ticker}'.",
            node=ticker_leaf,
            sources=exch_src,
            additional_instruction="Confirm the exact ticker from exchange or issuer pages. Allow case-insensitive match; ignore postfixes like '.A' if they are not part of the core ticker.",
        )
    else:
        ticker_leaf.score = 0.0
        ticker_leaf.status = "failed"

    # Regulatory Documentation (existence of URLs)
    reg_docs_present = False
    if etf.sources:
        reg_docs_present = bool((etf.sources.sec_approval_urls or []) and (etf.sources.exchange_listing_urls or []))
    evaluator.add_custom_node(
        result=reg_docs_present,
        id=f"etf_{idx}_reg_docs",
        desc="Provide verifiable URLs confirming SEC approval status and exchange listing",
        parent=reg_node,
        critical=True,
    )

    # ---------------- Cost Structure Analysis ----------------
    cost_node = evaluator.add_parallel(
        id=f"etf_{idx}_costs",
        desc=f"ETF #{etf_idx_human}: Evaluate expense ratio and fee waiver provisions",
        parent=etf_node,
        critical=True,
    )

    # Expense Ratio <= 0.50% (excluding temporary waivers)
    exp_leaf = evaluator.add_leaf(
        id=f"etf_{idx}_expense_ratio_check",
        desc="Verify the stated annual expense ratio is 0.50% or lower (excluding temporary waivers)",
        parent=cost_node,
        critical=True,
    )
    cost_src = _cost_sources(etf)
    if etf.stated_expense_ratio and cost_src:
        await evaluator.verify(
            claim=f"The ETF's stated annual expense ratio, excluding any temporary promotional waivers, is at most 0.50%; specifically it is {etf.stated_expense_ratio}.",
            node=exp_leaf,
            sources=cost_src,
            additional_instruction=(
                "Use the fee table from the prospectus/summary prospectus/fact sheet. "
                "If multiple figures are shown (gross/net or with/without waiver), choose the stated baseline cost "
                "that excludes temporary promotional waivers. Consider 'Sponsor fee' or 'Management fee' if that is the stated expense ratio."
            ),
        )
    else:
        exp_leaf.score = 0.0
        exp_leaf.status = "failed"

    # Fee Waiver Period (end date > 2026-03-31) with terms and end-date
    waiver_node = evaluator.add_parallel(
        id=f"etf_{idx}_waiver",
        desc=f"Confirm the ETF offers a fee waiver or promotional period extending beyond {WAIVER_DEADLINE_STR}",
        parent=cost_node,
        critical=True,
    )

    # Waiver End Date
    waiver_end_leaf = evaluator.add_leaf(
        id=f"etf_{idx}_waiver_end_date",
        desc="Specify the end date of the fee waiver period",
        parent=waiver_node,
        critical=True,
    )
    if etf.fee_waiver and etf.fee_waiver.end_date and cost_src:
        await evaluator.verify(
            claim=(
                f"The ETF currently offers a fee waiver or promotional reduced-fee period that ends on {etf.fee_waiver.end_date}, "
                f"which extends beyond {WAIVER_DEADLINE_STR}."
            ),
            node=waiver_end_leaf,
            sources=cost_src,
            additional_instruction=(
                f"Verify the explicit waiver/promotion end date and confirm it is strictly after {WAIVER_DEADLINE_STR}. "
                "Allow reasonable date format variants (e.g., 'Mar. 31, 2026' vs 'March 31, 2026'). "
                "If the page only states 'through' a date, interpret end-of-day inclusively."
            ),
        )
    else:
        waiver_end_leaf.score = 0.0
        waiver_end_leaf.status = "failed"

    # Waiver Terms
    waiver_terms_leaf = evaluator.add_leaf(
        id=f"etf_{idx}_waiver_terms",
        desc="Describe the terms of the fee waiver (percentage reduced or waived)",
        parent=waiver_node,
        critical=True,
    )
    if etf.fee_waiver and etf.fee_waiver.terms and cost_src:
        await evaluator.verify(
            claim=f"The ETF currently offers a fee waiver/promotion with the following terms: {etf.fee_waiver.terms}",
            node=waiver_terms_leaf,
            sources=cost_src,
            additional_instruction=(
                "Confirm the fee waiver specifics (e.g., 'Sponsor fee waived to 0.00%' or 'Net expense capped at X%') "
                "from official documents. Minor paraphrasing is acceptable if the substance clearly matches."
            ),
        )
    else:
        waiver_terms_leaf.score = 0.0
        waiver_terms_leaf.status = "failed"

    # Cost Documentation (existence of URLs)
    cost_docs_present = False
    if etf.sources:
        any_cost = (etf.sources.cost_structure_urls or [])
        any_waiver = (etf.fee_waiver.waiver_urls if etf.fee_waiver else []) or []
        cost_docs_present = bool(any_cost or any_waiver)
    evaluator.add_custom_node(
        result=cost_docs_present,
        id=f"etf_{idx}_cost_docs",
        desc="Provide verifiable URLs documenting the expense ratio and fee waiver details",
        parent=cost_node,
        critical=True,
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
    model: str = "o4-mini",
) -> Dict:
    """
    Evaluate an answer for the XRP ETF selection task and return a structured result dictionary.
    """
    # Initialize evaluator (root is non-critical by design; we add a top-level task node beneath)
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

    # Add top-level task node (parallel to allow partial credit if only one ETF qualifies)
    task_node = evaluator.add_parallel(
        id="Identify_XRP_ETFs_Meeting_Investment_Criteria",
        desc="Identify at least two distinct SEC-approved XRP ETFs that meet criteria (expense ratio <= 0.50%, "
             "fee waiver beyond March 31, 2026, and listing on NYSE or NASDAQ) with verifiable documentation",
        parent=root,
        critical=False,
    )

    # Extract structured ETF info from the answer
    extracted_list = await evaluator.extract(
        prompt=prompt_extract_etfs(),
        template_class=ETFListExtraction,
        extraction_name="extracted_xrp_etfs",
    )

    # Choose the first 2 ETFs (pad with empty if fewer than 2)
    etfs: List[ETFExtraction] = list(extracted_list.etfs[:2])
    while len(etfs) < 2:
        etfs.append(ETFExtraction())

    # Build verification subtrees for the first two ETFs
    for idx in range(2):
        # Create per-ETF top node (as in the rubric: First_Qualifying_XRP_ETF / Second_Qualifying_XRP_ETF)
        etf_top_node = evaluator.add_parallel(
            id=f"{'First' if idx == 0 else 'Second'}_Qualifying_XRP_ETF",
            desc=f"Identify the {'first' if idx == 0 else 'second'} XRP ETF that satisfies all specified investment criteria",
            parent=task_node,
            critical=False,
        )
        await verify_single_etf(evaluator, etf_top_node, etfs[idx], idx)

    # Optionally record custom info (e.g., evaluation date)
    evaluator.add_custom_info(
        info={
            "waiver_deadline_requirement": WAIVER_DEADLINE_STR,
            "evaluation_timestamp_utc": datetime.utcnow().isoformat(timespec="seconds"),
        },
        info_type="meta",
        info_name="evaluation_metadata",
    )

    # Return structured summary
    return evaluator.get_summary()