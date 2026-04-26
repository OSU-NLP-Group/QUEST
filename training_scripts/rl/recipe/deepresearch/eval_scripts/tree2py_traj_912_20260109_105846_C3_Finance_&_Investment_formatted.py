import asyncio
import logging
from typing import Any, List, Optional, Dict

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "sec_btc_etf_coinbase_lowest_expense"
TASK_DESCRIPTION = (
    "On January 10, 2024, the SEC approved 11 spot Bitcoin exchange-traded funds (ETFs). Among these ETFs, "
    "identify the one that uses Coinbase as its custodian and has the lowest standard ongoing expense ratio "
    "(excluding any promotional or temporary fee waivers). For the identified ETF, provide the following information: "
    "the ticker symbol, the name of the issuer/sponsor company, the standard ongoing expense ratio (as a percentage), "
    "and the exchange where it trades. Note: When comparing expense ratios, use only the standard ongoing rates, "
    "not any promotional or temporarily waived fees that were offered at launch."
)


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class ETFDetails(BaseModel):
    """Details of the ETF identified by the answer."""
    name: Optional[str] = None
    ticker: Optional[str] = None
    issuer: Optional[str] = None
    exchange: Optional[str] = None
    custodian: Optional[str] = None
    expense_ratio_standard: Optional[str] = None  # Keep as string for robustness (e.g., "0.20%")

    # Source URLs cited in the answer to support each field/claim
    main_sources: List[str] = Field(default_factory=list)           # general ETF profile pages cited
    approval_sources: List[str] = Field(default_factory=list)       # SEC approval / news sources cited
    custodian_sources: List[str] = Field(default_factory=list)      # pages stating Coinbase as custodian
    expense_sources: List[str] = Field(default_factory=list)        # pages stating the STANDARD ongoing expense ratio
    exchange_sources: List[str] = Field(default_factory=list)       # pages stating the trading exchange


class ETFComparisonItem(BaseModel):
    """Competitor ETFs that (according to the answer) use Coinbase as custodian."""
    name: Optional[str] = None
    ticker: Optional[str] = None
    custodian: Optional[str] = None
    expense_ratio_standard: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class CoinbaseCustodiedComparison(BaseModel):
    """List of Coinbase-custodied ETFs extracted for comparison."""
    items: List[ETFComparisonItem] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_selected_etf() -> str:
    return """
    Identify the single ETF selected by the answer that purportedly meets all constraints in the task:
    - It is one of the 11 spot Bitcoin ETFs that the SEC approved on January 10, 2024.
    - It uses Coinbase as its custodian (including Coinbase Custody Trust Company / Coinbase Custody / Coinbase Prime).
    - It has the lowest STANDARD ongoing expense ratio among the Coinbase-custodied spot Bitcoin ETFs (EXCLUDING any promotional or temporary fee waivers).

    Extract the following fields for the selected ETF exactly as stated in the answer:
    - name: The ETF fund name.
    - ticker: The ticker symbol.
    - issuer: The issuer/sponsor company name.
    - exchange: The exchange where it trades (e.g., NASDAQ, NYSE Arca, Cboe BZX).
    - custodian: The custodian name as stated (should mention Coinbase or its custody entity).
    - expense_ratio_standard: The STANDARD ongoing expense ratio as a percentage string (e.g., "0.20%"). 
      IMPORTANT: This must be the non-promotional, ongoing fee rate; DO NOT extract launch-time fee waivers or temporary/promotional rates.

    Also extract source URLs explicitly cited in the answer that support each field/claim:
    - main_sources: General ETF profile/issuer pages or official documents cited.
    - approval_sources: URLs in the answer that support the SEC approval on January 10, 2024 for spot Bitcoin ETFs or this ETF specifically.
    - custodian_sources: URLs that explicitly state Coinbase (or Coinbase Custody Trust Company / Coinbase Prime) is the custodian.
    - expense_sources: URLs that explicitly state the STANDARD ongoing expense ratio (exclude temporary/waived fee references).
    - exchange_sources: URLs that explicitly state where the ETF trades.

    RULES:
    - Only extract URLs that are explicitly present in the answer (plain URLs or markdown links). Do not invent URLs.
    - If any field is missing in the answer, set it to null. If any source type is not present, return an empty list for that field.
    """


def prompt_extract_coinbase_comparison() -> str:
    return """
    From the answer, extract the set of competitor spot Bitcoin ETFs that use Coinbase as custodian (including Coinbase Custody Trust Company / Coinbase Custody / Coinbase Prime), which are used to justify the 'lowest STANDARD ongoing expense ratio' comparison.

    For each competitor ETF mentioned in the answer (if any), extract:
    - name
    - ticker
    - custodian
    - expense_ratio_standard: The STANDARD ongoing expense ratio as a percentage string (exclude promotional/waived rates)
    - sources: All URLs cited in the answer that support the custodian and the STANDARD ongoing expense ratio.

    Return a JSON array under 'items'. If the answer does not provide any competitor information, return an empty array.
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def _combine_sources(*lists: List[str]) -> List[str]:
    """Combine and deduplicate multiple lists of URLs."""
    seen = set()
    combined: List[str] = []
    for lst in lists:
        for url in lst or []:
            if url and url not in seen:
                seen.add(url)
                combined.append(url)
    return combined


def _safe_str(v: Optional[str]) -> str:
    return v or ""


def _filter_coinbase_items(items: List[ETFComparisonItem]) -> List[ETFComparisonItem]:
    out = []
    for it in items:
        cust = (it.custodian or "").lower()
        if "coinbase" in cust:
            out.append(it)
    return out


# --------------------------------------------------------------------------- #
# Verification functions                                                      #
# --------------------------------------------------------------------------- #
async def verify_correct_etf_selected(
    evaluator: Evaluator,
    parent_node,
    selected: ETFDetails,
    comparison: CoinbaseCustodiedComparison
) -> None:
    """
    Build and verify the 'Correct_ETF_Selected' branch.
    """
    correct_node = evaluator.add_parallel(
        id="Correct_ETF_Selected",
        desc="The ETF named by the response satisfies all selection constraints from the prompt.",
        parent=parent_node,
        critical=True
    )

    # Optional presence check to gate subsequent verifications (critical under this parent)
    selected_present = evaluator.add_custom_node(
        result=(selected is not None and bool(_safe_str(selected.name)) and bool(_safe_str(selected.ticker))),
        id="Selected_ETF_Present",
        desc="An ETF (name and ticker) is identified in the answer.",
        parent=correct_node,
        critical=True
    )

    # 1) SEC approval on Jan 10, 2024 (one of the 11)
    sec_approved_leaf = evaluator.add_leaf(
        id="SEC_Approved_Spot_Bitcoin_ETF_Jan_10_2024",
        desc="The ETF is one of the 11 spot Bitcoin ETFs approved by the SEC on January 10, 2024.",
        parent=correct_node,
        critical=True
    )
    sec_claim = (
        f"The ETF {_safe_str(selected.name)} ({_safe_str(selected.ticker)}) is one of the "
        "SEC-approved spot Bitcoin ETFs announced on January 10, 2024."
    )
    sec_sources = _combine_sources(selected.approval_sources, selected.main_sources)
    await evaluator.verify(
        claim=sec_claim,
        node=sec_approved_leaf,
        sources=sec_sources,
        additional_instruction=(
            "Verify that the ETF was approved as a spot Bitcoin ETF by the U.S. SEC on January 10, 2024, "
            "as part of the cohort of approvals. Accept credible references (SEC order, reputable financial news, issuer disclosures)."
        )
    )

    # 2) Uses Coinbase as custodian
    custodian_leaf = evaluator.add_leaf(
        id="Uses_Coinbase_As_Custodian",
        desc="The ETF uses Coinbase (including Coinbase Custody Trust Company) as its custodian for holding Bitcoin.",
        parent=correct_node,
        critical=True
    )
    custodian_claim = (
        f"The ETF {_safe_str(selected.name)} ({_safe_str(selected.ticker)}) uses Coinbase (or Coinbase Custody Trust Company / "
        f"Coinbase Custody / Coinbase Prime) as its Bitcoin custodian."
    )
    custodian_sources = _combine_sources(selected.custodian_sources, selected.main_sources)
    await evaluator.verify(
        claim=custodian_claim,
        node=custodian_leaf,
        sources=custodian_sources,
        additional_instruction=(
            "Confirm that the ETF's custodian is Coinbase or a Coinbase custody entity (e.g., Coinbase Custody Trust Company, "
            "Coinbase Custody, Coinbase Prime)."
        )
    )

    # 3) Lowest STANDARD ongoing expense ratio among Coinbase-custodied ETFs
    lowest_leaf = evaluator.add_leaf(
        id="Lowest_Standard_Ongoing_Expense_Ratio_Among_Coinbase_Custodied",
        desc=(
            "Among spot Bitcoin ETFs that use Coinbase as custodian, the ETF has the lowest standard ongoing expense ratio, "
            "determined using only standard ongoing rates (excluding promotional/temporary waivers)."
        ),
        parent=correct_node,
        critical=True
    )
    coinbase_items = _filter_coinbase_items(comparison.items)
    competitor_sources_all: List[str] = []
    for it in coinbase_items:
        competitor_sources_all.extend(it.sources or [])
    lowest_sources = _combine_sources(selected.expense_sources, selected.main_sources, competitor_sources_all)

    lowest_claim = (
        f"Among spot Bitcoin ETFs that use Coinbase as custodian, {_safe_str(selected.name)} ({_safe_str(selected.ticker)}) "
        f"has the lowest (or tied-lowest) STANDARD ongoing expense ratio, excluding promotional or temporary fee waivers. "
        f"Its STANDARD ongoing expense ratio is {_safe_str(selected.expense_ratio_standard)}."
    )

    await evaluator.verify(
        claim=lowest_claim,
        node=lowest_leaf,
        sources=lowest_sources,
        additional_instruction=(
            "Compare ONLY the STANDARD ongoing expense ratios (not any launch-time promotional or temporary fee waivers). "
            "Use the provided competitor pages and the selected ETF page(s). If multiple ETFs share the same lowest standard rate, "
            "this still satisfies the 'lowest' requirement."
        )
    )


async def verify_requested_details(
    evaluator: Evaluator,
    parent_node,
    selected: ETFDetails
) -> None:
    """
    Build and verify the 'Requested_ETF_Details_Provided' branch, including each required field.
    """
    details_node = evaluator.add_parallel(
        id="Requested_ETF_Details_Provided",
        desc="All requested fields for the identified ETF are present and correct.",
        parent=parent_node,
        critical=True
    )

    # Ticker
    ticker_provided = evaluator.add_custom_node(
        result=bool(_safe_str(selected.ticker)),
        id="Ticker_Symbol_Provided",
        desc="Ticker symbol is provided in the answer.",
        parent=details_node,
        critical=True
    )
    ticker_leaf = evaluator.add_leaf(
        id="Ticker_Symbol",
        desc="Provides the ETF's correct ticker symbol.",
        parent=details_node,
        critical=True
    )
    ticker_claim = f"The ticker symbol for {_safe_str(selected.name)} is '{_safe_str(selected.ticker)}'."
    await evaluator.verify(
        claim=ticker_claim,
        node=ticker_leaf,
        sources=_combine_sources(selected.main_sources),
        additional_instruction="Confirm the ETF's ticker symbol as shown on official or authoritative pages."
    )

    # Issuer / Sponsor
    issuer_provided = evaluator.add_custom_node(
        result=bool(_safe_str(selected.issuer)),
        id="Issuer_Sponsor_Name_Provided",
        desc="Issuer/sponsor company name is provided in the answer.",
        parent=details_node,
        critical=True
    )
    issuer_leaf = evaluator.add_leaf(
        id="Issuer_Sponsor_Name",
        desc="Provides the ETF issuer/sponsor company name.",
        parent=details_node,
        critical=True
    )
    issuer_claim = f"The issuer/sponsor company for {_safe_str(selected.name)} ({_safe_str(selected.ticker)}) is '{_safe_str(selected.issuer)}'."
    await evaluator.verify(
        claim=issuer_claim,
        node=issuer_leaf,
        sources=_combine_sources(selected.main_sources),
        additional_instruction="Verify the sponsor/issuer name as stated by the issuer or authoritative sources."
    )

    # Standard ongoing expense ratio (percent)
    expense_provided = evaluator.add_custom_node(
        result=bool(_safe_str(selected.expense_ratio_standard)),
        id="Standard_Ongoing_Expense_Ratio_Percent_Provided",
        desc="Standard ongoing expense ratio (percent) is provided in the answer.",
        parent=details_node,
        critical=True
    )
    expense_leaf = evaluator.add_leaf(
        id="Standard_Ongoing_Expense_Ratio_Percent",
        desc="Provides the ETF's standard ongoing expense ratio as a percentage (not a promotional/waived rate).",
        parent=details_node,
        critical=True
    )
    expense_claim = (
        f"The STANDARD ongoing expense ratio (non-promotional) for {_safe_str(selected.name)} ({_safe_str(selected.ticker)}) "
        f"is {_safe_str(selected.expense_ratio_standard)}."
    )
    await evaluator.verify(
        claim=expense_claim,
        node=expense_leaf,
        sources=_combine_sources(selected.expense_sources, selected.main_sources),
        additional_instruction=(
            "Confirm that the fee is the STANDARD ongoing expense ratio (not a temporary launch waiver or promotional rate)."
        )
    )

    # Trading exchange
    exchange_provided = evaluator.add_custom_node(
        result=bool(_safe_str(selected.exchange)),
        id="Trading_Exchange_Provided",
        desc="Trading exchange is provided in the answer.",
        parent=details_node,
        critical=True
    )
    exchange_leaf = evaluator.add_leaf(
        id="Trading_Exchange",
        desc="Provides the exchange on which the ETF trades.",
        parent=details_node,
        critical=True
    )
    exchange_claim = f"The ETF {_safe_str(selected.name)} ({_safe_str(selected.ticker)}) trades on {_safe_str(selected.exchange)}."
    await evaluator.verify(
        claim=exchange_claim,
        node=exchange_leaf,
        sources=_combine_sources(selected.exchange_sources, selected.main_sources),
        additional_instruction="Confirm the listing exchange as stated on authoritative pages."
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
    Evaluate the agent's answer for the SEC-approved spot Bitcoin ETF selection task.
    """
    # Initialize evaluator
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.SEQUENTIAL,  # Follow rubric: overall evaluation proceeds logically
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

    # Perform extractions (run in parallel)
    selected_task = evaluator.extract(
        prompt=prompt_extract_selected_etf(),
        template_class=ETFDetails,
        extraction_name="selected_etf"
    )
    comparison_task = evaluator.extract(
        prompt=prompt_extract_coinbase_comparison(),
        template_class=CoinbaseCustodiedComparison,
        extraction_name="coinbase_comparison_set"
    )
    selected_etf, coinbase_comparison = await asyncio.gather(selected_task, comparison_task)

    # Build the rubric tree
    complete_node = evaluator.add_sequential(
        id="Complete_and_Accurate_Answer",
        desc="Response identifies the correct ETF meeting all selection criteria and provides all requested ETF details.",
        parent=root,
        critical=True
    )

    # Branch 1: Correct ETF selection verification
    await verify_correct_etf_selected(
        evaluator=evaluator,
        parent_node=complete_node,
        selected=selected_etf,
        comparison=coinbase_comparison
    )

    # Branch 2: Requested details verification
    await verify_requested_details(
        evaluator=evaluator,
        parent_node=complete_node,
        selected=selected_etf
    )

    # Return structured summary
    return evaluator.get_summary()