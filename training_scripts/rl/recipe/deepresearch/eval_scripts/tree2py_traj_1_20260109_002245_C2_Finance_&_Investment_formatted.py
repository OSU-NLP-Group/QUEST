import asyncio
import logging
from typing import Optional, List, Dict

from pydantic import BaseModel, Field
from obj_task_eval.llm_client.base_client import LLMClient

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "spot_btc_etf_lowest_fee"
TASK_DESCRIPTION = (
    "Among the spot Bitcoin ETFs that were approved by the U.S. Securities and Exchange Commission (SEC) on January 10, 2024, "
    "which one has the lowest standard expense ratio? Please provide the ETF name, ticker symbol, the expense ratio percentage, "
    "and a reference URL supporting this information."
)


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class ETFSelection(BaseModel):
    """Extraction model for the selected ETF answer."""
    etf_name: Optional[str] = None
    ticker: Optional[str] = None
    standard_expense_ratio: Optional[str] = None
    reference_urls: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_etf_selection() -> str:
    return """
    Extract the single ETF the answer selected as having the lowest standard (gross) expense ratio among the U.S. spot Bitcoin ETFs approved by the SEC on January 10, 2024.

    Return a JSON object with these fields:
    - etf_name: The full fund name of the selected ETF.
    - ticker: The ticker symbol for the selected ETF (e.g., EZBC, IBIT, FBTC, ARKB, etc.).
    - standard_expense_ratio: The STANDARD (GROSS) expense ratio percentage stated in the answer for this ETF (e.g., "0.19%"). 
      If the answer mentions both standard/gross and net/after-waiver expense ratios, extract the standard/gross one.
      Keep the value exactly as written in the answer (including the percent sign if present).
    - reference_urls: An array of all URLs cited in the answer that are intended to support the ETF selection and/or the expense ratio.
      These may include official fund pages, prospectuses, the SEC order, exchange listings, or reputable financial news/analysis sites.
      Extract only URLs that explicitly appear in the answer. If none are present, return an empty array.

    Do not invent or infer any information not explicitly present in the answer. If a field is missing, set it to null (or empty array for reference_urls).
    """


# --------------------------------------------------------------------------- #
# Verification helpers                                                        #
# --------------------------------------------------------------------------- #
async def verify_selection_correctness(
    evaluator: Evaluator,
    parent_node,
    selection: ETFSelection
) -> None:
    """
    Build and run the verification leaves for ETF selection correctness.
    This node verifies:
    - Spot ETF nature (holds actual Bitcoin)
    - SEC approval date (Jan 10, 2024)
    - Trading started on or shortly after Jan 11, 2024
    - Lowest standard expense ratio among those approved on Jan 10, 2024 (excluding temporary waivers)
    """
    urls = selection.reference_urls or []

    selection_node = evaluator.add_parallel(
        id="etf_selection_correctness",
        desc="Correct ETF is selected based on the stated constraints",
        parent=evaluator.root,
        critical=True
    )

    # 1) Is spot Bitcoin ETF (holds actual bitcoin, not futures)
    spot_leaf = evaluator.add_leaf(
        id="is_spot_bitcoin_etf",
        desc="Selected ETF is a spot Bitcoin ETF that holds actual Bitcoin (not futures/derivatives)",
        parent=selection_node,
        critical=True
    )
    claim_spot = (
        f"The ETF {selection.etf_name or ''} ({selection.ticker or ''}) is a U.S. spot Bitcoin ETF that holds Bitcoin directly "
        f"(e.g., in custody/trust), not a futures- or derivatives-based product."
    )
    await evaluator.verify(
        claim=claim_spot,
        node=spot_leaf,
        sources=urls,
        additional_instruction=(
            "Determine if the fund holds physical/actual bitcoin (spot) rather than bitcoin futures. "
            "Accept evidence from an official fund page, prospectus, or reputable financial source that clearly states it is a spot bitcoin ETF."
        )
    )

    # 2) SEC approval date is January 10, 2024
    sec_leaf = evaluator.add_leaf(
        id="sec_approval_date",
        desc="Selected ETF was approved by the SEC on January 10, 2024",
        parent=selection_node,
        critical=True
    )
    claim_sec = (
        f"The ETF {selection.etf_name or ''} ({selection.ticker or ''}) was approved by the U.S. SEC on January 10, 2024."
    )
    await evaluator.verify(
        claim=claim_sec,
        node=sec_leaf,
        sources=urls,
        additional_instruction=(
            "Look for mention of the SEC approval date or reference to the Jan 10, 2024 SEC order approving spot bitcoin ETFs. "
            "It is sufficient if a reliable source states that this ETF was among those approved on that date."
        )
    )

    # 3) Trading start timing: began trading on or shortly after Jan 11, 2024
    trading_leaf = evaluator.add_leaf(
        id="trading_start_timing",
        desc="Selected ETF began trading on or shortly after January 11, 2024",
        parent=selection_node,
        critical=True
    )
    claim_trading = (
        f"The ETF {selection.etf_name or ''} ({selection.ticker or ''}) began trading on or shortly after January 11, 2024 "
        f"(i.e., first trading day was Jan 11, 2024 or within the next few days)."
    )
    await evaluator.verify(
        claim=claim_trading,
        node=trading_leaf,
        sources=urls,
        additional_instruction=(
            "Check for listing or first trading date confirmation (e.g., exchange announcements, fund site, or reputable news). "
            "Dates of Jan 11, 2024 or the immediate days after are acceptable."
        )
    )

    # 4) Lowest standard expense ratio among those approved Jan 10, 2024 (excluding temporary waivers/promotions)
    lowest_leaf = evaluator.add_leaf(
        id="lowest_standard_expense_ratio",
        desc="Among ETFs meeting the above criteria, the selected ETF has the lowest standard expense ratio, excluding temporary fee waivers/promotional periods",
        parent=selection_node,
        critical=True
    )
    claim_lowest = (
        f"Among the spot Bitcoin ETFs approved by the SEC on January 10, 2024, "
        f"{selection.etf_name or ''} ({selection.ticker or ''}) has the lowest standard (gross) expense ratio. "
        f"Temporary fee waivers or promotional fee reductions should be ignored. Ties for the lowest are acceptable."
    )
    await evaluator.verify(
        claim=claim_lowest,
        node=lowest_leaf,
        sources=urls,
        additional_instruction=(
            "Focus on the standard/gross expense ratio as defined in prospectuses or fund documentation. "
            "Ignore temporary/introductory waivers, caps, or promotional reductions. "
            "It is acceptable if the ETF is tied for the lowest standard/gross expense ratio. "
            "Use comparison articles or reputable sources if provided."
        )
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
) -> Dict:
    """
    Evaluate an answer for the spot Bitcoin ETF lowest expense ratio task.
    """
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,
        agent_name=agent_name,
        answer_name=answer_name,
        client=client,
        task_description="Identify the spot Bitcoin ETF approved Jan 10, 2024 with the lowest standard expense ratio and provide all requested fields with a supporting URL",
        answer=answer,
        global_cache=cache,
        global_semaphore=semaphore,
        logger=logger,
        default_model=model
    )

    # 1) Extract the selected ETF info from the answer
    selection: ETFSelection = await evaluator.extract(
        prompt=prompt_extract_etf_selection(),
        template_class=ETFSelection,
        extraction_name="selected_etf"
    )

    # 2) Build Required Response Fields branch (critical)
    required_fields_node = evaluator.add_parallel(
        id="required_response_fields",
        desc="Answer includes all requested fields and a supporting source",
        parent=root,
        critical=True
    )

    # Leaf checks for presence (custom boolean nodes)
    evaluator.add_custom_node(
        result=bool(selection.etf_name and selection.etf_name.strip()),
        id="etf_name_provided",
        desc="ETF name is provided",
        parent=required_fields_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=bool(selection.ticker and selection.ticker.strip()),
        id="ticker_symbol_provided",
        desc="Ticker symbol is provided",
        parent=required_fields_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=bool(selection.standard_expense_ratio and selection.standard_expense_ratio.strip()),
        id="expense_ratio_percentage_provided",
        desc="Standard expense ratio percentage is stated",
        parent=required_fields_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=bool(selection.reference_urls and len(selection.reference_urls) > 0),
        id="reference_url_provided",
        desc="At least one reference URL from official fund documentation or another reputable financial source is provided to verify the expense ratio",
        parent=required_fields_node,
        critical=True
    )

    # 3) Build ETF selection correctness branch (critical)
    await verify_selection_correctness(evaluator, root, selection)

    # 4) Return evaluation summary
    return evaluator.get_summary()