import asyncio
import logging
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from obj_task_eval.llm_client.base_client import LLMClient
from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy

# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "largest_sp500_etf_aum_expense"
TASK_DESCRIPTION = """
As of late 2024 or early 2025, which S&P 500 ETF has the largest assets under management, and what is its expense ratio?
"""

# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class ETFExtraction(BaseModel):
    """
    Structured extraction of the ETF the answer claims is the largest-AUM S&P 500 tracker,
    along with its expense ratio and source URLs cited by the answer.
    """
    etf_name: Optional[str] = None                 # e.g., "SPDR S&P 500 ETF Trust"
    ticker: Optional[str] = None                   # e.g., "SPY"
    tracks_index: Optional[str] = None             # e.g., "S&P 500 Index"; exact phrasing from answer
    aum_value: Optional[str] = None                # e.g., "$500B", or textual description from answer
    aum_timeframe: Optional[str] = None            # e.g., "as of December 2024" or "early 2025"
    expense_ratio: Optional[str] = None            # e.g., "0.09%", textual as in the answer

    # URL sources explicitly cited in the answer (only URLs mentioned in the answer are allowed)
    sources_tracks_index: List[str] = Field(default_factory=list)         # URLs supporting that it tracks S&P 500
    sources_aum: List[str] = Field(default_factory=list)                  # URLs supporting "largest AUM" claim
    sources_expense_ratio_official: List[str] = Field(default_factory=list)  # Issuer site/fact sheet/prospectus URLs
    sources_expense_ratio_other: List[str] = Field(default_factory=list)  # Other URLs that mention expense ratio
    sources_general: List[str] = Field(default_factory=list)              # Any other URLs cited in the answer


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_etf_info() -> str:
    return """
    The task asks: “As of late 2024 or early 2025, which S&P 500 ETF has the largest assets under management, and what is its expense ratio?”
    Extract from the answer ONLY what the answer explicitly states or provides.

    You must extract a single ETF that the answer asserts is the largest-AUM S&P 500-tracking ETF (as of late 2024 / early 2025), plus its expense ratio, and group all cited URLs by their purpose.

    Return a JSON object with these fields:
    - etf_name: The name of the ETF (string). If multiple ETFs are mentioned, choose the one the answer claims is the largest by AUM.
    - ticker: The ETF ticker (string), e.g., "SPY", "IVV", or "VOO".
    - tracks_index: The exact phrasing in the answer describing the tracked index (likely "S&P 500 Index"). If omitted, set to null.
    - aum_value: The AUM stated or implied in the answer (string). If not stated, set to null.
    - aum_timeframe: Any date/timeframe the answer mentions for the AUM (string), e.g., "as of December 2024". If absent, set to null.
    - expense_ratio: The expense ratio value as stated (string), e.g., "0.09%". If absent, set to null.

    - sources_tracks_index: Array of URL strings the answer cites to support that the ETF tracks the S&P 500 Index. Prefer official issuer URLs if they appear in the answer. Only include URLs actually present in the answer.
    - sources_aum: Array of URL strings the answer cites to support the claim that this ETF has the largest AUM among S&P 500 ETFs as of late 2024/early 2025. Only include URLs actually present in the answer.
    - sources_expense_ratio_official: Array of URL strings from official issuer websites, fact sheets, or prospectuses that the answer cites for the expense ratio. Only include them if present in the answer.
    - sources_expense_ratio_other: Array of URL strings from non-issuer sites (media, aggregators) cited in the answer for the expense ratio. Only include URLs present in the answer.
    - sources_general: Array of any other URLs cited in the answer (that are not already captured above). Only include URLs present in the answer.

    IMPORTANT:
    - Extract ONLY URLs explicitly present in the answer (including markdown links). Do not invent URLs.
    - If a required field isn’t present in the answer, set it to null (or empty list for arrays).
    - If multiple ETFs are discussed, choose the one the answer claims is the largest by AUM; otherwise choose the first ETF the answer clearly identifies for this role.
    - Use strings for numerical values (e.g., "0.03%") exactly as written in the answer.
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def pick_sources(preferred: List[str], fallback: List[str]) -> List[str]:
    """
    Choose preferred sources list if not empty; otherwise return fallback list.
    """
    return preferred if preferred else fallback


# --------------------------------------------------------------------------- #
# Verification subroutines                                                    #
# --------------------------------------------------------------------------- #
async def verify_identification(
    evaluator: Evaluator,
    parent_node,
    extracted: ETFExtraction,
) -> None:
    """
    Build and evaluate the 'Identify_Largest_AUM_S_P_500_ETF' parallel critical sub-tree:
      - Basic existence (ETF identified)
      - Tracks S&P 500 Index
      - Largest AUM in timeframe (late 2024/early 2025)
    """
    identify_node = evaluator.add_parallel(
        id="Identify_Largest_AUM_S_P_500_ETF",
        desc="Correctly identifies the ETF that is the largest-AUM S&P 500-tracking ETF as of late 2024/early 2025.",
        parent=parent_node,
        critical=True
    )

    # Basic existence check: ETF name and ticker must be present
    exists = bool(extracted.etf_name and extracted.etf_name.strip()) and bool(extracted.ticker and extracted.ticker.strip())
    evaluator.add_custom_node(
        result=exists,
        id="ETF_Identified",
        desc="An ETF name and ticker are identified in the answer.",
        parent=identify_node,
        critical=True
    )

    # Tracks S&P 500 Index
    tracks_node = evaluator.add_leaf(
        id="Tracks_S_P_500_Index",
        desc="The identified ETF tracks the S&P 500 Index (i.e., is an S&P 500 index-tracking ETF).",
        parent=identify_node,
        critical=True
    )
    etf_display = f"{extracted.ticker or '[unknown ticker]'} ({extracted.etf_name or '[unknown name]'})"
    claim_tracks = f"The ETF {etf_display} tracks the S&P 500 Index."
    sources_for_tracks = pick_sources(extracted.sources_tracks_index, extracted.sources_general)
    await evaluator.verify(
        claim=claim_tracks,
        node=tracks_node,
        sources=sources_for_tracks,
        additional_instruction=(
            "Accept equivalent phrasings such as 'S&P 500', 'Standard & Poor's 500', or 'S&P 500 Index'. "
            "Rely on the provided webpage(s). If the sources are irrelevant, missing, or do not explicitly support that "
            f"{etf_display} tracks the S&P 500, mark as Not Supported."
        ),
    )

    # Largest AUM in timeframe
    largest_node = evaluator.add_leaf(
        id="Largest_AUM_In_Timeframe",
        desc="The identified ETF has the largest assets under management among S&P 500 ETFs as of late 2024/early 2025.",
        parent=identify_node,
        critical=True
    )
    claim_largest = (
        f"As of late 2024 or early 2025, {etf_display} has the largest assets under management among S&P 500 ETFs."
    )
    sources_for_aum = pick_sources(extracted.sources_aum, extracted.sources_general)
    await evaluator.verify(
        claim=claim_largest,
        node=largest_node,
        sources=sources_for_aum,
        additional_instruction=(
            "The webpage(s) must clearly support that the ETF is the largest by AUM among S&P 500-tracking ETFs "
            "in the specified timeframe (late 2024/early 2025). Accept explicit rankings or comparative statements. "
            "If the page lacks timeframe alignment or the claim is not explicitly supported, mark Not Supported."
        ),
    )


async def verify_expense_ratio(
    evaluator: Evaluator,
    parent_node,
    extracted: ETFExtraction,
) -> None:
    """
    Verify the expense ratio is provided and supported by OFFICIAL issuer sources (as described in the task).
    We add a small gating custom node to ensure official sources are present before verification.
    """
    # Add a critical custom node to ensure official issuer sources are present
    official_source_present = bool(extracted.sources_expense_ratio_official)
    evaluator.add_custom_node(
        result=official_source_present,
        id="Expense_Ratio_Official_Source_Present",
        desc="Official issuer source(s) for the expense ratio are present in the answer.",
        parent=parent_node,
        critical=True
    )

    # Leaf: Provide Expense Ratio (verified against official issuer sources)
    provide_node = evaluator.add_leaf(
        id="Provide_Expense_Ratio",
        desc="Provides the expense ratio for the identified ETF, and the value is verifiable from official issuer sources (e.g., issuer website/prospectus).",
        parent=parent_node,
        critical=True
    )
    etf_display = f"{extracted.ticker or '[unknown ticker]'} ({extracted.etf_name or '[unknown name]'})"
    er_value = extracted.expense_ratio or "[unknown expense ratio]"
    claim_expense = f"The expense ratio of {etf_display} is {er_value}."
    # Prefer official sources strictly; if none, the preceding custom node already fails and will block/skip this leaf.
    await evaluator.verify(
        claim=claim_expense,
        node=provide_node,
        sources=extracted.sources_expense_ratio_official,
        additional_instruction=(
            "Only accept if the claim is explicitly supported by official issuer documentation (e.g., issuer website, "
            "fund fact sheet, or prospectus). If the provided sources are non-official (e.g., media/aggregators) or missing, "
            "mark Not Supported even if the value appears plausible."
        ),
    )


# --------------------------------------------------------------------------- #
# Main evaluation entry point                                                 #
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
    Evaluate an answer for the largest S&P 500 ETF by AUM and its expense ratio (late 2024 / early 2025).
    """
    # Initialize evaluator (root is non-critical per framework; we add a critical child for the main task)
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

    # Extract structured ETF info from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_etf_info(),
        template_class=ETFExtraction,
        extraction_name="largest_sp500_etf_extraction"
    )

    # Build main critical sequential node corresponding to the rubric root
    task_node = evaluator.add_sequential(
        id="Largest_S_P_500_ETF_Task",
        desc="Identify the S&P 500-tracking ETF with the largest AUM as of late 2024/early 2025 and provide its (officially verifiable) expense ratio.",
        parent=root,
        critical=True
    )

    # 1) Identification sub-tree (parallel, critical)
    await verify_identification(evaluator, task_node, extracted)

    # 2) Expense ratio verification (critical leaf; gated by official source presence)
    await verify_expense_ratio(evaluator, task_node, extracted)

    # Return structured summary
    return evaluator.get_summary()