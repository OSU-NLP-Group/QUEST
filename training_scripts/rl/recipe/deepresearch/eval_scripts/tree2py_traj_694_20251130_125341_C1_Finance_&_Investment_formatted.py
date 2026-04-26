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
TASK_ID = "black_friday_2025_us_stock_hours"
TASK_DESCRIPTION = """
What time does the U.S. stock market close on Black Friday 2025, and is the market open or closed on that day?
"""

# Ground truth reference info (for transparency in the summary only)
GROUND_TRUTH_INFO = {
    "black_friday_2025_date": "November 28, 2025",
    "market_open_status": "Open with shortened hours",
    "early_close_time_et": "1:00 PM ET",
    "applies_to_markets": ["NYSE", "NASDAQ"],
    "regular_close_time_normal_days": "4:00 PM ET"
}

# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class MarketHoursInfo(BaseModel):
    """Information the answer claims regarding Black Friday 2025 market hours."""
    black_friday_2025_date: Optional[str] = None
    market_open_or_closed_phrase: Optional[str] = None
    early_close_time_et: Optional[str] = None
    applies_to_markets: List[str] = Field(default_factory=list)  # e.g., ["NYSE", "NASDAQ"]
    regular_close_time_normal_days_et: Optional[str] = None
    sources: List[str] = Field(default_factory=list)  # any URLs provided in the answer


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_market_hours() -> str:
    return """
    Extract the specific information the answer states about U.S. stock market hours on Black Friday 2025.
    Return a JSON object with the following fields:

    - black_friday_2025_date: The calendar date stated for "Black Friday 2025" (e.g., "November 28, 2025", "Nov 28, 2025", or "11/28/2025"). If not provided, return null.
    - market_open_or_closed_phrase: The exact phrase indicating whether the market is open or closed on that day (e.g., "open", "open with shortened hours", "closed"). If not provided, return null.
    - early_close_time_et: The early closing time on Black Friday in Eastern Time (ET), as stated (e.g., "1:00 PM ET", "1 pm ET", "13:00 ET", or "1:00 PM EST"). If not provided, return null.
    - applies_to_markets: A list of market names the answer explicitly says the early close applies to. Include "NYSE" and "NASDAQ" if they are mentioned (case-insensitive). Also accept synonyms like "New York Stock Exchange" for NYSE and "Nasdaq Stock Market" or "Nasdaq" for NASDAQ. If none are mentioned, return an empty list.
    - regular_close_time_normal_days_et: The regular closing time on normal trading days in Eastern Time (ET), as stated (e.g., "4:00 PM ET"). If not provided, return null.
    - sources: Extract all URLs cited in the answer. Include plain URLs or markdown links. If no URLs are provided, return an empty list.

    IMPORTANT:
    - Do NOT infer facts; only extract what the answer explicitly states.
    - Preserve the answer’s formatting for times and dates as strings.
    - The "applies_to_markets" list should only contain explicit mentions extracted from the answer.
    """


# --------------------------------------------------------------------------- #
# Verification builder                                                        #
# --------------------------------------------------------------------------- #
async def build_verification_tree(
    evaluator: Evaluator,
    root: Any,
    extracted: MarketHoursInfo
) -> None:
    """
    Build and execute verifications according to the rubric.
    All checks are critical and aggregated in parallel under a critical parent node.
    """

    # Critical parent node per rubric
    main_node = evaluator.add_parallel(
        id="Black_Friday_2025_US_Stock_Market_Hours",
        desc="Verify the answer satisfies all stated constraints about U.S. stock market status and hours on Black Friday 2025.",
        parent=root,
        critical=True
    )

    # 1) Black_Friday_2025_Date
    node_date = evaluator.add_leaf(
        id="Black_Friday_2025_Date",
        desc="The answer identifies Black Friday 2025 as November 28, 2025.",
        parent=main_node,
        critical=True
    )
    claim_date = (
        "The answer explicitly identifies Black Friday 2025 as November 28, 2025. "
        "Allow minor date formatting variants such as 'Nov 28, 2025' or '11/28/2025'. "
        "Mark as incorrect if the answer states a different date or does not specify the date."
    )
    await evaluator.verify(
        claim=claim_date,
        node=node_date,
        additional_instruction="Focus on whether the answer text states the date November 28, 2025 for Black Friday 2025."
    )

    # 2) Market_Open_Status
    node_open = evaluator.add_leaf(
        id="Market_Open_Status",
        desc="The answer indicates the U.S. stock market is open (not fully closed) on Black Friday 2025.",
        parent=main_node,
        critical=True
    )
    claim_open = (
        "The answer indicates that the U.S. stock market is open on Black Friday 2025 (i.e., not fully closed). "
        "Accept phrases such as 'open', 'open with shortened hours', 'trading day', or 'early close'. "
        "If the answer says 'closed', 'no trading', or implies full closure, mark as incorrect."
    )
    await evaluator.verify(
        claim=claim_open,
        node=node_open,
        additional_instruction="Check the answer wording for 'open', 'shortened hours', or equivalent; it must not indicate a full closure."
    )

    # 3) Early_Close_Time_ET
    node_close_time = evaluator.add_leaf(
        id="Early_Close_Time_ET",
        desc="The answer states the market closes early at 1:00 PM Eastern Time (ET) on Black Friday 2025.",
        parent=main_node,
        critical=True
    )
    claim_close_time = (
        "The answer explicitly states that the U.S. equity markets close early at 1:00 PM ET on Black Friday 2025. "
        "Accept equivalent time formatting: '1 pm ET', '1:00 p.m. ET', or '13:00 ET'. "
        "Treat 'EST' as equivalent to 'ET' for late November. "
        "Mark as incorrect if a different time or no early close time is stated."
    )
    await evaluator.verify(
        claim=claim_close_time,
        node=node_close_time,
        additional_instruction="Focus on the answer’s stated closing time; acceptable variants include '1 pm ET' or '1:00 PM EST'."
    )

    # 4) Applies_To_NYSE_And_NASDAQ
    node_markets = evaluator.add_leaf(
        id="Applies_To_NYSE_And_NASDAQ",
        desc="The answer states the early close applies to both NYSE and NASDAQ equity markets.",
        parent=main_node,
        critical=True
    )
    claim_markets = (
        "The answer states that the early close applies to both NYSE and NASDAQ equity markets. "
        "Accept synonyms such as 'New York Stock Exchange' for NYSE and 'Nasdaq Stock Market' or 'Nasdaq' for NASDAQ. "
        "If the answer mentions only one market or is ambiguous, mark as incorrect."
    )
    await evaluator.verify(
        claim=claim_markets,
        node=node_markets,
        additional_instruction="Ensure BOTH markets (NYSE and NASDAQ) are explicitly covered by the early close per the answer text."
    )

    # 5) Regular_Close_Time_Normal_Days
    node_regular = evaluator.add_leaf(
        id="Regular_Close_Time_Normal_Days",
        desc="The answer states the regular closing time on normal trading days is 4:00 PM ET.",
        parent=main_node,
        critical=True
    )
    claim_regular = (
        "The answer states that the regular closing time on normal trading days is 4:00 PM ET. "
        "Accept variants such as '4 pm ET' or '4:00 p.m. ET'. "
        "Mark as incorrect if a different normal closing time is stated or if the answer does not state it."
    )
    await evaluator.verify(
        claim=claim_regular,
        node=node_regular,
        additional_instruction="Check the answer text for the normal daily close of 4:00 PM ET."
    )


# --------------------------------------------------------------------------- #
# Main evaluation entry                                                       #
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
    Evaluate the answer for Black Friday 2025 U.S. stock market hours.
    Returns a structured summary with verification tree and score.
    """
    # Initialize evaluator
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # root container
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

    # Extract the market hours info from the answer text
    extracted = await evaluator.extract(
        prompt=prompt_extract_market_hours(),
        template_class=MarketHoursInfo,
        extraction_name="extracted_market_hours"
    )

    # Add ground truth info to the summary (for transparency only; not used in verification)
    evaluator.add_ground_truth(
        gt_info=GROUND_TRUTH_INFO,
        gt_type="reference_info"
    )

    # Build and run verifications according to rubric
    await build_verification_tree(evaluator, root, extracted)

    # Return the evaluation summary
    return evaluator.get_summary()