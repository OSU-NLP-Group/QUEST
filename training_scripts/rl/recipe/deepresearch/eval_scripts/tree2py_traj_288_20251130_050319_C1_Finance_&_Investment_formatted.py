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
TASK_ID = "us_stock_black_friday_2025_close_time"
TASK_DESCRIPTION = """
What time does the U.S. stock market close on Black Friday, November 28, 2025?
"""


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class MarketCloseExtraction(BaseModel):
    """
    Information extracted from the agent's answer about the Black Friday 2025 market close.
    """
    stated_close_time_text: Optional[str] = None
    timezone_expression: Optional[str] = None
    market_scope_text: Optional[str] = None
    source_urls: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_market_close_info() -> str:
    return """
    Extract from the answer the time information for when the U.S. stock market closes on Black Friday, November 28, 2025.

    Return the following fields:
    1) stated_close_time_text: The exact phrase in the answer that states the closing time for U.S. stock markets on Black Friday 2025. Include the number and any AM/PM marker and any timezone marker shown (e.g., "1:00 p.m. ET", "10 AM PT", "1 PM Eastern Time"). If multiple times are mentioned, choose the one that clearly refers to the U.S. stock market (NYSE/Nasdaq) closing time on Black Friday 2025. If no time is stated, return null.
    2) timezone_expression: The explicit timezone text used alongside the time in the answer (e.g., "ET", "EST", "EDT", "Eastern Time", "PT", "PST"). If none is shown, return null.
    3) market_scope_text: The phrase in the answer that indicates the scope or markets this time refers to (e.g., "U.S. stock market", "NYSE/Nasdaq", "major U.S. equity markets"). If the scope is not clearly stated, return null.
    4) source_urls: Extract every URL explicitly present in the answer (including any "Sources" section or inline links). Return an array. If none are present, return an empty array.

    Do not invent or normalize values. Preserve exact text from the answer. If the answer states a time in a different U.S. timezone (e.g., PT), still extract that exact phrase into stated_close_time_text.
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def get_canonical_equity_hours_urls() -> List[str]:
    """
    Provide canonical official sources for U.S. equity market holiday/early-close schedules.
    These are used when the answer does not supply sources, or to complement them.
    """
    return [
        # NYSE official calendars/hours (equity)
        "https://www.nyse.com/markets/hours-calendars",
        # NASDAQ official calendar/schedule
        "https://www.nasdaqtrader.com/Trader.aspx?id=Calendar",
        # Nasdaq public holiday schedule page
        "https://www.nasdaq.com/market-activity/stock-market-holidays",
    ]


def _merge_and_dedup_urls(primary: List[str], fallback: List[str]) -> List[str]:
    """Merge two URL lists while preserving order and removing duplicates."""
    seen = set()
    merged: List[str] = []
    for url in (primary + fallback):
        if not url:
            continue
        if url not in seen:
            seen.add(url)
            merged.append(url)
    return merged


# --------------------------------------------------------------------------- #
# Verification subroutines                                                    #
# --------------------------------------------------------------------------- #
async def verify_close_time_section(
    evaluator: Evaluator,
    parent_node,
    extracted: MarketCloseExtraction,
) -> None:
    """
    Build and verify the 'close time correctness' section with clear gating:
    1) Existence of a stated time in the answer
    2) Ground-truth support from official sources (1:00 p.m. ET early close)
    3) Alignment of the answer's stated time with that ground truth (allowing timezone equivalents)
    """
    close_time_node = evaluator.add_sequential(
        id="close_time_verification",
        desc="Verify the closing time for U.S. equity markets on Black Friday (Nov 28, 2025)",
        parent=parent_node,
        critical=True
    )

    # 1) Existence check: answer must state a closing time
    stated_time_present = extracted.stated_close_time_text is not None and extracted.stated_close_time_text.strip() != ""
    evaluator.add_custom_node(
        result=stated_time_present,
        id="close_time_stated",
        desc="Answer states a specific closing time for Black Friday 2025",
        parent=close_time_node,
        critical=True
    )

    # 2) Ground-truth support by official sources (independent of the answer's phrasing)
    sources = _merge_and_dedup_urls(extracted.source_urls if extracted.source_urls else [], get_canonical_equity_hours_urls())
    gt_leaf = evaluator.add_leaf(
        id="close_time_supported_by_sources",
        desc="Official sources support that NYSE/Nasdaq close early at 1:00 p.m. ET on Black Friday 2025",
        parent=close_time_node,
        critical=True
    )
    gt_claim = (
        "On Friday, November 28, 2025 (Black Friday, the day after Thanksgiving), "
        "the primary U.S. equity markets (NYSE and Nasdaq) have an early close at 1:00 p.m. Eastern Time (ET)."
    )
    await evaluator.verify(
        claim=gt_claim,
        node=gt_leaf,
        sources=sources,
        additional_instruction=(
            "Look for official holiday/early-close schedules. The relevant entry may be phrased as "
            "'Day after Thanksgiving — Early Close 1:00 p.m.' or similar. The early close applies to NYSE/Nasdaq "
            "equity markets. Verify the date corresponds to 2025 Black Friday (Nov 28, 2025)."
        )
    )

    # 3) Alignment: the answer's stated time should be equivalent to 1:00 p.m. ET
    match_leaf = evaluator.add_leaf(
        id="close_time_correctness",
        desc="Answer states the correct official closing time for U.S. stock markets on Black Friday (Nov 28, 2025), consistent with NYSE/Nasdaq holiday trading hours",
        parent=close_time_node,
        critical=True
    )
    stated_time = extracted.stated_close_time_text or ""
    match_claim = (
        f"The answer's stated closing time ('{stated_time}') is equivalent to 1:00 p.m. Eastern Time (ET) "
        f"on Black Friday, November 28, 2025."
    )
    await evaluator.verify(
        claim=match_claim,
        node=match_leaf,
        additional_instruction=(
            "Use the full answer context. Consider timezone conversions for late November in the U.S.: "
            "1:00 p.m. ET (EST) equals 12:00 p.m. CT (CST), 11:00 a.m. MT (MST), and 10:00 a.m. PT (PST). "
            "Minor textual variants such as '1 pm ET', '1 p.m. ET', '13:00 ET' are acceptable. "
            "If the answer provided only a different timezone (e.g., '10 AM PT') but it converts to 1:00 p.m. ET, "
            "treat this as correct."
        )
    )


async def verify_time_zone_clarity(
    evaluator: Evaluator,
    parent_node,
    extracted: MarketCloseExtraction,
) -> None:
    """
    Verify that the answer expresses the closing time in Eastern Time (ET) or provides an unambiguous conversion that includes ET.
    """
    tz_leaf = evaluator.add_leaf(
        id="time_zone_clarity",
        desc="Answer expresses the closing time in Eastern Time (ET) or provides an unambiguous conversion that includes ET",
        parent=parent_node,
        critical=True
    )
    tz_claim = (
        "The answer explicitly shows the closing time in Eastern Time (ET) (e.g., 'ET', 'EST', 'EDT', or 'Eastern Time'), "
        "or includes ET in a clear set of timezone conversions."
    )
    await evaluator.verify(
        claim=tz_claim,
        node=tz_leaf,
        additional_instruction=(
            "Accept explicit mentions like 'ET', 'EST', 'EDT', or 'Eastern Time'. "
            "If the answer only states another U.S. timezone (e.g., PT) without also providing ET or an explicit conversion that includes ET, "
            "this should not pass."
        )
    )


async def verify_market_scope_clarity(
    evaluator: Evaluator,
    parent_node,
    extracted: MarketCloseExtraction,
) -> None:
    """
    Verify that the answer clarifies the scope refers to major U.S. equity markets (NYSE/Nasdaq).
    Non-critical: helpful but not strictly required for full credit.
    """
    scope_leaf = evaluator.add_leaf(
        id="market_scope_clarity",
        desc="Answer clarifies the scope refers to major U.S. equity markets (e.g., NYSE and Nasdaq) rather than unrelated markets",
        parent=parent_node,
        critical=False
    )
    scope_claim = (
        "The answer makes it clear that the closing time refers to the primary U.S. equity exchanges, such as NYSE and Nasdaq, "
        "and not to unrelated markets like futures or the bond market."
    )
    await evaluator.verify(
        claim=scope_claim,
        node=scope_leaf,
        additional_instruction=(
            "Look for phrases like 'U.S. stock market', 'NYSE', 'Nasdaq', 'U.S. equities'. "
            "If the answer appears to refer to futures or the bond market instead of NYSE/Nasdaq equities, it should fail. "
            "If scope is reasonably clear from wording (e.g., 'U.S. stock market (NYSE/Nasdaq)'), consider it sufficient."
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
    Evaluate an answer for the U.S. stock market close time on Black Friday 2025 task.
    """
    # Initialize evaluator with a non-critical root to allow partial credit
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

    # Extract structured info from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_market_close_info(),
        template_class=MarketCloseExtraction,
        extraction_name="close_time_extraction",
    )

    # Add helpful ground truth info (for reporting; verification relies on URLs)
    evaluator.add_ground_truth({
        "expected_black_friday_date": "2025-11-28",
        "expected_equity_close_time_ET": "1:00 p.m. ET",
        "markets_in_scope": ["NYSE", "Nasdaq"],
        "reference_sources_used": get_canonical_equity_hours_urls(),
    })

    # Build verification tree sections
    await verify_close_time_section(evaluator, root, extracted)
    await verify_time_zone_clarity(evaluator, root, extracted)
    await verify_market_scope_clarity(evaluator, root, extracted)

    # Return summary
    return evaluator.get_summary()