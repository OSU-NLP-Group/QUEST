import asyncio
import logging
import re
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "bf2025_latest_close"
TASK_DESCRIPTION = "On Black Friday 2025, which major retailer among Walmart, Best Buy, and Ulta Beauty has the latest closing time, and what is that closing time?"


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class RetailerClosing(BaseModel):
    retailer: Optional[str] = None
    closing_time: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class ClosingsExtraction(BaseModel):
    # The retailer the answer claims has the latest closing time (one of the three)
    selected_retailer: Optional[str] = None
    selected_closing_time: Optional[str] = None
    selected_sources: List[str] = Field(default_factory=list)

    # Per-retailer details (as stated in the answer)
    walmart: Optional[RetailerClosing] = None
    best_buy: Optional[RetailerClosing] = None
    ulta_beauty: Optional[RetailerClosing] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_closings() -> str:
    return """
    Extract the Black Friday 2025 store closing time information from the answer for the three specified retailers: Walmart, Best Buy, and Ulta Beauty. Also extract which retailer the answer claims has the latest closing time and what that time is.

    Return a JSON object with the following fields:
    1) selected_retailer: the retailer name the answer identifies as having the latest closing time among the three (must be one of: "Walmart", "Best Buy", "Ulta Beauty"). If not explicitly stated, return null.
    2) selected_closing_time: the exact closing time string stated for the selected retailer (e.g., "11 PM", "10:30 p.m.", "23:00"). If not stated, return null.
    3) selected_sources: an array of URLs mentioned in the answer that support the selected retailer's Black Friday 2025 closing time. If none are given, return an empty array.

    4) walmart: an object with fields:
        - retailer: should be "Walmart" if Walmart's closing time is mentioned, otherwise null.
        - closing_time: the exact closing time string for Walmart on Black Friday 2025 as stated in the answer; null if missing.
        - sources: array of URLs explicitly cited for Walmart's Black Friday 2025 closing time; empty if none.

    5) best_buy: an object with fields:
        - retailer: should be "Best Buy" if Best Buy's closing time is mentioned, otherwise null.
        - closing_time: the exact closing time string for Best Buy on Black Friday 2025 as stated in the answer; null if missing.
        - sources: array of URLs explicitly cited for Best Buy's Black Friday 2025 closing time; empty if none.

    6) ulta_beauty: an object with fields:
        - retailer: should be "Ulta Beauty" if Ulta Beauty's closing time is mentioned, otherwise null.
        - closing_time: the exact closing time string for Ulta Beauty on Black Friday 2025 as stated in the answer; null if missing.
        - sources: array of URLs explicitly cited for Ulta Beauty's Black Friday 2025 closing time; empty if none.

    IMPORTANT:
    - Only extract information explicitly present in the answer text and its provided URLs list; do not infer or add new URLs.
    - The closing time should be for Black Friday 2025 specifically. If the answer mentions general or different date hours without clearly tying to Black Friday 2025, still extract what it claims, but do not invent any details.
    - Preserve the closing time string exactly as stated (including AM/PM or 24-hour format).
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def normalize_retailer_name(name: Optional[str]) -> Optional[str]:
    if not name:
        return None
    n = name.strip().lower()
    if n in {"walmart", "wal-mart"}:
        return "Walmart"
    if n in {"best buy", "bestbuy"}:
        return "Best Buy"
    if n in {"ulta beauty", "ulta", "ulta beauty, inc.", "ulta cosmetics"}:
        return "Ulta Beauty"
    return None  # Not one of the three


def parse_time_to_minutes(time_str: Optional[str]) -> Optional[int]:
    """
    Parse a closing time string into minutes since midnight (0..1440).
    Accepts formats like "11 PM", "10:30 p.m.", "23:00", "midnight".
    Returns None if parsing fails.
    """
    if not time_str:
        return None
    s = time_str.strip().lower()
    s = s.replace("p.m.", "pm").replace("a.m.", "am").replace("pm.", "pm").replace("am.", "am")
    s = re.sub(r"\s+", " ", s)

    # Common words
    if "midnight" in s:
        return 24 * 60
    if "noon" in s:
        return 12 * 60

    # 12-hour format: e.g., "10 pm", "10:30 pm", "10pm", "10:30pm"
    m12 = re.search(r"\b(\d{1,2})(?::(\d{2}))?\s*(a|p)m\b", s)
    if m12:
        hour = int(m12.group(1))
        minute = int(m12.group(2)) if m12.group(2) else 0
        ap = m12.group(3)
        if hour == 12:
            # 12 AM is 00:xx, 12 PM is 12:xx
            total = minute if ap == "a" else (12 * 60 + minute)
        else:
            if ap == "a":
                total = hour * 60 + minute
            else:
                total = (hour + 12) * 60 + minute
        return total

    # 24-hour format: "23:00", "22:30"
    m24 = re.search(r"\b([01]?\d|2[0-3]):([0-5]\d)\b", s)
    if m24:
        hour = int(m24.group(1))
        minute = int(m24.group(2))
        return hour * 60 + minute

    # Edge case: sometimes "11" without am/pm; not reliable -> return None
    return None


def get_minutes_for(extracted: ClosingsExtraction, key: str) -> Optional[int]:
    """
    Get minutes for a retailer key among 'Walmart', 'Best Buy', 'Ulta Beauty'
    based on the per-retailer fields in extraction.
    """
    obj: Optional[RetailerClosing] = None
    if key == "Walmart":
        obj = extracted.walmart
    elif key == "Best Buy":
        obj = extracted.best_buy
    elif key == "Ulta Beauty":
        obj = extracted.ulta_beauty
    else:
        return None

    if not obj or not obj.closing_time:
        return None
    return parse_time_to_minutes(obj.closing_time)


def choose_sources_for_selected(extracted: ClosingsExtraction, selected: Optional[str]) -> List[str]:
    """
    Prefer selected_sources, otherwise fallback to the per-retailer sources (if any).
    """
    if extracted.selected_sources:
        return extracted.selected_sources

    sel = normalize_retailer_name(selected)
    if sel == "Walmart":
        return (extracted.walmart.sources if extracted.walmart else []) or []
    if sel == "Best Buy":
        return (extracted.best_buy.sources if extracted.best_buy else []) or []
    if sel == "Ulta Beauty":
        return (extracted.ulta_beauty.sources if extracted.ulta_beauty else []) or []
    return []


def compute_correct_identification(extracted: ClosingsExtraction) -> Tuple[bool, Dict[str, Any]]:
    """
    Compute whether the identified retailer truly has the latest closing time among the three,
    based purely on times extracted from the answer text (internal consistency check).
    Returns (result_bool, debug_info_dict).
    """
    selected = normalize_retailer_name(extracted.selected_retailer)
    sel_time_str = extracted.selected_closing_time
    sel_minutes = parse_time_to_minutes(sel_time_str)

    walmart_minutes = get_minutes_for(extracted, "Walmart")
    bestbuy_minutes = get_minutes_for(extracted, "Best Buy")
    ulta_minutes = get_minutes_for(extracted, "Ulta Beauty")

    debug = {
        "selected_retailer": selected,
        "selected_closing_time": sel_time_str,
        "selected_minutes": sel_minutes,
        "walmart_minutes": walmart_minutes,
        "bestbuy_minutes": bestbuy_minutes,
        "ulta_minutes": ulta_minutes,
    }

    # Must be one of the three and have a parseable selected time
    if selected is None or sel_minutes is None:
        return False, debug

    # Need times for the other two to compare (both must be present to judge "latest compared to the other two")
    if walmart_minutes is None or bestbuy_minutes is None or ulta_minutes is None:
        return False, debug

    # Determine max of all three
    max_minutes = max(walmart_minutes, bestbuy_minutes, ulta_minutes)
    # Accept ties: as long as selected's time equals the maximum, it is "a latest closing time"
    result = (sel_minutes == max_minutes)
    return result, debug


# --------------------------------------------------------------------------- #
# Verification tree construction and checks                                   #
# --------------------------------------------------------------------------- #
async def build_and_verify(
    evaluator: Evaluator,
    extraction: ClosingsExtraction,
    parent_node,
) -> None:
    """
    Build the rubric tree nodes and run the checks required by the rubric.
    """
    # Create main rubric node as critical parallel aggregator
    latest_node = evaluator.add_parallel(
        id="Latest_Closing_Retailer_Identification",
        desc="Correctly identify which retailer among Walmart, Best Buy, and Ulta Beauty has the latest closing time on Black Friday 2025, and provide that closing time",
        parent=parent_node,
        critical=True,
    )

    # 1) Correct_Retailer_Identified (critical leaf) - implemented as a custom logical check
    correct_result, debug_info = compute_correct_identification(extraction)
    evaluator.add_custom_info(
        info=debug_info,
        info_type="debug",
        info_name="parsed_times_minutes"
    )

    evaluator.add_custom_node(
        result=correct_result,
        id="Correct_Retailer_Identified",
        desc="The answer identifies one of the three specified retailers (Walmart, Best Buy, or Ulta Beauty) and that retailer is factually the one with the latest closing time on Black Friday 2025 when compared to the other two",
        parent=latest_node,
        critical=True
    )

    # 2) Accurate_Closing_Time (critical leaf) - evidence-based verification against cited URLs
    accurate_leaf = evaluator.add_leaf(
        id="Accurate_Closing_Time",
        desc="The answer provides the closing time for the identified retailer in standard time format (e.g., X p.m., X:00 PM, or 24-hour format) and this time matches the factual closing time for that retailer on Black Friday 2025",
        parent=latest_node,
        critical=True
    )

    selected_name = normalize_retailer_name(extraction.selected_retailer)
    selected_time = extraction.selected_closing_time
    selected_sources = choose_sources_for_selected(extraction, extraction.selected_retailer)

    # If no selected retailer or time, mark as failed directly
    if not selected_name or not selected_time:
        # Overwrite leaf as failed (since there's nothing to verify)
        accurate_leaf.score = 0.0
        accurate_leaf.status = "failed"
        return

    # If no sources, we cannot fact-check; mark as failed to avoid relying on the agent-only text
    if not selected_sources:
        accurate_leaf.score = 0.0
        accurate_leaf.status = "failed"
        return

    claim = f"On Black Friday 2025, {selected_name} closes at {selected_time}."
    await evaluator.verify(
        claim=claim,
        node=accurate_leaf,
        sources=selected_sources,
        additional_instruction=(
            "Verify strictly against the provided webpage(s) whether they explicitly state "
            f"the closing time for {selected_name} on Black Friday 2025 equals '{selected_time}'. "
            "Allow minor formatting differences (e.g., PM vs p.m., omitted :00), but the stated time must match. "
            "If the source is ambiguous (e.g., hours vary by location, or not specifically Black Friday 2025), "
            "conclude not supported."
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
    model: str = "o4-mini"
) -> Dict:
    """
    Evaluate an answer for the Black Friday 2025 latest closing time among Walmart, Best Buy, and Ulta Beauty.
    """
    # Initialize evaluator
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

    # Extract structured information from the answer
    extraction = await evaluator.extract(
        prompt=prompt_extract_closings(),
        template_class=ClosingsExtraction,
        extraction_name="closings_extraction"
    )

    # Build the verification tree and run checks
    await build_and_verify(evaluator, extraction, root)

    # Return evaluation summary
    return evaluator.get_summary()