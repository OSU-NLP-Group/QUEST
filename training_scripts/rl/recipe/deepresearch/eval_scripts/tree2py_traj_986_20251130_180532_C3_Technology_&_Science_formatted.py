import asyncio
import logging
import re
from typing import Any, List, Optional, Dict

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "apple_first_silicon_2024"
TASK_DESCRIPTION = (
    "Identify the first Apple Silicon chip announced in 2024 (by official announcement date) "
    "and provide the required information for that chip."
)


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class ChipInfo(BaseModel):
    chip_model: Optional[str] = None
    announcement_date: Optional[str] = None
    source_urls: List[str] = Field(default_factory=list)
    neural_engine_tops: Optional[str] = None
    improvement_factor_vs_m3: Optional[str] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_chip_info() -> str:
    return """
    From the provided answer, extract the following fields for the first Apple Silicon chip announced in 2024:

    1) chip_model: The specific chip model name (e.g., "M4", "M4 Pro", "M4 Max").
    2) announcement_date: The exact official announcement date for the chip (e.g., "May 7, 2024" or "2024-05-07"). Return as a string exactly as written in the answer.
    3) source_urls: A list of URLs cited in the answer that confirm the chip announcement (chip model and announcement context/date). These should be explicit URLs. Prefer official Apple domains (e.g., apple.com) or credible technology/news publications with identifiable site name and article title (avoid user-generated platforms like wikis, forums, social media).
    4) neural_engine_tops: The Neural Engine performance for the chip measured in TOPS (trillions of operations per second). Return the value as shown in the answer (e.g., "38 TOPS", "Up to 38 TOPS"). Keep the unit text if present; return a single string.
    5) improvement_factor_vs_m3: The improvement factor of this chip's Neural Engine compared to the M3 family baseline of 18 TOPS. Expressed as a multiplication factor, e.g., "2x", "2.11x". Return exactly the string provided in the answer.

    Rules:
    - Extract only what is explicitly present in the answer. If any required field is missing, set it to null (for strings) or an empty list (for source_urls).
    - For URLs, extract actual URLs (including protocol), whether in plain text or markdown link format.
    - Do not infer or compute values; only extract what's written.
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def _non_empty_str(s: Optional[str]) -> bool:
    return bool(s and s.strip())


def _year_is_2024(date_str: Optional[str]) -> bool:
    if not _non_empty_str(date_str):
        return False
    # Accept any 4-digit year format containing 2024
    return bool(re.search(r"\b2024\b", date_str.strip()))


def _parse_tops_float(tops_str: Optional[str]) -> Optional[float]:
    """
    Attempt to parse a numeric TOPS value from strings like:
    - "38 TOPS"
    - "Up to 38 TOPS"
    - "approximately 38.0 TOPS"
    If "TOPS" is not present, fall back to first numeric.
    """
    if not _non_empty_str(tops_str):
        return None
    s = tops_str.strip()
    # Prefer number near TOPS
    m = re.search(r"([0-9]+(?:\.[0-9]+)?)\s*(?:TOPS)\b", s, flags=re.IGNORECASE)
    if m:
        try:
            return float(m.group(1))
        except Exception:
            return None
    # Fallback: first numeric in the string
    m2 = re.search(r"([0-9]+(?:\.[0-9]+)?)", s)
    if m2:
        try:
            return float(m2.group(1))
        except Exception:
            return None
    return None


def _parse_factor_float(factor_str: Optional[str]) -> Optional[float]:
    """
    Parse factor like "2x" or "2.11x" -> 2.0 or 2.11
    """
    if not _non_empty_str(factor_str):
        return None
    s = factor_str.strip().lower()
    m = re.search(r"([0-9]+(?:\.[0-9]+)?)\s*x\b", s)
    if m:
        try:
            return float(m.group(1))
        except Exception:
            return None
    return None


def _structured_format_ok(info: ChipInfo) -> bool:
    """
    Check that the answer provides all required fields in a structured manner:
    - chip_model present
    - announcement_date present
    - at least one source URL
    - neural_engine_tops present (and has a numeric parse)
    - improvement_factor_vs_m3 present (and parseable numeric)
    """
    if not _non_empty_str(info.chip_model):
        return False
    if not _non_empty_str(info.announcement_date):
        return False
    if not info.source_urls or len(info.source_urls) == 0:
        return False
    if not _non_empty_str(info.neural_engine_tops):
        return False
    if _parse_tops_float(info.neural_engine_tops) is None:
        return False
    if not _non_empty_str(info.improvement_factor_vs_m3):
        return False
    if _parse_factor_float(info.improvement_factor_vs_m3) is None:
        return False
    return True


# --------------------------------------------------------------------------- #
# Verification tree construction                                              #
# --------------------------------------------------------------------------- #
async def build_verification_tree(evaluator: Evaluator, chip: ChipInfo) -> None:
    """
    Build and execute the verification logic per the rubric.
    """
    # Top-level critical sequential node mirroring the rubric Root
    task_node = evaluator.add_sequential(
        id="task_main",
        desc=TASK_DESCRIPTION,
        parent=evaluator.root,
        critical=True
    )

    # --- Step 1: Identify the first 2024 chip (critical) ---
    # Precondition A: core fields (chip, date, sources) must exist
    core_fields_ok = evaluator.add_custom_node(
        result=(
            _non_empty_str(chip.chip_model) and
            _non_empty_str(chip.announcement_date) and
            bool(chip.source_urls)
        ),
        id="core_fields_provided",
        desc="Core fields provided: chip model, announcement date, and at least one source URL",
        parent=task_node,
        critical=True
    )

    # Precondition B: announcement date is in 2024
    date_in_2024_node = evaluator.add_custom_node(
        result=_year_is_2024(chip.announcement_date),
        id="announcement_in_2024",
        desc="Announcement date is in 2024",
        parent=task_node,
        critical=True
    )

    # Leaf: Identify the first Apple Silicon chip announced in 2024
    identify_leaf = evaluator.add_leaf(
        id="identify_first_2024_chip",
        desc="Correctly identify the first Apple Silicon chip announced in 2024 based on the official announcement date (not product release date).",
        parent=task_node,
        critical=True
    )

    identify_claim = (
        f"According to the provided sources, the first Apple Silicon chip announced in 2024 is '{chip.chip_model}', "
        f"and it was officially announced on '{chip.announcement_date}'."
    )
    await evaluator.verify(
        claim=identify_claim,
        node=identify_leaf,
        sources=chip.source_urls,
        additional_instruction=(
            "Verify that the sources explicitly confirm the chip model and the official announcement (newsroom/press release/article) date. "
            "Focus on the announcement context/date rather than release availability. "
            "If the sources do not state 'first', you should still PASS if they clearly confirm the chip announcement and date and "
            "there is no indication of an earlier Apple Silicon chip announcement in 2024 in the provided sources. "
            "Prefer official Apple newsroom posts or credible publications with identifiable site name and article title."
        )
    )

    # --- Step 2: Provide required info for the identified chip (critical, parallel) ---
    info_node = evaluator.add_parallel(
        id="provide_required_info",
        desc="Provide all required outputs for the identified chip.",
        parent=task_node,
        critical=True
    )

    # 2.a Structured Format (critical)
    structured_node = evaluator.add_custom_node(
        result=_structured_format_ok(chip),
        id="structured_format",
        desc=(
            "Present the answer in a structured format with labeled fields covering: "
            "chip model, announcement date, source URL, Neural Engine TOPS, improvement factor."
        ),
        parent=info_node,
        critical=True
    )

    # 2.b Announcement Date (critical)
    ann_date_leaf = evaluator.add_leaf(
        id="announcement_date",
        desc="Provide the exact official announcement date for the identified chip.",
        parent=info_node,
        critical=True
    )
    ann_date_claim = (
        f"The chip '{chip.chip_model}' was officially announced on '{chip.announcement_date}'."
    )
    await evaluator.verify(
        claim=ann_date_claim,
        node=ann_date_leaf,
        sources=chip.source_urls,
        additional_instruction=(
            "Confirm the official announcement date from the page content (Apple newsroom or credible publication article). "
            "Minor formatting variations (e.g., 'May 7, 2024' vs '2024-05-07') are acceptable if they represent the same date."
        )
    )

    # 2.c Announcement Source URL (critical)
    source_url_leaf = evaluator.add_leaf(
        id="announcement_source_url",
        desc=(
            "Provide at least one URL that explicitly confirms the chip announcement. "
            "The URL must be either an official Apple domain page or a credible technology/news publication article page with identifiable metadata."
        ),
        parent=info_node,
        critical=True
    )
    source_url_claim = (
        f"At least one provided source URL confirms the announcement of '{chip.chip_model}' on or around '{chip.announcement_date}', "
        "and is either an official Apple domain page or a credible technology/news publication article page with identifiable metadata."
    )
    await evaluator.verify(
        claim=source_url_claim,
        node=source_url_leaf,
        sources=chip.source_urls,
        additional_instruction=(
            "Pass if ANY one of the provided URLs is an Apple domain page (e.g., newsroom.apple.com or apple.com) confirming the announcement, "
            "OR a credible tech/news publication article page (e.g., theverge.com, techcrunch.com, wired.com, macrumors.com) "
            "that clearly shows a site name and an article title and confirms the announcement. "
            "Do NOT accept user-generated content platforms such as wikis, forums, or social media posts."
        )
    )

    # 2.d Neural Engine TOPS (critical)
    tops_leaf = evaluator.add_leaf(
        id="neural_engine_tops",
        desc="Report the Neural Engine performance specification measured in TOPS.",
        parent=info_node,
        critical=True
    )
    tops_claim = (
        f"The Neural Engine performance for '{chip.chip_model}' is '{chip.neural_engine_tops}'."
    )
    await evaluator.verify(
        claim=tops_claim,
        node=tops_leaf,
        sources=chip.source_urls,
        additional_instruction=(
            "Verify that the page explicitly states the Neural Engine performance in TOPS for this chip. "
            "Accept phrasing like 'up to X TOPS'. Focus on the numeric value and unit TOPS."
        )
    )

    # 2.e Neural Engine Improvement Factor vs M3 18 TOPS (critical)
    factor_leaf = evaluator.add_leaf(
        id="neural_engine_improvement_factor_vs_m3_18tops",
        desc="Report the improvement factor vs the M3-family baseline of 18 TOPS, consistent with chip_TOPS / 18.",
        parent=info_node,
        critical=True
    )

    chip_tops = _parse_tops_float(chip.neural_engine_tops)
    factor_val = _parse_factor_float(chip.improvement_factor_vs_m3)

    if chip_tops is not None and factor_val is not None:
        factor_claim = (
            f"Given the chip Neural Engine TOPS value of {chip_tops} and the M3-family baseline of 18 TOPS, "
            f"the improvement factor is approximately {chip_tops/18:.2f}x, which matches the stated '{chip.improvement_factor_vs_m3}'."
        )
    else:
        # If we cannot parse, craft a claim that will likely fail correctly
        factor_claim = (
            "The stated improvement factor matches chip_TOPS / 18, with chip_TOPS and the factor both provided and parsable."
        )

    await evaluator.verify(
        claim=factor_claim,
        node=factor_leaf,
        additional_instruction=(
            "Check the arithmetic: improvement_factor ≈ chip_TOPS / 18. "
            "Allow reasonable rounding (e.g., 2.11x for 38/18 ≈ 2.111...). "
            "FAIL if either value is missing or the factor does not match the computed ratio within typical rounding precision."
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
    model: str = "o4-mini",
) -> Dict:
    """
    Evaluate the agent's answer for the Apple 2024 first-announced Silicon chip task.
    """
    # Initialize evaluator
    evaluator = Evaluator()
    evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Root stays non-critical; we add a critical task node under it
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

    # Extract chip info from the answer
    chip_info = await evaluator.extract(
        prompt=prompt_extract_chip_info(),
        template_class=ChipInfo,
        extraction_name="chip_info"
    )

    # Build verification tree and execute checks
    await build_verification_tree(evaluator, chip_info)

    # Return structured summary
    return evaluator.get_summary()