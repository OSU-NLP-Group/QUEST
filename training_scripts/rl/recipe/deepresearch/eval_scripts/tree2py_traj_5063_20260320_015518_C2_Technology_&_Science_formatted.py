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
TASK_ID = "verizon_outage_comparison"
TASK_DESCRIPTION = """
Compare Verizon's two most recent major nationwide outages that occurred in January 2026 and September 2024. Identify the duration of each outage and determine which one lasted longer. Then, specify the compensation amount that Verizon offered to customers affected by the longer outage.
""".strip()


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class OutageEventExtraction(BaseModel):
    """Information about a single outage event extracted from the agent's answer."""
    month_year: Optional[str] = None  # e.g., "January 2026" or "September 2024"
    duration_text: Optional[str] = None  # e.g., "about 10 hours", "9.5 hours"
    duration_hours: Optional[str] = None  # numeric string if present, e.g., "10", "9.5"
    sources: List[str] = Field(default_factory=list)  # URLs cited for this outage's duration


class CompensationExtraction(BaseModel):
    """Compensation details for the longer outage extracted from the agent's answer."""
    applies_to: Optional[str] = None  # which outage it applies to (e.g., "January 2026")
    amount_text: Optional[str] = None  # e.g., "$20 account credit"
    amount_value: Optional[str] = None  # e.g., "20", "$20"
    sources: List[str] = Field(default_factory=list)  # URLs cited for compensation


class OutageComparisonExtraction(BaseModel):
    """Overall extraction encompassing both outages, longer outage identification, and compensation."""
    january_2026: Optional[OutageEventExtraction] = None
    september_2024: Optional[OutageEventExtraction] = None
    longer_outage: Optional[str] = None  # The one the answer states is longer, e.g., "January 2026"
    longer_outage_reasoning: Optional[str] = None
    compensation: Optional[CompensationExtraction] = None


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_outage_comparison() -> str:
    return """
    Extract from the answer structured details about the two Verizon nationwide outages (January 2026 and September 2024), the identified longer outage, and any compensation details.

    For each of the two outages, extract as much as the answer explicitly provides:
    - january_2026:
        - month_year: The temporal label used (e.g., "January 2026")
        - duration_text: The stated duration phrase (e.g., "about 10 hours", "roughly 10 hours", "10+ hours")
        - duration_hours: If the answer presents a numeric duration (e.g., 10, 9.5), provide it as a string (e.g., "10", "9.5"). Otherwise return null.
        - sources: All URLs explicitly cited that support the January 2026 outage duration
    - september_2024:
        - month_year: The temporal label used (e.g., "September 2024")
        - duration_text: The stated duration phrase (e.g., "9.5 hours", "nine and a half hours")
        - duration_hours: If the answer presents a numeric duration (e.g., 9.5), provide it as a string. Otherwise null.
        - sources: All URLs explicitly cited that support the September 2024 outage duration

    Also extract:
    - longer_outage: The outage that the answer states lasted longer (use "January 2026" or "September 2024" in canonical form if possible)
    - longer_outage_reasoning: Optional brief reasoning text if present
    - compensation:
        - applies_to: Which outage this compensation is tied to (e.g., "January 2026")
        - amount_text: The exact phrase for the compensation amount as stated in the answer (e.g., "$20 account credit")
        - amount_value: If a numeric amount is stated, provide it as a string with the currency symbol if present (e.g., "$20" or "20")
        - sources: All URLs explicitly cited that support the compensation information

    Important:
    - Only extract URLs explicitly present in the answer text.
    - Do not invent or infer values that are not stated.
    - If any field is missing, return null for that field (or an empty list for sources).
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def _unique_nonempty_urls(urls: Optional[List[str]]) -> List[str]:
    if not urls:
        return []
    seen = set()
    cleaned = []
    for u in urls:
        if not u:
            continue
        s = u.strip()
        if not s:
            continue
        if s not in seen:
            seen.add(s)
            cleaned.append(s)
    return cleaned


# --------------------------------------------------------------------------- #
# Verification subroutines                                                    #
# --------------------------------------------------------------------------- #
async def add_and_verify_duration_nodes(
    evaluator: Evaluator,
    parent_node,
    extraction: OutageComparisonExtraction,
) -> None:
    """
    Build and verify the OutageDurationComparison parallel node with two critical leaves:
    - January 2026 duration ≈ 10 hours (source-grounded)
    - September 2024 duration ≈ 9.5 hours (source-grounded)
    """
    duration_node = evaluator.add_parallel(
        id="OutageDurationComparison",
        desc="Correctly identify the duration of both the January 2026 and September 2024 Verizon outages",
        parent=parent_node,
        critical=True
    )

    # January 2026 duration verification (critical leaf)
    jan_leaf = evaluator.add_leaf(
        id="January2026Duration",
        desc="State that the January 2026 Verizon outage lasted approximately 10 hours",
        parent=duration_node,
        critical=True
    )
    jan_sources = _unique_nonempty_urls(extraction.january_2026.sources if extraction.january_2026 else [])

    jan_claim = (
        "The nationwide Verizon outage in January 2026 lasted approximately 10 hours "
        "(i.e., around ten hours)."
    )
    await evaluator.verify(
        claim=jan_claim,
        node=jan_leaf,
        sources=jan_sources,
        additional_instruction=(
            "Allow approximate phrasing such as 'about 10 hours', 'roughly 10 hours', or 'around ten hours'. "
            "Treat durations in the ~9.5 to ~10.5 hour range as equivalent to 'approximately 10 hours'. "
            "Verify that at least one cited source explicitly supports this duration range for the January 2026 outage."
        )
    )

    # September 2024 duration verification (critical leaf)
    sep_leaf = evaluator.add_leaf(
        id="September2024Duration",
        desc="State that the September 2024 Verizon outage lasted 9.5 hours",
        parent=duration_node,
        critical=True
    )
    sep_sources = _unique_nonempty_urls(extraction.september_2024.sources if extraction.september_2024 else [])

    sep_claim = (
        "The nationwide Verizon outage in September 2024 lasted approximately 9.5 hours "
        "(i.e., about nine and a half hours)."
    )
    await evaluator.verify(
        claim=sep_claim,
        node=sep_leaf,
        sources=sep_sources,
        additional_instruction=(
            "Consider '9.5 hours' equivalent to '9 hours 30 minutes' or phrasing like 'about nine and a half hours'. "
            "Allow small rounding tolerance (e.g., 9.4–9.6 hours). "
            "Verify that at least one cited source explicitly supports this duration for the September 2024 outage."
        )
    )


async def add_and_verify_longer_outage_node(
    evaluator: Evaluator,
    parent_node,
    extraction: OutageComparisonExtraction,
) -> None:
    """
    Add and verify the LongerOutageIdentification leaf (simple logical check based on durations).
    """
    longer_leaf = evaluator.add_leaf(
        id="LongerOutageIdentification",
        desc="Correctly identify that the January 2026 outage was longer than the September 2024 outage",
        parent=parent_node,
        critical=True
    )

    # Build a simple logical verification claim using the answer context
    jan_txt = extraction.january_2026.duration_text if extraction.january_2026 else None
    sep_txt = extraction.september_2024.duration_text if extraction.september_2024 else None

    claim = (
        "Between the two outages discussed (January 2026 and September 2024), "
        "the January 2026 outage was longer than the September 2024 outage."
    )
    add_ins = (
        "Use the answer context to reason: "
        f"January 2026 duration in the answer: '{jan_txt}'. "
        f"September 2024 duration in the answer: '{sep_txt}'. "
        "Conceptually, 'approximately 10 hours' is longer than 'approximately 9.5 hours'. "
        "Minor rounding differences do not change the relative ordering."
    )
    await evaluator.verify(
        claim=claim,
        node=longer_leaf,
        additional_instruction=add_ins
    )


async def add_and_verify_compensation_node(
    evaluator: Evaluator,
    parent_node,
    extraction: OutageComparisonExtraction,
) -> None:
    """
    Add a compensation info group (critical) and verify that Verizon offered $20 account credits
    to customers affected by the January 2026 outage (source-grounded).
    """
    comp_group = evaluator.add_parallel(
        id="CompensationInformation",
        desc="Identify the compensation amount offered for the longer outage",
        parent=parent_node,
        critical=True
    )

    comp_leaf = evaluator.add_leaf(
        id="CompensationAmount",
        desc="State that Verizon offered $20 account credits to customers affected by the January 2026 outage",
        parent=comp_group,
        critical=True
    )

    comp = extraction.compensation or CompensationExtraction()
    comp_sources = _unique_nonempty_urls(comp.sources)

    comp_claim = (
        "Verizon offered $20 account credits to customers affected by the January 2026 outage."
    )
    await evaluator.verify(
        claim=comp_claim,
        node=comp_leaf,
        sources=comp_sources,
        additional_instruction=(
            "Confirm via the cited sources that Verizon publicly offered $20 credits "
            "to customers impacted by the January 2026 nationwide outage. "
            "Treat 'account credit' and 'bill credit' as equivalent phrasing. "
            "Ensure the compensation is tied specifically to the January 2026 outage."
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
    Evaluate an answer for the Verizon outage comparison task.
    """
    # Initialize evaluator and root
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

    # Extract structured information from the answer
    extraction = await evaluator.extract(
        prompt=prompt_extract_outage_comparison(),
        template_class=OutageComparisonExtraction,
        extraction_name="outage_comparison_extraction"
    )

    # Add expected ground truth (for logging/traceability; not used directly in scoring)
    evaluator.add_ground_truth({
        "expected": {
            "january_2026_duration": "approximately 10 hours",
            "september_2024_duration": "approximately 9.5 hours",
            "longer_outage": "January 2026",
            "compensation_for_longer_outage": "$20 account credits (January 2026 outage)"
        }
    }, gt_type="ground_truth")

    # Build the verification tree according to the rubric
    comparison_root = evaluator.add_sequential(
        id="VerizonOutageComparison",
        desc="Compare Verizon's January 2026 and September 2024 outages to determine which lasted longer, and identify the compensation offered for the longer outage",
        parent=root,
        critical=True
    )

    # 1) Duration comparison (parallel, critical)
    await add_and_verify_duration_nodes(evaluator, comparison_root, extraction)

    # 2) Longer outage identification (leaf, critical)
    await add_and_verify_longer_outage_node(evaluator, comparison_root, extraction)

    # 3) Compensation information (group -> leaf, both critical)
    await add_and_verify_compensation_node(evaluator, comparison_root, extraction)

    # Return final structured summary
    return evaluator.get_summary()