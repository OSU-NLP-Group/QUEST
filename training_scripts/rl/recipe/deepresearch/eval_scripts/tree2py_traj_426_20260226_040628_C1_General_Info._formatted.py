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
TASK_ID = "honnold_birth_month"
TASK_DESCRIPTION = "In what month were both of Alex Honnold's daughters, June and Alice, born?"


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class BirthMonthsExtraction(BaseModel):
    """
    Extraction model for the answer's stated birth month(s) of Alex Honnold's daughters.
    - If the answer provides a single month for both daughters, put it in both_month.
    - If the answer specifies per-child months, fill june_month and/or alice_month.
    - Years are optional and not required; extract if explicitly present.
    - Extract any URLs present in the answer (as-is), regardless of whether they are tied to a specific daughter.
    """
    both_month: Optional[str] = None
    june_month: Optional[str] = None
    june_year: Optional[str] = None
    alice_month: Optional[str] = None
    alice_year: Optional[str] = None
    source_urls: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_birth_months() -> str:
    return """
    Extract the birth month information from the answer for Alex Honnold's daughters:
    - both_month: If the answer states a single month that applies to both daughters, put that month here (e.g., "February" or "Feb"). If not provided, set to null.
    - june_month: If the answer explicitly names a month for June Honnold, extract that month (e.g., "February", "Feb"); otherwise null.
    - june_year: If the answer explicitly gives a year for June's birth (e.g., "2022"), extract it; otherwise null.
    - alice_month: If the answer explicitly names a month for Alice Honnold, extract that month; otherwise null.
    - alice_year: If the answer explicitly gives a year for Alice's birth (e.g., "2024"), extract it; otherwise null.
    - source_urls: Extract every URL present in the answer text. Include URLs that appear in plain text or inside markdown links. Do not invent or infer URLs.

    Rules:
    - Do not infer or guess any months or years. Only extract what the answer explicitly states.
    - Accept common month abbreviations (e.g., "Feb" for "February") as valid values when extracting.
    - If the answer only says something like "both were born in February", put "February" (or the exact form used, like "Feb") in both_month and leave the per-child fields null unless they are also given explicitly.
    """


# --------------------------------------------------------------------------- #
# Verification helper                                                         #
# --------------------------------------------------------------------------- #
def additional_instruction_for_month_check(daughter_name: str) -> str:
    return (
        "Judge correctness by looking only at the provided answer text. "
        f"Count this claim as correct if the answer explicitly (or implicitly via 'both daughters') "
        f"indicates that {daughter_name} was born in February. "
        "Accept common abbreviations like 'Feb' and ignore letter case. "
        "If the answer states that BOTH daughters were born in February, "
        f"that counts for {daughter_name} even if her name is not repeated individually. "
        "Only the month matters; ignore any day or year details."
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
    model: str = "o4-mini"
) -> Dict:
    """
    Evaluate an answer for the question:
    'In what month were both of Alex Honnold's daughters, June and Alice, born?'
    The expected correct month for both is February (June: Feb 2022; Alice: Feb 2024).
    """
    # 1) Initialize evaluator
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
        default_model=model
    )

    # 2) Extract structured info (for record/robustness; verification focuses directly on answer content)
    extracted = await evaluator.extract(
        prompt=prompt_extract_birth_months(),
        template_class=BirthMonthsExtraction,
        extraction_name="birth_months_extraction"
    )

    # 3) Add ground truth info (for transparency)
    evaluator.add_ground_truth(
        {
            "expected_month": "February",
            "expected_details": {
                "June Honnold": "February 2022",
                "Alice Honnold": "February 2024"
            }
        },
        gt_type="ground_truth_birth_month"
    )

    # 4) Build verification tree per rubric
    # Top-level critical node
    main_node = evaluator.add_parallel(
        id="Correct_Birth_Month_Identified",
        desc="The answer correctly identifies the calendar month in which both of Alex Honnold's daughters were born",
        parent=root,
        critical=True
    )

    # Two critical leaf checks (must both pass)
    june_leaf = evaluator.add_leaf(
        id="June_Birth_Month",
        desc="June Honnold's birth month is correctly identified as February (born February 2022)",
        parent=main_node,
        critical=True
    )
    alice_leaf = evaluator.add_leaf(
        id="Alice_Birth_Month",
        desc="Alice Honnold's birth month is correctly identified as February (born February 2024)",
        parent=main_node,
        critical=True
    )

    # 5) Verify leaves (use batch to avoid sibling-precondition skipping effects)
    claims_and_sources = [
        (
            "According to the answer, June Honnold was born in February.",
            None,  # No external sources needed; we are checking the answer's stated month
            june_leaf,
            additional_instruction_for_month_check("June Honnold"),
        ),
        (
            "According to the answer, Alice Honnold was born in February.",
            None,
            alice_leaf,
            additional_instruction_for_month_check("Alice Honnold"),
        ),
    ]
    await evaluator.batch_verify(claims_and_sources)

    # Optionally record extracted summary as custom info (not required; extraction already recorded)
    evaluator.add_custom_info(
        {
            "both_month": extracted.both_month,
            "june_month": extracted.june_month,
            "june_year": extracted.june_year,
            "alice_month": extracted.alice_month,
            "alice_year": extracted.alice_year,
            "source_urls_found_in_answer": extracted.source_urls,
        },
        info_type="extracted_summary",
        info_name="extracted_birth_months_summary"
    )

    # 6) Return evaluation summary
    return evaluator.get_summary()