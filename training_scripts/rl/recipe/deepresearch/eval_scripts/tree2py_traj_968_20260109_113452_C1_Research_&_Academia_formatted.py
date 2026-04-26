import asyncio
import logging
import re
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "founding_month_1665_journal"
TASK_DESCRIPTION = "In what month was the world's longest-running scientific journal, which was founded in 1665, first published?"


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class FoundingMonthExtraction(BaseModel):
    """
    Extracted fields from the agent's answer:
    - first_publication_month: The explicit calendar month stated for the first publication.
    - journal_name: The journal name if mentioned in the answer.
    - source_urls: All URLs cited in the answer that could support the claim.
    """
    first_publication_month: Optional[str] = None
    journal_name: Optional[str] = None
    source_urls: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_founding_month() -> str:
    return """
    From the answer, extract the following fields:
    1) first_publication_month: The explicit calendar month (English) claimed for the journal's first publication in 1665.
       - If the answer does not clearly state a month (January–December), return null.
       - If multiple months appear in the text, choose the one explicitly associated with the journal’s first publication.
       - The value should be one of the full month names in English: 
         "January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December".
         If the answer uses an abbreviation (e.g., "Mar."), normalize to the full month name ("March").
    2) journal_name: The name of the journal if it is explicitly mentioned in the answer; otherwise return null.
    3) source_urls: All URLs explicitly present in the answer (if any) that could support the month claim or identify the journal.
       - Collect only valid URLs (plain or markdown links). If none are present, return an empty list.

    Do NOT invent or infer any information that is not explicitly in the answer.
    """


# --------------------------------------------------------------------------- #
# Helper: month normalization                                                  #
# --------------------------------------------------------------------------- #
_MONTH_ALIASES = {
    "January": ["january", "jan", "jan."],
    "February": ["february", "feb", "feb."],
    "March": ["march", "mar", "mar."],
    "April": ["april", "apr", "apr."],
    "May": ["may"],
    "June": ["june", "jun", "jun."],
    "July": ["july", "jul", "jul."],
    "August": ["august", "aug", "aug."],
    "September": ["september", "sep", "sep.", "sept", "sept."],
    "October": ["october", "oct", "oct."],
    "November": ["november", "nov", "nov."],
    "December": ["december", "dec", "dec."],
}


def normalize_month_name(text: Optional[str]) -> Optional[str]:
    """
    Normalize a textual month (possibly abbreviated) to a full English month name, or None if no month found.
    Uses word boundaries to avoid false positives.
    """
    if not text:
        return None
    text_l = text.strip().lower()
    for full, aliases in _MONTH_ALIASES.items():
        for a in aliases:
            pattern = r"\b" + re.escape(a) + r"\b"
            if re.search(pattern, text_l):
                return full
    return None


# --------------------------------------------------------------------------- #
# Verification logic                                                          #
# --------------------------------------------------------------------------- #
async def build_and_verify_tree(
    evaluator: Evaluator,
    extracted: FoundingMonthExtraction,
    parent_node_desc: str,
    answer_text: str,
) -> None:
    """
    Build the rubric tree and run verifications according to the provided rubric.
    """
    # Parent node (critical, parallel)
    root_task_node = evaluator.add_parallel(
        id="Founding_Month_Answer",
        desc=parent_node_desc,
        parent=evaluator.root,
        critical=True,
    )

    # Determine month presence from extraction, with light fallback from full answer text if extractor missed a clear month
    extracted_month_norm = normalize_month_name(extracted.first_publication_month)
    if not extracted_month_norm:
        # Light fallback: try to detect any month mention in the raw answer, just to avoid extractor miss
        detected_from_answer = normalize_month_name(answer_text)
        extracted_month_norm = detected_from_answer

    # Leaf 1: Mentions_A_Month_Not_Just_Year (custom binary node)
    mentions_month_node = evaluator.add_custom_node(
        result=extracted_month_norm is not None,
        id="Mentions_A_Month_Not_Just_Year",
        desc="Answer specifies an explicit calendar month (not only a year).",
        parent=root_task_node,
        critical=True,
    )

    # Leaf 2: Month_Is_Correct_For_Described_Journal (verification)
    correctness_node = evaluator.add_leaf(
        id="Month_Is_Correct_For_Described_Journal",
        desc="The stated month matches the actual first-publication month (in 1665) of the uniquely-identified journal described by the constraints.",
        parent=root_task_node,
        critical=True,
    )

    # Build the claim. Even if month is missing, we still call verify() so that it can be auto-skipped
    # due to the critical sibling precondition failure (Mentions_A_Month_...).
    month_for_claim = extracted_month_norm or "(no month provided)"
    # The journal described by the task constraints is:
    # – the world's longest-running scientific journal still published
    # – founded in 1665
    # – launched by Henry Oldenburg as first Secretary of the Royal Society
    # This uniquely refers to “Philosophical Transactions of the Royal Society”.
    claim = (
        f"The first issue of 'Philosophical Transactions of the Royal Society' was published in {month_for_claim} 1665."
    )

    # Use any URLs cited in the answer, if available; otherwise, verification falls back to simple verification.
    sources = extracted.source_urls if extracted and extracted.source_urls else []

    await evaluator.verify(
        claim=claim,
        node=correctness_node,
        sources=sources,
        additional_instruction=(
            "The journal referenced is the world's longest-running scientific journal still published, founded in 1665, "
            "and launched by Henry Oldenburg as the first Secretary of the Royal Society. "
            "Verify only the month of the first publication in 1665 for this journal. "
            "If the sources are provided, judge strictly by the webpage content (allowing minor variants like 'Mar.' vs 'March'). "
            "If no sources are provided, rely on your knowledge cautiously to judge the claim."
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
    model: str = "o4-mini",
) -> Dict:
    """
    Evaluate an answer for the 1665 founding-month question.

    Returns a structured summary with the verification tree and scores.
    """
    # Initialize evaluator
    evaluator = Evaluator()
    evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Root aggregation; our main node will be added under root
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

    # Extract fields from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_founding_month(),
        template_class=FoundingMonthExtraction,
        extraction_name="founding_month_extraction",
    )

    # Add ground truth info for transparency (not used for scoring directly)
    evaluator.add_ground_truth(
        {
            "journal_expected": "Philosophical Transactions of the Royal Society",
            "first_publication_month_expected": "March",
            "first_publication_year_expected": "1665",
        },
        gt_type="ground_truth",
    )

    # Optional: record normalized month we plan to verify
    evaluator.add_custom_info(
        info={
            "extracted_first_publication_month_raw": extracted.first_publication_month,
            "extracted_first_publication_month_normalized": normalize_month_name(extracted.first_publication_month),
            "journal_name_in_answer": extracted.journal_name,
            "source_urls_in_answer": extracted.source_urls,
        },
        info_type="extraction_debug",
        info_name="normalized_extraction_info",
    )

    # Build and verify according to rubric tree
    await build_and_verify_tree(
        evaluator=evaluator,
        extracted=extracted,
        parent_node_desc=(
            "Answer provides the correct month in which the journal described (world's longest-running scientific journal still "
            "published; founded in 1665; launched by Henry Oldenburg as first Secretary of the Royal Society) was first published."
        ),
        answer_text=answer,
    )

    # Return the final summary
    return evaluator.get_summary()