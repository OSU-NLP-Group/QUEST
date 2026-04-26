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
TASK_ID = "maduro_next_court_date_2026"
TASK_DESCRIPTION = "What is the date of Nicolás Maduro's next scheduled court appearance following his initial arraignment in U.S. federal court in January 2026?"


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class NextCourtDateExtraction(BaseModel):
    next_court_date: Optional[str] = None
    urls: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_next_court_date() -> str:
    return """
    From the provided answer text:
    1) Extract the specific date string that the answer claims is Nicolás Maduro's next scheduled court appearance following his initial arraignment in January 2026. Return it exactly as written in the answer (e.g., "January 30, 2026", "Jan. 30, 2026", "1/30/2026", or "2026-01-30").
       - If multiple dates are mentioned, choose the one labeled or described as the "next" scheduled court appearance after the January 2026 arraignment.
       - If no such date is provided, return null.
    2) Extract all URLs cited in the answer that are used to support this date (e.g., court records, government websites, or established news outlets). Return them in the 'urls' array. If none are present, return an empty array.

    Output fields:
    - next_court_date: string or null
    - urls: array of URL strings
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
_MONTH_REGEX = r"(?:Jan(?:\.|uary)?|Feb(?:\.|ruary)?|Mar(?:\.|ch)?|Apr(?:\.|il)?|May|Jun(?:\.|e)?|Jul(?:\.|y)?|Aug(?:\.|ust)?|Sep(?:\.|t\.|tember)?|Oct(?:\.|ober)?|Nov(?:\.|ember)?|Dec(?:\.|ember)?)"

def looks_like_specific_date(date_str: Optional[str]) -> bool:
    """
    Heuristic check that a provided string is a specific calendar date including month/day/year.
    Accepts formats like:
      - January 30, 2026 / Jan 30, 2026 / Jan. 30, 2026
      - 1/30/2026
      - 2026-01-30
    """
    if not date_str:
        return False
    s = date_str.strip()
    if not s:
        return False

    patterns = [
        rf"\b{_MONTH_REGEX}\s+\d{{1,2}},\s*\d{{4}}\b",   # MonthName DD, YYYY (with optional dot in abbrev)
        r"\b\d{1,2}/\d{1,2}/\d{4}\b",                   # MM/DD/YYYY
        r"\b\d{4}-\d{1,2}-\d{1,2}\b",                   # YYYY-MM-DD
    ]

    for pat in patterns:
        if re.search(pat, s, flags=re.IGNORECASE):
            return True
    return False


def _filter_valid_urls(urls: List[str]) -> List[str]:
    valid = []
    for u in urls:
        if isinstance(u, str):
            us = u.strip()
            if us.lower().startswith("http://") or us.lower().startswith("https://"):
                valid.append(us)
    return valid


# --------------------------------------------------------------------------- #
# Verification tree construction & checks                                     #
# --------------------------------------------------------------------------- #
async def build_and_verify(evaluator: Evaluator, parent_node, extracted: NextCourtDateExtraction) -> None:
    """
    Build the verification tree according to the rubric and execute verifications.
    """
    main_node = evaluator.add_parallel(
        id="next_court_date_identification",
        desc="Verify that the answer correctly identifies the date of Nicolás Maduro's next scheduled court appearance following his initial arraignment in January 2026",
        parent=parent_node,
        critical=True
    )

    # Child 1: Date Provided (custom, critical)
    date_is_specific = looks_like_specific_date(extracted.next_court_date)
    evaluator.add_custom_node(
        result=date_is_specific,
        id="date_provided",
        desc="A specific date (month, day, and year) is provided in the answer",
        parent=main_node,
        critical=True
    )

    # Prepare sources
    sources = _filter_valid_urls(extracted.urls if extracted and extracted.urls else [])
    date_str = extracted.next_court_date or ""

    # Child 2: Source Verification (leaf, critical)
    source_ver_node = evaluator.add_leaf(
        id="source_verification",
        desc="The date is supported by at least one credible reference URL (e.g., from court records, government websites, or established news organizations)",
        parent=main_node,
        critical=True
    )

    if sources and date_str:
        claim_src = f"According to the provided webpage, Nicolás Maduro's next scheduled court appearance is on {date_str}."
        await evaluator.verify(
            claim=claim_src,
            node=source_ver_node,
            sources=sources,
            additional_instruction=(
                "Verify that the page explicitly supports that this date is a scheduled court appearance. "
                "Allow synonyms like 'hearing', 'status conference', 'court date', 'appearance', or similar. "
                "If the page does not support this date, mark as not supported."
            ),
        )
    else:
        source_ver_node.score = 0.0
        source_ver_node.status = "failed"

    # Child 3: Temporal Accuracy (leaf, critical)
    temporal_node = evaluator.add_leaf(
        id="temporal_accuracy",
        desc="The provided date is explicitly described in the source as the next scheduled court appearance following the initial January 2026 arraignment",
        parent=main_node,
        critical=True
    )

    if sources and date_str:
        claim_temp = (
            f"The webpage states that after Nicolás Maduro's initial arraignment in January 2026, "
            f"the next scheduled court appearance is on {date_str}."
        )
        await evaluator.verify(
            claim=claim_temp,
            node=temporal_node,
            sources=sources,
            additional_instruction=(
                "Check that the webpage explicitly frames this date as the next scheduled court appearance "
                "following the initial arraignment in January 2026. Phrases such as 'next hearing', "
                "'next court date', or 'upcoming appearance after the arraignment' should count if clearly "
                "linked to the January 2026 arraignment as the preceding event."
            ),
        )
    else:
        temporal_node.score = 0.0
        temporal_node.status = "failed"


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
    Evaluate an answer for the 'next court date' task.
    """
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Root is parallel; main rubric node is critical parallel under root
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

    # Extract structured info from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_next_court_date(),
        template_class=NextCourtDateExtraction,
        extraction_name="next_court_date_info"
    )

    # Build and run verification checks
    await build_and_verify(evaluator, root, extracted)

    return evaluator.get_summary()