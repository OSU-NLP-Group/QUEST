import asyncio
import logging
import re
from typing import Optional, List, Dict, Any
from urllib.parse import urlparse

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.llm_client.base_client import LLMClient
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy

# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "cs_top5_study_abroad_gpa_2026"
TASK_DESCRIPTION = (
    "Identify a university that is ranked in the top 5 for undergraduate computer science programs "
    "according to the U.S. News & World Report 2026 rankings and has a minimum GPA requirement of 3.0 "
    "or lower for general study abroad program eligibility. Provide the university name, its exact ranking "
    "position for undergraduate computer science in the U.S. News 2026 rankings, the stated minimum GPA "
    "requirement for study abroad, and direct links to both the U.S. News ranking page and the university's "
    "official study abroad eligibility policy page."
)

# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class UniversitySelection(BaseModel):
    """All required fields describing the identified university and evidence links."""
    university_name: Optional[str] = None
    cs_ranking_position_2026: Optional[str] = None
    us_news_ranking_url: Optional[str] = None
    study_abroad_min_gpa: Optional[str] = None
    study_abroad_policy_url: Optional[str] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_university_selection() -> str:
    return """
    Extract the single university identified in the answer that claims to satisfy BOTH of the following:
    1) It is ranked in the TOP 5 (#1 through #5, ties included) for UNDERGRADUATE Computer Science in the U.S. News & World Report 2026 rankings.
    2) It has a publicly stated MINIMUM GPA REQUIREMENT of 3.0 or LOWER for GENERAL study abroad program eligibility (campus-wide policy; NOT a program-specific requirement).

    Return a JSON object with these fields:
    - university_name: The university's name as presented.
    - cs_ranking_position_2026: The EXACT ranking position string stated for UNDERGRADUATE computer science in the U.S. News 2026 rankings (e.g., "#3", "3 (tie)", "No. 5", "T-4").
    - us_news_ranking_url: A direct URL to the relevant U.S. News ranking page that supports the stated UG CS 2026 ranking for this university.
    - study_abroad_min_gpa: The stated minimum GPA requirement for GENERAL study abroad eligibility (e.g., "3.0", "2.75", "minimum GPA of 3.0").
    - study_abroad_policy_url: A direct URL to the university's OFFICIAL policy page describing general study abroad eligibility and minimum GPA requirement.

    Rules:
    - Extract ONLY what is explicitly given in the answer. Do not invent or infer missing information.
    - If any required field is missing, set it to null.
    - For URLs, extract the full URL strings. Accept plain URLs or markdown links; return the actual URL.
    - The U.S. News link SHOULD be on the usnews.com domain and should correspond to the 2026 UNDERGRADUATE Computer Science ranking context.
    - The study abroad policy link SHOULD be on the university's official domain (commonly ending in .edu).
    """


# --------------------------------------------------------------------------- #
# Helper parsing utilities                                                    #
# --------------------------------------------------------------------------- #
_RANK_WORD_TO_INT = {
    "first": 1, "1st": 1,
    "second": 2, "2nd": 2,
    "third": 3, "3rd": 3,
    "fourth": 4, "4th": 4,
    "fifth": 5, "5th": 5,
}


def parse_rank_position_to_int(rank_text: Optional[str]) -> Optional[int]:
    """
    Try to parse an integer ranking from a free-form ranking text.
    Robust against formats like "#3", "3 (tie)", "No. 5", "T-4".
    Also handles words like "first", "second", etc.
    Returns None if no reasonable integer 1..100 found.
    """
    if not rank_text:
        return None

    s = rank_text.strip().lower()

    # Check word-based ranks first
    for k, v in _RANK_WORD_TO_INT.items():
        if k in s:
            return v

    # Try explicit numeric patterns with signals
    candidates: List[int] = []

    # Common markers near rank numbers
    # Extract all integers; filter out very large ones like years (e.g., 2026)
    for m in re.finditer(r"\b(\d{1,3})\b", s):
        try:
            val = int(m.group(1))
            if 1 <= val <= 100:
                candidates.append(val)
        except Exception:
            continue

    if not candidates:
        return None

    # Heuristic: choose the smallest reasonable integer to avoid picking "2026" or similar
    return min(candidates)


def parse_gpa_value(gpa_text: Optional[str]) -> Optional[float]:
    """
    Extract a GPA numeric value from a free-form text. Looks for numbers between 0.0 and 4.0.
    Returns the smallest valid number if multiple are present (to be conservative).
    Returns None if not found.
    """
    if not gpa_text:
        return None

    s = gpa_text.lower()
    nums: List[float] = []
    for m in re.finditer(r"\b(\d(?:\.\d{1,2})?)\b", s):
        try:
            val = float(m.group(1))
            if 0.0 <= val <= 4.0:
                nums.append(val)
        except Exception:
            continue

    if not nums:
        return None

    return min(nums)


def is_usnews_url(url: Optional[str]) -> bool:
    """Check if the URL belongs to the U.S. News domain."""
    if not url:
        return False
    try:
        netloc = urlparse(url).netloc.lower()
        return "usnews.com" in netloc
    except Exception:
        return False


def is_edu_domain(url: Optional[str]) -> bool:
    """Check if the URL is on an .edu domain (typical for official university pages)."""
    if not url:
        return False
    try:
        netloc = urlparse(url).netloc.lower()
        return netloc.endswith(".edu") or ".edu" in netloc
    except Exception:
        return False


# --------------------------------------------------------------------------- #
# Verification subtrees                                                       #
# --------------------------------------------------------------------------- #
async def build_cs_ranking_verification(
    evaluator: Evaluator,
    parent_node,
    info: UniversitySelection,
) -> None:
    """
    Build and execute the CS Ranking Verification subtree.
    Critical, sequential:
      1) Required info present
      2) US News URL domain check (usnews.com)
      3) Verify-by-URL: UG CS 2026 ranking position matches stated
      4) Custom: Top 5 check (<= 5 after parsing; ties allowed)
    """
    node = evaluator.add_sequential(
        id="CS_Ranking_Verification",
        desc=(
            "The university must be ranked in the top 5 for undergraduate computer science programs "
            "according to the U.S. News & World Report 2026 rankings (positions #1 through #5, including ties). "
            "The exact ranking position must be stated, and a direct link to the U.S. News ranking page must be provided."
        ),
        parent=parent_node,
        critical=True,
    )

    # 1) Required info
    required_ok = all([
        bool(info.university_name and info.university_name.strip()),
        bool(info.cs_ranking_position_2026 and info.cs_ranking_position_2026.strip()),
        bool(info.us_news_ranking_url and info.us_news_ranking_url.strip()),
    ])
    evaluator.add_custom_node(
        result=required_ok,
        id="cs_required_info_present",
        desc="University name, exact UG CS 2026 ranking position, and U.S. News ranking URL are provided.",
        parent=node,
        critical=True,
    )

    # 2) Domain check for U.S. News URL
    evaluator.add_custom_node(
        result=is_usnews_url(info.us_news_ranking_url),
        id="cs_usnews_domain_valid",
        desc="U.S. News ranking URL is from usnews.com domain.",
        parent=node,
        critical=True,
    )

    # 3) Verify-by-URL: ranking position and category/year supported
    rank_leaf = evaluator.add_leaf(
        id="cs_ranking_supported_by_usnews",
        desc="U.S. News page supports the stated UG CS 2026 ranking position for the university.",
        parent=node,
        critical=True,
    )
    claim_rank = (
        f"On the U.S. News & World Report 2026 'Best Undergraduate Computer Science Programs' ranking page, "
        f"{info.university_name or ''} is ranked {info.cs_ranking_position_2026 or ''} for undergraduate computer science. "
        f"Ties count as the same position."
    )
    await evaluator.verify(
        claim=claim_rank,
        node=rank_leaf,
        sources=info.us_news_ranking_url,
        additional_instruction=(
            "Confirm the category is Undergraduate Computer Science and the year is 2026. "
            "Verify the stated position (allowing tie notation such as 'tie' or 'T-'). "
            "Minor formatting variations (e.g., '#3', '3 (tie)', 'No. 3') should be treated equivalently."
        ),
    )

    # 4) Top 5 check (custom, purely logical on extracted position)
    parsed_rank = parse_rank_position_to_int(info.cs_ranking_position_2026)
    evaluator.add_custom_node(
        result=(parsed_rank is not None and parsed_rank <= 5),
        id="cs_top5_check",
        desc="The stated UG CS ranking position is within top 5 (#1–#5 inclusive, ties allowed).",
        parent=node,
        critical=True,
    )

    # Record parsed value for transparency
    evaluator.add_custom_info(
        info={"parsed_cs_rank_int": parsed_rank},
        info_type="parsed_values",
        info_name="cs_ranking_parsed_value",
    )


async def build_study_abroad_gpa_verification(
    evaluator: Evaluator,
    parent_node,
    info: UniversitySelection,
) -> None:
    """
    Build and execute the Study Abroad GPA Verification subtree.
    Critical, sequential:
      1) Required info present
      2) Policy URL official domain (.edu)
      3) Verify-by-URL: page states general minimum GPA requirement value
      4) Custom: GPA <= 3.0 check
    """
    node = evaluator.add_sequential(
        id="Study_Abroad_GPA_Verification",
        desc=(
            "The university must have a publicly stated minimum GPA requirement of 3.0 or lower for general study abroad program eligibility "
            "(not program-specific requirements). The specific GPA requirement value must be stated, and a direct link to the university's official "
            "study abroad eligibility policy page must be provided."
        ),
        parent=parent_node,
        critical=True,
    )

    # 1) Required info
    required_ok = all([
        bool(info.study_abroad_min_gpa and info.study_abroad_min_gpa.strip()),
        bool(info.study_abroad_policy_url and info.study_abroad_policy_url.strip()),
    ])
    evaluator.add_custom_node(
        result=required_ok,
        id="gpa_required_info_present",
        desc="Minimum GPA requirement and official study abroad policy URL are provided.",
        parent=node,
        critical=True,
    )

    # 2) Policy URL official domain check (.edu)
    evaluator.add_custom_node(
        result=is_edu_domain(info.study_abroad_policy_url),
        id="gpa_policy_domain_official",
        desc="Study abroad policy URL is on an official university domain (.edu).",
        parent=node,
        critical=True,
    )

    # 3) Verify-by-URL that page states general minimum GPA requirement value (not program-specific)
    gpa_leaf = evaluator.add_leaf(
        id="gpa_requirement_supported_by_policy",
        desc="Policy page states the general minimum GPA requirement for campus-wide study abroad eligibility.",
        parent=node,
        critical=True,
    )
    claim_gpa = (
        f"The official study abroad eligibility policy page for {info.university_name or ''} states a general minimum GPA requirement of "
        f"{info.study_abroad_min_gpa or ''} for campus-wide study abroad eligibility (not program-specific)."
    )
    await evaluator.verify(
        claim=claim_gpa,
        node=gpa_leaf,
        sources=info.study_abroad_policy_url,
        additional_instruction=(
            "Confirm the page is a general, university-wide policy for study abroad eligibility, not a single program's requirement. "
            "Verify that the minimum GPA requirement value is explicitly stated on the page."
        ),
    )

    # 4) Custom GPA <= 3.0 check
    parsed_gpa = parse_gpa_value(info.study_abroad_min_gpa)
    evaluator.add_custom_node(
        result=(parsed_gpa is not None and parsed_gpa <= 3.0),
        id="gpa_threshold_leq_3_0",
        desc="The stated minimum GPA requirement is 3.0 or lower.",
        parent=node,
        critical=True,
    )

    # Record parsed value for transparency
    evaluator.add_custom_info(
        info={"parsed_min_gpa": parsed_gpa},
        info_type="parsed_values",
        info_name="study_abroad_gpa_parsed_value",
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
    model: str = "o4-mini",
) -> Dict[str, Any]:
    """
    Evaluate an answer for the CS top-5 & study abroad GPA 2026 task and return a structured summary.
    """
    # Initialize evaluator
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Overall aggregation; we'll add a critical child node for the actual rubric root.
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

    # Extract the selection info from the answer
    selection_info = await evaluator.extract(
        prompt=prompt_extract_university_selection(),
        template_class=UniversitySelection,
        extraction_name="university_selection",
    )

    # Build rubric root (critical)
    rubric_root = evaluator.add_parallel(
        id="University_Identification",
        desc=(
            "The identified university must satisfy both the CS ranking criterion and the study abroad GPA requirement criterion, "
            "with all required information and reference URLs provided."
        ),
        parent=root,
        critical=True,
    )

    # Build subtrees
    await build_cs_ranking_verification(evaluator, rubric_root, selection_info)
    await build_study_abroad_gpa_verification(evaluator, rubric_root, selection_info)

    # Return structured result
    return evaluator.get_summary()