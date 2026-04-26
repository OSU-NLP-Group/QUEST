import asyncio
import logging
from typing import List, Optional, Dict, Any

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "tn_transfer_law_2026"
TASK_DESCRIPTION = (
    "What are the key provisions of Tennessee's newly signed one-time transfer law for middle and high school athletes? "
    "Specifically, provide: (1) the date when the law was signed and when it becomes effective, "
    "(2) how many transfers are allowed during grades 6-8 and during grades 9-12, and "
    "(3) when transfers must occur for students to gain immediate eligibility."
)

EXPECTED_FACTS = {
    "signing_date": "February 23, 2026",
    "effective_date": "July 1, 2026",
    "effective_school_year": "2026-27",
    "middle_school_transfers": "one",
    "high_school_transfers": "one",
    "timing_requirement_keyword": "summer between school years",
}


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class LawInfoExtraction(BaseModel):
    # Dates
    signing_date: Optional[str] = None
    signing_sources: List[str] = Field(default_factory=list)

    effective_date: Optional[str] = None
    effective_school_year: Optional[str] = None
    effective_sources: List[str] = Field(default_factory=list)

    # Transfer counts
    middle_school_transfers: Optional[str] = None  # e.g., "one", "1", "one-time"
    ms_sources: List[str] = Field(default_factory=list)

    high_school_transfers: Optional[str] = None  # e.g., "one", "1", "one-time"
    hs_sources: List[str] = Field(default_factory=list)

    # Timing requirement
    timing_requirement: Optional[str] = None  # e.g., "during the summer between school years"
    timing_sources: List[str] = Field(default_factory=list)

    # Overall URLs cited anywhere in the answer (fallback pool)
    overall_sources: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_tn_transfer_law() -> str:
    return """
    Extract from the answer the key facts about Tennessee's newly signed one-time transfer law for middle and high school athletes.
    Return a JSON object with the following fields (use exact strings from the answer where possible):

    1) signing_date: The date when the bill/law was signed (e.g., "February 23, 2026"). If not present, null.
       signing_sources: Array of URLs cited in the answer specifically supporting the signing date. If none, [].

    2) effective_date: The date when the law becomes effective (e.g., "July 1, 2026"). If not present, null.
       effective_school_year: The school year the law applies to when it becomes effective (e.g., "2026-27"). If not present, null.
       effective_sources: Array of URLs for the effective date/school year. If none, [].

    3) middle_school_transfers: How many transfers are allowed during grades 6-8 under the law (e.g., "one", "1", "one time"). If not present, null.
       ms_sources: Array of URLs for the middle school transfer rule. If none, [].

    4) high_school_transfers: How many transfers are allowed during grades 9-12 under the law (e.g., "one", "1", "one time"). If not present, null.
       hs_sources: Array of URLs for the high school transfer rule. If none, [].

    5) timing_requirement: The timing required for a transfer to gain immediate eligibility (e.g., "during the summer between school years"). If not present, null.
       timing_sources: Array of URLs for the timing requirement rule. If none, [].

    6) overall_sources: Array of all URLs cited anywhere in the answer (include Google Docs/Drive, news, state or TSSAA pages, legislative pages, PDFs, etc.). If none, [].

    IMPORTANT:
    - Only extract URLs that are explicitly present in the answer (including markdown links).
    - Keep all values as strings exactly as they appear in the answer. Do not normalize numbers to words or vice versa.
    - If a field is not present in the answer, return null (for string fields) or [] (for URL arrays).
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _combine_sources(*lists: List[str]) -> List[str]:
    """Combine and deduplicate lists of URLs while preserving order."""
    seen = set()
    out: List[str] = []
    for lst in lists:
        for url in lst or []:
            if isinstance(url, str):
                u = url.strip()
                if u and u not in seen:
                    seen.add(u)
                    out.append(u)
    return out


# --------------------------------------------------------------------------- #
# Verification logic                                                          #
# --------------------------------------------------------------------------- #
async def build_and_verify_tree(evaluator: Evaluator, extracted: LawInfoExtraction) -> None:
    """
    Build the verification tree according to the rubric and submit verifications.
    All nodes under the main rubric node are critical, matching rubric requirements.
    """

    # Top-level rubric node (critical, parallel)
    tn_root = evaluator.add_parallel(
        id="Tennessee_Transfer_Law_Information",
        desc="Provide accurate information about Tennessee's one-time transfer law for high school athletes",
        parent=evaluator.root,
        critical=True
    )

    # ------------------ Effective Date Information (critical) -----------------
    effective_info = evaluator.add_parallel(
        id="Effective_Date_Information",
        desc="Identify when the Tennessee one-time transfer law was signed and when it becomes effective",
        parent=tn_root,
        critical=True
    )

    # Law Signing Date (critical leaf)
    signing_leaf = evaluator.add_leaf(
        id="Law_Signing_Date",
        desc="Correctly identify that Senate Bill 16 was signed on February 23, 2026",
        parent=effective_info,
        critical=True
    )
    signing_claim_value = extracted.signing_date or ""
    signing_claim = (
        f"Tennessee's one-time transfer law (Senate Bill 16) was signed on {signing_claim_value}."
    )
    signing_sources = _combine_sources(extracted.signing_sources, extracted.overall_sources)
    await evaluator.verify(
        claim=signing_claim,
        node=signing_leaf,
        sources=signing_sources if signing_sources else None,
        additional_instruction=(
            "Only mark as supported if the cited webpage(s) clearly state that Tennessee's Senate Bill 16 "
            f"was signed on {EXPECTED_FACTS['signing_date']}. "
            "If the date in the claim differs, is ambiguous, or the webpage is unrelated, mark incorrect."
        )
    )

    # Implementation / Effective Date (critical leaf)
    implementation_leaf = evaluator.add_leaf(
        id="Implementation_Date",
        desc="Correctly identify that the law takes effect July 1, 2026, for the 2026-27 school year",
        parent=effective_info,
        critical=True
    )
    effective_date_val = extracted.effective_date or ""
    effective_year_val = extracted.effective_school_year or ""
    implementation_claim = (
        f"The law takes effect on {effective_date_val} for the {effective_year_val} school year."
    )
    effective_sources = _combine_sources(extracted.effective_sources, extracted.overall_sources)
    await evaluator.verify(
        claim=implementation_claim,
        node=implementation_leaf,
        sources=effective_sources if effective_sources else None,
        additional_instruction=(
            "Only mark as supported if the cited webpage(s) clearly state BOTH that the effective date is "
            f"{EXPECTED_FACTS['effective_date']} AND that it applies starting with the "
            f"{EXPECTED_FACTS['effective_school_year']} school year (or equivalent phrasing). "
            "If either part is missing or contradictory, mark incorrect."
        )
    )

    # --------------- Grade Level Transfer Provisions (critical) --------------
    grade_level_node = evaluator.add_parallel(
        id="Grade_Level_Transfer_Provisions",
        desc="Specify how many transfers are allowed in each grade range under the new Tennessee law",
        parent=tn_root,
        critical=True
    )

    # Middle School Transfers (grades 6-8) (critical leaf)
    ms_leaf = evaluator.add_leaf(
        id="Middle_School_Transfers",
        desc="Correctly state that one transfer is permitted during grades 6-8",
        parent=grade_level_node,
        critical=True
    )
    ms_val = extracted.middle_school_transfers or ""
    ms_claim = f"Under the new Tennessee law, a student is permitted {ms_val} transfer(s) during grades 6–8."
    ms_sources = _combine_sources(extracted.ms_sources, extracted.overall_sources)
    await evaluator.verify(
        claim=ms_claim,
        node=ms_leaf,
        sources=ms_sources if ms_sources else None,
        additional_instruction=(
            "Evaluate whether the cited page(s) explicitly state that the law permits exactly ONE transfer "
            "for grades 6–8 (middle school). Treat 'one', '1', and 'one time/one-time' as equivalent to one. "
            "If the claim implies more than one or is inconsistent, mark incorrect."
        )
    )

    # High School Transfers (grades 9-12) (critical leaf)
    hs_leaf = evaluator.add_leaf(
        id="High_School_Transfers",
        desc="Correctly state that one transfer is permitted during grades 9-12",
        parent=grade_level_node,
        critical=True
    )
    hs_val = extracted.high_school_transfers or ""
    hs_claim = f"Under the new Tennessee law, a student is permitted {hs_val} transfer(s) during grades 9–12."
    hs_sources = _combine_sources(extracted.hs_sources, extracted.overall_sources)
    await evaluator.verify(
        claim=hs_claim,
        node=hs_leaf,
        sources=hs_sources if hs_sources else None,
        additional_instruction=(
            "Evaluate whether the cited page(s) explicitly state that the law permits exactly ONE transfer "
            "for grades 9–12 (high school). Treat 'one', '1', and 'one time/one-time' as equivalent to one. "
            "If the claim implies more than one or is inconsistent, mark incorrect."
        )
    )

    # ------------------- Transfer Timing Requirement (critical) --------------
    timing_leaf = evaluator.add_leaf(
        id="Transfer_Timing_Requirement",
        desc="Correctly identify that transfers must occur during the summer between school years for immediate eligibility",
        parent=tn_root,
        critical=True
    )
    timing_val = extracted.timing_requirement or ""
    timing_claim = (
        f"To gain immediate athletic eligibility after a transfer, the transfer must occur {timing_val}."
    )
    timing_sources = _combine_sources(extracted.timing_sources, extracted.overall_sources)
    await evaluator.verify(
        claim=timing_claim,
        node=timing_leaf,
        sources=timing_sources if timing_sources else None,
        additional_instruction=(
            "Only mark as supported if the cited webpage(s) clearly require that the transfer occur during "
            "the summer between school years (summer break) to obtain immediate eligibility. "
            "Equivalent wording such as 'during the summer between school years' or 'summer break between school years' "
            "is acceptable. Mid-year transfers should not qualify for immediate eligibility."
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
) -> Dict[str, Any]:
    """
    Evaluate an answer for the Tennessee one-time transfer law task.
    """
    # Initialize evaluator
    evaluator = Evaluator()
    evaluator.initialize(
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

    # Extract structured info from answer
    extracted_info = await evaluator.extract(
        prompt=prompt_extract_tn_transfer_law(),
        template_class=LawInfoExtraction,
        extraction_name="tn_transfer_law_extraction",
    )

    # Record expected facts as ground truth metadata (for transparency)
    evaluator.add_ground_truth(
        {
            "expected_signing_date": EXPECTED_FACTS["signing_date"],
            "expected_effective_date": EXPECTED_FACTS["effective_date"],
            "expected_effective_school_year": EXPECTED_FACTS["effective_school_year"],
            "expected_ms_transfers": EXPECTED_FACTS["middle_school_transfers"],
            "expected_hs_transfers": EXPECTED_FACTS["high_school_transfers"],
            "expected_timing_keyword": EXPECTED_FACTS["timing_requirement_keyword"],
        },
        gt_type="expected_facts",
    )

    # Build verification tree and run checks
    await build_and_verify_tree(evaluator, extracted_info)

    # Return summary
    return evaluator.get_summary()