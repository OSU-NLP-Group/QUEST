import asyncio
import logging
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.llm_client.base_client import LLMClient


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "jif_fast_decision_journal"
TASK_DESCRIPTION = """
Identify a multidisciplinary open access journal that has a 2024 Journal Impact Factor greater than 15.0 and provides a median time from submission to first editorial decision of 10 days or less. Provide the journal name and reference URL(s) supporting both metrics.
"""


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class JournalInfo(BaseModel):
    journal_name: Optional[str] = None

    # Journal type claims (as written in the answer; strings preferred for flexibility)
    open_access_claim: Optional[str] = None
    multidisciplinary_claim: Optional[str] = None

    # Impact factor details
    impact_factor_value: Optional[str] = None  # e.g., "17.2" or "17"
    impact_factor_year: Optional[str] = None   # ideally "2024"
    impact_factor_urls: List[str] = Field(default_factory=list)

    # Review speed details
    review_speed_median_days: Optional[str] = None  # e.g., "8" or "8 days"
    review_speed_urls: List[str] = Field(default_factory=list)

    # General URLs about the journal (homepage, aims & scope, about pages) if provided
    general_urls: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_journal_info() -> str:
    return """
    Extract the key details about the identified journal exactly as presented in the answer.

    Required fields to extract:
    1) journal_name: The full journal name mentioned.
    2) open_access_claim: If the answer explicitly states the journal is open access (OA), extract the exact phrase or short statement; otherwise null.
    3) multidisciplinary_claim: If the answer explicitly states the journal is multidisciplinary (covers multiple disciplines), extract the exact phrase or short statement; otherwise null.
    4) impact_factor_value: If the answer states a specific Journal Impact Factor numeric value (e.g., 17.2), extract it as a string; otherwise null.
    5) impact_factor_year: If the answer states a specific year for the Journal Impact Factor (e.g., 2024), extract that year as a string; otherwise null.
    6) impact_factor_urls: Extract all URLs that the answer uses specifically to support the Journal Impact Factor claim. Return every URL as a separate string in the array. If none, return an empty array.
    7) review_speed_median_days: If the answer provides a specific median time (in days) from submission to first editorial decision, extract it as a string (e.g., "8" or "8 days"); otherwise null.
    8) review_speed_urls: Extract all URLs that the answer uses specifically to support the median time to first editorial decision. If none, return an empty array.
    9) general_urls: Extract any other journal-related URLs mentioned (e.g., homepage, aims & scope, about, instructions for authors). If none, return an empty array.

    Rules:
    - Extract only what is explicitly present in the answer text. Do not infer or invent.
    - For URLs, return only valid and complete URLs. If a URL is missing the protocol, prepend http://.
    - If a required field is not present, set it to null (for a single value) or [] (for arrays).
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _merge_unique_urls(*url_lists: List[str]) -> List[str]:
    seen = set()
    merged: List[str] = []
    for lst in url_lists:
        for u in lst:
            if not u:
                continue
            if u not in seen:
                seen.add(u)
                merged.append(u)
    return merged


# --------------------------------------------------------------------------- #
# Verification builder                                                        #
# --------------------------------------------------------------------------- #
async def build_and_verify_tree(evaluator: Evaluator, info: JournalInfo) -> None:
    # Create the main critical node for the task
    main = evaluator.add_parallel(
        id="Journal_Identification",
        desc="Identify a single multidisciplinary open access journal meeting all criteria and provide the journal name plus reference URL(s) supporting both metrics.",
        parent=evaluator.root,
        critical=True,
    )

    # 1) Journal name must be provided
    evaluator.add_custom_node(
        result=bool(info.journal_name and info.journal_name.strip()),
        id="Journal_Name_Provided",
        desc="The response provides the journal's name.",
        parent=main,
        critical=True,
    )

    # 2) Journal type requirements (multidisciplinary + open access)
    # We split into two leaves under a critical parallel parent to ensure concrete checks are separate.
    jtype = evaluator.add_parallel(
        id="Journal_Type_Requirements",
        desc="The identified journal is multidisciplinary and open access.",
        parent=main,
        critical=True,
    )

    # Build a source pool for type checks: prefer general URLs, fallback to metric URLs
    type_sources = info.general_urls
    if not type_sources:
        type_sources = _merge_unique_urls(info.impact_factor_urls, info.review_speed_urls)

    # 2.a) Open access verification
    oa_node = evaluator.add_leaf(
        id="Journal_Open_Access",
        desc="The journal is open access.",
        parent=jtype,
        critical=True,
    )
    oa_claim = f"The journal '{info.journal_name}' is an open access journal."
    await evaluator.verify(
        claim=oa_claim,
        node=oa_node,
        sources=type_sources if type_sources else None,
        additional_instruction="Accept if the source indicates the journal is fully open access (OA). Publisher pages are preferred. If no source is provided, rely on the answer content."
    )

    # 2.b) Multidisciplinary verification
    multi_node = evaluator.add_leaf(
        id="Journal_Multidisciplinary",
        desc="The journal is multidisciplinary.",
        parent=jtype,
        critical=True,
    )
    multi_claim = f"The journal '{info.journal_name}' is multidisciplinary (publishes across multiple disciplines)."
    await evaluator.verify(
        claim=multi_claim,
        node=multi_node,
        sources=type_sources if type_sources else None,
        additional_instruction="Accept if the source (e.g., aims & scope) indicates broad, cross-disciplinary, or multidisciplinary coverage. If no source is provided, rely on the answer content."
    )

    # 3) Impact Factor requirement: 2024 JIF > 15.0
    if_req = evaluator.add_leaf(
        id="Impact_Factor_Requirement",
        desc="The identified journal has a 2024 Journal Impact Factor greater than 15.0.",
        parent=main,
        critical=True,
    )
    if info.impact_factor_value and info.impact_factor_value.strip():
        if_claim = f"The 2024 Journal Impact Factor (JIF) of {info.journal_name} is {info.impact_factor_value}, which is greater than 15.0."
    else:
        if_claim = f"The 2024 Journal Impact Factor (JIF) of {info.journal_name} is greater than 15.0."

    await evaluator.verify(
        claim=if_claim,
        node=if_req,
        sources=info.impact_factor_urls if info.impact_factor_urls else None,
        additional_instruction=(
            "Verify that the source explicitly states the 2024 Journal Impact Factor (JIF) for the journal. "
            "Prioritize official publisher pages or Clarivate JCR/Web of Science. "
            "Reject if the page refers to non-JIF metrics (e.g., CiteScore, SJR) or to a different year."
        ),
    )

    # 4) Review speed requirement: median time to first decision <= 10 days
    rs_req = evaluator.add_leaf(
        id="Review_Speed_Requirement",
        desc="The identified journal reports a median time from submission to first editorial decision of 10 days or less.",
        parent=main,
        critical=True,
    )
    if info.review_speed_median_days and info.review_speed_median_days.strip():
        rs_claim = (
            f"The journal {info.journal_name} reports a median time from submission to first editorial decision of "
            f"{info.review_speed_median_days}, which is 10 days or less."
        )
    else:
        rs_claim = (
            f"The journal {info.journal_name} reports a median time from submission to first editorial decision of 10 days or less."
        )
    await evaluator.verify(
        claim=rs_claim,
        node=rs_req,
        sources=info.review_speed_urls if info.review_speed_urls else None,
        additional_instruction=(
            "Confirm that the page explicitly reports the median time to first editorial decision (not average) and that it is 10 days or less. "
            "Synonyms include 'time to first decision' or 'median time to first decision'. "
            "Prefer official journal/publisher sources."
        ),
    )

    # 5) Evidence URLs (critical) for both metrics
    evid = evaluator.add_parallel(
        id="Evidence_URLs",
        desc="Provide reference URL(s) from official journal sources or recognized indexing databases that substantiate both the 2024 Journal Impact Factor and the median first-decision time.",
        parent=main,
        critical=True,
    )

    # 5.a) Impact Factor evidence group (sequential: presence -> support)
    if_evid_group = evaluator.add_sequential(
        id="Impact_Factor_Evidence_URL",
        desc="Impact Factor evidence URL(s) are provided and substantiate the claim.",
        parent=evid,
        critical=True,
    )

    # Presence
    evaluator.add_custom_node(
        result=len(info.impact_factor_urls) >= 1,
        id="Impact_Factor_Evidence_URL_Provided",
        desc="At least one URL is provided that supports the stated 2024 Journal Impact Factor.",
        parent=if_evid_group,
        critical=True,
    )

    # Support
    if_evid_support = evaluator.add_leaf(
        id="Impact_Factor_Evidence_URL_Support",
        desc="The provided URL(s) substantiate the 2024 JIF value/claim.",
        parent=if_evid_group,
        critical=True,
    )
    support_if_claim = (
        f"At least one of these pages explicitly supports the 2024 Journal Impact Factor (JIF) for {info.journal_name} "
        f"and its stated value (or confirms that it is greater than 15.0)."
    )
    await evaluator.verify(
        claim=support_if_claim,
        node=if_evid_support,
        sources=info.impact_factor_urls,
        additional_instruction=(
            "Accept only if the page explicitly mentions 'Journal Impact Factor' and '2024' for the specified journal. "
            "Prefer Clarivate JCR/Web of Science or official publisher pages. "
            "Reject pages that only show non-JIF metrics such as CiteScore/SJR."
        ),
    )

    # 5.b) Review speed evidence group (sequential: presence -> support)
    rs_evid_group = evaluator.add_sequential(
        id="Review_Speed_Evidence_URL",
        desc="Review speed evidence URL(s) are provided and substantiate the claim.",
        parent=evid,
        critical=True,
    )

    # Presence
    evaluator.add_custom_node(
        result=len(info.review_speed_urls) >= 1,
        id="Review_Speed_Evidence_URL_Provided",
        desc="At least one URL is provided that supports the stated median submission-to-first-decision time.",
        parent=rs_evid_group,
        critical=True,
    )

    # Support
    rs_evid_support = evaluator.add_leaf(
        id="Review_Speed_Evidence_URL_Support",
        desc="The provided URL(s) substantiate the median first-decision time value/claim.",
        parent=rs_evid_group,
        critical=True,
    )
    if info.review_speed_median_days and info.review_speed_median_days.strip():
        rs_support_claim = (
            f"At least one of these pages explicitly states the journal's median time from submission to first editorial decision "
            f"as {info.review_speed_median_days} (or equivalent phrasing)."
        )
    else:
        rs_support_claim = (
            f"At least one of these pages explicitly states the journal's median time from submission to first editorial decision "
            f"as 10 days or less."
        )
    await evaluator.verify(
        claim=rs_support_claim,
        node=rs_evid_support,
        sources=info.review_speed_urls,
        additional_instruction=(
            "Confirm that the source is an official journal/publisher page reporting the median time to first editorial decision "
            "(not average). Accept synonyms like 'time to first decision' when clearly referring to median."
        ),
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
    # Initialize evaluator with a parallel root
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

    # Extract structured information from the answer
    info: JournalInfo = await evaluator.extract(
        prompt=prompt_extract_journal_info(),
        template_class=JournalInfo,
        extraction_name="journal_info",
    )

    # Record the task constraints as "ground truth" context for transparency
    evaluator.add_ground_truth({
        "requirements": {
            "journal_type": ["multidisciplinary", "open access"],
            "impact_factor": {"year": "2024", "threshold": "> 15.0"},
            "review_speed": {"metric": "median time to first editorial decision", "threshold_days": "≤ 10"},
            "evidence": {
                "impact_factor": "official journal source or recognized indexing database (e.g., Clarivate JCR/Web of Science)",
                "review_speed": "official journal/publisher source"
            }
        }
    })

    # Build and run verification according to the rubric
    await build_and_verify_tree(evaluator, info)

    # Return evaluation summary
    return evaluator.get_summary()