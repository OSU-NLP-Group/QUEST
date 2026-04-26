import asyncio
import logging
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys

# -----------------------------------------------------------------------------
# Task constants
# -----------------------------------------------------------------------------
TASK_ID = "boston_2026_men_18_34_qual_time"
TASK_DESCRIPTION = "What is the official qualifying time standard for a male runner aged 18-34 to be eligible to apply for the 2026 Boston Marathon? Provide the time in hours and minutes format, and include a link to the official Boston Athletic Association (BAA) source."

EXPECTED_TIME = "2:55:00"
OFFICIAL_BAA_DOMAINS = ["baa.org", "bostonathleticassociation.org"]


# -----------------------------------------------------------------------------
# Extraction Models
# -----------------------------------------------------------------------------
class QualifyingTimeExtraction(BaseModel):
    """
    Information we need from the agent's answer:
    - The stated qualifying time for Men 18–34 for the 2026 Boston Marathon
    - Any official BAA URL(s) included in the answer
    """
    male_18_34_time: Optional[str] = None
    official_baa_urls: List[str] = Field(default_factory=list)


# -----------------------------------------------------------------------------
# Extraction Prompt Builders
# -----------------------------------------------------------------------------
def prompt_extract_qualifying_info() -> str:
    return """
    Extract the following fields from the answer text:

    1) male_18_34_time:
       - The qualifying time standard stated for the Men (male) 18–34 category for the 2026 Boston Marathon.
       - Return the time string exactly as written in the answer (e.g., "2:55:00", "2:55", "2 hours 55 minutes", etc.).
       - If not provided, return null.

    2) official_baa_urls:
       - All URLs in the answer that point to the official Boston Athletic Association (BAA) website.
       - Only include URLs whose domain clearly belongs to BAA, such as:
         • baa.org (e.g., https://www.baa.org/...)
         • bostonathleticassociation.org (if present)
       - Do NOT include non-BAA sources (e.g., news articles, blogs, social media).
       - If the answer includes no official BAA URLs, return an empty list.

    Note:
    - Extract only what is explicitly present in the answer.
    - Keep URLs complete and with protocol (http/https).
    """


# -----------------------------------------------------------------------------
# Main evaluation
# -----------------------------------------------------------------------------
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
    Evaluate an answer for the 2026 Boston Marathon (Men 18–34) qualifying time standard task.
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

    # Record ground truth info for transparency
    evaluator.add_ground_truth(
        {
            "expected_time": EXPECTED_TIME,
            "required_category": "Men 18–34",
            "event_year": 2026,
            "official_domains": OFFICIAL_BAA_DOMAINS,
        },
        gt_type="ground_truth",
    )

    # Extract structured info from the answer
    extraction: QualifyingTimeExtraction = await evaluator.extract(
        prompt=prompt_extract_qualifying_info(),
        template_class=QualifyingTimeExtraction,
        extraction_name="extracted_qualifying_time_info",
    )

    # Build the rubric tree per JSON
    # Parent node (critical, parallel): Qualifying_Time_Standard
    qt_node = evaluator.add_parallel(
        id="Qualifying_Time_Standard",
        desc="The correct 2026 Boston Marathon qualifying time standard for the specified age and gender category is identified",
        parent=root,
        critical=True,
    )

    # Child leaf 1 (critical): Correct_Time_Value
    correct_time_leaf = evaluator.add_leaf(
        id="Correct_Time_Value",
        desc=f"The qualifying time provided matches the official 2026 Boston Marathon standard for men aged 18-34, which is {EXPECTED_TIME} (2 hours 55 minutes)",
        parent=qt_node,
        critical=True,
    )

    # If time missing, fail immediately; otherwise verify equivalence
    extracted_time = extraction.male_18_34_time or ""
    if extracted_time.strip():
        time_claim = (
            f"The time '{extracted_time.strip()}' is equivalent to {EXPECTED_TIME} "
            f"(that is, two hours and fifty-five minutes; allowing optional ':00' seconds or formats such as '2:55' or '2 hours 55 minutes')."
        )
        await evaluator.verify(
            claim=time_claim,
            node=correct_time_leaf,
            additional_instruction=(
                "Judge only whether the two expressions denote the same duration. "
                "Accept minor, reasonable formatting variants such as '2:55' vs '2:55:00' or '2 hours 55 minutes'. "
                "Do not conflate '2.55 hours' with '2:55'."
            ),
        )
    else:
        # No time provided in the answer -> fail this critical leaf
        correct_time_leaf.score = 0.0
        correct_time_leaf.status = "failed"

    # Child leaf 2 (critical): Official_Source_Referenced
    official_src_leaf = evaluator.add_leaf(
        id="Official_Source_Referenced",
        desc="The answer includes a reference to the official Boston Athletic Association (BAA) qualifying standards page or official BAA source",
        parent=qt_node,
        critical=True,
    )

    # If no official BAA URLs were extracted, fail immediately. Otherwise, verify the URL(s) are indeed official BAA pages.
    baa_urls = extraction.official_baa_urls or []
    if len(baa_urls) == 0:
        official_src_leaf.score = 0.0
        official_src_leaf.status = "failed"
    else:
        # Verify that at least one provided URL is an official BAA page.
        # We focus ONLY on officialness for this rubric leaf (domain/branding),
        # not on whether the page states the time (that is covered by the time-value leaf).
        claim_official = (
            "This webpage is an official Boston Athletic Association (BAA) page (hosted on baa.org or a recognized BAA-owned domain) "
            "related to Boston Marathon qualifying standards or official race information."
        )
        await evaluator.verify(
            claim=claim_official,
            node=official_src_leaf,
            sources=baa_urls,
            additional_instruction=(
                "Pass if the page is clearly part of the official BAA website (e.g., domain contains 'baa.org' or 'bostonathleticassociation.org') "
                "and appears to be an official BAA page (branding, headers, footer, etc.). "
                "Do not accept third-party sites (news, blogs, social media) as official."
            ),
        )

    return evaluator.get_summary()