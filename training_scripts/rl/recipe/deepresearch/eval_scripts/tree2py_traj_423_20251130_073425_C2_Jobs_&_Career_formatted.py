import asyncio
import logging
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy, VerificationNode
from obj_task_eval.utils.cache_filesys import CacheFileSys


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "ca_lpcc_supervised_experience"
TASK_DESCRIPTION = (
    "You are planning to pursue Licensed Professional Clinical Counselor (LPCC) licensure in California after completing your master's degree in counseling. "
    "What are the supervised experience requirements you must fulfill? Specifically, provide: "
    "(1) the total number of supervised hours required, "
    "(2) the minimum number of hours that must be direct clinical counseling with clients, "
    "(3) the maximum number of hours that can be non-clinical practice activities, and "
    "(4) the minimum time period (in weeks) over which these hours must be accumulated. "
    "Include a reference URL from the California Board of Behavioral Sciences that documents these requirements."
)

GROUND_TRUTH = {
    "total_hours": "3,000",
    "min_direct_hours": "1,750",
    "max_nonclinical_hours": "1,250",
    "min_weeks": "104",
}


# --------------------------------------------------------------------------- #
# Extraction models                                                           #
# --------------------------------------------------------------------------- #
class RequirementsExtraction(BaseModel):
    """
    Structured extraction from the answer for California LPCC supervised experience requirements.
    All fields are strings to maximize robustness (answers may include formatting or commas).
    """
    total_hours: Optional[str] = None
    min_direct_hours: Optional[str] = None
    max_nonclinical_hours: Optional[str] = None
    min_weeks: Optional[str] = None
    bbs_urls: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_requirements() -> str:
    return """
    Extract the supervised experience requirements for California LPCC licensure as explicitly stated in the answer text.

    Return a JSON object containing these fields:
    - total_hours: The total number of supervised post-degree experience hours required (as it appears in the answer; keep formatting like "3,000" if present).
    - min_direct_hours: The minimum number of hours that must be direct clinical counseling with clients (string).
    - max_nonclinical_hours: The maximum number of hours that can be non-clinical practice activities (string).
    - min_weeks: The minimum accumulation period in weeks over which the hours must be accrued (string).
    - bbs_urls: An array of all URLs explicitly cited in the answer that are from the California Board of Behavioral Sciences (domain contains "bbs.ca.gov"). Include only valid URLs that appear in the answer text (plain or markdown links). If none are present, return an empty array.

    Rules:
    - Extract only what the answer explicitly states. Do not infer or invent values.
    - If a specific item is not stated, return null for that field (or empty list for bbs_urls).
    - For URLs: include only those with the domain "bbs.ca.gov" (case-insensitive). If the answer references the BBS without a URL, do not add anything.
    """


# --------------------------------------------------------------------------- #
# Verification helpers                                                        #
# --------------------------------------------------------------------------- #
def has_bbs_url(extracted: RequirementsExtraction) -> bool:
    return any(isinstance(u, str) and ("bbs.ca.gov" in u.lower()) for u in extracted.bbs_urls)


# --------------------------------------------------------------------------- #
# Build verification tree                                                     #
# --------------------------------------------------------------------------- #
async def build_and_verify_requirements_tree(
    evaluator: Evaluator,
    root: VerificationNode,
    extracted: RequirementsExtraction,
) -> None:
    """
    Build the verification tree according to the rubric and execute verifications.
    """

    # Critical main node that aggregates all required checks in parallel
    main_node = evaluator.add_parallel(
        id="California_LPCC_Supervised_Experience_Requirements",
        desc="Evaluates whether the answer provides all supervised-experience requirements requested for California LPCC licensure and includes a supporting California BBS reference URL.",
        parent=root,
        critical=True,
    )

    # 1) Supporting URL Reference (sequential: existence -> documentation check)
    url_node = evaluator.add_sequential(
        id="Supporting_URL_Reference",
        desc="Provides at least one valid reference URL from the California Board of Behavioral Sciences (bbs.ca.gov) documenting the LPCC supervised experience requirements.",
        parent=main_node,
        critical=True,
    )

    # 1.a Existence/domain check (custom, critical)
    evaluator.add_custom_node(
        result=has_bbs_url(extracted),
        id="Supporting_URL_Reference_exists",
        desc="At least one valid California BBS URL (domain contains bbs.ca.gov) is provided in the answer.",
        parent=url_node,
        critical=True,
    )

    # 1.b Verify that at least one provided BBS URL documents LPCC supervised experience requirements
    url_docs_node = evaluator.add_leaf(
        id="Supporting_URL_Reference_valid",
        desc="A provided BBS URL documents the LPCC supervised experience requirements.",
        parent=url_node,
        critical=True,
    )
    await evaluator.verify(
        claim=(
            "This California Board of Behavioral Sciences webpage documents the LPCC supervised experience requirements "
            "(e.g., total hours, minimum direct clinical hours, maximum non-clinical hours, and minimum accumulation period)."
        ),
        node=url_docs_node,
        sources=extracted.bbs_urls,
        additional_instruction=(
            "Confirm the page is about LPCC supervised experience requirements. "
            "Pages for other professions (e.g., LCSW, LMFT) are not acceptable unless they clearly document LPCC requirements."
        ),
    )

    # 2) Total Supervised Hours (sequential: answer states 3,000 -> source supports)
    total_node = evaluator.add_sequential(
        id="Total_Supervised_Hours",
        desc="States the total number of supervised post-degree experience hours required (3,000).",
        parent=main_node,
        critical=True,
    )

    total_value_leaf = evaluator.add_leaf(
        id="Total_Supervised_Hours_value",
        desc="The answer states the total supervised experience hours is 3,000.",
        parent=total_node,
        critical=True,
    )
    await evaluator.verify(
        claim="The answer states that the total required supervised experience hours is 3,000.",
        node=total_value_leaf,
        additional_instruction=(
            "Check the answer text directly. Allow minor formatting variants like '3000' or '3,000'."
        ),
    )

    total_supported_leaf = evaluator.add_leaf(
        id="Total_Supervised_Hours_supported",
        desc="California BBS source supports that the total supervised experience hours required is 3,000.",
        parent=total_node,
        critical=True,
    )
    await evaluator.verify(
        claim="California LPCC requires 3,000 total supervised post-degree experience hours.",
        node=total_supported_leaf,
        sources=extracted.bbs_urls,
        additional_instruction=(
            "Verify on the BBS site that the requirement is 3,000 hours for LPCC supervised experience."
        ),
    )

    # 3) Minimum Direct Clinical Hours (1,750) (sequential: answer states -> source supports)
    direct_node = evaluator.add_sequential(
        id="Minimum_Direct_Clinical_Hours",
        desc="States the minimum number of hours that must be direct clinical counseling with clients (1,750).",
        parent=main_node,
        critical=True,
    )

    direct_value_leaf = evaluator.add_leaf(
        id="Minimum_Direct_Clinical_Hours_value",
        desc="The answer states that at least 1,750 hours must be direct clinical counseling with clients.",
        parent=direct_node,
        critical=True,
    )
    await evaluator.verify(
        claim="The answer states that at least 1,750 of the supervised hours must be direct clinical counseling with clients.",
        node=direct_value_leaf,
        additional_instruction="Allow formatting variants like '1750' or '1,750'.",
    )

    direct_supported_leaf = evaluator.add_leaf(
        id="Minimum_Direct_Clinical_Hours_supported",
        desc="California BBS source supports that a minimum of 1,750 hours must be direct clinical counseling.",
        parent=direct_node,
        critical=True,
    )
    await evaluator.verify(
        claim="At least 1,750 hours must be direct clinical counseling with clients for LPCC supervised experience.",
        node=direct_supported_leaf,
        sources=extracted.bbs_urls,
        additional_instruction="Confirm on the BBS page that the LPCC requirement includes a minimum of 1,750 direct client-counseling hours.",
    )

    # 4) Maximum Non-Clinical Hours (1,250) (sequential: answer states -> source supports)
    nonclinical_node = evaluator.add_sequential(
        id="Maximum_Non_Clinical_Hours",
        desc="States the maximum number of hours that can be non-clinical practice activities (1,250).",
        parent=main_node,
        critical=True,
    )

    nonclinical_value_leaf = evaluator.add_leaf(
        id="Maximum_Non_Clinical_Hours_value",
        desc="The answer states that no more than 1,250 hours may be non-clinical practice activities.",
        parent=nonclinical_node,
        critical=True,
    )
    await evaluator.verify(
        claim="The answer states that no more than 1,250 hours may be non-clinical practice activities.",
        node=nonclinical_value_leaf,
        additional_instruction="Allow formatting variants like '1250' or '1,250'.",
    )

    nonclinical_supported_leaf = evaluator.add_leaf(
        id="Maximum_Non_Clinical_Hours_supported",
        desc="California BBS source supports that the non-clinical hours are capped at 1,250.",
        parent=nonclinical_node,
        critical=True,
    )
    await evaluator.verify(
        claim="No more than 1,250 hours may be non-clinical practice activities for LPCC supervised experience.",
        node=nonclinical_supported_leaf,
        sources=extracted.bbs_urls,
        additional_instruction="Confirm on the BBS page that the LPCC requirement caps non-clinical practice hours at 1,250.",
    )

    # 5) Minimum Time Period (104 weeks) (sequential: answer states -> source supports)
    weeks_node = evaluator.add_sequential(
        id="Minimum_Time_Period",
        desc="States the minimum accumulation period in weeks over which the hours must be accrued (104 weeks).",
        parent=main_node,
        critical=True,
    )

    weeks_value_leaf = evaluator.add_leaf(
        id="Minimum_Time_Period_value",
        desc="The answer states that the supervised experience must be accumulated over at least 104 weeks.",
        parent=weeks_node,
        critical=True,
    )
    await evaluator.verify(
        claim="The answer states that the supervised experience must be accumulated over at least 104 weeks.",
        node=weeks_value_leaf,
        additional_instruction="Allow formatting variants like '104 weeks' or 'at least 104 weeks'.",
    )

    weeks_supported_leaf = evaluator.add_leaf(
        id="Minimum_Time_Period_supported",
        desc="California BBS source supports that supervised experience must be accrued over a minimum of 104 weeks.",
        parent=weeks_node,
        critical=True,
    )
    await evaluator.verify(
        claim="Supervised experience for LPCC must be accrued over a minimum of 104 weeks.",
        node=weeks_supported_leaf,
        sources=extracted.bbs_urls,
        additional_instruction="Confirm on the BBS page that the LPCC requirement includes a minimum accumulation period of 104 weeks.",
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
    Evaluate an answer for California LPCC supervised experience requirements.
    """
    # Initialize evaluator (root is non-critical; we add a critical main node under it)
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

    # Extract structured info from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_requirements(),
        template_class=RequirementsExtraction,
        extraction_name="lpcc_requirements_extraction",
    )

    # Add ground truth for transparency
    evaluator.add_ground_truth(
        {
            "jurisdiction": "California",
            "license": "LPCC",
            "expected_requirements": GROUND_TRUTH,
        },
        gt_type="ground_truth_requirements",
    )

    # Build and run verification tree
    await build_and_verify_requirements_tree(evaluator, root, extracted)

    # Return evaluator summary
    return evaluator.get_summary()