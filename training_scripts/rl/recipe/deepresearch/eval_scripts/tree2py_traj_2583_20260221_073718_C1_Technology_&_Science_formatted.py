import asyncio
import logging
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys

# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "lenovo_premier_support_contact"
TASK_DESCRIPTION = "What is the dedicated phone number for Lenovo Premier Support, and what are their regular operating hours for technical assistance?"

OFFICIAL_PREMIER_SUPPORT_URL = "https://pcsupport.lenovo.com/us/en/premiersupport"

# Ground truth reference (for recording in summary; actual evaluation is evidence-based)
GROUND_TRUTH = {
    "phone_number": "855-669-3600",
    "weekday_hours": "7:00 AM to 20:00 (8:00 PM), Monday through Friday",
    "after_hours": "After-hours support available 7 days a week in English",
    "official_url": OFFICIAL_PREMIER_SUPPORT_URL,
}


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class SupportInfo(BaseModel):
    phone_number: Optional[str] = None
    weekday_hours: Optional[str] = None
    after_hours_support: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_support_info() -> str:
    return """
    Extract the Lenovo Premier Support contact information exactly as provided in the answer.
    Return a JSON object with the following fields:
    - phone_number: The dedicated Lenovo Premier Support phone number for technical assistance as written in the answer (string). If not present, return null.
    - weekday_hours: The regular operating hours for technical assistance on weekdays (Monday–Friday) as written in the answer (string). If not present, return null.
    - after_hours_support: Any statement in the answer about after-hours support availability (string). If not present, return null.
    - sources: All URLs explicitly cited in the answer related to Lenovo Premier Support (array of strings). Include official pages and any other URLs if present. If none, return an empty array.

    Notes:
    - Do not invent or infer any information; extract only what is explicitly present in the answer text.
    - Accept various formatting styles (e.g., 7 AM–8 PM, 7:00 a.m. to 8:00 p.m., 07:00–20:00, Mon–Fri, M–F).
    - For URLs, capture the full URL (including protocol).
    """


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
    model: str = "o4-mini",
) -> Dict:
    """
    Evaluate the answer for Lenovo Premier Support contact information:
    - Dedicated phone number
    - Regular operating hours (Mon–Fri)
    - After-hours support availability
    - Verifiable on official Premier Support page
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
        default_model=model,
    )

    # 2) Extract structured info from the answer
    extracted_support = await evaluator.extract(
        prompt=prompt_extract_support_info(),
        template_class=SupportInfo,
        extraction_name="lenovo_premier_support_extraction",
    )

    # 3) Record ground truth info for transparency
    evaluator.add_ground_truth(
        {
            "expected": GROUND_TRUTH,
            "note": "Ground truth recorded for transparency. Actual verification is performed against the official Lenovo page."
        },
        gt_type="expected_values",
    )

    # 4) Build verification tree nodes according to rubric
    # Root-level critical node matching rubric
    premier_root = evaluator.add_parallel(
        id="Lenovo_Premier_Support_Contact_Information",
        desc=(
            "Verify that the answer provides Lenovo Premier Support's dedicated phone number and operating hours for "
            "technical assistance, consistent with the given constraints and verifiable via the specified official Lenovo page."
        ),
        parent=root,
        critical=True,
    )

    # Leaf: Correct phone number provided in the answer
    phone_leaf = evaluator.add_leaf(
        id="Correct_Phone_Number",
        desc="Answer provides the dedicated Lenovo Premier Support phone number for technical assistance: 855-669-3600.",
        parent=premier_root,
        critical=True,
    )

    # Leaf: Regular operating hours Mon–Fri provided in the answer
    hours_leaf = evaluator.add_leaf(
        id="Regular_Operating_Hours_Mon_Fri",
        desc="Answer provides the regular operating hours for technical assistance: 7:00 AM to 20:00 (8:00 PM), Monday through Friday.",
        parent=premier_root,
        critical=True,
    )

    # Leaf: After-hours support availability provided in the answer
    after_hours_leaf = evaluator.add_leaf(
        id="After_Hours_Support_Availability",
        desc="Answer states that after-hours support is available 7 days a week in English.",
        parent=premier_root,
        critical=True,
    )

    # Leaf: Phone number and hours verifiable on official Lenovo Premier Support page
    official_verify_leaf = evaluator.add_leaf(
        id="Verifiable_On_Official_Premier_Support_Page",
        desc=f"The provided phone number and hours are verifiable on the official Lenovo Premier Support page at {OFFICIAL_PREMIER_SUPPORT_URL}.",
        parent=premier_root,
        critical=True,
    )

    # 5) Prepare claims and batch verify to avoid prerequisite gating across critical siblings
    claims_and_sources = [
        (
            # Phone number presence in the answer
            "The answer lists Lenovo Premier Support's dedicated technical assistance phone number as 855-669-3600.",
            None,
            phone_leaf,
            "Allow minor formatting differences (e.g., '(855) 669-3600', '855 669 3600', '855-669-3600'). Verify this is explicitly present in the answer text.",
        ),
        (
            # Weekday operating hours presence in the answer
            "The answer states the regular operating hours for Lenovo Premier Support technical assistance are 7:00 AM to 20:00 (8:00 PM), Monday through Friday.",
            None,
            hours_leaf,
            "Accept reasonable formatting variants such as '7 AM–8 PM', '07:00–20:00', 'Mon–Fri', 'M–F'. Verify this is explicitly present in the answer text.",
        ),
        (
            # After-hours support statement presence in the answer
            "The answer states that after-hours support is available 7 days a week in English.",
            None,
            after_hours_leaf,
            "Accept phrases like 'after-hours support 7 days a week', '24/7 in English', or 'English support available 7 days a week'. Verify this is explicitly present in the answer text.",
        ),
        (
            # Official page evidence check (URL verification)
            (
                "The Lenovo Premier Support technical assistance phone number is 855-669-3600 and regular operating hours are "
                "7:00 AM to 20:00 (8:00 PM), Monday through Friday. After-hours support is available 7 days a week in English."
            ),
            OFFICIAL_PREMIER_SUPPORT_URL,
            official_verify_leaf,
            (
                "Verify the claim strictly against the provided official Lenovo Premier Support page. "
                "Allow minor formatting variants for phone and time ranges (e.g., 7 AM–8 PM, 07:00–20:00, Mon–Fri). "
                "Focus on the 'Premier Support' context (technical assistance) and do not confuse with general or different Lenovo support programs."
            ),
        ),
    ]

    await evaluator.batch_verify(claims_and_sources)

    # 6) Optionally record custom info (what was extracted from the answer)
    evaluator.add_custom_info(
        {
            "extracted_phone_number": extracted_support.phone_number,
            "extracted_weekday_hours": extracted_support.weekday_hours,
            "extracted_after_hours_support": extracted_support.after_hours_support,
            "extracted_sources": extracted_support.sources,
        },
        info_type="extracted_summary",
        info_name="extracted_support_info",
    )

    # 7) Return evaluation summary
    return evaluator.get_summary()