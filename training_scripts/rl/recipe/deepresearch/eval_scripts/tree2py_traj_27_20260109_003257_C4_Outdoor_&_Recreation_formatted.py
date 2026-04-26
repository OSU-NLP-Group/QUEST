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
TASK_ID = "ca_state_parks_reservations"
TASK_DESCRIPTION = """
What are the complete booking procedures and requirements for making camping reservations at California State Parks? Your answer should include: how far in advance reservations can be made, what time new dates become available, how the reservation window operates, which platform or platforms can be used to make reservations (including any phone options), the nature of the reservations (whether they are site-specific or general), and information about the fee structure.
"""


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class ReservationPlatforms(BaseModel):
    """Reservation platforms and contact info mentioned in the answer."""
    online_platforms: List[str] = Field(default_factory=list)
    online_urls: List[str] = Field(default_factory=list)
    phone_numbers: List[str] = Field(default_factory=list)


class ReservationExtraction(BaseModel):
    """Structured extraction of the required reservation policies/details from the answer."""
    advance_window_text: Optional[str] = None
    opening_time_text: Optional[str] = None
    window_operation_text: Optional[str] = None
    site_specific_text: Optional[str] = None
    fee_structure_text: Optional[str] = None
    fee_details: List[str] = Field(default_factory=list)  # e.g., ["campsite fees", "reservation service fees"]
    platforms: Optional[ReservationPlatforms] = None
    sources: List[str] = Field(default_factory=list)  # any URLs cited in the answer


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_reservation_info() -> str:
    return """
    Extract the California State Parks camping reservation details as presented in the answer. Focus on presence and phrasing.
    Return a JSON matching this schema:

    - advance_window_text: The exact phrasing about how far in advance camping reservations can be made. If the answer states “6 months,” capture the sentence or phrase. If missing, return null.
    - opening_time_text: The exact phrasing for the time new reservation dates become available daily (e.g., “8:00 AM PST/PDT” or “8 AM PT”). If missing, return null.
    - window_operation_text: The phrasing describing how the reservation window operates (e.g., “rolling window,” “dates open on a rolling basis as they hit the 6‑month mark”). If missing, return null.
    - site_specific_text: The phrasing that indicates whether reservations are site-specific (e.g., “reserve a specific campsite”). If missing, return null.
    - fee_structure_text: The phrasing mentioning the fee structure (e.g., “campsite fees and reservation service fees”). If missing, return null.
    - fee_details: An array of fee components explicitly mentioned (e.g., ["campsite fees", "reservation service fees", "transaction fee"]). If none mentioned, return an empty array.
    - platforms:
        - online_platforms: Names of online platforms mentioned (e.g., ["ReserveCalifornia", "ReserveCalifornia.com"]).
        - online_urls: URLs for any online platforms or official pages cited (e.g., "https://www.reservecalifornia.com/").
        - phone_numbers: Any phone numbers related to reservations that appear in the answer (e.g., "1-800-444-7275", "800-444-PARK").
    - sources: All URLs cited anywhere in the answer (include duplicates from online_urls if easier). If none, return an empty array.

    Rules:
    - Extract exactly what the answer explicitly states. Do not add or infer unseen data.
    - For phone numbers, keep as-is (formatting differences are okay).
    - For URLs, extract the actual links (plain or markdown). If missing protocol, prepend http://.
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def collect_all_urls(extracted: ReservationExtraction) -> List[str]:
    """Collect all URLs found in the answer to be used as potential evidence sources."""
    urls: set[str] = set()
    if extracted.sources:
        for u in extracted.sources:
            if isinstance(u, str) and u.strip():
                urls.add(u.strip())
    if extracted.platforms and extracted.platforms.online_urls:
        for u in extracted.platforms.online_urls:
            if isinstance(u, str) and u.strip():
                urls.add(u.strip())
    return list(urls)


# --------------------------------------------------------------------------- #
# Verification logic                                                          #
# --------------------------------------------------------------------------- #
async def build_and_verify_requirements(
    evaluator: Evaluator,
    parent_node,
    extracted: ReservationExtraction,
) -> None:
    """
    Build the critical parallel node and attach 7 critical leaf verifications.
    Where URLs are available, verify facts against the webpages;
    otherwise, treat verification as a presence check against the answer text.
    """
    # Critical aggregator node (to match rubric: parent is critical and parallel)
    main_node = evaluator.add_parallel(
        id="California_State_Parks_Camping_Reservation_Requirements",
        desc="Evaluate whether the answer includes all required procedures/requirements for making camping reservations at California State Parks, as specified by the question and constraints.",
        parent=parent_node,
        critical=True,
    )

    urls = collect_all_urls(extracted)

    # Leaf nodes
    advance_leaf = evaluator.add_leaf(
        id="Advance_Booking_Window",
        desc="States that camping reservations can be made up to 6 months in advance of the arrival date.",
        parent=main_node,
        critical=True,
    )
    opening_time_leaf = evaluator.add_leaf(
        id="Daily_Opening_Time",
        desc="States that new reservation dates become available at 8:00 AM PST/PDT each day.",
        parent=main_node,
        critical=True,
    )
    window_op_leaf = evaluator.add_leaf(
        id="Reservation_Window_Operation",
        desc="Explains that the reservation availability operates on a rolling window (i.e., dates open on a rolling basis as they reach the 6-month-in-advance mark).",
        parent=main_node,
        critical=True,
    )
    platform_online_leaf = evaluator.add_leaf(
        id="Reservation_Platform_Online",
        desc="Identifies ReserveCalifornia.com as a platform that can be used to make reservations online.",
        parent=main_node,
        critical=True,
    )
    platform_phone_leaf = evaluator.add_leaf(
        id="Reservation_Platform_Phone",
        desc="Includes the phone reservation option and provides the phone number 1-800-444-7275.",
        parent=main_node,
        critical=True,
    )
    site_specific_leaf = evaluator.add_leaf(
        id="Reservation_Nature_Site_Specific",
        desc="States that campground reservations are site-specific (reserve a specific campsite rather than general availability).",
        parent=main_node,
        critical=True,
    )
    fee_structure_leaf = evaluator.add_leaf(
        id="Fee_Structure",
        desc="Mentions that the fee structure includes campsite fees and additional reservation service fees.",
        parent=main_node,
        critical=True,
    )

    # Prepare claims and instructions
    claims_and_sources: List[tuple[str, List[str] | str | None, Any, Optional[str]]] = []

    # 1) Advance booking window (6 months)
    claim_advance = "Camping reservations for California State Parks can be made up to 6 months in advance of the arrival date."
    add_ins_advance = (
        "If URL sources are provided, verify this policy is supported by the cited official pages (e.g., ReserveCalifornia). "
        "If no URLs are provided, treat this as a presence check: confirm the answer explicitly states this 6-month window. "
        "Allow equivalent phrasing like 'six months' or 'half a year'. Do not rely on outside knowledge when URLs are absent."
    )
    claims_and_sources.append((claim_advance, urls if urls else None, advance_leaf, add_ins_advance))

    # 2) Daily opening time (8:00 AM PST/PDT)
    claim_opening = "New reservation dates become available at 8:00 AM Pacific Time (PST/PDT) each day."
    add_ins_opening = (
        "If URLs are provided, verify the time is supported by the official page. "
        "Without URLs, check the answer text contains this time. Accept reasonable variants, e.g., '8 AM PT', '08:00 a.m. Pacific Time', '8:00 AM PDT/PST'."
    )
    claims_and_sources.append((claim_opening, urls if urls else None, opening_time_leaf, add_ins_opening))

    # 3) Rolling window operation
    claim_window = "Reservation availability operates on a rolling window: dates open on a rolling basis as they reach the 6-month-in-advance mark."
    add_ins_window = (
        "If URLs are provided, verify the rolling-window operation is stated. "
        "Without URLs, ensure the answer describes a rolling opening of dates as they hit the 6-month mark. Accept paraphrases such as 'released daily as they reach 6 months'."
    )
    claims_and_sources.append((claim_window, urls if urls else None, window_op_leaf, add_ins_window))

    # 4) Platform online: ReserveCalifornia.com
    claim_platform_online = "ReserveCalifornia.com is a platform that can be used to make camping reservations online for California State Parks."
    add_ins_platform_online = (
        "If URLs are provided, confirm the site is the official online reservation platform. "
        "Without URLs, verify the answer explicitly names 'ReserveCalifornia' or 'ReserveCalifornia.com' as the platform. "
        "Allow minor variants like 'Reserve California' or 'the ReserveCalifornia website'."
    )
    claims_and_sources.append((claim_platform_online, urls if urls else None, platform_online_leaf, add_ins_platform_online))

    # 5) Phone option and specific phone number
    claim_platform_phone = "There is a phone reservation option and the phone number is 1-800-444-7275."
    add_ins_platform_phone = (
        "If URLs are provided, verify the phone number on the official contact or reservation page. "
        "Without URLs, confirm the answer includes the phone option and the number '1-800-444-7275'. "
        "Accept equivalent formatting (e.g., '1 (800) 444-7275') and variants like '800-444-PARK (7275)'."
    )
    claims_and_sources.append((claim_platform_phone, urls if urls else None, platform_phone_leaf, add_ins_platform_phone))

    # 6) Site-specific nature
    claim_site_specific = "Campground reservations are site-specific; you reserve a specific campsite rather than general availability."
    add_ins_site_specific = (
        "If URLs are provided, verify the site-specific nature on the official page. "
        "Without URLs, check the answer explicitly indicates site-specific reservations (e.g., selecting a specific campsite). "
        "Allow paraphrases that clearly mean the same thing."
    )
    claims_and_sources.append((claim_site_specific, urls if urls else None, site_specific_leaf, add_ins_site_specific))

    # 7) Fee structure includes campsite fees and reservation service fees
    claim_fees = "The fee structure includes campsite fees and additional reservation service fees."
    add_ins_fees = (
        "If URLs are provided, verify that both campsite fees and reservation service/transaction fees are mentioned. "
        "Without URLs, confirm the answer mentions both components (accept synonyms such as 'service fee', 'transaction fee', 'processing fee')."
    )
    claims_and_sources.append((claim_fees, urls if urls else None, fee_structure_leaf, add_ins_fees))

    # Execute verifications (in parallel)
    await evaluator.batch_verify(claims_and_sources)


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
    Evaluate an answer for California State Parks camping reservations requirements.
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

    # Extract structured information from the answer
    extracted_info = await evaluator.extract(
        prompt=prompt_extract_reservation_info(),
        template_class=ReservationExtraction,
        extraction_name="reservation_requirements",
    )

    # Add ground truth expectations (for context in summary)
    evaluator.add_ground_truth({
        "expected_criteria": {
            "advance_booking_window": "Up to 6 months in advance of arrival date",
            "daily_opening_time": "8:00 AM PST/PDT",
            "reservation_window_operation": "Rolling window; dates open as they reach the 6-month mark",
            "platform_online": "ReserveCalifornia.com",
            "platform_phone": "1-800-444-7275",
            "reservation_nature": "Site-specific (reserve a specific campsite)",
            "fee_structure": "Includes campsite fees + reservation service/transaction fees"
        }
    }, gt_type="ground_truth")

    # Build verification tree and run checks
    await build_and_verify_requirements(evaluator, root, extracted_info)

    # Return summary
    return evaluator.get_summary()