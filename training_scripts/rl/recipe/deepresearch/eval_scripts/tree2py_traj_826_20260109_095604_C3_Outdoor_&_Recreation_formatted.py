import asyncio
import logging
from typing import List, Dict, Any, Optional

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "half_dome_permit_2026_day_hike"
TASK_DESCRIPTION = (
    "A group of 4 friends wants to day hike Half Dome in Yosemite National Park on September 10, 2026. "
    "They are currently in the planning phase and it is now March 15, 2026. One person will serve as the "
    "permit holder and another person will serve as the alternate permit holder. Provide a complete explanation "
    "of the permit application process they should follow, including: which lottery system applies to their "
    "situation, the specific application window dates and timezone, the maximum group size allowed and whether "
    "alternates are permitted for this lottery type, key application requirements they must follow, and the "
    "complete fee structure with specific amounts. Include URL references from official sources to support each "
    "major component of your answer."
)


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class SupportURLsExtraction(BaseModel):
    """
    Extract official-source URLs the answer cites, categorized by major components.
    Only URLs explicitly present in the answer should be included.
    Prefer official sources such as nps.gov and recreation.gov.
    """
    lottery_urls: List[str] = Field(default_factory=list)
    timing_urls: List[str] = Field(default_factory=list)
    group_alt_urls: List[str] = Field(default_factory=list)
    requirements_urls: List[str] = Field(default_factory=list)
    season_quota_validity_urls: List[str] = Field(default_factory=list)
    fee_urls: List[str] = Field(default_factory=list)
    all_urls: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_support_urls() -> str:
    return (
        "From the answer, extract official-source URLs (only those actually present in the answer text) and "
        "categorize them under the following fields:\n"
        "- lottery_urls: URLs that support which lottery type applies to this scenario.\n"
        "- timing_urls: URLs that support the application window (including timezone) and/or the date when results are announced.\n"
        "- group_alt_urls: URLs that support max group size, whether alternates are permitted for the preseason lottery, "
        "whether alternates are not permitted for the daily lottery, and the alternate acceptance deadline (72 hours).\n"
        "- requirements_urls: URLs that support key application requirements such as one application per individual per lottery, "
        "only one appearance per person per lottery (as holder or alternate), that applicants must use their legal name, "
        "and that the holder or alternate must show government-issued ID that matches the permit at the checkpoint.\n"
        "- season_quota_validity_urls: URLs that support when permits are needed (while cables are up), the typical cables season window, "
        "the daily day-hiker quota, and the validity window for preseason permits (single day, 12:00 AM–11:59 PM).\n"
        "- fee_urls: URLs that support the fee amounts (application fee and per-person recreation fee) and when they are charged.\n"
        "- all_urls: All URLs present in the answer text (include both official and non-official here).\n\n"
        "Rules:\n"
        "1) Extract only URLs explicitly present in the answer (plain URLs or markdown links). Do not invent URLs.\n"
        "2) When categorizing, prefer official sources (nps.gov, recreation.gov). If the answer includes non-official URLs, "
        "they can appear in all_urls but try to avoid including them in the category-specific fields unless the answer "
        "provides no official URL for that category.\n"
        "3) If the answer provides no URL for a category, return an empty list for that category.\n"
    )


# --------------------------------------------------------------------------- #
# Helper for URL-based verification instructions                              #
# --------------------------------------------------------------------------- #
def official_source_instruction(category_hint: str) -> str:
    return (
        f"Only accept as supported if the URL is an official government source (prefer nps.gov or recreation.gov). "
        f"Ignore blogs, commercial, or unofficial pages. If multiple URLs are provided, it's sufficient that at least one "
        f"clearly supports the claim(s) for the category: {category_hint}. "
        f"Allow minor wording variations and common synonyms."
    )


# --------------------------------------------------------------------------- #
# Verification subtrees                                                       #
# --------------------------------------------------------------------------- #
async def verify_lottery_type(evaluator: Evaluator, parent_node, urls: List[str]) -> None:
    # Parent container (critical)
    group_node = evaluator.add_parallel(
        id="Lottery_Type_Identification",
        desc="Correctly identifies which lottery system applies to their situation (applying in March for a September day hike).",
        parent=parent_node,
        critical=True,
    )

    # Presence in answer: preseason lottery applies
    leaf_applicable = evaluator.add_leaf(
        id="Applicable_Lottery_System",
        desc="Identifies the preseason lottery as the applicable lottery for this scenario.",
        parent=group_node,
        critical=True,
    )
    await evaluator.verify(
        claim="The answer explicitly states that the applicable lottery is the preseason lottery for a September day hike when applying in March (not the daily lottery).",
        node=leaf_applicable,
        additional_instruction="Judge only whether the answer states this. Accept reasonable variants such as 'pre-season' or 'preseason March lottery'."
    )

    # URL support for lottery identification
    leaf_url = evaluator.add_leaf(
        id="Lottery_Type_URL_Reference",
        desc="Provides an official-source URL supporting the applicable lottery identification.",
        parent=group_node,
        critical=True,
    )
    await evaluator.verify(
        claim="At least one official-source URL provided in the answer confirms that Half Dome day-hike permits for dates during the cables season are allocated by a preseason lottery held in March.",
        node=leaf_url,
        sources=urls,
        additional_instruction=official_source_instruction("lottery type (preseason lottery applies for dates like September when applying in March)")
    )


async def verify_timing_and_results(evaluator: Evaluator, parent_node, urls: List[str]) -> None:
    group_node = evaluator.add_parallel(
        id="Timing_And_Results",
        desc="Correctly states application window timing (including timezone) and results timing for the applicable lottery.",
        parent=parent_node,
        critical=True,
    )

    # Presence in answer: application window and timezone
    leaf_window_tz = evaluator.add_leaf(
        id="Application_Window_Dates_And_Timezone",
        desc="States the preseason lottery application period is March 1–31 (Eastern Time).",
        parent=group_node,
        critical=True,
    )
    await evaluator.verify(
        claim="The answer explicitly states that the Half Dome preseason lottery application period is March 1–31 and that the times are Eastern Time (ET).",
        node=leaf_window_tz,
        additional_instruction="Judge only whether the answer states this. Accept minor variants like 'Mar 1–31' and 'ET' for Eastern Time."
    )

    # Presence in answer: results timing mid-April
    leaf_results = evaluator.add_leaf(
        id="Results_Announcement_Timing",
        desc="States preseason lottery results are announced in mid-April.",
        parent=group_node,
        critical=True,
    )
    await evaluator.verify(
        claim="The answer states that preseason lottery results are announced in mid-April.",
        node=leaf_results,
        additional_instruction="Judge only whether the answer states this. Accept reasonable phrasings such as 'around mid-April' or 'by mid-April'."
    )

    # URL support for timing
    leaf_url = evaluator.add_leaf(
        id="Timing_URL_Reference",
        desc="Provides an official-source URL supporting the application window and/or results timing.",
        parent=group_node,
        critical=True,
    )
    await evaluator.verify(
        claim="At least one official-source URL provided in the answer confirms the preseason application window (March 1–31, in Eastern Time) and/or that results are announced in mid-April.",
        node=leaf_url,
        sources=urls,
        additional_instruction=official_source_instruction("application window (including ET timezone) and/or results timing (mid-April)")
    )


async def verify_group_size_and_alternates(evaluator: Evaluator, parent_node, urls: List[str]) -> None:
    group_node = evaluator.add_parallel(
        id="Group_Size_And_Alternate_Rules",
        desc="Correctly states group size limits and alternate rules for the relevant lottery type(s).",
        parent=parent_node,
        critical=True,
    )

    # Presence: max group size 6
    leaf_max = evaluator.add_leaf(
        id="Maximum_Group_Size",
        desc="States preseason lottery applications can request up to 6 permits (people) per application.",
        parent=group_node,
        critical=True,
    )
    await evaluator.verify(
        claim="The answer states that a preseason lottery application can request permits for up to 6 people.",
        node=leaf_max,
        additional_instruction="Judge only whether the answer states this. Accept '6 hikers/people/permits'."
    )

    # Presence: alternates allowed preseason, not daily
    leaf_alt_policy = evaluator.add_leaf(
        id="Alternates_Permitted_Preseason_Not_Daily",
        desc="States alternates are allowed in the preseason lottery but not in the daily lottery.",
        parent=group_node,
        critical=True,
    )
    await evaluator.verify(
        claim="The answer states that alternates are allowed for the preseason lottery but alternates are not permitted for the daily lottery.",
        node=leaf_alt_policy,
        additional_instruction="Judge only whether the answer states this. Accept minor phrasing differences."
    )

    # Presence: alternates must accept within 72 hours
    leaf_alt_72 = evaluator.add_leaf(
        id="Alternate_Acceptance_Deadline",
        desc="States alternates must accept their role within 72 hours (preseason lottery).",
        parent=group_node,
        critical=True,
    )
    await evaluator.verify(
        claim="The answer states that alternates must accept their role within 72 hours for preseason lottery applications.",
        node=leaf_alt_72,
        additional_instruction="Judge only whether the answer states this. Accept 'within 72 hours' or 'in 72 hours'."
    )

    # URL support for group size + alternates (including 72-hour acceptance)
    leaf_url = evaluator.add_leaf(
        id="Group_And_Alternate_URL_Reference",
        desc="Provides an official-source URL supporting group size and alternate rules (including the 72-hour acceptance rule).",
        parent=group_node,
        critical=True,
    )
    await evaluator.verify(
        claim="At least one official-source URL provided in the answer confirms the rules: up to 6 people per preseason application, alternates allowed in the preseason lottery but not in the daily lottery, and that alternates must accept within 72 hours.",
        node=leaf_url,
        sources=urls,
        additional_instruction=official_source_instruction("group size limit and alternate rules (including 72-hour acceptance)")
    )


async def verify_application_requirements(evaluator: Evaluator, parent_node, urls: List[str]) -> None:
    group_node = evaluator.add_parallel(
        id="Application_Requirements",
        desc="Correctly states key application requirements they must follow.",
        parent=parent_node,
        critical=True,
    )

    # Presence: one application per individual per lottery
    leaf_one_app = evaluator.add_leaf(
        id="One_Application_Per_Individual_Per_Lottery",
        desc="States each individual may submit only one application per lottery.",
        parent=group_node,
        critical=True,
    )
    await evaluator.verify(
        claim="The answer states that each individual may submit only one application per lottery.",
        node=leaf_one_app,
        additional_instruction="Judge only whether the answer states this."
    )

    # Presence: one appearance per person per lottery
    leaf_one_appearance = evaluator.add_leaf(
        id="One_Application_Appearance_Per_Person_Per_Lottery",
        desc="States each person may appear on only one application per lottery (as permit holder or alternate).",
        parent=group_node,
        critical=True,
    )
    await evaluator.verify(
        claim="The answer states that each person may appear on only one application per lottery, whether as the permit holder or as an alternate.",
        node=leaf_one_appearance,
        additional_instruction="Judge only whether the answer states this."
    )

    # Presence: legal name required
    leaf_legal_name = evaluator.add_leaf(
        id="Legal_Name_Required",
        desc="States applicants must use their legal name when applying.",
        parent=group_node,
        critical=True,
    )
    await evaluator.verify(
        claim="The answer states that applicants must use their legal name when applying.",
        node=leaf_legal_name,
        additional_instruction="Judge only whether the answer states this."
    )

    # Presence: ID required at checkpoint
    leaf_id = evaluator.add_leaf(
        id="ID_Required_At_Checkpoint",
        desc="States permit holder or alternate must show government-issued ID matching the permit at the checkpoint.",
        parent=group_node,
        critical=True,
    )
    await evaluator.verify(
        claim="The answer states that the permit holder or alternate must show a government-issued ID that matches the name on the permit at the checkpoint.",
        node=leaf_id,
        additional_instruction="Judge only whether the answer states this."
    )

    # URL support for requirements
    leaf_url = evaluator.add_leaf(
        id="Application_Requirements_URL_Reference",
        desc="Provides an official-source URL supporting the key application requirements.",
        parent=group_node,
        critical=True,
    )
    await evaluator.verify(
        claim="At least one official-source URL provided in the answer clearly states one or more of these requirements: one application per individual per lottery; each person may appear on only one application per lottery; applicants must use their legal name; and the permit holder or alternate must show government-issued ID matching the permit at the checkpoint.",
        node=leaf_url,
        sources=urls,
        additional_instruction=official_source_instruction("key application requirements (one application rule, one appearance rule, legal name, ID at checkpoint)")
    )


async def verify_permit_season_quota_validity(evaluator: Evaluator, parent_node, urls: List[str]) -> None:
    group_node = evaluator.add_parallel(
        id="Permit_Season_Quota_And_Validity",
        desc="Correctly states constraints that determine when permits are needed and what the permit covers.",
        parent=parent_node,
        critical=True,
    )

    # Presence: cables typical season window
    leaf_cables_season = evaluator.add_leaf(
        id="Cables_Typical_Season",
        desc="States Half Dome cables are typically up from the Friday before Memorial Day through the day after the second Monday in October.",
        parent=group_node,
        critical=True,
    )
    await evaluator.verify(
        claim="The answer states that the Half Dome cables are typically up from the Friday before Memorial Day through the day after the second Monday in October.",
        node=leaf_cables_season,
        additional_instruction="Judge only whether the answer states this. Accept minor phrasing variations; the endpoints of the window must match."
    )

    # Presence: permits required when cables up
    leaf_permits_when_up = evaluator.add_leaf(
        id="Permits_Required_When_Cables_Up",
        desc="States permits are required seven days per week when cables are up.",
        parent=group_node,
        critical=True,
    )
    await evaluator.verify(
        claim="The answer states that permits are required seven days per week when the cables are up.",
        node=leaf_permits_when_up,
        additional_instruction="Judge only whether the answer states this. Accept phrasing such as 'every day' or '7 days a week'."
    )

    # Presence: daily day-hiker quota
    leaf_quota = evaluator.add_leaf(
        id="Daily_Day_Hiker_Quota",
        desc="States the daily quota of 225 day-hiker permits.",
        parent=group_node,
        critical=True,
    )
    await evaluator.verify(
        claim="The answer states that the daily day-hiker permit quota is 225.",
        node=leaf_quota,
        additional_instruction="Judge only whether the answer states this. Accept variants like '225 day-hikers per day' or 'quota of 225'."
    )

    # Presence: permit validity window
    leaf_validity = evaluator.add_leaf(
        id="Preseason_Permit_Validity_Window",
        desc="States preseason lottery permits are valid for a single day from 12:00 AM to 11:59 PM.",
        parent=group_node,
        critical=True,
    )
    await evaluator.verify(
        claim="The answer states that a preseason lottery permit is valid for a single day from 12:00 AM through 11:59 PM.",
        node=leaf_validity,
        additional_instruction="Judge only whether the answer states this. Accept minor variants like 'midnight to 11:59 PM'."
    )

    # URL support for seasonality/permit-needed rules, quota, validity
    leaf_url = evaluator.add_leaf(
        id="Season_Quota_Validity_URL_Reference",
        desc="Provides an official-source URL supporting seasonality/permit-needed rules, quota, and/or permit validity window.",
        parent=group_node,
        critical=True,
    )
    await evaluator.verify(
        claim="At least one official-source URL provided in the answer confirms any of the following: typical cables season window; that permits are required when cables are up; the daily day-hiker quota is 225; or that preseason lottery permits are valid for a single day (12:00 AM–11:59 PM).",
        node=leaf_url,
        sources=urls,
        additional_instruction=official_source_instruction("seasonality, permit-needed rule, daily quota (225), or single-day validity window")
    )


async def verify_fee_structure(evaluator: Evaluator, parent_node, urls: List[str]) -> None:
    group_node = evaluator.add_parallel(
        id="Fee_Structure",
        desc="Correctly provides the complete fee structure with specific amounts.",
        parent=parent_node,
        critical=True,
    )

    # Presence: $10 application fee
    leaf_app_fee = evaluator.add_leaf(
        id="Application_Fee",
        desc="States the non-refundable application fee is $10 per application.",
        parent=group_node,
        critical=True,
    )
    await evaluator.verify(
        claim="The answer states that the non-refundable application fee is $10 per application.",
        node=leaf_app_fee,
        additional_instruction="Judge only whether the answer states this. Accept minor formatting such as '$10.00'."
    )

    # Presence: $10 per person recreation fee charged when awarded
    leaf_rec_fee = evaluator.add_leaf(
        id="Recreation_Fee",
        desc="States the recreation fee is $10 per person and is charged when the permit is awarded.",
        parent=group_node,
        critical=True,
    )
    await evaluator.verify(
        claim="The answer states that the recreation fee is $10 per person and is charged when the permit is awarded (i.e., upon success).",
        node=leaf_rec_fee,
        additional_instruction="Judge only whether the answer states this. Accept phrasing like 'upon award' or 'if selected'."
    )

    # URL support for fee amounts and timing
    leaf_url = evaluator.add_leaf(
        id="Fees_URL_Reference",
        desc="Provides an official-source URL supporting the fee amounts and when they are charged.",
        parent=group_node,
        critical=True,
    )
    await evaluator.verify(
        claim="At least one official-source URL provided in the answer confirms both that the application fee is $10 per application and that the recreation fee is $10 per person charged when the permit is awarded.",
        node=leaf_url,
        sources=urls,
        additional_instruction=official_source_instruction("fee amounts and when they are charged (application: $10/app; recreation: $10/person upon award)")
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
    Evaluate an answer for the Half Dome day-hike permit application process task.
    """
    # 1) Initialize evaluator and root
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

    # 2) Extract categorized official URLs from the answer
    support_urls = await evaluator.extract(
        prompt=prompt_extract_support_urls(),
        template_class=SupportURLsExtraction,
        extraction_name="support_urls",
        additional_instruction="Prefer NPS (nps.gov) and Recreation.gov URLs when categorizing. Extract only URLs actually present in the answer."
    )

    # 3) Build top-level critical aggregator matching rubric
    app_complete = evaluator.add_parallel(
        id="Application_Process_Complete",
        desc="Answer explains the correct Half Dome day-hike permit application process for a Sept 10, 2026 hike when planning/applying in March 2026, including lottery type, timing, group/alternate rules, application requirements, permit-season/validity constraints, and complete fee structure, with official URL citations for each major component.",
        parent=root,
        critical=True,
    )

    # 4) Build and verify each rubric subtree
    await verify_lottery_type(
        evaluator,
        app_complete,
        urls=support_urls.lottery_urls,
    )

    await verify_timing_and_results(
        evaluator,
        app_complete,
        urls=support_urls.timing_urls,
    )

    await verify_group_size_and_alternates(
        evaluator,
        app_complete,
        urls=support_urls.group_alt_urls,
    )

    await verify_application_requirements(
        evaluator,
        app_complete,
        urls=support_urls.requirements_urls,
    )

    await verify_permit_season_quota_validity(
        evaluator,
        app_complete,
        urls=support_urls.season_quota_validity_urls,
    )

    await verify_fee_structure(
        evaluator,
        app_complete,
        urls=support_urls.fee_urls,
    )

    # 5) Return summary
    return evaluator.get_summary()