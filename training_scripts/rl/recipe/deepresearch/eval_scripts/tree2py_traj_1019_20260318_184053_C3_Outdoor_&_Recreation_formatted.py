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
TASK_ID = "half_dome_planning_guide_2026"
TASK_DESCRIPTION = (
    "I am planning a Half Dome day hike in Yosemite National Park with a group of 4 people during the 2026 summer "
    "season. Please provide a comprehensive planning guide that includes the following information: "
    "(1) Permit Lottery System: Explain the two types of permit lotteries available for Half Dome day hikers "
    "(preseason and daily lotteries), including specific application period dates, notification timing, daily permit "
    "quotas, and the application time windows. Provide reference URLs from official sources. "
    "(2) Application Requirements and Fees: Detail all requirements for submitting a permit application, including the "
    "legal name requirement, the one-application-per-person rule, group size limits (maximum number of people per "
    "application), ID verification requirements at the permit checkpoint, and the complete fee structure (both "
    "application fees and recreation fees). Provide reference URLs from official sources. "
    "(3) Trail Specifications: Provide the round-trip distance range, total elevation gain, the specific trailhead "
    "starting point and shuttle stop number, parking location, and the typical seasonal dates when the Half Dome "
    "cables are installed. "
    "(4) Hiking Logistics: Include the expected time duration for most hikers to complete the hike and the specific "
    "location where permits are checked on the trail. All information should be sourced from official National Park "
    "Service or Recreation.gov websites, with reference URLs provided where applicable."
)

# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class PermitLotteryInfo(BaseModel):
    permits_required_when_cables_up_text: Optional[str] = None
    distributed_only_recreation_gov_text: Optional[str] = None
    preseason_application_period_dates: Optional[str] = None
    preseason_notification_timing: Optional[str] = None
    daily_submission_timing: Optional[str] = None
    daily_application_window: Optional[str] = None
    daily_notification_timing: Optional[str] = None
    day_hiker_daily_permit_quota: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class ApplicationReqFeesInfo(BaseModel):
    legal_name_requirement_text: Optional[str] = None
    one_application_per_person_text: Optional[str] = None
    group_size_limit_text: Optional[str] = None
    id_verification_text: Optional[str] = None
    application_fee_text: Optional[str] = None
    recreation_fee_text: Optional[str] = None
    permit_validity_window_text: Optional[str] = None
    non_transferable_text: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class TrailSpecsInfo(BaseModel):
    distance_range_text: Optional[str] = None
    elevation_gain_text: Optional[str] = None
    trailhead_shuttle_text: Optional[str] = None
    parking_location_text: Optional[str] = None
    cables_season_dates_text: Optional[str] = None
    final_400_feet_angle_text: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class HikingLogisticsInfo(BaseModel):
    expected_time_text: Optional[str] = None
    permit_check_location_text: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class HalfDomeGuideExtraction(BaseModel):
    permit_lottery: Optional[PermitLotteryInfo] = None
    application: Optional[ApplicationReqFeesInfo] = None
    trail_specs: Optional[TrailSpecsInfo] = None
    logistics: Optional[HikingLogisticsInfo] = None


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_half_dome_guide() -> str:
    return """
Extract the requested Half Dome day-hike planning guide details EXACTLY as stated in the provided answer text.

Return a JSON object with these top-level fields:
- permit_lottery: {
    permits_required_when_cables_up_text: string or null,
    distributed_only_recreation_gov_text: string or null,
    preseason_application_period_dates: string or null,             # e.g., "March 1 through March 31"
    preseason_notification_timing: string or null,                  # e.g., "mid-April"
    daily_submission_timing: string or null,                        # e.g., "two days in advance"
    daily_application_window: string or null,                       # e.g., "midnight to 4:00 PM Pacific Time"
    daily_notification_timing: string or null,                      # e.g., "late afternoon/early evening same day"
    day_hiker_daily_permit_quota: string or null,                   # e.g., "225 day hikers beyond the base of the subdome"
    sources: [urls...]                                              # Extract ALL URLs in the answer relevant to permit lotteries
}
- application: {
    legal_name_requirement_text: string or null,
    one_application_per_person_text: string or null,
    group_size_limit_text: string or null,                          # e.g., "up to 6 people"
    id_verification_text: string or null,                           # e.g., "government-issued ID matching the permit name is required at the checkpoint"
    application_fee_text: string or null,                           # e.g., "$10 application fee per application"
    recreation_fee_text: string or null,                            # e.g., "$10 recreation fee per person when awarded"
    permit_validity_window_text: string or null,                    # e.g., "valid for a single day from 12:00 AM to 11:59 PM"
    non_transferable_text: string or null,                          # e.g., "permits are non-transferable"
    sources: [urls...]                                              # Extract ALL URLs in the answer relevant to requirements/fees
}
- trail_specs: {
    distance_range_text: string or null,                            # e.g., "14–16 miles"
    elevation_gain_text: string or null,                            # e.g., "approximately 4,800 feet"
    trailhead_shuttle_text: string or null,                         # e.g., "Happy Isles, shuttle stop #16"
    parking_location_text: string or null,                          # e.g., "parking just beyond Curry Village"
    cables_season_dates_text: string or null,                       # e.g., "Friday before the last Monday in May through the day after the second Monday in October"
    final_400_feet_angle_text: string or null,                      # e.g., "final 400 feet at approximately a 45-degree angle on steel cables"
    sources: [urls...]                                              # Extract ALL URLs in the answer relevant to trail specs
}
- logistics: {
    expected_time_text: string or null,                             # e.g., "10–12 hours"
    permit_check_location_text: string or null,                     # e.g., "permits are checked at the subdome before the cables"
    sources: [urls...]                                              # Extract ALL URLs in the answer relevant to logistics
}

RULES:
1) Do NOT invent or infer any values; copy text as shown in the answer.
2) If a given item is not stated, return null for that field.
3) For 'sources' arrays, extract only the URLs explicitly present in the answer text (plain URLs or within markdown links).
4) Include all sources listed for that section, regardless of domain; do not filter. We'll filter later if needed.
5) Keep numeric/text formats exactly as written (e.g., "14–16 miles", "mid-April", "$10").
"""


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _is_official_url(u: Optional[str]) -> bool:
    if not isinstance(u, str):
        return False
    lu = u.lower()
    return ("nps.gov" in lu) or ("recreation.gov" in lu)


def _filter_official_urls(urls: Optional[List[str]]) -> List[str]:
    if not urls:
        return []
    return [u for u in urls if _is_official_url(u)]


def _nonempty(s: Optional[str]) -> bool:
    return isinstance(s, str) and s.strip() != ""


async def _add_existence_and_verify(
    evaluator: Evaluator,
    parent_node,
    *,
    id_prefix: str,
    existence_desc: str,
    claim_desc: str,
    raw_text: Optional[str],
    default_claim: str,
    sources: List[str],
    additional_instruction: str,
) -> None:
    """
    Adds two leaf checks:
    1) Existence check (custom node) – whether the answer explicitly provided the requested statement/value.
    2) URL-backed verification – verify the (extracted or default) claim against official sources (if any).
    """
    # 1) Existence check (critical)
    evaluator.add_custom_node(
        result=_nonempty(raw_text),
        id=f"{id_prefix}_provided",
        desc=existence_desc,
        parent=parent_node,
        critical=True,
    )

    # 2) Verification leaf (critical)
    leaf = evaluator.add_leaf(
        id=id_prefix,
        desc=claim_desc,
        parent=parent_node,
        critical=True,
    )
    claim_text = raw_text.strip() if _nonempty(raw_text) else default_claim

    await evaluator.verify(
        claim=claim_text,
        node=leaf,
        sources=sources if sources else None,
        additional_instruction=additional_instruction,
    )


# --------------------------------------------------------------------------- #
# Section verifiers                                                           #
# --------------------------------------------------------------------------- #
async def verify_permit_lottery_system(
    evaluator: Evaluator,
    parent_node,
    info: Optional[PermitLotteryInfo],
) -> None:
    node = evaluator.add_parallel(
        id="Permit_Lottery_System",
        desc="Explains preseason and daily lotteries and required operational details, with official source URL(s).",
        parent=parent_node,
        critical=True,
    )

    sources_all = info.sources if info else []
    official_sources = _filter_official_urls(sources_all)

    # Official source URL presence (critical)
    evaluator.add_custom_node(
        result=len(official_sources) > 0,
        id="Permit_Lottery_System_Official_Source_URL",
        desc="Provides at least one official reference URL (NPS.gov or Recreation.gov) supporting the permit lottery system information.",
        parent=node,
        critical=True,
    )

    # Individual critical claims
    await _add_existence_and_verify(
        evaluator,
        node,
        id_prefix="Permits_Required_When_Cables_Up",
        existence_desc="Answer states permits are required for Half Dome day hikers when cables are up (provided in the answer).",
        claim_desc="States that permits are required for Half Dome day hikers when cables are up.",
        raw_text=(info.permits_required_when_cables_up_text if info else None),
        default_claim="Permits are required for Half Dome day hikers when the cables are up.",
        sources=official_sources,
        additional_instruction="Confirm the official page explicitly indicates that day-hike permits are required when the Half Dome cables are up.",
    )

    await _add_existence_and_verify(
        evaluator,
        node,
        id_prefix="Permits_Distributed_Only_Through_RecreationGov_Lottery",
        existence_desc="Answer states permits are distributed only through Recreation.gov lottery (provided in the answer).",
        claim_desc="States that Half Dome day hiker permits are distributed only through the Recreation.gov lottery system.",
        raw_text=(info.distributed_only_recreation_gov_text if info else None),
        default_claim="Half Dome day hiker permits are distributed only through the Recreation.gov lottery system (preseason and daily).",
        sources=official_sources,
        additional_instruction="Verify that the official source explains the Recreation.gov lottery as the exclusive mechanism for Half Dome day-hike permits.",
    )

    await _add_existence_and_verify(
        evaluator,
        node,
        id_prefix="Preseason_Lottery_Application_Period_Dates",
        existence_desc="Answer provides the preseason lottery application period dates (provided in the answer).",
        claim_desc="States the preseason lottery application period dates (March 1 through March 31).",
        raw_text=(info.preseason_application_period_dates if info else None),
        default_claim="The preseason lottery application period is March 1 through March 31.",
        sources=official_sources,
        additional_instruction="Verify the exact preseason lottery application period window on the official source. Allow reasonable phrasing variants such as 'March 1–31'.",
    )

    await _add_existence_and_verify(
        evaluator,
        node,
        id_prefix="Preseason_Lottery_Notification_Timing",
        existence_desc="Answer provides the preseason lottery notification timing (provided in the answer).",
        claim_desc="States the preseason lottery notification timing (mid-April).",
        raw_text=(info.preseason_notification_timing if info else None),
        default_claim="Preseason lottery results notifications occur in mid-April.",
        sources=official_sources,
        additional_instruction="Verify the official page mentions when applicants are notified of the preseason lottery results (e.g., mid-April).",
    )

    await _add_existence_and_verify(
        evaluator,
        node,
        id_prefix="Daily_Lottery_Submission_Timing",
        existence_desc="Answer provides the daily lottery submission timing (provided in the answer).",
        claim_desc="States that daily lottery applications must be submitted two days in advance of the desired hiking date.",
        raw_text=(info.daily_submission_timing if info else None),
        default_claim="Daily lottery applications must be submitted two days in advance of the intended hiking date.",
        sources=official_sources,
        additional_instruction="Confirm the official source states that the daily lottery application is submitted two days before the hike date.",
    )

    await _add_existence_and_verify(
        evaluator,
        node,
        id_prefix="Daily_Lottery_Application_Window",
        existence_desc="Answer provides the daily lottery application time window (provided in the answer).",
        claim_desc="States the daily lottery application time window (midnight to 4:00 PM Pacific Time).",
        raw_text=(info.daily_application_window if info else None),
        default_claim="The daily lottery application window is from midnight to 4:00 PM Pacific Time.",
        sources=official_sources,
        additional_instruction="Verify the official source gives the daily lottery application window and time zone (Pacific Time).",
    )

    await _add_existence_and_verify(
        evaluator,
        node,
        id_prefix="Daily_Lottery_Notification_Timing",
        existence_desc="Answer provides the daily lottery notification timing (provided in the answer).",
        claim_desc="Provides the notification timing for the daily lottery (as stated by an official NPS.gov or Recreation.gov source).",
        raw_text=(info.daily_notification_timing if info else None),
        default_claim="Daily lottery results/notifications are sent later the same day after the application window closes (late afternoon/early evening, Pacific Time).",
        sources=official_sources,
        additional_instruction="Confirm the official page describes when daily lottery applicants are notified (e.g., later the same day). Allow phrasing variants.",
    )

    await _add_existence_and_verify(
        evaluator,
        node,
        id_prefix="Day_Hiker_Daily_Permit_Quota",
        existence_desc="Answer provides the daily permit quota for day hikers (provided in the answer).",
        claim_desc="States the daily permit quota for day hikers (225 day hikers beyond the base of the subdome).",
        raw_text=(info.day_hiker_daily_permit_quota if info else None),
        default_claim="The daily permit quota is 225 day hikers beyond the base of the subdome.",
        sources=official_sources,
        additional_instruction="Verify the official source explicitly states (or clearly implies) the daily quota for Half Dome day hikers (e.g., 225).",
    )


async def verify_application_requirements_and_fees(
    evaluator: Evaluator,
    parent_node,
    info: Optional[ApplicationReqFeesInfo],
) -> None:
    node = evaluator.add_parallel(
        id="Application_Requirements_and_Fees",
        desc="All requirements for submitting a permit application and complete fee structure, with official source URL(s).",
        parent=parent_node,
        critical=True,
    )

    sources_all = info.sources if info else []
    official_sources = _filter_official_urls(sources_all)

    evaluator.add_custom_node(
        result=len(official_sources) > 0,
        id="Application_Requirements_Official_Source_URL",
        desc="Provides at least one official reference URL (NPS.gov or Recreation.gov) supporting the application requirements and fees information.",
        parent=node,
        critical=True,
    )

    await _add_existence_and_verify(
        evaluator,
        node,
        id_prefix="Legal_Name_Requirement",
        existence_desc="Answer states the applicant must use their legal name (provided in the answer).",
        claim_desc="States that the applicant must use their legal name.",
        raw_text=(info.legal_name_requirement_text if info else None),
        default_claim="The applicant must use their legal name on the application.",
        sources=official_sources,
        additional_instruction="Verify the official page indicates applications must be submitted using the applicant's legal name.",
    )

    await _add_existence_and_verify(
        evaluator,
        node,
        id_prefix="One_Application_Per_Person_Rule",
        existence_desc="Answer states one-application-per-person rule (provided in the answer).",
        claim_desc="States that each individual may submit only one application per lottery.",
        raw_text=(info.one_application_per_person_text if info else None),
        default_claim="Each person may submit only one application per lottery.",
        sources=official_sources,
        additional_instruction="Verify the official page states that duplicate/multiple applications by the same individual are not allowed.",
    )

    await _add_existence_and_verify(
        evaluator,
        node,
        id_prefix="Group_Size_Limit",
        existence_desc="Answer states maximum group size per application (provided in the answer).",
        claim_desc="States the maximum number of people/permits per application (up to 6).",
        raw_text=(info.group_size_limit_text if info else None),
        default_claim="The maximum group size per application is up to 6 people.",
        sources=official_sources,
        additional_instruction="Verify the official page lists the maximum number of people allowed on one application (e.g., up to 6).",
    )

    await _add_existence_and_verify(
        evaluator,
        node,
        id_prefix="ID_Verification_Requirement",
        existence_desc="Answer states ID verification requirement at checkpoint (provided in the answer).",
        claim_desc="States that the permit holder must show a government-issued ID that matches the name on the permit at the checkpoint.",
        raw_text=(info.id_verification_text if info else None),
        default_claim="The permit holder must present a government-issued photo ID that matches the name on the permit at the checkpoint.",
        sources=official_sources,
        additional_instruction="Verify the official page describes ID matching requirements at the Half Dome checkpoint.",
    )

    await _add_existence_and_verify(
        evaluator,
        node,
        id_prefix="Application_Fee",
        existence_desc="Answer provides the application fee (provided in the answer).",
        claim_desc="States the non-refundable $10 application fee per application.",
        raw_text=(info.application_fee_text if info else None),
        default_claim="There is a non-refundable $10 application fee per application.",
        sources=official_sources,
        additional_instruction="Verify the official page states the application fee amount and that it is per application.",
    )

    await _add_existence_and_verify(
        evaluator,
        node,
        id_prefix="Recreation_Fee",
        existence_desc="Answer provides the recreation fee (provided in the answer).",
        claim_desc="States the $10 recreation fee per person when the permit is awarded.",
        raw_text=(info.recreation_fee_text if info else None),
        default_claim="A $10 recreation fee per person is charged when a permit is awarded (or when adding hikers).",
        sources=official_sources,
        additional_instruction="Verify the official page states the recreation fee is per person and is charged upon award/reservation.",
    )

    await _add_existence_and_verify(
        evaluator,
        node,
        id_prefix="Permit_Validity_Window",
        existence_desc="Answer provides the permit validity window (provided in the answer).",
        claim_desc="States that permits are valid for a single day from 12:00 AM to 11:59 PM.",
        raw_text=(info.permit_validity_window_text if info else None),
        default_claim="Half Dome day-hike permits are valid for a single day from 12:00 AM to 11:59 PM.",
        sources=official_sources,
        additional_instruction="Verify the official page clarifies the permit is valid for the single calendar day of the hike.",
    )

    await _add_existence_and_verify(
        evaluator,
        node,
        id_prefix="Permit_Non_Transferable",
        existence_desc="Answer states the non-transferable nature of permits (provided in the answer).",
        claim_desc="States that permits are non-transferable.",
        raw_text=(info.non_transferable_text if info else None),
        default_claim="Half Dome day-hike permits are non-transferable.",
        sources=official_sources,
        additional_instruction="Verify the official page states that permits cannot be transferred to another person.",
    )


async def verify_trail_specifications(
    evaluator: Evaluator,
    parent_node,
    info: Optional[TrailSpecsInfo],
) -> None:
    node = evaluator.add_parallel(
        id="Trail_Specifications",
        desc="Trail stats and access specifics requested, with official source URL(s).",
        parent=parent_node,
        critical=True,
    )

    sources_all = info.sources if info else []
    official_sources = _filter_official_urls(sources_all)

    evaluator.add_custom_node(
        result=len(official_sources) > 0,
        id="Trail_Specs_Official_Source_URL",
        desc="Provides at least one official reference URL (NPS.gov or Recreation.gov) supporting trail specification information.",
        parent=node,
        critical=True,
    )

    await _add_existence_and_verify(
        evaluator,
        node,
        id_prefix="Round_Trip_Distance_Range",
        existence_desc="Answer provides the round-trip distance range (provided in the answer).",
        claim_desc="States the round-trip distance range (14–16 miles).",
        raw_text=(info.distance_range_text if info else None),
        default_claim="The Half Dome day-hike round-trip distance is 14–16 miles (depending on route and start).",
        sources=official_sources,
        additional_instruction="Verify the official source states (or clearly implies) a round-trip distance range around 14–16 miles.",
    )

    await _add_existence_and_verify(
        evaluator,
        node,
        id_prefix="Total_Elevation_Gain",
        existence_desc="Answer provides the total elevation gain (provided in the answer).",
        claim_desc="States the total elevation gain (approximately 4,800 feet).",
        raw_text=(info.elevation_gain_text if info else None),
        default_claim="The total elevation gain is approximately 4,800 feet.",
        sources=official_sources,
        additional_instruction="Verify the official page indicates total elevation gain around 4,800 feet (allow approximate wording).",
    )

    await _add_existence_and_verify(
        evaluator,
        node,
        id_prefix="Trailhead_And_Shuttle_Stop",
        existence_desc="Answer provides the trailhead and shuttle stop (provided in the answer).",
        claim_desc="States the trailhead starting point and shuttle stop number (Happy Isles, shuttle stop #16).",
        raw_text=(info.trailhead_shuttle_text if info else None),
        default_claim="The Half Dome hike typically starts at Happy Isles (shuttle stop #16).",
        sources=official_sources,
        additional_instruction="Verify the official source identifies Happy Isles as the typical trailhead and includes the shuttle stop number (e.g., #16).",
    )

    await _add_existence_and_verify(
        evaluator,
        node,
        id_prefix="Parking_Location",
        existence_desc="Answer provides the parking location (provided in the answer).",
        claim_desc="States the parking location (just beyond Curry Village).",
        raw_text=(info.parking_location_text if info else None),
        default_claim="Trailhead parking is located just beyond Curry Village.",
        sources=official_sources,
        additional_instruction="Verify the official page describes appropriate parking for the trailhead area (just past/beyond Curry Village).",
    )

    await _add_existence_and_verify(
        evaluator,
        node,
        id_prefix="Typical_Cables_Season_Dates",
        existence_desc="Answer provides typical cables season dates (provided in the answer).",
        claim_desc='States typical seasonal dates when cables are installed/"up" (Friday before last Monday in May through the day after the second Monday in October), noting dates are subject to conditions.',
        raw_text=(info.cables_season_dates_text if info else None),
        default_claim="The Half Dome cables are typically up from the Friday before the last Monday in May through the day after the second Monday in October, conditions permitting.",
        sources=official_sources,
        additional_instruction="Verify the official page describes typical seasonal dates for cables being up; allow phrasing variants and note they depend on conditions.",
    )

    await _add_existence_and_verify(
        evaluator,
        node,
        id_prefix="Final_400_Feet_Cables_Angle",
        existence_desc="Answer describes the final 400 feet cables angle (provided in the answer).",
        claim_desc="States that the final 400 feet requires climbing the steel cables at approximately a 45-degree angle.",
        raw_text=(info.final_400_feet_angle_text if info else None),
        default_claim="The final 400 feet of Half Dome requires ascending steel cables at approximately a 45-degree angle.",
        sources=official_sources,
        additional_instruction="Verify the official page mentions the steepness and use of steel cables on the last ~400 feet (around 45 degrees).",
    )


async def verify_hiking_logistics(
    evaluator: Evaluator,
    parent_node,
    info: Optional[HikingLogisticsInfo],
) -> None:
    node = evaluator.add_parallel(
        id="Hiking_Logistics",
        desc="Time duration and where permits are checked, with official source URL(s).",
        parent=parent_node,
        critical=True,
    )

    sources_all = info.sources if info else []
    official_sources = _filter_official_urls(sources_all)

    evaluator.add_custom_node(
        result=len(official_sources) > 0,
        id="Hiking_Logistics_Official_Source_URL",
        desc="Provides at least one official reference URL (NPS.gov or Recreation.gov) supporting hiking logistics information.",
        parent=node,
        critical=True,
    )

    await _add_existence_and_verify(
        evaluator,
        node,
        id_prefix="Expected_Time_Duration",
        existence_desc="Answer provides the expected time duration (provided in the answer).",
        claim_desc="States expected time duration for most hikers (10–12 hours).",
        raw_text=(info.expected_time_text if info else None),
        default_claim="Most hikers take about 10–12 hours to complete the Half Dome day hike.",
        sources=official_sources,
        additional_instruction="Verify the official page indicates a typical total hiking time around 10–12 hours for most hikers.",
    )

    await _add_existence_and_verify(
        evaluator,
        node,
        id_prefix="Permit_Check_Location",
        existence_desc="Answer provides where permits are checked on the trail (provided in the answer).",
        claim_desc="States the specific location where permits are checked on the trail (at the subdome, before access to the cables section).",
        raw_text=(info.permit_check_location_text if info else None),
        default_claim="Half Dome day-hike permits are checked at the subdome, before the cables section.",
        sources=official_sources,
        additional_instruction="Verify the official page states permits are checked at/near the subdome prior to cable access.",
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
    Evaluate an answer for the Half Dome planning guide task.
    """
    # Initialize Evaluator
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
    extracted: HalfDomeGuideExtraction = await evaluator.extract(
        prompt=prompt_extract_half_dome_guide(),
        template_class=HalfDomeGuideExtraction,
        extraction_name="half_dome_guide_extraction",
    )

    # Build top-level critical node matching rubric root
    rubric_root = evaluator.add_parallel(
        id="Complete_Half_Dome_Planning_Guide",
        desc="Comprehensive planning guide for a Half Dome day hike covering permit lotteries, application requirements/fees, trail specs, and hiking logistics, with official NPS.gov or Recreation.gov citations where applicable.",
        parent=root,
        critical=True,
    )

    # Verify each major section (all critical)
    await verify_permit_lottery_system(evaluator, rubric_root, extracted.permit_lottery or PermitLotteryInfo())
    await verify_application_requirements_and_fees(evaluator, rubric_root, extracted.application or ApplicationReqFeesInfo())
    await verify_trail_specifications(evaluator, rubric_root, extracted.trail_specs or TrailSpecsInfo())
    await verify_hiking_logistics(evaluator, rubric_root, extracted.logistics or HikingLogisticsInfo())

    # Return structured evaluation summary
    return evaluator.get_summary()