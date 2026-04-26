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
TASK_ID = "nc_top3_districts_weather_policies"
TASK_DESCRIPTION = """
Identify the three largest public school districts in North Carolina by student enrollment that have publicly documented inclement weather closing and delay notification policies. For each of the three districts, provide the following information: (1) The district name and current student enrollment; (2) Documentation that the district has publicly available weather notification procedures; (3) The notification timeline target specified by the district for making weather-related decisions (such as by 5:30 AM, 6:00 AM, or the evening before); (4) At least three different communication methods the district uses to notify families of closures or delays (such as email, text message, phone calls, website updates, or mobile app notifications); (5) The name of the current district superintendent; (6) The district's main contact phone number; (7) The physical address of the district's main administrative office; (8) Who is documented as making the final decision on weather-related school closures or delays; (9) Documentation of the district's two-hour delay procedures or schedule; (10) The specific emergency notification system or technology platform the district uses for mass communications.
"""


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class District(BaseModel):
    # Core identification and ranking
    name: Optional[str] = None
    enrollment: Optional[str] = None
    enrollment_sources: List[str] = Field(default_factory=list)

    # Weather policy and timing
    weather_policy_urls: List[str] = Field(default_factory=list)
    notification_timeline: Optional[str] = None
    notification_timeline_sources: List[str] = Field(default_factory=list)

    # Communication channels
    communication_channels: List[str] = Field(default_factory=list)
    communication_sources: List[str] = Field(default_factory=list)

    # Superintendent
    superintendent_name: Optional[str] = None
    superintendent_sources: List[str] = Field(default_factory=list)

    # Contact info
    main_phone: Optional[str] = None
    phone_sources: List[str] = Field(default_factory=list)
    address: Optional[str] = None
    address_sources: List[str] = Field(default_factory=list)

    # Decision maker for closures/delays
    decision_maker: Optional[str] = None
    decision_sources: List[str] = Field(default_factory=list)

    # Two-hour delay details
    two_hour_delay_procedures: Optional[str] = None
    two_hour_delay_sources: List[str] = Field(default_factory=list)

    # Mass notification system/platform
    notification_system: Optional[str] = None
    notification_system_sources: List[str] = Field(default_factory=list)


class DistrictsExtraction(BaseModel):
    districts: List[District] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_districts() -> str:
    return """
    Extract up to three North Carolina public school districts discussed in the answer that are intended to be the three largest by student enrollment and that have publicly documented inclement weather closing/delay notification procedures.

    IMPORTANT:
    - Maintain the original order in which the districts appear in the answer. Do NOT reorder by size yourself.
    - Extract only what the answer explicitly states. If a field is not provided in the answer, set it to null (for strings) or [] (for arrays).
    - For any "sources" fields below, include only actual URLs explicitly present in the answer. If none are present, return an empty array.

    For each district, extract the following fields:
    1) name: The district name
    2) enrollment: The current student enrollment value (string as shown in the answer; keep formatting)
    3) enrollment_sources: Array of URL(s) in the answer that support the enrollment figure or ranking/top-3 claim
    4) weather_policy_urls: Array of URL(s) that directly point to the district's official inclement weather policy or procedures page(s)
    5) notification_timeline: The stated notification timeline target (e.g., "by 5:30 AM", "by 6:00 AM", "the evening before")
    6) notification_timeline_sources: Array of URL(s) that support the stated notification timeline
    7) communication_channels: Array of distinct communication methods mentioned (e.g., "email", "text message", "phone call", "website", "mobile app", "social media", "local media")
    8) communication_sources: Array of URL(s) that support the listed communications methods
    9) superintendent_name: The name of the current superintendent
    10) superintendent_sources: Array of URL(s) that support the superintendent's name
    11) main_phone: The district's main contact phone number (string as in the answer)
    12) phone_sources: Array of URL(s) that support the main phone number
    13) address: The physical address of the district's main administrative office
    14) address_sources: Array of URL(s) that support the physical address
    15) decision_maker: Who the district documents as making the final decision on closures/delays (e.g., "Superintendent" or "Superintendent in consultation with ...")
    16) decision_sources: Array of URL(s) that support who makes the decision
    17) two_hour_delay_procedures: The district’s description of two-hour delay procedures/schedule (short summary; keep as a string)
    18) two_hour_delay_sources: Array of URL(s) that support the two-hour delay procedures/schedule
    19) notification_system: The specific mass notification system or platform used (e.g., "Blackboard", "SchoolMessenger", "ParentSquare", "Remind")
    20) notification_system_sources: Array of URL(s) that support the identified system/platform

    Return JSON with a top-level "districts" array of up to three district objects in the same order as in the answer.
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def dedupe_urls(urls: List[str]) -> List[str]:
    seen = set()
    out = []
    for u in urls:
        if not u:
            continue
        if u not in seen:
            seen.add(u)
            out.append(u)
    return out


def coalesce_sources(*lists: List[str]) -> List[str]:
    combined: List[str] = []
    for lst in lists:
        combined.extend(lst or [])
    return dedupe_urls(combined)


def ordinal_from_index(idx1_based: int) -> str:
    mapping = {1: "First", 2: "Second", 3: "Third"}
    return mapping.get(idx1_based, f"#{idx1_based}")


def add_no_source_or_value_instruction(
    has_sources: bool,
    has_value: bool,
    base: str = ""
) -> str:
    suffix = ""
    if not has_sources:
        suffix += "\nImportant: No source URL was provided for this verification. Conclude the claim is NOT supported."
    if not has_value:
        suffix += "\nImportant: The answer does not provide a concrete value for this item. Conclude the claim is NOT supported."
    return (base + suffix).strip() if base or suffix else "None"


# --------------------------------------------------------------------------- #
# Verification logic for a single district                                    #
# --------------------------------------------------------------------------- #
async def verify_single_district(
    evaluator: Evaluator,
    parent_node,
    district: District,
    idx1_based: int,
) -> None:
    # Parent (parallel) node for this district
    district_node = evaluator.add_parallel(
        id=f"district_{idx1_based}",
        desc=f"{ordinal_from_index(idx1_based)} largest qualifying school district in North Carolina",
        parent=parent_node,
        critical=False
    )

    # Prepare common identifiers and fallbacks
    district_name = district.name or "the district"

    # Build leaves according to rubric
    # 1) Enrollment ranking (critical)
    node_enrollment_rank = evaluator.add_leaf(
        id=f"district_{idx1_based}_enrollment_ranking",
        desc="District is confirmed as one of the three largest school districts in North Carolina by current student enrollment",
        parent=district_node,
        critical=True
    )
    claim_enrollment = f"The district '{district_name}' is one of the three largest school districts in North Carolina by current student enrollment."
    sources_enrollment = district.enrollment_sources
    ins_enrollment = add_no_source_or_value_instruction(
        has_sources=bool(sources_enrollment),
        has_value=bool(district.name),
        base="Look for a North Carolina district enrollment ranking or a page listing the largest districts by student enrollment."
    )

    # 2) Weather policy documented (critical)
    node_policy = evaluator.add_leaf(
        id=f"district_{idx1_based}_weather_policy_documented",
        desc="District has publicly documented inclement weather closing/delay notification procedures accessible through official sources",
        parent=district_node,
        critical=True
    )
    claim_policy = f"This page documents the inclement weather closing/delay notification procedures for {district_name}."
    sources_policy = district.weather_policy_urls
    ins_policy = add_no_source_or_value_instruction(
        has_sources=bool(sources_policy),
        has_value=True,
        base="Verify the page is official (district or government) and directly discusses inclement weather procedures."
    )

    # 3) Notification timing (critical)
    node_timing = evaluator.add_leaf(
        id=f"district_{idx1_based}_notification_timing",
        desc="District specifies a notification timeline target for weather-related decisions (e.g., by 5:30 AM, 6:00 AM, or the evening before)",
        parent=district_node,
        critical=True
    )
    timing_value = (district.notification_timeline or "").strip()
    claim_timing = f"The district specifies the weather decision notification timeline target as '{timing_value}'."
    sources_timing = coalesce_sources(district.notification_timeline_sources, district.weather_policy_urls)
    ins_timing = add_no_source_or_value_instruction(
        has_sources=bool(sources_timing),
        has_value=bool(timing_value),
        base="Accept close paraphrases like 'no later than 5:30 a.m.' or 'by 6:00 AM', or 'the evening before when possible'."
    )

    # 4) Communication channels (critical; need >=3)
    node_channels = evaluator.add_leaf(
        id=f"district_{idx1_based}_communication_channels",
        desc="District documents the use of at least three different notification methods (such as email, text message, phone call, website, mobile app)",
        parent=district_node,
        critical=True
    )
    channels_list = [c.strip() for c in (district.communication_channels or []) if c and c.strip()]
    channels_preview = ", ".join(channels_list[:6])
    claim_channels = f"The district documents at least three different notification methods for closures/delays, including: {channels_preview}."
    sources_channels = coalesce_sources(district.communication_sources, district.weather_policy_urls)
    ins_channels = add_no_source_or_value_instruction(
        has_sources=bool(sources_channels),
        has_value=(len(channels_list) >= 3),
        base="Confirm that the page lists at least three distinct channels among email, text/SMS, phone/robocall, website, mobile app, social media, local media, etc."
    )

    # 5) Superintendent info (non-critical)
    node_superintendent = evaluator.add_leaf(
        id=f"district_{idx1_based}_superintendent_info",
        desc="District provides the name of the current superintendent",
        parent=district_node,
        critical=False
    )
    superintendent_value = (district.superintendent_name or "").strip()
    claim_superintendent = f"The current superintendent of {district_name} is '{superintendent_value}'."
    sources_superintendent = district.superintendent_sources or []
    if not sources_superintendent:
        # Fall back to any provided sources if superintendent-specific sources are missing
        sources_superintendent = coalesce_sources(
            district.weather_policy_urls,
            district.communication_sources,
            district.enrollment_sources
        )
    ins_superintendent = add_no_source_or_value_instruction(
        has_sources=bool(sources_superintendent),
        has_value=bool(superintendent_value),
        base="Verify on an official district page (leadership/superintendent page or similar). Allow minor name variations."
    )

    # 6) District main phone (non-critical)
    node_phone = evaluator.add_leaf(
        id=f"district_{idx1_based}_district_phone",
        desc="District provides a main contact phone number",
        parent=district_node,
        critical=False
    )
    phone_value = (district.main_phone or "").strip()
    claim_phone = f"The main contact phone number for {district_name} is '{phone_value}'."
    sources_phone = district.phone_sources or []
    if not sources_phone:
        sources_phone = coalesce_sources(district.address_sources, district.communication_sources, district.weather_policy_urls)
    ins_phone = add_no_source_or_value_instruction(
        has_sources=bool(sources_phone),
        has_value=bool(phone_value),
        base="Verify that this number appears as the district main or central office contact on an official page."
    )

    # 7) District address (non-critical)
    node_address = evaluator.add_leaf(
        id=f"district_{idx1_based}_district_address",
        desc="District provides the physical address of the main administrative office",
        parent=district_node,
        critical=False
    )
    address_value = (district.address or "").strip()
    claim_address = f"The main administrative office address of {district_name} is '{address_value}'."
    sources_address = district.address_sources or []
    if not sources_address:
        sources_address = coalesce_sources(district.phone_sources, district.communication_sources, district.weather_policy_urls)
    ins_address = add_no_source_or_value_instruction(
        has_sources=bool(sources_address),
        has_value=bool(address_value),
        base="Verify that the address is clearly indicated as the district central office/administration address."
    )

    # 8) Decision maker (non-critical)
    node_decision = evaluator.add_leaf(
        id=f"district_{idx1_based}_decision_maker",
        desc="District documents who makes the final decision on weather-related closures or delays",
        parent=district_node,
        critical=False
    )
    decision_value = (district.decision_maker or "").strip()
    claim_decision = f"The final decision on weather-related closures or delays for {district_name} is made by {decision_value}."
    sources_decision = district.decision_sources or coalesce_sources(district.weather_policy_urls)
    ins_decision = add_no_source_or_value_instruction(
        has_sources=bool(sources_decision),
        has_value=bool(decision_value),
        base="Accept formulations like 'the Superintendent' or 'the Superintendent in consultation with district leadership'."
    )

    # 9) Two-hour delay procedures (non-critical)
    node_two_hour = evaluator.add_leaf(
        id=f"district_{idx1_based}_two_hour_delay_procedures",
        desc="District documents specific procedures for two-hour delay schedules",
        parent=district_node,
        critical=False
    )
    two_hour_value_present = bool((district.two_hour_delay_procedures or "").strip())
    claim_two_hour = f"This page documents the district's two-hour delay procedures or schedule for {district_name}."
    sources_two_hour = district.two_hour_delay_sources or coalesce_sources(district.weather_policy_urls)
    ins_two_hour = add_no_source_or_value_instruction(
        has_sources=bool(sources_two_hour),
        has_value=two_hour_value_present,
        base="Look for 'Two-hour delay', '2-hour delay', or explicit delayed start schedule details."
    )

    # 10) Notification system/platform (non-critical)
    node_system = evaluator.add_leaf(
        id=f"district_{idx1_based}_notification_system",
        desc="District identifies the specific technology platform or system used for mass notifications",
        parent=district_node,
        critical=False
    )
    system_value = (district.notification_system or "").strip()
    claim_system = f"The district uses '{system_value}' as its mass/emergency notification system."
    sources_system = district.notification_system_sources or coalesce_sources(district.communication_sources, district.weather_policy_urls)
    ins_system = add_no_source_or_value_instruction(
        has_sources=bool(sources_system),
        has_value=bool(system_value),
        base="Verify that the platform name (e.g., Blackboard, SchoolMessenger, ParentSquare, Remind) is explicitly mentioned."
    )

    # Batch verify all leaves for this district
    verifications = [
        (claim_enrollment, sources_enrollment, node_enrollment_rank, ins_enrollment),
        (claim_policy, sources_policy, node_policy, ins_policy),
        (claim_timing, sources_timing, node_timing, ins_timing),
        (claim_channels, sources_channels, node_channels, ins_channels),
        (claim_superintendent, sources_superintendent, node_superintendent, ins_superintendent),
        (claim_phone, sources_phone, node_phone, ins_phone),
        (claim_address, sources_address, node_address, ins_address),
        (claim_decision, sources_decision, node_decision, ins_decision),
        (claim_two_hour, sources_two_hour, node_two_hour, ins_two_hour),
        (claim_system, sources_system, node_system, ins_system),
    ]

    await evaluator.batch_verify(verifications)


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
    # Initialize evaluator with a parallel root node (3 districts evaluated independently)
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

    # Extraction
    extracted = await evaluator.extract(
        prompt=prompt_extract_districts(),
        template_class=DistrictsExtraction,
        extraction_name="districts_extraction"
    )

    # Normalize to exactly 3 districts (pad if fewer)
    districts: List[District] = list(extracted.districts[:3])
    while len(districts) < 3:
        districts.append(District())

    # Build verification subtrees for each of the three districts
    tasks = []
    for i in range(3):
        tasks.append(verify_single_district(evaluator, root, districts[i], i + 1))
    await asyncio.gather(*tasks)

    # Return evaluation summary
    return evaluator.get_summary()