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
TASK_ID = "big_sur_camping_march_2026"
TASK_DESCRIPTION = """
A family of 6 adults is planning a camping trip to the Big Sur area during late March 2026, with plans to arrive on March 26, 2026. They will be traveling in a 28-foot motorhome and need to book a campsite at a California State Park campground that is located along the Big Sur River and offers hot shower facilities.

During their stay, they want to visit McWay Falls and go hiking in the Big Sur River Gorge area at Pfeiffer Big Sur State Park. One family member also wants to apply for the Half Dome day hike permit lottery to use during a potential future Yosemite trip.

Based on conditions and regulations as of March 2026, provide the following information:

1. Which specific state park campground should they book that meets all their requirements (river location, hot showers, can accommodate their 28-foot motorhome and group of 6 adults)?

2. What is the correct website to make camping reservations, and how far in advance can reservations be made?

3. Which of their planned activities (visiting McWay Falls Overlook Trail, hiking in Big Sur River Gorge) are actually accessible in March 2026, and what are the reasons for any closures?

4. What is the deadline to submit an application for the Half Dome preseason permit lottery that would be relevant during their late March camping trip?

For each piece of information, provide the supporting reference URL(s).
"""


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class CampgroundInfo(BaseModel):
    campground_name: Optional[str] = None
    campground_source_urls: List[str] = Field(default_factory=list)


class ReservationInfo(BaseModel):
    reservation_website_name: Optional[str] = None
    reservation_website_urls: List[str] = Field(default_factory=list)
    advance_booking_period_text: Optional[str] = None  # e.g., "6 months in advance"
    booking_time_detail: Optional[str] = None  # e.g., "8:00 AM PST/PDT"


class ActivityStatusInfo(BaseModel):
    # McWay Falls (Julia Pfeiffer Burns SP)
    mcway_status: Optional[str] = None  # e.g., "closed", "open", "partially closed"
    mcway_closure_reason: Optional[str] = None
    mcway_alternative_text: Optional[str] = None  # what alternative is suggested in the answer
    mcway_source_urls: List[str] = Field(default_factory=list)

    # Big Sur River Gorge (Pfeiffer Big Sur SP)
    gorge_status: Optional[str] = None  # e.g., "closed", "open with caution"
    gorge_closure_reason: Optional[str] = None
    gorge_source_urls: List[str] = Field(default_factory=list)


class HalfDomeLotteryInfo(BaseModel):
    preseason_lottery_start_date: Optional[str] = None  # e.g., "March 1, 2026"
    preseason_lottery_end_date: Optional[str] = None    # e.g., "March 31, 2026"
    lottery_platform_name: Optional[str] = None         # e.g., "Recreation.gov"
    lottery_source_urls: List[str] = Field(default_factory=list)


class PlanningExtraction(BaseModel):
    campground: Optional[CampgroundInfo] = None
    reservation: Optional[ReservationInfo] = None
    activities: Optional[ActivityStatusInfo] = None
    half_dome: Optional[HalfDomeLotteryInfo] = None


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_planning() -> str:
    return """
    Extract the required planning information exactly as presented in the answer.

    Return a JSON object with the following structure:
    {
      "campground": {
        "campground_name": string or null,
        "campground_source_urls": [array of URLs explicitly cited for campground info]
      },
      "reservation": {
        "reservation_website_name": string or null,
        "reservation_website_urls": [array of URLs explicitly cited for reservations],
        "advance_booking_period_text": string or null,
        "booking_time_detail": string or null
      },
      "activities": {
        "mcway_status": string or null,                    // e.g., "closed", "open", "partially closed"
        "mcway_closure_reason": string or null,
        "mcway_alternative_text": string or null,          // any alternative viewing suggestion the answer mentions
        "mcway_source_urls": [array of URLs],
        "gorge_status": string or null,                    // e.g., "closed", "open with caution"
        "gorge_closure_reason": string or null,
        "gorge_source_urls": [array of URLs]
      },
      "half_dome": {
        "preseason_lottery_start_date": string or null,    // expected like "March 1, 2026"
        "preseason_lottery_end_date": string or null,      // expected like "March 31, 2026"
        "lottery_platform_name": string or null,           // e.g., "Recreation.gov"
        "lottery_source_urls": [array of URLs]
      }
    }

    Rules:
    - Extract only what is explicitly present in the answer text.
    - For each URLs array, include only actual URLs present in the answer.
    - If a field is missing in the answer, set it to null (or empty array for URL lists).
    """


# --------------------------------------------------------------------------- #
# Verification helpers                                                        #
# --------------------------------------------------------------------------- #
async def verify_campground(evaluator: Evaluator, parent_node, cg: Optional[CampgroundInfo]) -> None:
    node = evaluator.add_parallel(
        id="campground_identification",
        desc="Identify the correct state park campground meeting all specified requirements",
        parent=parent_node,
        critical=True  # All listed constraints are essential
    )

    # Existence checks (gate the rest)
    name_provided = bool(cg and cg.campground_name and cg.campground_name.strip())
    src_provided = bool(cg and cg.campground_source_urls and len(cg.campground_source_urls) > 0)

    evaluator.add_custom_node(
        result=name_provided,
        id="campground_name_provided",
        desc="Specific campground name is provided",
        parent=node,
        critical=True
    )
    evaluator.add_custom_node(
        result=src_provided,
        id="campground_sources_provided",
        desc="Source URL(s) provided for campground information verification",
        parent=node,
        critical=True
    )

    # Prepare safe values
    name = (cg.campground_name or "").strip()
    urls = cg.campground_source_urls if src_provided else []

    # 1) California State Parks + Big Sur area
    n1 = evaluator.add_leaf(
        id="state_park_system",
        desc="Campground is within California State Parks system in Big Sur area",
        parent=node,
        critical=True
    )
    claim1 = f"{name} is a California State Parks campground located in the Big Sur area of California."
    await evaluator.verify(
        claim=claim1,
        node=n1,
        sources=urls,
        additional_instruction="Confirm that the campground is operated by California State Parks and is in the Big Sur area."
    )

    # 2) Along Big Sur River
    n2 = evaluator.add_leaf(
        id="river_location",
        desc="Campground is located along the Big Sur River",
        parent=node,
        critical=True
    )
    claim2 = f"{name} campground is located along the Big Sur River."
    await evaluator.verify(
        claim=claim2,
        node=n2,
        sources=urls,
        additional_instruction="Look for phrases indicating the campground is on/along the Big Sur River or that the river runs through the campground."
    )

    # 3) Motorhome accommodation (28ft requirement; expect max length at least 32ft per rubric)
    n3 = evaluator.add_leaf(
        id="motorhome_accommodation",
        desc="Campground can accommodate the 28-foot motorhome (maximum motorhome length is 32 feet or greater)",
        parent=node,
        critical=True
    )
    claim3 = f"The maximum allowed motorhome/RV length at {name} is at least 32 feet, thereby accommodating a 28-foot motorhome."
    await evaluator.verify(
        claim=claim3,
        node=n3,
        sources=urls,
        additional_instruction="Confirm RV or motorhome length limits from official campground info. It's acceptable if some sites meet or exceed 32 feet."
    )

    # 4) Group size accommodation (6 adults; expect max occupancy at least 8)
    n4 = evaluator.add_leaf(
        id="group_size_accommodation",
        desc="Campground can accommodate 6 adults (maximum occupancy is 8 people or greater)",
        parent=node,
        critical=True
    )
    claim4 = f"The maximum overnight occupancy per campsite at {name} is at least 8 people, which accommodates 6 adults."
    await evaluator.verify(
        claim=claim4,
        node=n4,
        sources=urls,
        additional_instruction="Verify standard campsite occupancy limits from official sources; 8 people per site or more satisfies this condition."
    )

    # 5) Hot showers available
    n5 = evaluator.add_leaf(
        id="hot_showers",
        desc="Campground provides hot shower facilities",
        parent=node,
        critical=True
    )
    claim5 = f"{name} campground provides hot showers (e.g., token-operated hot showers)."
    await evaluator.verify(
        claim=claim5,
        node=n5,
        sources=urls,
        additional_instruction="Confirm that showers are present and hot (token-operated hot showers count as hot showers)."
    )


async def verify_reservation(evaluator: Evaluator, parent_node, rsv: Optional[ReservationInfo]) -> None:
    node = evaluator.add_parallel(
        id="reservation_information",
        desc="Correct reservation platform and advance booking requirements",
        parent=parent_node,
        critical=False  # Allow partial here while still gating critical children
    )

    # Source existence (gate others)
    src_provided = bool(rsv and rsv.reservation_website_urls)
    evaluator.add_custom_node(
        result=src_provided,
        id="reservation_sources_provided",
        desc="Source URL(s) provided for reservation information verification",
        parent=node,
        critical=True
    )

    urls = rsv.reservation_website_urls if src_provided else []
    site_name = (rsv.reservation_website_name or "").strip()

    # 1) Correct reservation website (ReserveCalifornia)
    n1 = evaluator.add_leaf(
        id="reservation_website",
        desc="Correct reservation website (ReserveCalifornia.com) is identified",
        parent=node,
        critical=True
    )
    claim1 = "ReserveCalifornia (reservecalifornia.com) is the official website for making California State Parks camping reservations."
    await evaluator.verify(
        claim=claim1,
        node=n1,
        sources=urls,
        additional_instruction="Confirm that ReserveCalifornia is the official CA State Parks reservation platform."
    )

    # 2) Advance booking period (6 months)
    n2 = evaluator.add_leaf(
        id="advance_booking_period",
        desc="Correct advance booking period (6 months in advance) is stated",
        parent=node,
        critical=True
    )
    claim2 = "California State Parks campsite reservations open six months in advance of the arrival date on ReserveCalifornia."
    await evaluator.verify(
        claim=claim2,
        node=n2,
        sources=urls,
        additional_instruction="Confirm six-month booking window policy for CA State Parks on ReserveCalifornia."
    )

    # 3) Booking time detail (8:00 AM PT) - non-critical
    n3 = evaluator.add_leaf(
        id="booking_time_detail",
        desc="Specific booking time (8:00 AM PST/PDT) is provided",
        parent=node,
        critical=False
    )
    claim3 = "New reservations on ReserveCalifornia become available at 8:00 AM Pacific Time (PST/PDT) each day."
    await evaluator.verify(
        claim=claim3,
        node=n3,
        sources=urls,
        additional_instruction="Confirm the daily opening time (8:00 AM PT) for new inventory on ReserveCalifornia."
    )


async def verify_activities(evaluator: Evaluator, parent_node, act: Optional[ActivityStatusInfo]) -> None:
    node = evaluator.add_parallel(
        id="activity_accessibility_status",
        desc="Verification of which planned activities are accessible in March 2026 with reasons for closures",
        parent=parent_node,
        critical=False
    )

    # McWay Falls Overlook Trail
    mcway_node = evaluator.add_parallel(
        id="mcway_falls_assessment",
        desc="Complete assessment of McWay Falls Overlook Trail accessibility",
        parent=node,
        critical=False
    )
    mcway_src_provided = bool(act and act.mcway_source_urls)
    evaluator.add_custom_node(
        result=mcway_src_provided,
        id="mcway_source",
        desc="Source URL provided for McWay Falls status verification",
        parent=mcway_node,
        critical=True
    )
    mcway_urls = act.mcway_source_urls if mcway_src_provided else []
    # Closed status (critical)
    n_m_closed = evaluator.add_leaf(
        id="mcway_falls_closed",
        desc="McWay Falls Overlook Trail is identified as closed in March 2026",
        parent=mcway_node,
        critical=True
    )
    claim_mcway_closed = "As of March 2026, the McWay Falls Overlook Trail (at Julia Pfeiffer Burns State Park) is closed."
    await evaluator.verify(
        claim=claim_mcway_closed,
        node=n_m_closed,
        sources=mcway_urls,
        additional_instruction="Verify closure status as of March 2026 from official park alerts or notices."
    )
    # Closure reason (non-critical)
    n_m_reason = evaluator.add_leaf(
        id="mcway_closure_reason",
        desc="Closure reason (long-term retaining wall repair) is provided",
        parent=mcway_node,
        critical=False
    )
    claim_mcway_reason = "The McWay Falls Overlook Trail closure is due to long-term retaining wall repair (or equivalent infrastructure/retaining wall damage work)."
    await evaluator.verify(
        claim=claim_mcway_reason,
        node=n_m_reason,
        sources=mcway_urls,
        additional_instruction="Accept language that clearly indicates a retaining wall repair or equivalent structural repair as the reason."
    )
    # Alternative viewing option (non-critical - about the answer content)
    n_m_alt = evaluator.add_leaf(
        id="mcway_alternative",
        desc="Alternative viewing option is mentioned",
        parent=mcway_node,
        critical=False
    )
    alt_text = (act.mcway_alternative_text or "").strip() if act else ""
    claim_mcway_alt = (
        f"The answer mentions an alternative way to view McWay Falls (e.g., highway pullouts/overlooks). "
        f"Extracted mention: '{alt_text}'."
    )
    await evaluator.verify(
        claim=claim_mcway_alt,
        node=n_m_alt,
        sources=None,
        additional_instruction="Judge based on the answer text: it should mention an alternative viewing option such as viewing from Highway 1 overlooks or pullouts."
    )

    # Big Sur River Gorge (Pfeiffer Big Sur SP)
    gorge_node = evaluator.add_parallel(
        id="river_gorge_assessment",
        desc="Complete assessment of Big Sur River Gorge accessibility",
        parent=node,
        critical=False
    )
    gorge_src_provided = bool(act and act.gorge_source_urls)
    evaluator.add_custom_node(
        result=gorge_src_provided,
        id="gorge_source",
        desc="Source URL provided for Big Sur River Gorge status verification",
        parent=gorge_node,
        critical=True
    )
    gorge_urls = act.gorge_source_urls if gorge_src_provided else []
    # Closed status (critical)
    n_g_closed = evaluator.add_leaf(
        id="river_gorge_closed",
        desc="Big Sur River Gorge is identified as closed in March 2026",
        parent=gorge_node,
        critical=True
    )
    claim_gorge_closed = "As of March 2026, access to the Big Sur River Gorge area in Pfeiffer Big Sur State Park is closed."
    await evaluator.verify(
        claim=claim_gorge_closed,
        node=n_g_closed,
        sources=gorge_urls,
        additional_instruction="Verify closure status or explicit advisories prohibiting access to the River Gorge in March 2026."
    )
    # Closure reason (non-critical)
    n_g_reason = evaluator.add_leaf(
        id="gorge_closure_reason",
        desc="Closure reason (unsafe swiftwater conditions) is provided",
        parent=gorge_node,
        critical=False
    )
    claim_gorge_reason = "The Big Sur River Gorge is closed due to unsafe swiftwater/high-water conditions that make the area hazardous."
    await evaluator.verify(
        claim=claim_gorge_reason,
        node=n_g_reason,
        sources=gorge_urls,
        additional_instruction="Accept reasons referencing unsafe high water, swiftwater hazards, or similar safety concerns tied to the river conditions."
    )


async def verify_half_dome(evaluator: Evaluator, parent_node, hd: Optional[HalfDomeLotteryInfo]) -> None:
    node = evaluator.add_parallel(
        id="half_dome_lottery_deadline",
        desc="Half Dome permit lottery application deadline information",
        parent=parent_node,
        critical=False
    )

    # Source existence
    src_provided = bool(hd and hd.lottery_source_urls)
    evaluator.add_custom_node(
        result=src_provided,
        id="half_dome_source",
        desc="Source URL provided for Half Dome lottery deadline verification",
        parent=node,
        critical=True
    )
    urls = hd.lottery_source_urls if src_provided else []

    # Start date (critical)
    n_start = evaluator.add_leaf(
        id="lottery_start_date",
        desc="Preseason lottery start date (March 1, 2026) is stated",
        parent=node,
        critical=True
    )
    claim_start = "The 2026 Half Dome day hike preseason lottery opens on March 1, 2026."
    await evaluator.verify(
        claim=claim_start,
        node=n_start,
        sources=urls,
        additional_instruction="Confirm dates from official Yosemite NP or Recreation.gov pages."
    )

    # End date (critical)
    n_end = evaluator.add_leaf(
        id="lottery_end_date",
        desc="Preseason lottery end date (March 31, 2026) is stated",
        parent=node,
        critical=True
    )
    claim_end = "The 2026 Half Dome day hike preseason lottery closes on March 31, 2026."
    await evaluator.verify(
        claim=claim_end,
        node=n_end,
        sources=urls,
        additional_instruction="Confirm dates from official Yosemite NP or Recreation.gov pages."
    )

    # Platform (non-critical)
    n_platform = evaluator.add_leaf(
        id="lottery_platform",
        desc="Application platform (Recreation.gov) is identified",
        parent=node,
        critical=False
    )
    platform = (hd.lottery_platform_name or "").strip()
    claim_platform = "Applications for the Half Dome day hike permits are submitted through Recreation.gov."
    await evaluator.verify(
        claim=claim_platform,
        node=n_platform,
        sources=urls,
        additional_instruction="Verify the application platform is Recreation.gov."
    )


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
    Evaluate an answer for the Big Sur planning task (March 2026).
    """
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

    # Top-level planning node (non-critical to allow partial credit across sections)
    plan_root = evaluator.add_parallel(
        id="root_planning_requirements",
        desc="Complete planning requirements for Big Sur camping trip in late March 2026",
        parent=root,
        critical=False
    )

    # Extraction
    extracted: PlanningExtraction = await evaluator.extract(
        prompt=prompt_extract_planning(),
        template_class=PlanningExtraction,
        extraction_name="planning_extraction"
    )

    # Build and verify subtrees
    await verify_campground(evaluator, plan_root, extracted.campground)
    await verify_reservation(evaluator, plan_root, extracted.reservation)
    await verify_activities(evaluator, plan_root, extracted.activities)
    await verify_half_dome(evaluator, plan_root, extracted.half_dome)

    return evaluator.get_summary()