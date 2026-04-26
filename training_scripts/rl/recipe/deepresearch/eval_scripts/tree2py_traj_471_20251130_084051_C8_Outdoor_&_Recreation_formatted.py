import asyncio
import logging
import re
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys

# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "yosemite_half_dome_2025_planning"
TASK_DESCRIPTION = (
    "You are planning a 2-night backpacking trip to Half Dome in Yosemite National Park for a group of 4 people in July 2025, "
    "with the goal of camping near Little Yosemite Valley and summiting Half Dome. Provide a comprehensive planning guide that includes: "
    "(1) all required permit types and how they work together, (2) the complete application process including timing windows and booking platform, "
    "(3) at least three Half Dome-eligible trailheads with their specific daily quotas, (4) mandatory camping location rules, (5) all associated fees, "
    "(6) required gear and equipment, (7) seasonal access considerations, and (8) key logistics for permit pickup and validation. "
    "For each piece of information, provide supporting evidence with reference URLs from official sources."
)

# Ground-truth quota expectations (based on rubric requirements)
EXPECTED_HAPPY_ISLES_LYV_LOTTERY = 15
EXPECTED_HAPPY_ISLES_LYV_FCFS = 10
EXPECTED_GLACIER_POINT_LYV_LOTTERY = 6
EXPECTED_GLACIER_POINT_LYV_FCFS = 4

# Official domains to be considered authoritative
OFFICIAL_DOMAINS = ["nps.gov", "recreation.gov"]


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class PermitSection(BaseModel):
    wilderness_requirement_text: Optional[str] = None
    half_dome_requirement_text: Optional[str] = None
    integration_explanation: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class ApplicationSection(BaseModel):
    platform_text: Optional[str] = None
    wilderness_lottery_text: Optional[str] = None
    wilderness_fcfs_text: Optional[str] = None
    preseason_period_text: Optional[str] = None
    preseason_one_application_text: Optional[str] = None
    daily_two_days_prior_text: Optional[str] = None
    daily_window_text: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class TrailheadQuota(BaseModel):
    name: Optional[str] = None
    lottery_quota: Optional[str] = None
    fcfs_quota: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class TrailheadsSection(BaseModel):
    trailheads: List[TrailheadQuota] = Field(default_factory=list)
    sources: List[str] = Field(default_factory=list)


class CampingSection(BaseModel):
    lyv_first_night_rule_text: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class FeesSection(BaseModel):
    half_dome_application_fee_text: Optional[str] = None
    half_dome_permit_fee_text: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class GearSection(BaseModel):
    bear_canister_required_text: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class SeasonalSection(BaseModel):
    typical_cables_season_text: Optional[str] = None
    walkup_permits_rare_text: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class LogisticsSection(BaseModel):
    wilderness_center_hours_text: Optional[str] = None
    permit_holder_presence_text: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class PlanningGuideExtraction(BaseModel):
    permit: PermitSection = PermitSection()
    application: ApplicationSection = ApplicationSection()
    trailheads: TrailheadsSection = TrailheadsSection()
    camping: CampingSection = CampingSection()
    fees: FeesSection = FeesSection()
    gear: GearSection = GearSection()
    seasonal: SeasonalSection = SeasonalSection()
    logistics: LogisticsSection = LogisticsSection()


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_planning_guide() -> str:
    return """
    Extract a structured planning guide from the answer. Capture the exact statements (verbatim or closely paraphrased) and the reference URLs specifically cited for each section. Return JSON according to the following schema:

    Sections and fields:
    1) permit:
       - wilderness_requirement_text: statement that a wilderness permit is required year-round for overnight stays in Yosemite Wilderness.
       - half_dome_requirement_text: statement that Half Dome permits are required seven days per week when cables are up.
       - integration_explanation: explanation of how wilderness permits and Half Dome permits work together for an overnight backpacking trip.
       - sources: array of official reference URLs cited for permit requirements/integration (extract only URLs explicitly present in the answer).

    2) application:
       - platform_text: statement showing permits must be obtained via Recreation.gov.
       - wilderness_lottery_text: statement that 60% of wilderness permits are available via lottery 24 weeks in advance.
       - wilderness_fcfs_text: statement that 40% of wilderness permits are available first-come-first-served 7 days in advance.
       - preseason_period_text: statement that the Half Dome preseason lottery application period is March 1–31 (Eastern Time).
       - preseason_one_application_text: statement that each person may be a permit holder or alternate on only one preseason lottery application.
       - daily_two_days_prior_text: statement that daily Half Dome lottery applications must be submitted 2 days prior.
       - daily_window_text: statement that the daily Half Dome lottery application window is midnight to 4pm Pacific time.
       - sources: array of official reference URLs cited for the above platform + timing windows.

    3) trailheads:
       - trailheads: array of objects, each with:
         * name: the trailhead name (e.g., "Happy Isles to Little Yosemite Valley", "Glacier Point to Little Yosemite Valley").
         * lottery_quota: the stated daily lottery quota for that trailhead.
         * fcfs_quota: the stated daily first-come-first-served quota for that trailhead.
         * sources: array of official reference URLs specifically supporting this trailhead's eligibility/quotas.
       - sources: array of general URLs cited that discuss trailhead eligibility/quotas.

       Include at least 3 Half Dome-eligible trailheads if the answer provides them; otherwise include as many as are mentioned. Only include items that have at least one of the two quota values stated.

    4) camping:
       - lyv_first_night_rule_text: statement that starting from Happy Isles→LYV or Glacier Point→LYV trailheads requires spending the first night at Little Yosemite Valley Campground.
       - sources: array of official reference URLs supporting the camping rule.

    5) fees:
       - half_dome_application_fee_text: statement that the Half Dome application fee is $10 per application and is non-refundable.
       - half_dome_permit_fee_text: statement that the Half Dome permit fee is $10 per person (charged when the permit is issued).
       - sources: array of official reference URLs supporting these fees.

    6) gear:
       - bear_canister_required_text: statement that bear canisters are required for food storage in Yosemite wilderness.
       - sources: array of official reference URLs supporting the gear requirement.

    7) seasonal:
       - typical_cables_season_text: statement about the typical period when Half Dome cables are up (Friday before the last Monday in May through the day after the second Monday in October).
       - walkup_permits_rare_text: statement that walk-up wilderness permits are rare during peak season (late April through mid-October).
       - sources: array of official reference URLs supporting the above.

    8) logistics:
       - wilderness_center_hours_text: statement that wilderness centers operate 8am to 5pm for permit pickup.
       - permit_holder_presence_text: statement that the permit holder or alternate must be present with the group for permit pickup/validation.
       - sources: array of official reference URLs supporting pickup hours and presence requirements.

    URL extraction rules:
    - Extract only actual URLs explicitly present in the answer text (plain or markdown).
    - Prefer official sources (e.g., nps.gov, recreation.gov) if provided; otherwise still extract any cited URLs.
    - Do not invent URLs.

    If the answer does not provide a particular statement, set that field to null. If there are no URLs for a section, return an empty array for sources.
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def is_official_url(url: str) -> bool:
    if not url:
        return False
    u = url.lower()
    return any(domain in u for domain in OFFICIAL_DOMAINS)


def has_official_sources(urls: List[str]) -> bool:
    return any(is_official_url(u) for u in urls)


def extract_first_int(text: Optional[str]) -> Optional[int]:
    if not text:
        return None
    m = re.search(r"\b(\d{1,3})\b", text)
    if m:
        try:
            return int(m.group(1))
        except Exception:
            return None
    return None


def find_trailhead(trailheads: List[TrailheadQuota], keywords: List[str]) -> Optional[TrailheadQuota]:
    """
    Find a trailhead whose name contains all keywords (case-insensitive).
    """
    for th in trailheads:
        name = (th.name or "").lower()
        if all(k in name for k in keywords):
            return th
    return None


def union_sources(*lists: List[str]) -> List[str]:
    seen = set()
    result: List[str] = []
    for lst in lists:
        for u in lst:
            if u not in seen:
                seen.add(u)
                result.append(u)
    return result


# --------------------------------------------------------------------------- #
# Verification builders                                                       #
# --------------------------------------------------------------------------- #
async def build_permit_section(evaluator: Evaluator, parent, ex: PlanningGuideExtraction) -> None:
    sec_node = evaluator.add_parallel(
        id="PermitTypesRequired",
        desc="Identify all required permit types and explain how they work together (with official evidence)",
        parent=parent,
        critical=True
    )

    # Reference URL existence (official) – gate others
    evaluator.add_custom_node(
        result=has_official_sources(ex.permit.sources) and len(ex.permit.sources) > 0,
        id="PermitTypesRequired_ReferenceURL",
        desc="Provides supporting reference URL(s) from official sources for the permit requirements/integration",
        parent=sec_node,
        critical=True
    )

    # Wilderness permit requirement
    w_node = evaluator.add_leaf(
        id="WildernessPermitRequirement",
        desc="States that a wilderness permit is required year-round for any overnight stay in Yosemite Wilderness",
        parent=sec_node,
        critical=True
    )
    w_claim = ex.permit.wilderness_requirement_text or "A wilderness permit is required year-round for any overnight stay in the Yosemite Wilderness."
    await evaluator.verify(
        claim=w_claim,
        node=w_node,
        sources=ex.permit.sources,
        additional_instruction="Only accept support from official NPS/Recreation.gov sources."
    )

    # Half Dome permit requirement
    hd_node = evaluator.add_leaf(
        id="HalfDomePermitRequirement",
        desc="States that Half Dome permits are required seven days per week when cables are up",
        parent=sec_node,
        critical=True
    )
    hd_claim = ex.permit.half_dome_requirement_text or "Half Dome permits are required seven days per week when the cables are up."
    await evaluator.verify(
        claim=hd_claim,
        node=hd_node,
        sources=ex.permit.sources,
        additional_instruction="Only accept support from official NPS/Recreation.gov sources."
    )

    # Integration explanation
    integ_node = evaluator.add_leaf(
        id="PermitIntegrationExplanation",
        desc="Explains how wilderness permits and Half Dome permits relate/are used together for this trip plan",
        parent=sec_node,
        critical=True
    )
    integ_claim = ex.permit.integration_explanation or (
        "Overnight backpackers must have a wilderness permit for the trailhead and dates, and a separate Half Dome permit to ascend the cables; "
        "the Half Dome permit is used together with the wilderness permit on the permitted itinerary."
    )
    await evaluator.verify(
        claim=integ_claim,
        node=integ_node,
        sources=ex.permit.sources,
        additional_instruction="Confirm that Half Dome permits for backpackers are attached to a valid wilderness permit itinerary; use official sources."
    )


async def build_application_section(evaluator: Evaluator, parent, ex: PlanningGuideExtraction) -> None:
    sec_node = evaluator.add_parallel(
        id="ApplicationProcess_TimingAndPlatform",
        desc="Complete application process including timing windows and booking platform (with official evidence)",
        parent=parent,
        critical=True
    )

    # Reference URL existence (official) – gate others
    evaluator.add_custom_node(
        result=has_official_sources(ex.application.sources) and len(ex.application.sources) > 0,
        id="ApplicationProcess_ReferenceURL",
        desc="Provides supporting reference URL(s) from official sources for platform + timing windows above",
        parent=sec_node,
        critical=True
    )

    # Booking platform
    plat_node = evaluator.add_leaf(
        id="BookingPlatform_RecreationGov",
        desc="States that permits must be obtained through Recreation.gov",
        parent=sec_node,
        critical=True
    )
    plat_claim = ex.application.platform_text or "Permits must be obtained via Recreation.gov."
    await evaluator.verify(
        claim=plat_claim,
        node=plat_node,
        sources=ex.application.sources,
        additional_instruction="Verify the permit booking platform is Recreation.gov using official sources."
    )

    # Wilderness lottery window
    wl_node = evaluator.add_leaf(
        id="WildernessPermitLotteryWindow",
        desc="States that 60% of wilderness permits are available via lottery 24 weeks in advance",
        parent=sec_node,
        critical=True
    )
    wl_claim = ex.application.wilderness_lottery_text or "60% of wilderness permits are allocated via a lottery approximately 24 weeks in advance."
    await evaluator.verify(
        claim=wl_claim,
        node=wl_node,
        sources=ex.application.sources,
        additional_instruction="Confirm the lottery allocation percentage and the 24-weeks timing using official sources."
    )

    # Wilderness FCFS window
    wf_node = evaluator.add_leaf(
        id="WildernessPermitFCFSWindow",
        desc="States that 40% of wilderness permits are available first-come-first-served 7 days in advance",
        parent=sec_node,
        critical=True
    )
    wf_claim = ex.application.wilderness_fcfs_text or "40% of wilderness permits are available first-come-first-served starting 7 days in advance."
    await evaluator.verify(
        claim=wf_claim,
        node=wf_node,
        sources=ex.application.sources,
        additional_instruction="Confirm the FCFS percentage and 7-day timing using official sources."
    )

    # Half Dome preseason application period
    pre_node = evaluator.add_leaf(
        id="HalfDomePreseasonLotteryApplicationPeriod",
        desc="States preseason Half Dome lottery application period is March 1–31 (Eastern Time)",
        parent=sec_node,
        critical=True
    )
    pre_claim = ex.application.preseason_period_text or "The Half Dome preseason lottery application period is March 1–31 (Eastern Time)."
    await evaluator.verify(
        claim=pre_claim,
        node=pre_node,
        sources=ex.application.sources,
        additional_instruction="Confirm date range and time zone for the preseason lottery from official sources."
    )

    # One application rule
    one_node = evaluator.add_leaf(
        id="HalfDomePreseasonLotteryOneApplicationRule",
        desc="States that each person may be a permit holder or alternate on only ONE preseason lottery application",
        parent=sec_node,
        critical=True
    )
    one_claim = ex.application.preseason_one_application_text or "Each person may be listed as a permit holder or alternate on only one preseason lottery application."
    await evaluator.verify(
        claim=one_claim,
        node=one_node,
        sources=ex.application.sources,
        additional_instruction="Confirm the one-application rule using official sources."
    )

    # Daily lottery timing: two days prior
    dl_node = evaluator.add_leaf(
        id="HalfDomeDailyLotteryTwoDaysPrior",
        desc="States daily Half Dome lottery applications must be submitted 2 days prior",
        parent=sec_node,
        critical=True
    )
    dl_claim = ex.application.daily_two_days_prior_text or "Daily Half Dome lottery applications must be submitted two days prior to the intended hike date."
    await evaluator.verify(
        claim=dl_claim,
        node=dl_node,
        sources=ex.application.sources,
        additional_instruction="Confirm the daily lottery submission timing using official sources."
    )

    # Daily lottery window: midnight–4pm PT
    dw_node = evaluator.add_leaf(
        id="HalfDomeDailyLotteryTimeWindow",
        desc="States daily Half Dome lottery application window is midnight to 4pm Pacific time",
        parent=sec_node,
        critical=True
    )
    dw_claim = ex.application.daily_window_text or "The daily Half Dome lottery application window is from midnight to 4pm Pacific time."
    await evaluator.verify(
        claim=dw_claim,
        node=dw_node,
        sources=ex.application.sources,
        additional_instruction="Confirm the daily lottery time window using official sources."
    )


async def build_trailheads_section(evaluator: Evaluator, parent, ex: PlanningGuideExtraction) -> None:
    sec_node = evaluator.add_parallel(
        id="EligibleTrailheadsAndQuotas",
        desc="Provide at least three Half Dome-eligible trailheads with specific daily quotas (with official evidence)",
        parent=parent,
        critical=True
    )

    trailheads = ex.trailheads.trailheads or []

    # Identify specific required trailheads
    hi = find_trailhead(trailheads, ["happy isles", "little yosemite valley"])
    gp = find_trailhead(trailheads, ["glacier point", "little yosemite valley"])

    # Third trailhead: any other Half Dome-eligible trailhead with both quotas
    def has_both_quotas(th: TrailheadQuota) -> bool:
        return bool((th.lottery_quota or "").strip()) and bool((th.fcfs_quota or "").strip())

    third: Optional[TrailheadQuota] = None
    for th in trailheads:
        if th is hi or th is gp:
            continue
        if has_both_quotas(th):
            third = th
            break

    # Reference URL existence (official) – across all provided trailheads
    all_tr_sources = union_sources(*(th.sources for th in trailheads), ex.trailheads.sources)
    evaluator.add_custom_node(
        result=has_official_sources(all_tr_sources) and len(all_tr_sources) > 0,
        id="EligibleTrailheads_ReferenceURL",
        desc="Provides supporting reference URL(s) from official sources for all listed trailhead eligibility/quotas",
        parent=sec_node,
        critical=True
    )

    # Happy Isles → LYV quotas group (sequential: existence → expected numbers → URL support)
    hi_group = evaluator.add_sequential(
        id="HappyIslesToLYVTrailheadQuotas",
        desc="Includes Happy Isles to Little Yosemite Valley trailhead and its quotas (15 lottery, 10 FCFS)",
        parent=sec_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=hi is not None,
        id="HI_LYV_exists",
        desc="Happy Isles to Little Yosemite Valley trailhead is present in the answer",
        parent=hi_group,
        critical=True
    )
    hi_lq = extract_first_int(hi.lottery_quota if hi else None)
    hi_fq = extract_first_int(hi.fcfs_quota if hi else None)
    evaluator.add_custom_node(
        result=(hi_lq == EXPECTED_HAPPY_ISLES_LYV_LOTTERY and hi_fq == EXPECTED_HAPPY_ISLES_LYV_FCFS),
        id="HI_LYV_quota_match_expected",
        desc=f"Happy Isles → LYV quotas match expected ({EXPECTED_HAPPY_ISLES_LYV_LOTTERY} lottery, {EXPECTED_HAPPY_ISLES_LYV_FCFS} FCFS)",
        parent=hi_group,
        critical=True
    )
    hi_verify = evaluator.add_leaf(
        id="HI_LYV_quota_supported",
        desc="Official sources support the stated Happy Isles → LYV quotas",
        parent=hi_group,
        critical=True
    )
    hi_claim = (
        f"The 'Happy Isles to Little Yosemite Valley' trailhead has daily quotas of "
        f"{EXPECTED_HAPPY_ISLES_LYV_LOTTERY} lottery and {EXPECTED_HAPPY_ISLES_LYV_FCFS} first-come-first-served."
    )
    await evaluator.verify(
        claim=hi_claim,
        node=hi_verify,
        sources=hi.sources if hi else ex.trailheads.sources,
        additional_instruction="Verify quotas on official NPS/Recreation.gov sources; allow minor naming variants."
    )

    # Glacier Point → LYV quotas group
    gp_group = evaluator.add_sequential(
        id="GlacierPointToLYVTrailheadQuotas",
        desc="Includes Glacier Point to Little Yosemite Valley trailhead and its quotas (6 lottery, 4 FCFS)",
        parent=sec_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=gp is not None,
        id="GP_LYV_exists",
        desc="Glacier Point to Little Yosemite Valley trailhead is present in the answer",
        parent=gp_group,
        critical=True
    )
    gp_lq = extract_first_int(gp.lottery_quota if gp else None)
    gp_fq = extract_first_int(gp.fcfs_quota if gp else None)
    evaluator.add_custom_node(
        result=(gp_lq == EXPECTED_GLACIER_POINT_LYV_LOTTERY and gp_fq == EXPECTED_GLACIER_POINT_LYV_FCFS),
        id="GP_LYV_quota_match_expected",
        desc=f"Glacier Point → LYV quotas match expected ({EXPECTED_GLACIER_POINT_LYV_LOTTERY} lottery, {EXPECTED_GLACIER_POINT_LYV_FCFS} FCFS)",
        parent=gp_group,
        critical=True
    )
    gp_verify = evaluator.add_leaf(
        id="GP_LYV_quota_supported",
        desc="Official sources support the stated Glacier Point → LYV quotas",
        parent=gp_group,
        critical=True
    )
    gp_claim = (
        f"The 'Glacier Point to Little Yosemite Valley' trailhead has daily quotas of "
        f"{EXPECTED_GLACIER_POINT_LYV_LOTTERY} lottery and {EXPECTED_GLACIER_POINT_LYV_FCFS} first-come-first-served."
    )
    await evaluator.verify(
        claim=gp_claim,
        node=gp_verify,
        sources=gp.sources if gp else ex.trailheads.sources,
        additional_instruction="Verify quotas on official NPS/Recreation.gov sources; allow minor naming variants."
    )

    # Third eligible trailhead provided group
    third_group = evaluator.add_sequential(
        id="ThirdEligibleTrailheadProvided",
        desc="Includes at least one additional (third) Half Dome-eligible trailhead with both lottery and FCFS quotas stated",
        parent=sec_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=third is not None,
        id="ThirdTrailhead_exists",
        desc="A third Half Dome-eligible trailhead is provided",
        parent=third_group,
        critical=True
    )
    evaluator.add_custom_node(
        result=(third is not None and has_both_quotas(third)),
        id="ThirdTrailhead_both_quotas_stated",
        desc="Third trailhead has both lottery and FCFS quotas stated in the answer",
        parent=third_group,
        critical=True
    )
    third_verify = evaluator.add_leaf(
        id="ThirdTrailhead_quota_supported",
        desc="Official sources support the stated quotas for the third trailhead",
        parent=third_group,
        critical=True
    )
    third_claim = (
        f"The trailhead '{third.name}' has daily quotas of {third.lottery_quota} lottery and {third.fcfs_quota} first-come-first-served."
        if third else "A third Half Dome-eligible trailhead with both quotas exists."
    )
    await evaluator.verify(
        claim=third_claim,
        node=third_verify,
        sources=third.sources if third else ex.trailheads.sources,
        additional_instruction="Verify quotas on official NPS/Recreation.gov sources; allow minor naming variants."
    )


async def build_camping_section(evaluator: Evaluator, parent, ex: PlanningGuideExtraction) -> None:
    sec_node = evaluator.add_parallel(
        id="MandatoryCampingLocationRules",
        desc="Mandatory camping location rules (with official evidence)",
        parent=parent,
        critical=True
    )

    evaluator.add_custom_node(
        result=has_official_sources(ex.camping.sources) and len(ex.camping.sources) > 0,
        id="CampingRules_ReferenceURL",
        desc="Provides supporting reference URL(s) from official sources for the camping rule",
        parent=sec_node,
        critical=True
    )

    lyv_node = evaluator.add_leaf(
        id="LYVFirstNightRule",
        desc="States that if starting from Happy Isles to LYV or Glacier Point to LYV trailheads, the first night must be spent at Little Yosemite Valley Campground",
        parent=sec_node,
        critical=True
    )
    lyv_claim = ex.camping.lyv_first_night_rule_text or (
        "If starting from Happy Isles→Little Yosemite Valley or Glacier Point→Little Yosemite Valley trailheads, the first night must be spent at Little Yosemite Valley Campground."
    )
    await evaluator.verify(
        claim=lyv_claim,
        node=lyv_node,
        sources=ex.camping.sources,
        additional_instruction="Confirm camping location restrictions using official NPS sources."
    )


async def build_fees_section(evaluator: Evaluator, parent, ex: PlanningGuideExtraction) -> None:
    sec_node = evaluator.add_parallel(
        id="Fees",
        desc="All associated fees, including those specified in constraints (with official evidence)",
        parent=parent,
        critical=True
    )

    evaluator.add_custom_node(
        result=has_official_sources(ex.fees.sources) and len(ex.fees.sources) > 0,
        id="Fees_ReferenceURL",
        desc="Provides supporting reference URL(s) from official sources for the fees stated",
        parent=sec_node,
        critical=True
    )

    # Application fee group: existence → supported
    app_group = evaluator.add_sequential(
        id="HalfDomeApplicationFee",
        desc="States the Half Dome application fee is $10 per application and is non-refundable",
        parent=sec_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=(ex.fees.half_dome_application_fee_text or "").lower().count("$10") > 0 and "non-refundable" in (ex.fees.half_dome_application_fee_text or "").lower(),
        id="HalfDomeApplicationFee_stated",
        desc="Answer explicitly states $10 application fee and non-refundable",
        parent=app_group,
        critical=True
    )
    app_verify = evaluator.add_leaf(
        id="HalfDomeApplicationFee_supported",
        desc="Official sources support the stated Half Dome application fee and non-refundable policy",
        parent=app_group,
        critical=True
    )
    app_claim = ex.fees.half_dome_application_fee_text or "The Half Dome application fee is $10 per application and is non-refundable."
    await evaluator.verify(
        claim=app_claim,
        node=app_verify,
        sources=ex.fees.sources,
        additional_instruction="Confirm fee amount and non-refundable policy using official Recreation.gov/NPS sources."
    )

    # Permit fee group: existence → supported
    permit_group = evaluator.add_sequential(
        id="HalfDomePermitFee",
        desc="States the Half Dome permit fee is $10 per person (charged when permit is issued)",
        parent=sec_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=(ex.fees.half_dome_permit_fee_text or "").lower().count("$10") > 0 and "per person" in (ex.fees.half_dome_permit_fee_text or "").lower(),
        id="HalfDomePermitFee_stated",
        desc="Answer explicitly states $10 per person is charged when the permit is issued",
        parent=permit_group,
        critical=True
    )
    permit_verify = evaluator.add_leaf(
        id="HalfDomePermitFee_supported",
        desc="Official sources support the stated Half Dome permit per-person fee and timing",
        parent=permit_group,
        critical=True
    )
    permit_claim = ex.fees.half_dome_permit_fee_text or "The Half Dome permit fee is $10 per person and is charged when the permit is issued."
    await evaluator.verify(
        claim=permit_claim,
        node=permit_verify,
        sources=ex.fees.sources,
        additional_instruction="Confirm per-person fee and charge timing using official Recreation.gov/NPS sources."
    )


async def build_gear_section(evaluator: Evaluator, parent, ex: PlanningGuideExtraction) -> None:
    sec_node = evaluator.add_parallel(
        id="GearAndEquipment",
        desc="Required gear/equipment requirements grounded in constraints (with official evidence)",
        parent=parent,
        critical=True
    )

    evaluator.add_custom_node(
        result=has_official_sources(ex.gear.sources) and len(ex.gear.sources) > 0,
        id="Gear_ReferenceURL",
        desc="Provides supporting reference URL(s) from official sources for bear canister requirement",
        parent=sec_node,
        critical=True
    )

    bear_node = evaluator.add_leaf(
        id="BearCanisterRequired",
        desc="States that bear canisters are required for food storage in Yosemite wilderness",
        parent=sec_node,
        critical=True
    )
    bear_claim = ex.gear.bear_canister_required_text or "Bear canisters are required for food storage in Yosemite wilderness."
    await evaluator.verify(
        claim=bear_claim,
        node=bear_node,
        sources=ex.gear.sources,
        additional_instruction="Confirm bear canister requirements using official NPS sources."
    )


async def build_seasonal_section(evaluator: Evaluator, parent, ex: PlanningGuideExtraction) -> None:
    sec_node = evaluator.add_parallel(
        id="SeasonalAccessConsiderations",
        desc="Seasonal access considerations grounded in constraints (with official evidence)",
        parent=parent,
        critical=True
    )

    evaluator.add_custom_node(
        result=has_official_sources(ex.seasonal.sources) and len(ex.seasonal.sources) > 0,
        id="Seasonal_ReferenceURL",
        desc="Provides supporting reference URL(s) from official sources for the seasonal considerations above",
        parent=sec_node,
        critical=True
    )

    cables_node = evaluator.add_leaf(
        id="TypicalCableSeason",
        desc="States the typical period when Half Dome cables are up (Friday before last Monday in May through day after second Monday in October)",
        parent=sec_node,
        critical=True
    )
    cables_claim = ex.seasonal.typical_cables_season_text or (
        "Typically, Half Dome cables are up from the Friday before the last Monday in May through the day after the second Monday in October."
    )
    await evaluator.verify(
        claim=cables_claim,
        node=cables_node,
        sources=ex.seasonal.sources,
        additional_instruction="Confirm typical cable season dates using official NPS Half Dome pages."
    )

    walkup_node = evaluator.add_leaf(
        id="WalkUpPermitsRarePeakSeason",
        desc="States that walk-up wilderness permits are rare during peak season (late April through mid-October)",
        parent=sec_node,
        critical=True
    )
    walkup_claim = ex.seasonal.walkup_permits_rare_text or "Walk-up wilderness permits are rare during peak season (late April through mid-October)."
    await evaluator.verify(
        claim=walkup_claim,
        node=walkup_node,
        sources=ex.seasonal.sources,
        additional_instruction="Confirm walk-up permit availability realities during peak season using official sources."
    )


async def build_logistics_section(evaluator: Evaluator, parent, ex: PlanningGuideExtraction) -> None:
    sec_node = evaluator.add_parallel(
        id="PermitPickupAndValidationLogistics",
        desc="Key logistics for permit pickup and validation grounded in constraints (with official evidence)",
        parent=parent,
        critical=True
    )

    evaluator.add_custom_node(
        result=has_official_sources(ex.logistics.sources) and len(ex.logistics.sources) > 0,
        id="Logistics_ReferenceURL",
        desc="Provides supporting reference URL(s) from official sources for pickup hours and validation/presence requirements",
        parent=sec_node,
        critical=True
    )

    hours_node = evaluator.add_leaf(
        id="WildernessCenterHours",
        desc="States that wilderness centers operate 8am to 5pm for permit pickup",
        parent=sec_node,
        critical=True
    )
    hours_claim = ex.logistics.wilderness_center_hours_text or "Wilderness centers operate from 8am to 5pm for permit pickup."
    await evaluator.verify(
        claim=hours_claim,
        node=hours_node,
        sources=ex.logistics.sources,
        additional_instruction="Confirm permit pickup hours using official NPS Yosemite Wilderness Center information."
    )

    presence_node = evaluator.add_leaf(
        id="PermitHolderOrAlternatePresence",
        desc="States that the permit holder or alternate must be present with the group",
        parent=sec_node,
        critical=True
    )
    presence_claim = ex.logistics.permit_holder_presence_text or "The permit holder or alternate must be present with the group for permit pickup/validation."
    await evaluator.verify(
        claim=presence_claim,
        node=presence_node,
        sources=ex.logistics.sources,
        additional_instruction="Confirm permit pickup identity requirements using official sources."
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
    Evaluate a comprehensive Yosemite Half Dome backpacking planning guide answer.

    Returns a structured summary with verification tree and final score.
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

    # Extract structured info from the answer
    ex: PlanningGuideExtraction = await evaluator.extract(
        prompt=prompt_extract_planning_guide(),
        template_class=PlanningGuideExtraction,
        extraction_name="planning_guide_extraction",
    )

    # Create top-level critical node representing the whole planning guide
    guide_node = evaluator.add_parallel(
        id="HalfDomeBackpackingPlanningGuide",
        desc="Comprehensive planning guide matching the question + constraints, with official reference URLs supporting each required information area",
        parent=root,
        critical=True,
    )

    # Build sub-sections under the critical top-level node
    await build_permit_section(evaluator, guide_node, ex)
    await build_application_section(evaluator, guide_node, ex)
    await build_trailheads_section(evaluator, guide_node, ex)
    await build_camping_section(evaluator, guide_node, ex)
    await build_fees_section(evaluator, guide_node, ex)
    await build_gear_section(evaluator, guide_node, ex)
    await build_seasonal_section(evaluator, guide_node, ex)
    await build_logistics_section(evaluator, guide_node, ex)

    # Optionally record ground truth expectations that were explicitly specified in rubric for quotas
    evaluator.add_ground_truth({
        "expected_trailhead_quotas": {
            "Happy Isles → Little Yosemite Valley": {
                "lottery": EXPECTED_HAPPY_ISLES_LYV_LOTTERY,
                "fcfs": EXPECTED_HAPPY_ISLES_LYV_FCFS,
            },
            "Glacier Point → Little Yosemite Valley": {
                "lottery": EXPECTED_GLACIER_POINT_LYV_LOTTERY,
                "fcfs": EXPECTED_GLACIER_POINT_LYV_FCFS,
            },
        },
        "official_domains": OFFICIAL_DOMAINS,
    }, gt_type="rubric_expectations")

    return evaluator.get_summary()