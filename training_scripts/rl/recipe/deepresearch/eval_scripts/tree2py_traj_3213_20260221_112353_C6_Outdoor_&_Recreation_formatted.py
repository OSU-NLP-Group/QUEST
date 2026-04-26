import asyncio
import logging
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "rmnp_wilderness_plan"
TASK_DESCRIPTION = (
    "Plan a 5-night wilderness camping trip to Rocky Mountain National Park for June 2026 for a group of 9 U.S. residents, "
    "including 2 wheelchair users who require accessible facilities. Provide a complete trip plan that includes: "
    "(1) Site Selection Strategy: Explain how you will accommodate the group size of 9 people and meet the accessibility requirements for the 2 wheelchair users, "
    "citing specific Rocky Mountain National Park regulations and facilities. "
    "(2) Total Wilderness Permit Cost: Calculate the exact wilderness permit fee for this trip during June 2026. "
    "(3) Required Equipment: Identify all mandatory equipment requirements specific to Rocky Mountain National Park wilderness camping during this time period, "
    "including food storage requirements and placement rules. "
    "(4) Reservation Booking Procedure: Describe the complete procedure for booking wilderness camping reservations for this trip, including when reservations open, "
    "which platform(s) to use, advance booking requirements, and any special procedures for accessible sites. "
    "Ensure your answer addresses all Rocky Mountain National Park regulations for group size, accessibility, stay limits, equipment, and reservation procedures. "
    "Provide reference URLs for all key requirements."
)


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class TripScope(BaseModel):
    timing_text: Optional[str] = None  # e.g., "June 2026"
    month_year: Optional[str] = None
    duration_nights_text: Optional[str] = None  # e.g., "5 nights"
    nights_count: Optional[str] = None
    group_size_text: Optional[str] = None  # e.g., "9 people"
    group_size_number: Optional[str] = None
    wheelchair_users_text: Optional[str] = None  # e.g., "2 wheelchair users"
    wheelchair_users_count: Optional[str] = None


class SiteSelection(BaseModel):
    strategy_text: Optional[str] = None  # describes how group size + accessibility is handled
    uses_group_site: Optional[bool] = None
    splits_groups: Optional[bool] = None
    accessible_site_name: Optional[str] = None
    accessible_site_urls: List[str] = Field(default_factory=list)
    group_size_rule_urls: List[str] = Field(default_factory=list)
    stay_limit_urls: List[str] = Field(default_factory=list)
    capacity_text: Optional[str] = None  # any capacity statement
    capacity_number: Optional[str] = None
    accessibility_logistics_distance_text: Optional[str] = None  # distance from trailhead/parking
    accessibility_logistics_features_text: Optional[str] = None  # surface/grade/access features


class PermitCost(BaseModel):
    fee_rule_text: Optional[str] = None  # e.g., "$36 per trip (May 1–Oct 31)"
    total_cost_text: Optional[str] = None  # e.g., "$36 total"
    permit_fee_urls: List[str] = Field(default_factory=list)


class Equipment(BaseModel):
    bear_canister_required_text: Optional[str] = None
    placement_rule_text: Optional[str] = None  # e.g., "200 feet from camp & water"
    bear_canister_urls: List[str] = Field(default_factory=list)


class ReservationProcedure(BaseModel):
    reservation_window_text: Optional[str] = None  # e.g., "opens March 1 at 8:00 AM MST"
    platform_text: Optional[str] = None  # e.g., "Recreation.gov"
    advance_booking_text: Optional[str] = None  # e.g., "at least 3 days before start"
    accessible_reservation_text: Optional[str] = None  # e.g., "call Wilderness Office"
    reservation_rules_urls: List[str] = Field(default_factory=list)
    accessible_reservation_urls: List[str] = Field(default_factory=list)


class OtherConstraints(BaseModel):
    pets_prohibited_text: Optional[str] = None
    pet_rules_urls: List[str] = Field(default_factory=list)


class RMNPPlanExtraction(BaseModel):
    trip_scope: Optional[TripScope] = None
    site_selection: Optional[SiteSelection] = None
    permit_cost: Optional[PermitCost] = None
    equipment: Optional[Equipment] = None
    reservation: Optional[ReservationProcedure] = None
    other_constraints: Optional[OtherConstraints] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_plan() -> str:
    return """
    Extract structured information from the answer regarding the RMNP wilderness trip plan. Only extract details explicitly present in the answer.
    Organize the result into these sections and fields:

    trip_scope:
      - timing_text: exact phrasing indicating trip timing (e.g., "June 2026")
      - month_year: month + year if explicitly stated (e.g., "June 2026")
      - duration_nights_text: phrasing indicating the number of wilderness nights (e.g., "5 nights")
      - nights_count: the numeric count if stated (e.g., "5")
      - group_size_text: phrasing indicating group size (e.g., "group of 9")
      - group_size_number: numeric count if stated (e.g., "9")
      - wheelchair_users_text: phrasing indicating wheelchair users (e.g., "2 wheelchair users")
      - wheelchair_users_count: numeric count if stated (e.g., "2")

    site_selection:
      - strategy_text: how the plan addresses group size + accessibility (e.g., "use group wilderness site (8–12)" or "split into smaller groups 1+ mile apart")
      - uses_group_site: true/false if explicitly stated
      - splits_groups: true/false if explicitly stated
      - accessible_site_name: the named accessible wilderness campsite/facility if any (e.g., "Sprague Lake Accessible Wilderness Campsite")
      - accessible_site_urls: list of URLs provided that describe/confirm accessibility for the selected site/facility
      - group_size_rule_urls: list of URLs that support RMNP group size rules (e.g., group site capacity 8–12, or splitting groups ≥1 mile apart)
      - stay_limit_urls: list of URLs that support stay limits (e.g., max seasonal nights, max consecutive nights per site)
      - capacity_text: any explicit capacity statement the answer provides for the selected option (e.g., "accommodates 9")
      - capacity_number: numeric capacity if provided
      - accessibility_logistics_distance_text: any stated distance from parking/trailhead
      - accessibility_logistics_features_text: any stated wheelchair-accessible features (surface, grade, boardwalk, etc.)

    permit_cost:
      - fee_rule_text: fee rule used (e.g., "$36 per trip, May 1–Oct 31")
      - total_cost_text: computed total wilderness permit cost (e.g., "$36 total")
      - permit_fee_urls: list of URLs supporting RMNP wilderness permit fees

    equipment:
      - bear_canister_required_text: statement about requirement timing / elevation
      - placement_rule_text: statement about placement distance (e.g., "200 feet from camp and water sources")
      - bear_canister_urls: list of URLs supporting bear canister requirement and placement rules

    reservation:
      - reservation_window_text: statement for opening time (e.g., "March 1 at 8:00 AM MST")
      - platform_text: platform used (e.g., "Recreation.gov")
      - advance_booking_text: lead-time rule (e.g., "at least 3 days before start")
      - accessible_reservation_text: special procedure for accessible site (e.g., "call Wilderness Office")
      - reservation_rules_urls: list of URLs supporting opening time, platform usage, and lead-time
      - accessible_reservation_urls: list of URLs supporting accessible-site reservation procedure (e.g., phone reservations)

    other_constraints:
      - pets_prohibited_text: statement that pets are prohibited in RMNP wilderness areas
      - pet_rules_urls: list of URLs supporting pets prohibition

    Important:
    - Extract only URLs explicitly mentioned in the answer. Include full URLs with protocol.
    - If a field is missing, set it to null (for strings/booleans) or an empty list (for URLs).
    """


# --------------------------------------------------------------------------- #
# Verification helpers                                                        #
# --------------------------------------------------------------------------- #
async def verify_trip_scope(evaluator: Evaluator, parent_node, ext: RMNPPlanExtraction) -> None:
    node = evaluator.add_parallel(
        id="Trip_Scope",
        desc="Verify the plan matches the requested trip scope (timing, duration, and group composition).",
        parent=parent_node,
        critical=True,
    )

    # Timing: June 2026
    timing_node = evaluator.add_leaf(
        id="Timing_Is_June_2026",
        desc="Plan specifies the trip occurs in June 2026.",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim="The answer explicitly states that the trip occurs in June 2026.",
        node=timing_node,
        additional_instruction="Judge based on the answer content only. Pass if 'June 2026' is clearly indicated.",
    )

    # Duration: 5 nights
    duration_node = evaluator.add_leaf(
        id="Duration_Is_5_Nights",
        desc="Plan specifies a 5-night wilderness camping itinerary (or explicitly states 5 wilderness nights).",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim="The answer explicitly states that the wilderness camping portion is 5 nights.",
        node=duration_node,
        additional_instruction="Judge based on the answer content only. Pass if '5 nights' or equivalent is clearly indicated.",
    )

    # Group composition: 9 total including 2 wheelchair users
    group_node = evaluator.add_leaf(
        id="Group_Composition_Matches",
        desc="Plan accounts for a group of 9 people including 2 wheelchair users requiring accessible facilities.",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim="The answer clearly accounts for a group of nine people including two wheelchair users who require accessible facilities.",
        node=group_node,
        additional_instruction="Judge based on the answer content only. Minor phrasing variations are acceptable.",
    )


async def verify_site_selection(evaluator: Evaluator, parent_node, ext: RMNPPlanExtraction) -> None:
    s = ext.site_selection or SiteSelection()

    node = evaluator.add_parallel(
        id="Site_Selection_Strategy",
        desc="Verify site selection addresses group size + accessibility and aligns with stay-limit constraints, with URLs.",
        parent=parent_node,
        critical=True,
    )

    # Group size compliance statement in plan (content check)
    group_compliance_node = evaluator.add_leaf(
        id="Group_Size_Compliance",
        desc="Plan explains compliance for a 9-person group: uses a designated group wilderness site (8–12) OR splits into smaller groups camping at least 1 mile apart.",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim=("The plan explains how a nine-person group will comply with RMNP wilderness group-size rules by either "
               "using a designated group wilderness site (accommodating about 8–12 people) or by splitting into smaller "
               "groups camping at least one mile apart."),
        node=group_compliance_node,
        additional_instruction="Judge based on the answer content only; pass if either approach is clearly described.",
    )

    # URL presence for group size rule (existence gate)
    gs_url_present = evaluator.add_custom_node(
        result=bool(s.group_size_rule_urls),
        id="Group_Size_Rule_URL_Provided",
        desc="At least one URL is provided for the RMNP group-size rule(s).",
        parent=node,
        critical=True,
    )

    # URL-backed support for group size rule
    gs_rule_url_node = evaluator.add_leaf(
        id="URL_For_Group_Size_Rule",
        desc="Provides a reference URL supporting the group-size rule(s) used (group sites and/or splitting ≥1 mile apart).",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim=("RMNP wilderness regulations allow designated group wilderness sites (around 8–12 capacity) for larger parties; "
               "otherwise, parties must split into smaller groups that camp at least one mile apart."),
        node=gs_rule_url_node,
        sources=s.group_size_rule_urls,
        additional_instruction="Verify this rule content is supported by the provided RMNP source(s).",
        extra_prerequisites=[gs_url_present],
    )

    # Accessible site identified and supported
    acc_url_present = evaluator.add_custom_node(
        result=bool(s.accessible_site_urls),
        id="Accessible_Site_URL_Provided",
        desc="At least one URL is provided for the selected accessible campsite/facility.",
        parent=node,
        critical=True,
    )

    accessible_site_node = evaluator.add_leaf(
        id="Accessible_Site_Or_Facility_Identified",
        desc="Plan identifies at least one wheelchair-accessible wilderness campsite/facility in RMNP for the 2 wheelchair users.",
        parent=node,
        critical=True,
    )
    accessible_name = s.accessible_site_name or "the selected accessible site/facility"
    await evaluator.verify(
        claim=f"The identified option ({accessible_name}) is a wheelchair-accessible wilderness campsite or facility in Rocky Mountain National Park.",
        node=accessible_site_node,
        sources=s.accessible_site_urls,
        additional_instruction="Verify that the page(s) explicitly indicate accessibility features or wheelchair-accessible status.",
        extra_prerequisites=[acc_url_present],
    )

    # Accessible option capacity works (9 people)
    capacity_node = evaluator.add_leaf(
        id="Accessible_Option_Capacity_Works",
        desc="Plan states the selected accessible option can accommodate 9 campers and 2 wheelchair users (capacity/limits addressed).",
        parent=node,
        critical=True,
    )
    cap_sources = (s.accessible_site_urls or []) + (s.group_size_rule_urls or [])
    await evaluator.verify(
        claim=("The selected accessible option can reasonably accommodate a total of nine campers, including two wheelchair users; "
               "capacity or site limits support this plan."),
        node=capacity_node,
        sources=cap_sources,
        additional_instruction=("Confirm that capacity or site limits in the provided source(s) align with accommodating nine campers "
                                "and are compatible with accessibility needs."),
        extra_prerequisites=[acc_url_present, gs_url_present],
    )

    # Accessibility logistics: must state BOTH distance from parking/trailhead AND accessibility features
    logistics_node = evaluator.add_leaf(
        id="Accessibility_Logistics_Stated",
        desc="Plan states at least one concrete accessibility logistics detail: (a) distance from parking/trailhead AND (b) how the route/facility is wheelchair-accessible (surface/trail/access features).",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim=("The plan explicitly includes both: (a) a stated distance from parking or the trailhead, and (b) a description of "
               "how the route or facility is wheelchair-accessible (e.g., surface type, grade, boardwalk, accessible features)."),
        node=logistics_node,
        additional_instruction="Judge based on answer content only; both elements must be present to pass.",
    )

    # Stay limits compliance in content
    seasonal_stay_node = evaluator.add_leaf(
        id="Max_Seasonal_Trip_Length_Compliance",
        desc="Plan states compliance with the maximum stay limit of 7 nights during June–September.",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim="The plan explicitly states compliance with the RMNP seasonal stay limit of a maximum of seven nights during June through September.",
        node=seasonal_stay_node,
        additional_instruction="Judge based on answer content only; pass if 7-night seasonal limit is clearly addressed.",
    )

    consecutive_stay_node = evaluator.add_leaf(
        id="Max_Consecutive_Nights_Per_Campsite_Compliance",
        desc="Plan states compliance with the maximum of 3 consecutive nights at any single wilderness campsite (explains campsite rotation if needed).",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim="The plan explicitly states compliance with the RMNP rule limiting stays to a maximum of three consecutive nights at any single wilderness campsite.",
        node=consecutive_stay_node,
        additional_instruction="Judge based on answer content only; pass if 3-nights-per-site limit is clearly addressed.",
    )

    # URL presence for stay limits (existence gate)
    stay_url_present = evaluator.add_custom_node(
        result=bool(s.stay_limit_urls),
        id="Stay_Limits_URL_Provided",
        desc="At least one URL is provided supporting RMNP stay-limit rules.",
        parent=node,
        critical=True,
    )

    # URL-backed support for stay limits
    stay_url_node = evaluator.add_leaf(
        id="URL_For_Stay_Limits",
        desc="Provides a reference URL supporting the stay-limit rules used (7-night seasonal max and 3 consecutive nights/site).",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim=("RMNP stay limits include a maximum of seven nights during June–September and a maximum of three consecutive nights "
               "at any single wilderness campsite."),
        node=stay_url_node,
        sources=s.stay_limit_urls,
        additional_instruction="Verify that both limits are supported by the provided source(s).",
        extra_prerequisites=[stay_url_present],
    )


async def verify_permit_cost(evaluator: Evaluator, parent_node, ext: RMNPPlanExtraction) -> None:
    p = ext.permit_cost or PermitCost()

    node = evaluator.add_parallel(
        id="Total_Wilderness_Permit_Cost",
        desc="Verify the plan calculates the correct total wilderness permit fee for a June 2026 trip, with URL.",
        parent=parent_node,
        critical=True,
    )

    # URL presence (gate)
    fee_url_present = evaluator.add_custom_node(
        result=bool(p.permit_fee_urls),
        id="Permit_Fee_URL_Provided",
        desc="At least one URL is provided supporting RMNP wilderness permit fee.",
        parent=node,
        critical=True,
    )

    # Correct fee rule
    fee_rule_node = evaluator.add_leaf(
        id="Fee_Rule_Applied_Correctly",
        desc="Uses the correct rule: RMNP wilderness permits cost $36 per trip during May 1–Oct 31 (June qualifies).",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim="RMNP wilderness permits cost $36 per trip during May 1 through October 31; June is within this window.",
        node=fee_rule_node,
        sources=p.permit_fee_urls,
        additional_instruction="Verify the fee amount and seasonal window from the provided source(s).",
        extra_prerequisites=[fee_url_present],
    )

    # Computed as per-trip (content check)
    per_trip_node = evaluator.add_leaf(
        id="Total_Permit_Cost_Computed_As_Per_Trip",
        desc="Computes total wilderness permit cost correctly as a per-trip fee (not per-person or per-night).",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim="The plan computes the wilderness permit fee as a single per-trip charge rather than per-person or per-night.",
        node=per_trip_node,
        additional_instruction="Judge based on answer content only.",
    )

    # URL-backed reference
    fee_url_node = evaluator.add_leaf(
        id="URL_For_Permit_Fee",
        desc="Provides a reference URL supporting the $36-per-trip fee rule.",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim="The provided source(s) explicitly state that the RMNP wilderness permit is $36 per trip during the peak season.",
        node=fee_url_node,
        sources=p.permit_fee_urls,
        additional_instruction="Confirm both the $36 amount and the 'per-trip' nature from the source(s).",
        extra_prerequisites=[fee_url_present],
    )


async def verify_equipment(evaluator: Evaluator, parent_node, ext: RMNPPlanExtraction) -> None:
    e = ext.equipment or Equipment()

    node = evaluator.add_parallel(
        id="Required_Equipment",
        desc="Verify the plan includes mandatory equipment rules provided in constraints (bear canister + placement), with URL.",
        parent=parent_node,
        critical=True,
    )

    # URL presence (gate)
    bear_url_present = evaluator.add_custom_node(
        result=bool(e.bear_canister_urls),
        id="Bear_Canister_URL_Provided",
        desc="At least one URL is provided supporting bear canister rules.",
        parent=node,
        critical=True,
    )

    # Bear canister required (below treeline, Apr 1–Oct 31)
    bear_req_node = evaluator.add_leaf(
        id="Bear_Canister_Required_When_Applicable",
        desc="States bear-resistant food canisters are required from Apr 1–Oct 31 in wilderness areas below treeline (June trip must comply when below treeline).",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim=("RMNP requires bear-resistant food canisters from April 1 through October 31 in wilderness areas below treeline; "
               "a June trip must comply when camping below treeline."),
        node=bear_req_node,
        sources=e.bear_canister_urls,
        additional_instruction="Verify both the seasonal dates and the elevation condition from the source(s).",
        extra_prerequisites=[bear_url_present],
    )

    # Placement rule: 200 feet from camp & water
    placement_node = evaluator.add_leaf(
        id="Bear_Canister_Placement_Rule_Stated",
        desc="States canisters must be placed at least 200 feet (70 adult steps) from camp and water sources.",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim="Bear canisters must be placed at least 200 feet (about 70 adult steps) from camp and from water sources.",
        node=placement_node,
        sources=e.bear_canister_urls,
        additional_instruction="Verify the distance and placement guidance from the source(s).",
        extra_prerequisites=[bear_url_present],
    )

    # URL-backed rule support
    bear_url_node = evaluator.add_leaf(
        id="URL_For_Bear_Canister_Rules",
        desc="Provides a reference URL supporting the bear-canister requirement and the 200-feet placement rule.",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim="The provided source(s) clearly state both the seasonal bear canister requirement and the 200-feet placement guidance.",
        node=bear_url_node,
        sources=e.bear_canister_urls,
        additional_instruction="Confirm both requirement and placement guidance in the source(s).",
        extra_prerequisites=[bear_url_present],
    )


async def verify_reservation(evaluator: Evaluator, parent_node, ext: RMNPPlanExtraction) -> None:
    r = ext.reservation or ReservationProcedure()

    node = evaluator.add_parallel(
        id="Reservation_Booking_Procedure",
        desc="Verify the plan describes reservation booking timing, platform, lead-time rule, and accessible-site procedure, with URLs.",
        parent=parent_node,
        critical=True,
    )

    # URL presence for general reservation rules (gate)
    res_url_present = evaluator.add_custom_node(
        result=bool(r.reservation_rules_urls),
        id="Reservation_Rules_URL_Provided",
        desc="At least one URL is provided supporting RMNP reservation rules.",
        parent=node,
        critical=True,
    )

    # Reservation window opens
    window_node = evaluator.add_leaf(
        id="Reservation_Window_Opens",
        desc="States May–October wilderness reservations open March 1 at 8:00 AM Mountain Standard Time.",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim="RMNP wilderness reservations for May–October open on March 1 at 8:00 AM Mountain Standard Time.",
        node=window_node,
        sources=r.reservation_rules_urls,
        additional_instruction="Verify the opening date/time from the provided source(s).",
        extra_prerequisites=[res_url_present],
    )

    # Platform Recreation.gov
    platform_node = evaluator.add_leaf(
        id="Standard_Platform_RecreationGov",
        desc="States standard wilderness reservations are made via Recreation.gov.",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim="Standard RMNP wilderness reservations are made via Recreation.gov.",
        node=platform_node,
        sources=r.reservation_rules_urls,
        additional_instruction="Confirm platform usage in the source(s).",
        extra_prerequisites=[res_url_present],
    )

    # Advance booking minimum 3 days
    advance_node = evaluator.add_leaf(
        id="Advance_Booking_Minimum_3_Days",
        desc="States reservations must be made at least 3 days before the first camping date.",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim="Reservations must be made at least three days before the first camping date.",
        node=advance_node,
        sources=r.reservation_rules_urls,
        additional_instruction="Confirm the minimum lead-time from the source(s).",
        extra_prerequisites=[res_url_present],
    )

    # Accessible site special procedure
    acc_res_url_present = evaluator.add_custom_node(
        result=bool(r.accessible_reservation_urls),
        id="Accessible_Reservation_URL_Provided",
        desc="At least one URL is provided supporting the accessible-site reservation procedure.",
        parent=node,
        critical=True,
    )

    acc_proc_node = evaluator.add_leaf(
        id="Accessible_Site_Special_Procedure",
        desc=("Describes any special procedure for reserving the chosen accessible site; if using Sprague Lake accessible wilderness campsite, "
              "it states reservations are made by calling the Wilderness Office at 970-586-1242."),
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim=("The chosen accessible site requires a special reservation procedure (for example, initiating by phone with the RMNP Wilderness Office), "
               "as indicated by the provided source(s)."),
        node=acc_proc_node,
        sources=r.accessible_reservation_urls,
        additional_instruction="Confirm that the procedure differs from standard Recreation.gov flow (e.g., phone reservations for specific accessible sites).",
        extra_prerequisites=[acc_res_url_present],
    )

    # URL-backed rules summary
    res_rules_node = evaluator.add_leaf(
        id="URL_For_Reservation_Rules",
        desc="Provides reference URL(s) supporting: reservation opening time, Recreation.gov usage, and the 3-day minimum lead time.",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim=("The provided source(s) explicitly confirm the March 1 8:00 AM MST opening, Recreation.gov as the platform, "
               "and the minimum three-day advance booking rule."),
        node=res_rules_node,
        sources=r.reservation_rules_urls,
        additional_instruction="Confirm all three items are supported in the source(s).",
        extra_prerequisites=[res_url_present],
    )

    acc_rules_node = evaluator.add_leaf(
        id="URL_For_Accessible_Reservation_Procedure",
        desc="Provides a reference URL supporting the accessible-site reservation procedure used (including the phone-reservation procedure if Sprague Lake is used).",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim="The provided source(s) describe the accessible-site reservation procedure (e.g., via phone with the Wilderness Office).",
        node=acc_rules_node,
        sources=r.accessible_reservation_urls,
        additional_instruction="Confirm the accessible-site reservation instructions and contact method from the source(s).",
        extra_prerequisites=[acc_res_url_present],
    )


async def verify_other_constraints(evaluator: Evaluator, parent_node, ext: RMNPPlanExtraction) -> None:
    o = ext.other_constraints or OtherConstraints()

    node = evaluator.add_parallel(
        id="Other_Explicit_Constraints",
        desc="Verify additional explicit constraints are respected.",
        parent=parent_node,
        critical=True,
    )

    # URL presence (gate)
    pet_url_present = evaluator.add_custom_node(
        result=bool(o.pet_rules_urls),
        id="Pet_Rules_URL_Provided",
        desc="At least one URL is provided supporting pets prohibition.",
        parent=node,
        critical=True,
    )

    # Pets prohibited
    pets_node = evaluator.add_leaf(
        id="Pets_Prohibited_In_Wilderness",
        desc="States pets are not permitted in RMNP wilderness areas.",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim="Pets are not permitted in Rocky Mountain National Park wilderness areas.",
        node=pets_node,
        sources=o.pet_rules_urls,
        additional_instruction="Confirm the prohibition from the provided source(s).",
        extra_prerequisites=[pet_url_present],
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
    # Initialize evaluator (root is non-critical by design; we will add a critical plan root node)
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

    # Add critical plan root node mirroring rubric
    plan_root = evaluator.add_parallel(
        id="Complete_Wilderness_Trip_Plan",
        desc="Evaluate the trip plan for RMNP wilderness camping against the proposed question requirements and the provided constraints, including required citations (URLs).",
        parent=root,
        critical=True,
    )

    # Extract structured plan information
    extraction = await evaluator.extract(
        prompt=prompt_extract_plan(),
        template_class=RMNPPlanExtraction,
        extraction_name="rmnp_plan_extraction",
    )

    # Build and verify subtrees in parallel structure
    await verify_trip_scope(evaluator, plan_root, extraction)
    await verify_site_selection(evaluator, plan_root, extraction)
    await verify_permit_cost(evaluator, plan_root, extraction)
    await verify_equipment(evaluator, plan_root, extraction)
    await verify_reservation(evaluator, plan_root, extraction)
    await verify_other_constraints(evaluator, plan_root, extraction)

    # Summary result
    return evaluator.get_summary()