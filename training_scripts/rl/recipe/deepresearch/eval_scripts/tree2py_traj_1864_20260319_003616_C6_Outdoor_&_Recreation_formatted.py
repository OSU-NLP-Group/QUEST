import asyncio
import logging
from typing import Optional, List, Dict, Any, Tuple
from urllib.parse import urlparse

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "multi_park_backpacking_trip_planning_2026"
TASK_DESCRIPTION = """
You are planning a summer 2026 multi-park backpacking trip for a group of 5 adults visiting three western U.S. national parks. One member of your group is a 65-year-old U.S. resident eligible for a Senior Pass. The other four members are U.S. residents under age 62 who would need regular America the Beautiful Annual Passes.

Your trip will include:
1. A 4-night wilderness backpacking trip in Rocky Mountain National Park during late June 2026, staying in individual campsites
2. A 3-night backcountry camping trip in Grand Teton National Park
3. A wilderness backpacking trip in either Yosemite National Park or Mount Rainier National Park (you choose which park based on your research of their permit lottery systems)

For your trip planning, provide the following information with supporting URL references from official sources:

Part A - Rocky Mountain National Park:
1. What is the cost of a wilderness permit for your group's 4-night trip?
2. What are the group size limits for individual wilderness campsites?
3. What is the maximum number of consecutive nights you can stay in one camp area during June-September?
4. When do timed entry reservations first become available for late June 2026 entry dates, and what time do they open?

Part B - Grand Teton National Park:
1. What is the total permit cost for your group of 5 people for a 3-night trip? (Break down the base permit fee and per-person nightly fees)
2. Since your group has 5 people, what type of campsite should you use (individual or group)?
3. When do advance backcountry reservations typically open for the summer season?

Part C - Third Park (Yosemite or Mount Rainier):
1. Which park did you select and why based on the permit lottery system?
2. What percentage of wilderness permits are available through the early lottery system?
3. When does the lottery application window open?
4. When do the remaining permits (not allocated through lottery) become available?

Part D - Pass Selection and Cost Analysis:
1. What is the cost of a Senior Lifetime Pass for the 65-year-old group member?
2. What camping discount does the Senior Pass provide at federal campgrounds?
3. What is the cost of an America the Beautiful Resident Annual Pass for each of the four younger group members?
4. Calculate and compare the total trip costs (passes + all permits) for:
   - The scenario where the senior buys a Senior Pass and the others buy Annual Passes
   - The scenario where all 5 members buy Annual Passes (senior doesn't use Senior Pass benefit)
   Show which option is more cost-effective and by how much.

Part E - Reservation Strategy (Optional/Bonus):
1. How far in advance do Recreation.gov campground reservations typically open?
2. At what time (including time zone) are reservations released?

For all answers, provide specific URL references from official park websites (nps.gov), Recreation.gov, or USGS Store that support your information.
"""


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class RMNPSection(BaseModel):
    # Values
    permit_total_cost_for_group_4_nights: Optional[str] = None
    individual_campsite_group_size_limit: Optional[str] = None
    max_consecutive_nights_jun_sep_one_area: Optional[str] = None
    timed_entry_first_available_date_for_late_june_2026: Optional[str] = None
    timed_entry_open_time_tz: Optional[str] = None
    # Sources
    permit_cost_sources: List[str] = Field(default_factory=list)
    group_size_sources: List[str] = Field(default_factory=list)
    max_nights_sources: List[str] = Field(default_factory=list)
    timed_entry_sources: List[str] = Field(default_factory=list)


class GTNPSection(BaseModel):
    # Values
    total_permit_cost_for_5_people_3_nights: Optional[str] = None
    base_permit_fee: Optional[str] = None
    per_person_per_night_fee: Optional[str] = None
    campsite_type_for_group_of_5: Optional[str] = None
    advance_reservations_open_timeframe: Optional[str] = None
    # Sources
    cost_sources: List[str] = Field(default_factory=list)
    campsite_type_sources: List[str] = Field(default_factory=list)
    reservations_open_sources: List[str] = Field(default_factory=list)


class ThirdParkSection(BaseModel):
    # Values
    selected_park: Optional[str] = None  # "Yosemite" or "Mount Rainier"
    selection_rationale_based_on_lottery: Optional[str] = None
    early_lottery_allocation_percentage: Optional[str] = None  # or "no official percentage is published"
    lottery_application_window_open: Optional[str] = None
    remaining_permits_availability: Optional[str] = None
    # Optional: some answers may include an explicit permit total for this third park
    permit_total_cost_for_trip: Optional[str] = None
    # Sources
    selection_sources: List[str] = Field(default_factory=list)
    lottery_percentage_sources: List[str] = Field(default_factory=list)
    lottery_window_sources: List[str] = Field(default_factory=list)
    remaining_permits_sources: List[str] = Field(default_factory=list)
    permit_cost_sources: List[str] = Field(default_factory=list)


class PassSection(BaseModel):
    # Values
    senior_lifetime_pass_cost: Optional[str] = None
    senior_pass_camping_discount: Optional[str] = None
    annual_pass_cost_per_person: Optional[str] = None
    scenario1_total_cost: Optional[str] = None  # Senior uses Senior Pass + others Annual; include permits
    scenario2_total_cost: Optional[str] = None  # All five buy Annual Passes; include permits
    cheaper_option: Optional[str] = None
    savings_amount: Optional[str] = None
    # Sources
    senior_pass_sources: List[str] = Field(default_factory=list)
    camping_discount_sources: List[str] = Field(default_factory=list)
    annual_pass_sources: List[str] = Field(default_factory=list)


class RecGovSection(BaseModel):
    how_far_in_advance_open: Optional[str] = None
    release_time_with_timezone: Optional[str] = None
    advance_open_sources: List[str] = Field(default_factory=list)
    release_time_sources: List[str] = Field(default_factory=list)


class TripExtraction(BaseModel):
    rmnp: RMNPSection = Field(default_factory=RMNPSection)
    gtnp: GTNPSection = Field(default_factory=GTNPSection)
    third_park: ThirdParkSection = Field(default_factory=ThirdParkSection)
    passes: PassSection = Field(default_factory=PassSection)
    recreation_gov: RecGovSection = Field(default_factory=RecGovSection)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_all() -> str:
    return """
Extract all required planning details from the answer and map them exactly into the following JSON structure. Use strings for numbers, dates, times, percentages, and money values exactly as stated in the answer. For each factual item, also extract the specific supporting URLs cited by the answer from official sources only: nps.gov, recreation.gov, store.usgs.gov, or usgs.gov. If multiple relevant URLs are cited, include all of them. If a required value is not present in the answer, put null for the value and an empty list for its sources.

The JSON fields to fill:

rmnp:
  permit_total_cost_for_group_4_nights: string or null
  individual_campsite_group_size_limit: string or null
  max_consecutive_nights_jun_sep_one_area: string or null
  timed_entry_first_available_date_for_late_june_2026: string or null
  timed_entry_open_time_tz: string or null
  permit_cost_sources: list of URLs (official only)
  group_size_sources: list of URLs (official only)
  max_nights_sources: list of URLs (official only)
  timed_entry_sources: list of URLs (official only)

gtnp:
  total_permit_cost_for_5_people_3_nights: string or null
  base_permit_fee: string or null
  per_person_per_night_fee: string or null
  campsite_type_for_group_of_5: string or null
  advance_reservations_open_timeframe: string or null
  cost_sources: list of URLs (official only)
  campsite_type_sources: list of URLs (official only)
  reservations_open_sources: list of URLs (official only)

third_park:
  selected_park: "Yosemite" or "Mount Rainier" (string) or null
  selection_rationale_based_on_lottery: string or null
  early_lottery_allocation_percentage: string or null  (if not officially published, write exactly "no official percentage is published" or a similar explicit statement used in the answer)
  lottery_application_window_open: string or null
  remaining_permits_availability: string or null
  permit_total_cost_for_trip: string or null (only if the answer provided a total for the chosen third park)
  selection_sources: list of URLs (official only)
  lottery_percentage_sources: list of URLs (official only)
  lottery_window_sources: list of URLs (official only)
  remaining_permits_sources: list of URLs (official only)
  permit_cost_sources: list of URLs (official only)

passes:
  senior_lifetime_pass_cost: string or null
  senior_pass_camping_discount: string or null
  annual_pass_cost_per_person: string or null
  scenario1_total_cost: string or null
  scenario2_total_cost: string or null
  cheaper_option: string or null  (e.g., "Scenario 1", "Scenario 2", or a phrase clearly identifying the cheaper option)
  savings_amount: string or null
  senior_pass_sources: list of URLs (official only)
  camping_discount_sources: list of URLs (official only)
  annual_pass_sources: list of URLs (official only)

recreation_gov:
  how_far_in_advance_open: string or null
  release_time_with_timezone: string or null
  advance_open_sources: list of URLs (official only)
  release_time_sources: list of URLs (official only)

Rules:
- Only include URLs that are explicitly present in the answer text.
- Prefer official domains: nps.gov, recreation.gov, store.usgs.gov, usgs.gov.
- Do not invent or infer values; if missing, use null.
- Preserve time zones in any time fields (e.g., "8:00 a.m. Mountain Time" or "7:00 am PT").
"""


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
ALLOWED_DOMAINS = ["nps.gov", "recreation.gov", "store.usgs.gov", "usgs.gov"]


def _domain_allowed(url: str) -> bool:
    try:
        netloc = urlparse(url).netloc.lower()
        # Strip leading "www."
        if netloc.startswith("www."):
            netloc = netloc[4:]
        # Allow any subdomains of allowed domains
        return any(netloc == dom or netloc.endswith("." + dom) for dom in ALLOWED_DOMAINS)
    except Exception:
        return False


def _collect_required_sources(ex: TripExtraction) -> Dict[str, List[str]]:
    """
    Collect the URL lists for all required (critical) factual claims
    that must be supported by official sources.
    Optional Part E is excluded from the "all-claims" requirement.
    """
    srcs: Dict[str, List[str]] = {
        # Part A (RMNP)
        "RMNP_Wilderness_Permit_Cost_For_Trip": ex.rmnp.permit_cost_sources,
        "RMNP_Individual_Campsite_Group_Size_Limit": ex.rmnp.group_size_sources,
        "RMNP_Max_Consecutive_Nights_Per_Camp_Area_Jun_Sep": ex.rmnp.max_nights_sources,
        "RMNP_Timed_Entry_First_Availability_For_Late_June_2026": ex.rmnp.timed_entry_sources,
        # Part B (GTNP)
        "GTNP_Total_Permit_Cost_For_5_People_3_Nights_With_Breakdown": ex.gtnp.cost_sources,
        "GTNP_Campsite_Type_For_Group_Of_5": ex.gtnp.campsite_type_sources,
        "GTNP_Advance_Backcountry_Reservation_Opening_Timeframe": ex.gtnp.reservations_open_sources,
        # Part C (Third Park) - all lottery details + rationale should cite official sources
        "Third_Park_Selection_Rationale": ex.third_park.selection_sources,
        "Early_Lottery_Permit_Allocation_Percentage": ex.third_park.lottery_percentage_sources,
        "Lottery_Application_Window_Open": ex.third_park.lottery_window_sources,
        "Remaining_Permits_Availability_Not_Allocated_By_Lottery": ex.third_park.remaining_permits_sources,
        # Part D (Passes) - pass prices & discount must be source-backed
        "Senior_Lifetime_Pass_Cost": ex.passes.senior_pass_sources,
        "Senior_Pass_Camping_Discount": ex.passes.camping_discount_sources,
        "Annual_Pass_Cost_Per_Younger_Member": ex.passes.annual_pass_sources,
    }
    return srcs


def _check_official_sources_requirement(ex: TripExtraction) -> Tuple[bool, Dict[str, Any]]:
    required = _collect_required_sources(ex)
    missing: Dict[str, List[str]] = {}
    invalid_domains: Dict[str, List[str]] = {}

    for key, urls in required.items():
        # Presence check
        if not urls or len(urls) == 0:
            missing[key] = urls or []
        # Domain check
        bad = [u for u in (urls or []) if not _domain_allowed(u)]
        if bad:
            invalid_domains[key] = bad

    passed = (len(missing) == 0) and (len(invalid_domains) == 0)
    debug = {"missing_sources": missing, "invalid_domain_urls": invalid_domains}
    return passed, debug


# --------------------------------------------------------------------------- #
# Verification logic per part                                                 #
# --------------------------------------------------------------------------- #
async def verify_part_a_rmnp(evaluator: Evaluator, parent_node, ex: RMNPSection) -> None:
    part_a = evaluator.add_parallel(
        id="Part_A_RMNP",
        desc="Rocky Mountain National Park wilderness permit and timed-entry information requested in Part A.",
        parent=parent_node,
        critical=True,
    )

    # 1) Permit cost for 4-night trip
    leaf1 = evaluator.add_leaf(
        id="RMNP_Wilderness_Permit_Cost_For_Trip",
        desc="States the wilderness permit cost for the group’s 4-night RMNP trip.",
        parent=part_a,
        critical=True,
    )
    claim1 = (
        f"For a group of 5 adults doing a 4-night wilderness backpacking trip in Rocky Mountain National Park "
        f"during June–September 2026 using individual campsites, the total wilderness permit cost is "
        f"{ex.permit_total_cost_for_group_4_nights}."
    )
    await evaluator.verify(
        claim=claim1,
        node=leaf1,
        sources=ex.permit_cost_sources,
        additional_instruction="Verify the total cost using the fee structure shown on official RMNP/NPS materials; "
                               "it's acceptable if the source states per-person or per-night fees as long as the "
                               "implied total for 5 people over 4 nights matches the claim."
    )

    # 2) Group size limits for individual wilderness campsites
    leaf2 = evaluator.add_leaf(
        id="RMNP_Individual_Campsite_Group_Size_Limit",
        desc="States the group size limits for individual wilderness campsites.",
        parent=part_a,
        critical=True,
    )
    claim2 = (
        f"In Rocky Mountain National Park, individual wilderness campsites have a group size limit of "
        f"{ex.individual_campsite_group_size_limit}."
    )
    await evaluator.verify(
        claim=claim2,
        node=leaf2,
        sources=ex.group_size_sources,
        additional_instruction="Focus on the official wilderness/backcountry regulations or campsite capacity rules."
    )

    # 3) Max consecutive nights per camp area in June–September
    leaf3 = evaluator.add_leaf(
        id="RMNP_Max_Consecutive_Nights_Per_Camp_Area_Jun_Sep",
        desc="States the maximum number of consecutive nights allowed in one camp area during June–September.",
        parent=part_a,
        critical=True,
    )
    claim3 = (
        f"In Rocky Mountain National Park, during June–September, the maximum number of consecutive nights "
        f"allowed in one camp area is {ex.max_consecutive_nights_jun_sep_one_area}."
    )
    await evaluator.verify(
        claim=claim3,
        node=leaf3,
        sources=ex.max_nights_sources,
        additional_instruction="Look for season-specific stay limits in the official RMNP wilderness camping rules."
    )

    # 4) Timed-entry first availability for late June 2026 and opening time
    leaf4 = evaluator.add_leaf(
        id="RMNP_Timed_Entry_First_Availability_For_Late_June_2026",
        desc="States when timed-entry reservations first become available for late June 2026 entry dates and what time they open (including time zone).",
        parent=part_a,
        critical=True,
    )
    claim4 = (
        f"Timed-entry reservations for late June 2026 in Rocky Mountain National Park first become available on "
        f"{ex.timed_entry_first_available_date_for_late_june_2026} at "
        f"{ex.timed_entry_open_time_tz}."
    )
    await evaluator.verify(
        claim=claim4,
        node=leaf4,
        sources=ex.timed_entry_sources,
        additional_instruction="Use official RMNP/NPS or Recreation.gov guidance. The statement must include both the first availability date and the opening time with time zone."
    )


async def verify_part_b_gtnp(evaluator: Evaluator, parent_node, ex: GTNPSection) -> None:
    part_b = evaluator.add_parallel(
        id="Part_B_GTNP",
        desc="Grand Teton National Park backcountry permit cost, campsite type decision, and reservation opening timing requested in Part B.",
        parent=parent_node,
        critical=True,
    )

    # 1) Total permit cost with breakdown
    leaf1 = evaluator.add_leaf(
        id="GTNP_Total_Permit_Cost_For_5_People_3_Nights_With_Breakdown",
        desc="Computes the total permit cost for 5 people for 3 nights and breaks it down into base fee and per-person nightly fees; arithmetic is correct given the stated fee structure.",
        parent=part_b,
        critical=True,
    )
    claim1 = (
        f"In Grand Teton National Park, the total backcountry permit cost for 5 people on a 3-night trip is "
        f"{ex.total_permit_cost_for_5_people_3_nights}, consisting of a base permit fee of {ex.base_permit_fee} "
        f"plus a per-person per-night fee of {ex.per_person_per_night_fee} applied to 5 people for 3 nights."
    )
    await evaluator.verify(
        claim=claim1,
        node=leaf1,
        sources=ex.cost_sources,
        additional_instruction="Verify both the fee components and that the implied arithmetic total for 5 people over 3 nights equals the stated total."
    )

    # 2) Campsite type for a group of 5
    leaf2 = evaluator.add_leaf(
        id="GTNP_Campsite_Type_For_Group_Of_5",
        desc="Correctly determines whether a group of 5 should use an individual or group campsite type based on the stated capacity rules.",
        parent=part_b,
        critical=True,
    )
    claim2 = (
        f"In Grand Teton National Park, a group of 5 must use a {ex.campsite_type_for_group_of_5} backcountry campsite "
        f"based on the capacity rules."
    )
    await evaluator.verify(
        claim=claim2,
        node=leaf2,
        sources=ex.campsite_type_sources,
        additional_instruction="Confirm the group size thresholds for individual vs. group backcountry sites."
    )

    # 3) Advance backcountry reservations opening timeframe
    leaf3 = evaluator.add_leaf(
        id="GTNP_Advance_Backcountry_Reservation_Opening_Timeframe",
        desc="States when advance backcountry reservations typically open for the summer season (timeframe/date).",
        parent=part_b,
        critical=True,
    )
    claim3 = f"In Grand Teton National Park, advance backcountry reservations typically open {ex.advance_reservations_open_timeframe}."
    await evaluator.verify(
        claim=claim3,
        node=leaf3,
        sources=ex.reservations_open_sources,
        additional_instruction="Accept the official seasonal opening window/date (month/day or rule-of-thumb timing) as stated by NPS or Recreation.gov."
    )


async def verify_part_c_third_park(evaluator: Evaluator, parent_node, ex: ThirdParkSection) -> None:
    part_c = evaluator.add_parallel(
        id="Part_C_Third_Park_Lottery_System",
        desc="Select either Yosemite or Mount Rainier and provide the permit lottery system details requested in Part C.",
        parent=parent_node,
        critical=True,
    )

    # 1) Selected park clearly identified
    exists_ok = (ex.selected_park is not None) and (ex.selected_park.strip().lower() in ["yosemite", "mount rainier"])
    evaluator.add_custom_node(
        result=exists_ok,
        id="Third_Park_Selected",
        desc="Clearly identifies which park is selected (Yosemite OR Mount Rainier).",
        parent=part_c,
        critical=True,
    )

    # 2) Selection rationale based on lottery system
    leaf2 = evaluator.add_leaf(
        id="Selection_Rationale_Based_On_Lottery_System",
        desc="Justifies the park choice by referencing lottery-system characteristics (e.g., allocation method and timing windows) relevant to obtaining permits.",
        parent=part_c,
        critical=True,
    )
    claim2 = (
        f"The stated rationale for choosing {ex.selected_park} is correct based on official descriptions of its "
        f"permit lottery system: {ex.selection_rationale_based_on_lottery}"
    )
    await evaluator.verify(
        claim=claim2,
        node=leaf2,
        sources=ex.selection_sources,
        additional_instruction="Check that the described lottery characteristics (allocation, windows, process) match the official source."
    )

    # 3) Early lottery allocation percentage (or explicit statement that no official % is published)
    leaf3 = evaluator.add_leaf(
        id="Early_Lottery_Permit_Allocation_Percentage",
        desc="Provides the percentage/proportion of wilderness permits available through the early lottery system for the selected park, or (if no official percentage is published) explicitly states that with an official citation.",
        parent=part_c,
        critical=True,
    )
    claim3 = (
        f"For {ex.selected_park}, the early lottery allocation percentage is stated as "
        f"'{ex.early_lottery_allocation_percentage}'. If this states that no official percentage is published, "
        f"the cited official sources must support that."
    )
    await evaluator.verify(
        claim=claim3,
        node=leaf3,
        sources=ex.lottery_percentage_sources,
        additional_instruction="If the answer claims no official percentage is published, verify that the official source does not publish a percentage and/or explicitly says this."
    )

    # 4) Lottery application window opens
    leaf4 = evaluator.add_leaf(
        id="Lottery_Application_Window_Open",
        desc="States when the lottery application window opens for the selected park (date or lead-time rule).",
        parent=part_c,
        critical=True,
    )
    claim4 = f"The lottery application window for {ex.selected_park} opens {ex.lottery_application_window_open}."
    await evaluator.verify(
        claim=claim4,
        node=leaf4,
        sources=ex.lottery_window_sources,
        additional_instruction="Accept a concrete calendar date or a relative lead-time rule, as long as it aligns with the official source."
    )

    # 5) Remaining permits availability not allocated by lottery
    leaf5 = evaluator.add_leaf(
        id="Remaining_Permits_Availability_Not_Allocated_By_Lottery",
        desc="States when remaining permits (not allocated through the lottery) become available for the selected park.",
        parent=part_c,
        critical=True,
    )
    claim5 = f"In {ex.selected_park}, remaining wilderness permits not allocated by the lottery become available {ex.remaining_permits_availability}."
    await evaluator.verify(
        claim=claim5,
        node=leaf5,
        sources=ex.remaining_permits_sources,
        additional_instruction="Verify rolling release, first-come/first-served windows, or specific dates as stated in official sources."
    )


async def verify_part_d_passes_and_costs(evaluator: Evaluator, parent_node, ex_pass: PassSection, ex_rmnp: RMNPSection, ex_gtnp: GTNPSection, ex_third: ThirdParkSection) -> None:
    part_d = evaluator.add_parallel(
        id="Part_D_Pass_Selection_And_Cost_Analysis",
        desc="Pass prices/benefits and total trip cost comparison for the two scenarios requested in Part D.",
        parent=parent_node,
        critical=True,
    )

    # 1) Senior Lifetime Pass cost
    leaf1 = evaluator.add_leaf(
        id="Senior_Lifetime_Pass_Cost",
        desc="States the cost of the Senior Lifetime Pass for the 65-year-old group member.",
        parent=part_d,
        critical=True,
    )
    claim1 = f"The cost of the Senior Lifetime Pass is {ex_pass.senior_lifetime_pass_cost}."
    await evaluator.verify(
        claim=claim1,
        node=leaf1,
        sources=ex_pass.senior_pass_sources,
        additional_instruction="Validate on the USGS Store or official federal pass page. The value should be the standard lifetime Senior Pass price."
    )

    # 2) Senior Pass camping discount
    leaf2 = evaluator.add_leaf(
        id="Senior_Pass_Camping_Discount",
        desc="States the camping/amenity discount provided by the Senior Pass.",
        parent=part_d,
        critical=True,
    )
    claim2 = f"The Senior Pass provides a camping/amenity discount of {ex_pass.senior_pass_camping_discount} at eligible federal campgrounds."
    await evaluator.verify(
        claim=claim2,
        node=leaf2,
        sources=ex_pass.camping_discount_sources,
        additional_instruction="Confirm the discount percentage and applicable scope per official policy (e.g., 50% on certain amenity fees where applicable)."
    )

    # 3) Annual Pass cost per younger member
    leaf3 = evaluator.add_leaf(
        id="Annual_Pass_Cost_Per_Younger_Member",
        desc="States the cost of an America the Beautiful Resident Annual Pass for each of the four younger group members.",
        parent=part_d,
        critical=True,
    )
    claim3 = f"The cost of an America the Beautiful Annual Pass is {ex_pass.annual_pass_cost_per_person} per person."
    await evaluator.verify(
        claim=claim3,
        node=leaf3,
        sources=ex_pass.annual_pass_sources,
        additional_instruction="Validate on the USGS Store or official federal pass page."
    )

    # 4) Total costs for two scenarios compared (arithmetic/logical check)
    leaf4 = evaluator.add_leaf(
        id="Total_Costs_Two_Scenarios_Compared",
        desc="Calculates and compares total trip costs (passes + all required permits across all three parks) for (1) senior uses Senior Pass + others use Annual Passes and (2) all five use Annual Passes; identifies which is cheaper and by how much; includes correct arithmetic.",
        parent=part_d,
        critical=True,
    )
    # Build a reasoning-focused claim for arithmetic validation
    claim4 = (
        f"Based on the numbers stated in the answer, the total trip cost for Scenario 1 is {ex_pass.scenario1_total_cost} "
        f"and for Scenario 2 is {ex_pass.scenario2_total_cost}. The answer correctly identifies the cheaper option as "
        f"'{ex_pass.cheaper_option}' and the savings amount is {ex_pass.savings_amount}. The arithmetic is internally consistent."
    )
    await evaluator.verify(
        claim=claim4,
        node=leaf4,
        sources=None,  # Arithmetic/logical verification does not require URLs
        additional_instruction="Check only the arithmetic/logical consistency using the provided numbers. Allow reasonable rounding."
    )


async def verify_part_e_optional(evaluator: Evaluator, parent_node, ex: RecGovSection) -> None:
    part_e = evaluator.add_parallel(
        id="Part_E_RecreationGov_Optional_Bonus",
        desc="Optional/bonus: Recreation.gov campground reservation timing details requested in Part E.",
        parent=parent_node,
        critical=False,
    )

    # 1) How far in advance
    leaf1 = evaluator.add_leaf(
        id="RecreationGov_How_Far_In_Advance",
        desc="States how far in advance Recreation.gov campground reservations typically open.",
        parent=part_e,
        critical=False,
    )
    claim1 = f"On Recreation.gov, campground reservations typically open {ex.how_far_in_advance_open} in advance."
    await evaluator.verify(
        claim=claim1,
        node=leaf1,
        sources=ex.advance_open_sources,
        additional_instruction="Use Recreation.gov official help/article pages when possible."
    )

    # 2) Release time and time zone
    leaf2 = evaluator.add_leaf(
        id="RecreationGov_Release_Time_TimeZone",
        desc="States the time reservations are released, including time zone.",
        parent=part_e,
        critical=False,
    )
    claim2 = f"Recreation.gov releases reservations at {ex.release_time_with_timezone}."
    await evaluator.verify(
        claim=claim2,
        node=leaf2,
        sources=ex.release_time_sources,
        additional_instruction="Confirm official release time and include the time zone."
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
    Evaluate an agent's answer for the multi-park backpacking trip planning task.
    """
    # Initialize evaluator (root kept non-critical to allow optional partial credit node under it)
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
    extracted: TripExtraction = await evaluator.extract(
        prompt=prompt_extract_all(),
        template_class=TripExtraction,
        extraction_name="trip_planning_extraction",
    )

    # Build main parts under root
    # Part A
    await verify_part_a_rmnp(evaluator, root, extracted.rmnp)
    # Part B
    await verify_part_b_gtnp(evaluator, root, extracted.gtnp)
    # Part C
    await verify_part_c_third_park(evaluator, root, extracted.third_park)
    # Part D
    await verify_part_d_passes_and_costs(evaluator, root, extracted.passes, extracted.rmnp, extracted.gtnp, extracted.third_park)
    # Part E (Optional)
    await verify_part_e_optional(evaluator, root, extracted.recreation_gov)

    # Global official-URL citations check (critical leaf)
    official_urls_ok, url_debug = _check_official_sources_requirement(extracted)
    evaluator.add_custom_node(
        result=official_urls_ok,
        id="Official_URL_Citations_For_All_Claims",
        desc="All factual/numeric claims are supported by specific URL citations from allowed official domains (nps.gov, recreation.gov, store.usgs.gov, usgs.gov).",
        parent=root,
        critical=True,
    )
    # Record detailed diagnostics for missing/invalid URLs
    evaluator.add_custom_info(url_debug, info_type="url_citations_audit", info_name="official_url_citations_audit")

    # Return evaluation summary
    return evaluator.get_summary()