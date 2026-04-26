import asyncio
import logging
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.llm_client.base_client import LLMClient


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "winter_trip_planning"
TASK_DESCRIPTION = """Three families from Bangor, Maine (totaling 11 people: 6 adults and 5 children ages 4-14) are planning a comprehensive winter outdoor recreation trip for early January 2026. One family will be bringing their pet dog. They need help identifying four suitable destinations:

1. A nearby winter destination within 2 hours driving distance from Bangor that offers at least two winter outdoor activities (such as cross-country skiing, snowshoeing, or winter hiking) and accepts the America the Beautiful pass for entry.

2. A western US destination that offers winter snowmobiling access through a permit or access program for non-commercial groups, can accommodate their group size (with details on group size limits per permit), and is accessible for snowmobiling in early January 2026. Include permit fee information.

3. A pet-friendly camping destination (state park or similar) that allows pets in campsites, offers RV or tent camping, is open in early January 2026, and has a reservation system. Specify the advance booking window and any campsite capacity limits.

4. A family-friendly ski resort that offers skiing activities or programs suitable for children ages 4-14, has lodging options appropriate for three families (such as multiple rooms, condos, or cabins), and provides lift ticket pricing information. Note any group booking policies (acknowledging that 11 people is typically below the 20-person minimum for group discounts).

For each destination, provide:
- The destination name
- Verification of how it meets each specified criterion
- Supporting URL references for each key requirement
"""


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class NearbyDestination(BaseModel):
    name: Optional[str] = None
    distance_claim: Optional[str] = None
    distance_urls: List[str] = Field(default_factory=list)
    activities: List[str] = Field(default_factory=list)
    activities_urls: List[str] = Field(default_factory=list)
    pass_acceptance_urls: List[str] = Field(default_factory=list)


class SnowmobileDestination(BaseModel):
    name: Optional[str] = None
    location_urls: List[str] = Field(default_factory=list)
    permit_program_desc: Optional[str] = None
    permit_urls: List[str] = Field(default_factory=list)
    group_size_desc: Optional[str] = None
    group_size_urls: List[str] = Field(default_factory=list)
    january_access_desc: Optional[str] = None
    season_urls: List[str] = Field(default_factory=list)
    fee_desc: Optional[str] = None
    fee_urls: List[str] = Field(default_factory=list)


class CampingDestination(BaseModel):
    name: Optional[str] = None
    pet_policy_desc: Optional[str] = None
    pet_policy_urls: List[str] = Field(default_factory=list)
    camping_types: List[str] = Field(default_factory=list)  # e.g., ["RV", "Tent"]
    reservation_system_desc: Optional[str] = None
    reservation_urls: List[str] = Field(default_factory=list)
    advance_booking_window: Optional[str] = None
    campsite_capacity_desc: Optional[str] = None
    capacity_urls: List[str] = Field(default_factory=list)
    january_open_desc: Optional[str] = None
    operations_urls: List[str] = Field(default_factory=list)


class SkiDestination(BaseModel):
    name: Optional[str] = None
    youth_programs_desc: Optional[str] = None
    youth_programs_urls: List[str] = Field(default_factory=list)
    lodging_desc: Optional[str] = None
    lodging_urls: List[str] = Field(default_factory=list)
    ticket_pricing_desc: Optional[str] = None
    ticket_pricing_urls: List[str] = Field(default_factory=list)
    group_policies_desc: Optional[str] = None
    group_policies_urls: List[str] = Field(default_factory=list)


class WinterTripExtraction(BaseModel):
    nearby: Optional[NearbyDestination] = None
    snowmobile: Optional[SnowmobileDestination] = None
    camping: Optional[CampingDestination] = None
    ski: Optional[SkiDestination] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_winter_trip() -> str:
    return """
Extract the four destinations and their key verification details from the answer text. Return a JSON object matching the following structure. Only use information explicitly present in the answer; do not invent anything. For every URL field, extract actual URLs exactly as provided in the answer (plain or markdown). If some field is not mentioned, set it to null (for strings) or an empty array (for list fields).

Structure to extract:
{
  "nearby": {
    "name": string | null,
    "distance_claim": string | null,                         // e.g., "~1 hour from Bangor" or "~50 miles"
    "distance_urls": string[],                               // URLs that document driving distance/time from Bangor
    "activities": string[],                                  // list of winter activities mentioned (e.g., "cross-country skiing", "snowshoeing")
    "activities_urls": string[],                             // URLs that document available winter activities
    "pass_acceptance_urls": string[]                         // URLs confirming America the Beautiful pass acceptance
  },
  "snowmobile": {
    "name": string | null,
    "location_urls": string[],                               // URLs confirming it's in a western US state
    "permit_program_desc": string | null,                    // description of non-commercial group access/permits
    "permit_urls": string[],                                 // URLs for permit/access program
    "group_size_desc": string | null,                        // description of group size limits per permit / subgrouping
    "group_size_urls": string[],                             // URLs for group size policies
    "january_access_desc": string | null,                    // description indicating early Jan access
    "season_urls": string[],                                 // URLs confirming season/operating dates including early Jan
    "fee_desc": string | null,                               // description of permit fees (if any)
    "fee_urls": string[]                                     // URLs with fee information
  },
  "camping": {
    "name": string | null,
    "pet_policy_desc": string | null,                        // description confirming pets allowed in campsites
    "pet_policy_urls": string[],                             // URLs with pet policies
    "camping_types": string[],                                // e.g., ["RV", "Tent"]
    "reservation_system_desc": string | null,                // description confirming reservation system exists
    "reservation_urls": string[],                            // URLs with reservation system/policies
    "advance_booking_window": string | null,                 // booking window (e.g., "6 months in advance")
    "campsite_capacity_desc": string | null,                 // campsite limits (e.g., "1 RV + 1 tent")
    "capacity_urls": string[],                               // URLs with campsite specs/limits
    "january_open_desc": string | null,                      // description confirming open in early Jan 2026
    "operations_urls": string[]                              // URLs with winter/operating season
  },
  "ski": {
    "name": string | null,
    "youth_programs_desc": string | null,                    // children's lessons/programs ages 4–14
    "youth_programs_urls": string[],                         // URLs documenting kids/family programs
    "lodging_desc": string | null,                           // description of lodging for 3 families (rooms/condos/cabins)
    "lodging_urls": string[],                                // URLs for lodging options
    "ticket_pricing_desc": string | null,                    // lift ticket pricing summary
    "ticket_pricing_urls": string[],                         // URLs for lift ticket pricing
    "group_policies_desc": string | null,                    // group booking policy summary (note if 11 < group min)
    "group_policies_urls": string[]                          // URLs for group policy details (if provided)
  }
}

Special notes:
- For URL extraction, include only valid URLs explicitly present in the answer. If the answer references a site without a URL, leave the list empty.
- For the nearby destination, activities must include at least two items if provided.
- For the snowmobile destination, focus on non-commercial group access/permits, group size limits, January access, and fees.
- For the camping destination, ensure pet policy, reservation system + booking window, capacity limits, and January operations are captured if present.
- For the ski destination, capture youth suitability (ages 4–14), lodging options for multiple families, ticket pricing, and group policies if mentioned.
"""


# --------------------------------------------------------------------------- #
# Verification helpers                                                        #
# --------------------------------------------------------------------------- #
async def verify_nearby(evaluator: Evaluator, parent, data: Optional[NearbyDestination]) -> None:
    # Parent: Nearby_Destination (parallel, non-critical)
    nearby_node = evaluator.add_parallel(
        id="nearby_destination",
        desc="Identify a suitable winter recreation destination within 2 hours driving from Bangor, Maine",
        parent=parent,
        critical=False
    )

    # Guard against None data; still build leaves to reflect rubric, but claims may be generic
    name = (data.name if data and data.name else "the destination")
    distance_urls = data.distance_urls if data else []
    activities_urls = data.activities_urls if data else []
    pass_urls = data.pass_acceptance_urls if data else []

    # Location_Distance -> Distance_Requirement -> Within_Range + Distance_Reference
    loc_dist_node = evaluator.add_parallel(
        id="nearby_location_distance",
        desc="Verify the destination is within 2 hours driving distance from Bangor",
        parent=nearby_node,
        critical=True
    )
    dist_req_node = evaluator.add_parallel(
        id="nearby_distance_requirement",
        desc="Confirm proximity meets the 2-hour or 100-mile requirement",
        parent=loc_dist_node,
        critical=True
    )

    within_range_leaf = evaluator.add_leaf(
        id="nearby_within_range",
        desc="The destination is within 2 hours or approximately 100 miles driving distance from Bangor",
        parent=dist_req_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"{name} is within approximately 2 hours or about 100 miles driving distance from Bangor, Maine.",
        node=within_range_leaf,
        sources=distance_urls,
        additional_instruction="Accept reasonable approximations (e.g., ~1 hr 50 min, ~95–110 miles). The evidence page should explicitly or clearly imply driving time or distance from Bangor, ME."
    )

    distance_ref_leaf = evaluator.add_custom_node(
        result=bool(distance_urls),
        id="nearby_distance_reference",
        desc="Provide URL documenting the distance or driving time from Bangor",
        parent=dist_req_node,
        critical=True
    )

    # Winter_Activities -> Activity_Verification -> Multiple_Activities + Activity_Reference
    activities_node = evaluator.add_parallel(
        id="nearby_winter_activities",
        desc="Verify the destination offers at least two winter outdoor activities",
        parent=nearby_node,
        critical=True
    )
    activity_ver_node = evaluator.add_parallel(
        id="nearby_activity_verification",
        desc="Confirm availability of multiple winter activity options",
        parent=activities_node,
        critical=True
    )

    multi_acts_leaf = evaluator.add_leaf(
        id="nearby_multiple_activities",
        desc="The destination offers at least two winter activities such as cross-country skiing, snowshoeing, or winter hiking",
        parent=activity_ver_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"{name} offers at least two winter activities (e.g., cross-country skiing, snowshoeing, winter hiking).",
        node=multi_acts_leaf,
        sources=activities_urls,
        additional_instruction="Look for explicit mentions of two or more distinct winter activities. Minor naming variations are acceptable (e.g., 'XC skiing' for cross-country skiing)."
    )

    acts_ref_leaf = evaluator.add_custom_node(
        result=bool(activities_urls),
        id="nearby_activity_reference",
        desc="Provide URL documenting available winter activities",
        parent=activity_ver_node,
        critical=True
    )

    # Pass_Acceptance -> Pass_Verification -> Pass_Accepted + Pass_Reference
    pass_node = evaluator.add_parallel(
        id="nearby_pass_acceptance",
        desc="Verify the destination accepts America the Beautiful pass for entry",
        parent=nearby_node,
        critical=True
    )
    pass_ver_node = evaluator.add_parallel(
        id="nearby_pass_verification",
        desc="Confirm America the Beautiful pass is accepted",
        parent=pass_node,
        critical=True
    )

    pass_accepted_leaf = evaluator.add_leaf(
        id="nearby_pass_accepted",
        desc="The destination accepts America the Beautiful pass for entry",
        parent=pass_ver_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"{name} accepts the America the Beautiful pass for entry.",
        node=pass_accepted_leaf,
        sources=pass_urls,
        additional_instruction="The evidence should indicate acceptance of the America the Beautiful (Interagency) Pass for entry or fees. Acceptance via a parent agency (e.g., National Park Service) is acceptable if clearly applicable."
    )

    pass_ref_leaf = evaluator.add_custom_node(
        result=bool(pass_urls),
        id="nearby_pass_reference",
        desc="Provide URL confirming pass acceptance",
        parent=pass_ver_node,
        critical=True
    )


async def verify_snowmobile(evaluator: Evaluator, parent, data: Optional[SnowmobileDestination]) -> None:
    snow_node = evaluator.add_parallel(
        id="snowmobile_destination",
        desc="Identify a western US destination offering winter snowmobiling with group permits",
        parent=parent,
        critical=False
    )

    name = (data.name if data and data.name else "the destination")
    location_urls = data.location_urls if data else []
    permit_urls = data.permit_urls if data else []
    group_urls = data.group_size_urls if data else []
    season_urls = data.season_urls if data else []
    fee_urls = data.fee_urls if data else []

    # Western_Location -> Location_Verification -> Western_US + Location_Reference
    loc_node = evaluator.add_parallel(
        id="snow_western_location",
        desc="Verify the destination is located in the western United States",
        parent=snow_node,
        critical=True
    )
    loc_ver_node = evaluator.add_parallel(
        id="snow_location_verification",
        desc="Confirm western US location",
        parent=loc_node,
        critical=True
    )

    western_leaf = evaluator.add_leaf(
        id="snow_western_us",
        desc="The destination is located in the western United States",
        parent=loc_ver_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"{name} is located in a western U.S. state.",
        node=western_leaf,
        sources=location_urls,
        additional_instruction="Accept the following states as Western US: WA, OR, CA, NV, AZ, NM, CO, UT, WY, MT, ID, AK. Verify the page establishes the destination's state in this list."
    )

    loc_ref_leaf = evaluator.add_custom_node(
        result=bool(location_urls),
        id="snow_location_reference",
        desc="Provide URL confirming the destination's location",
        parent=loc_ver_node,
        critical=True
    )

    # Permit_Program -> Permit_Availability -> Non_Commercial_Permits + Permit_Reference
    permit_node = evaluator.add_parallel(
        id="snow_permit_program",
        desc="Verify snowmobile access through permits for non-commercial groups",
        parent=snow_node,
        critical=True
    )
    permit_avail_node = evaluator.add_parallel(
        id="snow_permit_availability",
        desc="Confirm permit program exists for non-commercial groups",
        parent=permit_node,
        critical=True
    )

    non_comm_leaf = evaluator.add_leaf(
        id="snow_non_commercial_permits",
        desc="The destination offers snowmobile permits or access programs for non-commercial groups",
        parent=permit_avail_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"There is a permit or access program allowing non-commercial groups to access snowmobiling at {name}.",
        node=non_comm_leaf,
        sources=permit_urls,
        additional_instruction="Evidence should reference non-commercial groups, non-commercial guiding, or similar. General snowmobile info pages without permit details are insufficient."
    )

    permit_ref_leaf = evaluator.add_custom_node(
        result=bool(permit_urls),
        id="snow_permit_reference",
        desc="Provide URL documenting the snowmobile permit program",
        parent=permit_avail_node,
        critical=True
    )

    # Group_Accommodation -> Group_Size_Details -> Size_Policy + Size_Reference
    group_node = evaluator.add_parallel(
        id="snow_group_accommodation",
        desc="Verify the permit program can accommodate the group size",
        parent=snow_node,
        critical=True
    )
    group_details_node = evaluator.add_parallel(
        id="snow_group_size_details",
        desc="Document group size limits and accommodation for 11 people",
        parent=group_node,
        critical=True
    )

    size_policy_leaf = evaluator.add_leaf(
        id="snow_size_policy",
        desc="Document the maximum group size allowed per permit and whether multiple permits can accommodate 11 people",
        parent=group_details_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The permit rules document group size limits per permit and whether multiple permits or subgroups can accommodate a total of 11 people at {name}.",
        node=size_policy_leaf,
        sources=group_urls,
        additional_instruction="Look for explicit numeric limits (e.g., max riders per permit) and whether multiple permits or staggered entries are allowed. If the policy allows subgrouping to comply, this satisfies the requirement."
    )

    size_ref_leaf = evaluator.add_custom_node(
        result=bool(group_urls),
        id="snow_size_reference",
        desc="Provide URL confirming group size policies",
        parent=group_details_node,
        critical=True
    )

    # January_Accessibility -> Season_Verification -> January_Operation + Season_Reference
    jan_node = evaluator.add_parallel(
        id="snow_january_accessibility",
        desc="Verify the destination is accessible for snowmobiling in early January 2026",
        parent=snow_node,
        critical=True
    )
    season_ver_node = evaluator.add_parallel(
        id="snow_season_verification",
        desc="Confirm winter season includes early January dates",
        parent=jan_node,
        critical=True
    )

    jan_op_leaf = evaluator.add_leaf(
        id="snow_january_operation",
        desc="The snowmobile season includes early January 2026 dates",
        parent=season_ver_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"Snowmobile access at {name} is open in early January 2026 (e.g., the winter season explicitly includes early January).",
        node=jan_op_leaf,
        sources=season_urls,
        additional_instruction="Accept typical winter season windows (e.g., Dec–Mar) as long as early January is clearly within the dates. Temporary closures for conditions do not negate the general season."
    )

    season_ref_leaf = evaluator.add_custom_node(
        result=bool(season_urls),
        id="snow_season_reference",
        desc="Provide URL confirming winter season operating dates",
        parent=season_ver_node,
        critical=True
    )

    # Fee_Information (non-critical) -> Cost_Details (non-critical) -> Fee_Amount (non-critical) + Fee_Reference (critical)
    fee_node = evaluator.add_parallel(
        id="snow_fee_information",
        desc="Provide permit fee information",
        parent=snow_node,
        critical=False
    )
    cost_details_node = evaluator.add_parallel(
        id="snow_cost_details",
        desc="Document permit fees",
        parent=fee_node,
        critical=False
    )

    fee_amount_leaf = evaluator.add_leaf(
        id="snow_fee_amount",
        desc="Specific fee amounts for snowmobile permits are documented",
        parent=cost_details_node,
        critical=False
    )
    await evaluator.verify(
        claim="This page documents permit fees for snowmobile permits/access (e.g., dollar amounts for permits).",
        node=fee_amount_leaf,
        sources=fee_urls,
        additional_instruction="Look for explicit dollar amounts or clear fee tables that apply to snowmobile access permits for non-commercial groups."
    )

    fee_ref_leaf = evaluator.add_custom_node(
        result=bool(fee_urls),
        id="snow_fee_reference",
        desc="Provide URL for fee information",
        parent=cost_details_node,
        critical=True
    )


async def verify_camping(evaluator: Evaluator, parent, data: Optional[CampingDestination]) -> None:
    camp_node = evaluator.add_parallel(
        id="camping_destination",
        desc="Identify a destination with pet-friendly RV or tent camping available in early January",
        parent=parent,
        critical=False
    )

    name = (data.name if data and data.name else "the campground")
    pet_urls = data.pet_policy_urls if data else []
    reservation_urls = data.reservation_urls if data else []
    capacity_urls = data.capacity_urls if data else []
    ops_urls = data.operations_urls if data else []
    booking_window_text = (data.advance_booking_window if data and data.advance_booking_window else "an advance booking window")

    # Pet_Policy -> Pet_Allowance -> Pets_Permitted + Pet_Reference
    pet_node = evaluator.add_parallel(
        id="camp_pet_policy",
        desc="Verify pets are allowed in campsites",
        parent=camp_node,
        critical=True
    )
    pet_allow_node = evaluator.add_parallel(
        id="camp_pet_allowance",
        desc="Confirm pet-friendly camping policies",
        parent=pet_node,
        critical=True
    )

    pets_perm_leaf = evaluator.add_leaf(
        id="camp_pets_permitted",
        desc="Pets are explicitly permitted in camping areas (noting any restrictions such as exclusion from buildings)",
        parent=pet_allow_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"Pets are permitted in camping areas at {name}, subject to typical restrictions (e.g., leashes, no pets in buildings).",
        node=pets_perm_leaf,
        sources=pet_urls,
        additional_instruction="Evidence should clearly state pets are allowed in campsites (not just day-use). Policies about leashes or excluding buildings are acceptable nuances."
    )

    pet_ref_leaf = evaluator.add_custom_node(
        result=bool(pet_urls),
        id="camp_pet_reference",
        desc="Provide URL documenting pet policies",
        parent=pet_allow_node,
        critical=True
    )

    # Reservation_System -> Booking_Window -> Advance_Booking + Booking_Reference
    res_node = evaluator.add_parallel(
        id="camp_reservation_system",
        desc="Verify reservation system and advance booking window",
        parent=camp_node,
        critical=True
    )
    booking_node = evaluator.add_parallel(
        id="camp_booking_window",
        desc="Document advance reservation requirements",
        parent=res_node,
        critical=True
    )

    advance_leaf = evaluator.add_leaf(
        id="camp_advance_booking",
        desc="State the advance booking window (how far ahead reservations can be made)",
        parent=booking_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The reservation policy specifies {booking_window_text}.",
        node=advance_leaf,
        sources=reservation_urls,
        additional_instruction="Accept explicit windows like '6 months in advance', or calendar rules that clearly imply the advance booking window."
    )

    booking_ref_leaf = evaluator.add_custom_node(
        result=bool(reservation_urls),
        id="camp_booking_reference",
        desc="Provide URL for reservation policies and booking windows",
        parent=booking_node,
        critical=True
    )

    # Campsite_Capacity -> Equipment_Accommodation -> Site_Limits + Capacity_Reference
    cap_node = evaluator.add_parallel(
        id="camp_campsite_capacity",
        desc="Verify campsite can accommodate RV or tent camping equipment",
        parent=camp_node,
        critical=True
    )
    equip_node = evaluator.add_parallel(
        id="camp_equipment_accommodation",
        desc="Confirm site specifications meet camping needs",
        parent=cap_node,
        critical=True
    )

    site_limits_leaf = evaluator.add_leaf(
        id="camp_site_limits",
        desc="Campsites can accommodate RV or tent camping per the stated site limits (typically 1 RV/camping unit plus 1 tent per site)",
        parent=equip_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"Campsites at {name} can accommodate RV or tent camping within stated site limits.",
        node=site_limits_leaf,
        sources=capacity_urls,
        additional_instruction="Look for explicit site specification pages indicating RV/tent allowance and any per-site capacity rules (e.g., one camping unit plus one tent)."
    )

    capacity_ref_leaf = evaluator.add_custom_node(
        result=bool(capacity_urls),
        id="camp_capacity_reference",
        desc="Provide URL for campsite specifications",
        parent=equip_node,
        critical=True
    )

    # Winter_Availability -> January_Operations -> Open_January + Operations_Reference
    ops_node = evaluator.add_parallel(
        id="camp_winter_availability",
        desc="Verify the campground is open in early January 2026",
        parent=camp_node,
        critical=True
    )
    jan_ops_node = evaluator.add_parallel(
        id="camp_january_operations",
        desc="Confirm campground operates during early January",
        parent=ops_node,
        critical=True
    )

    open_jan_leaf = evaluator.add_leaf(
        id="camp_open_january",
        desc="The campground is open and accepting reservations during early January 2026",
        parent=jan_ops_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"{name} is open and accepting reservations during early January 2026.",
        node=open_jan_leaf,
        sources=ops_urls,
        additional_instruction="Accept explicit winter/operating season pages or year-round operation statements. Temporary weather-dependent closures do not negate general availability."
    )

    ops_ref_leaf = evaluator.add_custom_node(
        result=bool(ops_urls),
        id="camp_operations_reference",
        desc="Provide URL confirming winter operating season or year-round operation",
        parent=jan_ops_node,
        critical=True
    )


async def verify_ski(evaluator: Evaluator, parent, data: Optional[SkiDestination]) -> None:
    ski_node = evaluator.add_parallel(
        id="ski_destination",
        desc="Identify a ski resort suitable for families with children ages 4-14",
        parent=parent,
        critical=False
    )

    name = (data.name if data and data.name else "the resort")
    youth_urls = data.youth_programs_urls if data else []
    lodging_urls = data.lodging_urls if data else []
    pricing_urls = data.ticket_pricing_urls if data else []
    group_urls = data.group_policies_urls if data else []

    # Family_Activities -> Children_Programs -> Youth_Suitability + Programs_Reference
    fam_act_node = evaluator.add_parallel(
        id="ski_family_activities",
        desc="Verify skiing activities or programs suitable for children ages 4-14",
        parent=ski_node,
        critical=True
    )
    child_prog_node = evaluator.add_parallel(
        id="ski_children_programs",
        desc="Confirm availability of children's skiing programs",
        parent=fam_act_node,
        critical=True
    )

    youth_suit_leaf = evaluator.add_leaf(
        id="ski_youth_suitability",
        desc="The resort offers skiing activities, lessons, or programs suitable for children ages 4-14",
        parent=child_prog_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"{name} offers skiing lessons or programs suitable for children ages 4–14.",
        node=youth_suit_leaf,
        sources=youth_urls,
        additional_instruction="Evidence should include children's lessons, ski school with age requirements allowing ages ~4–14, or family programs suited to this age range."
    )

    programs_ref_leaf = evaluator.add_custom_node(
        result=bool(youth_urls),
        id="ski_programs_reference",
        desc="Provide URL documenting family or children's programs",
        parent=child_prog_node,
        critical=True
    )

    # Lodging_Options -> Multi_Family_Lodging -> Family_Accommodations + Lodging_Reference
    lodge_node = evaluator.add_parallel(
        id="ski_lodging_options",
        desc="Verify lodging suitable for three families",
        parent=ski_node,
        critical=True
    )
    multi_family_node = evaluator.add_parallel(
        id="ski_multi_family_lodging",
        desc="Confirm appropriate accommodation types are available",
        parent=lodge_node,
        critical=True
    )

    family_accom_leaf = evaluator.add_leaf(
        id="ski_family_accommodations",
        desc="The resort offers lodging suitable for 3 families such as multiple hotel rooms, condos, or cabins",
        parent=multi_family_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"{name} offers lodging options suitable for three families (e.g., multiple rooms, condos, cabins).",
        node=family_accom_leaf,
        sources=lodging_urls,
        additional_instruction="Look for on-resort or partner lodging that can accommodate multiple families (multi-bedroom condos, townhomes, cabins, or availability of multiple rooms)."
    )

    lodging_ref_leaf = evaluator.add_custom_node(
        result=bool(lodging_urls),
        id="ski_lodging_reference",
        desc="Provide URL for lodging options and types",
        parent=multi_family_node,
        critical=True
    )

    # Lift_Pricing (non-critical) -> Ticket_Pricing (non-critical) -> Price_Documentation (non-critical) + Pricing_Reference (critical)
    pricing_node = evaluator.add_parallel(
        id="ski_lift_pricing",
        desc="Provide lift ticket pricing information",
        parent=ski_node,
        critical=False
    )
    ticket_pricing_node = evaluator.add_parallel(
        id="ski_ticket_pricing",
        desc="Document available lift ticket types and prices",
        parent=pricing_node,
        critical=False
    )

    price_doc_leaf = evaluator.add_leaf(
        id="ski_price_documentation",
        desc="Available lift ticket types and pricing are documented",
        parent=ticket_pricing_node,
        critical=False
    )
    await evaluator.verify(
        claim="This page documents lift ticket types and prices.",
        node=price_doc_leaf,
        sources=pricing_urls,
        additional_instruction="Accept day tickets, half-day, child/adult pricing tables, or season-specific ticket pages that list prices."
    )

    pricing_ref_leaf = evaluator.add_custom_node(
        result=bool(pricing_urls),
        id="ski_pricing_reference",
        desc="Provide URL for lift ticket pricing",
        parent=ticket_pricing_node,
        critical=True
    )

    # Group_Policies (non-critical) -> Group_Discount_Info (non-critical) -> Minimum_Requirements (non-critical) + Group_Reference (non-critical)
    group_node = evaluator.add_parallel(
        id="ski_group_policies",
        desc="Note group booking policies",
        parent=ski_node,
        critical=False
    )
    group_info_node = evaluator.add_parallel(
        id="ski_group_discount_info",
        desc="Document group discount requirements if applicable",
        parent=group_node,
        critical=False
    )

    min_req_leaf = evaluator.add_leaf(
        id="ski_minimum_requirements",
        desc="Note that this group of 11 people is below typical 20-person group minimums, if applicable",
        parent=group_info_node,
        critical=False
    )
    await evaluator.verify(
        claim="Group booking policies typically require a minimum around 15–20 people; thus, a group of 11 is often below the group discount minimum if a minimum is stated above 11.",
        node=min_req_leaf,
        sources=group_urls,
        additional_instruction="Accept if the page states a group minimum ≥ 12 and especially ≥ 15 or 20, implying 11 is below the threshold. If no group minimum is documented, do not support the claim."
    )

    group_ref_leaf = evaluator.add_custom_node(
        result=bool(group_urls),
        id="ski_group_reference",
        desc="Provide URL for group policies if available",
        parent=group_info_node,
        critical=False
    )


# --------------------------------------------------------------------------- #
# Main evaluation function                                                    #
# --------------------------------------------------------------------------- #
async def evaluate_answer(
    client: LLMClient,
    answer: str,
    agent_name: str,
    answer_name: str,
    cache: CacheFileSys,
    semaphore: asyncio.Semaphore,
    logger: logging.Logger,
    model: str = "o4-mini"
) -> Dict:
    """
    Evaluate an answer for the winter trip planning task. Builds a hierarchical verification
    tree and verifies each requirement using cited web sources when provided.
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
        default_model=model
    )

    # Top-level planning node (use non-critical to allow partial credit across categories)
    top_node = evaluator.add_parallel(
        id="winter_trip_planning",
        desc="Plan a comprehensive multi-destination winter outdoor recreation trip for 3 families totaling 11 people from Bangor, Maine in early January 2026",
        parent=root,
        critical=False
    )

    # Extract structured data
    extracted = await evaluator.extract(
        prompt=prompt_extract_winter_trip(),
        template_class=WinterTripExtraction,
        extraction_name="winter_trip_extraction"
    )

    # Build and verify each destination subtree
    await verify_nearby(evaluator, top_node, extracted.nearby)
    await verify_snowmobile(evaluator, top_node, extracted.snowmobile)
    await verify_camping(evaluator, top_node, extracted.camping)
    await verify_ski(evaluator, top_node, extracted.ski)

    return evaluator.get_summary()