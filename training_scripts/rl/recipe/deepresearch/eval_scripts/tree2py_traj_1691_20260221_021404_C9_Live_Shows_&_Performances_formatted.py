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
TASK_ID = "tour_venue_suitability"
TASK_DESCRIPTION = (
    "A major touring artist is planning a multi-city arena tour across the United States and needs to identify suitable venues in four different geographic regions. "
    "For each of the following regions, identify one major indoor arena concert venue that meets all the specified requirements:\n\n"
    "Regions:\n"
    "1. Northeast US (NY, NJ, PA, CT, MA, or nearby states)\n"
    "2. Southeast US (FL, GA, NC, SC, TN, or nearby states)\n"
    "3. Southwest US (TX, AZ, NM, OK, or nearby states)\n"
    "4. West US (CA, WA, OR, NV, or nearby states)\n\n"
    "Requirements for each venue:\n\n"
    "Capacity & Safety:\n"
    "- Minimum concert seating capacity of 15,000\n"
    "- Wheelchair accessible seating meeting ADA requirements (minimum 1% of total capacity)\n"
    "- Main entrance capable of accommodating at least 2/3 of total capacity for emergency evacuation per fire safety standards\n"
    "- General liability insurance with minimum $1 million coverage\n\n"
    "Technical Infrastructure:\n"
    "- 200-400 amp three-phase electrical service for concert production equipment\n"
    "- Stage depth of at least 16 feet to accommodate a full band with drums\n"
    "- Professional sound system support with adequate PA mounting infrastructure\n"
    "- Acoustic characteristics suitable for rock/pop concerts (reverberation time 0.6-1.2 seconds preferred)\n\n"
    "Logistical Requirements:\n"
    "- Available for booking with 3-6 months advance notice\n"
    "- Minimum 4 hours load-in time allowed before performance\n"
    "- Load-out access provided until at least 5 AM following the performance\n"
    "- Loading dock accommodating 53-foot semi-trailers\n\n"
    "Backstage Facilities:\n"
    "- Minimum 3 dressing rooms (star, band, crew)\n"
    "- Star dressing room at least 300 square feet\n"
    "- Green room with catering area and refrigeration\n"
    "- Audio/video stage monitors in dressing rooms (preferred)\n\n"
    "For each identified venue, provide:\n"
    "- The venue name and location (city, state)\n"
    "- Specific confirmation of how each requirement is met\n"
    "- Reference URLs supporting the venue's specifications and capabilities"
)

REGION_STATES: Dict[str, List[str]] = {
    "NE": ["NY", "NJ", "PA", "CT", "MA", "RI", "NH", "VT", "ME"],
    "SE": ["FL", "GA", "NC", "SC", "TN", "AL", "MS", "VA"],
    "SW": ["TX", "AZ", "NM", "OK"],
    "W": ["CA", "WA", "OR", "NV", "ID", "UT", "MT", "WY", "CO"]
}

# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class CapacitySafety(BaseModel):
    capacity: Optional[str] = None
    capacity_sources: List[str] = Field(default_factory=list)

    wheelchair_spaces: Optional[str] = None
    companion_seats: Optional[str] = None
    ada_sources: List[str] = Field(default_factory=list)

    emergency_egress: Optional[str] = None
    liability_insurance: Optional[str] = None
    safety_sources: List[str] = Field(default_factory=list)


class Technical(BaseModel):
    power_supply: Optional[str] = None
    power_sources: List[str] = Field(default_factory=list)

    stage_depth: Optional[str] = None
    sound_system: Optional[str] = None
    stage_sources: List[str] = Field(default_factory=list)

    acoustics_rt: Optional[str] = None  # reverberation time or description
    acoustics_sources: List[str] = Field(default_factory=list)


class Logistical(BaseModel):
    booking_timeline: Optional[str] = None
    booking_sources: List[str] = Field(default_factory=list)

    load_in_time: Optional[str] = None
    load_out_access: Optional[str] = None
    truck_access: Optional[str] = None
    logistics_sources: List[str] = Field(default_factory=list)


class Backstage(BaseModel):
    dressing_room_count: Optional[str] = None
    star_room_size: Optional[str] = None
    dressing_sources: List[str] = Field(default_factory=list)

    green_room_facilities: Optional[str] = None
    green_room_sources: List[str] = Field(default_factory=list)

    stage_monitors: Optional[str] = None
    monitors_sources: List[str] = Field(default_factory=list)


class Venue(BaseModel):
    venue_name: Optional[str] = None
    city: Optional[str] = None
    state: Optional[str] = None
    website_url: Optional[str] = None  # if provided by the answer

    capacity_safety: CapacitySafety = Field(default_factory=CapacitySafety)
    technical: Technical = Field(default_factory=Technical)
    logistical: Logistical = Field(default_factory=Logistical)
    backstage: Backstage = Field(default_factory=Backstage)


class TourVenuesExtraction(BaseModel):
    northeast: Optional[Venue] = None
    southeast: Optional[Venue] = None
    southwest: Optional[Venue] = None
    west: Optional[Venue] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_venues() -> str:
    return """
    Extract one major indoor arena concert venue for each of the four regions (Northeast, Southeast, Southwest, West) as presented in the answer.
    For each region, return a structured JSON object with the following fields:

    - venue_name: The full name of the arena venue.
    - city: The city where the venue is located.
    - state: The state abbreviation (e.g., NY, CA).
    - website_url: If the answer provides an official or primary venue page URL, include it; otherwise null.

    Capacity & Safety (capacity_safety):
    - capacity: The concert seating capacity as stated (string; include numbers or ranges exactly as written).
    - capacity_sources: Array of URLs supporting capacity claims.
    - wheelchair_spaces: The statement or metric regarding wheelchair accessible seating (string; include explicit % or descriptions if available).
    - companion_seats: The statement confirming adjacent companion seats for wheelchair spaces (string).
    - ada_sources: Array of URLs supporting ADA accessibility statements.
    - emergency_egress: The statement about main entrance egress (supporting evacuation of at least 2/3 capacity) (string).
    - liability_insurance: The statement confirming general liability insurance minimum $1M coverage (string).
    - safety_sources: Array of URLs supporting emergency egress and insurance statements.

    Technical Infrastructure (technical):
    - power_supply: The power service statement (must mention 200–400 amp three-phase service) (string).
    - power_sources: Array of URLs supporting power specifications.
    - stage_depth: The stage depth statement (should be at least 16 feet) (string).
    - sound_system: The statement confirming professional sound system/PA mounting support (string).
    - stage_sources: Array of URLs supporting stage and sound system specs.
    - acoustics_rt: The acoustics statement (preferably reverberation time: 0.6–1.2 seconds) (string).
    - acoustics_sources: Array of URLs supporting acoustics.

    Logistical Requirements (logistical):
    - booking_timeline: Statement that booking can be made 3–6 months in advance (string).
    - booking_sources: Array of URLs supporting booking timeline.
    - load_in_time: Statement that minimum 4 hours load-in is allowed (string).
    - load_out_access: Statement that load-out access is provided until at least 5 AM following performance (string).
    - truck_access: Statement that loading dock accommodates 53-foot semi-trailers (string).
    - logistics_sources: Array of URLs supporting load-in/out and truck access.

    Backstage Facilities (backstage):
    - dressing_room_count: Statement confirming at least 3 dressing rooms (star, band, crew) (string).
    - star_room_size: Statement confirming star dressing room ≥ 300 sq ft (string).
    - dressing_sources: Array of URLs supporting dressing rooms specs.
    - green_room_facilities: Statement confirming green room with catering area and refrigeration (string).
    - green_room_sources: Array of URLs supporting green room facilities.
    - stage_monitors: Statement that dressing rooms have audio/video stage monitors (string).
    - monitors_sources: Array of URLs supporting monitors statement.

    Regions to extract:
    - northeast: A venue in NY, NJ, PA, CT, MA, RI, NH, VT, or ME (or nearby if answer claims).
    - southeast: A venue in FL, GA, NC, SC, TN, AL, MS, or VA (or nearby if answer claims).
    - southwest: A venue in TX, AZ, NM, or OK (or nearby if answer claims).
    - west: A venue in CA, WA, OR, NV, ID, UT, MT, WY, or CO (or nearby if answer claims).

    Return a JSON object with keys: northeast, southeast, southwest, west. If any region is missing in the answer, set that region to null. For any missing subfield, set it to null or [] accordingly. Extract URLs exactly as provided in the answer (plain URLs or markdown links).
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def venue_label(v: Venue, region_tag: str) -> str:
    name = v.venue_name or "Unknown Venue"
    city = v.city or "Unknown City"
    state = v.state or "Unknown State"
    return f"{name} ({city}, {state}) [{region_tag}]"


def nonempty_urls(urls: Optional[List[str]]) -> List[str]:
    return [u for u in (urls or []) if isinstance(u, str) and u.strip()]


COMMON_VERIFY_INS = (
    "Focus on whether the claim is explicitly supported by the provided URL(s). "
    "Allow reasonable wording variants and minor numeric rounding. "
    "Use the screenshot and page text to locate the relevant specifications."
)


# --------------------------------------------------------------------------- #
# Region verification builder                                                 #
# --------------------------------------------------------------------------- #
async def build_region_verification(
    evaluator: Evaluator,
    parent_node,
    region_key: str,
    region_desc: str,
    venue: Optional[Venue],
) -> None:
    """
    Build verification subtree for one region.
    """
    # Region aggregator (non-critical to allow partial credit across sub-categories)
    region_node = evaluator.add_parallel(
        id=f"{region_key}_Region_Venue",
        desc=f"A suitable major arena venue identified in the {region_desc} that meets all technical, safety, logistical, and hospitality requirements",
        parent=parent_node,
        critical=False
    )

    # Existence gate: require name/location present
    exists_result = bool(venue and venue.venue_name and venue.city and venue.state)
    exists_node = evaluator.add_custom_node(
        result=exists_result,
        id=f"{region_key}_Venue_Identified",
        desc=f"{region_desc} venue identified with name and location",
        parent=region_node,
        critical=True
    )

    # If no venue provided, still build structure; leaf verifications will be skipped via prerequisites
    v = venue or Venue()

    # ------------------------- Capacity & Safety -------------------------- #
    capsafe_node = evaluator.add_parallel(
        id=f"{region_key}_Capacity_Safety",
        desc=f"The {region_desc} venue meets capacity and safety requirements",
        parent=region_node,
        critical=True
    )

    # Seating capacity sub-node
    capacity_node = evaluator.add_parallel(
        id=f"{region_key}_Seating_Capacity",
        desc="The venue has a concert seating capacity of at least 15,000",
        parent=capsafe_node,
        critical=True
    )
    # Capacity reference existence
    cap_ref_node = evaluator.add_custom_node(
        result=bool(nonempty_urls(v.capacity_safety.capacity_sources)),
        id=f"{region_key}_Capacity_Reference",
        desc=f"URL reference provided for {region_desc.lower()} venue capacity specifications",
        parent=capacity_node,
        critical=True
    )
    # Capacity value verification
    cap_val_leaf = evaluator.add_leaf(
        id=f"{region_key}_Capacity_Value",
        desc="Verified capacity number meets or exceeds 15,000",
        parent=capacity_node,
        critical=True
    )
    cap_claim = (
        f"The concert seating capacity of {venue_label(v, region_key)} is '{v.capacity_safety.capacity}', "
        "which meets or exceeds 15,000."
    )
    await evaluator.verify(
        claim=cap_claim,
        node=cap_val_leaf,
        sources=nonempty_urls(v.capacity_safety.capacity_sources),
        additional_instruction=COMMON_VERIFY_INS,
        extra_prerequisites=[exists_node, cap_ref_node]
    )

    # ADA compliance sub-node
    ada_node = evaluator.add_parallel(
        id=f"{region_key}_ADA_Compliance",
        desc="The venue provides wheelchair accessible seating and companion seats meeting ADA requirements",
        parent=capsafe_node,
        critical=True
    )
    ada_ref_node = evaluator.add_custom_node(
        result=bool(nonempty_urls(v.capacity_safety.ada_sources)),
        id=f"{region_key}_ADA_Reference",
        desc=f"URL reference provided for {region_desc.lower()} venue ADA accessibility specifications",
        parent=ada_node,
        critical=True
    )
    # Wheelchair spaces >=1%
    wc_leaf = evaluator.add_leaf(
        id=f"{region_key}_Wheelchair_Spaces",
        desc="Wheelchair accessible seating is at least 1% of total capacity",
        parent=ada_node,
        critical=True
    )
    wc_claim = (
        f"{venue_label(v, region_key)} provides wheelchair accessible seating meeting ADA minimum (at least 1% of capacity). "
        f"Stated detail: '{v.capacity_safety.wheelchair_spaces}'."
    )
    await evaluator.verify(
        claim=wc_claim,
        node=wc_leaf,
        sources=nonempty_urls(v.capacity_safety.ada_sources),
        additional_instruction=COMMON_VERIFY_INS,
        extra_prerequisites=[exists_node, ada_ref_node]
    )
    # Companion seats
    comp_leaf = evaluator.add_leaf(
        id=f"{region_key}_Companion_Seats",
        desc="Each wheelchair space has an adjacent companion seat as required by ADA",
        parent=ada_node,
        critical=True
    )
    comp_claim = (
        f"{venue_label(v, region_key)} provides an adjacent companion seat for each wheelchair space per ADA. "
        f"Stated detail: '{v.capacity_safety.companion_seats}'."
    )
    await evaluator.verify(
        claim=comp_claim,
        node=comp_leaf,
        sources=nonempty_urls(v.capacity_safety.ada_sources),
        additional_instruction=COMMON_VERIFY_INS,
        extra_prerequisites=[exists_node, ada_ref_node]
    )

    # Emergency safety sub-node
    safety_node = evaluator.add_parallel(
        id=f"{region_key}_Emergency_Safety",
        desc="The venue meets fire safety and emergency egress requirements",
        parent=capsafe_node,
        critical=True
    )
    safety_ref_node = evaluator.add_custom_node(
        result=bool(nonempty_urls(v.capacity_safety.safety_sources)),
        id=f"{region_key}_Safety_Reference",
        desc=f"URL reference provided for {region_desc.lower()} venue safety specifications",
        parent=safety_node,
        critical=True
    )
    # Emergency egress
    egress_leaf = evaluator.add_leaf(
        id=f"{region_key}_Emergency_Egress",
        desc="Main entrance can accommodate at least 2/3 of total capacity for emergency evacuation",
        parent=safety_node,
        critical=True
    )
    egress_claim = (
        f"The main entrance of {venue_label(v, region_key)} can accommodate at least 2/3 of total capacity for emergency evacuation. "
        f"Stated detail: '{v.capacity_safety.emergency_egress}'."
    )
    await evaluator.verify(
        claim=egress_claim,
        node=egress_leaf,
        sources=nonempty_urls(v.capacity_safety.safety_sources),
        additional_instruction=COMMON_VERIFY_INS,
        extra_prerequisites=[exists_node, safety_ref_node]
    )
    # Liability insurance
    insurance_leaf = evaluator.add_leaf(
        id=f"{region_key}_Liability_Insurance",
        desc="Venue maintains general liability insurance with minimum $1 million coverage",
        parent=safety_node,
        critical=True
    )
    ins_claim = (
        f"{venue_label(v, region_key)} maintains general liability insurance of at least $1,000,000 coverage. "
        f"Stated detail: '{v.capacity_safety.liability_insurance}'."
    )
    await evaluator.verify(
        claim=ins_claim,
        node=insurance_leaf,
        sources=nonempty_urls(v.capacity_safety.safety_sources),
        additional_instruction=COMMON_VERIFY_INS,
        extra_prerequisites=[exists_node, safety_ref_node]
    )

    # ---------------------- Technical Infrastructure ---------------------- #
    tech_node = evaluator.add_parallel(
        id=f"{region_key}_Technical_Infrastructure",
        desc=f"The {region_desc} venue meets technical infrastructure requirements for concert production",
        parent=region_node,
        critical=True
    )

    # Power supply
    power_node = evaluator.add_parallel(
        id=f"{region_key}_Power_Supply",
        desc="The venue provides 200-400 amp three-phase electrical service for concert production equipment",
        parent=tech_node,
        critical=True
    )
    power_ref_node = evaluator.add_custom_node(
        result=bool(nonempty_urls(v.technical.power_sources)),
        id=f"{region_key}_Power_Reference",
        desc="URL reference provided for power specifications",
        parent=power_node,
        critical=True
    )
    power_leaf = evaluator.add_leaf(
        id=f"{region_key}_Power_Specification",
        desc="Electrical service meets 200-400 amp three-phase requirement",
        parent=power_node,
        critical=True
    )
    power_claim = (
        f"{venue_label(v, region_key)} provides 200–400 amp three-phase electrical service suitable for concert production. "
        f"Stated detail: '{v.technical.power_supply}'."
    )
    await evaluator.verify(
        claim=power_claim,
        node=power_leaf,
        sources=nonempty_urls(v.technical.power_sources),
        additional_instruction=COMMON_VERIFY_INS,
        extra_prerequisites=[exists_node, power_ref_node]
    )

    # Stage requirements
    stage_node = evaluator.add_parallel(
        id=f"{region_key}_Stage_Requirements",
        desc="The venue's stage meets dimensional and technical requirements",
        parent=tech_node,
        critical=True
    )
    stage_ref_node = evaluator.add_custom_node(
        result=bool(nonempty_urls(v.technical.stage_sources)),
        id=f"{region_key}_Stage_Reference",
        desc="URL reference provided for stage specifications",
        parent=stage_node,
        critical=True
    )
    # Stage depth
    stage_depth_leaf = evaluator.add_leaf(
        id=f"{region_key}_Stage_Dimensions",
        desc="Stage is at least 16 feet deep to accommodate a full band with drums",
        parent=stage_node,
        critical=True
    )
    stage_depth_claim = (
        f"{venue_label(v, region_key)} has a stage depth of at least 16 feet. "
        f"Stated detail: '{v.technical.stage_depth}'."
    )
    await evaluator.verify(
        claim=stage_depth_claim,
        node=stage_depth_leaf,
        sources=nonempty_urls(v.technical.stage_sources),
        additional_instruction=COMMON_VERIFY_INS,
        extra_prerequisites=[exists_node, stage_ref_node]
    )
    # Sound system support
    sound_leaf = evaluator.add_leaf(
        id=f"{region_key}_Sound_System",
        desc="Venue supports professional sound system installation with adequate PA mounting infrastructure",
        parent=stage_node,
        critical=True
    )
    sound_claim = (
        f"{venue_label(v, region_key)} supports professional sound system installation with adequate PA mounting infrastructure. "
        f"Stated detail: '{v.technical.sound_system}'."
    )
    await evaluator.verify(
        claim=sound_claim,
        node=sound_leaf,
        sources=nonempty_urls(v.technical.stage_sources),
        additional_instruction=COMMON_VERIFY_INS,
        extra_prerequisites=[exists_node, stage_ref_node]
    )

    # Acoustics (optional / non-critical)
    if v.technical.acoustics_rt or nonempty_urls(v.technical.acoustics_sources):
        acoust_leaf = evaluator.add_leaf(
            id=f"{region_key}_Acoustics",
            desc="The venue has acoustic characteristics suitable for rock/pop concerts (reverberation time 0.6-1.2 seconds)",
            parent=region_node,  # place outside critical technical node to keep it non-critical
            critical=False
        )
        acoust_claim = (
            f"{venue_label(v, region_key)} has acoustics suitable for rock/pop concerts; preferred RT 0.6–1.2 seconds. "
            f"Stated detail: '{v.technical.acoustics_rt}'."
        )
        await evaluator.verify(
            claim=acoust_claim,
            node=acoust_leaf,
            sources=nonempty_urls(v.technical.acoustics_sources),
            additional_instruction=COMMON_VERIFY_INS,
            extra_prerequisites=[exists_node]
        )

    # ---------------------- Logistical Requirements ----------------------- #
    log_node = evaluator.add_parallel(
        id=f"{region_key}_Logistical_Requirements",
        desc=f"The {region_desc} venue meets logistical and operational requirements",
        parent=region_node,
        critical=True
    )

    # Booking timeline
    booking_node = evaluator.add_parallel(
        id=f"{region_key}_Booking_Timeline",
        desc="The venue can be booked with 3-6 months advance notice",
        parent=log_node,
        critical=True
    )
    booking_ref_node = evaluator.add_custom_node(
        result=bool(nonempty_urls(v.logistical.booking_sources)),
        id=f"{region_key}_Booking_Reference",
        desc="URL reference provided for booking information",
        parent=booking_node,
        critical=True
    )
    booking_leaf = evaluator.add_leaf(
        id=f"{region_key}_Booking_Availability",
        desc="Booking timeline meets 3-6 months advance requirement",
        parent=booking_node,
        critical=True
    )
    booking_claim = (
        f"{venue_label(v, region_key)} can be booked 3–6 months in advance. "
        f"Stated detail: '{v.logistical.booking_timeline}'."
    )
    await evaluator.verify(
        claim=booking_claim,
        node=booking_leaf,
        sources=nonempty_urls(v.logistical.booking_sources),
        additional_instruction=COMMON_VERIFY_INS,
        extra_prerequisites=[exists_node, booking_ref_node]
    )

    # Load operations
    loadops_node = evaluator.add_parallel(
        id=f"{region_key}_Load_Operations",
        desc="The venue provides adequate load-in and load-out access",
        parent=log_node,
        critical=True
    )
    log_ref_node = evaluator.add_custom_node(
        result=bool(nonempty_urls(v.logistical.logistics_sources)),
        id=f"{region_key}_Logistics_Reference",
        desc="URL reference provided for logistical information",
        parent=loadops_node,
        critical=True
    )
    # Load-in time
    loadin_leaf = evaluator.add_leaf(
        id=f"{region_key}_Load_In_Time",
        desc="Venue allows minimum 4 hours load-in time before performance",
        parent=loadops_node,
        critical=True
    )
    loadin_claim = (
        f"{venue_label(v, region_key)} allows a minimum of 4 hours load-in time before performance. "
        f"Stated detail: '{v.logistical.load_in_time}'."
    )
    await evaluator.verify(
        claim=loadin_claim,
        node=loadin_leaf,
        sources=nonempty_urls(v.logistical.logistics_sources),
        additional_instruction=COMMON_VERIFY_INS,
        extra_prerequisites=[exists_node, log_ref_node]
    )

    # Load-out access
    loadout_leaf = evaluator.add_leaf(
        id=f"{region_key}_Load_Out_Access",
        desc="Venue provides load-out access until at least 5 AM following performance",
        parent=loadops_node,
        critical=True
    )
    loadout_claim = (
        f"{venue_label(v, region_key)} provides load-out access until at least 5 AM following performance. "
        f"Stated detail: '{v.logistical.load_out_access}'."
    )
    await evaluator.verify(
        claim=loadout_claim,
        node=loadout_leaf,
        sources=nonempty_urls(v.logistical.logistics_sources),
        additional_instruction=COMMON_VERIFY_INS,
        extra_prerequisites=[exists_node, log_ref_node]
    )

    # Truck access
    truck_leaf = evaluator.add_leaf(
        id=f"{region_key}_Truck_Access",
        desc="Loading dock accommodates 53-foot semi-trailers",
        parent=loadops_node,
        critical=True
    )
    truck_claim = (
        f"The loading dock at {venue_label(v, region_key)} accommodates 53-foot semi-trailers. "
        f"Stated detail: '{v.logistical.truck_access}'."
    )
    await evaluator.verify(
        claim=truck_claim,
        node=truck_leaf,
        sources=nonempty_urls(v.logistical.logistics_sources),
        additional_instruction=COMMON_VERIFY_INS,
        extra_prerequisites=[exists_node, log_ref_node]
    )

    # ------------------------ Backstage Facilities ------------------------ #
    back_node = evaluator.add_parallel(
        id=f"{region_key}_Backstage_Facilities",
        desc=f"The {region_desc} venue meets backstage and hospitality requirements",
        parent=region_node,
        critical=True
    )

    # Dressing rooms
    dress_node = evaluator.add_parallel(
        id=f"{region_key}_Dressing_Rooms",
        desc="The venue provides adequate dressing room facilities",
        parent=back_node,
        critical=True
    )
    dress_ref_node = evaluator.add_custom_node(
        result=bool(nonempty_urls(v.backstage.dressing_sources)),
        id=f"{region_key}_Dressing_Reference",
        desc="URL reference provided for dressing room specifications",
        parent=dress_node,
        critical=True
    )
    # Count
    dress_count_leaf = evaluator.add_leaf(
        id=f"{region_key}_Dressing_Room_Count",
        desc="Minimum 3 dressing rooms provided (star, band, crew)",
        parent=dress_node,
        critical=True
    )
    dress_count_claim = (
        f"{venue_label(v, region_key)} provides at least 3 dressing rooms (star, band, crew). "
        f"Stated detail: '{v.backstage.dressing_room_count}'."
    )
    await evaluator.verify(
        claim=dress_count_claim,
        node=dress_count_leaf,
        sources=nonempty_urls(v.backstage.dressing_sources),
        additional_instruction=COMMON_VERIFY_INS,
        extra_prerequisites=[exists_node, dress_ref_node]
    )
    # Star room size
    star_room_leaf = evaluator.add_leaf(
        id=f"{region_key}_Star_Room_Size",
        desc="Star dressing room is at least 300 square feet",
        parent=dress_node,
        critical=True
    )
    star_room_claim = (
        f"The star dressing room at {venue_label(v, region_key)} is at least 300 square feet. "
        f"Stated detail: '{v.backstage.star_room_size}'."
    )
    await evaluator.verify(
        claim=star_room_claim,
        node=star_room_leaf,
        sources=nonempty_urls(v.backstage.dressing_sources),
        additional_instruction=COMMON_VERIFY_INS,
        extra_prerequisites=[exists_node, dress_ref_node]
    )

    # Green room
    green_node = evaluator.add_parallel(
        id=f"{region_key}_Green_Room",
        desc="The venue has a green room with catering area and refrigeration",
        parent=back_node,
        critical=True
    )
    green_ref_node = evaluator.add_custom_node(
        result=bool(nonempty_urls(v.backstage.green_room_sources)),
        id=f"{region_key}_Facilities_Reference",
        desc="URL reference provided for green room specifications",
        parent=green_node,
        critical=True
    )
    green_leaf = evaluator.add_leaf(
        id=f"{region_key}_Green_Room_Facilities",
        desc="Green room includes catering area and refrigeration",
        parent=green_node,
        critical=True
    )
    green_claim = (
        f"{venue_label(v, region_key)} has a green room with catering area and refrigeration. "
        f"Stated detail: '{v.backstage.green_room_facilities}'."
    )
    await evaluator.verify(
        claim=green_claim,
        node=green_leaf,
        sources=nonempty_urls(v.backstage.green_room_sources),
        additional_instruction=COMMON_VERIFY_INS,
        extra_prerequisites=[exists_node, green_ref_node]
    )

    # Stage monitors (optional)
    if v.backstage.stage_monitors or nonempty_urls(v.backstage.monitors_sources):
        monitors_leaf = evaluator.add_leaf(
            id=f"{region_key}_Stage_Monitors",
            desc="All dressing rooms have audio/video stage monitors installed",
            parent=region_node,  # place outside critical backstage node to keep it non-critical
            critical=False
        )
        monitors_claim = (
            f"Dressing rooms at {venue_label(v, region_key)} have audio/video stage monitors installed. "
            f"Stated detail: '{v.backstage.stage_monitors}'."
        )
        await evaluator.verify(
            claim=monitors_claim,
            node=monitors_leaf,
            sources=nonempty_urls(v.backstage.monitors_sources),
            additional_instruction=COMMON_VERIFY_INS,
            extra_prerequisites=[exists_node]
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
    model: str = "o4-mini"
) -> Dict:
    """
    Evaluate an answer for the tour venue suitability task across four US regions.
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

    # Top-level node for the tour venues (non-critical to allow partial scoring across regions)
    tour_node = evaluator.add_parallel(
        id="Tour_Venue_Suitability",
        desc="Evaluation of whether the identified concert venues meet all requirements for a major multi-city tour across four US regions",
        parent=root,
        critical=False
    )

    # Extract all venues info
    extracted = await evaluator.extract(
        prompt=prompt_extract_venues(),
        template_class=TourVenuesExtraction,
        extraction_name="tour_venues_extraction",
    )

    # Build verification for each region
    await build_region_verification(
        evaluator, tour_node, "NE", "Northeast US", extracted.northeast
    )
    await build_region_verification(
        evaluator, tour_node, "SE", "Southeast US", extracted.southeast
    )
    await build_region_verification(
        evaluator, tour_node, "SW", "Southwest US", extracted.southwest
    )
    await build_region_verification(
        evaluator, tour_node, "W", "West US", extracted.west
    )

    # Return summary
    return evaluator.get_summary()