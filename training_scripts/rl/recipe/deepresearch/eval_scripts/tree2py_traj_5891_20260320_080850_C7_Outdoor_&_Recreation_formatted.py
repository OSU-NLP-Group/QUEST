import asyncio
import logging
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.llm_client.base_client import LLMClient


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "mn_state_parks_camping_2026"
TASK_DESCRIPTION = (
    "I am planning a family camping trip to a Minnesota state park for July 2026. "
    "Please provide comprehensive information about the state park camping system including: "
    "(1) how far in advance I can make reservations and whether reservations are required, "
    "(2) the online booking system availability, "
    "(3) the total number of campsites available and types of lodging options, "
    "(4) wildlife viewing safety distances for general wildlife and elk specifically, "
    "(5) trail closure policies after rain, "
    "(6) safety guidelines regarding headphone use while hiking, "
    "(7) pet policies, "
    "(8) procedures for reporting sick or abandoned wildlife, and "
    "(9) visitor responsibilities for trash removal."
)


# --------------------------------------------------------------------------- #
# Extraction models                                                           #
# --------------------------------------------------------------------------- #
class CampingInfoSources(BaseModel):
    """
    Extract URLs explicitly provided in the answer that support each criterion.
    If the answer provides general source URLs that apply broadly, include them in general_sources as fallback.
    """
    reservation_window: List[str] = Field(default_factory=list)
    reservation_requirement: List[str] = Field(default_factory=list)
    online_booking: List[str] = Field(default_factory=list)
    total_campsite_count: List[str] = Field(default_factory=list)
    lodging_variety: List[str] = Field(default_factory=list)
    wildlife_distance_general: List[str] = Field(default_factory=list)
    wildlife_distance_elk: List[str] = Field(default_factory=list)
    trail_closure_rain_policy: List[str] = Field(default_factory=list)
    no_headphones_policy: List[str] = Field(default_factory=list)
    pet_leash_requirement: List[str] = Field(default_factory=list)
    reporting_sick_wildlife: List[str] = Field(default_factory=list)
    trash_removal_responsibility: List[str] = Field(default_factory=list)
    general_sources: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt helpers                                                   #
# --------------------------------------------------------------------------- #
def prompt_extract_sources() -> str:
    """
    Build an extraction prompt for per-criterion source URLs.
    """
    return """
    Extract, for each of the following criteria, the list of URLs explicitly present in the answer that support that criterion.
    Return a JSON object with these exact fields, each being an array of URLs (strings). If the answer provides no URL for a criterion, return an empty array for that field.

    Fields to extract (arrays of URLs):
    - reservation_window: URLs supporting how far in advance Minnesota state park campsite reservations can be made (e.g., maximum advance window, such as 120 days).
    - reservation_requirement: URLs supporting whether Minnesota state park campsites require a reservation before occupancy.
    - online_booking: URLs supporting whether online reservations can be made 24/7 through the reservation system.
    - total_campsite_count: URLs supporting the total number of campsites statewide (e.g., more than 5,000).
    - lodging_variety: URLs supporting that lodging options include cabins, guesthouses, or other lodging beyond tent/RV campsites.
    - wildlife_distance_general: URLs supporting general wildlife viewing distance guidance (e.g., stay at least 75 feet).
    - wildlife_distance_elk: URLs supporting elk-specific viewing distance guidance (e.g., at least 100 feet).
    - trail_closure_rain_policy: URLs supporting that trails may be closed for up to three days after rain.
    - no_headphones_policy: URLs supporting guidance that hikers should not wear headphones/earbuds to maintain awareness.
    - pet_leash_requirement: URLs supporting that pets must be kept on a leash (or otherwise left at home) per park/wildlife guidelines.
    - reporting_sick_wildlife: URLs supporting guidance on reporting seemingly abandoned or sick wildlife (whom to contact, etc.).
    - trash_removal_responsibility: URLs supporting visitor responsibility for trash removal (e.g., pack-in/pack-out).

    Also extract:
    - general_sources: An array of all URLs mentioned anywhere in the answer that relate to Minnesota state parks or Minnesota DNR policies/guidelines/reservations. This serves as a fallback if specific fields lack URLs.

    IMPORTANT:
    - Extract only actual URLs explicitly present in the answer (including those inside markdown links).
    - Do not fabricate or infer URLs.
    - If a URL is missing a protocol, prepend http:// as needed to form a valid URL.
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _dedup_preserve_order(urls: List[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for u in urls:
        if not isinstance(u, str):
            continue
        us = u.strip()
        if not us:
            continue
        if us not in seen:
            seen.add(us)
            out.append(us)
    return out


def _merge_with_fallback(primary: List[str], fallback: List[str]) -> List[str]:
    # Union primary and fallback to maximize available evidence
    return _dedup_preserve_order(list(primary or []) + list(fallback or []))


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
    model: str = "o4-mini",
) -> Dict:
    """
    Evaluate an answer for the Minnesota state parks camping information task.
    """
    # 1) Initialize evaluator and root node
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Independent checks
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

    # 2) Extract per-criterion source URLs (plus general fallback)
    extracted_sources: CampingInfoSources = await evaluator.extract(
        prompt=prompt_extract_sources(),
        template_class=CampingInfoSources,
        extraction_name="mn_state_parks_sources",
    )

    # 3) Build verification leaves per rubric item
    # Prepare sources for each criterion (union of specific + general)
    general = extracted_sources.general_sources or []

    sources_reservation_window = _merge_with_fallback(extracted_sources.reservation_window, general)
    sources_reservation_requirement = _merge_with_fallback(extracted_sources.reservation_requirement, general)
    sources_online_booking = _merge_with_fallback(extracted_sources.online_booking, general)
    sources_total_campsite_count = _merge_with_fallback(extracted_sources.total_campsite_count, general)
    sources_lodging_variety = _merge_with_fallback(extracted_sources.lodging_variety, general)
    sources_wildlife_general = _merge_with_fallback(extracted_sources.wildlife_distance_general, general)
    sources_wildlife_elk = _merge_with_fallback(extracted_sources.wildlife_distance_elk, general)
    sources_trail_rain = _merge_with_fallback(extracted_sources.trail_closure_rain_policy, general)
    sources_no_headphones = _merge_with_fallback(extracted_sources.no_headphones_policy, general)
    sources_pet_leash = _merge_with_fallback(extracted_sources.pet_leash_requirement, general)
    sources_reporting_sick = _merge_with_fallback(extracted_sources.reporting_sick_wildlife, general)
    sources_trash = _merge_with_fallback(extracted_sources.trash_removal_responsibility, general)

    # Create all leaf nodes
    n_res_window = evaluator.add_leaf(
        id="reservation_window_check",
        desc="The state park camping reservation system allows bookings up to 120 days in advance",
        parent=root,
        critical=True,
    )
    n_res_required = evaluator.add_leaf(
        id="reservation_requirement",
        desc="All campsites in Minnesota state parks require a reservation before they may be occupied",
        parent=root,
        critical=True,
    )
    n_online_booking = evaluator.add_leaf(
        id="online_booking_availability",
        desc="Online reservations can be made 24 hours a day through the reservation system",
        parent=root,
        critical=True,
    )
    n_total_sites = evaluator.add_leaf(
        id="total_campsite_count",
        desc="State parks and recreation areas offer more than 5,000 campsites total",
        parent=root,
        critical=False,
    )
    n_lodging = evaluator.add_leaf(
        id="lodging_variety",
        desc="The system offers a variety of camping options including cabins, guesthouses, and other lodging in addition to campsites",
        parent=root,
        critical=False,
    )
    n_wildlife_general = evaluator.add_leaf(
        id="wildlife_viewing_distance_general",
        desc="Wildlife viewing guidelines require staying at least 75 feet (about two bus lengths) away from most wildlife",
        parent=root,
        critical=True,
    )
    n_wildlife_elk = evaluator.add_leaf(
        id="wildlife_viewing_distance_elk",
        desc="Visitors must stay at least 100 feet (about two bus lengths) away from elk",
        parent=root,
        critical=True,
    )
    n_trail_rain = evaluator.add_leaf(
        id="trail_closure_rain_policy",
        desc="Trails may be closed for up to three days following rain for safety reasons",
        parent=root,
        critical=False,
    )
    n_no_headphones = evaluator.add_leaf(
        id="no_headphones_policy",
        desc="Safety guidelines advise against wearing headphones while hiking to maintain awareness of surroundings",
        parent=root,
        critical=False,
    )
    n_pet_leash = evaluator.add_leaf(
        id="pet_leash_requirement",
        desc="Pets must be kept on a leash or left at home according to wildlife viewing guidelines",
        parent=root,
        critical=True,
    )
    n_reporting_sick = evaluator.add_leaf(
        id="reporting_sick_wildlife",
        desc="The park system provides guidelines for reporting wildlife that seems abandoned or sick",
        parent=root,
        critical=False,
    )
    n_trash = evaluator.add_leaf(
        id="trash_removal_responsibility",
        desc="Visitors are expected to help with trash removal to protect wildlife",
        parent=root,
        critical=False,
    )

    # 4) Prepare verification claims with sources and additional instructions
    claims_and_sources: List[tuple[str, List[str] | None, Any, Optional[str]]] = [
        (
            "Minnesota State Parks campsite reservations can be made up to 120 days in advance (for campsites).",
            sources_reservation_window if sources_reservation_window else None,
            n_res_window,
            "Verify the maximum advance window for reserving Minnesota State Parks CAMPSITES is 120 days. "
            "Rely on official Minnesota DNR or the official reservation system pages. "
            "If the policy specifies a different number or only for some facilities, mark as not supported."
        ),
        (
            "All campsites in Minnesota State Parks require a reservation before they may be occupied (no first-come, first-served without a reservation).",
            sources_reservation_requirement if sources_reservation_requirement else None,
            n_res_required,
            "Check DNR policy language regarding required reservations before campsite occupancy. "
            "If exceptions exist (e.g., seasonal/walk-in without advance booking), this is not fully supported."
        ),
        (
            "Online reservations for Minnesota State Parks can be made 24 hours a day (24/7) through the official reservation system.",
            sources_online_booking if sources_online_booking else None,
            n_online_booking,
            "Look for explicit statements that online booking is available 24/7. "
            "If hours or downtime windows are specified instead of 24/7 availability, mark as not supported."
        ),
        (
            "Minnesota State Parks and Recreation Areas collectively offer more than 5,000 campsites in total.",
            sources_total_campsite_count if sources_total_campsite_count else None,
            n_total_sites,
            "Accept statements like 'over 5,000' or 'more than 5,000'. "
            "If totals are clearly below 5,000, mark as not supported."
        ),
        (
            "Minnesota State Parks offer lodging beyond standard campsites, including options such as cabins and guesthouses.",
            sources_lodging_variety if sources_lodging_variety else None,
            n_lodging,
            "Any official DNR mention of camper cabins, cabins, guesthouses, yurts, or other non-campsite lodging qualifies."
        ),
        (
            "Minnesota State Parks wildlife viewing guidance advises staying at least 75 feet (about two bus lengths) from most wildlife.",
            sources_wildlife_general if sources_wildlife_general else None,
            n_wildlife_general,
            "Confirm the general minimum distance guidance (≈75 feet / ~2 bus lengths) for typical wildlife. "
            "If guidance is a different distance, mark as not supported."
        ),
        (
            "Visitors must stay at least 100 feet (about two bus lengths) away from elk in Minnesota State Parks.",
            sources_wildlife_elk if sources_wildlife_elk else None,
            n_wildlife_elk,
            "Verify elk-specific distance is ≥100 feet. "
            "If a different figure is stated, mark as not supported."
        ),
        (
            "Trails in Minnesota State Parks may be closed for up to three days after rain for safety or resource protection.",
            sources_trail_rain if sources_trail_rain else None,
            n_trail_rain,
            "Accept language indicating closures may last '1–3 days' or 'up to 3 days' after rain."
        ),
        (
            "Safety guidance for Minnesota State Parks advises hikers not to wear headphones/earbuds to maintain awareness.",
            sources_no_headphones if sources_no_headphones else None,
            n_no_headphones,
            "Look for safety guidance discouraging headphone use to hear surroundings (wildlife, other users, hazards)."
        ),
        (
            "Per Minnesota State Parks guidelines, pets must be kept on a leash (or otherwise left at home) to protect wildlife and other visitors.",
            sources_pet_leash if sources_pet_leash else None,
            n_pet_leash,
            "Confirm a leash requirement (often a specified length, e.g., 6 feet). "
            "If off-leash allowance is indicated, mark as not supported."
        ),
        (
            "Minnesota State Parks provide instructions for reporting wildlife that appears abandoned, sick, or injured (e.g., contact park office or DNR).",
            sources_reporting_sick if sources_reporting_sick else None,
            n_reporting_sick,
            "Look for guidance on whom to contact or what steps to take (e.g., do not touch; call park, DNR, or wildlife rehabilitator)."
        ),
        (
            "Visitors are expected to help with trash removal (pack out trash) to protect wildlife and keep parks clean.",
            sources_trash if sources_trash else None,
            n_trash,
            "Accept 'pack in, pack out' or other language assigning trash responsibility to visitors."
        ),
    ]

    # 5) Run verifications (batch in parallel for efficiency)
    await evaluator.batch_verify(claims_and_sources=claims_and_sources)

    # 6) Return summary
    return evaluator.get_summary()