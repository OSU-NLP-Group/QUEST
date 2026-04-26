import asyncio
import logging
import re
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "wi_lake_michigan_rv_trip_12d_2026"
TASK_DESCRIPTION = (
    "Plan a comprehensive 12-day RV camping trip along Wisconsin's Lake Michigan shoreline for a family of 4 "
    "nonresident campers (from Illinois) traveling in a 28-foot RV during July 2026. Your trip plan must visit exactly "
    "3 different Wisconsin state parks and satisfy all requirements. Provide park-specific amenities and counts, ensure "
    "minimum stay and total duration, include all costs (nonresident fees, reservation fees, vehicle admission pass), "
    "confirm booking window feasibility from Feb 21, 2026, and ground claims with official Wisconsin DNR sources."
)


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class ParkInfo(BaseModel):
    name: Optional[str] = None
    has_lake_michigan_access: Optional[str] = None  # e.g., "yes", "direct beach"
    has_electric: Optional[str] = None              # e.g., "yes", "electric hookups available"
    electric_campsites_count: Optional[str] = None  # number or descriptive string
    has_showers_summer: Optional[str] = None        # e.g., "yes", "showers available in summer"
    has_dump_station: Optional[str] = None          # e.g., "yes"
    accommodates_28ft_rv: Optional[str] = None      # e.g., "yes", or "sites up to 30 ft"
    reference_urls: List[str] = Field(default_factory=list)  # park-specific references


class TripExtraction(BaseModel):
    parks: List[ParkInfo] = Field(default_factory=list)  # Extract the first three parks in plan order
    stay_nights: List[Optional[str]] = Field(default_factory=list)  # Nights per park in the same order
    total_duration_claimed: Optional[str] = None  # e.g., "12 days" or "12 nights"

    # Cost/calc inclusions (presence in the answer)
    includes_nonresident_fee_15: Optional[str] = None     # "yes" if $15/night/site mentioned in calc
    includes_reservation_fee_7_95: Optional[str] = None   # "yes" if $7.95 per site mentioned in calc
    addresses_vehicle_admission_pass: Optional[str] = None  # "yes" if addressed

    # References for fees and policies
    cost_reference_urls: List[str] = Field(default_factory=list)       # fee structure references
    booking_reference_urls: List[str] = Field(default_factory=list)    # booking window/time references
    min_stay_reference_urls: List[str] = Field(default_factory=list)   # minimum stay references

    # Booking time mention
    mentions_booking_time_9am_ct: Optional[str] = None  # "yes" if 9:00 a.m. CT noted


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_trip() -> str:
    return """
Extract the trip plan details for a 12-day Wisconsin Lake Michigan RV camping itinerary as presented in the answer. Return a JSON following the TripExtraction schema.

Instructions:
- Parks: Identify the first three Wisconsin state parks the answer plans to visit (in the itinerary order). For each of the three, extract:
  • name (as written)
  • has_lake_michigan_access: whether the text explicitly confirms direct Lake Michigan beach/shore access (write a short confirmation phrase like “yes, Lake Michigan beach” or null if not stated)
  • has_electric: whether the park has family campsites with electric hookups (short phrase like “yes, electric hookups” or null)
  • electric_campsites_count: the stated number of family campsites with electric hookups at that park (use the exact number or descriptive phrase; null if not stated)
  • has_showers_summer: confirmation that shower facilities operate in summer (short phrase; null if not stated)
  • has_dump_station: confirmation of a dump station for registered campers (short phrase; null if not stated)
  • accommodates_28ft_rv: whether a 28-foot RV can be accommodated (e.g., “yes, sites up to 30 ft”; null if not stated)
  • reference_urls: all URLs cited for that park’s information (include official Wisconsin DNR/goingtocamp URLs if present; include all relevant URLs mentioned for that park)
- stay_nights: For each of the three selected parks (in the same order), extract the number of nights planned. If the answer provides only days for a stop, convert to nights if clearly implied (e.g., “3 days” ≈ “2 nights”) and note the numeric result as a string. If unclear or missing, set null.
- total_duration_claimed: The total trip duration as stated in the answer (e.g., “12 days”).
- includes_nonresident_fee_15: “yes” if the cost calculation explicitly includes a nonresident additional fee of $15 per night per campsite; else “no” or null.
- includes_reservation_fee_7_95: “yes” if the cost calculation includes a reservation fee of $7.95 per site; else “no” or null.
- addresses_vehicle_admission_pass: “yes” if the answer addresses Wisconsin state park vehicle admission pass; else “no” or null.
- cost_reference_urls: All URLs cited that support fee structure (nonresident fees, reservation fees, admission pass). Include official DNR or the official reservation site if present.
- booking_reference_urls: All URLs cited for booking window/timing policies (e.g., 11-month window, 9:00 a.m. CT).
- min_stay_reference_urls: All URLs cited for minimum stay requirements in peak season (e.g., 2-night minimum).
- mentions_booking_time_9am_ct: “yes” if the answer mentions that reservations open at 9:00 a.m. Central Time; else “no” or null.

General rules:
- Extract exactly the first three parks referenced in the itinerary. If fewer than three parks are provided, return what is available and leave the remaining entries empty (nulls). If more than three, only keep the first three.
- Do not invent data. If any field is not explicitly present or cannot be inferred with high confidence as described, return null for that field.
"""


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def normalize_name(name: Optional[str]) -> Optional[str]:
    if name is None:
        return None
    return re.sub(r"\s+", " ", name).strip().lower()


def parse_int_from_str(s: Optional[str]) -> Optional[int]:
    if not s:
        return None
    m = re.search(r"(\d+)", s.replace(",", ""))
    if not m:
        return None
    try:
        return int(m.group(1))
    except Exception:
        return None


def is_official_dnr_url(url: str) -> bool:
    if not url:
        return False
    u = url.lower()
    return ("dnr.wisconsin.gov" in u) or ("dnr.wi.gov" in u) or ("wisconsin.goingtocamp.com" in u)


def filter_official_urls(urls: List[str]) -> List[str]:
    official = [u for u in urls if is_official_dnr_url(u)]
    return official if official else urls  # fall back to all if none classified as official


def has_valid_official_reference(urls: List[str]) -> bool:
    return any(is_official_dnr_url(u) for u in urls)


def ensure_length(lst: List[Any], length: int, filler: Any) -> List[Any]:
    out = list(lst[:length])
    while len(out) < length:
        out.append(filler)
    return out


# --------------------------------------------------------------------------- #
# Verification builders                                                       #
# --------------------------------------------------------------------------- #
async def verify_park(
    evaluator: Evaluator,
    parent_node,
    park_info: ParkInfo,
    park_index: int,
    other_parks: List[ParkInfo],
) -> None:
    """
    Build the verification subtree for one park, matching the rubric structure.
    """

    park_order = ["First", "Second", "Third"][park_index]
    park_id_prefix = f"park_{park_index + 1}"

    # Create main container for this park (non-critical per rubric)
    park_node = evaluator.add_parallel(
        id=park_id_prefix,
        desc=f"{park_order} selected park meets all specified requirements",
        parent=parent_node,
        critical=False
    )

    # References node (critical). We check that at least one official DNR or goingtocamp URL is provided.
    refs_node = evaluator.add_custom_node(
        result=has_valid_official_reference(park_info.reference_urls),
        id=f"{park_id_prefix}_references",
        desc=f"Valid URL references provided for {park_order.lower()} park information, specifications, and amenities",
        parent=park_node,
        critical=True
    )

    official_sources = filter_official_urls(park_info.reference_urls)

    # Eligibility group (critical)
    elig_node = evaluator.add_parallel(
        id=f"{park_id_prefix}_eligibility",
        desc=f"{park_order} park satisfies basic eligibility criteria",
        parent=park_node,
        critical=True
    )

    # Park is a Wisconsin state park
    wis_loc_leaf = evaluator.add_leaf(
        id=f"{park_id_prefix}_wisconsin_location",
        desc="Park is a Wisconsin state park",
        parent=elig_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The park '{park_info.name or ''}' is a Wisconsin state park managed by the Wisconsin DNR.",
        node=wis_loc_leaf,
        sources=official_sources,
        additional_instruction="Verify the page identifies this property as a Wisconsin State Park (not a State Forest or other property type). Prefer official DNR domains. Allow reasonable name variations.",
        extra_prerequisites=[refs_node],
    )

    # Direct Lake Michigan beach access
    lm_access_leaf = evaluator.add_leaf(
        id=f"{park_id_prefix}_lake_michigan_access",
        desc="Park provides direct Lake Michigan beach access",
        parent=elig_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The park '{park_info.name or ''}' provides direct Lake Michigan beach or shoreline access.",
        node=lm_access_leaf,
        sources=official_sources,
        additional_instruction="Look for explicit mentions of 'Lake Michigan', 'beach', or 'shoreline' access at the park. Verify the access is direct within the park.",
        extra_prerequisites=[refs_node],
    )

    # Electric availability (family campsites with electric hookups)
    electric_avail_leaf = evaluator.add_leaf(
        id=f"{park_id_prefix}_electric_availability",
        desc="Park has family campsites with electric hookups available",
        parent=elig_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The park '{park_info.name or ''}' has family campsites with electric (electrical) hookups available.",
        node=electric_avail_leaf,
        sources=official_sources,
        additional_instruction="Check campground or campsite amenities for 'electric' or 'electrical' hookups for family campsites.",
        extra_prerequisites=[refs_node],
    )

    # Distinctness checks for park 2 and park 3
    if park_index == 1:
        # Second park must be different from first
        p2_name = normalize_name(park_info.name)
        p1_name = normalize_name(other_parks[0].name if len(other_parks) > 0 else None)
        evaluator.add_custom_node(
            result=bool(p2_name) and bool(p1_name) and (p2_name != p1_name),
            id=f"{park_id_prefix}_different_from_park_1",
            desc="Second park is different from the first park",
            parent=elig_node,
            critical=True
        )
    if park_index == 2:
        # Third park must be different from first and second
        p3_name = normalize_name(park_info.name)
        p1_name = normalize_name(other_parks[0].name if len(other_parks) > 0 else None)
        p2_name = normalize_name(other_parks[1].name if len(other_parks) > 1 else None)
        evaluator.add_custom_node(
            result=bool(p3_name) and bool(p1_name) and bool(p2_name) and (p3_name != p1_name) and (p3_name != p2_name),
            id=f"{park_id_prefix}_different_from_others",
            desc="Third park is different from both the first and second parks",
            parent=elig_node,
            critical=True
        )

    # Specifications group (critical)
    specs_node = evaluator.add_parallel(
        id=f"{park_id_prefix}_specifications",
        desc=f"{park_order} park specifications are accurately provided",
        parent=park_node,
        critical=True
    )

    # Number of electric campsites is stated (presence check in answer)
    elec_count_leaf = evaluator.add_leaf(
        id=f"{park_id_prefix}_electric_count",
        desc="Number of electric campsites is stated",
        parent=specs_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The answer explicitly states the number of family campsites with electric hookups for '{park_info.name or ''}'.",
        node=elec_count_leaf,
        additional_instruction="Check the answer text (not the web page) for a stated number or clear quantity phrase for electric campsites at this park. Do not require a URL for this check.",
    )

    # RV accommodation (28-foot RV)
    rv_leaf = evaluator.add_leaf(
        id=f"{park_id_prefix}_rv_accommodation",
        desc="Park can accommodate a 28-foot RV",
        parent=specs_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The park '{park_info.name or ''}' can accommodate a 28-foot RV on family campsites (site length or vehicle length allows at least 28 ft).",
        node=rv_leaf,
        sources=official_sources,
        additional_instruction="Look for maximum site/vehicle lengths or guidance indicating sites accommodate at least 28 feet. If any family campsites support ≥28 ft, consider this supported.",
        extra_prerequisites=[refs_node],
    )

    # Amenities group (critical)
    am_node = evaluator.add_parallel(
        id=f"{park_id_prefix}_amenities",
        desc=f"{park_order} park has required summer amenities",
        parent=park_node,
        critical=True
    )

    # Shower facilities in summer
    showers_leaf = evaluator.add_leaf(
        id=f"{park_id_prefix}_shower_facilities",
        desc="Park has shower facilities available during summer season",
        parent=am_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The park '{park_info.name or ''}' provides shower facilities operating during the summer season.",
        node=showers_leaf,
        sources=official_sources,
        additional_instruction="Confirm 'showers' or 'flush toilets and showers' open during summer (e.g., Memorial Day–Labor Day).",
        extra_prerequisites=[refs_node],
    )

    # Dump station
    dump_leaf = evaluator.add_leaf(
        id=f"{park_id_prefix}_dump_station",
        desc="Park has a dump station available to registered campers",
        parent=am_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The park '{park_info.name or ''}' has a dump station available to registered campers.",
        node=dump_leaf,
        sources=official_sources,
        additional_instruction="Look for 'dump station' amenity on official park or campground info pages.",
        extra_prerequisites=[refs_node],
    )


async def verify_trip_requirements(
    evaluator: Evaluator,
    parent_node,
    trip: TripExtraction
) -> None:
    """
    Build the verification subtree for overall trip requirements.
    """
    trip_node = evaluator.add_parallel(
        id="trip_requirements",
        desc="Overall trip planning requirements are satisfied",
        parent=parent_node,
        critical=True
    )

    # Stay duration compliance (critical)
    stay_node = evaluator.add_parallel(
        id="stay_duration_compliance",
        desc="Stay duration at each park meets minimum requirements",
        parent=trip_node,
        critical=True
    )

    # Normalize nights list to length 3
    stay_nights = ensure_length(trip.stay_nights, 3, None)
    nights_parsed: List[Optional[int]] = [parse_int_from_str(n) for n in stay_nights]

    # First park minimum stay (2-night minimum during peak season)
    evaluator.add_custom_node(
        result=(nights_parsed[0] is not None and nights_parsed[0] >= 2),
        id="park_1_minimum_stay",
        desc="First park stay meets 2-night minimum for peak season bookings",
        parent=stay_node,
        critical=True
    )
    # Second park minimum stay
    evaluator.add_custom_node(
        result=(nights_parsed[1] is not None and nights_parsed[1] >= 2),
        id="park_2_minimum_stay",
        desc="Second park stay meets 2-night minimum for peak season bookings",
        parent=stay_node,
        critical=True
    )
    # Third park minimum stay
    evaluator.add_custom_node(
        result=(nights_parsed[2] is not None and nights_parsed[2] >= 2),
        id="park_3_minimum_stay",
        desc="Third park stay meets 2-night minimum for peak season bookings",
        parent=stay_node,
        critical=True
    )

    # Total duration exactly 12 days across parks
    # If we have all nights parsed, we can strictly enforce. Otherwise, verify via answer claim.
    if all(n is not None for n in nights_parsed):
        evaluator.add_custom_node(
            result=(sum(nights_parsed) == 12),
            id="total_duration",
            desc="Total trip duration is 12 days across all three parks",
            parent=stay_node,
            critical=True
        )
    else:
        total_duration_leaf = evaluator.add_leaf(
            id="total_duration",
            desc="Total trip duration is 12 days across all three parks",
            parent=stay_node,
            critical=True
        )
        await evaluator.verify(
            claim="The plan totals exactly 12 days across the three park stays.",
            node=total_duration_leaf,
            additional_instruction="Check the answer text for an explicit total of 12 days or equivalent schedule implying 12 days."
        )

    # Minimum stay reference (policy page)
    dur_ref_leaf = evaluator.add_leaf(
        id="duration_reference",
        desc="Valid URL reference for minimum stay requirements",
        parent=stay_node,
        critical=True
    )
    await evaluator.verify(
        claim="The referenced policy page states minimum stay requirements for peak season family campsite reservations (e.g., a 2-night minimum on certain dates/weekends).",
        node=dur_ref_leaf,
        sources=filter_official_urls(trip.min_stay_reference_urls),
        additional_instruction="Verify that the cited policy page discusses minimum night requirements that apply during peak season or weekends.",
    )

    # Cost calculation (critical)
    cost_node = evaluator.add_parallel(
        id="cost_calculation",
        desc="Trip cost calculation includes all required fees",
        parent=trip_node,
        critical=True
    )

    # Nonresident fees included in calculation ($15/night/site)
    nonres_leaf = evaluator.add_leaf(
        id="nonresident_fees",
        desc="Nonresident additional fees ($15/night/site) are included in cost calculation",
        parent=cost_node,
        critical=True
    )
    await evaluator.verify(
        claim="The cost calculation includes a nonresident additional fee of $15 per night per campsite.",
        node=nonres_leaf,
        additional_instruction="Check the answer’s cost section for inclusion of a $15 per night per site nonresident fee."
    )

    # Reservation fees ($7.95 per site) included
    resv_fee_leaf = evaluator.add_leaf(
        id="reservation_fees",
        desc="Reservation fees ($7.95 per site) for all three parks are included",
        parent=cost_node,
        critical=True
    )
    await evaluator.verify(
        claim="The cost calculation includes a reservation fee of $7.95 per site.",
        node=resv_fee_leaf,
        additional_instruction="Check the answer’s cost section for inclusion of a $7.95 per site reservation fee."
    )

    # Vehicle admission pass addressed
    admission_leaf = evaluator.add_leaf(
        id="admission_pass",
        desc="Vehicle admission pass requirement is addressed in the cost calculation",
        parent=cost_node,
        critical=True
    )
    await evaluator.verify(
        claim="The plan addresses the Wisconsin state park vehicle admission pass requirement in the cost calculation (daily or annual nonresident pass).",
        node=admission_leaf,
        additional_instruction="Look for mention of a Wisconsin state park vehicle admission pass requirement and any cost assumption."
    )

    # Cost references (policy pages)
    cost_ref_leaf = evaluator.add_leaf(
        id="cost_reference",
        desc="Valid URL reference for fee structure",
        parent=cost_node,
        critical=True
    )
    await evaluator.verify(
        claim="The cited fee policy page(s) substantiate the fee structure used in the cost calculation (e.g., nonresident camping additional fee and/or reservation fee amounts).",
        node=cost_ref_leaf,
        sources=filter_official_urls(trip.cost_reference_urls),
        additional_instruction="It is acceptable if different referenced official pages cover different parts of the fee structure. Verify that at least one referenced official page supports a key fee used in the calculation."
    )

    # Booking feasibility (critical)
    booking_node = evaluator.add_parallel(
        id="booking_feasibility",
        desc="Trip booking is feasible given reservation system constraints",
        parent=trip_node,
        critical=True
    )

    # Within booking window from Feb 21, 2026 (11 months in advance). July 2026 is within 11 months from Feb 21, 2026.
    evaluator.add_custom_node(
        result=True,  # July 2026 is about 5 months ahead of Feb 21, 2026, so within 11-month window.
        id="within_booking_window",
        desc="July 2026 dates are within the 11-month advance booking window from February 21, 2026",
        parent=booking_node,
        critical=True
    )

    # Note: The rubric marks this non-critical, but to satisfy critical-parent constraint we set it critical as well.
    booking_time_leaf = evaluator.add_leaf(
        id="booking_time_noted",
        desc="Recognition that reservations become available at 9:00 a.m. Central Time",
        parent=booking_node,
        critical=True  # Adjusted to satisfy framework constraint for critical parent
    )
    await evaluator.verify(
        claim="The answer notes that reservations become available at 9:00 a.m. Central Time.",
        node=booking_time_leaf,
        additional_instruction="Look for '9:00 a.m.' or '9am' with Central Time in the answer text."
    )

    # Booking reference (policy page)
    booking_ref_leaf = evaluator.add_leaf(
        id="booking_reference",
        desc="Valid URL reference for reservation booking window policy",
        parent=booking_node,
        critical=True
    )
    await evaluator.verify(
        claim="The referenced policy page states that Wisconsin state park reservations open 11 months in advance.",
        node=booking_ref_leaf,
        sources=filter_official_urls(trip.booking_reference_urls),
        additional_instruction="Verify the booking window policy (11 months). If the same page also mentions the 9:00 a.m. CT release time, that is fine but not required."
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
    Evaluate an answer for the Wisconsin Lake Michigan 12-day RV camping trip plan task.
    """
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Root aggregates major sections in parallel
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

    # IMPORTANT: Set root as non-critical to allow non-critical children under it (framework constraint)
    root.critical = False

    # Extract trip information from the answer
    trip: TripExtraction = await evaluator.extract(
        prompt=prompt_extract_trip(),
        template_class=TripExtraction,
        extraction_name="trip_plan_extraction"
    )

    # Ensure exactly 3 parks in order (pad with empty ParkInfo if needed)
    parks = ensure_length(trip.parks, 3, ParkInfo())
    # Build park verification subtrees
    for idx in range(3):
        # other_parks list used for distinctness checks
        others = [parks[j] for j in range(3) if j != idx]
        await verify_park(
            evaluator=evaluator,
            parent_node=root,
            park_info=parks[idx],
            park_index=idx,
            other_parks=others
        )

    # Trip-level requirements subtree
    await verify_trip_requirements(evaluator, root, trip)

    return evaluator.get_summary()