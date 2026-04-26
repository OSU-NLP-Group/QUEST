import asyncio
import logging
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys


# -----------------------------------------------------------------------------
# Task constants
# -----------------------------------------------------------------------------
TASK_ID = "allegiant_outdoor_trip_4_dests"
TASK_DESCRIPTION = """
You are planning a budget outdoor recreation trip for a group of 12 people to visit multiple national parks in the United States. To minimize travel costs, you want to fly exclusively on Allegiant Air, which is known for its low-cost nonstop routes to destinations near national parks.

Identify 4 national parks or outdoor recreation destinations in the United States that meet ALL of the following requirements:

1. The destination must be accessible via Allegiant Air nonstop service (i.e., Allegiant must operate nonstop flights to an airport that serves the destination)
2. The destination must be located within 150 miles of the airport served by Allegiant Air
3. The destination must offer camping facilities that can accommodate groups of at least 12 people
4. Camping reservations at the destination must be available through Recreation.gov
5. The destination must have hiking trails available for outdoor recreation

For each destination, provide:
- The name of the national park or outdoor recreation destination
- The airport code (e.g., TYS) of the Allegiant-served airport
- The approximate distance from the airport to the destination
- Confirmation that group camping facilities can accommodate your group size
- Reference URLs supporting each piece of information
""".strip()


# -----------------------------------------------------------------------------
# Extraction models
# -----------------------------------------------------------------------------
class AllegiantInfo(BaseModel):
    airport_code: Optional[str] = None
    nonstop: Optional[bool] = None
    route_urls: List[str] = Field(default_factory=list)


class DistanceInfo(BaseModel):
    approx_miles: Optional[str] = None  # Keep as string to allow ranges like "120–130" or "about 95"
    distance_urls: List[str] = Field(default_factory=list)


class CampingInfo(BaseModel):
    group_capacity_note: Optional[str] = None  # e.g., "Group sites up to 25 people"
    min_group_capacity: Optional[str] = None   # e.g., "25", "12–30", "at least 14"
    reservation_platform: Optional[str] = None  # Expect "Recreation.gov" if provided
    camping_urls: List[str] = Field(default_factory=list)
    recreation_urls: List[str] = Field(default_factory=list)


class ActivityInfo(BaseModel):
    hiking_available: Optional[bool] = None
    activity_urls: List[str] = Field(default_factory=list)


class DestinationItem(BaseModel):
    name: Optional[str] = None
    allegiant: AllegiantInfo = AllegiantInfo()
    distance: DistanceInfo = DistanceInfo()
    camping: CampingInfo = CampingInfo()
    activities: ActivityInfo = ActivityInfo()


class DestinationsExtraction(BaseModel):
    destinations: List[DestinationItem] = Field(default_factory=list)


# -----------------------------------------------------------------------------
# Extraction prompt
# -----------------------------------------------------------------------------
def prompt_extract_destinations() -> str:
    return """
Extract up to 6 distinct U.S. national parks or outdoor recreation destinations mentioned in the answer that are tied to Allegiant Air-served airports and include supporting sources for each required fact.

For each destination, return an object with the following fields (arrays may be empty if the answer does not provide them):

- name: Name of the national park or outdoor recreation destination (string)
- allegiant: {
    airport_code: 3-letter IATA airport code Allegiant serves for this destination (string),
    nonstop: whether Allegiant provides nonstop service to that airport (boolean if stated, else null),
    route_urls: list of URLs that support Allegiant service to this airport (ideally allegiantair.com route map, airport destination pages, or credible airport pages)
  }
- distance: {
    approx_miles: approximate distance in miles from the airport to the destination as stated in the answer (string; keep as written, do NOT parse to number),
    distance_urls: list of URLs that support the distance claim (maps results, official park/airport pages, tourism sites, etc.)
  }
- camping: {
    group_capacity_note: textual note from the answer about group capacity (e.g., "group sites up to 25"), or null,
    min_group_capacity: the minimum stated capacity for a single group site or group facility if provided (string; keep as written),
    reservation_platform: the named reservation platform if provided (expect "Recreation.gov" when applicable),
    camping_urls: list of URLs that support camping info (can include NPS, state park, BLM, county park, etc.),
    recreation_urls: list of URLs to Recreation.gov pages relevant to camping at/for this destination (extract only explicit rec.gov URLs from the answer)
  }
- activities: {
    hiking_available: whether the answer explicitly states or implies hiking trails are available (boolean if stated, else null),
    activity_urls: list of URLs that support hiking/activity availability (NPS/state park/trail pages, AllTrails, etc.)
  }

IMPORTANT:
- Only extract URLs that are explicitly present in the answer text (including markdown links).
- Do not invent URLs.
- Keep distance values as free-form strings (e.g., "95 miles", "about 120–130 miles").
- If the answer lists more than 4 destinations, return them all, we will select the first four later.
""".strip()


# -----------------------------------------------------------------------------
# Helper utilities
# -----------------------------------------------------------------------------
def _first_n_with_padding(items: List[DestinationItem], n: int) -> List[DestinationItem]:
    result = list(items[:n])
    while len(result) < n:
        result.append(DestinationItem())
    return result


def _norm_urls(urls: Optional[List[str]]) -> List[str]:
    if not urls:
        return []
    # Deduplicate while preserving order
    seen = set()
    out: List[str] = []
    for u in urls:
        if not u:
            continue
        if u not in seen:
            seen.add(u)
            out.append(u)
    return out


def _recgov_urls(camping: CampingInfo) -> List[str]:
    if camping.recreation_urls:
        return _norm_urls([u for u in camping.recreation_urls if isinstance(u, str)])
    # Fallback: infer rec.gov URLs among camping_urls if extractor didn’t separate them
    return _norm_urls([u for u in camping.camping_urls if isinstance(u, str) and "recreation.gov" in u.lower()])


# -----------------------------------------------------------------------------
# Destination verification builder
# -----------------------------------------------------------------------------
async def verify_destination(
    evaluator: Evaluator,
    parent_node,
    dest: DestinationItem,
    idx: int
) -> None:
    """
    Build verification subtree for one destination according to the rubric.
    idx is zero-based; use idx+1 for human-friendly numbering and node IDs (D1_, D2_, ...).
    """
    dnum = idx + 1
    did_prefix = f"D{dnum}_"

    # Collect/normalize frequently used fields
    park_name = (dest.name or "").strip()
    airport_code = (dest.allegiant.airport_code or "").strip().upper()
    allegiant_urls = _norm_urls(dest.allegiant.route_urls)
    distance_urls = _norm_urls(dest.distance.distance_urls)
    approx_miles = (dest.distance.approx_miles or "").strip()
    camping_urls = _norm_urls(dest.camping.camping_urls)
    rec_urls = _recgov_urls(dest.camping)
    activity_urls = _norm_urls(dest.activities.activity_urls)

    # -------------------------------------------------------------------------
    # Destination container (non-critical to allow partial credit across 4)
    # -------------------------------------------------------------------------
    dest_node = evaluator.add_parallel(
        id=f"Destination_{dnum}",
        desc=f"{['First','Second','Third','Fourth','Fifth','Sixth'][idx] if idx < 6 else f'Destination {dnum}'} eligible destination with all requirements met",
        parent=parent_node,
        critical=False,
    )

    # -------------------------------------------------------------------------
    # 1) Park Identification (critical group)
    # -------------------------------------------------------------------------
    id_node = evaluator.add_parallel(
        id=f"{did_prefix}Park_Identification",
        desc="Correctly identifies the national park or outdoor recreation destination with required details",
        parent=dest_node,
        critical=True
    )

    # D?_Park_Name (existence)
    evaluator.add_custom_node(
        result=bool(park_name),
        id=f"{did_prefix}Park_Name",
        desc="Provides the name of the national park or outdoor recreation destination",
        parent=id_node,
        critical=True
    )

    # D?_Airport_Code (existence, simple format check)
    airport_ok = (len(airport_code) == 3 and airport_code.isalpha())
    evaluator.add_custom_node(
        result=airport_ok,
        id=f"{did_prefix}Airport_Code",
        desc="Provides the airport code of the Allegiant-served airport",
        parent=id_node,
        critical=True
    )

    # D?_ID_Reference (existence of references for Allegiant route + distance/support)
    id_ref_ok = bool(allegiant_urls) and bool(distance_urls)
    evaluator.add_custom_node(
        result=id_ref_ok,
        id=f"{did_prefix}ID_Reference",
        desc="Provides valid reference URL for destination and Allegiant route information",
        parent=id_node,
        critical=True
    )

    # D?_Allegiant_Service (verify via URLs)
    allegiant_leaf = evaluator.add_leaf(
        id=f"{did_prefix}Allegiant_Service",
        desc="Verifies Allegiant Air provides nonstop service to the stated airport",
        parent=id_node,
        critical=True
    )
    allegiant_claim = (
        f"Allegiant Air provides scheduled nonstop service to {airport_code} airport."
        if airport_ok else
        "Allegiant Air provides scheduled nonstop service to the stated airport."
    )
    await evaluator.verify(
        claim=allegiant_claim,
        node=allegiant_leaf,
        sources=allegiant_urls,
        additional_instruction=(
            "Use Allegiant's official route map, destination pages, or credible airport pages. "
            "Confirm that Allegiant operates nonstop routes to this airport (terms like 'nonstop' or 'non-stop' or "
            "'route map showing nonstop connections' are acceptable)."
        ),
    )

    # D?_Distance_Value (verify distance value evidence; tolerate if answer phrased approximately)
    distance_val_leaf = evaluator.add_leaf(
        id=f"{did_prefix}Distance_Value",
        desc="Provides the approximate distance from the airport to the destination",
        parent=id_node,
        critical=True
    )
    if approx_miles:
        distance_val_claim = (
            f"The cited source(s) indicate that the distance from {airport_code} airport to {park_name} is approximately "
            f"{approx_miles} (allowing minor rounding or route variation)."
            if airport_ok and park_name else
            "The cited source(s) provide an approximate distance between the airport and the stated destination."
        )
    else:
        # If no explicit value, still require the source provides an approximate distance statement
        distance_val_claim = (
            f"The cited source(s) provide an approximate mileage distance between {airport_code} airport and {park_name}."
            if airport_ok and park_name else
            "The cited source(s) provide an approximate distance between the stated airport and the stated destination."
        )
    await evaluator.verify(
        claim=distance_val_claim,
        node=distance_val_leaf,
        sources=distance_urls,
        additional_instruction=(
            "Accept reasonable approximate or rounded values and small discrepancies across different routes. "
            "It suffices that the source supplies a plausible mileage figure linking the airport to the destination."
        ),
    )

    # D?_Distance_Constraint (verify <= 150 miles)
    distance_constraint_leaf = evaluator.add_leaf(
        id=f"{did_prefix}Distance_Constraint",
        desc="Confirms the stated distance is within 150 miles of the Allegiant-served airport",
        parent=id_node,
        critical=True
    )
    distance_constraint_claim = (
        f"The distance from {airport_code} airport to {park_name} is within 150 miles."
        if airport_ok and park_name else
        "The distance between the stated airport and the stated destination is within 150 miles."
    )
    await evaluator.verify(
        claim=distance_constraint_claim,
        node=distance_constraint_leaf,
        sources=distance_urls,
        additional_instruction=(
            "Use the provided distance/mapping sources. Consider typical driving route distances; "
            "accept minor deviations or approximations. If multiple distances are shown, accept if at least one common/typical route is ≤ 150 miles."
        ),
    )

    # -------------------------------------------------------------------------
    # 2) Camping Verification (critical group)
    # -------------------------------------------------------------------------
    camp_node = evaluator.add_parallel(
        id=f"{did_prefix}Camping_Verification",
        desc="Verifies camping availability and capacity at the destination",
        parent=dest_node,
        critical=True
    )

    # D?_Camping_Reference (existence of camping URLs and rec.gov presence)
    has_camping_refs = bool(camping_urls) and bool(rec_urls)
    evaluator.add_custom_node(
        result=has_camping_refs,
        id=f"{did_prefix}Camping_Reference",
        desc="Provides valid reference URL for camping information",
        parent=camp_node,
        critical=True
    )

    # D?_Group_Capacity (verify ≥ 12 people)
    group_capacity_leaf = evaluator.add_leaf(
        id=f"{did_prefix}Group_Capacity",
        desc="Confirms campsites can accommodate groups of 12 or more people",
        parent=camp_node,
        critical=True
    )
    if dest.camping.min_group_capacity:
        group_capacity_claim = (
            f"The camping facilities at {park_name} include group sites or group camping options that can accommodate "
            f"at least 12 people (the stated capacity is {dest.camping.min_group_capacity})."
            if park_name else
            "The camping facilities include group sites that can accommodate at least 12 people."
        )
    else:
        group_capacity_claim = (
            f"The camping facilities at {park_name} can accommodate groups of at least 12 people."
            if park_name else
            "The camping facilities can accommodate groups of at least 12 people."
        )
    await evaluator.verify(
        claim=group_capacity_claim,
        node=group_capacity_leaf,
        sources=(rec_urls + camping_urls),
        additional_instruction=(
            "Look for explicit group campsite capacity or statements that a single group site accommodates ≥12. "
            "Prefer Recreation.gov or official park/agency pages."
        ),
    )

    # D?_Reservation_System (verify Recreation.gov used)
    reservation_leaf = evaluator.add_leaf(
        id=f"{did_prefix}Reservation_System",
        desc="Verifies the destination uses Recreation.gov for camping reservations",
        parent=camp_node,
        critical=True
    )
    reservation_claim = (
        f"Camping reservations for {park_name} are available through Recreation.gov."
        if park_name else
        "Camping reservations for the destination are available through Recreation.gov."
    )
    await evaluator.verify(
        claim=reservation_claim,
        node=reservation_leaf,
        sources=rec_urls if rec_urls else camping_urls,
        additional_instruction=(
            "Confirm the relevant campground(s) for this destination are listed on Recreation.gov and available to book. "
            "The exact park/campground name on the Recreation.gov page should clearly correspond to the destination."
        ),
    )

    # -------------------------------------------------------------------------
    # 3) Activities (critical group)
    # -------------------------------------------------------------------------
    act_node = evaluator.add_parallel(
        id=f"{did_prefix}Activities",
        desc="Confirms availability of hiking trails at the destination",
        parent=dest_node,
        critical=True
    )

    # D?_Activity_Reference (existence)
    evaluator.add_custom_node(
        result=bool(activity_urls),
        id=f"{did_prefix}Activity_Reference",
        desc="Provides valid reference URL for hiking/activity information",
        parent=act_node,
        critical=True
    )

    # D?_Hiking_Available (verify by URLs)
    hiking_leaf = evaluator.add_leaf(
        id=f"{did_prefix}Hiking_Available",
        desc="Verifies hiking trails are available at the destination",
        parent=act_node,
        critical=True
    )
    hiking_claim = (
        f"Hiking trails are available at {park_name}."
        if park_name else
        "Hiking trails are available at the destination."
    )
    await evaluator.verify(
        claim=hiking_claim,
        node=hiking_leaf,
        sources=activity_urls,
        additional_instruction=(
            "Accept official park/agency pages (NPS, state park, BLM, etc.) or credible trail resources that explicitly "
            "indicate hiking trails exist at the destination."
        ),
    )


# -----------------------------------------------------------------------------
# Main evaluation entry
# -----------------------------------------------------------------------------
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
    Evaluate an answer for the Allegiant-only outdoor trip plan requiring 4 destinations.
    """
    # Initialize evaluator (root is a container; keep non-critical to allow partial credit)
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

    # Add explicit top-level node mirroring rubric root (set non-critical to avoid framework's strict critical-child rule)
    main_node = evaluator.add_parallel(
        id="Complete_Trip_Plan",
        desc="Identifies 4 national parks or outdoor recreation destinations in the United States that meet all transportation, facility, proximity, and activity requirements",
        parent=root,
        critical=False,  # Intentionally non-critical to allow partial credit across items
    )

    # Extract structured destinations
    extraction = await evaluator.extract(
        prompt=prompt_extract_destinations(),
        template_class=DestinationsExtraction,
        extraction_name="destinations_extraction",
    )

    # Select first 4 and pad if needed
    selected = _first_n_with_padding(extraction.destinations, 4)

    # Add a quick summary of extracted destination names for debugging/trace
    evaluator.add_custom_info(
        info={"extracted_destinations": [d.name for d in extraction.destinations]},
        info_type="extraction_summary",
        info_name="extracted_destination_names"
    )

    # Build and verify each destination subtree
    for i, dest in enumerate(selected):
        await verify_destination(evaluator, main_node, dest, i)

    # Return structured summary
    return evaluator.get_summary()