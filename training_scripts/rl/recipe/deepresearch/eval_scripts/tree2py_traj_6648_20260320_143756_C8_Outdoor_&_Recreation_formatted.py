import asyncio
import logging
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.llm_client.base_client import LLMClient
from obj_task_eval.evaluator import Evaluator, AggregationStrategy


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "byways_campgrounds"
TASK_DESCRIPTION = """
Identify 4 developed campgrounds in the United States that are located along or near roads designated as National Scenic Byways or All-American Roads as part of America's Byways program. For each campground, provide the following information: (1) Campground name and location (including state); (2) Associated National Scenic Byway or All-American Road - the specific designated route name; (3) Byway intrinsic qualities - verification that the byway meets federal designation requirements by possessing at least one of the six intrinsic qualities (archaeological, cultural, historic, natural, recreational, scenic) for National Scenic Byways, or at least two intrinsic qualities for All-American Roads; (4) Camping facilities - confirmation that this is a developed campground (not dispersed camping) with reservable sites, including site capacity information and the type of camping accommodations available (tent pads, RV sites, or both); (5) Hiking trail access - at least one hiking trail accessible within 5 miles of the campground, including the trail name and its difficulty level (easy, moderate, or strenuous) based on standard distance and elevation gain classifications; (6) Reference URLs providing: official information about the byway's designation and intrinsic qualities, the campground's reservation system or facility information, and information about the accessible hiking trail(s) and difficulty rating. Note: The six intrinsic qualities recognized for byway designation are: archaeological, cultural, historic, natural, recreational, and scenic.
"""

# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class CampgroundItem(BaseModel):
    campground_name: Optional[str] = None
    location: Optional[str] = None
    state: Optional[str] = None

    byway_route_name: Optional[str] = None
    byway_designation: Optional[str] = None  # "National Scenic Byway" or "All-American Road" (allow variants)
    intrinsic_qualities: List[str] = Field(default_factory=list)  # any of the six qualities

    developed_indicator_text: Optional[str] = None  # evidence text extracted from answer (e.g., "developed", "amenities")
    reservable_indicator_text: Optional[str] = None  # e.g., "reservations required", "reservable"

    site_capacity_info: Optional[str] = None  # e.g., "45 sites" or "up to 8 people per site"
    accommodation_types: List[str] = Field(default_factory=list)  # e.g., ["tent", "rv"], ["tent"], ["rv"]

    trail_name: Optional[str] = None
    trail_difficulty: Optional[str] = None  # easy, moderate, strenuous
    trail_distance_miles: Optional[str] = None  # as text (e.g., "4.2 miles")
    trail_elevation_gain_ft: Optional[str] = None  # as text (e.g., "800 ft")

    urls_byway: List[str] = Field(default_factory=list)  # official byway designation/qualities page(s)
    urls_campground: List[str] = Field(default_factory=list)  # reservation system or facility info page(s)
    urls_trail: List[str] = Field(default_factory=list)  # trail and difficulty source(s)
    urls_proximity: List[str] = Field(default_factory=list)  # optional extra URLs supporting proximity to byway


class CampgroundsExtraction(BaseModel):
    campgrounds: List[CampgroundItem] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_campgrounds() -> str:
    return """
    Extract up to 4 developed U.S. campgrounds from the answer that are along or near an FHWA America's Byways designated route (National Scenic Byway or All-American Road). For each campground, return an object with the following fields:

    Required identification and association:
    - campground_name: the campground name as written in the answer
    - location: the city/area and state if available (text as in the answer)
    - state: the U.S. state (use full state name or two-letter code if provided)
    - byway_route_name: the specific America's Byways route name the campground is along/near
    - byway_designation: "National Scenic Byway" or "All-American Road" if stated (allow variants like "NSB", "AAR", "All American Road"); if unknown, set to null
    - intrinsic_qualities: list of the byway’s intrinsic qualities mentioned in the answer (subset of: archaeological, cultural, historic, natural, recreational, scenic). If not explicitly listed, return an empty list.

    Campground facility details:
    - developed_indicator_text: the text in the answer indicating it is a developed campground (e.g., mentions of "developed campground", "designated sites", "amenities", etc.). If missing, set to null.
    - reservable_indicator_text: text indicating that sites are reservable (e.g., "reservations available on Recreation.gov"). If missing, set to null.
    - site_capacity_info: site capacity info as a text snippet (e.g., "45 sites" or "sites accommodate up to 8 people"). If missing, set to null.
    - accommodation_types: list capturing available accommodations from the answer among ["tent", "rv"] as appropriate; include both if both are mentioned; if unclear, return an empty list.

    Trail access within 5 miles:
    - trail_name: the trail name
    - trail_difficulty: difficulty label (easy, moderate, or strenuous) if provided; if not explicitly labeled, set to null
    - trail_distance_miles: trail distance text if provided (e.g., "3.8 miles"); else null
    - trail_elevation_gain_ft: elevation gain text if provided (e.g., "900 ft"); else null

    Reference URLs (extract only real URLs explicitly present in the answer; do not invent):
    - urls_byway: URLs to official info about byway designation and intrinsic qualities (prefer America's Byways/FHWA or official government/state sites)
    - urls_campground: URLs to campground reservation or facility info (e.g., Recreation.gov, ReserveAmerica, NPS/USFS park pages)
    - urls_trail: URLs providing the trail info and difficulty (or data to infer difficulty)
    - urls_proximity: any URL(s) that help show the campground is along/near the byway (optional; if none, return an empty list)

    Return a JSON object with a single field:
    - campgrounds: an array of up to 4 such campground objects (preserve the first 4 in the answer order; if fewer than 4 are present, return all available).

    Follow the SPECIAL RULES FOR URL SOURCES EXTRACTION: extract only URLs actually present in the answer (including markdown links), and return empty lists when missing.
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _normalize_designation(text: Optional[str]) -> str:
    if not text:
        return "UNKNOWN"
    t = text.strip().lower()
    if "all-american" in t or "all american" in t or "aar" in t:
        return "AAR"
    if "national scenic byway" in t or "nsb" in t or "scenic byway" in t:
        return "NSB"
    return "UNKNOWN"


def _join_list_readable(items: List[str]) -> str:
    return ", ".join(items) if items else ""


def _combined_sources(*url_lists: List[str]) -> List[str]:
    seen = set()
    merged: List[str] = []
    for lst in url_lists:
        for u in lst:
            if u and u not in seen:
                seen.add(u)
                merged.append(u)
    return merged


def _first_n(items: List[Any], n: int) -> List[Any]:
    return items[:n] if items else []


def _pad_to_n(items: List[Any], n: int, pad_item_factory) -> List[Any]:
    padded = list(items)
    while len(padded) < n:
        padded.append(pad_item_factory())
    return padded


# --------------------------------------------------------------------------- #
# Verification logic per campground                                           #
# --------------------------------------------------------------------------- #
async def verify_one_campground(
    evaluator: Evaluator,
    parent_node,
    item: CampgroundItem,
    idx_one_based: int,
) -> None:
    cg_node = evaluator.add_parallel(
        id=f"campground_{idx_one_based}",
        desc=f"Campground {idx_one_based} (one qualifying campground with all required fields and citations)",
        parent=parent_node,
        critical=False
    )

    # 1) Campground name + location (state) provided – presence check
    has_name_state = bool((item.campground_name or "").strip()) and bool((item.state or "").strip())
    evaluator.add_custom_node(
        result=has_name_state,
        id=f"cg{idx_one_based}_name_location_state",
        desc="Provides the campground name and location including state.",
        parent=cg_node,
        critical=True
    )

    # 2) Byway route name – verify against byway URLs (if provided), else simple check
    byway_name_leaf = evaluator.add_leaf(
        id=f"cg{idx_one_based}_byway_route_name",
        desc="Provides the specific designated route name of the associated America's Byways route.",
        parent=cg_node,
        critical=True
    )
    byway_name_claim = f"The associated America's Byways route name is '{(item.byway_route_name or '').strip()}'."
    await evaluator.verify(
        claim=byway_name_claim,
        node=byway_name_leaf,
        sources=item.urls_byway,
        additional_instruction="Verify that the provided route name matches what the cited page calls the official America's Byways route. Allow minor naming variants or punctuation/case differences."
    )

    # 3) Byway officially designated under America's Byways (NSB or AAR)
    designation_leaf = evaluator.add_leaf(
        id=f"cg{idx_one_based}_byway_is_fhwa_americas_byways_designated",
        desc="Includes a citation to an official source confirming the route is designated under FHWA America's Byways (as NSB or AAR).",
        parent=cg_node,
        critical=True
    )
    designation_short = _normalize_designation(item.byway_designation)
    if designation_short in ("NSB", "AAR"):
        designation_text = "National Scenic Byway" if designation_short == "NSB" else "All-American Road"
        designation_claim = f"The route '{item.byway_route_name or ''}' is officially designated in FHWA's America's Byways program as a {designation_text}."
    else:
        designation_claim = f"The route '{item.byway_route_name or ''}' is officially designated in FHWA's America's Byways program (either National Scenic Byway or All-American Road)."
    await evaluator.verify(
        claim=designation_claim,
        node=designation_leaf,
        sources=item.urls_byway,
        additional_instruction="Use an official America's Byways directory/government source (e.g., byways.org or FHWA/official state sites) to confirm designation. If the source is clearly the official America's Byways page, that suffices."
    )

    # 4) Campground along or near the identified byway route
    proximity_leaf = evaluator.add_leaf(
        id=f"cg{idx_one_based}_campground_along_or_near_byway",
        desc="Shows the campground is located along or near the identified byway route.",
        parent=cg_node,
        critical=True
    )
    proximity_sources = _combined_sources(item.urls_campground, item.urls_byway, item.urls_proximity)
    proximity_claim = f"The campground '{item.campground_name or ''}' is located along or near the byway '{item.byway_route_name or ''}'."
    await evaluator.verify(
        claim=proximity_claim,
        node=proximity_leaf,
        sources=proximity_sources,
        additional_instruction="Accept if sources indicate the campground sits on, adjacent to, or within the corridor of the named byway (e.g., same highway, within the same park the byway traverses, or clear proximity on official/map pages). Minor approximations are acceptable."
    )

    # 5) Intrinsic qualities meet threshold (NSB >=1; AAR >=2)
    qualities_leaf = evaluator.add_leaf(
        id=f"cg{idx_one_based}_intrinsic_qualities_meet_threshold",
        desc="Using the official designation (NSB vs AAR) and listed intrinsic quality(ies), verifies the threshold: NSB has ≥1 of {archaeological, cultural, historic, natural, recreational, scenic}; AAR has ≥2 of these.",
        parent=cg_node,
        critical=True
    )
    qualities_list_text = _join_list_readable(item.intrinsic_qualities)
    if designation_short == "AAR":
        threshold_text = "at least two (≥2)"
    else:
        threshold_text = "at least one (≥1)"
    qualities_claim = (
        f"According to official sources, the byway '{item.byway_route_name or ''}' lists the following intrinsic qualities: "
        f"{qualities_list_text}. Given its designation ({'All-American Road' if designation_short=='AAR' else 'National Scenic Byway or unspecified'}), "
        f"this satisfies the required threshold ({threshold_text}) for America's Byways designation."
    )
    await evaluator.verify(
        claim=qualities_claim,
        node=qualities_leaf,
        sources=item.urls_byway,
        additional_instruction="Check the official byway page for explicit intrinsic qualities (archaeological, cultural, historic, natural, recreational, scenic). "
                              "For NSB, at least one quality should be listed; for AAR, at least two. Allow minor wording variants (e.g., 'recreation' vs 'recreational')."
    )

    # 6) Developed (not dispersed)
    developed_leaf = evaluator.add_leaf(
        id=f"cg{idx_one_based}_developed_not_dispersed",
        desc="Confirms the campground is a developed campground (not dispersed camping).",
        parent=cg_node,
        critical=True
    )
    developed_claim = (
        f"The campground '{item.campground_name or ''}' is a developed campground (not dispersed), with designated sites and amenities."
    )
    await evaluator.verify(
        claim=developed_claim,
        node=developed_leaf,
        sources=item.urls_campground,
        additional_instruction="Look for indications of designated sites, amenities, reservations, facility listings, or campground classification that clearly indicate 'developed' rather than dispersed/primitive."
    )

    # 7) Reservable sites
    reservable_leaf = evaluator.add_leaf(
        id=f"cg{idx_one_based}_reservable_sites",
        desc="Confirms campsites are reservable (e.g., via a reservation system).",
        parent=cg_node,
        critical=True
    )
    reservable_claim = f"The campground '{item.campground_name or ''}' has reservable campsites (reservations accepted)."
    await evaluator.verify(
        claim=reservable_claim,
        node=reservable_leaf,
        sources=item.urls_campground,
        additional_instruction="Accept if the campground page or an official reservation portal (e.g., Recreation.gov, ReserveAmerica) indicates that reservations are available/required."
    )

    # 8) Site capacity information
    capacity_leaf = evaluator.add_leaf(
        id=f"cg{idx_one_based}_site_capacity_info",
        desc="Provides site capacity information (e.g., number of sites and/or occupancy/capacity per site as stated by a source).",
        parent=cg_node,
        critical=True
    )
    capacity_text = (item.site_capacity_info or "").strip()
    capacity_claim = f"Site capacity information for the campground is: '{capacity_text}'."
    await evaluator.verify(
        claim=capacity_claim,
        node=capacity_leaf,
        sources=item.urls_campground,
        additional_instruction="Verify that the cited page mentions either total number of sites and/or occupancy/capacity per site consistent with the text."
    )

    # 9) Accommodation types (tent pads, RV sites, or both)
    accom_leaf = evaluator.add_leaf(
        id=f"cg{idx_one_based}_accommodation_types",
        desc="Specifies accommodation types available: tent pads, RV sites, or both.",
        parent=cg_node,
        critical=True
    )
    accom_text = _join_list_readable(item.accommodation_types)
    accom_claim = f"The campground accommodates: {accom_text}."
    await evaluator.verify(
        claim=accom_claim,
        node=accom_leaf,
        sources=item.urls_campground,
        additional_instruction="Confirm from the facilities page whether tent sites and/or RV sites are available. Allow synonyms (e.g., 'standard nonelectric' for tent, 'RV hookup' for RV)."
    )

    # 10) Trail within 5 miles (trail name required)
    trail_within_leaf = evaluator.add_leaf(
        id=f"cg{idx_one_based}_trail_within_5_miles_name",
        desc="Provides at least one hiking trail accessible within 5 miles, including the trail name.",
        parent=cg_node,
        critical=True
    )
    trail_name = (item.trail_name or "").strip()
    within5_claim = (
        f"The hiking trail '{trail_name}' is accessible within 5 miles of the campground '{item.campground_name or ''}'."
    )
    within5_sources = _combined_sources(item.urls_trail, item.urls_campground)
    await evaluator.verify(
        claim=within5_claim,
        node=trail_within_leaf,
        sources=within5_sources,
        additional_instruction="Accept if the cited sources indicate the trailhead is ≤5 miles (~8 km) from the campground, or clearly within the same park area implying short access distance."
    )

    # 11) Trail difficulty and basis (distance/elevation)
    trail_diff_leaf = evaluator.add_leaf(
        id=f"cg{idx_one_based}_trail_difficulty_and_basis",
        desc="Provides the trail difficulty (easy/moderate/strenuous) based on standard distance/elevation-gain classification, supported by cited/quoted distance and elevation gain (or an authoritative source providing the needed data).",
        parent=cg_node,
        critical=True
    )
    diff_text = (item.trail_difficulty or "").strip()
    dist_text = (item.trail_distance_miles or "").strip()
    gain_text = (item.trail_elevation_gain_ft or "").strip()
    diff_claim = (
        f"The trail '{trail_name}' is rated '{diff_text}'. The cited source shows distance '{dist_text}' and elevation gain '{gain_text}', "
        f"which is consistent with that difficulty rating."
    )
    await evaluator.verify(
        claim=diff_claim,
        node=trail_diff_leaf,
        sources=item.urls_trail,
        additional_instruction="Verify that the source provides a stated difficulty or provides distance/elevation data that reasonably supports the stated difficulty. "
                              "As a rough guide: easy ≲3 miles or ≲500 ft gain; moderate ≈3–6 miles and/or 500–1500 ft gain; strenuous ≳7 miles or ≳1500 ft gain. Allow reasonable variation across sources."
    )

    # 12) URL: byway designation & qualities – presence check
    evaluator.add_custom_node(
        result=bool(item.urls_byway),
        id=f"cg{idx_one_based}_url_byway_designation_and_qualities",
        desc="Provides reference URL(s) for official information about the byway's designation and intrinsic qualities.",
        parent=cg_node,
        critical=True
    )

    # 13) URL: campground reservation/facilities – presence check
    evaluator.add_custom_node(
        result=bool(item.urls_campground),
        id=f"cg{idx_one_based}_url_campground_reservation_or_facilities",
        desc="Provides reference URL(s) for the campground's reservation system or facility information.",
        parent=cg_node,
        critical=True
    )

    # 14) URL: trail info & difficulty – presence check
    evaluator.add_custom_node(
        result=bool(item.urls_trail),
        id=f"cg{idx_one_based}_url_trail_info_and_difficulty",
        desc="Provides reference URL(s) for the hiking trail information and its difficulty (and/or the data needed to justify difficulty).",
        parent=cg_node,
        critical=True
    )


# --------------------------------------------------------------------------- #
# Main evaluation entry point                                                 #
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
    Evaluate an answer for the America's Byways campgrounds task using the obj_task_eval framework.
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

    # Extract campgrounds from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_campgrounds(),
        template_class=CampgroundsExtraction,
        extraction_name="campgrounds_extraction"
    )

    # Ensure exactly 4 campgrounds (truncate or pad)
    campgrounds = _first_n(extracted.campgrounds, 4)
    campgrounds = _pad_to_n(campgrounds, 4, pad_item_factory=CampgroundItem)

    # Build verification tree for 4 campgrounds (parallel, non-critical group; critical leaves inside)
    tasks = []
    for i, item in enumerate(campgrounds, start=1):
        tasks.append(verify_one_campground(evaluator, root, item, i))
    await asyncio.gather(*tasks)

    return evaluator.get_summary()