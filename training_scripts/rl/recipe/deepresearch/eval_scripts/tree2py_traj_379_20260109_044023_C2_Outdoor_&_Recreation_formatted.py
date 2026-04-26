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
TASK_ID = "co_highest_peak_northeast_ridge"
TASK_DESCRIPTION = (
    "I'm planning to hike the highest peak in Colorado and want to take the Northeast Ridge route, which I've read is the most popular trail to the summit. "
    "Please provide the following information:\n\n"
    "1. The name of the peak and its official elevation (in feet)\n"
    "2. The round-trip distance of the Northeast Ridge route (in miles)\n"
    "3. The total elevation gain from trailhead to summit (in feet)\n"
    "4. The name of the road where the trailhead is located and basic directions from the nearest town\n"
    "5. Whether I need any wilderness permits for a day hike\n\n"
    "Please include reference URLs for all information provided."
)


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class PeakInfo(BaseModel):
    peak_name: Optional[str] = None
    elevation_feet: Optional[str] = None
    elevation_measurement_note: Optional[str] = None  # e.g., mentions "LiDAR", "official"
    location_range: Optional[str] = None              # e.g., "Sawatch Range"
    location_forest: Optional[str] = None             # e.g., "San Isabel National Forest"
    sources_peak: List[str] = Field(default_factory=list)
    sources_elevation: List[str] = Field(default_factory=list)
    sources_location: List[str] = Field(default_factory=list)


class RouteInfo(BaseModel):
    route_name: Optional[str] = None                   # e.g., "Northeast Ridge"
    route_popularity_note: Optional[str] = None        # e.g., "most popular", "standard route"
    difficulty_class: Optional[str] = None             # e.g., "Class 1"
    maintained_trail_note: Optional[str] = None        # e.g., "maintained", "established trail"
    distance_round_trip_miles: Optional[str] = None
    elevation_gain_feet: Optional[str] = None
    sources_route: List[str] = Field(default_factory=list)
    sources_distance: List[str] = Field(default_factory=list)
    sources_gain: List[str] = Field(default_factory=list)
    sources_difficulty: List[str] = Field(default_factory=list)


class TrailheadInfo(BaseModel):
    trailhead_road_name: Optional[str] = None          # e.g., "Halfmoon Creek Road"
    directions_from_nearest_town: Optional[str] = None
    nearest_town_name: Optional[str] = None            # e.g., "Leadville"
    permit_required_day_hike: Optional[str] = None     # e.g., "yes"/"no" or statement
    permit_notes: Optional[str] = None
    sources_trailhead: List[str] = Field(default_factory=list)
    sources_permits: List[str] = Field(default_factory=list)


class HikeExtraction(BaseModel):
    peak: Optional[PeakInfo] = None
    route: Optional[RouteInfo] = None
    trailhead: Optional[TrailheadInfo] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_hike_info() -> str:
    return (
        "Extract the requested information exactly as presented in the answer. Do not infer or invent. "
        "Return a JSON object with the following nested structure and fields:\n\n"
        "peak:\n"
        "- peak_name: the peak's name\n"
        "- elevation_feet: the elevation in feet as stated\n"
        "- elevation_measurement_note: any note indicating it is an official LiDAR-based elevation or similar\n"
        "- location_range: the mountain range (e.g., 'Sawatch Range')\n"
        "- location_forest: the national forest (e.g., 'San Isabel National Forest')\n"
        "- sources_peak: URL(s) supporting the peak identification\n"
        "- sources_elevation: URL(s) supporting the elevation value\n"
        "- sources_location: URL(s) supporting the location details\n\n"
        "route:\n"
        "- route_name: the route name (e.g., 'Northeast Ridge')\n"
        "- route_popularity_note: any text indicating it is the most popular or standard route\n"
        "- difficulty_class: the difficulty/class rating (e.g., 'Class 1')\n"
        "- maintained_trail_note: text indicating the trail is established/maintained/nontechnical\n"
        "- distance_round_trip_miles: the round-trip distance in miles as stated\n"
        "- elevation_gain_feet: the total elevation gain in feet as stated\n"
        "- sources_route: URL(s) supporting the route identification/popularity/standard claim\n"
        "- sources_distance: URL(s) supporting the round-trip distance\n"
        "- sources_gain: URL(s) supporting the total elevation gain\n"
        "- sources_difficulty: URL(s) supporting the trail class/maintenance/nontechnical claim\n\n"
        "trailhead:\n"
        "- trailhead_road_name: the road name where the trailhead is located\n"
        "- directions_from_nearest_town: basic directions from the nearest town\n"
        "- nearest_town_name: the nearest town name\n"
        "- permit_required_day_hike: explicitly state whether a wilderness permit is required for a day hike (e.g., 'yes', 'no', or an explicit sentence)\n"
        "- permit_notes: any additional notes about permits\n"
        "- sources_trailhead: URL(s) supporting road and directions info\n"
        "- sources_permits: URL(s) supporting the day-hike permit requirement claim\n\n"
        "Follow these rules:\n"
        "1) Extract only from the answer's content. If any item is missing, set it to null; if URLs are missing, return an empty list.\n"
        "2) For URLs, extract the actual links present in the answer (including markdown links); ignore malformed ones.\n"
        "3) Do not normalize units or values; keep them exactly as stated in the answer."
    )


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _present_text(val: Optional[str]) -> bool:
    return bool(val and isinstance(val, str) and val.strip())

def _combine_sources(*lists: Optional[List[str]]) -> List[str]:
    seen = set()
    combined: List[str] = []
    for lst in lists:
        if not lst:
            continue
        for url in lst:
            if not url:
                continue
            if url not in seen:
                seen.add(url)
                combined.append(url)
    return combined

def _interpret_yes_no(val: Optional[str]) -> Optional[bool]:
    if not _present_text(val):
        return None
    s = val.strip().lower()
    tokens = {"yes": True, "true": True, "required": True, "no": False, "false": False, "not required": False}
    for key, tf in tokens.items():
        if key in s:
            return tf
    return None


# --------------------------------------------------------------------------- #
# Verification subtrees                                                       #
# --------------------------------------------------------------------------- #
async def build_peak_selection_and_constraints(
    evaluator: Evaluator,
    parent_node,
    data: HikeExtraction,
) -> None:
    peak_node = evaluator.add_parallel(
        id="Peak_Selection_and_Constraints",
        desc="Correctly identify the target peak and ensure it matches all stated peak-related constraints.",
        parent=parent_node,
        critical=True
    )

    peak = data.peak or PeakInfo()

    # Existence gate: Peak name provided
    evaluator.add_custom_node(
        result=_present_text(peak.peak_name),
        id="Peak_Name_Provided",
        desc="Peak name is provided in the answer",
        parent=peak_node,
        critical=True
    )

    # Peak name and highest point claim
    highest_leaf = evaluator.add_leaf(
        id="Peak_Name_and_Highest_Point",
        desc="Provides the peak’s name, and the named peak is the highest elevation point in Colorado.",
        parent=peak_node,
        critical=True
    )
    claim_highest = f"The peak named '{peak.peak_name or ''}' is the highest elevation point in the state of Colorado."
    peak_sources_all = _combine_sources(peak.sources_peak, peak.sources_elevation, peak.sources_location)
    await evaluator.verify(
        claim=claim_highest,
        node=highest_leaf,
        sources=peak_sources_all,
        additional_instruction="Verify the name refers to Colorado’s state high point. Allow minor name variants (e.g., Mt. vs. Mount)."
    )

    # Official elevation value supported
    elev_val_leaf = evaluator.add_leaf(
        id="Official_Elevation_Value_Supported",
        desc="The stated elevation in feet is supported by a cited source.",
        parent=peak_node,
        critical=True
    )
    claim_elev_val = f"The official elevation of '{peak.peak_name or ''}' is {peak.elevation_feet or ''} feet."
    await evaluator.verify(
        claim=claim_elev_val,
        node=elev_val_leaf,
        sources=_combine_sources(peak.sources_elevation, peak.sources_peak),
        additional_instruction="Check the elevation figure is explicitly supported by the provided source; minor rounding differences are acceptable."
    )

    # LiDAR indicator supported
    lidar_leaf = evaluator.add_leaf(
        id="Official_LiDAR_Elevation_Provided",
        desc="Provides the peak’s official elevation is LiDAR-based, supported by a source.",
        parent=peak_node,
        critical=True
    )
    claim_lidar = "The elevation value cited for this peak is reported as based on official LiDAR measurement."
    await evaluator.verify(
        claim=claim_lidar,
        node=lidar_leaf,
        sources=_combine_sources(peak.sources_elevation, peak.sources_peak),
        additional_instruction="Look for explicit mentions such as 'LiDAR', 'lidar-derived', 'official LiDAR measurement' on the source page."
    )

    # Location constraints: Sawatch Range
    sawatch_leaf = evaluator.add_leaf(
        id="Peak_In_Sawatch_Range",
        desc="States the peak is located in the Sawatch Range.",
        parent=peak_node,
        critical=True
    )
    claim_sawatch = f"The peak '{peak.peak_name or ''}' is located in the Sawatch Range."
    await evaluator.verify(
        claim=claim_sawatch,
        node=sawatch_leaf,
        sources=_combine_sources(peak.sources_location, peak.sources_peak),
        additional_instruction="Confirm the source explicitly associates the peak with the Sawatch Range."
    )

    # Location constraints: San Isabel National Forest
    forest_leaf = evaluator.add_leaf(
        id="Peak_In_San_Isabel_NF",
        desc="States the peak is within the San Isabel National Forest.",
        parent=peak_node,
        critical=True
    )
    claim_forest = f"The peak '{peak.peak_name or ''}' is within the San Isabel National Forest."
    await evaluator.verify(
        claim=claim_forest,
        node=forest_leaf,
        sources=_combine_sources(peak.sources_location, peak.sources_peak),
        additional_instruction="Confirm an explicit association with San Isabel National Forest; reasonable synonyms/USFS references allowed."
    )


async def build_route_and_trail_requirements(
    evaluator: Evaluator,
    parent_node,
    data: HikeExtraction,
) -> None:
    route_node = evaluator.add_parallel(
        id="Route_and_Trail_Requirements",
        desc="Provide route-specific details and verify route/trail constraints.",
        parent=parent_node,
        critical=True
    )

    route = data.route or RouteInfo()

    # Existence gate: Route name provided
    evaluator.add_custom_node(
        result=_present_text(route.route_name),
        id="Route_Name_Provided",
        desc="Route name is provided in the answer",
        parent=route_node,
        critical=True
    )

    # Route name is Northeast Ridge
    route_name_leaf = evaluator.add_leaf(
        id="Route_Is_Northeast_Ridge",
        desc="Specifies the route as the Northeast Ridge route.",
        parent=route_node,
        critical=True
    )
    claim_ne_ridge = "The standard summit route is known as the Northeast Ridge route."
    await evaluator.verify(
        claim=claim_ne_ridge,
        node=route_name_leaf,
        sources=_combine_sources(route.sources_route, route.sources_distance, route.sources_gain),
        additional_instruction="Allow minor naming variants (e.g., 'NE Ridge'). Confirm that this refers to the primary summit route."
    )

    # Route documented as most popular/standard
    route_pop_leaf = evaluator.add_leaf(
        id="Route_Is_Most_Popular_Standard",
        desc="Supports that the Northeast Ridge is documented as the most popular and/or standard route to the summit.",
        parent=route_node,
        critical=True
    )
    claim_pop = "The Northeast Ridge route is documented as the most popular and/or standard route to the summit."
    await evaluator.verify(
        claim=claim_pop,
        node=route_pop_leaf,
        sources=_combine_sources(route.sources_route),
        additional_instruction="Look for language like 'most popular', 'standard route', or 'primary route'."
    )

    # Trail class is Class 1 (nontechnical)
    class_leaf = evaluator.add_leaf(
        id="Trail_Is_Class1_Nontechnical",
        desc="Verifies the existence of a non-technical Class 1 trail to the summit.",
        parent=route_node,
        critical=True
    )
    claim_class = "This summit route is rated Class 1 (nontechnical) for hiking."
    await evaluator.verify(
        claim=claim_class,
        node=class_leaf,
        sources=_combine_sources(route.sources_difficulty, route.sources_route),
        additional_instruction="Look for explicit 'Class 1' rating or equivalent wording indicating nontechnical hiking."
    )

    # Trail is established/maintained
    maintained_leaf = evaluator.add_leaf(
        id="Trail_Is_Established_Maintained",
        desc="Verifies there is an established, maintained hiking trail to the summit.",
        parent=route_node,
        critical=True
    )
    claim_maintained = "There is an established, maintained trail to the summit via the Northeast Ridge."
    await evaluator.verify(
        claim=claim_maintained,
        node=maintained_leaf,
        sources=_combine_sources(route.sources_difficulty, route.sources_route),
        additional_instruction="Terms like 'well-defined', 'maintained', 'established trail' should be considered supportive."
    )

    # Distance: existence and support
    evaluator.add_custom_node(
        result=_present_text(route.distance_round_trip_miles),
        id="Round_Trip_Distance_Provided",
        desc="Provides the Northeast Ridge route round-trip distance in miles.",
        parent=route_node,
        critical=True
    )
    dist_leaf = evaluator.add_leaf(
        id="Round_Trip_Distance_Supported",
        desc="Round-trip distance value is supported by a reference URL.",
        parent=route_node,
        critical=True
    )
    claim_dist = f"The round-trip distance of the Northeast Ridge route is {route.distance_round_trip_miles or ''} miles."
    await evaluator.verify(
        claim=claim_dist,
        node=dist_leaf,
        sources=_combine_sources(route.sources_distance, route.sources_route),
        additional_instruction="Confirm the source reports a round-trip distance matching the stated value; allow minor rounding differences."
    )

    # Elevation gain: existence and support
    evaluator.add_custom_node(
        result=_present_text(route.elevation_gain_feet),
        id="Total_Elevation_Gain_Provided",
        desc="Provides the total elevation gain from trailhead to summit in feet.",
        parent=route_node,
        critical=True
    )
    gain_leaf = evaluator.add_leaf(
        id="Total_Elevation_Gain_Supported",
        desc="Total elevation gain value is supported by a reference URL.",
        parent=route_node,
        critical=True
    )
    claim_gain = f"The total elevation gain from the trailhead to the summit is {route.elevation_gain_feet or ''} feet."
    await evaluator.verify(
        claim=claim_gain,
        node=gain_leaf,
        sources=_combine_sources(route.sources_gain, route.sources_route),
        additional_instruction="Confirm the source reports an elevation gain matching the stated value; allow minor rounding differences."
    )


async def build_trailhead_access_and_permits(
    evaluator: Evaluator,
    parent_node,
    data: HikeExtraction,
) -> None:
    th_node = evaluator.add_parallel(
        id="Trailhead_Access_and_Permits",
        desc="Provide trailhead road/directions and permit requirements as requested.",
        parent=parent_node,
        critical=True
    )

    th = data.trailhead or TrailheadInfo()

    # Trailhead road is Halfmoon Creek Road
    road_leaf = evaluator.add_leaf(
        id="Trailhead_Road_and_Directions",
        desc="Identifies the trailhead road as Halfmoon Creek Road and provides basic directions from the nearest town (Leadville), supported by a reference URL.",
        parent=th_node,
        critical=True
    )
    claim_road = "The trailhead for the Northeast Ridge route is located along Halfmoon Creek Road, with basic directions from Leadville."
    await evaluator.verify(
        claim=claim_road,
        node=road_leaf,
        sources=_combine_sources(th.sources_trailhead),
        additional_instruction="Confirm mention of 'Halfmoon Creek Road' and reasonable directions originating from 'Leadville'. Allow minor naming variants (e.g., 'Halfmoon Road')."
    )

    # Day-hike permits requirement answered and supported
    permits_exist = _present_text(th.permit_required_day_hike)
    evaluator.add_custom_node(
        result=permits_exist,
        id="Permit_Answer_Provided",
        desc="The answer explicitly states whether a wilderness permit is required for a day hike.",
        parent=th_node,
        critical=True
    )
    permit_bool = _interpret_yes_no(th.permit_required_day_hike)
    permit_leaf = evaluator.add_leaf(
        id="Day_Hike_Permit_Requirement_Answered",
        desc="States whether any wilderness permits are required for a day hike and supports the claim with a reference URL.",
        parent=th_node,
        critical=True
    )
    if permit_bool is True:
        claim_permit = "A wilderness permit is required for a day hike on this route."
    elif permit_bool is False:
        claim_permit = "No wilderness permit is required for a day hike on this route."
    else:
        # If unknown, still verify the statement text as-is for support
        claim_permit = f"The permit requirement is: {th.permit_required_day_hike or ''}."
    await evaluator.verify(
        claim=claim_permit,
        node=permit_leaf,
        sources=_combine_sources(th.sources_permits),
        additional_instruction="Confirm the day-hike permit rule on an authoritative source (USFS or official land manager)."
    )


async def build_reference_url_coverage(
    evaluator: Evaluator,
    parent_node,
    data: HikeExtraction,
) -> None:
    cov_node = evaluator.add_parallel(
        id="Reference_URL_Coverage",
        desc="All requested information is backed by authoritative reference URLs.",
        parent=parent_node,
        critical=True
    )

    peak = data.peak or PeakInfo()
    route = data.route or RouteInfo()
    th = data.trailhead or TrailheadInfo()

    # Coverage checks via existence of sources
    peak_elev_sources = _combine_sources(peak.sources_peak, peak.sources_elevation)
    evaluator.add_custom_node(
        result=len(peak_elev_sources) > 0,
        id="Sources_For_Peak_Name_and_Elevation",
        desc="Includes at least one authoritative URL supporting the peak identification and official elevation.",
        parent=cov_node,
        critical=True
    )

    route_dist_gain_sources = _combine_sources(route.sources_route, route.sources_distance, route.sources_gain, route.sources_difficulty)
    evaluator.add_custom_node(
        result=len(route_dist_gain_sources) > 0,
        id="Sources_For_Route_Distance_and_Elevation_Gain",
        desc="Includes at least one authoritative URL supporting the Northeast Ridge round-trip distance and total elevation gain.",
        parent=cov_node,
        critical=True
    )

    th_perm_sources = _combine_sources(th.sources_trailhead, th.sources_permits)
    evaluator.add_custom_node(
        result=len(th_perm_sources) > 0,
        id="Sources_For_Trailhead_Access_and_Permits",
        desc="Includes at least one authoritative URL supporting trailhead access/directions and day-hike permit requirements.",
        parent=cov_node,
        critical=True
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
    Evaluate an answer for hiking Colorado’s highest peak via the Northeast Ridge route.
    """
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.SEQUENTIAL,
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

    # Top-level critical sequential node to mirror rubric root
    top = evaluator.add_sequential(
        id="Colorado_Highest_Peak_Northeast_Ridge_Hike_Info",
        desc="Provide complete, constraint-satisfying information for hiking Colorado’s highest peak via the Northeast Ridge route, with authoritative references.",
        parent=root,
        critical=True
    )

    # Extract structured data
    extracted = await evaluator.extract(
        prompt=prompt_extract_hike_info(),
        template_class=HikeExtraction,
        extraction_name="hike_extraction"
    )

    # Build verification subtrees in sequence
    await build_peak_selection_and_constraints(evaluator, top, extracted)
    await build_route_and_trail_requirements(evaluator, top, extracted)
    await build_trailhead_access_and_permits(evaluator, top, extracted)
    await build_reference_url_coverage(evaluator, top, extracted)

    # Return aggregated summary
    return evaluator.get_summary()