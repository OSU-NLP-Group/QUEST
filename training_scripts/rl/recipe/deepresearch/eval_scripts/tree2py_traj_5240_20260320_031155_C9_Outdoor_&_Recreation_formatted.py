import asyncio
import logging
from typing import Any, List, Optional, Dict

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "accessible_state_parks_se"
TASK_DESCRIPTION = """
Identify four different state parks in the southeastern United States (specifically in Georgia, Florida, Tennessee, Arkansas, Mississippi, or Alabama) that collectively meet ALL of the following accessibility and amenity requirements for each park:

1. The park must have at least 2 ADA-accessible RV campsites with full hookups (water, sewer, and electric service)
2. The accessible RV sites must be level and at least 16 feet wide to accommodate wheelchair lifts
3. The campground must have a minimum of 40 total campsites
4. The park must have at least one wheelchair-accessible paved trail that meets ADA specifications (minimum 36 inches wide with firm, paved surface)
5. The park must have an accessible fishing pier or accessible fishing area
6. The park must have ADA-compliant restrooms and showers accessible to campers
7. The park must allow pets in campground areas with leash requirements (typically 6 feet maximum leash length)
8. The park must accept advance reservations through an online reservation system
9. Reservations must be available at least 5 months in advance

For each of the four state parks, provide:
- The complete official name of the state park
- The state where it is located
- Supporting reference URLs from official park websites, state park systems, or reservation systems that verify the accessibility features and amenities
"""

ALLOWED_STATES = {"Georgia", "Florida", "Tennessee", "Arkansas", "Mississippi", "Alabama"}
STATE_ABBREV_TO_NAME = {
    "GA": "Georgia",
    "FL": "Florida",
    "TN": "Tennessee",
    "AR": "Arkansas",
    "MS": "Mississippi",
    "AL": "Alabama",
}


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class ParkURLs(BaseModel):
    basic_info: List[str] = Field(default_factory=list)
    rv_camping: List[str] = Field(default_factory=list)
    trail: List[str] = Field(default_factory=list)
    fishing: List[str] = Field(default_factory=list)
    restrooms: List[str] = Field(default_factory=list)
    pets: List[str] = Field(default_factory=list)
    reservations: List[str] = Field(default_factory=list)


class ParkFeatures(BaseModel):
    name: Optional[str] = None
    state: Optional[str] = None

    # RV camping
    rv_accessible_sites_count: Optional[str] = None
    rv_full_hookups: Optional[str] = None  # e.g., "full hookups", "W/S/E"
    rv_sites_level: Optional[str] = None   # e.g., "level", "nearly level"
    rv_site_min_width_ft: Optional[str] = None  # e.g., "16 ft", "18 feet"
    total_campsites: Optional[str] = None

    # Trails
    accessible_trail_exists: Optional[str] = None
    trail_min_width_inches: Optional[str] = None  # e.g., "36 in", "48 inches"
    trail_surface: Optional[str] = None  # e.g., "asphalt", "concrete", "boardwalk", "paved", "firm"

    # Fishing
    accessible_fishing: Optional[str] = None

    # Restrooms/Showers
    ada_restrooms_showers: Optional[str] = None

    # Pets
    pets_allowed: Optional[str] = None
    leash_length_max_ft: Optional[str] = None  # e.g., "6 ft"

    # Reservations
    online_reservations: Optional[str] = None
    reservation_window_months: Optional[str] = None  # e.g., "6 months", "11 months"

    urls: ParkURLs = Field(default_factory=ParkURLs)


class ParksExtraction(BaseModel):
    parks: List[ParkFeatures] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_parks() -> str:
    return """
    Extract at most four state parks mentioned in the answer that are in the southeastern U.S. (Georgia, Florida, Tennessee, Arkansas, Mississippi, Alabama).
    For each park, return the following fields using strings where possible (do not invent information not present in the answer; if missing, use null; for URLs use arrays of URLs exactly as shown in the answer):
    - name: complete official name of the state park as written in the answer
    - state: the U.S. state name or 2-letter abbreviation as stated in the answer
    - rv_accessible_sites_count: the stated number (or wording) of ADA-accessible RV sites
    - rv_full_hookups: the wording that indicates full hookups (e.g., "full hookups", "water/sewer/electric", "W/S/E")
    - rv_sites_level: the wording that indicates the accessible RV sites are level
    - rv_site_min_width_ft: the stated width specification for accessible RV pads (e.g., "16 feet", "18 ft")
    - total_campsites: the total number of campsites in the campground
    - accessible_trail_exists: wording indicating at least one wheelchair-accessible paved trail exists
    - trail_min_width_inches: stated minimum width of the accessible trail (in inches if mentioned)
    - trail_surface: stated surface type: paved/asphalt/concrete/boardwalk/firm (as mentioned)
    - accessible_fishing: wording indicating an accessible fishing pier/area exists
    - ada_restrooms_showers: wording indicating ADA-compliant restrooms/showers accessible to campers
    - pets_allowed: wording indicating pets are allowed in campground areas
    - leash_length_max_ft: the stated maximum leash length (e.g., "6 ft"), if mentioned
    - online_reservations: wording indicating advance online reservations are accepted
    - reservation_window_months: the stated advance booking window in months (e.g., "6 months", "11 months")
    - urls: an object with URL arrays:
        - basic_info: URLs for official park page(s) or reservation system page(s) that confirm name/location
        - rv_camping: URLs referencing RV site accessibility and hookups/capacity details
        - trail: URLs referencing accessible trail(s)
        - fishing: URLs referencing accessible fishing pier/area
        - restrooms: URLs referencing ADA restrooms/showers
        - pets: URLs referencing pet policy
        - reservations: URLs referencing online reservations and the advance window
    
    Notes:
    - Extract exactly what the answer states; do not infer or add details not present in the answer.
    - Include only valid and explicit URLs from the answer text.
    - If the answer lists more than four parks, return only the first four.
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def normalize_state_name(state: Optional[str]) -> Optional[str]:
    if not state:
        return None
    s = state.strip()
    if len(s) == 2:
        s = s.upper()
        return STATE_ABBREV_TO_NAME.get(s, None) or s
    # Title-case full names
    s_norm = s.title()
    # Handle common variants
    if s_norm in ALLOWED_STATES:
        return s_norm
    return s_norm  # Return as-is; verification will still check membership


def get_category_urls(park: ParkFeatures, category: str) -> List[str]:
    if not park or not park.urls:
        return []
    arr = getattr(park.urls, category, []) or []
    # De-duplicate while keeping order
    seen = set()
    deduped = []
    for u in arr:
        if u and u not in seen:
            seen.add(u)
            deduped.append(u)
    return deduped


def get_all_urls(park: ParkFeatures) -> List[str]:
    categories = ["basic_info", "rv_camping", "trail", "fishing", "restrooms", "pets", "reservations"]
    urls: List[str] = []
    seen = set()
    for cat in categories:
        for u in get_category_urls(park, cat):
            if u not in seen:
                urls.append(u)
                seen.add(u)
    return urls


def pick_sources(park: ParkFeatures, primary: List[str]) -> List[str]:
    return primary if primary else get_all_urls(park)


# --------------------------------------------------------------------------- #
# Verification builders                                                       #
# --------------------------------------------------------------------------- #
async def verify_single_park(evaluator: Evaluator, root, park: ParkFeatures, index: int) -> None:
    park_name = (park.name or "").strip()
    park_state_norm = normalize_state_name(park.state)

    # Park container node (NON-CRITICAL; categories will be marked CRITICAL)
    park_node = evaluator.add_parallel(
        id=f"park_{index+1}",
        desc=f"Park #{index+1}: {park_name or 'Unnamed'} - verification of all required features",
        parent=root,
        critical=False
    )

    # -------------------- Basic Info (CRITICAL) -------------------- #
    basic_node = evaluator.add_parallel(
        id=f"park_{index+1}_basicinfo",
        desc="Basic park identification and location information",
        parent=park_node,
        critical=True
    )

    # Name check (leaf, critical)
    name_leaf = evaluator.add_leaf(
        id=f"park_{index+1}_name",
        desc="Provide the complete official name of the state park",
        parent=basic_node,
        critical=True
    )
    name_sources = pick_sources(park, get_category_urls(park, "basic_info"))
    await evaluator.verify(
        claim=f"The official name of the park is '{park_name}'.",
        node=name_leaf,
        sources=name_sources,
        additional_instruction="Verify the park's official name as presented on its official page or reservation portal (e.g., page header or title). Allow reasonable punctuation or casing differences."
    )

    # State/location check (leaf, critical)
    state_leaf = evaluator.add_leaf(
        id=f"park_{index+1}_state",
        desc="State park must be located in Georgia, Florida, Tennessee, Arkansas, Mississippi, or Alabama",
        parent=basic_node,
        critical=True
    )
    allowed_list_str = ", ".join(sorted(ALLOWED_STATES))
    state_for_claim = park_state_norm or (park.state or "")
    await evaluator.verify(
        claim=f"This park is located in {state_for_claim}, which is one of the allowed states: {allowed_list_str}.",
        node=state_leaf,
        sources=name_sources,
        additional_instruction="Confirm the state location on the official park or state park system page. Treat state abbreviations (GA, FL, TN, AR, MS, AL) as equivalent to their full names."
    )

    # Basic info URL presence (leaf via custom judgment, critical under this category)
    basic_url_present = evaluator.add_custom_node(
        result=len(get_category_urls(park, "basic_info")) > 0,
        id=f"park_{index+1}_basicinfo_url",
        desc="Provide official park website or reservation system URL as reference for basic information",
        parent=basic_node,
        critical=True
    )

    # -------------------- RV Camping (CRITICAL) -------------------- #
    rv_node = evaluator.add_parallel(
        id=f"park_{index+1}_rv",
        desc="Accessible RV camping facilities with full hookups and adequate capacity",
        parent=park_node,
        critical=True
    )
    rv_sources = pick_sources(park, get_category_urls(park, "rv_camping"))

    # At least 2 ADA-accessible RV campsites
    rv_accessible_leaf = evaluator.add_leaf(
        id=f"park_{index+1}_rv_accessible_sites",
        desc="Park must have at least 2 ADA-accessible RV campsites",
        parent=rv_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"{park_name} has at least 2 ADA-accessible RV campsites (sites designated accessible/ADA/handicap).",
        node=rv_accessible_leaf,
        sources=rv_sources,
        additional_instruction="Look for mentions of accessible/ADA-designated RV sites, accessible pads, or specific site numbers labeled ADA. The requirement is at least two such sites."
    )

    # Full hookups for accessible RV sites
    rv_full_hookups_leaf = evaluator.add_leaf(
        id=f"park_{index+1}_rv_full_hookups",
        desc="Accessible RV sites must have full hookups (water, sewer, electric)",
        parent=rv_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The accessible RV sites at {park_name} provide full hookups (water, sewer, and electric).",
        node=rv_full_hookups_leaf,
        sources=rv_sources,
        additional_instruction="Accept equivalent wording such as 'full hookups', 'W/S/E', or explicit mention of water, sewer, and electric service at the site."
    )

    # Site specs: level and at least 16 feet wide
    rv_specs_leaf = evaluator.add_leaf(
        id=f"park_{index+1}_rv_site_specs",
        desc="Accessible sites must be level and adequately wide (minimum 16 feet) to accommodate wheelchair access",
        parent=rv_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The accessible RV sites at {park_name} are level and at least 16 feet wide.",
        node=rv_specs_leaf,
        sources=rv_sources,
        additional_instruction="Accept width stated in feet or inches (16 ft equals 192 inches). Wording like 'level', 'paved level pad', or 'level concrete pad' is acceptable."
    )

    # Total campsites >= 40
    total_sites_leaf = evaluator.add_leaf(
        id=f"park_{index+1}_rv_total_sites",
        desc="Campground must have at least 40 total campsites",
        parent=rv_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The campground at {park_name} has at least 40 total campsites.",
        node=total_sites_leaf,
        sources=rv_sources,
        additional_instruction="Confirm the total count from the official park or reservation page. Accept phrasing like 'X campsites' where X >= 40."
    )

    # RV camping URL presence (critical under this category)
    rv_url_present = evaluator.add_custom_node(
        result=len(get_category_urls(park, "rv_camping")) > 0,
        id=f"park_{index+1}_rv_url",
        desc="Provide URL reference for RV camping accessibility information",
        parent=rv_node,
        critical=True
    )

    # -------------------- Accessible Trail (CRITICAL) -------------------- #
    trail_node = evaluator.add_parallel(
        id=f"park_{index+1}_trail",
        desc="Wheelchair-accessible paved trail meeting ADA specifications",
        parent=park_node,
        critical=True
    )
    trail_sources = pick_sources(park, get_category_urls(park, "trail"))

    trail_exists_leaf = evaluator.add_leaf(
        id=f"park_{index+1}_trail_exists",
        desc="Park must have at least one wheelchair-accessible paved trail",
        parent=trail_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"{park_name} has at least one wheelchair-accessible paved trail.",
        node=trail_exists_leaf,
        sources=trail_sources,
        additional_instruction="Accept equivalent terms like accessible multi-use path, greenway, paved boardwalk. It must be specified as accessible or wheelchair friendly."
    )

    trail_specs_leaf = evaluator.add_leaf(
        id=f"park_{index+1}_trail_specs",
        desc="Trail must meet ADA specifications (minimum 36 inches wide, paved or firm surface)",
        parent=trail_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The accessible trail at {park_name} is at least 36 inches wide and has a firm, paved surface (e.g., asphalt, concrete, or boardwalk).",
        node=trail_specs_leaf,
        sources=trail_sources,
        additional_instruction="Confirm width where stated; otherwise confirm ADA-compliant width and surface. Accept surfaces described as paved/firm/hard-surfaced."
    )

    trail_url_present = evaluator.add_custom_node(
        result=len(get_category_urls(park, "trail")) > 0,
        id=f"park_{index+1}_trail_url",
        desc="Provide URL reference for accessible trail information",
        parent=trail_node,
        critical=True
    )

    # -------------------- Fishing Facility (CRITICAL) -------------------- #
    fish_node = evaluator.add_parallel(
        id=f"park_{index+1}_fishing",
        desc="Accessible fishing pier or accessible fishing area",
        parent=park_node,
        critical=True
    )
    fish_sources = pick_sources(park, get_category_urls(park, "fishing"))

    fish_leaf = evaluator.add_leaf(
        id=f"park_{index+1}_fishing_accessible",
        desc="Park must have accessible fishing pier or accessible fishing area",
        parent=fish_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"{park_name} provides an accessible fishing pier or an accessible fishing area.",
        node=fish_leaf,
        sources=fish_sources,
        additional_instruction="Look for phrases like 'accessible fishing pier', 'ADA-accessible pier', 'accessible fishing access'."
    )

    fish_url_present = evaluator.add_custom_node(
        result=len(get_category_urls(park, "fishing")) > 0,
        id=f"park_{index+1}_fishing_url",
        desc="Provide URL reference for accessible fishing facility information",
        parent=fish_node,
        critical=True
    )

    # -------------------- Restrooms/Showers (CRITICAL) -------------------- #
    rest_node = evaluator.add_parallel(
        id=f"park_{index+1}_restrooms",
        desc="ADA-compliant restroom and shower facilities",
        parent=park_node,
        critical=True
    )
    rest_sources = pick_sources(park, get_category_urls(park, "restrooms"))

    rest_leaf = evaluator.add_leaf(
        id=f"park_{index+1}_restrooms_accessible",
        desc="Park must have ADA-compliant restrooms and showers accessible to campers",
        parent=rest_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"{park_name} offers ADA-compliant restrooms and showers accessible to campers.",
        node=rest_leaf,
        sources=rest_sources,
        additional_instruction="Confirm ADA/accessible language for restrooms and showers in campground or nearby for camper use."
    )

    rest_url_present = evaluator.add_custom_node(
        result=len(get_category_urls(park, "restrooms")) > 0,
        id=f"park_{index+1}_restrooms_url",
        desc="Provide URL reference for accessible restroom information",
        parent=rest_node,
        critical=True
    )

    # -------------------- Pet Policy (CRITICAL) -------------------- #
    pet_node = evaluator.add_parallel(
        id=f"park_{index+1}_pets",
        desc="Pet-friendly camping policy with leash requirements",
        parent=park_node,
        critical=True
    )
    pet_sources = pick_sources(park, get_category_urls(park, "pets"))

    pets_leaf = evaluator.add_leaf(
        id=f"park_{index+1}_pets_allowed",
        desc="Park must allow pets in campground areas with leash requirements (typically 6 feet maximum)",
        parent=pet_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"Pets are allowed in the campground areas at {park_name} with a leash requirement of about 6 feet maximum.",
        node=pets_leaf,
        sources=pet_sources,
        additional_instruction="Accept common formulations like 'pets must be on a leash no longer than 6 feet' or similar language; policy can be on park or state parks policy page."
    )

    pet_url_present = evaluator.add_custom_node(
        result=len(get_category_urls(park, "pets")) > 0,
        id=f"park_{index+1}_pets_url",
        desc="Provide URL reference for pet policy information",
        parent=pet_node,
        critical=True
    )

    # -------------------- Reservations (CRITICAL) -------------------- #
    res_node = evaluator.add_parallel(
        id=f"park_{index+1}_reservations",
        desc="Online reservation system with adequate advance booking window",
        parent=park_node,
        critical=True
    )
    res_sources = pick_sources(park, get_category_urls(park, "reservations"))

    res_online_leaf = evaluator.add_leaf(
        id=f"park_{index+1}_reservations_online",
        desc="Park must accept advance reservations through an online reservation system",
        parent=res_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"{park_name} accepts advance reservations through an online reservation system.",
        node=res_online_leaf,
        sources=res_sources,
        additional_instruction="Accept official state reservation portals (e.g., ReserveAmerica/RA partners, state park booking systems) or the park's official booking page."
    )

    res_window_leaf = evaluator.add_leaf(
        id=f"park_{index+1}_reservations_window",
        desc="Reservations must be available at least 5 months in advance",
        parent=res_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"Reservations for {park_name} can be made at least 5 months in advance.",
        node=res_window_leaf,
        sources=res_sources,
        additional_instruction="Confirm minimum advance window; acceptable if policy states 6, 9, 11 or 12 months, etc., which satisfy 'at least 5 months'."
    )

    res_url_present = evaluator.add_custom_node(
        result=len(get_category_urls(park, "reservations")) > 0,
        id=f"park_{index+1}_reservations_url",
        desc="Provide URL reference for reservation system information",
        parent=res_node,
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
    # Initialize evaluator; Root kept NON-CRITICAL to allow partial scoring without
    # violating critical-child constraints in the framework.
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

    # Record allowed states as ground truth/context
    evaluator.add_ground_truth(
        {"allowed_states": sorted(list(ALLOWED_STATES))},
        gt_type="allowed_states"
    )

    # Extract parks and claimed details from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_parks(),
        template_class=ParksExtraction,
        extraction_name="parks_extraction"
    )

    # Normalize and keep only the first 4 parks; pad if fewer
    parks: List[ParkFeatures] = list(extracted.parks or [])[:4]
    while len(parks) < 4:
        parks.append(ParkFeatures())

    # Add a quick summary of URLs count per park for debugging
    summary_info = []
    for i, p in enumerate(parks, start=1):
        summary_info.append({
            "park_index": i,
            "name": p.name,
            "state": p.state,
            "url_counts": {
                "basic_info": len(get_category_urls(p, "basic_info")),
                "rv_camping": len(get_category_urls(p, "rv_camping")),
                "trail": len(get_category_urls(p, "trail")),
                "fishing": len(get_category_urls(p, "fishing")),
                "restrooms": len(get_category_urls(p, "restrooms")),
                "pets": len(get_category_urls(p, "pets")),
                "reservations": len(get_category_urls(p, "reservations")),
                "all_urls": len(get_all_urls(p)),
            }
        })
    evaluator.add_custom_info({"parks_url_summary": summary_info}, info_type="diagnostic", info_name="parks_url_summary")

    # Verify each park according to rubric
    for idx, park in enumerate(parks):
        await verify_single_park(evaluator, root, park, idx)

    # Return evaluation summary (includes verification tree and extraction)
    return evaluator.get_summary()