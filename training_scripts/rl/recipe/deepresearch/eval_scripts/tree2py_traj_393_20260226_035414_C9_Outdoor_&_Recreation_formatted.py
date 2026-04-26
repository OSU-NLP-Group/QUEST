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
TASK_ID = "ct_state_park_campgrounds_2025"
TASK_DESCRIPTION = """
Identify four Connecticut state park campgrounds that meet ALL of the following requirements: 
(1) The campground must offer campsites with both electric and water hookups; 
(2) The campground must have accessible (ADA-compliant) campsites available by reservation; 
(3) The campground must provide restroom facilities with showers; 
(4) The campground must have at least 80 total campsites; 
(5) The state park must contain hiking trails (either a single trail or combination of trails) totaling at least 2 miles in length AND with cumulative elevation gain of at least 150 feet; 
(6) Dogs must NOT be allowed in the campground area (following Connecticut state park campground policy); 
(7) The campground must be open for the 2025 camping season; 
(8) The campground must be reservable through the Connecticut ReserveAmerica online system; 
(9) The campground must provide accessible parking. 
For each of the four campgrounds, provide: the official name of the state park campground, the specific town or city location in Connecticut, a direct URL to the campground's page on either the Connecticut ReserveAmerica website or the official Connecticut State Parks website, a URL reference confirming the facility amenities (hookups, accessible sites, restrooms/showers, site count), a URL reference confirming the hiking trail specifications (length and elevation gain), a URL reference confirming the policies (pet policy, 2025 season dates, reservation system), and a URL reference confirming the accessibility features (accessible parking).
"""

MIN_TOTAL_SITES = 80
MIN_TRAIL_MILES = 2.0
MIN_ELEV_GAIN_FT = 150
SEASON_YEAR = 2025


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class CampgroundItem(BaseModel):
    """One campground entry extracted from the answer."""
    official_name: Optional[str] = None
    town_city: Optional[str] = None
    primary_url: Optional[str] = None  # ReserveAmerica or CT State Parks official page
    facilities_url: Optional[str] = None  # Amenities/supporting info page
    trails_url: Optional[str] = None      # Trail specs page
    policies_url: Optional[str] = None    # Policies and season/reservations page
    accessibility_url: Optional[str] = None  # Accessibility features page (accessible parking)


class CampgroundsExtraction(BaseModel):
    """All campgrounds extracted from the answer."""
    campgrounds: List[CampgroundItem] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_campgrounds() -> str:
    return """
    Extract up to six Connecticut state park campgrounds listed in the answer. For each campground, return:
    - official_name: The official state park campground name (string).
    - town_city: The specific town or city in Connecticut (string).
    - primary_url: A direct URL to the campground page on either ReserveAmerica (Connecticut) or the official Connecticut State Parks/DEEP website.
    - facilities_url: A URL reference confirming amenities (electric+water hookups, accessible campsites, restrooms/showers, and total site count).
    - trails_url: A URL reference confirming hiking trail specifications (total length >= 2 miles and elevation gain >= 150 ft).
    - policies_url: A URL reference confirming policies (dogs not allowed in campground), the 2025 camping season open, and that reservations are handled via Connecticut ReserveAmerica.
    - accessibility_url: A URL reference confirming accessibility features (including accessible parking).
    
    Rules:
    - Extract only URLs explicitly present in the answer.
    - If a specific URL for an aspect is not provided, set it to null.
    - If official_name or town_city is missing, set it to null.
    - Do not invent or infer URLs or data not present in the answer text.
    Return a JSON object with a single field "campgrounds" which is an array of objects of the above schema.
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _ordinal(n: int) -> str:
    mapping = {1: "first", 2: "second", 3: "third", 4: "fourth"}
    return mapping.get(n, f"#{n}")


def _non_empty_urls(urls: List[Optional[str]]) -> List[str]:
    return [u for u in urls if isinstance(u, str) and u.strip()]


# --------------------------------------------------------------------------- #
# Verification builders                                                       #
# --------------------------------------------------------------------------- #
async def build_basic_info_nodes(
    evaluator: Evaluator,
    parent_node,
    cg: CampgroundItem,
    idx: int
) -> Dict[str, Any]:
    """
    Build and verify 'Basic Info' subtree for one campground.
    Returns a dictionary with reference leaves for possible prerequisites (e.g., URL leaf).
    """
    basic_node = evaluator.add_parallel(
        id=f"Campground_{idx}_Basic_Info",
        desc=f"Basic identification and location information for the {_ordinal(idx)} campground",
        parent=parent_node,
        critical=True
    )

    # Name provided (existence check)
    evaluator.add_custom_node(
        result=bool(cg.official_name and cg.official_name.strip()),
        id=f"Campground_{idx}_Name",
        desc="Provide the official name of the state park campground",
        parent=basic_node,
        critical=True
    )

    # Location provided (existence check)
    evaluator.add_custom_node(
        result=bool(cg.town_city and cg.town_city.strip()),
        id=f"Campground_{idx}_Location",
        desc="Provide the specific town/city location in Connecticut",
        parent=basic_node,
        critical=True
    )

    # URL: verify that the provided URL is an official ReserveAmerica or CT State Parks page and corresponds to the campground
    url_leaf = evaluator.add_leaf(
        id=f"Campground_{idx}_URL",
        desc="Provide official ReserveAmerica or CT State Parks URL for this campground",
        parent=basic_node,
        critical=True
    )
    primary_url = cg.primary_url or None
    url_claim = (
        f"This page is the official ReserveAmerica or Connecticut State Parks page for the campground "
        f"'{cg.official_name or 'unknown'}' in Connecticut."
    )
    await evaluator.verify(
        claim=url_claim,
        node=url_leaf,
        sources=primary_url,
        additional_instruction=(
            "Confirm the page is either on ReserveAmerica (Connecticut) or an official CT State Parks/DEEP domain. "
            "The page should clearly correspond to the named campground. "
            "If the URL is missing or non-official, consider the claim not supported."
        ),
    )

    return {"basic_url_leaf": url_leaf}


async def build_facilities_nodes(
    evaluator: Evaluator,
    parent_node,
    cg: CampgroundItem,
    idx: int
) -> Dict[str, Any]:
    facilities_node = evaluator.add_parallel(
        id=f"Campground_{idx}_Facilities",
        desc=f"Camping facility requirements for the {_ordinal(idx)} campground",
        parent=parent_node,
        critical=True
    )

    # Facilities URL presence (used as prerequisite)
    fac_url_leaf = evaluator.add_custom_node(
        result=bool(cg.facilities_url and cg.facilities_url.strip()),
        id=f"Campground_{idx}_Facilities_URL",
        desc="URL reference confirming facility amenities",
        parent=facilities_node,
        critical=True
    )

    # Electric + Water hookups
    hookups_leaf = evaluator.add_leaf(
        id=f"Campground_{idx}_Electric_Water_Hookups",
        desc="Campground offers campsites with both electric and water hookups",
        parent=facilities_node,
        critical=True
    )
    hookups_claim = "The campground offers campsites with both electric and water hookups."
    await evaluator.verify(
        claim=hookups_claim,
        node=hookups_leaf,
        sources=_non_empty_urls([cg.facilities_url, cg.primary_url]),
        additional_instruction="Look for amenities listing that explicitly mentions both electric hookups and water hookups at campsites.",
        extra_prerequisites=[fac_url_leaf],
    )

    # Accessible (ADA) campsites available by reservation
    accessible_sites_leaf = evaluator.add_leaf(
        id=f"Campground_{idx}_Accessible_Sites",
        desc="Campground has accessible (ADA-compliant) campsites available by reservation",
        parent=facilities_node,
        critical=True
    )
    accessible_sites_claim = "The campground has accessible (ADA-compliant) campsites available by reservation."
    await evaluator.verify(
        claim=accessible_sites_claim,
        node=accessible_sites_leaf,
        sources=_non_empty_urls([cg.facilities_url, cg.primary_url]),
        additional_instruction="Confirm ADA/accessible campsites and that they are reservable (e.g., marked as accessible in the booking system or noted on the official page).",
        extra_prerequisites=[fac_url_leaf],
    )

    # Restrooms with showers
    showers_leaf = evaluator.add_leaf(
        id=f"Campground_{idx}_Restrooms_Showers",
        desc="Campground provides restroom facilities with showers",
        parent=facilities_node,
        critical=True
    )
    showers_claim = "The campground provides restroom facilities with showers."
    await evaluator.verify(
        claim=showers_claim,
        node=showers_leaf,
        sources=_non_empty_urls([cg.facilities_url, cg.primary_url]),
        additional_instruction="Look for bathhouse or restroom amenities explicitly specifying showers.",
        extra_prerequisites=[fac_url_leaf],
    )

    # Minimum total sites >= 80
    sites_leaf = evaluator.add_leaf(
        id=f"Campground_{idx}_Minimum_Sites",
        desc="Campground has at least 80 total campsites",
        parent=facilities_node,
        critical=True
    )
    sites_claim = f"The campground has at least {MIN_TOTAL_SITES} total campsites."
    await evaluator.verify(
        claim=sites_claim,
        node=sites_leaf,
        sources=_non_empty_urls([cg.facilities_url, cg.primary_url]),
        additional_instruction=f"Confirm total site count is >= {MIN_TOTAL_SITES}. Allow reasonable rounding or seasonal variation if clearly stated.",
        extra_prerequisites=[fac_url_leaf],
    )

    return {"fac_url_leaf": fac_url_leaf}


async def build_trails_nodes(
    evaluator: Evaluator,
    parent_node,
    cg: CampgroundItem,
    idx: int
) -> Dict[str, Any]:
    trails_node = evaluator.add_parallel(
        id=f"Campground_{idx}_Trails",
        desc=f"Hiking trail requirements within the {_ordinal(idx)} campground's park",
        parent=parent_node,
        critical=True
    )

    # Trails URL presence
    trail_url_leaf = evaluator.add_custom_node(
        result=bool(cg.trails_url and cg.trails_url.strip()),
        id=f"Campground_{idx}_Trail_URL",
        desc="URL reference confirming trail specifications",
        parent=trails_node,
        critical=True
    )

    # Total trail length >= MIN_TRAIL_MILES
    trail_len_leaf = evaluator.add_leaf(
        id=f"Campground_{idx}_Trail_Length",
        desc="Trails total at least 2 miles in length",
        parent=trails_node,
        critical=True
    )
    trail_len_claim = f"The state park's hiking trail system totals at least {MIN_TRAIL_MILES} miles in length (sum of one or more trails)."
    await evaluator.verify(
        claim=trail_len_claim,
        node=trail_len_leaf,
        sources=_non_empty_urls([cg.trails_url]),
        additional_instruction="Consider the sum of multiple trails or a single listed loop/trail; use miles or km conversion as needed. Reasonable rounding acceptable.",
        extra_prerequisites=[trail_url_leaf],
    )

    # Elevation gain >= MIN_ELEV_GAIN_FT
    trail_elev_leaf = evaluator.add_leaf(
        id=f"Campground_{idx}_Trail_Elevation",
        desc="Trails have cumulative elevation gain of at least 150 feet",
        parent=trails_node,
        critical=True
    )
    trail_elev_claim = f"The park's hiking trails have cumulative elevation gain of at least {MIN_ELEV_GAIN_FT} feet."
    await evaluator.verify(
        claim=trail_elev_claim,
        node=trail_elev_leaf,
        sources=_non_empty_urls([cg.trails_url]),
        additional_instruction="If elevation is shown per trail, consider typical loop or additive gain across network where indicated. Minor variations acceptable.",
        extra_prerequisites=[trail_url_leaf],
    )

    return {"trail_url_leaf": trail_url_leaf}


async def build_policies_nodes(
    evaluator: Evaluator,
    parent_node,
    cg: CampgroundItem,
    idx: int
) -> Dict[str, Any]:
    policies_node = evaluator.add_parallel(
        id=f"Campground_{idx}_Policies",
        desc=f"Policy and operational requirements for the {_ordinal(idx)} campground",
        parent=parent_node,
        critical=True
    )

    # Policies URL presence
    policy_url_leaf = evaluator.add_custom_node(
        result=bool(cg.policies_url and cg.policies_url.strip()),
        id=f"Campground_{idx}_Policy_URL",
        desc="URL reference confirming policies and season dates",
        parent=policies_node,
        critical=True
    )

    # Dogs NOT allowed in campground area
    no_dogs_leaf = evaluator.add_leaf(
        id=f"Campground_{idx}_No_Dogs",
        desc="Dogs are NOT allowed in the campground (Connecticut state park campground policy)",
        parent=policies_node,
        critical=True
    )
    no_dogs_claim = "Dogs are NOT allowed in the campground area."
    await evaluator.verify(
        claim=no_dogs_claim,
        node=no_dogs_leaf,
        sources=_non_empty_urls([cg.policies_url, cg.primary_url]),
        additional_instruction="CT State Parks camping policy generally prohibits pets in camping areas. Verify explicit 'no pets/dogs in campground' language for this specific campground.",
        extra_prerequisites=[policy_url_leaf],
    )

    # Open for the 2025 camping season
    season_leaf = evaluator.add_leaf(
        id=f"Campground_{idx}_2025_Season",
        desc="Campground is open for the 2025 camping season",
        parent=policies_node,
        critical=True
    )
    season_claim = f"The campground is open for the {SEASON_YEAR} camping season."
    await evaluator.verify(
        claim=season_claim,
        node=season_leaf,
        sources=_non_empty_urls([cg.policies_url, cg.primary_url]),
        additional_instruction=f"Confirm posted operating season includes year {SEASON_YEAR} or clearly states 2025 season dates/open status.",
        extra_prerequisites=[policy_url_leaf],
    )

    # Reservable via Connecticut ReserveAmerica system
    ra_leaf = evaluator.add_leaf(
        id=f"Campground_{idx}_ReserveAmerica",
        desc="Campground is reservable through Connecticut ReserveAmerica system",
        parent=policies_node,
        critical=True
    )
    ra_claim = "Reservations for this campground are made via the Connecticut ReserveAmerica online system."
    await evaluator.verify(
        claim=ra_claim,
        node=ra_leaf,
        sources=_non_empty_urls([cg.primary_url, cg.policies_url]),
        additional_instruction="Verify that reservations are handled through ReserveAmerica (Connecticut). Accept RA domain pages or CT pages that explicitly direct to ReserveAmerica.",
        extra_prerequisites=[policy_url_leaf],
    )

    return {"policy_url_leaf": policy_url_leaf}


async def build_accessibility_nodes(
    evaluator: Evaluator,
    parent_node,
    cg: CampgroundItem,
    idx: int
) -> Dict[str, Any]:
    accessibility_node = evaluator.add_parallel(
        id=f"Campground_{idx}_Accessibility",
        desc=f"Accessibility features for the {_ordinal(idx)} campground",
        parent=parent_node,
        critical=True
    )

    # Accessibility URL presence
    acc_url_leaf = evaluator.add_custom_node(
        result=bool(cg.accessibility_url and cg.accessibility_url.strip()),
        id=f"Campground_{idx}_Accessibility_URL",
        desc="URL reference confirming accessibility features",
        parent=accessibility_node,
        critical=True
    )

    # Accessible parking
    acc_parking_leaf = evaluator.add_leaf(
        id=f"Campground_{idx}_Accessible_Parking",
        desc="Campground provides accessible parking",
        parent=accessibility_node,
        critical=True
    )
    acc_parking_claim = "The campground provides accessible parking."
    await evaluator.verify(
        claim=acc_parking_claim,
        node=acc_parking_leaf,
        sources=_non_empty_urls([cg.accessibility_url, cg.primary_url]),
        additional_instruction="Look for ADA/accessible parking indications, accessible parking icons, or text confirming accessible parking availability.",
        extra_prerequisites=[acc_url_leaf],
    )

    return {"accessibility_url_leaf": acc_url_leaf}


async def verify_campground(
    evaluator: Evaluator,
    parent_node,
    cg: CampgroundItem,
    idx: int
) -> None:
    """
    Build and verify all subtrees for one campground.
    """
    # Campground aggregate node (non-critical to allow partial credit across the four)
    cg_node = evaluator.add_parallel(
        id=f"Campground_{idx}",
        desc=f"{_ordinal(idx).capitalize()} qualifying Connecticut state park campground meeting all requirements",
        parent=parent_node,
        critical=False
    )

    # Build and verify each category
    ref_basic = await build_basic_info_nodes(evaluator, cg_node, cg, idx)
    ref_fac = await build_facilities_nodes(evaluator, cg_node, cg, idx)
    ref_trails = await build_trails_nodes(evaluator, cg_node, cg, idx)
    ref_policies = await build_policies_nodes(evaluator, cg_node, cg, idx)
    ref_access = await build_accessibility_nodes(evaluator, cg_node, cg, idx)

    # Optionally attach custom info about used URLs for debugging
    used_urls = {
        "primary_url": cg.primary_url,
        "facilities_url": cg.facilities_url,
        "trails_url": cg.trails_url,
        "policies_url": cg.policies_url,
        "accessibility_url": cg.accessibility_url,
    }
    evaluator.add_custom_info(used_urls, info_type=f"campground_{idx}_urls")


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
    Evaluate an answer for the Connecticut state park campgrounds task.
    """
    # Initialize evaluator
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Root aggregates 4 campgrounds independently
        agent_name=agent_name,
        answer_name=answer_name,
        client=client,
        task_description="Identify four Connecticut state park campgrounds that meet all specified camping, hiking, accessibility, and policy requirements",
        answer=answer,
        global_cache=cache,
        global_semaphore=semaphore,
        logger=logger,
        default_model=model,
    )

    # Record constraints for transparency
    evaluator.add_custom_info(
        {
            "min_total_sites": MIN_TOTAL_SITES,
            "min_trail_miles": MIN_TRAIL_MILES,
            "min_elevation_gain_ft": MIN_ELEV_GAIN_FT,
            "season_year": SEASON_YEAR,
            "reservation_system": "ReserveAmerica (Connecticut)"
        },
        info_type="constraints"
    )

    # Extract campgrounds from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_campgrounds(),
        template_class=CampgroundsExtraction,
        extraction_name="campgrounds_extraction"
    )

    # Filter to first 4 campgrounds; pad with empty items if fewer
    campgrounds = list(extracted.campgrounds[:4])
    while len(campgrounds) < 4:
        campgrounds.append(CampgroundItem())

    # Build verification for each campground
    for i, cg in enumerate(campgrounds, start=1):
        await verify_campground(evaluator, root, cg, i)

    # Return structured summary
    return evaluator.get_summary()