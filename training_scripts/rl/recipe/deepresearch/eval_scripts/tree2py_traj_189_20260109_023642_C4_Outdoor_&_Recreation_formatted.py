import asyncio
import logging
import re
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "nv_state_park_campground_max_sites"
TASK_DESCRIPTION = """
Identify the Nevada state park campground that has the most total campsites and meets the following requirements: offers both water and electric hookups for RV camping, operates year-round (all 12 months), and is reservable through the Nevada State Parks reservation system. Provide the park name, the total number of campsites, and a reference URL from an official Nevada State Parks source.
"""


# --------------------------------------------------------------------------- #
# Data models for structured extraction                                       #
# --------------------------------------------------------------------------- #
class CampgroundSelection(BaseModel):
    """
    Extract the core elements from the answer for the identified Nevada State Park campground.
    All URLs must be explicitly present in the answer text.
    """
    park_name: Optional[str] = None                         # e.g., "Valley of Fire State Park"
    campground_name: Optional[str] = None                   # e.g., "Atlatl Rock Campground" (optional)
    total_campsites: Optional[str] = None                   # e.g., "72", "72 sites", or "72 total campsites"
    # Official Nevada State Parks URLs (parks.nv.gov subdomains, reservation portals run by NV State Parks)
    official_urls: List[str] = Field(default_factory=list)
    # Reservation portal URLs used by Nevada State Parks for booking (if provided in the answer)
    reservation_urls: List[str] = Field(default_factory=list)
    # All URLs cited in the answer (catch-all)
    all_urls: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt helpers                                                   #
# --------------------------------------------------------------------------- #
def prompt_extract_campground_selection() -> str:
    return """
    From the answer, extract the single Nevada State Park campground the answer identifies as the one with the most total campsites that also:
    - offers both water and electric hookups for RV camping,
    - operates year-round (all 12 months),
    - and is reservable through the Nevada State Parks reservation system.

    Extract the following fields (use null if missing):
    - park_name: The Nevada State Park name (e.g., "Valley of Fire State Park").
    - campground_name: The specific campground name if separate from the park (e.g., "Atlatl Rock Campground"). If the answer only names the park, set this to null.
    - total_campsites: The total number of campsites cited for the selected campground (use the exact text or number provided, e.g., "72" or "72 sites").
    - official_urls: A list of URLs from official Nevada State Parks sources, specifically domains containing "parks.nv.gov" (including subdomains like shop.parks.nv.gov or reservation.parks.nv.gov). Only include URLs that are explicitly present in the answer.
    - reservation_urls: A list of reservation/booking URLs used by Nevada State Parks if they are explicitly present in the answer (for example, links from parks.nv.gov to the booking portal, or branded Nevada State Parks reservation portals). Only include URLs that are explicitly present.
    - all_urls: A list of all URLs present in the answer (including non-official sites). If none, return an empty list.

    Rules:
    - Do not fabricate or infer values not present in the answer.
    - Keep URLs exactly as shown in the answer (valid URLs only).
    - If both park and campground names are given, keep both as separate fields.
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _dedup_preserve(seq: List[str]) -> List[str]:
    seen = {}
    for s in seq:
        if s is None:
            continue
        s = s.strip()
        if not s:
            continue
        if s not in seen:
            seen[s] = True
    return list(seen.keys())


def is_official_nv_parks_url(url: str) -> bool:
    """
    Heuristic check for an official Nevada State Parks URL.
    We treat the following as official:
      - Any URL containing "parks.nv.gov" (including subdomains such as reservation.parks.nv.gov, shop.parks.nv.gov)
      - Any URL containing "reservenevada" (some states brand their booking portals; being lenient here)
    """
    if not url:
        return False
    u = url.lower()
    return ("parks.nv.gov" in u) or ("reservenevada" in u) or ("reservation.parks.nv" in u)


def select_display_name(info: CampgroundSelection) -> str:
    if info.campground_name and info.campground_name.strip():
        return info.campground_name.strip()
    if info.park_name and info.park_name.strip():
        return info.park_name.strip()
    return "the selected campground"


def merge_sources_prefer_official(info: CampgroundSelection) -> List[str]:
    """
    Prefer official Nevada State Parks URLs, else fall back to any URLs.
    """
    official = [u for u in (info.official_urls or []) if is_official_nv_parks_url(u)]
    official = _dedup_preserve(official)
    if official:
        return official
    return _dedup_preserve(info.all_urls or [])


def reservation_sources(info: CampgroundSelection) -> List[str]:
    """
    For reservation checks, combine reservation URLs and official URLs.
    """
    cand = []
    cand.extend(info.reservation_urls or [])
    cand.extend(info.official_urls or [])
    # Keep only official-ish or clearly reservation-branded, if possible
    filtered = [u for u in cand if (is_official_nv_parks_url(u) or "reserve" in u.lower() or "reservation" in u.lower())]
    final_list = _dedup_preserve(filtered if filtered else cand)
    if final_list:
        return final_list
    # fall back to all urls if none found
    return _dedup_preserve(info.all_urls or [])


# --------------------------------------------------------------------------- #
# Verification logic                                                          #
# --------------------------------------------------------------------------- #
async def build_and_verify_tree(evaluator: Evaluator, root, info: CampgroundSelection) -> None:
    """
    Build the verification tree from the rubric and run all checks.
    """

    # Create the main critical node representing the whole identification rubric
    main_node = evaluator.add_parallel(
        id="Nevada_State_Park_Campground_Identification",
        desc="Verify that the identified campground meets all specified requirements and that required outputs are provided",
        parent=root,
        critical=True
    )

    # Convenience values
    display_name = select_display_name(info)
    official_srcs = merge_sources_prefer_official(info)
    reserve_srcs = reservation_sources(info)
    all_srcs = _dedup_preserve((info.all_urls or []) + official_srcs + reserve_srcs)

    # Park_Name_Provided (Existence check -> custom critical node)
    evaluator.add_custom_node(
        result=bool((info.park_name and info.park_name.strip()) or (info.campground_name and info.campground_name.strip())),
        id="Park_Name_Provided",
        desc="The answer explicitly provides the park/campground name being identified",
        parent=main_node,
        critical=True
    )

    # Nevada_Location
    node_nv_loc = evaluator.add_leaf(
        id="Nevada_Location",
        desc="The campground is located in Nevada",
        parent=main_node,
        critical=True
    )
    claim_nv_loc = f"{display_name} is located in Nevada."
    await evaluator.verify(
        claim=claim_nv_loc,
        node=node_nv_loc,
        sources=official_srcs,
        additional_instruction="Verify that the official page indicates the location is in the U.S. state of Nevada (NV). Allow reasonable variations like 'Nevada State Park' implying Nevada."
    )

    # State_Park_Status
    node_state_park = evaluator.add_leaf(
        id="State_Park_Status",
        desc="The facility is a Nevada state park (not a national park, county park, or private campground)",
        parent=main_node,
        critical=True
    )
    claim_state_park = f"{display_name} is a Nevada State Parks facility (a Nevada state park campground)."
    await evaluator.verify(
        claim=claim_state_park,
        node=node_state_park,
        sources=official_srcs,
        additional_instruction="Confirm that the campground/park is explicitly operated by Nevada State Parks and is a state park facility, not federal, county, or private."
    )

    # Water_Hookups
    node_water = evaluator.add_leaf(
        id="Water_Hookups",
        desc="The campground offers water hookups for RV camping",
        parent=main_node,
        critical=True
    )
    claim_water = f"The campground {display_name} offers water hookups for RV camping (at some or all RV sites)."
    await evaluator.verify(
        claim=claim_water,
        node=node_water,
        sources=official_srcs,
        additional_instruction="Look for phrases like 'water hookups', 'water at sites', or 'full hookups'. If at least some sites have water hookups, this should be considered True."
    )

    # Electric_Hookups
    node_electric = evaluator.add_leaf(
        id="Electric_Hookups",
        desc="The campground offers electric hookups for RV camping",
        parent=main_node,
        critical=True
    )
    claim_electric = f"The campground {display_name} offers electric hookups for RV camping (at some or all RV sites)."
    await evaluator.verify(
        claim=claim_electric,
        node=node_electric,
        sources=official_srcs,
        additional_instruction="Look for 'electric hookups', amperage references like '30-amp' or '50-amp', or 'power hookups'. If at least some sites have electric, consider this True."
    )

    # Year_Round_Operation
    node_year_round = evaluator.add_leaf(
        id="Year_Round_Operation",
        desc="The campground operates year-round (open all 12 months)",
        parent=main_node,
        critical=True
    )
    claim_year_round = f"The campground {display_name} operates year-round (open all 12 months)."
    await evaluator.verify(
        claim=claim_year_round,
        node=node_year_round,
        sources=official_srcs,
        additional_instruction="Confirm that the campground is described as 'open year-round', 'open all year', or equivalent. If the page indicates seasonal closure, this should be False."
    )

    # Reservation_System
    node_res_sys = evaluator.add_leaf(
        id="Reservation_System",
        desc="The campground is reservable through the Nevada State Parks reservation system",
        parent=main_node,
        critical=True
    )
    claim_res_sys = f"The campground {display_name} can be reserved through the Nevada State Parks reservation system or official booking portal."
    await evaluator.verify(
        claim=claim_res_sys,
        node=node_res_sys,
        sources=reserve_srcs if reserve_srcs else official_srcs,
        additional_instruction="Look for an official booking link or explicit instruction to reserve via the Nevada State Parks system (often linked from parks.nv.gov). Mentions of the official reservation portal satisfy this."
    )

    # Campsite_Count (Group: Provided + Accurate)
    campsite_group = evaluator.add_parallel(
        id="Campsite_Count",
        desc="The total number of campsites is provided and accurately stated",
        parent=main_node,
        critical=True
    )

    evaluator.add_custom_node(
        result=bool(info.total_campsites and info.total_campsites.strip()),
        id="Campsite_Count_Provided",
        desc="The answer provides the total number of campsites",
        parent=campsite_group,
        critical=True
    )

    node_campsite_accurate = evaluator.add_leaf(
        id="Campsite_Count_Accurate",
        desc="The total number of campsites is accurately stated",
        parent=campsite_group,
        critical=True
    )
    # If not provided, still perform verification (will likely fail)
    total_text = (info.total_campsites or "").strip()
    claim_count = f"The total number of campsites at {display_name} is {total_text}."
    await evaluator.verify(
        claim=claim_count,
        node=node_campsite_accurate,
        sources=official_srcs,
        additional_instruction="Verify the stated total campsite count on the official page. Allow minor formatting differences (e.g., '72', '72 sites'). If multiple areas/campgrounds exist, ensure the count corresponds to the specific campground identified in the answer."
    )

    # Maximum_Among_Qualifying
    node_maximum = evaluator.add_leaf(
        id="Maximum_Among_Qualifying",
        desc="Among campgrounds meeting all other criteria, this campground has the most total campsites",
        parent=main_node,
        critical=True
    )
    claim_max = (
        f"Among Nevada State Park campgrounds that operate year-round, offer both water and electric hookups, "
        f"and are reservable through the Nevada State Parks reservation system, {display_name} has the most total campsites."
    )
    await evaluator.verify(
        claim=claim_max,
        node=node_maximum,
        sources=all_srcs,
        additional_instruction=(
            "Use the provided sources to determine whether the selected campground truly has the greatest total number of campsites "
            "among Nevada State Park campgrounds that also meet the specified criteria (both water and electric hookups, year-round, reservable). "
            "If the sources do not provide sufficient comparative evidence, mark this as not supported."
        )
    )

    # Reference_URL (Group: valid/official + relevance)
    ref_group = evaluator.add_parallel(
        id="Reference_URL",
        desc="A valid reference URL from parks.nv.gov or another official Nevada State Parks source is provided",
        parent=main_node,
        critical=True
    )

    has_official_ref = any(is_official_nv_parks_url(u) for u in (info.official_urls or []))
    evaluator.add_custom_node(
        result=has_official_ref,
        id="Reference_URL_Provided_Official",
        desc="An official Nevada State Parks URL (e.g., parks.nv.gov) is provided in the answer",
        parent=ref_group,
        critical=True
    )

    node_ref_relevant = evaluator.add_leaf(
        id="Reference_URL_Relevant",
        desc="The official reference URL is relevant (about the identified park/campground)",
        parent=ref_group,
        critical=True
    )
    # Prefer the first official URL; fall back to any URL
    ref_sources = _dedup_preserve([u for u in (info.official_urls or []) if is_official_nv_parks_url(u)])
    if not ref_sources:
        ref_sources = official_srcs if official_srcs else all_srcs
    claim_ref_relevant = f"This webpage is an official Nevada State Parks page about {display_name}."
    await evaluator.verify(
        claim=claim_ref_relevant,
        node=node_ref_relevant,
        sources=ref_sources,
        additional_instruction="Confirm that the page is an official parks.nv.gov (or equivalent official NV State Parks) page clearly about the identified park/campground."
    )

    # Record some helpful custom info for debugging
    evaluator.add_custom_info(
        info={
            "display_name": display_name,
            "park_name": info.park_name,
            "campground_name": info.campground_name,
            "total_campsites": info.total_campsites,
            "official_urls_used": ref_sources,
            "reservation_sources_used": reserve_srcs,
            "all_sources_considered": all_srcs
        },
        info_type="debug_info",
        info_name="verification_inputs"
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
    Evaluate an answer for the Nevada state park campground maximum-sites task.
    """
    # Initialize evaluator and root
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,   # Root aggregation strategy
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

    # Extract core selection details from the answer
    extraction: CampgroundSelection = await evaluator.extract(
        prompt=prompt_extract_campground_selection(),
        template_class=CampgroundSelection,
        extraction_name="campground_selection"
    )

    # Build and run verification tree
    await build_and_verify_tree(evaluator, root, extraction)

    # Return structured summary
    return evaluator.get_summary()