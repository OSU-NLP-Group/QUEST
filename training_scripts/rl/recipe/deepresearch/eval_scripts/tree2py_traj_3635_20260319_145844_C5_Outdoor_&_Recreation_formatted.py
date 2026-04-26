import asyncio
import logging
from typing import Optional, List, Dict, Any, Tuple
from urllib.parse import urlparse

from pydantic import BaseModel, Field
from obj_task_eval.llm_client.base_client import LLMClient

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "fl_state_parks_regions_camping_beach_hiking"
TASK_DESCRIPTION = """
Identify three Florida state parks that each offer overnight camping facilities, direct beach access for swimming, and designated hiking trails open to general visitors. The three parks must represent different coastal regions of Florida: one from the Gulf Coast, one from the Atlantic Coast, and one from either the Florida Keys or South Florida coastal area. All parks must be accessible by car (not requiring ferry or boat access). For each park, provide: (1) The official park name, (2) The park's physical address, (3) A reference URL from the official Florida State Parks website (floridastateparks.org), (4) Confirmation that camping facilities are available (specify type: tent sites, RV sites, or cabins), (5) Confirmation of beach access for swimming, and (6) Confirmation of hiking trails (with brief trail information if available).
""".strip()


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class ParkItem(BaseModel):
    official_name: Optional[str] = None
    physical_address: Optional[str] = None
    fsp_url: Optional[str] = None  # Must be a URL on floridastateparks.org if provided
    camping_available: Optional[bool] = None
    camping_types: List[str] = Field(default_factory=list)  # e.g., ["tent", "RV", "cabins"]
    beach_swimming_access: Optional[bool] = None
    hiking_trails_available: Optional[bool] = None
    brief_trail_info: Optional[str] = None
    car_accessible: Optional[bool] = None
    region_claim: Optional[str] = None  # One of: "gulf", "atlantic", "keys", "south_florida"
    amenities_note: Optional[str] = None  # Any mention of amenities (restrooms, picnic area, parking, etc.)


class ParksExtraction(BaseModel):
    parks: List[ParkItem] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_parks() -> str:
    return """
    Extract all Florida State Parks mentioned in the answer (the agent's response). For each park, extract the following fields exactly as claimed in the answer text:
    - official_name: The official name of the park as written in the answer.
    - physical_address: The park's physical or mailing address as provided in the answer (single string).
    - fsp_url: A URL on the official Florida State Parks website for the specific park (must be a URL under 'floridastateparks.org'). If the answer does not provide a URL or provides a non-FSP URL, set this field to null.
    - camping_available: true/false if the answer explicitly confirms overnight camping is available. If unclear or not stated, set to null.
    - camping_types: A list of camping types explicitly mentioned (e.g., ["tent", "RV", "cabins"]). If not specified, return an empty list.
    - beach_swimming_access: true/false if the answer explicitly confirms direct beach access for swimming. If unclear or not stated, set to null.
    - hiking_trails_available: true/false if the answer explicitly confirms designated hiking trails open to general visitors. If unclear or not stated, set to null.
    - brief_trail_info: A brief trail note if the answer provides it (e.g., a named trail or length). If absent, set to null.
    - car_accessible: true/false if the answer explicitly confirms the park is accessible by car without a ferry/boat requirement. If unclear or not stated, set to null.
    - region_claim: If the answer explicitly assigns a region category for this park, extract one of: "gulf", "atlantic", "keys", "south_florida". If no explicit region is stated, set to null.
    - amenities_note: Any short note about basic visitor amenities (e.g., restrooms, picnic areas, parking) if provided; otherwise null.

    Important rules:
    - Do not invent information. Only extract what is explicitly present in the answer text.
    - If more than three parks are mentioned, still extract them all; downstream evaluation will only consider the first three slots but will also check if exactly three were provided.
    - fsp_url must be on 'floridastateparks.org'. If the answer provides a different domain, set fsp_url to null.
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def _is_fsp_url(u: Optional[str]) -> bool:
    if not u or not isinstance(u, str):
        return False
    try:
        host = urlparse(u).hostname or ""
        return host.endswith("floridastateparks.org")
    except Exception:
        return False


def _norm(s: Optional[str]) -> str:
    return (s or "").strip()


def _guess_region_from_text(park: ParkItem) -> Optional[str]:
    """Best-effort heuristic to guess region from provided name/address text if region_claim is missing."""
    txt = f"{park.official_name or ''} | {park.physical_address or ''}".lower()

    keys_terms = [
        "florida keys", "key west", "key largo", "islamorada", "marathon", "big pine key",
        "monroe county", "bahia honda", "molasses", "duval st"
    ]
    atlantic_terms = [
        "atlantic", "jacksonville", "fernandina", "amelia", "st. augustine", "st augustine",
        "daytona", "new smyrna", "cape canaveral", "cocoa", "melbourne", "vero beach",
        "fort pierce", "stuart", "jupiter", "palm beach", "delray", "boca raton",
        "hallandale", "fort lauderdale", "hollywood", "miami", "miami beach", "key biscayne",
        "broward county", "palm beach county", "miami-dade"
    ]
    gulf_terms = [
        "gulf", "panama city", "pensacola", "destin", "apalachicola", "sarasota", "bradenton",
        "clearwater", "st. petersburg", "st petersburg", "tampa", "dunedin", "honeymoon island",
        "fort de soto", "sanibel", "captiva", "fort myers", "naples", "mexico beach",
        "port st. joe", "port st joe", "st george island", "cedar key", "venice", "englewood"
    ]
    south_florida_terms = [
        "miami", "miami-dade", "broward", "palm beach", "fort lauderdale", "key biscayne", "delray", "boca raton"
    ]

    if any(k in txt for k in keys_terms):
        return "keys_or_south_fl"
    if any(a in txt for a in atlantic_terms):
        # Atlantic Coast (not necessarily South FL)
        # If it’s clearly in South Florida Atlantic counties, we can map to keys_or_south_fl later
        if any(s in txt for s in south_florida_terms):
            return "keys_or_south_fl"
        return "atlantic"
    if any(g in txt for g in gulf_terms):
        return "gulf"
    return None


def _normalize_region_claim(claim: Optional[str]) -> Optional[str]:
    if not claim:
        return None
    c = claim.strip().lower()
    if c in ["gulf", "gulf coast", "gulf_of_mexico", "gulfcoast"]:
        return "gulf"
    if c in ["atlantic", "atlantic coast", "atlanticcoast"]:
        return "atlantic"
    if c in ["keys", "florida keys", "south_florida", "south florida", "s. florida", "s florida"]:
        return "keys_or_south_fl"
    if c in ["keys_or_south_fl", "keys/south_florida", "keys/south_fl"]:
        return "keys_or_south_fl"
    return None


def _assign_parks_to_slots(all_parks: List[ParkItem]) -> Dict[str, Optional[ParkItem]]:
    """
    Assign parks to required region slots:
      - gulf
      - atlantic
      - keys_or_south_fl
    Try to maximize fit using region_claim first, then heuristics.
    """
    candidates = []
    for p in all_parks:
        # Normalize region_claim or guess
        region = _normalize_region_claim(p.region_claim)
        if not region:
            region = _guess_region_from_text(p)
        candidates.append((p, region))

    used: set = set()
    assignment: Dict[str, Optional[ParkItem]] = {
        "gulf": None,
        "atlantic": None,
        "keys_or_south_fl": None
    }

    # Helper to pick a park for a region
    def pick_for(region_key: str):
        for idx, (p, r) in enumerate(candidates):
            if idx in used:
                continue
            if r == region_key:
                used.add(idx)
                assignment[region_key] = p
                return True
        return False

    # First pass: exact matches
    for region_key in ["gulf", "atlantic", "keys_or_south_fl"]:
        pick_for(region_key)

    # Second pass: fill remaining by any leftover parks (best effort)
    for region_key in ["gulf", "atlantic", "keys_or_south_fl"]:
        if assignment[region_key] is None:
            for idx, (p, r) in enumerate(candidates):
                if idx in used:
                    continue
                used.add(idx)
                assignment[region_key] = p
                break

    return assignment


def _unique_nonempty_names(parks: List[Optional[ParkItem]]) -> bool:
    names = [_norm(p.official_name) for p in parks if p is not None]
    if any(n == "" for n in names):
        return False
    return len(set(names)) == len(names)


# --------------------------------------------------------------------------- #
# Verification builders                                                       #
# --------------------------------------------------------------------------- #
async def verify_park_slot(
    evaluator: Evaluator,
    parent_node,
    park: Optional[ParkItem],
    slot_node_id: str,
    slot_desc: str,
    region_target: str
) -> None:
    """
    Build verification subtree for a single park slot.
    region_target in {"gulf", "atlantic", "keys_or_south_fl"}.
    """
    # Create the park-level parallel node (non-critical to allow partial credit across slots)
    slot_node = evaluator.add_parallel(
        id=slot_node_id,
        desc=f"{slot_desc}: meets all constraints and includes all required fields.",
        parent=parent_node,
        critical=False
    )

    # Safe accessors
    name = _norm(park.official_name) if park else ""
    addr = _norm(park.physical_address) if park else ""
    url = _norm(park.fsp_url) if park else ""
    camping_types = park.camping_types if (park and park.camping_types) else []
    camping_types_str = ", ".join(camping_types) if camping_types else ""
    # Existence checks (as per rubric: required fields provided)
    n1 = evaluator.add_custom_node(
        result=(name != ""),
        id=f"{slot_node_id.replace('Park_', 'P')}_Official_Name",
        desc="Official park name is provided.",
        parent=slot_node,
        critical=True
    )
    n2 = evaluator.add_custom_node(
        result=(addr != ""),
        id=f"{slot_node_id.replace('Park_', 'P')}_Physical_Address",
        desc="Physical address is provided.",
        parent=slot_node,
        critical=True
    )

    # Official FSP URL existence and domain check (gatekeeper)
    url_provided_node = evaluator.add_custom_node(
        result=(_is_fsp_url(url)),
        id=f"{slot_node_id.replace('Park_', 'P')}_Official_FSP_URL_Provided",
        desc="A reference URL on floridastateparks.org for this specific park is provided.",
        parent=slot_node,
        critical=True
    )

    # Verify the URL actually corresponds to the named park page
    url_match_node = evaluator.add_leaf(
        id=f"{slot_node_id.replace('Park_', 'P')}_Official_FSP_URL_Matches_Park",
        desc="The provided FSP URL corresponds to the official page for the named park.",
        parent=slot_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"This Florida State Parks webpage is the official park page for '{name}' (allowing minor naming variants like 'State Park', 'Beach State Park', or historical designations).",
        node=url_match_node,
        sources=url if url else None,
        additional_instruction="Confirm the page is clearly a Florida State Park unit page for the specified park name (title/header should strongly indicate the park). Minor name variants and punctuation differences are acceptable."
    )

    # Confirm it's a Florida State Park (managed under the system)
    managed_node = evaluator.add_leaf(
        id=f"{slot_node_id.replace('Park_', 'P')}_Is_Florida_State_Park",
        desc="Park is confirmed to be managed by the Florida State Parks system.",
        parent=slot_node,
        critical=True
    )
    await evaluator.verify(
        claim="This page indicates that the site is a Florida State Parks unit managed by the Florida Department of Environmental Protection.",
        node=managed_node,
        sources=url if url else None,
        additional_instruction="Look for the official Florida State Parks branding and clear language that the unit is a State Park (e.g., 'Florida State Parks', 'FDEP')."
    )

    # Region check per slot
    region_map_text = {
        "gulf": "on or along Florida's Gulf Coast (i.e., Gulf of Mexico coastline in Florida, including Panhandle or west coast).",
        "atlantic": "on or along Florida's Atlantic Coast (i.e., Atlantic Ocean coastline in Florida).",
        "keys_or_south_fl": "in the Florida Keys OR within South Florida's Atlantic coastal counties (e.g., Miami-Dade, Broward, or Palm Beach)."
    }
    region_node = evaluator.add_leaf(
        id=f"{slot_node_id.replace('Park_', 'P')}_Region_Check",
        desc=f"Park is in required region for this slot ({region_target}).",
        parent=slot_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"This park is {region_map_text.get(region_target, 'in the required region')} ",
        node=region_node,
        sources=url if url else None,
        additional_instruction="Base your decision on explicit mentions such as 'Gulf of Mexico', 'Atlantic Ocean', 'Florida Keys', Monroe County, or the city/county location. Minor inference from address/city/county is allowed."
    )

    # Car accessibility (no ferry/boat-only requirement)
    car_node = evaluator.add_leaf(
        id=f"{slot_node_id.replace('Park_', 'P')}_Car_Accessible",
        desc="Park is accessible by car (no ferry-only or boat-only requirement).",
        parent=slot_node,
        critical=True
    )
    await evaluator.verify(
        claim="Visitors can reach the park by car via public roads; general entry does not require a ferry or boat.",
        node=car_node,
        sources=url if url else None,
        additional_instruction="If the page states 'accessible only by ferry or boat', mark as false. If the page provides driving directions, parking info, or a street address for arrival by car, mark as true. Optional boat tours do not negate car accessibility."
    )

    # Camping availability and type
    camp_node = evaluator.add_leaf(
        id=f"{slot_node_id.replace('Park_', 'P')}_Camping_Available_And_Type",
        desc="Overnight camping is confirmed and the camping type(s) are specified.",
        parent=slot_node,
        critical=True
    )
    type_hint = camping_types_str if camping_types_str else "at least one of: tent sites, RV sites, or cabins"
    await evaluator.verify(
        claim=f"The park offers official overnight camping; specifically, it provides {type_hint}.",
        node=camp_node,
        sources=url if url else None,
        additional_instruction="Pass if the official page confirms any overnight camping (campground or cabins). If specific types are listed, ensure at least one of the types in the claim appears on the page (allow synonyms like 'campground' for tent/RV, 'vacation cabins' for cabins)."
    )

    # Direct beach access for swimming
    beach_node = evaluator.add_leaf(
        id=f"{slot_node_id.replace('Park_', 'P')}_Direct_Beach_Swimming_Access",
        desc="Direct beach access for swimming is confirmed.",
        parent=slot_node,
        critical=True
    )
    await evaluator.verify(
        claim="The park has direct beach access where swimming is permitted (ocean or gulf).",
        node=beach_node,
        sources=url if url else None,
        additional_instruction="Look for amenities/activities like 'Beach' and 'Swimming'. If swimming is explicitly prohibited, mark as false."
    )

    # Hiking trails open to general visitors
    hike_node = evaluator.add_leaf(
        id=f"{slot_node_id.replace('Park_', 'P')}_Hiking_Trails_General_Visitors",
        desc="Designated hiking trails are confirmed and open to general visitors.",
        parent=slot_node,
        critical=True
    )
    await evaluator.verify(
        claim="The park offers designated hiking or nature trails that are open to general day-use visitors (no special permit required for normal day hiking).",
        node=hike_node,
        sources=url if url else None,
        additional_instruction="Accept boardwalks and multi-use nature trails. If trails exist only for restricted activities or by special permit, mark as false."
    )

    # Basic visitor amenities
    amenities_node = evaluator.add_leaf(
        id=f"{slot_node_id.replace('Park_', 'P')}_Basic_Visitor_Amenities",
        desc="Presence of basic visitor facilities/amenities is confirmed.",
        parent=slot_node,
        critical=True
    )
    await evaluator.verify(
        claim="The official page indicates typical visitor amenities such as restrooms, picnic areas/shelters, or parking.",
        node=amenities_node,
        sources=url if url else None,
        additional_instruction="Any one clear amenity (restrooms, picnic area/shelters, parking, bathhouse) suffices for a pass."
    )

    # Brief trail info if available (non-critical; if not provided, pass by design)
    if park and _norm(park.brief_trail_info):
        trail_info_leaf = evaluator.add_leaf(
            id=f"{slot_node_id.replace('Park_', 'P')}_Brief_Trail_Info_If_Available",
            desc="Brief trail information (if provided) is accurate.",
            parent=slot_node,
            critical=False
        )
        await evaluator.verify(
            claim=f"Trail info is accurate for this park: '{_norm(park.brief_trail_info)}'.",
            node=trail_info_leaf,
            sources=url if url else None,
            additional_instruction="Allow minor differences (e.g., rounding lengths). Verify the referenced trail name/feature exists at this park."
        )
    else:
        evaluator.add_custom_node(
            result=True,
            id=f"{slot_node_id.replace('Park_', 'P')}_Brief_Trail_Info_If_Available",
            desc="Brief trail info not provided (acceptable if not available on the official page).",
            parent=slot_node,
            critical=False
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
    model: str = "o4-mini",
) -> Dict:
    """
    Evaluate an answer for the Florida State Parks regional camping/beach/hiking task.
    """
    # Initialize evaluator (root is non-critical parallel aggregator)
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

    # 1) Extract park items from the answer
    extracted: ParksExtraction = await evaluator.extract(
        prompt=prompt_extract_parks(),
        template_class=ParksExtraction,
        extraction_name="parks_extraction",
    )

    # 2) Keep all extracted for bookkeeping; assignment will pick 3 for slots
    all_parks: List[ParkItem] = extracted.parks if extracted and extracted.parks else []

    # 3) Assignment to required region slots
    assignment = _assign_parks_to_slots(all_parks)
    p_gulf = assignment.get("gulf")
    p_atl = assignment.get("atlantic")
    p_keys = assignment.get("keys_or_south_fl")

    # Record assignment and counts for transparency
    evaluator.add_custom_info(
        info={
            "total_parks_extracted": len(all_parks),
            "assigned": {
                "gulf": p_gulf.official_name if p_gulf else None,
                "atlantic": p_atl.official_name if p_atl else None,
                "keys_or_south_fl": p_keys.official_name if p_keys else None,
            },
        },
        info_type="assignment",
        info_name="park_assignment"
    )

    # 4) Top-level general checks (treated as non-critical to allow partial scoring even if formatting differs)
    exactly_three = evaluator.add_custom_node(
        result=(len([p for p in all_parks if _norm(p.official_name)]) == 3),
        id="Exactly_Three_Parks_Provided",
        desc="Response provides exactly three parks (one per required region slot).",
        parent=root,
        critical=False
    )

    distinct_three = evaluator.add_custom_node(
        result=_unique_nonempty_names([p_gulf, p_atl, p_keys]),
        id="Distinct_Parks_Check",
        desc="All three parks are distinct (no park is reused across region slots).",
        parent=root,
        critical=False
    )

    # 5) Per-park verifications by slot
    await verify_park_slot(
        evaluator=evaluator,
        parent_node=root,
        park=p_gulf,
        slot_node_id="Park_1_Gulf_Coast",
        slot_desc="Park 1 (Gulf Coast)",
        region_target="gulf"
    )

    await verify_park_slot(
        evaluator=evaluator,
        parent_node=root,
        park=p_atl,
        slot_node_id="Park_2_Atlantic_Coast",
        slot_desc="Park 2 (Atlantic Coast)",
        region_target="atlantic"
    )

    await verify_park_slot(
        evaluator=evaluator,
        parent_node=root,
        park=p_keys,
        slot_node_id="Park_3_Keys_or_South_Florida",
        slot_desc="Park 3 (Florida Keys or South Florida coastal)",
        region_target="keys_or_south_fl"
    )

    # 6) Return structured summary
    return evaluator.get_summary()