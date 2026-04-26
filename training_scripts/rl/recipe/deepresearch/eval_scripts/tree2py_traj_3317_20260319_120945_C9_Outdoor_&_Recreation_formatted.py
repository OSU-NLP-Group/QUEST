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
TASK_ID = "wilderness_trailheads_ap_4"
TASK_DESCRIPTION = """
Identify 4 wilderness trailheads in Southern California that meet ALL of the following requirements for a wilderness education program:

1. Each trailhead must be located in one of the four Adventure Pass required national forests: Angeles National Forest, Cleveland National Forest, Los Padres National Forest, or San Bernardino National Forest.

2. Each trailhead must provide access to a designated wilderness area that requires a free wilderness permit for overnight use.

3. Each trailhead must accommodate groups of up to 12 people (the maximum wilderness group size limit).

4. Each trailhead must have the following facilities that qualify it as an Adventure Pass designated site:
   - Designated developed parking area
   - Permanent toilet facility
   - Permanent trash receptacle
   - Interpretive sign, exhibit, or kiosk
   - Picnic tables

5. From each trailhead, there must be trail access rated as Moderate or Difficult difficulty level (typically 3+ miles with elevation gain), suitable for multi-day wilderness backpacking.

6. Each trailhead must provide access to designated backcountry campsites or camping zones within the wilderness area for overnight stays.

For each of the 4 trailheads, provide:
- The specific trailhead name
- The wilderness area it accesses
- The national forest where it is located
- Confirmation of wilderness permit requirements
- Description of available facilities
- Trail characteristics (difficulty level, suitability for multi-day trips)
- URL reference(s) supporting each claim
"""

ALLOWED_FORESTS = [
    "Angeles National Forest",
    "Cleveland National Forest",
    "Los Padres National Forest",
    "San Bernardino National Forest",
]


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class TrailFacilities(BaseModel):
    parking: Optional[str] = None  # yes/no/unknown
    toilet: Optional[str] = None   # yes/no/unknown (permanent restroom, vault toilet)
    trash: Optional[str] = None    # yes/no/unknown (trash cans/receptacles)
    interpretive: Optional[str] = None  # yes/no/unknown (kiosk, info board)
    picnic: Optional[str] = None        # yes/no/unknown (picnic tables)


class TrailheadItem(BaseModel):
    trailhead_name: Optional[str] = None

    # Location / Forest
    national_forest: Optional[str] = None
    forest_sources: List[str] = Field(default_factory=list)

    # Wilderness access
    wilderness_area: Optional[str] = None
    wilderness_sources: List[str] = Field(default_factory=list)

    # Permits and group size
    permit_required_overnight: Optional[str] = None  # yes/no/unknown
    permit_free: Optional[str] = None                # yes/no/unknown (free/no fee)
    group_size_limit: Optional[str] = None           # e.g., "12"
    permit_sources: List[str] = Field(default_factory=list)

    # Facilities
    facilities: Optional[TrailFacilities] = None
    facilities_sources: List[str] = Field(default_factory=list)

    # Trail characteristics
    trail_difficulty: Optional[str] = None  # e.g., "Moderate", "Difficult", "Strenuous", or a phrase
    trail_length_miles: Optional[str] = None  # free text (e.g., "3.5 miles", "5–8 miles")
    trail_elevation_gain: Optional[str] = None  # free text (e.g., "1200 ft")
    multi_day_suitable: Optional[str] = None  # yes/no/unknown
    trail_sources: List[str] = Field(default_factory=list)

    # Camping access
    camping_access: Optional[str] = None  # yes/no/unknown; "designated campsites/zones available"
    camping_sources: List[str] = Field(default_factory=list)


class TrailheadsExtraction(BaseModel):
    trailheads: List[TrailheadItem] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_trailheads() -> str:
    return """
    Extract up to 4 distinct wilderness trailheads mentioned in the answer that meet the program's requirements.
    For each trailhead, return a JSON object with the following fields:

    1) trailhead_name: the specific trailhead name as written in the answer.
    2) national_forest: one of ["Angeles National Forest","Cleveland National Forest","Los Padres National Forest","San Bernardino National Forest"] if explicitly claimed; otherwise, return the exact name given in the answer (or null if missing).
    3) forest_sources: an array of URLs explicitly cited in the answer that confirm the trailhead's location within the specified national forest.

    4) wilderness_area: the designated wilderness area that the trailhead provides access to (as stated in the answer), exact string from the answer (or null if missing).
    5) wilderness_sources: URLs cited in the answer that confirm the wilderness access/designation.

    6) permit_required_overnight: "yes"/"no"/"unknown" based on the answer's claims for overnight wilderness permit requirement.
    7) permit_free: "yes"/"no"/"unknown" indicating whether the permit is free/no-fee as claimed.
    8) group_size_limit: the maximum wilderness group size value (e.g., "12") as claimed in the answer; if missing, null.
    9) permit_sources: URLs cited for permit requirements and group size limits.

    10) facilities: an object with five fields, each "yes"/"no"/"unknown" as claimed in the answer:
        - parking: designated developed parking area
        - toilet: permanent toilet facility (e.g., vault toilet, restroom)
        - trash: permanent trash receptacle (e.g., trash cans)
        - interpretive: interpretive sign, exhibit, or kiosk
        - picnic: picnic tables
    11) facilities_sources: URLs cited confirming the above facility amenities.

    12) trail_difficulty: the stated difficulty level if provided (e.g., "Moderate", "Difficult", "Strenuous"), otherwise any phrasing describing difficulty; null if missing.
    13) trail_length_miles: any mileage statement in the answer (free text), null if missing.
    14) trail_elevation_gain: any elevation gain statement in the answer (free text), null if missing.
    15) multi_day_suitable: "yes"/"no"/"unknown" whether the answer claims the trail(s) from this trailhead support multi-day backpacking.
    16) trail_sources: URLs cited describing trail characteristics/difficulty/length/elevation and multi-day suitability.

    17) camping_access: "yes"/"no"/"unknown" whether the answer claims designated backcountry campsites or camping zones are accessible from this trailhead within the wilderness.
    18) camping_sources: URLs cited confirming designated backcountry campsites or camping zones availability.

    RULES:
    - Extract only what is explicitly present in the answer. Do not invent or infer.
    - For each *sources* field, include all URLs (including markdown links) explicitly associated with that specific aspect in the answer.
    - If a URL lacks a protocol in the answer, prepend http://
    - If a value is not claimed in the answer, set it to null (or "unknown" for the boolean-like fields requested).
    - Return at most 4 trailheads in the 'trailheads' array, in the same order as they appear in the answer.
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _is_allowed_forest(forest: Optional[str]) -> bool:
    if not forest:
        return False
    normalized = forest.strip().lower()
    return any(normalized == af.lower() for af in ALLOWED_FORESTS)


def _boolish_yes(val: Optional[str]) -> Optional[bool]:
    if val is None:
        return None
    s = val.strip().lower()
    if s in {"yes", "y", "true", "present", "exists"}:
        return True
    if s in {"no", "n", "false", "absent", "not present"}:
        return False
    return None


def _safe_list(urls: Optional[List[str]]) -> List[str]:
    return urls or []


# --------------------------------------------------------------------------- #
# Verification for a single trailhead                                         #
# --------------------------------------------------------------------------- #
async def verify_single_trailhead(
    evaluator: Evaluator,
    parent_node,
    th: TrailheadItem,
    index: int
) -> None:
    tid = index + 1
    th_name = th.trailhead_name or f"Trailhead #{tid}"
    forest = th.national_forest or "Unknown Forest"
    wilderness = th.wilderness_area or "Unknown Wilderness"

    # Trailhead node (parallel, non-critical to allow partial credit per trailhead)
    th_node = evaluator.add_parallel(
        id=f"Trailhead_{tid}",
        desc=f"{['First','Second','Third','Fourth'][index]} qualifying trailhead meeting all requirements",
        parent=parent_node,
        critical=False
    )

    # ---------------- Forest Location ----------------
    forest_node = evaluator.add_parallel(
        id=f"T{tid}_Forest_Location",
        desc="Trailhead is located in one of the four Adventure Pass forests",
        parent=th_node,
        critical=True
    )

    # Identification: ensure the named forest is one of ALLOWED_FORESTS
    evaluator.add_custom_node(
        result=_is_allowed_forest(th.national_forest),
        id=f"T{tid}_Forest_Identification",
        desc="Identify which Adventure Pass forest (Angeles, Cleveland, Los Padres, or San Bernardino) contains the trailhead",
        parent=forest_node,
        critical=True
    )

    # Reference: verify with URLs
    t_forest_ref = evaluator.add_leaf(
        id=f"T{tid}_Forest_Reference",
        desc="Provide URL reference confirming the trailhead's location in the specified Adventure Pass forest",
        parent=forest_node,
        critical=True
    )
    forest_claim = f"The trailhead '{th_name}' is located within the {forest}."
    await evaluator.verify(
        claim=forest_claim,
        node=t_forest_ref,
        sources=_safe_list(th.forest_sources),
        additional_instruction="Confirm the page indicates the trailhead is inside or managed by the specified National Forest. Allow synonyms like 'in', 'within', or listing under that forest's sites."
    )

    # ---------------- Wilderness Access ----------------
    wilderness_node = evaluator.add_parallel(
        id=f"T{tid}_Wilderness_Access",
        desc="Trailhead provides access to a designated wilderness area",
        parent=th_node,
        critical=True
    )

    evaluator.add_custom_node(
        result=bool(th.wilderness_area and th.wilderness_area.strip()),
        id=f"T{tid}_Wilderness_Name",
        desc="Identify the specific wilderness area accessed from this trailhead",
        parent=wilderness_node,
        critical=True
    )

    t_wild_ref = evaluator.add_leaf(
        id=f"T{tid}_Wilderness_Reference",
        desc="Provide URL reference confirming the wilderness area designation",
        parent=wilderness_node,
        critical=True
    )
    wilderness_claim = f"The trailhead '{th_name}' provides access to the '{wilderness}' Wilderness (a designated wilderness area)."
    await evaluator.verify(
        claim=wilderness_claim,
        node=t_wild_ref,
        sources=_safe_list(th.wilderness_sources),
        additional_instruction="Verify that the cited page explicitly mentions the wilderness name and that it is a designated wilderness; look for 'Wilderness' designation language."
    )

    # ---------------- Permit System ----------------
    permit_node = evaluator.add_parallel(
        id=f"T{tid}_Permit_System",
        desc="Wilderness permit system requirements",
        parent=th_node,
        critical=True
    )

    t_permit_req = evaluator.add_leaf(
        id=f"T{tid}_Permit_Required",
        desc="Confirm that free wilderness permit is required for overnight use",
        parent=permit_node,
        critical=True
    )
    permit_req_claim = f"A free (no-fee) wilderness permit is required for overnight use in the '{wilderness}' Wilderness or for overnight trips starting at '{th_name}'."
    await evaluator.verify(
        claim=permit_req_claim,
        node=t_permit_req,
        sources=_safe_list(th.permit_sources),
        additional_instruction="The page should clearly indicate that an overnight wilderness permit is required and that the permit is free or no-fee (e.g., self-issue). If fee-based, mark as not supported."
    )

    t_group_limit = evaluator.add_leaf(
        id=f"T{tid}_Group_Size_Limit",
        desc="Confirm that groups of up to 12 people are permitted",
        parent=permit_node,
        critical=True
    )
    group_limit_value = th.group_size_limit.strip() if th.group_size_limit else "12"
    group_claim = f"The maximum wilderness group size limit is {group_limit_value} people for trips in the '{wilderness}' Wilderness (or per the applicable forest wilderness rules)."
    await evaluator.verify(
        claim=group_claim,
        node=t_group_limit,
        sources=_safe_list(th.permit_sources),
        additional_instruction="Verify that the official wilderness/forest source indicates a wilderness group size limit of 12 (or the value claimed in the answer). Allow 'up to 12' wording."
    )

    evaluator.add_custom_node(
        result=len(_safe_list(th.permit_sources)) > 0,
        id=f"T{tid}_Permit_Reference",
        desc="Provide URL reference for permit requirements and group size limits",
        parent=permit_node,
        critical=True
    )

    # ---------------- Facilities ----------------
    fac_node = evaluator.add_parallel(
        id=f"T{tid}_Facilities",
        desc="Required Adventure Pass facilities present at trailhead",
        parent=th_node,
        critical=True
    )

    # Helper to build facility verification
    def facility_leaf(fid: str, desc: str, claim_text: str):
        node = evaluator.add_leaf(
            id=f"T{tid}_{fid}",
            desc=desc,
            parent=fac_node,
            critical=True
        )
        return node, claim_text

    # Build claims for each required facility
    parking_node, parking_claim = facility_leaf(
        "Parking",
        "Designated developed parking area exists",
        f"There is a designated developed parking area at the '{th_name}' trailhead."
    )
    toilet_node, toilet_claim = facility_leaf(
        "Toilet",
        "Permanent toilet facility exists",
        f"A permanent toilet facility (e.g., vault toilet/restroom) is present at the '{th_name}' trailhead."
    )
    trash_node, trash_claim = facility_leaf(
        "Trash",
        "Permanent trash receptacle exists",
        f"Permanent trash receptacles (e.g., trash cans) are available at the '{th_name}' trailhead."
    )
    interp_node, interp_claim = facility_leaf(
        "Interpretive",
        "Interpretive sign, exhibit, or kiosk exists",
        f"An interpretive sign, exhibit, or informational kiosk is present at the '{th_name}' trailhead."
    )
    picnic_node, picnic_claim = facility_leaf(
        "Picnic",
        "Picnic tables exist",
        f"Picnic tables are available at the '{th_name}' trailhead."
    )

    # Batch verify facilities with the same sources list
    await evaluator.batch_verify([
        (parking_claim, _safe_list(th.facilities_sources), parking_node, "Accept synonyms: 'parking lot', 'designated parking area', 'developed parking'."),
        (toilet_claim, _safe_list(th.facilities_sources), toilet_node, "Accept synonyms: 'restroom', 'toilet', 'vault toilet'. Should be a permanent facility, not temporary."),
        (trash_claim, _safe_list(th.facilities_sources), trash_node, "Accept synonyms: 'trash can', 'trash receptacle', 'garbage can'. Should be permanent/installed."),
        (interp_claim, _safe_list(th.facilities_sources), interp_node, "Accept synonyms: 'information kiosk', 'interpretive display', 'interpretive sign', 'info board'."),
        (picnic_claim, _safe_list(th.facilities_sources), picnic_node, "Accept synonyms: 'picnic table(s)', 'picnic area'."),
    ])

    evaluator.add_custom_node(
        result=len(_safe_list(th.facilities_sources)) > 0,
        id=f"T{tid}_Facilities_Reference",
        desc="Provide URL reference confirming facility amenities",
        parent=fac_node,
        critical=True
    )

    # ---------------- Trail Characteristics ----------------
    trail_char_node = evaluator.add_parallel(
        id=f"T{tid}_Trail_Characteristics",
        desc="Trail meets difficulty and suitability requirements",
        parent=th_node,
        critical=True
    )

    t_difficulty = evaluator.add_leaf(
        id=f"T{tid}_Difficulty_Level",
        desc="Trail is rated as Moderate or Difficult (3+ miles with elevation gain)",
        parent=trail_char_node,
        critical=True
    )
    # Build difficulty claim using extracted hints if present
    diff_phrase = th.trail_difficulty or "Moderate or Difficult"
    miles_hint = th.trail_length_miles or "3+ miles"
    elev_hint = th.trail_elevation_gain or "appreciable elevation gain"
    diff_claim = f"From '{th_name}', there is trail access that is {diff_phrase} or otherwise roughly {miles_hint} with {elev_hint}, consistent with a Moderate/Difficult backcountry route."
    await evaluator.verify(
        claim=diff_claim,
        node=t_difficulty,
        sources=_safe_list(th.trail_sources),
        additional_instruction="Look for explicit difficulty ratings (Moderate/Difficult/Strenuous) OR route metrics indicating ~3+ miles and real elevation gain. Accept synonyms like 'strenuous' or 'challenging'."
    )

    t_multi = evaluator.add_leaf(
        id=f"T{tid}_Multi_Day_Suitable",
        desc="Trail is suitable for multi-day wilderness backpacking",
        parent=trail_char_node,
        critical=True
    )
    multi_claim = f"Routes from the '{th_name}' trailhead are suitable for multi-day wilderness backpacking (e.g., overnight itineraries, loops, or backcountry routes)."
    await evaluator.verify(
        claim=multi_claim,
        node=t_multi,
        sources=_safe_list(th.trail_sources),
        additional_instruction="Confirm mentions of overnights, backpacking trips, multi-day routes, or itineraries starting from this trailhead."
    )

    evaluator.add_custom_node(
        result=len(_safe_list(th.trail_sources)) > 0,
        id=f"T{tid}_Trail_Reference",
        desc="Provide URL reference for trail characteristics and difficulty rating",
        parent=trail_char_node,
        critical=True
    )

    # ---------------- Camping Access ----------------
    camping_node = evaluator.add_parallel(
        id=f"T{tid}_Camping_Access",
        desc="Access to designated backcountry campsites or camping zones",
        parent=th_node,
        critical=True
    )

    t_campsites = evaluator.add_leaf(
        id=f"T{tid}_Campsites_Available",
        desc="Designated backcountry campsites or camping zones are accessible from this trailhead",
        parent=camping_node,
        critical=True
    )
    camping_claim = f"Designated backcountry campsites or camping zones in the '{wilderness}' Wilderness are accessible via the '{th_name}' trailhead for overnight stays."
    await evaluator.verify(
        claim=camping_claim,
        node=t_campsites,
        sources=_safe_list(th.camping_sources),
        additional_instruction="Look for explicit mentions of designated backcountry campsites, camping zones, quota/zone systems, or named backcountry camps within the wilderness accessed by this trailhead."
    )

    evaluator.add_custom_node(
        result=len(_safe_list(th.camping_sources)) > 0,
        id=f"T{tid}_Camping_Reference",
        desc="Provide URL reference confirming backcountry camping availability",
        parent=camping_node,
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
    Evaluate an answer for the 4 Adventure Pass forest wilderness trailheads task.
    """
    evaluator = Evaluator()
    # Note: The provided JSON marks Root as critical. However, in the framework,
    # a critical parent requires all children to be critical. To allow partial
    # credit across trailheads, we initialize the root as non-critical.
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

    # Record reference info
    evaluator.add_custom_info({"allowed_forests": ALLOWED_FORESTS}, info_type="config", info_name="allowed_forests")

    # Extraction
    extracted = await evaluator.extract(
        prompt=prompt_extract_trailheads(),
        template_class=TrailheadsExtraction,
        extraction_name="trailheads_extraction"
    )

    # Normalize and pad/trim to exactly 4 items
    items: List[TrailheadItem] = list(extracted.trailheads or [])
    if len(items) > 4:
        items = items[:4]
    while len(items) < 4:
        items.append(TrailheadItem())

    # Build per-trailhead subtrees
    tasks = []
    for idx, th in enumerate(items):
        tasks.append(verify_single_trailhead(evaluator, root, th, idx))
    # Run verifications sequentially to respect shared network resources (can be parallelized if desired)
    for t in tasks:
        await t

    return evaluator.get_summary()