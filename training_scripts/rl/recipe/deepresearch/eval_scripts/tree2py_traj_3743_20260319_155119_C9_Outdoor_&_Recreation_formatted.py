import asyncio
import logging
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "multi_state_rv_trip_planning"
TASK_DESCRIPTION = (
    "You are planning a summer RV road trip across the United States and need to identify four state parks, each in a different U.S. state, "
    "that meet the specified requirements: full-hookup RV sites (water, 50A electric, sewer), ADA-accessible camping facilities, "
    "accommodate ≥40 ft RVs, correct reservation system, proximity to a major national park or outdoor recreation area, and state diversity. "
    "For each park: provide park name/state, confirm full hookups, describe ADA features, max RV length (≥40 ft), reservation system/platform, "
    "nearby attraction, and supporting URLs."
)


# --------------------------------------------------------------------------- #
# Data models for structured extraction                                       #
# --------------------------------------------------------------------------- #
class ParkItem(BaseModel):
    park_name: Optional[str] = None
    state: Optional[str] = None

    # URLs supporting each requirement
    identity_urls: List[str] = Field(default_factory=list)         # Official/state page confirming it's a state park in the given state
    full_hookup_urls: List[str] = Field(default_factory=list)      # Evidence of water + 50A electric + sewer at sites
    ada_urls: List[str] = Field(default_factory=list)              # Evidence of ADA accessible camping restrooms/paths
    rv_length_urls: List[str] = Field(default_factory=list)        # Evidence that ≥40 ft RVs are accommodated
    reservation_system: Optional[str] = None                       # Name of reservation system/platform
    reservation_urls: List[str] = Field(default_factory=list)      # Booking page or official reservation information
    nearby_attraction: Optional[str] = None                        # Major national park/outdoor area
    attraction_urls: List[str] = Field(default_factory=list)       # Evidence for proximity (distance/time or official “gateway/base” text)
    distance_or_time: Optional[str] = None                         # Distance or drive time stated in the answer (if any)
    rv_max_length: Optional[str] = None                            # Max RV length mentioned in the answer (string form)


class ParksExtraction(BaseModel):
    parks: List[ParkItem] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt helpers                                                   #
# --------------------------------------------------------------------------- #
def prompt_extract_parks() -> str:
    return """
    Extract every state park listed in the answer (do not limit to four if the answer lists more). Return an array 'parks' with one object per park.
    For each park, extract the following fields strictly from the answer text:
    - park_name: The park's name as written in the answer (do not invent).
    - state: The U.S. state name or abbreviation as written.
    - identity_urls: URLs (array) cited that demonstrate the unit is a state park in that state (e.g., official state parks website).
    - full_hookup_urls: URLs (array) that support availability of full RV hookups: water + 50-amp electric + sewer at campsites.
    - ada_urls: URLs (array) that support ADA-accessible camping facilities, including BOTH: (1) paved/wheelchair-suitable pathways and (2) accessible restrooms with grab bars and accessible stalls (or direct equivalent phrasing).
    - rv_length_urls: URLs (array) that support RVs of at least 40 feet are accommodated (e.g., “max length 40 ft+”, “big rigs up to 45’”).
    - reservation_system: The reservation platform or system named in the answer (e.g., state reservation site, ReserveCalifornia, Recreation.gov). Use null if missing.
    - reservation_urls: URLs (array) for the booking page or official reservation information.
    - nearby_attraction: The major national park or outdoor recreation area noted as nearby (name only). Use null if missing.
    - attraction_urls: URLs (array) that support proximity (distance/drive time) or clearly state the park is a gateway/base for that attraction.
    - distance_or_time: Any distance or drive-time wording present in the answer (e.g., "45 minutes", "30 miles"). Use null if missing.
    - rv_max_length: Any stated maximum RV length from the answer (string). Use null if missing.

    General rules:
    1) Extract only what the answer explicitly states; do not infer.
    2) If a requested field is not present, return null (or empty array for URL lists).
    3) Keep URLs exactly as written (support markdown links by extracting the actual URL).
    4) Do not add unrelated URLs.
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _has_urls(urls: Optional[List[str]]) -> bool:
    return bool(urls) and len(urls) > 0 and any(isinstance(u, str) and u.strip() for u in urls)


def _norm(s: Optional[str]) -> Optional[str]:
    return s.strip().lower() if isinstance(s, str) else None


# --------------------------------------------------------------------------- #
# Verification per-park                                                       #
# --------------------------------------------------------------------------- #
async def verify_state_park(
    evaluator: Evaluator,
    parent_node,
    park: ParkItem,
    idx_one_based: int
) -> None:
    """
    Build verification sub-tree for a single park.

    Structure (Parallel):
      - Park_Identity_{i} (+ presence checks)
      - Full_Hookups_{i} (+ presence checks)
      - ADA_Accessibility_{i} (+ presence checks)
      - RV_Length_{i} (+ presence checks)
      - Reservation_System_Identified_{i} (+ presence checks)
      - Reservation_System_Is_State_Or_Federal_{i} (depends on reservation presence)
      - Nearby_Attraction_{i} (+ presence checks)
    """
    park_node = evaluator.add_parallel(
        id=f"State_Park_{idx_one_based}",
        desc=f"Park #{idx_one_based} evaluation (meets required criteria and includes supporting URLs).",
        parent=parent_node,
        critical=False
    )

    # --------------------- Identity ---------------------
    # Presence checks (critical gating)
    evaluator.add_custom_node(
        result=(park.park_name is not None and park.state is not None and park.park_name.strip() != "" and park.state.strip() != ""),
        id=f"Park_Identity_{idx_one_based}_fields_present",
        desc=f"Park #{idx_one_based}: park name and state provided in the answer",
        parent=park_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=_has_urls(park.identity_urls),
        id=f"Park_Identity_{idx_one_based}_urls_present",
        desc=f"Park #{idx_one_based}: at least one supporting URL for identity/state-park status",
        parent=park_node,
        critical=True
    )
    identity_leaf = evaluator.add_leaf(
        id=f"Park_Identity_{idx_one_based}",
        desc="Identifies the park name and U.S. state, and the park is a state park (not a national park/private RV park).",
        parent=park_node,
        critical=True
    )
    identity_claim = (
        f"'{park.park_name}' is a state park located in the U.S. state of '{park.state}'. "
        f"It is not a national park and not a private RV park."
    )
    await evaluator.verify(
        claim=identity_claim,
        node=identity_leaf,
        sources=park.identity_urls,
        additional_instruction=(
            "Use the provided URLs to confirm the unit is a state-managed park in the specified state. "
            "Accept state-park system variants such as 'State Recreation Area', 'State Beach', or 'State Historic Park'. "
            "Reject if the evidence points to a national park, national forest campground, or a private/commercial RV resort."
        )
    )

    # --------------------- Full Hookups ---------------------
    evaluator.add_custom_node(
        result=_has_urls(park.full_hookup_urls),
        id=f"Full_Hookups_{idx_one_based}_urls_present",
        desc=f"Park #{idx_one_based}: at least one supporting URL for full hookups",
        parent=park_node,
        critical=True
    )
    hookups_leaf = evaluator.add_leaf(
        id=f"Full_Hookups_{idx_one_based}",
        desc="Confirms RV sites with full hookups including water, 50-amp electric service, and sewer, with at least one supporting URL.",
        parent=park_node,
        critical=True
    )
    hookups_claim = (
        f"The state park '{park.park_name}' offers RV campsites with full hookups, explicitly including: "
        f"(1) a water connection, (2) 50-amp electric service (30/50 acceptable so long as 50A is available), and (3) a sewer connection at the site."
    )
    await evaluator.verify(
        claim=hookups_claim,
        node=hookups_leaf,
        sources=park.full_hookup_urls,
        additional_instruction=(
            "Confirm all three components are available at RV sites: water, 50-amp electric, and sewer at the campsite. "
            "Do not count dump stations alone as sewer hookups. If the page only lists 'water and electric' without sewer, it does not qualify. "
            "If it lists '30/50 amp' that is acceptable for 50A."
        )
    )

    # --------------------- ADA Accessibility ---------------------
    evaluator.add_custom_node(
        result=_has_urls(park.ada_urls),
        id=f"ADA_Accessibility_{idx_one_based}_urls_present",
        desc=f"Park #{idx_one_based}: at least one supporting URL for ADA-accessible camping facilities",
        parent=park_node,
        critical=True
    )
    ada_leaf = evaluator.add_leaf(
        id=f"ADA_Accessibility_{idx_one_based}",
        desc="Confirms ADA-accessible camping facilities including paved wheelchair-suitable pathways and accessible restrooms with grab bars and accessible stalls, with at least one supporting URL.",
        parent=park_node,
        critical=True
    )
    ada_claim = (
        f"'{park.park_name}' provides ADA-accessible camping facilities that include paved or wheelchair-suitable pathways "
        f"and restrooms with grab bars and accessible stalls."
    )
    await evaluator.verify(
        claim=ada_claim,
        node=ada_leaf,
        sources=park.ada_urls,
        additional_instruction=(
            "Look for explicit mentions of accessible (ADA) campsites or facilities. "
            "Specifically verify BOTH aspects: (1) paved/wheelchair-suitable pathways (or equivalent accessible route) and "
            "(2) accessible restrooms with grab bars and accessible stalls. "
            "If only one aspect is present without the other, consider the claim not fully supported."
        )
    )

    # --------------------- RV Length (≥ 40 ft) ---------------------
    evaluator.add_custom_node(
        result=_has_urls(park.rv_length_urls),
        id=f"RV_Length_{idx_one_based}_urls_present",
        desc=f"Park #{idx_one_based}: at least one supporting URL for maximum RV length",
        parent=park_node,
        critical=True
    )
    rvlen_leaf = evaluator.add_leaf(
        id=f"RV_Length_{idx_one_based}",
        desc="Confirms the park can accommodate RVs with maximum allowed length ≥ 40 feet, with at least one supporting URL.",
        parent=park_node,
        critical=True
    )
    rvlen_claim = (
        f"The state park '{park.park_name}' accommodates RVs with a maximum allowable length of at least 40 feet "
        f"(e.g., 40', 45', or similar)."
    )
    await evaluator.verify(
        claim=rvlen_claim,
        node=rvlen_leaf,
        sources=park.rv_length_urls,
        additional_instruction=(
            "Accept phrasings like 'max RV length 40 ft', 'fits big rigs up to 45’', or per-site/loop maximums at or above 40'. "
            "If only smaller limits are shown (e.g., 35'), the claim fails."
        )
    )

    # --------------------- Reservation System ---------------------
    evaluator.add_custom_node(
        result=_has_urls(park.reservation_urls),
        id=f"Reservation_System_Identified_{idx_one_based}_urls_present",
        desc=f"Park #{idx_one_based}: at least one supporting URL for the reservation system",
        parent=park_node,
        critical=True
    )
    reserv_leaf = evaluator.add_leaf(
        id=f"Reservation_System_Identified_{idx_one_based}",
        desc="Identifies the reservation system/platform used to book campsites and provides a supporting URL (booking page or official reservation info).",
        parent=park_node,
        critical=True
    )
    reserv_name = park.reservation_system or "the official reservation website"
    reserv_claim = (
        f"Campsite reservations for '{park.park_name}' are made via {reserv_name} (as shown on the booking/official reservation page)."
    )
    await evaluator.verify(
        claim=reserv_claim,
        node=reserv_leaf,
        sources=park.reservation_urls,
        additional_instruction=(
            "Confirm that the provided reservation page is the correct system to book this state park's campsites "
            "(e.g., the state's official reservation portal, a state-branded vendor page, or Recreation.gov when federal). "
            "Generic travel aggregators without direct booking are not acceptable."
        )
    )

    # Official vs Aggregator check
    reserv_official_leaf = evaluator.add_leaf(
        id=f"Reservation_System_Is_State_Or_Federal_{idx_one_based}",
        desc="Confirms the reservation system is state-specific or federal/official (not merely a third-party aggregator), with supporting URL evidence.",
        parent=park_node,
        critical=True
    )
    reserv_official_claim = (
        f"The reservation platform used for '{park.park_name}' is an official state-managed system or a designated official booking portal "
        f"(such as a state parks website, a state-branded booking vendor linked from the official park page, or Recreation.gov), "
        f"not just a third-party aggregator."
    )
    await evaluator.verify(
        claim=reserv_official_claim,
        node=reserv_official_leaf,
        sources=(park.reservation_urls + park.identity_urls),
        additional_instruction=(
            "Judge officialness using the reservation page and, if needed, the official park page. "
            "Indicators include: .gov or state parks domain, explicit state-branded booking portals (e.g., ReserveCalifornia, ReserveAmerica page "
            "directly linked by the state, ReserveTexas, ReserveOhio, etc.), or Recreation.gov for federal lands. "
            "Reject if the evidence is only a general aggregator (e.g., campsite review/aggregator sites) without official linkage to booking."
        )
    )

    # --------------------- Nearby Major Attraction ---------------------
    evaluator.add_custom_node(
        result=_has_urls(park.attraction_urls),
        id=f"Nearby_Attraction_{idx_one_based}_urls_present",
        desc=f"Park #{idx_one_based}: at least one supporting URL for nearby major attraction and proximity",
        parent=park_node,
        critical=True
    )
    attraction_leaf = evaluator.add_leaf(
        id=f"Nearby_Attraction_{idx_one_based}",
        desc="Names a nearby major national park or outdoor recreation area and provides support that it is within reasonable driving distance (e.g., distance/drive time), with supporting URL(s).",
        parent=park_node,
        critical=True
    )
    attraction_name = park.nearby_attraction or "the referenced attraction"
    distance_phrase = park.distance_or_time or "a typical day-trip driving distance"
    attraction_claim = (
        f"'{park.park_name}' is within reasonable driving distance of '{attraction_name}', suitable as a base or gateway for visiting it "
        f"(approximately {distance_phrase} or comparable)."
    )
    await evaluator.verify(
        claim=attraction_claim,
        node=attraction_leaf,
        sources=park.attraction_urls,
        additional_instruction=(
            "Consider 'reasonable driving distance' as typically ≤ 2 hours or ≤ ~120 miles for a day trip. "
            "Accept explicit mileage/time evidence, official references describing the park as a base/gateway, or map directions pages indicating typical drive."
        )
    )


# --------------------------------------------------------------------------- #
# Main evaluation entry                                                       #
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
    Evaluate an answer for the multi-state RV trip planning task.
    """
    # Initialize evaluator (root should be non-critical to allow partial credit across items)
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

    # Create root node (parallel aggregation of the four parks)
    root_node = evaluator.add_parallel(
        id="Multi_State_RV_Trip_Planning",
        desc="Identify four U.S. state parks (all in different states) meeting the specified RV/ADA/reservation/proximity requirements, with supporting URLs.",
        parent=root,
        critical=False  # Note: JSON marks critical, but framework disallows critical parent with non-critical children; allow partial credit here.
    )

    # Extract all parks as presented in the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_parks(),
        template_class=ParksExtraction,
        extraction_name="parks_extraction"
    )

    # Keep a copy of raw extraction
    total_listed = len(extracted.parks) if extracted and extracted.parks else 0
    evaluator.add_custom_info(
        info={"total_parks_listed_in_answer": total_listed},
        info_type="extraction_stats",
        info_name="extraction_stats"
    )

    # Select first 4 parks for detailed verification; pad with empty if fewer
    selected: List[ParkItem] = list(extracted.parks[:4]) if extracted and extracted.parks else []
    while len(selected) < 4:
        selected.append(ParkItem())

    # Build subtrees for each of the four parks
    for i in range(4):
        await verify_state_park(evaluator, root_node, selected[i], i + 1)

    # Set-level constraints (placed under the 4th park node as per JSON)
    park4_node = evaluator.find_node("State_Park_4")
    set_level_parent = park4_node if park4_node is not None else root_node

    set_level = evaluator.add_parallel(
        id="Set_Level_Constraints",
        desc="Checks constraints that apply to the full set of parks (placed here to keep root second-level nodes as the 1st–4th items).",
        parent=set_level_parent,
        critical=True
    )

    # Exactly_Four_Parks_Total
    # Condition: The answer provides exactly four distinct state parks total (no fewer and no extras).
    # We judge by the extraction count and distinct park names in the extraction result.
    unique_names = {(_norm(p.park_name) or f"__null_{idx}") for idx, p in enumerate(extracted.parks)} if extracted and extracted.parks else set()
    exactly_four_total = (total_listed == 4) and (len(unique_names) == 4)
    evaluator.add_custom_node(
        result=exactly_four_total,
        id="Exactly_Four_Parks_Total",
        desc="Response provides exactly four distinct state parks total (no fewer and no additional parks beyond the four).",
        parent=set_level,
        critical=True
    )

    # All_Four_States_Distinct
    sel_states = [p.state for p in selected]
    norm_states = [s.strip().lower() for s in sel_states if isinstance(s, str) and s.strip()]
    all_four_states_distinct = (len(norm_states) == 4) and (len(set(norm_states)) == 4)
    evaluator.add_custom_node(
        result=all_four_states_distinct,
        id="All_Four_States_Distinct",
        desc="All four parks are located in four different U.S. states (no repeated state).",
        parent=set_level,
        critical=True
    )

    # Return evaluation summary
    return evaluator.get_summary()