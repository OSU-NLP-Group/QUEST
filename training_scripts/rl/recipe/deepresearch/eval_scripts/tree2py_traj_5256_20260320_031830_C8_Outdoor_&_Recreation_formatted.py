import asyncio
import logging
from typing import List, Optional, Dict, Any

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.llm_client.base_client import LLMClient


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "camping_reservations_multi_systems_2026"
TASK_DESCRIPTION = (
    "You are planning a summer camping road trip in 2026 and need to book campsites across multiple states. "
    "To properly coordinate your reservations, you need to identify campgrounds from different park systems and understand their specific reservation requirements. "
    "Identify 5 campgrounds that meet the following criteria: (1) 2 campgrounds must be in California State Parks, "
    "(2) 1 campground must be in Texas State Parks, and (3) 2 campgrounds must be in federal recreation areas (National Parks or National Forests). "
    "For each campground, provide: (a) The official reservation system used (specific website/platform name), "
    "(b) The reservation booking window (how many months in advance reservations can be made, and any specific timing details), "
    "(c) The time of day (including time zone) when new reservation dates become available, and (d) An official reference URL from the park system's website."
)


# --------------------------------------------------------------------------- #
# Data Models                                                                 #
# --------------------------------------------------------------------------- #
class CampgroundItem(BaseModel):
    name: Optional[str] = None
    # High-level categorization to help slotting (requested in the extraction prompt)
    # Expected values: "CA", "TX", "FEDERAL"
    category: Optional[str] = None

    # Additional descriptive fields (free text)
    park_system: Optional[str] = None          # e.g., "California State Parks", "Texas State Parks", "National Park", "National Forest"
    state: Optional[str] = None                # e.g., "CA", "California", "TX", "Texas", etc.
    federal_agency: Optional[str] = None       # e.g., "NPS", "USFS", "BLM" (for federal items)

    # Reservation details
    reservation_system: Optional[str] = None   # e.g., "ReserveCalifornia", "Recreation.gov", "TPWD Reservation System (ReserveAmerica portal)"
    booking_window: Optional[str] = None       # free-form text describing months in advance and rolling-window specifics
    opening_time: Optional[str] = None         # time-of-day release with time zone, e.g., "8:00 AM PT", "8:00 AM CT", "7:00 AM local time"
    reference_urls: List[str] = Field(default_factory=list)  # official references only (as stated in prompt)


class CampgroundsExtraction(BaseModel):
    campgrounds: List[CampgroundItem] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction Prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_campgrounds() -> str:
    return """
Identify and extract up to 10 campgrounds discussed in the answer, focusing on those that clearly belong to:
- California State Parks (Category: CA),
- Texas State Parks (Category: TX),
- Federal recreation areas (Category: FEDERAL), including National Parks (NPS), National Forests (USFS), or other federal agencies (e.g., BLM) that use Recreation.gov.

For each campground, extract the following fields exactly as they appear in the answer (do not invent anything):
- name: The official campground or park name.
- category: One of "CA", "TX", or "FEDERAL" based on the description in the answer.
- park_system: A short description of the system (e.g., "California State Parks", "Texas State Parks", "National Park", "National Forest", etc.), if available.
- state: The state (e.g., "CA", "California", "TX", "Texas", or another state for federal sites) if mentioned.
- federal_agency: If category is "FEDERAL", provide the agency name if stated (e.g., "NPS", "USFS", "BLM"); else null.
- reservation_system: The official reservation website/platform as stated (e.g., "ReserveCalifornia", "Recreation.gov", "Texas Parks & Wildlife reservation system (ReserveAmerica portal)").
- booking_window: The booking window text (e.g., "6 months rolling", "5 months in advance", "X months minus 1 day at 8 am", or specific period if provided).
- opening_time: The time of day when new inventory opens, including the time zone if it is stated (e.g., "8:00 AM PT", "8:00 AM CT", "7:00 AM local time"); if not present in the answer, return null.
- reference_urls: A list of official reference URLs mentioned in the answer for this campground’s reservation details.
  Official means:
  • California State Parks or ReserveCalifornia domains for CA sites (e.g., parks.ca.gov, reservecalifornia.com).
  • Texas Parks & Wildlife (TPWD) or the TX State Parks reservation portal (ReserveAmerica-branded) for TX sites (e.g., tpwd.texas.gov, texasstateparks.reserveamerica.com).
  • Recreation.gov or official federal agency domains (e.g., nps.gov, fs.usda.gov, blm.gov) for FEDERAL sites.
  Only include URLs that are explicitly present in the answer. Do not infer or construct URLs.

Return a JSON object with a 'campgrounds' array of CampgroundItem objects.
If any field is missing in the answer for a specific campground, set it to null (except 'reference_urls' which can be an empty list).
"""


# --------------------------------------------------------------------------- #
# Helper functions for selection and safety                                   #
# --------------------------------------------------------------------------- #
def _normalize_lower(s: Optional[str]) -> str:
    return (s or "").strip().lower()


def _has_official_domain(url: str, targets: List[str]) -> bool:
    u = url.lower()
    return any(t in u for t in targets)


def is_ca_item(cg: CampgroundItem) -> bool:
    if _normalize_lower(cg.category) == "ca":
        return True
    sys_l = _normalize_lower(cg.park_system)
    res_l = _normalize_lower(cg.reservation_system)
    # Domain heuristic from URLs as fallback
    url_has_ca = any(_has_official_domain(u, ["parks.ca.gov", "reservecalifornia.com"]) for u in cg.reference_urls)
    return (
        "california state park" in sys_l
        or "reserve california" in res_l
        or "reservecalifornia" in res_l
        or url_has_ca
    )


def is_tx_item(cg: CampgroundItem) -> bool:
    if _normalize_lower(cg.category) == "tx":
        return True
    sys_l = _normalize_lower(cg.park_system)
    res_l = _normalize_lower(cg.reservation_system)
    url_has_tx = any(_has_official_domain(u, ["tpwd.texas.gov", "texasstateparks.reserveamerica.com"]) for u in cg.reference_urls)
    return (
        "texas state park" in sys_l
        or "texas parks & wildlife" in sys_l
        or "tpwd" in sys_l
        or "reserveamerica" in res_l and "texas" in res_l
        or url_has_tx
    )


def is_federal_item(cg: CampgroundItem) -> bool:
    if _normalize_lower(cg.category) == "federal":
        return True
    sys_l = _normalize_lower(cg.park_system)
    fed_l = _normalize_lower(cg.federal_agency)
    res_l = _normalize_lower(cg.reservation_system)
    url_has_fed = any(_has_official_domain(u, ["recreation.gov", "nps.gov", "fs.usda.gov", "blm.gov"]) for u in cg.reference_urls)
    return (
        "national park" in sys_l
        or "national forest" in sys_l
        or fed_l in {"nps", "usfs", "blm"}
        or "recreation.gov" in res_l
        or url_has_fed
    )


def pick_campgrounds_for_slots(extracted: CampgroundsExtraction) -> Dict[str, List[CampgroundItem]]:
    ca_items = [cg for cg in extracted.campgrounds if is_ca_item(cg)]
    tx_items = [cg for cg in extracted.campgrounds if is_tx_item(cg)]
    fed_items = [cg for cg in extracted.campgrounds if is_federal_item(cg)]

    def empty_item() -> CampgroundItem:
        return CampgroundItem()

    selected = {
        "CA": ca_items[:2],
        "TX": tx_items[:1],
        "FED": fed_items[:2]
    }

    # Pad with empty items where needed
    while len(selected["CA"]) < 2:
        selected["CA"].append(empty_item())
    while len(selected["TX"]) < 1:
        selected["TX"].append(empty_item())
    while len(selected["FED"]) < 2:
        selected["FED"].append(empty_item())

    return selected


# --------------------------------------------------------------------------- #
# Verification builders                                                       #
# --------------------------------------------------------------------------- #
async def verify_california_campground(
    evaluator: Evaluator,
    parent_node,
    cg: CampgroundItem,
    group_idx: int
) -> None:
    """
    Build verification nodes for a California State Park campground.
    Enforces ReserveCalifornia, 6-month rolling window, 8:00 AM PT release, and official reference URLs.
    """
    group_id = f"california_campground_{group_idx}"
    group_desc = "First California State Park campground" if group_idx == 1 else "Second California State Park campground"

    group_node = evaluator.add_parallel(
        id=group_id,
        desc=group_desc,
        parent=parent_node,
        critical=False
    )

    # Existence gate (critical): requires a name and at least one reference URL
    exists = bool((cg.name or "").strip()) and len(cg.reference_urls) > 0
    exist_node = evaluator.add_custom_node(
        result=exists,
        id=f"{'ca1' if group_idx == 1 else 'ca2'}_exists",
        desc=f"{'CA#1' if group_idx == 1 else 'CA#2'}: Campground name and at least one official reference URL are provided",
        parent=group_node,
        critical=True
    )

    # Reservation system (critical)
    rs_node = evaluator.add_leaf(
        id=f"{'ca1' if group_idx == 1 else 'ca2'}_reservation_system",
        desc="Uses ReserveCalifornia.com as the reservation system",
        parent=group_node,
        critical=True
    )
    rs_claim = f"The official reservation system for {cg.name or 'this California State Parks campground'} is ReserveCalifornia (ReserveCalifornia.com)."
    await evaluator.verify(
        claim=rs_claim,
        node=rs_node,
        sources=cg.reference_urls,
        additional_instruction="Accept 'ReserveCalifornia', 'ReserveCalifornia.com', or the official CA State Parks reservation portal as equivalent."
    )

    # Booking window (critical) - 6 month rolling
    bw_node = evaluator.add_leaf(
        id=f"{'ca1' if group_idx == 1 else 'ca2'}_booking_window",
        desc="Has a 6-month rolling reservation window",
        parent=group_node,
        critical=True
    )
    bw_claim = f"Reservations for {cg.name or 'this California State Parks campground'} can be made six (6) months in advance on a rolling basis (ReserveCalifornia policy)."
    await evaluator.verify(
        claim=bw_claim,
        node=bw_node,
        sources=cg.reference_urls,
        additional_instruction="If the policy is stated as '6 months' or '6 months minus 1 day', consider it correct. If a different window (e.g., 5 or 7 months) is shown, mark as not supported."
    )

    # Opening time (critical) - 8:00 AM PT
    ot_node = evaluator.add_leaf(
        id=f"{'ca1' if group_idx == 1 else 'ca2'}_opening_time",
        desc="New reservation dates become available at 8:00 AM Pacific Time",
        parent=group_node,
        critical=True
    )
    ot_claim = f"New reservation dates for {cg.name or 'this California State Parks campground'} are released at 8:00 AM Pacific Time (PT)."
    await evaluator.verify(
        claim=ot_claim,
        node=ot_node,
        sources=cg.reference_urls,
        additional_instruction="Allow PT/PST/PDT equivalents. If the source states '8:00 a.m. PT' or similar, pass; otherwise, fail."
    )

    # Reference URL validity (critical)
    ref_node = evaluator.add_leaf(
        id=f"{'ca1' if group_idx == 1 else 'ca2'}_reference_url",
        desc="Provides valid reference URL from official California State Parks or ReserveCalifornia website",
        parent=group_node,
        critical=True
    )
    ref_claim = (
        "This page is an official California State Parks (parks.ca.gov) or ReserveCalifornia (reservecalifornia.com) page "
        "and provides official reservation/booking policy or timing details for the campground."
    )
    await evaluator.verify(
        claim=ref_claim,
        node=ref_node,
        sources=cg.reference_urls,
        additional_instruction="Check domain and page content. The page should be either parks.ca.gov or reservecalifornia.com and relevant to reservations/policy."
    )


async def verify_texas_campground(
    evaluator: Evaluator,
    parent_node,
    cg: CampgroundItem
) -> None:
    """
    Build verification nodes for a Texas State Park campground.
    Enforces TPWD/ReserveAmerica system, 5-month rolling, 8:00 AM CT, and official TPWD reference.
    """
    group_node = evaluator.add_parallel(
        id="texas_campground",
        desc="Texas State Park campground",
        parent=parent_node,
        critical=False
    )

    # Existence gate (critical)
    exists = bool((cg.name or "").strip()) and len(cg.reference_urls) > 0
    exist_node = evaluator.add_custom_node(
        result=exists,
        id="tx_exists",
        desc="TX: Campground name and at least one official reference URL are provided",
        parent=group_node,
        critical=True
    )

    # Reservation system (critical)
    rs_node = evaluator.add_leaf(
        id="tx_reservation_system",
        desc="Uses Texas Parks & Wildlife Department reservation system (ReserveAmerica platform)",
        parent=group_node,
        critical=True
    )
    rs_claim = (
        f"The official reservation system for {cg.name or 'this Texas State Parks campground'} is the Texas Parks & Wildlife Department "
        f"reservations system using the ReserveAmerica platform/portal."
    )
    await evaluator.verify(
        claim=rs_claim,
        node=rs_node,
        sources=cg.reference_urls,
        additional_instruction="Accept TPWD (tpwd.texas.gov) references or the Texas State Parks ReserveAmerica portal (texasstateparks.reserveamerica.com) as correct."
    )

    # Booking window (critical) - 5 months rolling
    bw_node = evaluator.add_leaf(
        id="tx_booking_window",
        desc="Has a 5-month rolling reservation window",
        parent=group_node,
        critical=True
    )
    bw_claim = f"Reservations for {cg.name or 'this Texas State Parks campground'} can be made up to five (5) months in advance on a rolling basis."
    await evaluator.verify(
        claim=bw_claim,
        node=bw_node,
        sources=cg.reference_urls,
        additional_instruction="If the page shows a 5-month advance window, pass; if a different window is stated, fail."
    )

    # Opening time (critical) - 8:00 AM CT
    ot_node = evaluator.add_leaf(
        id="tx_opening_time",
        desc="New reservation dates become available at 8:00 AM Central Time",
        parent=group_node,
        critical=True
    )
    ot_claim = f"New reservation dates for {cg.name or 'this Texas State Parks campground'} are released at 8:00 AM Central Time (CT)."
    await evaluator.verify(
        claim=ot_claim,
        node=ot_node,
        sources=cg.reference_urls,
        additional_instruction="Allow CT/CDT/CST equivalents. If the source lists 8:00 a.m. CT or equivalent, pass; else fail."
    )

    # Reference URL validity (critical)
    ref_node = evaluator.add_leaf(
        id="tx_reference_url",
        desc="Provides valid reference URL from official Texas Parks & Wildlife website",
        parent=group_node,
        critical=True
    )
    ref_claim = (
        "This page is an official Texas Parks & Wildlife Department page (tpwd.texas.gov) or the official Texas State Parks reservation portal "
        "(ReserveAmerica-branded) and provides reservation/booking policy or timing details."
    )
    await evaluator.verify(
        claim=ref_claim,
        node=ref_node,
        sources=cg.reference_urls,
        additional_instruction="Check domain and content. Accept tpwd.texas.gov and texasstateparks.reserveamerica.com as official."
    )


async def verify_federal_campground(
    evaluator: Evaluator,
    parent_node,
    cg: CampgroundItem,
    group_idx: int
) -> None:
    """
    Build verification nodes for a federal recreation area campground.
    Enforces Recreation.gov system, booking window specified, opening time/timing specified, accepts advance reservations, and official reference.
    """
    group_id = f"federal_campground_{group_idx}"
    group_desc = "First federal recreation area campground" if group_idx == 1 else "Second federal recreation area campground"

    group_node = evaluator.add_parallel(
        id=group_id,
        desc=group_desc,
        parent=parent_node,
        critical=False
    )

    # Existence gate (critical)
    exists = bool((cg.name or "").strip()) and len(cg.reference_urls) > 0
    exist_node = evaluator.add_custom_node(
        result=exists,
        id=f"fed{group_idx}_exists",
        desc=f"FED#{group_idx}: Campground name and at least one official reference URL are provided",
        parent=group_node,
        critical=True
    )

    # Reservation system (critical) - Recreation.gov
    rs_node = evaluator.add_leaf(
        id=f"fed{group_idx}_reservation_system",
        desc="Uses Recreation.gov as the reservation system",
        parent=group_node,
        critical=True
    )
    rs_claim = f"The official reservation website for {cg.name or 'this federal campground'} is Recreation.gov."
    await evaluator.verify(
        claim=rs_claim,
        node=rs_node,
        sources=cg.reference_urls,
        additional_instruction="The page should clearly indicate booking on Recreation.gov or provide a direct Recreation.gov linkage."
    )

    # Booking window specified (critical)
    bw_node = evaluator.add_leaf(
        id=f"fed{group_idx}_booking_window",
        desc="Specifies the advance booking window period for reservations",
        parent=group_node,
        critical=True
    )
    if (cg.booking_window or "").strip():
        bw_claim = f"The advance reservation window for {cg.name or 'this federal campground'} is: {cg.booking_window}."
        bw_instruction = "Verify that the source states an advance booking window compatible with the provided phrase (accept paraphrases and minor variations)."
    else:
        # Fall back to verifying that the source specifies an advance window at all
        bw_claim = f"The source specifies the advance booking/reservation window for {cg.name or 'this federal campground'}."
        bw_instruction = "Pass only if the page clearly states a specific advance booking window; otherwise fail."
    await evaluator.verify(
        claim=bw_claim,
        node=bw_node,
        sources=cg.reference_urls,
        additional_instruction=bw_instruction
    )

    # Opening time/timing specified (critical)
    ot_node = evaluator.add_leaf(
        id=f"fed{group_idx}_opening_time",
        desc="Specifies the time of day (including time zone) when new reservation dates become available, or provides specific release timing details",
        parent=group_node,
        critical=True
    )
    if (cg.opening_time or "").strip():
        ot_claim = f"The release time/timing for new reservation dates for {cg.name or 'this federal campground'} is: {cg.opening_time}."
        ot_instruction = "Verify that the page states this or an equivalent release time/timing detail (allow local time phrasing if applicable)."
    else:
        ot_claim = f"The source specifies a concrete release time or timing detail (with time zone or 'local time') for {cg.name or 'this federal campground'}."
        ot_instruction = "Pass only if the page clearly states a specific time or timing rule when new dates are released; otherwise fail."
    await evaluator.verify(
        claim=ot_claim,
        node=ot_node,
        sources=cg.reference_urls,
        additional_instruction=ot_instruction
    )

    # Accepts advance reservations (critical)
    avail_node = evaluator.add_leaf(
        id=f"fed{group_idx}_reservation_availability",
        desc="Confirms that the campground accepts advance reservations (not first-come-first-serve only)",
        parent=group_node,
        critical=True
    )
    avail_claim = f"{cg.name or 'This federal campground'} accepts advance reservations (i.e., it is not first-come, first-served only)."
    await evaluator.verify(
        claim=avail_claim,
        node=avail_node,
        sources=cg.reference_urls,
        additional_instruction="If the page indicates reservable sites or advance booking on Recreation.gov, pass. If it is walk-up/FCFS only, fail."
    )

    # Reference URL validity (critical)
    ref_node = evaluator.add_leaf(
        id=f"fed{group_idx}_reference_url",
        desc="Provides valid reference URL from Recreation.gov or official federal agency website",
        parent=group_node,
        critical=True
    )
    ref_claim = (
        "This page is either a Recreation.gov page or an official federal agency page (e.g., nps.gov, fs.usda.gov, blm.gov) "
        "and provides official reservation/booking information or policy for the campground."
    )
    await evaluator.verify(
        claim=ref_claim,
        node=ref_node,
        sources=cg.reference_urls,
        additional_instruction="Check domain and content relevance to reservations. Pass only if it's Recreation.gov or an official .gov agency site."
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
) -> Dict[str, Any]:
    """
    Evaluate an answer for the multi-system camping reservations task.
    """
    # Initialize evaluator (root must be non-critical to allow partial credit across groups)
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,
        agent_name=agent_name,
        answer_name=answer_name,
        client=client,
        task_description="Verify that all 5 required campgrounds are correctly identified with accurate reservation system details",
        answer=answer,
        global_cache=cache,
        global_semaphore=semaphore,
        logger=logger,
        default_model=model
    )

    # Extract structured campgrounds
    extracted = await evaluator.extract(
        prompt=prompt_extract_campgrounds(),
        template_class=CampgroundsExtraction,
        extraction_name="campgrounds_extraction"
    )

    # Selection for required slots
    selected = pick_campgrounds_for_slots(extracted)

    # Record expected composition as pseudo ground truth info to make the output self-describing
    evaluator.add_ground_truth({
        "required_slots": {
            "CA": 2,
            "TX": 1,
            "FED": 2
        },
        "selection_summary": {
            "CA_names": [c.name for c in selected["CA"]],
            "TX_names": [c.name for c in selected["TX"]],
            "FED_names": [c.name for c in selected["FED"]],
        }
    })

    # Build verifications for each slot
    # California (2)
    await verify_california_campground(evaluator, root, selected["CA"][0], 1)
    await verify_california_campground(evaluator, root, selected["CA"][1], 2)

    # Texas (1)
    await verify_texas_campground(evaluator, root, selected["TX"][0])

    # Federal (2)
    await verify_federal_campground(evaluator, root, selected["FED"][0], 1)
    await verify_federal_campground(evaluator, root, selected["FED"][1], 2)

    # Return final structured evaluation summary
    return evaluator.get_summary()