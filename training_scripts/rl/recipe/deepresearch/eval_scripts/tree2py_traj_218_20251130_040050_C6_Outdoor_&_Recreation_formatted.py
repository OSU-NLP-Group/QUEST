import asyncio
import logging
import re
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator, AggregationStrategy


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "pnw_lookout_rentals"
TASK_DESCRIPTION = """Identify three fire lookout tower rentals in the Pacific Northwest states (Washington, Oregon, Idaho, or Montana) that meet ALL of the following requirements:

1. The lookout must be managed by the US Forest Service and available for reservation through Recreation.gov
2. The lookout must be accessible by vehicle (you can drive directly to or very near the lookout without a required hiking trail to reach it)
3. The lookout must accommodate at least 4 people for overnight stays
4. The lookout must be located at an elevation between 3,500 and 6,000 feet (inclusive)

For each of the three lookouts, provide the following information:
- Official lookout name
- National Forest name and location (state)
- Exact elevation in feet
- Maximum occupancy (number of people)
- Vehicle access requirements (specify whether standard vehicles are sufficient or if high-clearance/4WD vehicles are needed)
- Check-in time
- Check-out time
- Direct link to the lookout's reservation page on Recreation.gov
- Ranger district phone number for pre-arrival contact

All information must be verified with URLs from official sources (Recreation.gov or US Forest Service websites).
"""

ALLOWED_STATES_FULL = {"washington", "oregon", "idaho", "montana"}
ALLOWED_STATES_ABBR = {"wa", "or", "id", "mt"}


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class LookoutItem(BaseModel):
    official_name: Optional[str] = None
    national_forest: Optional[str] = None
    state: Optional[str] = None
    elevation_ft: Optional[str] = None
    max_occupancy: Optional[str] = None
    vehicle_access_description: Optional[str] = None
    vehicle_requirements: Optional[str] = None
    check_in_time: Optional[str] = None
    check_out_time: Optional[str] = None
    recreation_gov_url: Optional[str] = None
    ranger_district_phone: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class LookoutsExtraction(BaseModel):
    lookouts: List[LookoutItem] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_lookouts() -> str:
    return """
    Extract up to three lookout tower rentals as presented in the answer. For each lookout, extract exactly the following fields from the answer text:

    - official_name: the official lookout name as written in the answer
    - national_forest: the National Forest name
    - state: the U.S. state (can be full name or abbreviation)
    - elevation_ft: the exact elevation in feet as written (e.g., "5,200 ft", "5200 feet")
    - max_occupancy: the maximum number of people allowed to stay overnight (as written, e.g., "4", "up to 4 people")
    - vehicle_access_description: a short description of the vehicle access (e.g., "drive-up access", "drive to the base")
    - vehicle_requirements: the specific vehicle requirements noted (e.g., "standard 2WD", "high-clearance 4WD required")
    - check_in_time: the check-in time as written (e.g., "2:00 PM")
    - check_out_time: the check-out time as written (e.g., "11:00 AM")
    - recreation_gov_url: the direct reservation page URL on Recreation.gov if provided
    - ranger_district_phone: the ranger district phone number as written
    - sources: all URLs explicitly mentioned in the answer for this lookout (include all; do not invent). Keep only valid URLs.

    Rules:
    - Extract exactly what is in the answer; do not infer or add missing values.
    - If a field is not provided in the answer, return null for that field (or an empty list for sources).
    - Include any official URLs mentioned (Recreation.gov, US Forest Service/USDA pages) in the 'sources' array.
    - If more than three lookouts are present in the answer, include only the first three in the 'lookouts' array.
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def normalize_state(state_str: Optional[str]) -> Optional[str]:
    if not state_str:
        return None
    s = state_str.strip().lower().replace(".", "")
    # Handle common abbreviations and full names
    mapping = {
        "wa": "washington",
        "washington": "washington",
        "or": "oregon",
        "oregon": "oregon",
        "id": "idaho",
        "idaho": "idaho",
        "mt": "montana",
        "montana": "montana",
    }
    return mapping.get(s, None)


def is_allowed_state(state_str: Optional[str]) -> bool:
    ns = normalize_state(state_str)
    return ns in ALLOWED_STATES_FULL if ns else False


def parse_first_int(value: Optional[str]) -> Optional[int]:
    if not value:
        return None
    m = re.search(r"(\d{1,6})", value.replace(",", ""))
    if not m:
        return None
    try:
        return int(m.group(1))
    except Exception:
        return None


def combine_sources(item: LookoutItem) -> List[str]:
    urls: List[str] = []
    if item.recreation_gov_url and isinstance(item.recreation_gov_url, str) and item.recreation_gov_url.strip():
        urls.append(item.recreation_gov_url.strip())
    for u in item.sources or []:
        if isinstance(u, str) and u.strip():
            urls.append(u.strip())
    # Deduplicate while preserving order
    seen = set()
    deduped = []
    for u in urls:
        if u not in seen:
            deduped.append(u)
            seen.add(u)
    return deduped


def is_official_url(url: str) -> bool:
    u = (url or "").lower()
    return ("recreation.gov" in u) or ("fs.usda.gov" in u) or (u.endswith(".usda.gov"))


def have_any_official_url(item: LookoutItem) -> bool:
    srcs = combine_sources(item)
    return any(is_official_url(u) for u in srcs)


def index_label(i: int) -> str:
    return f"L{i+1}"


# --------------------------------------------------------------------------- #
# Verification for a single lookout                                           #
# --------------------------------------------------------------------------- #
async def verify_lookout(
    evaluator: Evaluator,
    parent_node,
    item: LookoutItem,
    idx: int,
) -> None:
    label = index_label(idx)
    lk_num = idx + 1

    # Parent node for this lookout (parallel, non-critical to allow partial credit across lookouts)
    node = evaluator.add_parallel(
        id=f"lookout_{lk_num}",
        desc=f"Lookout {lk_num} qualifies and includes all required reported fields with official URL verification",
        parent=parent_node,
        critical=False
    )

    # 1) Official source URLs exist and are official (critical)
    # We require at least a Recreation.gov URL or an official USFS URL.
    official_sources_ok = have_any_official_url(item)
    evaluator.add_custom_node(
        result=official_sources_ok,
        id=f"official_url_verification_{label}",
        desc="Official source URLs (Recreation.gov and/or US Forest Service sites) are provided to verify the stated information",
        parent=node,
        critical=True
    )

    # 2) Official lookout name provided (critical)
    evaluator.add_custom_node(
        result=bool(item.official_name and item.official_name.strip()),
        id=f"official_name_provided_{label}",
        desc="Official lookout name is provided",
        parent=node,
        critical=True
    )

    # 3) National Forest name provided (critical)
    evaluator.add_custom_node(
        result=bool(item.national_forest and item.national_forest.strip()),
        id=f"national_forest_name_provided_{label}",
        desc="National Forest name is provided",
        parent=node,
        critical=True
    )

    # 4) State provided and allowed (critical)
    evaluator.add_custom_node(
        result=is_allowed_state(item.state),
        id=f"state_provided_and_allowed_{label}",
        desc="State is provided and is one of Washington, Oregon, Idaho, or Montana",
        parent=node,
        critical=True
    )

    # Collect sources for verification-heavy checks
    sources_all = combine_sources(item)
    recgov_url_only = item.recreation_gov_url if (item.recreation_gov_url and item.recreation_gov_url.strip()) else None

    # 5) USFS managed (critical) - verify using official sources
    usfs_managed_leaf = evaluator.add_leaf(
        id=f"usfs_managed_{label}",
        desc="Lookout is managed by the US Forest Service",
        parent=node,
        critical=True
    )
    claim_usfs = f"The lookout named '{item.official_name or 'UNKNOWN'}' is managed by the U.S. Forest Service (USFS)."
    await evaluator.verify(
        claim=claim_usfs,
        node=usfs_managed_leaf,
        sources=sources_all,
        additional_instruction="Check the facility's operator/agency or official description. Variants like 'USDA Forest Service' or 'U.S. Forest Service' count as USFS."
    )

    # 6) Recreation.gov reservable and direct link provided (critical) - verify by the Recreation.gov page
    recgov_leaf = evaluator.add_leaf(
        id=f"recreation_gov_reservable_and_link_{label}",
        desc="Direct Recreation.gov reservation page link is provided and the lookout is reservable via Recreation.gov",
        parent=node,
        critical=True
    )
    claim_recgov = (
        f"The provided URL is the official Recreation.gov reservation page for '{item.official_name or 'UNKNOWN'}', "
        f"and the facility is reservable through Recreation.gov (i.e., reservations or booking info are present)."
    )
    await evaluator.verify(
        claim=claim_recgov,
        node=recgov_leaf,
        sources=recgov_url_only,
        additional_instruction="Verify the page is on Recreation.gov and indicates booking/reservations or availability. Minor name variants are acceptable."
    )

    # 7) Vehicle access: no required hike to reach it (critical)
    veh_access_leaf = evaluator.add_leaf(
        id=f"vehicle_access_no_required_hike_{label}",
        desc="Accessible by vehicle without a required hiking trail to reach it",
        parent=node,
        critical=True
    )
    claim_vehicle_access = (
        "You can drive directly to the lookout (or to its immediate vicinity) without any required hiking trail to reach it."
    )
    await evaluator.verify(
        claim=claim_vehicle_access,
        node=veh_access_leaf,
        sources=sources_all,
        additional_instruction="Look for language like 'drive-up access', 'road access to the lookout', or parking adjacent to the facility. Optional short walks from parking are acceptable; any required hike/trail to reach the lookout fails this condition."
    )

    # 8) Vehicle requirements specified (critical)
    veh_req_leaf = evaluator.add_leaf(
        id=f"vehicle_requirements_specified_{label}",
        desc="Vehicle access requirements are specified (e.g., standard vehicle vs high-clearance/4WD)",
        parent=node,
        critical=True
    )
    veh_req_str = item.vehicle_requirements or "UNKNOWN"
    claim_vehicle_req = f"The required vehicle access for reaching this lookout is described as '{veh_req_str}'."
    await evaluator.verify(
        claim=claim_vehicle_req,
        node=veh_req_leaf,
        sources=sources_all,
        additional_instruction="Check whether the page mentions passenger car/2WD, high-clearance, snow-capable, chains, or 4WD requirements. If the answer's description contradicts the page, mark incorrect."
    )

    # 9) Occupancy provided and at least 4 people (critical)
    occ_leaf = evaluator.add_leaf(
        id=f"occupancy_provided_and_min4_{label}",
        desc="Maximum occupancy is provided and is at least 4 people",
        parent=node,
        critical=True
    )
    occ_str = item.max_occupancy or "UNKNOWN"
    claim_occ = f"The maximum overnight occupancy is '{occ_str}', and this number is at least 4 people."
    await evaluator.verify(
        claim=claim_occ,
        node=occ_leaf,
        sources=sources_all,
        additional_instruction="Focus on the maximum people allowed on the official page. If the page shows fewer than 4, this should fail. Accept minor wording variations."
    )

    # 10) Elevation provided and in [3,500, 6,000] feet inclusive (critical)
    elev_leaf = evaluator.add_leaf(
        id=f"elevation_provided_and_in_range_{label}",
        desc="Exact elevation (feet) is provided and is between 3,500 and 6,000 feet inclusive",
        parent=node,
        critical=True
    )
    elev_str = item.elevation_ft or "UNKNOWN"
    elev_num = parse_first_int(item.elevation_ft)
    if elev_num is not None:
        range_clause = f"which is between 3,500 and 6,000 feet inclusive ({3500} ≤ {elev_num} ≤ {6000})."
    else:
        range_clause = "and this elevation must be between 3,500 and 6,000 feet inclusive."
    claim_elev = f"The lookout's elevation is '{elev_str}', {range_clause}"
    await evaluator.verify(
        claim=claim_elev,
        node=elev_leaf,
        sources=sources_all,
        additional_instruction="If the page states an elevation outside the range, fail this check. Numeric parsing may be needed. Accept minor formatting (e.g., 5,200 ft == 5200 feet)."
    )

    # 11) Check-in time provided (critical)
    checkin_leaf = evaluator.add_leaf(
        id=f"checkin_time_provided_{label}",
        desc="Check-in time is provided",
        parent=node,
        critical=True
    )
    ci_str = item.check_in_time or "UNKNOWN"
    claim_checkin = f"The check-in time for '{item.official_name or 'UNKNOWN'}' is '{ci_str}'."
    await evaluator.verify(
        claim=claim_checkin,
        node=checkin_leaf,
        sources=recgov_url_only or sources_all,
        additional_instruction="Use the official page (preferably Recreation.gov) to verify the check-in time. Accept equivalent phrasing (e.g., 'after 2 PM')."
    )

    # 12) Check-out time provided (critical)
    checkout_leaf = evaluator.add_leaf(
        id=f"checkout_time_provided_{label}",
        desc="Check-out time is provided",
        parent=node,
        critical=True
    )
    co_str = item.check_out_time or "UNKNOWN"
    claim_checkout = f"The check-out time for '{item.official_name or 'UNKNOWN'}' is '{co_str}'."
    await evaluator.verify(
        claim=claim_checkout,
        node=checkout_leaf,
        sources=recgov_url_only or sources_all,
        additional_instruction="Use the official page (preferably Recreation.gov) to verify the check-out time. Accept equivalent phrasing (e.g., 'by 11 AM')."
    )

    # 13) Ranger district phone provided (critical)
    ranger_phone_leaf = evaluator.add_leaf(
        id=f"ranger_district_phone_provided_{label}",
        desc="Ranger district phone number for pre-arrival contact is provided",
        parent=node,
        critical=True
    )
    phone_str = item.ranger_district_phone or "UNKNOWN"
    claim_phone = f"The ranger district phone number for '{item.official_name or 'UNKNOWN'}' is '{phone_str}'."
    await evaluator.verify(
        claim=claim_phone,
        node=ranger_phone_leaf,
        sources=sources_all,
        additional_instruction="Verify a phone number associated with the lookout or the responsible ranger district is present on the official sources. Allow minor formatting differences (dashes, spaces, parentheses)."
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
    Evaluate an answer for the Pacific Northwest USFS-managed fire lookout rentals task.
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
        default_model=model,
    )

    # Extract the lookouts as listed in the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_lookouts(),
        template_class=LookoutsExtraction,
        extraction_name="lookouts_extraction"
    )

    # Select the first three (pad if fewer)
    lookouts = list(extracted.lookouts[:3])
    while len(lookouts) < 3:
        lookouts.append(LookoutItem())

    # Root-level critical leaf: exactly three distinct lookouts are provided (by name)
    def distinct_three_ok(items: List[LookoutItem]) -> bool:
        # Consider only entries with a non-empty official_name
        names = [((li.official_name or "").strip().lower()) for li in items if (li.official_name and li.official_name.strip())]
        if len(names) != 3:
            return False
        return len(set(names)) == 3

    evaluator.add_custom_node(
        result=distinct_three_ok(lookouts),
        id="three_distinct_lookouts_provided",
        desc="Exactly three lookouts are provided and they are all distinct (no duplicates)",
        parent=root,
        critical=True
    )

    # Build verification for each lookout (parallel under root)
    for idx in range(3):
        await verify_lookout(evaluator, root, lookouts[idx], idx)

    return evaluator.get_summary()