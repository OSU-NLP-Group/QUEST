import asyncio
import logging
import re
from typing import Optional, List, Dict, Any
from urllib.parse import urlparse

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "wilderness_permits_three_areas"
TASK_DESCRIPTION = """Identify three distinct wilderness areas in the United States that meet all of the following criteria for overnight backpacking:

1. Permit Reservation System: Wilderness permits for overnight use must be reserved through Recreation.gov during a designated quota season (not available through self-issue or walk-up only systems).

2. Quota Season Timing: The wilderness area's quota season must include at least part of the period from May through October.

3. Group Size: The wilderness area must officially allow groups of at least 8 people for overnight backpacking trips.

4. Geographic Diversity: At least one of the three wilderness areas must be located in California, and at least one must be located outside California.

5. Permit System Diversity: At least one wilderness area must allocate permits through a lottery system, and at least one must allocate permits through a first-come-first-served reservation system (non-lottery).

For each wilderness area, provide:
- The official name of the wilderness area or specific permit zone
- The managing federal agency (e.g., National Park Service, U.S. Forest Service)
- The state location
- A link to the official Recreation.gov permit page or the agency's official wilderness permit information page
- The quota season dates (start and end dates)
- Whether permits are allocated via lottery or first-come-first-served reservations
- How far in advance permits can be reserved
- The maximum group size allowed
- The per-person recreation fee for overnight wilderness permits
- Whether permits can be printed at home or must be picked up in person
- The no-show deadline or policy
- Direct URL references for all factual claims
"""


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class AreaItem(BaseModel):
    # Identification
    name: Optional[str] = None
    managing_agency: Optional[str] = None
    state: Optional[str] = None
    permit_page_url: Optional[str] = None  # Recreation.gov or official agency permit info page URL

    # Permit reservation system / timing
    quota_season_start: Optional[str] = None  # e.g., "May 1"
    quota_season_end: Optional[str] = None    # e.g., "Oct 31"
    allocation_method: Optional[str] = None   # e.g., "lottery" or "first-come-first-served"
    reservation_lead_time: Optional[str] = None  # e.g., "6 months", "168 days", "two weeks"

    # Group size
    max_group_size: Optional[str] = None  # Keep as string to be robust to various formats

    # Fees and logistics
    per_person_fee: Optional[str] = None
    print_or_pickup: Optional[str] = None      # e.g., "print at home" or "must pick up at ranger station"
    no_show_policy: Optional[str] = None       # e.g., "No-show if not picked up by 10am day of entry"

    # Direct references
    source_urls: List[str] = Field(default_factory=list)  # all supporting URLs cited in the answer for this area


class WildernessExtraction(BaseModel):
    areas: List[AreaItem] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_wilderness_areas() -> str:
    return """
Extract from the answer up to three wilderness areas (or specific permit zones) that the answer claims satisfy the task. For each area, return a JSON object with:

- name: The official wilderness area name or specific permit zone name used for the permit.
- managing_agency: The managing federal agency (e.g., National Park Service, U.S. Forest Service, Bureau of Land Management).
- state: The U.S. state(s) where the area is located (e.g., "California", "CA", "CA/NV").
- permit_page_url: The official Recreation.gov permit page URL or the official agency permit information page URL for the area/zone.
- quota_season_start: The claimed start date of the quota season (string as presented, e.g., "May 1" or "5/1").
- quota_season_end: The claimed end date of the quota season (string as presented, e.g., "Oct 31" or "10/31").
- allocation_method: "lottery" or "first-come-first-served" (or equivalent wording) as stated in the answer.
- reservation_lead_time: How far in advance permits can be reserved (e.g., "6 months", "26 weeks", "168 days", "March 1 release", etc.) exactly as written in the answer.
- max_group_size: The maximum group size allowed (string as written, e.g., "8", "12", "8-12", "up to 12").
- per_person_fee: The per-person recreation fee for the overnight wilderness permit (string as written; if none, return "0" or "none" if the answer states so).
- print_or_pickup: Whether permits can be printed at home or must be picked up in person (string as written).
- no_show_policy: The no-show deadline/policy (string as written).
- source_urls: An array of all direct URLs that the answer cites as supporting the above facts for this area (may include the same permit_page_url and/or additional official pages; include only actual URLs explicitly present in the answer text).

Rules:
- Extract information exactly as presented in the answer; do not invent.
- If a field is missing for an area, set it to null (except source_urls: return an empty list if not provided).
- Include up to three areas. If more are present, include only the first three in the order presented.
Return: {"areas": [ ... up to 3 AreaItem objects ... ]}
"""


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def _is_non_empty(s: Optional[str]) -> bool:
    return bool(s and str(s).strip())


def _looks_like_url(s: Optional[str]) -> bool:
    if not _is_non_empty(s):
        return False
    try:
        parsed = urlparse(s.strip())
        return parsed.scheme in ("http", "https") and _is_non_empty(parsed.netloc)
    except Exception:
        return False


def _combined_sources(area: AreaItem) -> List[str]:
    # Combine the main permit page URL and all cited source URLs, de-duplicated
    urls: List[str] = []
    if _looks_like_url(area.permit_page_url):
        urls.append(area.permit_page_url.strip())
    for u in area.source_urls:
        if _looks_like_url(u):
            urls.append(u.strip())
    # De-duplicate while preserving order
    seen = set()
    unique = []
    for u in urls:
        if u not in seen:
            unique.append(u)
            seen.add(u)
    return unique


def _is_official_url(url: str) -> bool:
    # Consider Recreation.gov and most .gov domains as official.
    try:
        netloc = urlparse(url).netloc.lower()
    except Exception:
        return False
    if "recreation.gov" in netloc:
        return True
    # Recognize common federal agency domains
    official_keywords = [
        ".gov",
        "nps.gov",
        "fs.usda.gov",
        "blm.gov",
        "doi.gov",
        "boreal"  # left placeholder; main check is .gov and specific known domains above
    ]
    return any(k in netloc for k in official_keywords)


def _has_any_official_url(urls: List[str]) -> bool:
    return any(_is_official_url(u) for u in urls)


def _parse_first_int(s: Optional[str]) -> Optional[int]:
    if not _is_non_empty(s):
        return None
    m = re.search(r"\d+", s)
    if not m:
        return None
    try:
        return int(m.group(0))
    except Exception:
        return None


def _normalize_allocation(method: Optional[str]) -> Optional[str]:
    if not _is_non_empty(method):
        return None
    m = method.strip().lower()
    if "lotter" in m:
        return "lottery"
    # treat variants as FCFS if they are clearly not lottery
    if "first" in m and "come" in m:
        return "fcfs"
    if "reservation" in m and "lotter" not in m:
        return "fcfs"
    if "rolling" in m and "lotter" not in m:
        return "fcfs"
    if "timed" in m and "release" in m and "lotter" not in m:
        return "fcfs"
    return m  # fallback (may still contain a recognizable keyword)


def _is_california(state_text: Optional[str]) -> bool:
    if not _is_non_empty(state_text):
        return False
    s = state_text.lower()
    return "california" in s or s.strip() == "ca" or " ca" in s or "ca/" in s or "/ca" in s or "ca," in s


def _is_non_california(state_text: Optional[str]) -> bool:
    if not _is_non_empty(state_text):
        return False
    s = state_text.lower()
    # If it contains CA exclusively, then not non-CA. If it includes another state without CA, count as non-CA.
    # Simple heuristic: if "california" or "ca" not present at all, treat as non-CA.
    return ("california" not in s) and (" ca" not in s) and (s.strip() != "ca") and ("ca/" not in s) and ("/ca" not in s) and ("ca," not in s)


def _claim_quota_overlap(start: Optional[str], end: Optional[str]) -> str:
    s = start if _is_non_empty(start) else "UNKNOWN"
    e = end if _is_non_empty(end) else "UNKNOWN"
    return f"The quota season is from '{s}' to '{e}', and it includes at least part of the period from May through October."


# --------------------------------------------------------------------------- #
# Area verification                                                           #
# --------------------------------------------------------------------------- #
async def verify_area(evaluator: Evaluator, parent_node, area: AreaItem, idx: int) -> None:
    # Build the subtree for one wilderness area
    area_node = evaluator.add_parallel(
        id=f"wilderness_area_{idx+1}",
        desc=f"{['First','Second','Third'][idx]} wilderness area meeting all specified criteria and providing required details",
        parent=parent_node,
        critical=False  # Keep non-critical per rubric for partial credit across areas
    )

    # 1) Area identification
    ident_node = evaluator.add_parallel(
        id=f"area_{idx+1}_identification",
        desc="Provide required identification fields for the area",
        parent=area_node,
        critical=True
    )

    evaluator.add_custom_node(
        result=_is_non_empty(area.name),
        id=f"area_{idx+1}_name",
        desc="Official name of the wilderness area or specific permit zone",
        parent=ident_node,
        critical=True
    )

    evaluator.add_custom_node(
        result=_is_non_empty(area.managing_agency),
        id=f"area_{idx+1}_managing_agency",
        desc="Managing federal agency",
        parent=ident_node,
        critical=True
    )

    evaluator.add_custom_node(
        result=_is_non_empty(area.state),
        id=f"area_{idx+1}_state_location",
        desc="State location",
        parent=ident_node,
        critical=True
    )

    evaluator.add_custom_node(
        result=_looks_like_url(area.permit_page_url),
        id=f"area_{idx+1}_official_permit_page_link",
        desc="Link to official Recreation.gov permit page or official agency permit information page",
        parent=ident_node,
        critical=True
    )

    # 2) Permit reservation system
    permit_node = evaluator.add_parallel(
        id=f"area_{idx+1}_permit_reservation_system",
        desc="Permit reservation system satisfies Recreation.gov + quota-season requirements",
        parent=area_node,
        critical=True
    )

    combined_urls = _combined_sources(area)

    # 2.1 Recreation.gov required during quota season
    recgov_leaf = evaluator.add_leaf(
        id=f"area_{idx+1}_recreation_gov_required_during_quota",
        desc="During the quota season, overnight wilderness permits must be reserved through Recreation.gov (i.e., not self-issue or walk-up-only during that season)",
        parent=permit_node,
        critical=True
    )
    claim_recgov = (
        f"During the quota season for '{area.name or 'this wilderness area'}', "
        f"overnight permits must be reserved via Recreation.gov (not self-issue or walk-up-only)."
    )
    await evaluator.verify(
        claim=claim_recgov,
        node=recgov_leaf,
        sources=combined_urls,
        additional_instruction=(
            "Verify that during the quota (peak) season, the process requires reserving permits through Recreation.gov. "
            "If the official page states that permits are obtained through Recreation.gov during quota season (even if off-season is walk-up or self-issue), the claim is supported."
        )
    )

    # 2.2 Quota season dates provided
    evaluator.add_custom_node(
        result=_is_non_empty(area.quota_season_start) and _is_non_empty(area.quota_season_end),
        id=f"area_{idx+1}_quota_season_dates_provided",
        desc="Quota season start and end dates are provided",
        parent=permit_node,
        critical=True
    )

    # 2.3 Quota season overlaps May–October
    quota_overlap_leaf = evaluator.add_leaf(
        id=f"area_{idx+1}_quota_season_overlaps_may_oct",
        desc="Quota season includes at least part of the period from May through October",
        parent=permit_node,
        critical=True
    )
    await evaluator.verify(
        claim=_claim_quota_overlap(area.quota_season_start, area.quota_season_end),
        node=quota_overlap_leaf,
        sources=combined_urls,
        additional_instruction=(
            "Check the dates of the quota season on the page(s). "
            "Confirm that at least some portion intersects the months May, June, July, August, September, or October."
        )
    )

    # 2.4 Reservation lead time ≥ 2 weeks
    lead_time_leaf = evaluator.add_leaf(
        id=f"area_{idx+1}_reservation_lead_time",
        desc="How far in advance permits can be reserved is stated and is at least 2 weeks",
        parent=permit_node,
        critical=True
    )
    lead_time_text = area.reservation_lead_time or "UNKNOWN"
    await evaluator.verify(
        claim=(
            f"The reservation lead time for this area is '{lead_time_text}', and this means permits can be reserved at least 14 days in advance."
        ),
        node=lead_time_leaf,
        sources=combined_urls,
        additional_instruction=(
            "Interpret the policy to determine how far in advance reservations can be made. "
            "If the policy says reservations open 30 days, 6 months, or similar before the entry date, that is ≥ 14 days. "
            "If it is 7 days or less, the claim is not supported."
        )
    )

    # 2.5 Allocation method (lottery vs first-come-first-served)
    allocation_leaf = evaluator.add_leaf(
        id=f"area_{idx+1}_allocation_method_stated",
        desc="Allocation method is stated (lottery vs first-come-first-served reservation)",
        parent=permit_node,
        critical=True
    )
    normalized_alloc = _normalize_allocation(area.allocation_method)
    if normalized_alloc == "lottery":
        alloc_claim = (
            f"Permits for '{area.name or 'this wilderness area'}' are allocated by lottery (not first-come-first-served)."
        )
    else:
        alloc_claim = (
            f"Permits for '{area.name or 'this wilderness area'}' are allocated by first-come-first-served reservations (non-lottery)."
        )
    await evaluator.verify(
        claim=alloc_claim,
        node=allocation_leaf,
        sources=combined_urls,
        additional_instruction=(
            "Determine whether the allocation mechanism is a lottery or first-come-first-served (FCFS). "
            "If the official page describes a lottery application/selection, that is lottery. "
            "If the page describes a release schedule or a window where users reserve available quota directly (without a lottery), that is FCFS."
        )
    )

    # 3) Group size requirement
    group_node = evaluator.add_parallel(
        id=f"area_{idx+1}_group_size_requirement",
        desc="Group size requirement is met and documented",
        parent=area_node,
        critical=True
    )

    group_leaf = evaluator.add_leaf(
        id=f"area_{idx+1}_max_group_size_stated",
        desc="Maximum group size is stated and is at least 8",
        parent=group_node,
        critical=True
    )
    max_group_text = area.max_group_size or "UNKNOWN"
    await evaluator.verify(
        claim=(
            f"The maximum overnight group size for '{area.name or 'this wilderness area'}' is '{max_group_text}', "
            f"and it allows at least 8 people."
        ),
        node=group_leaf,
        sources=combined_urls,
        additional_instruction=(
            "Find the official group size limit. "
            "If the stated maximum is 8 or greater, the claim is supported. "
            "If the limit is lower than 8, it is not supported."
        )
    )

    evaluator.add_custom_node(
        result=_has_any_official_url(combined_urls),
        id=f"area_{idx+1}_group_size_has_official_url_reference",
        desc="Direct URL reference is provided supporting the group size claim",
        parent=group_node,
        critical=True
    )

    # 4) Permit fee requirement
    fee_node = evaluator.add_parallel(
        id=f"area_{idx+1}_permit_fee_requirement",
        desc="Per-person recreation fee is stated and sourced",
        parent=area_node,
        critical=True
    )

    fee_leaf = evaluator.add_leaf(
        id=f"area_{idx+1}_per_person_fee_stated",
        desc="Per-person recreation fee for overnight wilderness permits is stated",
        parent=fee_node,
        critical=True
    )
    fee_text = area.per_person_fee or "UNKNOWN"
    await evaluator.verify(
        claim=f"The per-person recreation fee for overnight wilderness permits is '{fee_text}'.",
        node=fee_leaf,
        sources=combined_urls,
        additional_instruction=(
            "Confirm the per-person fee (or that no fee is charged if the page explicitly states $0 or 'no fee'). "
            "If fees vary by itinerary/entry point but the answer provides a correct example or a stated fee, that counts as supported."
        )
    )

    evaluator.add_custom_node(
        result=_has_any_official_url(combined_urls),
        id=f"area_{idx+1}_fee_has_official_url_reference",
        desc="Direct URL reference is provided supporting the fee claim",
        parent=fee_node,
        critical=True
    )

    # 5) Permit logistics: print/pickup and no-show
    logistics_node = evaluator.add_parallel(
        id=f"area_{idx+1}_permit_logistics_requirement",
        desc="Permit printing/pickup and no-show policy are stated and sourced",
        parent=area_node,
        critical=True
    )

    print_pickup_leaf = evaluator.add_leaf(
        id=f"area_{idx+1}_print_or_pickup_stated",
        desc="Whether permits can be printed at home or must be picked up in person is stated",
        parent=logistics_node,
        critical=True
    )
    print_pickup_text = area.print_or_pickup or "UNKNOWN"
    await evaluator.verify(
        claim=f"The permit pickup/printing policy is: '{print_pickup_text}'.",
        node=print_pickup_leaf,
        sources=combined_urls,
        additional_instruction=(
            "Verify whether the policy explicitly allows printing at home (e.g., email with PDF) or requires in-person pickup at a station. "
            "If both are allowed under conditions, the stated policy must match the official explanation."
        )
    )

    no_show_leaf = evaluator.add_leaf(
        id=f"area_{idx+1}_no_show_policy_stated",
        desc="No-show deadline or policy is stated",
        parent=logistics_node,
        critical=True
    )
    no_show_text = area.no_show_policy or "UNKNOWN"
    await evaluator.verify(
        claim=f"The no-show deadline/policy is: '{no_show_text}'.",
        node=no_show_leaf,
        sources=combined_urls,
        additional_instruction=(
            "Confirm the no-show definition or deadline (e.g., must pick up by a certain time, must enter on the entry date, etc.)."
        )
    )

    evaluator.add_custom_node(
        result=_has_any_official_url(combined_urls),
        id=f"area_{idx+1}_logistics_has_official_url_reference",
        desc="Direct URL reference is provided supporting print/pickup and no-show claims",
        parent=logistics_node,
        critical=True
    )

    # 6) URL references for claims (general)
    evaluator.add_custom_node(
        result=len(combined_urls) > 0,
        id=f"area_{idx+1}_url_references_for_claims",
        desc="Direct URL references are provided for factual claims made for this area (can reuse the same official page(s) if they support the claims)",
        parent=area_node,
        critical=True
    )


# --------------------------------------------------------------------------- #
# Main evaluation function                                                    #
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
    Evaluate an answer against the wilderness permit rubric.
    """
    evaluator = Evaluator()

    # Important: The original rubric marks root as critical, but the framework enforces
    # that critical parents must have all-critical children, which is not the case here.
    # We therefore initialize a non-critical root (default) to respect child-level criticality.
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

    # 1) Extract structured areas from the answer
    extraction = await evaluator.extract(
        prompt=prompt_extract_wilderness_areas(),
        template_class=WildernessExtraction,
        extraction_name="wilderness_areas_extraction"
    )

    # Ensure exactly 3 items for evaluation (pad with empties if fewer)
    areas: List[AreaItem] = list(extraction.areas[:3])
    while len(areas) < 3:
        areas.append(AreaItem())

    # 2) Build area subtrees and verify
    for idx in range(3):
        await verify_area(evaluator, root, areas[idx], idx)

    # 3) Global constraints under root: distinctness, geographic diversity, permit system diversity
    # Distinctness of names
    names = [a.name.strip().lower() for a in areas if _is_non_empty(a.name)]
    all_three_present = len([a for a in areas if _is_non_empty(a.name)]) == 3
    names_unique = len(set(names)) == 3 if all_three_present else False
    evaluator.add_custom_node(
        result=names_unique,
        id="distinctness",
        desc="All three wilderness areas/permit zones are distinct (not duplicates of the same area/zone)",
        parent=root,
        critical=True
    )

    # Geographic diversity: at least one CA and at least one non-CA
    has_ca = any(_is_california(a.state) for a in areas)
    has_non_ca = any(_is_non_california(a.state) for a in areas)
    evaluator.add_custom_node(
        result=has_ca and has_non_ca,
        id="geographic_diversity",
        desc="At least one wilderness area is located in California and at least one is located outside California",
        parent=root,
        critical=True
    )

    # Permit system diversity: at least one lottery and at least one FCFS
    methods_norm = [_normalize_allocation(a.allocation_method) for a in areas]
    has_lottery = any(m == "lottery" for m in methods_norm)
    has_fcfs = any(m == "fcfs" for m in methods_norm)
    evaluator.add_custom_node(
        result=has_lottery and has_fcfs,
        id="permit_system_diversity",
        desc="At least one wilderness area uses a lottery allocation system and at least one uses first-come-first-served reservations (non-lottery)",
        parent=root,
        critical=True
    )

    # 4) Return evaluation summary
    return evaluator.get_summary()