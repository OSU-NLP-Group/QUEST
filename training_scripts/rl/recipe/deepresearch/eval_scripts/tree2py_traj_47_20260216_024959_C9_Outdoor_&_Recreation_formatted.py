import asyncio
import logging
import re
from typing import Optional, List, Dict, Any, Tuple
from urllib.parse import urlparse

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.llm_client.base_client import LLMClient


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "sierra_trip_plan_2026"
TASK_DESCRIPTION = (
    "Complete 7-day backpacking trip plan for 10 people visiting exactly three Sierra Nevada wilderness areas "
    "during July 15-21, 2026, with required permit/regulation details and official sources."
)


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class PermitInfo(BaseModel):
    entry_trailhead_or_zone: Optional[str] = None
    permit_reservation_system: Optional[str] = None
    earliest_reservation_date_for_2026_07_15_entry: Optional[str] = None
    per_person_cost_and_fees: Optional[str] = None
    managing_agency: Optional[str] = None
    official_permit_urls: List[str] = Field(default_factory=list)


class AreaRegulationInfo(BaseModel):
    group_size_limit: Optional[str] = None
    group_size_url: Optional[str] = None
    bear_canister_requirement: Optional[str] = None  # e.g., "required", "recommended"
    bear_canister_url: Optional[str] = None
    camping_setback_feet: Optional[str] = None
    camping_setback_url: Optional[str] = None


class AreaInfo(BaseModel):
    name: Optional[str] = None
    order: Optional[int] = None  # 1, 2, or 3 indicating visit sequence
    nights: Optional[str] = None  # Prefer string to be robust (e.g., "2", "3 nights")
    permit: PermitInfo = PermitInfo()
    regulations: AreaRegulationInfo = AreaRegulationInfo()


class CampfirePermitInfo(BaseModel):
    required_for_stoves: Optional[str] = None  # e.g., "Yes" or "No"
    how_to_obtain: Optional[str] = None
    urls: List[str] = Field(default_factory=list)


class BearModelInfo(BaseModel):
    model_name: Optional[str] = None
    urls: List[str] = Field(default_factory=list)  # official sources confirming approval/validity


class CostSummary(BaseModel):
    total_permit_cost_for_10: Optional[str] = None  # store as string for flexibility (e.g., "$320")


class TripPlanExtraction(BaseModel):
    start_date: Optional[str] = None  # e.g., "July 15, 2026"
    end_date: Optional[str] = None    # e.g., "July 21, 2026"
    areas: List[AreaInfo] = Field(default_factory=list)
    campfire_permit: CampfirePermitInfo = CampfirePermitInfo()
    approved_bear_canister_model: BearModelInfo = BearModelInfo()
    total_cost: CostSummary = CostSummary()


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_trip_plan() -> str:
    return """
    Extract structured information about the 7-day Sierra Nevada backpacking trip plan described in the answer. Focus on exactly three wilderness areas visited in sequence during July 15–21, 2026 for a group of 10. Return a JSON object following this schema:

    Fields:
    - start_date: The trip start date string as stated in the answer (e.g., "July 15, 2026").
    - end_date: The trip end date string as stated in the answer (e.g., "July 21, 2026").
    - areas: An array of three objects, each with:
        - name: exact wilderness area name (e.g., "Desolation Wilderness", "Yosemite Wilderness").
        - order: visiting order number (1, 2, or 3).
        - nights: number of nights planned in that area (as a string, e.g., "2").
        - permit:
            - entry_trailhead_or_zone: entry trailhead/zone specified for the area.
            - permit_reservation_system: reservation/issuance system (e.g., "Recreation.gov", "self-issue", "Inyo NF wilderness permits", etc.).
            - earliest_reservation_date_for_2026_07_15_entry: earliest date you can reserve permits for a July 15, 2026 entry based on official rules (store the stated date string; do not compute if not specified).
            - per_person_cost_and_fees: per-person cost and any reservation fees during quota season (as a descriptive string including amounts when present).
            - managing_agency: agency/forest/park unit managing the permit (e.g., "National Park Service (Yosemite)", "US Forest Service – Inyo National Forest").
            - official_permit_urls: array of official reference URLs supporting requirements/cost/system for this area. Use only official sources when possible (e.g., recreation.gov, nps.gov, fs.usda.gov).
        - regulations:
            - group_size_limit: maximum on-trail group size limit for summer 2026 (string).
            - group_size_url: official URL supporting the stated group size.
            - bear_canister_requirement: whether bear canisters are required or recommended (string).
            - bear_canister_url: official URL supporting the bear canister requirement.
            - camping_setback_feet: minimum camping setback distance from water, in feet (string, e.g., "100 feet").
            - camping_setback_url: official URL supporting the camping setback rule.
    - campfire_permit:
        - required_for_stoves: whether a California Campfire Permit is required to use a backpacking/camping stove (string "Yes"/"No"/or explanation).
        - how_to_obtain: a short description of how to obtain it (e.g., "online training and permit at CAL FIRE site").
        - urls: array of official URLs supporting these statements (e.g., readyforwildfire.org, permit.preventwildfiresca.org, fs.usda.gov).
    - approved_bear_canister_model:
        - model_name: an example of an approved/allowed bear canister model valid across all three areas (e.g., "BearVault BV500").
        - urls: array of official URLs supporting approval/validity of the model across the Sierra.
    - total_cost:
        - total_permit_cost_for_10: the computed total wilderness permit cost for all 10 people across all three areas including all reservation fees and per-person charges, as a string (e.g., "$480").

    Rules:
    - Do NOT invent information; only extract what is explicitly present in the answer.
    - If some required item is missing in the answer, set the field to null (or empty array when appropriate).
    - Extract ALL official reference URLs that the answer provides for each required item.
    - For URLs, extract the actual link destinations (resolve markdown links to their target URLs). Include protocol (http/https).
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def _parse_int_from_str(s: Optional[str]) -> Optional[int]:
    if not s:
        return None
    m = re.search(r"(\d+)", s)
    if m:
        try:
            return int(m.group(1))
        except Exception:
            return None
    return None


def _domain_from_url(url: str) -> str:
    try:
        parsed = urlparse(url)
        return parsed.netloc.lower()
    except Exception:
        return ""


def _is_allowed_official_domain(domain: str) -> bool:
    # Allowed domains:
    # recreation.gov, nps.gov, fs.usda.gov, parks.ca.gov, readyforwildfire.org, permit.preventwildfiresca.org
    domain = domain.lower()
    allowed_suffixes = [
        "recreation.gov",
        "nps.gov",
        "fs.usda.gov",
        "parks.ca.gov",
        "readyforwildfire.org",
        "permit.preventwildfiresca.org",
    ]
    # Allow subdomains by suffix match
    for suf in allowed_suffixes:
        if domain == suf or domain.endswith("." + suf):
            return True
    return False


def _collect_all_urls(extracted: TripPlanExtraction) -> List[str]:
    urls: List[str] = []
    for area in extracted.areas:
        urls.extend(area.permit.official_permit_urls or [])
        if area.regulations.group_size_url:
            urls.append(area.regulations.group_size_url)
        if area.regulations.bear_canister_url:
            urls.append(area.regulations.bear_canister_url)
        if area.regulations.camping_setback_url:
            urls.append(area.regulations.camping_setback_url)
    urls.extend(extracted.campfire_permit.urls or [])
    urls.extend(extracted.approved_bear_canister_model.urls or [])
    # Deduplicate while preserving order
    seen = set()
    deduped = []
    for u in urls:
        if not u:
            continue
        if u not in seen:
            deduped.append(u)
            seen.add(u)
    return deduped


def _areas_distinct(areas: List[AreaInfo]) -> bool:
    names = [a.name.strip().lower() for a in areas if a.name]
    if len(names) != 3:
        return False
    return len(set(names)) == 3


def _orders_are_sequence(areas: List[AreaInfo]) -> bool:
    try:
        orders = sorted([int(a.order) for a in areas if a.order is not None])
        return orders == [1, 2, 3]
    except Exception:
        return False


def _nights_each_2_or_3(areas: List[AreaInfo]) -> bool:
    if len(areas) != 3:
        return False
    ok = True
    for a in areas:
        nights = _parse_int_from_str(a.nights)
        if nights is None or nights not in (2, 3):
            ok = False
            break
    return ok


# --------------------------------------------------------------------------- #
# Verification subroutines                                                    #
# --------------------------------------------------------------------------- #
async def build_itinerary_requirements(evaluator: Evaluator, parent_node, extracted: TripPlanExtraction) -> None:
    node = evaluator.add_parallel(
        id="Itinerary_Requirements",
        desc="Trip timing and allocation across the three areas.",
        parent=parent_node,
        critical=True,
    )

    # Trip_Dates (simple verify from answer text)
    leaf_dates = evaluator.add_leaf(
        id="Trip_Dates",
        desc="Trip dates are July 15-21, 2026 (7 days total).",
        parent=node,
        critical=True,
    )
    claim_dates = "The trip dates are July 15 through July 21, 2026 (inclusive), making 7 days total."
    await evaluator.verify(
        claim=claim_dates,
        node=leaf_dates,
        additional_instruction="Verify the dates exactly from the answer text; allow minor formatting variations like 'July 15–21, 2026'."
    )

    # Exactly_Three_Areas_In_Sequence (custom structural check)
    result_three_sequence = len(extracted.areas) == 3 and _orders_are_sequence(extracted.areas)
    evaluator.add_custom_node(
        result=result_three_sequence,
        id="Exactly_Three_Areas_In_Sequence",
        desc="Plan specifies exactly three wilderness areas in a clear visit order (Area 1 -> Area 2 -> Area 3).",
        parent=node,
        critical=True,
    )

    # Nights_Per_Area (custom structural check: each area 2–3 nights)
    evaluator.add_custom_node(
        result=_nights_each_2_or_3(extracted.areas),
        id="Nights_Per_Area",
        desc="Plan allocates 2–3 nights in each of the three wilderness areas.",
        parent=node,
        critical=True,
    )


async def build_wilderness_areas_identification(
    evaluator: Evaluator,
    parent_node,
    extracted: TripPlanExtraction,
    official_url_leaves: Dict[int, Any]  # map area_index -> VerificationNode for official URL existence (optional prerequisite)
) -> None:
    node = evaluator.add_parallel(
        id="Wilderness_Areas_Identification",
        desc="Identify three distinct wilderness areas meeting the stated geographic/permit constraints.",
        parent=parent_node,
        critical=True,
    )

    # Area name leaves with support claims (use official permit URLs if available)
    for idx in range(3):
        area = extracted.areas[idx] if idx < len(extracted.areas) else AreaInfo()
        leaf = evaluator.add_leaf(
            id=f"Area_{idx+1}_Name",
            desc=f"Provides the exact name of Wilderness Area {idx+1} (a California wilderness area in the Sierra Nevada).",
            parent=node,
            critical=True,
        )
        claim = f"The wilderness area named '{area.name or ''}' is located in the Sierra Nevada region of California."
        srcs = area.permit.official_permit_urls or None
        prereq = []
        if idx in official_url_leaves:
            prereq = [official_url_leaves[idx]]
        await evaluator.verify(
            claim=claim,
            node=leaf,
            sources=srcs if srcs else None,
            additional_instruction="Confirm the wilderness area location (Sierra Nevada in California) using the official source when provided.",
            extra_prerequisites=prereq if prereq else None
        )

    # All_Areas_Distinct (custom)
    evaluator.add_custom_node(
        result=_areas_distinct(extracted.areas),
        id="All_Areas_Distinct",
        desc="The three wilderness area names are all different from one another.",
        parent=node,
        critical=True,
    )

    # All_Areas_Require_Overnight_Permits -> break into 3 leaves (to avoid aggregating multiple checks in one leaf)
    subnode = evaluator.add_parallel(
        id="All_Areas_Require_Overnight_Permits",
        desc="Each selected area requires an overnight wilderness permit (per cited official sources).",
        parent=node,
        critical=True,
    )
    for idx in range(3):
        area = extracted.areas[idx] if idx < len(extracted.areas) else AreaInfo()
        leaf = evaluator.add_leaf(
            id=f"Area_{idx+1}_Overnight_Permit_Required",
            desc=f"Overnight wilderness permits are required in Area {idx+1}.",
            parent=subnode,
            critical=True,
        )
        claim = f"Overnight wilderness permits are required in {area.name or ''} during quota season (or generally for overnight trips)."
        srcs = area.permit.official_permit_urls or None
        await evaluator.verify(
            claim=claim,
            node=leaf,
            sources=srcs if srcs else None,
            additional_instruction="Verify on the official permit page whether overnight permits are required for this wilderness area."
        )


async def build_wilderness_permit_details(
    evaluator: Evaluator,
    parent_node,
    extracted: TripPlanExtraction
) -> Dict[int, Any]:
    node = evaluator.add_parallel(
        id="Wilderness_Permit_Details",
        desc="Permit details for each of the three wilderness areas.",
        parent=parent_node,
        critical=True,
    )

    official_url_leaves: Dict[int, Any] = {}

    for idx in range(3):
        area = extracted.areas[idx] if idx < len(extracted.areas) else AreaInfo()
        subnode = evaluator.add_parallel(
            id=f"Permit_For_Area_{idx+1}",
            desc=f"Permit details for Wilderness Area {idx+1}.",
            parent=node,
            critical=True,
        )

        # Entry trailhead/zone
        leaf_entry = evaluator.add_leaf(
            id=f"Area_{idx+1}_Entry_Trailhead_Or_Zone",
            desc=f"Entry trailhead or zone is specified for Area {idx+1}.",
            parent=subnode,
            critical=True,
        )
        claim_entry = f"The entry trailhead/zone for {area.name or ''} is '{area.permit.entry_trailhead_or_zone or ''}'."
        await evaluator.verify(
            claim=claim_entry,
            node=leaf_entry,
            sources=area.permit.official_permit_urls or None,
            additional_instruction="Confirm that the stated entry trailhead/zone is valid or listed for this wilderness permit system."
        )

        # Permit reservation system
        leaf_system = evaluator.add_leaf(
            id=f"Area_{idx+1}_Permit_Reservation_System",
            desc=f"Permit reservation/issuance system for Area {idx+1} is specified (e.g., Recreation.gov, self-issue, etc.).",
            parent=subnode,
            critical=True,
        )
        claim_system = f"Permits for {area.name or ''} are reserved/issued via '{area.permit.permit_reservation_system or ''}'."
        await evaluator.verify(
            claim=claim_system,
            node=leaf_system,
            sources=area.permit.official_permit_urls or None,
            additional_instruction="Verify the reservation/issuance system from official sources (e.g., Recreation.gov, NPS, USFS)."
        )

        # Earliest reservation date for July 15, 2026 entry
        leaf_earliest = evaluator.add_leaf(
            id=f"Area_{idx+1}_Earliest_Reservation_Date",
            desc=f"Earliest date permits can be reserved for a July 15, 2026 entry is stated and consistent with the system's rules (per cited official source).",
            parent=subnode,
            critical=True,
        )
        claim_earliest = (
            f"For an entry date of July 15, 2026 in {area.name or ''}, the earliest reservation date is "
            f"'{area.permit.earliest_reservation_date_for_2026_07_15_entry or ''}' per official rules."
        )
        await evaluator.verify(
            claim=claim_earliest,
            node=leaf_earliest,
            sources=area.permit.official_permit_urls or None,
            additional_instruction="If the page states a general rule (e.g., 6 months in advance), apply it to July 15, 2026 to confirm the earliest reservation date."
        )

        # Costs and fees
        leaf_costs = evaluator.add_leaf(
            id=f"Area_{idx+1}_Permit_Costs_And_Fees",
            desc=f"Per-person cost and reservation fees (if any) during quota season are stated for Area {idx+1}.",
            parent=subnode,
            critical=True,
        )
        claim_costs = (
            f"The per-person wilderness permit cost and reservation fees for {area.name or ''} are: "
            f"'{area.permit.per_person_cost_and_fees or ''}'."
        )
        await evaluator.verify(
            claim=claim_costs,
            node=leaf_costs,
            sources=area.permit.official_permit_urls or None,
            additional_instruction="Confirm costs/fees for permits during quota season from official sources."
        )

        # Managing agency
        leaf_agency = evaluator.add_leaf(
            id=f"Area_{idx+1}_Permit_Managing_Agency",
            desc=f"The agency (or forest district/park unit) managing the permit is identified for Area {idx+1}.",
            parent=subnode,
            critical=True,
        )
        claim_agency = f"The permit for {area.name or ''} is managed by '{area.permit.managing_agency or ''}'."
        await evaluator.verify(
            claim=claim_agency,
            node=leaf_agency,
            sources=area.permit.official_permit_urls or None,
            additional_instruction="Verify the managing agency/forest/park unit via official sources."
        )

        # Official permit URL existence (custom existence check)
        official_urls_present = bool(area.permit.official_permit_urls)
        leaf_urls_exist = evaluator.add_custom_node(
            result=official_urls_present,
            id=f"Area_{idx+1}_Permit_Official_URL",
            desc=f"Provides at least one official reference URL supporting Area {idx+1} permit requirements/costs/system.",
            parent=subnode,
            critical=True,
        )
        official_url_leaves[idx] = leaf_urls_exist

    # Permit_Systems_All_Different (custom distinct management systems/agencies)
    agencies = [a.permit.managing_agency or "" for a in extracted.areas[:3]]
    agencies_norm = [ag.strip().lower() for ag in agencies if ag]
    all_diff = len(agencies_norm) == 3 and len(set(agencies_norm)) == 3
    evaluator.add_custom_node(
        result=all_diff,
        id="Permit_Systems_All_Different",
        desc="The three areas use different permit management systems as defined in the question (managed by different agencies or forest districts/permit authorities), supported by cited sources.",
        parent=node,
        critical=True,
    )

    return official_url_leaves


async def build_campfire_permit(evaluator: Evaluator, parent_node, extracted: TripPlanExtraction) -> None:
    node = evaluator.add_parallel(
        id="California_Campfire_Permit",
        desc="Campfire permit requirement for backpacking stoves and how to obtain it.",
        parent=parent_node,
        critical=True,
    )

    leaf_required = evaluator.add_leaf(
        id="Campfire_Permit_Required_For_Stove",
        desc="States whether a California Campfire Permit is required for using a backpacking stove in these areas (with supporting official source URL).",
        parent=node,
        critical=True,
    )
    claim_required = (
        f"A California Campfire Permit is required to use a backpacking/camping stove in these wilderness areas: "
        f"'{extracted.campfire_permit.required_for_stoves or ''}'."
    )
    await evaluator.verify(
        claim=claim_required,
        node=leaf_required,
        sources=extracted.campfire_permit.urls or None,
        additional_instruction="Confirm the campfire permit requirement for stoves via official sources (e.g., CAL FIRE or USFS)."
    )

    leaf_how = evaluator.add_leaf(
        id="Campfire_Permit_How_To_Obtain",
        desc="Explains how to obtain the California Campfire Permit and provides an official source URL.",
        parent=node,
        critical=True,
    )
    claim_how = f"How to obtain the California Campfire Permit: '{extracted.campfire_permit.how_to_obtain or ''}'."
    await evaluator.verify(
        claim=claim_how,
        node=leaf_how,
        sources=extracted.campfire_permit.urls or None,
        additional_instruction="Verify steps for obtaining the CA Campfire Permit from official sources."
    )


async def build_group_size_limits(evaluator: Evaluator, parent_node, extracted: TripPlanExtraction) -> None:
    node = evaluator.add_parallel(
        id="Group_Size_Limits",
        desc="Maximum on-trail group size limits for summer 2026 for each area.",
        parent=parent_node,
        critical=True,
    )
    for idx in range(3):
        area = extracted.areas[idx] if idx < len(extracted.areas) else AreaInfo()
        leaf = evaluator.add_leaf(
            id=f"Group_Size_Limit_Area_{idx+1}",
            desc=f"States the maximum on-trail group size for Area {idx+1} during summer 2026 with an official source URL.",
            parent=node,
            critical=True,
        )
        claim = f"The maximum on-trail group size for {area.name or ''} during summer 2026 is '{area.regulations.group_size_limit or ''}'."
        sources = [area.regulations.group_size_url] if area.regulations.group_size_url else None
        await evaluator.verify(
            claim=claim,
            node=leaf,
            sources=sources,
            additional_instruction="Confirm the group size limit from the official area regulations page."
        )


async def build_bear_canister_requirements(evaluator: Evaluator, parent_node, extracted: TripPlanExtraction) -> None:
    node = evaluator.add_parallel(
        id="Bear_Canister_Requirements",
        desc="Bear canister rules per area and one example model valid across all three.",
        parent=parent_node,
        critical=True,
    )
    for idx in range(3):
        area = extracted.areas[idx] if idx < len(extracted.areas) else AreaInfo()
        leaf = evaluator.add_leaf(
            id=f"Bear_Canister_Status_Area_{idx+1}",
            desc=f"Indicates whether bear canisters are required or recommended in Area {idx+1} with an official source URL.",
            parent=node,
            critical=True,
        )
        claim = f"In {area.name or ''}, bear canisters are '{area.regulations.bear_canister_requirement or ''}'."
        sources = [area.regulations.bear_canister_url] if area.regulations.bear_canister_url else None
        await evaluator.verify(
            claim=claim,
            node=leaf,
            sources=sources,
            additional_instruction="Verify whether bear canisters are required/recommended via official area regulations."
        )

    # One approved model valid across all three
    leaf_model = evaluator.add_leaf(
        id="One_Approved_Model_Valid_Across_All_Three",
        desc="Provides at least one example of an approved bear canister model that is valid across all three areas, supported by official sources.",
        parent=node,
        critical=True,
    )
    # Combine area-specific bear canister URLs with dedicated model URLs
    model_sources: List[str] = []
    model_sources.extend(extracted.approved_bear_canister_model.urls or [])
    for a in extracted.areas[:3]:
        if a.regulations.bear_canister_url:
            model_sources.append(a.regulations.bear_canister_url)
    # Deduplicate
    seen = set()
    model_sources = [u for u in model_sources if u and (not (u in seen) and not seen.add(u))]
    claim_model = (
        f"The bear canister model '{extracted.approved_bear_canister_model.model_name or ''}' is an approved/allowed food storage container "
        f"valid across all three wilderness areas on this itinerary."
    )
    await evaluator.verify(
        claim=claim_model,
        node=leaf_model,
        sources=model_sources or None,
        additional_instruction="Confirm the model's approval/acceptance based on official sources; ensure applicability across all areas."
    )


async def build_camping_setback_requirements(evaluator: Evaluator, parent_node, extracted: TripPlanExtraction) -> None:
    node = evaluator.add_parallel(
        id="Camping_Setback_Requirements",
        desc="Minimum campsite setback distances from water sources for each area.",
        parent=parent_node,
        critical=True,
    )
    for idx in range(3):
        area = extracted.areas[idx] if idx < len(extracted.areas) else AreaInfo()
        leaf = evaluator.add_leaf(
            id=f"Setback_Area_{idx+1}_Feet",
            desc=f"States the minimum camping setback distance from water for Area {idx+1} in feet with an official source URL.",
            parent=node,
            critical=True,
        )
        claim = f"The minimum camping setback distance from water in {area.name or ''} is '{area.regulations.camping_setback_feet or ''}'."
        sources = [area.regulations.camping_setback_url] if area.regulations.camping_setback_url else None
        await evaluator.verify(
            claim=claim,
            node=leaf,
            sources=sources,
            additional_instruction="Verify the camping setback rule (distance from water) from official area regulations."
        )


async def build_total_permit_costs(evaluator: Evaluator, parent_node, extracted: TripPlanExtraction) -> None:
    node = evaluator.add_parallel(
        id="Total_Permit_Costs",
        desc="Total permit costs for all 10 people across all three areas including all fees/charges.",
        parent=parent_node,
        critical=True,
    )
    leaf = evaluator.add_leaf(
        id="Total_Cost_Computation",
        desc="Computes the total wilderness permit cost for 10 people across all three areas including reservation fees and per-person charges.",
        parent=node,
        critical=True,
    )
    claim = (
        f"The total wilderness permit cost for all 10 people across all three areas, including reservation fees and per-person charges, "
        f"is stated as '{extracted.total_cost.total_permit_cost_for_10 or ''}'."
    )
    await evaluator.verify(
        claim=claim,
        node=leaf,
        additional_instruction=(
            "Check that the answer explicitly provides a computed total cost covering all 10 people across all three areas, "
            "and that it claims to include both reservation fees and per-person charges."
        ),
    )


async def build_official_source_domains_constraint(evaluator: Evaluator, parent_node, extracted: TripPlanExtraction) -> None:
    node = evaluator.add_parallel(
        id="Official_Source_Domains_Constraint",
        desc="All citations use the explicitly allowed official source providers.",
        parent=parent_node,
        critical=True,
    )
    all_urls = _collect_all_urls(extracted)
    domains = [(_domain_from_url(u) if u else "") for u in all_urls]
    allowed_flags = [(_is_allowed_official_domain(d) if d else False) for d in domains]
    all_allowed = all(allowed_flags) if all_urls else False

    evaluator.add_custom_info(
        info={
            "total_urls_collected": len(all_urls),
            "domains": domains,
            "all_allowed": all_allowed,
        },
        info_type="url_domains_check",
        info_name="official_source_domains_check"
    )

    evaluator.add_custom_node(
        result=all_allowed,
        id="Allowed_Official_Source_Domains_Only",
        desc=(
            "All reference URLs used to support required claims are from Recreation.gov, National Park Service (nps.gov), "
            "US Forest Service (fs.usda.gov), or California official sites (parks.ca.gov, readyforwildfire.org, permit.preventwildfiresca.org)."
        ),
        parent=node,
        critical=True,
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
) -> Dict:
    """
    Evaluate the provided answer for the Sierra Nevada 7-day trip planning task.
    """
    # Initialize evaluator with a non-critical root, then add a critical Trip_Plan node under it.
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

    # Extract structured plan
    extracted: TripPlanExtraction = await evaluator.extract(
        prompt=prompt_extract_trip_plan(),
        template_class=TripPlanExtraction,
        extraction_name="trip_plan_extraction",
    )

    # Build Trip_Plan critical node
    trip_node = evaluator.add_parallel(
        id="Trip_Plan",
        desc=TASK_DESCRIPTION,
        parent=root,
        critical=True,
    )

    # Subtrees
    await build_itinerary_requirements(evaluator, trip_node, extracted)
    # Build permit details first to get official URL existence leaf nodes to use as prerequisites for area identification verification
    official_url_leaves = await build_wilderness_permit_details(evaluator, trip_node, extracted)
    await build_wilderness_areas_identification(evaluator, trip_node, extracted, official_url_leaves)
    await build_campfire_permit(evaluator, trip_node, extracted)
    await build_group_size_limits(evaluator, trip_node, extracted)
    await build_bear_canister_requirements(evaluator, trip_node, extracted)
    await build_camping_setback_requirements(evaluator, trip_node, extracted)
    await build_total_permit_costs(evaluator, trip_node, extracted)
    await build_official_source_domains_constraint(evaluator, trip_node, extracted)

    # Final summary
    return evaluator.get_summary()