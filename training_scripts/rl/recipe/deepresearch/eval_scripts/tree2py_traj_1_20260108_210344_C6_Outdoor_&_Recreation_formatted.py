import asyncio
import logging
from typing import Optional, List, Dict, Any
from urllib.parse import urlparse

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy

# --------------------------------------------------------------------------- #
# Task constants                                                              #
# --------------------------------------------------------------------------- #
TASK_ID = "onp_wilderness_group_camping_2026_07_25_27"
TASK_DESCRIPTION = (
    "You are planning a wilderness camping trip in Olympic National Park for a group of 10 adults (all age 16 or older) "
    "from July 25-27, 2026 (arriving July 25, departing July 27, spending 2 nights). Identify a suitable camping area for this trip "
    "and provide the following information: 1. Camping Area Name, 2. Quota Status, 3. Designated Group Site Requirement, "
    "4. Bear Canister Requirement, 5. Campfire Allowance, 6. Permit Reservation System and release timing, 7. Permit Fees total. "
    "Provide reference URLs from official National Park Service sources or Recreation.gov to support each requirement."
)

ADULTS_16_PLUS = 10
NIGHTS = 2
PER_PERSON_PER_NIGHT_FEE = 8  # USD, ages 16+
RESERVATION_FEE_PER_PERMIT = 6  # USD, non-refundable
EXPECTED_TOTAL_FEE = ADULTS_16_PLUS * NIGHTS * PER_PERSON_PER_NIGHT_FEE + RESERVATION_FEE_PER_PERMIT  # 166

# --------------------------------------------------------------------------- #
# Extraction Models                                                           #
# --------------------------------------------------------------------------- #
class CampingAreaInfo(BaseModel):
    name: Optional[str] = None
    within_onp_wilderness_statement: Optional[bool] = None
    designated_group_site_7_12: Optional[bool] = None
    quota_status: Optional[str] = None  # e.g., "quota-managed" or "non-quota"
    site_wilderness_urls: List[str] = Field(default_factory=list)
    quota_status_urls: List[str] = Field(default_factory=list)
    group_site_rule_urls: List[str] = Field(default_factory=list)


class BearInfo(BaseModel):
    required: Optional[bool] = None
    justification: Optional[str] = None
    bear_rule_urls: List[str] = Field(default_factory=list)


class CampfireInfo(BaseModel):
    allowed: Optional[bool] = None
    justification: Optional[str] = None
    elevation_ft: Optional[str] = None
    campfire_rule_urls: List[str] = Field(default_factory=list)


class PermitSystemInfo(BaseModel):
    system: Optional[str] = None  # Expect "Recreation.gov" or equivalent
    summer_release_timing: Optional[str] = None  # Expect "April 15 at 7 AM PT" for summer season (May 15–Oct 15)
    permit_system_urls: List[str] = Field(default_factory=list)


class FeesInfo(BaseModel):
    per_person_per_night: Optional[str] = None  # Expect "$8" or "8 USD"
    reservation_fee_per_permit: Optional[str] = None  # Expect "$6" or "6 USD"
    total_fee: Optional[str] = None  # The total computed by the answer (should be $166)
    fees_urls: List[str] = Field(default_factory=list)


class FoodStorageInfo(BaseModel):
    acknowledges_rule: Optional[bool] = None
    food_storage_urls: List[str] = Field(default_factory=list)


class TripPlanExtraction(BaseModel):
    camping_area: Optional[CampingAreaInfo] = None
    bear: Optional[BearInfo] = None
    campfire: Optional[CampfireInfo] = None
    permit: Optional[PermitSystemInfo] = None
    fees: Optional[FeesInfo] = None
    food_storage: Optional[FoodStorageInfo] = None


# --------------------------------------------------------------------------- #
# Extraction Prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_trip_plan() -> str:
    return """
    Extract the structured information for the proposed Olympic National Park wilderness camping plan for a group of 10 adults (all 16+) for July 25–27, 2026 (2 nights).
    If the answer mentions multiple camping areas, select the single specific camping area or designated group site the answer actually proposes for the group and extract fields for that primary choice only.
    Return JSON using the exact keys and types below. If a field is not clearly stated, set it to null or an empty array as appropriate.

    Structure and field semantics:
    - camping_area:
        - name (string or null): The specific camping area or designated group site name (not just a general region).
        - within_onp_wilderness_statement (boolean or null): Whether the answer explicitly states the site is within Olympic National Park wilderness.
        - designated_group_site_7_12 (boolean or null): Whether the answer explicitly confirms the selected site is one of ONP's designated group sites that groups of 7–12 must use.
        - quota_status (string or null): "quota-managed" or "non-quota" (or equivalent wording used by the answer). If unknown, null.
        - site_wilderness_urls (array of strings): Official URLs referenced for the site identification and/or wilderness eligibility (prefer NPS.gov).
        - quota_status_urls (array of strings): Official URLs referenced for quota vs non-quota status (NPS.gov or Recreation.gov).
        - group_site_rule_urls (array of strings): Official URLs referenced for designated group sites requirement or the designated group site list (NPS.gov preferred).

    - bear:
        - required (boolean or null): Whether the answer states that bear canisters (Animal Resistant Food Containers) are required at the selected location.
        - justification (string or null): Short justification text provided in the answer (e.g., which rule/list this falls under).
        - bear_rule_urls (array of strings): Official URLs used to justify the bear canister requirement determination (NPS.gov preferred).

    - campfire:
        - allowed (boolean or null): Whether the answer states campfires are allowed at the selected location.
        - justification (string or null): Explanation referencing ONP rule that campfires/wood-burning stoves are allowed only below 3,500 feet and the site’s elevation/context.
        - elevation_ft (string or null): Any elevation figure for the site the answer mentions (keep as text; do not convert).
        - campfire_rule_urls (array of strings): Official URLs supporting the campfire rule and/or area-specific allowance rules (NPS.gov preferred).

    - permit:
        - system (string or null): The reservation system used (expect "Recreation.gov" or equivalent phrasing).
        - summer_release_timing (string or null): The summer season release timing (expect "April 15 at 7 AM PT" for May 15–Oct 15).
        - permit_system_urls (array of strings): Official URLs supporting the system and release timing (NPS.gov or Recreation.gov).

    - fees:
        - per_person_per_night (string or null): The per-person (16+) per-night wilderness fee stated (expect "$8" but capture verbatim).
        - reservation_fee_per_permit (string or null): The reservation fee per permit (expect "$6" but capture verbatim).
        - total_fee (string or null): The total permit fees the answer computed for 10 adults and 2 nights (capture verbatim, e.g., "$166").
        - fees_urls (array of strings): Official URLs supporting the fee rules (NPS.gov or Recreation.gov).

    - food_storage:
        - acknowledges_rule (boolean or null): Whether the answer acknowledges that all food, garbage, and scented items must be secured 24/7.
        - food_storage_urls (array of strings): Official URLs supporting this rule (NPS.gov preferred).

    URL extraction requirements:
    - Extract only URLs explicitly present in the answer.
    - Include full URLs (with protocol). If a URL is malformed or missing protocol, attempt to add http:// or https:// if obvious; otherwise, omit.
    - Prefer official National Park Service (nps.gov) or Recreation.gov URLs for supporting evidence. If the answer provides multiple links, include them all in the relevant arrays.

    Do not invent content. If any piece of information is not given, set it to null (or [] for URLs).
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def filter_official_urls(urls: List[str]) -> List[str]:
    """Return only official NPS or Recreation.gov URLs."""
    official = []
    for u in urls or []:
        try:
            netloc = urlparse(u).netloc.lower()
            if "nps.gov" in netloc or "recreation.gov" in netloc:
                official.append(u)
        except Exception:
            continue
    # Deduplicate while preserving order
    seen = set()
    deduped = []
    for u in official:
        if u not in seen:
            seen.add(u)
            deduped.append(u)
    return deduped


def normalize_quota_status(text: Optional[str]) -> Optional[str]:
    """Normalize quota status to 'quota-managed' or 'non-quota' when possible."""
    if not text:
        return None
    t = text.strip().lower()
    # Heuristics
    if any(k in t for k in ["quota", "reservation required", "advance reservation", "limited quota", "lottery"]):
        if any(k in t for k in ["non-quota", "no quota", "walk-in", "walk up"]):
            # conflicting phrasing; return as-is to let judge decide
            return text
        return "quota-managed"
    if any(k in t for k in ["non-quota", "no quota", "no advance reservation", "walk-in", "walk up"]):
        return "non-quota"
    return text


# --------------------------------------------------------------------------- #
# Verification subroutines                                                    #
# --------------------------------------------------------------------------- #
async def add_camping_area_checks(evaluator: Evaluator, parent, plan: TripPlanExtraction) -> None:
    node = evaluator.add_parallel(
        id="camping_area",
        desc="Camping area selection satisfies ONP wilderness + group-size constraints and is clearly identified",
        parent=parent,
        critical=True,
    )

    ca = plan.camping_area or CampingAreaInfo()

    # camping_area_name (existence)
    has_name = bool(ca.name and str(ca.name).strip())
    evaluator.add_custom_node(
        result=has_name,
        id="camping_area_name",
        desc="Provides a specific camping area / designated group site name (not just a vague region)",
        parent=node,
        critical=True
    )

    # within_onp_wilderness (acknowledgement in the answer)
    within_leaf = evaluator.add_leaf(
        id="within_onp_wilderness",
        desc="Camping area is within Olympic National Park wilderness areas (as required)",
        parent=node,
        critical=True
    )
    area_name = ca.name or "the selected camping area"
    claim_within = f"The answer explicitly states that {area_name} is within Olympic National Park wilderness."
    await evaluator.verify(
        claim=claim_within,
        node=within_leaf,
        additional_instruction="Judge based on the answer text provided; this node checks acknowledgement. External support is verified separately in the citations section."
    )

    # group_size_within_max_12
    max12_leaf = evaluator.add_leaf(
        id="group_size_within_max_12",
        desc="Trip acknowledges/enforces maximum overnight wilderness group size of 12 and that the group of 10 complies",
        parent=node,
        critical=True
    )
    claim_max12 = "The answer acknowledges the ONP wilderness maximum overnight group size is 12 and confirms that a group of 10 complies with this limit."
    await evaluator.verify(
        claim=claim_max12,
        node=max12_leaf,
        additional_instruction="Look for explicit acknowledgement of the 12-person max wilderness group size and that a group of 10 is compliant."
    )

    # designated_group_site_for_7_to_12
    dgs_leaf = evaluator.add_leaf(
        id="designated_group_site_for_7_to_12",
        desc="For a group size of 10 (7–12), confirms the selected area is one of the park’s designated group sites required for groups of 7–12",
        parent=node,
        critical=True
    )
    # Prefer verifying with official URLs if provided; else fall back to answer-only verification
    dgs_urls = filter_official_urls(ca.group_site_rule_urls)
    claim_dgs = f"The selected area '{ca.name}' is one of Olympic National Park's designated group sites required for groups of 7–12 people."
    await evaluator.verify(
        claim=claim_dgs,
        node=dgs_leaf,
        sources=dgs_urls if dgs_urls else None,
        additional_instruction=(
            "If URLs are provided, verify from the official designated group site rule/list page(s) that this named site is indeed a designated group site for 7–12. "
            "Accept minor name variants (case, punctuation). If no URLs are provided, judge based on the answer text."
        )
    )


async def add_quota_status_check(evaluator: Evaluator, parent, plan: TripPlanExtraction) -> None:
    ca = plan.camping_area or CampingAreaInfo()
    qs_leaf = evaluator.add_leaf(
        id="quota_status",
        desc="States whether the selected area is quota-managed (advance reservation) or non-quota",
        parent=parent,
        critical=True
    )

    normalized = normalize_quota_status(ca.quota_status) or (ca.quota_status or "").strip()
    if normalized and normalized.lower() in ["quota-managed", "quota managed", "quota"]:
        qs_claim = "The selected camping area is quota-managed and requires an advance reservation."
    elif normalized and normalized.lower() in ["non-quota", "non quota", "no quota", "walk-in", "walk up"]:
        qs_claim = "The selected camping area is non-quota (i.e., no specific quota/advance reservation requirement)."
    else:
        # fallback to generic based on answer text
        qs_claim = "The answer clearly states whether the selected camping area is quota-managed or non-quota."

    qs_urls = filter_official_urls(ca.quota_status_urls)
    await evaluator.verify(
        claim=qs_claim,
        node=qs_leaf,
        sources=qs_urls if qs_urls else None,
        additional_instruction="If URLs are provided, verify the quota vs non-quota status from an official NPS or Recreation.gov page. If none, judge based on the answer text."
    )


async def add_bear_canister_checks(evaluator: Evaluator, parent, plan: TripPlanExtraction) -> None:
    node = evaluator.add_parallel(
        id="bear_canister_requirement",
        desc="Bear canister requirement for the selected area is correctly stated and justified from regulations",
        parent=parent,
        critical=True
    )
    bear = plan.bear or BearInfo()
    area_name = (plan.camping_area.name if plan.camping_area and plan.camping_area.name else "the selected location")

    # bear_canister_yes_no (answer acknowledgement)
    b_yes_no = evaluator.add_leaf(
        id="bear_canister_yes_no",
        desc="States whether bear canisters are required at the selected location (yes/no)",
        parent=node,
        critical=True
    )
    if bear.required is True:
        claim_yes_no = f"The answer states that bear canisters (Animal Resistant Food Containers) are required at {area_name}."
    elif bear.required is False:
        claim_yes_no = f"The answer states that bear canisters (Animal Resistant Food Containers) are not required at {area_name}."
    else:
        claim_yes_no = f"The answer clearly states whether bear canisters are required at {area_name}."
    await evaluator.verify(
        claim=claim_yes_no,
        node=b_yes_no,
        additional_instruction="Judge based on the answer text. External justification is checked in the next node."
    )

    # bear_canister_justification (evidence-based)
    b_just = evaluator.add_leaf(
        id="bear_canister_justification",
        desc="Justifies bear-canister determination using the park’s listed required areas (or other official rule) and explains why the selected area does/does not fall under it",
        parent=node,
        critical=True
    )
    b_urls = filter_official_urls(bear.bear_rule_urls)
    if bear.required is True:
        claim_just = f"Per official ONP rules, bear canisters are required at or for the area encompassing {area_name}."
    elif bear.required is False:
        claim_just = f"Per official ONP rules, bear canisters are not required for {area_name} or its immediate area."
    else:
        claim_just = f"Per official ONP rules, the bear canister requirement for {area_name} is correctly applied in the answer."
    await evaluator.verify(
        claim=claim_just,
        node=b_just,
        sources=b_urls if b_urls else None,
        additional_instruction="Only consider official NPS or Recreation.gov sources. The page(s) should explicitly state required ARFC areas or rules applicable to the location/zone."
    )


async def add_campfire_checks(evaluator: Evaluator, parent, plan: TripPlanExtraction) -> None:
    node = evaluator.add_parallel(
        id="campfire_allowance",
        desc="Campfire allowance at the selected location is correctly stated and justified using the park’s elevation-based rule and site elevation/context",
        parent=parent,
        critical=True
    )
    cf = plan.campfire or CampfireInfo()
    area_name = (plan.camping_area.name if plan.camping_area and plan.camping_area.name else "the selected location")
    elev_text = (cf.elevation_ft or "").strip()

    # campfire_yes_no (answer acknowledgement)
    cf_yes_no = evaluator.add_leaf(
        id="campfire_yes_no",
        desc="States whether campfires are allowed at the selected camping area (yes/no)",
        parent=node,
        critical=True
    )
    if cf.allowed is True:
        claim_cf = f"The answer states that campfires are allowed at {area_name}."
    elif cf.allowed is False:
        claim_cf = f"The answer states that campfires are not allowed at {area_name}."
    else:
        claim_cf = f"The answer clearly states whether campfires are allowed at {area_name}."
    await evaluator.verify(
        claim=claim_cf,
        node=cf_yes_no,
        additional_instruction="Judge based on the answer text. External rule justification is verified separately."
    )

    # campfire_justification (evidence-based rule)
    cf_just = evaluator.add_leaf(
        id="campfire_justification",
        desc="Explains why based on the park rule that campfires/wood-burning stoves are allowed only below 3,500 feet and the relevant elevation for the selected location",
        parent=node,
        critical=True
    )
    cf_urls = filter_official_urls(cf.campfire_rule_urls)
    allow_text = "allowed" if cf.allowed else "not allowed"
    elev_part = f" The answer references an elevation of approximately {elev_text} ft for context." if elev_text else ""
    claim_cf_rule = (
        f"In Olympic National Park wilderness, campfires (wood-burning) are allowed only below 3,500 feet. "
        f"According to official sources, the rule is correctly applied to conclude campfires are {allow_text} at {area_name}.{elev_part}"
    )
    await evaluator.verify(
        claim=claim_cf_rule,
        node=cf_just,
        sources=cf_urls if cf_urls else None,
        additional_instruction="Only consider official NPS sources. Focus on verifying the 3,500 ft threshold rule; site-specific context can be inferred if the rule clearly dictates the allowance."
    )


async def add_permit_system_checks(evaluator: Evaluator, parent, plan: TripPlanExtraction) -> None:
    node = evaluator.add_parallel(
        id="permit_reservation_system_and_timing",
        desc="Identifies the correct reservation system and the release timing for summer-season wilderness permits",
        parent=parent,
        critical=True
    )
    ps = plan.permit or PermitSystemInfo()

    # reservation_system_recgov
    sys_leaf = evaluator.add_leaf(
        id="reservation_system_recgov",
        desc="Identifies Recreation.gov as the wilderness permit reservation system",
        parent=node,
        critical=True
    )
    sys_urls = filter_official_urls(ps.permit_system_urls)
    claim_sys = "Olympic National Park wilderness/backcountry permits are reserved/managed via Recreation.gov."
    await evaluator.verify(
        claim=claim_sys,
        node=sys_leaf,
        sources=sys_urls if sys_urls else None,
        additional_instruction="Only consider official NPS or Recreation.gov pages."
    )

    # summer_release_timing
    timing_leaf = evaluator.add_leaf(
        id="summer_release_timing",
        desc="States that summer permits (May 15–Oct 15) are released April 15 at 7 AM PT (and applies this to the July 2026 trip context)",
        parent=node,
        critical=True
    )
    claim_timing = "In Olympic National Park, summer season wilderness permits (May 15–Oct 15) are released on April 15 at 7:00 AM Pacific Time."
    await evaluator.verify(
        claim=claim_timing,
        node=timing_leaf,
        sources=sys_urls if sys_urls else None,
        additional_instruction="Only consider official NPS or Recreation.gov pages that state the summer permit release timing."
    )


async def add_fees_checks(evaluator: Evaluator, parent, plan: TripPlanExtraction) -> None:
    node = evaluator.add_parallel(
        id="permit_fees_total",
        desc="Correctly computes total permit fees for 10 adults (16+) for 2 nights including the reservation fee",
        parent=parent,
        critical=True
    )
    fees = plan.fees or FeesInfo()

    # fee_components_correct
    comp_leaf = evaluator.add_leaf(
        id="fee_components_correct",
        desc="Uses the stated fee rules: $8 per person per night (16+) plus $6 non-refundable reservation fee per permit",
        parent=node,
        critical=True
    )
    comp_urls = filter_official_urls(fees.fees_urls)
    claim_components = "The fee rules are $8 per person per night (ages 16+) plus a $6 non-refundable reservation fee per permit."
    await evaluator.verify(
        claim=claim_components,
        node=comp_leaf,
        sources=comp_urls if comp_urls else None,
        additional_instruction="Verify the fee rules on an official NPS or Recreation.gov page."
    )

    # total_fee_calculation_correct
    total_leaf = evaluator.add_leaf(
        id="total_fee_calculation_correct",
        desc="Computes the correct total using the trip parameters (10 adults, 2 nights) and the stated fee rules",
        parent=node,
        critical=True
    )
    claim_total = f"The total permit fees for 10 adults (16+) for 2 nights, using $8/person/night plus one $6 reservation fee, equals ${EXPECTED_TOTAL_FEE}."
    await evaluator.verify(
        claim=claim_total,
        node=total_leaf,
        additional_instruction=(
            "Use the answer text and perform the arithmetic check: 10 people * 2 nights * $8 = $160; plus $6 reservation fee = $166 total. "
            "Pass only if the answer's stated total matches $166 (allow minor formatting variations)."
        )
    )


async def add_food_storage_check(evaluator: Evaluator, parent, plan: TripPlanExtraction) -> None:
    leaf = evaluator.add_leaf(
        id="food_storage_general_rule",
        desc="Acknowledges the rule that all food, garbage, and scented items must be secured 24/7",
        parent=parent,
        critical=True
    )
    claim_fs = "The answer acknowledges that all food, garbage, and scented items must be secured at all times (24/7)."
    await evaluator.verify(
        claim=claim_fs,
        node=leaf,
        additional_instruction="Judge based on the answer text."
    )


async def add_citations_checks(evaluator: Evaluator, parent, plan: TripPlanExtraction) -> None:
    node = evaluator.add_parallel(
        id="citations",
        desc="Provides official reference URLs (NPS.gov or Recreation.gov) supporting each required attribute",
        parent=parent,
        critical=True
    )

    ca = plan.camping_area or CampingAreaInfo()
    bear = plan.bear or BearInfo()
    cf = plan.campfire or CampfireInfo()
    ps = plan.permit or PermitSystemInfo()
    fees = plan.fees or FeesInfo()
    fs = plan.food_storage or FoodStorageInfo()

    # 1) Site and wilderness eligibility
    site_urls = filter_official_urls(ca.site_wilderness_urls)
    c1 = evaluator.add_leaf(
        id="citation_for_site_and_wilderness_eligibility",
        desc="Provides at least one official URL supporting the camping area identification and/or that it is within ONP wilderness",
        parent=node,
        critical=True
    )
    claim1 = (
        "An official page (NPS preferred) supports the camping area identification and/or indicates that the selected area is within Olympic National Park wilderness."
    )
    await evaluator.verify(
        claim=claim1,
        node=c1,
        sources=site_urls if site_urls else None,
        additional_instruction="Only consider URLs from nps.gov or recreation.gov as official."
    )

    # 2) Quota status
    quota_urls = filter_official_urls(ca.quota_status_urls)
    c2 = evaluator.add_leaf(
        id="citation_for_quota_status",
        desc="Provides at least one official URL supporting the stated quota vs non-quota status for the selected area",
        parent=node,
        critical=True
    )
    normalized = normalize_quota_status(ca.quota_status) or (ca.quota_status or "").strip()
    if normalized and normalized.lower() in ["quota-managed", "quota managed", "quota"]:
        quota_claim = "An official page confirms the selected area is quota-managed and requires an advance reservation."
    elif normalized and normalized.lower() in ["non-quota", "non quota", "no quota", "walk-in", "walk up"]:
        quota_claim = "An official page confirms the selected area is non-quota (no specific advance-reservation quota)."
    else:
        quota_claim = "An official page supports the stated quota vs non-quota status for the selected area."
    await evaluator.verify(
        claim=quota_claim,
        node=c2,
        sources=quota_urls if quota_urls else None,
        additional_instruction="Only consider nps.gov or recreation.gov URLs."
    )

    # 3) Designated group-site rule
    dgs_urls = filter_official_urls(ca.group_site_rule_urls)
    c3 = evaluator.add_leaf(
        id="citation_for_designated_group_site_rule",
        desc="Provides at least one official URL supporting the designated group-site requirement for groups of 7–12 and/or the designated group-site list",
        parent=node,
        critical=True
    )
    claim3 = "An official page states that groups of 7–12 in ONP wilderness must use designated group sites and/or provides the designated group site list."
    await evaluator.verify(
        claim=claim3,
        node=c3,
        sources=dgs_urls if dgs_urls else None,
        additional_instruction="Only consider nps.gov or recreation.gov URLs."
    )

    # 4) Bear canister rule
    bear_urls = filter_official_urls(bear.bear_rule_urls)
    c4 = evaluator.add_leaf(
        id="citation_for_bear_canister_rule",
        desc="Provides at least one official URL supporting the bear canister requirement rule used in the justification",
        parent=node,
        critical=True
    )
    claim4 = "An official page states the ONP bear canister (ARFC) requirements applicable to the selected location/area."
    await evaluator.verify(
        claim=claim4,
        node=c4,
        sources=bear_urls if bear_urls else None,
        additional_instruction="Only consider nps.gov or recreation.gov URLs."
    )

    # 5) Campfire rule (3,500 ft threshold)
    cf_urls = filter_official_urls(cf.campfire_rule_urls)
    c5 = evaluator.add_leaf(
        id="citation_for_campfire_rule",
        desc="Provides at least one official URL supporting the campfire rule (including the 3,500 ft threshold) and/or area-specific fire allowance",
        parent=node,
        critical=True
    )
    claim5 = "An official page states that in ONP wilderness, campfires (wood-burning) are allowed only below 3,500 feet (and/or area-specific rules consistent with this)."
    await evaluator.verify(
        claim=claim5,
        node=c5,
        sources=cf_urls if cf_urls else None,
        additional_instruction="Only consider nps.gov or recreation.gov URLs."
    )

    # 6) Permit system and summer release timing
    ps_urls = filter_official_urls(ps.permit_system_urls)
    c6 = evaluator.add_leaf(
        id="citation_for_permit_system_and_release_timing",
        desc="Provides at least one official URL supporting use of Recreation.gov and the April 15 7 AM PT summer permit release timing",
        parent=node,
        critical=True
    )
    claim6 = (
        "An official page states that winter/spring/summer permits are managed through Recreation.gov and that summer season permits "
        "(May 15–Oct 15) are released on April 15 at 7 AM Pacific Time."
    )
    await evaluator.verify(
        claim=claim6,
        node=c6,
        sources=ps_urls if ps_urls else None,
        additional_instruction="Only consider nps.gov or recreation.gov URLs."
    )

    # 7) Fees ($8/person/night, $6 reservation fee)
    fee_urls = filter_official_urls(fees.fees_urls)
    c7 = evaluator.add_leaf(
        id="citation_for_fees",
        desc="Provides at least one official URL supporting the $8/person/night fee and the $6 reservation fee",
        parent=node,
        critical=True
    )
    claim7 = "An official page states the ONP wilderness fees: $8 per person (age 16+) per night, and a $6 non-refundable reservation fee per permit."
    await evaluator.verify(
        claim=claim7,
        node=c7,
        sources=fee_urls if fee_urls else None,
        additional_instruction="Only consider nps.gov or recreation.gov URLs."
    )

    # 8) Food storage general rule
    fs_urls = filter_official_urls(fs.food_storage_urls)
    c8 = evaluator.add_leaf(
        id="citation_for_food_storage_general_rule",
        desc="Provides at least one official URL supporting the requirement to secure all food/garbage/scented items 24/7",
        parent=node,
        critical=True
    )
    claim8 = "An official page states that all food, garbage, and scented items must be secured at all times (24/7) in ONP wilderness."
    await evaluator.verify(
        claim=claim8,
        node=c8,
        sources=fs_urls if fs_urls else None,
        additional_instruction="Only consider nps.gov or recreation.gov URLs."
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
    model: str = "o4-mini",
) -> Dict:
    """
    Evaluate an answer for the ONP wilderness group camping plan (10 adults, July 25–27, 2026, 2 nights).
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

    # Create a critical wrapper node to mirror rubric root criticality
    overall = evaluator.add_parallel(
        id="overall_plan",
        desc="Answer provides a compliant wilderness camping plan with all requested attributes and supporting official citations",
        parent=root,
        critical=True
    )

    # 1) Extraction
    extracted: TripPlanExtraction = await evaluator.extract(
        prompt=prompt_extract_trip_plan(),
        template_class=TripPlanExtraction,
        extraction_name="trip_plan_extraction"
    )

    # 2) Build and verify tree according to rubric
    await add_camping_area_checks(evaluator, overall, extracted)
    await add_quota_status_check(evaluator, overall, extracted)
    await add_bear_canister_checks(evaluator, overall, extracted)
    await add_campfire_checks(evaluator, overall, extracted)
    await add_permit_system_checks(evaluator, overall, extracted)
    await add_fees_checks(evaluator, overall, extracted)
    await add_food_storage_check(evaluator, overall, extracted)
    await add_citations_checks(evaluator, overall, extracted)

    # 3) Add auxiliary info for transparency
    evaluator.add_custom_info(
        info={
            "assumed_group_size_16_plus": ADULTS_16_PLUS,
            "assumed_nights": NIGHTS,
            "expected_total_fee_usd": EXPECTED_TOTAL_FEE
        },
        info_type="assumptions",
        info_name="computed_expectations"
    )

    return evaluator.get_summary()