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
TASK_ID = "ca_np_campground_rv_accessible_2026"
TASK_DESCRIPTION = (
    "I'm planning a family RV camping trip to California in July 2026 with a group of 8 people. "
    "We have a 40-foot Class A motorhome and will be bringing our dog. Due to health considerations, "
    "we need a campground at lower elevation (below 5,000 feet) with full accessibility features. "
    "Identify a developed campground within a California national park that meets all of the following requirements: "
    "(1) managed by the National Park Service, (2) accepts reservations through Recreation.gov, "
    "(3) can accommodate RVs up to 40 feet in length, (4) allows pets on a 6-foot or shorter leash, "
    "(5) has wheelchair-accessible campsites and restrooms, (6) provides flush toilets, "
    "(7) has potable drinking water available, (8) allows stays of up to 14 consecutive days, "
    "(9) is open and operational in July 2026, (10) can accommodate our group of 8 people in standard campsites, "
    "and (11) is located at an elevation below 5,000 feet. Provide the campground name and a reference URL."
)


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class CampgroundExtraction(BaseModel):
    """
    Extract a single candidate campground (the first one the answer claims meets the requirements).
    """
    campground_name: Optional[str] = None
    park_name: Optional[str] = None  # e.g., "Yosemite National Park"
    state_text: Optional[str] = None  # free-form text if state is mentioned (e.g., "California")
    rec_gov_url: Optional[str] = None  # Recreation.gov link if present
    nps_campground_url: Optional[str] = None  # Official NPS campground page if present
    other_urls: List[str] = Field(default_factory=list)  # any other URLs tied to this campground
    reference_urls: List[str] = Field(default_factory=list)  # all URLs explicitly provided for the selected campground

    # Free-form fields the answer might explicitly mention; used only to aid claim phrasing (verification will still rely on sources)
    rv_max_length_text: Optional[str] = None
    pet_policy_text: Optional[str] = None
    accessibility_text: Optional[str] = None
    toilet_text: Optional[str] = None
    water_text: Optional[str] = None
    max_stay_text: Optional[str] = None
    seasonality_text: Optional[str] = None
    occupancy_text: Optional[str] = None
    elevation_text: Optional[str] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_campground() -> str:
    return """
    Extract exactly one campground (the first one the answer claims meets the specified requirements).
    If multiple campgrounds are listed, choose the first one and extract details only for that one.

    For the selected campground, extract:
    - campground_name: The specific campground name (not just the park name).
    - park_name: The National Park name that the campground is within (e.g., "Yosemite National Park").
    - state_text: If the answer mentions the state (e.g., California), capture it as free text; otherwise null.
    - rec_gov_url: The Recreation.gov page URL for this campground if provided explicitly in the answer; otherwise null.
    - nps_campground_url: The official NPS campground page URL (e.g., on nps.gov) if provided explicitly; otherwise null.
    - other_urls: Any other URLs in the answer specifically associated with the selected campground (exclude duplicates).
    - reference_urls: All URLs the answer provides for the selected campground (including Recreation.gov and NPS links). Deduplicate; preserve order of appearance.

    Also capture any explicit textual statements (verbatim or near-verbatim) provided in the answer related to these constraints:
    - rv_max_length_text: What the answer says about max RV length for this campground (free text).
    - pet_policy_text: Pet rules text (e.g., leash length).
    - accessibility_text: Accessibility details (e.g., accessible sites and restrooms).
    - toilet_text: Toilet facilities (e.g., flush toilets).
    - water_text: Drinking water availability.
    - max_stay_text: Max loop-stay or consecutive stay length.
    - seasonality_text: Open season or note about being open in July 2026 or open year-round.
    - occupancy_text: People per standard (non-group) campsite.
    - elevation_text: Elevation figure/phrase if stated.

    RULES:
    - Only extract information explicitly present in the answer text. Do not invent or infer.
    - For URLs, include only valid URLs explicitly present in the answer (plain, markdown, etc.). If missing protocol, prepend http://.
    - If a field is not present in the answer, set it to null (or empty list for arrays).
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def _unique_preserve_order(seq: List[str]) -> List[str]:
    seen = set()
    out = []
    for s in seq:
        if not s:
            continue
        if s not in seen:
            seen.add(s)
            out.append(s)
    return out


def collect_all_sources(data: CampgroundExtraction) -> List[str]:
    """
    Merge all provided URLs (rec.gov, nps, reference_urls, other_urls) in order, deduplicated.
    """
    urls: List[str] = []
    if data.rec_gov_url:
        urls.append(data.rec_gov_url)
    if data.nps_campground_url:
        urls.append(data.nps_campground_url)
    urls.extend(data.reference_urls or [])
    urls.extend(data.other_urls or [])
    urls = _unique_preserve_order(urls)
    return urls


def pick_rec_gov_url(data: CampgroundExtraction, all_sources: List[str]) -> Optional[str]:
    """
    Prefer the explicitly extracted rec_gov_url. If not present, try to find a recreation.gov URL among all sources.
    """
    if data.rec_gov_url:
        return data.rec_gov_url
    for u in all_sources:
        if "recreation.gov" in (u or "").lower():
            return u
    return None


def pick_nps_url(data: CampgroundExtraction, all_sources: List[str]) -> Optional[str]:
    """
    Prefer the explicitly extracted NPS campground URL. If not present, find an nps.gov URL among all sources.
    """
    if data.nps_campground_url:
        return data.nps_campground_url
    for u in all_sources:
        lu = (u or "").lower()
        if "nps.gov" in lu:
            return u
    return None


# --------------------------------------------------------------------------- #
# Verification logic                                                          #
# --------------------------------------------------------------------------- #
async def verify_campground_requirements(
    evaluator: Evaluator,
    parent_node,
    data: CampgroundExtraction
) -> None:
    """
    Build a critical parallel sub-tree for the selected campground and verify all rubric requirements.
    """
    name_for_claim = data.campground_name or "the selected campground"
    park_for_claim = data.park_name or "a U.S. National Park"

    all_sources = collect_all_sources(data)
    rec_url = pick_rec_gov_url(data, all_sources)
    nps_url = pick_nps_url(data, all_sources)

    # Critical parent node (mirrors rubric root as critical)
    camp_node = evaluator.add_parallel(
        id="campground_requirements",
        desc="Selected campground must satisfy all required constraints",
        parent=parent_node,
        critical=True
    )

    # Existence checks (critical) to gate verification
    evaluator.add_custom_node(
        result=bool(data.campground_name and data.campground_name.strip()),
        id="campground_name_provided",
        desc="The specific name of the campground is provided",
        parent=camp_node,
        critical=True
    )

    evaluator.add_custom_node(
        result=bool(len(all_sources) > 0),
        id="reference_url_provided",
        desc="A valid URL reference supporting the campground information is provided",
        parent=camp_node,
        critical=True
    )

    # Prepare leaves
    claims_and_sources: List[tuple[str, List[str] | str | None, Any, Optional[str]]] = []

    # 1) Location within a California National Park
    node_loc = evaluator.add_leaf(
        id="location_california",
        desc="The campground is located within a national park in California",
        parent=camp_node,
        critical=True
    )
    claim_loc = (
        f"{name_for_claim} is located within {park_for_claim}, which is a United States National Park "
        f"located in California. If the park spans multiple states, the campground itself is in the "
        f"California portion."
    )
    add_ins_loc = (
        "Verify that the campground lies inside the boundaries of an officially designated 'National Park' unit "
        "of the United States National Park Service and that it is in California. Pages that clearly identify "
        "the park and state are acceptable (e.g., nps.gov pages). If the park spans multiple states, it is sufficient "
        "that the campground is in the California portion."
    )
    claims_and_sources.append((claim_loc, [u for u in [nps_url] if u] or all_sources, node_loc, add_ins_loc))

    # 2) NPS managed
    node_nps = evaluator.add_leaf(
        id="nps_managed",
        desc="The campground is managed by the National Park Service",
        parent=camp_node,
        critical=True
    )
    claim_nps = f"{name_for_claim} is managed by the National Park Service (NPS)."
    add_ins_nps = (
        "Look for explicit statements such as 'Operator: National Park Service' on Recreation.gov, "
        "or language on NPS pages indicating NPS management. If managed by a concessioner, this should fail."
    )
    claims_and_sources.append((claim_nps, all_sources, node_nps, add_ins_nps))

    # 3) RV length up to 40 ft
    node_rv = evaluator.add_leaf(
        id="rv_length_40ft",
        desc="The campground can accommodate RVs up to 40 feet in length",
        parent=camp_node,
        critical=True
    )
    claim_rv = f"{name_for_claim} accommodates RVs up to at least 40 feet in length."
    add_ins_rv = (
        "Confirm that the stated maximum RV length is >= 40 ft (or explicitly 40 ft). "
        "Recreation.gov pages commonly list maximum vehicle length per site or campground; it is sufficient if "
        "the campground has standard sites that accommodate 40 ft RVs (not necessarily every site)."
    )
    claims_and_sources.append((claim_rv, [u for u in [rec_url] if u] or all_sources, node_rv, add_ins_rv))

    # 4) Reservations via Recreation.gov
    node_res = evaluator.add_leaf(
        id="reservable_recreation_gov",
        desc="The campground accepts reservations through Recreation.gov",
        parent=camp_node,
        critical=True
    )
    claim_res = f"Reservations for {name_for_claim} are accepted via Recreation.gov."
    add_ins_res = (
        "Confirm that the reservation platform is Recreation.gov (i.e., the reservation page is on the recreation.gov domain). "
        "Links to other systems (e.g., ReserveCalifornia) should not count."
    )
    # Prefer verifying directly against the Recreation.gov URL if available
    res_sources = [rec_url] if rec_url else all_sources
    claims_and_sources.append((claim_res, res_sources, node_res, add_ins_res))

    # 5) Pets allowed on 6-foot leash
    node_pets = evaluator.add_leaf(
        id="pets_allowed_leash",
        desc="The campground allows pets on a 6-foot or shorter leash",
        parent=camp_node,
        critical=True
    )
    claim_pets = f"Pets are allowed at {name_for_claim} when kept on a leash no longer than 6 feet."
    add_ins_pets = (
        "Verify the pet policy for this campground or the park's general pet rules. "
        "An explicit leash length of 6 feet (or equivalent 'six feet') should be present."
    )
    claims_and_sources.append((claim_pets, all_sources, node_pets, add_ins_pets))

    # 6) Wheelchair-accessible campsites and restrooms
    node_access = evaluator.add_leaf(
        id="wheelchair_accessible",
        desc="The campground has wheelchair-accessible campsites and restrooms",
        parent=camp_node,
        critical=True
    )
    claim_access = (
        f"{name_for_claim} offers wheelchair-accessible (ADA) campsites and wheelchair-accessible restrooms."
    )
    add_ins_access = (
        "The evidence should indicate both accessible campsites and accessible restrooms. "
        "Accept synonyms like 'ADA accessible' or 'accessible facilities' when it clearly applies to campsites and restrooms."
    )
    claims_and_sources.append((claim_access, all_sources, node_access, add_ins_access))

    # 7) Flush toilets
    node_flush = evaluator.add_leaf(
        id="flush_toilets_available",
        desc="The campground provides flush toilets",
        parent=camp_node,
        critical=True
    )
    claim_flush = f"{name_for_claim} provides flush toilets."
    add_ins_flush = "Check facilities list or description; 'flush toilets' must be explicitly indicated (not just vault toilets)."
    claims_and_sources.append((claim_flush, all_sources, node_flush, add_ins_flush))

    # 8) Potable drinking water
    node_water = evaluator.add_leaf(
        id="drinking_water_available",
        desc="The campground has potable drinking water available",
        parent=camp_node,
        critical=True
    )
    claim_water = f"Potable drinking water is available at {name_for_claim}."
    add_ins_water = "Look for 'drinking water', 'potable water', or equivalent statements."
    claims_and_sources.append((claim_water, all_sources, node_water, add_ins_water))

    # 9) Up to 14 consecutive days stay
    node_stay = evaluator.add_leaf(
        id="accommodates_14_day_stay",
        desc="The campground allows stays of up to 14 consecutive days",
        parent=camp_node,
        critical=True
    )
    claim_stay = f"The maximum stay limit at {name_for_claim} is at least 14 consecutive days."
    add_ins_stay = (
        "Park/campground rules often specify a 14-day limit. Accept exactly 14 days or any limit >=14 days. "
        "If only shorter limits are shown, fail."
    )
    claims_and_sources.append((claim_stay, all_sources, node_stay, add_ins_stay))

    # 10) Open and operational in July 2026
    node_open = evaluator.add_leaf(
        id="open_july_2026",
        desc="The campground is open and operational during July 2026",
        parent=camp_node,
        critical=True
    )
    claim_open = f"{name_for_claim} is open and operational during July 2026."
    add_ins_open = (
        "Verify using the campground's typical operating season or explicit 'open year-round' statements. "
        "If the official season includes July (e.g., May–September) or it is open year-round, count as supported. "
        "Do not assume special closures unless the page indicates July closures."
    )
    claims_and_sources.append((claim_open, all_sources, node_open, add_ins_open))

    # 11) Group size of 8 people in standard campsites
    node_group = evaluator.add_leaf(
        id="group_size_8_people",
        desc="The campground can accommodate a group of 8 people in standard (non-group) campsites",
        parent=camp_node,
        critical=True
    )
    claim_group = f"A standard, non-group campsite at {name_for_claim} allows up to 8 people."
    add_ins_group = (
        "Verify standard (individual) campsite occupancy/capacity. The requirement is 8 people. "
        "Do not use group campsite rules for this check."
    )
    claims_and_sources.append((claim_group, all_sources, node_group, add_ins_group))

    # 12) Elevation below 5,000 feet
    node_elev = evaluator.add_leaf(
        id="elevation_below_5000ft",
        desc="The campground is located at an elevation below 5,000 feet",
        parent=camp_node,
        critical=True
    )
    claim_elev = f"The elevation of {name_for_claim} is below 5,000 feet above sea level."
    add_ins_elev = (
        "Look for explicit elevation figures for the campground (not just the park). "
        "If the elevation value is shown and is < 5000 ft, count as supported."
    )
    claims_and_sources.append((claim_elev, all_sources, node_elev, add_ins_elev))

    # Execute all verifications in parallel
    await evaluator.batch_verify(claims_and_sources)


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
    Evaluate an answer for the California National Park RV-accessible campground task.
    """
    # Initialize evaluator
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Top-level aggregation across checks (we'll add a critical child node)
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

    # Extract the selected campground info from the answer
    extracted: CampgroundExtraction = await evaluator.extract(
        prompt=prompt_extract_campground(),
        template_class=CampgroundExtraction,
        extraction_name="selected_campground"
    )

    # Create a critical sub-root for campground requirement verification (since framework root is always non-critical)
    critical_root = evaluator.add_parallel(
        id="campground_main",
        desc="Identify at least one developed campground in a California national park that meets all specified requirements",
        parent=root,
        critical=True
    )

    # Build and verify all rubric leaves under the critical node
    await verify_campground_requirements(evaluator, critical_root, extracted)

    # Return full evaluation summary
    return evaluator.get_summary()