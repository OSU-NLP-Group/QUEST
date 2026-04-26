import asyncio
import logging
from typing import Optional, List, Dict, Any, Tuple

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "four_parks_co_ut_winter_facilities"
TASK_DESCRIPTION = (
    "Identify four National Park Service national parks located in Colorado and Utah that meet the following criteria: "
    "each park must have at least one visitor center with documented winter season operating hours (December 2025 through February 2026), "
    "at least one designated picnic area or picnic facilities, and at least one developed campground. For each of the four parks, provide the following information:\n\n"
    "1. The name of one visitor center and its current winter season operating hours (December 2025 through February 2026)\n"
    "2. The names or locations of at least two picnic areas or picnic facilities within the park\n"
    "3. The name of at least one developed campground and whether it operates during the winter season\n"
    "4. A link to the official NPS.gov page for that park's visitor center information or facilities page\n\n"
    "The four parks should include at least two parks from Colorado and at least two parks from Utah."
)


# --------------------------------------------------------------------------- #
# Data models                                                                 #
# --------------------------------------------------------------------------- #
class ParkFacilityInfo(BaseModel):
    park_name: Optional[str] = None
    state: Optional[str] = None  # expected values like "CO", "Colorado", "UT", "Utah"
    visitor_center_name: Optional[str] = None
    winter_hours_text: Optional[str] = None  # As written in the answer
    picnic_areas: List[str] = Field(default_factory=list)  # At least two names/locations
    campground_name: Optional[str] = None  # A developed campground
    campground_winter_status: Optional[str] = None  # e.g., "open in winter", "closed in winter", "partially open"
    nps_urls: List[str] = Field(default_factory=list)  # One or more NPS.gov URLs used as sources


class ParksExtraction(BaseModel):
    parks: List[ParkFacilityInfo] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_parks() -> str:
    return """
    Extract up to the first six park entries mentioned in the answer that are National Park Service "National Park" units in Colorado or Utah. For each park, extract:

    - park_name: The park's official name as written in the answer (e.g., "Arches National Park").
    - state: The state for the park, as given in the answer if present (e.g., "Utah"/"UT" or "Colorado"/"CO"). If missing in the answer, leave null.
    - visitor_center_name: The name of one specific visitor center (e.g., "Canyon Visitor Center", "Ben Reifel Visitor Center"). If missing, null.
    - winter_hours_text: The current winter season operating hours as stated in the answer for the period covering December 2025 through February 2026. Use the exact phrasing from the answer (e.g., "Open daily 9 am – 4 pm (Dec–Feb)"). If not provided, null.
    - picnic_areas: An array with the names or locations of at least two picnic areas/facilities within the park (e.g., ["Devils Garden Picnic Area", "Panorama Point Picnic Area"]). If fewer are provided, include whatever is mentioned; if none, empty array.
    - campground_name: The name of at least one developed campground (not strictly primitive or backcountry). If missing, null.
    - campground_winter_status: The winter operating status for that campground as provided in the answer (e.g., "open year-round", "closed in winter", "limited winter camping"). If not stated, null.
    - nps_urls: An array of all official NPS.gov URLs that the answer cites for this park’s visitor center or facilities information. Include only valid URLs; if none given, return an empty list.

    Rules:
    - Do not invent any values; only extract what is explicitly present in the answer.
    - If a field is missing for a park, set it to null (or empty array for picnic_areas/nps_urls).
    - Return a JSON object with a "parks" array.
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _normalize_state(s: Optional[str]) -> Optional[str]:
    if not s:
        return None
    t = s.strip().lower()
    if t in {"co", "colorado"}:
        return "CO"
    if t in {"ut", "utah"}:
        return "UT"
    return None


def _canonical_state_name(s: Optional[str]) -> str:
    if s is None:
        return "Colorado or Utah"
    if s.upper() == "CO":
        return "Colorado"
    if s.upper() == "UT":
        return "Utah"
    # If free text provided, keep it as-is for human-readable claims
    return s


def _normalize_name(name: Optional[str]) -> Optional[str]:
    if not name:
        return None
    return " ".join(name.strip().lower().split())


def _filter_nps_urls(urls: List[str]) -> List[str]:
    return [u for u in urls if isinstance(u, str) and "nps.gov" in u.lower()]


def _select_first_four_distinct(parks: List[ParkFacilityInfo]) -> List[ParkFacilityInfo]:
    seen = set()
    result: List[ParkFacilityInfo] = []
    for p in parks:
        key = _normalize_name(p.park_name)
        if key and key not in seen:
            seen.add(key)
            result.append(p)
        if len(result) == 4:
            break
    # pad to 4 if needed with empty placeholders
    while len(result) < 4:
        result.append(ParkFacilityInfo())
    return result


def _count_states(selected: List[ParkFacilityInfo]) -> Tuple[int, int]:
    co = 0
    ut = 0
    for p in selected:
        st = _normalize_state(p.state)
        if st == "CO":
            co += 1
        elif st == "UT":
            ut += 1
    return co, ut


# --------------------------------------------------------------------------- #
# Verification subroutine per park                                            #
# --------------------------------------------------------------------------- #
async def verify_park(
    evaluator: Evaluator,
    parent_node,
    park: ParkFacilityInfo,
    park_index: int,
) -> None:
    """
    Build the ParkX subtree with individual leaves and verifications.
    """
    park_label = f"Park{park_index + 1}"

    # Parent (`main_node`) is critical, so park-level children must also be critical.
    park_node = evaluator.add_parallel(
        id=park_label,
        desc=f"Evaluation of the {['first','second','third','fourth'][park_index] if park_index < 4 else f'#{park_index+1}'} park's eligibility and required facility details.",
        parent=parent_node,
        critical=True
    )

    # 1) Park name provided (existence)
    name_provided = bool(park.park_name and park.park_name.strip())
    evaluator.add_custom_node(
        result=name_provided,
        id=f"{park_label}_Park_Name_Provided",
        desc="The park name is provided.",
        parent=park_node,
        critical=True
    )

    # 2) Park is NPS National Park in CO or UT (verify by NPS URL(s) if provided)
    state_norm = _normalize_state(park.state)
    state_for_claim = _canonical_state_name(state_norm)
    nps_sources = _filter_nps_urls(park.nps_urls)

    node_is_nps_np = evaluator.add_leaf(
        id=f"{park_label}_Is_NPS_National_Park_In_CO_Or_UT",
        desc="The park is an NPS-designated National Park and is located in Colorado or Utah.",
        parent=park_node,
        critical=True
    )
    claim_is_nps_np = (
        f"{park.park_name or 'This park'} is an official U.S. National Park managed by the National Park Service "
        f"and is located in {state_for_claim}."
    )
    await evaluator.verify(
        claim=claim_is_nps_np,
        node=node_is_nps_np,
        sources=nps_sources if nps_sources else None,
        additional_instruction=(
            "Use the content of the provided NPS page(s) to confirm that the unit is designated a 'National Park' "
            "and is located in Colorado or Utah. Accept if the page header/title or description clearly indicates "
            "it is an NPS National Park in the specified state."
        )
    )

    # 3) Visitor center name provided (existence)
    vc_provided = bool(park.visitor_center_name and park.visitor_center_name.strip())
    evaluator.add_custom_node(
        result=vc_provided,
        id=f"{park_label}_Visitor_Center_Name_Provided",
        desc="At least one specific visitor center name is provided.",
        parent=park_node,
        critical=True
    )

    # 4) Winter hours documented for Dec 2025 – Feb 2026 (verify via NPS URLs)
    node_hours = evaluator.add_leaf(
        id=f"{park_label}_Winter_Hours_Dec2025_Feb2026_Documented",
        desc="Visitor center winter season operating hours are provided and are documented for the period Dec 2025 through Feb 2026.",
        parent=park_node,
        critical=True
    )
    vc_name = park.visitor_center_name or "the visitor center"
    hours_text = park.winter_hours_text or ""
    claim_hours = (
        f"The visitor center '{vc_name}' at {park.park_name or 'the park'} has winter-season operating hours that "
        f"cover December 2025, January 2026, and February 2026. The hours (as stated) are: {hours_text}"
    )
    await evaluator.verify(
        claim=claim_hours,
        node=node_hours,
        sources=nps_sources if nps_sources else None,
        additional_instruction=(
            "Check the NPS page(s) for operating hours information. Confirm that winter or off-season hours explicitly "
            "cover the months of December 2025, January 2026, and February 2026. Accept generic 'winter' season schedules "
            "that clearly include those months, or year-round hours that apply during those months."
        )
    )

    # 5) At least two picnic areas/facilities provided (verify via NPS URLs)
    node_picnic = evaluator.add_leaf(
        id=f"{park_label}_Two_Picnic_Areas_Provided",
        desc="Names or locations of at least two designated picnic areas or picnic facilities within the park are provided.",
        parent=park_node,
        critical=True
    )
    picnic_list_for_claim = ", ".join(park.picnic_areas) if park.picnic_areas else "none listed"
    claim_picnic = (
        f"Within {park.park_name or 'the park'}, at least two designated picnic areas or picnic facilities exist as listed: "
        f"{picnic_list_for_claim}. These are recognized by the park."
    )
    await evaluator.verify(
        claim=claim_picnic,
        node=node_picnic,
        sources=nps_sources if nps_sources else None,
        additional_instruction=(
            "Use the NPS page(s) to confirm there are at least two distinct picnic areas or picnic facilities in this park. "
            "Synonyms such as 'picnic area', 'day-use area with picnic tables', or named picnic sites are acceptable. "
            "They must be within the park."
        )
    )

    # 6) Developed campground and winter status (verify via NPS URLs)
    node_camp = evaluator.add_leaf(
        id=f"{park_label}_Developed_Campground_And_Winter_Status",
        desc="At least one developed campground (not primitive/backcountry-only) is named and its winter operating status is specified.",
        parent=park_node,
        critical=True
    )
    cg_name = park.campground_name or "the campground"
    cg_status = park.campground_winter_status or "unspecified"
    claim_camp = (
        f"The developed campground '{cg_name}' exists within {park.park_name or 'the park'}, and it is {cg_status} "
        f"during the winter period (December 2025 through February 2026)."
    )
    await evaluator.verify(
        claim=claim_camp,
        node=node_camp,
        sources=nps_sources if nps_sources else None,
        additional_instruction=(
            "Confirm from the NPS page(s) that the named campground is a developed front-country campground (not only primitive/backcountry), "
            "and determine whether it is open, closed, or partially open during December 2025 through February 2026."
        )
    )

    # 7) Official NPS URL provided and relevant (verify via provided URLs)
    node_nps_url = evaluator.add_leaf(
        id=f"{park_label}_Official_NPS_URL_Provided_And_Relevant",
        desc="A valid NPS.gov URL is provided that is an official visitor center information or facilities page for the park (usable to verify the claims).",
        parent=park_node,
        critical=True
    )
    claim_nps_url = (
        f"At least one of the provided webpages is an official NPS (nps.gov) page for {park.park_name or 'the park'} "
        f"that presents visitor center or facilities information relevant for verification."
    )
    await evaluator.verify(
        claim=claim_nps_url,
        node=node_nps_url,
        sources=nps_sources if nps_sources else None,
        additional_instruction=(
            "Verify that at least one URL is on the nps.gov domain and contains official visitor center or park facilities information "
            "for this specific park (e.g., hours, services, picnic, camping). If no URLs are provided in the answer, this should fail."
        )
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
) -> Dict[str, Any]:
    """
    Evaluate an answer for the four CO/UT National Parks with winter facilities task.
    """
    # Initialize evaluator with a neutral root; we'll create the critical top-level node beneath it.
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

    # Extract structured info from the answer
    extracted: ParksExtraction = await evaluator.extract(
        prompt=prompt_extract_parks(),
        template_class=ParksExtraction,
        extraction_name="parks_extraction"
    )

    # Select the first four distinct parks (by normalized park name), padding if fewer
    selected_parks = _select_first_four_distinct(extracted.parks)

    # Record helpful custom info for debugging
    all_park_names = [p.park_name for p in extracted.parks if p.park_name]
    selected_park_names = [p.park_name for p in selected_parks if p.park_name]
    distinct_total = len(set(_normalize_name(n) for n in all_park_names))
    co_count, ut_count = _count_states(selected_parks)
    evaluator.add_custom_info(
        info={
            "all_parks_extracted": all_park_names,
            "selected_parks": selected_park_names,
            "distinct_parks_in_answer": distinct_total,
            "selected_state_counts": {"CO": co_count, "UT": ut_count}
        },
        info_type="diagnostic",
        info_name="extraction_summary"
    )

    # Build top-level critical node matching rubric
    main_node = evaluator.add_parallel(
        id="Four_Parks_Colorado_Utah_Facilities",
        desc="Evaluate whether the response identifies four distinct NPS national parks in CO/UT and provides required winter visitor center hours, picnic areas, campground info, and official NPS.gov references for each.",
        parent=root,
        critical=True
    )

    # Global constraints node (critical)
    global_node = evaluator.add_parallel(
        id="Global_Requirements",
        desc="Global constraints across the full set of four parks.",
        parent=main_node,
        critical=True
    )

    # Global leaf: Exactly four distinct parks provided (interpretation: at least four distinct, per evaluation guidance)
    enough_distinct = distinct_total >= 4
    evaluator.add_custom_node(
        result=enough_distinct,
        id="Exactly_Four_Distinct_Parks_Provided",
        desc="The response identifies four distinct parks (no duplicates) as the four required items.",
        parent=global_node,
        critical=True
    )

    # Global leaf: State distribution (at least two CO and at least two UT among selected four)
    state_ok = (co_count >= 2 and ut_count >= 2)
    evaluator.add_custom_node(
        result=state_ok,
        id="State_Distribution_At_Least_Two_Each",
        desc="Among the four parks, at least two are in Colorado and at least two are in Utah.",
        parent=global_node,
        critical=True
    )

    # Park-specific verification for 4 parks
    for idx in range(4):
        await verify_park(evaluator, main_node, selected_parks[idx], idx)

    # Return summary
    return evaluator.get_summary()
