import asyncio
import logging
from typing import Any, List, Optional, Dict

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "yellowstone_accessibility_3_areas"
TASK_DESCRIPTION = (
    "A mobility-impaired visitor is planning a summer 2026 trip to Yellowstone National Park and wants to experience "
    "accessible natural features while learning about the park's geology and hydrothermal systems. Identify three "
    "distinct named areas within Yellowstone National Park that each meet ALL of the following requirements:\n\n"
    "1. The area must have a wheelchair-accessible paved trail or boardwalk that is at least 0.5 miles (approximately 800 meters) in length\n"
    "2. The area must have a visitor center with educational exhibits about natural features or geological/hydrothermal features\n"
    "3. The area must provide designated accessible parking at or near the trailhead or main viewing area\n"
    "4. The area must have accessible restroom facilities near the accessible trail\n\n"
    "For each of the three areas, provide:\n"
    "- The specific name of the area within Yellowstone\n"
    "- A reference URL from the National Park Service (nps.gov) that documents the accessibility features or visitor facilities"
)

# Short descriptions from the rubric tree for node texts
DESC_ROOT_GROUP = "Identify three distinct areas within Yellowstone National Park that each meet comprehensive accessibility, visitor facility, and educational program requirements"

AREA_COMPLETE_PROFILE_DESC = {
    1: "First qualifying area with complete accessibility and facility profile",
    2: "Second qualifying area with complete accessibility and facility profile, distinct from Area 1",
    3: "Third qualifying area with complete accessibility and facility profile, distinct from Areas 1 and 2",
}

AREA_BASIC_INFO_DESC = {
    1: "Basic identification and location information for Area 1",
    2: "Basic identification and location information for Area 2",
    3: "Basic identification and location information for Area 3",
}

AREA_NAME_DESC = {
    1: "The area is a named location within Yellowstone National Park",
    2: "The area is a named location within Yellowstone National Park, distinct from Area 1",
    3: "The area is a named location within Yellowstone National Park, distinct from Areas 1 and 2",
}

AREA_REF_URL_DESC = "A valid reference URL from nps.gov is provided to support the identification"

AREA_ACCESS_FEATURES_DESC = "Accessibility features meet requirements for wheelchair users"
AREA_TRAIL_DESC = "Has a wheelchair-accessible paved trail or boardwalk of at least 0.5 miles in length"
AREA_PARKING_DESC = "Provides designated accessible parking at the trailhead or main area"
AREA_RESTROOMS_DESC = "Has accessible restroom facilities near the accessible trail"

AREA_VISITOR_SERVICES_DESC = "Visitor center and educational services are available"
AREA_VC_PRESENCE_DESC = "Has a visitor center in the area"
AREA_EXHIBITS_DESC = "The visitor center features educational exhibits about natural or geological features"


# --------------------------------------------------------------------------- #
# Data models                                                                 #
# --------------------------------------------------------------------------- #
class AreaItem(BaseModel):
    name: Optional[str] = None
    nps_urls: List[str] = Field(default_factory=list)


class AreasExtraction(BaseModel):
    areas: List[AreaItem] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_areas() -> str:
    return """
    Extract up to three distinct named areas within Yellowstone National Park from the answer.
    For each area, extract:
    - name: the specific named location/area within Yellowstone (e.g., "Old Faithful", "West Thumb Geyser Basin", "Mammoth Hot Springs", etc.)
    - nps_urls: an array of one or more URLs from the National Park Service domain (nps.gov) that the answer uses to support this area’s accessibility features or visitor facilities; include only valid nps.gov URLs (e.g., https://www.nps.gov/...)

    Rules:
    - Only include URLs from the nps.gov domain.
    - If the answer lists more than three areas, return the first three mentioned.
    - If an area has no NPS URL mentioned, set nps_urls to an empty array (do not invent URLs).
    - Do not include non-NPS URLs.
    - Keep the area names exactly as given in the answer (minor normalization like trimming spaces is OK).

    Return a JSON with a single field:
    {
      "areas": [
        {"name": "...", "nps_urls": ["...", "..."]},
        {"name": "...", "nps_urls": ["..."]},
        {"name": "...", "nps_urls": []}
      ]
    }
    If fewer than three are present in the answer, return however many are available.
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def is_nps_url(url: str) -> bool:
    if not url:
        return False
    u = url.lower().strip()
    return "nps.gov" in u


def sanitize_area_item(area: AreaItem) -> AreaItem:
    """Ensure only NPS URLs and unique list."""
    urls = []
    seen = set()
    for u in area.nps_urls or []:
        if is_nps_url(u):
            nu = u.strip()
            if nu not in seen:
                urls.append(nu)
                seen.add(nu)
    return AreaItem(name=(area.name or "").strip() or None, nps_urls=urls)


def pad_or_trim_areas(areas: List[AreaItem], target: int = 3) -> List[AreaItem]:
    arr = list(areas[:target])
    while len(arr) < target:
        arr.append(AreaItem())
    return arr


# --------------------------------------------------------------------------- #
# Verification logic                                                          #
# --------------------------------------------------------------------------- #
async def verify_single_area(
    evaluator: Evaluator,
    parent_node,
    area: AreaItem,
    idx: int,
    previous_area_names: List[str],
) -> None:
    """
    Build the verification subtree for one area and run all checks.
    idx is 1-based: 1, 2, 3
    """
    # Construct titles with index
    area_node = evaluator.add_parallel(
        id=f"Area_{idx}_Complete_Profile",
        desc=AREA_COMPLETE_PROFILE_DESC[idx],
        parent=parent_node,
        critical=False,  # allow partial credit at the area level across the 3 areas
    )

    # ------------------ Basic Info (sequential, critical) ------------------ #
    basic_node = evaluator.add_sequential(
        id=f"Area_{idx}_Basic_Info",
        desc=AREA_BASIC_INFO_DESC[idx],
        parent=area_node,
        critical=True,
    )

    # First ensure we have at least one valid NPS URL (critical gate)
    has_valid_nps = any(is_nps_url(u) for u in (area.nps_urls or []))
    evaluator.add_custom_node(
        result=has_valid_nps,
        id=f"Area_{idx}_Reference_URL",
        desc=AREA_REF_URL_DESC,
        parent=basic_node,
        critical=True,
    )

    # Verify the area name against the NPS page(s)
    name_leaf = evaluator.add_leaf(
        id=f"Area_{idx}_Name",
        desc=AREA_NAME_DESC[idx],
        parent=basic_node,
        critical=True,
    )

    area_name = (area.name or "").strip()
    claim_name = (
        f"The provided NPS source page(s) are about the named location '{area_name}' within Yellowstone National Park."
    )
    await evaluator.verify(
        claim=claim_name,
        node=name_leaf,
        sources=area.nps_urls,  # may be multiple URLs
        additional_instruction=(
            "Look for the named place in page titles, headings, or body text. "
            "Allow minor variants (e.g., adding 'area', 'basin', or nearby feature names) if they clearly refer to the same named location in Yellowstone. "
            "If the provided name is empty or not present on the page, this should not pass."
        ),
    )

    # Distinctness checks for Area 2 and Area 3 (critical, under Basic Info)
    if idx >= 2:
        distinct_leaf = evaluator.add_leaf(
            id=f"Area_{idx}_Distinct_From_Previous",
            desc=f"Area {idx} name is distinct from previously identified areas",
            parent=basic_node,
            critical=True,
        )
        prev_list = [n for n in previous_area_names if n]
        claim_distinct = (
            f"The area name '{area_name}' represents a different named location than the previously listed names: {prev_list}."
        )
        await evaluator.verify(
            claim=claim_distinct,
            node=distinct_leaf,
            sources=None,
            additional_instruction=(
                "Judge distinctness logically by names. Treat as the same if they clearly refer to the same named area or sub-area; "
                "ignore superficial wording differences (e.g., 'Area' vs 'Basin') only if they denote the identical place. "
                "If the current name is empty, this should fail."
            ),
        )

    # --------------- Accessibility Features (parallel, critical) ----------- #
    access_node = evaluator.add_parallel(
        id=f"Area_{idx}_Accessibility_Features",
        desc=AREA_ACCESS_FEATURES_DESC,
        parent=area_node,
        critical=True,
    )

    trail_leaf = evaluator.add_leaf(
        id=f"Area_{idx}_Trail_Access",
        desc=AREA_TRAIL_DESC,
        parent=access_node,
        critical=True,
    )
    parking_leaf = evaluator.add_leaf(
        id=f"Area_{idx}_Parking",
        desc=AREA_PARKING_DESC,
        parent=access_node,
        critical=True,
    )
    restrooms_leaf = evaluator.add_leaf(
        id=f"Area_{idx}_Restrooms",
        desc=AREA_RESTROOMS_DESC,
        parent=access_node,
        critical=True,
    )

    claim_trail = (
        f"At {area_name}, there is a wheelchair-accessible paved trail or accessible boardwalk totaling at least 0.5 mile (approx. 0.8 km)."
    )
    claim_parking = (
        f"At {area_name}, designated accessible parking is provided at or near the trailhead or main viewing area."
    )
    claim_restrooms = (
        f"At {area_name}, accessible restroom facilities are available near the accessible trail or main viewing area."
    )

    # --------------- Visitor Services (parallel, critical) ----------------- #
    vc_node = evaluator.add_parallel(
        id=f"Area_{idx}_Visitor_Services",
        desc=AREA_VISITOR_SERVICES_DESC,
        parent=area_node,
        critical=True,
    )

    vc_presence_leaf = evaluator.add_leaf(
        id=f"Area_{idx}_VC_Presence",
        desc=AREA_VC_PRESENCE_DESC,
        parent=vc_node,
        critical=True,
    )
    exhibits_leaf = evaluator.add_leaf(
        id=f"Area_{idx}_Educational_Exhibits",
        desc=AREA_EXHIBITS_DESC,
        parent=vc_node,
        critical=True,
    )

    claim_vc_presence = (
        f"The area {area_name} has a visitor center in the area (e.g., a 'Visitor Center' or 'Visitor Education Center')."
    )
    claim_exhibits = (
        f"The visitor center in {area_name} features educational exhibits or interpretive displays about natural features or geological/hydrothermal systems."
    )

    # Batch verify the feature/VC claims (they auto-depend on Basic Info due to critical sibling + sequential gating)
    await evaluator.batch_verify(
        [
            (
                claim_trail,
                area.nps_urls,
                trail_leaf,
                "Accept phrasing like 'half-mile', '0.5 mi', '0.8 km', or longer distances. "
                "The route must be explicitly described as wheelchair accessible (paved or accessible boardwalk). "
                "If multiple pages are provided, any one page that clearly supports the claim suffices.",
            ),
            (
                claim_parking,
                area.nps_urls,
                parking_leaf,
                "Look for 'accessible parking', 'ADA parking', or similar language indicating designated accessible parking near the relevant trailhead or main viewing area.",
            ),
            (
                claim_restrooms,
                area.nps_urls,
                restrooms_leaf,
                "Look for 'accessible restrooms', 'wheelchair-accessible restrooms', or similar language; "
                "being at the visitor center or parking area near the trail counts as 'near the accessible trail'.",
            ),
            (
                claim_vc_presence,
                area.nps_urls,
                vc_presence_leaf,
                "Verify that a Visitor Center (or Visitor Education Center) exists in this area. "
                "Names may include the area (e.g., 'Old Faithful Visitor Education Center').",
            ),
            (
                claim_exhibits,
                area.nps_urls,
                exhibits_leaf,
                "Verify that the visitor center offers exhibits or interpretive displays about natural features, "
                "geology, hydrothermal systems, geysers, etc. General visitor services alone are insufficient.",
            ),
        ]
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
    Evaluate an answer for identifying three accessible Yellowstone areas with NPS sources.
    """
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Areas evaluated independently
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

    # Extract areas from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_areas(),
        template_class=AreasExtraction,
        extraction_name="extracted_areas",
    )

    # Sanitize extracted data: keep only NPS URLs; pad/trim to exactly 3 areas
    sanitized_areas = [sanitize_area_item(a) for a in (extracted.areas or [])]
    areas3 = pad_or_trim_areas(sanitized_areas, target=3)

    # Add a regrouping node mirroring the rubric top-level (non-critical for partial credit)
    top_group = evaluator.add_parallel(
        id="Yellowstone_Accessible_Areas_Identification",
        desc=DESC_ROOT_GROUP,
        parent=root,
        critical=False,
    )

    # Verify each area
    prev_names: List[str] = []
    for i, area in enumerate(areas3, start=1):
        await verify_single_area(
            evaluator=evaluator,
            parent_node=top_group,
            area=area,
            idx=i,
            previous_area_names=prev_names.copy(),
        )
        if area.name:
            prev_names.append(area.name)

    # Optionally record custom info for debugging
    evaluator.add_custom_info(
        info={
            "areas_extracted": [a.dict() for a in areas3],
            "note": "Only nps.gov URLs were retained for verification.",
        },
        info_type="debug",
        info_name="post_extraction_sanitization",
    )

    return evaluator.get_summary()