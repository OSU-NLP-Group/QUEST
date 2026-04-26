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
TASK_ID = "us_parks_pre1920_lodge_camp_trail_4"
TASK_DESCRIPTION = (
    "Identify 4 distinct United States national parks where each park meets all of the following criteria: "
    "(1) The park was officially designated as a national park before 1920. "
    "(2) The park contains at least one historic lodge that was originally constructed before 1920 and currently "
    "operates with a minimum of 70 guest rooms available for overnight accommodations. "
    "(3) The park has at least one developed campground facility that contains 100 or more individual campsites "
    "and accepts advance reservations through the Recreation.gov system. "
    "(4) The park provides at least one wheelchair-accessible trail that features either paved surfaces or boardwalk "
    "construction and measures less than 1 mile in total length. "
    "For each of the 4 parks you identify, provide the park's official name, its year of national park designation, "
    "the name and construction year of the qualifying historic lodge, the name and campsite count of a qualifying "
    "campground, and the name and length of a qualifying accessible trail."
)


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class EstablishmentInfo(BaseModel):
    designation_year: Optional[str] = None
    designation_source_urls: List[str] = Field(default_factory=list)


class LodgeInfo(BaseModel):
    lodge_name: Optional[str] = None
    construction_year: Optional[str] = None
    current_room_count: Optional[str] = None
    lodge_source_urls: List[str] = Field(default_factory=list)


class CampgroundInfo(BaseModel):
    campground_name: Optional[str] = None
    campsite_count: Optional[str] = None
    recreation_gov_url: Optional[str] = None
    campground_source_urls: List[str] = Field(default_factory=list)


class AccessibleTrailInfo(BaseModel):
    trail_name: Optional[str] = None
    length_miles: Optional[str] = None
    surface_type: Optional[str] = None  # e.g., "paved", "boardwalk"
    accessible_trail_source_urls: List[str] = Field(default_factory=list)


class ParkItem(BaseModel):
    park_name: Optional[str] = None
    establishment: Optional[EstablishmentInfo] = None
    lodge: Optional[LodgeInfo] = None
    campground: Optional[CampgroundInfo] = None
    accessible_trail: Optional[AccessibleTrailInfo] = None
    other_urls: List[str] = Field(default_factory=list)


class ParksExtraction(BaseModel):
    parks: List[ParkItem] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_parks() -> str:
    return """
Extract up to 6 candidate U.S. national parks mentioned in the answer, each with structured fields.
Return JSON with a top-level array "parks". For each park, extract the following fields exactly as written in the answer:

- park_name: string (the official park name as given in the answer)
- establishment: object:
  - designation_year: string (the year the park was officially designated as a U.S. National Park)
  - designation_source_urls: array of strings (URLs explicitly cited in the answer that support the designation year; include NPS, Wikipedia, or other authoritative pages if given)
- lodge: object:
  - lodge_name: string (name of a historic lodge in the park)
  - construction_year: string (the original construction year of the lodge)
  - current_room_count: string (the current number of guest rooms, as stated)
  - lodge_source_urls: array of strings (URLs explicitly cited for the lodge info)
- campground: object:
  - campground_name: string (name of a campground in the park)
  - campsite_count: string (the campground's number of individual campsites, as stated)
  - recreation_gov_url: string or null (the Recreation.gov reservation URL explicitly cited, if any, for this campground)
  - campground_source_urls: array of strings (additional URLs explicitly cited for the campground info)
- accessible_trail: object:
  - trail_name: string (name of a wheelchair-accessible trail)
  - length_miles: string (the trail length as stated in miles or a value convertible to miles, e.g., "0.8 miles" or "0.7 mi")
  - surface_type: string or null (e.g., "paved", "boardwalk", or similar wording if stated)
  - accessible_trail_source_urls: array of strings (URLs explicitly cited for the trail info)
- other_urls: array of strings (any other URLs the answer cites for this park that could be helpful)

Rules:
- Do NOT invent any URLs or values. Only include URLs that are explicitly present in the answer text (plain links or markdown links).
- If any field is missing from the answer for a particular park, set it to null (or empty array for url lists).
- Keep values as strings whenever reasonable (e.g., years, counts, lengths).
- Preserve up to 6 parks; the evaluator will only check the first 4.
    """.strip()


# --------------------------------------------------------------------------- #
# Helpers                                                                     #
# --------------------------------------------------------------------------- #
def _non_empty_str(s: Optional[str]) -> bool:
    return isinstance(s, str) and s.strip() != ""


def _has_urls(urls: Optional[List[str]]) -> bool:
    return bool(urls) and any(_non_empty_str(u) for u in urls or [])


def _combine_urls(*lists: Optional[List[str]]) -> List[str]:
    out: List[str] = []
    seen = set()
    for lst in lists:
        if not lst:
            continue
        for u in lst:
            if not _non_empty_str(u):
                continue
            if u not in seen:
                seen.add(u)
                out.append(u)
    return out


# --------------------------------------------------------------------------- #
# Verification for one park                                                   #
# --------------------------------------------------------------------------- #
async def verify_one_park(evaluator: Evaluator, parent_node, park: ParkItem, park_idx: int) -> None:
    """
    Build verification sub-tree for a single park, following the rubric.
    park_idx is 1-based to match rubric node IDs (park_1..park_4).
    """
    pid = f"park_{park_idx}"
    park_name = park.park_name or ""

    # Top-level node for this park (non-critical to allow partial credit across parks)
    park_node = evaluator.add_parallel(
        id=pid,
        desc=(
            "The {} national park meeting all criteria with all required information provided"
            .format(
                ["first", "second", "third", "fourth"][park_idx - 1]
                if 1 <= park_idx <= 4 else f"#{park_idx}"
            )
        ),
        parent=parent_node,
        critical=False
    )

    # ---------------- Establishment group ----------------
    estab = park.establishment or EstablishmentInfo()
    estab_node = evaluator.add_parallel(
        id=f"{pid}_establishment",
        desc="The park's designation information is complete and meets the time constraint",
        parent=park_node,
        critical=True
    )

    # Provided (require name, year, and at least one establishment source URL to enable grounded checks)
    estab_provided_ok = _non_empty_str(park.park_name) and _non_empty_str(estab.designation_year) and _has_urls(estab.designation_source_urls)
    evaluator.add_custom_node(
        result=estab_provided_ok,
        id=f"{pid}_establishment_provided",
        desc="The park's official name and year of national park designation are provided",
        parent=estab_node,
        critical=True
    )

    # Constraint verification (before 1920, and designated as National Park)
    estab_constraint_leaf = evaluator.add_leaf(
        id=f"{pid}_establishment_constraint",
        desc="The park was officially designated as a national park before 1920",
        parent=estab_node,
        critical=True
    )
    estab_claim = (
        f"The park named '{park_name}' was officially designated as a United States National Park in "
        f"{estab.designation_year}, which is earlier than 1920."
    )
    await evaluator.verify(
        claim=estab_claim,
        node=estab_constraint_leaf,
        sources=estab.designation_source_urls,
        additional_instruction=(
            "Confirm the year the site became a 'National Park' (not just a National Monument or other status). "
            "If the page indicates a later upgrade to National Park status after 1919, this claim is NOT supported. "
            "Allow minor name variants but ensure it's the same park."
        )
    )

    # ---------------- Lodge group ----------------
    lodge = park.lodge or LodgeInfo()
    lodge_node = evaluator.add_parallel(
        id=f"{pid}_lodge",
        desc="The park's historic lodge information is complete and meets all requirements",
        parent=park_node,
        critical=True
    )

    lodge_provided_ok = _non_empty_str(lodge.lodge_name) and _non_empty_str(lodge.construction_year) and _non_empty_str(lodge.current_room_count) and _has_urls(lodge.lodge_source_urls)
    evaluator.add_custom_node(
        result=lodge_provided_ok,
        id=f"{pid}_lodge_provided",
        desc="The name and construction year of the qualifying historic lodge are provided",
        parent=lodge_node,
        critical=True
    )

    lodge_constraint_leaf = evaluator.add_leaf(
        id=f"{pid}_lodge_constraint",
        desc="The park contains a historic lodge that was constructed before 1920 and currently has at least 70 guest rooms available for overnight accommodations",
        parent=lodge_node,
        critical=True
    )
    lodge_claim = (
        f"The lodge '{lodge.lodge_name}' in {park_name} National Park was originally constructed in "
        f"{lodge.construction_year} (before 1920) and currently operates with at least 70 guest rooms "
        f"(e.g., {lodge.current_room_count} rooms)."
    )
    await evaluator.verify(
        claim=lodge_claim,
        node=lodge_constraint_leaf,
        sources=lodge.lodge_source_urls,
        additional_instruction=(
            "Verify both parts: (1) original construction year is before 1920; (2) the lodge CURRENTLY "
            "has at least 70 guest rooms. Accept synonyms like 'guest rooms'/'rooms'/'accommodations'. "
            "Ensure the lodge is inside or directly associated with the specified national park."
        )
    )

    # ---------------- Campground group ----------------
    cg = park.campground or CampgroundInfo()
    cg_node = evaluator.add_parallel(
        id=f"{pid}_campground",
        desc="The park's campground information is complete and meets all requirements",
        parent=park_node,
        critical=True
    )

    cg_provided_ok = _non_empty_str(cg.campground_name) and _non_empty_str(cg.campsite_count) and (_non_empty_str(cg.recreation_gov_url) or _has_urls(cg.campground_source_urls))
    evaluator.add_custom_node(
        result=cg_provided_ok,
        id=f"{pid}_campground_provided",
        desc="The name and campsite count of a qualifying campground are provided",
        parent=cg_node,
        critical=True
    )

    cg_constraint_leaf = evaluator.add_leaf(
        id=f"{pid}_campground_constraint",
        desc="The park has at least one developed campground with 100 or more individual campsites that accepts reservations through Recreation.gov",
        parent=cg_node,
        critical=True
    )
    cg_sources = _combine_urls(cg.campground_source_urls, [cg.recreation_gov_url] if _non_empty_str(cg.recreation_gov_url) else None)
    cg_claim = (
        f"The campground '{cg.campground_name}' in {park_name} National Park has at least 100 individual campsites "
        f"and accepts advance reservations through Recreation.gov."
    )
    await evaluator.verify(
        claim=cg_claim,
        node=cg_constraint_leaf,
        sources=cg_sources,
        additional_instruction=(
            "You can use an NPS page and/or the Recreation.gov page as evidence. "
            "Count should reflect the total number of individual campsites (not group sites unless they are counted "
            "as individual sites). The presence of a valid Recreation.gov page for the campground is sufficient to "
            "confirm it accepts reservations via Recreation.gov."
        )
    )

    # ---------------- Accessible trail group ----------------
    trail = park.accessible_trail or AccessibleTrailInfo()
    trail_node = evaluator.add_parallel(
        id=f"{pid}_accessible_trail",
        desc="The park's accessible trail information is complete and meets all requirements",
        parent=park_node,
        critical=True
    )

    trail_provided_ok = _non_empty_str(trail.trail_name) and _non_empty_str(trail.length_miles) and _has_urls(trail.accessible_trail_source_urls)
    evaluator.add_custom_node(
        result=trail_provided_ok,
        id=f"{pid}_trail_provided",
        desc="The name and length of a qualifying accessible trail are provided",
        parent=trail_node,
        critical=True
    )

    trail_constraint_leaf = evaluator.add_leaf(
        id=f"{pid}_trail_constraint",
        desc="The park has at least one wheelchair-accessible paved or boardwalk trail that is less than 1 mile in total length",
        parent=trail_node,
        critical=True
    )
    surf_txt = trail.surface_type or "paved or boardwalk"
    trail_claim = (
        f"The trail '{trail.trail_name}' in {park_name} National Park is wheelchair-accessible, has a {surf_txt} surface, "
        f"and is less than 1 mile in total length (e.g., {trail.length_miles})."
    )
    await evaluator.verify(
        claim=trail_claim,
        node=trail_constraint_leaf,
        sources=trail.accessible_trail_source_urls,
        additional_instruction=(
            "Confirm the trail is explicitly described as wheelchair accessible (or equivalent language) and that the surface "
            "is paved or boardwalk. Verify that the total length is < 1.0 mile. If the length is given in feet or kilometers, "
            "convert conceptually (e.g., 0.8 mi ≈ 1.3 km; 0.5 mi ≈ 0.8 km; 2640 ft = 0.5 mi)."
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
) -> Dict:
    """
    Evaluate an answer for the '4 pre-1920 national parks with qualifying lodge, campground, accessible trail' task.
    """
    # Initialize evaluator; keep root non-critical to permit partial credit across the 4 parks
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

    # Extraction
    extracted = await evaluator.extract(
        prompt=prompt_extract_parks(),
        template_class=ParksExtraction,
        extraction_name="parks_extraction"
    )

    # Keep only the first 4 parks, padding with empty items if fewer
    parks = list(extracted.parks[:4])
    while len(parks) < 4:
        parks.append(ParkItem())

    # Build verification subtrees for each of the 4 parks
    for idx in range(1, 5):
        await verify_one_park(evaluator, root, parks[idx - 1], idx)

    # Optional: add custom info summary
    evaluator.add_custom_info(
        info={
            "parks_provided_in_answer": len(extracted.parks),
            "parks_evaluated": 4,
            "note": "Only the first 4 parks from the answer are evaluated; missing fields or sources reduce score."
        },
        info_type="evaluation_meta",
        info_name="meta"
    )

    return evaluator.get_summary()