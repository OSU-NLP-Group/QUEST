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
TASK_ID = "pa_state_parks_group_events"
TASK_DESCRIPTION = """Identify three Pennsylvania state parks that each meet all of the following comprehensive facility requirements for hosting large outdoor group events:

1. The park must be at least 5,000 acres in size
2. The park must have a swimming beach that operates during the summer season (approximately Memorial Day through Labor Day)
3. The park must have boat launch or marina facilities
4. The park must have designated fishing facilities (such as fishing pier, lake access, or fishing areas)
5. The park must have at least 10 miles of hiking trails
6. The park must have at least one reservable picnic shelter that accommodates a minimum of 50 people
7. The park must offer camping facilities (tent and/or RV camping sites)
8. The park must have a visitor center with documented operating hours

For each park, provide:
- The official name of the state park
- The park's acreage
- Specific details about each required facility
- Official Pennsylvania DCNR or state park website URLs documenting each facility and its specifications
"""


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class BeachInfo(BaseModel):
    description: Optional[str] = None
    season: Optional[str] = None
    urls: List[str] = Field(default_factory=list)


class BoatInfo(BaseModel):
    description: Optional[str] = None
    urls: List[str] = Field(default_factory=list)


class FishingInfo(BaseModel):
    description: Optional[str] = None
    urls: List[str] = Field(default_factory=list)


class HikingInfo(BaseModel):
    total_miles: Optional[str] = None
    urls: List[str] = Field(default_factory=list)


class PicnicInfo(BaseModel):
    description: Optional[str] = None
    capacity: Optional[str] = None  # e.g., "50", "up to 100", "50-80"
    urls: List[str] = Field(default_factory=list)


class CampingInfo(BaseModel):
    description: Optional[str] = None
    urls: List[str] = Field(default_factory=list)


class VisitorCenterInfo(BaseModel):
    description: Optional[str] = None
    hours: Optional[str] = None
    urls: List[str] = Field(default_factory=list)


class ParkItem(BaseModel):
    name: Optional[str] = None
    acreage: Optional[str] = None  # Keep as string to maximize robustness
    main_urls: List[str] = Field(default_factory=list)  # Official park/DCNR page(s)
    beach: Optional[BeachInfo] = None
    boat: Optional[BoatInfo] = None
    fishing: Optional[FishingInfo] = None
    hiking: Optional[HikingInfo] = None
    picnic: Optional[PicnicInfo] = None
    camping: Optional[CampingInfo] = None
    visitor_center: Optional[VisitorCenterInfo] = None


class ParksExtraction(BaseModel):
    parks: List[ParkItem] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_parks() -> str:
    return """
Extract up to three Pennsylvania state parks presented in the answer that purportedly meet ALL of the following requirements:
1) At least 5,000 acres in size
2) Swimming beach operating during the summer season (approx. Memorial Day–Labor Day)
3) Boat launch or marina facilities
4) Designated fishing facilities (fishing pier, lake access, or designated areas)
5) At least 10 miles of hiking trails
6) At least one reservable picnic shelter accommodating 50+ people
7) Camping facilities (tent and/or RV)
8) Visitor center with documented operating hours

For each park, return a JSON with:
- name: Official park name as written in the answer
- acreage: The acreage value or phrase (string as written)
- main_urls: A list of official Pennsylvania DCNR or state-run webpage URLs for the park (e.g., dcnr.pa.gov or other *.pa.gov or state.pa.us domains) explicitly cited in the answer

And a nested object for each facility with details and URLs, using these exact keys:
- beach: { description, season, urls }
- boat: { description, urls }
- fishing: { description, urls }
- hiking: { total_miles, urls }
- picnic: { description, capacity, urls }
- camping: { description, urls }
- visitor_center: { description, hours, urls }

Rules:
- Extract only what is explicitly in the answer. Do not invent any content or URLs.
- For all URL lists, include only URLs explicitly cited in the answer text. If none are provided for a facility, return an empty list.
- Use strings for numbers (e.g., acreage '5,286 acres', trails '17 miles').
- Prefer official Pennsylvania DCNR/state pages (dcnr.pa.gov, *.pa.gov, state.pa.us). If the answer only cites non-official pages for a facility, still extract them, but do not add any URL not present in the answer.
- If a field is not present in the answer, set it to null (for scalars) or [] (for lists).

Return a JSON object: { "parks": [ ParkItem, ParkItem, ParkItem ] }.
If fewer than three parks are present, include as many as are found.
"""


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def _dedup(urls: List[str]) -> List[str]:
    seen = set()
    out = []
    for u in urls:
        if not u:
            continue
        nu = u.strip()
        if nu and nu not in seen:
            seen.add(nu)
            out.append(nu)
    return out


def collect_all_urls_for_park(park: ParkItem) -> List[str]:
    urls: List[str] = []
    urls.extend(park.main_urls or [])
    if park.beach:
        urls.extend(park.beach.urls or [])
    if park.boat:
        urls.extend(park.boat.urls or [])
    if park.fishing:
        urls.extend(park.fishing.urls or [])
    if park.hiking:
        urls.extend(park.hiking.urls or [])
    if park.picnic:
        urls.extend(park.picnic.urls or [])
    if park.camping:
        urls.extend(park.camping.urls or [])
    if park.visitor_center:
        urls.extend(park.visitor_center.urls or [])
    return _dedup(urls)


async def verify_with_sources(
    evaluator: Evaluator,
    *,
    claim: str,
    node,
    urls: Optional[List[str]],
    additional_instruction: str
) -> bool:
    """
    Verify a claim requiring web evidence. If no URLs are provided, mark as failed to enforce source-grounding.
    """
    if urls and len(urls) > 0:
        return await evaluator.verify(
            claim=claim,
            node=node,
            sources=urls,
            additional_instruction=additional_instruction
        )
    else:
        node.score = 0.0
        node.status = "failed"
        return False


def official_pa_instruction(extra: str = "") -> str:
    base = (
        "Only consider the claim supported if the webpage is an official Pennsylvania DCNR or state-run page "
        "(e.g., dcnr.pa.gov, *.pa.gov, state.pa.us). If the URL is not an official PA/DCNR page, mark as not supported. "
    )
    return base + (extra or "")


# --------------------------------------------------------------------------- #
# Verification per-park                                                       #
# --------------------------------------------------------------------------- #
async def verify_single_park(
    evaluator: Evaluator,
    parent_node,
    park: ParkItem,
    park_index_one_based: int
) -> None:
    """
    Build and verify the subtree for a single park.
    All concrete checks are leaves; grouping follows the rubric.
    """
    idx = park_index_one_based
    park_node = evaluator.add_parallel(
        id=f"park_{idx}",
        desc=(
            "First Pennsylvania state park meeting all specified criteria"
            if idx == 1 else
            ("Second Pennsylvania state park meeting all specified criteria" if idx == 2
             else "Third Pennsylvania state park meeting all specified criteria")
        ),
        parent=parent_node,
        critical=False  # allow partial credit across different parks
    )

    park_name = park.name or "the park"
    all_urls = collect_all_urls_for_park(park)
    main_or_all_urls = _dedup((park.main_urls or []) + all_urls)

    # 1) Identification (leaf)
    ident_leaf = evaluator.add_leaf(
        id=f"park_{idx}_identification",
        desc="Park is officially designated as a Pennsylvania state park managed by DCNR",
        parent=park_node,
        critical=True
    )
    ident_claim = f"{park_name} is an official Pennsylvania state park managed by the Pennsylvania DCNR."
    await verify_with_sources(
        evaluator,
        claim=ident_claim,
        node=ident_leaf,
        urls=park.main_urls if (park.main_urls and len(park.main_urls) > 0) else main_or_all_urls,
        additional_instruction=official_pa_instruction(
            "Look for language on the official park page indicating it is a Pennsylvania state park managed by DCNR."
        )
    )

    # 2) Size (leaf)
    size_leaf = evaluator.add_leaf(
        id=f"park_{idx}_size",
        desc="Park encompasses at least 5,000 acres",
        parent=park_node,
        critical=True
    )
    acreage_text = park.acreage or ""
    size_claim = (
        f"The official documentation indicates that {park_name} has an area of at least 5,000 acres. "
        f"The reported acreage in the answer is: '{acreage_text}'."
    )
    await verify_with_sources(
        evaluator,
        claim=size_claim,
        node=size_leaf,
        urls=main_or_all_urls,
        additional_instruction=official_pa_instruction(
            "Confirm the acreage from the official page. Consider numeric interpretation and allow comma formatting. "
            "If multiple acreages appear, use the one that applies to the park as a whole. "
            "Pass if the acreage is >= 5,000."
        )
    )

    # 3) Water recreation (parallel group): beach, boating, fishing
    water_node = evaluator.add_parallel(
        id=f"park_{idx}_water_recreation",
        desc="Park provides water recreation facilities",
        parent=park_node,
        critical=True
    )

    # 3.a) Beach (sequential): existence -> season -> reference
    beach_node = evaluator.add_sequential(
        id=f"park_{idx}_beach",
        desc="Park has a swimming beach that operates during summer season",
        parent=water_node,
        critical=True
    )
    # Existence
    beach_exist_leaf = evaluator.add_leaf(
        id=f"park_{idx}_beach_existence",
        desc="Swimming beach facility exists",
        parent=beach_node,
        critical=True
    )
    beach_urls = park.beach.urls if (park.beach and park.beach.urls) else []
    beach_exist_claim = f"{park_name} has a public swimming beach."
    await verify_with_sources(
        evaluator,
        claim=beach_exist_claim,
        node=beach_exist_leaf,
        urls=beach_urls if beach_urls else main_or_all_urls,
        additional_instruction=official_pa_instruction(
            "The page should explicitly indicate a swimming beach at the park."
        )
    )
    # Season
    beach_season_leaf = evaluator.add_leaf(
        id=f"park_{idx}_beach_season",
        desc="Beach operates during summer months (approximately Memorial Day to Labor Day)",
        parent=beach_node,
        critical=True
    )
    beach_season_claim = (
        f"The swimming beach at {park_name} operates during the summer season, approximately Memorial Day through Labor Day."
    )
    await verify_with_sources(
        evaluator,
        claim=beach_season_claim,
        node=beach_season_leaf,
        urls=beach_urls if beach_urls else main_or_all_urls,
        additional_instruction=official_pa_instruction(
            "Allow phrasing variations like 'late May to early September'. The timeframe should clearly correspond to "
            "the US summer season window around Memorial Day to Labor Day."
        )
    )
    # Reference
    beach_ref_leaf = evaluator.add_leaf(
        id=f"park_{idx}_beach_reference",
        desc="Official source URL documenting beach facility and operating season",
        parent=beach_node,
        critical=True
    )
    beach_ref_claim = (
        f"At least one provided official Pennsylvania page documents the swimming beach at {park_name} and its summer operating season."
    )
    await verify_with_sources(
        evaluator,
        claim=beach_ref_claim,
        node=beach_ref_leaf,
        urls=beach_urls if beach_urls else main_or_all_urls,
        additional_instruction=official_pa_instruction(
            "The same page (or any single official page) should explicitly mention both the beach facility and its season/timing."
        )
    )

    # 3.b) Boat launch / marina (sequential): existence -> reference
    boat_node = evaluator.add_sequential(
        id=f"park_{idx}_boat_launch",
        desc="Park has boat launch or marina facilities",
        parent=water_node,
        critical=True
    )
    boat_exist_leaf = evaluator.add_leaf(
        id=f"park_{idx}_boat_existence",
        desc="Boat launch or marina facility exists",
        parent=boat_node,
        critical=True
    )
    boat_urls = park.boat.urls if (park.boat and park.boat.urls) else []
    boat_exist_claim = f"{park_name} has a boat launch or marina facility."
    await verify_with_sources(
        evaluator,
        claim=boat_exist_claim,
        node=boat_exist_leaf,
        urls=boat_urls if boat_urls else main_or_all_urls,
        additional_instruction=official_pa_instruction(
            "The page should clearly indicate a boat launch and/or marina accessible at the park."
        )
    )
    boat_ref_leaf = evaluator.add_leaf(
        id=f"park_{idx}_boat_reference",
        desc="Official source URL documenting boat launch or marina",
        parent=boat_node,
        critical=True
    )
    boat_ref_claim = f"An official Pennsylvania page documents the boat launch or marina at {park_name}."
    await verify_with_sources(
        evaluator,
        claim=boat_ref_claim,
        node=boat_ref_leaf,
        urls=boat_urls if boat_urls else main_or_all_urls,
        additional_instruction=official_pa_instruction(
            "The page should explicitly mention 'boat launch', 'ramp', 'marina', or equivalent facility."
        )
    )

    # 3.c) Fishing (sequential): existence -> reference
    fishing_node = evaluator.add_sequential(
        id=f"park_{idx}_fishing",
        desc="Park has designated fishing facilities",
        parent=water_node,
        critical=True
    )
    fish_exist_leaf = evaluator.add_leaf(
        id=f"park_{idx}_fishing_existence",
        desc="Fishing pier, lake access, or designated fishing area exists",
        parent=fishing_node,
        critical=True
    )
    fishing_urls = park.fishing.urls if (park.fishing and park.fishing.urls) else []
    fish_exist_claim = f"{park_name} provides designated fishing opportunities (e.g., pier, lake access, or fishing areas)."
    await verify_with_sources(
        evaluator,
        claim=fish_exist_claim,
        node=fish_exist_leaf,
        urls=fishing_urls if fishing_urls else main_or_all_urls,
        additional_instruction=official_pa_instruction(
            "The page should explicitly mention fishing facilities or access."
        )
    )
    fish_ref_leaf = evaluator.add_leaf(
        id=f"park_{idx}_fishing_reference",
        desc="Official source URL documenting fishing facilities",
        parent=fishing_node,
        critical=True
    )
    fish_ref_claim = f"An official Pennsylvania page documents designated fishing facilities at {park_name}."
    await verify_with_sources(
        evaluator,
        claim=fish_ref_claim,
        node=fish_ref_leaf,
        urls=fishing_urls if fishing_urls else main_or_all_urls,
        additional_instruction=official_pa_instruction(
            "It should mention 'fishing', 'pier', 'designated area', 'lake access', or similar."
        )
    )

    # 4) Land recreation (parallel): hiking + picnic
    land_node = evaluator.add_parallel(
        id=f"park_{idx}_land_recreation",
        desc="Park provides land-based recreation facilities",
        parent=park_node,
        critical=True
    )

    # 4.a) Hiking (sequential): length -> reference
    hiking_node = evaluator.add_sequential(
        id=f"park_{idx}_hiking",
        desc="Park has at least 10 miles of hiking trails",
        parent=land_node,
        critical=True
    )
    hiking_len_leaf = evaluator.add_leaf(
        id=f"park_{idx}_hiking_length",
        desc="Total hiking trail length is at least 10 miles",
        parent=hiking_node,
        critical=True
    )
    hiking_urls = park.hiking.urls if (park.hiking and park.hiking.urls) else []
    hiking_len_claim = f"{park_name} provides a trail system totaling at least 10 miles."
    await verify_with_sources(
        evaluator,
        claim=hiking_len_claim,
        node=hiking_len_leaf,
        urls=hiking_urls if hiking_urls else main_or_all_urls,
        additional_instruction=official_pa_instruction(
            "If units are in kilometers, convert mentally (>= 16 km ≈ 10 miles). "
            "If multiple values are shown, use the total hiking trail mileage at the park."
        )
    )
    hiking_ref_leaf = evaluator.add_leaf(
        id=f"park_{idx}_hiking_reference",
        desc="Official source URL documenting trail lengths",
        parent=hiking_node,
        critical=True
    )
    hiking_ref_claim = f"An official Pennsylvania page documents the total hiking trail mileage for {park_name}."
    await verify_with_sources(
        evaluator,
        claim=hiking_ref_claim,
        node=hiking_ref_leaf,
        urls=hiking_urls if hiking_urls else main_or_all_urls,
        additional_instruction=official_pa_instruction(
            "The page should explicitly state total miles (or km) of hiking trails."
        )
    )

    # 4.b) Picnic shelters (sequential): existence -> capacity -> reference
    picnic_node = evaluator.add_sequential(
        id=f"park_{idx}_picnic",
        desc="Park has reservable picnic shelter accommodating at least 50 people",
        parent=land_node,
        critical=True
    )
    picnic_exist_leaf = evaluator.add_leaf(
        id=f"park_{idx}_picnic_existence",
        desc="Reservable picnic shelter exists",
        parent=picnic_node,
        critical=True
    )
    picnic_urls = park.picnic.urls if (park.picnic and park.picnic.urls) else []
    picnic_exist_claim = f"{park_name} has at least one reservable picnic shelter."
    await verify_with_sources(
        evaluator,
        claim=picnic_exist_claim,
        node=picnic_exist_leaf,
        urls=picnic_urls if picnic_urls else main_or_all_urls,
        additional_instruction=official_pa_instruction(
            "Look for terms like 'picnic pavilion', 'shelter', and indication that it is reservable."
        )
    )
    picnic_capacity_leaf = evaluator.add_leaf(
        id=f"park_{idx}_picnic_capacity",
        desc="At least one shelter accommodates 50 or more people",
        parent=picnic_node,
        critical=True
    )
    capacity_text = park.picnic.capacity if (park.picnic and park.picnic.capacity) else ""
    picnic_capacity_claim = (
        f"At least one picnic shelter at {park_name} accommodates 50 or more people. "
        f"The capacity mentioned in the answer: '{capacity_text}'."
    )
    await verify_with_sources(
        evaluator,
        claim=picnic_capacity_claim,
        node=picnic_capacity_leaf,
        urls=picnic_urls if picnic_urls else main_or_all_urls,
        additional_instruction=official_pa_instruction(
            "The page should state a numeric capacity of >= 50 for at least one shelter."
        )
    )
    picnic_ref_leaf = evaluator.add_leaf(
        id=f"park_{idx}_picnic_reference",
        desc="Official source URL documenting picnic shelter capacity",
        parent=picnic_node,
        critical=True
    )
    picnic_ref_claim = f"An official Pennsylvania page documents picnic shelter capacity for {park_name}."
    await verify_with_sources(
        evaluator,
        claim=picnic_ref_claim,
        node=picnic_ref_leaf,
        urls=picnic_urls if picnic_urls else main_or_all_urls,
        additional_instruction=official_pa_instruction(
            "The page should include capacity details for picnic shelters."
        )
    )

    # 5) Accommodation (sequential): existence -> reference
    accom_node = evaluator.add_sequential(
        id=f"park_{idx}_accommodation",
        desc="Park provides overnight camping facilities",
        parent=park_node,
        critical=True
    )
    camping_exist_leaf = evaluator.add_leaf(
        id=f"park_{idx}_camping_existence",
        desc="Park offers tent and/or RV camping sites",
        parent=accom_node,
        critical=True
    )
    camping_urls = park.camping.urls if (park.camping and park.camping.urls) else []
    camping_exist_claim = f"{park_name} offers camping facilities (tent and/or RV)."
    await verify_with_sources(
        evaluator,
        claim=camping_exist_claim,
        node=camping_exist_leaf,
        urls=camping_urls if camping_urls else main_or_all_urls,
        additional_instruction=official_pa_instruction(
            "The page should explicitly mention camping availability at the park."
        )
    )
    camping_ref_leaf = evaluator.add_leaf(
        id=f"park_{idx}_camping_reference",
        desc="Official source URL documenting camping facilities",
        parent=accom_node,
        critical=True
    )
    camping_ref_claim = f"An official Pennsylvania page documents camping facilities at {park_name}."
    await verify_with_sources(
        evaluator,
        claim=camping_ref_claim,
        node=camping_ref_leaf,
        urls=camping_urls if camping_urls else main_or_all_urls,
        additional_instruction=official_pa_instruction(
            "The page should mention types of camping (e.g., tent, RV, sites)."
        )
    )

    # 6) Visitor services (sequential): existence -> hours -> reference
    visitor_node = evaluator.add_sequential(
        id=f"park_{idx}_visitor_services",
        desc="Park has visitor center with documented operating hours",
        parent=park_node,
        critical=True
    )
    vc_exist_leaf = evaluator.add_leaf(
        id=f"park_{idx}_visitor_center_existence",
        desc="Visitor center facility exists",
        parent=visitor_node,
        critical=True
    )
    visitor_urls = park.visitor_center.urls if (park.visitor_center and park.visitor_center.urls) else []
    vc_exist_claim = f"{park_name} has a visitor center."
    await verify_with_sources(
        evaluator,
        claim=vc_exist_claim,
        node=vc_exist_leaf,
        urls=visitor_urls if visitor_urls else main_or_all_urls,
        additional_instruction=official_pa_instruction(
            "The page should explicitly mention 'visitor center' or equivalent facility."
        )
    )
    vc_hours_leaf = evaluator.add_leaf(
        id=f"park_{idx}_visitor_center_hours",
        desc="Operating hours are documented and available",
        parent=visitor_node,
        critical=True
    )
    vc_hours_text = park.visitor_center.hours if park.visitor_center else ""
    vc_hours_claim = (
        f"The visitor center at {park_name} has documented operating hours available on the official page. "
        f"The answer-provided hours text (if any): '{vc_hours_text}'."
    )
    await verify_with_sources(
        evaluator,
        claim=vc_hours_claim,
        node=vc_hours_leaf,
        urls=visitor_urls if visitor_urls else main_or_all_urls,
        additional_instruction=official_pa_instruction(
            "Look for hours of operation (days/times). A seasonal schedule is acceptable."
        )
    )
    vc_ref_leaf = evaluator.add_leaf(
        id=f"park_{idx}_visitor_center_reference",
        desc="Official source URL documenting visitor center and hours",
        parent=visitor_node,
        critical=True
    )
    vc_ref_claim = f"An official Pennsylvania page documents the visitor center and its hours for {park_name}."
    await verify_with_sources(
        evaluator,
        claim=vc_ref_claim,
        node=vc_ref_leaf,
        urls=visitor_urls if visitor_urls else main_or_all_urls,
        additional_instruction=official_pa_instruction(
            "The page should include both the existence of a visitor center and its hours."
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
    Evaluate an answer for the Pennsylvania state parks group event facilities task.
    """
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Parks are independent; allow partial across parks
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

    # Extract parks and their facility details from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_parks(),
        template_class=ParksExtraction,
        extraction_name="parks_extraction"
    )

    # Normalize to exactly 3 parks (pad with empty if needed)
    parks: List[ParkItem] = list(extracted.parks or [])
    if len(parks) > 3:
        parks = parks[:3]
    while len(parks) < 3:
        parks.append(ParkItem())

    # Build and verify each park subtree
    for i, park in enumerate(parks, start=1):
        await verify_single_park(evaluator, root, park, i)

    return evaluator.get_summary()