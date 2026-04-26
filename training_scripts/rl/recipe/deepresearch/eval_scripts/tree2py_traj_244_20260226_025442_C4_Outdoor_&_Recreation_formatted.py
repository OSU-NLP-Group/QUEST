import asyncio
import logging
from typing import Optional, List, Dict, Any, Tuple

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.llm_client.base_client import LLMClient
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "california_mlk_2026_free_parks"
TASK_DESCRIPTION = (
    "I'm planning to visit California state parks on Martin Luther King Jr. Day in 2026 to take advantage of any free entry programs. "
    "Please provide the following information:\n\n"
    "1. What is the exact date of MLK Day in 2026?\n"
    "2. How many California state parks are participating in the free entry program on this day?\n"
    "3. What type of fees are waived under this program?\n"
    "4. Who announced this program and when?\n"
    "5. Identify at least three specific California state parks from different geographic regions (including at least one coastal park and one mountain park) "
    "that participate in this program, and provide an official website link for each park."
)


# Ground truth expectations used for strict value-match leaves (answer consistency checks)
GROUND_TRUTH = {
    "mlk_day_2026_date": "January 19, 2026",
    "participating_count_descriptor": "200+",
    "fee_waiver_type": "vehicle day-use entry fees",
    "announcement_by": "Governor Gavin Newsom",
    "announcement_date": "January 16, 2026",
}


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class ProgramExtraction(BaseModel):
    """Program-level facts extracted from the answer."""
    mlk_day_date: Optional[str] = None  # e.g., "January 19, 2026"
    participating_parks_count: Optional[str] = None  # e.g., "200+", "over 200", "more than 200"
    fee_waiver_type: Optional[str] = None  # e.g., "vehicle day-use entry fees"
    announcement_by: Optional[str] = None  # e.g., "Governor Gavin Newsom"
    announcement_date: Optional[str] = None  # e.g., "January 16, 2026"
    program_sources: List[str] = Field(default_factory=list)  # URLs cited in the answer that support program-level claims


class ParkItem(BaseModel):
    """A single park entry extracted from the answer."""
    name: Optional[str] = None
    region: Optional[str] = None  # free-text description, e.g., "coastal", "Sierra", "Central Valley", "desert"
    url: Optional[str] = None  # official park page or official reference URL


class ParksExtraction(BaseModel):
    """List of parks extracted from the answer."""
    parks: List[ParkItem] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_program_info() -> str:
    return (
        "Extract the program-level information about California's MLK Day 2026 free state parks program from the answer.\n"
        "Return a JSON object with the following fields:\n"
        "1. mlk_day_date: The exact date (e.g., 'January 19, 2026'). If not provided, return null.\n"
        "2. participating_parks_count: The stated number of participating parks as described in the answer (e.g., '200+', 'over 200', 'more than 200'). If not provided, return null.\n"
        "3. fee_waiver_type: The type of fees waived (e.g., 'vehicle day-use entry fees'). If not provided, return null.\n"
        "4. announcement_by: Who announced the program (e.g., 'Governor Gavin Newsom'). If not provided, return null.\n"
        "5. announcement_date: The announcement date (e.g., 'January 16, 2026'). If not provided, return null.\n"
        "6. program_sources: An array of all URLs explicitly cited in the answer that support any of the above program facts.\n"
        "Important:\n"
        "- Do not invent information; extract exactly what appears in the answer.\n"
        "- For URLs, include only valid, complete URLs mentioned in the answer (plain or markdown links)."
    )


def prompt_extract_parks() -> str:
    return (
        "Extract all parks the answer claims are participating in the MLK Day 2026 free entry program.\n"
        "For each park, return a JSON object with:\n"
        "1. name: The park name.\n"
        "2. region: The geographic region label provided in the answer for the park (e.g., 'coastal', 'Sierra', 'desert', 'Central Valley'). "
        "If the answer does not explicitly provide a region, return null.\n"
        "3. url: The official website URL or official reference URL for the park as provided in the answer.\n"
        "Do not infer or invent details. Only include parks that the answer explicitly claims participate and for which a URL is provided."
    )


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def classify_park_category(park: ParkItem) -> str:
    """
    Classify park into one of {'coastal', 'mountain', 'other', 'unknown'} using provided region text and name heuristics.
    This is only used for selection routing; formal verification will be done via URL evidence.
    """
    txt = (park.region or "").lower()
    name = (park.name or "").lower()

    coastal_tokens = ["coast", "coastal", "beach", "shore", "ocean", "bay", "pacific"]
    mountain_tokens = ["mountain", "sierra", "alpine", "tahoe", "yosemite", "peak", "ridge", "summit", "range"]

    if any(tok in txt for tok in coastal_tokens) or any(tok in name for tok in coastal_tokens):
        return "coastal"
    if any(tok in txt for tok in mountain_tokens) or any(tok in name for tok in mountain_tokens):
        return "mountain"
    if txt.strip() != "":
        # If a non-empty region provided but didn't match coastal/mountain, treat as 'other'
        return "other"

    # Heuristic from name if region missing
    if "beach" in name or "shore" in name or "coast" in name:
        return "coastal"
    if "mount" in name or "sierra" in name or "tahoe" in name or "yosemite" in name:
        return "mountain"

    return "unknown"


def select_parks_by_region(parks: List[ParkItem]) -> Tuple[Optional[ParkItem], Optional[ParkItem], Optional[ParkItem]]:
    """
    Select three parks by categories:
    - coastal: one park categorized as coastal
    - mountain: one park categorized as mountain or Sierra
    - other: one park categorized as neither coastal nor mountain
    Returns (coastal, mountain, other). Each element can be None if not found.
    """
    coastal: Optional[ParkItem] = None
    mountain: Optional[ParkItem] = None
    other: Optional[ParkItem] = None

    for p in parks:
        cat = classify_park_category(p)
        if coastal is None and cat == "coastal":
            coastal = p
        elif mountain is None and cat == "mountain":
            mountain = p
        elif other is None and cat == "other":
            other = p

        if coastal and mountain and other:
            break

    return coastal, mountain, other


# --------------------------------------------------------------------------- #
# Verification subtrees                                                       #
# --------------------------------------------------------------------------- #
async def verify_program_date(evaluator: Evaluator, parent, program: ProgramExtraction) -> None:
    """
    Parent node: MLK Day 2026 Date (critical).
    Children leaves:
    - Value match (answer consistency): extracted date equals the expected 'January 19, 2026'
    - Source-supported: program sources support that MLK Day in 2026 is January 19, 2026
    """
    node = evaluator.add_parallel(
        id="MLK_Day_2026_Date",
        desc="The date of Martin Luther King Jr. Day in 2026 is correctly identified as January 19, 2026",
        parent=parent,
        critical=True
    )

    # Leaf 1: Value match against ground truth (simple verify)
    val_leaf = evaluator.add_leaf(
        id="mlk_date_value_match",
        desc="Answer's stated MLK Day 2026 date matches 'January 19, 2026'",
        parent=node,
        critical=True
    )
    stated = program.mlk_day_date or ""
    claim = (
        f"The stated date '{stated}' equals 'January 19, 2026' allowing minor formatting variants "
        f"(e.g., 'Jan 19, 2026' or 'Monday, January 19, 2026')."
    )
    await evaluator.verify(
        claim=claim,
        node=val_leaf,
        additional_instruction="Judge textual equivalence under minor formatting or punctuation differences; focus on the actual calendar date."
    )

    # Leaf 2: Source-supported verification
    src_leaf = evaluator.add_leaf(
        id="mlk_date_supported_by_sources",
        desc="MLK Day 2026 date is supported by cited program sources",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="Martin Luther King Jr. Day in 2026 falls on January 19, 2026.",
        node=src_leaf,
        sources=program.program_sources,
        additional_instruction="Confirm the date referring specifically to MLK Day in 2026; accept clear statements or official calendar references."
    )


async def verify_program_count(evaluator: Evaluator, parent, program: ProgramExtraction) -> None:
    """
    Parent node: Participating Parks Count (critical).
    Children leaves:
    - Value implies 200+ (answer consistency): answer's count wording implies more than 200
    - Source-supported: program sources support 'more than 200' (200+)
    """
    node = evaluator.add_parallel(
        id="Participating_Parks_Count",
        desc="The number of participating parks is correctly stated as more than 200 (200+)",
        parent=parent,
        critical=True
    )

    # Leaf 1: Value implication check (simple verify)
    val_leaf = evaluator.add_leaf(
        id="parks_count_implies_200_plus",
        desc="Answer's stated count implies 200+ participating parks",
        parent=node,
        critical=True
    )
    count_text = (program.participating_parks_count or "").strip()
    claim = (
        f"The stated count '{count_text}' implies more than 200 participating parks (i.e., 200+). "
        f"Accept phrasing such as 'over 200', '200+', 'more than 200', or any explicit integer greater than 200."
    )
    await evaluator.verify(
        claim=claim,
        node=val_leaf,
        additional_instruction="Use common-sense interpretation of the phrase; allow synonyms and variations indicating >200."
    )

    # Leaf 2: Source-supported verification
    src_leaf = evaluator.add_leaf(
        id="parks_count_supported_by_sources",
        desc="Program sources support that more than 200 parks participate",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="More than 200 California state parks (200+) participate in the MLK Day 2026 free vehicle day-use entry program.",
        node=src_leaf,
        sources=program.program_sources,
        additional_instruction="Look for explicit counts or phrases like 'more than 200', '200+', or a list that clearly indicates the scale."
    )


async def verify_program_fee(evaluator: Evaluator, parent, program: ProgramExtraction) -> None:
    """
    Parent node: Fee Waiver Type (critical).
    Children leaves:
    - Value match (answer consistency): fee type equals 'vehicle day-use entry fees'
    - Source-supported: program sources support that only vehicle day-use entry fees are waived
    """
    node = evaluator.add_parallel(
        id="Fee_Waiver_Type",
        desc="Fee type waived is correctly identified as vehicle day-use entry fees (not camping or special fees)",
        parent=parent,
        critical=True
    )

    # Leaf 1: Value match (simple verify)
    val_leaf = evaluator.add_leaf(
        id="fee_type_value_match",
        desc="Answer's fee type matches 'vehicle day-use entry fees' only",
        parent=node,
        critical=True
    )
    fee_text = program.fee_waiver_type or ""
    claim = (
        f"The stated fee type '{fee_text}' matches 'vehicle day-use entry fees' and does not include camping or other special fees."
    )
    await evaluator.verify(
        claim=claim,
        node=val_leaf,
        additional_instruction="Equate 'free day-use entry' with vehicle day-use entry fees; ensure camping/special fees are explicitly excluded."
    )

    # Leaf 2: Source-supported verification
    src_leaf = evaluator.add_leaf(
        id="fee_type_supported_by_sources",
        desc="Program sources support vehicle day-use entry fees are waived (not camping/special fees)",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="Under the MLK Day 2026 program, vehicle day-use entry fees are waived; camping and special-use fees are not waived.",
        node=src_leaf,
        sources=program.program_sources,
        additional_instruction="Confirm scope of the fee waiver; accept explicit statements restricting the waiver to day-use vehicle entry fees."
    )


async def verify_program_announcement(evaluator: Evaluator, parent, program: ProgramExtraction) -> None:
    """
    Parent node: Program Announcement Source (critical).
    Children leaves:
    - Value match: announcer equals 'Governor Gavin Newsom'
    - Value match: announcement date equals 'January 16, 2026'
    - Source-supported: program sources support both announcer and date
    """
    node = evaluator.add_parallel(
        id="Program_Announcement_Source",
        desc="The announcement source is correctly identified as Governor Gavin Newsom, announced on January 16, 2026",
        parent=parent,
        critical=True
    )

    # Leaf 1: Announcer value match
    who_leaf = evaluator.add_leaf(
        id="announcement_who_value_match",
        desc="Announcer matches 'Governor Gavin Newsom'",
        parent=node,
        critical=True
    )
    who_text = program.announcement_by or ""
    await evaluator.verify(
        claim=f"The stated announcer '{who_text}' equals 'Governor Gavin Newsom' (allow minor variants like 'Gov. Gavin Newsom').",
        node=who_leaf,
        additional_instruction="Treat 'Governor Gavin Newsom' and reasonable minor variants (e.g., 'Gov. Gavin Newsom', 'Governor Newsom') as equivalent."
    )

    # Leaf 2: Announcement date value match
    date_leaf = evaluator.add_leaf(
        id="announcement_date_value_match",
        desc="Announcement date matches 'January 16, 2026'",
        parent=node,
        critical=True
    )
    ann_date = program.announcement_date or ""
    await evaluator.verify(
        claim=f"The stated announcement date '{ann_date}' equals 'January 16, 2026' allowing minor formatting variants.",
        node=date_leaf,
        additional_instruction="Judge textual equivalence under minor formatting or punctuation differences; focus on the actual calendar date."
    )

    # Leaf 3: Source-supported verification (who and date)
    src_leaf = evaluator.add_leaf(
        id="announcement_supported_by_sources",
        desc="Program sources support that Gov. Gavin Newsom announced the program on January 16, 2026",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="The MLK Day 2026 parks free-entry program was announced by Governor Gavin Newsom on January 16, 2026.",
        node=src_leaf,
        sources=program.program_sources,
        additional_instruction="Check official Governor/State Parks press releases or announcements confirming both announcer and date."
    )


async def verify_region_park(
    evaluator: Evaluator,
    parent,
    park: Optional[ParkItem],
    region_category: str,
    program_sources: List[str],
    node_id: str,
    node_desc: str
) -> None:
    """
    Build verification sub-tree for a specific region category.
    Parent node is NON-CRITICAL (partial credit allowed).
    Children leaves (all critical under this parent to gate semantics):
    - Existence: park name and url provided (custom)
    - Official URL validity: the URL is an official website/reference for the park (verify by URL)
    - Region classification: the park fits the requested region category (verify by URL)
    - Participation: the park participates in MLK Day 2026 free vehicle day-use entry program (verify by URLs with park.url + program_sources)
    """
    node = evaluator.add_parallel(
        id=node_id,
        desc=node_desc,
        parent=parent,
        critical=False
    )

    # Leaf: Existence check for name and URL (critical, gates subsequent leaves)
    exists = bool(park and park.name and park.url)
    exist_leaf = evaluator.add_custom_node(
        result=exists,
        id=f"{node_id}_exists",
        desc=f"{region_category.title()} park has name and official URL provided",
        parent=node,
        critical=True
    )

    # If we have a park, proceed with further leaves; they'll auto-skip if existence failed
    park_name = (park.name if park and park.name else "") or ""
    park_url = (park.url if park and park.url else "") or ""

    # Leaf: Official URL validity
    url_leaf = evaluator.add_leaf(
        id=f"{node_id}_official_url_valid",
        desc="Park URL is an official website/reference",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The URL '{park_url}' is an official website or official reference for the California state park '{park_name}'.",
        node=url_leaf,
        sources=park_url,
        additional_instruction="Prefer parks.ca.gov or other .ca.gov domains or official state sources; confirm the page clearly represents the park officially."
    )

    # Leaf: Region classification check
    region_leaf = evaluator.add_leaf(
        id=f"{node_id}_region_correct",
        desc=f"Park fits the '{region_category}' region classification",
        parent=node,
        critical=True
    )

    if region_category == "coastal":
        region_claim = (
            f"The park '{park_name}' is a coastal park on or directly adjacent to California's ocean coastline "
            f"(e.g., state beach or coastal unit)."
        )
    elif region_category == "mountain":
        region_claim = (
            f"The park '{park_name}' is a mountain/Sierra region park (e.g., Sierra Nevada, alpine terrain, mountainous area)."
        )
    else:
        # 'other' category explicitly requires not coastal or mountain
        region_claim = (
            f"The park '{park_name}' is in a different geographic region (not coastal and not mountain), "
            f"such as desert, inland valley, or other non-coastal, non-mountain area."
        )

    await evaluator.verify(
        claim=region_claim,
        node=region_leaf,
        sources=park_url,
        additional_instruction="Use the park's official page description/location cues to judge coastal/mountain/other classification; "
                              "allow reasonable inference from geography terms (beach/coast/ocean for coastal; Sierra/alpine/mountain for mountain; "
                              "desert/valley/inland for other)."
    )

    # Leaf: Participation check
    part_leaf = evaluator.add_leaf(
        id=f"{node_id}_participates_program",
        desc="Park participates in MLK Day 2026 free vehicle day-use entry program",
        parent=node,
        critical=True
    )
    srcs: List[str] = []
    if park_url:
        srcs.append(park_url)
    if program_sources:
        srcs.extend(program_sources)

    await evaluator.verify(
        claim=(
            f"The park '{park_name}' participates in the MLK Day (January 19, 2026) "
            f"free vehicle day-use entry program."
        ),
        node=part_leaf,
        sources=srcs if srcs else None,
        additional_instruction="Confirm participation via statewide program announcements listing the park or via the park's own official page."
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
    Evaluate the answer for California MLK Day 2026 free state parks program.
    Returns a standardized summary with verification tree.
    """
    # Initialize evaluator with root parallel aggregation (as per rubric)
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

    # Extract program-level info and parks list (can be done in parallel)
    program_task = evaluator.extract(
        prompt=prompt_extract_program_info(),
        template_class=ProgramExtraction,
        extraction_name="program_info"
    )
    parks_task = evaluator.extract(
        prompt=prompt_extract_parks(),
        template_class=ParksExtraction,
        extraction_name="parks_list"
    )

    program_info, parks_info = await asyncio.gather(program_task, parks_task)

    # Add ground truth for transparency
    evaluator.add_ground_truth({
        "expected_mlk_day_2026_date": GROUND_TRUTH["mlk_day_2026_date"],
        "expected_participating_count_descriptor": GROUND_TRUTH["participating_count_descriptor"],
        "expected_fee_waiver_type": GROUND_TRUTH["fee_waiver_type"],
        "expected_announcement_by": GROUND_TRUTH["announcement_by"],
        "expected_announcement_date": GROUND_TRUTH["announcement_date"]
    }, gt_type="expected_values")

    # Build program facts verification subtrees (critical)
    await verify_program_date(evaluator, root, program_info)
    await verify_program_count(evaluator, root, program_info)
    await verify_program_fee(evaluator, root, program_info)
    await verify_program_announcement(evaluator, root, program_info)

    # Parks from different regions (non-critical nodes allowing partial credit)
    coastal_pick, mountain_pick, other_pick = select_parks_by_region(parks_info.parks)

    await verify_region_park(
        evaluator=evaluator,
        parent=root,
        park=coastal_pick,
        region_category="coastal",
        program_sources=program_info.program_sources,
        node_id="Coastal_Region_Park",
        node_desc="At least one coastal region park participates with name and official URL provided"
    )

    await verify_region_park(
        evaluator=evaluator,
        parent=root,
        park=mountain_pick,
        region_category="mountain",
        program_sources=program_info.program_sources,
        node_id="Mountain_Region_Park",
        node_desc="At least one mountain/Sierra region park participates with name and official URL provided"
    )

    await verify_region_park(
        evaluator=evaluator,
        parent=root,
        park=other_pick,
        region_category="other",
        program_sources=program_info.program_sources,
        node_id="Third_Region_Park",
        node_desc="At least one additional park from a different (non-coastal, non-mountain) region participates with name and official URL provided"
    )

    # Optionally record custom info for debugging/trace
    evaluator.add_custom_info(
        info={
            "program_sources_count": len(program_info.program_sources),
            "parks_extracted": len(parks_info.parks),
            "selected_parks": {
                "coastal": coastal_pick.dict() if coastal_pick else None,
                "mountain": mountain_pick.dict() if mountain_pick else None,
                "other": other_pick.dict() if other_pick else None
            }
        },
        info_type="selection_summary"
    )

    # Return structured result
    return evaluator.get_summary()