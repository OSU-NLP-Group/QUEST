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
TASK_ID = "co_ski_mlk_2026"
TASK_DESCRIPTION = (
    "I'm planning a ski trip to Colorado during the Martin Luther King Jr. Day 2026 long weekend and need help "
    "identifying suitable resorts. First, please confirm the exact date of Martin Luther King Jr. Day in 2026. "
    "Then, I'm looking for two different Colorado ski resorts that meet the following criteria: base elevation of "
    "at least 9,000 feet above sea level, vertical drop of at least 2,500 feet, at least 20% of the terrain "
    "classified as intermediate difficulty, at least 2,000 acres of skiable terrain, uphill lift capacity of at least "
    "20,000 skiers per hour, and confirmed to be operating during January 2026. For each resort, please provide the "
    "specific measurements (base elevation, vertical drop, intermediate terrain percentage, skiable acres, and lift "
    "capacity) along with reference URLs that verify these specifications."
)

# Ground truth date for MLK Day 2026 per rubric
GROUND_TRUTH_MLK_2026 = "Monday, January 19, 2026"

# Helpful conversions for verifier guidance
FEET_9000_METERS = 2743  # 9000 ft ≈ 2743 m
FEET_2500_METERS = 762   # 2500 ft ≈ 762 m


# --------------------------------------------------------------------------- #
# Data models for extractions                                                 #
# --------------------------------------------------------------------------- #
class MLKDateExtraction(BaseModel):
    """Extraction of the MLK Day 2026 date and any cited URLs."""
    date_text: Optional[str] = None
    date_iso: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class ResortSpec(BaseModel):
    """Single resort specifications as presented in the answer."""
    name: Optional[str] = None
    location: Optional[str] = None  # e.g., "Colorado", or city, state
    base_elevation_ft: Optional[str] = None
    vertical_drop_ft: Optional[str] = None
    intermediate_percent: Optional[str] = None
    skiable_acres: Optional[str] = None
    lift_capacity_per_hour: Optional[str] = None
    operating_jan_2026: Optional[str] = None  # free-form statement or yes/no
    reference_urls: List[str] = Field(default_factory=list)


class ResortsExtraction(BaseModel):
    """All resorts extracted from the answer."""
    resorts: List[ResortSpec] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_mlk_date() -> str:
    return """
    Extract the answer's stated date for Martin Luther King Jr. Day in 2026 and any URLs cited to support it.

    Return a JSON object with:
    - date_text: The exact date string as stated in the answer (e.g., "Monday, January 19, 2026", "Jan 19, 2026 (Monday)").
    - date_iso: If the answer gives or implies an ISO-like date (YYYY-MM-DD), extract that; otherwise return null.
    - sources: An array of all URLs mentioned in the answer for confirming this date (official calendars, US government pages, reliable holiday sites, etc.).

    Rules:
    - Only extract what is explicitly in the answer. Do not infer new URLs.
    - If multiple date expressions appear, choose the one the answer claims is MLK Day 2026.
    - If no URLs are given, return an empty array for sources.
    """


def prompt_extract_resorts() -> str:
    return """
    Extract up to the first two Colorado ski resorts along with the specific measurements and reference URLs provided in the answer.

    For each resort, extract the following fields as strings (exactly as written in the answer when possible):
    - name
    - location: State or place text indicating location (e.g., "Colorado", "Breckenridge, Colorado"). If absent, return null.
    - base_elevation_ft: Base elevation value and unit as written (e.g., "9,600 ft", "2,926 m"). If absent, return null.
    - vertical_drop_ft: Vertical drop value and unit as written (e.g., "3,398 ft", "1,036 m"). If absent, return null.
    - intermediate_percent: Intermediate terrain percentage (e.g., "25%", "about 25%"). If absent, return null.
    - skiable_acres: Skiable terrain acreage (e.g., "2,908 acres"). If absent, return null.
    - lift_capacity_per_hour: Uphill lift capacity (e.g., "46,800 skiers/hour"). If absent, return null.
    - operating_jan_2026: The statement or indication that the resort operates during January 2026 (e.g., "2025–26 season", "open in January 2026"). If not provided, return null.
    - reference_urls: All URLs specifically cited for this resort's stats/operation. Include official resort pages and credible sources. Return an array; if none, return an empty array.

    Return a JSON object:
    {
      "resorts": [ResortSpec, ResortSpec, ...]
    }

    Notes:
    - Do NOT invent URLs; extract only those present in the answer (plain text or markdown links).
    - Keep numeric values as strings to preserve formatting (e.g., "≈", "about", commas).
    - If more than two resorts are mentioned, include only the first two.
    """


# --------------------------------------------------------------------------- #
# Verification helpers                                                        #
# --------------------------------------------------------------------------- #
async def verify_mlk_phase(
    evaluator: Evaluator,
    parent_node,
    mlk_info: MLKDateExtraction
) -> None:
    """
    Build and verify the 'date_identification_phase' subtree.
    """
    date_node = evaluator.add_parallel(
        id="date_identification_phase",
        desc="Identify and verify Martin Luther King Jr. Day 2026",
        parent=parent_node,
        critical=True  # Critical for the overall task
    )

    # Leaf: Verify date value itself (simple, logic/knowledge level)
    mlk_value_leaf = evaluator.add_leaf(
        id="mlk_day_date_value",
        desc="Date must be Monday, January 19, 2026",
        parent=date_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"In 2026, Martin Luther King Jr. Day falls on {GROUND_TRUTH_MLK_2026}.",
        node=mlk_value_leaf,
        additional_instruction=(
            "MLK Day in the U.S. is the third Monday in January. For 2026, "
            "that Monday is January 19, 2026. Judge this claim directly."
        )
    )

    # Leaf: Verify date against provided reference URL(s)
    mlk_ref_leaf = evaluator.add_leaf(
        id="mlk_day_reference_url",
        desc="Provide reference URL confirming the MLK Day 2026 date",
        parent=date_node,
        critical=True
    )
    # Use any provided sources from the answer; if none provided, this will fail
    await evaluator.verify(
        claim=f"Authoritative sources confirm that Martin Luther King Jr. Day in 2026 is {GROUND_TRUTH_MLK_2026}.",
        node=mlk_ref_leaf,
        sources=mlk_info.sources,
        additional_instruction=(
            "Use the provided URL(s) to confirm the exact 2026 MLK Day date. "
            "Accept official or reliable holiday calendars that explicitly list the 2026 date."
        )
    )


def _normalize_urls(urls: Optional[List[str]]) -> List[str]:
    if not urls:
        return []
    # Remove obvious duplicates / whitespace
    seen = set()
    clean = []
    for u in urls:
        if not u:
            continue
        s = u.strip()
        if s and s not in seen:
            seen.add(s)
            clean.append(s)
    return clean


async def verify_single_resort(
    evaluator: Evaluator,
    parent_node,
    resort: ResortSpec,
    idx: int
) -> None:
    """
    Build and verify the subtree for a single resort's specifications.
    """
    rid = idx + 1
    resort_node = evaluator.add_parallel(
        id=f"resort_{rid}_specifications",
        desc=f"{'First' if rid == 1 else 'Second'} Colorado ski resort meeting all requirements",
        parent=parent_node,
        critical=False  # Allow partial within this resort block
    )

    # Prepare sources list early and add the "reference_urls" critical existence node as a gate.
    resort_sources = _normalize_urls(resort.reference_urls)

    refs_exist_leaf = evaluator.add_custom_node(
        result=(len(resort_sources) > 0),
        id=f"resort_{rid}_reference_urls",
        desc="Reference URL(s) supporting all specifications",
        parent=resort_node,
        critical=True  # Critical gate for other checks
    )

    # Leaf: Location must be Colorado
    location_leaf = evaluator.add_leaf(
        id=f"resort_{rid}_location_verification",
        desc="Resort must be located in Colorado",
        parent=resort_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The ski resort '{resort.name or 'UNKNOWN'}' is located in Colorado.",
        node=location_leaf,
        sources=resort_sources,
        additional_instruction=(
            "Confirm that the resort is in the U.S. state of Colorado; acceptable if the page shows a Colorado address, "
            "Colorado map location, or text clearly stating Colorado."
        )
    )

    # Leaf: Base elevation >= 9,000 ft
    base_leaf = evaluator.add_leaf(
        id=f"resort_{rid}_base_elevation",
        desc="Base elevation must be at least 9,000 feet above sea level",
        parent=resort_node,
        critical=True
    )
    await evaluator.verify(
        claim=(
            f"The base elevation of the ski resort '{resort.name or 'UNKNOWN'}' is at least 9,000 feet "
            f"(i.e., ≥ {FEET_9000_METERS} meters)."
        ),
        node=base_leaf,
        sources=resort_sources,
        additional_instruction=(
            f"Check mountain statistics for base elevation. If only meters are given, convert: "
            f"9,000 ft ≈ {FEET_9000_METERS} m. Pass if base elevation ≥ 9,000 ft (or ≥ {FEET_9000_METERS} m). "
            "Minor rounding is acceptable."
        )
    )

    # Leaf: Vertical drop >= 2,500 ft
    vdrop_leaf = evaluator.add_leaf(
        id=f"resort_{rid}_vertical_drop",
        desc="Vertical drop must be at least 2,500 feet",
        parent=resort_node,
        critical=True
    )
    await evaluator.verify(
        claim=(
            f"The vertical drop of the ski resort '{resort.name or 'UNKNOWN'}' is at least 2,500 feet "
            f"(i.e., ≥ {FEET_2500_METERS} meters)."
        ),
        node=vdrop_leaf,
        sources=resort_sources,
        additional_instruction=(
            f"Check the listed vertical drop. If only meters are given, convert: "
            f"2,500 ft ≈ {FEET_2500_METERS} m. Pass if vertical drop ≥ 2,500 ft (or ≥ {FEET_2500_METERS} m). "
            "Minor rounding is acceptable."
        )
    )

    # Leaf: Intermediate terrain >= 20%
    interm_leaf = evaluator.add_leaf(
        id=f"resort_{rid}_intermediate_terrain",
        desc="Intermediate terrain must comprise at least 20% of total terrain",
        parent=resort_node,
        critical=True
    )
    await evaluator.verify(
        claim=(
            f"At least 20% of the terrain at the ski resort '{resort.name or 'UNKNOWN'}' is rated Intermediate."
        ),
        node=interm_leaf,
        sources=resort_sources,
        additional_instruction=(
            "Check the terrain breakdown. Accept if the page explicitly lists Intermediate ≥ 20% (e.g., 'Blue', "
            "'Intermediate'). Do not combine categories—only the specific 'Intermediate' percentage counts."
        )
    )

    # Leaf: Skiable acres >= 2,000
    acres_leaf = evaluator.add_leaf(
        id=f"resort_{rid}_skiable_acres",
        desc="Skiable terrain must be at least 2,000 acres",
        parent=resort_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The ski resort '{resort.name or 'UNKNOWN'}' has at least 2,000 acres of skiable terrain.",
        node=acres_leaf,
        sources=resort_sources,
        additional_instruction=(
            "Verify the skiable terrain acreage on the referenced page. Pass if acreage ≥ 2,000 acres. "
            "Minor rounding or 'about' phrasing is acceptable if clearly ≥ 2,000."
        )
    )

    # Leaf: Lift capacity ≥ 20,000 skiers/hour
    capacity_leaf = evaluator.add_leaf(
        id=f"resort_{rid}_lift_capacity",
        desc="Uphill lift capacity must be at least 20,000 skiers per hour",
        parent=resort_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The uphill lift capacity of the ski resort '{resort.name or 'UNKNOWN'}' is at least 20,000 skiers per hour.",
        node=capacity_leaf,
        sources=resort_sources,
        additional_instruction=(
            "Verify the listed uphill lift capacity (skiers per hour). Pass if the referenced page clearly supports ≥ 20,000."
        )
    )

    # Leaf: Operates during January 2026
    ops_leaf = evaluator.add_leaf(
        id=f"resort_{rid}_operational_status",
        desc="Resort operates during January 2026 (winter 2025-2026 season)",
        parent=resort_node,
        critical=True
    )
    await evaluator.verify(
        claim=(
            f"The ski resort '{resort.name or 'UNKNOWN'}' is operating and open for skiing during January 2026 "
            "(i.e., within the 2025–2026 winter season)."
        ),
        node=ops_leaf,
        sources=resort_sources,
        additional_instruction=(
            "Accept season calendars or operating schedules that show the resort is open in January 2026 (e.g., '2025–26 season', "
            "typical opening/closing dates indicating January operations). Reject pages with no clear evidence."
        )
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
    Evaluate an answer for the Colorado MLK 2026 ski planning task.
    """
    # Initialize evaluator; use SEQUENTIAL to respect the two-phase order (date -> resorts)
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.SEQUENTIAL,
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

    # Extract MLK date info
    mlk_info = await evaluator.extract(
        prompt=prompt_extract_mlk_date(),
        template_class=MLKDateExtraction,
        extraction_name="mlk_date_extraction"
    )

    # Extract resort specs (take first two only; pad if fewer)
    resorts_info = await evaluator.extract(
        prompt=prompt_extract_resorts(),
        template_class=ResortsExtraction,
        extraction_name="resorts_extraction"
    )
    resorts_list: List[ResortSpec] = list(resorts_info.resorts[:2])
    while len(resorts_list) < 2:
        resorts_list.append(ResortSpec())

    # Add ground truth/context info
    evaluator.add_ground_truth({
        "expected_mlk_2026": GROUND_TRUTH_MLK_2026,
        "resort_requirements": {
            "location": "Colorado",
            "base_elevation_ft_min": 9000,
            "vertical_drop_ft_min": 2500,
            "intermediate_percent_min": 20,
            "skiable_acres_min": 2000,
            "lift_capacity_per_hour_min": 20000,
            "operating_month": "January 2026"
        }
    })

    # Phase 1: Date identification (critical)
    await verify_mlk_phase(evaluator, root, mlk_info)

    # Phase 2: Resort collection (non-critical, parallel across two resorts)
    resorts_parent = evaluator.add_parallel(
        id="resort_collection_phase",
        desc="Identify two distinct Colorado ski resorts meeting all specified criteria",
        parent=root,
        critical=False
    )

    # Resort 1
    await verify_single_resort(evaluator, resorts_parent, resorts_list[0], idx=0)
    # Resort 2
    await verify_single_resort(evaluator, resorts_parent, resorts_list[1], idx=1)

    # Return the final structured evaluation summary
    return evaluator.get_summary()