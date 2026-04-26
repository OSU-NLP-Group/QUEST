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
TASK_ID = "tx_hs_stadiums_12000"
TASK_DESCRIPTION = """
Identify three high school football stadiums in Texas that each have a seating capacity of exactly 12,000 spectators. For each stadium, provide the following information with supporting reference URLs: (1) the official stadium name, (2) the total construction cost in millions of dollars, (3) the year the stadium opened, (4) the school district it serves, and (5) at least two specific key amenities or features (such as video board specifications, field house details, press box facilities, or other notable infrastructure).
"""

# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class StadiumItem(BaseModel):
    """One stadium item extracted from the answer."""
    stadium_name: Optional[str] = None
    capacity: Optional[str] = None  # Keep as string to allow formats like "12,000"
    construction_cost_millions: Optional[str] = None  # e.g., "60", "60 million", "$60M"
    year_opened: Optional[str] = None  # e.g., "2016"
    school_district: Optional[str] = None  # e.g., "Katy ISD"
    amenities: List[str] = Field(default_factory=list)  # At least two features required
    sources: List[str] = Field(default_factory=list)  # Reference URLs cited for this stadium


class StadiumsExtraction(BaseModel):
    """All stadiums mentioned in the answer."""
    stadiums: List[StadiumItem] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_stadiums() -> str:
    return """
    Extract all Texas high school football stadiums mentioned in the answer, each with structured fields.
    For each stadium, extract the following fields exactly as presented in the answer text:
    - stadium_name: The official stadium name (string).
    - capacity: Seating capacity (string; keep formatting like "12,000" if present).
    - construction_cost_millions: Total construction cost expressed in millions of dollars (string; e.g., "60", "$60 million", "approximately 60 million").
    - year_opened: Year the stadium opened (string).
    - school_district: The school district the stadium serves (string, e.g., "Katy ISD" or "Allen Independent School District").
    - amenities: A list of at least two specific amenities/features (strings), such as video board specs, field house details, press box facilities, turf type, locker room capacity, etc.
    - sources: All associated reference URLs explicitly cited for this stadium in the answer (list of strings). Include only actual URLs; do not invent or infer. Include duplicates only once.

    Important:
    - You must extract only URLs explicitly present in the answer (including markdown links).
    - If a field is missing for a stadium, set it to null (or empty list for amenities/sources).
    - Preserve up to all stadiums mentioned; the evaluator will consider only the first three.
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def _has_value(v: Optional[str]) -> bool:
    return v is not None and str(v).strip() != ""


def _first_n_or_pad(items: List[StadiumItem], n: int) -> List[StadiumItem]:
    lst = items[:n]
    while len(lst) < n:
        lst.append(StadiumItem())
    return lst


def _distinct_names(items: List[StadiumItem]) -> bool:
    names = [s.stadium_name.strip() for s in items if _has_value(s.stadium_name)]
    if len(names) != 3:
        return False
    lowered = [n.lower() for n in names]
    return len(set(lowered)) == 3


# --------------------------------------------------------------------------- #
# Verification logic per stadium                                              #
# --------------------------------------------------------------------------- #
async def verify_one_stadium(
    evaluator: Evaluator,
    parent_node,
    stadium: StadiumItem,
    idx_zero_based: int,
) -> None:
    """
    Build and verify the tree for one stadium. Enforces evidence-first gating by
    verifying references and name before other factual checks.
    """
    display_idx = idx_zero_based + 1
    node_desc_map = {
        1: "First stadium meeting all requirements",
        2: "Second stadium meeting all requirements",
        3: "Third stadium meeting all requirements",
    }
    stadium_node = evaluator.add_parallel(
        id=f"stadium_{display_idx}",
        desc=node_desc_map.get(display_idx, f"Stadium #{display_idx} verification"),
        parent=parent_node,
        critical=False  # Partial credit allowed per stadium
    )

    # 0) References provided (existence check)
    refs_node = evaluator.add_custom_node(
        result=(len(stadium.sources) > 0),
        id=f"stadium_{display_idx}_references",
        desc="Supporting reference URLs are provided",
        parent=stadium_node,
        critical=True
    )

    # 1) Official stadium name verification
    name_node = evaluator.add_leaf(
        id=f"stadium_{display_idx}_name",
        desc="Official stadium name is provided",
        parent=stadium_node,
        critical=True
    )
    if not _has_value(stadium.stadium_name):
        name_node.score = 0.0
        name_node.status = "failed"
    else:
        claim = f"The official stadium name is '{stadium.stadium_name}'."
        await evaluator.verify(
            claim=claim,
            node=name_node,
            sources=stadium.sources,
            additional_instruction="Accept minor naming variants or abbreviations only if they clearly refer to the same facility; do not accept unrelated sponsor names as distinct stadiums.",
        )

    # Common prerequisites for subsequent factual checks
    prerequisites = [refs_node, name_node]

    # 2) Stadium located in Texas
    loc_node = evaluator.add_leaf(
        id=f"stadium_{display_idx}_location",
        desc="Stadium is located in Texas",
        parent=stadium_node,
        critical=True
    )
    if not _has_value(stadium.stadium_name) or len(stadium.sources) == 0:
        loc_node.score = 0.0
        loc_node.status = "skipped"
    else:
        claim = f"The stadium '{stadium.stadium_name}' is located in Texas."
        await evaluator.verify(
            claim=claim,
            node=loc_node,
            sources=stadium.sources,
            additional_instruction="Confirm the facility is in the U.S. state of Texas (TX). City/Town should be in Texas.",
            extra_prerequisites=prerequisites
        )

    # 3) Facility is a high school football stadium
    type_node = evaluator.add_leaf(
        id=f"stadium_{display_idx}_type",
        desc="Facility is a high school football stadium",
        parent=stadium_node,
        critical=True
    )
    if not _has_value(stadium.stadium_name) or len(stadium.sources) == 0:
        type_node.score = 0.0
        type_node.status = "skipped"
    else:
        claim = f"The stadium '{stadium.stadium_name}' is a high school football stadium (used by a Texas K-12 school district)."
        await evaluator.verify(
            claim=claim,
            node=type_node,
            sources=stadium.sources,
            additional_instruction="Verify that the facility is explicitly for high school football (not solely college or professional). References to ISD usage or district teams suffice.",
            extra_prerequisites=prerequisites
        )

    # 4) Capacity exactly 12,000
    cap_node = evaluator.add_leaf(
        id=f"stadium_{display_idx}_capacity",
        desc="Stadium has exactly 12,000 seating capacity",
        parent=stadium_node,
        critical=True
    )
    if not _has_value(stadium.stadium_name) or len(stadium.sources) == 0:
        cap_node.score = 0.0
        cap_node.status = "skipped"
    else:
        claim = f"The seating capacity of the stadium '{stadium.stadium_name}' is exactly 12,000."
        await evaluator.verify(
            claim=claim,
            node=cap_node,
            sources=stadium.sources,
            additional_instruction="Require exact 12,000 seats (allow formatting variations like '12,000' or '12000'); do not accept 'about', 'over', or other numbers.",
            extra_prerequisites=prerequisites
        )

    # 5) Construction cost in millions
    cost_node = evaluator.add_leaf(
        id=f"stadium_{display_idx}_cost",
        desc="Total construction cost in millions is documented",
        parent=stadium_node,
        critical=True
    )
    if not _has_value(stadium.stadium_name) or len(stadium.sources) == 0:
        cost_node.score = 0.0
        cost_node.status = "skipped"
    elif not _has_value(stadium.construction_cost_millions):
        cost_node.score = 0.0
        cost_node.status = "failed"
    else:
        claim = f"The total construction cost of the stadium '{stadium.stadium_name}' was {stadium.construction_cost_millions} million dollars."
        await evaluator.verify(
            claim=claim,
            node=cost_node,
            sources=stadium.sources,
            additional_instruction="Match the cost expressed in millions. Allow common formats like '$60 million', '60M', or textual equivalents. Minor rounding (±1M) may be acceptable only if clearly indicated as approximate.",
            extra_prerequisites=prerequisites
        )

    # 6) Year opened
    year_node = evaluator.add_leaf(
        id=f"stadium_{display_idx}_year",
        desc="Year of opening is specified",
        parent=stadium_node,
        critical=True
    )
    if not _has_value(stadium.stadium_name) or len(stadium.sources) == 0:
        year_node.score = 0.0
        year_node.status = "skipped"
    elif not _has_value(stadium.year_opened):
        year_node.score = 0.0
        year_node.status = "failed"
    else:
        claim = f"The stadium '{stadium.stadium_name}' opened in {stadium.year_opened}."
        await evaluator.verify(
            claim=claim,
            node=year_node,
            sources=stadium.sources,
            additional_instruction="Accept 'opened in', 'completed in', or 'first season in' phrasing if clearly indicating the opening year.",
            extra_prerequisites=prerequisites
        )

    # 7) School district served
    dist_node = evaluator.add_leaf(
        id=f"stadium_{display_idx}_district",
        desc="School district is identified",
        parent=stadium_node,
        critical=True
    )
    if not _has_value(stadium.stadium_name) or len(stadium.sources) == 0:
        dist_node.score = 0.0
        dist_node.status = "skipped"
    elif not _has_value(stadium.school_district):
        dist_node.score = 0.0
        dist_node.status = "failed"
    else:
        claim = f"The stadium '{stadium.stadium_name}' serves the {stadium.school_district} school district."
        await evaluator.verify(
            claim=claim,
            node=dist_node,
            sources=stadium.sources,
            additional_instruction="Allow variants like 'ISD', 'Independent School District', or district shorthand if clearly the same entity.",
            extra_prerequisites=prerequisites
        )

    # 8) Amenities/features (at least two)
    am_node = evaluator.add_leaf(
        id=f"stadium_{display_idx}_amenities",
        desc="At least two specific key amenities or features are documented",
        parent=stadium_node,
        critical=True
    )
    if not _has_value(stadium.stadium_name) or len(stadium.sources) == 0:
        am_node.score = 0.0
        am_node.status = "skipped"
    elif len([a for a in stadium.amenities if _has_value(a)]) < 2:
        am_node.score = 0.0
        am_node.status = "failed"
    else:
        # Use first two amenities for verification claim
        amenities_clean = [a.strip() for a in stadium.amenities if _has_value(a)]
        a1, a2 = amenities_clean[0], amenities_clean[1]
        claim = f"The stadium '{stadium.stadium_name}' includes these features: '{a1}' and '{a2}'."
        await evaluator.verify(
            claim=claim,
            node=am_node,
            sources=stadium.sources,
            additional_instruction="Both listed features must be supported explicitly by the provided sources (text or screenshot). Allow minor wording variations for the same feature.",
            extra_prerequisites=prerequisites
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
    Evaluate an answer for the Texas high school stadiums (12,000 capacity) task.
    """
    # Initialize evaluator (root must be non-critical to allow non-critical stadium subtrees)
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,
        agent_name=agent_name,
        answer_name=answer_name,
        client=client,
        task_description="Complete identification and documentation of three Texas high school football stadiums with 12,000 seating capacity",
        answer=answer,
        global_cache=cache,
        global_semaphore=semaphore,
        logger=logger,
        default_model=model,
    )

    # Extract stadiums from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_stadiums(),
        template_class=StadiumsExtraction,
        extraction_name="stadiums_extraction",
    )

    # Build first-three list (padding if needed)
    first_three = _first_n_or_pad(extracted.stadiums, 3)

    # Distinct stadiums (critical leaf at root)
    distinct_node = evaluator.add_custom_node(
        result=_distinct_names(first_three),
        id="distinct_stadiums",
        desc="All three identified stadiums are distinct facilities (no duplicates)",
        parent=root,
        critical=True
    )

    # Count requirement (critical leaf at root) — must be exactly three identified in the answer
    count_node = evaluator.add_custom_node(
        result=(len(extracted.stadiums) == 3),
        id="count_requirement",
        desc="Exactly three stadiums are identified (not more, not fewer)",
        parent=root,
        critical=True
    )

    # Verify each of the first three stadiums
    for i, stadium in enumerate(first_three):
        await verify_one_stadium(evaluator, root, stadium, i)

    # Return structured summary
    return evaluator.get_summary()