import asyncio
import logging
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator, AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys


# -----------------------------------------------------------------------------
# Task-specific constants
# -----------------------------------------------------------------------------
TASK_ID = "dfw_hs_stadiums"
TASK_DESCRIPTION = (
    "I am researching large high school football stadiums in the Dallas-Fort Worth area for a documentary "
    "about Texas high school athletics. I need to identify at least three high school football stadiums that "
    "meet the following criteria: (1) Located in Collin County, Texas, or in cities within the immediate "
    "Dallas-Fort Worth metropolitan area; (2) Have a seating capacity of at least 12,000; (3) Are owned and "
    "operated by a public school district (not private schools); (4) Serve as the home stadium for at least one "
    "high school that is classified as either UIL Class 5A (enrollment 1,305-2,214) or UIL Class 6A "
    "(enrollment 2,215 and above) according to the 2026-2028 UIL reclassification. For each stadium, please "
    "provide: the official stadium name, the complete street address (including street number, street name, "
    "city, state, and ZIP code), the seating capacity, and the name of the public school district that owns "
    "and operates the stadium."
)


# -----------------------------------------------------------------------------
# Data Models for Extraction
# -----------------------------------------------------------------------------
class StadiumEntry(BaseModel):
    # Required descriptive fields
    official_name: Optional[str] = None
    address: Optional[str] = None  # Expected to be a single complete string including number, street, city, state, ZIP
    seating_capacity: Optional[str] = None  # Keep as string to be robust (e.g., "12,000+", "approx. 18,000")
    district_name: Optional[str] = None  # Public school district name (e.g., "Allen ISD")

    # Home schools served by this stadium
    home_high_schools: List[str] = Field(default_factory=list)

    # Source URLs explicitly included in the answer text
    capacity_sources: List[str] = Field(default_factory=list)            # Pages that state capacity
    location_sources: List[str] = Field(default_factory=list)            # Pages that show address/location/city/county
    ownership_sources: List[str] = Field(default_factory=list)           # Pages that state district ownership/operation
    uil_classification_sources: List[str] = Field(default_factory=list)  # UIL or equivalent pages showing 2026–2028 classes


class StadiumExtraction(BaseModel):
    stadiums: List[StadiumEntry] = Field(default_factory=list)


# -----------------------------------------------------------------------------
# Extraction Prompt
# -----------------------------------------------------------------------------
def prompt_extract_stadiums() -> str:
    return """
Extract up to 5 high school football stadium entries mentioned in the answer.

For each stadium, extract the following fields exactly as presented:
- official_name: The official name of the stadium.
- address: The complete street address as a single line string, including street number, street name, city, state, and 5-digit ZIP.
- seating_capacity: The stated seating capacity (leave formatting as-is, e.g., "12,000", "12,000+", "approx. 18,000").
- district_name: The name of the PUBLIC school district that owns/operates the stadium (e.g., "Allen ISD", "Plano ISD").
- home_high_schools: A list of at least one home high school that uses the stadium, as named in the answer.

Additionally, extract source URLs that are explicitly present in the answer text. Only include actual URLs:
- capacity_sources: URL(s) that explicitly state the stadium capacity.
- location_sources: URL(s) that show the address, city, or county of the stadium.
- ownership_sources: URL(s) that confirm that the stadium is owned/operated by the public school district.
- uil_classification_sources: URL(s) that confirm a listed home high school is classified as UIL Class 5A or Class 6A for 2026–2028.

Return a JSON object:
{
  "stadiums": [
    {
      "official_name": ...,
      "address": ...,
      "seating_capacity": ...,
      "district_name": ...,
      "home_high_schools": [...],
      "capacity_sources": [...],
      "location_sources": [...],
      "ownership_sources": [...],
      "uil_classification_sources": [...]
    },
    ...
  ]
}

Rules:
- Do NOT invent or infer any data not explicitly in the answer.
- If a field is missing for a stadium, set it to null (for strings) or [] (for lists).
- For URLs, extract only valid, explicit URLs shown in the answer (including markdown links).
"""


# -----------------------------------------------------------------------------
# Helper Utilities
# -----------------------------------------------------------------------------
def _combine_sources(*lists: List[str]) -> List[str]:
    """Combine multiple URL lists into a unique, ordered list."""
    seen = set()
    combined: List[str] = []
    for lst in lists:
        for url in lst or []:
            if url and url not in seen:
                combined.append(url)
                seen.add(url)
    return combined


def _normalize_name(name: Optional[str]) -> Optional[str]:
    if not name:
        return None
    return " ".join(name.lower().split())


# -----------------------------------------------------------------------------
# Per-Stadium Verification
# -----------------------------------------------------------------------------
async def verify_stadium(
    evaluator: Evaluator,
    parent_node,
    stadium: StadiumEntry,
    index: int,
) -> None:
    """
    Build and run verification checks for a single stadium.
    """
    # Parent node for this stadium (non-critical so partial credit can be assigned per stadium)
    stadium_node = evaluator.add_parallel(
        id=f"stadium_{index+1}",
        desc=f"Stadium entry {index+1} (counts toward the minimum if all critical checks pass)",
        parent=parent_node,
        critical=False,
    )

    # Prepare common sources (used for location/ownership if dedicated list is missing)
    all_common_sources = _combine_sources(
        stadium.location_sources,
        stadium.ownership_sources,
        stadium.capacity_sources,
        stadium.uil_classification_sources,
    )

    # 1) Official stadium name is provided (critical, existence check)
    evaluator.add_custom_node(
        result=bool(stadium.official_name and stadium.official_name.strip()),
        id=f"stadium_{index+1}_official_name_provided",
        desc="Official stadium name is provided",
        parent=stadium_node,
        critical=True,
    )

    # 2) Complete address is provided (critical)
    # Use LLM to judge completeness rather than regex (accepts minor formatting variants)
    address_leaf = evaluator.add_leaf(
        id=f"stadium_{index+1}_complete_address_provided",
        desc="Complete street address is provided (street number, street name, city, state, ZIP)",
        parent=stadium_node,
        critical=True,
    )
    addr_str = stadium.address or ""
    await evaluator.verify(
        claim=f"The following looks like a complete U.S. street address including street number, street name, city, state (TX/Texas), and a 5-digit ZIP code: '{addr_str}'.",
        node=address_leaf,
        additional_instruction="Judge completeness of the address string itself. Minor punctuation or abbreviation differences (e.g., 'TX' vs 'Texas') are acceptable. If any required component is missing, mark as Incorrect.",
    )

    # 3) Location constraint met (critical)
    # Verify via any provided URLs that the stadium is in Collin County or within the immediate DFW metro area.
    location_leaf = evaluator.add_leaf(
        id=f"stadium_{index+1}_location_constraint_met",
        desc="Stadium is located in Collin County, Texas, or in a city within the immediate Dallas–Fort Worth metropolitan area",
        parent=stadium_node,
        critical=True,
    )
    # Use all available sources that may show address/city info
    loc_sources = _combine_sources(stadium.location_sources, stadium.ownership_sources, stadium.capacity_sources)
    await evaluator.verify(
        claim=(
            f"The stadium '{stadium.official_name or 'the stadium'}' is located in Collin County, Texas, "
            f"or in a city that is part of the immediate Dallas–Fort Worth (DFW) metropolitan area."
        ),
        node=location_leaf,
        sources=loc_sources if loc_sources else None,
        additional_instruction=(
            "Use the address/city as shown on the provided page(s). Accept if the page shows a city widely recognized "
            "as part of DFW (e.g., Dallas, Fort Worth, Arlington, Plano, Irving, Garland, Mesquite, Grand Prairie, "
            "McKinney, Frisco, Carrollton, Richardson, Lewisville, Allen, Prosper, Wylie, Rockwall, Coppell, "
            "Grapevine, Southlake, Keller, Flower Mound, The Colony, Little Elm, etc.), or if the page explicitly "
            "indicates Collin County, Texas. Minor variants in city naming are acceptable. If the page(s) do not "
            "support either Collin County or DFW locality, mark as Not Supported."
        ),
    )

    # 4) Seating capacity source provided (critical)
    evaluator.add_custom_node(
        result=bool(stadium.capacity_sources),
        id=f"stadium_{index+1}_capacity_source_provided",
        desc="A verifiable source is provided for the stated seating capacity (official source or reliable stadium database)",
        parent=stadium_node,
        critical=True,
    )

    # 5) Seating capacity is at least 12,000 (critical; verify using capacity sources)
    capacity_leaf = evaluator.add_leaf(
        id=f"stadium_{index+1}_capacity_at_least_12000",
        desc="Seating capacity is stated and is at least 12,000",
        parent=stadium_node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"The stadium '{stadium.official_name or 'the stadium'}' has a seating capacity of at least 12,000.",
        node=capacity_leaf,
        sources=stadium.capacity_sources if stadium.capacity_sources else None,
        additional_instruction=(
            "Confirm from the page(s) that capacity is ≥ 12,000. Treat phrasings like '12,000+', 'approx. 12,000', "
            "'capacity 12,000', or any explicitly ≥12,000 as meeting the threshold. If capacity is clearly below 12,000 "
            "or not stated, mark as Not Supported."
        ),
    )

    # 6) Owning/operating district identified (critical, existence check)
    evaluator.add_custom_node(
        result=bool(stadium.district_name and stadium.district_name.strip()),
        id=f"stadium_{index+1}_owning_operating_district_identified",
        desc="Name of the owning/operating school district is provided",
        parent=stadium_node,
        critical=True,
    )

    # 7) Public district ownership/operation met (critical; verify using ownership sources or common sources)
    ownership_leaf = evaluator.add_leaf(
        id=f"stadium_{index+1}_public_district_ownership_operation_met",
        desc="The stadium is owned and operated by a public school district (not a private school/operator)",
        parent=stadium_node,
        critical=True,
    )
    own_sources = _combine_sources(stadium.ownership_sources, stadium.location_sources, stadium.capacity_sources)
    await evaluator.verify(
        claim=(
            f"The stadium '{stadium.official_name or 'the stadium'}' is owned and operated by the public school district "
            f"named '{stadium.district_name or 'the stated district'}'."
        ),
        node=ownership_leaf,
        sources=own_sources if own_sources else None,
        additional_instruction=(
            "Look for explicit ownership/operation language on the provided sources (district/stadium sites, official "
            "docs, or reputable databases). Accept 'ISD' (Independent School District) entities as public school "
            "districts. If ownership appears to be a private school, private foundation, or otherwise non-public "
            "operator, mark as Not Supported."
        ),
    )

    # 8) At least one home high school identified (critical, existence check)
    evaluator.add_custom_node(
        result=bool(stadium.home_high_schools),
        id=f"stadium_{index+1}_home_high_school_identified",
        desc="At least one UIL member high school that uses the stadium as its home stadium is identified",
        parent=stadium_node,
        critical=True,
    )

    # 9) UIL 5A or 6A (2026–2028) met for at least one identified home HS (critical; verify using UIL/classification sources)
    uil_leaf = evaluator.add_leaf(
        id=f"stadium_{index+1}_uil_5a_or_6a_2026_2028_met",
        desc="At least one identified home high school is classified as UIL Class 5A or Class 6A under the 2026–2028 UIL reclassification",
        parent=stadium_node,
        critical=True,
    )
    school_list_str = ", ".join(stadium.home_high_schools) if stadium.home_high_schools else "the listed home high schools"
    uil_sources = stadium.uil_classification_sources or []
    await evaluator.verify(
        claim=(
            f"At least one of the following high schools ({school_list_str}) is classified as UIL Class 5A or Class 6A "
            f"for the 2026–2028 cycle."
        ),
        node=uil_leaf,
        sources=uil_sources if uil_sources else None,
        additional_instruction=(
            "Prefer UIL official 2026–2028 reclassification lists (PDF/web). Accept credible district/UIL-region "
            "announcements that explicitly refer to the 2026–28 classification. If evidence only shows older cycles "
            "(e.g., 2024–26) without 2026–28 confirmation, mark as Not Supported."
        ),
    )


# -----------------------------------------------------------------------------
# Main Evaluation Entry Point
# -----------------------------------------------------------------------------
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
    Evaluate an answer for the DFW high school stadiums task using the Mind2Web2 framework.
    """
    # Initialize evaluator (set root as NON-CRITICAL to allow mixed children; we enforce the global minimum via a critical child node)
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Independent stadium entries
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

    # 1) Extract structured stadium info from the answer
    extraction = await evaluator.extract(
        prompt=prompt_extract_stadiums(),
        template_class=StadiumExtraction,
        extraction_name="stadium_extraction",
    )

    # Take the first 3 entries (pad with empty entries if fewer)
    selected: List[StadiumEntry] = list(extraction.stadiums[:3])
    while len(selected) < 3:
        selected.append(StadiumEntry())

    # 2) Build verification subtrees for each stadium
    stadium_parent_nodes = []
    for i in range(3):
        await verify_stadium(evaluator, root, selected[i], i)
        # Keep a direct handle for later aggregated scoring lookup
        stadium_parent_nodes.append(evaluator.find_node(f"stadium_{i+1}"))

    # 3) Global minimum check: at least three distinct qualifying stadiums
    # A stadium "qualifies" if its parent stadium node aggregated score == 1.0 (i.e., all critical children passed).
    # Distinctness is based on normalized official stadium name.
    qualifying_names: List[str] = []
    for i, node in enumerate(stadium_parent_nodes):
        if not node:
            continue
        # Ensure child's score/status are computed
        node.compute_score(mutate=True)
        if node.score == 1.0:
            norm_name = _normalize_name(selected[i].official_name)
            if norm_name:
                qualifying_names.append(norm_name)

    distinct_qualifying = len(set(qualifying_names)) >= 3

    # Add critical global node to enforce the "at least three distinct qualifying" rule
    evaluator.add_custom_node(
        result=distinct_qualifying,
        id="minimum_three_distinct_qualify",
        desc="At least three provided stadium entries are distinct and each passes all per-stadium critical checks",
        parent=root,
        critical=True,
    )

    # 4) Return standard evaluation summary
    return evaluator.get_summary()