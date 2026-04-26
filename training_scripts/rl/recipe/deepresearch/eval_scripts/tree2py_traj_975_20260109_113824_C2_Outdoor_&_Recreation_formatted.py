import asyncio
import logging
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


# -----------------------------------------------------------------------------
# Task-specific constants
# -----------------------------------------------------------------------------
TASK_ID = "wilderness_earliest_lottery_2026"
TASK_DESCRIPTION = (
    "You are planning a wilderness backpacking trip in the Western United States for summer 2026 and need to apply for permits. "
    "Many popular wilderness areas require overnight permits obtained through an advanced lottery system on Recreation.gov, with lottery application windows opening in early 2026.\n\n"
    "Identify the wilderness area in the Western United States that has the earliest lottery application opening date among all areas where:\n"
    "- An advanced lottery system is required for overnight wilderness permits during the 2026 season\n"
    "- The lottery is administered through Recreation.gov\n"
    "- The lottery application window opens in February or March 2026\n"
    "- The 2026 lottery dates are confirmed and published (not listed as pending)\n\n"
    "Provide the name of the wilderness area and the specific date when its lottery application window opens."
)


# -----------------------------------------------------------------------------
# Extraction Data Models
# -----------------------------------------------------------------------------
class CandidateArea(BaseModel):
    name: Optional[str] = None
    opening_date: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class SolutionExtraction(BaseModel):
    # The primary solution the answer claims as the earliest
    selected_area: Optional[str] = None
    opening_date: Optional[str] = None  # Keep as free-form string to maximize compatibility
    # URLs that directly support the selected area, its lottery platform, and opening date
    sources: List[str] = Field(default_factory=list)
    # Any other URLs used to compare across areas (roundups, agency posts, etc.)
    comparison_sources: List[str] = Field(default_factory=list)
    # Other areas mentioned with their dates and sources (if the answer provides them)
    competitors: List[CandidateArea] = Field(default_factory=list)


# -----------------------------------------------------------------------------
# Extraction Prompts
# -----------------------------------------------------------------------------
def prompt_extract_solution() -> str:
    return """
    From the answer, extract the single wilderness area the answer identifies as having the earliest lottery application opening date for the 2026 season.
    Return the following fields:
    - selected_area: the exact name of the identified wilderness area
    - opening_date: the specific date (as free-form string) when the lottery application window opens for 2026, as stated in the answer
    - sources: an array of all explicit URLs in the answer that support the selected area's 2026 lottery details (e.g., Recreation.gov lottery page, official agency updates)
    - comparison_sources: an array of any other URLs used to compare across multiple areas (e.g., roundups listing various areas with 2026 dates)
    - competitors: an array of up to 8 other areas mentioned in the answer, each with:
        - name: the other area's name
        - opening_date: the date string for that area's 2026 lottery application opening (if provided)
        - sources: an array of URLs that support that competitor's date/platform
    IMPORTANT:
    - Only extract URLs explicitly present in the answer text; do not invent or infer URLs.
    - Include URLs even if they are in markdown link format; extract the actual URL target.
    - If any field is missing in the answer, return null for that field (or empty array for URL lists).
    """


# -----------------------------------------------------------------------------
# Helper Utilities
# -----------------------------------------------------------------------------
def _unique_urls(*url_lists: List[str]) -> List[str]:
    seen = set()
    ordered: List[str] = []
    for lst in url_lists:
        for u in lst or []:
            if not isinstance(u, str):
                continue
            u = u.strip()
            if not u:
                continue
            if u not in seen:
                seen.add(u)
                ordered.append(u)
    return ordered


def _safe_str(s: Optional[str]) -> str:
    return s or ""


# -----------------------------------------------------------------------------
# Verification Tree Construction and Checks
# -----------------------------------------------------------------------------
async def build_verification_tree_and_verify(evaluator: Evaluator, extraction: SolutionExtraction) -> None:
    # Create a critical sequential node to represent the whole solution
    was_node = evaluator.add_sequential(
        id="Wilderness_Area_Solution",
        desc="Identify the qualifying wilderness area with the earliest lottery application opening date and provide the opening date.",
        parent=evaluator.root,
        critical=True
    )

    # 1) Answer components: both outputs must be present
    answer_components = evaluator.add_parallel(
        id="Answer_Components",
        desc="Answer includes both required outputs.",
        parent=was_node,
        critical=True
    )

    # 1.a) Wilderness area named
    evaluator.add_custom_node(
        result=bool(extraction.selected_area and extraction.selected_area.strip()),
        id="Wilderness_Area_Named",
        desc="The wilderness area name is explicitly provided.",
        parent=answer_components,
        critical=True
    )

    # 1.b) Opening date provided
    evaluator.add_custom_node(
        result=bool(extraction.opening_date and extraction.opening_date.strip()),
        id="Opening_Date_Provided",
        desc="The specific lottery application opening date is explicitly provided.",
        parent=answer_components,
        critical=True
    )

    # Prepare sources for subsequent verifications
    primary_sources = extraction.sources or []
    comparison_sources = extraction.comparison_sources or []
    competitor_sources = []
    for comp in (extraction.competitors or []):
        competitor_sources.extend(comp.sources or [])
    all_sources = _unique_urls(primary_sources, comparison_sources, competitor_sources)

    area_name = _safe_str(extraction.selected_area)
    opening_date = _safe_str(extraction.opening_date)

    # 2) Qualification Verification (critical parallel checks)
    qual_node = evaluator.add_parallel(
        id="Qualification_Verification",
        desc="The identified wilderness area satisfies all stated qualification constraints.",
        parent=was_node,
        critical=True
    )

    # 2.a) Western US location
    western_loc_leaf = evaluator.add_leaf(
        id="Western_US_Location",
        desc="The wilderness area is located in the Western United States.",
        parent=qual_node,
        critical=True
    )
    western_claim = (
        f"The wilderness area '{area_name}' is in the Western United States."
    )
    western_instruction = (
        "Consider the following states as Western US: WA, OR, CA, NV, ID, MT, WY, UT, CO, AZ, NM, AK, HI. "
        "If the provided sources indicate the area is within any of these states, treat this verification as supported."
    )
    await evaluator.verify(
        claim=western_claim,
        node=western_loc_leaf,
        sources=all_sources,
        additional_instruction=western_instruction
    )

    # 2.b) Advanced lottery system for overnight wilderness permits during 2026 season
    lottery_sys_leaf = evaluator.add_leaf(
        id="Advanced_Lottery_System",
        desc="An advanced lottery system is required for overnight wilderness permits during the 2026 season.",
        parent=qual_node,
        critical=True
    )
    lottery_sys_claim = (
        f"For the 2026 season, overnight wilderness permits for '{area_name}' are allocated via an advance lottery system (not first-come, not daily walk-up)."
    )
    lottery_sys_instruction = (
        "Look for explicit mentions of an advance or preseason lottery for overnight wilderness permits specifically for the 2026 season. "
        "It should not be a same-day or general rolling reservation; it must be an advance lottery mechanism for overnight wilderness use."
    )
    await evaluator.verify(
        claim=lottery_sys_claim,
        node=lottery_sys_leaf,
        sources=all_sources,
        additional_instruction=lottery_sys_instruction
    )

    # 2.c) Recreation.gov platform
    recgov_leaf = evaluator.add_leaf(
        id="Recreation_Gov_Platform",
        desc="The lottery is administered through Recreation.gov.",
        parent=qual_node,
        critical=True
    )
    recgov_claim = (
        f"The permit lottery for '{area_name}' is administered through Recreation.gov."
    )
    recgov_instruction = (
        "Verify that application and administration occur on Recreation.gov. "
        "A direct Recreation.gov lottery or permit page for this area is strong evidence."
    )
    await evaluator.verify(
        claim=recgov_claim,
        node=recgov_leaf,
        sources=all_sources,
        additional_instruction=recgov_instruction
    )

    # 2.d) February or March 2026 opening and matches stated date
    feb_mar_leaf = evaluator.add_leaf(
        id="Feb_Mar_2026_Opening",
        desc="The lottery application window opens in February or March 2026.",
        parent=qual_node,
        critical=True
    )
    feb_mar_claim = (
        f"The lottery application window for '{area_name}' opens on '{opening_date}', and this date is in February or March of 2026."
    )
    feb_mar_instruction = (
        "Confirm the specific opening date for the lottery application window is in 2026, and the month is February (02) or March (03). "
        "Interpret phrasing like 'applications open on' or 'lottery opens' as the opening date. "
        "Time zone/time-of-day annotations are acceptable."
    )
    await evaluator.verify(
        claim=feb_mar_claim,
        node=feb_mar_leaf,
        sources=all_sources,
        additional_instruction=feb_mar_instruction
    )

    # 2.e) Confirmed 2026 dates (not pending)
    confirmed_leaf = evaluator.add_leaf(
        id="Confirmed_2026_Dates",
        desc="The 2026 lottery dates are confirmed and published (not pending/TBD).",
        parent=qual_node,
        critical=True
    )
    confirmed_claim = (
        f"For '{area_name}', the 2026 lottery dates are explicitly confirmed and published (not marked pending or TBD)."
    )
    confirmed_instruction = (
        "Look for explicit 2026 date confirmations. If the page uses terms like 'TBD', 'pending', or 'to be announced' for 2026, this should fail."
    )
    await evaluator.verify(
        claim=confirmed_claim,
        node=confirmed_leaf,
        sources=all_sources,
        additional_instruction=confirmed_instruction
    )

    # 3) Earliest opening verification (critical leaf)
    earliest_leaf = evaluator.add_leaf(
        id="Earliest_Opening_Verification",
        desc="Among all wilderness areas that satisfy the stated constraints, the identified area's lottery application opening date is the earliest.",
        parent=was_node,
        critical=True
    )
    earliest_claim = (
        f"Among Western U.S. wilderness areas that (a) require an advance lottery for overnight permits in 2026, (b) use Recreation.gov, "
        f"(c) have confirmed 2026 dates, and (d) open in February or March 2026, '{area_name}' has the earliest lottery application opening date: '{opening_date}'."
    )
    earliest_instruction = (
        "Use ONLY the provided URLs to assess this claim. Prioritize roundup/comparison sources if available. "
        "If multiple qualifier areas and their dates are provided, compare their opening dates explicitly. "
        "If evidence is insufficient to determine earliest among qualifiers, return Incorrect. "
        "Treat ties where the chosen area shares the earliest date with another qualifying area as not strictly earliest (prefer strict earliest)."
    )
    await evaluator.verify(
        claim=earliest_claim,
        node=earliest_leaf,
        sources=all_sources,
        additional_instruction=earliest_instruction
    )


# -----------------------------------------------------------------------------
# Main evaluation entry point
# -----------------------------------------------------------------------------
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
    # Initialize evaluator and root
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.SEQUENTIAL,  # Follow task's sequential logic
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
    extraction: SolutionExtraction = await evaluator.extract(
        prompt=prompt_extract_solution(),
        template_class=SolutionExtraction,
        extraction_name="solution_extraction"
    )

    # Optional custom info about sources
    evaluator.add_custom_info(
        info={
            "selected_area": extraction.selected_area,
            "opening_date": extraction.opening_date,
            "primary_sources_count": len(extraction.sources or []),
            "comparison_sources_count": len(extraction.comparison_sources or []),
            "competitors_count": len(extraction.competitors or []),
        },
        info_type="extraction_stats"
    )

    # Build verification tree and run checks
    await build_verification_tree_and_verify(evaluator, extraction)

    # Return standardized summary
    return evaluator.get_summary()