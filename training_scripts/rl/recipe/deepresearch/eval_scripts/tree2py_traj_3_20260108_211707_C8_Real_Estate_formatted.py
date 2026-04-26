import asyncio
import logging
import re
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.llm_client.base_client import LLMClient


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "four_mixed_use_developments"
TASK_DESCRIPTION = """Identify four large-scale mixed-use real estate developments currently under construction or planned for completion between 2024 and 2027, with each development located in a different state among Texas, North Carolina, Arizona, or California. Each development must meet all of the following criteria:

1. Be a mixed-use development that combines both residential and commercial/retail components
2. Include at least 250 residential units
3. Include at least 20,000 square feet of commercial or retail space
4. Have a total project development cost of at least $300 million
5. Include at least one public amenity (such as a park, school, public space, or community facility)

For each development, provide:
- The project name
- The specific city and state location
- The number of residential units
- The amount of commercial/retail space (in square feet)
- The total project cost
- The expected completion timeline
- Description of the public amenity(ies) included
- A reference URL that verifies these details

Ensure that each of the four developments is located in a different state."""

ALLOWED_STATE_CANONICALS = {
    "texas": "Texas",
    "north carolina": "North Carolina",
    "arizona": "Arizona",
    "california": "California",
}
STATE_ABBREV_MAP = {
    "tx": "Texas",
    "nc": "North Carolina",
    "az": "Arizona",
    "ca": "California",
}


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class Development(BaseModel):
    project_name: Optional[str] = None
    city: Optional[str] = None
    state: Optional[str] = None
    residential_units: Optional[str] = None  # keep as string for flexibility
    commercial_sqft: Optional[str] = None    # keep as string (may include commas, ranges)
    total_cost: Optional[str] = None         # keep as string (e.g., "$300M", "$0.5B", "USD 325 million")
    completion_timeline: Optional[str] = None
    public_amenities: List[str] = Field(default_factory=list)
    reference_urls: List[str] = Field(default_factory=list)


class DevelopmentsExtraction(BaseModel):
    developments: List[Development] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_developments() -> str:
    return """
    Extract up to six mixed-use developments described in the answer. For each development, extract the following fields strictly from the answer text:

    - project_name: string (the development's official or commonly used name)
    - city: string (city where the development is located)
    - state: string (state where the development is located; can be full name or postal abbreviation)
    - residential_units: string (the stated number of residential units; preserve the exact text, e.g., "275", "approx. 300", "250-300")
    - commercial_sqft: string (the stated amount of commercial/retail square footage; preserve the exact text, e.g., "25,000 sq ft", "20k", "40,000 SF")
    - total_cost: string (the stated total development cost; preserve the exact text, e.g., "$350M", "$0.5B", "USD 400 million")
    - completion_timeline: string (the stated expected timeline; preserve the exact text, e.g., "2025", "2026-2027", "under construction with completion in 2025")
    - public_amenities: array of strings (each public amenity explicitly mentioned in the answer; examples: "park", "plaza", "school", "library", "community center")
    - reference_urls: array of strings (all URLs explicitly provided for this development; include any relevant press releases, developer pages, city planning documents, or news coverage)

    Return a JSON object with a 'developments' array of objects with these fields.
    If a field is missing for a specific development, set it to null (for strings) or [] (for arrays).
    Only include URLs that actually appear in the answer text (including markdown links).
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def normalize_state_name(state_text: Optional[str]) -> Optional[str]:
    if not state_text:
        return None
    s = state_text.strip().lower().replace(".", "")
    # handle common abbreviations
    if s in STATE_ABBREV_MAP:
        return STATE_ABBREV_MAP[s]
    # normalize "n c", "n. c." etc.
    s = re.sub(r"\s+", " ", s)
    s = s.replace("northcarolina", "north carolina")
    # canonical full names
    if s in ALLOWED_STATE_CANONICALS:
        return ALLOWED_STATE_CANONICALS[s]
    # title-case fallback (not strictly enforced)
    return state_text.strip().title()


def is_allowed_state(state_text: Optional[str]) -> bool:
    norm = normalize_state_name(state_text)
    if not norm:
        return False
    return norm in ALLOWED_STATE_CANONICALS.values()


def non_empty(s: Optional[str]) -> bool:
    return bool(s and s.strip())


def at_least_one_url(urls: Optional[List[str]]) -> bool:
    if not urls:
        return False
    valid = [u for u in urls if isinstance(u, str) and len(u.strip()) > 0]
    return len(valid) > 0


def join_list(values: List[str]) -> str:
    if not values:
        return ""
    if len(values) == 1:
        return values[0]
    return ", ".join(values[:-1]) + f", and {values[-1]}"


def states_are_all_distinct(projects: List[Development]) -> bool:
    # Consider only the first 4 developments (as evaluated)
    first_four = projects[:4]
    norm_states = []
    for p in first_four:
        norm_states.append(normalize_state_name(p.state) or "")
    # Need exactly 4 non-empty distinct states
    non_empty_states = [s for s in norm_states if s]
    return len(non_empty_states) == 4 and len(set(non_empty_states)) == 4


# --------------------------------------------------------------------------- #
# Verification for a single development                                       #
# --------------------------------------------------------------------------- #
async def verify_single_project(
    evaluator: Evaluator,
    parent_node,
    project: Development,
    project_index: int,
) -> None:
    """
    Build the verification subtree for a single project and run verifications.
    Node IDs mirror the rubric leaf names as much as possible.
    """
    proj_node = evaluator.add_parallel(
        id=f"Project_{project_index + 1}",
        desc=f"Development {project_index + 1} details and eligibility checks.",
        parent=parent_node,
        critical=False,  # Non-critical per rubric, allows partial credit per project
    )

    # --- Basic presence and validity checks (custom boolean leaves) ---
    name_ok = evaluator.add_custom_node(
        result=non_empty(project.project_name),
        id=f"Project_{project_index + 1}_Name",
        desc="Project name is provided.",
        parent=proj_node,
        critical=True,
    )

    city_state_ok = evaluator.add_custom_node(
        result=(non_empty(project.city) and non_empty(project.state)),
        id=f"Project_{project_index + 1}_City_State_Location",
        desc="Specific city and state location are provided.",
        parent=proj_node,
        critical=True,
    )

    state_allowed_ok = evaluator.add_custom_node(
        result=is_allowed_state(project.state),
        id=f"Project_{project_index + 1}_State_Allowed",
        desc="Project state is one of: Texas, North Carolina, Arizona, California.",
        parent=proj_node,
        critical=True,
    )

    # Reference URL leaf - first ensure at least one URL exists; then verify support
    ref_leaf = evaluator.add_leaf(
        id=f"Project_{project_index + 1}_Reference_URL_Verifies_Details",
        desc="At least one reference URL is provided that supports the reported project details (location, mixed-use nature, units, commercial/retail space, cost, timeline, and amenity).",
        parent=proj_node,
        critical=True,
    )

    sources = project.reference_urls or []

    # If no sources at all, immediately fail this leaf to prevent unsupported verifications
    if not at_least_one_url(sources):
        ref_leaf.score = 0.0
        ref_leaf.status = "failed"
    else:
        # Compose a comprehensive claim that the reference supports the listed key facts.
        claim_details_parts = []
        if non_empty(project.project_name):
            claim_details_parts.append(f"name '{project.project_name}'")
        if non_empty(project.city) or non_empty(project.state):
            loc = f"{project.city or ''}, {normalize_state_name(project.state) or (project.state or '')}".strip().strip(", ")
            claim_details_parts.append(f"location in {loc}")
        claim_details_parts.append("mixed-use combining residential and commercial/retail")
        claim_details_parts.append("at least 250 residential units")
        claim_details_parts.append("at least 20,000 square feet of commercial/retail space")
        claim_details_parts.append("total development cost at least $300 million")
        claim_details_parts.append("under construction or planned for completion between 2024 and 2027")
        if project.public_amenities:
            claim_details_parts.append(f"public amenity/amenities such as {join_list(project.public_amenities)}")
        else:
            claim_details_parts.append("at least one public amenity")

        full_claim = (
            f"For the development"
            f"{f' \"{project.project_name}\"' if non_empty(project.project_name) else ''}"
            f"{f' located in {project.city}, {normalize_state_name(project.state)}' if (non_empty(project.city) or non_empty(project.state)) else ''}, "
            f"the reference page(s) support all of the following facts: "
            + "; ".join(claim_details_parts)
            + "."
        )

        await evaluator.verify(
            claim=full_claim,
            node=ref_leaf,
            sources=sources,
            additional_instruction=(
                "Verify that the provided page(s) explicitly support the listed details for this project. "
                "Allow minor formatting differences (e.g., commas in numbers), rounding, or synonyms (e.g., 'retail' vs 'commercial'). "
                "If some details are clearly contradicted or absent, judge as not supported."
            ),
        )

    # --- Criterion: Mixed Use ---
    mixed_use_leaf = evaluator.add_leaf(
        id=f"Project_{project_index + 1}_Mixed_Use",
        desc="Project is mixed-use and includes both residential and commercial/retail components.",
        parent=proj_node,
        critical=True,
    )
    # Only run verification if references are present (ref leaf likely handles support)
    await evaluator.verify(
        claim=(
            "This development is a mixed-use project that includes both residential and commercial/retail components."
        ),
        node=mixed_use_leaf,
        sources=sources,
        additional_instruction=(
            "Check the page for explicit language that it is mixed-use and contains both residential and "
            "commercial/retail (or equivalent, e.g., shops, retail podium, offices) components."
        ),
    )

    # --- Criterion: Units >= 250 ---
    units_leaf = evaluator.add_leaf(
        id=f"Project_{project_index + 1}_Residential_Units_Min_And_Value",
        desc="Residential unit count is provided and is at least 250 units.",
        parent=proj_node,
        critical=True,
    )
    await evaluator.verify(
        claim=(
            f"The development includes at least 250 residential units."
            f"{f' The answer states: {project.residential_units}.' if non_empty(project.residential_units) else ''}"
        ),
        node=units_leaf,
        sources=sources,
        additional_instruction=(
            "Look for a statement of the number of residential units. If a range is given, the lower bound should be >= 250. "
            "If phases are clearly part of the same development and the unit total is given collectively, totals >= 250 count as meeting the requirement."
        ),
    )

    # --- Criterion: Commercial/Retail >= 20,000 sq ft ---
    commercial_leaf = evaluator.add_leaf(
        id=f"Project_{project_index + 1}_Commercial_Space_Min_And_Value",
        desc="Commercial/retail square footage is provided and is at least 20,000 sq ft.",
        parent=proj_node,
        critical=True,
    )
    await evaluator.verify(
        claim=(
            "The development includes at least 20,000 square feet of commercial or retail space."
            f"{f' The answer states: {project.commercial_sqft}.' if non_empty(project.commercial_sqft) else ''}"
        ),
        node=commercial_leaf,
        sources=sources,
        additional_instruction=(
            "Look for commercial/retail square footage. Numbers like '20,000 SF', '20k sf', or 'over 20,000 square feet' should count as meeting the threshold."
        ),
    )

    # --- Criterion: Total cost >= $300 million ---
    cost_leaf = evaluator.add_leaf(
        id=f"Project_{project_index + 1}_Total_Cost_Min_And_Value",
        desc="Total project development cost is provided and is at least $300 million.",
        parent=proj_node,
        critical=True,
    )
    await evaluator.verify(
        claim=(
            "The total development cost is at least $300 million."
            f"{f' The answer states: {project.total_cost}.' if non_empty(project.total_cost) else ''}"
        ),
        node=cost_leaf,
        sources=sources,
        additional_instruction=(
            "Interpret common formats such as '$300M', 'USD 0.3B', '$0.3 billion', etc., as equivalent. "
            "If the page shows cost clearly below $300M, judge as not meeting the requirement."
        ),
    )

    # --- Criterion: Timeline 2024-2027 or Under Construction ---
    timeline_leaf = evaluator.add_leaf(
        id=f"Project_{project_index + 1}_Timeline_Constraint_And_Value",
        desc="Expected completion timeline is provided, and the project is under construction or planned for completion between 2024 and 2027.",
        parent=proj_node,
        critical=True,
    )
    await evaluator.verify(
        claim=(
            "The project is under construction or is planned for completion between 2024 and 2027 inclusive."
            f"{f' The answer states: {project.completion_timeline}.' if non_empty(project.completion_timeline) else ''}"
        ),
        node=timeline_leaf,
        sources=sources,
        additional_instruction=(
            "Accept phrases like 'under construction', 'construction underway', or clear stated completion in 2024, 2025, 2026, or 2027. "
            "If only phases are given, any substantial phase delivering within 2024-2027 counts as meeting the timeline requirement."
        ),
    )

    # --- Criterion: Public amenity present ---
    amenity_leaf = evaluator.add_leaf(
        id=f"Project_{project_index + 1}_Public_Amenity",
        desc="At least one public amenity is described (e.g., park, school, public space, community facility).",
        parent=proj_node,
        critical=True,
    )
    amenity_clause = (
        f" such as {join_list(project.public_amenities)}" if project.public_amenities else ""
    )
    await evaluator.verify(
        claim=f"The development includes at least one public amenity{amenity_clause}.",
        node=amenity_leaf,
        sources=sources,
        additional_instruction=(
            "Public amenities include publicly accessible parks, plazas, schools, libraries, community centers, or comparable civic/public spaces. "
            "If the page explicitly mentions at least one such amenity, judge as supported."
        ),
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
    model: str = "o4-mini",
) -> Dict[str, Any]:
    """
    Evaluate an answer for the 'four_mixed_use_developments' task using obj_task_eval framework.
    """
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # overall evaluation considers independent sub-parts
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
        prompt=prompt_extract_developments(),
        template_class=DevelopmentsExtraction,
        extraction_name="developments_extraction",
    )

    # Record constraints info to help debugging
    evaluator.add_custom_info(
        info={
            "allowed_states": list(ALLOWED_STATE_CANONICALS.values()),
            "min_units": 250,
            "min_commercial_sqft": 20000,
            "min_total_cost_millions": 300,
            "timeline_years_inclusive": [2024, 2027],
        },
        info_type="constraints",
    )

    # Main rubric node (non-critical here to allow partial scoring across projects)
    main_node = evaluator.add_parallel(
        id="Four_Mixed_Use_Developments",
        desc="Identify four large-scale mixed-use developments meeting the stated constraints and provide the requested details and references.",
        parent=root,
        critical=False,
    )

    # Select up to four projects from the answer (pad with empty if fewer)
    projects: List[Development] = extracted.developments[:4]
    while len(projects) < 4:
        projects.append(Development())

    # Cross-project distinct states (critical leaf under main node)
    cross_states_node = evaluator.add_custom_node(
        result=states_are_all_distinct(projects),
        id="Cross_Project_Distinct_States",
        desc="The four developments are located in four different US states (i.e., the states reported for Projects 1–4 are all distinct).",
        parent=main_node,
        critical=True,
    )

    # Per-project verification
    for idx in range(4):
        await verify_single_project(evaluator, main_node, projects[idx], idx)

    # Return structured summary
    return evaluator.get_summary()