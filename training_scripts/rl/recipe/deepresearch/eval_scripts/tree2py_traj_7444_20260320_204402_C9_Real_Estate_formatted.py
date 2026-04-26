import asyncio
import logging
from typing import Any, Dict, List, Optional, Set

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.llm_client.base_client import LLMClient


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "ca_mixed_use_2024_2026"
TASK_DESCRIPTION = """
Identify four major mixed-use real estate development projects in California that were either completed or officially announced between January 2024 and March 2026. Each project must meet ALL of the following requirements:

1. LEED Certification: The project has achieved OR is officially targeting LEED Gold or Platinum certification level.

2. Mixed-Use Characteristics: The development includes at least two of the following component types: residential units, commercial/office space, or retail space.

3. Size Requirement: The total development size is at least 200,000 square feet.

4. Location: The project is located in one of California's four major metropolitan areas: Los Angeles metropolitan area, San Diego metropolitan area, San Francisco Bay Area, or Sacramento metropolitan area.

5. Special Features: The project either (a) includes an affordable housing component, OR (b) qualifies as a transit-oriented development (located within 0.25 miles of a major transit stop or station).

For each of the four projects, provide the following information with supporting URL references:
- Official project name
- Primary developer or development company name
- Completion date or official announcement date
- LEED certification level (achieved or targeted) and current status
- Component types included and their details (e.g., number of residential units, square footage of commercial/office space, square footage of retail space)
- Total project size in square feet
- Specific location (city, metropolitan area, and address or district if available)
- Special feature details (affordable housing percentage or transit proximity information)
- URL references supporting each category of information
"""

ALLOWED_METROS = [
    "Los Angeles metropolitan area",
    "San Diego metropolitan area",
    "San Francisco Bay Area",
    "Sacramento metropolitan area",
]

DATE_RANGE_DESC = "between January 1, 2024 and March 31, 2026 (inclusive)"


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class ProjectInfo(BaseModel):
    # Identification
    project_name: Optional[str] = None
    developer_name: Optional[str] = None
    timeline_date: Optional[str] = None  # e.g., "2025-02-10" or "Feb 2025"
    timeline_type: Optional[str] = None  # "announcement", "completion", "opening", etc.

    # LEED
    leed_level: Optional[str] = None  # "Gold" or "Platinum", possibly with extras (e.g., "LEED v4.1 Gold")
    leed_status: Optional[str] = None  # "achieved", "in-progress", "targeting", "registered", etc.

    # Components
    component_types: List[str] = Field(default_factory=list)  # e.g., ["residential", "office", "retail"]
    residential_units: Optional[str] = None
    office_sqft: Optional[str] = None
    retail_sqft: Optional[str] = None

    # Size
    total_sqft: Optional[str] = None

    # Location
    city: Optional[str] = None
    metro_area: Optional[str] = None
    address_or_district: Optional[str] = None

    # Special features
    special_feature_type: Optional[str] = None  # "affordable_housing" | "transit_oriented" | similar
    special_feature_details: Optional[str] = None

    # URL references per category
    identification_refs: List[str] = Field(default_factory=list)
    leed_refs: List[str] = Field(default_factory=list)
    components_refs: List[str] = Field(default_factory=list)
    size_refs: List[str] = Field(default_factory=list)
    location_refs: List[str] = Field(default_factory=list)
    features_refs: List[str] = Field(default_factory=list)


class ProjectsExtraction(BaseModel):
    projects: List[ProjectInfo] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_projects() -> str:
    return """
    Extract up to four qualifying mixed-use development projects in California that the answer claims were either completed or officially announced between January 2024 and March 2026.

    For each project found in the answer, extract the following fields exactly as written in the answer (use strings for numbers or dates; do not normalize):
    - project_name: official project name
    - developer_name: name of primary developer or development company
    - timeline_date: the completion or official announcement date mentioned (string, any natural format)
    - timeline_type: "announcement", "completion", "opening", "topped out", etc., if specified; otherwise null

    - leed_level: the claimed LEED level (e.g., "LEED Gold", "LEED Platinum", "LEED v4.1 Gold"); if unclear, return the closest phrase
    - leed_status: "achieved", "in-progress", "targeting", "registered", "pursuing", etc., if provided; otherwise null

    - component_types: list of component categories present, choose from ["residential", "office", "commercial", "retail"]; include at least those explicitly cited in the answer
    - residential_units: number of units if given (string), else null
    - office_sqft: office/commercial square footage if given (string), else null
    - retail_sqft: retail square footage if given (string), else null

    - total_sqft: total development size if mentioned (string), else null

    - city: city name in California if provided, else null
    - metro_area: one of ["Los Angeles metropolitan area", "San Diego metropolitan area", "San Francisco Bay Area", "Sacramento metropolitan area"] if the answer claims it; otherwise null
    - address_or_district: specific address, neighborhood, or district if provided; else null

    - special_feature_type: "affordable_housing" if the answer states affordable/Below-Market-Rate/inclusionary units; "transit_oriented" if TOD/within 0.25 miles of a transit station; otherwise null
    - special_feature_details: details about affordable housing %/unit count, or transit station + distance; else null

    - identification_refs: all URLs cited for project identification (name/developer/timeline)
    - leed_refs: all URLs cited for LEED certification level/status
    - components_refs: all URLs cited for mixed-use components and details
    - size_refs: all URLs cited for size/total square footage
    - location_refs: all URLs cited for location (city/metro/address)
    - features_refs: all URLs cited for special features (affordable housing or TOD)

    Return a JSON object: { "projects": [ ... up to four ProjectInfo objects ... ] }.
    If the answer lists more than four projects, only include the first four.
    If fewer than four, include all found; missing projects should not be fabricated.
    For any missing field, return null (or [] for lists).
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _norm_components(components: List[str]) -> Set[str]:
    out = set()
    for c in components or []:
        lc = (c or "").strip().lower()
        if lc in {"residential", "housing", "apartments", "apartment", "condo", "condominium"}:
            out.add("residential")
        elif lc in {"office", "commercial", "office/commercial", "commercial/office", "workspace"}:
            out.add("office")
        elif lc in {"retail", "shops", "shopping"}:
            out.add("retail")
        else:
            # attempt loose normalization
            if "residential" in lc or "apartment" in lc or "condo" in lc:
                out.add("residential")
            elif "office" in lc or "commercial" in lc:
                out.add("office")
            elif "retail" in lc or "shop" in lc:
                out.add("retail")
    return out


def _has_at_least_two_components(components: List[str]) -> bool:
    return len(_norm_components(components)) >= 2


def _exists_nonempty(value: Optional[str]) -> bool:
    return bool(value and str(value).strip())


def _has_any_url(urls: Optional[List[str]]) -> bool:
    return bool(urls and len([u for u in urls if isinstance(u, str) and u.strip()]) > 0)


def _first_n_or_pad(items: List[ProjectInfo], n: int) -> List[ProjectInfo]:
    lst = list(items[:n])
    while len(lst) < n:
        lst.append(ProjectInfo())
    return lst


# --------------------------------------------------------------------------- #
# Verification logic per project                                              #
# --------------------------------------------------------------------------- #
async def verify_project(evaluator: Evaluator, parent_node, project: ProjectInfo, idx: int) -> None:
    proj_prefix = f"P{idx+1}"

    # Create the project node (non-critical: allows partial scoring across projects)
    project_node = evaluator.add_parallel(
        id=f"Project_{idx+1}",
        desc=f"{['First','Second','Third','Fourth'][idx]} qualifying mixed-use development project",
        parent=parent_node,
        critical=False
    )

    # ---------------- Identification (Critical Group) ---------------- #
    identification_node = evaluator.add_parallel(
        id=f"{proj_prefix}_Project_Identification",
        desc="Basic project identification information",
        parent=project_node,
        critical=True
    )

    # Critical prerequisite: Reference URLs exist
    id_ref_exists = evaluator.add_custom_node(
        result=_has_any_url(project.identification_refs),
        id=f"{proj_prefix}_Identification_Reference",
        desc="URL reference supporting project identification details (exists)",
        parent=identification_node,
        critical=True
    )

    # Name verification
    name_node = evaluator.add_leaf(
        id=f"{proj_prefix}_Project_Name",
        desc="Official name of the development project is provided and supported",
        parent=identification_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The official project name is '{project.project_name}'.",
        node=name_node,
        sources=project.identification_refs,
        additional_instruction="Verify that the sources explicitly name the project with a matching or equivalent official name. Allow minor variations."
    )

    # Developer verification
    dev_node = evaluator.add_leaf(
        id=f"{proj_prefix}_Developer_Name",
        desc="Name of the primary developer or development company is identified and supported",
        parent=identification_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The primary developer/development company for the project '{project.project_name}' is '{project.developer_name}'.",
        node=dev_node,
        sources=project.identification_refs,
        additional_instruction="Confirm the developer on the cited source(s). Allow reasonable naming variants (LLC, Inc., etc.)."
    )

    # Timeline verification (in required date range)
    timeline_node = evaluator.add_leaf(
        id=f"{proj_prefix}_Timeline",
        desc=f"Project completion or announcement date falls {DATE_RANGE_DESC}",
        parent=identification_node,
        critical=True
    )
    tl_kind = project.timeline_type or "announcement or completion"
    tl_detail = f" on {project.timeline_date}" if _exists_nonempty(project.timeline_date) else ""
    await evaluator.verify(
        claim=f"The project was {tl_kind}{tl_detail}, which falls {DATE_RANGE_DESC}.",
        node=timeline_node,
        sources=project.identification_refs,
        additional_instruction="Accept synonyms like 'opening', 'grand opening', 'topped out' only if they clearly signal completion/announcement timing. Focus on the date range 2024-01-01 to 2026-03-31 inclusive."
    )

    # ---------------- LEED Certification (Critical Group) ---------------- #
    leed_node = evaluator.add_parallel(
        id=f"{proj_prefix}_LEED_Certification",
        desc="LEED certification status and level",
        parent=project_node,
        critical=True
    )

    # Critical prerequisite: LEED references exist
    leed_ref_exists = evaluator.add_custom_node(
        result=_has_any_url(project.leed_refs),
        id=f"{proj_prefix}_LEED_Reference",
        desc="URL reference confirming LEED certification status and level (exists)",
        parent=leed_node,
        critical=True
    )

    # Level: must be Gold or Platinum (achieved or targeting)
    leed_level_node = evaluator.add_leaf(
        id=f"{proj_prefix}_LEED_Level",
        desc="Project has achieved or is targeting LEED Gold or Platinum certification",
        parent=leed_node,
        critical=True
    )
    await evaluator.verify(
        claim="The project targets or has achieved LEED Gold or LEED Platinum certification.",
        node=leed_level_node,
        sources=project.leed_refs,
        additional_instruction="Accept formulations like 'LEED v4/v4.1 Gold/Platinum', 'pursuing Gold', 'registered for Gold', etc. The key is Gold or Platinum level is explicitly stated."
    )

    # Status provided (achieved / in-progress / targeted / registered)
    leed_status_node = evaluator.add_leaf(
        id=f"{proj_prefix}_LEED_Status",
        desc="Current certification status is clearly indicated (achieved, in-progress, or targeted)",
        parent=leed_node,
        critical=True
    )
    await evaluator.verify(
        claim="The sources clearly indicate the project's current LEED certification status (achieved, in-progress, targeting, or registered).",
        node=leed_status_node,
        sources=project.leed_refs,
        additional_instruction="Look for explicit phrases like 'achieved LEED Gold', 'targeting LEED Platinum', 'registered to pursue LEED', 'in progress', etc."
    )

    # ---------------- Mixed-Use Components ---------------- #
    # Core (Critical): at least two component types + have supporting refs
    components_core = evaluator.add_parallel(
        id=f"{proj_prefix}_Mixed_Use_Components_Core",
        desc="Verification of mixed-use core characteristics (types + refs)",
        parent=project_node,
        critical=True
    )

    # Critical prerequisite: Components references exist
    comps_ref_exists = evaluator.add_custom_node(
        result=_has_any_url(project.components_refs),
        id=f"{proj_prefix}_Components_Reference",
        desc="URL reference confirming mixed-use components and details (exists)",
        parent=components_core,
        critical=True
    )

    # At least two component categories (residential, office/commercial, retail)
    comp_types_node = evaluator.add_leaf(
        id=f"{proj_prefix}_Component_Types",
        desc="Project includes at least two of: residential units, commercial/office space, or retail space",
        parent=components_core,
        critical=True
    )
    await evaluator.verify(
        claim="The project includes at least two of the following components: residential, office/commercial, and retail.",
        node=comp_types_node,
        sources=project.components_refs,
        additional_instruction="Treat synonyms loosely (e.g., 'apartments/condos' => residential; 'commercial' => office; 'ground-floor shops' => retail)."
    )

    # Optional (Non-Critical): numeric/detail provision checks
    components_optional = evaluator.add_parallel(
        id=f"{proj_prefix}_Mixed_Use_Components_Details",
        desc="Optional component detail checks (non-critical)",
        parent=project_node,
        critical=False
    )

    norm_types = _norm_components(project.component_types)

    # Residential details provided if residential exists
    evaluator.add_custom_node(
        result=(("residential" not in norm_types) or _exists_nonempty(project.residential_units)),
        id=f"{proj_prefix}_Residential_Details",
        desc="If residential component exists, number of units is provided",
        parent=components_optional,
        critical=False
    )

    # Commercial/office details provided if office exists
    evaluator.add_custom_node(
        result=(("office" not in norm_types) or _exists_nonempty(project.office_sqft)),
        id=f"{proj_prefix}_Commercial_Details",
        desc="If commercial/office component exists, square footage is provided",
        parent=components_optional,
        critical=False
    )

    # Retail details provided if retail exists
    evaluator.add_custom_node(
        result=(("retail" not in norm_types) or _exists_nonempty(project.retail_sqft)),
        id=f"{proj_prefix}_Retail_Details",
        desc="If retail component exists, square footage is provided",
        parent=components_optional,
        critical=False
    )

    # ---------------- Size Requirements ---------------- #
    size_core = evaluator.add_parallel(
        id=f"{proj_prefix}_Size_Requirements_Core",
        desc="Total development size verification (core)",
        parent=project_node,
        critical=True
    )

    # Critical prerequisite: Size references exist
    size_ref_exists = evaluator.add_custom_node(
        result=_has_any_url(project.size_refs),
        id=f"{proj_prefix}_Size_Reference",
        desc="URL reference confirming size specifications (exists)",
        parent=size_core,
        critical=True
    )

    # Total size >= 200,000 sf
    total_sf_node = evaluator.add_leaf(
        id=f"{proj_prefix}_Total_Square_Footage",
        desc="Total development size is at least 200,000 square feet",
        parent=size_core,
        critical=True
    )
    await evaluator.verify(
        claim="The project's total development size is at least 200,000 square feet.",
        node=total_sf_node,
        sources=project.size_refs,
        additional_instruction="Accept GSF/GFA or similar if clearly equivalent. If multiple phases, confirm the combined/total exceeds 200,000 sf."
    )

    # Optional: Size calculation/breakdown clarity
    size_optional = evaluator.add_parallel(
        id=f"{proj_prefix}_Size_Requirements_Details",
        desc="Size calculation or breakdown clarity (non-critical)",
        parent=project_node,
        critical=False
    )
    size_calc_node = evaluator.add_leaf(
        id=f"{proj_prefix}_Size_Calculation",
        desc="Calculation method or breakdown of total square footage is clear",
        parent=size_optional,
        critical=False
    )
    await evaluator.verify(
        claim="The sources provide a clear calculation or breakdown supporting the stated total square footage (e.g., component areas or phasing totals).",
        node=size_calc_node,
        sources=project.size_refs,
        additional_instruction="Look for tables, bullet lists, or sentences attributing sf to components or phases summing to the total."
    )

    # ---------------- Location Verification ---------------- #
    location_core = evaluator.add_parallel(
        id=f"{proj_prefix}_Location_Verification_Core",
        desc="Location requirements compliance (core)",
        parent=project_node,
        critical=True
    )

    # Critical prerequisite: Location references exist
    loc_ref_exists = evaluator.add_custom_node(
        result=_has_any_url(project.location_refs),
        id=f"{proj_prefix}_Location_Reference",
        desc="URL reference confirming location details (exists)",
        parent=location_core,
        critical=True
    )

    # City & State (CA)
    city_state_node = evaluator.add_leaf(
        id=f"{proj_prefix}_City_State",
        desc="Specific city and state (California) are identified",
        parent=location_core,
        critical=True
    )
    city_part = f"in {project.city}, California" if _exists_nonempty(project.city) else "in California"
    await evaluator.verify(
        claim=f"The project is located {city_part}.",
        node=city_state_node,
        sources=project.location_refs,
        additional_instruction="Confirm the project is in California and identify the specific city if present on the page."
    )

    # Metro Area requirement
    metro_desc = project.metro_area if _exists_nonempty(project.metro_area) else "one of the required California metropolitan areas"
    metro_node = evaluator.add_leaf(
        id=f"{proj_prefix}_Metro_Area",
        desc="Project is located in Los Angeles, San Diego, San Francisco Bay Area, or Sacramento metropolitan area",
        parent=location_core,
        critical=True
    )
    await evaluator.verify(
        claim="The project is located in one of the following metropolitan areas: Los Angeles, San Diego, San Francisco Bay Area, or Sacramento.",
        node=metro_node,
        sources=project.location_refs,
        additional_instruction="Check the page for geographic context (city/region) to conclude which metro area applies; accept recognized synonyms like 'Greater Los Angeles', 'Bay Area', etc."
    )

    # Optional: Address or district details
    location_optional = evaluator.add_parallel(
        id=f"{proj_prefix}_Location_Verification_Details",
        desc="Address or district details (non-critical)",
        parent=project_node,
        critical=False
    )
    addr_node = evaluator.add_leaf(
        id=f"{proj_prefix}_Address_Details",
        desc="Specific address or district location is provided",
        parent=location_optional,
        critical=False
    )
    await evaluator.verify(
        claim="The sources provide a specific site address or a clearly named neighborhood/district for the project.",
        node=addr_node,
        sources=project.location_refs,
        additional_instruction="Look for street addresses or district names (e.g., Arts District, Mission Bay)."
    )

    # ---------------- Special Features ---------------- #
    features_core = evaluator.add_parallel(
        id=f"{proj_prefix}_Special_Features_Core",
        desc="Additional qualifying features verification (core)",
        parent=project_node,
        critical=True
    )

    # Critical prerequisite: Features references exist
    feat_ref_exists = evaluator.add_custom_node(
        result=_has_any_url(project.features_refs),
        id=f"{proj_prefix}_Features_Reference",
        desc="URL reference confirming special features (exists)",
        parent=features_core,
        critical=True
    )

    # Affordable housing OR Transit-Oriented Development within 0.25 miles
    feature_main_node = evaluator.add_leaf(
        id=f"{proj_prefix}_Affordable_Housing_OR_TOD",
        desc="Project includes affordable housing component OR qualifies as TOD (within 0.25 miles of transit)",
        parent=features_core,
        critical=True
    )
    await evaluator.verify(
        claim="The project includes an affordable housing component OR is transit-oriented (located within 0.25 miles of a major transit stop/station).",
        node=feature_main_node,
        sources=project.features_refs,
        additional_instruction="Affordable synonyms include 'below-market-rate (BMR)', 'deed-restricted', 'inclusionary'. TOD evidence can include adjacency to BART/Muni/Metro/Amtrak/Caltrain and explicit distance within 0.25 miles."
    )

    # Optional: specific details (percentage, station name/distance)
    features_optional = evaluator.add_parallel(
        id=f"{proj_prefix}_Special_Features_Details",
        desc="Special feature details (non-critical)",
        parent=project_node,
        critical=False
    )
    feat_details_node = evaluator.add_leaf(
        id=f"{proj_prefix}_Feature_Details",
        desc="Specific details about affordable housing percentage or transit proximity are provided",
        parent=features_optional,
        critical=False
    )
    await evaluator.verify(
        claim="The sources provide specific details, such as the affordable housing percentage/unit count or the exact transit station and approximate walking distance (or 'within 0.25 miles').",
        node=feat_details_node,
        sources=project.features_refs,
        additional_instruction="Look for numeric percentages (e.g., 15% affordable) or precise station/distance mentions."
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
) -> Dict:
    """
    Evaluate an answer for the California mixed-use developments (2024–2026) task.
    """

    evaluator = Evaluator()
    # Note: Although the original rubric marked the root as critical,
    # to comply with framework constraints (critical parent requires all children critical),
    # we keep the root non-critical to allow partial credit across projects.
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

    # Record task constraints as custom info
    evaluator.add_custom_info(
        info={
            "allowed_metro_areas": ALLOWED_METROS,
            "required_date_range": DATE_RANGE_DESC,
            "min_total_size_sqft": 200_000
        },
        info_type="task_constraints",
        info_name="constraints"
    )

    # 1) Extract up to four projects
    extraction: ProjectsExtraction = await evaluator.extract(
        prompt=prompt_extract_projects(),
        template_class=ProjectsExtraction,
        extraction_name="projects_extraction"
    )

    # Ensure exactly four entries (pad with empty)
    projects = _first_n_or_pad(extraction.projects or [], 4)

    # 2) Build verification tree for each project
    # Parent container for the task (non-critical root child to encapsulate all projects)
    task_parent = evaluator.add_parallel(
        id="California_Mixed_Use_Developments_Task",
        desc="Evaluate four major CA mixed-use development projects (2024–2026) against sustainability, size, and location criteria",
        parent=root,
        critical=False
    )

    # Verify each project subtree
    for i in range(4):
        await verify_project(evaluator, task_parent, projects[i], i)

    # 3) Return summary
    return evaluator.get_summary()