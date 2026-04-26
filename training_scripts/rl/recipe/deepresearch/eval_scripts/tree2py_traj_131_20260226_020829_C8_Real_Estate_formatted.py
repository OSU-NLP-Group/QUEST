import asyncio
import logging
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator, AggregationStrategy


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "ca_tod_2026"
TASK_DESCRIPTION = (
    "Identify a transit-oriented multifamily housing development project in California that meets all of the following requirements:\n\n"
    "1. The project broke ground or began construction in calendar year 2026\n"
    "2. The project is located within half a mile of a major transit station that qualifies under California's SB 79 (including BART stations, Metro Rail stations, Trolley stations, or high-frequency bus lines with 15-minute peak headways)\n"
    "3. The project includes at least 5 dwelling units\n"
    "4. The project meets a minimum density of 30 dwelling units per acre (or the local zoning minimum if higher)\n"
    "5. The project is classified as multifamily residential or mixed-use with a residential component\n"
    "6. The project includes designated affordable housing units (at any income level: very low, low, or moderate income)\n\n"
    "Provide the following information about the identified project:\n"
    "- Project name\n"
    "- Specific transit station it is near\n"
    "- Total number of residential units planned\n"
    "- Number or percentage of affordable units\n"
    "- Construction start timeframe in 2026\n"
    "- Supporting URL references for verification"
)


# --------------------------------------------------------------------------- #
# Data models                                                                 #
# --------------------------------------------------------------------------- #
class ProjectURLs(BaseModel):
    construction: List[str] = Field(default_factory=list)
    transit_proximity: List[str] = Field(default_factory=list)
    units_total: List[str] = Field(default_factory=list)
    affordable: List[str] = Field(default_factory=list)
    eligibility_other: List[str] = Field(default_factory=list)


class ProjectInfo(BaseModel):
    project_name: Optional[str] = None

    # Location
    location_city: Optional[str] = None
    location_county: Optional[str] = None
    location_state: Optional[str] = None
    address: Optional[str] = None

    # Transit and proximity
    transit_station_name: Optional[str] = None
    transit_type: Optional[str] = None       # e.g., "BART", "Metro Rail", "San Diego Trolley", "Bus"
    transit_line_name: Optional[str] = None  # e.g., "Orange Line", "Route 720"
    proximity_desc: Optional[str] = None     # e.g., "0.3 miles", "adjacent", "across the street"

    # Units and affordability
    total_units: Optional[str] = None        # keep as string to allow "xx-unit" phrasing
    affordable_units: Optional[str] = None   # number or percentage string

    # Timing and density
    construction_start_2026: Optional[str] = None  # e.g., "Jan 2026", "Q3 2026", "broke ground May 2026"
    density_info: Optional[str] = None             # e.g., "35 du/ac", "meets 30 du/ac minimum"

    # Classification
    classification: Optional[str] = None     # e.g., "multifamily", "mixed-use with residential"

    # URLs grouped by claim
    urls: ProjectURLs = Field(default_factory=ProjectURLs)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_project() -> str:
    return """
    Extract the details about a single California transit-oriented multifamily or mixed-use housing project described in the answer. Return exactly one project's details (the first if multiple are mentioned). Use the exact text from the answer without inventing new content.

    Required fields:
    - project_name: The project's official name or commonly used name.
    - location_city: City name if available.
    - location_county: County name if available.
    - location_state: State name if mentioned (should be 'California' or 'CA').
    - address: Street address or intersection if available.

    - transit_station_name: The specific transit station/stop/line named in the answer (e.g., 'MacArthur BART', '7th Street/Metro Center', 'Park & Market Trolley', or a bus line name).
    - transit_type: The transit mode/category as stated or inferable from the answer text; choose one of: 'BART', 'Metro Rail', 'Trolley', 'Bus', or 'Other'.
    - transit_line_name: The line/service name if applicable (e.g., 'Orange Line', 'Blue Line', 'Rapid 215').
    - proximity_desc: How the answer describes proximity/distance (e.g., '0.3 miles', 'within half a mile', 'adjacent', 'across the street', 'short walk').

    - total_units: The total number of residential units planned (string; can be like '150', '150 units', or 'approx. 150').
    - affordable_units: The number or percentage of affordable units (string; e.g., '20%', '30 affordable units', '15 very low-income units').

    - construction_start_2026: The stated timeframe in calendar year 2026 when construction began/broke ground (e.g., 'January 2026', 'Q2 2026', 'broke ground in 2026'). If only '2026' is stated, return '2026'.
    - density_info: Any explicit density info or statement supporting minimum 30 du/ac (e.g., '35 du/ac', 'meets 30 du/ac minimum', or a computable hint like '# units on # acres'). If missing, return null.

    - classification: The classification as stated (e.g., 'multifamily', 'apartment', 'mixed-use with residential'). If the answer indicates it is multifamily or mixed-use with a residential component, capture that exact phrasing.

    URL groupings (extract only URLs explicitly present in the answer and place them into all relevant buckets if they support multiple facts):
    - urls.construction: URLs that support the claim that construction began or broke ground in 2026.
    - urls.transit_proximity: URLs supporting the transit station/line identification and proximity to it (ideally within 0.5 miles).
    - urls.units_total: URLs supporting the total number of residential units.
    - urls.affordable: URLs supporting the number or percentage of affordable units (or that affordable units exist).
    - urls.eligibility_other: URLs supporting remaining eligibility details (California location, density requirement, and classification as multifamily or mixed-use).

    Rules:
    - Do not invent any URLs; only extract those explicitly present in the answer text (including markdown links).
    - Keep field values as strings; if a numeric is present, keep it as text (e.g., '150').
    - If a field is not present, return null for that field. For URL lists, return empty arrays if none are provided.
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def _unique_merge(*url_lists: List[str]) -> List[str]:
    seen = set()
    merged: List[str] = []
    for lst in url_lists:
        for u in lst or []:
            if isinstance(u, str):
                u2 = u.strip()
                if u2 and (u2 not in seen):
                    merged.append(u2)
                    seen.add(u2)
    return merged


def _all_urls(info: ProjectInfo) -> List[str]:
    return _unique_merge(
        info.urls.construction,
        info.urls.transit_proximity,
        info.urls.units_total,
        info.urls.affordable,
        info.urls.eligibility_other,
    )


# --------------------------------------------------------------------------- #
# Verification tree construction and checks                                   #
# --------------------------------------------------------------------------- #
async def build_and_verify_project_tree(evaluator: Evaluator, root, proj: ProjectInfo) -> None:
    # 1) Eligibility Constraints (critical, parallel)
    eligibility = evaluator.add_parallel(
        id="Eligibility_Constraints",
        desc="Project meets all eligibility requirements stated in the question/constraints.",
        parent=root,
        critical=True
    )

    # Helper URL pools
    urls_for_location = _unique_merge(proj.urls.eligibility_other, proj.urls.transit_proximity)
    urls_for_construction = proj.urls.construction
    urls_for_transit = proj.urls.transit_proximity
    urls_for_units = proj.urls.units_total
    urls_for_affordable = proj.urls.affordable
    urls_for_density = _unique_merge(proj.urls.eligibility_other, proj.urls.units_total)

    # a) Located in California
    leaf_loc = evaluator.add_leaf(
        id="Located_In_California",
        desc="Project is located in California.",
        parent=eligibility,
        critical=True
    )
    claim_loc = "The project is located in the State of California."
    await evaluator.verify(
        claim=claim_loc,
        node=leaf_loc,
        sources=urls_for_location or _all_urls(proj),
        additional_instruction="Use the provided URLs to confirm the project site is in California (city/county/state details are acceptable)."
    )

    # b) Construction started in 2026
    leaf_constr = evaluator.add_leaf(
        id="Construction_Started_In_2026",
        desc="Project broke ground or began construction in calendar year 2026.",
        parent=eligibility,
        critical=True
    )
    timeframe = proj.construction_start_2026 or "2026"
    claim_constr = f"The project broke ground or began construction in {timeframe}, which is within calendar year 2026."
    await evaluator.verify(
        claim=claim_constr,
        node=leaf_constr,
        sources=urls_for_construction,
        additional_instruction="Accept phrasings like 'broke ground in 2026' or 'construction started [month/quarter] 2026'. Do not count permits/approvals without construction start."
    )

    # c) Within half a mile of qualifying transit (proximity only)
    leaf_prox = evaluator.add_leaf(
        id="Within_Half_Mile_Of_Qualifying_Transit",
        desc="Project is located within half a mile of a major transit station/stop/line that qualifies under SB 79.",
        parent=eligibility,
        critical=True
    )
    near_name = proj.transit_station_name or "the cited transit station/line"
    prox_phrase = proj.proximity_desc or "within half a mile"
    claim_prox = f"The project site is {prox_phrase} of {near_name} (i.e., within 0.5 miles)."
    await evaluator.verify(
        claim=claim_prox,
        node=leaf_prox,
        sources=urls_for_transit,
        additional_instruction=(
            "Confirm that the sources indicate a distance within 0.5 miles. Explicit distances <= 0.5 miles qualify. "
            "If exact distance is not stated, treat phrases like 'adjacent', 'across the street', 'next to the station', or 'steps from' as within 0.5 miles when clearly implied."
        )
    )

    # d) Transit qualifies under SB 79 (modal/type check)
    leaf_sb79 = evaluator.add_leaf(
        id="Transit_Qualifies_Under_SB79",
        desc="The cited transit station/stop/line is an SB 79-qualifying type (BART, Metro Rail, Trolley, or a high-frequency bus line with 15-minute peak headways).",
        parent=eligibility,
        critical=True
    )
    tt = (proj.transit_type or "").strip()
    line_name = proj.transit_line_name or ""
    station_or_line = proj.transit_station_name or line_name or "the cited transit facility"
    claim_sb79 = (
        f"The cited facility '{station_or_line}' is SB 79 qualifying: either a BART, Metro Rail, or Trolley rail station, "
        f"or a high-frequency bus line with peak headways of 15 minutes or better."
    )
    await evaluator.verify(
        claim=claim_sb79,
        node=leaf_sb79,
        sources=urls_for_transit,
        additional_instruction=(
            "If the sources show the facility is BART, LA Metro Rail, or San Diego Trolley, mark as supported. "
            "If it is a bus service, only mark as supported if the sources indicate peak headways of 15 minutes or better during peak periods."
        )
    )

    # e) At least 5 dwelling units
    leaf_min_units = evaluator.add_leaf(
        id="At_Least_5_Dwelling_Units",
        desc="Project includes at least 5 dwelling units.",
        parent=eligibility,
        critical=True
    )
    units_text = proj.total_units or "the stated total units"
    claim_min_units = f"The project plans {units_text}, which is at least 5 dwelling units."
    await evaluator.verify(
        claim=claim_min_units,
        node=leaf_min_units,
        sources=urls_for_units,
        additional_instruction="Verify the total planned units from the sources. If the number is 5 or greater, this requirement is met."
    )

    # f) Meets minimum density (>= 30 du/ac or higher local minimum)
    leaf_density = evaluator.add_leaf(
        id="Meets_Minimum_Density",
        desc="Project meets a minimum density of 30 dwelling units per acre (or the local zoning minimum if higher).",
        parent=eligibility,
        critical=True
    )
    density_text = proj.density_info or "evidence provided in the sources"
    claim_density = (
        f"Based on {density_text}, the project meets a minimum density of at least 30 dwelling units per acre "
        f"or satisfies a higher local minimum density requirement."
    )
    await evaluator.verify(
        claim=claim_density,
        node=leaf_density,
        sources=urls_for_density,
        additional_instruction=(
            "Support can be: (1) an explicit density value >= 30 du/ac; (2) a statement that it meets/exceeds a minimum of 30 du/ac or a higher local minimum; "
            "(3) computable evidence (e.g., units and site acreage) demonstrating >= 30 du/ac."
        )
    )

    # g) Correct project classification
    leaf_class = evaluator.add_leaf(
        id="Correct_Project_Classification",
        desc="Project is classified as multifamily residential or mixed-use with a residential component.",
        parent=eligibility,
        critical=True
    )
    class_text = proj.classification or "the stated classification"
    claim_class = (
        f"The project is classified as {class_text}, which is either multifamily residential or mixed-use with a residential component."
    )
    await evaluator.verify(
        claim=claim_class,
        node=leaf_class,
        sources=_unique_merge(proj.urls.eligibility_other, urls_for_units, urls_for_affordable),
        additional_instruction="Treat 'apartment', 'multifamily', or 'mixed-use with housing' as qualifying classifications."
    )

    # h) Includes affordable housing units
    leaf_aff = evaluator.add_leaf(
        id="Includes_Affordable_Housing",
        desc="Project includes designated affordable housing units at any income level (very low, low, or moderate).",
        parent=eligibility,
        critical=True
    )
    aff_text = proj.affordable_units or "affordable units are designated"
    claim_aff = f"The project includes designated affordable housing units (e.g., {aff_text})."
    await evaluator.verify(
        claim=claim_aff,
        node=leaf_aff,
        sources=urls_for_affordable,
        additional_instruction="Evidence should indicate the presence of income-restricted/affordable units (any income tier)."
    )

    # 2) Required Output Fields (critical, parallel) - existence checks only
    required_fields = evaluator.add_parallel(
        id="Required_Output_Fields",
        desc="All requested information fields about the identified project are provided.",
        parent=root,
        critical=True
    )
    evaluator.add_custom_node(
        result=bool(proj.project_name and proj.project_name.strip()),
        id="Project_Name_Provided",
        desc="Project name is provided.",
        parent=required_fields,
        critical=True
    )
    evaluator.add_custom_node(
        result=bool(proj.transit_station_name and proj.transit_station_name.strip()),
        id="Specific_Transit_Station_Provided",
        desc="Specific transit station/stop/line the project is near is provided.",
        parent=required_fields,
        critical=True
    )
    evaluator.add_custom_node(
        result=bool(proj.total_units and proj.total_units.strip()),
        id="Total_Residential_Units_Provided",
        desc="Total number of residential units planned is provided.",
        parent=required_fields,
        critical=True
    )
    evaluator.add_custom_node(
        result=bool(proj.affordable_units and proj.affordable_units.strip()),
        id="Affordable_Units_Provided",
        desc="Number or percentage of affordable units is provided.",
        parent=required_fields,
        critical=True
    )
    evaluator.add_custom_node(
        result=bool(proj.construction_start_2026 and proj.construction_start_2026.strip()),
        id="Construction_Start_Timeframe_Provided",
        desc="Construction start timeframe in 2026 is provided.",
        parent=required_fields,
        critical=True
    )

    # 3) Supporting URL References (critical, parallel) - URL presence checks
    urls_group = evaluator.add_parallel(
        id="Supporting_URL_References",
        desc="Supporting URL references are provided for all major claims required by the constraints and requested output fields.",
        parent=root,
        critical=True
    )
    evaluator.add_custom_node(
        result=bool(proj.urls.construction and len(proj.urls.construction) > 0),
        id="URLs_For_Construction_Start_2026",
        desc="At least one supporting URL is provided for the claim that construction began/broke ground in 2026 (or the stated 2026 start timeframe).",
        parent=urls_group,
        critical=True
    )
    evaluator.add_custom_node(
        result=bool(proj.urls.transit_proximity and len(proj.urls.transit_proximity) > 0),
        id="URLs_For_Transit_And_Proximity",
        desc="At least one supporting URL is provided for the cited transit station/line and the claim that the project is within half a mile of it.",
        parent=urls_group,
        critical=True
    )
    evaluator.add_custom_node(
        result=bool(proj.urls.units_total and len(proj.urls.units_total) > 0),
        id="URLs_For_Total_Units",
        desc="At least one supporting URL is provided for the total number of residential units planned.",
        parent=urls_group,
        critical=True
    )
    evaluator.add_custom_node(
        result=bool(proj.urls.affordable and len(proj.urls.affordable) > 0),
        id="URLs_For_Affordable_Units",
        desc="At least one supporting URL is provided for the number or percentage of affordable units.",
        parent=urls_group,
        critical=True
    )
    evaluator.add_custom_node(
        result=bool(proj.urls.eligibility_other and len(proj.urls.eligibility_other) > 0),
        id="URLs_For_Remaining_Eligibility_Claims",
        desc="At least one supporting URL is provided for the remaining eligibility claims stated by the answer (e.g., California location, density, and classification/mixed-use or multifamily, and presence of affordable units).",
        parent=urls_group,
        critical=True
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
    Entry point for evaluating an answer for the California TOD 2026 project identification task.
    """
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Root aggregation
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

    # Extract structured project info from the answer
    project_info = await evaluator.extract(
        prompt=prompt_extract_project(),
        template_class=ProjectInfo,
        extraction_name="project_info"
    )

    # Build verification tree and run verifications
    await build_and_verify_project_tree(evaluator, root, project_info)

    # Return evaluation summary
    return evaluator.get_summary()