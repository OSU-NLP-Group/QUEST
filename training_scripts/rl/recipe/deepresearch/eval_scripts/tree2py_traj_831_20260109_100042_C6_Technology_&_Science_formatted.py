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
TASK_ID = "battery_facility_constraints"
TASK_DESCRIPTION = """
Identify the name of the battery manufacturing facility in the United States that satisfies all of the following criteria:

1. The facility is located in a U.S. state that borders the Mississippi River.
2. The facility is operated as a joint venture between at least two separate companies.
3. The facility is designed to produce Lithium Iron Phosphate (LFP) battery cells.
4. The facility's planned annual production capacity is between 20 gigawatt-hours (GWh) and 25 GWh (inclusive).
5. The facility site occupies at least 400 acres.
6. The main manufacturing building is at least 1.5 million square feet in area.
7. Ground was broken on the facility construction between January 2024 and December 2024 (inclusive).
8. The facility's planned initial production start date is in 2027 or later.
9. At least one of the joint venture partners is a company primarily involved in commercial vehicle (truck) manufacturing or commercial vehicle powertrain systems.
10. The facility is located within a designated industrial park or megasite.

Provide the official name of the facility.
"""

# States bordering the Mississippi River (for reference and simple checks)
MISSISSIPPI_BORDER_STATES = {
    "Minnesota", "Wisconsin", "Iowa", "Illinois", "Missouri",
    "Kentucky", "Tennessee", "Arkansas", "Mississippi", "Louisiana"
}


# --------------------------------------------------------------------------- #
# Data models for structured extraction                                       #
# --------------------------------------------------------------------------- #
class FacilityExtraction(BaseModel):
    """Structured information extracted from the agent's answer about the facility."""
    official_name: Optional[str] = None
    location_city_or_county: Optional[str] = None
    location_state: Optional[str] = None
    industrial_park_or_megasite: Optional[str] = None
    joint_venture_partners: List[str] = Field(default_factory=list)
    cell_chemistry: Optional[str] = None  # e.g., "LFP", "Lithium iron phosphate"
    capacity_planned_gwh: Optional[str] = None  # keep as string to allow ranges/text
    site_acres: Optional[str] = None  # keep as string to allow ranges/text
    main_building_sqft: Optional[str] = None  # keep as string to allow ranges/text
    groundbreaking_date: Optional[str] = None  # e.g., "September 2024"
    production_start_year: Optional[str] = None  # e.g., "2027"
    sources: List[str] = Field(default_factory=list)  # URLs cited in the answer


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_facility_info() -> str:
    return """
    Extract structured information about the single U.S. battery manufacturing facility the answer is referring to.
    Return a JSON object with the following fields exactly (use null when not present in the answer):

    - official_name: The official facility name (e.g., "XYZ Battery Cell Manufacturing Plant"). Do not return a vague description; return the proper name string if present, otherwise null.
    - location_city_or_county: City or county where the facility is located, if specified in the answer; otherwise null.
    - location_state: The U.S. state where the facility is located, as written in the answer; otherwise null.
    - industrial_park_or_megasite: The designated industrial park or megasite name (if provided); otherwise null.
    - joint_venture_partners: Array of JV partner company names mentioned (each as a string). If not mentioned, return an empty array.
    - cell_chemistry: The battery cell chemistry stated for the facility (e.g., "LFP", "Lithium Iron Phosphate"); otherwise null.
    - capacity_planned_gwh: The planned annual capacity as stated (e.g., "21 GWh", "20–25 GWh"); do not convert, keep as a string; otherwise null.
    - site_acres: The total site acreage as stated (e.g., "400 acres", "over 400 acres"); keep as a string; otherwise null.
    - main_building_sqft: The area of the main manufacturing building (e.g., "1.5 million sq ft"); keep as a string; otherwise null.
    - groundbreaking_date: The date or month/year when ground was broken on construction (e.g., "September 2024", "Q3 2024"); keep as a string; otherwise null.
    - production_start_year: The planned initial production start year (e.g., "2027", "2028"); keep as a string; otherwise null.
    - sources: Array of all URLs cited in the answer (include links present in any format in the answer such as markdown links or plain URLs). If none are present, return an empty array.

    Important:
    - Do not invent any values not present in the answer text.
    - For URLs, extract only actual links present in the answer (in any format). If a URL lacks protocol, prepend http://.
    - If multiple facilities are discussed, extract only the one used to satisfy the constraints and clearly presented as the answer's chosen facility.
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def normalize_sources(sources: Optional[List[str]]) -> List[str]:
    """Ensure a list of sources is returned; handle None gracefully."""
    if not sources:
        return []
    return [s for s in sources if isinstance(s, str) and s.strip()]


def name_or_placeholder(info: FacilityExtraction) -> str:
    """Return a readable facility identifier for claims."""
    if info.official_name and info.official_name.strip():
        return info.official_name.strip()
    # Construct a fallback based on location if name missing
    locality = info.location_city_or_county or ""
    state = info.location_state or ""
    fallback = f"the battery facility in {locality}, {state}".strip().strip(", ")
    return fallback if fallback else "the battery facility"


def partners_str(info: FacilityExtraction) -> str:
    return ", ".join(info.joint_venture_partners) if info.joint_venture_partners else "the listed JV partners"


# --------------------------------------------------------------------------- #
# Verification construction                                                   #
# --------------------------------------------------------------------------- #
async def build_and_verify_constraints(
    evaluator: Evaluator,
    parent_node,
    info: FacilityExtraction
) -> None:
    """
    Build the verification tree under the critical 'Correct_Facility_Identification' node
    and run verifications for each constraint using the extracted info and sources.
    """
    sources = normalize_sources(info.sources)
    facility_label = name_or_placeholder(info)

    # 1) Facility official name provided (existence check)
    evaluator.add_custom_node(
        result=bool(info.official_name and info.official_name.strip()),
        id="Facility_Official_Name_Provided",
        desc="Provides the official name of the facility (not just a vague description).",
        parent=parent_node,
        critical=True
    )

    # 2) Located in a Mississippi River bordering state
    state = (info.location_state or "").strip()
    mississippi_state_node = evaluator.add_leaf(
        id="Mississippi_River_Bordering_State",
        desc="Facility is located in a U.S. state that borders the Mississippi River.",
        parent=parent_node,
        critical=True
    )
    claim_state = (
        f"{facility_label} is located in the U.S. state '{state}' (or equivalent), "
        f"and that state borders the Mississippi River."
    )
    states_list_text = ", ".join(sorted(MISSISSIPPI_BORDER_STATES))
    await evaluator.verify(
        claim=claim_state,
        node=mississippi_state_node,
        sources=sources,
        additional_instruction=(
            "First, use the source(s) to confirm the facility's state. "
            f"Second, you may rely on general geographical knowledge or the provided list to confirm whether that state borders the Mississippi River. "
            f"States bordering the Mississippi River include: {states_list_text}. "
            "Minor naming variations (e.g., abbreviations) are acceptable."
        ),
    )

    # 3) Joint venture operated between >=2 companies
    jv_node = evaluator.add_leaf(
        id="Joint_Venture_Operated",
        desc="Facility is operated as a joint venture between at least two separate companies.",
        parent=parent_node,
        critical=True
    )
    claim_jv = (
        f"{facility_label} is operated as a joint venture between at least two companies "
        f"(e.g., {partners_str(info)})."
    )
    await evaluator.verify(
        claim=claim_jv,
        node=jv_node,
        sources=sources,
        additional_instruction=(
            "Confirm the facility is explicitly described as a joint venture and that at least two distinct partner companies are named."
        ),
    )

    # 4) Designed to produce LFP battery cells
    lfp_node = evaluator.add_leaf(
        id="LFP_Cell_Production",
        desc="Facility is designed to produce Lithium Iron Phosphate (LFP) battery cells.",
        parent=parent_node,
        critical=True
    )
    claim_lfp = f"{facility_label} is designed to produce Lithium Iron Phosphate (LFP) battery cells."
    await evaluator.verify(
        claim=claim_lfp,
        node=lfp_node,
        sources=sources,
        additional_instruction=(
            "Look for explicit mention of 'LFP' or 'Lithium Iron Phosphate' cells production at the facility."
        ),
    )

    # 5) Capacity between 20 and 25 GWh inclusive
    capacity_node = evaluator.add_leaf(
        id="Capacity_20_to_25_GWh_Inclusive",
        desc="Facility's planned annual production capacity is between 20 GWh and 25 GWh (inclusive).",
        parent=parent_node,
        critical=True
    )
    claim_capacity = (
        f"The planned annual production capacity for {facility_label} is between 20 and 25 GWh (inclusive)."
    )
    await evaluator.verify(
        claim=claim_capacity,
        node=capacity_node,
        sources=sources,
        additional_instruction=(
            "Check capacity statements; if a specific value such as 20 GWh, 21 GWh, 22 GWh, 23 GWh, 24 GWh, or 25 GWh is cited, consider it within the required range. "
            "Ranges like '20–25 GWh' are acceptable."
        ),
    )

    # 6) Site at least 400 acres
    acres_node = evaluator.add_leaf(
        id="Site_At_Least_400_Acres",
        desc="Facility site occupies at least 400 acres.",
        parent=parent_node,
        critical=True
    )
    claim_acres = f"The site area of {facility_label} is at least 400 acres."
    await evaluator.verify(
        claim=claim_acres,
        node=acres_node,
        sources=sources,
        additional_instruction=(
            "Look for site acreage details (e.g., '400 acres', 'over 400 acres'). Accept equivalent phrasings indicating ≥400 acres."
        ),
    )

    # 7) Main building at least 1.5 million sq ft
    main_bld_node = evaluator.add_leaf(
        id="Main_Building_At_Least_1_5M_SqFt",
        desc="Main manufacturing building is at least 1.5 million square feet in area.",
        parent=parent_node,
        critical=True
    )
    claim_bld = f"The main manufacturing building at {facility_label} is at least 1.5 million square feet in area."
    await evaluator.verify(
        claim=claim_bld,
        node=main_bld_node,
        sources=sources,
        additional_instruction=(
            "Check building area statements (e.g., '1.5 million sq ft', '≥1.5M sq ft'). Minor formatting differences are acceptable."
        ),
    )

    # 8) Groundbreaking in 2024 (Jan–Dec)
    groundbreaking_node = evaluator.add_leaf(
        id="Groundbreaking_In_2024",
        desc="Ground was broken on construction between January 2024 and December 2024 (inclusive).",
        parent=parent_node,
        critical=True
    )
    claim_groundbreaking = (
        f"Groundbreaking for {facility_label} occurred in 2024 (between January and December inclusive)."
    )
    await evaluator.verify(
        claim=claim_groundbreaking,
        node=groundbreaking_node,
        sources=sources,
        additional_instruction=(
            "Confirm any explicit references to groundbreaking dates or ceremonies in 2024 (e.g., 'September 2024', 'Q3 2024')."
        ),
    )

    # 9) Planned initial production start date in 2027 or later
    start_node = evaluator.add_leaf(
        id="Production_Start_2027_Or_Later",
        desc="Planned initial production start date is in 2027 or later.",
        parent=parent_node,
        critical=True
    )
    claim_start = f"The planned initial production start date for {facility_label} is in 2027 or later."
    await evaluator.verify(
        claim=claim_start,
        node=start_node,
        sources=sources,
        additional_instruction=(
            "Verify planned start year; acceptable values include 2027, 2028, etc. Allow phrasing such as 'production slated to begin in 2027'."
        ),
    )

    # 10) At least one JV partner is a commercial vehicle or powertrain company
    cv_partner_node = evaluator.add_leaf(
        id="JV_Partner_Commercial_Vehicle_Company",
        desc="At least one joint venture partner is primarily involved in commercial vehicle (truck) manufacturing or commercial vehicle powertrain systems.",
        parent=parent_node,
        critical=True
    )
    claim_cv = (
        f"Among the joint venture partners for {facility_label} (e.g., {partners_str(info)}), "
        "at least one is primarily involved in commercial vehicle (truck) manufacturing or commercial vehicle powertrain systems."
    )
    await evaluator.verify(
        claim=claim_cv,
        node=cv_partner_node,
        sources=sources,
        additional_instruction=(
            "Use source pages to determine whether any named JV partner is a truck manufacturer (commercial vehicles) or a company focused on commercial vehicle powertrains."
        ),
    )

    # 11) Located within a designated industrial park or megasite
    park_node = evaluator.add_leaf(
        id="Located_In_Industrial_Park_Or_Megasite",
        desc="Facility is located within a designated industrial park or megasite.",
        parent=parent_node,
        critical=True
    )
    park_name = (info.industrial_park_or_megasite or "").strip()
    claim_park = (
        f"{facility_label} is located within a designated industrial park or megasite"
        + (f", namely '{park_name}'." if park_name else ".")
    )
    await evaluator.verify(
        claim=claim_park,
        node=park_node,
        sources=sources,
        additional_instruction=(
            "Look for phrases such as 'industrial park', 'megasite', 'mega site', or a named site designation in the source content that explicitly places the facility within such a location."
        ),
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
    Evaluate an agent's answer for the battery facility constraints task.
    Returns a structured summary including the verification tree and final score.
    """
    # Initialize evaluator
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Root as parallel aggregator
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

    # Extract structured info from the answer
    facility_info = await evaluator.extract(
        prompt=prompt_extract_facility_info(),
        template_class=FacilityExtraction,
        extraction_name="facility_extraction",
    )

    # Record helpful custom info (e.g., border states reference)
    evaluator.add_custom_info(
        info={"border_states": sorted(list(MISSISSIPPI_BORDER_STATES))},
        info_type="reference",
        info_name="mississippi_border_states"
    )

    # Create the critical parent node corresponding to the rubric root
    main_node = evaluator.add_parallel(
        id="Correct_Facility_Identification",
        desc="Answer provides the official name of a U.S. battery manufacturing facility satisfying all listed constraints.",
        parent=root,
        critical=True
    )

    # Build and verify all constraints
    await build_and_verify_constraints(evaluator, main_node, facility_info)

    # Return the summary with verification tree and scores
    return evaluator.get_summary()