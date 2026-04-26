import asyncio
import logging
from typing import Any, List, Optional, Dict

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy

# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "us_airport_terminals_2024"
TASK_DESCRIPTION = (
    "Identify three major airport terminal projects in the United States that were completed and fully opened to passengers in 2024. "
    "For each terminal project, provide comprehensive information including: "
    "(1) The full official airport name, specific terminal designation, opening date (must be between January 1-December 31, 2024), U.S. state location, and project type (new terminal, expansion, or modernization); "
    "(2) Physical specifications including total square footage (must be at least 100,000 sq ft), number of gates (must be at least 4), and documented total project cost; "
    "(3) At least one documented sustainability feature (such as LEED certification, energy efficiency measures, renewable energy integration, or sustainable design elements); "
    "(4) At least three different types of passenger amenities (such as charging stations, nursing rooms, family restrooms, pet relief areas, companion care rooms, sensory rooms, or mother's rooms); "
    "(5) Information about retail and/or dining concessions with specific tenant names; "
    "(6) Construction timeline including documented start and completion dates; "
    "(7) At least one notable architectural or design feature. All information must be supported by reference URLs from credible sources."
)

# --------------------------------------------------------------------------- #
# Data models                                                                 #
# --------------------------------------------------------------------------- #
class TerminalProject(BaseModel):
    airport_official_name: Optional[str] = None
    terminal_designation: Optional[str] = None
    state: Optional[str] = None
    project_type: Optional[str] = None  # Expected values: "new terminal", "expansion", "modernization" (or equivalent wording)
    opening_date: Optional[str] = None

    square_footage: Optional[str] = None
    gates: Optional[str] = None
    project_cost: Optional[str] = None

    sustainability_features: List[str] = Field(default_factory=list)
    amenities: List[str] = Field(default_factory=list)
    concession_tenants: List[str] = Field(default_factory=list)

    construction_start_date: Optional[str] = None
    construction_completion_date: Optional[str] = None

    design_features: List[str] = Field(default_factory=list)
    accessibility_features: List[str] = Field(default_factory=list)

    source_urls: List[str] = Field(default_factory=list)


class TerminalProjectsExtraction(BaseModel):
    projects: List[TerminalProject] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_terminal_projects() -> str:
    return (
        "From the provided answer, extract all terminal projects mentioned that are in the United States and claimed to have fully opened to passengers in 2024. "
        "Return them in a 'projects' array (preserve the original order). For each project, extract the following fields exactly as stated in the answer:\n"
        "1) airport_official_name: Full official airport name (e.g., 'Los Angeles International Airport')\n"
        "2) terminal_designation: Specific terminal designation/name (e.g., 'Terminal A', 'Concourse B', 'New Terminal 1')\n"
        "3) state: U.S. state location (e.g., 'California', 'Texas')\n"
        "4) project_type: One of 'new terminal', 'expansion', or 'modernization' as described (use the best match to the answer; if unclear, return the closest descriptor from those three)\n"
        "5) opening_date: The date the terminal was fully opened to passengers in 2024 (string format as in answer)\n"
        "6) square_footage: Total square footage of the project (string, include units or qualifiers as present)\n"
        "7) gates: Number of gates (string; keep the exact phrasing, e.g., '18 gates')\n"
        "8) project_cost: Documented total project cost (string; include currency if present)\n"
        "9) sustainability_features: List of documented sustainability features (e.g., 'LEED Silver', 'solar panels', 'low-energy HVAC'); return an empty list if none are mentioned\n"
        "10) amenities: List with all passenger amenities explicitly mentioned (e.g., 'charging stations', 'nursing rooms', 'family restrooms', 'pet relief areas', 'sensory room', 'mother's room'); return an empty list if none are mentioned\n"
        "11) concession_tenants: List of specific retail/dining tenant names mentioned; return an empty list if none are mentioned\n"
        "12) construction_start_date: Documented construction start date (string; as in answer)\n"
        "13) construction_completion_date: Documented construction completion date (string; as in answer)\n"
        "14) design_features: List of notable architectural/design features (e.g., 'expansive glass facade', 'wood-accented ceilings', 'iconic atrium'); return an empty list if none are mentioned\n"
        "15) accessibility_features: List of modern accessibility features for passengers with disabilities or special needs (e.g., 'companion care rooms', 'adult changing tables', 'hearing loop systems'); return an empty list if none are mentioned\n"
        "16) source_urls: All URLs in the answer that support this project's claims (include official airport pages, government/local authority pages, reputable news, press releases, Google Drive PDFs, etc.). "
        "Only include valid URLs. Do not invent URLs.\n\n"
        "If any field is missing from the answer, return null for that string field or an empty list for list fields. "
        "Extract ONLY what is explicitly present in the answer."
    )


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _text_present(s: Optional[str]) -> bool:
    return bool(s and str(s).strip())

def _normalize_project_type(s: Optional[str]) -> Optional[str]:
    if not _text_present(s):
        return None
    t = s.strip().lower()
    # Simple normalization to allowed set based on keywords
    if "new" in t and ("terminal" in t or "concourse" in t or "building" in t):
        return "new terminal"
    if "expansion" in t or "expand" in t or "extension" in t or "addition" in t:
        return "expansion"
    if "modernization" in t or "modernize" in t or "renovation" in t or "redevelopment" in t or "upgrade" in t or "rebuild" in t:
        return "modernization"
    # Fallback to original text; evaluator will still verify via sources
    return s.strip()

NO_SOURCES_FAIL_INSTRUCTION = (
    "Important: If no source URLs are provided in the answer for this project, judge the claim as not supported/Incorrect."
)

# --------------------------------------------------------------------------- #
# Verification for one project                                                #
# --------------------------------------------------------------------------- #
async def verify_terminal_project(
    evaluator: Evaluator,
    parent_node,
    project: TerminalProject,
    idx: int,
) -> None:
    proj_idx = idx + 1
    proj_node = evaluator.add_parallel(
        id=f"Terminal_Project_{proj_idx}",
        desc=f"{proj_idx}st qualifying terminal project (partial credit item)" if proj_idx == 1 else (
             f"{proj_idx}nd qualifying terminal project (partial credit item)" if proj_idx == 2 else
             f"{proj_idx}rd qualifying terminal project (partial credit item)"),
        parent=parent_node,
        critical=False
    )

    # 1) Identification and Qualification
    ident_node = evaluator.add_parallel(
        id=f"Identification_and_Qualification_{proj_idx}",
        desc=f"Project {proj_idx} identification and qualification constraints",
        parent=proj_node,
        critical=True
    )

    # Required identification fields (existence checks as critical custom nodes)
    evaluator.add_custom_node(
        result=_text_present(project.airport_official_name),
        id=f"Airport_Official_Name_{proj_idx}",
        desc=f"Provide the full official airport name for Project {proj_idx}",
        parent=ident_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=_text_present(project.terminal_designation),
        id=f"Terminal_Designation_{proj_idx}",
        desc=f"Provide the specific terminal designation/name for Project {proj_idx}",
        parent=ident_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=_text_present(project.state),
        id=f"US_State_Location_{proj_idx}",
        desc=f"Provide the U.S. state location for Project {proj_idx}",
        parent=ident_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=_text_present(_normalize_project_type(project.project_type)),
        id=f"Project_Type_{proj_idx}",
        desc=(
            f"Specify whether Project {proj_idx} is a new terminal, major expansion, or significant modernization "
            "(not minor renovation/maintenance)"
        ),
        parent=ident_node,
        critical=True
    )

    # Major commercial US airport verification
    major_airport_leaf = evaluator.add_leaf(
        id=f"Major_Commercial_US_Airport_{proj_idx}",
        desc=f"Indicate that the airport for Project {proj_idx} is a major commercial airport in the United States (as supported by provided sources)",
        parent=ident_node,
        critical=True
    )
    claim_major_airport = (
        f"The airport '{project.airport_official_name or 'UNKNOWN AIRPORT'}' is a major commercial airport in the United States that "
        f"serves scheduled passenger flights."
    )
    await evaluator.verify(
        claim=claim_major_airport,
        node=major_airport_leaf,
        sources=project.source_urls,
        additional_instruction=(
            "Use the provided URLs to confirm that this is a major commercial U.S. airport (e.g., scheduled airline service, "
            "significant passenger volume, official classification). " + NO_SOURCES_FAIL_INSTRUCTION
        ),
    )

    # Opening to passengers in 2024
    opening_leaf = evaluator.add_leaf(
        id=f"Opened_To_Passengers_In_2024_{proj_idx}",
        desc=f"Provide an opening date showing Project {proj_idx} was completed and fully opened to passengers between Jan 1, 2024 and Dec 31, 2024",
        parent=ident_node,
        critical=True
    )
    claim_opening = (
        f"The terminal '{project.terminal_designation or 'UNKNOWN TERMINAL'}' at '{project.airport_official_name or 'UNKNOWN AIRPORT'}' "
        f"was completed and fully opened to passengers on {project.opening_date or 'an opening date in 2024'}, "
        f"which falls between January 1, 2024 and December 31, 2024."
    )
    await evaluator.verify(
        claim=claim_opening,
        node=opening_leaf,
        sources=project.source_urls,
        additional_instruction=(
            "Confirm that the source(s) explicitly indicate the terminal was completed and fully opened to passengers in 2024 "
            "(not just a soft/partial opening). Accept specific dates or clear statements of full opening in 2024. "
            + NO_SOURCES_FAIL_INSTRUCTION
        ),
    )

    # 2) Physical Specifications and Cost
    phys_node = evaluator.add_parallel(
        id=f"Physical_Specifications_{proj_idx}",
        desc=f"Project {proj_idx} physical constraints and cost",
        parent=proj_node,
        critical=True
    )

    sqft_leaf = evaluator.add_leaf(
        id=f"Square_Footage_Minimum_{proj_idx}",
        desc=f"Provide total added/renovated square footage for Project {proj_idx} and it must be ≥ 100,000 sq ft",
        parent=phys_node,
        critical=True
    )
    claim_sqft = (
        f"The total square footage associated with this terminal project is {project.square_footage or 'at least 100,000 square feet'}, "
        f"and it is at least 100,000 square feet."
    )
    await evaluator.verify(
        claim=claim_sqft,
        node=sqft_leaf,
        sources=project.source_urls,
        additional_instruction=(
            "Verify that the project's total square footage meets or exceeds 100,000 square feet. "
            "If the source uses different units (e.g., m² or 'million sq ft'), convert or interpret reasonably. "
            + NO_SOURCES_FAIL_INSTRUCTION
        ),
    )

    gates_leaf = evaluator.add_leaf(
        id=f"Gates_Minimum_{proj_idx}",
        desc=f"Provide number of gates for Project {proj_idx} and it must be ≥ 4 (new or renovated)",
        parent=phys_node,
        critical=True
    )
    claim_gates = (
        f"The terminal has at least 4 passenger gates; the number of gates is {project.gates or 'at least 4'}."
    )
    await evaluator.verify(
        claim=claim_gates,
        node=gates_leaf,
        sources=project.source_urls,
        additional_instruction=(
            "Confirm from the URLs the number of gates (new or renovated) is at least 4. "
            "Accept reasonable descriptions like 'X contact gates' or clear counts from official sources. "
            + NO_SOURCES_FAIL_INSTRUCTION
        ),
    )

    cost_leaf = evaluator.add_leaf(
        id=f"Documented_Project_Cost_{proj_idx}",
        desc=f"Provide a publicly documented/verifiable total project cost for Project {proj_idx}",
        parent=phys_node,
        critical=True
    )
    claim_cost = (
        f"The total project cost is documented as {project.project_cost or 'a specific amount stated in the sources'}."
    )
    await evaluator.verify(
        claim=claim_cost,
        node=cost_leaf,
        sources=project.source_urls,
        additional_instruction=(
            "Verify that the total project cost is explicitly documented in at least one provided URL (official page, government or reputable news). "
            + NO_SOURCES_FAIL_INSTRUCTION
        ),
    )

    # 3) Sustainability
    sust_node = evaluator.add_parallel(
        id=f"Sustainability_{proj_idx}",
        desc=f"Project {proj_idx} sustainability requirement",
        parent=proj_node,
        critical=True
    )
    sust_leaf = evaluator.add_leaf(
        id=f"At_Least_One_Sustainability_Feature_{proj_idx}",
        desc=f"Provide at least one documented sustainability feature for Project {proj_idx} (e.g., LEED, energy efficiency, renewables, sustainable design elements)",
        parent=sust_node,
        critical=True
    )
    sust_list_str = ", ".join(project.sustainability_features) if project.sustainability_features else "at least one sustainability feature"
    claim_sust = (
        f"The project includes documented sustainability feature(s), such as {sust_list_str}."
    )
    await evaluator.verify(
        claim=claim_sust,
        node=sust_leaf,
        sources=project.source_urls,
        additional_instruction=(
            "Pass if at least one sustainability-related feature (e.g., LEED certification, energy efficiency measures, renewable energy, sustainable design elements) "
            "is explicitly documented by the sources. " + NO_SOURCES_FAIL_INSTRUCTION
        ),
    )

    # 4) Passenger Amenities
    amen_node = evaluator.add_parallel(
        id=f"Passenger_Amenities_{proj_idx}",
        desc=f"Project {proj_idx} passenger amenities requirement",
        parent=proj_node,
        critical=True
    )
    amen_leaf = evaluator.add_leaf(
        id=f"At_Least_Three_Distinct_Amenity_Types_{proj_idx}",
        desc=f"Provide at least three different types of passenger amenities for Project {proj_idx}",
        parent=amen_node,
        critical=True
    )
    amen_list_str = ", ".join(project.amenities) if project.amenities else "at least three different amenity types"
    claim_amen = (
        f"The terminal provides at least three different types of passenger amenities (e.g., charging stations, nursing rooms, family restrooms, pet relief areas, sensory rooms, mother's rooms); "
        f"examples include {amen_list_str}."
    )
    await evaluator.verify(
        claim=claim_amen,
        node=amen_leaf,
        sources=project.source_urls,
        additional_instruction=(
            "Confirm from the sources that at least three distinct amenity types are present (count unique types). "
            "Minor phrasing differences are acceptable. " + NO_SOURCES_FAIL_INSTRUCTION
        ),
    )

    # 5) Concessions
    conc_node = evaluator.add_parallel(
        id=f"Concessions_{proj_idx}",
        desc=f"Project {proj_idx} retail/dining concessions requirement",
        parent=proj_node,
        critical=True
    )
    conc_leaf = evaluator.add_leaf(
        id=f"Concessions_With_Tenant_Names_{proj_idx}",
        desc=f"Provide retail and/or dining concessions information for Project {proj_idx} including specific tenant names",
        parent=conc_node,
        critical=True
    )
    tenants_str = ", ".join(project.concession_tenants) if project.concession_tenants else "specific tenant names listed in sources"
    claim_conc = (
        f"Retail and/or dining concessions for this terminal include specific tenant names such as {tenants_str}."
    )
    await evaluator.verify(
        claim=claim_conc,
        node=conc_leaf,
        sources=project.source_urls,
        additional_instruction=(
            "Verify that at least some specific concession tenant names are listed in the provided sources for this terminal. "
            + NO_SOURCES_FAIL_INSTRUCTION
        ),
    )

    # 6) Construction Timeline
    tl_node = evaluator.add_parallel(
        id=f"Construction_Timeline_{proj_idx}",
        desc=f"Project {proj_idx} construction timeline requirement",
        parent=proj_node,
        critical=True
    )
    tl_leaf = evaluator.add_leaf(
        id=f"Start_And_Completion_Dates_{proj_idx}",
        desc=f"Provide publicly documented construction start date and completion date for Project {proj_idx}",
        parent=tl_node,
        critical=True
    )
    claim_tl = (
        f"Construction for this terminal project started on {project.construction_start_date or 'the documented start date'} "
        f"and completed on {project.construction_completion_date or 'the documented completion date'}."
    )
    await evaluator.verify(
        claim=claim_tl,
        node=tl_leaf,
        sources=project.source_urls,
        additional_instruction=(
            "Confirm both a construction start date and a completion date from the sources. Month/year formats are acceptable if day is not provided. "
            + NO_SOURCES_FAIL_INSTRUCTION
        ),
    )

    # 7) Architecture / Design
    arch_node = evaluator.add_parallel(
        id=f"Architecture_Design_{proj_idx}",
        desc=f"Project {proj_idx} design requirement",
        parent=proj_node,
        critical=True
    )
    arch_leaf = evaluator.add_leaf(
        id=f"Notable_Architectural_Or_Design_Feature_{proj_idx}",
        desc=f"Provide at least one publicly documented notable architectural/design feature for Project {proj_idx}",
        parent=arch_node,
        critical=True
    )
    design_str = ", ".join(project.design_features) if project.design_features else "at least one notable architectural/design feature"
    claim_arch = (
        f"The terminal features notable architectural/design elements such as {design_str}."
    )
    await evaluator.verify(
        claim=claim_arch,
        node=arch_leaf,
        sources=project.source_urls,
        additional_instruction=(
            "Verify that at least one notable architectural or design feature is described in the provided sources. "
            + NO_SOURCES_FAIL_INSTRUCTION
        ),
    )

    # 8) Accessibility
    acc_node = evaluator.add_parallel(
        id=f"Accessibility_{proj_idx}",
        desc=f"Project {proj_idx} accessibility requirement",
        parent=proj_node,
        critical=True
    )
    acc_leaf = evaluator.add_leaf(
        id=f"Modern_Accessibility_Feature_{proj_idx}",
        desc=f"Provide at least one documented modern accessibility feature for passengers with disabilities or special needs in Project {proj_idx}",
        parent=acc_node,
        critical=True
    )
    acc_str = ", ".join(project.accessibility_features) if project.accessibility_features else "at least one modern accessibility feature"
    claim_acc = (
        f"The terminal provides modern accessibility features for passengers with disabilities or special needs, such as {acc_str}."
    )
    await evaluator.verify(
        claim=claim_acc,
        node=acc_leaf,
        sources=project.source_urls,
        additional_instruction=(
            "Confirm the presence of modern accessibility features (e.g., companion care rooms, adult changing tables, accessible restrooms, "
            "hearing loop systems, wayfinding aids). " + NO_SOURCES_FAIL_INSTRUCTION
        ),
    )

    # 9) References / Source URLs (simple structural check)
    refs_node = evaluator.add_parallel(
        id=f"References_{proj_idx}",
        desc=f"Project {proj_idx} source support requirement",
        parent=proj_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=(len(project.source_urls) >= 2),
        id=f"Credible_Source_URLs_{proj_idx}",
        desc=f"Provide reference URL(s) from credible sources that collectively support the required Project {proj_idx} claims/attributes",
        parent=refs_node,
        critical=True
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

    # Extract terminal projects data from the agent's answer
    extraction = await evaluator.extract(
        prompt=prompt_extract_terminal_projects(),
        template_class=TerminalProjectsExtraction,
        extraction_name="terminal_projects"
    )

    # Keep only the first 3 projects; pad with empty entries if fewer than 3
    projects: List[TerminalProject] = list(extraction.projects[:3])
    while len(projects) < 3:
        projects.append(TerminalProject())

    # Build verification tree and verify each project independently (parallel at root)
    for idx, project in enumerate(projects):
        await verify_terminal_project(evaluator, root, project, idx)

    # Return summary
    return evaluator.get_summary()