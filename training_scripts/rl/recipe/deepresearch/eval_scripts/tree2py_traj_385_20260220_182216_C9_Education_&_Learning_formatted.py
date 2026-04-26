import asyncio
import logging
from typing import Optional, List, Dict, Any, Set

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy

# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "abet_programs_4_regions"
TASK_DESCRIPTION = (
    "Identify 4 ABET-accredited undergraduate engineering programs, each located in a different U.S. state within the "
    "Southern, Midwestern, or Mid-Atlantic regions. Each program must meet the following mandatory requirements: "
    "(1) The engineering program must be accredited by ABET's Engineering Accreditation Commission (EAC); "
    "(2) The institution must hold regional accreditation from either the Middle States Commission on Higher Education (MSCHE), "
    "the Southern Association of Colleges and Schools Commission on Colleges (SACSCOC), or the Higher Learning Commission (HLC); "
    "(3) The institution must compete in NCAA Division I athletics; "
    "(4) The program must offer a cooperative education (co-op) or structured internship program for engineering students. "
    "For each program, provide: university name and specific engineering discipline/program, state location, ABET accreditation verification with URL reference, "
    "regional accreditation body and verification with URL reference, NCAA Division I confirmation with URL reference, co-op/internship program description with URL reference, "
    "minimum GPA requirement for admission or progression with URL reference, required mathematics coursework with URL reference, capstone design project requirement with URL reference, "
    "annual tuition costs with URL reference, on-campus housing costs with URL reference, career placement or outcomes data with URL reference, and average starting salary for graduates with URL reference."
)

# --------------------------------------------------------------------------- #
# Region/state helpers                                                        #
# --------------------------------------------------------------------------- #

MID_ATLANTIC_STATES = {
    "Delaware", "District of Columbia", "Maryland", "New Jersey", "New York",
    "Pennsylvania", "Virginia", "West Virginia"
}

MIDWESTERN_STATES = {
    "Illinois", "Indiana", "Iowa", "Kansas", "Michigan", "Minnesota", "Missouri",
    "Nebraska", "North Dakota", "Ohio", "South Dakota", "Wisconsin"
}

SOUTHERN_STATES = {
    "Alabama", "Arkansas", "Delaware", "District of Columbia", "Florida", "Georgia",
    "Kentucky", "Louisiana", "Maryland", "Mississippi", "North Carolina",
    "Oklahoma", "South Carolina", "Tennessee", "Texas", "Virginia", "West Virginia"
}

ALLOWED_STATES = MID_ATLANTIC_STATES | MIDWESTERN_STATES | SOUTHERN_STATES

STATE_ABBREVIATIONS = {
    "AL": "Alabama", "AK": "Alaska", "AR": "Arkansas", "AZ": "Arizona", "CA": "California",
    "CO": "Colorado", "CT": "Connecticut", "DC": "District of Columbia", "DE": "Delaware",
    "FL": "Florida", "GA": "Georgia", "HI": "Hawaii", "IA": "Iowa", "ID": "Idaho",
    "IL": "Illinois", "IN": "Indiana", "KS": "Kansas", "KY": "Kentucky", "LA": "Louisiana",
    "MA": "Massachusetts", "MD": "Maryland", "ME": "Maine", "MI": "Michigan", "MN": "Minnesota",
    "MO": "Missouri", "MS": "Mississippi", "MT": "Montana", "NC": "North Carolina", "ND": "North Dakota",
    "NE": "Nebraska", "NH": "New Hampshire", "NJ": "New Jersey", "NM": "New Mexico", "NV": "Nevada",
    "NY": "New York", "OH": "Ohio", "OK": "Oklahoma", "OR": "Oregon", "PA": "Pennsylvania",
    "RI": "Rhode Island", "SC": "South Carolina", "SD": "South Dakota", "TN": "Tennessee", "TX": "Texas",
    "UT": "Utah", "VA": "Virginia", "VT": "Vermont", "WA": "Washington", "WI": "Wisconsin",
    "WV": "West Virginia", "WY": "Wyoming"
}

def normalize_state(state: Optional[str]) -> Optional[str]:
    if not state:
        return None
    s = state.strip()
    if not s:
        return None
    upper = s.upper()
    if upper in STATE_ABBREVIATIONS:
        return STATE_ABBREVIATIONS[upper]
    # Handle DC variations
    if upper in {"D.C.", "DC", "WASHINGTON DC", "WASHINGTON, DC", "WASHINGTON, D.C."}:
        return "District of Columbia"
    # Title-case normalize
    return s.title()

def is_state_in_allowed_regions(state: Optional[str]) -> bool:
    full = normalize_state(state)
    return bool(full and full in ALLOWED_STATES)

def is_allowed_accreditor(name: Optional[str]) -> bool:
    if not name:
        return False
    s = name.strip().lower()
    tokens = [
        "middle states commission on higher education", "msche", "middle states",
        "southern association of colleges and schools commission on colleges", "sacs", "sacs coc", "sacs-coc", "sacsoc", "sacs coc", "sacs coc", "sacs-coc", "sacscoc",
        "higher learning commission", "hlc"
    ]
    return any(tok in s for tok in tokens)

# --------------------------------------------------------------------------- #
# Data models                                                                 #
# --------------------------------------------------------------------------- #

class ProgramAccreditationInfo(BaseModel):
    discipline_name: Optional[str] = None
    abet_url: Optional[str] = None

class RegionalAccreditationInfo(BaseModel):
    accreditor: Optional[str] = None  # e.g., MSCHE / SACSCOC / HLC (or full name)
    accred_url: Optional[str] = None

class AthleticsInfo(BaseModel):
    ncaa_division: Optional[str] = None  # e.g., "Division I"
    ncaa_url: Optional[str] = None

class CoopInfo(BaseModel):
    description: Optional[str] = None
    coop_url: Optional[str] = None

class AcademicRequirements(BaseModel):
    gpa_value: Optional[str] = None
    gpa_url: Optional[str] = None
    math_courses: Optional[str] = None
    math_url: Optional[str] = None
    capstone_desc: Optional[str] = None
    capstone_url: Optional[str] = None

class CostInfo(BaseModel):
    tuition_amount: Optional[str] = None
    tuition_url: Optional[str] = None
    housing_amount: Optional[str] = None
    housing_url: Optional[str] = None

class OutcomesInfo(BaseModel):
    placement_data: Optional[str] = None
    placement_url: Optional[str] = None
    starting_salary: Optional[str] = None
    salary_url: Optional[str] = None

class ProgramInfo(BaseModel):
    university: Optional[str] = None
    discipline: Optional[str] = None
    state: Optional[str] = None
    region: Optional[str] = None  # If provided in the answer; not strictly needed
    abet: ProgramAccreditationInfo = Field(default_factory=ProgramAccreditationInfo)
    regional_accreditation: RegionalAccreditationInfo = Field(default_factory=RegionalAccreditationInfo)
    athletics: AthleticsInfo = Field(default_factory=AthleticsInfo)
    coop: CoopInfo = Field(default_factory=CoopInfo)
    academics: AcademicRequirements = Field(default_factory=AcademicRequirements)
    costs: CostInfo = Field(default_factory=CostInfo)
    outcomes: OutcomesInfo = Field(default_factory=OutcomesInfo)

class ProgramsExtraction(BaseModel):
    programs: List[ProgramInfo] = Field(default_factory=list)

# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #

def prompt_extract_programs() -> str:
    return (
        "Extract up to 4 undergraduate engineering programs described in the answer that meet the task context.\n"
        "For each program, extract the following fields exactly as stated in the answer. If any field is missing in the answer, return null for that field.\n"
        "Return a JSON object with a single array field 'programs'. Each element must be an object with this schema:\n"
        "- university: University name\n"
        "- discipline: Specific engineering discipline/program name (e.g., Mechanical Engineering)\n"
        "- state: U.S. state where the university is located (either full name or USPS abbreviation)\n"
        "- region: If the answer mentions a region (Southern, Midwestern, or Mid-Atlantic), include it; otherwise null\n"
        "- abet: {\n"
        "    discipline_name: The specific program name as presented on the ABET page (if given),\n"
        "    abet_url: The URL to the ABET Accredited Programs database or official ABET page confirming accreditation. Prefer the ABET database.\n"
        "  }\n"
        "- regional_accreditation: {\n"
        "    accreditor: Name or acronym of the regional accrediting body (MSCHE, SACSCOC, HLC),\n"
        "    accred_url: URL confirming the institution's regional accreditation status.\n"
        "  }\n"
        "- athletics: {\n"
        "    ncaa_division: The division label stated (e.g., 'Division I'),\n"
        "    ncaa_url: A URL confirming NCAA Division I status (NCAA site or official athletics page).\n"
        "  }\n"
        "- coop: {\n"
        "    description: Brief description of the co-op or structured internship program (if provided),\n"
        "    coop_url: URL to the program page describing the co-op/internship.\n"
        "  }\n"
        "- academics: {\n"
        "    gpa_value: Minimum GPA requirement for admission or progression (string),\n"
        "    gpa_url: URL documenting this GPA requirement,\n"
        "    math_courses: Required mathematics coursework (string list or description; include Calculus I/II if shown),\n"
        "    math_url: URL with math/curriculum requirements,\n"
        "    capstone_desc: Description of capstone/senior design project requirement,\n"
        "    capstone_url: URL documenting the capstone requirement.\n"
        "  }\n"
        "- costs: {\n"
        "    tuition_amount: Annual tuition cost or range (string),\n"
        "    tuition_url: URL for tuition information,\n"
        "    housing_amount: On-campus housing cost or range (string),\n"
        "    housing_url: URL for housing cost information.\n"
        "  }\n"
        "- outcomes: {\n"
        "    placement_data: Career placement/outcomes data (string; e.g., placement rate or summary),\n"
        "    placement_url: URL for placement/outcomes data,\n"
        "    starting_salary: Average starting salary for graduates (string),\n"
        "    salary_url: URL for starting salary information.\n"
        "  }\n"
        "Rules:\n"
        "1) Extract only from the provided answer text; do not invent values. If a required URL or value is not present, use null.\n"
        "2) For URLs, extract the actual links as presented (plain or markdown). Include protocol; fix missing protocol by prepending http://.\n"
        "3) Preserve the wording of values exactly as shown (e.g., dollar amounts or ranges as text).\n"
    )

# --------------------------------------------------------------------------- #
# Verification helpers                                                        #
# --------------------------------------------------------------------------- #

def _safe(s: Optional[str]) -> str:
    return s or ""

def build_program_label(idx: int) -> str:
    return f"P{idx + 1}"

async def verify_program(
    evaluator: Evaluator,
    parent_node,
    program: ProgramInfo,
    idx: int,
    used_states: Set[str],
) -> None:
    """
    Build verification tree for a single program according to rubric.
    """
    pid = build_program_label(idx)
    uni = _safe(program.university)
    disc = _safe(program.discipline)
    state_full = normalize_state(program.state)

    # Top-level Program node (non-critical, parallel aggregation)
    prog_node = evaluator.add_parallel(
        id=f"Program_{idx + 1}",
        desc=[
            "First engineering program meeting all specified criteria",
            "Second engineering program meeting all specified criteria",
            "Third engineering program meeting all specified criteria",
            "Fourth engineering program meeting all specified criteria",
        ][idx],
        parent=parent_node,
        critical=False
    )

    # Unique State (critical, custom check)
    is_unique_state = bool(state_full) and (state_full not in used_states)
    evaluator.add_custom_node(
        result=is_unique_state,
        id=f"{pid}_Unique_State",
        desc="Program is located in a U.S. state not used by other identified programs",
        parent=prog_node,
        critical=True
    )
    if is_unique_state and state_full:
        used_states.add(state_full)

    # Regional Scope (critical, custom check)
    evaluator.add_custom_node(
        result=is_state_in_allowed_regions(state_full),
        id=f"{pid}_Regional_Scope",
        desc="Program is located in Southern, Midwestern, or Mid-Atlantic region",
        parent=prog_node,
        critical=True
    )

    # Accreditation Verification (critical, parallel)
    accred_node = evaluator.add_parallel(
        id=f"{pid}_Accreditation_Verification",
        desc="Program holds required accreditation credentials",
        parent=prog_node,
        critical=True
    )

    # ABET Accreditation (critical, parallel)
    abet_node = evaluator.add_parallel(
        id=f"{pid}_ABET_Accreditation",
        desc="Program is accredited by ABET's Engineering Accreditation Commission (EAC)",
        parent=accred_node,
        critical=True
    )

    # ABET URL Reference (critical leaf): verify EAC accreditation via ABET page
    abet_url_leaf = evaluator.add_leaf(
        id=f"{pid}_ABET_URL_Reference",
        desc="Verification URL from ABET database confirming accreditation",
        parent=abet_node,
        critical=True
    )
    abet_claim = (
        f"The ABET Accredited Programs database page confirms that the {disc} program at {uni} "
        f"is accredited by the Engineering Accreditation Commission (EAC)."
    )
    await evaluator.verify(
        claim=abet_claim,
        node=abet_url_leaf,
        sources=program.abet.abet_url,
        additional_instruction=(
            "Check the ABET page carefully; it should list the institution and the specific program/discipline, "
            "and indicate EAC accreditation. Minor naming variations are acceptable."
        )
    )

    # ABET Specific Program (critical leaf): program/discipline appears on ABET page
    abet_prog_leaf = evaluator.add_leaf(
        id=f"{pid}_ABET_Specific_Program",
        desc="Specific engineering discipline within the ABET-accredited programs",
        parent=abet_node,
        critical=True
    )
    abet_prog_claim = (
        f"The ABET page explicitly lists {disc} (or an equivalent program name) as an accredited program at {uni}."
    )
    await evaluator.verify(
        claim=abet_prog_claim,
        node=abet_prog_leaf,
        sources=program.abet.abet_url,
        additional_instruction="Allow reasonable naming variants (e.g., 'Mechanical Engineering (B.S.)')."
    )

    # Regional Accreditation (critical, parallel)
    reg_accred_node = evaluator.add_parallel(
        id=f"{pid}_Regional_Accreditation",
        desc="Institution holds regional accreditation from Middle States, SACSCOC, or HLC",
        parent=accred_node,
        critical=True
    )

    # Regional Accreditor (critical leaf): accreditor must be MSCHE/SACSCOC/HLC
    reg_acc_leaf_custom = evaluator.add_custom_node(
        result=is_allowed_accreditor(program.regional_accreditation.accreditor),
        id=f"{pid}_Regional_Accreditor",
        desc="Name of the regional accrediting body",
        parent=reg_accred_node,
        critical=True
    )

    # Regional URL Reference (critical leaf): verify accreditation via accreditor page
    reg_acc_leaf = evaluator.add_leaf(
        id=f"{pid}_Regional_URL_Reference",
        desc="Verification URL confirming regional accreditation status",
        parent=reg_accred_node,
        critical=True
    )
    reg_claim = (
        f"{uni} holds regional accreditation from {_safe(program.regional_accreditation.accreditor)}."
    )
    await evaluator.verify(
        claim=reg_claim,
        node=reg_acc_leaf,
        sources=program.regional_accreditation.accred_url,
        additional_instruction="Verify that the page explicitly confirms the institution's regional accreditation."
    )

    # Athletic Classification (critical, parallel)
    ath_node = evaluator.add_parallel(
        id=f"{pid}_Athletic_Classification",
        desc="Institution competes in NCAA Division I athletics",
        parent=prog_node,
        critical=True
    )

    # NCAA Division I (critical leaf)
    ncaa_div_leaf = evaluator.add_leaf(
        id=f"{pid}_NCAA_Division_I",
        desc="Confirmation that the institution is an NCAA Division I school",
        parent=ath_node,
        critical=True
    )
    ncaa_div_claim = f"{uni} competes in NCAA Division I athletics."
    await evaluator.verify(
        claim=ncaa_div_claim,
        node=ncaa_div_leaf,
        sources=program.athletics.ncaa_url,
        additional_instruction="Confirm Division I status; institution athletics page or NCAA roster page is acceptable."
    )

    # NCAA URL Reference (critical leaf)
    ncaa_url_leaf = evaluator.add_leaf(
        id=f"{pid}_NCAA_URL_Reference",
        desc="Verification URL confirming NCAA Division I status",
        parent=ath_node,
        critical=True
    )
    ncaa_url_claim = "This page confirms NCAA Division I status for the institution."
    await evaluator.verify(
        claim=ncaa_url_claim,
        node=ncaa_url_leaf,
        sources=program.athletics.ncaa_url,
        additional_instruction="The page should explicitly indicate Division I or provide enough context to confirm it."
    )

    # Cooperative Education (critical, parallel)
    coop_node = evaluator.add_parallel(
        id=f"{pid}_Cooperative_Education",
        desc="Program offers cooperative education or structured internship program",
        parent=prog_node,
        critical=True
    )

    # Coop Availability (critical leaf)
    coop_avail_leaf = evaluator.add_leaf(
        id=f"{pid}_Coop_Availability",
        desc="Documented availability of co-op or internship program for engineering students",
        parent=coop_node,
        critical=True
    )
    coop_avail_claim = (
        f"The engineering program/college at {uni} offers a cooperative education or structured internship program."
    )
    await evaluator.verify(
        claim=coop_avail_claim,
        node=coop_avail_leaf,
        sources=program.coop.coop_url,
        additional_instruction="Verify that the page describes a co-op or structured internship opportunity specifically for engineering students."
    )

    # Coop URL Reference (critical leaf)
    coop_url_leaf = evaluator.add_leaf(
        id=f"{pid}_Coop_URL_Reference",
        desc="Verification URL describing the co-op/internship program",
        parent=coop_node,
        critical=True
    )
    coop_url_claim = "This page describes the co-op or structured internship program for engineering students."
    await evaluator.verify(
        claim=coop_url_claim,
        node=coop_url_leaf,
        sources=program.coop.coop_url,
        additional_instruction="Page should be a university/college official page detailing co-op/internship."
    )

    # Academic Requirements (non-critical, parallel)
    acad_node = evaluator.add_parallel(
        id=f"{pid}_Academic_Requirements",
        desc="Program has documented academic admission and curriculum requirements",
        parent=prog_node,
        critical=False
    )

    # GPA Requirement (non-critical, parallel)
    gpa_node = evaluator.add_parallel(
        id=f"{pid}_GPA_Requirement",
        desc="Documented minimum GPA requirement for admission or progression",
        parent=acad_node,
        critical=False
    )

    gpa_val_leaf = evaluator.add_leaf(
        id=f"{pid}_GPA_Value",
        desc="Specific GPA threshold stated in program materials",
        parent=gpa_node,
        critical=False
    )
    gpa_val_claim = f"The minimum GPA requirement is {_safe(program.academics.gpa_value)} for admission or progression."
    await evaluator.verify(
        claim=gpa_val_claim,
        node=gpa_val_leaf,
        sources=program.academics.gpa_url,
        additional_instruction="Verify the GPA threshold as shown on the provided page. Minor rounding differences are acceptable."
    )

    gpa_url_leaf = evaluator.add_leaf(
        id=f"{pid}_GPA_URL_Reference",
        desc="URL reference for GPA requirement documentation",
        parent=gpa_node,
        critical=False
    )
    gpa_url_claim = "This page documents the minimum GPA requirement for admission or progression in the engineering program."
    await evaluator.verify(
        claim=gpa_url_claim,
        node=gpa_url_leaf,
        sources=program.academics.gpa_url,
        additional_instruction="The page should explicitly state a GPA requirement."
    )

    # Mathematics Requirement (non-critical, parallel)
    math_node = evaluator.add_parallel(
        id=f"{pid}_Mathematics_Requirement",
        desc="Program requires calculus coursework",
        parent=acad_node,
        critical=False
    )

    math_courses_leaf = evaluator.add_leaf(
        id=f"{pid}_Math_Courses",
        desc="Documentation of required mathematics courses (Calculus I, II minimum)",
        parent=math_node,
        critical=False
    )
    math_courses_claim = (
        f"Required mathematics courses for the program include {_safe(program.academics.math_courses)}, "
        f"which include or are equivalent to Calculus I and Calculus II."
    )
    await evaluator.verify(
        claim=math_courses_claim,
        node=math_courses_leaf,
        sources=program.academics.math_url,
        additional_instruction="Verify that the curriculum includes calculus (I and II or equivalent)."
    )

    math_url_leaf = evaluator.add_leaf(
        id=f"{pid}_Math_URL_Reference",
        desc="URL reference for mathematics requirements",
        parent=math_node,
        critical=False
    )
    math_url_claim = "This page documents the required mathematics/curriculum for the engineering program."
    await evaluator.verify(
        claim=math_url_claim,
        node=math_url_leaf,
        sources=program.academics.math_url,
        additional_instruction="The page should be an official curriculum or catalog page."
    )

    # Capstone Requirement (non-critical, parallel)
    cap_node = evaluator.add_parallel(
        id=f"{pid}_Capstone_Requirement",
        desc="Program includes mandatory capstone design project",
        parent=acad_node,
        critical=False
    )

    cap_desc_leaf = evaluator.add_leaf(
        id=f"{pid}_Capstone_Description",
        desc="Description of capstone or senior design project requirement",
        parent=cap_node,
        critical=False
    )
    cap_desc_claim = (
        f"The program requires a capstone or senior design project: {_safe(program.academics.capstone_desc)}."
    )
    await evaluator.verify(
        claim=cap_desc_claim,
        node=cap_desc_leaf,
        sources=program.academics.capstone_url,
        additional_instruction="Verify that the page describes a capstone/senior design requirement."
    )

    cap_url_leaf = evaluator.add_leaf(
        id=f"{pid}_Capstone_URL_Reference",
        desc="URL reference for capstone requirement documentation",
        parent=cap_node,
        critical=False
    )
    cap_url_claim = "This page documents the capstone/senior design requirement for the engineering program."
    await evaluator.verify(
        claim=cap_url_claim,
        node=cap_url_leaf,
        sources=program.academics.capstone_url,
        additional_instruction="The page should be official (catalog or department page) describing the requirement."
    )

    # Cost Documentation (non-critical, parallel)
    cost_node = evaluator.add_parallel(
        id=f"{pid}_Cost_Documentation",
        desc="Program has publicly available cost information",
        parent=prog_node,
        critical=False
    )

    tuition_node = evaluator.add_parallel(
        id=f"{pid}_Tuition_Costs",
        desc="Documented annual tuition costs for in-state or out-of-state students",
        parent=cost_node,
        critical=False
    )

    tuition_amt_leaf = evaluator.add_leaf(
        id=f"{pid}_Tuition_Amount",
        desc="Specific tuition amount or range",
        parent=tuition_node,
        critical=False
    )
    tuition_amt_claim = f"Annual tuition costs are {_safe(program.costs.tuition_amount)}."
    await evaluator.verify(
        claim=tuition_amt_claim,
        node=tuition_amt_leaf,
        sources=program.costs.tuition_url,
        additional_instruction="Verify tuition amounts or ranges from the official tuition page."
    )

    tuition_url_leaf = evaluator.add_leaf(
        id=f"{pid}_Tuition_URL_Reference",
        desc="URL reference for tuition information",
        parent=tuition_node,
        critical=False
    )
    tuition_url_claim = "This page provides official tuition information."
    await evaluator.verify(
        claim=tuition_url_claim,
        node=tuition_url_leaf,
        sources=program.costs.tuition_url,
        additional_instruction="Should be a bursar/tuition office or official financial page."
    )

    housing_node = evaluator.add_parallel(
        id=f"{pid}_Housing_Costs",
        desc="Documented on-campus housing costs",
        parent=cost_node,
        critical=False
    )

    housing_amt_leaf = evaluator.add_leaf(
        id=f"{pid}_Housing_Amount",
        desc="Specific housing cost amount or range",
        parent=housing_node,
        critical=False
    )
    housing_amt_claim = f"On-campus housing costs are {_safe(program.costs.housing_amount)}."
    await evaluator.verify(
        claim=housing_amt_claim,
        node=housing_amt_leaf,
        sources=program.costs.housing_url,
        additional_instruction="Verify housing cost amounts or ranges from the official housing page."
    )

    housing_url_leaf = evaluator.add_leaf(
        id=f"{pid}_Housing_URL_Reference",
        desc="URL reference for housing cost information",
        parent=housing_node,
        critical=False
    )
    housing_url_claim = "This page provides official on-campus housing cost information."
    await evaluator.verify(
        claim=housing_url_claim,
        node=housing_url_leaf,
        sources=program.costs.housing_url,
        additional_instruction="Should be a housing/residential life or official cost page."
    )

    # Career Outcomes (non-critical, parallel)
    career_node = evaluator.add_parallel(
        id=f"{pid}_Career_Outcomes",
        desc="Program has documented career outcomes information",
        parent=prog_node,
        critical=False
    )

    placement_node = evaluator.add_parallel(
        id=f"{pid}_Placement_Rate",
        desc="Documented job placement or career outcomes data",
        parent=career_node,
        critical=False
    )

    placement_data_leaf = evaluator.add_leaf(
        id=f"{pid}_Placement_Data",
        desc="Specific placement rate or outcome statistics",
        parent=placement_node,
        critical=False
    )
    placement_data_claim = f"Career outcomes data indicates {_safe(program.outcomes.placement_data)}."
    await evaluator.verify(
        claim=placement_data_claim,
        node=placement_data_leaf,
        sources=program.outcomes.placement_url,
        additional_instruction="Verify the placement/outcomes data (e.g., % employed) from the official outcomes page."
    )

    placement_url_leaf = evaluator.add_leaf(
        id=f"{pid}_Placement_URL_Reference",
        desc="URL reference for placement data",
        parent=placement_node,
        critical=False
    )
    placement_url_claim = "This page provides official career placement or outcomes information."
    await evaluator.verify(
        claim=placement_url_claim,
        node=placement_url_leaf,
        sources=program.outcomes.placement_url,
        additional_instruction="Should be a career services or college outcomes page."
    )

    salary_node = evaluator.add_parallel(
        id=f"{pid}_Starting_Salary",
        desc="Documented average starting salary for graduates",
        parent=career_node,
        critical=False
    )

    salary_amt_leaf = evaluator.add_leaf(
        id=f"{pid}_Salary_Amount",
        desc="Specific salary amount or range",
        parent=salary_node,
        critical=False
    )
    salary_amt_claim = f"Average starting salary for graduates is {_safe(program.outcomes.starting_salary)}."
    await evaluator.verify(
        claim=salary_amt_claim,
        node=salary_amt_leaf,
        sources=program.outcomes.salary_url,
        additional_instruction="Verify the average starting salary amount or range from the official outcomes/salary page."
    )

    salary_url_leaf = evaluator.add_leaf(
        id=f"{pid}_Salary_URL_Reference",
        desc="URL reference for salary information",
        parent=salary_node,
        critical=False
    )
    salary_url_claim = "This page provides official starting salary information for graduates."
    await evaluator.verify(
        claim=salary_url_claim,
        node=salary_url_leaf,
        sources=program.outcomes.salary_url,
        additional_instruction="Should be an outcomes or salary information page."
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
    Evaluate an answer for the ABET programs task.
    """
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

    # Extract programs from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_programs(),
        template_class=ProgramsExtraction,
        extraction_name="programs_extraction"
    )

    # Ensure we have exactly 4 programs (pad with empty if fewer; slice if more)
    programs = list(extracted.programs[:4])
    while len(programs) < 4:
        programs.append(ProgramInfo())

    # Build verification tree for each program
    used_states: Set[str] = set()
    for i in range(4):
        await verify_program(
            evaluator=evaluator,
            parent_node=root,
            program=programs[i],
            idx=i,
            used_states=used_states
        )

    return evaluator.get_summary()