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
TASK_ID = "three_large_school_districts_west_mississippi"
TASK_DESCRIPTION = """
Identify three public school districts in the United States that meet ALL of the following criteria:

1. Each district must be located in a different U.S. state
2. Each state must be located west of the Mississippi River
3. Each district must have total student enrollment of at least 150,000 students (based on the most recent available data from the 2023-2024 or 2024-2025 school year)
4. At least one of the three districts must be located in a state that requires standardized exit exams for high school graduation (Note: As of 2024-2025, only six states require such exams: Florida, Louisiana, New Jersey, Ohio, Texas, and Virginia)

For each of the three districts, provide the following information with supporting reference URLs:
- The official name of the school district
- The state where the district is located
- The exact total student enrollment figure for the 2023-2024 or 2024-2025 school year
- Whether the state requires standardized exit exams for high school graduation
- The total number of schools in the district (if publicly available)

All factual claims must be supported by reference URLs from official district websites, government education databases, or reliable news sources.
"""

EXIT_EXAM_STATES_2024_25 = {
    "FLORIDA", "LOUISIANA", "NEW JERSEY", "OHIO", "TEXAS", "VIRGINIA"
}

# States counted as west of the Mississippi River for this task (includes states that straddle the river but have territory west of it)
WEST_OF_MISSISSIPPI_STATES = {
    "ALASKA", "ARIZONA", "ARKANSAS", "CALIFORNIA", "COLORADO", "HAWAII", "IDAHO",
    "IOWA", "KANSAS", "LOUISIANA", "MINNESOTA", "MISSOURI", "MONTANA", "NEBRASKA",
    "NEVADA", "NEW MEXICO", "NORTH DAKOTA", "OKLAHOMA", "OREGON", "SOUTH DAKOTA",
    "TEXAS", "UTAH", "WASHINGTON", "WYOMING"
}

STATE_ABBR_TO_NAME = {
    "AK": "ALASKA", "AZ": "ARIZONA", "AR": "ARKANSAS", "CA": "CALIFORNIA", "CO": "COLORADO",
    "HI": "HAWAII", "ID": "IDAHO", "IA": "IOWA", "KS": "KANSAS", "LA": "LOUISIANA",
    "MN": "MINNESOTA", "MO": "MISSOURI", "MT": "MONTANA", "NE": "NEBRASKA",
    "NV": "NEVADA", "NM": "NEW MEXICO", "ND": "NORTH DAKOTA", "OK": "OKLAHOMA",
    "OR": "OREGON", "SD": "SOUTH DAKOTA", "TX": "TEXAS", "UT": "UTAH",
    "WA": "WASHINGTON", "WY": "WYOMING",
    # Include all for normalization completeness (east-of for robustness, though not used for west-of logic)
    "AL": "ALABAMA", "CT": "CONNECTICUT", "DE": "DELAWARE", "FL": "FLORIDA", "GA": "GEORGIA",
    "IL": "ILLINOIS", "IN": "INDIANA", "KY": "KENTUCKY", "ME": "MAINE", "MD": "MARYLAND",
    "MA": "MASSACHUSETTS", "MI": "MICHIGAN", "MS": "MISSISSIPPI", "NH": "NEW HAMPSHIRE",
    "NJ": "NEW JERSEY", "NY": "NEW YORK", "NC": "NORTH CAROLINA", "OH": "OHIO",
    "PA": "PENNSYLVANIA", "RI": "RHODE ISLAND", "SC": "SOUTH CAROLINA", "TN": "TENNESSEE",
    "VT": "VERMONT", "VA": "VIRGINIA", "WI": "WISCONSIN", "WV": "WEST VIRGINIA"
}


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class DistrictItem(BaseModel):
    name: Optional[str] = None
    state: Optional[str] = None
    enrollment: Optional[str] = None  # Keep as string to allow ranges/formatting
    enrollment_year: Optional[str] = None  # e.g., "2023-2024" or "2024-2025"
    number_of_schools: Optional[str] = None  # Keep as string
    exit_exam_required: Optional[str] = None  # "yes"/"no" or equivalent text

    # Source URLs
    identity_urls: List[str] = Field(default_factory=list)      # For name/identity/location confirmation
    enrollment_urls: List[str] = Field(default_factory=list)    # For enrollment figure confirmation
    exit_exam_urls: List[str] = Field(default_factory=list)     # For state exit exam policy confirmation
    size_urls: List[str] = Field(default_factory=list)          # For number of schools confirmation


class DistrictsExtraction(BaseModel):
    districts: List[DistrictItem] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def normalize_state_name(s: Optional[str]) -> Optional[str]:
    if not s:
        return None
    s = s.strip().upper()
    # Convert common punctuation variants
    s = s.replace(".", "")
    # Map abbreviations to full names
    if s in STATE_ABBR_TO_NAME:
        return STATE_ABBR_TO_NAME[s]
    return s  # Assume already a full state name in uppercase


def is_state_west_of_mississippi(state: Optional[str]) -> bool:
    norm = normalize_state_name(state)
    if not norm:
        return False
    return norm in WEST_OF_MISSISSIPPI_STATES


def parse_yes_no(value: Optional[str]) -> Optional[bool]:
    if value is None:
        return None
    s = value.strip().lower()
    # Positive forms
    positives = {"yes", "y", "true", "required", "requires", "require", "state requires", "mandatory", "required exit exam", "requires exit exam"}
    negatives = {"no", "n", "false", "not required", "does not require", "doesn't require", "optional", "no exit exam", "does not require exit exam"}
    if s in positives:
        return True
    if s in negatives:
        return False
    # Heuristic
    if "require" in s and "not" not in s and "does not" not in s and "no " not in s:
        return True
    if "does not require" in s or "not require" in s or "no " in s:
        return False
    return None


def merge_sources(*lists: List[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for lst in lists:
        for u in lst or []:
            if u and u not in seen:
                seen.add(u)
                out.append(u)
    return out


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_districts() -> str:
    return """
    Extract up to three (3) public school districts described in the answer. For each district, return:
    - name: The official district name exactly as stated.
    - state: The U.S. state where the district is located. Prefer full state name (e.g., "Texas"). If the answer uses a postal code (e.g., "TX"), keep it as provided.
    - enrollment: The exact total student enrollment figure text for the 2023-2024 or 2024-2025 school year (as stated in the answer, do not modify). If a range or approximate phrase is used, extract that string verbatim.
    - enrollment_year: The specified school year tied to the enrollment (e.g., "2023-2024" or "2024-2025"). If the answer mentions "2023-24" or "2024-25", keep that format.
    - number_of_schools: The total number of schools in the district, if mentioned. If not mentioned, set to null.
    - exit_exam_required: Whether the state requires standardized exit exams for high school graduation as claimed in the answer. Normalize to a short label if possible: "yes" or "no". If unspecified, set to null.

    Also extract supporting URLs (as they appear in the answer) for each claim:
    - identity_urls: URL(s) that confirm the official district name and location (official district site or authoritative source).
    - enrollment_urls: URL(s) that state the total enrollment figure for the given school year.
    - exit_exam_urls: URL(s) that state the state's exit exam requirement status (state DOE, statutes, reputable policy summaries, or reliable news).
    - size_urls: URL(s) that state the total number of schools, if provided.

    Rules:
    - Extract only URLs explicitly present in the answer. Do not invent URLs.
    - If more than three districts are present, only keep the first three.
    - If a field is missing in the answer, set it to null (or [] for URL lists).
    - Do not perform any calculations. Keep numbers and text exactly as shown in the answer.
    """

# --------------------------------------------------------------------------- #
# Verification subroutines                                                    #
# --------------------------------------------------------------------------- #
async def verify_district(
    evaluator: Evaluator,
    parent_node,
    district: DistrictItem,
    index: int,
) -> None:
    idx = index + 1
    # Parent node for this district
    district_node = evaluator.add_parallel(
        id=f"District_{idx}",
        desc=f"{['First', 'Second', 'Third'][index] if index < 3 else f'#{idx}'} qualifying school district",
        parent=parent_node,
        critical=False
    )

    # 1) Identity (critical, parallel)
    identity_node = evaluator.add_parallel(
        id=f"District_{idx}_Identity",
        desc=f"Official name and identity of the {['first','second','third'][index] if index < 3 else f'#{idx}'} district",
        parent=district_node,
        critical=True
    )

    # 1.a) Name provided (critical)
    name_present = bool(district.name and district.name.strip())
    evaluator.add_custom_node(
        result=name_present,
        id=f"District_{idx}_Name",
        desc="Official district name provided",
        parent=identity_node,
        critical=True
    )

    # 1.b) Name URL supports the name (critical)
    if name_present and district.identity_urls:
        leaf = evaluator.add_leaf(
            id=f"District_{idx}_Name_URL",
            desc="Reference URL confirming district name",
            parent=identity_node,
            critical=True
        )
        claim = f"The official name of the school district is '{district.name}'."
        await evaluator.verify(
            claim=claim,
            node=leaf,
            sources=district.identity_urls,
            additional_instruction="Verify that the page explicitly shows the district's official name, allowing minor formatting and punctuation variations."
        )
    else:
        evaluator.add_custom_node(
            result=False,
            id=f"District_{idx}_Name_URL",
            desc="Reference URL confirming district name",
            parent=identity_node,
            critical=True
        )

    # 2) Location (critical, parallel)
    location_node = evaluator.add_parallel(
        id=f"District_{idx}_Location",
        desc=f"Geographic location verification for {['first','second','third'][index] if index < 3 else f'#{idx}'} district",
        parent=district_node,
        critical=True
    )

    # 2.a) State location provided and supported (critical)
    merged_loc_sources = merge_sources(district.identity_urls, district.enrollment_urls)
    if district.state and merged_loc_sources and name_present:
        leaf = evaluator.add_leaf(
            id=f"District_{idx}_State",
            desc="State location provided",
            parent=location_node,
            critical=True
        )
        claim = f"The school district named '{district.name}' is located in the U.S. state of {district.state}."
        await evaluator.verify(
            claim=claim,
            node=leaf,
            sources=merged_loc_sources,
            additional_instruction="Confirm the district's state as indicated on the official or authoritative page. Minor variants like abbreviations are fine."
        )
    else:
        evaluator.add_custom_node(
            result=False,
            id=f"District_{idx}_State",
            desc="State location provided",
            parent=location_node,
            critical=True
        )

    # 2.b) West of Mississippi confirmation (critical) - logic-based
    evaluator.add_custom_node(
        result=is_state_west_of_mississippi(district.state),
        id=f"District_{idx}_West_of_Mississippi",
        desc="Confirmation that state is west of Mississippi River",
        parent=location_node,
        critical=True
    )

    # 3) Enrollment (critical, parallel)
    enrollment_node = evaluator.add_parallel(
        id=f"District_{idx}_Enrollment",
        desc=f"Enrollment verification for {['first','second','third'][index] if index < 3 else f'#{idx}'} district",
        parent=district_node,
        critical=True
    )

    # 3.c) Enrollment URL exists (critical)
    evaluator.add_custom_node(
        result=bool(district.enrollment_urls),
        id=f"District_{idx}_Enrollment_URL",
        desc="Reference URL confirming enrollment figure",
        parent=enrollment_node,
        critical=True
    )

    # 3.b) Exact enrollment figure for 2023-24 or 2024-25 (critical)
    has_year = bool(district.enrollment_year and district.enrollment_year.strip())
    has_enrollment = bool(district.enrollment and district.enrollment.strip())
    if has_enrollment and has_year and district.enrollment_urls and name_present:
        leaf = evaluator.add_leaf(
            id=f"District_{idx}_Enrollment_Figure",
            desc="Exact enrollment figure provided for 2023-24 or 2024-25 school year",
            parent=enrollment_node,
            critical=True
        )
        claim = f"For the {district.enrollment_year} school year, the total student enrollment of '{district.name}' is {district.enrollment}."
        await evaluator.verify(
            claim=claim,
            node=leaf,
            sources=district.enrollment_urls,
            additional_instruction="Accept minor rounding or formatting differences (e.g., 197,000 vs. 197k) as long as it reflects the same figure for the stated school year."
        )
    else:
        evaluator.add_custom_node(
            result=False,
            id=f"District_{idx}_Enrollment_Figure",
            desc="Exact enrollment figure provided for 2023-24 or 2024-25 school year",
            parent=enrollment_node,
            critical=True
        )

    # 3.a) Enrollment meets threshold >= 150,000 (critical)
    if has_enrollment and has_year and district.enrollment_urls and name_present:
        leaf = evaluator.add_leaf(
            id=f"District_{idx}_Enrollment_Threshold",
            desc="District enrollment meets or exceeds 150,000 students",
            parent=enrollment_node,
            critical=True
        )
        claim = f"According to the cited sources, '{district.name}' has total student enrollment of {district.enrollment} in {district.enrollment_year}, which is at least 150,000 students."
        await evaluator.verify(
            claim=claim,
            node=leaf,
            sources=district.enrollment_urls,
            additional_instruction="Judge whether the stated enrollment for the given school year is >= 150,000. Allow minor rounding as long as the underlying figure clearly meets/exceeds 150,000."
        )
    else:
        evaluator.add_custom_node(
            result=False,
            id=f"District_{idx}_Enrollment_Threshold",
            desc="District enrollment meets or exceeds 150,000 students",
            parent=enrollment_node,
            critical=True
        )

    # 4) Exit exam status (critical, parallel)
    exit_node = evaluator.add_parallel(
        id=f"District_{idx}_Exit_Exam_Status",
        desc=f"State exit exam requirement status for {['first','second','third'][index] if index < 3 else f'#{idx}'} district",
        parent=district_node,
        critical=True
    )

    # 4.a) Requirement value provided (critical)
    exit_val = parse_yes_no(district.exit_exam_required)
    evaluator.add_custom_node(
        result=(exit_val is not None),
        id=f"District_{idx}_Exit_Exam_Requirement",
        desc="Whether the state requires standardized exit exams for high school graduation is provided",
        parent=exit_node,
        critical=True
    )

    # 4.b) Requirement supported by URL(s) (critical)
    if exit_val is not None and district.exit_exam_urls and district.state:
        leaf = evaluator.add_leaf(
            id=f"District_{idx}_Exit_Exam_URL",
            desc="Reference URL confirming state's exit exam requirement status",
            parent=exit_node,
            critical=True
        )
        state_norm = normalize_state_name(district.state)
        if exit_val:
            claim = f"As of the 2024-2025 school year, the state of {state_norm or district.state} requires standardized exit exams for high school graduation."
        else:
            claim = f"As of the 2024-2025 school year, the state of {state_norm or district.state} does not require standardized exit exams for high school graduation."
        await evaluator.verify(
            claim=claim,
            node=leaf,
            sources=district.exit_exam_urls,
            additional_instruction="Use state DOE pages, official statutes/regulations, or reliable policy summaries/news confirming the exit exam requirement status as of 2024-2025."
        )
    else:
        evaluator.add_custom_node(
            result=False,
            id=f"District_{idx}_Exit_Exam_URL",
            desc="Reference URL confirming state's exit exam requirement status",
            parent=exit_node,
            critical=True
        )

    # 5) Size details (non-critical, parallel)
    size_node = evaluator.add_parallel(
        id=f"District_{idx}_Size_Details",
        desc=f"Additional size information for {['first','second','third'][index] if index < 3 else f'#{idx}'} district",
        parent=district_node,
        critical=False
    )

    # 5.a) Number of schools (non-critical)
    if district.number_of_schools and district.size_urls and name_present:
        leaf = evaluator.add_leaf(
            id=f"District_{idx}_Number_of_Schools",
            desc="Total number of schools in the district (if publicly available)",
            parent=size_node,
            critical=False
        )
        claim = f"The school district '{district.name}' has {district.number_of_schools} schools."
        await evaluator.verify(
            claim=claim,
            node=leaf,
            sources=district.size_urls,
            additional_instruction="Verify that the total count of schools matches the cited source (accept minor lag if the source is the official district page)."
        )
    else:
        evaluator.add_custom_node(
            result=False,
            id=f"District_{idx}_Number_of_Schools",
            desc="Total number of schools in the district (if publicly available)",
            parent=size_node,
            critical=False
        )

    # 5.b) Size URL present (non-critical)
    evaluator.add_custom_node(
        result=bool(district.size_urls),
        id=f"District_{idx}_Size_URL",
        desc="Reference URL confirming number of schools (if provided)",
        parent=size_node,
        critical=False
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
    Evaluate an answer for the 'three large public school districts west of Mississippi' task.
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
        default_model=model,
    )

    # Extract district info
    extracted: DistrictsExtraction = await evaluator.extract(
        prompt=prompt_extract_districts(),
        template_class=DistrictsExtraction,
        extraction_name="districts_extraction"
    )

    # Keep only first 3 districts; pad if fewer
    districts = (extracted.districts or [])[:3]
    while len(districts) < 3:
        districts.append(DistrictItem())

    # Record task ground-truth context/set definitions (for transparency)
    evaluator.add_ground_truth({
        "west_of_mississippi_states": sorted(list(WEST_OF_MISSISSIPPI_STATES)),
        "exit_exam_states_2024_25": sorted(list(EXIT_EXAM_STATES_2024_25)),
        "requirement_years": ["2023-2024", "2024-2025"]
    }, gt_type="policy_sets")

    # Top-level rubric node
    top = evaluator.add_parallel(
        id="Three_Qualifying_Large_School_Districts",
        desc="Task requires identifying three large public school districts in states west of the Mississippi River, each meeting specific enrollment and geographic criteria",
        parent=root,
        critical=False
    )

    # Verify each district subtree
    for i in range(3):
        await verify_district(evaluator, top, districts[i], i)

    # Cross-district constraints (critical)
    cross = evaluator.add_parallel(
        id="Cross_District_Constraints",
        desc="Constraints that apply across all three districts collectively",
        parent=top,
        critical=True
    )

    # Normalize states for cross checks
    norm_states = [normalize_state_name(d.state) or "" for d in districts]
    # 1) All different states (critical)
    all_diff = len([s for s in norm_states if s]) == len(set([s for s in norm_states if s])) and all(s for s in norm_states)
    evaluator.add_custom_node(
        result=all_diff,
        id="All_Different_States",
        desc="All three districts are located in different U.S. states",
        parent=cross,
        critical=True
    )

    # 2) At least one district in an exit-exam state (critical)
    # Use extracted exit_exam_required primarily (yes/no), otherwise fall back to known set by state names if provided.
    any_exit_yes_from_answer = any(parse_yes_no(d.exit_exam_required) is True for d in districts)
    any_state_in_known_exit_set = any((s in EXIT_EXAM_STATES_2024_25) for s in norm_states if s)
    evaluator.add_custom_node(
        result=bool(any_exit_yes_from_answer or any_state_in_known_exit_set),
        id="At_Least_One_Exit_Exam_State",
        desc="At least one of the three districts is in a state that requires standardized exit exams for high school graduation",
        parent=cross,
        critical=True
    )

    return evaluator.get_summary()