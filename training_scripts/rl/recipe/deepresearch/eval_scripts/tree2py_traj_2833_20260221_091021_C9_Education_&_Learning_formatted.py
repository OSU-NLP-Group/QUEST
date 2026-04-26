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
TASK_ID = "western_public_universities_fbs"
TASK_DESCRIPTION = (
    "Identify three public universities located in different states within the western United States "
    "(Arizona, California, Colorado, Idaho, Montana, Nevada, New Mexico, Oregon, Utah, Washington, or Wyoming) "
    "that meet all of the following criteria:\n\n"
    "1. Institutional Status: Each university must be a public (state-funded) institution and a member of a state university system.\n"
    "2. NCAA Athletics: Each university must be classified as NCAA Division I Football Bowl Subdivision (FBS) and must sponsor at least 16 varsity sports programs.\n"
    "3. Conference Membership: Each university must be a current member of an FBS athletic conference.\n"
    "4. Football Program: Each university must have an active football program with a current head coach whose name can be identified, and must have a football stadium with a published seating capacity.\n"
    "5. Academic Information: Each university must have publicly available current enrollment data (undergraduate or total enrollment for 2024-25 academic year) and published admission requirements (including GPA requirements, test score requirements, or test-optional policies).\n"
    "6. Geographic Diversity: The three universities must be located in three different states.\n\n"
    "For each university, provide:\n"
    "- University name\n"
    "- State location\n"
    "- State university system affiliation\n"
    "- Current enrollment figure\n"
    "- NCAA Division I FBS status\n"
    "- Athletic conference affiliation\n"
    "- Number of varsity sports (if available, or confirmation that it meets the 16-sport minimum)\n"
    "- Current football head coach name\n"
    "- Football stadium seating capacity\n"
    "- Admission requirements (GPA, test scores, or policy description)\n"
    "- Reference URLs supporting each piece of information"
)

WESTERN_STATES = {
    "AZ", "ARIZONA",
    "CA", "CALIFORNIA",
    "CO", "COLORADO",
    "ID", "IDAHO",
    "MT", "MONTANA",
    "NV", "NEVADA",
    "NM", "NEW MEXICO",
    "OR", "OREGON",
    "UT", "UTAH",
    "WA", "WASHINGTON",
    "WY", "WYOMING",
}


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class UniversitySourceLinks(BaseModel):
    location_urls: List[str] = Field(default_factory=list)
    institution_urls: List[str] = Field(default_factory=list)
    enrollment_urls: List[str] = Field(default_factory=list)
    ncaa_status_urls: List[str] = Field(default_factory=list)
    conference_urls: List[str] = Field(default_factory=list)
    sports_count_urls: List[str] = Field(default_factory=list)
    head_coach_urls: List[str] = Field(default_factory=list)
    stadium_urls: List[str] = Field(default_factory=list)
    admissions_urls: List[str] = Field(default_factory=list)


class UniversityInfo(BaseModel):
    name: Optional[str] = None
    state: Optional[str] = None  # Either full state name or USPS abbreviation is acceptable
    system_affiliation: Optional[str] = None
    enrollment: Optional[str] = None  # Prefer string to allow ranges or approximate values
    ncaa_fbs_status: Optional[str] = None  # e.g., "NCAA Division I FBS" or similar wording
    conference: Optional[str] = None
    varsity_sports_count: Optional[str] = None  # Can be numeric string or text "at least 16"
    head_coach: Optional[str] = None
    stadium_capacity: Optional[str] = None
    admissions_requirements: Optional[str] = None
    sources: UniversitySourceLinks = Field(default_factory=UniversitySourceLinks)


class UniversitiesExtraction(BaseModel):
    universities: List[UniversityInfo] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_universities() -> str:
    return (
        "Extract up to three universities from the answer that satisfy the task. For each, provide the following fields:\n"
        "1) name\n"
        "2) state (either full name or two-letter abbreviation)\n"
        "3) system_affiliation (e.g., 'University of California', 'California State University', 'Oregon University System', etc.)\n"
        "4) enrollment (current undergraduate or total enrollment figure for the 2024-25 academic year, or the most recent figure stated)\n"
        "5) ncaa_fbs_status (should indicate NCAA Division I FBS)\n"
        "6) conference (current FBS athletic conference)\n"
        "7) varsity_sports_count (if a number is provided, include it; otherwise, a textual confirmation like 'at least 16')\n"
        "8) head_coach (current football head coach name)\n"
        "9) stadium_capacity (football stadium published seating capacity)\n"
        "10) admissions_requirements (GPA/test score requirements or test-optional policy description)\n"
        "11) sources: Provide arrays of URLs for each facet: location_urls, institution_urls, enrollment_urls, ncaa_status_urls, "
        "conference_urls, sports_count_urls, head_coach_urls, stadium_urls, admissions_urls.\n\n"
        "Rules:\n"
        "- Only extract information explicitly present in the answer. Do not invent data.\n"
        "- If a particular URL is not provided, return an empty array for that facet.\n"
        "- If the answer lists more than three universities, return only the first three.\n"
        "- If any field is missing, set it to null.\n"
    )


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def normalize_state_str(s: Optional[str]) -> Optional[str]:
    if not s:
        return None
    t = s.strip().upper()
    # Normalize common forms to abbreviations when possible for consistency
    mapping = {
        "ARIZONA": "AZ", "CALIFORNIA": "CA", "COLORADO": "CO", "IDAHO": "ID", "MONTANA": "MT",
        "NEVADA": "NV", "NEW MEXICO": "NM", "OREGON": "OR", "UTAH": "UT", "WASHINGTON": "WA", "WYOMING": "WY"
    }
    if t in mapping:
        return mapping[t]
    return t


def is_western_state(s: Optional[str]) -> bool:
    if not s:
        return False
    ns = normalize_state_str(s)
    return ns in WESTERN_STATES


# --------------------------------------------------------------------------- #
# Verification for a single university                                        #
# --------------------------------------------------------------------------- #
async def verify_university(
    evaluator: Evaluator,
    parent_node,
    uni: UniversityInfo,
    index: int,
    prev_states: List[Optional[str]],
) -> None:
    # Create University node (non-critical, allows partial credit across universities)
    uni_node = evaluator.add_parallel(
        id=f"University_{index+1}",
        desc=f"{['First','Second','Third'][index]} qualifying university with all required attributes",
        parent=parent_node,
        critical=False,
    )

    # ---------------------- Basic Institutional Criteria ---------------------- #
    basic_node = evaluator.add_parallel(
        id=f"Basic_Institutional_Criteria_U{index+1}",
        desc=f"Core institutional characteristics for University {index+1}",
        parent=uni_node,
        critical=True,
    )

    # Geographic location sub-node
    geo_node = evaluator.add_parallel(
        id=f"Geographic_Location_U{index+1}",
        desc=f"Geographic requirements for University {index+1}",
        parent=basic_node,
        critical=True,
    )

    # State western US check
    state_leaf = evaluator.add_leaf(
        id=f"State_Western_US_U{index+1}",
        desc=f"University {index+1} is located in a western US state (AZ, CA, CO, ID, MT, NV, NM, OR, UT, WA, WY)",
        parent=geo_node,
        critical=True,
    )
    state_val = uni.state or ""
    claim_state_allowed = (
        f"The state '{state_val}' is within the allowed western US list: "
        f"Arizona (AZ), California (CA), Colorado (CO), Idaho (ID), Montana (MT), Nevada (NV), "
        f"New Mexico (NM), Oregon (OR), Utah (UT), Washington (WA), or Wyoming (WY)."
    )
    await evaluator.verify(
        claim=claim_state_allowed,
        node=state_leaf,
        additional_instruction="Treat either full state names or two-letter abbreviations as acceptable equivalents."
    )

    # Different state constraint for U2 and U3
    if index == 1:
        diff_state_leaf = evaluator.add_leaf(
            id=f"Different_State_U{index+1}",
            desc="University 2 is in a different state than University 1",
            parent=geo_node,
            critical=True,
        )
        prev_state = prev_states[0] or ""
        claim_diff_2 = f"The state '{state_val}' for University 2 is different from the state '{prev_state}' for University 1."
        await evaluator.verify(
            claim=claim_diff_2,
            node=diff_state_leaf,
            additional_instruction="This is a logical comparison only. If either state is missing, return Incorrect."
        )
    if index == 2:
        diff_state_leaf = evaluator.add_leaf(
            id=f"Different_State_U{index+1}",
            desc="University 3 is in a different state than Universities 1 and 2",
            parent=geo_node,
            critical=True,
        )
        prev_state_1 = prev_states[0] or ""
        prev_state_2 = prev_states[1] or ""
        claim_diff_3 = (
            f"The state '{state_val}' for University 3 is different from the state '{prev_state_1}' for University 1 "
            f"and different from the state '{prev_state_2}' for University 2."
        )
        await evaluator.verify(
            claim=claim_diff_3,
            node=diff_state_leaf,
            additional_instruction="This is a logical comparison only. If any state is missing, return Incorrect."
        )

    # Location URL support
    loc_url_leaf = evaluator.add_leaf(
        id=f"Location_URL_U{index+1}",
        desc=f"Provides URL verifying University {index+1}'s geographic location",
        parent=geo_node,
        critical=True,
    )
    claim_loc_url = f"{uni.name or 'The university'} is located in {state_val}."
    await evaluator.verify(
        claim=claim_loc_url,
        node=loc_url_leaf,
        sources=uni.sources.location_urls,
        additional_instruction=(
            "Only PASS if the provided URL(s) explicitly state the university's location (city/state) or clearly indicate the state. "
            "If no URL is provided, return NOT SUPPORTED / Incorrect."
        ),
    )

    # Public institution status
    pub_node = evaluator.add_parallel(
        id=f"Public_Institution_Status_U{index+1}",
        desc=f"Institutional type verification for University {index+1}",
        parent=basic_node,
        critical=True,
    )

    pub_uni_leaf = evaluator.add_leaf(
        id=f"Public_University_U{index+1}",
        desc=f"University {index+1} is a public (state-funded) institution",
        parent=pub_node,
        critical=True,
    )
    claim_public = f"{uni.name or 'The university'} is a public, state-funded institution."
    await evaluator.verify(
        claim=claim_public,
        node=pub_uni_leaf,
        sources=uni.sources.institution_urls,
        additional_instruction=(
            "Confirm that the page explicitly labels the institution as public/state-funded. "
            "If no URL is provided, return Incorrect."
        ),
    )

    system_member_leaf = evaluator.add_leaf(
        id=f"State_System_Member_U{index+1}",
        desc=f"University {index+1} is part of a state university system",
        parent=pub_node,
        critical=True,
    )
    claim_system = (
        f"{uni.name or 'The university'} is a member of a state university system"
        + (f" (affiliation: {uni.system_affiliation})." if uni.system_affiliation else ".")
    )
    await evaluator.verify(
        claim=claim_system,
        node=system_member_leaf,
        sources=uni.sources.institution_urls,
        additional_instruction=(
            "Look for official system membership indications (e.g., University of California, California State University, etc.). "
            "If no URL is provided, return Incorrect."
        ),
    )

    inst_type_url_leaf = evaluator.add_leaf(
        id=f"Institution_Type_URL_U{index+1}",
        desc=f"Provides URL verifying University {index+1}'s public status and system membership",
        parent=pub_node,
        critical=True,
    )
    claim_inst_url = (
        f"The provided page(s) confirm that {uni.name or 'the university'} is public and part of a state university system."
    )
    await evaluator.verify(
        claim=claim_inst_url,
        node=inst_type_url_leaf,
        sources=uni.sources.institution_urls,
        additional_instruction="If no URL is provided, return Incorrect."
    )

    # Enrollment data
    enroll_node = evaluator.add_parallel(
        id=f"Enrollment_Data_U{index+1}",
        desc=f"Current enrollment information for University {index+1}",
        parent=basic_node,
        critical=True,
    )

    enroll_num_leaf = evaluator.add_leaf(
        id=f"Enrollment_Number_U{index+1}",
        desc=f"Provides current undergraduate or total enrollment figure for University {index+1}",
        parent=enroll_node,
        critical=True,
    )
    claim_enroll_num = (
        f"The current enrollment figure for {uni.name or 'the university'} is {uni.enrollment or '[missing]'}."
    )
    await evaluator.verify(
        claim=claim_enroll_num,
        node=enroll_num_leaf,
        sources=uni.sources.enrollment_urls,
        additional_instruction=(
            "PASS only if the page explicitly provides an enrollment figure (undergraduate or total). "
            "Reasonable rounding is acceptable. If no URL provided, return Incorrect."
        ),
    )

    enroll_url_leaf = evaluator.add_leaf(
        id=f"Enrollment_URL_U{index+1}",
        desc=f"Provides URL verifying University {index+1}'s enrollment data",
        parent=enroll_node,
        critical=True,
    )
    claim_enroll_url = f"The provided page(s) report enrollment for {uni.name or 'the university'}."
    await evaluator.verify(
        claim=claim_enroll_url,
        node=enroll_url_leaf,
        sources=uni.sources.enrollment_urls,
        additional_instruction="If no URL is provided, return Incorrect."
    )

    # ---------------------- Athletic Program Requirements --------------------- #
    athletic_node = evaluator.add_parallel(
        id=f"Athletic_Program_Requirements_U{index+1}",
        desc=f"NCAA and conference requirements for University {index+1}",
        parent=uni_node,
        critical=True,
    )

    # NCAA Division Status
    ncaa_node = evaluator.add_parallel(
        id=f"NCAA_Division_Status_U{index+1}",
        desc=f"NCAA classification for University {index+1}",
        parent=athletic_node,
        critical=True,
    )

    ncaa_fbs_leaf = evaluator.add_leaf(
        id=f"Division_I_FBS_U{index+1}",
        desc=f"University {index+1} is classified as NCAA Division I FBS",
        parent=ncaa_node,
        critical=True,
    )
    claim_fbs = f"{uni.name or 'The university'} competes in NCAA Division I FBS."
    await evaluator.verify(
        claim=claim_fbs,
        node=ncaa_fbs_leaf,
        sources=uni.sources.ncaa_status_urls,
        additional_instruction="PASS only if the provided page(s) explicitly indicate FBS status. If no URL provided, return Incorrect."
    )

    ncaa_url_leaf = evaluator.add_leaf(
        id=f"NCAA_Status_URL_U{index+1}",
        desc=f"Provides URL verifying University {index+1}'s NCAA Division I FBS status",
        parent=ncaa_node,
        critical=True,
    )
    claim_ncaa_url = f"The provided page(s) confirm NCAA Division I FBS status for {uni.name or 'the university'}."
    await evaluator.verify(
        claim=claim_ncaa_url,
        node=ncaa_url_leaf,
        sources=uni.sources.ncaa_status_urls,
        additional_instruction="If no URL is provided, return Incorrect."
    )

    # Conference Affiliation
    conf_node = evaluator.add_parallel(
        id=f"Conference_Affiliation_U{index+1}",
        desc=f"Athletic conference membership for University {index+1}",
        parent=athletic_node,
        critical=True,
    )

    conf_name_leaf = evaluator.add_leaf(
        id=f"Conference_Name_U{index+1}",
        desc=f"Identifies University {index+1}'s current FBS athletic conference",
        parent=conf_node,
        critical=True,
    )
    conf_val = uni.conference or "[missing conference]"
    claim_conf = f"{uni.name or 'The university'} is a member of the {conf_val} conference."
    await evaluator.verify(
        claim=claim_conf,
        node=conf_name_leaf,
        sources=uni.sources.conference_urls,
        additional_instruction="PASS only if the page(s) explicitly confirm current conference membership. If no URL provided, return Incorrect."
    )

    conf_url_leaf = evaluator.add_leaf(
        id=f"Conference_URL_U{index+1}",
        desc=f"Provides URL verifying University {index+1}'s conference membership",
        parent=conf_node,
        critical=True,
    )
    claim_conf_url = f"The provided page(s) confirm conference membership for {uni.name or 'the university'}."
    await evaluator.verify(
        claim=claim_conf_url,
        node=conf_url_leaf,
        sources=uni.sources.conference_urls,
        additional_instruction="If no URL is provided, return Incorrect."
    )

    # Sports program count
    sports_node = evaluator.add_parallel(
        id=f"Sports_Program_Count_U{index+1}",
        desc=f"Varsity sports portfolio for University {index+1}",
        parent=athletic_node,
        critical=True,
    )

    min_16_leaf = evaluator.add_leaf(
        id=f"Minimum_Sixteen_Sports_U{index+1}",
        desc=f"University {index+1} sponsors at least 16 varsity sports (Division I minimum requirement)",
        parent=sports_node,
        critical=True,
    )
    claim_min16 = f"{uni.name or 'The university'} sponsors at least 16 varsity sports."
    await evaluator.verify(
        claim=claim_min16,
        node=min_16_leaf,
        sources=uni.sources.sports_count_urls,
        additional_instruction=(
            "Count only recognized varsity sports (men's/women's). If the page lists a count >= 16 or explicitly states 'at least 16', PASS. "
            "If no URL provided, return Incorrect."
        ),
    )

    sports_url_leaf = evaluator.add_leaf(
        id=f"Sports_Count_URL_U{index+1}",
        desc=f"Provides URL verifying University {index+1}'s sports program count",
        parent=sports_node,
        critical=True,
    )
    claim_sports_url = f"The provided page(s) enumerate or state the varsity sports count for {uni.name or 'the university'}."
    await evaluator.verify(
        claim=claim_sports_url,
        node=sports_url_leaf,
        sources=uni.sources.sports_count_urls,
        additional_instruction="If no URL is provided, return Incorrect."
    )

    # ---------------------- Football Program Details -------------------------- #
    football_node = evaluator.add_parallel(
        id=f"Football_Program_Details_U{index+1}",
        desc=f"Football program specifics for University {index+1}",
        parent=uni_node,
        critical=True,
    )

    coach_node = evaluator.add_parallel(
        id=f"Head_Coach_Information_U{index+1}",
        desc=f"Current football coaching staff for University {index+1}",
        parent=football_node,
        critical=True,
    )

    coach_leaf = evaluator.add_leaf(
        id=f"Active_Head_Coach_U{index+1}",
        desc=f"Identifies University {index+1}'s current football head coach by name",
        parent=coach_node,
        critical=True,
    )
    coach_val = uni.head_coach or "[missing coach]"
    claim_coach = f"The current football head coach for {uni.name or 'the university'} is {coach_val}."
    await evaluator.verify(
        claim=claim_coach,
        node=coach_leaf,
        sources=uni.sources.head_coach_urls,
        additional_instruction="PASS only if the page(s) clearly indicate the current head coach. If no URL provided, return Incorrect."
    )

    coach_url_leaf = evaluator.add_leaf(
        id=f"Head_Coach_URL_U{index+1}",
        desc=f"Provides URL verifying University {index+1}'s current head coach",
        parent=coach_node,
        critical=True,
    )
    claim_coach_url = f"The provided page(s) confirm the current head coach for {uni.name or 'the university'}."
    await evaluator.verify(
        claim=claim_coach_url,
        node=coach_url_leaf,
        sources=uni.sources.head_coach_urls,
        additional_instruction="If no URL is provided, return Incorrect."
    )

    stadium_node = evaluator.add_parallel(
        id=f"Stadium_Specifications_U{index+1}",
        desc=f"Football stadium information for University {index+1}",
        parent=football_node,
        critical=True,
    )

    capacity_leaf = evaluator.add_leaf(
        id=f"Stadium_Capacity_U{index+1}",
        desc=f"Provides University {index+1}'s football stadium seating capacity",
        parent=stadium_node,
        critical=True,
    )
    cap_val = uni.stadium_capacity or "[missing capacity]"
    claim_capacity = f"The football stadium seating capacity for {uni.name or 'the university'} is {cap_val}."
    await evaluator.verify(
        claim=claim_capacity,
        node=capacity_leaf,
        sources=uni.sources.stadium_urls,
        additional_instruction=(
            "PASS if the page shows a capacity number that matches or is reasonably equivalent (rounding acceptable). "
            "If no URL provided, return Incorrect."
        ),
    )

    stadium_url_leaf = evaluator.add_leaf(
        id=f"Stadium_URL_U{index+1}",
        desc=f"Provides URL verifying University {index+1}'s stadium capacity",
        parent=stadium_node,
        critical=True,
    )
    claim_stadium_url = f"The provided page(s) explicitly state the stadium seating capacity for {uni.name or 'the university'}."
    await evaluator.verify(
        claim=claim_stadium_url,
        node=stadium_url_leaf,
        sources=uni.sources.stadium_urls,
        additional_instruction="If no URL is provided, return Incorrect."
    )

    # ---------------------- Academic Profile ---------------------------------- #
    academic_node = evaluator.add_parallel(
        id=f"Academic_Profile_U{index+1}",
        desc=f"Academic standards and requirements for University {index+1}",
        parent=uni_node,
        critical=True,
    )

    admit_node = evaluator.add_parallel(
        id=f"Admission_Requirements_U{index+1}",
        desc=f"Published admission standards for University {index+1}",
        parent=academic_node,
        critical=True,
    )

    req_leaf = evaluator.add_leaf(
        id=f"GPA_or_Test_Requirements_U{index+1}",
        desc=f"Provides University {index+1}'s admission requirements (GPA, test scores, or test-optional policy)",
        parent=admit_node,
        critical=True,
    )
    req_text = uni.admissions_requirements or "[missing requirements]"
    claim_requirements = (
        f"The admissions page for {uni.name or 'the university'} describes current requirements, including GPA/test scores or a test-optional policy: {req_text}"
    )
    await evaluator.verify(
        claim=claim_requirements,
        node=req_leaf,
        sources=uni.sources.admissions_urls,
        additional_instruction=(
            "PASS only if the page provides current admissions requirements or explicitly states test-optional/test-required policies. "
            "If no URL provided, return Incorrect."
        ),
    )

    admit_url_leaf = evaluator.add_leaf(
        id=f"Admissions_URL_U{index+1}",
        desc=f"Provides URL verifying University {index+1}'s admission requirements",
        parent=admit_node,
        critical=True,
    )
    claim_admit_url = f"The provided page(s) present admissions requirements/policies for {uni.name or 'the university'}."
    await evaluator.verify(
        claim=claim_admit_url,
        node=admit_url_leaf,
        sources=uni.sources.admissions_urls,
        additional_instruction="If no URL is provided, return Incorrect."
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
    model: str = "o4-mini",
) -> Dict[str, Any]:
    # Initialize evaluator
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

    # Extract universities block
    extracted = await evaluator.extract(
        prompt=prompt_extract_universities(),
        template_class=UniversitiesExtraction,
        extraction_name="universities_extraction",
    )

    # Keep only the first 3 universities; pad if fewer
    unis: List[UniversityInfo] = list(extracted.universities[:3])
    while len(unis) < 3:
        unis.append(UniversityInfo())

    # Record helper info
    evaluator.add_custom_info(
        {"allowed_western_states": sorted(list(WESTERN_STATES))},
        info_type="constraints",
        info_name="western_states_constraint"
    )

    # Build the verification tree per university
    prev_states: List[Optional[str]] = []
    for i in range(3):
        uni = unis[i]
        # Store normalized states to use for different-state constraints
        prev_states.append(uni.state)
        await verify_university(
            evaluator=evaluator,
            parent_node=root,
            uni=uni,
            index=i,
            prev_states=prev_states
        )

    # Return evaluation summary
    return evaluator.get_summary()