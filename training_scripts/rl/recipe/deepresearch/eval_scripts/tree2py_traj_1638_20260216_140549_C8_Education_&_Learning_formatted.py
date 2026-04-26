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
TASK_ID = "tx_public_universities_mfg_eng"
TASK_DESCRIPTION = """
A community college student in Texas is completing an Associate of Science degree with 35 transferable credit hours and a 3.0 GPA. The student plans to transfer to a four-year public university in Texas for Spring 2026 to pursue a Bachelor of Science in Manufacturing Engineering. The student is particularly interested in gaining practical work experience through a cooperative education or internship program while completing their degree.

Identify TWO different Texas public universities that meet ALL of the following requirements:

1. The institution must be a public university located in Texas
2. The university must offer an ABET-accredited Bachelor of Science degree specifically in Manufacturing Engineering (not Manufacturing Engineering Technology)
3. The university must accept transfer students for the Spring 2026 semester  
4. The university must have documented cooperative education or internship opportunities available for engineering students

For each of the two universities you identify, provide:
- The university name and main campus location (city or cities)
- Confirmation that the Manufacturing Engineering B.S. program is ABET-accredited
- Any available specializations or concentrations within the Manufacturing Engineering program
- The Spring 2026 transfer application deadline (priority or final deadline)
- The minimum cumulative GPA requirement for transfer students with 30 or more credit hours
- The minimum GPA requirement for participation in the cooperative education or internship program (if specified)
- The total credit hours required to complete the Manufacturing Engineering degree
- The minimum number of advanced-level (3000 or 4000 level) credit hours required
- Valid reference URLs supporting each piece of information

Provide complete, verified information for both universities, ensuring each meets all specified criteria.
""".strip()


# --------------------------------------------------------------------------- #
# Data models for structured extraction                                       #
# --------------------------------------------------------------------------- #
class UniversityItem(BaseModel):
    # Institutional info
    university_name: Optional[str] = None
    campus_location: Optional[str] = None
    institution_urls: List[str] = Field(default_factory=list)

    # Program info
    degree_title: Optional[str] = None  # e.g., "Bachelor of Science in Manufacturing Engineering"
    program_specializations: List[str] = Field(default_factory=list)
    program_urls: List[str] = Field(default_factory=list)  # Should include ABET/program accreditation/details

    # Transfer admissions info
    spring_2026_acceptance: Optional[bool] = None  # If stated in the answer
    application_deadline: Optional[str] = None  # Spring 2026 deadline (priority/final)
    transfer_gpa_30_plus: Optional[str] = None  # Min cumulative GPA for 30+ credits
    transfer_urls: List[str] = Field(default_factory=list)

    # Co-op/Internship info
    coop_internship_program: Optional[bool] = None
    coop_gpa_requirement: Optional[str] = None  # If specified
    coop_urls: List[str] = Field(default_factory=list)

    # Degree requirements
    total_credit_hours: Optional[str] = None
    advanced_credit_hours: Optional[str] = None  # Min 3000/4000-level hours
    degree_requirements_urls: List[str] = Field(default_factory=list)


class UniversitiesExtraction(BaseModel):
    universities: List[UniversityItem] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_universities() -> str:
    return """
    Extract details for up to two Texas public universities mentioned in the answer that offer ABET-accredited Bachelor of Science in Manufacturing Engineering (not Manufacturing Engineering Technology), accept transfer students for Spring 2026, and provide cooperative education/internship opportunities.

    For each university mentioned in the answer (in the same order as presented), extract the following fields (use null or empty list when the answer does not specify; do not infer):
    - university_name: The university name
    - campus_location: The main campus city or cities as stated
    - institution_urls: All URLs that support the institution type/location (official site pages, system pages, Wikipedia pages if cited in the answer)
    - degree_title: The degree title as stated in the answer (e.g., "Bachelor of Science in Manufacturing Engineering")
    - program_specializations: A list of any specializations/concentrations within Manufacturing Engineering if provided
    - program_urls: All URLs supporting ABET accreditation and program details (ABET program list page or official program page)
    - spring_2026_acceptance: true/false only if explicitly indicated in the answer; otherwise null
    - application_deadline: The Spring 2026 transfer application deadline (priority or final) exactly as stated
    - transfer_gpa_30_plus: The minimum cumulative GPA required for applicants with 30 or more credits
    - transfer_urls: All URLs supporting transfer acceptance, deadlines, and GPA requirements (e.g., admissions pages)
    - coop_internship_program: true/false only if presence is explicitly stated in the answer; otherwise null
    - coop_gpa_requirement: The minimum GPA for co-op/internship participation if specified
    - coop_urls: All URLs supporting co-op/internship information for engineering students
    - total_credit_hours: The total credit hours required to complete the BS in Manufacturing Engineering
    - advanced_credit_hours: The minimum 3000/4000-level credit hours required if specified
    - degree_requirements_urls: All URLs supporting degree requirements (catalog, curriculum plans, etc.)

    Return a JSON object with a "universities" array containing up to two university objects with these fields.
    Ensure all URL fields include full absolute URLs. Do not fabricate any information that is not present in the answer.
    """.strip()


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _filter_valid_urls(urls: List[str]) -> List[str]:
    valid = []
    for u in urls or []:
        if isinstance(u, str):
            s = u.strip()
            if s.startswith("http://") or s.startswith("https://"):
                valid.append(s)
    # deduplicate while preserving order
    seen = set()
    uniq = []
    for u in valid:
        if u not in seen:
            seen.add(u)
            uniq.append(u)
    return uniq


async def _verify_with_urls_or_fail(
    evaluator: Evaluator,
    claim: str,
    node,
    urls: List[str],
    additional_instruction: str = "None",
) -> bool:
    valid_urls = _filter_valid_urls(urls)
    if not valid_urls:
        node.score = 0.0
        node.status = "failed"
        return False
    return await evaluator.verify(
        claim=claim,
        node=node,
        sources=valid_urls,
        additional_instruction=additional_instruction,
    )


async def _verify_optional_claim(
    evaluator: Evaluator,
    claim: str,
    node,
    urls: List[str],
    present: bool,
    additional_instruction: str = "None",
) -> bool:
    if not present:
        node.score = 0.0
        node.status = "skipped"
        return False
    return await _verify_with_urls_or_fail(evaluator, claim, node, urls, additional_instruction)


# --------------------------------------------------------------------------- #
# Verification logic for a single university                                  #
# --------------------------------------------------------------------------- #
async def verify_university(
    evaluator: Evaluator,
    parent_node,
    uni: UniversityItem,
    uni_index: int,
) -> None:
    """
    Build verification sub-tree for a single university and perform checks.
    """
    uni_node = evaluator.add_parallel(
        id=f"University_{uni_index}",
        desc="First qualifying Texas public university" if uni_index == 1 else "Second qualifying Texas public university",
        parent=parent_node,
        critical=False,  # allow partial credit for each university independently
    )

    # 1) Institution_Name (Critical): existence of name
    evaluator.add_custom_node(
        result=bool(uni.university_name and uni.university_name.strip()),
        id=f"u{uni_index}_Institution_Name",
        desc="University name is correctly identified",
        parent=uni_node,
        critical=True
    )

    # 2) Public_Institution_Texas (Critical): verify public + in Texas via institution URLs
    node_public_tx = evaluator.add_leaf(
        id=f"u{uni_index}_Public_Institution_Texas",
        desc="Confirmed as a public university located in Texas",
        parent=uni_node,
        critical=True
    )
    claim_public_tx = f"{uni.university_name or 'The university'} is a public university located in Texas."
    await _verify_with_urls_or_fail(
        evaluator,
        claim_public_tx,
        node_public_tx,
        uni.institution_urls,
        additional_instruction="Confirm the institution is public and located in Texas. Accept phrasings like 'public university', 'public research university', or 'state university'. Location must be in Texas."
    )

    # 3) Campus_Location (Critical): verify main campus location via institution URLs
    node_location = evaluator.add_leaf(
        id=f"u{uni_index}_Campus_Location",
        desc="Main campus location (city) is specified",
        parent=uni_node,
        critical=True
    )
    claim_location = f"The main campus location is {uni.campus_location}."
    await _verify_with_urls_or_fail(
        evaluator,
        claim_location,
        node_location,
        uni.institution_urls,
        additional_instruction="Verify the primary campus location (city or cities) as stated."
    )

    # 4) Institution_Reference (Critical): presence of institutional references
    evaluator.add_custom_node(
        result=len(_filter_valid_urls(uni.institution_urls)) > 0,
        id=f"u{uni_index}_Institution_Reference",
        desc="Valid reference URL for institutional information provided",
        parent=uni_node,
        critical=True
    )

    # 5) ABET_Accreditation (Critical): verify ABET accreditation via program URLs
    node_abet = evaluator.add_leaf(
        id=f"u{uni_index}_ABET_Accreditation",
        desc="Manufacturing Engineering B.S. program is confirmed ABET accredited",
        parent=uni_node,
        critical=True
    )
    claim_abet = "The Bachelor of Science in Manufacturing Engineering program is ABET-accredited."
    await _verify_with_urls_or_fail(
        evaluator,
        claim_abet,
        node_abet,
        uni.program_urls,
        additional_instruction="Look for ABET accreditation specific to Manufacturing Engineering, preferably 'accredited by the Engineering Accreditation Commission (EAC) of ABET' or an ABET listing page."
    )

    # 6) Degree_Type (Critical): confirm BS in Manufacturing Engineering (not Technology)
    node_degree_type = evaluator.add_leaf(
        id=f"u{uni_index}_Degree_Type",
        desc="Confirmed as Bachelor of Science in Manufacturing Engineering (not Manufacturing Engineering Technology)",
        parent=uni_node,
        critical=True
    )
    claim_degree_type = "The degree offered is a Bachelor of Science in Manufacturing Engineering, not Manufacturing Engineering Technology."
    await _verify_with_urls_or_fail(
        evaluator,
        claim_degree_type,
        node_degree_type,
        uni.program_urls,
        additional_instruction="Verify the program name includes 'Bachelor of Science' in 'Manufacturing Engineering'. Reject if it is 'Manufacturing Engineering Technology' or similar technology-only programs."
    )

    # 7) Program_Specializations (Non-critical): verify if provided; otherwise skip
    node_specs = evaluator.add_leaf(
        id=f"u{uni_index}_Program_Specializations",
        desc="Available specializations or concentrations are documented (if applicable)",
        parent=uni_node,
        critical=False
    )
    if uni.program_specializations and len(uni.program_specializations) > 0:
        claim_specs = f"The program offers specializations or concentrations such as: {', '.join(uni.program_specializations)}."
        await _verify_optional_claim(
            evaluator,
            claim_specs,
            node_specs,
            uni.program_urls,
            present=True,
            additional_instruction="Confirm that the named specializations/concentrations are listed for the BS in Manufacturing Engineering program."
        )
    else:
        node_specs.score = 0.0
        node_specs.status = "skipped"

    # 8) Program_Reference (Critical): presence of program refs
    evaluator.add_custom_node(
        result=len(_filter_valid_urls(uni.program_urls)) > 0,
        id=f"u{uni_index}_Program_Reference",
        desc="Valid reference URL for ABET accreditation and program details provided",
        parent=uni_node,
        critical=True
    )

    # 9) Spring_2026_Acceptance (Critical): verify acceptance for Spring 2026 via transfer URLs
    node_spring_accept = evaluator.add_leaf(
        id=f"u{uni_index}_Spring_2026_Acceptance",
        desc="University accepts transfer students for Spring 2026 semester",
        parent=uni_node,
        critical=True
    )
    claim_accept = "The university accepts transfer student applications for the Spring 2026 term."
    await _verify_with_urls_or_fail(
        evaluator,
        claim_accept,
        node_spring_accept,
        uni.transfer_urls,
        additional_instruction="Confirm that transfer applications are accepted for Spring 2026 (deadlines or term availability pages)."
    )

    # 10) Application_Deadline (Critical): verify the Spring 2026 transfer application deadline
    node_deadline = evaluator.add_leaf(
        id=f"u{uni_index}_Application_Deadline",
        desc="Spring 2026 transfer application deadline is specified",
        parent=uni_node,
        critical=True
    )
    claim_deadline = f"The Spring 2026 transfer application deadline is {uni.application_deadline}."
    await _verify_with_urls_or_fail(
        evaluator,
        claim_deadline,
        node_deadline,
        uni.transfer_urls,
        additional_instruction="Verify the stated deadline (priority or final) for Spring 2026 transfer applications."
    )

    # 11) Transfer_GPA_30_Plus (Critical): verify minimum cumulative GPA for 30+ credits
    node_tx_gpa = evaluator.add_leaf(
        id=f"u{uni_index}_Transfer_GPA_30_Plus",
        desc="Minimum cumulative GPA requirement for transfer students with 30+ credit hours is stated",
        parent=uni_node,
        critical=True
    )
    claim_tx_gpa = f"The minimum cumulative GPA requirement for transfer students with 30 or more credit hours is {uni.transfer_gpa_30_plus}."
    await _verify_with_urls_or_fail(
        evaluator,
        claim_tx_gpa,
        node_tx_gpa,
        uni.transfer_urls,
        additional_instruction="Confirm the minimum cumulative GPA threshold that applies to applicants with approximately 30 or more transferable credit hours."
    )

    # 12) Transfer_Reference (Critical): presence of transfer refs
    evaluator.add_custom_node(
        result=len(_filter_valid_urls(uni.transfer_urls)) > 0,
        id=f"u{uni_index}_Transfer_Reference",
        desc="Valid reference URL for transfer admission requirements provided",
        parent=uni_node,
        critical=True
    )

    # 13) Coop_Internship_Program (Critical): verify existence for engineering students
    node_coop = evaluator.add_leaf(
        id=f"u{uni_index}_Coop_Internship_Program",
        desc="Cooperative education or internship program for engineering students is documented",
        parent=uni_node,
        critical=True
    )
    claim_coop = "There is a cooperative education and/or internship program available for engineering students."
    await _verify_with_urls_or_fail(
        evaluator,
        claim_coop,
        node_coop,
        uni.coop_urls,
        additional_instruction="Confirm a co-op or internship program specifically available to engineering students (e.g., engineering co-op office, engineering internships page, or college of engineering career services)."
    )

    # 14) Coop_GPA_Requirement (Non-critical): verify if provided; otherwise skip
    node_coop_gpa = evaluator.add_leaf(
        id=f"u{uni_index}_Coop_GPA_Requirement",
        desc="Minimum GPA requirement for co-op/internship participation is specified (if available)",
        parent=uni_node,
        critical=False
    )
    if uni.coop_gpa_requirement and uni.coop_gpa_requirement.strip():
        claim_coop_gpa = f"The minimum GPA requirement for participation in co-op/internship is {uni.coop_gpa_requirement}."
        await _verify_optional_claim(
            evaluator,
            claim_coop_gpa,
            node_coop_gpa,
            uni.coop_urls,
            present=True,
            additional_instruction="Verify the minimum GPA threshold for eligibility in the engineering co-op or internship program, if the page states one."
        )
    else:
        node_coop_gpa.score = 0.0
        node_coop_gpa.status = "skipped"

    # 15) Coop_Reference (Critical): presence of coop refs
    evaluator.add_custom_node(
        result=len(_filter_valid_urls(uni.coop_urls)) > 0,
        id=f"u{uni_index}_Coop_Reference",
        desc="Valid reference URL for co-op/internship program information provided",
        parent=uni_node,
        critical=True
    )

    # 16) Total_Credit_Hours (Non-critical): verify if provided; otherwise skip
    node_total_hours = evaluator.add_leaf(
        id=f"u{uni_index}_Total_Credit_Hours",
        desc="Total credit hours required for Manufacturing Engineering degree completion is stated",
        parent=uni_node,
        critical=False
    )
    if uni.total_credit_hours and uni.total_credit_hours.strip():
        claim_total_hours = f"The total credit hours required to complete the Manufacturing Engineering degree is {uni.total_credit_hours}."
        await _verify_optional_claim(
            evaluator,
            claim_total_hours,
            node_total_hours,
            uni.degree_requirements_urls,
            present=True,
            additional_instruction="Confirm the total number of credit hours for the BS in Manufacturing Engineering in the official catalog or curriculum page."
        )
    else:
        node_total_hours.score = 0.0
        node_total_hours.status = "skipped"

    # 17) Advanced_Credit_Hours (Non-critical): verify if provided; otherwise skip
    node_adv_hours = evaluator.add_leaf(
        id=f"u{uni_index}_Advanced_Credit_Hours",
        desc="Minimum advanced (3000/4000 level) credit hours required is specified (if applicable)",
        parent=uni_node,
        critical=False
    )
    if uni.advanced_credit_hours and uni.advanced_credit_hours.strip():
        claim_adv_hours = f"The minimum number of advanced-level (3000/4000 level) credit hours required is {uni.advanced_credit_hours}."
        await _verify_optional_claim(
            evaluator,
            claim_adv_hours,
            node_adv_hours,
            uni.degree_requirements_urls,
            present=True,
            additional_instruction="Verify advanced-level (upper-division) credit hour requirements applicable to the BS in Manufacturing Engineering (e.g., 3000/4000-level credits)."
        )
    else:
        node_adv_hours.score = 0.0
        node_adv_hours.status = "skipped"

    # 18) Degree_Requirements_Reference (Critical): presence of degree requirement refs
    evaluator.add_custom_node(
        result=len(_filter_valid_urls(uni.degree_requirements_urls)) > 0,
        id=f"u{uni_index}_Degree_Requirements_Reference",
        desc="Valid reference URL for degree requirements provided",
        parent=uni_node,
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
    """
    Evaluate an answer for identifying two Texas public universities with ABET-accredited BS in Manufacturing Engineering,
    Spring 2026 transfer acceptance, and co-op/internship opportunities for engineering students.
    """
    # Initialize evaluator with a parallel root
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

    # Add top-level grouping node to mirror rubric
    top_node = evaluator.add_parallel(
        id="Texas_Public_Universities_Manufacturing_Engineering",
        desc="Identify two Texas public universities offering ABET-accredited Manufacturing Engineering programs that accept Spring 2026 transfer students and provide cooperative education or internship opportunities",
        parent=root,
        critical=False
    )

    # Extract structured university info from the answer
    extraction = await evaluator.extract(
        prompt=prompt_extract_universities(),
        template_class=UniversitiesExtraction,
        extraction_name="universities_extraction"
    )

    # Normalize to exactly two universities: take first two; pad with empty items if fewer
    universities: List[UniversityItem] = extraction.universities[:2]
    while len(universities) < 2:
        universities.append(UniversityItem())

    # Build verification subtrees for two universities
    await verify_university(evaluator, top_node, universities[0], 1)
    await verify_university(evaluator, top_node, universities[1], 2)

    # Return evaluation summary
    return evaluator.get_summary()