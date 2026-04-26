import asyncio
import logging
from typing import Any, List, Optional, Dict

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy

# --------------------------------------------------------------------------- #
# Task constants                                                              #
# --------------------------------------------------------------------------- #
TASK_ID = "pa_elem_ed_program"
TASK_DESCRIPTION = (
    "Identify a Pennsylvania university that offers an undergraduate elementary education program meeting all "
    "specified institutional, accreditation, degree, core academic, clinical experience, GPA, and certification "
    "requirements. Provide the university name, specific program name, and reference URLs confirming each major "
    "requirement category."
)

# --------------------------------------------------------------------------- #
# Data Models                                                                 #
# --------------------------------------------------------------------------- #
class ProgramExtraction(BaseModel):
    # Names
    university_name: Optional[str] = None
    program_name: Optional[str] = None

    # University basics
    location_urls: List[str] = Field(default_factory=list)
    accreditation_urls: List[str] = Field(default_factory=list)

    # Program accreditation
    caep_urls: List[str] = Field(default_factory=list)
    state_approval_urls: List[str] = Field(default_factory=list)

    # Degree + credits
    degree_urls: List[str] = Field(default_factory=list)
    credit_hours_value: Optional[str] = None
    credit_hours_urls: List[str] = Field(default_factory=list)

    # Certification grade levels
    certification_grade_levels: Optional[str] = None
    grade_levels_urls: List[str] = Field(default_factory=list)

    # Core academic requirements (12 areas)
    english_urls: List[str] = Field(default_factory=list)
    math_urls: List[str] = Field(default_factory=list)
    science_urls: List[str] = Field(default_factory=list)
    social_studies_urls: List[str] = Field(default_factory=list)
    arts_humanities_urls: List[str] = Field(default_factory=list)
    child_dev_urls: List[str] = Field(default_factory=list)
    special_ed_urls: List[str] = Field(default_factory=list)
    esl_urls: List[str] = Field(default_factory=list)
    literacy_urls: List[str] = Field(default_factory=list)
    classroom_mgmt_urls: List[str] = Field(default_factory=list)
    assessment_urls: List[str] = Field(default_factory=list)
    edtech_urls: List[str] = Field(default_factory=list)

    # Clinical experience
    early_field_urls: List[str] = Field(default_factory=list)
    student_teaching_duration_urls: List[str] = Field(default_factory=list)
    student_teaching_prereq_urls: List[str] = Field(default_factory=list)

    # GPA standards
    admission_gpa_value: Optional[str] = None
    admission_gpa_urls: List[str] = Field(default_factory=list)
    continuation_gpa_value: Optional[str] = None
    continuation_gpa_urls: List[str] = Field(default_factory=list)

    # Certification preparation
    instructional_I_urls: List[str] = Field(default_factory=list)
    exams_urls: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction Prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_program() -> str:
    return """
    Extract structured information for a Pennsylvania university's undergraduate elementary education program from the provided answer.
    Return fields strictly from the answer text, especially URLs. If a field is missing, return null (for scalar) or an empty list (for URLs).
    
    Required fields:
    - university_name: The university's name
    - program_name: The specific program name (e.g., "BS in Elementary Education")
    
    University basic info URLs:
    - location_urls: URLs confirming the university is located in Pennsylvania (can include the university's about/contact/campus page showing PA address)
    - accreditation_urls: URLs confirming the university holds regional or national accreditation (e.g., MSCHE, HLC)
    
    Program accreditation URLs:
    - caep_urls: URLs confirming the program holds CAEP accreditation (CAEP directory or program page that states CAEP)
    - state_approval_urls: URLs confirming Pennsylvania state approval as an educator preparation program (PDE listing or explicit program page)
    
    Degree & credits:
    - degree_urls: URLs confirming the awarded Bachelor degree type in Education (BS/BA, B.S.Ed./B.A.Ed.)
    - credit_hours_value: The stated number of required semester credits (e.g., "120", "124")
    - credit_hours_urls: URLs specifying the credit hour requirement
    
    Certification grade levels:
    - certification_grade_levels: Stated grade levels for PA elementary certification (e.g., "K-6", "PK-4", "4-8")
    - grade_levels_urls: URLs confirming certification grade levels
    
    Core academic requirement URLs (program must include coursework in all 12 areas):
    - english_urls
    - math_urls
    - science_urls
    - social_studies_urls
    - arts_humanities_urls
    - child_dev_urls  (child development or educational psychology)
    - special_ed_urls (special education or inclusive education)
    - esl_urls        (ESL/Multilingual Learner instruction)
    - literacy_urls   (reading & literacy instruction methods)
    - classroom_mgmt_urls (classroom management)
    - assessment_urls     (student assessment & evaluation methods)
    - edtech_urls         (educational technology integration)
    Each should include one or more URLs that clearly show curriculum/coursework meeting the area.
    
    Clinical experience URLs:
    - early_field_urls: URLs confirming early field observations in K-12 settings before student teaching
    - student_teaching_duration_urls: URLs confirming full-time student teaching lasting at least one semester in an accredited K-12 school
    - student_teaching_prereq_urls: URLs confirming minimum 60 credits required prior to enrolling in student teaching
    
    GPA standards URLs:
    - admission_gpa_value: Stated minimum GPA for admission to the education program (should be "2.8" if present)
    - admission_gpa_urls: URLs confirming admission GPA requirement
    - continuation_gpa_value: Stated minimum GPA to remain in the program
    - continuation_gpa_urls: URLs confirming continuation GPA requirement
    
    Certification preparation URLs:
    - instructional_I_urls: URLs confirming program completion leads to eligibility for Pennsylvania Instructional I Certificate
    - exams_urls: URLs detailing preparation for required Pennsylvania certification examinations (Praxis/PECT)
    
    Only include URLs explicitly present in the answer. If none are given for a field, return an empty list for that field.
    """


# --------------------------------------------------------------------------- #
# Helper: add "URL present" gating + verify leaf                              #
# --------------------------------------------------------------------------- #
async def add_url_supported_leaf(
    evaluator: Evaluator,
    parent_node,
    existence_id: str,
    existence_desc: str,
    urls: List[str],
    verify_id: str,
    verify_desc: str,
    claim: str,
    additional_instruction: str,
    critical: bool = True,
) -> None:
    # Existence (at least one URL provided) – gate verification
    evaluator.add_custom_node(
        result=bool(urls and len(urls) > 0),
        id=existence_id,
        desc=existence_desc,
        parent=parent_node,
        critical=critical,
    )

    # Verification leaf supported by URL(s)
    leaf = evaluator.add_leaf(
        id=verify_id,
        desc=verify_desc,
        parent=parent_node,
        critical=critical,
    )
    await evaluator.verify(
        claim=claim,
        node=leaf,
        sources=urls if urls else None,  # If empty, simple_verify will be used; existence node gates this
        additional_instruction=additional_instruction,
    )


# --------------------------------------------------------------------------- #
# Verification builders                                                       #
# --------------------------------------------------------------------------- #
async def build_university_basic_info(evaluator: Evaluator, parent_node, ex: ProgramExtraction) -> None:
    uni_node = evaluator.add_parallel(
        id="University_Basic_Information",
        desc="Basic institutional information and location",
        parent=parent_node,
        critical=True,
    )

    # Pennsylvania location
    loc_node = evaluator.add_parallel(
        id="Pennsylvania_Location",
        desc="University is located in Pennsylvania",
        parent=uni_node,
        critical=True,
    )
    await add_url_supported_leaf(
        evaluator,
        loc_node,
        existence_id="Pennsylvania_Location_URLs_Present",
        existence_desc="Provide URL(s) confirming Pennsylvania location: URLs present",
        urls=ex.location_urls,
        verify_id="URL_Reference_Location",
        verify_desc="Provide URL confirming Pennsylvania location (source-supported)",
        claim=f"The university '{ex.university_name or 'the identified university'}' is located in Pennsylvania.",
        additional_instruction="Accept if the referenced page shows a Pennsylvania address, 'PA' postal abbreviation, or explicitly states the university is in Pennsylvania.",
        critical=True,
    )

    # Accreditation status (regional/national)
    acc_node = evaluator.add_parallel(
        id="Accreditation_Status",
        desc="University holds regional or national accreditation",
        parent=uni_node,
        critical=True,
    )
    await add_url_supported_leaf(
        evaluator,
        acc_node,
        existence_id="Accreditation_Status_URLs_Present",
        existence_desc="Provide URL(s) confirming accreditation status: URLs present",
        urls=ex.accreditation_urls,
        verify_id="URL_Reference_Accreditation",
        verify_desc="Provide URL confirming accreditation status (source-supported)",
        claim="The university holds recognized regional or national accreditation.",
        additional_instruction="Accept if the page lists recognized accreditors (e.g., Middle States Commission on Higher Education (MSCHE), Higher Learning Commission, SACSCOC).",
        critical=True,
    )


async def build_program_accreditation(evaluator: Evaluator, parent_node, ex: ProgramExtraction) -> None:
    acc_req = evaluator.add_parallel(
        id="Program_Accreditation_Requirements",
        desc="Education program holds required specialized accreditation",
        parent=parent_node,
        critical=True,
    )

    # CAEP accreditation
    caep = evaluator.add_parallel(
        id="CAEP_Accreditation",
        desc="Program holds CAEP (Council for the Accreditation of Educator Preparation) accreditation",
        parent=acc_req,
        critical=True,
    )
    await add_url_supported_leaf(
        evaluator,
        caep,
        existence_id="CAEP_URLs_Present",
        existence_desc="Provide URL(s) confirming CAEP accreditation: URLs present",
        urls=ex.caep_urls,
        verify_id="URL_Reference_CAEP",
        verify_desc="Provide URL confirming CAEP accreditation (source-supported)",
        claim="The undergraduate elementary education program holds CAEP accreditation.",
        additional_instruction="Accept if the CAEP directory lists the institution/program or the program page explicitly states CAEP accreditation.",
        critical=True,
    )

    # State approval (PDE)
    state = evaluator.add_parallel(
        id="State_Approval",
        desc="Program is Pennsylvania state-approved educator preparation program",
        parent=acc_req,
        critical=True,
    )
    await add_url_supported_leaf(
        evaluator,
        state,
        existence_id="State_Approval_URLs_Present",
        existence_desc="Provide URL(s) confirming Pennsylvania state approval: URLs present",
        urls=ex.state_approval_urls,
        verify_id="URL_Reference_State_Approval",
        verify_desc="Provide URL confirming state approval (source-supported)",
        claim="The program is a Pennsylvania state-approved educator preparation program.",
        additional_instruction="Prefer a Pennsylvania Department of Education (PDE) approved programs list or official program page explicitly stating state approval.",
        critical=True,
    )


async def build_degree_requirements(evaluator: Evaluator, parent_node, ex: ProgramExtraction) -> None:
    deg_root = evaluator.add_parallel(
        id="Degree_Requirements",
        desc="Program degree and credit requirements",
        parent=parent_node,
        critical=True,
    )

    # Bachelor Degree Requirement parent
    bachelor = evaluator.add_parallel(
        id="Bachelor_Degree_Requirement",
        desc="Program awards Bachelor of Science or Bachelor of Arts in Education",
        parent=deg_root,
        critical=True,
    )
    # Degree type verification (BS/BA in Education)
    await add_url_supported_leaf(
        evaluator,
        bachelor,
        existence_id="Degree_Type_URLs_Present",
        existence_desc="Provide URL(s) confirming awarded Bachelor degree in Education: URLs present",
        urls=ex.degree_urls,
        verify_id="Bachelor_Degree_Type_Verification",
        verify_desc="Verify program awards BS or BA in Education (source-supported)",
        claim=f"The program '{ex.program_name or 'the identified program'}' awards a Bachelor's degree in Education (Bachelor of Science or Bachelor of Arts).",
        additional_instruction="Accept either 'Bachelor of Science in Education', 'Bachelor of Arts in Education', 'B.S.Ed.', or 'B.A.Ed.' as satisfying this requirement.",
        critical=True,
    )

    # Minimum credit hours (>=120)
    min_credits = evaluator.add_parallel(
        id="Minimum_Credit_Hours",
        desc="Program requires minimum 120 semester credit hours",
        parent=bachelor,
        critical=True,
    )
    credit_value_text = ex.credit_hours_value or "at least 120"
    await add_url_supported_leaf(
        evaluator,
        min_credits,
        existence_id="Credit_Hours_URLs_Present",
        existence_desc="Provide URL(s) specifying credit hour requirements: URLs present",
        urls=ex.credit_hours_urls,
        verify_id="URL_Reference_Credit_Hours",
        verify_desc="Provide URL specifying credit hour requirements (source-supported)",
        claim=f"The program requires {credit_value_text} semester credit hours for completion, satisfying the minimum of 120.",
        additional_instruction="If the page shows a number (e.g., 124), treat it as satisfying the 'minimum 120' requirement since 124 ≥ 120.",
        critical=True,
    )

    # Certification / grade levels
    cert = evaluator.add_parallel(
        id="Elementary_Education_Certification",
        desc="Program specifically prepares students for Pennsylvania elementary education certification",
        parent=deg_root,
        critical=True,
    )
    grade_levels = evaluator.add_parallel(
        id="Grade_Level_Specification",
        desc="Program specifies certification grade levels (e.g., K-6, K-4, 4-8)",
        parent=cert,
        critical=True,
    )
    grade_levels_text = ex.certification_grade_levels or "specified Pennsylvania elementary grade levels"
    await add_url_supported_leaf(
        evaluator,
        grade_levels,
        existence_id="Grade_Level_URLs_Present",
        existence_desc="Provide URL(s) confirming certification grade levels: URLs present",
        urls=ex.grade_levels_urls,
        verify_id="URL_Reference_Grade_Levels",
        verify_desc="Provide URL confirming certification grade levels (source-supported)",
        claim=f"The program prepares students for Pennsylvania elementary education certification at {grade_levels_text}.",
        additional_instruction="Accept common Pennsylvania grade spans like PK-4, 4-8, K-6, etc., explicitly stated for certification preparation.",
        critical=True,
    )


async def build_core_academics(evaluator: Evaluator, parent_node, ex: ProgramExtraction) -> None:
    core = evaluator.add_parallel(
        id="Core_Academic_Requirements",
        desc="Program requires completion of specified core academic courses in all 12 required areas",
        parent=parent_node,
        critical=True,
    )

    # Define areas mapping: (node_id, area_name, urls, verify_leaf_id)
    areas: List[Dict[str, Any]] = [
        {"node": "English_Language_Arts_Requirement", "area": "English Language Arts", "urls": ex.english_urls, "leaf": "URL_Reference_English"},
        {"node": "Mathematics_Requirement", "area": "Mathematics", "urls": ex.math_urls, "leaf": "URL_Reference_Mathematics"},
        {"node": "Science_Requirement", "area": "Science", "urls": ex.science_urls, "leaf": "URL_Reference_Science"},
        {"node": "Social_Studies_Requirement", "area": "Social Studies", "urls": ex.social_studies_urls, "leaf": "URL_Reference_Social_Studies"},
        {"node": "Arts_Humanities_Requirement", "area": "Arts and Humanities", "urls": ex.arts_humanities_urls, "leaf": "URL_Reference_Arts"},
        {"node": "Child_Development_Requirement", "area": "Child Development or Educational Psychology", "urls": ex.child_dev_urls, "leaf": "URL_Reference_Child_Development"},
        {"node": "Special_Education_Requirement", "area": "Special Education or Inclusive Education", "urls": ex.special_ed_urls, "leaf": "URL_Reference_Special_Ed"},
        {"node": "ESL_Requirement", "area": "ESL or Multilingual Learner instruction", "urls": ex.esl_urls, "leaf": "URL_Reference_ESL"},
        {"node": "Literacy_Instruction_Requirement", "area": "Reading and Literacy Instruction methods", "urls": ex.literacy_urls, "leaf": "URL_Reference_Literacy"},
        {"node": "Classroom_Management_Requirement", "area": "Classroom Management techniques", "urls": ex.classroom_mgmt_urls, "leaf": "URL_Reference_Management"},
        {"node": "Assessment_Methods_Requirement", "area": "Student Assessment and Evaluation methods", "urls": ex.assessment_urls, "leaf": "URL_Reference_Assessment"},
        {"node": "Technology_Integration_Requirement", "area": "Educational Technology integration", "urls": ex.edtech_urls, "leaf": "URL_Reference_Technology"},
    ]

    # Build nodes and schedule verifications
    batch_items = []
    for info in areas:
        area_parent = evaluator.add_parallel(
            id=info["node"],
            desc=f"Program includes {info['area']} coursework",
            parent=core,
            critical=True,
        )

        # Existence gate
        evaluator.add_custom_node(
            result=bool(info["urls"] and len(info["urls"]) > 0),
            id=f"{info['node']}_URLs_Present",
            desc=f"Provide URL(s) detailing {info['area']} requirements: URLs present",
            parent=area_parent,
            critical=True,
        )

        # Leaf node
        leaf = evaluator.add_leaf(
            id=info["leaf"],
            desc=f"Provide URL detailing {info['area']} requirements (source-supported)",
            parent=area_parent,
            critical=True,
        )

        # Prepare batch verify tuple
        claim = f"The program curriculum includes coursework in {info['area']}."
        add_ins = (
            "Accept if the curriculum or course list explicitly includes courses for this area, "
            "including reasonable synonyms (e.g., 'Language Arts' for ELA, 'Educational Psychology' for Child Development, "
            "'Inclusive Practices' for Special Education, 'ESL/ELL/ML' for English learner instruction)."
        )
        batch_items.append((claim, info["urls"], leaf, add_ins))

    # Execute verifications in parallel
    await evaluator.batch_verify(batch_items)


async def build_clinical_experience(evaluator: Evaluator, parent_node, ex: ProgramExtraction) -> None:
    clinical = evaluator.add_sequential(
        id="Clinical_Experience_Requirements",
        desc="Program includes required field and clinical experiences",
        parent=parent_node,
        critical=True,
    )

    # Early field experiences
    early = evaluator.add_parallel(
        id="Early_Field_Experiences",
        desc="Program requires early field observations in K-12 settings before student teaching",
        parent=clinical,
        critical=True,
    )
    await add_url_supported_leaf(
        evaluator,
        early,
        existence_id="Early_Field_URLs_Present",
        existence_desc="Provide URL(s) detailing early field experience requirements: URLs present",
        urls=ex.early_field_urls,
        verify_id="URL_Reference_Early_Field",
        verify_desc="Provide URL detailing early field experience requirements (source-supported)",
        claim="The program requires early field observations in K-12 settings before student teaching.",
        additional_instruction="Accept if the page states early fieldwork/observations/practicum in K-12 classrooms prior to student teaching.",
        critical=True,
    )

    # Student teaching requirement (parallel sub-node)
    st = evaluator.add_parallel(
        id="Student_Teaching_Requirement",
        desc="Program requires full-time student teaching experience",
        parent=clinical,
        critical=True,
    )

    # Duration + accredited K-12 school
    st_duration = evaluator.add_parallel(
        id="Student_Teaching_Duration",
        desc="Program requires full-time student teaching lasting at least one semester in accredited K-12 school",
        parent=st,
        critical=True,
    )
    await add_url_supported_leaf(
        evaluator,
        st_duration,
        existence_id="Student_Teaching_Duration_URLs_Present",
        existence_desc="Provide URL(s) detailing student teaching requirements and duration: URLs present",
        urls=ex.student_teaching_duration_urls,
        verify_id="URL_Reference_Student_Teaching",
        verify_desc="Provide URL detailing student teaching requirements and duration (source-supported)",
        claim="The program requires full-time student teaching lasting at least one semester and it takes place in an accredited K-12 school.",
        additional_instruction="Accept if the page indicates a one-semester/14+ weeks full-time placement; accredited K-12 school language may appear as 'approved/accredited school district'.",
        critical=True,
    )

    # Prerequisites (≥60 credits before student teaching)
    st_prereq = evaluator.add_parallel(
        id="Prerequisites_For_Student_Teaching",
        desc="Program requires completion of minimum 60 credit hours before student teaching enrollment",
        parent=st,
        critical=True,
    )
    await add_url_supported_leaf(
        evaluator,
        st_prereq,
        existence_id="Student_Teaching_Prereq_URLs_Present",
        existence_desc="Provide URL(s) confirming student teaching prerequisites: URLs present",
        urls=ex.student_teaching_prereq_urls,
        verify_id="URL_Reference_Prerequisites",
        verify_desc="Provide URL confirming student teaching prerequisites (source-supported)",
        claim="The program requires completion of a minimum of 60 credit hours before a student can enroll in student teaching.",
        additional_instruction="Accept if the page states 60+ credits (or equivalent semester hours) required prior to student teaching.",
        critical=True,
    )


async def build_gpa_requirements(evaluator: Evaluator, parent_node, ex: ProgramExtraction) -> None:
    gpa = evaluator.add_parallel(
        id="GPA_Requirements",
        desc="Program maintains specific GPA standards",
        parent=parent_node,
        critical=True,
    )

    # Admission GPA (must be 2.8)
    adm = evaluator.add_parallel(
        id="Program_Admission_GPA",
        desc="Program requires minimum 2.8 GPA for admission to education program",
        parent=gpa,
        critical=True,
    )
    await add_url_supported_leaf(
        evaluator,
        adm,
        existence_id="Admission_GPA_URLs_Present",
        existence_desc="Provide URL(s) confirming admission GPA requirement: URLs present",
        urls=ex.admission_gpa_urls,
        verify_id="URL_Reference_Admission_GPA",
        verify_desc="Provide URL confirming admission GPA requirement (source-supported)",
        claim="The program requires a minimum 2.8 GPA for admission to the education program.",
        additional_instruction="The page must explicitly state 2.8 minimum GPA for admission into the education program, not a higher or lower figure.",
        critical=True,
    )

    # Continuation GPA
    cont = evaluator.add_parallel(
        id="Continuation_GPA",
        desc="Program requires minimum GPA to remain in program",
        parent=gpa,
        critical=True,
    )
    cont_text = ex.continuation_gpa_value or "a specified minimum GPA"
    await add_url_supported_leaf(
        evaluator,
        cont,
        existence_id="Continuation_GPA_URLs_Present",
        existence_desc="Provide URL(s) confirming continuation GPA requirement: URLs present",
        urls=ex.continuation_gpa_urls,
        verify_id="URL_Reference_Continuation_GPA",
        verify_desc="Provide URL confirming continuation GPA requirement (source-supported)",
        claim=f"The program specifies {cont_text} requirement for students to remain in the program.",
        additional_instruction="Accept if the page states the ongoing minimum GPA standard for continuation/retention in the education program.",
        critical=True,
    )


async def build_certification_preparation(evaluator: Evaluator, parent_node, ex: ProgramExtraction) -> None:
    cert = evaluator.add_parallel(
        id="Certification_Preparation",
        desc="Program prepares students for Pennsylvania Instructional I Certificate",
        parent=parent_node,
        critical=True,
    )

    # Instructional I eligibility
    instr = evaluator.add_parallel(
        id="Pennsylvania_Level_I_Certificate",
        desc="Program completion leads to eligibility for Pennsylvania Instructional I Certificate",
        parent=cert,
        critical=True,
    )
    await add_url_supported_leaf(
        evaluator,
        instr,
        existence_id="Instructional_I_URLs_Present",
        existence_desc="Provide URL(s) confirming certification preparation: URLs present",
        urls=ex.instructional_I_urls,
        verify_id="URL_Reference_Certification",
        verify_desc="Provide URL confirming certification preparation (source-supported)",
        claim="Program completion leads to eligibility for the Pennsylvania Instructional I Certificate.",
        additional_instruction="Accept language such as 'eligible for Instructional I', 'recommendation for certification in Pennsylvania', or equivalent official statements.",
        critical=True,
    )

    # Required certification exams preparation
    exams = evaluator.add_parallel(
        id="Praxis_Or_Required_Exams",
        desc="Program prepares students for required Pennsylvania certification exams",
        parent=cert,
        critical=True,
    )
    await add_url_supported_leaf(
        evaluator,
        exams,
        existence_id="Exams_URLs_Present",
        existence_desc="Provide URL(s) detailing required certification exams: URLs present",
        urls=ex.exams_urls,
        verify_id="URL_Reference_Exams",
        verify_desc="Provide URL detailing required certification exams (source-supported)",
        claim="The program prepares students for the required Pennsylvania certification examinations (e.g., Praxis or PECT).",
        additional_instruction="Accept if the page lists required PA certification tests (Praxis/PECT) for the elementary certification sequence.",
        critical=True,
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
) -> Dict:
    # Initialize evaluator (root non-critical; create critical top node under root)
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

    # Extract structured program info
    ex: ProgramExtraction = await evaluator.extract(
        prompt=prompt_extract_program(),
        template_class=ProgramExtraction,
        extraction_name="program_extraction",
    )

    # Create critical main node to mirror rubric root
    main = evaluator.add_parallel(
        id="Pennsylvania_Elementary_Education_Program",
        desc="Identifies a Pennsylvania university undergraduate elementary education program meeting all specified criteria",
        parent=root,
        critical=True,
    )

    # Mandatory names provided
    evaluator.add_custom_node(
        result=bool(ex.university_name and ex.university_name.strip()),
        id="University_Name_Provided",
        desc="University name is provided",
        parent=main,
        critical=True,
    )
    evaluator.add_custom_node(
        result=bool(ex.program_name and ex.program_name.strip()),
        id="Program_Name_Provided",
        desc="Specific program name is provided",
        parent=main,
        critical=True,
    )

    # Build and verify categories
    await build_university_basic_info(evaluator, main, ex)
    await build_program_accreditation(evaluator, main, ex)
    await build_degree_requirements(evaluator, main, ex)
    await build_core_academics(evaluator, main, ex)
    await build_clinical_experience(evaluator, main, ex)
    await build_gpa_requirements(evaluator, main, ex)
    await build_certification_preparation(evaluator, main, ex)

    # Add custom info summary
    evaluator.add_custom_info(
        info={
            "university_name": ex.university_name,
            "program_name": ex.program_name,
            "grade_levels": ex.certification_grade_levels,
            "credit_hours_value": ex.credit_hours_value,
            "admission_gpa_value": ex.admission_gpa_value,
            "continuation_gpa_value": ex.continuation_gpa_value,
        },
        info_type="summary",
        info_name="extracted_summary",
    )

    return evaluator.get_summary()