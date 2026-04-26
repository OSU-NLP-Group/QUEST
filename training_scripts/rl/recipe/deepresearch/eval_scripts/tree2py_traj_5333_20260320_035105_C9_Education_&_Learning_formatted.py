import asyncio
import logging
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "hcs_dual_enrollment_partners"
TASK_DESCRIPTION = (
    "Huntsville City Schools in Alabama is expanding its dual enrollment program to serve high school students in "
    "grades 10-12 who maintain a minimum 2.5 GPA and have completed all required coursework through 10th grade. "
    "The district has identified three universities as potential partners: Miami University in Ohio, the University "
    "of North Carolina at Chapel Hill, and Case Western Reserve University in Ohio. Evaluate each of these three "
    "universities to determine their suitability as dual enrollment partners by providing the following information "
    "for each institution:\n\n"
    "1. Confirm the university holds regional accreditation\n"
    "2. Provide the total student enrollment (all levels combined)\n"
    "3. Identify the Carnegie Research Classification if the university is designated R1 or R2\n"
    "4. Confirm whether the university offers dual enrollment or early college programs\n"
    "5. Describe the types of academic programs or course levels available through such programs\n"
    "6. Identify whether the university offers an honors program and, if so, the minimum GPA requirement for admission\n"
    "7. Specify the university's NCAA Division status, if applicable\n"
    "8. Provide URL references that verify each piece of information\n\n"
    "For context, confirm that Huntsville City Schools' dual enrollment requirements align with Alabama state "
    "requirements for 10th-12th grade students with a 2.5 minimum GPA."
)


# --------------------------------------------------------------------------- #
# Data models for structured extraction                                       #
# --------------------------------------------------------------------------- #
class UniversityInfo(BaseModel):
    name: Optional[str] = None

    # Institutional profile
    accreditation_status: Optional[str] = None            # e.g., "regionally accredited", "yes", "no"
    accreditor: Optional[str] = None                      # e.g., "Higher Learning Commission", "SACSCOC"
    accreditation_urls: List[str] = Field(default_factory=list)

    total_enrollment: Optional[str] = None                # keep as string (allow ranges/approx)
    enrollment_urls: List[str] = Field(default_factory=list)

    carnegie_classification: Optional[str] = None         # e.g., "R1", "R2", "Doctoral/Professional"
    carnegie_urls: List[str] = Field(default_factory=list)

    institutional_profile_urls: List[str] = Field(default_factory=list)

    # Program offerings
    dual_enrollment_available: Optional[str] = None       # "yes"/"no"/"unknown"
    dual_enrollment_description: Optional[str] = None     # description of offerings / levels
    program_urls: List[str] = Field(default_factory=list)

    # Student support / Honors
    honors_program_available: Optional[str] = None        # "yes"/"no"/"unknown"
    honors_min_gpa: Optional[str] = None                  # e.g., "3.5", "3.7 weighted", "N/A"
    honors_urls: List[str] = Field(default_factory=list)

    # Athletics
    ncaa_division: Optional[str] = None                   # e.g., "NCAA Division I", "Division III"
    athletics_urls: List[str] = Field(default_factory=list)


class UniversitiesExtraction(BaseModel):
    miami_university_ohio: Optional[UniversityInfo] = None
    unc_chapel_hill: Optional[UniversityInfo] = None
    case_western_reserve: Optional[UniversityInfo] = None


class AlabamaRequirements(BaseModel):
    grades_covered_statement: Optional[str] = None        # e.g., "grades 10-12", "10th–12th"
    minimum_gpa_statement: Optional[str] = None           # e.g., "2.5 GPA"
    prerequisite_statement: Optional[str] = None          # e.g., "completed all required coursework through 10th grade"
    source_urls: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_universities() -> str:
    return """
    Extract structured information about the following three institutions exactly as presented in the answer:
    - Miami University (Ohio)
    - University of North Carolina at Chapel Hill
    - Case Western Reserve University (Ohio)

    For each institution, extract these fields from the answer text exactly as stated (do not infer or add):
    - name: The institution's name as written
    - accreditation_status: Whether the university holds regional accreditation (e.g., "regionally accredited", "yes", "no", or exact phrasing used)
    - accreditor: The regional accrediting body if stated (e.g., Higher Learning Commission, SACSCOC)
    - accreditation_urls: All URLs cited that support accreditation info
    - total_enrollment: The total student enrollment (all levels combined) as reported in the answer
    - enrollment_urls: All URLs cited that support enrollment figures
    - carnegie_classification: The Carnegie research classification as stated (e.g., "R1", "R2", or exact phrase)
    - carnegie_urls: All URLs cited that support Carnegie info
    - institutional_profile_urls: Any URLs cited that provide general institutional profile information
    - dual_enrollment_available: "yes", "no", or exact phrasing whether the university offers dual enrollment or early college
    - dual_enrollment_description: The description of the types/levels of courses or programs available to high school students via dual enrollment/early college
    - program_urls: All URLs cited that support dual enrollment or early college program info (include College Credit Plus pages for Ohio schools if mentioned)
    - honors_program_available: "yes", "no", or exact phrasing
    - honors_min_gpa: The minimum GPA for honors admission as stated; if not provided, return null
    - honors_urls: All URLs cited that support honors information
    - ncaa_division: The NCAA division classification as stated (e.g., "NCAA Division I", "Division III")
    - athletics_urls: All URLs cited that support NCAA division info

    Return an object with keys:
    - miami_university_ohio
    - unc_chapel_hill
    - case_western_reserve

    Each key should hold an object with the fields above. If any field is not explicitly given in the answer, set it to null or an empty list (for the URL fields).
    URLs must be exactly those explicitly present in the answer (plain or markdown). Do not invent URLs.
    """


def prompt_extract_alabama_requirements() -> str:
    return """
    Extract the Alabama dual enrollment alignment information as stated or cited in the answer.
    Fields to extract:
    - grades_covered_statement: The exact text/phrase the answer uses for eligible grades (e.g., "grades 10-12")
    - minimum_gpa_statement: The exact text/phrase for minimum GPA (e.g., "2.5 GPA")
    - prerequisite_statement: The exact text/phrase regarding completion of required coursework through 10th grade (or equivalent prerequisite)
    - source_urls: All URLs cited that support Alabama dual enrollment eligibility/requirements

    If any field is missing from the answer, set it to null or return an empty list for URLs.
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _merge_urls(*url_lists: Optional[List[str]]) -> List[str]:
    dedup: List[str] = []
    for lst in url_lists:
        if not lst:
            continue
        for u in lst:
            if isinstance(u, str):
                uu = u.strip()
                if uu and uu not in dedup:
                    dedup.append(uu)
    return dedup


def _bool_from_str(val: Optional[str]) -> Optional[bool]:
    if val is None:
        return None
    v = val.strip().lower()
    if v in {"yes", "y", "true", "t"}:
        return True
    if v in {"no", "n", "false", "f"}:
        return False
    return None


# --------------------------------------------------------------------------- #
# Verification builders                                                       #
# --------------------------------------------------------------------------- #
async def verify_alabama_requirements(
    evaluator: Evaluator,
    parent_node,
    alabama: AlabamaRequirements,
) -> None:
    # Alabama requirements node (parallel, non-critical as in rubric)
    alb_node = evaluator.add_parallel(
        id="alabama_requirements_verification",
        desc="Verify that stated requirements align with Alabama dual enrollment standards",
        parent=parent_node,
        critical=False
    )

    alb_sources = alabama.source_urls if alabama else []

    # Grade Level Alignment (critical leaf under this branch)
    grade_leaf = evaluator.add_leaf(
        id="alabama_grade_level_alignment",
        desc="Confirm that grades 10-12 requirement matches Alabama standards",
        parent=alb_node,
        critical=True
    )
    grade_claim = "Alabama's dual enrollment eligibility includes high school students in grades 10 through 12."
    await evaluator.verify(
        claim=grade_claim,
        node=grade_leaf,
        sources=alb_sources,
        additional_instruction="Judge only based on the provided Alabama policy/agency pages. Accept equivalent phrasing such as '10th-12th graders'."
    )

    # GPA Requirement Alignment (critical leaf)
    gpa_leaf = evaluator.add_leaf(
        id="alabama_gpa_requirement_alignment",
        desc="Confirm that 2.5 GPA minimum matches Alabama standards",
        parent=alb_node,
        critical=True
    )
    gpa_claim = "Alabama requires a minimum cumulative GPA of at least 2.5 for participation in dual enrollment."
    await evaluator.verify(
        claim=gpa_claim,
        node=gpa_leaf,
        sources=alb_sources,
        additional_instruction="Verify from Alabama state policy/agency sources. Allow small wording variants but the numeric threshold must be 2.5."
    )

    # Prerequisite Alignment (critical leaf)
    prereq_leaf = evaluator.add_leaf(
        id="alabama_prerequisite_alignment",
        desc="Confirm that completion of grades 9-10 coursework matches Alabama standards",
        parent=alb_node,
        critical=True
    )
    prereq_claim = (
        "Alabama dual enrollment policy requires that students have completed all required high school coursework "
        "through 10th grade (i.e., grades 9 and 10) or an equivalent prerequisite prior to dual enrollment."
    )
    await evaluator.verify(
        claim=prereq_claim,
        node=prereq_leaf,
        sources=alb_sources,
        additional_instruction="Confirm only if the cited Alabama policy/agency webpage explicitly states or clearly implies this prerequisite."
    )


async def verify_university(
    evaluator: Evaluator,
    parent_node,
    uni: UniversityInfo,
    slug: str,
    display_name: str,
) -> None:
    # Top-level university evaluation node (parallel, non-critical as in rubric)
    uni_node = evaluator.add_parallel(
        id=f"{slug}_evaluation",
        desc=f"Comprehensive evaluation of {display_name} as a potential partner",
        parent=parent_node,
        critical=False
    )

    # --------------------- Institutional Profile --------------------------
    # Mark non-critical to allow mixed criticality children without violating framework constraints
    inst_profile = evaluator.add_parallel(
        id=f"{slug}_institutional_profile",
        desc=f"Basic institutional characteristics of {display_name}",
        parent=uni_node,
        critical=False
    )

    # Regional Accreditation (critical)
    accred_leaf = evaluator.add_leaf(
        id=f"{slug}_regional_accreditation",
        desc=f"Verification that {display_name} holds regional accreditation",
        parent=inst_profile,
        critical=True
    )
    accred_sources = _merge_urls(uni.accreditation_urls, uni.institutional_profile_urls, uni.carnegie_urls, uni.enrollment_urls)
    if uni.accreditor and uni.accreditation_status:
        accred_claim = f"{display_name} holds regional accreditation from {uni.accreditor}."
    elif uni.accreditation_status:
        accred_claim = f"{display_name} holds regional accreditation."
    else:
        accred_claim = f"{display_name} holds regional accreditation."
    await evaluator.verify(
        claim=accred_claim,
        node=accred_leaf,
        sources=accred_sources,
        additional_instruction=(
            "Confirm that the page explicitly indicates institutional regional accreditation (not programmatic only). "
            "Accept equivalent accreditor naming (e.g., 'Higher Learning Commission' vs 'HLC')."
        )
    )

    # Total Enrollment (non-critical)
    enroll_leaf = evaluator.add_leaf(
        id=f"{slug}_total_enrollment",
        desc=f"Total student enrollment figure for {display_name}",
        parent=inst_profile,
        critical=False
    )
    if uni.total_enrollment:
        enrollment_claim = f"The total student enrollment (all levels combined) at {display_name} is reported as '{uni.total_enrollment}'."
    else:
        enrollment_claim = f"The total student enrollment (all levels combined) at {display_name} is reported as 'unknown' in the answer."
    await evaluator.verify(
        claim=enrollment_claim,
        node=enroll_leaf,
        sources=_merge_urls(uni.enrollment_urls, uni.institutional_profile_urls),
        additional_instruction="Allow approximate phrasings and rounding (e.g., 'about 20,000'). Match the value or an obviously equivalent figure on the page."
    )

    # Carnegie Classification (non-critical)
    carnegie_leaf = evaluator.add_leaf(
        id=f"{slug}_carnegie_classification",
        desc="Carnegie research classification (R1, R2, or other)",
        parent=inst_profile,
        critical=False
    )
    if uni.carnegie_classification:
        carnegie_claim = f"{display_name} is classified by the Carnegie Classification as '{uni.carnegie_classification}'."
    else:
        carnegie_claim = f"The Carnegie Classification for {display_name} is not specified."
    await evaluator.verify(
        claim=carnegie_claim,
        node=carnegie_leaf,
        sources=_merge_urls(uni.carnegie_urls, uni.institutional_profile_urls),
        additional_instruction=(
            "Accept equivalent labels like 'R1: Doctoral Universities – Very High Research Activity' for R1 and "
            "'R2: Doctoral Universities – High Research Activity' for R2."
        )
    )

    # Institutional URL Reference (critical)
    inst_url_leaf = evaluator.add_leaf(
        id=f"{slug}_institutional_url_reference",
        desc="URL supporting institutional profile information",
        parent=inst_profile,
        critical=True
    )
    inst_url_claim = (
        f"This webpage is an official or authoritative institutional profile page for {display_name} and contains "
        "information about accreditation, enrollment, or Carnegie classification."
    )
    await evaluator.verify(
        claim=inst_url_claim,
        node=inst_url_leaf,
        sources=_merge_urls(uni.institutional_profile_urls, uni.accreditation_urls, uni.enrollment_urls, uni.carnegie_urls),
        additional_instruction=(
            "Pages from the university's official domain, recognized accreditors, or Carnegie sites are acceptable. "
            "Reject unrelated or non-authoritative pages."
        )
    )

    # --------------------- Program Offerings ------------------------------
    # Mark non-critical to allow a non-critical child (course types) per framework rules
    prog_node = evaluator.add_parallel(
        id=f"{slug}_program_offerings",
        desc="Dual enrollment and academic program availability",
        parent=uni_node,
        critical=False
    )

    # Dual Enrollment Availability (critical)
    dual_leaf = evaluator.add_leaf(
        id=f"{slug}_dual_enrollment_availability",
        desc="Confirmation of dual enrollment or early college program existence",
        parent=prog_node,
        critical=True
    )
    de_bool = _bool_from_str(uni.dual_enrollment_available)
    if de_bool is True:
        dual_claim = f"{display_name} offers dual enrollment or early college programs for high school students."
    elif de_bool is False:
        dual_claim = f"{display_name} does not offer any dual enrollment or early college programs for high school students."
    else:
        dual_claim = f"{display_name} offers dual enrollment or early college programs for high school students."
    await evaluator.verify(
        claim=dual_claim,
        node=dual_leaf,
        sources=uni.program_urls,
        additional_instruction=(
            "Accept synonyms like 'dual enrollment', 'early college', 'pre-college', 'high school dual', or for Ohio "
            "schools 'College Credit Plus (CCP)'. The page must clearly indicate whether such programs exist."
        )
    )

    # Course Types Available (non-critical)
    course_leaf = evaluator.add_leaf(
        id=f"{slug}_course_types_available",
        desc="Description of course types/levels available to dual enrollment students",
        parent=prog_node,
        critical=False
    )
    if uni.dual_enrollment_description:
        course_claim = (
            f"The dual enrollment/early college offering at {display_name} provides the following types/levels: "
            f"'{uni.dual_enrollment_description}'."
        )
    else:
        course_claim = f"The dual enrollment/early college offering types or levels at {display_name} are unspecified."
    await evaluator.verify(
        claim=course_claim,
        node=course_leaf,
        sources=uni.program_urls,
        additional_instruction=(
            "Look for wording indicating course levels (e.g., 100/200-level), modalities (online/on-campus), or program "
            "structures (e.g., certificate pathways) for high school students."
        )
    )

    # Program URL Reference (critical)
    prog_url_leaf = evaluator.add_leaf(
        id=f"{slug}_program_url_reference",
        desc="URL supporting program information",
        parent=prog_node,
        critical=True
    )
    prog_url_claim = (
        f"This webpage is an official or authoritative page for {display_name} that describes dual enrollment, "
        "early college, or equivalent high school credit opportunities."
    )
    await evaluator.verify(
        claim=prog_url_claim,
        node=prog_url_leaf,
        sources=uni.program_urls,
        additional_instruction=(
            "Prefer official university pages, state program pages (e.g., Ohio College Credit Plus), or clearly "
            "authoritative sources. Reject unrelated pages."
        )
    )

    # --------------------- Student Support (Honors) -----------------------
    # Non-critical branch
    honors_node = evaluator.add_parallel(
        id=f"{slug}_student_support",
        desc="Honors program information",
        parent=uni_node,
        critical=False
    )

    honors_leaf = evaluator.add_leaf(
        id=f"{slug}_honors_program",
        desc="Availability and GPA requirement for honors program",
        parent=honors_node,
        critical=False
    )
    honors_bool = _bool_from_str(uni.honors_program_available)
    if honors_bool is True and uni.honors_min_gpa:
        honors_claim = (
            f"{display_name} offers an honors program, and the minimum GPA requirement for admission is '{uni.honors_min_gpa}'."
        )
    elif honors_bool is True and not uni.honors_min_gpa:
        honors_claim = f"{display_name} offers an honors program."
    elif honors_bool is False:
        honors_claim = f"{display_name} does not offer an undergraduate honors program."
    else:
        honors_claim = f"The availability of an honors program and minimum GPA requirement at {display_name} are unspecified."
    await evaluator.verify(
        claim=honors_claim,
        node=honors_leaf,
        sources=uni.honors_urls,
        additional_instruction="Accept pages from the university's honors college/program site or official admissions pages."
    )

    # --------------------- Athletics (NCAA) -------------------------------
    # Non-critical branch
    athletics_node = evaluator.add_parallel(
        id=f"{slug}_athletic_programs",
        desc="NCAA athletic program information",
        parent=uni_node,
        critical=False
    )

    ncaa_leaf = evaluator.add_leaf(
        id=f"{slug}_ncaa_division",
        desc="NCAA division classification",
        parent=athletics_node,
        critical=False
    )
    if uni.ncaa_division:
        ncaa_claim = f"{display_name} competes in {uni.ncaa_division}."
    else:
        ncaa_claim = f"The NCAA division for {display_name} is unspecified."
    await evaluator.verify(
        claim=ncaa_claim,
        node=ncaa_leaf,
        sources=_merge_urls(uni.athletics_urls, uni.institutional_profile_urls),
        additional_instruction=(
            "Accept official athletics pages (e.g., university athletics sites, NCAA.org, conference pages). "
            "Allow equivalent phrasing (e.g., 'Division I (FBS)' vs 'NCAA Division I')."
        )
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
    # Initialize evaluator (root is always non-critical by framework design)
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Root is parallel per rubric
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

    # 1) Extract structured info (universities + Alabama context)
    universities_task = evaluator.extract(
        prompt=prompt_extract_universities(),
        template_class=UniversitiesExtraction,
        extraction_name="universities_extraction",
    )
    alabama_task = evaluator.extract(
        prompt=prompt_extract_alabama_requirements(),
        template_class=AlabamaRequirements,
        extraction_name="alabama_requirements_extraction",
    )
    universities_ex, alabama_ex = await asyncio.gather(universities_task, alabama_task)

    # 2) Alabama requirements verification
    await verify_alabama_requirements(evaluator, root, alabama_ex)

    # 3) University verifications
    # Miami University (Ohio)
    await verify_university(
        evaluator=evaluator,
        parent_node=root,
        uni=universities_ex.miami_university_ohio or UniversityInfo(name="Miami University (Ohio)"),
        slug="miami_university_ohio",
        display_name="Miami University (Ohio)"
    )

    # UNC Chapel Hill
    await verify_university(
        evaluator=evaluator,
        parent_node=root,
        uni=universities_ex.unc_chapel_hill or UniversityInfo(name="University of North Carolina at Chapel Hill"),
        slug="unc_chapel_hill",
        display_name="University of North Carolina at Chapel Hill"
    )

    # Case Western Reserve University
    await verify_university(
        evaluator=evaluator,
        parent_node=root,
        uni=universities_ex.case_western_reserve or UniversityInfo(name="Case Western Reserve University"),
        slug="case_western_reserve",
        display_name="Case Western Reserve University"
    )

    # 4) Return final structured evaluation summary
    return evaluator.get_summary()