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
TASK_ID = "dual_enrollment_nacep_three_states"
TASK_DESCRIPTION = (
    "Identify three dual enrollment programs in the United States that meet ALL of the specified requirements, "
    "including NACEP accreditation (CEP/CPF), geographic coverage (PA, IL, WI), public/community institution type, "
    "multiple high school partnerships, GPA and grade eligibility, instructor qualifications, STEM offerings, and "
    "in-state credit transfer agreements. Provide URLs for all claims."
)

TARGET_STATES = {"pennsylvania": ["pennsylvania", "pa"], "illinois": ["illinois", "il"], "wisconsin": ["wisconsin", "wi"]}


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class ProgramInfo(BaseModel):
    institution_name: Optional[str] = None
    program_name: Optional[str] = None

    state: Optional[str] = None
    location_urls: List[str] = Field(default_factory=list)

    # Institution type
    institution_type: Optional[str] = None  # e.g., "community college", "public university"
    institution_type_urls: List[str] = Field(default_factory=list)

    # NACEP accreditation
    nacep_accreditation_model: Optional[str] = None  # "CEP", "CPF", or descriptive text
    accreditation_urls: List[str] = Field(default_factory=list)

    # High school partnerships
    partnership_description: Optional[str] = None
    partnership_urls: List[str] = Field(default_factory=list)

    # Student eligibility: GPA and grades
    gpa_requirement_text: Optional[str] = None  # e.g., "2.5 GPA on a 4.0 scale"
    gpa_urls: List[str] = Field(default_factory=list)
    grade_levels_text: Optional[str] = None     # e.g., "Grades 10-12"
    grade_urls: List[str] = Field(default_factory=list)

    # Instructor qualifications
    instructor_requirements_text: Optional[str] = None
    instructor_urls: List[str] = Field(default_factory=list)

    # STEM course offerings
    stem_subjects: List[str] = Field(default_factory=list)  # list of subject names (math, CS, etc.)
    course_examples: List[str] = Field(default_factory=list)
    course_urls: List[str] = Field(default_factory=list)

    # Credit transfer
    transfer_description: Optional[str] = None
    transfer_urls: List[str] = Field(default_factory=list)


class ProgramsExtraction(BaseModel):
    programs: List[ProgramInfo] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_programs() -> str:
    return """
Extract up to three dual enrollment programs as presented in the answer. For each program, extract the following fields and ensure you also extract the reference URLs that the answer cites for each claim.

For each program, extract:
- institution_name: The institution offering the program.
- program_name: The dual enrollment program name or label used by the institution.
- state: The U.S. state where the program is located (use full name if provided; otherwise extract as given).
- location_urls: All URLs that the answer cites to support the location/state of the program.

- institution_type: The institution classification (e.g., "community college" or "public university"). Do not infer; extract as written in the answer.
- institution_type_urls: All URLs that the answer cites to support the institution type.

- nacep_accreditation_model: The NACEP accreditation model text (e.g., "CEP" or "CPF") or the accreditation statement as cited in the answer.
- accreditation_urls: All URLs cited that directly support NACEP accreditation (e.g., NACEP directory page or the institution’s accreditation page).

- partnership_description: A short description of high school partnerships (e.g., “partnered with multiple high schools; courses taught at the high school or on campus”).
- partnership_urls: All URLs that support the partnerships claim.

- gpa_requirement_text: The minimum GPA requirement text as stated (e.g., "2.5 GPA on 4.0 scale"). Do not normalize; extract exactly as written.
- gpa_urls: All URLs that support the GPA requirement.
- grade_levels_text: The grade levels eligible (e.g., "Grades 10-12").
- grade_urls: All URLs that support the grade level eligibility.

- instructor_requirements_text: Instructor qualification requirement text (e.g., "Master’s degree in discipline OR Master’s + 18 graduate credits").
- instructor_urls: All URLs that support instructor qualification requirements.

- stem_subjects: A list of STEM subject areas explicitly offered by the program (e.g., ["mathematics", "computer science"]). Include only from {mathematics, engineering, computer science, natural sciences/biology/chemistry/physics}.
- course_examples: A list of example course titles (if mentioned).
- course_urls: All URLs that support course offerings.

- transfer_description: Short description of transfer/articulation agreements with public universities in the same state (if provided).
- transfer_urls: All URLs that support transfer or articulation agreements.

Return JSON with a top-level key "programs" that is an array of objects with these fields. If any field is missing in the answer, set it to null or [] accordingly. Do not fabricate URLs; only include those explicitly cited in the answer.
"""


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def normalize_state_token(s: Optional[str]) -> str:
    if not s:
        return ""
    s_norm = s.strip().lower()
    return s_norm


def is_state_match(state_text: Optional[str], target: str) -> bool:
    # target: "pennsylvania", "illinois", or "wisconsin"
    if not state_text:
        return False
    tnorm = target.lower()
    snorm = normalize_state_token(state_text)
    tokens = TARGET_STATES.get(tnorm, [])
    return any(tok in snorm for tok in tokens)


def non_empty_urls(urls: Optional[List[str]]) -> bool:
    return bool(urls) and len([u for u in urls if isinstance(u, str) and u.strip()]) > 0


def display_program_name(p: ProgramInfo, idx: int) -> str:
    parts = []
    if p.institution_name:
        parts.append(p.institution_name)
    if p.program_name:
        parts.append(p.program_name)
    if not parts:
        return f"Program #{idx + 1}"
    return " - ".join(parts)


# --------------------------------------------------------------------------- #
# Verification logic per program                                              #
# --------------------------------------------------------------------------- #
async def verify_program(evaluator: Evaluator, parent_node, program: ProgramInfo, idx: int) -> None:
    """
    Build the verification subtree for a single program.
    """
    prog_id = f"program_{idx + 1}"
    prog_node = evaluator.add_parallel(
        id=prog_id,
        desc=f"{['First','Second','Third'][idx]} qualifying dual enrollment program",
        parent=parent_node,
        critical=False  # allow partial credit across programs
    )

    pretty_name = display_program_name(program, idx)

    # ------------------ Accreditation ------------------
    acc_node = evaluator.add_parallel(
        id=f"{prog_id}_accreditation",
        desc="NACEP accreditation status verification",
        parent=prog_node,
        critical=True
    )

    # Existence of accreditation URLs (critical)
    evaluator.add_custom_node(
        result=non_empty_urls(program.accreditation_urls),
        id=f"{prog_id}_accreditation_url",
        desc="Reference URL confirming NACEP accreditation status",
        parent=acc_node,
        critical=True
    )

    # Accreditation status check (critical, source-grounded)
    acc_status_leaf = evaluator.add_leaf(
        id=f"{prog_id}_accreditation_status",
        desc="Program must have current NACEP accreditation (either CEP or CPF model)",
        parent=acc_node,
        critical=True
    )
    acc_model_text = program.nacep_accreditation_model or "CEP or CPF model"
    acc_claim = (
        f"{pretty_name} holds current NACEP accreditation, acceptable under the CEP (Concurrent Enrollment Program) "
        f"or CPF (College-Provided Faculty) model. The accreditation is described as '{acc_model_text}'."
    )
    await evaluator.verify(
        claim=acc_claim,
        node=acc_status_leaf,
        sources=program.accreditation_urls,
        additional_instruction="Verify that the provided page(s) explicitly indicate current NACEP accreditation. Accept CEP or CPF."
    )

    # ------------------ Location ------------------
    loc_node = evaluator.add_parallel(
        id=f"{prog_id}_location",
        desc="Geographic location requirements",
        parent=prog_node,
        critical=True
    )

    evaluator.add_custom_node(
        result=non_empty_urls(program.location_urls),
        id=f"{prog_id}_location_url",
        desc="Reference URL confirming program location",
        parent=loc_node,
        critical=True
    )

    state_text = program.state or ""
    state_leaf = evaluator.add_leaf(
        id=f"{prog_id}_state",
        desc="Program must be located in Pennsylvania, Illinois, or Wisconsin",
        parent=loc_node,
        critical=True
    )
    state_claim = (
        f"{pretty_name} is located in {state_text}, and thus is in one of the target states "
        f"(Pennsylvania, Illinois, or Wisconsin)."
    )
    await evaluator.verify(
        claim=state_claim,
        node=state_leaf,
        sources=program.location_urls,
        additional_instruction="Check the institution/program page for the state location."
    )

    # ------------------ Institution Type ------------------
    inst_node = evaluator.add_parallel(
        id=f"{prog_id}_institution_type",
        desc="Type of postsecondary institution offering the program",
        parent=prog_node,
        critical=True
    )

    evaluator.add_custom_node(
        result=non_empty_urls(program.institution_type_urls),
        id=f"{prog_id}_institution_url",
        desc="Reference URL confirming institution type",
        parent=inst_node,
        critical=True
    )

    inst_class_leaf = evaluator.add_leaf(
        id=f"{prog_id}_institution_classification",
        desc="Institution must be a community college or public university",
        parent=inst_node,
        critical=True
    )
    inst_type_text = program.institution_type or "community college or public university"
    inst_claim = (
        f"{pretty_name} is a {inst_type_text}, and not a private university."
    )
    await evaluator.verify(
        claim=inst_claim,
        node=inst_class_leaf,
        sources=program.institution_type_urls,
        additional_instruction="Confirm the institution is either a community college or a public university; reject private universities."
    )

    # ------------------ High School Partnerships ------------------
    partner_node = evaluator.add_parallel(
        id=f"{prog_id}_partnership",
        desc="High school partnership requirements",
        parent=prog_node,
        critical=True
    )

    evaluator.add_custom_node(
        result=non_empty_urls(program.partnership_urls),
        id=f"{prog_id}_partnership_url",
        desc="Reference URL confirming partnership details",
        parent=partner_node,
        critical=True
    )

    partnership_leaf = evaluator.add_leaf(
        id=f"{prog_id}_partnership_scope",
        desc="Program must have partnerships with multiple high schools allowing courses at high school or college campus",
        parent=partner_node,
        critical=True
    )
    partnership_desc = program.partnership_description or "multiple high school partnerships enabling courses on HS campus or college campus"
    partnership_claim = (
        f"{pretty_name} has partnerships with multiple high schools that allow students to take college courses "
        f"either at their high school or on the college campus (e.g., '{partnership_desc}')."
    )
    await evaluator.verify(
        claim=partnership_claim,
        node=partnership_leaf,
        sources=program.partnership_urls,
        additional_instruction="Look for phrases like 'partner high schools', 'multiple high schools', and course location (HS or college campus)."
    )

    # ------------------ Student Eligibility ------------------
    elig_node = evaluator.add_parallel(
        id=f"{prog_id}_student_eligibility",
        desc="Student eligibility requirements",
        parent=prog_node,
        critical=True
    )

    # GPA requirement subgroup
    gpa_group = evaluator.add_parallel(
        id=f"{prog_id}_gpa_requirement",
        desc="Minimum GPA requirement verification",
        parent=elig_node,
        critical=True
    )

    evaluator.add_custom_node(
        result=non_empty_urls(program.gpa_urls),
        id=f"{prog_id}_gpa_url",
        desc="Reference URL confirming GPA requirement",
        parent=gpa_group,
        critical=True
    )

    gpa_leaf = evaluator.add_leaf(
        id=f"{prog_id}_gpa_value",
        desc="Program must specify minimum GPA requirement of at least 2.5 on a 4.0 scale",
        parent=gpa_group,
        critical=True
    )
    gpa_text = program.gpa_requirement_text or "minimum GPA requirement"
    gpa_claim = (
        f"{pretty_name} specifies a minimum cumulative high school GPA requirement of at least 2.5 on a 4.0 scale "
        f"(stated as '{gpa_text}')."
    )
    await evaluator.verify(
        claim=gpa_claim,
        node=gpa_leaf,
        sources=program.gpa_urls,
        additional_instruction="Confirm the minimum GPA is 2.5 or higher on a 4.0 scale."
    )

    # Grade level requirement subgroup
    grade_group = evaluator.add_parallel(
        id=f"{prog_id}_grade_level",
        desc="Grade level eligibility verification",
        parent=elig_node,
        critical=True
    )

    evaluator.add_custom_node(
        result=non_empty_urls(program.grade_urls),
        id=f"{prog_id}_grade_url",
        desc="Reference URL confirming grade level requirements",
        parent=grade_group,
        critical=True
    )

    grade_leaf = evaluator.add_leaf(
        id=f"{prog_id}_grade_requirement",
        desc="Program must serve students in grades 10, 11, or 12",
        parent=grade_group,
        critical=True
    )
    grade_text = program.grade_levels_text or "grades 10–12"
    grade_claim = (
        f"{pretty_name} serves high school students at least in grades 10, 11, and 12 "
        f"(as described: '{grade_text}')."
    )
    await evaluator.verify(
        claim=grade_claim,
        node=grade_leaf,
        sources=program.grade_urls,
        additional_instruction="Confirm that eligible grade levels include 10th, 11th, and 12th grades (may include additional grades)."
    )

    # ------------------ Instructor Qualifications ------------------
    instr_node = evaluator.add_parallel(
        id=f"{prog_id}_instructor_qualifications",
        desc="Instructor qualification requirements",
        parent=prog_node,
        critical=True
    )

    evaluator.add_custom_node(
        result=non_empty_urls(program.instructor_urls),
        id=f"{prog_id}_instructor_url",
        desc="Reference URL confirming instructor qualification requirements",
        parent=instr_node,
        critical=True
    )

    instr_leaf = evaluator.add_leaf(
        id=f"{prog_id}_instructor_degree",
        desc="Program must require instructors to have master's degree in content area OR master's degree plus 18 graduate credits in teaching field",
        parent=instr_node,
        critical=True
    )
    instr_text = program.instructor_requirements_text or "Master’s in discipline OR Master’s + 18 graduate credits in subject"
    instr_claim = (
        f"{pretty_name} requires that instructors have either a master's degree in the subject area they teach, "
        f"or a master's degree in any field plus at least 18 graduate credits in the subject area (as stated: '{instr_text}')."
    )
    await evaluator.verify(
        claim=instr_claim,
        node=instr_leaf,
        sources=program.instructor_urls,
        additional_instruction="Look for explicit qualification statements referencing a master's degree and/or 18 graduate credits in discipline."
    )

    # ------------------ Course Offerings (STEM) ------------------
    course_node = evaluator.add_parallel(
        id=f"{prog_id}_course_offerings",
        desc="Course offerings in specified subject area",
        parent=prog_node,
        critical=True
    )

    evaluator.add_custom_node(
        result=non_empty_urls(program.course_urls),
        id=f"{prog_id}_course_url",
        desc="Reference URL confirming course offerings",
        parent=course_node,
        critical=True
    )

    subject_leaf = evaluator.add_leaf(
        id=f"{prog_id}_subject_availability",
        desc="Program must offer courses in STEM subject areas (mathematics, engineering, computer science, or natural sciences)",
        parent=course_node,
        critical=True
    )
    subjects_list = program.stem_subjects or []
    subjects_text = ", ".join(subjects_list) if subjects_list else "one or more STEM subjects"
    subject_claim = (
        f"{pretty_name} offers dual enrollment courses in at least one STEM subject area among "
        f"mathematics, engineering, computer science, or natural sciences (e.g., {subjects_text})."
    )
    await evaluator.verify(
        claim=subject_claim,
        node=subject_leaf,
        sources=program.course_urls,
        additional_instruction="Confirm that at least one listed course/subject is within mathematics, engineering, computer science, or natural sciences."
    )

    # ------------------ Credit Transfer ------------------
    transfer_node = evaluator.add_parallel(
        id=f"{prog_id}_credit_transfer",
        desc="Credit transfer agreement verification",
        parent=prog_node,
        critical=True
    )

    evaluator.add_custom_node(
        result=non_empty_urls(program.transfer_urls),
        id=f"{prog_id}_transfer_url",
        desc="Reference URL confirming transfer agreements",
        parent=transfer_node,
        critical=True
    )

    transfer_leaf = evaluator.add_leaf(
        id=f"{prog_id}_transfer_agreement",
        desc="Program must have established credit transfer agreements with at least one public university in its state",
        parent=transfer_node,
        critical=True
    )
    state_for_transfer = program.state or "the program's state"
    transfer_desc = program.transfer_description or "transfer/articulation details"
    transfer_claim = (
        f"{pretty_name} has established articulation/transfer agreements with at least one public university "
        f"in {state_for_transfer}. Example/description: '{transfer_desc}'."
    )
    await evaluator.verify(
        claim=transfer_claim,
        node=transfer_leaf,
        sources=program.transfer_urls,
        additional_instruction="Verify that at least one named receiving institution is a public university within the same state."
    )


# --------------------------------------------------------------------------- #
# Geographic diversity verification                                           #
# --------------------------------------------------------------------------- #
def add_geographic_diversity_checks(evaluator: Evaluator, parent_node, programs: List[ProgramInfo]) -> None:
    """
    Add critical geographic diversity checks ensuring coverage of PA, IL, and WI.
    """
    geo_node = evaluator.add_parallel(
        id="geographic_diversity",
        desc="Verify that the three programs collectively cover Pennsylvania, Illinois, and Wisconsin",
        parent=parent_node,
        critical=True
    )

    states_present = [p.state or "" for p in programs[:3]]

    pa_ok = any(is_state_match(s, "pennsylvania") for s in states_present)
    il_ok = any(is_state_match(s, "illinois") for s in states_present)
    wi_ok = any(is_state_match(s, "wisconsin") for s in states_present)

    evaluator.add_custom_node(
        result=pa_ok,
        id="pennsylvania_coverage",
        desc="At least one program must be located in Pennsylvania",
        parent=geo_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=il_ok,
        id="illinois_coverage",
        desc="At least one program must be located in Illinois",
        parent=geo_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=wi_ok,
        id="wisconsin_coverage",
        desc="At least one program must be located in Wisconsin",
        parent=geo_node,
        critical=True
    )

    evaluator.add_custom_info(
        info={"states_extracted": states_present},
        info_type="geographic_diversity_states",
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
    Evaluate an answer for the dual enrollment NACEP + three-state coverage task.
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

    # Extract up to three programs
    extracted = await evaluator.extract(
        prompt=prompt_extract_programs(),
        template_class=ProgramsExtraction,
        extraction_name="programs_extraction"
    )

    # Normalize count to exactly 3, padding with empty placeholders if needed
    programs: List[ProgramInfo] = list(extracted.programs[:3])
    while len(programs) < 3:
        programs.append(ProgramInfo())

    # Add geographic diversity critical checks
    add_geographic_diversity_checks(evaluator, root, programs)

    # Build verification subtrees for each program
    for i in range(3):
        await verify_program(evaluator, root, programs[i], i)

    return evaluator.get_summary()