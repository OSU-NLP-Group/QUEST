import asyncio
import logging
from typing import Any, List, Optional, Dict

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys

# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "pb_admissions_ms_4"
TASK_DESCRIPTION = (
    "Identify four accredited online Master's degree programs in Computer Science or Data Science that meet all of the following criteria: "
    "(1) offer performance-based admission where students can gain admission by completing pathway courses with a minimum cumulative GPA of 3.0 or grade B, "
    "without requiring a traditional application, GRE scores, or upfront bachelor's degree verification; "
    "(2) have total program tuition under $20,000 USD; "
    "(3) can be completed in 24 months or less at the recommended full-time pace; "
    "(4) are delivered 100% online with no mandatory on-campus requirements; "
    "(5) are accredited by a recognized regional or national accrediting body. "
    "For each program, provide the university name, specific degree title, total tuition cost, completion duration, performance-based admission pathway requirements "
    "(number of courses and GPA threshold), and reference URLs documenting each attribute."
)

# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class ProgramField(BaseModel):
    name: Optional[str] = None
    field_urls: List[str] = Field(default_factory=list)

class ProgramAdmission(BaseModel):
    pathway_offered_text: Optional[str] = None
    pathway_courses_count_text: Optional[str] = None
    grade_threshold_text: Optional[str] = None
    no_app_gre_degree_verification_text: Optional[str] = None
    admission_urls: List[str] = Field(default_factory=list)

class ProgramTuition(BaseModel):
    total_tuition_text: Optional[str] = None
    tuition_urls: List[str] = Field(default_factory=list)

class ProgramDuration(BaseModel):
    completion_duration_text: Optional[str] = None
    duration_urls: List[str] = Field(default_factory=list)

class ProgramDelivery(BaseModel):
    fully_online_text: Optional[str] = None
    delivery_urls: List[str] = Field(default_factory=list)

class ProgramAccreditation(BaseModel):
    accreditor_name: Optional[str] = None
    accreditation_urls: List[str] = Field(default_factory=list)

class ProgramItem(BaseModel):
    university_name: Optional[str] = None
    degree_title: Optional[str] = None
    field: ProgramField = Field(default_factory=ProgramField)
    institution_urls: List[str] = Field(default_factory=list)  # Established university documentation URLs
    admission: ProgramAdmission = Field(default_factory=ProgramAdmission)
    tuition: ProgramTuition = Field(default_factory=ProgramTuition)
    duration: ProgramDuration = Field(default_factory=ProgramDuration)
    delivery: ProgramDelivery = Field(default_factory=ProgramDelivery)
    accreditation: ProgramAccreditation = Field(default_factory=ProgramAccreditation)

class ProgramsExtraction(BaseModel):
    programs: List[ProgramItem] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_programs() -> str:
    return """
    Extract up to the first four Master's programs described in the answer that are intended to meet the specified criteria.
    For each program, extract the following fields strictly from the answer text:

    1) university_name: The awarding institution / university name.
    2) degree_title: The specific official degree title (e.g., "Master of Computer Science" or "MS in Data Science").
    3) field:
       - name: The program's field as stated (e.g., Computer Science, Data Science, Artificial Intelligence, Analytics, Software Engineering).
       - field_urls: A list of URLs cited in the answer that document the degree title or field (program page, catalog, etc.).
    4) institution_urls: A list of URLs cited that demonstrate the offering institution is a university or degree‑granting institution (official university pages).
    5) admission:
       - pathway_offered_text: A short phrase from the answer that indicates performance‑based admission via pathway courses.
       - pathway_courses_count_text: The number of pathway courses mentioned (keep as text as in the answer).
       - grade_threshold_text: The minimum GPA or grade threshold mentioned (e.g., "3.0 GPA" or "grade B").
       - no_app_gre_degree_verification_text: A phrase indicating no traditional application, no GRE, and no upfront bachelor's degree verification are required to start the pathway.
       - admission_urls: A list of URLs cited that document the performance‑based admission details (course count, GPA/grade threshold, no application/GRE/upfront degree verification).
    6) tuition:
       - total_tuition_text: The total program tuition amount as stated (keep the exact wording, e.g., "$15,000 total" or "approx. $18,500").
       - tuition_urls: A list of URLs cited that document the total tuition cost (tuition page, program cost page).
    7) duration:
       - completion_duration_text: The stated completion duration at the recommended full‑time pace (e.g., "12–18 months" or "24 months").
       - duration_urls: A list of URLs cited that document the completion duration.
    8) delivery:
       - fully_online_text: A phrase indicating 100% online delivery and no mandatory on‑campus requirements.
       - delivery_urls: A list of URLs cited that document the fully online/no residency requirement.
    9) accreditation:
       - accreditor_name: The accrediting body name (e.g., "HLC", "SACSCOC", "MSCHE", "WSCUC", "NECHE", "ABHES", etc.).
       - accreditation_urls: A list of URLs cited that document institutional or program accreditation.

    RULES:
    - Extract only what is explicitly present in the answer. Do not infer or invent any values or URLs.
    - If a requested field is missing, set it to null (for single values) or an empty list (for URLs arrays).
    - For URLs, include only valid URLs present in the answer (plain or markdown). If missing protocol, prepend "http://".
    - Return a JSON object with a "programs" array containing up to four ProgramItem objects with the above fields.
    """


# --------------------------------------------------------------------------- #
# Verification helpers                                                        #
# --------------------------------------------------------------------------- #
def _exists_str(s: Optional[str]) -> bool:
    return bool(s and str(s).strip())

def _exists_list(lst: Optional[List[str]]) -> bool:
    return bool(lst and len(lst) > 0)


# --------------------------------------------------------------------------- #
# Verification for one program                                                #
# --------------------------------------------------------------------------- #
async def verify_program(
    evaluator: Evaluator,
    parent_node,
    prog: ProgramItem,
    index: int,
) -> None:
    pid = index + 1
    program_node = evaluator.add_parallel(
        id=f"program_{pid}",
        desc=f"Program {pid} (one distinct qualifying program)",
        parent=parent_node,
        critical=True  # Child of root must be critical to gate overall pass/fail
    )

    # University name (existence)
    evaluator.add_custom_node(
        result=_exists_str(prog.university_name),
        id=f"program_{pid}_university_name",
        desc=f"Provides the university name for Program {pid}",
        parent=program_node,
        critical=True
    )

    # Degree title (existence)
    evaluator.add_custom_node(
        result=_exists_str(prog.degree_title),
        id=f"program_{pid}_degree_title",
        desc=f"Provides the specific degree title for Program {pid}",
        parent=program_node,
        critical=True
    )

    # Field group
    field_group = evaluator.add_parallel(
        id=f"program_{pid}_field",
        desc=f"Program {pid} is in Computer Science, Data Science, or directly related technical computing field",
        parent=program_node,
        critical=True
    )

    # Field verification via URLs
    field_verify_leaf = evaluator.add_leaf(
        id=f"program_{pid}_field_verification",
        desc=f"Degree title/description supports that the program field is CS/DS/related",
        parent=field_group,
        critical=True
    )
    field_claim = (
        f"The program '{prog.degree_title or 'this program'}' is in Computer Science, Data Science, or a closely related "
        f"technical computing field (e.g., AI, ML, Software Engineering, Data Analytics, Computing)."
    )
    await evaluator.verify(
        claim=field_claim,
        node=field_verify_leaf,
        sources=prog.field.field_urls,
        additional_instruction="Use the cited program/degree page(s) to confirm the field. Allow reasonable synonyms."
    )

    # Field URL existence
    evaluator.add_custom_node(
        result=_exists_list(prog.field.field_urls),
        id=f"program_{pid}_field_url",
        desc=f"URL documenting the degree title/field",
        parent=field_group,
        critical=True
    )

    # Established University group
    estu_group = evaluator.add_parallel(
        id=f"program_{pid}_established_university",
        desc=f"Program {pid} is offered by an established university (not solely a corporate training program)",
        parent=program_node,
        critical=True
    )

    estu_verify_leaf = evaluator.add_leaf(
        id=f"program_{pid}_established_university_verification",
        desc=f"Evidence that the degree is awarded/offered by a university or equivalent degree-granting institution",
        parent=estu_group,
        critical=True
    )
    estu_claim = (
        f"The offering institution '{prog.university_name or 'the institution'}' is a university or degree‑granting institution."
    )
    await evaluator.verify(
        claim=estu_claim,
        node=estu_verify_leaf,
        sources=prog.institution_urls,
        additional_instruction="Verify on the official institution site or equivalent that it is a recognized university/degree-granting institution."
    )

    evaluator.add_custom_node(
        result=_exists_list(prog.institution_urls),
        id=f"program_{pid}_established_university_url",
        desc=f"URL documenting the offering institution as a university/degree-granting institution",
        parent=estu_group,
        critical=True
    )

    # Performance-based Admission group
    pba_group = evaluator.add_parallel(
        id=f"program_{pid}_performance_based_admission",
        desc=f"Program {pid} offers performance-based admission via pathway courses meeting the stated requirements",
        parent=program_node,
        critical=True
    )

    # Admission URL existence
    evaluator.add_custom_node(
        result=_exists_list(prog.admission.admission_urls),
        id=f"program_{pid}_admission_url",
        desc=f"URL documenting the performance-based admission pathway details (course count + grade threshold + no application/GRE/upfront degree verification)",
        parent=pba_group,
        critical=True
    )

    # Pathway exists
    pathway_exists_leaf = evaluator.add_leaf(
        id=f"program_{pid}_pathway_exists",
        desc=f"Program explicitly offers admission by completing designated pathway courses (performance-based admission option)",
        parent=pba_group,
        critical=True
    )
    pathway_exists_claim = (
        "This program offers a performance‑based admission pathway where admission can be earned by completing designated pathway courses."
    )
    await evaluator.verify(
        claim=pathway_exists_claim,
        node=pathway_exists_leaf,
        sources=prog.admission.admission_urls,
        additional_instruction="Confirm the pathway option is explicitly described on the cited pages."
    )

    # Pathway course count (2–3)
    course_count_leaf = evaluator.add_leaf(
        id=f"program_{pid}_pathway_course_count",
        desc=f"Pathway requires completion of 2–3 courses",
        parent=pba_group,
        critical=True
    )
    course_count_claim = "The performance‑based admission pathway requires completion of 2 to 3 courses."
    await evaluator.verify(
        claim=course_count_claim,
        node=course_count_leaf,
        sources=prog.admission.admission_urls,
        additional_instruction="Check the pathway section for the number of courses needed to qualify for admission."
    )

    # Grade threshold (≥3.0 GPA or grade B)
    grade_threshold_leaf = evaluator.add_leaf(
        id=f"program_{pid}_pathway_grade_threshold",
        desc=f"Pathway requires minimum cumulative GPA of 3.0 or minimum grade B (as stated)",
        parent=pba_group,
        critical=True
    )
    grade_threshold_claim = (
        "The pathway requires at least a 3.0 cumulative GPA or minimum grade B across the pathway courses."
    )
    await evaluator.verify(
        claim=grade_threshold_claim,
        node=grade_threshold_leaf,
        sources=prog.admission.admission_urls,
        additional_instruction="Accept phrasing like 'grade B or better' or 'minimum 3.0 GPA' as meeting the requirement."
    )

    # No traditional application/GRE/upfront bachelor's degree verification to start
    no_req_leaf = evaluator.add_leaf(
        id=f"program_{pid}_no_traditional_app_gre_degree_verification",
        desc=f"Explicitly states no traditional application, no GRE, and no upfront bachelor's degree verification required to start the pathway",
        parent=pba_group,
        critical=True
    )
    no_req_claim = (
        "To start the performance‑based pathway, no traditional application, no GRE, and no upfront bachelor's degree verification are required."
    )
    await evaluator.verify(
        claim=no_req_claim,
        node=no_req_leaf,
        sources=prog.admission.admission_urls,
        additional_instruction="The page should clearly state that students can begin pathway courses without submitting a traditional application, GRE, or upfront degree verification."
    )

    # Tuition group
    tuition_group = evaluator.add_parallel(
        id=f"program_{pid}_tuition",
        desc=f"Program {pid} total tuition is under $20,000 USD",
        parent=program_node,
        critical=True
    )

    tuition_leaf = evaluator.add_leaf(
        id=f"program_{pid}_total_tuition_amount",
        desc=f"States a total tuition amount and it is < $20,000 USD",
        parent=tuition_group,
        critical=True
    )
    tuition_claim = "The total program tuition is under $20,000 USD."
    await evaluator.verify(
        claim=tuition_claim,
        node=tuition_leaf,
        sources=prog.tuition.tuition_urls,
        additional_instruction=(
            "If the page lists per‑credit rates, combine with the required credits shown to compute an estimated total. "
            "If the computed or stated total is < $20,000, consider the claim supported."
        )
    )

    evaluator.add_custom_node(
        result=_exists_list(prog.tuition.tuition_urls),
        id=f"program_{pid}_tuition_url",
        desc=f"URL documenting total tuition cost",
        parent=tuition_group,
        critical=True
    )

    # Duration group
    duration_group = evaluator.add_parallel(
        id=f"program_{pid}_duration",
        desc=f"Program {pid} can be completed in 24 months or less at recommended full-time pace",
        parent=program_node,
        critical=True
    )

    duration_leaf = evaluator.add_leaf(
        id=f"program_{pid}_duration_verification",
        desc=f"States completion duration and it is ≤ 24 months at recommended pace",
        parent=duration_group,
        critical=True
    )
    duration_claim = "The program can be completed in 24 months or less at the recommended full‑time pace."
    await evaluator.verify(
        claim=duration_claim,
        node=duration_leaf,
        sources=prog.duration.duration_urls,
        additional_instruction="Use program pages that state typical completion timeframes for full‑time students."
    )

    evaluator.add_custom_node(
        result=_exists_list(prog.duration.duration_urls),
        id=f"program_{pid}_duration_url",
        desc=f"URL documenting completion duration",
        parent=duration_group,
        critical=True
    )

    # Delivery group
    delivery_group = evaluator.add_parallel(
        id=f"program_{pid}_delivery",
        desc=f"Program {pid} is 100% online with no mandatory on-campus requirements",
        parent=program_node,
        critical=True
    )

    delivery_leaf = evaluator.add_leaf(
        id=f"program_{pid}_online_no_residency_verification",
        desc=f"Explicitly states fully/100% online and no required campus attendance/residency",
        parent=delivery_group,
        critical=True
    )
    delivery_claim = "The program is delivered 100% online with no mandatory on‑campus attendance or residency requirements."
    await evaluator.verify(
        claim=delivery_claim,
        node=delivery_leaf,
        sources=prog.delivery.delivery_urls,
        additional_instruction="Confirm the official program site states 'fully online' and that residency/campus attendance is not required."
    )

    evaluator.add_custom_node(
        result=_exists_list(prog.delivery.delivery_urls),
        id=f"program_{pid}_delivery_url",
        desc=f"URL documenting the online/no-on-campus requirement",
        parent=delivery_group,
        critical=True
    )

    # Accreditation group
    accred_group = evaluator.add_parallel(
        id=f"program_{pid}_accreditation",
        desc=f"Program {pid} is accredited by a recognized regional or national accrediting body",
        parent=program_node,
        critical=True
    )

    evaluator.add_custom_node(
        result=_exists_str(prog.accreditation.accreditor_name),
        id=f"program_{pid}_accrediting_body_named",
        desc=f"Names the accrediting body for the institution/program",
        parent=accred_group,
        critical=True
    )

    accred_leaf = evaluator.add_leaf(
        id=f"program_{pid}_accreditation_url",
        desc=f"URL documenting accreditation",
        parent=accred_group,
        critical=True
    )
    accred_claim = (
        f"The institution '{prog.university_name or 'the institution'}' or this program is accredited by "
        f"{prog.accreditation.accreditor_name or 'a recognized accrediting body'}."
    )
    await evaluator.verify(
        claim=accred_claim,
        node=accred_leaf,
        sources=prog.accreditation.accreditation_urls,
        additional_instruction=(
            "Use official accreditation listings (institutional accreditation) or university pages that state accreditation. "
            "Accept well‑recognized regional bodies (e.g., HLC, SACSCOC, MSCHE, WSCUC, NECHE) or recognized national accreditors."
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
    # Initialize evaluator (root is non-critical by framework design)
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

    # Extract structured programs info
    extracted = await evaluator.extract(
        prompt=prompt_extract_programs(),
        template_class=ProgramsExtraction,
        extraction_name="programs_extraction"
    )

    programs: List[ProgramItem] = extracted.programs[:4]
    # Pad to exactly 4 entries if fewer were provided
    while len(programs) < 4:
        programs.append(ProgramItem())

    # Add custom info about extraction
    evaluator.add_custom_info(
        info={
            "extracted_program_count": len(extracted.programs),
            "used_program_count": 4,
            "used_universities": [p.university_name for p in programs],
            "used_degree_titles": [p.degree_title for p in programs],
        },
        info_type="extraction_stats",
        info_name="programs_stats"
    )

    # Build verification tree per program
    for idx, prog in enumerate(programs, start=0):
        await verify_program(evaluator, root, prog, idx)

    # Return summary
    return evaluator.get_summary()