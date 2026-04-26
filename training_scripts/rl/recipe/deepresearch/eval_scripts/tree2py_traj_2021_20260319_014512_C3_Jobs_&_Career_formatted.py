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
TASK_ID = "va_superintendent_2024_career_pathway"
TASK_DESCRIPTION = """
Identify the name of a superintendent who meets all of the following criteria as of December 31, 2024: 
(1) Currently serves as a superintendent of a public school district in Virginia, 
(2) Was named the 2024 Superintendent of the Year by their state association, 
(3) Earned their bachelor's degree, master's degree, and doctorate degree all from the same university, 
(4) The university where they earned all three degrees offers an Ed.D. in Administration and Supervision (or an equivalent doctoral program in educational leadership or educational administration), 
(5) The university's doctoral program is accredited by a regional accrediting body, 
(6) The doctoral program has a typical completion timeline of 3-4 years for full-time students. 
Provide the superintendent's name, and include reference URLs documenting: 
(a) their 2024 award recognition, 
(b) their educational background, 
(c) the university's doctoral program offering, and 
(d) the program's accreditation and completion timeline.
"""

AS_OF_DATE = "December 31, 2024"


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class SuperintendentExtraction(BaseModel):
    # Person and role
    name: Optional[str] = None
    district: Optional[str] = None
    state: Optional[str] = None

    # Recognition/award
    recognition_title: Optional[str] = None
    recognition_awarder: Optional[str] = None
    recognition_year: Optional[str] = None
    recognition_urls: List[str] = Field(default_factory=list)

    # Position/supporting URLs (optional, if provided in answer)
    position_urls: List[str] = Field(default_factory=list)

    # Education - institutions for each degree
    bachelor_institution: Optional[str] = None
    master_institution: Optional[str] = None
    doctor_institution: Optional[str] = None
    education_urls: List[str] = Field(default_factory=list)

    # Program info at the (same) university
    doctoral_program_name: Optional[str] = None
    program_offering_urls: List[str] = Field(default_factory=list)

    # Program accreditation + duration details
    accrediting_body: Optional[str] = None
    typical_completion_years: Optional[str] = None
    program_details_urls: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_superintendent() -> str:
    return f"""
    From the provided answer, extract details about the identified superintendent and all required source URLs.

    Return the following fields exactly as stated in the answer (use null for any missing field):
    - name: Full name of the superintendent.
    - district: The school district/division they lead (if stated).
    - state: The U.S. state of the district (if stated).
    - recognition_title: The title of the 2024 award (e.g., "2024 Superintendent of the Year").
    - recognition_awarder: The awarding body (e.g., state superintendents association).
    - recognition_year: The award year (should be 2024 if stated).
    - recognition_urls: All URLs that document the 2024 award recognition.
    - position_urls: URLs (if any) that document the person's current superintendent role in Virginia (e.g., district bio page, press release).
    - bachelor_institution: University for the bachelor's degree.
    - master_institution: University for the master's degree.
    - doctor_institution: University for the doctorate (Ed.D./Ph.D.).
    - education_urls: All URLs that document the individual's educational background and institutions for each degree.
    - doctoral_program_name: Name of the relevant doctoral program at the university (e.g., "Ed.D. in Administration and Supervision" or an equivalent in educational leadership/administration).
    - program_offering_urls: Official university URLs that document the doctoral program offering (program pages, catalogs).
    - accrediting_body: Regional accreditor name if stated (e.g., SACSCOC, HLC, MSCHE, NECHE, NWCCU, WSCUC).
    - typical_completion_years: The textual description of typical full-time completion timeline if given (e.g., "3 years", "3-4 years").
    - program_details_urls: Official URLs that document accreditation and/or typical completion timeline (may include separate accreditation page and program page).

    Notes:
    - Extract only URLs explicitly present in the answer text; do not infer or invent new URLs.
    - If multiple URLs are provided for a category, include them all.
    - Do not conflate institutional accreditation with programmatic accreditation; for this task, institutional (regional) accreditation is acceptable to satisfy the accreditation requirement.
    - The same URL can appear in multiple lists if the answer used it for multiple purposes.
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _non_empty_urls(urls: Optional[List[str]]) -> List[str]:
    if not urls:
        return []
    return [u for u in urls if isinstance(u, str) and u.strip()]


def _first_non_empty_list(*lists: List[str]) -> List[str]:
    for lst in lists:
        if lst:
            return lst
    return []


def _choose_institution_for_program(ex: SuperintendentExtraction) -> Optional[str]:
    # Prefer doctoral institution; else master's; else bachelor's
    for inst in [ex.doctor_institution, ex.master_institution, ex.bachelor_institution]:
        if inst and inst.strip():
            return inst.strip()
    return None


def _all_three_same(ex: SuperintendentExtraction) -> Optional[str]:
    vals = [v.strip() for v in [ex.bachelor_institution, ex.master_institution, ex.doctor_institution] if v and v.strip()]
    if len(vals) < 3:
        return None
    norm = [v.lower().strip() for v in vals]
    if norm[0] == norm[1] == norm[2]:
        return vals[0]
    return None


# --------------------------------------------------------------------------- #
# Verification tree construction and checks                                   #
# --------------------------------------------------------------------------- #
async def build_and_verify_tree(evaluator: Evaluator, root, ex: SuperintendentExtraction) -> None:
    # Top-level critical sequential node (matches rubric root)
    task_node = evaluator.add_sequential(
        id="career_pathway_research_task",
        desc="Complete research task identifying a superintendent meeting specific career and educational criteria",
        parent=root,
        critical=True
    )

    # Superintendent Identification (critical sequential)
    sup_node = evaluator.add_sequential(
        id="superintendent_identification",
        desc="Identify the superintendent meeting geographic and recognition criteria",
        parent=task_node,
        critical=True
    )

    # 1) Current Position and Recognition (critical parallel)
    pos_rec_node = evaluator.add_parallel(
        id="current_position_and_recognition",
        desc="Verify the superintendent's current position and 2024 award recognition",
        parent=sup_node,
        critical=True
    )

    # Prepare sources
    recognition_urls = _non_empty_urls(ex.recognition_urls)
    position_urls = _non_empty_urls(ex.position_urls)
    education_urls = _non_empty_urls(ex.education_urls)
    program_offering_urls = _non_empty_urls(ex.program_offering_urls)
    program_details_urls = _non_empty_urls(ex.program_details_urls)

    # a) Virginia Superintendent Position (leaf, critical)
    leaf_va_pos = evaluator.add_leaf(
        id="virginia_superintendent_position",
        desc="The identified individual currently serves as a superintendent in Virginia",
        parent=pos_rec_node,
        critical=True
    )
    va_pos_sources = _first_non_empty_list(position_urls, recognition_urls, education_urls, program_offering_urls, program_details_urls)
    person = ex.name or "the identified individual"
    district_txt = f" of {ex.district}" if ex.district else ""
    claim_va_pos = f"As of {AS_OF_DATE}, {person} currently serves as superintendent{district_txt} in Virginia."
    add_ins_va = (
        "Verify that the person is a CURRENT superintendent (not former/interim unless stated as current), "
        "for a Virginia public school district/division. Accept reasonable titles like 'Division Superintendent'. "
        "Evidence should be from the provided URL(s) and need not explicitly mention the exact date if the page reasonably implies currency in 2024."
    )

    # b) State Award 2024 (leaf, critical)
    leaf_award = evaluator.add_leaf(
        id="state_award_2024",
        desc="The individual was named 2024 Superintendent of the Year by their state association",
        parent=pos_rec_node,
        critical=True
    )
    awarder = ex.recognition_awarder or "their state association"
    claim_award = f"{person} was named 2024 Superintendent of the Year by {awarder}."
    add_ins_award = (
        "Confirm the award year is 2024 and the awarding body is the person's state-level school superintendents association "
        "(e.g., Virginia Association of School Superintendents for a Virginia superintendent). "
        "The page should clearly indicate 'Superintendent of the Year' and the year 2024."
    )

    # c) Recognition Reference URL existence (custom, critical)
    evaluator.add_custom_node(
        result=len(recognition_urls) > 0,
        id="recognition_reference_url",
        desc="Provide a reference URL documenting the 2024 award recognition",
        parent=pos_rec_node,
        critical=True
    )

    # Batch verify the two factual leaves under the parallel node
    await evaluator.batch_verify([
        (claim_va_pos, va_pos_sources, leaf_va_pos, add_ins_va),
        (claim_award, recognition_urls, leaf_award, add_ins_award),
    ])

    # 2) Educational Background Verification (critical sequential)
    edu_node = evaluator.add_sequential(
        id="educational_background_verification",
        desc="Verify the superintendent's educational credentials and institutional affiliation",
        parent=sup_node,
        critical=True
    )

    # 2.1) Degree Institutional Consistency (critical parallel)
    degree_consistency = evaluator.add_parallel(
        id="degree_institutional_consistency",
        desc="Verify all degrees are from the same institution",
        parent=edu_node,
        critical=True
    )

    # a) Three degrees same institution (leaf, critical)
    leaf_same_inst = evaluator.add_leaf(
        id="three_degrees_same_institution",
        desc="The individual earned bachelor's, master's, and doctorate degrees all from the same institution",
        parent=degree_consistency,
        critical=True
    )
    same_inst_name = _all_three_same(ex)
    if same_inst_name:
        claim_same = f"{person} earned their bachelor's, master's, and doctorate degrees all from {same_inst_name}."
    else:
        claim_same = f"{person} earned their bachelor's, master's, and doctorate degrees all from the same university."
    add_ins_same = (
        "Verify from the provided URL(s) that the individual's bachelor's, master's, and doctoral degrees are all from the same university. "
        "It's acceptable if this information is presented across one or more official bios, press releases, or district/organization pages, "
        "as long as, taken together, they confirm the three degrees are from the same institution."
    )

    # b) Educational background reference URL existence (custom, critical)
    evaluator.add_custom_node(
        result=len(education_urls) > 0,
        id="educational_background_reference_url",
        desc="Provide a reference URL documenting the individual's educational background",
        parent=degree_consistency,
        critical=True
    )

    await evaluator.verify(
        claim=claim_same,
        node=leaf_same_inst,
        sources=education_urls,
        additional_instruction=add_ins_same
    )

    # 2.2) Institution Program Analysis (critical sequential)
    program_analysis = evaluator.add_sequential(
        id="institution_program_analysis",
        desc="Verify the institution's doctoral program offerings and characteristics",
        parent=edu_node,
        critical=True
    )

    # 2.2.1) Doctoral Program Availability (critical parallel)
    program_availability = evaluator.add_parallel(
        id="doctoral_program_availability",
        desc="Verify the institution offers the relevant doctoral program",
        parent=program_analysis,
        critical=True
    )

    # a) EdD Administration Program Offered or equivalent (leaf, critical)
    leaf_program_offered = evaluator.add_leaf(
        id="edd_administration_program_offered",
        desc="The institution offers an Ed.D. in Administration and Supervision or an equivalent doctoral program in educational leadership/administration",
        parent=program_availability,
        critical=True
    )
    inst_for_program = _choose_institution_for_program(ex) or "the same university where all three degrees were earned"
    prog_name = ex.doctoral_program_name or "a doctoral program in educational leadership/administration"
    claim_prog_offered = f"{inst_for_program} offers {prog_name}, which qualifies as an Ed.D. in Administration and Supervision or an equivalent doctoral program in educational leadership/administration."
    add_ins_prog = (
        "Accept official pages (program page, catalog, graduate school page) that clearly indicate the university offers an Ed.D. in Administration and Supervision "
        "OR an equivalent doctoral program in educational leadership/administration (e.g., 'Ed.D. in Educational Leadership', 'Ed.D. in Educational Administration', or a Ph.D. in Educational Leadership/Administration). "
        "The focus should be clearly on educational leadership/administration."
    )

    # b) Program Offering Reference URL existence (custom, critical)
    evaluator.add_custom_node(
        result=len(program_offering_urls) > 0,
        id="program_offering_reference_url",
        desc="Provide a reference URL from the institution documenting the doctoral program offering",
        parent=program_availability,
        critical=True
    )

    await evaluator.verify(
        claim=claim_prog_offered,
        node=leaf_program_offered,
        sources=program_offering_urls,
        additional_instruction=add_ins_prog
    )

    # 2.2.2) Program Accreditation and Duration (critical parallel)
    accred_duration = evaluator.add_parallel(
        id="program_accreditation_and_duration",
        desc="Verify the program's accreditation status and typical completion timeline",
        parent=program_analysis,
        critical=True
    )

    # a) Regional Accreditation (leaf, critical)
    leaf_accred = evaluator.add_leaf(
        id="regional_accreditation",
        desc="The institution's doctoral program is accredited by a regional accrediting body (e.g., Higher Learning Commission, SACSCOC, Middle States, etc.)",
        parent=accred_duration,
        critical=True
    )
    if ex.accrediting_body:
        claim_accred = f"{inst_for_program} is accredited by {ex.accrediting_body}, a US regional accrediting body, and the doctoral program is covered by this institutional accreditation."
    else:
        claim_accred = f"{inst_for_program} is accredited by a US regional accrediting body (e.g., SACSCOC, HLC, MSCHE, NECHE, NWCCU, or WSCUC), which covers the doctoral program."
    add_ins_accred = (
        "Institutional (regional) accreditation suffices for this criterion; it does not need to be program-specific. "
        "Verify via an official university accreditation page, the accreditor's directory/listing page, or an official catalog statement."
    )

    # b) Standard Completion Timeline (leaf, critical)
    leaf_timeline = evaluator.add_leaf(
        id="standard_completion_timeline",
        desc="The doctoral program's typical completion time for full-time students is 3-4 years",
        parent=accred_duration,
        critical=True
    )
    claim_timeline = (
        f"The typical full-time completion timeline for the doctoral program at {inst_for_program} is between 3 and 4 years."
    )
    add_ins_timeline = (
        "Accept statements like '3 years', 'approximately 3 years', '3-4 years', or clearly equivalent timelines for full-time students. "
        "If multiple timelines are given, prefer full-time standard paths; reasonable equivalence (e.g., ~3 years) should count as within the 3-4 year window."
    )

    # c) Program Details Reference URL existence (custom, critical)
    evaluator.add_custom_node(
        result=len(program_details_urls) > 0,
        id="program_details_reference_url",
        desc="Provide a reference URL documenting program accreditation and completion timeline",
        parent=accred_duration,
        critical=True
    )

    # Batch verify accreditation and timeline using provided program details URLs
    await evaluator.batch_verify([
        (claim_accred, program_details_urls, leaf_accred, add_ins_accred),
        (claim_timeline, program_details_urls, leaf_timeline, add_ins_timeline),
    ])


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
    Evaluate an answer for the Virginia superintendent 2024 career pathway research task.
    """
    # Initialize evaluator (use sequential at root to align with rubric gating)
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.SEQUENTIAL,
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
    extracted = await evaluator.extract(
        prompt=prompt_extract_superintendent(),
        template_class=SuperintendentExtraction,
        extraction_name="superintendent_extraction",
    )

    # Record as-of date and minimal context for debugging
    evaluator.add_custom_info(
        info={"as_of_date": AS_OF_DATE},
        info_type="metadata",
        info_name="evaluation_context"
    )

    # Build verification tree and run checks
    await build_and_verify_tree(evaluator, root, extracted)

    # Return structured evaluation summary
    return evaluator.get_summary()