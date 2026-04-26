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
TASK_ID = "online_elem_ed_bachelors"
TASK_DESCRIPTION = (
    "I am researching affordable online pathways to become a licensed elementary school teacher. "
    "Please identify three different online bachelor's degree programs in elementary education that meet ALL of the following requirements:\n\n"
    "1. The program must be accredited by CAEP (Council for the Accreditation of Educator Preparation) for initial teacher preparation\n"
    "2. The institution must have regional accreditation from a recognized U.S. regional accrediting body\n"
    "3. The program coursework must be available 100% online\n"
    "4. The program must include required in-person clinical practice or student teaching components\n"
    "5. The program must explicitly prepare candidates for initial elementary teacher licensure\n"
    "6. The tuition cost must not exceed $4,000 per 6-month term (or equivalent per-credit hour basis, calculated as approximately $333 or less per credit hour for a typical 12-credit semester)\n"
    "7. The program must specify the elementary grade levels covered (e.g., K-6, K-8, or P-5)\n"
    "8. The program must include supervised student teaching experience of at least 12 weeks\n"
    "9. The program must be offered by a U.S.-based institution\n"
    "10. The program must clearly state admission requirements on its official website\n"
    "11. The program must specify which state(s) it prepares candidates for teacher licensure\n"
    "12. The program must be currently accepting applications for online enrollment\n\n"
    "For each of the three programs, provide:\n"
    "- The institution name\n"
    "- The specific program name/degree title\n"
    "- The current tuition cost (specify whether per term, per credit, or per year)\n"
    "- The grade levels the program prepares teachers for\n"
    "- A direct link to the official program page on the institution's website"
)


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class ProgramItem(BaseModel):
    institution: Optional[str] = None
    program_name: Optional[str] = None
    program_url: Optional[str] = None

    # Tuition info
    tuition_value: Optional[str] = None  # e.g., "$3,950", "$315/credit", "$7,800 per year"
    tuition_basis: Optional[str] = None  # e.g., "per 6-month term", "per credit", "per semester", "per year"
    tuition_url: Optional[str] = None

    # Delivery and structure
    grade_levels: Optional[str] = None  # e.g., "K-6", "P-5", "K-8"
    online_delivery_url: Optional[str] = None

    # Clinical and student teaching
    student_teaching_weeks: Optional[str] = None  # e.g., "12 weeks", "one semester", "16 weeks"
    student_teaching_url: Optional[str] = None

    # Accreditation
    caep_url: Optional[str] = None
    regional_accreditation_url: Optional[str] = None

    # Licensure
    licensure_states: Optional[str] = None  # free-text list or description
    licensure_states_url: Optional[str] = None

    # Admissions
    admissions_url: Optional[str] = None

    # Extra supporting URLs (official sources only if possible)
    additional_urls: List[str] = Field(default_factory=list)


class ProgramsExtraction(BaseModel):
    programs: List[ProgramItem] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_programs() -> str:
    return """
    Extract details for up to three distinct online bachelor's degree programs in elementary education that the answer proposes. 
    Extract the first three programs if more are mentioned. For each program, provide a JSON object with the following fields:

    - institution: Institution name (string)
    - program_name: Exact degree/program title (string)
    - program_url: Direct URL to the official program page on the institution website (string URL)
    - tuition_value: Tuition figure as written (string, keep currency symbol and units if present)
    - tuition_basis: The basis of the tuition (e.g., "per 6-month term", "per credit", "per semester", "per year")
    - tuition_url: URL where the tuition figure is stated if provided (string URL or null)
    - grade_levels: The specific elementary grades coverage (e.g., "K-6", "P-5", "K-8")
    - online_delivery_url: URL where the online/100% online delivery is stated if provided (string URL or null)
    - student_teaching_weeks: Duration of supervised student teaching as written (e.g., "12 weeks", "one semester", "16 weeks")
    - student_teaching_url: URL where student teaching/clinical requirement is stated if provided (string URL or null)
    - caep_url: URL evidencing CAEP accreditation (program-level or provider-level) if provided (string URL or null)
    - regional_accreditation_url: URL evidencing regional (institutional) accreditation if provided (string URL or null)
    - licensure_states: Text description of which state(s) the program prepares candidates for (string, concise; null if not stated)
    - licensure_states_url: URL where the licensure states are specified if provided (string URL or null)
    - admissions_url: URL where admission requirements for the program are stated if provided (string URL or null)
    - additional_urls: Array of any other official URLs from the institution or accreditor that are cited in the answer and relevant as supporting evidence

    RULES:
    - Only extract URLs explicitly present in the answer. Do not invent or infer URLs.
    - Prefer official institution or accreditor domains; avoid third-party listings unless the answer explicitly cites them.
    - If a field is missing, set it to null (or [] for arrays).
    - Do not normalize numbers; keep the original text as written for tuition_value and student_teaching_weeks.

    Output a JSON object with a single field:
    {
      "programs": [ ... up to three ProgramItem objects ... ]
    }
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def _combine_sources(*args: Optional[Any]) -> List[str]:
    """Combine strings or lists of URLs into a de-duplicated ordered list."""
    seen = set()
    ordered: List[str] = []
    for x in args:
        if not x:
            continue
        if isinstance(x, str):
            url = x.strip()
            if url and url not in seen:
                seen.add(url)
                ordered.append(url)
        elif isinstance(x, list):
            for u in x:
                if isinstance(u, str):
                    uu = u.strip()
                    if uu and uu not in seen:
                        seen.add(uu)
                        ordered.append(uu)
    return ordered


def _program_display_name(p: ProgramItem, idx: int) -> str:
    base = f"Program #{idx}"
    if p.program_name and p.institution:
        return f"{p.program_name} at {p.institution} ({base})"
    if p.program_name:
        return f"{p.program_name} ({base})"
    if p.institution:
        return f"{p.institution} ({base})"
    return base


# --------------------------------------------------------------------------- #
# Verification logic per program                                              #
# --------------------------------------------------------------------------- #
async def verify_program(
    evaluator: Evaluator,
    parent_node,
    program: ProgramItem,
    program_idx: int,
) -> None:
    """
    Build verification nodes and execute checks for one program.
    program_idx is 1-based (1, 2, 3) to match rubric IDs.
    """
    idx = program_idx
    prog_label = _program_display_name(program, idx)

    # Create the parent node for this program (non-critical to allow partial scoring across the 3 programs)
    prog_node = evaluator.add_parallel(
        id=f"program_{idx}",
        desc=[
            "First qualifying elementary education bachelor's program",
            "Second qualifying elementary education bachelor's program",
            "Third qualifying elementary education bachelor's program",
        ][idx - 1],
        parent=parent_node,
        critical=False
    )

    # Critical: URL provided (custom existence check)
    url_exists = isinstance(program.program_url, str) and program.program_url.strip() != ""
    evaluator.add_custom_node(
        result=url_exists,
        id=f"program_{idx}_url_reference",
        desc="A direct URL to the official program page is provided",
        parent=prog_node,
        critical=True
    )

    # Gather common sources
    common_sources = _combine_sources(
        program.program_url,
        program.additional_urls
    )

    # 1) CAEP accreditation (critical)
    caep_node = evaluator.add_leaf(
        id=f"program_{idx}_caep_accreditation",
        desc="The program is accredited by CAEP for initial teacher preparation",
        parent=prog_node,
        critical=True
    )
    caep_claim = (
        f"The program {_program_display_name(program, idx)} is accredited by CAEP (Council for the Accreditation of "
        f"Educator Preparation) for initial teacher preparation."
    )
    await evaluator.verify(
        claim=caep_claim,
        node=caep_node,
        sources=_combine_sources(program.caep_url, common_sources),
        additional_instruction="Accept evidence from CAEP's directory or the institution's/accreditor's site. "
                               "Look for explicit mention of CAEP accreditation. "
                               "It must be for initial teacher preparation (initial licensure/certification), "
                               "not only advanced programs."
    )

    # 2) Regional accreditation (critical)
    regional_node = evaluator.add_leaf(
        id=f"program_{idx}_regional_accreditation",
        desc="The institution has regional accreditation from a recognized U.S. regional accrediting body",
        parent=prog_node,
        critical=True
    )
    regional_claim = (
        f"The institution for {_program_display_name(program, idx)} holds regional (institutional) accreditation from "
        f"a recognized U.S. regional accrediting body (e.g., HLC, MSCHE, SACSCOC, NECHE, NWCCU, WSCUC)."
    )
    await evaluator.verify(
        claim=regional_claim,
        node=regional_node,
        sources=_combine_sources(program.regional_accreditation_url, common_sources),
        additional_instruction="Verify that the institution lists a recognized U.S. regional accreditor "
                               "(HLC, MSCHE, SACSCOC, NECHE, NWCCU, WSCUC) on an official page or the accreditor's site."
    )

    # 3) 100% online delivery (critical)
    online_node = evaluator.add_leaf(
        id=f"program_{idx}_online_delivery",
        desc="The program coursework is available 100% online",
        parent=prog_node,
        critical=True
    )
    online_claim = (
        f"The coursework for {_program_display_name(program, idx)} is offered 100% online (no required on-campus classes), "
        f"though in-person clinical placements may be required."
    )
    await evaluator.verify(
        claim=online_claim,
        node=online_node,
        sources=_combine_sources(program.online_delivery_url, common_sources),
        additional_instruction="Accept phrasing like 'fully online', '100% online', or clear statements that coursework can "
                               "be completed entirely online."
    )

    # 4) Clinical/student teaching components required (critical)
    clinical_node = evaluator.add_leaf(
        id=f"program_{idx}_clinical_component",
        desc="The program includes required in-person clinical practice or student teaching components",
        parent=prog_node,
        critical=True
    )
    clinical_claim = (
        f"The program {_program_display_name(program, idx)} includes required in-person clinical experiences or "
        f"student teaching components."
    )
    await evaluator.verify(
        claim=clinical_claim,
        node=clinical_node,
        sources=_combine_sources(program.student_teaching_url, common_sources),
        additional_instruction="Look for 'field experiences', 'clinical practice', 'practicum', or 'student teaching' "
                               "as required components."
    )

    # 5) Licensure preparation (critical)
    licprep_node = evaluator.add_leaf(
        id=f"program_{idx}_licensure_preparation",
        desc="The program explicitly prepares candidates for initial elementary teacher licensure",
        parent=prog_node,
        critical=True
    )
    licprep_claim = (
        f"The program {_program_display_name(program, idx)} explicitly prepares candidates for initial elementary "
        f"teacher licensure/certification."
    )
    await evaluator.verify(
        claim=licprep_claim,
        node=licprep_node,
        sources=_combine_sources(program.licensure_states_url, common_sources),
        additional_instruction="Accept terms like 'initial licensure', 'initial certification', 'prepares for teacher licensure'."
    )

    # 6) Tuition cost threshold (critical)
    tuition_node = evaluator.add_leaf(
        id=f"program_{idx}_tuition_cost",
        desc="The tuition cost does not exceed $4,000 per 6-month term or equivalent per-credit basis",
        parent=prog_node,
        critical=True
    )
    tuition_descr = ""
    if program.tuition_value and program.tuition_basis:
        tuition_descr = f" The page indicates tuition is {program.tuition_value} {program.tuition_basis}."
    tuition_claim = (
        f"The program's tuition pricing satisfies the affordability requirement: it does not exceed $4,000 per 6-month term "
        f"OR is equivalent to approximately $333 or less per credit hour for a typical 12-credit semester."
        f"{tuition_descr}"
    )
    await evaluator.verify(
        claim=tuition_claim,
        node=tuition_node,
        sources=_combine_sources(program.tuition_url, common_sources),
        additional_instruction="If tuition is quoted per credit, verify that it is $333/credit or less. "
                               "If per 6‑month term, verify it is $4,000 or less. "
                               "If per semester/year, assess equivalence (e.g., <= $8,000 per year roughly corresponds to two 6‑month terms). "
                               "Use only values explicitly stated on the provided official pages."
    )

    # 7) Grade levels specified (critical)
    grade_node = evaluator.add_leaf(
        id=f"program_{idx}_grade_levels",
        desc="The program specifies the elementary grade levels covered",
        parent=prog_node,
        critical=True
    )
    if program.grade_levels:
        grade_claim = (
            f"The program page specifies the grade levels for elementary education as '{program.grade_levels}'."
        )
    else:
        grade_claim = "The program page specifies the elementary grade levels covered (e.g., K-6, K-8, P-5)."
    await evaluator.verify(
        claim=grade_claim,
        node=grade_node,
        sources=common_sources,
        additional_instruction="Look for explicit grade range notation such as K-6, K-8, P-5, 1-6, etc."
    )

    # 8) Student teaching duration >= 12 weeks (critical)
    duration_node = evaluator.add_leaf(
        id=f"program_{idx}_student_teaching_duration",
        desc="The program includes supervised student teaching experience of at least 12 weeks",
        parent=prog_node,
        critical=True
    )
    if program.student_teaching_weeks:
        dur_claim = (
            f"The supervised student teaching duration is at least 12 weeks (page indicates '{program.student_teaching_weeks}')."
        )
    else:
        dur_claim = "The supervised student teaching duration is at least 12 weeks (approximately one academic semester)."
    await evaluator.verify(
        claim=dur_claim,
        node=duration_node,
        sources=_combine_sources(program.student_teaching_url, common_sources),
        additional_instruction="Accept phrasing like '12 weeks' or 'one semester' (typically 12–16 weeks) as meeting the threshold."
    )

    # 9) U.S.-based institution (critical)
    us_node = evaluator.add_leaf(
        id=f"program_{idx}_us_institution",
        desc="The program is offered by a U.S.-based institution",
        parent=prog_node,
        critical=True
    )
    us_claim = (
        f"The institution offering {_program_display_name(program, idx)} is based in the United States."
    )
    await evaluator.verify(
        claim=us_claim,
        node=us_node,
        sources=common_sources,
        additional_instruction="Look for a U.S. address, a U.S. state name/abbreviation, or explicit reference to being in the United States on the official site."
    )

    # 10) Admission requirements clearly stated (critical)
    adm_node = evaluator.add_leaf(
        id=f"program_{idx}_admission_requirements",
        desc="The program clearly states admission requirements",
        parent=prog_node,
        critical=True
    )
    adm_claim = (
        f"The official program or admissions page clearly states admission requirements for {_program_display_name(program, idx)}."
    )
    await evaluator.verify(
        claim=adm_claim,
        node=adm_node,
        sources=_combine_sources(program.admissions_url, common_sources),
        additional_instruction="Look for an 'Admissions' or 'Requirements' section with explicit criteria (e.g., GPA, test scores, prerequisites, application materials)."
    )

    # 11) States for licensure specified (critical)
    states_node = evaluator.add_leaf(
        id=f"program_{idx}_state_licensure",
        desc="The program specifies which state(s) it prepares candidates for teacher licensure",
        parent=prog_node,
        critical=True
    )
    if program.licensure_states:
        states_claim = (
            f"The program specifies the state(s) for which it prepares candidates for teacher licensure: {program.licensure_states}."
        )
    else:
        states_claim = (
            "The program specifies which U.S. state(s) it prepares candidates for teacher licensure (e.g., an authorization or disclosures page listing states)."
        )
    await evaluator.verify(
        claim=states_claim,
        node=states_node,
        sources=_combine_sources(program.licensure_states_url, common_sources),
        additional_instruction="Look for a 'state authorization' or 'licensure disclosures' page that lists states for which the program meets initial licensure requirements (or notes exceptions)."
    )

    # 12) Currently accepting applications (critical)
    open_node = evaluator.add_leaf(
        id=f"program_{idx}_enrollment_availability",
        desc="The program is currently accepting applications for online enrollment",
        parent=prog_node,
        critical=True
    )
    open_claim = (
        f"The program {_program_display_name(program, idx)} is currently accepting applications for online enrollment."
    )
    await evaluator.verify(
        claim=open_claim,
        node=open_node,
        sources=common_sources,
        additional_instruction="Accept indicators like 'Apply Now', 'Now accepting applications', upcoming start dates, "
                               "or an active application portal for the online program."
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
    """
    Evaluate an answer for the task of identifying three qualifying online elementary education bachelor's programs.
    """
    evaluator = Evaluator()

    # Note: Root set to non-critical to allow partial credit across programs and to comply with framework constraint
    # (critical parent cannot have non-critical children). The JSON marks root critical, but we relax it for validity.
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

    # Extract structured program info from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_programs(),
        template_class=ProgramsExtraction,
        extraction_name="programs_extraction"
    )

    # Ensure exactly three program slots by padding with empty items if necessary
    programs: List[ProgramItem] = list(extracted.programs[:3])
    while len(programs) < 3:
        programs.append(ProgramItem())

    # Build verification for each of the three programs
    for i in range(3):
        await verify_program(
            evaluator=evaluator,
            parent_node=root,
            program=programs[i],
            program_idx=i + 1
        )

    # Optionally record custom info with brief summary of extracted program names/urls
    summary_info = []
    for i, p in enumerate(programs, start=1):
        summary_info.append({
            "index": i,
            "institution": p.institution,
            "program_name": p.program_name,
            "program_url": p.program_url,
            "tuition_value": p.tuition_value,
            "tuition_basis": p.tuition_basis,
            "grade_levels": p.grade_levels
        })
    evaluator.add_custom_info({"program_summaries": summary_info}, info_type="extraction_summary")

    return evaluator.get_summary()