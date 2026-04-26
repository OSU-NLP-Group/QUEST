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
TASK_ID = "edu_grad_programs_fall_2026"
TASK_DESCRIPTION = (
    "I am planning to pursue a graduate degree in an education-related field starting in Fall 2026. "
    "Please identify four different graduate programs at regionally accredited institutions in the United States that meet all of the following criteria:\n\n"
    "1. The program must be in an education-related field (such as educational leadership, curriculum and instruction, special education, higher education administration, school counseling, instructional design, elementary education, or similar education fields).\n\n"
    "2. The institution must be regionally accredited in the United States.\n\n"
    "3. The program must be currently accepting applications for Fall 2026 semester enrollment (as of March 20, 2026).\n\n"
    "4. The institution must offer graduate assistantship opportunities specifically for education graduate students.\n\n"
    "For each of the four programs, please provide:\n"
    "- The name of the institution\n"
    "- The specific graduate program name and degree type (e.g., M.Ed., M.A., Ed.S., Ph.D.)\n"
    "- Verification that it meets all four criteria listed above\n"
    "- A direct URL to the program page, graduate admissions page, or assistantship page that supports the provided information"
)

AS_OF_DATE_TEXT = "March 20, 2026"
AS_OF_DATE_ISO = "2026-03-20"

REGIONAL_ACCREDITORS = [
    "Higher Learning Commission",
    "HLC",
    "Middle States Commission on Higher Education",
    "MSCHE",
    "New England Commission of Higher Education",
    "NECHE",
    "Northwest Commission on Colleges and Universities",
    "NWCCU",
    "Southern Association of Colleges and Schools Commission on Colleges",
    "SACSCOC",
    "WASC Senior College and University Commission",
    "WSCUC",
]


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class ProgramExtract(BaseModel):
    institution: Optional[str] = None
    program_name: Optional[str] = None
    degree_type: Optional[str] = None
    program_url: Optional[str] = None
    admissions_url: Optional[str] = None
    assistantship_url: Optional[str] = None
    accreditation_url: Optional[str] = None
    extra_urls: List[str] = Field(default_factory=list)


class ProgramsExtraction(BaseModel):
    programs: List[ProgramExtract] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt builder                                                   #
# --------------------------------------------------------------------------- #
def prompt_extract_programs() -> str:
    return f"""
Extract up to FOUR distinct graduate programs in education-related fields as presented in the answer.

Return a JSON object with:
- programs: an array of objects, each with fields:
  - institution: The institution's full name (e.g., "University of X")
  - program_name: The specific program (e.g., "Educational Leadership")
  - degree_type: The degree type (e.g., "M.Ed.", "M.A.", "Ed.S.", "Ph.D.")
  - program_url: Direct URL to the program page (if provided)
  - admissions_url: Direct URL to graduate admissions or application instructions for this program (if provided)
  - assistantship_url: Direct URL that mentions graduate assistantships specifically for education graduate students (e.g., College/School of Education assistantships, departmental GA/TA/RA pages) (if provided)
  - accreditation_url: Direct URL proving regional accreditation (institutional accreditation page or accreditor directory entry) (if provided)
  - extra_urls: Any other relevant URLs cited for this program

GUIDELINES:
- Only extract information explicitly present in the answer.
- If a field is missing, set it to null (or [] for extra_urls).
- Prioritize URLs that directly support: the education field nature of the program, Fall 2026 application availability, assistantships for education students, and regional accreditation.
- Do NOT invent URLs.
"""


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _dedupe_urls(urls: List[Optional[str]]) -> List[str]:
    seen = set()
    out: List[str] = []
    for u in urls:
        if not u:
            continue
        s = u.strip()
        if not s:
            continue
        if s not in seen:
            out.append(s)
            seen.add(s)
    return out


def _program_display(program: ProgramExtract) -> str:
    inst = program.institution or "the institution"
    pname = program.program_name or "the program"
    dtype = program.degree_type or ""
    dtype_part = f" ({dtype})" if dtype else ""
    return f"{pname}{dtype_part} at {inst}"


# --------------------------------------------------------------------------- #
# Verification builders per program                                           #
# --------------------------------------------------------------------------- #
async def verify_single_program(
    evaluator: Evaluator,
    parent_node,
    program: ProgramExtract,
    index_zero_based: int,
) -> None:
    """
    Build and execute verification nodes for one program, following the rubric.
    """
    ordinal = index_zero_based + 1
    prog_node = evaluator.add_parallel(
        id=f"Program_{ordinal}",
        desc=f"{['First','Second','Third','Fourth'][index_zero_based]} graduate education program meeting all specified criteria",
        parent=parent_node,
        critical=False,  # Program-level node is non-critical to allow partial credit across programs
    )

    # Collect URL sources
    all_urls = _dedupe_urls([
        program.program_url,
        program.admissions_url,
        program.assistantship_url,
        program.accreditation_url,
        *program.extra_urls,
    ])
    prog_urls = _dedupe_urls([program.program_url, *program.extra_urls])
    adm_urls = _dedupe_urls([program.admissions_url, program.program_url, *program.extra_urls])
    ga_urls = _dedupe_urls([program.assistantship_url, program.program_url, *program.extra_urls])
    accr_urls = _dedupe_urls([program.accreditation_url, *program.extra_urls, program.program_url])

    # 1) U.S. Location (critical)
    node_us = evaluator.add_leaf(
        id=f"Program_{ordinal}_US_Location",
        desc="The program must be offered by a U.S. institution",
        parent=prog_node,
        critical=True,
    )
    claim_us = (
        f"The institution '{program.institution or ''}' is located in the United States (U.S.). "
        "Evidence should include a U.S. address, mention of a U.S. state, or explicit statement indicating the institution is in the U.S."
    )
    await evaluator.verify(
        claim=claim_us,
        node=node_us,
        sources=all_urls,
        additional_instruction=(
            "Confirm the institution is in the United States. Accept explicit address/state references, "
            "or clear indicators on official pages. If pages show a non-U.S. location, mark as not supported."
        ),
    )

    # 2) Education-related Field (critical)
    node_field = evaluator.add_leaf(
        id=f"Program_{ordinal}_Education_Field",
        desc="The program must be in an education-related field (e.g., educational leadership, curriculum and instruction, special education, counseling, instructional design)",
        parent=prog_node,
        critical=True,
    )
    claim_field = (
        f"The program '{program.program_name or ''}' ({program.degree_type or ''}) at '{program.institution or ''}' "
        "is an education-related graduate program (e.g., educational leadership, curriculum and instruction, "
        "special education, higher education administration, school counseling, instructional design, elementary education, or similar)."
    )
    await evaluator.verify(
        claim=claim_field,
        node=node_field,
        sources=prog_urls or all_urls,
        additional_instruction=(
            "Look for program titles and descriptions that clearly place the program within education or schools/colleges of education. "
            "Allow close variants and synonyms. If the program is clearly non-education (e.g., computer science), fail."
        ),
    )

    # 3) Accepting applications for Fall 2026 as of March 20, 2026 (critical)
    node_fall = evaluator.add_leaf(
        id=f"Program_{ordinal}_Fall_2026_Application",
        desc=f"The program must accept applications for Fall 2026 semester enrollment with a deadline that has not yet passed as of the current date ({AS_OF_DATE_TEXT})",
        parent=prog_node,
        critical=True,
    )
    claim_fall = (
        f"As of {AS_OF_DATE_TEXT} (ISO {AS_OF_DATE_ISO}), the program {_program_display(program)} is accepting applications "
        "for Fall 2026 (or an equivalent term label such as Autumn 2026), and the deadline for Fall 2026 has not yet passed by that date."
    )
    await evaluator.verify(
        claim=claim_fall,
        node=node_fall,
        sources=adm_urls or all_urls,
        additional_instruction=(
            f"Confirm that Fall 2026 (or Autumn 2026) is an available entry term and applications are open or deadlines are on/after {AS_OF_DATE_TEXT}. "
            "If the only listed deadlines for Fall 2026 are earlier than this date and marked closed, then the claim is not supported. "
            "Accept explicit notes of rolling admissions that remain open beyond {AS_OF_DATE_TEXT}."
        ),
    )

    # 4) Graduate assistantships specifically for education graduate students (critical)
    node_ga = evaluator.add_leaf(
        id=f"Program_{ordinal}_Assistantship_Available",
        desc="The institution must offer graduate assistantships for education graduate students",
        parent=prog_node,
        critical=True,
    )
    claim_ga = (
        "There are graduate assistantship opportunities specifically for education graduate students (e.g., College/School of Education GA/TA/RA roles, "
        "department-based assistantships in education programs) at the institution."
    )
    await evaluator.verify(
        claim=claim_ga,
        node=node_ga,
        sources=ga_urls or all_urls,
        additional_instruction=(
            "Prefer explicit references to assistantships offered by the College/School/Department of Education, or pages listing GA/TA/RA positions for education programs. "
            "General university-wide GA info is insufficient unless it explicitly includes education graduate students as eligible and commonly placed in education units."
        ),
    )

    # 5) Regional accreditation (critical)
    node_accr = evaluator.add_leaf(
        id=f"Program_{ordinal}_Regional_Accreditation",
        desc="The institution must be regionally accredited",
        parent=prog_node,
        critical=True,
    )
    accreditors_list = ", ".join(REGIONAL_ACCREDITORS)
    claim_accr = (
        f"The institution '{program.institution or ''}' holds regional accreditation by a recognized U.S. regional accreditor "
        f"(e.g., {accreditors_list})."
    )
    await evaluator.verify(
        claim=claim_accr,
        node=node_accr,
        sources=accr_urls or all_urls,
        additional_instruction=(
            "Accept evidence from the institution's accreditation page or the accreditor's official directory page. "
            "Recognized regional accreditors include: HLC, MSCHE, NECHE, NWCCU, SACSCOC, and WSCUC. "
            "Specialized/professional accreditations (e.g., CACREP, CAEP) do NOT satisfy this criterion on their own."
        ),
    )

    # 6) Valid supporting URL(s) provided (critical)
    node_url = evaluator.add_leaf(
        id=f"Program_{ordinal}_URL",
        desc="Provide a valid URL to the program page or graduate admissions page that confirms the above information",
        parent=prog_node,
        critical=True,
    )
    claim_url = (
        f"At least one of the provided URLs is an official page (e.g., program page, admissions page, assistantship page, or accreditor directory) "
        f"that is directly relevant to {_program_display(program)} and contains supporting information for at least one of the required criteria."
    )
    await evaluator.verify(
        claim=claim_url,
        node=node_url,
        sources=all_urls,
        additional_instruction=(
            "Verify that at least one URL is clearly relevant and from an authoritative source (e.g., .edu institutional page or the official accreditor site). "
            "The page should mention the program/admissions/assistantships/accreditation in a way that supports one or more required criteria."
        ),
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
    Evaluate an answer for the task:
    Identify four U.S. graduate programs in education that accept applications for Fall 2026 and offer graduate assistantships,
    ensuring the institutions are regionally accredited and U.S.-based, with supporting URLs.
    """
    # Initialize evaluator (root: parallel aggregation across programs)
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

    # Extract structured program information from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_programs(),
        template_class=ProgramsExtraction,
        extraction_name="programs_extraction"
    )

    # Always process exactly four programs (pad with empty placeholders if fewer)
    programs: List[ProgramExtract] = (extracted.programs or [])[:4]
    while len(programs) < 4:
        programs.append(ProgramExtract())

    # Add parent node per rubric
    parent = evaluator.add_parallel(
        id="Graduate_Programs",
        desc="Identify four U.S. graduate programs in education that accept applications for Fall 2026 semester and offer graduate assistantships",
        parent=root,
        critical=False
    )

    # Verify each program
    for idx in range(4):
        await verify_single_program(evaluator, parent, programs[idx], idx)

    # Optionally record evaluation configuration/context
    evaluator.add_custom_info(
        info={
            "as_of_date_text": AS_OF_DATE_TEXT,
            "as_of_date_iso": AS_OF_DATE_ISO,
            "recognized_regional_accreditors": REGIONAL_ACCREDITORS,
            "programs_count_extracted": len(extracted.programs or []),
        },
        info_type="context",
        info_name="evaluation_context"
    )

    # Return evaluation summary
    return evaluator.get_summary()