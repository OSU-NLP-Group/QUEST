import asyncio
import logging
from typing import Any, List, Optional, Dict

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "principal_cert_search"
TASK_DESCRIPTION = (
    "I am a certified teacher with three years of classroom experience, currently working full-time, and interested in pursuing a career transition to become a school principal or educational administrator. "
    "I am looking for principal certification or educational leadership programs that would allow me to continue working while completing my studies.\n\n"
    "Please identify four principal certification or educational leadership master's programs from accredited universities in the United States that meet the following requirements:\n\n"
    "1. The program must be offered fully online or in a hybrid format that accommodates working professionals\n"
    "2. The program must be from a regionally accredited university\n"
    "3. The program must lead to principal certification or administrative services certification in a specific U.S. state\n"
    "4. For each program, provide the following information:\n"
    "   - University name and program title\n"
    "   - State where the certification is valid\n"
    "   - Program format (fully online or hybrid)\n"
    "   - Program duration (estimated time to completion)\n"
    "   - Program cost or tuition information\n"
    "   - A direct URL link to the official program page\n\n"
    "The programs can be from different states, and can be either certification-only programs (for those who already have a master's degree) or master's degree programs that include certification."
)


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class ProgramItem(BaseModel):
    university: Optional[str] = None
    program_title: Optional[str] = None
    state: Optional[str] = None  # state where certification is valid
    format: Optional[str] = None  # "fully online" or "hybrid" or similar wording
    duration: Optional[str] = None  # e.g., "12 months", "2 years", "12–24 months"
    cost: Optional[str] = None  # e.g., "$15,000 total", "$650/credit", "Tuition varies"
    program_url: Optional[str] = None  # official program page
    accreditation_url: Optional[str] = None  # university accreditation page if provided
    additional_urls: List[str] = Field(default_factory=list)  # any other cited URL(s)
    certification_type: Optional[str] = None  # e.g., "Principal Certification", "Administrative Services Credential"


class ProgramsExtraction(BaseModel):
    programs: List[ProgramItem] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_programs() -> str:
    return (
        "Extract up to the first four principal certification or educational leadership programs mentioned in the answer. "
        "Only include programs from accredited U.S. universities that the answer claims meet the requirements. "
        "For each program, return an object with the following fields:\n"
        "• university: University name\n"
        "• program_title: Program title (e.g., 'MEd Educational Leadership – Principal Certification')\n"
        "• state: The U.S. state where the resulting certification/credential is valid (e.g., 'Texas', 'California')\n"
        "• format: The format as stated (e.g., 'fully online', 'online', 'hybrid', 'online with occasional on-campus')\n"
        "• duration: The time-to-completion as stated (e.g., '12 months', '18 months', '2 years')\n"
        "• cost: Tuition or program cost info as stated (e.g., '$650/credit', 'Total cost ~$18,000')\n"
        "• program_url: A direct URL to the official program page; if not present, return null\n"
        "• accreditation_url: A URL to the university’s accreditation page if cited; if not present, return null\n"
        "• additional_urls: Any other relevant URLs explicitly cited for this program (exclude duplicates); may be empty\n"
        "• certification_type: The certification name/type (e.g., 'Principal Certification', 'Administrative Services Credential'); if ambiguous or absent, return null\n\n"
        "IMPORTANT:\n"
        "- Extract exactly what is written in the answer; do not invent or normalize terms.\n"
        "- Only include URLs that are explicitly present; include full URLs with protocol.\n"
        "- If any field is missing for a program, set it to null (or empty array for additional_urls).\n"
        "- Return a JSON object with a 'programs' array of up to four items."
    )


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def program_sources(prog: ProgramItem) -> List[str]:
    """Aggregate unique sources for a program for verification."""
    urls = []
    if prog.program_url and prog.program_url.strip():
        urls.append(prog.program_url.strip())
    if prog.accreditation_url and prog.accreditation_url.strip():
        urls.append(prog.accreditation_url.strip())
    for u in prog.additional_urls:
        if isinstance(u, str) and u.strip():
            urls.append(u.strip())
    # Deduplicate while preserving order
    seen = set()
    unique = []
    for u in urls:
        if u not in seen:
            unique.append(u)
            seen.add(u)
    return unique


def make_program_display_name(prog: ProgramItem) -> str:
    uni = prog.university or "Unknown University"
    title = prog.program_title or "Unknown Program"
    return f"{uni} — {title}"


# --------------------------------------------------------------------------- #
# Verification for one program                                                #
# --------------------------------------------------------------------------- #
async def verify_single_program(
    evaluator: Evaluator,
    parent_node,
    prog: ProgramItem,
    index: int,
) -> None:
    """
    Build verification sub-tree for one program and run checks.
    """
    # Program container node (parallel; non-critical to allow partial credit across programs)
    program_node = evaluator.add_parallel(
        id=f"program_{index + 1}",
        desc=f"{['First','Second','Third','Fourth'][index]} principal certification program meeting all requirements",
        parent=parent_node,
        critical=False,
    )

    # 1) University name and program title are provided (Critical existence check)
    uni_title_exists = bool(prog.university and prog.university.strip()) and bool(prog.program_title and prog.program_title.strip())
    evaluator.add_custom_node(
        result=uni_title_exists,
        id=f"program_{index + 1}_university_and_title",
        desc="The university name and program title are provided",
        parent=program_node,
        critical=True,
    )

    # 2) University accreditation (Critical) – verify regionally accredited
    acc_leaf = evaluator.add_leaf(
        id=f"program_{index + 1}_university_accreditation",
        desc="The program is offered by a regionally accredited university in the United States",
        parent=program_node,
        critical=True,
    )
    acc_claim = (
        f"The university '{prog.university or ''}' is regionally accredited by a recognized U.S. regional accrediting body "
        "(e.g., HLC, SACSCOC, MSCHE, WSCUC, NECHE, NWCCU)."
    )
    await evaluator.verify(
        claim=acc_claim,
        node=acc_leaf,
        sources=program_sources(prog),
        additional_instruction=(
            "Check the page(s) for explicit evidence of regional/institutional accreditation (e.g., statements like "
            "'accredited by the Higher Learning Commission' or 'SACSCOC'). If only national, programmatic, or missing, treat as not supported."
        ),
    )

    # 3) Program format (Critical) – fully online or hybrid suitable for working professionals
    fmt_leaf = evaluator.add_leaf(
        id=f"program_{index + 1}_format",
        desc="The program is offered fully online or in a hybrid format suitable for working professionals",
        parent=program_node,
        critical=True,
    )
    fmt_text = (prog.format or "").strip()
    fmt_claim = (
        f"The program '{prog.program_title or ''}' at '{prog.university or ''}' is offered in a format suitable for working professionals."
        " Specifically, it is either fully online or hybrid with limited in-person requirements."
    )
    await evaluator.verify(
        claim=fmt_claim,
        node=fmt_leaf,
        sources=prog.program_url or program_sources(prog),
        additional_instruction=(
            "Look for phrases such as '100% online', 'fully online', 'online', 'hybrid', 'online with occasional on-campus', "
            "or similar. Flexibility for working professionals should be evident."
        ),
    )

    # 4) Leads to principal or administrative services certification in a specific state (Critical)
    cert_leaf = evaluator.add_leaf(
        id=f"program_{index + 1}_state_certification",
        desc="The program leads to principal or administrative services certification in a specific U.S. state",
        parent=program_node,
        critical=True,
    )
    state_text = (prog.state or "").strip()
    cert_type_text = (prog.certification_type or "principal or administrative services").strip()
    cert_claim = (
        f"This program leads to {cert_type_text} certification in {state_text}."
        if state_text else
        "This program leads to principal or administrative services certification in a specific U.S. state."
    )
    await evaluator.verify(
        claim=cert_claim,
        node=cert_leaf,
        sources=program_sources(prog),
        additional_instruction=(
            "Confirm that the program page explicitly states it leads to a principal license/certification or an administrative services "
            "credential in the stated U.S. state (e.g., TX, CA, PA)."
        ),
    )

    # 5) The specific state is identified (Critical existence check)
    state_identified = bool(prog.state and prog.state.strip())
    evaluator.add_custom_node(
        result=state_identified,
        id=f"program_{index + 1}_state_identified",
        desc="The specific state where the certification is valid is identified",
        parent=program_node,
        critical=True,
    )

    # 6) Program duration (Non-critical) – verify time-to-completion
    dur_leaf = evaluator.add_leaf(
        id=f"program_{index + 1}_duration",
        desc="The program duration is specified (typically 12-24 months for completion)",
        parent=program_node,
        critical=False,
    )
    dur_text = (prog.duration or "").strip()
    dur_claim = (
        f"The stated estimated time to completion is '{dur_text}'."
        if dur_text else
        "The program provides an estimated time to completion."
    )
    await evaluator.verify(
        claim=dur_claim,
        node=dur_leaf,
        sources=prog.program_url or program_sources(prog),
        additional_instruction=(
            "Check for timeline language like 'complete in 12 months', 'finish in 18–24 months', '2 years', etc. "
            "Minor variations and rounding are acceptable."
        ),
    )

    # 7) Program cost/tuition (Non-critical) – verify price info
    cost_leaf = evaluator.add_leaf(
        id=f"program_{index + 1}_cost",
        desc="The program cost or tuition information is provided",
        parent=program_node,
        critical=False,
    )
    cost_text = (prog.cost or "").strip()
    cost_claim = (
        f"The program's tuition/cost information includes: '{cost_text}'."
        if cost_text else
        "The program provides tuition or cost information."
    )
    await evaluator.verify(
        claim=cost_claim,
        node=cost_leaf,
        sources=prog.program_url or program_sources(prog),
        additional_instruction=(
            "Verify that tuition or cost details are present (per credit, per course, total estimate, or fee tables). "
            "Minor formatting differences are acceptable."
        ),
    )

    # 8) Official program URL provided and valid (Critical) – verify page corresponds to program
    if prog.program_url and prog.program_url.strip():
        url_leaf = evaluator.add_leaf(
            id=f"program_{index + 1}_url",
            desc="A valid URL to the program's official page is provided",
            parent=program_node,
            critical=True,
        )
        url_claim = (
            f"This webpage is the official program page for '{prog.program_title or ''}' at '{prog.university or ''}'."
        )
        await evaluator.verify(
            claim=url_claim,
            node=url_leaf,
            sources=prog.program_url,
            additional_instruction=(
                "Confirm that the page is the official program page (e.g., hosted on the university domain and clearly "
                "shows program name/title and related details). Allow fuzzy matching for naming variations."
            ),
        )
    else:
        evaluator.add_custom_node(
            result=False,
            id=f"program_{index + 1}_url",
            desc="A valid URL to the program's official page is provided",
            parent=program_node,
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
    """
    Evaluate an answer for principal/educational leadership programs meeting specified criteria.
    """
    # Initialize evaluator; root is always non-critical by framework design.
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Programs evaluated independently
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

    # Extract up to 4 programs from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_programs(),
        template_class=ProgramsExtraction,
        extraction_name="programs_extraction",
    )

    # Normalize to exactly 4 items (slice or pad with empty ProgramItem)
    programs = list(extracted.programs[:4])
    while len(programs) < 4:
        programs.append(ProgramItem())

    evaluator.add_custom_info(
        info={"extracted_program_count": len(extracted.programs), "evaluated_program_count": 4},
        info_type="extraction_stats",
    )

    # Build verification subtrees for each program
    for i, prog in enumerate(programs):
        await verify_single_program(evaluator, root, prog, i)

    # Return structured summary
    return evaluator.get_summary()