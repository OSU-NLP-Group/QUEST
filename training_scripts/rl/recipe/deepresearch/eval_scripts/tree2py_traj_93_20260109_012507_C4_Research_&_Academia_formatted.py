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
TASK_ID = "neuro_postdoc_2026"
TASK_DESCRIPTION = """
A recent PhD graduate in neuroscience who is a U.S. citizen is seeking postdoctoral fellowship opportunities to begin in 2026. They require programs that offer a minimum commitment of 24 months (2 years) of funding support. Identify four different postdoctoral fellowship programs that meet all of the following criteria:

1. The fellowship must be specifically in the field of neuroscience or cognitive neuroscience
2. The fellowship must require or explicitly accept U.S. citizens as eligible applicants
3. The fellowship must provide a minimum duration of 24 months (2 years) of support
4. The fellowship must have a start date in calendar year 2026
5. The application deadline for the fellowship must be in 2025 or by March 2026
6. The fellowship must publicly disclose the annual stipend or salary amount
7. Applicants must have completed their PhD degree before the fellowship begins
8. The program must be explicitly at the postdoctoral level (not predoctoral or graduate student positions)

For each of the four fellowship programs, provide: the program name, the host institution or organization, and the reference URL where the fellowship information can be verified.
"""


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class FellowshipProgram(BaseModel):
    program_name: Optional[str] = None
    host: Optional[str] = None
    verification_urls: List[str] = Field(default_factory=list)

    # Free-text fields captured from the answer (not strictly required for verification,
    # but useful to craft claims or context if present)
    field: Optional[str] = None
    us_citizenship: Optional[str] = None
    duration: Optional[str] = None
    start_date: Optional[str] = None
    application_deadline: Optional[str] = None
    stipend: Optional[str] = None
    phd_requirement: Optional[str] = None
    level: Optional[str] = None


class FellowshipList(BaseModel):
    programs: List[FellowshipProgram] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_fellowships() -> str:
    return """
    Extract up to four distinct postdoctoral fellowship programs mentioned in the answer. For each program, extract the following fields exactly as presented:

    - program_name: The official program name
    - host: The host institution or organization name
    - verification_urls: An array of one or more URLs cited in the answer where the fellowship information (eligibility/terms) can be verified
    - field: The stated field or area (e.g., neuroscience, cognitive neuroscience)
    - us_citizenship: The eligibility statement related to U.S. citizenship (e.g., “U.S. citizens eligible”, “U.S. citizenship required”, “open to U.S. citizens and others”)
    - duration: The stated duration or support term (e.g., “two-year fellowship”, “24 months minimum”)
    - start_date: The stated start date or start window (e.g., “start in 2026”, “January 2026”)
    - application_deadline: The stated application deadline(s) relevant to the cycle (e.g., “deadline October 2025”, “rolling through March 2026”)
    - stipend: The stipend/salary disclosure statement (ideally including the annual amount if shown)
    - phd_requirement: The requirement regarding PhD completion timing (e.g., “PhD must be completed before start date”)
    - level: The program level description (e.g., “postdoctoral fellowship”, “postdoc”)

    Rules:
    - Only extract programs explicitly mentioned in the answer.
    - If more than four programs are mentioned, extract only the first four in the order they appear.
    - If fewer than four are mentioned, return only those mentioned (do not invent programs).
    - If any field is not mentioned for a program, set it to null (except verification_urls, which should be an empty array if not provided).
    - For verification_urls, extract actual URLs; if the answer uses markdown links, include the underlying URL.

    Return a JSON object with a single key 'programs' which is an array of up to four FellowshipProgram objects.
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def _canonicalize_url(url: Optional[str]) -> Optional[str]:
    if not url:
        return None
    u = url.strip()
    # normalize scheme and www, and strip trailing slash
    if u.startswith("http://"):
        u = u[len("http://") :]
    if u.startswith("https://"):
        u = u[len("https://") :]
    if u.startswith("www."):
        u = u[len("www.") :]
    if u.endswith("/"):
        u = u[:-1]
    return u.lower()


def _first_valid_url(urls: List[str]) -> Optional[str]:
    for u in urls:
        if isinstance(u, str) and u.strip():
            return u.strip()
    return None


# --------------------------------------------------------------------------- #
# Verification functions                                                      #
# --------------------------------------------------------------------------- #
async def verify_program(
    evaluator: Evaluator,
    parent_node,
    program: FellowshipProgram,
    index: int,
) -> None:
    """
    Verify all aspects of a single fellowship program according to rubric.
    The program node uses parallel aggregation, with critical leaves gating other checks.
    """
    # Create program node (non-critical to allow partial scoring per program)
    program_node = evaluator.add_parallel(
        id=f"Fellowship_Program_{index}",
        desc=f"Evaluate the {index}th fellowship program against all required constraints and required reported fields.",
        parent=parent_node,
        critical=False,
    )

    # Provide required identifying fields (critical existence checks)
    evaluator.add_custom_node(
        result=bool(program.program_name and program.program_name.strip()),
        id=f"Program_{index}_Provide_Name",
        desc=f"Provides the program name for fellowship program #{index}.",
        parent=program_node,
        critical=True,
    )

    evaluator.add_custom_node(
        result=bool(program.host and program.host.strip()),
        id=f"Program_{index}_Provide_Host",
        desc=f"Provides the host institution or organization for fellowship program #{index}.",
        parent=program_node,
        critical=True,
    )

    evaluator.add_custom_node(
        result=bool(program.verification_urls and len(program.verification_urls) > 0),
        id=f"Program_{index}_Provide_Verification_URL",
        desc=f"Provides at least one reference URL where eligibility/terms for fellowship program #{index} can be verified.",
        parent=program_node,
        critical=True,
    )

    # Build shared sources list for verification
    sources_list = program.verification_urls if program.verification_urls else []

    # Field: neuroscience or cognitive neuroscience
    field_node = evaluator.add_leaf(
        id=f"Program_{index}_Field",
        desc=f"Fellowship program #{index} is specifically in neuroscience or cognitive neuroscience.",
        parent=program_node,
        critical=True,
    )
    claim_field = (
        "This fellowship program is specifically in neuroscience or cognitive neuroscience."
    )
    await evaluator.verify(
        claim=claim_field,
        node=field_node,
        sources=sources_list,
        additional_instruction=(
            "Check the program page to confirm the field is explicitly neuroscience or cognitive neuroscience. "
            "Accept synonyms like 'neural science' or 'brain science' when clearly framed as a neuroscience program. "
            "Do not accept general biology/biomedical unless explicitly narrowed to neuroscience."
        ),
    )

    # U.S. citizen eligibility
    us_node = evaluator.add_leaf(
        id=f"Program_{index}_US_Citizen_Eligibility",
        desc=f"Fellowship program #{index} requires or explicitly accepts U.S. citizens as eligible applicants.",
        parent=program_node,
        critical=True,
    )
    claim_us = (
        "U.S. citizens are explicitly eligible applicants for this fellowship program (or U.S. citizenship is required)."
    )
    await evaluator.verify(
        claim=claim_us,
        node=us_node,
        sources=sources_list,
        additional_instruction=(
            "Look for explicit mention of U.S. citizens being eligible (e.g., 'U.S. citizens eligible', "
            "'U.S. citizenship required', or inclusive language that clearly includes U.S. citizens). "
            "If the page only mentions non-U.S. citizens or excludes U.S. citizens, this should fail."
        ),
    )

    # Duration: minimum 24 months
    duration_node = evaluator.add_leaf(
        id=f"Program_{index}_Duration",
        desc=f"Fellowship program #{index} provides a minimum of 24 months (2 years) of support.",
        parent=program_node,
        critical=True,
    )
    claim_duration = "The fellowship provides at least 24 months (two years) of support."
    await evaluator.verify(
        claim=claim_duration,
        node=duration_node,
        sources=sources_list,
        additional_instruction=(
            "Confirm the stated support duration is two years (24 months) or more (e.g., 'two-year appointment', "
            "'initial two-year term', '24 months minimum')."
        ),
    )

    # Start date: in calendar year 2026
    start_node = evaluator.add_leaf(
        id=f"Program_{index}_Start_Date",
        desc=f"Fellowship program #{index} has a start date in calendar year 2026.",
        parent=program_node,
        critical=True,
    )
    claim_start = "The fellowship start date is in calendar year 2026 (any month in 2026)."
    await evaluator.verify(
        claim=claim_start,
        node=start_node,
        sources=sources_list,
        additional_instruction=(
            "Verify that the program indicates fellows start in 2026 (e.g., 'Start: 2026', 'January 2026', "
            "'Fall 2026', or similar). Flexible ranges that include 2026 (e.g., 'start window includes 2026') are acceptable."
        ),
    )

    # Application deadline: in 2025 or by March 2026
    deadline_node = evaluator.add_leaf(
        id=f"Program_{index}_Deadline",
        desc=f"Fellowship program #{index} has an application deadline in 2025 or by March 2026.",
        parent=program_node,
        critical=True,
    )
    claim_deadline = (
        "The application deadline for the fellowship is in 2025 or no later than March 2026."
    )
    await evaluator.verify(
        claim=claim_deadline,
        node=deadline_node,
        sources=sources_list,
        additional_instruction=(
            "Check the stated application deadline(s). Accept any deadline dates in 2025 or up to March 31, 2026. "
            "If multiple cycles are listed, ensure that a cycle relevant to a 2026 start has a deadline that meets this window."
        ),
    )

    # Stipend disclosed: annual amount publicly disclosed
    stipend_node = evaluator.add_leaf(
        id=f"Program_{index}_Stipend_Disclosed",
        desc=f"Fellowship program #{index} publicly discloses the annual stipend or salary amount.",
        parent=program_node,
        critical=True,
    )
    claim_stipend = (
        "The program publicly discloses the annual stipend or salary amount (e.g., a numeric per-year figure)."
    )
    await evaluator.verify(
        claim=claim_stipend,
        node=stipend_node,
        sources=sources_list,
        additional_instruction=(
            "Look for an explicit annual amount (e.g., '$XX,XXX per year', 'annual stipend', 'salary: $YY,YYY/year'). "
            "General phrases like 'salary commensurate with experience' without a number should not pass."
        ),
    )

    # PhD completion before start
    phd_node = evaluator.add_leaf(
        id=f"Program_{index}_PhD_Completion_Timing",
        desc=f"Fellowship program #{index} requires applicants to have completed the PhD before the fellowship begins.",
        parent=program_node,
        critical=True,
    )
    claim_phd = "Applicants must have completed the PhD before the fellowship begins."
    await evaluator.verify(
        claim=claim_phd,
        node=phd_node,
        sources=sources_list,
        additional_instruction=(
            "Verify language such as 'PhD must be completed by the start date', 'doctoral degree required upon appointment', "
            "or equivalent. 'ABD' or 'near completion' without requiring completion by start should fail."
        ),
    )

    # Postdoctoral level explicitly
    level_node = evaluator.add_leaf(
        id=f"Program_{index}_Postdoc_Level",
        desc=f"Fellowship program #{index} is explicitly at the postdoctoral level (not predoctoral/graduate).",
        parent=program_node,
        critical=True,
    )
    claim_level = "This program is explicitly a postdoctoral-level fellowship (postdoc)."
    await evaluator.verify(
        claim=claim_level,
        node=level_node,
        sources=sources_list,
        additional_instruction=(
            "Confirm explicitly 'postdoctoral', 'postdoc', 'postdoctoral fellowship'. "
            "If the page describes predoctoral/graduate student programs, it should fail."
        ),
    )


# --------------------------------------------------------------------------- #
# Main evaluation function                                                    #
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
    Evaluate the answer for the neuroscience/cognitive neuroscience postdoc fellowships task.
    """
    # Initialize evaluator (root: non-critical parallel to allow partial scoring across programs)
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

    # Extract fellowship programs from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_fellowships(),
        template_class=FellowshipList,
        extraction_name="fellowship_programs",
    )

    # Keep only the first 4 programs; pad if needed with empty placeholders for consistent structure
    programs: List[FellowshipProgram] = list(extracted.programs[:4])
    while len(programs) < 4:
        programs.append(FellowshipProgram())

    # Distinct Programs check (critical under root)
    # Use canonicalized first verification URL per program to enforce uniqueness.
    normalized_urls: List[Optional[str]] = []
    for p in programs:
        first_url = _first_valid_url(p.verification_urls)
        normalized_urls.append(_canonicalize_url(first_url))

    # All four must have distinct non-null canonical URLs to pass distinctness
    has_four_urls = all(u is not None for u in normalized_urls)
    url_set = set(u for u in normalized_urls if u is not None)
    distinct_ok = has_four_urls and len(url_set) == 4

    evaluator.add_custom_node(
        result=distinct_ok,
        id="Distinct_Programs",
        desc="All four programs are different (no duplicates).",
        parent=root,
        critical=True,
    )

    # Build verification subtrees for each of the 4 programs
    # Index labels as 1..4 for IDs to match rubric naming style
    for idx, program in enumerate(programs, start=1):
        await verify_program(evaluator, root, program, idx)

    # Optional: record a quick summary of extracted programs for debugging
    summary_list = []
    for i, p in enumerate(programs, start=1):
        summary_list.append(
            {
                "index": i,
                "program_name": p.program_name,
                "host": p.host,
                "first_url": _first_valid_url(p.verification_urls),
            }
        )
    evaluator.add_custom_info(
        info={"extracted_programs_overview": summary_list},
        info_type="extraction_overview",
        info_name="extracted_programs_overview",
    )

    # Return evaluation summary
    return evaluator.get_summary()