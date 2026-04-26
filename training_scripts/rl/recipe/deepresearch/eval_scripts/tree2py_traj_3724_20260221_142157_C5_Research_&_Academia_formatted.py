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
TASK_ID = "postdoc_fall_2026_fellowships"
TASK_DESCRIPTION = (
    "I am an early-career researcher who recently completed my Ph.D. in planetary science in 2023. "
    "I am currently working as a postdoctoral researcher at a university in California, but I am interested "
    "in applying for competitive postdoctoral fellowship programs that would allow me to move to a different "
    "institution and pursue my own independent research direction. I want to identify suitable fellowship "
    "opportunities with application deadlines in the latter part of 2026, specifically between September and December, "
    "so I can prepare strong applications over the summer. Please identify three distinct postdoctoral fellowship programs "
    "that meet all of the following criteria: (1) the program explicitly supports independent research in planetary science, "
    "space science, astronomy, or astrophysics; (2) the program allows fellows to choose their own U.S. host institution rather "
    "than being restricted to a single predetermined location; (3) the program's next application deadline falls between "
    "September 1, 2026 and December 31, 2026; (4) the program is open to recent Ph.D. recipients who completed their doctoral "
    "degree within the past 5 years; and (5) the program provides at least one year of fellowship support with the potential "
    "for renewal to additional years."
)

DEADLINE_WINDOW_START = "September 1, 2026"
DEADLINE_WINDOW_END = "December 31, 2026"

# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class FellowshipProgram(BaseModel):
    program_name: Optional[str] = None
    reference_urls: List[str] = Field(default_factory=list)
    next_application_deadline: Optional[str] = None
    field_scope_text: Optional[str] = None  # e.g., "planetary science, astronomy"
    independence_support_text: Optional[str] = None  # e.g., "independent research; propose own project"
    host_institution_policy_text: Optional[str] = None  # e.g., "choose US host institution"
    eligibility_timing_text: Optional[str] = None  # e.g., "within 5 years of PhD"
    duration_text: Optional[str] = None  # e.g., "2 years, renewable"
    renewal_policy_text: Optional[str] = None  # e.g., "may renew for additional year(s)"


class FellowshipExtraction(BaseModel):
    programs: List[FellowshipProgram] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_programs() -> str:
    return (
        "Extract up to three distinct postdoctoral fellowship programs mentioned in the answer that are relevant to "
        "planetary science, space science, astronomy, or astrophysics and that are described with the specified criteria. "
        "For each identified program, extract the following fields exactly as they appear in the answer:\n"
        "1. program_name: The official name of the fellowship program.\n"
        "2. reference_urls: A list of URLs explicitly provided in the answer that reference official program sources "
        "(e.g., program pages hosted by agencies, institutes, or official organizations). If none are provided, return an empty list.\n"
        "3. next_application_deadline: The next application deadline date or date phrase mentioned in the answer (e.g., 'October 15, 2026'). "
        "If not provided, return null.\n"
        "4. field_scope_text: Any text in the answer indicating the program fields (planetary science, space science, astronomy, astrophysics). "
        "If not provided, return null.\n"
        "5. independence_support_text: Any text in the answer indicating the program supports independent research or allows fellows to propose "
        "their own research direction. If not provided, return null.\n"
        "6. host_institution_policy_text: Any text in the answer indicating the program allows fellows to choose their own U.S. host institution. "
        "If not provided, return null.\n"
        "7. eligibility_timing_text: Any text in the answer indicating eligibility timing related to Ph.D. year (e.g., within 5 years). "
        "If not provided, return null.\n"
        "8. duration_text: Any text indicating the fellowship duration. If not provided, return null.\n"
        "9. renewal_policy_text: Any text indicating whether renewal/extension beyond the initial term is possible. If not provided, return null.\n\n"
        "Return a JSON object with a 'programs' array of up to 3 objects. If the answer mentions more than 3 programs, include only the first 3. "
        "If fewer than 3 programs are mentioned, include as many as are available."
    )


# --------------------------------------------------------------------------- #
# Verification functions                                                      #
# --------------------------------------------------------------------------- #
async def verify_program(
    evaluator: Evaluator,
    parent_node,
    program: FellowshipProgram,
    program_index: int,
) -> None:
    """
    Build verification subtree for one fellowship program and perform verifications.
    All leaf nodes correspond directly to rubric criteria and are critical.
    """
    # Create program node (non-critical, parallel aggregation across criteria)
    program_node = evaluator.add_parallel(
        id=f"fellowship_program_{program_index + 1}",
        desc=(
            "First postdoctoral fellowship program meeting all criteria"
            if program_index == 0
            else ("Second postdoctoral fellowship program meeting all criteria" if program_index == 1
                  else "Third postdoctoral fellowship program meeting all criteria")
        ),
        parent=parent_node,
        critical=False,
    )

    name_for_claim = program.program_name or "the fellowship program"
    urls = program.reference_urls if program.reference_urls else None
    deadline_text = program.next_application_deadline or "unknown"

    # Leaf 1: Field requirement (independent research in planetary/space/astronomy/astrophysics)
    field_node = evaluator.add_leaf(
        id=f"fellowship_program_{program_index + 1}_field_requirement",
        desc="Program explicitly supports independent research in planetary science, space science, astronomy, or astrophysics",
        parent=program_node,
        critical=True,
    )
    field_claim = (
        f"{name_for_claim} supports independent research in one or more of the following fields: "
        f"planetary science, space science, astronomy, or astrophysics."
    )
    field_instruction = (
        "Confirm that the official program page explicitly covers at least one of these fields and supports independent research "
        "(fellows propose their own projects or have research independence). Accept clear synonyms such as 'astrophysics', "
        "'planetary', 'space sciences', 'astronomy', and phrases like 'fellows define their research', 'independent research', "
        "or 'self-directed research'."
    )

    # Leaf 2: Host institution flexibility (choose own U.S. host institution)
    host_node = evaluator.add_leaf(
        id=f"fellowship_program_{program_index + 1}_host_institution_flexibility",
        desc="Program allows fellows to choose their own U.S. host institution (not restricted to a single predetermined location)",
        parent=program_node,
        critical=True,
    )
    host_claim = (
        f"{name_for_claim} allows the fellow to choose their own host institution located in the United States, "
        f"rather than being restricted to a single predetermined location."
    )
    host_instruction = (
        "Look for language such as 'fellows select a U.S. host institution', 'host institution of choice', "
        "'choose a host/mentor at any U.S. institution', or similar. Programs restricted to one fixed site do NOT satisfy this criterion."
    )

    # Leaf 3: Application deadline window (Sep 1, 2026 – Dec 31, 2026)
    deadline_node = evaluator.add_leaf(
        id=f"fellowship_program_{program_index + 1}_application_deadline",
        desc="Program's next application deadline falls between September 1, 2026 and December 31, 2026",
        parent=program_node,
        critical=True,
    )
    deadline_claim = (
        f"The program's next application deadline is {deadline_text}, and it falls between {DEADLINE_WINDOW_START} and {DEADLINE_WINDOW_END}."
    )
    deadline_instruction = (
        "From the official page, identify the next application deadline for the 2026 cycle (or next cycle that occurs in late 2026). "
        "Judge whether the date is within the inclusive window Sep 1, 2026 to Dec 31, 2026. "
        "If the page only provides a month (e.g., 'October 2026'), treat that as within the window. "
        "If multiple deadlines appear, use the next upcoming deadline relevant to the main fellowship application."
    )

    # Leaf 4: Eligibility timing (within 5 years of PhD)
    eligibility_node = evaluator.add_leaf(
        id=f"fellowship_program_{program_index + 1}_eligibility_timing",
        desc="Program is open to recent Ph.D. recipients who completed their doctoral degree within the past 5 years",
        parent=program_node,
        critical=True,
    )
    eligibility_claim = (
        f"{name_for_claim} is open to applicants who earned their Ph.D. within the past five years (or an equivalent rule clearly within 5 years)."
    )
    eligibility_instruction = (
        "Verify eligibility language such as 'within 5 years of Ph.D.', 'no more than five years since Ph.D. conferral', "
        "or cycle-specific wording implying a 5-year window. Accept stricter rules (e.g., <= 5 years) as satisfying the criterion."
    )

    # Leaf 5: Fellowship duration (>=1 year) with potential renewal
    duration_node = evaluator.add_leaf(
        id=f"fellowship_program_{program_index + 1}_fellowship_duration",
        desc="Program provides at least one year of fellowship support with potential for renewal",
        parent=program_node,
        critical=True,
    )
    duration_claim = (
        f"{name_for_claim} provides at least one year of support and includes potential for renewal or extension to additional years."
    )
    duration_instruction = (
        "Confirm the program offers an initial term of one year or longer and explicitly allows renewal/extension beyond the initial term. "
        "Examples include 'two-year fellowship renewable for a third year' or 'one-year fellowship with possible renewal'. "
        "A strictly fixed multi-year term with no renewal does NOT satisfy 'potential for renewal'."
    )

    # Leaf 6: Reference URL is official program source
    refurl_node = evaluator.add_leaf(
        id=f"fellowship_program_{program_index + 1}_reference_url",
        desc="A valid reference URL from an official program source is provided",
        parent=program_node,
        critical=True,
    )
    refurl_claim = (
        f"At least one provided URL is an official program source page for {name_for_claim} (or an official page from the administering organization)."
    )
    refurl_instruction = (
        "Evaluate whether any provided URL is an authoritative program source (e.g., pages hosted by government (.gov), official institute/observatory, "
        "university (.edu), or the program's official organization. Generic news articles or third-party summaries do NOT count as official sources)."
    )

    # Perform batch verification for the six leaves (parallel under the program node)
    await evaluator.batch_verify(
        [
            (field_claim, urls, field_node, field_instruction),
            (host_claim, urls, host_node, host_instruction),
            (deadline_claim, urls, deadline_node, deadline_instruction),
            (eligibility_claim, urls, eligibility_node, eligibility_instruction),
            (duration_claim, urls, duration_node, duration_instruction),
            (refurl_claim, urls, refurl_node, refurl_instruction),
        ]
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
    Evaluate an answer for the postdoctoral fellowship programs task.
    """
    # Initialize evaluator and root node
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

    # Extract fellowship programs from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_programs(),
        template_class=FellowshipExtraction,
        extraction_name="fellowship_programs_extraction",
    )

    # Record useful context in summary
    evaluator.add_custom_info(
        {
            "deadline_window_start": DEADLINE_WINDOW_START,
            "deadline_window_end": DEADLINE_WINDOW_END,
            "expected_program_count": 3,
        },
        info_type="deadline_window",
        info_name="constraints"
    )

    # Prepare up to 3 programs (pad with empty ones if fewer provided)
    programs: List[FellowshipProgram] = list(extracted.programs[:3])
    while len(programs) < 3:
        programs.append(FellowshipProgram())

    # Build verification tree for each program
    for idx in range(3):
        await verify_program(evaluator, root, programs[idx], idx)

    # Return the evaluation summary
    return evaluator.get_summary()