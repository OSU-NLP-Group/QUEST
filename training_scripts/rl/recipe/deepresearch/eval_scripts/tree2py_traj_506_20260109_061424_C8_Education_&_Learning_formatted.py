import asyncio
import logging
from typing import Any, Optional, List, Dict

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "online_masters_affordable_flexible"
TASK_DESCRIPTION = """
I am considering pursuing an online master's degree in business or data analytics and want to explore affordable, flexible options from accredited institutions. Please identify three distinct online master's degree programs that meet all of the following requirements:

Program Requirements:
1. The program must be offered entirely online (100% online delivery)
2. The program must be regionally accredited by a recognized U.S. accrediting agency
3. The program must be in business, management, or data analytics
4. The total program tuition cost must be under $25,000 for all students, with no out-of-state tuition differential
5. The program must be completable in 24 months or less for full-time students
6. The program must not require GRE or GMAT scores for admission
7. The program must offer at least two different concentrations or specializations
8. The program must have multiple start dates per year (at least 3 start opportunities annually)
9. The program must require 36 credits or fewer to complete
10. The program must offer asynchronous course delivery (students can access materials on their own schedule)
11. The program must provide dedicated career services for online students
12. The program must accept transfer credits (at least 6 graduate credits)
13. The program must include a capstone project or applied learning experience
14. The program must have courses delivered in 10-week terms or shorter
15. The program must provide access to an alumni network for online students

For each program, provide:
- The university name and specific program title
- A direct link to the program's official page
- A direct link to the tuition information page
- The total program tuition cost
- The number of credits required
- The available concentrations or specializations (list at least two)
- The typical completion time for full-time students
- The number of start dates offered per year
"""


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class ProgramItem(BaseModel):
    university: Optional[str] = None
    program_title: Optional[str] = None
    program_url: Optional[str] = None
    tuition_url: Optional[str] = None

    total_tuition: Optional[str] = None
    credits_required: Optional[str] = None
    concentrations: List[str] = Field(default_factory=list)
    completion_time: Optional[str] = None
    start_dates_per_year: Optional[str] = None

    field_category: Optional[str] = None  # e.g., "business", "management", "data analytics"

    accreditation_agency: Optional[str] = None
    accreditation_url: Optional[str] = None

    asynchronous_delivery: Optional[str] = None
    gre_gmat_required: Optional[str] = None  # e.g., "not required", "optional", "required", etc.

    transfer_credits: Optional[str] = None  # e.g., "up to 12 credits"

    capstone_component: Optional[str] = None
    term_length: Optional[str] = None  # e.g., "8-week courses", "10-week terms"

    career_services_url: Optional[str] = None
    alumni_network_url: Optional[str] = None


class ProgramsExtraction(BaseModel):
    programs: List[ProgramItem] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_programs() -> str:
    return """
    Extract up to the first three distinct online master's programs mentioned in the answer. For each program, return a JSON object with these fields:
    - university: University name
    - program_title: Specific program title
    - program_url: Direct URL to the official program page
    - tuition_url: Direct URL to a tuition/costs page for this program (if provided)
    - total_tuition: Total program tuition cost mentioned in the answer (string; keep as written)
    - credits_required: Number of credits required to complete the program (string; keep as written)
    - concentrations: List of at least two concentrations/specializations if provided in the answer (strings; keep as written)
    - completion_time: Typical completion duration for full-time students (string; keep as written, e.g., "18 months", "12-24 months")
    - start_dates_per_year: Number or description of annual start dates (string; keep as written)
    - field_category: One of {"business", "management", "data analytics"} if stated or reasonably implied; otherwise null
    - accreditation_agency: Regional accreditor (e.g., "SACSCOC", "HLC", "MSCHE", "NECHE", "WSCUC", "NWCCU") if provided; otherwise null
    - accreditation_url: URL pointing to accreditation information (university or program page) if provided; otherwise null
    - asynchronous_delivery: Whether asynchronous delivery is available (string as mentioned)
    - gre_gmat_required: Whether GRE/GMAT is required (string as mentioned, e.g., "not required", "optional", "waived", "required")
    - transfer_credits: Transfer credit acceptance (string as mentioned, e.g., "up to 12 graduate credits")
    - capstone_component: Whether capstone/applied learning is included (string as mentioned)
    - term_length: Course/term length (string as mentioned, e.g., "8-week", "5-week", "10-week")
    - career_services_url: URL for online career services or support if provided
    - alumni_network_url: URL mentioning alumni network access if provided

    Important:
    - Extract only what appears in the answer; do not invent or infer missing fields.
    - If a field is not present in the answer, set it to null (or empty list for concentrations).
    - Return a JSON object with an array 'programs' containing up to three ProgramItem objects in the order they appear.
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def valid_url(url: Optional[str]) -> bool:
    if not url:
        return False
    u = url.strip()
    return u.startswith("http://") or u.startswith("https://")


def non_empty_urls(*urls: Optional[str]) -> List[str]:
    return [u.strip() for u in urls if u and u.strip()]


def program_display_name(p: ProgramItem, idx: int) -> str:
    uni = (p.university or "").strip()
    title = (p.program_title or "").strip()
    if uni and title:
        return f"{uni} — {title} (Program #{idx})"
    if title:
        return f"{title} (Program #{idx})"
    if uni:
        return f"{uni} (Program #{idx})"
    return f"Program #{idx}"


# --------------------------------------------------------------------------- #
# Verification logic per program                                              #
# --------------------------------------------------------------------------- #
async def verify_single_program(
    evaluator: Evaluator,
    parent_node,
    program: ProgramItem,
    ordinal_index: int
) -> None:
    """
    Build the verification tree for a single program and run checks.
    ordinal_index is 1-based (1, 2, 3).
    """
    # Program container node (non-critical to allow partial scoring across programs)
    prog_node = evaluator.add_parallel(
        id=f"program_{ordinal_index}",
        desc=(
            "First qualifying online master's program" if ordinal_index == 1 else
            ("Second qualifying online master's program (distinct from first program)" if ordinal_index == 2
             else "Third qualifying online master's program (distinct from first two programs)")
        ),
        parent=parent_node,
        critical=False
    )

    # -------------------------- Basic Eligibility -------------------------- #
    basic_node = evaluator.add_parallel(
        id=f"p{ordinal_index}_basic_eligibility",
        desc="Program meets fundamental eligibility requirements",
        parent=prog_node,
        critical=True
    )

    # URL existence (critical custom node)
    url_ok = valid_url(program.program_url)
    url_node = evaluator.add_custom_node(
        result=url_ok,
        id=f"p{ordinal_index}_url_reference",
        desc="Valid URL provided to the program's official page",
        parent=basic_node,
        critical=True
    )

    # 100% online delivery (critical leaf)
    online_leaf = evaluator.add_leaf(
        id=f"p{ordinal_index}_online_delivery",
        desc="Program is offered 100% online",
        parent=basic_node,
        critical=True
    )
    await evaluator.verify(
        claim="This master's program is delivered entirely online (100% online).",
        node=online_leaf,
        sources=program.program_url,
        additional_instruction=(
            "Verify on the program page whether it states 'online', 'fully online', '100% online', or equivalent. "
            "If any mandatory on-campus/hybrid component exists, treat as not 100% online."
        ),
        extra_prerequisites=[url_node]
    )

    # Regional accreditation (critical leaf)
    accred_leaf = evaluator.add_leaf(
        id=f"p{ordinal_index}_accreditation",
        desc="Program is regionally accredited by a recognized U.S. accrediting agency",
        parent=basic_node,
        critical=True
    )
    accred_claim_agency = (program.accreditation_agency or "a recognized regional accreditor")
    await evaluator.verify(
        claim=f"The institution offering this program is regionally accredited by {accred_claim_agency}.",
        node=accred_leaf,
        sources=non_empty_urls(program.accreditation_url, program.program_url),
        additional_instruction=(
            "Recognized U.S. institutional/regional accreditors include: HLC, MSCHE, SACSCOC, NECHE, WSCUC, NWCCU. "
            "Programmatic accreditations (e.g., AACSB, ABET) do NOT count for this criterion. "
            "Confirm institutional/regional accreditation on the accreditation page or the university site."
        ),
        extra_prerequisites=[url_node]
    )

    # Field category (critical leaf)
    field_leaf = evaluator.add_leaf(
        id=f"p{ordinal_index}_field",
        desc="Program is in business, management, or data analytics field",
        parent=basic_node,
        critical=True
    )
    field_label = (program.field_category or "business, management, or data analytics")
    await evaluator.verify(
        claim=f"This program is a master's program in {field_label}.",
        node=field_leaf,
        sources=program.program_url,
        additional_instruction=(
            "Accept common variants: MBA, Master of Science in Business Analytics, MS in Management, "
            "MS in Data Analytics, etc. Confirm the program fits one of the specified fields."
        ),
        extra_prerequisites=[url_node]
    )

    # --------------------------- Cost Requirements ------------------------- #
    cost_node = evaluator.add_parallel(
        id=f"p{ordinal_index}_cost_requirements",
        desc="Program meets cost and financial requirements",
        parent=prog_node,
        critical=True
    )

    # Tuition URL existence (critical custom node)
    tuition_url_ok = valid_url(program.tuition_url)
    tuition_url_node = evaluator.add_custom_node(
        result=tuition_url_ok,
        id=f"p{ordinal_index}_cost_url",
        desc="Valid URL provided showing tuition information",
        parent=cost_node,
        critical=True
    )

    # Tuition under $25,000 (critical leaf)
    tuition_leaf = evaluator.add_leaf(
        id=f"p{ordinal_index}_tuition_cost",
        desc="Total program tuition is under $25,000 for all students",
        parent=cost_node,
        critical=True
    )
    await evaluator.verify(
        claim="The total program tuition (excluding fees) is under $25,000.",
        node=tuition_leaf,
        sources=non_empty_urls(program.tuition_url, program.program_url),
        additional_instruction=(
            "If the page lists per-credit tuition and number of credits, estimate total tuition by multiplying. "
            "Ignore fees for this check. If total shown is clearly under $25,000, pass."
        ),
        extra_prerequisites=[tuition_url_node, url_node]
    )

    # No out-of-state differential (critical leaf)
    residency_leaf = evaluator.add_leaf(
        id=f"p{ordinal_index}_no_out_of_state",
        desc="Program has no out-of-state tuition differential or charges the same rate for all students",
        parent=cost_node,
        critical=True
    )
    await evaluator.verify(
        claim="The program charges the same tuition rate for in-state and out-of-state students (no out-of-state differential).",
        node=residency_leaf,
        sources=non_empty_urls(program.tuition_url, program.program_url),
        additional_instruction=(
            "Confirm that online tuition is a single rate regardless of residency. "
            "Wording like 'same tuition for all students', 'single tuition rate', or explicit absence of "
            "in-state vs. out-of-state differences should pass."
        ),
        extra_prerequisites=[tuition_url_node, url_node]
    )

    # -------------------------- Program Structure -------------------------- #
    structure_node = evaluator.add_parallel(
        id=f"p{ordinal_index}_program_structure",
        desc="Program structure meets specified requirements",
        parent=prog_node,
        critical=True
    )

    # Completion time <= 24 months (critical leaf)
    time_leaf = evaluator.add_leaf(
        id=f"p{ordinal_index}_completion_time",
        desc="Program is completable in 24 months or less for full-time students",
        parent=structure_node,
        critical=True
    )
    await evaluator.verify(
        claim="The program can be completed within 24 months or less for full-time students.",
        node=time_leaf,
        sources=program.program_url,
        additional_instruction=(
            "Check typical duration statements like '12 months', '18 months', '24 months' for full-time pacing. "
            "If only ranges are provided (e.g., 12–24 months), that satisfies the requirement."
        ),
        extra_prerequisites=[url_node]
    )

    # Credits <= 36 (critical leaf)
    credits_leaf = evaluator.add_leaf(
        id=f"p{ordinal_index}_credit_requirement",
        desc="Program requires 36 credits or fewer to complete",
        parent=structure_node,
        critical=True
    )
    await evaluator.verify(
        claim="This program requires 36 credit hours or fewer to complete.",
        node=credits_leaf,
        sources=non_empty_urls(program.program_url, program.tuition_url),
        additional_instruction=(
            "Accept synonyms like 'credits', 'credit hours', 'semester hours'. "
            "If the page shows a number <= 36 (e.g., 30, 33, 36), pass."
        ),
        extra_prerequisites=[url_node]
    )

    # Asynchronous delivery (critical leaf)
    async_leaf = evaluator.add_leaf(
        id=f"p{ordinal_index}_course_format",
        desc="Program offers asynchronous course delivery",
        parent=structure_node,
        critical=True
    )
    await evaluator.verify(
        claim="The program offers asynchronous course delivery for online students.",
        node=async_leaf,
        sources=program.program_url,
        additional_instruction=(
            "Look for wording such as 'asynchronous', 'self-paced', 'on your schedule'. "
            "If some synchronous sessions exist but asynchronous delivery is clearly available, pass."
        ),
        extra_prerequisites=[url_node]
    )

    # Term length <= 10 weeks (critical leaf)
    term_leaf = evaluator.add_leaf(
        id=f"p{ordinal_index}_term_length",
        desc="Program has courses delivered in 10-week terms or shorter",
        parent=structure_node,
        critical=True
    )
    await evaluator.verify(
        claim="Courses in this program are delivered in terms of 10 weeks or shorter.",
        node=term_leaf,
        sources=program.program_url,
        additional_instruction=(
            "Accept explicit term/course module lengths of 10 weeks or shorter (e.g., 8-week, 7.5-week, 5-week). "
            "If exact term length is unclear or longer than 10 weeks, fail."
        ),
        extra_prerequisites=[url_node]
    )

    # --------------------- Admission & Scheduling Flexibility --------------- #
    admission_node = evaluator.add_parallel(
        id=f"p{ordinal_index}_admission_flexibility",
        desc="Program meets admission and scheduling requirements",
        parent=prog_node,
        critical=True
    )

    # No GRE/GMAT required (critical leaf)
    tests_leaf = evaluator.add_leaf(
        id=f"p{ordinal_index}_no_gre_gmat",
        desc="Program does not require GRE or GMAT scores for admission",
        parent=admission_node,
        critical=True
    )
    await evaluator.verify(
        claim="GRE or GMAT scores are not required for admission to this program.",
        node=tests_leaf,
        sources=program.program_url,
        additional_instruction=(
            "If tests are explicitly 'optional', 'not required', or universally 'waived', pass. "
            "If they are required unless individual waivers apply, fail."
        ),
        extra_prerequisites=[url_node]
    )

    # At least 3 start dates per year (critical leaf)
    starts_leaf = evaluator.add_leaf(
        id=f"p{ordinal_index}_start_dates",
        desc="Program offers at least 3 start dates per year",
        parent=admission_node,
        critical=True
    )
    await evaluator.verify(
        claim="This program offers at least three start dates per year.",
        node=starts_leaf,
        sources=program.program_url,
        additional_instruction=(
            "Wording like 'multiple start dates', '6 annual starts', 'monthly starts', or 3+ terms per year should pass. "
            "If only one or two starts per year, fail."
        ),
        extra_prerequisites=[url_node]
    )

    # Accepts transfer credits (>=6 graduate credits) (critical leaf)
    transfer_leaf = evaluator.add_leaf(
        id=f"p{ordinal_index}_transfer_credits",
        desc="Program accepts transfer credits (at least 6 graduate credits)",
        parent=admission_node,
        critical=True
    )
    await evaluator.verify(
        claim="This program accepts at least 6 graduate transfer credits.",
        node=transfer_leaf,
        sources=program.program_url,
        additional_instruction=(
            "Look for 'transfer up to X credits' or similar. If X >= 6, pass."
        ),
        extra_prerequisites=[url_node]
    )

    # -------------------------- Program Features --------------------------- #
    features_node = evaluator.add_parallel(
        id=f"p{ordinal_index}_program_features",
        desc="Program provides required features and support",
        parent=prog_node,
        critical=True
    )

    # At least two specializations (critical leaf)
    specs_leaf = evaluator.add_leaf(
        id=f"p{ordinal_index}_specializations",
        desc="Program offers at least two different concentrations or specializations",
        parent=features_node,
        critical=True
    )
    listed_specs = ", ".join(program.concentrations[:2]) if program.concentrations else ""
    await evaluator.verify(
        claim=("The program offers at least two different concentrations or specializations"
               + (f": {listed_specs}." if listed_specs else ".")),
        node=specs_leaf,
        sources=program.program_url,
        additional_instruction=(
            "Accept synonyms like 'concentrations', 'tracks', 'focus areas'. "
            "If the page lists two or more options, pass."
        ),
        extra_prerequisites=[url_node]
    )

    # Capstone/applied learning (critical leaf)
    capstone_leaf = evaluator.add_leaf(
        id=f"p{ordinal_index}_capstone",
        desc="Program includes a capstone project or applied learning experience",
        parent=features_node,
        critical=True
    )
    await evaluator.verify(
        claim="This program includes a capstone project or an applied learning experience.",
        node=capstone_leaf,
        sources=program.program_url,
        additional_instruction=(
            "Look for 'capstone', 'culminating project', 'applied project', 'practicum', or similar."
        ),
        extra_prerequisites=[url_node]
    )

    # Career services for online students (critical leaf)
    career_leaf = evaluator.add_leaf(
        id=f"p{ordinal_index}_career_services",
        desc="Program provides dedicated career services for online students",
        parent=features_node,
        critical=True
    )
    await evaluator.verify(
        claim="Dedicated career services are available and accessible to online students in this program.",
        node=career_leaf,
        sources=non_empty_urls(program.career_services_url, program.program_url),
        additional_instruction=(
            "Accept institution-level career services that explicitly support online/remote students. "
            "General career services without online accessibility may fail."
        ),
        extra_prerequisites=[url_node]
    )

    # Alumni network access for online students (critical leaf)
    alumni_leaf = evaluator.add_leaf(
        id=f"p{ordinal_index}_alumni_network",
        desc="Program provides access to an alumni network for online students",
        parent=features_node,
        critical=True
    )
    await evaluator.verify(
        claim="An alumni network is available to students and accessible to online graduates.",
        node=alumni_leaf,
        sources=non_empty_urls(program.alumni_network_url, program.program_url),
        additional_instruction=(
            "Institution-level alumni networks that include online alumni should pass."
        ),
        extra_prerequisites=[url_node]
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
    Evaluate an answer for the affordable, flexible online master's programs task.
    """
    # Initialize evaluator (root should be non-critical to allow partial scoring and avoid critical consistency constraints)
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

    # Extract programs
    extracted = await evaluator.extract(
        prompt=prompt_extract_programs(),
        template_class=ProgramsExtraction,
        extraction_name="programs_extraction"
    )

    # Take first 3 programs; pad with empty if fewer
    programs = list(extracted.programs[:3])
    while len(programs) < 3:
        programs.append(ProgramItem())

    # Distinctness check (critical under root)
    keys = []
    for p in programs:
        key = (p.program_url or "").strip()
        if not key:
            uni = (p.university or "").strip()
            title = (p.program_title or "").strip()
            key = f"{uni}::{title}" if (uni or title) else ""
        if key:
            keys.append(key)
    distinct = len(keys) == 3 and len(set(keys)) == 3
    evaluator.add_custom_node(
        result=distinct,
        id="distinct_programs",
        desc="Three programs are distinct (different URLs or titles/universities).",
        parent=root,
        critical=True
    )

    # Build verification for each program
    for idx, program in enumerate(programs, start=1):
        await verify_single_program(evaluator, root, program, idx)

    # Return structured summary
    return evaluator.get_summary()