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
TASK_ID = "astro_postdoc_fellowships_2026"
TASK_DESCRIPTION = """A postdoctoral researcher in astronomy who completed their PhD in December 2023 at a US university (US citizen) is seeking highly competitive fellowship programs to apply for. Identify two distinct named postdoctoral fellowship programs in astronomy or astrophysics that meet ALL of the following criteria:

1. Eligible for US citizens
2. PhD completion in December 2023 falls within the program's eligible timeframe
3. Support observational, theoretical, or computational astronomy/astrophysics research
4. Provide annual stipend of at least $75,000
5. Include annual research budget or allowance of at least $15,000
6. Offer fellowship duration of at least 2 years
7. Have an application deadline occurring between February 22, 2026 and December 31, 2026
8. Provide or fund health insurance benefits for fellows

For each of the two fellowship programs, provide:
- Official program name
- Administering institution or organization
- Specific application deadline for the 2026 cycle
- Annual stipend amount
- Annual research budget/allowance amount
- Total fellowship duration in years
- At least one reference URL from official sources documenting these program details
"""

DEADLINE_START = "2026-02-22"
DEADLINE_END = "2026-12-31"

CANDIDATE_PROFILE = {
    "citizenship": "US",
    "phd_completion_date": "2023-12-15",  # representative date in December 2023
    "field": "astronomy/astrophysics",
}

# Shared instruction to enforce official-source grounding across verifications
OFFICIAL_SOURCE_POLICY = (
    "Use only official program pages (e.g., NASA, NSF, associated observatories, or the host university/organization) "
    "as valid evidence. If the provided URL is not an official source, treat the claim as NOT supported. Prefer explicit "
    "language on the page (Eligibility, Benefits, Stipend, Duration, Application Deadline) rather than inference."
)

# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class FellowshipProgram(BaseModel):
    name: Optional[str] = None
    organization: Optional[str] = None
    deadline_date: Optional[str] = None  # the 2026 cycle deadline as stated in the answer
    stipend: Optional[str] = None        # annual stipend amount (string; can be range or exact)
    research_budget: Optional[str] = None  # annual research/allowance amount (string; can be range or exact)
    duration_years: Optional[str] = None   # total duration (string; e.g., "2", "3 years", "2-3 years")
    urls: List[str] = Field(default_factory=list)


class FellowshipsExtraction(BaseModel):
    programs: List[FellowshipProgram] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_programs() -> str:
    return """
    Identify up to TWO distinct named postdoctoral fellowship programs in astronomy or astrophysics mentioned in the answer.
    For each program, extract the following fields exactly as stated in the answer:
    - name: The official program name (string). If not stated, return null.
    - organization: The administering institution or organization (string). If not stated, return null.
    - deadline_date: The specific application deadline for the 2026 cycle (string; any reasonable format). If not stated, return null.
    - stipend: The annual stipend amount (string; allow ranges or textual expressions). If not stated, return null.
    - research_budget: The annual research budget or allowance amount (string; allow ranges or textual expressions). If not stated, return null.
    - duration_years: The total fellowship duration in years (string; allow ranges or textual expressions). If not stated, return null.
    - urls: An array of all explicit URLs provided for this program (must be actual URLs visible in the answer; include full http/https).
    
    Rules:
    - Extract only what is explicitly in the answer. Do not infer or invent values.
    - If the answer mentions more than two programs, extract only the first two.
    - If the answer mentions fewer than two programs, still return up to two with missing fields as null and an empty urls array for missing programs.
    - For URLs, include only valid, complete URLs. If a URL is missing a protocol, prepend 'http://'.
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def _filter_valid_urls(urls: List[str]) -> List[str]:
    """Keep only plausible web URLs."""
    if not urls:
        return []
    cleaned = []
    for u in urls:
        if not isinstance(u, str):
            continue
        x = u.strip()
        if not x:
            continue
        if x.startswith("http://") or x.startswith("https://"):
            cleaned.append(x)
        else:
            # naive fix per extraction rules
            cleaned.append("http://" + x)
    return cleaned


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
    Build the verification subtree for a single fellowship program and run checks.
    """
    # Top-level node for this program (non-critical to allow partial credit across programs)
    prog_node = evaluator.add_parallel(
        id=f"Fellowship_Program_{program_index+1}",
        desc=f"{'First' if program_index == 0 else 'Second'} fellowship program identified with complete information meeting all criteria",
        parent=parent_node,
        critical=False
    )

    valid_urls = _filter_valid_urls(program.urls)

    # ---------------------- Identification ---------------------- #
    ident_node = evaluator.add_parallel(
        id=f"Program_{program_index+1}_Identification",
        desc="Program correctly identified with name and administering organization",
        parent=prog_node,
        critical=True
    )

    # Reference URL presence - do this first to gate other checks
    ref_url_exists = evaluator.add_custom_node(
        result=(len(valid_urls) >= 1),
        id=f"Program_{program_index+1}_Reference_URL",
        desc="At least one valid reference URL from official sources provided",
        parent=ident_node,
        critical=True
    )

    # Official program name
    name_leaf = evaluator.add_leaf(
        id=f"Program_{program_index+1}_Official_Name",
        desc="Official program name provided accurately",
        parent=ident_node,
        critical=True
    )
    name_val = program.name or ""
    await evaluator.verify(
        claim=f"The official program name is '{name_val}'.",
        node=name_leaf,
        sources=valid_urls,
        additional_instruction=(
            OFFICIAL_SOURCE_POLICY +
            " Confirm the program's official name exactly or with minor acceptable variations "
            "(e.g., inclusion/exclusion of 'Fellowship' if the official page indicates the same program)."
        ),
        extra_prerequisites=[ref_url_exists],
    )

    # Administering organization
    org_leaf = evaluator.add_leaf(
        id=f"Program_{program_index+1}_Administering_Organization",
        desc="Administering institution or organization correctly identified",
        parent=ident_node,
        critical=True
    )
    org_val = program.organization or ""
    await evaluator.verify(
        claim=f"The administering institution or organization for this program is '{org_val}'.",
        node=org_leaf,
        sources=valid_urls,
        additional_instruction=OFFICIAL_SOURCE_POLICY + " Verify the organization designated as the program's administrator.",
        extra_prerequisites=[ref_url_exists],
    )

    # ---------------------- Eligibility ---------------------- #
    elig_node = evaluator.add_parallel(
        id=f"Program_{program_index+1}_Eligibility_Verification",
        desc="Program eligibility requirements verified for the candidate profile",
        parent=prog_node,
        critical=True
    )

    # Citizenship check
    citizen_leaf = evaluator.add_leaf(
        id=f"Program_{program_index+1}_Citizenship_Check",
        desc="Program accepts US citizens as eligible applicants",
        parent=elig_node,
        critical=True
    )
    await evaluator.verify(
        claim="US citizens are eligible to apply for this fellowship program.",
        node=citizen_leaf,
        sources=valid_urls,
        additional_instruction=OFFICIAL_SOURCE_POLICY + " Look for an Eligibility section explicitly allowing US citizens.",
        extra_prerequisites=[ref_url_exists],
    )

    # PhD timeline check (Dec 2023 eligibility for 2026 cycle)
    timeline_leaf = evaluator.add_leaf(
        id=f"Program_{program_index+1}_PhD_Timeline_Check",
        desc="PhD completion in December 2023 falls within program's eligible timeframe",
        parent=elig_node,
        critical=True
    )
    await evaluator.verify(
        claim="A PhD awarded in December 2023 is within the program's eligible timeframe for the 2026 application cycle.",
        node=timeline_leaf,
        sources=valid_urls,
        additional_instruction=OFFICIAL_SOURCE_POLICY + " Check rules like 'PhD within N years of application' or specific date windows; evaluate Dec 2023 against the 2026 cycle.",
        extra_prerequisites=[ref_url_exists],
    )

    # Research area check
    research_leaf = evaluator.add_leaf(
        id=f"Program_{program_index+1}_Research_Area_Check",
        desc="Program supports observational, theoretical, or computational astronomy research",
        parent=elig_node,
        critical=True
    )
    await evaluator.verify(
        claim="This fellowship supports research in observational, theoretical, or computational astronomy/astrophysics.",
        node=research_leaf,
        sources=valid_urls,
        additional_instruction=OFFICIAL_SOURCE_POLICY + " Confirm support for astronomy/astrophysics research (observational/theoretical/computational).",
        extra_prerequisites=[ref_url_exists],
    )

    # ---------------------- Financial Package ---------------------- #
    fin_node = evaluator.add_parallel(
        id=f"Program_{program_index+1}_Financial_Package",
        desc="Financial support meets specified minimum requirements",
        parent=prog_node,
        critical=True
    )

    # Stipend verification (>= $75,000)
    stipend_leaf = evaluator.add_leaf(
        id=f"Program_{program_index+1}_Stipend_Verification",
        desc="Annual stipend amount is at least $75,000",
        parent=fin_node,
        critical=True
    )
    stipend_val = program.stipend or "unspecified"
    await evaluator.verify(
        claim=f"The fellowship's annual stipend is at least $75,000 (answer states: {stipend_val}).",
        node=stipend_leaf,
        sources=valid_urls,
        additional_instruction=OFFICIAL_SOURCE_POLICY + " Use explicit stipend numbers/ranges on the page to judge whether it meets or exceeds $75,000.",
        extra_prerequisites=[ref_url_exists],
    )

    # Research budget verification (>= $15,000)
    budget_leaf = evaluator.add_leaf(
        id=f"Program_{program_index+1}_Research_Budget_Verification",
        desc="Annual research budget or allowance is at least $15,000",
        parent=fin_node,
        critical=True
    )
    budget_val = program.research_budget or "unspecified"
    await evaluator.verify(
        claim=f"The annual research budget/allowance is at least $15,000 (answer states: {budget_val}).",
        node=budget_leaf,
        sources=valid_urls,
        additional_instruction=OFFICIAL_SOURCE_POLICY + " Confirm the page indicates a research budget/allowance meeting or exceeding $15,000 per year.",
        extra_prerequisites=[ref_url_exists],
    )

    # ---------------------- Timeline & Structure ---------------------- #
    time_node = evaluator.add_parallel(
        id=f"Program_{program_index+1}_Timeline_and_Structure",
        desc="Program timeline and structure meet requirements",
        parent=prog_node,
        critical=True
    )

    # Duration check (>= 2 years)
    duration_leaf = evaluator.add_leaf(
        id=f"Program_{program_index+1}_Duration_Check",
        desc="Fellowship duration is at least 2 years",
        parent=time_node,
        critical=True
    )
    duration_val = program.duration_years or "unspecified"
    await evaluator.verify(
        claim=f"The fellowship duration is at least 2 years (answer states: {duration_val}).",
        node=duration_leaf,
        sources=valid_urls,
        additional_instruction=OFFICIAL_SOURCE_POLICY + " Confirm total fellowship term from the program page.",
        extra_prerequisites=[ref_url_exists],
    )

    # Deadline verification (between 2026-02-22 and 2026-12-31)
    deadline_leaf = evaluator.add_leaf(
        id=f"Program_{program_index+1}_Deadline_Verification",
        desc="Application deadline occurs between February 22, 2026 and December 31, 2026",
        parent=time_node,
        critical=True
    )
    deadline_val = program.deadline_date or "unspecified"
    await evaluator.verify(
        claim=f"The 2026 application deadline occurs between {DEADLINE_START} and {DEADLINE_END} (answer states: {deadline_val}).",
        node=deadline_leaf,
        sources=valid_urls,
        additional_instruction=OFFICIAL_SOURCE_POLICY + " Verify the specific 2026 cycle deadline date from official sources and ensure it falls within the stated range.",
        extra_prerequisites=[ref_url_exists],
    )

    # ---------------------- Benefits ---------------------- #
    ben_node = evaluator.add_parallel(
        id=f"Program_{program_index+1}_Benefits",
        desc="Program provides required benefits",
        parent=prog_node,
        critical=True
    )

    health_leaf = evaluator.add_leaf(
        id=f"Program_{program_index+1}_Health_Insurance_Check",
        desc="Program provides or funds health insurance benefits",
        parent=ben_node,
        critical=True
    )
    await evaluator.verify(
        claim="The program provides or funds health insurance benefits for fellows.",
        node=health_leaf,
        sources=valid_urls,
        additional_instruction=OFFICIAL_SOURCE_POLICY + " Look for explicit mention of health insurance coverage or funding in the benefits section.",
        extra_prerequisites=[ref_url_exists],
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
    Evaluate an answer for identifying two astronomy/astrophysics postdoctoral fellowship programs
    that meet the specified criteria for a US citizen with PhD completion in December 2023.
    """
    # Initialize evaluator (root node is non-critical to allow partial credit aggregation)
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

    # Create a rubric-aligned top-level node (set non-critical to comply with framework constraints)
    task_completion_node = evaluator.add_parallel(
        id="Task_Completion",
        desc="Successfully identified two distinct named postdoctoral fellowship programs meeting all specified criteria with complete and accurate information",
        parent=root,
        critical=False
    )

    # Extract programs from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_programs(),
        template_class=FellowshipsExtraction,
        extraction_name="extracted_programs"
    )

    # Keep only first two programs; pad if fewer
    programs: List[FellowshipProgram] = list(extracted.programs[:2])
    while len(programs) < 2:
        programs.append(FellowshipProgram())

    # Add ground truth-like criteria expectations
    evaluator.add_ground_truth({
        "candidate_profile": CANDIDATE_PROFILE,
        "required_minimums": {
            "annual_stipend_usd": ">= 75,000",
            "annual_research_budget_usd": ">= 15,000",
            "duration_years": ">= 2",
            "deadline_window_2026": [DEADLINE_START, DEADLINE_END],
            "citizenship": "US citizens eligible",
            "research_area": "observational/theoretical/computational astronomy/astrophysics",
            "benefits": "health insurance provided or funded"
        }
    })

    evaluator.add_custom_info(
        info={"deadline_start": DEADLINE_START, "deadline_end": DEADLINE_END},
        info_type="constants",
        info_name="deadline_window_2026"
    )

    # Verify each program subtree
    for idx, prog in enumerate(programs):
        await verify_program(evaluator, task_completion_node, prog, idx)

    # Return structured summary
    return evaluator.get_summary()