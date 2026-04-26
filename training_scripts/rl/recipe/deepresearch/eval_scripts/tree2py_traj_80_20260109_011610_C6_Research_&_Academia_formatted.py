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
TASK_ID = "ai_ml_postdoc_2026"
TASK_DESCRIPTION = (
    "Identify four distinct postdoctoral fellowship programs in artificial intelligence or machine learning that meet ALL of the following criteria: "
    "(1) The program must be open to U.S. citizens who hold a PhD in computer science or a closely related computational field and have completed 2 years of postdoctoral training; "
    "(2) The program must support research at institutions within the University of California system, Big Ten Academic Alliance, Russell Group universities, or equivalent major research university networks; "
    "(3) The program must provide annual funding or stipend of at least $65,000; "
    "(4) The program must explicitly support or prioritize research in artificial intelligence, machine learning, or closely related computational fields; "
    "(5) The program must accept applications in 2026. "
    "For each fellowship program identified, provide the program name, a brief description, and a reference URL to the official program information page."
)


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class FellowshipProgram(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    official_url: Optional[str] = None
    extra_urls: List[str] = Field(default_factory=list)


class ProgramsExtraction(BaseModel):
    programs: List[FellowshipProgram] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_programs() -> str:
    return """
    Extract all fellowship program entries mentioned in the answer. For each entry, return:
    - name: the program name (string)
    - description: a brief description of the program as stated in the answer (string)
    - official_url: the official program information URL (must be a URL explicitly present in the answer; if none is present, set to null)
    - extra_urls: any other URLs cited for this program (array of URLs)
    
    IMPORTANT:
    - Extract every program mentioned in the answer (not just the first four).
    - Only include URLs explicitly present in the answer text (including markdown links). Do not invent URLs.
    - If any field is missing, set it to null (for name/description/official_url) or an empty array (for extra_urls).
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def _combine_sources(program: FellowshipProgram) -> List[str]:
    urls: List[str] = []
    if program.official_url and program.official_url.strip():
        urls.append(program.official_url.strip())
    if program.extra_urls:
        urls.extend([u for u in program.extra_urls if isinstance(u, str) and u.strip()])
    return urls


def _safe_text(val: Optional[str]) -> str:
    return val or ""


# --------------------------------------------------------------------------- #
# Verification logic for a single program                                     #
# --------------------------------------------------------------------------- #
async def verify_program(
    evaluator: Evaluator,
    parent_node,
    program: FellowshipProgram,
    program_index: int
) -> None:
    """
    Build and verify the subtree for a single fellowship program.
    Each check is a single binary verification step.
    """
    # Create the program node (parallel aggregation, non-critical to allow partial credit across programs)
    prog_node = evaluator.add_parallel(
        id=f"Fellowship_Program_{program_index}",
        desc=f"{program_index}th fellowship program entry meets all criteria and includes required fields" if program_index != 1
             else "1st fellowship program entry meets all criteria and includes required fields",
        parent=parent_node,
        critical=False
    )

    # Critical existence checks for name/description (as per rubric "Provides ...")
    name_node = evaluator.add_custom_node(
        result=bool(program.name and program.name.strip()),
        id=f"Program_Name_{program_index}",
        desc="Provides the program name.",
        parent=prog_node,
        critical=True
    )

    desc_node = evaluator.add_custom_node(
        result=bool(program.description and program.description.strip()),
        id=f"Program_Description_{program_index}",
        desc="Provides a brief description of the program.",
        parent=prog_node,
        critical=True
    )

    # Official URL presence and validity check
    official_url_leaf = evaluator.add_leaf(
        id=f"Official_URL_{program_index}",
        desc="Provides a valid URL to an official program information page.",
        parent=prog_node,
        critical=True
    )
    # Verify that the provided URL (if any) is indeed a program information page for the named fellowship
    await evaluator.verify(
        claim=f"The provided URL is an official program information page describing the fellowship program named '{_safe_text(program.name)}'. "
              f"If no URL is provided, this claim should be considered incorrect.",
        node=official_url_leaf,
        sources=program.official_url,
        additional_instruction=(
            "Assess whether the URL points to an official program page (e.g., on the host institution or program's official domain) "
            "that clearly describes the fellowship. If the URL is missing or obviously invalid, mark as incorrect."
        )
    )

    # Prepare shared sources for subsequent checks (official + extra)
    all_sources = _combine_sources(program)

    # Eligibility: U.S. citizens allowed (or no restrictions excluding U.S. citizens)
    us_citizen_leaf = evaluator.add_leaf(
        id=f"Eligibility_US_Citizen_Allowed_{program_index}",
        desc="Program is open to U.S. citizens OR has no citizenship restrictions that would exclude U.S. citizens.",
        parent=prog_node,
        critical=True
    )
    await evaluator.verify(
        claim="U.S. citizens are eligible to apply to this fellowship (or there are no citizenship restrictions that exclude U.S. citizens).",
        node=us_citizen_leaf,
        sources=all_sources,
        additional_instruction=(
            "Pass if the page states 'open to U.S. citizens' OR 'open to all nationalities' OR similar. "
            "Fail if it explicitly excludes U.S. citizens or restricts to a specific foreign citizenship."
        )
    )

    # Eligibility: PhD in CS or closely related computational field
    phd_field_leaf = evaluator.add_leaf(
        id=f"Eligibility_PhD_Field_{program_index}",
        desc="Program requires/accepts a completed PhD in computer science or a closely related computational field.",
        parent=prog_node,
        critical=True
    )
    await evaluator.verify(
        claim="Applicants must have (or the program accepts) a completed PhD in computer science or a closely related computational field.",
        node=phd_field_leaf,
        sources=all_sources,
        additional_instruction=(
            "Consider closely related computational fields such as computer engineering, electrical engineering (with computing/AI/ML focus), "
            "applied mathematics, statistics, data science, robotics, computational neuroscience, or information science."
        )
    )

    # Eligibility: Candidate with 2 years of postdoctoral training falls within eligible experience window
    two_years_postdoc_leaf = evaluator.add_leaf(
        id=f"Eligibility_2_Years_Postdoc_{program_index}",
        desc="A candidate with 2 years of postdoctoral training falls within the program's eligible experience window.",
        parent=prog_node,
        critical=True
    )
    await evaluator.verify(
        claim="A candidate with approximately two years of postdoctoral training is eligible under the program's stated experience/eligibility window.",
        node=two_years_postdoc_leaf,
        sources=all_sources,
        additional_instruction=(
            "Pass if eligibility includes early-career postdocs or sets windows like '0–5 years post-PhD' or similar that encompass 2 years. "
            "Fail if the program excludes postdocs at ~2 years (e.g., requires >2 years independent PI or sets a max below 2 years)."
        )
    )

    # Host institution network support (UC, BTAA, Russell Group, or equivalent major research university network)
    host_network_leaf = evaluator.add_leaf(
        id=f"Host_Institution_Network_{program_index}",
        desc="Program supports/allows research at institutions in the UC system, Big Ten Academic Alliance, Russell Group, or an equivalent major research university network.",
        parent=prog_node,
        critical=True
    )
    await evaluator.verify(
        claim="The program supports or allows research at institutions belonging to the University of California system, Big Ten Academic Alliance (BTAA), Russell Group, or an equivalent major research university network.",
        node=host_network_leaf,
        sources=all_sources,
        additional_instruction=(
            "Pass if the program is hosted by or allows placements at institutions in UC, BTAA, or Russell Group. "
            "Also pass if it clearly allows placements at equivalent major research university networks (e.g., AAU, Ivy League), indicating comparable stature. "
            "Fail if the program is limited to institutions outside such networks without equivalency."
        )
    )

    # Funding minimum of $65,000
    funding_leaf = evaluator.add_leaf(
        id=f"Funding_Min_65000_{program_index}",
        desc="Program provides annual funding/stipend of at least $65,000.",
        parent=prog_node,
        critical=True
    )
    await evaluator.verify(
        claim="The program provides annual funding or an annualized stipend of at least $65,000 (USD or equivalent).",
        node=funding_leaf,
        sources=all_sources,
        additional_instruction=(
            "Pass if the page lists a salary/stipend >= $65,000 per year or a range that includes amounts >= $65,000. "
            "If only monthly/weekly rates are given, consider the annualized amount. If currency is not USD, consider approximate equivalence."
        )
    )

    # AI/ML focus
    ai_ml_focus_leaf = evaluator.add_leaf(
        id=f"AI_ML_Focus_{program_index}",
        desc="Program explicitly supports or prioritizes AI/ML or closely related computational fields (as stated in the question/constraints).",
        parent=prog_node,
        critical=True
    )
    await evaluator.verify(
        claim="The program explicitly supports or prioritizes research in artificial intelligence, machine learning, or closely related computational fields.",
        node=ai_ml_focus_leaf,
        sources=all_sources,
        additional_instruction=(
            "Look for explicit mentions of AI, artificial intelligence, machine learning, deep learning, data science, computer vision, NLP, robotics with ML, "
            "or closely related computational disciplines. General STEM without computational focus is insufficient."
        )
    )

    # Accepts applications in 2026
    accepts_2026_leaf = evaluator.add_leaf(
        id=f"Accepts_Applications_2026_{program_index}",
        desc="Program accepts applications during calendar year 2026 (fixed deadline(s), rolling, or ongoing window).",
        parent=prog_node,
        critical=True
    )
    await evaluator.verify(
        claim="The program accepts applications during calendar year 2026 (either via fixed deadlines in 2026 or rolling/ongoing acceptance covering 2026).",
        node=accepts_2026_leaf,
        sources=all_sources,
        additional_instruction=(
            "Pass if the page shows an application cycle with deadlines in 2026 or states rolling/ongoing acceptance that includes 2026. "
            "Fail if deadlines are confined to 2025 or earlier and there's no indication of a 2026 cycle."
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
    """
    Evaluate an answer for the AI/ML postdoc fellowship program identification task.
    """
    # Initialize evaluator
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Root aggregates in parallel to allow partial credit on program entries
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

    # Extract all programs from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_programs(),
        template_class=ProgramsExtraction,
        extraction_name="programs_extraction"
    )

    total_programs = len(extracted.programs)
    evaluator.add_custom_info(
        info={"extracted_program_count": total_programs},
        info_type="stats",
        info_name="extraction_stats"
    )

    # Global requirements node (critical)
    set_requirements = evaluator.add_parallel(
        id="Program_Set_Requirements",
        desc="Global requirements about the set of programs returned",
        parent=root,
        critical=True
    )

    # Exactly four programs (critical)
    exactly_four_node = evaluator.add_custom_node(
        result=(total_programs == 4),
        id="Exactly_Four_Programs",
        desc="Response identifies exactly four fellowship programs (no more, no fewer).",
        parent=set_requirements,
        critical=True
    )

    # Programs are distinct (critical) - LLM checks distinctness
    # Build a readable list of program names for the claim
    names_for_claim = "; ".join([_safe_text(p.name) for p in extracted.programs[:4]])
    distinct_leaf = evaluator.add_leaf(
        id="Programs_Are_Distinct",
        desc="All four programs are distinct fellowship opportunities (not merely different tracks/components of the same program).",
        parent=set_requirements,
        critical=True
    )
    await evaluator.verify(
        claim=f"The listed programs are four distinct fellowship programs (not tracks of the same program): {names_for_claim}.",
        node=distinct_leaf,
        additional_instruction=(
            "Use names/descriptions to judge distinctness. Programs with the same overarching name but different tracks/components should be considered NOT distinct."
        )
    )

    # Build verification subtrees for up to the first four programs
    # Pad with empty placeholders if fewer than 4 were provided (to keep consistent structure)
    programs_for_eval: List[FellowshipProgram] = list(extracted.programs[:4])
    while len(programs_for_eval) < 4:
        programs_for_eval.append(FellowshipProgram())

    # Create each program node and verify
    for idx, program in enumerate(programs_for_eval, start=1):
        await verify_program(evaluator, root, program, idx)

    # Return the evaluation summary
    return evaluator.get_summary()