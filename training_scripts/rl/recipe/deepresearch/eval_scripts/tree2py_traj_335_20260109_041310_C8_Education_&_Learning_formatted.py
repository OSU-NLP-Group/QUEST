import asyncio
import logging
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.llm_client.base_client import LLMClient


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "aacsb_online_mba_tx_ca_fl"
TASK_DESCRIPTION = (
    "Identify three AACSB-accredited online MBA programs in the United States, with one program from each of the following states: "
    "Texas, California, and Florida. Each program must meet all of the following criteria: (1) The program must be accredited by AACSB International; "
    "(2) The program must be offered 100% online with no required campus visits; (3) The program must be completable in 36 months or less; "
    "(4) The program must offer GMAT/GRE test score waivers for applicants with sufficient professional work experience; "
    "(5) The program's minimum undergraduate GPA requirement for admission must be 3.0 or lower; "
    "(6) The program must require 48 credit hours or fewer to complete the degree; "
    "(7) The program must offer at least 3 different specialization or concentration options; "
    "(8) The total program tuition must be $40,000 or less for the entire degree; "
    "(9) The program must be ranked in the top 50 for online MBA programs by at least one major national ranking organization "
    "(such as U.S. News & World Report, The Princeton Review, or Financial Times). "
    "For each of the three programs, provide the name of the university, the program name, verification that each criterion is met, "
    "and reference URLs supporting your findings."
)


# --------------------------------------------------------------------------- #
# Extraction models                                                           #
# --------------------------------------------------------------------------- #
class ProgramInfo(BaseModel):
    state: Optional[str] = None
    university: Optional[str] = None
    program_name: Optional[str] = None
    reference_urls: List[str] = Field(default_factory=list)

    # Optional, descriptive fields (strings for robustness)
    online_format_note: Optional[str] = None
    duration_note: Optional[str] = None
    gmat_waiver_note: Optional[str] = None
    gpa_requirement_note: Optional[str] = None
    credit_hours_note: Optional[str] = None
    tuition_total_note: Optional[str] = None
    specializations: List[str] = Field(default_factory=list)
    ranking_note: Optional[str] = None
    ranking_urls: List[str] = Field(default_factory=list)


class ProgramsExtraction(BaseModel):
    texas: Optional[ProgramInfo] = None
    california: Optional[ProgramInfo] = None
    florida: Optional[ProgramInfo] = None


# --------------------------------------------------------------------------- #
# Extraction prompt helpers                                                   #
# --------------------------------------------------------------------------- #
def prompt_extract_programs() -> str:
    return """
Extract exactly one online MBA program for each of the following states from the provided answer text (do not infer or invent):
- texas
- california
- florida

For each state, return a JSON object with these fields:
- state: The U.S. state associated with the university (must be one of Texas, California, Florida)
- university: The university name offering the online MBA
- program_name: The official name of the online MBA program
- reference_urls: An array of all URLs cited in the answer that support any of the claims (program page, tuition, credit hours, format, AACSB accreditation, GMAT/GRE waiver, GPA, specializations, ranking, etc.). Only include valid full URLs that appear in the answer.
- online_format_note: A short snippet from the answer describing the online delivery (if provided)
- duration_note: A short snippet indicating time-to-completion (if provided)
- gmat_waiver_note: A short snippet indicating GMAT/GRE waivers for work experience (if provided)
- gpa_requirement_note: A short snippet indicating the minimum GPA (if provided)
- credit_hours_note: A short snippet indicating total credits (if provided)
- tuition_total_note: A short snippet indicating total program tuition (if provided)
- specializations: List up to 10 specialization/concentration names stated in the answer (if provided)
- ranking_note: A short snippet stating a top-50 online MBA ranking (if provided)
- ranking_urls: An array of any ranking URLs cited in the answer (if provided)

Important:
- Do NOT fabricate information. Only extract what is explicitly stated in the answer.
- If a field is not mentioned, set it to null (or [] for arrays).
- For URLs, extract the actual links (including protocol). If none are provided, return an empty array.
- For each state, if the answer does not include a program that matches, return null for that state.

Return the JSON with top-level keys: texas, california, florida, each mapping to the described object or null.
    """.strip()


# --------------------------------------------------------------------------- #
# Verification helper                                                         #
# --------------------------------------------------------------------------- #
def _program_sources(program: Optional[ProgramInfo]) -> List[str]:
    if not program:
        return []
    # Merge reference URLs and ranking URLs; de-duplicate while preserving order
    seen = set()
    merged: List[str] = []
    for url in (program.reference_urls or []):
        if url and url not in seen:
            merged.append(url)
            seen.add(url)
    for url in (program.ranking_urls or []):
        if url and url not in seen:
            merged.append(url)
            seen.add(url)
    return merged


async def _verify_program(
    evaluator: Evaluator,
    parent_node,
    program: Optional[ProgramInfo],
    program_label: str,  # "P1", "P2", "P3"
    target_state: str,   # "Texas" | "California" | "Florida"
    program_node_desc: str
) -> None:
    """
    Build and execute verification nodes for a single state's program.
    """
    # Create the program node (parallel aggregation, non-critical)
    prog_node = evaluator.add_parallel(
        id=f"{program_label}_node",
        desc=program_node_desc,
        parent=parent_node,
        critical=False
    )

    # Extract fields safely
    state = (program.state if program and program.state else None)
    university = (program.university if program and program.university else None)
    prog_name = (program.program_name if program and program.program_name else None)
    sources = _program_sources(program)

    # 1) Reference URLs presence (critical and evaluated first to gate others)
    urls_present = bool(sources)
    urls_node = evaluator.add_custom_node(
        result=urls_present,
        id=f"{program_label}_Reference_URLs",
        desc="Provide reference URL(s) supporting the above claims for this program",
        parent=prog_node,
        critical=True
    )

    # 2) University name present (critical)
    uni_present = bool(university and str(university).strip())
    uni_node = evaluator.add_custom_node(
        result=uni_present,
        id=f"{program_label}_University_Name",
        desc="Provide the name of the university offering the program",
        parent=prog_node,
        critical=True
    )

    # 3) Program name present (critical)
    pname_present = bool(prog_name and str(prog_name).strip())
    pname_node = evaluator.add_custom_node(
        result=pname_present,
        id=f"{program_label}_Program_Name",
        desc="Provide the program name (online MBA program name as offered by the university)",
        parent=prog_node,
        critical=True
    )

    # Helper: Common instructions that apply to every check
    base_instruction = (
        f"University: {university or 'Unknown'}; Program: {prog_name or 'Unknown'}; Target state: {target_state}.\n"
        "Use only the provided URLs to verify the claim. If the relevant information cannot be found on the provided URLs, "
        "mark the claim as not supported. Prefer official program/university or AACSB pages for accreditation; for ranking, "
        "prefer ranking organization pages (e.g., U.S. News). Allow minor wording variations. If a residency exists but is explicitly optional, "
        "that counts as 'no required campus visits'."
    )

    # 4) State verification (critical)
    state_leaf = evaluator.add_leaf(
        id=f"{program_label}_State_Verification",
        desc=f"Verify the university/program is located in {target_state} (USA)",
        parent=prog_node,
        critical=True
    )
    state_claim = (
        f"The university '{university or 'Unknown University'}' offering the program '{prog_name or 'Online MBA'}' "
        f"is located in the U.S. state of {target_state}."
    )
    await evaluator.verify(
        claim=state_claim,
        node=state_leaf,
        sources=sources,
        additional_instruction=(
            base_instruction
            + " Confirm the state location (campus/main campus) belongs to the specified state. "
              "If multiple campuses exist, it suffices that the business school/program campus is in the target state."
        )
    )

    # 5) AACSB accreditation (critical)
    aacsb_leaf = evaluator.add_leaf(
        id=f"{program_label}_AACSB_Accreditation",
        desc="Verify the program (or the business school offering it) is AACSB-accredited",
        parent=prog_node,
        critical=True
    )
    aacsb_claim = (
        "The business school offering this online MBA is accredited by AACSB International "
        "(Association to Advance Collegiate Schools of Business)."
    )
    await evaluator.verify(
        claim=aacsb_claim,
        node=aacsb_leaf,
        sources=sources,
        additional_instruction=(
            base_instruction
            + " Accept explicit statements on the school's site or AACSB's official directory that the business school is AACSB-accredited. "
              "Program-level or school-level AACSB accreditation both count."
        )
    )

    # 6) Online format 100% with no required campus visits (critical)
    online_leaf = evaluator.add_leaf(
        id=f"{program_label}_Online_Format",
        desc="Verify the program is offered 100% online with no required campus visits",
        parent=prog_node,
        critical=True
    )
    online_claim = (
        "The program is delivered 100% online and does not require any campus visits or in-person residencies."
    )
    await evaluator.verify(
        claim=online_claim,
        node=online_leaf,
        sources=sources,
        additional_instruction=(
            base_instruction
            + " Confirm that the program is fully online. Optional residencies are acceptable; required residencies are NOT."
        )
    )

    # 7) Duration: 36 months or less (critical)
    duration_leaf = evaluator.add_leaf(
        id=f"{program_label}_Duration",
        desc="Verify the program can be completed in 36 months or less",
        parent=prog_node,
        critical=True
    )
    duration_claim = "The program can be completed in 36 months or less."
    await evaluator.verify(
        claim=duration_claim,
        node=duration_leaf,
        sources=sources,
        additional_instruction=(
            base_instruction
            + " Accept statements like 'complete in as few as 12-24 months' or any explicit timeframe ≤ 36 months."
        )
    )

    # 8) GMAT/GRE waiver for sufficient work experience (critical)
    gmat_leaf = evaluator.add_leaf(
        id=f"{program_label}_GMAT_Waiver",
        desc="Verify the program offers GMAT/GRE test score waivers for applicants with professional work experience",
        parent=prog_node,
        critical=True
    )
    gmat_claim = (
        "The program offers GMAT or GRE test score waivers for applicants who have sufficient professional work experience."
    )
    await evaluator.verify(
        claim=gmat_claim,
        node=gmat_leaf,
        sources=sources,
        additional_instruction=(
            base_instruction
            + " Look for policy pages indicating work-experience-based GMAT/GRE waiver options. If waivers exist for other reasons only (not work experience), "
              "this does not satisfy the requirement."
        )
    )

    # 9) GPA requirement: 3.0 or lower (critical)
    gpa_leaf = evaluator.add_leaf(
        id=f"{program_label}_GPA_Requirement",
        desc="Verify the minimum undergraduate GPA requirement is 3.0 or lower",
        parent=prog_node,
        critical=True
    )
    gpa_claim = "The minimum undergraduate GPA required for admission is 3.0 or lower."
    await evaluator.verify(
        claim=gpa_claim,
        node=gpa_leaf,
        sources=sources,
        additional_instruction=(
            base_instruction
            + " Confirm an explicit minimum GPA threshold; 'preferred' GPAs do not count unless clearly stated as minimum. "
              "Accept thresholds like 3.0, 2.75, or 2.5."
        )
    )

    # 10) Credit hours: 48 or fewer (critical)
    credits_leaf = evaluator.add_leaf(
        id=f"{program_label}_Credit_Hours",
        desc="Verify the program requires 48 credit hours or fewer",
        parent=prog_node,
        critical=True
    )
    credits_claim = "The program requires 48 credit hours or fewer to complete the degree."
    await evaluator.verify(
        claim=credits_claim,
        node=credits_leaf,
        sources=sources,
        additional_instruction=(
            base_instruction
            + " Confirm total credit hours or units required. If a range or multiple tracks exist, accept if any standard completion path is ≤ 48 credits."
        )
    )

    # 11) Specializations: at least 3 (critical)
    specs_leaf = evaluator.add_leaf(
        id=f"{program_label}_Specializations",
        desc="Verify the program offers at least 3 specialization/concentration options",
        parent=prog_node,
        critical=True
    )
    specs_claim = "The program offers at least 3 different specializations or concentration options."
    await evaluator.verify(
        claim=specs_claim,
        node=specs_leaf,
        sources=sources,
        additional_instruction=(
            base_instruction
            + " Count any clearly listed concentrations or tracks. If the page lists three or more named options, this passes."
        )
    )

    # 12) Total tuition: $40,000 or less (critical)
    tuition_leaf = evaluator.add_leaf(
        id=f"{program_label}_Tuition",
        desc="Verify the total program tuition is $40,000 or less for the entire degree",
        parent=prog_node,
        critical=True
    )
    tuition_claim = "The total tuition for the entire online MBA program is $40,000 or less."
    await evaluator.verify(
        claim=tuition_claim,
        node=tuition_leaf,
        sources=sources,
        additional_instruction=(
            base_instruction
            + " If only per-credit tuition is given, multiply by total credits to estimate. Exclude fees if only tuition is clearly specified. "
              "If multiple rates exist (in-state/out-of-state), accept if at least one applicable published tuition total is ≤ $40,000."
        )
    )

    # 13) Ranking: top 50 for online MBA by at least one major org (critical)
    ranking_leaf = evaluator.add_leaf(
        id=f"{program_label}_Ranking",
        desc="Verify the program is ranked in the top 50 online MBA programs by at least one of: U.S. News & World Report, The Princeton Review, or Financial Times",
        parent=prog_node,
        critical=True
    )
    ranking_claim = (
        "This online MBA program is ranked within the top 50 for online MBA programs by at least one of the following: "
        "U.S. News & World Report, The Princeton Review, or Financial Times."
    )
    ranking_specific_sources = program.ranking_urls if (program and program.ranking_urls) else sources
    await evaluator.verify(
        claim=ranking_claim,
        node=ranking_leaf,
        sources=ranking_specific_sources,
        additional_instruction=(
            base_instruction
            + " Verify that the ranking pertains specifically to Online MBA programs and shows a rank between 1 and 50. "
              "Accept ranking pages from the listed organizations only."
        )
    )


# --------------------------------------------------------------------------- #
# Main evaluation entry point                                                 #
# --------------------------------------------------------------------------- #
async def evaluate_answer(
    client: LLMClient,
    answer: str,
    agent_name: str,
    answer_name: str,
    cache: CacheFileSys,
    semaphore: asyncio.Semaphore,
    logger: logging.Logger,
    model: str = "o4-mini",
) -> Dict:
    """
    Evaluate an answer for the AACSB-accredited online MBA program selection task across Texas, California, and Florida.
    """
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Root combines three independent state programs
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

    # Extract structured programs info from the answer
    extracted: ProgramsExtraction = await evaluator.extract(
        prompt=prompt_extract_programs(),
        template_class=ProgramsExtraction,
        extraction_name="programs_by_state",
    )

    # Record helpful GT/context info for transparency (not used for scoring)
    evaluator.add_ground_truth(
        {
            "required_states": ["Texas", "California", "Florida"],
            "constraints": [
                "AACSB accredited",
                "100% online, no required campus visits",
                "≤ 36 months",
                "GMAT/GRE waiver for sufficient work experience",
                "Min GPA ≤ 3.0",
                "Credit hours ≤ 48",
                "≥ 3 specializations",
                "Total tuition ≤ $40,000",
                "Top 50 online MBA ranking by U.S. News, Princeton Review, or Financial Times",
            ],
        },
        gt_type="task_requirements",
    )

    # Build three parallel program subtrees (all non-critical at program level for partial credit)
    # Program 1: Texas
    await _verify_program(
        evaluator=evaluator,
        parent_node=root,
        program=extracted.texas,
        program_label="P1",
        target_state="Texas",
        program_node_desc="Online MBA program from a Texas university meeting all specified criteria",
    )

    # Program 2: California
    await _verify_program(
        evaluator=evaluator,
        parent_node=root,
        program=extracted.california,
        program_label="P2",
        target_state="California",
        program_node_desc="Online MBA program from a California university meeting all specified criteria",
    )

    # Program 3: Florida
    await _verify_program(
        evaluator=evaluator,
        parent_node=root,
        program=extracted.florida,
        program_label="P3",
        target_state="Florida",
        program_node_desc="Online MBA program from a Florida university meeting all specified criteria",
    )

    # Return evaluation summary
    return evaluator.get_summary()