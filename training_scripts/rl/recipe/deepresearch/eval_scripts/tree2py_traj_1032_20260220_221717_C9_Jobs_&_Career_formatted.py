import asyncio
import logging
from datetime import datetime
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "nc_professional_programs"
TASK_DESCRIPTION = (
    "Identify three university professional preparation programs in North Carolina, each in a different field, "
    "that meet the following comprehensive criteria: (1) the program must be at a four-year university in North Carolina, "
    "(2) the program must prepare students for a specific professional license or certification, "
    "(3) the program must hold active specialized accreditation from the relevant professional accrediting body for its field, "
    "(4) the program must publicly report licensure/certification exam pass rates that meet or exceed applicable state or national standards, "
    "(5) the program must publicly report employment or continuing education outcomes for graduates within six months of graduation from a recent graduating class "
    "(within the past three years), and (6) the program must be approved by the relevant North Carolina professional board or state agency. "
    "For each program, provide: the university name, specific program name, professional field, degree level, specialized accreditation body and status, "
    "licensure/certification type and exam pass rate with performance standard met, employment/continuing education rate and timeframe, state approval authority, "
    "and source URLs documenting accreditation, exam performance, career outcomes, and state approval."
)

CURRENT_YEAR = datetime.utcnow().year
RECENT_YEARS = [CURRENT_YEAR - i for i in range(0, 3)]  # e.g., [2026, 2025, 2024] if CURRENT_YEAR=2026


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class ProgramSources(BaseModel):
    accreditation_urls: List[str] = Field(default_factory=list)
    exam_urls: List[str] = Field(default_factory=list)
    outcome_urls: List[str] = Field(default_factory=list)
    state_approval_urls: List[str] = Field(default_factory=list)


class ProgramInfo(BaseModel):
    university: Optional[str] = None
    program_name: Optional[str] = None
    professional_field: Optional[str] = None
    degree_level: Optional[str] = None

    accreditation_body: Optional[str] = None
    accreditation_status: Optional[str] = None

    licensure_type: Optional[str] = None
    exam_pass_rate: Optional[str] = None
    exam_standard_met: Optional[str] = None

    employment_rate: Optional[str] = None
    outcome_timeframe: Optional[str] = None

    state_approval_authority: Optional[str] = None

    sources: ProgramSources = Field(default_factory=ProgramSources)


class ProgramsExtraction(BaseModel):
    programs: List[ProgramInfo] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_programs() -> str:
    return (
        "Extract up to three university professional preparation programs described in the answer. "
        "For each program, return the following fields as a JSON object:\n"
        "- university: Name of the North Carolina four-year university offering the program\n"
        "- program_name: Specific degree program name\n"
        "- professional_field: Professional field (e.g., nursing, teaching, allied health, pharmacy, social work, etc.)\n"
        "- degree_level: Degree level offered (e.g., Bachelor's, Master's, Doctorate, etc.)\n"
        "- accreditation_body: Specialized accrediting body for the program (e.g., CCNE, CAEP, CAAHEP, ACPE, COARC, NASM/NASAD/NASM, etc.)\n"
        "- accreditation_status: The current accreditation status text (e.g., 'accredited', 'active', 'approved') exactly as stated in the answer\n"
        "- licensure_type: Specific professional license or certification the program prepares students to obtain\n"
        "- exam_pass_rate: The licensure/certification exam pass rate as stated (e.g., '95%', '90% first-time pass rate')\n"
        "- exam_standard_met: The applicable state or national performance standard met or exceeded (e.g., 'meets NC Board minimum 80% threshold')\n"
        "- employment_rate: Employment or continuing education rate within approximately six months of graduation (as stated)\n"
        "- outcome_timeframe: The timeframe for the outcomes and the graduating class year (e.g., 'within 6 months, Class of 2024')\n"
        "- state_approval_authority: The relevant North Carolina professional board or state agency approving the program\n"
        "- sources: Categorized source URLs explicitly mentioned in the answer text with the following arrays:\n"
        "    • accreditation_urls: URLs documenting accreditation status/body (official accreditor directory or official program page)\n"
        "    • exam_urls: URLs documenting licensure/certification exam performance\n"
        "    • outcome_urls: URLs documenting employment/continuing education outcomes\n"
        "    • state_approval_urls: URLs documenting NC board/state approval\n\n"
        "IMPORTANT:\n"
        "1) Only extract information explicitly mentioned in the answer; do not invent. If a field is missing, set it to null.\n"
        "2) For each URL list, include only URLs explicitly provided in the answer (plain URLs or markdown links). If none are provided, return an empty array.\n"
        "3) Preserve the exact phrasing for rates and statuses as stated (e.g., preserve '%' and qualifiers like 'first-time').\n"
        "4) If the answer lists more than three programs, include only the first three. If fewer than three are listed, return what's available.\n"
    )


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _non_empty(val: Optional[str]) -> bool:
    return bool(val and str(val).strip())


def _list_non_empty(urls: Optional[List[str]]) -> bool:
    return bool(urls and len(urls) > 0)


# --------------------------------------------------------------------------- #
# Verification for one program                                                #
# --------------------------------------------------------------------------- #
async def verify_program(
    evaluator: Evaluator,
    root_parent,
    program: ProgramInfo,
    index: int,
    prior_fields: List[str],
) -> None:
    """
    Build the verification subtree for a single program following the rubric.

    - index: 0-based program index (Program_1 => 0, Program_2 => 1, Program_3 => 2)
    - prior_fields: professional fields extracted for earlier programs, used to enforce distinct fields
    """
    prog_num = index + 1

    # Program node (sequential): Basic Identification first, then All Verifications
    program_node = evaluator.add_sequential(
        id=f"Program_{prog_num}",
        desc=f"{['First','Second','Third'][index]} qualifying professional preparation program",
        parent=root_parent,
        critical=False  # Keep non-critical to allow partial credit across programs
    )

    # --------------------- Basic Identification (parallel, critical) --------------------- #
    basic_node = evaluator.add_parallel(
        id=f"Program_{prog_num}_Basic_Identification",
        desc="Identify the university and specific program",
        parent=program_node,
        critical=True
    )

    # University (existence)
    evaluator.add_custom_node(
        result=_non_empty(program.university),
        id=f"Program_{prog_num}_University",
        desc="Provide the name of the North Carolina university offering this program",
        parent=basic_node,
        critical=True
    )

    # Program name (existence)
    evaluator.add_custom_node(
        result=_non_empty(program.program_name),
        id=f"Program_{prog_num}_Program_Name",
        desc="Provide the specific degree program name",
        parent=basic_node,
        critical=True
    )

    # Professional field (existence)
    evaluator.add_custom_node(
        result=_non_empty(program.professional_field),
        id=f"Program_{prog_num}_Professional_Field",
        desc="Identify the professional field",
        parent=basic_node,
        critical=True
    )

    # Degree level (existence)
    evaluator.add_custom_node(
        result=_non_empty(program.degree_level),
        id=f"Program_{prog_num}_Degree_Level",
        desc="Specify the degree level offered (Bachelor's, Master's, etc.)",
        parent=basic_node,
        critical=True
    )

    # Field uniqueness checks as separate leaves (simple verify without sources)
    if index == 1 and prior_fields and _non_empty(program.professional_field):
        # Must be different from Program 1
        node_diff_p1 = evaluator.add_leaf(
            id=f"Program_{prog_num}_Field_Different_From_Program_1",
            desc="Program 2 field differs from Program 1",
            parent=basic_node,
            critical=True
        )
        claim = (
            f"The professional field '{program.professional_field}' is different from '{prior_fields[0]}'. "
            "Allow reasonable synonyms to be considered the same and treat them as NOT different."
        )
        await evaluator.verify(
            claim=claim,
            node=node_diff_p1,
            additional_instruction="Judge difference semantically, not just by wording. If they refer to the same profession despite minor phrasing, consider them the same."
        )

    if index == 2 and _non_empty(program.professional_field):
        # Must be different from Program 1 and Program 2 (two leaves)
        if len(prior_fields) >= 1:
            node_diff_p1 = evaluator.add_leaf(
                id=f"Program_{prog_num}_Field_Different_From_Program_1",
                desc="Program 3 field differs from Program 1",
                parent=basic_node,
                critical=True
            )
            claim = (
                f"The professional field '{program.professional_field}' is different from '{prior_fields[0]}'. "
                "Allow reasonable synonyms to be considered the same and treat them as NOT different."
            )
            await evaluator.verify(
                claim=claim,
                node=node_diff_p1,
                additional_instruction="Judge difference semantically, not just by wording. If they refer to the same profession despite minor phrasing, consider them the same."
            )
        if len(prior_fields) >= 2:
            node_diff_p2 = evaluator.add_leaf(
                id=f"Program_{prog_num}_Field_Different_From_Program_2",
                desc="Program 3 field differs from Program 2",
                parent=basic_node,
                critical=True
            )
            claim = (
                f"The professional field '{program.professional_field}' is different from '{prior_fields[1]}'. "
                "Allow reasonable synonyms to be considered the same and treat them as NOT different."
            )
            await evaluator.verify(
                claim=claim,
                node=node_diff_p2,
                additional_instruction="Judge difference semantically, not just by wording. If they refer to the same profession despite minor phrasing, consider them the same."
            )

    # --------------------- All Verifications (parallel, critical) --------------------- #
    all_verif_node = evaluator.add_parallel(
        id=f"Program_{prog_num}_All_Verifications",
        desc="Verify all program requirements in parallel",
        parent=program_node,
        critical=True
    )

    # ---- Accreditation Verification (parallel, critical) ---- #
    accred_node = evaluator.add_parallel(
        id=f"Program_{prog_num}_Accreditation_Verification",
        desc="Verify the program meets accreditation requirements",
        parent=all_verif_node,
        critical=True
    )

    # Accreditation source URL existence (critical sibling first)
    evaluator.add_custom_node(
        result=_list_non_empty(program.sources.accreditation_urls),
        id=f"Program_{prog_num}_Accreditation_Source_URL",
        desc="Provide URL documenting the accreditation status",
        parent=accred_node,
        critical=True
    )

    # Specialized accreditation (verify by URLs)
    spec_acc_leaf = evaluator.add_leaf(
        id=f"Program_{prog_num}_Specialized_Accreditation",
        desc="Confirm the program holds specialized accreditation from the relevant professional accrediting body for its field",
        parent=accred_node,
        critical=True
    )
    claim_acc = (
        f"The program '{program.program_name}' at '{program.university}' is accredited by '{program.accreditation_body}'. "
        "Minor naming variations or abbreviations should be accepted if they clearly refer to the same accrediting body."
    )
    await evaluator.verify(
        claim=claim_acc,
        node=spec_acc_leaf,
        sources=program.sources.accreditation_urls,
        additional_instruction="Use the accrediting body’s official directory or authoritative listing; a university page explicitly citing the accreditor also suffices."
    )

    # Accreditation status active (verify by URLs)
    status_leaf = evaluator.add_leaf(
        id=f"Program_{prog_num}_Accreditation_Status",
        desc="Verify the current accreditation status is active/approved",
        parent=accred_node,
        critical=True
    )
    claim_status = (
        "The accreditation status for this program is currently active/approved (i.e., accredited). "
        "Accept equivalent phrasings like 'currently accredited', 'active accreditation', or 'approved'."
    )
    await evaluator.verify(
        claim=claim_status,
        node=status_leaf,
        sources=program.sources.accreditation_urls,
        additional_instruction="Confirm status wording indicates present, active accreditation rather than historical or lapsed status."
    )

    # ---- Licensure Preparation (parallel, critical) ---- #
    lic_node = evaluator.add_parallel(
        id=f"Program_{prog_num}_Licensure_Preparation",
        desc="Verify the program prepares students for professional licensure/certification",
        parent=all_verif_node,
        critical=True
    )

    # Exam source URL existence (critical sibling first)
    evaluator.add_custom_node(
        result=_list_non_empty(program.sources.exam_urls),
        id=f"Program_{prog_num}_Exam_Source_URL",
        desc="Provide URL documenting the exam performance data",
        parent=lic_node,
        critical=True
    )

    # Licensure type supported by sources
    lic_type_leaf = evaluator.add_leaf(
        id=f"Program_{prog_num}_Licensure_Type",
        desc="Identify the specific professional license or certification the program prepares students to obtain",
        parent=lic_node,
        critical=True
    )
    claim_lic = (
        f"The program '{program.program_name}' at '{program.university}' prepares students for '{program.licensure_type}'. "
        "Accept equivalent license/certification names or common abbreviations when clearly referring to the same credential."
    )
    await evaluator.verify(
        claim=claim_lic,
        node=lic_type_leaf,
        sources=program.sources.exam_urls,
        additional_instruction="Use program/licensure pages or exam performance reports that explicitly tie the program to the specific credential."
    )

    # Exam performance (rate)
    exam_perf_leaf = evaluator.add_leaf(
        id=f"Program_{prog_num}_Exam_Performance",
        desc="Provide the program's licensure/certification exam pass rate",
        parent=lic_node,
        critical=True
    )
    claim_exam = (
        f"The reported licensure/certification exam pass rate for this program is '{program.exam_pass_rate}'. "
        "Allow minor numeric rounding differences to be considered equivalent."
    )
    await evaluator.verify(
        claim=claim_exam,
        node=exam_perf_leaf,
        sources=program.sources.exam_urls,
        additional_instruction="Look for clear pass rate statements (e.g., 'first-time pass rate', 'overall pass rate') and match the reported figure."
    )

    # Exam performance standard met
    exam_std_leaf = evaluator.add_leaf(
        id=f"Program_{prog_num}_Exam_Performance_Standard",
        desc="Verify the pass rate meets or exceeds the applicable state or national standard",
        parent=lic_node,
        critical=True
    )
    claim_std = (
        f"The program's exam pass rate meets or exceeds the applicable standard: '{program.exam_standard_met}'. "
        "If the source provides a threshold or explicitly states compliance, that suffices."
    )
    await evaluator.verify(
        claim=claim_std,
        node=exam_std_leaf,
        sources=program.sources.exam_urls,
        additional_instruction="If a numeric threshold is present, compare the reported pass rate; otherwise, accept explicit claims of meeting or exceeding the standard."
    )

    # ---- Career Outcomes (parallel, critical) ---- #
    outcomes_node = evaluator.add_parallel(
        id=f"Program_{prog_num}_Career_Outcomes",
        desc="Verify the program reports career outcome metrics",
        parent=all_verif_node,
        critical=True
    )

    # Outcome source URL existence (critical sibling first)
    evaluator.add_custom_node(
        result=_list_non_empty(program.sources.outcome_urls),
        id=f"Program_{prog_num}_Outcome_Source_URL",
        desc="Provide URL documenting the career outcome data",
        parent=outcomes_node,
        critical=True
    )

    # Employment or continuing education rate
    emp_rate_leaf = evaluator.add_leaf(
        id=f"Program_{prog_num}_Employment_Rate",
        desc="Provide the employment or continuing education rate within 6 months of graduation",
        parent=outcomes_node,
        critical=True
    )
    claim_emp = (
        f"The program reports an employment or continuing education rate of '{program.employment_rate}' within approximately six months of graduation."
    )
    await evaluator.verify(
        claim=claim_emp,
        node=emp_rate_leaf,
        sources=program.sources.outcome_urls,
        additional_instruction="Confirm the percentage and that the timeframe is approximately within 6 months."
    )

    # Outcome timeframe recent (past three years)
    timeframe_leaf = evaluator.add_leaf(
        id=f"Program_{prog_num}_Outcome_Timeframe",
        desc="Confirm the outcome data is from a recent graduating class (within past 3 years)",
        parent=outcomes_node,
        critical=True
    )
    years_str = ", ".join(str(y) for y in sorted(RECENT_YEARS, reverse=True))
    claim_time = (
        f"The outcomes are reported for a graduating class within the past three years (acceptable class years include: {years_str}). "
        f"The reported timeframe is '{program.outcome_timeframe}'."
    )
    await evaluator.verify(
        claim=claim_time,
        node=timeframe_leaf,
        sources=program.sources.outcome_urls,
        additional_instruction="Look for explicit graduating class years or dates indicating recency within ~3 years and a 6‑month post‑graduation window."
    )

    # ---- State Approval (parallel, critical) ---- #
    state_node = evaluator.add_parallel(
        id=f"Program_{prog_num}_State_Approval",
        desc="Verify the program meets North Carolina state requirements",
        parent=all_verif_node,
        critical=True
    )

    # State approval source URL existence (critical sibling first)
    evaluator.add_custom_node(
        result=_list_non_empty(program.sources.state_approval_urls),
        id=f"Program_{prog_num}_State_Authorization_URL",
        desc="Provide URL or documentation source for state approval",
        parent=state_node,
        critical=True
    )

    # State authorization claim
    state_auth_leaf = evaluator.add_leaf(
        id=f"Program_{prog_num}_State_Authorization",
        desc="Confirm the program is approved by the relevant North Carolina professional board or state agency",
        parent=state_node,
        critical=True
    )
    claim_state = (
        f"The program '{program.program_name}' at '{program.university}' is approved by '{program.state_approval_authority}'. "
        "Approval may be demonstrated via official listings, approval letters, or authoritative state board pages."
    )
    await evaluator.verify(
        claim=claim_state,
        node=state_auth_leaf,
        sources=program.sources.state_approval_urls,
        additional_instruction="Confirm listing or explicit approval for the specific program (not just institution-level authorization)."
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
    Evaluate the answer for the North Carolina professional preparation programs task.

    Returns a structured summary including the verification tree and aggregated score.
    """
    evaluator = Evaluator()

    # IMPORTANT: Although the JSON marked the root as critical, obj_task_eval enforces that
    # a critical parent cannot have non-critical children. Since each Program node is
    # non-critical (to allow partial credit across programs), we set the root to non-critical.
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Programs evaluated independently
        agent_name=agent_name,
        answer_name=answer_name,
        client=client,
        task_description=(
            "Identify three university programs in North Carolina that prepare students for professional licensure/certification in high-demand fields, "
            "meet accreditation standards, and demonstrate strong career outcomes"
        ),
        answer=answer,
        global_cache=cache,
        global_semaphore=semaphore,
        logger=logger,
        default_model=model,
    )

    # Extract programs
    extraction = await evaluator.extract(
        prompt=prompt_extract_programs(),
        template_class=ProgramsExtraction,
        extraction_name="programs_extraction",
    )

    # Take first three programs; pad with empty ProgramInfo if fewer
    programs: List[ProgramInfo] = list(extraction.programs[:3])
    while len(programs) < 3:
        programs.append(ProgramInfo())

    # Track prior fields for uniqueness checks
    prior_fields: List[str] = []
    for i, prog in enumerate(programs):
        # Record prior field before verifying next
        await verify_program(
            evaluator=evaluator,
            root_parent=root,
            program=prog,
            index=i,
            prior_fields=prior_fields.copy(),
        )
        # Update prior fields
        if _non_empty(prog.professional_field):
            prior_fields.append(prog.professional_field.strip())

    # Return structured summary
    return evaluator.get_summary()