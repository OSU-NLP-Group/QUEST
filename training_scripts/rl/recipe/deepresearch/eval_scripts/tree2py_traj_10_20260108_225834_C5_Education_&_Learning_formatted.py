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
TASK_ID = "cacrep_programs_se_us"
TASK_DESCRIPTION = (
    "Find three CACREP-accredited online master's degree programs in Clinical Mental Health Counseling offered by "
    "universities located in the Southeastern United States, specifically in Florida, Georgia, or North Carolina. "
    "For each program, provide the following information:\n\n"
    "1. Institution Name and Program Name: The full name of the university and the specific degree program title\n"
    "2. CACREP Accreditation Verification: Confirm that the program holds current CACREP accreditation for the Clinical "
    "Mental Health Counseling specialty, and provide a link to the program's listing in the official CACREP directory "
    "(https://www.cacrep.org/directory/)\n"
    "3. Total Credit Hours: State the total number of credit hours required to complete the program, which must meet the "
    "CACREP minimum requirement of 60 semester credit hours (or 90 quarter credit hours)\n"
    "4. Delivery Format: Specify whether the online program is delivered in a synchronous format (scheduled live virtual "
    "sessions), asynchronous format (self-paced without live sessions), or hybrid format (combination of both)\n"
    "5. Tuition Cost: Provide the current tuition cost, either as cost per credit hour or total program cost\n"
    "6. Minimum GPA Requirement: State the minimum undergraduate GPA required for admission to the program\n"
    "7. Standardized Test Requirement: Indicate whether the program requires GRE or GMAT scores, or if these tests are "
    "waived under certain conditions\n\n"
    "For each piece of information, include a direct URL reference to the source webpage where this information can be verified."
)

ALLOWED_STATES = {"florida", "ga", "georgia", "nc", "north carolina", "fl"}

# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class ProgramItem(BaseModel):
    institution_name: Optional[str] = None
    program_name: Optional[str] = None
    program_url: Optional[str] = None

    state: Optional[str] = None
    state_source_url: Optional[str] = None

    cacrep_directory_url: Optional[str] = None

    credit_hours: Optional[str] = None
    credit_hours_source_url: Optional[str] = None

    delivery_format: Optional[str] = None
    delivery_format_source_url: Optional[str] = None

    tuition_cost: Optional[str] = None
    tuition_source_url: Optional[str] = None

    min_gpa: Optional[str] = None
    min_gpa_source_url: Optional[str] = None

    test_policy: Optional[str] = None
    test_policy_source_url: Optional[str] = None

    practicum_hours: Optional[str] = None
    practicum_source_url: Optional[str] = None

    internship_hours: Optional[str] = None
    internship_source_url: Optional[str] = None

    regional_accreditation: Optional[str] = None
    regional_accreditation_source_url: Optional[str] = None


class ProgramsExtraction(BaseModel):
    programs: List[ProgramItem] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_programs() -> str:
    return (
        "Extract up to three CACREP-accredited online or online-blended master's programs in Clinical Mental Health Counseling "
        "as presented in the answer. For each program, return a JSON object with the following fields and their direct source URLs "
        "from the answer:\n"
        "- institution_name: Full university name\n"
        "- program_name: Full degree/program title\n"
        "- program_url: Direct URL to the program page or official university page describing the program\n"
        "- state: The U.S. state where the university is located (e.g., Florida, Georgia, or North Carolina). If an abbreviation "
        "is used (FL, GA, NC), extract that\n"
        "- state_source_url: Direct URL that verifies the institution's location/state\n"
        "- cacrep_directory_url: Direct URL to the program's listing in CACREP's official directory page\n"
        "- credit_hours: Total required credits (e.g., '60 credits', '63 semester hours', '90 quarter hours')\n"
        "- credit_hours_source_url: Direct URL that states the total required credits\n"
        "- delivery_format: One of 'synchronous', 'asynchronous', 'hybrid', or a descriptive phrase; extract what the answer states\n"
        "- delivery_format_source_url: Direct URL that describes the delivery format\n"
        "- tuition_cost: Current tuition (either per-credit amount or total program cost); extract the text as shown\n"
        "- tuition_source_url: Direct URL to tuition information\n"
        "- min_gpa: Minimum undergraduate GPA required for admission (e.g., '3.0', '2.75')\n"
        "- min_gpa_source_url: Direct URL that states the minimum GPA\n"
        "- test_policy: GRE/GMAT requirement or waiver policy stated (e.g., 'GRE not required', 'GRE waived for GPA ≥3.0')\n"
        "- test_policy_source_url: Direct URL that states standardized test policy\n"
        "- practicum_hours: Practicum hours requirement (e.g., '100 hours')\n"
        "- practicum_source_url: Direct URL that states practicum hours requirement\n"
        "- internship_hours: Internship hours requirement (e.g., '600 hours')\n"
        "- internship_source_url: Direct URL that states internship hours requirement\n"
        "- regional_accreditation: Name of the institution's regional accreditor (e.g., 'SACSCOC')\n"
        "- regional_accreditation_source_url: Direct URL that verifies regional accreditation\n\n"
        "Return a JSON object with a 'programs' array of up to three program objects. If an item or URL is not present in the answer, "
        "set the value to null."
    )


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def _norm_text(s: Optional[str]) -> str:
    if not s:
        return ""
    return "".join(ch.lower() for ch in s if ch.isalnum() or ch.isspace()).strip()


def _is_allowed_state(state_text: Optional[str]) -> bool:
    if not state_text:
        return False
    st = state_text.strip().lower()
    return st in ALLOWED_STATES


def _program_identity_key(p: ProgramItem) -> str:
    return (_norm_text(p.institution_name) + " | " + _norm_text(p.program_name)).strip()


# --------------------------------------------------------------------------- #
# Verification for a single program                                           #
# --------------------------------------------------------------------------- #
async def verify_program(
    evaluator: Evaluator,
    parent_node,
    program: ProgramItem,
    idx: int,
) -> None:
    prog_node = evaluator.add_parallel(
        id=f"program_{idx+1}",
        desc=f"{['First','Second','Third'][idx]} qualifying program",
        parent=parent_node,
        critical=False,
    )

    # 1) Institution & Program Name with URL
    exists_names_url = evaluator.add_custom_node(
        result=bool(program.institution_name and program.program_name and program.program_url),
        id=f"program_{idx+1}_names_url_exists",
        desc="Names and direct program URL are provided",
        parent=prog_node,
        critical=True,
    )
    names_leaf = evaluator.add_leaf(
        id=f"program_{idx+1}_institution_program_name_with_url",
        desc="Provides institution name and program name, with a direct source URL.",
        parent=prog_node,
        critical=True,
    )
    claim_names = (
        f"This webpage describes the program '{program.program_name or ''}' at '{program.institution_name or ''}'. "
        f"The page should clearly indicate the institution and the Clinical Mental Health Counseling master's program."
    )
    await evaluator.verify(
        claim=claim_names,
        node=names_leaf,
        sources=program.program_url,
        additional_instruction="Allow minor naming variations, acronyms, or formatting differences. Confirm the page corresponds to the specified program at the specified institution.",
    )

    # 2) State FL/GA/NC with URL
    exists_state_url = evaluator.add_custom_node(
        result=bool(program.state and program.state_source_url),
        id=f"program_{idx+1}_state_url_exists",
        desc="Location/state and verification URL are provided",
        parent=prog_node,
        critical=True,
    )
    state_leaf = evaluator.add_leaf(
        id=f"program_{idx+1}_state_fl_ga_nc_with_url",
        desc="Confirms the university is located in Florida, Georgia, or North Carolina, with a direct source URL.",
        parent=prog_node,
        critical=True,
    )
    state_claim = (
        f"The institution is located in one of the allowed states (Florida, Georgia, or North Carolina). "
        f"The answer states: '{program.state or ''}'. Confirm that the source page shows the institution is in this state."
    )
    await evaluator.verify(
        claim=state_claim,
        node=state_leaf,
        sources=program.state_source_url,
        additional_instruction="Accept common abbreviations (FL, GA, NC) and verify via address, campus location, or official profile page.",
    )

    # 3) CACREP CMHC accreditation with directory URL
    exists_cacrep_dir = evaluator.add_custom_node(
        result=bool(program.cacrep_directory_url),
        id=f"program_{idx+1}_cacrep_dir_exists",
        desc="CACREP directory listing URL is provided",
        parent=prog_node,
        critical=True,
    )
    cacrep_leaf = evaluator.add_leaf(
        id=f"program_{idx+1}_cacrep_cmhc_accreditation_with_directory_url",
        desc="Confirms current CACREP accreditation for the Clinical Mental Health Counseling specialty and includes a direct CACREP directory listing URL.",
        parent=prog_node,
        critical=True,
    )
    cacrep_claim = (
        "This CACREP directory listing page confirms that the program has current CACREP accreditation for the "
        "Clinical Mental Health Counseling specialty."
    )
    await evaluator.verify(
        claim=cacrep_claim,
        node=cacrep_leaf,
        sources=program.cacrep_directory_url,
        additional_instruction="Verify that the CACREP directory entry lists 'Clinical Mental Health Counseling' and shows the program as accredited. Allow minor naming variations.",
    )

    # 4) Credit hours ≥60 semester or ≥90 quarter with URL
    exists_credits_url = evaluator.add_custom_node(
        result=bool(program.credit_hours and program.credit_hours_source_url),
        id=f"program_{idx+1}_credit_hours_url_exists",
        desc="Total credit hours and verification URL are provided",
        parent=prog_node,
        critical=True,
    )
    credits_leaf = evaluator.add_leaf(
        id=f"program_{idx+1}_credit_hours_at_least_60_or_90_with_url",
        desc="States total credit hours and verifies they meet CACREP minimum (≥60 semester or ≥90 quarter), with a direct source URL.",
        parent=prog_node,
        critical=True,
    )
    credits_claim = (
        "The program requires at least 60 semester credit hours or at least 90 quarter credit hours. "
        "Confirm the requirement on the provided source page."
    )
    await evaluator.verify(
        claim=credits_claim,
        node=credits_leaf,
        sources=program.credit_hours_source_url,
        additional_instruction="If the page states '60 credits', it satisfies 'at least 60 semester credits'. If it states '90 quarter hours', it satisfies the minimum. Allow small textual variations (e.g., 'semester hours').",
    )

    # 5) Delivery format with URL
    exists_format_url = evaluator.add_custom_node(
        result=bool(program.delivery_format and program.delivery_format_source_url),
        id=f"program_{idx+1}_delivery_format_url_exists",
        desc="Delivery format and verification URL are provided",
        parent=prog_node,
        critical=True,
    )
    format_leaf = evaluator.add_leaf(
        id=f"program_{idx+1}_delivery_format_with_url",
        desc="Specifies delivery format (synchronous, asynchronous, or hybrid/online-blended), with a direct source URL.",
        parent=prog_node,
        critical=True,
    )
    format_claim = (
        f"The program is delivered online in a '{program.delivery_format or ''}' format (synchronous, asynchronous, or hybrid). "
        "Confirm this on the source page."
    )
    await evaluator.verify(
        claim=format_claim,
        node=format_leaf,
        sources=program.delivery_format_source_url,
        additional_instruction="Accept synonyms like 'live online', 'self-paced online', or 'combination of live and self-paced'.",
    )

    # 6) Tuition cost with URL
    exists_tuition_url = evaluator.add_custom_node(
        result=bool(program.tuition_cost and program.tuition_source_url),
        id=f"program_{idx+1}_tuition_url_exists",
        desc="Tuition information and verification URL are provided",
        parent=prog_node,
        critical=True,
    )
    tuition_leaf = evaluator.add_leaf(
        id=f"program_{idx+1}_tuition_cost_with_url",
        desc="Provides current tuition cost (per-credit or total), with a direct source URL.",
        parent=prog_node,
        critical=True,
    )
    tuition_claim = (
        f"The tuition information provided ('{program.tuition_cost or ''}') is accurate for this program. "
        "Confirm the cost on the source page."
    )
    await evaluator.verify(
        claim=tuition_claim,
        node=tuition_leaf,
        sources=program.tuition_source_url,
        additional_instruction="Prices may be listed per-credit or as total program cost. Accept either as long as it matches the page.",
    )

    # 7) Minimum GPA with URL
    exists_gpa_url = evaluator.add_custom_node(
        result=bool(program.min_gpa and program.min_gpa_source_url),
        id=f"program_{idx+1}_gpa_url_exists",
        desc="Minimum GPA and verification URL are provided",
        parent=prog_node,
        critical=True,
    )
    gpa_leaf = evaluator.add_leaf(
        id=f"program_{idx+1}_minimum_gpa_with_url",
        desc="States minimum undergraduate GPA requirement for admission, with a direct source URL.",
        parent=prog_node,
        critical=True,
    )
    gpa_claim = (
        f"The minimum undergraduate GPA required for admission is '{program.min_gpa or ''}'. "
        "Confirm this requirement on the source page."
    )
    await evaluator.verify(
        claim=gpa_claim,
        node=gpa_leaf,
        sources=program.min_gpa_source_url,
        additional_instruction="Allow variations like '3.0 on a 4.0 scale' or 'minimum cumulative GPA of 2.75'.",
    )

    # 8) Standardized test policy with URL
    exists_test_url = evaluator.add_custom_node(
        result=bool(program.test_policy and program.test_policy_source_url),
        id=f"program_{idx+1}_test_policy_url_exists",
        desc="Standardized test policy and verification URL are provided",
        parent=prog_node,
        critical=True,
    )
    test_leaf = evaluator.add_leaf(
        id=f"program_{idx+1}_standardized_test_policy_with_url",
        desc="States whether GRE/GMAT is required or waived (and under what conditions if applicable), with a direct source URL.",
        parent=prog_node,
        critical=True,
    )
    test_claim = (
        f"The program's standardized test policy is: '{program.test_policy or ''}'. "
        "Confirm the GRE/GMAT requirement or waiver details on the source page."
    )
    await evaluator.verify(
        claim=test_claim,
        node=test_leaf,
        sources=program.test_policy_source_url,
        additional_instruction="Accept statements like 'GRE not required' or 'GRE waived if GPA ≥ X'. Confirm any conditions noted.",
    )

    # 9) Practicum ≥100 hours with URL
    exists_prac_url = evaluator.add_custom_node(
        result=bool(program.practicum_hours and program.practicum_source_url),
        id=f"program_{idx+1}_practicum_url_exists",
        desc="Practicum hours and verification URL are provided",
        parent=prog_node,
        critical=True,
    )
    practicum_leaf = evaluator.add_leaf(
        id=f"program_{idx+1}_practicum_at_least_100_with_url",
        desc="Confirms practicum requirement is at least 100 clock hours, with a direct source URL.",
        parent=prog_node,
        critical=True,
    )
    practicum_claim = "This program requires at least 100 clock hours of practicum."
    await evaluator.verify(
        claim=practicum_claim,
        node=practicum_leaf,
        sources=program.practicum_source_url,
        additional_instruction="If the page states 100 hours (or more), it satisfies the requirement.",
    )

    # 10) Internship ≥600 hours with URL
    exists_intern_url = evaluator.add_custom_node(
        result=bool(program.internship_hours and program.internship_source_url),
        id=f"program_{idx+1}_internship_url_exists",
        desc="Internship hours and verification URL are provided",
        parent=prog_node,
        critical=True,
    )
    internship_leaf = evaluator.add_leaf(
        id=f"program_{idx+1}_internship_at_least_600_with_url",
        desc="Confirms internship requirement is at least 600 clock hours, with a direct source URL.",
        parent=prog_node,
        critical=True,
    )
    internship_claim = "This program requires at least 600 clock hours of internship."
    await evaluator.verify(
        claim=internship_claim,
        node=internship_leaf,
        sources=program.internship_source_url,
        additional_instruction="If the page states 600 hours (or more), it satisfies the requirement.",
    )

    # 11) Regional accreditation with URL
    exists_regacc_url = evaluator.add_custom_node(
        result=bool(program.regional_accreditation and program.regional_accreditation_source_url),
        id=f"program_{idx+1}_regional_accreditation_url_exists",
        desc="Regional accreditation and verification URL are provided",
        parent=prog_node,
        critical=True,
    )
    regacc_leaf = evaluator.add_leaf(
        id=f"program_{idx+1}_regional_accreditation_with_url",
        desc="Confirms the institution holds regional accreditation from a recognized accrediting body, with a direct source URL.",
        parent=prog_node,
        critical=True,
    )
    regacc_claim = (
        f"The institution is regionally accredited by '{program.regional_accreditation or ''}'. Confirm the accreditor on the source page."
    )
    await evaluator.verify(
        claim=regacc_claim,
        node=regacc_leaf,
        sources=program.regional_accreditation_source_url,
        additional_instruction="Recognized regional accreditors include SACSCOC, MSCHE, HLC, NECHE, NWCCU, and WSCUC.",
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

    # Top-level aggregator for the task (set non-critical to allow partial credit)
    top = evaluator.add_parallel(
        id="find_cacrep_programs",
        desc="Identify up to three qualifying CACREP-accredited online/blended CMHC master's programs in FL, GA, or NC, with required attributes and direct source URLs.",
        parent=root,
        critical=False,
    )

    # Extract programs from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_programs(),
        template_class=ProgramsExtraction,
        extraction_name="programs_extraction",
    )

    # Only consider first 3 programs; pad if fewer
    programs = list(extracted.programs[:3])
    while len(programs) < 3:
        programs.append(ProgramItem())

    # Program count & uniqueness check (non-critical)
    # Check that first three have distinct (institution, program) pairs and at least three provided
    valid_keys = [k for k in [_program_identity_key(p) for p in programs] if k.strip() != "|" and k.strip() != ""]
    unique_ok = len(valid_keys) == 3 and len(set(valid_keys)) == 3
    evaluator.add_custom_node(
        result=unique_ok,
        id="program_count_and_uniqueness",
        desc="Provides three distinct (non-duplicate) programs.",
        parent=top,
        critical=False,
    )

    # Build verification subtree for each program
    for i, program in enumerate(programs):
        await verify_program(evaluator, top, program, i)

    # Add custom info to summary for transparency
    evaluator.add_custom_info(
        info={
            "allowed_states": ["Florida (FL)", "Georgia (GA)", "North Carolina (NC)"],
            "cacrep_minimum": "≥60 semester credits or ≥90 quarter credits",
            "programs_extracted_count": len(extracted.programs),
        },
        info_type="requirements_overview",
    )

    return evaluator.get_summary()