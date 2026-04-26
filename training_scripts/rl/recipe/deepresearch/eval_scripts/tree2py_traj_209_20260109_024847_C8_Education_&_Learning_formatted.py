import asyncio
import logging
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from obj_task_eval.llm_client.base_client import LLMClient
from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "stem_opt_online_masters"
TASK_DESCRIPTION = (
    "I am an international student interested in pursuing an online master's degree in the United States that would "
    "qualify me for STEM OPT extension. I need maximum flexibility due to my work schedule and cannot travel to campus.\n\n"
    "Please identify 4 online master's degree programs from regionally accredited U.S. universities that meet ALL of the following mandatory requirements:\n\n"
    "1. STEM-designated: The program must be officially designated as a STEM program, making it eligible for the 24-month STEM OPT extension for F-1 visa holders\n"
    "2. Specialized Accreditation: The program must have specialized/programmatic accreditation from a recognized accrediting body (such as AACSB for business, CEPH for public health, NASPAA for public administration, CSWE for social work, CAHME for healthcare administration, or CCNE for nursing)\n"
    "3. 100% Online: The program must be offered completely online\n"
    "4. No Residency Requirement: The program must not require any on-campus visits, residencies, or in-person attendance\n"
    "5. Asynchronous Format: The program must offer asynchronous coursework (no scheduled live class sessions)\n"
    "6. Multiple Start Dates: The program must offer multiple enrollment start dates throughout the year (not just fall semester)\n\n"
    "For each of the 4 programs, please provide:\n"
    "- Program name and degree title\n"
    "- University name\n"
    "- Confirmation of STEM designation\n"
    "- Specialized accreditation body\n"
    "- Total credit hours required\n"
    "- Tuition cost (per credit hour or total program cost)\n"
    "- Program duration/completion timeline\n"
    "- Available concentrations or specializations\n"
    "- Minimum GPA requirement for admission\n"
    "- GRE/GMAT requirement policy\n"
    "- Direct URL to the official program webpage\n\n"
    "The 4 programs should represent different fields or specializations to provide diverse options."
)

RECOGNIZED_PROGRAMMATIC = {"AACSB", "CEPH", "NASPAA", "CSWE", "CAHME", "CCNE"}
RECOGNIZED_REGIONAL = {"HLC", "MSCHE", "NECHE", "NWCCU", "SACSCOC", "WSCUC"}


# --------------------------------------------------------------------------- #
# Data models                                                                 #
# --------------------------------------------------------------------------- #
class ProgramItem(BaseModel):
    program_name: Optional[str] = None
    degree_title: Optional[str] = None
    degree_level: Optional[str] = None
    university_name: Optional[str] = None

    field_or_specialization: Optional[str] = None

    specialized_accreditation_body: Optional[str] = None
    specialized_accreditation_url: Optional[str] = None

    regional_accreditor_name: Optional[str] = None
    university_accreditation_url: Optional[str] = None

    stem_designation_statement: Optional[str] = None
    stem_info_url: Optional[str] = None

    online_format_statement: Optional[str] = None
    no_residency_statement: Optional[str] = None
    asynchronous_statement: Optional[str] = None
    multiple_start_dates_statement: Optional[str] = None

    credit_hours_text: Optional[str] = None
    tuition_cost_text: Optional[str] = None
    program_duration_text: Optional[str] = None

    concentrations: List[str] = Field(default_factory=list)
    gpa_requirement_text: Optional[str] = None
    test_policy_text: Optional[str] = None

    program_url: Optional[str] = None
    supporting_urls: List[str] = Field(default_factory=list)


class ProgramsExtraction(BaseModel):
    programs: List[ProgramItem] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_programs() -> str:
    return (
        "Extract up to 4 online master's degree programs mentioned in the answer. For each program, return a JSON "
        "object with the following fields:\n"
        "- program_name: the program name\n"
        "- degree_title: the full degree title (e.g., MS in Data Science)\n"
        "- degree_level: the degree level text provided (should be a master's)\n"
        "- university_name: the university offering the program\n"
        "- field_or_specialization: the academic field or specialization of the program (e.g., data science, public health)\n"
        "- specialized_accreditation_body: the named specialized/programmatic accreditor (e.g., AACSB, CEPH, NASPAA, CSWE, CAHME, CCNE)\n"
        "- specialized_accreditation_url: a URL on the university/program or accreditor site confirming this specialized accreditation\n"
        "- regional_accreditor_name: the named U.S. regional accreditor for the university (e.g., HLC, MSCHE, NECHE, NWCCU, SACSCOC, WSCUC)\n"
        "- university_accreditation_url: URL to the university's accreditation page or accreditor listing\n"
        "- stem_designation_statement: text indicating STEM designation or CIP code flagged as STEM\n"
        "- stem_info_url: URL confirming STEM designation for the program or CIP listing on university/program site\n"
        "- online_format_statement: text confirming 100% online delivery\n"
        "- no_residency_statement: text confirming no on-campus visits/residencies/in-person attendance required\n"
        "- asynchronous_statement: text confirming asynchronous coursework availability (no required scheduled live sessions)\n"
        "- multiple_start_dates_statement: text confirming multiple enrollment start dates throughout the year\n"
        "- credit_hours_text: text stating total credit hours required\n"
        "- tuition_cost_text: text stating tuition cost (per credit or total)\n"
        "- program_duration_text: text stating typical program duration/completion timeline\n"
        "- concentrations: list of available concentrations or specializations (array of strings)\n"
        "- gpa_requirement_text: text stating minimum GPA requirement for admission (if any)\n"
        "- test_policy_text: text stating GRE/GMAT requirement policy (required/waived/optional)\n"
        "- program_url: direct URL to the official program webpage (must be a valid URL if provided)\n"
        "- supporting_urls: array of any other official URLs cited for this program (e.g., admissions, tuition, start dates)\n\n"
        "Return a JSON object: {\"programs\": [ ... ]}. If any field is missing in the answer, set it to null (or an empty list for arrays). "
        "Use only URLs explicitly present in the answer text."
    )


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _nonempty(s: Optional[str]) -> bool:
    return bool(s and str(s).strip())


def _is_valid_url(u: Optional[str]) -> bool:
    if not _nonempty(u):
        return False
    u = u.strip().lower()
    return u.startswith("http://") or u.startswith("https://")


def _is_masters_degree(degree_title: Optional[str], degree_level: Optional[str]) -> bool:
    text = " ".join([t for t in [degree_title, degree_level] if _nonempty(t)]).lower()
    tokens = [
        "master", "m.s", "ms", "msc", "ma", "m.a", "mba", "mph", "mpa",
        "msn", "m.eng", "meng", "med", "m.ed", "macc", "m.acc", "mfa",
        "mps", "m.p.s", "mse", "m.s.e"
    ]
    return any(tok in text for tok in tokens)


def _unique_ordered(urls: List[str]) -> List[str]:
    seen = set()
    out = []
    for u in urls:
        if not _is_valid_url(u):
            continue
        if u not in seen:
            seen.add(u)
            out.append(u)
    return out


def _gather_sources(p: ProgramItem) -> List[str]:
    urls: List[str] = []
    urls.append(p.program_url or "")
    urls.extend(p.supporting_urls or [])
    urls.append(p.specialized_accreditation_url or "")
    urls.append(p.stem_info_url or "")
    urls.append(p.university_accreditation_url or "")
    return _unique_ordered(urls)


# --------------------------------------------------------------------------- #
# Verification builders                                                       #
# --------------------------------------------------------------------------- #
async def verify_field_diversity(evaluator: Evaluator, parent_node, programs: List[ProgramItem]) -> None:
    fields = []
    for i, p in enumerate(programs[:4]):
        label = p.field_or_specialization or p.degree_title or p.program_name or f"Program {i + 1}"
        fields.append(label)

    node = evaluator.add_leaf(
        id="field_diversity",
        desc="The 4 programs represent different academic fields or specializations",
        parent=parent_node,
        critical=True
    )

    claim = (
        f"The four programs represent different fields/specializations: "
        + "; ".join([f"{i + 1}: {f}" for i, f in enumerate(fields)])
        + ". Judge whether they are meaningfully distinct (e.g., business vs. public health vs. computer science vs. nursing)."
    )

    await evaluator.verify(
        claim=claim,
        node=node,
        additional_instruction=(
            "Consider broad discipline differences and reasonable synonyms. "
            "Minor naming variations for the same field should not count as distinct. "
            "If any two are essentially the same field or specialization, mark Incorrect."
        )
    )


async def verify_program(evaluator: Evaluator, parent_node, program: ProgramItem, index: int) -> None:
    prog_id = f"program_{index + 1}"
    prog_node = evaluator.add_parallel(
        id=prog_id,
        desc=f"Program {index + 1}: qualifying program identification and required details",
        parent=parent_node,
        critical=False
    )

    # Critical presence and basic checks (custom nodes)
    evaluator.add_custom_node(
        result=_nonempty(program.program_name) and _nonempty(program.degree_title),
        id=f"{prog_id}_program_name",
        desc="Program name and degree title provided",
        parent=prog_node,
        critical=True
    )

    evaluator.add_custom_node(
        result=_is_masters_degree(program.degree_title, program.degree_level),
        id=f"{prog_id}_degree_level_masters",
        desc="Program is a master's degree program",
        parent=prog_node,
        critical=True
    )

    evaluator.add_custom_node(
        result=_nonempty(program.university_name),
        id=f"{prog_id}_university_name",
        desc="University name provided",
        parent=prog_node,
        critical=True
    )

    evaluator.add_custom_node(
        result=_is_valid_url(program.program_url),
        id=f"{prog_id}_program_url",
        desc="Direct URL to the official program webpage provided",
        parent=prog_node,
        critical=True
    )

    # Critical verifications via sources
    # Regional accreditation (U.S. regionally accredited)
    reg_acc_node = evaluator.add_leaf(
        id=f"{prog_id}_regional_accreditation_us_university",
        desc="University is a U.S. university and regionally accredited by a recognized U.S. regional accreditor",
        parent=prog_node,
        critical=True
    )
    acc_body = program.regional_accreditor_name or "a recognized U.S. regional accreditor"
    claim_reg = (
        f"{program.university_name or 'The university'} is regionally accredited in the United States "
        f"by {acc_body}."
    )
    await evaluator.verify(
        claim=claim_reg,
        node=reg_acc_node,
        sources=_gather_sources(program),
        additional_instruction=(
            "Verify that the university is regionally accredited by one of the following: "
            "HLC, MSCHE, NECHE, NWCCU, SACSCOC, WSCUC. Use the provided accreditation page or official sources."
        )
    )

    # STEM designation
    stem_node = evaluator.add_leaf(
        id=f"{prog_id}_stem_designation",
        desc="Program is STEM-designated (eligible for the STEM OPT extension)",
        parent=prog_node,
        critical=True
    )
    claim_stem = (
        f"The program '{program.program_name or 'this program'}' at "
        f"{program.university_name or 'the university'} is officially STEM-designated and eligible for the 24-month STEM OPT extension."
    )
    await evaluator.verify(
        claim=claim_stem,
        node=stem_node,
        sources=_gather_sources(program),
        additional_instruction=(
            "Look for explicit mention of 'STEM-designated' on the program/university site or a university page mapping CIP to STEM. "
            "Do not rely on external knowledge; rely on the page statements."
        )
    )

    # Specialized accreditation (must be among recognized)
    spec_acc_node = evaluator.add_leaf(
        id=f"{prog_id}_specialized_accreditation",
        desc="Program has specialized/programmatic accreditation from a recognized body (AACSB, CEPH, NASPAA, CSWE, CAHME, or CCNE)",
        parent=prog_node,
        critical=True
    )
    spec_body = program.specialized_accreditation_body or "a recognized accreditor"
    claim_spec = (
        f"The program holds specialized/programmatic accreditation from {spec_body}, "
        f"which is one of AACSB, CEPH, NASPAA, CSWE, CAHME, or CCNE."
    )
    await evaluator.verify(
        claim=claim_spec,
        node=spec_acc_node,
        sources=_gather_sources(program),
        additional_instruction=(
            "Confirm that the program page or the accreditor page explicitly states the program's specialized accreditation. "
            "Also verify that the named accreditor is one of: AACSB, CEPH, NASPAA, CSWE, CAHME, CCNE."
        )
    )

    # 100% Online
    online_node = evaluator.add_leaf(
        id=f"{prog_id}_online_format",
        desc="Program is offered 100% online",
        parent=prog_node,
        critical=True
    )
    await evaluator.verify(
        claim="The program is delivered 100% online.",
        node=online_node,
        sources=_gather_sources(program),
        additional_instruction="Verify the program page explicitly indicates fully online delivery."
    )

    # No residency requirement
    residency_node = evaluator.add_leaf(
        id=f"{prog_id}_no_residency",
        desc="Program requires no on-campus visits/residencies/in-person attendance",
        parent=prog_node,
        critical=True
    )
    await evaluator.verify(
        claim="The program requires no on-campus visits, residencies, or in-person attendance.",
        node=residency_node,
        sources=_gather_sources(program),
        additional_instruction=(
            "Confirm that the page indicates no in-person components. If any residency or on-campus session is required, mark Incorrect."
        )
    )

    # Asynchronous coursework
    async_node = evaluator.add_leaf(
        id=f"{prog_id}_asynchronous",
        desc="Program offers asynchronous coursework (no scheduled live class sessions required)",
        parent=prog_node,
        critical=True
    )
    await evaluator.verify(
        claim="The program offers asynchronous coursework with no required scheduled live sessions.",
        node=async_node,
        sources=_gather_sources(program),
        additional_instruction=(
            "Check for statements like 'asynchronous', 'no set class times', or similar phrasing that clearly indicates asynchronous availability."
        )
    )

    # Multiple start dates
    start_node = evaluator.add_leaf(
        id=f"{prog_id}_multiple_start_dates",
        desc="Program offers multiple start dates throughout the year (not only fall)",
        parent=prog_node,
        critical=True
    )
    await evaluator.verify(
        claim="The program offers multiple enrollment start dates throughout the year (not only fall).",
        node=start_node,
        sources=_gather_sources(program),
        additional_instruction=(
            "Look for multiple intakes (e.g., fall, spring, summer, or monthly starts). 'Rolling admissions' alone without multiple start dates is insufficient."
        )
    )

    # Non-critical information presence verifications
    credit_node = evaluator.add_leaf(
        id=f"{prog_id}_credit_hours",
        desc="Total credit hours required specified",
        parent=prog_node,
        critical=False
    )
    await evaluator.verify(
        claim="The official program webpage states the total credit hours required for completion.",
        node=credit_node,
        sources=_gather_sources(program),
        additional_instruction="You do not need to match an exact number; verify that the page clearly provides total credit hours."
    )

    tuition_node = evaluator.add_leaf(
        id=f"{prog_id}_tuition_cost",
        desc="Tuition cost provided (per credit hour or total program cost)",
        parent=prog_node,
        critical=False
    )
    await evaluator.verify(
        claim="The official program webpage provides tuition cost (per credit hour or total program cost).",
        node=tuition_node,
        sources=_gather_sources(program),
        additional_instruction="Verify presence of tuition information on the official page."
    )

    duration_node = evaluator.add_leaf(
        id=f"{prog_id}_program_duration",
        desc="Program duration/completion timeline provided",
        parent=prog_node,
        critical=False
    )
    await evaluator.verify(
        claim="The official program webpage provides a typical program duration or completion timeline.",
        node=duration_node,
        sources=_gather_sources(program),
        additional_instruction="Look for duration, typical time to completion, or pacing information."
    )

    conc_node = evaluator.add_leaf(
        id=f"{prog_id}_concentrations",
        desc="Available concentrations/specializations provided",
        parent=prog_node,
        critical=False
    )
    await evaluator.verify(
        claim="The official program webpage lists available concentrations or specializations.",
        node=conc_node,
        sources=_gather_sources(program),
        additional_instruction="Verify presence of concentration/specialization listings."
    )

    gpa_node = evaluator.add_leaf(
        id=f"{prog_id}_gpa_requirement",
        desc="Minimum GPA requirement for admission provided",
        parent=prog_node,
        critical=False
    )
    await evaluator.verify(
        claim="The official program webpage states a minimum GPA requirement for admission (or explicitly states that no minimum is set).",
        node=gpa_node,
        sources=_gather_sources(program),
        additional_instruction="Check admissions requirements for GPA information."
    )

    test_node = evaluator.add_leaf(
        id=f"{prog_id}_test_policy",
        desc="GRE/GMAT requirement policy provided",
        parent=prog_node,
        critical=False
    )
    await evaluator.verify(
        claim="The official program webpage states a GRE/GMAT policy (required, waived, optional, or not required).",
        node=test_node,
        sources=_gather_sources(program),
        additional_instruction="Check admissions requirements for standardized test policies."
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
    model: str = "o4-mini"
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
        default_model=model
    )

    extracted = await evaluator.extract(
        prompt=prompt_extract_programs(),
        template_class=ProgramsExtraction,
        extraction_name="programs_extraction"
    )

    # Record recognized accreditors info for transparency
    evaluator.add_ground_truth({
        "recognized_programmatic_bodies": sorted(list(RECOGNIZED_PROGRAMMATIC)),
        "recognized_regional_accreditors": sorted(list(RECOGNIZED_REGIONAL)),
        "requirements": [
            "STEM-designated",
            "Specialized accreditation (AACSB, CEPH, NASPAA, CSWE, CAHME, CCNE)",
            "100% online",
            "No residency requirement",
            "Asynchronous coursework offered",
            "Multiple start dates"
        ]
    }, gt_type="requirements_and_recognized_bodies")

    # Ensure exactly 4 programs for evaluation (pad with empty if fewer; take first 4 if more)
    programs: List[ProgramItem] = (extracted.programs or [])[:4]
    while len(programs) < 4:
        programs.append(ProgramItem())

    # Field diversity critical check
    await verify_field_diversity(evaluator, root, programs)

    # Build verification subtrees for each program
    for idx in range(4):
        await verify_program(evaluator, root, programs[idx], idx)

    return evaluator.get_summary()