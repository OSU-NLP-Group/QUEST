import asyncio
import logging
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field
from obj_task_eval.llm_client.base_client import LLMClient

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys


TASK_ID = "online_masters_flex_affordable_highstandard"
TASK_DESCRIPTION = (
    "A working professional in the United States seeks to pursue an online master's degree that offers maximum "
    "flexibility and affordability while maintaining high academic standards. Identify a specific online master's "
    "degree program offered by a regionally accredited U.S. university that meets ALL of the following eight requirements: "
    "(1) The institution must hold regional accreditation recognized by CHEA (Council for Higher Education Accreditation), "
    "(2) The program must not require GRE or GMAT scores for admission (for applicants meeting minimum GPA requirements), "
    "(3) The program must accept applicants with an undergraduate GPA of 3.0 or lower, "
    "(4) All courses must be offered in a fully asynchronous format with no required synchronous class sessions, "
    "(5) The program must require 36 credit hours or fewer for degree completion, "
    "(6) The tuition must be $500 per credit hour or less, "
    "(7) The program must not require a traditional thesis for graduation (capstone projects or other culminating experiences are acceptable), "
    "and (8) The program must be completable 100% online with no on-campus attendance requirements. "
    "For your answer, provide: (1) the specific degree program name, (2) the institution name, (3) the URL of the official program page, "
    "and (4) brief confirmation of how each of the eight criteria is met."
)


# ----------------------------- Data Models ---------------------------------- #
class CriteriaConfirmations(BaseModel):
    regional_accreditation: Optional[str] = None
    test_optional: Optional[str] = None
    gpa_requirement: Optional[str] = None
    asynchronous_delivery: Optional[str] = None
    credit_hours: Optional[str] = None
    affordable_tuition: Optional[str] = None
    no_thesis_required: Optional[str] = None
    fully_online: Optional[str] = None


class ProgramSelection(BaseModel):
    program_name: Optional[str] = None
    institution_name: Optional[str] = None
    official_url: Optional[str] = None
    additional_urls: List[str] = Field(default_factory=list)
    confirmations: Optional[CriteriaConfirmations] = None


# --------------------------- Extraction Prompt ------------------------------ #
def prompt_extract_program_selection() -> str:
    return (
        "Extract the structured information about the single online master's program identified in the answer. "
        "Return the following fields:\n"
        "1) program_name: The specific degree program name.\n"
        "2) institution_name: The university/institution offering the program.\n"
        "3) official_url: The URL of the official university program page (not a third-party site).\n"
        "4) additional_urls: A list of any other URLs mentioned in the answer that relate to accreditation, tuition, admissions, online delivery details, or program requirements.\n"
        "5) confirmations: A JSON object containing brief confirmation/explanations from the answer for each of the eight criteria with keys:\n"
        "   - regional_accreditation\n"
        "   - test_optional\n"
        "   - gpa_requirement\n"
        "   - asynchronous_delivery\n"
        "   - credit_hours\n"
        "   - affordable_tuition\n"
        "   - no_thesis_required\n"
        "   - fully_online\n\n"
        "Rules:\n"
        "- Extract only what is explicitly present in the answer text. Do not invent or infer missing info.\n"
        "- If a field is not present, set it to null.\n"
        "- For URLs, include full URLs; ignore obviously invalid URLs.\n"
        "- For confirmations, if the answer provides them, include the brief text; otherwise set those fields to null.\n"
        "- If more than one program is mentioned, focus on the first one presented as the recommended program."
    )


# ------------------------------ Helpers ------------------------------------- #
def _nonempty_str(s: Optional[str]) -> bool:
    return isinstance(s, str) and s.strip() != ""


def _all_confirmations_present(conf: Optional[CriteriaConfirmations]) -> bool:
    if conf is None:
        return False
    return all(
        _nonempty_str(getattr(conf, key))
        for key in [
            "regional_accreditation",
            "test_optional",
            "gpa_requirement",
            "asynchronous_delivery",
            "credit_hours",
            "affordable_tuition",
            "no_thesis_required",
            "fully_online",
        ]
    )


def _collect_sources(program: ProgramSelection) -> List[str]:
    urls: List[str] = []
    if _nonempty_str(program.official_url):
        urls.append(program.official_url.strip())  # type: ignore
    if program.additional_urls:
        for u in program.additional_urls:
            if _nonempty_str(u):
                urls.append(u.strip())
    # Deduplicate preserving order
    seen = set()
    deduped = []
    for u in urls:
        if u not in seen:
            deduped.append(u)
            seen.add(u)
    return deduped


# ---------------------------- Verification Logic ---------------------------- #
async def build_tree_and_verify(evaluator: Evaluator, program: ProgramSelection) -> None:
    # Create a critical root node for this specific rubric
    suitable_node = evaluator.add_parallel(
        id="SuitableProgramResponse",
        desc=(
            "Response identifies one specific online master's program from a regionally accredited U.S. university "
            "and includes all required fields and confirmations while meeting all eight constraints."
        ),
        parent=evaluator.root,
        critical=True,
    )

    # Required response fields
    req_fields = evaluator.add_parallel(
        id="RequiredResponseFields",
        desc="Response includes all required identification fields and confirmations requested by the question.",
        parent=suitable_node,
        critical=True,
    )

    evaluator.add_custom_node(
        result=_nonempty_str(program.program_name),
        id="ProgramNameProvided",
        desc="Provides the specific degree program name.",
        parent=req_fields,
        critical=True,
    )

    evaluator.add_custom_node(
        result=_nonempty_str(program.institution_name),
        id="InstitutionNameProvided",
        desc="Provides the institution (university) name offering the program.",
        parent=req_fields,
        critical=True,
    )

    evaluator.add_custom_node(
        result=_nonempty_str(program.official_url),
        id="OfficialProgramURLProvided",
        desc="Provides a URL to the official program page (official university web page for the program).",
        parent=req_fields,
        critical=True,
    )

    evaluator.add_custom_node(
        result=_all_confirmations_present(program.confirmations),
        id="BriefConfirmationForAllCriteria",
        desc="Includes brief confirmation/explanation addressing how each of the eight criteria is met (criterion-by-criterion).",
        parent=req_fields,
        critical=True,
    )

    # Criteria checks
    criteria_node = evaluator.add_parallel(
        id="MeetsAllEightCriteria",
        desc="The identified program satisfies all eight listed constraints.",
        parent=suitable_node,
        critical=True,
    )

    sources = _collect_sources(program)
    institution_label = program.institution_name or "the institution offering this program"
    program_label = program.program_name or "this program"

    # Create leaf nodes for each criterion
    node_regional = evaluator.add_leaf(
        id="RegionalAccreditation",
        desc="Institution holds regional accreditation recognized by CHEA.",
        parent=criteria_node,
        critical=True,
    )

    node_test_optional = evaluator.add_leaf(
        id="TestOptional",
        desc="Program does not require GRE/GMAT for admission when applicants meet minimum GPA requirements.",
        parent=criteria_node,
        critical=True,
    )

    node_gpa_req = evaluator.add_leaf(
        id="GPARequirement",
        desc="Program accepts applicants with an undergraduate GPA of 3.0 or lower.",
        parent=criteria_node,
        critical=True,
    )

    node_async = evaluator.add_leaf(
        id="AsynchronousDelivery",
        desc="All courses are fully asynchronous with no required synchronous class sessions.",
        parent=criteria_node,
        critical=True,
    )

    node_credits = evaluator.add_leaf(
        id="CreditHours",
        desc="Requires 36 credit hours or fewer for degree completion.",
        parent=criteria_node,
        critical=True,
    )

    node_tuition = evaluator.add_leaf(
        id="AffordableTuition",
        desc="Tuition is $500 per credit hour or less.",
        parent=criteria_node,
        critical=True,
    )

    node_no_thesis = evaluator.add_leaf(
        id="NoThesisRequired",
        desc="No traditional thesis required (capstone/other culminating experience acceptable).",
        parent=criteria_node,
        critical=True,
    )

    node_fully_online = evaluator.add_leaf(
        id="FullyOnline",
        desc="Completable 100% online with no on-campus attendance requirements.",
        parent=criteria_node,
        critical=True,
    )

    # Construct claims and additional instructions
    claims_and_sources = [
        (
            f"The institution {institution_label} is regionally accredited by a U.S. regional accreditor that is recognized by CHEA.",
            sources,
            node_regional,
            (
                "Look for explicit institutional accreditation statements (e.g., HLC, MSCHE, SACSCOC, NECHE, NWCCU, WSCUC). "
                "If the page indicates accreditation by one of these regional accreditors, treat this criterion as satisfied. "
                "Use only the provided webpage(s); if no accreditation is present on these pages, mark as not supported."
            ),
        ),
        (
            f"{program_label} does not require GRE or GMAT for admission when the applicant meets the minimum GPA requirement.",
            sources,
            node_test_optional,
            (
                "Verify the admissions policy: acceptable signals include 'GRE not required', 'GMAT not required', "
                "'tests optional', or 'tests waived when GPA meets minimum'. If tests are required only below a GPA threshold, "
                "this still satisfies the criterion."
            ),
        ),
        (
            f"{program_label} accepts applicants with an undergraduate GPA of 3.0 or lower (e.g., minimum GPA ≤ 3.0 or conditional admission below 3.0).",
            sources,
            node_gpa_req,
            (
                "Check the admissions requirements: passing examples include minimum GPA set to 3.0 or lower, "
                "or explicit conditional/provisional admission routes for GPAs below 3.0."
            ),
        ),
        (
            f"All courses in {program_label} are offered fully asynchronously and do not require any synchronous class sessions.",
            sources,
            node_async,
            (
                "Confirm asynchronous delivery: acceptable signals include 'fully asynchronous', 'no live class meetings', "
                "'no required synchronous sessions', 'self-paced within deadlines'. If any synchronous attendance is required, fail."
            ),
        ),
        (
            f"{program_label} requires 36 credit hours or fewer to complete the degree.",
            sources,
            node_credits,
            (
                "Verify total program credits/semester hours/units in curriculum or program overview. "
                "If the listed total is ≤ 36, pass; otherwise, fail."
            ),
        ),
        (
            f"The tuition for {program_label} is $500 per credit hour or less.",
            sources,
            node_tuition,
            (
                "Check tuition details: look for per-credit pricing. If clearly ≤ $500 per credit hour, pass. "
                "If tuition is presented differently (per course/semester), only pass if it is explicitly equivalent "
                "to ≤ $500 per credit hour according to the provided page(s)."
            ),
        ),
        (
            f"{program_label} does not require a traditional thesis to graduate; a capstone or similar culminating experience is acceptable instead.",
            sources,
            node_no_thesis,
            (
                "Verify graduation requirements: acceptable signals include 'no thesis required', 'capstone project required', "
                "'comprehensive exam', or similar non-thesis culminating experience."
            ),
        ),
        (
            f"{program_label} can be completed 100% online and does not require any on-campus attendance.",
            sources,
            node_fully_online,
            (
                "Confirm fully online delivery: acceptable signals include '100% online', 'no campus visits', "
                "'no residency requirements'. If any on-campus attendance is required, fail."
            ),
        ),
    ]

    # Run all criterion verifications concurrently to avoid premature skips due to sibling failures
    await evaluator.batch_verify(claims_and_sources)


# ---------------------------- Main Entry Point ------------------------------ #
async def evaluate_answer(
    client: LLMClient,
    answer: str,
    agent_name: str,
    answer_name: str,
    cache: CacheFileSys,
    semaphore: asyncio.Semaphore,
    logger: logging.Logger,
    model: str = "o4-mini",
) -> Dict[str, Any]:
    evaluator = Evaluator()
    evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Root aggregator
        agent_name=agent_name,
        answer_name=answer_name,
        client=client,
        task_description=TASK_DESCRIPTION,
        answer=answer,
        global_cache=cache,
        global_semaphore=semaphore,
        logger=logger,
        default_model=model,
        verify_model=model,
        extract_model=model,
    )

    # Extract structured program selection info from the answer
    program = await evaluator.extract(
        prompt=prompt_extract_program_selection(),
        template_class=ProgramSelection,
        extraction_name="program_selection",
    )

    # Build verification tree and verify criteria
    await build_tree_and_verify(evaluator, program)

    # Return evaluation summary
    return evaluator.get_summary()