import asyncio
import logging
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys


TASK_ID = "med_curr_instr_program_eval"
TASK_DESCRIPTION = (
    "Identify a Master of Education program that meets ALL of the following requirements:\n\n"
    "1. The program must specialize in Curriculum and Instruction (as the degree name, concentration, or major)\n"
    "2. The program must require exactly 30 semester credit hours to complete\n"
    "3. The program must be offered 100% online\n"
    "4. The institution's educator preparation programs must hold specialized accreditation from either CAEP (Council for the Accreditation of Educator Preparation) or AAQEP (Association for Advancing Quality in Educator Preparation)\n"
    "5. The institution must hold regional accreditation from one of the six U.S. regional accrediting bodies\n"
    "6. The program must NOT require GRE scores for admission\n"
    "7. The program must have a minimum GPA admission requirement of 3.0 or lower\n"
    "8. The program must require a bachelor's degree from an accredited institution as a prerequisite\n"
    "9. The program must have a stated application deadline for Fall semester admission\n"
    "10. The program must be designed for current K-12 educators, teachers, or education professionals\n\n"
    "For your answer, provide:\n"
    "- The name of the institution\n"
    "- The specific program name/title\n"
    "- Documentation (with reference URLs) verifying that the program meets each of the requirements listed above"
)

REGIONAL_ACCREDITORS = [
    "NECHE", "New England Commission of Higher Education",
    "MSCHE", "Middle States Commission on Higher Education",
    "HLC", "Higher Learning Commission",
    "SACSCOC", "Southern Association of Colleges and Schools Commission on Colleges",
    "WSCUC", "WASC Senior College and University Commission",
    "NWCCU", "Northwest Commission on Colleges and Universities",
]


class ProgramExtraction(BaseModel):
    institution_name: Optional[str] = None
    program_title: Optional[str] = None

    ci_focus_urls: List[str] = Field(default_factory=list)
    credit_hours_urls: List[str] = Field(default_factory=list)
    online_delivery_urls: List[str] = Field(default_factory=list)
    epp_accreditation_urls: List[str] = Field(default_factory=list)
    regional_accreditation_urls: List[str] = Field(default_factory=list)
    no_gre_urls: List[str] = Field(default_factory=list)
    min_gpa_urls: List[str] = Field(default_factory=list)
    bachelor_prereq_urls: List[str] = Field(default_factory=list)
    fall_deadline_urls: List[str] = Field(default_factory=list)
    k12_design_urls: List[str] = Field(default_factory=list)


def prompt_extract_program() -> str:
    return (
        "Extract the selected program identification and the supporting URLs provided in the answer. "
        "Return the following fields:\n"
        "1) institution_name: The institution offering the program.\n"
        "2) program_title: The exact program name/title as stated in the answer.\n"
        "For each requirement below, extract all reference URLs explicitly mentioned in the answer (include markdown-linked URLs and plain URLs):\n"
        "3) ci_focus_urls: URLs that show the program specializes in Curriculum and Instruction (degree name, concentration, or major).\n"
        "4) credit_hours_urls: URLs that show the program requires exactly 30 semester credit hours.\n"
        "5) online_delivery_urls: URLs that show the program is offered 100% online.\n"
        "6) epp_accreditation_urls: URLs that show the institution’s educator preparation programs hold CAEP or AAQEP accreditation.\n"
        "7) regional_accreditation_urls: URLs that show the institution holds U.S. regional accreditation (NECHE, MSCHE, HLC, SACSCOC, WSCUC, NWCCU).\n"
        "8) no_gre_urls: URLs that show GRE scores are NOT required for admission to the program.\n"
        "9) min_gpa_urls: URLs that show the minimum GPA admission requirement is 3.0 or lower.\n"
        "10) bachelor_prereq_urls: URLs that show a bachelor's degree from an accredited institution is required.\n"
        "11) fall_deadline_urls: URLs that show a stated application deadline exists for Fall semester admission.\n"
        "12) k12_design_urls: URLs that show the program is designed for current K-12 educators/teachers/education professionals (not initial certification).\n\n"
        "Rules:\n"
        "- Extract only URLs explicitly present in the answer. If a requirement lacks any URLs, return an empty list for that field.\n"
        "- Do not invent or infer URLs.\n"
        "- If an item (institution_name or program_title) is missing, return null.\n"
        "- Return all fields in a single JSON object."
    )


def _build_ci_additional_instruction(program_title: Optional[str], institution_name: Optional[str]) -> str:
    return (
        f"Verify that the selected program{(' '+program_title) if program_title else ''} "
        f"at{(' '+institution_name) if institution_name else ''} specializes in 'Curriculum and Instruction' "
        "as the degree name, concentration, or major. Accept reasonable variants such as 'Curriculum & Instruction', "
        "'M.Ed in Curriculum and Instruction', 'Master of Education — Curriculum and Instruction', or a listed concentration. "
        "Do not accept unrelated specializations."
    )


def _build_credit_hours_additional_instruction() -> str:
    return (
        "Confirm the program requires exactly 30 semester credit hours. Accept phrasing like '30 credits', "
        "'30 semester hours', or '30 credit hours', but do not accept ranges or values other than exactly 30."
    )


def _build_online_delivery_additional_instruction() -> str:
    return (
        "Confirm the program is offered 100% online (fully online). Accept synonyms like 'fully online', '100% online', "
        "and 'online program'. Do not accept hybrid or on-campus formats."
    )


def _build_epp_acc_additional_instruction(institution_name: Optional[str]) -> str:
    return (
        f"Confirm that the educator preparation programs at{(' '+institution_name) if institution_name else ''} "
        "hold specialized accreditation from CAEP or AAQEP. Evidence may be from the institution’s accreditation page, "
        "college of education page, or CAEP/AAQEP directories. Look explicitly for 'CAEP' or 'AAQEP'."
    )


def _build_regional_acc_additional_instruction(institution_name: Optional[str]) -> str:
    bodies = ", ".join(REGIONAL_ACCREDITORS)
    return (
        f"Confirm that{(' '+institution_name) if institution_name else ''} holds U.S. regional accreditation from one of "
        f"the standard regional accreditors ({bodies}). The page should explicitly name the accreditor or clearly indicate regional accreditation."
    )


def _build_no_gre_additional_instruction(program_title: Optional[str], institution_name: Optional[str]) -> str:
    return (
        f"Confirm that GRE scores are NOT required for admission to the program{(' '+program_title) if program_title else ''} "
        f"at{(' '+institution_name) if institution_name else ''}. Accept statements such as 'GRE not required', 'No GRE required', or 'GRE optional'. "
        "Do not count 'GRE required' or ambiguous statements as meeting this requirement."
    )


def _build_min_gpa_additional_instruction(program_title: Optional[str], institution_name: Optional[str]) -> str:
    return (
        f"Confirm that the minimum GPA admission requirement for{(' '+program_title) if program_title else ''} "
        f"at{(' '+institution_name) if institution_name else ''} is 3.0 or lower (e.g., 2.75, 2.5, 3.0). "
        "Look for explicit 'minimum GPA' language; 'preferred GPA' alone is not sufficient unless minimum is stated and is ≤ 3.0."
    )


def _build_bachelor_prereq_additional_instruction(program_title: Optional[str], institution_name: Optional[str]) -> str:
    return (
        f"Confirm that a bachelor's degree from an accredited institution is required for admission to{(' '+program_title) if program_title else ''} "
        f"at{(' '+institution_name) if institution_name else ''}."
    )


def _build_fall_deadline_additional_instruction(program_title: Optional[str], institution_name: Optional[str]) -> str:
    return (
        f"Confirm there is a stated application deadline for Fall semester admission for{(' '+program_title) if program_title else ''} "
        f"at{(' '+institution_name) if institution_name else ''}. Accept 'Fall application deadline', 'Fall priority deadline', or equivalent explicit deadline statements."
    )


def _build_k12_design_additional_instruction(program_title: Optional[str], institution_name: Optional[str]) -> str:
    return (
        f"Confirm that{(' '+program_title) if program_title else ''} "
        f"at{(' '+institution_name) if institution_name else ''} is designed for current K-12 educators, teachers, or education professionals. "
        "Accept phrases like 'for practicing teachers', 'for in-service educators', 'for licensed educators', or 'for education professionals'; "
        "do not count initial licensure/first-certification programs that target non-educators."
    )


async def _add_constraint_group(
    evaluator: Evaluator,
    parent_node,
    group_id: str,
    group_desc: str,
    urls: List[str],
    support_leaf_id: str,
    support_leaf_desc: str,
    claim: str,
    additional_instruction: str,
) -> None:
    group_node = evaluator.add_parallel(
        id=group_id,
        desc=group_desc,
        parent=parent_node,
        critical=True,
    )

    urls_provided = evaluator.add_custom_node(
        result=(isinstance(urls, list) and len(urls) > 0),
        id=f"{support_leaf_id}_URLs_Provided",
        desc=f"At least one supporting URL is provided for: {support_leaf_desc}",
        parent=group_node,
        critical=True,
    )

    leaf_node = evaluator.add_leaf(
        id=support_leaf_id,
        desc=support_leaf_desc,
        parent=group_node,
        critical=True,
    )

    await evaluator.verify(
        claim=claim,
        node=leaf_node,
        sources=urls,
        additional_instruction=additional_instruction,
    )


async def verify_program(
    evaluator: Evaluator,
    extraction: ProgramExtraction,
) -> None:
    root_node = evaluator.add_parallel(
        id="Program_Identification",
        desc="Return one Master of Education program that satisfies all listed constraints, and include URL documentation supporting each constraint.",
        parent=evaluator.root,
        critical=True,
    )

    required_fields_node = evaluator.add_parallel(
        id="Required_Output_Fields",
        desc="The answer includes the required identification fields for the selected program.",
        parent=root_node,
        critical=True,
    )

    evaluator.add_custom_node(
        result=(extraction.institution_name is not None and str(extraction.institution_name).strip() != ""),
        id="Institution_Name_Provided",
        desc="Provide the name of the institution offering the program.",
        parent=required_fields_node,
        critical=True,
    )

    evaluator.add_custom_node(
        result=(extraction.program_title is not None and str(extraction.program_title).strip() != ""),
        id="Program_Title_Provided",
        desc="Provide the specific program name/title.",
        parent=required_fields_node,
        critical=True,
    )

    constraints_node = evaluator.add_parallel(
        id="Program_Constraints_With_Documentation",
        desc="The program meets every stated constraint, and each constraint is supported by at least one relevant reference URL.",
        parent=root_node,
        critical=True,
    )

    inst = extraction.institution_name or ""
    prog = extraction.program_title or ""

    # 1) Curriculum & Instruction specialization
    await _add_constraint_group(
        evaluator,
        constraints_node,
        group_id="Curriculum_Instruction_Focus_Group",
        group_desc="Constraint: Curriculum and Instruction specialization with URL documentation.",
        urls=extraction.ci_focus_urls,
        support_leaf_id="Curriculum_Instruction_Focus",
        support_leaf_desc="Program specializes in Curriculum and Instruction (degree name, concentration, or major) AND includes URL evidence.",
        claim=f"The program '{prog}' at '{inst}' specializes in Curriculum and Instruction (degree name, concentration, or major).",
        additional_instruction=_build_ci_additional_instruction(extraction.program_title, extraction.institution_name),
    )

    # 2) Exactly 30 semester credit hours
    await _add_constraint_group(
        evaluator,
        constraints_node,
        group_id="Credit_Hours_Exactly_30_Group",
        group_desc="Constraint: Exactly 30 semester credit hours with URL documentation.",
        urls=extraction.credit_hours_urls,
        support_leaf_id="Credit_Hours_Exactly_30",
        support_leaf_desc="Program requires exactly 30 semester credit hours to complete AND includes URL evidence.",
        claim=f"The program '{prog}' at '{inst}' requires exactly 30 semester credit hours to complete.",
        additional_instruction=_build_credit_hours_additional_instruction(),
    )

    # 3) Fully online
    await _add_constraint_group(
        evaluator,
        constraints_node,
        group_id="Fully_Online_Group",
        group_desc="Constraint: 100% online delivery with URL documentation.",
        urls=extraction.online_delivery_urls,
        support_leaf_id="Fully_Online",
        support_leaf_desc="Program is offered 100% online AND includes URL evidence.",
        claim=f"The program '{prog}' at '{inst}' is offered 100% online (fully online).",
        additional_instruction=_build_online_delivery_additional_instruction(),
    )

    # 4) EPP specialized accreditation (CAEP or AAQEP)
    await _add_constraint_group(
        evaluator,
        constraints_node,
        group_id="Specialized_EPP_Accreditation_Group",
        group_desc="Constraint: EPP specialized accreditation (CAEP or AAQEP) with URL documentation.",
        urls=extraction.epp_accreditation_urls,
        support_leaf_id="Specialized_EPP_Accreditation",
        support_leaf_desc="Institution’s educator preparation programs hold specialized accreditation from CAEP or AAQEP AND includes URL evidence.",
        claim=f"The educator preparation programs at '{inst}' are accredited by CAEP or AAQEP.",
        additional_instruction=_build_epp_acc_additional_instruction(extraction.institution_name),
    )

    # 5) Regional accreditation
    await _add_constraint_group(
        evaluator,
        constraints_node,
        group_id="Regional_Accreditation_Group",
        group_desc="Constraint: Regional accreditation with URL documentation.",
        urls=extraction.regional_accreditation_urls,
        support_leaf_id="Regional_Accreditation",
        support_leaf_desc="Institution holds regional accreditation from one of the six U.S. regional accrediting bodies AND includes URL evidence.",
        claim=f"The institution '{inst}' holds regional accreditation from a recognized U.S. regional accrediting body.",
        additional_instruction=_build_regional_acc_additional_instruction(extraction.institution_name),
    )

    # 6) No GRE required
    await _add_constraint_group(
        evaluator,
        constraints_node,
        group_id="No_GRE_Required_Group",
        group_desc="Constraint: GRE not required for admission with URL documentation.",
        urls=extraction.no_gre_urls,
        support_leaf_id="No_GRE_Required",
        support_leaf_desc="Program does NOT require GRE scores for admission AND includes URL evidence.",
        claim=f"GRE scores are not required for admission to the program '{prog}' at '{inst}'.",
        additional_instruction=_build_no_gre_additional_instruction(extraction.program_title, extraction.institution_name),
    )

    # 7) Minimum GPA requirement ≤ 3.0
    await _add_constraint_group(
        evaluator,
        constraints_node,
        group_id="Min_GPA_Requirement_3_0_or_Lower_Group",
        group_desc="Constraint: Minimum GPA requirement is 3.0 or lower with URL documentation.",
        urls=extraction.min_gpa_urls,
        support_leaf_id="Min_GPA_Requirement_3_0_or_Lower",
        support_leaf_desc="Program’s stated minimum GPA admission requirement is 3.0 or lower AND includes URL evidence.",
        claim=f"The program '{prog}' at '{inst}' has a minimum GPA admission requirement that is 3.0 or lower.",
        additional_instruction=_build_min_gpa_additional_instruction(extraction.program_title, extraction.institution_name),
    )

    # 8) Bachelor's degree prerequisite
    await _add_constraint_group(
        evaluator,
        constraints_node,
        group_id="Bachelors_Degree_Prerequisite_Group",
        group_desc="Constraint: Bachelor's degree prerequisite with URL documentation.",
        urls=extraction.bachelor_prereq_urls,
        support_leaf_id="Bachelors_Degree_Prerequisite",
        support_leaf_desc="Program requires a bachelor’s degree from an accredited institution as a prerequisite AND includes URL evidence.",
        claim=f"The program '{prog}' at '{inst}' requires a bachelor's degree from an accredited institution as a prerequisite.",
        additional_instruction=_build_bachelor_prereq_additional_instruction(extraction.program_title, extraction.institution_name),
    )

    # 9) Fall admission deadline stated
    await _add_constraint_group(
        evaluator,
        constraints_node,
        group_id="Fall_Admission_Deadline_Stated_Group",
        group_desc="Constraint: Fall admission application deadline stated with URL documentation.",
        urls=extraction.fall_deadline_urls,
        support_leaf_id="Fall_Admission_Deadline_Stated",
        support_leaf_desc="Program has a stated application deadline for Fall semester admission AND includes URL evidence.",
        claim=f"The program '{prog}' at '{inst}' has a stated application deadline for Fall semester admission.",
        additional_instruction=_build_fall_deadline_additional_instruction(extraction.program_title, extraction.institution_name),
    )

    # 10) Designed for current K-12 educators
    await _add_constraint_group(
        evaluator,
        constraints_node,
        group_id="Designed_For_Current_K12_Educators_Group",
        group_desc="Constraint: Designed for current K-12 educators/teachers/education professionals with URL documentation.",
        urls=extraction.k12_design_urls,
        support_leaf_id="Designed_For_Current_K12_Educators",
        support_leaf_desc="Program is designed for current K-12 educators/teachers/education professionals (not initial certification) AND includes URL evidence.",
        claim=f"The program '{prog}' at '{inst}' is designed for current K-12 educators, teachers, or education professionals (not initial certification).",
        additional_instruction=_build_k12_design_additional_instruction(extraction.program_title, extraction.institution_name),
    )


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
    evaluator.initialize(
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

    extracted = await evaluator.extract(
        prompt=prompt_extract_program(),
        template_class=ProgramExtraction,
        extraction_name="program_extraction",
    )

    # Optional: record a brief summary of URL counts
    url_counts = {
        "ci_focus_urls_count": len(extracted.ci_focus_urls),
        "credit_hours_urls_count": len(extracted.credit_hours_urls),
        "online_delivery_urls_count": len(extracted.online_delivery_urls),
        "epp_accreditation_urls_count": len(extracted.epp_accreditation_urls),
        "regional_accreditation_urls_count": len(extracted.regional_accreditation_urls),
        "no_gre_urls_count": len(extracted.no_gre_urls),
        "min_gpa_urls_count": len(extracted.min_gpa_urls),
        "bachelor_prereq_urls_count": len(extracted.bachelor_prereq_urls),
        "fall_deadline_urls_count": len(extracted.fall_deadline_urls),
        "k12_design_urls_count": len(extracted.k12_design_urls),
    }
    evaluator.add_custom_info(url_counts, info_type="url_counts", info_name="url_counts")

    await verify_program(evaluator, extracted)

    return evaluator.get_summary()