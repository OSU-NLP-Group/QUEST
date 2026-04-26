import asyncio
import logging
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.llm_client.base_client import LLMClient


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "eds_edlead_requirements"
TASK_DESCRIPTION = (
    "Identify a university that offers an Education Specialist (Ed.S.) degree program in Educational Leadership which "
    "requires applicants to have both a minimum of three years of teaching experience and current certification as a school principal. "
    "Provide the name of the university and a reference URL documenting these specific admission requirements."
)


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class ProgramExtraction(BaseModel):
    university: Optional[str] = None
    program_name: Optional[str] = None
    degree_type: Optional[str] = None
    specialization: Optional[str] = None
    teaching_experience_requirement_text: Optional[str] = None
    principal_certification_requirement_text: Optional[str] = None
    reference_urls: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_program_info() -> str:
    return """
    Extract the key details provided in the answer about an Education Specialist (Ed.S.) program in Educational Leadership.
    Return a JSON object with the following fields:
    - university: The exact name of the university or institution.
    - program_name: The program name as written in the answer (e.g., "Ed.S. in Educational Leadership").
    - degree_type: The degree type explicitly stated for the program (e.g., "Ed.S.", "Education Specialist", "Specialist in Education").
    - specialization: The specialization/field of the program (should be "Educational Leadership" if claimed).
    - teaching_experience_requirement_text: The verbatim phrase that indicates the minimum teaching experience requirement (e.g., "minimum of three years teaching experience", "at least 3 years").
    - principal_certification_requirement_text: The verbatim phrase that indicates a requirement for current certification as a school principal (or equivalent phrasing, e.g., "principal license", "principal certificate", "administrator/principal endorsement").
    - reference_urls: An array of all URLs cited in the answer that purportedly document the program details and admission requirements (prefer program/admissions pages). Include only actual URLs present in the answer. If a URL is missing a protocol, prepend http://.

    Do not invent any information. If a field is not explicitly present in the answer, set it to null (or [] for lists).
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def _filter_valid_urls(urls: List[str]) -> List[str]:
    """Keep only plausible HTTP(S) URLs and deduplicate while preserving order."""
    seen = set()
    cleaned: List[str] = []
    for u in urls or []:
        if not u:
            continue
        uu = u.strip()
        if not uu:
            continue
        if not (uu.startswith("http://") or uu.startswith("https://")):
            # Basic normalization if missing protocol
            uu = "http://" + uu
        if uu not in seen:
            seen.add(uu)
            cleaned.append(uu)
    return cleaned


# --------------------------------------------------------------------------- #
# Verification tree construction and checks                                   #
# --------------------------------------------------------------------------- #
async def build_and_verify_tree(
    evaluator: Evaluator,
    root_node,
    extracted: ProgramExtraction
) -> None:
    """
    Build the verification tree according to the rubric and run verifications.
    """
    # Top-level critical node
    main_node = evaluator.add_parallel(
        id="Education_Specialist_Program_Identification",
        desc="Identify an institution offering an Education Specialist (Ed.S.) degree in Educational Leadership with specific admission requirements",
        parent=root_node,
        critical=True,
    )

    # Prepare sources from extracted URLs
    all_urls = _filter_valid_urls(extracted.reference_urls)

    # Record some custom info into the summary for debugging
    evaluator.add_custom_info(
        {
            "university": extracted.university,
            "program_name": extracted.program_name,
            "degree_type": extracted.degree_type,
            "specialization": extracted.specialization,
            "teaching_experience_requirement_text": extracted.teaching_experience_requirement_text,
            "principal_certification_requirement_text": extracted.principal_certification_requirement_text,
            "reference_urls_cleaned": all_urls,
        },
        info_type="extracted_program_info",
    )

    # Reference URL existence check (Critical)
    # This enforces that at least one URL is provided; other critical siblings will be auto-skipped if this fails.
    evaluator.add_custom_node(
        result=len(all_urls) > 0,
        id="Reference_URL",
        desc="Provide a valid URL reference documenting these requirements",
        parent=main_node,
        critical=True,
    )

    # Program degree + specialization (Critical group)
    prog_node = evaluator.add_parallel(
        id="Program_Degree_and_Specialization",
        desc="Verify the program offers Ed.S. degree specifically in Educational Leadership",
        parent=main_node,
        critical=True,
    )

    # Degree Type Verification (Critical leaf)
    degree_leaf = evaluator.add_leaf(
        id="Degree_Type_Verification",
        desc="The program must offer an Education Specialist (Ed.S.) degree",
        parent=prog_node,
        critical=True,
    )

    degree_claim = (
        "This webpage describes an Education Specialist degree program (Ed.S., EdS, or Specialist in Education) "
        "rather than a master's (M.Ed./MA) or doctoral (Ed.D./Ph.D.) program."
    )
    degree_add_ins = (
        "Accept synonyms such as 'Ed.S.', 'EdS', 'Education Specialist', or 'Specialist in Education'. "
        "Do not accept pages that only describe M.Ed., Ed.D., or Ph.D. degrees."
    )

    # Specialization Verification (Critical leaf)
    spec_leaf = evaluator.add_leaf(
        id="Specialization_Verification",
        desc="The specialization must be in Educational Leadership",
        parent=prog_node,
        critical=True,
    )

    spec_claim = (
        "This webpage indicates the program specialization/field is Educational Leadership."
    )
    spec_add_ins = (
        "Accept reasonable naming variants such as 'Educational Leadership', "
        "'Educational Leadership & Administration', 'School Leadership', or 'K-12 Educational Leadership' "
        "if clearly describing the Educational Leadership specialization. "
        "Do not accept unrelated specializations."
    )

    # Admission Requirements (Critical group)
    admit_node = evaluator.add_parallel(
        id="Admission_Requirements",
        desc="Verify the specific admission requirements for the program",
        parent=main_node,
        critical=True,
    )

    # Teaching Experience Requirement (Critical leaf)
    teach_leaf = evaluator.add_leaf(
        id="Teaching_Experience_Requirement",
        desc="The program requires a minimum of three years of teaching experience",
        parent=admit_node,
        critical=True,
    )

    teach_claim = (
        "The admission requirements on this webpage include a minimum of three (3) years of teaching experience."
    )
    teach_add_ins = (
        "Explicitly confirm phrases like 'at least three years', 'minimum of 3 years', or '3+ years' "
        "of teaching experience. Numeric or word forms (e.g., 'three (3)') should count as a match."
    )

    # Principal Certification Requirement (Critical leaf)
    principal_leaf = evaluator.add_leaf(
        id="Principal_Certification_Requirement",
        desc="The program requires applicants to be certified as a school principal",
        parent=admit_node,
        critical=True,
    )

    principal_claim = (
        "The admission requirements on this webpage require applicants to hold current certification as a school principal."
    )
    principal_add_ins = (
        "Accept equivalent language such as 'principal certificate', 'principal license', "
        "'principal endorsement', 'administrator/principal license', or 'eligible for principal certification', "
        "provided it clearly indicates certification as a school principal is required. "
        "If only teacher certification is required (and not principal certification), this should fail."
    )

    # Prepare batch verifications; critical sibling 'Reference_URL' will auto-gate these if it fails
    claims_and_sources = [
        (degree_claim, all_urls, degree_leaf, degree_add_ins),
        (spec_claim, all_urls, spec_leaf, spec_add_ins),
        (teach_claim, all_urls, teach_leaf, teach_add_ins),
        (principal_claim, all_urls, principal_leaf, principal_add_ins),
    ]

    await evaluator.batch_verify(claims_and_sources)


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
) -> Dict[str, Any]:
    """
    Evaluate an answer for the Ed.S. Educational Leadership admissions requirements task.
    """
    # Initialize evaluator
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Overall criteria evaluated in parallel with critical gating
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

    # Extract structured info from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_program_info(),
        template_class=ProgramExtraction,
        extraction_name="program_extraction",
    )

    # Add GT-style context for transparency (not strict GT, but expected criteria)
    evaluator.add_ground_truth(
        {
            "required_degree_type": "Ed.S. (Education Specialist)",
            "required_specialization": "Educational Leadership",
            "required_admission_criteria": [
                "Minimum of three (3) years teaching experience",
                "Current certification as a school principal",
            ],
            "must_provide_reference_url": True,
        },
        gt_type="expected_criteria",
    )

    # Build verification tree and run checks
    await build_and_verify_tree(evaluator, root, extracted)

    # Return structured summary
    return evaluator.get_summary()