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
TASK_ID = "ca_tx_pe_requirements"
TASK_DESCRIPTION = (
    "A Professional Engineer is researching state-specific licensing requirements as part of their career planning. "
    "They need to understand: (1) What are the two state-specific exams that California requires civil engineering "
    "applicants to pass in addition to the national PE exam? (2) For Texas PE license renewal, how many total "
    "Professional Development Hours (PDH) are required annually, and what is the minimum number of PDH hours that "
    "must be completed in Ethics or Act/Rules each year?"
)


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class CAExamsExtraction(BaseModel):
    """
    California state-specific exams for civil engineering licensure (beyond the national PE exam),
    as stated in the answer.
    """
    exams: List[str] = Field(default_factory=list, description="List of exam names explicitly mentioned in the answer.")
    sources: List[str] = Field(default_factory=list, description="Any URLs cited in the answer for California requirements.")


class TXPDHExtraction(BaseModel):
    """
    Texas annual PDH requirements for PE license renewal, as stated in the answer.
    """
    total_pdh: Optional[str] = Field(default=None, description="Total annual PDH stated in the answer (e.g., '15').")
    ethics_pdh_min: Optional[str] = Field(default=None, description="Minimum annual PDH in Ethics or Act/Rules (e.g., '1').")
    sources: List[str] = Field(default_factory=list, description="Any URLs cited in the answer for Texas renewal requirements.")


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_ca_exams() -> str:
    return """
    Extract the California state-specific exams required for civil engineering licensure beyond the national PE exam,
    as stated in the provided answer. Return:
    - exams: an array of the exam names exactly as mentioned (e.g., "Civil Seismic Principles", "Seismic Principles exam",
      "Civil Engineering Surveying", "Surveying exam"). Include reasonable variants and synonyms if they appear.
    - sources: any URLs explicitly cited in the answer relevant to the California licensure requirements (extract actual URLs).

    Only include California civil engineering state-specific exams (not FE/PE national exams or other states' requirements).
    If the answer mentions more than two exams, include them all as long as they are California-specific exams beyond the national PE.
    If none are mentioned, return an empty array for exams and sources.
    """


def prompt_extract_tx_pdh() -> str:
    return """
    Extract the Texas PE license renewal annual Professional Development Hours (PDH) requirements, as stated in the answer.
    Return:
    - total_pdh: the total number of PDH required annually (as a simple string like '15' if possible; otherwise the phrase).
    - ethics_pdh_min: the minimum annual PDH in Ethics or Act/Rules (as a simple string like '1' if possible; otherwise the phrase).
    - sources: any URLs explicitly cited in the answer relevant to Texas renewal requirements (extract actual URLs).

    If a field is not mentioned, return null for that field. If no URLs are cited, return an empty array for sources.
    Accept variants such as 'hours', 'PDH units', 'per year', 'annually', 'ethics or professional responsibility', 'Act/Rules'.
    """


# --------------------------------------------------------------------------- #
# Verification builders                                                       #
# --------------------------------------------------------------------------- #
async def build_california_checks(
    evaluator: Evaluator,
    parent_node,
) -> None:
    """
    Build California verification subtree:
    - Seismic exam required
    - Surveying exam required
    All nodes are critical, per rubric.
    """
    ca_node = evaluator.add_parallel(
        id="California_Civil_PE_Requirements",
        desc="Identify the two state-specific exams required for California civil engineering licensure beyond the national PE exam",
        parent=parent_node,
        critical=True,
    )

    # Leaf: Seismic exam stated
    seismic_leaf = evaluator.add_leaf(
        id="Seismic_Exam",
        desc="States that the Civil Seismic Principles exam is required",
        parent=ca_node,
        critical=True,
    )
    seismic_claim = (
        "The answer states that California requires civil engineering applicants to pass the Civil Seismic Principles "
        "exam (aka seismic principles/seismic exam) as a state-specific requirement beyond the national PE exam."
    )
    await evaluator.verify(
        claim=seismic_claim,
        node=seismic_leaf,
        additional_instruction=(
            "Your job is to check whether the ANSWER TEXT explicitly mentions or clearly implies that the California "
            "civil licensure requires the Civil Seismic Principles exam. Accept minor wording variants such as "
            "'Seismic Principles exam', 'Civil Seismic exam', 'Seismic exam for civil engineers', etc. "
            "Focus ONLY on whether the answer states this requirement; do not rely on external knowledge."
        ),
    )

    # Leaf: Surveying exam stated
    surveying_leaf = evaluator.add_leaf(
        id="Surveying_Exam",
        desc="States that the Civil Engineering Surveying exam is required",
        parent=ca_node,
        critical=True,
    )
    surveying_claim = (
        "The answer states that California requires civil engineering applicants to pass the Civil Engineering "
        "Surveying exam (aka surveying exam for civil engineers) as a state-specific requirement beyond the national PE exam."
    )
    await evaluator.verify(
        claim=surveying_claim,
        node=surveying_leaf,
        additional_instruction=(
            "Check whether the ANSWER TEXT explicitly mentions or clearly implies that the California civil licensure "
            "requires the Civil Engineering Surveying exam. Accept minor wording variants such as 'Surveying exam', "
            "'Civil Engineering Surveying', etc. Focus ONLY on whether the answer states this requirement; "
            "do not rely on external knowledge."
        ),
    )


async def build_texas_checks(
    evaluator: Evaluator,
    parent_node,
) -> None:
    """
    Build Texas verification subtree:
    - Total PDH annually (15)
    - Ethics or Act/Rules minimum PDH annually (at least 1)
    All nodes are critical, per rubric.
    """
    tx_node = evaluator.add_parallel(
        id="Texas_PE_Continuing_Education",
        desc="Provide the annual continuing education (PDH) requirements for Texas PE license renewal",
        parent=parent_node,
        critical=True,
    )

    # Leaf: Total PDH (15 annually)
    total_pdh_leaf = evaluator.add_leaf(
        id="Total_PDH",
        desc="States the total annual PDH required for renewal (15 PDH annually)",
        parent=tx_node,
        critical=True,
    )
    total_pdh_claim = (
        "The answer states that Texas requires a total of 15 Professional Development Hours (PDH) annually to renew a PE license."
    )
    await evaluator.verify(
        claim=total_pdh_claim,
        node=total_pdh_leaf,
        additional_instruction=(
            "Check the ANSWER TEXT for an explicit statement equivalent to '15 PDH annually' for Texas PE renewal. "
            "Accept reasonable phrasing variants (e.g., '15 hours per year', '15 PDH units annually', etc.). "
            "Focus ONLY on whether the answer states 15 PDH annually; do not rely on external knowledge."
        ),
    )

    # Leaf: Ethics/Act-Rules minimum (at least 1 annually)
    ethics_leaf = evaluator.add_leaf(
        id="Ethics_Requirement",
        desc="States the minimum annual PDH in Ethics or Act/Rules (at least 1 PDH annually)",
        parent=tx_node,
        critical=True,
    )
    ethics_claim = (
        "The answer states that Texas requires at least 1 PDH annually to be completed in Ethics or Act/Rules for PE renewal."
    )
    await evaluator.verify(
        claim=ethics_claim,
        node=ethics_leaf,
        additional_instruction=(
            "Check the ANSWER TEXT for an explicit statement equivalent to 'at least 1 PDH annually in Ethics or Act/Rules'. "
            "Accept phrasing variants such as '1 hour in ethics', '1 PDH in ethics or professional responsibility', or "
            "'1 PDH in rules/laws/regulations (Act/Rules)'. Focus ONLY on whether the answer states this minimum; "
            "do not rely on external knowledge."
        ),
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
    Evaluate an answer for California state-specific civil PE exam requirements and Texas PDH renewal requirements.

    Returns a structured evaluation summary containing:
    - Extraction results
    - Verification tree with node statuses and scores
    - Final aggregated score
    """
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

    # Ground Truth (for reference in summary; not used to judge directly)
    evaluator.add_ground_truth({
        "california_expected_exams": [
            "Civil Seismic Principles",
            "Civil Engineering Surveying",
        ],
        "texas_expected_pdh": {
            "total_pdh_annually": "15",
            "ethics_or_act_rules_min_annually": "1",
        }
    }, gt_type="expected_requirements")

    # Run extractions in parallel
    ca_extract_task = evaluator.extract(
        prompt=prompt_extract_ca_exams(),
        template_class=CAExamsExtraction,
        extraction_name="california_exams"
    )
    tx_extract_task = evaluator.extract(
        prompt=prompt_extract_tx_pdh(),
        template_class=TXPDHExtraction,
        extraction_name="texas_pdh_requirements"
    )
    ca_extraction, tx_extraction = await asyncio.gather(ca_extract_task, tx_extract_task)

    # Build the rubric tree: Root (critical, parallel)
    task_root = evaluator.add_parallel(
        id="Root",
        desc="Answer both parts of the question: California state-specific exams and Texas PE renewal PDH requirements",
        parent=root,
        critical=True,
    )

    # California subtree
    await build_california_checks(evaluator, task_root)

    # Texas subtree
    await build_texas_checks(evaluator, task_root)

    # Optional: Record extra custom information about extraction to aid debugging
    evaluator.add_custom_info(
        info={
            "california_exams_extracted": ca_extraction.exams,
            "california_sources": ca_extraction.sources,
            "texas_total_pdh_extracted": tx_extraction.total_pdh,
            "texas_ethics_min_pdh_extracted": tx_extraction.ethics_pdh_min,
            "texas_sources": tx_extraction.sources,
        },
        info_type="extraction_summary",
        info_name="extraction_overview"
    )

    return evaluator.get_summary()