import asyncio
import logging
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "ca_arch_reciprocity_exam_requirements"
TASK_DESCRIPTION = (
    "An architect currently holds an active NCARB Certificate and is licensed to practice architecture "
    "in good standing in another U.S. state. This architect now wishes to obtain reciprocal licensure in "
    "California to practice there. What are the complete examination requirements that this architect must "
    "fulfill to obtain California licensure? Specifically, identify all examinations that must be completed, "
    "whether holding an NCARB Certificate provides any exemptions from California's examination requirements, "
    "and provide the specific format and characteristics of any state-specific examinations required."
)


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class NcarbAreInfo(BaseModel):
    """
    Information in the answer related to ARE status via NCARB Certificate.
    """
    ncarb_implies_are_completed_statement: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class CseInfo(BaseModel):
    """
    Information in the answer related to California Supplemental Examination (CSE).
    """
    cse_required_statement: Optional[str] = None
    cse_no_exemption_statement: Optional[str] = None
    cse_format_statement: Optional[str] = None
    cse_length_questions_statement: Optional[str] = None
    cse_time_limit_statement: Optional[str] = None
    cse_ca_specific_content_statement: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class ExamRequirementsExtraction(BaseModel):
    """
    Complete extraction model capturing ARE via NCARB and CSE details plus any cited URLs.
    """
    ncarb_are: Optional[NcarbAreInfo] = None
    cse: Optional[CseInfo] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_exam_requirements() -> str:
    return """
    Extract, from the provided answer, all statements and URLs related to California reciprocal licensure exam requirements for an NCARB-certified architect.

    Organize your extraction into two sections:

    1) ncarb_are:
       - ncarb_implies_are_completed_statement: The exact or paraphrased statement (if present) indicating that holding an active NCARB Certificate implies the ARE requirement is satisfied (e.g., completion of all divisions of ARE 5.0).
       - sources: An array of all URLs explicitly cited in the answer that substantively support the NCARB/ARE statement.

    2) cse:
       - cse_required_statement: The statement indicating California requires passage of the California Supplemental Examination (CSE) for licensure, including reciprocal applicants.
       - cse_no_exemption_statement: The statement indicating there is no waiver/exemption from the CSE requirement for NCARB Certificate holders.
       - cse_format_statement: The statement describing the CSE as a computer-delivered, multiple-choice examination.
       - cse_length_questions_statement: The statement indicating the CSE consists of 100 multiple-choice questions (if provided).
       - cse_time_limit_statement: The statement indicating the CSE has a time limit of 3.5 hours (or equivalently “3 hours 30 minutes”).
       - cse_ca_specific_content_statement: The statement that the CSE tests entry-level competence in areas specific to California architectural practice, covering CA-specific requirements/conditions.
       - sources: An array of all URLs explicitly cited in the answer that substantively support the CSE-related statements.

    URL extraction rules:
    - Extract only URLs explicitly present in the answer (plain URLs or in markdown link form).
    - If a URL lacks http/https protocol, prepend http://.
    - If the answer provides no URLs for a section, return an empty list for 'sources'.

    If a particular statement is not present in the answer, return null for that field.
    """


# --------------------------------------------------------------------------- #
# Verification helpers                                                        #
# --------------------------------------------------------------------------- #
async def verify_are_status_via_ncarb(
    evaluator: Evaluator,
    parent_node,
    extracted: ExamRequirementsExtraction,
) -> None:
    """
    Build the ARE via NCARB node and verify the claim.
    """
    are_parent = evaluator.add_parallel(
        id="ARE_Status_Via_NCARB",
        desc="Addresses whether the national examination requirement (ARE) is satisfied via holding an active NCARB Certificate.",
        parent=parent_node,
        critical=True,
    )

    leaf_ncarb_implies_are = evaluator.add_leaf(
        id="NCARB_Implies_ARE_Completed",
        desc="States that holding an active NCARB Certificate implies completion of all six divisions of ARE 5.0 (i.e., the ARE requirement is already satisfied).",
        parent=are_parent,
        critical=True,
    )

    ncarb_sources = []
    if extracted and extracted.ncarb_are and extracted.ncarb_are.sources:
        ncarb_sources = extracted.ncarb_are.sources

    claim = (
        "Holding an active NCARB Certificate implies the holder has completed the Architect Registration Examination (ARE 5.0) "
        "across all required divisions, meaning the national ARE requirement is satisfied."
    )

    await evaluator.verify(
        claim=claim,
        node=leaf_ncarb_implies_are,
        sources=ncarb_sources if ncarb_sources else None,
        additional_instruction=(
            "Confirm via the cited source(s) that NCARB Certification requires passing the ARE (ARE 5.0 divisions). "
            "Equivalent phrasing such as 'must pass the ARE' should be considered supporting evidence."
        ),
    )


async def verify_cse_requirements(
    evaluator: Evaluator,
    parent_node,
    extracted: ExamRequirementsExtraction,
) -> None:
    """
    Build the California CSE parent node and verify each state-specific exam detail.
    """
    cse_parent = evaluator.add_parallel(
        id="California_Supplemental_Examination_CSE",
        desc="Covers California’s state-specific examination requirement (CSE), including exemptions (if any) and exam format/characteristics.",
        parent=parent_node,
        critical=True,
    )

    cse_sources = []
    if extracted and extracted.cse and extracted.cse.sources:
        cse_sources = extracted.cse.sources

    # Define all leaf nodes first
    leaf_required = evaluator.add_leaf(
        id="CSE_Is_Required",
        desc="States that California requires passage of the California Supplemental Examination (CSE) for licensure (including reciprocal applicants).",
        parent=cse_parent,
        critical=True,
    )

    leaf_no_exemption = evaluator.add_leaf(
        id="No_CSE_Exemption_For_NCARB",
        desc="States there is no waiver/exemption from the CSE requirement for NCARB Certificate holders.",
        parent=cse_parent,
        critical=True,
    )

    leaf_format = evaluator.add_leaf(
        id="CSE_Format",
        desc="States the CSE is a computer-delivered, multiple-choice examination.",
        parent=cse_parent,
        critical=True,
    )

    leaf_length = evaluator.add_leaf(
        id="CSE_Length_Questions",
        desc="States the CSE consists of 100 multiple-choice questions.",
        parent=cse_parent,
        critical=True,
    )

    leaf_time_limit = evaluator.add_leaf(
        id="CSE_Time_Limit",
        desc="States the CSE has a time limit of 3.5 hours.",
        parent=cse_parent,
        critical=True,
    )

    leaf_ca_specific = evaluator.add_leaf(
        id="CSE_CA_Specific_Content",
        desc="States the CSE tests entry-level competence in areas specific to California architectural practice / covers unique CA-specific requirements and conditions.",
        parent=cse_parent,
        critical=True,
    )

    claims_and_sources: List[tuple[str, Optional[List[str]] | Optional[str], Any, Optional[str]]] = [
        (
            "California requires passage of the California Supplemental Examination (CSE) for licensure, including for reciprocal applicants.",
            cse_sources if cse_sources else None,
            leaf_required,
            (
                "Look for Board policy indicating the CSE is required for all applicants seeking licensure in California, "
                "including those applying by reciprocity. Phrases like 'all applicants must pass the CSE' support this."
            ),
        ),
        (
            "There is no waiver or exemption from the California Supplemental Examination (CSE) requirement for NCARB Certificate holders.",
            cse_sources if cse_sources else None,
            leaf_no_exemption,
            (
                "Confirm that NCARB Certification does not exempt applicants from the CSE. "
                "Evidence includes statements that all applicants must pass the CSE and no exemptions/waivers are offered."
            ),
        ),
        (
            "The California Supplemental Examination (CSE) is a computer-delivered, multiple-choice examination.",
            cse_sources if cse_sources else None,
            leaf_format,
            (
                "Accept synonymous phrasing such as 'computer-based' or 'selected-response/multiple-choice items' administered at a test center."
            ),
        ),
        (
            "The California Supplemental Examination (CSE) consists of 100 multiple-choice questions.",
            cse_sources if cse_sources else None,
            leaf_length,
            (
                "Accept equivalent language such as '100 items' or '100 scored questions' where clearly indicating multiple-choice."
            ),
        ),
        (
            "The California Supplemental Examination (CSE) has a time limit of 3.5 hours (i.e., 3 hours and 30 minutes).",
            cse_sources if cse_sources else None,
            leaf_time_limit,
            (
                "Accept equivalent phrasing like '3 hours 30 minutes' or '210 minutes'."
            ),
        ),
        (
            "The California Supplemental Examination (CSE) tests entry-level competence in areas specific to California architectural practice, covering California-specific requirements and conditions.",
            cse_sources if cse_sources else None,
            leaf_ca_specific,
            (
                "Look for descriptions of CSE content focused on California laws, regulations, practice conditions, and unique state requirements at an entry-level competency."
            ),
        ),
    ]

    await evaluator.batch_verify(claims_and_sources)


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
    Evaluate an answer describing California reciprocal licensure examination requirements for an NCARB-certified architect.
    """
    # Initialize evaluator and root
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # The two main sections can be checked independently
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

    # Extract structured information from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_exam_requirements(),
        template_class=ExamRequirementsExtraction,
        extraction_name="exam_requirements_extraction",
    )

    # Add the top-level critical node consolidating all exam requirement verifications
    complete_req_node = evaluator.add_parallel(
        id="Complete_Examination_Requirements",
        desc=(
            "States the complete examination requirements for an NCARB-certified architect seeking reciprocal licensure "
            "in California, including any exemptions and details of state-specific exams."
        ),
        parent=root,
        critical=True,
    )

    # Build and verify ARE via NCARB section
    await verify_are_status_via_ncarb(evaluator, complete_req_node, extracted)

    # Build and verify California CSE section
    await verify_cse_requirements(evaluator, complete_req_node, extracted)

    # Optional: record ground truth expectations (for transparency)
    evaluator.add_ground_truth({
        "expectations": {
            "ARE_status_via_NCARB": "NCARB Certificate holders have satisfied the national ARE requirement.",
            "CSE_required": "All applicants (including reciprocity) must pass the California Supplemental Examination (CSE).",
            "No_CSE_exemption": "NCARB Certificate does not waive the CSE.",
            "CSE_format": "Computer-delivered, multiple-choice.",
            "CSE_length": "100 multiple-choice questions.",
            "CSE_time_limit": "3.5 hours.",
            "CSE_content": "Entry-level competence, California-specific practice requirements and conditions.",
        }
    }, gt_type="expected_requirements")

    # Return structured summary
    return evaluator.get_summary()