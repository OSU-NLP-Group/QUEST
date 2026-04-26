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
TASK_ID = "tri_state_teacher_prep_selection"
TASK_DESCRIPTION = """
A prospective teacher is evaluating universities across three states to determine where to pursue teacher certification. They want to identify one university in each of Pennsylvania, Maryland, and Oklahoma that meets the following requirements: (1) The university must be located in the specified state; (2) The university's teacher preparation programs must hold CAEP (Council for the Accreditation of Educator Preparation) accreditation; (3) The university's teacher preparation programs must be approved by the respective state education department (Pennsylvania Department of Education, Maryland State Department of Education, or Oklahoma State Department of Education); (4) The university must require a minimum cumulative GPA of 3.0 for admission to its teacher preparation programs; (5) The university must offer teacher certification in at least 3 different subject areas or grade levels; (6) The university's teacher preparation program must be currently operational and accepting students; (7) For Oklahoma specifically, the University of Tulsa must be excluded as it lost state accreditation for its teacher preparation program in 2017. Identify one qualifying university in each of the three states and provide reference URLs that verify each requirement.
"""


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class StateSelection(BaseModel):
    name: Optional[str] = None

    # Source URLs grouped by requirement
    location_urls: List[str] = Field(default_factory=list)
    regional_accreditation_urls: List[str] = Field(default_factory=list)
    caep_urls: List[str] = Field(default_factory=list)
    state_approval_urls: List[str] = Field(default_factory=list)
    initial_cert_urls: List[str] = Field(default_factory=list)
    undergraduate_program_urls: List[str] = Field(default_factory=list)
    certification_areas_urls: List[str] = Field(default_factory=list)
    operational_status_urls: List[str] = Field(default_factory=list)
    gpa_requirement_urls: List[str] = Field(default_factory=list)

    # Optional auxiliary fields from the answer (free text or lists)
    gpa_text: Optional[str] = None
    certification_areas_list: List[str] = Field(default_factory=list)


class TeacherPrepSelection(BaseModel):
    pennsylvania: Optional[StateSelection] = None
    maryland: Optional[StateSelection] = None
    oklahoma: Optional[StateSelection] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_selection() -> str:
    return """
    Extract exactly one selected university per each of the following states from the answer: Pennsylvania, Maryland, and Oklahoma.
    For each state, return the university name and group every reference URL under the correct requirement bucket.
    Only include explicit URLs present in the answer (plain or markdown links).
    Do not fabricate or infer URLs. If a category has no explicit URL mentioned, return an empty list for that category.

    For each state object, include:
    - name: university name (string)
    - location_urls: list of URLs that show the university is located in the state (e.g., About/Contact page listing address)
    - regional_accreditation_urls: list of URLs proving institutional "regional" accreditation (e.g., accreditor directory pages like MSCHE, HLC, SACSCOC, NECHE, NWCCU, WSCUC, or the university's accreditation page)
    - caep_urls: list of URLs proving CAEP accreditation for the educator preparation provider/programs (e.g., CAEP provider directory, institution's page citing CAEP)
    - state_approval_urls: list of URLs proving state approval (PDE for Pennsylvania, MSDE for Maryland, OSDE for Oklahoma; state department lists/directories/approval pages)
    - initial_cert_urls: list of URLs proving the institution offers initial teacher certification/licensure (not just add-on/advanced endorsements)
    - undergraduate_program_urls: URLs proving there is an undergraduate (bachelor's-level) pathway that leads to initial teacher certification
    - certification_areas_urls: URLs listing the certification/licensure areas the institution offers
    - operational_status_urls: URLs indicating the program is currently active and accepting students/applications (e.g., current application/admissions/program page)
    - gpa_requirement_urls: URLs proving the minimum cumulative GPA requirement for admission to the teacher preparation program(s)
    - gpa_text: the GPA requirement phrase as stated in the answer (string if present; otherwise null)
    - certification_areas_list: list of the distinct certification/grade-level areas the answer explicitly listed for that university (empty if not listed)

    Return a JSON object with three top-level fields: 'pennsylvania', 'maryland', 'oklahoma', each being a state object as defined above.
    If the answer does not include a university for a state, set the field to null.
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _state_params(state_key: str) -> Dict[str, str]:
    state_key = state_key.lower()
    if state_key == "pennsylvania":
        return {
            "prefix": "PA",
            "state_full": "Pennsylvania",
            "dept_full": "Pennsylvania Department of Education",
            "dept_acronym": "PDE"
        }
    if state_key == "maryland":
        return {
            "prefix": "MD",
            "state_full": "Maryland",
            "dept_full": "Maryland State Department of Education",
            "dept_acronym": "MSDE"
        }
    if state_key == "oklahoma":
        return {
            "prefix": "OK",
            "state_full": "Oklahoma",
            "dept_full": "Oklahoma State Department of Education",
            "dept_acronym": "OSDE"
        }
    raise ValueError(f"Unsupported state_key: {state_key}")


async def _add_status_and_reference(
    evaluator: Evaluator,
    parent_node,
    *,
    status_leaf_id: str,
    status_desc: str,
    claim: str,
    sources: List[str],
    reference_leaf_id: str,
    reference_desc: str,
    additional_instruction: str,
    critical: bool = True
):
    """
    Create a pair of leaves under a SEQUENTIAL parent:
      1) Reference existence (custom, critical)
      2) Status verification (URL-grounded, critical), blocked if reference fails
    The creation order follows the rubric (status first, then reference), but verification
    of the status leaf is conditioned on the reference leaf via explicit prerequisites.
    """
    # Create status leaf first (to match rubric order)
    status_leaf = evaluator.add_leaf(
        id=status_leaf_id,
        desc=status_desc,
        parent=parent_node,
        critical=critical
    )

    # Create reference existence as a custom leaf (critical)
    ref_exists = evaluator.add_custom_node(
        result=bool(sources),
        id=reference_leaf_id,
        desc=reference_desc,
        parent=parent_node,
        critical=critical
    )

    # Now verify status, but require the reference existence as a prerequisite
    await evaluator.verify(
        claim=claim,
        node=status_leaf,
        sources=sources,
        additional_instruction=additional_instruction,
        extra_prerequisites=[ref_exists]
    )
    return status_leaf, ref_exists


# --------------------------------------------------------------------------- #
# State verification builder                                                  #
# --------------------------------------------------------------------------- #
async def verify_state_university(
    evaluator: Evaluator,
    task_parent,
    state_key: str,
    selection: Optional[StateSelection]
) -> None:
    params = _state_params(state_key)
    prefix = params["prefix"]
    state_full = params["state_full"]
    dept_full = params["dept_full"]
    dept_acr = params["dept_acronym"]

    sel = selection or StateSelection()
    uni_name = sel.name or "Unknown University"

    # Create the state container node (non-critical, parallel grouping)
    state_node = evaluator.add_parallel(
        id=f"{state_full}_University",
        desc=f"Identify a {state_full} university meeting all teacher preparation criteria",
        parent=task_parent,
        critical=False
    )

    # 1) Geographic location (critical leaf)
    loc_leaf = evaluator.add_leaf(
        id=f"{prefix}_Geographic_Location",
        desc=f"University is located in {state_full}",
        parent=state_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"{uni_name} is a university located in {state_full}.",
        node=loc_leaf,
        sources=sel.location_urls,
        additional_instruction=f"Verify the institution is in {state_full}. Accept official pages listing campus address or About/Contact pages. If multiple campuses exist in different states, confirm the primary/main campus is in {state_full}."
    )

    # 2) Institutional Accreditation (sequential)
    inst_acc_node = evaluator.add_sequential(
        id=f"{prefix}_Institutional_Accreditation",
        desc="University is regionally accredited as an institution of higher education",
        parent=state_node,
        critical=True
    )
    await _add_status_and_reference(
        evaluator,
        inst_acc_node,
        status_leaf_id=f"{prefix}_Regional_Accreditation_Status",
        status_desc="University holds regional accreditation",
        claim=f"{uni_name} holds institutional regional accreditation from a recognized U.S. regional accreditor (e.g., MSCHE, HLC, SACSCOC, NECHE, NWCCU, or WSCUC).",
        sources=sel.regional_accreditation_urls,
        reference_leaf_id=f"{prefix}_Regional_Accreditation_Reference",
        reference_desc="Reference URL for regional accreditation status",
        additional_instruction="Confirm institutional (university-wide) regional accreditation by a recognized regional accreditor, not just programmatic accreditation. Accept official accreditor directory pages or the university's accreditation page explicitly stating the regional accreditor."
    )

    # 3) Accreditation Status (parallel): CAEP and State approval
    accred_status_node = evaluator.add_parallel(
        id=f"{prefix}_Accreditation_Status",
        desc="University meets national and state accreditation requirements for teacher preparation",
        parent=state_node,
        critical=True
    )

    # 3a) CAEP Accreditation (sequential)
    caep_node = evaluator.add_sequential(
        id=f"{prefix}_CAEP_Accreditation",
        desc="University holds CAEP accreditation for its teacher preparation programs",
        parent=accred_status_node,
        critical=True
    )
    await _add_status_and_reference(
        evaluator,
        caep_node,
        status_leaf_id=f"{prefix}_CAEP_Status",
        status_desc="Programs are CAEP-accredited",
        claim=f"{uni_name}'s educator preparation provider (EPP) or teacher preparation programs are accredited by CAEP (Council for the Accreditation of Educator Preparation).",
        sources=sel.caep_urls,
        reference_leaf_id=f"{prefix}_CAEP_Reference",
        reference_desc="Reference URL for CAEP accreditation status",
        additional_instruction="Look for explicit 'CAEP accredited' statements, CAEP provider directory entries, or official institutional pages citing CAEP accreditation. Do not treat legacy-only NCATE/TEAC mentions as CAEP unless the page clearly states CAEP accreditation."
    )

    # 3b) State Approval (sequential)
    state_approval_node = evaluator.add_sequential(
        id=f"{prefix}_State_Approval",
        desc=f"University has {dept_acr}-approved teacher preparation programs",
        parent=accred_status_node,
        critical=True
    )
    await _add_status_and_reference(
        evaluator,
        state_approval_node,
        status_leaf_id=f"{prefix}_{dept_acr}_Approval_Status",
        status_desc=f"Programs approved by {dept_full}",
        claim=f"{uni_name}'s teacher preparation program(s) are approved by the {dept_full}.",
        sources=sel.state_approval_urls,
        reference_leaf_id=f"{prefix}_State_Approval_Reference",
        reference_desc=f"Reference URL for {dept_acr} approval status",
        additional_instruction=f"Verify on {dept_full}'s official site or equivalent state directory that the institution/program is an approved educator preparation provider and/or specific programs are approved."
    )

    # 4) Program Characteristics (parallel): initial cert, undergraduate level, breadth, operational status
    prog_node = evaluator.add_parallel(
        id=f"{prefix}_Program_Characteristics",
        desc="University's teacher preparation program meets specified operational and breadth requirements",
        parent=state_node,
        critical=True
    )

    # 4a) Certification Type (sequential)
    cert_type_node = evaluator.add_sequential(
        id=f"{prefix}_Certification_Type",
        desc="Programs lead to initial teacher certification",
        parent=prog_node,
        critical=True
    )
    await _add_status_and_reference(
        evaluator,
        cert_type_node,
        status_leaf_id=f"{prefix}_Initial_Certification",
        status_desc="Offers initial teacher certification (not only add-on certifications)",
        claim=f"{uni_name} offers initial teacher certification/licensure programs (not only add-on endorsements).",
        sources=sel.initial_cert_urls,
        reference_leaf_id=f"{prefix}_Certification_Type_Reference",
        reference_desc="Reference URL for initial certification offerings",
        additional_instruction="Accept phrases like 'initial teacher certification', 'initial licensure', or equivalent indicating entry-level certification leading programs."
    )

    # 4b) Program Level (sequential)
    level_node = evaluator.add_sequential(
        id=f"{prefix}_Program_Level",
        desc="Programs offer undergraduate-level pathways to certification",
        parent=prog_node,
        critical=True
    )
    await _add_status_and_reference(
        evaluator,
        level_node,
        status_leaf_id=f"{prefix}_Undergraduate_Programs",
        status_desc="Offers bachelor's degree programs leading to teacher certification",
        claim=f"{uni_name} offers undergraduate (bachelor's-level) pathways that lead to initial teacher certification/licensure.",
        sources=sel.undergraduate_program_urls,
        reference_leaf_id=f"{prefix}_Program_Level_Reference",
        reference_desc="Reference URL for undergraduate program availability",
        additional_instruction="Look for bachelor's degree programs that explicitly state they lead to teacher certification/licensure (e.g., BA/BS in Education leading to certification)."
    )

    # 4c) Certification Breadth (sequential)
    breadth_node = evaluator.add_sequential(
        id=f"{prefix}_Certification_Breadth",
        desc="University offers teacher certification in multiple subject areas",
        parent=prog_node,
        critical=True
    )
    await _add_status_and_reference(
        evaluator,
        breadth_node,
        status_leaf_id=f"{prefix}_Multiple_Certifications",
        status_desc="Offers at least 3 different teacher certification areas",
        claim=f"{uni_name} offers at least three distinct teacher certification/licensure areas or grade levels.",
        sources=sel.certification_areas_urls,
        reference_leaf_id=f"{prefix}_Certification_Reference",
        reference_desc="Reference URL for certification areas offered",
        additional_instruction="Check program lists showing multiple certification/endorsement areas (e.g., Elementary, Secondary Math, Secondary English, Special Education, Early Childhood). It must be at least three distinct areas/grade levels."
    )

    # 4d) Operational Status (sequential)
    op_node = evaluator.add_sequential(
        id=f"{prefix}_Operational_Status",
        desc="University's teacher preparation program is currently operational",
        parent=prog_node,
        critical=True
    )
    await _add_status_and_reference(
        evaluator,
        op_node,
        status_leaf_id=f"{prefix}_Currently_Operational",
        status_desc="Program currently accepts students",
        claim=f"{uni_name}'s teacher preparation program is currently active and accepting new students/applications.",
        sources=sel.operational_status_urls,
        reference_leaf_id=f"{prefix}_Operational_Reference",
        reference_desc="Reference URL for operational status",
        additional_instruction="Accept current admissions/apply pages, program pages stating applications are open, or catalog pages indicating the program is active (avoid archived/defunct notices)."
    )

    # 5) Admission Requirements (sequential)
    admit_node = evaluator.add_sequential(
        id=f"{prefix}_Admission_Requirements",
        desc="University's admission standards meet the specified GPA threshold",
        parent=state_node,
        critical=True
    )
    await _add_status_and_reference(
        evaluator,
        admit_node,
        status_leaf_id=f"{prefix}_GPA_Requirement",
        status_desc="Requires minimum 3.0 GPA for admission to teacher preparation program",
        claim=f"{uni_name} requires a minimum cumulative GPA of at least 3.0 for admission to its teacher preparation program(s).",
        sources=sel.gpa_requirement_urls,
        reference_leaf_id=f"{prefix}_GPA_Reference",
        reference_desc="Reference URL for GPA admission requirements",
        additional_instruction="Verify that the minimum GPA threshold for admission to the teacher preparation program(s) is 3.0 or higher. Accept policies stating 'minimum 3.0 GPA', 'GPA of 3.0 or above', etc."
    )

    # 6) Oklahoma-specific exclusion check (critical)
    if prefix == "OK":
        is_not_tulsa = False
        if sel.name:
            lowered = sel.name.strip().lower()
            is_not_tulsa = lowered not in {
                "university of tulsa",
                "the university of tulsa"
            }
        evaluator.add_custom_node(
            result=is_not_tulsa,
            id="OK_Exclusion_Check",
            desc="University is not the University of Tulsa, which lost state accreditation in 2017",
            parent=state_node,
            critical=True
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
    Evaluate an answer for the tri-state teacher preparation program selection task.
    """
    # Initialize evaluator with a parallel root (states are independent)
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

    # Top-level task node (parallel across states)
    task_node = evaluator.add_parallel(
        id="Task_Completion",
        desc="Identify one university in each of Pennsylvania, Maryland, and Oklahoma that meets all specified teacher preparation program criteria",
        parent=root,
        critical=False
    )

    # Extract structured selection from the answer
    selection: TeacherPrepSelection = await evaluator.extract(
        prompt=prompt_extract_selection(),
        template_class=TeacherPrepSelection,
        extraction_name="tri_state_selection"
    )

    # Add custom info summary
    evaluator.add_custom_info(
        info={
            "pennsylvania_university": selection.pennsylvania.name if selection.pennsylvania else None,
            "maryland_university": selection.maryland.name if selection.maryland else None,
            "oklahoma_university": selection.oklahoma.name if selection.oklahoma else None
        },
        info_type="extracted_universities",
        info_name="extracted_universities"
    )

    # Build and verify each state's subtree
    await verify_state_university(evaluator, task_node, "pennsylvania", selection.pennsylvania)
    await verify_state_university(evaluator, task_node, "maryland", selection.maryland)
    await verify_state_university(evaluator, task_node, "oklahoma", selection.oklahoma)

    # Return structured summary
    return evaluator.get_summary()