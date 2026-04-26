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
TASK_ID = "superintendent_cert_analysis"
TASK_DESCRIPTION = (
    "You currently hold a Master of Education degree in Educational Leadership from an accredited university, "
    "a valid Texas principal certificate, and have accumulated 5 years of combined teaching and administrative "
    "experience in Texas public schools. You are exploring superintendent certification opportunities and need "
    "to answer the following: Part 1: What are the complete requirements (degree, prior certification, program "
    "completion, and examination) for obtaining Texas superintendent certification? Provide official reference "
    "URLs for each requirement category. Part 2: Among Washington, California, and North Carolina, which state "
    "requires the LEAST additional degree-level education beyond your current master's degree to qualify for "
    "superintendent certification? Provide the specific degree requirements for all three states with reference "
    "URLs, then identify which requires the minimum additional degree work and explain your reasoning. Part 3: For "
    "the state you identified in Part 2, what are ALL the additional requirements (beyond degree requirements) you "
    "would need to complete to obtain superintendent certification? Your answer must include: experience requirements "
    "(type, duration, and reference URL), prerequisite certifications or licenses needed (specific certificates, "
    "transfer process, and reference URL), required preparation programs (type, approval requirements, and reference "
    "URL), and required assessments or exams (if applicable, with reference URL). Provide comprehensive details with "
    "official reference URLs for each requirement category."
)


# --------------------------------------------------------------------------- #
# Extraction Models                                                           #
# --------------------------------------------------------------------------- #
class TexasRequirements(BaseModel):
    degree_text: Optional[str] = None
    degree_urls: List[str] = Field(default_factory=list)
    prior_cert_text: Optional[str] = None
    prior_cert_urls: List[str] = Field(default_factory=list)
    program_text: Optional[str] = None
    program_urls: List[str] = Field(default_factory=list)
    exam_text: Optional[str] = None
    exam_urls: List[str] = Field(default_factory=list)


class StateDegreeRequirements(BaseModel):
    washington_degree_text: Optional[str] = None
    washington_urls: List[str] = Field(default_factory=list)
    california_degree_text: Optional[str] = None
    california_urls: List[str] = Field(default_factory=list)
    north_carolina_degree_text: Optional[str] = None
    north_carolina_urls: List[str] = Field(default_factory=list)


class SelectionExtraction(BaseModel):
    selected_state: Optional[str] = None  # Expect one of {"Washington", "California", "North Carolina"}
    justification: Optional[str] = None
    selection_urls: List[str] = Field(default_factory=list)


class SelectedStateAdditionalRequirements(BaseModel):
    state_name: Optional[str] = None  # Should match selected_state above
    experience_text: Optional[str] = None
    experience_urls: List[str] = Field(default_factory=list)
    prereq_text: Optional[str] = None
    prereq_urls: List[str] = Field(default_factory=list)
    program_text: Optional[str] = None
    program_urls: List[str] = Field(default_factory=list)
    assessment_text: Optional[str] = None
    assessment_urls: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction Prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_texas_requirements() -> str:
    return (
        "Extract the Texas superintendent certification requirements as presented in the answer. "
        "You must return four categories, each with a concise requirement statement and the official reference URLs cited in the answer. "
        "Categories:\n"
        "1) degree_text: The degree requirement for Texas superintendent certification.\n"
        "   degree_urls: All official URLs cited for the degree requirement.\n"
        "2) prior_cert_text: The prior certification requirement (e.g., principal certificate or allowable alternatives).\n"
        "   prior_cert_urls: All official URLs cited for prior certification requirement.\n"
        "3) program_text: The educator preparation program requirement.\n"
        "   program_urls: All official URLs cited for program requirement.\n"
        "4) exam_text: The examination requirement (e.g., TExES superintendent exam).\n"
        "   exam_urls: All official URLs cited for exam requirement.\n\n"
        "Rules:\n"
        "- Extract statements exactly from the answer; do not invent. If a category is missing, set the text to null and return an empty list for URLs.\n"
        "- For URLs, extract only actual URLs explicitly provided. Include all relevant official references mentioned (TEA or TAC pages preferred but not required if the answer cites others)."
    )


def prompt_extract_state_degree_requirements() -> str:
    return (
        "Extract the superintendent certification degree requirements for Washington, California, and North Carolina "
        "as stated in the answer, with reference URLs for each.\n"
        "Return fields:\n"
        "- washington_degree_text: The Washington degree requirement statement.\n"
        "- washington_urls: All URLs cited for Washington degree requirements.\n"
        "- california_degree_text: The California degree requirement statement (Administrative Services Credential context).\n"
        "- california_urls: All URLs cited for California degree requirements.\n"
        "- north_carolina_degree_text: The North Carolina degree requirement statement.\n"
        "- north_carolina_urls: All URLs cited for North Carolina degree requirements.\n\n"
        "Rules:\n"
        "- Use exactly what the answer states. If any state's degree requirement is missing, set its text to null and return an empty list of URLs."
    )


def prompt_extract_selection() -> str:
    return (
        "Extract the state identified in the answer as requiring the least additional degree-level education beyond a master's degree "
        "among Washington, California, and North Carolina, along with the justification and reference URLs.\n"
        "Return fields:\n"
        "- selected_state: The chosen state name (expected one of 'Washington', 'California', 'North Carolina').\n"
        "- justification: The explanation provided in the answer for the selection.\n"
        "- selection_urls: All URLs cited specifically to support the selection.\n\n"
        "Rules:\n"
        "- If the state is not clearly identified, set selected_state to null.\n"
        "- If no justification is present, set justification to null.\n"
        "- Extract only URLs explicitly provided in the answer."
    )


def prompt_extract_selected_state_additional_requirements() -> str:
    return (
        "For the state identified in Part 2 (the selection in the answer), extract ALL additional requirements beyond degree "
        "needed for superintendent certification, including official reference URLs. If the answer did not clearly identify a state, "
        "extract the additional requirements for the state the answer discusses in Part 3.\n"
        "Return fields:\n"
        "- state_name: The state for which the additional requirements are described.\n"
        "- experience_text: The experience requirements (type and duration).\n"
        "- experience_urls: URLs cited for experience.\n"
        "- prereq_text: Prerequisite certifications/licenses and any transfer/application process details.\n"
        "- prereq_urls: URLs cited for prerequisite certifications/licenses.\n"
        "- program_text: Required preparation program type and approval requirements.\n"
        "- program_urls: URLs cited for preparation programs.\n"
        "- assessment_text: Required exams or assessments, or 'no exam required' if the answer states that.\n"
        "- assessment_urls: URLs cited for assessments/exams if applicable.\n\n"
        "Rules:\n"
        "- Extract exactly from the answer; do not add or infer.\n"
        "- For any missing category, set text to null and return an empty list for URLs."
    )


# --------------------------------------------------------------------------- #
# Helper Functions                                                            #
# --------------------------------------------------------------------------- #
def _has_text_and_urls(text: Optional[str], urls: List[str]) -> bool:
    return bool(text and text.strip()) and bool(urls and len(urls) > 0)


def _norm_state_name(name: Optional[str]) -> str:
    if not name:
        return ""
    n = name.strip().lower()
    if "wash" in n:
        return "Washington"
    if "calif" in n:
        return "California"
    if "north car" in n or n == "nc":
        return "North Carolina"
    return name.strip()


# --------------------------------------------------------------------------- #
# Verification Builders                                                       #
# --------------------------------------------------------------------------- #
async def verify_texas_requirements(evaluator: Evaluator, parent_node, texas: TexasRequirements) -> None:
    """
    Build verification nodes for Texas requirements (degree, prior cert, program, exam).
    Parent is a critical parallel node.
    Each requirement is a critical sequential sub-node with existence check and source-supported verification.
    """
    tx_node = evaluator.add_parallel(
        id="Texas_Complete_Requirements",
        desc="Identify all requirements for Texas superintendent certification with reference URLs",
        parent=parent_node,
        critical=True
    )

    # Degree
    deg_seq = evaluator.add_sequential(
        id="Texas_Degree_Requirement",
        desc="Texas degree requirement stated with official reference URL(s)",
        parent=tx_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=_has_text_and_urls(texas.degree_text, texas.degree_urls),
        id="Texas_Degree_Requirement_exists",
        desc="Texas degree requirement provided with at least one reference URL",
        parent=deg_seq,
        critical=True
    )
    deg_verify = evaluator.add_leaf(
        id="Texas_Degree_Requirement_supported",
        desc="Texas degree requirement is supported by the cited URL(s)",
        parent=deg_seq,
        critical=True
    )
    await evaluator.verify(
        claim=f"Texas superintendent certification degree requirement: {texas.degree_text or ''}",
        node=deg_verify,
        sources=texas.degree_urls,
        additional_instruction=(
            "Confirm on official Texas sources (e.g., TEA/TAC) whether a master's degree from an accredited institution "
            "is required for the superintendent certificate. Allow minor paraphrasing."
        )
    )

    # Prior Certification
    prior_seq = evaluator.add_sequential(
        id="Texas_Prior_Certification_Requirement",
        desc="Texas prior certification requirement stated with official reference URL(s)",
        parent=tx_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=_has_text_and_urls(texas.prior_cert_text, texas.prior_cert_urls),
        id="Texas_Prior_Certification_Requirement_exists",
        desc="Texas prior certification requirement provided with at least one reference URL",
        parent=prior_seq,
        critical=True
    )
    prior_verify = evaluator.add_leaf(
        id="Texas_Prior_Certification_Requirement_supported",
        desc="Texas prior certification requirement is supported by the cited URL(s)",
        parent=prior_seq,
        critical=True
    )
    await evaluator.verify(
        claim=f"Texas superintendent certification prior certification requirement: {texas.prior_cert_text or ''}",
        node=prior_verify,
        sources=texas.prior_cert_urls,
        additional_instruction=(
            "Verify whether Texas requires a Principal certificate OR explicitly permits TEA-approved managerial experience "
            "in lieu of the principal certificate (if the answer claims that). Use the cited official reference."
        )
    )

    # Program
    prog_seq = evaluator.add_sequential(
        id="Texas_Program_Requirement",
        desc="Texas program completion requirement stated with official reference URL(s)",
        parent=tx_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=_has_text_and_urls(texas.program_text, texas.program_urls),
        id="Texas_Program_Requirement_exists",
        desc="Texas program completion requirement provided with at least one reference URL",
        parent=prog_seq,
        critical=True
    )
    prog_verify = evaluator.add_leaf(
        id="Texas_Program_Requirement_supported",
        desc="Texas program completion requirement is supported by the cited URL(s)",
        parent=prog_seq,
        critical=True
    )
    await evaluator.verify(
        claim=f"Texas superintendent certification educator preparation program requirement: {texas.program_text or ''}",
        node=prog_verify,
        sources=texas.program_urls,
        additional_instruction=(
            "Confirm that completion of an approved superintendent educator preparation program (EPP) is required by Texas, "
            "as stated in the cited official reference."
        )
    )

    # Exam
    exam_seq = evaluator.add_sequential(
        id="Texas_Exam_Requirement",
        desc="Texas examination requirement stated with official reference URL(s)",
        parent=tx_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=_has_text_and_urls(texas.exam_text, texas.exam_urls),
        id="Texas_Exam_Requirement_exists",
        desc="Texas examination requirement provided with at least one reference URL",
        parent=exam_seq,
        critical=True
    )
    exam_verify = evaluator.add_leaf(
        id="Texas_Exam_Requirement_supported",
        desc="Texas examination requirement is supported by the cited URL(s)",
        parent=exam_seq,
        critical=True
    )
    await evaluator.verify(
        claim=f"Texas superintendent certification exam requirement: {texas.exam_text or ''}",
        node=exam_verify,
        sources=texas.exam_urls,
        additional_instruction=(
            "Verify that passing the superintendent TExES (or the current official Texas superintendent assessment) is required, "
            "according to the cited official reference."
        )
    )


async def verify_state_degree_requirements(
    evaluator: Evaluator,
    parent_node,
    states: StateDegreeRequirements
) -> None:
    """
    Build verification nodes for Washington, California, and North Carolina degree requirements.
    Enforce critical verification under the 'Individual_State_Degree_Requirements' node (parallel).
    Each state is a critical sequential sub-node with existence and source-supported verification.
    """
    indiv_node = evaluator.add_parallel(
        id="Individual_State_Degree_Requirements",
        desc="Research and state degree requirements for each comparison state with reference URLs",
        parent=parent_node,
        critical=True
    )

    # Washington
    wa_seq = evaluator.add_sequential(
        id="Washington_Degree_Requirements",
        desc="Washington superintendent degree requirements with reference URL(s)",
        parent=indiv_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=_has_text_and_urls(states.washington_degree_text, states.washington_urls),
        id="Washington_Degree_Requirements_exists",
        desc="Washington degree requirement provided with at least one reference URL",
        parent=wa_seq,
        critical=True
    )
    wa_verify = evaluator.add_leaf(
        id="Washington_Degree_Requirements_supported",
        desc="Washington degree requirements supported by the cited URL(s)",
        parent=wa_seq,
        critical=True
    )
    await evaluator.verify(
        claim=f"Washington superintendent certification degree requirements: {states.washington_degree_text or ''}",
        node=wa_verify,
        sources=states.washington_urls,
        additional_instruction=(
            "Confirm Washington's superintendent certificate degree-related requirements (e.g., master's plus specified graduate credits, "
            "and/or doctorate for professional tiers) per the cited official reference."
        )
    )

    # California
    ca_seq = evaluator.add_sequential(
        id="California_Degree_Requirements",
        desc="California Administrative Services Credential degree requirements with reference URL(s)",
        parent=indiv_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=_has_text_and_urls(states.california_degree_text, states.california_urls),
        id="California_Degree_Requirements_exists",
        desc="California degree requirement provided with at least one reference URL",
        parent=ca_seq,
        critical=True
    )
    ca_verify = evaluator.add_leaf(
        id="California_Degree_Requirements_supported",
        desc="California degree requirements supported by the cited URL(s)",
        parent=ca_seq,
        critical=True
    )
    await evaluator.verify(
        claim=f"California superintendent-related credential degree requirements: {states.california_degree_text or ''}",
        node=ca_verify,
        sources=states.california_urls,
        additional_instruction=(
            "Confirm the degree-level requirements associated with the Administrative Services Credential (Preliminary/Clear) relevant to superintendent roles, "
            "per the cited official reference (e.g., CTC)."
        )
    )

    # North Carolina
    nc_seq = evaluator.add_sequential(
        id="North_Carolina_Degree_Requirements",
        desc="North Carolina superintendent degree requirements with reference URL(s)",
        parent=indiv_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=_has_text_and_urls(states.north_carolina_degree_text, states.north_carolina_urls),
        id="North_Carolina_Degree_Requirements_exists",
        desc="North Carolina degree requirement provided with at least one reference URL",
        parent=nc_seq,
        critical=True
    )
    nc_verify = evaluator.add_leaf(
        id="North_Carolina_Degree_Requirements_supported",
        desc="North Carolina degree requirements supported by the cited URL(s)",
        parent=nc_seq,
        critical=True
    )
    await evaluator.verify(
        claim=f"North Carolina superintendent certification degree requirements: {states.north_carolina_degree_text or ''}",
        node=nc_verify,
        sources=states.north_carolina_urls,
        additional_instruction=(
            "Confirm North Carolina superintendent license routes and degree-related requirements (e.g., specialist/doctoral level or alternative path) "
            "using the cited official reference."
        )
    )


async def verify_minimum_degree_selection(
    evaluator: Evaluator,
    parent_node,
    selection: SelectionExtraction,
    states: StateDegreeRequirements
) -> None:
    """
    Build verification nodes to identify which state requires the least additional degree-level education.
    Parallel critical node with three critical leaves:
    - Selected_State: simple logical verification using the answer context and extracted state requirements.
    - Selection_Justification: logical check that justification explains the comparative reasoning.
    - Selection_Reference_URL: evidence-based verification that the cited URL(s) support the selection.
    """
    min_node = evaluator.add_parallel(
        id="Minimum_Degree_State_Identification",
        desc="Identify which state requires least additional degree work with justification and reference URL",
        parent=parent_node,
        critical=True
    )

    # Selected State
    sel_leaf = evaluator.add_leaf(
        id="Selected_State",
        desc="Correctly identify the state requiring minimum additional degree-level education",
        parent=min_node,
        critical=True
    )
    # Construct a claim that prompts the judge to re-evaluate based on stated requirements in the answer context.
    wa_txt = states.washington_degree_text or ""
    ca_txt = states.california_degree_text or ""
    nc_txt = states.north_carolina_degree_text or ""
    await evaluator.verify(
        claim=(
            f"Among Washington, California, and North Carolina, the state requiring the least additional degree-level education "
            f"beyond a master's degree is {selection.selected_state or ''}."
        ),
        node=sel_leaf,
        additional_instruction=(
            "Use the degree requirement statements provided in the answer:\n"
            f"- Washington: {wa_txt}\n"
            f"- California: {ca_txt}\n"
            f"- North Carolina: {nc_txt}\n"
            "Judge which state requires the least additional degree-level work beyond a master's degree."
        )
    )

    # Selection Justification
    just_leaf = evaluator.add_leaf(
        id="Selection_Justification",
        desc="Provide reasoning explaining why this state requires least degree work",
        parent=min_node,
        critical=True
    )
    await evaluator.verify(
        claim=(
            f"The justification provided explains why {selection.selected_state or ''} requires the least additional "
            f"degree-level education compared with the other two states."
        ),
        node=just_leaf,
        additional_instruction=(
            "Check that the justification compares the degree requirements across Washington, California, and North Carolina, "
            "and clearly argues which involves minimal or no additional graduate credits or advanced degrees."
        )
    )

    # Selection Reference URL supports the selection
    ref_leaf = evaluator.add_leaf(
        id="Selection_Reference_URL",
        desc="Provide reference URL supporting the selection",
        parent=min_node,
        critical=True
    )
    # Combine selection-specific URLs plus the chosen state's degree URLs to strengthen evidence
    chosen = _norm_state_name(selection.selected_state)
    chosen_urls: List[str] = list(selection.selection_urls)
    if chosen == "Washington":
        chosen_urls += states.washington_urls
    elif chosen == "California":
        chosen_urls += states.california_urls
    elif chosen == "North Carolina":
        chosen_urls += states.north_carolina_urls

    await evaluator.verify(
        claim=(
            f"The provided reference(s) support the selection that {chosen} requires the least additional degree-level "
            "education beyond a master's degree among the three states."
        ),
        node=ref_leaf,
        sources=chosen_urls,
        additional_instruction=(
            "Verify that the cited reference(s) substantively support the chosen state's degree requirement claim and the conclusion "
            "that it entails the least additional degree-level education compared to the other two states."
        )
    )


async def verify_selected_state_additional_requirements(
    evaluator: Evaluator,
    parent_node,
    selection: SelectionExtraction,
    addl: SelectedStateAdditionalRequirements
) -> None:
    """
    Build verification nodes for ALL additional requirements beyond degree for the selected state.
    Parent node is set to NON-CRITICAL (to respect framework constraint for a non-critical child).
    Each category is a critical sequential sub-node with existence and source-supported verification,
    except Assessment_Requirements which remains non-critical as per rubric.
    """
    # Parent adjusted to NON-CRITICAL to avoid critical-parent/non-critical-child constraint in framework
    details_node = evaluator.add_parallel(
        id="Selected_State_Additional_Requirements",
        desc="Detail all additional requirements beyond degree for the selected minimum-degree state",
        parent=parent_node,
        critical=False
    )

    state_label = _norm_state_name(addl.state_name or selection.selected_state or "")

    # Experience
    exp_seq = evaluator.add_sequential(
        id="Experience_Requirements",
        desc="Identify and describe experience requirements including type and duration, with reference URL",
        parent=details_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=_has_text_and_urls(addl.experience_text, addl.experience_urls),
        id="Experience_Requirements_exists",
        desc="Experience requirements provided with at least one reference URL",
        parent=exp_seq,
        critical=True
    )
    exp_verify = evaluator.add_leaf(
        id="Experience_Requirements_supported",
        desc="Experience requirements supported by the cited URL(s)",
        parent=exp_seq,
        critical=True
    )
    await evaluator.verify(
        claim=f"{state_label} superintendent certification experience requirements: {addl.experience_text or ''}",
        node=exp_verify,
        sources=addl.experience_urls,
        additional_instruction="Confirm the type and required duration of experience on the cited official reference."
    )

    # Certification Prerequisites
    cert_seq = evaluator.add_sequential(
        id="Certification_Prerequisites",
        desc="Identify prerequisite certifications/licenses, transfer/application process, with reference URL",
        parent=details_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=_has_text_and_urls(addl.prereq_text, addl.prereq_urls),
        id="Certification_Prerequisites_exists",
        desc="Prerequisite certifications/licenses provided with at least one reference URL",
        parent=cert_seq,
        critical=True
    )
    cert_verify = evaluator.add_leaf(
        id="Certification_Prerequisites_supported",
        desc="Prerequisite certifications/licenses supported by the cited URL(s)",
        parent=cert_seq,
        critical=True
    )
    await evaluator.verify(
        claim=f"{state_label} superintendent certification prerequisite certifications/licenses and process: {addl.prereq_text or ''}",
        node=cert_verify,
        sources=addl.prereq_urls,
        additional_instruction=(
            "Verify required certificates/licenses and any transfer/application process details per the cited official reference."
        )
    )

    # Program Completion Requirements
    prog_seq = evaluator.add_sequential(
        id="Program_Completion_Requirements",
        desc="Identify required preparation program type and approval requirements, with reference URL",
        parent=details_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=_has_text_and_urls(addl.program_text, addl.program_urls),
        id="Program_Completion_Requirements_exists",
        desc="Preparation program requirements provided with at least one reference URL",
        parent=prog_seq,
        critical=True
    )
    prog_verify = evaluator.add_leaf(
        id="Program_Completion_Requirements_supported",
        desc="Preparation program requirements supported by the cited URL(s)",
        parent=prog_seq,
        critical=True
    )
    await evaluator.verify(
        claim=f"{state_label} superintendent preparation program requirements: {addl.program_text or ''}",
        node=prog_verify,
        sources=addl.program_urls,
        additional_instruction="Confirm program type and state approval requirements per the cited official reference."
    )

    # Assessment Requirements (non-critical)
    assess_seq = evaluator.add_sequential(
        id="Assessment_Requirements",
        desc="Identify required assessments/exams if applicable, with reference URL",
        parent=details_node,
        critical=False
    )
    evaluator.add_custom_node(
        result=_has_text_and_urls(addl.assessment_text, addl.assessment_urls),
        id="Assessment_Requirements_exists",
        desc="Assessment requirements provided with at least one reference URL",
        parent=assess_seq,
        critical=False
    )
    assess_verify = evaluator.add_leaf(
        id="Assessment_Requirements_supported",
        desc="Assessment requirements supported by the cited URL(s)",
        parent=assess_seq,
        critical=False
    )
    await evaluator.verify(
        claim=f"{state_label} superintendent certification assessments/exams: {addl.assessment_text or ''}",
        node=assess_verify,
        sources=addl.assessment_urls,
        additional_instruction=(
            "Verify assessment or exam requirements (or verify that no exam is required if that is claimed) per the cited reference."
        )
    )


# --------------------------------------------------------------------------- #
# Main evaluation function                                                    #
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
    Entry point for evaluating the superintendent certification analysis answer.
    Constructs the verification tree and runs extraction + verification steps.
    """
    # Initialize evaluator with root sequential strategy (per rubric)
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.SEQUENTIAL,
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

    # Extract all necessary structured information (can be concurrent)
    texas_task = evaluator.extract(
        prompt=prompt_extract_texas_requirements(),
        template_class=TexasRequirements,
        extraction_name="texas_requirements"
    )
    states_task = evaluator.extract(
        prompt=prompt_extract_state_degree_requirements(),
        template_class=StateDegreeRequirements,
        extraction_name="state_degree_requirements"
    )
    selection_task = evaluator.extract(
        prompt=prompt_extract_selection(),
        template_class=SelectionExtraction,
        extraction_name="minimum_degree_selection"
    )
    addl_task = evaluator.extract(
        prompt=prompt_extract_selected_state_additional_requirements(),
        template_class=SelectedStateAdditionalRequirements,
        extraction_name="selected_state_additional_requirements"
    )

    texas, states, selection, addl = await asyncio.gather(
        texas_task, states_task, selection_task, addl_task
    )

    # Build tree: Part 1 – Texas requirements (critical parallel block)
    await verify_texas_requirements(evaluator, root, texas)

    # Build tree: Part 2 – Comparative degree analysis (critical sequential block)
    comp_node = evaluator.add_sequential(
        id="Comparative_Degree_Analysis",
        desc="Determine which state requires least additional degree-level education beyond master's",
        parent=root,
        critical=True
    )
    await verify_state_degree_requirements(evaluator, comp_node, states)
    await verify_minimum_degree_selection(evaluator, comp_node, selection, states)

    # Build tree: Part 3 – Additional requirements for selected state (NON-CRITICAL parallel block to satisfy framework constraints)
    await verify_selected_state_additional_requirements(evaluator, root, selection, addl)

    # Return evaluation summary
    return evaluator.get_summary()