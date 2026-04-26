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
TASK_ID = "school_counselor_cert_compare"
TASK_DESCRIPTION = """
You are considering a career change to become a school counselor and are evaluating opportunities in three states: Florida, New York, and Texas. For each state, research and document the complete certification requirements, including:

1. Degree Requirements: What is the minimum graduate degree requirement (including specific semester hours if applicable)?
2. Certification Examinations: What specific examination(s) must be passed, including exam names/numbers and passing scores if specified?
3. Program Requirements: Are there any required preparation programs (such as educator preparation programs or practicum experiences)?
4. Experience Requirements: Are there any required years of teaching or counseling experience? Document current policies.
5. Renewal Requirements: What are the continuing education or professional development requirements for certificate renewal, including the number of hours and renewal cycle period?
6. Official Documentation: Provide an official reference URL from each state's education department or authorized source that verifies these requirements.

For each state, ensure your information is current and sourced from official state education department websites or their authorized certification resources.
""".strip()


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class StateRequirement(BaseModel):
    degree_requirement: Optional[str] = None
    certification_exam: Optional[str] = None
    preparation_program: Optional[str] = None
    practicum_requirement: Optional[str] = None
    experience_requirement: Optional[str] = None
    renewal_requirements: Optional[str] = None
    official_sources: List[str] = Field(default_factory=list)


class CounselorCertificationExtraction(BaseModel):
    florida: Optional[StateRequirement] = None
    new_york: Optional[StateRequirement] = None
    texas: Optional[StateRequirement] = None


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_counselor_requirements() -> str:
    return """
Extract the school counselor certification requirements as presented in the answer for the following three states: Florida, New York, and Texas.

For each state, extract the following fields as strings (verbatim or faithful paraphrase from the answer). If any field is missing, set it to null. Also extract the official source URL(s) exactly as written (full URLs). Use arrays for URLs even if there is only one.

Per state fields:
- degree_requirement: Minimum graduate degree requirement (include any semester-hour specifics if provided; e.g., “a master’s degree with X semester hours”).
- certification_exam: Required certification exam(s), including exam names/numbers and passing scores if provided (e.g., “FTCE School Counseling PK–12” or “TExES 252, passing score 240”).
- preparation_program: Any required educator/approved preparation program or specific approved program requirement.
- practicum_requirement: Any supervised field experience/practicum requirement, with hours if mentioned.
- experience_requirement: Required years of teaching or counseling experience (if any). Include notes on “current policy” when stated.
- renewal_requirements: Continuing education/professional development for certificate renewal, including hours and cycle (e.g., “150 hours every 5 years”).
- official_sources: An array of official reference URLs (from state education departments or authorized certification resources).

State-specific guidance:
- Florida: Include degree and any specified semester hours if mentioned; include FTCE exam name and any passing score if provided.
- New York: If applicable, include both Initial and Professional certificate degree hour requirements (e.g., initial 48 hours, professional/Professional with 60 hours). Include required NYSED exams and CTLE renewal requirements.
- Texas: Include TExES exam number and passing score if mentioned; document EPP (Educator Preparation Program) requirement; mention current experience policy (recognize any noted changes such as around September 2023); include CPE renewal hours and cycle.

Return a JSON object with keys: florida, new_york, texas. Each key maps to an object with the above fields. For any missing information, set to null; for official_sources set to [] if none.
""".strip()


# --------------------------------------------------------------------------- #
# Verification helpers                                                        #
# --------------------------------------------------------------------------- #
def _safe_sources(info: Optional[StateRequirement]) -> List[str]:
    if info and info.official_sources:
        # Deduplicate while preserving order
        seen = set()
        unique = []
        for u in info.official_sources:
            if isinstance(u, str) and u.strip() and u not in seen:
                unique.append(u.strip())
                seen.add(u.strip())
        return unique
    return []


async def _verify_reference_url(
    evaluator: Evaluator,
    parent_node,
    node_id: str,
    state_name: str,
    sources: List[str],
    desc: str,
) -> Any:
    """
    Create and verify the 'Reference URL' leaf node for a given state.
    If no sources are provided, fail the node directly.
    """
    ref_node = evaluator.add_leaf(
        id=node_id,
        desc=desc,
        parent=parent_node,
        critical=True,
    )

    if not sources:
        # No official sources provided -> fail this critical node
        ref_node.score = 0.0
        ref_node.status = "failed"
        return ref_node

    claim = (
        f"This webpage is an official {state_name} state education department page (or an authorized certification resource) "
        f"that documents school counselor certification requirements."
    )

    # State-specific hints for official domains (just examples to aid the judge):
    domain_hints = {
        "Florida": "Examples of official/authorized domains include fldoe.org or other *.fl.us government education domains. Verify the page presents certification requirements for school counselors.",
        "New York": "Examples of official/authorized domains include nysed.gov and highered.nysed.gov/tcert. Verify the page presents certification requirements for school counselors.",
        "Texas": "Examples of official/authorized domains include tea.texas.gov. Verify the page presents certification requirements for school counselors."
    }
    add_ins = domain_hints.get(state_name, "Verify that this page is an official state education department or authorized certification resource and it documents counselor certification requirements.")

    await evaluator.verify(
        claim=claim,
        node=ref_node,
        sources=sources,
        additional_instruction=add_ins
    )
    return ref_node


async def _verify_requirement_leaf(
    evaluator: Evaluator,
    parent_node,
    node_id: str,
    desc: str,
    state_name: str,
    field_value: Optional[str],
    sources: List[str],
    extra_prereq_node,
    additional_instruction: str
) -> None:
    """
    Generic helper to create a critical leaf and verify its field value against provided sources,
    depending on the 'ReferenceURL' node as a prerequisite.
    """
    leaf = evaluator.add_leaf(
        id=node_id,
        desc=desc,
        parent=parent_node,
        critical=True,
    )

    # Build a clear, verifiable claim tying the state, topic, and value to official page(s)
    value_text = field_value if (field_value and field_value.strip()) else "(no specific details provided)"
    claim = f"The official {state_name} source supports the following information: {desc}. Stated details: '{value_text}'."

    await evaluator.verify(
        claim=claim,
        node=leaf,
        sources=sources if sources else None,
        additional_instruction=additional_instruction,
        extra_prerequisites=[extra_prereq_node] if extra_prereq_node else None
    )


# --------------------------------------------------------------------------- #
# Per-state verification builders                                             #
# --------------------------------------------------------------------------- #
async def verify_florida(evaluator: Evaluator, parent_node, info: Optional[StateRequirement]) -> None:
    state = "Florida"
    sources = _safe_sources(info)

    # Reference URL first
    ref_node = await _verify_reference_url(
        evaluator=evaluator,
        parent_node=parent_node,
        node_id="FL_ReferenceURL",
        state_name=state,
        sources=sources,
        desc="Provide official reference URL from Florida Department of Education or FLDOE authorized source"
    )

    # Degree Requirement
    await _verify_requirement_leaf(
        evaluator=evaluator,
        parent_node=parent_node,
        node_id="FL_DegreeRequirement",
        desc="Identify the minimum graduate degree requirement for Florida school counselor certification, including specific semester hours if applicable",
        state_name=state,
        field_value=info.degree_requirement if info else None,
        sources=sources,
        extra_prereq_node=ref_node,
        additional_instruction="Look for degree minimums on the official Florida source. Accept reasonable paraphrasing and minor variations; ensure any semester-hour counts match or are explicitly supported."
    )

    # Certification Exam
    await _verify_requirement_leaf(
        evaluator=evaluator,
        parent_node=parent_node,
        node_id="FL_CertificationExam",
        desc="Identify the required certification examination for Florida school counselors, including exam name and passing score if specified",
        state_name=state,
        field_value=info.certification_exam if info else None,
        sources=sources,
        extra_prereq_node=ref_node,
        additional_instruction="Verify the specific exam name/number (e.g., FTCE School Counseling PK–12) and passing score if present. Allow minor formatting differences."
    )

    # Preparation Program
    await _verify_requirement_leaf(
        evaluator=evaluator,
        parent_node=parent_node,
        node_id="FL_PreparationProgram",
        desc="Document any required educator preparation program or approved program requirements for Florida",
        state_name=state,
        field_value=info.preparation_program if info else None,
        sources=sources,
        extra_prereq_node=ref_node,
        additional_instruction="Confirm that an educator/approved preparation program requirement is stated for school counselor certification."
    )

    # Practicum Requirement
    await _verify_requirement_leaf(
        evaluator=evaluator,
        parent_node=parent_node,
        node_id="FL_PracticumRequirement",
        desc="Document supervised field experience or practicum requirements for Florida",
        state_name=state,
        field_value=info.practicum_requirement if info else None,
        sources=sources,
        extra_prereq_node=ref_node,
        additional_instruction="Verify supervised field experience/practicum language and any specified hours."
    )

    # Experience Requirement
    await _verify_requirement_leaf(
        evaluator=evaluator,
        parent_node=parent_node,
        node_id="FL_ExperienceRequirement",
        desc="Document any required years of teaching or counseling experience for Florida certification",
        state_name=state,
        field_value=info.experience_requirement if info else None,
        sources=sources,
        extra_prereq_node=ref_node,
        additional_instruction="Check for any explicit teaching or counseling experience requirements and capture the current policy."
    )

    # Renewal Requirements
    await _verify_requirement_leaf(
        evaluator=evaluator,
        parent_node=parent_node,
        node_id="FL_RenewalRequirements",
        desc="Document renewal requirements including the number of continuing education hours and renewal cycle period for Florida",
        state_name=state,
        field_value=info.renewal_requirements if info else None,
        sources=sources,
        extra_prereq_node=ref_node,
        additional_instruction="Verify continuing education/professional development hour totals and the renewal cycle period; allow rounding or minor phrasing differences."
    )


async def verify_new_york(evaluator: Evaluator, parent_node, info: Optional[StateRequirement]) -> None:
    state = "New York"
    sources = _safe_sources(info)

    # Reference URL first
    ref_node = await _verify_reference_url(
        evaluator=evaluator,
        parent_node=parent_node,
        node_id="NY_ReferenceURL",
        state_name=state,
        sources=sources,
        desc="Provide official reference URL from NYSED or New York State Education Department authorized source"
    )

    # Degree Requirement (Initial and Professional details encouraged)
    await _verify_requirement_leaf(
        evaluator=evaluator,
        parent_node=parent_node,
        node_id="NY_DegreeRequirement",
        desc="Identify the minimum graduate degree requirement for New York school counselor certification, including specific semester hours for both Initial (48 hours) and Professional (60 hours) certificates",
        state_name=state,
        field_value=info.degree_requirement if info else None,
        sources=sources,
        extra_prereq_node=ref_node,
        additional_instruction="For NYSED, verify if requirements include Initial certificate (often ~48 graduate hours) and Professional certificate (often ~60 hours). Accept reasonable paraphrase if both are clearly indicated."
    )

    # Certification Exam
    await _verify_requirement_leaf(
        evaluator=evaluator,
        parent_node=parent_node,
        node_id="NY_CertificationExam",
        desc="Identify the required certification examination for New York school counselors, including exam name/number and passing score if specified",
        state_name=state,
        field_value=info.certification_exam if info else None,
        sources=sources,
        extra_prereq_node=ref_node,
        additional_instruction="Verify the required NY State certification exams (names/numbers) for school counselors and any passing score language if present."
    )

    # Preparation Program
    await _verify_requirement_leaf(
        evaluator=evaluator,
        parent_node=parent_node,
        node_id="NY_PreparationProgram",
        desc="Document any required graduate program or preparation program requirements for New York",
        state_name=state,
        field_value=info.preparation_program if info else None,
        sources=sources,
        extra_prereq_node=ref_node,
        additional_instruction="Confirm a NYSED-approved program or equivalent preparation requirement for school counseling certification."
    )

    # Practicum Requirement
    await _verify_requirement_leaf(
        evaluator=evaluator,
        parent_node=parent_node,
        node_id="NY_PracticumRequirement",
        desc="Document supervised field experience or practicum requirements for New York",
        state_name=state,
        field_value=info.practicum_requirement if info else None,
        sources=sources,
        extra_prereq_node=ref_node,
        additional_instruction="Verify field experience/practicum requirements and any hours or scope described for NY school counseling programs."
    )

    # Experience Requirement (including mentored experience)
    await _verify_requirement_leaf(
        evaluator=evaluator,
        parent_node=parent_node,
        node_id="NY_ExperienceRequirement",
        desc="Document any required years of teaching or counseling experience for New York certification, including mentored experience requirements",
        state_name=state,
        field_value=info.experience_requirement if info else None,
        sources=sources,
        extra_prereq_node=ref_node,
        additional_instruction="Check for mentored experience or other experience requirements stated by NYSED for Initial/Professional certificates."
    )

    # Renewal Requirements (CTLE)
    await _verify_requirement_leaf(
        evaluator=evaluator,
        parent_node=parent_node,
        node_id="NY_RenewalRequirements",
        desc="Document CTLE renewal requirements for New York, including the number of hours and renewal cycle period",
        state_name=state,
        field_value=info.renewal_requirements if info else None,
        sources=sources,
        extra_prereq_node=ref_node,
        additional_instruction="Verify CTLE requirements (e.g., total hours and applicable cycle) for NY certificate maintenance."
    )


async def verify_texas(evaluator: Evaluator, parent_node, info: Optional[StateRequirement]) -> None:
    state = "Texas"
    sources = _safe_sources(info)

    # Reference URL first
    ref_node = await _verify_reference_url(
        evaluator=evaluator,
        parent_node=parent_node,
        node_id="TX_ReferenceURL",
        state_name=state,
        sources=sources,
        desc="Provide official reference URL from Texas Education Agency (TEA) or TEA authorized source"
    )

    # Degree Requirement
    await _verify_requirement_leaf(
        evaluator=evaluator,
        parent_node=parent_node,
        node_id="TX_DegreeRequirement",
        desc="Identify the minimum graduate degree hour requirement for Texas school counselor certification, including specific semester hours",
        state_name=state,
        field_value=info.degree_requirement if info else None,
        sources=sources,
        extra_prereq_node=ref_node,
        additional_instruction="Verify degree requirements on TEA or authorized pages; ensure any semester-hour counts are supported."
    )

    # Certification Exam (TExES # and passing score)
    await _verify_requirement_leaf(
        evaluator=evaluator,
        parent_node=parent_node,
        node_id="TX_CertificationExam",
        desc="Identify the required TExES examination number and passing score for Texas school counselors",
        state_name=state,
        field_value=info.certification_exam if info else None,
        sources=sources,
        extra_prereq_node=ref_node,
        additional_instruction="Verify the TExES exam name/number (e.g., School Counselor 252) and passing score (e.g., 240) if stated."
    )

    # EPP Requirement
    await _verify_requirement_leaf(
        evaluator=evaluator,
        parent_node=parent_node,
        node_id="TX_EPPRequirement",
        desc="Document the Educator Preparation Program (EPP) requirement for Texas",
        state_name=state,
        field_value=info.preparation_program if info else None,
        sources=sources,
        extra_prereq_node=ref_node,
        additional_instruction="Confirm an approved Educator Preparation Program (EPP) is required for Texas school counselor certification."
    )

    # Practicum Requirement
    await _verify_requirement_leaf(
        evaluator=evaluator,
        parent_node=parent_node,
        node_id="TX_PracticumRequirement",
        desc="Document supervised field experience or practicum requirements for Texas",
        state_name=state,
        field_value=info.practicum_requirement if info else None,
        sources=sources,
        extra_prereq_node=ref_node,
        additional_instruction="Verify any practicum/field experience requirements and hours as specified by TEA/authorized sources."
    )

    # Experience Requirement (current policy; mention Sep 2023 change if applicable)
    await _verify_requirement_leaf(
        evaluator=evaluator,
        parent_node=parent_node,
        node_id="TX_ExperienceRequirement",
        desc="Document the current policy regarding teaching or counseling experience requirements for Texas, including the September 2023 policy change",
        state_name=state,
        field_value=info.experience_requirement if info else None,
        sources=sources,
        extra_prereq_node=ref_node,
        additional_instruction="Verify the current TEA policy for experience requirements for school counselors; if a change around September 2023 is mentioned, ensure the stated policy aligns with current requirements."
    )

    # Renewal Requirements (CPE)
    await _verify_requirement_leaf(
        evaluator=evaluator,
        parent_node=parent_node,
        node_id="TX_RenewalRequirements",
        desc="Document CPE renewal hour requirements and renewal cycle for Texas school counselor certificate",
        state_name=state,
        field_value=info.renewal_requirements if info else None,
        sources=sources,
        extra_prereq_node=ref_node,
        additional_instruction="Verify required Continuing Professional Education (CPE) hours and the renewal cycle period for Texas certificates."
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
    Evaluate an answer for the School Counselor Certification Comparison task.
    """
    evaluator = Evaluator()

    # Root: Use PARALLEL aggregation. Set critical=False to allow partial credit across states
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

    # Extraction
    extracted = await evaluator.extract(
        prompt=prompt_extract_counselor_requirements(),
        template_class=CounselorCertificationExtraction,
        extraction_name="state_requirements",
    )

    # Build tree: top-level comparison node (parallel)
    comparison_node = evaluator.add_parallel(
        id="SchoolCounselorCertificationComparison",
        desc="Compare school counselor certification requirements across Florida, New York, and Texas",
        parent=root,
        critical=False  # Non-critical to avoid hard fail when one state fails
    )

    # Florida subtree
    fl_node = evaluator.add_parallel(
        id="Florida",
        desc="School counselor certification requirements for Florida",
        parent=comparison_node,
        critical=False
    )
    await verify_florida(
        evaluator=evaluator,
        parent_node=fl_node,
        info=extracted.florida
    )

    # New York subtree
    ny_node = evaluator.add_parallel(
        id="NewYork",
        desc="School counselor certification requirements for New York",
        parent=comparison_node,
        critical=False
    )
    await verify_new_york(
        evaluator=evaluator,
        parent_node=ny_node,
        info=extracted.new_york
    )

    # Texas subtree
    tx_node = evaluator.add_parallel(
        id="Texas",
        desc="School counselor certification requirements for Texas",
        parent=comparison_node,
        critical=False
    )
    await verify_texas(
        evaluator=evaluator,
        parent_node=tx_node,
        info=extracted.texas
    )

    # Return summary
    return evaluator.get_summary()