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
TASK_ID = "admin_cert_4_states"
TASK_DESCRIPTION = """
An experienced school administrator is considering career advancement opportunities and wants to understand their options across different U.S. states. The administrator has the following credentials: Master's degree in Educational Leadership from an accredited university, 8 years of successful classroom teaching experience (grades 3-5), 3 years of experience as an Assistant Principal at an elementary school, and currently holds valid teaching and administrative certification in Virginia. The administrator is interested in pursuing principal or assistant superintendent positions in public school districts and wants to understand the certification pathways in different states. Identify 4 different U.S. states (excluding Virginia, since the candidate is already certified there) where this candidate would qualify for principal or assistant superintendent certification. For each state, provide: (1) State and Certification Type: The state name and the specific name of the administrative certification required; (2) Educational Requirements: The required degree level and field, and verification that the candidate's master's degree in educational leadership meets the requirement; (3) Experience Requirements: Minimum required years of teaching experience, any required administrative experience, and verification that the candidate's 8 years teaching + 3 years as assistant principal meets the requirement; (4) Certification Pathway Details: Required certification exams, required preparation programs or additional coursework, and other requirements such as background checks or fees. All information must be verified through official state education department websites or recognized educational certification resources. Each requirement must include a supporting URL reference.
"""

CANDIDATE_PROFILE = (
    "Candidate profile: Master's degree in Educational Leadership (accredited), "
    "8 years classroom teaching experience (grades 3–5), "
    "3 years Assistant Principal at an elementary school, "
    "holds valid teaching and administrative certification in Virginia."
)

# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class StateIdentification(BaseModel):
    state_name: Optional[str] = None
    certification_type: Optional[str] = None
    identification_urls: List[str] = Field(default_factory=list)


class StateEducation(BaseModel):
    degree_level: Optional[str] = None
    degree_field: Optional[str] = None
    education_urls: List[str] = Field(default_factory=list)


class StateExperience(BaseModel):
    teaching_experience_required: Optional[str] = None
    admin_experience_required: Optional[str] = None
    experience_urls: List[str] = Field(default_factory=list)


class StatePathway(BaseModel):
    exams: List[str] = Field(default_factory=list)
    program_or_coursework: Optional[str] = None
    other_requirements: Optional[str] = None
    pathway_urls: List[str] = Field(default_factory=list)


class StateEntry(BaseModel):
    identification: StateIdentification = Field(default_factory=StateIdentification)
    education: StateEducation = Field(default_factory=StateEducation)
    experience: StateExperience = Field(default_factory=StateExperience)
    pathway: StatePathway = Field(default_factory=StatePathway)


class StatesExtraction(BaseModel):
    states: List[StateEntry] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_states() -> str:
    return """
    Extract up to four U.S. states (excluding Virginia) discussed in the answer where the candidate could qualify for principal or assistant superintendent certification. For each state, extract the following structured fields. Only extract information explicitly present in the answer, and only extract URLs that are explicitly included in the answer text (plain URLs or markdown links).

    For each state, extract an object with:
    - identification:
        - state_name: the U.S. state name (must not be Virginia)
        - certification_type: the specific administrative certification name for principal or assistant superintendent (e.g., "Standard School Principal", "Initial Administrator License – Principal", "Assistant Superintendent", etc.)
        - identification_urls: URL(s) cited for general certification overview or identification of the credential in that state
    - education:
        - degree_level: the required degree level (e.g., "master's or higher", "doctoral", "bachelor's + approved program", etc.)
        - degree_field: the acceptable or required field(s) (e.g., "educational leadership/administration", "education", etc.). Use the wording provided in the answer.
        - education_urls: URL(s) that support the education requirement
    - experience:
        - teaching_experience_required: the minimum years of teaching experience (as stated, e.g., "3 years teaching", "2 years", etc.)
        - admin_experience_required: any explicitly required administrative experience (e.g., "2 years building-level leadership", "none", "not specified", etc.)
        - experience_urls: URL(s) that support the experience requirement
    - pathway:
        - exams: list of named exam(s) if stated (e.g., "Praxis 5412", "SLLA", "SSLLA", Pearson exams). If the answer explicitly states none are required, include ["none"]. If exams are not mentioned in the answer, leave empty.
        - program_or_coursework: the required approved preparation program or coursework description as stated in the answer (e.g., "state-approved principal preparation program", "X credits in administration", etc.). If not mentioned, set to null.
        - other_requirements: other requirements as stated (e.g., "background check", "fees", "fingerprinting", etc.). If not mentioned, set to null.
        - pathway_urls: URL(s) that support exam/program/other requirement details

    Return a JSON with a top-level "states" array of these objects in the same order as the answer presents them. Exclude Virginia if it appears. If more than 4 states are present, include only the first 4 unique non-Virginia states. If fewer than 4 appear, return however many are present.
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _valid_url_list(urls: Optional[List[str]]) -> List[str]:
    if not urls:
        return []
    # keep clearly valid URLs
    cleaned = []
    for u in urls:
        if isinstance(u, str):
            u = u.strip()
            if u and (u.startswith("http://") or u.startswith("https://")):
                cleaned.append(u)
    return cleaned


def _is_nonempty_text(s: Optional[str]) -> bool:
    return bool(s and isinstance(s, str) and s.strip())


def _ordinal(n: int) -> str:
    return ["First", "Second", "Third", "Fourth", "Fifth"][n - 1] if 1 <= n <= 5 else f"#{n}"


def _urls_required_instruction(has_urls: bool) -> str:
    if has_urls:
        return (
            "Judge strictly based on the provided source page(s). "
            "Do not rely on the answer text alone. Mark Incorrect if the page(s) do not explicitly support the claim."
        )
    else:
        return (
            "No URL evidence was provided for this claim. Per instructions, if no URL evidence is available, "
            "you must judge the claim as Incorrect (unsupported)."
        )


def _role_hint_from_cert(cert_type: Optional[str]) -> str:
    if not _is_nonempty_text(cert_type):
        return "administrator (principal or assistant superintendent)"
    return cert_type.strip()


# --------------------------------------------------------------------------- #
# Verification for one state                                                  #
# --------------------------------------------------------------------------- #
async def verify_state(
    evaluator: Evaluator,
    parent_node,
    state: StateEntry,
    state_index: int,
) -> None:
    """
    Build the verification subtree for a single state, following the rubric structure.
    """
    ord_name = _ordinal(state_index)
    state_name = state.identification.state_name or ""
    cert_type = _role_hint_from_cert(state.identification.certification_type)

    # Top-level node for this state (non-critical; allows partial credit across states)
    state_node = evaluator.add_parallel(
        id=f"state_{state_index}",
        desc=f"{ord_name} state's certification pathway fully documented and verified",
        parent=parent_node,
        critical=False
    )

    # ---------------- Identification ----------------
    ident_node = evaluator.add_parallel(
        id=f"state_{state_index}_identification",
        desc="State name and certification type are clearly identified",
        parent=state_node,
        critical=True
    )

    # name provided and not Virginia
    name_ok = _is_nonempty_text(state_name) and state_name.strip().lower() != "virginia"
    evaluator.add_custom_node(
        result=name_ok,
        id=f"state_{state_index}_name",
        desc="U.S. state name is provided",
        parent=ident_node,
        critical=True
    )

    # certification type provided
    cert_ok = _is_nonempty_text(state.identification.certification_type)
    evaluator.add_custom_node(
        result=cert_ok,
        id=f"state_{state_index}_cert_type",
        desc="Specific certification name for principal or assistant superintendent role is provided",
        parent=ident_node,
        critical=True
    )

    # identification URL provided (existence)
    ident_urls = _valid_url_list(state.identification.identification_urls)
    ident_url_ok = len(ident_urls) > 0
    evaluator.add_custom_node(
        result=ident_url_ok,
        id=f"state_{state_index}_id_url",
        desc="URL reference to state certification information",
        parent=ident_node,
        critical=True
    )

    # ---------------- Education Requirements ----------------
    edu_node = evaluator.add_parallel(
        id=f"state_{state_index}_education_req",
        desc="Educational degree requirements are documented",
        parent=state_node,
        critical=True
    )

    edu_urls = _valid_url_list(state.education.education_urls)
    edu_urls_ok = len(edu_urls) > 0
    evaluator.add_custom_node(
        result=edu_urls_ok,
        id=f"state_{state_index}_education_url",
        desc="URL reference supporting education requirements",
        parent=edu_node,
        critical=True
    )

    # Degree level specified and supported
    degree_level_claim = (
        f"The required degree level for {_role_hint_from_cert(state.identification.certification_type)} "
        f"in {state_name or 'the state'} is stated as: {state.education.degree_level if _is_nonempty_text(state.education.degree_level) else 'UNKNOWN/NOT PROVIDED'}."
    )
    degree_level_leaf = evaluator.add_leaf(
        id=f"state_{state_index}_degree_type",
        desc="Required degree level (master's, doctoral, etc.) is specified",
        parent=edu_node,
        critical=True
    )
    await evaluator.verify(
        claim=degree_level_claim,
        node=degree_level_leaf,
        sources=edu_urls,
        additional_instruction=(
            _urls_required_instruction(edu_urls_ok)
            + " If the degree text shows 'UNKNOWN/NOT PROVIDED', judge Incorrect. "
              "Consider equivalent phrasings (e.g., 'master's or higher', 'advanced degree') as matching the claimed level."
        )
    )

    # Degree field specified and supported
    degree_field_claim = (
        f"The acceptable or required degree field for {_role_hint_from_cert(state.identification.certification_type)} "
        f"in {state_name or 'the state'} includes: "
        f"{state.education.degree_field if _is_nonempty_text(state.education.degree_field) else 'UNKNOWN/NOT PROVIDED'}."
    )
    degree_field_leaf = evaluator.add_leaf(
        id=f"state_{state_index}_degree_field",
        desc="Required or acceptable degree field(s) are specified",
        parent=edu_node,
        critical=True
    )
    await evaluator.verify(
        claim=degree_field_claim,
        node=degree_field_leaf,
        sources=edu_urls,
        additional_instruction=(
            _urls_required_instruction(edu_urls_ok)
            + " If the field text shows 'UNKNOWN/NOT PROVIDED', judge Incorrect. "
              "Treat 'educational leadership/administration' and similar variants as equivalent."
        )
    )

    # Candidate meets education requirement
    meets_edu_claim = (
        f"Given the candidate's master's degree in Educational Leadership, the candidate meets the education "
        f"requirement for {_role_hint_from_cert(state.identification.certification_type)} in {state_name or 'the state'} "
        f"as described on the provided page(s)."
    )
    meets_edu_leaf = evaluator.add_leaf(
        id=f"state_{state_index}_candidate_meets_education",
        desc="Verification that candidate's master's degree in educational leadership meets the requirement",
        parent=edu_node,
        critical=True
    )
    await evaluator.verify(
        claim=meets_edu_claim,
        node=meets_edu_leaf,
        sources=edu_urls,
        additional_instruction=(
            _urls_required_instruction(edu_urls_ok)
            + " " + CANDIDATE_PROFILE + " Base your judgment solely on the page(s). "
              "Focus only on degree level/field equivalence (do not consider exams/programs in this check). "
              "If the page requires a master's in educational leadership/administration or accepts a master's in education/related fields, this should be supported."
        )
    )

    # ---------------- Experience Requirements ----------------
    exp_node = evaluator.add_parallel(
        id=f"state_{state_index}_experience_req",
        desc="Teaching and administrative experience requirements are documented",
        parent=state_node,
        critical=True
    )

    exp_urls = _valid_url_list(state.experience.experience_urls)
    exp_urls_ok = len(exp_urls) > 0
    evaluator.add_custom_node(
        result=exp_urls_ok,
        id=f"state_{state_index}_experience_url",
        desc="URL reference supporting experience requirements",
        parent=exp_node,
        critical=True
    )

    # Teaching years requirement
    teach_years_text = state.experience.teaching_experience_required if _is_nonempty_text(state.experience.teaching_experience_required) else "UNKNOWN/NOT PROVIDED"
    teach_claim = (
        f"The minimum required years of teaching experience for {_role_hint_from_cert(state.identification.certification_type)} "
        f"in {state_name or 'the state'} is: {teach_years_text}."
    )
    teach_leaf = evaluator.add_leaf(
        id=f"state_{state_index}_teaching_years",
        desc="Minimum required years of teaching experience is specified",
        parent=exp_node,
        critical=True
    )
    await evaluator.verify(
        claim=teach_claim,
        node=teach_leaf,
        sources=exp_urls,
        additional_instruction=(
            _urls_required_instruction(exp_urls_ok)
            + " If the value shows 'UNKNOWN/NOT PROVIDED', judge Incorrect. "
              "Allow reasonable phrasing variants (e.g., 'X years successful teaching')."
        )
    )

    # Administrative years requirement
    admin_years_text = state.experience.admin_experience_required if _is_nonempty_text(state.experience.admin_experience_required) else "UNKNOWN/NOT PROVIDED"
    admin_claim = (
        f"The required administrative experience (if any) for {_role_hint_from_cert(state.identification.certification_type)} "
        f"in {state_name or 'the state'} is: {admin_years_text}."
    )
    admin_leaf = evaluator.add_leaf(
        id=f"state_{state_index}_admin_years",
        desc="Any required administrative experience is specified (may be 0)",
        parent=exp_node,
        critical=True
    )
    await evaluator.verify(
        claim=admin_claim,
        node=admin_leaf,
        sources=exp_urls,
        additional_instruction=(
            _urls_required_instruction(exp_urls_ok)
            + " If the value shows 'UNKNOWN/NOT PROVIDED', judge Incorrect. "
              "If the page explicitly states no admin experience required, treat 'none' as supported."
        )
    )

    # Candidate meets experience requirement
    meets_exp_claim = (
        "Given 8 years of classroom teaching and 3 years as an Assistant Principal, the candidate meets or exceeds the "
        f"experience requirements stated for {_role_hint_from_cert(state.identification.certification_type)} "
        f"in {state_name or 'the state'} on the provided page(s)."
    )
    meets_exp_leaf = evaluator.add_leaf(
        id=f"state_{state_index}_candidate_meets_experience",
        desc="Verification that candidate's 8 years teaching + 3 years as assistant principal meets the requirement",
        parent=exp_node,
        critical=True
    )
    await evaluator.verify(
        claim=meets_exp_claim,
        node=meets_exp_leaf,
        sources=exp_urls,
        additional_instruction=(
            _urls_required_instruction(exp_urls_ok)
            + " " + CANDIDATE_PROFILE + " Base your judgment solely on the page(s). "
              "Allow reasonable interpretation (e.g., 'successful teaching' or 'three years' etc.). "
              "If the page lists multiple pathways, judge based on the pathway that matches the claimed certification route."
        )
    )

    # ---------------- Pathway Details ----------------
    path_node = evaluator.add_parallel(
        id=f"state_{state_index}_pathway_details",
        desc="Certification pathway steps and requirements are documented",
        parent=state_node,
        critical=True
    )

    path_urls = _valid_url_list(state.pathway.pathway_urls)
    path_urls_ok = len(path_urls) > 0
    evaluator.add_custom_node(
        result=path_urls_ok,
        id=f"state_{state_index}_pathway_url",
        desc="URL reference supporting pathway details",
        parent=path_node,
        critical=True
    )

    # Exam requirement
    exams_text = ", ".join([e for e in state.pathway.exams if _is_nonempty_text(e)]) if state.pathway.exams else "none"
    exam_claim = (
        f"For {_role_hint_from_cert(state.identification.certification_type)} in {state_name or 'the state'}, "
        f"the required certification exam(s) include: {exams_text}. "
        f"This list does not need to be exhaustive; it must include at least one required exam if exams are required."
    )
    exam_leaf = evaluator.add_leaf(
        id=f"state_{state_index}_exam_requirement",
        desc="Any required certification exams are identified",
        parent=path_node,
        critical=True
    )
    await evaluator.verify(
        claim=exam_claim,
        node=exam_leaf,
        sources=path_urls,
        additional_instruction=(
            _urls_required_instruction(path_urls_ok)
            + " If the answer claims 'none', only mark Correct if the page(s) explicitly indicate no exam is required. "
              "If the page lists a required exam and the answer's list is non-empty but incomplete, still mark Correct."
        )
    )

    # Program/coursework requirement
    program_text = state.pathway.program_or_coursework if _is_nonempty_text(state.pathway.program_or_coursework) else "UNKNOWN/NOT PROVIDED"
    program_claim = (
        f"The certification pathway for {_role_hint_from_cert(state.identification.certification_type)} in {state_name or 'the state'} "
        f"includes this required preparation program or coursework: {program_text}."
    )
    program_leaf = evaluator.add_leaf(
        id=f"state_{state_index}_program_requirement",
        desc="Any required preparation program or coursework is identified",
        parent=path_node,
        critical=True
    )
    await evaluator.verify(
        claim=program_claim,
        node=program_leaf,
        sources=path_urls,
        additional_instruction=(
            _urls_required_instruction(path_urls_ok)
            + " If the value shows 'UNKNOWN/NOT PROVIDED', judge Incorrect. "
              "Accept reasonable variants (e.g., 'state-approved administrator preparation program')."
        )
    )

    # Other requirements
    other_text = state.pathway.other_requirements if _is_nonempty_text(state.pathway.other_requirements) else "UNKNOWN/NOT PROVIDED"
    other_claim = (
        f"Additional requirements (e.g., background checks, fees, fingerprinting) for "
        f"{_role_hint_from_cert(state.identification.certification_type)} in {state_name or 'the state'} include: {other_text}. "
        f"This list may not be exhaustive but must be explicitly supported by the page(s)."
    )
    other_leaf = evaluator.add_leaf(
        id=f"state_{state_index}_other_requirements",
        desc="Any additional requirements (fingerprinting, fees, etc.) are identified",
        parent=path_node,
        critical=True
    )
    await evaluator.verify(
        claim=other_claim,
        node=other_leaf,
        sources=path_urls,
        additional_instruction=(
            _urls_required_instruction(path_urls_ok)
            + " If the value shows 'UNKNOWN/NOT PROVIDED', judge Incorrect. "
              "If the page mentions such requirements (e.g., background check or fees), mark Correct."
        )
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
    Evaluate an answer for the '4 different states administrator certification' task.
    """
    # Initialize evaluator (root is non-critical to allow partial credit across states)
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

    # Extract structured info
    extracted = await evaluator.extract(
        prompt=prompt_extract_states(),
        template_class=StatesExtraction,
        extraction_name="states_extraction",
    )

    # Select first 4 unique non-Virginia states, keep order
    unique_selected: List[StateEntry] = []
    seen_states = set()
    for st in extracted.states:
        name = (st.identification.state_name or "").strip()
        if not name:
            continue
        if name.lower() == "virginia":
            continue
        key = name.lower()
        if key in seen_states:
            continue
        unique_selected.append(st)
        seen_states.add(key)
        if len(unique_selected) == 4:
            break

    # Pad to 4 with empty placeholders if needed
    while len(unique_selected) < 4:
        unique_selected.append(StateEntry())

    evaluator.add_custom_info(
        info={"selected_states": [s.identification.state_name for s in unique_selected]},
        info_type="selection",
        info_name="selected_states_for_evaluation"
    )

    # Global critical check: all 4 are distinct and none is Virginia
    names = [s.identification.state_name for s in unique_selected]
    nonempty = all(_is_nonempty_text(n) for n in names)
    lowered = [n.strip().lower() for n in names] if nonempty else []
    uniqueness_ok = nonempty and (len(set(lowered)) == 4) and ("virginia" not in lowered)

    evaluator.add_custom_node(
        result=uniqueness_ok,
        id="state_uniqueness",
        desc="All 4 identified states are different from each other and none is Virginia",
        parent=root,
        critical=True
    )

    # Build 4 state subtrees
    for idx in range(1, 5):
        await verify_state(
            evaluator=evaluator,
            parent_node=root,
            state=unique_selected[idx - 1],
            state_index=idx
        )

    # Return structured summary
    return evaluator.get_summary()