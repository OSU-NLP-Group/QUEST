import asyncio
import logging
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "principal_license_two_states"
TASK_DESCRIPTION = """
Identify two U.S. states that require exactly three years of full-time teaching experience as a prerequisite for obtaining principal licensure eligibility. Both states must also require a master's degree from an accredited institution and possession of a valid teaching license. Provide the official state government or state administrative code URLs that document these requirements for each identified state.
"""


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class StateRequirement(BaseModel):
    state: Optional[str] = None
    experience_requirement: Optional[str] = None  # as mentioned in the answer (free text)
    masters_requirement: Optional[str] = None     # as mentioned in the answer (free text)
    teaching_license_requirement: Optional[str] = None  # as mentioned in the answer (free text)
    acceptable_school_types: Optional[str] = None  # e.g., "public or accredited nonpublic"
    source_urls: List[str] = Field(default_factory=list)  # URLs explicitly cited in the answer


class TwoStatesExtraction(BaseModel):
    states: List[StateRequirement] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_two_states() -> str:
    return """
    From the provided answer, extract up to two U.S. states that the answer claims meet the specified principal licensure prerequisites.
    For each identified state, extract the following fields strictly as stated in the answer:
    - state: The U.S. state's name (e.g., "Minnesota", "New Jersey"). Do not extract territories or districts.
    - experience_requirement: The description of the teaching experience requirement as claimed (e.g., "exactly three years of full-time teaching").
    - masters_requirement: The description of the master's degree requirement as claimed (e.g., "master's degree from an accredited institution").
    - teaching_license_requirement: The description of the teaching license/certificate prerequisite as claimed (e.g., "must hold a valid teaching license").
    - acceptable_school_types: Any mention of acceptable school types for counting experience (e.g., "public or accredited nonpublic schools").
    - source_urls: All URLs the answer provides for this state that document these requirements. Only include URLs explicitly present in the answer text. If none are provided, return an empty list.
    
    Return a JSON object with a single field:
    - states: an array with at most two objects, each following the schema above, in the same order as mentioned in the answer.
    
    Important rules:
    - Do not invent information. If a field is not stated in the answer, set it to null (for strings) or an empty list (for URLs).
    - For URLs, extract only valid URLs present in the answer (including markdown links). Do not infer or guess.
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def _non_empty(s: Optional[str]) -> bool:
    return bool(s and isinstance(s, str) and s.strip() != "")


def _urls_present(urls: Optional[List[str]]) -> bool:
    return bool(urls and isinstance(urls, list) and len(urls) > 0)


def _state_label(state: Optional[str], fallback: str) -> str:
    return state.strip() if _non_empty(state) else fallback


# --------------------------------------------------------------------------- #
# Verification builders                                                       #
# --------------------------------------------------------------------------- #
async def build_identification_nodes(
    evaluator: Evaluator,
    root,
    states: List[StateRequirement]
) -> Dict[str, Any]:
    """
    Build identification nodes for first and second state.
    Returns a dict with references to certain prerequisite nodes for later dependencies.
    """
    prereq_nodes = {}

    # First state identification
    first_node = evaluator.add_parallel(
        id="First_State_Identified",
        desc="A valid U.S. state requiring exactly three years of full-time teaching experience for principal licensure is correctly identified",
        parent=root,
        critical=False
    )

    s0 = states[0] if len(states) > 0 else StateRequirement()
    first_state_name = _state_label(s0.state, "First State")
    s0_name_present = evaluator.add_custom_node(
        result=_non_empty(s0.state),
        id="state_0_name_present",
        desc="First state name is provided in the answer",
        parent=first_node,
        critical=True
    )
    prereq_nodes["state_0_name_present"] = s0_name_present

    # Second state identification
    second_node = evaluator.add_parallel(
        id="Second_State_Identified",
        desc="A second valid U.S. state (different from the first) requiring exactly three years of full-time teaching experience for principal licensure is correctly identified",
        parent=root,
        critical=False
    )

    s1 = states[1] if len(states) > 1 else StateRequirement()
    second_state_name = _state_label(s1.state, "Second State")
    s1_name_present = evaluator.add_custom_node(
        result=_non_empty(s1.state),
        id="state_1_name_present",
        desc="Second state name is provided in the answer",
        parent=second_node,
        critical=True
    )
    prereq_nodes["state_1_name_present"] = s1_name_present

    # Ensure the second state is different from the first
    diff_leaf = evaluator.add_leaf(
        id="state_1_different_from_first",
        desc="The second identified state is different from the first identified state",
        parent=second_node,
        critical=True
    )
    # Compose a simple verification claim
    s0n = s0.state or ""
    s1n = s1.state or ""
    await evaluator.verify(
        claim=f"The two names '{s0n}' and '{s1n}' refer to different U.S. states.",
        node=diff_leaf,
        additional_instruction="Treat names as the same if they only differ by casing, common suffixes like 'State of', or whitespace. Territories or districts do not count as states.",
        extra_prerequisites=[s1_name_present, s0_name_present] if "state_0_name_present" in prereq_nodes else [s1_name_present]
    )

    return prereq_nodes


async def build_masters_requirement_nodes(
    evaluator: Evaluator,
    root,
    states: List[StateRequirement]
) -> None:
    """
    Build and verify: both states require a master's degree from an accredited institution.
    """
    masters_node = evaluator.add_parallel(
        id="Masters_Degree_Requirement",
        desc="Documentation confirms that both identified states require a master's degree from an accredited institution for principal licensure",
        parent=root,
        critical=True
    )

    for i in range(2):
        s = states[i] if i < len(states) else StateRequirement()
        st_name = _state_label(s.state, f"State {i+1}")

        # Existence of sources to ground verification
        src_exist = evaluator.add_custom_node(
            result=_urls_present(s.source_urls),
            id=f"masters_state_{i}_sources_present",
            desc=f"{st_name}: At least one source URL is provided for verifying master's degree requirement",
            parent=masters_node,
            critical=True
        )

        # Verification leaf
        masters_leaf = evaluator.add_leaf(
            id=f"masters_state_{i}_supported",
            desc=f"{st_name}: Requires a master's degree from an accredited institution for principal licensure",
            parent=masters_node,
            critical=True
        )
        await evaluator.verify(
            claim=f"For {st_name}, a master's degree from an accredited (e.g., regionally accredited) institution is required to be eligible for principal licensure.",
            node=masters_leaf,
            sources=s.source_urls if _urls_present(s.source_urls) else None,
            additional_instruction=(
                "Pass only if the page explicitly requires a master's degree and indicates the institution must be accredited "
                "(e.g., 'regionally accredited', 'accredited institution'). Closely check wording under eligibility or requirements."
            )
        )


async def build_experience_requirement_nodes(
    evaluator: Evaluator,
    root,
    states: List[StateRequirement]
) -> None:
    """
    Build and verify: both states require exactly three years of full-time teaching experience.
    """
    exp_node = evaluator.add_parallel(
        id="Teaching_Experience_Three_Years",
        desc="Documentation confirms that both identified states require exactly three years of full-time teaching experience (not more, not less) for principal licensure eligibility",
        parent=root,
        critical=True
    )

    for i in range(2):
        s = states[i] if i < len(states) else StateRequirement()
        st_name = _state_label(s.state, f"State {i+1}")

        src_exist = evaluator.add_custom_node(
            result=_urls_present(s.source_urls),
            id=f"exp_state_{i}_sources_present",
            desc=f"{st_name}: At least one source URL is provided for verifying the three-years full-time experience requirement",
            parent=exp_node,
            critical=True
        )

        exp_leaf = evaluator.add_leaf(
            id=f"exp_state_{i}_three_years_full_time",
            desc=f"{st_name}: Requires exactly three years of full-time teaching experience for principal licensure eligibility",
            parent=exp_node,
            critical=True
        )
        await evaluator.verify(
            claim=(
                f"For {st_name}, principal licensure eligibility requires exactly three years of full-time teaching experience."
            ),
            node=exp_leaf,
            sources=s.source_urls if _urls_present(s.source_urls) else None,
            additional_instruction=(
                "Confirm the requirement is exactly three (3) years and specifies full-time (or full-time equivalent/FTE). "
                "Do NOT pass if the language is 'at least 3 years', 'minimum 3 years', or gives a range. "
                "Accept synonyms like 'full-time equivalent (FTE)'."
            )
        )


async def build_license_prereq_nodes(
    evaluator: Evaluator,
    root,
    states: List[StateRequirement]
) -> None:
    """
    Build and verify: both states require holding a valid/active teaching license as a prerequisite.
    """
    license_node = evaluator.add_parallel(
        id="Teaching_License_Prerequisite",
        desc="Documentation confirms that both identified states require holding a valid/active teaching license before applying for principal licensure",
        parent=root,
        critical=True
    )

    for i in range(2):
        s = states[i] if i < len(states) else StateRequirement()
        st_name = _state_label(s.state, f"State {i+1}")

        src_exist = evaluator.add_custom_node(
            result=_urls_present(s.source_urls),
            id=f"license_state_{i}_sources_present",
            desc=f"{st_name}: At least one source URL is provided for verifying the valid teaching license prerequisite",
            parent=license_node,
            critical=True
        )

        lic_leaf = evaluator.add_leaf(
            id=f"license_state_{i}_valid_license_required",
            desc=f"{st_name}: Applicants must hold a valid/active teaching license to qualify for principal licensure",
            parent=license_node,
            critical=True
        )
        await evaluator.verify(
            claim=(
                f"For {st_name}, applicants must currently hold a valid or active teaching license/certificate to be eligible for principal licensure."
            ),
            node=lic_leaf,
            sources=s.source_urls if _urls_present(s.source_urls) else None,
            additional_instruction=(
                "Look for explicit language such as 'hold a valid teaching license/certificate', 'current professional educator license', "
                "or equivalent phrasing in eligibility/prerequisites sections."
            )
        )


async def build_school_types_nodes(
    evaluator: Evaluator,
    root,
    states: List[StateRequirement]
) -> None:
    """
    Build and verify: both states accept experience from public or accredited nonpublic schools.
    """
    school_node = evaluator.add_parallel(
        id="Acceptable_School_Types",
        desc="Documentation confirms that both identified states accept teaching experience from public schools or accredited nonpublic schools",
        parent=root,
        critical=True
    )

    for i in range(2):
        s = states[i] if i < len(states) else StateRequirement()
        st_name = _state_label(s.state, f"State {i+1}")

        src_exist = evaluator.add_custom_node(
            result=_urls_present(s.source_urls),
            id=f"school_types_state_{i}_sources_present",
            desc=f"{st_name}: At least one source URL is provided for verifying acceptable school types",
            parent=school_node,
            critical=True
        )

        school_leaf = evaluator.add_leaf(
            id=f"school_types_state_{i}_public_or_accredited_nonpublic",
            desc=f"{st_name}: Accepts teaching experience from public schools OR accredited nonpublic/private schools",
            parent=school_node,
            critical=True
        )
        await evaluator.verify(
            claim=(
                f"For {st_name}, qualifying teaching experience for principal licensure can be obtained in public schools or in accredited nonpublic/private schools."
            ),
            node=school_leaf,
            sources=s.source_urls if _urls_present(s.source_urls) else None,
            additional_instruction=(
                "Pass only if the page explicitly allows experience from public schools and also allows experience from accredited nonpublic/approved private schools. "
                "Accept synonyms like 'accredited private', 'approved nonpublic', 'recognized nonpublic'."
            )
        )


async def build_official_sources_nodes(
    evaluator: Evaluator,
    root,
    states: List[StateRequirement]
) -> None:
    """
    Build and verify: official sources (state education department or state administrative code) OR accredited university program page referencing official state requirements are provided for both states.
    """
    official_node = evaluator.add_parallel(
        id="Official_Source_URLs",
        desc="Official state education department, state administrative code, or accredited university program URLs referencing official state requirements are provided for both identified states",
        parent=root,
        critical=True
    )

    for i in range(2):
        s = states[i] if i < len(states) else StateRequirement()
        st_name = _state_label(s.state, f"State {i+1}")

        src_exist = evaluator.add_custom_node(
            result=_urls_present(s.source_urls),
            id=f"official_state_{i}_sources_present",
            desc=f"{st_name}: At least one source URL is provided",
            parent=official_node,
            critical=True
        )

        official_leaf = evaluator.add_leaf(
            id=f"official_state_{i}_has_official_reference",
            desc=f"{st_name}: At least one provided URL is an official state page or state administrative code page, OR an accredited university program page that references official state requirements",
            parent=official_node,
            critical=True
        )
        await evaluator.verify(
            claim=(
                f"At least one of these pages is either: (a) an official {st_name} state government or state administrative code page that documents principal licensure requirements; "
                f"or (b) an accredited university (.edu) program page that explicitly references or cites the official {st_name} state principal licensure requirements."
            ),
            node=official_leaf,
            sources=s.source_urls if _urls_present(s.source_urls) else None,
            additional_instruction=(
                "Prefer .gov or state-admin (e.g., state.xx.us) domains and pages labeled as administrative code, state department of education, etc. "
                "If a .edu page is used, it must clearly and directly reference or cite the official state requirements."
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
    Evaluate an answer for the principal licensure requirements (two states) task.
    """
    # Initialize evaluator
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Parallel: independent checks with critical gates
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

    # Extraction
    extracted = await evaluator.extract(
        prompt=prompt_extract_two_states(),
        template_class=TwoStatesExtraction,
        extraction_name="extracted_states_requirements"
    )

    # Keep only the first two states; pad if fewer than 2
    states: List[StateRequirement] = list(extracted.states[:2])
    while len(states) < 2:
        states.append(StateRequirement())

    # Record a summary of extracted info
    evaluator.add_custom_info(
        info={
            "state_1": states[0].dict(),
            "state_2": states[1].dict()
        },
        info_type="extraction_summary",
        info_name="parsed_states_overview"
    )

    # Build tree: identification nodes
    prereqs = await build_identification_nodes(evaluator, root, states)

    # Build tree: critical requirement nodes
    await build_masters_requirement_nodes(evaluator, root, states)
    await build_experience_requirement_nodes(evaluator, root, states)
    await build_license_prereq_nodes(evaluator, root, states)
    await build_school_types_nodes(evaluator, root, states)
    await build_official_sources_nodes(evaluator, root, states)

    # Return evaluation summary
    return evaluator.get_summary()