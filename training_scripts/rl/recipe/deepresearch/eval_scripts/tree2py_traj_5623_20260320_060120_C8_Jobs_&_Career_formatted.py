import asyncio
import logging
from typing import Any, List, Optional, Dict

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "athletic_director_states"
TASK_DESCRIPTION = """
In the United States, high school athletic director positions vary significantly in their requirements and compensation across different states. Identify 4 U.S. states where high school athletic director positions meet ALL of the following criteria:

1. The average annual salary for high school athletic directors is at least $100,000
2. The National Interscholastic Athletic Administrators Association (NIAAA) Certified Athletic Administrator (CAA) certification is recognized or accepted for athletic director positions
3. A valid state teaching certificate is required for high school athletic director positions
4. A minimum of 2 years of coaching or athletic administration experience is typically required for athletic director positions

For each of the 4 states you identify, provide:
- The state name
- Documentation of the average annual salary meeting the $100,000 threshold
- Evidence that NIAAA CAA certification is recognized or accepted
- Verification that a state teaching certificate is required
- Confirmation of the minimum experience requirement

Each piece of information must be supported by a reference URL from your research.
"""


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class StateCriteria(BaseModel):
    state_name: Optional[str] = None

    # Salary
    salary_note: Optional[str] = None  # e.g., "Average salary is $105,000"
    salary_urls: List[str] = Field(default_factory=list)

    # NIAAA CAA recognition
    niaaa_note: Optional[str] = None   # e.g., "CAA certification is required/preferred/accepted"
    niaaa_urls: List[str] = Field(default_factory=list)

    # Teaching certificate requirement
    teaching_cert_note: Optional[str] = None  # e.g., "Valid teaching certificate required"
    teaching_cert_urls: List[str] = Field(default_factory=list)

    # Experience requirement
    experience_note: Optional[str] = None  # e.g., "Minimum two (2) years of coaching experience"
    experience_years: Optional[str] = None  # e.g., "2", "two", "2+"
    experience_urls: List[str] = Field(default_factory=list)


class StatesExtraction(BaseModel):
    states: List[StateCriteria] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_states() -> str:
    return """
    Extract up to 4 U.S. states that the answer claims meet ALL four criteria for high school athletic director positions (salary ≥ $100,000; NIAAA CAA recognized/accepted; valid state teaching certificate required; ≥ 2 years experience required).
    
    For each state mentioned in the answer, extract the following fields:
    - state_name: The state name as written in the answer.
    - salary_note: The salary figure or phrase claimed in the answer (string, do not normalize).
    - salary_urls: All URLs cited in the answer that support the salary claim for this state.
    - niaaa_note: The statement or phrase indicating NIAAA CAA recognition/acceptance.
    - niaaa_urls: All URLs cited that support the NIAAA CAA recognition/acceptance claim.
    - teaching_cert_note: The statement indicating a valid state teaching certificate is required.
    - teaching_cert_urls: All URLs cited that support the teaching certificate requirement claim.
    - experience_note: The statement indicating the minimum years of experience.
    - experience_years: The number of years mentioned (as a string; keep as written, e.g., "2", "two", "2+").
    - experience_urls: All URLs cited that support the experience requirement claim.

    IMPORTANT:
    - Only extract URLs explicitly present in the answer text. If no URL is given for a field, return an empty list for that field.
    - If a field is not present in the answer for a state, set it to null (or empty list for URLs).
    - Return up to the first 4 states if more are provided. Preserve the order in which they appear in the answer.
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _filter_valid_urls(urls: Optional[List[str]]) -> List[str]:
    if not urls:
        return []
    cleaned = []
    for u in urls:
        if not isinstance(u, str):
            continue
        s = u.strip()
        if not s:
            continue
        # Allow any non-empty string; verifier will attempt retrieval/validation.
        # Prefer full URLs; if missing protocol, keep it as-is to remain faithful to answer.
        cleaned.append(s)
    # Deduplicate while preserving order
    seen = set()
    deduped = []
    for u in cleaned:
        if u not in seen:
            seen.add(u)
            deduped.append(u)
    return deduped


def _has_at_least_one_url(urls: Optional[List[str]]) -> bool:
    return len(_filter_valid_urls(urls)) > 0


# --------------------------------------------------------------------------- #
# Verification for one state                                                  #
# --------------------------------------------------------------------------- #
async def verify_one_state(
    evaluator: Evaluator,
    parent_node,
    state: StateCriteria,
    idx: int,
) -> None:
    """
    Build verification subtree for a single state with four critical criteria groups:
    - Salary ≥ $100,000
    - NIAAA CAA recognized/accepted
    - Valid state teaching certificate required
    - ≥ 2 years experience required
    """
    display_idx = idx + 1
    st_name = state.state_name or f"State #{display_idx}"

    state_node = evaluator.add_parallel(
        id=f"state_{display_idx}",
        desc=f"{['First','Second','Third','Fourth'][idx]} qualifying state meeting all criteria",
        parent=parent_node,
        critical=False
    )

    # Optionally record a compact snapshot for debugging
    evaluator.add_custom_info(
        info={
            "state_name": state.state_name,
            "salary_note": state.salary_note,
            "salary_urls": state.salary_urls,
            "niaaa_note": state.niaaa_note,
            "niaaa_urls": state.niaaa_urls,
            "teaching_cert_note": state.teaching_cert_note,
            "teaching_cert_urls": state.teaching_cert_urls,
            "experience_note": state.experience_note,
            "experience_years": state.experience_years,
            "experience_urls": state.experience_urls,
        },
        info_type="extracted_state_info",
        info_name=f"state_{display_idx}_extracted"
    )

    # -------------------- Salary ≥ $100,000 -------------------- #
    salary_group = evaluator.add_sequential(
        id=f"state_{display_idx}_salary_group",
        desc="Salary verification sequence",
        parent=state_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=_has_at_least_one_url(state.salary_urls),
        id=f"state_{display_idx}_salary_sources_provided",
        desc="Salary criterion includes at least one reference URL",
        parent=salary_group,
        critical=True
    )
    salary_leaf = evaluator.add_leaf(
        id=f"state_{display_idx}_salary",
        desc="Average annual salary for athletic directors is at least $100,000, supported by reference URL",
        parent=salary_group,
        critical=True
    )
    salary_claim = (
        f"In {st_name}, the average annual salary for high school athletic directors (K-12) "
        f"is at least $100,000."
    )
    await evaluator.verify(
        claim=salary_claim,
        node=salary_leaf,
        sources=_filter_valid_urls(state.salary_urls),
        additional_instruction=(
            "Use the provided source(s) only. Confirm that the page supports a statewide figure "
            "(average/mean/median/typical) of $100,000 or higher for HIGH SCHOOL (K-12) athletic directors. "
            "If the number clearly applies to colleges/universities or unrelated roles, it should not pass. "
            "If a value ≥ $100,000 is clearly shown for high school athletic directors, mark as supported."
        ),
    )

    # -------------------- NIAAA CAA recognized/accepted -------------------- #
    niaaa_group = evaluator.add_sequential(
        id=f"state_{display_idx}_niaaa_group",
        desc="NIAAA CAA recognition verification sequence",
        parent=state_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=_has_at_least_one_url(state.niaaa_urls),
        id=f"state_{display_idx}_niaaa_sources_provided",
        desc="NIAAA criterion includes at least one reference URL",
        parent=niaaa_group,
        critical=True
    )
    niaaa_leaf = evaluator.add_leaf(
        id=f"state_{display_idx}_niaaa",
        desc="NIAAA CAA certification is recognized or accepted for athletic director positions, supported by reference URL",
        parent=niaaa_group,
        critical=True
    )
    niaaa_claim = (
        f"In {st_name}, for high school athletic director (K-12) positions, the NIAAA Certified Athletic "
        f"Administrator (CAA) credential is recognized or accepted (e.g., required, preferred, or explicitly "
        f"listed as an accepted/recognized credential)."
    )
    await evaluator.verify(
        claim=niaaa_claim,
        node=niaaa_leaf,
        sources=_filter_valid_urls(state.niaaa_urls),
        additional_instruction=(
            "Verify that the provided page(s) explicitly connect the NIAAA CAA certification to K-12/high school "
            "athletic director roles (e.g., in job postings, district requirements, or state guidelines). "
            "Mentions of CAA unrelated to AD hiring should not pass."
        ),
    )

    # -------------------- Teaching certificate required -------------------- #
    teach_group = evaluator.add_sequential(
        id=f"state_{display_idx}_teaching_cert_group",
        desc="Teaching certificate requirement verification sequence",
        parent=state_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=_has_at_least_one_url(state.teaching_cert_urls),
        id=f"state_{display_idx}_teaching_cert_sources_provided",
        desc="Teaching certificate criterion includes at least one reference URL",
        parent=teach_group,
        critical=True
    )
    teach_leaf = evaluator.add_leaf(
        id=f"state_{display_idx}_teaching_cert",
        desc="State teaching certificate is required for athletic director positions, supported by reference URL",
        parent=teach_group,
        critical=True
    )
    teach_claim = (
        f"In {st_name}, a valid state teaching certificate (teaching license) is required for high school "
        f"athletic director positions."
    )
    await evaluator.verify(
        claim=teach_claim,
        node=teach_leaf,
        sources=_filter_valid_urls(state.teaching_cert_urls),
        additional_instruction=(
            "The evidence must tie the requirement to K-12/high school athletic director roles. "
            "Synonyms like 'teaching license' count. If the page clearly states the role must hold "
            "a valid state teaching certificate/license, mark as supported."
        ),
    )

    # -------------------- ≥ 2 years experience required -------------------- #
    exp_group = evaluator.add_sequential(
        id=f"state_{display_idx}_experience_group",
        desc="Experience requirement verification sequence",
        parent=state_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=_has_at_least_one_url(state.experience_urls),
        id=f"state_{display_idx}_experience_sources_provided",
        desc="Experience criterion includes at least one reference URL",
        parent=exp_group,
        critical=True
    )
    exp_leaf = evaluator.add_leaf(
        id=f"state_{display_idx}_experience",
        desc="Minimum 2 years of coaching or athletic administration experience is typically required, supported by reference URL",
        parent=exp_group,
        critical=True
    )
    exp_claim = (
        f"In {st_name}, high school athletic director positions typically require a minimum of 2 years of "
        f"coaching or athletic administration experience."
    )
    await evaluator.verify(
        claim=exp_claim,
        node=exp_leaf,
        sources=_filter_valid_urls(state.experience_urls),
        additional_instruction=(
            "Confirm that the page shows a requirement of at least two (2) years (e.g., 'minimum two years', '2+ years') "
            "of relevant experience for K-12/high school athletic director roles. If it says three years, it's still "
            "acceptable since it is ≥ 2. If less than two years, do not support."
        ),
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
    Entry point for evaluating the answer for the athletic director states task.
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

    # Extract structured info from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_states(),
        template_class=StatesExtraction,
        extraction_name="states_extraction"
    )

    # Ensure we have exactly 4 state entries (truncate or pad)
    states: List[StateCriteria] = list(extracted.states[:4])
    while len(states) < 4:
        states.append(StateCriteria())

    # Build verification subtrees per state
    # We will run them sequentially in code, but each state's internal checks run under the evaluator logic.
    for i in range(4):
        await verify_one_state(evaluator, root, states[i], i)

    # Return the unified evaluation summary
    return evaluator.get_summary()