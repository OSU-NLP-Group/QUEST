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
TASK_ID = "three_fifths_capital_most_populous"
TASK_DESCRIPTION = (
    "Among U.S. states that require a three-fifths (3/5) vote by the state legislature to override a governor's veto, "
    "identify the state capital of the most populous state."
)
AS_OF_YEAR = 2025


# --------------------------------------------------------------------------- #
# Data models for structured extraction                                       #
# --------------------------------------------------------------------------- #
class ThreeFifthsStateItem(BaseModel):
    state: Optional[str] = None
    # URLs specifically cited to support the 3/5 override threshold for this state (ideally NCSL)
    threshold_sources: List[str] = Field(default_factory=list)
    # URLs cited (if any) to support population data for this state
    population_sources: List[str] = Field(default_factory=list)


class MostPopulousInfo(BaseModel):
    state: Optional[str] = None
    population_value: Optional[str] = None
    population_sources: List[str] = Field(default_factory=list)


class CapitalInfo(BaseModel):
    city: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class CoreExtraction(BaseModel):
    three_fifths_states: List[ThreeFifthsStateItem] = Field(default_factory=list)
    # Any general NCSL URLs the answer cites (not necessarily tied to one state item)
    ncsl_general_urls: List[str] = Field(default_factory=list)
    most_populous: Optional[MostPopulousInfo] = None
    capital: Optional[CapitalInfo] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_core() -> str:
    return """
    Extract structured information from the answer relevant to this task.

    1) three_fifths_states (array):
       - Include every U.S. state the answer explicitly identifies as requiring a three-fifths (3/5) vote to override a governor's veto.
       - For each state object, extract:
         • state: Full state name as written.
         • threshold_sources: All URLs the answer cites to support the 3/5 override requirement for this specific state (prefer the NCSL comprehensive table if present).
         • population_sources: Any URLs the answer cites that provide population figures or rankings for this state (if any; otherwise empty).

    2) ncsl_general_urls (array):
       - Extract any NCSL URLs (domain contains 'ncsl.org') that the answer cites as a general reference for the veto override thresholds.

    3) most_populous (object):
       - state: The single state the answer claims is the most populous among those requiring a 3/5 veto-override threshold.
       - population_value: The population number or textual value provided in the answer (if any).
       - population_sources: All URLs the answer cites to support this 'most populous' determination (Census Bureau, reputable rankings, etc.).

    4) capital (object):
       - city: The capital city the answer provides for the identified most populous state.
       - sources: All URLs the answer cites to support that this city is the official capital (official state site, encyclopedia, etc.).

    Rules:
    - Only extract what is explicitly present in the answer.
    - For any missing field, return null for scalars and [] for arrays.
    - Preserve exact strings from the answer; do not infer or add information.
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _unique_nonempty(urls: List[str]) -> List[str]:
    seen = set()
    out = []
    for u in urls:
        if not u:
            continue
        if u not in seen:
            seen.add(u)
            out.append(u)
    return out


def _flatten(list_of_lists: List[List[str]]) -> List[str]:
    out: List[str] = []
    for lst in list_of_lists:
        out.extend(lst or [])
    return out


def _has_ncsl_url(urls: List[str]) -> bool:
    for u in urls:
        if isinstance(u, str) and "ncsl.org" in u.lower():
            return True
    return False


def _normalize_name(s: Optional[str]) -> str:
    return (s or "").strip().lower()


# --------------------------------------------------------------------------- #
# Verification subroutines                                                    #
# --------------------------------------------------------------------------- #
async def verify_three_fifths_states(
    evaluator: Evaluator,
    parent_node,
    extracted: CoreExtraction,
) -> List[str]:
    """
    Step 1. Verify the identification of states that require a 3/5 veto-override threshold.
    Returns the list of identified state names for use by later steps.
    """
    step_node = evaluator.add_parallel(
        id="identify_three_fifths_states",
        desc="Identifies U.S. states that require a three-fifths (3/5) vote threshold to override a governor's veto according to the NCSL comprehensive table",
        parent=parent_node,
        critical=False,
    )

    # Existence of at least one state listed
    states_listed = len(extracted.three_fifths_states) > 0
    evaluator.add_custom_node(
        result=states_listed,
        id="three_fifths_states_listed",
        desc="At least one U.S. state is listed as requiring a 3/5 veto-override threshold",
        parent=step_node,
        critical=True
    )

    # Combined sources for threshold support
    combined_threshold_sources = _unique_nonempty(
        _flatten([s.threshold_sources for s in extracted.three_fifths_states]) + extracted.ncsl_general_urls
    )

    # Require that an NCSL link is present somewhere as the comprehensive table source
    evaluator.add_custom_node(
        result=_has_ncsl_url(combined_threshold_sources),
        id="ncsl_source_present",
        desc="The NCSL comprehensive table is cited among the sources for the 3/5 threshold identification",
        parent=step_node,
        critical=True
    )

    # Per-state verification leaves (each must be supported by cited sources; prefer NCSL)
    state_names = []
    for idx, item in enumerate(extracted.three_fifths_states):
        st_name = (item.state or "").strip()
        if not st_name:
            # Still add a failing custom node to maintain a clear trace if an empty entry appears
            evaluator.add_custom_node(
                result=False,
                id=f"state_{idx}_name_present",
                desc=f"State #{idx + 1} name is present",
                parent=step_node,
                critical=True
            )
            continue

        state_names.append(st_name)

        # Ensure we have at least one URL for this state's threshold check; fallback to general NCSL URLs
        urls = item.threshold_sources if item.threshold_sources else extracted.ncsl_general_urls

        # Require sources present for this state
        evaluator.add_custom_node(
            result=len(urls) > 0,
            id=f"state_{idx}_threshold_sources_present",
            desc=f"Sources are provided for the 3/5 threshold claim for {st_name}",
            parent=step_node,
            critical=True
        )

        leaf = evaluator.add_leaf(
            id=f"state_{idx}_three_fifths_supported",
            desc=f"{st_name} requires a three-fifths (3/5) vote by the state legislature to override the governor's veto",
            parent=step_node,
            critical=True
        )
        claim = f"{st_name} requires a three-fifths (3/5) vote by the state legislature to override the governor's veto."
        await evaluator.verify(
            claim=claim,
            node=leaf,
            sources=urls,
            additional_instruction="Use the cited NCSL comprehensive table when available. Confirm that the override threshold is exactly 3/5 (60%), not a different fraction."
        )

    # Optional aggregate check mirroring the rubric leaf wording
    if state_names:
        agg_leaf = evaluator.add_leaf(
            id="states_have_three_fifths_threshold",
            desc="All identified states are documented in the NCSL table as requiring a 3/5 vote threshold to override a governor's veto",
            parent=step_node,
            critical=True
        )
        joined_states = ", ".join(state_names)
        agg_claim = f"Each of the following states requires a three-fifths (3/5) legislative vote to override the governor's veto: {joined_states}."
        await evaluator.verify(
            claim=agg_claim,
            node=agg_leaf,
            sources=combined_threshold_sources,
            additional_instruction="Validate this list against the NCSL comprehensive table. If any listed state does not match 3/5 exactly, mark as not supported."
        )

    return state_names


async def verify_most_populous(
    evaluator: Evaluator,
    parent_node,
    extracted: CoreExtraction,
    identified_states: List[str],
) -> None:
    """
    Step 2. Verify that the answer correctly determines the most populous state among the identified 3/5-threshold states.
    """
    step_node = evaluator.add_parallel(
        id="determine_most_populous",
        desc=f"Determines which state among the identified 3/5-threshold states has the highest population as of {AS_OF_YEAR}",
        parent=parent_node,
        critical=False
    )

    mp = extracted.most_populous or MostPopulousInfo()
    mp_state = (mp.state or "").strip()

    # Existence checks
    evaluator.add_custom_node(
        result=bool(mp_state),
        id="most_populous_provided",
        desc="A most populous state among the 3/5-threshold set is provided",
        parent=step_node,
        critical=True
    )

    # Ensure the identified state is within the previously identified set
    in_set = _normalize_name(mp_state) in {_normalize_name(s) for s in identified_states}
    evaluator.add_custom_node(
        result=in_set if identified_states else False,
        id="most_populous_within_identified_set",
        desc="The identified most populous state is among the previously identified 3/5-threshold states",
        parent=step_node,
        critical=True
    )

    # Require population sources for the identified state
    evaluator.add_custom_node(
        result=bool(mp.population_sources),
        id="most_populous_sources_present",
        desc="Population source(s) provided for the identified most populous state",
        parent=step_node,
        critical=True
    )

    # Build a single aggregate verification that the identified state exceeds all others
    others = [s for s in identified_states if _normalize_name(s) != _normalize_name(mp_state)]
    all_compare_urls = _unique_nonempty(mp.population_sources + _flatten([
        item.population_sources for item in extracted.three_fifths_states
        if _normalize_name(item.state) in {_normalize_name(x) for x in others}
    ]))

    leaf = evaluator.add_leaf(
        id="most_populous_correctly_identified",
        desc="The identified state has the highest population among the identified 3/5 states, supported by cited sources",
        parent=step_node,
        critical=True
    )

    if others:
        comp_list = ", ".join(others)
        claim = (
            f"As of {AS_OF_YEAR} (or the latest available official estimates), the population of {mp_state} is higher "
            f"than the populations of the following states: {comp_list}."
        )
    else:
        # Degenerate case: only one state identified; then it is trivially the most populous in that set.
        claim = (
            f"As of {AS_OF_YEAR} (or the latest available official estimates), {mp_state} is the most populous among the "
            f"identified set of three-fifths-threshold states."
        )

    await evaluator.verify(
        claim=claim,
        node=leaf,
        sources=all_compare_urls if all_compare_urls else mp.population_sources,
        additional_instruction="Use the cited population pages (e.g., U.S. Census Bureau or reputable ranking sites). If comparison states lack direct URLs, rely on provided sources that clearly show numeric counts or rankings sufficient to establish the comparison."
    )


async def verify_capital(
    evaluator: Evaluator,
    parent_node,
    extracted: CoreExtraction,
) -> None:
    """
    Step 3. Verify the provided capital for the most populous state.
    """
    step_node = evaluator.add_parallel(
        id="provide_state_capital",
        desc="Provides the official capital city of the state identified as most populous",
        parent=parent_node,
        critical=False
    )

    mp_state = (extracted.most_populous.state if extracted.most_populous else "") or ""
    cap_city = (extracted.capital.city if extracted.capital else "") or ""
    cap_sources = (extracted.capital.sources if extracted.capital else []) or []

    # Existence checks
    evaluator.add_custom_node(
        result=bool(cap_city),
        id="capital_provided",
        desc="A capital city for the identified most populous state is provided",
        parent=step_node,
        critical=True
    )

    evaluator.add_custom_node(
        result=bool(cap_sources),
        id="capital_sources_present",
        desc="Source(s) provided to support the stated capital city",
        parent=step_node,
        critical=True
    )

    # Capital verification leaf
    leaf = evaluator.add_leaf(
        id="capital_matches_identified_state",
        desc="The provided capital city is the official capital of the identified state",
        parent=step_node,
        critical=True
    )

    claim = f"The official capital of {mp_state} is {cap_city}."
    await evaluator.verify(
        claim=claim,
        node=leaf,
        sources=cap_sources,
        additional_instruction="Check authoritative sources (official state website, reputable encyclopedia). Allow minor formatting differences in city naming."
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
    Run the evaluation for: Among U.S. states requiring a 3/5 veto-override vote, identify the state capital of the most populous state.
    """
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.SEQUENTIAL,  # Follow the logical order: identify set -> pick most populous -> give capital
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
        prompt=prompt_extract_core(),
        template_class=CoreExtraction,
        extraction_name="core_extraction",
    )

    # Step 1: Identify and verify 3/5-threshold states
    identified_states = await verify_three_fifths_states(evaluator, root, extracted)

    # Step 2: Determine whether the chosen state is indeed the most populous in that set
    await verify_most_populous(evaluator, root, extracted, identified_states)

    # Step 3: Verify the capital of the identified most populous state
    await verify_capital(evaluator, root, extracted)

    return evaluator.get_summary()