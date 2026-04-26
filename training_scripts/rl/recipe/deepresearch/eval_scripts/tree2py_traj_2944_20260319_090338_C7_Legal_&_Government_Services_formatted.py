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
TASK_ID = "voter_policies_2026_four_states"
TASK_DESCRIPTION = (
    "As of March 2026, identify exactly four US states that simultaneously meet ALL of the "
    "following criteria: (1) Do NOT require voters to present any form of identification at polling places "
    "for in-person voting, (2) Offer same-day voter registration (allowing eligible voters to register and vote "
    "on Election Day or during early voting), and (3) Have implemented online voter registration systems. "
    "For each of the four states, provide the state name and reference URLs documenting that the state meets "
    "all three requirements."
)

# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class StateEntry(BaseModel):
    state_name: Optional[str] = None
    # General sources cited for this state (any/all of the three requirements)
    urls: List[str] = Field(default_factory=list)
    # If the answer explicitly attributes sources per criterion, capture them as well
    urls_no_id: List[str] = Field(default_factory=list)
    urls_same_day: List[str] = Field(default_factory=list)
    urls_online: List[str] = Field(default_factory=list)


class StatesExtraction(BaseModel):
    states: List[StateEntry] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_states() -> str:
    return """
    Extract the list of US states identified in the answer that purportedly meet all three requirements:
    (1) No ID required at in-person polling places for voting, (2) Same-day voter registration,
    and (3) Online voter registration.

    Return a JSON object with a single field:
    - states: an array of objects, each with the fields:
        - state_name: the full name of the US state (e.g., "Minnesota"). Do not include territories or DC.
        - urls: an array of all reference URLs cited for this state in the answer (collect every URL associated with this state).
        - urls_no_id: an array of URLs the answer explicitly cites for the "no ID required" criterion (if any; otherwise empty).
        - urls_same_day: an array of URLs the answer explicitly cites for the "same-day registration" criterion (if any; otherwise empty).
        - urls_online: an array of URLs the answer explicitly cites for the "online registration" criterion (if any; otherwise empty).

    Rules:
    - Extract ONLY what appears in the answer; do not invent or infer states or URLs.
    - Accept URLs provided in plain text or markdown; output only the resolved URL strings.
    - Deduplicate URLs within each array while preserving the original order.
    - Include ALL states the answer names, even if more than 4; we'll consider only the first 4 later.
    - If some fields are missing, set them to null (for strings) or empty array (for lists).
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _ordinal_word(idx1: int) -> str:
    mapping = {1: "first", 2: "second", 3: "third", 4: "fourth"}
    return mapping.get(idx1, f"{idx1}th")


def _unique_preserve_order(urls: List[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for u in urls:
        if not u:
            continue
        if u not in seen:
            seen.add(u)
            out.append(u)
    return out


def _gather_all_sources(s: StateEntry) -> List[str]:
    all_urls = []
    all_urls.extend(s.urls or [])
    all_urls.extend(s.urls_no_id or [])
    all_urls.extend(s.urls_same_day or [])
    all_urls.extend(s.urls_online or [])
    return _unique_preserve_order(all_urls)


def _preferred_sources(s: StateEntry, which: str) -> List[str]:
    if which == "no_id" and s.urls_no_id:
        return _unique_preserve_order(s.urls_no_id)
    if which == "same_day" and s.urls_same_day:
        return _unique_preserve_order(s.urls_same_day)
    if which == "online" and s.urls_online:
        return _unique_preserve_order(s.urls_online)
    # Fallback to all sources if criterion-specific sources are missing
    return _gather_all_sources(s)


# --------------------------------------------------------------------------- #
# Verification helpers                                                        #
# --------------------------------------------------------------------------- #
NO_ID_ADDITIONAL_INSTRUCTION = (
    "Interpret 'no ID required' specifically for general in-person voting at polling places for most voters. "
    "Ignore ID checks that apply ONLY to federal HAVA first-time voters who registered by mail, or to special "
    "contexts such as absentee/by-mail voting or provisional ballot cure. The source should indicate that voters "
    "generally do not need to present identification to vote in person at polling places."
)

SAME_DAY_ADDITIONAL_INSTRUCTION = (
    "Verify that the state offers same-day voter registration (also known as Election Day Registration or EDR), "
    "either on Election Day and/or during in-person early voting. Accept synonymous phrases such as 'register and "
    "vote on the same day.' If it is only available in limited localities and not a statewide policy, treat as not supported."
)

ONLINE_REG_ADDITIONAL_INSTRUCTION = (
    "Verify that the state has implemented an official online voter registration (OVR) system or portal that allows "
    "eligible voters to submit registrations online."
)


async def verify_state(
    evaluator: Evaluator,
    parent_node,
    state_entry: StateEntry,
    state_index_zero_based: int,
) -> None:
    idx1 = state_index_zero_based + 1
    ord_word = _ordinal_word(idx1)
    state_name = state_entry.state_name or ""

    # Create the state-level node (critical to align with overall critical gating)
    state_node = evaluator.add_parallel(
        id=f"State_{idx1}",
        desc=f"{ord_word.capitalize()} identified state meets all three voting requirement criteria",
        parent=parent_node,
        critical=True,
    )

    # References presence (critical) - ensures we don't verify factual leaves without sources
    all_sources = _gather_all_sources(state_entry)
    evaluator.add_custom_node(
        result=len(all_sources) > 0,
        id=f"State_{idx1}_References",
        desc=f"Valid reference URLs are provided documenting that the {ord_word} state meets all requirements",
        parent=state_node,
        critical=True,
    )

    # Leaf: No ID requirement for in-person voting (critical)
    no_id_leaf = evaluator.add_leaf(
        id=f"State_{idx1}_No_ID_Requirement",
        desc=f"The {ord_word} identified state does not require voters to present identification at polling places for in-person voting",
        parent=state_node,
        critical=True,
    )
    no_id_claim = (
        f"As of March 2026, the state of {state_name} does not require voters to present identification at "
        f"polling places for in-person voting."
    )
    await evaluator.verify(
        claim=no_id_claim,
        node=no_id_leaf,
        sources=_preferred_sources(state_entry, "no_id"),
        additional_instruction=NO_ID_ADDITIONAL_INSTRUCTION,
    )

    # Leaf: Same-day registration (critical)
    sdr_leaf = evaluator.add_leaf(
        id=f"State_{idx1}_Same_Day_Registration",
        desc=f"The {ord_word} identified state offers same-day voter registration on Election Day or during early voting",
        parent=state_node,
        critical=True,
    )
    sdr_claim = (
        f"As of March 2026, the state of {state_name} offers same-day voter registration on Election Day or "
        f"during in-person early voting."
    )
    await evaluator.verify(
        claim=sdr_claim,
        node=sdr_leaf,
        sources=_preferred_sources(state_entry, "same_day"),
        additional_instruction=SAME_DAY_ADDITIONAL_INSTRUCTION,
    )

    # Leaf: Online voter registration (critical)
    ovr_leaf = evaluator.add_leaf(
        id=f"State_{idx1}_Online_Registration",
        desc=f"The {ord_word} identified state has implemented an online voter registration system",
        parent=state_node,
        critical=True,
    )
    ovr_claim = (
        f"As of March 2026, the state of {state_name} provides an official online voter registration system."
    )
    await evaluator.verify(
        claim=ovr_claim,
        node=ovr_leaf,
        sources=_preferred_sources(state_entry, "online"),
        additional_instruction=ONLINE_REG_ADDITIONAL_INSTRUCTION,
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
    Evaluate an answer for the 'four states voter policy' task and return a structured summary.
    """
    # Initialize evaluator (root is always non-critical in framework)
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

    # Extract structured list of states and their sources from the answer
    extraction = await evaluator.extract(
        prompt=prompt_extract_states(),
        template_class=StatesExtraction,
        extraction_name="states_extraction",
    )

    # Add a critical top-level node representing the rubric root
    main_node = evaluator.add_parallel(
        id="Root",
        desc="Complete and accurate identification of exactly four US states meeting all specified voting requirements",
        parent=root,
        critical=True,
    )

    # Check exactly four states were identified in the answer (critical)
    total_states_reported = len(extraction.states or [])
    evaluator.add_custom_node(
        result=(total_states_reported == 4),
        id="Exactly_Four_States",
        desc="Exactly four states are identified, no more and no fewer",
        parent=main_node,
        critical=True,
    )

    # Prepare up to the first 4 states (pad with empty entries if fewer)
    prepared_states: List[StateEntry] = list(extraction.states[:4]) if extraction.states else []
    while len(prepared_states) < 4:
        prepared_states.append(StateEntry())

    # Build and verify per-state subtrees (each state node is critical under the main critical node)
    for i in range(4):
        await verify_state(evaluator, main_node, prepared_states[i], i)

    # Record custom info about evaluation context
    evaluator.add_custom_info(
        info={"as_of": "March 2026", "reported_states_count": total_states_reported},
        info_type="context",
        info_name="evaluation_context",
    )

    # Return structured summary
    return evaluator.get_summary()