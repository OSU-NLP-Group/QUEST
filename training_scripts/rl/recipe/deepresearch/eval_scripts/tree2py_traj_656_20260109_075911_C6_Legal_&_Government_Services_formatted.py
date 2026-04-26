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
TASK_ID = "veto_override_3_5"
TASK_DESCRIPTION = """
Identify three U.S. states where the state legislature requires a three-fifths (3/5) supermajority vote in both legislative chambers to override a gubernatorial veto. For each of these three states, provide: (1) the state name, (2) the specific constitutional citation (article and section) establishing the veto override requirement, (3) the exact threshold formula as stated in the state constitution (e.g., '3/5 of elected members', '3/5 of present and voting members', etc.), (4) confirmation that the requirement applies to both the state House and Senate, and (5) reference URLs to official government sources that verify this constitutional provision. Ensure all information is drawn from authoritative state government or legislative sources.
"""


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class StateEntry(BaseModel):
    state_name: Optional[str] = None
    constitutional_citation: Optional[str] = None  # e.g., "Article IV, Section 9"
    threshold_formula: Optional[str] = None        # e.g., "three-fifths of the members elected to each house"
    both_chambers_confirmation: Optional[str] = None  # textual confirmation, e.g., "applies to each house"
    official_sources: List[str] = Field(default_factory=list)  # URLs claimed as official government/legislature


class StatesExtraction(BaseModel):
    states: List[StateEntry] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_states() -> str:
    return """
    Extract up to three U.S. states listed in the answer that claim to require a three-fifths (3/5) vote in BOTH legislative chambers (House and Senate) to override a gubernatorial veto.

    For each state, extract exactly these fields:
    1. state_name: The name of the U.S. state.
    2. constitutional_citation: The precise constitutional citation (article and section) where the veto-override requirement is stated (e.g., "Article IV, Section 9").
    3. threshold_formula: The exact wording of the threshold requirement as stated in the constitution (e.g., "three-fifths of the members elected to each house" or "three-fifths of the members present and voting").
    4. both_chambers_confirmation: A textual confirmation (quoted or paraphrased from the answer) that the requirement applies to BOTH chambers (House and Senate). Phrases like "each house" should be captured here.
    5. official_sources: An array of URLs that the answer cites as official state government or legislative sources verifying the provision. Extract the URLs exactly as provided in the answer. Include only URLs; ignore non-URL citations.

    IMPORTANT:
    - Extract strictly from the given answer; do not invent or infer additional states or citations.
    - If the answer provides more than three states, return only the first three mentioned.
    - If a field is missing for a state, set it to null or an empty array as appropriate.
    - For official_sources, keep only the URLs the answer explicitly gives (plain URLs or markdown links). If none are provided, return an empty array.
    - Do NOT transform the wording; keep the threshold_formula text as stated in the answer.
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def _clean_urls(urls: List[str]) -> List[str]:
    """Normalize URLs: strip whitespace and drop empty strings."""
    return [u.strip() for u in urls if isinstance(u, str) and u.strip()]


def _state_label(idx: int) -> str:
    return f"State {idx + 1}: required information is complete and correct"


# --------------------------------------------------------------------------- #
# Verification functions                                                      #
# --------------------------------------------------------------------------- #
async def verify_state(
    evaluator: Evaluator,
    parent_node,
    state: StateEntry,
    idx: int,
) -> None:
    """
    Build and verify the five required checks for a single state.
    Children checks are critical under this state node, and gating is handled by
    the framework's precondition logic (critical sibling failure causes subsequent
    verifications to be skipped).
    """
    # Create a parallel node for this state (non-critical to allow partial credit across states)
    state_node = evaluator.add_parallel(
        id=f"state_{idx + 1}",
        desc=_state_label(idx),
        parent=parent_node,
        critical=False,
    )

    # 1) State name existence (critical)
    state_name_exists = bool(state.state_name and state.state_name.strip())
    evaluator.add_custom_node(
        result=state_name_exists,
        id=f"state_{idx + 1}_state_name",
        desc="Provides the U.S. state name",
        parent=state_node,
        critical=True,
    )

    # Prepare sources (used by subsequent verifications)
    sources_list = _clean_urls(state.official_sources)

    # 5) Official sources (critical) — verify at least one official URL supports the provision and is authoritative
    # If there are no sources, fail immediately via a custom node to avoid empty multi-URL verification
    if not sources_list:
        evaluator.add_custom_node(
            result=False,
            id=f"state_{idx + 1}_official_sources",
            desc="Provides reference URL(s) to authoritative official state government or legislative sources that verify the cited constitutional provision",
            parent=state_node,
            critical=True,
        )
        # When official sources fail, subsequent critical checks will be auto-skipped by the framework
    else:
        official_sources_leaf = evaluator.add_leaf(
            id=f"state_{idx + 1}_official_sources",
            desc="Provides reference URL(s) to authoritative official state government or legislative sources that verify the cited constitutional provision",
            parent=state_node,
            critical=True,
        )
        urls_str = "; ".join(sources_list)
        claim_official = (
            f"At least one of the following URLs is an authoritative official {state.state_name or 'state'} "
            f"government or legislature page and it explicitly states the constitutional veto-override requirement "
            f"as a three-fifths threshold applying to both chambers: {urls_str}"
        )
        await evaluator.verify(
            claim=claim_official,
            node=official_sources_leaf,
            sources=sources_list,
            additional_instruction=(
                "Evaluate both officialness (e.g., .gov domains, legislature-hosted sites) and substantive support. "
                "Pass only if the page is an authoritative state government or legislative source and it clearly states "
                "the 3/5 veto-override requirement applying to both chambers (phrases like 'each house' should count)."
            ),
        )

    # 2) Constitutional citation (critical) — verify article/section location via sources
    citation_leaf = evaluator.add_leaf(
        id=f"state_{idx + 1}_constitutional_citation",
        desc="Provides the constitutional citation including article and section for the veto override requirement",
        parent=state_node,
        critical=True,
    )
    citation_text = state.constitutional_citation or ""
    state_name_text = state.state_name or "the state"
    claim_citation = (
        f"In {state_name_text}, the gubernatorial veto-override provision is located at {citation_text} in the state constitution."
    )
    # Route verification based on whether we have sources; if no sources, the precondition official_sources should fail
    await evaluator.verify(
        claim=claim_citation,
        node=citation_leaf,
        sources=sources_list if sources_list else None,
        additional_instruction=(
            "Check the constitutional text or official codification page and confirm that the veto-override provision "
            f"is indeed located at the cited article/section '{citation_text}'. Minor formatting variations are acceptable."
        ),
    )

    # 3) Threshold formula (critical) — verify exact formula wording via sources (must reflect three-fifths and the base)
    threshold_leaf = evaluator.add_leaf(
        id=f"state_{idx + 1}_threshold_formula",
        desc="Provides the exact threshold formula as written in the constitution, explicitly reflecting a three-fifths (3/5) requirement (including the base such as 'elected members' vs 'present and voting')",
        parent=state_node,
        critical=True,
    )
    formula_text = state.threshold_formula or ""
    claim_threshold = (
        f"The constitutional text for {state_name_text} states the veto-override threshold as: '{formula_text}', "
        "and it explicitly reflects a three-fifths (3/5) requirement along with the base (e.g., members elected or present and voting)."
    )
    await evaluator.verify(
        claim=claim_threshold,
        node=threshold_leaf,
        sources=sources_list if sources_list else None,
        additional_instruction=(
            "Confirm that the threshold wording matches the constitution's text and includes an explicit three-fifths requirement. "
            "Also confirm the base (e.g., 'members elected', 'members present', or 'present and voting'). Allow minor formatting differences."
        ),
    )

    # 4) Both chambers confirmation (critical) — verify requirement applies to both House and Senate
    both_leaf = evaluator.add_leaf(
        id=f"state_{idx + 1}_both_chambers_confirmation",
        desc="Confirms the 3/5 veto override requirement applies to both legislative chambers",
        parent=state_node,
        critical=True,
    )
    confirm_text = state.both_chambers_confirmation or ""
    claim_both = (
        f"In {state_name_text}, the constitutional veto-override requirement applies to both chambers (House and Senate). "
        f"Text such as '{confirm_text}' or 'each house' should be treated as confirmation that it applies to both."
    )
    await evaluator.verify(
        claim=claim_both,
        node=both_leaf,
        sources=sources_list if sources_list else None,
        additional_instruction=(
            "Verify that the constitution's text indicates the requirement applies to both chambers. "
            "Phrases like 'each house' should be treated as confirmation."
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
    Evaluate an answer for the veto-override (3/5 in both chambers) task.
    """
    # Initialize evaluator; NOTE: set root as non-critical to satisfy framework constraint
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Parallel aggregation across the three states
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

    # Extract states info from the answer
    extracted_states = await evaluator.extract(
        prompt=prompt_extract_states(),
        template_class=StatesExtraction,
        extraction_name="states_extraction",
    )

    # Prepare exactly three states (pad with empty entries or truncate to first three)
    states_list = list(extracted_states.states or [])
    if len(states_list) > 3:
        states_list = states_list[:3]
    while len(states_list) < 3:
        states_list.append(StateEntry())

    # Build verification tree for each of the three states
    for i in range(3):
        await verify_state(evaluator, root, states_list[i], i)

    # Return structured summary
    return evaluator.get_summary()