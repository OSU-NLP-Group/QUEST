import asyncio
import logging
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys


# -----------------------------------------------------------------------------
# Task constants
# -----------------------------------------------------------------------------
TASK_ID = "simple_majority_veto_override_states"
TASK_DESCRIPTION = (
    "Identify 4 US states that currently have bicameral legislatures and require only a simple majority vote "
    "of all elected members (not members present) in both legislative chambers to override a gubernatorial veto. "
    "For each state, provide: (1) the state name, (2) confirmation that it has a bicameral legislature structure, "
    "(3) verification that only a simple majority (not a supermajority like two-thirds or three-fifths) is required "
    "in both chambers for veto override, (4) confirmation that the vote threshold is calculated based on all elected "
    "members of each chamber rather than just those present, and (5) a reference source (such as the state constitution, "
    "official state legislative website, or authoritative government document) that verifies the veto override procedure."
)


# -----------------------------------------------------------------------------
# Extraction models
# -----------------------------------------------------------------------------
class StateEntry(BaseModel):
    """
    One state's information as stated in the answer.
    All fields should be extracted directly from the answer text; sources should be the URLs the answer cites.
    """
    state_name: Optional[str] = None
    bicameral_statement: Optional[str] = None          # The answer's own wording confirming bicameral structure
    override_threshold_statement: Optional[str] = None  # The answer's statement that only a simple majority is required in both chambers
    vote_counting_basis_statement: Optional[str] = None # The answer's statement that vote is based on all elected members (not just present)
    sources: List[str] = Field(default_factory=list)    # URLs cited by the answer for this state


class StatesExtraction(BaseModel):
    """
    The list of states mentioned in the answer, in the order presented.
    """
    states: List[StateEntry] = Field(default_factory=list)


# -----------------------------------------------------------------------------
# Extraction prompt
# -----------------------------------------------------------------------------
def prompt_extract_states() -> str:
    return """
    Extract up to the first four (4) US states from the answer that the answer claims meet the following:
    - Bicameral legislature (two chambers)
    - A gubernatorial veto can be overridden by only a simple majority in both chambers (not a supermajority)
    - The threshold is calculated based on all elected members of each chamber (not just members present)
    For each of the first four states mentioned in the answer, extract:
    1) state_name: The state's full name (e.g., "Nebraska", "Illinois"); return null if not provided.
    2) bicameral_statement: Exactly how the answer text confirms bicameral structure (a short quote or paraphrase from the answer); null if not provided.
    3) override_threshold_statement: Exactly how the answer asserts that only a simple majority is required in both chambers (short quote or paraphrase); null if not provided.
    4) vote_counting_basis_statement: Exactly how the answer asserts that the threshold is based on all elected members, not just those present (short quote or paraphrase); null if not provided.
    5) sources: A list of all URLs (only actual URLs) the answer cites that are intended to support the veto‑override procedure and/or bicameral structure for that specific state.
       - Include only URLs explicitly present in the answer text for that state.
       - Accept plain URLs or markdown links; extract the actual URL.
       - If no URLs are provided for the state, return an empty list [].

    Return a JSON object:
    {
      "states": [
        {
          "state_name": ...,
          "bicameral_statement": ...,
          "override_threshold_statement": ...,
          "vote_counting_basis_statement": ...,
          "sources": [...]
        },
        ...
      ]
    }

    If the answer mentions fewer than 4 states, include only those present.
    Do not infer or invent URLs or statements that are not explicitly present in the answer.
    """


# -----------------------------------------------------------------------------
# Verification helpers
# -----------------------------------------------------------------------------
def _has_minimal_state_info(entry: StateEntry) -> bool:
    """We minimally require a state_name and at least one source URL to run web-grounded checks."""
    return bool((entry.state_name or "").strip()) and bool(entry.sources)


def _fail_leaf_due_to_missing_sources(node) -> None:
    """Mark a verification leaf as failed when sources are missing (avoid ungrounded verification)."""
    node.score = 0.0
    node.status = "failed"


async def verify_one_state(
    evaluator: Evaluator,
    parent_node,
    entry: StateEntry,
    idx: int
) -> None:
    """
    Build the verification subtree for a single state and run the checks.

    Leaves (all critical under the state node):
    - state_{i}_bicameral
    - state_{i}_majority_override
    - state_{i}_vote_counting
    - state_{i}_reference
    """
    state_label = f"state_{idx+1}"
    stated_name = (entry.state_name or "").strip() or f"State #{idx+1}"

    # Parent node for this state (non-critical, parallel as per rubric)
    state_node = evaluator.add_parallel(
        id=state_label,
        desc=f"{stated_name}: meets all specified criteria",
        parent=parent_node,
        critical=False
    )

    # Prepare leaves
    bicameral_node = evaluator.add_leaf(
        id=f"{state_label}_bicameral",
        desc="The state has a bicameral legislature (two legislative chambers: Senate and House/Assembly)",
        parent=state_node,
        critical=True
    )
    majority_node = evaluator.add_leaf(
        id=f"{state_label}_majority_override",
        desc="The state requires only a simple majority vote (not a supermajority like two-thirds or three-fifths) in both chambers to override a gubernatorial veto",
        parent=state_node,
        critical=True
    )
    counting_node = evaluator.add_leaf(
        id=f"{state_label}_vote_counting",
        desc="The veto override vote threshold is calculated based on all elected members of each chamber, not just members present",
        parent=state_node,
        critical=True
    )
    reference_node = evaluator.add_leaf(
        id=f"{state_label}_reference",
        desc="Provides a verifiable reference source (state constitution, official legislative website, or authoritative government document) confirming the veto override procedure",
        parent=state_node,
        critical=True
    )

    # If we don't have minimal info (state_name and sources), fail all four checks for this state
    if not _has_minimal_state_info(entry):
        for n in (bicameral_node, majority_node, counting_node, reference_node):
            _fail_leaf_due_to_missing_sources(n)
        return

    # Build verification tasks
    tasks: List[tuple[str, List[str], Any, Optional[str]]] = []

    # Bicameral verification
    bicameral_claim = (
        f"The legislature of {entry.state_name} is bicameral, i.e., it clearly has two chambers "
        f"(such as a Senate and a House/Assembly)."
    )
    bicameral_instruction = (
        "Verify that the provided page(s) explicitly or very clearly indicate the state legislature has two chambers "
        "(e.g., 'Senate and House of Representatives', 'Senate and Assembly', or similar). "
        "Text such as 'both houses', 'each house', or pages that list the two chambers count as support. "
        "Reject if the page is irrelevant, does not mention two chambers, or implies a unicameral structure."
    )
    tasks.append((bicameral_claim, entry.sources, bicameral_node, bicameral_instruction))

    # Majority override verification
    majority_claim = (
        f"In {entry.state_name}, overriding a gubernatorial veto requires only a simple majority (>50%) vote in "
        f"both legislative chambers (not any supermajority such as two‑thirds, three‑fifths, etc.)."
    )
    majority_instruction = (
        "Look for constitutional or official descriptions of the veto override threshold. "
        "To PASS, the page must indicate only a simple majority (>50%) is required in both chambers (e.g., 'each house', 'both houses'). "
        "If the page indicates a supermajority (e.g., two‑thirds, three‑fifths) or leaves the threshold ambiguous, FAIL. "
        "Do not accept interpretations; the text must suggest/confirm simple majority for an override."
    )
    tasks.append((majority_claim, entry.sources, majority_node, majority_instruction))

    # Vote counting basis verification
    counting_claim = (
        f"In {entry.state_name}, the majority threshold for overriding a gubernatorial veto is calculated from "
        f"all elected members of each chamber (e.g., 'a majority of all the members elected to each house'), "
        f"and not just those present and voting."
    )
    counting_instruction = (
        "To PASS, the page must indicate that the majority is computed relative to all elected (or 'all members elected') "
        "to each chamber (e.g., 'a majority of all the members elected to each house'). "
        "If the text indicates it is based on members present and voting, or is silent/unclear on this point, FAIL."
    )
    tasks.append((counting_claim, entry.sources, counting_node, counting_instruction))

    # Reference source verification (authoritativeness + relevance)
    reference_claim = (
        f"At least one of these pages is an official or authoritative government source for {entry.state_name} "
        f"(e.g., state constitution, official legislature or governor site, or codified statutes on an official .gov/.state.xx.us domain), "
        f"and it explicitly describes the gubernatorial veto override procedure."
    )
    reference_instruction = (
        "PASS only if at least one URL is an official or authoritative government source (e.g., state legislature website, "
        "state constitution on an official portal, statute code on an official .gov/.state.xx.us domain) AND that page explicitly addresses "
        "the veto override procedure. Ballotpedia or general news/secondary aggregators alone are not sufficient for PASS."
    )
    tasks.append((reference_claim, entry.sources, reference_node, reference_instruction))

    # Run all verifications for this state in parallel
    await evaluator.batch_verify(tasks)


# -----------------------------------------------------------------------------
# Main evaluation function
# -----------------------------------------------------------------------------
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
    Entry point to evaluate an answer for the 'simple majority of all elected members veto override' task.
    """
    # Initialize evaluator
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Root parallel aggregation (four states judged independently)
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

    # Extract up to the first 4 states as provided in the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_states(),
        template_class=StatesExtraction,
        extraction_name="states_extraction"
    )

    # Keep only the first 4 states (pad with empty entries if fewer)
    states: List[StateEntry] = list(extracted.states[:4])
    while len(states) < 4:
        states.append(StateEntry())

    # Add a compact summary of extracted state names to the report
    evaluator.add_custom_info(
        info={"extracted_state_names": [s.state_name for s in states]},
        info_type="extraction_overview",
        info_name="extracted_overview"
    )

    # Build four state subtrees (as parallel children under root per rubric)
    verify_tasks = []
    for i in range(4):
        verify_tasks.append(verify_one_state(evaluator, root, states[i], i))

    # Execute all state verifications concurrently
    await asyncio.gather(*verify_tasks, return_exceptions=False)

    # Return evaluation summary
    return evaluator.get_summary()