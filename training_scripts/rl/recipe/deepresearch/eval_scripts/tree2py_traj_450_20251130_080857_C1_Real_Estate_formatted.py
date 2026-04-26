import asyncio
import logging
import re
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy

# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "real_estate_min_prelic_hours"
TASK_DESCRIPTION = (
    "I am considering becoming a real estate agent and want to minimize the time spent on pre-licensing education. "
    "Among the following five states—Michigan, Florida, Virginia, Pennsylvania, and Georgia—which state requires the "
    "fewest pre-licensing education hours for a real estate salesperson license, and how many hours are required?"
)

ALLOWED_STATES = ["Michigan", "Florida", "Virginia", "Pennsylvania", "Georgia"]
STATE_ABBREVIATIONS = {
    "mi": "Michigan",
    "fl": "Florida",
    "va": "Virginia",
    "pa": "Pennsylvania",
    "ga": "Georgia",
}


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class SelectionExtraction(BaseModel):
    """
    Extracted selection details from the answer:
    - selected_state: the state the answer identifies as having the fewest pre-licensing education hours.
    - hours: the stated numeric hours for that state's pre-licensing requirement (string as presented).
    - citations: URLs cited to support the hours requirement for the identified state.
    """
    selected_state: Optional[str] = None
    hours: Optional[str] = None
    citations: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_selection() -> str:
    return """
    From the answer, extract the single state among this set — Michigan, Florida, Virginia, Pennsylvania, Georgia — 
    that the answer claims has the fewest state-mandated pre-licensing education hours for a real estate salesperson 
    (a.k.a. sales agent or, in Florida, sales associate) license.

    Extract the following fields:
    1) selected_state: The name of the state as written in the answer that is claimed to have the fewest pre-licensing hours among the five specified states. 
       If the answer does not clearly choose one state, return null.
    2) hours: The specific number of pre-licensing education hours that the answer states for the identified state. 
       Return it exactly as written (e.g., "40", "63 hours"). If not specified, return null.
    3) citations: An array of all URL(s) the answer provides that specifically support the stated pre-licensing hour requirement for the identified state. 
       Include official regulator sites (e.g., state real estate commission/regulator), approved education providers, or recognized real estate education platforms if present. 
       Only include actual URLs found in the answer. If none are provided, return an empty array.

    Do NOT infer or invent any information. Only extract what is explicitly present in the answer.
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def normalize_state_name(name: Optional[str]) -> Optional[str]:
    if not name:
        return None
    raw = re.sub(r"[^a-zA-Z ]", "", name).strip().lower()
    # direct match
    for st in ALLOWED_STATES:
        if raw == st.lower():
            return st
    # abbreviation match
    abbr = raw.replace(".", "").replace(" ", "")
    if abbr in STATE_ABBREVIATIONS:
        return STATE_ABBREVIATIONS[abbr]
    # fuzzy contain (e.g., "State of Michigan")
    for st in ALLOWED_STATES:
        if st.lower() in raw:
            return st
    return None


def extract_numeric_hours(text: Optional[str]) -> Optional[int]:
    """
    Extract the first reasonable integer number from an hours text (e.g., "63 hours", "40-hour", "75").
    Returns None if no digits found.
    """
    if not text:
        return None
    match = re.search(r"(\d{1,3})", text)
    if match:
        try:
            val = int(match.group(1))
            if 1 <= val <= 300:
                return val
        except Exception:
            return None
    return None


# --------------------------------------------------------------------------- #
# Verification construction                                                   #
# --------------------------------------------------------------------------- #
async def build_and_verify(
    evaluator: Evaluator,
    parent_node,
    selection: SelectionExtraction,
) -> None:
    """
    Build the verification tree under 'answer_validation' and perform checks per rubric.
    """
    # Create the critical parallel node for overall validation
    val_node = evaluator.add_parallel(
        id="answer_validation",
        desc=(
            "Validate the answer identifying which of the five specified states has the fewest state-mandated "
            "pre-licensing education hours for a real estate salesperson license, and report the required hours "
            "with authoritative citation(s)."
        ),
        parent=parent_node,
        critical=True,
    )

    # Normalize selected state
    normalized_state = normalize_state_name(selection.selected_state)

    # 1) state_in_allowed_set (critical)
    state_ok = normalized_state in ALLOWED_STATES
    evaluator.add_custom_node(
        result=state_ok,
        id="state_in_allowed_set",
        desc="The answer names a state that is one of: Michigan, Florida, Virginia, Pennsylvania, Georgia.",
        parent=val_node,
        critical=True,
    )

    # 2) license_type_correct (critical) — verify the answer talks about salesperson/agent (not broker)
    lic_type_node = evaluator.add_leaf(
        id="license_type_correct",
        desc="The answer clearly addresses a real estate salesperson (salesperson/agent) license requirement, not a broker license requirement.",
        parent=val_node,
        critical=True,
    )
    await evaluator.verify(
        claim=(
            "The licensing requirement discussed in the answer refers to a real estate 'salesperson' or 'sales agent' "
            "(Florida uses the term 'sales associate') license, and not a broker license."
        ),
        node=lic_type_node,
        additional_instruction=(
            "Judge only based on the answer text. Accept synonyms like 'real estate agent' or 'sales associate' as "
            "equivalent to 'salesperson'. If the answer is ambiguous or mentions broker requirements, mark incorrect."
        ),
    )

    # 3) requirement_type_correct (critical) — verify reported hours are for pre-licensing education
    req_type_node = evaluator.add_leaf(
        id="requirement_type_correct",
        desc="The hours reported are for state-mandated pre-licensing education (not continuing education, post-licensing, renewal, or other training).",
        parent=val_node,
        critical=True,
    )
    await evaluator.verify(
        claim=(
            "The hours being discussed in the answer specifically refer to the required pre-licensing education "
            "that must be completed before applying for the salesperson license (not continuing ed, post-licensing, or renewal)."
        ),
        node=req_type_node,
        additional_instruction=(
            "Look for the explicit phrases like pre-licensing, prelicense, or pre-license. If unclear or mixed with "
            "other types of training, mark incorrect."
        ),
    )

    # 4) minimum_among_five (critical) — verify the answer explicitly claims the identified state has the fewest
    min_claim_node = evaluator.add_leaf(
        id="minimum_among_five",
        desc="The answer claims the identified state has the fewest required pre-licensing education hours among the five specified states.",
        parent=val_node,
        critical=True,
    )
    claim_txt = (
        f"The answer explicitly claims that {normalized_state or 'the identified state'} has the fewest required "
        "pre-licensing education hours among Michigan, Florida, Virginia, Pennsylvania, and Georgia."
    )
    await evaluator.verify(
        claim=claim_txt,
        node=min_claim_node,
        additional_instruction=(
            "You are only checking whether the answer makes this comparative claim. If the answer states or clearly implies "
            "that the chosen state has the lowest (or is tied for the lowest) among the five, mark correct; otherwise incorrect."
        ),
    )

    # 5) hours_reported (critical) — ensure a specific numeric number is stated
    hours_numeric = extract_numeric_hours(selection.hours)
    evaluator.add_custom_node(
        result=hours_numeric is not None,
        id="hours_reported",
        desc="The answer states a specific numeric number of required pre-licensing education hours for the identified state.",
        parent=val_node,
        critical=True,
    )

    # 6) authoritative_citations (critical) — at least one URL that supports the hours for the identified state
    citations_node = evaluator.add_leaf(
        id="authoritative_citations",
        desc=(
            "The answer provides at least one URL citation to an authoritative source (e.g., official state real estate "
            "commission/regulator, approved education provider, or recognized real estate education platform) that supports "
            "the stated pre-licensing hour requirement for the identified state."
        ),
        parent=val_node,
        critical=True,
    )

    # If no citations, mark failed directly; otherwise verify against the provided URLs
    if not selection.citations:
        citations_node.score = 0.0
        citations_node.status = "failed"
    else:
        # Build claim for URL verification
        state_for_claim = normalized_state or (selection.selected_state or "the identified state")
        hours_for_claim = selection.hours or (str(hours_numeric) if hours_numeric is not None else "the stated number of")
        url_claim = (
            f"The cited source supports that {state_for_claim} requires {hours_for_claim} hours of "
            "pre-licensing education for a real estate salesperson (sales agent/sales associate) license."
        )
        await evaluator.verify(
            claim=url_claim,
            node=citations_node,
            sources=selection.citations,
            additional_instruction=(
                "Confirm that the webpage explicitly states or clearly supports the exact pre-licensing hours requirement "
                f"for {state_for_claim}. Prefer official regulator sites (e.g., Michigan LARA, Florida DBPR/FREC, Virginia DPOR, "
                "Pennsylvania Real Estate Commission, Georgia Real Estate Commission), or reputable education providers/platforms. "
                "Minor wording variations are acceptable, but the hours and license type must match."
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
    Evaluate an answer for the real estate pre-licensing minimum-hours task.
    """
    # Initialize evaluator with a parallel root to host the critical sub-tree
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

    # Record ground-truth context (allowed states)
    evaluator.add_ground_truth({
        "allowed_states": ALLOWED_STATES,
        "note": "Only states within this list are valid for selection."
    })

    # Extract selection details from the answer
    selection = await evaluator.extract(
        prompt=prompt_extract_selection(),
        template_class=SelectionExtraction,
        extraction_name="selection_extraction",
    )

    # Build and execute verification
    await build_and_verify(evaluator, root, selection)

    # Return structured summary
    return evaluator.get_summary()