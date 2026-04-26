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
TASK_ID = "open_meetings_48h_4_states"
TASK_DESCRIPTION = """
Identify 4 U.S. states that require at least 48 hours advance notice for regular public meetings under their open meetings laws. For each of the 4 states, provide the following information:

1. The specific minimum advance notice period required for regular meetings (e.g., '48 hours', '72 hours')
2. Whether the state law requires online posting of meeting documents, agendas, or materials before the meeting (answer 'Yes' with brief details, or 'No')
3. Any specific deadline or timeframe required for filing or publishing meeting minutes after a meeting occurs (if such a requirement is specified in the state's open meetings law; if not specified, state 'Not specified')
4. A reference URL to an official state government source that documents these requirements

Present your answer in a structured format with each state clearly labeled.
"""

# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class StateRequirement(BaseModel):
    """Structured info for a single state's open meetings requirements."""
    state_name: Optional[str] = None
    notice_period: Optional[str] = None  # e.g., "48 hours", "72 hours", "two business days"
    document_posting: Optional[str] = None  # e.g., "Yes: agenda must be posted online 48 hours prior", or "No"
    minutes_deadline: Optional[str] = None  # e.g., "Within 10 days", "By next meeting", or "Not specified"
    reference_urls: List[str] = Field(default_factory=list)  # Official state gov sources


class RequirementsExtraction(BaseModel):
    """Collection of up to four states extracted from the answer."""
    states: List[StateRequirement] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_states() -> str:
    return """
    Extract up to 4 U.S. states and their open meetings law requirements as presented in the answer. For each identified state, extract the following fields exactly as stated in the answer:

    - state_name: The U.S. state's name (e.g., "Texas", "Arizona").
    - notice_period: The specific minimum advance notice for regular public meetings (e.g., "48 hours", "72 hours", "two business days"). Do not convert units; return the text as given in the answer.
    - document_posting: Whether the state requires online posting of meeting documents/agendas/materials before the meeting. Return "Yes: ..." with brief details if the answer states such a requirement is mandatory; otherwise return "No".
    - minutes_deadline: Any specific statutory deadline/timeframe for publishing or filing meeting minutes after the meeting (e.g., "Within 10 days", "By next meeting"). If the answer says there is no specified deadline, return "Not specified". If not mentioned at all, return null.
    - reference_urls: A list of the official state government source URLs cited in the answer that document these requirements. Include full URLs; the answer may present them in plain form or markdown. If none are provided, return an empty list.

    Rules:
    1. Extract ONLY what is explicitly present in the answer text; do not invent or infer.
    2. If any field is missing for a state, set it to null (except reference_urls which should be an empty list if none given).
    3. Return a JSON object with a single key "states" that is an array of up to 4 objects matching the schema above. Preserve the order the states appear in the answer.
    4. Do not include more than 4 states in the output even if the answer mentions more; return only the first 4.
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def ordinal(n: int) -> str:
    mapping = {1: "First", 2: "Second", 3: "Third", 4: "Fourth"}
    return mapping.get(n, f"#{n}")


def _has_reference_urls(state: StateRequirement) -> bool:
    return bool(state and state.reference_urls and len(state.reference_urls) > 0)


def _state_label(idx: int, state_name: Optional[str]) -> str:
    if state_name and state_name.strip():
        return f"{ordinal(idx + 1)} identified state ({state_name})"
    return f"{ordinal(idx + 1)} identified state"


# --------------------------------------------------------------------------- #
# Verification functions                                                      #
# --------------------------------------------------------------------------- #
async def verify_state(
    evaluator: Evaluator,
    parent_node,
    state: StateRequirement,
    idx: int,
) -> None:
    """
    Build the verification subtree for one state and run verifications.
    """
    # Parent node for this state (non-critical to allow partial scoring across states)
    state_node = evaluator.add_parallel(
        id=f"state_{idx + 1}",
        desc=f"{_state_label(idx, state.state_name)} has accurate requirements",
        parent=parent_node,
        critical=False
    )

    # Optional existence gate: require state name and at least one reference URL
    evaluator.add_custom_node(
        result=bool(state and state.state_name and state.state_name.strip()) and _has_reference_urls(state),
        id=f"state_{idx + 1}_exists",
        desc=f"{_state_label(idx, state.state_name)} includes a state name and at least one reference URL",
        parent=state_node,
        critical=True
    )

    # 1) Reference validity (Critical)
    ref_leaf = evaluator.add_leaf(
        id=f"state_{idx + 1}_reference",
        desc="A valid reference URL to official state government source is provided",
        parent=state_node,
        critical=True
    )
    ref_claim = (
        f"At least one of these URLs is an official state government source for {state.state_name} "
        f"and it documents open meeting requirements (e.g., notice, agenda posting, minutes)."
    )
    await evaluator.verify(
        claim=ref_claim,
        node=ref_leaf,
        sources=state.reference_urls,
        additional_instruction=(
            "Treat official state government sources as pages under .gov domains or official state legislature/assembly websites. "
            "Accept pages that directly present statutes, administrative code, or official guidance about open/public meetings. "
            "Do not accept third-party or advocacy sites. Verify that the page content pertains to open/public meetings requirements."
        ),
    )

    # 2) Notice period (Critical)
    notice_leaf = evaluator.add_leaf(
        id=f"state_{idx + 1}_notice_period",
        desc="The specific minimum advance notice period for regular meetings is correctly stated and meets the at least 48 hours requirement",
        parent=state_node,
        critical=True
    )
    notice_text = state.notice_period or ""
    notice_claim = (
        f"Under {state.state_name}'s open meetings law, regular public meetings require a minimum advance notice of '{notice_text}', "
        f"and this minimum is at least 48 hours."
    )
    await evaluator.verify(
        claim=notice_claim,
        node=notice_leaf,
        sources=state.reference_urls,
        additional_instruction=(
            "Focus on regular (non-emergency) meetings. Accept equivalent phrasing such as 'two days', 'two business days', "
            "'48 hours', '72 hours', etc. If the law's minimum for regular meetings is less than 48 hours, mark as not supported. "
            "If the page only discusses special/emergency meetings, do not use that to support the minimum for regular meetings."
        ),
    )

    # 3) Document posting requirement (Critical)
    posting_leaf = evaluator.add_leaf(
        id=f"state_{idx + 1}_document_posting",
        desc="Whether the state requires online posting of meeting documents/agendas before meetings is correctly identified",
        parent=state_node,
        critical=True
    )
    posting_text = state.document_posting or ""
    # Build claim depending on Yes/No wording to aid verification
    if posting_text.strip().lower().startswith("yes"):
        posting_claim = (
            f"{state.state_name}'s open meetings law requires online posting of meeting agendas/documents/materials before the meeting. "
            f"Details: {posting_text}."
        )
        posting_instruction = (
            "Confirm that the statute or official guidance indicates a mandatory requirement to post agendas or meeting materials online "
            "prior to the meeting (e.g., 'shall post on website', 'must publish online'). If the requirement is optional or "
            "only for physical posting (e.g., bulletin board), do not accept as 'Yes'."
        )
    else:
        posting_claim = (
            f"{state.state_name}'s open meetings law does not require online posting of meeting agendas/documents/materials before the meeting; "
            f"no mandatory requirement is specified."
        )
        posting_instruction = (
            "If the official page does not specify any mandatory requirement to post agendas or materials on a website prior to the meeting, "
            "consider the 'No' claim supported. Absence of a stated online posting requirement should be treated as supportive for 'No'. "
            "If the page explicitly mandates online posting prior to meetings, the 'No' claim is not supported."
        )
    await evaluator.verify(
        claim=posting_claim,
        node=posting_leaf,
        sources=state.reference_urls,
        additional_instruction=posting_instruction,
    )

    # 4) Minutes deadline (Non-Critical)
    minutes_leaf = evaluator.add_leaf(
        id=f"state_{idx + 1}_minutes_deadline",
        desc="The deadline for filing or publishing meeting minutes after the meeting is correctly stated (if specified in state law)",
        parent=state_node,
        critical=False
    )
    if (state.minutes_deadline or "").strip().lower() in ("not specified", "none", ""):
        minutes_claim = (
            f"{state.state_name}'s open meetings law does not specify a fixed deadline for publishing or filing meeting minutes after the meeting."
        )
        minutes_instruction = (
            "Treat 'Not specified' as supported if the official page does not set a specific timeframe (e.g., a number of days) "
            "for publishing or filing minutes. General wording such as 'promptly' without a fixed timeframe counts as 'Not specified'."
        )
    else:
        minutes_claim = (
            f"{state.state_name}'s open meetings law specifies the following deadline/timeframe for minutes after the meeting: "
            f"'{state.minutes_deadline}'."
        )
        minutes_instruction = (
            "Verify that the official page explicitly states the cited timeframe or deadline for publishing or filing minutes after the meeting. "
            "Accept variants such as 'by the next meeting', 'within X days', or similar."
        )
    await evaluator.verify(
        claim=minutes_claim,
        node=minutes_leaf,
        sources=state.reference_urls,
        additional_instruction=minutes_instruction,
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
    model: str = "o4-mini"
) -> Dict:
    """
    Evaluate an answer for the open meetings law requirements task.
    """
    # Initialize evaluator
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # States evaluated independently
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

    # Extract structured states info from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_states(),
        template_class=RequirementsExtraction,
        extraction_name="states_requirements",
    )

    # Limit to first 4 states and pad if fewer
    states: List[StateRequirement] = list(extracted.states[:4])
    while len(states) < 4:
        states.append(StateRequirement())

    # Build verification subtrees for each of the 4 states
    for i, st in enumerate(states):
        await verify_state(evaluator, root, st, i)

    # Return structured result
    return evaluator.get_summary()