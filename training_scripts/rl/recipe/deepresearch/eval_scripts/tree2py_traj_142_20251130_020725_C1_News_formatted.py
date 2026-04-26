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
TASK_ID = "nec_state_identification"
TASK_DESCRIPTION = """
Evaluate whether the correct U.S. state has been identified for the person announced in November 2024 to serve as Director of the White House National Economic Council in President-elect Donald Trump's incoming administration.
"""


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class AppointeeInfo(BaseModel):
    """Information about the NEC Director appointee as stated in the answer."""
    name: Optional[str] = None
    role: Optional[str] = None
    announcement_month: Optional[str] = None
    announcement_year: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class StateInfo(BaseModel):
    """State answer and any sources provided for the state claim."""
    state_name: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class NECAppointmentExtraction(BaseModel):
    """Combined extraction for the NEC appointment and the state answer."""
    appointee: Optional[AppointeeInfo] = None
    state: Optional[StateInfo] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_nec_appointment() -> str:
    return """
    Extract the specific information from the answer needed to verify the question:
    "What U.S. state is the person from who was announced in November 2024 to serve as Director of the White House National Economic Council in President-elect Donald Trump's incoming administration?"

    You must extract:
    - appointee.name: The full name of the person the answer claims was announced to serve as Director of the White House National Economic Council (NEC).
    - appointee.role: The role/title as written in the answer for that person (e.g., "Director of the National Economic Council", "NEC Director").
    - appointee.announcement_month: The month of the announcement as stated in the answer, if present.
    - appointee.announcement_year: The year of the announcement as stated in the answer, if present.
    - appointee.sources: All URL sources explicitly provided in the answer that support the appointment claim. Include every URL mentioned (plain URL or within markdown). If none, return an empty list.

    - state.state_name: The U.S. state given as the final answer to the question (e.g., "Florida", "New York"). If the answer provides multiple states, choose the one explicitly presented as the final answer. If no state is provided, set to null.
    - state.sources: All URL sources explicitly provided in the answer that support the state claim. If none, return an empty list.

    Rules:
    - Extract only what is explicitly present in the answer text.
    - Do not invent names, roles, dates, or URLs.
    - For URLs, extract the actual URL string (normalize if missing protocol by prepending http://).
    - If some fields are missing in the answer, set them to null or empty lists as appropriate.
    """


# --------------------------------------------------------------------------- #
# Helpers                                                                     #
# --------------------------------------------------------------------------- #
def combine_sources(extracted: NECAppointmentExtraction) -> List[str]:
    """Combine and deduplicate all URLs from appointee and state sections."""
    urls: List[str] = []
    if extracted and extracted.appointee and extracted.appointee.sources:
        urls.extend(extracted.appointee.sources)
    if extracted and extracted.state and extracted.state.sources:
        urls.extend(extracted.state.sources)
    # Deduplicate while preserving order
    seen = set()
    deduped: List[str] = []
    for u in urls:
        if u and u not in seen:
            seen.add(u)
            deduped.append(u)
    return deduped


# --------------------------------------------------------------------------- #
# Verification tree construction                                              #
# --------------------------------------------------------------------------- #
async def build_state_identification_tree(
    evaluator: Evaluator,
    parent_node,
    extracted: NECAppointmentExtraction,
) -> None:
    """
    Build the verification tree following the rubric:
    1) Appointee_Meets_Constraints (critical leaf)
    2) State_Answer_Provided (critical leaf)
    3) State_Is_Correct (critical leaf)
    All under a critical sequential node State_Identification_Task.
    """
    # Top-level critical sequential node (represents the rubric root)
    task_node = evaluator.add_sequential(
        id="State_Identification_Task",
        desc="Evaluate whether the correct U.S. state has been identified for the person announced in November 2024 as Director of the White House National Economic Council in President-elect Trump's incoming administration",
        parent=parent_node,
        critical=True,
    )

    # Prepare extracted values
    person_name = extracted.appointee.name if extracted and extracted.appointee else None
    role_title = extracted.appointee.role if extracted and extracted.appointee else None
    # We require Nov 2024 per task constraints; enforce in the claim
    sources_for_appointee = extracted.appointee.sources if extracted and extracted.appointee else []
    state_name = extracted.state.state_name if extracted and extracted.state else None
    sources_for_state = extracted.state.sources if extracted and extracted.state else []
    all_sources = combine_sources(extracted)

    # 1) Appointee_Meets_Constraints (critical verification leaf)
    appointee_node = evaluator.add_leaf(
        id="Appointee_Meets_Constraints",
        desc="The person whose state is provided must be the individual who was announced to serve as Director of the White House National Economic Council in November 2024 for President-elect Donald Trump's incoming administration",
        parent=task_node,
        critical=True,
    )

    # Build claim for appointee constraints (explicitly requires November 2024, NEC Director, and Trump incoming admin)
    appointee_claim = (
        f"{person_name or '[UNKNOWN PERSON]'} was announced in November 2024 to serve as Director of the White House National Economic Council "
        f"in President-elect Donald Trump's incoming administration."
    )
    await evaluator.verify(
        claim=appointee_claim,
        node=appointee_node,
        sources=sources_for_appointee,
        additional_instruction=(
            "This must be supported directly by the provided URLs. Accept equivalent wording like 'NEC Director', "
            "'Director of the National Economic Council', or 'to lead the National Economic Council'. "
            "The announcement timing must be November 2024. If the person name is missing/unknown in the claim or if no URLs are provided, "
            "or if URLs do not explicitly support this appointment and timing, then the claim is not supported."
        ),
    )

    # 2) State_Answer_Provided (critical existence leaf)
    state_provided_node = evaluator.add_custom_node(
        result=bool(state_name and state_name.strip()),
        id="State_Answer_Provided",
        desc="A U.S. state name is clearly provided as the answer",
        parent=task_node,
        critical=True,
    )

    # 3) State_Is_Correct (critical verification leaf)
    state_correct_node = evaluator.add_leaf(
        id="State_Is_Correct",
        desc="The provided state correctly identifies the home state of the appointee who meets all the specified constraints",
        parent=task_node,
        critical=True,
    )

    # Build claim for state correctness (home state / 'from' state)
    state_claim = (
        f"The home state (the U.S. state they are from) of {person_name or '[UNKNOWN PERSON]'} is {state_name or '[UNKNOWN STATE]'}."
    )
    await evaluator.verify(
        claim=state_claim,
        node=state_correct_node,
        sources=all_sources if all_sources else None,
        additional_instruction=(
            "Judge strictly based on the provided URLs. 'From' or 'home state' may refer to birthplace or well-established home/residence; "
            "focus on widely recognized attribution (e.g., 'from Florida'). If no URLs are provided or the URLs do not clearly support the "
            "state attribution, mark as not supported. Also ensure the person in sources is the same NEC Director appointee verified earlier."
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
) -> Dict[str, Any]:
    """
    Evaluate an answer for the NEC Director state identification task.
    """
    # Initialize evaluator
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
        prompt=prompt_extract_nec_appointment(),
        template_class=NECAppointmentExtraction,
        extraction_name="nec_appointment_extraction",
    )

    # Build and run verification tree
    await build_state_identification_tree(evaluator, root, extracted)

    # Return structured summary
    return evaluator.get_summary()