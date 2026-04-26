import asyncio
import logging
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator, AggregationStrategy


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "eclipse_2024_np_longer_totality"
TASK_DESCRIPTION = (
    "Which of the two U.S. national parks located in the path of totality for the April 8, 2024 total solar eclipse "
    "experienced the longer duration of totality? Provide the park name, the state it is located in, and the approximate "
    "totality duration in minutes and seconds."
)


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class ParkAnswerExtraction(BaseModel):
    # The park that the answer claims had the longer totality duration
    selected_park_name: Optional[str] = None
    selected_state: Optional[str] = None
    totality_duration: Optional[str] = None  # string as provided in the answer (e.g., "3m 50s" or "3 minutes 50 seconds")

    # If the answer mentions the other park (for comparison), capture it too
    other_park_name: Optional[str] = None
    other_park_duration: Optional[str] = None

    # All URLs cited in the answer (as a fallback pool)
    source_urls: List[str] = Field(default_factory=list)

    # Optional categorization of sources if the answer presents them distinctly
    path_sources: List[str] = Field(default_factory=list)        # URLs supporting the park being in the totality path
    duration_sources: List[str] = Field(default_factory=list)    # URLs supporting the selected park's duration
    state_sources: List[str] = Field(default_factory=list)       # URLs supporting location/state info
    comparison_sources: List[str] = Field(default_factory=list)  # URLs directly comparing durations of the two parks


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_park_answer() -> str:
    return """
    Extract from the provided answer the single U.S. national park that the answer claims had the longer duration
    of totality among the two U.S. national parks that were in the path of totality for the April 8, 2024 total solar eclipse.
    Return the following fields:
    - selected_park_name: the name of the park that is claimed to have the longer duration of totality.
    - selected_state: the U.S. state for the selected park (as stated in the answer).
    - totality_duration: the approximate totality duration for the selected park in the answer (keep it as written, e.g., "3m 50s" or "3 minutes 50 seconds").
    - other_park_name: if the answer mentions the other national park in the path of totality, provide its name; otherwise null.
    - other_park_duration: if the answer provides a duration (approximate or specific) for the other park, extract it; otherwise null.
    - source_urls: a list of all URLs cited in the answer.
    - path_sources: URLs used in the answer to support that the selected park is in the April 8, 2024 path of totality (if any).
    - duration_sources: URLs used to support the totality duration of the selected park (if any).
    - state_sources: URLs used to support the location/state of the selected park (if any).
    - comparison_sources: URLs used to directly compare the totality durations of the two parks (if any).
    
    SPECIAL RULES FOR URL EXTRACTION:
    - Extract only URLs explicitly present in the answer (plain or markdown links). Do not invent URLs.
    - If the answer shows multiple URLs but doesn't categorize them, include all of them in 'source_urls' and leave the specific lists empty.
    - If a URL is missing a protocol, prepend "http://".
    - If any field is missing in the answer, set it to null (or [] for lists).
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _pick_sources(preferred_lists: List[List[str]], fallback: List[str]) -> List[str]:
    """Pick the first non-empty list from preferred_lists; otherwise return fallback (which can be empty)."""
    for lst in preferred_lists:
        if lst:
            return lst
    return fallback or []


# --------------------------------------------------------------------------- #
# Verification logic                                                          #
# --------------------------------------------------------------------------- #
async def build_and_verify_tree(evaluator: Evaluator, extraction: ParkAnswerExtraction) -> None:
    """
    Build the verification tree according to the rubric and run verifications using the evaluator.
    """
    # Create top-level critical sequential node per rubric
    np_root = evaluator.add_sequential(
        id="National_Park_Identification",
        desc="Correctly identify which of the two U.S. national parks in the April 8, 2024 total solar eclipse path of totality experienced the longer duration of totality, and provide all required information",
        parent=evaluator.root,
        critical=True
    )

    # ---------------------- Park Selection Criteria ---------------------- #
    criteria_node = evaluator.add_parallel(
        id="Park_Selection_Criteria",
        desc="The identified park meets the selection criteria specified in the question",
        parent=np_root,
        critical=True
    )

    # Park_In_Totality_Path (critical leaf)
    pitp_node = evaluator.add_leaf(
        id="Park_In_Totality_Path",
        desc="The identified park is one of the two U.S. national parks that were in the path of totality for the April 8, 2024 eclipse",
        parent=criteria_node,
        critical=True
    )

    selected_park = extraction.selected_park_name or ""
    # Choose sources: prefer path_sources, else all source_urls
    pitp_sources = _pick_sources([extraction.path_sources], extraction.source_urls)

    if not pitp_sources or not selected_park:
        # Missing sources or missing park name => fail this critical leaf
        pitp_node.score = 0.0
        pitp_node.status = "failed"
    else:
        pitp_claim = f"{selected_park} was in the path of totality for the April 8, 2024 total solar eclipse."
        await evaluator.verify(
            claim=pitp_claim,
            node=pitp_node,
            sources=pitp_sources,
            additional_instruction="Confirm that this national park lay within the path of totality on April 8, 2024. Do not accept partial eclipse."
        )

    # Longer_Totality_Duration (critical leaf)
    longer_node = evaluator.add_leaf(
        id="Longer_Totality_Duration",
        desc="The identified park had a longer totality duration than the other U.S. national park in the path of totality",
        parent=criteria_node,
        critical=True
    )

    # Prefer direct comparison sources; otherwise use all URLs
    longer_sources = _pick_sources([extraction.comparison_sources, extraction.duration_sources], extraction.source_urls)
    if not longer_sources or not selected_park:
        longer_node.score = 0.0
        longer_node.status = "failed"
    else:
        if extraction.other_park_name:
            longer_claim = f"During the April 8, 2024 total solar eclipse, the totality duration at {selected_park} was longer than at {extraction.other_park_name}."
        else:
            longer_claim = (
                f"During the April 8, 2024 total solar eclipse, the totality duration at {selected_park} "
                f"was longer than at the other U.S. national park that lay in the path of totality."
            )
        await evaluator.verify(
            claim=longer_claim,
            node=longer_node,
            sources=longer_sources,
            additional_instruction="Look for an explicit comparison or clearly implied longer duration at the selected park than the other U.S. national park that was in the 2024 path of totality. Minor wording or rounding differences are acceptable, but the comparison must be supported."
        )

    # ---------------------- Required Information ------------------------- #
    required_node = evaluator.add_parallel(
        id="Required_Information",
        desc="All information explicitly requested in the question is provided accurately",
        parent=np_root,
        critical=True
    )

    # Park_Name_Provided (critical existence check)
    name_provided = bool(extraction.selected_park_name and extraction.selected_park_name.strip())
    evaluator.add_custom_node(
        result=name_provided,
        id="Park_Name_Provided",
        desc="The name of the national park is provided",
        parent=required_node,
        critical=True
    )

    # State_Location_Provided (critical leaf with verification)
    state_node = evaluator.add_leaf(
        id="State_Location_Provided",
        desc="The U.S. state in which the park is located is provided correctly",
        parent=required_node,
        critical=True
    )
    state_sources = _pick_sources([extraction.state_sources], extraction.source_urls)
    if not state_sources or not extraction.selected_state or not selected_park:
        state_node.score = 0.0
        state_node.status = "failed"
    else:
        state_claim = f"{selected_park} is located in {extraction.selected_state}."
        await evaluator.verify(
            claim=state_claim,
            node=state_node,
            sources=state_sources,
            additional_instruction="Use authoritative sources (e.g., NPS or Wikipedia) to confirm the state. If the park spans multiple states, consider it correct if the stated state is one of them."
        )

    # Totality_Duration_Provided (critical leaf with verification)
    duration_node = evaluator.add_leaf(
        id="Totality_Duration_Provided",
        desc="The approximate totality duration in minutes and seconds is provided",
        parent=required_node,
        critical=True
    )
    duration_sources = _pick_sources([extraction.duration_sources], extraction.source_urls)
    if not duration_sources or not extraction.totality_duration or not selected_park:
        duration_node.score = 0.0
        duration_node.status = "failed"
    else:
        dur_str = extraction.totality_duration
        duration_claim = f"The totality duration at {selected_park} on April 8, 2024 was approximately {dur_str}."
        await evaluator.verify(
            claim=duration_claim,
            node=duration_node,
            sources=duration_sources,
            additional_instruction="Allow reasonable approximations or rounding (e.g., ±10 seconds). Verify using the cited page(s) that the stated duration is plausible and supported."
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
    Entrypoint to evaluate an answer for the eclipse national park totality question.
    Returns the evaluation summary dictionary.
    """
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.SEQUENTIAL,
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

    # Extract structured information from the answer
    extraction = await evaluator.extract(
        prompt=prompt_extract_park_answer(),
        template_class=ParkAnswerExtraction,
        extraction_name="park_answer_extraction"
    )

    # Optionally record custom info for debugging
    evaluator.add_custom_info(
        info={
            "selected_park_name": extraction.selected_park_name,
            "selected_state": extraction.selected_state,
            "totality_duration": extraction.totality_duration,
            "other_park_name": extraction.other_park_name,
            "other_park_duration": extraction.other_park_duration,
            "source_counts": {
                "source_urls": len(extraction.source_urls),
                "path_sources": len(extraction.path_sources),
                "duration_sources": len(extraction.duration_sources),
                "state_sources": len(extraction.state_sources),
                "comparison_sources": len(extraction.comparison_sources),
            }
        },
        info_type="extraction_overview",
        info_name="extraction_overview"
    )

    # Build verification tree and run checks
    await build_and_verify_tree(evaluator, extraction)

    # Return evaluation summary
    return evaluator.get_summary()