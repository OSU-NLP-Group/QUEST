import asyncio
import logging
from typing import Optional, List, Dict

from pydantic import BaseModel, Field
from obj_task_eval.llm_client.base_client import LLMClient

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "western_states_gov_2024"
TASK_DESCRIPTION = (
    "Identify at least three U.S. states that satisfy ALL of the following governmental structure requirements as of 2024: "
    "(1) The state must have a bicameral legislature consisting of two separate legislative chambers. "
    "(2) The state legislature must require a two-thirds vote in both chambers to override a gubernatorial veto. "
    "(3) The state's Attorney General must be directly elected by voters in statewide elections, not appointed by the governor, legislature, or courts. "
    "(4) The state's supreme court justices must be selected through nonpartisan elections where candidates appear on the ballot without party labels. "
    "(5) The state must be geographically located in the Western United States region, including Mountain and Pacific states west of the Great Plains. "
    "For each state you identify, provide the state name, confirmation of bicameral legislature structure, the specific veto override requirement, "
    "the Attorney General selection method, the state supreme court justice selection method, the state's geographic region, and reference URLs supporting each requirement."
)


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class StateRecord(BaseModel):
    name: Optional[str] = None

    bicameral_desc: Optional[str] = None
    bicameral_urls: List[str] = Field(default_factory=list)

    veto_override_desc: Optional[str] = None
    veto_override_urls: List[str] = Field(default_factory=list)

    ag_selection_desc: Optional[str] = None
    ag_selection_urls: List[str] = Field(default_factory=list)

    sc_selection_desc: Optional[str] = None
    sc_selection_urls: List[str] = Field(default_factory=list)

    region_desc: Optional[str] = None
    region_urls: List[str] = Field(default_factory=list)


class StatesExtraction(BaseModel):
    states: List[StateRecord] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_states() -> str:
    return """
    From the provided answer, extract all U.S. states that the answer claims meet the given 2024 governmental structure requirements.
    For each state, extract the following fields exactly as stated in the answer:
    - name: The state's name.
    - bicameral_desc: The description in the answer of the state's bicameral legislature and its two chambers (e.g., "Senate and House of Representatives" or "Senate and Assembly"). If the answer only generally says "bicameral" without naming the chambers, still extract the text provided.
    - bicameral_urls: All URLs provided that support the bicameral legislature claim.
    - veto_override_desc: The description of the veto override requirement (it must say two-thirds in both chambers to override the governor's veto).
    - veto_override_urls: All URLs that support the veto override rule.
    - ag_selection_desc: The description of how the Attorney General is selected (must be directly elected statewide, not appointed).
    - ag_selection_urls: All URLs that support the Attorney General selection method.
    - sc_selection_desc: The description of how state supreme court justices are selected (must be nonpartisan elections with no party labels on the ballot).
    - sc_selection_urls: All URLs that support the supreme court selection method.
    - region_desc: The description of the state's geographic classification in the Western U.S. (Mountain or Pacific, west of the Great Plains).
    - region_urls: All URLs that support the state's Western region classification/location.

    Return a JSON object with a single field:
    {
      "states": [
        {
          "name": "...",
          "bicameral_desc": "...",
          "bicameral_urls": ["...", "..."],
          "veto_override_desc": "...",
          "veto_override_urls": ["...", "..."],
          "ag_selection_desc": "...",
          "ag_selection_urls": ["...", "..."],
          "sc_selection_desc": "...",
          "sc_selection_urls": ["...", "..."],
          "region_desc": "...",
          "region_urls": ["...", "..."]
        },
        ...
      ]
    }

    SPECIAL URL RULES:
    - Extract only valid URLs explicitly present in the answer (plain URLs or markdown links). Do not invent.
    - If no URL is provided for a field, return an empty list for that field.

    If the answer provides more than three states, extract them all. Missing fields should be null (for text) or empty list (for URLs).
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def normalize_state_name(name: Optional[str]) -> str:
    if not name:
        return ""
    return "".join(ch for ch in name.lower().strip() if ch.isalnum())


def ensure_list(urls: Optional[List[str]]) -> List[str]:
    return urls if urls else []


# --------------------------------------------------------------------------- #
# Verification functions                                                      #
# --------------------------------------------------------------------------- #
async def verify_single_state(
    evaluator: Evaluator,
    parent_node,
    state: StateRecord,
    index: int,
) -> None:
    """
    Build verification subtree for one state, with critical existence/URL checks gating evidence-based verifications.
    """
    # Create state-level parallel node (non-critical to allow partial credit across states)
    state_node = evaluator.add_parallel(
        id=f"state_{index+1}",
        desc=f"Evaluate the {'first' if index == 0 else ('second' if index == 1 else 'third')} identified state against all constraints and required outputs",
        parent=parent_node,
        critical=False,
    )

    # Name provided (critical)
    name_provided = bool(state.name and state.name.strip())
    evaluator.add_custom_node(
        result=name_provided,
        id=f"State{index+1}_Name_Provided",
        desc="State name is explicitly provided",
        parent=state_node,
        critical=True,
    )

    # Bicameral legislature URL presence (critical)
    evaluator.add_custom_node(
        result=len(ensure_list(state.bicameral_urls)) > 0,
        id=f"State{index+1}_Bicameral_Legislature_URL",
        desc="Provides a reference URL supporting the bicameral/two-chamber legislature claim",
        parent=state_node,
        critical=True,
    )

    # Bicameral legislature correctness (critical, verify by URLs)
    bicameral_leaf = evaluator.add_leaf(
        id=f"State{index+1}_Bicameral_Legislature_Correct",
        desc="State is described as having a bicameral legislature with two separate chambers (and the description identifies the two chambers)",
        parent=state_node,
        critical=True,
    )
    bicameral_claim = (
        f"The state legislature of {state.name} is bicameral with two chambers "
        f"(e.g., Senate and House of Representatives/Assembly)."
    )
    await evaluator.verify(
        claim=bicameral_claim,
        node=bicameral_leaf,
        sources=ensure_list(state.bicameral_urls),
        additional_instruction=(
            "Confirm that the provided page(s) explicitly show the state has exactly two legislative chambers "
            "(upper and lower), typically named Senate and House of Representatives (or Assembly). "
            "Allow synonyms (e.g., House, Assembly)."
        ),
    )

    # Veto override URL presence (critical)
    evaluator.add_custom_node(
        result=len(ensure_list(state.veto_override_urls)) > 0,
        id=f"State{index+1}_Veto_Override_URL",
        desc="Provides a reference URL supporting the veto-override rule",
        parent=state_node,
        critical=True,
    )

    # Veto override correctness (critical, verify by URLs)
    veto_leaf = evaluator.add_leaf(
        id=f"State{index+1}_Veto_Override_Correct",
        desc="State is described as requiring a two-thirds vote in both chambers to override a gubernatorial veto",
        parent=state_node,
        critical=True,
    )
    veto_claim = (
        f"In {state.name}, overriding a gubernatorial veto requires a two-thirds (2/3) vote in both legislative chambers."
    )
    await evaluator.verify(
        claim=veto_claim,
        node=veto_leaf,
        sources=ensure_list(state.veto_override_urls),
        additional_instruction=(
            "Verify that both the upper and lower chamber must reach two-thirds to override the governor's veto. "
            "Do not accept simple majority or three-fifths. The rule must be two-thirds in both chambers."
        ),
    )

    # Attorney General selection URL presence (critical)
    evaluator.add_custom_node(
        result=len(ensure_list(state.ag_selection_urls)) > 0,
        id=f"State{index+1}_AG_Elected_URL",
        desc="Provides a reference URL supporting the Attorney General selection method",
        parent=state_node,
        critical=True,
    )

    # Attorney General selection correctness (critical, verify by URLs)
    ag_leaf = evaluator.add_leaf(
        id=f"State{index+1}_AG_Elected_Correct",
        desc="State is described as having an Attorney General directly elected by voters statewide (not appointed)",
        parent=state_node,
        critical=True,
    )
    ag_claim = (
        f"In {state.name}, the Attorney General is directly elected by voters statewide (not appointed by the governor, legislature, or courts)."
    )
    await evaluator.verify(
        claim=ag_claim,
        node=ag_leaf,
        sources=ensure_list(state.ag_selection_urls),
        additional_instruction=(
            "Confirm the page(s) explicitly state that the Attorney General is chosen via statewide election, "
            "and not appointed. Accept phrasing like 'elected statewide' or 'popular election'."
        ),
    )

    # Supreme Court selection URL presence (critical)
    evaluator.add_custom_node(
        result=len(ensure_list(state.sc_selection_urls)) > 0,
        id=f"State{index+1}_Supreme_Court_URL",
        desc="Provides a reference URL supporting the supreme court justice selection method",
        parent=state_node,
        critical=True,
    )

    # Supreme Court selection correctness (critical, verify by URLs)
    sc_leaf = evaluator.add_leaf(
        id=f"State{index+1}_Supreme_Court_Nonpartisan_Correct",
        desc="State is described as selecting supreme court justices via nonpartisan elections with no party labels on the ballot",
        parent=state_node,
        critical=True,
    )
    sc_claim = (
        f"In {state.name}, state supreme court justices are selected through nonpartisan elections where candidates appear without party labels on the ballot."
    )
    await evaluator.verify(
        claim=sc_claim,
        node=sc_leaf,
        sources=ensure_list(state.sc_selection_urls),
        additional_instruction=(
            "Confirm the page(s) indicate the election is nonpartisan for supreme court justices; "
            "candidates should be listed without party labels on the ballot."
        ),
    )

    # Western region URL presence (critical)
    evaluator.add_custom_node(
        result=len(ensure_list(state.region_urls)) > 0,
        id=f"State{index+1}_Western_Region_URL",
        desc="Provides a reference URL supporting the state's Western U.S. location/region classification",
        parent=state_node,
        critical=True,
    )

    # Western region correctness (critical, verify by URLs)
    region_leaf = evaluator.add_leaf(
        id=f"State{index+1}_Western_Region_Correct",
        desc="State is described as being geographically located in the Western U.S. (Mountain or Pacific), west of the Great Plains",
        parent=state_node,
        critical=True,
    )
    region_claim = (
        f"The U.S. state {state.name} is classified in the Western United States (Mountain or Pacific), west of the Great Plains."
    )
    await evaluator.verify(
        claim=region_claim,
        node=region_leaf,
        sources=ensure_list(state.region_urls),
        additional_instruction=(
            "Accept authoritative geographic classifications (e.g., U.S. Census Bureau regions or reputable sources) "
            "indicating the state is in the Western U.S. (Mountain or Pacific)."
        ),
    )


# --------------------------------------------------------------------------- #
# Main evaluation function                                                    #
# --------------------------------------------------------------------------- #
async def evaluate_answer(
    client: LLMClient,
    answer: str,
    agent_name: str,
    answer_name: str,
    cache: CacheFileSys,
    semaphore: asyncio.Semaphore,
    logger: logging.Logger,
    model: str = "o4-mini",
) -> Dict:
    """
    Evaluate an answer for the Western U.S. states governmental structure requirements (2024).
    """
    # Initialize evaluator with a parallel root (to allow independent state checks)
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

    # Extract states data
    extracted_states = await evaluator.extract(
        prompt=prompt_extract_states(),
        template_class=StatesExtraction,
        extraction_name="states_extraction",
    )

    # Prepare first three states (pad with empty if needed)
    provided_states = [s for s in extracted_states.states if s.name and s.name.strip()]
    selected_three = provided_states[:3]
    while len(selected_three) < 3:
        selected_three.append(StateRecord())

    # Critical gates node (critical) for global constraints
    gates_node = evaluator.add_parallel(
        id="critical_gates",
        desc="Critical gating: minimum count and distinctness checks",
        parent=root,
        critical=True,
    )

    # At least three states provided (critical)
    evaluator.add_custom_node(
        result=len(provided_states) >= 3,
        id="At_Least_Three_States_Provided",
        desc="Answer identifies at least three states",
        parent=gates_node,
        critical=True,
    )

    # States are distinct among the first three (critical)
    names_first_three = [normalize_state_name(s.name) for s in selected_three if s.name]
    distinct = len(names_first_three) == len(set(names_first_three)) and len(names_first_three) == 3
    evaluator.add_custom_node(
        result=distinct,
        id="States_Are_Distinct",
        desc="The identified states are three different states (no duplicates)",
        parent=gates_node,
        critical=True,
    )

    # Build verification for each of the first three states
    for idx in range(3):
        await verify_single_state(evaluator, root, selected_three[idx], idx)

    # Return summary
    return evaluator.get_summary()