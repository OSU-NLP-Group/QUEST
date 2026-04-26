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
TASK_ID = "np_relocation_states"
TASK_DESCRIPTION = (
    "A registered nurse currently holds a multistate compact license with Florida as their primary state of residence. "
    "They are completing their Doctor of Nursing Practice (DNP) degree and planning to relocate to work as an independent nurse practitioner. "
    "They are evaluating three potential destination states: Montana, Tennessee, and Rhode Island. For each of these three states, determine: "
    "(1) Whether the state is a member of the Nurse Licensure Compact (NLC), which would allow the nurse to practice using their existing Florida multistate RN license; "
    "(2) Whether the state grants full practice authority to nurse practitioners, allowing independent practice without physician supervision or collaborative agreements; "
    "(3) What specific licensure requirements the nurse must fulfill within the first 60 days of establishing residency in that state (if relocating from another NLC compact state); "
    "(4) The official website URL of the state's board of nursing or authoritative source for verification. Provide this information for all three states (Montana, Tennessee, and Rhode Island)."
)

# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class StateInfo(BaseModel):
    """Information extracted for a single state."""
    state_name: Optional[str] = None

    # (1) NLC membership as stated in the answer (yes/no/unknown or textual)
    nlc_membership: Optional[str] = None
    nlc_sources: List[str] = Field(default_factory=list)

    # (2) Full practice authority (FPA) as stated in the answer (yes/no/unknown or textual)
    practice_authority: Optional[str] = None
    practice_sources: List[str] = Field(default_factory=list)

    # (3) 60-day licensure requirement statement (free text as stated)
    requirement_60_day_text: Optional[str] = None
    requirements_sources: List[str] = Field(default_factory=list)

    # (4) Official board/authoritative website URL
    board_url: Optional[str] = None


class StatesExtraction(BaseModel):
    """Container for all states' extracted info."""
    montana: Optional[StateInfo] = None
    tennessee: Optional[StateInfo] = None
    rhode_island: Optional[StateInfo] = None


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_states() -> str:
    return (
        "Extract the information provided in the answer for each of these states: Montana, Tennessee, and Rhode Island.\n"
        "For each state, return a JSON object with the following fields:\n"
        "1) state_name: The state's name.\n"
        "2) nlc_membership: Whether the state is a member of the Nurse Licensure Compact (NLC), as explicitly stated in the answer (use 'yes', 'no', or a short textual phrase; if not stated, use 'unknown').\n"
        "3) nlc_sources: All URLs cited in the answer that support the NLC membership determination for this state.\n"
        "4) practice_authority: Whether the state grants full practice authority (independent practice without physician supervision) to nurse practitioners, as explicitly stated in the answer (use 'yes', 'no', or a short textual phrase; if not stated, use 'unknown').\n"
        "5) practice_sources: All URLs cited in the answer that support the practice authority determination.\n"
        "6) requirement_60_day_text: The specific licensure requirement within the first 60 days of establishing residency (if relocating from another NLC state) as presented in the answer. If not stated, use null.\n"
        "7) requirements_sources: All URLs cited in the answer that support the 60-day requirement.\n"
        "8) board_url: The official website URL of the state’s board of nursing or authoritative licensing source, as presented in the answer. If not provided, use null.\n"
        "Return a JSON object with three fields: 'montana', 'tennessee', and 'rhode_island', each following the above schema. "
        "Only extract URLs explicitly present in the answer."
    )


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def parse_yes_no(val: Optional[str]) -> Optional[bool]:
    """Parse a yes/no/unknown style string to a tri-state boolean."""
    if not val:
        return None
    v = val.strip().lower()
    yes_vals = {"yes", "true", "y", "full", "independent", "member", "is member"}
    no_vals = {"no", "false", "n", "not", "non-member", "restricted", "reduced"}
    if v in yes_vals:
        return True
    if v in no_vals:
        return False
    # Heuristic: detect explicit negation patterns
    if any(neg in v for neg in ["not", "no ", "does not", "isn't", "is not"]):
        return False
    if any(pos in v for pos in ["yes", "is a member", "is member", "full practice", "independent"]):
        return True
    return None


def ensure_list(urls: Optional[List[str]]) -> List[str]:
    return urls if urls else []


def combine_sources(primary: List[str], fallback_url: Optional[str]) -> List[str]:
    """Use primary URLs if present; otherwise fallback to a single board_url if available."""
    if primary:
        return primary
    return [fallback_url] if fallback_url else []


def nlc_claim_for_state(state_name: str, is_member: Optional[bool]) -> str:
    """Construct the NLC membership claim based on the answer's stance if available."""
    if is_member is True:
        return f"{state_name} is an active member of the Nurse Licensure Compact (NLC)."
    if is_member is False:
        return f"{state_name} is not a member of the Nurse Licensure Compact (NLC)."
    # Unknown in answer: still phrase neutrally so the verifier checks support; we default to positive phrasing.
    return f"{state_name} is an active member of the Nurse Licensure Compact (NLC)."


def practice_claim_for_state(state_name: str, has_fpa: Optional[bool]) -> str:
    """Construct the full practice authority claim based on the answer's stance if available."""
    if has_fpa is True:
        return (
            f"{state_name} grants full practice authority to nurse practitioners—independent practice without "
            f"physician supervision or collaborative agreements."
        )
    if has_fpa is False:
        return (
            f"{state_name} does not grant full practice authority to nurse practitioners; physician supervision or "
            f"collaborative agreements are required."
        )
    return (
        f"{state_name} grants full practice authority to nurse practitioners—independent practice without "
        f"physician supervision or collaborative agreements."
    )


def requirement_60d_claim(state_name: str, requirement_text: Optional[str]) -> str:
    """Construct the 60-day licensure requirement claim."""
    # Prefer using the user's statement; otherwise, verify the standard NLC 60-day expectation phrasing.
    if requirement_text and requirement_text.strip():
        return (
            f"For {state_name}, the answer states: '{requirement_text.strip()}'. "
            f"Verify whether the authoritative source confirms this 60-day licensing requirement for nurses relocating from another compact state."
        )
    return (
        f"In {state_name}, nurses relocating from another NLC state must apply for licensure within 60 days of establishing residency."
    )


def reference_url_claim(state_name: str) -> str:
    """Claim to verify that the provided URL is an official or authoritative site."""
    return (
        f"This webpage is the official or authoritative site for {state_name}'s Board of Nursing or the state's licensing "
        f"authority responsible for nursing regulation."
    )


def nlc_additional_instruction(state_name: str) -> str:
    """Additional instruction for NLC verification, with state-specific nuance."""
    base = (
        "Verify Nurse Licensure Compact (NLC) membership for RN multistate licensure. "
        "Rely on official sources (state board, NCSBN). Do not confuse RN NLC with the separate APRN Compact."
    )
    if state_name.lower() == "rhode island":
        return base + " Rhode Island rejoined the RN NLC effective January 1, 2024; explicit mention of this effective date is acceptable."
    return base


def practice_additional_instruction() -> str:
    return (
        "Full practice authority (FPA) means independent nurse practitioner practice without physician supervision or "
        "mandatory collaborative agreements. Verify using state statutes, board of nursing pages, or other official sources."
    )


def requirement_additional_instruction() -> str:
    return (
        "Verify the NLC residency change requirement specific to this state (or the general NLC requirement) that nurses "
        "moving from one compact state to another must apply for a license in the new primary state within 60 days of establishing residency."
    )


def reference_additional_instruction() -> str:
    return (
        "Confirm that the page is the official state board of nursing or an authoritative governmental licensing body "
        "(e.g., state .gov domains, Department of Health, professional regulation). The page should explicitly reference "
        "nursing licensure/board functions for the state."
    )


# --------------------------------------------------------------------------- #
# Verification functions                                                      #
# --------------------------------------------------------------------------- #
async def verify_state(
    evaluator: Evaluator,
    parent_node,
    label_id: str,
    human_desc: str,
    info: Optional[StateInfo],
) -> None:
    """
    Build the verification subtree for a single state:
    - NLC membership
    - Full practice authority
    - 60-day licensure requirement
    - Official board/authoritative URL
    """
    state_node = evaluator.add_parallel(
        id=label_id,
        desc=human_desc,
        parent=parent_node,
        critical=False  # Each state's analysis contributes partial credit independently
    )

    # Basic sources availability check (critical gate to avoid source-free web verifications)
    sources_available = False
    if info:
        sources_available = bool(
            (info.board_url and info.board_url.strip()) or
            ensure_list(info.nlc_sources) or
            ensure_list(info.practice_sources) or
            ensure_list(info.requirements_sources)
        )

    evaluator.add_custom_node(
        result=sources_available,
        id=f"{label_id}_sources_available",
        desc=f"At least one authoritative URL (board or supporting sources) is provided for {info.state_name if info and info.state_name else 'state'}.",
        parent=state_node,
        critical=True
    )

    # Prepare values
    state_name = info.state_name if info and info.state_name else label_id.replace("_Analysis", "").replace("_", " ").title()
    nlc_bool = parse_yes_no(info.nlc_membership if info else None)
    fpa_bool = parse_yes_no(info.practice_authority if info else None)

    # 1) NLC Membership
    nlc_node = evaluator.add_leaf(
        id=f"{label_id.replace('_Analysis','')}_NLC_Membership",
        desc=f"Verify that {state_name} is an active member of the Nurse Licensure Compact, allowing multistate RN license holders to practice",
        parent=state_node,
        critical=True,
    )
    nlc_claim = nlc_claim_for_state(state_name, nlc_bool)
    nlc_sources = combine_sources(ensure_list(info.nlc_sources if info else []), info.board_url if info else None)
    await evaluator.verify(
        claim=nlc_claim,
        node=nlc_node,
        sources=nlc_sources,
        additional_instruction=nlc_additional_instruction(state_name),
    )

    # 2) Full Practice Authority
    fpa_node = evaluator.add_leaf(
        id=f"{label_id.replace('_Analysis','')}_Practice_Authority",
        desc=f"Verify whether {state_name} grants full practice authority to nurse practitioners (independent practice without physician supervision)",
        parent=state_node,
        critical=True,
    )
    fpa_claim = practice_claim_for_state(state_name, fpa_bool)
    fpa_sources = combine_sources(ensure_list(info.practice_sources if info else []), info.board_url if info else None)
    await evaluator.verify(
        claim=fpa_claim,
        node=fpa_node,
        sources=fpa_sources,
        additional_instruction=practice_additional_instruction(),
    )

    # 3) 60-Day Requirement
    req_node = evaluator.add_leaf(
        id=f"{label_id.replace('_Analysis','')}_60Day_Requirement",
        desc=f"Identify the requirement that nurses relocating to {state_name} from another compact state must apply for licensure within 60 days",
        parent=state_node,
        critical=True,
    )
    req_claim = requirement_60d_claim(state_name, info.requirement_60_day_text if info else None)
    req_sources = combine_sources(ensure_list(info.requirements_sources if info else []), info.board_url if info else None)
    await evaluator.verify(
        claim=req_claim,
        node=req_node,
        sources=req_sources,
        additional_instruction=requirement_additional_instruction(),
    )

    # 4) Official Board / Authoritative URL
    ref_node = evaluator.add_leaf(
        id=f"{label_id.replace('_Analysis','')}_Reference_URL",
        desc=f"Provide the official {state_name} Board of Nursing website URL or authoritative source for verification",
        parent=state_node,
        critical=True,
    )
    # If we have a board_url, verify it's official; otherwise, this leaf will be skipped by the sources gate
    # (critical sibling) or fail due to lack of evidence.
    board_url = info.board_url if info else None
    await evaluator.verify(
        claim=reference_url_claim(state_name),
        node=ref_node,
        sources=board_url if board_url else None,
        additional_instruction=reference_additional_instruction(),
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
    Evaluate the relocation analysis for Montana, Tennessee, and Rhode Island.
    """
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # States are evaluated independently
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
    extracted_states = await evaluator.extract(
        prompt=prompt_extract_states(),
        template_class=StatesExtraction,
        extraction_name="states_extraction",
    )

    # Build the root analysis node
    analysis_root = evaluator.add_parallel(
        id="Complete_State_Analysis",
        desc="Comprehensive analysis of three states for nurse practitioner relocation decision",
        parent=root,
        critical=False
    )

    # Verify each state
    await verify_state(
        evaluator=evaluator,
        parent_node=analysis_root,
        label_id="Montana_Analysis",
        human_desc="Complete evaluation of Montana for NLC membership, practice authority, and requirements",
        info=extracted_states.montana
    )

    await verify_state(
        evaluator=evaluator,
        parent_node=analysis_root,
        label_id="Tennessee_Analysis",
        human_desc="Complete evaluation of Tennessee for NLC membership, practice authority, and requirements",
        info=extracted_states.tennessee
    )

    await verify_state(
        evaluator=evaluator,
        parent_node=analysis_root,
        label_id="Rhode_Island_Analysis",
        human_desc="Complete evaluation of Rhode Island for NLC membership, practice authority, and requirements",
        info=extracted_states.rhode_island
    )

    # Return summary
    return evaluator.get_summary()