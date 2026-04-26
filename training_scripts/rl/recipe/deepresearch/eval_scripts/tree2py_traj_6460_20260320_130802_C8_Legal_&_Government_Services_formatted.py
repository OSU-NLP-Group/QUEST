import asyncio
import logging
from typing import Any, List, Optional, Dict

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys

# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "us_states_2026_session_access"
TASK_DESCRIPTION = (
    "Identify four U.S. states where the 2026 regular legislative session begins in January or February 2026. "
    "For each of these four states, provide the following information: "
    "(1) The state name, "
    "(2) The official start date and adjournment/end date of the 2026 regular legislative session, "
    "(3) A direct URL to an official state government source (such as the state legislature's official website) that confirms these session dates, "
    "(4) Whether the state requires public records requesters to be state residents (answer 'Yes' if residency is required, 'No' if any person can request), "
    "(5) Confirmation that state legislative committee hearings are open to public attendance, and "
    "(6) The official URL of the state legislature's website where the public can track bill status and legislative information."
)

# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class StateInfo(BaseModel):
    state_name: Optional[str] = None
    session_start_date: Optional[str] = None          # e.g., "January 8, 2026" (free text ok)
    session_end_date: Optional[str] = None            # e.g., "May 15, 2026" (free text ok)
    session_dates_url: Optional[str] = None           # Official gov/legislature URL confirming the dates

    public_records_residency: Optional[str] = None    # "Yes" or "No"
    public_records_source_url: Optional[str] = None   # Official source confirming residency requirement (or not)

    committee_hearings_public: Optional[str] = None   # "Yes" or "No"
    committee_access_source_url: Optional[str] = None # Official source confirming public access

    bill_tracking_url: Optional[str] = None           # Official legislature bill-tracking URL


class StatesExtraction(BaseModel):
    states: List[StateInfo] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_states() -> str:
    return """
    Extract up to FOUR states from the answer that meet ALL of the following:
    – The 2026 REGULAR legislative session begins in January or February 2026.

    For each extracted state, provide a single object with these fields:
    - state_name: the state's full name (e.g., "Utah")
    - session_start_date: the official start date for the 2026 regular legislative session (free-text as shown in the answer)
    - session_end_date: the official end/adjournment date for the 2026 regular legislative session (free-text as shown in the answer)
    - session_dates_url: a single DIRECT URL to an OFFICIAL state government or legislature webpage that explicitly lists or confirms the 2026 regular session dates
    - public_records_residency: "Yes" if the state restricts public records requests to residents/citizens of that state; "No" if any person can request (normalize strictly to "Yes" or "No")
    - public_records_source_url: a single DIRECT official government or statutory URL supporting the residency rule ("Yes") or lack of residency requirement ("No")
    - committee_hearings_public: "Yes" if legislative committee hearings are open to public attendance; otherwise "No" (normalize strictly to "Yes" or "No")
    - committee_access_source_url: a single DIRECT official legislature/government URL that states committee hearings are open to the public (or indicates access policy)
    - bill_tracking_url: a single DIRECT OFFICIAL state legislature site that allows the public to track bills/legislation (not third-party aggregators)

    Requirements and notes:
    - Only extract URLs that are explicitly present in the answer text. Do not invent URLs.
    - Prefer official .gov or official legislature domains (e.g., legis.state.xx.us, leg.state.xx.us, legislature.xx.gov).
    - If any required value is missing from the answer, set that field to null.
    - If more than four states are present, include only the first four in the order they appear.
    - If the answer provides fewer than four states, include as many as are present.
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def yn_normalized(value: Optional[str]) -> str:
    if not value:
        return ""
    v = value.strip().lower()
    if v in {"yes", "y", "true"}:
        return "Yes"
    if v in {"no", "n", "false"}:
        return "No"
    return value.strip()


def safe(s: Optional[str], fallback: str = "") -> str:
    return s if s is not None else fallback


# --------------------------------------------------------------------------- #
# Verification per-state                                                      #
# --------------------------------------------------------------------------- #
async def verify_single_state(evaluator: Evaluator, parent_node, state: StateInfo, index: int) -> None:
    """
    Build the verification subtree and run checks for one state entry.
    The structure creates logical groups to avoid unintended cross-dependencies, while keeping each rubric leaf as a leaf node.
    """
    display_idx = index + 1
    state_label = safe(state.state_name, "the state")

    # Parent node for this state (non-critical; allows partial credit across states)
    state_node = evaluator.add_parallel(
        id=f"state_{display_idx}",
        desc=f"State #{display_idx} meeting criteria with all required information",
        parent=parent_node,
        critical=False
    )

    # Group 1: Session info (sequential: URL -> dates -> identification)
    session_group = evaluator.add_sequential(
        id=f"state_{display_idx}_session_group",
        desc=f"State #{display_idx} 2026 regular session info verification",
        parent=state_node,
        critical=False
    )

    # Leaf A: session_dates_url (official source confirming session dates)
    leaf_session_url = evaluator.add_leaf(
        id=f"state_{display_idx}_session_dates_url",
        desc="Provide official government source URL confirming the session dates",
        parent=session_group,
        critical=True
    )
    session_url_claim = (
        f"This webpage is an OFFICIAL state government or legislature source for {state_label} and it explicitly lists "
        f"or confirms the 2026 regular legislative session dates."
    )
    await evaluator.verify(
        claim=session_url_claim,
        node=leaf_session_url,
        sources=safe(state.session_dates_url, None),
        additional_instruction=(
            "Accept state legislature or other official state .gov domains. The content should clearly show the 2026 "
            "regular session calendar/dates; it's okay if phrased as 'Regular Session' or equivalent (e.g., 'General Session'). "
            "If the page is not official or does not mention the 2026 regular session dates, mark as not supported."
        )
    )

    # Leaf B: session_dates (start/end correctness on the official page)
    leaf_session_dates = evaluator.add_leaf(
        id=f"state_{display_idx}_session_dates",
        desc="Provide the official start date and end date for the state's 2026 regular legislative session",
        parent=session_group,
        critical=True
    )
    start_txt = safe(state.session_start_date, "")
    end_txt = safe(state.session_end_date, "")
    session_dates_claim = (
        f"According to the cited official source, {state_label}'s 2026 REGULAR legislative session starts on '{start_txt}' "
        f"and ends/adjourns on '{end_txt}'."
    )
    await evaluator.verify(
        claim=session_dates_claim,
        node=leaf_session_dates,
        sources=safe(state.session_dates_url, None),
        additional_instruction=(
            "Verify that BOTH the start and end (adjournment) dates for the 2026 regular session match the claim. "
            "Allow reasonable variants like 'adjourn sine die' for the end. Minor date-format differences are fine "
            "as long as the dates match."
        )
    )

    # Leaf C: identification (start month is Jan or Feb 2026; regular session)
    leaf_identification = evaluator.add_leaf(
        id=f"state_{display_idx}_identification",
        desc="State has a 2026 regular legislative session that begins in January or February 2026",
        parent=session_group,
        critical=True
    )
    identification_claim = (
        f"For {state_label}, the 2026 REGULAR legislative session begins on '{start_txt}', which is in January or February 2026."
    )
    await evaluator.verify(
        claim=identification_claim,
        node=leaf_identification,
        sources=safe(state.session_dates_url, None),
        additional_instruction=(
            "Confirm BOTH that it is a regular (not special) session AND that the start date month is January or February 2026. "
            "Consider common phrasing like 'General Session' as equivalent to 'Regular Session'. "
            "If the start date is not in Jan/Feb 2026, or the page only lists special sessions, mark as not supported."
        )
    )

    # Group 2: Public access policies (parallel: FOI residency + committee access)
    access_group = evaluator.add_parallel(
        id=f"state_{display_idx}_public_access_group",
        desc=f"State #{display_idx} public access requirements verification",
        parent=state_node,
        critical=False
    )

    # Leaf D: public records residency requirement
    leaf_residency = evaluator.add_leaf(
        id=f"state_{display_idx}_public_records_residency",
        desc="Specify whether the state requires public records requesters to be state residents (yes or no)",
        parent=access_group,
        critical=True
    )
    residency_val = yn_normalized(state.public_records_residency)
    if residency_val == "Yes":
        residency_claim = (
            f"{state_label} restricts public records requests to its residents/citizens (a residency requirement exists)."
        )
    else:
        residency_claim = (
            f"{state_label} does NOT restrict public records requests to state residents; any person may request."
        )
    await evaluator.verify(
        claim=residency_claim,
        node=leaf_residency,
        sources=safe(state.public_records_source_url, None),
        additional_instruction=(
            "Look for statutory text or official policy from the state indicating either (a) residency/citizenship is required "
            "for requests, or (b) requests are open to 'any person' or the general public. Phrases like 'citizens of [state]' "
            "or 'residents' indicate a residency requirement. If ambiguous or non-official, mark as not supported."
        )
    )

    # Leaf E: committee hearings public access
    leaf_committee = evaluator.add_leaf(
        id=f"state_{display_idx}_committee_access",
        desc="Confirm that state legislative committee hearings are open to public attendance",
        parent=access_group,
        critical=True
    )
    committee_claim = (
        f"In {state_label}, state legislative committee hearings are open to public attendance."
    )
    await evaluator.verify(
        claim=committee_claim,
        node=leaf_committee,
        sources=safe(state.committee_access_source_url, None),
        additional_instruction=(
            "Accept language such as 'open to the public', 'public may attend', 'public welcome', or similar. "
            "It's acceptable if the policy allows public attendance subject to space/security rules. "
            "Reject non-official pages or pages that do not address public attendance."
        )
    )

    # Group 3: Bill tracking URL (parallel with single leaf)
    tracking_group = evaluator.add_parallel(
        id=f"state_{display_idx}_bill_tracking_group",
        desc=f"State #{display_idx} bill tracking site verification",
        parent=state_node,
        critical=False
    )

    # Leaf F: bill_tracking_url (official tracking site)
    leaf_tracking = evaluator.add_leaf(
        id=f"state_{display_idx}_bill_tracking_url",
        desc="Provide the official state legislature website URL for public bill tracking",
        parent=tracking_group,
        critical=True
    )
    tracking_claim = (
        f"This webpage is the OFFICIAL state legislature site for {state_label} where the public can search/track bills and view legislative information."
    )
    await evaluator.verify(
        claim=tracking_claim,
        node=leaf_tracking,
        sources=safe(state.bill_tracking_url, None),
        additional_instruction=(
            "Confirm the site is an official legislature/government property (not a third-party aggregator). "
            "The page should present bill lookup, bill status, or measure history features accessible to the public."
        )
    )


# --------------------------------------------------------------------------- #
# Main evaluation function                                                    #
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
    Evaluate an answer for the '2026 state legislative sessions and public access' task.
    """
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,   # States are independent
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

    # Extract up to 4 qualifying states from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_states(),
        template_class=StatesExtraction,
        extraction_name="states_extraction"
    )

    # Keep at most first 4 states; pad if fewer
    states: List[StateInfo] = list(extracted.states[:4])
    while len(states) < 4:
        states.append(StateInfo())

    # Build verification subtrees for each of the four states
    # Root node should be non-critical to avoid critical child constraint at root level
    for i in range(4):
        await verify_single_state(evaluator, root, states[i], i)

    # Optional: record custom meta info
    evaluator.add_custom_info(
        info={
            "extracted_states_count": len(extracted.states),
            "used_states": min(4, len(extracted.states))
        },
        info_type="extraction_stats",
        info_name="extraction_statistics"
    )

    return evaluator.get_summary()