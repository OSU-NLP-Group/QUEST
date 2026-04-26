import asyncio
import logging
from datetime import datetime
from typing import List, Optional, Dict, Any, Tuple

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy, VerificationNode

# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "us_state_legislatures_2025_accessibility"
TASK_DESCRIPTION = """
Identify at least 3 U.S. states whose state legislatures meet ALL of the following requirements: 
(1) The state has a regular annual legislative session scheduled for 2025 (not a special or extraordinary session) that begins between January 1 and March 31, 2025. 
(2) The state's 2025 legislative session has a clearly defined duration (either a specific day count or a defined end date), and the state publishes an accessible online legislative calendar showing which days the legislature is in session. 
(3) The state capitol building allows public visitors to access legislative sessions or observe proceedings, with documented visitor access policies available online. 
(4) The state provides an online bill tracking system or legislative information system that allows users to search for and follow legislation. 
(5) The state publishes committee meeting schedules online in an accessible format. 
(6) The state does NOT require requesters to be state residents to make public records requests (i.e., the state must NOT be one of the following seven states that have residency requirements: Alabama, Arkansas, Delaware, New Jersey, Kentucky, Tennessee, or Virginia). 

For each qualifying state, provide: the state name, the official state legislature website URL, the 2025 session start date, a URL documenting the session schedule, a URL for the bill tracking system, a URL for committee schedules, and a URL documenting the public records law.
"""

RESIDENCY_REQUIREMENT_STATES = {
    "alabama", "arkansas", "delaware", "new jersey", "kentucky", "tennessee", "virginia"
}

# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class StateEntry(BaseModel):
    state_name: Optional[str] = None
    legislature_url: Optional[str] = None
    session_start_date: Optional[str] = None
    session_schedule_url: Optional[str] = None
    capitol_public_access_url: Optional[str] = None
    bill_tracking_url: Optional[str] = None
    committee_schedules_url: Optional[str] = None
    public_records_law_url: Optional[str] = None


class StatesExtraction(BaseModel):
    states: List[StateEntry] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_states() -> str:
    return """
    From the answer, extract up to the first five (5) distinct U.S. states that the answer proposes as qualifying for the task.
    For each extracted state, return the following fields exactly as presented in the answer:
    - state_name: The name of the U.S. state (e.g., "Texas", "Washington", "New York"). Use the state name, not a city.
    - legislature_url: The official state legislature website URL.
    - session_start_date: The stated start date for the 2025 regular annual legislative session (any textual date format is acceptable; do not invent).
    - session_schedule_url: A URL that documents the 2025 session schedule/calendar.
    - capitol_public_access_url: A URL that documents that the public can access or observe legislative sessions and includes visitor/observer policy details online.
    - bill_tracking_url: A URL for an online legislative information/bill tracking system for that state (where users can search and follow legislation).
    - committee_schedules_url: A URL that publishes committee meeting schedules online.
    - public_records_law_url: A URL documenting the state's public records law (e.g., FOIA/Public Records Act page).

    Rules:
    - Extract only what is explicitly present in the answer. If a field is missing, set it to null.
    - Include only valid, complete URLs (accept both HTTP and HTTPS); do not infer or invent URLs.
    - Preserve the original text formatting for the date string; do not normalize it.
    - Ensure that 'state_name' refers to a U.S. state (if unclear, still extract the provided text; do not invent).
    - If the answer lists more than five states, include only the first five in the order presented.
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _normalize_state_name(name: Optional[str]) -> Optional[str]:
    if not name:
        return None
    return name.strip().lower()


def _is_restricted_residency_state(state_name: Optional[str]) -> bool:
    norm = _normalize_state_name(state_name)
    return (norm in RESIDENCY_REQUIREMENT_STATES) if norm else False


def _distinct_passed_states(candidate_nodes: List[Tuple[VerificationNode, Optional[str]]]) -> List[str]:
    """
    Compute which candidate state nodes fully passed (score == 1.0) and return distinct state names (case-insensitive).
    """
    seen = set()
    passed = []
    for node, name in candidate_nodes:
        try:
            score = node.compute_score(mutate=False)
        except Exception:
            score = 0.0
        if score == 1.0 and name:
            key = name.strip().lower()
            if key not in seen:
                seen.add(key)
                passed.append(name.strip())
    return passed


# --------------------------------------------------------------------------- #
# Verification logic per state                                                #
# --------------------------------------------------------------------------- #
async def verify_candidate_state(
    evaluator: Evaluator,
    parent_node: VerificationNode,
    idx: int,
    st: StateEntry,
) -> VerificationNode:
    """
    Build verification checks for one candidate state. All leaf nodes are critical under the candidate node,
    matching the rubric's per-state "must satisfy all constraints" requirement.
    """
    cand = evaluator.add_parallel(
        id=f"candidate_state_{idx+1}",
        desc=f"Candidate State {idx+1}: evaluate whether it satisfies all constraints and includes all required fields.",
        parent=parent_node,
        critical=False
    )

    # 1) State name is a U.S. state
    name_leaf = evaluator.add_leaf(
        id=f"state_{idx+1}_name_us_state",
        desc="Provides a U.S. state name.",
        parent=cand,
        critical=True
    )
    nm = st.state_name or ""
    await evaluator.verify(
        claim=f"'{nm}' is the name of a U.S. state.",
        node=name_leaf,
        additional_instruction="Treat only the 50 U.S. states as valid. Do not count territories, D.C., or regions. If the name is missing or blank, mark incorrect."
    )

    # 2) Official legislature website URL
    leg_leaf = evaluator.add_leaf(
        id=f"state_{idx+1}_official_legislature_website_url",
        desc="Provides the official state legislature website URL.",
        parent=cand,
        critical=True
    )
    await evaluator.verify(
        claim=f"This URL is the official website or primary official portal for the {nm} state legislature.",
        node=leg_leaf,
        sources=st.legislature_url or None,
        additional_instruction="Look for explicit signals that this site is operated by the state legislature (official branding, domain, about statements). Reject third-party or unofficial aggregators."
    )

    # 3) 2025 Regular annual session start date in range (Jan 1–Mar 31, 2025)
    start_leaf = evaluator.add_leaf(
        id=f"state_{idx+1}_2025_regular_session_start_in_range",
        desc="Provides the start date for a 2025 regular annual (not special) session that begins between 2025-01-01 and 2025-03-31.",
        parent=cand,
        critical=True
    )
    start_str = st.session_start_date or ""
    await evaluator.verify(
        claim=(
            f"The 2025 regular (not special or extraordinary) session of the {nm} legislature begins on '{start_str}', "
            f"and that start date is between January 1, 2025 and March 31, 2025 inclusive."
        ),
        node=start_leaf,
        sources=st.session_schedule_url or None,
        additional_instruction=(
            "Use the provided session schedule/calendar page to confirm the start date and that it is a regular session (not special/extraordinary). "
            "If the date falls outside the specified range, or if the page suggests it's a special session, mark incorrect. "
            "Minor date formatting differences are fine; focus on the semantics."
        )
    )

    # 4) Session schedule URL includes duration and in-session days
    schedule_leaf = evaluator.add_leaf(
        id=f"state_{idx+1}_session_schedule_has_duration_and_calendar",
        desc="Provides a URL documenting the 2025 session schedule/calendar including duration (end date or day count) and which days are in session.",
        parent=cand,
        critical=True
    )
    await evaluator.verify(
        claim=(
            f"On the provided 2025 session schedule page for {nm}, the session's duration is clearly defined "
            f"(either an explicit end date or a specified day count), and the page also shows which days the legislature is in session."
        ),
        node=schedule_leaf,
        sources=st.session_schedule_url or None,
        additional_instruction=(
            "Look for a defined session end date or explicit total day count, and also a calendar or list indicating in-session days. "
            "If either element is missing, mark incorrect."
        )
    )

    # 5) Capitol public access policies documented online
    access_leaf = evaluator.add_leaf(
        id=f"state_{idx+1}_capitol_public_access_policies",
        desc="State capitol allows public visitors to access/observe legislative sessions or proceedings, with visitor policies documented online.",
        parent=cand,
        critical=True
    )
    await evaluator.verify(
        claim=(
            f"The provided page documents that members of the public may visit the {nm} State Capitol to observe legislative sessions or proceedings, "
            f"and it includes visitor/observer policy details (e.g., hours, security, gallery access rules)."
        ),
        node=access_leaf,
        sources=st.capitol_public_access_url or None,
        additional_instruction=(
            "Accept pages from the legislature, capitol visitor services, or an official state agency. "
            "There should be explicit mention of public observation or visitor access to legislative sessions or galleries."
        )
    )

    # 6) Bill tracking system URL and functionality
    bill_leaf = evaluator.add_leaf(
        id=f"state_{idx+1}_bill_tracking_url_and_functionality",
        desc="Provides a URL for an online bill tracking/legislative information system with search/follow functionality.",
        parent=cand,
        critical=True
    )
    await evaluator.verify(
        claim=(
            f"The provided page is {nm}'s online legislative information/bill tracking system that allows users to search for and follow legislation "
            f"(e.g., search by bill number/keyword and view bill status/history)."
        ),
        node=bill_leaf,
        sources=st.bill_tracking_url or None,
        additional_instruction="Reject generic news sites or static PDFs. Accept official LIS portals or equivalent systems clearly supporting bill search and tracking."
    )

    # 7) Committee meeting schedules URL
    cmte_leaf = evaluator.add_leaf(
        id=f"state_{idx+1}_committee_meeting_schedules_url",
        desc="Provides a URL where committee meeting schedules are published online in an accessible format.",
        parent=cand,
        critical=True
    )
    await evaluator.verify(
        claim=(
            f"The provided page publishes committee meeting schedules for the {nm} legislature in an accessible format (e.g., listings, calendar, agenda pages)."
        ),
        node=cmte_leaf,
        sources=st.committee_schedules_url or None,
        additional_instruction="Look for schedule listings, calendars, or agendas that are clearly about committee meetings. PDFs are acceptable if clearly presented as schedules."
    )

    # 8) Public records law URL
    pr_law_leaf = evaluator.add_leaf(
        id=f"state_{idx+1}_public_records_law_url",
        desc="Provides a URL documenting the state's public records law.",
        parent=cand,
        critical=True
    )
    await evaluator.verify(
        claim=(
            f"The provided page documents {nm}'s public records law (e.g., FOIA/Public Records Act or equivalent), describing rights or process to request records."
        ),
        node=pr_law_leaf,
        sources=st.public_records_law_url or None,
        additional_instruction="Accept official statute pages, attorney general guidance, or official state portals discussing the public records law."
    )

    # 9) No public records residency requirement: not one of the seven listed states
    residency_leaf = evaluator.add_custom_node(
        result=(not _is_restricted_residency_state(st.state_name)),
        id=f"state_{idx+1}_no_pr_residency_requirement",
        desc="State is NOT one of the seven states with a residency requirement for public records (AL, AR, DE, NJ, KY, TN, VA).",
        parent=cand,
        critical=True
    )

    return cand


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
    model: str = "o4-mini"
) -> Dict:
    """
    Evaluate an answer for the 'U.S. state legislatures 2025 accessibility' task.
    """
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

    # 1) Extract candidate states from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_states(),
        template_class=StatesExtraction,
        extraction_name="states_extraction"
    )

    # Normalize and cap to at most 5 candidates
    candidates: List[StateEntry] = list(extracted.states or [])[:5]

    # 2) Build per-state verification subtrees
    candidate_nodes: List[Tuple[VerificationNode, Optional[str]]] = []
    for i, st in enumerate(candidates):
        node = await verify_candidate_state(evaluator, root, i, st)
        candidate_nodes.append((node, st.state_name))

    # 3) Compute how many distinct states fully qualified (all critical checks passed)
    passed_states = _distinct_passed_states(candidate_nodes)
    passed_count = len(passed_states)

    # Record diagnostic info
    evaluator.add_custom_info(
        info={
            "passed_states": passed_states,
            "passed_count": passed_count,
            "residency_restriction_list": sorted(list(RESIDENCY_REQUIREMENT_STATES))
        },
        info_type="diagnostics",
        info_name="qualification_summary"
    )

    # 4) Root-level critical gate: At least 3 distinct states qualify
    evaluator.add_custom_node(
        result=(passed_count >= 3),
        id="at_least_3_distinct_states_qualify",
        desc="At least 3 DISTINCT provided states satisfy ALL per-state constraints and include ALL required fields (state name, official legislature website URL, 2025 session start date, URL documenting session schedule, URL for bill tracking, URL for committee schedules, URL documenting public records law).",
        parent=root,
        critical=True
    )

    # 5) Return standardized summary
    return evaluator.get_summary()