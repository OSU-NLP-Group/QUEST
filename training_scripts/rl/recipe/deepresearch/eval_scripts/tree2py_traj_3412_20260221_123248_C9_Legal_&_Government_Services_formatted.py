import asyncio
import logging
from typing import Optional, List, Dict, Any
from urllib.parse import urlparse

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy

# -----------------------------------------------------------------------------
# Task Constants
# -----------------------------------------------------------------------------
TASK_ID = "jan2026_state_sessions"
TASK_DESCRIPTION = """Identify four U.S. states whose 2026 legislative sessions convened in January 2026. For each of these four states, provide the following information from official state legislature websites:

1. Session Information:
   - The official start date of the 2026 legislative session
   - The official end date or estimated adjournment date of the 2026 session
   - The deadline for introducing bills in at least one chamber (House or Senate)
   - A URL reference to the official state legislature page containing session information

2. Committee Information:
   - The official name of one standing committee in the state legislature
   - The name of that committee's chair or co-chairs
   - The committee's regular meeting schedule or next scheduled meeting date
   - A URL reference to the official committee information page

3. Chamber Leadership:
   - The official title of one chamber's leader (e.g., Speaker of the House or Senate President)
   - The name of the current leader holding that position
   - Official contact information for that leader (phone number, email address, or office address)
   - A URL reference to the official leadership information page

4. Legislative Calendar:
   - The pattern or schedule of regular session days (e.g., which days of the week the legislature typically meets)
   - A URL reference to the official legislative calendar

All information must be current, verifiable, and obtained from official state government or state legislature websites.
"""

# -----------------------------------------------------------------------------
# Data Models for Extraction
# -----------------------------------------------------------------------------
class SessionInfo(BaseModel):
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    bill_introduction_deadline: Optional[str] = None
    url: Optional[str] = None


class CommitteeInfo(BaseModel):
    name: Optional[str] = None
    chair: Optional[str] = None
    meeting_schedule: Optional[str] = None
    url: Optional[str] = None


class LeadershipInfo(BaseModel):
    title: Optional[str] = None
    name: Optional[str] = None
    contact: Optional[str] = None
    url: Optional[str] = None


class CalendarInfo(BaseModel):
    regular_session_days: Optional[str] = None
    url: Optional[str] = None


class StateInfo(BaseModel):
    state_name: Optional[str] = None
    session: Optional[SessionInfo] = None
    committee: Optional[CommitteeInfo] = None
    leadership: Optional[LeadershipInfo] = None
    calendar: Optional[CalendarInfo] = None


class StatesExtraction(BaseModel):
    states: List[StateInfo] = Field(default_factory=list)


# -----------------------------------------------------------------------------
# Extraction Prompt
# -----------------------------------------------------------------------------
def prompt_extract_states() -> str:
    return """
Extract up to four U.S. states mentioned in the answer for which the 2026 legislative session convened in January 2026. For each of the selected states (take the first four in the order they appear in the answer), extract exactly what the answer provides for the following structured fields:

- state_name
- session:
  - start_date (string as written in the answer)
  - end_date (string as written)
  - bill_introduction_deadline (string as written for at least one chamber)
  - url (the URL to the official session information page)
- committee:
  - name (official standing committee name)
  - chair (chair or co-chairs as written)
  - meeting_schedule (regular meeting schedule or next scheduled meeting date)
  - url (URL to the official committee page)
- leadership:
  - title (e.g., "Speaker of the House" or "Senate President")
  - name (current leader’s name)
  - contact (an official contact detail: phone/email/office address exactly as written)
  - url (URL to the official leadership information page)
- calendar:
  - regular_session_days (the stated pattern of regular session days)
  - url (URL to the official legislative calendar)

STRICT RULES:
- Only extract information explicitly present in the answer text.
- All URLs must be the exact URLs shown in the answer (do not invent or modify).
- If any requested field is missing in the answer, return null for that field (or empty string if the answer shows an empty value).
- Do not add more than four states. If fewer than four are present, return only those available.
"""


# -----------------------------------------------------------------------------
# Utility: Basic Official URL Heuristic (for presence checks when needed)
# -----------------------------------------------------------------------------
def is_official_url(url: Optional[str]) -> bool:
    if not url or not isinstance(url, str):
        return False
    try:
        parsed = urlparse(url.strip())
        if not parsed.scheme or not parsed.netloc:
            return False
        host = parsed.netloc.lower()
        # Heuristics for official state legislature/government domains
        if ".gov" in host:
            return True
        if ".us" in host and (
            "leg" in host or "legis" in host or "legislature" in host or "assembly" in host or
            "senate" in host or "house" in host or "state" in host or "capitol" in host
        ):
            return True
        return False
    except Exception:
        return False


# -----------------------------------------------------------------------------
# Verification Builder per State
# -----------------------------------------------------------------------------
async def verify_state_block(evaluator: Evaluator, parent, state: StateInfo, idx: int) -> None:
    state_label = state.state_name or f"State #{idx + 1}"
    session = state.session or SessionInfo()
    committee = state.committee or CommitteeInfo()
    leadership = state.leadership or LeadershipInfo()
    calendar = state.calendar or CalendarInfo()

    # State node (non-critical to allow partial credit across states)
    state_node = evaluator.add_parallel(
        id=f"state_{idx+1}",
        desc=f"{['First','Second','Third','Fourth'][idx] if idx < 4 else f'State {idx+1}'} state with complete legislative information",
        parent=parent,
        critical=False
    )

    # 1) State selection (critical)
    selection_node = evaluator.add_parallel(
        id=f"state_{idx+1}_selection",
        desc="State has an active 2026 legislative session that convened in January 2026",
        parent=state_node,
        critical=True
    )

    # 1.a) Session convened in January 2026 (critical leaf)
    convened_leaf = evaluator.add_leaf(
        id=f"state_{idx+1}_session_convened_january",
        desc="Legislative session convened in January 2026",
        parent=selection_node,
        critical=True
    )
    convened_claim = (
        f"According to the provided session information page, the 2026 legislative session of {state_label} "
        f"convened in January 2026 (i.e., the official start date is between January 1 and January 31, 2026)."
    )
    await evaluator.verify(
        claim=convened_claim,
        node=convened_leaf,
        sources=session.url,
        additional_instruction=(
            "Use the session information page to confirm the 2026 session's convene/start date is in January 2026. "
            "If the URL is missing or not accessible, or if the page does not provide a clear 2026 start date in January, mark as not supported."
        )
    )

    # 1.b) State selection URL presence and officialness (critical leaf)
    # We implement as a custom node result based on simple heuristics to ensure presence; content support verified by other leaves.
    selection_url_ok = is_official_url(session.url)
    evaluator.add_custom_node(
        result=selection_url_ok,
        id=f"state_{idx+1}_state_selection_url",
        desc="URL reference for state legislative session information",
        parent=selection_node,
        critical=True
    )

    # 2) Session information (critical)
    session_node = evaluator.add_parallel(
        id=f"state_{idx+1}_session_information",
        desc="2026 legislative session details",
        parent=state_node,
        critical=True
    )

    # 2.a) Session start date (critical leaf)
    start_leaf = evaluator.add_leaf(
        id=f"state_{idx+1}_session_start_date",
        desc="Official start date of the 2026 legislative session",
        parent=session_node,
        critical=True
    )
    start_claim = (
        f"The official start date of the 2026 legislative session in {state_label} is '{session.start_date}'."
    )
    await evaluator.verify(
        claim=start_claim,
        node=start_leaf,
        sources=session.url,
        additional_instruction=(
            "Verify the 2026 session start date exactly as stated. Minor formatting differences are acceptable. "
            "If the URL is missing/invalid or the page does not clearly indicate the 2026 start date, mark as not supported."
        )
    )

    # 2.b) Session end/adjournment date (critical leaf)
    end_leaf = evaluator.add_leaf(
        id=f"state_{idx+1}_session_end_date",
        desc="Official end date or estimated adjournment date of the 2026 session",
        parent=session_node,
        critical=True
    )
    end_claim = (
        f"The official end date or adjournment date of the 2026 legislative session in {state_label} is '{session.end_date}'."
    )
    await evaluator.verify(
        claim=end_claim,
        node=end_leaf,
        sources=session.url,
        additional_instruction=(
            "Verify the 2026 end/adjournment date as stated on the page. If multiple adjournments exist, "
            "a clearly labeled final or estimated adjournment date is acceptable. "
            "If the URL is missing/invalid or the page does not provide it, mark as not supported."
        )
    )

    # 2.c) Bill introduction deadline (critical leaf)
    bill_deadline_leaf = evaluator.add_leaf(
        id=f"state_{idx+1}_bill_introduction_deadline",
        desc="Deadline for introducing bills in at least one chamber",
        parent=session_node,
        critical=True
    )
    bill_deadline_claim = (
        f"For the 2026 session in {state_label}, the deadline for introducing bills in at least one chamber "
        f"(House or Senate) is '{session.bill_introduction_deadline}'."
    )
    await evaluator.verify(
        claim=bill_deadline_claim,
        node=bill_deadline_leaf,
        sources=session.url,
        additional_instruction=(
            "Confirm that the page lists a bill introduction deadline for at least one chamber for the 2026 session. "
            "If the page does not provide such a deadline or the URL is missing/invalid, mark as not supported."
        )
    )

    # 2.d) Session info URL (critical leaf) - verify officialness and that it provides 2026 session info
    session_url_leaf = evaluator.add_leaf(
        id=f"state_{idx+1}_session_info_url",
        desc="URL reference from official state legislature website",
        parent=session_node,
        critical=True
    )
    session_url_claim = (
        f"This webpage is an official state legislature or government site that provides 2026 session information for {state_label} "
        f"(such as session dates, calendars, or deadlines)."
    )
    await evaluator.verify(
        claim=session_url_claim,
        node=session_url_leaf,
        sources=session.url,
        additional_instruction=(
            "Accept .gov or state .us legislative domains (e.g., ncleg.gov, le.utah.gov, legis.state.xx.us, etc.) as official. "
            "The page should clearly pertain to 2026 session information. If the URL is missing/invalid or not official, mark as not supported."
        )
    )

    # 3) Committee information (critical)
    committee_node = evaluator.add_parallel(
        id=f"state_{idx+1}_committee_information",
        desc="Information about one standing committee in the state legislature",
        parent=state_node,
        critical=True
    )

    # 3.a) Committee name
    committee_name_leaf = evaluator.add_leaf(
        id=f"state_{idx+1}_committee_name",
        desc="Official name of a standing committee",
        parent=committee_node,
        critical=True
    )
    committee_name_claim = f"The official committee name is '{committee.name}'."
    await evaluator.verify(
        claim=committee_name_claim,
        node=committee_name_leaf,
        sources=committee.url,
        additional_instruction=(
            "Verify the committee's official name as shown on the official committee page. "
            "If the URL is missing/invalid or the committee name cannot be found, mark as not supported."
        )
    )

    # 3.b) Committee chair
    committee_chair_leaf = evaluator.add_leaf(
        id=f"state_{idx+1}_committee_chair",
        desc="Name of the committee chair or co-chairs",
        parent=committee_node,
        critical=True
    )
    committee_chair_claim = (
        f"The chair or co-chairs of the '{committee.name}' committee is/are '{committee.chair}'."
    )
    await evaluator.verify(
        claim=committee_chair_claim,
        node=committee_chair_leaf,
        sources=committee.url,
        additional_instruction=(
            "Verify the committee chair or co-chairs as listed on the official committee page. "
            "Minor formatting variations in names are acceptable. If the URL is missing/invalid or "
            "the chair info is not present, mark as not supported."
        )
    )

    # 3.c) Committee meeting schedule
    committee_sched_leaf = evaluator.add_leaf(
        id=f"state_{idx+1}_committee_meeting_schedule",
        desc="Regular meeting schedule or next scheduled meeting date",
        parent=committee_node,
        critical=True
    )
    committee_sched_claim = (
        f"The committee's regular meeting schedule or next scheduled meeting is '{committee.meeting_schedule}'."
    )
    await evaluator.verify(
        claim=committee_sched_claim,
        node=committee_sched_leaf,
        sources=committee.url,
        additional_instruction=(
            "Verify that the committee page states a regular meeting schedule (e.g., 'Mondays at 9 AM') "
            "or a next scheduled meeting date. If not present or URL is missing/invalid, mark as not supported."
        )
    )

    # 3.d) Committee info URL (official page)
    committee_url_leaf = evaluator.add_leaf(
        id=f"state_{idx+1}_committee_info_url",
        desc="URL reference from official state legislature website",
        parent=committee_node,
        critical=True
    )
    committee_url_claim = (
        f"This webpage is an official state legislature or government page for the '{committee.name}' committee."
    )
    await evaluator.verify(
        claim=committee_url_claim,
        node=committee_url_leaf,
        sources=committee.url,
        additional_instruction=(
            "Accept .gov or state .us legislative domains as official. "
            "If the URL is missing/invalid or not clearly an official committee page, mark as not supported."
        )
    )

    # 4) Chamber leadership (critical)
    leadership_node = evaluator.add_parallel(
        id=f"state_{idx+1}_chamber_leadership",
        desc="Leadership information for one legislative chamber",
        parent=state_node,
        critical=True
    )

    # 4.a) Leadership title
    leadership_title_leaf = evaluator.add_leaf(
        id=f"state_{idx+1}_leadership_title",
        desc="Official title of the chamber leader (e.g., Speaker of the House, Senate President)",
        parent=leadership_node,
        critical=True
    )
    leadership_title_claim = f"The leadership title is '{leadership.title}'."
    await evaluator.verify(
        claim=leadership_title_claim,
        node=leadership_title_leaf,
        sources=leadership.url,
        additional_instruction=(
            "Verify the official leadership title on the leadership page. If URL is missing/invalid or title not present, mark as not supported."
        )
    )

    # 4.b) Leadership name
    leadership_name_leaf = evaluator.add_leaf(
        id=f"state_{idx+1}_leadership_name",
        desc="Name of the current chamber leader",
        parent=leadership_node,
        critical=True
    )
    leadership_name_claim = f"The current {leadership.title} is '{leadership.name}'."
    await evaluator.verify(
        claim=leadership_name_claim,
        node=leadership_name_leaf,
        sources=leadership.url,
        additional_instruction=(
            "Verify the current holder of the leadership position on the official page. "
            "Allow minor name formatting differences. If URL is missing/invalid or name not present, mark as not supported."
        )
    )

    # 4.c) Leadership contact
    leadership_contact_leaf = evaluator.add_leaf(
        id=f"state_{idx+1}_leadership_contact",
        desc="Official contact information (phone, email, or office address)",
        parent=leadership_node,
        critical=True
    )
    leadership_contact_claim = (
        f"The official contact information for {leadership.title} {leadership.name} includes '{leadership.contact}'."
    )
    await evaluator.verify(
        claim=leadership_contact_claim,
        node=leadership_contact_leaf,
        sources=leadership.url,
        additional_instruction=(
            "Verify that at least one official contact detail (phone/email/office address) is provided for the leader. "
            "If URL is missing/invalid or contact details not present, mark as not supported."
        )
    )

    # 4.d) Leadership info URL (official page)
    leadership_url_leaf = evaluator.add_leaf(
        id=f"state_{idx+1}_leadership_info_url",
        desc="URL reference from official state legislature or government website",
        parent=leadership_node,
        critical=True
    )
    leadership_url_claim = (
        f"This webpage is an official state legislature or government page that provides leadership information for '{leadership.title}'."
    )
    await evaluator.verify(
        claim=leadership_url_claim,
        node=leadership_url_leaf,
        sources=leadership.url,
        additional_instruction=(
            "Accept .gov or state .us legislative domains as official. "
            "If the URL is missing/invalid or not an official leadership page, mark as not supported."
        )
    )

    # 5) Legislative calendar (critical)
    calendar_node = evaluator.add_parallel(
        id=f"state_{idx+1}_legislative_calendar",
        desc="Official legislative calendar information for 2026 session",
        parent=state_node,
        critical=True
    )

    # 5.a) Regular session days
    calendar_days_leaf = evaluator.add_leaf(
        id=f"state_{idx+1}_regular_session_days",
        desc="Pattern or schedule of regular session days (e.g., which days of the week the legislature meets)",
        parent=calendar_node,
        critical=True
    )
    calendar_days_claim = (
        f"The pattern/schedule of regular session days for the 2026 session is '{calendar.regular_session_days}'."
    )
    await evaluator.verify(
        claim=calendar_days_claim,
        node=calendar_days_leaf,
        sources=calendar.url,
        additional_instruction=(
            "Verify that the legislative calendar explicitly indicates a pattern of session days (e.g., days of week) "
            "for 2026. If URL is missing/invalid or the pattern is not present, mark as not supported."
        )
    )

    # 5.b) Calendar URL (official page)
    calendar_url_leaf = evaluator.add_leaf(
        id=f"state_{idx+1}_calendar_url",
        desc="URL reference to official legislative calendar",
        parent=calendar_node,
        critical=True
    )
    calendar_url_claim = (
        f"This webpage is an official state legislature or government legislative calendar for the 2026 session."
    )
    await evaluator.verify(
        claim=calendar_url_claim,
        node=calendar_url_leaf,
        sources=calendar.url,
        additional_instruction=(
            "Accept .gov or state .us legislative domains as official. "
            "If the URL is missing/invalid or not clearly an official 2026 legislative calendar, mark as not supported."
        )
    )


# -----------------------------------------------------------------------------
# Main Evaluation Entry Point
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
    # Initialize evaluator; root is non-critical parallel aggregator
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
        default_model=model
    )

    # Extract structured state information from the answer
    extraction = await evaluator.extract(
        prompt=prompt_extract_states(),
        template_class=StatesExtraction,
        extraction_name="states_extraction"
    )

    # Filter/pad to exactly 4 states (first four)
    states: List[StateInfo] = extraction.states[:4]
    while len(states) < 4:
        states.append(StateInfo())

    # Build per-state verification blocks
    # Each state block is a parallel subtree as per rubric; internal critical groups enforce required info
    for i in range(4):
        await verify_state_block(evaluator, root, states[i], i)

    return evaluator.get_summary()