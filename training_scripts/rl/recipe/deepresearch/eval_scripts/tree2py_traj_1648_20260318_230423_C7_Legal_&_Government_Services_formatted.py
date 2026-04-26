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
TASK_ID = "state_legislative_compliance_2026"
TASK_DESCRIPTION = """
Identify a U.S. state whose 2026 regular legislative session meets all of the following procedural, transparency, and accessibility requirements: 
(1) The session must begin between January 5 and January 20, 2026; 
(2) The session must be scheduled to run for at least 60 consecutive days; 
(3) The state must have a clearly defined and published deadline for bill introduction during the session; 
(4) The state must require advance public notice for legislative committee meetings; 
(5) All legislative committee and floor meetings must be open to public attendance; 
(6) The state must require recorded roll call votes for final passage of legislation; 
(7) The state must have a defined constitutional or statutory timeline for gubernatorial action on passed bills; 
(8) The state must provide public access to legislative records, bills, and amendments; 
(9) The state must have a specified maximum response time for public records requests; 
(10) The state must maintain a publicly accessible online bill tracking system; 
(11) The state legislature must have established standing committees with defined subject matter jurisdictions; 
(12) The state must require fiscal impact analysis for legislation with budgetary implications; 
(13) The state must provide electronic online access to bill text, amendments, and committee reports; 
(14) The state must maintain and publish minutes or recordings of legislative meetings. 
Provide the name of the state and, for each requirement, provide the specific information demonstrating compliance along with a reference URL supporting the claim.
"""


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class Criterion(BaseModel):
    detail: Optional[str] = None               # Key statement or info provided in the answer
    urls: List[str] = Field(default_factory=list)  # All supporting URLs explicitly cited in the answer


class LegislativeComplianceExtraction(BaseModel):
    state_name: Optional[str] = None

    session_start: Optional[Criterion] = None
    session_duration: Optional[Criterion] = None
    bill_intro_deadline: Optional[Criterion] = None
    committee_meeting_notice: Optional[Criterion] = None
    meeting_public_access: Optional[Criterion] = None
    recorded_votes: Optional[Criterion] = None
    governor_action_timeline: Optional[Criterion] = None
    public_records_access: Optional[Criterion] = None
    records_request_response: Optional[Criterion] = None
    online_bill_tracking: Optional[Criterion] = None
    standing_committees: Optional[Criterion] = None
    fiscal_analysis: Optional[Criterion] = None
    electronic_document_access: Optional[Criterion] = None
    meeting_minutes: Optional[Criterion] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_legislative_compliance() -> str:
    return """
    From the answer, extract the single U.S. state's name and, for each listed requirement, extract:
    - detail: the specific statement or fact the answer uses to demonstrate compliance (quote or paraphrase from the answer).
    - urls: all source URLs explicitly cited in the answer that support that requirement. Include only valid, complete URLs that appear in the answer; do not infer or add any.

    If an item is missing, set its field to null (for detail) and an empty array for urls.

    Return JSON with this schema:
    {
      "state_name": string | null,

      "session_start": {"detail": string | null, "urls": string[]},
      "session_duration": {"detail": string | null, "urls": string[]},
      "bill_intro_deadline": {"detail": string | null, "urls": string[]},
      "committee_meeting_notice": {"detail": string | null, "urls": string[]},
      "meeting_public_access": {"detail": string | null, "urls": string[]},
      "recorded_votes": {"detail": string | null, "urls": string[]},
      "governor_action_timeline": {"detail": string | null, "urls": string[]},
      "public_records_access": {"detail": string | null, "urls": string[]},
      "records_request_response": {"detail": string | null, "urls": string[]},
      "online_bill_tracking": {"detail": string | null, "urls": string[]},
      "standing_committees": {"detail": string | null, "urls": string[]},
      "fiscal_analysis": {"detail": string | null, "urls": string[]},
      "electronic_document_access": {"detail": string | null, "urls": string[]},
      "meeting_minutes": {"detail": string | null, "urls": string[]}
    }

    Special instructions:
    - state_name: If multiple states are mentioned, return the primary one the answer claims satisfies all requirements. If unclear, pick the first explicitly asserted as compliant.
    - urls: Extract only URLs that are explicitly shown in the answer text (including markdown links). Do not fabricate URLs. If no URL is provided for a requirement, return an empty list for that requirement.
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def sanitize_urls(urls: Optional[List[str]]) -> List[str]:
    if not urls:
        return []
    cleaned = []
    for u in urls:
        if not u:
            continue
        s = u.strip()
        if not s:
            continue
        cleaned.append(s)
    # Deduplicate while preserving order
    seen = set()
    deduped = []
    for u in cleaned:
        if u not in seen:
            seen.add(u)
            deduped.append(u)
    return deduped


# --------------------------------------------------------------------------- #
# Verification builders                                                       #
# --------------------------------------------------------------------------- #
async def verify_state_identification(evaluator: Evaluator, parent_node, state_name: Optional[str]) -> None:
    """
    Build the State_Identification subtree:
      - Existence check (critical)
      - Simple verification that the named entity is one of the 50 U.S. states (critical)
    """
    group = evaluator.add_sequential(
        id="State_Identification",
        desc="A U.S. state must be clearly identified by name",
        parent=parent_node,
        critical=True
    )

    # Existence
    exists = evaluator.add_custom_node(
        result=(state_name is not None and str(state_name).strip() != ""),
        id="State_Identification_info_and_sources",
        desc="State name is provided in the answer",
        parent=group,
        critical=True
    )

    # Simple verification (no URL needed; general factual check)
    leaf = evaluator.add_leaf(
        id="State_Identification_supported",
        desc="A U.S. state must be clearly identified by name",
        parent=group,
        critical=True
    )
    claim = f"'{state_name}' is one of the 50 U.S. states of the United States of America."
    await evaluator.verify(
        claim=claim,
        node=leaf,
        additional_instruction="Treat District of Columbia and U.S. territories as not states. Minor spelling variants that still clearly refer to a real U.S. state should be accepted."
    )


async def verify_requirement(
    evaluator: Evaluator,
    parent_node,
    *,
    node_id: str,
    description: str,
    criterion: Optional[Criterion],
    state_name: Optional[str],
    requirement_specific_instruction: str
) -> None:
    """
    Generic builder for each requirement:
      - Sequential group (critical)
      - Existence node: requires non-empty detail and at least one URL (critical)
      - URL-grounded verification leaf (critical)
    """
    group = evaluator.add_sequential(
        id=node_id,
        desc=description,
        parent=parent_node,
        critical=True
    )

    detail = (criterion.detail if criterion else None) or ""
    urls = sanitize_urls(criterion.urls if criterion else [])

    # Existence: must have detail AND at least one source URL
    evaluator.add_custom_node(
        result=(detail.strip() != "" and len(urls) > 0),
        id=f"{node_id}_info_and_sources",
        desc=f"{description} — detail and at least one supporting URL are provided",
        parent=group,
        critical=True
    )

    # URL-grounded verification
    leaf = evaluator.add_leaf(
        id=f"{node_id}_supported",
        desc=description,
        parent=group,
        critical=True
    )

    generic_base_instruction = (
        "Use only the provided webpage(s). Focus strictly on the named state's 2026 REGULAR legislative session "
        "and the specific requirement being checked. Ignore special sessions or past-year pages unless they "
        "explicitly and unambiguously state the 2026 rule. If pages are unrelated, inaccessible, or do not "
        "explicitly support the claim, return NOT SUPPORTED."
    )

    # Build a precise claim tying the requirement, state, and provided detail
    claim = (
        f"According to the cited source(s), for the state of '{state_name}', this statement about the 2026 regular "
        f"legislative session is correct: {description}. Specifically, the answer asserts: '{detail}'."
    )

    await evaluator.verify(
        claim=claim,
        node=leaf,
        sources=urls,
        additional_instruction=f"{generic_base_instruction}\n\nRequirement-specific guidance:\n{requirement_specific_instruction}"
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
    Evaluate an answer for the 2026 state legislative compliance task.
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
        default_model=model
    )

    # Create the top-level critical node representing complete compliance
    compliance_root = evaluator.add_parallel(
        id="Complete_State_Legislative_Compliance",
        desc="Verify that a U.S. state is identified and that its 2026 legislative session meets all 14 specified procedural, transparency, and accessibility requirements",
        parent=root,
        critical=True
    )

    # Extract structured info from the answer
    extracted: LegislativeComplianceExtraction = await evaluator.extract(
        prompt=prompt_extract_legislative_compliance(),
        template_class=LegislativeComplianceExtraction,
        extraction_name="legislative_compliance_extraction"
    )

    # Record basic custom info for easier debugging
    evaluator.add_custom_info(
        {
            "state_name": extracted.state_name,
            "has_urls_per_requirement": {
                "session_start": len(sanitize_urls(getattr(extracted.session_start or Criterion(), "urls", []))),
                "session_duration": len(sanitize_urls(getattr(extracted.session_duration or Criterion(), "urls", []))),
                "bill_intro_deadline": len(sanitize_urls(getattr(extracted.bill_intro_deadline or Criterion(), "urls", []))),
                "committee_meeting_notice": len(sanitize_urls(getattr(extracted.committee_meeting_notice or Criterion(), "urls", []))),
                "meeting_public_access": len(sanitize_urls(getattr(extracted.meeting_public_access or Criterion(), "urls", []))),
                "recorded_votes": len(sanitize_urls(getattr(extracted.recorded_votes or Criterion(), "urls", []))),
                "governor_action_timeline": len(sanitize_urls(getattr(extracted.governor_action_timeline or Criterion(), "urls", []))),
                "public_records_access": len(sanitize_urls(getattr(extracted.public_records_access or Criterion(), "urls", []))),
                "records_request_response": len(sanitize_urls(getattr(extracted.records_request_response or Criterion(), "urls", []))),
                "online_bill_tracking": len(sanitize_urls(getattr(extracted.online_bill_tracking or Criterion(), "urls", []))),
                "standing_committees": len(sanitize_urls(getattr(extracted.standing_committees or Criterion(), "urls", []))),
                "fiscal_analysis": len(sanitize_urls(getattr(extracted.fiscal_analysis or Criterion(), "urls", []))),
                "electronic_document_access": len(sanitize_urls(getattr(extracted.electronic_document_access or Criterion(), "urls", []))),
                "meeting_minutes": len(sanitize_urls(getattr(extracted.meeting_minutes or Criterion(), "urls", []))),
            }
        },
        info_type="extraction_meta",
        info_name="extraction_meta_overview"
    )

    # Build and run verifications
    await verify_state_identification(evaluator, compliance_root, extracted.state_name)

    # 1) Session start date window
    await verify_requirement(
        evaluator, compliance_root,
        node_id="Session_Start_Date",
        description="The state's 2026 regular legislative session must begin between January 5 and January 20, 2026",
        criterion=extracted.session_start,
        state_name=extracted.state_name,
        requirement_specific_instruction=(
            "Confirm the opening date ('convenes', 'first day', 'opening day') falls within 2026-01-05 to 2026-01-20 inclusive. "
            "If the page shows a different year, a special session, or an out-of-window start date, mark as NOT SUPPORTED."
        )
    )

    # 2) Session duration >= 60 consecutive days
    await verify_requirement(
        evaluator, compliance_root,
        node_id="Session_Duration",
        description="The legislative session must be scheduled to run for at least 60 consecutive days",
        criterion=extracted.session_duration,
        state_name=extracted.state_name,
        requirement_specific_instruction=(
            "Accept explicit statements such as 'at least 60 consecutive days' or a published start/end date range spanning ≥60 calendar days. "
            "If only 'legislative days' are mentioned without a continuous calendar range, do not assume consecutiveness."
        )
    )

    # 3) Bill introduction deadline
    await verify_requirement(
        evaluator, compliance_root,
        node_id="Bill_Introduction_Deadline",
        description="The state must have a clearly defined and published deadline for bill introduction during the session",
        criterion=extracted.bill_intro_deadline,
        state_name=extracted.state_name,
        requirement_specific_instruction=(
            "Look for 'bill introduction deadline', 'final day to introduce/file bills', or equivalent with a specific date or rule for the 2026 regular session."
        )
    )

    # 4) Advance notice for committee meetings
    await verify_requirement(
        evaluator, compliance_root,
        node_id="Committee_Meeting_Notice",
        description="The state must require advance public notice for legislative committee meetings",
        criterion=extracted.committee_meeting_notice,
        state_name=extracted.state_name,
        requirement_specific_instruction=(
            "Confirm a rule, statute, or formal policy requiring advance public notice (posting) for committee meetings."
        )
    )

    # 5) Public access to committee and floor meetings
    await verify_requirement(
        evaluator, compliance_root,
        node_id="Meeting_Public_Access",
        description="All legislative committee and floor meetings must be open to public attendance",
        criterion=extracted.meeting_public_access,
        state_name=extracted.state_name,
        requirement_specific_instruction=(
            "Verify rules or statutes requiring that committee meetings and floor sessions are open to the public (exceptions like executive session may exist but the default must be open)."
        )
    )

    # 6) Recorded roll call votes
    await verify_requirement(
        evaluator, compliance_root,
        node_id="Recorded_Votes",
        description="The state must require recorded roll call votes for final passage of legislation",
        criterion=extracted.recorded_votes,
        state_name=extracted.state_name,
        requirement_specific_instruction=(
            "Confirm that final passage on the floor requires a recorded roll call vote and that the votes are recorded (e.g., in journals or official records)."
        )
    )

    # 7) Governor action timeline
    await verify_requirement(
        evaluator, compliance_root,
        node_id="Governor_Action_Timeline",
        description="The state must have a defined constitutional or statutory timeline for gubernatorial action on passed bills",
        criterion=extracted.governor_action_timeline,
        state_name=extracted.state_name,
        requirement_specific_instruction=(
            "Confirm the constitution or statute defines how many days the governor has to sign, veto, or otherwise act on a bill; an explicit timeline must be stated."
        )
    )

    # 8) Public access to legislative records, bills, and amendments
    await verify_requirement(
        evaluator, compliance_root,
        node_id="Public_Records_Access",
        description="The state must provide public access to legislative records, bills, and amendments",
        criterion=extracted.public_records_access,
        state_name=extracted.state_name,
        requirement_specific_instruction=(
            "Confirm the public can access legislative records including bills and amendments via official channels (e.g., legislature website)."
        )
    )

    # 9) Specified maximum response time for public records requests
    await verify_requirement(
        evaluator, compliance_root,
        node_id="Records_Request_Response",
        description="The state must have a specified maximum response time for public records requests",
        criterion=extracted.records_request_response,
        state_name=extracted.state_name,
        requirement_specific_instruction=(
            "Look for a Public Records Act/FOIA statute specifying a maximum response time (e.g., within X business or calendar days). "
            "Vague terms like 'promptly' without a maximum number of days should be marked NOT SUPPORTED."
        )
    )

    # 10) Publicly accessible online bill tracking
    await verify_requirement(
        evaluator, compliance_root,
        node_id="Online_Bill_Tracking",
        description="The state must maintain a publicly accessible online bill tracking system",
        criterion=extracted.online_bill_tracking,
        state_name=extracted.state_name,
        requirement_specific_instruction=(
            "Verify an official, public, no-login-required portal for tracking/searching bills exists (prefer official legislature domain; third-party aggregators generally do not qualify)."
        )
    )

    # 11) Standing committees with defined jurisdictions
    await verify_requirement(
        evaluator, compliance_root,
        node_id="Standing_Committees",
        description="The state legislature must have established standing committees with defined subject matter jurisdictions",
        criterion=extracted.standing_committees,
        state_name=extracted.state_name,
        requirement_specific_instruction=(
            "Check that the legislature has standing committees and that each has a described subject-matter jurisdiction (via rules or official listings)."
        )
    )

    # 12) Fiscal impact analysis required
    await verify_requirement(
        evaluator, compliance_root,
        node_id="Fiscal_Analysis",
        description="The state must require fiscal impact analysis for legislation with budgetary implications",
        criterion=extracted.fiscal_analysis,
        state_name=extracted.state_name,
        requirement_specific_instruction=(
            "Confirm statutes, rules, or manuals require fiscal notes/impact analyses for bills with budgetary or fiscal effects."
        )
    )

    # 13) Electronic online access to bill text, amendments, and committee reports
    await verify_requirement(
        evaluator, compliance_root,
        node_id="Electronic_Document_Access",
        description="The state must provide electronic online access to bill text, amendments, and committee reports",
        criterion=extracted.electronic_document_access,
        state_name=extracted.state_name,
        requirement_specific_instruction=(
            "Verify that bill text, amendments, and committee reports are accessible online in electronic form to the public."
        )
    )

    # 14) Publish minutes or recordings of legislative meetings
    await verify_requirement(
        evaluator, compliance_root,
        node_id="Meeting_Minutes",
        description="The state must maintain and publish minutes or recordings of legislative meetings",
        criterion=extracted.meeting_minutes,
        state_name=extracted.state_name,
        requirement_specific_instruction=(
            "Confirm that official minutes and/or audio/video recordings of committee and/or floor meetings are maintained and publicly available online."
        )
    )

    # Return structured result
    return evaluator.get_summary()