import asyncio
import logging
from typing import Any, List, Optional, Dict

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "compliance_profile_may2025"
TASK_DESCRIPTION = (
    "A legal compliance consulting firm is expanding its services and needs to compile comparative regulatory information "
    "for state governments with specific characteristics.\n\n"
    "Identify four U.S. states that meet BOTH of the following criteria:\n"
    "1. The state's 2025 regular legislative session adjourns during May 2025 (between May 1 and May 31, 2025)\n"
    "2. The state requires annual or periodic report filings for Limited Liability Companies (LLCs) or corporations\n\n"
    "For each of the four states you identify, provide a structured compliance profile containing the following information "
    "(with supporting URL references for each data point):\n\n"
    "Required Information:\n"
    "- The exact adjournment date of the state's 2025 regular legislative session\n"
    "- The due date or filing frequency for LLC annual/periodic reports\n"
    "- The filing fee amount for LLC annual/periodic reports\n"
    "- The due date or filing frequency for corporation annual/periodic reports\n"
    "- The filing fee amount for corporation annual/periodic reports\n\n"
    "Additional Information (if readily available):\n"
    "- The state's mandated response time for FOIA or public records requests\n"
    "- The minimum number of employees that triggers mandatory workers' compensation insurance requirements\n"
    "- Whether the state offers bar admission reciprocity with other states\n"
    "- The number of years of experience required to obtain a real estate broker license in that state\n\n"
    "Present the answer clearly with each state labeled and all information cited with URLs from official state government sources "
    "or authoritative legal compliance resources."
)


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class StateProfile(BaseModel):
    name: Optional[str] = None

    # Required info + URLs
    adjournment_date: Optional[str] = None
    adjournment_url: Optional[str] = None

    llc_report_due: Optional[str] = None
    llc_due_url: Optional[str] = None
    llc_fee_amount: Optional[str] = None
    llc_fee_url: Optional[str] = None

    corp_report_due: Optional[str] = None
    corp_due_url: Optional[str] = None
    corp_fee_amount: Optional[str] = None
    corp_fee_url: Optional[str] = None

    # Optional info + URLs
    foia_response_time: Optional[str] = None
    foia_url: Optional[str] = None

    workers_comp_threshold: Optional[str] = None
    workers_comp_url: Optional[str] = None

    bar_reciprocity_status: Optional[str] = None
    bar_reciprocity_url: Optional[str] = None

    broker_experience_requirement: Optional[str] = None
    broker_license_url: Optional[str] = None


class ComplianceExtraction(BaseModel):
    states: List[StateProfile] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_states() -> str:
    return """
    Extract up to four U.S. states that the answer claims meet BOTH of the following criteria:
    (1) The state's 2025 regular legislative session adjourns during May 2025 (between May 1 and May 31, 2025), and
    (2) The state requires annual or periodic report filings for LLCs or corporations.

    For each identified state, extract the following fields exactly as presented in the answer text. Include supporting URLs
    for each data point if they are provided in the answer (extract the actual URLs). If a field is not present, set it to null.

    Required information for each state:
    - name: The state's name
    - adjournment_date: The exact adjournment date of the 2025 regular legislative session
    - adjournment_url: A URL that supports the adjournment date
    - llc_report_due: The due date or filing frequency for LLC annual/periodic reports (e.g., "Annually by April 1", "Biennially in odd years", etc.)
    - llc_due_url: A URL supporting the LLC due date/frequency
    - llc_fee_amount: The filing fee amount for LLC annual/periodic reports (string, e.g., "$50", "USD 60", "varies by report")
    - llc_fee_url: A URL supporting the LLC fee amount
    - corp_report_due: The due date or filing frequency for corporation (for-profit) annual/periodic reports
    - corp_due_url: A URL supporting the corporation due date/frequency
    - corp_fee_amount: The filing fee amount for corporation annual/periodic reports (string)
    - corp_fee_url: A URL supporting the corporation fee amount

    Additional information (if present in the answer):
    - foia_response_time: The state's mandated response time for FOIA/public records requests
    - foia_url: A URL supporting the FOIA response time
    - workers_comp_threshold: The minimum number of employees that triggers mandatory workers' compensation insurance
    - workers_comp_url: A URL supporting the workers' comp threshold
    - bar_reciprocity_status: Whether the state offers bar admission reciprocity with other states (e.g., "Yes, limited", "No")
    - bar_reciprocity_url: A URL supporting the reciprocity status
    - broker_experience_requirement: Years of experience required to obtain a real estate broker license
    - broker_license_url: A URL supporting the broker experience requirement

    Return a JSON object with a single field:
    { "states": [ ... up to 4 StateProfile objects ... ] }

    IMPORTANT:
    - Extract only what is explicitly in the answer. Do not infer missing values.
    - For URLs, include the full URL string appearing in the answer (including http/https). If the answer uses markdown links, extract the URL part.
    - If the answer lists more than four states, extract all you find; we will pick the first four later.
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def is_valid_url(url: Optional[str]) -> bool:
    if url is None:
        return False
    u = url.strip().lower()
    return u.startswith("http://") or u.startswith("https://")


def nonempty_list(*urls: Optional[str]) -> List[str]:
    return [u for u in urls if is_valid_url(u)]


# --------------------------------------------------------------------------- #
# Verification per-state                                                      #
# --------------------------------------------------------------------------- #
async def verify_state_profile(
    evaluator: Evaluator,
    parent_node,
    idx: int,
    state: StateProfile,
) -> None:
    """Build and verify the compliance profile subtree for one state."""

    # Create State node (parallel, non-critical overall)
    state_node = evaluator.add_parallel(
        id=f"State_{idx + 1}",
        desc=f"{['First','Second','Third','Fourth'][idx]} qualifying state with complete compliance information",
        parent=parent_node,
        critical=False,
    )

    # Existence checks for required URLs (critical under state as per rubric)
    adj_url_exists = evaluator.add_custom_node(
        result=is_valid_url(state.adjournment_url),
        id=f"state_{idx + 1}_Legislative_Adjournment_URL",
        desc="Provides URL reference supporting the legislative session adjournment date",
        parent=state_node,
        critical=True,
    )

    llc_due_url_exists = evaluator.add_custom_node(
        result=is_valid_url(state.llc_due_url),
        id=f"state_{idx + 1}_LLC_Due_Date_URL",
        desc="Provides URL reference supporting LLC annual report due date",
        parent=state_node,
        critical=True,
    )
    llc_fee_url_exists = evaluator.add_custom_node(
        result=is_valid_url(state.llc_fee_url),
        id=f"state_{idx + 1}_LLC_Fee_URL",
        desc="Provides URL reference supporting LLC annual report fee",
        parent=state_node,
        critical=True,
    )

    corp_due_url_exists = evaluator.add_custom_node(
        result=is_valid_url(state.corp_due_url),
        id=f"state_{idx + 1}_Corporation_Due_Date_URL",
        desc="Provides URL reference supporting corporation annual report due date",
        parent=state_node,
        critical=True,
    )
    corp_fee_url_exists = evaluator.add_custom_node(
        result=is_valid_url(state.corp_fee_url),
        id=f"state_{idx + 1}_Corporation_Fee_URL",
        desc="Provides URL reference supporting corporation annual report fee",
        parent=state_node,
        critical=True,
    )

    # Meets selection criteria (critical gate, parallel aggregation)
    criteria_node = evaluator.add_parallel(
        id=f"state_{idx + 1}_Meets_Selection_Criteria",
        desc="State satisfies both primary selection criteria",
        parent=state_node,
        critical=True,
    )

    # Criterion 1: May 2025 adjournment
    may_adj_node = evaluator.add_leaf(
        id=f"state_{idx + 1}_May_2025_Adjournment",
        desc="State's 2025 regular legislative session adjourns between May 1-31, 2025",
        parent=criteria_node,
        critical=True,
    )
    claim_adj_may = (
        f"The 2025 regular legislative session in {state.name or 'the state'} adjourned on {state.adjournment_date}, "
        f"and this date falls between May 1 and May 31, 2025."
    )
    await evaluator.verify(
        claim=claim_adj_may,
        node=may_adj_node,
        sources=state.adjournment_url,
        additional_instruction=(
            "Verify the adjournment date and confirm that the date is in May 2025. "
            "Treat minor date format variations as acceptable as long as the date clearly indicates a day in May 2025."
        ),
    )

    # Criterion 2: Has annual/periodic report requirement for LLCs or corporations
    has_reports_node = evaluator.add_leaf(
        id=f"state_{idx + 1}_Has_Annual_Report_Requirement",
        desc="State requires annual or periodic reports for LLCs or corporations",
        parent=criteria_node,
        critical=True,
    )
    claim_has_reports = (
        f"{state.name or 'The state'} requires annual or periodic report filings for at least one of: LLCs or corporations."
    )
    await evaluator.verify(
        claim=claim_has_reports,
        node=has_reports_node,
        sources=nonempty_list(state.llc_due_url, state.corp_due_url),
        additional_instruction=(
            "Pass if any provided source indicates that either LLCs or corporations must file periodic or annual reports. "
            "It is sufficient that only one entity type (LLCs or corporations) has such a requirement."
        ),
    )

    # Required info verifications
    # Legislative adjournment date exact value check
    adj_date_node = evaluator.add_leaf(
        id=f"state_{idx + 1}_Legislative_Adjournment_Date",
        desc="Provides exact adjournment date of 2025 regular legislative session",
        parent=state_node,
        critical=True,
    )
    claim_adj_date = (
        f"The exact adjournment date for the {state.name or 'state'} 2025 regular legislative session is '{state.adjournment_date}'."
    )
    await evaluator.verify(
        claim=claim_adj_date,
        node=adj_date_node,
        sources=state.adjournment_url,
        additional_instruction="Confirm that the page explicitly states this adjournment date.",
    )

    # LLC due date/frequency
    llc_due_node = evaluator.add_leaf(
        id=f"state_{idx + 1}_LLC_Annual_Report_Due_Date",
        desc="Provides LLC annual report due date or filing frequency",
        parent=state_node,
        critical=True,
    )
    claim_llc_due = (
        f"For {state.name or 'the state'}, the LLC annual/periodic report due date or filing frequency is '{state.llc_report_due}'."
    )
    await evaluator.verify(
        claim=claim_llc_due,
        node=llc_due_node,
        sources=state.llc_due_url,
        additional_instruction=(
            "Confirm that the cited page indicates the timing requirement (due date or filing frequency) for LLC reports. "
            "Allow reasonable phrasing variants (e.g., 'annually by May 1', 'biennially in odd years')."
        ),
    )

    # LLC filing fee
    llc_fee_node = evaluator.add_leaf(
        id=f"state_{idx + 1}_LLC_Annual_Report_Fee",
        desc="Provides LLC annual report filing fee amount",
        parent=state_node,
        critical=True,
    )
    claim_llc_fee = (
        f"For {state.name or 'the state'}, the LLC annual/periodic report filing fee amount is '{state.llc_fee_amount}'."
    )
    await evaluator.verify(
        claim=claim_llc_fee,
        node=llc_fee_node,
        sources=state.llc_fee_url,
        additional_instruction=(
            "Confirm that the page indicates the fee for filing the LLC annual/periodic report. "
            "Allow minor formatting variations in currency symbols or wording."
        ),
    )

    # Corporation due date/frequency
    corp_due_node = evaluator.add_leaf(
        id=f"state_{idx + 1}_Corporation_Annual_Report_Due_Date",
        desc="Provides corporation annual report due date or filing frequency",
        parent=state_node,
        critical=True,
    )
    claim_corp_due = (
        f"For {state.name or 'the state'}, the corporation (for‑profit) annual/periodic report due date or filing frequency is '{state.corp_report_due}'."
    )
    await evaluator.verify(
        claim=claim_corp_due,
        node=corp_due_node,
        sources=state.corp_due_url,
        additional_instruction=(
            "Confirm the timing requirement for corporation annual/periodic reports on the cited page."
        ),
    )

    # Corporation filing fee
    corp_fee_node = evaluator.add_leaf(
        id=f"state_{idx + 1}_Corporation_Annual_Report_Fee",
        desc="Provides corporation annual report filing fee amount",
        parent=state_node,
        critical=True,
    )
    claim_corp_fee = (
        f"For {state.name or 'the state'}, the corporation annual/periodic report filing fee amount is '{state.corp_fee_amount}'."
    )
    await evaluator.verify(
        claim=claim_corp_fee,
        node=corp_fee_node,
        sources=state.corp_fee_url,
        additional_instruction=(
            "Confirm the reported fee amount for corporation annual/periodic report filings."
        ),
    )

    # Optional information verifications (non-critical)
    foia_node = evaluator.add_leaf(
        id=f"state_{idx + 1}_FOIA_Response_Time",
        desc="Provides state's FOIA or public records request response time requirement",
        parent=state_node,
        critical=False,
    )
    claim_foia = (
        f"For {state.name or 'the state'}, the mandated response time for FOIA/public records requests is '{state.foia_response_time}'."
    )
    await evaluator.verify(
        claim=claim_foia,
        node=foia_node,
        sources=state.foia_url,
        additional_instruction=(
            "Verify that the page indicates a statutory or policy-based response time (e.g., '5 business days', '10 days')."
        ),
    )

    wc_node = evaluator.add_leaf(
        id=f"state_{idx + 1}_Workers_Comp_Threshold",
        desc="Provides minimum number of employees that triggers workers' compensation insurance requirement",
        parent=state_node,
        critical=False,
    )
    claim_wc = (
        f"For {state.name or 'the state'}, the minimum number of employees that triggers mandatory workers’ compensation insurance is '{state.workers_comp_threshold}'."
    )
    await evaluator.verify(
        claim=claim_wc,
        node=wc_node,
        sources=state.workers_comp_url,
        additional_instruction=(
            "Verify the threshold (number of employees) that mandates workers’ compensation coverage."
        ),
    )

    bar_node = evaluator.add_leaf(
        id=f"state_{idx + 1}_Bar_Reciprocity_Status",
        desc="Indicates whether state offers bar admission reciprocity with other states",
        parent=state_node,
        critical=False,
    )
    claim_bar = (
        f"For {state.name or 'the state'}, the bar admission reciprocity status is '{state.bar_reciprocity_status}'."
    )
    await evaluator.verify(
        claim=claim_bar,
        node=bar_node,
        sources=state.bar_reciprocity_url,
        additional_instruction=(
            "Verify whether the state accepts admission on motion/reciprocity (full or limited) or does not offer reciprocity."
        ),
    )

    broker_node = evaluator.add_leaf(
        id=f"state_{idx + 1}_Broker_Experience_Requirement",
        desc="Provides years of experience required to obtain a real estate broker license",
        parent=state_node,
        critical=False,
    )
    claim_broker = (
        f"For {state.name or 'the state'}, the years of experience required to obtain a real estate broker license is '{state.broker_experience_requirement}'."
    )
    await evaluator.verify(
        claim=claim_broker,
        node=broker_node,
        sources=state.broker_license_url,
        additional_instruction=(
            "Verify the experience requirement for broker licensure; allow wording variations such as 'active salesperson experience', 'completed transactions', etc."
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
    Evaluate an answer for the Compliance Profile task.
    """
    # Initialize evaluator with root node using parallel aggregation
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,
        agent_name=agent_name,
        answer_name=answer_name,
        client=client,
        task_description="Identifies 4 U.S. states where the 2025 regular legislative session adjourns in May 2025 and which have annual report requirements for business entities, providing comprehensive compliance information for each state",
        answer=answer,
        global_cache=cache,
        global_semaphore=semaphore,
        logger=logger,
        default_model=model,
    )

    # Create top-level Compliance_Profile node (parallel, non-critical)
    compliance_node = evaluator.add_parallel(
        id="Compliance_Profile",
        desc="Identifies 4 U.S. states where the 2025 regular legislative session adjourns in May 2025 and which have annual report requirements for business entities, providing comprehensive compliance information for each state",
        parent=root,
        critical=False,
    )

    # Extract states and compliance info from the answer
    extraction = await evaluator.extract(
        prompt=prompt_extract_states(),
        template_class=ComplianceExtraction,
        extraction_name="state_compliance_profiles",
    )

    # Normalize to exactly 4 states (pad with empty placeholders if fewer)
    extracted_states: List[StateProfile] = list(extraction.states or [])
    if len(extracted_states) > 4:
        extracted_states = extracted_states[:4]
    while len(extracted_states) < 4:
        extracted_states.append(StateProfile())

    # Verify each state's compliance profile
    for i, st in enumerate(extracted_states):
        await verify_state_profile(evaluator, compliance_node, i, st)

    # Return structured evaluation summary
    return evaluator.get_summary()