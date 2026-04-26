import asyncio
import logging
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys


TASK_ID = "state_cpa_cpe_requirements"
TASK_DESCRIPTION = """Identify a U.S. state that meets all of the following CPA (Certified Public Accountant) continuing professional education (CPE) and license renewal requirements:

1. The state must have a biennial (2-year) CPA license renewal cycle
2. The state must require exactly 80 CPE hours every 2 years for license renewal
3. The state must require exactly 4 hours of ethics CPE within each renewal period
4. The ethics CPE courses must be state-specific or board-approved by that state's board of accountancy
5. The state must require a specific minimum number of CPE hours in Accounting and Auditing subjects
6. The state must impose a maximum limit on behavioral subject CPE hours
7. The state must meet NASBA substantially equivalent requirements under the UAA 3E criteria
8. The state's CPE reporting period must end on June 30th

For your answer, provide:
- The name of the state
- The specific number of Accounting and Auditing CPE hours required
- The maximum number of behavioral subject CPE hours allowed
- Supporting reference URLs for each requirement
"""


class StateCPEExtraction(BaseModel):
    """Structured extraction of the state CPE answer content and source URLs."""
    # Required reported fields
    state_name: Optional[str] = None
    aa_min_hours: Optional[str] = None
    behavioral_max_hours: Optional[str] = None
    # Optional reported fields (still extracted if present)
    total_cpe_hours: Optional[str] = None
    ethics_hours: Optional[str] = None

    # Source URLs categorized by requirement
    renewal_cycle_urls: List[str] = Field(default_factory=list)
    cpe_total_urls: List[str] = Field(default_factory=list)
    ethics_hours_urls: List[str] = Field(default_factory=list)
    ethics_course_approval_urls: List[str] = Field(default_factory=list)
    aa_min_urls: List[str] = Field(default_factory=list)
    behavioral_max_urls: List[str] = Field(default_factory=list)
    nasba_uua_urls: List[str] = Field(default_factory=list)
    reporting_end_urls: List[str] = Field(default_factory=list)


def prompt_extract_state_cpe_info() -> str:
    return """
    Extract the requested information from the answer about a single U.S. state's CPA CPE and renewal requirements.

    1) Required reported fields (return the text exactly as provided; use digits if available):
       - state_name: The name of the state identified in the answer.
       - aa_min_hours: The specific numeric minimum number of Accounting and Auditing (A&A) CPE hours required per renewal period for the chosen state. Prefer digits (e.g., "8") rather than words.
       - behavioral_max_hours: The specific numeric maximum number of behavioral subject CPE hours allowed per renewal period for the chosen state. Prefer digits (e.g., "24") rather than words.
       - total_cpe_hours: The total CPE hours required per 2-year renewal cycle, if stated (e.g., "80"). If not explicitly stated, return null.
       - ethics_hours: The ethics CPE hours required per renewal period, if stated (e.g., "4"). If not explicitly stated, return null.

    2) Supporting reference URLs for each requirement. Extract actual URLs (plain or in markdown links) used in the answer and categorize them:
       - renewal_cycle_urls: URLs supporting the biennial (2-year) CPA license renewal cycle.
       - cpe_total_urls: URLs supporting the requirement of exactly 80 CPE hours per 2-year cycle.
       - ethics_hours_urls: URLs supporting the requirement of exactly 4 ethics CPE hours per renewal period.
       - ethics_course_approval_urls: URLs supporting that ethics CPE must be state-specific or board-approved by the state's board of accountancy.
       - aa_min_urls: URLs supporting the existence of a specific minimum A&A CPE hours requirement.
       - behavioral_max_urls: URLs supporting the existence of a maximum limit on behavioral subject CPE hours.
       - nasba_uua_urls: URLs supporting that the state meets NASBA substantially equivalent requirements under UAA 3E criteria.
       - reporting_end_urls: URLs supporting that the state's CPE reporting period ends on June 30.

    RULES:
    - Extract only URLs explicitly present in the answer (including markdown links). Do not invent URLs.
    - If a category has no URLs provided, return an empty list for that category.
    - If a numeric field is not present in the answer, return null.
    - The output must be a single JSON object conforming exactly to the specified schema.
    """


def _has_numberlike(value: Optional[str]) -> bool:
    """Check if a string contains at least one digit, indicating a numeric value."""
    return bool(value) and any(ch.isdigit() for ch in value)


async def _build_constraints_checks(
    evaluator: Evaluator,
    parent_node,
    extracted: StateCPEExtraction
) -> None:
    """Build and execute verification leaf checks for all eight constraints."""
    constraints_node = evaluator.add_parallel(
        id="Meets_All_CPE_and_Renewal_Constraints",
        desc="Chosen state satisfies all eight stated constraints",
        parent=parent_node,
        critical=True
    )

    state_ref = extracted.state_name or "the chosen state"

    # 1. Biennial renewal cycle
    node_renewal = evaluator.add_leaf(
        id="Biennial_Renewal_Cycle",
        desc="State has a biennial (2-year) CPA license renewal cycle",
        parent=constraints_node,
        critical=True
    )
    claim_renewal = f"{state_ref} has a biennial (every 2 years) CPA license renewal cycle."
    add_ins_renewal = "Look for wording such as 'biennial', 'every two years', or a 2-year renewal cycle for the CPA license (not just the CPE reporting period)."
    await evaluator.verify(
        claim=claim_renewal,
        node=node_renewal,
        sources=extracted.renewal_cycle_urls,
        additional_instruction=add_ins_renewal
    )

    # 2. Exactly 80 CPE hours per 2-year cycle
    node_total = evaluator.add_leaf(
        id="Total_CPE_Hours",
        desc="State requires exactly 80 CPE hours every 2 years for renewal",
        parent=constraints_node,
        critical=True
    )
    claim_total = f"{state_ref} requires exactly 80 total CPE hours per biennial (2-year) renewal cycle for CPA license renewal."
    add_ins_total = "Confirm the exact number '80' hours per 2-year cycle for renewal; accept reasonable variants like '80 hours biennially'."
    await evaluator.verify(
        claim=claim_total,
        node=node_total,
        sources=extracted.cpe_total_urls,
        additional_instruction=add_ins_total
    )

    # 3. Exactly 4 ethics CPE hours per renewal period
    node_ethics_hours = evaluator.add_leaf(
        id="Ethics_Hours_Required",
        desc="State requires exactly 4 hours of ethics CPE within each renewal period",
        parent=constraints_node,
        critical=True
    )
    claim_ethics_hours = f"{state_ref} requires exactly 4 hours of ethics CPE in each renewal period."
    add_ins_ethics_hours = "Verify the explicit requirement of 4 ethics hours per renewal period; allow synonyms like 'ethics course' or 'professional ethics'."
    await evaluator.verify(
        claim=claim_ethics_hours,
        node=node_ethics_hours,
        sources=extracted.ethics_hours_urls,
        additional_instruction=add_ins_ethics_hours
    )

    # 4. Ethics course must be state-specific or board-approved by that state's board
    node_ethics_approval = evaluator.add_leaf(
        id="Ethics_Course_Approval",
        desc="Ethics CPE courses must be state-specific or board-approved by that state's board of accountancy",
        parent=constraints_node,
        critical=True
    )
    claim_ethics_approval = f"For {state_ref}, ethics CPE must be state-specific or approved by that state's board of accountancy."
    add_ins_ethics_approval = "Look for phrases like 'state-specific ethics', 'board-approved ethics', or a requirement that the ethics course be approved by the state's board of accountancy."
    await evaluator.verify(
        claim=claim_ethics_approval,
        node=node_ethics_approval,
        sources=extracted.ethics_course_approval_urls,
        additional_instruction=add_ins_ethics_approval
    )

    # 5. Specific minimum A&A hours requirement exists
    node_aa_min = evaluator.add_leaf(
        id="Accounting_Auditing_Minimum_Exists",
        desc="State requires a specific minimum number of CPE hours in Accounting and Auditing subjects (a stated numeric minimum exists)",
        parent=constraints_node,
        critical=True
    )
    aa_min_text = extracted.aa_min_hours or "a stated numeric minimum"
    claim_aa_min = f"{state_ref} requires a specific minimum number of CPE hours in Accounting and Auditing subjects (e.g., {aa_min_text})."
    add_ins_aa_min = "Verify that the policy explicitly sets a numeric minimum of A&A hours within a renewal period; this check does not require verifying the exact number in the answer, only that a numeric minimum exists."
    await evaluator.verify(
        claim=claim_aa_min,
        node=node_aa_min,
        sources=extracted.aa_min_urls,
        additional_instruction=add_ins_aa_min
    )

    # 6. Maximum limit on behavioral subject hours exists
    node_behavioral_max = evaluator.add_leaf(
        id="Behavioral_Subject_Maximum_Exists",
        desc="State imposes a maximum limit on behavioral subject CPE hours (a stated numeric cap exists)",
        parent=constraints_node,
        critical=True
    )
    behavioral_max_text = extracted.behavioral_max_hours or "a stated numeric maximum"
    claim_behavioral_max = f"{state_ref} imposes a maximum limit on behavioral subject CPE hours (e.g., {behavioral_max_text})."
    add_ins_behavioral_max = "Behavioral subjects may be called 'soft skills', 'personal development', or similar. Verify that a numeric maximum/cap exists for behavioral subject hours."
    await evaluator.verify(
        claim=claim_behavioral_max,
        node=node_behavioral_max,
        sources=extracted.behavioral_max_urls,
        additional_instruction=add_ins_behavioral_max
    )

    # 7. NASBA substantial equivalency under UAA 3E
    node_nasba = evaluator.add_leaf(
        id="NASBA_Substantial_Equivalency_UAA_3E",
        desc="State meets NASBA substantial equivalency requirements under UAA 3E criteria",
        parent=constraints_node,
        critical=True
    )
    claim_nasba = f"{state_ref} meets NASBA substantial equivalency under UAA 3E criteria."
    add_ins_nasba = "Confirm that NASBA recognizes the state as substantially equivalent under UAA 3E; look for '3E', 'substantial equivalency', or equivalent phrasing on NASBA or authoritative sources."
    await evaluator.verify(
        claim=claim_nasba,
        node=node_nasba,
        sources=extracted.nasba_uua_urls,
        additional_instruction=add_ins_nasba
    )

    # 8. CPE reporting period ends on June 30
    node_reporting_end = evaluator.add_leaf(
        id="CPE_Reporting_Period_End_June_30",
        desc="State's CPE reporting period ends on June 30th",
        parent=constraints_node,
        critical=True
    )
    claim_reporting_end = f"For {state_ref}, the CPE reporting period ends on June 30."
    add_ins_reporting_end = "Verify the end date of the CPE reporting period is June 30; the requirement may be stated as 'period ending June 30' or similar."
    await evaluator.verify(
        claim=claim_reporting_end,
        node=node_reporting_end,
        sources=extracted.reporting_end_urls,
        additional_instruction=add_ins_reporting_end
    )


async def _build_required_fields_checks(
    evaluator: Evaluator,
    parent_node,
    extracted: StateCPEExtraction
) -> None:
    """Build existence checks for required reported fields."""
    fields_node = evaluator.add_parallel(
        id="Required_Answer_Fields_Provided",
        desc="Answer includes all explicitly requested output fields",
        parent=parent_node,
        critical=True
    )

    # 1. Provides state name
    evaluator.add_custom_node(
        result=bool(extracted.state_name and extracted.state_name.strip()),
        id="Provides_State_Name",
        desc="Answer provides the name of the state",
        parent=fields_node,
        critical=True
    )

    # 2. Provides A&A minimum number
    evaluator.add_custom_node(
        result=_has_numberlike(extracted.aa_min_hours),
        id="Provides_Accounting_Auditing_Hours_Number",
        desc="Answer provides the specific numeric minimum number of Accounting and Auditing CPE hours required for the chosen state",
        parent=fields_node,
        critical=True
    )

    # 3. Provides behavioral max number
    evaluator.add_custom_node(
        result=_has_numberlike(extracted.behavioral_max_hours),
        id="Provides_Behavioral_Subject_Max_Number",
        desc="Answer provides the specific numeric maximum number of behavioral subject CPE hours allowed for the chosen state",
        parent=fields_node,
        critical=True
    )

    # 4. Provides supporting reference URLs (at least one per requirement category)
    all_sources_present = all([
        len(extracted.renewal_cycle_urls) > 0,
        len(extracted.cpe_total_urls) > 0,
        len(extracted.ethics_hours_urls) > 0,
        len(extracted.ethics_course_approval_urls) > 0,
        len(extracted.aa_min_urls) > 0,
        len(extracted.behavioral_max_urls) > 0,
        len(extracted.nasba_uua_urls) > 0,
        len(extracted.reporting_end_urls) > 0,
    ])
    evaluator.add_custom_node(
        result=all_sources_present,
        id="Provides_Supporting_Reference_URLs",
        desc="Answer provides supporting reference URL(s) that substantiate each stated requirement (i.e., the constraints and the reported A&A minimum and behavioral maximum values)",
        parent=fields_node,
        critical=True
    )


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
    Evaluate an answer for the CPA CPE/license renewal requirement matching task.
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
        default_model=model
    )

    # Top-level critical aggregation node mirroring rubric root
    state_main = evaluator.add_parallel(
        id="State_Identification_and_Reporting",
        desc="Identify a U.S. state meeting all specified CPA CPE/license renewal requirements and provide the requested supporting details and citations",
        parent=root,
        critical=True
    )

    # Extract structured content and sources
    extracted: StateCPEExtraction = await evaluator.extract(
        prompt=prompt_extract_state_cpe_info(),
        template_class=StateCPEExtraction,
        extraction_name="state_cpe_extraction"
    )

    # Build verification tree for constraints and required fields
    await _build_constraints_checks(evaluator, state_main, extracted)
    await _build_required_fields_checks(evaluator, state_main, extracted)

    # Return final summary
    return evaluator.get_summary()