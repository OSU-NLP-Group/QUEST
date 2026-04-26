import asyncio
import logging
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from obj_task_eval.llm_client.base_client import LLMClient
from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy

# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "ca_pain_cme_option1_march2024"
TASK_DESCRIPTION = (
    "A California physician who obtained their initial Physician and Surgeon (P&S) license in March 2024 is not a pathologist or radiologist. "
    "They must complete the one-time pain management CME course requirement. Determine: "
    "(1) by which license renewal deadline they must complete the requirement if choosing Option 1 (pain management/terminally ill treatment), "
    "(2) the total number of CME hours required, "
    "(3) what accrediting organizations are acceptable for course providers, "
    "(4) the minimum retention period for CME documentation, "
    "and (5) what professional conduct violation occurs if they certify completion but cannot provide documentation when audited."
)

# Ground truth information (reference expectations; used only for summary)
EXPECTED_TOTAL_HOURS = "12"
EXPECTED_ORGS = ["ACCME", "AMA", "CMA", "AAFP"]

# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class DeadlineInfo(BaseModel):
    """Structured information about the Option 1 deadline."""
    rule_text: Optional[str] = None
    rule_sources: List[str] = Field(default_factory=list)
    applied_text: Optional[str] = None
    applied_month_year: Optional[str] = None
    applied_sources: List[str] = Field(default_factory=list)


class CMEComplianceExtraction(BaseModel):
    """All fields to extract from the agent's answer for verification."""
    eligibility_context: Optional[str] = None
    eligibility_sources: List[str] = Field(default_factory=list)

    deadline: DeadlineInfo = Field(default_factory=DeadlineInfo)

    total_cme_hours: Optional[str] = None
    total_cme_hours_sources: List[str] = Field(default_factory=list)

    accrediting_orgs: List[str] = Field(default_factory=list)
    accrediting_sources: List[str] = Field(default_factory=list)

    retention_period: Optional[str] = None
    retention_sources: List[str] = Field(default_factory=list)

    violation_label: Optional[str] = None
    violation_sources: List[str] = Field(default_factory=list)

    discipline_note: Optional[str] = None
    discipline_sources: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_cme_option1() -> str:
    return (
        "Extract the specific compliance details for the California one-time pain management CME (Option 1) from the answer. "
        "Return a JSON object with the following fields:\n"
        "1) eligibility_context: Summarize whether the physician (not a pathologist or radiologist) is subject to the one-time pain management CME requirement.\n"
        "2) eligibility_sources: List of URLs cited in the answer specifically supporting eligibility or applicability.\n"
        "3) deadline.rule_text: The rule for when Option 1 must be completed (e.g., 'by the second license renewal date after initial licensure').\n"
        "4) deadline.rule_sources: URLs cited supporting the rule.\n"
        "5) deadline.applied_text: The answer's explicit application of the deadline to a March 2024 initial license (e.g., a month/year or clear timing phrase).\n"
        "6) deadline.applied_month_year: If the answer provides a specific month/year for the second renewal deadline relative to March 2024, extract it verbatim (e.g., 'May 2028'); otherwise null.\n"
        "7) deadline.applied_sources: URLs cited supporting the renewal-cycle timing application.\n"
        "8) total_cme_hours: The stated total number of CME hours for Option 1 (e.g., '12'). Extract exactly as written (string).\n"
        "9) total_cme_hours_sources: URLs cited supporting the total hours.\n"
        "10) accrediting_orgs: The list of accrediting organizations the answer claims are acceptable for course providers (e.g., ['ACCME','AMA','CMA','AAFP']).\n"
        "11) accrediting_sources: URLs cited supporting these accrediting organizations.\n"
        "12) retention_period: The minimum required retention period for CME documentation (e.g., '4 years'). Extract exactly as written (string).\n"
        "13) retention_sources: URLs cited supporting the retention period.\n"
        "14) violation_label: The professional conduct violation label if a physician certifies completion but cannot provide documentation when audited (e.g., 'unprofessional conduct').\n"
        "15) violation_sources: URLs cited supporting that violation characterization.\n"
        "16) discipline_note: Any optional note in the answer about potential disciplinary outcomes (e.g., 'citation and fine'). If none, return null.\n"
        "17) discipline_sources: URLs cited supporting the optional disciplinary action note.\n\n"
        "IMPORTANT:\n"
        "- Extract only facts explicitly presented in the answer text. Do not invent information.\n"
        "- For any item not present, return null (for scalars) or an empty list (for arrays).\n"
        "- For all 'sources' fields, extract actual URLs the answer cites (including markdown links). If none cited, return an empty list.\n"
        "- Preserve the exact wording for scalar string fields as they appear in the answer."
    )


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def _add_months(year: int, month: int, months_to_add: int) -> (int, int):
    """Add months_to_add to (year, month). month is 1-12."""
    total = (year * 12 + (month - 1)) + months_to_add
    new_year = total // 12
    new_month = (total % 12) + 1
    return new_year, new_month


def _month_name(month: int) -> str:
    names = [
        "January", "February", "March", "April", "May", "June",
        "July", "August", "September", "October", "November", "December"
    ]
    return names[month - 1]


def expected_second_renewal_for_march_2024() -> str:
    """
    Compute the expected second renewal month/year given:
    - Initial license month/year: March 2024
    - Initial license valid ~26 months
    - Subsequent renewals are 24-month cycles
    Returns a string like 'May 2028'.
    """
    # Initial issuance: March 2024
    init_year, init_month = 2024, 3
    # First renewal ≈ init + 26 months
    first_year, first_month = _add_months(init_year, init_month, 26)
    # Second renewal ≈ first + 24 months
    second_year, second_month = _add_months(first_year, first_month, 24)
    return f"{_month_name(second_month)} {second_year}"


# --------------------------------------------------------------------------- #
# Verification subroutines                                                    #
# --------------------------------------------------------------------------- #
async def verify_eligibility(evaluator: Evaluator, parent_node, ex: CMEComplianceExtraction) -> None:
    node = evaluator.add_leaf(
        id="Eligibility_Context",
        desc="Recognize that the physician (not a pathologist/radiologist) is subject to the one-time pain management CME requirement",
        parent=parent_node,
        critical=False,
    )
    claim = (
        "The answer explicitly recognizes that this physician (not a pathologist or radiologist) is subject to the one-time pain management CME requirement."
    )
    await evaluator.verify(
        claim=claim,
        node=node,
        sources=ex.eligibility_sources if ex.eligibility_sources else None,
        additional_instruction=(
            "Judge whether the answer clearly states applicability of the one-time pain management CME for this physician. "
            "If the answer equivocates or omits this, mark as incorrect."
        ),
    )


async def verify_deadline(evaluator: Evaluator, parent_node, ex: CMEComplianceExtraction) -> None:
    # NOTE: The rubric marks Deadline as critical; we keep children critical and use a sequential aggregator.
    deadline_node = evaluator.add_sequential(
        id="Deadline",
        desc="State by which renewal deadline Option 1 must be completed for a March 2024 initial licensee",
        parent=parent_node,
        critical=True,
    )

    # 1) Rule: by second license renewal date after initial licensure
    rule_leaf = evaluator.add_leaf(
        id="Deadline_Rule",
        desc="Specify that Option 1 must be completed by the physician’s second license renewal date after initial licensure",
        parent=deadline_node,
        critical=True,
    )
    rule_claim = (
        "For Option 1 (pain management/terminally ill treatment), the requirement must be completed by the physician’s second license renewal date after initial licensure."
    )
    await evaluator.verify(
        claim=rule_claim,
        node=rule_leaf,
        sources=ex.deadline.rule_sources if ex.deadline.rule_sources else None,
        additional_instruction=(
            "Verify the second-renewal deadline rule for the one-time Option 1 requirement using the cited sources. "
            "Allow equivalent phrasing that unambiguously means the second renewal after the initial license."
        ),
    )

    # 2) Applied to March 2024: month/year (or equivalent clear timing)
    applied_leaf = evaluator.add_leaf(
        id="Deadline_Applied_To_March_2024",
        desc="Apply the given renewal-cycle constraints (initial license valid 26 months; subsequent renewals are 24-month cycles) to express the second-renewal deadline relative to March 2024 initial licensure (e.g., month/year or equivalent clear timing)",
        parent=deadline_node,
        critical=True,
    )

    expected_deadline = expected_second_renewal_for_march_2024()  # e.g., "May 2028"
    stated_applied = ex.deadline.applied_month_year or ex.deadline.applied_text or "N/A"

    applied_claim = (
        f"Given initial licensure in March 2024 and the constraints (initial ~26 months, then 24-month renewals), "
        f"the second renewal deadline would be around {expected_deadline}. "
        f"The answer’s stated timing ('{stated_applied}') correctly reflects this second-renewal timing."
    )

    await evaluator.verify(
        claim=applied_claim,
        node=applied_leaf,
        sources=ex.deadline.applied_sources if ex.deadline.applied_sources else None,
        additional_instruction=(
            "Focus on whether the answer correctly applies the provided cycle lengths to March 2024. "
            "Accept reasonable approximations and equivalent timing descriptions that match the expected second renewal (e.g., same month/year or clearly equivalent phrasing)."
        ),
    )


async def verify_total_hours(evaluator: Evaluator, parent_node, ex: CMEComplianceExtraction) -> None:
    node = evaluator.add_leaf(
        id="Total_CME_Hours",
        desc="Specify that the one-time Option 1 requirement totals 12 CME hours",
        parent=parent_node,
        critical=True,
    )
    claim = "The one-time Option 1 (pain management/terminally ill treatment) requirement totals 12 CME hours."
    await evaluator.verify(
        claim=claim,
        node=node,
        sources=ex.total_cme_hours_sources if ex.total_cme_hours_sources else None,
        additional_instruction=(
            "Treat '12 units' or '12 hours' equivalently as 12 CME hours for Option 1. Use the cited sources."
        ),
    )


async def verify_accrediting_orgs(evaluator: Evaluator, parent_node, ex: CMEComplianceExtraction) -> None:
    node = evaluator.add_leaf(
        id="Acceptable_Accrediting_Organizations",
        desc="List the acceptable accrediting organizations for course providers: ACCME, AMA, CMA, and AAFP",
        parent=parent_node,
        critical=True,
    )
    claim = (
        "Acceptable accrediting organizations for course providers include ACCME (Accreditation Council for Continuing Medical Education), "
        "AMA (American Medical Association—for AMA PRA Category 1 Credit), CMA (California Medical Association), and AAFP (American Academy of Family Physicians)."
    )
    await evaluator.verify(
        claim=claim,
        node=node,
        sources=ex.accrediting_sources if ex.accrediting_sources else None,
        additional_instruction=(
            "Verify that these organizations are acceptable for course providers according to the cited sources. "
            "Allow equivalent naming (full names vs acronyms)."
        ),
    )


async def verify_retention(evaluator: Evaluator, parent_node, ex: CMEComplianceExtraction) -> None:
    node = evaluator.add_leaf(
        id="CME_Documentation_Retention",
        desc="Specify the minimum required retention period for CME documentation",
        parent=parent_node,
        critical=True,
    )
    retention_value = ex.retention_period or "N/A"
    claim = f"The minimum required retention period for CME documentation is {retention_value}."
    await evaluator.verify(
        claim=claim,
        node=node,
        sources=ex.retention_sources if ex.retention_sources else None,
        additional_instruction=(
            "Verify the minimum retention timeframe required by California authorities (e.g., Medical Board of California). "
            "If the answer’s value is not supported by the sources, mark incorrect."
        ),
    )


async def verify_violation(evaluator: Evaluator, parent_node, ex: CMEComplianceExtraction) -> None:
    # NOTE: The rubric marks this parent critical but includes a non-critical child. To satisfy framework constraints
    # (critical parents cannot have non-critical children), we set this parent as non-critical, while keeping the core
    # violation label leaf critical.
    audit_node = evaluator.add_parallel(
        id="Audit_Noncompliance_Conduct_Violation",
        desc="Identify the professional conduct violation if the physician certifies compliance but cannot provide documentation when audited",
        parent=parent_node,
        critical=False,
    )

    # Core violation label (critical)
    violation_leaf = evaluator.add_leaf(
        id="Violation_Label",
        desc="State that failure to provide verification after certifying compliance constitutes unprofessional conduct",
        parent=audit_node,
        critical=True,
    )
    claim = (
        "If a physician certifies completion of CME but cannot provide documentation when audited, this constitutes unprofessional conduct."
    )
    await evaluator.verify(
        claim=claim,
        node=violation_leaf,
        sources=ex.violation_sources if ex.violation_sources else None,
        additional_instruction=(
            "Use California regulatory sources (e.g., MBC) to verify that certifying compliance without documentation constitutes unprofessional conduct."
        ),
    )

    # Optional disciplinary note (non-critical)
    optional_leaf = evaluator.add_leaf(
        id="Possible_Disciplinary_Action",
        desc="Optionally note that this may result in disciplinary action (e.g., citation and fine)",
        parent=audit_node,
        critical=False,
    )
    discipline_text = ex.discipline_note or "N/A"
    claim_opt = (
        f"The answer optionally notes that this may result in disciplinary action (e.g., citation and fine). "
        f"Stated: '{discipline_text}'."
    )
    await evaluator.verify(
        claim=claim_opt,
        node=optional_leaf,
        sources=ex.discipline_sources if ex.discipline_sources else None,
        additional_instruction=(
            "Verify whether the answer’s optional disciplinary note aligns with California board practices (e.g., citation and fine). "
            "If no source support is provided, mark as incorrect but non-critical."
        ),
    )


# --------------------------------------------------------------------------- #
# Main evaluation entry point                                                 #
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
) -> Dict[str, Any]:
    """
    Evaluate an answer for the California one-time pain management CME (Option 1) compliance details
    for a March 2024 initial licensee.
    """
    # Initialize evaluator (root is always non-critical by design in framework)
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

    # Extract structured information from the answer
    ex: CMEComplianceExtraction = await evaluator.extract(
        prompt=prompt_extract_cme_option1(),
        template_class=CMEComplianceExtraction,
        extraction_name="cme_option1_extraction",
    )

    # Add ground truth info for summary (non-binding)
    evaluator.add_ground_truth(
        {
            "expected_second_renewal_for_march_2024": expected_second_renewal_for_march_2024(),
            "expected_total_cme_hours_option1": EXPECTED_TOTAL_HOURS,
            "expected_accrediting_orgs": EXPECTED_ORGS,
        },
        gt_type="reference_expectations",
    )

    # Build and verify according to rubric tree
    await verify_eligibility(evaluator, root, ex)
    await verify_deadline(evaluator, root, ex)
    await verify_total_hours(evaluator, root, ex)
    await verify_accrediting_orgs(evaluator, root, ex)
    await verify_retention(evaluator, root, ex)
    await verify_violation(evaluator, root, ex)

    # Return structured summary with verification tree and scores
    return evaluator.get_summary()