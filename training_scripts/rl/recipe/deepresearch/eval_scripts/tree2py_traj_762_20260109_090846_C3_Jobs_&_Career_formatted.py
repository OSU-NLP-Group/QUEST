import asyncio
import logging
from typing import Any, Dict, List, Optional
from datetime import date
import calendar

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys


# --------------------------------------------------------------------------- #
# Task metadata                                                               #
# --------------------------------------------------------------------------- #
TASK_ID = "ca_cpa_first_renewal_ce_2024_2026"
TASK_DESCRIPTION = (
    "A California CPA received their initial license on September 1, 2024, with a license expiration date of "
    "January 31, 2026. They participate in attest engagements during this period. What are all the continuing "
    "education requirements they must complete for this first license renewal?"
)

ISSUANCE_DATE = date(2024, 9, 1)
EXPIRATION_DATE = date(2026, 1, 31)


# --------------------------------------------------------------------------- #
# Helpers for date math                                                       #
# --------------------------------------------------------------------------- #
def _last_day_of_month(year: int, month: int) -> int:
    return calendar.monthrange(year, month)[1]


def _add_months(dt: date, months: int) -> date:
    y = dt.year + (dt.month - 1 + months) // 12
    m = (dt.month - 1 + months) % 12 + 1
    d = min(dt.day, _last_day_of_month(y, m))
    return date(y, m, d)


def compute_full_six_month_periods(start: date, end: date) -> int:
    count = 0
    cur = start
    while True:
        next_point = _add_months(cur, 6)
        if next_point <= end:
            count += 1
            cur = next_point
        else:
            break
    return count


FULL_6MO_COUNT = compute_full_six_month_periods(ISSUANCE_DATE, EXPIRATION_DATE)
EXPECTED_TOTAL_HOURS = FULL_6MO_COUNT * 20  # As per rubric: 20 hours per full 6‑month period


# --------------------------------------------------------------------------- #
# Extraction models                                                           #
# --------------------------------------------------------------------------- #
class CERequirementsExtraction(BaseModel):
    total_hours: Optional[str] = None
    total_hours_sources: List[str] = Field(default_factory=list)

    technical_min_per_full_year: Optional[str] = None
    technical_sources: List[str] = Field(default_factory=list)

    regulatory_review_required: Optional[str] = None  # expected values like "required" or "not required"
    regulatory_review_hours: Optional[str] = None
    regulatory_sources: List[str] = Field(default_factory=list)

    ethics_applicability: Optional[str] = None  # "required"/"applies" vs "not required"/"does not apply"
    ethics_sources: List[str] = Field(default_factory=list)

    attest_aa_fraud_applicability: Optional[str] = None  # "applies"/"required" vs "does not apply"/"not required"
    attest_sources: List[str] = Field(default_factory=list)

    completion_window_statement: Optional[str] = None
    completion_window_sources: List[str] = Field(default_factory=list)

    carryover_statement: Optional[str] = None
    carryover_sources: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_ce_requirements() -> str:
    return (
        "Extract from the answer the specific continuing education (CE) requirements the CPA must complete for the "
        "first renewal, given the license issuance date 9/1/2024 and expiration date 1/31/2026. Only extract what the "
        "answer explicitly states.\n\n"
        "For each item below, extract the stated value and the URLs cited in the answer that support it:\n"
        "1) total_hours: The total CE hours required for this first renewal period.\n"
        "   – total_hours_sources: All URLs cited that support the total hours rule/amount.\n"
        "2) technical_min_per_full_year: The minimum technical subject matter hours required during each full year of the license period.\n"
        "   – technical_sources: URLs cited supporting this technical minimum.\n"
        "3) regulatory_review_required: Whether a Board-approved Regulatory Review course is required for the first renewal (use values like 'required' or 'not required').\n"
        "   – regulatory_review_hours: The number of hours for that course (e.g., '2 hours').\n"
        "   – regulatory_sources: URLs cited supporting the Regulatory Review requirement.\n"
        "4) ethics_applicability: Whether the 4-hour ethics education requirement applies to this first renewal (use values like 'required' or 'not required').\n"
        "   – ethics_sources: URLs cited supporting the ethics applicability.\n"
        "5) attest_aa_fraud_applicability: Whether the Accounting & Auditing (24 hours) plus fraud (4 hours) requirements apply to this first renewal (use values like 'applies/required' or 'does not apply/not required').\n"
        "   – attest_sources: URLs cited supporting this applicability.\n"
        "6) completion_window_statement: The statement about when CE must be completed (e.g., 'between issuance and expiration').\n"
        "   – completion_window_sources: URLs cited supporting the completion window.\n"
        "7) carryover_statement: The statement about CE carryover (e.g., 'no carryover permitted').\n"
        "   – carryover_sources: URLs cited supporting the carryover rule.\n\n"
        "Rules:\n"
        "- Extract only information explicitly present in the answer.\n"
        "- If the answer does not provide a value for a field, set it to null. If no URLs are provided, return an empty list for the corresponding sources field.\n"
        "- Extract actual URLs (including protocol), not just site names.\n"
    )


# --------------------------------------------------------------------------- #
# Small helpers for stance normalization                                      #
# --------------------------------------------------------------------------- #
def stance_is_required(text: Optional[str]) -> Optional[bool]:
    if text is None:
        return None
    t = text.strip().lower()
    positives = {"required", "applies", "does apply", "is required", "must complete", "must be completed"}
    negatives = {"not required", "does not apply", "is not required", "no", "not applicable"}
    if t in positives:
        return True
    if t in negatives:
        return False
    return None


def ensure_hours_phrase(h: Optional[str], default: str) -> str:
    if not h or not h.strip():
        return default
    s = h.strip().lower()
    if "hour" in s:
        return h.strip()
    # If it looks like a number, append "hours"
    try:
        _ = float(s.replace(" ", "").replace("hours", "").replace("hour", ""))
        return f"{s} hours".strip()
    except Exception:
        return h.strip()


# --------------------------------------------------------------------------- #
# Verification construction                                                   #
# --------------------------------------------------------------------------- #
async def build_and_verify_nodes(
    evaluator: Evaluator,
    main_node_parent,
    ex: CERequirementsExtraction,
) -> None:
    # Create main critical parallel node as per rubric
    ce_main = evaluator.add_parallel(
        id="Complete_Continuing_Education_Requirements",
        desc="Identify all continuing education (CE) requirements applicable to the CPA’s first renewal given the stated dates, attest participation, and the provided constraints.",
        parent=main_node_parent,
        critical=True,
    )

    # Leaf: Total CE Hours Required
    total_hours_node = evaluator.add_leaf(
        id="Total_CE_Hours_Required",
        desc="States the total CE hours required for the period using the rule: 20 hours per full 6-month period from issuance to expiration (applied to 9/1/2024–1/31/2026).",
        parent=ce_main,
        critical=True,
    )
    total_hours_value = ex.total_hours.strip() if ex.total_hours else None
    total_hours_claim_value = total_hours_value if total_hours_value else f"{EXPECTED_TOTAL_HOURS} hours"
    total_hours_claim = (
        f"For the first renewal period from September 1, 2024 to January 31, 2026, the total continuing education (CE) "
        f"hours required is {total_hours_claim_value}."
    )
    total_hours_add_ins = (
        "Apply California Board of Accountancy proration for first renewal: 20 hours per full 6‑month period only. "
        f"The period 9/1/2024–1/31/2026 contains {FULL_6MO_COUNT} full six‑month segments, for a total of {EXPECTED_TOTAL_HOURS} hours. "
        "Verify the stated total against the cited rules/pages."
    )
    await evaluator.verify(
        claim=total_hours_claim,
        node=total_hours_node,
        sources=ex.total_hours_sources,
        additional_instruction=total_hours_add_ins,
    )

    # Leaf: Technical Subject Minimum
    tech_node = evaluator.add_leaf(
        id="Technical_Subject_Minimum",
        desc="States the minimum technical subject matter hours required during each full year of the license period (apply to any full year contained in 9/1/2024–1/31/2026).",
        parent=ce_main,
        critical=True,
    )
    tech_val = ex.technical_min_per_full_year.strip() if ex.technical_min_per_full_year else None
    tech_claim_val = tech_val if tech_val else "the minimum technical subject matter hours specified by the Board per full year"
    tech_claim = (
        f"The minimum technical subject matter hours required during each full year within the license period is {tech_claim_val}."
    )
    tech_add_ins = (
        "This license period spans one full year (9/1/2024–8/31/2025) and a partial final 5‑month segment. "
        "Verify, from the cited sources, the Board's minimum technical subject matter hours required per full year."
    )
    await evaluator.verify(
        claim=tech_claim,
        node=tech_node,
        sources=ex.technical_sources,
        additional_instruction=tech_add_ins,
    )

    # Leaf: Regulatory Review Course
    rr_node = evaluator.add_leaf(
        id="Regulatory_Review_Course",
        desc="States that a 2-hour Board-approved Regulatory Review course is required for the first renewal because the license was issued on or after 7/1/2024.",
        parent=ce_main,
        critical=True,
    )
    rr_required = stance_is_required(ex.regulatory_review_required)
    rr_hours_phrase = ensure_hours_phrase(ex.regulatory_review_hours, "2 hours")
    rr_required_phrase = "is required" if rr_required is not False else "is not required"
    rr_claim = (
        f"A Board‑approved Regulatory Review course of {rr_hours_phrase} {rr_required_phrase} for the first renewal "
        "because the initial license date (September 1, 2024) is on or after July 1, 2024."
    )
    rr_add_ins = (
        "Confirm the introduction/effective date and requirement language for California CPA first renewal. "
        "Specifically check that a Board‑approved Regulatory Review course (2 hours) applies to first renewals for licenses "
        "issued on or after July 1, 2024."
    )
    await evaluator.verify(
        claim=rr_claim,
        node=rr_node,
        sources=ex.regulatory_sources,
        additional_instruction=rr_add_ins,
    )

    # Leaf: Ethics Requirement Applicability
    ethics_node = evaluator.add_leaf(
        id="Ethics_Requirement_Applicability",
        desc="Correctly determines whether the 4-hour ethics education requirement applies, using the given condition that it is only required when the license has been held for a full two years and 80 hours of CE is required.",
        parent=ce_main,
        critical=True,
    )
    ethics_required = stance_is_required(ex.ethics_applicability)
    ethics_phrase = "does apply" if ethics_required else "does not apply"
    ethics_claim = (
        f"For this first renewal (9/1/2024–1/31/2026), the 4‑hour ethics education requirement {ethics_phrase}. "
        "It only applies when the license has been held for a full two years and the total CE requirement is 80 hours."
    )
    ethics_add_ins = (
        f"This renewal is approximately 17 months; {FULL_6MO_COUNT} full six‑month segments, total {EXPECTED_TOTAL_HOURS} hours, not 80. "
        "Verify that per the Board's rules the 4‑hour ethics requirement applies only to full two‑year/80‑hour cycles, and thus does not apply here."
    )
    await evaluator.verify(
        claim=ethics_claim,
        node=ethics_node,
        sources=ex.ethics_sources,
        additional_instruction=ethics_add_ins,
    )

    # Leaf: Attest A&A + Fraud Applicability
    attest_node = evaluator.add_leaf(
        id="Attest_AA_Fraud_Applicability",
        desc="Correctly determines whether the accounting & auditing (24 hours) plus fraud (4 hours) requirements apply, using the given condition that they only apply when 80 total CE hours is required (and considering the user’s statement that the CPA participates in attest engagements).",
        parent=ce_main,
        critical=True,
    )
    attest_required = stance_is_required(ex.attest_aa_fraud_applicability)
    attest_phrase = "do apply" if attest_required else "do not apply"
    attest_claim = (
        f"The Accounting & Auditing (24 hours) and fraud (4 hours) requirements {attest_phrase} for this first renewal. "
        "They only apply when the total CE requirement is 80 hours (full cycle). Participation in attest engagements does not trigger these in a prorated first renewal cycle."
    )
    attest_add_ins = (
        f"Confirm from the Board's CE rules that A&A (24) and fraud (4) special requirements are tied to full 80‑hour cycles. "
        f"This first renewal is {EXPECTED_TOTAL_HOURS} hours, not 80. Even though the CPA participates in attest, verify that the A&A and fraud blocks do not apply in this prorated cycle."
    )
    await evaluator.verify(
        claim=attest_claim,
        node=attest_node,
        sources=ex.attest_sources,
        additional_instruction=attest_add_ins,
    )

    # Leaf: Completion Window
    window_node = evaluator.add_leaf(
        id="Completion_Window",
        desc="States that CE must be completed between the license issuance date and the first expiration date.",
        parent=ce_main,
        critical=True,
    )
    window_claim = (
        "Continuing education must be completed between the license issuance date (September 1, 2024) and the first expiration date (January 31, 2026)."
    )
    window_add_ins = (
        "Verify that CE must be earned within the license period window for first renewal—as defined by issuance to expiration."
    )
    await evaluator.verify(
        claim=window_claim,
        node=window_node,
        sources=ex.completion_window_sources,
        additional_instruction=window_add_ins,
    )

    # Leaf: No Carryover
    carry_node = evaluator.add_leaf(
        id="No_Carryover",
        desc="States that CE hours cannot be carried over from one renewal period to another.",
        parent=ce_main,
        critical=True,
    )
    carry_claim = "CE hours cannot be carried over from one renewal period to another."
    carry_add_ins = "Confirm the Board's rule that CE carryover is not allowed."
    await evaluator.verify(
        claim=carry_claim,
        node=carry_node,
        sources=ex.carryover_sources,
        additional_instruction=carry_add_ins,
    )


# --------------------------------------------------------------------------- #
# Main evaluation entry                                                       #
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

    # Extract structured CE requirements from the answer
    ex = await evaluator.extract(
        prompt=prompt_extract_ce_requirements(),
        template_class=CERequirementsExtraction,
        extraction_name="ce_requirements_extraction",
    )

    # Record ground truth contextual info (dates and computed expectation)
    evaluator.add_ground_truth(
        {
            "issuance_date": ISSUANCE_DATE.isoformat(),
            "expiration_date": EXPIRATION_DATE.isoformat(),
            "full_six_month_segments": FULL_6MO_COUNT,
            "expected_total_hours_by_rule": EXPECTED_TOTAL_HOURS,
            "notes": "Expected total hours are computed strictly per rubric: 20 hours per full six‑month period only."
        },
        gt_type="context_computation"
    )

    # Build verification nodes and run checks
    await build_and_verify_nodes(evaluator, root, ex)

    # Return final summary
    return evaluator.get_summary()