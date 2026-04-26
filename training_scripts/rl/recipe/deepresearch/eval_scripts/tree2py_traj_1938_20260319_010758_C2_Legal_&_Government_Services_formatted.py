import asyncio
import logging
import re
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "compliance_deadlines_2025"
TASK_DESCRIPTION = """
A U.S.-based company operates as a calendar-year C corporation with its fiscal year ending on December 31, 2025. The company is classified as a Large Accelerated Filer by the SEC and employs 250 workers at a single establishment.

Identify three specific federal regulatory filing deadlines that this company must meet in 2026 for compliance with regulations related to the 2025 fiscal/calendar year. For each deadline, provide:
1. The exact filing deadline date (month, day, and year)
2. The specific form name or filing system designation
3. A direct URL link to an official government source (such as SEC.gov, OSHA.gov, or IRS.gov) or an authoritative compliance publication that explicitly states or confirms this deadline

The three regulatory filing requirements you should identify are:
- The SEC annual report filing requirement for publicly traded companies
- The OSHA electronic injury and illness data submission requirement for establishments with 250+ employees
- The federal corporate income tax return filing requirement for C corporations
"""

FISCAL_YEAR_END = "December 31, 2025"
FILING_YEAR = 2026


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class SECInfo(BaseModel):
    form_name: Optional[str] = None
    deadline_date: Optional[str] = None
    source_urls: List[str] = Field(default_factory=list)


class OSHAInfo(BaseModel):
    submission_system: Optional[str] = None  # e.g., "OSHA Injury Tracking Application (ITA)" or "OSHA Form 300A electronic submission"
    deadline_date: Optional[str] = None
    source_urls: List[str] = Field(default_factory=list)


class IRSInfo(BaseModel):
    form_name: Optional[str] = None  # e.g., "Form 1120"
    deadline_date: Optional[str] = None
    source_urls: List[str] = Field(default_factory=list)


class ComplianceDeadlinesExtraction(BaseModel):
    sec: Optional[SECInfo] = None
    osha: Optional[OSHAInfo] = None
    tax: Optional[IRSInfo] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_compliance_deadlines() -> str:
    return f"""
    Extract the three required 2026 compliance deadlines related to the 2025 year for the specified U.S. company context:
    - A U.S. calendar-year C corporation (fiscal year ends on {FISCAL_YEAR_END})
    - Classified as a Large Accelerated Filer by the SEC
    - Employs 250 workers at a single establishment

    For each of the three required items below, extract the fields exactly as they appear in the answer:

    1) SEC annual report filing requirement for publicly traded companies
       - form_name: The name/designation of the SEC annual report form (e.g., "Form 10-K")
       - deadline_date: The specific deadline date in 2026 (include month, day, and 4-digit year, e.g., "March 1, 2026" or "2026-03-01")
       - source_urls: All URLs cited as official/authoritative references that state or confirm the deadline (SEC.gov or reputable compliance/law firm sources). Return an array of URLs.

    2) OSHA electronic injury and illness data submission for establishments with 250+ employees
       - submission_system: The name of the system or requirement (e.g., "OSHA Injury Tracking Application (ITA)" or "OSHA Form 300A electronic submission")
       - deadline_date: The specific deadline date in 2026 (include month, day, and 4-digit year, commonly March 2, 2026)
       - source_urls: All URLs cited as official/authoritative references that state or confirm the deadline (OSHA.gov or reputable compliance sources). Return an array of URLs.

    3) Federal corporate income tax return filing for C corporations
       - form_name: The IRS form name/designation (e.g., "Form 1120")
       - deadline_date: The specific deadline date in 2026 (include month, day, and 4-digit year, commonly April 15, 2026)
       - source_urls: All URLs cited as official/authoritative references that state or confirm the deadline (IRS.gov, IRS Publication 509, instructions, or reputable tax compliance sources). Return an array of URLs.

    Constraints and formatting:
    - If any field is not present in the answer, set it to null (for strings) or [] (for lists).
    - For deadline_date, prefer a full explicit date with month, day, and year (e.g., "March 2, 2026", "04/15/2026", or "2026-03-02").
    - Only include URLs that are explicitly present in the answer. Extract the actual links even if given in markdown format.
    - Return a single JSON object with the top-level keys: sec, osha, tax.
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
DATE_PATTERNS = [
    re.compile(r"\b(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|"
               r"Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\s+\d{{1,2}},\s*\d{{4}}\b",
               re.IGNORECASE),
    re.compile(r"\b\d{1,2}/\d{1,2}/\d{4}\b"),
    re.compile(r"\b\d{4}-\d{2}-\d{2}\b")
]


def has_full_date(s: Optional[str]) -> bool:
    if not s or not isinstance(s, str):
        return False
    for pat in DATE_PATTERNS:
        if pat.search(s):
            return True
    return False


def normalize_urls(urls: Optional[List[str]]) -> List[str]:
    if not urls:
        return []
    out = []
    seen = set()
    for u in urls:
        if not isinstance(u, str):
            continue
        u = u.strip()
        if not u:
            continue
        if not (u.startswith("http://") or u.startswith("https://")):
            u = "http://" + u
        if u not in seen:
            seen.add(u)
            out.append(u)
    return out


# --------------------------------------------------------------------------- #
# Verification builders                                                       #
# --------------------------------------------------------------------------- #
async def build_sec_verification(evaluator: Evaluator, parent) -> None:
    node = evaluator.add_parallel(
        id="SEC_Annual_Report_Requirement",
        desc="Identify the SEC annual report filing deadline applicable to Large Accelerated Filers for fiscal year ending December 31, 2025",
        parent=parent,
        critical=False
    )

    # Retrieve extracted info
    extraction: ComplianceDeadlinesExtraction = next(
        (info.get("deadlines_extraction") for info in evaluator.get_summary()["eval_breakdown"][0]["info"]
         if "deadlines_extraction" in info), None
    )
    # Fallback if above summary path not yet populated; use evaluator's internal extraction record access pattern
    # Safer approach: re-extract reference from evaluator by searching last recorded extraction
    # But the framework does not expose directly; Hence, we will pass in via closure in main to avoid this complexity.
    # We will instead get the latest extraction result through a captured variable.
    # This function will be redefined dynamically inside evaluate_answer with closure variables.
    pass  # Placeholder to satisfy linter (this function will be overridden in evaluate_answer)


async def build_osha_verification(evaluator: Evaluator, parent) -> None:
    pass  # Will be overridden in evaluate_answer (closure)


async def build_tax_verification(evaluator: Evaluator, parent) -> None:
    pass  # Will be overridden in evaluate_answer (closure)


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
    Evaluate an answer for the 2025-year compliance deadlines due in 2026.
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
        default_model=model,
    )

    # Extract structured info
    extracted: ComplianceDeadlinesExtraction = await evaluator.extract(
        prompt=prompt_extract_compliance_deadlines(),
        template_class=ComplianceDeadlinesExtraction,
        extraction_name="deadlines_extraction",
    )

    # Helper: local builders that can access "extracted" via closure
    async def _build_sec_verification():
        sec_node = evaluator.add_parallel(
            id="SEC_Annual_Report_Requirement",
            desc="Identify the SEC annual report filing deadline applicable to Large Accelerated Filers for fiscal year ending December 31, 2025",
            parent=root,
            critical=False
        )

        sec = extracted.sec or SECInfo()
        urls = normalize_urls(sec.source_urls)

        # 1) Form Identification (leaf, critical) — simple verification
        form_leaf = evaluator.add_leaf(
            id="SEC_Form_Identification",
            desc="The SEC annual report form applicable to public companies is correctly identified",
            parent=sec_node,
            critical=True
        )
        form_name = sec.form_name or ""
        form_claim = (
            f"The answer identifies the SEC annual report form as '{form_name}'. "
            f"This should be the correct annual report filing form for a U.S. domestic large accelerated filer (i.e., Form 10-K)."
        )
        await evaluator.verify(
            claim=form_claim,
            node=form_leaf,
            sources=None,  # Use general knowledge check; not strictly requiring sources for this identification leaf
            additional_instruction=(
                "Accept 'Form 10-K' or equivalent phrasing like 'Annual Report on Form 10-K' as correct for U.S. domestic issuers. "
                "Do not accept 'Form 10-Q' (quarterly), 'Form 20-F' (foreign private issuers), or 'Form 40-F' (Canadian issuers)."
            )
        )

        # 2) Deadline Provided (leaf, critical) — existence/format check
        deadline_ok = has_full_date(sec.deadline_date)
        evaluator.add_custom_node(
            result=deadline_ok,
            id="SEC_Deadline_Provided",
            desc="A specific deadline date for the SEC annual report filing is provided",
            parent=sec_node,
            critical=True
        )

        # 3) Official Reference (sequential critical group): URL exists + support by URLs
        official_seq = evaluator.add_sequential(
            id="SEC_Official_Reference",
            desc="An official reference URL is provided and supports the SEC annual report deadline",
            parent=sec_node,
            critical=True
        )
        # 3.1 existence
        evaluator.add_custom_node(
            result=len(urls) > 0,
            id="SEC_Official_Reference_URL_Provided",
            desc="At least one official/authoritative URL is provided to verify the SEC deadline",
            parent=official_seq,
            critical=True
        )
        # 3.2 support
        support_leaf = evaluator.add_leaf(
            id="SEC_Official_Reference_Support",
            desc="The provided source(s) support the SEC annual report deadline",
            parent=official_seq,
            critical=True
        )
        sec_support_claim = (
            f"For a Large Accelerated Filer with fiscal year ending {FISCAL_YEAR_END}, the due date to file the annual report "
            f"on {form_name or 'Form 10-K'} is {sec.deadline_date or '[DATE MISSING]'}. "
            f"It is acceptable if the source states the rule (e.g., '60 days after fiscal year end for Large Accelerated Filers') "
            f"from which this specific date follows, including standard weekend/holiday adjustments."
        )
        await evaluator.verify(
            claim=sec_support_claim,
            node=support_leaf,
            sources=urls,
            additional_instruction=(
                "Consider the claim supported if the page explicitly states either: "
                "(a) the specific date provided, or (b) the general timing rule (e.g., 60 days after FY end for Large Accelerated Filers) "
                "that leads to the stated date for a December 31, 2025 year-end. "
                "If URLs are irrelevant or do not mention the rule/date, mark as not supported."
            )
        )

    async def _build_osha_verification():
        osha_node = evaluator.add_parallel(
            id="OSHA_Electronic_Submission_Requirement",
            desc="Identify the OSHA electronic injury and illness data submission deadline for establishments with 250 or more employees for calendar year 2025 data",
            parent=root,
            critical=False
        )

        osha = extracted.osha or OSHAInfo()
        urls = normalize_urls(osha.source_urls)

        # 1) Submission System Identified (leaf, critical) — simple verification
        sys_leaf = evaluator.add_leaf(
            id="OSHA_Submission_System_Identified",
            desc="The OSHA electronic submission system or requirement for injury and illness data is correctly identified",
            parent=osha_node,
            critical=True
        )
        system_name = osha.submission_system or ""
        system_claim = (
            f"The answer identifies the OSHA electronic submission mechanism as '{system_name}', "
            f"which should correspond to OSHA's Injury Tracking Application (ITA) for electronic submission of injury and illness data "
            f"(commonly Form 300A summary for large establishments)."
        )
        await evaluator.verify(
            claim=system_claim,
            node=sys_leaf,
            sources=None,
            additional_instruction=(
                "Accept 'OSHA Injury Tracking Application', 'OSHA ITA', or 'electronic submission of OSHA Form 300A' as correct. "
                "The key is that the mechanism is OSHA's ITA system for annual injury/illness data."
            )
        )

        # 2) Deadline Provided (leaf, critical) — existence/format check
        deadline_ok = has_full_date(osha.deadline_date)
        evaluator.add_custom_node(
            result=deadline_ok,
            id="OSHA_Deadline_Provided",
            desc="A specific deadline date for the OSHA electronic submission is provided",
            parent=osha_node,
            critical=True
        )

        # 3) Official Reference (sequential critical group): URL exists + support by URLs
        official_seq = evaluator.add_sequential(
            id="OSHA_Official_Reference",
            desc="An official reference URL is provided and supports the OSHA submission deadline",
            parent=osha_node,
            critical=True
        )
        # 3.1 existence
        evaluator.add_custom_node(
            result=len(urls) > 0,
            id="OSHA_Official_Reference_URL_Provided",
            desc="At least one official/authoritative URL is provided to verify the OSHA deadline",
            parent=official_seq,
            critical=True
        )
        # 3.2 support
        support_leaf = evaluator.add_leaf(
            id="OSHA_Official_Reference_Support",
            desc="The provided source(s) support the OSHA electronic submission deadline",
            parent=official_seq,
            critical=True
        )
        osha_support_claim = (
            f"For an establishment with 250 or more employees, the deadline to electronically submit CY 2025 injury and illness "
            f"summary data via OSHA's ITA is {osha.deadline_date or '[DATE MISSING]'}. "
            f"It is acceptable if the source states 'by March 2' each year for the prior calendar year data, which implies March 2, {FILING_YEAR} for 2025 data."
        )
        await evaluator.verify(
            claim=osha_support_claim,
            node=support_leaf,
            sources=urls,
            additional_instruction=(
                "Consider the claim supported if the page explicitly states the date (e.g., March 2, 2026) or the general rule "
                "('by March 2' for the prior year's data). If URLs do not mention the rule/date, mark as not supported."
            )
        )

    async def _build_tax_verification():
        tax_node = evaluator.add_parallel(
            id="Federal_Corporate_Tax_Requirement",
            desc="Identify the federal corporate income tax return filing deadline for calendar-year C corporations with fiscal year ending December 31, 2025",
            parent=root,
            critical=False
        )

        tax = extracted.tax or IRSInfo()
        urls = normalize_urls(tax.source_urls)

        # 1) Tax Form Identification (leaf, critical) — simple verification
        form_leaf = evaluator.add_leaf(
            id="Tax_Form_Identification",
            desc="The federal corporate income tax return form applicable to C corporations is correctly identified",
            parent=tax_node,
            critical=True
        )
        tax_form = tax.form_name or ""
        tax_form_claim = (
            f"The answer identifies the corporate income tax return form as '{tax_form}', "
            f"which should be the correct form for U.S. C corporations (i.e., Form 1120)."
        )
        await evaluator.verify(
            claim=tax_form_claim,
            node=form_leaf,
            sources=None,
            additional_instruction=(
                "Accept 'Form 1120' (U.S. Corporation Income Tax Return) as correct for C corporations. "
                "Do not accept partnership (Form 1065) or S corporation (Form 1120-S) forms."
            )
        )

        # 2) Deadline Provided (leaf, critical) — existence/format check
        deadline_ok = has_full_date(tax.deadline_date)
        evaluator.add_custom_node(
            result=deadline_ok,
            id="Tax_Deadline_Provided",
            desc="A specific deadline date for the corporate income tax return filing is provided",
            parent=tax_node,
            critical=True
        )

        # 3) Official Reference (sequential critical group): URL exists + support by URLs
        official_seq = evaluator.add_sequential(
            id="IRS_Official_Reference",
            desc="An official reference URL is provided and supports the corporate income tax return deadline",
            parent=tax_node,
            critical=True
        )
        # 3.1 existence
        evaluator.add_custom_node(
            result=len(urls) > 0,
            id="IRS_Official_Reference_URL_Provided",
            desc="At least one official/authoritative URL is provided to verify the tax deadline",
            parent=official_seq,
            critical=True
        )
        # 3.2 support
        support_leaf = evaluator.add_leaf(
            id="IRS_Official_Reference_Support",
            desc="The provided source(s) support the corporate income tax return deadline",
            parent=official_seq,
            critical=True
        )
        tax_support_claim = (
            f"A calendar-year C corporation with tax year ending {FISCAL_YEAR_END} must file {tax_form or 'Form 1120'} by {tax.deadline_date or '[DATE MISSING]'}. "
            f"It is acceptable if the source states the general rule that the return is due by the 15th day of the 4th month after the end of the tax year, "
            f"which would be April 15, {FILING_YEAR} for a December 31, 2025 year-end, allowing weekend/holiday adjustments."
        )
        await evaluator.verify(
            claim=tax_support_claim,
            node=support_leaf,
            sources=urls,
            additional_instruction=(
                "Consider the claim supported if the page explicitly states the specific date or the general rule "
                "('15th day of the 4th month after year-end') that yields the provided date. "
                "If URLs are irrelevant or do not mention the rule/date, mark as not supported."
            )
        )

    # Build all three requirement groups
    await _build_sec_verification()
    await _build_osha_verification()
    await _build_tax_verification()

    # Return structured result
    return evaluator.get_summary()