import asyncio
import logging
import re
from datetime import datetime
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy

# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "fr_major_rule_2024"
TASK_DESCRIPTION = (
    "Identify one major final rule that was published in the Federal Register in 2024, had an estimated annual "
    "economic effect of $200 million or more, and went through the standard notice-and-comment rulemaking process. "
    "For this rule, provide: (1) The complete Federal Register citation (including volume, number, date, and page) "
    "and a working URL to the final rule, (2) The complete Federal Register citation and a working URL to the "
    "corresponding Notice of Proposed Rulemaking (NPRM) that preceded the final rule, (3) The public comment period "
    "start date (the NPRM publication date in the Federal Register) and the comment period end date (the deadline "
    "stated in the NPRM for submitting comments), and (4) The calculated duration of the public comment period in "
    "calendar days, and verification that it met the typical minimum standard of at least 30 calendar days."
)

# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class FRDoc(BaseModel):
    """Federal Register document information."""
    title: Optional[str] = None
    url: Optional[str] = None
    citation_text: Optional[str] = None  # e.g., "89 FR 12345 (June 15, 2024)" or "Vol. 89, No. 200 (Oct 15, 2024), pp. 12345-12360"
    volume: Optional[str] = None         # e.g., "89"
    number: Optional[str] = None         # e.g., "No. 200"
    date_text: Optional[str] = None      # e.g., "June 15, 2024"
    page_text: Optional[str] = None      # e.g., "12345" or "12345-12360"


class EconomicsEvidence(BaseModel):
    """Evidence and sources supporting major rule classification (≥ $200M)."""
    major_claim_text: Optional[str] = None  # e.g., "This is a major rule" or "economically significant"
    economic_effect_amount_text: Optional[str] = None  # e.g., "$220 million annually"
    sources: List[str] = Field(default_factory=list)   # URLs explicitly cited in the answer supporting major classification


class CommentPeriodInfo(BaseModel):
    """Public comment period dates and duration from the NPRM."""
    start_date_text: Optional[str] = None    # NPRM publication date (start of comment period)
    end_date_text: Optional[str] = None      # Comment deadline stated in NPRM
    duration_days_text: Optional[str] = None # e.g., "45 days", "30 calendar days", "60"


class RulePackageExtraction(BaseModel):
    """Aggregate extraction for the selected rule package."""
    final_rule: Optional[FRDoc] = None
    nprm: Optional[FRDoc] = None
    economics: Optional[EconomicsEvidence] = None
    comments: Optional[CommentPeriodInfo] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_rule_package() -> str:
    return """
    Extract from the answer all information needed to evaluate a single chosen rule package that meets the prompt.
    Return a JSON object with the following fields:

    - final_rule:
        - title: The title of the final rule (as stated in the answer).
        - url: The working URL to the final rule page (prefer FederalRegister.gov).
        - citation_text: The complete Federal Register citation string for the final rule provided in the answer.
        - volume: The Federal Register volume number if explicitly included (string).
        - number: The Federal Register issue number if explicitly included (string), often styled as "No. X".
        - date_text: The publication date of the final rule, as stated in the citation text or answer (string).
        - page_text: The page number or range included in the citation (string).

    - nprm:
        - title: The NPRM title (if stated in the answer, otherwise null).
        - url: The working URL to the NPRM page (prefer FederalRegister.gov).
        - citation_text: The complete Federal Register citation string for the NPRM provided in the answer.
        - volume: The Federal Register volume number if explicitly included (string).
        - number: The issue number if explicitly included (string).
        - date_text: The NPRM publication date (this is the comment period start date per the prompt; string).
        - page_text: The page number or range included in the citation (string).

    - economics:
        - major_claim_text: Any text in the answer claiming the rule is "major" or meets/exceeds the ≥ $200 million annual effect threshold.
        - economic_effect_amount_text: Any annual economic effect estimate stated in the answer (string).
        - sources: An array of URLs explicitly cited in the answer that support the major rule classification or economic effect (include agency RIA links, OMB pages, etc. if present).

    - comments:
        - start_date_text: The NPRM publication date as stated in the answer (string), which the prompt defines as the comment period start date.
        - end_date_text: The stated comment deadline from the NPRM (string).
        - duration_days_text: The answer’s stated computed duration in calendar days (e.g., "45 days" or "45") (string).

    GENERAL RULES:
    - Extract exactly what appears in the answer. Do not invent or infer missing values.
    - If any requested field is missing from the answer, set it to null (for strings) or [] (for arrays).
    - For URLs: extract only actual URLs explicitly present; include full URLs (prepend http:// if missing protocol).
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
DATE_FORMATS = [
    "%B %d, %Y",    # January 15, 2024
    "%b %d, %Y",    # Jan 15, 2024
    "%Y-%m-%d",     # 2024-01-15
    "%m/%d/%Y",     # 01/15/2024
    "%m-%d-%Y",     # 01-15-2024
    "%d %B %Y",     # 15 January 2024
    "%d %b %Y",     # 15 Jan 2024
    "%Y/%m/%d",     # 2024/01/15
]


def parse_date(text: Optional[str]) -> Optional[datetime]:
    if not text:
        return None
    t = text.strip()
    # Common clean-ups
    t = re.sub(r"^Published[:\s]+", "", t, flags=re.I).strip()
    t = re.sub(r"^Comments\s+Close[:\s]+", "", t, flags=re.I).strip()
    t = re.sub(r"[\(\)]", "", t).strip()

    for fmt in DATE_FORMATS:
        try:
            return datetime.strptime(t, fmt)
        except Exception:
            continue

    # Try to normalize months like "Sept." to "Sep"
    t2 = re.sub(r"\bSept\.\b", "Sep", t)
    if t2 != t:
        for fmt in DATE_FORMATS:
            try:
                return datetime.strptime(t2, fmt)
            except Exception:
                continue

    return None


def parse_int_from_text(text: Optional[str]) -> Optional[int]:
    if not text:
        return None
    m = re.search(r"(\d{1,4})", text)
    if m:
        try:
            return int(m.group(1))
        except Exception:
            return None
    return None


def sanitize_urls(urls: List[Optional[str]]) -> List[str]:
    return [u.strip() for u in urls if isinstance(u, str) and u.strip() != ""]


def compute_duration_days(start_text: Optional[str], end_text: Optional[str]) -> Optional[int]:
    start_dt = parse_date(start_text)
    end_dt = parse_date(end_text)
    if not start_dt or not end_dt:
        return None
    delta = (end_dt - start_dt).days
    return delta if delta >= 0 else None


# --------------------------------------------------------------------------- #
# Verification tree construction and checks                                   #
# --------------------------------------------------------------------------- #
async def build_and_verify(
    evaluator: Evaluator,
    extraction: RulePackageExtraction
) -> None:
    """
    Build the verification tree according to the rubric and run checks.
    """
    # Create the overall critical sequential node (acts as root for the rubric tree)
    reg_root = evaluator.add_sequential(
        id="Regulatory_Process_Verification",
        desc="Verify that the response identifies an eligible major final rule and provides all required Federal Register documentation and comment-period calculations.",
        parent=evaluator.root,
        critical=True
    )

    # ------------------------------------------------------------------ #
    # Step 1: Rule Eligibility Verification (parallel, critical)         #
    # ------------------------------------------------------------------ #
    step1 = evaluator.add_parallel(
        id="Step_1_Rule_Eligibility_Verification",
        desc="Verify the chosen rule satisfies all eligibility constraints (final rule in 2024 FR, major ≥$200M annual effect, APA notice-and-comment).",
        parent=reg_root,
        critical=True
    )

    final_rule_url = extraction.final_rule.url if extraction.final_rule else None
    nprm_url = extraction.nprm.url if extraction.nprm else None
    econ_sources = extraction.economics.sources if extraction.economics else []
    maj_sources = sanitize_urls(([final_rule_url] if final_rule_url else []) + econ_sources)

    # 1.1 Final rule published in 2024 (verify via FR final rule page)
    leaf_final_2024 = evaluator.add_leaf(
        id="Final_Rule_Published_In_2024",
        desc="The identified rule is a final rule published in the Federal Register in 2024.",
        parent=step1,
        critical=True
    )
    claim_final_2024 = (
        "This Federal Register page is a final rule and its publication year is 2024."
    )
    await evaluator.verify(
        claim=claim_final_2024,
        node=leaf_final_2024,
        sources=final_rule_url,
        additional_instruction="Verify that the document type indicates 'Final rule' (or equivalent) and that the FR page shows a 2024 publication date."
    )

    # 1.2 Major rule economic effect ≥ $200M (verify via provided sources)
    leaf_major_rule = evaluator.add_leaf(
        id="Major_Rule_Economic_Effect_At_Least_200M",
        desc="The response provides evidence that the rule is classified as a major rule with estimated annual economic effect ≥ $200 million (EO 14094 threshold).",
        parent=step1,
        critical=True
    )
    claim_major_rule = (
        "This rule is a 'major rule' (or otherwise shows an estimated annual economic effect of at least $200 million)."
    )
    await evaluator.verify(
        claim=claim_major_rule,
        node=leaf_major_rule,
        sources=maj_sources if maj_sources else None,
        additional_instruction="Check the cited pages for explicit statements of 'major rule' classification or evidence the annual economic effect meets/exceeds $200 million."
    )

    # 1.3 APA notice-and-comment process via NPRM (verify using NPRM and final rule)
    leaf_apa = evaluator.add_leaf(
        id="APA_Notice_And_Comment_Process",
        desc="The response indicates the rule went through standard APA §553 notice-and-comment rulemaking (i.e., it was proposed via NPRM and finalized after receiving comments).",
        parent=step1,
        critical=True
    )
    claim_apa = (
        "This rule was proposed via an NPRM and later finalized after receiving public comments under APA notice-and-comment procedures."
    )
    await evaluator.verify(
        claim=claim_apa,
        node=leaf_apa,
        sources=sanitize_urls([nprm_url, final_rule_url]),
        additional_instruction="Confirm the NPRM preceded the final rule and that the process reflects standard APA notice-and-comment rulemaking."
    )

    # ------------------------------------------------------------------ #
    # Step 2: Final Rule Documentation (parallel, critical)              #
    # ------------------------------------------------------------------ #
    step2 = evaluator.add_parallel(
        id="Step_2_Final_Rule_Documentation",
        desc="Provide the required Federal Register citation and working URL for the final rule.",
        parent=reg_root,
        critical=True
    )

    # 2.1 Final Rule Citation Complete (simple verification against answer content)
    leaf_final_cite = evaluator.add_leaf(
        id="Final_Rule_Citation_Complete",
        desc="Provide the complete Federal Register citation for the final rule, including volume, number, date, and page.",
        parent=step2,
        critical=True
    )
    final_citation = extraction.final_rule.citation_text if extraction.final_rule else None
    claim_final_cite = (
        f"The answer includes a complete Federal Register citation for the final rule with volume, number, date, and page: '{final_citation}'."
    )
    await evaluator.verify(
        claim=claim_final_cite,
        node=leaf_final_cite,
        additional_instruction="Look only at the provided answer text. Confirm the final rule citation contains all of: volume, issue number, publication date, and page (or page range)."
    )

    # 2.2 Final Rule URL Working (verify via URL)
    leaf_final_url = evaluator.add_leaf(
        id="Final_Rule_URL_Working",
        desc="Provide a publicly accessible, working URL to the final rule on FederalRegister.gov (or equivalent official Federal Register page).",
        parent=step2,
        critical=True
    )
    claim_final_url = "This URL leads to the official Federal Register page for the final rule and is accessible."
    await evaluator.verify(
        claim=claim_final_url,
        node=leaf_final_url,
        sources=final_rule_url,
        additional_instruction="Verify the URL loads and clearly corresponds to the final rule page with Federal Register citation details."
    )

    # ------------------------------------------------------------------ #
    # Step 3: NPRM Documentation and Linkage (parallel, critical)        #
    # ------------------------------------------------------------------ #
    step3 = evaluator.add_parallel(
        id="Step_3_NPRM_Documentation_And_Linkage",
        desc="Provide the NPRM citation/URL and verify it corresponds to (and precedes) the final rule.",
        parent=reg_root,
        critical=True
    )

    # 3.1 NPRM Citation Complete (simple verification)
    leaf_nprm_cite = evaluator.add_leaf(
        id="NPRM_Citation_Complete",
        desc="Provide the complete Federal Register citation for the NPRM, including volume, number, date, and page.",
        parent=step3,
        critical=True
    )
    nprm_citation = extraction.nprm.citation_text if extraction.nprm else None
    claim_nprm_cite = (
        f"The answer includes a complete Federal Register citation for the NPRM with volume, number, date, and page: '{nprm_citation}'."
    )
    await evaluator.verify(
        claim=claim_nprm_cite,
        node=leaf_nprm_cite,
        additional_instruction="Look only at the provided answer text. Confirm the NPRM citation string includes volume, issue number, publication date, and page (or page range)."
    )

    # 3.2 NPRM URL Working (verify via URL)
    leaf_nprm_url = evaluator.add_leaf(
        id="NPRM_URL_Working",
        desc="Provide a publicly accessible, working URL to the NPRM on FederalRegister.gov (or equivalent official Federal Register page).",
        parent=step3,
        critical=True
    )
    claim_nprm_url = "This URL leads to the official Federal Register page for the NPRM and is accessible."
    await evaluator.verify(
        claim=claim_nprm_url,
        node=leaf_nprm_url,
        sources=nprm_url,
        additional_instruction="Verify the URL loads and corresponds to the NPRM page with Federal Register citation details."
    )

    # 3.3 NPRM precedes final rule (logical check via extracted dates)
    nprm_pub_date_text = extraction.nprm.date_text if extraction.nprm else None
    final_pub_date_text = extraction.final_rule.date_text if extraction.final_rule else None
    nprm_dt = parse_date(nprm_pub_date_text)
    final_dt = parse_date(final_pub_date_text)
    precedes_result = bool(nprm_dt and final_dt and (nprm_dt <= final_dt))
    evaluator.add_custom_node(
        result=precedes_result,
        id="NPRM_Precedes_Final_Rule",
        desc="The NPRM publication occurred prior to the final rule publication (chronologically precedes it).",
        parent=step3,
        critical=True
    )

    # ------------------------------------------------------------------ #
    # Step 4: Comment Period Dates from NPRM (parallel, critical)        #
    # ------------------------------------------------------------------ #
    step4 = evaluator.add_parallel(
        id="Step_4_Comment_Period_Dates_From_NPRM",
        desc="Extract the comment period start/end dates as defined in the question (start = NPRM publication date; end = deadline stated in NPRM).",
        parent=reg_root,
        critical=True
    )

    comment_start_text = extraction.comments.start_date_text if extraction.comments else None
    comment_end_text = extraction.comments.end_date_text if extraction.comments else None

    # 4.1 NPRM includes a stated public comment period (deadline)
    leaf_comment_period_present = evaluator.add_leaf(
        id="Comment_Period_Provided_In_NPRM",
        desc="The NPRM includes a stated public comment period (i.e., a deadline for submitting comments).",
        parent=step4,
        critical=True
    )
    claim_comment_period_present = (
        "The NPRM page includes a comment deadline or 'comments close' date indicating a public comment period."
    )
    await evaluator.verify(
        claim=claim_comment_period_present,
        node=leaf_comment_period_present,
        sources=nprm_url,
        additional_instruction="Check the NPRM page for a 'Comments close' date or explicit 'submit comments by' deadline."
    )

    # 4.2 Comment Start Date (NPRM publication date)
    leaf_comment_start = evaluator.add_leaf(
        id="Comment_Start_Date_NPRM_Publication_Date",
        desc="Provide the public comment period start date, defined as the NPRM publication date in the Federal Register.",
        parent=step4,
        critical=True
    )
    claim_comment_start = f"The NPRM publication date (comment period start) is {comment_start_text}."
    await evaluator.verify(
        claim=claim_comment_start,
        node=leaf_comment_start,
        sources=nprm_url,
        additional_instruction="Verify the NPRM 'Published' date shown on the Federal Register page matches the provided start date."
    )

    # 4.3 Comment End Date (NPRM deadline)
    leaf_comment_end = evaluator.add_leaf(
        id="Comment_End_Date_NPRM_Deadline",
        desc="Provide the public comment period end date, defined as the comment deadline stated in the NPRM.",
        parent=step4,
        critical=True
    )
    claim_comment_end = f"The comment period deadline (end date) stated in the NPRM is {comment_end_text}."
    await evaluator.verify(
        claim=claim_comment_end,
        node=leaf_comment_end,
        sources=nprm_url,
        additional_instruction="Verify the NPRM page states the provided comment deadline (e.g., 'Comments must be received by ...')."
    )

    # ------------------------------------------------------------------ #
    # Step 5: Duration Calculation and 30-day Check (sequential, critical)
    # ------------------------------------------------------------------ #
    step5 = evaluator.add_sequential(
        id="Step_5_Duration_Calculation_And_30_Day_Check",
        desc="Compute the calendar-day duration from start to end date and verify it meets the ≥30-day minimum standard.",
        parent=reg_root,
        critical=True
    )

    stated_days = parse_int_from_text(extraction.comments.duration_days_text if extraction.comments else None)
    computed_days = compute_duration_days(comment_start_text, comment_end_text)

    # 5.1 Duration Calculated Correctly (custom check comparing answer vs computed)
    duration_correct = bool(stated_days is not None and computed_days is not None and stated_days == computed_days)
    evaluator.add_custom_node(
        result=duration_correct,
        id="Duration_Calculated_Correctly",
        desc="Correctly calculate the public comment period duration in calendar days from the provided start date to the provided end date.",
        parent=step5,
        critical=True
    )

    # 5.2 Meets at least 30 calendar days (custom check)
    meets_30 = bool(computed_days is not None and computed_days >= 30)
    evaluator.add_custom_node(
        result=meets_30,
        id="Meets_At_Least_30_Calendar_Days",
        desc="Verify the calculated duration is at least 30 calendar days (typical minimum standard referenced in the prompt).",
        parent=step5,
        critical=True
    )

    # Add diagnostic info for transparency
    evaluator.add_custom_info(
        info={
            "nprm_publication_date_text": nprm_pub_date_text,
            "final_rule_publication_date_text": final_pub_date_text,
            "nprm_precedes_final_rule": precedes_result,
            "comment_start_date_text": comment_start_text,
            "comment_end_date_text": comment_end_text,
            "stated_duration_days": stated_days,
            "computed_duration_days": computed_days
        },
        info_type="diagnostics",
        info_name="computed_checks"
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
) -> Dict[str, Any]:
    """
    Entry point for evaluating an answer to the Federal Register major rule task.
    """
    # Initialize evaluator (root is non-critical by framework design; we add a critical child node for the rubric)
    evaluator = Evaluator()
    evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Overall root parallel; rubric subtree is added as critical sequential under it
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

    # Extract structured info from the answer
    extraction: RulePackageExtraction = await evaluator.extract(
        prompt=prompt_extract_rule_package(),
        template_class=RulePackageExtraction,
        extraction_name="rule_package_extraction"
    )

    # Add ground truth constraints (for context)
    evaluator.add_ground_truth({
        "required_publication_year": 2024,
        "required_economic_threshold": ">= $200 million annual effect",
        "required_process": "APA notice-and-comment via NPRM preceding final rule",
        "required_citations_fields": ["volume", "number", "date", "page"],
        "comment_period_definition": {
            "start": "NPRM publication date",
            "end": "comment deadline stated in NPRM"
        },
        "minimum_duration_days": 30
    }, gt_type="rubric_constraints")

    # Build the verification tree and run checks
    await build_and_verify(evaluator, extraction)

    # Return evaluation summary
    return evaluator.get_summary()