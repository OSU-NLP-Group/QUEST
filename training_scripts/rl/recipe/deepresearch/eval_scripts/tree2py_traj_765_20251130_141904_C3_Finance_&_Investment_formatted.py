import asyncio
import logging
from typing import Any, Optional, List, Dict
from datetime import datetime, timedelta

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "awk_ceo_transition_2025"
TASK_DESCRIPTION = """
For American Water Works Company, Inc. (NYSE: AWK), a publicly traded water utility company, verify the company's compliance with SEC disclosure requirements related to its 2025 CEO transition. Specifically: 
(1) Identify the SEC Form 8-K filing under Item 5.02 that disclosed the departure of the former CEO and appointment of the new CEO, and provide the direct URL to this filing on the SEC EDGAR database along with the filing date. 
(2) Determine the date when the CEO transition was publicly announced (as stated in the Form 8-K or associated press release), and provide the URL to the source document containing this announcement date. 
(3) Verify whether the Form 8-K was filed within 4 business days of the announcement date, as required by SEC regulations. Show your calculation of business days or provide a reference confirming compliance. 
(4) From the Form 8-K and related SEC filings or company announcements, extract and provide: the exact date when the new CEO officially assumed the CEO position, the date of American Water's 2025 annual shareholder meeting, and verify that the new CEO assumed the position on or before the annual meeting date. 
(5) Provide URLs to all supporting documents used to verify these details (such as the Form 8-K text, proxy statement, or relevant investor relations pages).
"""


# --------------------------------------------------------------------------- #
# Data models for extracting structured information from the answer           #
# --------------------------------------------------------------------------- #
class Form8KExtraction(BaseModel):
    """Item 5.02 Form 8-K essentials for AWK CEO transition."""
    eight_k_url: Optional[str] = None
    eight_k_filing_date: Optional[str] = None  # Prefer exact as displayed on EDGAR (string)


class AnnouncementExtraction(BaseModel):
    """CEO transition announcement details."""
    announcement_date: Optional[str] = None  # As stated in Form 8-K or press release
    announcement_source_url: Optional[str] = None  # Primary source URL (8-K text or press release)


class AssumptionMeetingExtraction(BaseModel):
    """New CEO effective date and 2025 annual meeting date."""
    ceo_assumption_date: Optional[str] = None
    ceo_assumption_source_url: Optional[str] = None
    annual_meeting_date_2025: Optional[str] = None
    annual_meeting_source_url: Optional[str] = None  # Typically the DEF 14A or meeting notice


class ProxyExtraction(BaseModel):
    """Proxy statement (DEF 14A) URL."""
    def14a_url: Optional[str] = None


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_form8k() -> str:
    return """
    Extract the specific SEC Form 8-K (Item 5.02) referenced in the answer for American Water Works Company, Inc. (NYSE: AWK) regarding its 2025 CEO transition.
    Return:
    - eight_k_url: The direct EDGAR URL to the exact Form 8-K filing page (not a search page or company filings list). Accept URLs such as https://www.sec.gov/ixviewer/doc?action=display&... or https://www.sec.gov/Archives/edgar/data/... pointing to the filing document itself.
    - eight_k_filing_date: The filing date string exactly as shown on EDGAR for this 8-K (e.g., "January 5, 2025" or "2025-01-05").
    If either the direct EDGAR URL or the filing date is not explicitly provided in the answer, return null for that field.
    """


def prompt_extract_announcement() -> str:
    return """
    Identify the public announcement date of the CEO transition (as stated in the Form 8-K or an associated company press release).
    Return:
    - announcement_date: The CEO transition announcement date string exactly as presented in the answer (e.g., "January 2, 2025", "2025-01-02").
    - announcement_source_url: The URL of the primary source document that explicitly states that announcement date (preferably the Form 8-K text page on EDGAR or the official company press release).
    If either is missing in the answer, return null for that field.
    """


def prompt_extract_assumption_and_meeting() -> str:
    return """
    Extract the new CEO official assumption date and the company's 2025 annual shareholder meeting date.
    Return:
    - ceo_assumption_date: The exact date the new CEO officially assumed the CEO role (string).
    - ceo_assumption_source_url: A primary-source URL supporting that assumption date (e.g., Form 8-K, press release, or investor relations page).
    - annual_meeting_date_2025: The date of the company's 2025 annual shareholder meeting (string).
    - annual_meeting_source_url: A primary-source URL supporting the meeting date (typically the 2025 DEF 14A or the official meeting notice).
    If any of these are not present in the answer, return null for those fields.
    """


def prompt_extract_def14a() -> str:
    return """
    Extract the URL to American Water Works Company, Inc.’s 2025 DEF 14A proxy statement for the annual meeting.
    Return:
    - def14a_url: The EDGAR (or the filed-document) URL to the 2025 DEF 14A proxy statement.
    If the URL is not present in the answer, return null.
    """


# --------------------------------------------------------------------------- #
# Date parsing and business-day calculation helpers                           #
# --------------------------------------------------------------------------- #
def _try_dateutil_parse(date_str: str) -> Optional[datetime]:
    """Attempt robust parsing with dateutil if available."""
    try:
        from dateutil import parser as date_parser  # type: ignore
        return date_parser.parse(date_str).date()
    except Exception:
        return None


def parse_date(date_str: Optional[str]) -> Optional[datetime]:
    """Best-effort parser for common date formats; return date (no time)."""
    if not date_str or not isinstance(date_str, str):
        return None
    s = date_str.strip()

    # Try dateutil first if available
    d = _try_dateutil_parse(s)
    if d:
        return d

    # Try common formats
    fmts = [
        "%Y-%m-%d",
        "%m/%d/%Y",
        "%Y/%m/%d",
        "%B %d, %Y",   # January 5, 2025
        "%b %d, %Y",   # Jan 5, 2025
        "%d %B %Y",    # 5 January 2025
        "%d %b %Y",    # 5 Jan 2025
    ]
    for fmt in fmts:
        try:
            return datetime.strptime(s, fmt).date()
        except Exception:
            continue

    return None


def business_days_between(announcement: Optional[datetime], filing: Optional[datetime]) -> Optional[int]:
    """
    Compute business days from the day AFTER announcement_date up to and including filing_date.
    Excludes Saturdays and Sundays. (Federal holidays are not excluded due to limited context.)
    Returns None if dates are invalid or filing precedes announcement.
    """
    if not announcement or not filing:
        return None
    if filing < announcement:
        return None

    # Counting starts the next business day after the announcement event
    current = announcement + timedelta(days=1)
    count = 0
    while current <= filing:
        if current.weekday() < 5:  # Monday=0 .. Friday=4
            count += 1
        current += timedelta(days=1)
    return count


# --------------------------------------------------------------------------- #
# Verification step builders                                                  #
# --------------------------------------------------------------------------- #
async def build_step_1_identify_8k(
    evaluator: Evaluator,
    parent_node,
    form8k: Form8KExtraction
) -> None:
    """
    Step 1: Identify Item 5.02 Form 8-K, provide EDGAR direct URL, filing date, and confirm coverage.
    """
    step_node = evaluator.add_parallel(
        id="Step_1_Identify_Item_5_02_Form_8K",
        desc="Identify the SEC Form 8-K (Item 5.02) that discloses the former CEO’s departure and the new CEO’s appointment, and provide required filing metadata.",
        parent=parent_node,
        critical=True
    )

    # Existence prerequisites
    url_exists = evaluator.add_custom_node(
        result=bool(form8k.eight_k_url),
        id="Form_8K_URL_Provided",
        desc="Form 8-K EDGAR URL is provided in the answer",
        parent=step_node,
        critical=True
    )
    date_exists = evaluator.add_custom_node(
        result=bool(form8k.eight_k_filing_date),
        id="Form_8K_Filing_Date_Provided",
        desc="Form 8-K filing date is provided in the answer",
        parent=step_node,
        critical=True
    )

    # 8-K direct EDGAR URL verification
    edgar_url_node = evaluator.add_leaf(
        id="Form_8K_EDGAR_Direct_URL",
        desc="Provide the direct SEC EDGAR URL to the specific Form 8-K filing used for verification (not just a search results page).",
        parent=step_node,
        critical=True
    )
    await evaluator.verify(
        claim="This URL points directly to an SEC EDGAR Form 8-K filing for American Water Works Company, Inc., not a search results page.",
        node=edgar_url_node,
        sources=form8k.eight_k_url,
        additional_instruction="Confirm that the page is the specific Form 8-K filing (e.g., shows 'Form 8-K' and the company name), rather than a general search or listing page."
    )

    # Filing date verification (via the same EDGAR 8-K page)
    filing_date_node = evaluator.add_leaf(
        id="Form_8K_Filing_Date",
        desc="State the Form 8-K filing date (as shown on EDGAR) for the identified filing.",
        parent=step_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The Form 8-K filing date shown on EDGAR is '{form8k.eight_k_filing_date}'.",
        node=filing_date_node,
        sources=form8k.eight_k_url,
        additional_instruction="Check the EDGAR page for the displayed 'Filed' or 'Filing Date' and verify it matches exactly."
    )

    # Item 5.02 and coverage verification
    item_502_node = evaluator.add_leaf(
        id="Form_8K_Is_Item_5_02_And_Covers_CEO_Transition",
        desc="Verify the identified Form 8-K is under Item 5.02 and explicitly discloses the CEO departure and new CEO appointment (verifiable from the filing text).",
        parent=step_node,
        critical=True
    )
    await evaluator.verify(
        claim="This Form 8-K includes Item 5.02 and explicitly discloses the CEO departure and the appointment of the new CEO.",
        node=item_502_node,
        sources=form8k.eight_k_url,
        additional_instruction="Search the filing text for 'Item 5.02' and for language describing the CEO departure and the appointment of the successor."
    )


async def build_step_2_announcement_date(
    evaluator: Evaluator,
    parent_node,
    ann: AnnouncementExtraction,
    fallback_8k_url: Optional[str]
) -> None:
    """
    Step 2: Determine the public announcement date and cite its source.
    """
    step_node = evaluator.add_parallel(
        id="Step_2_Announcement_Date",
        desc="Determine the public announcement date of the CEO transition and cite its source.",
        parent=parent_node,
        critical=True
    )

    # Existence prerequisites
    ann_date_exists = evaluator.add_custom_node(
        result=bool(ann.announcement_date),
        id="Announcement_Date_Provided",
        desc="CEO transition announcement date is provided in the answer",
        parent=step_node,
        critical=True
    )
    ann_src_exists = evaluator.add_custom_node(
        result=bool(ann.announcement_source_url),
        id="Announcement_Source_URL_Provided",
        desc="Source URL for the announcement date is provided in the answer",
        parent=step_node,
        critical=True
    )

    # Announcement date verification against source
    ann_date_node = evaluator.add_leaf(
        id="Announcement_Date_Value",
        desc="State the CEO transition announcement date (as stated in the Form 8-K or an associated press release).",
        parent=step_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The CEO transition announcement date was '{ann.announcement_date}'.",
        node=ann_date_node,
        sources=ann.announcement_source_url or fallback_8k_url,
        additional_instruction="Verify that the source document explicitly states this announcement date."
    )

    # Source document clarity verification
    ann_src_node = evaluator.add_leaf(
        id="Announcement_Date_Source_URL",
        desc="Provide a URL to the primary source document (Form 8-K text or associated company press release) that clearly states the announcement date.",
        parent=step_node,
        critical=True
    )
    await evaluator.verify(
        claim="The provided source document clearly states the CEO transition announcement date.",
        node=ann_src_node,
        sources=ann.announcement_source_url or fallback_8k_url,
        additional_instruction="Confirm that the page includes the announcement date in clear text (matching or equivalent to the provided date)."
    )


async def build_step_3_timeliness(
    evaluator: Evaluator,
    parent_node,
    ann: AnnouncementExtraction,
    form8k: Form8KExtraction
) -> None:
    """
    Step 3: Verify the 8-K was filed within 4 business days of the announcement date.
    """
    step_node = evaluator.add_parallel(
        id="Step_3_4_Business_Day_Timeliness",
        desc="Verify the 8-K was filed within 4 business days of the announcement date (SEC requirement).",
        parent=parent_node,
        critical=True
    )

    # Parse dates
    ann_dt = parse_date(ann.announcement_date)
    filing_dt = parse_date(form8k.eight_k_filing_date)
    bd_count = business_days_between(ann_dt, filing_dt)

    # Add calculation details to summary for transparency
    evaluator.add_custom_info(
        info={
            "announcement_date_raw": ann.announcement_date,
            "filing_date_raw": form8k.eight_k_filing_date,
            "announcement_date_parsed": str(ann_dt) if ann_dt else None,
            "filing_date_parsed": str(filing_dt) if filing_dt else None,
            "business_day_count": bd_count,
            "calculation_rule": "Count business days from the day AFTER announcement_date through filing_date, excluding Saturdays and Sundays (federal holidays not excluded)."
        },
        info_type="business_day_calculation",
        info_name="Timeliness_Calculation_Detail"
    )

    # Calculation availability/validity check
    calc_node = evaluator.add_custom_node(
        result=(bd_count is not None),
        id="Business_Day_Calculation",
        desc=f"Show a business-day count from announcement date to filing date; Computed business-day count: {bd_count if bd_count is not None else 'unavailable'}",
        parent=step_node,
        critical=True
    )

    # Timeliness <= 4 business days
    timely = (bd_count is not None) and (bd_count <= 4)
    conclusion_node = evaluator.add_custom_node(
        result=timely,
        id="Timeliness_Conclusion",
        desc="Explicitly conclude whether the filing occurred within 4 business days of the announcement date.",
        parent=step_node,
        critical=True
    )


async def build_step_4_effective_and_alignment(
    evaluator: Evaluator,
    parent_node,
    am: AssumptionMeetingExtraction,
    proxy: ProxyExtraction
) -> None:
    """
    Step 4: Extract new CEO effective date and 2025 annual meeting date, verify alignment.
    """
    step_node = evaluator.add_parallel(
        id="Step_4_Effective_Date_And_Annual_Meeting_Alignment",
        desc="Extract the new CEO effective date and the 2025 annual meeting date, and verify timing relative to the annual meeting.",
        parent=parent_node,
        critical=True
    )

    # Existence prerequisites
    ceo_date_exists = evaluator.add_custom_node(
        result=bool(am.ceo_assumption_date),
        id="New_CEO_Assumption_Date_Provided",
        desc="New CEO official assumption date is provided in the answer",
        parent=step_node,
        critical=True
    )
    ceo_src_exists = evaluator.add_custom_node(
        result=bool(am.ceo_assumption_source_url),
        id="New_CEO_Assumption_Source_URL_Provided",
        desc="Source URL for the new CEO assumption date is provided",
        parent=step_node,
        critical=True
    )

    mtg_date_exists = evaluator.add_custom_node(
        result=bool(am.annual_meeting_date_2025),
        id="Annual_Meeting_Date_2025_Provided",
        desc="2025 annual shareholder meeting date is provided",
        parent=step_node,
        critical=True
    )
    mtg_src_exists = evaluator.add_custom_node(
        result=bool(am.annual_meeting_source_url),
        id="Annual_Meeting_Source_URL_Provided",
        desc="Source URL for the 2025 annual meeting date is provided",
        parent=step_node,
        critical=True
    )

    # Verify assumption date with source
    ceo_assume_node = evaluator.add_leaf(
        id="New_CEO_Official_Assumption_Date_With_Source",
        desc="State the exact date the new CEO officially assumed the CEO role and provide a primary-source URL supporting that date.",
        parent=step_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The new CEO officially assumed the CEO role on '{am.ceo_assumption_date}'.",
        node=ceo_assume_node,
        sources=am.ceo_assumption_source_url,
        additional_instruction="Confirm the document explicitly states this effective/assumption date."
    )

    # Verify 2025 annual meeting date with source
    mtg_date_node = evaluator.add_leaf(
        id="Annual_Meeting_Date_2025_With_Source",
        desc="State the date of the company’s 2025 annual shareholder meeting and provide a primary-source URL supporting that date (typically the 2025 DEF 14A).",
        parent=step_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The company’s 2025 annual shareholder meeting was held on '{am.annual_meeting_date_2025}'.",
        node=mtg_date_node,
        sources=am.annual_meeting_source_url,
        additional_instruction="Verify the meeting date from the proxy statement (DEF 14A) or official meeting notice."
    )

    # Logical check: Assumption on or before meeting date
    ceo_dt = parse_date(am.ceo_assumption_date)
    mtg_dt = parse_date(am.annual_meeting_date_2025)
    on_or_before = (ceo_dt is not None and mtg_dt is not None and ceo_dt <= mtg_dt)

    assumption_timing_node = evaluator.add_custom_node(
        result=on_or_before,
        id="Assumption_On_Or_Before_Annual_Meeting_Date",
        desc="Verify the new CEO assumed the CEO role on or before the 2025 annual shareholder meeting date.",
        parent=step_node,
        critical=True
    )

    # Verify sources state assumption at or immediately following annual meeting
    at_or_immediately_node = evaluator.add_leaf(
        id="Assumption_At_Or_Immediately_Following_Annual_Meeting",
        desc="Verify the source(s) state the new CEO assumed the CEO role at or immediately following the company’s 2025 annual shareholder meeting.",
        parent=step_node,
        critical=True
    )
    multi_sources: List[str] = []
    if am.ceo_assumption_source_url:
        multi_sources.append(am.ceo_assumption_source_url)
    if am.annual_meeting_source_url:
        multi_sources.append(am.annual_meeting_source_url)
    if proxy.def14a_url:
        multi_sources.append(proxy.def14a_url)

    await evaluator.verify(
        claim="The new CEO assumed the CEO role at the annual meeting or immediately following it.",
        node=at_or_immediately_node,
        sources=multi_sources if multi_sources else None,
        additional_instruction="Look for phrasing such as 'effective at the annual meeting', 'effective immediately following the annual meeting', or equivalent wording."
    )


async def build_step_5_def14a(
    evaluator: Evaluator,
    parent_node,
    proxy: ProxyExtraction
) -> None:
    """
    Step 5: Verify DEF 14A URL and that it mentions the 2025 CEO transition.
    """
    step_node = evaluator.add_parallel(
        id="Step_5_DEF14A_Proxy_References_Transition",
        desc="Verify the company filed a 2025 DEF 14A proxy statement for the annual meeting that references/discusses the CEO transition.",
        parent=parent_node,
        critical=True
    )

    # Existence prerequisite
    def14a_exists = evaluator.add_custom_node(
        result=bool(proxy.def14a_url),
        id="DEF14A_URL_Provided",
        desc="DEF 14A URL is provided in the answer",
        parent=step_node,
        critical=True
    )

    # Verify DEF 14A URL is indeed the proxy statement for 2025 annual meeting
    def14a_url_node = evaluator.add_leaf(
        id="DEF14A_URL",
        desc="Provide the EDGAR (or filed-document) URL to the company’s 2025 DEF 14A proxy statement for the annual meeting.",
        parent=step_node,
        critical=True
    )
    await evaluator.verify(
        claim="This document is the company’s 2025 DEF 14A proxy statement for the annual meeting.",
        node=def14a_url_node,
        sources=proxy.def14a_url,
        additional_instruction="Verify the title and filing type indicate 'DEF 14A' for the 2025 annual meeting of American Water Works Company, Inc."
    )

    # Verify DEF 14A mentions the CEO transition
    def14a_mentions_node = evaluator.add_leaf(
        id="DEF14A_Mentions_CEO_Transition",
        desc="Verify the DEF 14A includes text that references/discusses the 2025 CEO transition (verifiable from the proxy content).",
        parent=step_node,
        critical=True
    )
    await evaluator.verify(
        claim="The DEF 14A includes text that references or discusses the 2025 CEO transition.",
        node=def14a_mentions_node,
        sources=proxy.def14a_url,
        additional_instruction="Search the proxy statement for discussion of leadership changes, CEO transition, or related sections."
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
    Evaluate an answer for AWK's 2025 CEO transition SEC compliance verification.
    """
    # Initialize evaluator
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.SEQUENTIAL,  # Overall steps must be checked in order
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

    # Create the critical, sequential top-level compliance node (rubric root)
    compliance_root = evaluator.add_sequential(
        id="CEO_Transition_Compliance_Verification",
        desc="Verify American Water Works Company (NYSE: AWK) complied with SEC disclosure requirements for its 2025 CEO transition: identify the correct Item 5.02 Form 8-K and its EDGAR URL/date, identify the public announcement date and source URL, verify 4-business-day timeliness with a shown calculation, extract CEO assumption date and 2025 annual meeting date with sources, verify assumption timing relative to annual meeting, and confirm the 2025 DEF 14A references the transition with an EDGAR URL.",
        parent=root,
        critical=True
    )

    # Extract structured info from the answer
    form8k_info = await evaluator.extract(
        prompt=prompt_extract_form8k(),
        template_class=Form8KExtraction,
        extraction_name="form_8k_info"
    )
    announcement_info = await evaluator.extract(
        prompt=prompt_extract_announcement(),
        template_class=AnnouncementExtraction,
        extraction_name="announcement_info"
    )
    assumption_meeting_info = await evaluator.extract(
        prompt=prompt_extract_assumption_and_meeting(),
        template_class=AssumptionMeetingExtraction,
        extraction_name="assumption_meeting_info"
    )
    proxy_info = await evaluator.extract(
        prompt=prompt_extract_def14a(),
        template_class=ProxyExtraction,
        extraction_name="def14a_info"
    )

    # Build verification steps (sequential under compliance_root)
    await build_step_1_identify_8k(evaluator, compliance_root, form8k_info)
    await build_step_2_announcement_date(evaluator, compliance_root, announcement_info, form8k_info.eight_k_url)
    await build_step_3_timeliness(evaluator, compliance_root, announcement_info, form8k_info)
    await build_step_4_effective_and_alignment(evaluator, compliance_root, assumption_meeting_info, proxy_info)
    await build_step_5_def14a(evaluator, compliance_root, proxy_info)

    # Return summary with verification tree and recorded extra info
    return evaluator.get_summary()