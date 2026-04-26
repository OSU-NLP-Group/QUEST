import asyncio
import logging
import re
from datetime import datetime, date
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "currie_active_service_duration"
TASK_DESCRIPTION = (
    "Cameron McGowan Currie serves as a federal judge for the United States District Court for the District of South Carolina. "
    "Using the Federal Judicial Center's official biographical directory for Article III federal judges as your authoritative source, "
    "determine the exact duration of Judge Currie's active federal judicial service by calculating the number of calendar days between "
    "her commission date and the date she assumed senior status. Your answer must include: (1) the official commission date, "
    "(2) the official senior status date, (3) the precise number of calendar days of active service between these two dates "
    "(counting from the commission date up to but not including the senior status date), and (4) the URL of the Federal Judicial Center "
    "biographical page used for verification."
)


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class JudgeServiceExtraction(BaseModel):
    """Information extracted from the agent's answer."""
    fjc_url: Optional[str] = None
    commission_date: Optional[str] = None
    senior_status_date: Optional[str] = None
    active_service_days: Optional[str] = None


class FJCDatesExtraction(BaseModel):
    """Official dates extracted from the FJC biographical page."""
    commission_date_official: Optional[str] = None
    senior_status_date_official: Optional[str] = None


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_answer_fields() -> str:
    return """
    Extract the following four items exactly as presented in the answer text:
    1. fjc_url: The URL to the Federal Judicial Center (FJC) biographical page for Judge Cameron McGowan Currie. It should be on the fjc.gov domain. If multiple URLs are present, return the one explicitly used for verification of dates. If none is present, return null.
    2. commission_date: The commission date stated in the answer (any reasonable date format, e.g., 'November 15, 1994', '11/15/1994', or '1994-11-15'). If not present, return null.
    3. senior_status_date: The date stated in the answer for when she assumed senior status (any reasonable format). If not present, return null.
    4. active_service_days: The number of calendar days of active service as stated in the answer. Extract it as a string (e.g., '6,235', '6235', or '6,235 days'). If not present, return null.
    Do not invent or infer any values; extract only what is explicitly provided in the answer.
    """


def prompt_extract_fjc_dates() -> str:
    return """
    From this Federal Judicial Center (FJC) biographical page for Judge Cameron McGowan Currie, extract:
    1. commission_date_official: The official 'Received commission' (or equivalent commission date) for her Article III service on the United States District Court for the District of South Carolina.
    2. senior_status_date_official: The official 'Assumed senior status' date shown on the page.
    Return the dates as strings exactly as shown (do not transform format). If either date is not clearly stated, return null for that field.
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
_MONTH_ABBREV_FIXES = {
    "Sept.": "Sep",
    "Sept": "Sep",
    "Jun.": "Jun",
    "Jul.": "Jul",
    "Apr.": "Apr",
}


def _normalize_date_string(s: str) -> str:
    if not s:
        return s
    s = s.strip()
    # Replace common abbreviated month variants to help strptime
    for k, v in _MONTH_ABBREV_FIXES.items():
        s = s.replace(k, v)
    # Remove ordinal suffixes (st, nd, rd, th)
    s = re.sub(r'(\d+)(st|nd|rd|th)', r'\1', s)
    # Collapse extra spaces
    s = re.sub(r'\s+', ' ', s)
    return s


def parse_date_str(s: Optional[str]) -> Optional[date]:
    """Attempt to parse a date string in common formats."""
    if not s:
        return None
    s = _normalize_date_string(s)
    fmts = [
        "%B %d, %Y",   # November 15, 1994
        "%b %d, %Y",   # Nov 15, 1994
        "%m/%d/%Y",    # 11/15/1994
        "%Y-%m-%d",    # 1994-11-15
        "%d %B %Y",    # 15 November 1994
        "%d %b %Y",    # 15 Nov 1994
    ]
    for fmt in fmts:
        try:
            return datetime.strptime(s, fmt).date()
        except Exception:
            continue
    # Try to parse a simple 'Month YYYY' by assuming day=1
    try:
        d = datetime.strptime(s, "%B %Y").date()
        return date(d.year, d.month, 1)
    except Exception:
        pass
    try:
        d = datetime.strptime(s, "%b %Y").date()
        return date(d.year, d.month, 1)
    except Exception:
        pass
    return None


def parse_int_from_string(s: Optional[str]) -> Optional[int]:
    """Extract first integer (allowing commas) from a string."""
    if not s:
        return None
    m = re.search(r"\d{1,3}(?:,\d{3})*|\d+", s)
    if not m:
        return None
    val = m.group(0).replace(",", "")
    try:
        return int(val)
    except Exception:
        return None


# --------------------------------------------------------------------------- #
# Verification tree construction                                              #
# --------------------------------------------------------------------------- #
async def build_verification_tree(
    evaluator: Evaluator,
    parent_node,
    answer_info: JudgeServiceExtraction,
    fjc_official: Optional[FJCDatesExtraction],
) -> None:
    """
    Build the verification tree and run checks for the four critical criteria.
    """

    # 1) FJC Source URL Provided And Valid
    fjc_url_node = evaluator.add_leaf(
        id="FJC_Source_URL_Provided_And_Valid",
        desc="Answer includes a URL to the Federal Judicial Center (fjc.gov) biographical page for Judge Cameron McGowan Currie used as the authoritative verification source.",
        parent=parent_node,
        critical=True,
    )
    fjc_url = answer_info.fjc_url or ""
    fjc_claim = (
        f"The answer includes the FJC biographical page URL '{fjc_url}', and this URL is on fjc.gov and is the official "
        f"Federal Judicial Center biographical page for Judge Cameron McGowan Currie."
    )
    await evaluator.verify(
        claim=fjc_claim,
        node=fjc_url_node,
        sources=fjc_url if fjc_url else None,
        additional_instruction=(
            "Confirm that the URL domain is fjc.gov and that the page is the FJC biographical directory entry for "
            "Article III judge Cameron McGowan Currie. Use the provided URL and the page content."
        ),
    )

    # 2) Commission Date Provided And Accurate
    commission_node = evaluator.add_leaf(
        id="Commission_Date_Provided_And_Accurate",
        desc="Answer states the official commission date (month/day/year) and it matches the commission date shown on the cited FJC biographical page.",
        parent=parent_node,
        critical=True,
    )
    commission_answer = answer_info.commission_date or ""
    commission_claim = (
        f"The official commission date for Judge Cameron McGowan Currie shown on this FJC page is '{commission_answer}'."
    )
    await evaluator.verify(
        claim=commission_claim,
        node=commission_node,
        sources=fjc_url if fjc_url else None,
        additional_instruction=(
            "Locate the 'Received commission' or equivalent commission date for her Article III District Court service. "
            "Allow minor formatting differences (e.g., abbreviations, leading zeros). The stated date in the answer must match."
        ),
    )

    # 3) Senior Status Date Provided And Accurate
    senior_node = evaluator.add_leaf(
        id="Senior_Status_Date_Provided_And_Accurate",
        desc="Answer states the official senior status date (month/day/year) and it matches the senior status date shown on the cited FJC biographical page.",
        parent=parent_node,
        critical=True,
    )
    senior_answer = answer_info.senior_status_date or ""
    senior_claim = (
        f"The official 'Assumed senior status' date for Judge Cameron McGowan Currie shown on this FJC page is '{senior_answer}'."
    )
    await evaluator.verify(
        claim=senior_claim,
        node=senior_node,
        sources=fjc_url if fjc_url else None,
        additional_instruction=(
            "Find the 'Assumed senior status' date on the FJC page. Allow minor formatting differences. "
            "The date stated in the answer must match what the page shows."
        ),
    )

    # 4) Active Service Days Calculated Correctly (Custom check via computation)
    # Compute based on official FJC dates if available; otherwise mark as failed
    official_commission = fjc_official.commission_date_official if fjc_official else None
    official_senior = fjc_official.senior_status_date_official if fjc_official else None
    d_commission = parse_date_str(official_commission)
    d_senior = parse_date_str(official_senior)
    computed_days: Optional[int] = None
    if d_commission and d_senior and d_senior >= d_commission:
        # Inclusive of commission date, exclusive of senior status date -> difference in days:
        computed_days = (d_senior - d_commission).days

    stated_days = parse_int_from_string(answer_info.active_service_days)

    calc_correct = bool(computed_days is not None and stated_days is not None and computed_days == stated_days)

    evaluator.add_custom_node(
        result=calc_correct,
        id="Active_Service_Days_Calculated_Correctly",
        desc=(
            "Answer provides the precise number of calendar days of active service computed as the difference between the "
            "commission date (inclusive) and the senior status date (exclusive), consistent with the stated counting rule."
        ),
        parent=parent_node,
        critical=True,
    )

    # Add supporting info to summary for transparency/debugging
    evaluator.add_ground_truth({
        "official_commission_date": official_commission,
        "official_senior_status_date": official_senior,
        "computed_active_service_days": computed_days,
    }, gt_type="fjc_official_dates")

    evaluator.add_custom_info({
        "answer_fjc_url": answer_info.fjc_url,
        "answer_commission_date": answer_info.commission_date,
        "answer_senior_status_date": answer_info.senior_status_date,
        "answer_active_service_days": answer_info.active_service_days,
        "parsed_answer_active_days": stated_days,
    }, info_type="answer_extracted_fields")


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
    Evaluate an answer for Judge Currie's active federal judicial service duration using the FJC directory.
    """
    # Initialize evaluator with a critical parallel root node
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

    # Create critical root node as per rubric (children must also be critical)
    main_node = evaluator.add_parallel(
        id="Judicial_Career_Duration_Analysis",
        desc="Verify and calculate the duration of active federal judicial service for Judge Cameron McGowan Currie from commission date to senior status using the Federal Judicial Center (FJC) biographical directory.",
        parent=root,
        critical=True,
    )

    # 1) Extract fields from the agent's answer
    answer_info = await evaluator.extract(
        prompt=prompt_extract_answer_fields(),
        template_class=JudgeServiceExtraction,
        extraction_name="answer_fields",
    )

    # 2) Extract official dates from the FJC page (if URL available)
    fjc_official: Optional[FJCDatesExtraction] = None
    if answer_info.fjc_url:
        fjc_official = await evaluator.extract(
            prompt=prompt_extract_fjc_dates(),
            template_class=FJCDatesExtraction,
            extraction_name="fjc_official_dates_extraction",
            source=answer_info.fjc_url,
            additional_instruction="Focus on the 'Received commission' and 'Assumed senior status' fields."
        )
    else:
        fjc_official = FJCDatesExtraction()  # Empty if no URL

    # 3) Build verification tree and run checks
    await build_verification_tree(
        evaluator=evaluator,
        parent_node=main_node,
        answer_info=answer_info,
        fjc_official=fjc_official,
    )

    # 4) Return structured summary
    return evaluator.get_summary()