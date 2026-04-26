import asyncio
import logging
import re
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "verizon_fcc_final_due_date"
TASK_DESCRIPTION = (
    "On January 14, 2026, Verizon experienced a major wireless network outage that disrupted service for customers "
    "across the United States. The outage, which lasted more than six hours and affected 911 emergency services in "
    "multiple cities including New York City, was attributed to a software issue related to server failures in New Jersey.\n\n"
    "Under the Federal Communications Commission's (FCC) Network Outage Reporting System (NORS) regulations codified "
    "in 47 CFR § 4.9, telecommunications carriers must report certain network outages. For wireless providers, a Final "
    "Communications Outage Report must be submitted no later than a specified number of days after discovering a reportable outage.\n\n"
    "Based on the FCC's reporting requirements for wireless carriers and the facts of the January 14, 2026 Verizon outage, "
    "on what date was Verizon's Final Communications Outage Report due to the FCC?"
)

# Reference for evaluation context (not used for verification-by-URL)
FCC_RULE_SNIPPET = (
    "Per 47 CFR § 4.9, for Commercial Mobile Radio Service (wireless) providers, a Final Communications Outage Report "
    "is due no later than 30 days after discovery of a reportable outage."
)

# --------------------------------------------------------------------------- #
# Utilities                                                                   #
# --------------------------------------------------------------------------- #
MONTH_ABBR_WITH_DOTS = {
    "Jan.": "Jan", "Feb.": "Feb", "Mar.": "Mar", "Apr.": "Apr", "Jun.": "Jun",
    "Jul.": "Jul", "Aug.": "Aug", "Sep.": "Sep", "Sept.": "Sep", "Oct.": "Oct",
    "Nov.": "Nov", "Dec.": "Dec", "May.": "May"
}

DATE_FORMATS = [
    "%B %d, %Y",     # January 14, 2026
    "%b %d, %Y",     # Jan 14, 2026
    "%B %d %Y",      # January 14 2026
    "%b %d %Y",      # Jan 14 2026
    "%Y-%m-%d",      # 2026-01-14
    "%Y/%m/%d",      # 2026/01/14
    "%m/%d/%Y",      # 01/14/2026
    "%m/%d/%y",      # 01/14/26
    "%d %B %Y",      # 14 January 2026
    "%d %b %Y",      # 14 Jan 2026
]


def _clean_date_str(s: str) -> str:
    if not s:
        return s
    s = s.strip()
    # Remove ordinal suffixes like "14th", "1st", "2nd", "3rd"
    s = re.sub(r'(\d{1,2})(st|nd|rd|th)\b', r'\1', s, flags=re.IGNORECASE)
    # Remove periods in month abbreviations (e.g., "Feb." -> "Feb")
    for k, v in MONTH_ABBR_WITH_DOTS.items():
        s = s.replace(k, v)
    # Normalize whitespace and commas
    s = re.sub(r'\s+', ' ', s)
    s = s.replace(" ,", ",")
    return s


def parse_date_str(s: Optional[str]) -> Optional[datetime]:
    if not s or not isinstance(s, str):
        return None
    s2 = _clean_date_str(s)
    for fmt in DATE_FORMATS:
        try:
            return datetime.strptime(s2, fmt)
        except Exception:
            continue
    # Try to handle cases like "2026-2-3" (single-digit month/day)
    try:
        parts = re.split(r'[\/\-\s,]+', s2)
        if len(parts) == 3 and all(p.isdigit() for p in parts):
            # Heuristics: prefer YYYY-M-D (e.g., 2026-2-3)
            if len(parts[0]) == 4:
                y, m, d = int(parts[0]), int(parts[1]), int(parts[2])
                return datetime(year=y, month=m, day=d)
            # Or M/D/YYYY
            if len(parts[2]) == 4:
                m, d, y = int(parts[0]), int(parts[1]), int(parts[2])
                return datetime(year=y, month=m, day=d)
    except Exception:
        pass
    return None


def to_iso_date_str(dt: Optional[datetime]) -> Optional[str]:
    return dt.strftime("%Y-%m-%d") if dt else None


def add_30_days(dt: Optional[datetime]) -> Optional[datetime]:
    if not dt:
        return None
    return dt + timedelta(days=30)


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class FCCDueDateExtraction(BaseModel):
    discovery_date: Optional[str] = None
    due_date: Optional[str] = None
    used_30_day_rule: Optional[bool] = None
    rule_citation_text: Optional[str] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_due_date_fields() -> str:
    return (
        "Extract the following fields strictly from the answer text (do not infer):\n"
        "1) discovery_date: The specific calendar date the answer identifies as the 'discovery' date used for FCC reporting purposes.\n"
        "   - If multiple dates are mentioned, select the one explicitly used to compute the due date.\n"
        "   - If no discovery date is stated, set to null.\n"
        "2) due_date: The specific final report due date given as a calendar date.\n"
        "   - If no due date is stated, set to null.\n"
        "3) used_30_day_rule: true if the answer explicitly applies the rule that the Final Communications Outage Report is due no later than 30 days after discovery; otherwise false (or null if unclear).\n"
        "4) rule_citation_text: any quoted rule text or CFR citation used to justify the 30-day rule (may be null).\n"
        "Return the values exactly as they appear in the answer without normalization."
    )


# --------------------------------------------------------------------------- #
# Verification subtrees                                                       #
# --------------------------------------------------------------------------- #
async def build_reportable_outage_check(evaluator: Evaluator, parent_node) -> None:
    """
    Verify the outage is reportable under the provided definition for wireless carriers:
    duration ≥ 30 minutes AND potentially affected a 911 special facility.
    """
    reportable_node = evaluator.add_parallel(
        id="Reportable_Outage_Check",
        desc="Verify the outage is reportable under the provided definition for wireless carriers (duration ≥ 30 minutes AND potentially affected a 911 special facility).",
        parent=parent_node,
        critical=True
    )

    # Duration threshold leaf
    duration_leaf = evaluator.add_leaf(
        id="Meets_Duration_Threshold",
        desc="Outage duration meets/exceeds 30 minutes (constraints state it lasted > 6 hours).",
        parent=reportable_node,
        critical=True
    )
    duration_claim = (
        "The January 14, 2026 Verizon wireless outage lasted more than six hours, which clearly exceeds the 30-minute "
        "threshold for a reportable outage for wireless providers."
    )
    await evaluator.verify(
        claim=duration_claim,
        node=duration_leaf,
        additional_instruction=(
            "Use only the task description provided above as the authoritative facts; do not search the web. "
            "Focus solely on whether the claim matches those provided facts."
        ),
    )

    # 911 impact criterion leaf
    impact_leaf = evaluator.add_leaf(
        id="Meets_911_Impact_Criterion",
        desc="Outage potentially affected a 911 special facility (constraints state 911 services were affected in multiple cities).",
        parent=reportable_node,
        critical=True
    )
    impact_claim = (
        "The outage affected 911 emergency services in multiple cities, including New York City, meeting the 'special "
        "facility' (911) impact criterion for a reportable outage."
    )
    await evaluator.verify(
        claim=impact_claim,
        node=impact_leaf,
        additional_instruction=(
            "Use only the task description provided above as the authoritative facts; do not search the web. "
            "Focus solely on whether the claim matches those provided facts."
        ),
    )


async def build_discovery_date_checks(
    evaluator: Evaluator,
    parent_node,
    extracted: FCCDueDateExtraction,
) -> None:
    """
    Check that the discovery date is stated and consistent with the provided definition and facts.
    """
    discovery_node = evaluator.add_parallel(
        id="Discovery_Date",
        desc="Identify the discovery date (calendar date) for FCC reporting purposes using the provided definition of discovery and the provided outage facts.",
        parent=parent_node,
        critical=True
    )

    # Existence: discovery date stated
    has_discovery = extracted.discovery_date is not None and extracted.discovery_date.strip() != ""
    evaluator.add_custom_node(
        result=has_discovery,
        id="Discovery_Date_Stated",
        desc="The answer states a discovery date (calendar date) used for the deadline calculation.",
        parent=discovery_node,
        critical=True
    )

    # Consistency with definition and facts
    consistency_leaf = evaluator.add_leaf(
        id="Discovery_Date_Consistent_With_Definition_And_Facts",
        desc="The stated discovery date is consistent with: (a) the definition that discovery occurs when the provider determines a reportable outage has occurred, and (b) the provided outage timeline/facts.",
        parent=discovery_node,
        critical=True
    )

    stated_date_text = extracted.discovery_date or ""
    consistency_claim = (
        f"The answer's stated discovery date for FCC reporting is '{stated_date_text}'. This is consistent with the "
        "definition that discovery occurs when the provider determines a reportable outage has occurred and with the "
        "provided outage facts (a major, reportable outage occurred on January 14, 2026 and 911 services were affected)."
    )
    await evaluator.verify(
        claim=consistency_claim,
        node=consistency_leaf,
        additional_instruction=(
            "Judge consistency using only the provided definition and outage facts in the task description. "
            "Absent any explicit justification in the answer to delay 'discovery' to a later date, a discovery date of "
            "January 14, 2026 is considered consistent. If the answer picks a later date without justification tied to "
            "when the provider determined the outage was reportable, consider it inconsistent."
        ),
    )


async def build_due_date_calculation_checks(
    evaluator: Evaluator,
    parent_node,
    extracted: FCCDueDateExtraction,
    computed_due_from_extracted_discovery: Optional[datetime],
) -> None:
    """
    Verify the due date calculation process:
    - The 30-day rule was used
    - The 30-day addition was calculated correctly
    - The due date is output as an unambiguous calendar date
    """
    due_calc_node = evaluator.add_parallel(
        id="Due_Date_Calculation",
        desc="Apply the rule that the Final Communications Outage Report is due no later than 30 days after discovery, and output the resulting calendar date.",
        parent=parent_node,
        critical=True
    )

    # Uses 30-day rule (LLM judgment from the answer)
    uses_rule_leaf = evaluator.add_leaf(
        id="Uses_30_Day_Rule",
        desc="The answer uses the provided deadline rule: due date = discovery date + 30 days.",
        parent=due_calc_node,
        critical=True
    )
    uses_rule_claim = (
        "The answer explicitly applies the rule that the Final Communications Outage Report is due no later than "
        "30 days after the discovery date (i.e., due date = discovery date + 30 days)."
    )
    await evaluator.verify(
        claim=uses_rule_claim,
        node=uses_rule_leaf,
        additional_instruction=(
            "Accept equivalent phrasings such as 'within 30 days of discovery' or 'no later than 30 days after discovery'. "
            "If the answer uses a different rule (e.g., 7 days, business days, or a different offset), mark this incorrect."
        ),
    )

    # Outputs due date as an unambiguous calendar date (programmatic check: parse succeeds)
    due_dt_parsed = parse_date_str(extracted.due_date)
    evaluator.add_custom_node(
        result=due_dt_parsed is not None,
        id="Outputs_Due_Date_As_Date",
        desc="The answer outputs the resulting due date as an unambiguous calendar date.",
        parent=due_calc_node,
        critical=True
    )

    # Adds 30 days correctly (programmatic check comparing extracted due date to computed (discovery + 30))
    correct_addition = False
    if computed_due_from_extracted_discovery is not None and due_dt_parsed is not None:
        correct_addition = (due_dt_parsed.date() == computed_due_from_extracted_discovery.date())
    evaluator.add_custom_node(
        result=bool(correct_addition),
        id="Adds_30_Days_Correctly",
        desc="The answer correctly performs the calendar addition of 30 days from the stated discovery date.",
        parent=due_calc_node,
        critical=True
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
    Evaluate an answer for the FCC Final Communications Outage Report due date task.
    """
    # Initialize evaluator
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.SEQUENTIAL,  # Overall task logically sequential
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

    # Extract structured fields from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_due_date_fields(),
        template_class=FCCDueDateExtraction,
        extraction_name="fcc_due_date_extraction",
    )

    # Compute derived dates based on the extracted discovery date
    disc_dt = parse_date_str(extracted.discovery_date)
    computed_due = add_30_days(disc_dt)

    # Add GT and computed info for transparency
    evaluator.add_ground_truth({
        "fcc_rule_reference": FCC_RULE_SNIPPET,
        "facts_baseline_outage_date": "2026-01-14",
        "expected_due_if_discovery_2026_01_14": "2026-02-13",
    }, gt_type="reference_info")

    evaluator.add_custom_info(
        info={
            "extracted_discovery_date_raw": extracted.discovery_date,
            "extracted_due_date_raw": extracted.due_date,
            "used_30_day_rule_extracted": extracted.used_30_day_rule,
            "parsed_discovery_iso": to_iso_date_str(disc_dt),
            "computed_due_from_extracted_discovery_iso": to_iso_date_str(computed_due),
        },
        info_type="computed_dates",
        info_name="computed_dates_from_answer"
    )

    # Build the verification tree according to rubric
    fcc_final_node = evaluator.add_sequential(
        id="FCC_Final_Report_Due_Date",
        desc="Determine the calendar due date for Verizon's FCC Final Communications Outage Report for the January 14, 2026 outage using the provided FCC rule and the provided definition of discovery.",
        parent=root,
        critical=True
    )

    # 1) Reportable Outage Check
    await build_reportable_outage_check(evaluator, fcc_final_node)

    # 2) Discovery Date checks
    await build_discovery_date_checks(evaluator, fcc_final_node, extracted)

    # 3) Due Date Calculation checks
    await build_due_date_calculation_checks(evaluator, fcc_final_node, extracted, computed_due)

    # Return evaluation summary
    return evaluator.get_summary()