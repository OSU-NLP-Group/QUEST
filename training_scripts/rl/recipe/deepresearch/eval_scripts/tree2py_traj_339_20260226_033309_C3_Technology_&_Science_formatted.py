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
TASK_ID = "fcc_nors_20260114"
TASK_DESCRIPTION = (
    "A major U.S. telecommunications carrier experienced a network outage on January 14, 2026, that lasted approximately 5 hours and affected over 100,000 wireless customers across multiple states. "
    "As a compliance analyst, you need to verify whether this carrier met all FCC Network Outage Reporting System (NORS) requirements for this incident. Specifically, determine: "
    "(1) Whether the outage met the minimum duration threshold (at least 30 minutes) requiring mandatory FCC reporting, "
    "(2) Whether the carrier submitted the initial NORS notification within the required 120-minute timeframe after determining the outage was reportable, "
    "(3) Whether the carrier submitted the initial outage report within 3 calendar days as mandated, "
    "(4) Whether the carrier submitted (or committed to submit) the final outage report within 30 days after discovering the outage. "
    "Provide the carrier's name, the exact outage duration, and verify each of the four compliance requirements listed above with supporting evidence from official sources or credible news reports."
)


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class NORSComplianceExtraction(BaseModel):
    carrier_name: Optional[str] = None
    outage_duration: Optional[str] = None  # e.g., "about 5 hours", "5 hours 10 minutes"
    # General incident/documentation sources (official or credible news)
    sources: List[str] = Field(default_factory=list)

    # Requirement-specific statements (as mentioned in the answer) and sources
    notification_within_120: Optional[str] = None
    notification_sources: List[str] = Field(default_factory=list)

    initial_report_within_3_days: Optional[str] = None
    initial_report_sources: List[str] = Field(default_factory=list)

    final_report_within_30_days: Optional[str] = None
    final_report_sources: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_nors() -> str:
    return """
Extract the following information from the answer exactly as provided (do not invent anything):

1) carrier_name: The name of the telecommunications carrier that experienced the outage on January 14, 2026.
2) outage_duration: The exact outage duration as stated (e.g., "5 hours", "approximately 5 hours", "5 hours 20 minutes").
3) sources: A list of all URLs in the answer that document the incident itself (official sources or credible news reports). Include URLs that discuss the outage details such as date, affected customers, and/or duration.

For each of the FCC NORS reporting requirements below, extract:
4) notification_within_120: The text (if any) in the answer that claims whether the initial NORS notification was submitted within 120 minutes of determining the outage was reportable (e.g., "filed within two hours").
5) notification_sources: A list of URLs (if any) that specifically relate to the initial NORS notification timing for this incident.

6) initial_report_within_3_days: The text (if any) in the answer that claims whether the initial outage report was submitted within 3 calendar days.
7) initial_report_sources: A list of URLs (if any) that specifically relate to the initial 3-day outage report.

8) final_report_within_30_days: The text (if any) in the answer that claims whether the final outage report was submitted or the carrier committed to submit it within 30 days after discovering the outage.
9) final_report_sources: A list of URLs (if any) that specifically relate to the 30-day final report (submission or commitment).

Rules for URL fields:
- Extract only URLs explicitly present in the answer text. Do not create or infer any URLs.
- Include full URLs (with http:// or https://). If a URL is missing a protocol, prepend http://.
- If a field is not mentioned in the answer, return null for the text field and an empty list for the URLs.
"""


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def parse_duration_minutes(duration_text: Optional[str]) -> Optional[int]:
    """
    Parse a human-readable duration string into total minutes (best effort).
    Handles common formats like:
      - "5 hours", "5 hrs", "5h", "5-hour", "5 hours 20 minutes", "4h 35m", "300 minutes"
    Returns None if cannot parse any numeric duration.
    """
    if not duration_text:
        return None

    text = duration_text.lower().strip()

    # Try HH:MM patterns (e.g., 5:00)
    m_time = re.search(r"\b(\d{1,2}):(\d{2})\b", text)
    if m_time:
        try:
            hours = int(m_time.group(1))
            minutes = int(m_time.group(2))
            return hours * 60 + minutes
        except Exception:
            pass

    hours = 0
    minutes = 0

    # Hours patterns
    m_hours = re.search(r"(\d+)\s*(?:hours?|hrs?|h)\b", text)
    if m_hours:
        try:
            hours = int(m_hours.group(1))
        except Exception:
            hours = 0

    # Hyphenated "5-hour"
    if hours == 0:
        m_hours_hyphen = re.search(r"(\d+)\s*-\s*hour", text)
        if m_hours_hyphen:
            try:
                hours = int(m_hours_hyphen.group(1))
            except Exception:
                hours = 0

    # Minutes patterns
    m_minutes = re.search(r"(\d+)\s*(?:minutes?|mins?|m)\b", text)
    if m_minutes:
        try:
            minutes = int(m_minutes.group(1))
        except Exception:
            minutes = 0

    # If neither hours nor minutes matched, try a bare "300 minutes" or similar:
    if hours == 0 and minutes == 0:
        m_only_minutes = re.search(r"\b(\d+)\s*(?:minutes?|mins?|m)\b", text)
        if m_only_minutes:
            try:
                minutes = int(m_only_minutes.group(1))
            except Exception:
                minutes = 0

    # If still nothing, consider bare number + 'hour' word variants covered above; otherwise fail
    if hours == 0 and minutes == 0:
        return None

    return hours * 60 + minutes


def merge_sources(*lists: List[str]) -> List[str]:
    """Merge multiple URL lists into a unique, order-preserving list and keep only plausible http(s) URLs."""
    seen = set()
    merged: List[str] = []
    for lst in lists:
        for url in lst or []:
            if not url:
                continue
            u = url.strip()
            if not u:
                continue
            # normalize minimal
            if not (u.startswith("http://") or u.startswith("https://")):
                if u.startswith("www."):
                    u = "http://" + u
                else:
                    # Skip obviously malformed URLs
                    continue
            if u not in seen:
                seen.add(u)
                merged.append(u)
    return merged


# --------------------------------------------------------------------------- #
# Verification tree builders                                                  #
# --------------------------------------------------------------------------- #
async def build_incident_identification(
    evaluator: Evaluator,
    parent_node,
    data: NORSComplianceExtraction
) -> None:
    """
    Build and verify the Incident Identification subtree.
    """
    incident_node = evaluator.add_parallel(
        id="Incident_Identification",
        desc="Identify the telecommunications carrier and document the outage characteristics",
        parent=parent_node,
        critical=True
    )

    # Combine all available sources for robust grounding
    all_sources = merge_sources(
        data.sources,
        data.notification_sources,
        data.initial_report_sources,
        data.final_report_sources
    )

    # Carrier Name Identification (leaf)
    carrier_leaf = evaluator.add_leaf(
        id="Carrier_Name_Identification",
        desc="Provide the name of the telecommunications carrier that experienced the outage on January 14, 2026",
        parent=incident_node,
        critical=True
    )
    carrier_name = data.carrier_name or ""
    await evaluator.verify(
        claim=f"The network outage on January 14, 2026 was experienced by {carrier_name}.",
        node=carrier_leaf,
        sources=all_sources,
        additional_instruction=(
            "Confirm that the page identifies the carrier associated with the January 14, 2026 outage. "
            "Allow minor variants of the company name (e.g., legal suffixes like Inc., LLC). "
            "If the claim contains an empty or missing name, mark as not supported."
        )
    )

    # Outage Duration Documentation (leaf)
    duration_leaf = evaluator.add_leaf(
        id="Outage_Duration_Documentation",
        desc="Document the exact duration of the network outage in hours and minutes",
        parent=incident_node,
        critical=True
    )
    duration_text = data.outage_duration or ""
    await evaluator.verify(
        claim=f"The network outage on January 14, 2026 lasted {duration_text}.",
        node=duration_leaf,
        sources=all_sources,
        additional_instruction=(
            "Verify that the page documents the outage duration in hours and/or minutes "
            "(e.g., 'about 5 hours' or '5 hours 20 minutes'). Minor phrasing differences are acceptable."
        )
    )

    # Supporting Evidence URLs (leaf)
    evidence_leaf = evaluator.add_leaf(
        id="Supporting_Evidence_URLs",
        desc="Provide reference URLs from official sources or credible news reports that document the incident",
        parent=incident_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"This webpage is an official source (e.g., FCC, carrier) or a credible news report documenting the January 14, 2026 outage of {carrier_name}.",
        node=evidence_leaf,
        sources=all_sources,
        additional_instruction=(
            "Pass if at least one provided URL credibly documents the January 14, 2026 outage (e.g., carrier newsroom, FCC, well-known news outlets)."
        )
    )


async def build_nors_compliance_verification(
    evaluator: Evaluator,
    parent_node,
    data: NORSComplianceExtraction
) -> None:
    """
    Build and verify the NORS Compliance subtree in sequence.
    Note: Due to framework constraints, child nodes under a critical parent must also be critical.
    """
    nors_node = evaluator.add_sequential(
        id="NORS_Compliance_Verification",
        desc="Verify compliance with all four FCC Network Outage Reporting System requirements in the logical sequence of the reporting process",
        parent=parent_node,
        critical=True
    )

    # --- Stage 1: Reportability Determination ---
    reportability_node = evaluator.add_parallel(
        id="Reportability_Determination",
        desc="Verify that the outage met the FCC threshold for mandatory reporting (minimum 30-minute duration)",
        parent=nors_node,
        critical=True  # Adjusted to satisfy critical parent constraint
    )

    # Thirty-minute threshold check (custom / binary)
    minutes = parse_duration_minutes(data.outage_duration)
    threshold_met = minutes is not None and minutes >= 30
    evaluator.add_custom_node(
        result=threshold_met,
        id="Thirty_Minute_Threshold_Check",
        desc="Confirm that the documented outage duration meets or exceeds 30 minutes, making it reportable under FCC regulations",
        parent=reportability_node,
        critical=True
    )

    # Reportability Evidence URL (leaf verified by sources)
    reportability_evidence_leaf = evaluator.add_leaf(
        id="Reportability_Evidence_URL",
        desc="Provide reference URL documenting that the outage exceeded the 30-minute threshold",
        parent=reportability_node,
        critical=True
    )
    await evaluator.verify(
        claim="The outage on January 14, 2026 lasted at least 30 minutes (half an hour), making it reportable under FCC NORS rules.",
        node=reportability_evidence_leaf,
        sources=merge_sources(data.sources),
        additional_instruction=(
            "Verify that the page explicitly or implicitly indicates the outage duration was ≥ 30 minutes. "
            "Accept phrasing like 'about 5 hours' or a clearly stated duration ≥ 30 minutes."
        )
    )

    # --- Stage 2: Initial Notification within 120 minutes ---
    init_notif_node = evaluator.add_parallel(
        id="Initial_Notification_Verification",
        desc="Verify that the carrier submitted the NORS notification within 120 minutes of determining the outage was reportable",
        parent=nors_node,
        critical=True  # Adjusted to satisfy critical parent constraint
    )

    notif_check_leaf = evaluator.add_leaf(
        id="120_Minute_Notification_Check",
        desc="Confirm that the initial NORS notification was submitted within the required 120-minute timeframe",
        parent=init_notif_node,
        critical=True
    )
    await evaluator.verify(
        claim="The carrier submitted the initial NORS notification within 120 minutes (2 hours) after determining the outage was reportable for the January 14, 2026 incident.",
        node=notif_check_leaf,
        sources=merge_sources(data.notification_sources),
        additional_instruction=(
            "Pass if the page confirms the NORS initial notification was filed within 2 hours of determining reportability, "
            "or provides timestamps from which this ≤120-minute interval can be inferred."
        )
    )

    notif_evidence_leaf = evaluator.add_leaf(
        id="Notification_Evidence_URL",
        desc="Provide reference URL or official documentation confirming the timing of the initial NORS notification",
        parent=init_notif_node,
        critical=True
    )
    await evaluator.verify(
        claim="This webpage documents the timing of the NORS initial notification for the January 14, 2026 outage and supports that it met the 120-minute requirement.",
        node=notif_evidence_leaf,
        sources=merge_sources(data.notification_sources),
        additional_instruction=(
            "Pass if the page mentions the initial NORS notification timing for this incident and supports compliance with the 120-minute rule."
        )
    )

    # --- Stage 3: Initial Report within 3 days ---
    init_report_node = evaluator.add_parallel(
        id="Initial_Report_Verification",
        desc="Verify that the carrier submitted the initial outage report within 3 calendar days",
        parent=nors_node,
        critical=True  # Adjusted to satisfy critical parent constraint
    )

    three_day_check_leaf = evaluator.add_leaf(
        id="Three_Day_Report_Check",
        desc="Confirm that the initial outage report was submitted within 3 calendar days as required by FCC regulations",
        parent=init_report_node,
        critical=True
    )
    await evaluator.verify(
        claim="The carrier submitted the initial outage report within 3 calendar days of discovering the January 14, 2026 outage.",
        node=three_day_check_leaf,
        sources=merge_sources(data.initial_report_sources),
        additional_instruction=(
            "Pass if the page states or implies that the initial outage report was filed within 3 calendar days after discovery of the outage; "
            "if dates are provided, infer whether the difference is ≤3 days."
        )
    )

    init_report_evidence_leaf = evaluator.add_leaf(
        id="Initial_Report_Evidence_URL",
        desc="Provide reference URL or official documentation confirming the submission of the 3-day initial report",
        parent=init_report_node,
        critical=True
    )
    await evaluator.verify(
        claim="This webpage confirms the submission (or documented requirement compliance) of the 3-day initial outage report for the January 14, 2026 incident.",
        node=init_report_evidence_leaf,
        sources=merge_sources(data.initial_report_sources),
        additional_instruction=(
            "Pass if the page explicitly references the initial (3-day) outage report submission for this incident."
        )
    )

    # --- Stage 4: Final Report within 30 days ---
    final_report_node = evaluator.add_parallel(
        id="Final_Report_Verification",
        desc="Verify that the carrier submitted or committed to submit the final outage report within 30 days",
        parent=nors_node,
        critical=True  # Adjusted to satisfy critical parent constraint
    )

    thirty_day_check_leaf = evaluator.add_leaf(
        id="Thirty_Day_Report_Check",
        desc="Confirm that the final outage report was submitted or will be submitted within 30 days after discovering the outage",
        parent=final_report_node,
        critical=True
    )
    await evaluator.verify(
        claim="The carrier submitted or committed to submit the final outage report within 30 days after discovering the January 14, 2026 outage.",
        node=thirty_day_check_leaf,
        sources=merge_sources(data.final_report_sources),
        additional_instruction=(
            "Pass if the page confirms the final outage report was filed within 30 days or explicitly states a commitment to file within 30 days."
        )
    )

    final_report_evidence_leaf = evaluator.add_leaf(
        id="Final_Report_Evidence_URL",
        desc="Provide reference URL or official documentation confirming the submission or commitment to submit the 30-day final report",
        parent=final_report_node,
        critical=True
    )
    await evaluator.verify(
        claim="This webpage confirms the submission or explicit commitment to submit the final (30-day) outage report for the January 14, 2026 incident.",
        node=final_report_evidence_leaf,
        sources=merge_sources(data.final_report_sources),
        additional_instruction=(
            "Pass if the page explicitly references the final (30-day) outage report submission or commitment for this incident."
        )
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
    Evaluate an answer for the FCC NORS compliance task (Jan 14, 2026 outage).
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

    # Extract structured information from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_nors(),
        template_class=NORSComplianceExtraction,
        extraction_name="nors_extraction"
    )

    # Add high-level overall node
    overall_node = evaluator.add_parallel(
        id="Overall_Compliance_Assessment",
        desc="Comprehensive evaluation of telecommunications carrier's outage reporting compliance, including incident identification and FCC NORS requirement verification",
        parent=root,
        critical=True
    )

    # Build Incident Identification subtree
    await build_incident_identification(evaluator, overall_node, extracted)

    # Build NORS Compliance Verification subtree (sequential inside)
    await build_nors_compliance_verification(evaluator, overall_node, extracted)

    # Record custom info for debugging/traceability
    parsed_minutes = parse_duration_minutes(extracted.outage_duration)
    evaluator.add_custom_info(
        {
            "carrier_name": extracted.carrier_name,
            "outage_duration_text": extracted.outage_duration,
            "parsed_duration_minutes": parsed_minutes,
            "source_counts": {
                "incident_sources": len(extracted.sources or []),
                "notification_sources": len(extracted.notification_sources or []),
                "initial_report_sources": len(extracted.initial_report_sources or []),
                "final_report_sources": len(extracted.final_report_sources or []),
            }
        },
        info_type="extraction_summary"
    )

    return evaluator.get_summary()