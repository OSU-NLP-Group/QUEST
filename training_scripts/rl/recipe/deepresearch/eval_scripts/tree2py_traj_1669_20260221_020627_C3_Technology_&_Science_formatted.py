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
TASK_ID = "fcc_nors_outage"
TASK_DESCRIPTION = """Identify a major wireless telecommunications outage that occurred in the United States between January 2024 and January 2026 that would have triggered FCC Network Outage Reporting System (NORS) reporting requirements.

For the outage you identify, provide the following information:

1. Outage Identification:
   - The name of the wireless carrier that experienced the outage
   - The specific date the outage occurred
   - The approximate start time (or discovery time), duration, and resolution time of the outage
   - URL reference(s) to publicly available sources (news articles, carrier statements, or official announcements) documenting this outage

2. FCC Threshold Verification:
   - Demonstrate that the outage met the 30-minute minimum duration threshold for FCC NORS reporting
   - Demonstrate that the outage potentially affected at least 900,000 user-minutes of telephony service (the FCC threshold for wireless provider reporting)
   - Cite the specific FCC regulation (47 CFR Part 4) sections that establish these thresholds for wireless providers

3. Reporting Deadline Calculations:
   Based on when the wireless carrier discovered the outage, calculate the three FCC NORS reporting deadlines that would have applied:
   - The electronic notification deadline (within 120 minutes of discovery)
   - The Initial Communications Outage Report deadline (within 72 hours of discovery)
   - The Final Communications Outage Report deadline (within 30 days of discovery)
   
   For each deadline, specify the exact date and time when the report would have been due, and cite the relevant FCC regulation section.

Your answer should demonstrate that the outage you selected meets all FCC NORS reporting thresholds and should include all necessary documentation and regulatory citations to support your analysis.
"""

# Fallback official regulation URLs if the answer did not provide citation URLs
E_CFR_PART4_URL = "https://www.ecfr.gov/current/title-47/part-4"
E_CFR_4_9_URL = "https://www.ecfr.gov/current/title-47/part-4/section-4.9"

# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class OutageInfo(BaseModel):
    """Core outage identification and timeline details extracted from the answer."""
    carrier_name: Optional[str] = None
    outage_date: Optional[str] = None  # Prefer YYYY-MM-DD; allow textual date
    start_time: Optional[str] = None   # Include timezone label if available
    discovery_time: Optional[str] = None  # Include timezone label if available; can be same as start_time
    resolution_time: Optional[str] = None  # Include timezone label if available
    duration_minutes: Optional[str] = None  # Keep as a string to maximize compatibility
    outage_sources: List[str] = Field(default_factory=list)

    impacted_users_estimate: Optional[str] = None  # e.g., "1,500,000 customers", "millions"
    impacted_user_minutes_estimate: Optional[str] = None  # e.g., "over 3,000,000 user-minutes"
    impact_sources: List[str] = Field(default_factory=list)


class RegulationCitations(BaseModel):
    """FCC regulation citations and URLs required."""
    duration_threshold_citation_text: Optional[str] = None    # e.g., "47 CFR § 4.9"
    duration_threshold_url: Optional[str] = None

    user_minutes_threshold_citation_text: Optional[str] = None  # e.g., "47 CFR § 4.9(e)"
    user_minutes_threshold_url: Optional[str] = None

    notification_deadline_citation_text: Optional[str] = None  # e.g., "47 CFR § 4.9(e)(1)"
    notification_deadline_url: Optional[str] = None

    initial_report_citation_text: Optional[str] = None         # e.g., "47 CFR § 4.9(e)(4)"
    initial_report_url: Optional[str] = None

    final_report_citation_text: Optional[str] = None           # e.g., "47 CFR § 4.9(e)(4)"
    final_report_url: Optional[str] = None


class DeadlinesInfo(BaseModel):
    """Computed deadlines based on discovery time."""
    discovery_timestamp_iso: Optional[str] = None  # e.g., "2024-02-22T09:00:00-05:00"
    notification_deadline_iso: Optional[str] = None  # discovery + 120 minutes
    initial_report_deadline_iso: Optional[str] = None  # discovery + 72 hours
    final_report_deadline_iso: Optional[str] = None  # discovery + 30 days


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_outage() -> str:
    return """
    Select and extract details for ONE major U.S. wireless telecommunications outage that occurred between January 1, 2024 and January 31, 2026, and that reasonably would have triggered FCC NORS reporting for wireless providers.

    Provide the following fields EXTRACTED EXACTLY FROM THE ANSWER:
    - carrier_name: Name of the wireless carrier (e.g., AT&T, Verizon, T-Mobile)
    - outage_date: The specific date of the outage (prefer ISO YYYY-MM-DD; if not available, use textual date as given)
    - start_time: Approximate start time of the outage (include timezone abbreviation if provided)
    - discovery_time: When the carrier discovered the outage (include timezone)
    - resolution_time: When the outage was resolved (include timezone)
    - duration_minutes: Outage duration in minutes (if the answer expresses in hours or a range, provide the best numeric minutes estimate; otherwise leave as the exact text)
    - outage_sources: All URLs in the answer that document the outage (news articles, carrier statements, FCC, etc.)

    Impact information:
    - impacted_users_estimate: The approximate number of users/customers affected (extract text as-is, including qualifiers like "millions")
    - impacted_user_minutes_estimate: If the answer provides a calculated or stated user-minutes figure, extract it as-is (otherwise null)
    - impact_sources: All URLs in the answer that specifically discuss impact/scale (can overlap with outage_sources)

    IMPORTANT RULES:
    - Extract ONLY from the answer text. Do not invent or infer new URLs or values.
    - If any field is not present, return null (or empty list for URLs).
    - Preserve the original formatting for times and dates if ISO format isn't provided.
    """


def prompt_extract_reg_citations() -> str:
    return """
    Extract the FCC regulation citations and URLs mentioned in the answer that establish:
    1) The minimum reportable outage duration threshold for wireless providers (30 minutes), and
    2) The 900,000 user-minutes threshold for wireless providers.
    Also extract the citations for NORS reporting deadlines (notification, initial report, final report).

    Provide:
    - duration_threshold_citation_text
    - duration_threshold_url
    - user_minutes_threshold_citation_text
    - user_minutes_threshold_url
    - notification_deadline_citation_text
    - notification_deadline_url
    - initial_report_citation_text
    - initial_report_url
    - final_report_citation_text
    - final_report_url

    RULES:
    - Extract only citations and URLs explicitly present in the answer.
    - If any specific citation or URL is missing, return null for that field.
    """


def prompt_extract_deadlines(discovery_timestamp_hint: Optional[str]) -> str:
    hint_text = discovery_timestamp_hint or "null"
    return f"""
    Based on the outage discovery time provided in the answer, compute the three FCC NORS reporting deadlines.

    You must use the discovery timestamp from the answer. If the answer provides multiple times, prefer the explicit "discovery" time. If no ISO timestamp is present, convert to ISO if reasonably possible; otherwise, return the original text.

    Provide:
    - discovery_timestamp_iso: ISO-8601 timestamp of discovery (e.g., 2024-02-22T09:00:00-05:00). If the answer only provides textual time, return that text.
    - notification_deadline_iso: discovery + 120 minutes (2 hours)
    - initial_report_deadline_iso: discovery + 72 hours (3 days)
    - final_report_deadline_iso: discovery + 30 days

    RULES:
    - If timezone is ambiguous, keep the timezone label from the answer. If none, assume the event local timezone.
    - If you cannot compute exactly due to missing precision, provide best-effort computed timestamps with clear approximations (e.g., round to nearest hour) — but keep original discovery text in discovery_timestamp_iso if ISO rendering is impossible.
    - Do NOT fabricate any new times not reasonably supported by the answer.
    """

# --------------------------------------------------------------------------- #
# Utility helpers                                                             #
# --------------------------------------------------------------------------- #
def _parse_minutes_from_text(text: Optional[str]) -> Optional[int]:
    """Attempt to extract a minute count from freeform text. Supports minutes or hours."""
    if not text:
        return None
    s = text.strip().lower()

    # Match "X minutes", "X min"
    m_min = re.search(r'(\d+)\s*(minutes|minute|min)\b', s)
    if m_min:
        try:
            return int(m_min.group(1))
        except Exception:
            pass

    # Match "X.Y hours", "X hours", "X hr", "X h"
    m_hr = re.search(r'(\d+(?:\.\d+)?)\s*(hours|hour|hrs|hr|h)\b', s)
    if m_hr:
        try:
            hours = float(m_hr.group(1))
            return int(round(hours * 60))
        except Exception:
            pass

    # Fallback: bare integer (assume minutes)
    m_int = re.search(r'\b(\d{1,4})\b', s)
    if m_int:
        try:
            return int(m_int.group(1))
        except Exception:
            pass

    return None


def _parse_int_from_text(text: Optional[str]) -> Optional[int]:
    """Extract a large integer (e.g., number of users) from freeform text."""
    if not text:
        return None
    # Handle words like "millions"
    s = text.strip().lower()
    # "X million", "X.X million"
    m_million = re.search(r'(\d+(?:\.\d+)?)\s*million', s)
    if m_million:
        try:
            val = float(m_million.group(1)) * 1_000_000
            return int(round(val))
        except Exception:
            pass

    # Raw digits with commas
    m_digits = re.search(r'(\d{1,3}(?:,\d{3})+|\d{4,})', s)
    if m_digits:
        try:
            return int(m_digits.group(1).replace(",", ""))
        except Exception:
            pass

    return None


def _pick_reg_url(primary: Optional[str], fallback: str) -> str:
    return primary if (primary and primary.strip()) else fallback


# --------------------------------------------------------------------------- #
# Verification subtrees                                                       #
# --------------------------------------------------------------------------- #
async def build_outage_identification(
    evaluator: Evaluator,
    parent_node,
    outage: OutageInfo,
) -> None:
    """
    Build the 'outage_identification' subtree:
    - Basic info (carrier/date/timeline/sources)
    - Threshold verification (duration >= 30; user-minutes >= 900,000)
    """
    # Parent: outage_identification (sequential)
    ident_node = evaluator.add_sequential(
        id="outage_identification",
        desc="Identify a recent major wireless carrier outage that meets FCC NORS reporting thresholds",
        parent=parent_node,
        critical=False,
    )

    # 1) Basic Information (parallel, critical)
    basic_node = evaluator.add_parallel(
        id="outage_basic_information",
        desc="Document the outage's carrier name, date, and general timeline",
        parent=ident_node,
        critical=True,
    )

    # 1.1) Carrier & Date existence check (critical precondition)
    carrier_date_exists = evaluator.add_custom_node(
        result=bool(outage.carrier_name and outage.carrier_name.strip() and outage.outage_date and outage.outage_date.strip()),
        id="carrier_and_date_exists",
        desc="Carrier name and outage date are provided",
        parent=basic_node,
        critical=True,
    )

    # 1.2) Carrier & Date supported by sources (leaf)
    carrier_date_supported = evaluator.add_leaf(
        id="carrier_and_date",
        desc="Provide the name of the wireless carrier and the specific date of the outage",
        parent=basic_node,
        critical=True,
    )
    claim_cd = f"The carrier {outage.carrier_name or ''} experienced a network outage on {outage.outage_date or ''}."
    await evaluator.verify(
        claim=claim_cd,
        node=carrier_date_supported,
        sources=outage.outage_sources,
        additional_instruction="Verify that the sources document the stated carrier and the specific outage date. Allow minor formatting differences (e.g., 'Feb 22, 2024' vs '2024-02-22').",
    )

    # 1.3) Timeline (parallel, critical)
    timeline_node = evaluator.add_parallel(
        id="outage_timeline",
        desc="Provide the approximate start time, discovery time, and resolution time of the outage",
        parent=basic_node,
        critical=True,
    )

    # Separate leaves to verify each timeline component with sources
    start_leaf = evaluator.add_leaf(
        id="timeline_start_time_supported",
        desc="Outage start time is supported by cited sources",
        parent=timeline_node,
        critical=True,
    )
    disc_leaf = evaluator.add_leaf(
        id="timeline_discovery_time_supported",
        desc="Outage discovery time is supported by cited sources",
        parent=timeline_node,
        critical=True,
    )
    resolve_leaf = evaluator.add_leaf(
        id="timeline_resolution_time_supported",
        desc="Outage resolution time is supported by cited sources",
        parent=timeline_node,
        critical=True,
    )

    # Batch verify timeline components
    tl_claims = [
        (
            f"The outage started around {outage.start_time or 'N/A'}.",
            outage.outage_sources,
            start_leaf,
            "Verify that the sources provide or imply an approximate start time; allow hour-level approximations and timezone labels."
        ),
        (
            f"The outage was discovered around {outage.discovery_time or outage.start_time or 'N/A'}.",
            outage.outage_sources,
            disc_leaf,
            "Verify that the sources mention the carrier's discovery or acknowledgment time; allow approximations."
        ),
        (
            f"The outage was resolved around {outage.resolution_time or 'N/A'}.",
            outage.outage_sources,
            resolve_leaf,
            "Verify that the sources indicate when service was restored; allow approximations."
        ),
    ]
    await evaluator.batch_verify(tl_claims)

    # 1.4) Source Documentation existence
    src_exists = evaluator.add_custom_node(
        result=bool(outage.outage_sources),
        id="source_documentation_exists",
        desc="Outage documentation sources are provided",
        parent=basic_node,
        critical=True,
    )
    # 1.5) Source Documentation relevance
    src_relevance = evaluator.add_leaf(
        id="source_documentation",
        desc="Provide URL reference(s) to publicly available sources documenting the outage",
        parent=basic_node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"These sources document the outage of {outage.carrier_name or ''} on {outage.outage_date or ''}.",
        node=src_relevance,
        sources=outage.outage_sources,
        additional_instruction="Confirm that the URLs are about the stated outage and carrier. If multiple URLs are provided, any single URL that supports the claim suffices.",
    )

    # 2) Threshold Verification (parallel, critical)
    thr_node = evaluator.add_parallel(
        id="threshold_verification",
        desc="Verify that the outage meets both FCC NORS reporting thresholds: 30-minute duration and 900,000 user-minutes impact",
        parent=ident_node,
        critical=True,
    )

    # 2.1) Duration threshold (parallel, critical)
    dur_node = evaluator.add_parallel(
        id="duration_threshold",
        desc="Verify and document that the outage lasted at least 30 minutes",
        parent=thr_node,
        critical=True,
    )

    # Duration calculation: rely on duration_minutes extracted (convert if possible)
    duration_minutes_int = _parse_minutes_from_text(outage.duration_minutes)
    duration_calc_leaf = evaluator.add_custom_node(
        result=bool(duration_minutes_int is not None and duration_minutes_int >= 30),
        id="duration_calculation",
        desc="Calculate the total outage duration from start to resolution time (must be ≥ 30 minutes)",
        parent=dur_node,
        critical=True,
    )

    # Duration threshold reference (verify with FCC)
    dur_ref_leaf = evaluator.add_leaf(
        id="duration_threshold_reference",
        desc="Cite the FCC regulation (47 CFR § 4.9) that establishes the 30-minute threshold",
        parent=dur_node,
        critical=True,
    )
    await evaluator.verify(
        claim="47 CFR § 4.9 establishes a minimum reportable outage duration threshold of 30 minutes for providers subject to Part 4 reporting.",
        node=dur_ref_leaf,
        sources=_pick_reg_url(None, E_CFR_4_9_URL),
        additional_instruction="Verify that the regulation text in Part 4 or §4.9 indicates that outages lasting at least 30 minutes are reportable.",
    )

    # 2.2) User-minutes threshold (parallel, critical)
    um_node = evaluator.add_parallel(
        id="user_minutes_threshold",
        desc="Verify that the outage potentially affected at least 900,000 user-minutes of telephony service",
        parent=thr_node,
        critical=True,
    )

    impacted_users_int = _parse_int_from_text(outage.impacted_users_estimate)
    # If impacted_user_minutes_estimate given, parse it; otherwise compute users * minutes
    impacted_user_minutes_int = _parse_int_from_text(outage.impacted_user_minutes_estimate)
    if impacted_user_minutes_int is None and (impacted_users_int is not None and duration_minutes_int is not None):
        impacted_user_minutes_int = impacted_users_int * duration_minutes_int

    um_doc_leaf = evaluator.add_leaf(
        id="impact_documentation",
        desc="Document evidence that the outage affected enough users for enough duration to meet the 900,000 user-minutes threshold",
        parent=um_node,
        critical=True,
    )
    claim_um = (
        f"The outage meets or exceeds 900,000 user-minutes (users × minutes). "
        f"Impacted users: {outage.impacted_users_estimate or 'N/A'}; "
        f"Duration: {outage.duration_minutes or 'N/A'}; "
        f"Computed/estimated user-minutes: {impacted_user_minutes_int if impacted_user_minutes_int is not None else 'N/A'}."
    )
    await evaluator.verify(
        claim=claim_um,
        node=um_doc_leaf,
        sources=(outage.impact_sources if outage.impact_sources else outage.outage_sources),
        additional_instruction="Confirm that the sources support a plausible user-minutes figure ≥ 900,000, either directly or by reasonable multiplication of users affected by duration.",
    )

    um_ref_leaf = evaluator.add_leaf(
        id="user_minutes_threshold_reference",
        desc="Cite the FCC regulation (47 CFR § 4.9(e)) that establishes the 900,000 user-minutes threshold for wireless providers",
        parent=um_node,
        critical=True,
    )
    await evaluator.verify(
        claim="47 CFR § 4.9(e) sets a 900,000 user-minutes threshold for wireless providers' reportable outages.",
        node=um_ref_leaf,
        sources=_pick_reg_url(None, E_CFR_4_9_URL),
        additional_instruction="Verify that the regulation indicates the 900,000 user-minutes criterion for wireless providers (often in §4.9(e)).",
    )


async def build_reporting_deadlines(
    evaluator: Evaluator,
    parent_node,
    deadlines: DeadlinesInfo,
    regs: RegulationCitations,
) -> None:
    """
    Build the 'reporting_deadlines' subtree, verifying computed deadlines and regulation citations.
    """
    rep_node = evaluator.add_parallel(
        id="reporting_deadlines",
        desc="Calculate all three FCC NORS reporting deadlines based on the outage discovery time",
        parent=parent_node,
        critical=False,
    )

    # For each deadline, we verify the arithmetic via simple verification (non-web factual math),
    # and we verify the regulatory citation via URL separately.

    # 1) Notification (120 minutes)
    notif_calc_leaf = evaluator.add_leaf(
        id="notification_deadline_calc",
        desc="Electronic notification deadline is correctly calculated (120 minutes after discovery)",
        parent=rep_node,
        critical=True,
    )
    claim_notif = (
        f"Given discovery time '{deadlines.discovery_timestamp_iso or 'N/A'}', "
        f"the electronic notification deadline (120 minutes after discovery) is '{deadlines.notification_deadline_iso or 'N/A'}'."
    )
    await evaluator.verify(
        claim=claim_notif,
        node=notif_calc_leaf,
        sources=None,
        additional_instruction="Check that adding exactly 120 minutes to the discovery time yields the stated notification deadline.",
    )

    notif_cite_leaf = evaluator.add_leaf(
        id="notification_deadline",
        desc="Cite 47 CFR § 4.9(e)(1) for the electronic notification deadline",
        parent=rep_node,
        critical=True,
    )
    await evaluator.verify(
        claim="47 CFR § 4.9(e)(1) requires electronic notification within 120 minutes of discovery for wireless providers.",
        node=notif_cite_leaf,
        sources=_pick_reg_url(regs.notification_deadline_url, E_CFR_4_9_URL),
        additional_instruction="Verify that the regulation text requires notification within 120 minutes of discovery.",
    )

    # 2) Initial report (72 hours)
    initial_calc_leaf = evaluator.add_leaf(
        id="initial_report_deadline_calc",
        desc="Initial Communications Outage Report deadline is correctly calculated (72 hours after discovery)",
        parent=rep_node,
        critical=True,
    )
    claim_initial = (
        f"Given discovery time '{deadlines.discovery_timestamp_iso or 'N/A'}', "
        f"the Initial Report deadline (72 hours after discovery) is '{deadlines.initial_report_deadline_iso or 'N/A'}'."
    )
    await evaluator.verify(
        claim=claim_initial,
        node=initial_calc_leaf,
        sources=None,
        additional_instruction="Check that adding exactly 72 hours to the discovery time yields the stated Initial Report deadline.",
    )

    initial_cite_leaf = evaluator.add_leaf(
        id="initial_report_deadline",
        desc="Cite 47 CFR § 4.9(e)(4) for the Initial Communications Outage Report deadline",
        parent=rep_node,
        critical=True,
    )
    await evaluator.verify(
        claim="47 CFR § 4.9(e)(4) requires the Initial Communications Outage Report within 72 hours of discovery.",
        node=initial_cite_leaf,
        sources=_pick_reg_url(regs.initial_report_url, E_CFR_4_9_URL),
        additional_instruction="Verify that the regulation text requires the Initial Report within 72 hours of discovery.",
    )

    # 3) Final report (30 days)
    final_calc_leaf = evaluator.add_leaf(
        id="final_report_deadline_calc",
        desc="Final Communications Outage Report deadline is correctly calculated (30 days after discovery)",
        parent=rep_node,
        critical=True,
    )
    claim_final = (
        f"Given discovery time '{deadlines.discovery_timestamp_iso or 'N/A'}', "
        f"the Final Report deadline (30 days after discovery) is '{deadlines.final_report_deadline_iso or 'N/A'}'."
    )
    await evaluator.verify(
        claim=claim_final,
        node=final_calc_leaf,
        sources=None,
        additional_instruction="Check that adding exactly 30 days to the discovery time yields the stated Final Report deadline.",
    )

    final_cite_leaf = evaluator.add_leaf(
        id="final_report_deadline",
        desc="Cite 47 CFR § 4.9(e)(4) for the Final Communications Outage Report deadline",
        parent=rep_node,
        critical=True,
    )
    await evaluator.verify(
        claim="47 CFR § 4.9(e)(4) requires the Final Communications Outage Report within 30 days of discovery.",
        node=final_cite_leaf,
        sources=_pick_reg_url(regs.final_report_url, E_CFR_4_9_URL),
        additional_instruction="Verify that the regulation text requires the Final Report within 30 days of discovery.",
    )


# --------------------------------------------------------------------------- #
# Main evaluation function                                                    #
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
) -> Dict[str, Any]:
    """
    Evaluate an answer for the FCC NORS outage compliance task.
    """
    # Initialize evaluator
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.SEQUENTIAL,  # Evaluate in order: identification -> thresholds -> deadlines
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

    # Note: Make root non-critical to allow mixed critical children (per framework constraints)
    root.critical = False

    # Record ground truth thresholds for context
    evaluator.add_ground_truth({
        "fcc_thresholds": {
            "duration_minutes_min": 30,
            "user_minutes_min": 900_000,
            "regulations_expected": [
                "47 CFR § 4.9",
                "47 CFR § 4.9(e)",
                "47 CFR § 4.9(e)(1)",
                "47 CFR § 4.9(e)(4)"
            ]
        }
    }, gt_type="thresholds")

    # 1) Extract outage info
    outage_info: OutageInfo = await evaluator.extract(
        prompt=prompt_extract_outage(),
        template_class=OutageInfo,
        extraction_name="outage_info",
    )

    # 2) Extract regulation citations
    reg_citations: RegulationCitations = await evaluator.extract(
        prompt=prompt_extract_reg_citations(),
        template_class=RegulationCitations,
        extraction_name="regulation_citations",
    )

    # 3) Extract deadlines (provide discovery time from previous extraction as hint)
    deadlines_info: DeadlinesInfo = await evaluator.extract(
        prompt=prompt_extract_deadlines(discovery_timestamp_hint=outage_info.discovery_time or outage_info.start_time),
        template_class=DeadlinesInfo,
        extraction_name="deadlines_info",
        additional_instruction="Use the discovery time extracted from the answer to compute the exact deadlines as specified."
    )

    # Add custom info: computed numeric helper values (for debugging/clarity)
    duration_minutes_int = _parse_minutes_from_text(outage_info.duration_minutes)
    impacted_users_int = _parse_int_from_text(outage_info.impacted_users_estimate)
    impacted_user_minutes_int = _parse_int_from_text(outage_info.impacted_user_minutes_estimate)
    if impacted_user_minutes_int is None and (impacted_users_int is not None and duration_minutes_int is not None):
        impacted_user_minutes_int = impacted_users_int * duration_minutes_int

    evaluator.add_custom_info(
        info={
            "duration_minutes_int": duration_minutes_int,
            "impacted_users_int": impacted_users_int,
            "impacted_user_minutes_int": impacted_user_minutes_int
        },
        info_type="computed_helper",
        info_name="numeric_estimates"
    )

    # Build verification subtrees
    await build_outage_identification(evaluator, root, outage_info)

    # Reporting deadlines come after identification/thresholds due to root sequential strategy
    await build_reporting_deadlines(evaluator, root, deadlines_info, reg_citations)

    # Return evaluation summary
    return evaluator.get_summary()