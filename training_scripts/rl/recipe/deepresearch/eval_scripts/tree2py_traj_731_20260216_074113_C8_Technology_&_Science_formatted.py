import asyncio
import logging
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "telecom_outages_2024_2026"
TASK_DESCRIPTION = """
Identify four major telecommunications carrier outages in the United States that occurred between January 2024 and February 2026. For each outage, the incident must have met the FCC Network Outage Reporting System (NORS) minimum thresholds: lasting at least 30 minutes and affecting at least 30,000 users. The outages should have impacted multiple major U.S. cities across different states.

For each of the four outages, provide the following information:

1. Carrier Name
2. Outage Date
3. Outage Start Time (with time zone)
4. Outage Duration (hours)
5. Root Cause (publicly acknowledged by the carrier)
6. Three Affected Cities (at least)
7. Number of Affected States
8. Estimated Affected Users (≥ 30,000)
9. NORS Reporting Compliance (filed within 72 hours if publicly available; otherwise explicitly state not publicly available)
10. PSAP Notification (if 911 impacted; else not applicable / not publicly available)
11. Customer Compensation
12. FCC Investigation Status
13. Service Restoration Confirmation
14. Primary Source URL
15. Secondary Source URL

Ensure the four outages are distinct incidents and all information is verifiable via publicly available sources.
"""


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class OutageItem(BaseModel):
    carrier_name: Optional[str] = None
    outage_date: Optional[str] = None
    outage_start_time: Optional[str] = None
    outage_duration_hours: Optional[str] = None
    root_cause: Optional[str] = None
    affected_cities: List[str] = Field(default_factory=list)
    affected_states_count: Optional[str] = None
    estimated_affected_users: Optional[str] = None
    nors_reporting_compliance: Optional[str] = None
    psap_notification: Optional[str] = None
    customer_compensation: Optional[str] = None
    fcc_investigation_status: Optional[str] = None
    service_restoration_confirmation: Optional[str] = None
    primary_source_url: Optional[str] = None
    secondary_source_url: Optional[str] = None


class OutagesExtraction(BaseModel):
    outages: List[OutageItem] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_outages() -> str:
    return """
Extract all outages mentioned in the answer and structure them as a list of objects with the following fields. Extract ONLY what is explicitly stated in the answer (do not infer). If a field is not provided in the answer, return null (or an empty list for cities).

For each outage, extract:
- carrier_name
- outage_date (e.g., "Feb 22, 2024" or "2024-02-22")
- outage_start_time (include a time zone if present, e.g., "10:15 AM ET", "07:00 PST", "14:30 UTC-5")
- outage_duration_hours (keep as a string, e.g., "3 hours", "0.75 hours", "45 minutes")
- root_cause (as publicly acknowledged by the carrier, keep category-like phrasing if present)
- affected_cities (array of at least three city names if provided)
- affected_states_count (keep as a string or number-like string; do not infer)
- estimated_affected_users (keep as a string exactly as stated; e.g., "100,000+", "around 45,000")
- nors_reporting_compliance (one of: "filed within 72 hours", "filed later than 72 hours", "not publicly available", or exactly as stated)
- psap_notification (e.g., "not applicable", "PSAPs notified within 30 minutes", "not publicly available", or exactly as stated)
- customer_compensation (e.g., "bill credit", "data credit", "refund", or a sentence describing compensation)
- fcc_investigation_status (e.g., "investigation launched", "no public investigation", "not publicly available")
- service_restoration_confirmation (e.g., "restoration confirmed", "service restored at 5 PM ET")
- primary_source_url (must be an explicit URL in the answer; if missing, null)
- secondary_source_url (must be an explicit URL in the answer; if missing, null)

Important:
- Only extract outages that the answer explicitly lists.
- Do not invent URLs; only extract URLs that are actually present in the answer text. If a URL is missing protocol, prepend http:// as needed.
"""


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _is_nonempty(s: Optional[str]) -> bool:
    return bool(s and str(s).strip())


def _valid_url(s: Optional[str]) -> bool:
    if not _is_nonempty(s):
        return False
    st = s.strip()
    return st.startswith("http://") or st.startswith("https://")


def _outage_signature(oi: OutageItem) -> str:
    """
    Signature for distinctness check.
    Prefer primary_source_url; else combine carrier, date, start_time, duration.
    """
    if _is_nonempty(oi.primary_source_url):
        return oi.primary_source_url.strip().lower()
    parts = [
        (oi.carrier_name or "").strip().lower(),
        (oi.outage_date or "").strip().lower(),
        (oi.outage_start_time or "").strip().lower(),
        (oi.outage_duration_hours or "").strip().lower(),
        (oi.estimated_affected_users or "").strip().lower(),
    ]
    return "|".join(parts)


def _first_k_or_pad(items: List[OutageItem], k: int) -> List[OutageItem]:
    out = items[:k]
    while len(out) < k:
        out.append(OutageItem())
    return out


def _safe_url_list(*urls: Optional[str]) -> List[str]:
    return [u for u in urls if _valid_url(u)]


def _fmt_list(vals: List[str]) -> str:
    return ", ".join(vals) if vals else ""


# --------------------------------------------------------------------------- #
# Verification for one outage                                                 #
# --------------------------------------------------------------------------- #
async def verify_one_outage(evaluator: Evaluator, parent_node, outage: OutageItem, idx: int) -> None:
    """
    Build and verify the subtree for a single outage.
    """
    outage_node = evaluator.add_parallel(
        id=f"outage_{idx+1}",
        desc=f"Outage {idx + 1} (one incident) details and evidence",
        parent=parent_node,
        critical=False  # allow partial credit across outages
    )

    # Convenience
    urls = _safe_url_list(outage.primary_source_url, outage.secondary_source_url)
    cities = outage.affected_cities or []
    cities_str = _fmt_list(cities)

    # 1) Carrier name provided (critical)
    evaluator.add_custom_node(
        result=_is_nonempty(outage.carrier_name),
        id=f"outage_{idx+1}_carrier_name",
        desc="Carrier name is provided",
        parent=outage_node,
        critical=True
    )

    # 2) Date in range (critical) – LLM simple check
    date_node = evaluator.add_leaf(
        id=f"outage_{idx+1}_date_in_range",
        desc="Outage date is provided and falls between January 2024 and February 2026 (inclusive)",
        parent=outage_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The outage date '{outage.outage_date}' falls between Jan 1, 2024 and Feb 29, 2026 (inclusive).",
        node=date_node,
        additional_instruction="Interpret common US date formats. If the date is a single day within this window, mark Correct. If missing or clearly outside, mark Incorrect."
    )

    # 3) Start time includes timezone (critical) – LLM simple check
    st_node = evaluator.add_leaf(
        id=f"outage_{idx+1}_start_time_with_timezone",
        desc="Approximate outage start time is provided and includes a time zone",
        parent=outage_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The start time string '{outage.outage_start_time}' includes an explicit time zone indicator (e.g., ET/EST/EDT/CT/CDT/PT/PDT/MST, UTC, or a UTC offset like UTC-5).",
        node=st_node,
        additional_instruction="If the string contains a recognizable US time zone abbreviation or a UTC/offset notation, mark Correct; otherwise Incorrect."
    )

    # 4) Duration meets NORS minimum ≥ 0.5 hours (critical) – LLM simple check
    dur_node = evaluator.add_leaf(
        id=f"outage_{idx+1}_duration_meets_threshold",
        desc="Outage duration is provided in hours and meets the NORS minimum threshold (≥ 30 minutes / ≥ 0.5 hours)",
        parent=outage_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The outage duration '{outage.outage_duration_hours}' indicates at least 0.5 hours (30 minutes).",
        node=dur_node,
        additional_instruction="Interpret common duration strings (e.g., '45 minutes', '0.75 hours', '3 hours'). If ≥ 30 minutes, mark Correct."
    )

    # 5) Users meet threshold ≥ 30,000 (critical) – LLM simple check
    users_node = evaluator.add_leaf(
        id=f"outage_{idx+1}_users_meet_threshold",
        desc="Estimated affected users is provided and meets the NORS minimum threshold (≥ 30,000 users)",
        parent=outage_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The estimated affected users '{outage.estimated_affected_users}' is at least 30,000.",
        node=users_node,
        additional_instruction="Allow format variations like '100k', '100,000+', 'around 45,000'. If clearly ≥ 30,000, mark Correct."
    )

    # 6) Root cause publicly acknowledged (critical) – verify by URLs
    rc_node = evaluator.add_leaf(
        id=f"outage_{idx+1}_root_cause_publicly_acknowledged",
        desc="Root-cause category is provided and is publicly acknowledged by the carrier",
        parent=outage_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The carrier publicly attributed this outage to: {outage.root_cause}.",
        node=rc_node,
        sources=urls,
        additional_instruction="Look for explicit attribution by the carrier (e.g., software issue, configuration error, fiber cut). If a source explicitly attributes the cause, mark Correct."
    )

    # 7) Affected cities requirement (critical) – LLM simple check on list content
    cities_node = evaluator.add_leaf(
        id=f"outage_{idx+1}_affected_cities_requirement",
        desc="At least three affected major U.S. cities are identified, and the listed cities span at least two different U.S. states",
        parent=outage_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The following cities were affected: [{cities_str}]. These include at least three major U.S. cities and span at least two different U.S. states.",
        node=cities_node,
        additional_instruction="Judge based on the city list alone: if it has at least 3 recognizable major US cities and they are from at least two states, mark Correct."
    )

    # 8) Number of affected states provided (critical) – existence check
    evaluator.add_custom_node(
        result=_is_nonempty(outage.affected_states_count),
        id=f"outage_{idx+1}_affected_states_count_provided",
        desc="Number of affected U.S. states is provided",
        parent=outage_node,
        critical=True
    )

    # 9) NORS reporting compliance if public (critical) – content compliance check
    nors_node = evaluator.add_leaf(
        id=f"outage_{idx+1}_nors_reporting_compliance_if_public",
        desc="Indicates whether an initial NORS report was filed within 72 hours if publicly available; otherwise explicitly states that this information is not publicly available",
        parent=outage_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The NORS compliance statement '{outage.nors_reporting_compliance}' either indicates filing within 72 hours (or later) OR explicitly states that the information is not publicly available.",
        node=nors_node,
        additional_instruction="Acceptable if the text clearly conveys 'filed within 72 hours', 'filed late', 'not publicly available', or equivalent. Focus on the statement itself, not external evidence."
    )

    # 10) PSAP notification if applicable (critical) – content compliance check
    psap_node = evaluator.add_leaf(
        id=f"outage_{idx+1}_psap_notification_if_applicable",
        desc="If the outage impacted 911 service and this is publicly available, indicates whether affected PSAPs were notified within 30 minutes; otherwise states not applicable/not publicly available",
        parent=outage_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The PSAP notification statement '{outage.psap_notification}' indicates either timely PSAP notification (≤30 minutes) when applicable, or explicitly 'not applicable'/'not publicly available'.",
        node=psap_node,
        additional_instruction="Judge based on the phrasing. Do not require external evidence here; accept 'not applicable' or 'not publicly available' when stated."
    )

    # 11) Customer compensation info (critical) – existence check
    evaluator.add_custom_node(
        result=_is_nonempty(outage.customer_compensation),
        id=f"outage_{idx+1}_customer_compensation_info",
        desc="Describes customer compensation offered (credits/refunds/other consideration)",
        parent=outage_node,
        critical=True
    )

    # 12) FCC investigation status (critical) – conditional verification
    fcc_node = evaluator.add_leaf(
        id=f"outage_{idx+1}_fcc_investigation_status",
        desc="Indicates whether the FCC launched an official investigation into the outage",
        parent=outage_node,
        critical=True
    )
    # Build conditional claim
    fcc_text = (outage.fcc_investigation_status or "").strip().lower()
    if any(k in fcc_text for k in ["yes", "launched", "opened", "announced"]):
        claim = "The FCC launched an official investigation into this outage."
        add_ins = "Look for explicit statements from the FCC or credible reporting that the FCC launched an investigation."
        await evaluator.verify(
            claim=claim,
            node=fcc_node,
            sources=urls,
            additional_instruction=add_ins
        )
    elif any(k in fcc_text for k in ["no", "not publicly available", "unknown"]):
        # For lack of public info or explicit 'no', judge the statement itself (non-web factual check)
        await evaluator.verify(
            claim=f"The FCC investigation status statement '{outage.fcc_investigation_status}' is a valid indication (e.g., 'no public investigation' or 'not publicly available').",
            node=fcc_node,
            additional_instruction="Check if the statement clearly communicates 'no public investigation' or 'not publicly available'. Do not require external evidence for this negative/availability statement."
        )
    else:
        # If unclear text, likely to fail by LLM judgment
        await evaluator.verify(
            claim=f"The FCC investigation status statement '{outage.fcc_investigation_status}' clearly indicates whether an official investigation was launched.",
            node=fcc_node,
            additional_instruction="If the statement is unclear or empty, mark Incorrect."
        )

    # 13) Service restoration confirmation (critical) – verify by URLs
    restore_node = evaluator.add_leaf(
        id=f"outage_{idx+1}_service_restoration_public_confirmation",
        desc="Confirms the carrier publicly announced service restoration",
        parent=outage_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The carrier publicly announced that service was restored for this outage.",
        node=restore_node,
        sources=urls,
        additional_instruction="Look for explicit acknowledgement that service was restored/resolved, from the carrier or credible coverage quoting the carrier."
    )

    # 14) Primary source URL provided (critical) – existence/format
    evaluator.add_custom_node(
        result=_valid_url(outage.primary_source_url),
        id=f"outage_{idx+1}_primary_source_url",
        desc="Provides a primary source URL documenting the outage",
        parent=outage_node,
        critical=True
    )

    # 15) Secondary source URL provided (critical) – existence/format
    evaluator.add_custom_node(
        result=_valid_url(outage.secondary_source_url),
        id=f"outage_{idx+1}_secondary_source_url",
        desc="Provides a secondary source URL corroborating the outage information",
        parent=outage_node,
        critical=True
    )

    # 16) Sources substantiate key claims (critical) – verify by URLs
    support_node = evaluator.add_leaf(
        id=f"outage_{idx+1}_sources_support_claims",
        desc="Sources substantiate date/time window, duration/impact thresholds, multi-city/multi-state impact, and restoration acknowledgement",
        parent=outage_node,
        critical=True
    )
    # Summarize key claims into one verification
    claim_summary = (
        f"For this outage involving {outage.carrier_name}, on {outage.outage_date} starting around '{outage.outage_start_time}', "
        f"lasting '{outage.outage_duration_hours}', affecting at least '{outage.estimated_affected_users}' users, "
        f"impacting multiple major cities including [{cities_str}] across multiple states (claimed {outage.affected_states_count}), "
        f"and with the carrier later announcing service restoration."
    )
    await evaluator.verify(
        claim=claim_summary,
        node=support_node,
        sources=urls,
        additional_instruction=(
            "Using the provided URLs, verify that the pages collectively support: (1) date/time window, "
            "(2) duration/impact meeting or exceeding 30 minutes and 30,000 users, "
            "(3) multi-city and multi-state impact, and (4) carrier restoration acknowledgement. "
            "Allow reasonable approximations (e.g., 'morning' vs exact time) and minor rounding."
        )
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
    model: str = "o4-mini"
) -> Dict:
    """
    Evaluate a single answer for the telecom outages (Jan 2024–Feb 2026) task and return a structured result dictionary.
    """
    # 1) Initialize evaluator (root non-critical to avoid critical-consistency constraint)
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

    # 2) Extract structured outages from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_outages(),
        template_class=OutagesExtraction,
        extraction_name="extracted_outages"
    )

    outages: List[OutageItem] = extracted.outages or []

    # 3) Set-level requirements node
    set_level = evaluator.add_parallel(
        id="set_level_requirements",
        desc="Answer-level requirements that apply to the full set of outages",
        parent=root,
        critical=True  # Critical as per rubric
    )

    # Compute exact-count and distinctness
    nonempty_outages = [o for o in outages if _is_nonempty(o.carrier_name)]
    count = len(nonempty_outages)
    signatures = [_outage_signature(o) for o in nonempty_outages]
    unique_count = len(set(signatures))

    exactly_four_distinct = (count == 4) and (unique_count == 4)

    evaluator.add_custom_node(
        result=exactly_four_distinct,
        id="provides_exactly_four_distinct_outages",
        desc="Provides exactly four outages, and they are four distinct incidents",
        parent=set_level,
        critical=True
    )

    # Record custom info
    evaluator.add_custom_info(
        info={
            "extracted_total": len(outages),
            "nonempty_count": count,
            "unique_count": unique_count,
            "signatures": signatures[:6]
        },
        info_type="extraction_stats",
        info_name="extraction_statistics"
    )

    # 4) Evaluate exactly four outages (pad or truncate to four for verification)
    to_verify = _first_k_or_pad(outages, 4)

    # Build four outage subtrees
    tasks = []
    for i in range(4):
        tasks.append(verify_one_outage(evaluator, root, to_verify[i], i))
    # Run sequentially to maintain predictable logs; could also gather
    for t in tasks:
        await t

    # 5) Return evaluation summary
    return evaluator.get_summary()