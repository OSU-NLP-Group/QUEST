import asyncio
import logging
import re
from datetime import datetime, date
from typing import Optional, List, Dict, Any, Tuple

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "major_us_telco_outage_2024_2026"
TASK_DESCRIPTION = """
Identify and document a major telecommunications outage that occurred in the United States between January 1, 2024 and February 21, 2026, involving one of the three major wireless carriers (Verizon, AT&T, or T-Mobile). The outage must meet the following criteria: (1) The outage lasted at least 30 minutes, meeting the FCC's Network Outage Reporting System (NORS) minimum duration threshold for reportable outages; (2) The outage affected at least 100,000 users at its peak; (3) The outage affected multiple US states (not limited to a single state); (4) The outage disrupted wireless services such as voice calls, text messaging, and/or mobile data. For the identified outage, provide the following comprehensive documentation: the name of the wireless carrier, the specific date the outage occurred, the types of wireless services that were disrupted, a URL reference to an official statement or announcement from the carrier about the outage, and a URL reference to credible news media coverage of the outage. Additionally, if available, include: major cities that were affected, the states that were affected, information about when service was restored, and details about any customer remediation actions taken by the carrier (such as account credits or compensation).
"""

DATE_RANGE_START = date(2024, 1, 1)
DATE_RANGE_END = date(2026, 2, 21)


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class OutageExtraction(BaseModel):
    """
    Structured extraction of a single major US wireless outage as presented in the answer.
    NOTE: Extract exactly what the answer provides; do not invent details.
    """
    carrier: Optional[str] = None  # e.g., "AT&T", "Verizon", "T-Mobile"
    outage_date: Optional[str] = None  # e.g., "February 22, 2024", "2024-02-22"
    services_disrupted: List[str] = Field(default_factory=list)  # canonical: "voice calls", "text messaging", "mobile data"
    duration_description: Optional[str] = None  # e.g., "several hours", "about 8 hours", "at least 30 minutes"
    impact_description: Optional[str] = None  # e.g., "millions of users", "over 100,000 customers"
    major_cities_affected: List[str] = Field(default_factory=list)  # e.g., ["New York", "Los Angeles"]
    states_affected: List[str] = Field(default_factory=list)  # e.g., ["California", "Texas"]
    official_statement_url: Optional[str] = None  # carrier official site or official social media post about the outage
    news_coverage_url: Optional[str] = None  # credible media coverage URL
    restoration_info: Optional[str] = None  # e.g., "Service restored by 3 PM ET"
    remediation_actions: Optional[str] = None  # e.g., "Bill credits were offered to affected customers"


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_outage() -> str:
    return """
    You must extract a single major US wireless outage as described in the answer. Extract only what is explicitly stated in the answer text.

    Required fields:
    1) carrier: The wireless carrier involved in the outage. Must be one of: "AT&T", "Verizon", or "T-Mobile" (normalize common variants like ATT/AT T/AT&T → "AT&T"; Verizon Wireless → "Verizon"; TMobile/T Mobile/T-Mobile → "T-Mobile"). If the answer mentions multiple carriers, choose the primary one associated with the described outage.
    2) outage_date: The specific calendar date for when the outage occurred (e.g., "February 22, 2024" or "2024-02-22"). If multiple dates are mentioned, pick the primary occurrence date.
    3) services_disrupted: A list of the wireless service types disrupted. Normalize to canonical values among: "voice calls", "text messaging", "mobile data". Map synonyms:
       - voice calls: calling, phone calls, voice service, voice, call failures
       - text messaging: SMS, texts, messaging
       - mobile data: cellular data, LTE/5G data, internet on phone, data services
       Include only those the answer explicitly claims were disrupted.
    4) duration_description: A textual description of how long the outage lasted (e.g., "several hours", "around 8 hours", "at least 30 minutes").
    5) impact_description: A textual description of the impact scale (e.g., "millions of users", "over 100,000 customers", "hundreds of thousands"). Extract exactly as claimed in the answer.
    6) major_cities_affected: A list of specific major cities named as affected (e.g., "New York", "Los Angeles", "Chicago"). If none are mentioned, return an empty list.
    7) states_affected: A list of specific US states named as affected (e.g., "California", "Texas"). If none are mentioned, return an empty list.
    8) official_statement_url: A URL to an official statement or announcement from the carrier about the outage (carrier website newsroom/support page or the carrier's official social media post). If not provided, return null.
    9) news_coverage_url: A URL to a credible news media article about the outage. If not provided, return null.
    10) restoration_info: Text summarizing when service was restored or the restoration timeline if the answer provides it. Else null.
    11) remediation_actions: Text about any carrier response actions (e.g., bill credits, compensation). Else null.

    Constraints:
    - Extract ONLY from the provided answer. Do NOT infer or invent any values.
    - For URLs, extract explicit URLs present in the answer (plain or markdown formats).
    - Keep dates as strings exactly as presented; do NOT reformat.
    - If a field is missing, set it to null (for scalars) or [] for lists.
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def normalize_carrier(carrier: Optional[str]) -> Optional[str]:
    if not carrier:
        return None
    s = carrier.strip().lower()
    s = re.sub(r'[\s\.\-]+', '', s)
    # Normalize common variants
    if s in {"att", "atandt", "at&t", "attwireless", "atandtwireless"}:
        return "AT&T"
    if s in {"verizon", "verizonwireless"}:
        return "Verizon"
    if s in {"tmobile", "t-mobile", "tmobileus", "tmobileusa"}:
        return "T-Mobile"
    # Try broader matching
    if "att" in s or "atandt" in s:
        return "AT&T"
    if "verizon" in s:
        return "Verizon"
    if "tmobile" in s or "t-mobile" in s:
        return "T-Mobile"
    return carrier.strip()


def normalize_services(services: List[str]) -> List[str]:
    canonical = []
    for item in services or []:
        t = (item or "").strip().lower()
        if not t:
            continue
        # Voice synonyms
        if any(k in t for k in ["voice", "call", "phone call"]):
            val = "voice calls"
        # Text/SMS synonyms
        elif any(k in t for k in ["text", "sms", "messag"]):
            val = "text messaging"
        # Data synonyms
        elif any(k in t for k in ["data", "internet", "lte", "5g", "cellular data"]):
            val = "mobile data"
        else:
            # Keep original if we can't classify
            val = item.strip()
        if val not in canonical:
            canonical.append(val)
    # Keep only canonical names if recognized
    result = []
    for v in canonical:
        if v in {"voice calls", "text messaging", "mobile data"}:
            result.append(v)
        else:
            # unrecognized; keep it as-is for transparency
            result.append(v)
    return result


def join_list_natural(items: List[str]) -> str:
    items = [i for i in items if i]
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    return ", ".join(items[:-1]) + " and " + items[-1]


def _extract_date_candidate(s: str) -> Optional[str]:
    if not s:
        return None
    # Try to find common patterns
    patterns = [
        r"(\d{4}-\d{1,2}-\d{1,2})",                     # 2024-02-22
        r"(\d{4}/\d{1,2}/\d{1,2})",                     # 2024/02/22
        r"(\d{1,2}/\d{1,2}/\d{4})",                     # 02/22/2024
        r"((Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)\s+\d{1,2},\s*\d{4})",
        r"((January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},\s*\d{4})",
    ]
    for pat in patterns:
        m = re.search(pat, s, flags=re.IGNORECASE)
        if m:
            return m.group(1)
    # If ISO datetime present, take date part
    m2 = re.search(r"(\d{4}-\d{2}-\d{2})T", s)
    if m2:
        return m2.group(1)
    return None


def parse_date_fuzzy(s: Optional[str]) -> Optional[date]:
    if not s:
        return None
    cand = _extract_date_candidate(s) or s.strip()
    # Try known formats
    fmts = [
        "%Y-%m-%d",
        "%Y/%m/%d",
        "%m/%d/%Y",
        "%b %d, %Y",
        "%B %d, %Y",
    ]
    for fmt in fmts:
        try:
            return datetime.strptime(cand, fmt).date()
        except Exception:
            continue
    # Try first 10 chars if looks like ISO
    try:
        if len(cand) >= 10 and cand[4] == "-" and cand[7] == "-":
            return datetime.strptime(cand[:10], "%Y-%m-%d").date()
    except Exception:
        pass
    return None


def within_range(d: Optional[date], start: date, end: date) -> bool:
    if d is None:
        return False
    return start <= d <= end


def _normalize_sources_list(sources: Optional[List[Optional[str]] | str]) -> List[str]:
    if sources is None:
        return []
    if isinstance(sources, str):
        return [sources] if sources.strip() else []
    # list
    lst: List[str] = []
    for u in sources:
        if u and str(u).strip():
            lst.append(str(u).strip())
    # dedup preserve order
    seen = set()
    result = []
    for u in lst:
        if u not in seen:
            seen.add(u)
            result.append(u)
    return result


async def verify_guarded(
    evaluator: Evaluator,
    node,
    claim: str,
    sources: Optional[List[str] | str],
    additional_instruction: str,
    if_no_source_status: str = "failed",
) -> bool:
    src_list = _normalize_sources_list(sources)
    if len(src_list) == 0:
        node.score = 0.0
        node.status = "skipped" if if_no_source_status == "skipped" else "failed"
        return False
    return await evaluator.verify(
        claim=claim,
        node=node,
        sources=src_list,
        additional_instruction=additional_instruction,
    )


# --------------------------------------------------------------------------- #
# Verification builder                                                        #
# --------------------------------------------------------------------------- #
async def build_and_verify_tree(evaluator: Evaluator, data: OutageExtraction) -> None:
    root = evaluator.root  # Already initialized as PARALLEL

    # Normalize some fields
    norm_carrier = normalize_carrier(data.carrier)
    norm_services = normalize_services(data.services_disrupted or [])

    # 1) Carrier_Identity (Critical)
    carrier_ok = norm_carrier in {"AT&T", "Verizon", "T-Mobile"}
    evaluator.add_custom_node(
        result=carrier_ok,
        id="Carrier_Identity",
        desc="Identifies the carrier as one of the major US wireless carriers (Verizon, AT&T, or T-Mobile)",
        parent=root,
        critical=True,
    )

    # Aggregate sources convenience
    both_sources = [u for u in [data.official_statement_url, data.news_coverage_url] if u and u.strip()]
    both_sources = _normalize_sources_list(both_sources)

    # 2) Outage_Date (Critical) - verify with URLs that the outage occurred on the provided date
    node_od = evaluator.add_leaf(
        id="Outage_Date",
        desc="Provides the specific date when the outage occurred",
        parent=root,
        critical=True,
    )
    if data.outage_date and data.outage_date.strip():
        claim = f"The {norm_carrier or (data.carrier or 'named')} outage occurred on or about {data.outage_date} (allowing for local time zones or late-night spills)."
        await verify_guarded(
            evaluator,
            node_od,
            claim=claim,
            sources=both_sources,
            additional_instruction="Verify that the provided date is explicitly indicated as the occurrence date of the outage (not merely the article publish date). Allow minor timezone date shifts (e.g., late night leading to next-day reports).",
            if_no_source_status="failed",
        )
    else:
        node_od.score = 0.0
        node_od.status = "failed"

    # 3) Timeframe_Constraint (Critical) - purely logical date range check
    parsed_date = parse_date_fuzzy(data.outage_date or "")
    in_range = within_range(parsed_date, DATE_RANGE_START, DATE_RANGE_END)
    evaluator.add_custom_node(
        result=in_range,
        id="Timeframe_Constraint",
        desc="Confirms the outage occurred between January 1, 2024 and February 21, 2026",
        parent=root,
        critical=True,
    )

    # 4) Duration_Threshold (Critical) - at least 30 minutes
    node_dt = evaluator.add_leaf(
        id="Duration_Threshold",
        desc="Documents that the outage lasted at least 30 minutes, meeting the FCC NORS reporting threshold for duration",
        parent=root,
        critical=True,
    )
    claim_dt = f"For the {norm_carrier or (data.carrier or 'named')} outage on {data.outage_date or 'the reported date'}, the duration was at least 30 minutes (e.g., described as 30+ minutes, hours-long, multiple hours, etc.)."
    await verify_guarded(
        evaluator,
        node_dt,
        claim=claim_dt,
        sources=both_sources,
        additional_instruction="Treat the claim as supported if the page explicitly states 30 minutes or more, or terms like 'hours-long', 'several hours', 'multi-hour', or similar phrasing indicating ≥30 minutes.",
        if_no_source_status="failed",
    )

    # 5) User_Impact_Scale (Critical) - >=100,000 affected
    node_ui = evaluator.add_leaf(
        id="User_Impact_Scale",
        desc="Documents that the outage affected at least 100,000 users at its peak or provides evidence of impact scale meeting FCC reporting thresholds",
        parent=root,
        critical=True,
    )
    claim_ui = f"The {norm_carrier or (data.carrier or 'named')} outage on {data.outage_date or 'the reported date'} affected at least 100,000 users/customers at its peak."
    await verify_guarded(
        evaluator,
        node_ui,
        claim=claim_ui,
        sources=both_sources,
        additional_instruction="Consider the claim supported if the page says 'hundreds of thousands', 'over 100,000', 'hundreds of thousands or more', 'millions', or any phrase clearly ≥100,000 affected.",
        if_no_source_status="failed",
    )

    # 6) Multiple_States_Affected (Critical)
    node_ms = evaluator.add_leaf(
        id="Multiple_States_Affected",
        desc="Confirms that the outage affected multiple US states (not limited to a single state), regardless of whether specific state names are enumerated",
        parent=root,
        critical=True,
    )
    claim_ms = f"The {norm_carrier or (data.carrier or 'named')} outage on {data.outage_date or 'the reported date'} affected multiple US states (e.g., multi-state or nationwide impact)."
    await verify_guarded(
        evaluator,
        node_ms,
        claim=claim_ms,
        sources=both_sources,
        additional_instruction="Evidence may include phrases like 'nationwide', 'across the U.S.', 'multiple states', or explicitly listing more than one state. If only a single state is mentioned with no broader impact, do not support.",
        if_no_source_status="failed",
    )

    # 7) Service_Types_Disrupted (Critical)
    node_st = evaluator.add_leaf(
        id="Service_Types_Disrupted",
        desc="Specifies which wireless services were disrupted (e.g., voice calls, text messaging, mobile data)",
        parent=root,
        critical=True,
    )
    if norm_services:
        services_text = join_list_natural(norm_services)
        claim_st = f"During the {norm_carrier or (data.carrier or 'named')} outage on {data.outage_date or 'the reported date'}, the following services were disrupted: {services_text}."
        await verify_guarded(
            evaluator,
            node_st,
            claim=claim_st,
            sources=both_sources,
            additional_instruction="Treat as supported if each listed service type is clearly reported as impacted (allowing synonyms, e.g., voice/calling, SMS/text, cellular data/mobile internet). It's fine if the sources mention additional impacted services beyond those listed.",
            if_no_source_status="failed",
        )
    else:
        # Missing explicit enumeration in the answer is a critical failure per task requirement
        node_st.score = 0.0
        node_st.status = "failed"

    # 8) Major_Cities_Identified (Non-Critical)
    node_mc = evaluator.add_leaf(
        id="Major_Cities_Identified",
        desc="Lists specific major cities that were affected by the outage",
        parent=root,
        critical=False,
    )
    if data.major_cities_affected:
        cities_text = join_list_natural(data.major_cities_affected)
        claim_mc = f"Major affected cities included: {cities_text}."
        await verify_guarded(
            evaluator,
            node_mc,
            claim=claim_mc,
            sources=both_sources,
            additional_instruction="Support the claim if at least one of the listed cities is explicitly mentioned as impacted by the outage. The list need not be exhaustive.",
            if_no_source_status="skipped",
        )
    else:
        node_mc.score = 0.0
        node_mc.status = "skipped"

    # 9) Geographic_States_Documentation (Non-Critical)
    node_gs = evaluator.add_leaf(
        id="Geographic_States_Documentation",
        desc="Provides an enumeration or list of specific US states that were affected by the outage",
        parent=root,
        critical=False,
    )
    if data.states_affected:
        states_text = join_list_natural(data.states_affected)
        claim_gs = f"Affected states included: {states_text}."
        await verify_guarded(
            evaluator,
            node_gs,
            claim=claim_gs,
            sources=both_sources,
            additional_instruction="Support the claim if at least one of the listed states is explicitly mentioned as impacted by the outage. The list need not be exhaustive.",
            if_no_source_status="skipped",
        )
    else:
        node_gs.score = 0.0
        node_gs.status = "skipped"

    # 10) Official_Carrier_Statement (Critical)
    node_off = evaluator.add_leaf(
        id="Official_Carrier_Statement",
        desc="Provides a reference URL to an official statement or announcement from the carrier about the outage",
        parent=root,
        critical=True,
    )
    if data.official_statement_url and data.official_statement_url.strip():
        claim_off = f"This page is an official statement or announcement from {norm_carrier or (data.carrier or 'the carrier')} about the outage on {data.outage_date or 'the reported date'}."
        await verify_guarded(
            evaluator,
            node_off,
            claim=claim_off,
            sources=data.official_statement_url,
            additional_instruction="Treat as 'official' if the page is on the carrier's official website (e.g., att.com / about.att.com, verizon.com, t-mobile.com) OR is a post from the carrier's verified/official social account (e.g., X/Twitter). The content must explicitly discuss the outage.",
            if_no_source_status="failed",
        )
    else:
        node_off.score = 0.0
        node_off.status = "failed"

    # 11) News_Media_Coverage (Critical)
    node_news = evaluator.add_leaf(
        id="News_Media_Coverage",
        desc="Provides a reference URL to credible news coverage of the outage",
        parent=root,
        critical=True,
    )
    if data.news_coverage_url and data.news_coverage_url.strip():
        claim_news = f"This page is credible news media coverage of the {norm_carrier or (data.carrier or 'named')} outage on {data.outage_date or 'the reported date'}."
        await verify_guarded(
            evaluator,
            node_news,
            claim=claim_news,
            sources=data.news_coverage_url,
            additional_instruction="Treat as credible news media if it is a recognized media outlet (national, regional, or reputable local). The article must explicitly discuss the outage.",
            if_no_source_status="failed",
        )
    else:
        node_news.score = 0.0
        node_news.status = "failed"

    # 12) Restoration_Timeline (Non-Critical)
    node_rest = evaluator.add_leaf(
        id="Restoration_Timeline",
        desc="Documents when service was restored or the timeline for restoration",
        parent=root,
        critical=False,
    )
    if data.restoration_info and data.restoration_info.strip():
        claim_rest = f"The sources report the following restoration timing or timeline for the outage: {data.restoration_info.strip()}."
        await verify_guarded(
            evaluator,
            node_rest,
            claim=claim_rest,
            sources=both_sources,
            additional_instruction="Support the claim if the sources describe when service was restored or provide a clear timeline of restoration stages (partial or full). Allow paraphrasing.",
            if_no_source_status="skipped",
        )
    else:
        node_rest.score = 0.0
        node_rest.status = "skipped"

    # 13) Company_Response_Actions (Non-Critical)
    node_resp = evaluator.add_leaf(
        id="Company_Response_Actions",
        desc="Documents the carrier's response actions such as account credits, compensation, or other customer remediation",
        parent=root,
        critical=False,
    )
    if data.remediation_actions and data.remediation_actions.strip():
        claim_resp = f"The sources state the carrier took remediation actions such as compensation/credits: {data.remediation_actions.strip()}."
        await verify_guarded(
            evaluator,
            node_resp,
            claim=claim_resp,
            sources=both_sources,
            additional_instruction="Support the claim if the sources explicitly say the carrier offered bill credits, compensation, refunds, free service, or similar actions for affected customers.",
            if_no_source_status="skipped",
        )
    else:
        node_resp.score = 0.0
        node_resp.status = "skipped"

    # Optionally record some normalized info for debugging
    evaluator.add_custom_info(
        {
            "normalized_carrier": norm_carrier,
            "normalized_services": norm_services,
            "parsed_outage_date": parsed_date.isoformat() if parsed_date else None,
            "date_in_range": in_range,
            "sources_checked": both_sources,
        },
        info_type="debug_info",
        info_name="normalization_and_range_checks"
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
    Evaluate an answer for the major US telecommunications outage documentation task.
    """
    evaluator = Evaluator()
    evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # As specified by the rubric
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

    # Extract structured outage info from the answer
    extracted: OutageExtraction = await evaluator.extract(
        prompt=prompt_extract_outage(),
        template_class=OutageExtraction,
        extraction_name="outage_extraction",
    )

    # Build verification tree and run checks
    await build_and_verify_tree(evaluator, extracted)

    # Return summary
    return evaluator.get_summary()