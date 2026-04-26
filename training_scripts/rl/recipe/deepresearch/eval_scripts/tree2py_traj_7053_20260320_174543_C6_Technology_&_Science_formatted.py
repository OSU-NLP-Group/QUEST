import asyncio
import logging
import re
from datetime import datetime, date
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "us_outages_q1_2026"
TASK_DESCRIPTION = """
Identify four major network outages from four different U.S. telecommunications or internet service providers that occurred between January 1, 2026 and March 15, 2026. Each outage must have lasted at least 30 minutes, meeting the FCC's Network Outage Reporting System (NORS) reporting threshold. For each outage, provide comprehensive documentation including: the provider name, the specific date when the outage occurred, the total duration of the outage, the geographic regions or areas affected, the number of customers impacted or scope of impact, the stated or identified technical cause, the provider's official response or statement, and whether customer compensation (such as credits or refunds) was offered and if so, the amount. Support all information with reference URLs from credible sources such as news articles, official company statements, industry reports, or recognized outage tracking platforms.
"""

DATE_RANGE_START = date(2026, 1, 1)
DATE_RANGE_END = date(2026, 3, 15)
NORS_MINUTES = 30


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class OutageItem(BaseModel):
    provider: Optional[str] = None
    date: Optional[str] = None  # Keep as free-form string extracted from answer
    duration: Optional[str] = None  # Free-form (e.g., "45 minutes", "1 hour 20 minutes")
    regions: List[str] = Field(default_factory=list)
    impact: Optional[str] = None  # Free-form (e.g., "over 100,000 customers", "nationwide")
    technical_cause: Optional[str] = None
    official_response: Optional[str] = None
    compensation: Optional[str] = None  # e.g., "Yes: $5 credit", "No", "TBD"
    compensation_amount: Optional[str] = None  # e.g., "$5", "$10 credit", etc.

    # Evidence URLs per category
    temporal_urls: List[str] = Field(default_factory=list)   # Supports date and/or duration
    geographic_urls: List[str] = Field(default_factory=list)
    impact_urls: List[str] = Field(default_factory=list)
    technical_urls: List[str] = Field(default_factory=list)
    response_urls: List[str] = Field(default_factory=list)


class OutagesExtraction(BaseModel):
    outages: List[OutageItem] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_outages() -> str:
    return """
    Extract up to four (4) specific U.S. telecom or internet service provider network outages from the answer text.
    Each outage must have occurred between January 1, 2026 and March 15, 2026 (inclusive), and the duration must be at least 30 minutes.
    If the answer contains more than four qualifying outages, extract only the first four that meet the constraints. If fewer, extract as many as are provided.

    For each outage, extract these fields exactly as they appear in the answer (do not fabricate or infer):
    - provider: The provider name (e.g., AT&T, Verizon, Comcast Xfinity, T-Mobile, Spectrum, Frontier, Cox, etc.).
    - date: The specific date of the outage as written in the answer (keep original format).
    - duration: The outage duration as written in the answer (e.g., "45 minutes", "1 hour 15 minutes", "about 2 hours").
    - regions: A list of geographic regions/areas affected (states, cities, or broader scope); split multiple regions into separate list entries.
    - impact: The number of customers affected or a textual description of the scope (e.g., "hundreds of thousands", "nationwide", "millions").
    - technical_cause: The stated or identified technical cause, if any (e.g., "fiber cut", "configuration error", "software update").
    - official_response: The provider's official response or statement summary as presented in the answer (paraphrase or exact quote used in the answer).
    - compensation: Whether customer compensation was offered (e.g., "Yes: $5 credit", "No", "Unknown"). If not mentioned, set to null.
    - compensation_amount: The compensation amount, if any (e.g., "$5 credit"); otherwise null.

    Evidence URLs:
    The answer should cite URLs. Assign them to categories based on what they support:
    - temporal_urls: URLs supporting the date and/or duration.
    - geographic_urls: URLs supporting the geographic scope/regions.
    - impact_urls: URLs supporting the number of customers affected or scope of impact.
    - technical_urls: URLs supporting the technical cause.
    - response_urls: URLs supporting the provider's official response and/or compensation details.
    If a single URL supports multiple categories, include it in each relevant category. Include only URLs that are explicitly present in the answer text.
    Always include full URLs with protocol.

    Return JSON with a single field:
    {
      "outages": [ OutageItem, OutageItem, OutageItem, OutageItem (optional) ]
    }
    For any missing field, return null (or an empty list for lists).
    """


# --------------------------------------------------------------------------- #
# Helper parsing utilities                                                    #
# --------------------------------------------------------------------------- #
MONTHS = [
    "january", "february", "march", "april", "may", "june",
    "july", "august", "september", "october", "november", "december"
]
MONTHS_ABBR = ["jan", "feb", "mar", "apr", "may", "jun",
               "jul", "aug", "sep", "sept", "oct", "nov", "dec"]


def _try_parse_date(date_str: Optional[str]) -> Optional[date]:
    if not date_str:
        return None
    s = date_str.strip()
    # Try common formats first
    fmts = [
        "%Y-%m-%d",
        "%Y/%m/%d",
        "%Y.%m.%d",
        "%m/%d/%Y",
        "%m/%d/%y",
        "%B %d, %Y",
        "%b %d, %Y",
        "%d %B %Y",
        "%d %b %Y",
        "%B %d %Y",
        "%b %d %Y",
    ]
    for f in fmts:
        try:
            return datetime.strptime(s, f).date()
        except Exception:
            pass

    # Try to locate a clean date substring (e.g., handles extra words)
    # 1) ISO-like yyyy-mm-dd
    m = re.search(r"\b(2026)-(0[1-9]|1[0-2])-(0[1-9]|[12]\d|3[01])\b", s)
    if m:
        try:
            return datetime.strptime(m.group(0), "%Y-%m-%d").date()
        except Exception:
            pass

    # 2) mm/dd/yyyy
    m = re.search(r"\b(0?[1-9]|1[0-2])/(0?[1-9]|[12]\d|3[01])/(\d{4})\b", s)
    if m:
        mm, dd, yyyy = m.groups()
        try:
            return date(int(yyyy), int(mm), int(dd))
        except Exception:
            pass

    # 3) "Month dd, 2026"
    month_names = "|".join(MONTHS)
    month_abbr = "|".join(MONTHS_ABBR)
    m = re.search(rf"\b({month_names}|{month_abbr})\s+(\d{{1,2}}),?\s*(2026)\b", s, re.IGNORECASE)
    if m:
        mon, dd, yyyy = m.groups()
        try:
            # Normalize month
            try:
                mon_num = MONTHS.index(mon.lower()) + 1
            except ValueError:
                # abbr mapping
                ab = mon.lower()
                # map "sept" to "sep"
                if ab == "sept":
                    ab = "sep"
                mon_num = MONTHS_ABBR.index(ab) + 1
            return date(int(yyyy), mon_num, int(dd))
        except Exception:
            pass

    return None


def _parse_duration_minutes(duration_str: Optional[str]) -> Optional[int]:
    if not duration_str:
        return None
    s = duration_str.lower()

    # Special phrase: half an hour
    if "half an hour" in s or "half-hour" in s or "half hour" in s:
        return 30

    total_minutes = 0
    # Match hours (e.g., "1.5 hours", "2 hours", "1 hr", "hr", "hrs")
    for m in re.finditer(r"(\d+(?:\.\d+)?)\s*(hour|hours|hr|hrs)\b", s):
        try:
            hours = float(m.group(1))
            total_minutes += int(round(hours * 60))
        except Exception:
            pass

    # Match minutes (e.g., "45 minutes", "30 mins", "min")
    for m in re.finditer(r"(\d+)\s*(minute|minutes|min|mins)\b", s):
        try:
            mins = int(m.group(1))
            total_minutes += mins
        except Exception:
            pass

    # Fallback: a bare number followed by '+' and 'minutes' context (e.g., "30+ minutes")
    if total_minutes == 0:
        m = re.search(r"(\d+)\s*\+\s*(minutes|min|mins)\b", s)
        if m:
            try:
                total_minutes += int(m.group(1))
            except Exception:
                pass

    # Another fallback: a single bare integer with context "for X minutes"
    if total_minutes == 0:
        m = re.search(r"for\s+(\d+)\s*(minutes|min|mins)\b", s)
        if m:
            try:
                total_minutes += int(m.group(1))
            except Exception:
                pass

    return total_minutes if total_minutes > 0 else None


def _within_required_timeframe(d: Optional[date]) -> bool:
    if d is None:
        return False
    return DATE_RANGE_START <= d <= DATE_RANGE_END


def _nonempty(s: Optional[str]) -> bool:
    return bool(s and str(s).strip())


def _normalize_provider_name(name: str) -> str:
    # Lowercase, remove punctuation and common corporate suffixes to reduce trivial variants.
    s = name.lower()
    s = re.sub(r"[^a-z0-9\s&]+", "", s)  # keep letters/numbers/&/spaces
    tokens = [t for t in s.split() if t not in {
        "inc", "incorporated", "corp", "corporation", "llc", "l.l.c", "co", "company",
        "communications", "communication", "comms", "services", "service", "limited", "ltd", "plc"
    }]
    return " ".join(tokens).strip()


# --------------------------------------------------------------------------- #
# Verification for a single outage                                            #
# --------------------------------------------------------------------------- #
async def verify_single_outage(
    evaluator: Evaluator,
    parent_node,
    outage: OutageItem,
    outage_index: int
) -> None:
    """
    Build verification sub-tree for one outage.
    """
    # Create the outage container node (parallel, non-critical, as rubric specifies "analysis" nodes are non-critical)
    outage_node = evaluator.add_parallel(
        id=f"outage_{outage_index}",
        desc=f"{['First','Second','Third','Fourth'][outage_index-1]} provider outage analysis",
        parent=parent_node,
        critical=False
    )

    # 1) Provider identification (critical existence)
    provider_exists = evaluator.add_custom_node(
        result=_nonempty(outage.provider),
        id=f"outage_{outage_index}_provider_identification",
        desc="Provider name is identified",
        parent=outage_node,
        critical=True
    )

    # 2) Temporal information (critical group)
    temporal_group = evaluator.add_parallel(
        id=f"outage_{outage_index}_temporal_information",
        desc="Complete temporal information about the outage",
        parent=outage_node,
        critical=True
    )
    # 2.1 date specified
    evaluator.add_custom_node(
        result=_nonempty(outage.date),
        id=f"outage_{outage_index}_date_specified",
        desc="Specific date of outage is provided",
        parent=temporal_group,
        critical=True
    )
    # 2.2 duration specified
    evaluator.add_custom_node(
        result=_nonempty(outage.duration),
        id=f"outage_{outage_index}_duration_specified",
        desc="Total duration of outage is provided in minutes or hours",
        parent=temporal_group,
        critical=True
    )
    # 2.3 duration threshold met (>= 30 minutes)
    duration_minutes = _parse_duration_minutes(outage.duration)
    evaluator.add_custom_node(
        result=(duration_minutes is not None and duration_minutes >= NORS_MINUTES),
        id=f"outage_{outage_index}_duration_threshold_met",
        desc="Outage duration is at least 30 minutes, meeting FCC NORS reporting threshold",
        parent=temporal_group,
        critical=True
    )
    # 2.4 timeframe compliance
    parsed_date = _try_parse_date(outage.date)
    evaluator.add_custom_node(
        result=_within_required_timeframe(parsed_date),
        id=f"outage_{outage_index}_timeframe_compliance",
        desc="Outage occurred between January 1, 2026 and March 15, 2026",
        parent=temporal_group,
        critical=True
    )
    # 2.5 temporal reference exists (additional gating existence check for URLs)
    evaluator.add_custom_node(
        result=bool(outage.temporal_urls),
        id=f"outage_{outage_index}_temporal_reference_urls_provided",
        desc="At least one temporal reference URL is provided",
        parent=temporal_group,
        critical=True
    )
    # 2.6 temporal reference verification
    temporal_ref_leaf = evaluator.add_leaf(
        id=f"outage_{outage_index}_temporal_reference",
        desc="Reference URL provided supporting the temporal information",
        parent=temporal_group,
        critical=True
    )
    temporal_claim_parts = []
    if _nonempty(outage.date):
        temporal_claim_parts.append(f"the outage occurred on {outage.date}")
    if _nonempty(outage.duration):
        temporal_claim_parts.append(f"and lasted about {outage.duration}")
    temporal_claim_text = f"For provider '{outage.provider}', " + " ".join(temporal_claim_parts) + "."
    await evaluator.verify(
        claim=temporal_claim_text.strip(),
        node=temporal_ref_leaf,
        sources=outage.temporal_urls,
        additional_instruction="Verify that at least one provided URL explicitly mentions the outage date and (if stated) the duration. Allow minor rounding or wording differences."
    )

    # 3) Geographic information (critical group)
    geo_group = evaluator.add_parallel(
        id=f"outage_{outage_index}_geographic_information",
        desc="Geographic scope of the outage",
        parent=outage_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=bool(outage.regions),
        id=f"outage_{outage_index}_regions_identified",
        desc="Geographic regions, states, cities, or scope are identified",
        parent=geo_group,
        critical=True
    )
    evaluator.add_custom_node(
        result=bool(outage.geographic_urls),
        id=f"outage_{outage_index}_geographic_reference_urls_provided",
        desc="At least one geographic reference URL is provided",
        parent=geo_group,
        critical=True
    )
    geo_leaf = evaluator.add_leaf(
        id=f"outage_{outage_index}_geographic_reference",
        desc="Reference URL provided supporting the geographic scope information",
        parent=geo_group,
        critical=True
    )
    regions_str = ", ".join(outage.regions[:6]) if outage.regions else "the stated regions"
    await evaluator.verify(
        claim=f"The outage affected the following regions/areas: {regions_str}.",
        node=geo_leaf,
        sources=outage.geographic_urls,
        additional_instruction="Verify that at least one URL substantiates the geographic scope (specific cities, states, or broader regions). Fuzzy-match names and allow synonyms."
    )

    # 4) Impact information (critical group)
    impact_group = evaluator.add_parallel(
        id=f"outage_{outage_index}_impact_information",
        desc="Customer impact scope of the outage",
        parent=outage_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=_nonempty(outage.impact),
        id=f"outage_{outage_index}_impact_scope_documented",
        desc="Number of customers affected or description of impact scope is provided",
        parent=impact_group,
        critical=True
    )
    evaluator.add_custom_node(
        result=bool(outage.impact_urls),
        id=f"outage_{outage_index}_impact_reference_urls_provided",
        desc="At least one impact reference URL is provided",
        parent=impact_group,
        critical=True
    )
    impact_leaf = evaluator.add_leaf(
        id=f"outage_{outage_index}_impact_reference",
        desc="Reference URL provided supporting the customer impact information",
        parent=impact_group,
        critical=True
    )
    await evaluator.verify(
        claim=f"The outage's customer impact is described as: {outage.impact or 'N/A'}.",
        node=impact_leaf,
        sources=outage.impact_urls,
        additional_instruction="Verify that at least one URL states the number of affected customers or clearly describes the impact scope (e.g., 'nationwide', 'hundreds of thousands')."
    )

    # 5) Technical information (critical group)
    technical_group = evaluator.add_parallel(
        id=f"outage_{outage_index}_technical_information",
        desc="Technical cause and details of the outage",
        parent=outage_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=_nonempty(outage.technical_cause),
        id=f"outage_{outage_index}_cause_identified",
        desc="Technical cause of the outage is stated or identified",
        parent=technical_group,
        critical=True
    )
    evaluator.add_custom_node(
        result=bool(outage.technical_urls),
        id=f"outage_{outage_index}_technical_reference_urls_provided",
        desc="At least one technical-cause reference URL is provided",
        parent=technical_group,
        critical=True
    )
    technical_leaf = evaluator.add_leaf(
        id=f"outage_{outage_index}_technical_reference",
        desc="Reference URL provided supporting the technical cause information",
        parent=technical_group,
        critical=True
    )
    await evaluator.verify(
        claim=f"The technical cause of the outage is: {outage.technical_cause or 'N/A'}.",
        node=technical_leaf,
        sources=outage.technical_urls,
        additional_instruction="Verify that at least one URL explicitly mentions the technical cause or a highly plausible root cause. Accept reasonable paraphrases."
    )

    # 6) Response information
    # Note: To allow a non-critical 'compensation_documented' child, we keep the parent non-critical (framework constraint).
    response_group = evaluator.add_parallel(
        id=f"outage_{outage_index}_response_information",
        desc="Provider response and compensation details",
        parent=outage_node,
        critical=False
    )
    evaluator.add_custom_node(
        result=_nonempty(outage.official_response),
        id=f"outage_{outage_index}_official_response_documented",
        desc="Provider's official response or statement about the outage is documented",
        parent=response_group,
        critical=True
    )
    # Non-critical: compensation documented (if present)
    evaluator.add_custom_node(
        result=_nonempty(outage.compensation) or _nonempty(outage.compensation_amount),
        id=f"outage_{outage_index}_compensation_documented",
        desc="Information about whether customer compensation was offered is documented, including amount if disclosed",
        parent=response_group,
        critical=False
    )
    evaluator.add_custom_node(
        result=bool(outage.response_urls),
        id=f"outage_{outage_index}_response_reference_urls_provided",
        desc="At least one response/compensation reference URL is provided",
        parent=response_group,
        critical=True
    )
    response_leaf = evaluator.add_leaf(
        id=f"outage_{outage_index}_response_reference",
        desc="Reference URL provided supporting the provider response information",
        parent=response_group,
        critical=True
    )
    comp_snippet = ""
    if _nonempty(outage.compensation_amount):
        comp_snippet = f" Compensation amount mentioned: {outage.compensation_amount}."
    elif _nonempty(outage.compensation):
        comp_snippet = f" Compensation info: {outage.compensation}."
    await evaluator.verify(
        claim=f"The provider's official response/statement is: {outage.official_response or 'N/A'}.{comp_snippet}".strip(),
        node=response_leaf,
        sources=outage.response_urls,
        additional_instruction="Verify that at least one URL contains the provider's official statement/response to the outage. If compensation is claimed, check if the same or other provided URLs mention it."
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
    Evaluate an answer for the Q1 2026 U.S. provider outage task.
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

    # Extract structured outages
    extracted = await evaluator.extract(
        prompt=prompt_extract_outages(),
        template_class=OutagesExtraction,
        extraction_name="outages_extraction"
    )

    outages: List[OutageItem] = list(extracted.outages)[:4]
    while len(outages) < 4:
        outages.append(OutageItem())

    # Ground truth info (constraints)
    evaluator.add_ground_truth({
        "timeframe": {"start": str(DATE_RANGE_START), "end": str(DATE_RANGE_END)},
        "nors_minimum_minutes": NORS_MINUTES,
        "required_outages": 4
    }, gt_type="constraints")

    # Provider uniqueness (critical)
    provider_names = [o.provider for o in outages if _nonempty(o.provider)]
    normalized = [_normalize_provider_name(p) for p in provider_names]
    uniqueness_ok = (len(provider_names) == 4 and len(set(normalized)) == 4)
    evaluator.add_custom_node(
        result=uniqueness_ok,
        id="provider_uniqueness",
        desc="All four providers are distinct and different from each other",
        parent=root,
        critical=True
    )

    # Build per-outage verification subtrees
    for idx, outage in enumerate(outages, start=1):
        await verify_single_outage(evaluator, root, outage, idx)

    return evaluator.get_summary()