import asyncio
import logging
import re
from typing import Optional, List, Dict, Any, Tuple
from urllib.parse import urlparse

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "nace_knowledge_rate_university_excellence"
TASK_DESCRIPTION = """
According to the National Association of Colleges and Employers (NACE) First Destination Survey Standards and Protocols, what is the recommended minimum knowledge rate that institutions should strive to achieve when collecting and reporting post-graduation career outcomes? Additionally, identify a major U.S. research university whose career services office has consistently achieved a knowledge rate that exceeds the NACE minimum by at least 20 percentage points in recent years. For this university, provide: (1) the university's name, (2) the specific knowledge rate range achieved in recent years, and (3) verification that this performance level is characterized as being among the best in the nation for comparable institutions. Include supporting URL references for all claims.
"""


# --------------------------------------------------------------------------- #
# Data models for structured extraction                                       #
# --------------------------------------------------------------------------- #
class NACEInfo(BaseModel):
    min_knowledge_rate_value: Optional[str] = None
    nace_citation_urls: List[str] = Field(default_factory=list)


class UniversityInfo(BaseModel):
    name: Optional[str] = None
    research_status_text: Optional[str] = None
    research_status_urls: List[str] = Field(default_factory=list)
    knowledge_rate_range: Optional[str] = None
    knowledge_rate_urls: List[str] = Field(default_factory=list)
    recent_years_indicator: Optional[bool] = None


class BestInNationInfo(BaseModel):
    claim_text: Optional[str] = None
    citation_urls: List[str] = Field(default_factory=list)


class ExtractionOutput(BaseModel):
    nace: Optional[NACEInfo] = None
    university: Optional[UniversityInfo] = None
    best_in_nation: Optional[BestInNationInfo] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_all() -> str:
    return """
    Extract the following information from the provided answer. If multiple universities are mentioned, extract details for the first university that clearly meets the criteria.

    Return a JSON object with fields: nace, university, best_in_nation.

    1) nace:
       - min_knowledge_rate_value (string): The NACE-recommended minimum "knowledge rate" value (e.g., "65%" or "65 percent"). Use exactly the phrasing/number given in the answer.
       - nace_citation_urls (array of strings): URL(s) to official NACE documentation supporting the minimum knowledge rate (prefer naceweb.org or nacecenter.org). Include all such URLs explicitly present in the answer.

    2) university:
       - name (string): The university's name.
       - research_status_text (string): The statement/evidence indicating it is a major U.S. research university (e.g., "Carnegie R1," "AAU member," or similar). Use a short phrase from the answer when possible.
       - research_status_urls (array of strings): URL(s) supporting the research university status (e.g., Carnegie Classification page, AAU page, or credible equivalent). Extract URLs exactly as shown in the answer.
       - knowledge_rate_range (string): The specific knowledge rate range achieved in recent years. Prefer a concise form like "92%-97%" or "95%+"; it can also be a sentence if no compact range is given. If multiple year-specific percentages are given, summarize as "min%-max%".
       - knowledge_rate_urls (array of strings): Official university URL(s) that support the knowledge rate figures/range and time span (prefer .edu domains). Extract all such URLs from the answer.
       - recent_years_indicator (boolean): True if the answer indicates the knowledge rate range corresponds to multiple recent years (not just one year); otherwise False or null.

    3) best_in_nation:
       - claim_text (string): The statement/phrase indicating the performance is "among the best in the nation for comparable institutions" (or closely equivalent phrasing). Use the wording from the answer if possible.
       - citation_urls (array of strings): URL(s) that support this characterization. Extract all such URLs.

    Rules:
    - Do not fabricate any data; only extract what is explicitly present in the answer.
    - For all URL fields, extract actual URLs (plain or markdown links).
    - If a requested item is missing, set it to null (or [] for arrays).
    """


# --------------------------------------------------------------------------- #
# Helper utilities for numeric parsing                                        #
# --------------------------------------------------------------------------- #
def _to_float_safe(s: str) -> Optional[float]:
    try:
        return float(s)
    except Exception:
        return None


def parse_percentage_value(text: Optional[str]) -> Optional[float]:
    """
    Parse a single percentage value from text into a float in [0, 100].
    Accepts forms like '65%', '65 percent', '65 pct', '0.65' (interpreted as 65 if <= 1).
    Returns None if not parsable.
    """
    if not text:
        return None
    t = text.lower().strip()

    # Common direct patterns with percent words/symbol
    m = re.search(r'(\d{1,3}(?:\.\d+)?)\s*(%|percent|pct|per\s*cent)\b', t, flags=re.I)
    if m:
        val = _to_float_safe(m.group(1))
        if val is None:
            return None
        return max(0.0, min(100.0, val))

    # Plain number, potentially a decimal like 0.65 meaning 65%
    m2 = re.search(r'\b(\d{1,3}(?:\.\d+)?)\b', t)
    if m2:
        val = _to_float_safe(m2.group(1))
        if val is None:
            return None
        if val <= 1.0:
            return val * 100.0
        return max(0.0, min(100.0, val))

    return None


def parse_percentage_range(text: Optional[str]) -> Tuple[Optional[float], Optional[float]]:
    """
    Parse a percentage range from text, returning (min_percent, max_percent).
    Handles patterns:
      - 'X% - Y%', 'X%-Y%', 'X%–Y%', 'X% to Y%', 'between X% and Y%'
      - '>=X%', 'X%+', 'at least X%', 'X% or higher' -> (X, None)
      - Single value 'X%' -> (X, X)
    Returns (None, None) if not parsable.
    """
    if not text:
        return (None, None)
    t = text.lower().strip()

    # between X% and Y%
    m_between = re.search(
        r'between\s+(\d{1,3}(?:\.\d+)?)\s*%?\s+and\s+(\d{1,3}(?:\.\d+)?)\s*%?',
        t, flags=re.I
    )
    if m_between:
        a = _to_float_safe(m_between.group(1))
        b = _to_float_safe(m_between.group(2))
        if a is not None and b is not None:
            lo, hi = sorted([a, b])
            return (max(0.0, min(100.0, lo)), max(0.0, min(100.0, hi)))

    # X% - Y% (including en/em dashes) or 'to'
    m_dash = re.search(
        r'(\d{1,3}(?:\.\d+)?)\s*%?\s*(?:-|–|—|to)\s*(\d{1,3}(?:\.\d+)?)\s*%?',
        t, flags=re.I
    )
    if m_dash:
        a = _to_float_safe(m_dash.group(1))
        b = _to_float_safe(m_dash.group(2))
        if a is not None and b is not None:
            lo, hi = sorted([a, b])
            return (max(0.0, min(100.0, lo)), max(0.0, min(100.0, hi)))

    # >= X% / at least X% / X%+ / X% or higher
    m_floor = re.search(
        r'(?:>=|at\s+least|minimum\s+of|no\s+less\s+than)\s*(\d{1,3}(?:\.\d+)?)\s*%|\b(\d{1,3}(?:\.\d+)?)\s*%\s*(?:\+|or\s+higher|or\s+more)\b',
        t, flags=re.I
    )
    if m_floor:
        num = m_floor.group(1) or m_floor.group(2)
        a = _to_float_safe(num)
        if a is not None:
            return (max(0.0, min(100.0, a)), None)

    # Single value X%
    m_single = re.search(r'(\d{1,3}(?:\.\d+)?)\s*%(\s|$)', t, flags=re.I)
    if m_single:
        a = _to_float_safe(m_single.group(1))
        if a is not None:
            a = max(0.0, min(100.0, a))
            return (a, a)

    # Fallback: single number without % (assume percent)
    m_single2 = re.search(r'\b(\d{1,3}(?:\.\d+)?)\b', t)
    if m_single2:
        a = _to_float_safe(m_single2.group(1))
        if a is not None:
            if a <= 1.0:
                a = a * 100.0
            a = max(0.0, min(100.0, a))
            return (a, a)

    return (None, None)


def is_official_nace_url(url: str) -> bool:
    try:
        netloc = urlparse(url).netloc.lower()
        return ("naceweb.org" in netloc) or ("nacecenter.org" in netloc)
    except Exception:
        return False


def is_edu_url(url: str) -> bool:
    try:
        netloc = urlparse(url).netloc.lower()
        return netloc.endswith(".edu") or ".edu" in netloc
    except Exception:
        return False


# --------------------------------------------------------------------------- #
# Verification subroutines                                                    #
# --------------------------------------------------------------------------- #
async def verify_nace_minimum_section(
    evaluator: Evaluator,
    parent_node,
    extracted: ExtractionOutput,
) -> None:
    nace = extracted.nace or NACEInfo()

    # Parallel node for NACE minimum knowledge rate (Critical)
    nace_node = evaluator.add_parallel(
        id="NACE_Minimum_Knowledge_Rate",
        desc="Identify NACE's recommended minimum knowledge rate for First Destination Surveys, supported by citation.",
        parent=parent_node,
        critical=True
    )

    # Critical existence + domain check for official NACE URLs
    official_nace_present = any(is_official_nace_url(u) for u in (nace.nace_citation_urls or []))
    evaluator.add_custom_node(
        result=official_nace_present,
        id="Minimum_Knowledge_Rate_Citation_URL",
        desc="Provides a URL to official NACE documentation supporting the stated minimum knowledge rate.",
        parent=nace_node,
        critical=True
    )

    # Verify the stated minimum knowledge rate value against the provided NACE URL(s)
    min_val_text = nace.min_knowledge_rate_value or ""
    claim = f"NACE recommends a minimum First Destination Survey knowledge rate of {min_val_text}."
    min_val_node = evaluator.add_leaf(
        id="Minimum_Knowledge_Rate_Value",
        desc="States the NACE-recommended minimum knowledge rate value (as specified in the constraints).",
        parent=nace_node,
        critical=True
    )
    await evaluator.verify(
        claim=claim,
        node=min_val_node,
        sources=nace.nace_citation_urls,
        additional_instruction="Use only official NACE documentation pages (e.g., naceweb.org or nacecenter.org) to verify the recommended minimum knowledge rate."
    )


async def verify_university_section(
    evaluator: Evaluator,
    parent_node,
    extracted: ExtractionOutput,
) -> Dict[str, Any]:
    uni = extracted.university or UniversityInfo()
    nace = extracted.nace or NACEInfo()

    uni_node = evaluator.add_parallel(
        id="University_That_Exceeds_Minimum",
        desc="Identify a major U.S. research university whose outcomes reporting shows a knowledge rate exceeding the NACE minimum by at least 20 percentage points in recent years, with citations.",
        parent=parent_node,
        critical=True
    )

    # University name provided (existence)
    evaluator.add_custom_node(
        result=bool(uni.name and uni.name.strip()),
        id="University_Name",
        desc="Provides the university's name.",
        parent=uni_node,
        critical=True
    )

    # Research university status URL provided (existence)
    evaluator.add_custom_node(
        result=bool(uni.research_status_urls),
        id="Major_US_Research_University_Status_Citation_URL",
        desc="Provides a URL supporting the claim that the institution qualifies as a major U.S. research university.",
        parent=uni_node,
        critical=True
    )

    # Verify research university status evidence
    status_claim = f"{uni.name or 'The institution'} qualifies as a major U.S. research university (e.g., Carnegie R1 or AAU member)."
    status_node = evaluator.add_leaf(
        id="Major_US_Research_University_Status_Evidence",
        desc="Provides evidence that the institution qualifies as a major U.S. research university (e.g., classification/listing or equivalent).",
        parent=uni_node,
        critical=True
    )
    await evaluator.verify(
        claim=status_claim,
        node=status_node,
        sources=uni.research_status_urls,
        additional_instruction="Acceptable evidence includes Carnegie Classification R1 (Very High Research Activity), AAU membership, or comparable authoritative classifications/listings. The source(s) should clearly indicate this standing."
    )

    # Knowledge rate URLs are official university URLs (existence/officialness check)
    official_uni_present = any(is_edu_url(u) for u in (uni.knowledge_rate_urls or []))
    evaluator.add_custom_node(
        result=official_uni_present and bool(uni.knowledge_rate_urls),
        id="Knowledge_Rate_Range_Citation_URL",
        desc="Provides an official university URL supporting the stated knowledge rate range and time span.",
        parent=uni_node,
        critical=True
    )

    # Verify knowledge rate range value against the provided university URL(s)
    kr_range_text = uni.knowledge_rate_range or ""
    kr_claim = f"According to official university sources, {uni.name or 'the university'} reports First Destination Survey knowledge rate(s) in the range {kr_range_text} in recent years."
    kr_range_node = evaluator.add_leaf(
        id="Knowledge_Rate_Range_Value",
        desc="Provides the specific knowledge rate range achieved in recent years.",
        parent=uni_node,
        critical=True
    )
    await evaluator.verify(
        claim=kr_claim,
        node=kr_range_node,
        sources=uni.knowledge_rate_urls,
        additional_instruction="It is acceptable if the range is supported by multiple year-by-year figures or by a page summarizing several recent graduating classes. Allow minor rounding differences."
    )

    # Verify that the range corresponds to multiple recent years (not a single year)
    multi_year_claim = f"The sources indicate that the reported knowledge rate(s) for {uni.name or 'the university'} correspond to multiple recent years (not just a single year)."
    multi_year_node = evaluator.add_leaf(
        id="Knowledge_Rate_Range_Is_Recent_Years",
        desc="Indicates the knowledge rate range corresponds to multiple recent years (not a single isolated year).",
        parent=uni_node,
        critical=True
    )
    await evaluator.verify(
        claim=multi_year_claim,
        node=multi_year_node,
        sources=uni.knowledge_rate_urls,
        additional_instruction="Confirm that the documentation references several recent graduating classes (e.g., last 3–5 years) or explicitly states that the results reflect multiple recent years."
    )

    # Numeric comparison: Exceeds NACE minimum by at least 20 percentage points
    nace_min = parse_percentage_value(nace.min_knowledge_rate_value)
    kr_lo, kr_hi = parse_percentage_range(uni.knowledge_rate_range)
    exceeds_20pp = False
    explained = {
        "parsed_nace_min": nace_min,
        "parsed_knowledge_rate_min": kr_lo,
        "parsed_knowledge_rate_max": kr_hi,
        "meets_20pp_margin": None,
        "computed_margin": None
    }
    if (nace_min is not None) and (kr_lo is not None):
        margin = kr_lo - nace_min
        exceeds_20pp = margin >= 20.0
        explained["computed_margin"] = round(margin, 4)
        explained["meets_20pp_margin"] = exceeds_20pp

    evaluator.add_custom_node(
        result=exceeds_20pp,
        id="Exceeds_NACE_Minimum_By_20pp",
        desc="Demonstrates (via numeric comparison) that the university's reported knowledge rate exceeds the NACE minimum by at least 20 percentage points.",
        parent=uni_node,
        critical=True
    )

    return explained


async def verify_best_in_nation_section(
    evaluator: Evaluator,
    parent_node,
    extracted: ExtractionOutput,
) -> None:
    uni = extracted.university or UniversityInfo()
    best = extracted.best_in_nation or BestInNationInfo()

    best_node = evaluator.add_parallel(
        id="Best_in_Nation_Comparable_Institutions_Verification",
        desc="Verify the characterization that this knowledge rate performance is among the best in the nation for comparable institutions, with citation.",
        parent=parent_node,
        critical=True
    )

    # Citation URL presence (existence)
    evaluator.add_custom_node(
        result=bool(best.citation_urls),
        id="Best_in_Nation_Claim_Citation_URL",
        desc="Provides a URL reference supporting the 'among the best in the nation for comparable institutions' characterization.",
        parent=best_node,
        critical=True
    )

    # Verify claim text using provided sources
    best_claim = f"The sources characterize the knowledge rate performance of {uni.name or 'the university'} as being among the best in the nation for comparable institutions (allow reasonable synonymy)."
    best_text_node = evaluator.add_leaf(
        id="Best_in_Nation_Claim_Text",
        desc="Includes a statement that the knowledge rate performance is described as among the best in the nation for comparable institutions.",
        parent=best_node,
        critical=True
    )
    await evaluator.verify(
        claim=best_claim,
        node=best_text_node,
        sources=best.citation_urls,
        additional_instruction="Accept close variants such as 'best in the nation', 'among the top nationally', or 'leading among comparable peer institutions'. The statement must pertain specifically to knowledge rate/outcomes coverage."
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
    Evaluate an answer for the NACE knowledge rate + university performance task.
    """
    # Initialize evaluator with a neutral root; create a critical sequential child as the true root of this task.
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Actual task root below will be sequential/critical
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

    # Extract all structured fields from the answer
    extracted: ExtractionOutput = await evaluator.extract(
        prompt=prompt_extract_all(),
        template_class=ExtractionOutput,
        extraction_name="extracted_fields"
    )

    # Build a critical sequential parent for the three major sections (as per rubric)
    task_root = evaluator.add_sequential(
        id="NACE_Knowledge_Rate_and_University_Performance",
        desc="Verify NACE recommended minimum knowledge rate and identify a qualifying major U.S. research university exceeding it by ≥20 percentage points in recent years, with supported citations.",
        parent=root,
        critical=True
    )

    # Section 1: NACE minimum knowledge rate
    await verify_nace_minimum_section(evaluator, task_root, extracted)

    # Section 2: University that exceeds minimum
    comparison_info = await verify_university_section(evaluator, task_root, extracted)

    # Section 3: "Best in nation" characterization
    await verify_best_in_nation_section(evaluator, task_root, extracted)

    # Record useful computed info for transparency
    evaluator.add_custom_info(
        info={
            "parsed_and_computed": comparison_info
        },
        info_type="debug_numeric_parsing",
        info_name="numeric_parsing_and_margins"
    )

    return evaluator.get_summary()