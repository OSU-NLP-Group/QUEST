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
TASK_ID = "nc_public_universities_enrollment_acres"
TASK_DESCRIPTION = """
Find three public universities in North Carolina that each meet the following criteria: total enrollment (undergraduate and graduate students combined) of at least 25,000 students, and main campus size of at least 1,000 acres. For each university, provide: (1) the official name of the university, (2) the city where its main campus is located, (3) the total enrollment figure (most recent available data), and (4) the main campus size in acres. Include reference URLs that support each piece of information.
"""


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class UniversityRecord(BaseModel):
    # Identity and eligibility
    official_name: Optional[str] = None
    eligibility_urls: List[str] = Field(default_factory=list)  # URLs confirming "public university in North Carolina"

    # Main campus city
    main_campus_city: Optional[str] = None
    city_urls: List[str] = Field(default_factory=list)

    # Total enrollment (most recent)
    total_enrollment: Optional[str] = None  # Keep as string to be robust to formatting (e.g., "~30,000")
    enrollment_as_of: Optional[str] = None  # e.g., "Fall 2024", "2024-25", or "most recent"
    enrollment_urls: List[str] = Field(default_factory=list)

    # Main campus size (acres)
    main_campus_acres: Optional[str] = None  # Keep as string to be robust
    campus_urls: List[str] = Field(default_factory=list)

    # Official name support URLs (optional but encouraged)
    name_urls: List[str] = Field(default_factory=list)


class UniversitiesExtraction(BaseModel):
    universities: List[UniversityRecord] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_universities() -> str:
    return """
    Extract up to all universities mentioned in the answer that are intended to satisfy the task. For each university, extract the following fields exactly as presented in the answer:

    - official_name: The university’s full official or formal name as provided in the answer.
    - name_urls: URL(s) that the answer cites to support the official name (can be the official site, UNC system page, or reputable encyclopedia).
    - eligibility_urls: URL(s) that support that the institution is a public university in North Carolina (e.g., "public university", "public research university", "UNC system", and located in North Carolina). If the answer provides only one set of sources, reuse those here.
    - main_campus_city: The city where the main campus is located, as provided in the answer.
    - city_urls: URL(s) that support the main campus city.
    - total_enrollment: The total enrollment (UG + Grad combined) figure as written in the answer (keep formatting, e.g., "30,000+" or "~29,500").
    - enrollment_as_of: Any as-of term/year/date or an explicit phrase like "most recent" that the answer associates with the enrollment; if not present, return null.
    - enrollment_urls: URL(s) that support the total enrollment figure (and ideally the as-of/recency).
    - main_campus_acres: The size of the MAIN campus in acres as written in the answer (keep formatting, e.g., "~1,200 acres", "over 1,000 acres").
    - campus_urls: URL(s) that support the main campus acreage.

    Important extraction rules for URLs:
    - Extract only URLs explicitly present in the answer (including URLs in markdown links).
    - Do not fabricate URLs. If none are given for a field, return an empty list.
    - Return full URLs with protocol. If a URL is missing protocol, prepend "http://".

    Return JSON with a single top-level field:
    {
      "universities": [ UniversityRecord, ... ]
    }
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _first_number_from_text(text: Optional[str]) -> Optional[float]:
    if not text:
        return None
    # Normalize unicode and lowercase for safety
    s = str(text)
    # Common numeric patterns with optional commas and decimals; ignore percent or year contexts
    matches = re.findall(r"(?:(?:approx\.?|about|around|over|nearly|~)\s*)?(\d{1,3}(?:,\d{3})*(?:\.\d+)?|\d+(?:\.\d+)?)", s, flags=re.IGNORECASE)
    if not matches:
        return None
    # Choose the first meaningful number; remove commas
    try:
        val = float(matches[0].replace(",", ""))
        return val
    except Exception:
        return None


def _at_least_threshold(value_text: Optional[str], threshold: float) -> bool:
    val = _first_number_from_text(value_text)
    return (val is not None) and (val >= threshold)


def _dedupe_urls(urls: List[str]) -> List[str]:
    seen = set()
    out = []
    for u in urls:
        if not u:
            continue
        u = u.strip()
        if not u:
            continue
        # Normalize very simply here; framework handles further normalization
        if u.lower() not in seen:
            seen.add(u.lower())
            out.append(u)
    return out


def _combine_urls(*url_lists: List[str]) -> List[str]:
    combined = []
    for lst in url_lists:
        combined.extend(lst or [])
    return _dedupe_urls(combined)


# --------------------------------------------------------------------------- #
# Verification for a single university                                        #
# --------------------------------------------------------------------------- #
async def verify_university(
    evaluator: Evaluator,
    parent_node,
    uni: UniversityRecord,
    idx: int,
) -> None:
    """
    Build verification leaves for a single university under parent_node.
    """
    u_node = evaluator.add_parallel(
        id=f"university_{idx+1}",
        desc=f"University #{idx+1} (scored independently for partial credit).",
        parent=parent_node,
        critical=False
    )

    # 1) Eligibility: Public university in North Carolina (requires URLs)
    elig_node = evaluator.add_leaf(
        id=f"U{idx+1}_Eligibility_Public_In_NC",
        desc="The identified institution is a public university located in North Carolina.",
        parent=u_node,
        critical=True
    )
    elig_urls = _combine_urls(uni.eligibility_urls, uni.name_urls, uni.city_urls, uni.enrollment_urls, uni.campus_urls)
    elig_claim = f"{uni.official_name or 'The institution'} is a public university located in the state of North Carolina, United States."
    await evaluator.verify(
        claim=elig_claim,
        node=elig_node,
        sources=elig_urls,
        additional_instruction="Verify the page clearly indicates the institution is a public (e.g., 'public university', 'public research university', 'land‑grant public') and that it is located in North Carolina (e.g., shows city in NC or states 'North Carolina'). Accept UNC System membership as evidence of 'public in NC'."
    )

    # 2) Official name with supporting URLs
    name_node = evaluator.add_leaf(
        id=f"U{idx+1}_Official_Name_With_URL",
        desc="Provides the university’s full official name and includes reference URL(s) supporting the name.",
        parent=u_node,
        critical=True
    )
    name_claim = f"The official or formal name of this institution is '{uni.official_name}'. Treat common official short forms as equivalent if they clearly refer to the same institution."
    await evaluator.verify(
        claim=name_claim,
        node=name_node,
        sources=_dedupe_urls(uni.name_urls),
        additional_instruction="Accept reasonable official variants (e.g., 'NC State University' vs 'North Carolina State University') if the page clearly indicates they are the same institution."
    )

    # 3) Main campus city with supporting URLs
    city_node = evaluator.add_leaf(
        id=f"U{idx+1}_Main_Campus_City_With_URL",
        desc="Provides the city where the main campus is located and includes reference URL(s) supporting the city.",
        parent=u_node,
        critical=True
    )
    city_name = uni.main_campus_city or ""
    city_claim = f"The main campus of {uni.official_name or 'the university'} is located in {city_name}, North Carolina."
    await evaluator.verify(
        claim=city_claim,
        node=city_node,
        sources=_dedupe_urls(uni.city_urls),
        additional_instruction="Focus on MAIN campus location. If multiple campuses are listed, the page should indicate which one is the main campus (or the city commonly recognized as main). Minor phrasing differences are acceptable."
    )

    # 4) Total enrollment (most recent) with supporting URLs
    enroll_node = evaluator.add_leaf(
        id=f"U{idx+1}_Total_Enrollment_Most_Recent_With_URL",
        desc="Reports the most recent available total enrollment (UG+Grad combined) and includes reference URL(s) supporting the figure and its as-of year/term/date (or explicit 'latest/most recent' indicator).",
        parent=u_node,
        critical=True
    )
    if uni.enrollment_as_of:
        enroll_claim = f"The total (undergraduate + graduate) enrollment is {uni.total_enrollment}, as of {uni.enrollment_as_of} (or otherwise indicated as a most recent/latest figure) on the cited page."
    else:
        enroll_claim = f"The total (undergraduate + graduate) enrollment is {uni.total_enrollment}, and the cited page indicates this figure is the most recent available (via an as-of term/year/date or explicit 'most recent/latest' phrasing)."
    await evaluator.verify(
        claim=enroll_claim,
        node=enroll_node,
        sources=_dedupe_urls(uni.enrollment_urls),
        additional_instruction="Verify that the page supports BOTH the total enrollment number (or an acceptably close rounded equivalent) and that it is the most recent (either by explicit 'as-of' term/year/date or an explicit 'latest/most recent' statement). Allow small rounding differences and formatting variants."
    )

    # 5) Total enrollment >= 25,000 (computed check)
    enroll_threshold_ok = _at_least_threshold(uni.total_enrollment, 25000)
    evaluator.add_custom_node(
        result=enroll_threshold_ok,
        id=f"U{idx+1}_Total_Enrollment_At_Least_25000",
        desc="The reported total enrollment figure is ≥ 25,000.",
        parent=u_node,
        critical=True
    )

    # 6) Main campus size (acres) with supporting URLs
    acres_node = evaluator.add_leaf(
        id=f"U{idx+1}_Main_Campus_Size_Acres_With_URL",
        desc="Reports main campus size in acres and includes reference URL(s) supporting the acreage figure.",
        parent=u_node,
        critical=True
    )
    acres_claim = f"The main campus size of {uni.official_name or 'the university'} is {uni.main_campus_acres} acres (approximate phrasing acceptable if equivalent)."
    await evaluator.verify(
        claim=acres_claim,
        node=acres_node,
        sources=_dedupe_urls(uni.campus_urls),
        additional_instruction="Verify the page states the MAIN campus acreage. Accept approximate phrases such as 'about', 'over', or '~' when they clearly indicate an equivalent value."
    )

    # 7) Main campus size >= 1,000 acres (computed check)
    acres_threshold_ok = _at_least_threshold(uni.main_campus_acres, 1000)
    evaluator.add_custom_node(
        result=acres_threshold_ok,
        id=f"U{idx+1}_Main_Campus_Size_At_Least_1000_Acres",
        desc="The reported main campus size is ≥ 1,000 acres.",
        parent=u_node,
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
    model: str = "o4-mini"
) -> Dict:
    """
    Evaluate an answer for the 'NC public universities with enrollment and acreage thresholds' task.
    """
    # Initialize evaluator (root as non-critical parallel to allow partial credit across universities)
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

    # Extract structured data from answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_universities(),
        template_class=UniversitiesExtraction,
        extraction_name="universities_extraction",
    )

    # Select first 3 distinct by official name (case-insensitive), preserving order
    selected: List[UniversityRecord] = []
    seen_names = set()
    for item in extracted.universities:
        name = (item.official_name or "").strip()
        if not name:
            continue
        key = name.lower()
        if key in seen_names:
            continue
        seen_names.add(key)
        selected.append(item)
        if len(selected) == 3:
            break

    # Pad to 3 if fewer provided
    while len(selected) < 3:
        selected.append(UniversityRecord())

    # Three distinct universities check (non-critical to allow partial credit)
    distinct_names = [u.official_name.strip() for u in selected if u.official_name and u.official_name.strip()]
    are_three_distinct = (len(distinct_names) == 3) and (len({n.lower() for n in distinct_names}) == 3)
    evaluator.add_custom_node(
        result=are_three_distinct,
        id="Three_Distinct_Universities",
        desc="Response includes exactly three non-duplicate universities (distinct institutions).",
        parent=root,
        critical=False
    )

    # Build verification subtrees for each university
    for i in range(3):
        await verify_university(evaluator, root, selected[i], i)

    # Optional: record a concise summary of selected items
    evaluator.add_custom_info(
        info={
            "selected_universities": [
                {
                    "official_name": u.official_name,
                    "main_campus_city": u.main_campus_city,
                    "total_enrollment": u.total_enrollment,
                    "enrollment_as_of": u.enrollment_as_of,
                    "main_campus_acres": u.main_campus_acres,
                } for u in selected
            ]
        },
        info_type="selection_summary"
    )

    return evaluator.get_summary()