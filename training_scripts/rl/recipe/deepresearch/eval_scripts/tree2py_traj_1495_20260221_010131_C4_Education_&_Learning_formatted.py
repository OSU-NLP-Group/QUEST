import asyncio
import logging
import re
from typing import Optional, List, Dict, Any, Tuple

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.llm_client.base_client import LLMClient


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "ma_public_highschools_usnews_2025_3schools"
TASK_DESCRIPTION = """
Identify 3 public high schools in Massachusetts that are ranked in the U.S. News 2025-2026 Best High Schools rankings and collectively meet ALL of the following requirements:

1. All 3 schools must be public high schools located in Massachusetts and eligible for U.S. News ranking (offering 12th grade with at least 15 students enrolled in 12th grade)

2. All 3 schools must be ranked in the U.S. News 2025-2026 Best High Schools rankings for Massachusetts

3. All 3 schools must have College Readiness Index (CRI) data available in the U.S. News rankings, indicating the presence of AP or IB programs with at least 10 students taking exams

4. At least 2 of the 3 schools must have total enrollment (grades 9-12) between 1,500 and 2,500 students

5. At least 1 of the 3 schools must have total minority enrollment exceeding 45%

6. At least 2 of the 3 schools must have a student-teacher ratio of 12:1 or better (lower)

7. At least 1 of the 3 schools must be ranked in the top 30 public high schools in Massachusetts by U.S. News

8. The 3 schools must be located in at least 2 different municipalities (cities/towns) in Massachusetts

For each identified school, provide: (a) the school name, (b) its U.S. News ranking position in Massachusetts, (c) total enrollment for grades 9-12, (d) total minority enrollment percentage, (e) student-teacher ratio, (f) municipality location, and (g) a reference URL from U.S. News or official school sources supporting these details.
""".strip()


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class SchoolItem(BaseModel):
    name: Optional[str] = None
    ma_ranking: Optional[str] = None
    total_enrollment: Optional[str] = None
    minority_enrollment_pct: Optional[str] = None
    student_teacher_ratio: Optional[str] = None
    municipality: Optional[str] = None
    source_urls: List[str] = Field(default_factory=list)


class SchoolsExtraction(BaseModel):
    schools: List[SchoolItem] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_schools() -> str:
    return """
    Extract the schools presented in the answer. We are looking for public high schools in Massachusetts as identified by the answer text.

    For each school mentioned in the answer, extract the following fields exactly as written in the answer:
    1) name: the school name
    2) ma_ranking: the school's U.S. News Massachusetts ranking position (as written, e.g., "#18 in Massachusetts" or "MA rank: 18")
    3) total_enrollment: the total enrollment for grades 9-12 (as written, e.g., "1,850" or "about 1,800")
    4) minority_enrollment_pct: the total minority enrollment percentage (as written, e.g., "46%" or "~47%")
    5) student_teacher_ratio: the student-teacher ratio (as written, e.g., "12:1" or "11 to 1")
    6) municipality: the municipality (city or town) in Massachusetts where the school is located (as written)
    7) source_urls: all reference URLs cited in the answer for that school. Include any U.S. News profile URLs and any official school sources cited. If multiple URLs are provided, include them all. Do not infer URLs; only include those explicitly present in the answer (plain or in markdown).

    Return a JSON object with a single key "schools" that is an array of school objects with the above fields. If a field is missing for a school, set it to null (or an empty array for source_urls). Extract all schools mentioned in the answer (not just three); we will select the first three later.
    """


# --------------------------------------------------------------------------- #
# Helper parsing functions                                                    #
# --------------------------------------------------------------------------- #
def parse_first_int(text: Optional[str]) -> Optional[int]:
    if not text:
        return None
    # Remove commas
    cleaned = re.sub(r"[,\s]+", " ", text).strip()
    # Look for the first integer-like token
    m = re.search(r"(\d+)", cleaned)
    if not m:
        return None
    try:
        return int(m.group(1))
    except Exception:
        return None


def parse_enrollment(text: Optional[str]) -> Optional[int]:
    # Try to parse a single integer representing total enrollment
    return parse_first_int(text)


def parse_ratio_to_number(text: Optional[str]) -> Optional[float]:
    if not text:
        return None
    s = text.strip().lower()
    # Try formats like "12:1" or "11 to 1"
    if ":" in s:
        try:
            left = s.split(":")[0].strip()
            val = float(re.sub(r"[^\d\.]", "", left))
            return val
        except Exception:
            pass
    # "11 to 1"
    m = re.findall(r"(\d+(?:\.\d+)?)", s)
    if m:
        try:
            # Return first number as students per 1 teacher
            return float(m[0])
        except Exception:
            return None
    return None


def parse_percent(text: Optional[str]) -> Optional[float]:
    if not text:
        return None
    s = text.strip().lower()
    # find the first numeric token (could be "45", "45.6", "0.456")
    m = re.search(r"(\d+(?:\.\d+)?)", s)
    if not m:
        return None
    try:
        val = float(m.group(1))
        # If explicitly contains "%" we treat as percentage already
        if "%" in s:
            return val
        # otherwise, if between 0 and 1, interpret as fraction
        if 0.0 < val <= 1.0:
            return val * 100.0
        return val
    except Exception:
        return None


def parse_ma_rank(text: Optional[str]) -> Optional[int]:
    # Expect formats like "#18 in Massachusetts", "MA rank: 22", "Rank 30 (MA)", etc.
    return parse_first_int(text)


def normalize_municipality(text: Optional[str]) -> Optional[str]:
    if not text:
        return None
    return re.sub(r"\s+", " ", text.strip().lower())


def pick_urls(urls: List[str]) -> List[str]:
    # Filter obviously invalid entries and keep unique
    dedup = []
    for u in urls:
        if not u:
            continue
        u = u.strip()
        if not re.match(r"^https?://", u):
            # If missing protocol, prepend http:// per rules
            u = "http://" + u
        if u not in dedup:
            dedup.append(u)
    return dedup


# --------------------------------------------------------------------------- #
# Per‑school verification helpers                                             #
# --------------------------------------------------------------------------- #
class SchoolVerifyResult(BaseModel):
    public_ma_ok: bool = False
    ranked_cycle_ok: bool = False
    cri_ok: bool = False


async def verify_school_critical(
    evaluator: Evaluator,
    parent_node,
    school: SchoolItem,
    idx: int,
) -> SchoolVerifyResult:
    """
    Add critical per-school checks that gate subsequent group criteria.
    All children within this branch are critical (to satisfy framework constraints).
    """
    school_node = evaluator.add_parallel(
        id=f"school_{idx}_critical",
        desc=f"School #{idx+1} critical validation",
        parent=parent_node,
        critical=True
    )

    # Existence: must have name and at least one source URL
    exists_ok = bool(school and school.name and (school.source_urls and len(school.source_urls) > 0))
    evaluator.add_custom_node(
        result=exists_ok,
        id=f"school_{idx}_exists",
        desc=f"School #{idx+1} has name and at least one source URL",
        parent=school_node,
        critical=True
    )

    urls = pick_urls(school.source_urls if school and school.source_urls else [])

    # Check public MA eligibility for U.S. News ranking
    n_public = evaluator.add_leaf(
        id=f"school_{idx}_public_ma_eligible",
        desc=f"School #{idx+1} is a public high school in Massachusetts offering grade 12 and eligible for U.S. News ranking",
        parent=school_node,
        critical=True
    )
    public_claim = (
        f"{school.name or 'This school'} is a public high school located in Massachusetts, "
        f"serves grade 12, and is eligible for U.S. News Best High Schools ranking."
    )
    public_ok = await evaluator.verify(
        claim=public_claim,
        node=n_public,
        sources=urls,
        additional_instruction=(
            "Prefer evidence from the U.S. News school profile if present. "
            "If the page shows grades including 12 (e.g., 'Grades 9-12') and indicates it's a public high school in MA, "
            "treat the school as eligible. If explicit 12th-grade headcount isn't shown, the presence of a U.S. News ranking/profile "
            "for the school implies eligibility under U.S. News criteria."
        )
    )

    # Check the school is ranked in 2025-2026 MA list (or page clearly indicates 2025-2026 cycle)
    n_ranked = evaluator.add_leaf(
        id=f"school_{idx}_ranked_2025_2026",
        desc=f"School #{idx+1} is ranked in Massachusetts in the 2025-2026 U.S. News Best High Schools",
        parent=school_node,
        critical=True
    )
    ranked_claim = (
        f"{school.name or 'This school'} is shown as ranked in Massachusetts in the 2025-2026 U.S. News Best High Schools cycle."
    )
    ranked_ok = await evaluator.verify(
        claim=ranked_claim,
        node=n_ranked,
        sources=urls,
        additional_instruction=(
            "Look for explicit '2025-2026' cycle indicators on the U.S. News page. "
            "If the page clearly displays the Massachusetts rank for the school within the 2025-2026 Best High Schools, mark as supported. "
            "If multiple pages are cited, it's sufficient that one clearly shows the 2025-2026 MA ranking."
        )
    )

    # Check CRI (College Readiness Index) is present
    n_cri = evaluator.add_leaf(
        id=f"school_{idx}_cri_present",
        desc=f"School #{idx+1} has College Readiness Index (CRI) data present on U.S. News",
        parent=school_node,
        critical=True
    )
    cri_claim = (
        f"The U.S. News profile for {school.name or 'this school'} shows a College Readiness Index (CRI) metric or section."
    )
    cri_ok = await evaluator.verify(
        claim=cri_claim,
        node=n_cri,
        sources=urls,
        additional_instruction=(
            "Verify the page includes a 'College Readiness Index' value or explicit section. "
            "U.S. News generally reports a CRI only if AP/IB participation meets minimum thresholds; "
            "presence of the CRI metric is sufficient to indicate AP/IB participation with at least ~10 exam takers per methodology."
        )
    )

    return SchoolVerifyResult(public_ma_ok=bool(public_ok), ranked_cycle_ok=bool(ranked_ok), cri_ok=bool(cri_ok))


async def verify_school_fields(
    evaluator: Evaluator,
    parent_node,
    school: SchoolItem,
    idx: int
) -> None:
    """
    Non-critical per-school field support checks using the provided URLs.
    """
    fields_node = evaluator.add_parallel(
        id=f"school_{idx}_fields",
        desc=f"School #{idx+1} field support verifications",
        parent=parent_node,
        critical=False
    )
    urls = pick_urls(school.source_urls if school and school.source_urls else [])

    # Ranking support
    if school and school.ma_ranking:
        n_rank = evaluator.add_leaf(
            id=f"school_{idx}_ma_rank_supported",
            desc=f"School #{idx+1} Massachusetts ranking matches the page",
            parent=fields_node,
            critical=False
        )
        claim = (
            f"The Massachusetts rank reported for {school.name or 'this school'} is {school.ma_ranking}."
        )
        await evaluator.verify(
            claim=claim,
            node=n_rank,
            sources=urls,
            additional_instruction=(
                "Check the school's Massachusetts rank shown on the U.S. News page. "
                "Allow minor formatting differences (e.g., 'Tied at #20', '#20 in MA')."
            )
        )

    # Total enrollment support
    if school and school.total_enrollment:
        n_enr = evaluator.add_leaf(
            id=f"school_{idx}_enrollment_supported",
            desc=f"School #{idx+1} total enrollment is supported by sources",
            parent=fields_node,
            critical=False
        )
        claim = (
            f"The total student enrollment for grades 9-12 at {school.name or 'this school'} is approximately {school.total_enrollment}."
        )
        await evaluator.verify(
            claim=claim,
            node=n_enr,
            sources=urls,
            additional_instruction=(
                "Verify the total enrollment value on U.S. News (or official school's statistics page). "
                "Allow minor rounding differences."
            )
        )

    # Minority enrollment support
    if school and school.minority_enrollment_pct:
        n_min = evaluator.add_leaf(
            id=f"school_{idx}_minority_supported",
            desc=f"School #{idx+1} minority enrollment percentage is supported by sources",
            parent=fields_node,
            critical=False
        )
        claim = (
            f"The total minority enrollment at {school.name or 'this school'} is approximately {school.minority_enrollment_pct}."
        )
        await evaluator.verify(
            claim=claim,
            node=n_min,
            sources=urls,
            additional_instruction=(
                "Verify the minority enrollment percentage on the U.S. News page (or official school's report). "
                "Allow minor rounding differences."
            )
        )

    # Student-teacher ratio support
    if school and school.student_teacher_ratio:
        n_ratio = evaluator.add_leaf(
            id=f"school_{idx}_ratio_supported",
            desc=f"School #{idx+1} student-teacher ratio is supported by sources",
            parent=fields_node,
            critical=False
        )
        claim = (
            f"The student-teacher ratio at {school.name or 'this school'} is approximately {school.student_teacher_ratio}."
        )
        await evaluator.verify(
            claim=claim,
            node=n_ratio,
            sources=urls,
            additional_instruction=(
                "Verify the student-teacher ratio on the U.S. News page. "
                "Accept equivalent representations (e.g., '11 to 1' vs '11:1')."
            )
        )

    # Municipality support
    if school and school.municipality:
        n_loc = evaluator.add_leaf(
            id=f"school_{idx}_municipality_supported",
            desc=f"School #{idx+1} municipality location is supported by sources",
            parent=fields_node,
            critical=False
        )
        claim = (
            f"{school.name or 'This school'} is located in {school.municipality}, Massachusetts."
        )
        await evaluator.verify(
            claim=claim,
            node=n_loc,
            sources=urls,
            additional_instruction=(
                "Verify the school location (city/town) as shown on U.S. News or the official school website. "
                "Allow neighborhood or district naming variants where appropriate."
            )
        )


# --------------------------------------------------------------------------- #
# Aggregated criteria computation                                             #
# --------------------------------------------------------------------------- #
def compute_group_criteria(
    schools: List[SchoolItem],
    critical_results: List[SchoolVerifyResult]
) -> Dict[str, bool]:
    # Numeric/parsed arrays
    enrollments = [parse_enrollment(s.total_enrollment) for s in schools]
    minority_pcts = [parse_percent(s.minority_enrollment_pct) for s in schools]
    ratios = [parse_ratio_to_number(s.student_teacher_ratio) for s in schools]
    ranks = [parse_ma_rank(s.ma_ranking) for s in schools]
    municipalities = [normalize_municipality(s.municipality) for s in schools]

    # Criterion 1: All 3 are public MA and eligible -> all True from critical_results.public_ma_ok
    c1 = all(res.public_ma_ok for res in critical_results)

    # Criterion 2: All 3 are ranked in MA for 2025-2026 -> all True from critical_results.ranked_cycle_ok
    c2 = all(res.ranked_cycle_ok for res in critical_results)

    # Criterion 3: At least 2 enrollments between 1500 and 2500
    enr_in_range = [e for e in enrollments if e is not None and 1500 <= e <= 2500]
    c3 = len(enr_in_range) >= 2

    # Criterion 4: At least 1 minority > 45%
    c4 = any((p is not None and p > 45.0) for p in minority_pcts)

    # Criterion 5: At least 2 have ratio <= 12:1
    ratio_good = [r for r in ratios if r is not None and r <= 12.0]
    c5 = len(ratio_good) >= 2

    # Criterion 6: All 3 have CRI -> use critical_results.cri_ok
    c6 = all(res.cri_ok for res in critical_results)

    # Criterion 7: At least 1 ranked in top 30 (MA)
    c7 = any((rk is not None and rk <= 30) for rk in ranks)

    # Criterion 8: At least 2 different municipalities
    muni_set = {m for m in municipalities if m}
    c8 = len(muni_set) >= 2

    return {
        "criterion_1_public_massachusetts": c1,
        "criterion_2_us_news_ranked": c2,
        "criterion_3_enrollment_range": c3,
        "criterion_4_minority_enrollment": c4,
        "criterion_5_student_teacher_ratio": c5,
        "criterion_6_college_readiness": c6,
        "criterion_7_state_ranking": c7,
        "criterion_8_geographic_diversity": c8,
    }


# --------------------------------------------------------------------------- #
# Main evaluation function                                                    #
# --------------------------------------------------------------------------- #
async def evaluate_answer(
    client: LLMClient,
    answer: str,
    agent_name: str,
    answer_name: str,
    cache: CacheFileSys,
    semaphore: asyncio.Semaphore,
    logger: logging.Logger,
    model: str = "o4-mini"
) -> Dict[str, Any]:
    """
    Evaluation entry point for the Massachusetts public high schools selection task.
    """
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Root non-critical to allow mixed children
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

    # Extract all schools mentioned in the answer
    extracted: SchoolsExtraction = await evaluator.extract(
        prompt=prompt_extract_schools(),
        template_class=SchoolsExtraction,
        extraction_name="schools_extraction"
    )

    # Select first 3 schools; pad with empty if fewer
    selected: List[SchoolItem] = list(extracted.schools[:3])
    while len(selected) < 3:
        selected.append(SchoolItem())

    # Build a sequential pipeline: (1) critical per-school checks -> (2) non-critical field support -> (3) aggregated criteria
    pipeline = evaluator.add_sequential(
        id="evaluation_pipeline",
        desc="Pipeline: per-school critical checks -> field support -> aggregated group criteria",
        parent=root,
        critical=False
    )

    # 1) Per-school CRITICAL checks (as a critical parent with critical children only)
    critical_parent = evaluator.add_parallel(
        id="per_school_critical",
        desc="Per-school critical validations (public MA + ranked 2025-2026 + CRI present)",
        parent=pipeline,
        critical=True
    )

    # Gather results
    critical_results: List[SchoolVerifyResult] = []

    for i, school in enumerate(selected):
        res = await verify_school_critical(evaluator, critical_parent, school, i)
        critical_results.append(res)

    # 2) Per-school NON-CRITICAL field support (evaluated only if previous critical block passes, due to sequential gating)
    field_parent = evaluator.add_parallel(
        id="per_school_fields",
        desc="Per-school field support verifications (ranking, enrollment, minority %, ratio, municipality)",
        parent=pipeline,
        critical=False
    )

    for i, school in enumerate(selected):
        await verify_school_fields(evaluator, field_parent, school, i)

    # 3) Aggregated group criteria (reflect rubric criteria). This block is skipped if critical block fails.
    group_parent = evaluator.add_parallel(
        id="group_criteria",
        desc="Aggregated criteria across the 3 selected schools",
        parent=pipeline,
        critical=False
    )

    # Compute results based on extracted values and critical checks
    group_results = compute_group_criteria(selected, critical_results)

    # Criterion 1: All 3 public MA & eligible (CRITICAL)
    evaluator.add_custom_node(
        result=group_results["criterion_1_public_massachusetts"],
        id="criterion_1_public_massachusetts",
        desc="All 3 identified schools are public high schools in Massachusetts and eligible for U.S. News ranking (offer grade 12)",
        parent=group_parent,
        critical=True
    )

    # Criterion 2: All 3 ranked in U.S. News 2025-2026 MA (CRITICAL)
    evaluator.add_custom_node(
        result=group_results["criterion_2_us_news_ranked"],
        id="criterion_2_us_news_ranked",
        desc="All 3 identified schools are ranked in the U.S. News 2025-2026 Best High Schools for Massachusetts",
        parent=group_parent,
        critical=True
    )

    # Criterion 3: At least 2 enrollments between 1,500 and 2,500 (NON-CRITICAL)
    evaluator.add_custom_node(
        result=group_results["criterion_3_enrollment_range"],
        id="criterion_3_enrollment_range",
        desc="At least 2 of the 3 schools have total enrollment (grades 9-12) between 1,500 and 2,500 students",
        parent=group_parent,
        critical=False
    )

    # Criterion 4: At least 1 minority > 45% (NON-CRITICAL)
    evaluator.add_custom_node(
        result=group_results["criterion_4_minority_enrollment"],
        id="criterion_4_minority_enrollment",
        desc="At least 1 of the 3 schools has total minority enrollment exceeding 45%",
        parent=group_parent,
        critical=False
    )

    # Criterion 5: At least 2 ratio <= 12:1 (NON-CRITICAL)
    evaluator.add_custom_node(
        result=group_results["criterion_5_student_teacher_ratio"],
        id="criterion_5_student_teacher_ratio",
        desc="At least 2 of the 3 schools have a student-teacher ratio of 12:1 or better (lower)",
        parent=group_parent,
        critical=False
    )

    # Criterion 6: All 3 have CRI data available (CRITICAL)
    evaluator.add_custom_node(
        result=group_results["criterion_6_college_readiness"],
        id="criterion_6_college_readiness",
        desc="All 3 schools have College Readiness Index (CRI) data available in U.S. News rankings",
        parent=group_parent,
        critical=True
    )

    # Criterion 7: At least 1 ranked top 30 in MA (NON-CRITICAL)
    evaluator.add_custom_node(
        result=group_results["criterion_7_state_ranking"],
        id="criterion_7_state_ranking",
        desc="At least 1 of the 3 schools is ranked in the top 30 public high schools in Massachusetts by U.S. News",
        parent=group_parent,
        critical=False
    )

    # Criterion 8: At least 2 different municipalities (NON-CRITICAL)
    evaluator.add_custom_node(
        result=group_results["criterion_8_geographic_diversity"],
        id="criterion_8_geographic_diversity",
        desc="The 3 schools are located in at least 2 different municipalities (cities/towns) in Massachusetts",
        parent=group_parent,
        critical=False
    )

    # Add a compact custom info block to show parsed numbers that drove group criteria
    parsed_debug = []
    for s in selected:
        parsed_debug.append({
            "name": s.name,
            "parsed_ma_rank": parse_ma_rank(s.ma_ranking),
            "parsed_enrollment": parse_enrollment(s.total_enrollment),
            "parsed_minority_pct": parse_percent(s.minority_enrollment_pct),
            "parsed_ratio": parse_ratio_to_number(s.student_teacher_ratio),
            "municipality_norm": normalize_municipality(s.municipality),
            "num_sources": len(s.source_urls or [])
        })
    evaluator.add_custom_info(
        info={"schools_parsed": parsed_debug, "group_results": group_results},
        info_type="computed_metrics",
        info_name="parsed_and_group_results"
    )

    return evaluator.get_summary()