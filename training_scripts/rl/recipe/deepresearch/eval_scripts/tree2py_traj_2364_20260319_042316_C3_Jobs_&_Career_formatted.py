import asyncio
import logging
import re
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "hbs_cme_program_selection"
TASK_DESCRIPTION = (
    "A senior professional with 12 years of experience is planning to earn Harvard Business School's Certificate of "
    "Management Excellence. They have the following constraints: (1) Budget: The total cost of all three required "
    "programs must not exceed $45,000; (2) Timeline: All programs must start in May 2026 or later, and all three "
    "programs must be completable within 36 consecutive months from the start date of the first program; "
    "(3) Format requirement: Due to work travel limitations, at least one of the three programs must be offered in "
    "Virtual or Blended format (not all three can be In-Person only); (4) Program requirements: As per the certificate "
    "structure, they must complete exactly one qualifying leadership program, one qualifying strategy program, and one "
    "qualifying elective program. Identify the three specific programs (by their official program names) that this "
    "professional should select to satisfy all constraints, and provide the email address they should contact to "
    "initiate their Certificate of Management Excellence application."
)
BUDGET_LIMIT = 45000.0
MIN_START_YEAR = 2026
MIN_START_MONTH = 5  # May
TIMELINE_MONTH_WINDOW = 36


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class ProgramItem(BaseModel):
    name: Optional[str] = None
    category: Optional[str] = None  # leadership / strategy / elective (as claimed in the answer)
    start_date: Optional[str] = None  # e.g., "May 2026", "May 12–16, 2026", etc.
    format: Optional[str] = None  # e.g., "In-Person", "Virtual", "Blended", "Hybrid", "Live Online"
    cost: Optional[str] = None  # tuition/program fee text
    urls: List[str] = Field(default_factory=list)  # official HBS Exec Ed page(s) or relevant HBS pages


class CMESelectionExtraction(BaseModel):
    leadership_program: Optional[ProgramItem] = None
    strategy_program: Optional[ProgramItem] = None
    elective_program: Optional[ProgramItem] = None
    contact_email: Optional[str] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_programs() -> str:
    return """
Extract exactly three Harvard Business School Executive Education programs from the answer that are intended for the Certificate of Management Excellence (CME), one per category:
- leadership_program
- strategy_program
- elective_program

For each of the three programs, extract:
1) name: The official program name as shown on the HBS Executive Education website.
2) category: One of ["leadership", "strategy", "elective"] as asserted by the answer.
3) start_date: The specific session start month and year (e.g., "May 2026" or "May 12–16, 2026") claimed in the answer for the cohort the user should take.
4) format: The delivery format as claimed by the answer (e.g., "In-Person", "Virtual", "Live Online", "Blended", or "Hybrid").
5) cost: The tuition/program fee amount as claimed by the answer (in USD, if provided).
6) urls: All URLs the answer cites for this program (prefer official HBS Executive Education pages). Include every URL explicitly shown in the answer text for this program.

Also extract:
- contact_email: The email address the answer says to use to initiate the CME application.

Rules:
- Extract only what appears in the answer text; do not invent details.
- URLs can be in plain form or markdown links; extract the actual URL(s).
- If any field is missing, set it to null (or [] for urls).
- Normalize category to one of: leadership, strategy, elective (lowercase).
"""


# --------------------------------------------------------------------------- #
# Helpers: parsing and normalization                                          #
# --------------------------------------------------------------------------- #
MONTH_MAP = {
    "january": 1, "jan": 1,
    "february": 2, "feb": 2,
    "march": 3, "mar": 3,
    "april": 4, "apr": 4,
    "may": 5,
    "june": 6, "jun": 6,
    "july": 7, "jul": 7,
    "august": 8, "aug": 8,
    "september": 9, "sep": 9, "sept": 9,
    "october": 10, "oct": 10,
    "november": 11, "nov": 11,
    "december": 12, "dec": 12,
}


def parse_money_usd(text: Optional[str]) -> Optional[float]:
    if not text:
        return None
    # Remove everything except digits and dot, then handle commas
    # Prefer extracting the largest numeric token that looks like an amount
    nums = re.findall(r"[0-9][0-9,]*(?:\.[0-9]{1,2})?", text)
    if not nums:
        return None
    # Choose the largest numerical value among tokens (tuition usually the largest)
    values = []
    for tok in nums:
        try:
            values.append(float(tok.replace(",", "")))
        except ValueError:
            continue
    if not values:
        return None
    return max(values)


def parse_month_year(text: Optional[str]) -> Optional[Tuple[int, int]]:
    """
    Extract (year, month) with priority to the first recognizable month-year pair.
    Accepts forms like:
      - "May 2026"
      - "May 12–16, 2026"
      - "Starting May 2026 (virtual)"
    """
    if not text:
        return None
    t = text.strip()
    # Look for a month name and a 4-digit year
    month_pattern = r"(January|Jan|February|Feb|March|Mar|April|Apr|May|June|Jun|July|Jul|August|Aug|September|Sept|Sep|October|Oct|November|Nov|December|Dec)"
    year_pattern = r"([12][0-9]{3})"
    regex = re.compile(month_pattern + r"[^0-9A-Za-z]{0,10}" + year_pattern, flags=re.IGNORECASE)
    m = regex.search(t)
    if not m:
        # Try year-first formats like "2026 May"
        regex2 = re.compile(year_pattern + r"[^0-9A-Za-z]{0,10}" + month_pattern, flags=re.IGNORECASE)
        m2 = regex2.search(t)
        if not m2:
            return None
        year = int(m2.group(1))
        month_name = m2.group(2).lower()
    else:
        month_name = m.group(1).lower()
        year = int(m.group(2))
    month = MONTH_MAP.get(month_name.lower())
    if not month:
        return None
    return (year, month)


def months_between(a: Tuple[int, int], b: Tuple[int, int]) -> int:
    """Return absolute month difference between two (year, month) tuples."""
    (ya, ma), (yb, mb) = a, b
    return abs((yb - ya) * 12 + (mb - ma))


def normalize_format(fmt: Optional[str]) -> Optional[str]:
    if not fmt:
        return None
    s = fmt.strip().lower()
    # Synonyms
    if any(k in s for k in ["blended", "hybrid"]):
        return "blended"
    if any(k in s for k in ["virtual", "live online", "online (live)", "online—live", "online - live", "online live", "online (synchronous)"]):
        return "virtual"
    if any(k in s for k in ["in-person", "in person", "on campus", "on-campus", "campus"]):
        return "in-person"
    # If ambiguous multiple, prefer blended
    if "online" in s and ("in-person" in s or "on campus" in s):
        return "blended"
    return s  # fallback: return raw lowered text


def any_virtual_or_blended(programs: List[ProgramItem]) -> bool:
    for p in programs:
        nf = normalize_format(p.format)
        if nf in {"virtual", "blended"}:
            return True
    return False


def collect_urls(*programs: ProgramItem) -> List[str]:
    urls: List[str] = []
    for p in programs:
        if p and p.urls:
            for u in p.urls:
                if isinstance(u, str) and u.strip():
                    urls.append(u.strip())
    # Deduplicate preserving order
    seen = set()
    deduped = []
    for u in urls:
        if u not in seen:
            deduped.append(u)
            seen.add(u)
    return deduped


# --------------------------------------------------------------------------- #
# Verification subroutines                                                    #
# --------------------------------------------------------------------------- #
async def verify_program(
    evaluator: Evaluator,
    parent_node,
    program: ProgramItem,
    category_tag: str,           # "leadership" | "strategy" | "elective"
    category_title: str,         # "Leadership" | "Strategy" | "Elective"
    node_prefix: str             # e.g., "leadership", "strategy", "elective"
) -> Dict[str, Any]:
    """
    Build verification nodes for a single program category and run checks.

    Returns a dict with parsed values for downstream global constraints:
      - parsed_cost: Optional[float]
      - start_ym: Optional[Tuple[int,int]]
    """
    # Parent node for this category
    cat_node = evaluator.add_parallel(
        id=f"{node_prefix}_program",
        desc=f"One program selected from the qualifying {category_title.lower()} programs list",
        parent=parent_node,
        critical=True
    )

    # Existence check (must have name and at least one URL)
    existence_ok = bool(program and program.name and program.name.strip()) and bool(program and program.urls and len(program.urls) > 0)
    evaluator.add_custom_node(
        result=existence_ok,
        id=f"{node_prefix}_program_provided",
        desc=f"{category_title} program is provided with official name and at least one source URL",
        parent=cat_node,
        critical=True
    )

    # Correctness block: all leaves critical
    correctness = evaluator.add_parallel(
        id=f"{node_prefix}_program_correctness",
        desc=f"The selected {category_title.lower()} program is correctly identified and its details are accurate",
        parent=cat_node,
        critical=True
    )

    p_name = program.name or ""
    p_urls = program.urls or []
    p_start = program.start_date or ""
    p_format = program.format or ""
    p_cost = program.cost or ""
    claimed_category = (program.category or "").strip().lower()

    # 1) Official name matches page
    name_node = evaluator.add_leaf(
        id=f"{node_prefix}_name_match",
        desc=f"The official program name matches the claimed name",
        parent=correctness,
        critical=True
    )
    await evaluator.verify(
        claim=f"The official program name on the linked page is '{p_name}'. Allow for minor punctuation or capitalization differences but confirm it refers to the same HBS Executive Education program.",
        node=name_node,
        sources=p_urls,
        additional_instruction="Confirm the page shows the same program name; small punctuation, hyphenation, or case differences are acceptable if clearly the same program."
    )

    # 2) CME category qualification
    cat_node_leaf = evaluator.add_leaf(
        id=f"{node_prefix}_cme_category_qualified",
        desc=f"The program qualifies for the HBS Certificate of Management Excellence in the {category_title} category",
        parent=correctness,
        critical=True
    )
    await evaluator.verify(
        claim=f"This program counts toward the Harvard Business School Certificate of Management Excellence (CME) and is categorized under {category_title}.",
        node=cat_node_leaf,
        sources=p_urls,
        additional_instruction="Look for mentions of 'Certificate of Management Excellence' or 'CME' and the category label (Leadership/Strategy/Elective) on the program or CME-related HBS pages linked."
    )

    # 3) Start date accuracy and window
    start_node = evaluator.add_leaf(
        id=f"{node_prefix}_start_date_valid",
        desc=f"The program has a session starting as claimed and it is May 2026 or later",
        parent=correctness,
        critical=True
    )
    await evaluator.verify(
        claim=f"The program has a cohort/session that starts in {p_start}, and that start date is in May 2026 or later.",
        node=start_node,
        sources=p_urls,
        additional_instruction="Confirm that the page lists a session start aligned with the claimed month/year and that it is not earlier than May 2026."
    )

    # 4) Format accuracy
    format_node = evaluator.add_leaf(
        id=f"{node_prefix}_format_correct",
        desc=f"The program format matches the claimed delivery (In-Person, Virtual/Live Online, or Blended/Hybrid)",
        parent=correctness,
        critical=True
    )
    await evaluator.verify(
        claim=f"The program is offered in {p_format} format (accept synonyms: Virtual ≈ Live Online; Blended ≈ Hybrid; In-Person ≈ On-Campus).",
        node=format_node,
        sources=p_urls,
        additional_instruction="Check the delivery format(s) listed on the page. Accept synonyms and minor wording variations."
    )

    # 5) Cost accuracy
    cost_node = evaluator.add_leaf(
        id=f"{node_prefix}_cost_correct",
        desc=f"The program tuition/program fee amount matches the claimed cost",
        parent=correctness,
        critical=True
    )
    await evaluator.verify(
        claim=f"The tuition/program fee for the program is {p_cost}. Accept currency formatting variants like '$', 'USD', and thousands separators.",
        node=cost_node,
        sources=p_urls,
        additional_instruction="Look for 'Program Fee', 'Tuition', or similar. Minor formatting differences (commas, $ sign, USD) are acceptable if the numeric amount matches."
    )

    # Parsed info for downstream checks
    parsed_cost = parse_money_usd(p_cost)
    start_ym = parse_month_year(p_start)

    return {
        "parsed_cost": parsed_cost,
        "start_ym": start_ym,
        "normalized_format": normalize_format(p_format),
        "claimed_category": claimed_category,
        "urls": p_urls,
        "name": p_name
    }


# --------------------------------------------------------------------------- #
# Main evaluation                                                             #
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
    Evaluate an answer for selecting three HBS Executive Education programs to satisfy the
    Certificate of Management Excellence constraints and provide the correct contact email.
    """
    # Initialize evaluator (root strategy sequential per rubric)
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.SEQUENTIAL,
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

    # Extract structured selection
    extracted = await evaluator.extract(
        prompt=prompt_extract_programs(),
        template_class=CMESelectionExtraction,
        extraction_name="cme_program_selection",
    )

    # Safeguard empty items
    leadership = extracted.leadership_program or ProgramItem()
    strategy = extracted.strategy_program or ProgramItem()
    elective = extracted.elective_program or ProgramItem()
    contact_email = (extracted.contact_email or "").strip()

    # 1) ProgramSelection (critical, parallel)
    program_sel = evaluator.add_parallel(
        id="program_selection",
        desc="Three specific programs correctly identified and satisfy category and format requirements",
        parent=root,
        critical=True
    )

    # 1.a) CategoryComposition (critical, parallel)
    cat_comp = evaluator.add_parallel(
        id="category_composition",
        desc="Program selection satisfies category requirements: exactly one leadership program, one strategy program, and one elective program from the CME qualifying lists",
        parent=program_sel,
        critical=True
    )

    # Verify each category program with detailed checks
    leadership_res = await verify_program(evaluator, cat_comp, leadership, "leadership", "Leadership", "leadership")
    strategy_res = await verify_program(evaluator, cat_comp, strategy, "strategy", "Strategy", "strategy")
    elective_res = await verify_program(evaluator, cat_comp, elective, "elective", "Elective", "elective")

    # 1.b) FormatRequirement (critical leaf): At least one Virtual/Blended
    fmt_req = evaluator.add_leaf(
        id="format_requirement",
        desc="At least one of the three selected programs is offered in Virtual or Blended format",
        parent=program_sel,
        critical=True
    )
    all_urls = collect_urls(
        leadership if leadership else ProgramItem(),
        strategy if strategy else ProgramItem(),
        elective if elective else ProgramItem(),
    )
    await evaluator.verify(
        claim=(
            f"Among these three programs — '{leadership_res.get('name','')}', '{strategy_res.get('name','')}', "
            f"and '{elective_res.get('name','')}' — at least one is offered in a Virtual (a.k.a. 'Live Online') "
            f"or Blended/Hybrid delivery format per the linked official pages."
        ),
        node=fmt_req,
        sources=all_urls,
        additional_instruction="Treat 'Virtual' ≈ 'Live Online' and 'Blended' ≈ 'Hybrid'. If any one of the three pages lists such a format, this check should pass."
    )

    # 2) FinancialTimingConstraints (critical, parallel)
    fin_time = evaluator.add_parallel(
        id="financial_timing_constraints",
        desc="The selected programs satisfy both budget and timeline constraints",
        parent=root,
        critical=True
    )

    # Compute budget totals
    l_cost = leadership_res.get("parsed_cost")
    s_cost = strategy_res.get("parsed_cost")
    e_cost = elective_res.get("parsed_cost")
    costs_available = all(v is not None for v in [l_cost, s_cost, e_cost])
    total_cost = (l_cost or 0.0) + (s_cost or 0.0) + (e_cost or 0.0)
    evaluator.add_custom_info(
        {
            "leadership_cost": l_cost,
            "strategy_cost": s_cost,
            "elective_cost": e_cost,
            "total_cost": total_cost,
            "budget_limit": BUDGET_LIMIT,
            "all_costs_present": costs_available
        },
        info_type="budget_calc",
        info_name="budget_calculation"
    )
    evaluator.add_custom_node(
        result=(costs_available and total_cost <= BUDGET_LIMIT),
        id="budget_limit",
        desc="The total cost of the three selected programs does not exceed $45,000",
        parent=fin_time,
        critical=True
    )

    # Timeline compliance: start dates
    l_start = leadership_res.get("start_ym")
    s_start = strategy_res.get("start_ym")
    e_start = elective_res.get("start_ym")
    starts = [x for x in [l_start, s_start, e_start] if x is not None]
    all_starts_present = len(starts) == 3

    min_allowed = (MIN_START_YEAR, MIN_START_MONTH)
    all_starts_after_min = all(all([ym[0] > MIN_START_YEAR]) or (ym[0] == MIN_START_YEAR and ym[1] >= MIN_START_MONTH) for ym in starts) if starts else False

    # Check 36-month window across the three starts (use earliest and latest)
    within_window = False
    if len(starts) == 3:
        earliest = min(starts)
        latest = max(starts)
        within_window = months_between(earliest, latest) <= TIMELINE_MONTH_WINDOW

    evaluator.add_custom_info(
        {
            "leadership_start": l_start,
            "strategy_start": s_start,
            "elective_start": e_start,
            "min_allowed_start": min_allowed,
            "all_starts_present": all_starts_present,
            "all_starts_after_or_equal_min": all_starts_after_min,
            "window_months": TIMELINE_MONTH_WINDOW,
            "earliest_to_latest_months": (months_between(min(starts), max(starts)) if len(starts) == 3 else None),
            "within_36_months_window": within_window
        },
        info_type="timeline_calc",
        info_name="timeline_calculation"
    )
    evaluator.add_custom_node(
        result=(all_starts_present and all_starts_after_min and within_window),
        id="timeline_compliance",
        desc="All three programs start in May 2026 or later and can be completed within 36 consecutive months from the first program's start date",
        parent=fin_time,
        critical=True
    )

    # 3) ApplicationContact (critical leaf)
    app_contact = evaluator.add_leaf(
        id="application_contact",
        desc="The correct email address to initiate the Certificate of Management Excellence application is provided: executive_education@hbs.edu",
        parent=root,
        critical=True
    )
    await evaluator.verify(
        claim=f"The provided email address to initiate the Certificate of Management Excellence application is '{contact_email}', and it exactly matches 'executive_education@hbs.edu' (case-insensitive).",
        node=app_contact,
        additional_instruction="Judge true only if the provided email equals executive_education@hbs.edu ignoring case; ignore surrounding spaces or a leading 'mailto:'."
    )

    # Record constraint info for convenience
    evaluator.add_custom_info(
        {
            "constraints": {
                "budget_limit_usd": BUDGET_LIMIT,
                "min_start": {"year": MIN_START_YEAR, "month": MIN_START_MONTH},
                "timeline_month_window": TIMELINE_MONTH_WINDOW,
                "format_requirement": "At least one program must be Virtual/Live Online or Blended/Hybrid"
            }
        },
        info_type="constraints",
        info_name="task_constraints"
    )

    return evaluator.get_summary()