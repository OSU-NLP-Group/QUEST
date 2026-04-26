import asyncio
import logging
import re
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys


# -----------------------------------------------------------------------------
# Task constants
# -----------------------------------------------------------------------------
TASK_ID = "largest_enrollment_2024_2025"
TASK_DESCRIPTION = (
    "Among the following three educational institutions—Seton Hall University, Purdue University, "
    "and Jefferson County Public Schools (JCPS)—which one has the largest total student enrollment "
    "based on the most recent 2024-2025 academic year data? Provide the specific enrollment number "
    "and include at least one reference URL that verifies this information."
)

CANONICAL_INSTITUTIONS = [
    "Seton Hall University",
    "Purdue University",
    "Jefferson County Public Schools (JCPS)",
]


# -----------------------------------------------------------------------------
# Data models for extraction
# -----------------------------------------------------------------------------
class InstitutionEntry(BaseModel):
    name: Optional[str] = None  # Expected to be one of the three, but allow None
    enrollment: Optional[str] = None  # Keep as string to be flexible (e.g., "49,639", "about 50k")
    year_or_term: Optional[str] = None  # e.g., "2024–2025", "Fall 2024", "Most recent"
    urls: List[str] = Field(default_factory=list)  # All URLs associated with this institution


class FullExtraction(BaseModel):
    # The selected (claimed largest) institution details
    selected_institution: Optional[str] = None
    selected_enrollment: Optional[str] = None
    selected_year_or_term: Optional[str] = None
    selected_urls: List[str] = Field(default_factory=list)

    # Any institution entries mentioned in the answer (from the three)
    institutions: List[InstitutionEntry] = Field(default_factory=list)


# -----------------------------------------------------------------------------
# Extraction prompt builders
# -----------------------------------------------------------------------------
def prompt_extract_full() -> str:
    return """
    Extract the final answer's chosen (claimed largest) institution among exactly these three:
    - Seton Hall University
    - Purdue University
    - Jefferson County Public Schools (JCPS)

    1) selected_institution: The institution the answer claims has the largest total student enrollment among the three.
       The value should be one of: "Seton Hall University", "Purdue University", or "Jefferson County Public Schools (JCPS)".
       If the answer uses variants (e.g., "Seton Hall"), normalize it to the canonical name above.
    2) selected_enrollment: The specific student enrollment value stated for the chosen institution, as a string exactly as written in the answer (e.g., "49,639", "about 50,000", "97k+").
    3) selected_year_or_term: Any explicit academic year or timeframe tied to the selected enrollment (e.g., "2024–2025", "Fall 2024", "Most recent"). If not specified, return null.
    4) selected_urls: All URLs in the answer that are cited as references supporting the chosen institution's enrollment figure (do not invent URLs).

    Also extract any details provided for the three institutions (even if not selected) under `institutions`:
    - name: Use the canonical name if mentioned (normalize variations):
        • "Seton Hall University"
        • "Purdue University"
        • "Jefferson County Public Schools (JCPS)"
    - enrollment: The enrollment value string associated with that institution in the answer (exactly as written), if any.
    - year_or_term: The linked timeframe (e.g., "2024–2025", "Most recent"), if any.
    - urls: All URLs that the answer ties to that institution's enrollment or enrollment context.

    Rules:
    - Only extract information explicitly present in the answer text.
    - For URLs, extract actual URLs (including those in markdown link formats).
    - If any field is not mentioned, set it to null or an empty list as appropriate.
    - Deduplicate URLs when obvious duplicates occur.
    """


# -----------------------------------------------------------------------------
# Helper utilities
# -----------------------------------------------------------------------------
def normalize_institution_name(name: Optional[str]) -> Optional[str]:
    if not name:
        return None
    n = name.strip().lower()
    if "purdue" in n:
        return "Purdue University"
    if "seton hall" in n:
        return "Seton Hall University"
    if "jefferson county public schools" in n or "jcps" in n:
        return "Jefferson County Public Schools (JCPS)"
    return None


def extract_numeric_from_text(text: Optional[str]) -> Optional[int]:
    """
    Extract a best-effort integer enrollment from a free-form string.
    Handles formats like "49,639", "about 50k", "50 thousand", "1.2 million", etc.
    Returns None if no plausible numeric value can be parsed.
    """
    if not text:
        return None
    s = text.strip().lower()

    # Try capturing forms like '50k', '50 k', '50 thousand', '1.2m', '1.2 million'
    # First, check for explicit thousand/million suffixes
    km = re.search(r'(\d{1,3}(?:,\d{3})+|\d+(?:\.\d+)?)(\s*[km]|(?:\s*thousand)|(?:\s*million))\b', s)
    if km:
        num_str = km.group(1)
        suffix = km.group(2).strip()
        try:
            val = float(num_str.replace(",", ""))
        except ValueError:
            val = None
        if val is not None:
            if suffix in ("k", " thousand") or "thousand" in suffix:
                return int(round(val * 1_000))
            if suffix in ("m",) or "million" in suffix:
                return int(round(val * 1_000_000))

    # Otherwise pick the first big integer-like number in the string
    m = re.search(r'(\d{1,3}(?:,\d{3})+|\d+)', s)
    if m:
        try:
            return int(m.group(1).replace(",", ""))
        except ValueError:
            return None
    return None


def unique_urls(urls: List[str]) -> List[str]:
    seen = set()
    result = []
    for u in urls:
        if not u:
            continue
        u = u.strip()
        if u and u not in seen:
            seen.add(u)
            result.append(u)
    return result


def build_institution_numeric_map(
    extraction: FullExtraction,
) -> Dict[str, Optional[int]]:
    """
    Build a mapping from canonical institution name -> numeric enrollment (int or None),
    using information from the detailed 'institutions' list, and falling back to the selected
    enrollment if the selected institution is missing from that list.
    """
    mapping: Dict[str, Optional[int]] = {c: None for c in CANONICAL_INSTITUTIONS}

    # Populate from institutions list
    for entry in extraction.institutions:
        canon = normalize_institution_name(entry.name)
        if canon and canon in mapping and mapping[canon] is None:
            mapping[canon] = extract_numeric_from_text(entry.enrollment)

    # Ensure the selected institution is filled if missing
    selected_canon = normalize_institution_name(extraction.selected_institution)
    if selected_canon and mapping.get(selected_canon) is None:
        mapping[selected_canon] = extract_numeric_from_text(extraction.selected_enrollment)

    return mapping


def compute_largest_correct(
    extraction: FullExtraction,
) -> Tuple[bool, Dict[str, Optional[int]]]:
    """
    Determine whether the selected institution is indeed (one of) the largest by numeric enrollment
    among the three canonical institutions, based solely on the enrollment numbers present in the answer.
    Returns (is_correct, numeric_map).
    """
    numeric_map = build_institution_numeric_map(extraction)

    # We need all three numeric values to confidently decide "largest among three".
    values = [numeric_map.get(c) for c in CANONICAL_INSTITUTIONS]
    if any(v is None for v in values):
        return False, numeric_map

    selected_canon = normalize_institution_name(extraction.selected_institution)
    if not selected_canon:
        return False, numeric_map

    max_val = max(values)  # type: ignore
    winners = [c for c in CANONICAL_INSTITUTIONS if numeric_map[c] == max_val]
    return selected_canon in winners, numeric_map


# -----------------------------------------------------------------------------
# Verification builder
# -----------------------------------------------------------------------------
async def build_and_verify_nodes(evaluator: Evaluator, extraction: FullExtraction) -> None:
    """
    Construct the verification tree according to the rubric and execute verifications.
    """
    # Create the top-level task node as critical (as per rubric)
    main_node = evaluator.add_parallel(
        id="Largest_Enrollment_Institution_Identification",
        desc=(
            "Determine which of the three specified institutions has the largest total student enrollment "
            "using 2024–2025 academic year data or the most recent available, and provide the verified "
            "enrollment number with a reference URL."
        ),
        parent=evaluator.root,
        critical=True,
    )

    # Normalize selected institution and URLs
    selected_canon = normalize_institution_name(extraction.selected_institution)
    selected_enrollment_str = extraction.selected_enrollment or None
    selected_urls = unique_urls(extraction.selected_urls or [])
    selected_year_term = extraction.selected_year_or_term or None

    # 1) Institution_Is_One_Of_Three (critical)
    institution_in_set = selected_canon in CANONICAL_INSTITUTIONS if selected_canon else False
    evaluator.add_custom_node(
        result=institution_in_set,
        id="Institution_Is_One_Of_Three",
        desc=(
            "The answer names an institution and it is exactly one of: Seton Hall University, "
            "Purdue University, or Jefferson County Public Schools (JCPS)."
        ),
        parent=main_node,
        critical=True,
    )

    # 2) Enrollment_Figure_Provided (critical): must contain a specific numeric value
    has_numeric_enrollment = extract_numeric_from_text(selected_enrollment_str) is not None
    evaluator.add_custom_node(
        result=bool(selected_enrollment_str) and has_numeric_enrollment,
        id="Enrollment_Figure_Provided",
        desc="The answer provides a specific numeric value for the identified institution’s total student enrollment.",
        parent=main_node,
        critical=True,
    )

    # Auxiliary: Reference URL presence (critical gating to ensure URL exists for verification)
    has_reference_url = len(selected_urls) > 0
    has_url_node = evaluator.add_custom_node(
        result=has_reference_url,
        id="Reference_URL_Present",
        desc="At least one reference URL is provided to support the enrollment figure.",
        parent=main_node,
        critical=True,
    )

    # 3) Valid_Reference_URL_Verifies_Enrollment (critical) — verify the enrollment number with the provided URLs
    verify_enrollment_node = evaluator.add_leaf(
        id="Valid_Reference_URL_Verifies_Enrollment",
        desc=(
            "The answer includes at least one credible reference URL that supports the stated enrollment figure "
            "(and the stated year/recency, if provided by the source)."
        ),
        parent=main_node,
        critical=True,
    )
    claim_enrollment = f"The total student enrollment for {selected_canon or 'the chosen institution'} is {selected_enrollment_str}."
    # If no URLs, this leaf will be auto-skipped due to the critical sibling "Reference_URL_Present" failing.
    await evaluator.verify(
        claim=claim_enrollment,
        node=verify_enrollment_node,
        sources=selected_urls if selected_urls else None,
        additional_instruction=(
            "Verify that the page supports the numeric enrollment claim for the named institution, "
            "allowing minor formatting or rounding differences (e.g., 49,639 vs 49.6k). "
            "The claim refers to total overall student enrollment for the entire institution/district."
        ),
    )

    # 4) Enrollment_Data_Is_2024_2025_Or_Most_Recent (critical)
    verify_year_node = evaluator.add_leaf(
        id="Enrollment_Data_Is_2024_2025_Or_Most_Recent",
        desc=(
            "The enrollment figure is explicitly tied to the 2024–2025 academic year, or the answer/source clearly "
            "indicates it is the most recent available enrollment data."
        ),
        parent=main_node,
        critical=True,
    )
    claim_year = (
        f"The cited source indicates that the enrollment figure for {selected_canon or 'the chosen institution'} "
        f"is for the 2024–2025 academic year or is explicitly described as the most recent available figure."
    )
    await evaluator.verify(
        claim=claim_year,
        node=verify_year_node,
        sources=selected_urls if selected_urls else None,
        additional_instruction=(
            "Accept '2024–2025', '2024-25', 'AY 2024–25', 'Fall 2024', 'as of 2024/2025', "
            "or clear language like 'current'/'most recent' indicating the figure is the latest available. "
            "If the page only references older years without indicating that it is the latest, do not support the claim."
        ),
    )

    # 5) Correct_Largest_Institution_Among_Three (critical) — computed from numbers present in the answer
    largest_correct, numeric_map = compute_largest_correct(extraction)
    evaluator.add_custom_node(
        result=largest_correct,
        id="Correct_Largest_Institution_Among_Three",
        desc="The answer identifies the institution with the largest total student enrollment among the three specified institutions.",
        parent=main_node,
        critical=True,
    )

    # Record some helpful info for debugging/analysis
    evaluator.add_custom_info(
        info={
            "selected_institution_normalized": selected_canon,
            "selected_enrollment_numeric": extract_numeric_from_text(selected_enrollment_str),
            "selected_year_or_term": selected_year_term,
            "selected_urls": selected_urls,
            "parsed_enrollment_map": numeric_map,
        },
        info_type="parsed_values",
        info_name="parsed_values_summary",
    )


# -----------------------------------------------------------------------------
# Main evaluation entry
# -----------------------------------------------------------------------------
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
    Evaluate an answer for the 'largest enrollment among three institutions' task.
    """
    evaluator = Evaluator()
    evaluator.initialize(
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

    # Extract structured info from the answer
    extraction = await evaluator.extract(
        prompt=prompt_extract_full(),
        template_class=FullExtraction,
        extraction_name="selection_and_details",
    )

    # Add ground-truth context (names only; no numeric ground truth to avoid knowledge leakage)
    evaluator.add_ground_truth(
        {
            "institutions_under_comparison": CANONICAL_INSTITUTIONS,
            "requirement": "Identify which has the largest total student enrollment using 2024–2025 or the most recent available data; provide the numeric figure and a verifying URL.",
        },
        gt_type="task_requirements",
    )

    # Build verification tree and run checks
    await build_and_verify_nodes(evaluator, extraction)

    # Return summary
    return evaluator.get_summary()