import asyncio
import logging
import re
from typing import Optional, List, Dict, Any, Tuple

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy

# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "highest_teacher_salary_2025_26_four_districts"
TASK_DESCRIPTION = (
    "Among Frisco Independent School District (Texas), Prince William County Public Schools (Virginia), "
    "Gwinnett County Public Schools (Georgia), and Duval County Public Schools (Florida), which district offers "
    "the highest starting salary for a newly hired teacher with a Master's degree and zero years of prior teaching "
    "experience for the 2025-26 school year? Please provide the specific salary amount for each district and identify "
    "which district has the highest starting salary."
)

# Mapping for friendly names and short keys
DISTRICT_KEYS = ["frisco", "pwcs", "gcps", "duval"]
DISTRICT_FRIENDLY = {
    "frisco": "Frisco ISD",
    "pwcs": "Prince William County Public Schools",
    "gcps": "Gwinnett County Public Schools",
    "duval": "Duval County Public Schools",
}

DISTRICT_DESCRIPTIONS = {
    "frisco": "Correctly identify Frisco ISD's 2025-26 starting salary for a teacher with Master's degree and 0 years experience (base salary plus any Master's degree stipend)",
    "pwcs": "Correctly identify Prince William County Public Schools' 2025-26 starting salary for a teacher with Master's degree (MA column) and 0 years experience (Step 1)",
    "gcps": "Correctly identify Gwinnett County Public Schools' 2025-26 starting salary for a teacher with Master's degree (Level 2) and 0-2 years external experience (Performance Step 0)",
    "duval": "Correctly identify Duval County Public Schools' 2025-26 starting salary for a teacher with Master's degree and 0-14 years experience (Performance Pay base plus Master's supplement of $1,150)",
}

# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class DistrictSalary(BaseModel):
    total: Optional[str] = None  # Total starting salary for MA + 0 years as presented in the answer
    base: Optional[str] = None   # If the answer decomposes base + stipend, capture base amount
    masters_stipend: Optional[str] = None  # If present in the answer
    education_level: Optional[str] = None  # e.g., "Master's", "MA", "Level 2"
    step_or_experience: Optional[str] = None  # e.g., "0 years", "Step 0", "Step 1", "Performance Step 0"
    school_year: Optional[str] = None  # e.g., "2025-26", "2025-2026"
    sources: List[str] = Field(default_factory=list)  # URLs cited for this district


class SalaryExtraction(BaseModel):
    frisco: Optional[DistrictSalary] = None
    pwcs: Optional[DistrictSalary] = None
    gcps: Optional[DistrictSalary] = None
    duval: Optional[DistrictSalary] = None
    highest_district: Optional[str] = None  # The district (or districts) the answer claims is/are highest
    highest_amount: Optional[str] = None    # The highest salary amount as stated in the answer (optional)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_salary_info() -> str:
    return """
    Extract, exactly as presented in the answer, the starting salary information for the 2025-26 school year for a newly hired teacher with a Master's degree and 0 years of prior teaching experience for the following four districts:
    - Frisco ISD (Texas)
    - Prince William County Public Schools (Virginia)
    - Gwinnett County Public Schools (Georgia)
    - Duval County Public Schools (Florida)
    
    For each district, extract the following fields as they appear in the answer:
    - total: The total starting salary amount the answer reports for a Master's degree teacher with 0 years for 2025-26. Keep the original formatting (e.g., include $ and commas if present).
    - base: If the answer breaks the number down, extract the base salary component separately (otherwise null).
    - masters_stipend: If the answer mentions a Master's degree stipend/supplement explicitly, extract it (otherwise null).
    - education_level: The degree/category label used in the answer, if mentioned (e.g., "Master's", "MA", "Level 2", "Advanced degree").
    - step_or_experience: The step/experience wording used in the answer for this number (e.g., "0 years", "Step 0", "Step 1", "Performance Step 0").
    - school_year: The school year string mentioned for this salary (should be "2025-26" or similar), if present in the answer.
    - sources: All URLs cited in the answer that are used to support this district’s salary.
    
    Also extract the overall identification of the highest salary among the four districts, as stated in the answer:
    - highest_district: The district name(s) the answer says has/have the highest starting salary (use the exact text provided in the answer; if a tie is claimed, include all districts as written).
    - highest_amount: The amount mentioned as the highest (if any).
    
    Return a JSON object with top-level keys: frisco, pwcs, gcps, duval, highest_district, highest_amount.
    Each of frisco, pwcs, gcps, duval must be an object with the fields above (or null if the district is missing in the answer).
    Make sure URLs in 'sources' are actual links explicitly present in the answer. Do not invent or infer URLs.
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def _parse_amount(amount_str: Optional[str]) -> Optional[float]:
    """
    Parse a currency-like string into a numeric value.
    Accepts formats like "$62,345", "62,345", "62345", "62k" (interpreted as 62000).
    Returns None if cannot parse.
    """
    if not amount_str:
        return None
    s = amount_str.strip().lower()
    # Handle shorthand like "62k"
    m = re.fullmatch(r"\s*([0-9]+(\.[0-9]+)?)\s*k\s*", s)
    if m:
        try:
            return float(m.group(1)) * 1000.0
        except Exception:
            return None
    # Remove everything except digits and dot
    cleaned = re.sub(r"[^0-9.]", "", s)
    if cleaned == "" or cleaned == ".":
        return None
    try:
        return float(cleaned)
    except Exception:
        return None


def _compute_total_numeric(d: Optional[DistrictSalary]) -> Tuple[Optional[float], Dict[str, Optional[float]]]:
    """
    Compute a numeric total for a district using the 'total' field if parseable;
    otherwise try base + masters_stipend if both parseable.
    Returns (numeric_total, {"total": v, "base": v, "masters_stipend": v})
    """
    parts = {
        "total": _parse_amount(d.total if d else None),
        "base": _parse_amount(d.base if d else None),
        "masters_stipend": _parse_amount(d.masters_stipend if d else None),
    }
    if parts["total"] is not None:
        return parts["total"], parts
    if parts["base"] is not None and parts["masters_stipend"] is not None:
        return parts["base"] + parts["masters_stipend"], parts
    return None, parts


def _district_category_instruction(dist_key: str) -> str:
    """
    Additional instruction tailored to each district for verifying category/time context.
    """
    common = (
        "Treat 'Master's' equivalently with labels such as 'MA', 'M.Ed.', or in GCPS 'Level 2'. "
        "Treat 0 years experience as the first step on the schedule, which may be labeled 'Step 0', 'Step 1', or 'Performance Step 0 (PS0)' depending on the district. "
        "Verify that the page(s) are for the 2025-26 school year (or explicitly labeled as such), a teacher position, and the specified degree/experience category."
    )
    if dist_key == "frisco":
        return (
            common
            + " For Frisco ISD, note that a Master's stipend might be shown separately from the base teacher salary schedule; "
              "accept either a Master's-specific column at Step 0/1 OR a separate stipend page, as long as it clearly pertains to 2025-26."
        )
    if dist_key == "pwcs":
        return (
            common
            + " For Prince William County Public Schools (PWCS), Master's may be labeled MA column; 0 years is often Step 1."
        )
    if dist_key == "gcps":
        return (
            common
            + " For Gwinnett County Public Schools (GCPS), Master's is typically 'Level 2'; 0-2 years external experience generally maps to 'Performance Step 0'."
        )
    if dist_key == "duval":
        return (
            common
            + " For Duval County Public Schools (DCPS), the base may be on a performance pay schedule and the Master's supplement (often $1,150) may be shown separately."
        )
    return common


def _amount_support_instruction(dist_key: str) -> str:
    """
    Additional instruction for verifying the specific salary amount.
    """
    return (
        "Check whether the specific starting salary figure for a newly hired teacher with a Master's degree and 0 years of experience "
        "for the 2025-26 school year is explicitly shown on the page. Accept reasonable labeling variations (e.g., Master's/MA/Level 2; "
        "0 years may be labeled Step 0/1 or PS0). If a district splits base and Master's stipend on different pages, it is acceptable if a "
        "single page directly shows the Master's total for that step/experience; otherwise, this specific 'total' claim may not be supported by a single page."
    )


def _format_amount(a: Optional[str]) -> str:
    return a if a is not None else "N/A"


# --------------------------------------------------------------------------- #
# Verification subroutines                                                    #
# --------------------------------------------------------------------------- #
async def verify_single_district(
    evaluator: Evaluator,
    parent_node,
    dist_key: str,
    dist_info: Optional[DistrictSalary],
) -> None:
    """
    Build verification nodes for a single district.
    """
    friendly = DISTRICT_FRIENDLY[dist_key]
    desc_main = DISTRICT_DESCRIPTIONS[dist_key]

    # Parent node for this district (Parallel, non-critical to allow partial credit)
    district_node = evaluator.add_parallel(
        id=f"{dist_key}_salary",
        desc=desc_main,
        parent=parent_node,
        critical=False
    )

    # Existence check: must have a total salary reported and at least one source
    exists = bool(dist_info and dist_info.total and dist_info.total.strip())
    has_source = bool(dist_info and dist_info.sources and len(dist_info.sources) > 0)

    evaluator.add_custom_node(
        result=exists and has_source,
        id=f"{dist_key}_exists",
        desc=f"{friendly}: Answer provides a starting salary figure and at least one supporting source URL",
        parent=district_node,
        critical=True
    )

    # Category/time context supported by sources (critical)
    cat_node = evaluator.add_leaf(
        id=f"{dist_key}_category_supported",
        desc=f"{friendly}: The 2025-26 Master's (or equivalent) and 0-years/initial step category is present/supported by cited sources",
        parent=district_node,
        critical=True
    )
    category_claim = (
        f"This page pertains to the 2025-26 teacher salary information for {friendly} and includes the category for a teacher with a Master's degree "
        f"(or equivalent: MA, M.Ed., Level 2) at zero years of experience (first step, labeled Step 0/1 or PS0)."
    )
    await evaluator.verify(
        claim=category_claim,
        node=cat_node,
        sources=(dist_info.sources if dist_info else []),
        additional_instruction=_district_category_instruction(dist_key)
    )

    # Amount supported by at least one cited source (non-critical; some districts may only show base/stipend separately)
    amt_node = evaluator.add_leaf(
        id=f"{dist_key}_amount_supported",
        desc=f"{friendly}: The stated starting salary amount for MA, 0 years in 2025-26 is supported by a cited source",
        parent=district_node,
        critical=False
    )
    amt = dist_info.total if dist_info else None
    amount_claim = (
        f"The starting salary for a newly hired teacher with a Master's degree and 0 years of experience in the 2025-26 school year at {friendly} is {amt}."
    )
    await evaluator.verify(
        claim=amount_claim,
        node=amt_node,
        sources=(dist_info.sources if dist_info else []),
        additional_instruction=_amount_support_instruction(dist_key)
    )

    # If components are provided, verify them as well (non-critical partial credit)
    if dist_info and dist_info.base:
        base_node = evaluator.add_leaf(
            id=f"{dist_key}_base_supported",
            desc=f"{friendly}: The base starting salary component used for 2025-26 is supported by a cited source",
            parent=district_node,
            critical=False
        )
        base_claim = (
            f"The base starting salary component relevant to a newly hired teacher (initial step) for the 2025-26 school year at {friendly} is {dist_info.base}."
        )
        await evaluator.verify(
            claim=base_claim,
            node=base_node,
            sources=dist_info.sources,
            additional_instruction=(
                "Verify that this amount appears on the teacher salary schedule or compensation page for 2025-26, "
                "as the initial/base step amount. Minor labeling variations are acceptable."
            )
        )

    if dist_info and dist_info.masters_stipend:
        stipend_node = evaluator.add_leaf(
            id=f"{dist_key}_masters_stipend_supported",
            desc=f"{friendly}: The Master's degree stipend/supplement amount for 2025-26 is supported by a cited source",
            parent=district_node,
            critical=False
        )
        stipend_claim = (
            f"The Master's degree stipend or supplement for the 2025-26 school year at {friendly} is {dist_info.masters_stipend}."
        )
        await evaluator.verify(
            claim=stipend_claim,
            node=stipend_node,
            sources=dist_info.sources,
            additional_instruction=(
                "Verify that this amount appears on an official page (e.g., HR/compensation/benefits) and explicitly corresponds to a Master's degree supplement for 2025-26."
            )
        )

    # Math consistency check (non-critical): base + stipend == total
    if dist_info and dist_info.total and (dist_info.base or dist_info.masters_stipend):
        total_num, parts = _compute_total_numeric(dist_info)
        computed_ok = False
        if parts["total"] is not None:
            if parts["base"] is not None and parts["masters_stipend"] is not None:
                computed_ok = abs(parts["base"] + parts["masters_stipend"] - parts["total"]) < 0.5
            else:
                # If only 'total' present, we cannot judge math consistency
                computed_ok = True  # Consider consistent by default if components are incomplete
        evaluator.add_custom_node(
            result=computed_ok,
            id=f"{dist_key}_math_consistent",
            desc=f"{friendly}: The presented total equals the sum of base and Master's stipend when both are provided",
            parent=district_node,
            critical=False
        )


async def verify_highest_identification(
    evaluator: Evaluator,
    parent_node,
    extracted: SalaryExtraction
) -> None:
    """
    Verify the correctness of the identified highest salary based on the four extracted amounts.
    Implemented as a critical sequential group with a primary logical check using simple verification.
    """
    highest_group = evaluator.add_sequential(
        id="correct_highest_identification",
        desc="Based on the four salary figures, correctly identify which district offers the highest total starting salary",
        parent=parent_node,
        critical=True
    )

    # Leaf: Check that the answer actually made an identification (existence)
    present_leaf = evaluator.add_custom_node(
        result=bool(extracted.highest_district and extracted.highest_district.strip()),
        id="highest_identification_present",
        desc="The answer explicitly identifies which district has the highest starting salary (or a tie)",
        parent=highest_group,
        critical=True
    )

    # Leaf: Logical correctness check (critical) using simple verification
    logic_leaf = evaluator.add_leaf(
        id="highest_identification_correct",
        desc="The identified highest district(s) is/are correct given the four stated salary amounts",
        parent=highest_group,
        critical=True
    )

    frisco_amt = _format_amount(extracted.frisco.total if extracted.frisco else None)
    pwcs_amt = _format_amount(extracted.pwcs.total if extracted.pwcs else None)
    gcps_amt = _format_amount(extracted.gcps.total if extracted.gcps else None)
    duval_amt = _format_amount(extracted.duval.total if extracted.duval else None)
    claimed_highest = extracted.highest_district if extracted.highest_district else "N/A"
    claimed_highest_amt = extracted.highest_amount if extracted.highest_amount else "N/A"

    logic_claim = (
        "Consider only the four amounts stated in the answer for the 2025-26 starting salary for a newly hired teacher with a Master's degree and 0 years:\n"
        f"- Frisco ISD: {frisco_amt}\n"
        f"- Prince William County Public Schools: {pwcs_amt}\n"
        f"- Gwinnett County Public Schools: {gcps_amt}\n"
        f"- Duval County Public Schools: {duval_amt}\n\n"
        f"The answer claims the highest is: {claimed_highest} (amount: {claimed_highest_amt}). "
        "Verify whether this identification is correct given the four amounts above. "
        "If two or more districts are tied for the highest amount, the claim is only correct if it acknowledges all tied districts. "
        "Ignore any amounts not listed above and rely strictly on the four provided figures."
    )

    await evaluator.verify(
        claim=logic_claim,
        node=logic_leaf,
        sources=None,
        additional_instruction=(
            "Treat currency formatting leniently (ignore $ and commas) and compare numeric values. "
            "If any listed amount is 'N/A' or missing, treat the identification as not verifiable/correct."
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
) -> Dict[str, Any]:
    """
    Evaluate an answer for the task:
    Identify which of four specified districts offers the highest 2025-26 starting salary for a Master's degree teacher with 0 years,
    verify each district's salary (with sources), and verify the correctness of the highest identification.
    """
    # Initialize evaluator (root as PARALLEL to avoid over-strict sequential precondition skipping)
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

    # Extraction
    extracted: SalaryExtraction = await evaluator.extract(
        prompt=prompt_extract_salary_info(),
        template_class=SalaryExtraction,
        extraction_name="salary_extraction"
    )

    # Add debug/custom info for downstream inspection
    debug_amounts = {}
    for k in DISTRICT_KEYS:
        d: Optional[DistrictSalary] = getattr(extracted, k)
        total_num, parts = _compute_total_numeric(d)
        debug_amounts[k] = {
            "friendly": DISTRICT_FRIENDLY[k],
            "total_str": d.total if d else None,
            "base_str": d.base if d else None,
            "masters_stipend_str": d.masters_stipend if d else None,
            "numeric_total": total_num,
            "numeric_parts": parts
        }
    evaluator.add_custom_info(debug_amounts, info_type="extracted_numeric_debug", info_name="numeric_interpretation")

    # Build district verification branches
    # Frisco
    await verify_single_district(evaluator, root, "frisco", extracted.frisco)
    # PWCS
    await verify_single_district(evaluator, root, "pwcs", extracted.pwcs)
    # GCPS
    await verify_single_district(evaluator, root, "gcps", extracted.gcps)
    # Duval
    await verify_single_district(evaluator, root, "duval", extracted.duval)

    # Highest identification check (critical group)
    await verify_highest_identification(evaluator, root, extracted)

    # Return evaluation summary
    return evaluator.get_summary()