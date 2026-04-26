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
TASK_ID = "ce_requirements_professions_states"
TASK_DESCRIPTION = (
    "A career services coordinator at a university needs to compile information about continuing education (CE) "
    "requirements for various professional licenses to help students understand long-term professional development commitments. "
    "Based on publicly available state licensing board information, answer the following four questions about annual continuing education requirements:\n\n"
    "1. Among these five profession-state combinations: Illinois CPA, Illinois Attorney (practicing law), Illinois Registered Nurse (RN), Texas Attorney (practicing law), and Florida Registered Nurse (RN), which combination requires the highest number of annual continuing education hours?\n\n"
    "2. Which of these five combinations requires the lowest number of annual continuing education hours?\n\n"
    "3. If a professional holds both an Illinois Attorney license and a Florida RN license, what is their total annual continuing education requirement in hours?\n\n"
    "4. How many of these five profession-state combinations require more than 12 continuing education hours per year?\n\n"
    "For each answer, provide supporting reference URL(s) from official state licensing boards or regulatory sources."
)

# Canonical combination labels
CANONICAL_COMBOS = [
    "Illinois CPA",
    "Illinois Attorney (practicing law)",
    "Illinois Registered Nurse (RN)",
    "Texas Attorney (practicing law)",
    "Florida Registered Nurse (RN)",
]

# Keys for the five combinations (internal use)
COMBO_KEYS = ["il_cpa", "il_attorney", "il_rn", "tx_attorney", "fl_rn"]
KEY_TO_CANONICAL = {
    "il_cpa": "Illinois CPA",
    "il_attorney": "Illinois Attorney (practicing law)",
    "il_rn": "Illinois Registered Nurse (RN)",
    "tx_attorney": "Texas Attorney (practicing law)",
    "fl_rn": "Florida Registered Nurse (RN)",
}


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class CEItem(BaseModel):
    """Single profession-state combination CE info extracted from the answer."""
    name: Optional[str] = None
    annual_hours: Optional[str] = None
    cycle_hours: Optional[str] = None  # e.g., "24 hours", "30"
    cycle_length_years: Optional[str] = None  # e.g., "2 years", "biennial"
    sources: List[str] = Field(default_factory=list)


class CEAnswers(BaseModel):
    """Question-specific answer extraction."""
    # Q1
    q1_highest_name: Optional[str] = None
    q1_highest_hours: Optional[str] = None
    q1_sources: List[str] = Field(default_factory=list)

    # Q2
    q2_lowest_name: Optional[str] = None
    q2_lowest_hours: Optional[str] = None
    q2_sources: List[str] = Field(default_factory=list)

    # Q3
    q3_il_attorney_annual_hours: Optional[str] = None
    q3_fl_rn_annual_hours: Optional[str] = None
    q3_total_annual_hours: Optional[str] = None
    q3_il_attorney_sources: List[str] = Field(default_factory=list)
    q3_fl_rn_sources: List[str] = Field(default_factory=list)

    # Q4
    q4_count_above_12: Optional[str] = None
    q4_combos_above_12: List[str] = Field(default_factory=list)
    q4_sources: List[str] = Field(default_factory=list)


class CEExtraction(BaseModel):
    """Full extraction payload."""
    il_cpa: Optional[CEItem] = None
    il_attorney: Optional[CEItem] = None
    il_rn: Optional[CEItem] = None
    tx_attorney: Optional[CEItem] = None
    fl_rn: Optional[CEItem] = None
    answers: CEAnswers = Field(default_factory=CEAnswers)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_ce_data() -> str:
    return (
        "Extract the continuing education (CE) information exactly as presented in the answer for the following five "
        "profession-state combinations (use these exact canonical labels when populating the 'name' fields if the answer refers to them):\n"
        "- Illinois CPA\n"
        "- Illinois Attorney (practicing law)\n"
        "- Illinois Registered Nurse (RN)\n"
        "- Texas Attorney (practicing law)\n"
        "- Florida Registered Nurse (RN)\n\n"
        "For each combination, extract:\n"
        "1) name: The profession-state combo label used in the answer (normalize to the canonical label above if it's clearly the same combo).\n"
        "2) annual_hours: The annual CE hours stated in the answer (as a string); if only a multi-year cycle is given, you may copy the annual value if the answer itself computed it; otherwise leave this null.\n"
        "3) cycle_hours: The total CE hours per full cycle (as a string) stated in the answer, such as '24' or '30 hours'. If not mentioned, set null.\n"
        "4) cycle_length_years: The stated cycle length (as a string), such as '2 years', 'biennial', 'annual', etc. If not mentioned, set null.\n"
        "5) sources: All URLs in the answer that are clearly tied to that combination's CE requirement. Only extract actual URLs present in the answer.\n\n"
        "Then extract the four question-specific answers and their cited reference URLs:\n"
        "Q1: Among the five combinations, which requires the highest annual CE hours?\n"
        "- q1_highest_name: The identified combo name (string)\n"
        "- q1_highest_hours: The annual hours the answer claims for that combo (string)\n"
        "- q1_sources: URLs cited for Q1 (list)\n\n"
        "Q2: Which requires the lowest annual CE hours among the five?\n"
        "- q2_lowest_name: The identified combo name (string)\n"
        "- q2_lowest_hours: The annual hours the answer claims for that combo (string)\n"
        "- q2_sources: URLs cited for Q2 (list)\n\n"
        "Q3: If a professional holds both an Illinois Attorney license and a Florida RN license, what is the total annual CE?\n"
        "- q3_il_attorney_annual_hours: The annual CE hours the answer uses for IL Attorney (string)\n"
        "- q3_fl_rn_annual_hours: The annual CE hours the answer uses for FL RN (string)\n"
        "- q3_total_annual_hours: The sum stated in the answer (string)\n"
        "- q3_il_attorney_sources: URLs cited for IL Attorney CE in Q3 (list)\n"
        "- q3_fl_rn_sources: URLs cited for FL RN CE in Q3 (list)\n\n"
        "Q4: How many of these five combinations require more than 12 hours per year?\n"
        "- q4_count_above_12: The count stated in the answer (string)\n"
        "- q4_combos_above_12: The list of combo names the answer counted as >12 hours (list of strings)\n"
        "- q4_sources: URLs cited for Q4 (list)\n\n"
        "Rules:\n"
        "- Extract only what appears in the answer text; do not invent new numbers or URLs.\n"
        "- List all URLs exactly as they appear (plain or markdown), using full URLs. If a URL is missing a protocol, prepend http://.\n"
        "- If a field is missing in the answer, return null for that field or an empty list for URLs.\n"
    )


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def merge_sources(*lists: List[str]) -> List[str]:
    """Merge multiple URL lists, deduplicate, preserve order."""
    seen = set()
    result: List[str] = []
    for lst in lists:
        for url in lst or []:
            if url and url not in seen:
                seen.add(url)
                result.append(url)
    return result


def get_item_by_key(data: CEExtraction, key: str) -> Optional[CEItem]:
    return getattr(data, key, None)


def guess_key_from_name(name: Optional[str]) -> Optional[str]:
    """Fuzzy map a free-form combo name to one of our internal keys."""
    if not name:
        return None
    n = name.lower()
    if "illinois" in n and "cpa" in n:
        return "il_cpa"
    if "illinois" in n and ("attorney" in n or "lawyer" in n or "mcle" in n):
        return "il_attorney"
    if "illinois" in n and ("registered nurse" in n or "rn" in n):
        return "il_rn"
    if "texas" in n and ("attorney" in n or "lawyer" in n or "mcle" in n):
        return "tx_attorney"
    if "florida" in n and ("registered nurse" in n or "rn" in n):
        return "fl_rn"
    return None


def extract_first_number(text: Optional[str]) -> Optional[float]:
    """Extract the first numeric value from a string."""
    if not text:
        return None
    m = re.search(r"(\d+(?:\.\d+)?)", text)
    if not m:
        return None
    try:
        return float(m.group(1))
    except Exception:
        return None


def cycle_years_from_text(text: Optional[str]) -> Optional[float]:
    """Heuristically parse cycle length (years) from text."""
    if not text:
        return None
    t = text.lower()
    # direct numeric
    m = re.search(r"(\d+(?:\.\d+)?)\s*year", t)
    if m:
        try:
            return float(m.group(1))
        except Exception:
            pass
    # keywords
    if "biennial" in t or "biannual" in t or "every two years" in t or "two-year" in t or "2-year" in t:
        return 2.0
    if "annual" in t or "per year" in t or "yearly" in t:
        return 1.0
    if "triennial" in t or "every three years" in t or "three-year" in t or "3-year" in t:
        return 3.0
    return None


def describe_combo(item: Optional[CEItem], canonical_name: str) -> str:
    """Create a readable description line for a combo based on extracted fields."""
    if not item:
        return f"{canonical_name}: (no details provided in the answer)"
    parts = []
    annual = item.annual_hours or "not stated"
    parts.append(f"{canonical_name}: annual {annual}")
    if item.cycle_hours or item.cycle_length_years:
        cyc = item.cycle_hours or "?"
        yrs = item.cycle_length_years or "?"
        parts.append(f"(cycle: {cyc} over {yrs})")
    return " ".join(parts)


def collect_all_combo_descriptions(data: CEExtraction) -> List[str]:
    """Collect readable descriptions for all five combos (based on the answer)."""
    descs = []
    for key in COMBO_KEYS:
        item = get_item_by_key(data, key)
        descs.append(describe_combo(item, KEY_TO_CANONICAL[key]))
    return descs


# --------------------------------------------------------------------------- #
# Verification builders                                                       #
# --------------------------------------------------------------------------- #
async def build_question_1(
    evaluator: Evaluator,
    parent_node,
    data: CEExtraction,
) -> None:
    """Build verification nodes for Question 1: identify highest annual CE and provide supporting references."""
    q1_node = evaluator.add_parallel(
        id="Question_1_Identify_Highest_Annual_CE",
        desc="Identify which profession-state combination among the five specified requires the highest number of annual continuing education hours",
        parent=parent_node,
        critical=False,
    )

    # Q1 Correct Identification
    q1_correct_leaf = evaluator.add_leaf(
        id="Q1_Correct_Identification",
        desc="Correctly identifies the profession-state combination with the highest annual CE requirement by comparing calculated annual hours from state licensing requirements",
        parent=q1_node,
        critical=True,
    )

    combo_descs = collect_all_combo_descriptions(data)
    claimed_highest = data.answers.q1_highest_name or "(not stated)"
    claimed_hours = data.answers.q1_highest_hours or "(not stated)"
    q1_claim = (
        "Based solely on the numbers explicitly stated in the answer, determine the highest annual continuing education hours among:\n"
        f"- " + "\n- ".join(combo_descs) + "\n"
        f"The answer claims the highest is '{claimed_highest}' with '{claimed_hours}' hours per year. "
        "Verify whether this identification is logically correct using only the provided numbers. "
        "If the answer provided cycle totals and cycle length instead of annual hours, derive annual by dividing cycle hours by cycle length. "
        "If there is a tie, the identification is acceptable if the claimed combo is among the tied maximums."
    )
    await evaluator.verify(
        claim=q1_claim,
        node=q1_correct_leaf,
        additional_instruction="Use only the answer-provided numbers to rank annual CE hours; do not use external knowledge.",
    )

    # Q1 Supporting Reference (verify the hours for the identified highest combo with cited sources)
    q1_support_leaf = evaluator.add_leaf(
        id="Q1_Supporting_Reference",
        desc="Provides valid reference URL(s) from official state licensing board or regulatory source documenting the CE requirement",
        parent=q1_node,
        critical=True,
    )

    highest_key = guess_key_from_name(data.answers.q1_highest_name)
    highest_item = get_item_by_key(data, highest_key) if highest_key else None
    highest_annual = data.answers.q1_highest_hours or (highest_item.annual_hours if highest_item else None) or "(not stated)"
    sources = merge_sources(
        data.answers.q1_sources,
        highest_item.sources if highest_item else [],
    )

    ref_claim = (
        f"The annual continuing education requirement for {data.answers.q1_highest_name or 'the claimed highest combination'} "
        f"is {highest_annual} hours per year (or equivalent based on cycle). "
        "Verify that at least one of the cited official licensing board or regulatory URLs supports this requirement for the specified state and profession."
    )
    await evaluator.verify(
        claim=ref_claim,
        node=q1_support_leaf,
        sources=sources if sources else None,
        additional_instruction=(
            "Prefer official regulatory sources (state licensing boards, state supreme court MCLE boards, state nursing boards, etc.). "
            "If the source states multi-year cycle totals (e.g., 24 hours every 2 years), treat derivation to annual (12 per year) as valid support."
        ),
    )


async def build_question_2(
    evaluator: Evaluator,
    parent_node,
    data: CEExtraction,
) -> None:
    """Build verification nodes for Question 2: identify lowest annual CE and provide supporting references."""
    q2_node = evaluator.add_parallel(
        id="Question_2_Identify_Lowest_Annual_CE",
        desc="Identify which profession-state combination among the five specified requires the lowest number of annual continuing education hours",
        parent=parent_node,
        critical=False,
    )

    # Q2 Correct Identification
    q2_correct_leaf = evaluator.add_leaf(
        id="Q2_Correct_Identification",
        desc="Correctly identifies the profession-state combination with the lowest annual CE requirement by comparing calculated annual hours from state licensing requirements",
        parent=q2_node,
        critical=True,
    )

    combo_descs = collect_all_combo_descriptions(data)
    claimed_lowest = data.answers.q2_lowest_name or "(not stated)"
    claimed_hours = data.answers.q2_lowest_hours or "(not stated)"
    q2_claim = (
        "Based solely on the numbers explicitly stated in the answer, determine the lowest annual continuing education hours among:\n"
        f"- " + "\n- ".join(combo_descs) + "\n"
        f"The answer claims the lowest is '{claimed_lowest}' with '{claimed_hours}' hours per year. "
        "Verify whether this identification is logically correct using only the provided numbers. "
        "If the answer provided cycle totals and cycle length instead of annual hours, derive annual by dividing cycle hours by cycle length. "
        "If there is a tie for the minimum, the identification is acceptable if the claimed combo is among the tied minimums."
    )
    await evaluator.verify(
        claim=q2_claim,
        node=q2_correct_leaf,
        additional_instruction="Use only the answer-provided numbers to rank annual CE hours; do not use external knowledge.",
    )

    # Q2 Supporting Reference
    q2_support_leaf = evaluator.add_leaf(
        id="Q2_Supporting_Reference",
        desc="Provides valid reference URL(s) from official state licensing board or regulatory source documenting the CE requirement",
        parent=q2_node,
        critical=True,
    )

    lowest_key = guess_key_from_name(data.answers.q2_lowest_name)
    lowest_item = get_item_by_key(data, lowest_key) if lowest_key else None
    lowest_annual = data.answers.q2_lowest_hours or (lowest_item.annual_hours if lowest_item else None) or "(not stated)"
    sources = merge_sources(
        data.answers.q2_sources,
        lowest_item.sources if lowest_item else [],
    )

    ref_claim = (
        f"The annual continuing education requirement for {data.answers.q2_lowest_name or 'the claimed lowest combination'} "
        f"is {lowest_annual} hours per year (or equivalent based on cycle). "
        "Verify that at least one of the cited official licensing board or regulatory URLs supports this requirement for the specified state and profession."
    )
    await evaluator.verify(
        claim=ref_claim,
        node=q2_support_leaf,
        sources=sources if sources else None,
        additional_instruction=(
            "Prefer official regulatory sources (state licensing boards, state supreme court MCLE boards, state nursing boards, etc.). "
            "If the source states multi-year cycle totals (e.g., X hours every Y years), treat derivation to annual (X/Y per year) as valid support."
        ),
    )


async def build_question_3(
    evaluator: Evaluator,
    parent_node,
    data: CEExtraction,
) -> None:
    """Build verification nodes for Question 3: combined total annual CE for IL Attorney + FL RN, with references."""
    q3_node = evaluator.add_parallel(
        id="Question_3_Calculate_Combined_Total",
        desc="Calculate the total annual continuing education requirement for an individual holding both Illinois Attorney and Florida RN licenses",
        parent=parent_node,
        critical=False,
    )

    # Q3 Calculation (Sequential)
    calc_node = evaluator.add_sequential(
        id="Q3_Calculation",
        desc="Correctly performs the calculation steps to determine the combined annual CE requirement",
        parent=q3_node,
        critical=True,
    )

    # Q3 Illinois Attorney annual hours (verified via references)
    q3_il_att_leaf = evaluator.add_leaf(
        id="Q3_Illinois_Attorney_Annual_Hours",
        desc="Correctly determines the annual CE hours required for Illinois Attorney license (from biennial requirement divided by 2)",
        parent=calc_node,
        critical=True,
    )

    il_att_item = data.il_attorney or CEItem()
    il_att_sources = merge_sources(
        data.answers.q3_il_attorney_sources,
        il_att_item.sources,
    )
    il_att_hours_str = data.answers.q3_il_attorney_annual_hours or il_att_item.annual_hours or "(not stated)"
    il_att_claim = (
        f"For Illinois Attorney (practicing law), the annual CE requirement used in the answer is {il_att_hours_str} hours per year. "
        "Verify this using the cited official source(s). If only a biennial cycle total is shown (e.g., 30 hours every 2 years), treat derivation to annual (e.g., 15/year) as valid."
    )
    await evaluator.verify(
        claim=il_att_claim,
        node=q3_il_att_leaf,
        sources=il_att_sources if il_att_sources else None,
        additional_instruction=(
            "Confirm the Illinois MCLE requirement from official sources (e.g., Illinois MCLE Board). "
            "If only biennial totals are provided, accept annual derived by dividing by the cycle length."
        ),
    )

    # Q3 Florida RN annual hours (verified via references)
    q3_fl_rn_leaf = evaluator.add_leaf(
        id="Q3_Florida_RN_Annual_Hours",
        desc="Correctly determines the annual CE hours required for Florida RN license (from biennial requirement divided by 2)",
        parent=calc_node,
        critical=True,
    )

    fl_rn_item = data.fl_rn or CEItem()
    fl_rn_sources = merge_sources(
        data.answers.q3_fl_rn_sources,
        fl_rn_item.sources,
    )
    fl_rn_hours_str = data.answers.q3_fl_rn_annual_hours or fl_rn_item.annual_hours or "(not stated)"
    fl_rn_claim = (
        f"For Florida Registered Nurse (RN), the annual CE requirement used in the answer is {fl_rn_hours_str} hours per year. "
        "Verify this using the cited official source(s). If only a biennial cycle total is shown (e.g., X hours every 2 years), treat derivation to annual (X/2 per year) as valid."
    )
    await evaluator.verify(
        claim=fl_rn_claim,
        node=q3_fl_rn_leaf,
        sources=fl_rn_sources if fl_rn_sources else None,
        additional_instruction=(
            "Confirm from official sources (e.g., Florida Board of Nursing). "
            "If only biennial totals are provided, accept annual derived by dividing by the cycle length."
        ),
    )

    # Q3 Sum Calculation (arithmetic check)
    q3_sum_leaf = evaluator.add_leaf(
        id="Q3_Sum_Calculation",
        desc="Correctly calculates the sum of Illinois Attorney and Florida RN annual CE hours",
        parent=calc_node,
        critical=True,
    )

    # Arithmetic simple verification using the numbers from the answer
    il_val = extract_first_number(il_att_hours_str)
    fl_val = extract_first_number(fl_rn_hours_str)
    total_str = data.answers.q3_total_annual_hours or "(not stated)"
    tot_val = extract_first_number(total_str)
    sum_claim = (
        f"Check the arithmetic: {il_att_hours_str} + {fl_rn_hours_str} equals {total_str}. "
        "Allow minor rounding differences (e.g., within 0.5 hours) if applicable."
    )
    await evaluator.verify(
        claim=sum_claim,
        node=q3_sum_leaf,
        additional_instruction=(
            "Verify purely by arithmetic using the stated numbers in the answer; disregard external knowledge."
        ),
    )

    # Q3 Supporting References (require both IL Attorney and FL RN to be supported)
    # We implement as a critical parallel group with two leaves (one per combo).
    refs_node = evaluator.add_parallel(
        id="Q3_Supporting_References",
        desc="Provides valid reference URLs from official state licensing boards for both Illinois Attorney and Florida RN CE requirements",
        parent=q3_node,
        critical=True,
    )

    # IL Attorney reference support
    q3_il_ref_leaf = evaluator.add_leaf(
        id="Q3_IL_Attorney_Reference_Support",
        desc="Official sources support the Illinois Attorney annual CE requirement used",
        parent=refs_node,
        critical=True,
    )
    await evaluator.verify(
        claim=(f"The Illinois Attorney annual CE requirement used ({il_att_hours_str} per year or equivalent) is supported by the cited official source(s)."),
        node=q3_il_ref_leaf,
        sources=il_att_sources if il_att_sources else None,
        additional_instruction=(
            "Prefer official Illinois MCLE Board/Illinois Supreme Court sources. "
            "If a biennial cycle is stated, accept derivation to annual by dividing by the cycle length."
        ),
    )

    # FL RN reference support
    q3_fl_ref_leaf = evaluator.add_leaf(
        id="Q3_FL_RN_Reference_Support",
        desc="Official sources support the Florida RN annual CE requirement used",
        parent=refs_node,
        critical=True,
    )
    await evaluator.verify(
        claim=(f"The Florida RN annual CE requirement used ({fl_rn_hours_str} per year or equivalent) is supported by the cited official source(s)."),
        node=q3_fl_ref_leaf,
        sources=fl_rn_sources if fl_rn_sources else None,
        additional_instruction=(
            "Prefer official Florida Board of Nursing sources. "
            "If a biennial cycle is stated, accept derivation to annual by dividing by the cycle length."
        ),
    )


async def build_question_4(
    evaluator: Evaluator,
    parent_node,
    data: CEExtraction,
) -> None:
    """Build verification nodes for Question 4: count >12 hours per year and provide references."""
    q4_node = evaluator.add_parallel(
        id="Question_4_Count_Above_Threshold",
        desc="Count how many of the five specified profession-state combinations require more than 12 continuing education hours per year",
        parent=parent_node,
        critical=False,
    )

    # Q4 Correct Count
    q4_count_leaf = evaluator.add_leaf(
        id="Q4_Correct_Count",
        desc="Correctly counts the number of profession-state combinations with annual CE hours exceeding 12 hours based on calculated annual requirements",
        parent=q4_node,
        critical=True,
    )

    combo_descs = collect_all_combo_descriptions(data)
    count_str = data.answers.q4_count_above_12 or "(not stated)"
    listed_above = data.answers.q4_combos_above_12 or []
    q4_claim = (
        "Using only the numbers explicitly stated in the answer (including derivations from cycle totals divided by cycle length when appropriate), "
        "determine how many of the five combinations exceed 12 hours per year:\n"
        f"- " + "\n- ".join(combo_descs) + "\n"
        f"The answer claims the count is '{count_str}', listing these combos as >12: {listed_above}. "
        "Verify whether this count is logically correct given the provided numbers."
    )
    await evaluator.verify(
        claim=q4_claim,
        node=q4_count_leaf,
        additional_instruction="Use only the answer-provided numbers (or their simple cycle-based derivations) to decide; do not use external knowledge.",
    )

    # Q4 Supporting References (parallel verification per counted combo)
    refs_node = evaluator.add_parallel(
        id="Q4_Supporting_References",
        desc="Provides valid reference URLs from official state licensing boards documenting the CE requirements for the combinations counted",
        parent=q4_node,
        critical=True,
    )

    # Determine which combos to verify for references: use listed_above; if empty, verify all five combos.
    targets: List[Tuple[str, CEItem]] = []
    if listed_above:
        for name in listed_above:
            key = guess_key_from_name(name)
            item = get_item_by_key(data, key) if key else None
            if key and item:
                targets.append((KEY_TO_CANONICAL[key], item))
    else:
        # Fallback: verify all five combos to ensure broad source support
        for key in COMBO_KEYS:
            item = get_item_by_key(data, key)
            if item:
                targets.append((KEY_TO_CANONICAL[key], item))

    # Create a leaf per target combo
    for idx, (canon_name, item) in enumerate(targets):
        leaf = evaluator.add_leaf(
            id=f"Q4_Reference_{idx+1}",
            desc=f"Official sources support the CE requirement used for {canon_name}",
            parent=refs_node,
            critical=True,
        )
        # Build claim based on available fields
        annual_str = item.annual_hours or "(not stated)"
        cyc_str = item.cycle_hours or None
        yrs_str = item.cycle_length_years or None
        parts = [f"For {canon_name}, the annual CE requirement used is {annual_str} per year."]
        if cyc_str or yrs_str:
            parts.append(f"(The answer also mentions a cycle of {cyc_str or '?'} over {yrs_str or '?'})")
        claim = " ".join(parts) + " Verify that at least one cited official source supports this requirement."

        await evaluator.verify(
            claim=claim,
            node=leaf,
            sources=item.sources if item.sources else (data.answers.q4_sources if data.answers.q4_sources else None),
            additional_instruction=(
                "Prefer official regulatory sources (e.g., state boards of accountancy, state MCLE boards, IDFPR, "
                "State Bar of Texas, Florida Board of Nursing). "
                "If a multi-year cycle is stated, accept derivation to annual by dividing cycle total by cycle length."
            ),
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
    Evaluate an answer for continuing education requirements across specified profession-state combinations.
    """
    # Initialize evaluator (root is parallel per rubric)
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

    # Create top-level rubric root node (non-critical, parallel)
    complete_node = evaluator.add_parallel(
        id="Complete_All_Questions",
        desc="Answer all four questions about continuing education requirements for professional licenses across different states",
        parent=root,
        critical=False,
    )

    # Extract structured information from the answer
    extraction = await evaluator.extract(
        prompt=prompt_extract_ce_data(),
        template_class=CEExtraction,
        extraction_name="ce_extraction",
    )

    # Build question subtrees
    await build_question_1(evaluator, complete_node, extraction)
    await build_question_2(evaluator, complete_node, extraction)
    await build_question_3(evaluator, complete_node, extraction)
    await build_question_4(evaluator, complete_node, extraction)

    # Return evaluation summary
    return evaluator.get_summary()