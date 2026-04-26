import asyncio
import logging
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field
from obj_task_eval.llm_client.base_client import LLMClient

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy

# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "wootton_profile_2025_2026"
TASK_DESCRIPTION = (
    "I am compiling a comprehensive school profile for Thomas S. Wootton High School in Rockville, Maryland for the "
    "2025-2026 academic year. Please provide the following verified information about the school:\n\n"
    "1. Complete school address (street address, city, state, and ZIP code)\n"
    "2. Year the school was founded/opened\n"
    "3. Official school colors\n"
    "4. Official school mascot\n"
    "5. Current principal's name (2025-2026 school year)\n"
    "6. Current student enrollment (grades 9-12)\n"
    "7. Student-teacher ratio\n"
    "8. US News national ranking (2026 edition)\n"
    "9. Ranking among Maryland high schools (2026)\n"
    "10. Number of Advanced Placement (AP) courses offered\n"
    "11. MCPS graduation credit requirement\n"
    "12. Names of the feeder middle schools\n"
    "13. Regular school day start time (Period 1)\n"
    "14. Main school phone number\n\n"
    "For each piece of information, provide the specific factual detail along with at least one supporting reference URL "
    "from an official or authoritative source."
)

# Ground truth / expected references (used for "match expected" checks)
EXPECTED = {
    "address": "2100 Wootton Parkway, Rockville, MD 20850",
    "year_founded": "1970",
    "colors": "red, white, and blue",
    "mascot": "Patriots",
    "principal": "Dr. Joseph Bostic, Jr.",
    "enrollment_min": 1870,
    "enrollment_max": 1875,
    "ratio_min": 17,
    "ratio_max": 19,
    "ratio_target": "18:1",
    "us_news_national": "#191",
    "md_rank": "#3",
    "ap_courses": "30",
    "grad_credits": "22",
    "feeders": ["Cabin John Middle School", "Robert Frost Middle School"],
    "start_time": "7:45 AM",
    "phone": "240-740-1500",
}

# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class FactWithSources(BaseModel):
    value: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class FactArrayWithSources(BaseModel):
    values: List[str] = Field(default_factory=list)
    sources: List[str] = Field(default_factory=list)


class WoottonProfileExtraction(BaseModel):
    address: Optional[FactWithSources] = None
    year_founded: Optional[FactWithSources] = None
    colors: Optional[FactWithSources] = None
    mascot: Optional[FactWithSources] = None
    principal: Optional[FactWithSources] = None
    enrollment: Optional[FactWithSources] = None
    student_teacher_ratio: Optional[FactWithSources] = None
    us_news_national_rank_2026: Optional[FactWithSources] = None
    maryland_state_rank_2026: Optional[FactWithSources] = None
    ap_courses_offered: Optional[FactWithSources] = None
    graduation_credits_requirement: Optional[FactWithSources] = None
    feeder_middle_schools: Optional[FactArrayWithSources] = None
    start_time_period1: Optional[FactWithSources] = None
    main_phone_number: Optional[FactWithSources] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_wootton_profile() -> str:
    return """
    Extract the following information for Thomas S. Wootton High School (Rockville, Maryland) from the provided answer.
    For each item, return both the factual value and an array of URLs (sources) explicitly cited in the answer.

    Return a JSON object with this structure:
    {
      "address": {"value": string|null, "sources": [urls...]},
      "year_founded": {"value": string|null, "sources": [urls...]},
      "colors": {"value": string|null, "sources": [urls...]},
      "mascot": {"value": string|null, "sources": [urls...]},
      "principal": {"value": string|null, "sources": [urls...]},
      "enrollment": {"value": string|null, "sources": [urls...]},
      "student_teacher_ratio": {"value": string|null, "sources": [urls...]},
      "us_news_national_rank_2026": {"value": string|null, "sources": [urls...]},
      "maryland_state_rank_2026": {"value": string|null, "sources": [urls...]},
      "ap_courses_offered": {"value": string|null, "sources": [urls...]},
      "graduation_credits_requirement": {"value": string|null, "sources": [urls...]},
      "feeder_middle_schools": {"values": [strings...], "sources": [urls...]},
      "start_time_period1": {"value": string|null, "sources": [urls...]},
      "main_phone_number": {"value": string|null, "sources": [urls...]}
    }

    Extraction rules:
    - Extract values exactly as stated in the answer. Do not invent information.
    - Sources must be full URLs explicitly present in the answer (plaintext or markdown links).
    - If an item is missing in the answer or no source is provided, set value to null (or empty list for values) and sources to [].
    - Prefer official/authoritative sources (MCPS, school website, US News), but extract whatever URLs are cited in the answer.
    - For colors, a single string is fine (e.g., "red, white, and blue").
    - For feeder middle schools, provide an array of school names.
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _has_value_and_sources(fact: Optional[FactWithSources]) -> bool:
    return bool(fact and fact.value and fact.value.strip() and fact.sources and len(fact.sources) > 0)


def _has_array_and_sources(fact: Optional[FactArrayWithSources]) -> bool:
    return bool(fact and fact.values and len(fact.values) > 0 and fact.sources and len(fact.sources) > 0)


def _safe_value(fact: Optional[FactWithSources]) -> str:
    return (fact.value or "").strip() if fact else ""


def _safe_values(fact: Optional[FactArrayWithSources]) -> List[str]:
    return fact.values if fact and fact.values else []


# --------------------------------------------------------------------------- #
# Verification building blocks                                                #
# --------------------------------------------------------------------------- #
async def add_fact_verification(
    evaluator: Evaluator,
    parent_node,
    node_id: str,
    node_desc: str,
    fact: Optional[FactWithSources],
    *,
    parent_critical: bool,
    expected_match_instruction: Optional[str],
    expected_value: Optional[str],
    support_claim_template: str,
    support_instruction: str
) -> None:
    """
    Add a verification sub-tree for a single string fact:
    - existence (critical)
    - match expected (optional, non-critical unless parent is critical and caller wants it)
    - supported by sources (critical)
    """
    criterion_node = evaluator.add_parallel(
        id=node_id,
        desc=node_desc,
        parent=parent_node,
        critical=parent_critical
    )

    # Existence check (critical)
    evaluator.add_custom_node(
        result=_has_value_and_sources(fact),
        id=f"{node_id}_exists",
        desc=f"{node_desc} – value present and at least one source URL provided",
        parent=criterion_node,
        critical=True
    )

    # Match expected (non-critical except when caller sets expected_value and wants strict match for critical parents)
    if expected_value is not None:
        match_node = evaluator.add_leaf(
            id=f"{node_id}_match_expected",
            desc=f"{node_desc} – matches expected value",
            parent=criterion_node,
            critical=True if parent_critical else False
        )
        claim = f"The provided value '{_safe_value(fact)}' matches the expected value '{expected_value}'."
        await evaluator.verify(
            claim=claim,
            node=match_node,
            additional_instruction=(expected_match_instruction or "Allow minor formatting/casing differences.")
        )

    # Source-supported (critical)
    support_node = evaluator.add_leaf(
        id=f"{node_id}_supported_by_sources",
        desc=f"{node_desc} – supported by cited sources",
        parent=criterion_node,
        critical=True
    )
    # Build claim text using template and extracted value
    claim_text = support_claim_template.format(value=_safe_value(fact))
    await evaluator.verify(
        claim=claim_text,
        node=support_node,
        sources=(fact.sources if fact else []),
        additional_instruction=support_instruction
    )


async def add_array_fact_verification(
    evaluator: Evaluator,
    parent_node,
    node_id: str,
    node_desc: str,
    fact: Optional[FactArrayWithSources],
    *,
    parent_critical: bool,
    expected_values: Optional[List[str]],
    expected_match_instruction: Optional[str],
    support_instruction: str
) -> None:
    """
    Add verification sub-tree for an array fact (e.g., feeder middle schools).
    """
    criterion_node = evaluator.add_parallel(
        id=node_id,
        desc=node_desc,
        parent=parent_node,
        critical=parent_critical
    )

    # Existence (critical)
    evaluator.add_custom_node(
        result=_has_array_and_sources(fact),
        id=f"{node_id}_exists",
        desc=f"{node_desc} – list present and at least one source URL provided",
        parent=criterion_node,
        critical=True
    )

    # Match expected (non-critical unless parent is critical and strict match is desired)
    if expected_values is not None:
        match_node = evaluator.add_leaf(
            id=f"{node_id}_match_expected",
            desc=f"{node_desc} – includes all expected items",
            parent=criterion_node,
            critical=True if parent_critical else False
        )
        extracted_list = _safe_values(fact)
        claim = f"The extracted list {extracted_list} includes all expected items {expected_values} (allowing minor naming variations)."
        await evaluator.verify(
            claim=claim,
            node=match_node,
            additional_instruction=(expected_match_instruction or "Allow abbreviations (e.g., 'MS' for Middle School), casing differences, and minor punctuation variations.")
        )

    # Supported by sources (critical)
    support_node = evaluator.add_leaf(
        id=f"{node_id}_supported_by_sources",
        desc=f"{node_desc} – supported by cited sources",
        parent=criterion_node,
        critical=True
    )
    extracted_list = _safe_values(fact)
    claim_text = f"The feeder middle schools are {extracted_list} for Thomas S. Wootton High School."
    await evaluator.verify(
        claim=claim_text,
        node=support_node,
        sources=(fact.sources if fact else []),
        additional_instruction=support_instruction
    )


# --------------------------------------------------------------------------- #
# Item-specific verification orchestration                                    #
# --------------------------------------------------------------------------- #
async def build_verifications(evaluator: Evaluator, root, data: WoottonProfileExtraction) -> None:
    # 1. Address (critical parent)
    await add_fact_verification(
        evaluator=evaluator,
        parent_node=root,
        node_id="School_Location",
        node_desc="Complete school address is provided",
        fact=data.address,
        parent_critical=True,
        expected_value=EXPECTED["address"],
        expected_match_instruction="Check if the provided address equals the expected address, allowing minor punctuation or formatting differences.",
        support_claim_template="The complete address for Thomas S. Wootton High School is '{value}'.",
        support_instruction="Verify the full street address (street, city, state, ZIP) on official sources (MCPS or the school's website)."
    )

    # 2. Year founded (non-critical)
    await add_fact_verification(
        evaluator=evaluator,
        parent_node=root,
        node_id="Year_Founded",
        node_desc="Year the school was founded/opened is provided",
        fact=data.year_founded,
        parent_critical=False,
        expected_value=EXPECTED["year_founded"],
        expected_match_instruction="Treat 'opened' or 'established' as equivalent. Minor differences like 'around 1970' should not be a match.",
        support_claim_template="Thomas S. Wootton High School was founded/opened in '{value}'.",
        support_instruction="Confirm the founding/opening year from authoritative sources (MCPS historical page, school history page, or reputable directory)."
    )

    # 3. School colors (non-critical)
    await add_fact_verification(
        evaluator=evaluator,
        parent_node=root,
        node_id="School_Colors",
        node_desc="Official school colors are provided",
        fact=data.colors,
        parent_critical=False,
        expected_value=EXPECTED["colors"],
        expected_match_instruction="Allow color name casing differences; 'red, white, and blue' equivalent to 'Red/White/Blue'.",
        support_claim_template="The official school colors are '{value}'.",
        support_instruction="Verify colors via the school's official site, athletics/branding pages, or MCPS resources."
    )

    # 4. Mascot (non-critical)
    await add_fact_verification(
        evaluator=evaluator,
        parent_node=root,
        node_id="School_Mascot",
        node_desc="Official school mascot is provided",
        fact=data.mascot,
        parent_critical=False,
        expected_value=EXPECTED["mascot"],
        expected_match_instruction="Allow singular/plural and articles (e.g., 'the Patriots').",
        support_claim_template="The official school mascot is '{value}'.",
        support_instruction="Confirm mascot on school site (about/athletics) or MCPS cluster pages."
    )

    # 5. Current principal (non-critical)
    await add_fact_verification(
        evaluator=evaluator,
        parent_node=root,
        node_id="Current_Principal",
        node_desc="Current principal's name (2025-2026) is provided",
        fact=data.principal,
        parent_critical=False,
        expected_value=EXPECTED["principal"],
        expected_match_instruction="Allow minor formatting differences, initials, suffix (Jr.), and titles (Dr.).",
        support_claim_template="During the 2025-2026 school year, the principal is '{value}'.",
        support_instruction="Confirm from the school's official site (administration/staff pages), MCPS directory, or official announcements/newsletters."
    )

    # 6. Student enrollment (non-critical, accept range 1870–1875)
    enrollment_value = _safe_value(data.enrollment)
    match_enrollment_instruction = (
        f"Consider a match if the extracted enrollment '{enrollment_value}' is between {EXPECTED['enrollment_min']} and {EXPECTED['enrollment_max']} inclusive. "
        "Allow approximate wording like '~1,874' or 'around 1,874'."
    )
    await add_fact_verification(
        evaluator=evaluator,
        parent_node=root,
        node_id="Student_Enrollment",
        node_desc="Current student enrollment (grades 9-12) is provided",
        fact=data.enrollment,
        parent_critical=False,
        expected_value=f"{EXPECTED['enrollment_min']}-{EXPECTED['enrollment_max']}",
        expected_match_instruction=match_enrollment_instruction,
        support_claim_template="The current student enrollment for Thomas S. Wootton High School (grades 9–12) is '{value}'.",
        support_instruction="Verify the enrollment number from MCPS data dashboards, school profile pages, or official reports. Accept slight rounding/approximation."
    )

    # 7. Student-teacher ratio (non-critical, accept ~18:1 within 17–19)
    ratio_value = _safe_value(data.student_teacher_ratio)
    match_ratio_instruction = (
        f"Consider a match if the extracted ratio '{ratio_value}' is approximately {EXPECTED['ratio_target']} and within {EXPECTED['ratio_min']}:1 to {EXPECTED['ratio_max']}:1. "
        "Allow formatting differences (e.g., '18 to 1', '18/1')."
    )
    await add_fact_verification(
        evaluator=evaluator,
        parent_node=root,
        node_id="Student_Teacher_Ratio",
        node_desc="Student-teacher ratio is provided",
        fact=data.student_teacher_ratio,
        parent_critical=False,
        expected_value=f"~{EXPECTED['ratio_target']} (acceptable range {EXPECTED['ratio_min']}:1–{EXPECTED['ratio_max']}:1)",
        expected_match_instruction=match_ratio_instruction,
        support_claim_template="The student-teacher ratio at Thomas S. Wootton High School is '{value}'.",
        support_instruction="Confirm ratio from MCPS statistics, state report cards, or reputable aggregators. Accept minor approximation."
    )

    # 8. US News national ranking (2026 edition) (non-critical)
    await add_fact_verification(
        evaluator=evaluator,
        parent_node=root,
        node_id="US_News_National_Ranking",
        node_desc="US News national ranking (2026 edition) is provided",
        fact=data.us_news_national_rank_2026,
        parent_critical=False,
        expected_value=EXPECTED["us_news_national"],
        expected_match_instruction="Allow '#' symbol or absence (e.g., '191'). Ensure edition year is 2026.",
        support_claim_template="In the U.S. News 2026 edition, Thomas S. Wootton High School has a national ranking of '{value}'.",
        support_instruction="Verify on the official U.S. News school ranking page for the 2026 edition. The claim must correspond to the 2026 rankings."
    )

    # 9. Maryland state ranking (2026) (non-critical)
    await add_fact_verification(
        evaluator=evaluator,
        parent_node=root,
        node_id="Maryland_State_Ranking",
        node_desc="Maryland high schools ranking (2026) is provided",
        fact=data.maryland_state_rank_2026,
        parent_critical=False,
        expected_value=EXPECTED["md_rank"],
        expected_match_instruction="Allow '#' or numeric only (e.g., '#3' vs '3'). Ensure edition year is 2026.",
        support_claim_template="In the U.S. News 2026 edition, Thomas S. Wootton High School is ranked '{value}' among Maryland high schools.",
        support_instruction="Verify using U.S. News (2026) or authoritative ranking summaries; ensure the year is correct."
    )

    # 10. AP courses offered (non-critical)
    await add_fact_verification(
        evaluator=evaluator,
        parent_node=root,
        node_id="AP_Courses_Offered",
        node_desc="Number of AP courses offered is provided",
        fact=data.ap_courses_offered,
        parent_critical=False,
        expected_value=EXPECTED["ap_courses"],
        expected_match_instruction="Allow numeric formatting variations; must indicate count of AP courses.",
        support_claim_template="Thomas S. Wootton High School offers '{value}' Advanced Placement (AP) courses.",
        support_instruction="Confirm using the school's course catalog, academic program page, or MCPS curriculum resources."
    )

    # 11. Graduation credits requirement (non-critical)
    await add_fact_verification(
        evaluator=evaluator,
        parent_node=root,
        node_id="Graduation_Credits",
        node_desc="MCPS graduation credit requirement is provided",
        fact=data.graduation_credits_requirement,
        parent_critical=False,
        expected_value=EXPECTED["grad_credits"],
        expected_match_instruction="Confirm MCPS standard graduation credit requirement. Minor wording like 'minimum 22 credits' is acceptable.",
        support_claim_template="The MCPS graduation credit requirement is '{value}' credits.",
        support_instruction="Verify on MCPS official graduation requirements page or official policy documents."
    )

    # 12. Feeder middle schools (non-critical)
    await add_array_fact_verification(
        evaluator=evaluator,
        parent_node=root,
        node_id="Feeder_Middle_Schools",
        node_desc="Feeder middle schools are provided",
        fact=data.feeder_middle_schools,
        parent_critical=False,
        expected_values=EXPECTED["feeders"],
        expected_match_instruction="Allow abbreviations (e.g., 'Cabin John MS', 'Robert Frost MS') and minor variations.",
        support_instruction="Verify feeder patterns on MCPS cluster/feeder pattern pages or school site; both Cabin John MS and Robert Frost MS must be included."
    )

    # 13. School start time (non-critical)
    await add_fact_verification(
        evaluator=evaluator,
        parent_node=root,
        node_id="School_Start_Time",
        node_desc="Regular school day start time (Period 1) is provided",
        fact=data.start_time_period1,
        parent_critical=False,
        expected_value=EXPECTED["start_time"],
        expected_match_instruction="Allow minor formatting like '7:45AM' without space; ensure it's the standard start of period 1.",
        support_claim_template="The regular school day start time (Period 1) at Thomas S. Wootton High School is '{value}'.",
        support_instruction="Confirm on the school's bell schedule or MCPS schedule documentation."
    )

    # 14. Main phone number (non-critical)
    await add_fact_verification(
        evaluator=evaluator,
        parent_node=root,
        node_id="Main_Phone_Number",
        node_desc="Main school phone number is provided",
        fact=data.main_phone_number,
        parent_critical=False,
        expected_value=EXPECTED["phone"],
        expected_match_instruction="Allow formatting variants like '(240) 740-1500' vs '240-740-1500'.",
        support_claim_template="The main phone number for Thomas S. Wootton High School is '{value}'.",
        support_instruction="Verify via the school's contact page or MCPS directory."
    )


# --------------------------------------------------------------------------- #
# Main evaluation entry point                                                 #
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
    Evaluate an answer for the Wootton profile task (2025-2026).
    """
    # Initialize evaluator with a non-critical root to allow partial credit across many independent criteria
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

    # Extract structured data
    extracted = await evaluator.extract(
        prompt=prompt_extract_wootton_profile(),
        template_class=WoottonProfileExtraction,
        extraction_name="wootton_profile_extraction"
    )

    # Add ground-truth expectations (for transparency; used only for match checks)
    evaluator.add_ground_truth(
        {
            "expected": EXPECTED,
            "notes": "Expected values used for 'match expected' checks. Source-supported checks rely on the URLs provided by the answer."
        },
        gt_type="expected_values"
    )

    # Build verification tree and run checks
    await build_verifications(evaluator, root, extracted)

    # Return normalized summary
    return evaluator.get_summary()