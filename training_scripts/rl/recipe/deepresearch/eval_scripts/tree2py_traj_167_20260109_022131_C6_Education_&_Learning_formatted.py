import asyncio
import logging
import re
from typing import Any, List, Optional, Dict, Tuple

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "stackable_data_analytics_certificate_pathway"
TASK_DESCRIPTION = (
    "A Licensed Professional Counselor in Texas is seeking to transition into data analytics while maintaining "
    "professional credentials. They want to pursue an online graduate certificate that is part of a stackable "
    "credential pathway at a major U.S. university.\n\n"
    "Find ONE graduate certificate program that meets ALL of the following requirements:\n\n"
    "1. Offered by a regionally accredited U.S. university with an established stackable credentials program\n"
    "2. The certificate is specifically in data analytics or a closely related field\n"
    "3. The certificate consists of 9-16 credits\n"
    "4. Can be completed within 8-18 months\n"
    "5. Offered 100% online\n"
    "6. Stacks into a master's degree in data analytics or a closely related field\n"
    "7. Per-credit tuition is $1,500 or less (based on 2025-2026 academic year rates)\n\n"
    "Provide the following information about the identified program:\n"
    "- University name\n"
    "- Graduate certificate program name\n"
    "- Exact number of credit hours\n"
    "- Estimated completion timeline\n"
    "- The specific master's degree program it stacks into\n"
    "- Per-credit tuition rate for 2025-2026\n"
    "- Total certificate cost"
)


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class ProgramEntry(BaseModel):
    university_name: Optional[str] = None
    certificate_name: Optional[str] = None
    credit_hours: Optional[str] = None
    completion_timeline: Optional[str] = None  # e.g., "12 months", "8–18 months", "1 year"
    masters_program_name: Optional[str] = None
    per_credit_tuition: Optional[str] = None   # string as stated
    tuition_year: Optional[str] = None         # e.g., "2025-2026", "AY 2025-2026"
    total_certificate_cost: Optional[str] = None

    # URLs referenced in the answer (critical for verification)
    program_page_urls: List[str] = Field(default_factory=list)
    tuition_page_urls: List[str] = Field(default_factory=list)
    accreditation_urls: List[str] = Field(default_factory=list)
    stackability_page_urls: List[str] = Field(default_factory=list)
    masters_program_page_urls: List[str] = Field(default_factory=list)
    online_modality_urls: List[str] = Field(default_factory=list)
    stackable_initiative_urls: List[str] = Field(default_factory=list)


class ProgramSet(BaseModel):
    programs: List[ProgramEntry] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_programs() -> str:
    return """
    Extract all graduate certificate program(s) mentioned in the answer. The user requests exactly ONE program,
    but you must extract any programs mentioned in the answer verbatim. For each identified program, return the following fields:
    - university_name: The U.S. university name
    - certificate_name: The graduate certificate program name
    - credit_hours: Exact number of credit hours as stated (string)
    - completion_timeline: The estimated time to complete (string; e.g., "12 months", "8–18 months")
    - masters_program_name: The specific master’s program it stacks into
    - per_credit_tuition: Per-credit tuition rate (string, with currency symbol if present)
    - tuition_year: The academic year for the per-credit tuition (string, e.g., "2025-2026", "AY 2025-2026")
    - total_certificate_cost: Total certificate cost (string, with currency symbol if present)
    - program_page_urls: URLs cited that describe the certificate program
    - tuition_page_urls: URLs cited for tuition information
    - accreditation_urls: URLs cited that document regional accreditation of the university
    - stackability_page_urls: URLs cited that explicitly document this certificate stacks into the named master’s program
    - masters_program_page_urls: URLs cited that describe the master’s program
    - online_modality_urls: URLs cited that explicitly document the 100% online delivery (if different from program page)
    - stackable_initiative_urls: URLs cited that document the university’s stackable credentials initiative/program

    RULES:
    - Extract ONLY what is explicitly present in the answer.
    - If a field is missing for a program, return null for scalar fields or an empty array for URL lists.
    - Extract ALL URLs in the answer relevant to the fields (program page, tuition page, accreditation, stackability, masters program page, online modality, stackable initiative).
    - If the answer includes multiple program options, include all of them in the 'programs' array (we will later check that exactly one is identified).
    """


# --------------------------------------------------------------------------- #
# Helpers for numeric parsing                                                 #
# --------------------------------------------------------------------------- #
def _extract_numbers(text: Optional[str]) -> List[float]:
    if not text:
        return []
    nums = re.findall(r"(\d+(?:\.\d+)?)", text.replace(",", ""))
    try:
        return [float(n) for n in nums]
    except Exception:
        return []


def _contains_range_indicator(text: Optional[str]) -> bool:
    if not text:
        return False
    t = text.lower()
    return any(ind in t for ind in ["-", "–", "to", "through"])


def _parse_credit_hours_exact(text: Optional[str]) -> Optional[int]:
    """
    Returns an integer credit hours if the text appears to specify a single exact number.
    """
    if not text:
        return None
    numbers = _extract_numbers(text)
    if len(numbers) != 1 or _contains_range_indicator(text):
        return None
    # Credits should be a whole number commonly; cast to int if close
    return int(round(numbers[0]))


def _parse_credit_hours_any(text: Optional[str]) -> Optional[int]:
    """
    Returns a plausible integer credit hours (first number found) even if the text contains multiple numbers.
    """
    if not text:
        return None
    numbers = _extract_numbers(text)
    if not numbers:
        return None
    return int(round(numbers[0]))


def _parse_months(text: Optional[str]) -> List[int]:
    """
    Attempt to derive months from a timeline string.
    Handles 'months' and 'year(s)'. Returns list of month values (for ranges returns both).
    """
    if not text:
        return []
    t = text.lower()
    numbers = _extract_numbers(t)

    # If mentions year(s), convert numbers to months * 12
    if "year" in t or "yr" in t or "years" in t:
        return [int(round(n * 12)) for n in numbers]
    # Otherwise assume months
    return [int(round(n)) for n in numbers]


def _parse_money(text: Optional[str]) -> Optional[float]:
    """
    Parse a monetary amount from text (e.g., "$1,250", "USD 1400", "1,500.00").
    """
    if not text:
        return None
    nums = _extract_numbers(text)
    if not nums:
        return None
    return float(nums[0])


def _is_within(value: Optional[int], min_v: int, max_v: int) -> bool:
    if value is None:
        return False
    return min_v <= value <= max_v


def _months_within_range(months_list: List[int], min_v: int, max_v: int) -> bool:
    if not months_list:
        return False
    # If any value in the list is within range, accept
    return any(min_v <= m <= max_v for m in months_list)


def _cost_consistent(credits: Optional[int], rate: Optional[float], total: Optional[float]) -> bool:
    """
    Check arithmetic consistency: total ≈ credits * rate, allowing small rounding tolerance.
    """
    if credits is None or rate is None or total is None:
        return False
    expected = credits * rate
    diff = abs(total - expected)
    tolerance = max(0.01 * expected, 25.0)  # allow 1% or $25 rounding
    return diff <= tolerance


def _merge_sources(*lists: List[str]) -> List[str]:
    seen = set()
    merged: List[str] = []
    for lst in lists:
        for url in lst or []:
            if url and url not in seen:
                seen.add(url)
                merged.append(url)
    return merged


# --------------------------------------------------------------------------- #
# Verification subtrees                                                       #
# --------------------------------------------------------------------------- #
async def build_and_verify_institution_requirements(
    evaluator: Evaluator,
    parent_node,
    program: ProgramEntry,
    program_count_node,
) -> None:
    inst_node = evaluator.add_parallel(
        id="Institution_Requirements",
        desc="Institution satisfies the university-level constraints and is properly identified.",
        parent=parent_node,
        critical=True,
    )

    # University name provided
    uni_name_ok = bool(program.university_name and program.university_name.strip())
    uni_name_node = evaluator.add_custom_node(
        result=uni_name_ok,
        id="University_Name_Provided",
        desc="University name is provided.",
        parent=inst_node,
        critical=True,
    )

    # Regionally accredited U.S. institution (verify via URLs if available)
    accred_leaf = evaluator.add_leaf(
        id="University_Is_US_Regionally_Accredited",
        desc="University is a U.S. institution and is regionally accredited.",
        parent=inst_node,
        critical=True,
    )
    claim = (
        f"The university '{program.university_name or ''}' is a regionally accredited U.S. institution "
        f"(e.g., HLC, MSCHE, SACSCOC, NECHE, NWCCU, or WSCUC)."
    )
    await evaluator.verify(
        claim=claim,
        node=accred_leaf,
        sources=program.accreditation_urls,
        additional_instruction=(
            "Look for explicit statements indicating accreditation by a recognized U.S. regional accreditor "
            "(HLC, MSCHE, SACSCOC, NECHE, NWCCU, WSCUC). If the answer provides an accreditation page, verify there."
        ),
        extra_prerequisites=[program_count_node, uni_name_node],
    )

    # Stackable credentials initiative exists (university-level)
    stack_init_leaf = evaluator.add_leaf(
        id="Stackable_Credentials_Initiative_Exists",
        desc="University has an established stackable credentials program/initiative (documented/claimed by the solution).",
        parent=inst_node,
        critical=True,
    )
    claim = (
        f"The university '{program.university_name or ''}' has an established stackable credentials initiative or program "
        f"(e.g., stackable certificates, stacked microcredentials, formal stackable pathways)."
    )
    stack_init_sources = _merge_sources(program.stackable_initiative_urls, program.stackability_page_urls, program.program_page_urls)
    await evaluator.verify(
        claim=claim,
        node=stack_init_leaf,
        sources=stack_init_sources,
        additional_instruction="Accept reasonable synonyms such as 'stackable certificates', 'stacked credentials', or 'stackable pathways' when clearly documented.",
        extra_prerequisites=[program_count_node, uni_name_node],
    )


async def build_and_verify_certificate_requirements(
    evaluator: Evaluator,
    parent_node,
    program: ProgramEntry,
    program_count_node,
) -> None:
    cert_node = evaluator.add_parallel(
        id="Certificate_Requirements",
        desc="Certificate satisfies program-format, field, credit, timeline, and modality constraints and required reporting fields.",
        parent=parent_node,
        critical=True,
    )

    # Certificate name provided
    cert_name_ok = bool(program.certificate_name and program.certificate_name.strip())
    cert_name_node = evaluator.add_custom_node(
        result=cert_name_ok,
        id="Certificate_Name_Provided",
        desc="Graduate certificate program name is provided.",
        parent=cert_node,
        critical=True,
    )

    # Field is data analytics or closely related
    field_leaf = evaluator.add_leaf(
        id="Certificate_Field_Is_Data_Analytics_Related",
        desc="Certificate is specifically in data analytics or a closely related field (e.g., data science, business analytics).",
        parent=cert_node,
        critical=True,
    )
    claim = (
        f"The certificate program '{program.certificate_name or ''}' is in data analytics or a closely related field "
        f"(e.g., data science, business analytics, applied analytics)."
    )
    await evaluator.verify(
        claim=claim,
        node=field_leaf,
        sources=_merge_sources(program.program_page_urls),
        additional_instruction="Check the program title, description, and curriculum for clear alignment with analytics-related fields.",
        extra_prerequisites=[program_count_node, cert_name_node],
    )

    # Exact credit hours provided
    exact_credits_val = _parse_credit_hours_exact(program.credit_hours)
    exact_credits_provided = exact_credits_val is not None
    exact_credits_node = evaluator.add_custom_node(
        result=exact_credits_provided,
        id="Exact_Credit_Hours_Provided",
        desc="Exact number of credit hours is provided.",
        parent=cert_node,
        critical=True,
    )

    # Credits within 9–16
    any_credits_val = _parse_credit_hours_any(program.credit_hours)
    credits_within = _is_within(any_credits_val, 9, 16)
    credits_range_node = evaluator.add_custom_node(
        result=credits_within,
        id="Credits_Within_9_to_16",
        desc="Certificate credit total is within 9–16 credits (inclusive).",
        parent=cert_node,
        critical=True,
    )

    # Completion timeline provided
    timeline_provided = bool(program.completion_timeline and program.completion_timeline.strip())
    timeline_provided_node = evaluator.add_custom_node(
        result=timeline_provided,
        id="Completion_Timeline_Provided",
        desc="Estimated completion timeline is provided.",
        parent=cert_node,
        critical=True,
    )

    # Timeline within 8–18 months
    months_vals = _parse_months(program.completion_timeline)
    timeline_within = _months_within_range(months_vals, 8, 18)
    timeline_range_node = evaluator.add_custom_node(
        result=timeline_within,
        id="Timeline_Within_8_to_18_Months",
        desc="Certificate can be completed within 8–18 months (inclusive), as stated by the solution.",
        parent=cert_node,
        critical=True,
    )

    # Offered 100% online
    online_leaf = evaluator.add_leaf(
        id="Offered_100_Percent_Online",
        desc="Program is offered 100% online (no required in-person components).",
        parent=cert_node,
        critical=True,
    )
    claim = (
        f"The certificate program '{program.certificate_name or ''}' is offered 100% online with no required on-campus components."
    )
    online_sources = _merge_sources(program.online_modality_urls, program.program_page_urls)
    await evaluator.verify(
        claim=claim,
        node=online_leaf,
        sources=online_sources,
        additional_instruction="Look for explicit statements such as '100% online', 'fully online', or 'no campus visits required'.",
        extra_prerequisites=[program_count_node, cert_name_node],
    )


async def build_and_verify_stackability_requirements(
    evaluator: Evaluator,
    parent_node,
    program: ProgramEntry,
    program_count_node,
) -> None:
    stack_node = evaluator.add_parallel(
        id="Stackability_Requirements",
        desc="Certificate stacks into a relevant master's program and the required master's details are provided.",
        parent=parent_node,
        critical=True,
    )

    # Master's program name provided
    masters_name_ok = bool(program.masters_program_name and program.masters_program_name.strip())
    masters_name_node = evaluator.add_custom_node(
        result=masters_name_ok,
        id="Masters_Program_Name_Provided",
        desc="Specific master's degree program it stacks into is named.",
        parent=stack_node,
        critical=True,
    )

    # Master's field is analytics-related
    masters_field_leaf = evaluator.add_leaf(
        id="Masters_Field_Is_Data_Analytics_Related",
        desc="The named master's program is in data analytics or a closely related field.",
        parent=stack_node,
        critical=True,
    )
    claim = (
        f"The master's program '{program.masters_program_name or ''}' is in data analytics or a closely related field "
        f"(e.g., data science, business analytics, applied analytics)."
    )
    masters_sources = _merge_sources(program.masters_program_page_urls, program.stackability_page_urls)
    await evaluator.verify(
        claim=claim,
        node=masters_field_leaf,
        sources=masters_sources,
        additional_instruction="Confirm the master's program focus aligns with analytics-related fields.",
        extra_prerequisites=[program_count_node, masters_name_node],
    )

    # Explicit stackability documented
    explicit_stack_leaf = evaluator.add_leaf(
        id="Explicit_Stackability_Documented",
        desc="Solution includes an explicit statement/documentation that the certificate stacks into the named master's program (i.e., a defined pathway/credit application relationship).",
        parent=stack_node,
        critical=True,
    )
    claim = (
        f"The certificate '{program.certificate_name or ''}' explicitly stacks into the master's program "
        f"'{program.masters_program_name or ''}', with a defined pathway and/or credit application relationship."
    )
    await evaluator.verify(
        claim=claim,
        node=explicit_stack_leaf,
        sources=_merge_sources(program.stackability_page_urls, program.program_page_urls),
        additional_instruction=(
            "Look for explicit statements such as 'stacks into', 'credits apply toward', 'pathway to the master's', "
            "'share credits', or 'stackable certificate into the master's'."
        ),
        extra_prerequisites=[program_count_node, masters_name_node],
    )


async def build_and_verify_cost_requirements(
    evaluator: Evaluator,
    parent_node,
    program: ProgramEntry,
    program_count_node,
) -> None:
    cost_node = evaluator.add_parallel(
        id="Cost_Requirements",
        desc="Tuition constraints are satisfied and required cost fields are provided.",
        parent=parent_node,
        critical=True,
    )

    # Per-credit tuition provided
    tuition_provided = bool(program.per_credit_tuition and program.per_credit_tuition.strip())
    tuition_provided_node = evaluator.add_custom_node(
        result=tuition_provided,
        id="Per_Credit_Tuition_Provided",
        desc="Per-credit tuition rate is provided.",
        parent=cost_node,
        critical=True,
    )

    # Tuition year is 2025–2026 (verify via tuition page if possible)
    tuition_year_leaf = evaluator.add_leaf(
        id="Tuition_Year_Is_2025_2026",
        desc="Per-credit tuition rate is explicitly for the 2025–2026 academic year.",
        parent=cost_node,
        critical=True,
    )
    claim = (
        "The per-credit tuition rate shown applies specifically to the 2025–2026 academic year (e.g., 'AY 2025-2026')."
    )
    await evaluator.verify(
        claim=claim,
        node=tuition_year_leaf,
        sources=program.tuition_page_urls,
        additional_instruction="Look for explicit labeling of the academic year '2025–2026' or equivalent phrasing on the tuition page.",
        extra_prerequisites=[program_count_node, tuition_provided_node],
    )

    # Per-credit tuition at most $1,500
    rate_val = _parse_money(program.per_credit_tuition)
    per_credit_ok = (rate_val is not None) and (rate_val <= 1500.0)
    per_credit_node = evaluator.add_custom_node(
        result=per_credit_ok,
        id="Per_Credit_Tuition_At_Most_1500",
        desc="Per-credit tuition is $1,500 or less.",
        parent=cost_node,
        critical=True,
    )

    # Total certificate cost provided
    total_cost_provided = bool(program.total_certificate_cost and program.total_certificate_cost.strip())
    total_cost_provided_node = evaluator.add_custom_node(
        result=total_cost_provided,
        id="Total_Certificate_Cost_Provided",
        desc="Total certificate cost is provided.",
        parent=cost_node,
        critical=True,
    )

    # Total cost consistent with rate * credits
    total_val = _parse_money(program.total_certificate_cost)
    credits_val = _parse_credit_hours_any(program.credit_hours)
    total_consistent = _cost_consistent(credits_val, rate_val, total_val)
    evaluator.add_custom_node(
        result=total_consistent,
        id="Total_Cost_Consistent_With_Rate_And_Credits",
        desc="Total certificate cost is arithmetically consistent with (credits × per-credit tuition) as reported in the solution.",
        parent=cost_node,
        critical=True,
    )

    # Record parsed numbers for transparency
    evaluator.add_custom_info(
        info={
            "parsed_per_credit_tuition": rate_val,
            "parsed_total_cost": total_val,
            "parsed_credits_any": credits_val,
            "parsed_credits_exact": _parse_credit_hours_exact(program.credit_hours),
        },
        info_type="parsed_numbers",
        info_name="parsed_numbers",
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
    Evaluate the agent's answer for the stackable data analytics certificate pathway task.
    """
    # Initialize evaluator (root node is non-critical by design)
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

    # Extract programs from the answer
    extraction: ProgramSet = await evaluator.extract(
        prompt=prompt_extract_programs(),
        template_class=ProgramSet,
        extraction_name="program_extraction",
    )

    # Build the top-level critical node that represents the rubric root
    complete_node = evaluator.add_parallel(
        id="Complete_Valid_Solution",
        desc="Identifies exactly one online graduate certificate program that satisfies all stated constraints and provides all required fields.",
        parent=root,
        critical=True,
    )

    # Check program count (must be exactly one)
    num_programs = len(extraction.programs)
    program_count_node = evaluator.add_custom_node(
        result=(num_programs == 1),
        id="Program_Count",
        desc="Exactly ONE program is identified (not multiple options).",
        parent=complete_node,
        critical=True,
    )

    # Select the first program for further checks (even if count != 1, other checks will get skipped via prerequisites)
    selected_program = extraction.programs[0] if extraction.programs else ProgramEntry()

    # Build and verify institution requirements
    await build_and_verify_institution_requirements(
        evaluator=evaluator,
        parent_node=complete_node,
        program=selected_program,
        program_count_node=program_count_node,
    )

    # Build and verify certificate requirements
    await build_and_verify_certificate_requirements(
        evaluator=evaluator,
        parent_node=complete_node,
        program=selected_program,
        program_count_node=program_count_node,
    )

    # Build and verify stackability requirements
    await build_and_verify_stackability_requirements(
        evaluator=evaluator,
        parent_node=complete_node,
        program=selected_program,
        program_count_node=program_count_node,
    )

    # Build and verify cost requirements
    await build_and_verify_cost_requirements(
        evaluator=evaluator,
        parent_node=complete_node,
        program=selected_program,
        program_count_node=program_count_node,
    )

    # Return evaluation summary
    return evaluator.get_summary()