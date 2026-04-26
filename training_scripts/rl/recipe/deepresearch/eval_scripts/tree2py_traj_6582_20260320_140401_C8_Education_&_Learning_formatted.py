import asyncio
import logging
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "qualifying_us_school_districts_3"
TASK_DESCRIPTION = """
Identify three public school districts in the United States that meet ALL of the following criteria:

1. The district must currently enroll more than 90,000 students
2. The district must be located in one of these three states: Virginia, Georgia, or Massachusetts
3. The district must operate at least 100 schools and centers (including elementary, middle, and high schools)
4. The state where the district is located must require students to complete a minimum of 22 credits to earn a standard high school diploma
5. The state where the district is located must require at least 180 days of instruction per school year
6. The district's current superintendent must hold appropriate professional certification in educational leadership or administration
7. The district must serve a student population where no single racial or ethnic group comprises more than 50% of total enrollment (demonstrating demographic diversity)
8. The district must provide special education services to at least 15% of its student population
9. The district must have achieved an on-time graduation rate of at least 85% for its most recent graduating class
10. The district must operate all three types of schools: elementary schools, middle schools, and high schools

For each district you identify, provide: (a) the district name, (b) the state, (c) current enrollment figure with source URL, (d) number of schools with source URL, (e) superintendent name and confirmation of certification with source URL, (f) demographic breakdown with source URL, (g) special education percentage with source URL, (h) graduation rate with source URL, and (i) confirmation that all three school types are operated with source URL.
"""


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class DistrictItem(BaseModel):
    # Basic identity
    name: Optional[str] = None
    state: Optional[str] = None

    # Enrollment
    enrollment: Optional[str] = None
    enrollment_urls: List[str] = Field(default_factory=list)

    # Number of schools
    schools_count: Optional[str] = None
    schools_urls: List[str] = Field(default_factory=list)

    # Superintendent
    superintendent_name: Optional[str] = None
    superintendent_certification: Optional[str] = None  # e.g., certificate/license details or statement
    superintendent_urls: List[str] = Field(default_factory=list)

    # Demographics
    demographics_text: Optional[str] = None  # free-text breakdown as presented in the answer
    demographics_urls: List[str] = Field(default_factory=list)

    # Special education
    special_ed_percentage: Optional[str] = None
    special_ed_urls: List[str] = Field(default_factory=list)

    # Graduation
    graduation_rate: Optional[str] = None
    graduation_urls: List[str] = Field(default_factory=list)

    # School types
    school_types: List[str] = Field(default_factory=list)  # expected tokens like "elementary", "middle", "high"
    school_types_urls: List[str] = Field(default_factory=list)

    # State policies
    state_credits_requirement_text: Optional[str] = None
    state_credits_urls: List[str] = Field(default_factory=list)

    state_instruction_days_text: Optional[str] = None
    state_instruction_days_urls: List[str] = Field(default_factory=list)


class DistrictsExtraction(BaseModel):
    districts: List[DistrictItem] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_districts() -> str:
    return """
Extract up to three public school districts from the answer that the writer claims meet ALL of the required criteria. For each district, extract exactly the following fields:

- name: The district name (string)
- state: The U.S. state of the district (string; use full state name if present, otherwise the two-letter postal abbreviation)
- enrollment: The current enrollment figure as stated (string; keep formatting, e.g., "181,000" or "~180k")
- enrollment_urls: An array of the URL(s) explicitly cited for enrollment
- schools_count: The stated number of schools and centers (string; keep formatting)
- schools_urls: An array of the URL(s) explicitly cited for number of schools
- superintendent_name: The current superintendent's full name (string)
- superintendent_certification: The stated certification/licensure detail or a short statement confirming appropriate professional certification in educational leadership/administration (string)
- superintendent_urls: An array of the URL(s) explicitly cited for superintendent identity/certification
- demographics_text: A short, verbatim or close paraphrase summary of the racial/ethnic breakdown as stated (string)
- demographics_urls: An array of the URL(s) explicitly cited for demographics
- special_ed_percentage: The stated percentage of students receiving special education services (string; e.g., "16%" or "0.16")
- special_ed_urls: An array of the URL(s) explicitly cited for the special education figure
- graduation_rate: The stated most recent on-time graduation rate (string; e.g., "88%" or "0.88")
- graduation_urls: An array of the URL(s) explicitly cited for the graduation rate
- school_types: An array of strings indicating which school types the district operates; use lowercase tokens from this set only: ["elementary", "middle", "high"]. Include all that are explicitly stated or clearly implied in the answer.
- school_types_urls: An array of the URL(s) explicitly cited for school types operated
- state_credits_requirement_text: A brief statement of the state's minimum standard diploma credit requirement as presented in the answer (string)
- state_credits_urls: An array of the URL(s) explicitly cited for the state's graduation credit requirement
- state_instruction_days_text: A brief statement of the state's minimum required instructional days per school year as presented (string)
- state_instruction_days_urls: An array of the URL(s) explicitly cited for the instructional days requirement

IMPORTANT:
- Only extract URLs that are explicitly present in the answer text. Do not invent URLs.
- Normalize school type tokens to exactly: "elementary", "middle", "high".
- If any field is missing in the answer for a district, set it to null (for strings) or an empty array (for URL arrays).
- Return up to the first three districts mentioned in the answer, in the same order they appear.
"""


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _ordinal(n: int) -> str:
    return ["First", "Second", "Third"][n] if 0 <= n < 3 else f"#{n+1}"


def _non_empty_str(s: Optional[str]) -> bool:
    return bool(s and str(s).strip())


def _merge_sources(*lists: List[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for lst in lists:
        for u in lst or []:
            u_norm = (u or "").strip()
            if not u_norm:
                continue
            if u_norm not in seen:
                seen.add(u_norm)
                out.append(u_norm)
    return out


def _safe(s: Optional[str], fallback: str = "") -> str:
    return s if _non_empty_str(s) else fallback


# --------------------------------------------------------------------------- #
# Verification for a single district                                          #
# --------------------------------------------------------------------------- #
async def verify_one_district(
    evaluator: Evaluator,
    parent_node,
    district: DistrictItem,
    index: int,
) -> None:
    # Create district node (parallel aggregation; allow partial credit across districts at root)
    district_node = evaluator.add_parallel(
        id=f"district_{index+1}",
        desc=f"{_ordinal(index)} qualifying school district meeting all ten criteria",
        parent=parent_node,
        critical=False,
    )

    # 1) Enrollment > 90,000 (Critical)
    node_enroll = evaluator.add_leaf(
        id=f"district_{index+1}_enrollment_over_90000",
        desc="The district currently enrolls more than 90,000 students",
        parent=district_node,
        critical=True,
    )
    enroll_claim = (
        f"The district {_safe(district.name, 'the district')} currently enrolls more than 90,000 students; "
        f"the reported enrollment is {_safe(district.enrollment, 'unknown')}."
    )
    await evaluator.verify(
        claim=enroll_claim,
        node=node_enroll,
        sources=district.enrollment_urls,
        additional_instruction="Verify the enrollment figure on the cited page(s). "
                               "If the source shows a number ≥ 90,001 (or equivalent wording like 'over 90,000'), consider this satisfied. "
                               "Allow reasonable rounding/approximation (e.g., '~180k'). If the webpage does not support it, mark as not supported.",
    )

    # 2) State location is VA/GA/MA (Critical)
    node_state = evaluator.add_leaf(
        id=f"district_{index+1}_state_location_verified",
        desc="The district is located in Virginia, Georgia, or Massachusetts",
        parent=district_node,
        critical=True,
    )
    state_claim = (
        f"The district {_safe(district.name, 'the district')} is located in {_safe(district.state, 'an unknown state')}, "
        f"which must be one of: Virginia, Georgia, or Massachusetts."
    )
    state_sources = _merge_sources(
        district.enrollment_urls,
        district.schools_urls,
        district.superintendent_urls,
        district.school_types_urls,
    )
    await evaluator.verify(
        claim=state_claim,
        node=node_state,
        sources=state_sources,
        additional_instruction="Confirm the district's state from the provided webpages. "
                               "Treat 'VA' as Virginia, 'GA' as Georgia, and 'MA' as Massachusetts. "
                               "If the district is not clearly in one of these three states, mark as not supported.",
    )

    # 3) School count >= 100 (Critical)
    node_schools = evaluator.add_leaf(
        id=f"district_{index+1}_school_count_ge_100",
        desc="The district operates at least 100 schools and centers",
        parent=district_node,
        critical=True,
    )
    schools_claim = (
        f"The district {_safe(district.name, 'the district')} operates at least 100 schools and centers; "
        f"the reported number is {_safe(district.schools_count, 'unknown')}."
    )
    await evaluator.verify(
        claim=schools_claim,
        node=node_schools,
        sources=district.schools_urls,
        additional_instruction="Verify the total number of schools/centers from the cited webpages. "
                               "If the page shows ≥ 100 (or equivalent wording such as 'over 100'), consider satisfied.",
    )

    # 4) State requires ≥ 22 credits (Critical)
    node_credits = evaluator.add_leaf(
        id=f"district_{index+1}_state_credits_ge_22",
        desc="The state where the district is located requires a minimum of 22 credits for a standard high school diploma",
        parent=district_node,
        critical=True,
    )
    credits_claim = (
        f"The state of {_safe(district.state, 'the district state')} requires at least 22 credits to earn a standard high school diploma."
    )
    await evaluator.verify(
        claim=credits_claim,
        node=node_credits,
        sources=district.state_credits_urls,
        additional_instruction="Verify on the provided state-level policy page(s) that the minimum for a standard high school diploma is ≥ 22 credits. "
                               "If the page shows 22 or more units/credits required, consider this satisfied.",
    )

    # 5) State requires ≥ 180 instructional days (Critical)
    node_days = evaluator.add_leaf(
        id=f"district_{index+1}_state_days_ge_180",
        desc="The state where the district is located requires at least 180 days of instruction per school year",
        parent=district_node,
        critical=True,
    )
    days_claim = (
        f"The state of {_safe(district.state, 'the district state')} requires at least 180 instructional days per school year."
    )
    await evaluator.verify(
        claim=days_claim,
        node=node_days,
        sources=district.state_instruction_days_urls,
        additional_instruction="Verify on the provided state policy page(s) that the minimum instructional days per school year is at least 180.",
    )

    # 6) Superintendent name provided (Non-critical existence check)
    evaluator.add_custom_node(
        result=_non_empty_str(district.superintendent_name),
        id=f"district_{index+1}_superintendent_name_provided",
        desc="The current superintendent's name is identified and provided",
        parent=district_node,
        critical=False,
    )

    # 7) Superintendent holds appropriate certification (Critical)
    node_sup_cert = evaluator.add_leaf(
        id=f"district_{index+1}_superintendent_cert_verified",
        desc="The superintendent holds appropriate professional certification in educational leadership or administration",
        parent=district_node,
        critical=True,
    )
    sup_claim = (
        f"The current superintendent {_safe(district.superintendent_name, 'of the district')} holds an appropriate professional certification "
        f"in educational leadership/administration (e.g., superintendent license/certificate). "
        f"Stated details: {_safe(district.superintendent_certification, 'not specified')}."
    )
    await evaluator.verify(
        claim=sup_claim,
        node=node_sup_cert,
        sources=district.superintendent_urls,
        additional_instruction="Confirm from the provided page(s) that the superintendent has a relevant certification, "
                               "such as a superintendent license or administrator/leadership certificate in the state.",
    )

    # 8) Demographic diversity: no single group > 50% (Critical)
    node_diversity = evaluator.add_leaf(
        id=f"district_{index+1}_demographic_diversity_verified",
        desc="No single racial or ethnic group comprises more than 50% of total enrollment",
        parent=district_node,
        critical=True,
    )
    diversity_claim = (
        f"Based on the cited demographic breakdown for {_safe(district.name, 'the district')} "
        f"({_safe(district.demographics_text, 'no breakdown provided')}), "
        f"no single racial or ethnic group comprises more than 50% of total enrollment."
    )
    await evaluator.verify(
        claim=diversity_claim,
        node=node_diversity,
        sources=district.demographics_urls,
        additional_instruction="Examine any demographic table/figures in the provided source(s). "
                               "If any group is clearly > 50%, mark as not supported.",
    )

    # 9) Special education ≥ 15% (Critical)
    node_sped = evaluator.add_leaf(
        id=f"district_{index+1}_special_ed_ge_15",
        desc="The district provides special education services to at least 15% of its student population",
        parent=district_node,
        critical=True,
    )
    sped_claim = (
        f"In {_safe(district.name, 'the district')}, the share of students receiving special education services is "
        f"{_safe(district.special_ed_percentage, 'unknown')}, which is at least 15%."
    )
    await evaluator.verify(
        claim=sped_claim,
        node=node_sped,
        sources=district.special_ed_urls,
        additional_instruction="Verify the special education percentage on the provided source(s). "
                               "If the percentage is ≥ 15% (or equivalent), consider this satisfied.",
    )

    # 10) Graduation rate ≥ 85% (Critical)
    node_grad = evaluator.add_leaf(
        id=f"district_{index+1}_graduation_rate_ge_85",
        desc="The district has achieved an on-time graduation rate of at least 85% for its most recent graduating class",
        parent=district_node,
        critical=True,
    )
    grad_claim = (
        f"The most recent on-time graduation rate for {_safe(district.name, 'the district')} is "
        f"{_safe(district.graduation_rate, 'unknown')}, which is at least 85%."
    )
    await evaluator.verify(
        claim=grad_claim,
        node=node_grad,
        sources=district.graduation_urls,
        additional_instruction="Verify the most recent on-time graduation rate on the provided source(s). "
                               "If the rate is ≥ 85% (or equivalent), consider this satisfied.",
    )

    # 11) Elementary schools present (Critical)
    node_elem = evaluator.add_leaf(
        id=f"district_{index+1}_elementary_present",
        desc="The district operates elementary schools",
        parent=district_node,
        critical=True,
    )
    elem_claim = f"The district {_safe(district.name, 'the district')} operates elementary schools."
    await evaluator.verify(
        claim=elem_claim,
        node=node_elem,
        sources=district.school_types_urls,
        additional_instruction="Confirm from the provided source(s) (e.g., school directory) that the district operates elementary schools.",
    )

    # 12) Middle schools present (Critical)
    node_mid = evaluator.add_leaf(
        id=f"district_{index+1}_middle_present",
        desc="The district operates middle schools",
        parent=district_node,
        critical=True,
    )
    mid_claim = f"The district {_safe(district.name, 'the district')} operates middle schools."
    await evaluator.verify(
        claim=mid_claim,
        node=node_mid,
        sources=district.school_types_urls,
        additional_instruction="Confirm from the provided source(s) that the district operates middle schools.",
    )

    # 13) High schools present (Critical)
    node_high = evaluator.add_leaf(
        id=f"district_{index+1}_high_present",
        desc="The district operates high schools",
        parent=district_node,
        critical=True,
    )
    high_claim = f"The district {_safe(district.name, 'the district')} operates high schools."
    await evaluator.verify(
        claim=high_claim,
        node=node_high,
        sources=district.school_types_urls,
        additional_instruction="Confirm from the provided source(s) that the district operates high schools.",
    )


# --------------------------------------------------------------------------- #
# Main evaluation entry                                                       #
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
    # Initialize evaluator with a parallel root (three districts evaluated independently)
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

    # Root node (non-critical): evaluation of up to three public school districts
    root_node = evaluator.add_parallel(
        id="Qualifying_School_Districts",
        desc="Evaluation of up to three public school districts that meet all specified criteria regarding enrollment, location, school count, state requirements, superintendent qualifications, demographic diversity, special education services, graduation rates, and school type variety",
        parent=evaluator.root,
        critical=False,
    )

    # 1) Extract up to 3 districts from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_districts(),
        template_class=DistrictsExtraction,
        extraction_name="districts_extraction",
    )

    # Ensure exactly 3 entries (pad with empty if fewer)
    districts: List[DistrictItem] = list(extracted.districts[:3])
    while len(districts) < 3:
        districts.append(DistrictItem())

    # 2) Build verification subtrees for each district
    for idx in range(3):
        await verify_one_district(
            evaluator=evaluator,
            parent_node=root_node,
            district=districts[idx],
            index=idx,
        )

    # 3) Return evaluation summary
    return evaluator.get_summary()