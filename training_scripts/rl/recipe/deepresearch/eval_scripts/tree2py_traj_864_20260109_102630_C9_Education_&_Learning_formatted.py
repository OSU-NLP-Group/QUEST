import asyncio
import logging
from typing import Any, List, Dict, Optional

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "tx_r1_d1_housing_ratio_international_honors_credits_abroad_service_career_coop_language_teachered"
TASK_DESCRIPTION = """
Identify three public universities in Texas that simultaneously meet all of the following comprehensive criteria:

1. Classified as R1 (Research 1) universities in the 2025 Carnegie Classification, meaning they spend at least $50 million annually on research and award at least 70 research doctorates per year
2. Participate in NCAA Division I athletics programs
3. Offer on-campus housing capacity for at least 7,000 students in residence halls and apartments
4. Maintain a student-faculty ratio of 18:1 or lower for undergraduate programs
5. Enroll at least 5% international students in their total student population
6. Offer an honors program with a maximum GPA requirement of 3.5 or lower for admission
7. Require exactly 120 credit hours for bachelor's degree completion as a standard requirement
8. Provide study abroad programs requiring students to enroll in a minimum of 12 credit hours while abroad
9. Offer service learning or community engagement opportunities with defined hour requirements
10. Maintain career placement rates of at least 85% for graduates (employed or pursuing further education within six months)
11. Provide cooperative education or paid internship programs for academic credit
12. Have foreign language requirements at the 202-level (intermediate proficiency) or equivalent for certain bachelor's degrees
13. Offer teacher education programs with accreditation from CAEP, AAQEP, or state education agency approval

For each of the three universities you identify, provide:
- The university's complete official name
- Verification that it meets each of the 13 criteria listed above
- At least one reference URL supporting each major criterion
"""


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class UniversityData(BaseModel):
    # General
    official_name: Optional[str] = None

    # C2: Public university in Texas
    public_in_texas_sources: List[str] = Field(default_factory=list)

    # C1: R1 (2025 Carnegie Classification) with definition thresholds
    r1_2025_value: Optional[str] = None
    r1_2025_sources: List[str] = Field(default_factory=list)

    # C2: NCAA Division I participation
    ncaa_d1_sources: List[str] = Field(default_factory=list)

    # C3: On-campus housing capacity ≥ 7,000
    housing_capacity_value: Optional[str] = None
    housing_capacity_sources: List[str] = Field(default_factory=list)

    # C4: Student-faculty ratio ≤ 18:1
    ratio_value: Optional[str] = None
    ratio_sources: List[str] = Field(default_factory=list)

    # C5: International students ≥ 5%
    intl_pct_value: Optional[str] = None
    intl_pct_sources: List[str] = Field(default_factory=list)

    # C6: Honors program GPA requirement ≤ 3.5
    honors_gpa_value: Optional[str] = None
    honors_gpa_sources: List[str] = Field(default_factory=list)

    # C7: Bachelor's require exactly 120 credits
    bachelors_credits_value: Optional[str] = None
    bachelors_credits_sources: List[str] = Field(default_factory=list)

    # C8: Study abroad requires minimum 12 credits while abroad
    study_abroad_credits_value: Optional[str] = None
    study_abroad_credits_sources: List[str] = Field(default_factory=list)

    # C9: Service learning/community engagement hours requirements defined
    service_learning_hours_value: Optional[str] = None
    service_learning_hours_sources: List[str] = Field(default_factory=list)

    # C10: Career placement rate ≥ 85% within six months
    career_placement_rate_value: Optional[str] = None
    career_placement_rate_sources: List[str] = Field(default_factory=list)

    # C11: Cooperative education or paid internship for academic credit
    coop_or_paid_internship_sources: List[str] = Field(default_factory=list)

    # C12: Foreign language requirement at 202-level (intermediate) for certain degrees
    language_202_requirement_value: Optional[str] = None
    language_202_requirement_sources: List[str] = Field(default_factory=list)

    # C13: Teacher education accreditation (CAEP/AAQEP/state)
    teacher_ed_accreditation_value: Optional[str] = None
    teacher_ed_accreditation_sources: List[str] = Field(default_factory=list)


class UniversityExtraction(BaseModel):
    universities: List[UniversityData] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_universities() -> str:
    return """
    Extract all universities mentioned in the answer and, for each, collect the official name plus criterion-specific values and URLs. We will later evaluate only the first three distinct universities. Extract exactly the fields below for each university, using null for missing values and empty arrays for missing URLs. For URLs, extract actual URLs explicitly present in the answer (plain URLs or those inside markdown links). Do not invent any information.

    For each university, extract:
    - official_name: The complete official university name (string)
    - public_in_texas_sources: URLs proving it is a public institution in Texas (array of URLs)
    - r1_2025_value: Any text/value the answer provides about 2025 Carnegie R1 status or thresholds (string or null)
    - r1_2025_sources: URLs supporting R1 classification in 2025 and/or the stated thresholds (array of URLs)
    - ncaa_d1_sources: URLs showing NCAA Division I participation (array of URLs)
    - housing_capacity_value: Any text/value about on-campus housing capacity (string or null)
    - housing_capacity_sources: URLs supporting housing capacity ≥ 7000 (array of URLs)
    - ratio_value: Any text/value for student-faculty ratio (string or null)
    - ratio_sources: URLs supporting ratio ≤ 18:1 (array of URLs)
    - intl_pct_value: Any text/value for international student percentage (string or null)
    - intl_pct_sources: URLs supporting ≥ 5% international students (array of URLs)
    - honors_gpa_value: Any text/value for honors program GPA requirement (string or null)
    - honors_gpa_sources: URLs supporting maximum GPA requirement ≤ 3.5 (array of URLs)
    - bachelors_credits_value: Any text/value about bachelor's total required credits (string or null)
    - bachelors_credits_sources: URLs supporting exactly 120 required credits (array of URLs)
    - study_abroad_credits_value: Any text/value for credit-hour requirement while abroad (string or null)
    - study_abroad_credits_sources: URLs supporting minimum 12 credits while abroad (array of URLs)
    - service_learning_hours_value: Any text/value for service learning/community engagement hour requirements (string or null)
    - service_learning_hours_sources: URLs supporting defined hour requirements (array of URLs)
    - career_placement_rate_value: Any text/value for career placement rate within six months (string or null)
    - career_placement_rate_sources: URLs supporting ≥ 85% placement (array of URLs)
    - coop_or_paid_internship_sources: URLs supporting cooperative education or paid internships for academic credit (array of URLs)
    - language_202_requirement_value: Any text/value for foreign language requirement level (string or null)
    - language_202_requirement_sources: URLs supporting 202-level (intermediate) requirement for certain bachelor's degrees (array of URLs)
    - teacher_ed_accreditation_value: Any text/value for teacher education accreditation or approval (string or null)
    - teacher_ed_accreditation_sources: URLs supporting CAEP/AAQEP accreditation or state education agency approval (array of URLs)

    Return as:
    {
      "universities": [ { ...fields above... }, ... ]
    }

    Notes:
    - If more than three universities are mentioned, include all; the evaluator will filter to the first three distinct official_name entries.
    - Ensure you extract URLs exactly as shown in the answer. If the answer names a site without a URL, do not include it.
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def safe_name(name: Optional[str], fallback: str) -> str:
    return name.strip() if name else fallback


def has_sources(urls: Optional[List[str]]) -> bool:
    return bool(urls) and len(urls) > 0


async def add_sources_check(
    evaluator: Evaluator,
    parent_node,
    node_id: str,
    desc: str,
    urls: List[str],
) -> None:
    evaluator.add_custom_node(
        result=has_sources(urls),
        id=f"{node_id}_sources_provided",
        desc=f"{desc} - at least one supporting URL provided",
        parent=parent_node,
        critical=True
    )


async def add_claim_verify(
    evaluator: Evaluator,
    parent_node,
    node_id: str,
    desc: str,
    claim: str,
    urls: Optional[List[str]],
    additional_instruction: str,
) -> None:
    leaf = evaluator.add_leaf(
        id=node_id,
        desc=desc,
        parent=parent_node,
        critical=True
    )
    await evaluator.verify(
        claim=claim,
        node=leaf,
        sources=urls if urls else None,
        additional_instruction=additional_instruction
    )


# --------------------------------------------------------------------------- #
# University verification                                                     #
# --------------------------------------------------------------------------- #
async def verify_university(
    evaluator: Evaluator,
    parent_node,
    uni: UniversityData,
    uni_index: int
) -> None:
    # Create a parallel node for this university
    uni_node = evaluator.add_parallel(
        id=f"University_{uni_index+1}",
        desc=f"Evaluate the {'first' if uni_index==0 else 'second' if uni_index==1 else 'third'} university against all requirements",
        parent=parent_node,
        critical=False
    )

    uname = safe_name(uni.official_name, f"University_{uni_index+1}")

    # U{n}_Official_Name: check official name provided
    evaluator.add_custom_node(
        result=bool(uni.official_name) and uni.official_name.strip() != "",
        id=f"U{uni_index+1}_Official_Name",
        desc=f"Provide the complete official name of the {'first' if uni_index==0 else 'second' if uni_index==1 else 'third'} university",
        parent=uni_node,
        critical=True
    )

    # U{n}_Public_In_Texas
    await add_sources_check(
        evaluator, uni_node, f"U{uni_index+1}_Public_In_Texas",
        f"{uname} is a public institution located in Texas",
        uni.public_in_texas_sources
    )
    await add_claim_verify(
        evaluator, uni_node, f"U{uni_index+1}_Public_In_Texas",
        "First university is a public institution located in Texas" if uni_index==0
        else "Second university is a public institution located in Texas" if uni_index==1
        else "Third university is a public institution located in Texas",
        claim=f"{uname} is a public university located in the state of Texas.",
        urls=uni.public_in_texas_sources,
        additional_instruction="Verify that the institution is public (not private) and located in Texas using the provided official pages."
    )

    # U{n}_C1_R1_2025
    await add_sources_check(
        evaluator, uni_node, f"U{uni_index+1}_C1_R1_2025",
        f"{uname} meets R1 (2025 Carnegie Classification) definition thresholds with references",
        uni.r1_2025_sources
    )
    await add_claim_verify(
        evaluator, uni_node, f"U{uni_index+1}_C1_R1_2025",
        "Meets criterion 1: R1 in 2025 Carnegie Classification (as defined in prompt: ≥$50M research spend and ≥70 research doctorates/year) AND provides ≥1 supporting reference URL",
        claim=(
            f"{uname} is classified as R1 (very high research activity) in the 2025 Carnegie Classification. "
            "The R1 classification requires at least $50 million in annual research expenditures and at least 70 research doctorates per year."
        ),
        urls=uni.r1_2025_sources,
        additional_instruction=(
            "Confirm both that the university is listed as R1 for 2025 AND that the cited source(s) acknowledge the thresholds "
            "($50M research expenditures and ≥70 research doctorates/year) as part of this classification definition."
        )
    )

    # U{n}_C2_NCAA_D1
    await add_sources_check(
        evaluator, uni_node, f"U{uni_index+1}_C2_NCAA_D1",
        f"{uname} participates in NCAA Division I athletics with references",
        uni.ncaa_d1_sources
    )
    await add_claim_verify(
        evaluator, uni_node, f"U{uni_index+1}_C2_NCAA_D1",
        "Meets criterion 2: Participates in NCAA Division I athletics AND provides ≥1 supporting reference URL",
        claim=f"{uname} participates in NCAA Division I athletics.",
        urls=uni.ncaa_d1_sources,
        additional_instruction="Verify that the institution is an NCAA Division I member (not Division II/III). Use athletics or NCAA references."
    )

    # U{n}_C3_Housing_7000
    await add_sources_check(
        evaluator, uni_node, f"U{uni_index+1}_C3_Housing_7000",
        f"{uname} has on-campus housing capacity of at least 7,000 with references",
        uni.housing_capacity_sources
    )
    await add_claim_verify(
        evaluator, uni_node, f"U{uni_index+1}_C3_Housing_7000",
        "Meets criterion 3: On-campus housing capacity ≥7,000 students AND provides ≥1 supporting reference URL",
        claim=f"{uname} offers on-campus housing capacity for at least 7,000 students in residence halls and apartments.",
        urls=uni.housing_capacity_sources,
        additional_instruction=(
            "Confirm that on-campus housing capacity (residence halls and apartments) is 7,000 or higher. "
            f"If a specific capacity is given (e.g., {uni.housing_capacity_value}), check it meets the threshold."
        )
    )

    # U{n}_C4_Ratio_18_1
    await add_sources_check(
        evaluator, uni_node, f"U{uni_index+1}_C4_Ratio_18_1",
        f"{uname} maintains undergraduate student-faculty ratio ≤ 18:1 with references",
        uni.ratio_sources
    )
    await add_claim_verify(
        evaluator, uni_node, f"U{uni_index+1}_C4_Ratio_18_1",
        "Meets criterion 4: Student-faculty ratio ≤18:1 (as specified) AND provides ≥1 supporting reference URL",
        claim=f"The undergraduate student-faculty ratio at {uname} is 18:1 or lower.",
        urls=uni.ratio_sources,
        additional_instruction=(
            f"Verify that the student-faculty ratio is 18:1 or lower. If the answer provides a figure like '{uni.ratio_value}', "
            "use it for comparison."
        )
    )

    # U{n}_C5_International_5pct
    await add_sources_check(
        evaluator, uni_node, f"U{uni_index+1}_C5_International_5pct",
        f"{uname} has ≥ 5% international students with references",
        uni.intl_pct_sources
    )
    await add_claim_verify(
        evaluator, uni_node, f"U{uni_index+1}_C5_International_5pct",
        "Meets criterion 5: International students are ≥5% of total enrollment AND provides ≥1 supporting reference URL",
        claim=f"International students comprise at least 5% of total enrollment at {uname}.",
        urls=uni.intl_pct_sources,
        additional_instruction=(
            f"Confirm that the international student percentage is at least 5%. If provided (e.g., '{uni.intl_pct_value}'), "
            "use the stated figure to assess the threshold."
        )
    )

    # U{n}_C6_Honors_GPA_3_5_or_lower
    await add_sources_check(
        evaluator, uni_node, f"U{uni_index+1}_C6_Honors_GPA_3_5_or_lower",
        f"{uname} honors program GPA requirement is ≤ 3.5 with references",
        uni.honors_gpa_sources
    )
    await add_claim_verify(
        evaluator, uni_node, f"U{uni_index+1}_C6_Honors_GPA_3_5_or_lower",
        "Meets criterion 6: Offers an honors program with maximum GPA requirement for admission ≤3.5 AND provides ≥1 supporting reference URL",
        claim=f"The honors program at {uname} sets a maximum GPA requirement for admission of 3.5 or lower.",
        urls=uni.honors_gpa_sources,
        additional_instruction=(
            f"Verify that the stated honors GPA requirement (e.g., '{uni.honors_gpa_value}') does not exceed 3.5. "
            "If a range or multiple requirements exist, the maximum must be ≤ 3.5."
        )
    )

    # U{n}_C7_Bachelors_120_hours_exact
    await add_sources_check(
        evaluator, uni_node, f"U{uni_index+1}_C7_Bachelors_120_hours_exact",
        f"{uname} requires exactly 120 credits for bachelor's degrees with references",
        uni.bachelors_credits_sources
    )
    await add_claim_verify(
        evaluator, uni_node, f"U{uni_index+1}_C7_Bachelors_120_hours_exact",
        "Meets criterion 7: Standard bachelor's completion requires exactly 120 credit hours AND provides ≥1 supporting reference URL",
        claim=f"The standard bachelor's degree at {uname} requires exactly 120 credit hours to complete.",
        urls=uni.bachelors_credits_sources,
        additional_instruction=(
            f"Confirm that the standard requirement is exactly 120 credits (not a range). If provided value is '{uni.bachelors_credits_value}', "
            "ensure it states 120 credits as the standard."
        )
    )

    # U{n}_C8_Study_Abroad_12_hours_min
    await add_sources_check(
        evaluator, uni_node, f"U{uni_index+1}_C8_Study_Abroad_12_hours_min",
        f"{uname} requires ≥ 12 credit hours while abroad with references",
        uni.study_abroad_credits_sources
    )
    await add_claim_verify(
        evaluator, uni_node, f"U{uni_index+1}_C8_Study_Abroad_12_hours_min",
        "Meets criterion 8: Study abroad programs require enrollment in at least 12 credit hours while abroad AND provides ≥1 supporting reference URL",
        claim=f"Study abroad programs at {uname} require students to enroll in a minimum of 12 credit hours while abroad.",
        urls=uni.study_abroad_credits_sources,
        additional_instruction=(
            f"Verify that study abroad policies specify at least 12 credit hours during the term abroad. "
            f"If the answer includes '{uni.study_abroad_credits_value}', use it."
        )
    )

    # U{n}_C9_Service_Learning_hours_defined
    await add_sources_check(
        evaluator, uni_node, f"U{uni_index+1}_C9_Service_Learning_hours_defined",
        f"{uname} defines service learning/community engagement hour requirements with references",
        uni.service_learning_hours_sources
    )
    await add_claim_verify(
        evaluator, uni_node, f"U{uni_index+1}_C9_Service_Learning_hours_defined",
        "Meets criterion 9: Service learning/community engagement opportunities include defined hour requirements AND provides ≥1 supporting reference URL",
        claim=f"{uname} offers service learning or community engagement opportunities with specified, defined hour requirements.",
        urls=uni.service_learning_hours_sources,
        additional_instruction=(
            f"Confirm that the program descriptions include explicit hour requirements (e.g., '{uni.service_learning_hours_value}')."
        )
    )

    # U{n}_C10_Career_placement_85pct
    await add_sources_check(
        evaluator, uni_node, f"U{uni_index+1}_C10_Career_placement_85pct",
        f"{uname} reports ≥ 85% career placement within six months with references",
        uni.career_placement_rate_sources
    )
    await add_claim_verify(
        evaluator, uni_node, f"U{uni_index+1}_C10_Career_placement_85pct",
        "Meets criterion 10: Career placement rate ≥85% within six months (employed or further education) AND provides ≥1 supporting reference URL",
        claim=f"The career placement rate for graduates of {uname} (employed or further education within six months) is at least 85%.",
        urls=uni.career_placement_rate_sources,
        additional_instruction=(
            f"Verify that the reported rate meets or exceeds 85%. If a value like '{uni.career_placement_rate_value}' is given, compare it."
        )
    )

    # U{n}_C11_Coop_or_paid_internship_for_credit
    await add_sources_check(
        evaluator, uni_node, f"U{uni_index+1}_C11_Coop_or_paid_internship_for_credit",
        f"{uname} offers cooperative education or paid internships for academic credit with references",
        uni.coop_or_paid_internship_sources
    )
    await add_claim_verify(
        evaluator, uni_node, f"U{uni_index+1}_C11_Coop_or_paid_internship_for_credit",
        "Meets criterion 11: Cooperative education or paid internship programs for academic credit AND provides ≥1 supporting reference URL",
        claim=f"{uname} offers cooperative education or paid internship programs that grant academic credit.",
        urls=uni.coop_or_paid_internship_sources,
        additional_instruction="Verify that co-op or paid internship programs are offered for academic credit (not just unpaid, non-credit experiences)."
    )

    # U{n}_C12_Language_202_level
    await add_sources_check(
        evaluator, uni_node, f"U{uni_index+1}_C12_Language_202_level",
        f"{uname} requires 202-level (intermediate) foreign language for certain bachelor's degrees with references",
        uni.language_202_requirement_sources
    )
    await add_claim_verify(
        evaluator, uni_node, f"U{uni_index+1}_C12_Language_202_level",
        "Meets criterion 12: Foreign language requirements at the 202-level (intermediate) or equivalent for certain bachelor's degrees AND provides ≥1 supporting reference URL",
        claim=f"Certain bachelor's degrees at {uname} require foreign language proficiency at the 202-level (intermediate) or an equivalent requirement.",
        urls=uni.language_202_requirement_sources,
        additional_instruction=(
            "Confirm that degree requirements explicitly state a 202-level language requirement or equivalent intermediate proficiency "
            f"(if a value like '{uni.language_202_requirement_value}' is provided, use it)."
        )
    )

    # U{n}_C13_Teacher_ed_accreditation
    await add_sources_check(
        evaluator, uni_node, f"U{uni_index+1}_C13_Teacher_ed_accreditation",
        f"{uname} teacher education accredited by CAEP/AAQEP or has state approval with references",
        uni.teacher_ed_accreditation_sources
    )
    await add_claim_verify(
        evaluator, uni_node, f"U{uni_index+1}_C13_Teacher_ed_accreditation",
        "Meets criterion 13: Teacher education programs accredited by CAEP or AAQEP or have state education agency approval (as specified) AND provides ≥1 supporting reference URL",
        claim=(
            f"The teacher education programs at {uname} are accredited by CAEP or AAQEP, "
            "or have approval from the state education agency."
        ),
        urls=uni.teacher_ed_accreditation_sources,
        additional_instruction=(
            f"Verify accreditation or approval status. If the answer includes '{uni.teacher_ed_accreditation_value}', "
            "use it to confirm CAEP/AAQEP accreditation or state approval."
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
) -> Dict:
    """
    Evaluate the answer for the Texas public universities multi-criteria verification task.
    """
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.SEQUENTIAL,  # Sequential: gate universities after the initial distinct-universities check
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
    extracted: UniversityExtraction = await evaluator.extract(
        prompt=prompt_extract_universities(),
        template_class=UniversityExtraction,
        extraction_name="universities_extraction"
    )

    # Prepare list: filter to the first three distinct official names
    # Build unique list maintaining order
    seen = set()
    filtered_unis: List[UniversityData] = []
    for uni in extracted.universities:
        name = (uni.official_name or "").strip()
        if not name:
            continue
        key = name.lower()
        if key in seen:
            continue
        seen.add(key)
        filtered_unis.append(uni)
        if len(filtered_unis) == 3:
            break

    # Record custom info for visibility
    evaluator.add_custom_info(
        {
            "total_universities_extracted": len(extracted.universities),
            "distinct_universities_count": len(seen),
            "selected_universities": [u.official_name for u in filtered_unis]
        },
        info_type="extraction_stats",
        info_name="selection_summary"
    )

    # Critical gating: Three distinct universities provided (we evaluate first three when available)
    evaluator.add_custom_node(
        result=len(filtered_unis) >= 3,
        id="Three_Distinct_Universities_Provided",
        desc="Response identifies exactly three distinct universities (we evaluate the first three distinct ones)",
        parent=root,
        critical=True
    )

    # Build and verify for each of the three universities (if available)
    for idx in range(3):
        # Create University parent parallel node (non-critical to allow partial credit across universities)
        uni_parent = evaluator.add_parallel(
            id=f"University_{idx+1}",
            desc=f"Evaluate the {'first' if idx==0 else 'second' if idx==1 else 'third'} university against all requirements",
            parent=root,
            critical=False
        )

        # If we don't have enough universities, add placeholder nodes with failed name check to make structure explicit
        if idx >= len(filtered_unis):
            # Explicitly fail missing university entries
            evaluator.add_custom_node(
                result=False,
                id=f"U{idx+1}_Official_Name",
                desc=f"Provide the complete official name of the {'first' if idx==0 else 'second' if idx==1 else 'third'} university",
                parent=uni_parent,
                critical=True
            )
            # Add minimal placeholders for each criterion so the tree is complete (skipped via preconditions)
            placeholder_claims = [
                ("U{n}_Public_In_Texas", "University is a public institution located in Texas"),
                ("U{n}_C1_R1_2025", "R1 in 2025 Carnegie Classification with defined thresholds"),
                ("U{n}_C2_NCAA_D1", "Participates in NCAA Division I athletics"),
                ("U{n}_C3_Housing_7000", "On-campus housing capacity ≥ 7,000"),
                ("U{n}_C4_Ratio_18_1", "Student-faculty ratio ≤ 18:1"),
                ("U{n}_C5_International_5pct", "International students ≥ 5%"),
                ("U{n}_C6_Honors_GPA_3_5_or_lower", "Honors program maximum GPA requirement ≤ 3.5"),
                ("U{n}_C7_Bachelors_120_hours_exact", "Bachelor's requires exactly 120 credits"),
                ("U{n}_C8_Study_Abroad_12_hours_min", "Study abroad minimum 12 credits while abroad"),
                ("U{n}_C9_Service_Learning_hours_defined", "Service learning/community engagement hour requirements defined"),
                ("U{n}_C10_Career_placement_85pct", "Career placement rate ≥ 85% within six months"),
                ("U{n}_C11_Coop_or_paid_internship_for_credit", "Co-op or paid internship for academic credit"),
                ("U{n}_C12_Language_202_level", "Foreign language 202-level requirement for certain degrees"),
                ("U{n}_C13_Teacher_ed_accreditation", "Teacher education accredited (CAEP/AAQEP) or state approval")
            ]
            for node_id_fmt, desc in placeholder_claims:
                node_id = node_id_fmt.format(n=idx+1)
                leaf = evaluator.add_leaf(
                    id=node_id,
                    desc=desc,
                    parent=uni_parent,
                    critical=True
                )
                await evaluator.verify(
                    claim="This is a placeholder claim. No sources provided.",
                    node=leaf,
                    sources=None,
                    additional_instruction="No verification possible due to missing university; mark as failed."
                )
            continue

        # Verify actual university
        await verify_university(evaluator, uni_parent, filtered_unis[idx], idx)

    return evaluator.get_summary()