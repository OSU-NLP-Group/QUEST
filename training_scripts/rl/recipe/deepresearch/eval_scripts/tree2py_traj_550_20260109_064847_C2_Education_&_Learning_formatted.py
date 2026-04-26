import asyncio
import logging
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


TASK_ID = "cs_ms_ta_requirements_fall_2026"
TASK_DESCRIPTION = """
I am an international student planning to apply for Master's programs in Computer Science in the United States for Fall 2026. Due to my academic profile and financial needs, I am looking for a program that meets very specific criteria.

My profile:
- Undergraduate GPA: 3.0 on a 4.0 scale
- TOEFL iBT scores: Total 90, with Speaking subsection score of 24
- I need financial support through a Teaching Assistantship (TA) position
- Due to my work commitments, I can only enroll in 9 graduate credit hours per semester

Please identify ONE U.S. university that offers a Master's degree program in Computer Science that meets ALL of the following requirements:

1. The program must accept a minimum undergraduate GPA of 3.0 on a 4.0 scale for admission consideration.
2. The program must accept a TOEFL iBT total score of 90 or lower for general graduate admission.
3. The program must offer Teaching Assistantship (TA) positions as a form of financial support for graduate students.
4. For TA eligibility, the program must accept a TOEFL iBT speaking subsection score of 24 or lower.
5. For TA positions, the program must allow students to maintain assistantship eligibility while enrolled in 9 or fewer graduate credit hours per semester.

For the identified program, please provide:
- The university name
- The specific program name
- Official documentation or web page references that verify each of the five requirements listed above
"""


# ----------------------------- Data Models ---------------------------------- #
class ProgramExtraction(BaseModel):
    university_name: Optional[str] = None
    program_name: Optional[str] = None

    # General program or university official pages (program overview, department, grad school)
    general_program_urls: List[str] = Field(default_factory=list)

    # Requirement 1: Minimum GPA (≤ 3.0)
    gpa_requirement_text: Optional[str] = None
    gpa_urls: List[str] = Field(default_factory=list)

    # Requirement 2: TOEFL iBT total (≤ 90)
    toefl_total_requirement_text: Optional[str] = None
    toefl_total_urls: List[str] = Field(default_factory=list)

    # Requirement 3: TA positions offered
    ta_availability_text: Optional[str] = None
    ta_urls: List[str] = Field(default_factory=list)

    # Requirement 4: TA eligibility TOEFL Speaking (≤ 24)
    ta_speaking_requirement_text: Optional[str] = None
    ta_speaking_urls: List[str] = Field(default_factory=list)

    # Requirement 5: TA enrollment eligibility with ≤ 9 credits/semester
    ta_enrollment_requirement_text: Optional[str] = None
    ta_enrollment_urls: List[str] = Field(default_factory=list)

    # Optional location hint extracted from answer (e.g., "United States", state/city)
    location_text: Optional[str] = None


# --------------------------- Extraction Prompt ------------------------------ #
def prompt_extract_program() -> str:
    return """
    Extract exactly ONE identified graduate program (choose the first one if multiple are mentioned) from the answer and provide both the core identifiers and the verification sources for each of the five enumerated requirements.

    Fields to extract:
    1. university_name: The full name of the U.S. university.
    2. program_name: The specific name of the Master's program in Computer Science (e.g., "MS in Computer Science", "Master of Computer Science", "MEng in Computer Science").
    3. general_program_urls: All official URLs provided in the answer that describe the program/department/university (e.g., the program page, department page, graduate school page). Extract only URLs explicitly present in the answer.
    4. gpa_requirement_text: The text snippet or phrase (as it appears in the answer) that describes the minimum GPA requirement (if present).
    5. gpa_urls: All URLs cited for verifying the minimum GPA requirement.
    6. toefl_total_requirement_text: The text snippet or phrase (as it appears in the answer) that describes the TOEFL iBT total score requirement (if present).
    7. toefl_total_urls: All URLs cited for verifying the TOEFL iBT total score requirement.
    8. ta_availability_text: The text snippet or phrase (as it appears in the answer) about TA positions availability (if present).
    9. ta_urls: All URLs cited for verifying TA positions availability.
    10. ta_speaking_requirement_text: The text snippet or phrase (as it appears in the answer) describing TOEFL iBT speaking subscore required for TA eligibility (if present).
    11. ta_speaking_urls: All URLs cited for verifying TA speaking score requirement.
    12. ta_enrollment_requirement_text: The text snippet or phrase (as it appears in the answer) describing enrollment credit hour conditions for assistantship eligibility (if present).
    13. ta_enrollment_urls: All URLs cited for verifying TA enrollment requirement.
    14. location_text: Any location descriptor mentioned in the answer (e.g., "United States", a city/state), if present.

    IMPORTANT URL EXTRACTION RULES:
    - Extract only complete, valid URLs explicitly present in the answer (plain or markdown link forms).
    - If a URL is missing a protocol, prepend "http://".
    - Do not invent URLs.

    Return all fields. If any field is missing in the answer, set it to null (strings) or an empty list (for urls).
    """


# --------------------------- Verification Helpers --------------------------- #
async def verify_basic_admission_requirements(
    evaluator: Evaluator,
    parent_node,
    extracted: ProgramExtraction,
) -> None:
    basic_node = evaluator.add_parallel(
        id="Basic_Admission_Requirements",
        desc="Verify the program meets fundamental admission requirements",
        parent=parent_node,
        critical=True,
    )

    # Masters degree in Computer Science
    masters_leaf = evaluator.add_leaf(
        id="Masters_Degree_in_Computer_Science",
        desc="The program must be a Master's degree in Computer Science",
        parent=basic_node,
        critical=True,
    )
    masters_claim = (
        f"The identified program '{extracted.program_name or 'UNKNOWN PROGRAM'}' "
        f"is a Master's degree program in Computer Science offered by "
        f"{extracted.university_name or 'UNKNOWN UNIVERSITY'}."
    )
    await evaluator.verify(
        claim=masters_claim,
        node=masters_leaf,
        sources=extracted.general_program_urls,
        additional_instruction=(
            "Verify from the official program/department/university page(s) that the program is a Master's "
            "degree in Computer Science. Accept equivalent naming like 'MS in CS', 'M.S. in Computer Science', "
            "'Master of Computer Science (MCS)', or 'MEng in Computer Science' if clearly a graduate master's "
            "program within Computer Science."
        ),
    )

    # US University
    us_leaf = evaluator.add_leaf(
        id="US_University",
        desc="The program must be offered by a university in the United States",
        parent=basic_node,
        critical=True,
    )
    us_claim = (
        f"The university '{extracted.university_name or 'UNKNOWN UNIVERSITY'}' is located in the United States."
    )
    await evaluator.verify(
        claim=us_claim,
        node=us_leaf,
        sources=extracted.general_program_urls,
        additional_instruction=(
            "Confirm the university is a US institution using the official page(s)—look for a US address, "
            "state/city, or other location indicators. Pages on .edu domains and explicit mention of US location "
            "are strong signals. If the official page makes the US location clear, pass."
        ),
    )

    # Minimum GPA requirement (≤ 3.0)
    gpa_leaf = evaluator.add_leaf(
        id="Minimum_GPA_Requirement",
        desc="The program must accept a minimum undergraduate GPA of 3.0 or lower on a 4.0 scale",
        parent=basic_node,
        critical=True,
    )
    gpa_claim = (
        "The minimum undergraduate GPA requirement for admission consideration is at most 3.0 on a 4.0 scale "
        "(i.e., ≤ 3.0). Statements such as 'minimum GPA 3.0' satisfy this."
    )
    await evaluator.verify(
        claim=gpa_claim,
        node=gpa_leaf,
        sources=extracted.gpa_urls,
        additional_instruction=(
            "Check the official admission page(s). Accept if the minimum GPA requirement is listed as 3.0 or lower "
            "on a 4.0 scale. Department-level or graduate school-level policy are acceptable as long as they apply "
            "to the program."
        ),
    )

    # TOEFL iBT total requirement (≤ 90)
    toefl_total_leaf = evaluator.add_leaf(
        id="TOEFL_Total_Score_Requirement",
        desc="The program must accept a TOEFL iBT total score of 90 or lower for general admission",
        parent=basic_node,
        critical=True,
    )
    toefl_total_claim = (
        "For general graduate admission, the minimum required TOEFL iBT total score is at most 90 "
        "(i.e., ≤ 90). A threshold of 90 or any lower number qualifies."
    )
    await evaluator.verify(
        claim=toefl_total_claim,
        node=toefl_total_leaf,
        sources=extracted.toefl_total_urls,
        additional_instruction=(
            "Verify from official admission policy pages (graduate school or department) that the minimum TOEFL iBT "
            "total score for admission is 90 or lower. If the policy shows ≤90 (e.g., 79, 80, 85, 90), pass."
        ),
    )


async def verify_ta_requirements(
    evaluator: Evaluator,
    parent_node,
    extracted: ProgramExtraction,
) -> None:
    ta_node = evaluator.add_parallel(
        id="TA_Requirements",
        desc="Verify the program offers TA positions and meets TA eligibility requirements",
        parent=parent_node,
        critical=True,
    )

    # TA availability
    ta_avail_leaf = evaluator.add_leaf(
        id="TA_Position_Availability",
        desc="The program must offer Teaching Assistantship positions as a form of financial support",
        parent=ta_node,
        critical=True,
    )
    ta_avail_claim = (
        "Teaching Assistantship (TA) positions are offered to graduate students in this program/department/university."
    )
    await evaluator.verify(
        claim=ta_avail_claim,
        node=ta_avail_leaf,
        sources=extracted.ta_urls,
        additional_instruction=(
            "Confirm from official department/program/graduate school funding pages that TA positions exist for "
            "graduate students (even if competitive). Any official mention of TA opportunities suffices."
        ),
    )

    # TA speaking requirement (≤ 24)
    ta_speaking_leaf = evaluator.add_leaf(
        id="TOEFL_Speaking_Score_for_TA",
        desc="For TA eligibility, the program must accept a TOEFL iBT speaking subsection score of 24 or lower",
        parent=ta_node,
        critical=True,
    )
    ta_speaking_claim = (
        "For TA eligibility, the required TOEFL iBT Speaking subscore threshold is at most 24 "
        "(i.e., ≤ 24). A minimum requirement of 24 qualifies; any requirement above 24 fails."
    )
    await evaluator.verify(
        claim=ta_speaking_claim,
        node=ta_speaking_leaf,
        sources=extracted.ta_speaking_urls,
        additional_instruction=(
            "Use official TA eligibility or English proficiency policy pages. If the TA eligibility policy sets the "
            "TOEFL iBT Speaking subscore requirement at 24 or lower, pass. If it requires >24 (e.g., 26), fail."
        ),
    )

    # TA enrollment requirement (≤ 9 credits)
    ta_enroll_leaf = evaluator.add_leaf(
        id="Enrollment_Requirement_for_TA",
        desc="For TA positions, the program must allow assistantship eligibility with enrollment in 9 or fewer graduate credit hours per semester",
        parent=ta_node,
        critical=True,
    )
    ta_enroll_claim = (
        "Assistantship eligibility is permitted while enrolled in 9 or fewer graduate credit hours per semester "
        "(i.e., the enrollment threshold is ≤ 9 credits)."
    )
    await evaluator.verify(
        claim=ta_enroll_claim,
        node=ta_enroll_leaf,
        sources=extracted.ta_enrollment_urls,
        additional_instruction=(
            "Check official assistantship policies (graduate school or department). If the policy allows TAs to be "
            "eligible at 9 or fewer credits (e.g., minimum enrollment requirement of 6 or 9 credits), pass."
        ),
    )


async def verify_required_information(
    evaluator: Evaluator,
    parent_node,
    extracted: ProgramExtraction,
) -> None:
    info_node = evaluator.add_parallel(
        id="Required_Information_Provided",
        desc="Verify that the solution provides all requested information about the identified program",
        parent=parent_node,
        critical=True,
    )

    # University name provided
    uni_name_exists = bool(extracted.university_name and extracted.university_name.strip())
    evaluator.add_custom_node(
        result=uni_name_exists,
        id="University_Name_Provided",
        desc="The solution must provide the name of the university",
        parent=info_node,
        critical=True,
    )

    # Program name provided
    program_name_exists = bool(extracted.program_name and extracted.program_name.strip())
    evaluator.add_custom_node(
        result=program_name_exists,
        id="Program_Name_Provided",
        desc="The solution must provide the specific name of the graduate program",
        parent=info_node,
        critical=True,
    )

    # Documentation provided for each of the five enumerated requirements
    # (GPA, TOEFL total, TA availability, TA speaking, TA enrollment)
    docs_per_requirement = all([
        len(extracted.gpa_urls) > 0,
        len(extracted.toefl_total_urls) > 0,
        len(extracted.ta_urls) > 0,
        len(extracted.ta_speaking_urls) > 0,
        len(extracted.ta_enrollment_urls) > 0,
    ])
    evaluator.add_custom_node(
        result=docs_per_requirement,
        id="Documentation_Provided",
        desc="The solution must provide official documentation or web page references that verify the requirements",
        parent=info_node,
        critical=True,
    )


# ---------------------------- Main Evaluation ------------------------------- #
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

    extracted = await evaluator.extract(
        prompt=prompt_extract_program(),
        template_class=ProgramExtraction,
        extraction_name="program_extraction",
    )

    # Top-level critical node representing the rubric root
    suitable_program_node = evaluator.add_parallel(
        id="Suitable_Graduate_Program_Identification",
        desc="Identify a U.S. university graduate program in Computer Science that meets all specified admission and financial support criteria, and provide required information",
        parent=root,
        critical=True,
    )

    # Program requirements met (critical)
    reqs_node = evaluator.add_parallel(
        id="Program_Requirements_Met",
        desc="Verify that the identified program meets all admission and TA requirements",
        parent=suitable_program_node,
        critical=True,
    )

    # Subtrees under requirements
    await verify_basic_admission_requirements(evaluator, reqs_node, extracted)
    await verify_ta_requirements(evaluator, reqs_node, extracted)

    # Required information provided (critical)
    await verify_required_information(evaluator, suitable_program_node, extracted)

    # Optional: record a compact view of extracted URLs counts
    evaluator.add_custom_info(
        info={
            "general_program_urls_count": len(extracted.general_program_urls),
            "gpa_urls_count": len(extracted.gpa_urls),
            "toefl_total_urls_count": len(extracted.toefl_total_urls),
            "ta_urls_count": len(extracted.ta_urls),
            "ta_speaking_urls_count": len(extracted.ta_speaking_urls),
            "ta_enrollment_urls_count": len(extracted.ta_enrollment_urls),
        },
        info_type="url_counts_summary",
        info_name="url_counts_summary",
    )

    return evaluator.get_summary()