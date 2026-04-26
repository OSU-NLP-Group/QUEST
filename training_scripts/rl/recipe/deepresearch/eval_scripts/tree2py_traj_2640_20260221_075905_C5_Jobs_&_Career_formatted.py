import asyncio
import logging
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "tx_superintendent_identification"
TASK_DESCRIPTION = (
    "Identify a current superintendent of a Texas school district who meets all of the following criteria:\n\n"
    "1. The district is classified as suburban (not urban or rural)\n"
    "2. The district serves more than 30,000 students\n"
    "3. The district operates at least 50 schools or campuses\n"
    "4. The superintendent began their teaching career in the 1990s\n"
    "5. The superintendent's career progression included serving as a classroom teacher, assistant principal, and principal before becoming superintendent\n"
    "6. The superintendent holds a master's degree in educational administration or educational leadership\n"
    "7. The superintendent is currently pursuing or holds a doctoral degree (Ed.D. or Ph.D.)\n"
    "8. The superintendent was appointed to their current position between 2020 and 2023\n"
    "9. The superintendent's total compensation in the 2023-24 school year exceeded $300,000\n\n"
    "Provide the superintendent's full name, the name of their school district, and supporting reference URLs that verify each of the above criteria."
)


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class SuperintendentData(BaseModel):
    superintendent_name: Optional[str] = None
    district_name: Optional[str] = None

    # Source URLs grouped by verification category
    district_sources: List[str] = Field(default_factory=list)             # Location, classification, enrollment, school count
    career_sources: List[str] = Field(default_factory=list)               # Teaching start decade, roles, master's, doctoral
    appointment_comp_sources: List[str] = Field(default_factory=list)     # Appointment timeframe, 2023-24 compensation


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_superintendent_data() -> str:
    return """
    Extract the following information as JSON from the provided answer. Do not invent any information.

    Required fields:
    1. superintendent_name: The full name of the superintendent identified in the answer.
    2. district_name: The official name of the superintendent's school district (e.g., "Katy ISD", "Frisco ISD").
    3. district_sources: A list of URLs that the answer cites to support district facts, including location (Texas), suburban classification, total student enrollment, and number of schools/campuses.
    4. career_sources: A list of URLs that the answer cites to support the superintendent's career background and qualifications: teaching start decade (1990s), progression through classroom teacher → assistant principal → principal before superintendent, master's degree field (educational administration or educational leadership), and doctoral pursuit/degree (Ed.D. or Ph.D.).
    5. appointment_comp_sources: A list of URLs that the answer cites to support the superintendent's appointment timeframe (between 2020 and 2023) and total compensation exceeding $300,000 in the 2023–24 school year.

    Rules:
    - Extract only URLs explicitly present in the answer. Include full URLs (with http/https). Ignore malformed URLs.
    - If multiple URLs are given without categorization, place each URL into all relevant lists it appears to support.
    - If the answer does not provide URLs for a category, return an empty list for that category.
    - If the superintendent_name or district_name is not clearly provided, set them to null.

    Return a single JSON object with fields: superintendent_name, district_name, district_sources, career_sources, appointment_comp_sources.
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def _safe_name(name: Optional[str]) -> str:
    return name.strip() if name else "the superintendent"

def _safe_district(district: Optional[str]) -> str:
    return district.strip() if district else "the school district"


# --------------------------------------------------------------------------- #
# Verification subtree builders                                               #
# --------------------------------------------------------------------------- #
async def build_district_characteristics_checks(
    evaluator: Evaluator,
    parent_node,
    data: SuperintendentData,
) -> None:
    """
    Build and verify the 'District_Characteristics' subtree.
    """
    district_node = evaluator.add_parallel(
        id="District_Characteristics",
        desc="The superintendent's current district meets all geographic and size requirements",
        parent=parent_node,
        critical=True,
    )

    # URL reference existence (gate)
    evaluator.add_custom_node(
        result=bool(data.district_sources),
        id="District_Characteristics_URL_Reference",
        desc="Provide reference URL(s) that document the district's location, classification, enrollment, and number of schools",
        parent=district_node,
        critical=True
    )

    district = _safe_district(data.district_name)

    # Texas location
    texas_node = evaluator.add_leaf(
        id="Texas_Location",
        desc="The district is located in Texas",
        parent=district_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The school district named '{district}' is located in Texas.",
        node=texas_node,
        sources=data.district_sources,
        additional_instruction="Confirm that the district is in the state of Texas. Accept official district websites, state education agency pages, or credible profiles as evidence."
    )

    # Suburban classification
    suburban_node = evaluator.add_leaf(
        id="Suburban_Classification",
        desc="The district is classified as suburban (not urban or rural)",
        parent=district_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The school district '{district}' is classified as suburban (not urban or rural).",
        node=suburban_node,
        sources=data.district_sources,
        additional_instruction=(
            "Look for locale classification from NCES (e.g., Suburb: Large/Medium/Small) or credible sources "
            "that explicitly describe the district as suburban. If a source states an NCES code corresponding to Suburb, "
            "that counts. Do not accept 'urban' or 'rural' classifications."
        )
    )

    # Student enrollment > 30,000
    enrollment_node = evaluator.add_leaf(
        id="Student_Enrollment",
        desc="The district serves more than 30,000 students",
        parent=district_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The school district '{district}' serves more than 30,000 students.",
        node=enrollment_node,
        sources=data.district_sources,
        additional_instruction=(
            "Check recent enrollment figures on official sources (district profile, accountability reports, NCES). "
            "If the number is clearly above 30,000, consider the claim supported. Reasonable rounding is acceptable."
        )
    )

    # Number of schools >= 50
    schools_node = evaluator.add_leaf(
        id="Number_of_Schools",
        desc="The district operates at least 50 schools/campuses",
        parent=district_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The school district '{district}' operates at least 50 schools or campuses.",
        node=schools_node,
        sources=data.district_sources,
        additional_instruction=(
            "Verify the number of schools/campuses per official district pages or credible data sources. "
            "Count should be 50 or greater."
        )
    )


async def build_career_background_checks(
    evaluator: Evaluator,
    parent_node,
    data: SuperintendentData,
) -> None:
    """
    Build and verify the 'Career_Background_and_Qualifications' subtree.
    """
    career_node = evaluator.add_parallel(
        id="Career_Background_and_Qualifications",
        desc="The superintendent's career trajectory and educational qualifications meet all specified requirements",
        parent=parent_node,
        critical=True,
    )

    # URL reference existence (gate)
    evaluator.add_custom_node(
        result=bool(data.career_sources),
        id="Career_Background_URL_Reference",
        desc="Provide reference URL(s) that document the superintendent's teaching start date, career progression, master's degree, and doctoral pursuit/completion",
        parent=career_node,
        critical=True
    )

    name = _safe_name(data.superintendent_name)

    # Teaching start decade = 1990s
    start_decade_node = evaluator.add_leaf(
        id="Teaching_Start_Decade",
        desc="The superintendent began their teaching career in the 1990s",
        parent=career_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"{name} began their teaching career in the 1990s.",
        node=start_decade_node,
        sources=data.career_sources,
        additional_instruction=(
            "Check biography pages, interviews, or credible profiles that state the first year/period of teaching. "
            "Any start year between 1990–1999 qualifies."
        )
    )

    # Progressive roles: teacher, assistant principal, principal
    roles_node = evaluator.add_leaf(
        id="Progressive_Roles",
        desc="The superintendent served as classroom teacher, assistant principal, and principal before becoming superintendent",
        parent=career_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"Before becoming superintendent, {name} served as a classroom teacher, assistant principal, AND principal.",
        node=roles_node,
        sources=data.career_sources,
        additional_instruction=(
            "Confirm that all three roles (teacher, assistant principal, principal) appear in the career history prior to becoming superintendent. "
            "Minor title variants are acceptable (e.g., 'AP' for assistant principal). All three must be present."
        )
    )

    # Master's degree field: educational administration or educational leadership
    masters_node = evaluator.add_leaf(
        id="Masters_Degree_Field",
        desc="The superintendent holds a master's degree in educational administration or educational leadership",
        parent=career_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"{name} holds a master's degree in educational administration or educational leadership.",
        node=masters_node,
        sources=data.career_sources,
        additional_instruction=(
            "Accept synonyms like 'Master of Education (M.Ed.) in Educational Leadership', 'MS in Educational Administration', "
            "or similar formulations clearly indicating the field is educational administration/leadership."
        )
    )

    # Doctoral pursuit or holder: Ed.D. or Ph.D.
    doctoral_node = evaluator.add_leaf(
        id="Doctoral_Pursuit",
        desc="The superintendent is pursuing or holds a doctoral degree (Ed.D. or Ph.D.)",
        parent=career_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"{name} is currently pursuing or holds a doctoral degree (Ed.D. or Ph.D.).",
        node=doctoral_node,
        sources=data.career_sources,
        additional_instruction=(
            "The evidence may state 'pursuing a doctorate', 'doctoral candidate', 'earned Ed.D.', or 'earned Ph.D.'. "
            "Either pursuit or completion qualifies."
        )
    )


async def build_appointment_comp_checks(
    evaluator: Evaluator,
    parent_node,
    data: SuperintendentData,
) -> None:
    """
    Build and verify the 'Appointment_and_Compensation' subtree.
    """
    appt_comp_node = evaluator.add_parallel(
        id="Appointment_and_Compensation",
        desc="The superintendent's appointment timing and compensation meet specified thresholds",
        parent=parent_node,
        critical=True,
    )

    # URL reference existence (gate)
    evaluator.add_custom_node(
        result=bool(data.appointment_comp_sources),
        id="Appointment_Compensation_URL_Reference",
        desc="Provide reference URL(s) that document the superintendent's appointment date and total compensation for 2023-24",
        parent=appt_comp_node,
        critical=True
    )

    name = _safe_name(data.superintendent_name)
    district = _safe_district(data.district_name)

    # Appointment timeframe (2020–2023)
    appointment_node = evaluator.add_leaf(
        id="Appointment_Timeframe",
        desc="The superintendent was appointed to their current position between 2020 and 2023",
        parent=appt_comp_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"{name} was appointed as superintendent of {district} between 2020 and 2023 (inclusive).",
        node=appointment_node,
        sources=data.appointment_comp_sources,
        additional_instruction=(
            "Check board announcements, district press releases, or credible news reports for the appointment date. "
            "The appointment year must be one of 2020, 2021, 2022, or 2023."
        )
    )

    # Compensation level > $300,000 in 2023–24
    compensation_node = evaluator.add_leaf(
        id="Compensation_Level",
        desc="The superintendent's total compensation in the 2023-24 school year exceeded $300,000",
        parent=appt_comp_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"In the 2023–24 school year, {name}'s total compensation exceeded $300,000.",
        node=compensation_node,
        sources=data.appointment_comp_sources,
        additional_instruction=(
            "Use official contracts, board agenda materials, district financial documents, or credible reports. "
            "Total compensation may include base salary plus allowances/stipends (e.g., travel, housing, car allowance). "
            "If a source explicitly says 'total compensation' or clearly sums to > $300,000, consider it supported."
        )
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
) -> Dict:
    """
    Evaluate the answer for identifying a Texas superintendent that meets the specified criteria.
    """
    # Initialize evaluator
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

    # Extract structured information
    extracted: SuperintendentData = await evaluator.extract(
        prompt=prompt_extract_superintendent_data(),
        template_class=SuperintendentData,
        extraction_name="superintendent_data",
    )

    # Add a critical top-level node to mirror rubric root
    super_node = evaluator.add_parallel(
        id="Superintendent_Identification",
        desc="Correctly identify a superintendent who meets all specified criteria for district characteristics, career background, qualifications, and appointment details",
        parent=root,
        critical=True,
    )

    # Basic identification presence check (gate)
    evaluator.add_custom_node(
        result=bool(extracted.superintendent_name) and bool(extracted.district_name),
        id="Basics_Provided",
        desc="Superintendent's full name and school district are provided in the answer",
        parent=super_node,
        critical=True
    )

    # Build subtree checks
    await build_district_characteristics_checks(evaluator, super_node, extracted)
    await build_career_background_checks(evaluator, super_node, extracted)
    await build_appointment_comp_checks(evaluator, super_node, extracted)

    # Add custom info for thresholds used
    evaluator.add_custom_info(
        info={
            "district_classification_required": "suburban",
            "enrollment_threshold": "> 30,000",
            "schools_threshold": ">= 50",
            "teaching_start_decade": "1990s",
            "required_roles": ["classroom teacher", "assistant principal", "principal"],
            "masters_field": ["educational administration", "educational leadership"],
            "doctoral_degree": ["Ed.D.", "Ph.D."],
            "appointment_year_range": [2020, 2021, 2022, 2023],
            "compensation_threshold_2023_24": "> $300,000",
        },
        info_type="constraints",
        info_name="evaluation_constraints"
    )

    # Return final structured summary
    return evaluator.get_summary()