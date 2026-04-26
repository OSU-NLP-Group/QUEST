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
TASK_ID = "wake_forest_head_coach_dec_2024"
TASK_DESCRIPTION = (
    "In December 2024, Wake Forest University hired a new head football coach. "
    "What is the 2024 population of the city where this coach earned his bachelor's degree? "
    "Provide the following information with supporting URL references: "
    "(1) The coach's full name, (2) His undergraduate alma mater (institution name), "
    "(3) The city and state where that institution is located, and (4) The 2024 population of that city."
)

# Ground truth expectations based on rubric
EXPECTEDS = {
    "undergrad_institution": "University of Wisconsin–Stevens Point",
    "institution_city": "Stevens Point",
    "institution_state": "Wisconsin",
    "population_2024": "26,465",
    "hiring_date": "December 18, 2024",
    "program_history_ordinal": "33rd",
}


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class CoachInfo(BaseModel):
    full_name: Optional[str] = None
    hiring_date_text: Optional[str] = None  # e.g., "December 18, 2024"
    ordinal_text: Optional[str] = None  # e.g., "33rd"
    hiring_sources: List[str] = Field(default_factory=list)


class UndergraduateInfo(BaseModel):
    institution_name: Optional[str] = None
    bachelors_field: Optional[str] = None
    bachelors_completion_year: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class InstitutionLocationInfo(BaseModel):
    city: Optional[str] = None
    state: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class PopulationInfo(BaseModel):
    population_2024: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class AnswerExtraction(BaseModel):
    coach: Optional[CoachInfo] = None
    undergraduate: Optional[UndergraduateInfo] = None
    location: Optional[InstitutionLocationInfo] = None
    population: Optional[PopulationInfo] = None


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_all() -> str:
    return """
    Extract structured information strictly from the provided answer text that addresses the following task:

    In December 2024, Wake Forest University hired a new head football coach. We need:
    (1) The coach's full name,
    (2) His undergraduate alma mater (institution name),
    (3) The city and state where that institution is located,
    (4) The 2024 population of that city,
    Each accompanied by relevant source URLs cited in the answer.

    Return a JSON object with these fields:

    coach:
      - full_name: The full name of the individual hired as Wake Forest head football coach in December 2024.
      - hiring_date_text: The hiring date as stated in the answer (e.g., "December 18, 2024"). If the date isn't stated, return null.
      - ordinal_text: The program-history ordinal if stated (e.g., "33rd"). If not stated, return null.
      - hiring_sources: An array of URLs explicitly cited in the answer that support the coach’s identity and the Dec 18, 2024 Wake Forest head coach hiring claim. If none provided, return an empty array.

    undergraduate:
      - institution_name: The name of the institution where he earned his bachelor's degree, as stated in the answer.
      - bachelors_field: The field/major of the bachelor's degree (e.g., "secondary math education"), as stated in the answer. If not stated, return null.
      - bachelors_completion_year: The completion year (e.g., "2007"), as stated in the answer. If not stated, return null.
      - sources: An array of URLs explicitly cited in the answer that support the bachelor's-degree institution and at least the bachelor’s-degree claim. If none provided, return an empty array.

    location:
      - city: The city where the undergraduate institution is located, as stated in the answer (e.g., "Stevens Point").
      - state: The state where the undergraduate institution is located, as stated in the answer (e.g., "Wisconsin").
      - sources: An array of URLs explicitly cited in the answer that support the institution’s city/state location. If none provided, return an empty array.

    population:
      - population_2024: The 2024 population value of the relevant city, exactly as stated in the answer (e.g., "26,465"). If not stated, return null.
      - sources: An array of URLs explicitly cited in the answer that support the 2024 population value. If none provided, return an empty array.

    Rules:
    - Extract only what is explicitly mentioned in the answer; do not infer or invent.
    - For URLs, only include actual URLs present in the answer (plain or markdown), and include the full protocol if missing by prepending http://.
    - If any item is missing from the answer, set it to null or an empty array as specified.
    """


# --------------------------------------------------------------------------- #
# Verification helpers                                                        #
# --------------------------------------------------------------------------- #
async def build_coach_section(evaluator: Evaluator, parent_node, data: AnswerExtraction) -> None:
    coach = data.coach or CoachInfo()
    # Create section node (critical per rubric; all children must be critical)
    coach_node = evaluator.add_parallel(
        id="Coach_Identification_And_Hiring",
        desc="Identifies the Wake Forest head football coach hired on Dec 18, 2024 and includes required hiring details with sources.",
        parent=parent_node,
        critical=True,
    )

    # Existence of hiring sources (gate)
    coach_sources_exist = bool(coach.hiring_sources)
    evaluator.add_custom_node(
        result=coach_sources_exist,
        id="Coach_Hiring_Source_URL",
        desc="Provides at least one valid URL that supports the coach’s identity and the Dec 18, 2024 Wake Forest head coach hiring claim.",
        parent=coach_node,
        critical=True,
    )

    # Coach full name verification
    coach_name_node = evaluator.add_leaf(
        id="Coach_Full_Name",
        desc="Provides the full name of the individual hired as Wake Forest head football coach on December 18, 2024.",
        parent=coach_node,
        critical=True,
    )
    coach_full_name = coach.full_name or ""
    claim_full_name = (
        f"The individual hired as head football coach at Wake Forest University on December 18, 2024 is {coach_full_name}."
    )
    await evaluator.verify(
        claim=claim_full_name,
        node=coach_name_node,
        sources=coach.hiring_sources,
        additional_instruction="Verify that the cited page(s) explicitly identify the person hired on December 18, 2024 as Wake Forest's head football coach, and that the full name matches the answer (allow minor variations like middle initials).",
    )

    # Hiring date and role verification
    hiring_date_role_node = evaluator.add_leaf(
        id="Hiring_Date_And_Role",
        desc="States that the individual was hired as head football coach at Wake Forest University on December 18, 2024.",
        parent=coach_node,
        critical=True,
    )
    claim_hiring = (
        f"On December 18, 2024, Wake Forest University hired {coach_full_name} as its head football coach."
    )
    await evaluator.verify(
        claim=claim_hiring,
        node=hiring_date_role_node,
        sources=coach.hiring_sources,
        additional_instruction="Confirm the announcement date is December 18, 2024 and the role is head football coach at Wake Forest University.",
    )

    # Program history ordinal verification (33rd head coach)
    ordinal_node = evaluator.add_leaf(
        id="Program_History_Ordinal",
        desc="States that the individual is the 33rd head coach in Wake Forest football program history.",
        parent=coach_node,
        critical=True,
    )
    claim_ordinal = f"{coach_full_name} is the 33rd head coach in Wake Forest football program history."
    await evaluator.verify(
        claim=claim_ordinal,
        node=ordinal_node,
        sources=coach.hiring_sources,
        additional_instruction="Verify the ordinal stated on the source page as thirty-third (33rd) head coach in program history.",
    )


async def build_undergrad_section(evaluator: Evaluator, parent_node, data: AnswerExtraction) -> None:
    ug = data.undergraduate or UndergraduateInfo()

    ug_node = evaluator.add_parallel(
        id="Undergraduate_Education",
        desc="Provides the coach’s undergraduate alma mater details per constraints, with sources.",
        parent=parent_node,
        critical=True,
    )

    # Existence of undergrad sources (gate)
    ug_sources_exist = bool(ug.sources)
    evaluator.add_custom_node(
        result=ug_sources_exist,
        id="Undergrad_Source_URL",
        desc="Provides at least one valid URL supporting the bachelor's-degree institution and (at minimum) the bachelor’s-degree claim.",
        parent=ug_node,
        critical=True,
    )

    # Institution name verification (UW–Stevens Point)
    ug_inst_node = evaluator.add_leaf(
        id="Undergrad_Institution_Name",
        desc="Identifies University of Wisconsin–Stevens Point as the institution where the individual earned his bachelor's degree.",
        parent=ug_node,
        critical=True,
    )
    claim_inst = (
        "The coach earned his bachelor's degree from University of Wisconsin–Stevens Point."
    )
    await evaluator.verify(
        claim=claim_inst,
        node=ug_inst_node,
        sources=ug.sources,
        additional_instruction="Confirm the institution name explicitly matches University of Wisconsin–Stevens Point (allow minor dash/typographical variations such as 'UW–Stevens Point' or 'University of Wisconsin-Stevens Point').",
    )

    # Bachelor's field verification (secondary math education)
    ug_field_node = evaluator.add_leaf(
        id="Bachelors_Field",
        desc="States that the bachelor's degree was in secondary math education.",
        parent=ug_node,
        critical=True,
    )
    claim_field = "The coach's bachelor's degree field was secondary math education."
    await evaluator.verify(
        claim=claim_field,
        node=ug_field_node,
        sources=ug.sources,
        additional_instruction="Verify that the source page(s) specify the bachelor’s field as secondary math education (accept reasonable variations like 'secondary mathematics education').",
    )

    # Bachelor's completion year verification (2007)
    ug_year_node = evaluator.add_leaf(
        id="Bachelors_Completion_Year",
        desc="States that the bachelor's degree was completed in 2007.",
        parent=ug_node,
        critical=True,
    )
    claim_year = "The coach completed his bachelor's degree in 2007."
    await evaluator.verify(
        claim=claim_year,
        node=ug_year_node,
        sources=ug.sources,
        additional_instruction="Verify the source page(s) state that the bachelor’s degree was completed in 2007.",
    )


async def build_location_section(evaluator: Evaluator, parent_node, data: AnswerExtraction) -> None:
    loc = data.location or InstitutionLocationInfo()

    loc_node = evaluator.add_parallel(
        id="Institution_Location",
        desc="Provides the city and state where the undergraduate institution is located, with sources.",
        parent=parent_node,
        critical=True,
    )

    # Existence of location sources (gate)
    loc_sources_exist = bool(loc.sources)
    evaluator.add_custom_node(
        result=loc_sources_exist,
        id="Location_Source_URL",
        desc="Provides at least one valid URL supporting the institution’s city/state location.",
        parent=loc_node,
        critical=True,
    )

    # City and State verification (Stevens Point, Wisconsin)
    city_state_node = evaluator.add_leaf(
        id="Institution_City_State",
        desc="States that University of Wisconsin–Stevens Point is located in Stevens Point, Wisconsin (city and state).",
        parent=loc_node,
        critical=True,
    )
    claim_city_state = (
        "University of Wisconsin–Stevens Point is located in Stevens Point, Wisconsin."
    )
    await evaluator.verify(
        claim=claim_city_state,
        node=city_state_node,
        sources=loc.sources,
        additional_instruction="Verify the institution’s official city and state (Stevens Point, Wisconsin). Accept minor naming variations of the university but the location must be Stevens Point, WI.",
    )


async def build_population_section(evaluator: Evaluator, parent_node, data: AnswerExtraction) -> None:
    pop = data.population or PopulationInfo()

    pop_node = evaluator.add_parallel(
        id="City_Population_2024",
        desc="Provides the 2024 population of the relevant city (Stevens Point, Wisconsin) with sources.",
        parent=parent_node,
        critical=True,
    )

    # Existence of population sources (gate)
    pop_sources_exist = bool(pop.sources)
    evaluator.add_custom_node(
        result=pop_sources_exist,
        id="Population_Source_URL",
        desc="Provides at least one valid URL supporting the city’s 2024 population value.",
        parent=pop_node,
        critical=True,
    )

    # 2024 population value verification (26,465)
    population_node = evaluator.add_leaf(
        id="Population_Value_And_Year",
        desc="Gives the 2024 population for Stevens Point, Wisconsin as 26,465 and clearly indicates the year is 2024.",
        parent=pop_node,
        critical=True,
    )
    claim_population = "The 2024 population for Stevens Point, Wisconsin is 26,465."
    await evaluator.verify(
        claim=claim_population,
        node=population_node,
        sources=pop.sources,
        additional_instruction="Verify that the cited source explicitly indicates a 2024 population estimate/value of 26,465 for Stevens Point, Wisconsin. Accept minor formatting like commas or spacing; the year must be 2024.",
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
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Root aggregation across sections
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

    # Top-level critical node representing the rubric root
    task_root = evaluator.add_parallel(
        id="Root",
        desc="Answer provides all requested information (coach identity, undergraduate institution, institution location, and 2024 city population) with supporting URL references, consistent with the stated constraints.",
        parent=root,
        critical=True,
    )

    # Extract structured info from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_all(),
        template_class=AnswerExtraction,
        extraction_name="answer_extraction",
    )

    # Optional: record ground truth expectations for transparency
    evaluator.add_ground_truth({
        "expected_undergrad_institution": EXPECTEDS["undergrad_institution"],
        "expected_institution_city_state": f"{EXPECTEDS['institution_city']}, {EXPECTEDS['institution_state']}",
        "expected_population_2024": EXPECTEDS["population_2024"],
        "expected_hiring_date": EXPECTEDS["hiring_date"],
        "expected_program_history_ordinal": EXPECTEDS["program_history_ordinal"],
    })

    # Build and verify each rubric section
    await build_coach_section(evaluator, task_root, extracted)
    await build_undergrad_section(evaluator, task_root, extracted)
    await build_location_section(evaluator, task_root, extracted)
    await build_population_section(evaluator, task_root, extracted)

    return evaluator.get_summary()