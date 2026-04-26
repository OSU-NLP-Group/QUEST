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
TASK_ID = "panthers_coach_education_first_role"
TASK_DESCRIPTION = (
    "What university did the current Carolina Panthers head coach graduate from and in what year? "
    "What degree and field of study did he earn? "
    "Where was his first coaching position and during what years did he serve in that role?"
)

# Expected constraints (ground truth targets)
EXPECTED_CONSTRAINTS = {
    "coach_name": "Dave Canales",
    "season_current": "2024–2025 NFL season",
    "hire_date": "January 25, 2024",
    "university": "Azusa Pacific University",
    "graduation_year": "2003",
    "degree_type": "Bachelor of Arts",
    "field_of_study": "Business Administration",
    "first_coaching_level": "High school",
    "first_coaching_institution": "Carson High School",
    "first_coaching_years": "2004–2005",
    "first_coaching_role_desc": "Head coach and offensive coordinator for the freshman/sophomore team",
}

# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class CoachInfoExtraction(BaseModel):
    """Structured information extracted from the agent's answer."""
    coach_name: Optional[str] = None
    current_status_text: Optional[str] = None  # e.g., "current head coach", "as of 2024–2025"
    hire_date: Optional[str] = None

    university: Optional[str] = None
    graduation_year: Optional[str] = None
    degree_type: Optional[str] = None
    field_of_study: Optional[str] = None

    first_coaching_level: Optional[str] = None
    first_coaching_institution: Optional[str] = None
    first_coaching_years: Optional[str] = None
    first_coaching_role_desc: Optional[str] = None

    source_urls: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_coach_info() -> str:
    return """
    Extract the following information strictly from the provided answer text. Do not invent any facts.

    Required fields to extract:
    1) coach_name: The name of the Carolina Panthers head coach mentioned in the answer.
    2) current_status_text: Any phrasing indicating he is the "current" head coach and any timing context (e.g., "as of 2024–2025").
    3) hire_date: The hire date the answer mentions for when he became Panthers head coach (e.g., "January 25, 2024").
    4) university: The university from which he graduated (e.g., "Azusa Pacific University").
    5) graduation_year: The year he graduated (e.g., "2003").
    6) degree_type: The degree type earned (e.g., "Bachelor of Arts" / "BA").
    7) field_of_study: The major or field of study (e.g., "Business Administration").
    8) first_coaching_level: The level of his first coaching role (e.g., "High school", "College").
    9) first_coaching_institution: The institution name for the first coaching role (e.g., "Carson High School").
    10) first_coaching_years: The years served in that first role (e.g., "2004–2005").
    11) first_coaching_role_desc: The role description in that first position (e.g., "Head coach and offensive coordinator for the freshman/sophomore team").
    12) source_urls: Extract all explicit URLs cited in the answer that support any of the above (web links or markdown links). If none are provided, return an empty array.

    If the answer does not mention a particular field, set it to null. For URLs, include only valid URLs explicitly present in the answer text; do not infer or create any URLs.

    Return a single JSON object with the above fields.
    """


# --------------------------------------------------------------------------- #
# Verification tree builders                                                  #
# --------------------------------------------------------------------------- #
async def add_coach_identity_and_status_nodes(
    evaluator: Evaluator,
    parent_node,
    sources: List[str]
) -> None:
    """Build and verify 'Coach_Identity_And_Status' subtree."""
    coach_node = evaluator.add_parallel(
        id="Coach_Identity_And_Status",
        desc="Correctly identifies the constraint-defined coach and status/timing constraints.",
        parent=parent_node,
        critical=True
    )

    # Leaf 1: Coach Identity Is Dave Canales
    leaf_identity = evaluator.add_leaf(
        id="Coach_Identity_Is_Dave_Canales",
        desc="Identifies the head coach as Dave Canales.",
        parent=coach_node,
        critical=True
    )
    claim_identity = "The Carolina Panthers head coach is Dave Canales."
    await evaluator.verify(
        claim=claim_identity,
        node=leaf_identity,
        sources=sources,
        additional_instruction="Verify that the sources explicitly identify Dave Canales as the head coach of the Carolina Panthers."
    )

    # Leaf 2: Coach Is Current As Of 2024–2025 NFL Season
    leaf_current = evaluator.add_leaf(
        id="Coach_Is_Current_As_Of_2024_2025_NFL_Season",
        desc="States/implies the individual is the current Carolina Panthers head coach as of the 2024–2025 NFL season.",
        parent=coach_node,
        critical=True
    )
    claim_current = "Dave Canales is the current Carolina Panthers head coach as of the 2024–2025 NFL season."
    await evaluator.verify(
        claim=claim_current,
        node=leaf_current,
        sources=sources,
        additional_instruction=(
            "Accept if the page indicates he is the head coach during 2024 or 2025, or explicitly states current status in that timeframe."
        )
    )

    # Leaf 3: Coach Hire Date Is 2024-01-25
    leaf_hire_date = evaluator.add_leaf(
        id="Coach_Hire_Date_Is_2024_01_25",
        desc="States the head coach was hired on January 25, 2024.",
        parent=coach_node,
        critical=True
    )
    claim_hire_date = "Dave Canales was hired on January 25, 2024 as the Carolina Panthers head coach."
    await evaluator.verify(
        claim=claim_hire_date,
        node=leaf_hire_date,
        sources=sources,
        additional_instruction="Allow minor formatting variations (e.g., 'Jan. 25, 2024'). The page must clearly support the hire date."
    )


async def add_education_details_nodes(
    evaluator: Evaluator,
    parent_node,
    sources: List[str]
) -> None:
    """Build and verify 'Education_Details' subtree."""
    edu_node = evaluator.add_parallel(
        id="Education_Details",
        desc="Provides the constraint-defined university, graduation year, degree, and field of study.",
        parent=parent_node,
        critical=True
    )

    # University
    leaf_univ = evaluator.add_leaf(
        id="University_Is_Azusa_Pacific_University",
        desc="States the university is Azusa Pacific University.",
        parent=edu_node,
        critical=True
    )
    claim_univ = "Dave Canales graduated from Azusa Pacific University."
    await evaluator.verify(
        claim=claim_univ,
        node=leaf_univ,
        sources=sources,
        additional_instruction="Verify that the sources explicitly state Azusa Pacific University as his alma mater."
    )

    # Graduation Year
    leaf_grad_year = evaluator.add_leaf(
        id="Graduation_Year_Is_2003",
        desc="States the graduation year is 2003.",
        parent=edu_node,
        critical=True
    )
    claim_grad_year = "Dave Canales graduated in 2003."
    await evaluator.verify(
        claim=claim_grad_year,
        node=leaf_grad_year,
        sources=sources,
        additional_instruction="Check that sources mention the year of graduation as 2003."
    )

    # Degree Type
    leaf_degree = evaluator.add_leaf(
        id="Degree_Type_Is_Bachelor_of_Arts",
        desc="States the degree type is Bachelor of Arts (BA).",
        parent=edu_node,
        critical=True
    )
    claim_degree = "Dave Canales earned a Bachelor of Arts (BA) degree."
    await evaluator.verify(
        claim=claim_degree,
        node=leaf_degree,
        sources=sources,
        additional_instruction=(
            "Allow minor variations like 'Bachelor’s' or 'B.A.' but it must clearly be a Bachelor of Arts degree rather than another type."
        )
    )

    # Field of Study
    leaf_field = evaluator.add_leaf(
        id="Field_Of_Study_Is_Business_Administration",
        desc="States the field of study/major is business administration.",
        parent=edu_node,
        critical=True
    )
    claim_field = "Dave Canales’s field of study (major) was Business Administration."
    await evaluator.verify(
        claim=claim_field,
        node=leaf_field,
        sources=sources,
        additional_instruction="Verify that sources indicate Business Administration as his major/field."
    )


async def add_first_coaching_role_nodes(
    evaluator: Evaluator,
    parent_node,
    sources: List[str]
) -> None:
    """Build and verify 'First_Coaching_Role_Details' subtree."""
    first_role_node = evaluator.add_parallel(
        id="First_Coaching_Role_Details",
        desc="Provides the constraint-defined first coaching position details.",
        parent=parent_node,
        critical=True
    )

    # High school level
    leaf_level = evaluator.add_leaf(
        id="Coaching_Career_Began_At_High_School_Level",
        desc="States he began his coaching career at the high school level.",
        parent=first_role_node,
        critical=True
    )
    claim_level = "Dave Canales began his coaching career at the high school level."
    await evaluator.verify(
        claim=claim_level,
        node=leaf_level,
        sources=sources,
        additional_instruction="Confirm that the sources state his initial coaching role was at a high school."
    )

    # First Position Institution
    leaf_institution = evaluator.add_leaf(
        id="First_Coaching_Position_Is_Carson_High_School",
        desc="States his first coaching position was at Carson High School.",
        parent=first_role_node,
        critical=True
    )
    claim_institution = "Dave Canales’s first coaching position was at Carson High School."
    await evaluator.verify(
        claim=claim_institution,
        node=leaf_institution,
        sources=sources,
        additional_instruction="Verify that the sources name Carson High School as his first coaching stop."
    )

    # Years
    leaf_years = evaluator.add_leaf(
        id="First_Coaching_Position_Years_Are_2004_2005",
        desc="States he served in that first coaching role from 2004 to 2005.",
        parent=first_role_node,
        critical=True
    )
    claim_years = "Dave Canales served in his first coaching role from 2004 to 2005."
    await evaluator.verify(
        claim=claim_years,
        node=leaf_years,
        sources=sources,
        additional_instruction="Allow minor formatting like '2004-2005' or '2004–2005'. The timeframe must match."
    )

    # Role Description
    leaf_role_desc = evaluator.add_leaf(
        id="First_Coaching_Position_Role_Matches",
        desc="States he served as head coach and offensive coordinator for the freshman/sophomore team in that first role.",
        parent=first_role_node,
        critical=True
    )
    claim_role_desc = (
        "In his first coaching role, Dave Canales served as head coach and offensive coordinator for the freshman/sophomore team."
    )
    await evaluator.verify(
        claim=claim_role_desc,
        node=leaf_role_desc,
        sources=sources,
        additional_instruction=(
            "Accept equivalent phrasing such as 'frosh-soph' or 'freshman & sophomore team'. "
            "Both duties (head coach and offensive coordinator) must be supported."
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
    Evaluate an answer for the Carolina Panthers head coach education and first coaching role task.
    """
    # Initialize evaluator with a parallel root
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

    # Extract structured info from the answer
    extracted_info = await evaluator.extract(
        prompt=prompt_extract_coach_info(),
        template_class=CoachInfoExtraction,
        extraction_name="coach_info_extraction"
    )

    # Record ground truth/expected constraints
    evaluator.add_ground_truth({
        "expected_constraints": EXPECTED_CONSTRAINTS
    }, gt_type="expected_constraints")

    # Build the critical top-level group node
    main_node = evaluator.add_parallel(
        id="Carolina_Panthers_Head_Coach_Education_And_First_Coaching_Role",
        desc="Answer matches the constraint-defined current Carolina Panthers head coach and provides the constraint-defined education and first coaching role details.",
        parent=root,
        critical=True
    )

    # Prepare sources for verification (must be URLs from the answer)
    sources_list = extracted_info.source_urls if extracted_info and extracted_info.source_urls else []

    # Subtrees
    await add_coach_identity_and_status_nodes(evaluator, main_node, sources_list)
    await add_education_details_nodes(evaluator, main_node, sources_list)
    await add_first_coaching_role_nodes(evaluator, main_node, sources_list)

    # Return structured evaluation summary
    return evaluator.get_summary()