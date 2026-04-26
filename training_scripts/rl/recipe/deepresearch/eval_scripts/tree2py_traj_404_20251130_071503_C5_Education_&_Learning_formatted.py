import asyncio
import logging
from typing import Any, List, Optional, Dict

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "edu_institutions_football_oh_in_2025"
TASK_DESCRIPTION = (
    "Identify the following two educational institutions and provide the specified information for each:\n\n"
    "Institution A: Identify the high school located in Ohio whose varsity football team advanced to compete in the "
    "OHSAA Division I state championship game scheduled for December 2025. Provide the following information:\n"
    "- Complete official name of the high school\n"
    "- City where the school is located\n"
    "- First and last name of the school's current principal\n"
    "- First and last name of the head football coach\n"
    "- Name of the stadium/venue where the championship game is scheduled to be held\n\n"
    "Institution B: Identify the university located in Indiana that announced the hiring of a new head football coach "
    "on December 8, 2024. Provide the following information:\n"
    "- Complete official name of the university\n"
    "- First and last name of the university's current president\n"
    "- First and last name of the newly hired head football coach\n"
    "- Year the newly hired coach was born\n\n"
    "For each institution, include reference URL(s) that verify the provided information."
)


# --------------------------------------------------------------------------- #
# Extraction models                                                           #
# --------------------------------------------------------------------------- #
class InstitutionAData(BaseModel):
    school_name: Optional[str] = None
    city: Optional[str] = None
    principal_name: Optional[str] = None
    coach_name: Optional[str] = None
    venue_name: Optional[str] = None
    reference_urls: List[str] = Field(default_factory=list)


class InstitutionBData(BaseModel):
    university_name: Optional[str] = None
    president_name: Optional[str] = None
    coach_name: Optional[str] = None
    coach_birth_year: Optional[str] = None
    reference_urls: List[str] = Field(default_factory=list)


class InstitutionsExtraction(BaseModel):
    institution_a: Optional[InstitutionAData] = None
    institution_b: Optional[InstitutionBData] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_institutions() -> str:
    return (
        "Extract the two educational institutions and all requested fields exactly as presented in the answer.\n\n"
        "Institution A (Ohio high school; OHSAA Division I state championship game scheduled for December 2025):\n"
        "- school_name: Complete official name of the high school\n"
        "- city: City where the school is located\n"
        "- principal_name: First and last name of the school's current principal\n"
        "- coach_name: First and last name of the head football coach\n"
        "- venue_name: Name of the stadium/venue where the championship game is scheduled to be held\n"
        "- reference_urls: An array of URLs cited in the answer that verify this institution and its attributes\n\n"
        "Institution B (Indiana university; announced hiring of new head football coach on Dec 8, 2024):\n"
        "- university_name: Complete official name of the university\n"
        "- president_name: First and last name of the university's current president\n"
        "- coach_name: First and last name of the newly hired head football coach\n"
        "- coach_birth_year: The year the newly hired coach was born\n"
        "- reference_urls: An array of URLs cited in the answer that verify this institution and its attributes\n\n"
        "Rules:\n"
        "1) Extract only what is explicitly stated in the answer. If a field is missing, return null for that field.\n"
        "2) For reference_urls, extract only actual URLs shown in the answer (plain URLs or URLs inside markdown links). "
        "Return an empty array if no URLs are provided for an institution.\n"
        "3) Do not infer or invent information.\n"
        "Return a JSON object with two top-level keys: institution_a and institution_b, each as an object with the fields above."
    )


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def _non_empty_str(s: Optional[str]) -> bool:
    return bool(s) and bool(s.strip())


def _valid_urls(urls: Optional[List[str]]) -> List[str]:
    if not urls:
        return []
    # Keep all non-empty entries; framework will normalize and handle
    return [u for u in urls if isinstance(u, str) and u.strip()]


# --------------------------------------------------------------------------- #
# Verification logic for Institution A                                        #
# --------------------------------------------------------------------------- #
async def verify_institution_a(evaluator: Evaluator, parent_node, data: Optional[InstitutionAData]) -> None:
    # Create Institution A parent node (parallel aggregator, non-critical to allow partial credit)
    inst_a_node = evaluator.add_parallel(
        id="InstitutionA",
        desc="Institution A: Ohio high school whose varsity football team advanced to compete in the OHSAA Division I state championship game scheduled for December 2025; provide all required fields and references",
        parent=parent_node,
        critical=False,
    )

    # Critical fields presence group (critical gate)
    fields_gate = evaluator.add_parallel(
        id="InstitutionA_fields_gate",
        desc="Institution A required fields presence gate",
        parent=inst_a_node,
        critical=True,
    )

    # Presence checks (each critical, as specified by rubric)
    evaluator.add_custom_node(
        result=_non_empty_str(data.school_name) if data else False,
        id="InstitutionA_SchoolName",
        desc="Complete official name of the high school provided",
        parent=fields_gate,
        critical=True,
    )

    evaluator.add_custom_node(
        result=_non_empty_str(data.city) if data else False,
        id="InstitutionA_City",
        desc="City where the school is located provided",
        parent=fields_gate,
        critical=True,
    )

    evaluator.add_custom_node(
        result=_non_empty_str(data.principal_name) if data else False,
        id="InstitutionA_PrincipalName",
        desc="First and last name of the school's current principal provided",
        parent=fields_gate,
        critical=True,
    )

    evaluator.add_custom_node(
        result=_non_empty_str(data.coach_name) if data else False,
        id="InstitutionA_CoachName",
        desc="First and last name of the head football coach provided",
        parent=fields_gate,
        critical=True,
    )

    evaluator.add_custom_node(
        result=_non_empty_str(data.venue_name) if data else False,
        id="InstitutionA_VenueName",
        desc="Name of the stadium/venue where the championship game is scheduled to be held provided",
        parent=fields_gate,
        critical=True,
    )

    evaluator.add_custom_node(
        result=len(_valid_urls(data.reference_urls if data else [])) > 0,
        id="InstitutionA_Reference",
        desc="Reference URL(s) provided that verify the Institution A identification and the provided attributes",
        parent=fields_gate,
        critical=True,
    )

    # Verification leaves (critical), using references
    # 1) Located in Ohio
    located_leaf = evaluator.add_leaf(
        id="InstitutionA_LocatedInOhio",
        desc="Institution A is a high school located in Ohio",
        parent=inst_a_node,
        critical=True,
    )

    school = data.school_name if data and data.school_name else "the high school"
    city = data.city if data and data.city else None
    if city:
        located_claim = f"The high school '{school}' is located in Ohio, specifically in {city}, Ohio."
    else:
        located_claim = f"The high school '{school}' is located in the state of Ohio."
    await evaluator.verify(
        claim=located_claim,
        node=located_leaf,
        sources=_valid_urls(data.reference_urls if data else []),
        additional_instruction="Use the provided references to confirm the school is located in Ohio. Accept confirmation via the school website, OHSAA pages, or reputable news pages.",
    )

    # 2) Advanced to OHSAA Division I state championship scheduled for December 2025
    advanced_leaf = evaluator.add_leaf(
        id="InstitutionA_AdvancedToOHSAA_D1_Dec2025",
        desc="Institution A's varsity football team advanced to compete in the OHSAA Division I state championship game scheduled for December 2025",
        parent=inst_a_node,
        critical=True,
    )

    coach = data.coach_name if data and data.coach_name else ""
    advanced_claim = (
        f"The varsity football team of '{school}' advanced to compete in the OHSAA Division I state championship game scheduled for December 2025."
    )
    await evaluator.verify(
        claim=advanced_claim,
        node=advanced_leaf,
        sources=_valid_urls(data.reference_urls if data else []),
        additional_instruction=(
            "Confirm that the references explicitly indicate the school's varsity football team reached or qualified for "
            "the OHSAA Division I state championship game in December 2025. Accept synonyms such as 'state final', "
            "'title game', or 'championship game'."
        ),
    )


# --------------------------------------------------------------------------- #
# Verification logic for Institution B                                        #
# --------------------------------------------------------------------------- #
async def verify_institution_b(evaluator: Evaluator, parent_node, data: Optional[InstitutionBData]) -> None:
    # Create Institution B parent node (parallel aggregator, non-critical to allow partial credit)
    inst_b_node = evaluator.add_parallel(
        id="InstitutionB",
        desc="Institution B: Indiana university that publicly announced hiring of a new head football coach on Dec 8, 2024; provide all required fields and references",
        parent=parent_node,
        critical=False,
    )

    # Critical fields presence group (critical gate)
    fields_gate = evaluator.add_parallel(
        id="InstitutionB_fields_gate",
        desc="Institution B required fields presence gate",
        parent=inst_b_node,
        critical=True,
    )

    # Presence checks (each critical, as specified by rubric)
    evaluator.add_custom_node(
        result=_non_empty_str(data.university_name) if data else False,
        id="InstitutionB_UniversityName",
        desc="Complete official name of the university provided",
        parent=fields_gate,
        critical=True,
    )

    evaluator.add_custom_node(
        result=_non_empty_str(data.president_name) if data else False,
        id="InstitutionB_PresidentName",
        desc="First and last name of the university's current president provided",
        parent=fields_gate,
        critical=True,
    )

    evaluator.add_custom_node(
        result=_non_empty_str(data.coach_name) if data else False,
        id="InstitutionB_CoachName",
        desc="First and last name of the newly hired head football coach provided",
        parent=fields_gate,
        critical=True,
    )

    evaluator.add_custom_node(
        result=_non_empty_str(data.coach_birth_year) if data else False,
        id="InstitutionB_CoachBirthYear",
        desc="Birth year of the newly hired coach provided",
        parent=fields_gate,
        critical=True,
    )

    evaluator.add_custom_node(
        result=len(_valid_urls(data.reference_urls if data else [])) > 0,
        id="InstitutionB_Reference",
        desc="Reference URL(s) provided that verify the Institution B identification and the provided attributes",
        parent=fields_gate,
        critical=True,
    )

    # Verification leaves (critical), using references
    # 1) Located in Indiana
    located_leaf = evaluator.add_leaf(
        id="InstitutionB_LocatedInIndiana",
        desc="Institution B is a university located in Indiana",
        parent=inst_b_node,
        critical=True,
    )

    uni = data.university_name if data and data.university_name else "the university"
    located_claim = f"The university '{uni}' is located in the state of Indiana."
    await evaluator.verify(
        claim=located_claim,
        node=located_leaf,
        sources=_valid_urls(data.reference_urls if data else []),
        additional_instruction="Use the provided references to confirm the university is located in Indiana. Accept confirmation via official university pages or reputable sources.",
    )

    # 2) Announced hiring on December 8, 2024
    hire_leaf = evaluator.add_leaf(
        id="InstitutionB_AnnouncedHireOnDec8_2024",
        desc="Institution B publicly announced the hiring of a new head football coach on December 8, 2024",
        parent=inst_b_node,
        critical=True,
    )

    coach_name = data.coach_name if data and data.coach_name else "the new head coach"
    hire_claim = (
        f"On December 8, 2024, '{uni}' announced the hiring of '{coach_name}' as its new head football coach."
    )
    await evaluator.verify(
        claim=hire_claim,
        node=hire_leaf,
        sources=_valid_urls(data.reference_urls if data else []),
        additional_instruction=(
            "Verify the announcement date (December 8, 2024) and that it is specifically about hiring the new head football coach. "
            "Accept official press releases, athletics department pages, or reputable news coverage."
        ),
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
    Evaluate the agent's answer for the educational institutions task.
    """
    # Initialize evaluator with parallel root (non-critical to avoid critical-child consistency constraint)
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

    # Extract structured information from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_institutions(),
        template_class=InstitutionsExtraction,
        extraction_name="institutions_extraction",
    )

    # Build and verify Institution A subtree
    await verify_institution_a(
        evaluator=evaluator,
        parent_node=root,
        data=extracted.institution_a if extracted and extracted.institution_a else None,
    )

    # Build and verify Institution B subtree
    await verify_institution_b(
        evaluator=evaluator,
        parent_node=root,
        data=extracted.institution_b if extracted and extracted.institution_b else None,
    )

    # Return structured summary
    return evaluator.get_summary()