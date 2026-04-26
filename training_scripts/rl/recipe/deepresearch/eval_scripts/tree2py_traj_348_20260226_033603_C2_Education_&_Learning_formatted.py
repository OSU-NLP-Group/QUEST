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
TASK_ID = "ky_superintendent_ttu_90k"
TASK_DESCRIPTION = """
Identify the superintendent of a Kentucky school district who holds a doctoral degree in educational leadership or a related field from Texas Tech University and currently leads a district serving more than 90,000 students.
"""


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class SuperintendentExtraction(BaseModel):
    """
    Structured extraction for the identified superintendent and supporting sources.
    """
    candidate_name: Optional[str] = None
    district_name: Optional[str] = None
    district_state: Optional[str] = None  # e.g., "KY" or "Kentucky"
    enrollment_text: Optional[str] = None  # any textual enrollment figure, e.g., "96,000", "over 90,000"

    doctorate_institution: Optional[str] = None  # expected: "Texas Tech University" or variants
    doctorate_degree_type: Optional[str] = None  # e.g., "Ed.D.", "Ph.D."
    doctorate_field: Optional[str] = None  # e.g., "Educational Leadership", "Educational Administration"

    sources_education: List[str] = Field(default_factory=list)  # URLs that document education/background
    sources_position: List[str] = Field(default_factory=list)   # URLs that document position/district facts


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_superintendent() -> str:
    return """
    Extract the single superintendent identified in the answer who is claimed to meet the task requirements.

    Return a JSON object with the following fields:
    - candidate_name: The full name of the identified superintendent.
    - district_name: The name of the school district the person leads (as mentioned).
    - district_state: The state of that district if stated (e.g., "Kentucky" or "KY"); if not explicitly stated, return null.
    - enrollment_text: Any enrollment figure mentioned (e.g., "96,000", "over 90,000", "about 100,000"); if not provided, return null.
    - doctorate_institution: The doctoral degree institution (e.g., "Texas Tech University"); if not clearly stated, return null.
    - doctorate_degree_type: The degree type (e.g., "Ed.D.", "Ph.D."); if not mentioned, return null.
    - doctorate_field: The field or program for the doctoral degree (e.g., "Educational Leadership", "Educational Administration"); if not mentioned, return null.
    - sources_education: A list of all URLs cited in the answer that document the superintendent's education credentials (bios, university pages, profiles, reputable news). Include only URLs explicitly present in the answer text.
    - sources_position: A list of all URLs cited in the answer that document the superintendent's current position, district location (Kentucky), and/or district enrollment. Include only URLs explicitly present in the answer text.

    Rules:
    1) Extract only what is explicitly in the answer. Do not invent.
    2) For URLs, include only valid URLs present in the answer (plain or markdown links).
    3) If the answer provides a single combined "Sources" list without separating education vs position URLs, copy all listed URLs into both sources_education and sources_position.
    4) If any field is not present in the answer, set it to null (or [] for URL lists).
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _safe_name(name: Optional[str], fallback: str) -> str:
    return name.strip() if isinstance(name, str) and name.strip() else fallback


def _safe_district(district: Optional[str], fallback: str) -> str:
    return district.strip() if isinstance(district, str) and district.strip() else fallback


def _unique_urls(urls: List[str]) -> List[str]:
    seen = set()
    unique = []
    for u in urls:
        if not isinstance(u, str):
            continue
        uu = u.strip()
        if not uu:
            continue
        if uu not in seen:
            unique.append(uu)
            seen.add(uu)
    return unique


# --------------------------------------------------------------------------- #
# Verification builder                                                        #
# --------------------------------------------------------------------------- #
async def build_verification_tree(evaluator: Evaluator, extracted: SuperintendentExtraction) -> None:
    """
    Build the verification tree according to the rubric and run verifications.
    """

    # 1) Top-level critical, sequential node for the overall task
    task_node = evaluator.add_sequential(
        id="Superintendent_Identification_Task",
        desc="Identify the school district superintendent who meets all specified educational and district criteria",
        parent=evaluator.root,
        critical=True
    )

    # 2) Main verification node (parallel), still critical
    main_verify_node = evaluator.add_parallel(
        id="Verify_Identified_Superintendent",
        desc="Verify that the identified superintendent satisfies all required credentials and position characteristics",
        parent=task_node,
        critical=True
    )

    # Prepare names/strings for claims
    candidate = _safe_name(extracted.candidate_name, "the identified superintendent")
    district = _safe_district(extracted.district_name, "the identified district")
    district_state = _safe_name(extracted.district_state, "Kentucky")
    doctorate_institution = _safe_name(extracted.doctorate_institution, "Texas Tech University")
    doctorate_field = _safe_name(extracted.doctorate_field, "Educational Leadership or a closely related field")

    # Prepare URL source sets
    edu_urls = _unique_urls(extracted.sources_education or [])
    pos_urls = _unique_urls(extracted.sources_position or [])
    all_urls = _unique_urls((extracted.sources_education or []) + (extracted.sources_position or []))

    # 3) Educational_Credentials (parallel, critical)
    edu_node = evaluator.add_parallel(
        id="Educational_Credentials",
        desc="Verify the superintendent's doctoral education credentials",
        parent=main_verify_node,
        critical=True
    )

    # 3.1) Educational_Documentation_URL (critical leaf - existence check)
    evaluator.add_custom_node(
        result=len(edu_urls) > 0,
        id="Educational_Documentation_URL",
        desc="Provide official source URL documenting the superintendent's educational background",
        parent=edu_node,
        critical=True
    )

    # 3.2) Texas_Tech_Doctorate (critical leaf - verify by URLs)
    ttu_doctorate_leaf = evaluator.add_leaf(
        id="Texas_Tech_Doctorate",
        desc="The superintendent earned a doctoral degree from Texas Tech University",
        parent=edu_node,
        critical=True
    )
    ttu_claim = (
        f"{candidate} earned a doctoral degree (e.g., Ed.D. or Ph.D.) from Texas Tech University."
    )
    await evaluator.verify(
        claim=ttu_claim,
        node=ttu_doctorate_leaf,
        sources=all_urls,  # allow any cited page to support the claim
        additional_instruction=(
            "Confirm that the page explicitly states the person earned a doctoral degree from Texas Tech University. "
            "Accept reasonable variants such as 'TTU', 'Texas Tech', or references to Texas Tech University's College of Education. "
            "If the page mentions a doctorate from another institution instead, mark as not supported."
        ),
    )

    # 3.3) Educational_Leadership_Field (critical leaf - verify by URLs)
    field_leaf = evaluator.add_leaf(
        id="Educational_Leadership_Field",
        desc="The doctoral degree is in educational leadership, educational administration, or a closely related field",
        parent=edu_node,
        critical=True
    )
    field_claim = (
        f"The doctoral degree that {candidate} earned from Texas Tech University is in educational leadership, "
        f"educational administration, or a closely related field (e.g., Educational Leadership & Policy, "
        f"Educational Leadership and Administration)."
    )
    await evaluator.verify(
        claim=field_claim,
        node=field_leaf,
        sources=all_urls,
        additional_instruction=(
            "Accept field wording variants such as 'Educational Leadership', 'Education Leadership', "
            "'Educational Administration', 'Educational Leadership & Policy', or obviously equivalent program names. "
            "If the field is clearly unrelated (e.g., Chemistry), mark as not supported."
        ),
    )

    # 4) Position_and_District (parallel, critical)
    pos_node = evaluator.add_parallel(
        id="Position_and_District",
        desc="Verify the superintendent's current position and district characteristics",
        parent=main_verify_node,
        critical=True
    )

    # 4.1) Position_Documentation_URL (critical leaf - existence check)
    evaluator.add_custom_node(
        result=len(pos_urls) > 0,
        id="Position_Documentation_URL",
        desc="Provide official source URL documenting the superintendent's current position and district information",
        parent=pos_node,
        critical=True
    )

    # 4.2) Kentucky_Superintendent (critical leaf - verify by URLs)
    ky_leaf = evaluator.add_leaf(
        id="Kentucky_Superintendent",
        desc="The individual currently serves as superintendent of a school district in Kentucky",
        parent=pos_node,
        critical=True
    )
    ky_claim = (
        f"{candidate} currently serves as the superintendent of {district}, and this district is in Kentucky."
    )
    await evaluator.verify(
        claim=ky_claim,
        node=ky_leaf,
        sources=all_urls,
        additional_instruction=(
            "Verify both: (1) the individual holds the title 'Superintendent' (or clear equivalent, e.g., "
            "'Superintendent of Schools') for the district, and (2) the district is located in Kentucky. "
            "Allow reasonable name variants for the district and titles."
        ),
    )

    # 4.3) Large_District_Enrollment (critical leaf - verify by URLs)
    large_leaf = evaluator.add_leaf(
        id="Large_District_Enrollment",
        desc="The school district serves more than 90,000 students",
        parent=pos_node,
        critical=True
    )
    enrollment_claim = (
        f"{district} serves more than 90,000 students (enrollment > 90,000)."
    )
    await evaluator.verify(
        claim=enrollment_claim,
        node=large_leaf,
        sources=all_urls,
        additional_instruction=(
            "Confirm that the cited page(s) indicate a student population exceeding 90,000. "
            "Accept expressions like 'over 90,000', 'more than 90,000', 'approximately 96,000', or similar. "
            "If the page shows a number clearly below 90,000, mark as not supported."
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
    model: str = "o4-mini"
) -> Dict:
    """
    Evaluate an answer for the Kentucky superintendent with Texas Tech doctoral credential and >90,000 enrollment district.
    """
    # Initialize evaluator and root
    evaluator = Evaluator()
    evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.SEQUENTIAL,  # Root strategy not critical; we add a critical child node per rubric
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

    # Extract structured info from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_superintendent(),
        template_class=SuperintendentExtraction,
        extraction_name="superintendent_extraction",
        additional_instruction=(
            "If the sources list is not separated by topic, duplicate all URLs into both sources_education and sources_position."
        )
    )

    # Optionally record some custom info for debugging transparency
    evaluator.add_custom_info(
        info={
            "candidate_name": extracted.candidate_name,
            "district_name": extracted.district_name,
            "district_state": extracted.district_state,
            "doctorate_institution": extracted.doctorate_institution,
            "doctorate_degree_type": extracted.doctorate_degree_type,
            "doctorate_field": extracted.doctorate_field,
            "sources_education_count": len(extracted.sources_education or []),
            "sources_position_count": len(extracted.sources_position or []),
        },
        info_type="extraction_overview",
        info_name="extraction_overview"
    )

    # Build and run verification checks
    await build_verification_tree(evaluator, extracted)

    # Return evaluator's structured result summary
    return evaluator.get_summary()