import asyncio
import logging
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "nys_ctle_course_selection"
TASK_DESCRIPTION = (
    "I am a New York State teacher holding a professional certificate, and I need to complete continuing education to "
    "maintain my certification. I am looking for an online professional development course that meets the following "
    "requirements:\n\n"
    "1. The course must be from a provider that is an approved CTLE (Continuing Teacher and Leader Education) sponsor by the New York State Education Department (NYSED)\n"
    "2. The course must provide graduate-level university credit that counts toward both my CTLE hour requirements and salary advancement\n"
    "3. The course must be offered 100% online in a self-paced format so I can complete it around my teaching schedule\n"
    "4. The course must focus on English Language Learners (ELL) or English as a New Language (ENL) instruction, as I need to fulfill my ELL-specific CTLE hours\n"
    "5. The course provider must be affiliated with or partnered with a regionally accredited university to ensure the credits are legitimate\n\n"
    "Please identify one specific course that meets all these requirements, including the course provider name, the course title, and a reference URL where I can verify this information."
)


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class CourseItem(BaseModel):
    provider_name: Optional[str] = None
    course_title: Optional[str] = None
    reference_urls: List[str] = Field(default_factory=list)


class CoursesExtraction(BaseModel):
    courses: List[CourseItem] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_courses() -> str:
    return """
    Extract all distinct course entries mentioned in the answer that are proposed to meet the user's CTLE requirements.
    For each course entry, extract:
    - provider_name: The name of the course provider (e.g., Learners Edge, Advancement Courses, etc.). If not explicitly named, return null.
    - course_title: The specific course title (not a program name or general category). If no specific title is given, return null.
    - reference_urls: An array of all URLs present in the answer that directly relate to verifying this specific course and/or the provider's status/claims (include the course page, provider info pages, NYSED pages, partner university pages if included). Only include URLs explicitly present in the answer text.

    Return a JSON object with a top-level field:
    - courses: an array of up to 5 course objects as described above, preserving the order they appear in the answer.

    Important:
    - Do NOT invent or infer URLs. Only include those explicitly appearing in the answer (plain or markdown links).
    - Do not merge multiple courses into one; if the answer lists multiple distinct courses, include them as separate items in order.
    - If the answer mentions only a provider but no specific course, include one item with provider_name set and course_title as null.
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def select_primary_course(extraction: CoursesExtraction) -> Optional[CourseItem]:
    if extraction.courses:
        return extraction.courses[0]
    return None


def _safe_str(v: Optional[str]) -> str:
    return v or ""


# --------------------------------------------------------------------------- #
# Verification tree construction and checks                                   #
# --------------------------------------------------------------------------- #
async def verify_ctle_course(
    evaluator: Evaluator,
    parent_node,
    extraction: CoursesExtraction,
) -> None:
    # Create top-level critical node for this task (as per rubric)
    ctle_node = evaluator.add_parallel(
        id="CTLE_Course_Identification",
        desc="Identify one specific online professional development course that meets all specified NYS CTLE requirements and provide the required identifying information.",
        parent=parent_node,
        critical=True,
    )

    primary = select_primary_course(extraction)

    # ----------------- Specific_Course_Identified (leaf, critical) -----------------
    exactly_one_specific = (
        len(extraction.courses) == 1
        and primary is not None
        and isinstance(primary.course_title, str)
        and primary.course_title.strip() != ""
    )
    evaluator.add_custom_node(
        result=exactly_one_specific,
        id="Specific_Course_Identified",
        desc="Response clearly identifies exactly one specific course (not a list of multiple courses and not only a provider/program without a specific course).",
        parent=ctle_node,
        critical=True,
    )

    # ----------------- Course_Details_Provided (parallel, critical) -----------------
    details_node = evaluator.add_parallel(
        id="Course_Details_Provided",
        desc="Response includes the required identifying details for the course.",
        parent=ctle_node,
        critical=True,
    )

    provider_provided = bool(primary and primary.provider_name and primary.provider_name.strip())
    evaluator.add_custom_node(
        result=provider_provided,
        id="Provider_Name_Provided",
        desc="Course provider name is provided.",
        parent=details_node,
        critical=True,
    )

    title_provided = bool(primary and primary.course_title and primary.course_title.strip())
    evaluator.add_custom_node(
        result=title_provided,
        id="Course_Title_Provided",
        desc="Course title is provided.",
        parent=details_node,
        critical=True,
    )

    url_provided = bool(primary and primary.reference_urls and len(primary.reference_urls) > 0)
    evaluator.add_custom_node(
        result=url_provided,
        id="Reference_URL_Provided",
        desc="At least one reference URL is provided where the course information can be verified.",
        parent=details_node,
        critical=True,
    )

    # Prepare strings and URLs for verification leaves
    provider_name = _safe_str(primary.provider_name if primary else None)
    course_title = _safe_str(primary.course_title if primary else None)
    sources = primary.reference_urls if primary else []

    # ----------------- State_Approval_Verification (leaf, critical) -----------------
    state_approval_node = evaluator.add_leaf(
        id="State_Approval_Verification",
        desc="Verify that the course provider is an approved CTLE sponsor by the New York State Education Department (NYSED).",
        parent=ctle_node,
        critical=True,
    )
    state_claim = f"{provider_name} is an approved CTLE sponsor by the New York State Education Department (NYSED)."
    await evaluator.verify(
        claim=state_claim,
        node=state_approval_node,
        sources=sources,
        additional_instruction=(
            "Use only the provided URLs. Look for explicit language such as 'NYSED-approved CTLE sponsor', "
            "'CTLE sponsor', 'CTLE Provider', 'CTLE Sponsor ID', or a listing on the official NYSED CTLE sponsor search page. "
            "The page must clearly indicate NYSED approval for CTLE sponsorship. If not explicit, mark as not supported."
        ),
    )

    # ----------------- Course_Characteristics (parallel, critical) -----------------
    characteristics_node = evaluator.add_parallel(
        id="Course_Characteristics",
        desc="Verify that the course meets the specified requirements for credit, format, and subject focus.",
        parent=ctle_node,
        critical=True,
    )

    # Credit_Type (leaf, critical)
    credit_node = evaluator.add_leaf(
        id="Credit_Type",
        desc="Course provides graduate-level university credit that counts toward both CTLE hour requirements and salary advancement.",
        parent=characteristics_node,
        critical=True,
    )
    credit_claim = (
        f"The course '{course_title}' provides graduate-level university credit and those credits count toward CTLE hours "
        f"and are acceptable for salary advancement."
    )
    await evaluator.verify(
        claim=credit_claim,
        node=credit_node,
        sources=sources,
        additional_instruction=(
            "Confirm BOTH of the following from the provided pages: "
            "(1) the course yields graduate-level university credit (semester or quarter credits) issued by a university; and "
            "(2) those credits count toward NYS CTLE hours and are acceptable for salary advancement/lane change. "
            "Accept synonymous phrases such as 'graduate credits', 'university credit', 'CTLE-eligible', 'salary advancement', "
            "'lane change', or 'salary increment'. If either part is missing or only PD hours/CEUs are mentioned without "
            "university graduate credit, mark as not supported."
        ),
    )

    # Delivery_Format (leaf, critical)
    format_node = evaluator.add_leaf(
        id="Delivery_Format",
        desc="Course is offered 100% online in a self-paced format.",
        parent=characteristics_node,
        critical=True,
    )
    format_claim = f"The course '{course_title}' is delivered 100% online and is self-paced (asynchronous)."
    await evaluator.verify(
        claim=format_claim,
        node=format_node,
        sources=sources,
        additional_instruction=(
            "Verify that the course is entirely online and explicitly self-paced/asynchronous. "
            "Accept synonymous phrasing such as 'self-paced', 'asynchronous', 'at your own pace', or 'start anytime'. "
            "If the page indicates fixed live meeting times or in-person components, this should fail."
        ),
    )

    # Subject_Area_Focus (leaf, critical)
    subject_node = evaluator.add_leaf(
        id="Subject_Area_Focus",
        desc="Course focuses on English Language Learners (ELL) or English as a New Language (ENL) instruction.",
        parent=characteristics_node,
        critical=True,
    )
    subject_claim = (
        f"The course '{course_title}' focuses on English Language Learners (ELL) / English as a New Language (ENL) instruction."
    )
    await evaluator.verify(
        claim=subject_claim,
        node=subject_node,
        sources=sources,
        additional_instruction=(
            "Confirm that the central topic of the course is ELL/ENL instruction. "
            "Accept synonyms or equivalent terminology such as ESL, ELD, multilingual learners (MLs), or English learners. "
            "If ELL/ENL is only a minor or incidental mention, mark as not supported."
        ),
    )

    # ----------------- Provider_Accreditation (leaf, critical) -----------------
    accreditation_node = evaluator.add_leaf(
        id="Provider_Accreditation",
        desc="Verify that the course provider is affiliated with or partnered with a regionally accredited university.",
        parent=ctle_node,
        critical=True,
    )
    accreditation_claim = (
        f"{provider_name} is affiliated with or partnered with a regionally accredited university to offer the graduate credit."
    )
    await evaluator.verify(
        claim=accreditation_claim,
        node=accreditation_node,
        sources=sources,
        additional_instruction=(
            "From the provided pages, verify that the provider partners with a specific regionally accredited university "
            "for issuing the graduate credit. Look for explicit mention of the university partner and regional accreditation "
            "or accrediting bodies (e.g., MSCHE, NECHE, HLC, WSCUC, SACSCOC, NWCCU). "
            "Generic 'accredited' statements without a named regionally accredited university are insufficient."
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
    Evaluate an answer for the NYS CTLE-compliant ELL/ENL course selection task.
    """
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

    # Extraction
    extraction = await evaluator.extract(
        prompt=prompt_extract_courses(),
        template_class=CoursesExtraction,
        extraction_name="courses_extraction",
    )

    # Record a brief summary of what was extracted (for transparency)
    primary = select_primary_course(extraction)
    evaluator.add_custom_info(
        info={
            "course_count_extracted": len(extraction.courses),
            "primary_course": {
                "provider_name": primary.provider_name if primary else None,
                "course_title": primary.course_title if primary else None,
                "reference_urls": primary.reference_urls if primary else [],
            },
        },
        info_type="extraction_summary",
        info_name="selected_course_summary",
    )

    # Build verification tree and run verifications
    await verify_ctle_course(evaluator, root, extraction)

    return evaluator.get_summary()