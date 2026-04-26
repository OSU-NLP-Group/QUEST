import asyncio
import logging
from typing import List, Optional, Dict, Any

from pydantic import BaseModel, Field
from obj_task_eval.llm_client.base_client import LLMClient

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "engineering_coop_mandatory_4_universities"
TASK_DESCRIPTION = (
    "I am planning to pursue an engineering degree and am specifically interested in universities that have strong, "
    "structured cooperative education (co-op) programs. I want to find 4 U.S. universities where the engineering co-op "
    "program meets ALL of the following requirements:\n\n"
    "1. The co-op program must be mandatory (required for degree completion) for engineering students\n"
    "2. The program must require students to complete at least 3 co-op rotations or terms\n"
    "3. The program must have a minimum GPA requirement of 2.5 or higher for co-op eligibility\n"
    "4. Students must have completed at least 30 credit hours (or equivalent coursework) before being eligible for their first co-op\n\n"
    "For each university, please provide:\n"
    "- The university name\n"
    "- The specific engineering college or school name (e.g., \"College of Engineering\")\n"
    "- Confirmation that the co-op program is mandatory\n"
    "- The number of required co-op rotations/terms\n"
    "- The minimum GPA requirement for co-op eligibility\n"
    "- The minimum credit hours requirement before first co-op\n"
    "- Reference URLs to official university pages that verify each of these requirements"
)


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class UniversityItem(BaseModel):
    university_name: Optional[str] = None
    engineering_college_name: Optional[str] = None

    # Free-text fields as stated in the answer (keep as strings for robustness)
    mandatory_status: Optional[str] = None
    required_rotations: Optional[str] = None
    minimum_gpa: Optional[str] = None
    minimum_credit_hours_before_first_coop: Optional[str] = None

    # Dedicated URL lists per requirement (explicitly mentioned in the answer)
    mandatory_urls: List[str] = Field(default_factory=list)
    rotations_urls: List[str] = Field(default_factory=list)
    gpa_urls: List[str] = Field(default_factory=list)
    credit_urls: List[str] = Field(default_factory=list)


class UniversitiesExtraction(BaseModel):
    universities: List[UniversityItem] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_universities() -> str:
    return """
    Extract from the answer a list of universities (in the United States) that the answer claims meet ALL specified co-op program criteria.
    For each university mentioned in the answer, extract the following fields exactly as stated in the answer text:

    - university_name: The university name (string)
    - engineering_college_name: The specific engineering college or school name (string), if provided
    - mandatory_status: The description confirming the co-op is mandatory/required (string, verbatim from the answer if present; else null)
    - required_rotations: The number of required co-op rotations/terms as described (string, e.g., '3', 'three terms', 'at least 3'; else null)
    - minimum_gpa: The minimum GPA requirement as described (string, e.g., '2.5', '3.0', '2.7'; else null)
    - minimum_credit_hours_before_first_coop: The minimum credit hours requirement before first co-op as described (string, e.g., '30 credits', '30 semester hours'; else null)

    Also extract the reference URLs that the answer uses to support each specific requirement. Only include URLs that are explicitly present in the answer text (plain or in markdown). Do not invent or infer any URLs.
    - mandatory_urls: list of URLs that substantiate the mandatory/required co-op status for engineering students
    - rotations_urls: list of URLs that substantiate the required number of co-op rotations/terms
    - gpa_urls: list of URLs that substantiate the minimum GPA requirement for co-op eligibility
    - credit_urls: list of URLs that substantiate the minimum credit hours (or equivalent coursework) required before the first co-op

    Rules:
    - If a given field is not mentioned in the answer, return null for that field (or an empty list for URLs).
    - For each university, keep URLs separated by requirement as listed above. Do not merge them.
    - Return all universities found in the answer in the 'universities' array, preserving order of appearance.
    """


# --------------------------------------------------------------------------- #
# Verification helpers                                                        #
# --------------------------------------------------------------------------- #
def _bool_nonempty_str(s: Optional[str]) -> bool:
    return bool(s and str(s).strip())


def _has_urls(urls: Optional[List[str]]) -> bool:
    return bool(urls and len(urls) > 0)


# --------------------------------------------------------------------------- #
# University verification                                                     #
# --------------------------------------------------------------------------- #
async def verify_one_university(
    evaluator: Evaluator,
    parent_node,
    uni: UniversityItem,
    idx: int,
) -> None:
    """
    Build the verification subtree for a single university and execute all checks.
    The structure follows the rubric:
    - Basic Info (critical)
    - Mandatory Status (critical)
    - Rotation Count (critical)
    - GPA Requirement (critical)
    - Credit Hours (critical)
    """
    uni_label = f"University #{idx + 1}"
    uni_node = evaluator.add_parallel(
        id=f"University_{idx + 1}",
        desc=f"{uni_label} meets all specified co-op program requirements",
        parent=parent_node,
        critical=False  # allow partial credit across different universities
    )

    # -------------------- Basic Info (Critical group) -------------------- #
    basic_info = evaluator.add_parallel(
        id=f"University_{idx + 1}_Basic_Info",
        desc="University name and engineering college/school name are provided",
        parent=uni_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=_bool_nonempty_str(uni.university_name),
        id=f"University_{idx + 1}_Name",
        desc="Valid U.S. university name is provided",
        parent=basic_info,
        critical=True
    )
    evaluator.add_custom_node(
        result=_bool_nonempty_str(uni.engineering_college_name),
        id=f"University_{idx + 1}_College_Name",
        desc="Specific engineering college or school name is provided",
        parent=basic_info,
        critical=True
    )

    # ---------------- Mandatory Status (Critical group) ------------------ #
    mandatory_grp = evaluator.add_parallel(
        id=f"University_{idx + 1}_Mandatory_Status",
        desc="The co-op program is mandatory (required for degree completion) for engineering students",
        parent=uni_node,
        critical=True
    )
    # URL existence check (critical sibling)
    evaluator.add_custom_node(
        result=_has_urls(uni.mandatory_urls),
        id=f"University_{idx + 1}_Mandatory_URL",
        desc="Reference URL provided for mandatory status verification",
        parent=mandatory_grp,
        critical=True
    )
    # Evidence-backed verification
    mandatory_leaf = evaluator.add_leaf(
        id=f"University_{idx + 1}_Mandatory_Verification",
        desc="Official university documentation confirms co-op is a degree requirement",
        parent=mandatory_grp,
        critical=True
    )
    mandatory_claim = (
        f"The official page(s) indicate that the engineering co-op program at "
        f"{uni.university_name or 'the university'} is mandatory (i.e., required for degree completion) "
        f"for engineering students."
    )

    # ---------------- Rotation Count (Critical group) -------------------- #
    rotation_grp = evaluator.add_parallel(
        id=f"University_{idx + 1}_Rotation_Count",
        desc="The program requires at least 3 co-op rotations/terms",
        parent=uni_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=_has_urls(uni.rotations_urls),
        id=f"University_{idx + 1}_Rotation_URL",
        desc="Reference URL provided for rotation count verification",
        parent=rotation_grp,
        critical=True
    )
    rotation_leaf = evaluator.add_leaf(
        id=f"University_{idx + 1}_Rotation_Number",
        desc="The stated number of required co-op rotations is 3 or more",
        parent=rotation_grp,
        critical=True
    )
    rotation_claim = (
        f"The engineering co-op curriculum at {uni.university_name or 'the university'} requires "
        f"at least 3 co-op rotations/terms (e.g., three or more co-op semesters)."
    )

    # ---------------- GPA Requirement (Critical group) ------------------- #
    gpa_grp = evaluator.add_parallel(
        id=f"University_{idx + 1}_GPA_Requirement",
        desc="The program has minimum GPA requirement of 2.5 or higher for co-op eligibility",
        parent=uni_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=_has_urls(uni.gpa_urls),
        id=f"University_{idx + 1}_GPA_URL",
        desc="Reference URL provided for GPA requirement verification",
        parent=gpa_grp,
        critical=True
    )
    gpa_leaf = evaluator.add_leaf(
        id=f"University_{idx + 1}_GPA_Value",
        desc="The stated minimum GPA is 2.5 or higher",
        parent=gpa_grp,
        critical=True
    )
    gpa_claim = (
        f"The minimum GPA requirement for co-op eligibility at {uni.university_name or 'the university'} "
        f"is at least 2.5 (e.g., 2.5, 2.7, 3.0 all satisfy this)."
    )

    # ---------------- Credit Hours (Critical group) ---------------------- #
    credit_grp = evaluator.add_parallel(
        id=f"University_{idx + 1}_Credit_Hours",
        desc="The program requires at least 30 credit hours completed before first co-op",
        parent=uni_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=_has_urls(uni.credit_urls),
        id=f"University_{idx + 1}_Credit_URL",
        desc="Reference URL provided for credit hours requirement verification",
        parent=credit_grp,
        critical=True
    )
    credit_leaf = evaluator.add_leaf(
        id=f"University_{idx + 1}_Credit_Value",
        desc="The stated minimum credit hours is 30 or more",
        parent=credit_grp,
        critical=True
    )
    credit_claim = (
        f"Students must have completed at least 30 credit hours (or equivalent coursework) "
        f"before being eligible for their first engineering co-op at "
        f"{uni.university_name or 'the university'}."
    )

    # ---------------- Execute evidence-backed verifications -------------- #
    # Each verification will be auto-gated by its critical sibling URL-existence check
    # due to Evaluator's automatic precondition mechanism.
    claims_and_sources = [
        (
            mandatory_claim,
            uni.mandatory_urls,
            mandatory_leaf,
            "Verify the page(s) explicitly indicate the co-op is required/mandatory to graduate for engineering students. "
            "Synonyms like 'required co-op', 'mandatory co-op', or 'co-op required for degree' should count. "
            "If the page says 'optional', 'voluntary', or 'encouraged' only, it should fail."
        ),
        (
            rotation_claim,
            uni.rotations_urls,
            rotation_leaf,
            "Confirm that the page(s) explicitly require at least 3 co-op rotations/terms (e.g., '3 required co-ops', "
            "'three co-op terms minimum', or 'at least three co-op semesters'). Accept phrasing like '3 or more'."
        ),
        (
            gpa_claim,
            uni.gpa_urls,
            gpa_leaf,
            "Identify the minimum GPA for co-op eligibility and ensure it is 2.5 or higher. "
            "If multiple minima are listed for subgroups, use the lowest listed value; pass only if that lowest value is >= 2.5."
        ),
        (
            credit_claim,
            uni.credit_urls,
            credit_leaf,
            "Identify the minimum completed credits (or equivalent coursework/semester hours) required before the first co-op. "
            "Pass only if it is 30 or higher. Recognize synonyms like '30 hours', 'sophomore standing with 30 credits', etc."
        ),
    ]
    await evaluator.batch_verify(claims_and_sources)


# --------------------------------------------------------------------------- #
# Main evaluation entry point                                                 #
# --------------------------------------------------------------------------- #
async def evaluate_answer(
    client: LLMClient,
    answer: str,
    agent_name: str,
    answer_name: str,
    cache: CacheFileSys,
    semaphore: asyncio.Semaphore,
    logger: logging.Logger,
    model: str = "o4-mini",
) -> Dict[str, Any]:
    """
    Evaluate an answer for the 'mandatory engineering co-op' task using the Mind2Web2 evaluation framework.
    Returns a standardized summary dictionary with the verification tree and scores.
    """
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Independent universities → parallel aggregation
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
    extracted = await evaluator.extract(
        prompt=prompt_extract_universities(),
        template_class=UniversitiesExtraction,
        extraction_name="universities_extraction",
    )

    # Normalize to exactly 4 entries (pad with empty stubs if fewer; take first 4 if more)
    universities = list(extracted.universities or [])
    if len(universities) < 4:
        universities.extend([UniversityItem() for _ in range(4 - len(universities))])
    else:
        universities = universities[:4]

    # Add a note of the constraints for transparency (non-scoring)
    evaluator.add_custom_info(
        info={
            "requirements": {
                "mandatory_coop": True,
                "min_rotations": 3,
                "min_gpa": ">= 2.5",
                "min_credits_before_first_coop": ">= 30"
            },
            "expected_universities_count": 4
        },
        info_type="constraints",
        info_name="task_requirements"
    )

    # Build verification subtrees for each university
    tasks = []
    for i in range(4):
        tasks.append(verify_one_university(evaluator, root, universities[i], i))
    await asyncio.gather(*tasks)

    # Return structured summary
    return evaluator.get_summary()