import asyncio
import logging
import re
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy

# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "msu_presidential_scholarship_renewal_eligibility"
TASK_DESCRIPTION = (
    "A first-year student at Montana State University was awarded a Presidential Scholarship and has just completed "
    "their first academic year with a cumulative GPA of 3.6 and 31 credit hours. Based on the university's official "
    "renewal requirements for the Presidential Scholarship, does this student meet the criteria to renew their "
    "scholarship for the second year? Provide your answer with specific reference to the GPA and credit hour requirements."
)

# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class RenewalPolicy(BaseModel):
    policy_urls: List[str] = Field(default_factory=list)
    gpa_min_required: Optional[str] = None
    credit_hours_required_per_year: Optional[str] = None


class StudentRecord(BaseModel):
    cumulative_gpa: Optional[str] = None
    credit_hours_completed: Optional[str] = None


class RenewalExtraction(BaseModel):
    policy: RenewalPolicy = RenewalPolicy()
    student: StudentRecord = StudentRecord()


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_renewal_info() -> str:
    return (
        "Extract the following information from the answer:\n"
        "1) policy.policy_urls: A list of official Montana State University URL(s) that the answer cites for the "
        "Presidential Scholarship renewal requirements. Only include real URLs explicitly present in the answer. If none, return an empty list.\n"
        "2) policy.gpa_min_required: The minimum cumulative GPA required for renewal as stated in the answer (e.g., '3.5'). If not stated, return null.\n"
        "3) policy.credit_hours_required_per_year: The number of credit hours per academic year required for renewal as stated in the answer (e.g., '32'). If not stated, return null.\n"
        "4) student.cumulative_gpa: The student's cumulative GPA mentioned in the answer. If not stated, return null.\n"
        "5) student.credit_hours_completed: The student's total credit hours completed in the academic year mentioned in the answer. If not stated, return null.\n"
        "Return a JSON object matching the RenewalExtraction schema."
    )


# --------------------------------------------------------------------------- #
# Helper parsing functions                                                    #
# --------------------------------------------------------------------------- #
def safe_parse_float(text: Optional[str]) -> Optional[float]:
    if text is None:
        return None
    try:
        # Extract the first float-like number in the string
        match = re.search(r"(\d+(?:\.\d+)?)", text.strip())
        if match:
            return float(match.group(1))
        return None
    except Exception:
        return None


def safe_parse_int(text: Optional[str]) -> Optional[int]:
    if text is None:
        return None
    try:
        match = re.search(r"(\d+)", text.strip())
        if match:
            return int(match.group(1))
        return None
    except Exception:
        return None


# --------------------------------------------------------------------------- #
# Verification tree construction                                              #
# --------------------------------------------------------------------------- #
async def build_renewal_verification_tree(
    evaluator: Evaluator,
    parent_node,
    extraction: RenewalExtraction,
) -> None:
    """
    Build the verification tree according to the rubric:
    - Presidential_Scholarship_Renewal_Eligibility (critical, parallel)
      - GPA_Requirement (critical, sequential)
          • gpa_threshold_stated (critical, custom)
          • gpa_policy_supported (critical, verify_by_urls)
          • student_gpa_stated (critical, custom)
          • gpa_meets_requirement_calc (critical, custom)
      - Credit_Hour_Requirement (critical, sequential)
          • credit_threshold_stated (critical, custom)
          • credit_policy_supported (critical, verify_by_urls)
          • student_credit_hours_stated (critical, custom)
          • credit_hours_meet_requirement_calc (critical, custom)
      - policy_sources_present (critical, custom) – used as a prerequisite for policy verifications
    """
    # Container node mirroring the rubric root (critical, parallel)
    elig_node = evaluator.add_parallel(
        id="Presidential_Scholarship_Renewal_Eligibility",
        desc="Evaluates whether the student meets Montana State University Presidential Scholarship renewal requirements",
        parent=parent_node,
        critical=True
    )

    # Ensure official policy URLs are provided (critical prerequisite for policy verification)
    policy_urls_present = evaluator.add_custom_node(
        result=bool(extraction.policy.policy_urls),
        id="policy_sources_present",
        desc="Official MSU renewal policy URL(s) are provided in the answer",
        parent=elig_node,
        critical=True
    )

    # ---------------------- GPA Requirement Subtree ---------------------- #
    gpa_req = evaluator.add_sequential(
        id="GPA_Requirement",
        desc="Student maintains a minimum cumulative GPA of 3.5",
        parent=elig_node,
        critical=True
    )

    # 1) GPA threshold stated in the answer (critical)
    gpa_threshold_stated = evaluator.add_custom_node(
        result=bool(extraction.policy.gpa_min_required and extraction.policy.gpa_min_required.strip()),
        id="gpa_threshold_stated",
        desc="The minimum cumulative GPA requirement is stated in the answer",
        parent=gpa_req,
        critical=True
    )

    # 2) GPA policy supported by cited official source(s) (critical)
    gpa_policy_leaf = evaluator.add_leaf(
        id="gpa_policy_supported",
        desc="Official source(s) confirm the minimum cumulative GPA requirement",
        parent=gpa_req,
        critical=True
    )
    gpa_policy_claim = (
        f"Montana State University's Presidential Scholarship renewal requires a minimum cumulative GPA of "
        f"{extraction.policy.gpa_min_required}."
    )
    await evaluator.verify(
        claim=gpa_policy_claim,
        node=gpa_policy_leaf,
        sources=extraction.policy.policy_urls,
        additional_instruction=(
            "Verify this exact GPA threshold on the official MSU page(s) cited in the answer. "
            "Allow minor phrasing variations (e.g., 'minimum cumulative GPA of 3.5 on a 4.0 scale')."
        ),
        extra_prerequisites=[policy_urls_present, gpa_threshold_stated]
    )

    # 3) Student GPA stated (critical)
    student_gpa_stated = evaluator.add_custom_node(
        result=bool(extraction.student.cumulative_gpa and extraction.student.cumulative_gpa.strip()),
        id="student_gpa_stated",
        desc="The student's cumulative GPA is stated in the answer",
        parent=gpa_req,
        critical=True
    )

    # 4) Student GPA meets the threshold (critical, computed)
    gpa_threshold_val = safe_parse_float(extraction.policy.gpa_min_required)
    student_gpa_val = safe_parse_float(extraction.student.cumulative_gpa)
    gpa_meets = (student_gpa_val is not None and gpa_threshold_val is not None and student_gpa_val >= gpa_threshold_val)

    evaluator.add_custom_node(
        result=gpa_meets,
        id="gpa_meets_requirement_calc",
        desc=(
            f"GPA comparison: student GPA {extraction.student.cumulative_gpa} "
            f"vs required minimum {extraction.policy.gpa_min_required}"
        ),
        parent=gpa_req,
        critical=True
    )

    # ------------------- Credit Hour Requirement Subtree ----------------- #
    credit_req = evaluator.add_sequential(
        id="Credit_Hour_Requirement",
        desc="Student completes 32 credit hours per academic year",
        parent=elig_node,
        critical=True
    )

    # 1) Credit-hour threshold stated in the answer (critical)
    credit_threshold_stated = evaluator.add_custom_node(
        result=bool(extraction.policy.credit_hours_required_per_year and extraction.policy.credit_hours_required_per_year.strip()),
        id="credit_threshold_stated",
        desc="The annual credit-hour requirement is stated in the answer",
        parent=credit_req,
        critical=True
    )

    # 2) Credit-hour policy supported by cited official source(s) (critical)
    credit_policy_leaf = evaluator.add_leaf(
        id="credit_policy_supported",
        desc="Official source(s) confirm the annual credit-hour requirement",
        parent=credit_req,
        critical=True
    )
    credit_policy_claim = (
        f"Montana State University's Presidential Scholarship renewal requires completion of "
        f"{extraction.policy.credit_hours_required_per_year} credit hours per academic year."
    )
    await evaluator.verify(
        claim=credit_policy_claim,
        node=credit_policy_leaf,
        sources=extraction.policy.policy_urls,
        additional_instruction=(
            "Verify on the official MSU page(s) that the annual credit requirement is exactly as stated. "
            "The page may phrase this as 'complete at least 32 credits during the academic year' "
            "or similar language combining Fall + Spring."
        ),
        extra_prerequisites=[policy_urls_present, credit_threshold_stated]
    )

    # 3) Student credit hours stated (critical)
    student_credits_stated = evaluator.add_custom_node(
        result=bool(extraction.student.credit_hours_completed and extraction.student.credit_hours_completed.strip()),
        id="student_credit_hours_stated",
        desc="The student's completed credit hours are stated in the answer",
        parent=credit_req,
        critical=True
    )

    # 4) Student credit hours meet the threshold (critical, computed)
    credit_threshold_val = safe_parse_int(extraction.policy.credit_hours_required_per_year)
    student_credits_val = safe_parse_int(extraction.student.credit_hours_completed)
    credits_meet = (student_credits_val is not None and credit_threshold_val is not None and student_credits_val >= credit_threshold_val)

    evaluator.add_custom_node(
        result=credits_meet,
        id="credit_hours_meet_requirement_calc",
        desc=(
            f"Credit-hours comparison: student completed {extraction.student.credit_hours_completed} "
            f"vs required {extraction.policy.credit_hours_required_per_year}"
        ),
        parent=credit_req,
        critical=True
    )

    # Record custom info summary
    evaluator.add_custom_info(
        info={
            "policy_urls": extraction.policy.policy_urls,
            "gpa_min_required": extraction.policy.gpa_min_required,
            "credit_hours_required": extraction.policy.credit_hours_required_per_year,
            "student_gpa": extraction.student.cumulative_gpa,
            "student_credit_hours": extraction.student.credit_hours_completed,
            "computed_gpa_meets": gpa_meets,
            "computed_credits_meet": credits_meet
        },
        info_type="extraction_summary",
        info_name="renewal_extraction_summary"
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
    Evaluate the answer for MSU Presidential Scholarship renewal eligibility.
    """
    # Initialize evaluator (root is non-critical by framework design)
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

    # Extract structured information from the answer
    extraction = await evaluator.extract(
        prompt=prompt_extract_renewal_info(),
        template_class=RenewalExtraction,
        extraction_name="renewal_extraction"
    )

    # Optionally, record ground truth context (from task description)
    evaluator.add_ground_truth({
        "student_given_context": {
            "cumulative_gpa": "3.6",
            "credit_hours_completed": "31"
        },
        "note": "These values are from the task description for contextual reference."
    }, gt_type="task_context")

    # Build the verification tree according to rubric
    await build_renewal_verification_tree(evaluator, root, extraction)

    # Return the structured summary
    return evaluator.get_summary()