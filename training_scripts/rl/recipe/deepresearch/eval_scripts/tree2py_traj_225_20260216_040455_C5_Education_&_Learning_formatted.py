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
TASK_ID = "nchsaa_transfer_eligibility_2026"
TASK_DESCRIPTION = (
    "A family is relocating from Birmingham, Alabama to Raleigh, North Carolina in July 2026. "
    "Their child is a 10th-grade student-athlete who will be enrolling in a Wake County Public Schools high school "
    "at the start of 11th grade (August 2026) and wants to participate in varsity athletics. "
    "The family is making a permanent move (selling their Alabama home and purchasing a home in North Carolina). "
    "Based on the North Carolina High School Athletic Association (NCHSAA) rules and regulations, provide a comprehensive "
    "explanation of all athletic eligibility requirements the student must meet, including: "
    "(1) Previous Semester Requirements - What attendance and academic performance standards must the student have met in "
    "their last semester at their Alabama school? "
    "(2) Current Enrollment Requirements - What are the age, enrollment timing, and enrollment status requirements the student "
    "must satisfy at their new North Carolina school? "
    "(3) Transfer Rules and Exceptions - What is the standard waiting period for transfer students, and what exceptions might "
    "allow immediate varsity athletic eligibility in this situation? "
    "(4) Documentation - What records or documentation will be needed to verify athletic eligibility? "
    "For each requirement, provide specific details such as percentage thresholds, time limits, numerical standards, and criteria, "
    "along with supporting references from official sources."
)

# --------------------------------------------------------------------------- #
# Expected rule texts (for claims to verify against cited sources)            #
# --------------------------------------------------------------------------- #
EXPECTED_RULES = {
    "prev_attendance_claim": (
        "NCHSAA requires that to be eligible, a student must have attended at least 85% of class days in the previous semester; "
        "for a 90‑day semester this equates to 77 days."
    ),
    "prev_min_load_claim": (
        "Under NCHSAA rules, in the previous semester a student must have passed a minimum load of courses: "
        "either 5 courses on a traditional schedule, or 3 courses on a block schedule, or 6 of 8 courses on an A/B block schedule."
    ),
    "age_limit_claim": (
        "Under NCHSAA rules, a student is ineligible if they turn 19 on or before August 31 of the current school year."
    ),
    "enroll_timing_claim": (
        "Under NCHSAA rules, a student must enroll within the first 15 days of the semester to be eligible."
    ),
    "regular_enrollment_claim": (
        "Under NCHSAA rules, the student must be regularly enrolled at the school; "
        "if there is no local board policy defining regular enrollment, the student must be enrolled in at least one‑half of the minimum load."
    ),
    "transfer_wait_claim": (
        "Under NCHSAA transfer rules, a student who transfers schools is ineligible for varsity athletics at the new school for 365 days "
        "unless an exception applies."
    ),
    "bona_fide_claim": (
        "Under NCHSAA rules, a bona fide change of residence by the student's parents or legal guardians into the new school's "
        "attendance zone can grant immediate varsity eligibility."
    ),
    "mutual_waiver_claim": (
        "Under NCHSAA rules, the sending and receiving schools/districts (principals and superintendents or their designees) may "
        "mutually agree to waive the 365‑day transfer ineligibility, allowing immediate eligibility."
    ),
    "transcript_required_claim": (
        "To verify athletic eligibility, the student's official transcript or academic records from the prior school are required."
    ),
    "attendance_records_required_claim": (
        "To verify athletic eligibility, attendance records from the prior school are required."
    ),
}

# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class ClauseSources(BaseModel):
    text: Optional[str] = None
    urls: List[str] = Field(default_factory=list)


class NCHSAAExtraction(BaseModel):
    prev_attendance: Optional[ClauseSources] = None
    prev_min_load: Optional[ClauseSources] = None
    age_limit: Optional[ClauseSources] = None
    enrollment_timing: Optional[ClauseSources] = None
    regular_enrollment: Optional[ClauseSources] = None
    transfer_wait: Optional[ClauseSources] = None
    bona_fide: Optional[ClauseSources] = None
    mutual_waiver: Optional[ClauseSources] = None
    transcript_required: Optional[ClauseSources] = None
    attendance_records_required: Optional[ClauseSources] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_nchsaa_requirements() -> str:
    return """
Extract, exactly as presented in the answer, the requirement statements and the official source URLs associated with each category below. 
For each category, return:
- text: the exact sentence(s) from the answer describing this rule/requirement. If the answer did not state it, return null.
- urls: a list of official source URLs (as shown in the answer) that support this specific category. If none are cited, return an empty list.

Categories and what to look for:
1) prev_attendance:
   - Look for the previous-semester attendance rule with the numeric threshold "85%" AND include the regular semester example "77 of 90 days" if present.
2) prev_min_load:
   - Look for the previous-semester minimum load PASSED rule with the three options:
     "5 (traditional) OR 3 (block) OR 6 of 8 (A/B block)".
3) age_limit:
   - Look for the age rule: "must NOT turn 19 on or before August 31 of the current school year".
4) enrollment_timing:
   - Look for the enrollment timing rule: "must enroll within the first 15 days of the semester".
5) regular_enrollment:
   - Look for the regular enrollment status rule: "must be regularly enrolled; if no local policy, ≥ one-half of the minimum load".
6) transfer_wait:
   - Look for the transfer rule: "standard 365-day waiting period" for varsity eligibility unless exceptions apply.
7) bona_fide:
   - Look for the exception: "bona fide change of residence into the new school's attendance zone allows immediate eligibility".
8) mutual_waiver:
   - Look for the exception/waiver: "both schools/districts can mutually agree to waive the 365-day period".
9) transcript_required:
   - Look for documentation: "official transcript/academic records from prior school are required to verify eligibility".
10) attendance_records_required:
   - Look for documentation: "prior-school attendance records are required to verify eligibility".

Important:
- Extract only URLs explicitly included in the answer. Include multiple URLs if the answer associates them with the category.
- Do NOT invent any URL. If the answer lists general references without indicating which rule they support, assign them to the most relevant category based on context and wording.
- If the answer cites no sources for a category, return an empty list for URLs for that category.
"""


# --------------------------------------------------------------------------- #
# Helper verification builders                                                #
# --------------------------------------------------------------------------- #
async def verify_clause_with_text_and_sources(
    evaluator: Evaluator,
    parent_node,
    *,
    id_prefix: str,
    parent_desc: str,
    stated_claim: str,
    support_claim: str,
    clause: Optional[ClauseSources],
    require_official_sources_instruction: Optional[str] = None,
    critical: bool = True,
) -> None:
    """
    Build a sequential critical node that:
      1) Checks the answer states the required detail (simple verify against answer text).
      2) Ensures official source URLs were provided (custom existence node).
      3) Verifies the claim is supported by the cited URLs (verify_by_urls).
    """
    seq_node = evaluator.add_sequential(
        id=id_prefix,
        desc=parent_desc,
        parent=parent_node,
        critical=critical
    )

    # 1) Answer states the detail
    stated_leaf = evaluator.add_leaf(
        id=f"{id_prefix}_stated",
        desc=f"Answer states the required detail for: {id_prefix}",
        parent=seq_node,
        critical=True
    )
    await evaluator.verify(
        claim=stated_claim,
        node=stated_leaf,
        # No URL evidence here; this is a check against the answer text itself
        additional_instruction="Judge this solely by reading the provided answer text above. "
                               "Minor wording variations are acceptable, but the required numeric criteria must be explicitly present."
    )

    # 2) Official source URLs provided
    urls = clause.urls if clause and clause.urls else []
    sources_present = evaluator.add_custom_node(
        result=(len(urls) > 0),
        id=f"{id_prefix}_sources_present",
        desc=f"Official source URLs are provided in the answer for: {id_prefix}",
        parent=seq_node,
        critical=True
    )

    # 3) Claim supported by cited official sources
    supported_leaf = evaluator.add_leaf(
        id=f"{id_prefix}_supported",
        desc=f"Cited official sources support the rule for: {id_prefix}",
        parent=seq_node,
        critical=True
    )
    instructions = require_official_sources_instruction or (
        "Treat only official NCHSAA Handbook/policy pages or official North Carolina K‑12 district/school policy pages as acceptable. "
        "Verify the exact numerical thresholds/limits/dates in the claim are stated or clearly implied on the cited page(s). "
        "If URLs are irrelevant, unofficial, or do not support the claim, return not supported."
    )
    await evaluator.verify(
        claim=support_claim,
        node=supported_leaf,
        sources=urls,
        additional_instruction=instructions
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
    Evaluate an answer for the NCHSAA transfer eligibility scenario.
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
        default_model=model
    )

    # Extract structured info: requirement statements and their cited URLs
    extraction = await evaluator.extract(
        prompt=prompt_extract_nchsaa_requirements(),
        template_class=NCHSAAExtraction,
        extraction_name="nchsaa_extraction"
    )

    # Record expected rule set (for transparency/debugging only)
    evaluator.add_ground_truth({
        "expected_rules": EXPECTED_RULES
    }, gt_type="expected_rules")

    # Build the rubric tree reflecting the provided JSON

    # Top-level: NCHSAA_Transfer_Eligibility (critical parallel)
    nchsaa_root = evaluator.add_parallel(
        id="NCHSAA_Transfer_Eligibility",
        desc="Explain the required NCHSAA eligibility rules for this transfer scenario, using the provided constraints and citing official sources.",
        parent=root,
        critical=True
    )

    # 1) Previous Semester Requirements (critical parallel)
    prev_semester_node = evaluator.add_parallel(
        id="Previous_Semester_Requirements",
        desc="Previous-semester eligibility standards from the prior (Alabama) school term.",
        parent=nchsaa_root,
        critical=True
    )

    # 1.a) Previous Semester Attendance
    await verify_clause_with_text_and_sources(
        evaluator,
        prev_semester_node,
        id_prefix="Previous_Semester_Attendance",
        parent_desc="States the prior-semester attendance requirement (≥85%, including the provided 77-of-90-days regular semester example) and provides an official-source reference.",
        stated_claim="The answer explicitly states a previous-semester attendance requirement of at least 85% and includes the example '77 of 90 days' (or equivalent phrasing) for a regular 90-day semester.",
        support_claim=EXPECTED_RULES["prev_attendance_claim"],
        clause=extraction.prev_attendance,
        require_official_sources_instruction=(
            "Accept only official NCHSAA or official NC district/school policy pages. "
            "Verify the page states an 85% prior-semester attendance requirement and gives (or is consistent with) the example of 77 days for a 90-day semester."
        ),
        critical=True
    )

    # 1.b) Previous Semester Minimum Load Passed
    await verify_clause_with_text_and_sources(
        evaluator,
        prev_semester_node,
        id_prefix="Previous_Semester_Minimum_Load_Passed",
        parent_desc="States the prior-semester academic minimum-load passed requirement (5 traditional OR 3 block OR 6 of 8 A/B block) and provides an official-source reference.",
        stated_claim="The answer states the previous-semester minimum load passed requirement as: 5 courses (traditional) OR 3 courses (block) OR 6 of 8 courses (A/B block).",
        support_claim=EXPECTED_RULES["prev_min_load_claim"],
        clause=extraction.prev_min_load,
        require_official_sources_instruction=(
            "Accept only official NCHSAA or official NC district/school policy pages. "
            "Verify the minimum passed course load options: 5 traditional OR 3 block OR 6 of 8 A/B block."
        ),
        critical=True
    )

    # 2) Current Enrollment Requirements (critical parallel)
    current_enroll_node = evaluator.add_parallel(
        id="Current_Enrollment_Requirements",
        desc="Current eligibility requirements at the new North Carolina school.",
        parent=nchsaa_root,
        critical=True
    )

    # 2.a) Age Requirement
    await verify_clause_with_text_and_sources(
        evaluator,
        current_enroll_node,
        id_prefix="Age_Requirement",
        parent_desc="States the age limit (must not turn 19 on or before Aug 31 of the current school year) and provides an official-source reference.",
        stated_claim="The answer states that a student must not turn 19 on or before August 31 of the current school year.",
        support_claim=EXPECTED_RULES["age_limit_claim"],
        clause=extraction.age_limit,
        require_official_sources_instruction=(
            "Accept only official NCHSAA or official NC district/school policy pages. "
            "Verify the Aug 31 age cutoff policy (ineligible if 19 on/before Aug 31)."
        ),
        critical=True
    )

    # 2.b) Enrollment Timing Requirement
    await verify_clause_with_text_and_sources(
        evaluator,
        current_enroll_node,
        id_prefix="Enrollment_Timing_Requirement",
        parent_desc="States the enrollment timing requirement (must enroll within the first 15 days of the semester) and provides an official-source reference.",
        stated_claim="The answer states that a student must enroll within the first 15 days of the semester.",
        support_claim=EXPECTED_RULES["enroll_timing_claim"],
        clause=extraction.enrollment_timing,
        require_official_sources_instruction=(
            "Accept only official NCHSAA or official NC district/school policy pages. "
            "Verify the 'within the first 15 days of the semester' eligibility requirement."
        ),
        critical=True
    )

    # 2.c) Regular Enrollment Status Requirement
    await verify_clause_with_text_and_sources(
        evaluator,
        current_enroll_node,
        id_prefix="Regular_Enrollment_Status_Requirement",
        parent_desc="States the regular-enrollment status requirement (regularly enrolled; if no local policy then ≥ half the minimum load) and provides an official-source reference.",
        stated_claim="The answer states that the student must be regularly enrolled; if no local policy defines regular enrollment, the student must be enrolled in at least one-half of the minimum load.",
        support_claim=EXPECTED_RULES["regular_enrollment_claim"],
        clause=extraction.regular_enrollment,
        require_official_sources_instruction=(
            "Accept only official NCHSAA or official NC district/school policy pages. "
            "Verify both parts: regularly enrolled AND, if no local policy, at least one-half of the minimum load."
        ),
        critical=True
    )

    # 3) Transfer Rules and Exceptions (critical parallel)
    transfer_node = evaluator.add_parallel(
        id="Transfer_Rules_and_Exceptions",
        desc="Transfer rule and the exceptions/waivers relevant to the stated permanent move.",
        parent=nchsaa_root,
        critical=True
    )

    # 3.a) Standard Transfer Waiting Period
    await verify_clause_with_text_and_sources(
        evaluator,
        transfer_node,
        id_prefix="Standard_Transfer_Waiting_Period",
        parent_desc="States the standard transfer waiting period (365 days) and provides an official-source reference.",
        stated_claim="The answer states that the standard transfer waiting period is 365 days for varsity eligibility at the new school unless an exception applies.",
        support_claim=EXPECTED_RULES["transfer_wait_claim"],
        clause=extraction.transfer_wait,
        require_official_sources_instruction=(
            "Accept only official NCHSAA or official NC district/school policy pages. "
            "Verify that the standard transfer ineligibility period is 365 days."
        ),
        critical=True
    )

    # 3.b) Bona Fide Change of Residence Exception
    await verify_clause_with_text_and_sources(
        evaluator,
        transfer_node,
        id_prefix="Bona_Fide_Change_of_Residence_Exception",
        parent_desc="States that a bona fide change of residence can allow immediate eligibility and provides an official-source reference.",
        stated_claim="The answer states that a bona fide change of residence into the new school's attendance zone can allow immediate varsity eligibility.",
        support_claim=EXPECTED_RULES["bona_fide_claim"],
        clause=extraction.bona_fide,
        require_official_sources_instruction=(
            "Accept only official NCHSAA or official NC district/school policy pages. "
            "Verify that a bona fide change of residence allows immediate eligibility."
        ),
        critical=True
    )

    # 3.c) Mutual School Agreement Waiver
    await verify_clause_with_text_and_sources(
        evaluator,
        transfer_node,
        id_prefix="Mutual_School_Agreement_Waiver",
        parent_desc="States that both schools can mutually agree to waive the 365-day period and provides an official-source reference.",
        stated_claim="The answer states that the sending and receiving schools/districts can mutually agree to waive the 365-day transfer ineligibility period to allow immediate eligibility.",
        support_claim=EXPECTED_RULES["mutual_waiver_claim"],
        clause=extraction.mutual_waiver,
        require_official_sources_instruction=(
            "Accept only official NCHSAA or official NC district/school policy pages. "
            "Verify that a mutual agreement/waiver process exists allowing immediate eligibility despite the 365-day rule."
        ),
        critical=True
    )

    # 4) Documentation Requirements (critical parallel)
    docs_node = evaluator.add_parallel(
        id="Documentation_Requirements",
        desc="Documents needed to verify eligibility.",
        parent=nchsaa_root,
        critical=True
    )

    # 4.a) Transcript Required
    await verify_clause_with_text_and_sources(
        evaluator,
        docs_node,
        id_prefix="Transcript_Required",
        parent_desc="Identifies that the prior-school academic transcript/records are required for verification.",
        stated_claim="The answer identifies that the student's official transcript or academic records from the prior school are required to verify athletic eligibility.",
        support_claim=EXPECTED_RULES["transcript_required_claim"],
        clause=extraction.transcript_required,
        require_official_sources_instruction=(
            "Accept only official NCHSAA or official NC district/school policy pages. "
            "Verify that prior-school academic records/transcripts are required for eligibility verification."
        ),
        critical=True
    )

    # 4.b) Attendance Records Required
    await verify_clause_with_text_and_sources(
        evaluator,
        docs_node,
        id_prefix="Attendance_Records_Required",
        parent_desc="Identifies that the prior-school attendance records are required for verification.",
        stated_claim="The answer identifies that prior-school attendance records are required to verify athletic eligibility.",
        support_claim=EXPECTED_RULES["attendance_records_required_claim"],
        clause=extraction.attendance_records_required,
        require_official_sources_instruction=(
            "Accept only official NCHSAA or official NC district/school policy pages. "
            "Verify that prior-school attendance records are required for eligibility verification."
        ),
        critical=True
    )

    # Return structured evaluation summary
    return evaluator.get_summary()