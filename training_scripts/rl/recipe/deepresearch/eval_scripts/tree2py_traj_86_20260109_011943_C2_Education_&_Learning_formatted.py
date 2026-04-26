import asyncio
import logging
import re
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from obj_task_eval.llm_client.base_client import LLMClient
from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


TASK_ID = "il_pd_course_selection"
TASK_DESCRIPTION = (
    "An Illinois public school teacher with a Professional Educator License needs to earn professional development "
    "hours toward their 120-hour, five-year license renewal requirement. They want to take a graduate-level course "
    "focused on educational technology integration or STEM education that can be completed online. Identify one specific "
    "course or program from an Illinois-approved provider that meets these requirements. Provide the course/program name, "
    "the institution offering it, how many professional development hours it provides, and a reference URL."
)


# ----------------------------- Extraction Models ----------------------------- #
class CourseInfo(BaseModel):
    course_name: Optional[str] = None
    institution_name: Optional[str] = None
    reference_url: Optional[str] = None

    # PD hours and basis
    pd_hours_value: Optional[str] = None  # numeric string preferred; may contain units
    pd_hours_basis: Optional[str] = None  # one of: "direct_pd_hours", "semester_hours", "unknown"
    credits_value: Optional[str] = None   # numeric string for semester hours when basis == "semester_hours"

    # Eligibility descriptors derived from answer (will be verified via sources)
    graduate_level_indicator: Optional[str] = None  # "yes"/"no"/"unknown"
    online_indicator: Optional[str] = None          # "yes"/"no"/"unknown"
    topic_focus_label: Optional[str] = None         # "edtech_integration"/"stem_education"/"other"

    institution_state: Optional[str] = None         # e.g., "Illinois", "IL"
    additional_urls: List[str] = Field(default_factory=list)

    # Helpful meta for checks
    course_candidates: List[str] = Field(default_factory=list)   # all distinct course/program names mentioned
    mentions_120_hour_context: Optional[bool] = None             # whether the answer mentions 120-hour renewal context


# ----------------------------- Extraction Prompt ----------------------------- #
def prompt_extract_course_info() -> str:
    return """
    Extract details for the single specific course or program the answer recommends for Illinois teachers seeking PD hours.

    Return a JSON object with the following fields:
    - course_name: The exact name/title of the identified course or program (string).
    - institution_name: The name of the institution offering it (string).
    - reference_url: The main official page URL for this course/program (string URL). If multiple are given, choose the most specific official course/program page. If none, return null.
    - pd_hours_value: A numeric PD hours value provided by the answer (string containing just the number; e.g., "45"). If the answer only provides semester hours/credits, leave this as null.
    - pd_hours_basis: One of "direct_pd_hours", "semester_hours", or "unknown". Use "semester_hours" when the answer derives PD hours from college/semester credits. Use "direct_pd_hours" when the answer directly states PD hours. Use "unknown" if unclear.
    - credits_value: If pd_hours_basis == "semester_hours", provide the number of semester credit hours as a string (e.g., "3"). Otherwise null.
    - graduate_level_indicator: "yes", "no", or "unknown" based on the answer's claim (e.g., graduate credit, master's-level).
    - online_indicator: "yes", "no", or "unknown" based on whether the course/program can be completed online according to the answer.
    - topic_focus_label: Use "edtech_integration" if the focus is educational/instructional technology integration, "stem_education" if it is STEM education, otherwise "other".
    - institution_state: If the answer indicates the institution is in Illinois, return "Illinois" or "IL"; otherwise null or another state string.
    - additional_urls: Array of any other URLs the answer cites about the institution or program (e.g., accreditation page, about page). If none, return [].
    - course_candidates: Array of all distinct named course/program titles the answer mentions. Include the primary one first.
    - mentions_120_hour_context: true/false depending on whether the answer explicitly mentions contributing to the 120-hour, five-year Illinois renewal requirement.

    If any field is missing in the answer text, use null for scalars and [] for arrays.
    """


# ----------------------------- Helper Utilities ----------------------------- #
def parse_first_number(text: Optional[str]) -> Optional[float]:
    if not text:
        return None
    matches = re.findall(r"-?\d+(?:\.\d+)?", text)
    if not matches:
        return None
    try:
        return float(matches[0])
    except Exception:
        return None


def join_sources(primary: Optional[str], extras: List[str]) -> List[str]:
    srcs = []
    if primary and isinstance(primary, str) and primary.strip():
        srcs.append(primary.strip())
    for u in extras or []:
        if isinstance(u, str) and u.strip():
            srcs.append(u.strip())
    return srcs


# ----------------------------- Verification Builders ----------------------------- #
async def build_course_program_eligibility(
    evaluator: Evaluator,
    parent_node,
    info: CourseInfo,
) -> None:
    """
    Build and verify the 'Course_Program_Eligibility' node and its children.
    """
    eligibility_node = evaluator.add_parallel(
        id="Course_Program_Eligibility",
        desc="Selected course/program meets all eligibility constraints from the prompt/constraints.",
        parent=parent_node,
        critical=True
    )

    # Sources for verification across multiple pages
    sources = join_sources(info.reference_url, info.additional_urls)

    # 1) Exactly one specific course/program
    exactly_one = False
    if info.course_name and info.course_name.strip():
        # Count distinct course/program mentions
        unique_candidates = {c.strip().lower() for c in (info.course_candidates or []) if c and c.strip()}
        if not unique_candidates:
            exactly_one = True
        else:
            # If candidates list contains exactly one distinct title that matches course_name, still acceptable
            if len(unique_candidates) == 1:
                # accept
                exactly_one = True
            else:
                exactly_one = False
    one_specific_node = evaluator.add_custom_node(
        result=exactly_one,
        id="One_Specific_Course_Or_Program",
        desc="Identifies exactly one specific course or program (not multiple options).",
        parent=eligibility_node,
        critical=True
    )

    # 2) Graduate level
    graduate_leaf = evaluator.add_leaf(
        id="Graduate_Level",
        desc="Course/program is graduate-level (explicitly stated or clearly designated by the provider).",
        parent=eligibility_node,
        critical=True
    )
    grad_claim = (
        f"The course or program '{info.course_name or 'the identified course/program'}' offered by "
        f"{info.institution_name or 'the institution'} is graduate-level or provides graduate credit."
    )
    await evaluator.verify(
        claim=grad_claim,
        node=graduate_leaf,
        sources=sources,
        additional_instruction=(
            "Accept evidence such as phrases like 'graduate credit', 'master's-level', '500-level course', 'M.Ed.' "
            "or 'graduate program'. If only undergraduate-level credit is indicated, consider it not graduate-level."
        ),
    )

    # 3) Topic focus: educational technology integration or STEM education
    topic_leaf = evaluator.add_leaf(
        id="Topic_Focus_Matches",
        desc="Course/program focus is educational technology integration or STEM education.",
        parent=eligibility_node,
        critical=True
    )
    topic_claim = (
        f"The course/program focuses on either educational technology/instructional technology integration "
        f"or on STEM education."
    )
    await evaluator.verify(
        claim=topic_claim,
        node=topic_leaf,
        sources=sources,
        additional_instruction=(
            "Accept synonyms like 'instructional technology', 'technology integration', 'learning technologies' for edtech. "
            "Accept STEM-related wording: science, technology, engineering, math, or integrated STEM education."
        ),
    )

    # 4) Online availability
    online_leaf = evaluator.add_leaf(
        id="Online_Availability",
        desc="Course/program can be completed online.",
        parent=eligibility_node,
        critical=True
    )
    online_claim = (
        f"The course/program '{info.course_name or 'the identified course/program'}' can be completed online "
        f"(e.g., 'online', 'fully online', '100% online', 'distance learning')."
    )
    await evaluator.verify(
        claim=online_claim,
        node=online_leaf,
        sources=sources,
        additional_instruction=(
            "Look for explicit indications like 'online', 'fully online', or similar. If only 'hybrid' is indicated "
            "and there is no clear statement it can be completed fully online, consider not supported."
        ),
    )

    # 5) Regionally accredited institution in Illinois
    # Split into two leaf verifications under a parent to avoid combining multiple checks in one leaf.
    region_il_parent = evaluator.add_parallel(
        id="Regionally_Accredited_Illinois_Institution",
        desc="The provider is a regionally accredited institution located in Illinois (as required by the constraints).",
        parent=eligibility_node,
        critical=True
    )

    # 5a) Institution Located in Illinois
    il_loc_leaf = evaluator.add_leaf(
        id="Institution_Location_Illinois",
        desc="The institution is located in Illinois.",
        parent=region_il_parent,
        critical=True
    )
    il_loc_claim = (
        f"The institution '{info.institution_name or 'the institution'}' is located in Illinois."
    )
    await evaluator.verify(
        claim=il_loc_claim,
        node=il_loc_leaf,
        sources=sources,
        additional_instruction=(
            "Accept evidence such as campus address in Illinois, city/state listing in Illinois (e.g., Chicago, IL), "
            "or explicit statements indicating the institution is an Illinois university/college."
        ),
    )

    # 5b) Institution Regionally Accredited
    accred_leaf = evaluator.add_leaf(
        id="Institution_Regionally_Accredited",
        desc="The institution is regionally accredited.",
        parent=region_il_parent,
        critical=True
    )
    accred_claim = (
        f"The institution '{info.institution_name or 'the institution'}' is regionally accredited (e.g., by the Higher Learning Commission)."
    )
    await evaluator.verify(
        claim=accred_claim,
        node=accred_leaf,
        sources=sources,
        additional_instruction=(
            "Look for accreditation statements such as 'accredited by the Higher Learning Commission (HLC)', "
            "'regional accreditation', or other recognized U.S. regional accreditors (MSCHE, NECHE, NWCCU, SACSCOC, WSCUC). "
            "If only programmatic accreditation is mentioned and no regional institutional accreditation, consider not supported."
        ),
    )


async def build_required_output_fields(
    evaluator: Evaluator,
    parent_node,
    info: CourseInfo,
) -> None:
    """
    Build and verify the 'Required_Output_Fields' node and its children.
    Note: Parent is set to non-critical to allow the non-critical 'Mentions_120_Hour_Context' leaf.
    Critical children under this node will still gate its own score to 0 if any fail.
    """
    outputs_node = evaluator.add_parallel(
        id="Required_Output_Fields",
        desc="All required fields are provided for the identified course/program, with correct PD-hour handling.",
        parent=parent_node,
        critical=False
    )

    # Presence checks for name and institution
    name_present = bool(info.course_name and info.course_name.strip())
    inst_present = bool(info.institution_name and info.institution_name.strip())

    evaluator.add_custom_node(
        result=name_present,
        id="Provides_Course_Program_Name",
        desc="Provides the course/program name.",
        parent=outputs_node,
        critical=True
    )

    evaluator.add_custom_node(
        result=inst_present,
        id="Provides_Institution_Name",
        desc="Provides the institution offering the course/program.",
        parent=outputs_node,
        critical=True
    )

    # PD hours numeric value presence
    pd_numeric = parse_first_number(info.pd_hours_value)
    evaluator.add_custom_node(
        result=(pd_numeric is not None),
        id="Provides_PD_Hours_Value",
        desc="Provides a numeric professional development hours (PD hours) value for the course/program.",
        parent=outputs_node,
        critical=True
    )

    # PD hours positive
    evaluator.add_custom_node(
        result=(pd_numeric is not None and pd_numeric > 0),
        id="PD_Hours_Positive_Meaningful_Progress",
        desc="The provided PD hours value is greater than 0, indicating non-zero progress toward the 120-hour renewal requirement.",
        parent=outputs_node,
        critical=True
    )

    # PD Hours Basis Correct (verification leaf)
    pd_basis_leaf = evaluator.add_leaf(
        id="PD_Hours_Basis_Correct",
        desc="If PD hours are derived from semester hours/college credit, applies the required conversion (1 semester hour = 15 PD hours) correctly; otherwise cites PD hours directly from the source.",
        parent=outputs_node,
        critical=True
    )

    sources = join_sources(info.reference_url, info.additional_urls)
    credits = parse_first_number(info.credits_value)
    # Construct claim depending on basis
    if (info.pd_hours_basis or "").lower() == "semester_hours" and (credits is not None) and (pd_numeric is not None):
        expected_pd = credits * 15.0
        # Allow small rounding differences
        pd_basis_claim = (
            f"The official page(s) indicate the course/program carries {credits} semester credit hours. "
            f"Using the Illinois PD conversion (1 semester hour = 15 PD hours), this yields approximately {expected_pd:.0f} PD hours, "
            f"which matches the provided PD hours value ({int(pd_numeric)})."
        )
        add_ins = (
            "Verify that the webpage(s) explicitly state the number of semester credit hours. "
            "Then apply the Illinois PD conversion: 1 semester hour equals 15 PD hours. "
            "Allow minor rounding differences (e.g., 44.9 ~ 45)."
        )
    elif (info.pd_hours_basis or "").lower() == "direct_pd_hours" and (pd_numeric is not None):
        pd_basis_claim = (
            f"The official page(s) explicitly state that the course/program provides {int(pd_numeric)} professional development hours (PD hours), "
            f"and this matches the provided PD hours value."
        )
        add_ins = (
            "Confirm that the PD hours are stated or clearly implied on the official page(s). "
            "If the PD hours are not stated but only credits are mentioned, this claim should be considered not supported."
        )
    else:
        # Unknown basis or insufficient data; ask verifier to judge availability on page(s)
        pd_basis_claim = (
            "There is sufficient evidence on the official page(s) to confirm the provided PD hours value and that any required conversion "
            "(1 semester hour = 15 PD hours) has been correctly applied."
        )
        add_ins = (
            "If the webpage(s) do not explicitly state PD hours or semester credit hours, or if conversion cannot be confirmed, "
            "consider the claim not supported."
        )

    await evaluator.verify(
        claim=pd_basis_claim,
        node=pd_basis_leaf,
        sources=sources,
        additional_instruction=add_ins,
    )

    # Reference URL presence
    url_present = bool(info.reference_url and info.reference_url.strip())
    evaluator.add_custom_node(
        result=url_present,
        id="Provides_Reference_URL",
        desc="Provides a reference URL to an official source page for the identified course/program.",
        parent=outputs_node,
        critical=True
    )

    # Mentions 120-hour context (non-critical)
    mention_leaf = evaluator.add_leaf(
        id="Mentions_120_Hour_Context",
        desc="Mentions that the PD hours/credits earned contribute toward the stated 120-hour, five-year Illinois renewal requirement (no additional calculations required).",
        parent=outputs_node,
        critical=False
    )
    # Verify against the answer text itself (simple verify without sources)
    mention_claim = (
        "The answer mentions that the PD hours or credits earned contribute toward the 120-hour, five-year Illinois renewal requirement."
    )
    await evaluator.verify(
        claim=mention_claim,
        node=mention_leaf,
        additional_instruction=(
            "Check the provided answer text for an explicit mention of contributing to the 120-hour five-year renewal requirement. "
            "Paraphrases are acceptable as long as the intent is clear."
        ),
    )


# ----------------------------- Main Evaluation Entry ----------------------------- #
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
    Evaluate an answer for the Illinois PD course/program selection task.
    """
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.SEQUENTIAL,
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

    # Extract structured course/program info from the answer
    info: CourseInfo = await evaluator.extract(
        prompt=prompt_extract_course_info(),
        template_class=CourseInfo,
        extraction_name="course_info",
    )

    # Build and run verifications according to rubric tree
    # Root-level child representing the overall solution (critical sequential aggregation)
    solution_node = evaluator.add_sequential(
        id="PD_Course_Solution",
        desc="Identify exactly one Illinois-appropriate online graduate-level course/program (edtech integration or STEM) and provide required details and a source URL.",
        parent=root,
        critical=True
    )

    await build_course_program_eligibility(evaluator, solution_node, info)
    await build_required_output_fields(evaluator, solution_node, info)

    # Return summary dict
    return evaluator.get_summary()