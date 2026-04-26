import asyncio
import logging
from typing import Any, List, Optional, Dict

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys


TASK_ID = "big_ten_universities_multi_constraints_2024_2025"
TASK_DESCRIPTION = """
Identify three Big Ten Conference universities where each university satisfies all of the following criteria:
(1) The university is a current member of the Big Ten Conference as of the 2024-2025 academic year;
(2) The university's research expenditures reached or exceeded $900 million in fiscal year 2024 or fiscal year 2025;
(3) The university operates an undergraduate honors college or honors program that requires a minimum cumulative GPA of at least 3.3 for admission or continuation in the program;
(4) The university's football program participated in a bowl game during the 2024-2025 season;
(5) The university has an undergraduate enrollment of at least 30,000 students;
(6) The university participates in an interstate tuition reciprocity program such as the Midwest Student Exchange Program or a state-specific reciprocity agreement like the Minnesota-Wisconsin tuition reciprocity.

For each of the three universities identified, provide:
- the full official name of the university,
- the specific research expenditure amount for FY2024 or FY2025,
- the name of the honors college or honors program,
- the minimum GPA requirement for the honors program,
- the bowl game the university participated in during the 2024-2025 season,
- the undergraduate enrollment figure,
- and the specific tuition reciprocity program(s) in which the university participates.

Include reference URLs that verify each piece of information.
"""


# ------------------------------ Data Models ------------------------------ #
class UniversityItem(BaseModel):
    name: Optional[str] = None

    # Big Ten membership
    conference_sources: List[str] = Field(default_factory=list)

    # Research expenditure
    research_amount: Optional[str] = None
    research_fy: Optional[str] = None  # e.g., "FY2024" or "FY2025"
    research_sources: List[str] = Field(default_factory=list)

    # Honors program
    honors_name: Optional[str] = None
    honors_gpa: Optional[str] = None
    honors_sources: List[str] = Field(default_factory=list)

    # Football bowl participation
    bowl_name: Optional[str] = None
    bowl_sources: List[str] = Field(default_factory=list)

    # Undergraduate enrollment
    enrollment: Optional[str] = None
    enrollment_sources: List[str] = Field(default_factory=list)

    # Tuition reciprocity
    reciprocity_programs: List[str] = Field(default_factory=list)
    reciprocity_sources: List[str] = Field(default_factory=list)


class UniversitiesExtraction(BaseModel):
    universities: List[UniversityItem] = Field(default_factory=list)


# --------------------------- Extraction Prompt --------------------------- #
def prompt_extract_universities() -> str:
    return """
    Extract up to three universities listed in the answer that the answer claims meet the task requirements.
    For each university, strictly extract only what is explicitly provided in the answer, without adding or inferring new information.

    For each university, extract the following fields:
    - name: Full official university name (string).
    - conference_sources: URLs that the answer cites to verify current Big Ten membership (array of URLs).
    - research_amount: The specific research expenditure amount string for FY2024 or FY2025 (string as shown in the answer, e.g., "$1.05 billion" or "$950,000,000").
    - research_fy: The fiscal year associated with the research expenditure (e.g., "FY2024" or "FY2025"). If mentioned differently (e.g., "Fiscal Year 2024"), normalize to "FY2024". If not provided, return null.
    - research_sources: URLs cited to support the research expenditure figure (array of URLs).
    - honors_name: The name of the honors college or honors program (string).
    - honors_gpa: The minimum cumulative GPA requirement string for admission or continuation (string as shown, e.g., "3.5", "3.3 cumulative", "3.40 GPA").
    - honors_sources: URLs cited to support honors program info and GPA requirement (array of URLs).
    - bowl_name: The specific bowl game name participated in during the 2024-2025 season (string).
    - bowl_sources: URLs cited to verify bowl participation (array of URLs).
    - enrollment: The undergraduate enrollment figure string (string as shown in the answer).
    - enrollment_sources: URLs cited to verify the undergraduate enrollment (array of URLs).
    - reciprocity_programs: The list of tuition reciprocity program names the answer claims (e.g., "Midwest Student Exchange Program", "Minnesota–Wisconsin Reciprocity") (array of strings).
    - reciprocity_sources: URLs cited to verify reciprocity participation (array of URLs).

    Rules:
    - If any field is missing in the answer, set it to null (for singular fields) or an empty array (for list fields).
    - For URLs, extract actual URLs as they appear, including those inside markdown links.
    - Return exactly the first three universities mentioned if more than three are present.
    """


# ------------------------------ Helper Text ------------------------------ #
MISSING_URL_FAIL_INS = (
    "If no URL is provided for this verification or the provided URL is irrelevant or inaccessible, "
    "you must judge the claim as NOT SUPPORTED/INCORRECT."
)

RELIABLE_SPORTS_SITES = "ESPN.com, NCAA.com, CBSSports.com, FoxSports.com, Sports-Reference.com, official university athletics sites, or bigten.org."
RELIABLE_RESEARCH_SOURCES = "NSF HERD reports, official university institutional research/financial reports, or credible news citing these."
RELIABLE_ENROLLMENT_SOURCES = "official university institutional research/fact book pages, Common Data Set pages, or other official university pages."
RELIABLE_RECIPROCITY_SOURCES = "official reciprocity program websites (e.g., msep.mhec.org), state higher education agency pages, or official university tuition/financial aid pages."


def _fmt_list_str(items: List[str]) -> str:
    return ", ".join(items) if items else "none"


# --------------------------- Verification Logic -------------------------- #
async def verify_university(evaluator: Evaluator, parent_node, uni: UniversityItem, uni_idx: int) -> None:
    # University-level container (non-critical to allow partial credit across different universities)
    uni_node = evaluator.add_parallel(
        id=f"University_{uni_idx}",
        desc=f"{['First','Second','Third'][uni_idx - 1]} university identified meets all required criteria",
        parent=parent_node,
        critical=False,
    )

    # 1) University Name provided (critical existence check)
    evaluator.add_custom_node(
        result=bool(uni.name and uni.name.strip()),
        id=f"U{uni_idx}_University_Name",
        desc="Full official name of the university is provided",
        parent=uni_node,
        critical=True
    )

    # 2) Conference Membership (critical)
    conf_node = evaluator.add_parallel(
        id=f"U{uni_idx}_Conference_Membership",
        desc="University is a current member of the Big Ten Conference",
        parent=uni_node,
        critical=True
    )

    # 2.1 Conference status
    leaf_conf_status = evaluator.add_leaf(
        id=f"U{uni_idx}_Conference_Status",
        desc="University is verified as current Big Ten member as of 2024-2025",
        parent=conf_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"{uni.name} is a current member of the Big Ten Conference as of the 2024-2025 academic year.",
        node=leaf_conf_status,
        sources=uni.conference_sources,
        additional_instruction=(
            f"{MISSING_URL_FAIL_INS} Prefer official Big Ten or NCAA pages, or official university athletics pages. "
            "If any provided URL clearly lists Big Ten members and includes this university, mark as supported."
        )
    )

    # 2.2 Conference URL quality
    leaf_conf_url = evaluator.add_leaf(
        id=f"U{uni_idx}_Conference_URL",
        desc="Conference membership verified through official Big Ten Conference source or reliable reference",
        parent=conf_node,
        critical=True
    )
    await evaluator.verify(
        claim=(
            f"At least one provided URL is either an official Big Ten source (bigten.org) or a widely recognized and "
            f"reliable sports reference site (e.g., {RELIABLE_SPORTS_SITES}) explicitly confirming that {uni.name} is a Big Ten member."
        ),
        node=leaf_conf_url,
        sources=uni.conference_sources,
        additional_instruction=MISSING_URL_FAIL_INS
    )

    # 3) Research Expenditure >= $900M in FY2024 or FY2025 (critical)
    res_node = evaluator.add_parallel(
        id=f"U{uni_idx}_Research_Expenditure",
        desc="University's research expenditures meet or exceed $900 million annually",
        parent=uni_node,
        critical=True
    )

    # 3.1 Specific research amount and threshold
    leaf_res_amount = evaluator.add_leaf(
        id=f"U{uni_idx}_Research_Amount",
        desc="Specific research expenditure amount for FY2024 or FY2025 is provided and is at least $900 million",
        parent=res_node,
        critical=True
    )
    fy_str = uni.research_fy or "FY2024 or FY2025"
    amt_str = uni.research_amount or "(amount not provided)"
    await evaluator.verify(
        claim=(
            f"For {fy_str}, the research expenditures for {uni.name} are reported as {amt_str}, "
            "and this value is at least $900 million (allowing reasonable rounding). "
            "If the provided year string is ambiguous but the page clearly shows FY2024 or FY2025 >= $900M, consider it supported."
        ),
        node=leaf_res_amount,
        sources=uni.research_sources,
        additional_instruction=(
            f"{MISSING_URL_FAIL_INS} Prefer {RELIABLE_RESEARCH_SOURCES} "
            "Rounding or minor formatting differences are acceptable."
        )
    )

    # 3.2 Research URL quality
    leaf_res_url = evaluator.add_leaf(
        id=f"U{uni_idx}_Research_URL",
        desc="Research expenditure data verified through NSF HERD survey, university official announcement, or credible news source",
        parent=res_node,
        critical=True
    )
    await evaluator.verify(
        claim=(
            f"At least one provided URL is a credible authority (e.g., {RELIABLE_RESEARCH_SOURCES}) confirming "
            f"{uni.name}'s research expenditures at or above $900 million in FY2024 or FY2025."
        ),
        node=leaf_res_url,
        sources=uni.research_sources,
        additional_instruction=MISSING_URL_FAIL_INS
    )

    # 4) Honors College / Program with GPA >= 3.3 (critical)
    honors_node = evaluator.add_parallel(
        id=f"U{uni_idx}_Honors_College",
        desc="University has an undergraduate honors college or honors program with appropriate GPA requirements",
        parent=uni_node,
        critical=True
    )

    # 4.1 Honors name provided
    evaluator.add_custom_node(
        result=bool(uni.honors_name and uni.honors_name.strip()),
        id=f"U{uni_idx}_Honors_Name",
        desc="Name of the honors college or honors program is provided",
        parent=honors_node,
        critical=True
    )

    # 4.2 Honors existence verification
    leaf_honors_exist = evaluator.add_leaf(
        id=f"U{uni_idx}_Honors_Existence",
        desc="Formal honors college or program exists for undergraduate students",
        parent=honors_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"{uni.name} operates an official undergraduate honors college or honors program named '{uni.honors_name}'.",
        node=leaf_honors_exist,
        sources=uni.honors_sources,
        additional_instruction=(
            f"{MISSING_URL_FAIL_INS} Prefer official university pages (.edu), honors program pages, or official PDFs."
        )
    )

    # 4.3 Honors GPA threshold (>= 3.3)
    leaf_honors_gpa = evaluator.add_leaf(
        id=f"U{uni_idx}_Honors_GPA",
        desc="Honors program minimum GPA requirement is provided and is at least 3.3 for admission or continuation",
        parent=honors_node,
        critical=True
    )
    gpa_str = uni.honors_gpa or "(GPA not provided)"
    await evaluator.verify(
        claim=(
            f"The honors program at {uni.name} requires a minimum cumulative GPA of at least 3.3 for admission or continuation. "
            f"The answer states the requirement as '{gpa_str}'. Either admission or continuation threshold counts as long as it is >= 3.3."
        ),
        node=leaf_honors_gpa,
        sources=uni.honors_sources,
        additional_instruction=(
            f"{MISSING_URL_FAIL_INS} Accept clear statements such as minimum cumulative GPA for admission or for remaining in good standing. "
            "If multiple thresholds exist, any explicit minimum >= 3.3 satisfies the requirement."
        )
    )

    # 4.4 Honors URL quality
    leaf_honors_url = evaluator.add_leaf(
        id=f"U{uni_idx}_Honors_URL",
        desc="Honors college information and GPA requirements verified through official university source",
        parent=honors_node,
        critical=True
    )
    await evaluator.verify(
        claim=(
            f"At least one provided URL is an official {uni.name} website (.edu or equivalent official subdomain) explicitly stating the honors program and its GPA requirement."
        ),
        node=leaf_honors_url,
        sources=uni.honors_sources,
        additional_instruction=MISSING_URL_FAIL_INS
    )

    # 5) Football Bowl Participation in 2024-2025 (critical)
    bowl_node = evaluator.add_parallel(
        id=f"U{uni_idx}_Bowl_Participation",
        desc="University's football program participated in a bowl game during the 2024-2025 season",
        parent=uni_node,
        critical=True
    )

    # 5.1 Bowl name provided
    evaluator.add_custom_node(
        result=bool(uni.bowl_name and uni.bowl_name.strip()),
        id=f"U{uni_idx}_Bowl_Name",
        desc="Specific bowl game name is provided",
        parent=bowl_node,
        critical=True
    )

    # 5.2 Bowl participation verified
    leaf_bowl_ver = evaluator.add_leaf(
        id=f"U{uni_idx}_Bowl_Verification",
        desc="Bowl game participation during 2024-2025 season is verified",
        parent=bowl_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"{uni.name}'s football team participated in the {uni.bowl_name} during the 2024-2025 season.",
        node=leaf_bowl_ver,
        sources=uni.bowl_sources,
        additional_instruction=(
            f"{MISSING_URL_FAIL_INS} Accept official bowl websites, {RELIABLE_SPORTS_SITES}, or Big Ten/official athletics sites. "
            "The bowl may be played in late Dec 2024 or early Jan 2025; it still counts as the 2024-2025 season."
        )
    )

    # 5.3 Bowl URL quality
    leaf_bowl_url = evaluator.add_leaf(
        id=f"U{uni_idx}_Bowl_URL",
        desc="Bowl game participation verified through official conference, bowl game, or sports media source",
        parent=bowl_node,
        critical=True
    )
    await evaluator.verify(
        claim=(
            f"At least one provided URL is an official conference or bowl site, or a nationally recognized sports media outlet (e.g., {RELIABLE_SPORTS_SITES}) confirming {uni.name}'s participation in the {uni.bowl_name}."
        ),
        node=leaf_bowl_url,
        sources=uni.bowl_sources,
        additional_instruction=MISSING_URL_FAIL_INS
    )

    # 6) Undergraduate enrollment >= 30,000 (critical)
    enr_node = evaluator.add_parallel(
        id=f"U{uni_idx}_Enrollment",
        desc="University has undergraduate enrollment of at least 30,000 students",
        parent=uni_node,
        critical=True
    )

    # 6.1 Specific enrollment figure and threshold
    leaf_enr_fig = evaluator.add_leaf(
        id=f"U{uni_idx}_Enrollment_Figure",
        desc="Specific undergraduate enrollment figure is provided and is at least 30,000",
        parent=enr_node,
        critical=True
    )
    enr_str = uni.enrollment or "(enrollment not provided)"
    await evaluator.verify(
        claim=(
            f"The undergraduate enrollment for {uni.name} is reported as {enr_str}, and it is at least 30,000 students. "
            "If the page provides clearly labeled undergraduate enrollment >= 30,000 (allowing reasonable rounding), the claim is supported. "
            "If only total/university-wide enrollment (not undergraduate) is available without a clear undergraduate figure >= 30,000, consider it not supported."
        ),
        node=leaf_enr_fig,
        sources=uni.enrollment_sources,
        additional_instruction=(
            f"{MISSING_URL_FAIL_INS} Prefer {RELIABLE_ENROLLMENT_SOURCES} "
            "Use undergraduate figure specifically; do not substitute graduate or total if undergraduate is not clearly indicated."
        )
    )

    # 6.2 Enrollment URL quality
    leaf_enr_url = evaluator.add_leaf(
        id=f"U{uni_idx}_Enrollment_URL",
        desc="Enrollment data verified through official university institutional data or reliable source",
        parent=enr_node,
        critical=True
    )
    await evaluator.verify(
        claim=(
            f"At least one provided URL is an official institutional data source for {uni.name} (e.g., fact book, Common Data Set, or official stats page) confirming undergraduate enrollment."
        ),
        node=leaf_enr_url,
        sources=uni.enrollment_sources,
        additional_instruction=MISSING_URL_FAIL_INS
    )

    # 7) Tuition reciprocity participation (critical)
    rec_node = evaluator.add_parallel(
        id=f"U{uni_idx}_Tuition_Reciprocity",
        desc="University participates in an interstate tuition reciprocity program",
        parent=uni_node,
        critical=True
    )

    # 7.1 Program name(s) provided
    evaluator.add_custom_node(
        result=bool(uni.reciprocity_programs and len(uni.reciprocity_programs) > 0),
        id=f"U{uni_idx}_Reciprocity_Program",
        desc="Specific tuition reciprocity program name(s) provided (MSEP or state-specific agreement such as Minnesota-Wisconsin)",
        parent=rec_node,
        critical=True
    )

    # 7.2 Participation verified
    leaf_rec_ver = evaluator.add_leaf(
        id=f"U{uni_idx}_Reciprocity_Verification",
        desc="University participation in stated reciprocity program is verified",
        parent=rec_node,
        critical=True
    )
    await evaluator.verify(
        claim=(
            f"{uni.name} participates in the following tuition reciprocity program(s): {_fmt_list_str(uni.reciprocity_programs)}."
        ),
        node=leaf_rec_ver,
        sources=uni.reciprocity_sources,
        additional_instruction=(
            f"{MISSING_URL_FAIL_INS} Prefer {RELIABLE_RECIPROCITY_SOURCES} "
            "Explicit statements of eligibility or participation count as verification."
        )
    )

    # 7.3 Reciprocity URL quality
    leaf_rec_url = evaluator.add_leaf(
        id=f"U{uni_idx}_Reciprocity_URL",
        desc="Reciprocity participation verified through official program website, state higher education agency, or university source",
        parent=rec_node,
        critical=True
    )
    await evaluator.verify(
        claim=(
            f"At least one provided URL is an official reciprocity program website, a state higher education agency page, or an official {uni.name} page confirming reciprocity participation."
        ),
        node=leaf_rec_url,
        sources=uni.reciprocity_sources,
        additional_instruction=MISSING_URL_FAIL_INS
    )


# -------------------------- Main Evaluation Flow ------------------------- #
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

    # Extract up to 3 universities as structured data
    extracted = await evaluator.extract(
        prompt=prompt_extract_universities(),
        template_class=UniversitiesExtraction,
        extraction_name="universities_extraction",
    )

    # Normalize to exactly three entries (pad with empty if fewer)
    universities: List[UniversityItem] = (extracted.universities or [])[:3]
    while len(universities) < 3:
        universities.append(UniversityItem())

    # Task_Root as non-critical parallel aggregator (allow partial credit across universities)
    task_root = evaluator.add_parallel(
        id="Task_Root",
        desc="Identify exactly three Big Ten Conference universities that simultaneously meet all specified criteria regarding research expenditures, honors college requirements, bowl game participation, enrollment size, and tuition reciprocity program participation",
        parent=root,
        critical=False
    )

    # Build verification subtrees for each university (1..3)
    for idx, uni in enumerate(universities, start=1):
        await verify_university(evaluator, task_root, uni, idx)

    return evaluator.get_summary()