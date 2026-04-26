import asyncio
import logging
from typing import Any, List, Optional, Dict

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.llm_client.base_client import LLMClient


TASK_ID = "student_services_coordinator_unis"
TASK_DESCRIPTION = (
    "Identify four public universities in the United States that meet all of the following criteria for employment "
    "as a student services coordinator: (1) Be a public, state-funded institution (not private); "
    "(2) Have a Student Services, Student Affairs, or equivalent department; "
    "(3) Have a dedicated Career Services or Career Center department; "
    "(4) Employ coordinator-level administrative staff in student services or related areas; "
    "(5) Require a minimum of a bachelor's degree for coordinator-level positions; "
    "(6) Typically require 2-5 years of professional experience for coordinator positions; "
    "(7) Have a student enrollment of at least 5,000 students; "
    "(8) Provide standard employee benefits including health insurance to full-time staff; "
    "(9) Have a publicly accessible careers or employment website where job postings can be viewed; "
    "(10) Be accredited by a regional accrediting body recognized by the US Department of Education. "
    "For each university, provide its official name and reference URLs to verify the information."
)


# ------------------------- Data Models ------------------------- #
class UniversityItem(BaseModel):
    name: Optional[str] = None
    reference_urls: List[str] = Field(default_factory=list)


class UniversitiesExtraction(BaseModel):
    universities: List[UniversityItem] = Field(default_factory=list)


# ------------------------- Extraction Prompt ------------------------- #
def prompt_extract_universities() -> str:
    return (
        "From the provided answer, extract up to all universities mentioned that are proposed as meeting the student "
        "services coordinator employment criteria. For each university, extract:\n"
        "1) name: The official university name exactly as written in the answer. If multiple variants are given, "
        "   prefer the official full name.\n"
        "2) reference_urls: A list of all URLs cited in the answer that are intended to support verification of the "
        "   criteria for this specific university (e.g., university website pages, student affairs pages, career services, "
        "   HR/benefits pages, job postings, accreditation pages, or credible sources like Wikipedia). Extract only URLs "
        "   explicitly present in the answer text.\n\n"
        "Return a JSON object with a 'universities' array of objects having 'name' and 'reference_urls'. "
        "If the answer lists more than four universities, still include them all; downstream evaluation will select the first four. "
        "If an item lacks a name, set 'name' to null; if no URLs were provided for that item, return an empty list for 'reference_urls'."
    )


# ------------------------- Helper Functions ------------------------- #
def _name_or_placeholder(i: int, uni: UniversityItem) -> str:
    return uni.name.strip() if uni and uni.name else f"University #{i + 1}"


def _has_nonempty_name(uni: UniversityItem) -> bool:
    return bool(uni and uni.name and uni.name.strip())


def _has_urls(uni: UniversityItem) -> bool:
    return bool(uni and isinstance(uni.reference_urls, list) and len(uni.reference_urls) > 0)


# ------------------------- Verification per University ------------------------- #
async def verify_university(
    evaluator: Evaluator,
    parent_node,
    uni: UniversityItem,
    idx: int,
) -> Dict[str, Any]:
    """
    Build the sequential verification subtree for a single university.
    Returns a dictionary with references to prerequisite nodes for potential cross-checks.
    """
    uni_name = _name_or_placeholder(idx, uni)

    uni_node = evaluator.add_sequential(
        id=f"university_{idx}",
        desc=f"University #{idx + 1} verification: {uni_name}",
        parent=parent_node,
        critical=False  # Allow partial credit per university
    )

    # Existence: name provided
    name_provided_node = evaluator.add_custom_node(
        result=_has_nonempty_name(uni),
        id=f"university_{idx}_name_provided",
        desc=f"University #{idx + 1}: A qualifying university is identified and named",
        parent=uni_node,
        critical=True  # Gate further checks
    )

    # Existence: reference URLs provided
    urls_provided_node = evaluator.add_custom_node(
        result=_has_urls(uni),
        id=f"university_{idx}_urls_provided",
        desc=f"University #{idx + 1}: Reference URLs for verification are provided",
        parent=uni_node,
        critical=True  # Gate further checks
    )

    # Constraints group (parallel, all checks within are critical)
    constraints_node = evaluator.add_parallel(
        id=f"university_{idx}_constraints",
        desc=f"University #{idx + 1}: Constraint verifications",
        parent=uni_node,
        critical=True  # All constraints are mandatory
    )

    # 1) University (awards higher education degrees)
    leaf_is_university = evaluator.add_leaf(
        id=f"university_{idx}_is_university",
        desc=f"{uni_name} is a higher education university (not a community college or non-degree institution)",
        parent=constraints_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The institution '{uni_name}' is a university that confers higher education degrees.",
        node=leaf_is_university,
        sources=uni.reference_urls,
        additional_instruction=(
            "Accept pages that clearly indicate the institution is a 'University' and awards bachelor's, master's, or doctoral degrees. "
            "Do not count community colleges or non-degree institutions."
        ),
    )

    # 2) Public institution
    leaf_is_public = evaluator.add_leaf(
        id=f"university_{idx}_is_public",
        desc=f"{uni_name} is a public, state-funded institution (not private)",
        parent=constraints_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"'{uni_name}' is a public, state-funded university (not private).",
        node=leaf_is_public,
        sources=uni.reference_urls,
        additional_instruction=(
            "Look for explicit language such as 'public university', 'state university', or equivalent statements."
        ),
    )

    # 3) United States-based
    leaf_us_based = evaluator.add_leaf(
        id=f"university_{idx}_us_based",
        desc=f"{uni_name} has its main campus located in the United States",
        parent=constraints_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The main campus of '{uni_name}' is located in the United States.",
        node=leaf_us_based,
        sources=uni.reference_urls,
        additional_instruction=(
            "Confirm that the institution is in the U.S. (city/state or explicit 'United States')."
        ),
    )

    # 4) Student Services / Student Affairs department
    leaf_student_services = evaluator.add_leaf(
        id=f"university_{idx}_student_services_dept",
        desc=f"{uni_name} has a Student Services, Student Affairs, or equivalent department",
        parent=constraints_node,
        critical=True
    )
    await evaluator.verify(
        claim=(
            f"'{uni_name}' has a Student Affairs, Student Services, or an equivalent division/department supporting students."
        ),
        node=leaf_student_services,
        sources=uni.reference_urls,
        additional_instruction=(
            "Accept department names like 'Student Affairs', 'Student Services', 'Division of Student Affairs', "
            "'Student Experience', or other clearly equivalent student-support administrative units."
        ),
    )

    # 5) Career Services / Career Center department
    leaf_career_services = evaluator.add_leaf(
        id=f"university_{idx}_career_services_dept",
        desc=f"{uni_name} has a dedicated Career Services or Career Center department",
        parent=constraints_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"'{uni_name}' has a dedicated Career Services or Career Center department.",
        node=leaf_career_services,
        sources=uni.reference_urls,
        additional_instruction=(
            "Look for 'Career Services', 'Career Center', 'Career Development', or equivalent offices supporting students' careers."
        ),
    )

    # 6) Employ coordinator-level staff
    leaf_employ_coordinators = evaluator.add_leaf(
        id=f"university_{idx}_employs_coordinators",
        desc=f"{uni_name} employs staff in coordinator-level administrative positions within student services or related areas",
        parent=constraints_node,
        critical=True
    )
    await evaluator.verify(
        claim=(
            f"'{uni_name}' employs 'Coordinator' level administrative staff within student services or closely related areas "
            "(e.g., student life, advising, career services, residence life)."
        ),
        node=leaf_employ_coordinators,
        sources=uni.reference_urls,
        additional_instruction=(
            "Job postings, HR pages, or org charts indicating roles titled 'Coordinator' in student services or related areas are acceptable."
        ),
    )

    # 7) Bachelor's degree required for coordinator positions
    leaf_bachelors_required = evaluator.add_leaf(
        id=f"university_{idx}_bachelors_required",
        desc=f"{uni_name} requires a minimum of a bachelor's degree for coordinator-level positions",
        parent=constraints_node,
        critical=True
    )
    await evaluator.verify(
        claim=(
            f"Coordinator-level positions at '{uni_name}' require at least a bachelor's degree."
        ),
        node=leaf_bachelors_required,
        sources=uni.reference_urls,
        additional_instruction=(
            "Look for minimum qualifications on coordinator job postings or classification specs stating bachelor's degree is required. "
            "Equivalent (e.g., 'bachelor's required or equivalent experience') can count if clearly requiring at least bachelor's in typical cases."
        ),
    )

    # 8) 2–5 years professional experience typical for coordinator positions
    leaf_experience_required = evaluator.add_leaf(
        id=f"university_{idx}_experience_2to5",
        desc=f"{uni_name} typically requires 2–5 years of professional experience for coordinator positions",
        parent=constraints_node,
        critical=True
    )
    await evaluator.verify(
        claim=(
            f"Coordinator-level positions at '{uni_name}' typically require between 2 and 5 years of professional experience."
        ),
        node=leaf_experience_required,
        sources=uni.reference_urls,
        additional_instruction=(
            "Accept statements such as '2 years required', '3–5 years preferred/required', or equivalent ranges "
            "within 2–5 years on coordinator job postings or HR specs."
        ),
    )

    # 9) Enrollment >= 5,000
    leaf_min_enrollment = evaluator.add_leaf(
        id=f"university_{idx}_enrollment_min_5000",
        desc=f"{uni_name} has an enrollment of at least 5,000 students",
        parent=constraints_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"'{uni_name}' has total student enrollment of at least 5,000.",
        node=leaf_min_enrollment,
        sources=uni.reference_urls,
        additional_instruction=(
            "Use institutional facts pages, credible sources (e.g., Wikipedia infobox), or reports. "
            "Allow rounding and recent-year approximations (e.g., 4,950 ≈ 5,000 if explicitly described as ~5k)."
        ),
    )

    # 10) Offer full-time staff positions
    leaf_full_time_positions = evaluator.add_leaf(
        id=f"university_{idx}_full_time_positions",
        desc=f"{uni_name} offers full-time staff employment opportunities",
        parent=constraints_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"'{uni_name}' offers full-time staff employment opportunities.",
        node=leaf_full_time_positions,
        sources=uni.reference_urls,
        additional_instruction=(
            "Careers site or job postings indicating 'Full-time' staff positions suffice."
        ),
    )

    # 11) Provide benefits incl. health insurance to full-time staff
    leaf_benefits_health = evaluator.add_leaf(
        id=f"university_{idx}_benefits_health_insurance",
        desc=f"{uni_name} provides standard employee benefits including health insurance to full-time staff",
        parent=constraints_node,
        critical=True
    )
    await evaluator.verify(
        claim=(
            f"Full-time staff at '{uni_name}' receive standard employee benefits including health insurance."
        ),
        node=leaf_benefits_health,
        sources=uni.reference_urls,
        additional_instruction=(
            "HR/benefits pages, employee handbook, or official statements listing medical/health insurance for full-time staff are acceptable."
        ),
    )

    # 12) Publicly accessible careers/employment website with postings
    leaf_public_careers = evaluator.add_leaf(
        id=f"university_{idx}_public_careers_site",
        desc=f"{uni_name} has a publicly accessible careers or employment website where job postings can be viewed",
        parent=constraints_node,
        critical=True
    )
    await evaluator.verify(
        claim=(
            f"'{uni_name}' maintains a publicly accessible careers/employment website showing job postings."
        ),
        node=leaf_public_careers,
        sources=uni.reference_urls,
        additional_instruction=(
            "Accept official ATS portals (e.g., 'jobs.university.edu', 'careers.university.edu') or linked employment pages where postings are visible without special access."
        ),
    )

    # 13) Regionally accredited by US DoE-recognized body
    leaf_regionally_accredited = evaluator.add_leaf(
        id=f"university_{idx}_regionally_accredited",
        desc=f"{uni_name} is accredited by a regional accrediting body recognized by the US Department of Education",
        parent=constraints_node,
        critical=True
    )
    await evaluator.verify(
        claim=(
            f"'{uni_name}' is accredited by a regional accrediting body recognized by the US Department of Education."
        ),
        node=leaf_regionally_accredited,
        sources=uni.reference_urls,
        additional_instruction=(
            "Accept regional accreditors like HLC, SACSCOC, WASC (WSCUC), MSCHE, NECHE, or NWCCU, etc., if clearly indicated."
        ),
    )

    return {
        "name_provided_node": name_provided_node,
        "urls_provided_node": urls_provided_node,
        "constraints_node": constraints_node,
    }


# ------------------------- Main Evaluation ------------------------- #
async def evaluate_answer(
    client: LLMClient,
    answer: str,
    agent_name: str,
    answer_name: str,
    cache: CacheFileSys,
    semaphore: asyncio.Semaphore,
    logger: logging.Logger,
    model: str = "o4-mini",
) -> Dict:
    """
    Evaluate an answer for the student services coordinator universities task.
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

    # Create task grouping node (non-critical to allow partial credit, given framework constraints)
    task_node = evaluator.add_parallel(
        id="Four_Universities_Identification",
        desc="Identify four public universities in the U.S. meeting all required employment criteria (student services coordinator).",
        parent=root,
        critical=False
    )

    # Extract universities and reference URLs
    extracted_unis = await evaluator.extract(
        prompt=prompt_extract_universities(),
        template_class=UniversitiesExtraction,
        extraction_name="universities_extraction"
    )

    # Select first four; pad if fewer than four
    selected_unis: List[UniversityItem] = list(extracted_unis.universities[:4])
    while len(selected_unis) < 4:
        selected_unis.append(UniversityItem())

    # Add provided-name checks corresponding to rubric leaf items
    first_provided = evaluator.add_custom_node(
        result=_has_nonempty_name(selected_unis[0]),
        id="First_University_Provided",
        desc="A first qualifying university is identified and named",
        parent=task_node,
        critical=False
    )
    second_provided = evaluator.add_custom_node(
        result=_has_nonempty_name(selected_unis[1]),
        id="Second_University_Provided",
        desc="A second qualifying university is identified and named",
        parent=task_node,
        critical=False
    )
    third_provided = evaluator.add_custom_node(
        result=_has_nonempty_name(selected_unis[2]),
        id="Third_University_Provided",
        desc="A third qualifying university is identified and named",
        parent=task_node,
        critical=False
    )
    fourth_provided = evaluator.add_custom_node(
        result=_has_nonempty_name(selected_unis[3]),
        id="Fourth_University_Provided",
        desc="A fourth qualifying university is identified and named",
        parent=task_node,
        critical=False
    )

    # Build per-university verification subtrees
    prereq_nodes_per_uni: List[Dict[str, Any]] = []
    for i, uni in enumerate(selected_unis):
        nodes = await verify_university(evaluator, task_node, uni, i)
        prereq_nodes_per_uni.append(nodes)

    # Additional rubric item: Valid reference URLs provided for each university
    valid_urls_group = evaluator.add_parallel(
        id="Valid_Reference_URLs_Provided",
        desc="Valid reference URLs are provided for each university to verify the information",
        parent=task_node,
        critical=True
    )
    for i, uni in enumerate(selected_unis):
        leaf_valid_url = evaluator.add_leaf(
            id=f"university_{i}_valid_references",
            desc=f"University #{i + 1}: At least one provided URL is a publicly accessible page relevant to the university",
            parent=valid_urls_group,
            critical=True
        )
        await evaluator.verify(
            claim=(
                f"At least one of the provided URLs for '{_name_or_placeholder(i, uni)}' is a publicly accessible page relevant to this university."
            ),
            node=leaf_valid_url,
            sources=uni.reference_urls,
            additional_instruction=(
                "Accept official .edu pages, careers portals, HR/benefits pages, accreditation statements, or credible sources like Wikipedia. "
                "The page should be accessible without special credentials."
            ),
            extra_prerequisites=[
                prereq_nodes_per_uni[i]["name_provided_node"],
                prereq_nodes_per_uni[i]["urls_provided_node"]
            ],
        )

    # Return evaluation summary
    return evaluator.get_summary()