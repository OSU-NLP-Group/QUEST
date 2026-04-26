import asyncio
import logging
from typing import Any, List, Dict, Optional

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "business_admin_universities"
TASK_DESCRIPTION = (
    "Find three regionally accredited universities in the United States that each offer a bachelor's degree program in "
    "Business Administration and meet ALL of the following requirements:\n"
    "1) Regional accreditation (NECHE, MSCHE, HLC, SACSCOC, WSCUC, or NWCCU);\n"
    "2) Program requires at least 120 semester credit hours;\n"
    "3) Minimum cumulative GPA requirement of at least 2.0;\n"
    "4) Minimum GPA of at least 2.0 in the major;\n"
    "5) Study abroad programs allow students to earn at least 15 credits per semester abroad;\n"
    "6) Internship component requiring at least 135 hours;\n"
    "7) Experiential learning requirement of at least 50 hours;\n"
    "8) Capstone project completed during the final year;\n"
    "9) Honors program minimum GPA requirement between 3.2 and 3.5;\n"
    "10) Combined bachelor's/master's (4+1 or 5-year) program in Business Administration or related business field;\n"
    "11) Mandatory first-year on-campus housing for unmarried full-time undergraduates;\n"
    "12) Mandatory new-student orientation requirement;\n"
    "13) Health insurance requirement or university-sponsored plan; and\n"
    "14) Registered student organizations relevant to Business Administration.\n"
    "For each university, provide the name and reference URL(s) that verify these requirements."
)


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class UniversityEntry(BaseModel):
    name: Optional[str] = None
    evidence_urls: List[str] = Field(default_factory=list)


class UniversitiesExtraction(BaseModel):
    universities: List[UniversityEntry] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_universities() -> str:
    return """
    Extract up to the FIRST THREE distinct universities listed in the answer that are claimed to meet the stated constraints.
    For each university, return:
    - name: the university name exactly as written in the answer (string)
    - evidence_urls: an array of ALL reference URLs cited in the answer that pertain to this university and are intended to verify its accreditation, the Business Administration bachelor's program details (credits/GPA/capstone/internship/experiential learning), study abroad credits, honors program, combined degree option, first-year housing requirement, orientation requirement, health insurance requirement, and business-related student organizations.
    
    IMPORTANT:
    - Only include URLs that are explicitly present in the answer text (plain URLs or markdown links).
    - Do not fabricate or infer any URLs.
    - Preserve full URLs, including the protocol (http or https).
    - If fewer than three universities are present, return as many as exist.
    - If a university has no URLs in the answer, set evidence_urls to an empty array.
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def _first_n_universities(extraction: UniversitiesExtraction, n: int = 3) -> List[UniversityEntry]:
    items = extraction.universities[:n]
    # If fewer than n, pad with empty entries (to keep a fixed structure)
    while len(items) < n:
        items.append(UniversityEntry())
    return items


def _valid_urls(urls: List[str]) -> List[str]:
    # Basic filtering to keep URLs that look valid
    return [u for u in urls if isinstance(u, str) and u.strip().lower().startswith(("http://", "https://"))]


# --------------------------------------------------------------------------- #
# Verification logic per university                                           #
# --------------------------------------------------------------------------- #
async def verify_university(
    evaluator: Evaluator,
    parent_node,
    uni: UniversityEntry,
    index_1_based: int
) -> None:
    """
    Build the verification subtree for one university and execute all leaf verifications.
    Following the rubric tree: a parallel node with a set of critical leaves.
    """
    uni_node = evaluator.add_parallel(
        id=f"university_{index_1_based}",
        desc=f"University {index_1_based} satisfies all constraints and includes verifying URL(s)",
        parent=parent_node,
        critical=True  # Parent is critical; children must also be critical per framework constraints
    )

    name = (uni.name or "").strip()
    urls = _valid_urls(uni.evidence_urls)

    # Existence checks (custom nodes) - create early to gate other leaf verifications
    evaluator.add_custom_node(
        result=bool(name),
        id=f"u{index_1_based}_university_name_provided",
        desc="University name is provided",
        parent=uni_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=len(urls) > 0,
        id=f"u{index_1_based}_evidence_urls_verify_requirements",
        desc="Reference URL(s) are provided that collectively verify the above requirements for this university",
        parent=uni_node,
        critical=True
    )

    claims_and_nodes: List[tuple[str, List[str], Any, Optional[str]]] = []

    def add_leaf_and_prepare(
        node_id: str,
        desc: str,
        claim: str,
        add_ins: Optional[str] = None
    ):
        node = evaluator.add_leaf(
            id=node_id,
            desc=desc,
            parent=uni_node,
            critical=True
        )
        claims_and_nodes.append((claim, urls, node, add_ins))

    # Common instruction snippets
    accreditors_list = "NECHE (New England), MSCHE (Middle States), HLC (North Central), SACSCOC (Southern), WSCUC/WASC (Western), NWCCU (Northwest)"
    addins_program_names = (
        "Treat 'Business Administration' as including BBA, BSBA, BA in Business Administration, or equivalent titles."
    )
    addins_credits = (
        "Interpret 'semester credit hours' synonymously with 'credit hours' or 'units' where applicable. "
        "If the institution uses quarter hours, 180+ quarter hours is equivalent to 120+ semester hours."
    )
    addins_gpa = (
        "A requirement of 2.0 or higher satisfies 'at least 2.0'. Treat a 'C average' as equivalent to 2.0 GPA."
    )
    addins_study_abroad = (
        "The page should indicate students can earn at least 15 credits per semester abroad. "
        "A range that includes 15 (e.g., 12–18) or 'up to 18' also satisfies this."
    )
    addins_internship = (
        "Confirm an internship component for the Business Administration program requiring at least 135 hours of work "
        "(e.g., a course-based internship listing minimum hours)."
    )
    addins_experiential = (
        "Confirm there is an experiential learning requirement totaling at least 50 hours. "
        "This may be referred to as experiential learning, service-learning, co-curricular, career readiness hours, "
        "or similar, provided it is a formal requirement for degree completion."
    )
    addins_capstone = (
        "Confirm there is a capstone (senior capstone, culminating project, or equivalent) that is completed in the final year."
    )
    addins_honors = (
        "Confirm the university has an honors program with a minimum GPA requirement between 3.2 and 3.5 inclusive."
    )
    addins_combined = (
        "Confirm a combined bachelor's/master's program (4+1 or 5-year accelerated) in Business Administration or a related business field "
        "(e.g., accounting, finance, MBA, MS in business)."
    )
    addins_housing = (
        "Confirm a mandatory on-campus housing requirement for first-year full-time undergraduates. "
        "The 'unmarried' qualifier is conventional; accept policies that require first-year students to live on campus with listed exceptions."
    )
    addins_orientation = "Confirm a mandatory orientation requirement for new students."
    addins_insurance = (
        "Confirm health insurance requirements for all enrolled students (e.g., mandatory enrollment in a student health plan or proof of comparable coverage)."
    )
    addins_orgs = (
        "Confirm there are registered student organizations relevant to Business Administration (e.g., business club, accounting society, finance club, "
        "marketing association, entrepreneurship club, management or supply chain associations)."
    )

    # 1) US location
    add_leaf_and_prepare(
        node_id=f"u{index_1_based}_us_location",
        desc="University is located in the United States",
        claim=f"The university '{name}' is located in the United States.",
        add_ins="Accept evidence such as campus address including a US state, or explicit mentions of being a US university."
    )

    # 2) Regional accreditation
    add_leaf_and_prepare(
        node_id=f"u{index_1_based}_accreditation_requirement",
        desc="University holds regional accreditation from one of the six U.S. regional accrediting bodies (New England, Middle States, North Central, Southern, Western, or Northwest)",
        claim=(
            f"The university '{name}' holds regional accreditation from one of these accreditors: {accreditors_list}."
        ),
        add_ins="Look for explicit accreditation statements listing one of the six recognized regional accreditors."
    )

    # 3) Program availability (Business Administration bachelor's)
    add_leaf_and_prepare(
        node_id=f"u{index_1_based}_program_availability",
        desc="University offers a bachelor's degree program in Business Administration",
        claim=f"The university '{name}' offers a bachelor's degree program in Business Administration.",
        add_ins=addins_program_names
    )

    # 4) Credit requirements (>=120 semester credit hours)
    add_leaf_and_prepare(
        node_id=f"u{index_1_based}_credit_requirements",
        desc="Business Administration bachelor's program requires at least 120 semester credit hours for graduation",
        claim=(
            f"The Business Administration bachelor's program at '{name}' requires at least 120 semester credit hours (or equivalent) to graduate."
        ),
        add_ins=addins_credits
    )

    # 5) Overall GPA requirement (>=2.0)
    add_leaf_and_prepare(
        node_id=f"u{index_1_based}_overall_gpa_requirement",
        desc="Program has a minimum cumulative graduation GPA requirement of at least 2.0",
        claim=(
            f"The Business Administration program at '{name}' requires a minimum cumulative GPA of at least 2.0 to graduate."
        ),
        add_ins=addins_gpa
    )

    # 6) Major GPA requirement (>=2.0)
    add_leaf_and_prepare(
        node_id=f"u{index_1_based}_major_gpa_requirement",
        desc="Program has a minimum GPA requirement of at least 2.0 in major coursework",
        claim=(
            f"The Business Administration program at '{name}' requires a minimum GPA of at least 2.0 in the major (major coursework)."
        ),
        add_ins=addins_gpa
    )

    # 7) Study abroad credits (>=15 per semester)
    add_leaf_and_prepare(
        node_id=f"u{index_1_based}_study_abroad",
        desc="University offers study abroad programs where students can earn at least 15 credits per semester abroad",
        claim=(
            f"Students at '{name}' can earn at least 15 credits per semester while studying abroad."
        ),
        add_ins=addins_study_abroad
    )

    # 8) Internship component (>=135 hours)
    add_leaf_and_prepare(
        node_id=f"u{index_1_based}_internship_component",
        desc="Business Administration program includes an internship component requiring at least 135 hours of work",
        claim=(
            f"The Business Administration program at '{name}' includes an internship requiring at least 135 hours of work."
        ),
        add_ins=addins_internship
    )

    # 9) Experiential learning requirement (>=50 hours)
    add_leaf_and_prepare(
        node_id=f"u{index_1_based}_experiential_learning",
        desc="University has an experiential learning requirement of at least 50 hours for degree completion",
        claim=(
            f"'{name}' has an experiential learning requirement of at least 50 hours for degree completion."
        ),
        add_ins=addins_experiential
    )

    # 10) Capstone project in final year
    add_leaf_and_prepare(
        node_id=f"u{index_1_based}_capstone_project",
        desc="Program includes a capstone project requirement completed during the student's final year",
        claim=(
            f"The Business Administration program at '{name}' requires a capstone project (or equivalent) completed in the final year."
        ),
        add_ins=addins_capstone
    )

    # 11) Honors program GPA between 3.2 and 3.5
    add_leaf_and_prepare(
        node_id=f"u{index_1_based}_honors_program",
        desc="University offers an honors program with a minimum GPA requirement between 3.2 and 3.5 for participation",
        claim=(
            f"'{name}' offers an honors program that requires a minimum GPA between 3.2 and 3.5 (inclusive) to participate."
        ),
        add_ins=addins_honors
    )

    # 12) Combined bachelor's/master's (4+1 or 5-year) in Business/Admin or related business
    add_leaf_and_prepare(
        node_id=f"u{index_1_based}_combined_degree",
        desc="University offers a combined bachelor's/master's (4+1 or 5-year accelerated) program in Business Administration or a related business field",
        claim=(
            f"'{name}' offers a combined bachelor's/master's (4+1 or 5-year accelerated) program in Business Administration or a related business field."
        ),
        add_ins=addins_combined
    )

    # 13) Housing requirement (first-year on-campus)
    add_leaf_and_prepare(
        node_id=f"u{index_1_based}_housing_requirement",
        desc="University has mandatory first-year on-campus housing for unmarried full-time undergraduate students",
        claim=(
            f"'{name}' requires first-year full-time undergraduate students to live on campus (with standard exceptions)."
        ),
        add_ins=addins_housing
    )

    # 14) Orientation requirement (mandatory)
    add_leaf_and_prepare(
        node_id=f"u{index_1_based}_orientation_requirement",
        desc="University requires new students to complete a mandatory orientation program",
        claim=(
            f"'{name}' requires new students to complete a mandatory orientation program."
        ),
        add_ins=addins_orientation
    )

    # 15) Health insurance requirement / plan
    add_leaf_and_prepare(
        node_id=f"u{index_1_based}_health_insurance",
        desc="University maintains health insurance requirements for all enrolled students or provides a university-sponsored health insurance plan",
        claim=(
            f"'{name}' maintains a health insurance requirement for enrolled students, either requiring enrollment in a student plan or proof of comparable coverage."
        ),
        add_ins=addins_insurance
    )

    # 16) Student organizations related to Business Administration
    add_leaf_and_prepare(
        node_id=f"u{index_1_based}_student_organizations",
        desc="University has registered student organizations relevant to Business Administration that students can join",
        claim=(
            f"There are registered student organizations at '{name}' relevant to Business Administration (e.g., business, accounting, finance, marketing, entrepreneurship)."
        ),
        add_ins=addins_orgs
    )

    # Execute verifications in parallel under this university node
    await evaluator.batch_verify(claims_and_nodes)


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
    Evaluate an answer for the Business Administration universities task.
    """
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,
        agent_name=agent_name,
        answer_name=answer_name,
        client=client,
        task_description="Provide exactly three universities, each satisfying all listed institutional/program constraints for a Business Administration bachelor's program, and provide verifying reference URL(s) for each university",
        answer=answer,
        global_cache=cache,
        global_semaphore=semaphore,
        logger=logger,
        default_model=model
    )

    # NOTE: The rubric's root node is critical and aggregates 3 universities.
    # To satisfy the framework's rule that a critical parent can only have critical children,
    # we will set each university node as critical=True within verify_university().

    # Extraction
    extracted = await evaluator.extract(
        prompt=prompt_extract_universities(),
        template_class=UniversitiesExtraction,
        extraction_name="universities_extraction"
    )

    universities = _first_n_universities(extracted, 3)

    # Build verification subtrees for exactly three universities
    # All three are required (root critical → children must all pass).
    verify_tasks = []
    for i, uni in enumerate(universities, start=1):
        verify_tasks.append(verify_university(evaluator, root, uni, i))
    await asyncio.gather(*verify_tasks)

    # Return summary
    return evaluator.get_summary()