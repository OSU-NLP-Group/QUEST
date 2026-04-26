import asyncio
import logging
from typing import Any, List, Optional, Dict

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy

# --------------------------------------------------------------------------- #
# Task constants                                                              #
# --------------------------------------------------------------------------- #
TASK_ID = "online_grad_cert_data_analytics_constraints"
TASK_DESCRIPTION = (
    "Identify one online graduate certificate program in data analytics that is offered by a regionally-accredited "
    "university in the United States and meets ALL of the following requirements: "
    "(1) The program requires between 12 and 18 credit hours, "
    "(2) The program consists of 4 or 5 courses, "
    "(3) The curriculum includes training in at least two of the following programming languages: Python, R, or SQL, "
    "(4) The program can be completed entirely online, "
    "(5) Earned credits from the certificate can be applied toward a master's degree (either at the same institution or at partner institutions), "
    "(6) The program specifies a maximum completion timeframe for certificate completion."
)

RECOGNIZED_REGIONAL_ACCREDITORS = [
    "Higher Learning Commission", "HLC",
    "Southern Association of Colleges and Schools Commission on Colleges", "SACSCOC",
    "WASC Senior College and University Commission", "WSCUC",
    "Middle States Commission on Higher Education", "MSCHE",
    "New England Commission of Higher Education", "NECHE",
    "Northwest Commission on Colleges and Universities", "NWCCU"
]


# --------------------------------------------------------------------------- #
# Data models                                                                 #
# --------------------------------------------------------------------------- #
class ProgramSources(BaseModel):
    program_pages: List[str] = Field(default_factory=list)
    curriculum_pages: List[str] = Field(default_factory=list)
    delivery_pages: List[str] = Field(default_factory=list)
    credit_transfer_pages: List[str] = Field(default_factory=list)
    timeframe_pages: List[str] = Field(default_factory=list)
    accreditation_pages: List[str] = Field(default_factory=list)
    institution_pages: List[str] = Field(default_factory=list)


class ProgramExtraction(BaseModel):
    institution_name: Optional[str] = None
    program_name: Optional[str] = None
    program_field: Optional[str] = None  # e.g., Data Analytics / Data Science / Business Analytics
    credential_level: Optional[str] = None  # expecting "Graduate Certificate" or equivalent
    total_credits: Optional[str] = None  # keep as string (e.g., "12", "12-15", "15 credit hours")
    course_count: Optional[str] = None  # keep as string (e.g., "4", "5", "4 courses")
    languages: List[str] = Field(default_factory=list)  # subset of ["Python", "R", "SQL"] mentioned in the answer
    fully_online_statement: Optional[str] = None  # snippet from the answer stating fully online
    credit_transfer_statement: Optional[str] = None  # snippet stating credits apply to master's
    completion_timeframe: Optional[str] = None  # snippet specifying a max timeframe (e.g., "within 3 years")
    sources: ProgramSources = Field(default_factory=ProgramSources)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_program() -> str:
    return """
    Extract details for exactly one (the first mentioned if multiple) online graduate certificate program that the answer proposes.
    Only extract information explicitly present in the answer. Do not infer or add anything that is not stated.
    
    Return the following fields:
    - institution_name: Name of the university offering the program (string or null)
    - program_name: The name/title of the certificate program (string or null)
    - program_field: The field designation (e.g., "Data Analytics", "Data Science", "Business Analytics") (string or null)
    - credential_level: The credential designation as stated (e.g., "Graduate Certificate", "Postgraduate Certificate") (string or null)
    - total_credits: The total credit hours required as written in the answer (keep as string, e.g., "12", "12-18", "15 credit hours") (string or null)
    - course_count: The number of courses as written (keep as string, e.g., "4", "5", "4 courses") (string or null)
    - languages: List including any of the following that are explicitly mentioned as being part of the curriculum: "Python", "R", "SQL". Include only those explicitly mentioned for this program.
    - fully_online_statement: A short quoted snippet from the answer indicating the program is 100% online / fully online / can be completed entirely online (string or null)
    - credit_transfer_statement: A short quoted snippet indicating that earned certificate credits can apply to a master's degree (string or null)
    - completion_timeframe: A short quoted snippet stating a maximum time limit to complete the certificate (e.g., "must be completed within X years") (string or null)
    
    Also extract the URLs (only those explicitly present in the answer text) grouped as:
    - sources.program_pages: URLs for the main program page(s)
    - sources.curriculum_pages: URLs describing courses, curriculum, credits, or course count
    - sources.delivery_pages: URLs that mention online delivery / fully online
    - sources.credit_transfer_pages: URLs that mention applying earned credits to a master's degree
    - sources.timeframe_pages: URLs that specify a maximum completion timeframe (or time limit policy applying to the certificate)
    - sources.accreditation_pages: URLs indicating the institution's accreditation and accrediting agency
    - sources.institution_pages: URLs showing the university’s U.S. location or general institution info
    
    IMPORTANT:
    - Only include URLs explicitly present in the answer. If a relevant URL type is not present in the answer, return an empty list for that field.
    - If the answer mentions multiple programs, select and extract only the first one described in detail.
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _gather_sources(*url_lists: List[str]) -> List[str]:
    """Merge and deduplicate multiple URL lists, keep order of first occurrence."""
    seen = set()
    merged: List[str] = []
    for lst in url_lists:
        for u in lst or []:
            if isinstance(u, str) and u.strip() and u not in seen:
                seen.add(u)
                merged.append(u)
    return merged


def _name_or_generic(name: Optional[str], generic: str) -> str:
    if name and name.strip():
        return name.strip()
    return generic


# --------------------------------------------------------------------------- #
# Verification builder                                                        #
# --------------------------------------------------------------------------- #
async def build_verification_tree(evaluator: Evaluator, root_node, info: ProgramExtraction) -> None:
    """
    Build the verification tree following the rubric and run verifications.
    A top-level critical aggregator node is created to enforce 'all constraints must be met'.
    """
    # Overall critical aggregator (since evaluator root is non-critical by design)
    overall = evaluator.add_parallel(
        id="overall_constraints",
        desc="Overall: All constraints for the online graduate certificate program must be satisfied",
        parent=root_node,
        critical=True
    )

    # Group: Institution requirements (critical)
    institution_group = evaluator.add_parallel(
        id="institution_requirements",
        desc="Institution meets location and accreditation constraints",
        parent=overall,
        critical=True
    )

    # Group: Program type and structure (critical)
    type_structure_group = evaluator.add_parallel(
        id="program_type_and_structure",
        desc="Program type and structure meet credit-hour and course-count constraints",
        parent=overall,
        critical=True
    )

    # Group: Curriculum requirements (critical)
    curriculum_group = evaluator.add_parallel(
        id="curriculum_requirements",
        desc="Program field and curriculum content constraints are satisfied",
        parent=overall,
        critical=True
    )

    # Group: Delivery and academic policy requirements (critical)
    delivery_policy_group = evaluator.add_parallel(
        id="delivery_and_policy_requirements",
        desc="Online delivery and academic policy constraints are satisfied",
        parent=overall,
        critical=True
    )

    # Prepare commonly used source bundles
    program_urls = info.sources.program_pages
    curriculum_urls = info.sources.curriculum_pages
    delivery_urls = info.sources.delivery_pages
    transfer_urls = info.sources.credit_transfer_pages
    timeframe_urls = info.sources.timeframe_pages
    accreditation_urls = info.sources.accreditation_pages
    institution_urls = info.sources.institution_pages

    # Combined bundles for broader checks
    inst_related_sources = _gather_sources(program_urls, institution_urls, accreditation_urls)
    accreditation_related_sources = _gather_sources(accreditation_urls, institution_urls)
    curriculum_related_sources = _gather_sources(curriculum_urls, program_urls)
    delivery_related_sources = _gather_sources(delivery_urls, program_urls)
    transfer_related_sources = _gather_sources(transfer_urls, program_urls)
    timeframe_related_sources = _gather_sources(timeframe_urls, program_urls, institution_urls)

    institution_name = _name_or_generic(info.institution_name, "the institution")
    program_name = _name_or_generic(info.program_name, "the program")

    # ---------------- Institution requirements ----------------
    # us_institution
    us_inst_leaf = evaluator.add_leaf(
        id="us_institution",
        desc="The program is offered by a university located in the United States",
        parent=institution_group,
        critical=True
    )
    claim_us = f"The university offering {program_name} is located in the United States."
    await evaluator.verify(
        claim=claim_us,
        node=us_inst_leaf,
        sources=inst_related_sources,
        additional_instruction=(
            "Verify that the institution is a U.S. university. Evidence may include a U.S. state/city address on the page, "
            "references to being a U.S. public/private university, '.edu' domain context, or similar. If the page clearly indicates "
            "a U.S. location (e.g., 'California, USA', 'United States'), consider it supported."
        ),
    )

    # regional_accreditation_status
    accred_leaf = evaluator.add_leaf(
        id="regional_accreditation_status",
        desc="The university is regionally accredited (recognized U.S. regional accreditor)",
        parent=institution_group,
        critical=True
    )
    claim_accred = (
        f"{institution_name} is regionally accredited by a recognized U.S. regional accreditor "
        f"(e.g., HLC, SACSCOC, WSCUC, MSCHE, NECHE, or NWCCU)."
    )
    await evaluator.verify(
        claim=claim_accred,
        node=accred_leaf,
        sources=accreditation_related_sources,
        additional_instruction=(
            "Look for explicit statements of regional accreditation. Accept if the page indicates accreditation by any of: "
            + ", ".join(RECOGNIZED_REGIONAL_ACCREDITORS)
            + ". An accreditor directory listing page for the institution also counts."
        ),
    )

    # ---------------- Program type & structure ----------------
    # graduate_certificate_designation
    grad_cert_leaf = evaluator.add_leaf(
        id="graduate_certificate_designation",
        desc="The credential is explicitly designated as a graduate certificate (not undergraduate/professional certificate)",
        parent=type_structure_group,
        critical=True
    )
    claim_grad_cert = (
        f"{program_name} is explicitly designated as a graduate certificate program."
    )
    await evaluator.verify(
        claim=claim_grad_cert,
        node=grad_cert_leaf,
        sources=program_urls,
        additional_instruction=(
            "Confirm the page uses wording like 'Graduate Certificate'. Do not accept 'Undergraduate Certificate' or unrelated 'Professional Certificate' "
            "unless it also clearly states it's a graduate-level certificate."
        ),
    )

    # credit_hour_range
    credits_leaf = evaluator.add_leaf(
        id="credit_hour_range",
        desc="The program requires between 12 and 18 credit hours",
        parent=type_structure_group,
        critical=True
    )
    claim_credits = "This certificate requires between 12 and 18 total credit hours (inclusive)."
    await evaluator.verify(
        claim=claim_credits,
        node=credits_leaf,
        sources=curriculum_related_sources,
        additional_instruction=(
            "Use the curriculum/program page to verify the stated total credits fall within 12-18 credits inclusive. "
            "Ranges like '12-15 credits' are acceptable."
        ),
    )

    # course_count_requirement
    course_count_leaf = evaluator.add_leaf(
        id="course_count_requirement",
        desc="The program consists of 4 or 5 courses",
        parent=type_structure_group,
        critical=True
    )
    claim_courses = "This certificate consists of 4 or 5 courses in total."
    await evaluator.verify(
        claim=claim_courses,
        node=course_count_leaf,
        sources=curriculum_related_sources,
        additional_instruction=(
            "Confirm that the program requirements indicate either 4 or 5 courses. "
            "If the page clearly lists 4 required courses or states '4 courses' (or 5), it qualifies. "
            "If only credits are listed, ensure the page explicitly equates this to 4 or 5 courses (e.g., 12 credits at 3 credits each = 4 courses)."
        ),
    )

    # ---------------- Curriculum requirements ----------------
    # field_designation
    field_leaf = evaluator.add_leaf(
        id="field_designation",
        desc="The program is designated as a data analytics, data science, or business analytics certificate",
        parent=curriculum_group,
        critical=True
    )
    claim_field = (
        f"{program_name} is designated in the analytics domain (data analytics, data science, or business analytics)."
    )
    await evaluator.verify(
        claim=claim_field,
        node=field_leaf,
        sources=program_urls,
        additional_instruction=(
            "Confirm the program name/description explicitly indicates it is a data analytics, data science, or business analytics certificate. "
            "Close synonyms like 'analytics' with clear data focus are acceptable."
        ),
    )

    # programming_languages_requirement
    lang_leaf = evaluator.add_leaf(
        id="programming_languages_requirement",
        desc="The curriculum includes training in at least two of: Python, R, SQL",
        parent=curriculum_group,
        critical=True
    )
    claim_langs = (
        "The certificate's curriculum includes training in at least two of the following programming languages: Python, R, SQL."
    )
    langs_mentioned = ", ".join(info.languages) if info.languages else "None mentioned in the answer"
    await evaluator.verify(
        claim=claim_langs,
        node=lang_leaf,
        sources=curriculum_related_sources,
        additional_instruction=(
            "Verify using course lists/descriptions. Look for explicit mentions of 'Python', 'R' (R language), or 'SQL' (Structured Query Language). "
            "At least two of these must be clearly included in course titles/descriptions. "
            f"Languages reported in the answer: {langs_mentioned}."
        ),
    )

    # ---------------- Delivery & policy requirements ----------------
    # fully_online_delivery
    online_leaf = evaluator.add_leaf(
        id="fully_online_delivery",
        desc="The program can be completed entirely online with no required on-campus attendance",
        parent=delivery_policy_group,
        critical=True
    )
    claim_online = "This certificate can be completed entirely online with no required on-campus attendance."
    await evaluator.verify(
        claim=claim_online,
        node=online_leaf,
        sources=delivery_related_sources,
        additional_instruction=(
            "Accept terms such as '100% online', 'fully online', 'no campus visits required'. "
            "If optional campus experiences are offered but not required, it still qualifies."
        ),
    )

    # credit_transfer_provision
    transfer_leaf = evaluator.add_leaf(
        id="credit_transfer_provision",
        desc="The program explicitly states that earned credits can be applied toward a master's degree (same or partner institution)",
        parent=delivery_policy_group,
        critical=True
    )
    claim_transfer = (
        "Credits earned from this certificate can be applied toward a master's degree at the same institution or partner institutions."
    )
    await evaluator.verify(
        claim=claim_transfer,
        node=transfer_leaf,
        sources=transfer_related_sources,
        additional_instruction=(
            "Look for explicit stacking/transfer language such as 'stackable to a master's', "
            "'credits apply toward the MS in X', or equivalent statements."
        ),
    )

    # completion_timeframe_specified
    timeframe_leaf = evaluator.add_leaf(
        id="completion_timeframe_specified",
        desc="The program specifies a maximum timeframe for certificate completion",
        parent=delivery_policy_group,
        critical=True
    )
    claim_timeframe = "There is a specified maximum timeframe to complete the certificate (e.g., must be completed within N years)."
    await evaluator.verify(
        claim=claim_timeframe,
        node=timeframe_leaf,
        sources=timeframe_related_sources,
        additional_instruction=(
            "Accept if the page or an official academic policy page states a maximum time limit (e.g., 3/5/6 years) that applies to this graduate certificate. "
            "General graduate policies that explicitly include certificates also count."
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
    Evaluate an agent's answer for the online data analytics graduate certificate constraints task.
    """
    # Initialize evaluator (root is non-critical by design; we add a critical aggregator node under it)
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

    # Extract structured data from the answer
    program_info = await evaluator.extract(
        prompt=prompt_extract_program(),
        template_class=ProgramExtraction,
        extraction_name="program_extraction",
    )

    # Build verification tree and run checks
    await build_verification_tree(evaluator, root, program_info)

    # Return evaluation summary
    return evaluator.get_summary()