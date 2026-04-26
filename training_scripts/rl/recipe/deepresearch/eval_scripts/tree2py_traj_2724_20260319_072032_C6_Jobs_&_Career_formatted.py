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
TASK_ID = "uva_10th_president_profile"
TASK_DESCRIPTION = (
    "Identify the 10th president of the University of Virginia and provide comprehensive details about their educational "
    "credentials and career progression. Specifically, your answer must include: (1) The person's full name, "
    "(2) The year they were appointed or began serving as UVA's 10th president, "
    "(3) Complete details of their doctoral degree including the type of degree (EdD or PhD), the specific field of study or "
    "program concentration, the university that granted the degree, and the year the degree was completed, and "
    "(4) Details of the administrative position they held immediately prior to becoming university president including the specific "
    "position title, the name of the school, college, or unit where they served, and the institution where this role was held. "
    "All information must be verifiable through publicly available sources and URLs must be provided as supporting evidence."
)


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class UVA10thPresidentExtraction(BaseModel):
    # Person identification and appointment
    full_name: Optional[str] = None
    appointment_year: Optional[str] = None
    presidential_appointment_urls: List[str] = Field(default_factory=list)

    # Doctoral credentials
    degree_type: Optional[str] = None  # e.g., "PhD", "EdD", or textual variants provided by the answer
    doctoral_field_primary: Optional[str] = None  # e.g., "Higher Education", "Education Administration"
    doctoral_field_specific: Optional[str] = None  # e.g., "Higher Education Management", "Higher Education Administration"
    doctoral_institution_name: Optional[str] = None
    doctoral_institution_location: Optional[str] = None  # state or location string if provided
    doctoral_completion_year: Optional[str] = None
    doctoral_urls: List[str] = Field(default_factory=list)

    # Administrative role immediately prior to presidency
    prior_role_title: Optional[str] = None
    prior_role_school_or_unit: Optional[str] = None
    prior_role_institution: Optional[str] = None
    prior_role_urls: List[str] = Field(default_factory=list)

    # Optional timeline information
    prior_role_start_year: Optional[str] = None
    prior_role_duration_years: Optional[str] = None  # keep as string to allow "approx. 5"


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_uva_10th_profile() -> str:
    return """
    Extract all of the following information exactly as presented in the answer text. Do not invent or infer anything not explicitly stated.

    Person and appointment details:
    - full_name: the full name of the person identified as the 10th president of the University of Virginia.
    - appointment_year: the year they were appointed or began serving as UVA's 10th president (just the year string if possible).
    - presidential_appointment_urls: an array of all URLs cited in the answer that document the presidential appointment (appointment announcement, official bio, reputable news coverage, etc.).

    Doctoral credentials:
    - degree_type: the type of doctoral degree (e.g., 'PhD', 'EdD', or a textual variant provided).
    - doctoral_field_primary: the primary field or discipline for the doctoral degree (e.g., 'Higher Education', 'Education Administration', 'Education Management', or similar).
    - doctoral_field_specific: the specific program name or concentration if provided (e.g., 'Higher Education Management', 'Higher Education Administration'); null if not specified.
    - doctoral_institution_name: the full name of the university that granted the doctoral degree.
    - doctoral_institution_location: the state or location of the granting institution if provided; null if not stated.
    - doctoral_completion_year: the year the doctoral degree was completed/awarded; null if not stated.
    - doctoral_urls: an array of all URLs cited in the answer that document the doctoral credential.

    Prior administrative role (immediately before becoming president):
    - prior_role_title: title of the administrative position (e.g., 'Dean', 'Provost', 'Vice President').
    - prior_role_school_or_unit: specific school/college/unit where the person served (e.g., 'School of Education'); null if not stated.
    - prior_role_institution: institution where the prior role was held (e.g., 'University of X').
    - prior_role_urls: an array of all URLs cited in the answer that document the prior administrative role.

    Optional timeline details (if provided in the answer):
    - prior_role_start_year: the year they began serving in the prior administrative role; null if not stated.
    - prior_role_duration_years: approximate number of years served in that role as stated or computed in the answer; null if not stated.

    Special rules for URL extraction:
    - Extract only URLs explicitly shown in the answer (plain links or links embedded in markdown).
    - If a field is not mentioned in the answer, set it to null (or an empty array for URL lists).
    """


# --------------------------------------------------------------------------- #
# Verification helpers                                                        #
# --------------------------------------------------------------------------- #
async def build_person_identification(
    evaluator: Evaluator,
    parent,
    data: UVA10thPresidentExtraction,
) -> None:
    # Person Identification (critical, parallel)
    person_node = evaluator.add_parallel(
        id="person_identification",
        desc="Correctly identify the specific individual who became the 10th president of the University of Virginia",
        parent=parent,
        critical=True,
    )

    # Presidential Appointment (critical, sequential; gate with URL presence first)
    appointment_node = evaluator.add_sequential(
        id="presidential_appointment",
        desc="Verify the appointment details for the 10th president position at the University of Virginia",
        parent=person_node,
        critical=True,
    )

    # URL existence check (critical)
    pres_url_exist = evaluator.add_custom_node(
        result=bool(data.presidential_appointment_urls),
        id="presidential_appointment_url_provided",
        desc="Provide a valid URL reference documenting the presidential appointment (existence check)",
        parent=appointment_node,
        critical=True,
    )

    # Name verification (critical leaf) - depends on URL existence
    name_leaf = evaluator.add_leaf(
        id="name_verification",
        desc="Provide the correct full name of the person who was appointed as UVA's 10th president",
        parent=person_node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"This page identifies the 10th president of the University of Virginia as '{data.full_name}'. "
              f"Allow reasonable variations (middle initials, case, punctuation).",
        node=name_leaf,
        sources=data.presidential_appointment_urls,
        additional_instruction="Focus on whether the page clearly names the University of Virginia's 10th president as the extracted person. "
                               "If the page states a different ordinal (e.g., 9th), it should be considered not supported.",
    )

    # University verification (critical leaf)
    univ_leaf = evaluator.add_leaf(
        id="university_verification",
        desc="Confirm that the University of Virginia is the institution where this person serves/served as the 10th president",
        parent=appointment_node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"This page states that {data.full_name} serves or served as the 10th president of the University of Virginia.",
        node=univ_leaf,
        sources=data.presidential_appointment_urls,
        additional_instruction="Treat '10th president' wording flexibly (e.g., 'tenth'), but the ordinal must correspond to 10. "
                               "If the page explicitly states a different ordinal for UVA presidency, do not support the claim.",
    )

    # Appointment Year (critical leaf)
    appt_year_leaf = evaluator.add_leaf(
        id="appointment_year",
        desc="Identify the year when this person was appointed or began serving as UVA's 10th president",
        parent=appointment_node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"This page indicates that {data.full_name} was appointed or began serving as UVA's 10th president in {data.appointment_year}.",
        node=appt_year_leaf,
        sources=data.presidential_appointment_urls,
        additional_instruction="Allow month/day details; the year must match exactly.",
    )


async def build_doctoral_credentials(
    evaluator: Evaluator,
    parent,
    data: UVA10thPresidentExtraction,
) -> Dict[str, Any]:
    # Doctoral Credentials (critical, parallel)
    doc_node = evaluator.add_parallel(
        id="doctoral_credentials",
        desc="Accurate identification and verification of the education leader's doctorate degree",
        parent=parent,
        critical=True,
    )

    # URLs existence (critical)
    doc_urls_exist = evaluator.add_custom_node(
        result=bool(data.doctoral_urls),
        id="doctoral_urls_provided",
        desc="Provide a valid URL reference documenting the doctoral degree completion (existence check)",
        parent=doc_node,
        critical=True,
    )

    # Degree type identification (critical)
    degree_type_leaf = evaluator.add_leaf(
        id="degree_type_identification",
        desc="Correctly identify whether the doctorate is an EdD or PhD",
        parent=doc_node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"This page indicates that {data.full_name} holds a doctoral degree of type '{data.degree_type}' "
              f"(e.g., PhD/Ph.D. or EdD/Ed.D.).",
        node=degree_type_leaf,
        sources=data.doctoral_urls,
        additional_instruction="Allow common textual variants, such as 'Doctor of Philosophy (Ph.D.)' for PhD or "
                               "'Doctor of Education (Ed.D.)' for EdD.",
    )

    # Primary field (critical)
    primary_field_leaf = evaluator.add_leaf(
        id="primary_field",
        desc="Identify the primary field as Higher Education, Education Management, or closely related discipline",
        parent=doc_node,
        critical=True,
    )
    await evaluator.verify(
        claim=(
            f"For {data.full_name}'s doctoral degree, the field or program is '{data.doctoral_field_primary}', which is within "
            f"Higher Education / Education Administration / Education Management or a closely related discipline."
        ),
        node=primary_field_leaf,
        sources=data.doctoral_urls,
        additional_instruction="Accept closely related labels such as 'Higher Education Administration', 'Higher Education Leadership', "
                               "'Education Administration/Management', or equivalent Higher Education-focused programs.",
    )

    # Granting institution (critical, parallel) - only critical children beneath this
    grant_node = evaluator.add_parallel(
        id="granting_institution",
        desc="Identify the university that granted the doctoral degree",
        parent=doc_node,
        critical=True,
    )

    institution_name_leaf = evaluator.add_leaf(
        id="institution_name",
        desc="Provide the complete name of the institution that awarded the doctorate",
        parent=grant_node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"This page indicates that the doctoral degree was awarded by {data.doctoral_institution_name}.",
        node=institution_name_leaf,
        sources=data.doctoral_urls,
        additional_instruction="The page should clearly attribute the doctoral degree to the named institution.",
    )

    doctorate_doc_leaf = evaluator.add_leaf(
        id="doctorate_documentation_verified",
        desc="A provided source page documents the doctoral degree completion",
        parent=grant_node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"This page documents the completion/award of {data.full_name}'s doctoral degree at {data.doctoral_institution_name}.",
        node=doctorate_doc_leaf,
        sources=data.doctoral_urls,
        additional_instruction="The page should explicitly mention the person's doctoral degree, ideally with institution and/or program.",
    )

    # Year of completion (critical)
    year_completion_leaf = evaluator.add_leaf(
        id="year_of_completion",
        desc="Identify the year the doctoral degree was completed or awarded",
        parent=doc_node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"This page indicates that the doctoral degree was completed or awarded in {data.doctoral_completion_year}.",
        node=year_completion_leaf,
        sources=data.doctoral_urls,
        additional_instruction="Allow month/day details; the year must match exactly.",
    )

    # Optional doctoral details (non-critical, parallel) - placed under root later
    return {"doc_urls_exist": doc_urls_exist}


async def build_doctoral_optional_details(
    evaluator: Evaluator,
    parent,
    data: UVA10thPresidentExtraction,
    doc_urls_exist_node,
) -> None:
    # Optional doctoral details (non-critical, parallel)
    optional_node = evaluator.add_parallel(
        id="doctoral_optional_details",
        desc="Optional specificity for doctoral credentials (non-critical)",
        parent=parent,
        critical=False,
    )

    # Field specificity (non-critical)
    field_spec_leaf = evaluator.add_leaf(
        id="field_specificity",
        desc="Provide the specific program name or concentration (if available)",
        parent=optional_node,
        critical=False,
    )
    await evaluator.verify(
        claim=f"This page mentions the specific program or concentration '{data.doctoral_field_specific}' for the doctoral degree.",
        node=field_spec_leaf,
        sources=data.doctoral_urls,
        additional_instruction="Treat reasonable variants (e.g., 'Higher Education Administration' vs 'Administration of Higher Education') as acceptable.",
    )

    # Institution location (non-critical)
    inst_loc_leaf = evaluator.add_leaf(
        id="institution_location",
        desc="Identify the state where the granting institution is located (if available)",
        parent=optional_node,
        critical=False,
    )
    await evaluator.verify(
        claim=f"This page indicates that {data.doctoral_institution_name} is located in {data.doctoral_institution_location}.",
        node=inst_loc_leaf,
        sources=data.doctoral_urls,
        additional_instruction="Only pass if the page provides sufficient evidence for the location claim.",
    )


async def build_admin_career_path(
    evaluator: Evaluator,
    parent,
    data: UVA10thPresidentExtraction,
) -> Dict[str, Any]:
    # Administrative Career Path (critical, sequential) - gate with URL presence first
    admin_node = evaluator.add_sequential(
        id="administrative_career_path",
        desc="Verification of the administrative position held immediately before assuming the university presidency",
        parent=parent,
        critical=True,
    )

    prior_url_exist = evaluator.add_custom_node(
        result=bool(data.prior_role_urls),
        id="prior_role_url_provided",
        desc="Provide a valid URL reference documenting the prior administrative role (existence check)",
        parent=admin_node,
        critical=True,
    )

    # Position Title (critical)
    pos_title_leaf = evaluator.add_leaf(
        id="position_title",
        desc="Provide the correct title of the administrative position (e.g., Dean, Provost, Vice President)",
        parent=admin_node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"This page indicates that, prior to becoming president of UVA, {data.full_name} served as '{data.prior_role_title}'.",
        node=pos_title_leaf,
        sources=data.prior_role_urls,
        additional_instruction="Allow reasonable wording variants but ensure the role/title is accurately reflected.",
    )

    # School or Unit (critical)
    school_unit_leaf = evaluator.add_leaf(
        id="school_or_unit",
        desc="Identify the specific school, college, or unit within the university where this position was held",
        parent=admin_node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"This page indicates that the position was held in/at the '{data.prior_role_school_or_unit}'.",
        node=school_unit_leaf,
        sources=data.prior_role_urls,
        additional_instruction="The unit/school/college should be clearly referenced on the page.",
    )

    # Institution of Prior Role (critical)
    inst_prior_leaf = evaluator.add_leaf(
        id="institution_of_prior_role",
        desc="Identify the institution where the prior administrative role was held",
        parent=admin_node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"This page indicates that the prior administrative role was held at {data.prior_role_institution}.",
        node=inst_prior_leaf,
        sources=data.prior_role_urls,
        additional_instruction="The institution name should appear explicitly.",
    )

    # An additional explicit relevance check (critical)
    prior_role_doc_leaf = evaluator.add_leaf(
        id="prior_role_documentation",
        desc="A provided source page documents the prior administrative role",
        parent=admin_node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"This page documents {data.full_name}'s prior administrative role before becoming president of UVA.",
        node=prior_role_doc_leaf,
        sources=data.prior_role_urls,
        additional_instruction="It should be clear that the page is about the same person and describes the administrative role in question.",
    )

    return {"prior_url_exist": prior_url_exist}


async def build_admin_timeline(
    evaluator: Evaluator,
    parent,
    data: UVA10thPresidentExtraction,
) -> None:
    # Administrative Experience Timeline (non-critical, parallel)
    timeline_node = evaluator.add_parallel(
        id="administrative_experience_timeline",
        desc="Verify the timeline of administrative service in the role prior to presidency",
        parent=parent,
        critical=False,
    )

    # Start Year (non-critical)
    start_year_leaf = evaluator.add_leaf(
        id="prior_role_start_year",
        desc="Identify the year when the person began serving in the administrative role prior to presidency",
        parent=timeline_node,
        critical=False,
    )
    await evaluator.verify(
        claim=f"This page indicates that {data.full_name} began the prior administrative role in {data.prior_role_start_year}.",
        node=start_year_leaf,
        sources=data.prior_role_urls,
        additional_instruction="Allow month/day details; the year must match exactly when stated.",
    )

    # Duration of Service (non-critical)
    duration_leaf = evaluator.add_leaf(
        id="duration_of_service",
        desc="Calculate or identify the approximate number of years served in that administrative role",
        parent=timeline_node,
        critical=False,
    )
    await evaluator.verify(
        claim=f"This page indicates that {data.full_name} served in the prior role for approximately {data.prior_role_duration_years} years.",
        node=duration_leaf,
        sources=data.prior_role_urls,
        additional_instruction="Allow approximations if the page provides start/end years enabling a reasonable calculation.",
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
    # Initialize evaluator with a parallel root (non-critical to allow mixed critical children)
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
    extracted = await evaluator.extract(
        prompt=prompt_extract_uva_10th_profile(),
        template_class=UVA10thPresidentExtraction,
        extraction_name="uva_10th_president_profile_extraction",
    )

    # Build verification tree according to rubric (with safe structural adjustments)
    # 1) Person Identification + Presidential Appointment
    await build_person_identification(evaluator, root, extracted)

    # 2) Doctoral Credentials (critical) + Optional Doctoral Details (non-critical)
    doc_info = await build_doctoral_credentials(evaluator, root, extracted)
    await build_doctoral_optional_details(evaluator, root, extracted, doc_info.get("doc_urls_exist"))

    # 3) Administrative Career Path (critical)
    admin_info = await build_admin_career_path(evaluator, root, extracted)

    # 4) Administrative Experience Timeline (non-critical)
    await build_admin_timeline(evaluator, root, extracted)

    # Return summary
    return evaluator.get_summary()