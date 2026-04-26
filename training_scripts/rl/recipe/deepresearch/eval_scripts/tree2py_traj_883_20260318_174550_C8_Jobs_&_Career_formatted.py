import asyncio
import logging
from typing import Optional, List, Dict, Any, Tuple

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator, AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "ga_edu_leadership_requirements"
TASK_DESCRIPTION = (
    "A Georgia-based education career consulting firm is developing a comprehensive reference guide for aspiring "
    "education leaders. For each of the following four senior education leadership positions in Georgia's public "
    "education system, determine and document the complete minimum qualification requirements: (1) School "
    "Superintendent for Gwinnett County Schools, Georgia's largest school district with over 178,000 students; "
    "(2) School Superintendent for Forsyth County Schools, Georgia's fifth-largest school district; (3) Athletic "
    "Director for Georgia State University, a Division I FBS institution in the Sun Belt Conference; and (4) "
    "Athletic Director for Georgia Southern University, a Division I FBS institution in the Sun Belt Conference. "
    "For each position, provide: the minimum educational degree level and field of study required; all required "
    "Georgia state certifications or professional licenses (with specific regulation or rule citations where "
    "applicable); the minimum number of years of teaching experience (for superintendent positions) or coaching/"
    "athletic administration experience (for athletic director positions) typically required; the minimum "
    "administrative experience requirements specifying the types and duration of leadership roles needed (e.g., "
    "assistant principal, principal, assistant athletic director); the total minimum career timeline in years from "
    "entry-level position to eligibility for the senior role; and the current typical salary range based on available "
    "public data from 2024-2025."
)


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class FieldWithSources(BaseModel):
    """
    A generic container for a textual requirement/value plus URLs explicitly cited
    in the answer as sources for this specific requirement/value.
    """
    text: Optional[str] = None
    source_urls: List[str] = Field(default_factory=list)


class SuperintendentRequirements(BaseModel):
    """Structured minimum requirement set for a Georgia school district superintendent."""
    degree_level: Optional[FieldWithSources] = None
    degree_field: Optional[FieldWithSources] = None
    required_state_certs_licenses_with_citation: Optional[FieldWithSources] = None
    prerequisite_leadership_certification: Optional[FieldWithSources] = None
    required_assessments: Optional[FieldWithSources] = None
    teaching_experience_years: Optional[FieldWithSources] = None
    admin_experience_roles_duration: Optional[FieldWithSources] = None
    total_career_timeline_years: Optional[FieldWithSources] = None
    salary_range_2024_2025: Optional[FieldWithSources] = None


class AthleticDirectorRequirements(BaseModel):
    """Structured minimum requirement set for a Division I FBS Athletic Director."""
    degree_level: Optional[FieldWithSources] = None
    degree_field: Optional[FieldWithSources] = None
    required_licenses_or_certs: Optional[FieldWithSources] = None
    ncaa_compliance_requirement: Optional[FieldWithSources] = None
    title_ix_requirement: Optional[FieldWithSources] = None
    budget_management_requirement: Optional[FieldWithSources] = None
    experience_years_coaching_or_admin: Optional[FieldWithSources] = None
    admin_experience_roles_duration: Optional[FieldWithSources] = None
    total_career_timeline_years: Optional[FieldWithSources] = None
    salary_range_2024_2025: Optional[FieldWithSources] = None


class FullExtraction(BaseModel):
    """Aggregated extraction for all four target positions."""
    gwinnett_superintendent: Optional[SuperintendentRequirements] = None
    forsyth_superintendent: Optional[SuperintendentRequirements] = None
    gsu_athletic_director: Optional[AthleticDirectorRequirements] = None
    gsouthern_athletic_director: Optional[AthleticDirectorRequirements] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_requirements() -> str:
    return """
    Extract a structured summary of the minimum qualification requirements as the answer presents them for each of the four positions:
    1) School Superintendent – Gwinnett County Schools,
    2) School Superintendent – Forsyth County Schools,
    3) Athletic Director – Georgia State University (Division I FBS),
    4) Athletic Director – Georgia Southern University (Division I FBS).

    IMPORTANT:
    – Extract only what is explicitly claimed in the answer text.
    – For every field, also extract the list of URLs explicitly cited in the answer that directly support ONLY that specific field. Do not invent URLs.
    – If the answer says “none specified” or similar, set text accordingly and still include any URLs that support that statement (if any).
    – If the answer lacks either the text or any supporting URLs for a field, set text to null if missing; set source_urls to an empty array if no URLs are present in the answer for that specific field.

    Return a JSON with the following top-level keys and nested fields:

    {
      "gwinnett_superintendent": {
        "degree_level": {"text": string|null, "source_urls": [urls...]},
        "degree_field": {"text": string|null, "source_urls": [urls...]},
        "required_state_certs_licenses_with_citation": {"text": string|null, "source_urls": [urls...]},
        "prerequisite_leadership_certification": {"text": string|null, "source_urls": [urls...]},
        "required_assessments": {"text": string|null, "source_urls": [urls...]},
        "teaching_experience_years": {"text": string|null, "source_urls": [urls...]},
        "admin_experience_roles_duration": {"text": string|null, "source_urls": [urls...]},
        "total_career_timeline_years": {"text": string|null, "source_urls": [urls...]},
        "salary_range_2024_2025": {"text": string|null, "source_urls": [urls...]}
      },
      "forsyth_superintendent": {
        "degree_level": {"text": string|null, "source_urls": [urls...]},
        "degree_field": {"text": string|null, "source_urls": [urls...]},
        "required_state_certs_licenses_with_citation": {"text": string|null, "source_urls": [urls...]},
        "prerequisite_leadership_certification": {"text": string|null, "source_urls": [urls...]},
        "required_assessments": {"text": string|null, "source_urls": [urls...]},
        "teaching_experience_years": {"text": string|null, "source_urls": [urls...]},
        "admin_experience_roles_duration": {"text": string|null, "source_urls": [urls...]},
        "total_career_timeline_years": {"text": string|null, "source_urls": [urls...]},
        "salary_range_2024_2025": {"text": string|null, "source_urls": [urls...]}
      },
      "gsu_athletic_director": {
        "degree_level": {"text": string|null, "source_urls": [urls...]},
        "degree_field": {"text": string|null, "source_urls": [urls...]},
        "required_licenses_or_certs": {"text": string|null, "source_urls": [urls...]},
        "ncaa_compliance_requirement": {"text": string|null, "source_urls": [urls...]},
        "title_ix_requirement": {"text": string|null, "source_urls": [urls...]},
        "budget_management_requirement": {"text": string|null, "source_urls": [urls...]},
        "experience_years_coaching_or_admin": {"text": string|null, "source_urls": [urls...]},
        "admin_experience_roles_duration": {"text": string|null, "source_urls": [urls...]},
        "total_career_timeline_years": {"text": string|null, "source_urls": [urls...]},
        "salary_range_2024_2025": {"text": string|null, "source_urls": [urls...]}
      },
      "gsouthern_athletic_director": {
        "degree_level": {"text": string|null, "source_urls": [urls...]},
        "degree_field": {"text": string|null, "source_urls": [urls...]},
        "required_licenses_or_certs": {"text": string|null, "source_urls": [urls...]},
        "ncaa_compliance_requirement": {"text": string|null, "source_urls": [urls...]},
        "title_ix_requirement": {"text": string|null, "source_urls": [urls...]},
        "budget_management_requirement": {"text": string|null, "source_urls": [urls...]},
        "experience_years_coaching_or_admin": {"text": string|null, "source_urls": [urls...]},
        "admin_experience_roles_duration": {"text": string|null, "source_urls": [urls...]},
        "total_career_timeline_years": {"text": string|null, "source_urls": [urls...]},
        "salary_range_2024_2025": {"text": string|null, "source_urls": [urls...]}
      }
    }

    Special notes for URL extraction:
    – Only include URLs that are explicitly present in the answer for the specific field. If a URL is given in Markdown, extract the actual URL.
    – Do not infer or fabricate any URL. If missing, leave the list empty.
    """


# --------------------------------------------------------------------------- #
# Helper functions for verification                                           #
# --------------------------------------------------------------------------- #
def _exists_with_sources(field: Optional[FieldWithSources]) -> bool:
    return bool(field and field.text and str(field.text).strip()) and bool(field and field.source_urls and len(field.source_urls) > 0)


def _build_additional_instruction_generic(field_label: str, position_label: str) -> str:
    """
    A generic verification helper instruction reused across leaves.
    """
    return (
        f"Verify strictly against the provided webpage(s) whether the claim about '{field_label}' for the role "
        f"'{position_label}' is explicitly stated or very clearly implied. Allow minor wording variations, "
        f"synonyms, or formatting differences (e.g., 'Master's degree from an accredited institution' vs 'graduate degree'). "
        f"If the answer claims 'none specified' or 'not required', accept only when the provided source does not list a "
        f"specific requirement or explicitly notes that none is required. If a page is irrelevant/inaccessible, treat as not supported. "
        f"For salary ranges, accept ranges from public, current postings or official public data that reasonably fall within the 2024–2025 timeframe."
    )


async def _add_field_verification(
    evaluator: Evaluator,
    parent_node,
    *,
    field_node_id: str,
    field_desc: str,
    claim_text: str,
    position_label: str,
    field_value: Optional[FieldWithSources],
    additional_instruction: Optional[str] = None,
    critical: bool = True
) -> None:
    """
    Create a sequential verification node for a single requirement field:
    1) Existence with at least one source URL (custom leaf).
    2) Claim supported by cited sources (verify_by_urls).
    """
    # Sequential node for this field (critical as per rubric)
    seq_node = evaluator.add_sequential(
        id=field_node_id,
        desc=field_desc,
        parent=parent_node,
        critical=critical
    )

    # Existence check (gates verification)
    evaluator.add_custom_node(
        result=_exists_with_sources(field_value),
        id=f"{field_node_id}_exists",
        desc=f"{field_desc} is provided in the answer with at least one supporting source URL",
        parent=seq_node,
        critical=True
    )

    # Verification leaf
    verify_leaf = evaluator.add_leaf(
        id=f"{field_node_id}_supported",
        desc=f"{field_desc} is supported by the cited source(s)",
        parent=seq_node,
        critical=True
    )

    urls = field_value.source_urls if field_value else []
    add_ins = additional_instruction or _build_additional_instruction_generic(field_desc, position_label)

    await evaluator.verify(
        claim=claim_text,
        node=verify_leaf,
        sources=urls,
        additional_instruction=add_ins
    )


# --------------------------------------------------------------------------- #
# Position-specific verification builders                                     #
# --------------------------------------------------------------------------- #
async def verify_superintendent_position(
    evaluator: Evaluator,
    parent_node,
    *,
    position_node_id: str,
    position_label: str,
    data: Optional[SuperintendentRequirements]
) -> None:
    """
    Build verification sub-tree for a superintendent role.
    """
    pos_node = evaluator.add_parallel(
        id=position_node_id,
        desc=f"Minimum qualification requirements for {position_label}",
        parent=parent_node,
        critical=False
    )

    data = data or SuperintendentRequirements()

    # Degree level
    await _add_field_verification(
        evaluator,
        pos_node,
        field_node_id=f"{position_node_id.split('_')[0]}_education_degree_level" if position_node_id.startswith(("gwinnett", "forsyth")) else f"{position_node_id}_education_degree_level",
        field_desc="States the minimum required educational degree level (including any accreditation requirement if specified by governing requirement/constraint).",
        claim_text=f"According to the provided source(s), the minimum required educational degree level for {position_label} is: {data.degree_level.text if data.degree_level and data.degree_level.text else ''}.",
        position_label=position_label,
        field_value=data.degree_level
    )

    # Degree field of study
    await _add_field_verification(
        evaluator,
        pos_node,
        field_node_id=f"{position_node_id.split('_')[0]}_education_field_of_study" if position_node_id.startswith(("gwinnett", "forsyth")) else f"{position_node_id}_education_field_of_study",
        field_desc="States the required/acceptable field of study for the minimum degree (or explicitly notes if no specific field is stated by the governing requirement/constraint).",
        claim_text=f"According to the provided source(s), the required/acceptable degree field(s) for {position_label} is/are: {data.degree_field.text if data.degree_field and data.degree_field.text else ''}.",
        position_label=position_label,
        field_value=data.degree_field
    )

    # Required GA state certs/licenses (with citation)
    await _add_field_verification(
        evaluator,
        pos_node,
        field_node_id=f"{position_node_id.split('_')[0]}_required_state_certs_licenses_with_citation",
        field_desc="Identifies all required Georgia state certifications/licenses for superintendent eligibility, including applicable rule/regulation citations.",
        claim_text=f"According to the provided source(s), the required Georgia certifications/licenses (with relevant citation where applicable) for {position_label} include: {data.required_state_certs_licenses_with_citation.text if data.required_state_certs_licenses_with_citation and data.required_state_certs_licenses_with_citation.text else ''}.",
        position_label=position_label,
        field_value=data.required_state_certs_licenses_with_citation
    )

    # Prerequisite leadership certification
    await _add_field_verification(
        evaluator,
        pos_node,
        field_node_id=f"{position_node_id.split('_')[0]}_prerequisite_leadership_certification",
        field_desc="States prerequisite leadership certification(s) required before superintendent certification/eligibility.",
        claim_text=f"According to the provided source(s), the prerequisite leadership certification(s) prior to superintendent certification/eligibility for {position_label} are: {data.prerequisite_leadership_certification.text if data.prerequisite_leadership_certification and data.prerequisite_leadership_certification.text else ''}.",
        position_label=position_label,
        field_value=data.prerequisite_leadership_certification
    )

    # Required assessments
    await _add_field_verification(
        evaluator,
        pos_node,
        field_node_id=f"{position_node_id.split('_')[0]}_required_assessments",
        field_desc="States required assessments for superintendent certification/eligibility (e.g., GACE/Georgia Educator Ethics for Educational Leadership).",
        claim_text=f"According to the provided source(s), the required assessment(s) for superintendent certification/eligibility for {position_label} include: {data.required_assessments.text if data.required_assessments and data.required_assessments.text else ''}.",
        position_label=position_label,
        field_value=data.required_assessments
    )

    # Teaching experience (years)
    await _add_field_verification(
        evaluator,
        pos_node,
        field_node_id=f"{position_node_id.split('_')[0]}_teaching_experience_years",
        field_desc="Provides the minimum number of years of successful classroom teaching experience required (in years) for superintendent eligibility.",
        claim_text=f"According to the provided source(s), the minimum years of successful classroom teaching experience required for {position_label} is: {data.teaching_experience_years.text if data.teaching_experience_years and data.teaching_experience_years.text else ''}.",
        position_label=position_label,
        field_value=data.teaching_experience_years
    )

    # Administrative experience – roles and duration
    await _add_field_verification(
        evaluator,
        pos_node,
        field_node_id=f"{position_node_id.split('_')[0]}_admin_experience_roles_duration",
        field_desc="Provides the minimum administrative/leadership experience requirements, specifying role types and duration.",
        claim_text=f"According to the provided source(s), the minimum administrative/leadership experience (roles and duration) for {position_label} is: {data.admin_experience_roles_duration.text if data.admin_experience_roles_duration and data.admin_experience_roles_duration.text else ''}.",
        position_label=position_label,
        field_value=data.admin_experience_roles_duration
    )

    # Total career timeline (years)
    await _add_field_verification(
        evaluator,
        pos_node,
        field_node_id=f"{position_node_id.split('_')[0]}_total_career_timeline_years",
        field_desc="Provides the total minimum career timeline in years from entry-level to superintendent eligibility.",
        claim_text=f"According to the provided source(s), the total minimum career timeline from entry-level to eligibility for {position_label} is: {data.total_career_timeline_years.text if data.total_career_timeline_years and data.total_career_timeline_years.text else ''}.",
        position_label=position_label,
        field_value=data.total_career_timeline_years
    )

    # Salary range (2024–2025)
    await _add_field_verification(
        evaluator,
        pos_node,
        field_node_id=f"{position_node_id.split('_')[0]}_salary_range_2024_2025",
        field_desc="Provides a current typical salary range based on available public data from 2024–2025.",
        claim_text=f"According to the provided source(s), a typical current salary range (2024–2025) for {position_label} is: {data.salary_range_2024_2025.text if data.salary_range_2024_2025 and data.salary_range_2024_2025.text else ''}.",
        position_label=position_label,
        field_value=data.salary_range_2024_2025
    )


async def verify_athletic_director_position(
    evaluator: Evaluator,
    parent_node,
    *,
    position_node_id: str,
    position_label: str,
    data: Optional[AthleticDirectorRequirements]
) -> None:
    """
    Build verification sub-tree for a Division I FBS Athletic Director role.
    """
    pos_node = evaluator.add_parallel(
        id=position_node_id,
        desc=f"Minimum qualification requirements for {position_label}",
        parent=parent_node,
        critical=False
    )

    data = data or AthleticDirectorRequirements()

    # Degree level
    await _add_field_verification(
        evaluator,
        pos_node,
        field_node_id=f"{position_node_id.split('_')[0]}_education_degree_level",
        field_desc="States the minimum required educational degree level for the athletic director role.",
        claim_text=f"According to the provided source(s), the minimum required educational degree level for {position_label} is: {data.degree_level.text if data.degree_level and data.degree_level.text else ''}.",
        position_label=position_label,
        field_value=data.degree_level
    )

    # Degree field
    await _add_field_verification(
        evaluator,
        pos_node,
        field_node_id=f"{position_node_id.split('_')[0]}_education_field_of_study",
        field_desc="States the required/acceptable field(s) of study for the minimum degree (or explicitly notes if none specified).",
        claim_text=f"According to the provided source(s), the required/acceptable degree field(s) for {position_label} is/are: {data.degree_field.text if data.degree_field and data.degree_field.text else ''}.",
        position_label=position_label,
        field_value=data.degree_field
    )

    # Required licenses or certifications (if any)
    await _add_field_verification(
        evaluator,
        pos_node,
        field_node_id=f"{position_node_id.split('_')[0]}_required_licenses_or_certs",
        field_desc="Identifies any required Georgia state certifications or professional licenses for the position (or explicitly states none are specified).",
        claim_text=f"According to the provided source(s), the required licenses/certifications for {position_label} are: {data.required_licenses_or_certs.text if data.required_licenses_or_certs and data.required_licenses_or_certs.text else ''}.",
        position_label=position_label,
        field_value=data.required_licenses_or_certs
    )

    # NCAA compliance requirement
    await _add_field_verification(
        evaluator,
        pos_node,
        field_node_id=f"{position_node_id.split('_')[0]}_ncaa_compliance_requirement",
        field_desc="Addresses the NCAA Division I rules/compliance knowledge/experience requirement, if specified.",
        claim_text=f"According to the provided source(s), NCAA Division I compliance knowledge/experience requirements for {position_label} are: {data.ncaa_compliance_requirement.text if data.ncaa_compliance_requirement and data.ncaa_compliance_requirement.text else ''}.",
        position_label=position_label,
        field_value=data.ncaa_compliance_requirement
    )

    # Title IX requirement
    await _add_field_verification(
        evaluator,
        pos_node,
        field_node_id=f"{position_node_id.split('_')[0]}_title_ix_requirement",
        field_desc="Addresses the Title IX compliance knowledge/experience requirement, if specified.",
        claim_text=f"According to the provided source(s), Title IX compliance knowledge/experience requirements for {position_label} are: {data.title_ix_requirement.text if data.title_ix_requirement and data.title_ix_requirement.text else ''}.",
        position_label=position_label,
        field_value=data.title_ix_requirement
    )

    # Budget management requirement
    await _add_field_verification(
        evaluator,
        pos_node,
        field_node_id=f"{position_node_id.split('_')[0]}_budget_management_requirement",
        field_desc="Addresses required/typical budget management experience for Division I FBS athletic directors, if specified.",
        claim_text=f"According to the provided source(s), budget management experience requirements for {position_label} are: {data.budget_management_requirement.text if data.budget_management_requirement and data.budget_management_requirement.text else ''}.",
        position_label=position_label,
        field_value=data.budget_management_requirement
    )

    # Experience years: coaching or athletic administration
    await _add_field_verification(
        evaluator,
        pos_node,
        field_node_id=f"{position_node_id.split('_')[0]}_experience_years_coaching_or_admin",
        field_desc="Provides the minimum years of coaching and/or athletic administration experience typically required.",
        claim_text=f"According to the provided source(s), the minimum years of coaching and/or athletic administration experience for {position_label} is: {data.experience_years_coaching_or_admin.text if data.experience_years_coaching_or_admin and data.experience_years_coaching_or_admin.text else ''}.",
        position_label=position_label,
        field_value=data.experience_years_coaching_or_admin
    )

    # Administrative experience – roles and duration
    await _add_field_verification(
        evaluator,
        pos_node,
        field_node_id=f"{position_node_id.split('_')[0]}_admin_experience_roles_duration",
        field_desc="Provides minimum administrative/leadership experience requirements (types of athletic-department roles and duration).",
        claim_text=f"According to the provided source(s), the minimum administrative/leadership experience (roles and duration) for {position_label} is: {data.admin_experience_roles_duration.text if data.admin_experience_roles_duration and data.admin_experience_roles_duration.text else ''}.",
        position_label=position_label,
        field_value=data.admin_experience_roles_duration
    )

    # Total career timeline (years)
    await _add_field_verification(
        evaluator,
        pos_node,
        field_node_id=f"{position_node_id.split('_')[0]}_total_career_timeline_years",
        field_desc="Provides the total minimum career timeline in years from entry-level to Division I FBS AD eligibility.",
        claim_text=f"According to the provided source(s), the total minimum career timeline from entry-level to eligibility for {position_label} is: {data.total_career_timeline_years.text if data.total_career_timeline_years and data.total_career_timeline_years.text else ''}.",
        position_label=position_label,
        field_value=data.total_career_timeline_years
    )

    # Salary range (2024–2025)
    await _add_field_verification(
        evaluator,
        pos_node,
        field_node_id=f"{position_node_id.split('_')[0]}_salary_range_2024_2025",
        field_desc="Provides a current typical salary range based on available public data from 2024–2025.",
        claim_text=f"According to the provided source(s), a typical current salary range (2024–2025) for {position_label} is: {data.salary_range_2024_2025.text if data.salary_range_2024_2025 and data.salary_range_2024_2025.text else ''}.",
        position_label=position_label,
        field_value=data.salary_range_2024_2025
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
    Evaluate an answer for the Georgia education leadership minimum-qualification requirements task.
    """
    # 1) Initialize evaluator
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Positions evaluated independently
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

    # 2) Extract structured information from the answer
    extraction = await evaluator.extract(
        prompt=prompt_extract_requirements(),
        template_class=FullExtraction,
        extraction_name="structured_requirements_extraction"
    )

    # 3) Build a top-level organizer (kept non-critical to allow partial credit across positions)
    positions_parent = evaluator.add_parallel(
        id="all_positions_analysis",
        desc="Complete minimum-qualification requirements for all four specified senior education leadership positions in Georgia",
        parent=root,
        critical=False
    )

    # 4) Verify each position block in parallel (tree creation + URL-based checks)
    # Gwinnett County Schools Superintendent
    await verify_superintendent_position(
        evaluator,
        positions_parent,
        position_node_id="gwinnett_superintendent",
        position_label="Gwinnett County Schools – School Superintendent",
        data=extraction.gwinnett_superintendent
    )

    # Forsyth County Schools Superintendent
    await verify_superintendent_position(
        evaluator,
        positions_parent,
        position_node_id="forsyth_superintendent",
        position_label="Forsyth County Schools – School Superintendent",
        data=extraction.forsyth_superintendent
    )

    # Georgia State University Athletic Director
    await verify_athletic_director_position(
        evaluator,
        positions_parent,
        position_node_id="gsu_athletic_director",
        position_label="Georgia State University – Athletic Director (Division I FBS, Sun Belt)",
        data=extraction.gsu_athletic_director
    )

    # Georgia Southern University Athletic Director
    await verify_athletic_director_position(
        evaluator,
        positions_parent,
        position_node_id="gsouthern_athletic_director",
        position_label="Georgia Southern University – Athletic Director (Division I FBS, Sun Belt)",
        data=extraction.gsouthern_athletic_director
    )

    # 5) Return evaluation summary
    return evaluator.get_summary()