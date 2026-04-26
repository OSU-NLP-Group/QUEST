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
TASK_ID = "wcpss_superintendent_qualifications_march_2026"
TASK_DESCRIPTION = (
    "As of March 2026, who is the current superintendent of Wake County Public School System in North Carolina, "
    "and does this individual meet all of the following standard qualifications for a North Carolina school district "
    "superintendent position: (1) Holds a doctoral degree (EdD or PhD) in educational leadership, educational "
    "administration, or a closely related field; (2) Holds or has held an active North Carolina Principal License; "
    "(3) Has served as a principal in at least one North Carolina public school; (4) Has at least 3 years of "
    "cumulative experience in school administrative positions (such as principal, assistant principal, or equivalent roles). "
    "Provide the superintendent's full name, and for each qualification requirement, specify whether it is met and provide "
    "supporting details along with reference URLs that verify each credential."
)

AS_OF_TIMEFRAME = "March 2026"


# --------------------------------------------------------------------------- #
# Data models for structured extraction                                       #
# --------------------------------------------------------------------------- #
class DegreeDetails(BaseModel):
    degree_type: Optional[str] = None  # e.g., "EdD", "PhD"
    field_of_study: Optional[str] = None  # e.g., "Educational Leadership"
    institution: Optional[str] = None  # e.g., "University of North Carolina at Chapel Hill"
    year_awarded: Optional[str] = None  # free-form year or date (string)
    sources: List[str] = Field(default_factory=list)


class LicenseInfo(BaseModel):
    license_status: Optional[str] = None  # e.g., "active", "previously held", "expired", "unknown"
    license_details: Optional[str] = None  # any clarifying text (e.g., license number if given)
    sources: List[str] = Field(default_factory=list)


class PrincipalService(BaseModel):
    school_name: Optional[str] = None  # e.g., "Example High School"
    district: Optional[str] = None  # e.g., "Wake County Public School System"
    city_or_location: Optional[str] = None  # e.g., "Raleigh, NC"
    years_of_service: Optional[str] = None  # e.g., "2016–2019" or "circa 2015–2018"


class PrincipalExperience(BaseModel):
    experiences: List[PrincipalService] = Field(default_factory=list)  # At least one NC public school principal role
    sources: List[str] = Field(default_factory=list)


class AdminPosition(BaseModel):
    role_title: Optional[str] = None  # e.g., "Assistant Principal", "Principal", "Executive Director"
    organization: Optional[str] = None  # e.g., school name or district/department
    timespan_or_duration: Optional[str] = None  # e.g., "2019–2023" or "~4 years"


class AdministrativeExperience(BaseModel):
    positions: List[AdminPosition] = Field(default_factory=list)
    total_years_estimate: Optional[str] = None  # free-form text, e.g., "at least 3 years", "~5+ years"
    sources: List[str] = Field(default_factory=list)


class SuperintendentQualificationExtraction(BaseModel):
    superintendent_name: Optional[str] = None
    official_title: Optional[str] = None  # as written in the answer (e.g., "Superintendent, Wake County Public School System")
    identity_sources: List[str] = Field(default_factory=list)  # URLs verifying identity and current role

    doctoral_degree: Optional[DegreeDetails] = None
    nc_principal_license: Optional[LicenseInfo] = None
    principal_experience: Optional[PrincipalExperience] = None
    administrative_experience: Optional[AdministrativeExperience] = None


# --------------------------------------------------------------------------- #
# Extraction prompt helpers                                                   #
# --------------------------------------------------------------------------- #
def prompt_extract_superintendent_qualifications() -> str:
    return """
    Extract the following structured information exactly as presented in the answer text. Do not infer or invent details. 
    When URLs are requested, only include explicit URLs mentioned in the answer; ignore named references without links.

    Fields to extract:
    1) superintendent_name: The full name of the identified superintendent of the Wake County Public School System.
    2) official_title: The official title as stated (e.g., "Superintendent" or "Superintendent, Wake County Public School System").
    3) identity_sources: An array of URL strings that verify the identity and current employment as superintendent (e.g., WCPSS official pages, press releases, or credible news articles). If none are provided in the answer, return an empty array.

    4) doctoral_degree: 
       - degree_type: "EdD" or "PhD" (or other doctoral designation) as written.
       - field_of_study: The degree field (e.g., "Educational Leadership", "Educational Administration", or clearly related).
       - institution: The granting institution.
       - year_awarded: As provided (if available).
       - sources: Array of URLs that verify this doctoral credential. If none provided, return an empty array.

    5) nc_principal_license:
       - license_status: The status as stated in the answer (e.g., "active", "previously held", "expired", "unknown").
       - license_details: Any textual detail such as license number, issue date, or notes (if provided).
       - sources: Array of URLs that verify NC principal licensure status or history. If none provided, return an empty array.

    6) principal_experience:
       - experiences: An array; each object includes:
           * school_name: Name of at least one North Carolina public school where the person served as principal.
           * district: If provided (e.g., "Wake County Public School System").
           * city_or_location: If provided (e.g., "Raleigh, NC").
           * years_of_service: Approximate years of service (free text if needed), as provided in the answer.
       - sources: Array of URLs that document the principal service specifically. If none provided, return an empty array.

    7) administrative_experience:
       - positions: An array; each object includes:
           * role_title: Title of the administrative position (e.g., "Principal", "Assistant Principal", "Executive Director").
           * organization: The school or district/department.
           * timespan_or_duration: The years or approximate duration as provided in the answer.
       - total_years_estimate: A free-text estimate of total administrative experience years as claimed in the answer (e.g., "at least 3 years", "~5+ years").
       - sources: Array of URLs that document the administrative experience and timeframes. If none provided, return an empty array.

    Return a single JSON object matching the specified schema. 
    - For any missing field, use null for strings/objects and an empty array for URL lists.
    - Do not add fields not specified.
    """


# --------------------------------------------------------------------------- #
# Utility helpers                                                             #
# --------------------------------------------------------------------------- #
def _first_nonempty(items: List[Optional[str]]) -> Optional[str]:
    for x in items:
        if x and str(x).strip():
            return x.strip()
    return None


def _format_principal_experience_for_claim(px: Optional[PrincipalExperience]) -> str:
    if not px or not px.experiences:
        return "no specific principal service details provided"
    exp = None
    for e in px.experiences:
        if (e.school_name and e.school_name.strip()):
            exp = e
            break
    if exp is None:
        exp = px.experiences[0]
    school = exp.school_name or "an NC public school"
    district = exp.district or ""
    city = exp.city_or_location or ""
    years = exp.years_of_service or "unspecified years"
    extras = []
    if district:
        extras.append(district)
    if city:
        extras.append(city)
    extra_txt = f" ({', '.join(extras)})" if extras else ""
    return f"{school}{extra_txt} around {years}"


def _format_admin_positions_for_claim(ax: Optional[AdministrativeExperience]) -> str:
    if not ax or not ax.positions:
        return "no detailed positions provided"
    parts = []
    for p in ax.positions[:5]:
        role = p.role_title or "administrative role"
        org = p.organization or "an NC school/district"
        span = p.timespan_or_duration or "unspecified duration"
        parts.append(f"{role} at {org} ({span})")
    return "; ".join(parts)


# --------------------------------------------------------------------------- #
# Verification subtree builders                                               #
# --------------------------------------------------------------------------- #
async def build_identity_verification(
    evaluator: Evaluator,
    parent_node,
    data: SuperintendentQualificationExtraction
) -> None:
    """
    Superintendent_Identity_Confirmation (parallel, critical)
      - (custom critical) Identity_References_Provided  [gates the following verifications]
      - (leaf critical) Superintendent_Name_and_Title
      - (leaf critical) Current_Position_Verification
    """
    identity_node = evaluator.add_parallel(
        id="Superintendent_Identity_Confirmation",
        desc="Identify the current superintendent of Wake County Public School System as of March 2026",
        parent=parent_node,
        critical=True,
    )

    # Gate: require at least one identity source URL
    identity_sources = data.identity_sources or []
    evaluator.add_custom_node(
        result=bool(identity_sources),
        id="Identity_References_Provided",
        desc="At least one URL is provided to verify the superintendent's identity and role",
        parent=identity_node,
        critical=True
    )

    # Leaf 1: Superintendent_Name_and_Title
    name_title_leaf = evaluator.add_leaf(
        id="Superintendent_Name_and_Title",
        desc="Correctly identify the superintendent by full name and official title",
        parent=identity_node,
        critical=True
    )
    name = data.superintendent_name or "[NAME NOT PROVIDED]"
    title = data.official_title or "Superintendent"
    claim_nt = (
        f"The full name of the superintendent of the Wake County Public School System is '{name}', "
        f"and the individual holds the title of Superintendent (or equivalent) of the district."
    )
    await evaluator.verify(
        claim=claim_nt,
        node=name_title_leaf,
        sources=identity_sources,
        additional_instruction=(
            "Verify that the provided URL(s) clearly show the individual's name and identify them as the "
            "Superintendent of the Wake County Public School System (WCPSS). Accept reasonable title variants "
            "that clearly mean Superintendent of WCPSS. If the URLs do not confirm both name and superintendent title, "
            "mark as Incorrect."
        ),
    )

    # Leaf 2: Current_Position_Verification
    current_pos_leaf = evaluator.add_leaf(
        id="Current_Position_Verification",
        desc="Confirm the individual's current employment status as superintendent of Wake County Public School System as of March 2026",
        parent=identity_node,
        critical=True
    )
    claim_current = (
        f"As of {AS_OF_TIMEFRAME}, {name} is serving as the Superintendent of the Wake County Public School System."
    )
    await evaluator.verify(
        claim=claim_current,
        node=current_pos_leaf,
        sources=identity_sources,
        additional_instruction=(
            "From the provided URL(s), determine whether the individual is the CURRENT superintendent as of March 2026. "
            "Use page timestamps, publication dates, or explicit present-tense statements indicating the person is the current "
            "superintendent. If a source indicates the tenure ended before March 2026 or names a different person as current "
            "superintendent, mark as Incorrect. If no clear timing evidence is shown but the official WCPSS page lists them "
            "as superintendent with no end date, treat as correct."
        ),
    )


async def build_doctoral_degree_verification(
    evaluator: Evaluator,
    parent_node,
    data: SuperintendentQualificationExtraction
) -> None:
    """
    Doctoral_Degree_Verification (parallel, critical)
      - (custom critical) Doctoral_Degree_Reference  [checks URL presence]
      - (leaf critical) Doctoral_Degree_Details      [URL-backed fact check]
    """
    node = evaluator.add_parallel(
        id="Doctoral_Degree_Verification",
        desc="Verify the superintendent holds a doctoral degree (EdD or PhD) in educational leadership/administration or closely related field",
        parent=parent_node,
        critical=True
    )

    dd = data.doctoral_degree or DegreeDetails()
    doc_sources = dd.sources or []

    # Reference presence (critical)
    evaluator.add_custom_node(
        result=bool(doc_sources),
        id="Doctoral_Degree_Reference",
        desc="Provide a reference URL verifying the superintendent's doctoral degree credential",
        parent=node,
        critical=True
    )

    # Details verification (critical)
    details_leaf = evaluator.add_leaf(
        id="Doctoral_Degree_Details",
        desc="Provide the specific doctoral degree type (EdD or PhD), the granting institution, and the field of study",
        parent=node,
        critical=True
    )
    degree_type = dd.degree_type or "a doctoral degree"
    field = dd.field_of_study or "a field closely related to educational leadership or educational administration"
    institution = dd.institution or "an accredited institution"
    name = data.superintendent_name or "[NAME NOT PROVIDED]"

    claim = (
        f"{name} holds {degree_type} in {field} from {institution}."
    )
    await evaluator.verify(
        claim=claim,
        node=details_leaf,
        sources=doc_sources,
        additional_instruction=(
            "Confirm that the URL(s) explicitly state a doctoral degree (EdD or PhD) and identify the field and institution. "
            "Allow reasonable equivalents for field names closely related to educational leadership/administration. "
            "If only a master's degree is shown, or the doctorate is in an unrelated field with no indication of 'closely related', "
            "mark as Incorrect."
        ),
    )


async def build_nc_principal_license_verification(
    evaluator: Evaluator,
    parent_node,
    data: SuperintendentQualificationExtraction
) -> None:
    """
    NC_Principal_License_Verification (parallel, critical)
      - (custom critical) License_Reference
      - (leaf critical) License_Status_Confirmation
    """
    node = evaluator.add_parallel(
        id="NC_Principal_License_Verification",
        desc="Verify the superintendent holds or has held an active North Carolina Principal License",
        parent=parent_node,
        critical=True
    )

    lic = data.nc_principal_license or LicenseInfo()
    lic_sources = lic.sources or []

    # Reference presence (critical)
    evaluator.add_custom_node(
        result=bool(lic_sources),
        id="License_Reference",
        desc="Provide a reference URL verifying the superintendent's NC licensure status or history",
        parent=node,
        critical=True
    )

    # License status confirmation (critical)
    status_leaf = evaluator.add_leaf(
        id="License_Status_Confirmation",
        desc="Confirm the existence and status of the superintendent's North Carolina Principal License",
        parent=node,
        critical=True
    )
    name = data.superintendent_name or "[NAME NOT PROVIDED]"
    claim = (
        f"{name} holds or has previously held a North Carolina Principal License (it may be currently active or previously active)."
    )
    await evaluator.verify(
        claim=claim,
        node=status_leaf,
        sources=lic_sources,
        additional_instruction=(
            "Confirm from the URL(s) that the person either currently holds or has held a North Carolina Principal License. "
            "Accept evidence from official state licensure lookup, district bios, resumes, or credible sources explicitly mentioning NC principal licensure. "
            "If the links do not explicitly confirm NC principal licensure, mark as Incorrect."
        ),
    )


async def build_principal_experience_verification(
    evaluator: Evaluator,
    parent_node,
    data: SuperintendentQualificationExtraction
) -> None:
    """
    NC_Principal_Experience_Verification (parallel, critical)
      - (custom critical) Principal_Experience_Reference
      - (leaf critical) Principal_Service_Details
    """
    node = evaluator.add_parallel(
        id="NC_Principal_Experience_Verification",
        desc="Verify the superintendent has served as a principal in at least one North Carolina public school",
        parent=parent_node,
        critical=True
    )

    px = data.principal_experience or PrincipalExperience()
    px_sources = px.sources or []

    # Reference presence (critical)
    evaluator.add_custom_node(
        result=bool(px_sources),
        id="Principal_Experience_Reference",
        desc="Provide a reference URL documenting the superintendent's service as a principal in NC public schools",
        parent=node,
        critical=True
    )

    # Principal service details (critical)
    details_leaf = evaluator.add_leaf(
        id="Principal_Service_Details",
        desc="Identify at least one specific North Carolina public school where the superintendent served as principal, including approximate years of service",
        parent=node,
        critical=True
    )
    name = data.superintendent_name or "[NAME NOT PROVIDED]"
    exp_text = _format_principal_experience_for_claim(px)
    claim = (
        f"{name} served as a principal at {exp_text} in North Carolina."
    )
    await evaluator.verify(
        claim=claim,
        node=details_leaf,
        sources=px_sources,
        additional_instruction=(
            "From the URL(s), confirm at least one principal role in a North Carolina PUBLIC school and include approximate years if provided. "
            "If the pages do not clearly indicate principal service in an NC public school, mark as Incorrect."
        ),
    )


async def build_admin_experience_verification(
    evaluator: Evaluator,
    parent_node,
    data: SuperintendentQualificationExtraction
) -> None:
    """
    Administrative_Experience_Verification (parallel, critical)
      - (custom critical) Experience_Reference
      - (leaf critical) Experience_Summary
    """
    node = evaluator.add_parallel(
        id="Administrative_Experience_Verification",
        desc="Verify the superintendent has at least 3 years of cumulative experience in school administrative positions",
        parent=parent_node,
        critical=True
    )

    ax = data.administrative_experience or AdministrativeExperience()
    ax_sources = ax.sources or []

    # Reference presence (critical)
    evaluator.add_custom_node(
        result=bool(ax_sources),
        id="Experience_Reference",
        desc="Provide a reference URL documenting the superintendent's administrative experience history",
        parent=node,
        critical=True
    )

    # Experience summary (critical)
    summary_leaf = evaluator.add_leaf(
        id="Experience_Summary",
        desc="Provide a summary documenting the superintendent's administrative positions and demonstrating at least 3 cumulative years of experience",
        parent=node,
        critical=True
    )
    name = data.superintendent_name or "[NAME NOT PROVIDED]"
    positions_text = _format_admin_positions_for_claim(ax)
    total_years_txt = ax.total_years_estimate or "at least 3 years"
    claim = (
        f"{name} has {total_years_txt} of cumulative school administrative experience across the following roles: {positions_text}."
    )
    await evaluator.verify(
        claim=claim,
        node=summary_leaf,
        sources=ax_sources,
        additional_instruction=(
            "Using the provided URL(s), verify that the listed administrative roles and their durations cumulatively meet or exceed 3 years. "
            "If explicit durations are not provided but the dates imply at least 3 years, that is acceptable. "
            "If the evidence clearly totals less than 3 years, mark as Incorrect."
        ),
    )


async def build_qualifications_block(
    evaluator: Evaluator,
    parent_node,
    data: SuperintendentQualificationExtraction
) -> None:
    """
    All_Qualifications_Verification (parallel, critical)
      - Doctoral_Degree_Verification (parallel, critical)
      - NC_Principal_License_Verification (parallel, critical)
      - NC_Principal_Experience_Verification (parallel, critical)
      - Administrative_Experience_Verification (parallel, critical)
    """
    quals_node = evaluator.add_parallel(
        id="All_Qualifications_Verification",
        desc="Verify the identified superintendent meets all four standard qualification requirements for a North Carolina school district superintendent position",
        parent=parent_node,
        critical=True
    )

    # Build each qualification subtree
    await build_doctoral_degree_verification(evaluator, quals_node, data)
    await build_nc_principal_license_verification(evaluator, quals_node, data)
    await build_principal_experience_verification(evaluator, quals_node, data)
    await build_admin_experience_verification(evaluator, quals_node, data)


# --------------------------------------------------------------------------- #
# Main evaluation function                                                    #
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
    Entry point for evaluating an answer for the WCPSS superintendent qualifications task.
    """
    # Initialize evaluator and root
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.SEQUENTIAL,  # Root-level aggregation per rubric (sequential)
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

    # Create top-level critical sequential node (since initialize root is always non-critical)
    main_node = evaluator.add_sequential(
        id="Superintendent_Qualification_Complete_Verification",
        desc="Complete verification of the Wake County Public School System superintendent's identity and qualifications as of March 2026",
        parent=root,
        critical=True
    )

    # Extract structured data
    extracted = await evaluator.extract(
        prompt=prompt_extract_superintendent_qualifications(),
        template_class=SuperintendentQualificationExtraction,
        extraction_name="superintendent_qualifications_extraction"
    )

    # Build identity verification (first in sequence)
    await build_identity_verification(evaluator, main_node, extracted)

    # Build all-qualifications verification (second in sequence)
    await build_qualifications_block(evaluator, main_node, extracted)

    # Return standardized summary
    return evaluator.get_summary()