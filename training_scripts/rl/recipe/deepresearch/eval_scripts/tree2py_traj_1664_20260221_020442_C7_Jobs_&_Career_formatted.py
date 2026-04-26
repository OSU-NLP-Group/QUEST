import asyncio
import logging
from datetime import datetime
from typing import Any, List, Optional, Dict

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys

# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "oh_admin_jobs_4"
TASK_DESCRIPTION = (
    "Find four currently open school district administrative job positions in Ohio. For each position, provide:\n"
    "1) Job Title and Location; 2) Educational Requirements; 3) Experience Requirements; "
    "4) Application Information (deadline and contact email or apply link). "
    "All positions must be administrative (not classroom teaching) and must have future application deadlines."
)

CURRENT_DATE_ISO = "2026-02-22"


# --------------------------------------------------------------------------- #
# Data models for structured extraction                                       #
# --------------------------------------------------------------------------- #
class EducationRequirements(BaseModel):
    degree_level: Optional[str] = None  # e.g., "bachelor's", "master's", "doctoral"
    field: Optional[str] = None         # e.g., "education administration", "curriculum & instruction"
    not_specified: Optional[bool] = False  # true only if explicitly stated or clearly no minimum specified


class ExperienceRequirements(BaseModel):
    min_years: Optional[str] = None       # e.g., "3", "3+"
    experience_type: Optional[str] = None # e.g., "administrative", "teaching", "coaching", "leadership"
    not_specified: Optional[bool] = False


class CertificationsInfo(BaseModel):
    certifications: List[str] = Field(default_factory=list)  # e.g., ["Ohio Principal License"]
    not_specified: Optional[bool] = False


class ApplicationInfo(BaseModel):
    deadline: Optional[str] = None        # Prefer ISO format YYYY-MM-DD; else raw text
    contact_email: Optional[str] = None
    apply_link: Optional[str] = None
    instructions_text: Optional[str] = None  # optional free text from answer


class PositionInfo(BaseModel):
    job_title: Optional[str] = None
    district: Optional[str] = None
    location_city_or_county: Optional[str] = None
    location_state: Optional[str] = None  # e.g., "Ohio" or "OH"
    education: EducationRequirements = Field(default_factory=EducationRequirements)
    experience: ExperienceRequirements = Field(default_factory=ExperienceRequirements)
    certifications: CertificationsInfo = Field(default_factory=CertificationsInfo)
    application: ApplicationInfo = Field(default_factory=ApplicationInfo)
    source_urls: List[str] = Field(default_factory=list)


class PositionsExtraction(BaseModel):
    positions: List[PositionInfo] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_positions() -> str:
    return (
        "Extract up to four school district administrative job positions mentioned in the answer. "
        "Return an object with a 'positions' array (max length 4). For each position, extract exactly the following fields:\n"
        "- job_title: the exact job title string as stated.\n"
        "- district: the name of the school district.\n"
        "- location_city_or_county: city or county if provided; else null.\n"
        "- location_state: state string (e.g., 'Ohio', 'OH') if present; else null.\n"
        "- education: an object with:\n"
        "    • degree_level: one of \"bachelor's\", \"master's\", \"doctoral\" if explicitly required; else null.\n"
        "    • field: field of study if specified (e.g., 'education administration'); else null.\n"
        "    • not_specified: true ONLY if the answer explicitly indicates no minimum education requirement is specified.\n"
        "- experience: an object with:\n"
        "    • min_years: minimum years required as a string (e.g., '3', '3+'); else null.\n"
        "    • experience_type: the type (e.g., 'administrative', 'teaching', 'leadership'); else null.\n"
        "    • not_specified: true ONLY if the answer explicitly indicates no specific experience requirement is stated.\n"
        "- certifications: an object with:\n"
        "    • certifications: list of required certifications/licenses explicitly named; empty if none listed.\n"
        "    • not_specified: true ONLY if the answer explicitly indicates no certification/license requirement is listed.\n"
        "- application: an object with:\n"
        "    • deadline: application deadline string (prefer ISO YYYY-MM-DD if answer provides a clear date; else use raw text like 'Open until filled').\n"
        "    • contact_email: contact email if provided; else null.\n"
        "    • apply_link: direct link to apply if provided; else null.\n"
        "    • instructions_text: any brief application instruction text if present; else null.\n"
        "- source_urls: list of ALL URLs that the answer cites for this position (job board postings, district HR pages, etc.). "
        "Include valid URLs only; if none are provided in the answer, return an empty list.\n\n"
        "Rules:\n"
        "1) Extract ONLY what appears in the answer; do not invent or infer.\n"
        "2) If the answer provides more than four positions, include ONLY the first four in the array, preserving order. "
        "If fewer than four, include what is available.\n"
        "3) If any field is missing in the answer, return null (or empty list for array fields).\n"
        "4) Do not normalize or reinterpret requirements (e.g., 'preferred' is not a 'required' minimum).\n"
    )


# --------------------------------------------------------------------------- #
# Helper to build position node descriptions                                  #
# --------------------------------------------------------------------------- #
def position_node_desc(idx: int) -> str:
    labels = ["First", "Second", "Third", "Fourth"]
    return f"{labels[idx]} administrative position with complete required information"


# --------------------------------------------------------------------------- #
# Verification per position                                                   #
# --------------------------------------------------------------------------- #
async def verify_single_position(
    evaluator: Evaluator,
    parent_node,
    position: PositionInfo,
    idx: int,
) -> None:
    """
    Build verification nodes and run checks for a single position.
    """
    # Create the position node (non-critical, parallel aggregation)
    pos_node = evaluator.add_parallel(
        id=f"position_{idx+1}",
        desc=position_node_desc(idx),
        parent=parent_node,
        critical=False,
    )

    # Precondition: basic existence and sources
    has_basic = bool(position.job_title and position.district)
    has_sources = bool(position.source_urls and len(position.source_urls) > 0)

    evaluator.add_custom_node(
        result=(has_basic and has_sources),
        id=f"position_{idx+1}_sources_available",
        desc="Basic fields present (job_title and district) AND at least one source URL provided",
        parent=pos_node,
        critical=True,
    )

    # 1) Job Title & Location (Administrative + Ohio)
    job_node = evaluator.add_leaf(
        id=f"position_{idx+1}_job_title_location",
        desc="Job title clearly stated as an administrative position (not classroom teaching) and school district location in Ohio identified",
        parent=pos_node,
        critical=True,
    )

    job_title = position.job_title or ""
    district = position.district or ""
    job_claim = (
        f"The job posting describes an administrative position titled '{job_title}' at the school district '{district}' in Ohio (OH), "
        f"and it is NOT a classroom teaching position."
    )
    job_add_ins = (
        "Administrative roles include principal, assistant principal, athletic director, curriculum director, special education director, "
        "superintendent, or other district-level administrative positions. Confirm the district is in Ohio by matching 'Ohio' or 'OH' on the page. "
        "If the posting is for a teacher (classroom teaching), the claim is incorrect."
    )

    # 2) Educational Requirements
    edu_node = evaluator.add_leaf(
        id=f"position_{idx+1}_educational_requirements",
        desc="Minimum educational requirements (degree level and field) specified or explicitly stated as not specified",
        parent=pos_node,
        critical=True,
    )

    if position.education and position.education.not_specified:
        edu_claim = "The job posting does not specify any minimum educational requirement (degree level or field)."
        edu_add_ins = (
            "Check whether the posting explicitly lists a minimum degree requirement. If the posting only mentions 'preferred' degrees "
            "and no explicit minimum required degree, treat the claim as correct."
        )
    else:
        deg = position.education.degree_level or ""
        fld = position.education.field or ""
        if deg and fld:
            edu_claim = f"The job posting specifies a minimum education requirement of a {deg} degree in {fld} (or closely related field)."
        elif deg and not fld:
            edu_claim = f"The job posting specifies a minimum education requirement of a {deg} degree."
        else:
            edu_claim = "The job posting does not specify any minimum educational requirement (degree level or field)."
        edu_add_ins = (
            "Focus on explicit 'required' language. If only 'preferred' is mentioned without a minimum requirement, consider 'not specified' as correct."
        )

    # 3) Experience Requirements
    exp_node = evaluator.add_leaf(
        id=f"position_{idx+1}_experience_requirements",
        desc="Minimum years and type of experience required specified, or explicitly stated as no specific experience requirement",
        parent=pos_node,
        critical=True,
    )

    if position.experience and position.experience.not_specified:
        exp_claim = "The job posting does not state any specific minimum years or type of experience requirement."
        exp_add_ins = (
            "If the posting only indicates 'experience preferred' without a clear minimum years/type, treat this claim as correct."
        )
    else:
        yrs = position.experience.min_years or ""
        typ = position.experience.experience_type or ""
        if yrs and typ:
            exp_claim = f"The job posting requires at least {yrs} years of {typ} experience."
        elif yrs and not typ:
            exp_claim = f"The job posting requires at least {yrs} years of relevant experience."
        elif (not yrs) and typ:
            exp_claim = f"The job posting requires {typ} experience (minimum years not specified)."
        else:
            exp_claim = "The job posting does not state any specific minimum years or type of experience requirement."
        exp_add_ins = (
            "Verify that the posting explicitly lists minimum years and/or an experience type. Ignore 'preferred' statements if not required."
        )

    # 4) Certifications/Licenses
    cert_node = evaluator.add_leaf(
        id=f"position_{idx+1}_certification_license",
        desc="Required certifications or licenses specified if any are required by the posting",
        parent=pos_node,
        critical=True,
    )

    if position.certifications and position.certifications.certifications:
        cert_list = ", ".join(position.certifications.certifications)
        cert_claim = f"The job posting requires the following certification(s) or license(s): {cert_list}."
        cert_add_ins = (
            "Confirm explicit 'required' credentials (e.g., Ohio Principal License, Superintendent License, Administrative License). "
            "If the posting mentions only 'preferred/recommended' credentials, do NOT treat them as required."
        )
    else:
        cert_claim = "The job posting does not list any specific required certification or license."
        cert_add_ins = (
            "If the posting does not explicitly require a certification/license, this claim is correct. Ignore 'preferred' credentials."
        )

    # 5) Application Info (deadline future + contact/apply link)
    app_node = evaluator.add_leaf(
        id=f"position_{idx+1}_application_contact",
        desc="Application instructions or contact information provided with application deadline that is in the future (not expired)",
        parent=pos_node,
        critical=True,
    )

    deadline = position.application.deadline or ""
    email = position.application.contact_email or ""
    apply_link = position.application.apply_link or ""
    app_claim = (
        f"The job posting includes application instructions or contact information (email '{email}' or apply link '{apply_link}'), "
        f"and the application deadline is '{deadline}', which is after {CURRENT_DATE_ISO}."
    )
    app_add_ins = (
        f"Today's date is {CURRENT_DATE_ISO}. If the posting states 'Open until filled' or similar, treat it as a future/unexpired deadline. "
        "Confirm that at least one of: contact email or apply link is present on the page."
    )

    # Build claims and nodes for batch verification
    claims_and_sources = [
        (job_claim, position.source_urls, job_node, job_add_ins),
        (edu_claim, position.source_urls, edu_node, edu_add_ins),
        (exp_claim, position.source_urls, exp_node, exp_add_ins),
        (cert_claim, position.source_urls, cert_node, cert_add_ins),
        (app_claim, position.source_urls, app_node, app_add_ins),
    ]

    # Run verifications in parallel under this position node
    await evaluator.batch_verify(claims_and_sources)


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
    Evaluate an agent's answer for four Ohio school district administrative job positions.
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

    # Record current date for reference in the summary
    evaluator.add_custom_info(
        {"current_date_iso": CURRENT_DATE_ISO},
        info_type="context",
        info_name="current_date",
    )

    # Extract positions from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_positions(),
        template_class=PositionsExtraction,
        extraction_name="positions_extraction",
    )

    # Ensure we have exactly four positions (pad with empty placeholders if needed)
    positions = list(extracted.positions[:4]) if extracted.positions else []
    while len(positions) < 4:
        positions.append(PositionInfo())

    # Build verification tree for each position
    for i in range(4):
        await verify_single_position(evaluator, root, positions[i], i)

    # Return the evaluation summary with the verification tree
    return evaluator.get_summary()