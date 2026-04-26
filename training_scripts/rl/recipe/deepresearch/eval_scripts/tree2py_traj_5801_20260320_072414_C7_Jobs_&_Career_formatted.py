import asyncio
import logging
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "superintendent_2024_cert_req"
TASK_DESCRIPTION = """
Identify a U.S. public school district where a new superintendent was appointed during 2024. The district must be located in a state where the superintendent certification or licensure requirements include: (1) a master's degree as the minimum educational qualification, (2) completion of graduate-level coursework in educational administration or leadership, (3) prior teaching experience, and (4) a minimum specified amount of prior administrative or supervisory experience. Provide the following information: the name of the school district, the superintendent's name, the superintendent's documented start date, the employment contract duration, and the state's specific minimum administrative experience requirement (stated in years or months) for superintendent certification.
"""


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class DistrictSubmission(BaseModel):
    district_name: Optional[str] = None
    state: Optional[str] = None  # Full name preferred; abbreviation acceptable
    superintendent_name: Optional[str] = None

    # Appointment and employment info
    appointment_year: Optional[str] = None  # e.g., "2024"
    start_date: Optional[str] = None        # e.g., "July 1, 2024"
    contract_duration: Optional[str] = None # e.g., "3-year", "36 months", "through June 30, 2027"

    # State certification requirement specific numeric value
    admin_experience_requirement_value: Optional[str] = None  # e.g., "3 years", "24 months"

    # Source URLs explicitly mentioned in the answer
    district_urls: List[str] = Field(default_factory=list)        # district homepage, about page, board listing, etc.
    appointment_urls: List[str] = Field(default_factory=list)     # press releases, board minutes, news on appointment
    contract_urls: List[str] = Field(default_factory=list)        # contract document, board agenda/minutes with contract
    certification_urls: List[str] = Field(default_factory=list)   # state DOE/licensing or administrative code pages
    cert_status_urls: List[str] = Field(default_factory=list)     # sources attesting superintendent holds/is eligible


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_submission() -> str:
    return """
    Extract the single U.S. public school district and superintendent information presented in the answer.

    Return a JSON object with the following fields:
    - district_name: The school district name exactly as stated.
    - state: The U.S. state of the district (prefer full state name if available; abbreviation is acceptable).
    - superintendent_name: The superintendent's full name as provided.
    - appointment_year: The calendar year in which the superintendent was appointed/named/hired/approved (e.g., "2024"). If not explicitly stated, set null.
    - start_date: The documented start/effective date for the superintendent (e.g., "July 1, 2024"). If not provided, set null.
    - contract_duration: The specified employment contract duration/term (e.g., "3-year term", "36 months", "through June 30, 2027"). If not provided, set null.
    - admin_experience_requirement_value: The state's specific minimum prior administrative or supervisory experience requirement for superintendent certification, explicitly stated in years or months (e.g., "3 years", "24 months"). If not provided, set null.

    Also extract URL sources explicitly mentioned in the answer. Only include valid URLs actually present in the answer:
    - district_urls: URLs to the district's official pages (homepage, board pages, etc.) that help verify the district's identity/location/public status.
    - appointment_urls: URLs (press releases, board minutes, reputable local news, district pages) that document the superintendent's 2024 appointment and/or the start date.
    - contract_urls: URLs (contract PDF, board agenda/minutes, district docs) that specify the employment contract duration/term.
    - certification_urls: URLs from the state's department of education/licensing authority/administrative code/official program pages that document superintendent certification or licensure requirements for the specified state.
    - cert_status_urls: URLs that explicitly state the superintendent holds or is eligible for the state's required certification/licensure (press release, resume/CV posted by district, board docs, or state database, etc.).

    Rules:
    - Do not invent URLs; include only URLs explicitly present in the answer (plain links or markdown links). 
    - If a field is not mentioned, set it to null (or empty list for URL arrays).
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _has_text(s: Optional[str]) -> bool:
    return s is not None and isinstance(s, str) and s.strip() != ""


def _dedup_urls(*url_lists: List[str]) -> List[str]:
    seen = set()
    result: List[str] = []
    for lst in url_lists:
        for u in lst or []:
            if isinstance(u, str):
                uu = u.strip()
                if uu and uu not in seen:
                    seen.add(uu)
                    result.append(uu)
    return result


# --------------------------------------------------------------------------- #
# Verification logic                                                          #
# --------------------------------------------------------------------------- #
async def verify_submission(evaluator: Evaluator, root, sub: DistrictSubmission) -> None:
    # Presence checks (from answer text)
    evaluator.add_custom_node(
        result=_has_text(sub.district_name),
        id="District_Name_Provided",
        desc="The answer provides the name of the school district",
        parent=root,
        critical=True
    )

    evaluator.add_custom_node(
        result=_has_text(sub.superintendent_name),
        id="Superintendent_Name_Provided",
        desc="The answer provides the superintendent's name",
        parent=root,
        critical=True
    )

    evaluator.add_custom_node(
        result=_has_text(sub.admin_experience_requirement_value),
        id="Specific_Admin_Experience_Value_Provided",
        desc="The answer provides the state's specific minimum administrative experience requirement stated in years or months",
        parent=root,
        critical=True
    )

    # Common URL pools
    district_related_urls = _dedup_urls(sub.district_urls, sub.appointment_urls, sub.contract_urls)
    appointment_related_urls = _dedup_urls(sub.appointment_urls, sub.district_urls)
    startdate_related_urls = _dedup_urls(sub.appointment_urls, sub.contract_urls, sub.district_urls)
    contract_related_urls = _dedup_urls(sub.contract_urls, sub.appointment_urls)
    certification_urls = _dedup_urls(sub.certification_urls)
    cert_status_urls = _dedup_urls(sub.cert_status_urls, sub.appointment_urls, sub.district_urls)

    # Create verification leaves
    leaves_and_payloads: List[tuple] = []

    # US_State_Location
    node_state_loc = evaluator.add_leaf(
        id="US_State_Location",
        desc="The district is located within a U.S. state",
        parent=root,
        critical=True
    )
    claim_state_loc = f"The school district '{sub.district_name or ''}' is located in the U.S. state of '{sub.state or ''}'."
    add_ins_state_loc = "Verify using official district pages or authoritative sources on the provided URLs. Minor naming or abbreviation differences are acceptable."
    leaves_and_payloads.append((claim_state_loc, district_related_urls, node_state_loc, add_ins_state_loc))

    # Public_School_District
    node_public = evaluator.add_leaf(
        id="Public_School_District",
        desc="The district operates as a public school system",
        parent=root,
        critical=True
    )
    claim_public = f"'{sub.district_name or ''}' is a public K-12 school district (i.e., a public school system)."
    add_ins_public = "Confirm that the entity is a public school district (not a solely private/independent/charter company). Use district or state pages."
    leaves_and_payloads.append((claim_public, district_related_urls, node_public, add_ins_public))

    # State_Certification_Exists
    node_cert_exists = evaluator.add_leaf(
        id="State_Certification_Exists",
        desc="The state has documented superintendent certification or licensure requirements",
        parent=root,
        critical=True
    )
    claim_cert_exists = f"The provided source pages document superintendent certification/licensure requirements for the state of '{sub.state or ''}'."
    add_ins_cert_exists = "Prefer state DOE/licensure authority pages or administrative codes. The page(s) should explicitly describe superintendent (or district-level leader) certification requirements."
    leaves_and_payloads.append((claim_cert_exists, certification_urls, node_cert_exists, add_ins_cert_exists))

    # Masters_Degree_Required
    node_masters = evaluator.add_leaf(
        id="Masters_Degree_Required",
        desc="The state requires a master's degree as minimum education for superintendent certification",
        parent=root,
        critical=True
    )
    claim_masters = f"The state of '{sub.state or ''}' requires at least a master's degree (or higher) as the minimum educational qualification for superintendent certification/licensure."
    add_ins_masters = "Allow equivalents like Ed.S., master's degree or higher. Verify requirement text on the state certification pages."
    leaves_and_payloads.append((claim_masters, certification_urls, node_masters, add_ins_masters))

    # Graduate_Coursework_Required
    node_grad_course = evaluator.add_leaf(
        id="Graduate_Coursework_Required",
        desc="The state certification requirements include specific graduate-level coursework in educational administration or leadership",
        parent=root,
        critical=True
    )
    claim_grad_course = "The superintendent certification requirements include graduate-level coursework in educational administration or educational leadership (or equivalent)."
    add_ins_grad_course = "Look for graduate credits/courses in admin/leadership, e.g., 'school administration', 'educational leadership', or program-specific graduate coursework."
    leaves_and_payloads.append((claim_grad_course, certification_urls, node_grad_course, add_ins_grad_course))

    # Teaching_Experience_Required
    node_teaching = evaluator.add_leaf(
        id="Teaching_Experience_Required",
        desc="The state requires prior teaching experience as part of the superintendent certification requirements",
        parent=root,
        critical=True
    )
    claim_teaching = "The superintendent certification/licensure requirements include prior teaching experience (e.g., licensed teaching experience)."
    add_ins_teaching = "Accept phrasing like 'teaching experience', 'experience as a teacher', or 'licensed P-12 experience that includes teaching'."
    leaves_and_payloads.append((claim_teaching, certification_urls, node_teaching, add_ins_teaching))

    # Administrative_Experience_Required
    node_admin_exp_req = evaluator.add_leaf(
        id="Administrative_Experience_Required",
        desc="The state requires a minimum specified amount of prior administrative or supervisory experience for superintendent certification",
        parent=root,
        critical=True
    )
    claim_admin_exp_req = "The superintendent certification/licensure requirements include a minimum specified amount of prior administrative or supervisory experience (stated in years or months)."
    add_ins_admin_exp_req = "The page should specify an explicit minimum (e.g., N years or months) of administrative/supervisory/leadership experience."
    leaves_and_payloads.append((claim_admin_exp_req, certification_urls, node_admin_exp_req, add_ins_admin_exp_req))

    # Superintendent_Appointed_2024
    node_appt_2024 = evaluator.add_leaf(
        id="Superintendent_Appointed_2024",
        desc="A new superintendent was appointed to the district in 2024",
        parent=root,
        critical=True
    )
    claim_appt_2024 = f"In calendar year 2024, '{sub.superintendent_name or ''}' was appointed (named/hired/approved/selected) as superintendent of '{sub.district_name or ''}'."
    add_ins_appt_2024 = "Verify using district press releases, board minutes, or reputable local news pages provided. Accept synonyms like 'named', 'hired', 'approved', 'selected'."
    leaves_and_payloads.append((claim_appt_2024, appointment_related_urls, node_appt_2024, add_ins_appt_2024))

    # Start_Date_Documented_And_Provided
    node_start_date = evaluator.add_leaf(
        id="Start_Date_Documented_And_Provided",
        desc="The superintendent's appointment has a documented start date and the answer provides this specific start date",
        parent=root,
        critical=True
    )
    claim_start_date = f"The superintendent '{sub.superintendent_name or ''}' has a documented start (effective) date of '{sub.start_date or ''}'."
    add_ins_start_date = "Confirm the exact start/effective date using the provided sources (press release, contract, minutes). Accept phrasing like 'begins on', 'effective', or 'start'."
    leaves_and_payloads.append((claim_start_date, startdate_related_urls, node_start_date, add_ins_start_date))

    # Contract_Duration_Specified_And_Provided
    node_contract = evaluator.add_leaf(
        id="Contract_Duration_Specified_And_Provided",
        desc="The superintendent's employment contract specifies a duration period and the answer provides this duration",
        parent=root,
        critical=True
    )
    claim_contract = f"The employment contract for '{sub.superintendent_name or ''}' specifies a duration/term of '{sub.contract_duration or ''}'."
    add_ins_contract = "Verify the contract term (e.g., 'three-year term', '36 months', or a date range like 'through June 30, 2027')."
    leaves_and_payloads.append((claim_contract, contract_related_urls, node_contract, add_ins_contract))

    # Superintendent_Certification_Status
    node_cert_status = evaluator.add_leaf(
        id="Superintendent_Certification_Status",
        desc="The superintendent holds or is eligible for the state's required certification or licensure",
        parent=root,
        critical=True
    )
    claim_cert_status = f"'{sub.superintendent_name or ''}' holds or is eligible for the required superintendent certification/licensure in '{sub.state or ''}'."
    add_ins_cert_status = "Accept either 'holds' or 'eligible' language. Verify using district press releases, state databases, resumes/CVs posted by the district, or board documents."
    leaves_and_payloads.append((claim_cert_status, cert_status_urls, node_cert_status, add_ins_cert_status))

    # Execute all URL-based verifications (in parallel)
    await evaluator.batch_verify(leaves_and_payloads)


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
    # Initialize evaluator (root is non-critical by framework; children enforce critical gating)
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

    # Extract structured data from answer
    submission = await evaluator.extract(
        prompt=prompt_extract_submission(),
        template_class=DistrictSubmission,
        extraction_name="submission_extraction"
    )

    # Build verification tree and run checks
    await verify_submission(evaluator, root, submission)

    # Return summary
    return evaluator.get_summary()