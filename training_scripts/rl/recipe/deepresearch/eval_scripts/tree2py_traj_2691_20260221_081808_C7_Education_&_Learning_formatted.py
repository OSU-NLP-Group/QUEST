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
TASK_ID = "tx_public_univ_ba_biz"
TASK_DESCRIPTION = (
    "Identify a public university in Texas that meets all of the following criteria for its Business Administration "
    "bachelor's degree program:\n\n"
    "1. Holds regional accreditation from one of the six recognized U.S. regional accrediting organizations\n"
    "2. Has AACSB accreditation for its business program (verifiable through the AACSB accredited schools database)\n"
    "3. Offers a Bachelor's degree (B.S. or B.A.) specifically in Business Administration\n"
    "4. Requires 120 credit hours for degree completion\n"
    "5. Offers at least three distinct concentrations or specializations within the Business Administration major\n"
    "6. Operates at least three distinct campus locations within Texas\n"
    "7. Offers the Business Administration degree in an online format or with online course availability\n\n"
    "For the identified university, provide the following information with supporting URLs:\n\n"
    "- The university name\n"
    "- Direct URL to the Business Administration program webpage showing concentrations and requirements\n"
    "- Direct URL to verify AACSB accreditation (from aacsb.edu)\n"
    "- Direct URL to the university's regional accreditation information\n"
    "- Direct URL to information about the university's multiple campus locations within Texas\n"
    "- Direct URL to information about online degree availability for Business Administration\n"
    "- List the names of at least three concentrations offered within the Business Administration program\n"
    "- Direct URL to the published academic calendar for 2025-2026\n"
    "- Direct URL to published general education requirements for bachelor's degrees\n"
    "- Direct URL to undergraduate admission requirements including minimum GPA information"
)

RECOGNIZED_US_REGIONAL_ACCREDITORS = [
    "SACSCOC",  # Southern Association of Colleges and Schools Commission on Colleges
    "HLC",      # Higher Learning Commission
    "NECHE",    # New England Commission of Higher Education
    "MSCHE",    # Middle States Commission on Higher Education
    "NWCCU",    # Northwest Commission on Colleges and Universities
    "WSCUC",    # WASC Senior College and University Commission
]


# --------------------------------------------------------------------------- #
# Data models                                                                 #
# --------------------------------------------------------------------------- #
class UniversityInfo(BaseModel):
    """
    Structured extraction of all required fields for the university and program.
    All URL fields should be direct links as explicitly provided in the answer text.
    """
    university_name: Optional[str] = None

    program_url: Optional[str] = None  # Business Administration program page showing concentrations and requirements
    aacsb_url: Optional[str] = None    # Must be on aacsb.edu
    regional_accreditation_url: Optional[str] = None
    campuses_url: Optional[str] = None
    online_ba_url: Optional[str] = None

    concentrations: List[str] = Field(default_factory=list)

    academic_calendar_url: Optional[str] = None  # For 2025-2026
    general_education_url: Optional[str] = None
    admission_requirements_url: Optional[str] = None
    minimum_gpa_url: Optional[str] = None  # If separate; can duplicate admission_requirements_url if needed

    degree_type: Optional[str] = None  # e.g., "BS in Business Administration", "BA in Business Administration", "BBA"
    credit_hours: Optional[str] = None  # e.g., "120"

    public_status_url: Optional[str] = None  # Page indicating public/state institution status
    texas_location_url: Optional[str] = None  # Page confirming Texas location (could be About/Contact/Campus Locations)

    aacsb_school_name: Optional[str] = None  # e.g., "College of Business", "School of Business"


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_university_info() -> str:
    return """
    Extract the following information exactly as presented in the answer. If a field is not mentioned, return null for single values or [] for lists. Do not invent or infer any URLs or values.

    Required fields:
    - university_name: The name of the university identified.
    - program_url: Direct URL to the Business Administration bachelor's program webpage that lists concentrations and requirements.
    - aacsb_url: Direct URL on aacsb.edu that verifies AACSB accreditation for the university's business school.
    - regional_accreditation_url: Direct URL to the university's regional accreditation information page.
    - campuses_url: Direct URL to information about the university's multiple campus locations within Texas.
    - online_ba_url: Direct URL to information indicating the Business Administration bachelor's degree is offered online, or has online course availability.
    - concentrations: Array of the names of concentrations/specializations/tracks for Business Administration (extract at least three if present).
    - academic_calendar_url: Direct URL to the published academic calendar for the 2025-2026 academic year.
    - general_education_url: Direct URL to published general education requirements for bachelor's degrees (often called "Core Curriculum" or similar).
    - admission_requirements_url: Direct URL to undergraduate admission requirements page.
    - minimum_gpa_url: Direct URL that explicitly states a minimum GPA requirement for freshman admission; if not separate, use the same as admission_requirements_url.
    - degree_type: The specific bachelor’s degree type in Business Administration (e.g., "BS in Business Administration", "BA in Business Administration", "BBA").
    - credit_hours: The number of credit hours required to complete the Business Administration bachelor's degree (e.g., "120").
    - public_status_url: Direct URL that indicates the university is a public/state institution (if provided).
    - texas_location_url: Direct URL that indicates the university is located in Texas (if provided).
    - aacsb_school_name: The business school name shown on the AACSB page (e.g., "College of Business"), if mentioned in the answer.

    Special rules:
    - aacsb_url must be on the domain aacsb.edu. If the answer mentions AACSB accreditation without a URL, return null.
    - Extract only URLs explicitly present in the answer text (including markdown links). Do not construct or infer URLs.
    - For concentrations, extract exactly the names as listed in the answer text. If more than three are listed, include all.
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def _has_url(u: Optional[str]) -> bool:
    return bool(u and u.strip())


def _urls(*args: Optional[str]) -> List[str]:
    return [u for u in args if _has_url(u)]


# --------------------------------------------------------------------------- #
# Verification builder                                                        #
# --------------------------------------------------------------------------- #
async def build_university_criteria_tree(evaluator: Evaluator, root, info: UniversityInfo) -> None:
    """
    Build verification tree according to the rubric and run verifications.
    Each top-level criterion is represented as a non-leaf aggregator under the root.
    For critical criteria, all child checks are critical.
    For non-critical criteria, existence checks are critical (to gate) and verification leaves are non-critical.
    """

    # University_Meeting_Criteria root aggregator (parallel, non-critical)
    top_node = evaluator.add_parallel(
        id="University_Meeting_Criteria",
        desc="A public university in Texas offering an AACSB-accredited Business Administration bachelor's degree with multiple concentrations and campus locations",
        parent=root,
        critical=False
    )

    # -------------------------- Regional Accreditation (Critical) --------------------------
    reg_node = evaluator.add_parallel(
        id="Regional_Accreditation",
        desc="The university holds regional accreditation from one of the six recognized U.S. regional accrediting organizations",
        parent=top_node,
        critical=True
    )
    reg_exist = evaluator.add_custom_node(
        result=_has_url(info.regional_accreditation_url),
        id="Regional_Accreditation_Url_Provided",
        desc="Regional accreditation URL is provided",
        parent=reg_node,
        critical=True
    )
    reg_verify = evaluator.add_leaf(
        id="Regional_Accreditation_Verified",
        desc="Regional accreditation is supported by the cited page and is one of the six recognized U.S. regional accreditors",
        parent=reg_node,
        critical=True
    )
    reg_claim = f"The university {info.university_name or 'the identified university'} is regionally accredited by a recognized U.S. regional accreditor."
    await evaluator.verify(
        claim=reg_claim,
        node=reg_verify,
        sources=info.regional_accreditation_url,
        additional_instruction=(
            "Confirm that the accreditor is one of: SACSCOC, HLC, NECHE, MSCHE, NWCCU, WSCUC. "
            "The page should clearly indicate institutional regional accreditation."
        )
    )

    # -------------------------- AACSB Accreditation (Critical) ---------------------------
    aacsb_node = evaluator.add_parallel(
        id="AACSB_Accreditation",
        desc="The university's business program holds AACSB accreditation, verifiable through the AACSB accredited schools database",
        parent=top_node,
        critical=True
    )
    aacsb_exist = evaluator.add_custom_node(
        result=_has_url(info.aacsb_url) and ("aacsb.edu" in (info.aacsb_url or "")),
        id="AACSB_Accreditation_Url_Provided",
        desc="AACSB accreditation URL on aacsb.edu is provided",
        parent=aacsb_node,
        critical=True
    )
    aacsb_verify = evaluator.add_leaf(
        id="AACSB_Accreditation_Verified",
        desc="AACSB accreditation is confirmed by the page on aacsb.edu",
        parent=aacsb_node,
        critical=True
    )
    aacsb_claim = (
        f"The AACSB page shows that the business school at {info.university_name or 'this university'} "
        f"is AACSB accredited."
    )
    await evaluator.verify(
        claim=aacsb_claim,
        node=aacsb_verify,
        sources=info.aacsb_url,
        additional_instruction=(
            "The AACSB page may list the business school name (e.g., College/School of Business) rather than the university directly. "
            "It should clearly indicate AACSB accreditation for that unit affiliated with the university."
        )
    )

    # -------------------------- Business Administration Degree (Critical) ----------------
    ba_node = evaluator.add_parallel(
        id="Business_Administration_Degree",
        desc="The university offers a Bachelor of Science or Bachelor of Arts degree specifically in Business Administration",
        parent=top_node,
        critical=True
    )
    ba_exist = evaluator.add_custom_node(
        result=_has_url(info.program_url),
        id="Business_Administration_Program_Url_Provided",
        desc="Business Administration program URL is provided",
        parent=ba_node,
        critical=True
    )
    ba_verify = evaluator.add_leaf(
        id="Business_Administration_Degree_Verified",
        desc="The program page shows a bachelor's degree specifically in Business Administration",
        parent=ba_node,
        critical=True
    )
    ba_claim = (
        "This page describes a bachelor's degree specifically in Business Administration (e.g., BS or BA in Business Administration; "
        "Bachelor of Business Administration (BBA) also qualifies as a bachelor's degree in Business Administration)."
    )
    await evaluator.verify(
        claim=ba_claim,
        node=ba_verify,
        sources=info.program_url,
        additional_instruction=(
            "Confirm the page is for the undergraduate Business Administration major and indicates a bachelor's-level credential "
            "(BS, BA, or BBA) specifically in Business Administration."
        )
    )

    # -------------------------- Credit Hour Requirement (Critical) -----------------------
    cred_node = evaluator.add_parallel(
        id="Credit_Hour_Requirement",
        desc="The Business Administration bachelor's degree program requires 120 credit hours for completion",
        parent=top_node,
        critical=True
    )
    cred_exist = evaluator.add_custom_node(
        result=_has_url(info.program_url),
        id="Credit_Hour_Requirement_Source_Provided",
        desc="A source URL for degree requirements is provided",
        parent=cred_node,
        critical=True
    )
    cred_verify = evaluator.add_leaf(
        id="Credit_Hour_Requirement_Verified",
        desc="The program requires 120 credit hours",
        parent=cred_node,
        critical=True
    )
    cred_claim = "The Business Administration bachelor's degree requires 120 credit hours to complete."
    await evaluator.verify(
        claim=cred_claim,
        node=cred_verify,
        sources=_urls(info.program_url, info.general_education_url),
        additional_instruction=(
            "Verify the total credit hours stated for completing the bachelor's degree in Business Administration are 120."
        )
    )

    # -------------------------- Concentration Options (Critical) ------------------------
    conc_node = evaluator.add_parallel(
        id="Concentration_Options",
        desc="The Business Administration program offers at least three distinct concentrations or specializations within the major",
        parent=top_node,
        critical=True
    )
    conc_exist = evaluator.add_custom_node(
        result=len(info.concentrations) >= 3 and _has_url(info.program_url),
        id="Concentration_List_Provided",
        desc="At least three concentration names are provided and a program URL exists",
        parent=conc_node,
        critical=True
    )
    conc_verify = evaluator.add_leaf(
        id="Concentration_Options_Verified",
        desc="The program page lists the provided concentrations (at least three distinct options)",
        parent=conc_node,
        critical=True
    )
    top_three = info.concentrations[:3]
    conc_claim = (
        f"This page lists at least three distinct concentrations/specializations in Business Administration, "
        f"including: {', '.join(top_three)}."
    )
    await evaluator.verify(
        claim=conc_claim,
        node=conc_verify,
        sources=info.program_url,
        additional_instruction=(
            "Confirm the page explicitly lists at least three concentrations/specializations/tracks for the Business Administration major, "
            "and that the named examples appear on the page."
        )
    )

    # -------------------------- Public University (Critical) ----------------------------
    pub_node = evaluator.add_parallel(
        id="Public_University",
        desc="The university is a public institution (state university)",
        parent=top_node,
        critical=True
    )
    pub_exist = evaluator.add_custom_node(
        result=_has_url(info.public_status_url),
        id="Public_Status_Url_Provided",
        desc="A URL indicating public/state status is provided",
        parent=pub_node,
        critical=True
    )
    pub_verify = evaluator.add_leaf(
        id="Public_University_Verified",
        desc="The university is confirmed to be a public/state institution",
        parent=pub_node,
        critical=True
    )
    pub_claim = f"{info.university_name or 'The identified university'} is a public (state) university."
    await evaluator.verify(
        claim=pub_claim,
        node=pub_verify,
        sources=info.public_status_url,
        additional_instruction=(
            "The page should indicate the institution is a public/state university (e.g., part of a state system or described as public)."
        )
    )

    # -------------------------- Texas Location (Critical) -------------------------------
    tx_node = evaluator.add_parallel(
        id="Texas_Location",
        desc="The university is located in the state of Texas",
        parent=top_node,
        critical=True
    )
    tx_exist = evaluator.add_custom_node(
        result=_has_url(info.texas_location_url) or _has_url(info.campuses_url),
        id="Texas_Location_Url_Provided",
        desc="A URL confirming Texas location is provided",
        parent=tx_node,
        critical=True
    )
    tx_verify = evaluator.add_leaf(
        id="Texas_Location_Verified",
        desc="The university is confirmed to be in Texas",
        parent=tx_node,
        critical=True
    )
    tx_claim = f"{info.university_name or 'The identified university'} is located in Texas."
    await evaluator.verify(
        claim=tx_claim,
        node=tx_verify,
        sources=_urls(info.texas_location_url, info.campuses_url),
        additional_instruction=(
            "Confirm the institution is in the state of Texas; campus location pages or About pages that clearly state Texas are valid."
        )
    )

    # -------------------------- Multiple Campuses (Critical) ----------------------------
    campuses_node = evaluator.add_parallel(
        id="Multiple_Campuses",
        desc="The university operates at least three distinct campus locations within Texas",
        parent=top_node,
        critical=True
    )
    campuses_exist = evaluator.add_custom_node(
        result=_has_url(info.campuses_url),
        id="Campuses_Url_Provided",
        desc="A URL listing campus locations is provided",
        parent=campuses_node,
        critical=True
    )
    campuses_verify = evaluator.add_leaf(
        id="Multiple_Campuses_Verified",
        desc="The university operates at least three distinct campus locations in Texas",
        parent=campuses_node,
        critical=True
    )
    campuses_claim = (
        f"{info.university_name or 'The identified university'} operates at least three distinct campus locations in Texas."
    )
    await evaluator.verify(
        claim=campuses_claim,
        node=campuses_verify,
        sources=info.campuses_url,
        additional_instruction=(
            "Confirm that the page lists three or more distinct campus locations within Texas."
        )
    )

    # -------------------------- Online Availability (Critical) --------------------------
    online_node = evaluator.add_parallel(
        id="Online_Availability",
        desc="The university offers the Business Administration bachelor's degree in an online format or with online courses available",
        parent=top_node,
        critical=True
    )
    online_exist = evaluator.add_custom_node(
        result=_has_url(info.online_ba_url),
        id="Online_Availability_Url_Provided",
        desc="A URL indicating online availability for Business Administration is provided",
        parent=online_node,
        critical=True
    )
    online_verify = evaluator.add_leaf(
        id="Online_Availability_Verified",
        desc="Business Administration bachelor's degree is offered online or has online course availability",
        parent=online_node,
        critical=True
    )
    online_claim = (
        "The Business Administration bachelor's program is available online (fully online) or offers online course options."
    )
    await evaluator.verify(
        claim=online_claim,
        node=online_verify,
        sources=info.online_ba_url,
        additional_instruction=(
            "The page should explicitly indicate online format availability for the BA/BS/BBA in Business Administration "
            "or clearly state online course options within the program."
        )
    )

    # -------------------------- Academic Calendar 2025-2026 (Non-Critical) --------------
    cal_node = evaluator.add_parallel(
        id="Academic_Calendar_2025_2026",
        desc="The university has a publicly available academic calendar for the 2025-2026 academic year",
        parent=top_node,
        critical=False
    )
    cal_exist = evaluator.add_custom_node(
        result=_has_url(info.academic_calendar_url),
        id="Academic_Calendar_Url_Provided",
        desc="Academic calendar 2025-2026 URL is provided",
        parent=cal_node,
        critical=True
    )
    cal_verify = evaluator.add_leaf(
        id="Academic_Calendar_Verified",
        desc="The page is the published academic calendar for 2025-2026",
        parent=cal_node,
        critical=False
    )
    cal_claim = "This page is the institution's academic calendar for the 2025–2026 academic year."
    await evaluator.verify(
        claim=cal_claim,
        node=cal_verify,
        sources=info.academic_calendar_url,
        additional_instruction=(
            "Confirm that the calendar explicitly covers the 2025–2026 academic year (terms, dates, semesters)."
        )
    )

    # -------------------------- General Education Requirements (Non-Critical) -----------
    gened_node = evaluator.add_parallel(
        id="General_Education_Requirements",
        desc="The university publishes specific general education requirements for bachelor's degrees, including required credit hours",
        parent=top_node,
        critical=False
    )
    gened_exist = evaluator.add_custom_node(
        result=_has_url(info.general_education_url),
        id="General_Education_Url_Provided",
        desc="General education/core curriculum URL is provided",
        parent=gened_node,
        critical=True
    )
    gened_verify = evaluator.add_leaf(
        id="General_Education_Verified",
        desc="Published general education requirements for bachelor's degrees are available",
        parent=gened_node,
        critical=False
    )
    gened_claim = "This page publishes general education (core curriculum) requirements for bachelor's degrees at the university."
    await evaluator.verify(
        claim=gened_claim,
        node=gened_verify,
        sources=info.general_education_url,
        additional_instruction=(
            "The page should present official general education/core requirements applicable to undergraduate degrees. "
            "It may also indicate associated credit hours."
        )
    )

    # -------------------------- Minimum GPA Requirement (Non-Critical) ------------------
    gpa_node = evaluator.add_parallel(
        id="Minimum_GPA_Requirement",
        desc="The university publishes a minimum GPA requirement for freshman admission to undergraduate programs",
        parent=top_node,
        critical=False
    )
    gpa_exist = evaluator.add_custom_node(
        result=_has_url(info.minimum_gpa_url or info.admission_requirements_url),
        id="Minimum_GPA_Url_Provided",
        desc="A URL stating minimum GPA for freshman admission is provided",
        parent=gpa_node,
        critical=True
    )
    gpa_verify = evaluator.add_leaf(
        id="Minimum_GPA_Verified",
        desc="Minimum GPA requirement for freshman undergraduate admission is published",
        parent=gpa_node,
        critical=False
    )
    gpa_claim = "This page states a minimum GPA requirement for freshman undergraduate admission."
    await evaluator.verify(
        claim=gpa_claim,
        node=gpa_verify,
        sources=(info.minimum_gpa_url or info.admission_requirements_url),
        additional_instruction=(
            "Confirm the page explicitly mentions a numeric minimum GPA for freshman/first-year undergraduate admission."
        )
    )

    # -------------------------- Program Website (Non-Critical) --------------------------
    prog_node = evaluator.add_parallel(
        id="Program_Website",
        desc="The Business Administration program has a dedicated webpage listing program details, concentrations, and requirements",
        parent=top_node,
        critical=False
    )
    prog_exist = evaluator.add_custom_node(
        result=_has_url(info.program_url),
        id="Program_Website_Url_Provided",
        desc="Business Administration program page URL is provided",
        parent=prog_node,
        critical=True
    )
    prog_verify = evaluator.add_leaf(
        id="Program_Website_Verified",
        desc="The program page lists program details, concentrations, and requirements",
        parent=prog_node,
        critical=False
    )
    prog_claim = "This webpage is the Business Administration bachelor's program page and lists program details, concentrations, and requirements."
    await evaluator.verify(
        claim=prog_claim,
        node=prog_verify,
        sources=info.program_url,
        additional_instruction=(
            "Confirm the page is specifically for the undergraduate Business Administration program and includes or links to concentrations and degree requirements."
        )
    )

    # -------------------------- Admission Requirements Page (Non-Critical) --------------
    adm_node = evaluator.add_parallel(
        id="Admission_Requirements_Page",
        desc="The university has a publicly accessible webpage detailing admission requirements for undergraduate students",
        parent=top_node,
        critical=False
    )
    adm_exist = evaluator.add_custom_node(
        result=_has_url(info.admission_requirements_url),
        id="Admission_Requirements_Url_Provided",
        desc="Undergraduate admission requirements URL is provided",
        parent=adm_node,
        critical=True
    )
    adm_verify = evaluator.add_leaf(
        id="Admission_Requirements_Verified",
        desc="Undergraduate admission requirements are detailed on the page",
        parent=adm_node,
        critical=False
    )
    adm_claim = "This webpage details undergraduate admission requirements for the university."
    await evaluator.verify(
        claim=adm_claim,
        node=adm_verify,
        sources=info.admission_requirements_url,
        additional_instruction="Confirm the page presents undergraduate admission requirements."
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
    Evaluate the answer for the Texas public university Business Administration program criteria.
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

    # Extract structured information from the answer
    info: UniversityInfo = await evaluator.extract(
        prompt=prompt_extract_university_info(),
        template_class=UniversityInfo,
        extraction_name="university_info",
    )

    # Add contextual info for evaluation output (optional helpful metadata)
    evaluator.add_custom_info(
        info={
            "recognized_regional_accreditors": RECOGNIZED_US_REGIONAL_ACCREDITORS,
        },
        info_type="reference",
        info_name="regional_accreditor_list"
    )

    # Build verification tree and run checks
    await build_university_criteria_tree(evaluator, root, info)

    # Return structured summary
    return evaluator.get_summary()