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
TASK_ID = "us_public_school_ad_2025"
TASK_DESCRIPTION = """
Identify a person who was appointed as an athletic director (or equivalent district-level student activities/athletics leadership position) at a U.S. public school district in 2025, and provide comprehensive information about their background. Your answer must include: (1) Identification: The full name of the person; (2) Appointment Details: The year of appointment (must be 2025), the specific position title, confirmation that it is a U.S. public school district position, the effective date or month of the appointment, and a reference URL confirming the appointment; (3) Educational Background: Confirmation that they hold a bachelor's degree, the specific field/major of the bachelor's degree, the institution that granted the bachelor's degree, whether they hold a graduate degree (master's or higher), if applicable the field/major and institution of the graduate degree, and a reference URL confirming educational background; (4) Professional Experience: Confirmation of at least 10 years of professional experience in education, athletics, or related fields, confirmation of previous position(s) involving athletic administration, coaching, or supervision of athletic programs, the title of their most recent position before the athletic director appointment, the organization/school/district where that previous position was held, information about any coaching experience, information about any school administrative experience, and a reference URL confirming professional experience; (5) Specialized Qualifications (if available): Knowledge of state athletic association rules, professional recognition, awards, or leadership positions, and NIAAA certifications. All information must be verifiable through public sources and accompanied by appropriate reference URLs.
"""


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class AppointmentInfo(BaseModel):
    year: Optional[str] = None
    position_title: Optional[str] = None
    district_name: Optional[str] = None
    district_type: Optional[str] = None  # e.g., "public school district"
    country: Optional[str] = None        # e.g., "United States" / "USA"
    effective_date_or_month: Optional[str] = None  # e.g., "July 2025" or "2025-07-01"
    urls: List[str] = Field(default_factory=list)


class EducationInfo(BaseModel):
    has_bachelors: Optional[str] = None  # e.g., "yes", "Bachelor of Science", etc.
    bachelors_field: Optional[str] = None
    bachelors_institution: Optional[str] = None
    has_graduate_degree: Optional[str] = None  # e.g., "yes", "Master of Education", etc.
    graduate_field: Optional[str] = None
    graduate_institution: Optional[str] = None
    urls: List[str] = Field(default_factory=list)


class ExperienceInfo(BaseModel):
    years_experience: Optional[str] = None  # e.g., "12 years", "over 15 years"
    previous_athletic_leadership: Optional[str] = None  # description or "yes"
    previous_position_title: Optional[str] = None
    previous_organization: Optional[str] = None
    coaching_experience: Optional[str] = None
    administrative_experience: Optional[str] = None
    urls: List[str] = Field(default_factory=list)


class QualificationsInfo(BaseModel):
    state_rules_knowledge: Optional[str] = None
    professional_recognition: Optional[str] = None
    niaaa_certifications: List[str] = Field(default_factory=list)  # e.g., ["RAA", "CAA"]
    urls: List[str] = Field(default_factory=list)


class CandidateExtraction(BaseModel):
    full_name: Optional[str] = None
    appointment: Optional[AppointmentInfo] = None
    education: Optional[EducationInfo] = None
    experience: Optional[ExperienceInfo] = None
    qualifications: Optional[QualificationsInfo] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_candidate() -> str:
    return """
    Extract structured information about a single identified person appointed as an athletic director (or equivalent district-level student activities/athletics leadership position) at a U.S. public school district in 2025 from the provided answer.

    Return a JSON object with the following fields:
    - full_name: The full name of the identified person.

    - appointment: {
        year: The appointment year explicitly mentioned (e.g., "2025").
        position_title: The specific position title (e.g., "Athletic Director", "Director of Athletics", "Director of Student Activities", "Activities Director", etc.).
        district_name: The name of the U.S. public school district (e.g., "Springfield Public Schools").
        district_type: The stated type of the employer (aim for wording like "public school district", "public schools", "school district").
        country: The country (should be "United States", "U.S.", "USA", or similar).
        effective_date_or_month: The effective date or month for the appointment, if provided (e.g., "July 2025", "2025-07", "effective July 1, 2025").
        urls: A list of URL(s) in the answer that support or confirm the appointment (include district/news release links; only include URLs explicitly present in the answer).
      }

    - education: {
        has_bachelors: A phrase that clearly indicates the person holds a bachelor's degree (e.g., "Bachelor of Science", "earned a bachelor's degree"), or "yes" if only confirmation is given.
        bachelors_field: The specific field/major of the bachelor's degree, if available.
        bachelors_institution: The institution that granted the bachelor's degree, if available.
        has_graduate_degree: A phrase indicating they hold a graduate degree (e.g., "Master of Education", "Doctorate") or "yes" if only confirmation is given; if not present, set to null.
        graduate_field: The specific field/major of the graduate degree, if available.
        graduate_institution: The institution for the graduate degree, if available.
        urls: A list of URL(s) explicitly present in the answer that confirm the educational background (e.g., district bio page, press release).
      }

    - experience: {
        years_experience: A phrase that states or implies total professional experience (e.g., "over 10 years", "15 years").
        previous_athletic_leadership: A phrase indicating prior roles involving athletic administration/coaching/supervision (e.g., "assistant athletic director", "head coach"), or "yes" if generally asserted.
        previous_position_title: The title of the most recent position before the athletic director appointment, if provided.
        previous_organization: The organization/school/district of that most recent prior role, if provided.
        coaching_experience: A phrase confirming any coaching experience, if provided (sport/level if present).
        administrative_experience: A phrase confirming any school administrative experience (principal, AP, dean, etc.), if provided.
        urls: A list of URL(s) explicitly present in the answer that confirm professional experience.
      }

    - qualifications: {
        state_rules_knowledge: A phrase indicating knowledge of state athletic association rules (e.g., UIL, VHSL, WIAA, FHSAA), if available.
        professional_recognition: A phrase listing notable awards or leadership roles in professional organizations, if available.
        niaaa_certifications: An array of NIAAA certifications mentioned (e.g., ["RAA","CAA","CMAA"]); if none, return an empty array.
        urls: A list of URL(s), if any are explicitly present for specialized qualifications (may overlap with experience/education URLs).
      }

    Special rules:
    - Extract only from the provided answer text. Do not invent or infer details not present.
    - For all URL fields, include only valid URLs explicitly present in the answer (plain or markdown). If a URL lacks a protocol, prepend "http://".
    - If a field is missing in the answer, return null for singular fields or [] for arrays.
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _safe_list(urls: Optional[List[str]]) -> List[str]:
    if not urls:
        return []
    # Basic de-duplication and cleanup
    cleaned = []
    for u in urls:
        if not isinstance(u, str):
            continue
        s = u.strip()
        if not s:
            continue
        if s not in cleaned:
            cleaned.append(s)
    return cleaned


def _merge_sources(*lists: List[str]) -> List[str]:
    merged: List[str] = []
    for lst in lists:
        for u in lst:
            if u not in merged:
                merged.append(u)
    return merged


# --------------------------------------------------------------------------- #
# Verification subroutines                                                    #
# --------------------------------------------------------------------------- #
async def verify_appointment_details(
    evaluator: Evaluator,
    parent_node,
    full_name: Optional[str],
    appt: Optional[AppointmentInfo],
):
    # Parent node for appointment details (non-critical to allow mixed children criticality)
    appt_node = evaluator.add_parallel(
        id="Verify_Appointment_Details",
        desc="Verify the appointment details of the identified candidate",
        parent=parent_node,
        critical=False,
    )

    # Prepare values and sources
    appt = appt or AppointmentInfo()
    appt_sources = _safe_list(appt.urls)

    # Reference URL presence (critical) — evaluate first to gate other checks
    ref_url_node = evaluator.add_custom_node(
        result=len(appt_sources) > 0,
        id="Reference_URL_Appointment",
        desc="Provide a URL reference that confirms the appointment",
        parent=appt_node,
        critical=True,
    )

    # Appointment Year (critical)
    year_leaf = evaluator.add_leaf(
        id="Appointment_Year",
        desc="The person was appointed to their athletic director (or equivalent) position in 2025",
        parent=appt_node,
        critical=True,
    )
    year_claim_name = full_name or "the candidate"
    pos_title = appt.position_title or "an athletics leadership role"
    district_phrase = appt.district_name or "a school district"
    year_claim = (
        f"The appointment of {year_claim_name} to {pos_title} at {district_phrase} occurred in 2025 "
        f"(announcement/press release or effective date in 2025)."
    )
    await evaluator.verify(
        claim=year_claim,
        node=year_leaf,
        sources=appt_sources,
        additional_instruction="Accept if the announcement/publication date is in 2025 or if it explicitly states the appointment is effective in 2025.",
    )

    # Position Title (critical)
    title_leaf = evaluator.add_leaf(
        id="Position_Title",
        desc="The position title is Athletic Director, Director of Athletics, Director of Student Activities, or equivalent athletics leadership role at the district level",
        parent=appt_node,
        critical=True,
    )
    title_claim = (
        f"The appointed position is a district-level athletics leadership role (e.g., athletic director or equivalent). "
        f"If specified, the title is '{appt.position_title}' or an equivalent."
    )
    await evaluator.verify(
        claim=title_claim,
        node=title_leaf,
        sources=appt_sources,
        additional_instruction="Look for titles such as Athletic Director, Director of Athletics, Activities Director, or Director of Student Activities; confirm that it is a DISTRICT-level role.",
    )

    # School District Type (critical)
    sd_leaf = evaluator.add_leaf(
        id="School_District_Type",
        desc="The appointment is at a U.S. public school district (not a private school or collegiate institution)",
        parent=appt_node,
        critical=True,
    )
    country_phrase = appt.country or "United States"
    district_name = appt.district_name or "the district"
    sd_claim = (
        f"The appointment is at {district_name}, which is a U.S. public K-12 school district in the {country_phrase} (not a private school and not a college/university)."
    )
    await evaluator.verify(
        claim=sd_claim,
        node=sd_leaf,
        sources=appt_sources,
        additional_instruction="Confirm that the organization is a U.S. public K-12 school district (district website, official press release, or reputable news). Exclude private schools and higher-education institutions.",
    )

    # Effective date/month provided (non-critical, presence check)
    eff_node = evaluator.add_custom_node(
        result=bool(appt.effective_date_or_month and appt.effective_date_or_month.strip()),
        id="Effective_Date_Provided",
        desc="Provide the specific effective date or month when the appointment took effect",
        parent=appt_node,
        critical=False,
    )


async def verify_educational_background(
    evaluator: Evaluator,
    parent_node,
    full_name: Optional[str],
    edu: Optional[EducationInfo],
):
    edu_node = evaluator.add_parallel(
        id="Verify_Educational_Background",
        desc="Verify the identified candidate's educational qualifications",
        parent=parent_node,
        critical=False,
    )

    edu = edu or EducationInfo()
    edu_sources = _safe_list(edu.urls)

    # Reference URL presence (critical) — evaluate first to gate other checks
    ref_url_node = evaluator.add_custom_node(
        result=len(edu_sources) > 0,
        id="Reference_URL_Education",
        desc="Provide a URL reference that confirms the educational background",
        parent=edu_node,
        critical=True,
    )

    # Bachelor's Degree (critical)
    bachelor_leaf = evaluator.add_leaf(
        id="Bachelors_Degree",
        desc="The person holds a bachelor's degree",
        parent=edu_node,
        critical=True,
    )
    name_for_claim = full_name or "the candidate"
    bachelor_claim = (
        f"According to the cited sources, {name_for_claim} holds a bachelor's degree (BA, BS, BEd, or similar)."
    )
    await evaluator.verify(
        claim=bachelor_claim,
        node=bachelor_leaf,
        sources=edu_sources,
        additional_instruction="Accept if the page clearly indicates the person earned a bachelor's degree (BA/BS/B.Ed or equivalent).",
    )

    # Bachelor's Field provided (non-critical; presence check)
    bf_node = evaluator.add_custom_node(
        result=bool(edu.bachelors_field and edu.bachelors_field.strip()),
        id="Bachelors_Field",
        desc="Provide the specific field or major of the bachelor's degree",
        parent=edu_node,
        critical=False,
    )

    # Bachelor's Institution provided (non-critical; presence check)
    bi_node = evaluator.add_custom_node(
        result=bool(edu.bachelors_institution and edu.bachelors_institution.strip()),
        id="Bachelors_Institution",
        desc="Provide the name of the institution that granted the bachelor's degree",
        parent=edu_node,
        critical=False,
    )

    # Graduate Degree (non-critical, verify if claimed/present)
    grad_leaf = evaluator.add_leaf(
        id="Graduate_Degree",
        desc="The person holds a master's degree or higher graduate degree",
        parent=edu_node,
        critical=False,
    )
    grad_claim = (
        f"The sources indicate that {name_for_claim} holds a graduate degree (master's or higher)."
    )
    await evaluator.verify(
        claim=grad_claim,
        node=grad_leaf,
        sources=edu_sources,
        additional_instruction="Accept if the page mentions a master's degree, doctorate, or equivalent graduate credential.",
    )

    # Graduate Field provided (non-critical; presence check)
    gf_node = evaluator.add_custom_node(
        result=bool(edu.graduate_field and edu.graduate_field.strip()),
        id="Graduate_Field",
        desc="If a graduate degree is held, provide the specific field or major",
        parent=edu_node,
        critical=False,
    )

    # Graduate Institution provided (non-critical; presence check)
    gi_node = evaluator.add_custom_node(
        result=bool(edu.graduate_institution and edu.graduate_institution.strip()),
        id="Graduate_Institution",
        desc="If a graduate degree is held, provide the name of the institution",
        parent=edu_node,
        critical=False,
    )


async def verify_professional_experience(
    evaluator: Evaluator,
    parent_node,
    full_name: Optional[str],
    exp: Optional[ExperienceInfo],
):
    exp_node = evaluator.add_parallel(
        id="Verify_Professional_Experience",
        desc="Verify the identified candidate's professional experience and career progression",
        parent=parent_node,
        critical=False,
    )

    exp = exp or ExperienceInfo()
    exp_sources = _safe_list(exp.urls)

    # Reference URL presence (critical) — evaluate first to gate other checks
    ref_url_node = evaluator.add_custom_node(
        result=len(exp_sources) > 0,
        id="Reference_URL_Experience",
        desc="Provide a URL reference that confirms the professional experience",
        parent=exp_node,
        critical=True,
    )

    # Years in Education (critical)
    years_leaf = evaluator.add_leaf(
        id="Years_in_Education",
        desc="The person has at least 10 years of professional experience in education, athletics, or related fields",
        parent=exp_node,
        critical=True,
    )
    years_claim = (
        f"The sources indicate that {full_name or 'the candidate'} has at least 10 years of professional experience in education, athletics, or related fields."
    )
    await evaluator.verify(
        claim=years_claim,
        node=years_leaf,
        sources=exp_sources,
        additional_instruction="Accept phrasing like 'over 10 years', 'more than a decade', '15 years', etc.",
    )

    # Previous Athletic Leadership (critical)
    pal_leaf = evaluator.add_leaf(
        id="Previous_Athletic_Leadership",
        desc="The person held a previous position involving athletic administration, coaching, or supervision of athletic programs",
        parent=exp_node,
        critical=True,
    )
    pal_claim = (
        f"The sources confirm that {full_name or 'the candidate'} previously held roles involving athletic administration, coaching, or supervision of athletic programs."
    )
    await evaluator.verify(
        claim=pal_claim,
        node=pal_leaf,
        sources=exp_sources,
        additional_instruction="Accept roles such as assistant/associate athletic director, activities director, coach, athletics supervisor, etc.",
    )

    # Previous Position Title provided (non-critical; presence check)
    ppt_node = evaluator.add_custom_node(
        result=bool(exp.previous_position_title and exp.previous_position_title.strip()),
        id="Previous_Position_Title",
        desc="Provide the title of the most recent position held before the athletic director appointment",
        parent=exp_node,
        critical=False,
    )

    # Previous Organization provided (non-critical; presence check)
    po_node = evaluator.add_custom_node(
        result=bool(exp.previous_organization and exp.previous_organization.strip()),
        id="Previous_Organization",
        desc="Provide the name of the organization/school/district where the most recent previous position was held",
        parent=exp_node,
        critical=False,
    )

    # Coaching Experience (non-critical; verify if claimed/present)
    coach_leaf = evaluator.add_leaf(
        id="Coaching_Experience",
        desc="The person has coaching experience (any sport, any level)",
        parent=exp_node,
        critical=False,
    )
    coach_claim = f"The sources indicate that {full_name or 'the candidate'} has coaching experience (any sport and any level)."
    await evaluator.verify(
        claim=coach_claim,
        node=coach_leaf,
        sources=exp_sources,
        additional_instruction="Look for mentions of coaching titles, seasons coached, or sports teams coached.",
    )

    # Administrative Experience (non-critical; verify if claimed/present)
    admin_leaf = evaluator.add_leaf(
        id="Administrative_Experience",
        desc="The person has school administrative experience (e.g., principal, associate principal, dean, coordinator)",
        parent=exp_node,
        critical=False,
    )
    admin_claim = f"The sources indicate that {full_name or 'the candidate'} has school administrative experience (e.g., principal, assistant/associate principal, dean, coordinator)."
    await evaluator.verify(
        claim=admin_claim,
        node=admin_leaf,
        sources=exp_sources,
        additional_instruction="Accept any K-12 administrative role, not limited to athletics.",
    )


async def verify_specialized_qualifications(
    evaluator: Evaluator,
    parent_node,
    full_name: Optional[str],
    qual: Optional[QualificationsInfo],
    exp: Optional[ExperienceInfo],
):
    qual_node = evaluator.add_parallel(
        id="Verify_Specialized_Qualifications",
        desc="Verify any specialized qualifications, certifications, or notable achievements",
        parent=parent_node,
        critical=False,
    )

    qual = qual or QualificationsInfo()
    exp = exp or ExperienceInfo()
    # Use qualifications URLs; if none, fall back to experience URLs
    qual_sources = _safe_list(qual.urls)
    if not qual_sources:
        qual_sources = _safe_list(exp.urls)

    # State athletic rules knowledge (non-critical)
    state_leaf = evaluator.add_leaf(
        id="State_Athletic_Rules_Knowledge",
        desc="The person has demonstrated knowledge of state athletic association rules (e.g., UIL, VHSL, or equivalent)",
        parent=qual_node,
        critical=False,
    )
    state_claim = f"The sources show that {full_name or 'the candidate'} has demonstrated knowledge of state athletic association rules (e.g., UIL, VHSL, WIAA, FHSAA, etc.)."
    await evaluator.verify(
        claim=state_claim,
        node=state_leaf,
        sources=qual_sources,
        additional_instruction="Accept service on state association committees, compliance oversight, or references to implementing or understanding state association rules.",
    )

    # Professional recognition (non-critical)
    recog_leaf = evaluator.add_leaf(
        id="Professional_Recognition",
        desc="The person has received professional recognition, awards, or held leadership positions in professional athletic administration organizations",
        parent=qual_node,
        critical=False,
    )
    recog_claim = f"The sources indicate that {full_name or 'the candidate'} has professional recognition (awards) or leadership roles in athletic administration organizations."
    await evaluator.verify(
        claim=recog_claim,
        node=recog_leaf,
        sources=qual_sources,
        additional_instruction="Accept awards, honors, or leadership positions in organizations such as NIAAA, state athletic associations, or AD associations.",
    )

    # NIAAA certification (non-critical)
    niaaa_leaf = evaluator.add_leaf(
        id="NIAAA_Certification",
        desc="The person holds or has pursued NIAAA certification (RAA, CAA, or CMAA)",
        parent=qual_node,
        critical=False,
    )
    niaaa_claim = f"The sources indicate that {full_name or 'the candidate'} holds or has pursued NIAAA certification (e.g., RAA, CAA, or CMAA)."
    await evaluator.verify(
        claim=niaaa_claim,
        node=niaaa_leaf,
        sources=qual_sources,
        additional_instruction="Look for explicit mentions of NIAAA and credentials RAA, CAA, or CMAA.",
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
    Evaluate an answer for the 2025 U.S. public school district athletic director appointment task.
    """
    # Initialize evaluator
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
        prompt=prompt_extract_candidate(),
        template_class=CandidateExtraction,
        extraction_name="candidate_extraction",
    )

    # Build main sequential evaluation node
    ad_main = evaluator.add_sequential(
        id="Athletic_Director_Identification",
        desc="Correctly identify an athletic director appointed in 2025 who meets all specified educational, experiential, and appointment criteria",
        parent=root,
        critical=False,  # Keep non-critical to allow mixed child criticality under this parent
    )

    # 1) Identify Candidate (critical existence check)
    identify_node = evaluator.add_custom_node(
        result=bool(extracted.full_name and extracted.full_name.strip()),
        id="Identify_Candidate",
        desc="Provide the full name of a person appointed as athletic director (or equivalent student activities/athletics leadership position) at a U.S. public school district in 2025",
        parent=ad_main,
        critical=True,
    )

    # 2) Verify Appointment Details
    await verify_appointment_details(
        evaluator=evaluator,
        parent_node=ad_main,
        full_name=extracted.full_name,
        appt=extracted.appointment,
    )

    # 3) Verify Educational Background
    await verify_educational_background(
        evaluator=evaluator,
        parent_node=ad_main,
        full_name=extracted.full_name,
        edu=extracted.education,
    )

    # 4) Verify Professional Experience
    await verify_professional_experience(
        evaluator=evaluator,
        parent_node=ad_main,
        full_name=extracted.full_name,
        exp=extracted.experience,
    )

    # 5) Verify Specialized Qualifications (if available)
    await verify_specialized_qualifications(
        evaluator=evaluator,
        parent_node=ad_main,
        full_name=extracted.full_name,
        qual=extracted.qualifications,
        exp=extracted.experience,
    )

    # Return evaluation summary
    return evaluator.get_summary()