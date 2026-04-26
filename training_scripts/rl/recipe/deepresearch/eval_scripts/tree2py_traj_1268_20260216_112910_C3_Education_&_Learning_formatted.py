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
TASK_ID = "ecu_head_coach_career_progression"
TASK_DESCRIPTION = (
    "East Carolina University currently competes in the American Athletic Conference as an NCAA Division I FBS program. "
    "Identify the individual who currently serves as ECU's head football coach by verifying the following sequential career progression:\n\n"
    "1. Educational Background: The individual must have earned a bachelor's degree in physical education from Western Carolina University in 2002, followed by an MBA from Lenoir-Rhyne University in 2012.\n\n"
    "2. Early Coaching Foundation: The individual must have been part of a coaching staff at Lenoir-Rhyne University that helped the team reach the NCAA Division II National Championship Game in 2013, and must have subsequently been promoted to defensive coordinator at The Citadel in February 2016.\n\n"
    "3. Path to ECU: The individual must have been hired as defensive coordinator and inside linebackers coach at East Carolina University on January 29, 2020, then named interim head coach on October 20, 2024 (following the termination of the previous head coach Mike Houston), and finally officially named as East Carolina's 23rd head football coach on November 27, 2024.\n\n"
    "Provide the name of this individual along with URLs that verify each stage of this career progression."
)

# --------------------------------------------------------------------------- #
# Data models for structured extraction                                       #
# --------------------------------------------------------------------------- #
class CoachIdentity(BaseModel):
    name: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class DegreeInfo(BaseModel):
    degree: Optional[str] = None
    field: Optional[str] = None
    institution: Optional[str] = None
    year: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class RoleInfo(BaseModel):
    role: Optional[str] = None
    institution: Optional[str] = None
    date_or_year: Optional[str] = None   # e.g., "2001-2002", "2019"
    description: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class AppointmentInfo(BaseModel):
    title: Optional[str] = None
    institution: Optional[str] = None
    date: Optional[str] = None           # e.g., "January 29, 2020", "February 2016"
    extra: Optional[str] = None          # any extra descriptors (e.g., "23rd head coach")
    sources: List[str] = Field(default_factory=list)


class ConferenceStatus(BaseModel):
    conference: Optional[str] = None     # e.g., "American Athletic Conference" or "AAC"
    classification: Optional[str] = None # e.g., "NCAA Division I FBS"
    sources: List[str] = Field(default_factory=list)


class CareerExtraction(BaseModel):
    identity: Optional[CoachIdentity] = None

    undergrad: Optional[DegreeInfo] = None
    student_assistant: Optional[RoleInfo] = None
    grad_degree: Optional[DegreeInfo] = None

    lenoir_rhyne_achievement: Optional[RoleInfo] = None
    citadel_dc_promotion: Optional[AppointmentInfo] = None
    kennesaw_dc: Optional[RoleInfo] = None

    ecu_conference_status: Optional[ConferenceStatus] = None
    ecu_dc_appointment: Optional[AppointmentInfo] = None
    interim_appointment: Optional[AppointmentInfo] = None
    permanent_appointment: Optional[AppointmentInfo] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_career_progression() -> str:
    return """
Extract the following structured information from the answer text. Only extract what is explicitly stated in the answer. For each item that mentions or requires evidence, also extract any explicit URL(s) provided in the answer for that item.

Return a single JSON object with the following fields and subfields:

- identity:
  - name: The full name of the individual identified as East Carolina University's current head football coach.
  - sources: A list of explicit URL(s) in the answer that support the identity (if any).

- undergrad:
  - degree: The degree name (e.g., "Bachelor's", "B.S.").
  - field: The field/major (e.g., "physical education").
  - institution: The institution name (e.g., "Western Carolina University").
  - year: The year the degree was obtained (e.g., "2002").
  - sources: URL(s) in the answer supporting the undergrad degree.

- student_assistant:
  - role: The role title (e.g., "student assistant").
  - institution: The institution (e.g., "Western Carolina University").
  - date_or_year: The time period (e.g., "2001-2002").
  - description: Any brief description if present.
  - sources: URL(s) supporting the student assistant role.

- grad_degree:
  - degree: The degree name (e.g., "MBA", "Master of Business Administration").
  - field: If present.
  - institution: The institution (e.g., "Lenoir-Rhyne University").
  - year: The year obtained (e.g., "2012").
  - sources: URL(s) supporting the MBA degree.

- lenoir_rhyne_achievement:
  - role: The role or position the coach held at Lenoir-Rhyne, if stated.
  - institution: "Lenoir-Rhyne University".
  - date_or_year: The relevant year (e.g., "2013").
  - description: Must mention the team reached the NCAA Division II National Championship Game in 2013.
  - sources: URL(s) verifying this achievement.

- citadel_dc_promotion:
  - title: Must indicate promotion to defensive coordinator.
  - institution: "The Citadel".
  - date: The date/timeframe (e.g., "February 2016").
  - extra: Any additional descriptors if present.
  - sources: URL(s) verifying the February 2016 promotion.

- kennesaw_dc:
  - role: Must indicate defensive coordinator.
  - institution: "Kennesaw State University".
  - date_or_year: The year (e.g., "2019").
  - description: If present.
  - sources: URL(s) verifying the 2019 DC role at KSU.

- ecu_conference_status:
  - conference: e.g., "American Athletic Conference" or "AAC".
  - classification: e.g., "NCAA Division I FBS".
  - sources: URL(s) verifying ECU's conference/classification status.

- ecu_dc_appointment:
  - title: Must indicate defensive coordinator and inside linebackers coach.
  - institution: "East Carolina University" (or "ECU").
  - date: The date (e.g., "January 29, 2020").
  - extra: Any additional descriptors if present.
  - sources: URL(s) verifying this appointment.

- interim_appointment:
  - title: Should indicate interim head coach.
  - institution: "East Carolina University" (or "ECU").
  - date: The date (e.g., "October 20, 2024").
  - extra: Should mention that this followed Mike Houston's termination/firing/parting ways wording if given.
  - sources: URL(s) verifying the interim head coach appointment on Oct 20, 2024.

- permanent_appointment:
  - title: Should indicate head football coach.
  - institution: "East Carolina University" (or "ECU").
  - date: The date (e.g., "November 27, 2024").
  - extra: Should indicate "23rd head football coach" if given.
  - sources: URL(s) verifying the official appointment on Nov 27, 2024.

Extraction rules:
- Extract only URLs explicitly present in the answer. Do not invent any URL.
- If a field is missing, set it to null (for strings) or [] (for lists).
- Preserve textual details as strings rather than converting to numbers/dates.
"""


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _dedup_urls(urls: List[str]) -> List[str]:
    seen = set()
    result = []
    for u in urls:
        if not u:
            continue
        if u in seen:
            continue
        seen.add(u)
        result.append(u)
    return result


def _merge_sources(*lists: Optional[List[str]]) -> List[str]:
    merged: List[str] = []
    for lst in lists:
        if lst:
            merged.extend([u for u in lst if isinstance(u, str) and u.strip() != ""])
    return _dedup_urls(merged)


# --------------------------------------------------------------------------- #
# Verification logic                                                          #
# --------------------------------------------------------------------------- #
async def build_and_verify_career_progression(
    evaluator: Evaluator,
    parent_node,
    data: CareerExtraction
) -> None:
    """
    Build the verification tree and run all checks following the rubric, using the provided extracted data.
    All child nodes of critical parents are kept critical to satisfy framework constraints.
    """

    # Root critical sequential node representing the whole progression
    cpv_node = evaluator.add_sequential(
        id="Career_Progression_Verification",
        desc="Verify the complete career progression of East Carolina University's current head football coach through a sequential chain of educational credentials and professional positions",
        parent=parent_node,
        critical=True
    )

    # -------------------- Coach Identification ---------------------------- #
    coach_name = data.identity.name if (data and data.identity and data.identity.name) else ""
    identity_sources = data.identity.sources if (data and data.identity) else []
    ecu_apt_sources = _merge_sources(
        identity_sources,
        data.permanent_appointment.sources if data and data.permanent_appointment else [],
        data.interim_appointment.sources if data and data.interim_appointment else [],
        data.ecu_dc_appointment.sources if data and data.ecu_dc_appointment else []
    )

    coach_ident_leaf = evaluator.add_leaf(
        id="Coach_Identification",
        desc="Identify the individual by name who currently serves as ECU's head football coach",
        parent=cpv_node,
        critical=True
    )
    coach_ident_claim = f"The individual named '{coach_name}' is the current head football coach at East Carolina University."
    await evaluator.verify(
        claim=coach_ident_claim,
        node=coach_ident_leaf,
        sources=ecu_apt_sources,
        additional_instruction=(
            "Verify that the named person is ECU's current head football coach. Accept minor variations in phrasing. "
            "Use the provided URLs to confirm current status. If multiple pages are provided, only one needs to support the claim."
        )
    )

    # -------------------- Educational Foundation (sequential) ------------- #
    edu_node = evaluator.add_sequential(
        id="Educational_Foundation",
        desc="Verify the educational credentials and early coaching experience obtained before entering collegiate coaching",
        parent=cpv_node,
        critical=True
    )

    # Undergraduate Degree (parallel)
    ug_node = evaluator.add_parallel(
        id="Undergraduate_Degree",
        desc="Verify bachelor's degree in physical education from Western Carolina University obtained in 2002",
        parent=edu_node,
        critical=True
    )
    # Reference presence
    ug_ref = evaluator.add_custom_node(
        result=bool(data and data.undergrad and data.undergrad.sources and len(data.undergrad.sources) > 0),
        id="Undergraduate_Reference",
        desc="Provide a URL that verifies the undergraduate degree information",
        parent=ug_node,
        critical=True
    )
    # Details verification
    ug_details_leaf = evaluator.add_leaf(
        id="Undergraduate_Degree_Details",
        desc="The coach must have earned a bachelor's degree in physical education from Western Carolina University in 2002",
        parent=ug_node,
        critical=True
    )
    ug_degree = data.undergrad.degree if data and data.undergrad else ""
    ug_field = data.undergrad.field if data and data.undergrad else ""
    ug_inst = data.undergrad.institution if data and data.undergrad else ""
    ug_year = data.undergrad.year if data and data.undergrad else ""
    ug_claim = (
        f"{coach_name} earned a bachelor's degree ({ug_degree}) in {ug_field} from {ug_inst} in {ug_year}."
    )
    await evaluator.verify(
        claim=ug_claim,
        node=ug_details_leaf,
        sources=(data.undergrad.sources if data and data.undergrad else []),
        additional_instruction=(
            "Confirm the undergraduate credential and year. Allow common synonyms like 'Bachelor's'/'B.S.' "
            "and minor formatting differences. Institution should be Western Carolina University and year 2002."
        ),
        extra_prerequisites=[ug_ref]
    )

    # Student Assistant Role (parallel)
    sa_node = evaluator.add_parallel(
        id="Student_Assistant_Role",
        desc="Verify student assistant position at Western Carolina during undergraduate years",
        parent=edu_node,
        critical=True
    )
    sa_ref = evaluator.add_custom_node(
        result=bool(data and data.student_assistant and data.student_assistant.sources and len(data.student_assistant.sources) > 0),
        id="Student_Assistant_Reference",
        desc="Provide a URL that verifies the student assistant role at Western Carolina",
        parent=sa_node,
        critical=True
    )
    sa_leaf = evaluator.add_leaf(
        id="Student_Assistant_Details",
        desc="The coach must have served as a student assistant at Western Carolina University during 2001-2002",
        parent=sa_node,
        critical=True
    )
    sa_role = data.student_assistant.role if data and data.student_assistant else ""
    sa_inst = data.student_assistant.institution if data and data.student_assistant else ""
    sa_period = data.student_assistant.date_or_year if data and data.student_assistant else ""
    sa_claim = f"{coach_name} served as a {sa_role} at {sa_inst} during {sa_period}."
    await evaluator.verify(
        claim=sa_claim,
        node=sa_leaf,
        sources=(data.student_assistant.sources if data and data.student_assistant else []),
        additional_instruction=(
            "Verify the student assistant role and the time period (e.g., 2001–2002). Allow minor variations in the dash or spacing."
        ),
        extra_prerequisites=[sa_ref]
    )

    # Graduate Degree (parallel)
    gd_node = evaluator.add_parallel(
        id="Graduate_Degree",
        desc="Verify MBA from Lenoir-Rhyne University obtained in 2012",
        parent=edu_node,
        critical=True
    )
    gd_ref = evaluator.add_custom_node(
        result=bool(data and data.grad_degree and data.grad_degree.sources and len(data.grad_degree.sources) > 0),
        id="Graduate_Reference",
        desc="Provide a URL that verifies the graduate degree information",
        parent=gd_node,
        critical=True
    )
    gd_leaf = evaluator.add_leaf(
        id="Graduate_Degree_Details",
        desc="The coach must have earned an MBA from Lenoir-Rhyne University in 2012",
        parent=gd_node,
        critical=True
    )
    gd_degree = data.grad_degree.degree if data and data.grad_degree else ""
    gd_field = data.grad_degree.field if data and data.grad_degree else ""
    gd_inst = data.grad_degree.institution if data and data.grad_degree else ""
    gd_year = data.grad_degree.year if data and data.grad_degree else ""
    gd_claim = f"{coach_name} earned an {gd_degree} {('in ' + gd_field) if gd_field else ''} from {gd_inst} in {gd_year}."
    await evaluator.verify(
        claim=gd_claim,
        node=gd_leaf,
        sources=(data.grad_degree.sources if data and data.grad_degree else []),
        additional_instruction=(
            "Confirm this is an MBA from Lenoir-Rhyne in 2012. Accept 'Master of Business Administration' as equivalent to 'MBA'."
        ),
        extra_prerequisites=[gd_ref]
    )

    # -------------------- Early Coaching Foundation (sequential) ---------- #
    ecf_node = evaluator.add_sequential(
        id="Early_Coaching_Foundation",
        desc="Verify early coaching positions establishing the foundation for defensive coordinator roles",
        parent=cpv_node,
        critical=True
    )

    # Lenoir-Rhyne Achievement (parallel)
    lr_node = evaluator.add_parallel(
        id="Lenoir_Rhyne_Achievement",
        desc="Verify coaching role at Lenoir-Rhyne that led to 2013 NCAA Division II National Championship Game appearance",
        parent=ecf_node,
        critical=True
    )
    lr_ref = evaluator.add_custom_node(
        result=bool(data and data.lenoir_rhyne_achievement and data.lenoir_rhyne_achievement.sources and len(data.lenoir_rhyne_achievement.sources) > 0),
        id="Achievement_Reference",
        desc="Provide a URL that verifies the 2013 Lenoir-Rhyne NCAA Division II Championship Game achievement",
        parent=lr_node,
        critical=True
    )
    lr_leaf = evaluator.add_leaf(
        id="Championship_Game_Details",
        desc="The coach must have been part of a Lenoir-Rhyne University coaching staff that helped the team reach the NCAA Division II National Championship Game in 2013",
        parent=lr_node,
        critical=True
    )
    lr_inst = data.lenoir_rhyne_achievement.institution if data and data.lenoir_rhyne_achievement else "Lenoir-Rhyne University"
    lr_year = data.lenoir_rhyne_achievement.date_or_year if data and data.lenoir_rhyne_achievement else "2013"
    lr_claim = (
        f"{coach_name} was part of the {lr_inst} coaching staff that helped the team reach the NCAA Division II National Championship Game in {lr_year}."
    )
    await evaluator.verify(
        claim=lr_claim,
        node=lr_leaf,
        sources=(data.lenoir_rhyne_achievement.sources if data and data.lenoir_rhyne_achievement else []),
        additional_instruction="Confirm involvement on the Lenoir-Rhyne coaching staff and that the team reached the 2013 NCAA Division II National Championship Game.",
        extra_prerequisites=[lr_ref]
    )

    # First DC Promotion at The Citadel (parallel)
    cit_node = evaluator.add_parallel(
        id="First_DC_Promotion",
        desc="Verify promotion to defensive coordinator at The Citadel in February 2016",
        parent=ecf_node,
        critical=True
    )
    cit_ref = evaluator.add_custom_node(
        result=bool(data and data.citadel_dc_promotion and data.citadel_dc_promotion.sources and len(data.citadel_dc_promotion.sources) > 0),
        id="Citadel_Reference",
        desc="Provide a URL that verifies the February 2016 defensive coordinator promotion at The Citadel",
        parent=cit_node,
        critical=True
    )
    cit_leaf = evaluator.add_leaf(
        id="Citadel_DC_Details",
        desc="The coach must have been promoted to defensive coordinator at The Citadel in February 2016",
        parent=cit_node,
        critical=True
    )
    cit_date = data.citadel_dc_promotion.date if data and data.citadel_dc_promotion else "February 2016"
    cit_inst = data.citadel_dc_promotion.institution if data and data.citadel_dc_promotion else "The Citadel"
    cit_title = data.citadel_dc_promotion.title if data and data.citadel_dc_promotion else "defensive coordinator"
    cit_claim = f"{coach_name} was promoted to {cit_title} at {cit_inst} in {cit_date}."
    await evaluator.verify(
        claim=cit_claim,
        node=cit_leaf,
        sources=(data.citadel_dc_promotion.sources if data and data.citadel_dc_promotion else []),
        additional_instruction="Verify the promotion to defensive coordinator at The Citadel in February 2016.",
        extra_prerequisites=[cit_ref]
    )

    # Kennesaw State DC in 2019 (parallel)
    ksu_node = evaluator.add_parallel(
        id="Kennesaw_State_DC",
        desc="Verify defensive coordinator position at Kennesaw State University in 2019",
        parent=ecf_node,
        critical=True
    )
    ksu_ref = evaluator.add_custom_node(
        result=bool(data and data.kennesaw_dc and data.kennesaw_dc.sources and len(data.kennesaw_dc.sources) > 0),
        id="Kennesaw_Reference",
        desc="Provide a URL that verifies the 2019 defensive coordinator position at Kennesaw State",
        parent=ksu_node,
        critical=True
    )
    ksu_leaf = evaluator.add_leaf(
        id="Kennesaw_DC_Details",
        desc="The coach must have served as defensive coordinator at Kennesaw State University in 2019",
        parent=ksu_node,
        critical=True
    )
    ksu_role = data.kennesaw_dc.role if data and data.kennesaw_dc else "defensive coordinator"
    ksu_inst = data.kennesaw_dc.institution if data and data.kennesaw_dc else "Kennesaw State University"
    ksu_year = data.kennesaw_dc.date_or_year if data and data.kennesaw_dc else "2019"
    ksu_claim = f"{coach_name} served as {ksu_role} at {ksu_inst} in {ksu_year}."
    await evaluator.verify(
        claim=ksu_claim,
        node=ksu_leaf,
        sources=(data.kennesaw_dc.sources if data and data.kennesaw_dc else []),
        additional_instruction="Verify the defensive coordinator role at Kennesaw State University in 2019.",
        extra_prerequisites=[ksu_ref]
    )

    # -------------------- ECU Career Trajectory (sequential) -------------- #
    ecu_node = evaluator.add_sequential(
        id="ECU_Career_Trajectory",
        desc="Verify the coaching progression at East Carolina University from defensive coordinator to head coach",
        parent=cpv_node,
        critical=True
    )

    # ECU Conference Status (parallel)
    conf_node = evaluator.add_parallel(
        id="ECU_Conference_Status",
        desc="Verify that East Carolina University competes in the American Athletic Conference as an NCAA Division I FBS program",
        parent=ecu_node,
        critical=True
    )
    conf_ref = evaluator.add_custom_node(
        result=bool(data and data.ecu_conference_status and data.ecu_conference_status.sources and len(data.ecu_conference_status.sources) > 0),
        id="Conference_Reference",
        desc="Provide a URL that verifies ECU's conference and division status",
        parent=conf_node,
        critical=True
    )
    conf_leaf = evaluator.add_leaf(
        id="Conference_Status_Details",
        desc="East Carolina University must compete in the American Athletic Conference as an NCAA Division I FBS program",
        parent=conf_node,
        critical=True
    )
    conf_conf = data.ecu_conference_status.conference if data and data.ecu_conference_status else "American Athletic Conference"
    conf_class = data.ecu_conference_status.classification if data and data.ecu_conference_status else "NCAA Division I FBS"
    conf_claim = f"East Carolina University competes in the {conf_conf} and is part of {conf_class}."
    await evaluator.verify(
        claim=conf_claim,
        node=conf_leaf,
        sources=(data.ecu_conference_status.sources if data and data.ecu_conference_status else []),
        additional_instruction=(
            "Accept 'AAC' as equivalent to 'American Athletic Conference' and 'FBS' as equivalent to 'NCAA Division I FBS'."
        ),
        extra_prerequisites=[conf_ref]
    )

    # ECU DC Appointment (parallel)
    dc_node = evaluator.add_parallel(
        id="ECU_DC_Appointment",
        desc="Verify appointment as defensive coordinator at East Carolina on January 29, 2020",
        parent=ecu_node,
        critical=True
    )
    dc_ref = evaluator.add_custom_node(
        result=bool(data and data.ecu_dc_appointment and data.ecu_dc_appointment.sources and len(data.ecu_dc_appointment.sources) > 0),
        id="ECU_DC_Reference",
        desc="Provide a URL that verifies the January 29, 2020 defensive coordinator appointment at ECU",
        parent=dc_node,
        critical=True
    )
    dc_leaf = evaluator.add_leaf(
        id="ECU_DC_Details",
        desc="The coach must have been hired as defensive coordinator and inside linebackers coach at East Carolina University on January 29, 2020",
        parent=dc_node,
        critical=True
    )
    dc_title = data.ecu_dc_appointment.title if data and data.ecu_dc_appointment else "defensive coordinator and inside linebackers coach"
    dc_inst = data.ecu_dc_appointment.institution if data and data.ecu_dc_appointment else "East Carolina University"
    dc_date = data.ecu_dc_appointment.date if data and data.ecu_dc_appointment else "January 29, 2020"
    dc_claim = f"{coach_name} was hired as {dc_title} at {dc_inst} on {dc_date}."
    await evaluator.verify(
        claim=dc_claim,
        node=dc_leaf,
        sources=(data.ecu_dc_appointment.sources if data and data.ecu_dc_appointment else []),
        additional_instruction="Verify the ECU hiring on January 29, 2020, including both DC and inside linebackers coach responsibilities.",
        extra_prerequisites=[dc_ref]
    )

    # Head Coach Appointment (sequential)
    hc_node = evaluator.add_sequential(
        id="Head_Coach_Appointment",
        desc="Verify the two-stage process of becoming East Carolina's head football coach",
        parent=ecu_node,
        critical=True
    )

    # Interim Appointment (parallel)
    interim_node = evaluator.add_parallel(
        id="Interim_Appointment",
        desc="Verify interim head coach appointment on October 20, 2024, following Mike Houston's termination",
        parent=hc_node,
        critical=True
    )
    interim_ref = evaluator.add_custom_node(
        result=bool(data and data.interim_appointment and data.interim_appointment.sources and len(data.interim_appointment.sources) > 0),
        id="Interim_Reference",
        desc="Provide a URL that verifies the October 20, 2024 interim head coach appointment after Houston's firing",
        parent=interim_node,
        critical=True
    )
    interim_leaf = evaluator.add_leaf(
        id="Interim_Details",
        desc="The coach must have been named interim head coach at East Carolina on October 20, 2024, after the previous head coach Mike Houston was fired",
        parent=interim_node,
        critical=True
    )
    interim_date = data.interim_appointment.date if data and data.interim_appointment else "October 20, 2024"
    interim_inst = data.interim_appointment.institution if data and data.interim_appointment else "East Carolina University"
    interim_title = data.interim_appointment.title if data and data.interim_appointment else "interim head coach"
    interim_claim = (
        f"{coach_name} was named {interim_title} at {interim_inst} on {interim_date}, following Mike Houston's termination."
    )
    await evaluator.verify(
        claim=interim_claim,
        node=interim_leaf,
        sources=(data.interim_appointment.sources if data and data.interim_appointment else []),
        additional_instruction=(
            "Verify the interim appointment on Oct 20, 2024, and that it followed Mike Houston's removal. "
            "Treat 'fired', 'dismissed', 'terminated', or 'parted ways' as equivalent descriptions."
        ),
        extra_prerequisites=[interim_ref]
    )

    # Permanent Appointment (parallel)
    perm_node = evaluator.add_parallel(
        id="Permanent_Appointment",
        desc="Verify official appointment as ECU's 23rd head football coach on November 27, 2024",
        parent=hc_node,
        critical=True
    )
    perm_ref = evaluator.add_custom_node(
        result=bool(data and data.permanent_appointment and data.permanent_appointment.sources and len(data.permanent_appointment.sources) > 0),
        id="Permanent_Reference",
        desc="Provide a URL that verifies the November 27, 2024 official appointment as 23rd head coach",
        parent=perm_node,
        critical=True
    )
    perm_leaf = evaluator.add_leaf(
        id="Permanent_Details",
        desc="The coach must have been officially named as East Carolina's 23rd head football coach on November 27, 2024",
        parent=perm_node,
        critical=True
    )
    perm_date = data.permanent_appointment.date if data and data.permanent_appointment else "November 27, 2024"
    perm_inst = data.permanent_appointment.institution if data and data.permanent_appointment else "East Carolina University"
    perm_title = data.permanent_appointment.title if data and data.permanent_appointment else "head football coach"
    perm_extra = data.permanent_appointment.extra if data and data.permanent_appointment else "23rd head football coach"
    perm_claim = f"On {perm_date}, {coach_name} was officially named {perm_inst}'s {perm_extra} ({perm_title})."
    await evaluator.verify(
        claim=perm_claim,
        node=perm_leaf,
        sources=(data.permanent_appointment.sources if data and data.permanent_appointment else []),
        additional_instruction=(
            "Verify the official appointment on Nov 27, 2024, and that it identifies the individual as the 23rd head football coach."
        ),
        extra_prerequisites=[perm_ref]
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
    Evaluate an answer for the ECU head coach career progression verification task.
    Builds a sequential, fully grounded verification tree as specified by the rubric.
    """
    # Initialize evaluator with a sequential root
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.SEQUENTIAL,
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

    # Extract structured information
    extracted: CareerExtraction = await evaluator.extract(
        prompt=prompt_extract_career_progression(),
        template_class=CareerExtraction,
        extraction_name="career_progression_extraction"
    )

    # Build and verify the tree according to the rubric
    await build_and_verify_career_progression(evaluator, root, extracted)

    # Return summary
    return evaluator.get_summary()