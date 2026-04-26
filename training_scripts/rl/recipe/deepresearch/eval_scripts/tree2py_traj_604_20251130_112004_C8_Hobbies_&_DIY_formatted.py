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
TASK_ID = "heb_parade_volunteer_2025"
TASK_DESCRIPTION = """
You are interested in volunteering at the H-E-B Thanksgiving Day Parade in Houston, Texas on Thanksgiving Day 2025. Provide comprehensive information about this volunteer opportunity, including: the exact date and start time of the parade, all age requirements for volunteers, mandatory training session details (dates, times, and location), work hours required on parade day, available volunteer roles, registration method, and contact information for the volunteer coordinator.
"""


# --------------------------------------------------------------------------- #
# Data models for structured extraction                                       #
# --------------------------------------------------------------------------- #
class BasicEventInfo(BaseModel):
    name: Optional[str] = None
    city: Optional[str] = None
    state: Optional[str] = None
    date: Optional[str] = None
    start_time: Optional[str] = None
    route_location: Optional[str] = None


class AgeRequirements(BaseModel):
    general_min_age: Optional[str] = None
    marshal_min_age: Optional[str] = None


class TrainingInfo(BaseModel):
    required: Optional[str] = None
    session_one_date: Optional[str] = None
    session_one_time: Optional[str] = None
    session_two_date: Optional[str] = None
    session_two_time: Optional[str] = None
    location: Optional[str] = None


class ParticipationInfo(BaseModel):
    work_date: Optional[str] = None
    work_start_time: Optional[str] = None
    work_end_time: Optional[str] = None
    roles: List[str] = Field(default_factory=list)
    registration_method: Optional[str] = None
    signup_url: Optional[str] = None


class CoordinatorInfo(BaseModel):
    name: Optional[str] = None
    email: Optional[str] = None
    phone: Optional[str] = None


class VolunteerExtraction(BaseModel):
    basic: Optional[BasicEventInfo] = None
    age_requirements: Optional[AgeRequirements] = None
    training: Optional[TrainingInfo] = None
    participation: Optional[ParticipationInfo] = None
    coordinator: Optional[CoordinatorInfo] = None
    sources: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_volunteer_info() -> str:
    return """
    Extract comprehensive structured information about the volunteer opportunity for the H-E-B Thanksgiving Day Parade in Houston, Texas for Thanksgiving Day 2025, as presented in the answer.

    Return a JSON object with the following structure and fields. If any field is missing from the answer, return null for that field (or an empty list where appropriate). Do not invent information.

    {
      "basic": {
        "name": string | null,  // e.g., "H-E-B Thanksgiving Day Parade"
        "city": string | null,  // e.g., "Houston"
        "state": string | null, // e.g., "Texas" or "TX"
        "date": string | null,  // e.g., "Thursday, November 27, 2025"
        "start_time": string | null, // e.g., "9:00 a.m."
        "route_location": string | null // e.g., "downtown Houston business district"
      },
      "age_requirements": {
        "general_min_age": string | null, // e.g., "16+", "at least 16 years old"
        "marshal_min_age": string | null  // e.g., "18+", "at least 18 years old"
      },
      "training": {
        "required": string | null, // e.g., "All volunteers must attend one mandatory training session"
        "session_one_date": string | null, // e.g., "Thursday, November 20, 2025"
        "session_one_time": string | null, // e.g., "6:30 p.m."
        "session_two_date": string | null, // e.g., "Monday, November 24, 2025"
        "session_two_time": string | null, // e.g., "6:30 p.m."
        "location": string | null // e.g., "Legacy Room at City Hall, 901 Bagby Street, Houston"
      },
      "participation": {
        "work_date": string | null, // e.g., "Thursday, November 27, 2025"
        "work_start_time": string | null, // e.g., "6:00 a.m."
        "work_end_time": string | null, // e.g., "11:00 a.m."
        "roles": string[] | [], // e.g., ["Balloon Handlers", "Banner Carriers", ...]
        "registration_method": string | null, // e.g., "SignUp.com"
        "signup_url": string | null // e.g., a direct SignUp.com signup link URL if provided
      },
      "coordinator": {
        "name": string | null,
        "email": string | null,
        "phone": string | null
      },
      "sources": string[] // Extract ALL URLs explicitly mentioned in the answer (including any SignUp.com link). Return [] if none.
    }

    SPECIAL RULES FOR URL EXTRACTION:
    - Extract only URLs explicitly present in the answer text. Do not invent or infer URLs.
    - Accept plain URLs or markdown links. Always include the full URL with protocol.
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _all_urls(extracted: VolunteerExtraction) -> List[str]:
    urls: List[str] = []
    if extracted and extracted.sources:
        urls.extend([u for u in extracted.sources if isinstance(u, str) and u.strip() != ""])
    if extracted and extracted.participation and extracted.participation.signup_url:
        su = extracted.participation.signup_url.strip()
        if su and su not in urls:
            urls.append(su)
    # Deduplicate while preserving order
    seen = set()
    unique = []
    for u in urls:
        if u not in seen:
            seen.add(u)
            unique.append(u)
    return unique


def _nonempty(s: Optional[str]) -> bool:
    return bool(s and str(s).strip())


# --------------------------------------------------------------------------- #
# Verification subtrees                                                       #
# --------------------------------------------------------------------------- #
async def verify_basic_event_information(
    evaluator: Evaluator,
    parent_node,
    extracted: VolunteerExtraction,
) -> None:
    node = evaluator.add_parallel(
        id="basic_event_information",
        desc="Core parade event details including identity/name, date, start time, and route/location",
        parent=parent_node,
        critical=True,
    )

    urls = _all_urls(extracted)
    basic = extracted.basic if extracted and extracted.basic else BasicEventInfo()

    # Existence checks (critical)
    evaluator.add_custom_node(
        result=_nonempty(basic.name) and _nonempty(basic.city) and _nonempty(basic.state),
        id="parade_identity_provided",
        desc="Parade identity (name and city/state) is provided in the answer",
        parent=node,
        critical=True,
    )
    evaluator.add_custom_node(
        result=_nonempty(basic.date),
        id="parade_date_provided",
        desc="Parade date is provided in the answer",
        parent=node,
        critical=True,
    )
    evaluator.add_custom_node(
        result=_nonempty(basic.start_time),
        id="parade_start_time_provided",
        desc="Parade start time is provided in the answer",
        parent=node,
        critical=True,
    )
    evaluator.add_custom_node(
        result=_nonempty(basic.route_location),
        id="parade_route_location_provided",
        desc="Parade route/location is provided in the answer",
        parent=node,
        critical=True,
    )

    # Leaf verifications (critical)
    # 1) Identity
    leaf_identity = evaluator.add_leaf(
        id="parade_identity",
        desc="Parade is identified as the H-E-B Thanksgiving Day Parade in Houston, Texas",
        parent=node,
        critical=True,
    )
    claim_identity = "The parade is the H-E-B Thanksgiving Day Parade in Houston, Texas."
    await evaluator.verify(
        claim=claim_identity,
        node=leaf_identity,
        sources=urls,
        additional_instruction="Verify that the event name clearly matches 'H-E-B Thanksgiving Day Parade' and the location is in Houston, Texas.",
    )

    # 2) Date
    leaf_date = evaluator.add_leaf(
        id="parade_date",
        desc="Parade date is Thursday, November 27, 2025 (Thanksgiving Day 2025)",
        parent=node,
        critical=True,
    )
    claim_date = "The parade date is Thursday, November 27, 2025 (Thanksgiving Day 2025)."
    await evaluator.verify(
        claim=claim_date,
        node=leaf_date,
        sources=urls,
        additional_instruction="Check the official event information to confirm the 2025 date listed above.",
    )

    # 3) Start time
    leaf_start = evaluator.add_leaf(
        id="parade_start_time",
        desc="Parade starts at 9:00 a.m.",
        parent=node,
        critical=True,
    )
    claim_start = "The parade starts at 9:00 a.m."
    await evaluator.verify(
        claim=claim_start,
        node=leaf_start,
        sources=urls,
        additional_instruction="Verify the published start time on the official or authoritative event pages.",
    )

    # 4) Route/location
    leaf_route = evaluator.add_leaf(
        id="parade_route_location",
        desc="Parade route is in downtown Houston business district",
        parent=node,
        critical=True,
    )
    claim_route = "The parade route is in the downtown Houston business district."
    await evaluator.verify(
        claim=claim_route,
        node=leaf_route,
        sources=urls,
        additional_instruction="Confirm the described parade route location on authoritative sources.",
    )


async def verify_age_requirements(
    evaluator: Evaluator,
    parent_node,
    extracted: VolunteerExtraction,
) -> None:
    node = evaluator.add_parallel(
        id="volunteer_age_requirements",
        desc="Age requirements for different volunteer positions",
        parent=parent_node,
        critical=True,
    )

    urls = _all_urls(extracted)
    ages = extracted.age_requirements if extracted and extracted.age_requirements else AgeRequirements()

    # Existence checks
    evaluator.add_custom_node(
        result=_nonempty(ages.general_min_age),
        id="general_volunteer_age_provided",
        desc="General minimum volunteer age is provided in the answer",
        parent=node,
        critical=True,
    )
    evaluator.add_custom_node(
        result=_nonempty(ages.marshal_min_age),
        id="marshal_age_provided",
        desc="Parade Marshal minimum age is provided in the answer",
        parent=node,
        critical=True,
    )

    # Leaf verifications
    leaf_general = evaluator.add_leaf(
        id="general_volunteer_age",
        desc="Minimum age requirement for general volunteers is at least 16 years old",
        parent=node,
        critical=True,
    )
    claim_general = "The minimum age requirement for general volunteers is at least 16 years old."
    await evaluator.verify(
        claim=claim_general,
        node=leaf_general,
        sources=urls,
        additional_instruction="Confirm that general volunteers must be 16 years of age or older.",
    )

    leaf_marshal = evaluator.add_leaf(
        id="marshal_age",
        desc="Minimum age requirement for Parade Marshals is at least 18 years old",
        parent=node,
        critical=True,
    )
    claim_marshal = "The minimum age requirement for Parade Marshals is at least 18 years old."
    await evaluator.verify(
        claim=claim_marshal,
        node=leaf_marshal,
        sources=urls,
        additional_instruction="Confirm that Parade Marshals must be 18 years of age or older.",
    )


async def verify_training_requirements(
    evaluator: Evaluator,
    parent_node,
    extracted: VolunteerExtraction,
) -> None:
    node = evaluator.add_parallel(
        id="training_requirements",
        desc="Mandatory training session requirements and details",
        parent=parent_node,
        critical=True,
    )

    urls = _all_urls(extracted)
    training = extracted.training if extracted and extracted.training else TrainingInfo()

    # Existence checks
    evaluator.add_custom_node(
        result=_nonempty(training.required),
        id="training_requirement_provided",
        desc="Statement about mandatory training requirement is provided in the answer",
        parent=node,
        critical=True,
    )
    evaluator.add_custom_node(
        result=_nonempty(training.session_one_date) and _nonempty(training.session_one_time),
        id="training_session_one_provided",
        desc="Training session option 1 date and time are provided in the answer",
        parent=node,
        critical=True,
    )
    evaluator.add_custom_node(
        result=_nonempty(training.session_two_date) and _nonempty(training.session_two_time),
        id="training_session_two_provided",
        desc="Training session option 2 date and time are provided in the answer",
        parent=node,
        critical=True,
    )
    evaluator.add_custom_node(
        result=_nonempty(training.location),
        id="training_location_provided",
        desc="Training session location is provided in the answer",
        parent=node,
        critical=True,
    )

    # Leaf verifications
    leaf_req = evaluator.add_leaf(
        id="training_requirement",
        desc="ALL volunteer positions must attend ONE mandatory training session",
        parent=node,
        critical=True,
    )
    claim_req = "All volunteer positions must attend one mandatory training session."
    await evaluator.verify(
        claim=claim_req,
        node=leaf_req,
        sources=urls,
        additional_instruction="Verify the mandatory nature of training for volunteers.",
    )

    leaf_s1 = evaluator.add_leaf(
        id="training_session_one",
        desc="Training session option 1 is Thursday, November 20, 2025 at 6:30 p.m.",
        parent=node,
        critical=True,
    )
    claim_s1 = "One training session option is Thursday, November 20, 2025 at 6:30 p.m."
    await evaluator.verify(
        claim=claim_s1,
        node=leaf_s1,
        sources=urls,
        additional_instruction="Confirm the exact date and time for the first training session option.",
    )

    leaf_s2 = evaluator.add_leaf(
        id="training_session_two",
        desc="Training session option 2 is Monday, November 24, 2025 at 6:30 p.m.",
        parent=node,
        critical=True,
    )
    claim_s2 = "Another training session option is Monday, November 24, 2025 at 6:30 p.m."
    await evaluator.verify(
        claim=claim_s2,
        node=leaf_s2,
        sources=urls,
        additional_instruction="Confirm the exact date and time for the second training session option.",
    )

    leaf_loc = evaluator.add_leaf(
        id="training_location",
        desc="Training sessions take place in the Legacy Room at City Hall, 901 Bagby Street, Houston",
        parent=node,
        critical=True,
    )
    claim_loc = "Training sessions take place in the Legacy Room at City Hall, 901 Bagby Street, Houston."
    await evaluator.verify(
        claim=claim_loc,
        node=leaf_loc,
        sources=urls,
        additional_instruction="Verify the precise training location address and room.",
    )


async def verify_participation_logistics(
    evaluator: Evaluator,
    parent_node,
    extracted: VolunteerExtraction,
) -> None:
    node = evaluator.add_parallel(
        id="participation_logistics",
        desc="Work requirements, available roles, and registration process",
        parent=parent_node,
        critical=True,
    )

    urls = _all_urls(extracted)
    participation = extracted.participation if extracted and extracted.participation else ParticipationInfo()

    # Existence checks
    evaluator.add_custom_node(
        result=_nonempty(participation.work_start_time) and _nonempty(participation.work_end_time),
        id="volunteer_work_hours_provided",
        desc="Volunteer work hours are provided in the answer",
        parent=node,
        critical=True,
    )
    evaluator.add_custom_node(
        result=bool(participation.roles and len(participation.roles) > 0),
        id="volunteer_roles_provided",
        desc="At least one volunteer role is listed in the answer",
        parent=node,
        critical=True,
    )
    evaluator.add_custom_node(
        result=_nonempty(participation.signup_url) and ("signup.com" in (participation.signup_url or "").lower()),
        id="registration_signup_url_provided",
        desc="A concrete SignUp.com signup URL is provided in the answer",
        parent=node,
        critical=True,
    )

    # Leaf verifications
    leaf_hours = evaluator.add_leaf(
        id="volunteer_work_hours",
        desc="All volunteer positions must work on Thursday, November 27, 2025 from 6:00 a.m. to 11:00 a.m.",
        parent=node,
        critical=True,
    )
    claim_hours = "All volunteer positions must work on Thursday, November 27, 2025 from 6:00 a.m. to 11:00 a.m."
    await evaluator.verify(
        claim=claim_hours,
        node=leaf_hours,
        sources=urls,
        additional_instruction="Confirm the stated volunteer commitment window on parade day.",
    )

    leaf_roles = evaluator.add_leaf(
        id="volunteer_roles",
        desc="Available volunteer roles include Balloon Handlers, Banner Carriers, Parade Marshals, Seating Ushers, ADA Ushers, and Back Lot helpers",
        parent=node,
        critical=True,
    )
    claim_roles = (
        "Available volunteer roles include Balloon Handlers, Banner Carriers, Parade Marshals, "
        "Seating Ushers, ADA Ushers, and Back Lot helpers."
    )
    await evaluator.verify(
        claim=claim_roles,
        node=leaf_roles,
        sources=urls,
        additional_instruction="Verify that these specific roles are listed among the volunteer opportunities.",
    )

    leaf_reg = evaluator.add_leaf(
        id="registration_method",
        desc="Registration is through SignUp.com and the answer provides a concrete SignUp.com signup URL",
        parent=node,
        critical=True,
    )
    claim_reg = "Registration for this volunteer opportunity is through SignUp.com."
    # Prefer the signup URL directly as strongest source if available
    reg_sources = [participation.signup_url] if participation and participation.signup_url else urls
    await evaluator.verify(
        claim=claim_reg,
        node=leaf_reg,
        sources=reg_sources,
        additional_instruction="Check that the referenced registration platform is SignUp.com.",
    )


async def verify_coordinator_contact_information(
    evaluator: Evaluator,
    parent_node,
    extracted: VolunteerExtraction,
) -> None:
    node = evaluator.add_parallel(
        id="coordinator_contact_information",
        desc="Volunteer coordinator contact information",
        parent=parent_node,
        critical=True,
    )

    urls = _all_urls(extracted)
    coord = extracted.coordinator if extracted and extracted.coordinator else CoordinatorInfo()

    # Existence checks
    name_present = _nonempty(coord.name)
    email_present = _nonempty(coord.email)
    phone_present = _nonempty(coord.phone)

    evaluator.add_custom_node(
        result=name_present,
        id="coordinator_name_provided",
        desc="Volunteer coordinator name is provided in the answer",
        parent=node,
        critical=True,
    )
    evaluator.add_custom_node(
        result=email_present,
        id="coordinator_email_provided",
        desc="Volunteer coordinator email is provided in the answer",
        parent=node,
        critical=True,
    )
    evaluator.add_custom_node(
        result=phone_present,
        id="coordinator_phone_provided",
        desc="Volunteer coordinator phone number is provided in the answer",
        parent=node,
        critical=True,
    )

    # Leaf verifications referencing the provided values (gated by existence via critical siblings)
    leaf_name = evaluator.add_leaf(
        id="coordinator_name",
        desc="Volunteer coordinator name is provided",
        parent=node,
        critical=True,
    )
    claim_name = f"The volunteer coordinator's name is {coord.name}." if name_present else "The volunteer coordinator's name is provided."
    await evaluator.verify(
        claim=claim_name,
        node=leaf_name,
        sources=urls,
        additional_instruction="Verify that the stated person is explicitly identified as the volunteer coordinator or primary volunteer contact for the parade.",
    )

    leaf_email = evaluator.add_leaf(
        id="coordinator_email",
        desc="Volunteer coordinator email is provided",
        parent=node,
        critical=True,
    )
    claim_email = f"The volunteer coordinator's email is {coord.email}." if email_present else "The volunteer coordinator email is provided."
    await evaluator.verify(
        claim=claim_email,
        node=leaf_email,
        sources=urls,
        additional_instruction="Verify that this email is listed as the volunteer coordinator/contact email for the parade.",
    )

    leaf_phone = evaluator.add_leaf(
        id="coordinator_phone",
        desc="Volunteer coordinator phone number is provided",
        parent=node,
        critical=True,
    )
    claim_phone = f"The volunteer coordinator's phone number is {coord.phone}." if phone_present else "The volunteer coordinator phone number is provided."
    await evaluator.verify(
        claim=claim_phone,
        node=leaf_phone,
        sources=urls,
        additional_instruction="Verify that this phone number is listed as the volunteer coordinator/contact phone for the parade.",
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

    # A critical main node representing the rubric root (since Evaluator.root is non-critical by design)
    main = evaluator.add_parallel(
        id="heb_parade_volunteer_main",
        desc="Comprehensive information about the H-E-B Thanksgiving Day Parade volunteer opportunity in Houston, Texas for Thanksgiving 2025",
        parent=root,
        critical=True,
    )

    # Extract structured information from the answer
    extracted: VolunteerExtraction = await evaluator.extract(
        prompt=prompt_extract_volunteer_info(),
        template_class=VolunteerExtraction,
        extraction_name="volunteer_info_extraction",
    )

    # Add optional ground truth expectations for transparency (not used for scoring)
    evaluator.add_ground_truth({
        "expected_core_facts": [
            "Parade identity: H-E-B Thanksgiving Day Parade in Houston, Texas",
            "Parade date: Thursday, November 27, 2025",
            "Parade start time: 9:00 a.m.",
            "Route location: downtown Houston business district",
            "General volunteer minimum age: at least 16",
            "Parade Marshal minimum age: at least 18",
            "Mandatory training: attend one session",
            "Training options: Thu Nov 20, 2025 6:30 p.m.; Mon Nov 24, 2025 6:30 p.m.",
            "Training location: Legacy Room at City Hall, 901 Bagby St, Houston",
            "Parade day work hours: Thu Nov 27, 2025 6:00 a.m.–11:00 a.m.",
            "Roles include: Balloon Handlers, Banner Carriers, Parade Marshals, Seating Ushers, ADA Ushers, Back Lot helpers",
            "Registration via SignUp.com with concrete signup link",
            "Coordinator contact: name, email, and phone"
        ]
    })

    # Build the verification tree according to rubric
    await verify_basic_event_information(evaluator, main, extracted)
    await verify_age_requirements(evaluator, main, extracted)
    await verify_training_requirements(evaluator, main, extracted)
    await verify_participation_logistics(evaluator, main, extracted)
    await verify_coordinator_contact_information(evaluator, main, extracted)

    return evaluator.get_summary()