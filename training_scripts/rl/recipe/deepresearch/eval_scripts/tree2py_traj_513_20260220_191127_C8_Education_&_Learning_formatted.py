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
TASK_ID = "edu_admins_us_2026"
TASK_DESCRIPTION = (
    "Identify four current or recent educational administrators in the United States who each meet all of the "
    "following respective criteria as of February 2026:\n\n"
    "Administrator 1:\n"
    "- Holds or held a superintendent position in a North Carolina school district\n"
    "- The district must be one of the two largest school districts in North Carolina by student enrollment\n"
    "- Holds a doctoral degree (Ed.D. or Ph.D.)\n"
    "- Was appointed to their current superintendent position between 2022-2024 (inclusive)\n\n"
    "Administrator 2:\n"
    "- Serves as executive director of a state-level high school activities or athletics association\n"
    "- The association is located in Mississippi\n"
    "- Began serving as executive director in January 2021\n"
    "- Succeeded an executive director whose first name begins with the letter 'D'\n\n"
    "Administrator 3:\n"
    "- Works at a university that participates in NCAA Division II athletics\n"
    "- The university is located in New York State\n"
    "- The university has a campus in Westchester County, New York\n"
    "- Serves or served as Director of Athletics at the institution\n\n"
    "Administrator 4:\n"
    "- Works at a high school (serving grades 9-12 or similar)\n"
    "- The high school is located in Georgia\n"
    "- The school's football team won a state championship between 2010-2015 (inclusive)\n"
    "- Holds a position as principal, assistant principal, or athletic director at the school\n\n"
    "For each administrator, provide:\n"
    "- Full name (including titles and credentials)\n"
    "- Current position title\n"
    "- Institution/organization name\n"
    "- A reference URL that verifies their qualifications and meeting the specified criteria"
)


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class Administrator(BaseModel):
    # Core requested fields
    full_name: Optional[str] = None
    position_title: Optional[str] = None
    organization: Optional[str] = None
    reference_urls: List[str] = Field(default_factory=list)

    # Admin 1-specific fields
    district_name: Optional[str] = None
    district_state: Optional[str] = None
    doctoral_degree: Optional[str] = None
    appointment_month_year: Optional[str] = None
    appointment_year: Optional[str] = None

    # Admin 2-specific fields
    association_name: Optional[str] = None
    association_state: Optional[str] = None
    start_month_year: Optional[str] = None
    predecessor_name: Optional[str] = None

    # Admin 3-specific fields
    university_name: Optional[str] = None
    ncaa_division: Optional[str] = None
    university_state: Optional[str] = None
    campus_county: Optional[str] = None
    athletics_role: Optional[str] = None

    # Admin 4-specific fields
    school_name: Optional[str] = None
    school_state: Optional[str] = None
    football_championship_year: Optional[str] = None
    school_position: Optional[str] = None
    school_level: Optional[str] = None


class AdminListExtraction(BaseModel):
    administrators: List[Administrator] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_administrators() -> str:
    return (
        "Extract up to the first four educational administrators mentioned in the answer, capturing the following "
        "fields for each administrator. If any field is missing, return null or an empty list as appropriate.\n\n"
        "For each administrator, extract:\n"
        "1) full_name: The person's full name, including titles and credentials if provided (e.g., 'Dr. Jane Smith, Ed.D.')\n"
        "2) position_title: The person's current role/title as stated (e.g., 'Superintendent', 'Executive Director', 'Director of Athletics', 'Principal')\n"
        "3) organization: The name of the institution/organization (e.g., 'Charlotte-Mecklenburg Schools', 'MHSAA', 'Pace University', 'Buford High School')\n"
        "4) reference_urls: An array of URLs explicitly provided in the answer that verify the administrator meeting the relevant criteria. "
        "   Extract only actual URLs (including markdown links). If none are provided, return an empty array.\n\n"
        "Additionally, extract any of the following criteria-specific fields IF present in the answer (otherwise null):\n"
        "- For Superintendent (NC) case: district_name, district_state, doctoral_degree, appointment_month_year, appointment_year\n"
        "- For State association ED case: association_name, association_state, start_month_year, predecessor_name\n"
        "- For NCAA Division II athletics case: university_name, ncaa_division, university_state, campus_county, athletics_role\n"
        "- For Georgia high school case: school_name, school_state, football_championship_year, school_position, school_level\n\n"
        "Return a JSON object with a single key 'administrators', an array of objects with these fields."
    )


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def _nonempty_str(s: Optional[str]) -> bool:
    return bool(s and str(s).strip())


def _sources(admin: Administrator) -> List[str]:
    # Ensure we pass a list for URL verifications
    return admin.reference_urls if admin.reference_urls else []


# --------------------------------------------------------------------------- #
# Verification functions per administrator                                   #
# --------------------------------------------------------------------------- #
async def verify_administrator_1(evaluator: Evaluator, parent_node, admin: Administrator) -> None:
    """
    Administrator 1:
    - Superintendent in North Carolina
    - District is one of the two largest by enrollment (NC)
    - Holds doctoral degree (Ed.D. or Ph.D.)
    - Appointed 2022-2024 inclusive
    - Provides verifying reference URL
    """
    node = evaluator.add_parallel(
        id="administrator_1",
        desc="First administrator meeting all requirements",
        parent=parent_node,
        critical=False
    )

    # Basic required info and reference presence (pre-checks)
    req_info = evaluator.add_custom_node(
        result=(_nonempty_str(admin.full_name) and _nonempty_str(admin.position_title) and _nonempty_str(admin.organization)),
        id="admin1_required_info",
        desc="Administrator 1 has basic info (name, position, organization)",
        parent=node,
        critical=True
    )
    ref_present = evaluator.add_custom_node(
        result=(len(admin.reference_urls) > 0),
        id="admin1_reference_present",
        desc="Administrator 1 provides at least one reference URL",
        parent=node,
        critical=True
    )

    # 1) Superintendent in NC
    pos_leaf = evaluator.add_leaf(
        id="admin1_position",
        desc="Holds or held a superintendent position in a North Carolina school district",
        parent=node,
        critical=True
    )
    name = admin.full_name or "the administrator"
    district = admin.district_name or admin.organization or "the district"
    claim_pos = (
        f"The provided source shows that {name} holds or held the position of Superintendent "
        f"(including titles like 'Superintendent of Schools' or 'Interim Superintendent') for {district} "
        f"in North Carolina."
    )
    await evaluator.verify(
        claim=claim_pos,
        node=pos_leaf,
        sources=_sources(admin),
        additional_instruction="Allow reasonable title variants (Superintendent, Interim Superintendent). "
                               "Confirm the district is in North Carolina; minor naming variations are acceptable."
    )

    # 2) District is one of the two largest NC districts by enrollment
    size_leaf = evaluator.add_leaf(
        id="admin1_district_size",
        desc="The district is one of the two largest school districts in North Carolina by enrollment",
        parent=node,
        critical=True
    )
    claim_size = (
        f"The provided source(s) explicitly indicate that {district} is one of the two largest school districts "
        f"in North Carolina by student enrollment (e.g., Wake County Public School System or Charlotte-Mecklenburg Schools)."
    )
    await evaluator.verify(
        claim=claim_size,
        node=size_leaf,
        sources=_sources(admin),
        additional_instruction="Look for explicit statements or credible evidence about district enrollment ranking. "
                               "Accept synonyms and references to 'largest', 'second-largest', or equivalent phrasing."
    )

    # 3) Holds doctoral degree (Ed.D. or Ph.D.)
    edu_leaf = evaluator.add_leaf(
        id="admin1_education",
        desc="Holds a doctoral degree (Ed.D. or Ph.D.)",
        parent=node,
        critical=True
    )
    doc_degree = admin.doctoral_degree or "a doctoral degree"
    claim_degree = (
        f"The provided source shows that {name} holds a doctoral degree, such as Ed.D./Doctor of Education or Ph.D./Doctor of Philosophy."
    )
    await evaluator.verify(
        claim=claim_degree,
        node=edu_leaf,
        sources=_sources(admin),
        additional_instruction="Accept variations like 'EdD', 'Ed.D.', 'Doctor of Education', 'Ph.D.', 'PhD', "
                               "and nearby phrasing that clearly indicates a doctoral degree."
    )

    # 4) Appointed within 2022–2024 inclusive
    appoint_leaf = evaluator.add_leaf(
        id="admin1_appointment",
        desc="Was appointed to their current superintendent position between 2022-2024 (inclusive)",
        parent=node,
        critical=True
    )
    appoint_year_text = admin.appointment_year or admin.appointment_month_year or "a date"
    claim_appoint = (
        f"The provided source shows that {name} was appointed as superintendent in 2022, 2023, or 2024 "
        f"(appointment mentioned as {appoint_year_text} if available)."
    )
    await evaluator.verify(
        claim=claim_appoint,
        node=appoint_leaf,
        sources=_sources(admin),
        additional_instruction="Confirm the appointment occurred during 2022–2024 inclusive; "
                               "accept announcement or board approval dates indicating the start."
    )

    # 5) Reference URL confirms position and qualifications
    ref_leaf = evaluator.add_leaf(
        id="admin1_reference",
        desc="Provides a verifiable reference URL confirming the administrator's position and qualifications",
        parent=node,
        critical=True
    )
    claim_ref = (
        f"The provided source page(s) mention {name} and confirm their superintendent role at {district} "
        f"and relevant qualifications (e.g., doctoral degree or appointment details)."
    )
    await evaluator.verify(
        claim=claim_ref,
        node=ref_leaf,
        sources=_sources(admin),
        additional_instruction="The page should clearly identify the person and their role at the district; "
                               "minor formatting differences in names/titles are acceptable."
    )


async def verify_administrator_2(evaluator: Evaluator, parent_node, admin: Administrator) -> None:
    """
    Administrator 2:
    - Executive Director of state-level HS activities/athletics association
    - Located in Mississippi
    - Began serving in January 2021
    - Succeeded executive director whose first name begins with 'D'
    - Provides verifying reference URL
    """
    node = evaluator.add_parallel(
        id="administrator_2",
        desc="Second administrator meeting all requirements",
        parent=parent_node,
        critical=False
    )

    req_info = evaluator.add_custom_node(
        result=_nonempty_str(admin.full_name) and _nonempty_str(admin.position_title) and (_nonempty_str(admin.organization) or _nonempty_str(admin.association_name)),
        id="admin2_required_info",
        desc="Administrator 2 has basic info (name, position, association/organization)",
        parent=node,
        critical=True
    )
    ref_present = evaluator.add_custom_node(
        result=(len(admin.reference_urls) > 0),
        id="admin2_reference_present",
        desc="Administrator 2 provides at least one reference URL",
        parent=node,
        critical=True
    )

    # 1) Executive Director of state-level HS activities/athletics association
    org_leaf = evaluator.add_leaf(
        id="admin2_organization",
        desc="Serves as executive director of a state-level high school activities/athletics association",
        parent=node,
        critical=True
    )
    name = admin.full_name or "the administrator"
    assoc = admin.association_name or admin.organization or "the association"
    claim_org = (
        f"The provided source shows that {name} serves as Executive Director of {assoc}, "
        f"which is a state-level high school activities or athletics association."
    )
    await evaluator.verify(
        claim=claim_org,
        node=org_leaf,
        sources=_sources(admin),
        additional_instruction="Accept variants like 'Executive Director' or 'ED'. Confirm it is a state high school association "
                               "(e.g., activities/athletics)."
    )

    # 2) Association located in Mississippi
    state_leaf = evaluator.add_leaf(
        id="admin2_state",
        desc="The association is located in Mississippi",
        parent=node,
        critical=True
    )
    assoc_state = admin.association_state or "Mississippi"
    claim_state = (
        f"The provided source indicates {assoc} is the Mississippi state association or is based in Mississippi."
    )
    await evaluator.verify(
        claim=claim_state,
        node=state_leaf,
        sources=_sources(admin),
        additional_instruction="Confirm Mississippi presence (e.g., 'Mississippi High School Activities Association'). "
                               "Accept 'MS' abbreviations and clear Mississippi context."
    )

    # 3) Began serving in January 2021
    start_leaf = evaluator.add_leaf(
        id="admin2_appointment_year",
        desc="Began serving as executive director in January 2021",
        parent=node,
        critical=True
    )
    start_m_y = admin.start_month_year or "January 2021"
    claim_start = (
        f"The provided source shows that {name} began serving as Executive Director in January 2021 "
        f"(e.g., appointment or start date stated as {start_m_y})."
    )
    await evaluator.verify(
        claim=claim_start,
        node=start_leaf,
        sources=_sources(admin),
        additional_instruction="Confirm the start month and year as January 2021; accept announcement dates clearly indicating start timing."
    )

    # 4) Succeeded an ED whose first name starts with 'D'
    pred_leaf = evaluator.add_leaf(
        id="admin2_predecessor",
        desc="Succeeded an executive director whose first name begins with 'D'",
        parent=node,
        critical=True
    )
    pred_name = admin.predecessor_name or "a predecessor whose first name starts with 'D'"
    claim_pred = (
        f"The provided source indicates that {name} succeeded an Executive Director whose first name begins with 'D' "
        f"(e.g., {pred_name})."
    )
    await evaluator.verify(
        claim=claim_pred,
        node=pred_leaf,
        sources=_sources(admin),
        additional_instruction="Look for explicit mentions of the predecessor name. Accept common 'D' names such as Don, Dana, David, Dylan, etc."
    )

    # 5) Reference URL confirms appointment details
    ref_leaf = evaluator.add_leaf(
        id="admin2_reference",
        desc="Provides a verifiable reference URL confirming the executive director's appointment details",
        parent=node,
        critical=True
    )
    claim_ref = (
        f"The provided source page(s) mention {name}, confirm their Executive Director role at {assoc}, "
        f"and include appointment details (start date and predecessor context)."
    )
    await evaluator.verify(
        claim=claim_ref,
        node=ref_leaf,
        sources=_sources(admin),
        additional_instruction="Ensure the page connects the person, role, and timing/predecessor information; "
                               "minor formatting differences are acceptable."
    )


async def verify_administrator_3(evaluator: Evaluator, parent_node, admin: Administrator) -> None:
    """
    Administrator 3:
    - University participates in NCAA Division II
    - University is located in New York State
    - University has a campus in Westchester County, New York
    - Administrator serves/served as Director of Athletics
    - Provides verifying reference URL
    """
    node = evaluator.add_parallel(
        id="administrator_3",
        desc="Third administrator meeting all requirements",
        parent=parent_node,
        critical=False
    )

    req_info = evaluator.add_custom_node(
        result=_nonempty_str(admin.full_name) and _nonempty_str(admin.position_title) and (_nonempty_str(admin.organization) or _nonempty_str(admin.university_name)),
        id="admin3_required_info",
        desc="Administrator 3 has basic info (name, position, university/institution)",
        parent=node,
        critical=True
    )
    ref_present = evaluator.add_custom_node(
        result=(len(admin.reference_urls) > 0),
        id="admin3_reference_present",
        desc="Administrator 3 provides at least one reference URL",
        parent=node,
        critical=True
    )

    # 1) NCAA Division II participant
    inst_leaf = evaluator.add_leaf(
        id="admin3_institution_type",
        desc="Works at a university that participates in NCAA Division II athletics",
        parent=node,
        critical=True
    )
    uni = admin.university_name or admin.organization or "the university"
    claim_div2 = (
        f"The provided source indicates that {uni} participates in NCAA Division II athletics (Division II, DII, or equivalent phrasing)."
    )
    await evaluator.verify(
        claim=claim_div2,
        node=inst_leaf,
        sources=_sources(admin),
        additional_instruction="Accept 'Division II', 'NCAA DII', or synonyms indicating D2 participation."
    )

    # 2) University located in New York State
    loc_leaf = evaluator.add_leaf(
        id="admin3_location",
        desc="The university is located in New York State",
        parent=node,
        critical=True
    )
    claim_loc = (
        f"The provided source indicates that {uni} is located in New York State."
    )
    await evaluator.verify(
        claim=claim_loc,
        node=loc_leaf,
        sources=_sources(admin),
        additional_instruction="Confirm New York location; accept campus address or institutional description indicating New York State."
    )

    # 3) Campus in Westchester County, NY
    campus_leaf = evaluator.add_leaf(
        id="admin3_campus",
        desc="The university has a campus in Westchester County, New York",
        parent=node,
        critical=True
    )
    county = admin.campus_county or "Westchester County"
    claim_campus = (
        f"The provided source indicates that {uni} has a campus in Westchester County, New York "
        f"(e.g., campuses like Pleasantville in Westchester)."
    )
    await evaluator.verify(
        claim=claim_campus,
        node=campus_leaf,
        sources=_sources(admin),
        additional_instruction="Look for explicit mention of a campus in Westchester County (e.g., Pleasantville campus)."
    )

    # 4) Serves/served as Director of Athletics
    role_leaf = evaluator.add_leaf(
        id="admin3_role",
        desc="Serves or served as Director of Athletics at the institution",
        parent=node,
        critical=True
    )
    name = admin.full_name or "the administrator"
    role_text = admin.athletics_role or admin.position_title or "Director of Athletics"
    claim_role = (
        f"The provided source indicates that {name} serves or served as {role_text} at {uni}."
    )
    await evaluator.verify(
        claim=claim_role,
        node=role_leaf,
        sources=_sources(admin),
        additional_instruction="Accept past or present service. Minor variants like 'Athletics Director' are acceptable."
    )

    # 5) Reference URL confirms role and institutional details
    ref_leaf = evaluator.add_leaf(
        id="admin3_reference",
        desc="Provides a verifiable reference URL confirming the athletics director's role and institutional details",
        parent=node,
        critical=True
    )
    claim_ref = (
        f"The provided source page(s) mention {name}, confirm their athletics director role at {uni}, "
        f"and include institutional details (NCAA Division II participation, NY/Westchester context)."
    )
    await evaluator.verify(
        claim=claim_ref,
        node=ref_leaf,
        sources=_sources(admin),
        additional_instruction="Ensure the page connects person, role, and institutional facts; "
                               "minor formatting differences are acceptable."
    )


async def verify_administrator_4(evaluator: Evaluator, parent_node, admin: Administrator) -> None:
    """
    Administrator 4:
    - Works at a high school (grades 9–12 or similar)
    - High school located in Georgia
    - School's football team won a state championship between 2010–2015 (inclusive)
    - Holds position as principal, assistant principal, or athletic director
    - Provides verifying reference URL
    """
    node = evaluator.add_parallel(
        id="administrator_4",
        desc="Fourth administrator meeting all requirements",
        parent=parent_node,
        critical=False
    )

    req_info = evaluator.add_custom_node(
        result=_nonempty_str(admin.full_name) and _nonempty_str(admin.position_title) and (_nonempty_str(admin.organization) or _nonempty_str(admin.school_name)),
        id="admin4_required_info",
        desc="Administrator 4 has basic info (name, position, school)",
        parent=node,
        critical=True
    )
    ref_present = evaluator.add_custom_node(
        result=(len(admin.reference_urls) > 0),
        id="admin4_reference_present",
        desc="Administrator 4 provides at least one reference URL",
        parent=node,
        critical=True
    )

    # 1) Works at a high school (grades 9–12 or similar)
    level_leaf = evaluator.add_leaf(
        id="admin4_school_level",
        desc="Works at a high school (grades 9-12 or similar)",
        parent=node,
        critical=True
    )
    school = admin.school_name or admin.organization or "the high school"
    school_level_txt = admin.school_level or "high school"
    claim_level = (
        f"The provided source indicates that {school} is a high school (grades 9–12 or similar), and that the administrator works there."
    )
    await evaluator.verify(
        claim=claim_level,
        node=level_leaf,
        sources=_sources(admin),
        additional_instruction="Confirm the school level as high school; accept minor variants (e.g., secondary school serving grades 9–12)."
    )

    # 2) Located in Georgia
    loc_leaf = evaluator.add_leaf(
        id="admin4_location",
        desc="The high school is located in Georgia",
        parent=node,
        critical=True
    )
    claim_loc = (
        f"The provided source indicates that {school} is located in Georgia."
    )
    await evaluator.verify(
        claim=claim_loc,
        node=loc_leaf,
        sources=_sources(admin),
        additional_instruction="Accept abbreviations like 'GA' and references to 'Georgia High School Association' indicating Georgia location."
    )

    # 3) Football team won a state championship between 2010–2015 inclusive
    ath_leaf = evaluator.add_leaf(
        id="admin4_athletics",
        desc="The school's football team won a state championship between 2010-2015 (inclusive)",
        parent=node,
        critical=True
    )
    champ_year = admin.football_championship_year or "one of the years 2010–2015"
    claim_champ = (
        f"The provided source indicates that {school}'s football team won a Georgia state championship in the period 2010–2015 "
        f"(e.g., {champ_year} if specified)."
    )
    await evaluator.verify(
        claim=claim_champ,
        node=ath_leaf,
        sources=_sources(admin),
        additional_instruction="Look for GHSA or equivalent state championship mentions; accept newspaper articles or official records."
    )

    # 4) Holds position as principal, assistant principal, or athletic director at the school
    pos_leaf = evaluator.add_leaf(
        id="admin4_position",
        desc="Holds a position as principal, assistant principal, or athletic director at the school",
        parent=node,
        critical=True
    )
    name = admin.full_name or "the administrator"
    role_text = admin.school_position or admin.position_title or "a leadership role"
    claim_role = (
        f"The provided source indicates that {name} holds the role of principal, assistant principal, or athletic director at {school} "
        f"(e.g., {role_text})."
    )
    await evaluator.verify(
        claim=claim_role,
        node=pos_leaf,
        sources=_sources(admin),
        additional_instruction="Accept minor title variants such as 'Athletics Director' or 'AP'."
    )

    # 5) Reference confirms role and achievements
    ref_leaf = evaluator.add_leaf(
        id="admin4_reference",
        desc="Provides a verifiable reference URL confirming the administrator's role and school athletic achievements",
        parent=node,
        critical=True
    )
    claim_ref = (
        f"The provided source page(s) mention {name}, confirm their role at {school}, and include the football state championship achievement "
        f"in 2010–2015."
    )
    await evaluator.verify(
        claim=claim_ref,
        node=ref_leaf,
        sources=_sources(admin),
        additional_instruction="Ensure the page connects person, role, and the school's football championship; "
                               "minor formatting differences are acceptable."
    )


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
    model: str = "o4-mini",
) -> Dict:
    """
    Entry point to evaluate an answer for the educational administrators task.
    """
    # Initialize evaluator with a non-critical root (to allow partial credit across admins)
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,
        agent_name=agent_name,
        answer_name=answer_name,
        client=client,
        task_description="Find four educational administrators who meet all specified criteria",
        answer=answer,
        global_cache=cache,
        global_semaphore=semaphore,
        logger=logger,
        default_model=model,
    )

    # Extract administrator entries
    extracted = await evaluator.extract(
        prompt=prompt_extract_administrators(),
        template_class=AdminListExtraction,
        extraction_name="administrators_extraction",
    )

    # Limit to first 4 administrators, pad with empty entries if fewer
    admins: List[Administrator] = list(extracted.administrators[:4])
    while len(admins) < 4:
        admins.append(Administrator())

    # Build top-level admin nodes under root (parallel, non-critical per rubric)
    # And run verifications per administrator
    await verify_administrator_1(evaluator, root, admins[0])
    await verify_administrator_2(evaluator, root, admins[1])
    await verify_administrator_3(evaluator, root, admins[2])
    await verify_administrator_4(evaluator, root, admins[3])

    # Return summary
    return evaluator.get_summary()