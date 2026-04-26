import asyncio
import logging
from typing import Any, List, Optional, Dict

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy

# ------------------------------------------------------------------------------
# Task metadata
# ------------------------------------------------------------------------------
TASK_ID = "se_superintendent_march2026"
TASK_DESCRIPTION = """Identify the current school district superintendent who meets ALL of the following criteria as of March 2026:

1. The superintendent leads a school district located in a U.S. state in the Southeastern United States
2. The district is the 5th largest school district in its state by student enrollment
3. The district currently serves between 50,000 and 60,000 students
4. The district is a regular local public school district (not a charter or private school system)
5. The superintendent assumed their current role (not as interim) on July 1, 2024
6. The superintendent holds or has completed a doctoral degree (Ed.D. or Ph.D.) in Educational Leadership
7. The superintendent has at least 20 years of total experience in education
8. The superintendent previously worked within the same school district in other roles before being appointed superintendent

Provide the superintendent's full name, the name of the school district, and the state where it is located. All information must be verifiable through official district websites or credible news sources published between 2024 and 2026.
"""

# A reasonable canonical list for "Southeastern United States" (common definitions)
SOUTHEASTERN_STATES = [
    "Alabama", "Arkansas", "Florida", "Georgia", "Kentucky",
    "Louisiana", "Mississippi", "North Carolina", "South Carolina",
    "Tennessee", "Virginia", "West Virginia"
]

# Common instruction snippet to enforce source credibility/date window
SOURCE_CREDIBILITY_INS = (
    "When judging support from a URL, prefer official school district websites or reputable news media. "
    "Verify the page/article is published between Jan 1, 2024 and Mar 31, 2026 (inclusive). "
    "If publication date is missing/unclear or falls outside this window, or the site is not credible/official, "
    "treat the source as not sufficient to support the claim."
)

# ------------------------------------------------------------------------------
# Extraction models
# ------------------------------------------------------------------------------
class DistrictCore(BaseModel):
    district_name: Optional[str] = None
    state: Optional[str] = None
    location_sources: List[str] = Field(default_factory=list)


class DistrictSize(BaseModel):
    enrollment: Optional[str] = None
    size_sources: List[str] = Field(default_factory=list)


class DistrictRanking(BaseModel):
    rank_position: Optional[str] = None  # e.g., "5th", "No. 5", "5"
    ranking_sources: List[str] = Field(default_factory=list)


class DistrictType(BaseModel):
    district_type: Optional[str] = None  # e.g., "Public school district"
    type_sources: List[str] = Field(default_factory=list)


class SuperintendentIdentity(BaseModel):
    name: Optional[str] = None
    title: Optional[str] = None
    identity_sources: List[str] = Field(default_factory=list)


class AppointmentInfo(BaseModel):
    start_date: Optional[str] = None
    appointment_sources: List[str] = Field(default_factory=list)


class CareerInfo(BaseModel):
    prior_roles: List[str] = Field(default_factory=list)
    has_teaching_experience: Optional[str] = None  # "yes"/"no"/null
    has_admin_experience: Optional[str] = None     # "yes"/"no"/null
    career_sources: List[str] = Field(default_factory=list)


class EducationInfo(BaseModel):
    doctoral_degree: Optional[str] = None  # e.g., "Ed.D. in Educational Leadership"
    doctoral_institution: Optional[str] = None
    masters_degree: Optional[str] = None
    bachelors_degree: Optional[str] = None
    education_sources: List[str] = Field(default_factory=list)


class ExperienceInfo(BaseModel):
    total_years: Optional[str] = None  # e.g., "22 years", "over 25 years"
    experience_sources: List[str] = Field(default_factory=list)


class AdditionalDistrictInfo(BaseModel):
    number_of_schools: Optional[str] = None
    recognition: Optional[str] = None
    district_website: Optional[str] = None
    additional_sources: List[str] = Field(default_factory=list)


class TaskExtraction(BaseModel):
    district: Optional[DistrictCore] = None
    size: Optional[DistrictSize] = None
    ranking: Optional[DistrictRanking] = None
    dtype: Optional[DistrictType] = None
    superintendent: Optional[SuperintendentIdentity] = None
    appointment: Optional[AppointmentInfo] = None
    career: Optional[CareerInfo] = None
    education: Optional[EducationInfo] = None
    experience: Optional[ExperienceInfo] = None
    extras: Optional[AdditionalDistrictInfo] = None


# ------------------------------------------------------------------------------
# Extraction prompt
# ------------------------------------------------------------------------------
def prompt_extract_task() -> str:
    return """
Extract the following structured information exactly as presented in the answer. Do NOT invent. If any field is not present in the answer, return null or an empty list accordingly. Include all URLs that the answer cites for each corresponding item.

Return a single JSON object with these fields:

- district:
  - district_name: Full official district name
  - state: U.S. state where the district is located
  - location_sources: All URLs cited that confirm the district’s location

- size:
  - enrollment: The current student enrollment number or phrase
  - size_sources: All URLs cited that confirm enrollment

- ranking:
  - rank_position: The rank by student enrollment within the state (e.g., "5th", "No. 5", "5")
  - ranking_sources: All URLs cited that confirm the state ranking

- dtype:
  - district_type: The district type as stated (e.g., "public school district")
  - type_sources: All URLs cited that confirm the district type

- superintendent:
  - name: Full legal name of the superintendent
  - title: Their official title as stated (e.g., "Superintendent")
  - identity_sources: All URLs cited that confirm identity and title

- appointment:
  - start_date: The stated non-interim start date (e.g., "July 1, 2024")
  - appointment_sources: All URLs cited that confirm the appointment date and non-interim status

- career:
  - prior_roles: List of prior roles the person held within the SAME district (e.g., "Deputy Superintendent", "Chief Academic Officer"), if mentioned
  - has_teaching_experience: "yes" if the answer states they previously held teaching roles (e.g., teacher), "no" if the answer explicitly negates, otherwise null
  - has_admin_experience: "yes" if the answer states they previously held administrative/leadership roles, "no" if explicitly negates, otherwise null
  - career_sources: All URLs cited that confirm the internal career path within the district

- education:
  - doctoral_degree: Stated doctoral degree (e.g., "Ed.D. in Educational Leadership" or "Ph.D. in Educational Leadership")
  - doctoral_institution: Institution awarding the doctoral degree if stated
  - masters_degree: Stated master's degree (string) if any
  - bachelors_degree: Stated bachelor's degree (string) if any
  - education_sources: All URLs cited that confirm educational background (especially the doctoral degree)

- experience:
  - total_years: The total years of experience in education as stated (e.g., "over 20 years", "22 years")
  - experience_sources: All URLs cited that confirm years of experience

- extras:
  - number_of_schools: Total number of schools in the district, if stated
  - recognition: Any notable recognitions/awards mentioned for the district
  - district_website: The official district website URL if provided
  - additional_sources: Any other URLs cited about the district that are not covered above

Important:
- URL extraction: Capture actual URLs from the answer (plain or markdown links). Do not fabricate URLs.
- If a required item lacks any cited URL in the answer, set the corresponding sources list to an empty list.
"""


# ------------------------------------------------------------------------------
# Helper to enforce "sources required"
# ------------------------------------------------------------------------------
async def verify_with_sources_required(
    evaluator: Evaluator,
    *,
    claim: str,
    node,
    urls: Optional[List[str]],
    base_instruction: str = "",
    use_screenshot: bool = True
) -> bool:
    """
    Verify a claim with URLs when available. If no URLs are provided, instruct the judge to mark as not supported.
    """
    if urls and len(urls) > 0:
        return await evaluator.verify(
            claim=claim,
            node=node,
            sources=urls,
            additional_instruction=(base_instruction + " " + SOURCE_CREDIBILITY_INS).strip(),
            use_screenshot=use_screenshot
        )
    else:
        # Force a fail due to missing sources per policy
        return await evaluator.verify(
            claim=claim,
            node=node,
            sources=None,
            additional_instruction=(
                base_instruction
                + " IMPORTANT: No URL sources were provided in the answer to support this claim. "
                "You must judge this claim as Not Supported."
            ).strip(),
            use_screenshot=use_screenshot
        )


# ------------------------------------------------------------------------------
# Subtree builders
# ------------------------------------------------------------------------------
async def build_district_identification(evaluator: Evaluator, parent, data: TaskExtraction) -> None:
    # Parent grouping node (non-critical to allow mixing inside)
    district_node = evaluator.add_parallel(
        id="District_Identification",
        desc="Correctly identify the school district matching all geographic and size criteria",
        parent=parent,
        critical=False
    )

    # Existence gate for basic info (district name/state + at least one location source)
    basic_exists = evaluator.add_custom_node(
        result=(
            data and data.district
            and (data.district.district_name is not None and data.district.district_name.strip() != "")
            and (data.district.state is not None and data.district.state.strip() != "")
        ),
        id="district_basic_info_exists",
        desc="District basic info present (district name and state provided)",
        parent=district_node,
        critical=True
    )

    # 1) Geographic Location
    geo_node = evaluator.add_parallel(
        id="Geographic_Location",
        desc="Verify the district's geographic location meets specified criteria",
        parent=district_node,
        critical=False
    )

    # State Identification (fact, requires sources)
    state_id_leaf = evaluator.add_leaf(
        id="State_Identification",
        desc="Identify the correct U.S. state where the district is located",
        parent=geo_node,
        critical=True
    )
    district_name = data.district.district_name if data and data.district else ""
    state_name = data.district.state if data and data.district else ""
    await verify_with_sources_required(
        evaluator,
        claim=f"The school district named '{district_name}' is located in the U.S. state of {state_name}.",
        node=state_id_leaf,
        urls=(data.district.location_sources if data and data.district else []),
        base_instruction="Confirm the district’s state exactly as stated on the cited page(s)."
    )

    # Regional Verification (simple)
    region_leaf = evaluator.add_leaf(
        id="Regional_Verification",
        desc="Verify the state is in the Southeastern United States",
        parent=geo_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The state '{state_name}' is considered part of the Southeastern United States.",
        node=region_leaf,
        sources=None,
        additional_instruction=(
            "Use common U.S. geographic conventions. Consider the following states as Southeastern: "
            + ", ".join(SOUTHEASTERN_STATES)
            + ". Allow reasonable boundary interpretations, but if clearly outside, mark incorrect."
        )
    )

    # Location Source (enforce credible/dates)
    loc_src_leaf = evaluator.add_leaf(
        id="Location_Source",
        desc="Provide valid URL confirming the district's location",
        parent=geo_node,
        critical=True
    )
    await verify_with_sources_required(
        evaluator,
        claim=f"These source(s) explicitly confirm that '{district_name}' is located in {state_name}.",
        node=loc_src_leaf,
        urls=(data.district.location_sources if data and data.district else []),
        base_instruction="At least one source must explicitly state the district location."
    )

    # 2) District Size
    size_node = evaluator.add_parallel(
        id="District_Size",
        desc="Verify the district enrollment meets the specified size range",
        parent=district_node,
        critical=False
    )

    size_exists = evaluator.add_custom_node(
        result=(
            data and data.size and data.size.enrollment is not None and data.size.enrollment.strip() != ""
        ),
        id="size_info_exists",
        desc="Enrollment info is provided",
        parent=size_node,
        critical=True
    )

    # Current Enrollment (fact w/ sources)
    curr_enr_leaf = evaluator.add_leaf(
        id="Current_Enrollment",
        desc="Provide the district's current student enrollment count",
        parent=size_node,
        critical=True
    )
    enrollment_text = data.size.enrollment if data and data.size else ""
    await verify_with_sources_required(
        evaluator,
        claim=f"The current student enrollment of '{district_name}' is {enrollment_text}.",
        node=curr_enr_leaf,
        urls=(data.size.size_sources if data and data.size else []),
        base_instruction="Accept reasonable rounding or approximations only if the page clearly implies the same figure."
    )

    # Enrollment range match (simple)
    enr_range_leaf = evaluator.add_leaf(
        id="Enrollment_Range_Match",
        desc="Confirm enrollment falls within the specified range (50,000-60,000 students)",
        parent=size_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The enrollment value '{enrollment_text}' indicates a total between 50,000 and 60,000 students (inclusive).",
        node=enr_range_leaf,
        sources=None,
        additional_instruction="Interpret numbers in the text; allow typical formatting (commas) and phrases like 'about 55,000'."
    )

    # Size Source (credibility/date window)
    size_src_leaf = evaluator.add_leaf(
        id="Size_Source",
        desc="Provide valid URL confirming enrollment numbers",
        parent=size_node,
        critical=True
    )
    await verify_with_sources_required(
        evaluator,
        claim=f"These source(s) confirm the enrollment figure {enrollment_text} for '{district_name}'.",
        node=size_src_leaf,
        urls=(data.size.size_sources if data and data.size else []),
        base_instruction="The page must explicitly include or imply the current enrollment number mentioned."
    )

    # 3) State Ranking
    rank_node = evaluator.add_parallel(
        id="State_Ranking",
        desc="Verify the district's ranking within its state",
        parent=district_node,
        critical=False
    )

    rank_exists = evaluator.add_custom_node(
        result=(
            data and data.ranking and data.ranking.rank_position is not None and data.ranking.rank_position.strip() != ""
        ),
        id="ranking_info_exists",
        desc="Ranking info is provided",
        parent=rank_node,
        critical=True
    )

    # Rank position (fact w/ sources)
    rank_pos_leaf = evaluator.add_leaf(
        id="Rank_Position",
        desc="Identify the district's ranking among all districts in the state by enrollment",
        parent=rank_node,
        critical=True
    )
    rank_text = data.ranking.rank_position if data and data.ranking else ""
    await verify_with_sources_required(
        evaluator,
        claim=f"'{district_name}' is the {rank_text} largest school district by enrollment in {state_name}.",
        node=rank_pos_leaf,
        urls=(data.ranking.ranking_sources if data and data.ranking else []),
        base_instruction="The page must explicitly list or rank the district by student enrollment within the state."
    )

    # Rank specification match = 5th
    rank_match_leaf = evaluator.add_leaf(
        id="Rank_Specification_Match",
        desc="Confirm the district ranks as the 5th largest in its state",
        parent=rank_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The ranking text '{rank_text}' indicates that the district is 5th largest in the state.",
        node=rank_match_leaf,
        sources=None,
        additional_instruction="Treat ordinal/cardy equivalences as matching (e.g., '5th', '#5', 'No. 5')."
    )

    # Ranking Source
    ranking_src_leaf = evaluator.add_leaf(
        id="Ranking_Source",
        desc="Provide valid URL confirming the district's state ranking",
        parent=rank_node,
        critical=True
    )
    await verify_with_sources_required(
        evaluator,
        claim=f"These source(s) confirm that '{district_name}' ranks 5th by enrollment in {state_name}.",
        node=ranking_src_leaf,
        urls=(data.ranking.ranking_sources if data and data.ranking else []),
        base_instruction="The page must explicitly mention the ranking position or provide a list where 5th is clear."
    )

    # 4) District Type
    dtype_node = evaluator.add_parallel(
        id="District_Type",
        desc="Verify the district is a regular local public school district",
        parent=district_node,
        critical=False
    )

    dtype_exists = evaluator.add_custom_node(
        result=(
            data and data.dtype and data.dtype.district_type is not None and data.dtype.district_type.strip() != ""
        ),
        id="type_info_exists",
        desc="District type info is provided",
        parent=dtype_node,
        critical=True
    )

    # Type verification (fact w/ sources)
    type_ver_leaf = evaluator.add_leaf(
        id="Type_Verification",
        desc="Confirm the district type (must be regular local public school district, not charter or private)",
        parent=dtype_node,
        critical=True
    )
    district_type_text = data.dtype.district_type if data and data.dtype else ""
    await verify_with_sources_required(
        evaluator,
        claim=f"'{district_name}' is a regular local public school district (not a charter or private system).",
        node=type_ver_leaf,
        urls=(data.dtype.type_sources if data and data.dtype else []),
        base_instruction="If the page indicates charter-only, private, or state-run special system, mark incorrect."
    )

    # Type Source
    type_src_leaf = evaluator.add_leaf(
        id="Type_Source",
        desc="Provide valid URL confirming district type",
        parent=dtype_node,
        critical=True
    )
    await verify_with_sources_required(
        evaluator,
        claim=f"These source(s) confirm that '{district_name}' is a regular local public school district.",
        node=type_src_leaf,
        urls=(data.dtype.type_sources if data and data.dtype else []),
        base_instruction="The source should clearly indicate the public nature of the district."
    )


async def build_superintendent_identification(evaluator: Evaluator, parent, data: TaskExtraction) -> None:
    sup_node = evaluator.add_sequential(
        id="Superintendent_Identification",
        desc="Correctly identify the current superintendent of the identified district",
        parent=parent,
        critical=False
    )

    # A) Current Superintendent (parallel cluster)
    curr_sup_node = evaluator.add_parallel(
        id="Current_Superintendent",
        desc="Identify the person currently serving as superintendent",
        parent=sup_node,
        critical=False
    )

    identity_exists = evaluator.add_custom_node(
        result=(
            data and data.superintendent
            and (data.superintendent.name is not None and data.superintendent.name.strip() != "")
        ),
        id="identity_info_exists",
        desc="Superintendent name provided",
        parent=curr_sup_node,
        critical=True
    )

    # Full Name (presence check)
    full_name_leaf = evaluator.add_custom_node(
        result=(
            data and data.superintendent
            and (data.superintendent.name is not None and data.superintendent.name.strip() != "")
        ),
        id="Full_Name",
        desc="Provide the superintendent's full legal name",
        parent=curr_sup_node,
        critical=True
    )

    # Title Verification (fact with sources)
    title_ver_leaf = evaluator.add_leaf(
        id="Title_Verification",
        desc="Confirm the person holds the official title of 'Superintendent' (not interim, assistant, or associate)",
        parent=curr_sup_node,
        critical=True
    )
    sup_name = data.superintendent.name if data and data.superintendent else ""
    await verify_with_sources_required(
        evaluator,
        claim=f"{sup_name} holds the official title of Superintendent for the district (not interim/acting/assistant/associate).",
        node=title_ver_leaf,
        urls=(data.superintendent.identity_sources if data and data.superintendent else []),
        base_instruction="The page should clearly indicate the official Superintendent title without interim qualifiers."
    )

    # Identity Source (credibility/date)
    id_src_leaf = evaluator.add_leaf(
        id="Identity_Source",
        desc="Provide valid URL confirming the superintendent's identity and title",
        parent=curr_sup_node,
        critical=True
    )
    await verify_with_sources_required(
        evaluator,
        claim=f"These source(s) confirm that {sup_name} is the current Superintendent as of March 2026.",
        node=id_src_leaf,
        urls=(data.superintendent.identity_sources if data and data.superintendent else []),
        base_instruction="Check that the page implies currency through date or updated roster; if outdated, mark not supported."
    )

    # B) Appointment Details (parallel cluster)
    appt_node = evaluator.add_parallel(
        id="Appointment_Details",
        desc="Verify the superintendent's appointment information",
        parent=sup_node,
        critical=False
    )

    # Appointment Date (sub-parallel)
    appt_date_node = evaluator.add_parallel(
        id="Appointment_Date",
        desc="Verify the superintendent's exact start date",
        parent=appt_node,
        critical=False
    )

    # Start Date Verification (fact w/ sources)
    start_date_leaf = evaluator.add_leaf(
        id="Start_Date_Verification",
        desc="Confirm the superintendent began serving on July 1, 2024",
        parent=appt_date_node,
        critical=True
    )
    start_date_text = data.appointment.start_date if data and data.appointment else ""
    await verify_with_sources_required(
        evaluator,
        claim=f"{sup_name} began serving as Superintendent on July 1, 2024 (non-interim appointment).",
        node=start_date_leaf,
        urls=(data.appointment.appointment_sources if data and data.appointment else []),
        base_instruction="The page must explicitly match this date and indicate not interim/acting."
    )

    # Timeline Source
    appt_src_leaf = evaluator.add_leaf(
        id="Timeline_Source",
        desc="Provide valid URL confirming appointment date",
        parent=appt_date_node,
        critical=True
    )
    await verify_with_sources_required(
        evaluator,
        claim=f"These source(s) explicitly confirm {sup_name}'s Superintendent start date of July 1, 2024.",
        node=appt_src_leaf,
        urls=(data.appointment.appointment_sources if data and data.appointment else []),
        base_instruction="Check for explicit date; if ambiguous or different, mark not supported."
    )

    # Internal Career Path (parallel)
    career_node = evaluator.add_parallel(
        id="Internal_Career_Path",
        desc="Verify the superintendent previously worked within the same district",
        parent=appt_node,
        critical=False
    )

    # Prior District Employment (fact w/ sources)
    prior_emp_leaf = evaluator.add_leaf(
        id="Prior_District_Employment",
        desc="Confirm the superintendent held previous positions within the same school district",
        parent=career_node,
        critical=True
    )
    await verify_with_sources_required(
        evaluator,
        claim=f"Before becoming Superintendent, {sup_name} previously worked within the same district in other roles.",
        node=prior_emp_leaf,
        urls=(data.career.career_sources if data and data.career else []),
        base_instruction="The page must clearly indicate prior roles within the same district."
    )

    # Career Roles (non-critical, extraction-based presence)
    roles_present_leaf = evaluator.add_custom_node(
        result=bool(data and data.career and data.career.prior_roles and len(data.career.prior_roles) > 0),
        id="Career_Roles",
        desc="Document the progression of roles held within the district",
        parent=career_node,
        critical=False
    )

    # Career Source (credibility/date)
    career_src_leaf = evaluator.add_leaf(
        id="Career_Source",
        desc="Provide valid URL confirming internal career progression",
        parent=career_node,
        critical=True
    )
    await verify_with_sources_required(
        evaluator,
        claim=f"These source(s) confirm {sup_name}'s internal career progression within the district before July 1, 2024.",
        node=career_src_leaf,
        urls=(data.career.career_sources if data and data.career else []),
        base_instruction="Look for explicit mention of previous district roles (e.g., deputy, chief officer, principal)."
    )


async def build_educational_qualifications(evaluator: Evaluator, parent, data: TaskExtraction) -> None:
    edu_node = evaluator.add_parallel(
        id="Educational_Qualifications",
        desc="Verify the superintendent meets required educational qualifications",
        parent=parent,
        critical=False
    )

    # Doctoral Degree cluster
    doc_node = evaluator.add_parallel(
        id="Doctoral_Degree",
        desc="Verify the superintendent holds or has completed a doctoral degree",
        parent=edu_node,
        critical=False
    )

    # Degree Possession (fact w/ sources)
    degree_possess_leaf = evaluator.add_leaf(
        id="Degree_Possession",
        desc="Confirm the superintendent has earned or completed a doctoral degree (Ed.D. or Ph.D.) in Educational Leadership",
        parent=doc_node,
        critical=True
    )
    doc_deg_text = data.education.doctoral_degree if data and data.education else ""
    await verify_with_sources_required(
        evaluator,
        claim=f"{data.superintendent.name if data and data.superintendent else 'The superintendent'} "
              f"has completed a doctoral degree in Educational Leadership (e.g., Ed.D. or Ph.D.).",
        node=degree_possess_leaf,
        urls=(data.education.education_sources if data and data.education else []),
        base_instruction="The page must explicitly confirm a doctoral degree and that it is in Educational Leadership (or equivalent)."
    )

    # Degree Institution (non-critical; try verifying institution if given)
    degree_inst_leaf = evaluator.add_leaf(
        id="Degree_Institution",
        desc="Identify the institution that awarded the doctoral degree",
        parent=doc_node,
        critical=False
    )
    doc_inst_text = data.education.doctoral_institution if data and data.education else ""
    await verify_with_sources_required(
        evaluator,
        claim=f"The superintendent earned the doctoral degree from {doc_inst_text}.",
        node=degree_inst_leaf,
        urls=(data.education.education_sources if data and data.education else []),
        base_instruction="If the institution is not clearly stated, mark not supported."
    )

    # Education Source (credibility/date window focus on doctoral degree)
    edu_src_leaf = evaluator.add_leaf(
        id="Education_Source",
        desc="Provide valid URL confirming doctoral degree",
        parent=doc_node,
        critical=True
    )
    await verify_with_sources_required(
        evaluator,
        claim="These source(s) explicitly confirm the superintendent's doctoral degree in Educational Leadership.",
        node=edu_src_leaf,
        urls=(data.education.education_sources if data and data.education else []),
        base_instruction="Ensure explicit confirmation of the doctoral credential."
    )

    # Prior degrees cluster (non-critical)
    prior_deg_node = evaluator.add_parallel(
        id="Prior_Degrees",
        desc="Document the superintendent's prior degrees",
        parent=edu_node,
        critical=False
    )

    masters_leaf = evaluator.add_leaf(
        id="Masters_Degree",
        desc="Confirm the superintendent holds a master's degree",
        parent=prior_deg_node,
        critical=False
    )
    masters_text = data.education.masters_degree if data and data.education else ""
    await verify_with_sources_required(
        evaluator,
        claim=f"The superintendent holds a master's degree ({masters_text}).",
        node=masters_leaf,
        urls=(data.education.education_sources if data and data.education else []),
        base_instruction="If no master's degree is mentioned, mark not supported."
    )

    bachelors_leaf = evaluator.add_leaf(
        id="Bachelors_Degree",
        desc="Confirm the superintendent holds a bachelor's degree",
        parent=prior_deg_node,
        critical=False
    )
    bachelors_text = data.education.bachelors_degree if data and data.education else ""
    await verify_with_sources_required(
        evaluator,
        claim=f"The superintendent holds a bachelor's degree ({bachelors_text}).",
        node=bachelors_leaf,
        urls=(data.education.education_sources if data and data.education else []),
        base_instruction="If no bachelor's degree is mentioned, mark not supported."
    )


async def build_professional_experience(evaluator: Evaluator, parent, data: TaskExtraction) -> None:
    exp_node = evaluator.add_parallel(
        id="Professional_Experience",
        desc="Verify the superintendent's professional experience in education",
        parent=parent,
        critical=False
    )

    years_node = evaluator.add_parallel(
        id="Years_in_Education",
        desc="Verify the superintendent has substantial experience in education",
        parent=exp_node,
        critical=False
    )

    # Total years (fact w/ sources)
    total_years_leaf = evaluator.add_leaf(
        id="Total_Years",
        desc="Confirm the superintendent has at least 20 years of experience in education",
        parent=years_node,
        critical=True
    )
    years_text = data.experience.total_years if data and data.experience else ""
    await verify_with_sources_required(
        evaluator,
        claim=f"The superintendent has at least 20 years of total experience in education (e.g., '{years_text}').",
        node=total_years_leaf,
        urls=(data.experience.experience_sources if data and data.experience else []),
        base_instruction="If the page implies fewer than 20 years, mark not supported."
    )

    # Experience Source
    exp_src_leaf = evaluator.add_leaf(
        id="Experience_Source",
        desc="Provide valid URL confirming years of experience",
        parent=years_node,
        critical=True
    )
    await verify_with_sources_required(
        evaluator,
        claim="These source(s) explicitly confirm the superintendent's total years of experience (≥ 20 years).",
        node=exp_src_leaf,
        urls=(data.experience.experience_sources if data and data.experience else []),
        base_instruction="Look for explicit totals or totals inferable from dated roles."
    )

    # Career progression (non-critical cluster)
    prog_node = evaluator.add_parallel(
        id="Career_Progression",
        desc="Document the superintendent's career progression within education",
        parent=exp_node,
        critical=False
    )

    # Teaching experience (non-critical)
    teach_leaf = evaluator.add_leaf(
        id="Teaching_Experience",
        desc="Identify any teaching roles held by the superintendent",
        parent=prog_node,
        critical=False
    )
    has_teaching = (data.career.has_teaching_experience or "").strip().lower() if data and data.career else ""
    await verify_with_sources_required(
        evaluator,
        claim=f"The superintendent has previously held teaching roles: {has_teaching == 'yes'}.",
        node=teach_leaf,
        urls=(data.career.career_sources if data and data.career else []),
        base_instruction="If no explicit teaching role is shown, mark not supported."
    )

    # Administrative roles (non-critical)
    admin_leaf = evaluator.add_leaf(
        id="Administrative_Roles",
        desc="Identify previous administrative or leadership positions",
        parent=prog_node,
        critical=False
    )
    has_admin = (data.career.has_admin_experience or "").strip().lower() if data and data.career else ""
    await verify_with_sources_required(
        evaluator,
        claim=f"The superintendent has held administrative/leadership roles prior to their appointment: {has_admin == 'yes'}.",
        node=admin_leaf,
        urls=(data.career.career_sources if data and data.career else []),
        base_instruction="If no explicit admin/leadership role is shown, mark not supported."
    )

    # Optional role documentation source (non-critical)
    role_src_leaf = evaluator.add_leaf(
        id="Role_Documentation_Source",
        desc="Provide URL documenting career progression if available",
        parent=prog_node,
        critical=False
    )
    await verify_with_sources_required(
        evaluator,
        claim="These source(s) document the superintendent's prior roles progression.",
        node=role_src_leaf,
        urls=(data.career.career_sources if data and data.career else []),
        base_instruction="If roles are not explicitly listed, this should fail."
    )


async def build_additional_info(evaluator: Evaluator, parent, data: TaskExtraction) -> None:
    add_node = evaluator.add_parallel(
        id="Additional_District_Information",
        desc="Provide supplementary information about the identified district",
        parent=parent,
        critical=False
    )

    # Number of schools (non-critical) - try verify with any provided extras/additional sources or district site
    num_schools_leaf = evaluator.add_leaf(
        id="Number_of_Schools",
        desc="Provide the total number of schools in the district",
        parent=add_node,
        critical=False
    )
    schools_text = data.extras.number_of_schools if data and data.extras else ""
    add_urls = (data.extras.additional_sources if data and data.extras else [])
    # Prefer district website if provided
    if data and data.extras and data.extras.district_website:
        add_urls = [data.extras.district_website] + add_urls
    await verify_with_sources_required(
        evaluator,
        claim=f"The total number of schools in the district is {schools_text}.",
        node=num_schools_leaf,
        urls=add_urls,
        base_instruction="If the page does not clearly state the total number, mark not supported."
    )

    # Recognition (non-critical)
    recog_leaf = evaluator.add_leaf(
        id="District_Recognition",
        desc="Note any state or national recognition the district has received",
        parent=add_node,
        critical=False
    )
    recog_text = data.extras.recognition if data and data.extras else ""
    await verify_with_sources_required(
        evaluator,
        claim=f"The district has received the following recognition: {recog_text}.",
        node=recog_leaf,
        urls=add_urls,
        base_instruction="Look for explicit mention of awards or recognitions."
    )

    # District website (non-critical) - verify URL is official site
    website_leaf = evaluator.add_leaf(
        id="District_Website",
        desc="Provide the official website URL for the district",
        parent=add_node,
        critical=False
    )
    district_site = data.extras.district_website if data and data.extras else None
    await verify_with_sources_required(
        evaluator,
        claim=f"This URL is the official website of the school district: {district_site}.",
        node=website_leaf,
        urls=([district_site] if district_site else []),
        base_instruction="The homepage should clearly represent the official school district."
    )


# ------------------------------------------------------------------------------
# Main evaluation entry point
# ------------------------------------------------------------------------------
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
    Build and run the evaluation for the superintendent identification task.
    """
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.SEQUENTIAL,  # Evaluate major clusters in sequence
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

    # Extraction
    extracted = await evaluator.extract(
        prompt=prompt_extract_task(),
        template_class=TaskExtraction,
        extraction_name="superintendent_task_extraction"
    )

    # Root-level container to mimic "Task_Completion"
    task_node = evaluator.add_sequential(
        id="Task_Completion",
        desc="Complete identification and verification of a school district superintendent meeting all specified criteria",
        parent=root,
        critical=False  # Keep non-critical to allow mixed strictness within subtrees
    )

    # Subtrees
    await build_district_identification(evaluator, task_node, extracted)
    await build_superintendent_identification(evaluator, task_node, extracted)
    await build_educational_qualifications(evaluator, task_node, extracted)
    await build_professional_experience(evaluator, task_node, extracted)
    await build_additional_info(evaluator, task_node, extracted)

    # Return structured summary
    return evaluator.get_summary()