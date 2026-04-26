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
TASK_ID = "edu_leadership_appointments_Q1_2026"
TASK_DESCRIPTION = """Between January and March 2026, several educational institutions announced significant leadership appointments. Identify the following four individuals based on the specified criteria:

Individual 1: An athletic director appointed at a private university in New York State in March 2026, who previously served as athletic director at a Mid-American Conference institution starting in 2022, achieved 282% fundraising growth at their previous institution, and was named to the Sports Business Journal's Forty Under 40 list in 2024.

Individual 2: A university president appointed at a public Big Ten university in March 2026, who had been serving as that same university's executive vice president and provost since January 2025, and previously served as provost at Emory University and as dean of the Pratt School of Engineering at Duke University.

Individual 3: A head football coach hired by a public Big Ten university in December 2025, who is 66 years old, previously served as head coach at the University of Utah for 21 seasons with a 177-88 record, and signed a five-year contract averaging $8.2 million per year.

Individual 4: A school district superintendent appointed in a city in western New York State in January 2026, who had worked in that same school district for 20 years before being selected from a field of four internal candidates.

For each individual, provide:
- The person's full name
- The specific institution or school district name
- The exact appointment announcement date
- Verification that all specified qualifying criteria are met
- Reference URLs from authoritative sources supporting the information
"""


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class Individual1AD(BaseModel):
    name: Optional[str] = None
    institution: Optional[str] = None
    appointment_date: Optional[str] = None
    previous_institution: Optional[str] = None
    previous_role_title: Optional[str] = None
    previous_start_year: Optional[str] = None
    fundraising_growth_percent: Optional[str] = None
    sbj_recognition_year: Optional[str] = None

    institution_urls: List[str] = Field(default_factory=list)
    appointment_date_urls: List[str] = Field(default_factory=list)
    previous_position_urls: List[str] = Field(default_factory=list)
    fundraising_urls: List[str] = Field(default_factory=list)
    sbj_urls: List[str] = Field(default_factory=list)


class Individual2President(BaseModel):
    name: Optional[str] = None
    institution: Optional[str] = None
    appointment_date: Optional[str] = None
    provost_since_date: Optional[str] = None  # e.g., "January 2025"

    institution_urls: List[str] = Field(default_factory=list)
    conference_urls: List[str] = Field(default_factory=list)  # Big Ten membership proof
    appointment_date_urls: List[str] = Field(default_factory=list)
    provost_urls: List[str] = Field(default_factory=list)
    emory_urls: List[str] = Field(default_factory=list)
    duke_urls: List[str] = Field(default_factory=list)


class Individual3Coach(BaseModel):
    name: Optional[str] = None
    institution: Optional[str] = None
    hiring_date: Optional[str] = None
    age_years: Optional[str] = None  # "66"
    utah_seasons: Optional[str] = None  # "21"
    utah_record: Optional[str] = None  # "177-88"
    contract_years: Optional[str] = None  # "5"
    contract_avg_per_year: Optional[str] = None  # "$8.2 million" or "8.2 million"

    institution_urls: List[str] = Field(default_factory=list)
    conference_urls: List[str] = Field(default_factory=list)
    hiring_date_urls: List[str] = Field(default_factory=list)
    age_urls: List[str] = Field(default_factory=list)
    utah_urls: List[str] = Field(default_factory=list)
    contract_urls: List[str] = Field(default_factory=list)


class Individual4Superintendent(BaseModel):
    name: Optional[str] = None
    district: Optional[str] = None
    city: Optional[str] = None
    appointment_date: Optional[str] = None
    years_in_district: Optional[str] = None  # "20"
    selection_candidates_count: Optional[str] = None  # "4"

    district_urls: List[str] = Field(default_factory=list)
    appointment_date_urls: List[str] = Field(default_factory=list)
    years_service_urls: List[str] = Field(default_factory=list)
    selection_urls: List[str] = Field(default_factory=list)


class AllIndividualsExtraction(BaseModel):
    individual1: Optional[Individual1AD] = Field(default_factory=Individual1AD)
    individual2: Optional[Individual2President] = Field(default_factory=Individual2President)
    individual3: Optional[Individual3Coach] = Field(default_factory=Individual3Coach)
    individual4: Optional[Individual4Superintendent] = Field(default_factory=Individual4Superintendent)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_all() -> str:
    return """
Extract structured information for four specified individuals exactly as stated in the provided answer. Do not infer or fabricate any field. If a field is not explicitly present, set it to null (or [] for URL lists). Extract explicit URLs only.

Return a JSON object with keys: individual1, individual2, individual3, individual4, each mapping to an object with the fields below.

individual1 (Athletic Director in March 2026 at a private NY university):
- name
- institution
- appointment_date
- previous_institution
- previous_role_title
- previous_start_year
- fundraising_growth_percent
- sbj_recognition_year
- institution_urls (URLs confirming institution identity/type and NY location)
- appointment_date_urls (URLs confirming the announcement date/month-year)
- previous_position_urls (URLs confirming prior role, institution, start year, MAC affiliation)
- fundraising_urls (URLs confirming 282% fundraising growth)
- sbj_urls (URLs confirming Sports Business Journal Forty Under 40 recognition and year)

individual2 (University President in March 2026 at a public Big Ten university):
- name
- institution
- appointment_date
- provost_since_date  (e.g., "January 2025")
- institution_urls (URLs confirming institution identity and public status)
- conference_urls (URLs confirming Big Ten membership)
- appointment_date_urls (URLs confirming announcement date/month-year)
- provost_urls (URLs confirming EVP & Provost role at same university and start date)
- emory_urls (URLs confirming service as Emory provost)
- duke_urls (URLs confirming service as Duke Pratt engineering dean)

individual3 (Head football coach hired in Dec 2025 at a public Big Ten university):
- name
- institution
- hiring_date
- age_years
- utah_seasons
- utah_record
- contract_years
- contract_avg_per_year
- institution_urls (URLs confirming institution identity and public status)
- conference_urls (URLs confirming Big Ten membership)
- hiring_date_urls (URLs confirming hiring announcement/date/month-year)
- age_urls (URLs confirming age)
- utah_urls (URLs confirming Utah head coach role, seasons, and 177-88 record)
- contract_urls (URLs confirming 5-year contract and $8.2 million average per year)

individual4 (School superintendent in western New York in Jan 2026):
- name
- district
- city
- appointment_date
- years_in_district
- selection_candidates_count
- district_urls (URLs confirming district identity and city/location)
- appointment_date_urls (URLs confirming announcement date/month-year)
- years_service_urls (URLs confirming 20 years in district)
- selection_urls (URLs confirming selection from four INTERNAL candidates)

Rules:
- For any URL list, include only explicit URLs present in the answer (plain or markdown). If none, use an empty array.
- Keep all date fields as strings exactly as written in the answer.
- For numeric descriptors (e.g., "282%"), keep them as strings exactly as written.
"""


# --------------------------------------------------------------------------- #
# Utility helpers                                                             #
# --------------------------------------------------------------------------- #
def has_text(s: Optional[str]) -> bool:
    return bool(s and isinstance(s, str) and s.strip() != "")


def has_urls(urls: Optional[List[str]]) -> bool:
    return bool(urls and isinstance(urls, list) and any((u or "").strip() for u in urls))


def combine_urls(*url_lists: Optional[List[str]]) -> List[str]:
    urls: List[str] = []
    for lst in url_lists:
        if lst:
            for u in lst:
                if isinstance(u, str) and u.strip():
                    urls.append(u)
    return urls


# --------------------------------------------------------------------------- #
# Verification builders                                                       #
# --------------------------------------------------------------------------- #
async def verify_individual_1(evaluator: Evaluator, parent_node, data: Individual1AD):
    ind_node = evaluator.add_parallel(
        id="Individual_1_Athletic_Director",
        desc="Correctly identify the athletic director appointed at a New York private university in March 2026",
        parent=parent_node,
        critical=False
    )

    # Name provided
    evaluator.add_custom_node(
        result=has_text(data.name),
        id="AD_Name_Provided",
        desc="Provide the individual's full name",
        parent=ind_node,
        critical=True
    )

    # Institution identity (critical)
    inst_node = evaluator.add_parallel(
        id="AD_Institution_Identity",
        desc="Provide the specific institution name and verify it is a private university in New York State",
        parent=ind_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=has_text(data.institution),
        id="AD_Institution_Name_Provided",
        desc="Provide the institution name",
        parent=inst_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=has_urls(data.institution_urls),
        id="AD_Institution_URL",
        desc="Provide URL confirming institution identity and type",
        parent=inst_node,
        critical=True
    )
    # Verify private
    ad_private_leaf = evaluator.add_leaf(
        id="AD_Institution_Is_Private",
        desc="The institution is a private university",
        parent=inst_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"{data.institution} is a private university.",
        node=ad_private_leaf,
        sources=data.institution_urls,
        additional_instruction="Judge strictly based on the provided webpages. If pages indicate the university is private (non-public), mark as supported."
    )
    # Verify in New York State
    ad_in_ny_leaf = evaluator.add_leaf(
        id="AD_Institution_In_NY",
        desc="The institution is located in New York State",
        parent=inst_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"{data.institution} is located in New York State.",
        node=ad_in_ny_leaf,
        sources=data.institution_urls,
        additional_instruction="Confirm the institution is in the State of New York (NY)."
    )

    # Appointment timing (critical)
    timing_node = evaluator.add_parallel(
        id="AD_Appointment_Timing",
        desc="Verify the appointment was announced in March 2026",
        parent=ind_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=has_text(data.appointment_date),
        id="AD_Announcement_Date_Provided",
        desc="Provide the exact announcement date",
        parent=timing_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=has_urls(data.appointment_date_urls),
        id="AD_Date_URL",
        desc="Provide URL confirming announcement date",
        parent=timing_node,
        critical=True
    )
    ad_date_month_leaf = evaluator.add_leaf(
        id="AD_Date_In_March_2026",
        desc="The announcement date is in March 2026",
        parent=timing_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The announcement of {data.name} at {data.institution} occurred in March 2026.",
        node=ad_date_month_leaf,
        sources=data.appointment_date_urls,
        additional_instruction="Determine from the page's posted/publication date or the announcement date string if it is in March 2026."
    )

    # Previous position at MAC (critical)
    prev_node = evaluator.add_parallel(
        id="AD_Previous_Position_MAC",
        desc="Verify previous position at a Mid-American Conference institution starting in 2022",
        parent=ind_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=has_text(data.previous_institution),
        id="AD_Previous_Institution_Provided",
        desc="Provide the previous institution name",
        parent=prev_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=has_urls(data.previous_position_urls),
        id="AD_Previous_Position_URL",
        desc="Provide URL confirming previous position details",
        parent=prev_node,
        critical=True
    )
    # Served as AD at previous institution
    prev_role_leaf = evaluator.add_leaf(
        id="AD_Previous_Role_Was_AD",
        desc="The individual served as athletic director at the previous institution",
        parent=prev_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"{data.name} served as athletic director at {data.previous_institution}.",
        node=prev_role_leaf,
        sources=data.previous_position_urls,
        additional_instruction="Confirm the title as Athletic Director (Director of Athletics)."
    )
    # Previous institution was MAC member
    prev_mac_leaf = evaluator.add_leaf(
        id="AD_Previous_Was_MAC",
        desc="The previous institution is in the Mid-American Conference",
        parent=prev_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"{data.previous_institution} competes in the Mid-American Conference (MAC).",
        node=prev_mac_leaf,
        sources=data.previous_position_urls,
        additional_instruction="The page should explicitly state MAC membership for the institution/its athletics."
    )
    # Started in 2022
    started_2022_leaf = evaluator.add_leaf(
        id="AD_Started_2022",
        desc="The individual started the previous position in 2022",
        parent=prev_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"{data.name} started the athletic director position at {data.previous_institution} in 2022.",
        node=started_2022_leaf,
        sources=data.previous_position_urls,
        additional_instruction="If hired late 2021 with a start date in 2022, consider that as starting in 2022."
    )

    # Fundraising growth (critical)
    fund_node = evaluator.add_parallel(
        id="AD_Fundraising_Growth",
        desc="Verify 282% fundraising growth achievement at previous institution",
        parent=ind_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=has_urls(data.fundraising_urls),
        id="AD_Fundraising_URL",
        desc="Provide URL confirming fundraising achievement",
        parent=fund_node,
        critical=True
    )
    fund_leaf = evaluator.add_leaf(
        id="AD_Fundraising_Achievement_Documented",
        desc="Document the 282% fundraising growth at previous institution",
        parent=fund_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"During {data.name}'s tenure at {data.previous_institution}, fundraising grew by 282%.",
        node=fund_leaf,
        sources=data.fundraising_urls,
        additional_instruction="Look for explicit mention of '282%' fundraising growth."
    )

    # SBJ recognition (critical)
    sbj_node = evaluator.add_parallel(
        id="AD_SBJ_Recognition",
        desc="Verify Sports Business Journal Forty Under 40 recognition in 2024",
        parent=ind_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=has_urls(data.sbj_urls),
        id="AD_SBJ_URL",
        desc="Provide URL confirming SBJ recognition",
        parent=sbj_node,
        critical=True
    )
    sbj_award_leaf = evaluator.add_leaf(
        id="AD_SBJ_Award_Received",
        desc="Individual was named to Sports Business Journal's Forty Under 40 list",
        parent=sbj_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"{data.name} was named to Sports Business Journal's Forty Under 40 list.",
        node=sbj_award_leaf,
        sources=data.sbj_urls,
        additional_instruction="Confirm the named person appears on SBJ 'Forty Under 40'."
    )
    sbj_year_leaf = evaluator.add_leaf(
        id="AD_SBJ_Year_2024",
        desc="The recognition was received in 2024",
        parent=sbj_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"{data.name} received the SBJ Forty Under 40 recognition in 2024.",
        node=sbj_year_leaf,
        sources=data.sbj_urls,
        additional_instruction="Year must be 2024."
    )


async def verify_individual_2(evaluator: Evaluator, parent_node, data: Individual2President):
    ind_node = evaluator.add_parallel(
        id="Individual_2_University_President",
        desc="Correctly identify the university president appointed at a Big Ten university in March 2026",
        parent=parent_node,
        critical=False
    )

    evaluator.add_custom_node(
        result=has_text(data.name),
        id="Pres_Name_Provided",
        desc="Provide the individual's full name",
        parent=ind_node,
        critical=True
    )

    inst_node = evaluator.add_parallel(
        id="Pres_Institution_Identity",
        desc="Provide the specific institution name and verify it is a public Big Ten university",
        parent=ind_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=has_text(data.institution),
        id="Pres_Institution_Name_Provided",
        desc="Provide the institution name",
        parent=inst_node,
        critical=True
    )
    # URL presence for identity/membership
    evaluator.add_custom_node(
        result=has_urls(data.institution_urls) or has_urls(data.conference_urls),
        id="Pres_Institution_URL",
        desc="Provide URL confirming institution identity and conference membership",
        parent=inst_node,
        critical=True
    )
    # Public university
    pub_leaf = evaluator.add_leaf(
        id="Pres_Institution_Is_Public",
        desc="The institution is a public university",
        parent=inst_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"{data.institution} is a public university.",
        node=pub_leaf,
        sources=data.institution_urls,
        additional_instruction="Confirm public status using provided pages."
    )
    # Big Ten membership
    bigten_leaf = evaluator.add_leaf(
        id="Pres_Institution_Is_Big_Ten",
        desc="The institution is a Big Ten university",
        parent=inst_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"{data.institution} is a member of the Big Ten Conference.",
        node=bigten_leaf,
        sources=combine_urls(data.conference_urls, data.institution_urls),
        additional_instruction="Prefer Big Ten official site or institution page explicitly stating Big Ten membership."
    )

    # Appointment timing
    timing_node = evaluator.add_parallel(
        id="Pres_Appointment_Timing",
        desc="Verify the appointment was announced in March 2026",
        parent=ind_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=has_text(data.appointment_date),
        id="Pres_Announcement_Date_Provided",
        desc="Provide the exact announcement date",
        parent=timing_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=has_urls(data.appointment_date_urls),
        id="Pres_Date_URL",
        desc="Provide URL confirming announcement date",
        parent=timing_node,
        critical=True
    )
    pres_date_leaf = evaluator.add_leaf(
        id="Pres_Date_In_March_2026",
        desc="The announcement date is in March 2026",
        parent=timing_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The appointment of {data.name} as president at {data.institution} was announced in March 2026.",
        node=pres_date_leaf,
        sources=data.appointment_date_urls,
        additional_instruction="Use the page timestamp or announcement date."
    )

    # Provost at same institution since Jan 2025
    prov_node = evaluator.add_parallel(
        id="Pres_Was_Provost_Same_Institution",
        desc="Verify the individual was serving as executive vice president and provost at the same university since January 2025",
        parent=ind_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=has_urls(data.provost_urls),
        id="Pres_Provost_URL",
        desc="Provide URL confirming provost position and start date",
        parent=prov_node,
        critical=True
    )
    prov_role_leaf = evaluator.add_leaf(
        id="Pres_Held_Provost_Role",
        desc="Individual served as executive vice president and provost at the same university",
        parent=prov_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"{data.name} served as executive vice president and provost at {data.institution}.",
        node=prov_role_leaf,
        sources=data.provost_urls,
        additional_instruction="Look for 'Executive Vice President and Provost' at the same institution."
    )
    prov_since_leaf = evaluator.add_leaf(
        id="Pres_Provost_Since_Jan_2025",
        desc="Individual started provost position in January 2025",
        parent=prov_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"{data.name} started the EVP & Provost role at {data.institution} in January 2025.",
        node=prov_since_leaf,
        sources=data.provost_urls,
        additional_instruction="Exact month and year must match January 2025 (allow reasonable phrasing like 'effective Jan. 2025')."
    )

    # Emory provost (critical)
    emory_node = evaluator.add_parallel(
        id="Pres_Previous_Emory_Provost",
        desc="Verify the individual previously served as provost at Emory University",
        parent=ind_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=has_urls(data.emory_urls),
        id="Pres_Emory_URL",
        desc="Provide URL confirming Emory provost position",
        parent=emory_node,
        critical=True
    )
    emory_leaf = evaluator.add_leaf(
        id="Pres_Was_Emory_Provost",
        desc="Individual served as provost at Emory University",
        parent=emory_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"{data.name} previously served as provost at Emory University.",
        node=emory_leaf,
        sources=data.emory_urls,
        additional_instruction="Confirm title 'Provost' at Emory."
    )

    # Duke Pratt dean (critical)
    duke_node = evaluator.add_parallel(
        id="Pres_Previous_Duke_Dean",
        desc="Verify the individual previously served as dean of the Pratt School of Engineering at Duke University",
        parent=ind_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=has_urls(data.duke_urls),
        id="Pres_Duke_URL",
        desc="Provide URL confirming Duke engineering dean position",
        parent=duke_node,
        critical=True
    )
    duke_leaf = evaluator.add_leaf(
        id="Pres_Was_Duke_Dean",
        desc="Individual served as dean of the Pratt School of Engineering at Duke University",
        parent=duke_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"{data.name} previously served as dean of the Pratt School of Engineering at Duke University.",
        node=duke_leaf,
        sources=data.duke_urls,
        additional_instruction="Look specifically for 'Dean' of Pratt School of Engineering."
    )


async def verify_individual_3(evaluator: Evaluator, parent_node, data: Individual3Coach):
    ind_node = evaluator.add_parallel(
        id="Individual_3_Football_Coach",
        desc="Correctly identify the head football coach hired by a Big Ten university in December 2025",
        parent=parent_node,
        critical=False
    )

    evaluator.add_custom_node(
        result=has_text(data.name),
        id="Coach_Name_Provided",
        desc="Provide the individual's full name",
        parent=ind_node,
        critical=True
    )

    inst_node = evaluator.add_parallel(
        id="Coach_Institution_Identity",
        desc="Provide the specific institution name and verify it is a public Big Ten university",
        parent=ind_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=has_text(data.institution),
        id="Coach_Institution_Name_Provided",
        desc="Provide the institution name",
        parent=inst_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=has_urls(data.institution_urls) or has_urls(data.conference_urls),
        id="Coach_Institution_URL",
        desc="Provide URL confirming institution identity and conference membership",
        parent=inst_node,
        critical=True
    )
    coach_public_leaf = evaluator.add_leaf(
        id="Coach_Institution_Is_Public",
        desc="The institution is a public university",
        parent=inst_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"{data.institution} is a public university.",
        node=coach_public_leaf,
        sources=data.institution_urls,
        additional_instruction="Confirm public status explicitly."
    )
    coach_bigten_leaf = evaluator.add_leaf(
        id="Coach_Institution_Is_Big_Ten",
        desc="The institution is a Big Ten university",
        parent=inst_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"{data.institution} is a member of the Big Ten Conference.",
        node=coach_bigten_leaf,
        sources=combine_urls(data.conference_urls, data.institution_urls),
        additional_instruction="Check Big Ten membership."
    )

    # Hiring timing
    timing_node = evaluator.add_parallel(
        id="Coach_Hiring_Timing",
        desc="Verify the hiring was announced in December 2025",
        parent=ind_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=has_text(data.hiring_date),
        id="Coach_Announcement_Date_Provided",
        desc="Provide the exact announcement date",
        parent=timing_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=has_urls(data.hiring_date_urls),
        id="Coach_Date_URL",
        desc="Provide URL confirming hiring date",
        parent=timing_node,
        critical=True
    )
    coach_date_leaf = evaluator.add_leaf(
        id="Coach_Date_In_Dec_2025",
        desc="The announcement date is in December 2025",
        parent=timing_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The hiring of {data.name} as head football coach at {data.institution} was announced in December 2025.",
        node=coach_date_leaf,
        sources=data.hiring_date_urls,
        additional_instruction="Use page timestamp or announcement date."
    )

    # Age 66
    age_node = evaluator.add_parallel(
        id="Coach_Age_66",
        desc="Verify the coach is 66 years old at time of hiring",
        parent=ind_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=has_urls(data.age_urls),
        id="Coach_Age_URL",
        desc="Provide URL confirming age",
        parent=age_node,
        critical=True
    )
    age_leaf = evaluator.add_leaf(
        id="Coach_Age_Documented",
        desc="Document that the coach is 66 years old",
        parent=age_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"As of December 2025, {data.name} is 66 years old.",
        node=age_leaf,
        sources=data.age_urls,
        additional_instruction="If a DOB is given, infer age as of Dec 2025; otherwise look for explicit age."
    )

    # Previous Utah position, seasons, record
    utah_node = evaluator.add_parallel(
        id="Coach_Previous_Utah_Position",
        desc="Verify previous position as head coach at University of Utah for 21 seasons with 177-88 record",
        parent=ind_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=has_urls(data.utah_urls),
        id="Coach_Previous_URL",
        desc="Provide URL confirming previous position and record",
        parent=utah_node,
        critical=True
    )
    utah_head_leaf = evaluator.add_leaf(
        id="Coach_Was_Utah_Head_Coach",
        desc="Individual served as head coach at the University of Utah",
        parent=utah_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"{data.name} served as head football coach at the University of Utah.",
        node=utah_head_leaf,
        sources=data.utah_urls,
        additional_instruction="Confirm head coach role at Utah."
    )
    utah_seasons_leaf = evaluator.add_leaf(
        id="Coach_Utah_21_Seasons",
        desc="Individual served for 21 seasons at Utah",
        parent=utah_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"{data.name} served as Utah's head coach for 21 seasons.",
        node=utah_seasons_leaf,
        sources=data.utah_urls,
        additional_instruction="Look for total seasons count."
    )
    utah_record_leaf = evaluator.add_leaf(
        id="Coach_Utah_Record_177_88",
        desc="Individual had a career record of 177-88 at Utah",
        parent=utah_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"{data.name}'s head coaching record at Utah was 177-88.",
        node=utah_record_leaf,
        sources=data.utah_urls,
        additional_instruction="Look for an all-time record summary."
    )

    # Contract terms
    contract_node = evaluator.add_parallel(
        id="Coach_Contract_Terms",
        desc="Verify five-year contract averaging $8.2 million per year",
        parent=ind_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=has_urls(data.contract_urls),
        id="Coach_Contract_URL",
        desc="Provide URL confirming contract details",
        parent=contract_node,
        critical=True
    )
    contract_years_leaf = evaluator.add_leaf(
        id="Coach_Contract_Five_Years",
        desc="Individual signed a five-year contract",
        parent=contract_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"{data.name} signed a five-year contract with {data.institution}.",
        node=contract_years_leaf,
        sources=data.contract_urls,
        additional_instruction="Confirm 5-year term."
    )
    contract_avg_leaf = evaluator.add_leaf(
        id="Coach_Contract_Avg_8_2M",
        desc="Contract averages $8.2 million per year",
        parent=contract_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"{data.name}'s contract averages $8.2 million per year.",
        node=contract_avg_leaf,
        sources=data.contract_urls,
        additional_instruction="Look for 'average annual value' or similar phrasing; accept $8.2M."
    )


async def verify_individual_4(evaluator: Evaluator, parent_node, data: Individual4Superintendent):
    ind_node = evaluator.add_parallel(
        id="Individual_4_School_Superintendent",
        desc="Correctly identify the school superintendent appointed in western New York in January 2026",
        parent=parent_node,
        critical=False
    )

    evaluator.add_custom_node(
        result=has_text(data.name),
        id="Supt_Name_Provided",
        desc="Provide the individual's full name",
        parent=ind_node,
        critical=True
    )

    # District identity and location in Western NY
    dist_node = evaluator.add_parallel(
        id="Supt_District_Identity",
        desc="Provide the school district name and verify it is in a city in western New York State",
        parent=ind_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=has_text(data.district),
        id="Supt_District_Name_Provided",
        desc="Provide the school district name",
        parent=dist_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=has_urls(data.district_urls),
        id="Supt_District_URL",
        desc="Provide URL confirming district identity and location",
        parent=dist_node,
        critical=True
    )
    western_leaf = evaluator.add_leaf(
        id="Supt_Location_Western_NY",
        desc="The district is in a city in western New York State",
        parent=dist_node,
        critical=True
    )
    city_txt = data.city if has_text(data.city) else "the stated city"
    await evaluator.verify(
        claim=f"{data.district} is located in {city_txt}, which is in western New York State.",
        node=western_leaf,
        sources=data.district_urls,
        additional_instruction="Use common geographic definitions. Cities such as Buffalo, Rochester, Niagara Falls, Jamestown, Olean, Batavia are in Western NY."
    )

    # Appointment timing Jan 2026
    timing_node = evaluator.add_parallel(
        id="Supt_Appointment_Timing",
        desc="Verify the appointment was announced in January 2026",
        parent=ind_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=has_text(data.appointment_date),
        id="Supt_Announcement_Date_Provided",
        desc="Provide the exact announcement date",
        parent=timing_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=has_urls(data.appointment_date_urls),
        id="Supt_Date_URL",
        desc="Provide URL confirming announcement date",
        parent=timing_node,
        critical=True
    )
    jan_leaf = evaluator.add_leaf(
        id="Supt_Date_In_Jan_2026",
        desc="The announcement date is in January 2026",
        parent=timing_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The superintendent appointment of {data.name} at {data.district} was announced in January 2026.",
        node=jan_leaf,
        sources=data.appointment_date_urls,
        additional_instruction="Verify using page date/announcement date."
    )

    # 20 years in same district
    yrs_node = evaluator.add_parallel(
        id="Supt_Twenty_Years_Service",
        desc="Verify the individual worked in the same school district for 20 years prior to appointment",
        parent=ind_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=has_urls(data.years_service_urls),
        id="Supt_Years_URL",
        desc="Provide URL confirming years of service",
        parent=yrs_node,
        critical=True
    )
    yrs_leaf = evaluator.add_leaf(
        id="Supt_20_Years_Documented",
        desc="Document that individual worked in the district for 20 years",
        parent=yrs_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"{data.name} worked in {data.district} for 20 years prior to the appointment.",
        node=yrs_leaf,
        sources=data.years_service_urls,
        additional_instruction="Look for explicit '20 years' language."
    )

    # Selected from four internal candidates
    sel_node = evaluator.add_parallel(
        id="Supt_Internal_Selection",
        desc="Verify the individual was selected from a field of four internal candidates",
        parent=ind_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=has_urls(data.selection_urls),
        id="Supt_Selection_URL",
        desc="Provide URL confirming selection process",
        parent=sel_node,
        critical=True
    )
    four_leaf = evaluator.add_leaf(
        id="Supt_Four_Candidates",
        desc="Individual was selected from a field of four candidates",
        parent=sel_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"{data.name} was selected from a field of four candidates.",
        node=four_leaf,
        sources=data.selection_urls,
        additional_instruction="The page should indicate four candidates were considered."
    )
    internal_leaf = evaluator.add_leaf(
        id="Supt_All_Candidates_Internal",
        desc="All four candidates were internal to the district",
        parent=sel_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"All four candidates for the superintendent role at {data.district} were internal candidates from the district.",
        node=internal_leaf,
        sources=data.selection_urls,
        additional_instruction="Confirm that each candidate was 'internal' to the district."
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
    # Initialize evaluator (root parallel: four independent individuals)
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
    extracted: AllIndividualsExtraction = await evaluator.extract(
        prompt=prompt_extract_all(),
        template_class=AllIndividualsExtraction,
        extraction_name="extracted_individuals",
    )

    # Build subtrees per individual
    await verify_individual_1(evaluator, root, extracted.individual1 or Individual1AD())
    await verify_individual_2(evaluator, root, extracted.individual2 or Individual2President())
    await verify_individual_3(evaluator, root, extracted.individual3 or Individual3Coach())
    await verify_individual_4(evaluator, root, extracted.individual4 or Individual4Superintendent())

    # Return evaluator summary
    return evaluator.get_summary()