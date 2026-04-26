import asyncio
import logging
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy

# --------------------------------------------------------------------------- #
# Task constants                                                              #
# --------------------------------------------------------------------------- #
TASK_ID = "pa_superintendent_2025"
TASK_DESCRIPTION = """
A Pennsylvania school district appointed a new superintendent in 2025 who officially began their tenure on July 1, 2025. This individual holds a valid Pennsylvania Superintendent Letter of Eligibility or Commission Qualification Letter, having completed a State-approved graduate-level program of educational administrative study for the preparation of chief school administrators. The superintendent holds a doctoral degree (Ed.D. or Ph.D.) and has at least 6 years of satisfactory educational or student support service in K-12 schools or an accredited institution of higher education, with at least 3 of those years in supervisory or administrative positions. The individual is a native of Western Pennsylvania who grew up in a rural area. The appointment and the superintendent's background are documented on an official district biography page or announcement accessible through the district's website. Identify the full name (including title) of this superintendent and the name of the Pennsylvania school district where they serve. Provide reference URLs confirming: (1) their identity and title, (2) the district name, (3) their July 1, 2025 start date, (4) their Western Pennsylvania origin, (5) their rural upbringing, and (6) the official biography or announcement page.
"""

# --------------------------------------------------------------------------- #
# Extraction models                                                           #
# --------------------------------------------------------------------------- #
class SuperintendentExtraction(BaseModel):
    # Core identification
    full_name_with_title: Optional[str] = None
    district_name: Optional[str] = None

    # Dates and appointment
    start_date: Optional[str] = None          # e.g., "July 1, 2025"
    appointment_year: Optional[str] = None    # e.g., "2025"

    # Qualifications
    certification: Optional[str] = None       # e.g., "PA Superintendent Letter of Eligibility"
    graduate_program_description: Optional[str] = None
    doctoral_degree: Optional[str] = None     # e.g., "Ed.D." or "Ph.D."

    # Experience
    total_educational_service: Optional[str] = None   # narrative with years; should imply >= 6
    administrative_experience: Optional[str] = None   # narrative with years; should imply >= 3

    # Origin and background
    western_pa_origin: Optional[str] = None
    rural_upbringing: Optional[str] = None

    # URL sources by category
    urls_identity_title: List[str] = Field(default_factory=list)
    urls_district_name: List[str] = Field(default_factory=list)
    urls_start_date: List[str] = Field(default_factory=list)
    urls_western_pa_origin: List[str] = Field(default_factory=list)
    urls_rural_upbringing: List[str] = Field(default_factory=list)
    urls_bio_or_announcement: List[str] = Field(default_factory=list)

    # Additional supporting URLs for rubric items
    urls_certification: List[str] = Field(default_factory=list)
    urls_education: List[str] = Field(default_factory=list)
    urls_total_experience: List[str] = Field(default_factory=list)
    urls_admin_experience: List[str] = Field(default_factory=list)
    urls_appointment_year: List[str] = Field(default_factory=list)
    urls_district_location: List[str] = Field(default_factory=list)
    urls_district_appointment: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_superintendent() -> str:
    return """
    Extract, from the provided answer text, the structured information about the identified Pennsylvania superintendent and the exact URLs cited by the answer.

    REQUIRED FIELDS (strings; if missing, return null):
    - full_name_with_title: The superintendent's full name INCLUDING doctoral title formatting if provided (e.g., "Dr. Jane Q. Smith, Ph.D." or "Jane Q. Smith, Ed.D."). Do not invent formatting.
    - district_name: The exact Pennsylvania school district name where the person serves as superintendent.
    - start_date: The official start date as written in the answer (e.g., "July 1, 2025" or "7/1/2025").
    - appointment_year: The 4-digit year of appointment if explicitly stated in the answer (e.g., "2025"); else null.
    - certification: The form of Pennsylvania superintendent certification explicitly mentioned in the answer (e.g., "Pennsylvania Superintendent Letter of Eligibility", "Commission Qualification Letter"); else null.
    - graduate_program_description: Text describing completion of the state-approved graduate-level administrative program for preparation of chief school administrators (if mentioned); else null.
    - doctoral_degree: The doctoral degree abbreviation exactly as shown (e.g., "Ed.D." or "Ph.D."); else null.
    - total_educational_service: The exact phrasing about total years of educational or student support service in K-12 or higher education (must imply 6 or more); else null.
    - administrative_experience: The exact phrasing about years in supervisory/administrative roles (must imply 3 or more); else null.
    - western_pa_origin: The exact phrasing indicating the person is a native of Western Pennsylvania; else null.
    - rural_upbringing: The exact phrasing indicating the person grew up in a rural area (preferably in Western Pennsylvania as stated); else null.

    REQUIRED URL LISTS PER CATEGORY (extract only URLs explicitly present in the answer; if none, return an empty list):
    - urls_identity_title: URL(s) confirming the individual's identity and superintendent title.
    - urls_district_name: URL(s) confirming the district name.
    - urls_start_date: URL(s) confirming the July 1, 2025 start date.
    - urls_western_pa_origin: URL(s) confirming Western Pennsylvania origin.
    - urls_rural_upbringing: URL(s) confirming rural upbringing.
    - urls_bio_or_announcement: Official district biography/announcement URL(s) hosted on the district’s website.

    ADDITIONAL SUPPORTING URL LISTS (if provided in the answer; else empty list):
    - urls_certification: URL(s) confirming superintendent certification (Letter of Eligibility or Commission Qualification Letter).
    - urls_education: URL(s) confirming graduate program completion and/or doctoral degree.
    - urls_total_experience: URL(s) confirming total years of educational/student support service.
    - urls_admin_experience: URL(s) confirming years in supervisory/administrative roles.
    - urls_appointment_year: URL(s) confirming the 2025 appointment year.
    - urls_district_location: URL(s) confirming the district is in Pennsylvania.
    - urls_district_appointment: URL(s) confirming the specific appointment as superintendent to the district.

    RULES:
    - Return null for any missing string field; return [] for any missing URL list.
    - Do not fabricate URLs. Extract only URLs that appear in the answer in any reasonable format (plain, markdown link, etc.).
    - Preserve the text exactly for narrative fields; do not paraphrase.
    """


# --------------------------------------------------------------------------- #
# Helpers                                                                     #
# --------------------------------------------------------------------------- #
def _merge_urls(*lists: List[str]) -> List[str]:
    seen = set()
    merged: List[str] = []
    for lst in lists:
        for url in lst or []:
            if url and (url not in seen):
                seen.add(url)
                merged.append(url)
    return merged


def _strip_title(name_with_title: Optional[str]) -> str:
    if not name_with_title:
        return ""
    s = name_with_title.strip()
    # Remove common doctoral prefixes/suffixes conservatively
    lowers = s.lower()
    if lowers.startswith("dr. "):
        s = s[4:].strip()
    # Remove common suffixes
    for suf in [", ph.d.", " ph.d.", ", phd", " phd", ", ed.d.", " ed.d.", ", edd", " edd", ", d.ed.", " d.ed."]:
        if s.lower().endswith(suf):
            s = s[: -len(suf)].strip(", ").strip()
    return s


def _has_any_url(urls: Optional[List[str]]) -> bool:
    return bool(urls and len([u for u in urls if isinstance(u, str) and u.strip()]) > 0)


# --------------------------------------------------------------------------- #
# Subtree builders (each creates nodes and runs verifications)                #
# --------------------------------------------------------------------------- #
async def build_certification_subtree(evaluator: Evaluator, parent, data: SuperintendentExtraction, name_core: str):
    node = evaluator.add_sequential(
        id="Pennsylvania_Certification",
        desc="Verify the individual holds valid Pennsylvania superintendent certification",
        parent=parent,
        critical=True,
    )

    # Leaf: Superintendent Letter of Eligibility / Commission Qualification Letter
    leaf_cert = evaluator.add_leaf(
        id="Superintendent_Letter_of_Eligibility",
        desc="Individual holds Pennsylvania Superintendent Letter of Eligibility or Commission Qualification Letter",
        parent=node,
        critical=True,
    )
    cert_sources = _merge_urls(
        data.urls_certification,
        data.urls_bio_or_announcement,
        data.urls_identity_title,
        data.urls_education,
    )
    cert_claim = f"This source confirms that {name_core} holds a Pennsylvania Superintendent Letter of Eligibility or a Commission Qualification Letter (superintendent credential)."
    await evaluator.verify(
        claim=cert_claim,
        node=leaf_cert,
        sources=cert_sources,
        additional_instruction="Accept synonyms such as 'Letter of Eligibility for the Superintendency', 'PA superintendent letter of eligibility', or 'Commission Qualification Letter'. The page must clearly indicate Pennsylvania superintendent certification.",
    )

    # Leaf: Reference URL presence (quality/completeness)
    evaluator.add_custom_node(
        result=_has_any_url(data.urls_certification),
        id="Certification_Reference_URL",
        desc="Provide reference URL confirming certification requirement compliance",
        parent=node,
        critical=True,
    )


async def build_graduate_education_subtree(evaluator: Evaluator, parent, data: SuperintendentExtraction, name_core: str):
    node = evaluator.add_sequential(
        id="Graduate_Education",
        desc="Verify completion of required graduate-level administrative preparation program",
        parent=parent,
        critical=True,
    )

    # Leaf: Graduate Program Completion (state-approved program)
    grad_leaf = evaluator.add_leaf(
        id="Graduate_Program_Completion",
        desc="Completed State-approved graduate-level program for chief school administrator preparation",
        parent=node,
        critical=True,
    )
    grad_sources = _merge_urls(
        data.urls_education,
        data.urls_certification,
        data.urls_bio_or_announcement,
    )
    grad_claim = f"This source confirms that {name_core} completed a state-approved graduate-level program preparing chief school administrators (e.g., superintendent Letter of Eligibility program)."
    await evaluator.verify(
        claim=grad_claim,
        node=grad_leaf,
        sources=grad_sources,
        additional_instruction="Accept program names such as 'Superintendent Letter of Eligibility Program', 'Educational Leadership (Superintendent certification)', or equivalent state-approved C.S.A. preparation programs.",
    )

    # Leaf: Doctoral Degree (Ed.D. or Ph.D.)
    doc_leaf = evaluator.add_leaf(
        id="Doctoral_Degree",
        desc="Holds a doctoral degree (Ed.D. or Ph.D.)",
        parent=node,
        critical=True,
    )
    doc_sources = _merge_urls(
        data.urls_education,
        data.urls_identity_title,
        data.urls_bio_or_announcement,
    )
    doc_claim = f"This source confirms that {name_core} holds a doctoral degree (Ed.D. or Ph.D.)."
    await evaluator.verify(
        claim=doc_claim,
        node=doc_leaf,
        sources=doc_sources,
        additional_instruction="Allow equivalent formatting (e.g., 'EdD', 'PhD') and accept presence of 'Dr.' as evidence of a doctoral title if the page also indicates Ed.D./Ph.D. explicitly or implicitly.",
    )

    # Leaf: Reference URL presence
    evaluator.add_custom_node(
        result=_has_any_url(data.urls_education),
        id="Education_Reference_URL",
        desc="Provide reference URL confirming educational qualifications",
        parent=node,
        critical=True,
    )


async def build_total_experience_subtree(evaluator: Evaluator, parent, data: SuperintendentExtraction, name_core: str):
    node = evaluator.add_sequential(
        id="Total_Experience_Requirement",
        desc="Verify minimum 6 years of educational service in K-12 or higher education",
        parent=parent,
        critical=True,
    )

    # Leaf: >= 6 years total educational/student support service
    exp_leaf = evaluator.add_leaf(
        id="Six_Years_Educational_Service",
        desc="Has at least 6 years of satisfactory educational or student support service in K-12 schools or accredited higher education institution",
        parent=node,
        critical=True,
    )
    exp_sources = _merge_urls(
        data.urls_total_experience,
        data.urls_bio_or_announcement,
    )
    exp_claim = f"This source states that {name_core} has at least six (6) years of educational or student support service in K-12 or an accredited higher education institution."
    await evaluator.verify(
        claim=exp_claim,
        node=exp_leaf,
        sources=exp_sources,
        additional_instruction="Accept phrases like 'over six years', 'more than 6 years', or any wording clearly indicating ≥ 6 years.",
    )

    # Leaf: Reference URL presence
    evaluator.add_custom_node(
        result=_has_any_url(data.urls_total_experience),
        id="Experience_Reference_URL",
        desc="Provide reference URL confirming total years of experience",
        parent=node,
        critical=True,
    )


async def build_admin_experience_subtree(evaluator: Evaluator, parent, data: SuperintendentExtraction, name_core: str):
    node = evaluator.add_sequential(
        id="Administrative_Experience_Requirement",
        desc="Verify minimum 3 years in supervisory or administrative positions",
        parent=parent,
        critical=True,
    )

    # Leaf: >= 3 years supervisory/administrative
    admin_leaf = evaluator.add_leaf(
        id="Three_Years_Administrative_Service",
        desc="Has at least 3 years of service in supervisory or administrative positions",
        parent=node,
        critical=True,
    )
    admin_sources = _merge_urls(
        data.urls_admin_experience,
        data.urls_bio_or_announcement,
    )
    admin_claim = f"This source states that {name_core} has at least three (3) years in supervisory or administrative positions (e.g., principal, director, assistant superintendent)."
    await evaluator.verify(
        claim=admin_claim,
        node=admin_leaf,
        sources=admin_sources,
        additional_instruction="Accept wording that clearly implies ≥ 3 years of supervisory/administrative experience, even if titles vary.",
    )

    # Leaf: Reference URL presence
    evaluator.add_custom_node(
        result=_has_any_url(data.urls_admin_experience),
        id="Administrative_Experience_Reference_URL",
        desc="Provide reference URL confirming administrative experience",
        parent=node,
        critical=True,
    )


async def build_pa_district_subtree(evaluator: Evaluator, parent, data: SuperintendentExtraction, name_core: str):
    node = evaluator.add_sequential(
        id="Pennsylvania_School_District",
        desc="Verify appointment to a Pennsylvania school district",
        parent=parent,
        critical=True,
    )

    # Leaf: Appointment to PA district (and the district is in Pennsylvania)
    appoint_leaf = evaluator.add_leaf(
        id="Pennsylvania_District_Appointment",
        desc="Individual was appointed as superintendent of a school district located in Pennsylvania",
        parent=node,
        critical=True,
    )
    appoint_sources = _merge_urls(
        data.urls_district_appointment,
        data.urls_identity_title,
        data.urls_bio_or_announcement,
        data.urls_district_name,
        data.urls_district_location,
    )
    if data.district_name:
        appoint_claim = f"This source confirms that {name_core} was appointed superintendent of {data.district_name}, a public school district in Pennsylvania (PA)."
    else:
        appoint_claim = f"This source confirms that {name_core} was appointed superintendent of a public school district in Pennsylvania (PA)."
    await evaluator.verify(
        claim=appoint_claim,
        node=appoint_leaf,
        sources=appoint_sources,
        additional_instruction="Verify that the district is located in the U.S. state of Pennsylvania. Accept variants such as 'PA'.",
    )

    # Leaf: Reference URL presence
    evaluator.add_custom_node(
        result=_has_any_url(data.urls_district_location),
        id="District_Location_Reference_URL",
        desc="Provide reference URL confirming Pennsylvania district location",
        parent=node,
        critical=True,
    )


async def build_appointment_year_subtree(evaluator: Evaluator, parent, data: SuperintendentExtraction, name_core: str):
    node = evaluator.add_sequential(
        id="Appointment_Year",
        desc="Verify appointment occurred in calendar year 2025",
        parent=parent,
        critical=True,
    )

    # Leaf: Appointment in 2025
    year_leaf = evaluator.add_leaf(
        id="Year_2025_Appointment",
        desc="Individual was appointed as superintendent in 2025",
        parent=node,
        critical=True,
    )
    year_sources = _merge_urls(
        data.urls_appointment_year,
        data.urls_bio_or_announcement,
        data.urls_identity_title,
    )
    year_claim = f"This source confirms that {name_core} was appointed as superintendent in calendar year 2025 (e.g., board vote/announcement in 2025)."
    await evaluator.verify(
        claim=year_claim,
        node=year_leaf,
        sources=year_sources,
        additional_instruction="Accept descriptions such as 'appointed in May 2025', 'board approved in 2025', or similar clear 2025 appointment phrasing.",
    )

    # Leaf: Reference URL presence
    evaluator.add_custom_node(
        result=_has_any_url(data.urls_appointment_year),
        id="Appointment_Year_Reference_URL",
        desc="Provide reference URL confirming 2025 appointment",
        parent=node,
        critical=True,
    )


async def build_start_date_subtree(evaluator: Evaluator, parent, data: SuperintendentExtraction, name_core: str):
    node = evaluator.add_sequential(
        id="Start_Date",
        desc="Verify official start date as superintendent",
        parent=parent,
        critical=True,
    )

    # Leaf: Start date July 1, 2025
    start_leaf = evaluator.add_leaf(
        id="July_1_2025_Start",
        desc="Individual officially began their tenure as superintendent on July 1, 2025",
        parent=node,
        critical=True,
    )
    start_sources = _merge_urls(
        data.urls_start_date,
        data.urls_bio_or_announcement,
        data.urls_identity_title,
    )
    start_claim = f"This source confirms that {name_core} officially began as superintendent on July 1, 2025."
    await evaluator.verify(
        claim=start_claim,
        node=start_leaf,
        sources=start_sources,
        additional_instruction="Be strict about the date being July 1, 2025; accept equivalent formats like 'July 1st, 2025' or '7/1/2025'.",
    )

    # Leaf: Reference URL presence
    evaluator.add_custom_node(
        result=_has_any_url(data.urls_start_date),
        id="Start_Date_Reference_URL",
        desc="Provide reference URL confirming July 1, 2025 start date",
        parent=node,
        critical=True,
    )


async def build_western_pa_subtree(evaluator: Evaluator, parent, data: SuperintendentExtraction, name_core: str):
    node = evaluator.add_sequential(
        id="Western_Pennsylvania_Native",
        desc="Verify individual is a native of Western Pennsylvania",
        parent=parent,
        critical=True,
    )

    # Leaf: Western PA origin
    origin_leaf = evaluator.add_leaf(
        id="Western_PA_Origin",
        desc="Individual is a native of Western Pennsylvania",
        parent=node,
        critical=True,
    )
    origin_sources = _merge_urls(
        data.urls_western_pa_origin,
        data.urls_bio_or_announcement,
    )
    origin_claim = f"This source states that {name_core} is a native of Western Pennsylvania."
    await evaluator.verify(
        claim=origin_claim,
        node=origin_leaf,
        sources=origin_sources,
        additional_instruction="Accept phrasing like 'hails from Western Pennsylvania', 'Western PA native', or clear equivalent.",
    )

    # Leaf: Reference URL presence
    evaluator.add_custom_node(
        result=_has_any_url(data.urls_western_pa_origin),
        id="Origin_Reference_URL",
        desc="Provide reference URL confirming Western Pennsylvania origin",
        parent=node,
        critical=True,
    )


async def build_rural_background_subtree(evaluator: Evaluator, parent, data: SuperintendentExtraction, name_core: str):
    node = evaluator.add_sequential(
        id="Rural_Background",
        desc="Verify individual grew up in a rural area",
        parent=parent,
        critical=True,
    )

    # Leaf: Rural upbringing (preferably in Western PA per task)
    rural_leaf = evaluator.add_leaf(
        id="Rural_Upbringing",
        desc="Individual grew up in a rural area of Western Pennsylvania",
        parent=node,
        critical=True,
    )
    rural_sources = _merge_urls(
        data.urls_rural_upbringing,
        data.urls_western_pa_origin,
        data.urls_bio_or_announcement,
    )
    rural_claim = f"This source states that {name_core} grew up in a rural area (preferably indicated as in Western Pennsylvania)."
    await evaluator.verify(
        claim=rural_claim,
        node=rural_leaf,
        sources=rural_sources,
        additional_instruction="Accept phrases such as 'grew up on a farm', 'from a small rural town/community', or clear equivalents indicating rural upbringing.",
    )

    # Leaf: Reference URL presence
    evaluator.add_custom_node(
        result=_has_any_url(data.urls_rural_upbringing),
        id="Rural_Background_Reference_URL",
        desc="Provide reference URL confirming rural upbringing",
        parent=node,
        critical=True,
    )


async def build_district_name_subtree(evaluator: Evaluator, parent, data: SuperintendentExtraction, name_core: str):
    node = evaluator.add_sequential(
        id="District_Name_Identification",
        desc="Correctly identify the specific Pennsylvania school district",
        parent=parent,
        critical=True,
    )

    # Leaf: School district name correctness
    dist_leaf = evaluator.add_leaf(
        id="School_District_Name",
        desc="Provide the correct name of the Pennsylvania school district where the individual serves as superintendent",
        parent=node,
        critical=True,
    )
    dist_sources = _merge_urls(
        data.urls_district_name,
        data.urls_bio_or_announcement,
    )
    if data.district_name:
        dist_claim = f"This source confirms the school district name as '{data.district_name}' where {name_core} serves as superintendent."
    else:
        # If missing extracted name, still verify that the source clearly identifies the district name
        dist_claim = f"This source clearly identifies the Pennsylvania school district where {name_core} serves as superintendent."
    await evaluator.verify(
        claim=dist_claim,
        node=dist_leaf,
        sources=dist_sources,
        additional_instruction="Allow common naming variants such as 'Area School District', 'ASD', or 'School District (SD)'.",
    )

    # Leaf: Reference URL presence
    evaluator.add_custom_node(
        result=_has_any_url(data.urls_district_name),
        id="District_Name_Reference_URL",
        desc="Provide reference URL confirming the school district name",
        parent=node,
        critical=True,
    )


async def build_name_title_subtree(evaluator: Evaluator, parent, data: SuperintendentExtraction, name_core: str):
    node = evaluator.add_sequential(
        id="Superintendent_Name_Identification",
        desc="Correctly identify the full name and title of the superintendent",
        parent=parent,
        critical=True,
    )

    # Leaf: Full name with title correctness
    name_leaf = evaluator.add_leaf(
        id="Full_Name_With_Title",
        desc="Provide the superintendent's full name including doctoral title (Dr./Ph.D.)",
        parent=node,
        critical=True,
    )
    name_sources = _merge_urls(
        data.urls_identity_title,
        data.urls_bio_or_announcement,
    )
    if data.full_name_with_title:
        name_claim = f"This source confirms that the superintendent's full name and title is '{data.full_name_with_title}'."
        name_instr = "Consider formatting variants equivalent (e.g., with/without middle initial, 'Dr.' prefix vs. 'Ph.D.'/'Ed.D.' suffix). The page must clearly show the same person and a doctoral title."
    else:
        # If missing formatted name, require the source to still confirm identity + doctoral title
        name_claim = f"This source confirms the superintendent's identity as {name_core} and that they hold a doctoral title (e.g., Dr., Ph.D., or Ed.D.)."
        name_instr = "Even if the provided formatting is missing in the answer, confirm identity and presence of a doctoral title."
    await evaluator.verify(
        claim=name_claim,
        node=name_leaf,
        sources=name_sources,
        additional_instruction=name_instr,
    )

    # Leaf: Reference URL presence
    evaluator.add_custom_node(
        result=_has_any_url(data.urls_identity_title),
        id="Name_Reference_URL",
        desc="Provide reference URL confirming the superintendent's name and title",
        parent=node,
        critical=True,
    )


async def build_official_bio_subtree(evaluator: Evaluator, parent, data: SuperintendentExtraction, name_core: str):
    node = evaluator.add_sequential(
        id="Official_Biography_Documentation",
        desc="Verify existence of official district biography or announcement",
        parent=parent,
        critical=True,
    )

    # Leaf: Official biography/announcement page exists (on district site)
    bio_leaf = evaluator.add_leaf(
        id="Official_Biography_Page",
        desc="Official district biography page or announcement exists documenting the superintendent's appointment and background",
        parent=node,
        critical=True,
    )
    bio_sources = _merge_urls(data.urls_bio_or_announcement)
    bio_claim = f"This URL is an official page on the school district's own website that documents {name_core}'s superintendent appointment and/or biographical background."
    await evaluator.verify(
        claim=bio_claim,
        node=bio_leaf,
        sources=bio_sources,
        additional_instruction="The page should be hosted on the district's domain (not external media). Accept pages like 'About the Superintendent', 'News/Announcement' on the district site.",
    )

    # Leaf: Reference URL presence
    evaluator.add_custom_node(
        result=_has_any_url(data.urls_bio_or_announcement),
        id="Biography_Reference_URL",
        desc="Provide reference URL to official biography or appointment announcement",
        parent=node,
        critical=True,
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
    # Initialize evaluator
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Root aggregation
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
    extracted: SuperintendentExtraction = await evaluator.extract(
        prompt=prompt_extract_superintendent(),
        template_class=SuperintendentExtraction,
        extraction_name="superintendent_extraction",
    )

    # Prepare main (critical) task node
    main = evaluator.add_parallel(
        id="Identify_Pennsylvania_Superintendent",
        desc="Correctly identify the superintendent who meets all specified criteria",
        parent=root,
        critical=True,
    )

    # Derived helper values
    name_core = _strip_title(extracted.full_name_with_title)

    # Build and verify each rubric subtree (all critical under the main node)
    await build_certification_subtree(evaluator, main, extracted, name_core)
    await build_graduate_education_subtree(evaluator, main, extracted, name_core)
    await build_total_experience_subtree(evaluator, main, extracted, name_core)
    await build_admin_experience_subtree(evaluator, main, extracted, name_core)
    await build_pa_district_subtree(evaluator, main, extracted, name_core)
    await build_appointment_year_subtree(evaluator, main, extracted, name_core)
    await build_start_date_subtree(evaluator, main, extracted, name_core)
    await build_western_pa_subtree(evaluator, main, extracted, name_core)
    await build_rural_background_subtree(evaluator, main, extracted, name_core)
    await build_district_name_subtree(evaluator, main, extracted, name_core)
    await build_name_title_subtree(evaluator, main, extracted, name_core)
    await build_official_bio_subtree(evaluator, main, extracted, name_core)

    # Return evaluation summary
    return evaluator.get_summary()